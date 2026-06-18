#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
フルモデルマージ (vLLM 用 save_pretrained)

対応モデル:
  - GLM-5-FP8 (FP8 base, partial BF16 trained layers)
  - Qwen3.5-397B-A17B (BF16 native, VLM architecture)
  - 他の FSDP sharded checkpoint も同様のパターンで対応可能

対応モード:
  - Full fine-tune: 単一/複数レイヤー × attn/mlp/full モジュール
  - LoRA: lora_A/lora_B → W_merged = W_base + B @ A * (α/r)

VLM 自動対応 (Phase 5):
  - save_pretrained が language_model サブモデルのみ保存する問題を自動修正
  - ベースから config/tokenizer/processor 等の非 weight ファイルを補完
  - visual encoder の weight シャードをベースからコピー + index 更新
  - checkpoint の key mapping: model.layers.* ↔ model.language_model.layers.*

自動検出:
  - ファイル: params.rank*.pt / attn_params.rank*.pt
  - メタ: params_meta.json / attn_params_meta.json
  - モード: meta の use_lora / param名の lora_A/lora_B

処理フロー:
  Phase 1: rank*.pt をストリーミングマージ
  Phase 2: ベースモデルを CPU ロード
  Phase 3: Full → weight 置換 / LoRA → W += B@A*(α/r)
  Phase 4: save_pretrained (bf16=ストリーミング / keep=通常)
  Phase 5: VLM 補完 (config/tokenizer/visual encoder) + 検証

使い方:
  # GLM-5-FP8 (78 layers)
  python merge_full_model.py \
    --base_model /path/to/GLM-5-FP8 \
    --checkpoint_dir ./output/glm5-sft \
    --out_dir ./output/merged \
    --num_layers 78 --output_dtype bf16

  # Qwen3.5 (60 layers, VLM)
  python merge_full_model.py \
    --base_model /path/to/Qwen3.5-397B-A17B \
    --checkpoint_dir ./output/qwen35-lastlayer-attn-sft \
    --out_dir ./output/merged \
    --num_layers 60 --output_dtype keep
"""

from __future__ import annotations

import os
import sys
import json
import glob
import gc
import time
import shutil
import argparse
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from safetensors.torch import save_file as safetensors_save_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    from safetensors import safe_open
    HAS_SAFE_OPEN = True
except ImportError:
    HAS_SAFE_OPEN = False


# ============================================================
# ShardedTensor unpickle hack
# ============================================================

HACK_ENABLED = False
HackShardedTensor = None

try:
    import torch.distributed._shard.sharded_tensor.api as st_api

    class HackShardedTensor(torch.Tensor):
        class ProcessGroupState:
            pass
        class Metadata:
            pass
        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)
        def __init__(self):
            self._local_shards = []
        def __setstate__(self, state):
            self._local_shards = []
            def is_shard_list(x):
                return isinstance(x, list) and len(x) > 0 and hasattr(x[0], "tensor")
            if isinstance(state, dict):
                for k in ("_local_shards", "local_shards", "_shards", "shards"):
                    v = state.get(k)
                    if is_shard_list(v):
                        self._local_shards = v
                        return
                for v in state.values():
                    if is_shard_list(v):
                        self._local_shards = v
                        return
                    if isinstance(v, dict):
                        for vv in v.values():
                            if is_shard_list(vv):
                                self._local_shards = vv
                                return
            elif isinstance(state, (list, tuple)):
                for it in state:
                    if is_shard_list(it):
                        self._local_shards = it
                        return
                for it in state:
                    if isinstance(it, dict):
                        for v in it.values():
                            if is_shard_list(v):
                                self._local_shards = v
                                return
        def local_shards(self):
            return self._local_shards
        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            raise RuntimeError("HackShardedTensor: not for computation")
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            raise RuntimeError("HackShardedTensor: not for computation")

    st_api.ShardedTensor = HackShardedTensor
    HACK_ENABLED = True
except Exception:
    pass


def _build_safe_globals():
    safe_list = []
    if HackShardedTensor is not None:
        safe_list.append(HackShardedTensor)
    try:
        import torch.distributed._shard.sharded_tensor.shard as st_shard
        for n in ("Shard", "ShardMetadata", "TensorProperties", "ShardedTensorMetadata"):
            if hasattr(st_shard, n):
                safe_list.append(getattr(st_shard, n))
    except Exception:
        pass
    return safe_list


def load_pt(path, safe_list):
    def _torch_load(**kwargs):
        if hasattr(torch.serialization, "safe_globals") and safe_list:
            with torch.serialization.safe_globals(safe_list):
                return torch.load(path, map_location="cpu", **kwargs)
        return torch.load(path, map_location="cpu", **kwargs)
    for use_mmap in (True, False):
        try:
            return _torch_load(weights_only=False, mmap=use_mmap)
        except TypeError:
            try:
                return _torch_load(mmap=True) if use_mmap else _torch_load()
            except TypeError:
                continue
        except Exception:
            if use_mmap:
                continue
            raise
    return _torch_load()


def clean_key(name):
    name = name.replace("_fsdp_wrapped_module.", "")
    name = name.replace("_checkpoint_wrapped_module.", "")
    name = name.replace("module.", "")
    while name.startswith("model.model."):
        name = "model." + name[len("model.model."):]
    return name


def get_offsets_sizes(meta):
    offsets = sizes = None
    for k in ("shard_offsets", "offsets"):
        if hasattr(meta, k):
            try:
                offsets = [int(x) for x in getattr(meta, k)]
                break
            except Exception:
                pass
    for k in ("shard_sizes", "sizes"):
        if hasattr(meta, k):
            try:
                sizes = [int(x) for x in getattr(meta, k)]
                break
            except Exception:
                pass
    return offsets, sizes


# ============================================================
# VLM detection
# ============================================================

def detect_vlm(base_model_dir):
    """ベースモデルが VLM かどうかを判定。"""
    config_path = os.path.join(base_model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        archs = config.get("architectures", [])
        model_type = config.get("model_type", "")
        for arch in archs:
            if any(kw in arch for kw in ("Conditional", "VL", "Visual", "Multimodal")):
                return True
        if any(kw in model_type for kw in ("vl", "visual", "multimodal")):
            return True

    index_path = os.path.join(base_model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        for key in index.get("weight_map", {}):
            if "visual" in key.lower():
                return True
    return False


# ============================================================
# FP8 → BF16 dequantization
# ============================================================

def _blockwise_dequant(w_fp8, scale_inv, block_size=(128, 128)):
    block_h, block_w = block_size
    if w_fp8.dim() == 2:
        out_f, in_f = w_fp8.shape
        num_bh = (out_f + block_h - 1) // block_h
        num_bw = (in_f + block_w - 1) // block_w
        pad_h = num_bh * block_h - out_f
        pad_w = num_bw * block_w - in_f
        w_bf16 = w_fp8.to(torch.bfloat16)
        if pad_h > 0 or pad_w > 0:
            w_bf16 = torch.nn.functional.pad(w_bf16, (0, pad_w, 0, pad_h))
        w_blocks = w_bf16.view(num_bh, block_h, num_bw, block_w)
        scale_bf16 = scale_inv[:num_bh, :num_bw].to(torch.bfloat16)
        w_blocks.mul_(scale_bf16.unsqueeze(1).unsqueeze(3))
        result = w_blocks.reshape(num_bh * block_h, num_bw * block_w)[:out_f, :in_f]
        return result.contiguous() if not result.is_contiguous() else result
    elif w_fp8.dim() == 3:
        n_exp, out_f, in_f = w_fp8.shape
        w_bf16 = torch.empty(n_exp, out_f, in_f, dtype=torch.bfloat16, device=w_fp8.device)
        for e in range(n_exp):
            w_bf16[e] = _blockwise_dequant(w_fp8[e], scale_inv[e], block_size)
        return w_bf16
    else:
        return w_fp8.to(torch.bfloat16)


def dequantize_layer_inplace(layer, block_size=(128, 128)):
    """1レイヤーの全 FP8 パラメータを in-place BF16 変換。"""
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    converted = 0
    for mod_name, module in layer.named_modules():
        if (hasattr(module, "weight")
                and isinstance(module.weight, torch.nn.Parameter)
                and module.weight.dtype in fp8_dtypes
                and hasattr(module, "weight_scale_inv")):
            w = module.weight.data
            scale = module.weight_scale_inv
            scale_data = scale.data if isinstance(scale, torch.nn.Parameter) else scale
            w_bf16 = _blockwise_dequant(w, scale_data, block_size)
            module.weight = torch.nn.Parameter(w_bf16, requires_grad=False)
            del w, w_bf16
            converted += 1
        for attr in list(module._parameters.keys()):
            param = module._parameters[attr]
            if param is None or param.dtype not in fp8_dtypes:
                continue
            if attr.endswith("_scale_inv"):
                continue
            scale_attr = attr + "_scale_inv"
            scale = getattr(module, scale_attr, None)
            if scale is not None:
                scale_data = scale.data if isinstance(scale, torch.nn.Parameter) else scale
                old_data = param.data
                w_bf16 = _blockwise_dequant(old_data, scale_data, block_size)
                module._parameters[attr] = torch.nn.Parameter(w_bf16, requires_grad=False)
                del old_data, w_bf16
            else:
                module._parameters[attr] = torch.nn.Parameter(
                    param.data.to(torch.bfloat16), requires_grad=False)
            converted += 1
    return converted


def dequantize_module_params(module, block_size=(128, 128)):
    """トップレベルモジュール用 FP8→BF16。"""
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    converted = 0
    for mod_name, mod in module.named_modules():
        if (hasattr(mod, "weight")
                and isinstance(mod.weight, torch.nn.Parameter)
                and mod.weight.dtype in fp8_dtypes
                and hasattr(mod, "weight_scale_inv")):
            w = mod.weight.data
            scale = mod.weight_scale_inv
            scale_data = scale.data if isinstance(scale, torch.nn.Parameter) else scale
            w_bf16 = _blockwise_dequant(w, scale_data, block_size)
            mod.weight = torch.nn.Parameter(w_bf16, requires_grad=False)
            del w, w_bf16
            converted += 1
        for attr in list(mod._parameters.keys()):
            param = mod._parameters[attr]
            if param is None or param.dtype not in fp8_dtypes:
                continue
            if attr.endswith("_scale_inv"):
                continue
            scale_attr = attr + "_scale_inv"
            scale = getattr(mod, scale_attr, None)
            if scale is not None:
                scale_data = scale.data if isinstance(scale, torch.nn.Parameter) else scale
                old_data = param.data
                w_bf16 = _blockwise_dequant(old_data, scale_data, block_size)
                mod._parameters[attr] = torch.nn.Parameter(w_bf16, requires_grad=False)
                del old_data, w_bf16
            else:
                mod._parameters[attr] = torch.nn.Parameter(
                    param.data.to(torch.bfloat16), requires_grad=False)
            converted += 1
    return converted


# ============================================================
# Phase 1: ストリーミング attn パラメータマージ
# ============================================================

def streaming_merge_params(checkpoint_dir, world_size, strict=True):
    """
    2パスストリーミングでシャードされたチェックポイントをマージ。
    自動検出するファイルパターン:
      - attn_params.rank*.pt + attn_params_meta.json (GLM-5)
      - params.rank*.pt + params_meta.json (Qwen3.5)
      - *rank*.pt (フォールバック)
    """
    safe_list = _build_safe_globals()

    # Auto-detect meta file
    meta = {}
    for meta_name in ("params_meta.json", "attn_params_meta.json"):
        meta_path = os.path.join(checkpoint_dir, meta_name)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            print(f"  meta file: {meta_name}", flush=True)
            print(f"  checkpoint meta: {json.dumps(meta, indent=2)}", flush=True)
            break

    if world_size <= 0:
        world_size = meta.get("world_size", 0)

    # Auto-detect rank file pattern
    # Priority: params.rank*.pt > attn_params.rank*.pt > *rank*.pt
    FILE_PATTERNS = [
        "params.rank{r}.pt",
        "attn_params.rank{r}.pt",
    ]
    GLOB_PATTERNS = [
        "params.rank*.pt",
        "attn_params.rank*.pt",
        "*rank*.pt",
    ]

    rank_files = []
    if world_size > 0:
        for r in range(world_size):
            found = False
            for pat in FILE_PATTERNS:
                fpath = os.path.join(checkpoint_dir, pat.format(r=r))
                if os.path.exists(fpath):
                    rank_files.append((r, fpath))
                    found = True
                    break
            if not found:
                cands = sorted(glob.glob(os.path.join(checkpoint_dir, f"*rank{r}*.pt")))
                cands = [c for c in cands if "meta" not in c and "full" not in c]
                if cands:
                    rank_files.append((r, cands[0]))
    else:
        for gpat in GLOB_PATTERNS:
            all_f = sorted(glob.glob(os.path.join(checkpoint_dir, gpat)))
            all_f = [f for f in all_f if "meta" not in f and "full" not in f]
            if all_f:
                rank_files = [(i, f) for i, f in enumerate(all_f)]
                break

    if not rank_files:
        raise FileNotFoundError(f"No rank files in {checkpoint_dir}")
    print(f"  rank files: {len(rank_files)} (pattern: {os.path.basename(rank_files[0][1])})",
          flush=True)

    # Pass 1: metadata scan
    print("  [Pass 1] Scanning metadata...", flush=True)
    all_shard_metas: Dict[str, List[Dict]] = {}
    all_full_tensors: Dict[str, torch.Tensor] = {}

    for rank_idx, fpath in rank_files:
        sd = load_pt(fpath, safe_list)
        for raw_name, obj in sd.items():
            name = clean_key(str(raw_name))
            local_shards = None
            if hasattr(obj, "local_shards") and callable(getattr(obj, "local_shards")):
                try:
                    local_shards = obj.local_shards()
                except Exception:
                    pass
            if isinstance(local_shards, list):
                for sh in local_shards:
                    st = getattr(sh, "tensor", None)
                    sm = getattr(sh, "metadata", None)
                    if st is None or sm is None or not torch.is_tensor(st):
                        continue
                    offsets, sizes = get_offsets_sizes(sm)
                    if offsets is None or sizes is None:
                        continue
                    all_shard_metas.setdefault(name, []).append({
                        "raw_name": str(raw_name),
                        "offsets": offsets,
                        "sizes": sizes,
                        "dtype": str(st.dtype),
                    })
                continue
            if torch.is_tensor(obj) and name not in all_full_tensors:
                all_full_tensors[name] = obj.detach().contiguous().cpu()
        del sd
        gc.collect()

    # Validate & allocate
    output_tensors: Dict[str, torch.Tensor] = {}
    for name, metas in all_shard_metas.items():
        dtype_str = metas[0]["dtype"]
        dtype = torch.bfloat16
        if "float32" in dtype_str:
            dtype = torch.float32
        elif "float16" in dtype_str:
            dtype = torch.float16

        ndim = len(metas[0]["offsets"])
        max_dim0 = max(m["offsets"][0] + m["sizes"][0] for m in metas)
        full_shape = [max_dim0] + [metas[0]["sizes"][d] for d in range(1, ndim)]

        if strict:
            sorted_m = sorted(metas, key=lambda m: m["offsets"][0])
            expect = 0
            for m in sorted_m:
                if m["offsets"][0] != expect:
                    raise RuntimeError(f"[STRICT] {name}: gap at dim0 expected={expect} got={m['offsets'][0]}")
                expect += m["sizes"][0]
            if expect != full_shape[0]:
                raise RuntimeError(f"[STRICT] {name}: incomplete {expect}/{full_shape[0]}")

        output_tensors[name] = torch.zeros(full_shape, dtype=dtype)

    # Pass 2: stream write
    print("  [Pass 2] Streaming write...", flush=True)
    for rank_idx, fpath in rank_files:
        sd = load_pt(fpath, safe_list)
        for raw_name, obj in sd.items():
            name = clean_key(str(raw_name))
            local_shards = None
            if hasattr(obj, "local_shards") and callable(getattr(obj, "local_shards")):
                try:
                    local_shards = obj.local_shards()
                except Exception:
                    pass
            if not isinstance(local_shards, list) or name not in output_tensors:
                continue
            out = output_tensors[name]
            for sh in local_shards:
                st = getattr(sh, "tensor", None)
                sm = getattr(sh, "metadata", None)
                if st is None or sm is None:
                    continue
                offsets, sizes = get_offsets_sizes(sm)
                if offsets is None or sizes is None:
                    continue
                sl = tuple(slice(int(o), int(o)+int(s)) for o, s in zip(offsets, sizes))
                data = st.detach().contiguous().cpu()
                if data.dtype != out.dtype:
                    data = data.to(out.dtype)
                out[sl].copy_(data.reshape(*sizes))
                del data
        del sd
        gc.collect()

    merged = dict(output_tensors)
    for name, tensor in all_full_tensors.items():
        if name not in merged:
            merged[name] = tensor

    del output_tensors, all_shard_metas, all_full_tensors
    gc.collect()

    print(f"  Merged {len(merged)} params", flush=True)
    return merged, meta


# ============================================================
# Module navigation helper (VLM compatible)
# ============================================================

def _navigate_to_module(model, path):
    """
    Navigate to a module by dotted path, trying multiple VLM-compatible variants:
      - path as-is
      - without "model." prefix
      - with "model.language_model." instead of "model."
      - with "language_model." prefix
    """
    variants = [path]
    if path.startswith("model."):
        stripped = path[len("model."):]
        variants.append(stripped)
        variants.append("model.language_model." + stripped)
        variants.append("language_model." + stripped)
    else:
        variants.append("model." + path)
        variants.append("model.language_model." + path)

    for nav_path in variants:
        obj = model
        try:
            for part in nav_path.split("."):
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            return obj
        except (AttributeError, IndexError, KeyError):
            continue
    return None


# ============================================================
# LoRA merge: W_merged = W_base + B @ A * (alpha / rank)
# ============================================================

def merge_lora_into_model(model, merged_params, lora_rank, lora_alpha,
                          train_layer_indices, block_size=(128, 128)):
    """
    LoRA パラメータをベースモデルの weight に統合。

    チェックポイントのキー例:
      model.layers.77.self_attn.q_a_proj.lora_A  → shape (r, in_features)
      model.layers.77.self_attn.q_a_proj.lora_B  → shape (out_features, r)

    対応するベースモデルの weight:
      model.layers.77.self_attn.q_a_proj.weight   → shape (out_features, in_features)

    マージ: weight += lora_B @ lora_A * (alpha / rank)

    FP8 weight は事前に dequant 済みであること。
    """
    scaling = lora_alpha / lora_rank
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)

    # Collect lora_A/lora_B pairs
    lora_pairs: Dict[str, Dict[str, torch.Tensor]] = {}

    for name, tensor in merged_params.items():
        if ".lora_A" in name:
            mod_path = name.rsplit(".lora_A", 1)[0]
            lora_pairs.setdefault(mod_path, {})["lora_A"] = tensor
        elif ".lora_B" in name:
            mod_path = name.rsplit(".lora_B", 1)[0]
            lora_pairs.setdefault(mod_path, {})["lora_B"] = tensor

    if not lora_pairs:
        print("  [WARN] No lora_A/lora_B pairs found in checkpoint!", flush=True)
        return 0

    print(f"  Found {len(lora_pairs)} LoRA pairs, scaling={scaling} (α={lora_alpha}/r={lora_rank})",
          flush=True)

    merged_count = 0

    for mod_path, lora_dict in sorted(lora_pairs.items()):
        lora_A = lora_dict.get("lora_A")
        lora_B = lora_dict.get("lora_B")
        if lora_A is None or lora_B is None:
            print(f"  [WARN] Incomplete LoRA pair for {mod_path}", flush=True)
            continue

        # Navigate to the module (VLM compatible)
        module = _navigate_to_module(model, mod_path)
        if module is None:
            print(f"  [WARN] Cannot navigate to module: {mod_path}", flush=True)
            continue

        # Find the weight to merge into
        w = getattr(module, "weight", None)
        if w is None or not isinstance(w, torch.nn.Parameter):
            print(f"  [WARN] No weight Parameter on {mod_path}", flush=True)
            continue

        if w.dtype in fp8_dtypes:
            # Need to dequant first
            scale = getattr(module, "weight_scale_inv", None)
            if scale is not None:
                scale_data = scale.data if isinstance(scale, torch.nn.Parameter) else scale
                w_bf16 = _blockwise_dequant(w.data, scale_data, block_size)
                module.weight = torch.nn.Parameter(w_bf16, requires_grad=False)
                w = module.weight
            else:
                w.data = w.data.to(torch.bfloat16)

        # Merge: W += B @ A * scaling
        delta = (lora_B.to(torch.bfloat16) @ lora_A.to(torch.bfloat16)) * scaling
        w.data.add_(delta.to(w.dtype))

        merged_count += 1
        del delta

        if merged_count <= 5:
            print(f"    [LoRA-MERGE] {mod_path}: "
                  f"A={tuple(lora_A.shape)}, B={tuple(lora_B.shape)} → "
                  f"W{tuple(w.shape)} += B@A*{scaling:.4f}", flush=True)

    if merged_count > 5:
        print(f"    ... and {merged_count - 5} more modules", flush=True)

    print(f"  LoRA merge complete: {merged_count} modules updated", flush=True)
    return merged_count


# ============================================================
# Full fine-tune: apply checkpoint weights to model
# ============================================================

def find_base_key(name, model_keys):
    """
    checkpoint key → model key のマッピング。
    VLM アーキテクチャでは model.language_model.* prefix を試行。
    """
    candidates = [name]

    if name.startswith("model."):
        stripped = name[len("model."):]
        candidates.append(stripped)
        candidates.append("model.model." + stripped)
        # VLM: model.layers.* → model.language_model.layers.*
        candidates.append("model.language_model." + stripped)
        candidates.append("language_model." + stripped)
    else:
        candidates.append("model." + name)
        candidates.append("model.model." + name)
        candidates.append("model.language_model." + name)
        candidates.append("language_model." + name)

    for c in candidates:
        if c in model_keys:
            return c
    return None


def apply_full_finetune_params(model, merged_params):
    """Full fine-tune の checkpoint weight をモデルに適用。"""
    model_keys = set(n for n, _ in model.named_parameters())
    model_keys.update(n for n, _ in model.named_buffers())

    # lora_A/lora_B は除外 (LoRA モードで間違って呼ばれた場合の安全策)
    regular_params = {k: v for k, v in merged_params.items()
                      if "lora_A" not in k and "lora_B" not in k}

    applied = 0
    skipped = []

    for name, tensor in regular_params.items():
        bk = find_base_key(name, model_keys)
        if bk is None:
            skipped.append(name)
            continue

        # Navigate to actual param
        parts = bk.split(".")
        obj = model
        try:
            for p in parts[:-1]:
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)
        except (AttributeError, IndexError):
            skipped.append(name)
            continue

        attr = parts[-1]
        target = getattr(obj, attr, None)
        if target is None:
            skipped.append(name)
            continue

        if isinstance(target, torch.nn.Parameter):
            if tensor.shape == target.shape:
                target.data.copy_(tensor.to(target.dtype))
                applied += 1
                if applied <= 5:
                    print(f"    [{name}] → [{bk}]  shape={tuple(tensor.shape)}", flush=True)
            else:
                print(f"    [WARN] shape mismatch: {bk} "
                      f"ckpt={tuple(tensor.shape)} model={tuple(target.shape)}", flush=True)
                skipped.append(name)
        elif isinstance(target, torch.Tensor):
            # buffer
            if tensor.shape == target.shape:
                target.copy_(tensor.to(target.dtype))
                applied += 1
            else:
                skipped.append(name)

    if applied > 5:
        print(f"    ... and {applied - 5} more params", flush=True)

    if skipped:
        print(f"    [WARN] {len(skipped)} keys not matched:", flush=True)
        for s in skipped[:10]:
            print(f"      - {s}", flush=True)
        if len(skipped) > 10:
            print(f"      ... and {len(skipped)-10} more", flush=True)

    print(f"  Applied {applied}/{len(regular_params)} params", flush=True)
    return applied


# ============================================================
# Streaming safetensors save (bf16 mode)
# ============================================================

def collect_layer_state(layer, layer_prefix):
    sd = OrderedDict()
    for name, param in layer.named_parameters():
        full_name = f"{layer_prefix}.{name}"
        # LoRA params (lora_A, lora_B) をスキップ — 既に weight にマージ済み
        if "lora_A" in name or "lora_B" in name:
            continue
        sd[full_name] = param.data.contiguous().cpu()
    for name, buf in layer.named_buffers():
        full_name = f"{layer_prefix}.{name}"
        sd[full_name] = buf.data.contiguous().cpu()
    return sd


def evict_layer_to_meta(layer):
    for name, param in layer.named_parameters():
        try:
            param.data = torch.empty(param.shape, dtype=param.dtype, device="meta")
        except (RuntimeError, TypeError):
            # FP8 quantized tensors may not support meta device — use empty bf16 instead
            try:
                param.data = torch.empty(param.shape, dtype=torch.bfloat16, device="meta")
            except (RuntimeError, TypeError):
                # Last resort: just release the storage
                param.data = torch.empty(0, device="cpu")
    for name, buf in layer.named_buffers():
        def _safe_meta(shape, dtype):
            try:
                return torch.empty(shape, dtype=dtype, device="meta")
            except (RuntimeError, TypeError):
                try:
                    return torch.empty(shape, dtype=torch.bfloat16, device="meta")
                except (RuntimeError, TypeError):
                    return torch.empty(0, device="cpu")

        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, buf_name = parts
            parent = dict(layer.named_modules()).get(parent_name)
            if parent is not None and buf_name in parent._buffers:
                parent._buffers[buf_name] = _safe_meta(buf.shape, buf.dtype)
                continue
        if name in layer._buffers:
            layer._buffers[name] = _safe_meta(buf.shape, buf.dtype)


def collect_toplevel_state(model, inner_model, layers_attr_name):
    sd = OrderedDict()
    layers_module = getattr(inner_model, layers_attr_name)
    inner_prefix = "model." if (hasattr(model, "model") and model.model is inner_model) else ""

    def _collect(module, prefix):
        for pname, param in module.named_parameters():
            if param.device.type == "meta" or param.numel() == 0:
                continue
            sd[f"{prefix}.{pname}"] = param.data.contiguous().cpu()
        for bname, buf in module.named_buffers():
            if buf.device.type == "meta" or buf.numel() == 0:
                continue
            sd[f"{prefix}.{bname}"] = buf.data.contiguous().cpu()

    for child_name, child_module in inner_model.named_children():
        if child_module is layers_module:
            continue
        _collect(child_module, f"{inner_prefix}{child_name}")

    if hasattr(model, "model") and model.model is inner_model:
        for child_name, child_module in model.named_children():
            if child_module is inner_model:
                continue
            _collect(child_module, child_name)

    return sd


def _is_quant_artifact_key(key):
    """FP8量子化の付随テンソル (dequant後は不要) かどうか。

    dequantize後も weight_scale_inv 等のパラメータはモジュール上に残るため、
    BF16出力に含めるとロード時に量子化モデルと誤認される。除外する。
    """
    return key.endswith("_scale_inv") or "activation_scale" in key


def streaming_save_bf16(model, num_layers, out_dir, block_size=(128, 128)):
    """
    レイヤー単位ストリーミング BF16 保存。
    各レイヤー: dequant(未変換分) → BF16化 → safetensors書き出し → meta退避
    """
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors required: pip install safetensors")

    os.makedirs(out_dir, exist_ok=True)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        inner_model = model.model
        layers = model.model.layers
        layers_prefix = "model.layers"
        layers_attr_name = "layers"
    elif hasattr(model, "layers"):
        inner_model = model
        layers = model.layers
        layers_prefix = "layers"
        layers_attr_name = "layers"
    else:
        raise RuntimeError("Cannot find model.layers")

    weight_map = {}
    shard_files = []
    total_dequant = 0

    for layer_idx in range(num_layers):
        t0 = time.time()
        layer = layers[layer_idx]
        layer_prefix = f"{layers_prefix}.{layer_idx}"

        # Dequant any remaining FP8 in this layer
        n_dq = dequantize_layer_inplace(layer, block_size)
        total_dequant += n_dq

        # BF16 化 (non-BF16 残り)
        for pname, param in layer.named_parameters():
            if param.dtype != torch.bfloat16 and param.device.type != "meta":
                param.data = param.data.to(torch.bfloat16)

        # Collect & write
        layer_sd = collect_layer_state(layer, layer_prefix)
        layer_sd = {k: v for k, v in layer_sd.items()
                    if v.device.type != "meta" and v.numel() > 0
                    and not _is_quant_artifact_key(k)}
        for k in layer_sd:
            if layer_sd[k].dtype != torch.bfloat16:
                layer_sd[k] = layer_sd[k].to(torch.bfloat16)

        shard_name = f"model-{layer_idx:05d}-of-{num_layers:05d}.safetensors"
        shard_path = os.path.join(out_dir, shard_name)
        safetensors_save_file(layer_sd, shard_path)
        shard_files.append(shard_name)
        for k in layer_sd:
            weight_map[k] = shard_name

        shard_mb = os.path.getsize(shard_path) / (1024 * 1024)
        evict_layer_to_meta(layer)
        del layer_sd
        gc.collect()

        elapsed = time.time() - t0
        if (layer_idx + 1) % 5 == 0 or layer_idx == num_layers - 1 or layer_idx == 0:
            print(f"    layer {layer_idx:3d}/{num_layers}: "
                  f"dequant={n_dq}, {shard_mb:.0f}MB, {elapsed:.1f}s", flush=True)

    # Top-level modules
    print(f"  Processing top-level modules...", flush=True)
    for child_name, child_module in inner_model.named_children():
        if child_module is getattr(inner_model, layers_attr_name):
            continue
        n_dq = dequantize_module_params(child_module, block_size)
        total_dequant += n_dq

    if hasattr(model, "model") and model.model is inner_model:
        for child_name, child_module in model.named_children():
            if child_module is inner_model:
                continue
            n_dq = dequantize_module_params(child_module, block_size)
            total_dequant += n_dq

    toplevel_sd = collect_toplevel_state(model, inner_model, layers_attr_name)
    toplevel_sd = {k: v for k, v in toplevel_sd.items()
                   if v.device.type != "meta" and v.numel() > 0
                   and not _is_quant_artifact_key(k)}
    for k in toplevel_sd:
        if toplevel_sd[k].dtype != torch.bfloat16:
            toplevel_sd[k] = toplevel_sd[k].to(torch.bfloat16)

    if toplevel_sd:
        shard_name = "model-toplevel.safetensors"
        shard_path = os.path.join(out_dir, shard_name)
        safetensors_save_file(toplevel_sd, shard_path)
        shard_files.append(shard_name)
        for k in toplevel_sd:
            weight_map[k] = shard_name

    del toplevel_sd
    gc.collect()

    # Index
    total_size = sum(
        os.path.getsize(os.path.join(out_dir, f))
        for f in shard_files if os.path.exists(os.path.join(out_dir, f))
    )
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"  Streaming BF16 save: {len(shard_files)} shards, "
          f"{total_size/(1024**3):.1f}GB, dequanted={total_dequant}", flush=True)


# ============================================================
# Phase 5: VLM 補完 (config / tokenizer / visual encoder)
# ============================================================

def copy_base_nonweight_files(base_model_dir, out_dir):
    """
    ベースモデルから weight (*.safetensors / model*.bin) 以外の全ファイルをコピー。
    out_dir の model.safetensors.index.json は保護する (上書きしない)。

    修正対象:
      - config.json          (VLM top-level ↔ text submodel の model_type 不一致)
      - tokenizer_config.json (TokenizersBackend 参照エラー)
      - tokenizer.json, vocab.json, merges.txt, special_tokens_map.json
      - preprocessor_config.json, processor 系ファイル (VLM 画像処理)
      - chat_template.jinja, generation_config.json
      - カスタム .py ファイル (modeling_*.py 等)
    """
    # Backup merged index.json (Phase 4 で生成済み)
    merged_index_path = os.path.join(out_dir, "model.safetensors.index.json")
    merged_index_bak = merged_index_path + ".merged_bak"
    if os.path.exists(merged_index_path):
        shutil.copy2(merged_index_path, merged_index_bak)

    copied = []
    for fname in os.listdir(base_model_dir):
        src = os.path.join(base_model_dir, fname)
        if not os.path.isfile(src):
            continue
        # Skip weight files
        if fname.endswith(".safetensors"):
            continue
        if fname.endswith(".bin") and "model" in fname:
            continue

        dst = os.path.join(out_dir, fname)
        shutil.copy2(src, dst)
        copied.append(fname)

    # Restore merged index.json (base のを上書きしてしまった場合の保護)
    if os.path.exists(merged_index_bak):
        shutil.copy2(merged_index_bak, merged_index_path)
        os.remove(merged_index_bak)

    print(f"  Copied {len(copied)} non-weight files from base:", flush=True)
    for f in sorted(copied):
        print(f"    {f}", flush=True)
    return copied


def copy_visual_encoder_weights(base_model_dir, out_dir):
    """
    ベースモデルの visual encoder weight をマージ出力に補完。

    save_pretrained が language_model サブモデルのみ保存した場合、
    visual.* / vision_tower.* 等の weight が欠落する。
    ベースから該当シャードをコピーし、マージ済み index.json に追加。

    Returns:
        (num_keys, num_shards): 追加された key 数とシャード数
    """
    base_index_path = os.path.join(base_model_dir, "model.safetensors.index.json")
    merged_index_path = os.path.join(out_dir, "model.safetensors.index.json")

    if not os.path.exists(base_index_path) or not os.path.exists(merged_index_path):
        print("  [WARN] Cannot find index files, skipping visual weight copy", flush=True)
        return 0, 0

    with open(base_index_path) as f:
        base_index = json.load(f)
    with open(merged_index_path) as f:
        merged_index = json.load(f)

    merged_keys = set(merged_index.get("weight_map", {}).keys())

    # Visual / multi-modal key prefixes (各モデルアーキテクチャに対応)
    VISUAL_PREFIXES = (
        "visual.", "model.visual.",
        "vision_tower.", "model.vision_tower.",
        "multi_modal_projector.", "model.multi_modal_projector.",
        "image_newline", "model.image_newline",
    )

    visual_keys = {}
    visual_shards = set()
    for key, shard in base_index.get("weight_map", {}).items():
        is_visual = any(key.startswith(pfx) for pfx in VISUAL_PREFIXES)
        if is_visual and key not in merged_keys:
            visual_keys[key] = shard
            visual_shards.add(shard)

    if not visual_keys:
        print("  No missing visual weights found", flush=True)
        return 0, 0

    print(f"  Found {len(visual_keys)} missing visual keys in {len(visual_shards)} shard(s)",
          flush=True)

    # Copy shards from base
    for shard in sorted(visual_shards):
        src = os.path.join(base_model_dir, shard)
        dst = os.path.join(out_dir, shard)
        if os.path.exists(dst):
            print(f"    [SKIP] {shard} already exists in output", flush=True)
            continue
        if not os.path.exists(src):
            print(f"    [WARN] {shard} not found in base!", flush=True)
            continue
        t0 = time.time()
        shutil.copy2(src, dst)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"    Copied {shard} ({size_mb:.0f} MB, {time.time()-t0:.1f}s)", flush=True)

    # Update merged index
    merged_index["weight_map"].update(visual_keys)

    # Recalculate total size
    all_shards = set(merged_index["weight_map"].values())
    total_size = 0
    for shard in all_shards:
        shard_path = os.path.join(out_dir, shard)
        if os.path.exists(shard_path):
            total_size += os.path.getsize(shard_path)
    merged_index["metadata"]["total_size"] = total_size

    with open(merged_index_path, "w") as f:
        json.dump(merged_index, f, indent=2)

    print(f"  Updated index: {len(merged_index['weight_map'])} total keys, "
          f"{total_size / (1024**3):.1f} GB", flush=True)

    return len(visual_keys), len(visual_shards)


def verify_output(base_model_dir, out_dir):
    """出力ディレクトリの健全性チェック。"""
    issues = []

    # 1. config.json — model_type がベースと一致するか
    config_path = os.path.join(out_dir, "config.json")
    if not os.path.exists(config_path):
        issues.append("config.json missing")
    else:
        with open(config_path) as f:
            config = json.load(f)
        base_config_path = os.path.join(base_model_dir, "config.json")
        if os.path.exists(base_config_path):
            with open(base_config_path) as f:
                base_config = json.load(f)
            if config.get("model_type") != base_config.get("model_type"):
                issues.append(f"config.json model_type mismatch: "
                              f"output={config.get('model_type')} vs "
                              f"base={base_config.get('model_type')}")

    # 2. tokenizer ファイル
    if not os.path.exists(os.path.join(out_dir, "tokenizer_config.json")):
        issues.append("tokenizer_config.json missing")

    # 3. index.json が参照するシャードが全て存在するか
    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shards = set(index.get("weight_map", {}).values())
        missing_shards = [s for s in shards
                          if not os.path.exists(os.path.join(out_dir, s))]
        if missing_shards:
            issues.append(f"index references {len(missing_shards)} missing shard(s): "
                          f"{missing_shards[0]}...")

    # 4. VLM: visual weights が含まれるか
    if detect_vlm(base_model_dir):
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            has_visual = any("visual" in k.lower()
                            for k in index.get("weight_map", {}))
            if not has_visual:
                issues.append("VLM model but no visual.* weights in output")

    if issues:
        print(f"\n  ⚠️  Verification issues ({len(issues)}):", flush=True)
        for iss in issues:
            print(f"    - {iss}", flush=True)
        return False
    else:
        print(f"\n  ✅ Verification passed", flush=True)
        return True


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Full model merge for vLLM (GLM-5/Qwen3.5/etc)")
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--checkpoint_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_layers", type=int, default=78)
    ap.add_argument("--world_size", type=int, default=0)
    ap.add_argument("--output_dtype", type=str, default="keep",
                    choices=["keep", "bf16"])
    ap.add_argument("--max_shard_size", type=str, default="5GB",
                    help="save_pretrained shard size (keep mode only)")
    ap.add_argument("--strict", action="store_true", default=True)
    # LoRA overrides (normally read from meta, but can be forced)
    ap.add_argument("--force_lora_rank", type=int, default=0,
                    help="Override lora_rank from meta (0=use meta)")
    ap.add_argument("--force_lora_alpha", type=float, default=0,
                    help="Override lora_alpha from meta (0=use meta)")
    ap.add_argument("--skip_vlm_fix", action="store_true", default=False,
                    help="Skip VLM post-processing (config/visual copy)")
    args = ap.parse_args()

    is_vlm = detect_vlm(args.base_model)

    print("=" * 64)
    print("  Full Model Merge (for vLLM)")
    print("=" * 64)
    print(f"  base_model     : {args.base_model}")
    print(f"  checkpoint_dir : {args.checkpoint_dir}")
    print(f"  out_dir        : {args.out_dir}")
    print(f"  num_layers     : {args.num_layers}")
    print(f"  output_dtype   : {args.output_dtype}")
    print(f"  VLM detected   : {is_vlm}")
    print(f"  hack_enabled   : {HACK_ENABLED}")
    print("=" * 64, flush=True)

    total_start = time.time()

    # ================================================================
    # Phase 1: ストリーミングマージ
    # ================================================================
    print("\n" + "=" * 64)
    print("  Phase 1: Streaming merge of checkpoint parameters")
    print("=" * 64, flush=True)

    merged_params, ckpt_meta = streaming_merge_params(
        args.checkpoint_dir, world_size=args.world_size, strict=args.strict,
    )

    # Detect mode from meta
    use_lora = ckpt_meta.get("use_lora", False)
    train_layers = ckpt_meta.get("train_layers", [args.num_layers - 1])
    train_modules = ckpt_meta.get("train_modules", "attn")
    lora_rank = args.force_lora_rank or ckpt_meta.get("lora_rank", 16)
    lora_alpha = args.force_lora_alpha or ckpt_meta.get("lora_alpha", 32)

    # Auto-detect LoRA from param names if meta is missing
    has_lora_params = any("lora_A" in k or "lora_B" in k for k in merged_params)
    if has_lora_params and not use_lora:
        print("  [AUTO] Detected lora_A/lora_B in params → LoRA mode", flush=True)
        use_lora = True

    mode_str = f"LoRA (r={lora_rank}, α={lora_alpha})" if use_lora else f"Full ({train_modules})"
    print(f"\n  Mode         : {mode_str}")
    print(f"  Train layers : {train_layers}")
    print(f"  Params count : {len(merged_params)}")
    for name, t in sorted(merged_params.items()):
        print(f"    {name}: {tuple(t.shape)} {t.dtype}", flush=True)

    # ================================================================
    # Phase 2: ベースモデルロード
    # ================================================================
    print("\n" + "=" * 64)
    print("  Phase 2: Loading base model to CPU")
    print("=" * 64, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    t0 = time.time()
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)

    import transformers.modeling_utils as _tm_utils
    _orig_finalize = _tm_utils.PreTrainedModel._finalize_model_loading

    @classmethod
    def _skip_finalize(cls, model, *a, **kw):
        return {"missing_keys": [], "unexpected_keys": [],
                "mismatched_keys": [], "error_msgs": []}

    _tm_utils.PreTrainedModel._finalize_model_loading = _skip_finalize
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, config=config,
            device_map=None, trust_remote_code=True, low_cpu_mem_usage=True,
        )
    finally:
        _tm_utils.PreTrainedModel._finalize_model_loading = _orig_finalize

    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s", flush=True)

    # Show model structure for VLM debugging
    if is_vlm:
        print(f"  Model type: {type(model).__name__}", flush=True)
        top_children = [(n, type(m).__name__) for n, m in model.named_children()]
        print(f"  Top-level children: {top_children}", flush=True)

    # ================================================================
    # Phase 3: モードに応じた適用
    # ================================================================
    print("\n" + "=" * 64)
    print(f"  Phase 3: Apply checkpoint ({mode_str})")
    print("=" * 64, flush=True)

    # Find layers — VLM ではネスト構造が異なるため複数パターンを試行
    layers = None
    for attr_chain in [
        ("model", "model", "layers"),                    # Standard: model.model.layers
        ("model", "layers"),                              # model.layers
        ("language_model", "model", "layers"),            # VLM: language_model.model.layers
        ("model", "language_model", "model", "layers"),   # model.language_model.model.layers
        ("model", "language_model", "layers"),            # model.language_model.layers
        ("layers",),                                      # layers
    ]:
        obj = model
        try:
            for attr in attr_chain:
                obj = getattr(obj, attr)
            if hasattr(obj, "__len__"):
                layers = obj
                print(f"  Found layers at: {'.'.join(attr_chain)} ({len(layers)} layers)", flush=True)
                break
        except AttributeError:
            continue

    if layers is None:
        raise RuntimeError("Cannot find model layers. Tried: model.model.layers, "
                           "model.language_model.model.layers, etc.")

    if use_lora:
        # LoRA: trainable layers を dequant → lora_B @ lora_A を weight に加算
        print(f"  Dequantizing layers {train_layers} for LoRA merge...", flush=True)
        for lidx in train_layers:
            n_dq = dequantize_layer_inplace(layers[lidx])
            print(f"    layer {lidx}: {n_dq} tensors dequanted", flush=True)
        gc.collect()

        merge_lora_into_model(
            model, merged_params,
            lora_rank=lora_rank, lora_alpha=lora_alpha,
            train_layer_indices=train_layers,
        )
    else:
        # Full fine-tune: trainable layers を dequant → checkpoint の weight で置換
        print(f"  Dequantizing layers {train_layers}...", flush=True)
        for lidx in train_layers:
            n_dq = dequantize_layer_inplace(layers[lidx])
            print(f"    layer {lidx}: {n_dq} tensors dequanted", flush=True)
        gc.collect()

        apply_full_finetune_params(model, merged_params)

    del merged_params
    gc.collect()

    # ================================================================
    # Phase 4: Save
    # ================================================================
    print("\n" + "=" * 64)
    print(f"  Phase 4: Save → {args.out_dir}")
    print("=" * 64, flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.output_dtype == "bf16":
        print("  Mode: streaming layer-by-layer BF16 save")
        print(f"  Peak RAM ≈ FP8 base + 1 layer BF16", flush=True)
        streaming_save_bf16(model, args.num_layers, args.out_dir)
    else:
        print("  Mode: save_pretrained (keep original dtypes)", flush=True)
        t0 = time.time()
        model.save_pretrained(
            args.out_dir, safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        print(f"  Saved in {time.time()-t0:.1f}s", flush=True)

    # ================================================================
    # Phase 5: VLM 補完 + 検証
    # ================================================================
    print("\n" + "=" * 64)
    print(f"  Phase 5: Post-processing (VLM={is_vlm})")
    print("=" * 64, flush=True)

    # 5a. ベースから非 weight ファイルをコピー
    #     save_pretrained が text submodel の config/tokenizer を保存する問題を修正
    #     index.json はバックアップ→リストアで保護
    print("\n  [5a] Copying non-weight files from base...", flush=True)
    copy_base_nonweight_files(args.base_model, args.out_dir)

    # BF16出力時: base からコピーされた config.json に quantization_config が
    # 残っていると、ロード側が FP8 量子化モデルと誤認する。削除する。
    if args.output_dtype == "bf16":
        out_config_path = os.path.join(args.out_dir, "config.json")
        if os.path.exists(out_config_path):
            with open(out_config_path) as f:
                out_config = json.load(f)
            removed = False
            if "quantization_config" in out_config:
                del out_config["quantization_config"]
                removed = True
            for sub_key in ("text_config", "language_config", "llm_config"):
                sub = out_config.get(sub_key)
                if isinstance(sub, dict) and "quantization_config" in sub:
                    del sub["quantization_config"]
                    removed = True
            if removed:
                with open(out_config_path, "w") as f:
                    json.dump(out_config, f, indent=2)
                print("  Removed quantization_config from config.json (bf16 output)",
                      flush=True)

    # 5b. Visual encoder の weight をベースからコピー (VLM のみ)
    #     save_pretrained が language_model サブモデルのみ保存 → visual.* 欠落
    if is_vlm and not args.skip_vlm_fix:
        print("\n  [5b] Copying visual encoder weights from base...", flush=True)
        n_keys, n_shards = copy_visual_encoder_weights(args.base_model, args.out_dir)
        if n_keys > 0:
            print(f"  Added {n_keys} visual keys from {n_shards} shard(s)", flush=True)
    else:
        if not is_vlm:
            print("\n  [5b] Not a VLM model, skipping visual weight copy", flush=True)
        else:
            print("\n  [5b] Skipped (--skip_vlm_fix)", flush=True)

    # 5c. 検証
    print("\n  [5c] Verifying output...", flush=True)
    verify_output(args.base_model, args.out_dir)

    # Summary
    total_elapsed = time.time() - total_start
    total_size = sum(
        os.path.getsize(os.path.join(args.out_dir, f))
        for f in os.listdir(args.out_dir)
        if os.path.isfile(os.path.join(args.out_dir, f))
    )

    print("\n" + "=" * 64)
    print(f"  ✅ マージ完了!")
    print(f"  出力先     : {args.out_dir}")
    print(f"  出力サイズ : {total_size / (1024**3):.1f} GB")
    print(f"  モード     : {mode_str}")
    print(f"  VLM        : {is_vlm}")
    print(f"  dtype      : {args.output_dtype}")
    print(f"  合計時間   : {int(total_elapsed // 60)}m {int(total_elapsed % 60)}s")
    print("=" * 64)
    print(f"\n  vllm serve {args.out_dir} --trust-remote-code")


if __name__ == "__main__":
    main()
