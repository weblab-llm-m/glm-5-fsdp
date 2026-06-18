#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partial-layer SFT for GLM-5-FP8 (FSDP).

Supports:
  - Full fine-tuning of selected modules (attn, mlp, or full layer)
  - LoRA adaptation (--use_lora) — base FP8 weights stay frozen, ~100x fewer params
  - Configurable number of trainable layers (--train_layers N)
  - Sequence packing (--packing) for GPU utilization
  - Gradient accumulation (--gradient_accumulation_steps)
  - W&B logging (--wandb_project)

Model loading strategy:
  - rank 0: from_pretrained() loads FP8 weights to CPU (~760GB)
  - rank != 0: from_pretrained() with meta context → zero CPU RAM
  - FSDP sync_module_states=True broadcasts from rank 0 to all ranks

FP8 handling:
  - Forward pre/post hooks on submodules for block-wise dequant
  - Full fine-tune: trainable layers dequanted FP8 → BF16
  - LoRA: base weights stay FP8, LoRA A/B added as BF16
  - Detach hook at frozen→trainable boundary eliminates activation storage
"""

import os
import json
import math
import random
import argparse
import datetime
import time
from typing import List, Tuple
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import InitProcessGroupKwargs, GradientAccumulationPlugin
from torch.distributed.fsdp import (
    ShardingStrategy,
    StateDictType,
    ShardedStateDictConfig,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    _FSDP_AVAILABLE = True
except Exception:
    _FSDP_AVAILABLE = False

# Monkey-patch: FSDP _construct_padding_tensor doesn't support FP8 dtypes
# (torch.ones * torch.ones triggers "mul_cuda" not implemented for Float8_e4m3fn).
# When world_size doesn't evenly divide the flat param numel, FSDP needs padding
# and hits this code path. Fix: create padding in BF16, then cast to FP8.
import torch.distributed.fsdp._flat_param as _fsdp_flat_param_mod
_orig_construct_padding = _fsdp_flat_param_mod._construct_padding_tensor

def _fp8_safe_construct_padding(padding_numel, dtype, requires_grad, device):
    _fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    if dtype in _fp8_dtypes:
        return torch.zeros(padding_numel, dtype=torch.bfloat16, device=device).to(dtype)
    return _orig_construct_padding(padding_numel, dtype, requires_grad, device)

_fsdp_flat_param_mod._construct_padding_tensor = _fp8_safe_construct_padding


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_sdp_backends():
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


# ============================================================
# FP8 → BF16 dequantization (trainable layer only)
# ============================================================

def _blockwise_dequant(w_fp8, scale_inv, block_size=(128, 128)):
    """Vectorized block-wise FP8 → BF16 dequantization.
    2D: fully vectorized, in-place mul.
    3D: per-expert to limit peak memory."""
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


# ============================================================
# On-the-fly FP8 dequant hooks for FSDP forward pass
# ============================================================

def install_fp8_dequant_hooks(model, block_size=(128, 128)):
    """
    FP8 weight を持つ個別 submodule に forward hook を登録し、
    forward 時に on-the-fly で FP8→BF16 dequant を行う。

    重要: hook は decoder layer (= FSDP unit) ではなく、
    その内側の submodule (q_a_proj, kv_b_proj 等) に登録する。
    理由: FSDP unit に登録すると、FSDP unshard より先に hook が走り、
    weight がまだ sharded 状態で dequant してしまう。
    Submodule は FSDP boundary の内側なので、unshard 完了後に実行される。

    FSDP hook 実行順序:
      1. FSDP pre-hook (all-gather → full FP8 weight on GPU)
      2. layer.forward() 開始
      3. submodule.forward() 呼び出し
         → OUR pre_hook: FP8 data 保存、BF16 dequant 版に差し替え
         → F.linear(input, weight_bf16) → 正常計算 ✓
         → OUR post_hook: FP8 data 復元
      4. FSDP post-hook (reshard FP8)
    """
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    hooks = []

    # Diagnostic counter (shared across hooks)
    _diag = {"count": 0, "done": False}

    def _make_pre_hook(w_attr, s_attr, bs, mod_name=""):
        def hook(module, args):
            w = getattr(module, w_attr)
            if w is None or w.dtype not in fp8_dtypes:
                return
            scale = getattr(module, s_attr, None)
            if scale is None:
                return
            scale_data = scale.data if isinstance(scale, torch.nn.Parameter) else scale
            if not hasattr(module, "_fp8_backups"):
                module._fp8_backups = {}
            module._fp8_backups[w_attr] = w.data

            # Diagnostic: first 5 hooks on first forward, rank 0 only
            do_diag = (not _diag["done"] and _diag["count"] < 5
                       and torch.distributed.is_initialized()
                       and torch.distributed.get_rank() == 0)

            if do_diag:
                _diag["count"] += 1
                print(f"  [HOOK-DIAG] {mod_name}.{w_attr}: "
                      f"w.dtype={w.dtype}, w.shape={tuple(w.shape)}, "
                      f"w.device={w.device}, "
                      f"scale.shape={tuple(scale_data.shape)}, "
                      f"scale.dtype={scale_data.dtype}, "
                      f"scale_nan={torch.isnan(scale_data.float().view(-1)[:100]).any().item()}, "
                      f"scale_min={scale_data.float().view(-1)[:100].min().item():.6f}, "
                      f"scale_max={scale_data.float().view(-1)[:100].max().item():.6f}",
                      flush=True)

            w.data = _blockwise_dequant(w.data, scale_data, bs)

            if do_diag:
                flat = w.data.view(-1)[:1000].float()
                print(f"           → dequant: dtype={w.data.dtype}, "
                      f"has_nan={torch.isnan(flat).any().item()}, "
                      f"has_inf={torch.isinf(flat).any().item()}, "
                      f"min={flat[~torch.isnan(flat)].min().item():.6f}, "
                      f"max={flat[~torch.isnan(flat)].max().item():.6f}",
                      flush=True)
                if _diag["count"] >= 5:
                    _diag["done"] = True
        return hook

    def _make_post_hook(w_attr):
        def hook(module, args, output):
            backups = getattr(module, "_fp8_backups", {})
            if w_attr in backups:
                getattr(module, w_attr).data = backups.pop(w_attr)
            if hasattr(module, "_fp8_backups") and not module._fp8_backups:
                del module._fp8_backups
            return output
        return hook

    installed = 0
    for name, mod in model.named_modules():
        # Case 1: Standard Linear with weight + weight_scale_inv
        if (hasattr(mod, "weight")
                and isinstance(mod.weight, (torch.nn.Parameter, torch.Tensor))
                and mod.weight.dtype in fp8_dtypes
                and hasattr(mod, "weight_scale_inv")):
            h1 = mod.register_forward_pre_hook(_make_pre_hook("weight", "weight_scale_inv", block_size, name))
            h2 = mod.register_forward_hook(_make_post_hook("weight"))
            hooks.extend([h1, h2])
            installed += 1
            continue

        # Case 2: Expert modules with gate_up_proj + gate_up_proj_scale_inv etc.
        if not hasattr(mod, "_parameters"):
            continue
        for attr in list(mod._parameters.keys()):
            p = mod._parameters.get(attr)
            if p is None or p.dtype not in fp8_dtypes:
                continue
            if attr.endswith("_scale_inv"):
                continue
            scale_attr = attr + "_scale_inv"
            if hasattr(mod, scale_attr):
                h1 = mod.register_forward_pre_hook(_make_pre_hook(attr, scale_attr, block_size, name))
                h2 = mod.register_forward_hook(_make_post_hook(attr))
                hooks.extend([h1, h2])
                installed += 1

    return hooks, installed


def dequantize_single_layer(model, layer_idx, rank=0, block_size=(128, 128)):
    """
    1つの decoder layer 内の全 FP8 Parameters を BF16 に dequant する。
    trainable layer のみ対象。

    ★ FSDP は1つの wrapping unit 内で uniform dtype を要求する。
    trainable layer 内のモジュール選択（attn/mlp/full）に関わらず、
    layer 全体を BF16 に統一する必要がある。
    モジュール選択は mark_trainable() の requires_grad で行う。
    """
    import gc
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer = model.model.layers[layer_idx]
    elif hasattr(model, "layers"):
        layer = model.layers[layer_idx]
    else:
        raise RuntimeError("Cannot find model.layers")

    converted = 0
    for mod_name, module in layer.named_modules():
        # FP8Linear-style: weight + weight_scale_inv
        if (hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter)
                and module.weight.dtype in fp8_dtypes
                and hasattr(module, "weight_scale_inv")):
            w = module.weight.data
            if w.device.type == "meta":
                module.weight = torch.nn.Parameter(
                    torch.empty(w.shape, dtype=torch.bfloat16, device="meta"),
                    requires_grad=False,
                )
            else:
                bs = getattr(module, "block_size", block_size)
                scale = module.weight_scale_inv
                scale_data = scale.data if isinstance(scale, torch.nn.Parameter) else scale
                w_bf16 = _blockwise_dequant(w, scale_data, bs)
                module.weight = torch.nn.Parameter(w_bf16, requires_grad=False)
                if rank == 0:
                    print(f"  [DEQUANT] {mod_name}: {tuple(w.shape)} FP8→BF16", flush=True)
                del w, w_bf16
                gc.collect()
            converted += 1

        # Expert composite tensors (gate_up_proj, down_proj, etc.)
        for attr in list(module._parameters.keys()):
            param = module._parameters[attr]
            if param is None or param.dtype not in fp8_dtypes:
                continue
            if attr.endswith("_scale_inv"):
                continue
            scale_attr = attr + "_scale_inv"
            scale = getattr(module, scale_attr, None)

            if param.device.type == "meta":
                module._parameters[attr] = torch.nn.Parameter(
                    torch.empty(param.shape, dtype=torch.bfloat16, device="meta"),
                    requires_grad=False,
                )
            elif scale is not None:
                scale_data = scale.data if isinstance(scale, torch.nn.Parameter) else scale
                old_data = param.data
                w_bf16 = _blockwise_dequant(old_data, scale_data, block_size)
                module._parameters[attr] = torch.nn.Parameter(w_bf16, requires_grad=False)
                if rank == 0:
                    print(f"  [DEQUANT] {mod_name}.{attr}: {tuple(old_data.shape)} FP8→BF16", flush=True)
                del old_data, w_bf16
                gc.collect()
            else:
                old_data = param.data
                module._parameters[attr] = torch.nn.Parameter(
                    old_data.to(torch.bfloat16), requires_grad=False,
                )
                del old_data
            converted += 1

    gc.collect()
    return converted


def convert_frozen_non_fp8_to_buffers(model, trainable_layer_indices):
    """
    frozen layers の非 FP8 Parameters を buffer に変換。
    → frozen layers の FSDP flat param group には FP8 Parameters のみ残る → uniform dtype ✓

    ★ trainable layers は全体をスキップ。
    理由: dequantize_single_layer() で layer 全体を BF16 に変換済みなので、
    trainable layer の FSDP unit は uniform BF16。
    buffer 化すると FSDP が shard しなくなり、MoE の場合 OOM する。

    ★ embed_tokens, norm, lm_head 等のトップレベル modules も除外。
    """
    if isinstance(trainable_layer_indices, int):
        trainable_layer_indices = [trainable_layer_indices]

    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    count = 0

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        inner_model = model.model
    elif hasattr(model, "layers"):
        layers = model.layers
        inner_model = model
    else:
        raise RuntimeError("Cannot find model.layers")

    # Build set of module IDs to skip (entire trainable layers + top-level modules)
    skip_module_ids = set()
    for idx in trainable_layer_indices:
        for m in layers[idx].modules():
            skip_module_ids.add(id(m))

    # Skip top-level modules: embed_tokens, lm_head, final norm, etc.
    top_level_attr_names = (
        "embed_tokens", "lm_head", "norm", "final_layernorm",
        "embed_layernorm", "output",
    )
    for container in [model, inner_model]:
        if container is None:
            continue
        for attr_name in top_level_attr_names:
            top_mod = getattr(container, attr_name, None)
            if top_mod is not None:
                skip_module_ids.add(id(top_mod))
                for m in top_mod.modules():
                    skip_module_ids.add(id(m))

    for mod_name, module in model.named_modules():
        if id(module) in skip_module_ids:
            continue
        for attr in list(module._parameters.keys()):
            param = module._parameters[attr]
            if param is None:
                continue
            if param.dtype in fp8_dtypes:
                continue  # FP8 params stay as Parameters for FSDP sharding
            data = param.data
            module._parameters.pop(attr)
            module.register_buffer(attr, data)
            count += 1

    return count


def convert_scales_to_buffers(model):
    """
    weight_scale_inv, activation_scale 等の scale Parameters を buffer に変換。
    FSDP flat param に含めないため。buffers はモデルコードから
    self.weight_scale_inv で引き続きアクセス可能。
    """
    scale_suffixes = ("_scale_inv", "activation_scale")
    count = 0
    for mod_name, module in model.named_modules():
        for attr in list(module._parameters.keys()):
            if any(attr.endswith(s) or attr == s for s in scale_suffixes):
                param = module._parameters.pop(attr)
                if param is not None:
                    module.register_buffer(attr, param.data)
                    count += 1
    return count



# ============================================================
# Data utilities
# ============================================================

def extract_text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [extract_text_from_content(x) for x in content]
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        inner = content.get("content")
        if isinstance(inner, str):
            return inner
        if isinstance(inner, (list, dict)):
            return extract_text_from_content(inner)
        parts = [v for v in content.values() if isinstance(v, str)]
        return "\n".join(parts)
    return ""


def _global_shard_params(rank: int, world_size: int) -> Tuple[int, int]:
    wi = get_worker_info()
    if wi is None:
        worker_id, num_workers = 0, 1
    else:
        worker_id, num_workers = wi.id, wi.num_workers
    return rank * num_workers + worker_id, world_size * num_workers


def tokenize_prompt_ref(tok, prompt, ref, seq_len):
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        msgs_user = [{"role": "user", "content": prompt}]
        msgs_full = [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": ref}]

        ids_user_raw = tok.apply_chat_template(
            msgs_user, add_generation_prompt=True, tokenize=True)
        ids_full_raw = tok.apply_chat_template(
            msgs_full, add_generation_prompt=False, tokenize=True)

        # apply_chat_template can return: list[int], tensor, dict, or Encoding
        def _to_list(x):
            if isinstance(x, list):
                return x
            if isinstance(x, torch.Tensor):
                return x.view(-1).tolist()
            if isinstance(x, dict):
                return _to_list(x["input_ids"])
            if hasattr(x, "ids"):  # tokenizers.Encoding
                return x.ids
            # nested list
            if hasattr(x, "__getitem__") and len(x) > 0:
                return _to_list(x[0])
            return list(x)

        ids_user = _to_list(ids_user_raw)
        ids_full = _to_list(ids_full_raw)

        start = len(ids_user)
        ids_full_t = torch.tensor(ids_full, dtype=torch.long)
        labels = torch.full_like(ids_full_t, -100)
        if start < len(ids_full):
            labels[start:] = ids_full_t[start:]
        else:
            labels[:] = ids_full_t

        ids_list = ids_full
        labels_list = labels.tolist()
    else:
        full_text = prompt + "\n" + ref
        enc = tok(full_text, return_tensors="pt")
        ids = enc["input_ids"][0]
        enc_p = tok(prompt, return_tensors="pt", add_special_tokens=False)
        p_len = int(enc_p["input_ids"].shape[1])
        labels = torch.full_like(ids, -100)
        if p_len < ids.shape[0]:
            labels[p_len:] = ids[p_len:]
        else:
            labels[:] = ids
        ids_list = ids.tolist()
        labels_list = labels.tolist()

    if len(ids_list) > seq_len:
        ids_list = ids_list[:seq_len]
        labels_list = labels_list[:seq_len]
    return ids_list, labels_list


class OnTheFlyTokenizedChatMLJsonl(IterableDataset):
    def __init__(self, raw_path, tok, seq_len, rank, world_size):
        self.raw_path = raw_path
        self.tok = tok
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        global_rank, global_world = _global_shard_params(self.rank, self.world_size)
        with open(self.raw_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % global_world != global_rank:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                msgs = rec.get("messages") or []
                umsgs = [m for m in msgs if m.get("role") == "user"]
                amsgs = [m for m in msgs if m.get("role") == "assistant"]
                if not umsgs:
                    continue
                prompt = extract_text_from_content(umsgs[0].get("content")).strip()
                ref = extract_text_from_content(amsgs[0].get("content")).strip() if amsgs else ""
                if not prompt or not ref:
                    continue
                ids_list, labels_list = tokenize_prompt_ref(self.tok, prompt, ref, self.seq_len)
                if not ids_list:
                    continue
                yield {
                    "input_ids": torch.tensor(ids_list, dtype=torch.long),
                    "labels": torch.tensor(labels_list, dtype=torch.long),
                    "length": len(ids_list),
                }


class BucketedBatcher(IterableDataset):
    def __init__(self, base, batch_size, buffer_size, pad_token_id, seq_len,
                 max_batch_len=0, drop_last=False):
        self.base = base
        self.batch_size = batch_size
        self.buffer_size = max(buffer_size, batch_size)
        self.pad_token_id = pad_token_id
        self.seq_len = seq_len
        self.max_batch_len = int(max_batch_len) if max_batch_len else 0
        self.drop_last = drop_last

    def _pad_and_stack(self, samples):
        lengths = [int(s["length"]) for s in samples]
        max_len = min(max(lengths), self.seq_len)
        if self.max_batch_len > 0:
            max_len = min(max_len, self.max_batch_len)

        input_ids_list, labels_list, attn_list = [], [], []
        for s in samples:
            ids = s["input_ids"][:max_len]
            lbl = s["labels"][:max_len]
            cur_len = int(ids.shape[0])
            pad_len = max_len - cur_len
            if pad_len > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_len), value=self.pad_token_id)
                lbl = torch.nn.functional.pad(lbl, (0, pad_len), value=-100)
                attn = torch.cat([torch.ones(cur_len, dtype=torch.long),
                                  torch.zeros(pad_len, dtype=torch.long)])
            else:
                attn = torch.ones(max_len, dtype=torch.long)
            input_ids_list.append(ids)
            labels_list.append(lbl)
            attn_list.append(attn)

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_list),
            "labels": torch.stack(labels_list),
        }

    def __iter__(self):
        buf = []
        def flush(buffer):
            buffer.sort(key=lambda x: int(x["length"]))
            for i in range(0, len(buffer), self.batch_size):
                chunk = buffer[i : i + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                yield self._pad_and_stack(chunk)

        for ex in self.base:
            buf.append(ex)
            if len(buf) >= self.buffer_size:
                yield from flush(buf)
                buf = []
        if buf:
            yield from flush(buf)


class PackedSequenceDataset(IterableDataset):
    """
    複数の短いサンプルを seq_len に詰め込む（packing）。
    GPU utilization を最大化し、パディングの無駄を排除。

    仕組み:
      - base dataset から (ids, labels) を順に受け取る
      - 内部バッファに連結し、seq_len に達したら yield
      - サンプル境界では EOS トークンを挿入
      - labels: prompt 部分は既に -100、サンプル間境界も -100
      - Causal mask により後続サンプルが前のサンプルに attend するが、
        labels が -100 の部分は loss に影響しないため実用上問題なし

    注意: cross-attention contamination を完全に防ぐには
    FlashAttention2 の varlen 等が必要だが、SFT では通常不要。
    """

    def __init__(self, base, seq_len, pad_token_id, eos_token_id, batch_size=1):
        self.base = base
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.batch_size = batch_size

    def _pack_iter(self):
        """1 sequence ずつ packed サンプルを yield する。"""
        ids_buf = []
        lbl_buf = []

        for sample in self.base:
            s_ids = sample["input_ids"].tolist()
            s_lbl = sample["labels"].tolist()

            # seq_len を超えるサンプルはトランケート
            if len(s_ids) > self.seq_len:
                s_ids = s_ids[:self.seq_len]
                s_lbl = s_lbl[:self.seq_len]

            # バッファに追加したら seq_len を超える → まず現在のバッファを flush
            # サンプル間に EOS を挿入するため +1
            need = len(s_ids) + (1 if ids_buf else 0)
            if ids_buf and len(ids_buf) + need > self.seq_len:
                yield self._finalize(ids_buf, lbl_buf)
                ids_buf = []
                lbl_buf = []

            # サンプル間境界: EOS トークンを挿入（loss対象外）
            if ids_buf:
                ids_buf.append(self.eos_token_id)
                lbl_buf.append(-100)

            ids_buf.extend(s_ids)
            lbl_buf.extend(s_lbl)

            # ぴったり seq_len に達したら flush
            if len(ids_buf) >= self.seq_len:
                yield self._finalize(ids_buf[:self.seq_len], lbl_buf[:self.seq_len])
                ids_buf = []
                lbl_buf = []

        # 残りのバッファ（seq_len 未満）を flush
        if ids_buf:
            yield self._finalize(ids_buf, lbl_buf)

    def _finalize(self, ids_list, lbl_list):
        """ids/labels をパディングして dict 化。"""
        cur_len = len(ids_list)
        pad_len = self.seq_len - cur_len

        ids = torch.tensor(ids_list, dtype=torch.long)
        lbl = torch.tensor(lbl_list, dtype=torch.long)

        if pad_len > 0:
            ids = torch.nn.functional.pad(ids, (0, pad_len), value=self.pad_token_id)
            lbl = torch.nn.functional.pad(lbl, (0, pad_len), value=-100)
            attn = torch.cat([
                torch.ones(cur_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ])
        else:
            attn = torch.ones(self.seq_len, dtype=torch.long)

        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": lbl,
        }

    def __iter__(self):
        """batch_size 個の packed sequence をまとめて yield。"""
        batch = []
        for packed in self._pack_iter():
            batch.append(packed)
            if len(batch) >= self.batch_size:
                yield {
                    "input_ids": torch.stack([b["input_ids"] for b in batch]),
                    "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                    "labels": torch.stack([b["labels"] for b in batch]),
                }
                batch = []
        # Drop partial batch to prevent FSDP hang from rank-imbalanced iterations
        # (when some ranks have more micro-batches than others)


# ============================================================
# Trainable parameter selection
# ============================================================

# Module name patterns for each training mode.
# Keys are substrings matched against param names within a layer.
TRAIN_MODULE_PATTERNS = {
    "attn": {
        "include": ("self_attn.", "input_layernorm."),
        "description": "self_attn + input_layernorm",
    },
    "mlp": {
        "include": ("mlp.", "moe.", "feed_forward.", "ffn.",
                    "post_attention_layernorm.", "post_attn_layernorm."),
        "description": "mlp/moe + post_attention_layernorm",
    },
    "full": {
        "include": None,  # None = all params in the layer
        "description": "all modules in layer",
    },
}


def inject_lora_adapters(model, train_layer_indices, train_modules,
                         lora_rank, lora_alpha, lora_dropout, rank=0):
    """
    Inject LoRA A/B parameters into target linear layers.
    Base weights stay frozen (FP8 or otherwise). Only LoRA params are trainable.

    For each target linear:
      output = original_forward(x) + dropout(x @ A.T) @ B.T * (alpha / r)

    LoRA B is zero-initialized → initial model output is unchanged.

    Args:
        model: base model (pre-FSDP)
        train_layer_indices: list of layer indices to apply LoRA
        train_modules: "attn", "mlp", or "full"
        lora_rank: LoRA rank (r)
        lora_alpha: LoRA alpha (scaling = alpha / r)
        lora_dropout: dropout rate on LoRA path
        rank: distributed rank (for logging)
    Returns:
        number of injected adapters
    """
    pattern_info = TRAIN_MODULE_PATTERNS.get(train_modules)
    if pattern_info is None:
        raise ValueError(f"Unknown train_modules: {train_modules}")

    include_patterns = pattern_info["include"]
    train_set = set(train_layer_indices)

    if include_patterns is not None:
        target_prefixes = []
        for i in train_set:
            for pat in include_patterns:
                target_prefixes.append(f"layers.{i}.{pat}")
    else:
        target_prefixes = [f"layers.{i}." for i in train_set]

    # Skip non-linear modules
    skip_lower = ("layernorm", "layer_norm", "rmsnorm", "norm")

    scaling = lora_alpha / lora_rank
    dropout_mod = nn.Dropout(lora_dropout) if lora_dropout > 0 else None

    injected = 0
    total_lora_params = 0

    for name, module in model.named_modules():
        if not any(pref in name for pref in target_prefixes):
            continue
        # Skip normalization layers
        name_lower = name.lower().split(".")[-1]
        if any(s in name_lower for s in skip_lower):
            continue

        # Must have a weight tensor with 2+ dims (linear-like)
        w = getattr(module, "weight", None)
        if w is None or not isinstance(w, (nn.Parameter, torch.Tensor)):
            continue
        if w.dim() < 2:
            continue
        # Skip 3D+ (MoE expert composite tensors) for now
        if w.dim() > 2:
            continue

        out_features, in_features = w.shape[0], w.shape[1]

        # Create LoRA params: A (rank, in), B (out, rank)
        lora_A = nn.Parameter(
            torch.empty(lora_rank, in_features, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        lora_B = nn.Parameter(
            torch.zeros(out_features, lora_rank, dtype=torch.bfloat16))

        module.register_parameter("lora_A", lora_A)
        module.register_parameter("lora_B", lora_B)

        # Forward hook: add LoRA contribution
        def _make_lora_hook(s, drop):
            def _hook(mod, inp, out):
                x = inp[0] if isinstance(inp, tuple) else inp
                h = x.to(mod.lora_A.dtype) @ mod.lora_A.T   # (..., rank)
                if drop is not None:
                    h = drop(h)
                h = h @ mod.lora_B.T                          # (..., out)
                h = h * s
                if isinstance(out, tuple):
                    return (out[0] + h,) + out[1:]
                return out + h
            return _hook

        module.register_forward_hook(_make_lora_hook(scaling, dropout_mod))
        injected += 1
        total_lora_params += lora_rank * (in_features + out_features)

        if rank == 0 and injected <= 5:
            print(f"  [LoRA] {name}: ({out_features}, {in_features}) → r={lora_rank}",
                  flush=True)

    if rank == 0:
        if injected > 5:
            print(f"  [LoRA] ... and {injected - 5} more modules", flush=True)
        print(f"[LoRA] Injected {injected} adapters | r={lora_rank} α={lora_alpha} "
              f"dropout={lora_dropout} | {total_lora_params:,} LoRA params", flush=True)

    return injected


def mark_trainable(model, train_layer_indices, train_modules="attn",
                   use_lora=False):
    """
    指定された layer indices の指定モジュールを trainable に。それ以外は freeze。

    Args:
        model: the model
        train_layer_indices: list of int, e.g. [77] or [76, 77]
        train_modules: str, one of "attn", "mlp", "full"
        use_lora: if True, only lora_A/lora_B are trainable
    """
    if train_modules not in TRAIN_MODULE_PATTERNS:
        raise ValueError(
            f"Unknown --train_modules '{train_modules}'. "
            f"Choose from: {list(TRAIN_MODULE_PATTERNS.keys())}"
        )

    description = TRAIN_MODULE_PATTERNS[train_modules]["description"]
    train_set = set(train_layer_indices)

    total = 0
    trainable = 0

    if use_lora:
        # LoRA mode: only lora_A/lora_B are trainable, everything else frozen
        for name, p in model.named_parameters():
            total += p.numel()
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad = True
                trainable += p.numel()
            else:
                p.requires_grad = False
        mode_str = f"lora({train_modules})"
    else:
        # Full fine-tune mode
        include_patterns = TRAIN_MODULE_PATTERNS[train_modules]["include"]
        if include_patterns is not None:
            all_prefixes = []
            for i in train_set:
                for pat in include_patterns:
                    all_prefixes.append(f"layers.{i}.{pat}")
        else:
            all_prefixes = [f"layers.{i}." for i in train_set]

        skip_suffixes = ("weight_scale_inv", "activation_scale")

        for name, p in model.named_parameters():
            total += p.numel()
            if any(pref in name for pref in all_prefixes) and not name.endswith(skip_suffixes):
                if p.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                    trainable += p.numel()
            else:
                p.requires_grad = False
        mode_str = train_modules

    layers_str = ",".join(str(i) for i in sorted(train_set))
    print(
        f"[INFO] trainable({mode_str})={trainable:,} / total={total:,} "
        f"({trainable / max(1, total) * 100:.4f}%) "
        f"[layers {layers_str}, modules={description}]",
        flush=True,
    )
    return trainable


# ============================================================
# Trainer
# ============================================================

class LastLayerAttnSFTTrainer:
    def __init__(self, args):
        self.args = args

        if not _FSDP_AVAILABLE:
            raise RuntimeError("torch.distributed.fsdp not available.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required.")

        configure_sdp_backends()

        # --------------------------------------------------
        # 1) meta device でレイヤークラスを検出
        # --------------------------------------------------
        if int(os.environ.get("RANK", "0")) == 0:
            print("[INIT] Detecting layer class on meta device...", flush=True)

        temp_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        with torch.device("meta"):
            temp_model = AutoModelForCausalLM.from_config(temp_config, trust_remote_code=True)

        if hasattr(temp_model, "model") and hasattr(temp_model.model, "layers"):
            layer_cls = temp_model.model.layers[0].__class__
        elif hasattr(temp_model, "layers"):
            layer_cls = temp_model.layers[0].__class__
        else:
            layer_cls = None
        del temp_model

        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[INIT] Layer Class: {layer_cls}", flush=True)

        auto_wrap_policy = None
        if layer_cls is not None:
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={layer_cls},
            )

        # --------------------------------------------------
        # 2) FSDP plugin
        #    sync_module_states=True:
        #      → rank 0 の実weight を全rank に broadcast
        #    param_init_fn:
        #      → meta テンソルがある場合のみ to_empty で materialize
        #      → rank 0 の実データは保護される
        # --------------------------------------------------
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            cpu_ram_efficient_loading=True,
            sync_module_states=True,
            limit_all_gathers=True,
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing=False,  # Disabled: we use detach hooks instead
        )
        # FSDP MixedPrecision を手動設定:
        #   param_dtype=None: パラメータの dtype をそのまま維持
        #     → FP8 frozen params: FP8 のまま保持 → hook が scale 付き dequant
        #     → BF16 trainable params (layer 77): BF16 のまま保持
        #   reduce_dtype=bf16: gradient all-reduce を BF16 で実行
        #   buffer_dtype=None: buffers keep original dtype (scale_inv は f32 維持)
        # 注意: mixed_precision="bf16" は FSDP が FP8→BF16 を scale なしで行い NaN になる
        fsdp_plugin.mixed_precision_policy = MixedPrecision(
            param_dtype=None,       # FP8 stays FP8, BF16 stays BF16
            reduce_dtype=torch.bfloat16,  # gradient all-reduce in BF16
            buffer_dtype=None,      # buffers keep original dtype
        )

        # ★ 修正: param_init_fn は meta テンソルがある場合のみ to_empty を実行。
        # 旧コード: 無条件に module.to_empty() → rank 0 の実データも破壊 → NaN
        # 新コード: meta デバイス上のテンソルがある場合のみ materialize。
        #   rank 0: 全テンソルが CPU 上の実データ → to_empty スキップ → データ保護
        #   rank != 0: meta テンソル → to_empty(cpu) → 空テンソル作成 → sync で埋まる
        def _param_init_fn(module):
            has_meta = any(
                p.device.type == "meta"
                for p in module.parameters(recurse=False)
            ) or any(
                b is not None and b.device.type == "meta"
                for b in module.buffers(recurse=False)
            )
            if has_meta:
                module.to_empty(device=torch.device("cpu"), recurse=False)
            return module

        fsdp_plugin.param_init_fn = _param_init_fn

        # NCCL timeout 延長: rank 0 のモデルロードに時間がかかるため
        pg_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))
        # sync_each_batch=True: disable no_sync to avoid +16GB full gradient
        # buffer that causes OOM during gradient accumulation with FSDP.
        ga_plugin = GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps,
            sync_each_batch=True,
        )
        self.accelerator = Accelerator(
            mixed_precision="no",  # FSDP MixedPrecision で手動管理
            fsdp_plugin=fsdp_plugin,
            gradient_accumulation_plugin=ga_plugin,
            kwargs_handlers=[pg_kwargs],
        )

        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        self.device = self.accelerator.device

        if self.accelerator.is_main_process:
            print(f"[DIST] rank={self.rank} / world_size={self.world_size}", flush=True)

        # --------------------------------------------------
        # 3) モデルロード
        #    rank 0: 実ウェイトを CPU にロード (~760GB FP8)
        #    rank != 0: meta device（CPU RAM = 0）
        #      - _meta_model_loading_context: safetensors 読み込みスキップ
        #      - torch.device("meta"): FP8Linear 等の tensor 生成を meta に
        #    FSDP sync_module_states=True で rank 0 → 全rank に broadcast
        # --------------------------------------------------
        self.tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

        if self.accelerator.is_main_process:
            print(f"[LOAD] model: {args.model}", flush=True)

        if self.rank == 0:
            print("[LOAD] rank 0: loading real weights to CPU...", flush=True)
            import transformers.modeling_utils as _tm_utils
            # _finalize_model_loading が "missing" keys を normal_() で初期化しようとするが、
            # FP8 weight は normal_() 未対応で crash する。
            # checkpoint は self_attn.indexer.* 形式だが model は self_attn.wk.* 形式 →
            # key 名不一致で MISSING 判定されるが実データはロード済み。
            _orig_finalize_r0 = _tm_utils.PreTrainedModel._finalize_model_loading
            @classmethod
            def _skip_finalize(cls, model, *args, **kwargs):
                return {"missing_keys": [], "unexpected_keys": [],
                        "mismatched_keys": [], "error_msgs": []}
            _tm_utils.PreTrainedModel._finalize_model_loading = _skip_finalize
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    config=config,
                    device_map=None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            finally:
                _tm_utils.PreTrainedModel._finalize_model_loading = _orig_finalize_r0
        else:
            # Non-rank-0: モデル構造だけ作成、weight ロードをスキップ
            print(f"[LOAD] rank {self.rank}: creating meta model (zero CPU RAM)...", flush=True)
            import transformers.modeling_utils as _tm_utils

            _orig_check = getattr(_tm_utils, "check_and_set_device_map", None)
            _orig_load_model = _tm_utils.PreTrainedModel._load_pretrained_model
            _orig_finalize = _tm_utils.PreTrainedModel._finalize_model_loading

            @classmethod
            def _noop_load(cls, model, *args, **kwargs):
                return {"missing_keys": [], "unexpected_keys": [],
                        "mismatched_keys": [], "error_msgs": []}, None

            @classmethod
            def _noop_finalize(cls, model, *args, **kwargs):
                return {"missing_keys": [], "unexpected_keys": [],
                        "mismatched_keys": [], "error_msgs": []}

            _tm_utils.PreTrainedModel._load_pretrained_model = _noop_load
            _tm_utils.PreTrainedModel._finalize_model_loading = _noop_finalize
            if _orig_check is not None:
                _tm_utils.check_and_set_device_map = lambda device_map, **kw: device_map

            try:
                with torch.device("meta"):
                    base_model = AutoModelForCausalLM.from_pretrained(
                        args.model,
                        config=config,
                        device_map=None,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
            finally:
                _tm_utils.PreTrainedModel._load_pretrained_model = _orig_load_model
                _tm_utils.PreTrainedModel._finalize_model_loading = _orig_finalize
                if _orig_check is not None:
                    _tm_utils.check_and_set_device_map = _orig_check
            print(f"[LOAD] rank {self.rank}: meta model created successfully.", flush=True)

        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False

        # ★ Do NOT enable gradient_checkpointing or enable_input_require_grads here.
        # With FSDP, gradient checkpointing causes each layer to be re-all-gathered
        # during backward (~18.45 GiB per layer), causing OOM.
        # Instead, we use a detach hook (installed post-FSDP) so that:
        #   - Layers 0-(N-1): forward runs without grad → no activations saved, no backward
        #   - Layer N+: forward runs with grad → activations saved, backward computes gradients
        # This is more memory-efficient than checkpointing for partial-layer training.

        base_model.train()

        # ---- Handle FSDP1 uniform dtype + correct FP8 forward ----
        last_layer_idx = args.num_layers - 1
        train_layer_indices = list(range(last_layer_idx - args.train_layers + 1, last_layer_idx + 1))
        first_train_idx = train_layer_indices[0]
        self.train_layer_indices = train_layer_indices

        # Step 1: scales → buffers
        n_scales = convert_scales_to_buffers(base_model)
        if self.accelerator.is_main_process:
            print(f"[DTYPE] Converted {n_scales} scale Parameters to buffers.", flush=True)

        # ★ Scale 値の健全性チェック (rank 0 のみ)
        if self.rank == 0:
            for name, buf in base_model.named_buffers():
                if "weight_scale_inv" in name and "layers.0." in name:
                    flat = buf.data.float().view(-1)[:100]
                    print(f"[SCALE-CHECK] {name}: "
                          f"min={flat.min().item():.6f}, "
                          f"max={flat.max().item():.6f}, "
                          f"mean={flat.mean().item():.6f}, "
                          f"any_neg={bool((flat < 0).any())}, "
                          f"any_nan={bool(torch.isnan(flat).any())}",
                          flush=True)
                    break

        # Step 2: dequant / LoRA injection
        layers_str = ",".join(str(i) for i in train_layer_indices)
        if args.use_lora:
            # LoRA mode: base weights stay FP8. No dequant needed.
            # LoRA A/B are added as BF16 Parameters → much less memory.
            if self.accelerator.is_main_process:
                print(f"[LoRA] Injecting adapters into layers [{layers_str}]...", flush=True)
            n_lora = inject_lora_adapters(
                base_model, train_layer_indices, args.train_modules,
                lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout, rank=self.rank,
            )
            if n_lora == 0:
                raise RuntimeError(
                    f"No LoRA adapters injected for layers {train_layer_indices} "
                    f"with --train_modules={args.train_modules}."
                )
        else:
            # Full fine-tune: dequant full trainable layers FP8 → BF16
            # ★ Layer 全体を BF16 化して FSDP uniform dtype を維持。
            # モジュール選択は mark_trainable() の requires_grad で行う。
            if self.accelerator.is_main_process:
                print(f"[DEQUANT] Converting layers [{layers_str}] FP8 → BF16 (full layer)...", flush=True)
            total_dequant = 0
            for lidx in train_layer_indices:
                n_dequant = dequantize_single_layer(base_model, lidx, rank=self.rank)
                total_dequant += n_dequant
                if self.accelerator.is_main_process:
                    print(f"[DEQUANT] Layer {lidx}: {n_dequant} tensors converted.", flush=True)
            if self.accelerator.is_main_process:
                print(f"[DEQUANT] Total: {total_dequant} tensors in "
                      f"{len(train_layer_indices)} layers.", flush=True)

        # Step 3: frozen non-FP8 params → buffers (trainable layers excluded entirely)
        n_buf = convert_frozen_non_fp8_to_buffers(base_model, train_layer_indices)
        if self.accelerator.is_main_process:
            print(f"[DTYPE] Converted {n_buf} frozen non-FP8 Parameters to buffers.", flush=True)

        # ---- Freeze all except trainable layers' selected modules ----
        trainable_count = mark_trainable(
            base_model, train_layer_indices, args.train_modules,
            use_lora=args.use_lora,
        )
        if trainable_count == 0:
            raise RuntimeError(
                f"No trainable params for layers {train_layer_indices} "
                f"with --train_modules={args.train_modules} "
                f"{'(LoRA)' if args.use_lora else ''}. "
                f"Check --num_layers ({args.num_layers}) and --train_layers ({args.train_layers})."
            )

        # NOTE: Detach hooks are installed AFTER FSDP wrapping (see below)
        # Pre-FSDP hooks would land inside CheckpointWrapper and be ineffective.

        # --------------------------------------------------
        # 4) FSDP wrap (sync_module_states で rank0 → 全rank に配布)
        # --------------------------------------------------
        # Non-rank-0: meta buffers を CPU に materialize
        if self.rank != 0:
            for module in base_model.modules():
                for bname in list(module._buffers.keys()):
                    buf = module._buffers[bname]
                    if buf is not None and buf.device.type == "meta":
                        module._buffers[bname] = torch.empty(
                            buf.shape, dtype=buf.dtype, device="cpu"
                        )

        # Rank 0 の処理完了まで全 rank で同期
        self.accelerator.wait_for_everyone()

        # Lightweight pre-FSDP diagnostic: dtype counts only (no tensor scanning)
        if self.rank == 0:
            dtype_counts = {}
            for name, p in base_model.named_parameters():
                dt = str(p.dtype)
                dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
            print(f"[DIAG-PRE] Parameter dtypes: {dtype_counts}", flush=True)
            buf_dtypes = {}
            for name, b in base_model.named_buffers():
                if b is not None:
                    dt = str(b.dtype)
                    buf_dtypes[dt] = buf_dtypes.get(dt, 0) + 1
            print(f"[DIAG-PRE] Buffer dtypes: {buf_dtypes}", flush=True)

        if self.accelerator.is_main_process:
            print("[FSDP] Wrapping model (sync_module_states=True)...", flush=True)

        self.model = self.accelerator.prepare(base_model)
        self.is_fsdp = isinstance(self.model, FSDP)

        if self.accelerator.is_main_process:
            print("[FSDP] Model wrapped successfully.", flush=True)
            # Lightweight post-FSDP dtype check
            dtype_counts = {}
            for name, p in self.model.named_parameters():
                dt = str(p.dtype)
                dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
            print(f"[DIAG] Post-FSDP parameter dtypes: {dtype_counts}", flush=True)

        # ---- Install FP8 on-the-fly dequant hooks ----
        fp8_hooks, n_fp8_patched = install_fp8_dequant_hooks(self.model)
        if self.accelerator.is_main_process:
            print(f"[FP8] Installed dequant hooks on {n_fp8_patched} FP8 submodules ({len(fp8_hooks)} hooks).", flush=True)

        # ---- Post-FSDP detach hooks for memory optimization ----
        # These MUST be installed after FSDP wrapping so they sit OUTSIDE
        # the FSDP/CheckpointWrapper boundary.
        #
        # Strategy: Since we disabled gradient checkpointing AND
        # don't call enable_input_require_grads(), frozen layers (0..N-1)
        # forward without requires_grad on their inputs → no autograd graph
        # → no activation storage → no backward. We only need to ensure
        # the first trainable layer's input gets requires_grad=True.

        # --- Robust FSDP layer discovery ---
        # Walk the wrapped model's hierarchy to find the `layers` ModuleList,
        # then index directly. This avoids fragile name-pattern matching.
        def _find_layers_list(model):
            """Find the nn.ModuleList containing decoder layers."""
            for mod in model.modules():
                if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
                    # Heuristic: the largest ModuleList is the decoder layer stack
                    # Verify by checking expected count
                    if len(mod) == args.num_layers:
                        return mod
            # Fallback: try known attribute paths
            for attr_path in ["model.layers", "layers",
                              "_fsdp_wrapped_module.model.layers",
                              "_fsdp_wrapped_module.layers"]:
                obj = model
                try:
                    for part in attr_path.split("."):
                        obj = getattr(obj, part)
                    if isinstance(obj, torch.nn.ModuleList):
                        return obj
                except AttributeError:
                    continue
            return None

        fsdp_layers = _find_layers_list(self.model)

        fsdp_train_layer = None
        if fsdp_layers is not None and first_train_idx < len(fsdp_layers):
            fsdp_train_layer = fsdp_layers[first_train_idx]
            if self.accelerator.is_main_process:
                print(f"[DETACH] Found layers ModuleList ({len(fsdp_layers)} layers), "
                      f"target: index {first_train_idx} → {type(fsdp_train_layer).__name__}",
                      flush=True)
        else:
            if self.accelerator.is_main_process:
                print(f"[WARN] Could not find layers ModuleList. "
                      f"fsdp_layers={'None' if fsdp_layers is None else len(fsdp_layers)}, "
                      f"first_train_idx={first_train_idx}", flush=True)

        # Hook: Detach + requires_grad on first trainable layer
        if fsdp_train_layer is not None:
            def _detach_and_enable_grad(module, args):
                """At frozen→trainable boundary: detach to cut graph, enable grad."""
                new_args = []
                for a in args:
                    if isinstance(a, torch.Tensor):
                        new_args.append(a.detach().requires_grad_(True))
                    else:
                        new_args.append(a)
                return tuple(new_args)

            fsdp_train_layer.register_forward_pre_hook(_detach_and_enable_grad)
            if self.accelerator.is_main_process:
                print(f"[DETACH] Installed detach+grad hook on layer {first_train_idx} "
                      f"(frozen layers 0-{first_train_idx-1} run without grad → no activation storage)",
                      flush=True)
        else:
            # Last-resort fallback: scan by name
            if self.accelerator.is_main_process:
                print(f"[WARN] Falling back to name-based module search.", flush=True)
            for name, module in self.model.named_modules():
                if f"layers.{first_train_idx}" in name and isinstance(module, FSDP):
                    # Verify it's the layer-level FSDP (not a sub-module FSDP)
                    name_parts = name.split(".")
                    try:
                        idx_pos = name_parts.index(str(first_train_idx))
                        if idx_pos > 0 and name_parts[idx_pos - 1] == "layers":
                            # Check no deeper path after layers.{idx}
                            if idx_pos == len(name_parts) - 1:
                                def _detach_and_enable_grad_fb(module, args):
                                    new_args = []
                                    for a in args:
                                        if isinstance(a, torch.Tensor):
                                            new_args.append(a.detach().requires_grad_(True))
                                        else:
                                            new_args.append(a)
                                    return tuple(new_args)
                                module.register_forward_pre_hook(_detach_and_enable_grad_fb)
                                fsdp_train_layer = module
                                if self.accelerator.is_main_process:
                                    print(f"[DETACH] Fallback: hook on {name}", flush=True)
                                break
                    except ValueError:
                        continue

        if self.accelerator.is_main_process:
            print(f"[DETACH] Summary: layer-input detach={'OK' if fsdp_train_layer else 'FAILED'}",
                  flush=True)

        # ---- MLP output detach hooks ----
        # Prevent backward through MLP/MoE in trainable layers.
        # Without this, FSDP all-gathers the massive MoE params (~18 GiB) during backward → OOM.
        #
        # Gradient approximation:
        #   Full:  dL/dh_mid = dL/dh_out × (I + J_mlp)
        #   Approx: dL/dh_mid = dL/dh_out  (residual identity only)
        # The identity path dominates in deep residual networks.
        # MLP params are frozen anyway — only attn params receive gradients.
        # Only install when training attn-only: with train_modules mlp/full the
        # MLP params ARE trainable, and detaching their output would cut their
        # gradients entirely (training becomes a no-op).
        mlp_detach_count = 0
        if args.train_modules == "attn":
            for tidx in train_layer_indices:
                layer_mod = fsdp_layers[tidx] if fsdp_layers is not None else None
                if layer_mod is None:
                    continue

                # Find the inner (unwrapped) layer module
                inner = getattr(layer_mod, '_fsdp_wrapped_module', layer_mod)
                # Try to find checkpoint wrapper
                inner = getattr(inner, '_checkpoint_wrapped_module', inner)

                mlp_module = None
                for attr_name in ("mlp", "moe", "feed_forward", "ffn"):
                    mlp_module = getattr(inner, attr_name, None)
                    if mlp_module is not None:
                        break

                if mlp_module is not None:
                    def _make_mlp_detach_hook(layer_idx):
                        def _mlp_output_detach(module, input, output):
                            """Detach MLP/MoE output to prevent backward through frozen MoE."""
                            if isinstance(output, torch.Tensor) and output.requires_grad:
                                return output.detach().requires_grad_(True)
                            elif isinstance(output, tuple):
                                new_out = []
                                for o in output:
                                    if isinstance(o, torch.Tensor) and o.requires_grad:
                                        new_out.append(o.detach().requires_grad_(True))
                                    else:
                                        new_out.append(o)
                                return tuple(new_out)
                            return output
                        return _mlp_output_detach

                    mlp_module.register_forward_hook(_make_mlp_detach_hook(tidx))
                    mlp_detach_count += 1
                    if self.accelerator.is_main_process:
                        print(f"[DETACH] MLP output detach on layer {tidx} ({type(mlp_module).__name__})",
                              flush=True)
                else:
                    if self.accelerator.is_main_process:
                        print(f"[WARN] No MLP/MoE found in layer {tidx}.", flush=True)

        if self.accelerator.is_main_process:
            print(f"[DETACH] MLP detach: {mlp_detach_count} layers "
                  f"(approx gradients via residual identity path)", flush=True)

        # Optimizer (trainable params only)
        # ★ FSDP use_orig_params=True exposes both original params AND internal
        # FlatParameter objects via model.parameters(). FlatParameter is a 1D
        # concatenation of all params in the FSDP unit, while original params
        # are 2D views. Mixing them causes shape mismatch in AdamW's _foreach ops.
        # Fix: use named_parameters() which only returns original params.
        # Also use foreach=False to disable multi-tensor batching for safety.
        trainable_params = [p for _, p in self.model.named_parameters()
                           if p.requires_grad and p.numel() > 0]

        if self.accelerator.is_main_process:
            print(f"[OPT] Trainable params for optimizer: {len(trainable_params)}", flush=True)
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.numel() > 0:
                    print(f"  {n}: shape={tuple(p.shape)}, numel={p.numel()}", flush=True)

        # Some ranks may have ALL params as 0-numel → empty optimizer is fine
        has_params = len(trainable_params) > 0
        if self.accelerator.is_main_process and not has_params:
            raise RuntimeError("No trainable parameters on rank 0 after FSDP wrapping.")

        if not trainable_params:
            _dummy = torch.nn.Parameter(torch.zeros(1, device=self.device, dtype=torch.bfloat16))
            _dummy.requires_grad = True
            trainable_params = [_dummy]
            self._dummy_param = _dummy

        self.opt = torch.optim.AdamW(
            trainable_params, lr=args.lr, betas=(0.9, 0.98),
            weight_decay=args.weight_decay,
            foreach=False,  # Disable multi-tensor to avoid FSDP shape issues
        )
        # Prepare optimizer so accelerator.accumulate() properly handles
        # grad_accum > 1 (no-ops opt.step on non-boundary micro-steps).
        self.opt = self.accelerator.prepare(self.opt)

        # Pad token
        if self.tok.pad_token_id is not None:
            self.pad_token_id = int(self.tok.pad_token_id)
        elif self.tok.eos_token_id is not None:
            self.pad_token_id = int(self.tok.eos_token_id)
        else:
            raise RuntimeError("Tokenizer has no pad/eos token id.")

        # Dataset
        if self.accelerator.is_main_process:
            print(f"[DATA] dataset: {args.dataset}", flush=True)
            print(f"[DATA] packing: {args.packing}", flush=True)

        base_ds = OnTheFlyTokenizedChatMLJsonl(
            args.dataset, tok=self.tok, seq_len=args.seq_len,
            rank=self.rank, world_size=self.world_size,
        )

        if args.packing:
            eos_id = int(self.tok.eos_token_id) if self.tok.eos_token_id is not None else self.pad_token_id
            batched_ds = PackedSequenceDataset(
                base=base_ds,
                seq_len=args.seq_len,
                pad_token_id=self.pad_token_id,
                eos_token_id=eos_id,
                batch_size=args.batch_size,
            )
        else:
            batched_ds = BucketedBatcher(
                base=base_ds, batch_size=args.batch_size,
                buffer_size=args.bucket_buffer,
                pad_token_id=self.pad_token_id, seq_len=args.seq_len,
                max_batch_len=args.max_batch_len, drop_last=True,
            )
        self.loader = DataLoader(batched_ds, batch_size=None, num_workers=0)

        self.global_step = 0
        self.global_epoch = 0

        # ---- Estimate total steps for progress tracking ----
        # "step" = optimizer step (= grad_accum micro-batches)
        self.total_lines = 0
        self.estimated_total_steps = 0
        gas = args.gradient_accumulation_steps
        if self.accelerator.is_main_process:
            try:
                total_tokens = 0
                with open(args.dataset, "r", encoding="utf-8") as f:
                    for line in f:
                        self.total_lines += 1
                        if args.packing:
                            # Rough token estimate for packing efficiency calc
                            line = line.strip()
                            if line:
                                try:
                                    rec = json.loads(line)
                                    msgs = rec.get("messages") or []
                                    text = " ".join(
                                        extract_text_from_content(m.get("content", ""))
                                        for m in msgs
                                    )
                                    # Rough: 1 token ≈ 4 chars (for estimation only)
                                    total_tokens += max(len(text) // 4, 10)
                                except Exception:
                                    total_tokens += 100  # fallback

                samples_per_rank = self.total_lines // self.world_size

                if args.packing:
                    # Packing: samples are packed into seq_len sequences
                    tokens_per_rank = total_tokens // self.world_size
                    packed_seqs_per_rank = max(1, tokens_per_rank // args.seq_len)
                    micro_batches_per_epoch = (packed_seqs_per_rank + args.batch_size - 1) // args.batch_size
                    packing_ratio = samples_per_rank / max(1, packed_seqs_per_rank)
                    print(f"[DATA] packing estimate: ~{tokens_per_rank:,} tokens/rank, "
                          f"~{packed_seqs_per_rank:,} packed_seqs/rank, "
                          f"packing_ratio≈{packing_ratio:.1f}x",
                          flush=True)
                else:
                    micro_batches_per_epoch = (samples_per_rank + args.batch_size - 1) // args.batch_size

                optim_steps_per_epoch = micro_batches_per_epoch // gas
                num_epochs = args.epochs if args.epochs > 0 else 1
                self.estimated_total_steps = optim_steps_per_epoch * num_epochs
                if args.steps and args.steps > 0:
                    self.estimated_total_steps = min(self.estimated_total_steps, args.steps)

                eff_batch = args.batch_size * gas * self.world_size
                print(f"[DATA] total_lines={self.total_lines:,}, "
                      f"micro_batches/epoch≈{micro_batches_per_epoch:,}, "
                      f"optim_steps/epoch≈{optim_steps_per_epoch:,}, "
                      f"estimated_total_steps≈{self.estimated_total_steps:,}, "
                      f"effective_batch_size={eff_batch}",
                      flush=True)
            except Exception as e:
                print(f"[DATA] Could not count lines: {e}", flush=True)

        # Timing state
        self._train_start_time = None
        self._step_times = []  # rolling window for speed estimation
        self._loss_accum = 0.0
        self._loss_count = 0
        self._micro_step = 0       # micro-batch counter
        self._micro_loss_accum = 0.0
        self._micro_loss_count = 0

        if self.accelerator.is_main_process:
            eff_batch = args.batch_size * args.gradient_accumulation_steps * self.world_size
            print("=" * 56)
            print(f"  GLM-5 Partial-Layer SFT {'(LoRA)' if args.use_lora else '(Full)'}")
            print(f"  model        : {args.model}")
            print(f"  num_layers   : {args.num_layers}")
            print(f"  train_layers : {layers_str} ({len(train_layer_indices)} layers)")
            print(f"  train_modules: {args.train_modules} ({TRAIN_MODULE_PATTERNS[args.train_modules]['description']})")
            if args.use_lora:
                print(f"  lora         : r={args.lora_rank}, α={args.lora_alpha}, dropout={args.lora_dropout}")
            print(f"  dataset      : {args.dataset}")
            print(f"  packing      : {args.packing}")
            print(f"  batch_size   : {args.batch_size} (micro)")
            print(f"  grad_accum   : {args.gradient_accumulation_steps}")
            print(f"  eff_batch    : {eff_batch} (micro×accum×GPUs={args.batch_size}×{args.gradient_accumulation_steps}×{self.world_size})")
            print(f"  seq_len      : {args.seq_len}")
            print(f"  optim        : {args.optim}")
            print(f"  lr           : {args.lr}")
            print(f"  epochs       : {args.epochs}")
            print(f"  steps(cap)   : {args.steps}")
            print(f"  save_dir     : {args.save_dir}")
            print("=" * 56, flush=True)

        # ---- W&B logging (rank 0 only) ----
        self.use_wandb = False
        if args.wandb_project and self.accelerator.is_main_process:
            try:
                import wandb
                eff_batch = args.batch_size * args.gradient_accumulation_steps * self.world_size
                wandb_config = {
                    "model": args.model,
                    "num_layers": args.num_layers,
                    "train_layer": train_layer_indices,
                    "train_layers": args.train_layers,
                    "train_modules": args.train_modules,
                    "use_lora": args.use_lora,
                    "lora_rank": args.lora_rank if args.use_lora else None,
                    "lora_alpha": args.lora_alpha if args.use_lora else None,
                    "lora_dropout": args.lora_dropout if args.use_lora else None,
                    "trainable_params": trainable_count,
                    "dataset": args.dataset,
                    "packing": args.packing,
                    "batch_size_micro": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "effective_batch_size": eff_batch,
                    "seq_len": args.seq_len,
                    "optim": args.optim,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "grad_clip": args.grad_clip,
                    "epochs": args.epochs,
                    "steps_cap": args.steps,
                    "world_size": self.world_size,
                    "estimated_total_steps": self.estimated_total_steps,
                    "total_lines": self.total_lines,
                }
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name or None,
                    entity=args.wandb_entity or None,
                    config=wandb_config,
                    resume="allow",
                )
                self.use_wandb = True
                print(f"[WANDB] Initialized: project={args.wandb_project}, "
                      f"run={wandb.run.name}", flush=True)
            except ImportError:
                print("[WANDB] wandb not installed. Skipping.", flush=True)
            except Exception as e:
                print(f"[WANDB] Init failed: {e}. Continuing without wandb.", flush=True)

    def step(self, batch):
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        # First-step NaN debugging: register hooks to find where NaN appears
        hooks = []
        if self.global_step == 0 and self._micro_step == 0 and self.accelerator.is_main_process:
            def make_hook(name):
                def hook(module, input, output):
                    if "layers.0" in name and name.count(".") <= 3:
                        if isinstance(input, tuple) and len(input) > 0:
                            inp = input[0] if isinstance(input[0], torch.Tensor) else None
                            if inp is not None and torch.isnan(inp).any():
                                print(f"[NaN] {name} INPUT has NaN (shape={tuple(inp.shape)}, dtype={inp.dtype})", flush=True)
                            elif inp is not None:
                                flat = inp.view(-1)[:500].float()
                                print(f"[DBG] {name} input OK: dtype={inp.dtype}, min={flat.min():.4f}, max={flat.max():.4f}", flush=True)

                    if isinstance(output, tuple):
                        tensors = [o for o in output if isinstance(o, torch.Tensor)]
                    elif isinstance(output, torch.Tensor):
                        tensors = [output]
                    else:
                        return
                    for i, t in enumerate(tensors):
                        if torch.isnan(t).any():
                            nan_frac = torch.isnan(t).float().mean().item()
                            print(f"[NaN] {name} output[{i}] has NaN ({nan_frac*100:.1f}%) (shape={tuple(t.shape)}, dtype={t.dtype})", flush=True)
                            return
                return hook
            for name, mod in self.model.named_modules():
                if "embed_tokens" in name:
                    hooks.append(mod.register_forward_hook(make_hook(name)))
                elif "layers.0." in name:
                    hooks.append(mod.register_forward_hook(make_hook(name)))
                elif "layers." in name and name.count(".") <= 3:
                    hooks.append(mod.register_forward_hook(make_hook(name)))

        self.model.train()

        # accelerator.accumulate() handles:
        #   - gradient scaling (loss / grad_accum_steps)
        #   - no_sync for non-boundary micro-steps (skips all-reduce)
        #   - optimizer.step() only on accumulation boundary
        with self.accelerator.accumulate(self.model):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                    output_router_logits=False,
                    return_dict=True,
                )
                loss = outputs.loss

            # Clean up hooks
            for h in hooks:
                h.remove()

            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients and self.args.grad_clip > 0:
                self.accelerator.clip_grad_norm_(
                    [p for _, p in self.model.named_parameters() if p.requires_grad],
                    self.args.grad_clip,
                )
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

        loss_val = loss.item()
        self._micro_step += 1

        # Track per-micro-step loss for averaging
        if not (loss_val != loss_val):  # not NaN
            self._micro_loss_accum += loss_val
            self._micro_loss_count += 1

        # Check if this was an optimizer step (accumulation boundary)
        gas = self.args.gradient_accumulation_steps
        if self._micro_step % gas == 0:
            self.global_step += 1
            current_step = self.global_step

            # Compute avg loss over the accumulated micro-steps
            accum_avg_loss = self._micro_loss_accum / max(1, self._micro_loss_count)
            self._micro_loss_accum = 0.0
            self._micro_loss_count = 0

            # Global running average
            self._loss_accum += accum_avg_loss
            self._loss_count += 1

            # Timing
            step_end = time.time()
            if self._train_start_time is not None:
                self._step_times.append(step_end)

            if current_step % self.args.log_every == 0 and self.accelerator.is_main_process:
                elapsed = step_end - self._train_start_time if self._train_start_time else 0
                avg_loss = self._loss_accum / max(1, self._loss_count)

                # Speed: use last 50 optimizer steps for smoothed estimate
                if len(self._step_times) >= 2:
                    window = min(50, len(self._step_times))
                    recent_elapsed = self._step_times[-1] - self._step_times[-window]
                    steps_per_sec = (window - 1) / max(recent_elapsed, 1e-6)
                    sec_per_step = 1.0 / max(steps_per_sec, 1e-6)
                else:
                    steps_per_sec = 0
                    sec_per_step = 0

                # ETA
                eta_str = "?"
                pct_str = ""
                if self.estimated_total_steps > 0:
                    pct = current_step / self.estimated_total_steps * 100
                    pct_str = f" ({pct:.1f}%)"
                    remaining_steps = max(0, self.estimated_total_steps - current_step)
                    if steps_per_sec > 0:
                        eta_sec = remaining_steps / steps_per_sec
                        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                elif steps_per_sec > 0:
                    eta_str = "unknown"

                elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))

                # Progress bar (50 chars wide)
                bar = ""
                if self.estimated_total_steps > 0:
                    filled = int(50 * current_step / self.estimated_total_steps)
                    filled = min(filled, 50)
                    bar = f" |{'█' * filled}{'░' * (50 - filled)}|"

                print(
                    f"[ep {self.global_epoch} step {current_step:>6d}"
                    f"{'/' + str(self.estimated_total_steps) if self.estimated_total_steps > 0 else ''}"
                    f"{pct_str}]"
                    f" loss={accum_avg_loss:.4f} avg={avg_loss:.4f}"
                    f" | {sec_per_step:.2f}s/step"
                    f" | elapsed={elapsed_str} ETA={eta_str}"
                    f"{bar}",
                    flush=True,
                )

                # W&B logging
                if self.use_wandb:
                    import wandb
                    log_dict = {
                        "train/loss": accum_avg_loss,
                        "train/loss_avg": avg_loss,
                        "train/epoch": self.global_epoch,
                        "train/lr": self.opt.param_groups[0].get("lr", self.args.lr),
                        "perf/sec_per_step": sec_per_step,
                        "perf/steps_per_sec": steps_per_sec,
                    }
                    if self.estimated_total_steps > 0:
                        log_dict["train/progress_pct"] = current_step / self.estimated_total_steps * 100
                    wandb.log(log_dict, step=current_step)

    def _should_stop(self):
        return (self.args.steps and self.args.steps > 0
                and self.global_step >= self.args.steps)

    @torch.no_grad()
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        def clean_key(k):
            k = k.replace("_fsdp_wrapped_module.", "")
            k = k.replace("_checkpoint_wrapped_module.", "")
            k = k.replace("module.", "")
            if k.startswith("model."):
                k = k[6:]
            return k

        if self.is_fsdp:
            with FSDP.state_dict_type(
                self.model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
            ):
                sd = self.model.state_dict()

            unwrapped = self.accelerator.unwrap_model(self.model)
            trainable_names = {
                clean_key(n) for n, p in unwrapped.named_parameters()
                if p.requires_grad
            }

            sub = {k: v for k, v in sd.items()
                   if clean_key(k) in trainable_names}

            path = os.path.join(save_dir, f"attn_params.rank{self.rank}.pt")
            torch.save(sub, path)

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                meta = {
                    "format": "lora_sharded" if self.args.use_lora else "partial_layer_sharded",
                    "pattern": "attn_params.rank{rank}.pt",
                    "world_size": int(self.world_size),
                    "model": self.args.model,
                    "train_layers": list(range(
                        self.args.num_layers - self.args.train_layers,
                        self.args.num_layers)),
                    "train_modules": self.args.train_modules,
                    "use_lora": self.args.use_lora,
                    "lora_rank": self.args.lora_rank if self.args.use_lora else None,
                    "lora_alpha": self.args.lora_alpha if self.args.use_lora else None,
                    "matched_keys": len(sub),
                }
                with open(os.path.join(save_dir, "attn_params_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"[SAVE] -> {save_dir} ({len(sub)} keys)", flush=True)
        else:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped = self.accelerator.unwrap_model(self.model)
                sub = {n: p.detach().cpu()
                       for n, p in unwrapped.named_parameters() if p.requires_grad}
                path = os.path.join(save_dir, "attn_params.pt")
                torch.save(sub, path)
                print(f"[SAVE] -> {path} ({len(sub)} keys)", flush=True)

    def train(self):
        self.model.train()
        self._train_start_time = time.time()
        self._step_times = [self._train_start_time]
        self._loss_accum = 0.0
        self._loss_count = 0
        self._micro_step = 0
        self._micro_loss_accum = 0.0
        self._micro_loss_count = 0

        if self.args.epochs and self.args.epochs > 0:
            for ep in range(1, self.args.epochs + 1):
                self.global_epoch = ep
                epoch_start = time.time()
                epoch_step_start = self.global_step
                if self.accelerator.is_main_process:
                    print(f"\n[EPOCH] start {ep}/{self.args.epochs}", flush=True)

                # Materialize batches and sync count across ranks to prevent
                # FSDP hang from rank-imbalanced iteration counts.
                # With IterableDataset + packing, each rank can produce a
                # different number of micro-batches depending on token lengths.
                epoch_batches = list(self.loader)
                local_count = torch.tensor([len(epoch_batches)], device=self.device, dtype=torch.long)
                dist.all_reduce(local_count, op=dist.ReduceOp.MIN)
                safe_count = int(local_count.item())
                if self.accelerator.is_main_process:
                    print(f"[SYNC] local_batches={len(epoch_batches)}, "
                          f"min_across_ranks={safe_count}", flush=True)

                for batch in epoch_batches[:safe_count]:
                    self.step(batch)
                    if self._should_stop():
                        break
                del epoch_batches
                # Epoch summary
                if self.accelerator.is_main_process:
                    epoch_elapsed = time.time() - epoch_start
                    epoch_steps = self.global_step - epoch_step_start
                    avg_loss = self._loss_accum / max(1, self._loss_count)
                    print(
                        f"\n[EPOCH] finished {ep}/{self.args.epochs} "
                        f"| steps={epoch_steps} "
                        f"| avg_loss={avg_loss:.4f} "
                        f"| time={str(datetime.timedelta(seconds=int(epoch_elapsed)))}",
                        flush=True,
                    )
                if self._should_stop():
                    break
        else:
            if not (self.args.steps and self.args.steps > 0):
                raise RuntimeError("Need --steps > 0 or --epochs > 0")
            while self.global_step < self.args.steps:
                for batch in self.loader:
                    self.step(batch)
                    if self.global_step >= self.args.steps:
                        break

        # Final summary
        if self.accelerator.is_main_process:
            total_elapsed = time.time() - self._train_start_time
            avg_loss = self._loss_accum / max(1, self._loss_count)
            print(
                f"\n{'=' * 56}\n"
                f"  Training complete!\n"
                f"  Total steps : {self.global_step:,}\n"
                f"  Avg loss    : {avg_loss:.4f}\n"
                f"  Total time  : {str(datetime.timedelta(seconds=int(total_elapsed)))}\n"
                f"  Avg speed   : {total_elapsed / max(1, self.global_step):.2f}s/step\n"
                f"{'=' * 56}",
                flush=True,
            )

            if self.use_wandb:
                import wandb
                wandb.summary["final_avg_loss"] = avg_loss
                wandb.summary["total_steps"] = self.global_step
                wandb.summary["total_time_sec"] = total_elapsed
                wandb.summary["avg_sec_per_step"] = total_elapsed / max(1, self.global_step)

        self.accelerator.wait_for_everyone()
        self.save(self.args.save_dir)

        # Finish wandb after save
        if self.use_wandb and self.accelerator.is_main_process:
            import wandb
            wandb.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--num_layers", type=int, default=78)
    ap.add_argument("--train_layers", type=int, default=1,
                    help="Number of layers from the end to train (e.g., 1=layer77, 2=layers76-77)")
    ap.add_argument("--train_modules", type=str, default="attn",
                    choices=["attn", "mlp", "full"],
                    help="Which modules to train: attn (self_attn+layernorm), "
                         "mlp (mlp/moe+layernorm), full (entire layer)")
    ap.add_argument("--use_lora", action="store_true", default=False,
                    help="Use LoRA instead of full fine-tuning")
    ap.add_argument("--lora_rank", type=int, default=16,
                    help="LoRA rank (r)")
    ap.add_argument("--lora_alpha", type=float, default=32,
                    help="LoRA alpha (scaling = alpha / rank)")
    ap.add_argument("--lora_dropout", type=float, default=0.0,
                    help="LoRA dropout rate")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of micro-batches before optimizer step")
    ap.add_argument("--bucket_buffer", type=int, default=256)
    ap.add_argument("--packing", action="store_true", default=False,
                    help="Pack multiple short samples into seq_len to maximize GPU utilization")
    ap.add_argument("--max_batch_len", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--optim", type=str, default="adamw",
                    choices=["adamw"])
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--wandb_project", type=str, default="",
                    help="W&B project name. Empty = wandb disabled.")
    ap.add_argument("--wandb_run_name", type=str, default="",
                    help="W&B run name. Empty = auto-generated.")
    ap.add_argument("--wandb_entity", type=str, default="",
                    help="W&B entity (team/user). Empty = default.")
    ap.add_argument("--save_dir", type=str,
                    default="output/glm5_lastlayer_attn_sft")
    args = ap.parse_args()

    set_seed(42)
    trainer = LastLayerAttnSFTTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
