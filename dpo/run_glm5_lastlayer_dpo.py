#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partial-layer DPO for GLM-5-FP8 (FSDP).

Based on run_lastlayer_attn_sft.py — same model loading / FSDP / FP8 handling /
layer selection logic, with DPO-specific changes:
  - Reads JSONL with {prompt, chosen, rejected} fields
  - Precomputes reference log-probs (distributed) before training
  - DPO loss: -log σ(β · (Δ_chosen - Δ_rejected))
  - Concatenates chosen+rejected into one forward pass per step
  - Optional label smoothing, length normalisation

Supports:
  - Full fine-tuning of selected modules (attn, mlp, or full layer)
  - LoRA adaptation (--use_lora) — base FP8 weights stay frozen
  - Configurable number of trainable layers (--train_layers N)
  - Gradient accumulation (--gradient_accumulation_steps)
  - W&B logging (--wandb_project)

FP8 handling:
  - Forward pre/post hooks on submodules for block-wise dequant
  - Full fine-tune: trainable layers dequanted FP8 → BF16
  - LoRA: base weights stay FP8, LoRA A/B added as BF16

Loading strategy:
  - rank 0: from_pretrained() loads FP8 weights to CPU (~760GB)
  - rank != 0: meta context → zero CPU RAM
  - FSDP sync_module_states=True broadcasts from rank 0
"""

import os
import json
import math
import random
import argparse
import contextlib
import datetime
import time
from typing import List, Dict, Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
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


# ============================================================
# Utilities
# ============================================================

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
# FP8 → BF16 dequantization (identical to SFT)
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


# ============================================================
# FP8 dequant hooks (identical to SFT)
# ============================================================

def install_fp8_dequant_hooks(model, block_size=(128, 128)):
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    hooks = []
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
            w.data = _blockwise_dequant(w.data, scale_data, bs)
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
        if (hasattr(mod, "weight")
                and isinstance(mod.weight, (torch.nn.Parameter, torch.Tensor))
                and mod.weight.dtype in fp8_dtypes
                and hasattr(mod, "weight_scale_inv")):
            h1 = mod.register_forward_pre_hook(_make_pre_hook("weight", "weight_scale_inv", block_size, name))
            h2 = mod.register_forward_hook(_make_post_hook("weight"))
            hooks.extend([h1, h2])
            installed += 1
            continue

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

    skip_module_ids = set()
    for idx in trainable_layer_indices:
        for m in layers[idx].modules():
            skip_module_ids.add(id(m))

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
                continue
            data = param.data
            module._parameters.pop(attr)
            module.register_buffer(attr, data)
            count += 1

    return count


def convert_scales_to_buffers(model):
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
# Trainable parameter selection (identical to SFT)
# ============================================================

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
        "include": None,
        "description": "all modules in layer",
    },
}


def inject_lora_adapters(model, train_layer_indices, train_modules,
                         lora_rank, lora_alpha, lora_dropout, rank=0):
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

    skip_lower = ("layernorm", "layer_norm", "rmsnorm", "norm")
    scaling = lora_alpha / lora_rank
    dropout_mod = nn.Dropout(lora_dropout) if lora_dropout > 0 else None

    injected = 0
    total_lora_params = 0

    for name, module in model.named_modules():
        if not any(pref in name for pref in target_prefixes):
            continue
        name_lower = name.lower().split(".")[-1]
        if any(s in name_lower for s in skip_lower):
            continue
        w = getattr(module, "weight", None)
        if w is None or not isinstance(w, (nn.Parameter, torch.Tensor)):
            continue
        if w.dim() < 2 or w.dim() > 2:
            continue

        out_features, in_features = w.shape[0], w.shape[1]
        lora_A = nn.Parameter(
            torch.empty(lora_rank, in_features, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        lora_B = nn.Parameter(
            torch.zeros(out_features, lora_rank, dtype=torch.bfloat16))

        module.register_parameter("lora_A", lora_A)
        module.register_parameter("lora_B", lora_B)

        def _make_lora_hook(s, drop):
            def _hook(mod, inp, out):
                x = inp[0] if isinstance(inp, tuple) else inp
                h = x.to(mod.lora_A.dtype) @ mod.lora_A.T
                if drop is not None:
                    h = drop(h)
                h = h @ mod.lora_B.T * s
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
        for name, p in model.named_parameters():
            total += p.numel()
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad = True
                trainable += p.numel()
            else:
                p.requires_grad = False
    else:
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

    mode_str = f"lora({train_modules})" if use_lora else train_modules
    layers_str = ",".join(str(i) for i in sorted(train_set))
    print(
        f"[INFO] trainable({mode_str})={trainable:,} / total={total:,} "
        f"({trainable / max(1, total) * 100:.4f}%) "
        f"[layers {layers_str}, modules={description}]",
        flush=True,
    )
    return trainable


# ============================================================
# Data: Tokenization helper (from SFT)
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


def tokenize_prompt_ref(tok, prompt, ref, seq_len):
    """Tokenize prompt+response, masking prompt tokens in labels."""
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        msgs_user = [{"role": "user", "content": prompt}]
        msgs_full = [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": ref}]

        ids_user_raw = tok.apply_chat_template(
            msgs_user, add_generation_prompt=True, tokenize=True)
        ids_full_raw = tok.apply_chat_template(
            msgs_full, add_generation_prompt=False, tokenize=True)

        def _to_list(x):
            if isinstance(x, list):
                return x
            if isinstance(x, torch.Tensor):
                return x.view(-1).tolist()
            if isinstance(x, dict):
                return _to_list(x["input_ids"])
            if hasattr(x, "ids"):
                return x.ids
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


# ============================================================
# DPO Dataset
# ============================================================

class DPOMapDataset(Dataset):
    """Map-style dataset for DPO training.

    Reads JSONL with {prompt, chosen, rejected, metadata?} fields.
    Tokenises (prompt+chosen) and (prompt+rejected) with prompt masking.
    """

    def __init__(self, path: str, tok, seq_len: int):
        self.tok = tok
        self.seq_len = seq_len
        self.samples: List[Dict] = []

        self.ref_chosen_logps: Optional[List[float]] = None
        self.ref_rejected_logps: Optional[List[float]] = None

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                result = self._tokenize_pair(obj)
                if result is not None:
                    self.samples.append(result)

        print(f"[DPO-DS] Loaded {len(self.samples)} pairs from {path}", flush=True)

    def _tokenize_pair(self, obj: dict) -> Optional[Dict]:
        prompt = obj.get("prompt")
        chosen = obj.get("chosen")
        rejected = obj.get("rejected")

        if not prompt or not chosen or not rejected:
            return None

        chosen_ids, chosen_labels = tokenize_prompt_ref(
            self.tok, prompt, chosen, self.seq_len)
        rejected_ids, rejected_labels = tokenize_prompt_ref(
            self.tok, prompt, rejected, self.seq_len)

        if not chosen_ids or not rejected_ids:
            return None

        n_chosen_train = sum(1 for l in chosen_labels if l != -100)
        n_rejected_train = sum(1 for l in rejected_labels if l != -100)
        if n_chosen_train == 0 or n_rejected_train == 0:
            return None

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        result = {
            "chosen_input_ids": s["chosen_input_ids"],
            "chosen_labels": s["chosen_labels"],
            "rejected_input_ids": s["rejected_input_ids"],
            "rejected_labels": s["rejected_labels"],
            "index": idx,
        }
        if self.ref_chosen_logps is not None:
            result["ref_chosen_logps"] = self.ref_chosen_logps[idx]
            result["ref_rejected_logps"] = self.ref_rejected_logps[idx]
        return result


def dpo_collate_fn(batch, pad_id, seq_len):
    """Collate DPO batch — chosen and rejected are padded to the SAME length.

    This is essential for concatenated forward (torch.cat along batch dim)
    which requires matching seq_len across chosen and rejected.
    """
    all_c_ids = [b["chosen_input_ids"] for b in batch]
    all_c_labels = [b["chosen_labels"] for b in batch]
    all_r_ids = [b["rejected_input_ids"] for b in batch]
    all_r_labels = [b["rejected_labels"] for b in batch]

    # Single max_len across both chosen AND rejected
    max_len = min(
        max(
            max(len(ids) for ids in all_c_ids),
            max(len(ids) for ids in all_r_ids),
        ),
        seq_len,
    )

    def _pad_to(all_ids, all_labels):
        ids_t = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        mask_t = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels_t = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, (ids, labs) in enumerate(zip(all_ids, all_labels)):
            slen = min(len(ids), max_len)
            ids_t[i, :slen] = torch.tensor(ids[:slen], dtype=torch.long)
            mask_t[i, :slen] = 1
            labels_t[i, :slen] = torch.tensor(labs[:slen], dtype=torch.long)

        return ids_t, mask_t, labels_t

    c_ids, c_mask, c_labels = _pad_to(all_c_ids, all_c_labels)
    r_ids, r_mask, r_labels = _pad_to(all_r_ids, all_r_labels)

    result = {
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_mask,
        "chosen_labels": c_labels,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_mask,
        "rejected_labels": r_labels,
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }

    if "ref_chosen_logps" in batch[0]:
        result["ref_chosen_logps"] = torch.tensor(
            [b["ref_chosen_logps"] for b in batch], dtype=torch.float32)
        result["ref_rejected_logps"] = torch.tensor(
            [b["ref_rejected_logps"] for b in batch], dtype=torch.float32)

    return result


# ============================================================
# DPO Loss
# ============================================================

def _cross_entropy_no_save(logits_flat: torch.Tensor, labels_flat: torch.Tensor) -> torch.Tensor:
    """Pure function for checkpoint: F.cross_entropy with reduction='none'.

    torch.utils.checkpoint will recompute this during backward instead of
    saving the softmax probabilities tensor (shape (B*T, V) float32 ≈ 2.3 GiB
    for V=150K, T=4096). This is the single biggest saved-tensor in the
    autograd graph.
    """
    return F.cross_entropy(logits_flat, labels_flat, reduction='none')


def compute_per_sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Memory-efficient per-sequence log-probability.

    Uses checkpointed F.cross_entropy so that the softmax saved tensor
    (~2.3 GiB for GLM-5 vocab) is NOT held during backward — it is
    recomputed on the fly instead.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    mask = (shift_labels != -100)
    B, T, V = shift_logits.shape

    # Checkpointed CE: avoids saving (B*T, V) float32 softmax for backward
    per_token_ce = torch.utils.checkpoint.checkpoint(
        _cross_entropy_no_save,
        shift_logits.view(-1, V),
        shift_labels.clamp(min=0).view(-1),
        use_reentrant=False,
    ).view(B, T)

    per_token_logps = -per_token_ce * mask.float()
    return per_token_logps.sum(dim=-1)


def compute_per_sequence_logps_with_length(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same as compute_per_sequence_logps but also returns sequence lengths."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    mask = (shift_labels != -100)
    B, T, V = shift_logits.shape

    per_token_ce = torch.utils.checkpoint.checkpoint(
        _cross_entropy_no_save,
        shift_logits.view(-1, V),
        shift_labels.clamp(min=0).view(-1),
        use_reentrant=False,
    ).view(B, T)

    per_token_logps = -per_token_ce * mask.float()
    seq_logps = per_token_logps.sum(dim=-1)
    seq_lengths = mask.float().sum(dim=-1)
    return seq_logps, seq_lengths


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
    length_normalise: bool = False,
    chosen_lengths: Optional[torch.Tensor] = None,
    rejected_lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pc = policy_chosen_logps
    pr = policy_rejected_logps
    rc = ref_chosen_logps.to(pc.device)
    rr = ref_rejected_logps.to(pr.device)

    if length_normalise and chosen_lengths is not None and rejected_lengths is not None:
        pc = pc / chosen_lengths.clamp(min=1).to(pc.device)
        pr = pr / rejected_lengths.clamp(min=1).to(pr.device)
        rc = rc / chosen_lengths.clamp(min=1).to(rc.device)
        rr = rr / rejected_lengths.clamp(min=1).to(rr.device)

    chosen_rewards = beta * (pc - rc)
    rejected_rewards = beta * (pr - rr)

    if loss_type == "sigmoid":
        logits = chosen_rewards - rejected_rewards
        if label_smoothing > 0:
            loss = (
                (1 - label_smoothing) * F.logsigmoid(-logits)
                + label_smoothing * F.logsigmoid(logits)
            )
            loss = -loss.mean()
        else:
            loss = -F.logsigmoid(logits).mean()
    elif loss_type == "hinge":
        loss = torch.relu(1.0 - (chosen_rewards - rejected_rewards)).mean()
    elif loss_type == "ipo":
        logits = (pc - pr) - (rc - rr)
        loss = ((logits - 1.0 / (2.0 * beta)) ** 2).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, chosen_rewards.detach(), rejected_rewards.detach()


# ============================================================
# DPO Trainer
# ============================================================

class DPOTrainer:
    def __init__(self, args):
        self.args = args

        if not _FSDP_AVAILABLE:
            raise RuntimeError("torch.distributed.fsdp not available.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required.")

        set_seed(args.seed)
        configure_sdp_backends()
        self.global_step = 0
        self.global_epoch = 0
        self.use_wandb = False

        # --------------------------------------------------
        # 1) Detect layer class on meta device
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
        # --------------------------------------------------
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            cpu_ram_efficient_loading=True,
            sync_module_states=True,
            limit_all_gathers=True,
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing=False,
        )

        fsdp_plugin.mixed_precision_policy = MixedPrecision(
            param_dtype=None,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=None,
        )

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

        pg_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))
        # sync_each_batch=True: disable no_sync to avoid +16GB full gradient
        # buffer that causes OOM during gradient accumulation with FSDP.
        # Trade-off: extra reduce-scatter per micro step (trainable layer only,
        # ~few hundred MB, negligible overhead).
        ga_plugin = GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps,
            sync_each_batch=True,
        )
        self.accelerator = Accelerator(
            mixed_precision="no",
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
        # 3) Model loading
        # --------------------------------------------------
        self.tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

        if self.accelerator.is_main_process:
            print(f"[LOAD] model: {args.model}", flush=True)

        if self.rank == 0:
            print("[LOAD] rank 0: loading real weights to CPU...", flush=True)
            import transformers.modeling_utils as _tm_utils
            _orig_finalize_r0 = _tm_utils.PreTrainedModel._finalize_model_loading
            @classmethod
            def _skip_finalize(cls, model, *args, **kwargs):
                return {"missing_keys": [], "unexpected_keys": [],
                        "mismatched_keys": [], "error_msgs": []}
            _tm_utils.PreTrainedModel._finalize_model_loading = _skip_finalize
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.model, config=config, device_map=None,
                    trust_remote_code=True, low_cpu_mem_usage=True,
                )
            finally:
                _tm_utils.PreTrainedModel._finalize_model_loading = _orig_finalize_r0
        else:
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
                        args.model, config=config, device_map=None,
                        trust_remote_code=True, low_cpu_mem_usage=True,
                    )
            finally:
                _tm_utils.PreTrainedModel._load_pretrained_model = _orig_load_model
                _tm_utils.PreTrainedModel._finalize_model_loading = _orig_finalize
                if _orig_check is not None:
                    _tm_utils.check_and_set_device_map = _orig_check
            print(f"[LOAD] rank {self.rank}: meta model created successfully.", flush=True)

        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False

        base_model.train()

        # --------------------------------------------------
        # 4) FP8 dtype handling + trainable layer preparation
        # --------------------------------------------------
        last_layer_idx = args.num_layers - 1
        train_layer_indices = list(range(
            last_layer_idx - args.train_layers + 1, last_layer_idx + 1))
        first_train_idx = train_layer_indices[0]
        self.train_layer_indices = train_layer_indices
        layers_str = ",".join(str(i) for i in train_layer_indices)

        # Step 1: scales → buffers
        n_scales = convert_scales_to_buffers(base_model)
        if self.accelerator.is_main_process:
            print(f"[DTYPE] Converted {n_scales} scale Parameters to buffers.", flush=True)

        # Step 2: dequant / LoRA injection
        if args.use_lora:
            if self.accelerator.is_main_process:
                print(f"[LoRA] Injecting adapters into layers [{layers_str}]...", flush=True)
            n_lora = inject_lora_adapters(
                base_model, train_layer_indices, args.train_modules,
                lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout, rank=self.rank,
            )
            if n_lora == 0:
                raise RuntimeError(
                    f"No LoRA adapters injected for layers {train_layer_indices}")
        else:
            if self.accelerator.is_main_process:
                print(f"[DEQUANT] Converting layers [{layers_str}] FP8 → BF16...", flush=True)
            total_dequant = 0
            for lidx in train_layer_indices:
                n_dequant = dequantize_single_layer(base_model, lidx, rank=self.rank)
                total_dequant += n_dequant
                if self.accelerator.is_main_process:
                    print(f"[DEQUANT] Layer {lidx}: {n_dequant} tensors.", flush=True)

        # Step 3: frozen non-FP8 params → buffers
        n_buf = convert_frozen_non_fp8_to_buffers(base_model, train_layer_indices)
        if self.accelerator.is_main_process:
            print(f"[DTYPE] Converted {n_buf} frozen non-FP8 Parameters to buffers.", flush=True)

        # Freeze all except trainable
        trainable_count = mark_trainable(
            base_model, train_layer_indices, args.train_modules,
            use_lora=args.use_lora,
        )
        if trainable_count == 0:
            raise RuntimeError(f"No trainable params for layers {train_layer_indices}")

        # --------------------------------------------------
        # 5) FSDP wrap
        # --------------------------------------------------
        if self.rank != 0:
            for module in base_model.modules():
                for bname in list(module._buffers.keys()):
                    buf = module._buffers[bname]
                    if buf is not None and buf.device.type == "meta":
                        module._buffers[bname] = torch.empty(
                            buf.shape, dtype=buf.dtype, device="cpu")

        self.accelerator.wait_for_everyone()

        if self.rank == 0:
            dtype_counts = {}
            for name, p in base_model.named_parameters():
                dt = str(p.dtype)
                dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
            print(f"[DIAG-PRE] Parameter dtypes: {dtype_counts}", flush=True)

        if self.accelerator.is_main_process:
            print("[FSDP] Wrapping model (sync_module_states=True)...", flush=True)

        self.model = self.accelerator.prepare(base_model)
        self.is_fsdp = isinstance(self.model, FSDP)

        if self.accelerator.is_main_process:
            print("[FSDP] Model wrapped successfully.", flush=True)

        # Install FP8 dequant hooks
        fp8_hooks, n_fp8_patched = install_fp8_dequant_hooks(self.model)
        if self.accelerator.is_main_process:
            print(f"[FP8] Installed dequant hooks on {n_fp8_patched} FP8 submodules.", flush=True)

        # --------------------------------------------------
        # 6) Post-FSDP detach hooks
        # --------------------------------------------------
        def _find_layers_list(model):
            for mod in model.modules():
                if isinstance(mod, torch.nn.ModuleList) and len(mod) == args.num_layers:
                    return mod
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
                print(f"[DETACH] Found layers ({len(fsdp_layers)} layers), "
                      f"target: index {first_train_idx} → {type(fsdp_train_layer).__name__}",
                      flush=True)

        if fsdp_train_layer is not None:
            def _detach_and_enable_grad(module, args_tuple):
                new_args = []
                for a in args_tuple:
                    if isinstance(a, torch.Tensor):
                        new_args.append(a.detach().requires_grad_(True))
                    else:
                        new_args.append(a)
                return tuple(new_args)

            fsdp_train_layer.register_forward_pre_hook(_detach_and_enable_grad)
            if self.accelerator.is_main_process:
                print(f"[DETACH] Installed detach+grad hook on layer {first_train_idx} "
                      f"(frozen layers 0-{first_train_idx-1} run without grad)", flush=True)
        else:
            # Fallback: name-based search
            if self.accelerator.is_main_process:
                print(f"[WARN] Falling back to name-based module search.", flush=True)
            for name, module in self.model.named_modules():
                if f"layers.{first_train_idx}" in name and isinstance(module, FSDP):
                    name_parts = name.split(".")
                    try:
                        idx_pos = name_parts.index(str(first_train_idx))
                        if idx_pos > 0 and name_parts[idx_pos - 1] == "layers":
                            if idx_pos == len(name_parts) - 1:
                                def _detach_and_enable_grad_fb(module, args_tuple):
                                    new_args = []
                                    for a in args_tuple:
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
            print(f"[DETACH] Summary: layer-input detach="
                  f"{'OK' if fsdp_train_layer else 'FAILED'}", flush=True)

        # MLP output detach hooks
        # Only for attn-only training: with train_modules mlp/full the MLP
        # params are trainable and detaching would cut their gradients.
        mlp_detach_count = 0
        if args.train_modules == "attn":
            for tidx in train_layer_indices:
                layer_mod = fsdp_layers[tidx] if fsdp_layers is not None else None
                if layer_mod is None:
                    continue
                inner = getattr(layer_mod, '_fsdp_wrapped_module', layer_mod)
                inner = getattr(inner, '_checkpoint_wrapped_module', inner)
                mlp_module = None
                for attr_name in ("mlp", "moe", "feed_forward", "ffn"):
                    mlp_module = getattr(inner, attr_name, None)
                    if mlp_module is not None:
                        break
                if mlp_module is not None:
                    def _make_mlp_detach(idx):
                        def _hook(module, input, output):
                            if isinstance(output, torch.Tensor) and output.requires_grad:
                                return output.detach().requires_grad_(True)
                            elif isinstance(output, tuple):
                                new = []
                                for o in output:
                                    if isinstance(o, torch.Tensor) and o.requires_grad:
                                        new.append(o.detach().requires_grad_(True))
                                    else:
                                        new.append(o)
                                return tuple(new)
                            return output
                        return _hook
                    mlp_module.register_forward_hook(_make_mlp_detach(tidx))
                    mlp_detach_count += 1
                    if self.accelerator.is_main_process:
                        print(f"[DETACH] MLP output detach on layer {tidx} "
                              f"({type(mlp_module).__name__})", flush=True)

        if self.accelerator.is_main_process and mlp_detach_count > 0:
            print(f"[DETACH] MLP detach: {mlp_detach_count} layers", flush=True)

        # --------------------------------------------------
        # 7) Optimizer
        # --------------------------------------------------
        trainable_params = [p for _, p in self.model.named_parameters()
                           if p.requires_grad and p.numel() > 0]

        if self.accelerator.is_main_process:
            print(f"[OPT] Trainable params for optimizer: {len(trainable_params)}", flush=True)

        has_params = len(trainable_params) > 0
        if self.accelerator.is_main_process and not has_params:
            raise RuntimeError("No trainable parameters on rank 0 after FSDP wrapping.")

        if not trainable_params:
            _dummy = nn.Parameter(torch.zeros(1, device=self.device, dtype=torch.bfloat16))
            _dummy.requires_grad = True
            trainable_params = [_dummy]
            self._dummy_param = _dummy

        self.opt = torch.optim.AdamW(
            trainable_params, lr=args.lr, betas=(0.9, 0.98),
            weight_decay=args.weight_decay, foreach=False,
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

        # --------------------------------------------------
        # 8) Dataset
        # --------------------------------------------------
        if self.accelerator.is_main_process:
            print(f"[DATA] dataset: {args.dataset}", flush=True)

        self.dataset = DPOMapDataset(args.dataset, tok=self.tok, seq_len=args.seq_len)

        if self.accelerator.is_main_process and len(self.dataset) > 0:
            s0 = self.dataset.samples[0]
            for side in ("chosen", "rejected"):
                ids = s0[f"{side}_input_ids"]
                labs = s0[f"{side}_labels"]
                n_train = sum(1 for l in labs if l != -100)
                n_mask = sum(1 for l in labs if l == -100)
                print(f"[DATA] sample 0 {side}: {len(ids)} tokens, "
                      f"{n_train} trainable, {n_mask} masked", flush=True)

        # --------------------------------------------------
        # 9) Precompute reference log-probs
        # --------------------------------------------------
        if self.accelerator.is_main_process:
            print(f"\n[REF] Precomputing reference log-probs "
                  f"({len(self.dataset)} pairs)...", flush=True)

        self._precompute_ref_logprobs()

        if self.accelerator.is_main_process:
            rc = self.dataset.ref_chosen_logps
            rr = self.dataset.ref_rejected_logps
            avg_c = sum(rc) / max(len(rc), 1)
            avg_r = sum(rr) / max(len(rr), 1)
            print(f"[REF] Done. avg ref_chosen_logp={avg_c:.2f}, "
                  f"avg ref_rejected_logp={avg_r:.2f}", flush=True)

        # --------------------------------------------------
        # 10) Training DataLoader
        # --------------------------------------------------
        self.train_sampler = DistributedSampler(
            self.dataset, num_replicas=self.world_size, rank=self.rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        _collate = partial(dpo_collate_fn, pad_id=self.pad_token_id, seq_len=args.seq_len)
        self.loader = DataLoader(
            self.dataset, batch_size=args.batch_size,
            sampler=self.train_sampler, collate_fn=_collate,
            num_workers=2, pin_memory=True, drop_last=True,
        )

        # Estimate steps
        gas = args.gradient_accumulation_steps
        eff_batch = args.batch_size * gas * self.world_size
        samples_per_rank = len(self.dataset) // self.world_size
        micro_per_epoch = max(1, samples_per_rank // args.batch_size)
        optim_per_epoch = micro_per_epoch // gas
        total_steps = optim_per_epoch * (args.epochs or 1)
        self.estimated_total_steps = max(1, total_steps)

        if self.accelerator.is_main_process:
            print(f"[DATA] total_pairs={len(self.dataset)}, "
                  f"micro_batches/epoch≈{micro_per_epoch}, "
                  f"optim_steps/epoch≈{optim_per_epoch}, "
                  f"estimated_total_steps≈{total_steps}, "
                  f"effective_batch_size={eff_batch}", flush=True)

        # Summary
        if self.accelerator.is_main_process:
            print("=" * 56)
            print(f"  GLM-5-FP8 Partial-Layer DPO "
                  f"({'LoRA' if args.use_lora else 'Full'})")
            print(f"  model        : {args.model}")
            print(f"  num_layers   : {args.num_layers}")
            print(f"  train_layers : {layers_str} ({len(train_layer_indices)} layers)")
            print(f"  train_modules: {args.train_modules} "
                  f"({TRAIN_MODULE_PATTERNS[args.train_modules]['description']})")
            if args.use_lora:
                print(f"  lora         : r={args.lora_rank}, α={args.lora_alpha}, "
                      f"dropout={args.lora_dropout}")
            print(f"  dataset      : {args.dataset}")
            print(f"  batch_size   : {args.batch_size} (micro, per rank)")
            print(f"  grad_accum   : {gas}")
            print(f"  eff_batch    : {eff_batch}")
            print(f"  seq_len      : {args.seq_len}")
            print(f"  beta         : {args.beta}")
            print(f"  loss_type    : {args.loss_type}")
            print(f"  label_smooth : {args.label_smoothing}")
            print(f"  len_normalise: {args.length_normalise}")
            print(f"  lr           : {args.lr}")
            print(f"  epochs       : {args.epochs}")
            print(f"  steps(cap)   : {args.steps}")
            print(f"  save_dir     : {args.save_dir}")
            print("=" * 56, flush=True)

        # W&B
        if self.accelerator.is_main_process and args.wandb_project:
            try:
                import wandb
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name or None,
                    entity=args.wandb_entity or None,
                    config={
                        "model": args.model,
                        "num_layers": args.num_layers,
                        "train_layers": train_layer_indices,
                        "train_modules": args.train_modules,
                        "use_lora": args.use_lora,
                        "lora_rank": args.lora_rank if args.use_lora else None,
                        "dataset": args.dataset,
                        "batch_size": args.batch_size,
                        "grad_accum": gas,
                        "eff_batch": eff_batch,
                        "seq_len": args.seq_len,
                        "beta": args.beta,
                        "loss_type": args.loss_type,
                        "lr": args.lr,
                        "epochs": args.epochs,
                        "world_size": self.world_size,
                    },
                    resume="allow",
                )
                self.use_wandb = True
                print(f"[WANDB] Initialized: project={args.wandb_project}, "
                      f"run={wandb.run.name}", flush=True)
            except Exception as e:
                print(f"[WANDB] Init failed: {e}. Continuing without wandb.", flush=True)

    # ==========================================================
    # Reference log-prob precomputation
    # ==========================================================

    @torch.no_grad()
    def _precompute_ref_logprobs(self):
        self.model.eval()

        n = len(self.dataset)
        n_padded = math.ceil(n / self.world_size) * self.world_size
        all_indices = list(range(n)) + [0] * (n_padded - n)

        per_rank = n_padded // self.world_size
        my_start = self.rank * per_rank
        my_end = my_start + per_rank
        my_indices = all_indices[my_start:my_end]

        # Force ref_bs=1: each forward is 1 DPO pair concatenated → batch=2,
        # exactly matching training's forward. For MoE models, expert routing
        # depends on batch composition, so batch=4 (ref_bs=2) would produce
        # different logits than training's batch=2, causing loss != log(2) at step 1.
        ref_bs = 1
        _collate = partial(dpo_collate_fn, pad_id=self.pad_token_id, seq_len=self.args.seq_len)

        local_results = {}

        t0 = time.time()
        num_batches = math.ceil(len(my_indices) / ref_bs)

        for bi in range(num_batches):
            batch_indices = my_indices[bi * ref_bs : (bi + 1) * ref_bs]
            batch_items = [self.dataset[i] for i in batch_indices]
            batch = _collate(batch_items)

            # Concatenate chosen + rejected into batch=2*ref_bs
            # This MUST match training forward exactly so that MoE expert
            # routing produces identical logits (routing depends on batch).
            c_ids = batch["chosen_input_ids"].to(self.device, non_blocking=True)
            c_mask = batch["chosen_attention_mask"].to(self.device, non_blocking=True)
            c_labels = batch["chosen_labels"].to(self.device, non_blocking=True)
            r_ids = batch["rejected_input_ids"].to(self.device, non_blocking=True)
            r_mask = batch["rejected_attention_mask"].to(self.device, non_blocking=True)
            r_labels = batch["rejected_labels"].to(self.device, non_blocking=True)

            all_ids = torch.cat([c_ids, r_ids], dim=0)
            all_mask = torch.cat([c_mask, r_mask], dim=0)
            all_labels = torch.cat([c_labels, r_labels], dim=0)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=all_ids, attention_mask=all_mask,
                    use_cache=False, output_router_logits=False, return_dict=True,
                )
            all_logps = compute_per_sequence_logps(outputs.logits, all_labels)
            bs = c_ids.size(0)
            c_logps = all_logps[:bs]
            r_logps = all_logps[bs:]
            del outputs, all_ids, all_mask, all_labels, all_logps

            for j, idx in enumerate(batch_indices):
                if idx < n:
                    local_results[idx] = (c_logps[j].item(), r_logps[j].item())

            if self.accelerator.is_main_process and (bi + 1) % max(1, num_batches // 5) == 0:
                elapsed = time.time() - t0
                print(f"  [REF] {bi+1}/{num_batches} batches, "
                      f"elapsed={elapsed:.1f}s", flush=True)

        # All-gather
        all_results = [None] * self.world_size
        dist.all_gather_object(all_results, local_results)

        # Merge
        ref_chosen = [0.0] * n
        ref_rejected = [0.0] * n
        for rank_dict in all_results:
            for idx, (c, r) in rank_dict.items():
                ref_chosen[idx] = c
                ref_rejected[idx] = r

        self.dataset.ref_chosen_logps = ref_chosen
        self.dataset.ref_rejected_logps = ref_rejected

        self.model.train()

        if self.accelerator.is_main_process:
            elapsed = time.time() - t0
            print(f"  [REF] Precomputation finished in {elapsed:.1f}s", flush=True)

    # ==========================================================
    # DPO gradient weight computation (detached)
    # ==========================================================

    @torch.no_grad()
    def _compute_dpo_weight(self, pc_val, pr_val, ref_c, ref_r,
                            c_lengths=None, r_lengths=None):
        """Compute the detached DPO gradient weight w such that:
            ∂L/∂θ = -w · ∂pc/∂θ  +  w · ∂pr/∂θ
        where pc, pr may be length-normalised.

        Returns (w, loss_val, chosen_rewards, rejected_rewards).
        """
        beta = self.args.beta
        pc, pr, rc, rr = pc_val, pr_val, ref_c, ref_r

        if self.args.length_normalise and c_lengths is not None and r_lengths is not None:
            pc = pc / c_lengths.clamp(min=1)
            pr = pr / r_lengths.clamp(min=1)
            rc = rc / c_lengths.clamp(min=1)
            rr = rr / r_lengths.clamp(min=1)

        chosen_rewards = beta * (pc - rc)
        rejected_rewards = beta * (pr - rr)
        reward_diff = chosen_rewards - rejected_rewards

        if self.args.loss_type == "sigmoid":
            eps = self.args.label_smoothing
            if eps > 0:
                w = ((1 - eps) * torch.sigmoid(-reward_diff)
                     - eps * torch.sigmoid(reward_diff)) * beta
                loss_val = -((1 - eps) * F.logsigmoid(reward_diff)
                             + eps * F.logsigmoid(-reward_diff)).mean().item()
            else:
                w = torch.sigmoid(-reward_diff) * beta
                loss_val = -F.logsigmoid(reward_diff).mean().item()
        elif self.args.loss_type == "hinge":
            active = (reward_diff < 1.0).float()
            w = beta * active
            loss_val = torch.relu(1.0 - reward_diff).mean().item()
        elif self.args.loss_type == "ipo":
            d = (pc - pr) - (rc - rr)
            residual = d - 1.0 / (2.0 * beta)
            w = -2.0 * residual
            loss_val = (residual ** 2).mean().item()
        else:
            raise ValueError(f"Unknown loss_type: {self.args.loss_type}")

        return w, loss_val, chosen_rewards, rejected_rewards

    # ==========================================================
    # Training step — concatenated forward + single backward
    # ==========================================================

    def step(self, batch):
        """DPO training step: concatenated chosen+rejected forward, single backward.

        Memory profile is identical to SFT batch=2:
          - Forward: batch=2 through full model (peak ~41 GB from FSDP unshards)
          - Log-prob: checkpointed F.cross_entropy (no saved softmax tensor)
          - Backward: single pass (peak ~71 GB)

        This avoids the sequential backward problem where gradients from
        the first backward (~16 GB) persist during the second backward,
        causing OOM (28 + 59 = 87 GB > 79.4 GB).
        """
        # DPO requires consistent log-probs between policy and reference.
        # Ref logps were precomputed in eval() mode (no dropout).
        # We must also forward in eval() mode so that at step 1 (before any
        # weight update) policy_logps == ref_logps and loss ≈ log(2) ≈ 0.693.
        # Gradients still flow through trainable params regardless of eval mode.
        self.model.eval()

        c_ids = batch["chosen_input_ids"].to(self.device, non_blocking=True)
        c_mask = batch["chosen_attention_mask"].to(self.device, non_blocking=True)
        c_labels = batch["chosen_labels"].to(self.device, non_blocking=True)
        r_ids = batch["rejected_input_ids"].to(self.device, non_blocking=True)
        r_mask = batch["rejected_attention_mask"].to(self.device, non_blocking=True)
        r_labels = batch["rejected_labels"].to(self.device, non_blocking=True)
        ref_c = batch["ref_chosen_logps"].to(self.device, non_blocking=True)
        ref_r = batch["ref_rejected_logps"].to(self.device, non_blocking=True)

        if self._micro_step == 0:
            torch.cuda.reset_peak_memory_stats()

        with self.accelerator.accumulate(self.model):
            # ---- Concatenate chosen + rejected into batch=2 ----
            # collator already pads both to the same seq_len
            bs = c_ids.size(0)  # per-side batch size (chosen == rejected)
            all_ids = torch.cat([c_ids, r_ids], dim=0)
            all_mask = torch.cat([c_mask, r_mask], dim=0)
            all_labels = torch.cat([c_labels, r_labels], dim=0)
            del c_ids, r_ids, c_mask, r_mask, c_labels, r_labels

            # ---- Single forward pass (same as SFT batch=2) ----
            torch.cuda.empty_cache()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=all_ids, attention_mask=all_mask,
                    use_cache=False, output_router_logits=False, return_dict=True,
                )
            del all_ids, all_mask

            # ---- Per-sequence log-probs via checkpointed CE ----
            # checkpoint avoids saving (B*T, V) float32 softmax tensor (~2.3 GB)
            logits = outputs.logits
            del outputs

            if self.args.length_normalise:
                all_logps, all_lengths = compute_per_sequence_logps_with_length(
                    logits, all_labels)
                pc_logps = all_logps[:bs]
                pr_logps = all_logps[bs:]
                c_lengths = all_lengths[:bs]
                r_lengths = all_lengths[bs:]
            else:
                all_logps = compute_per_sequence_logps(logits, all_labels)
                pc_logps = all_logps[:bs]
                pr_logps = all_logps[bs:]
                c_lengths = r_lengths = None
            del logits, all_labels, all_logps

            if self._micro_step == 0 and self.accelerator.is_local_main_process:
                _alloc = torch.cuda.memory_allocated() / 1e9
                _peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[MEM] after fwd+logps: "
                      f"alloc={_alloc:.1f}GB peak={_peak:.1f}GB")

            # ---- DPO loss (standard, single graph) ----
            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps=pc_logps,
                policy_rejected_logps=pr_logps,
                ref_chosen_logps=ref_c,
                ref_rejected_logps=ref_r,
                beta=self.args.beta,
                label_smoothing=self.args.label_smoothing,
                loss_type=self.args.loss_type,
                length_normalise=self.args.length_normalise,
                chosen_lengths=c_lengths,
                rejected_lengths=r_lengths if self.args.length_normalise else None,
            )
            del pc_logps, pr_logps

            # ---- Single backward (same peak as SFT batch=2 ~71 GB) ----
            torch.cuda.empty_cache()
            self.accelerator.backward(loss)

            if self._micro_step == 0 and self.accelerator.is_local_main_process:
                _alloc = torch.cuda.memory_allocated() / 1e9
                _peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[MEM] after backward: "
                      f"alloc={_alloc:.1f}GB peak={_peak:.1f}GB")

            # clip + step + zero_grad only on accumulation boundary.
            # accumulate() makes opt.step/zero_grad no-ops on non-boundary,
            # but clip_grad_norm_ is NOT wrapped — must guard explicitly.
            if self.accelerator.sync_gradients and self.args.grad_clip > 0:
                self.accelerator.clip_grad_norm_(
                    [p for _, p in self.model.named_parameters() if p.requires_grad],
                    self.args.grad_clip,
                )
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

        loss_val = loss.item()
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        self._micro_step += 1

        if not (loss_val != loss_val):  # not NaN
            self._micro_loss_accum += loss_val
            self._micro_margin_accum += reward_margin
            self._micro_acc_accum += accuracy
            self._micro_count += 1

        gas = self.args.gradient_accumulation_steps
        if self._micro_step % gas == 0:
            self.global_step += 1
            current_step = self.global_step

            avg_loss = self._micro_loss_accum / max(1, self._micro_count)
            avg_margin = self._micro_margin_accum / max(1, self._micro_count)
            avg_acc = self._micro_acc_accum / max(1, self._micro_count)

            self._epoch_loss_accum += avg_loss
            self._epoch_margin_accum += avg_margin
            self._epoch_acc_accum += avg_acc
            self._epoch_count += 1

            self._micro_loss_accum = 0.0
            self._micro_margin_accum = 0.0
            self._micro_acc_accum = 0.0
            self._micro_count = 0

            step_end = time.time()
            self._step_times.append(step_end)

            if current_step % self.args.log_every == 0 and self.accelerator.is_main_process:
                elapsed = step_end - self._train_start_time if self._train_start_time else 0
                run_avg_loss = self._epoch_loss_accum / max(1, self._epoch_count)

                if len(self._step_times) >= 2:
                    window = min(50, len(self._step_times))
                    recent = self._step_times[-1] - self._step_times[-window]
                    steps_per_sec = (window - 1) / max(recent, 1e-6)
                    sec_per_step = 1.0 / max(steps_per_sec, 1e-6)
                else:
                    steps_per_sec = sec_per_step = 0

                eta_str = "?"
                pct_str = ""
                if self.estimated_total_steps > 0:
                    pct = current_step / self.estimated_total_steps * 100
                    pct_str = f" ({pct:.1f}%)"
                    remaining = max(0, self.estimated_total_steps - current_step)
                    if steps_per_sec > 0:
                        eta_sec = remaining / steps_per_sec
                        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

                print(
                    f"  step {current_step}{pct_str} | "
                    f"loss={avg_loss:.4f} (avg={run_avg_loss:.4f}) | "
                    f"margin={avg_margin:.3f} | acc={avg_acc:.3f} | "
                    f"lr={self.opt.param_groups[0]['lr']:.2e} | "
                    f"{sec_per_step:.1f}s/step | "
                    f"elapsed={str(datetime.timedelta(seconds=int(elapsed)))} | "
                    f"ETA={eta_str}",
                    flush=True,
                )

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/avg_loss": run_avg_loss,
                        "train/reward_margin": avg_margin,
                        "train/accuracy": avg_acc,
                        "train/lr": self.opt.param_groups[0]["lr"],
                        "train/epoch": self.global_epoch,
                        "perf/sec_per_step": sec_per_step,
                    }, step=current_step)

            if (self.args.save_every and current_step % self.args.save_every == 0
                    and current_step > 0):
                self.save(os.path.join(self.args.save_dir, f"step-{current_step}"))

    def _should_stop(self):
        return (self.args.steps and self.args.steps > 0
                and self.global_step >= self.args.steps)

    # ==========================================================
    # Save
    # ==========================================================

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

            path = os.path.join(save_dir, f"params.rank{self.rank}.pt")
            torch.save(sub, path)

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                meta = {
                    "format": "dpo_lora_sharded" if self.args.use_lora else "dpo_partial_layer_sharded",
                    "pattern": "params.rank{rank}.pt",
                    "world_size": int(self.world_size),
                    "model": self.args.model,
                    "train_layers": list(self.train_layer_indices),
                    "train_modules": self.args.train_modules,
                    "use_lora": self.args.use_lora,
                    "lora_rank": self.args.lora_rank if self.args.use_lora else None,
                    "lora_alpha": self.args.lora_alpha if self.args.use_lora else None,
                    "beta": self.args.beta,
                    "loss_type": self.args.loss_type,
                    "matched_keys": len(sub),
                }
                with open(os.path.join(save_dir, "params_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"[SAVE] -> {save_dir} ({len(sub)} keys)", flush=True)
        else:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped = self.accelerator.unwrap_model(self.model)
                sub = {n: p.detach().cpu()
                       for n, p in unwrapped.named_parameters() if p.requires_grad}
                path = os.path.join(save_dir, "params.pt")
                torch.save(sub, path)
                print(f"[SAVE] -> {path} ({len(sub)} keys)", flush=True)

    # ==========================================================
    # Training loop
    # ==========================================================

    def train(self):

        self.model.train()
        self._train_start_time = time.time()
        self._step_times = [self._train_start_time]
        self._epoch_loss_accum = 0.0
        self._epoch_margin_accum = 0.0
        self._epoch_acc_accum = 0.0
        self._epoch_count = 0
        self._micro_step = 0
        self._micro_loss_accum = 0.0
        self._micro_margin_accum = 0.0
        self._micro_acc_accum = 0.0
        self._micro_count = 0

        if self.args.epochs and self.args.epochs > 0:
            for ep in range(1, self.args.epochs + 1):
                self.global_epoch = ep
                self.train_sampler.set_epoch(ep)
                epoch_start = time.time()
                epoch_step_start = self.global_step

                self._epoch_loss_accum = 0.0
                self._epoch_margin_accum = 0.0
                self._epoch_acc_accum = 0.0
                self._epoch_count = 0

                if self.accelerator.is_main_process:
                    print(f"\n[EPOCH] start {ep}/{self.args.epochs}", flush=True)

                for batch in self.loader:
                    self.step(batch)
                    if self._should_stop():
                        break

                if self.accelerator.is_main_process:
                    epoch_elapsed = time.time() - epoch_start
                    epoch_steps = self.global_step - epoch_step_start
                    avg_loss = self._epoch_loss_accum / max(1, self._epoch_count)
                    avg_margin = self._epoch_margin_accum / max(1, self._epoch_count)
                    avg_acc = self._epoch_acc_accum / max(1, self._epoch_count)
                    print(
                        f"\n[EPOCH] finished {ep}/{self.args.epochs} "
                        f"| steps={epoch_steps} "
                        f"| avg_loss={avg_loss:.4f} "
                        f"| avg_margin={avg_margin:.3f} "
                        f"| avg_acc={avg_acc:.3f} "
                        f"| time={str(datetime.timedelta(seconds=int(epoch_elapsed)))}",
                        flush=True,
                    )
                if self._should_stop():
                    break

        # Step-only mode
        if not self.args.epochs or self.args.epochs <= 0:
            if self.accelerator.is_main_process:
                print(f"\n[TRAIN] step-only mode (steps={self.args.steps})", flush=True)
            while not self._should_stop():
                self.train_sampler.set_epoch(self.global_step)
                for batch in self.loader:
                    self.step(batch)
                    if self._should_stop():
                        break

        # Final save
        if self.accelerator.is_main_process:
            total_elapsed = time.time() - self._train_start_time
            avg_loss = self._epoch_loss_accum / max(1, self._epoch_count)
            print(
                f"\n[DONE] steps={self.global_step} "
                f"| avg_loss={avg_loss:.4f} "
                f"| total_time={str(datetime.timedelta(seconds=int(total_elapsed)))}",
                flush=True,
            )

        self.save(self.args.save_dir)

        if self.use_wandb:
            import wandb
            wandb.finish()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GLM-5-FP8 Partial-Layer DPO")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_layers", type=int, default=78)
    parser.add_argument("--train_layers", type=int, default=1)
    parser.add_argument("--train_modules", type=str, default="attn",
                        choices=list(TRAIN_MODULE_PATTERNS.keys()))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw"])
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./output/glm5-lastlayer-dpo")
    parser.add_argument("--seed", type=int, default=42)
    # DPO-specific
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="sigmoid",
                        choices=["sigmoid", "hinge", "ipo"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--length_normalise", action="store_true")
    parser.add_argument("--ref_batch_size", type=int, default=1,
                        help="Ignored (forced to 1 for MoE routing consistency)")
    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    # W&B
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    args = parser.parse_args()

    trainer = DPOTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
