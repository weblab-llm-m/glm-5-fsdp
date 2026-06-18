#!/usr/bin/env python3
"""
streaming_bf16_to_fp8.py
safetensorsシャードを1つずつ読んでFP8に変換する。
RAMピーク = 1シャード分のみ
"""
import os
import json
import glob
import gc
import shutil
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open


def quantize_fp8_blockwise(w: torch.Tensor, block_size=(128, 128)):
    if w.dim() < 2:
        return w.to(torch.float8_e4m3fn), torch.ones(1, dtype=torch.float32)
    bh, bw = block_size
    if w.dim() == 2:
        out_f, in_f = w.shape
        num_bh = (out_f + bh - 1) // bh
        num_bw = (in_f + bw - 1) // bw
        pad_h = num_bh * bh - out_f
        pad_w = num_bw * bw - in_f
        w_f = w.float()
        if pad_h > 0 or pad_w > 0:
            w_f = torch.nn.functional.pad(w_f, (0, pad_w, 0, pad_h))
        w_b = w_f.view(num_bh, bh, num_bw, bw)
        scale = w_b.abs().amax(dim=(1, 3)) / 448.0
        scale = scale.clamp(min=1e-12)
        w_q = (w_b / scale.unsqueeze(1).unsqueeze(3)).clamp(-448, 448)
        w_q = w_q.view(num_bh * bh, num_bw * bw)[:out_f, :in_f]
        return w_q.to(torch.float8_e4m3fn), scale
    else:
        # 3D以上: MoEエキスパート (n_exp, out_f, in_f) など
        n_exp, out_f, in_f = w.shape[0], w.shape[-2], w.shape[-1]
        batch = w.shape[:-2]
        num_bh = (out_f + bh - 1) // bh
        num_bw = (in_f + bw - 1) // bw
        pad_h = num_bh * bh - out_f
        pad_w = num_bw * bw - in_f
        w_f = w.reshape(-1, out_f, in_f).float()
        if pad_h > 0 or pad_w > 0:
            w_f = torch.nn.functional.pad(w_f, (0, pad_w, 0, pad_h))
        w_b = w_f.view(n_exp, num_bh, bh, num_bw, bw)
        scale = w_b.abs().amax(dim=(2, 4)) / 448.0  # (n_exp, num_bh, num_bw)
        scale = scale.clamp(min=1e-12)
        w_q = (w_b / scale.unsqueeze(2).unsqueeze(4)).clamp(-448, 448)
        w_q = w_q.view(n_exp, num_bh * bh, num_bw * bw)[:, :out_f, :in_f]
        return w_q.to(torch.float8_e4m3fn), scale


def convert_shard(src_path, dst_path):
    t0 = time.time()
    tensors = load_file(src_path, device="cpu")
    out = {}
    n_fp8 = 0
    SKIP_KEYS = ("embed_tokens", "lm_head", "norm", "layernorm", "bias",
                 "scale_inv", "e_score_correction")
    for key, t in tensors.items():
        # scale テンソルの扱い:
        #   - 対応する weight がこのスクリプトで再量子化される (BF16等) 場合、
        #     scale はここで再生成するため入力の値は破棄する (旧値は不整合の元)。
        #   - 対応する weight が FP8 のまま素通しされる場合 (keep モードの
        #     マージ出力に含まれる凍結層)、既存 scale が必須なので保持する。
        if key.endswith("_scale_inv") or "activation_scale" in key:
            if key.endswith("_scale_inv"):
                base_key = key[: -len("_scale_inv")]
            else:
                base_key = key.rsplit(".", 1)[0] + ".weight"
            base = tensors.get(base_key)
            if base is not None and base.dtype in (torch.float8_e4m3fn,
                                                   torch.float8_e5m2):
                out[key] = t
            continue
        is_weight_like = (
            t.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and t.dim() >= 2
            and not any(s in key for s in SKIP_KEYS)
        )
        if is_weight_like:
            w_fp8, scale = quantize_fp8_blockwise(t)
            out[key] = w_fp8
            scale_key = (key.replace("weight", "weight_scale_inv")
                         if "weight" in key else key + "_scale_inv")
            out[scale_key] = scale.to(torch.float32)
            n_fp8 += 1
        else:
            out[key] = t
    save_file(out, dst_path)
    mb_in = os.path.getsize(src_path) / 1024**2
    mb_out = os.path.getsize(dst_path) / 1024**2
    print(
        f"  fp8={n_fp8} tensors, {mb_in:.0f}MB -> {mb_out:.0f}MB, {time.time()-t0:.1f}s",
        flush=True,
    )
    del tensors, out
    gc.collect()


def main(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"ERROR: {index_path} not found")
        sys.exit(1)
    with open(index_path) as f:
        index = json.load(f)

    shards = sorted(set(index["weight_map"].values()))
    print(f"Shards: {len(shards)}", flush=True)

    weight_map_new = {}
    for i, name in enumerate(shards):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"[{i+1}/{len(shards)}] {name} SKIP (already exists)", flush=True)
        else:
            print(f"[{i+1}/{len(shards)}] {name}", flush=True)
            convert_shard(src, dst)
        # 出力シャードの実キー一覧からweight_map構築
        with safe_open(dst, framework="pt", device="cpu") as f:
            for k in f.keys():
                weight_map_new[k] = name

    total_size = sum(
        os.path.getsize(os.path.join(dst_dir, f))
        for f in os.listdir(dst_dir)
        if f.endswith(".safetensors")
    )
    with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map_new},
            f,
            indent=2,
        )

    for fname in os.listdir(src_dir):
        if fname.endswith((".json", ".py", ".tiktoken", ".model")) and not fname.endswith(
            "safetensors.index.json"
        ) and not os.path.exists(os.path.join(dst_dir, fname)):
            shutil.copy2(
                os.path.join(src_dir, fname), os.path.join(dst_dir, fname)
            )

    print(f"\nDone -> {dst_dir}  ({total_size/1024**3:.1f} GB)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python streaming_bf16_to_fp8.py <src_dir> <dst_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
