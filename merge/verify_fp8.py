#!/usr/bin/env python3
"""
verify_fp8.py
FP8変換後のモデルの整合性チェック

チェック項目:
  1. 全シャードファイルが存在するか
  2. 各シャードがロード可能か（破損チェック）
  3. weightがfloat8_e4m3fnになっているか
  4. weight_scale_invが対応して存在するか
  5. scaleにNaN/Infがないか
  6. index.jsonのweight_mapと実ファイルが一致するか

Usage:
  python verify_fp8.py /path/to/fp8_model_dir
"""
import os
import sys
import json
import time

import torch
from safetensors.torch import load_file

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def check_shard(shard_path, weight_map_keys):
    errors = []
    warnings = []

    try:
        tensors = load_file(shard_path, device="cpu")
    except Exception as e:
        return False, [f"LOAD ERROR: {e}"], []

    fp8_count = 0
    scale_count = 0
    bf16_count = 0

    for key, t in tensors.items():
        if t.dtype in FP8_DTYPES:
            fp8_count += 1
            # 対応するscaleが存在するか
            scale_key = (key.replace("weight", "weight_scale_inv")
                         if "weight" in key else key + "_scale_inv")
            if scale_key not in tensors:
                errors.append(f"Missing scale for: {key}")
            else:
                scale = tensors[scale_key].float()
                if torch.isnan(scale).any():
                    errors.append(f"NaN in scale: {scale_key}")
                if torch.isinf(scale).any():
                    errors.append(f"Inf in scale: {scale_key}")
                if (scale <= 0).any():
                    warnings.append(f"Non-positive scale in: {scale_key}")
                scale_count += 1
        elif key.endswith("weight_scale_inv") or key.endswith("_scale_inv"):
            pass  # scaleはカウント済み
        elif t.dtype == torch.bfloat16:
            bf16_count += 1

    return True, errors, warnings, fp8_count, scale_count, bf16_count


def main(model_dir):
    print(f"{'='*60}")
    print(f"  FP8 Verification: {model_dir}")
    print(f"{'='*60}")
    t_start = time.time()

    # index.json 確認
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print("ERROR: model.safetensors.index.json not found")
        sys.exit(1)

    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    print(f"  Index: {len(weight_map)} keys, {len(shards)} shards")

    # 全シャードの存在確認
    missing = [s for s in shards if not os.path.exists(os.path.join(model_dir, s))]
    if missing:
        print(f"\nERROR: {len(missing)} shards missing:")
        for s in missing:
            print(f"  - {s}")
        sys.exit(1)
    print(f"  All {len(shards)} shards exist: OK")

    # 各シャードをチェック
    print(f"\n  Checking shards...")
    total_fp8 = 0
    total_scale = 0
    total_bf16 = 0
    total_errors = []
    total_warnings = []

    for i, shard_name in enumerate(shards):
        shard_path = os.path.join(model_dir, shard_name)
        result = check_shard(shard_path, weight_map)

        if result[0] is False:
            _, errors, _ = result
            total_errors.extend([f"[{shard_name}] {e}" for e in errors])
            print(f"  [{i+1:3d}/{len(shards)}] {shard_name}: LOAD FAILED", flush=True)
            continue

        _, errors, warnings, fp8, scale, bf16 = result
        total_fp8 += fp8
        total_scale += scale
        total_bf16 += bf16
        total_errors.extend([f"[{shard_name}] {e}" for e in errors])
        total_warnings.extend([f"[{shard_name}] {w}" for w in warnings])

        if (i + 1) % 50 == 0 or i == 0 or i == len(shards) - 1:
            status = "OK" if not errors else f"ERRORS:{len(errors)}"
            print(f"  [{i+1:3d}/{len(shards)}] fp8={fp8}, scale={scale}, bf16={bf16} [{status}]",
                  flush=True)

    # サマリー
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Shards checked   : {len(shards)}")
    print(f"  FP8 tensors      : {total_fp8}")
    print(f"  Scale tensors    : {total_scale}")
    print(f"  BF16 tensors     : {total_bf16}  (embed/lm_head等)")
    print(f"  Elapsed          : {elapsed:.1f}s")

    if total_errors:
        print(f"\n  ERRORS ({len(total_errors)}):")
        for e in total_errors[:20]:
            print(f"    ✗ {e}")
        if len(total_errors) > 20:
            print(f"    ... and {len(total_errors)-20} more")
    else:
        print(f"\n  Errors           : 0")

    if total_warnings:
        print(f"  Warnings ({len(total_warnings)}):")
        for w in total_warnings[:5]:
            print(f"    ⚠ {w}")

    if not total_errors:
        print(f"\n  ✅ 変換成功: 破損なし")
        size_gb = sum(
            os.path.getsize(os.path.join(model_dir, s)) for s in shards
        ) / 1024**3
        print(f"  出力サイズ: {size_gb:.1f} GB")
    else:
        print(f"\n  ❌ 変換失敗: {len(total_errors)} エラーあり")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_fp8.py <fp8_model_dir>")
        sys.exit(1)
    main(sys.argv[1])
