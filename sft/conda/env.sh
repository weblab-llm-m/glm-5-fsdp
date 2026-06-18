#!/bin/bash
set -euo pipefail

ENV_NAME="glm5-sft"
ENV_PATH="${CONDA_BASE:-$(conda info --base)}/envs/${ENV_NAME}"

echo "=== GLM-5 SFT 環境構築 ==="
echo "開始: $(date)"

# ============================================================
# 1. 既存環境の削除
# ============================================================
echo "[1/4] 既存環境を削除..."
conda deactivate 2>/dev/null || true
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
# パス指定の場合
if [ -d "${ENV_PATH}" ]; then
    rm -rf "${ENV_PATH}"
    echo "  ${ENV_PATH} を削除しました"
fi

# ============================================================
# 2. 新規環境作成
# ============================================================
echo "[2/4] conda 環境作成 (Python 3.11)..."
conda create -p "${ENV_PATH}" python=3.11 -y

eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"

# ============================================================
# 3. パッケージインストール
# ============================================================
echo "[3/4] パッケージインストール..."

# PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# transformers dev版（GLM-5 の glm_moe_dsa は 5.0.2.dev0 必須）
pip install git+https://github.com/huggingface/transformers.git

# FSDP 分散学習
pip install accelerate

# tokenizer 依存
pip install sentencepiece protobuf

# ロギング（任意）
pip install wandb

# ============================================================
# 4. 確認
# ============================================================
echo "[4/4] 環境確認..."
python -c "
import torch
import transformers
import accelerate

print(f'torch:        {torch.__version__}')
print(f'CUDA:         {torch.version.cuda}')
print(f'GPU count:    {torch.cuda.device_count()}')
print(f'transformers: {transformers.__version__}')
print(f'accelerate:   {accelerate.__version__}')

# GLM-5 config テスト
from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained(
        '/path/to/base_model', trust_remote_code=True)
    print(f'model_type:   {cfg.model_type}')
    print(f'num_layers:   {cfg.num_hidden_layers}')
    print('GLM-5 config: OK')
except Exception as e:
    print(f'GLM-5 config: FAILED ({e})')
"

echo ""
echo "=== 完了: $(date) ==="
echo "使用方法: conda activate ${ENV_PATH}"
