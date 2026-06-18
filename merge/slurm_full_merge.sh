#!/bin/bash
#SBATCH --job-name=merge-glm5-sft
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodelist=node-XX
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail

# リポジトリルート (このスクリプトの1つ上のディレクトリ)。
# sbatch を任意の cwd から実行しても .py を確実に参照できるようにする。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# sbatch はスクリプトをスプールへコピーして実行するため、BASH_SOURCE が
# リポジトリ外を指す場合は提出ディレクトリ(リポジトリルート想定)へフォールバック
if [[ ! -f "${REPO_ROOT}/merge/merge_full_model.py" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
if [[ ! -f "${REPO_ROOT}/merge/merge_full_model.py" ]]; then
  echo "ERROR: merge/merge_full_model.py が見つかりません (REPO_ROOT=${REPO_ROOT})。リポジトリルートから sbatch してください" >&2
  exit 1
fi

echo "=== GLM-5-FP8 SFT Merge ==="
echo "開始時刻: $(date)"
echo "ノード: $(hostname)"

mkdir -p outputs/logs

eval "$(conda shell.bash hook)"
set +u
conda activate glm5-sft
set -u

python "${REPO_ROOT}/merge/merge_full_model.py" \
  --base_model /path/to/base_model \
  --checkpoint_dir ./output/glm5-fp8-lastlayer-attn-sft \
  --out_dir ./output/GLM-5-FP8-lastlayer-attn-sft-merged \
  --num_layers 78 \
  --output_dtype bf16

echo "=== 完了: $(date) ==="
