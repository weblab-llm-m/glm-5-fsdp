#!/bin/bash
#SBATCH --job-name=glm5-attn-dpo
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodelist=node-XX,node-XX,node-XX,node-XX
#SBATCH --nodes=12
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=240
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail

# リポジトリルート (このスクリプトの1つ上のディレクトリ)。
# sbatch を任意の cwd から実行しても .py を確実に参照できるようにする。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# sbatch はスクリプトをスプールへコピーして実行するため、BASH_SOURCE が
# リポジトリ外を指す場合は提出ディレクトリ(リポジトリルート想定)へフォールバック
if [[ ! -f "${REPO_ROOT}/dpo/run_glm5_lastlayer_dpo.py" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
if [[ ! -f "${REPO_ROOT}/dpo/run_glm5_lastlayer_dpo.py" ]]; then
  echo "ERROR: dpo/run_glm5_lastlayer_dpo.py が見つかりません (REPO_ROOT=${REPO_ROOT})。リポジトリルートから sbatch してください" >&2
  exit 1
fi

echo "=== GLM-5-FP8 Last-Layer Attention DPO (FSDP) ==="
echo "開始時刻: $(date)"

mkdir -p outputs/logs

# ============================================================
# ネットワーク設定
# ============================================================
export TARGET_IFNAME="eth0"

module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load cuda/12.4

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="${TARGET_IFNAME}"
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_NCHANNELS=4
export NCCL_NCHANNELS_PER_NET_PEER=1
export NCCL_CUMEM_ENABLE=0

export GLOO_SOCKET_IFNAME="${TARGET_IFNAME}"
export TP_SOCKET_IFNAME="${TARGET_IFNAME}"

export OMP_NUM_THREADS=2
ulimit -n 262144 || true

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200

# ============================================================
# conda 環境
# ============================================================
eval "$(conda shell.bash hook)"
set +u
conda activate glm5-sft
set -u

# ============================================================
# マスターノード IP 取得
# ============================================================
MASTER_HOSTNAME="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)"

HOST_ADDR=$(srun --nodes=1 --ntasks=1 -w "${MASTER_HOSTNAME}" \
    ip -4 addr show "${TARGET_IFNAME}" \
    | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1)

if [ -z "${HOST_ADDR}" ]; then
    echo "WARNING: ${TARGET_IFNAME} のIP取得に失敗。ホスト名を使用します。"
    HOST_ADDR="${MASTER_HOSTNAME}"
fi

export MASTER_ADDR="${HOST_ADDR}"
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "MASTER_ADDR  : ${MASTER_ADDR}"
echo "MASTER_PORT  : ${MASTER_PORT}"
echo "SLURM_NNODES : ${SLURM_NNODES}"
echo "Total GPUs   : $((SLURM_NNODES * 8))"

# ============================================================
# 設定値
# ============================================================
BASE_MODEL="/path/to/base_model"
DATASET="/path/to/train_dpo.jsonl"
OUTPUT_DIR="./output/glm5-fp8-lastlayer-attn-dpo"
SCRIPT="${REPO_ROOT}/dpo/run_glm5_lastlayer_dpo.py"

# バッチ設定:
#   DPO: chosen+rejected concatenated forward → 実質2倍のメモリ
#   micro batch_size=1 × grad_accum=4 × 96 GPUs (12ノード×8) = effective batch 384 pairs
BATCH_SIZE=1
GRAD_ACCUM=4

# DPO ハイパーパラメータ
BETA=0.1
LOSS_TYPE="sigmoid"
LR="5e-6"
LABEL_SMOOTHING=0.0

# W&B 設定 (空文字で無効化)
WANDB_PROJECT="glm5-dpo"
WANDB_RUN_NAME="lastlayer-attn-beta${BETA}"

echo "=== accelerate launch ==="
echo "micro_batch=${BATCH_SIZE}, grad_accum=${GRAD_ACCUM}, GPUs=$((SLURM_NNODES * 8))"
echo "effective_batch=$((BATCH_SIZE * GRAD_ACCUM * SLURM_NNODES * 8)) pairs"
echo "beta=${BETA}, loss_type=${LOSS_TYPE}, lr=${LR}"

# ============================================================
# 実行
# ============================================================
srun -l --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c '
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR='"${MASTER_ADDR}"'
export MASTER_PORT='"${MASTER_PORT}"'

NODE_RANK="${SLURM_NODEID}"
NNODES='"${SLURM_NNODES}"'
NPROC=8

echo "[Node ${NODE_RANK}] hostname=$(hostname), MASTER=${MASTER_ADDR}:${MASTER_PORT}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true

accelerate launch \
  --multi_gpu \
  --num_machines ${NNODES} \
  --num_processes $((NNODES * NPROC)) \
  --machine_rank ${NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --mixed_precision no \
  '"${SCRIPT}"' \
    --model '"${BASE_MODEL}"' \
    --num_layers 78 \
    --train_layers 1 \
    --train_modules attn \
    --dataset '"${DATASET}"' \
    --seq_len 4096 \
    --batch_size '"${BATCH_SIZE}"' \
    --gradient_accumulation_steps '"${GRAD_ACCUM}"' \
    --epochs 3 \
    --steps 0 \
    --lr '"${LR}"' \
    --optim adamw \
    --grad_clip 1.0 \
    --beta '"${BETA}"' \
    --loss_type '"${LOSS_TYPE}"' \
    --label_smoothing '"${LABEL_SMOOTHING}"' \
    --log_every 1 \
    --save_every 50 \
    --wandb_project '"${WANDB_PROJECT}"' \
    --wandb_run_name '"${WANDB_RUN_NAME}"' \
    --save_dir '"${OUTPUT_DIR}"'
'

echo "=== ジョブ完了: $(date) ==="
