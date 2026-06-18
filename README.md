# GLM-5-FSDP

A partial-layer fine-tuning framework for **GLM-5-FP8** using FSDP (Fully Sharded Data Parallel). Provides two training pipelines (SFT / DPO) plus FSDP checkpoint merging and BF16 → FP8 quantization for vLLM inference. Designed for SLURM + H100 multi-node clusters.

## Model Overview

| Item | Value |
|---|---|
| Model | GLM-5 (FP8 variant) |
| Architecture | `glm_moe_dsa` (`GlmMoeDsaForCausalLM`) |
| Layers | 78 |
| Precision | Native FP8 (on-the-fly BF16 dequantization during training) |

## Features

- **Partial-layer fine-tuning** — Train only the last N layers while freezing the rest
- **Module selection** — Choose attention / MLP / full modules per layer
- **LoRA support** — Parameter-efficient training (~100× parameter reduction); base FP8 weights stay frozen
- **FP8 weight handling** — On-the-fly block-wise (128×128) FP8 → BF16 dequantization via forward hooks
- **MoE-aware DPO** — Concatenated chosen+rejected forward for routing consistency + checkpointed cross-entropy
- **Detach hooks** — Cut the autograd graph at the frozen→trainable boundary to slash VRAM
- **Sequence packing** — Pack short samples into `seq_len` to maximize GPU utilization (SFT)
- **VLM-aware merge** — Auto-completes visual encoder weights, config, tokenizer for VLM base models
- **BF16 → FP8 quantization** — Streaming block-wise (128×128) FP8 conversion + integrity verification
- **W&B integration** — Optional Weights & Biases experiment tracking

## Directory Structure

```
.
├── sft/                                       # SFT (Supervised Fine-Tuning)
│   ├── run_lastlayer_attn_sft.py
│   ├── slurm_glm5_lastlayer_attn_sft.sh
│   └── conda/
│       └── env.sh                             # SFT/DPO/Merge environment setup
├── dpo/                                       # DPO (Direct Preference Optimization)
│   ├── run_glm5_lastlayer_dpo.py
│   └── slurm_glm5_lastlayer_attn_dpo.sh
└── merge/                                     # Checkpoint Merge & Quantization
    ├── merge_full_model.py                    # FSDP merge (Full / LoRA → vLLM-ready)
    ├── slurm_full_merge.sh                    # SLURM job script
    ├── streaming_bf16_to_fp8.py               # BF16 → FP8 blockwise quantization
    └── verify_fp8.py                          # FP8 integrity checker
```

## Prerequisites

- **GPU**: Multi-node NVIDIA H100 cluster (8 GPUs per node; 10–12 nodes for training)
- **CUDA**: 12.4 driver
- **Conda**: Python 3.11
- **SLURM**: Job scheduler
- **NCCL**: 2.22.3
- **Network**: High-speed interconnect (InfiniBand / RoCE) for NCCL
- **CPU memory**: ~760 GB on rank 0 for model loading

> **Note**: GLM-5-FP8 requires approximately 760 GB of CPU memory on rank 0 just to load the model. Single-node execution is not expected; multi-node SLURM is assumed.

## Setup

### 1. Create the Conda Environment

```bash
bash sft/conda/env.sh
conda activate glm5-sft
```

The script automatically performs:

1. Delete and recreate the `glm5-sft` environment (Python 3.11)
2. Install PyTorch + torchvision + torchaudio (CUDA 12.4 wheels)
3. Install transformers **dev version** from GitHub (the `glm_moe_dsa` architecture requires `5.0.2.dev0`)
4. Install `accelerate` (FSDP distributed training)
5. Install `sentencepiece`, `protobuf` (tokenizer dependencies)
6. Install `wandb` (optional logging)
7. Verify torch / CUDA / transformers / accelerate and the GLM-5 config

The same environment is shared by SFT, DPO, and Merge.

### 2. Prepare the Data

**SFT** — ChatML JSONL. Each line has a `messages` array; the first `user` message and first `assistant` message are used as prompt/reference (the prompt part is masked with `-100` in the labels):

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The `content` field is normally a text string, but nested dictionaries/arrays (multimodal-style content) are also flattened to text.

**DPO** — prompt / chosen / rejected JSONL:

```jsonl
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

### 3. Configure Paths

Edit each `.sh` file to match your environment:

```bash
BASE_MODEL="/path/to/base_model"
DATASET="/path/to/dataset.jsonl"
```

Also adjust the SLURM settings (`--nodelist`, `--partition`) and the network interface (`TARGET_IFNAME`, default `eth0`).

---

## 1. Supervised Fine-Tuning (SFT)

Fine-tunes attention / MLP / full modules of the last N layers using ChatML JSONL data. Supports LoRA, sequence packing, length-bucketed batching, assistant-turn label masking, and detach hooks for memory efficiency.

- **Node config**: 10 nodes × 8 GPUs = 80 GPUs (FSDP FULL_SHARD)
- **Optimizer**: AdamW (betas 0.9/0.98, `foreach=False`), random seed fixed to 42

### Launch

```bash
sbatch sft/slurm_glm5_lastlayer_attn_sft.sh
```

### Basic Usage

```bash
accelerate launch \
    --multi_gpu \
    --num_machines 10 \
    --num_processes 80 \
    --mixed_precision no \
    sft/run_lastlayer_attn_sft.py \
    --model /path/to/GLM-5-FP8 \
    --dataset /path/to/train_sft.jsonl \
    --num_layers 78 \
    --train_layers 1 \
    --train_modules attn \
    --seq_len 4096 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --packing \
    --epochs 1 \
    --lr 2e-5 \
    --save_dir ./output/glm5-fp8-lastlayer-attn-sft \
    --wandb_project glm5-sft
```

Note that `--mixed_precision no` is intentional: precision is managed manually via the FSDP `MixedPrecision` policy (`param_dtype=None` so FP8 stays FP8 and BF16 stays BF16, `reduce_dtype=bf16`), because `mixed_precision="bf16"` would make FSDP cast FP8 → BF16 without scales and produce NaN.

### Full Argument Reference

#### Model & Layer Selection

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | **required** | Path to base model (GLM-5-FP8) |
| `--num_layers` | int | 78 | Total number of model layers |
| `--train_layers` | int | 1 | Number of layers from the end to train (e.g., 1=layer 77, 2=layers 76–77) |
| `--train_modules` | str | attn | Modules to train: `attn` (self_attn + input_layernorm), `mlp` (mlp/moe + post_attention_layernorm), `full` (entire layer) |

#### LoRA

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_lora` | flag | False | Use LoRA instead of full fine-tuning (base FP8 weights stay frozen) |
| `--lora_rank` | int | 16 | LoRA rank (r) |
| `--lora_alpha` | float | 32 | LoRA alpha (scaling = alpha / rank) |
| `--lora_dropout` | float | 0.0 | LoRA dropout rate |

#### Data & Batching

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | **required** | Path to ChatML JSONL training data |
| `--seq_len` | int | 4096 | Maximum sequence length |
| `--batch_size` | int | 1 | Micro-batch size per GPU |
| `--gradient_accumulation_steps` | int | 1 | Number of micro-batches before each optimizer step |
| `--packing` | flag | False | Pack multiple short samples into `seq_len` to maximize GPU utilization |
| `--bucket_buffer` | int | 256 | Buffer size for length-bucketed batching (used when packing is off) |
| `--max_batch_len` | int | 0 | Max padded length per batch in bucketed mode (0 = no cap) |

#### Training Control

| Argument | Type | Default | Description |
|---|---|---|---|
| `--epochs` | int | 1 | Number of epochs (epoch mode when > 0) |
| `--steps` | int | 0 | Maximum optimizer steps (acts as cap; 0 = no cap; either epochs or steps must be > 0) |
| `--lr` | float | 2e-5 | Learning rate |
| `--weight_decay` | float | 0.0 | Weight decay |
| `--grad_clip` | float | 1.0 | Gradient clipping threshold (0 to disable) |
| `--optim` | str | adamw | Optimizer (only `adamw` is available) |

#### Save & Logging

| Argument | Type | Default | Description |
|---|---|---|---|
| `--save_dir` | str | output/glm5_lastlayer_attn_sft | Checkpoint save directory |
| `--log_every` | int | 1 | Log output interval (optimizer steps) |
| `--wandb_project` | str | "" | W&B project name (empty = wandb disabled) |
| `--wandb_run_name` | str | "" | W&B run name (empty = auto-generated) |
| `--wandb_entity` | str | "" | W&B entity (team/user; empty = default) |

### Processing Flow

1. **Layer class detection** — instantiate the model on the meta device to find the decoder layer class for the FSDP `transformer_auto_wrap_policy`
2. **Model loading** — rank 0 loads real FP8 weights to CPU (~760 GB); all other ranks build a meta-device skeleton (zero CPU RAM)
3. **Dtype preparation**:
   - `weight_scale_inv` / `activation_scale` Parameters → buffers (kept out of FSDP flat params)
   - Trainable layers dequantized FP8 → BF16 in full (FSDP requires uniform dtype per unit), or LoRA A/B injected as BF16 with base FP8 frozen
   - Frozen non-FP8 Parameters → buffers (uniform FP8 dtype per frozen FSDP unit)
4. **Trainable marking** — `requires_grad=True` only on the selected modules of the last N layers (or on `lora_A`/`lora_B` in LoRA mode)
5. **FSDP wrap** — FULL_SHARD, `use_orig_params=True`, `sync_module_states=True` broadcasts rank 0 weights to all ranks; a monkey patch makes FSDP flat-param padding FP8-safe
6. **Hook installation** — on-the-fly FP8 dequant pre/post hooks on FP8 submodules; a detach + `requires_grad_(True)` pre-hook on the first trainable layer; MLP-output detach hooks in trainable layers
7. **Training loop** — micro-batch counts are all-reduced (MIN) across ranks each epoch to prevent FSDP hangs from rank-imbalanced iterations; `accelerator.accumulate()` handles gradient accumulation with `sync_each_batch=True`
8. **Save** — FSDP `SHARDED_STATE_DICT` with `offload_to_cpu=True`; only trainable parameters are written as `attn_params.rank{rank}.pt` plus an `attn_params_meta.json` describing format, world size, trained layers/modules, and LoRA settings

---

## 2. Direct Preference Optimization (DPO)

Optimizes attention / MLP / full modules of the last N layers via DPO using {prompt, chosen, rejected} JSONL. Shares the conda environment and the model loading / FSDP / FP8 machinery with SFT, and adds several MoE-specific DPO fixes:

- **Reference log-prob precompute** — reference log-probs are computed distributed across ranks before training (no second model in memory)
- **Concat forward** — chosen and rejected are padded to the same length and concatenated into one batch, exactly matching the reference forward so MoE expert routing is identical (loss ≈ log 2 at step 1)
- **Checkpointed cross-entropy** — `torch.utils.checkpoint` on `F.cross_entropy` avoids saving the `(B·T, V)` float32 softmax tensor (~2.3 GB per forward)
- **Single backward** — one concatenated forward + one backward keeps peak memory equal to SFT at batch=2

- **Node config**: 12 nodes × 8 GPUs = 96 GPUs (FSDP FULL_SHARD)
- **Optimizer**: AdamW (betas 0.9/0.98, `foreach=False`)

### Launch

```bash
sbatch dpo/slurm_glm5_lastlayer_attn_dpo.sh
```

### Basic Usage

```bash
accelerate launch \
    --multi_gpu \
    --num_machines 12 \
    --num_processes 96 \
    --mixed_precision no \
    dpo/run_glm5_lastlayer_dpo.py \
    --model /path/to/GLM-5-FP8 \
    --dataset /path/to/train_dpo.jsonl \
    --num_layers 78 \
    --train_layers 1 \
    --train_modules attn \
    --seq_len 4096 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --epochs 3 \
    --lr 5e-6 \
    --beta 0.1 \
    --loss_type sigmoid \
    --save_every 50 \
    --save_dir ./output/glm5-fp8-lastlayer-attn-dpo \
    --wandb_project glm5-dpo
```

### Full Argument Reference

#### Model & Layer Selection

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | **required** | Path to base model (GLM-5-FP8) |
| `--num_layers` | int | 78 | Total number of model layers |
| `--train_layers` | int | 1 | Number of layers from the end to train |
| `--train_modules` | str | attn | Modules to train (`attn` / `mlp` / `full`) |

#### Data & Batching

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | **required** | Path to DPO JSONL (prompt / chosen / rejected) |
| `--seq_len` | int | 4096 | Maximum sequence length |
| `--batch_size` | int | 1 | Micro-batch size per GPU (in pairs; concat forward doubles the effective batch) |
| `--gradient_accumulation_steps` | int | 4 | Gradient accumulation steps |

#### DPO Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--beta` | float | 0.1 | DPO temperature β (controls deviation from the reference policy) |
| `--loss_type` | str | sigmoid | DPO loss variant (`sigmoid` / `hinge` / `ipo`) |
| `--label_smoothing` | float | 0.0 | Label smoothing factor (sigmoid loss only) |
| `--length_normalise` | flag | False | Normalize log-probabilities by sequence length |
| `--ref_batch_size` | int | 1 | **Ignored** — internally forced to 1 so the reference forward exactly matches training (MoE routing depends on batch composition) |

#### Training Control

| Argument | Type | Default | Description |
|---|---|---|---|
| `--epochs` | int | 1 | Number of epochs (step-only mode when <= 0) |
| `--steps` | int | 0 | Maximum optimizer steps (0 = no cap) |
| `--lr` | float | 5e-6 | Learning rate |
| `--weight_decay` | float | 0.01 | Weight decay |
| `--grad_clip` | float | 1.0 | Gradient clipping threshold (0 to disable) |
| `--optim` | str | adamw | Optimizer (only `adamw` is available) |
| `--seed` | int | 42 | Random seed (also used by the DistributedSampler) |

#### LoRA

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_lora` | flag | False | Use LoRA instead of full fine-tuning |
| `--lora_rank` | int | 16 | LoRA rank (r) |
| `--lora_alpha` | int | 32 | LoRA alpha (scaling = alpha / rank) |
| `--lora_dropout` | float | 0.0 | LoRA dropout rate |

#### Save & Logging

| Argument | Type | Default | Description |
|---|---|---|---|
| `--save_dir` | str | ./output/glm5-lastlayer-dpo | Checkpoint save directory |
| `--save_every` | int | 0 | Save an intermediate checkpoint (`step-N/` subdirectory) every N optimizer steps (0 = save only at end) |
| `--log_every` | int | 1 | Log output interval (optimizer steps; logs loss, reward margin, accuracy) |
| `--wandb_project` | str | "" | W&B project name (empty = wandb disabled) |
| `--wandb_run_name` | str | "" | W&B run name |
| `--wandb_entity` | str | "" | W&B entity |

### Processing Flow

1. **Model loading & FSDP setup** — identical to SFT (rank 0 CPU load / meta ranks, scale-to-buffer conversion, trainable-layer dequant or LoRA injection, FSDP FULL_SHARD wrap, FP8 dequant hooks, detach hooks). MLP-output detach hooks are installed only when `--train_modules != full`
2. **Dataset tokenization** — each pair is tokenized as (prompt+chosen) and (prompt+rejected) with prompt masking; pairs with no trainable tokens are dropped
3. **Reference log-prob precompute** — indices are sharded across ranks; each pair is forwarded once (concat batch=2, eval mode, no grad), per-sequence log-probs computed via checkpointed CE, then results are all-gathered
4. **Training step** — model is kept in `eval()` mode (matching the reference computation; gradients still flow), chosen+rejected concatenated into one forward, DPO loss `-log σ(β·(Δ_chosen − Δ_rejected))`, single backward, clip + step on accumulation boundaries
5. **Save** — FSDP `SHARDED_STATE_DICT`, trainable parameters only, written as `params.rank{rank}.pt` + `params_meta.json` (includes β, loss type, LoRA settings)

---

## 3. Model Merge

`merge/merge_full_model.py` merges FSDP sharded checkpoints into a `save_pretrained`-compatible model directory for vLLM inference. Supports both full fine-tune and LoRA checkpoints, with automatic format detection and automatic VLM completion (visual encoder weights, config, tokenizer).

- **Supported base models**: GLM-5-FP8 (FP8 base, partial BF16 trained layers), Qwen3.5-397B-A17B (BF16 native, VLM architecture), and other FSDP sharded checkpoints with the same pattern
- **Runs on**: 1 node, CPU processing (the SLURM script reserves 1 GPU and 64 CPUs, but the merge itself runs on CPU)
- **Auto-detection**: rank files (`params.rank*.pt` / `attn_params.rank*.pt` / `*rank*.pt`), meta files (`params_meta.json` / `attn_params_meta.json`), and LoRA mode (from meta `use_lora` or `lora_A`/`lora_B` param names)

### Launch

```bash
sbatch merge/slurm_full_merge.sh
```

### Basic Usage

```bash
# GLM-5-FP8 (78 layers) — streaming BF16 output
python merge/merge_full_model.py \
    --base_model /path/to/GLM-5-FP8 \
    --checkpoint_dir ./output/glm5-fp8-lastlayer-attn-sft \
    --out_dir ./output/GLM-5-FP8-lastlayer-attn-sft-merged \
    --num_layers 78 \
    --output_dtype bf16

# Qwen3.5 (60 layers, VLM) — keep original dtypes
python merge/merge_full_model.py \
    --base_model /path/to/Qwen3.5-397B-A17B \
    --checkpoint_dir ./output/qwen35-lastlayer-attn-sft \
    --out_dir ./output/merged \
    --num_layers 60 \
    --output_dtype keep
```

The merged model can be served directly:

```bash
vllm serve ./output/merged --trust-remote-code
```

### Full Argument Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--base_model` | str | **required** | HF local path to the base model |
| `--checkpoint_dir` | str | **required** | FSDP sharded checkpoint directory (rank files + meta JSON) |
| `--out_dir` | str | **required** | Output directory |
| `--num_layers` | int | 78 | Total number of model layers |
| `--world_size` | int | 0 | Number of FSDP ranks (0 = auto-detect from meta JSON / file glob) |
| `--output_dtype` | str | keep | Output dtype: `bf16` (streaming layer-by-layer BF16 save with FP8 dequant) / `keep` (standard `save_pretrained` keeping original dtypes) |
| `--max_shard_size` | str | 5GB | Maximum shard size for `save_pretrained` (`keep` mode only) |
| `--strict` | flag | True | Abort if sharded tensors have dim-0 gaps or are incomplete |
| `--force_lora_rank` | int | 0 | Override LoRA rank from meta (0 = use meta value, fallback 16) |
| `--force_lora_alpha` | float | 0 | Override LoRA alpha from meta (0 = use meta value, fallback 32) |
| `--skip_vlm_fix` | flag | False | Skip VLM post-processing (config / visual encoder copy) |

### Processing Flow

1. **Phase 1 — Streaming shard merge**: two-pass over `rank*.pt` files (Pass 1 scans `ShardedTensor` metadata and allocates full tensors; Pass 2 streams shard data into place), using an unpickle shim so `ShardedTensor` objects can be read without a process group. Strict mode validates dim-0 contiguity
2. **Phase 2 — Base model load**: full base model loaded to CPU with `low_cpu_mem_usage=True` (the `_finalize_model_loading` step is bypassed because FP8 weights do not support `normal_()` init)
3. **Phase 3 — Apply checkpoint**: trained layers are dequantized FP8 → BF16 in place, then either checkpoint weights replace base weights (full fine-tune) or `W += B @ A × (alpha/rank)` (LoRA merge). Layer lookup tries multiple attribute chains for VLM nesting (`model.language_model.layers` etc.)
4. **Phase 4 — Save**: `bf16` mode streams one safetensors shard per layer (dequantizing remaining FP8, evicting each layer to meta afterwards — peak RAM ≈ FP8 base + 1 BF16 layer) and writes `model.safetensors.index.json`; `keep` mode uses `save_pretrained`
5. **Phase 5 — VLM completion & verification**: copies all non-weight files (config, tokenizer, processor, chat template, custom `.py`) from the base while protecting the merged index; for VLMs, copies missing `visual.*` / `vision_tower.*` / `multi_modal_projector.*` shards from the base and updates the index; finally verifies `model_type` consistency, tokenizer presence, shard existence, and visual-weight presence

---

## 4. FP8 Quantization

### 4.1 streaming_bf16_to_fp8.py

Converts a merged BF16 model to FP8 (block-wise, 128×128, `float8_e4m3fn`) for vLLM FP8 inference. Shards are processed one at a time, so peak RAM is one shard only.

```bash
python merge/streaming_bf16_to_fp8.py <src_dir> <dst_dir>
```

| Argument | Type | Description |
|---|---|---|
| `src_dir` | positional | BF16 model directory (must contain `model.safetensors.index.json`) |
| `dst_dir` | positional | Output FP8 model directory (created if missing) |

**Quantization details:**

- Per 128×128 block: `scale = absmax / 448` (clamped to min 1e-12), values clamped to ±448, stored as `float8_e4m3fn` plus a float32 `*_scale_inv` tensor (`weight` → `weight_scale_inv`)
- 3D+ tensors (MoE expert stacks `(n_exp, out, in)`) are quantized per expert with a `(n_exp, num_bh, num_bw)` scale
- Only floating-point tensors with 2+ dims are quantized; keys containing `embed_tokens`, `lm_head`, `norm`, `layernorm`, `bias`, `scale_inv`, or `e_score_correction` are kept as-is

**Processing Flow:**

1. Read the source `model.safetensors.index.json` and enumerate shards
2. Convert each shard in order; already-existing non-empty output shards are **skipped** (resume support)
3. Rebuild `model.safetensors.index.json` from the actual keys of the output shards
4. Copy auxiliary files (`.json`, `.py`, `.tiktoken`, `.model`) from the source directory

### 4.2 verify_fp8.py

Verifies the integrity of the converted FP8 model. Exits with a non-zero code on errors (usable in CI / job pipelines).

```bash
python merge/verify_fp8.py <fp8_model_dir>
```

| Argument | Type | Description |
|---|---|---|
| `fp8_model_dir` | positional | FP8 model directory to verify |

**Checks performed:**

1. All shards referenced by `model.safetensors.index.json` exist
2. Every shard loads without corruption
3. Weights are `float8_e4m3fn` / `float8_e5m2`
4. Every FP8 tensor has a corresponding `*_scale_inv` tensor
5. Scales contain no NaN / Inf (non-positive scales are reported as warnings)
6. Summary of FP8 / scale / BF16 tensor counts and total size

---

## 5. SLURM Configuration

### Environment Variables

Settings used by both training job scripts (`module load nccl/2.22.3 hpcx/2.18.1-gcc-cuda12/hpcx-mt cuda/12.4`):

```bash
# Network interface used for NCCL / gloo (adjust to your cluster)
export TARGET_IFNAME="eth0"

# NCCL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL                     # Use NVLink for P2P
export NCCL_IB_DISABLE=0                      # Enable InfiniBand
export NCCL_SOCKET_IFNAME="${TARGET_IFNAME}"
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_NCHANNELS=4
export NCCL_NCHANNELS_PER_NET_PEER=1
export NCCL_CUMEM_ENABLE=0
export GLOO_SOCKET_IFNAME="${TARGET_IFNAME}"
export TP_SOCKET_IFNAME="${TARGET_IFNAME}"

# Timeouts (rank 0 model loading takes a long time)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200

# PyTorch
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=2
ulimit -n 262144
```

The training scripts also extend the in-process NCCL timeout to 7200 s via `InitProcessGroupKwargs`.

### Node Configuration

| Pipeline | Nodes | GPUs/Node | Total GPUs | Notes |
|---|---|---|---|---|
| SFT | 10 | 8 | 80 | accelerate launch, FSDP FULL_SHARD, 240 CPUs/task |
| DPO | 12 | 8 | 96 | accelerate launch, FSDP FULL_SHARD, 240 CPUs/task |
| Merge | 1 | 1 | - | CPU processing, 64 CPUs, `--mem=0` (all node memory) |
| FP8 quantization / verify | 1 | - | - | Plain Python, CPU only, streaming per shard |

### Accelerate Launch Pattern

Both SLURM training scripts resolve the master node IP from `TARGET_IFNAME`, pick a free port, and run one launcher per node:

```bash
accelerate launch \
    --multi_gpu \
    --num_machines ${SLURM_NNODES} \
    --num_processes $((SLURM_NNODES * 8)) \
    --machine_rank ${SLURM_NODEID} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --mixed_precision no \
    script.py [args...]
```

Effective batch size = `batch_size × gradient_accumulation_steps × total_GPUs` (for DPO, in preference pairs).

---

## Memory Efficiency Techniques

This framework employs several techniques to minimize GPU and CPU memory:

- **Rank 0-only model loading** — Only rank 0 loads real weights to CPU; other ranks use the meta device. FSDP broadcasts via `sync_module_states=True`
- **FSDP FULL_SHARD** — Shards parameters, gradients, and optimizer states (ZeRO Stage 3 equivalent)
- **Detach hooks** — Cut the autograd graph at the frozen→trainable boundary; frozen layers forward without grad, so no activations are stored and no backward runs (lower VRAM than gradient checkpointing, which would re-all-gather every layer during backward)
- **MLP output detach hooks** — Detach MLP/MoE outputs in trainable layers to prevent backward from all-gathering the massive MoE parameters (~18 GiB per layer); gradients flow through the residual identity path
- **Checkpointed cross-entropy (DPO)** — `torch.utils.checkpoint` on `F.cross_entropy` avoids saving the `(B·T, V)` float32 softmax tensor (~2.3 GB per forward)
- **Frozen param → buffer conversion** — Non-FP8 params in frozen layers converted to buffers so each frozen FSDP flat-param group has a uniform (FP8) dtype
- **Scale → buffer conversion** — `weight_scale_inv` / `activation_scale` kept out of FSDP flat params while remaining accessible to the dequant hooks
- **`sync_each_batch=True` gradient accumulation** — Avoids the +16 GB full-gradient `no_sync` buffer during accumulation
- **`limit_all_gathers=True`** — FSDP limits concurrent all-gathers to reduce peak memory
- **`offload_to_cpu`** — Checkpoint saving uses CPU offload to avoid GPU OOM
- **Streaming merge / save / quantization** — Two-pass shard merging, layer-by-layer safetensors writing with meta-device eviction, and per-shard FP8 conversion keep CPU RAM bounded

## Configuration Examples

**LoRA fine-tuning:**

```bash
--use_lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.1
```

**Full-module training on the last 2 layers:**

```bash
--train_layers 2 --train_modules full
```

**MLP-only training:**

```bash
--train_layers 1 --train_modules mlp
```

## End-to-End Pipeline

```
sbatch sft/slurm_glm5_lastlayer_attn_sft.sh            # or dpo/slurm_glm5_lastlayer_attn_dpo.sh
sbatch merge/slurm_full_merge.sh                        # FSDP shards → BF16 model
python merge/streaming_bf16_to_fp8.py merged/ fp8/      # BF16 → FP8
python merge/verify_fp8.py fp8/                         # integrity check
vllm serve fp8/ --trust-remote-code
```

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  GLM-5-FP8 Base Model (78 layers, native FP8)                 │
│  rank 0: CPU load (~760 GB) / rank != 0: meta device          │
└──────────────────────────┬────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────┐
│  Layer / Module Selection                                     │
│  Last N layers × {attn | mlp | full}, optionally LoRA         │
│  Trainable layers: FP8 → BF16 dequant (or LoRA A/B in BF16)   │
└──────────────────────────┬────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────┐
│  FSDP FULL_SHARD (sync_module_states, use_orig_params)        │
│  + On-the-fly FP8 dequant hooks (frozen layers)               │
│  + Detach hooks at frozen→trainable boundary                  │
├───────────────────────────┬───────────────────────────────────┤
│  SFT (80 GPUs)            │  DPO (96 GPUs)                    │
│  ChatML JSONL, packing    │  {prompt,chosen,rejected} JSONL   │
│  assistant-turn masking   │  ref logp precompute +            │
│                           │  concat forward + ckpt CE         │
└─────────────┬─────────────┴─────────────┬─────────────────────┘
              ▼                           ▼
┌───────────────────────────────────────────────────────────────┐
│  Sharded Checkpoint (trainable params only)                   │
│  attn_params.rank*.pt / params.rank*.pt + *_meta.json         │
└──────────────────────────┬────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────┐
│  merge_full_model.py (CPU)                                    │
│  Phase 1-5: streaming merge → base load → apply (Full/LoRA)   │
│  → streaming BF16 save → VLM completion + verification        │
└──────────────────────────┬────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────┐
│  streaming_bf16_to_fp8.py → verify_fp8.py                     │
│  Block-wise (128×128) FP8 e4m3fn + weight_scale_inv           │
└──────────────────────────┬────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────┐
│  vllm serve <model_dir> --trust-remote-code                   │
└───────────────────────────────────────────────────────────────┘
```

## System Requirements

- CUDA 12.4 driver
- conda (Python 3.11)
- SLURM job scheduler
- Multi-node H100 cluster (8+ nodes × 8 GPUs)
- Sufficient CPU memory on rank 0 (~760 GB for model loading)
- High-speed interconnect (InfiniBand / RoCE) for NCCL

## License

[MIT](LICENSE)

## Acknowledgements

This program is based on results obtained from a project, JPNP25006, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
