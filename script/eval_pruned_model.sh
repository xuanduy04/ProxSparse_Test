#!/usr/bin/env bash
set -e

# ------------ Hyperparams ------------ #
model_dir=Qwen
model_subdir=Qwen2.5-1.5B

lr=1e-04
ctx_len=4096
per_device_train_batch_size=16
lambda_=0.2

# -------------- Constants -------------- #
lr_float=$(printf "%g" "$lr")
DIR="${model_subdir}-lr${lr_float}_len${ctx_len}_batch${per_device_train_batch_size}_lambda${lambda_}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------- MAIN ---------------- #
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=7 PYTHONPATH="$PROJECT_ROOT" \
python -m eval.eval_mask \
  --model_name "${model_dir}/${model_subdir}" \
  --mask_dir "$DIR" \
  --ckpt "${1:-last}" \
  --batch_size "${2:-auto}"
