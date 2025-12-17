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
echo -e "Starting learning mask with ProxSparse"

CUDA_VISIBLE_DEVICES=7 python "$PROJECT_ROOT/end-to-end/main.py" \
  --model "${model_dir}/${model_subdir}" \
  --lambda_value $lambda_ \
  --ctx_len $ctx_len \
  --per_device_train_batch_size $per_device_train_batch_size \
  --learning_rate $lr

echo -e "Finished learning, now extracting binary mask. Mask stored in proximal_* directory"

python "$PROJECT_ROOT/end-to-end/mask_op.py" \
  --model "$DIR" \
  --ckpt "last"
