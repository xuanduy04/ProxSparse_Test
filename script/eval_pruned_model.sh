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
DIR="${model_subdir}-lr${lr}_len${ctx_len}_batch${per_device_train_batch_size}_lambda${lambda_}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------- MAIN ---------------- #
echo -e "Evaluating ${DIR}\n(checkpoint: ${checkpoint})"

python "$PROJECT_ROOT/eval/eval_mask.py" \
  --mask "$DIR/checkpoint-$checkpoint" \
  --model "${model_dir}/${model_subdir}" \
  --batch_size "${1:-auto}"
