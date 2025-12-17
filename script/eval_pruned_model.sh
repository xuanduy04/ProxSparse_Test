#!/usr/bin/env bash
set -e

# ------------ Hyperparams ------------ #
model_dir=Qwen
model_subdir=Qwen2.5-1.5B
lambda_=0.2
batch_size=1
ctx_len=4096
samples=400
lr=1e-04
checkpoint=$samples

# -------------- Constants -------------- #
DIR="${model_subdir}-en_sft_final_${samples}_lr${lr}_len${ctx_len}_batch${batch_size}_lambda${lambda_}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------- MAIN ---------------- #
echo -e "Evaluating ${DIR}\n(checkpoint: ${checkpoint})"

python "$PROJECT_ROOT/eval/eval_mask.py" \
  --mask "$DIR/checkpoint-$checkpoint" \
  --model "${model_dir}/${model_subdir}" \
  --batch_size "auto"
