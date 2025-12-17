#!/usr/bin/env bash
set -e


model_dir=Qwen
model_subdir=Qwen2.5-1.5B
lambda_=0.2
batch_size=1
ctx_len=4096
samples=400
lr=1e-04
checkpoint=$samples

DIR="${model_subdir}-en_sft_final_${samples}_lr${lr}_len${ctx_len}_batch${batch_size}_lambda${lambda_}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"


echo -e "Starting learning mask with ProxSparse"

CUDA_VISIBLE_DEVICES=6 python "$PROJECT_ROOT/end-to-end/main.py" \
  --model "${model_dir}/${model_subdir}" \
  --lambda_value $lambda_ \
  --ctx_len $ctx_len \
  --batch_size $batch_size \
  --samples $samples \
  --learning_rate $lr

echo -e "Finished learning, now extracting binary mask. Mask stored in proximal_* directory"

python "$PROJECT_ROOT/end-to-end/mask_op.py" \
  --model "$DIR/checkpoint-$checkpoint"

# Optional eval
# python "$PROJECT_ROOT/eval/eval_mask_ppl.py" \
#   --mask "$DIR/checkpoint-$checkpoint" \
#   --model "${model_dir}/${model_subdir}" \
#   --method else \
#   --ctx_len 4096
