SEED = 1836

import argparse
import os
import random

import numpy as np
import torch

os.environ["TRL_USE_LIGER_KERNEL"] = "False"
os.environ["HF_TRACKIO_DISABLE"] = "1"
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM

from pathlib import Path

import end_to_end.config as config  # configurations for lambdas
from patch_transformers_trainer import prox_inner_training_loop, prox_compute_loss
from prox_linear import replace_prox_linear


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def main():
    seed_everything(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Llama-2-7b",
                        help='Provide the model name for finetuning')
    parser.add_argument('--lambda_value', type=float, default=0.1,
                        help='lambda value')
    parser.add_argument('--lambda2_value', type=float, default=0.1,
                        help='lambda2 value')
    parser.add_argument('--ctx_len', type=int, default=4096,
                        help='ctx length ratio')
    parser.add_argument('--per_device_train_batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--project_lambda2', type=int, default="0",
                        help='projected descent')
    parser.add_argument('--epsilon', type=float, default="0.1",
                        help='lambda2 epsilon')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='lr')

    BASE_DIR = Path(__file__).resolve().parent.parent  # {etc}/ProxSparse_Test
    args = parser.parse_args()
    model_name = args.model
    lr = args.learning_rate
    ctx_len = args.ctx_len
    per_device_train_batch_size = args.per_device_train_batch_size
    assert per_device_train_batch_size <= 256
    assert 256 % per_device_train_batch_size == 0
    epsilon = args.epsilon

    config.lambda_ = args.lambda_value
    config.epsilon = epsilon
    config.lambda2_ = args.lambda2_value
    config.project_lambda2 = args.project_lambda2

    print(f'Loading dataset at `{str(BASE_DIR.parent / "data" / "for_susi")}`... ', end='')
    raw_dataset = load_dataset(
        "parquet",
        data_files={
            "train": str(BASE_DIR.parent / "data" / "for_susi" / "*.parquet")
        },
    )
    dataset = DatasetDict({
        'train': raw_dataset['train'].shuffle(seed=SEED),
        'validation': raw_dataset['train'].select(range(4)),
        'test': raw_dataset['train'].select(range(4))
    })
    del raw_dataset, dataset["test"]
    print('Done')

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # float 32
        device_map="auto",
    )

    if config.project_lambda2 == 1:
        print("# begin replacing layers")
        replace_prox_linear(model)
        print("# replacing sucessful!")
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    BASE_MODEL = model_name

    if config.project_lambda2 == 1:
        raise NotImplementedError
    else:
        repository_id = f"{BASE_MODEL.split('/')[1]}-lr{lr}_len{ctx_len}_batch{per_device_train_batch_size}_lambda{config.lambda_}"
        # DIR="${model_subdir}-lr${lr}_len${ctx_len}_batch${per_device_train_batch_size}_lambda${lambda_}"


    sft_config = SFTConfig(
        dataset_text_field="text",
        output_dir=repository_id,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=256 // per_device_train_batch_size,
        max_length=ctx_len,
        learning_rate=lr,
        num_train_epochs=1,
        optim="adamw_torch_fused",
        warmup_ratio=0.1,
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=0.1,
        logging_first_step=True,
        eval_strategy="no",
        # eval_steps=0.1,
        # eval_accumulation_steps=2,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=None,
        # load_best_model_at_end=True,
        save_only_model=True,
        report_to="none",
    )

    # patching with Proxsparse operator
    print("Patching inner training loop and loss computation")
    SFTTrainer._inner_training_loop = prox_inner_training_loop
    SFTTrainer.compute_loss = prox_compute_loss

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=sft_config,
    )

    trainer.train()
    print("Finished!")


if __name__ == "__main__":
    main()
