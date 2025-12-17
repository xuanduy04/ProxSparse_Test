SEED = 1836

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from transformers import AutoModelForCausalLM


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def explicit_sparsify_magnitude_row(weight, n, m):  # prune and leave n out of m in 2d array
    dim = weight.shape
    weight = weight.view(-1, m)  # for a 2d array, reshale into 4 blocks
    w_mask = torch.zeros_like(weight)
    w_mask.scatter_(1, torch.topk(torch.abs(weight), n, dim=1, largest=True)[1], True)
    w_mask = w_mask.view(dim[0], dim[1])
    return w_mask


def reshape_weights(weight_matrix):
    m, n = weight_matrix.shape
    weight_matrix = weight_matrix.view(-1, 4)
    return weight_matrix, m, n


def main():
    seed_everything(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="meta-llama/Meta-Llama-3-8B")

    def ckpt_type(value):
        if value in {"all", "last"}:
            return value
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "ckpt must be an integer, 'all', or 'last'"
            )
    parser.add_argument("--ckpt", type=ckpt_type, default="last",
        help="Which checkpoint to load (int, 'all', or 'last')")


    args = parser.parse_args()
    model_dir = args.model_dir
    ckpt = args.ckpt

    checkpoints = sorted(
        (
            p.name for p in Path(model_dir).iterdir()
            if p.is_dir() and p.name.startswith("checkpoint-")
        ),
        key=lambda name: int(name.split("-", 1)[1])
    )
    if not checkpoints:
        raise ValueError("No checkpoint directories found.")

    if ckpt == "all":
        selected_checkpoints = checkpoints
    elif ckpt == "last":
        selected_checkpoints = [checkpoints[-1]]
    else:
        try:
            selected_checkpoints = [
                next(
                    name for name in checkpoints
                    if int(name.split("-", 1)[1]) == ckpt
                )
            ]
        except StopIteration:
            raise ValueError(f"Checkpoint {ckpt} not found.")

    for checkpoint in selected_checkpoints:
        model_name = model_dir + "/" + checkpoint
        print(f"Getting the mask of checkpoint {checkpoint}")
        get_mask(model_name)
        print("Done")


def get_mask(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    one_sparse = 0
    two_sparse = 0
    three_sparse = 0
    dense = 0
    total = 0

    for n, m in model.named_parameters():
        if "bias" not in n and (
                "k_proj" in n or "q_proj" in n or "v_proj" in n or "o_proj" in n or "up_proj" in n or "down_proj" in n or "gate_proj" in n or "out_proj" in n or "fc1" in n or "fc2" in n):
            mm, _, __ = reshape_weights(m)
            total += mm.shape[0]
            zero_counts = (mm == 0).sum(dim=1)
            num_zeros_0 = (zero_counts == 0).sum().item()
            num_zeros_1 = (zero_counts == 1).sum().item()
            num_zeros_2 = (zero_counts == 2).sum().item()
            num_zeros_3 = (zero_counts == 3).sum().item()

            dense += num_zeros_0
            one_sparse += num_zeros_1
            two_sparse += num_zeros_2
            three_sparse += num_zeros_3

            mask = explicit_sparsify_magnitude_row(m, 2, 4)
            if not os.path.isdir(f"./proximal_{model_name.split('/')[0]}"):
                os.mkdir(f"./proximal_{model_name.split('/')[0]}")
            if not os.path.isdir(f"./proximal_{model_name}"):
                os.mkdir(f"./proximal_{model_name}")
            torch.save(mask, f"./proximal_{model_name}/{n}.pt")

    print("one sparse ratio: ", one_sparse / total)
    print("two sparse ratio: ", two_sparse / total)
    print("three sparse ratio: ", three_sparse / total)
    print("dense ratio: ", dense / total)


if __name__ == "__main__":
    main()
