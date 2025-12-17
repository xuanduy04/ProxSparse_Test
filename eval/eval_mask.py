SEED = 1836

import argparse
import random
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.ckpt_loading import *

try:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
except ImportError:
    subprocess.run(["pip", "install", "lm_eval"], check=True)
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def explicit_sparsify_magnitude_row(weight, n, m):
    dim = weight.shape
    weight = weight.view(-1, m)
    w_mask = torch.zeros_like(weight)
    w_mask.scatter_(1, torch.topk(torch.abs(weight), n, dim=1, largest=True)[1], True)
    w_mask = w_mask.view(dim[0], dim[1])
    return w_mask


class ProxLinear(Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, mask=None):
        super(ProxLinear, self).__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.mask = mask  # inverse

    def forward(self, input):
        qw = self.weight * self.mask
        return F.linear(input, qw, self.bias)

    def modify_weight(self, weight, mask):
        qw = weight * mask
        return qw

    def sparsify_magnitude(self, weight, n, m):
        dim = weight.shape
        weight = weight.view(-1, m)
        w_mask = torch.zeros_like(weight)
        w_mask.scatter_(1, torch.topk(torch.abs(weight), n, dim=1, largest=True)[1], True)
        weight = w_mask * weight
        weight = weight.view(dim[0], dim[1])
        return weight


def reshape_weights(weight_matrix):
    m, n = weight_matrix.shape
    weight_matrix = weight_matrix.view(-1, 4)
    return weight_matrix, m, n


def replace_linear(model, mask_set):
    with torch.no_grad():
        for name, m in model.named_children():
            if isinstance(m, nn.Linear) and (
                    "bias" not in name and ("k_proj" in name or "v_proj" in name or "q_proj" in name or "o_proj" in name
                                            or "up_proj" in name or "down_proj" in name or "gate_proj" in name or "out_proj" in name or "fc1" in name or "fc2" in name)):
                newlinear = ProxLinear(m.in_features, m.out_features, m.bias is not None, device=m.weight.device,
                                       dtype=m.weight.dtype, mask=(mask_set[0]).to(m.weight.device).to(m.weight.dtype))
                del mask_set[0]
                newlinear.weight.data.copy_(m.weight.data * newlinear.mask)
                if m.bias is not None:
                    newlinear.bias.data.copy_(m.bias.data)
                setattr(model, name, newlinear)
            elif isinstance(m, torch.nn.LayerNorm):
                pass
            else:
                replace_linear(m, mask_set)


def main():
    seed_everything(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument('--mask_dir', type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument('--method', type=str, default="else")
    parser.add_argument('--batch_size', type=str, default='auto')
    parser = add_ckpt_argument(parser)

    args = parser.parse_args()
    model_name = args.model_name
    method = args.method
    mask_dir = args.mask_dir
    ckpt = args.ckpt

    if ckpt == 'all':
        raise NotImplementedError("Please eval checkpoints one-by-one")
    selected_checkpoints = select_checkpoints(mask_dir, ckpt)
    mask_name = mask_dir + "/" + selected_checkpoints[0]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    mask_set = []

    # def loop_lambda(model):
    for n, m in model.named_parameters():
        if "bias" not in n and (
                "k_proj" in n or "q_proj" in n or "v_proj" in n or "o_proj" in n or "up_proj" in n or "down_proj" in n or "gate_proj" in n or "out_proj" in n or "fc1" in n or "fc2" in n):
            # import ipdb; ipdb.set_trace()
            if method == "projected":
                mask = torch.load(f"./projected_mask/{n}.pt")
                mask_set.append(mask.bool())
            if method == "proximal":
                mask = torch.load(f"./proximal_mask/{n}.pt")
                mask_set.append(mask.bool())
            if method == "else":
                mask = torch.load(f"./proximal_{mask_name}/{n}.pt")
                mask_set.append(mask.bool())

    print("# Replacing layers... ", end="")
    replace_linear(model, mask_set)
    print("Done.")

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

    print(
        f"2/4 sparsity is {two_sparse / total}, one sparse is {one_sparse / total}, dense is {dense / total},  three sparse is {three_sparse / total}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    evaluate(model, tokenizer, args.batch_size)


def evaluate(model, tokenizer, batch_size=None):
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        max_length=4096,
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[
            "arc_easy",
            "arc_challenge",
            "boolq",
            "hellaswag",
            "piqa",
            "race",
            "sciq",
            "winogrande",
            "wikitext",
        ],
        batch_size=batch_size if batch_size is not None else "auto",
    )
    print(results)


if __name__ == "__main__":
    main()
