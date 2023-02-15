import csv
from typing import List

import torch

IGNORE_USERWARNING = "ignore::UserWarning"


def check_grad(result, model):
    """Checks the model's grads"""
    loss = result.mean()
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None


def get_mask(seq_len: int, pad_lens: list):
    mask = [[1] * (seq_len - item) + [0] * item for item in pad_lens]
    mask = torch.BoolTensor(mask)
    return mask


def create_csv_file(file_path, data: List[dict], encoding="utf-8", sep=","):
    with open(file_path, "w", encoding=encoding) as f:
        writer = csv.DictWriter(f, data[0].keys(), delimiter=sep)
        writer.writeheader()
        writer.writerows(data)
