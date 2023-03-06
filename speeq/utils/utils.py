import json
import os
import platform
from csv import DictReader
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.optim import Optimizer

from speeq.constants import FileKeys, StateKeys


def clear():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


def get_text_list(data: List[dict], key=FileKeys.text_key.value) -> List[str]:
    return [item[key] for item in data]


def load_json(file_path, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as f:
        data = json.load(f)
    return data


def load_text(file_path, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as f:
        data = f.read()
    return data


def save_json(file_path, data: Union[dict, list], encoding="utf-8") -> None:
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f)


def save_text(file_path, data: str, encoding="utf-8") -> None:
    with open(file_path, "w", encoding=encoding) as f:
        f.write(data)


def load_csv(file_path, encoding="utf-8", sep=","):
    with open(file_path, "r", encoding=encoding) as f:
        data = [*DictReader(f, delimiter=sep)]
    return data


def get_pad_mask(seq_len: int, pad_len: int):
    if seq_len <= 0:
        raise ValueError("seq_len must be greater than 0!")
    mask = [i < seq_len for i in range(seq_len + pad_len)]
    return torch.BoolTensor(mask)


def get_state_dict(
    model: Module, optimizer: Optimizer, step: int, history: dict
) -> dict:
    model = {
        key.replace("module.", ""): value for key, value in model.state_dict().items()
    }
    return {
        StateKeys.model.value: model,
        StateKeys.optimizer.value: optimizer.state_dict(),
        StateKeys.step.value: step,
        StateKeys.history.value: history,
    }


def save_state_dict(
    model_name: str,
    outdir: Union[str, Path],
    model: Module,
    optimizer: Optimizer,
    step: int,
    history: dict,
) -> None:
    ckpt_path = "{}_{}.pt".format(model_name, step)
    ckpt_path = os.path.join(outdir, ckpt_path)
    state = get_state_dict(model=model, optimizer=optimizer, step=step, history=history)
    torch.save(state, ckpt_path)
    print(f"checkpoint save to {ckpt_path}!")


def load_state_dict(state_path: Union[str, Path]) -> tuple:
    state = torch.load(state_path)
    model = state[StateKeys.model.value]
    optimizer = state[StateKeys.optimizer.value]
    steps = state[StateKeys.step.value]
    history = state[StateKeys.history.value]
    return model, optimizer, steps, history


def set_state_dict(
    model: Module, state_path: Union[Path, str], optimizer: Optional[Optimizer] = None
):
    model_state, optimizer_state, steps, history = load_state_dict(
        state_path=state_path
    )
    model.load_state_dict(model_state)
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state)
    return steps, history


def get_key_tag(key: str, category: str) -> str:
    return f"{key}_key_{category}"


def calc_data_len(
    result_len: int,
    pad_len: Union[Tensor, int],
    data_len: Union[Tensor, int],
    kernel_size: int,
    stride: int,
) -> Union[Tensor, int]:
    """Calculates the new data portion size after applying convolution on a padded tensor

    Args:

        result_len (int): The length after the convolution is applied.

        pad_len Union[Tensor, int]: The original padding portion length.

        data_len Union[Tensor, int]: The original data portion legnth.

        kernel_size (int): The convolution kernel size.

        stride (int): The convolution stride.

    Returns:

        Union[Tensor, int]: The new data portion length.

    """
    if type(pad_len) != type(data_len):
        raise ValueError(
            f"""expected both pad_len and data_len to be of the same type
            but {type(pad_len)}, and {type(data_len)} passed"""
        )
    inp_len = data_len + pad_len
    new_pad_len = 0
    # if padding size less than the kernel size
    # then it will be convolved with the data.
    convolved_pad_mask = pad_len >= kernel_size
    # calculating the size of the discarded items (not convolved)
    unconvolved = (inp_len - kernel_size) % stride
    undiscarded_pad_mask = unconvolved < pad_len
    convolved = pad_len - unconvolved
    new_pad_len = (convolved - kernel_size) // stride + 1
    # setting any condition violation to zeros using masks
    new_pad_len *= convolved_pad_mask
    new_pad_len *= undiscarded_pad_mask
    return result_len - new_pad_len


def get_positional_encoding(max_length: int, d_model: int) -> Tensor:
    """Create positional encoding tensor as described in
    https://arxiv.org/abs/1706.03762

    Args:

        max_length (int): The maximum length of the positionals sequence.

        d_model (int): The dimensionality of the positionals sequence.

    Returns:

        Tensor: Positional tensor of shape [1, max_length, d_model]

    """
    if d_model % 2 == 1:
        raise ValueError("Even number is expected for d_model, but odd is given!")
    result = torch.zeros(max_length, d_model, dtype=torch.float)
    feat_range = torch.arange(0, d_model // 2)
    time_range = torch.arange(0, max_length)
    denominator = pow(10000, 2 * feat_range / d_model)
    result[:, 0::2] = torch.sin(time_range[:, None] / denominator)
    result[:, 1::2] = torch.cos(time_range[:, None] / denominator)
    result = result.unsqueeze(dim=0)
    return result


def get_mask_from_lens(lengths: Tensor, max_len: int) -> Tensor:
    """Creates a mask tensor from lengths tensor.

    Args:
        lengths (Tensor): The lengths of the original tensors of shape [B].

        max_len (int): the maximum lengths.

    Returns:
        Tensor: The mask of shape [B, max_len] and True whenever the index in the data portion.
    """
    indices = torch.arange(max_len).to(lengths.device)
    indices = indices.expand(len(lengths), max_len)
    return indices < lengths.unsqueeze(dim=1)


def add_pos_enc(x: Tensor) -> Tensor:
    """Adds positional encodings to the input tensor x.

    Args:

        x (Tensor): The input tensor of shape [B, M, d].

    Returns:

        Tensor: The input added to at the positional encoding.

    """
    d_model = x.shape[-1]
    pe = get_positional_encoding(x.shape[1], d_model)
    pe = pe.to(x.device)
    return pe + x


def truncate_attention_mask(mask: Tensor, right_size: int, left_size: int) -> Tensor:
    """creates a truncation mask that can be used to mask attention to only look
    at the time steps with a certain range. Specifically, it allows attention
    to look at right_size steps to the right and left_size steps to the left of
    each time step.


    Args:

        mask (Tensor): The original mask, which is True for the data positions
        and False for the padding ones. It has a shape of [B, M].

        right_size (int): The size of the right window that each time step is
        allowed to look at.

        left_size (int): The size of the left window that each time step is
        allowed to look at.


    Returns:
        Tensor: The new mask tensor of shape [B, M, M]
    """
    max_len = mask.shape[1]
    window_size = right_size + left_size + 1
    new_mask = torch.zeros(max_len**2, dtype=torch.bool).to(mask.device)
    # creating the original positions that will be the center of the window
    centers = torch.arange(0, max_len, device=mask.device)

    # the start and the end of each window
    start = torch.clamp_min(centers - left_size, 0)
    end = torch.clamp_max(centers + right_size, max_len - 1)

    # defining the indices in each window
    indices = (
        torch.arange(0, window_size, device=mask.device)
        .repeat(max_len)
        .view(max_len, -1)
    )
    indices = torch.clamp_max(start.view(-1, 1) + indices, end.view(-1, 1))
    indices += (torch.arange(0, max_len, device=mask.device) * max_len).view(-1, 1)

    # setting the indices to True
    new_mask = new_mask.index_put((indices,), torch.tensor(True)).view(max_len, max_len)
    # merging the original tensor with the new one
    return mask.unsqueeze(dim=1) & new_mask.unsqueeze(dim=0) & mask.unsqueeze(dim=-1)


def has_bnorm(model: Module) -> bool:
    """Checks if a model contains a batch normalization layer.

    Args:
        model (Module): The model to check.

    Returns:
        bool: A boolean value indicating whether the provided model contains
        batch normalization or not.
    """
    for layer in model.children():
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True
        if has_bnorm(layer):
            return True
    return False
