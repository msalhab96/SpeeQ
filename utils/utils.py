import os
import json
import torch
import platform
from torch import Tensor
from pathlib import Path
from typing import Union, List
from csv import DictReader
from constants import FileKeys
from torch.nn import Module
from torch.optim import Optimizer
from constants import StateKeys


def clear():
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')


def get_text_list(data: List[dict]) -> List[str]:
    return [item[FileKeys.text_key] for item in data]


def load_json(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        data = json.load(f)
    return data


def load_text(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        data = f.read()
    return data


def save_json(
        file_path,
        data: Union[dict, list],
        encoding='utf-8'
        ) -> None:
    with open(file_path, 'w', encoding=encoding) as f:
        json.dump(data, f)


def save_text(
        file_path,
        data: str,
        encoding='utf-8'
        ) -> None:
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(data)


def load_csv(
        file_path,
        encoding='utf-8',
        sep=','
        ):
    with open(file_path, 'r', encoding=encoding) as f:
        data = [*DictReader(f, delimiter=sep)]
    return data


def get_pad_mask(seq_len: int, pad_len: int):
    mask = [i < seq_len for i in range(seq_len + pad_len)]
    return torch.BoolTensor(mask)


def get_state_dict(
        model: Module,
        optimizer: Optimizer,
        step: int,
        history: dict
        ) -> dict:
    model = {
        key.replace('module.', ''): value
        for key, value in model.state_dict().items()
        }
    return {
        StateKeys.model.value: model,
        StateKeys.optimizer.value: optimizer.state_dict(),
        StateKeys.step.value: step,
        StateKeys.history.value: history
    }


def save_state_dict(
        model_name: str,
        outdir: Union[str, Path],
        model: Module,
        optimizer: Optimizer,
        step: int,
        history: dict
        ) -> None:
    ckpt_path = '{}_{}.pt'.format(
        model_name, step
    )
    ckpt_path = os.path.join(
        outdir, ckpt_path
        )
    state = get_state_dict(
        model=model,
        optimizer=optimizer,
        step=step,
        history=history
        )
    torch.save(state, ckpt_path)
    print(f'checkpoint save to {ckpt_path}!')


def load_state_dict(state_path: Union[str, Path]) -> tuple:
    state = torch.load(state_path)
    model = state[StateKeys.model]
    optimizer = state[StateKeys.optimizer]
    steps = state[StateKeys.steps]
    history = state[StateKeys.history]
    return model, optimizer, steps, history


def set_state_dict(model, optimizer, state_path):
    model_state, optimizer_state, steps, history = load_state_dict(
        state_path=state_path
    )
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return steps, history


def get_key_tag(key: str, category: str) -> str:
    return f'{key}_key_{category}'


def calc_data_len(
        result_len: int,
        pad_len: Union[Tensor, int],
        data_len: Union[Tensor, int],
        kernel_size: int,
        stride: int
        ) -> Union[Tensor, int]:
    """Calculates the new data portion size
    after applying convolution on a padded tensor

    Args:
        result_len int: The length after the
            convolution iss applied.
        pad_len Union[Tensor, int]: The original padding portion length.
        data_len Union[Tensor, int]: The original data portion legnth.
        kernel_size (int): The convolution kernel size.
        stride (int): The convolution stride.

    Returns:
        Union[Tensor, int]: The new data portion length.
    """
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
