import os
import torch
from pathlib import Path
from typing import Union
from torch.nn import Module
from torch.optim import Optimizer
from constants import StateKeys


def get_state_dict(
        model: Module,
        optimizer: Optimizer,
        epoch: int,
        step: int,
        history: dict
        ) -> dict:
    model = {
        key.replace('module.', ''): value
        for key, value in model.state_dict().items()
        }
    return {
        StateKeys.model: model,
        StateKeys.epoch: epoch,
        StateKeys.optimizer: optimizer.state_dict(),
        StateKeys.step: step,
        StateKeys.history: history
    }


def save_state_dict(
        model_name: str,
        outdir: Union[str, Path],
        model: Module,
        optimizer: Optimizer,
        epoch: int,
        step: int,
        history: dict
        ) -> None:
    ckpt_path = '{}_{}_{}.pt'.format(
        model_name, epoch, step
    )
    ckpt_path = os.path.join(
        outdir, ckpt_path
        )
    state = get_state_dict(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        history=history
        )
    torch.save(state, ckpt_path)


def load_state_dict(state_path: Union[str, Path]) -> tuple:
    state = torch.load(state_path)
    model = state[StateKeys.model]
    optimizer = state[StateKeys.optimizer]
    epoch = state[StateKeys.epoch]
    steps = state[StateKeys.steps]
    history = state[StateKeys.history]
    return model, optimizer, epoch, steps, history


def set_state_dict(model, optimizer, state_path):
    model_state, optimizer_state, epoch, steps, history = load_state_dict(
        state_path=state_path
    )
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return epoch, steps, history
