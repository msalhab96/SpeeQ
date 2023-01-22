from math import sqrt
from typing import Iterable
from constants import OPTIMIZER_STATE_KEY
from interfaces import ISchedular


class Schedular(ISchedular):
    """Implements the base scheduler class.

    Args:
        params (Iterable): The mdoel's parameters.
        optimizer (str): The optimizer's name.
        optimizer_args (dict): The optimizer's arguments.
    """

    def __init__(
            self,
            params: Iterable,
            optimizer: str,
            optimizer_args: dict
            ) -> None:
        super().__init__()
        from .registry import OPTIMIZERS
        self.optimizer = OPTIMIZERS[optimizer](
            params, **optimizer_args
        )

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def _update_lr(self) -> None:
        self.counter += 1
        lr = self.get_lr(self.counter)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step(self) -> None:
        self.optimizer.step()
        self._update_lr()

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict[OPTIMIZER_STATE_KEY])
        state_dict.pop(OPTIMIZER_STATE_KEY)
        self.__dict__.update(state_dict)


class NoamSchedular(Schedular):
    """Implements the noam scheduler  proposed in
    https://arxiv.org/abs/1706.03762

    Args:
        params (Iterable): The mdoel's parameters.
        optimizer (str): The optimizer's name.
        optimizer_args (dict): The optimizer's arguments.
        warmup_staps (int): The warmup steps.
        d_model (int): The model dimension.
    """

    def __init__(
            self,
            params,
            optimizer: str,
            optimizer_args: dict,
            warmup_staps: int,
            d_model: int,
            *args, **kwargs
            ) -> None:
        super().__init__(
            params=params,
            optimizer=optimizer,
            optimizer_args=optimizer_args
            )
        self.peak = 1 / sqrt(d_model)
        self.counter = 0
        self.warmup_staps = warmup_staps
        self._update_lr()

    def get_lr(self) -> float:
        return self.peak * min(
            1 / sqrt(self.counter),
            self.counter * pow(self.warmup_staps, -1.5)
        )

    def state_dict(self) -> dict:
        return {
            'peak': self.peak,
            'warmup_staps': self.warmup_staps,
            'counter': self.counter,
            OPTIMIZER_STATE_KEY: self.optimizer.state_dict()
        }
