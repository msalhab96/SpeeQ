from typing import Union

import torch
from torch import Tensor

from speeq.interfaces import IPadder


class DynamicPadder(IPadder):
    """Pads the input sequence across a dim for the maximum length

    Args:
        dim (int): The dimension to do the padding across.
        pad_val (Union[int, Tensor, float]): The padding value that
        will be used to fill the padding sequence.
        left_pad (int): The side to pad the padding sequence to.
    """

    def __init__(
            self,
            dim: int,
            pad_val: Union[int, Tensor, float],
            left_pad=False,
            *args, **kwargs
    ) -> None:
        super().__init__()
        self.dim = dim
        self.left_pad = left_pad
        self.pad_val = pad_val

    def pad(self, x: Tensor, max_len: int):
        seq_len = x.shape[self.dim]
        pad_len = max_len - seq_len
        assert pad_len >= 0
        if pad_len == 0:
            return x, pad_len
        pad = torch.zeros(
            *x.shape[:self.dim],
            pad_len,
            *x.shape[1 + self.dim:],
            dtype=x.dtype
        ).to(x.device)
        pad = pad + self.pad_val
        if self.left_pad:
            x = torch.cat([pad, x], dim=self.dim)
        else:
            x = torch.cat([x, pad], dim=self.dim)
        return x, pad_len


class StaticPadder(DynamicPadder):
    """Pads the input sequence across a dim for the maximum length

    Args:
        dim (int): The dimension to do the padding across.
        pad_val (Union[int, Tensor, float]): The padding value that
        will be used to fill the padding sequence.
        max_len (int): The maximum sequence length.
        left_pad (int): The side to pad the padding sequence to.
    """

    def __init__(
            self,
            dim: int,
            pad_val: Union[int, Tensor, float],
            max_len: int,
            left_pad=False,
            *args, **kwargs
    ) -> None:
        super().__init__(dim, pad_val, left_pad)
        self.max_len = max_len

    def pad(self, x: Tensor, *args, **kwargs):
        return super().pad(x, self.max_len)
