"""The `padders` module provides two classes for padding input sequences:
`DynamicPadder` and `StaticPadder`.

`DynamicPadder` pads an input sequence along a specified dimension to match
the maximum sequence length, while `StaticPadder` is a subclass of
`DynamicPadder` that also allows the user to specify the maximum length
of the sequence to pad to.

Both classes have a `pad` method that accepts an input tensor and the maximum
length to pad to, and returns the padded tensor and the length of the padding added.


Usage:


.. code-block:: python

    import torch
    from speeq.data.padders import DynamicPadder, StaticPadder

    # create a dummy input
    input_tensor = torch.randn(1, 3, 7)

    # Example usage of DynamicPadder
    dynamic_padder = DynamicPadder(dim=1, pad_val=0)
    padded_tensor, padding_length = dynamic_padder.pad(input_tensor, max_len=10)

    # Example usage of StaticPadder
    static_padder = StaticPadder(dim=1, pad_val=0, max_len=10)
    padded_tensor, padding_length = static_padder.pad(input_tensor)

"""
from typing import Tuple, Union

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
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.dim = dim
        self.left_pad = left_pad
        self.pad_val = pad_val

    def pad(self, x: Tensor, max_len: int) -> Tuple[Tensor, int]:
        """Pads the input tensor to match the specified maximum length along the
        pre-defined dimension.

        Args:
            x (Tensor): The input tensor to be padded.

            max_len (int): The maximum length to pad the input tensor to.

        Returns:
            Tuple[Tensor, int]: A tuple containing the padded tensor and the
            length of the padding added.
        """
        seq_len = x.shape[self.dim]
        pad_len = max_len - seq_len
        assert pad_len >= 0
        if pad_len == 0:
            return x, pad_len
        pad = torch.zeros(
            *x.shape[: self.dim], pad_len, *x.shape[1 + self.dim :], dtype=x.dtype
        ).to(x.device)
        pad = pad + self.pad_val
        if self.left_pad:
            x = torch.cat([pad, x], dim=self.dim)
        else:
            x = torch.cat([x, pad], dim=self.dim)
        return x, pad_len


class StaticPadder(DynamicPadder):
    """A subclass of `DynamicPadder` that pads an input sequence to match
    a pre-defined maximum length along a specified dimension.


    Args:
        dim (int): The dimension to pad across.

        pad_val (Union[int, Tensor, float]): The value used to fill the padded sequence.

        max_len (int): The maximum length of the sequence to pad to.

        left_pad (int): The side to which the sequence will be padded.

    """

    def __init__(
        self,
        dim: int,
        pad_val: Union[int, Tensor, float],
        max_len: int,
        left_pad=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(dim, pad_val, left_pad)
        self.max_len = max_len

    def pad(self, x: Tensor, *args, **kwargs):
        """Pads the input tensor to match the specified maximum length along the
        pre-defined dimension.

        Args:
            x (Tensor): The input tensor to be padded.

        Returns:
            Tuple[Tensor, int]: A tuple containing the padded tensor and the
            length of the padding added.
        """
        return super().pad(x, self.max_len)
