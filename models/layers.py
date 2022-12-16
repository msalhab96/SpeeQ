import torch.nn as nn
from torch import Tensor
from typing import List, Union
from torch.nn.utils.rnn import (
    pack_padded_sequence, pad_packed_sequence
)


class PackedRNN(nn.Module):
    """Packed RNN Module utilizes the RNN built in torch
    with the padding functionalities provided in torch.

    Args:
        input_size (int): The RNN input size
        hidden_size (int): The RNN hidden size
        batch_first (bool): whether the batch will be in the
        first dimension or not. Default to True.
        enforce_sorted (bool): If the inputs are sorted based
        on their length. Default to False.
        bidirectional (bool): If the RNN is bidirectional or not.
        Default to False.
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            batch_first=True,
            enforce_sorted=False,
            bidirectional=False
            ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
        self.batch_first = batch_first
        self.enforce_sorted = enforce_sorted

    def forward(self, x: Tensor, lens: Union[List[int], Tensor]):
        packed = pack_padded_sequence(
            x, lens,
            batch_first=self.batch_first,
            enforce_sorted=self.enforce_sorted
            )
        out, h = self.rnn(packed)
        out, lens = pad_packed_sequence(out, batch_first=self.batch_first)
        return out, h, lens


class PackedLSTM(PackedRNN):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            batch_first=True,
            enforce_sorted=False,
            bidirectional=False
            ) -> None:
        super().__init__(
            input_size, hidden_size, batch_first, enforce_sorted
            )
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
