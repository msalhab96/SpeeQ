import torch
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


class PackedGRU(PackedRNN):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            batch_first=True,
            enforce_sorted=False,
            bidirectional=False
            ) -> None:
        super().__init__(
            input_size, hidden_size, batch_first, enforce_sorted, bidirectional
            )
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional
        )


class PredModule(nn.Module):
    """A prediction module that consist of a signle
    feed forward layer followed by a pre-defined activation
    function.

    Args:
        in_features (int): The input feature size.
        n_classes (int): The number of classes to produce.
        activation (Module): The activation function to be used.
    """
    def __init__(
            self,
            in_features: int,
            n_classes: int,
            activation: nn.Module
            ) -> None:
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=n_classes
        )
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.fc(x))


class CReLu(nn.Module):
    """clipped rectified-linear unit, can be described as
    min{max{0, x}, max_value}

    as described in: https://arxiv.org/abs/1412.5567

    Args:
        max_val (int): the maximum clipping value.
    """
    def __init__(self, max_val: int) -> None:
        super().__init__()
        self.max_val = max_val

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(
            x, min=0, max=self.max_val
            )


class FeedForwardModule(nn.Module):
    """Implements the feed-forward module
    described in https://arxiv.org/abs/1706.03762

    Args:
        d_model (int): The model dimensionality.
        hidden_size (int): The inner layer's dimensionality.
    """
    def __init__(
            self,
            d_model: int,
            hidden_size: int
            ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=d_model,
            out_features=hidden_size
        )
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(
            in_features=hidden_size,
            out_features=d_model
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
