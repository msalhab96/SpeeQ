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


class AddAndNorm(nn.Module):
    """Implements the Add and norm module
    described in https://arxiv.org/abs/1706.03762

    Args:
        d_model (int): The model dimensionality.
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.lnorm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: Tensor, sub_x: Tensor):
        return self.lnorm(x + sub_x)


class MultiHeadSelfAtt(nn.Module):
    """Implements the multi-head self attention module
    described in https://arxiv.org/abs/1706.03762

    Args:
        d_model (int): The model dimensionality.
        h (int): The number of heads.
    """
    def __init__(
            self,
            d_model: int,
            h: int
            ) -> None:
        super().__init__()
        self.h = h
        self.dk = d_model // h
        self.d_model = d_model
        assert d_model % h == 0, ValueError
        self.query_fc = nn.Linear(
            in_features=d_model,
            out_features=d_model
        )
        self.key_fc = nn.Linear(
            in_features=d_model,
            out_features=d_model
        )
        self.value_fc = nn.Linear(
            in_features=d_model,
            out_features=d_model
        )
        self.softmax = nn.Softmax(dim=-1)

    def _reshape(self, x: Tensor) -> List[Tensor]:
        batch_size, max_len, _ = x.shape
        x = x.view(
            batch_size, max_len, self.h, self.dk
            )
        return x

    def _mask(
            self,
            att: Tensor,
            mask: Tensor
            ):
        # mask of shape [B, M]
        mask = mask.unsqueeze(dim=1)
        mask = mask.unsqueeze(dim=2) | mask.unsqueeze(dim=-1)
        return att.masked_fill(mask, 1e-15)

    def perform_attention(
            self,
            key: Tensor,
            query: Tensor,
            value: Tensor,
            mask: Union[Tensor, None]
            ) -> Tensor:
        key = self._reshape(key)  # B, M, h, dk
        query = self._reshape(query)  # B, M, h, dk
        value = self._reshape(value)  # B, M, h, dk
        key = key.permute(0, 2, 3, 1)  # B, h, dk, M
        query = query.permute(0, 2, 1, 3)  # B, h, M, dk
        value = value.permute(0, 2, 1, 3)  # B, h, M, dk
        att = self.softmax(
            torch.matmul(query, key) / self.d_model
            )
        if mask is not None:
            att = self._mask(att, mask)
        out = torch.matmul(att, value)
        out = out.permute(0, 2, 1, 3)
        out = out.contiguous()
        out = out.view(
            out.shape[0], out.shape[1], -1
            )
        return out

    def forward(
            self,
            key: Tensor,
            query: Tensor,
            value: Tensor,
            mask: Union[Tensor, None]
            ) -> Tensor:
        key = self.key_fc(key)
        query = self.query_fc(query)
        value = self.value_fc(value)
        return self.perform_attention(
            key=key, query=query, value=value, mask=mask
        )
