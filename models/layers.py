import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Union
from torch.nn.utils.rnn import (
    pack_padded_sequence, pad_packed_sequence
)
from utils.utils import calc_data_len


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


class TransformerEncLayer(nn.Module):
    """Implements a single encoder layer of the transformer
    as described in https://arxiv.org/abs/1706.03762

    Args:
        d_model (int): The model dimensionality.
        hidden_size (int): The feed forward inner
            layer dimensionality..
        h (int): The number of heads.
    """
    def __init__(
            self,
            d_model: int,
            hidden_size: int,
            h: int
            ) -> None:
        super().__init__()
        self.mhsa = MultiHeadSelfAtt(
            d_model=d_model, h=h
            )
        self.add_and_norm1 = AddAndNorm(
            d_model=d_model
            )
        self.ff = FeedForwardModule(
            d_model=d_model, hidden_size=hidden_size
            )
        self.add_and_norm2 = AddAndNorm(
            d_model=d_model
            )

    def forward(
            self,
            x: Tensor,
            mask: Tensor
            ) -> Tensor:
        out = self.mhsa(
            key=x, query=x,
            value=x, mask=mask
            )
        out = self.add_and_norm1(x, out)
        result = self.ff(out)
        return self.add_and_norm2(
            out, result
            )


class RowConv1D(nn.Module):
    """Implements the row convolution module
    proposed in https://arxiv.org/abs/1512.02595

    Args:
        tau (int): The size of future context.
        hidden_size (int): The input feature size.
    """
    def __init__(
            self,
            tau: int,
            hidden_size: int
            ) -> None:
        super().__init__()
        self.tau = tau
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=tau, stride=1,
            padding=0, dilation=1
        )

    def _pad(self, x: Tensor):
        """pads the input with zeros along the
        time dim.

        Args:
            x (Tensor): The input tensor of shape [B, d, M].

        Returns:
            Tensor: The padded tensor.
        """
        zeros = torch.zeros(
            *x.shape[:-1], self.tau
            )
        zeros = zeros.to(x.device)
        return torch.cat(
            [x, zeros], dim=-1
        )

    def forward(self, x: Tensor):
        # x of shape [B, M, d]
        max_len = x.shape[1]
        x = x.transpose(1, 2)
        x = self._pad(x)
        out = self.conv(x)
        # remove the conv on the padding if there is any
        out = out[..., :max_len]
        out = out.transpose(1, 2)
        return out


class Conv1DLayers(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            kernel_size: int,
            stride: int,
            n_layers: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_size if i == 0 else out_size,
                out_channels=out_size,
                kernel_size=kernel_size,
                stride=stride
            )
            for i in range(n_layers)
        ])
        self.dropout = nn.Dropout(p_dropout)

    def forward(
            self, x: Tensor, mask: Tensor
            ) -> Tuple[Tensor, Tensor]:
        # x of shape [B, M, d]
        x = x.transpose(1, 2)
        out = x
        data_len = mask.sum(dim=-1)
        pad_len = mask.shape[-1] - data_len
        for layer in self.layers:
            out = layer(out)
            out = self.dropout(out)
            result_len = out.shape[-1]
            data_len = calc_data_len(
                result_len=result_len,
                pad_len=pad_len,
                data_len=data_len,
                kernel_size=layer.kernel_size[0],
                stride=layer.stride[0]
            )
            pad_len = result_len - data_len
        out = out.transpose(1, 2)
        return out, data_len
