from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.activations import Sigmax
from utils.utils import add_pos_enc, calc_data_len, get_positional_encoding


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

    def forward(
            self, x: Tensor,
            lens: Union[List[int], Tensor],
            h: Union[Tensor, None] = None
    ):
        packed = pack_padded_sequence(
            x, lens,
            batch_first=self.batch_first,
            enforce_sorted=self.enforce_sorted
        )
        if h is not None:
            out, h = self.rnn(packed, h)
        else:
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


class ConvPredModule(nn.Module):
    """A prediction module that consist of a signle
    Conv1d layer followed by a pre-defined activation
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
        self.activation = activation
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=n_classes,
            kernel_size=1
        )

    def forward(self, x: Tensor) -> Tensor:
        # B, d, M
        out = self.conv(x)
        out = self.activation(out)
        out = out.transpose(-1, -2)
        return out


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


class MultiHeadAtt(nn.Module):
    """Implements the multi-head attention module
    described in https://arxiv.org/abs/1706.03762

    Args:
        d_model (int): The model dimensionality.
        h (int): The number of heads.
        masking_value (int): The masking value. Default -1e15
    """

    def __init__(
            self,
            d_model: int,
            h: int,
            masking_value: int = -1e15
    ) -> None:
        super().__init__()
        self.h = h
        self.dk = d_model // h
        self.d_model = d_model
        self.masking_value = masking_value
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
            self, att: Tensor, key_mask: Tensor, query_mask: Tensor
    ) -> Tensor:
        key_max_len = key_mask.shape[-1]
        query_max_len = query_mask.shape[-1]
        key_mask = key_mask.repeat(1, query_max_len)
        key_mask = key_mask.view(-1, query_max_len, key_max_len)
        if query_mask.dim() != key_mask.dim():
            query_mask = query_mask.unsqueeze(dim=-1)
        mask = key_mask & query_mask
        mask = mask.unsqueeze(dim=1)
        return att.masked_fill(~mask, self.masking_value)

    def perform_attention(
            self,
            key: Tensor,
            query: Tensor,
            value: Tensor,
            key_mask: Union[Tensor, None],
            query_mask: Union[Tensor, None]
    ) -> Tensor:
        key = self._reshape(key)  # B, M, h, dk
        query = self._reshape(query)  # B, M, h, dk
        value = self._reshape(value)  # B, M, h, dk
        key = key.permute(0, 2, 3, 1)  # B, h, dk, M
        query = query.permute(0, 2, 1, 3)  # B, h, M, dk
        value = value.permute(0, 2, 1, 3)  # B, h, M, dk
        att = torch.matmul(query, key)
        if key_mask is not None and query_mask is not None:
            att = self._mask(
                att=att, key_mask=key_mask, query_mask=query_mask
            )
        att = self.softmax(
            att / self.d_model
        )
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
            key_mask: Union[Tensor, None],
            query_mask: Union[Tensor, None]
    ) -> Tensor:
        key = self.key_fc(key)
        query = self.query_fc(query)
        value = self.value_fc(value)
        return self.perform_attention(
            key=key, query=query, value=value,
            key_mask=key_mask, query_mask=query_mask
        )


class MaskedMultiHeadAtt(MultiHeadAtt):
    """Implements the multi-head attention module
    described in https://arxiv.org/abs/1706.03762

    Args:
        d_model (int): The model dimensionality.
        h (int): The number of heads.
        masking_value (int): The masking value. Default -1e15
    """

    def __init__(
            self,
            d_model: int,
            h: int,
            masking_value: int = -1e15
    ) -> None:
        super().__init__(
            d_model=d_model, h=h, masking_value=masking_value
        )

    def forward(
            self,
            key: Tensor,
            query: Tensor,
            value: Tensor,
            key_mask: Union[Tensor, None],
            query_mask: Union[Tensor, None]
    ) -> Tensor:
        if key_mask is not None:
            batch_size, max_len = key_mask.shape
            query_mask = torch.tril(torch.ones(batch_size, max_len, max_len))
            query_mask = query_mask.bool()
            query_mask = query_mask.to(key_mask.device)
            query_mask &= key_mask.unsqueeze(dim=-1) & query_mask
        return super().forward(
            key=key,
            query=query,
            value=value,
            key_mask=key_mask,
            query_mask=query_mask
        )


class TransformerEncLayer(nn.Module):
    """Implements a single encoder layer of the transformer
    as described in https://arxiv.org/abs/1706.03762

    Args:
        d_model (int): The model dimensionality.
        hidden_size (int): The feed forward inner layer dimensionality.
        h (int): The number of heads.
        masking_value (int): The masking value. Default -1e15
    """

    def __init__(
            self,
            d_model: int,
            hidden_size: int,
            h: int,
            masking_value: int = -1e15
    ) -> None:
        super().__init__()
        self.mhsa = MultiHeadAtt(
            d_model=d_model, h=h, masking_value=masking_value
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
            value=x, key_mask=mask,
            query_mask=mask
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
            self, x: Tensor, data_len: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # x of shape [B, M, d]
        x = x.transpose(1, 2)
        out = x
        pad_len = x.shape[-1] - data_len
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


class GlobalMulAttention(nn.Module):
    """Implements the global multiplicative
    attention mechanism as described in
    https://arxiv.org/abs/1508.04025 with direct
    dot product for scoring.
    Args:
        enc_feat_size (int): The encoder feature size.
        dec_feat_size (int): The decoder feature size.
        scaling_factor (Union[float, int]): The scaling factor
            for numerical stability used inside the softmax.
            Default 1.
        mask_val (float): the masking value. Default -1e12.
    """

    def __init__(
            self,
            enc_feat_size: int,
            dec_feat_size: int,
            scaling_factor: Union[float, int] = 1,
            mask_val: float = -1e12
    ) -> None:
        super().__init__()
        self.fc_query = nn.Linear(
            in_features=dec_feat_size,
            out_features=dec_feat_size
        )
        self.fc_key = nn.Linear(
            in_features=enc_feat_size,
            out_features=dec_feat_size
        )
        self.fc_value = nn.Linear(
            in_features=enc_feat_size,
            out_features=dec_feat_size
        )
        self.fc = nn.Linear(
            in_features=2 * dec_feat_size,
            out_features=dec_feat_size
        )
        self.scaling_factor = scaling_factor
        self.mask_val = mask_val

    def forward(
            self,
            key: Tensor,
            query: Tensor,
            mask=None
    ) -> Tensor:
        # key of shape [B, M, feat_size]
        # query of shape [B, 1, feat_size]
        # mask of shape [B, M], False for padding
        value = self.fc_value(key)
        key = self.fc_key(key)
        query = self.fc_query(query)
        att_weights = torch.matmul(
            query, key.transpose(-1, -2)
        )
        if mask is not None:
            mask = mask.unsqueeze(dim=-2)
            att_weights = att_weights.masked_fill(
                ~mask, self.mask_val
            )
        att_weights = torch.softmax(
            att_weights / self.scaling_factor, dim=-1
        )
        context = torch.matmul(att_weights, value)
        results = torch.cat([context, query], dim=-1)
        results = self.fc(results)
        results = torch.tanh(results)
        return results


class ConformerFeedForward(nn.Module):
    """Implements the conformer feed-forward module
    as described in https://arxiv.org/abs/2005.08100

    Args:
        d_model (int): The model dimension.
        expansion_factor (int): The linear layer's expansion
            factor.
        p_dropout (float): The dropout rate.
    """

    def __init__(
            self,
            d_model: int,
            expansion_factor: int,
            p_dropout: float
    ) -> None:
        super().__init__()
        self.lnrom = nn.LayerNorm(
            normalized_shape=d_model
        )
        self.fc1 = nn.Linear(
            in_features=d_model,
            out_features=expansion_factor * d_model
        )
        self.fc2 = nn.Linear(
            in_features=expansion_factor * d_model,
            out_features=d_model
        )
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.lnrom(x)
        out = self.fc1(out)
        out = self.swish(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out


class ConformerConvModule(nn.Module):
    """Implements the conformer convolution module
    as described in https://arxiv.org/abs/2005.08100

    Args:
        d_model (int): The model dimension.
        kernel_size (int): The depth-wise convolution kernel size.
        p_dropout (float): The dropout rate.
    """

    def __init__(
            self,
            d_model: int,
            kernel_size: int,
            p_dropout: float
    ) -> None:
        super().__init__()
        self.lnorm = nn.LayerNorm(
            normalized_shape=d_model
        )
        self.pwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=2 * d_model,
            kernel_size=1
        )
        self.act1 = nn.GLU(dim=1)
        self.dwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding='same'
        )
        self.bnorm = nn.BatchNorm1d(
            num_features=d_model
        )
        self.act2 = nn.SiLU()
        self.pwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1
        )
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x of shape [B, M, d]
        out = self.lnorm(x)
        out = out.transpose(-1, -2)  # [B, d, M]
        out = self.pwise_conv1(out)  # [B, 2d, M]
        out = self.act1(out)  # [B, d, M]
        out = self.dwise_conv(out)
        out = self.bnorm(out)
        out = self.act2(out)
        out = self.pwise_conv2(out)
        out = self.dropout(out)
        out = out.transpose(-1, -2)  # [B, M, d]
        return out


class ConformerRelativeMHSA(MultiHeadAtt):
    """Implements the multi-head self attention module with
    relative positional encoding as described in
    https://arxiv.org/abs/2005.08100

    Args:
        d_model (int): The model dimension.
        h (int): The number of heads.
        p_dropout (float): The dropout rate.
        masking_value (int): The masking value. Default -1e15
    """

    def __init__(
            self,
            d_model: int,
            h: int,
            p_dropout: float,
            masking_value: int = -1e15
    ) -> None:
        super().__init__(
            d_model=d_model, h=h, masking_value=masking_value
        )
        self.lnrom = nn.LayerNorm(
            normalized_shape=d_model
        )
        self.dropout = nn.Dropout(p_dropout)

    def forward(
            self,
            x: Tensor,
            mask: Union[None, Tensor]
    ) -> Tensor:
        out = self.lnrom(x)
        out = add_pos_enc(out, self.d_model)
        out = super().forward(
            key=out, query=out,
            value=out, query_mask=mask,
            key_mask=mask
        )
        out = self.dropout(out)
        return out


class ConformerBlock(nn.Module):
    """Implements the conformer block
    described in https://arxiv.org/abs/2005.08100

    Args:
        d_model (int): The model dimension.
        ff_expansion_factor (int): The linear layer's expansion factor.
        h (int): The number of heads.
        kernel_size (int): The depth-wise convolution kernel size.
        p_dropout (float): The dropout rate.
        res_scaling (float): The residual connection multiplier.
    """

    def __init__(
            self,
            d_model: int,
            ff_expansion_factor: int,
            h: int,
            kernel_size: int,
            p_dropout: float,
            res_scaling: float = 0.5
    ) -> None:
        super().__init__()
        self.ff1 = ConformerFeedForward(
            d_model=d_model,
            expansion_factor=ff_expansion_factor,
            p_dropout=p_dropout
        )
        self.mhsa = ConformerRelativeMHSA(
            d_model=d_model,
            h=h, p_dropout=p_dropout
        )
        self.conv = ConformerConvModule(
            d_model=d_model,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )
        self.ff2 = ConformerFeedForward(
            d_model=d_model,
            expansion_factor=ff_expansion_factor,
            p_dropout=p_dropout
        )
        self.lnrom = nn.LayerNorm(
            normalized_shape=d_model
        )
        self.res_scaling = res_scaling

    def forward(self, x: Tensor, mask: Union[None, Tensor]) -> Tensor:
        out = self.ff1(x)
        out = x + self.res_scaling * out
        out = out + self.mhsa(out, mask)
        out = out + self.conv(out)
        out = out + self.res_scaling * self.ff2(out)
        out = self.lnrom(out)
        return out


class ConformerPreNet(nn.Module):
    """Implements the pre-conformer blocks that contains
    the subsampling as described in https://arxiv.org/abs/2005.08100

    Args:
        in_features (int): The input/speech feature size.
        kernel_size (Union[int, List[int]]): The kernel size of the
            subsampling layer.
        stride (Union[int, List[int]]): The stride of the subsampling layer.
        n_conv_layers (int): The number of convolutional layers.
        d_model (int): The model dimension.
        p_dropout (float): The dropout rate.
        groups (Union[int, List[int]]): The convolution groups size.
    """

    def __init__(
            self,
            in_features: int,
            kernel_size: Union[int, List[int]],
            stride: Union[int, List[int]],
            n_conv_layers: int,
            d_model: int,
            p_dropout: float,
            groups: Union[int, List[int]] = 1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_features if i == 0 else d_model,
                out_channels=d_model,
                kernel_size=kernel_size if isinstance(kernel_size, int)
                else kernel_size[i],
                stride=stride if isinstance(stride, int) else stride[i],
                groups=groups if isinstance(groups, int) else groups[i]
            )
            for i in range(n_conv_layers)
        ])

        self.fc = nn.Linear(
            in_features=d_model,
            out_features=d_model
        )
        self.drpout = nn.Dropout(p_dropout)

    def forward(
            self, x: Tensor, lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # x of shape [B, M, d]
        out = x.transpose(-1, -2)  # [B, d, M]
        for conv in self.layers:
            length = out.shape[-1]
            out = conv(out)
            lengths = calc_data_len(
                result_len=out.shape[-1],
                pad_len=length - lengths,
                data_len=lengths,
                kernel_size=conv.kernel_size[0],
                stride=conv.stride[0]
            )
        out = out.transpose(-1, -2)
        out = self.fc(out)
        out = self.drpout(out)
        return out, lengths


class JasperSubBlock(nn.Module):
    """Implements the subblock of the
    Jasper model as described in
    https://arxiv.org/abs/1904.03288

    Args:
        in_channels (int): The number of the input's channels.
        out_channels (int): The number of the output's channels.
        kernel_size (int): The convolution layer's kernel size.
        p_dropout (float): The dropout rate.
        stride (int): The convolution layer's stride. Default 1.
        padding (Union[str, int]): The padding mood/size. Default 'same'.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            p_dropout: float,
            stride: int = 1,
            padding: Union[str, int] = 'same'
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
        self.bnorm = nn.BatchNorm1d(
            num_features=out_channels
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(
            self, x: Tensor,
            residual: Union[Tensor, None] = None
    ) -> Tensor:
        # x and residual of shape [B, d, M]
        out = self.conv(x)
        out = self.bnorm(out)
        if residual is not None:
            out = out + residual
        out = self.relu(out)
        out = self.dropout(out)
        return out


class JasperResidual(nn.Module):
    """Implements the the residual connection
    module as described in https://arxiv.org/abs/1904.03288

    Args:
        in_channels (int): The number of the input's channels.
        out_channels (int): The number of the output's channels.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        self.bnorm = nn.BatchNorm1d(
            num_features=out_channels
        )

    def forward(self, x: Tensor) -> Tensor:
        # x of shape [B, d, M]
        out = self.conv(x)
        out = self.bnorm(out)
        return out


class JasperBlock(nn.Module):
    """Implements the main jasper block of the
    Jasper model as described in
    https://arxiv.org/abs/1904.03288

    Args:
        num_sub_blocks (int): The number of subblocks, which is
            denoted as 'R' in the paper.
        in_channels (int): The number of the input's channels.
        out_channels (int): The number of the output's channels.
        kernel_size (int): The convolution layer's kernel size.
        p_dropout (float): The dropout rate.
    """

    def __init__(
            self,
            num_sub_blocks: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            p_dropout: float
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            JasperSubBlock(
                in_channels=in_channels if i == 1 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                p_dropout=p_dropout
            )
            for i in range(1, 1 + num_sub_blocks)
        ])
        self.residual_layer = JasperResidual(
            in_channels=in_channels,
            out_channels=out_channels
        )
        self.num_sub_blocks = num_sub_blocks

    def forward(self, x: Tensor) -> Tensor:
        # x of shape [B, d, M]
        out = x
        for i, block in enumerate(self.blocks):
            if (i + 1) == self.num_sub_blocks:
                out = block(
                    out, residual=self.residual_layer(x)
                )
            else:
                out = block(out)
        return out


class JasperBlocks(nn.Module):
    """Implements the jasper's series of blocks
    as described in https://arxiv.org/abs/1904.03288

    Args:
        num_blocks (int): The number of jasper blocks, denoted
            as 'B' in the paper.
        num_sub_blocks (int): The number of jasper subblocks, denoted
            as 'R' in the paper.
        in_channels (int): The number of the input's channels.
        channel_inc (int): The rate to increase the number of channels
            across the blocks.
        kernel_size (Union[int, List[int]]): The convolution layer's
            kernel size of each block.
        p_dropout (float): The dropout rate.
    """

    def __init__(
            self,
            num_blocks: int,
            num_sub_blocks: int,
            in_channels: int,
            channel_inc: int,
            kernel_size: Union[int, List[int]],
            p_dropout: float
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            JasperBlock(
                num_sub_blocks=num_sub_blocks,
                in_channels=in_channels + channel_inc * i,
                out_channels=in_channels + channel_inc * (1 + i),
                kernel_size=kernel_size if isinstance(
                    kernel_size, int
                ) else kernel_size[i],
                p_dropout=p_dropout
            )
            for i in range(num_blocks)
        ])

    def forward(self, x: Tensor) -> Tensor:
        # x of shape [B, d, M]
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class LocAwareGlobalAddAttention(nn.Module):
    """Implements the location-aware global additive attention
    proposed in https://arxiv.org/abs/1506.07503

    Args:
        enc_feat_size (int): The encoder feature size.
        dec_feat_size (int): The decoder feature size.
        kernel_size (int): The attention kernel size.
        activation (str): The activation function to use.
            it can be either softmax or sigmax.
        inv_temperature (Union[float, int]): The inverse temperature value.
            Default 1.
        mask_val (float): The masking value. Default -1e12.
    """

    def __init__(
            self,
            enc_feat_size: int,
            dec_feat_size: int,
            kernel_size: int,
            activation: str,
            inv_temperature: Union[float, int] = 1,
            mask_val: float = -1e12
    ) -> None:
        super().__init__()
        activations = {
            'softmax': nn.Softmax,
            'sigmax': Sigmax
        }
        assert activation in activations
        self.activation = activations[activation](dim=-2)
        self.fc_query = nn.Linear(
            in_features=dec_feat_size,
            out_features=dec_feat_size
        )
        self.fc_key = nn.Linear(
            in_features=enc_feat_size,
            out_features=dec_feat_size
        )
        self.fc_value = nn.Linear(
            in_features=enc_feat_size,
            out_features=dec_feat_size
        )
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=dec_feat_size,
            kernel_size=kernel_size,
            padding='same'
        )
        self.pos_fc = nn.Linear(
            in_features=dec_feat_size,
            out_features=dec_feat_size
        )
        self.w = nn.parameter.Parameter(
            data=torch.randn(dec_feat_size, 1)
        )
        self.mask_val = mask_val
        self.inv_temperature = inv_temperature

    def forward(
            self,
            key: Tensor,
            query: Tensor,
            alpha: Tensor,
            mask=None
    ) -> Tuple[Tensor, Tensor]:
        # alpha of shape [B, 1, M_enc]
        value = self.fc_value(key)
        key = self.fc_key(key)
        query = self.fc_query(query)
        f = self.conv(alpha)  # [B, d, M_enc]
        f = f.transpose(-1, -2)
        f = self.pos_fc(f)
        # [B, 1, d] + [B, M_enc,  d] +  [B, M_enc, d]
        e = torch.tanh(
            query + key + f
        )  # [B, M_dec, d]
        att_weights = torch.matmul(e, self.w)
        if mask is not None:
            mask = mask.unsqueeze(dim=-1)
            att_weights = att_weights.masked_fill(
                ~mask, self.mask_val
            )
        att_weights = self.activation(
            att_weights * self.inv_temperature
        )
        att_weights = att_weights.transpose(-1, -2)
        context = torch.matmul(att_weights, value)
        return context, att_weights


class MultiHeadAtt2d(MultiHeadAtt):
    """Implements the 2-dimensional multi-head self-attention
    proposed in https://ieeexplore.ieee.org/document/8462506

    Args:
        d_model (int): The model dimensionality.
        h (int): The number of heads.
        out_channels (int): The number of output channels of the convolution
        kernel_size (int): The convolutional layers' kernel size.
    """

    def __init__(
            self,
            d_model: int,
            h: int,
            out_channels: int,
            kernel_size: int
    ) -> None:
        super().__init__(out_channels, h)
        assert out_channels % h == 0
        self.query_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same'
        )
        self.key_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same'
        )
        self.value_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same'
        )
        self.fc = nn.Linear(
            in_features=2 * out_channels,
            out_features=d_model
        )

    def perform_frequency_attention(
            self,
            key: Tensor,
            query: Tensor,
            value: Tensor,
    ) -> Tensor:
        key = self._reshape(key)  # B, M, h, dk
        query = self._reshape(query)  # B, M, h, dk
        value = self._reshape(value)  # B, M, h, dk
        key = key.permute(0, 2, 1, 3)  # B, h, M, dk
        query = query.permute(0, 2, 3, 1)  # B, h, dk, M
        value = value.permute(0, 2, 3, 1)  # B, h, dk, M
        att = self.softmax(
            torch.matmul(query, key) / self.d_model
        )
        out = torch.matmul(att, value)
        out = out.permute(0, 3, 2, 1)
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
        key = key.transpose(-1, -2)
        query = query.transpose(-1, -2)
        value = value.transpose(-1, -2)
        key = self.key_conv(key)
        query = self.query_conv(query)
        value = self.value_conv(value)
        key = key.transpose(-1, -2)
        query = query.transpose(-1, -2)
        value = value.transpose(-1, -2)
        time_att_result = self.perform_attention(
            key=key, query=query, value=value,
            query_mask=mask, key_mask=mask
        )
        freq_att_result = self.perform_frequency_attention(
            key=key, query=query, value=value
        )
        result = torch.cat(
            [time_att_result, freq_att_result], dim=-1
        )
        result = self.fc(result)
        return result


class SpeechTransformerEncLayer(TransformerEncLayer):
    """Implements a single encoder layer of the speech transformer
    as described in https://ieeexplore.ieee.org/document/8462506

    Args:
        d_model (int): The model dimensionality.
        hidden_size (int): The feed-forward inner layer dimensionality.
        h (int): The number of heads.
        out_channels (int): The number of output channels of the convolution
        kernel_size (int): The convolutional layers' kernel size.
    """

    def __init__(
            self,
            d_model: int,
            hidden_size: int,
            h: int,
            out_channels: int,
            kernel_size: int
    ) -> None:
        # TODO: pass masking value
        # TODO: rename hidden size to ff_size
        super().__init__(
            d_model=d_model,
            hidden_size=hidden_size,
            h=h
        )
        self.mhsa = MultiHeadAtt2d(
            d_model=d_model,
            h=h,
            out_channels=out_channels,
            kernel_size=kernel_size
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        out = self.mhsa(
            key=x, query=x, value=x, mask=mask
        )
        out = self.add_and_norm1(x, out)
        result = self.ff(out)
        return self.add_and_norm2(
            out, result
        )


class TransformerDecLayer(nn.Module):
    """Implements a single decoder layer of the transformer
    as described in https://arxiv.org/abs/1706.03762

    Args:
        d_model (int): The model dimensionality.
        hidden_size (int): The feed forward inner layer dimensionality.
        h (int): The number of heads.
        masking_value (int): The masking value. Default -1e15
    """

    def __init__(
            self,
            d_model: int,
            hidden_size: int,
            h: int,
            masking_value: int = -1e15
    ) -> None:
        super().__init__()
        self.mmhsa = MaskedMultiHeadAtt(
            d_model=d_model,
            h=h, masking_value=masking_value
        )
        self.add_and_norm1 = AddAndNorm(
            d_model=d_model
        )
        self.mha = MultiHeadAtt(
            d_model=d_model, h=h, masking_value=masking_value
        )
        self.add_and_norm2 = AddAndNorm(
            d_model=d_model
        )
        self.ff = FeedForwardModule(
            d_model=d_model, hidden_size=hidden_size
        )
        self.add_and_norm3 = AddAndNorm(
            d_model=d_model
        )

    def forward(
            self,
            enc_out: Tensor,
            enc_mask: Union[Tensor, None],
            dec_inp: Tensor,
            dec_mask: Union[Tensor, None],
    ) -> Tensor:
        out = self.mmhsa(
            key=dec_inp,
            query=dec_inp,
            value=dec_inp,
            key_mask=dec_mask,
            query_mask=dec_mask
        )
        out = self.add_and_norm1(out, dec_inp)
        out = self.add_and_norm2(
            self.mha(
                key=enc_out,
                query=out,
                value=enc_out,
                key_mask=enc_mask,
                query_mask=dec_mask
            ),
            out
        )
        out = self.add_and_norm3(
            self.ff(out), out
        )
        return out


class PositionalEmbedding(nn.Module):
    """Implements the positional embedding proposed in
    https://arxiv.org/abs/1706.03762

    Args:
        vocab_size (int): The vocabulary size.
        embed_dim (int): The embedding size.
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )
        self.d_model = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        max_len = x.shape[-1]
        pe = get_positional_encoding(
            max_length=max_len, d_model=self.d_model
        )
        pe = pe.to(x.device)
        return self.emb(x) + pe


class GroupsShuffle(nn.Module):
    """Implements the group shuffle proposed in
    https://arxiv.org/abs/1707.01083

    Args:
        groups (int): The groups size.
    """

    def __init__(self, groups: int) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, x: Tensor) -> Tensor:
        # x of shape [B, C, ...]
        batch_size, channels, *_ = x.shape
        dims = x.shape[2:]
        x = x.view(
            batch_size, self.groups, channels // self.groups, *dims
        )
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(batch_size, channels, *dims)
        return x


class QuartzSubBlock(JasperSubBlock):
    """Implements Quartznet's subblock module
    described in https://arxiv.org/abs/1910.10261

    Args:
        in_channels (int): The number of the input's channels.
        out_channels (int): The number of the output's channels.
        kernel_size (int): The convolution layer's kernel size.
        p_dropout (float): The dropout rate.
        groups (int): The groups size.
        stride (int): The convolution layer's stride. Default 1.
        padding (Union[str, int]): The padding mood/size. Default 'same'.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            p_dropout: float,
            groups: int,
            stride: int = 1,
            padding: Union[str, int] = 'same'
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            p_dropout,
            stride,
            padding
        )
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                groups=groups
            ),
            GroupsShuffle(
                groups=groups
            )
        )
        self.dwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            padding='same'
        )

    def forward(
            self, x: Tensor,
            residual: Union[Tensor, None] = None
    ) -> Tensor:
        # x and residual of shape [B, d, M]
        x = self.dwise_conv(x)
        return super().forward(x=x, residual=residual)


class QuartzBlock(JasperBlock):
    """Implements the main quartznet block of the quartznet
    model as described in https://arxiv.org/abs/1904.03288

    Args:
        num_sub_blocks (int): The number of subblocks, which is
            denoted as 'R' in the paper.
        in_channels (int): The number of the input's channels.
        out_channels (int): The number of the output's channels.
        kernel_size (int): The convolution layer's kernel size.
        groups (int): The groups size.
        p_dropout (float): The dropout rate.
    """

    def __init__(
            self,
            num_sub_blocks: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            groups: int,
            p_dropout: float
    ) -> None:
        super().__init__(
            num_sub_blocks,
            in_channels,
            out_channels,
            kernel_size,
            p_dropout
        )
        self.blocks = nn.ModuleList([
            QuartzSubBlock(
                in_channels=in_channels if i == 1 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                groups=groups,
                p_dropout=p_dropout
            )
            for i in range(1, 1 + num_sub_blocks)
        ])


class QuartzBlocks(JasperBlocks):
    """Implements the quartznet's series of blocks
    as described in https://arxiv.org/abs/1910.10261

    Args:
        num_blocks (int): The number of QuartzNet blocks, denoted
            as 'B' in the paper.
        block_repetition (int): The nubmer of times to repeat each block.
            denoted as S in the paper.
        num_sub_blocks (int): The number of QuartzNet subblocks, denoted
            as 'R' in the paper.
        in_channels (int): The number of the input's channels.
        channels_size (List[int]): The channel size of each block.
        kernel_size (Union[int, List[int]]): The convolution layer's
            kernel size of each block.
        groups (int): The groups size.
        p_dropout (float): The dropout rate.
    """

    def __init__(
            self,
            num_blocks: int,
            block_repetition: int,
            num_sub_blocks: int,
            in_channels: int,
            channels_size: List[int],
            kernel_size: Union[int, List[int]],
            groups: int,
            p_dropout: float
    ) -> None:
        super().__init__(
            num_blocks=num_blocks,
            num_sub_blocks=num_sub_blocks,
            in_channels=in_channels,
            channel_inc=0,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )
        assert len(channels_size) == num_blocks
        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            channel_size = channels_size[i - 1] if i != 0 else in_channels
            self.blocks.append(
                torch.nn.Sequential(
                    *[
                        QuartzBlock(
                            num_sub_blocks=num_sub_blocks,
                            in_channels=channel_size if j == 0
                            else channels_size[i],
                            out_channels=channels_size[i],
                            kernel_size=kernel_size if isinstance(
                                kernel_size, int
                            ) else kernel_size[i],
                            groups=groups,
                            p_dropout=p_dropout
                        )
                        for j in range(block_repetition)
                    ]
                )
            )


class Scaling1d(nn.Module):
    """Implements the scaling layer proposed in
    https://arxiv.org/abs/2206.00888

    Args:
        d_model (int): The model dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(
            torch.randn(1, 1, d_model)
        )
        self.beta = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.gamma * x + self.beta


class SqueezeformerConvModule(ConformerConvModule):
    """Implements the conformer convolution module
    with the modification as described in
    https://arxiv.org/abs/2206.00888

    Args:
        d_model (int): The model dimension.
        kernel_size (int): The depth-wise convolution kernel size.
        p_dropout (float): The dropout rate.
    """

    def __init__(
            self, d_model: int, kernel_size: int, p_dropout: float
    ) -> None:
        super().__init__(d_model, kernel_size, p_dropout)
        self.pwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1
        )
        self.act1 = nn.SiLU()
        self.scaler = Scaling1d(d_model=d_model)
        del self.lnorm

    def forward(self, x: Tensor) -> Tensor:
        # x of shape [B, M, d]
        out = self.scaler(x)
        out = out.transpose(-1, -2)
        out = self.pwise_conv1(out)
        out = self.act1(out)
        out = self.dwise_conv(out)
        out = self.bnorm(out)
        out = self.act2(out)
        out = self.pwise_conv2(out)
        out = self.dropout(out)
        out = out.transpose(-1, -2)  # [B, M, d]
        return out


class SqueezeformerRelativeMHSA(MultiHeadAtt):
    """Implements the multi-head self attention module with
    relative positional encoding and pre-scaling module.

    Args:
        d_model (int): The model dimension.
        h (int): The number of heads.
        p_dropout (float): The dropout rate.
        masking_value (int): The masking value. Default -1e15
    """

    def __init__(
            self,
            d_model: int,
            h: int,
            p_dropout: float,
            masking_value: int = -1e15
    ) -> None:
        super().__init__(
            d_model=d_model, h=h, masking_value=masking_value
        )
        self.dropout = nn.Dropout(p_dropout)
        self.scaler = Scaling1d(d_model=d_model)

    def forward(
            self,
            x: Tensor,
            mask: Union[None, Tensor]
    ) -> Tensor:
        out = self.scaler(x)
        out = add_pos_enc(out, self.d_model)
        out = super().forward(
            key=out, query=out,
            value=out, query_mask=mask,
            key_mask=mask
        )
        out = self.dropout(out)
        return out


class SqueezeformerFeedForward(ConformerFeedForward):
    """Implements the conformer feed-forward module
    with the modifications presented in
    https://arxiv.org/abs/2206.00888

    Args:
        d_model (int): The model dimension.
        expansion_factor (int): The linear layer's expansion
            factor.
        p_dropout (float): The dropout rate.
    """

    def __init__(
            self,
            d_model: int,
            expansion_factor: int,
            p_dropout: float
    ) -> None:
        super().__init__(
            d_model=d_model,
            expansion_factor=expansion_factor,
            p_dropout=p_dropout
        )
        del self.lnrom
        self.scaler = Scaling1d(d_model=d_model)

    def forward(self, x: Tensor) -> Tensor:
        out = self.scaler(x)
        out = self.fc1(out)
        out = self.swish(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out


class SqueezeformerBlock(nn.Module):
    """Implements the Squeezeformer block
    described in https://arxiv.org/abs/2206.00888

    Args:
        d_model (int): The model dimension.
        ff_expansion_factor (int): The linear layer's expansion factor.
        h (int): The number of heads.
        kernel_size (int): The depth-wise convolution kernel size.
        p_dropout (float): The dropout rate.
        masking_value (int): The masking value. Default -1e15
    """

    def __init__(
            self, d_model: int, ff_expansion_factor: int,
            h: int, kernel_size: int, p_dropout: float,
            masking_value: int = -1e15
    ) -> None:
        super().__init__()
        self.mhsa = SqueezeformerRelativeMHSA(
            d_model=d_model, h=h, p_dropout=p_dropout,
            masking_value=masking_value
        )
        self.add_and_norm1 = AddAndNorm(d_model=d_model)
        self.ff1 = SqueezeformerFeedForward(
            d_model=d_model,
            expansion_factor=ff_expansion_factor,
            p_dropout=p_dropout
        )
        self.add_and_norm2 = AddAndNorm(d_model=d_model)
        self.conv = SqueezeformerConvModule(
            d_model=d_model,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )
        self.add_and_norm3 = AddAndNorm(d_model=d_model)
        self.ff2 = SqueezeformerFeedForward(
            d_model=d_model,
            expansion_factor=ff_expansion_factor,
            p_dropout=p_dropout
        )
        self.add_and_norm4 = AddAndNorm(d_model=d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        out = self.add_and_norm1(self.mhsa(x, mask), x)
        out = self.add_and_norm2(self.ff1(out), out)
        out = self.add_and_norm3(self.conv(out), out)
        out = self.add_and_norm4(self.ff2(out), out)
        return out
