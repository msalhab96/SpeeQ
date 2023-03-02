"""
This module contains implementations of various atomic layers used in neural network models.

Layers:

- PackedRNN: RNN layer with support for PackedSequence.
- PackedLSTM: LSTM layer with support for PackedSequence.
- PackedGRU: GRU layer with support for PackedSequence.
- PredModule: A simple feedforward prediction module.
- ConvPredModule: A convolutional prediction module.
- FeedForwardModule: A transformer feedforward module.
- AddAndNorm: A layer that performs residual connection and layer normalization.
- MultiHeadAtt: Multi-Head Attention layer.
- MaskedMultiHeadAtt: Masked Multi-Head Attention layer.
- TransformerEncLayer: Transformer Encoder layer.
- RowConv1D: A 1D convolution layer that convolves each row separately.
- Conv1DLayers: A stack of 1D convolutional layers.
- GlobalMulAttention: Global Multiplicative Attention layer.
- ConformerFeedForward: A feedforward module used in Conformer model.
- ConformerConvModule: A convolutional module used in Conformer model.
- ConformerRelativeMHSA: Conformer Relative Multi-Head Self-Attention layer.
- ConformerBlock: Conformer block.
- ConformerPreNet: A pre-processing network used in Conformer model.
- JasperSubBlock: Jasper Sub-block.
- JasperResidual: Jasper Residual module.
- JasperBlock: Jasper Block.
- JasperBlocks: A stack of Jasper Blocks.
- LocAwareGlobalAddAttention: Location-Aware Global Additive Attention layer.
- MultiHeadAtt2d: 2D Multi-Head Attention layer.
- SpeechTransformerEncLayer: Speech Transformer Encoder layer.
- TransformerDecLayer: Transformer Decoder layer.
- PositionalEmbedding: Positional embedding layer.
- GroupsShuffle: Group Shuffle layer.
- QuartzSubBlock: Quartz Sub-block.
- QuartzBlock: Quartz Block.
- QuartzBlocks: A stack of Quartz Blocks.
- Scaling1d: A learnable scaling layer.
- SqueezeformerConvModule: A convolutional module used in Squeezeformer model.
- SqueezeformerRelativeMHSA: Squeezeformer Relative Multi-Head Self-Attention layer.
- SqueezeformerFeedForward: A feedforward module used in Squeezeformer model.
- SqueezeformerBlock: Squeezeformer block.
- SqueezeAndExcit1D: Squeeze-and-Excitation layer for 1D inputs.
- ContextNetConvLayer: ContextNet convolution layer.
- ContextNetResidual: ContextNet residual module.
- ContextNetBlock: ContextNet block.
- CausalVGGBlock: Causal VGG Block.
- TruncatedSelfAttention: Truncated self attention.
- TransformerEncLayerWithAttTruncation: Transformer Encoder layer with truncated self attention.
- VGGTransformerPreNet: VGG Transformer prenet.
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from speeq.utils.utils import (
    add_pos_enc,
    calc_data_len,
    get_mask_from_lens,
    truncate_attention_mask,
)

from .activations import Sigmax


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
        bidirectional=False,
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        self.batch_first = batch_first
        self.enforce_sorted = enforce_sorted

    def forward(
        self, x: Tensor, lens: Union[List[int], Tensor], h: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Passes the input tensor x of shape [B, M, d], along with tensor or
        list of lengths lens of shape [B] representing the length of each
        sequence without padding, through the layer. An optional tensor h
        representing the last hidden state can also be provided.


        Args:
            x (Tensor): The input sequence tensor of shape [B, M, d].

            lens (Union[List[int], Tensor]): The lengths of the data without
            padding for each sequence of length [B].

            h (Tensor, optional): The last hidden state if there's any. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple of three tensors containing
            the output sequence of shape [B, max(lens), hidden_size], the last
            hidden state of shape [D, B, hidden_size], and the new lengths.
        """
        packed = pack_padded_sequence(
            x, lens, batch_first=self.batch_first, enforce_sorted=self.enforce_sorted
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
        bidirectional=False,
    ) -> None:
        super().__init__(input_size, hidden_size, batch_first, enforce_sorted)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )


class PackedGRU(PackedRNN):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        batch_first=True,
        enforce_sorted=False,
        bidirectional=False,
    ) -> None:
        super().__init__(
            input_size, hidden_size, batch_first, enforce_sorted, bidirectional
        )
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )


class PredModule(nn.Module):
    """This is a Prediction Module class that comprises a single feedforward
    layer followed by a pre-defined activation function.


    Args:
        in_features (int): The input feature size.

        n_classes (int): The number of classes to be produced.

        activation (Module): The activation function to be used.
    """

    def __init__(self, in_features: int, n_classes: int, activation: nn.Module) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=n_classes)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input thought the layers' modules, where the input x of
        shape [B, M, d]

        Args:
            x (Tensor): The input tensor of shape [B, M, d]

        Returns:
            Tensor: The output tensor of shape [B, M, C] obtained after passing
            through the layers' modules.
        """
        return self.activation(self.fc(x))


class ConvPredModule(nn.Module):
    """A prediction module that consist of a signle
    Conv1d layer followed by a pre-defined activation
    function.

    Args:
        in_features (int): The input feature size.

        n_classes (int): The number of classes to be produced.

        activation (Module): The activation function to be used.
    """

    def __init__(self, in_features: int, n_classes: int, activation: nn.Module) -> None:
        super().__init__()
        self.activation = activation
        self.conv = nn.Conv1d(
            in_channels=in_features, out_channels=n_classes, kernel_size=1
        )

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input thought the layers' modules, where the input x of
        shape [B, M, C]

        Args:
            x (Tensor): The input tensor of shape [B, M, d]

        Returns:
            Tensor: The output tensor of shape [B, M, C] obtained after passing
            through the layers' modules.
        """
        x = x.transpose(-1, -2)
        out = self.conv(x)
        out = out.transpose(-1, -2)
        out = self.activation(out)
        return out


class FeedForwardModule(nn.Module):
    """Implements the feed-forward module of the transformer architecture as
    described in the paper https://arxiv.org/abs/1706.03762

    Args:
        d_model (int): The model dimensionality.

        ff_size (int): The dimensionality of the inner layer.
    """

    def __init__(self, d_model: int, ff_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=d_model, out_features=ff_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=ff_size, out_features=d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input to the layer

        Args:
            x (Tensor): The input tensor of shape [B, M, d]

        Returns:
            Tensor: The output tensor of shape [B, M, d] obtained after passing
            through the layer.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class AddAndNorm(nn.Module):
    """Implements the Add and Norm module of the transformer model as described
    in the paper https://arxiv.org/abs/1706.03762

    Args:

        d_model (int): The model dimensionality.

    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.lnorm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: Tensor, sub_x: Tensor) -> Tensor:
        """takes the output tensor `x` from the last layer and the output tensor
        `sub_x` from the sub-layer, adds them, and then normalizes the sum
        using layer normalization.

        Args:
            x (Tensor): The output tensor of the last layer with shape [B, M, d].

            sub_x (Tensor): The output tensor of the sub-layer with shape
            [B, M, d].

        Returns:
            Tensor: The result tensor obtained after normalizing the sum of
            the inputs with shape [B, M, d].

        """
        return self.lnorm(x + sub_x)


class MultiHeadAtt(nn.Module):
    """A module that implements the multi-head attention mechanism described in
    https://arxiv.org/abs/1706.03762.

    Args:
        d_model (int): The dimensionality of the model.

        h (int): The number of heads to use in the attention mechanism.

        masking_value (float, optional): The value used for masking. Defaults
        to -1e15.
    """

    def __init__(self, d_model: int, h: int, masking_value: int = -1e15) -> None:
        super().__init__()
        self.h = h
        self.dk = d_model // h
        self.d_model = d_model
        self.masking_value = masking_value
        assert d_model % h == 0, ValueError
        self.query_fc = nn.Linear(in_features=d_model, out_features=d_model)
        self.key_fc = nn.Linear(in_features=d_model, out_features=d_model)
        self.value_fc = nn.Linear(in_features=d_model, out_features=d_model)
        self.softmax = nn.Softmax(dim=-1)

    def _reshape(self, x: Tensor) -> List[Tensor]:
        batch_size, max_len, _ = x.shape
        x = x.view(batch_size, max_len, self.h, self.dk)
        return x

    def _mask(self, att: Tensor, key_mask: Tensor, query_mask: Tensor) -> Tensor:
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
        key_mask: Optional[Tensor] = None,
        query_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Performs multi-head attention by computing a weighted sum of the
        values using queries and keys. The weights are computed as a softmax
        over the dot products of the queries and keys for each attention head.
        Optionally, attention can be masked using key and query masks.

        Args:
            key (Tensor): The key input tensor of shape [B, M, d]

            query (Tensor): The query of shape [B, M, d]

            value (Tensor): Teh value tensor of shape [B, M, d]

            key_mask (Tensor, optional): A boolean tensor of shape
            [B, M] where True indicates that the corresponding key position
            contains data, not padding, and should not be masked

            query_mask (Tensor, optional): A boolean tensor of shape
            [B, M] where True indicates that the corresponding query position
            contains data, not padding, and should not be masked

        Returns:
            Tensor: The tensor of shape [B, M, d] resulting from the multi-head
            attention computation.
        """
        key = self._reshape(key)  # B, M, h, dk
        query = self._reshape(query)  # B, M, h, dk
        value = self._reshape(value)  # B, M, h, dk
        key = key.permute(0, 2, 3, 1)  # B, h, dk, M
        query = query.permute(0, 2, 1, 3)  # B, h, M, dk
        value = value.permute(0, 2, 1, 3)  # B, h, M, dk
        att = torch.matmul(query, key)
        if key_mask is not None and query_mask is not None:
            att = self._mask(att=att, key_mask=key_mask, query_mask=query_mask)
        att = self.softmax(att / self.d_model)
        out = torch.matmul(att, value)
        out = out.permute(0, 2, 1, 3)
        out = out.contiguous()
        out = out.view(out.shape[0], out.shape[1], -1)
        return out

    def forward(
        self,
        key: Tensor,
        query: Tensor,
        value: Tensor,
        key_mask: Union[Tensor, None] = None,
        query_mask: Union[Tensor, None] = None,
    ) -> Tensor:
        """passes the input to the multi-head attention by computing a weighted
        sum of the values using queries and keys. The weights are computed as a softmax
        over the dot products of the queries and keys for each attention head.
        Optionally, attention can be masked using key and query masks.

        Args:
            key (Tensor): The key input tensor of shape [B, M, d]

            query (Tensor): The query of shape [B, M, d]

            value (Tensor): Teh value tensor of shape [B, M, d]

            key_mask (Tensor, optional): A boolean tensor of shape
            [B, M] where True indicates that the corresponding key position
            contains data, not padding, and should not be masked

            query_mask (Tensor, optional): A boolean tensor of shape
            [B, M] where True indicates that the corresponding query position
            contains data, not padding, and should not be masked

        Returns:
            Tensor: The tensor of shape [B, M, d] resulting from the multi-head
            attention computation.
        """
        key = self.key_fc(key)
        query = self.query_fc(query)
        value = self.value_fc(value)
        return self.perform_attention(
            key=key, query=query, value=value, key_mask=key_mask, query_mask=query_mask
        )


class MaskedMultiHeadAtt(MultiHeadAtt):
    """A multi-head attention module that performs masking to handle padded sequences.
    This implementation is based on the architecture described in https://arxiv.org/abs/1706.03762

    Args:

        d_model (int): The model dimensionality.

        h (int): The number of heads in the attention mechanism.

        masking_value (float, optional): The value to use for masking padded
        elements. Defaults to -1e15.
    """

    def __init__(self, d_model: int, h: int, masking_value: float = -1e15) -> None:
        super().__init__(d_model=d_model, h=h, masking_value=masking_value)

    def get_looking_ahead_mask(self, key_mask: Tensor) -> Tensor:
        batch_size, max_len = key_mask.shape
        query_mask = torch.tril(torch.ones(batch_size, max_len, max_len))
        query_mask = query_mask.bool()
        query_mask = query_mask.to(key_mask.device)
        query_mask &= key_mask.unsqueeze(dim=-1) & query_mask
        return query_mask

    def forward(
        self,
        key: Tensor,
        query: Tensor,
        value: Tensor,
        key_mask: Union[Tensor, None],
    ) -> Tensor:
        """Applies masked multi-head attention to the input.

        Args:
            key (Tensor): The key input tensor of shape [B, M, d].

            query (Tensor): The query input tensor of shape [B, M, d].

            value (Tensor): The value input tensor of shape [B, M, d].

            key_mask (Union[Tensor, None]): The mask tensor of the key of shape
            [B, M] where True indicates that the corresponding key position
            contains data not padding and therefore should not be masked.
            If None, the function will act as a normal multi-head attention.

        Returns:
            Tensor: The attention result tensor of shape [B, M, d].
        """

        query_mask = None
        if key_mask is not None:
            query_mask = self.get_looking_ahead_mask(key_mask=key_mask)
        return super().forward(
            key=key, query=query, value=value, key_mask=key_mask, query_mask=query_mask
        )


class TransformerEncLayer(nn.Module):
    """Implements a single layer of the transformer encoder model as
    presented in the paper https://arxiv.org/abs/1706.03762

    Args:
        d_model (int): The model dimensionality.

        ff_size (int): The feed forward inner layer dimensionality.

        h (int): The number of heads in the attention mechanism.

        masking_value (float, optional): The value to use for masking padded
        elements. Defaults to -1e15.

    """

    def __init__(
        self, d_model: int, ff_size: int, h: int, masking_value: int = -1e15
    ) -> None:
        super().__init__()
        self.mhsa = MultiHeadAtt(d_model=d_model, h=h, masking_value=masking_value)
        self.add_and_norm1 = AddAndNorm(d_model=d_model)
        self.ff = FeedForwardModule(d_model=d_model, ff_size=ff_size)
        self.add_and_norm2 = AddAndNorm(d_model=d_model)

    def forward(self, x: Tensor, mask: Union[Tensor, None] = None) -> Tensor:
        """Performs a forward pass of the transformer encoder layer.

        Args:

            x (Tensor): The input tensor of shape [B, M, d].

            mask (Union[Tensor, None], optional): Boolean tensor of the input of shape
            [B, M] where True indicates that the corresponding key position
            contains data not padding and therefore should not be masked.
            If None, the function will act as a normal multi-head attention. Defaults to None.

        Returns:
            Tensor: Result tensor of the same shape as x.
        """
        out = self.mhsa(key=x, query=x, value=x, key_mask=mask, query_mask=mask)
        out = self.add_and_norm1(x, out)
        result = self.ff(out)
        return self.add_and_norm2(out, result)


class RowConv1D(nn.Module):
    """Implements the row convolution module
    proposed in https://arxiv.org/abs/1512.02595

    Args:

        tau (int): The size of future context.

        feat_size (int): The size of the input feature.

    """

    def __init__(self, tau: int, feat_size: int) -> None:
        super().__init__()
        self.tau = tau
        self.conv = nn.Conv1d(
            in_channels=feat_size,
            out_channels=feat_size,
            kernel_size=tau,
            stride=1,
            padding=0,
            dilation=1,
        )

    def _pad(self, x: Tensor):
        """pads the input with zeros along the
        time dim.

        Args:
            x (Tensor): The input tensor of shape [B, d, M].

        Returns:
            Tensor: The padded tensor.
        """
        zeros = torch.zeros(*x.shape[:-1], self.tau)
        zeros = zeros.to(x.device)
        return torch.cat([x, zeros], dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input tensor x through the row convolution layer.

        Args:
            x (Tensor): The input tensor of shape [B, M, feat_size].

        Returns:
            Tensor: The result tensor of the same shape [B, M, feat_size].
        """
        max_len = x.shape[1]
        x = x.transpose(1, 2)
        x = self._pad(x)
        out = self.conv(x)
        # remove the conv on the padding if there is any
        out = out[..., :max_len]
        out = out.transpose(1, 2)
        return out


class Conv1DLayers(nn.Module):
    """Implements stack of Conv1d layers.

    Args:

        in_size (int): The input feature size.

        out_size (Union[List[int], int]): The output feature size(s) of each
        layer. If a list is passed, it has to be of length equal to `n_layers`.

        kernel_size (Union[List[int], int]): The kernel size(s) of the Conv1d
        layers. If a list is passed, it has to be of length equal to `n_layers`.

        stride (Union[List[int], int]): The stride size(s) of the Conv1d layers.
        If a list is passed, it has to be of length equal to `n_layers`.

        n_layers (int): The number of Conv1d layers to stack.

        p_dropout (float): The dropout rate.

        groups (Union[List[int], int]): The groups size of the conv layers, if
        a list is passed it has to be of length equal to n_layers. Default 1.

    """

    def __init__(
        self,
        in_size: int,
        out_size: Union[List[int], int],
        kernel_size: Union[List[int], int],
        stride: Union[List[int], int],
        n_layers: int,
        p_dropout: float,
        groups: Union[List[int], int] = 1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        _kernel_size = kernel_size
        _stride = stride
        _groups = groups
        for i in range(n_layers):
            in_channels = out_size
            if i == 0:
                in_channels = in_size
            elif isinstance(out_size, list):
                in_channels = out_size[i - 1]

            out_channels = out_size
            if isinstance(out_size, list):
                out_channels = out_size[i]

            if isinstance(kernel_size, list):
                _kernel_size = kernel_size[i]

            if isinstance(stride, list):
                _stride = stride[i]

            if isinstance(groups, list):
                _groups = groups[i]

            self.layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=_kernel_size,
                    stride=_stride,
                    groups=_groups,
                )
            )
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor, data_len: Tensor) -> Tuple[Tensor, Tensor]:
        """Passes the input tensor x through the Conv1D layers and returns the
        result.

        Args:
            x (Tensor): The input tensor of shape [B, M, in_size].

            data_len (Tensor):  A tensor of shape [B] containing the length of
            each sequence in x.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the result tensor of shape
            [B, M, out_size] and a tensor of shape [B] containing the new length
            of each sequence after applying the conv layers.

        """

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
                stride=layer.stride[0],
            )
            pad_len = result_len - data_len
        out = out.transpose(1, 2)
        return out, data_len


class GlobalMulAttention(nn.Module):
    """Implements the global multiplicative attention mechanism as described
    in https://arxiv.org/abs/1508.04025, using direct dot product for scoring.

    Args:
        enc_feat_size (int): The size of encoder features.

        dec_feat_size (int): The size of decoder features.

        scaling_factor (Union[float, int]): The scaling factor for numerical
        stability used inside the softmax. Default: 1.

        mask_val (float): the masking value. Default -1e12.

    """

    def __init__(
        self,
        enc_feat_size: int,
        dec_feat_size: int,
        scaling_factor: Union[float, int] = 1,
        mask_val: float = -1e12,
    ) -> None:
        super().__init__()
        self.fc_query = nn.Linear(in_features=dec_feat_size, out_features=dec_feat_size)
        self.fc_key = nn.Linear(in_features=enc_feat_size, out_features=dec_feat_size)
        self.fc_value = nn.Linear(in_features=enc_feat_size, out_features=dec_feat_size)
        self.fc = nn.Linear(in_features=2 * dec_feat_size, out_features=dec_feat_size)
        self.scaling_factor = scaling_factor
        self.mask_val = mask_val

    def forward(
        self, key: Tensor, query: Tensor, mask: Union[None, Tensor] = None
    ) -> Tensor:
        """Applies the global multiplicative attention mechanism
        to the input key and query.

        Args:
            key (Tensor): The key tensor of shape [B, M, enc_feat_size].

            query (Tensor): The query tensor of shape [B, 1, dec_feat_size].

            mask (Union[None, Tensor], optional): The boolean mask tensor of shape
            [B, M], where False for padding. Default None.

        Returns:
            Tensor: The attention tensor of shape [B, enc_feat_size].
        """
        value = self.fc_value(key)
        key = self.fc_key(key)
        query = self.fc_query(query)
        att_weights = torch.matmul(query, key.transpose(-1, -2))
        if mask is not None:
            mask = mask.unsqueeze(dim=-2)
            att_weights = att_weights.masked_fill(~mask, self.mask_val)
        att_weights = torch.softmax(att_weights / self.scaling_factor, dim=-1)
        context = torch.matmul(att_weights, value)
        results = torch.cat([context, query], dim=-1)
        results = self.fc(results)
        results = torch.tanh(results)
        return results


class ConformerFeedForward(nn.Module):
    """Implements the feed-forward module used in Conformer models
    as described in https://arxiv.org/abs/2005.08100

    Args:
        d_model (int): The input feature dimensionality.

        expansion_factor (int): The expansion factor used by the linear layer.

        p_dropout (float): The dropout rate.
    """

    def __init__(self, d_model: int, expansion_factor: int, p_dropout: float) -> None:
        super().__init__()
        self.lnrom = nn.LayerNorm(normalized_shape=d_model)
        self.fc1 = nn.Linear(
            in_features=d_model, out_features=expansion_factor * d_model
        )
        self.fc2 = nn.Linear(
            in_features=expansion_factor * d_model, out_features=d_model
        )
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input x through the conformer feed-forward module.

        Args:
            x (Tensor): The input tensor of shape [B, M, d].

        Returns:
            Tensor: The output tensor of shape [B, M, d].
        """
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

    def __init__(self, d_model: int, kernel_size: int, p_dropout: float) -> None:
        super().__init__()
        self.lnorm = nn.LayerNorm(normalized_shape=d_model)
        self.pwise_conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=2 * d_model, kernel_size=1
        )
        self.act1 = nn.GLU(dim=1)
        self.dwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding="same",
        )
        self.bnorm = nn.BatchNorm1d(num_features=d_model)
        self.act2 = nn.SiLU()
        self.pwise_conv2 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1
        )
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Passes the input tensor through the Conformer Convolutional Module.

        Args:
            x (Tensor): Input tensor of shape [B, M, d].

        Returns:
            Tensor: Result tensor of shape [B, M, d].
        """

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
    """Multi-Head Self-Attention module with relative positional encoding,
    based on the paper https://arxiv.org/abs/2005.08100

    Args:
        d_model (int): The input and output feature dimension.

        h (int): The number of attention heads.

        p_dropout (float): The dropout rate.

        masking_value (int): The masking value used for padding. Default -1e15.
    """

    def __init__(
        self, d_model: int, h: int, p_dropout: float, masking_value: int = -1e15
    ) -> None:
        super().__init__(d_model=d_model, h=h, masking_value=masking_value)
        self.lnrom = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor, mask: Union[None, Tensor] = None) -> Tensor:
        """Performs Multi-Head Self-Attention operation with relative positional
        encoding on input tensor x.

        Args:

            x (Tensor): Input tensor of shape [B, M, d].

            mask (Tensor, optional): Boolean tensor of shape [B, M], where
            False for padding. If None is provided, no masking is applied.
            Default is None.

        Returns:

            Tensor: Result tensor of shape [B, M, d].

        """
        out = self.lnrom(x)
        out = add_pos_enc(out)
        out = super().forward(
            key=out, query=out, value=out, query_mask=mask, key_mask=mask
        )
        out = self.dropout(out)
        return out


class ConformerBlock(nn.Module):
    """Implements the conformer block described in https://arxiv.org/abs/2005.08100

    Args:

        d_model (int): The model dimension.

        ff_expansion_factor (int): The expansion factor of the linear layer.

        h (int): The number of heads.

        kernel_size (int): The kernel size of depth-wise convolution.

        p_dropout (float): The dropout rate.

        res_scaling (float): The multiplier for residual connection.
    """

    def __init__(
        self,
        d_model: int,
        ff_expansion_factor: int,
        h: int,
        kernel_size: int,
        p_dropout: float,
        res_scaling: float = 0.5,
    ) -> None:
        super().__init__()
        self.ff1 = ConformerFeedForward(
            d_model=d_model, expansion_factor=ff_expansion_factor, p_dropout=p_dropout
        )
        self.mhsa = ConformerRelativeMHSA(d_model=d_model, h=h, p_dropout=p_dropout)
        self.conv = ConformerConvModule(
            d_model=d_model, kernel_size=kernel_size, p_dropout=p_dropout
        )
        self.ff2 = ConformerFeedForward(
            d_model=d_model, expansion_factor=ff_expansion_factor, p_dropout=p_dropout
        )
        self.lnrom = nn.LayerNorm(normalized_shape=d_model)
        self.res_scaling = res_scaling

    def forward(self, x: Tensor, mask: Union[None, Tensor] = None) -> Tensor:
        """Passes the input to the conformer block.

        Args:

            x (torch.Tensor): The input tensor of shape [B, M, d].

            mask (Tensor, optional): Boolean tensor of shape [B, M], where
            False for padding. If None is provided, no masking is applied.
            Default is None.

        Returns:

            Tensor: The output tensor of the same shape as the input tensor `x`.
        """
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

        kernel_size (Union[int, List[int]]): The kernel size of the subsampling layer.

        stride (Union[int, List[int]]): The stride of the subsampling layer.

        n_conv_layers (int): The number of convolutional layers.

        d_model (int): The model dimension.

        p_dropout (float): The dropout rate.

        groups (Union[int, List[int]]): The convolution groups size. Default 1.
    """

    def __init__(
        self,
        in_features: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]],
        n_conv_layers: int,
        d_model: int,
        p_dropout: float,
        groups: Union[int, List[int]] = 1,
    ) -> None:
        super().__init__()
        self.layers = Conv1DLayers(
            in_size=in_features,
            out_size=d_model,
            kernel_size=kernel_size,
            stride=stride,
            n_layers=n_conv_layers,
            p_dropout=p_dropout,
            groups=groups,
        )
        self.fc = nn.Linear(in_features=d_model, out_features=d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` to the pre-conformer blocks that contains
        the subsampling convolutional.

        Args:

            x (Tensor): The input tensor of shape [B, M, d].

            lengths (Tensor): A tensor of shape [B] containing the lengths of
            each sequence in `x` before subsampling.

        Returns:

            Tuple[Tensor, Tensor]: A tuple containing two tensors. The first
            tensor is the output of the pre-conformer block of shape [B, N, d].
            The second tensor is a tensor of shape [B] containing the lengths of
            each sequence in the output tensor after subsampling.
        """
        out, lengths = self.layers(x, lengths)
        out = self.fc(out)
        out = self.dropout(out)
        return out, lengths


class JasperSubBlock(nn.Module):
    """Implements the subblock of the Jasper model as described in
    https://arxiv.org/abs/1904.03288

    Args:

        in_channels (int): The number of input channels.

        out_channels (int): The number of output channels.

        kernel_size (int): The kernel size of the convolutional layer.

        p_dropout (float): The dropout rate.

        stride (int): The stride of the convolutional layer. Default is 1.

        padding (Union[str, int]): The padding mode or size. Default is 'same'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        p_dropout: float,
        stride: int = 1,
        padding: Union[str, int] = "same",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.bnorm = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor, residual: Union[Tensor, None] = None) -> Tensor:
        """Passes the input to the layer

        Args:

            x (Tensor): The input tensor of shape [B, d, M].

            residual (Union[Tensor, None], optional): An optional tensor of shape
            [B, out_channels, M]. If not None, it is added element-wise to the
            output tensor. Defaults to None.

        Returns:

            Tensor: The output tensor of shape [B, out_channels, M].

        """

        out = self.conv(x)
        out = self.bnorm(out)
        if residual is not None:
            out = out + residual
        out = self.relu(out)
        out = self.dropout(out)
        return out


class JasperResidual(nn.Module):
    """Implements the residual connection module described in https://arxiv.org/abs/1904.03288

    Args:

        in_channels (int): The number of input channels.

        out_channels (int): The number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        self.bnorm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input x through the residual branch.

        Args:
            x (Tensor): The input tensor of shape [B, in_channels, M]

        Returns:

            Tensor: The result tensor of shape [B, out_channels, M]
        """
        out = self.conv(x)
        out = self.bnorm(out)
        return out


class JasperBlock(nn.Module):
    """Implements the main jasper block of the Jasper model as described in
    https://arxiv.org/abs/1904.03288

    Args:

        num_sub_blocks (int): The number of subblocks, which is denoted as
        'R' in the paper.

        in_channels (int): The number of input channels.

        out_channels (int): The number of output channels.

        kernel_size (int): The kernel size of the convolutional layer.

        p_dropout (float): The dropout rate.
    """

    def __init__(
        self,
        num_sub_blocks: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                JasperSubBlock(
                    in_channels=in_channels if i == 1 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    p_dropout=p_dropout,
                )
                for i in range(1, 1 + num_sub_blocks)
            ]
        )
        self.residual_layer = JasperResidual(
            in_channels=in_channels, out_channels=out_channels
        )
        self.num_sub_blocks = num_sub_blocks

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input x through the layer.

        Args:
            x (Tensor): The input tensor of shape [B, in_channels, M]

        Returns:

            Tensor: The result tensor of shape [B, out_channels, M]
        """
        out = x
        for i, block in enumerate(self.blocks):
            if (i + 1) == self.num_sub_blocks:
                out = block(out, residual=self.residual_layer(x))
            else:
                out = block(out)
        return out


class JasperBlocks(nn.Module):
    """Implements the jasper's series of blocks as described in
    https://arxiv.org/abs/1904.03288

    Args:

        num_blocks (int): The number of Jasper blocks (denoted as 'B' in the paper).

        num_sub_blocks (int): The number of Jasper subblocks (denoted as 'R' in the paper).

        in_channels (int): The number of input channels.

        channel_inc (int): The rate to increase the number of channels across the blocks.

        kernel_size (Union[int, List[int]]): The kernel size(s) of the convolution layer for each block.

        p_dropout (float): The dropout rate.
    """

    def __init__(
        self,
        num_blocks: int,
        num_sub_blocks: int,
        in_channels: int,
        channel_inc: int,
        kernel_size: Union[int, List[int]],
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                JasperBlock(
                    num_sub_blocks=num_sub_blocks,
                    in_channels=in_channels + channel_inc * i,
                    out_channels=in_channels + channel_inc * (1 + i),
                    kernel_size=kernel_size
                    if isinstance(kernel_size, int)
                    else kernel_size[i],
                    p_dropout=p_dropout,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input tensor through the JasperBlocks layer.

        Args:
            x (Tensor): The input tensor of shape [B, in_channels, M].

        Returns:

            Tensor: The output tensor of shape [B, in_channels + channel_inc * num_blocks, M].
                This tensor is the result of applying the JasperBlocks layer to the input tensor x.

        """
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class LocAwareGlobalAddAttention(nn.Module):
    """Implements the location-aware global additive attention proposed in
    https://arxiv.org/abs/1506.07503

    Args:
        enc_feat_size (int): The encoder feature size.

        dec_feat_size (int): The decoder feature size.

        kernel_size (int): The size of the attention kernel.

        activation (str): The activation function to use. Can be either 'softmax' or 'sigmoid'.

        inv_temperature (Union[float, int], optional): The value of the inverse temperature parameter. Default is 1.

        mask_val (float, optional): The masking value for the attention weights. Default is -1e12.

    """

    def __init__(
        self,
        enc_feat_size: int,
        dec_feat_size: int,
        kernel_size: int,
        activation: str,
        inv_temperature: Optional[Union[float, int]] = 1,
        mask_val: Optional[float] = -1e12,
    ) -> None:
        super().__init__()
        activations = {"softmax": nn.Softmax, "sigmax": Sigmax}
        assert activation in activations
        self.activation = activations[activation](dim=-2)
        self.fc_query = nn.Linear(in_features=dec_feat_size, out_features=dec_feat_size)
        self.fc_key = nn.Linear(in_features=enc_feat_size, out_features=dec_feat_size)
        self.fc_value = nn.Linear(in_features=enc_feat_size, out_features=dec_feat_size)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=dec_feat_size,
            kernel_size=kernel_size,
            padding="same",
        )
        self.pos_fc = nn.Linear(in_features=dec_feat_size, out_features=dec_feat_size)
        self.w = nn.parameter.Parameter(data=torch.randn(dec_feat_size, 1))
        self.mask_val = mask_val
        self.inv_temperature = inv_temperature

    def forward(
        self, key: Tensor, query: Tensor, alpha: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Computes the forward pass of the location-aware global additive attention mechanism.

        Args:

            key (Tensor): The encoder feature maps of shape [B, M_enc, enc_feat_size].

            query (Tensor): The decoder feature maps of shape [B, 1, dec_feat_size].

            alpha (Tensor): The previous attention weights of shape [B, 1, M_enc].

            mask (Tensor, optional): The mask tensor of shape [B, M_enc], with zeros/False in the
                                     positions that should be masked. Default is None.

        Returns:
            A tuple of two tensors:

            - context (Tensor): The context tensor of shape [B, 1, M_dec].
            - attn_weights (Tensor): The attention weights tensor of shape [B, 1, M_enc].
        """
        value = self.fc_value(key)
        key = self.fc_key(key)
        query = self.fc_query(query)
        f = self.conv(alpha)  # [B, d, M_enc]
        f = f.transpose(-1, -2)
        f = self.pos_fc(f)
        # [B, 1, d] + [B, M_enc,  d] +  [B, M_enc, d]
        e = torch.tanh(query + key + f)  # [B, M_dec, d]
        att_weights = torch.matmul(e, self.w)
        if mask is not None:
            mask = mask.unsqueeze(dim=-1)
            att_weights = att_weights.masked_fill(~mask, self.mask_val)
        att_weights = self.activation(att_weights * self.inv_temperature)
        att_weights = att_weights.transpose(-1, -2)
        context = torch.matmul(att_weights, value)
        return context, att_weights


class MultiHeadAtt2d(MultiHeadAtt):
    """Implements the 2-dimensional multi-head self-attention
    proposed in https://ieeexplore.ieee.org/document/8462506

    Args:

        d_model (int): The input feature dimensionality.

        h (int): The number of attention heads.

        out_channels (int): The number of output channels of the convolution layer.

        kernel_size (int): The size of the convolutional kernel to apply.

    """

    def __init__(
        self, d_model: int, h: int, out_channels: int, kernel_size: int
    ) -> None:
        super().__init__(out_channels, h)
        assert out_channels % h == 0
        self.query_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.key_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.value_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.fc = nn.Linear(in_features=2 * out_channels, out_features=d_model)
        del self.query_fc, self.key_fc, self.value_fc

    def perform_frequency_attention(
        self,
        key: Tensor,
        query: Tensor,
        value: Tensor,
    ) -> Tensor:
        """
        Applies frequency-domain multi-head self-attention.

        Args:
            key (Tensor): A tensor of shape [B, M, d].
            query (Tensor): A tensor of shape [B, M, d].
            value (Tensor): A tensor of shape [B, M, d].

        Returns:
            Tensor: A tensor of shape [B, M, d], representing the result
            after performing the attention mechanism on the frequency domain.
        """
        key = self._reshape(key)  # B, M, h, dk
        query = self._reshape(query)  # B, M, h, dk
        value = self._reshape(value)  # B, M, h, dk
        key = key.permute(0, 2, 1, 3)  # B, h, M, dk
        query = query.permute(0, 2, 3, 1)  # B, h, dk, M
        value = value.permute(0, 2, 3, 1)  # B, h, dk, M
        att = self.softmax(torch.matmul(query, key) / self.d_model)
        out = torch.matmul(att, value)
        out = out.permute(0, 3, 2, 1)
        out = out.contiguous()
        out = out.view(out.shape[0], out.shape[1], -1)
        return out

    def forward(
        self,
        key: Tensor,
        query: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Applies both time-domain and frequency-domain multi-head self-attention
        on the input.

        Args:

            key (Tensor): A tensor of shape [B, M,d].

            query (Tensor): A tensor of shape [B, M,d].

            value (Tensor): A tensor of shape [B, M,d].

            mask (Tensor, optional): Boolean tensor of shape [B, M], where
            False for padding. If None is provided, no masking is applied.
            Default is None.

        Returns:
            Tensor: The result tensor of shape [B, M, d].
        """
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
            key=key, query=query, value=value, query_mask=mask, key_mask=mask
        )
        freq_att_result = self.perform_frequency_attention(
            key=key, query=query, value=value
        )
        result = torch.cat([time_att_result, freq_att_result], dim=-1)
        result = self.fc(result)
        return result


class SpeechTransformerEncLayer(TransformerEncLayer):
    """Implements a single encoder layer of the speech transformer
    as described in https://ieeexplore.ieee.org/document/8462506

    Args:
        d_model (int): The model dimensionality.

        ff_size (int): The dimensionality of the inner layer of the feed-forward module.

        h (int): The number of attention heads.

        out_channels (int): The number of output channels of the convolution layer.

        kernel_size (int): The kernel size of the convolutional layers.
    """

    def __init__(
        self, d_model: int, ff_size: int, h: int, out_channels: int, kernel_size: int
    ) -> None:
        # TODO: pass masking value
        super().__init__(d_model=d_model, ff_size=ff_size, h=h)
        del self.add_and_norm2
        self.mhsa = MultiHeadAtt2d(
            d_model=d_model, h=h, out_channels=out_channels, kernel_size=kernel_size
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Passes the input tensor `x` through a single encoder layer of the speech
        transformer.

        Args:

            x (Tensor): The input tensor of shape [B, M, d].

            mask (Tensor, optional): The mask tensor of shape [B, M],
            or None if no mask is needed. Default None.

        Returns:

            Tensor: The output tensor of shape [B, B, d].

        """
        out = self.layer_norm(x)
        out = self.mhsa(key=out, query=out, value=out, mask=mask)
        out = self.add_and_norm1(x, out)
        result = self.ff(out)
        return out + result


class TransformerDecLayer(nn.Module):
    """Implements a single decoder layer of the transformer
    as described in https://arxiv.org/abs/1706.03762

    Args:

        d_model (int): The model dimensionality.

        ff_size (int): The feed forward inner layer dimensionality.

        h (int): The number of attention heads.

        masking_value (int): The masking value. Default -1e15
    """

    def __init__(
        self, d_model: int, ff_size: int, h: int, masking_value: int = -1e15
    ) -> None:
        super().__init__()
        self.mmhsa = MaskedMultiHeadAtt(
            d_model=d_model, h=h, masking_value=masking_value
        )
        self.add_and_norm1 = AddAndNorm(d_model=d_model)
        self.mha = MultiHeadAtt(d_model=d_model, h=h, masking_value=masking_value)
        self.add_and_norm2 = AddAndNorm(d_model=d_model)
        self.ff = FeedForwardModule(d_model=d_model, ff_size=ff_size)
        self.add_and_norm3 = AddAndNorm(d_model=d_model)

    def forward(
        self,
        enc_out: Tensor,
        enc_mask: Union[Tensor, None],
        dec_inp: Tensor,
        dec_mask: Union[Tensor, None],
    ) -> Tensor:
        """Applies a single decoder layer of the transformer to the input.

        Args:
            enc_out (Tensor): The output of the encoder. Its shape is [B, M_enc, d].

            enc_mask (Tensor, optional): The mask tensor for the encoder output.
            Its shape is [B, M_enc], where it is False for the padding positions.

            dec_inp (Tensor): The input to the decoder layer. Its shape is
            [B, M_dec, d_model].

            dec_mask (Tensor, optional): The mask tensor for the decoder input.
            Its shape is [B, M_dec], where it is False for the padding.

        Returns:
            The output of the decoder layer. Its shape is [B, M_dec, d_model].

        """
        out = self.mmhsa(key=dec_inp, query=dec_inp, value=dec_inp, key_mask=dec_mask)
        out = self.add_and_norm1(out, dec_inp)
        out = self.add_and_norm2(
            self.mha(
                key=enc_out,
                query=out,
                value=enc_out,
                key_mask=enc_mask,
                query_mask=dec_mask,
            ),
            out,
        )
        out = self.add_and_norm3(self.ff(out), out)
        return out


class SpeechTransformerDecLayer(TransformerDecLayer):
    """Implements a single decoder layer of the speech transformer
    as described in https://ieeexplore.ieee.org/document/8462506

    Args:

        d_model (int): The model dimensionality.

        ff_size (int): The feed forward inner layer dimensionality.

        h (int): The number of attention heads.

        masking_value (int): The masking value. Default -1e15
    """

    def __init__(
        self, d_model: int, ff_size: int, h: int, masking_value: int = -1e15
    ) -> None:
        super().__init__(d_model, ff_size, h, masking_value)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        del self.add_and_norm3

    def forward(
        self,
        enc_out: Tensor,
        enc_mask: Union[Tensor, None],
        dec_inp: Tensor,
        dec_mask: Union[Tensor, None],
    ) -> Tensor:
        """Applies a single decoder layer of speech transformer to the input.

        Args:
            enc_out (Tensor): The output of the encoder. Its shape is [B, M_enc, d].

            enc_mask (Tensor, optional): The mask tensor for the encoder output.
            Its shape is [B, M_enc], where it is False for the padding positions.

            dec_inp (Tensor): The input to the decoder layer. Its shape is
            [B, M_dec, d_model].

            dec_mask (Tensor, optional): The mask tensor for the decoder input.
            Its shape is [B, M_dec], where it is False for the padding.

        Returns:
            The output of the decoder layer. Its shape is [B, M_dec, d_model].

        """
        out = self.layer_norm(dec_inp)
        out = self.mmhsa(key=out, query=out, value=out, key_mask=dec_mask)
        out = self.add_and_norm1(out, dec_inp)
        out = self.add_and_norm2(
            self.mha(
                key=enc_out,
                query=out,
                value=enc_out,
                key_mask=enc_mask,
                query_mask=dec_mask,
            ),
            out,
        )
        out = self.ff(out) + out
        return out


class PositionalEmbedding(nn.Module):
    """Implements the positional embedding proposed in
    https://arxiv.org/abs/1706.03762

    output = positional_encoding + Embedding(input)

    Args:
        vocab_size (int): The vocabulary size.

        embed_dim (int): The embedding size.

    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.d_model = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        """Applies the positional embedding to the input tensor.

        Args:

            x (Tensor): The input tensor of shape [B, M].

        Returns:

            Tensor: The output tensor of shape [B, M, d].

        """
        out = self.emb(x)
        out = add_pos_enc(out)
        return out


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
        """Applies the group shuffle on the input tensor `x`.

        Args:

            x (Tensor): The input tensor of shape [B, C, ...].

        Returns:

            Tensor: The output tensor after applying group shuffle of shape [B, C, ...].
        """
        batch_size, channels, *_ = x.shape
        dims = x.shape[2:]
        x = x.view(batch_size, self.groups, channels // self.groups, *dims)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(batch_size, channels, *dims)
        return x


class QuartzSubBlock(JasperSubBlock):
    """Implements the subblock module of Quartznet as described in https://arxiv.org/abs/1910.10261

    Args:

        in_channels (int): The number of channels of the input.

        out_channels (int): The number of channels of the output.

        kernel_size (int): The kernel size of the convolution layer.

        p_dropout (float): The dropout rate.

        groups (int): The number of groups in the convolution layer.

        stride (int): The stride of the convolution layer. Default is 1.

        padding (Union[str, int]): The padding mode or size. Default is 'same'.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        p_dropout: float,
        groups: int,
        stride: int = 1,
        padding: Union[str, int] = "same",
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, p_dropout, stride, padding
        )
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                groups=groups,
            ),
            GroupsShuffle(groups=groups),
        )
        self.dwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            padding="same",
        )

    def forward(self, x: Tensor, residual: Union[Tensor, None] = None) -> Tensor:
        """The forward method applies the Quartznet's subblock module to the input tensor
        x and an optional residual tensor.

        Args:

            x (Tensor): The input tensor of shape [B, in_channels, M].

            residual (Tensor, optional): The residual tensor of shape [B, out_channels, M]. Default is None.

        Returns:
            Tensor: The output tensor of shape [B, out_channels, M].
        """
        x = self.dwise_conv(x)
        return super().forward(x=x, residual=residual)


class QuartzBlock(JasperBlock):
    """Implements the main block of the QuartzNet model as described
    in https://arxiv.org/abs/1904.03288

    Args:

        num_sub_blocks (int): Number of subblocks, denoted as 'R' in the paper.

        in_channels (int): Number of input channels of the convolution layer.

        out_channels (int): Number of output channels of the convolution layer.

        kernel_size (int): Convolution layer's kernel size.

        groups (int): Group size for the convolution layer.

        p_dropout (float): Dropout rate.

    """

    def __init__(
        self,
        num_sub_blocks: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        p_dropout: float,
    ) -> None:
        super().__init__(
            num_sub_blocks, in_channels, out_channels, kernel_size, p_dropout
        )
        self.blocks = nn.ModuleList(
            [
                QuartzSubBlock(
                    in_channels=in_channels if i == 1 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    groups=groups,
                    p_dropout=p_dropout,
                )
                for i in range(1, 1 + num_sub_blocks)
            ]
        )


class QuartzBlocks(JasperBlocks):
    """Implements the series of blocks in the QuartzNet model, as described in
    https://arxiv.org/abs/1910.10261

    Args:

        num_blocks (int): The number of QuartzNet blocks, denoted as 'B' in the paper.

        block_repetition (int): The number of times to repeat each block, denoted as 'S' in the paper.

        num_sub_blocks (int): The number of QuartzNet subblocks, denoted as 'R' in the paper.

        in_channels (int): The number of channels in the input.

        channels_size (List[int]): A list of integers representing the number of output channels
        for each block.

        kernel_size (Union[int, List[int]]): An integer or a list of integers representing the
        kernel size(s) for each block's convolutional layer.

        groups (int): The group size.

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
        p_dropout: float,
    ) -> None:
        super().__init__(
            num_blocks=num_blocks,
            num_sub_blocks=num_sub_blocks,
            in_channels=in_channels,
            channel_inc=0,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
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
                            in_channels=channel_size if j == 0 else channels_size[i],
                            out_channels=channels_size[i],
                            kernel_size=kernel_size
                            if isinstance(kernel_size, int)
                            else kernel_size[i],
                            groups=groups,
                            p_dropout=p_dropout,
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
        self.gamma = nn.Parameter(torch.randn(1, 1, d_model))
        self.beta = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: Tensor) -> Tensor:
        """Scales the input x.

        Args:
            x (Tensor): The input tensor of shape [B, M, d].

        Returns:
            Tensor: The scaled and shifted tensor of shape [B, M, d].

        """
        return self.gamma * x + self.beta


class SqueezeformerConvModule(ConformerConvModule):
    """Implements the conformer convolution module with the modification as described in
    https://arxiv.org/abs/2206.00888

    Args:

        d_model (int): The model dimension.

        kernel_size (int): The size of the depth-wise convolution kernel.

        p_dropout (float): The dropout rate.

    """

    def __init__(self, d_model: int, kernel_size: int, p_dropout: float) -> None:
        super().__init__(d_model, kernel_size, p_dropout)
        self.pwise_conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1
        )
        self.act1 = nn.SiLU()
        self.scaler = Scaling1d(d_model=d_model)
        del self.lnorm

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input x through the layers of the SqueezeformerConvModule.

        Args:

            x (torch.Tensor): A tensor of shape [B, M, d].

        Returns:
            Tensor: The result tensor of shape [B, M, d]
        """
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

        h (int): The number of attention heads.

        p_dropout (float): The dropout rate.

        masking_value (int): The masking value. Default -1e15

    """

    def __init__(
        self, d_model: int, h: int, p_dropout: float, masking_value: int = -1e15
    ) -> None:
        super().__init__(d_model=d_model, h=h, masking_value=masking_value)
        self.dropout = nn.Dropout(p_dropout)
        self.scaler = Scaling1d(d_model=d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """computes the multi-head self-attention of the input tensor with
        optional mask tensor.

        Args:

            x (Tensor): The input tensor of shape [B, M, d].

            mask (Tensor, optional): Boolean tensor of shape [B, M], where
            it's set to False for padding positions. If None is provided, no
            masking is applied. Default is None.

        Returns:
            Tensor: A tensor of shape [B, M, d] representing the output of
            the multi-head self-attention module.
        """
        out = self.scaler(x)
        out = add_pos_enc(out)
        out = super().forward(
            key=out, query=out, value=out, query_mask=mask, key_mask=mask
        )
        out = self.dropout(out)
        return out


class SqueezeformerFeedForward(ConformerFeedForward):
    """Implements the conformer feed-forward module with the modifications
    introduced in https://arxiv.org/abs/2206.00888

    Args:
        d_model (int): The model dimension.

        expansion_factor (int): The expansion factor of the linear layer.

        p_dropout (float): The dropout rate.

    """

    def __init__(self, d_model: int, expansion_factor: int, p_dropout: float) -> None:
        super().__init__(
            d_model=d_model, expansion_factor=expansion_factor, p_dropout=p_dropout
        )
        del self.lnrom
        self.scaler = Scaling1d(d_model=d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input to the feed-forward layers

        Args:
            x (Tensor): Input tensor of shape [B, M, d].

        Returns:
            Tensor: Output tensor of shape [B, M, d].
        """
        out = self.scaler(x)
        out = self.fc1(out)
        out = self.swish(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out


class SqueezeformerBlock(nn.Module):
    """Implements the Squeezeformer block described in
    https://arxiv.org/abs/2206.00888

    Args:
        d_model (int): The model dimension.

        ff_expansion_factor (int): The linear layer's expansion factor.

        h (int): The number of atention heads.

        kernel_size (int): The kernel size of the depth-wise convolution layer.

        p_dropout (float): The dropout rate.

        masking_value (int): The masking value. Default -1e15
    """

    def __init__(
        self,
        d_model: int,
        ff_expansion_factor: int,
        h: int,
        kernel_size: int,
        p_dropout: float,
        masking_value: int = -1e15,
    ) -> None:
        super().__init__()
        self.mhsa = SqueezeformerRelativeMHSA(
            d_model=d_model, h=h, p_dropout=p_dropout, masking_value=masking_value
        )
        self.add_and_norm1 = AddAndNorm(d_model=d_model)
        self.ff1 = SqueezeformerFeedForward(
            d_model=d_model, expansion_factor=ff_expansion_factor, p_dropout=p_dropout
        )
        self.add_and_norm2 = AddAndNorm(d_model=d_model)
        self.conv = SqueezeformerConvModule(
            d_model=d_model, kernel_size=kernel_size, p_dropout=p_dropout
        )
        self.add_and_norm3 = AddAndNorm(d_model=d_model)
        self.ff2 = SqueezeformerFeedForward(
            d_model=d_model, expansion_factor=ff_expansion_factor, p_dropout=p_dropout
        )
        self.add_and_norm4 = AddAndNorm(d_model=d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the Squeezeformer block.

        Args:

            x (Tensor): The input tensor of shape [B, M, d].

            mask (Optional[Tensor]): The optional mask tensor of shape [B, M].
            Default None.

        Returns:

            Tensor: The output tensor of shape [B, M, d].

        """
        out = self.add_and_norm1(self.mhsa(x, mask), x)
        out = self.add_and_norm2(self.ff1(out), out)
        out = self.add_and_norm3(self.conv(out), out)
        out = self.add_and_norm4(self.ff2(out), out)
        return out


class SqueezeAndExcit1D(nn.Module):
    """Implements the squeeze and excite module proposed in https://arxiv.org/abs/1709.01507
    and used in https://arxiv.org/abs/2005.03191

    Args:

        in_feature (int): The number of channels or feature size.

        reduction_factor (int): The feature reduction size.
    """

    def __init__(self, in_feature: int, reduction_factor: int) -> None:
        super().__init__()
        self.swish = nn.SiLU()
        self.fc1 = nn.Linear(
            in_features=in_feature, out_features=in_feature // reduction_factor
        )
        self.fc2 = nn.Linear(
            in_features=in_feature // reduction_factor, out_features=in_feature
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor, mask: Tensor):
        """Applies the squeeze and excite operation to the input tensor.

        Args:

            x (Tensor): The input tensor of shape [B, d, M].

            mask (Tensor): The masking tensor of shape [B, M].

        Returns:
            Tensor: The output tensor of shape [B, d, M] after applying the
            squeeze and excite operation.

        """
        lengths = mask.sum(dim=-1)  # [B]
        x = mask.unsqueeze(dim=1) * x  # zeroing out padded values
        x_pooled = x.sum(dim=-1)  # [B, d]
        x_pooled = x_pooled / lengths.unsqueeze(dim=1)
        x_pooled = self.fc1(x_pooled)
        x_pooled = self.swish(x_pooled)
        x_pooled = self.fc2(x_pooled)
        x_pooled = self.sigmoid(x_pooled)
        x_pooled = x_pooled.unsqueeze(dim=-1)  # [B, d, 1]
        return x_pooled * x


class ContextNetConvLayer(nn.Module):
    """Implements the convolution layer of the ContextNet model proposed in
    https://arxiv.org/abs/2005.03191. This layer applies a convolution operation
    followed by batch normalization and an activation function.

    Args:

        in_channels (int): The number of input channels.

        out_channels (int): The number of output channels.

        kernel_size (int): The convolution layer kernel size.

        stride (int): The stride of the convolution layer. Default 1.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same" if stride == 1 else 0,
            groups=in_channels,
        )
        self.bnorm = nn.BatchNorm1d(num_features=out_channels)
        self.swish = nn.SiLU()

    def forward(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """Passes the input tensor to the ContextNet convolution layer and
        returns a tuple of the output tensor and the updated lengths tensor.

        Args:

            x (Tensor): The input tensor of shape [B, in_channels, M].

            lengths (Tensor): The tensor of shape [B] containing the lengths of
            each sequence in x.

        Returns:

            Tuple[Tensor, Tensor]: A tuple of two tensors. The first tensor is
            the output tensor after applying convolution of shape
            [B, out_channels, N], and the second tensor is the updated lengths
            tensor of shape [B], after applying the convolution layer.

        """
        out = self.conv(x)
        out = self.bnorm(out)
        out = self.swish(out)
        if self.conv.stride[0] != 1:
            lengths = calc_data_len(
                result_len=out.shape[-1],
                pad_len=x.shape[-1] - lengths,
                data_len=lengths,
                kernel_size=self.conv.kernel_size[0],
                stride=self.conv.stride[0],
            )
        return out, lengths


class ContextNetResidual(nn.Module):
    """Implements the residual branch of the ContextNet block
    as proposed in https://arxiv.org/abs/2005.03191

    Args:

        in_channels (int): The number of input channels.

        out_channels (int): The number of output channels.

        kernel_size (int): The convolution kernel size.

        stride (int): The convolution stride size.

    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same" if stride == 1 else 0,
        )
        self.bnorm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x: Tensor, out: Tensor) -> Tensor:
        """
        Args:

            x (Tensor): The input tensor of shape [B, d, M].

            out (Tensor): The output tensor of the previous block of shape [B, d/s, M]
            where s is the stride value. If the block has no stride, s is set to 1.

        Returns:
            Tensor: The output tensor after applying the residual connection of
            shape [B, d, M].

        """
        x = self.conv(x)
        x = self.bnorm(x)
        return x + out


class ContextNetBlock(nn.Module):
    """Implements the convolution block of the ContextNet
    model proposd in https://arxiv.org/abs/2005.03191

    Args:

        n_layers (int): The number of convolutional layers in the block.

        in_channels (int): The number of channels in the input.

        out_channels (int): The number of output channels.

        kernel_size (int): The convolution kernel size.

        reduction_factor (int):The reduction factor for the Squeeze-and-excitation module.

        add_residual (bool):  A flag indicating whether to include a residual connection.

        last_layer_stride (int): The stride size of the last convolutional layer.
    """

    def __init__(
        self,
        n_layers: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        reduction_factor: int,
        add_residual: bool,
        last_layer_stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                ContextNetConvLayer(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1 if i < n_layers - 1 else last_layer_stride,
                )
                for i in range(n_layers)
            ]
        )
        self.squeeze_and_excite = SqueezeAndExcit1D(
            in_feature=out_channels, reduction_factor=reduction_factor
        )
        if add_residual is True:
            self.residual = ContextNetResidual(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=last_layer_stride,
            )
        self.swish = nn.SiLU()
        self.add_residual = add_residual

    def forward(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Passes the input through the convolution block of the ContextNet.

        Args:
            x (Tensor): The input tensor of shape [B, in_channels, M].
            lengths (Tensor): The tensor of shape [B] containing the lengths of each sequence in x.

        Returns:
            Tuple[Tensor, Tensor]: The output tensor after passing through the convolution block,
            of shape [B, out_channels, N], and the updated lengths tensor of shape [B], after
            passing through the convolution block.
        """

        out = x
        for layer in self.conv_layers:
            out, lengths = layer(out, lengths)
        mask = get_mask_from_lens(lengths=lengths, max_len=out.shape[-1])
        out = self.squeeze_and_excite(out, mask)
        if self.add_residual is True:
            out = self.residual(x, out)
        out = self.swish(out)
        return out, lengths


class CausalVGGBlock(nn.Module):
    """Implements a causal VGG block consisting of causal 2D convolution layers,
    as described in the paper https://arxiv.org/pdf/1910.12977.pdf.



    Args:
        n_conv (int): Specifies the number of convolution layers.

        in_channels (int): Specifies the number of input channels.

        out_channels (List[int]): A list of integers that specifies the number
        of channels in each convolution layer

        kernel_sizes (List[int]): A list of integers that specifies the kernel size of each convolution layer.

        pooling_kernel_size (int): Specifies the kernel size of the pooling layer.

    """

    def __init__(
        self,
        n_conv: int,
        in_channels: int,
        out_channels: List[int],
        kernel_sizes: List[int],
        pooling_kernel_size: int,
    ) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels if i == 0 else out_channels[i - 1],
                    out_channels=out_channels[i],
                    kernel_size=kernel_sizes[i],
                )
                for i in range(n_conv)
            ]
        )
        self.pooling = nn.MaxPool2d(kernel_size=pooling_kernel_size)

    def _pad(self, x: Tensor, kernel_size: Tuple[int, int]):
        batch_size, channels, max_len, feat_size = x.shape
        seq_pad = torch.zeros(batch_size, channels, kernel_size[0] - 1, feat_size).to(
            x.device
        )
        feat_pad = torch.zeros(
            batch_size, channels, kernel_size[0] - 1 + max_len, kernel_size[1] - 1
        ).to(x.device)
        x = torch.cat([seq_pad, x], dim=2)
        x = torch.cat([feat_pad, x], dim=3)
        return x

    def forward(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """passes the input x of shape [B, C, M, f] to the network.

        Args:
            x (Tensor): The input tensor if shape [B, C, M, f].
            lengths (Tensor): The legnths tensor of shape [B].

        Returns:
            Tuple[Tensor, Tensor]: A tuple where the first is the result of shape
            [B, C', M', f'] and the updated lengths of shape [B]
        """
        for conv_layer in self.conv_layers:
            kernel_size = conv_layer.kernel_size
            x = self._pad(x, kernel_size=kernel_size)
            x = conv_layer(x)
        x = self.pooling(x)
        lengths = lengths // self.pooling.kernel_size
        return x, lengths


class TruncatedSelfAttention(MultiHeadAtt):
    """Builds the truncated self attention module used
    in https://arxiv.org/abs/1910.12977

    Args:

        d_model (int): The model dimension.

        h (int): The number of attention heads.

        left_size (int): The size of the left window that each time step is
        allowed to look at.

        right_size (int): The size of the right window that each time step is
        allowed to look at.

        masking_value (float): The attention masking value.
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        left_size: int,
        right_size: int,
        masking_value: float = -1e15,
    ) -> None:
        super().__init__(d_model=d_model, h=h, masking_value=masking_value)
        self.left_size = left_size
        self.right_size = right_size

    def get_looking_ahead_mask(self, mask: Tensor) -> Tensor:
        truncated_mask = truncate_attention_mask(mask, self.right_size, self.left_size)
        return truncated_mask

    def _mask(self, att: Tensor, query_mask: Tensor, *args, **kwargs) -> Tensor:
        query_mask = query_mask.unsqueeze(dim=1)
        return att.masked_fill(~query_mask, self.masking_value)

    def forward(
        self,
        x: Tensor,
        mask: Union[Tensor, None],
    ) -> Tensor:
        """Applies truncated masked multi-head self attention to the input.

        Args:

            x (Tensor): The input tensor of shape [B, M, d].

            mask (Union[Tensor, None]): The mask tensor of the input of shape
            [B, M] where True indicates that the corresponding input position
            contains data not padding and therefore should not be masked.
            If None, the function will act as a normal multi-head self attention.

        Returns:

            Tensor: The attention result tensor of shape [B, M, d].

        """
        query_mask = None
        if mask is not None:
            query_mask = self.get_looking_ahead_mask(mask=mask)
        return super().forward(
            key=x, query=x, value=x, key_mask=mask, query_mask=query_mask
        )


class TransformerEncLayerWithAttTruncation(TransformerEncLayer):
    """Implements a single encoder layer of the transformer
    with truncated self attention as described in https://arxiv.org/abs/1910.12977

    Args:

        d_model (int): The model dimensionality.

        ff_size (int): The feed forward inner layer dimensionality.

        h (int): The number of heads in the attention mechanism.

        left_size (int): The size of the left window that each time step is
        allowed to look at.

        right_size (int): The size of the right window that each time step is
        allowed to look at.

        masking_value (float, optional): The value to use for masking padded
        elements. Defaults to -1e15.
    """

    def __init__(
        self,
        d_model: int,
        ff_size: int,
        h: int,
        left_size: int,
        right_size: int,
        masking_value: int = -1e15,
    ) -> None:
        super().__init__(
            d_model=d_model, ff_size=ff_size, h=h, masking_value=masking_value
        )
        self.mhsa = TruncatedSelfAttention(
            d_model=d_model,
            h=h,
            left_size=left_size,
            right_size=right_size,
            masking_value=masking_value,
        )

    def forward(self, x: Tensor, mask: Union[Tensor, None] = None) -> Tensor:
        """Performs a forward pass of the transformer encoder layer.

        Args:

            x (Tensor): The input tensor of shape [B, M, d].

            mask (Union[Tensor, None], optional): Boolean tensor of the input of shape
            [B, M] where True indicates that the corresponding key position
            contains data not padding and therefore should not be masked.
            If None, the function will act as a normal multi-head attention. Defaults to None.

        Returns:
            Tensor: Result tensor of the same shape as x.
        """
        out = self.mhsa(x=x, mask=mask)
        out = self.add_and_norm1(x, out)
        result = self.ff(out)
        return self.add_and_norm2(out, result)


class VGGTransformerPreNet(nn.Module):
    """Implements the VGGTransformer prenet module as described in
    https://arxiv.org/abs/1910.12977

    Args:

    in_features (int): The input feature size.

    n_vgg_blocks (int): The number of VGG blocks to use.

    n_layers_per_block (List[int]): A list of integers that specifies the number
    of convolution layers in each block.

    kernel_sizes_per_block (List[List[int]]): A list of lists that contains the
    kernel size for each layer in each block. The length of the outer list
    should match `n_vgg_blocks`, and each inner list should be the same length
    as the corresponding block's number of layers.

    n_channels_per_block (List[List[int]]): A list of lists that contains the
    number of channels for each convolution layer in each block. This argument
    should also have length equal to `n_vgg_blocks`, and each sublist should
    have length equal to the number of layers in the corresponding block.

    pooling_kernel_size (List[int]): A list of integers that specifies the size
    of the max pooling layer in each block. The length of this list should be
    equal to `n_vgg_blocks`.

    d_model (int): The size of the output feature

    """

    def __init__(
        self,
        in_features: int,
        n_vgg_blocks: int,
        n_layers_per_block: List[int],
        kernel_sizes_per_block: List[List[int]],
        n_channels_per_block: List[List[int]],
        pooling_kernel_size: List[int],
        d_model: int,
    ) -> None:
        super().__init__()
        self.vgg_blocks = nn.ModuleList(
            [
                CausalVGGBlock(
                    n_conv=n_layers_per_block[i],
                    in_channels=1 if i == 0 else n_channels_per_block[i - 1][-1],
                    out_channels=n_channels_per_block[i],
                    kernel_sizes=kernel_sizes_per_block[i],
                    pooling_kernel_size=pooling_kernel_size[i],
                )
                for i in range(n_vgg_blocks)
            ]
        )
        for i in range(n_vgg_blocks):
            in_features //= pooling_kernel_size[i]
        in_features *= n_channels_per_block[-1][-1]
        self.fc = nn.Linear(in_features=in_features, out_features=d_model)

    def forward(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` through the VGGTransformer prenet and returns
        a tuple of tensors.

        Args:
            x (Tensor): Input tensor of shape [B, M, in_features].

            lengths (Tensor): Lengths of shape [B] that has the length for each
            sequence in `x`.

        Returns:
            A tuple of tensors (output, updated_lengths).
            - output (Tensor): Output tensor of shape [B, M, d_model].
            - updated_lengths (Tensor): Updated lengths of shape [B].
        """
        x = x.unsqueeze(dim=1)  # [B, 1, M, d]
        for block in self.vgg_blocks:
            x, lengths = block(x, lengths)
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous()
        x = x.view(*x.shape[:2], -1)
        return self.fc(x), lengths


class TruncatedRelativeMHSA(TruncatedSelfAttention):
    """Builds the truncated self attention with relative positional encoding
    module proposed in https://arxiv.org/abs/2002.02562

    Args:

        d_model (int): The model dimension.

        h (int): The number of attention heads.

        left_size (int): The size of the left window that each time step is
        allowed to look at.

        right_size (int): The size of the right window that each time step is
        allowed to look at.

        masking_value (float): The attention masking value.
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        left_size: int,
        right_size: int,
        masking_value: float = -1e15,
    ) -> None:
        super().__init__(
            d_model=d_model,
            h=h,
            left_size=left_size,
            right_size=right_size,
            masking_value=masking_value,
        )

    def forward(
        self,
        x: Tensor,
        mask: Union[Tensor, None],
    ) -> Tensor:
        """Applies truncated masked rekative multi-head self attention to the input.

        Args:

            x (Tensor): The input tensor of shape [B, M, d].

            mask (Union[Tensor, None]): The mask tensor of the input of shape
            [B, M] where True indicates that the corresponding input position
            contains data not padding and therefore should not be masked.
            If None, the function will act as a normal multi-head self attention.

        Returns:

            Tensor: The attention result tensor of shape [B, M, d].

        """
        x = add_pos_enc(x)
        return super().forward(x=x, mask=mask)
