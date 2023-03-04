"""This module provides various speech encoders.

The available encoders are:

- DeepSpeechV1Encoder: The encoder implementation of the DeepSpeech V1 model.
- DeepSpeechV2Encoder: The encoder implementation of the DeepSpeech V2 model.
- ConformerEncoder: The encoder implementation of the Conformer model.
- JasperEncoder: The encoder implementation of the Jasper model.
- Wav2LetterEncoder: The encoder implementation of the Wav2Letter model.
- QuartzNetEncoder: The encoder implementation of the QuartzNet model.
- SqueezeformerEncoder: The encoder implementation of the Squeezeformer model.
- SpeechTransformerEncoder: The encoder implementation of the Speech Transformer model.
- RNNEncoder: The encoder implementation of a general RNN model.
- PyramidRNNEncoder: The encoder implementation of the Pyramid RNN model.
- ContextNetEncoder: The encoder implementation of the ContextNet model.
- VGGTransformerEncoder: The encoder implementation of the VGG-Transformer.
- TransformerTransducerEncoder: The encoder implementation of the transformer transducer with relative truncated multi-head self-attention.

Each encoder takes a speech input of shape [B, M, d], and the lengths if
shape [B], where B is the batch size, M is the length of
the speech sequence, and d is the number of features.
"""
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from speeq.utils.utils import add_pos_enc, calc_data_len, get_mask_from_lens

from .activations import CReLu
from .layers import (
    ConformerBlock,
    ConformerPreNet,
    ContextNetBlock,
    Conv1DLayers,
    JasperBlocks,
    JasperSubBlock,
    QuartzBlocks,
    RowConv1D,
    SpeechTransformerEncLayer,
    SqueezeformerBlock,
    TransformerEncLayer,
    TransformerEncLayerWithAttTruncation,
    TransformerTransducerLayer,
    VGGTransformerPreNet,
)


class DeepSpeechV1Encoder(nn.Module):
    """Builds the DeepSpeech encoder described in
    https://arxiv.org/abs/1412.5567

    Args:

        in_features (int): The input feature size.

        hidden_size (int): The hidden size of the rnn layers.

        n_linear_layers (int): The number of feed-forward layers.

        bidirectional (bool): if the rnn is bidirectional or not.

        max_clip_value (int): The maximum relu clipping value.

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.

        p_dropout (float): The dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        n_linear_layers: int,
        bidirectional: bool,
        max_clip_value: int,
        rnn_type: str,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.ff_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        in_features=in_features if i == 0 else hidden_size,
                        out_features=hidden_size,
                    ),
                    CReLu(max_val=max_clip_value),
                    nn.Dropout(p=p_dropout),
                )
                for i in range(n_linear_layers)
            ]
        )
        from .registry import PACKED_RNN_REGISTRY

        self.rnn = PACKED_RNN_REGISTRY[rnn_type](
            input_size=hidden_size, hidden_size=hidden_size, bidirectional=bidirectional
        )
        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
        )
        self.crelu = CReLu(max_val=max_clip_value)
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

    def forward(
        self, x: Tensor, mask: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` through the encoder layers.

        Args:

            x (Tensor): The input speech tensor of shape [B, M, d]

            mask (Tensor): The input boolean input mask of shape [B, M], where it's True
            if there is no padding.

        Returns:

            Tuple[Tensor, Tensor]: A tuple where the first element is the encoded speech of shape
            [B, M, F] and the second element is the lengths of shape [B].
        """
        lengths = mask.sum(dim=-1)
        for layer in self.ff_layers:
            x = layer(x)
        out, _, lengths = self.rnn(x, lengths.cpu())
        out = self.crelu(out)
        if self.bidirectional is True:
            out = out[..., : self.hidden_size] + out[..., self.hidden_size :]
        out = self.crelu(self.fc(out))
        return out, lengths


class DeepSpeechV2Encoder(nn.Module):
    """Implements the deep speech 2 encoder proposed in
    https://arxiv.org/abs/1512.02595

    Args:
        n_conv (int): The number of convolution layers.

        kernel_size (int): The kernel size of the convolution layers.

        stride (int): The stride size of the convolution layer.

        in_features (int): The input/speech feature size.

        hidden_size (int): The hidden size of the RNN layers.

        bidirectional (bool): A flag indicating if the rnn is bidirectional or not.

        n_rnn (int): The number of RNN layers.

        n_linear_layers (int): The number of linear layers.

        max_clip_value (int): The maximum relu clipping value.

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.

        tau (int): The future context size.

        p_dropout (float): The dropout rate.
    """

    def __init__(
        self,
        n_conv: int,
        kernel_size: int,
        stride: int,
        in_features: int,
        hidden_size: int,
        bidirectional: bool,
        n_rnn: int,
        n_linear_layers: int,
        max_clip_value: int,
        rnn_type: str,
        tau: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.conv = Conv1DLayers(
            in_size=in_features,
            out_size=hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            n_layers=n_conv,
            p_dropout=p_dropout,
            activation=CReLu(max_val=max_clip_value),
        )
        from .registry import PACKED_RNN_REGISTRY

        self.rnns = nn.ModuleList(
            [
                PACKED_RNN_REGISTRY[rnn_type](
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    bidirectional=bidirectional,
                )
                for _ in range(n_rnn)
            ]
        )
        self.rnn_bnorms = nn.ModuleList(
            [nn.BatchNorm1d(num_features=hidden_size) for _ in range(n_rnn)]
        )
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(in_features=hidden_size, out_features=hidden_size)
                for _ in range(n_linear_layers)
            ]
        )
        self.linear_bnorms = nn.ModuleList(
            [nn.BatchNorm1d(num_features=hidden_size) for _ in range(n_linear_layers)]
        )
        self.crelu = CReLu(max_val=max_clip_value)
        self.context_conv = RowConv1D(tau=tau, feat_size=hidden_size)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def forward(
        self, x: Tensor, mask: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` through the encoder layers.

        Args:

            x (Tensor): The input speech tensor of shape [B, M, d]

            mask (Tensor): The input boolean input mask of shape [B, M], where it's True
            if there is no padding.

        Returns:

            Tuple[Tensor, Tensor]: A tuple where the first element is the encoded speech of shape
            [B, M, F] and the second element is the lengths of shape [B].
        """
        lengths = mask.sum(dim=-1)
        lengths = lengths.cpu()
        out, lengths = self.conv(x, lengths)
        out = self.crelu(out)
        for bnorm, layer in zip(self.rnn_bnorms, self.rnns):
            out = out.transpose(-1, -2)
            out = bnorm(out)
            out = out.transpose(-1, -2)
            out, _, lengths = layer(out, lengths)
            if self.bidirectional is True:
                out = out[..., : self.hidden_size] + out[..., self.hidden_size :]
            out = self.crelu(out)
        out = self.context_conv(out)
        for bnorm, layer in zip(self.linear_bnorms, self.linear_layers):
            out = layer(out)
            out = out.transpose(-1, -2)
            out = bnorm(out)
            out = out.transpose(-1, -2)
            out = self.crelu(out)
        return out, lengths


class ConformerEncoder(nn.Module):
    """Implements the conformer encoder proposed in
    https://arxiv.org/abs/2005.08100

    Args:
        d_model (int): The model dimension.

        n_conf_layers (int): The number of conformer blocks.

        ff_expansion_factor (int): The feed-forward expansion factor.

        h (int): The number of attention heads.

        kernel_size (int): The convolution module kernel size.

        ss_kernel_size (int): The subsampling layer kernel size.

        ss_stride (int): The subsampling layer stride size.

        ss_num_conv_layers (int): The number of subsampling convolutional layers.

        in_features (int): The input/speech feature size.

        res_scaling (float): The residual connection multiplier.

        p_dropout (float): The dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_conf_layers: int,
        ff_expansion_factor: int,
        h: int,
        kernel_size: int,
        ss_kernel_size: int,
        ss_stride: int,
        ss_num_conv_layers: int,
        in_features: int,
        res_scaling: float,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.sub_sampling = ConformerPreNet(
            in_features=in_features,
            kernel_size=ss_kernel_size,
            stride=ss_stride,
            n_conv_layers=ss_num_conv_layers,
            d_model=d_model,
            p_dropout=p_dropout,
        )
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    ff_expansion_factor=ff_expansion_factor,
                    h=h,
                    kernel_size=kernel_size,
                    p_dropout=p_dropout,
                    res_scaling=res_scaling,
                )
                for _ in range(n_conf_layers)
            ]
        )

    def forward(
        self, x: Tensor, mask: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` through the encoder layers.

        Args:

            x (Tensor): The input speech tensor of shape [B, M, d]

            mask (Tensor): The input boolean input mask of shape [B, M], where it's True
            if there is no padding.

        Returns:

            Tuple[Tensor, Tensor]: A tuple where the first element is the encoded speech of shape
            [B, M, F] and the second element is the lengths of shape [B].
        """
        lengths = mask.sum(dim=-1)
        lengths = lengths.cpu()
        out, lengths = self.sub_sampling(x, lengths)
        mask = get_mask_from_lens(lengths, lengths.max().item())
        mask = mask.to(x.device)
        for layer in self.blocks:
            out = layer(out, mask)
        return out, lengths


class JasperEncoder(nn.Module):
    """Implements Jasper's encoder proposed in https://arxiv.org/abs/1904.03288

    Args:

        in_features (int): The input/speech feature size.

        num_blocks (int): The number of Jasper blocks (denoted as 'B' in the paper).

        num_sub_blocks (int): The number of Jasper subblocks (denoted as 'R' in the paper).

        channel_inc (int): The rate to increase the number of channels across the blocks.

        epilog_kernel_size (int): The kernel size of the epilog block convolution layer.

        prelog_kernel_size (int): The kernel size of the prelog block ocnvolution layer.

        prelog_stride (int): The stride size of the prelog block convolution layer.

        prelog_n_channels (int): The output channnels of the prelog block convolution layer.

        blocks_kernel_size (Union[int, List[int]]): The kernel size(s) of the convolution layer for each block.

        p_dropout (float): The dropout rate.

    """

    def __init__(
        self,
        in_features: int,
        num_blocks: int,
        num_sub_blocks: int,
        channel_inc: int,
        epilog_kernel_size: int,
        prelog_kernel_size: int,
        prelog_stride: int,
        prelog_n_channels: int,
        blocks_kernel_size: Union[int, List[int]],
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.prelog = JasperSubBlock(
            in_channels=in_features,
            out_channels=prelog_n_channels,
            kernel_size=prelog_kernel_size,
            p_dropout=p_dropout,
            padding=0,
            stride=prelog_stride,
        )
        self.prelog_stride = prelog_stride
        self.prelog_kernel_size = prelog_kernel_size
        self.blocks = JasperBlocks(
            num_blocks=num_blocks,
            num_sub_blocks=num_sub_blocks,
            in_channels=prelog_n_channels,
            channel_inc=channel_inc,
            kernel_size=blocks_kernel_size,
            p_dropout=p_dropout,
        )
        self.epilog1 = JasperSubBlock(
            in_channels=prelog_n_channels + channel_inc * num_blocks,
            out_channels=prelog_n_channels + channel_inc * (1 + num_blocks),
            kernel_size=epilog_kernel_size,
            p_dropout=p_dropout,
        )
        self.epilog2 = JasperSubBlock(
            in_channels=prelog_n_channels + channel_inc * (1 + num_blocks),
            out_channels=prelog_n_channels + channel_inc * (2 + num_blocks),
            kernel_size=1,
            p_dropout=p_dropout,
        )

    def forward(
        self, x: Tensor, mask: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` through the encoder layers.

        Args:

            x (Tensor): The input speech tensor of shape [B, M, d]

            mask (Tensor): The input boolean input mask of shape [B, M], where it's True
            if there is no padding.

        Returns:

            Tuple[Tensor, Tensor]: A tuple where the first element is the encoded speech of shape
            [B, M, F] and the second element is the lengths of shape [B].
        """
        lengths = mask.sum(dim=-1)
        lengths = lengths.cpu()
        x = x.transpose(-1, -2)
        out = self.prelog(x)
        lengths = calc_data_len(
            result_len=out.shape[-1],
            pad_len=x.shape[-1] - lengths,
            data_len=lengths,
            kernel_size=self.prelog_kernel_size,
            stride=self.prelog_stride,
        )
        out = self.blocks(out)
        out = self.epilog1(out)
        out = self.epilog2(out)
        out = out.transpose(-1, -2)  # [B, M, d']
        return out, lengths


class Wav2LetterEncoder(nn.Module):
    """Implements the Wav2Letter encoder proposed in
    https://arxiv.org/abs/1609.03193

    Args:

        in_features (int): The input/speech feature size.

        n_conv_layers (int): The number of convolution layers.

        layers_kernel_size (int): The kernel size of the convolution layers.

        layers_channels_size (int): The number of output channels of each convolution layer.

        pre_conv_stride (int): The stride of the prenet convolution layer.

        pre_conv_kernel_size (int): The kernel size of the prenet convolution layer.

        post_conv_channels_size (int): The number of output channels of the
        postnet convolution layer.

        post_conv_kernel_size (int): The kernel size of the postnet convolution layer.

        p_dropout (float): The dropout rate.

        wav_kernel_size (Optional[int]): The kernel size of the first layer that
        processes the wav samples directly if wav is modeled. Default None.

        wav_stride (Optional[int]): The stride size of the first layer that
        processes the wav samples directly if wav is modeled. Default None.
    """

    def __init__(
        self,
        in_features: int,
        n_conv_layers: int,
        layers_kernel_size: int,
        layers_channels_size: int,
        pre_conv_stride: int,
        pre_conv_kernel_size: int,
        post_conv_channels_size: int,
        post_conv_kernel_size: int,
        p_dropout: float,
        wav_kernel_size: Optional[int] = None,
        wav_stride: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.is_wav = in_features == 1
        if self.is_wav:
            assert wav_kernel_size is not None
            assert wav_stride is not None
            self.raw_conv = nn.Conv1d(
                in_channels=1,
                out_channels=layers_channels_size,
                kernel_size=wav_kernel_size,
                stride=wav_stride,
            )
        self.pre_conv = nn.Conv1d(
            in_channels=layers_channels_size if self.is_wav else in_features,
            out_channels=layers_channels_size,
            kernel_size=pre_conv_kernel_size,
            stride=pre_conv_stride,
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=layers_channels_size,
                    out_channels=layers_channels_size,
                    kernel_size=layers_kernel_size,
                    padding="same",
                )
                for _ in range(n_conv_layers - 1)
            ]
        )
        self.convs.append(
            nn.Conv1d(
                in_channels=layers_channels_size,
                out_channels=post_conv_channels_size,
                kernel_size=post_conv_kernel_size,
                padding="same",
            )
        )
        self.post_conv = nn.Conv1d(
            in_channels=post_conv_channels_size,
            out_channels=post_conv_channels_size,
            kernel_size=1,
            padding="same",
        )
        self.dropout = nn.Dropout(p_dropout)

    def forward(
        self, x: Tensor, mask: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` through the encoder layers.

        Args:

            x (Tensor): The input speech tensor of shape [B, M, d]

            mask (Tensor): The input boolean input mask of shape [B, M], where it's True
            if there is no padding.

        Returns:

            Tuple[Tensor, Tensor]: A tuple where the first element is the encoded speech of shape
            [B, M, F] and the second element is the lengths of shape [B].
        """
        lengths = mask.sum(dim=-1)
        lengths = lengths.cpu()
        x = x.transpose(-1, -2)
        out = x
        if self.is_wav:
            out = self.raw_conv(out)
            out = torch.tanh(out)
            out = self.dropout(out)
            lengths = calc_data_len(
                result_len=out.shape[-1],
                pad_len=x.shape[-1] - lengths,
                data_len=lengths,
                kernel_size=self.raw_conv.kernel_size[0],
                stride=self.raw_conv.stride[0],
            )
        results = self.pre_conv(out)
        lengths = calc_data_len(
            result_len=results.shape[-1],
            pad_len=out.shape[-1] - lengths,
            data_len=lengths,
            kernel_size=self.pre_conv.kernel_size[0],
            stride=self.pre_conv.stride[0],
        )
        out = results
        out = torch.tanh(out)
        out = self.dropout(out)
        for layer in self.convs:
            out = layer(out)
            out = torch.tanh(out)
            out = self.dropout(out)
        out = self.post_conv(out)
        out = torch.tanh(out)
        out = self.dropout(out)
        out = out.transpose(-1, -2)  # [B, M, d]
        return out, lengths


class QuartzNetEncoder(JasperEncoder):
    """Implements QuartzNet encoder proposed in https://arxiv.org/abs/1910.10261

    Args:

        in_features (int): The input/speech feature size.

        num_blocks (int): The number of QuartzNet blocks (denoted as 'B' in the paper).

        block_repetition (int): The number of times to repeat each block (denoted as 'S' in the paper).

        num_sub_blocks (int): The number of QuartzNet subblocks, (denoted as 'R' in the paper).

        channels_size (List[int]): A list of integers representing the number of output channels
        for each block.

        epilog_kernel_size (int): The kernel size of the convolution layer in the epilog block.

        epilog_channel_size (Tuple[int, int]): A tuple for both epilog layers
        of the convolution layer .

        prelog_kernel_size (int): The kernel size pf the convolution layer in the prelog block.

        prelog_stride (int): The stride size of the of the convoltuional layer
        in the prelog block.

        prelog_n_channels (int): The number of output channels of the convolutional
        layer in the prelog block.

        groups (int): The groups size.

        blocks_kernel_size (Union[int, List[int]]): An integer or a list of integers representing the
        kernel size(s) for each block's convolutional layer.

        p_dropout (float): The dropout rate.

    """

    def __init__(
        self,
        in_features: int,
        num_blocks: int,
        block_repetition: int,
        num_sub_blocks: int,
        channels_size: List[int],
        epilog_kernel_size: int,
        epilog_channel_size: Tuple[int, int],
        prelog_kernel_size: int,
        prelog_stride: int,
        prelog_n_channels: int,
        groups: int,
        blocks_kernel_size: Union[int, List[int]],
        p_dropout: float,
    ) -> None:
        super().__init__(
            in_features=in_features,
            num_blocks=num_blocks,
            num_sub_blocks=num_sub_blocks,
            channel_inc=0,
            epilog_kernel_size=epilog_kernel_size,
            prelog_kernel_size=prelog_kernel_size,
            prelog_stride=prelog_stride,
            prelog_n_channels=prelog_n_channels,
            blocks_kernel_size=blocks_kernel_size,
            p_dropout=p_dropout,
        )
        self.blocks = QuartzBlocks(
            num_blocks=num_blocks,
            block_repetition=block_repetition,
            num_sub_blocks=num_sub_blocks,
            in_channels=prelog_n_channels,
            channels_size=channels_size,
            kernel_size=blocks_kernel_size,
            groups=groups,
            p_dropout=p_dropout,
        )
        self.epilog1 = JasperSubBlock(
            in_channels=channels_size[-1],
            out_channels=epilog_channel_size[0],
            kernel_size=epilog_kernel_size,
            p_dropout=p_dropout,
        )
        self.epilog2 = JasperSubBlock(
            in_channels=epilog_channel_size[0],
            out_channels=epilog_channel_size[1],
            kernel_size=1,
            p_dropout=p_dropout,
        )


class SqueezeformerEncoder(nn.Module):
    """Implements the Squeezeformer encoder
    as described in https://arxiv.org/abs/2206.00888

    Args:

        in_features (int): The input/speech feature size.

        n (int): The number of layers per block, (denoted as N in the paper).

        d_model (int): The model dimension.

        ff_expansion_factor (int): The expansion factor of linear layer in the
        feed forward module.

        h (int): The number of attention heads.

        kernel_size (int): The kernel size of the depth-wise convolution layer.

        pooling_kernel_size (int): The kernel size of the pooling convolution layer.

        pooling_stride (int): The stride size of the pooling convolution layer.

        ss_kernel_size (Union[int, List[int]]): The kernel size of the subsampling layer(s).

        ss_stride (Union[int, List[int]]): The stride of the subsampling layer(s).

        ss_n_conv_layers (int): The number of subsampling convolutional layers.

        p_dropout (float): The dropout rate.

        ss_groups (Union[int, List[int]]): The subsampling convolution groups size(s).

        masking_value (int): The masking value. Default -1e15
    """

    def __init__(
        self,
        in_features: int,
        n: int,
        d_model: int,
        ff_expansion_factor: int,
        h: int,
        kernel_size: int,
        pooling_kernel_size: int,
        pooling_stride: int,
        ss_kernel_size: Union[int, List[int]],
        ss_stride: Union[int, List[int]],
        ss_n_conv_layers: int,
        p_dropout: float,
        ss_groups: Union[int, List[int]] = 1,
        masking_value: int = -1e15,
    ) -> None:
        super().__init__()
        self.subsampling = ConformerPreNet(
            in_features=in_features,
            kernel_size=ss_kernel_size,
            stride=ss_stride,
            n_conv_layers=ss_n_conv_layers,
            d_model=d_model,
            p_dropout=p_dropout,
            groups=ss_groups,
        )
        self.layers1 = nn.ModuleList(
            [
                SqueezeformerBlock(
                    d_model=d_model,
                    ff_expansion_factor=ff_expansion_factor,
                    h=h,
                    kernel_size=kernel_size,
                    p_dropout=p_dropout,
                    masking_value=masking_value,
                )
                for _ in range(n - 1)
            ]
        )
        self.pooling = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=pooling_kernel_size,
            stride=pooling_stride,
            groups=d_model,
        )
        self.layers2 = nn.ModuleList(
            [
                SqueezeformerBlock(
                    d_model=d_model,
                    ff_expansion_factor=ff_expansion_factor,
                    h=h,
                    kernel_size=kernel_size,
                    p_dropout=p_dropout,
                    masking_value=masking_value,
                )
                for _ in range(n)
            ]
        )
        self.upsampling_conv = nn.ConvTranspose1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=pooling_kernel_size,
            stride=pooling_stride,
        )
        self.sf_layer = SqueezeformerBlock(
            d_model=d_model,
            ff_expansion_factor=ff_expansion_factor,
            h=h,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            masking_value=masking_value,
        )

    def _pass_through_layers(
        self, x: Tensor, mask: Tensor, layers: nn.ModuleList
    ) -> Tensor:
        for layer in layers:
            x = layer(x, mask)
        return x

    def _upsample(self, x: Tensor, target_len: int):
        # x of shape [B, M, d]
        x = x.transpose(-1, -2)
        out = self.upsampling_conv(x)
        res_len = target_len - out.shape[-1]
        out = torch.cat([out, torch.zeros(*x.shape[:2], res_len).to(x.device)], dim=-1)
        out = out.transpose(-1, -2)
        return out

    def _time_pooling(self, x: Tensor):
        x = x.transpose(-1, -2)
        out = self.pooling(x)
        out = out.transpose(-1, -2)
        return out

    def forward(
        self, x: Tensor, mask: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` through the encoder layers.

        Args:

            x (Tensor): The input speech tensor of shape [B, M, d]

            mask (Tensor): The input boolean input mask of shape [B, M], where it's True
            if there is no padding.

        Returns:

            Tuple[Tensor, Tensor]: A tuple where the first element is the encoded speech of shape
            [B, M, F] and the second element is the lengths of shape [B].
        """
        lengths = mask.sum(dim=-1)
        out, lengths = self.subsampling(x, lengths)
        mask = get_mask_from_lens(lengths=lengths, max_len=out.shape[1])
        out = self._pass_through_layers(out, mask, self.layers1)
        result = self._time_pooling(out)
        pooled_len = calc_data_len(
            result_len=result.shape[1],
            pad_len=out.shape[1] - lengths,
            data_len=lengths,
            kernel_size=self.pooling.kernel_size[0],
            stride=self.pooling.stride[0],
        )
        pooled_mask = get_mask_from_lens(lengths=pooled_len, max_len=result.shape[1])
        result = self._pass_through_layers(result, pooled_mask, self.layers2)
        result = self._upsample(result, out.shape[1])
        out = result + out
        out = self.sf_layer(out, mask)
        return out, lengths


class SpeechTransformerEncoder(nn.Module):
    """Implements the speech transformer encoder
    described in https://ieeexplore.ieee.org/document/8462506

    Args:

        in_features (int): The input/speech feature size.

        n_conv_layers (int): The number of down-sampling convolutional layers.

        kernel_size (int): The kernel size of the down-sampling convolutional layers.

        stride (int): The stride size of the down-sampling convolutional layers.

        d_model (int): The model dimensionality.

        n_layers (int): The number of encoder layers.

        ff_size (int):  The dimensionality of the inner layer of the feed-forward module.

        h (int): The number of attention heads.

        att_kernel_size (int): The kernel size of the attentional convolutional layers.

        att_out_channels (int): The number of output channels of the attentional convolution layers.
    """

    def __init__(
        self,
        in_features: int,
        n_conv_layers: int,
        kernel_size: int,
        stride: int,
        d_model: int,
        n_layers: int,
        ff_size: int,
        h: int,
        att_kernel_size: int,
        att_out_channels: int,
    ) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=kernel_size,
                    stride=stride,
                )
                for _ in range(n_conv_layers)
            ]
        )
        self.relu = nn.ReLU()
        for _ in range(n_conv_layers):
            in_features = (in_features - kernel_size) // stride + 1
        self.fc = nn.Linear(in_features=in_features, out_features=d_model)
        self.layers = nn.ModuleList(
            [
                SpeechTransformerEncLayer(
                    d_model=d_model,
                    ff_size=ff_size,
                    h=h,
                    out_channels=att_out_channels,
                    kernel_size=att_kernel_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.d_model = d_model

    def _pre_process(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.unsqueeze(dim=1)  # B, 1, M, d
        lengths = mask.sum(dim=-1)
        for layer in self.conv_layers:
            length = x.shape[-2]
            x = layer(x)
            lengths = calc_data_len(
                result_len=x.shape[-2],
                pad_len=length - lengths,
                data_len=lengths,
                kernel_size=layer.kernel_size[0],
                stride=layer.stride[0],
            )
            x = self.relu(x)
        x = x.squeeze(dim=1)
        x = self.fc(x)
        x = add_pos_enc(x)
        mask = get_mask_from_lens(lengths=lengths, max_len=x.shape[1])
        mask = mask.to(x.device)
        return x, mask

    def forward(
        self, x: Tensor, mask: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` through the encoder layers.

        Args:

            x (Tensor): The input speech tensor of shape [B, M, d]

            mask (Tensor): The input boolean input mask of shape [B, M], where it's True
            if there is no padding.

        Returns:

            Tuple[Tensor, Tensor]: A tuple where the first element is the encoded speech of shape
            [B, M, F] and the second element is the lengths of shape [B].
        """
        out, mask = self._pre_process(x, mask)
        for layer in self.layers:
            out = layer(out, mask)
        lengths = mask.sum(dim=-1)
        out = self.layer_norm(out)
        return out, lengths


class RNNEncoder(nn.Module):
    """Implements a stack of RNN layers.

    Args:

        in_features (int): The input features size.

        hidden_size (int): The RNN hidden size.

        bidirectional (bool): A flag indicating if the rnn is bidirectional or not.

        n_layers (int): The number of RNN layers.

        p_dropout (float): The dropout rate.

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.

    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        bidirectional: bool,
        n_layers: int,
        p_dropout: float,
        rnn_type: str = "rnn",
    ) -> None:
        super().__init__()
        from .registry import PACKED_RNN_REGISTRY

        if bidirectional is True:
            assert hidden_size % 2 == 0
        self.rnns = nn.ModuleList(
            [
                PACKED_RNN_REGISTRY[rnn_type](
                    input_size=in_features if i == 0 else hidden_size,
                    hidden_size=hidden_size // 2 if bidirectional else hidden_size,
                    batch_first=True,
                    enforce_sorted=False,
                    bidirectional=bidirectional,
                )
                for i in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(p_dropout)
        self.n_layers = n_layers

    def forward(
        self, x: Tensor, mask: Tensor, return_h=False, *args, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Passes the input `x` through the encoder layers.

        Args:

            x (Tensor): The input speech tensor of shape [B, M, d]

            mask (Tensor): The input boolean input mask of shape [B, M], where it's True
            if there is no padding.

        Returns:

            Tuple[Tensor, Tensor]: A tuple where the first element is the encoded speech of shape
            [B, M, F] and the second element is the lengths of shape [B].
        """
        out = x
        lengths = mask.sum(dim=-1).cpu()
        for i, layer in enumerate(self.rnns):
            out, h, lengths = layer(out, lengths)
            if (i + 1) != self.n_layers:
                out = self.dropout(out)
        if return_h is True:
            return out, h, lengths
        return out, lengths


class PyramidRNNEncoder(nn.Module):
    """Implements a pyramid of RNN as described in
    https://arxiv.org/abs/1508.01211.

    Args:

        in_features (int): The input features size.

        hidden_size (int): The RNN hidden size.

        reduction_factor (int): The time resolution reduction factor.

        bidirectional (bool): A flag indicating if the rnn is bidirectional or not.

        n_layers (int): The number of RNN layers.

        p_dropout (float): The dropout rate.

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.

    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        reduction_factor: int,
        bidirectional: bool,
        n_layers: int,
        p_dropout: float,
        rnn_type: str = "rnn",
    ) -> None:
        super().__init__()
        self.reduction_factor = reduction_factor
        from .registry import PACKED_RNN_REGISTRY

        if bidirectional is True:
            assert hidden_size % 2 == 0
        if bidirectional is True:
            hidden_size = hidden_size // 2
        self.rnns = nn.ModuleList(
            [
                PACKED_RNN_REGISTRY[rnn_type](
                    input_size=in_features,
                    hidden_size=hidden_size,
                    batch_first=True,
                    enforce_sorted=False,
                    bidirectional=bidirectional,
                )
            ]
        )
        for _ in range(n_layers - 1):
            inp_size = (1 + bidirectional) * hidden_size * reduction_factor
            self.rnns.append(
                PACKED_RNN_REGISTRY[rnn_type](
                    input_size=inp_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    enforce_sorted=False,
                    bidirectional=bidirectional,
                )
            )
        self.dropout = nn.Dropout(p_dropout)
        self.n_layers = n_layers

    def _reduce(self, x: Tensor) -> Tensor:
        # x of shape [B, M, d]
        max_len = x.shape[1]
        assert max_len > self.reduction_factor
        # making sure it's divisible by the reduction factor
        res_len = max_len % self.reduction_factor
        res_len = self.reduction_factor if res_len == 0 else res_len
        pad_len = self.reduction_factor - res_len
        # adding trailing zeros to make the sequence divisible
        x = torch.cat(
            [x, torch.zeros(x.shape[0], pad_len, x.shape[-1]).to(x.device)], dim=1
        )
        x = x.view(x.shape[0], x.shape[1] // self.reduction_factor, -1)
        return x

    def forward(
        self, x: Tensor, mask: Tensor, return_h=False, *args, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Passes the input `x` through the encoder layers.

        Args:

            x (Tensor): The input speech tensor of shape [B, M, d]

            mask (Tensor): The input boolean input mask of shape [B, M], where it's True
            if there is no padding.

        Returns:

            Tuple[Tensor, Tensor]: A tuple where the first element is the encoded speech of shape
            [B, M, F] and the second element is the lengths of shape [B].
        """
        out = x
        lengths = mask.sum(dim=-1).cpu()
        for i, layer in enumerate(self.rnns):
            out, h, lengths = layer(out, lengths)
            if (i + 1) != self.n_layers:
                out = self._reduce(out)
                lengths = torch.ceil(lengths / self.reduction_factor)
                out = self.dropout(out)
        lengths = lengths.long()
        if return_h is True:
            return out, h, lengths
        return out, lengths


class ContextNetEncoder(nn.Module):
    """Implements the ContextNet encoder proposed in
    https://arxiv.org/abs/2005.03191

    Args:
        in_features (int): The input feature size.

        n_layers (int): The number of ContextNet blocks.

        n_sub_layers (Union[int, List[int]]): The number of convolutional
        layers per block. If list is passed, it has to be of length equal to `n_layers`.

        stride (Union[int, List[int]]): The stride of the last convolutional
        layers per block. If list is passed, it has to be of length equal to
        `n_layers`.

        out_channels (Union[int, List[int]]): The channels size of the
        convolutional layers per block. If list is passed, it has to be of
        length equal to `n_layers`.

        kernel_size (int): The convolutional layers kernel size.

        reduction_factor (int): The feature reduction size of the Squeeze-and-excitation module.
    """

    def __init__(
        self,
        in_features: int,
        n_layers: int,
        n_sub_layers: Union[int, List[int]],
        stride: Union[int, List[int]],
        out_channels: Union[int, List[int]],
        kernel_size: int,
        reduction_factor: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            if i == 0:
                self.layers.append(
                    ContextNetBlock(
                        n_layers=1,
                        in_channels=in_features,
                        out_channels=out_channels[i]
                        if isinstance(out_channels, list)
                        else out_channels,
                        kernel_size=kernel_size,
                        reduction_factor=reduction_factor,
                        add_residual=False,
                        last_layer_stride=1,
                    )
                )
            elif i == n_layers - 1:
                self.layers.append(
                    ContextNetBlock(
                        n_layers=1,
                        in_channels=out_channels[i - 1]
                        if isinstance(out_channels, list)
                        else out_channels,
                        out_channels=out_channels[i]
                        if isinstance(out_channels, list)
                        else out_channels,
                        kernel_size=kernel_size,
                        reduction_factor=reduction_factor,
                        add_residual=False,
                        last_layer_stride=1,
                    )
                )
            else:
                self.layers.append(
                    ContextNetBlock(
                        n_layers=n_sub_layers[i]
                        if isinstance(n_sub_layers, list)
                        else n_sub_layers,
                        in_channels=out_channels[i - 1]
                        if isinstance(out_channels, list)
                        else out_channels,
                        out_channels=out_channels[i]
                        if isinstance(out_channels, list)
                        else out_channels,
                        kernel_size=kernel_size,
                        reduction_factor=reduction_factor,
                        add_residual=True,
                        last_layer_stride=stride[i]
                        if isinstance(stride, list)
                        else stride,
                    )
                )

    def forward(
        self, x: Tensor, mask: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` through the encoder layers.

        Args:

            x (Tensor): The input speech tensor of shape [B, M, d]

            mask (Tensor): The input boolean input mask of shape [B, M], where it's True
            if there is no padding.

        Returns:

            Tuple[Tensor, Tensor]: A tuple where the first element is the encoded speech of shape
            [B, M, F] and the second element is the lengths of shape [B].
        """
        lengths = mask.sum(dim=-1)
        out = x.transpose(-1, -2)  # [B, d, M]
        for layer in self.layers:
            out, lengths = layer(out, lengths)
        out = out.transpose(-1, -2)  # [B, M, d']
        return out, lengths


class VGGTransformerEncoder(nn.Module):
    """Implements the VGGTransformer encoder as described in
    https://arxiv.org/abs/1910.12977

    Args:

        in_features (int): The input feature size.

        n_layers (int): The number of transformer encoder layers with truncated
        self attention.

        n_vgg_blocks (int): The number of VGG blocks to use.

        n_conv_layers_per_vgg_block (List[int]): A list of integers that specifies the number
        of convolution layers in each block.

        kernel_sizes_per_vgg_block (List[List[int]]): A list of lists that contains the
        kernel size for each layer in each block. The length of the outer list
        should match `n_vgg_blocks`, and each inner list should be the same length
        as the corresponding block's number of layers.

        n_channels_per_vgg_block (List[List[int]]): A list of lists that contains the
        number of channels for each convolution layer in each block. This argument
        should also have length equal to `n_vgg_blocks`, and each sublist should
        have length equal to the number of layers in the corresponding block.

        vgg_pooling_kernel_size (List[int]): A list of integers that specifies the size
        of the max pooling layer in each block. The length of this list should be
        equal to `n_vgg_blocks`.

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
        in_features: int,
        n_layers: int,
        n_vgg_blocks: int,
        n_conv_layers_per_vgg_block: List[int],
        kernel_sizes_per_vgg_block: List[List[int]],
        n_channels_per_vgg_block: List[List[int]],
        vgg_pooling_kernel_size: List[int],
        d_model: int,
        ff_size: int,
        h: int,
        left_size: int,
        right_size: int,
        masking_value: int = -1e15,
    ) -> None:
        super().__init__()
        self.pre_net = VGGTransformerPreNet(
            in_features=in_features,
            n_vgg_blocks=n_vgg_blocks,
            n_layers_per_block=n_conv_layers_per_vgg_block,
            kernel_sizes_per_block=kernel_sizes_per_vgg_block,
            n_channels_per_block=n_channels_per_vgg_block,
            pooling_kernel_size=vgg_pooling_kernel_size,
            d_model=d_model,
        )
        self.enc_layers = nn.ModuleList(
            [
                TransformerEncLayerWithAttTruncation(
                    d_model=d_model,
                    ff_size=ff_size,
                    h=h,
                    left_size=left_size,
                    right_size=right_size,
                    masking_value=masking_value,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` through the encoder layers.

        Args:

            x (Tensor): The input speech tensor of shape [B, M, d]

            mask (Tensor): The input boolean mask of shape [B, M], where it's True
            if there is no padding.

        Returns:

            Tuple[Tensor, Tensor]: A tuple where the first element is the encoded speech of shape
            [B, M, F] and the second element is the lengths of shape [B].
        """
        lengths = mask.sum(dim=-1)
        out, lengths = self.pre_net(x, lengths)
        mask = get_mask_from_lens(lengths=lengths, max_len=out.shape[1])
        for layer in self.enc_layers:
            out = layer(out, mask)
        return out, lengths


class TransformerTransducerEncoder(nn.Module):
    """Implements the Transformer-Transducer encoder with relative truncated
    multi-head self attention as described in https://arxiv.org/abs/2002.02562

    Args:

        in_features (int): The input feature size.

        n_layers (int): The number of transformer encoder layers with truncated
        self attention and relative positional encoding.

        d_model (int): The model dimensionality.

        ff_size (int): The feed forward inner layer dimensionality.

        h (int): The number of heads in the attention mechanism.

        left_size (int): The size of the left window that each time step is
        allowed to look at.

        right_size (int): The size of the right window that each time step is
        allowed to look at.

        p_dropout (float): The dropout rate.

        stride (int): The stride of the convolution layer. Default 1.

        kernel_size (int): The kernel size of the convolution layer. Default 1.

        masking_value (float, optional): The value to use for masking padded
        elements. Defaults to -1e15.
    """

    def __init__(
        self,
        in_features: int,
        n_layers: int,
        d_model: int,
        ff_size: int,
        h: int,
        left_size: int,
        right_size: int,
        p_dropout: float,
        stride: int = 1,
        kernel_size: int = 1,
        masking_value: int = -1e15,
    ) -> None:
        super().__init__()
        self.pre_net = nn.Conv1d(
            in_channels=in_features,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.enc_layers = nn.ModuleList(
            [
                TransformerTransducerLayer(
                    d_model=d_model,
                    ff_size=ff_size,
                    h=h,
                    left_size=left_size,
                    right_size=right_size,
                    p_dropout=p_dropout,
                    masking_value=masking_value,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Passes the input `x` through the encoder layers.

        Args:

            x (Tensor): The input speech tensor of shape [B, M, d]

            mask (Tensor): The input boolean mask of shape [B, M], where it's True
            if there is no padding.

        Returns:

            Tuple[Tensor, Tensor]: A tuple where the first element is the encoded speech of shape
            [B, M, F] and the second element is the lengths of shape [B].
        """
        lengths = mask.sum(dim=-1)
        out = x.transpose(-1, -2)
        out = self.pre_net(out)
        out = out.transpose(-1, -2)
        lengths = calc_data_len(
            result_len=out.shape[1],
            pad_len=x.shape[1] - lengths,
            data_len=lengths,
            kernel_size=self.pre_net.kernel_size[0],
            stride=self.pre_net.stride[0],
        )
        mask = get_mask_from_lens(lengths=lengths, max_len=out.shape[1])
        for layer in self.enc_layers:
            out = layer(out, mask)
        return out, lengths
