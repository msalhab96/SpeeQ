from typing import List, Optional, Tuple, Union
from models.layers import (
    CReLu, ConformerBlock, ConformerPreNet, Conv1DLayers,
    JasperBlocks, JasperSubBlock, RowConv1D
)
from torch import nn
from torch import Tensor
from utils.utils import calc_data_len, get_mask_from_lens
import torch


class DeepSpeechV1Encoder(nn.Module):
    """Builds the DeepSpeech encoder described in
    https://arxiv.org/abs/1412.5567

    Args:
        in_features (int): The input feature size.
        hidden_size (int): The layers' hidden size.
        n_linear_layers (int): The number of feed-forward layers.
        bidirectional (bool): if the rnn is bidirectional or not.
        max_clip_value (int): The maximum relu value.
        rnn_type (str): rnn, gru or lstm.
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
            p_dropout: float
    ) -> None:
        super().__init__()
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    in_features=in_features if i == 0 else hidden_size,
                    out_features=hidden_size
                ),
                CReLu(
                    max_val=max_clip_value
                ),
                nn.Dropout(
                    p=p_dropout
                )
            )
            for i in range(n_linear_layers)
        ])
        from .registry import PACKED_RNN_REGISTRY
        self.rnn = PACKED_RNN_REGISTRY[rnn_type](
            input_size=hidden_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional
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
        # mask of shape [B, M] and True if there's no padding
        # x of shape [B, T, F]
        lengths = mask.sum(dim=-1)
        for layer in self.ff_layers:
            x = layer(x)
        out, _, lengths = self.rnn(x, lengths.cpu())
        if self.bidirectional is True:
            out = out[..., :self.hidden_size] + out[..., self.hidden_size:]
        out = self.crelu(self.fc(out))
        return out, lengths


class DeepSpeechV2Encoder(nn.Module):
    """Implements the deep speech 2 encoder
    proposed in https://arxiv.org/abs/1512.02595

    Args:
        n_conv (int): The number of convolution layers.
        kernel_size (int): The convolution layers' kernel size.
        stride (int): The convolution layers' stride.
        in_features (int): The input/speech feature size.
        hidden_size (int): The layers' hidden size.
        bidirectional (bool): if the rnn is bidirectional or not.
        n_rnn (int): The number of RNN layers.
        n_linear_layers (int): The number of linear layers.
        max_clip_value (int): The maximum relu value.
        rnn_type (str): rnn, gru or lstm.
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
            p_dropout: float
    ) -> None:
        super().__init__()
        self.conv = Conv1DLayers(
            in_size=in_features,
            out_size=hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            n_layers=n_conv,
            p_dropout=p_dropout
        )
        from .registry import PACKED_RNN_REGISTRY
        self.rnns = nn.ModuleList(
            [
                PACKED_RNN_REGISTRY[rnn_type](
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    bidirectional=bidirectional
                )
                for _ in range(n_rnn)
            ]
        )
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=hidden_size,
                    out_features=hidden_size
                )
                for _ in range(n_linear_layers)
            ]
        )
        self.crelu = CReLu(max_val=max_clip_value)
        self.context_conv = RowConv1D(
            tau=tau, hidden_size=hidden_size
        )
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def forward(
            self, x: Tensor, mask: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        lengths = mask.sum(dim=-1)
        lengths = lengths.cpu()
        out, lengths = self.conv(x, lengths)
        out = self.crelu(out)
        for layer in self.rnns:
            out, _, lengths = layer(
                out, lengths
            )
            if self.bidirectional is True:
                out = out[..., :self.hidden_size] +\
                    out[..., self.hidden_size:]
            out = self.crelu(out)
        out = self.context_conv(out)
        for layer in self.linear_layers:
            out = layer(out)
            out = self.crelu(out)
        return out, lengths


class ConformerEncoder(nn.Module):
    """Implements the conformer encoder proposed in
    https://arxiv.org/abs/2005.08100

    Args:
        d_model (int): The model dimension.
        n_conf_layers (int): The number of conformer blocks.
        ff_expansion_factor (int): The feed-forward expansion factor.
        h (int): The number of heads.
        kernel_size (int): The kernel size of conv module.
        ss_kernel_size (int): The kernel size of the subsampling layer.
        ss_stride (int): The stride of the subsampling layer.
        ss_num_conv_layers (int): The number of subsampling layers.
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
            p_dropout: float
    ) -> None:
        super().__init__()
        self.sub_sampling = ConformerPreNet(
            in_features=in_features,
            kernel_size=ss_kernel_size,
            stride=ss_stride,
            n_conv_layers=ss_num_conv_layers,
            d_model=d_model,
            p_dropout=p_dropout
        )
        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                ff_expansion_factor=ff_expansion_factor,
                h=h, kernel_size=kernel_size,
                p_dropout=p_dropout, res_scaling=res_scaling
            )
            for _ in range(n_conf_layers)
        ])

    def forward(
            self, x: Tensor, mask: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        lengths = mask.sum(dim=-1)
        lengths = lengths.cpu()
        out, lengths = self.sub_sampling(x, lengths)
        mask = get_mask_from_lens(
            lengths, lengths.max().item()
        )
        mask = mask.to(x.device)
        for layer in self.blocks:
            out = layer(out, mask)
        return out, lengths


class JasperEncoder(nn.Module):
    """Implements Jasper model architecture's encoder proposed
    in https://arxiv.org/abs/1904.03288

    Args:
        in_features (int): The input/speech feature size.
        num_blocks (int): The number of jasper blocks, denoted
            as 'B' in the paper.
        num_sub_blocks (int): The number of jasper subblocks, denoted
            as 'R' in the paper.
        channel_inc (int): The rate to increase the number of channels
            across the blocks.
        epilog_kernel_size (int): The epilog block convolution's kernel size.
        prelog_kernel_size (int): The prelog block convolution's kernel size.
        prelog_stride (int): The prelog block convolution's stride.
        prelog_n_channels (int): The prelog block convolution's number of
            output channnels.
        blocks_kernel_size (Union[int, List[int]]): The convolution layer's
            kernel size of each jasper block.
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
            p_dropout: float
    ) -> None:
        super().__init__()
        self.prelog = JasperSubBlock(
            in_channels=in_features,
            out_channels=prelog_n_channels,
            kernel_size=prelog_kernel_size,
            p_dropout=p_dropout,
            padding=0,
            stride=prelog_stride
        )
        self.prelog_stride = prelog_stride
        self.prelog_kernel_size = prelog_kernel_size
        self.blocks = JasperBlocks(
            num_blocks=num_blocks,
            num_sub_blocks=num_sub_blocks,
            in_channels=prelog_n_channels,
            channel_inc=channel_inc,
            kernel_size=blocks_kernel_size,
            p_dropout=p_dropout
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
            p_dropout=p_dropout
        )

    def forward(
            self, x: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # x of shape [B, M, d]
        lengths = mask.sum(dim=-1)
        lengths = lengths.cpu()
        x = x.transpose(-1, -2)
        out = self.prelog(x)
        lengths = calc_data_len(
            result_len=out.shape[-1],
            pad_len=x.shape[-1] - lengths,
            data_len=lengths,
            kernel_size=self.prelog_kernel_size,
            stride=self.prelog_stride
        )
        out = self.blocks(out)
        out = self.epilog1(out)
        out = self.epilog2(out)
        return out, lengths


class Wav2LetterEncoder(nn.Module):
    """Implements Wav2Letter's encoder proposed in
    https://arxiv.org/abs/1609.03193

    Args:
        in_features (int): The input/speech feature size.
        n_conv_layers (int): The number of convolution layers.
        layers_kernel_size (int): The convolution layers' kernel size.
        layers_channels_size (int): The convolution layers' channel size.
        pre_conv_stride (int): The prenet convolution stride.
        pre_conv_kernel_size (int): The prenet convolution kernel size.
        post_conv_channels_size (int): The postnet convolution channel size.
        post_conv_kernel_size (int): The postnet convolution kernel size.
        p_dropout (float): The dropout rate.
        wav_kernel_size (Optional[int]): The kernel size of the first
            layer that processes the wav samples directly if wav is modeled.
            Default None.
        wav_stride (Optional[int]): The stride size of the first
            layer that processes the wav samples directly if wav is modeled.
            Default None.
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
            wav_stride: Optional[int] = None
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
                stride=wav_stride
            )
        self.pre_conv = nn.Conv1d(
            in_channels=layers_channels_size if self.is_wav else in_features,
            out_channels=layers_channels_size,
            kernel_size=pre_conv_kernel_size,
            stride=pre_conv_stride
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=layers_channels_size,
                out_channels=layers_channels_size,
                kernel_size=layers_kernel_size,
                padding='same'
            )
            for _ in range(n_conv_layers - 1)
        ])
        self.convs.append(
            nn.Conv1d(
                in_channels=layers_channels_size,
                out_channels=post_conv_channels_size,
                kernel_size=post_conv_kernel_size,
                padding='same'
            )
        )
        self.post_conv = nn.Conv1d(
            in_channels=post_conv_channels_size,
            out_channels=post_conv_channels_size,
            kernel_size=1,
            padding='same'
        )
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        # x of shape [B, M, d]
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
                stride=self.raw_conv.stride[0]
            )
        results = self.pre_conv(out)
        lengths = calc_data_len(
            result_len=results.shape[-1],
            pad_len=out.shape[-1] - lengths,
            data_len=lengths,
            kernel_size=self.pre_conv.kernel_size[0],
            stride=self.pre_conv.stride[0]
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
        return out, lengths
