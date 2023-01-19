import torch
from typing import List, Optional, Tuple, Union

from utils.utils import calc_data_len, get_mask_from_lens
from .layers import (
    CReLu, ConformerBlock, ConformerPreNet,
    Conv1DLayers, JasperBlocks, JasperSubBlock,
    PredModule, QuartzBlocks, RowConv1D, TransformerEncLayer
    )
from torch import nn
from torch import Tensor


class DeepSpeechV1(nn.Module):
    """Builds the DeepSpeech model described in
    https://arxiv.org/abs/1412.5567

    Args:
        in_features (int): The input feature size.
        hidden_size (int): The layers' hidden size.
        n_linear_layers (int): The number of feed-forward layers.
        bidirectional (bidirectional): if the rnn is bidirectional
        or not.
        n_clases (int): The number of classes to predict.
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
            n_classes: int,
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
        self.pred_net = PredModule(
            in_features=hidden_size,
            n_classes=n_classes,
            activation=nn.LogSoftmax(dim=-1)
        )
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.has_bnorm = False

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        # mask of shape [B, M] and True if there's no padding
        # x of shape [B, T, F]
        lengths = mask.sum(dim=-1)
        for layer in self.ff_layers:
            x = layer(x)
        out, _, lengths = self.rnn(x, lengths.cpu())
        if self.bidirectional is True:
            out = out[..., :self.hidden_size] + out[..., self.hidden_size:]
        out = self.crelu(self.fc(out))
        preds = self.pred_net(out)
        preds = preds.permute(1, 0, 2)
        return preds, lengths

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        # x of shape [1, T, F]
        mask = torch.ones(1, x.shape[1]).long()
        preds, _ = self(x, mask)
        return preds


class BERT(nn.Module):
    """Implements the BERT Model as
    described in https://arxiv.org/abs/1810.04805

    Args:
        max_len (int): The maximum length for positional
            encoding.
        in_feature (int): The input/speech feature size.
        d_model (int): The model dimensionality.
        h (int): The number of heads.
        hidden_size (int): The inner size of the feed forward
            module.
        n_layers (int): The number of transformer encoders.
        n_classes (int): The number of classes.
        p_dropout (float): The dropout rate.
    """
    def __init__(
            self,
            max_len: int,
            in_feature: int,
            d_model: int,
            h: int,
            hidden_size: int,
            n_layers: int,
            n_classes: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_feature,
            out_features=d_model,
        )
        self.pos_emb = nn.Parameter(
            torch.randn(max_len, d_model)
            )
        self.layers = nn.ModuleList([
            TransformerEncLayer(
                d_model=d_model,
                hidden_size=hidden_size,
                h=h
                )
            for _ in range(n_layers)
        ])
        self.pred_module = PredModule(
            in_features=d_model,
            n_classes=n_classes,
            activation=nn.LogSoftmax(dim=-1)
        )
        self.dropout = nn.Dropout(p_dropout)
        self.has_bnorm = False

    def embed(self, x: Tensor, mask: Tensor):
        # this is valid as long the padding is dynamic!
        # TODO
        max_len = mask.shape[-1]
        emb = self.pos_emb[:max_len]  # M, d
        emb = emb.unsqueeze(dim=0)  # 1, M, d
        emb = emb.repeat(
            mask.shape[0], 1, 1
            )  # B, M , d
        mask = mask.unsqueeze(dim=-1)  # B, M, 1
        emb = mask * emb
        return emb + x

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        # mask of shape [B, M] and True if there's no padding
        # x of shape [B, T, F]
        lengths = mask.sum(dim=-1)
        out = self.fc(x)
        out = self.embed(out, mask)
        for layer in self.layers:
            out = layer(out, mask)
            out = self.dropout(out)
        preds = self.pred_module(out)
        preds = preds.permute(1, 0, 2)
        return preds, lengths


class DeepSpeechV2(nn.Module):
    """Implements the deep speech model
    proposed in https://arxiv.org/abs/1512.02595

    Args:
        n_conv (int): The number of convolution layers.
        kernel_size (int): The convolution layers' kernel size.
        stride (int): The convolution layers' stride.
        in_features (int): The input/speech feature size.
        hidden_size (int): The layers' hidden size.
        bidirectional (bidirectional): if the rnn is bidirectional
            or not.
        n_rnn (int): The number of RNN layers.
        n_linear_layers (int): The number of linear layers.
        n_classes (int): The number of classes.
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
            n_classes: int,
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
        self.pred_net = PredModule(
            in_features=hidden_size,
            n_classes=n_classes,
            activation=nn.LogSoftmax(dim=-1)
        )
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.has_bnorm = False

    def forward(self, x: Tensor, mask: Tensor):
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
        preds = self.pred_net(out)
        preds = preds.permute(1, 0, 2)
        return preds, lengths


class Conformer(nn.Module):
    """Implements the confformer model proposed in
    https://arxiv.org/abs/2005.08100, this model used
    with CTC, while in the paper used RNN-T.

    Args:
        n_classes (int): The number of classes.
        d_model (int): The model dimension.
        n_conf_layers (int): The number of conformer blocks.
        ff_expansion_factor (int): The feed-forward expansion factor.
        h (int): The number of heads.
        kernel_size (int): The kernel size of conv module.
        ss_kernel_size (int): The kernel size of the subsampling layer.
        in_features (int): The input/speech feature size.
        res_scaling (float): The residual connection multiplier.
        p_dropout (float): The dropout rate.
    """
    def __init__(
            self,
            n_classes: int,
            d_model: int,
            n_conf_layers: int,
            ff_expansion_factor: int,
            h: int,
            kernel_size: int,
            ss_kernel_size: int,
            in_features: int,
            res_scaling: float,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.sub_sampling = ConformerPreNet(
            in_features=in_features,
            kernel_size=ss_kernel_size,
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
        self.pred_net = PredModule(
            in_features=d_model,
            n_classes=n_classes,
            activation=nn.LogSoftmax(dim=-1)
        )
        self.has_bnorm = True

    def forward(
            self, x: Tensor, mask: Tensor
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
        preds = self.pred_net(out)
        preds = preds.permute(1, 0, 2)
        return preds, lengths


class Jasper(nn.Module):
    """Implements Jasper model architecture proposed
    in https://arxiv.org/abs/1904.03288

    Args:
        n_classes (int): The number of classes.
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
            n_classes: int,
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
        # TODO: Add activation function options
        # TODO: Add normalization options
        # TODO: Add residual connections options
        # TODO: passing dropout list
        self.has_bnorm = True
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
        self.pred_net = nn.Conv1d(
            in_channels=prelog_n_channels + channel_inc * (2 + num_blocks),
            out_channels=n_classes,
            kernel_size=1
        )
        self.log_softmax = nn.LogSoftmax(dim=-2)

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
        preds = self.pred_net(out)
        preds = self.log_softmax(preds)
        preds = preds.permute(2, 0, 1)
        return preds, lengths


class Wav2Letter(nn.Module):
    """Implements Wav2Letter model proposed in
    https://arxiv.org/abs/1609.03193

    Args:
        in_features (int): The input/speech feature size.
        n_classes (int): The number of classes.
        n_conv_layers (int): The number of convolution layers.
        layers_kernel_size (int): The convolution layers' kernel size.
        layers_channels_size (int): The convolution layers' channel size.
        pre_conv_stride (int): The prenet convolution stride.
        pre_conv_kernel_size (int): The prenet convolution kernel size.
        post_conv_channels_size (int): The postnet convolution channel size.
        post_conv_kernel_size (int): The postnet convolution kernel size.
        p_dropout (float): The dropout rate.
        wav_kernel_size (Optional[int]): The kernel size of the first
            layer that process the wav samples directly if wav is modeled.
            Default None.
        wav_stride (Optional[int]): The stride size of the first
            layer that process the wav samples directly if wav is modeled.
            Default None.
    """
    def __init__(
            self,
            in_features: int,
            n_classes: int,
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
        self.has_bnorm = False
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
        self.pred_net = nn.Conv1d(
            in_channels=post_conv_channels_size,
            out_channels=n_classes,
            kernel_size=1
        )
        self.log_softmax = nn.LogSoftmax(dim=-2)
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
        preds = self.pred_net(out)
        preds = self.log_softmax(preds)
        preds = preds.permute(2, 0, 1)
        return preds, lengths


class QuartzNet(Jasper):
    """Implements QuartzNet model architecture proposed
    in https://arxiv.org/abs/1910.10261

    Args:
        n_classes (int): The number of classes.
        in_features (int): The input/speech feature size.
        num_blocks (int): The number of QuartzNet blocks, denoted
            as 'B' in the paper.
        block_repetition (int): The nubmer of times to repeat each block.
            denoted as S in the paper.
        num_sub_blocks (int): The number of QuartzNet subblocks, denoted
            as 'R' in the paper.
        channels_size (List[int]): The channel size of each block. it has to
            be of length equal to num_blocks
        epilog_kernel_size (int): The epilog block convolution's kernel size.
        epilog_channel_size (Tuple[int, int]): The epilog blocks channels size.
        prelog_kernel_size (int): The prelog block convolution's kernel size.
        prelog_stride (int): The prelog block convolution's stride.
        prelog_n_channels (int): The prelog block convolution's number of
            output channnels.
        groups (int): The groups size.
        blocks_kernel_size (Union[int, List[int]]): The convolution layer's
            kernel size of each jasper block.
        p_dropout (float): The dropout rate.
    """
    def __init__(
            self,
            n_classes: int,
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
            p_dropout: float
            ) -> None:
        super().__init__(
            n_classes=n_classes,
            in_features=in_features,
            num_blocks=num_blocks,
            num_sub_blocks=num_sub_blocks,
            channel_inc=0,
            epilog_kernel_size=epilog_kernel_size,
            prelog_kernel_size=prelog_kernel_size,
            prelog_stride=prelog_stride,
            prelog_n_channels=prelog_n_channels,
            blocks_kernel_size=blocks_kernel_size,
            p_dropout=p_dropout
            )
        self.blocks = QuartzBlocks(
            num_blocks=num_blocks,
            block_repetition=block_repetition,
            num_sub_blocks=num_sub_blocks,
            in_channels=prelog_n_channels,
            channels_size=channels_size,
            kernel_size=blocks_kernel_size,
            groups=groups,
            p_dropout=p_dropout
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
            p_dropout=p_dropout
        )
        self.pred_net = nn.Conv1d(
            in_channels=epilog_channel_size[1],
            out_channels=n_classes,
            kernel_size=1
        )
