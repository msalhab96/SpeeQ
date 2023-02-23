"""This module contains various CTC (Connectionist Temporal Classification) models for speech recognition. The CTC models are implemented as subclasses of the base class CTCModel.

Classes:

- CTCModel(nn.Module): Base class for CTC models.
- DeepSpeechV1(CTCModel): DeepSpeech version 1 model.
- BERT(nn.Module): Bidirectional Encoder Representations from Transformers (BERT) model.
- DeepSpeechV2(CTCModel): DeepSpeech version 2 model.
- Conformer(CTCModel): Conformer model.
- Jasper(CTCModel): Jasper model.
- Wav2Letter(CTCModel): Wav2Letter model.
- QuartzNet(CTCModel): QuartzNet model.
- Squeezeformer(CTCModel): Squeezeformer model.
"""
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .encoders import (
    ConformerEncoder,
    DeepSpeechV1Encoder,
    DeepSpeechV2Encoder,
    JasperEncoder,
    QuartzNetEncoder,
    SqueezeformerEncoder,
    Wav2LetterEncoder,
)
from .layers import ConvPredModule, PredModule, TransformerEncLayer


class CTCModel(nn.Module):
    """Builds the base of CTC model, if used encoder paramters has to be added,
    otherwise the forward module will raise error.
    """

    def __init__(self, pred_in_size: int, n_classes: int) -> None:
        super().__init__()
        self.has_bnorm = False
        self.pred_net = PredModule(
            in_features=pred_in_size,
            n_classes=n_classes,
            activation=nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: Tensor, mask: Tensor, *args, **kwargs):
        """passes the speech input to the model.

        Args:

            x (Tensor): The input speech signal of shape [B, M, d]

            mask (Tensor): The speech mask of shape [B, M], where it's false
            for the positions that contains padding.

        Returns:
            Tuple[Tensor, Tensor]: A tuple where the first is the predictions of shape
            [M, B, C], and the lengths tensor of shape [B].
        """
        out, lengths = self.encoder(x, mask, *args, **kwargs)  # B, M, d
        preds = self.pred_net(out)  # B, M, C
        preds = preds.permute(1, 0, 2)  # M, B, C
        return preds, lengths


class DeepSpeechV1(CTCModel):
    """Builds the DeepSpeech model described in
    https://arxiv.org/abs/1412.5567

    Args:
        in_features (int): The input feature size.

        hidden_size (int): The hidden size of the rnn layers.

        n_linear_layers (int): The number of feed-forward layers.

        bidirectional (bool): A flag indicating if the rnn is bidirectional or not.

        n_clases (int): The number of classes to predict.

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
        n_classes: int,
        max_clip_value: int,
        rnn_type: str,
        p_dropout: float,
    ) -> None:
        super().__init__(pred_in_size=hidden_size, n_classes=n_classes)
        self.encoder = DeepSpeechV1Encoder(
            in_features=in_features,
            hidden_size=hidden_size,
            n_linear_layers=n_linear_layers,
            bidirectional=bidirectional,
            max_clip_value=max_clip_value,
            rnn_type=rnn_type,
            p_dropout=p_dropout,
        )

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

        max_len (int): The maximum length for positional encoding.

        in_features (int): The input/speech feature size.

        d_model (int): The model dimensionality.

        h (int): The number of attention heads.

        ff_size (int): The inner size of the feed forward module.

        n_layers (int): The number of transformer encoders.

        n_classes (int): The number of classes.

        p_dropout (float): The dropout rate.
    """

    def __init__(
        self,
        max_len: int,
        in_features: int,
        d_model: int,
        h: int,
        ff_size: int,
        n_layers: int,
        n_classes: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=d_model,
        )
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model))
        self.layers = nn.ModuleList(
            [
                TransformerEncLayer(d_model=d_model, ff_size=ff_size, h=h)
                for _ in range(n_layers)
            ]
        )
        self.pred_module = PredModule(
            in_features=d_model, n_classes=n_classes, activation=nn.LogSoftmax(dim=-1)
        )
        self.dropout = nn.Dropout(p_dropout)
        self.has_bnorm = False

    def embed(self, x: Tensor, mask: Tensor):
        max_len = mask.sum(dim=-1).max().item()
        emb = self.pos_emb[:max_len]  # M, d
        emb = emb.unsqueeze(dim=0)  # 1, M, d
        emb = emb.repeat(mask.shape[0], 1, 1)  # B, M , d
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


class DeepSpeechV2(CTCModel):
    """Implements the deep speech model
    proposed in https://arxiv.org/abs/1512.02595

    Args:
        n_conv (int): The number of convolution layers.

        kernel_size (int): The kernel size of the convolution layers.

        stride (int): The stride size of the convolution layer.

        in_features (int): The input/speech feature size.

        hidden_size (int): The hidden size of the RNN layers.

        bidirectional (bool): A flag indicating if the rnn is bidirectional or not.

        n_rnn (int): The number of RNN layers.

        n_linear_layers (int): The number of linear layers.

        n_classes (int): The number of classes.

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
        n_classes: int,
        max_clip_value: int,
        rnn_type: str,
        tau: int,
        p_dropout: float,
    ) -> None:
        super().__init__(pred_in_size=hidden_size, n_classes=n_classes)
        self.encoder = DeepSpeechV2Encoder(
            n_conv=n_conv,
            kernel_size=kernel_size,
            stride=stride,
            in_features=in_features,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            n_rnn=n_rnn,
            n_linear_layers=n_linear_layers,
            max_clip_value=max_clip_value,
            rnn_type=rnn_type,
            tau=tau,
            p_dropout=p_dropout,
        )


class Conformer(CTCModel):
    """Implements the conformer model proposed in
    https://arxiv.org/abs/2005.08100, this model used
    with CTC, while in the paper used RNN-T.

    Args:

        n_classes (int): The number of classes.

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
        n_classes: int,
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
        super().__init__(pred_in_size=d_model, n_classes=n_classes)
        self.encoder = ConformerEncoder(
            d_model=d_model,
            n_conf_layers=n_conf_layers,
            ff_expansion_factor=ff_expansion_factor,
            h=h,
            kernel_size=kernel_size,
            ss_kernel_size=ss_kernel_size,
            ss_stride=ss_stride,
            ss_num_conv_layers=ss_num_conv_layers,
            in_features=in_features,
            res_scaling=res_scaling,
            p_dropout=p_dropout,
        )
        self.has_bnorm = True


class Jasper(CTCModel):
    """Implements Jasper model architecture proposed
    in https://arxiv.org/abs/1904.03288

    Args:

        n_classes (int): The number of classes.

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
        p_dropout: float,
    ) -> None:
        super().__init__(1, 1)
        # TODO: Add activation function options
        # TODO: Add normalization options
        # TODO: Add residual connections options
        # TODO: passing dropout list
        self.has_bnorm = True
        self.encoder = JasperEncoder(
            in_features=in_features,
            num_blocks=num_blocks,
            num_sub_blocks=num_sub_blocks,
            channel_inc=channel_inc,
            epilog_kernel_size=epilog_kernel_size,
            prelog_kernel_size=prelog_kernel_size,
            prelog_stride=prelog_stride,
            prelog_n_channels=prelog_n_channels,
            blocks_kernel_size=blocks_kernel_size,
            p_dropout=p_dropout,
        )
        self.pred_net = ConvPredModule(
            in_features=prelog_n_channels + channel_inc * (2 + num_blocks),
            n_classes=n_classes,
            activation=nn.LogSoftmax(dim=-1),
        )


class Wav2Letter(CTCModel):
    """Implements Wav2Letter model proposed in
    https://arxiv.org/abs/1609.03193

    Args:

        in_features (int): The input/speech feature size.

        n_classes (int): The number of classes.

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
        wav_stride: Optional[int] = None,
    ) -> None:
        super().__init__(1, 1)
        self.encoder = Wav2LetterEncoder(
            in_features=in_features,
            n_conv_layers=n_conv_layers,
            layers_kernel_size=layers_kernel_size,
            layers_channels_size=layers_channels_size,
            pre_conv_stride=pre_conv_stride,
            pre_conv_kernel_size=pre_conv_kernel_size,
            post_conv_channels_size=post_conv_channels_size,
            post_conv_kernel_size=post_conv_kernel_size,
            p_dropout=p_dropout,
            wav_kernel_size=wav_kernel_size,
            wav_stride=wav_stride,
        )
        self.pred_net = ConvPredModule(
            in_features=post_conv_channels_size,
            n_classes=n_classes,
            activation=nn.LogSoftmax(dim=-1),
        )


class QuartzNet(CTCModel):
    """Implements QuartzNet model architecture proposed
    in https://arxiv.org/abs/1910.10261

    Args:

        n_classes (int): The number of classes.

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
        p_dropout: float,
    ) -> None:
        super().__init__(1, 1)
        self.encoder = QuartzNetEncoder(
            in_features=in_features,
            num_blocks=num_blocks,
            block_repetition=block_repetition,
            num_sub_blocks=num_sub_blocks,
            channels_size=channels_size,
            epilog_kernel_size=epilog_kernel_size,
            epilog_channel_size=epilog_channel_size,
            prelog_kernel_size=prelog_kernel_size,
            prelog_stride=prelog_stride,
            prelog_n_channels=prelog_n_channels,
            groups=groups,
            blocks_kernel_size=blocks_kernel_size,
            p_dropout=p_dropout,
        )
        self.pred_net = ConvPredModule(
            in_features=epilog_channel_size[1],
            n_classes=n_classes,
            activation=nn.LogSoftmax(dim=-1),
        )


class Squeezeformer(CTCModel):
    """Implements the Squeezeformer model architecture
    as described in https://arxiv.org/abs/2206.00888

    Args:

        n_classes (int): The number of classes.

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
        n_classes: int,
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
        super().__init__(pred_in_size=d_model, n_classes=n_classes)
        self.encoder = SqueezeformerEncoder(
            in_features=in_features,
            n=n,
            d_model=d_model,
            ff_expansion_factor=ff_expansion_factor,
            h=h,
            kernel_size=kernel_size,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            ss_kernel_size=ss_kernel_size,
            ss_stride=ss_stride,
            ss_n_conv_layers=ss_n_conv_layers,
            p_dropout=p_dropout,
            ss_groups=ss_groups,
            masking_value=masking_value,
        )
        self.has_bnorm = True
