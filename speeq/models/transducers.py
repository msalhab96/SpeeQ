"""The transducer module provides implementations for different models used
in speech recognition based on the transducer architecture.

Classes:

- RNNTransducer: An implementation of the RNN transducer model.
- ConformerTransducer: An implementation of the Conformer transducer model.
- ContextNet: An implementation of the ContextNet transducer model.

"""
from typing import List, Tuple, Union

import torch
from torch import Tensor, nn

from speeq.constants import (
    DECODER_OUT_KEY,
    ENC_OUT_KEY,
    HIDDEN_STATE_KEY,
    PREDS_KEY,
    PREV_HIDDEN_STATE_KEY,
    SPEECH_IDX_KEY,
)

from .decoders import TransducerRNNDecoder
from .encoders import ConformerEncoder, ContextNetEncoder, RNNEncoder


class _BaseTransducer(nn.Module):
    def __init__(self, feat_size: int, n_classes: int) -> None:
        super().__init__()
        self.has_bnorm = False
        self.join_net = nn.Linear(in_features=feat_size, out_features=n_classes)

    def forward(
        self,
        speech: Tensor,
        speech_mask: Tensor,
        text: Tensor,
        text_mask: Tensor,
        *args,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Passes the input to the model

        Args:

            speech (Tensor): The input speech of shape [B, M, d]

            speech_mask (Tensor): The speech mask of shape [B, M]

            text (Tensor): The text input of shape [B, N]

            text_mask (Tensor): The text mask of shape [B, N]

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple of 3 tensors where the first
            is the predictions of shape [B, M, N, C], the last two tensor are
            the speech and text length of shape [B]
        """
        speech, speech_len = self.encoder(speech, speech_mask)
        text, text_len = self.decoder(text, text_mask)
        speech = speech.unsqueeze(-2)
        text = text.unsqueeze(1)
        result = speech + text
        result = self.join_net(result)
        speech_len, text_len = (
            speech_len.to(speech.device),
            text_len.to(speech.device),
        )
        return result, speech_len, text_len

    def predict(self, x: Tensor, mask: Tensor, state: dict) -> dict:
        if ENC_OUT_KEY not in state:
            state[ENC_OUT_KEY], _ = self.encoder(x, mask)
            state[SPEECH_IDX_KEY] = 0
            state[HIDDEN_STATE_KEY] = None
        last_hidden_state = state[HIDDEN_STATE_KEY]
        state = self.decoder.predict(state)
        speech_idx = state[SPEECH_IDX_KEY]
        out = (
            state[DECODER_OUT_KEY]
            + state[ENC_OUT_KEY][:, speech_idx : speech_idx + 1, :]
        )
        out = self.join_net(out)
        out = torch.nn.functional.log_softmax(out, dim=-1)
        out = torch.argmax(out, dim=-1)
        state[PREDS_KEY] = torch.cat([state[PREDS_KEY], out], dim=-1)
        state[PREV_HIDDEN_STATE_KEY] = last_hidden_state
        return state


class RNNTransducer(_BaseTransducer):
    """Implements the RNN transducer model proposed in
    https://arxiv.org/abs/1211.3711

    Args:

        in_features (int): The input feature size.

        n_classes (int): The number of classes/vocabulary.

        emb_dim (int): The embedding layer's size.

        n_layers (int): The number of the RNN layers in the encoder.

        hidden_size (int): The hidden size of the RNN layers.

        bidirectional (bool): A flag indicating if the rnn is bidirectional or not.

        rnn_type (str): The RNN type.

        p_dropout (float): The dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        emb_dim: int,
        n_layers: int,
        hidden_size: int,
        bidirectional: bool,
        rnn_type: str,
        p_dropout: float,
    ) -> None:
        super().__init__(feat_size=hidden_size, n_classes=n_classes)
        self.encoder = RNNEncoder(
            in_features=in_features,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            n_layers=n_layers,
            p_dropout=p_dropout,
            rnn_type=rnn_type,
        )
        self.decoder = TransducerRNNDecoder(
            vocab_size=n_classes,
            emb_dim=emb_dim,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
        )


class ConformerTransducer(RNNTransducer):
    """Implements the conformer transducer model proposed in
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

        n_classes (int): The number of classes/vocabulary.

        emb_dim (int): The embedding layer's size.

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.

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
        n_classes: int,
        emb_dim: int,
        rnn_type: str,
        p_dropout: float,
    ) -> None:
        super().__init__(
            in_features, n_classes, emb_dim, 1, d_model, False, rnn_type, p_dropout
        )
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


class ContextNet(_BaseTransducer):
    """Implements the ContextNet transducer model proposed in
    https://arxiv.org/abs/2005.03191

    Args:

        in_features (int): The input feature size.

        n_classes (int): The number of classes/vocabulary.

        emb_dim (int): The embedding layer's size.

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

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        emb_dim: int,
        n_layers: int,
        n_sub_layers: Union[int, List[int]],
        stride: Union[int, List[int]],
        out_channels: Union[int, List[int]],
        kernel_size: int,
        reduction_factor: int,
        rnn_type: str,
    ) -> None:
        super().__init__(
            out_channels[-1] if isinstance(out_channels, list) else out_channels,
            n_classes,
        )
        self.has_bnorm = True
        self.encoder = ContextNetEncoder(
            in_features=in_features,
            n_layers=n_layers,
            n_sub_layers=n_sub_layers,
            stride=stride,
            out_channels=out_channels,
            kernel_size=kernel_size,
            reduction_factor=reduction_factor,
        )
        self.decoder = TransducerRNNDecoder(
            vocab_size=n_classes,
            emb_dim=emb_dim,
            hidden_size=out_channels[-1]
            if isinstance(out_channels, list)
            else out_channels,
            rnn_type=rnn_type,
        )
