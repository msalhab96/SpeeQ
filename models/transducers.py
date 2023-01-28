from typing import Tuple

import torch
from torch import Tensor, nn

from constants import (DECODER_OUT_KEY, ENC_OUT_KEY, HIDDEN_STATE_KEY,
                       PREDS_KEY, PREV_HIDDEN_STATE_KEY, SPEECH_IDX_KEY)
from models.decoders import RNNDecoder
from models.encoders import ConformerEncoder, RNNEncoder


class BaseTransducer(nn.Module):
    def __init__(
            self,
            feat_size: int,
            n_classes: int
    ) -> None:
        super().__init__()
        self.has_bnorm = False
        self.join_net = nn.Linear(
            in_features=feat_size,
            out_features=n_classes
        )

    def forward(
            self, speech: Tensor,
            speech_mask: Tensor, text: Tensor,
            text_mask: Tensor,
            *args, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        speech, speech_len = self.encoder(
            speech, speech_mask
        )
        text, text_len = self.decoder(
            text, text_mask
        )
        speech = speech.unsqueeze(-2)
        text = text.unsqueeze(1)
        result = speech + text
        result = self.join_net(result)
        speech_len, text_len = (
            speech_len.to(speech.device), text_len.to(speech.device)
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
        out = state[DECODER_OUT_KEY] + \
            state[ENC_OUT_KEY][:, speech_idx: speech_idx + 1, :]
        out = self.join_net(out)
        out = torch.nn.functional.log_softmax(out, dim=-1)
        out = torch.argmax(out, dim=-1)
        state[PREDS_KEY] = torch.cat([state[PREDS_KEY], out], dim=-1)
        state[PREV_HIDDEN_STATE_KEY] = last_hidden_state
        return state


class RNNTransducer(BaseTransducer):
    """Implements the RNN transducer model proposed in
    https://arxiv.org/abs/1211.3711

    Args:
        in_features (int): The input feature size.
        n_classes (int): The number of classes/vocabulary.
        emb_dim (int): The embedding layer's size.
        n_layers (int): The number of the encoder's RNN layers.
        hidden_size (int): The RNN's hidden size.
        bidirectional (bool): If the RNN is bidirectional or not.
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
            p_dropout: float
    ) -> None:
        super().__init__(
            feat_size=hidden_size, n_classes=n_classes
        )
        self.encoder = RNNEncoder(
            in_features=in_features,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            n_layers=n_layers,
            p_dropout=p_dropout,
            rnn_type=rnn_type
        )
        self.decoder = RNNDecoder(
            vocab_size=n_classes,
            emb_dim=emb_dim,
            hidden_size=hidden_size,
            rnn_type=rnn_type
        )


class ConformerTransducer(RNNTransducer):
    """Implements the conformer transducer model proposed in
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
        n_classes (int): The number of classes/vocabulary.
        emb_dim (int): The embedding layer's size.
        rnn_type (str): The RNN type.
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
            p_dropout: float
    ) -> None:
        super().__init__(in_features, n_classes, emb_dim, 1,
                         d_model, False, rnn_type, p_dropout)
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
            p_dropout=p_dropout
        )
