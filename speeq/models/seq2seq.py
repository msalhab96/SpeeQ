"""
The module contains implementations of various sequence-to-sequence (seq2seq) speech recognition models

Classes:

- BasicAttSeq2SeqRNN: A basic seq2seq model with an RNN encoder and an attention-based RNN decoder.
- LAS: A Listen, Attend and Spell (LAS) model.
- RNNWithLocationAwareAtt: An RNN-based seq2seq model with location-aware attention mechanism.
- SpeechTransformer: A transformer-based seq2seq model for speech processing.
"""
from typing import Union

from torch import Tensor, nn

from speeq.constants import ENC_OUT_KEY, HIDDEN_STATE_KEY
from speeq.utils.utils import get_mask_from_lens

from .decoders import GlobAttRNNDecoder, LocationAwareAttDecoder, TransformerDecoder
from .encoders import PyramidRNNEncoder, RNNEncoder, SpeechTransformerEncoder


class BasicAttSeq2SeqRNN(nn.Module):
    """Implements The basic RNN encoder decoder ASR.

    Args:

        in_features (int): The encoder's input feature speech size.

        n_classes (int): The number of classes/vocabulary.

        hidden_size (int): The hidden size of the RNN layers.

        enc_num_layers (int): The number of layers in the encoder.

        bidirectional (bool): A flag indicating if the rnn is bidirectional or not.

        dec_num_layers (int): The number of the RNN layers in the decoder.

        emb_dim (int): The embedding size.

        p_dropout (float): The dropout rate.

        pred_activation (Module): An instance of an activation function.

        teacher_forcing_rate (float): The teacher forcing rate. Default 0.0

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.
        Default 'rnn'.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        hidden_size: int,
        enc_num_layers: int,
        bidirectional: bool,
        dec_num_layers: int,
        emb_dim: int,
        p_dropout: float,
        pred_activation: nn.Module,
        teacher_forcing_rate: float = 0.0,
        rnn_type: str = "rnn",
    ) -> None:
        super().__init__()
        self.has_bnorm = False
        self.encoder = RNNEncoder(
            in_features=in_features,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            n_layers=enc_num_layers,
            p_dropout=p_dropout,
            rnn_type=rnn_type,
        )
        self.decoder = GlobAttRNNDecoder(
            embed_dim=emb_dim,
            hidden_size=hidden_size,
            n_layers=dec_num_layers,
            n_classes=n_classes,
            pred_activation=pred_activation,
            teacher_forcing_rate=teacher_forcing_rate,
            rnn_type=rnn_type,
        )
        self.bidirectional = bidirectional

    def _process_hiddens(self, h):
        batch_size = h.shape[1]
        h = h.permute(1, 0, 2)
        h = h.contiguous()
        h = h.view(1, batch_size, -1)
        return h

    def forward(
        self, speech: Tensor, speech_mask: Tensor, text: Tensor, *args, **kwargs
    ) -> Tensor:
        """Passes the input to the model

        Args:
            speech (Tensor): The input speech of shape [B, M, d]

            mask (Union[Tensor, None]): The speech mask of shape [B, M],
            which is True for the data positions and False for the padding ones.


            text (Tensor): The text tensor of shape [B, M_dec]

        Returns:
            Tensor: The prediction tensor of shape [B, M_dec, C]
        """
        out, h, lengths = self.encoder(speech, speech_mask, return_h=True)
        if self.bidirectional is True:
            if isinstance(h, tuple):
                # if LSTM is used
                h = (self._process_hiddens(h[0]), self._process_hiddens(h[1]))
            else:
                h = self._process_hiddens(h)
        speech_mask = get_mask_from_lens(lengths=lengths, max_len=out.shape[1])
        speech_mask = speech_mask.to(speech.device)
        preds = self.decoder(h=h, enc_out=out, enc_mask=speech_mask, dec_inp=text)
        return preds

    def predict(self, x: Tensor, mask: Tensor, state: dict) -> dict:
        if ENC_OUT_KEY not in state:
            enc_out, h, _ = self.encoder(x, mask, return_h=True)
            if self.bidirectional is True:
                if isinstance(h, tuple):
                    # if LSTM is used
                    h = (self._process_hiddens(h[0]), self._process_hiddens(h[1]))
                else:
                    h = self._process_hiddens(h)
            state[HIDDEN_STATE_KEY] = h
            state[ENC_OUT_KEY] = enc_out
        state = self.decoder.predict(state)
        return state


class LAS(BasicAttSeq2SeqRNN):
    """Implements Listen, Attend and Spell model
    proposed in https://arxiv.org/abs/1508.01211

    Args:

        in_features (int): The encoder's input feature speech size.

        n_classes (int): The number of classes/vocabulary.

        hidden_size (int): The hidden size of the RNN layers.

        enc_num_layers (int): The number of layers in the encoder.

        reduction_factor (int): The time resolution reduction factor.

        bidirectional (bool): A flag indicating if the rnn is bidirectional or not.

        dec_num_layers (int): The number of the RNN layers in the decoder.

        emb_dim (int): The embedding size.

        p_dropout (float): The dropout rate.

        pred_activation (Module): An instance of an activation function to be
        applied on the last dimension of the predicted logits..

        teacher_forcing_rate (float): The teacher forcing rate. Default 0.0

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.
        Default 'rnn'.

    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        hidden_size: int,
        enc_num_layers: int,
        reduction_factor: int,
        bidirectional: bool,
        dec_num_layers: int,
        emb_dim: int,
        p_dropout: float,
        pred_activation: nn.Module,
        teacher_forcing_rate: float = 0.0,
        rnn_type: str = "rnn",
    ) -> None:
        super().__init__(
            in_features=in_features,
            n_classes=n_classes,
            hidden_size=hidden_size,
            enc_num_layers=enc_num_layers,
            bidirectional=bidirectional,
            dec_num_layers=dec_num_layers,
            emb_dim=emb_dim,
            p_dropout=p_dropout,
            pred_activation=pred_activation,
            teacher_forcing_rate=teacher_forcing_rate,
            rnn_type=rnn_type,
        )
        self.reduction_factor = reduction_factor
        self.encoder = PyramidRNNEncoder(
            in_features=in_features,
            hidden_size=hidden_size,
            reduction_factor=reduction_factor,
            bidirectional=bidirectional,
            n_layers=enc_num_layers,
            p_dropout=p_dropout,
            rnn_type=rnn_type,
        )


class RNNWithLocationAwareAtt(BasicAttSeq2SeqRNN):
    """Implements RNN seq2seq model proposed
        in https://arxiv.org/abs/1506.07503

    Args:

        in_features (int): The encoder's input feature speech size.

        n_classes (int): The number of classes/vocabulary.

        hidden_size (int): The hidden size of the RNN layers.

        enc_num_layers (int): The number of layers in the encoder.

        bidirectional (bool): A flag indicating if the rnn is bidirectional or not.

        dec_num_layers (int): The number of the RNN layers in the decoder.

        emb_dim (int): The embedding size.

        kernel_size (int): The attention kernel size.

        activation (str): The activation function to use in the attention layer.
        it can be either softmax or sigmax.

        p_dropout (float): The dropout rate.

        pred_activation (Module): An instance of an activation function to be
        applied on the last dimension of the predicted logits..

        inv_temperature (Union[float, int]): The inverse temperature value. Default 1.

        teacher_forcing_rate (float): The teacher forcing rate. Default 0.0

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.
        Default 'rnn'.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        hidden_size: int,
        enc_num_layers: int,
        bidirectional: bool,
        dec_num_layers: int,
        emb_dim: int,
        kernel_size: int,
        activation: str,
        p_dropout: float,
        pred_activation: nn.Module,
        inv_temperature: Union[float, int] = 1,
        teacher_forcing_rate: float = 0.0,
        rnn_type: str = "rnn",
    ) -> None:
        super().__init__(
            in_features=in_features,
            n_classes=n_classes,
            hidden_size=hidden_size,
            enc_num_layers=enc_num_layers,
            bidirectional=bidirectional,
            dec_num_layers=dec_num_layers,
            emb_dim=emb_dim,
            pred_activation=pred_activation,
            p_dropout=p_dropout,
            rnn_type=rnn_type,
        )
        self.decoder = LocationAwareAttDecoder(
            embed_dim=emb_dim,
            hidden_size=hidden_size,
            n_layers=dec_num_layers,
            n_classes=n_classes,
            pred_activation=pred_activation,
            kernel_size=kernel_size,
            activation=activation,
            inv_temperature=inv_temperature,
            teacher_forcing_rate=teacher_forcing_rate,
            rnn_type=rnn_type,
        )


class SpeechTransformer(nn.Module):
    """Implements the Speech Transformer model proposed in
    https://ieeexplore.ieee.org/document/8462506

    Args:

        in_features (int): The input/speech feature size.

        n_classes (int): The number of classes.

        n_conv_layers (int): The number of down-sampling convolutional layers.

        kernel_size (int): The kernel size of the down-sampling convolutional layers.

        stride (int): The stride size of the down-sampling convolutional layers.

        d_model (int): The model dimensionality.

        n_enc_layers (int): The number of encoder layers.

        n_dec_layers (int): The number of decoder layers.

        ff_size (int):  The dimensionality of the inner layer of the feed-forward module.

        h (int): The number of attention heads.

        att_kernel_size (int): The kernel size of the attentional convolutional layers.

        att_out_channels (int): The number of output channels of the attentional convolution layers.

        pred_activation (Module): An activation function instance to be applied on
        the last dimension of the predicted logits.

        masking_value (int): The attentin masking value. Default -1e15
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        n_conv_layers: int,
        kernel_size: int,
        stride: int,
        d_model: int,
        n_enc_layers: int,
        n_dec_layers: int,
        ff_size: int,
        h: int,
        att_kernel_size: int,
        att_out_channels: int,
        pred_activation: nn.Module,
        masking_value: int = -1e15,
    ) -> None:
        super().__init__()
        self.has_bnorm = False
        self.encoder = SpeechTransformerEncoder(
            in_features=in_features,
            n_conv_layers=n_conv_layers,
            kernel_size=kernel_size,
            stride=stride,
            d_model=d_model,
            n_layers=n_enc_layers,
            ff_size=ff_size,
            h=h,
            att_kernel_size=att_kernel_size,
            att_out_channels=att_out_channels,
        )
        self.decoder = TransformerDecoder(
            n_classes=n_classes,
            n_layers=n_dec_layers,
            d_model=d_model,
            ff_size=ff_size,
            h=h,
            pred_activation=pred_activation,
            masking_value=masking_value,
        )

    def forward(
        self,
        speech: Tensor,
        speech_mask: Tensor,
        text: Tensor,
        text_mask: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        """Passes the input to the model

        Args:

            speech (Tensor): The input speech of shape [B, M, d]

            speech_mask (Union[Tensor, None]): The speech mask of shape [B, M],
            which is True for the data positions and False for the padding ones.

            text (Tensor): The text tensor of shape [B, M_dec]

            text_mask (Union[Tensor, None]): The text mask of shape [B, M_dec],
            which is True for the data positions and False for the padding ones.

        Returns:
            Tensor: The prediction tensor of shape [B, M_dec, C]
        """
        speech, lengths = self.encoder(speech, speech_mask)
        speech_mask = get_mask_from_lens(lengths, speech.shape[1])
        preds = self.decoder(
            enc_out=speech, enc_mask=speech_mask, dec_inp=text, dec_mask=text_mask
        )
        return preds

    def predict(self, speech: Tensor, mask: Tensor, state: dict):
        if ENC_OUT_KEY not in state:
            state[ENC_OUT_KEY], _ = self.encoder(speech, mask)
        state = self.decoder.predict(state)
        return state
