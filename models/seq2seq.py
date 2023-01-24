from typing import Union
from models.decoders import GlobAttRNNDecoder, LocationAwareAttDecoder
from models.encoders import SpeechTransformerEncoder
from models.layers import PyramidRNNLayers, RNNLayers, TransformerDecoder
from torch import Tensor
from torch import nn
from utils.utils import get_mask_from_lens


class BasicAttSeq2SeqRNN(nn.Module):
    """Implements The basic RNN encoder decoder ASR.

    Args:
        in_features (int): The encoder's input feature speech size.
        n_classes (int): The number of classes/vocabulary.
        hidden_size (int): The RNNs' hidden size.
        enc_num_layers (int): The number of the encoder's layers.
        bidirectional (bool): If the encoder's RNNs are bidirectional or not.
        dec_num_layers (int): The number of the decoders' RNN layers.
        emb_dim (int): The embedding size.
        p_dropout (float): The dropout rate.
        teacher_forcing_rate (float): The teacher forcing rate. Default 0.0
        rnn_type (str): The rnn type. default 'rnn'.
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
            teacher_forcing_rate: float = 0.0,
            rnn_type: str = 'rnn',
            ) -> None:
        super().__init__()
        self.encoder = RNNLayers(
            in_features=in_features,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            n_layers=enc_num_layers,
            p_dropout=p_dropout,
            rnn_type=rnn_type
        )
        self.decoder = GlobAttRNNDecoder(
            embed_dim=emb_dim,
            hidden_size=hidden_size,
            n_layers=dec_num_layers,
            n_classes=n_classes,
            pred_activation=nn.LogSoftmax(dim=-1),
            teacher_forcing_rate=teacher_forcing_rate,
            rnn_type=rnn_type
        )
        self.bidirectional = bidirectional

    def _process_hiddens(self, h):
        batch_size = h.shape[1]
        h = h.permute(1, 0, 2)
        h = h.contiguous()
        h = h.view(1, batch_size, -1)
        return h

    def forward(
            self,
            enc_inp: Tensor,
            enc_mask: Tensor,
            dec_inp: Tensor,
            *args, **kwargs
            ) -> Tensor:
        out, h, lengths = self.encoder(enc_inp, enc_mask)
        if self.bidirectional is True:
            if isinstance(h, tuple):
                # if LSTM is used
                h = (
                    self._process_hiddens(h[0]),
                    self._process_hiddens(h[1])
                )
            else:
                h = self._process_hiddens(h)
        enc_mask = get_mask_from_lens(lengths=lengths, max_len=out.shape[1])
        enc_mask = enc_mask.to(enc_inp.device)
        preds = self.decoder(
            h=h,
            enc_h=out,
            enc_mask=enc_mask,
            target=dec_inp
            )
        return preds


class LAS(BasicAttSeq2SeqRNN):
    """Implements Listen, Attend and Spell model
    proposed in https://arxiv.org/abs/1508.01211

    Args:
        in_features (int): The encoder's input feature speech size.
        n_classes (int): The number of classes/vocabulary.
        hidden_size (int): The RNNs' hidden size.
        enc_num_layers (int): The number of the encoder's layers.
        reduction_factor (int): The time resolution reduction factor.
        bidirectional (bool): If the encoder's RNNs are bidirectional or not.
        dec_num_layers (int): The number of the decoders' RNN layers.
        emb_dim (int): The embedding size.
        p_dropout (float): The dropout rate.
        teacher_forcing_rate (float): The teacher forcing rate. Default 0.0
        rnn_type (str): The rnn type. default 'rnn'.
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
            teacher_forcing_rate: float = 0.0,
            rnn_type: str = 'rnn',
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
            teacher_forcing_rate=teacher_forcing_rate,
            rnn_type=rnn_type
            )
        self.reduction_factor = reduction_factor
        self.encoder = PyramidRNNLayers(
            in_features=in_features,
            hidden_size=hidden_size,
            reduction_factor=reduction_factor,
            bidirectional=bidirectional,
            n_layers=enc_num_layers,
            p_dropout=p_dropout,
            rnn_type=rnn_type
        )


class RNNWithLocationAwareAtt(BasicAttSeq2SeqRNN):
    """Implements RNN seq2seq model proposed
        in https://arxiv.org/abs/1506.07503

    Args:
        in_features (int): The encoder's input feature speech size.
        n_classes (int): The number of classes/vocabulary.
        hidden_size (int): The RNNs' hidden size.
        enc_num_layers (int): The number of the encoder's layers.
        bidirectional (bool): If the encoder's RNNs are bidirectional or not.
        dec_num_layers (int): The number of the decoders' RNN layers.
        emb_dim (int): The embedding size.
        kernel_size (int): The attention kernel size.
        activation (str): The activation function to use in the
            attention layer. it can be either softmax or sigmax.
        p_dropout (float): The dropout rate.
        inv_temperature (Union[float, int]): The inverse temperature value of
            the attention. Default 1.
        teacher_forcing_rate (float): The teacher forcing rate. Default 0.0
        rnn_type (str): The rnn type. default 'rnn'.
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
            inv_temperature: Union[float, int] = 1,
            teacher_forcing_rate: float = 0.0,
            rnn_type: str = 'rnn'
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
            rnn_type=rnn_type
            )
        self.decoder = LocationAwareAttDecoder(
            embed_dim=emb_dim,
            hidden_size=hidden_size,
            n_layers=dec_num_layers,
            n_classes=n_classes,
            pred_activation=nn.LogSoftmax(dim=-1),
            kernel_size=kernel_size,
            activation=activation,
            inv_temperature=inv_temperature,
            teacher_forcing_rate=teacher_forcing_rate,
            rnn_type=rnn_type
        )


class SpeechTransformer(nn.Module):
    """Implements the Speech Transformer model proposed in
    https://ieeexplore.ieee.org/document/8462506

    Args:
        in_features (int): The input/speech feature size.
        n_classes (int): The number of classes.
        n_conv_layers (int): The number of down-sampling convolutional layers.
        kernel_size (int): The down-sampling convolutional layers kernel size.
        stride (int): The down-sampling convolutional layers stride.
        d_model (int): The model dimensionality.
        n_enc_layers (int): The number of encoder layers.
        n_dec_layers (int): The number of decoder layers.
        ff_size (int): The feed-forward inner layer dimensionality.
        h (int): The number of attention heads.
        att_kernel_size (int): The attentional convolutional
            layers' kernel size.
        att_out_channels (int): The number of output channels of the
            attentional convolution
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
            masking_value: int = -1e15
            ) -> None:
        super().__init__()
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
            att_out_channels=att_out_channels
        )
        self.decoder = TransformerDecoder(
            n_classes=n_classes,
            n_layers=n_dec_layers,
            d_model=d_model,
            ff_size=ff_size,
            h=h,
            masking_value=masking_value
        )

    def forward(
            self, speech: Tensor,
            speech_mask: Tensor, text: Tensor,
            text_mask: Tensor,
            *args, **kwargs
            ) -> Tensor:
        speech, lengths = self.encoder(
            speech, speech_mask
        )
        speech_mask = get_mask_from_lens(
            lengths, speech.shape[1]
            )
        preds = self.decoder(
            enc_out=speech,
            enc_mask=speech_mask,
            dec_inp=text,
            dec_mask=text_mask
        )
        return preds
