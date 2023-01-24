from typing import Tuple

from torch import Tensor, nn

from models.decoders import RNNDecoder
from models.encoders import RNNEncoder


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
        encoder = RNNEncoder(
            in_features=in_features,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            n_layers=n_layers,
            p_dropout=p_dropout,
            rnn_type=rnn_type
        )
        decoder = RNNDecoder(
            vocab_size=n_classes,
            emb_dim=emb_dim,
            hidden_size=hidden_size,
            rnn_type=rnn_type
        )
