from models.decoders import GlobAttRNNDecoder
from models.layers import PyramidRNNLayers
from torch import Tensor
from torch import nn
from utils.utils import get_mask_from_lens


class LAS(nn.Module):
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
            rnn_type: str = 'rnn',
            ) -> None:
        super().__init__()
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
        self.decoder = GlobAttRNNDecoder(
            embed_dim=emb_dim,
            hidden_size=hidden_size,
            n_layers=dec_num_layers,
            n_classes=n_classes,
            pred_activation=nn.LogSoftmax(dim=-1),
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
            enc_mask=None,
            target=dec_inp
            )
        return preds
