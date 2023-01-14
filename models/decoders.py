import torch
from torch import nn
from torch import Tensor
from typing import Tuple, Union
from models.layers import GlobalMulAttention, PredModule


class GlobAttRNNDecoder(nn.Module):
    """Implements RNN decoder with global attention.

    Args:
        embed_dim (int): The embedding size.
        hidden_size (int): The RNN hidden size.
        n_layers (int): The number of RNN layers.
        n_classes (int): The number of classes.
        pred_activation (Module): An activation function instance.
        rnn_type (str): The RNN type to use. Default 'rnn'.
    """
    def __init__(
            self,
            embed_dim: int,
            hidden_size: int,
            n_layers: int,
            n_classes: int,
            pred_activation: nn.Module,
            rnn_type: str = 'rnn'
            ) -> None:
        super().__init__()
        self.emb = nn.Embedding(
            num_embeddings=hidden_size,
            embedding_dim=embed_dim
        )
        from .registry import RNN_REGISTRY
        self.rnn_layers = nn.ModuleList([
            RNN_REGISTRY[rnn_type](
                input_size=embed_dim,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=False
            )
            for _ in range(n_layers)
        ])
        self.att_layers = nn.ModuleList([
            GlobalMulAttention(
                enc_feat_size=hidden_size,
                dec_feat_size=hidden_size
                )
            for _ in range(n_layers)
            ])
        self.pred_net = PredModule(
            in_features=hidden_size,
            n_classes=n_classes,
            activation=pred_activation
        )
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.is_lstm = rnn_type == 'lstm'

    def forward(
            self,
            h: Union[Tensor, Tuple[Tensor, Tensor]],
            enc_h: Tensor,
            enc_mask: Tensor,
            target: Tensor,
            *args, **kwargs
            ) -> Tensor:
        # h is the encoder's last hidden state
        # target of shape [B, M]
        max_len = target.shape[-1]
        results = None
        for i in range(max_len):
            out = self.emb(target[:, i:i + 1])
            for rnn, att in zip(self.rnn_layers, self.att_layers):
                out, h = rnn(out, h)
                if self.is_lstm:
                    (h, c) = h
                h = h.permute(1, 0, 2)
                h = att(key=enc_h, query=h, mask=enc_mask)
                h = h.permute(1, 0, 2)
                if self.is_lstm:
                    h = (h, c)
            out = self.pred_net(out)
            results = out if results is None \
                else torch.cat([results, out], dim=1)
        return results
