from typing import Tuple, Union

import torch
from torch import Tensor, nn

from models.layers import (GlobalMulAttention, LocAwareGlobalAddAttention,
                           PositionalEmbedding, PredModule,
                           TransformerDecLayer)


class GlobAttRNNDecoder(nn.Module):
    """Implements RNN decoder with global attention.

    Args:
        embed_dim (int): The embedding size.
        hidden_size (int): The RNN hidden size.
        n_layers (int): The number of RNN layers.
        n_classes (int): The number of classes.
        pred_activation (Module): An activation function instance.
        teacher_forcing_rate (float): The teacher forcing rate. Default 0.0
        rnn_type (str): The RNN type to use. Default 'rnn'.
    """

    def __init__(
            self,
            embed_dim: int,
            hidden_size: int,
            n_layers: int,
            n_classes: int,
            pred_activation: nn.Module,
            teacher_forcing_rate: float = 0.0,
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
                input_size=embed_dim if i == 0 else hidden_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=False
            )
            for i in range(n_layers)
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
        self.teacher_forcing_rate = teacher_forcing_rate

    def _apply_teacher_forcing(self, y: Tensor, out: Tensor) -> Tensor:
        # y of shape [B, 1]
        # out of shape [B, 1, C]
        """Applies teacher forcing on the decoder's input.

        Args:
            y (Tensor): The original target labels.
            out (Tensor): The latest predicted probabilities.

        Returns:
            Tensor: The new decoder input tensor.
        """
        mask = torch.rand(y.shape[0]) <= self.teacher_forcing_rate
        mask = mask.to(y.device)
        mask = mask.unsqueeze(dim=-1)
        out = torch.argmax(out, dim=-1)
        return (~mask) * y + mask * out

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
        out = self.emb(target[:, 0: 1])
        h = [h] * len(self.rnn_layers)
        for i in range(max_len):
            for j, (rnn, att) in enumerate(
                    zip(self.rnn_layers, self.att_layers)
            ):
                out, h_ = rnn(out, h[j])
                if self.is_lstm:
                    (h_, c_) = h_
                h_ = h_.permute(1, 0, 2)
                h_ = att(key=enc_h, query=h_, mask=enc_mask)
                h_ = h_.permute(1, 0, 2)
                if self.is_lstm:
                    h[j] = (h_, c_)
                else:
                    h[j] = h_
            out = self.pred_net(out)
            results = out if results is None \
                else torch.cat([results, out], dim=1)
            y = target[:, i: i + 1]
            if self.teacher_forcing_rate > 0:
                y = self._apply_teacher_forcing(y=y, out=out)
            out = self.emb(y)
        return results


class LocationAwareAttDecoder(GlobAttRNNDecoder):
    """Implements RNN decoder with location aware attention.

    Args:
        embed_dim (int): The embedding size.
        hidden_size (int): The RNN hidden size.
        n_layers (int): The number of RNN layers.
        n_classes (int): The number of classes.
        pred_activation (Module): An activation function instance.
        kernel_size (int): The attention kernel size.
        activation (str): The activation function to use.
            it can be either softmax or sigmax.
        inv_temperature (Union[float, int]): The inverse temperature value.
            Default 1.
        teacher_forcing_rate (float): The teacher forcing rate. Default 0.0
        rnn_type (str): The RNN type to use. Default 'rnn'.
    """

    def __init__(
            self,
            embed_dim: int,
            hidden_size: int,
            n_layers: int,
            n_classes: int,
            pred_activation: nn.Module,
            kernel_size: int,
            activation: str,
            inv_temperature: Union[float, int] = 1,
            teacher_forcing_rate: float = 0.0,
            rnn_type: str = 'rnn'
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_classes=n_classes,
            pred_activation=pred_activation,
            teacher_forcing_rate=teacher_forcing_rate,
            rnn_type=rnn_type
        )
        self.att_layers = nn.ModuleList([
            LocAwareGlobalAddAttention(
                enc_feat_size=hidden_size,
                dec_feat_size=hidden_size,
                kernel_size=kernel_size,
                activation=activation,
                inv_temperature=inv_temperature
            )
            for _ in range(n_layers)
        ])

    def forward(
            self,
            h: Union[Tensor, Tuple[Tensor, Tensor]],
            enc_h: Tensor,
            enc_mask: Tensor,
            target: Tensor,
            *args, **kwargs
    ) -> Tensor:
        # target of shape [B, M]
        batch_size, max_len = target.shape
        results = None
        alpha = torch.zeros(batch_size, 1, enc_h.shape[1]).to(enc_h.device)
        out = self.emb(target[:, 0: 1])
        h = [h] * len(self.rnn_layers)
        for i in range(max_len):
            for j, (rnn, att) in enumerate(
                    zip(self.rnn_layers, self.att_layers)
            ):
                out, h_ = rnn(out, h[j])
                if self.is_lstm:
                    (h_, c_) = h_
                h_ = h_.permute(1, 0, 2)
                h_, alpha = att(
                    key=enc_h, query=h_, alpha=alpha, mask=enc_mask
                )
                h = h.permute(1, 0, 2)
                if self.is_lstm:
                    h[j] = (h_, c_)
                else:
                    h[j] = h_
            out = self.pred_net(out)
            results = out if results is None \
                else torch.cat([results, out], dim=1)
            y = target[:, i: i + 1]
            if self.teacher_forcing_rate > 0:
                y = self._apply_teacher_forcing(y=y, out=out)
            out = self.emb(y)
        return results


class RNNDecoder(nn.Module):
    """Builds a simple RNN-decoder that contains embedding layer
    and a single RNN layer

    Args:
        vocab_size (int): The vocabulary size.
        emb_dim (int): The embedding dimension.
        hidden_size (int): The RNN's hidden size.
        rnn_type (str): The RNN type.
    """

    def __init__(
            self,
            vocab_size: int,
            emb_dim: int,
            hidden_size: int,
            rnn_type: str
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=emb_dim
        )
        from .registry import PACKED_RNN_REGISTRY
        self.rnn = PACKED_RNN_REGISTRY[rnn_type](
            input_size=emb_dim,
            hidden_size=hidden_size,
            batch_first=True,
            enforce_sorted=False,
            bidirectional=False
        )

    def forward(
            self, x: Tensor, mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        lengths = mask.sum(dim=-1).cpu()
        out = self.emb(x)
        out, _, lens = self.rnn(out, lengths)
        return out, lens


class TransformerDecoder(nn.Module):
    """Implements the transformer decoder as described in
    https://arxiv.org/abs/1706.03762

    Args:
        n_classes (int): The number of classes.
        n_layers (int): The nnumber of decoder layers.
        d_model (int): The model dimensionality.
        ff_size (int): The feed-forward inner layer dimensionality.
        h (int): The number of attentional heads.
        masking_value (int): The attentin masking value. Default -1e15
    """

    def __init__(
            self,
            n_classes: int,
            n_layers: int,
            d_model: int,
            ff_size: int,
            h: int,
            masking_value: int = -1e15
    ) -> None:
        super().__init__()
        self.emb = PositionalEmbedding(
            vocab_size=n_classes,
            embed_dim=d_model
        )
        self.layers = nn.ModuleList(
            [
                TransformerDecLayer(
                    d_model=d_model,
                    hidden_size=ff_size,
                    h=h, masking_value=masking_value
                )
                for _ in range(n_layers)
            ]
        )
        self.pred_net = PredModule(
            in_features=d_model,
            n_classes=n_classes,
            activation=nn.LogSoftmax(dim=-1)
        )

    def forward(
            self,
            enc_out: Tensor,
            enc_mask: Union[Tensor, None],
            dec_inp: Tensor,
            dec_mask: Union[Tensor, None],
    ) -> Tensor:
        out = self.emb(dec_inp)
        for layer in self.layers:
            out = layer(
                enc_out=enc_out,
                enc_mask=enc_mask,
                dec_inp=out,
                dec_mask=dec_mask
            )
        out = self.pred_net(out)
        return out
