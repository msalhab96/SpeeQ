from typing import Tuple, Union

import torch
from torch import Tensor, nn

from speeq.constants import DECODER_OUT_KEY, ENC_OUT_KEY, HIDDEN_STATE_KEY, PREDS_KEY

from .layers import (
    GlobalMulAttention,
    LocAwareGlobalAddAttention,
    PositionalEmbedding,
    PredModule,
    TransformerDecLayer,
)


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
        rnn_type: str = "rnn",
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=n_classes, embedding_dim=embed_dim)
        from .registry import RNN_REGISTRY

        self.rnn_layers = nn.ModuleList(
            [
                RNN_REGISTRY[rnn_type](
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    bidirectional=False,
                )
                for i in range(n_layers)
            ]
        )
        self.fc_layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=hidden_size + embed_dim if i == 0 else 2 * hidden_size,
                    out_features=hidden_size,
                )
                for i in range(n_layers)
            ]
        )
        self.att_layers = nn.ModuleList(
            [
                GlobalMulAttention(enc_feat_size=hidden_size, dec_feat_size=hidden_size)
                for _ in range(n_layers)
            ]
        )
        self.pred_net = PredModule(
            in_features=hidden_size, n_classes=n_classes, activation=pred_activation
        )
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.is_lstm = rnn_type == "lstm"
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

    def _init_hidden_state(self, batch_size, device):
        if self.is_lstm:
            return (
                torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device),
            )
        return torch.zeros(1, batch_size, self.hidden_size).to(device)

    def forward(
        self,
        h: Union[Tensor, Tuple[Tensor, Tensor], None],
        enc_out: Tensor,
        enc_mask: Tensor,
        dec_inp: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        # h is the encoder's last hidden state
        # dec_inp of shape [B, M]
        batch_size, max_len = dec_inp.shape
        if h is None:
            self._init_hidden_state(batch_size=batch_size, device=dec_inp.device)
        results = None
        out = self.emb(dec_inp[:, 0:1])
        h = [h] * len(self.rnn_layers)
        for i in range(max_len):
            layers = enumerate(zip(self.fc_layers, self.rnn_layers, self.att_layers))
            for j, (fc, rnn, att) in layers:
                h_ = h[j]
                if self.is_lstm:
                    (h_, c_) = h_
                h_ = h_.permute(1, 0, 2)
                out = torch.cat([out, h_], dim=-1)
                out = fc(out)
                out = att(key=enc_out, query=out, mask=enc_mask)
                out, h[j] = rnn(out, h[j])
            out = self.pred_net(out)
            results = out if results is None else torch.cat([results, out], dim=1)
            y = dec_inp[:, i : i + 1]
            if self.teacher_forcing_rate > 0:
                y = self._apply_teacher_forcing(y=y, out=out)
            out = self.emb(y)
        return results

    def predict(self, state: dict) -> Tuple[Tensor, dict, Tensor]:
        enc_out = state[ENC_OUT_KEY]
        preds = state[PREDS_KEY]  # [B, M]
        h = state[HIDDEN_STATE_KEY]
        last_pred = preds[:, -1:]
        if isinstance(h, list) is False:
            # for the first prediction iteration
            h = [h] * len(self.rnn_layers)
        out = self.emb(last_pred)
        layers = enumerate(zip(self.fc_layers, self.rnn_layers, self.att_layers))
        for i, (fc, rnn, att) in layers:
            h_ = h[i]
            if self.is_lstm:
                (h_, c_) = h_
            h_ = h_.permute(1, 0, 2)
            out = torch.cat([out, h_], dim=-1)
            out = fc(out)
            out = att(key=enc_out, query=out, mask=None)
            out, h[i] = rnn(out, h[i])
        out = self.pred_net(out)
        state[PREDS_KEY] = torch.cat(
            [state[PREDS_KEY], torch.argmax(out, dim=-1)], dim=-1
        )
        state[HIDDEN_STATE_KEY] = h
        return state


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
        rnn_type: str = "rnn",
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_classes=n_classes,
            pred_activation=pred_activation,
            teacher_forcing_rate=teacher_forcing_rate,
            rnn_type=rnn_type,
        )
        self.att_layers = nn.ModuleList(
            [
                LocAwareGlobalAddAttention(
                    enc_feat_size=hidden_size,
                    dec_feat_size=hidden_size,
                    kernel_size=kernel_size,
                    activation=activation,
                    inv_temperature=inv_temperature,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        h: Union[Tensor, Tuple[Tensor, Tensor], None],
        enc_out: Tensor,
        enc_mask: Tensor,
        dec_inp: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        # dec_inp of shape [B, M]
        batch_size, max_len = dec_inp.shape
        results = None
        if h is None:
            self._init_hidden_state(batch_size=batch_size, device=dec_inp.device)
        alpha = torch.zeros(batch_size, 1, enc_out.shape[1]).to(enc_out.device)
        out = self.emb(dec_inp[:, 0:1])
        h = [h] * len(self.rnn_layers)
        for i in range(max_len):
            for j, (fc, rnn, att) in enumerate(
                zip(self.fc_layers, self.rnn_layers, self.att_layers)
            ):
                h_ = h[j]
                if self.is_lstm:
                    (h_, c_) = h_
                h_ = h_.permute(1, 0, 2)
                out = torch.cat([out, h_], dim=-1)
                out = fc(out)
                out, alpha = att(key=enc_out, query=out, alpha=alpha, mask=enc_mask)
                out, h[j] = rnn(out, h[j])
            out = self.pred_net(out)
            results = out if results is None else torch.cat([results, out], dim=1)
            y = dec_inp[:, i : i + 1]
            if self.teacher_forcing_rate > 0:
                y = self._apply_teacher_forcing(y=y, out=out)
            out = self.emb(y)
        return results

    def predict(self, state: dict) -> Tuple[Tensor, dict, Tensor]:
        alpha_key = "alpha"
        enc_out = state[ENC_OUT_KEY]
        batch_size, _, hidden_size = enc_out.shape
        last_pred = state[PREDS_KEY][:, -1:]
        h = state[HIDDEN_STATE_KEY]
        alpha = state.get(alpha_key, torch.zeros(batch_size, 1, hidden_size))
        alpha = alpha.to(enc_out.device)
        if isinstance(h, list) is False:
            h = [h] * len(self.rnn_layers)
        out = self.emb(last_pred)
        for i, (fc, rnn, att) in enumerate(
            zip(self.fc_layers, self.rnn_layers, self.att_layers)
        ):
            h_ = h[i]
            if self.is_lstm:
                (h_, c_) = h_
            h_ = h_.permute(1, 0, 2)
            out = torch.cat([out, h_], dim=-1)
            out = fc(out)
            out, alpha = att(key=enc_out, query=out, alpha=alpha, mask=None)
            out, h[i] = rnn(out, h[i])
        out = self.pred_net(out)
        state[PREDS_KEY] = torch.cat(
            [state[PREDS_KEY], torch.argmax(out, dim=-1)], dim=-1
        )
        state[HIDDEN_STATE_KEY] = h
        state[alpha_key] = alpha
        return state


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
        self, vocab_size: int, emb_dim: int, hidden_size: int, rnn_type: str
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        from .registry import PACKED_RNN_REGISTRY

        self.rnn = PACKED_RNN_REGISTRY[rnn_type](
            input_size=emb_dim,
            hidden_size=hidden_size,
            batch_first=True,
            enforce_sorted=False,
            bidirectional=False,
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        lengths = mask.sum(dim=-1).cpu()
        out = self.emb(x)
        out, _, lens = self.rnn(out, lengths)
        return out, lens

    def predict(self, state: dict) -> dict:
        h = state[HIDDEN_STATE_KEY]
        last_pred = state[PREDS_KEY][:, -1:]
        lens = torch.ones(last_pred.shape[0], dtype=torch.long)
        out = self.emb(last_pred)
        out, h, _ = self.rnn(out, lens, h)
        state[HIDDEN_STATE_KEY] = h
        state[DECODER_OUT_KEY] = out
        return state


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
        masking_value: int = -1e15,
    ) -> None:
        super().__init__()
        self.emb = PositionalEmbedding(vocab_size=n_classes, embed_dim=d_model)
        self.layers = nn.ModuleList(
            [
                TransformerDecLayer(
                    d_model=d_model, ff_size=ff_size, h=h, masking_value=masking_value
                )
                for _ in range(n_layers)
            ]
        )
        self.pred_net = PredModule(
            in_features=d_model, n_classes=n_classes, activation=nn.LogSoftmax(dim=-1)
        )

    def forward(
        self,
        enc_out: Tensor,
        enc_mask: Union[Tensor, None],
        dec_inp: Tensor,
        dec_mask: Union[Tensor, None],
        *args,
        **kwargs
    ) -> Tensor:
        out = self.emb(dec_inp)
        for layer in self.layers:
            out = layer(
                enc_out=enc_out, enc_mask=enc_mask, dec_inp=out, dec_mask=dec_mask
            )
        out = self.pred_net(out)
        return out

    def predict(self, state: dict) -> dict:
        preds = state[PREDS_KEY]
        out = self.emb(preds)
        for layer in self.layers:
            out = layer(
                enc_out=state[ENC_OUT_KEY], enc_mask=None, dec_inp=out, dec_mask=None
            )
        out = self.pred_net(out[:, -1:, :])
        last_pred = torch.argmax(out, dim=-1)
        state[PREDS_KEY] = torch.cat([state[PREDS_KEY], last_pred], dim=-1)
        return state
