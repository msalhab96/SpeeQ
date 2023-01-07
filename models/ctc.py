import torch
from typing import Tuple
from .layers import (
    CReLu, Conv1DLayers, PredModule,
    RowConv1D, TransformerEncLayer
    )
from torch import nn
from torch import Tensor


class DeepSpeechV1(nn.Module):
    """Builds the DeepSpeech model described in
    https://arxiv.org/abs/1412.5567

    Args:
        in_features (int): The input feature size.
        hidden_size (int): The layers' hidden size.
        n_linear_layers (int): The number of feed-forward layers.
        bidirectional (bidirectional): if the rnn is bidirectional
        or not.
        n_clases (int): The number of classes to predict.
        max_clip_value (int): The maximum relu value.
        rnn_type (str): rnn, gru or lstm.
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
            p_dropout: float
            ) -> None:
        super().__init__()
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    in_features=in_features if i == 0 else hidden_size,
                    out_features=hidden_size
                    ),
                CReLu(
                    max_val=max_clip_value
                    ),
                nn.Dropout(
                    p=p_dropout
                    )
            )
            for i in range(n_linear_layers)
        ])
        from .registry import RNN_REGISTRY
        self.rnn = RNN_REGISTRY[rnn_type](
            input_size=hidden_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
        )
        self.crelu = CReLu(max_val=max_clip_value)
        self.pred_net = PredModule(
            in_features=hidden_size,
            n_classes=n_classes,
            activation=nn.LogSoftmax(dim=-1)
        )
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        # mask of shape [B, M] and True if there's no padding
        # x of shape [B, T, F]
        lengths = mask.sum(dim=-1)
        for layer in self.ff_layers:
            x = layer(x)
        out, _, lengths = self.rnn(x, lengths.cpu())
        if self.bidirectional is True:
            out = out[..., :self.hidden_size] + out[..., self.hidden_size:]
        out = self.crelu(self.fc(out))
        preds = self.pred_net(out)
        preds = preds.permute(1, 0, 2)
        return preds, lengths

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
        max_len (int): The maximum length for positional
            encoding.
        in_feature (int): The input/speech feature size.
        d_model (int): The model dimensionality.
        h (int): The number of heads.
        hidden_size (int): The inner size of the feed forward
            module.
        n_layers (int): The number of transformer encoders.
        n_classes (int): The number of classes.
        p_dropout (float): The dropout rate.
    """
    def __init__(
            self,
            max_len: int,
            in_feature: int,
            d_model: int,
            h: int,
            hidden_size: int,
            n_layers: int,
            n_classes: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_feature,
            out_features=d_model,
        )
        self.pos_emb = nn.Parameter(
            torch.randn(max_len, d_model)
            )
        self.layers = nn.ModuleList([
            TransformerEncLayer(
                d_model=d_model,
                hidden_size=hidden_size,
                h=h
                )
            for _ in range(n_layers)
        ])
        self.pred_module = PredModule(
            in_features=d_model,
            n_classes=n_classes,
            activation=nn.LogSoftmax(dim=-1)
        )
        self.dropout = nn.Dropout(p_dropout)

    def embed(self, x: Tensor, mask: Tensor):
        # this is valid as long the padding is dynamic!
        # TODO
        max_len = mask.shape[-1]
        emb = self.pos_emb[:max_len]  # M, d
        emb = emb.unsqueeze(dim=0)  # 1, M, d
        emb = emb.repeat(
            mask.shape[0], 1, 1
            )  # B, M , d
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


class DeepSpeechV2(nn.Module):
    """Implements the deep speech model
    proposed in https://arxiv.org/abs/1512.02595

    Args:
        n_conv (int): The number of convolution layers.
        kernel_size (int): The convolution layers' kernel size.
        stride (int): The convolution layers' stride.
        in_features (int): The input/speech feature size.
        hidden_size (int): The layers' hidden size.
        bidirectional (bidirectional): if the rnn is bidirectional
            or not.
        n_rnn (int): The number of RNN layers.
        n_linear_layers (int): The number of linear layers.
        n_classes (int): The number of classes.
        max_clip_value (int): The maximum relu value.
        rnn_type (str): rnn, gru or lstm.
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
            p_dropout: float
            ) -> None:
        super().__init__()
        self.conv = Conv1DLayers(
            in_size=in_features,
            out_size=hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            n_layers=n_conv,
            p_dropout=p_dropout
        )
        from .registry import RNN_REGISTRY
        self.rnns = nn.ModuleList(
            [
                RNN_REGISTRY[rnn_type](
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    bidirectional=bidirectional
                    )
                for _ in range(n_rnn)
                ]
            )
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=hidden_size,
                    out_features=hidden_size
                    )
                for _ in range(n_linear_layers)
                ]
            )
        self.crelu = CReLu(max_val=max_clip_value)
        self.context_conv = RowConv1D(
            tau=tau, hidden_size=hidden_size
        )
        self.pred_net = PredModule(
            in_features=hidden_size,
            n_classes=n_classes,
            activation=nn.LogSoftmax(dim=-1)
        )
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def forward(self, x: Tensor, mask: Tensor):
        lengths = mask.sum(dim=-1)
        lengths = lengths.cpu()
        out, lengths = self.conv(x, lengths)
        out = self.crelu(out)
        for layer in self.rnns:
            out, _, lengths = layer(
                out, lengths
                )
            if self.bidirectional is True:
                out = out[..., :self.hidden_size] +\
                    out[..., self.hidden_size:]
            out = self.crelu(out)
        out = self.context_conv(out)
        for layer in self.linear_layers:
            out = layer(out)
            out = self.crelu(out)
        preds = self.pred_net(out)
        preds = preds.permute(1, 0, 2)
        return preds, lengths
