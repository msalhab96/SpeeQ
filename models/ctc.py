import torch
from typing import Tuple
from .layers import CReLu, PredModule
from .registry import RNN_REGISTRY
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
            n_clases: int,
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
            n_classes=n_clases,
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
        out, _, lengths = self.rnn(x, lengths)
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
