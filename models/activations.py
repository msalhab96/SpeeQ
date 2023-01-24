import torch
from torch import nn
from torch import Tensor


class Sigmax(nn.Module):
    """Implements the custom activation function for attention
    proposed in https://arxiv.org/abs/1506.07503
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        sigma = torch.sigmoid(x)
        sum = sigma.sum(dim=self.dim, keepdim=True)
        return sigma / sum
