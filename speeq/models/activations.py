import torch
from torch import Tensor, nn


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


class CReLu(nn.Module):
    """clipped rectified-linear unit, can be described as
    min{max{0, x}, max_value}

    as described in: https://arxiv.org/abs/1412.5567

    Args:
        max_val (int): the maximum clipping value.
    """

    def __init__(self, max_val: int) -> None:
        super().__init__()
        self.max_val = max_val

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(
            x, min=0, max=self.max_val
        )
