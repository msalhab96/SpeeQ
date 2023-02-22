"""
This module contains custom activation functions that can be used in PyTorch models.

Available functions:

- Sigmax: Implements the custom activation function for attention.
- CReLu: Implements the Clipped Rectified Linear Unit (CReLu) activation function.
"""
import torch
from torch import Tensor, nn


class Sigmax(nn.Module):
    """Implements the custom activation function for attention
    proposed in https://arxiv.org/abs/1506.07503

    Args:

        dim (int): The dimension to apply the activation function on.

    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input tensor `x` through the Sigmax activation function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The result tensor after applying the activation function.
        """
        sigma = torch.sigmoid(x)
        sum = sigma.sum(dim=self.dim, keepdim=True)
        return sigma / sum


class CReLu(nn.Module):
    """implements the Clipped Rectified Linear Unit (CReLu) activation function
    as described in: https://arxiv.org/abs/1412.5567

    Args:

        max_val (int): The maximum value for clipping

    """

    def __init__(self, max_val: int) -> None:
        super().__init__()
        self.max_val = max_val

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input tensor `x` through the Clipped Rectified Linear Unit
        (CReLu) activation function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The result tensor after applying the activation function.
        """
        return torch.clamp(x, min=0, max=self.max_val)
