from processors import StochasticProcess
from torch import Tensor
import random
import torch


class WhiteNoiseInjector(StochasticProcess):
    """Injects random Gaussian noise to the original signal,
    this is done by adding the inpus signal x to randomly generated
    Gaussian noise multiplied by a random gain as the below equation
    show:
    x_aumgneted = x + noise * gain

    Args:
        ratio (float): The ratio/rate that the augmentation will be
        applied to the data. Default 1.0
    """
    def __init__(self, ratio=1.0) -> None:
        super().__init__(ratio)

    def func(self, x: Tensor) -> Tensor:
        gain = random.random()
        return x + gain * torch.randn_like(x).to(x.device)
