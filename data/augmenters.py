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


class VolumeChanger(StochasticProcess):
    """Changes the amplitude of the input signal by
    a random gain.

    Args:
        ratio (float): The ratio/rate that the augmentation will be
        applied to the data. Default 1.0
        min_gain (float): The minimum gain that will be multiplied by
        the signal.
        max_gain (float): The maximum gain that will be multiplied by
        the signal.
    """
    def __init__(
            self,
            min_gain: float,
            max_gain: float,
            ratio=1.0
            ) -> None:
        super().__init__(ratio)
        self.min_gain = min_gain
        self.max_gain = max_gain
        self._diff = self.max_gain - self.min_gain

    @property
    def _gain(self):
        return self._diff * random.random() + self.min_gain

    def func(self, x: Tensor) -> Tensor:
        return self._gain * x


class ConsistentAttenuator(VolumeChanger):
    """Attenuates the amplitude of the input signal by
    a random gain less than 1, such that the gain is consistant
    across all time steps.

    Args:
        ratio (float): The ratio/rate that the augmentation will be
        applied to the data. Default 1.0
        min_gain (float): The minimum gain that will be multiplied by
        the signal. Default 0.1
    """
    def __init__(self, ratio=1.0, min_gain=0.1) -> None:
        super().__init__(ratio, min_gain, max_gain=1)
