"""
This module offers a variety of data augmentation techniques that operate on
either the time or frequency domain. All of the augmenters implemented in this
module have an abstract method named run, which is either an instance of
StochasticProcess or IProcess. This method applies the augmentation to the
input signal.

The classes implemented in this module can be divided into two groups based
on the domain they operate on.

The time domain augmenters include:

- WhiteNoiseInjector: Adds white noise to the input signal.
- VolumeChanger: Changes the volume of the input signal by applying a random gain.
- ConsistentAttenuator: Attenuates the amplitude of the input signal by a random single value.
- VariableAttenuator: Attenuates the amplitude of the input signal by applying a random gain, where the gain varies across time steps.
- Reverberation: Adds a reverberation effect to the input signal.

The frequency domain augmenters are:

- FrequencyMasking: Masks a random frequency pins in the input spectrogram.
- TimeMasking: Masks a random time segment in the input spectrogram.


.. code-block:: python

    # Import the module
    import torch
    from speeq.data import augmenters

    # creating dummy signal
    signal = torch.randn(1, 100)

    # Create an instance of the augmenter
    # Will use WhiteNoiseInjector example for illustration
    noise_injector = augmenters.WhiteNoiseInjector()

    # Apply the augmentation to the signal
    augmented_signal = noise_injector.run(signal)

"""
import random

import torch
from torch import Tensor

from .processors import StochasticProcess


class WhiteNoiseInjector(StochasticProcess):
    """Injects random Gaussian noise to the original signal,
    this is done by adding the inpus signal x to randomly generated
    Gaussian noise multiplied by a random gain as the below equation
    shows:

        .. math::

            x_{augmented} = x + noise \cdot gain \cdot gain\_mul

    where `gain` is a random number between 0 and 1, and x is a signal in the time domain.

    Args:
        ratio (float): The ratio/rate that the augmentation will be applied to
        the data. Default 1.0

        gain_mul (float): The gain multiplier factor to control the strength of
        the noise. Default 0.05
    """

    def __init__(self, ratio=1.0, gain_mul=5e-2) -> None:
        super().__init__(ratio)
        self.gain_mul = gain_mul

    def func(self, x: Tensor) -> Tensor:
        gain = random.random() * self.gain_mul
        return x + gain * torch.randn_like(x).to(x.device)


class VolumeChanger(StochasticProcess):
    """Amplifies the input signal by a random gain.

    This changes the amplitude of the input time domain signal `x` by
    multiplying it with a random gain `gain`, which is computed using the
    following equation:

    .. math::

        gain = (max\_gain - min\_gain) \cdot U + min\_gain

    where `U` is a random number between 0 and 1. The resulting amplified signal
    `x_augmented` can be computed as follows:

    .. math::

        x_{augmented} = x \cdot gain

    Args:
        ratio (float): The ratio/rate that the augmentation will be applied to
        the data. Default 1.0

        min_gain (float): The minimum gain that will be multiplied by the signal.

        max_gain (float): The maximum gain that will be multiplied by the signal.
    """

    def __init__(self, min_gain: float, max_gain: float, ratio=1.0) -> None:
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
    """applies amplitude attenuation to the input signal by multiplying it by a
    random gain that is less than 1, such that the gain is consistent across all time
    steps. The augmented signal x_augmented is given by the following equation:

        .. math::

            x_{augmented} = x \cdot U

    where `U` is a random number between `min_gain` and 1, and x is input time
    domain signal.

    Args:
        ratio (float): The ratio/rate that the augmentation will be applied to
        the data. Default 1.0

        min_gain (float): The minimum gain that will be multiplied by the
        signal. Default 0.1
    """

    def __init__(self, ratio=1.0, min_gain=0.1) -> None:
        super().__init__(ratio=ratio, min_gain=min_gain, max_gain=1)


class VariableAttenuator(StochasticProcess):
    """applies random attenuation to an input signal by multiplying it with a random
    gain less than 1. The amount of attenuation varies across time steps. The
    function uses the following equation to apply the attenuation

        .. math::

            x_{augmented} = x \cdot U \cdot noise\_mul

    where x is the input time-domain signal, U is a random Gaussian noise with
    values between 0 and 1 and the same shape as x.

    Args:
        ratio (float): The ratio/rate that the augmentation will be applied to
        the data. Default 1.0

        noise_mul (float): The noise multiplier. Default 0.5
    """

    def __init__(self, ratio=1.0, noise_mul=0.5) -> None:
        super().__init__(ratio)
        self.noise_mul = noise_mul

    def func(self, x: Tensor):
        return x + x * self.noise_mul * torch.randn_like(x).to(x.device)


class Reverberation(StochasticProcess):
    """Reverberates the input signal by generating an impulse response
    and convolve it with the speech signal.

    Args:
        ratio (float): The ratio/rate that the augmentation will be applied to
        the data. Default 1.0

        min_len (int): The minimum impulse response to generate. Default 1000.

        max_len (int): The maximum impulse response length. Default 4000.

        start_val (int): The starting value of the impulse response genration
        function. Default -10.

        end_val (int): The end value of the impulse response genration
        function. Default 10.

        eps (float): smoothing value, to prevent devision by 0. Default to 1e-3.
    """

    def __init__(
        self, ratio=1.0, min_len=1000, max_len=4000, start_val=-10, end_val=10, eps=1e-3
    ) -> None:
        super().__init__(ratio)
        self.min_len = min_len
        self.max_len = max_len
        self.start_val = start_val
        self.end_val = end_val
        self.eps = eps

    def _get_impulse_response(self) -> Tensor:
        length = random.randint(self.min_len, self.max_len)
        x = torch.linspace(self.start_val, self.end_val, length)
        alpha = self.eps + random.random()
        x /= alpha
        denominator = torch.exp(x) + torch.exp(-x)
        numerator = torch.exp(x) - torch.exp(-x)
        envelope = 1 - (numerator / denominator) ** 2
        envelope = envelope.nan_to_num()
        h = torch.randn_like(envelope) * envelope
        return h.view(1, 1, length)

    def func(self, x: Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(dim=0)
        ir = self._get_impulse_response()
        ir = ir.to(x.device)
        ir = ir.flip(dims=[-1])
        ir_length = ir.shape[-1]
        is_odd = int(ir_length % 2 != 0)
        x = torch.cat(
            [
                torch.zeros(1, 1, ir_length // 2).to(x.device),
                x,
                torch.zeros(1, 1, ir_length // 2 + is_odd).to(x.device),
            ],
            dim=-1,
        )
        return torch.nn.functional.conv1d(x, ir).squeeze(dim=0)


class _BaseMasking(StochasticProcess):
    def __init__(self, n: int, max_length: int, ratio=1.0) -> None:
        super().__init__(ratio)
        self.n = n
        self.max_length = max_length

    def _get_mask(self, x: Tensor, dim=-1):
        mask = torch.ones_like(x, device=x.device)
        length = x.shape[dim]
        for _ in range(self.n):
            start = random.randint(0, length)
            end = random.randint(start, start + self.max_length)
            end = min(length, end)
            indices = torch.arange(start, end, device=x.device)
            mask = mask.index_fill(dim=dim, index=indices, value=0)
        return mask


class FrequencyMasking(_BaseMasking):
    """Mask the inpus spectrogram, on the frequency axis.

    Args:
        n (int): The number of times to apply the masking operation.

        max_length (int): The maximum masking length.

        ratio (float): The ratio/rate that the augmentation will be applied to
        the data. Default 1.0
    """

    def __init__(self, n: int, max_length: int, ratio=1.0) -> None:
        super().__init__(ratio=ratio, n=n, max_length=max_length)

    def func(self, x: Tensor) -> Tensor:
        """
        x (Tensor): the input spectrogram to be augmented of
        shape [..., time, freq].
        """
        return x * self._get_mask(x, dim=-1)


class TimeMasking(_BaseMasking):
    """Mask the inpus spectrogram, on the time axis.

    Args:
        n (int): The number of times to apply the masking operation.

        max_length (int): The maximum masking length.

        ratio (float): The ratio/rate that the augmentation will be applied to
        the data. Default 1.0
    """

    def __init__(self, n: int, max_length: int, ratio=1.0) -> None:
        super().__init__(ratio=ratio, n=n, max_length=max_length)

    def func(self, x: Tensor) -> Tensor:
        """
        x (Tensor): the input spectrogram to be augmented of
        shape [..., time, freq].
        """
        return x * self._get_mask(x, dim=-2)
