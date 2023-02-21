"""
This module contains classes for speech processing that implement the IProcess interface.

Classes:

- AudioLoader: Loads and resamples an audio file to the targeted sample rate.
- FeatExtractor: Extracts frequency features from a given time domain signal, supporting mfcc and mel spectrogram.
- FeatStacker: Implements feature stacking operation by stacking consecutive time stamps along the feature space.
- FrameContextualizer: Implements frame contextualizer through time as described in https://arxiv.org/abs/1412.5567

All classes have a run method as an abstract method that applies the process on the input signal.

Example usage:


    .. code-block:: python

        # Import required packages and modules
        import torch
        from speeq.data.processes import AudioLoader, FeatExtractor, FeatStacker, FrameContextualizer

        # Define the audio file path
        audio_path = 'path/to/audio.wav'

        # Create an instance of AudioLoader
        audio_loader = AudioLoader(sample_rate=16000)

        # Load the audio file using AudioLoader
        audio_tensor = audio_loader.run(audio_path)

        # Create an instance of FeatExtractor
        feat_extractor = FeatExtractor(feat_ext_name='mfcc', feat_ext_args={'n_mfcc': 13})

        # Extract the MFCC features of the audio tensor using FeatExtractor
        feat_tensor = feat_extractor.run(audio_tensor)

        # Create an instance of FeatStacker
        feat_stacker = FeatStacker(feat_stack_factor=2)

        # Stack the features using FeatStacker
        stacked_feat_tensor = feat_stacker.run(feat_tensor)

        # Create an instance of FrameContextualizer
        frame_contextualizer = FrameContextualizer(contex_size=2)

        # Add context to the features using FrameContextualizer
        contextualized_feat_tensor = frame_contextualizer.run(stacked_feat_tensor)
"""

import functools
from pathlib import Path
from typing import Union

import torch
import torchaudio
from torch import Tensor, nn
from torchaudio import transforms

from speeq.interfaces import IProcess

SAMPLER_CACHE_SIZE = 5


class AudioLoader(IProcess):
    """Loads and resamples audio to the specified sample rate.

    .. note::

        This class utilizes the `load` function provided by `torchaudio` framework
        for loading audio. For additional details on supported file formats and
        further information, please refer to
        `the documentation <https://pytorch.org/audio/stable/index.html>`_.

    Args:
        sample_rate (int): The target sampling rate.
    """

    def __init__(self, sample_rate: int) -> None:
        super().__init__()
        self.sample_rate = sample_rate

    @functools.lru_cache(SAMPLER_CACHE_SIZE)
    def _get_resampler(self, original_sr: int):
        return transforms.Resample(orig_freq=original_sr, new_freq=self.sample_rate)

    def run(self, file_path: Union[Path, str]) -> Tensor:
        """Load and resample an audio file.

        Args:
            file_path (Union[Path, str]): The path to the audio file to be loaded.

        Returns:
            Tensor: A tensor containing the speech data of shape [C, M].
        """
        x, sr = torchaudio.load(file_path)
        return self._get_resampler(sr)(x)


class FeatExtractor(IProcess):
    """A class for extracting frequency features from a given time domain signal,
    supporting `mfcc` and `mel spectrogram` features.


    .. note::

        This class utilizes the `transforms.MelSpectrogram` and `transforms.MFCC`
        classes provided by `torchaudio` framework for feature extraction.
        For additional details and parameter information, please refer to
        `the documentation <https://pytorch.org/audio/stable/index.html>`_.

    Args:
        feat_ext_name (str): The name of the feature extractor to be used. either `mfcc` or `melspec`.

        feat_ext_args (dict): The arguments to be passed to the specified feature
        extractor. For more information on parameters, please refer to the `torchaudio` documentation.
    """

    __feat_extractor = {"mfcc": transforms.MFCC, "melspec": transforms.MelSpectrogram}

    def __init__(
        self,
        feat_ext_name: str,
        feat_ext_args: dict,
    ) -> None:
        super().__init__()
        self.feat_extractor = self.__feat_extractor[feat_ext_name](**feat_ext_args)

    def run(self, x: Tensor) -> Tensor:
        """Transforms the input signal `x` from time domain to frequency domain using the
        predefined feature extractor.


        Args:
            x (Tensor): A time domain tensor of shape [..., T, F].

        Returns:
            Tensor: A tensor containing the frequency domain features of shape [..., T, F].
        """
        x = self.feat_extractor(x)
        x = x.swapaxes(-1, -2)  # (..., T, F)
        return x


class FeatStacker(IProcess):
    """A class that implements feature stacking by stacking `n` consecutive time stamps
    along the feature space.


    Args:
        feat_stack_factor (int): The factor by which to stack the features.


        Example:

        .. code-block:: python

            # Import required packages
            import torch
            from speeq.data.processes import FeatStacker

            batch_size = 3
            max_len = 10
            feat_size = 15
            stacking_factor = 2
            # creating dummy data
            input = torch.randn(batch_size, max_len, feat_size)

            # Create an instance of the class
            stacker = FeatStacker(feat_stack_factor=stacking_factor)

            # Apply the process to the input
            result = stacker.run(input)

            # Print the result's shape
            print(result.shape)  # torch.Size([3, 5, 30])

    """

    def __init__(self, feat_stack_factor: int) -> None:
        super().__init__()
        assert feat_stack_factor > 1
        self.feat_stack_factor = feat_stack_factor

    def run(self, x: Tensor):
        """Applies feature stacking to the input tensor x by stacking `n` consecutive
        time frames along the feature space.

        Args:
            x (Tensor): The input tensor of shape [..., T, F]

        Returns:
            Tensor: The result tensor after applying feature stacking. The shape of the result tensor
            is [batch_size, seq_len // n, feat_dim * n].
        """
        if self.feat_stack_factor == 1:
            return x
        residual = x.shape[-2] % self.feat_stack_factor
        if residual != 0:
            size = list(x.shape)
            size[-2] = self.feat_stack_factor - residual
            zeros = torch.zeros(*size).to(x.device)
            x = torch.cat([x, zeros], dim=-2)
        x = x.view(*x.shape[:-2], x.shape[-2] // self.feat_stack_factor, -1)
        return x


class FrameContextualizer(IProcess):
    """Implements frame contextualization through time, as described in
    https://arxiv.org/abs/1412.5567

    Args:
        contex_size (int): The context size, i.e., the number of left or right
        frames to consider with the current frame.


        Example:

        .. code-block:: python

            # Import required packages
            import torch
            from speeq.data.processes import FrameContextualizer

            max_len = 10
            feat_size = 15

            # 2 to the left, the current time step and 2 to the right
            contex_size = 2

            # creating dummy data
            input = torch.randn(1, max_len, feat_size)

            # Create an instance of the class
            contextualizer = FrameContextualizer(contex_size=contex_size)

            # Apply the process to the input
            result = contextualizer.run(input)

            # Print the result's shape
            print(result.shape)  # torch.Size([1, 10, 75])

    """

    def __init__(self, contex_size: int) -> None:
        super().__init__()
        self.contex_size = contex_size
        self.win_size = self.contex_size * 2 + 1
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.win_size,
            kernel_size=self.win_size,
            bias=False,
        )
        self.conv.weight.data = torch.eye(self.win_size).view(
            self.win_size, 1, self.win_size
        )
        self.conv.weight.requires_grad = False

    def run(self, x: Tensor) -> Tensor:
        """Applies frame contextualization on the input tensor x.

        Args:
            x (Tensor): The input tensor of shape [1, M, F]

        Returns:
            Tensor: The output tensor of shape [1, M, F * (2 * context_size + 1)]
        """
        x = x.permute(2, 0, 1)  # [F, 1, T]
        zeros = torch.zeros(x.shape[0], 1, self.contex_size)
        x = torch.cat([zeros, x, zeros], dim=-1)
        x = self.conv(x)  # [F, W, T]
        x = x.permute(2, 1, 0).contiguous()  # [T, W, F]
        x = x.view(1, x.shape[0], -1)  # [1, T, W * F]
        return x
