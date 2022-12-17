import torch
import functools
import torchaudio
from torch import nn
from torch import Tensor
from typing import Union
from pathlib import Path
from .processors import IProcess
from torchaudio import transforms
SAMPLER_CACHE_SIZE = 5


class AudioLoader(IProcess):
    def __init__(
            self,
            sample_rate: int
            ) -> None:
        super().__init__()
        self.sample_rate = sample_rate

    @functools.lru_cache(SAMPLER_CACHE_SIZE)
    def _get_resampler(self, original_sr: int):
        return transforms.Resample(
            orig_freq=original_sr,
            new_freq=self.sample_rate
        )

    def run(self, file_path: Union[Path, str]):
        x, sr = torchaudio.load(file_path)
        return self._get_resampler(sr)(x)


class FeatExtractor(IProcess):
    __feat_extractor = {
        'mfcc': transforms.MFCC,
        'melspec': transforms.MelSpectrogram
    }

    def __init__(
            self,
            feat_ext_name: str,
            feat_ext_args: dict,
            ) -> None:
        super().__init__()
        self.feat_extractor = self.__feat_extractor[feat_ext_name](
            **feat_ext_args
            )

    def run(self, x: Tensor):
        x = self.feat_extractor(x)
        x = x.swapaxes(-1, -2)  # (..., T, F)
        return x


class FeatStacker(IProcess):
    """Implements the feature stacking operation.

    Args:
        feat_stack_factor (int): The feature stacking
        ratio.
    """
    def __init__(self, feat_stack_factor: int) -> None:
        super().__init__()
        assert feat_stack_factor > 1
        self.feat_stack_factor = feat_stack_factor

    def run(self, x: Tensor):
        # x of shape (..., T, F)
        if self.feat_stack_factor == 1:
            return x
        residual = x.shape[-2] % self.feat_stack_factor
        if residual != 0:
            size = list(x.shape)
            size[1] = self.feat_stack_factor - residual
            zeros = torch.zeros(*size).to(x.device)
            x = torch.cat([x, zeros])
        x = x.view(
            *x.shape[:-2],
            x.shape[-2] // self.feat_stack_factor,
            -1
            )
        return x


class FrameContextualizer(IProcess):
    """Implements frame contextualizer through time
    as described in https://arxiv.org/abs/1412.5567

    Args:
        contex_size (int): The context size, or the number
        of left and right frames to take with the current frame.
    """
    def __init__(self, contex_size: int) -> None:
        super().__init__()
        self.contex_size = contex_size
        self.win_size = self.contex_size * 2 + 1
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.win_size,
            kernel_size=self.win_size,
            bias=False
        )
        self.conv.weight.data = torch.eye(self.win_size).view(
            self.win_size, 1, self.win_size
            )
        self.conv.weight.requires_grad = False

    def run(self, x: Tensor) -> Tensor:
        # x of shape [1, T, F]
        x = x.permute(2, 0, 1)  # [F, 1, T]
        zeros = torch.zeros(
            x.shape[0], 1, self.contex_size
            )
        x = torch.cat([zeros, x, zeros], dim=-1)
        x = self.conv(x)  # [F, W, T]
        x = x.permute(2, 1, 0).contiguous()  # [T, W, F]
        x = x.view(1, x.shape[0], -1)  # [1, T, W * F]
        return x
