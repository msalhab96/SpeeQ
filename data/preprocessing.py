import torch
import functools
import torchaudio
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
