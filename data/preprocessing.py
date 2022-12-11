import functools
import torchaudio
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
