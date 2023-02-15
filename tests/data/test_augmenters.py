import pytest
import torch

from speeq.data import augmenters


class BaseAudioTest:
    def check(self, audio_generator, n_samples: int, aug_args: dict):
        input = audio_generator(n_samples)
        augmenter = self.augmenter(**aug_args)
        result = augmenter.func(input)
        assert result.shape == input.shape
        if aug_args.get("ratio", -1) == 0:
            result = augmenter.run(input)
            assert torch.allclose(input, result)


class TestWhiteNoiseInjector(BaseAudioTest):
    augmenter = augmenters.WhiteNoiseInjector

    @pytest.mark.parametrize(
        ("n_samples", "aug_args"),
        (
            (1, {"ratio": 1}),
            (2, {"ratio": 1}),
            (1, {"ratio": 0}),
            (2, {"ratio": 0}),
        ),
    )
    def test(self, audio_generator, n_samples, aug_args):
        self.check(audio_generator, n_samples=n_samples, aug_args=aug_args)


class TestVolumeChanger(BaseAudioTest):
    augmenter = augmenters.VolumeChanger

    @pytest.mark.parametrize(
        ("n_samples", "aug_args"),
        (
            (1, {"min_gain": 0.1, "max_gain": 1, "ratio": 1}),
            (2, {"min_gain": 0.5, "max_gain": 1, "ratio": 1}),
            (1, {"min_gain": 0.1, "max_gain": 1, "ratio": 0}),
            (2, {"min_gain": 0.5, "max_gain": 1, "ratio": 0}),
        ),
    )
    def test(self, audio_generator, n_samples, aug_args):
        self.check(audio_generator, n_samples=n_samples, aug_args=aug_args)


class TestConsistentAttenuator(BaseAudioTest):
    augmenter = augmenters.ConsistentAttenuator

    @pytest.mark.parametrize(
        ("n_samples", "aug_args"),
        (
            (1, {"min_gain": 0.1, "ratio": 1}),
            (2, {"min_gain": 0.5, "ratio": 1}),
            (1, {"min_gain": 0.1, "ratio": 0}),
            (2, {"min_gain": 0.5, "ratio": 0}),
        ),
    )
    def test(self, audio_generator, n_samples, aug_args):
        self.check(audio_generator, n_samples=n_samples, aug_args=aug_args)


class TestVariableAttenuator(BaseAudioTest):
    augmenter = augmenters.VariableAttenuator

    @pytest.mark.parametrize(
        ("n_samples", "aug_args"),
        (
            (1, {"ratio": 1}),
            (2, {"ratio": 1}),
            (1, {"ratio": 0}),
            (2, {"ratio": 0}),
        ),
    )
    def test(self, audio_generator, n_samples, aug_args):
        self.check(audio_generator, n_samples=n_samples, aug_args=aug_args)


class TestFrequencyMasking:
    @pytest.mark.parametrize(
        ("n_samples", "feat_size", "n", "max_length", "ratio"),
        (
            (2, 8, 1, 2, 1),
            (2, 8, 0, 2, 1),
            (2, 8, 1, 2, 0),
        ),
    )
    def test(self, spectrogram_generator, n_samples, feat_size, n, max_length, ratio):
        input = spectrogram_generator(n_samples, feat_size)
        augmenter = augmenters.FrequencyMasking(n=n, max_length=max_length, ratio=ratio)
        result = augmenter.func(input)
        assert result.shape == input.shape
        assert torch.allclose(result * input, result**2)
        if ratio == 0 or n == 0:
            result = augmenter.run(input)
            assert torch.allclose(input, result)


class TestTimeMasking:
    @pytest.mark.parametrize(
        ("n_samples", "feat_size", "n", "max_length", "ratio"),
        (
            (2, 8, 1, 2, 1),
            (2, 8, 0, 2, 1),
            (2, 8, 1, 2, 0),
        ),
    )
    def test(self, spectrogram_generator, n_samples, feat_size, n, max_length, ratio):
        input = spectrogram_generator(n_samples, feat_size)
        augmenter = augmenters.TimeMasking(n=n, max_length=max_length, ratio=ratio)
        result = augmenter.func(input)
        assert result.shape == input.shape
        assert torch.allclose(result * input, result**2)
        if ratio == 0 or n == 0:
            result = augmenter.run(input)
            assert torch.allclose(input, result)
