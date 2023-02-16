import pytest
import torch

from speeq.data import preprocessing


class TestAudioLoader:
    @pytest.mark.parametrize(
        ("sr", "file_path"),
        (
            (16000, "tests/files/1.wav"),
            (16000, "tests/files/2.wav"),
        ),
    )
    def test_run(self, sr, file_path):
        loader = preprocessing.AudioLoader(sample_rate=sr)
        result = loader.run(file_path=file_path)
        assert len(result.shape) == 2


class TestFeatExtractor:
    @pytest.mark.parametrize(
        ("seq_len", "feat_extractor", "args", "expected_shape"),
        (
            (
                16000,
                "melspec",
                {
                    "sample_rate": 16000,
                    "n_fft": 400,
                    "win_length": 400,
                    "hop_length": 200,
                    "n_mels": 80,
                },
                (1, 81, 80),
            ),
            (
                16000,
                "mfcc",
                {
                    "sample_rate": 16000,
                    "n_mfcc": 40,
                    "melkwargs": {
                        "n_fft": 400,
                        "win_length": 400,
                        "hop_length": 200,
                        "n_mels": 80,
                    },
                },
                (1, 81, 40),
            ),
        ),
    )
    def test_run(self, seq_len, feat_extractor, args, expected_shape):
        input = torch.randn(1, seq_len)
        extractor = preprocessing.FeatExtractor(
            feat_ext_name=feat_extractor, feat_ext_args=args
        )
        result = extractor.run(input)
        assert result.shape == expected_shape


class TestFeatStacker:
    @pytest.mark.parametrize(
        ("inp_shape", "stacking_factor", "expected_shape"),
        (
            ((6, 20), 2, (3, 40)),
            ((6, 20), 3, (2, 60)),
            ((6, 20), 4, (2, 80)),
            ((6, 20), 5, (2, 100)),
            ((6, 20), 6, (1, 120)),
            ((1, 6, 20), 3, (1, 2, 60)),
            ((1, 6, 20), 4, (1, 2, 80)),
            ((3, 6, 20), 3, (3, 2, 60)),
            ((3, 6, 20), 4, (3, 2, 80)),
        ),
    )
    def test_run(self, inp_shape, stacking_factor, expected_shape):
        input = torch.randn(*inp_shape)
        feat_stacker = preprocessing.FeatStacker(feat_stack_factor=stacking_factor)
        result = feat_stacker.run(input)
        assert result.shape == expected_shape


class TestFrameContextualizer:
    @pytest.mark.parametrize(
        ("context_size", "inp_shape", "expected_shape"),
        (
            (1, (1, 10, 8), (1, 10, 24)),
            (2, (1, 10, 8), (1, 10, 40)),
            (3, (1, 10, 8), (1, 10, 56)),
            (10, (1, 10, 8), (1, 10, 168)),
        ),
    )
    def test_run(self, context_size, inp_shape, expected_shape):
        processor = preprocessing.FrameContextualizer(contex_size=context_size)
        x = torch.randn(*inp_shape)
        result = processor.run(x)
        assert result.shape == expected_shape
