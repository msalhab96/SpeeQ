import pytest
import torch

from speeq.data import padders


class TestDynamicPadder:
    @pytest.mark.parametrize(
        ("dim", "pad_val", "left_pad", "max_len", "seq_shape"),
        (
            (0, 0, False, 5, (1, 2, 3)),
            (1, 0, False, 5, (1, 2, 3)),
            (2, 0, False, 5, (1, 2, 3)),
            (0, 0, True, 5, (1, 2, 3)),
        ),
    )
    def test_pad(self, dim, pad_val, left_pad, max_len, seq_shape):
        x = torch.randn(*seq_shape)
        padder = padders.DynamicPadder(dim=dim, pad_val=pad_val, left_pad=left_pad)
        expected_pad_len = max_len - seq_shape[dim]
        expected_shape = (*seq_shape[:dim], max_len, *seq_shape[1 + dim :])
        x, pad_len = padder.pad(x, max_len=max_len)
        assert x.shape == expected_shape
        assert pad_len == expected_pad_len


class TestStaticPadder:
    @pytest.mark.parametrize(
        ("dim", "pad_val", "left_pad", "max_len", "seq_shape"),
        (
            (0, 0, False, 5, (1, 2, 3)),
            (1, 0, False, 5, (1, 2, 3)),
            (2, 0, False, 5, (1, 2, 3)),
            (0, 0, True, 5, (1, 2, 3)),
        ),
    )
    def test_pad(self, dim, pad_val, left_pad, max_len, seq_shape):
        x = torch.randn(*seq_shape)
        padder = padders.StaticPadder(
            dim=dim, pad_val=pad_val, left_pad=left_pad, max_len=max_len
        )
        expected_pad_len = max_len - seq_shape[dim]
        expected_shape = (*seq_shape[:dim], max_len, *seq_shape[1 + dim :])
        x, pad_len = padder.pad(x)
        assert x.shape == expected_shape
        assert pad_len == expected_pad_len
