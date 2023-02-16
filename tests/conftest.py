import string

import pytest
import torch
from pytest import fixture
from torch import LongTensor

from speeq.constants import FileKeys


@fixture
def char2id():
    chars = string.ascii_letters
    return dict(zip(chars, range(len(chars))))


@fixture
def text():
    return string.ascii_letters


@fixture
def dict_csv_data():
    return [
        {
            FileKeys.speech_key.value: "tests/files/1.wav",
            FileKeys.text_key.value: "be at prudence's to night at eight",
        },
        {
            FileKeys.speech_key.value: "tests/files/2.wav",
            FileKeys.text_key.value: "in the course of the day i received this note",
        },
    ]


masking_params_mark = pytest.mark.parametrize(
    ("seq_len", "pad_len"), ((1, 0), (1, 1), (5, 4), (3, 5))
)


@fixture
def positional_enc_1_5_4():
    x = [
        [0.0, 1.0, 0.0, 1.0],
        [0.8415, 0.5403, 0.0100, 0.9999],
        [0.9093, -0.4161, 0.0200, 0.9998],
        [0.1411, -0.9900, 0.0300, 0.9996],
        [-0.7568, -0.6536, 0.0400, 0.9992],
    ]
    x = torch.tensor(x)
    x = x.unsqueeze(dim=0)
    return x


@fixture
def batched_speech_feat():
    def func(batch_size, seq_len, feat_size):
        return torch.ones(batch_size, seq_len, feat_size)

    return func


mask_from_lens_mark = pytest.mark.parametrize(
    ("lengths", "max_len"),
    (
        (LongTensor([1, 2, 3]), 3),
        (LongTensor([1]), 1),
        (LongTensor([1, 2]), 4),
    ),
)


@fixture
def batcher():
    def func(batch_size, seq_len, feat_size, *args, **kwargs):
        return torch.randn(batch_size, seq_len, feat_size)

    return func


@fixture
def int_batcher():
    def func(batch_size, seq_len, max_val, min_val=0):
        return torch.randint(min_val, max_val, size=(batch_size, seq_len))

    return func


@fixture
def audio_generator():
    def func(n_samples: int, n_channels=1):
        return torch.randn(n_channels, n_samples)

    return func


@fixture
def spectrogram_generator():
    def func(n_samples: int, feat_size: int, n_channels=1):
        return torch.randn(n_channels, n_samples, feat_size)

    return func


@fixture
def char_tokenizer_dict():
    return {
        "type": "char_tokenizer",
        "token_to_id": {
            "<OOV>": 0,
            "<PAD>": 1,
            "<SOS>": 2,
            "a": 4,
            "b": 5,
            "c": 6,
        },
        "special_tokens": {
            "oov": ["<OOV>", 0],
            "pad": ["<PAD>", 1],
            "sos": ["<SOS>", 2],
        },
    }


@fixture
def word_tokenizer_dict():
    return {
        "type": "word_tokenizer",
        "token_to_id": {
            "<OOV>": 0,
            "<PAD>": 1,
            "<SOS>": 2,
            "a": 4,
            "b": 5,
            "c": 6,
        },
        "special_tokens": {
            "oov": ["<OOV>", 0],
            "pad": ["<PAD>", 1],
            "sos": ["<SOS>", 2],
        },
    }
