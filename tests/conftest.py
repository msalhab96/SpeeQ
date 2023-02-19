import csv
import os
import string

import pytest
import torch
from pytest import fixture
from torch import LongTensor

from speeq.constants import FileKeys
from speeq.data.loaders import SpeechTextDataset, SpeechTextLoader
from speeq.data.padders import DynamicPadder
from speeq.data.processors import OrderedProcessor
from speeq.data.tokenizers import CharTokenizer


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


@fixture
def speech_text_dataset(dict_csv_data, tmp_path):
    def func(use_mel_spec=False):
        # mocking class
        class SpeechProcessor:
            def execute(self, *args, **kwargs):
                if use_mel_spec is False:
                    return torch.randn(1, 100)
                return torch.randn(1, 50, 60)

        encoding = "utf-8"
        sep = ","
        data = [item[FileKeys.text_key.value] for item in dict_csv_data]
        file_path = os.path.join(tmp_path, "file.csv")
        with open(file_path, "w", encoding=encoding) as f:
            writer = csv.DictWriter(f, dict_csv_data[0].keys(), delimiter=sep)
            writer.writeheader()
            writer.writerows(dict_csv_data)
        text_processor = OrderedProcessor([])
        char_tokenizer = CharTokenizer()
        char_tokenizer.set_tokenizer(data)
        return SpeechTextDataset(
            data_path=file_path,
            tokenizer=char_tokenizer,
            speech_processor=SpeechProcessor(),
            text_processor=text_processor,
            sep=sep,
            encoding=encoding,
        )

    return func


@fixture
def speech_text_loader(speech_text_dataset):
    def func(batch_size, rank=0, world_size=1, use_mel_spec=False):
        dataset = speech_text_dataset(use_mel_spec=use_mel_spec)
        speech_padder = DynamicPadder(dim=-2 if use_mel_spec else -1, pad_val=0)
        text_padder = DynamicPadder(dim=-1, pad_val=0)
        print("*" * 10)
        print(len(dataset), batch_size)
        return SpeechTextLoader(
            dataset=dataset,
            batch_size=batch_size,
            text_padder=text_padder,
            speech_padder=speech_padder,
            rank=rank,
            world_size=world_size,
        )

    return func
