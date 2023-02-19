import os

import pytest

from speeq.data import loaders
from tests.helpers import create_csv_file


class TestCSVDataset:
    def test(self, dict_csv_data, tmp_path):
        file_path = os.path.join(tmp_path, "file.csv")
        sep = ","
        create_csv_file(file_path, data=dict_csv_data, sep=sep)
        csv_data = loaders.CSVDataset(data_path=file_path, sep=sep)
        assert len(dict_csv_data) == len(csv_data)
        for result, target in zip(csv_data, dict_csv_data):
            assert result == target


class TestSpeechTextDataset:
    def test_wav(self, speech_text_dataset):
        dataset = speech_text_dataset()
        assert isinstance(dataset[0], tuple)
        speech, speech_len, text, text_len = dataset[0]
        assert speech.shape[1] == speech_len
        assert text.shape[-1] == text_len
        with pytest.raises(IndexError):
            dataset[len(dataset)]

    def test_spec(self, speech_text_dataset):
        dataset = speech_text_dataset(use_mel_spec=True)
        assert isinstance(dataset[0], tuple)
        speech, speech_len, text, text_len = dataset[0]
        assert speech.shape[1] == speech_len
        assert text.shape[-1] == text_len
        with pytest.raises(IndexError):
            dataset[len(dataset)]


class TestSpeechTextLoader:
    @pytest.mark.parametrize(
        ("batch_size", "rank", "world_size", "n_runs"),
        (
            (1, 0, 1, 2),
            (1, 0, 2, 1),
            (2, 0, 1, 1),
        ),
    )
    def test(self, speech_text_loader, batch_size, rank, world_size, n_runs):
        loader = speech_text_loader(
            batch_size=batch_size, rank=rank, world_size=world_size
        )
        assert len(loader) == n_runs
