import csv
import os
from unittest import mock

import pytest

from speeq.constants import FileKeys
from speeq.data import loaders
from speeq.data.preprocessing import AudioLoader
from speeq.data.processors import OrderedProcessor
from speeq.data.tokenizers import CharTokenizer
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
    def test(self, dict_csv_data, tmp_path):
        encoding = "utf-8"
        sep = ","
        data = [item[FileKeys.text_key.value] for item in dict_csv_data]
        file_path = os.path.join(tmp_path, "file.csv")
        with open(file_path, "w", encoding=encoding) as f:
            writer = csv.DictWriter(f, dict_csv_data[0].keys(), delimiter=sep)
            writer.writeheader()
            writer.writerows(dict_csv_data)
        speech_processor = OrderedProcessor([AudioLoader(16000)])
        text_processor = OrderedProcessor([])
        char_tokenizer = CharTokenizer()
        char_tokenizer.set_tokenizer(data)
        dataset = loaders.SpeechTextDataset(
            data_path=file_path,
            tokenizer=char_tokenizer,
            speech_processor=speech_processor,
            text_processor=text_processor,
            sep=sep,
            encoding=encoding,
        )
        assert len(dataset) == len(dict_csv_data)
        assert isinstance(dataset[0], tuple)
        speech, speech_len, text, text_len = dataset[0]
        assert speech.shape[1] == speech_len
        assert text.shape[-1] == text_len
        with pytest.raises(IndexError):
            dataset[len(dataset)]
