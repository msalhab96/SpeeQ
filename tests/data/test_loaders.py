import os
from unittest import mock

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
