from typing import Union
from pathlib import Path
from .utils import load_csv


class CSVDataset:
    """The base dataset class that handles
    CSV datasets.

    Args:
        data_path (Union[str, Path]): The file path.
        sep (str): The CSV separator.
        encoding (str): The file encoding. Default "utf-8".
    """
    def __init__(
            self,
            data_path: Union[str, Path],
            sep: str,
            encoding='utf-8'
            ) -> None:
        super().__init__()
        self.data_path = data_path
        self.sep = sep
        self.data = load_csv(
            file_path=data_path,
            encoding=encoding,
            sep=sep
        )

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)
