import torch
from torch import Tensor
from typing import Tuple, Union
from pathlib import Path
from .utils import load_csv
from .processors import IProcessor
from .interfaces import ITokenizer
from constants import FileKeys


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


class SpeechTextDataset(CSVDataset):
    """Implements the basic dataset for speech-text pairs
    for speech-recognition application.

    Args:
        data_path (Union[str, Path]): The file path.
        tokenizer (ITokenizerITokenizer): The tokenizer that will be used.
        speech_processor (IProcessor): The speech processor.
        text_processor (IProcessor): The text processor.
        sep (str): The CSV separator.
        encoding (str): The file encoding. Default "utf-8".
    """
    def __init__(
            self,
            data_path: Union[str, Path],
            tokenizer: ITokenizer,
            speech_processor: IProcessor,
            text_processor: IProcessor,
            sep: str,
            encoding='utf-8'
            ) -> None:
        super().__init__(
            data_path, sep, encoding
            )
        self.tokenizer = tokenizer
        self.speech_processor = speech_processor
        self.text_processor = text_processor

    def process_text(self, text: str) -> Tuple[Tensor, int]:
        text = self.text_processor.execute(text)
        # FIX: eos and sos for CTC-based model
        tokens = self.tokenizer.tokenize(
            text, add_sos=True, add_eos=True
            )
        return torch.LongTensor(tokens), len(tokens)

    def process_speech(
            self, file_path: Union[Path, str]
            ) -> Tensor:
        speech = self.speech_processor.execute(file_path)
        # FIX: handle MONO and STEREO issue
        return speech, speech.shape[-2]

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        text, text_len = self.process_text(item[FileKeys.text_key.value])
        speech, speech_len = self.process_speech(
            item[FileKeys.speech_key.value]
            )
        return speech, speech_len, text, text_len
