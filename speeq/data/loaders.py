import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from speeq.constants import FileKeys
from speeq.interfaces import IDataLoader, IDataset, IPadder, ITokenizer
from speeq.utils.utils import get_pad_mask, load_csv

from .processors import IProcessor


class CSVDataset(IDataset):
    """The base dataset class that handles
    CSV datasets.

    Args:
        data_path (Union[str, Path]): The file path.
        sep (str): The CSV separator.
        encoding (str): The file encoding. Default "utf-8".
        sort_key (Optional[str]): The key to sort the data on. Default ''.
        reverse (bool): Used if sorting key is passed, False will sort ascending,
            True will sort descending. Default is False.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        sep: str,
        encoding="utf-8",
        sort_key: Optional[str] = "",
        reverse: bool = False,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.sep = sep
        self.data = load_csv(file_path=data_path, encoding=encoding, sep=sep)
        if sort_key != "":
            self.data = list(
                sorted(self.data, key=lambda x: x[sort_key], reverse=reverse)
            )

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)


class SpeechTextDataset(CSVDataset):
    """Implements the basic dataset for speech-text pairs
    for speech-recognition application.

    Args:
        data_path (Union[str, Path]): The data csv file path.
        tokenizer (ITokenizerITokenizer): The tokenizer that will be used.
        speech_processor (IProcessor): The speech processor, where the `run`
            method returns the speech of shape [B] or [1, M], or [..., M, F].
        text_processor (IProcessor): The text processor.
        sep (str): The CSV separator.
        add_sos (bool): a flag indicates if SOS token shall be added
            to the text sequence. Default False.
        add_eos (bool): a flag indicates if EOS token shall be added
            to the text sequence. Default False.
        encoding (str): The file encoding. Default "utf-8".
        sort_key (Optional[str]): The key to sort the data on. Default ''.
        reverse (bool): Used if sorting key is passed, False will sort ascending,
            True will sort descending. Default is False.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: ITokenizer,
        speech_processor: IProcessor,
        text_processor: IProcessor,
        sep: str,
        add_sos=False,
        add_eos=False,
        encoding="utf-8",
        sort_key: Optional[str] = "",
        reverse: bool = False,
    ) -> None:
        super().__init__(
            data_path=data_path,
            sep=sep,
            encoding=encoding,
            sort_key=sort_key,
            reverse=reverse,
        )
        self.tokenizer = tokenizer
        self.speech_processor = speech_processor
        self.text_processor = text_processor
        self.add_sos = add_sos
        self.add_eos = add_eos

    def process_text(self, text: str) -> Tuple[Tensor, int]:
        text = self.text_processor.execute(text)
        tokens = self.tokenizer.tokenize(
            text, add_sos=self.add_sos, add_eos=self.add_eos
        )
        return torch.LongTensor(tokens), len(tokens)

    def process_speech(self, file_path: Union[Path, str]) -> Tensor:
        speech = self.speech_processor.execute(file_path)
        if speech.dim() == 1:
            # [M]
            speech_len = speech.shape[0]
        elif speech.dim() == 2:
            # [B, M]
            speech_len = speech.shape[-1]
        else:
            speech_len = speech.shape[-2]
        return speech, speech_len

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        text, text_len = self.process_text(item[FileKeys.text_key.value])
        speech, speech_len = self.process_speech(item[FileKeys.speech_key.value])
        return speech, speech_len, text, text_len


class _DataLoader(IDataLoader):
    """Builds the iterable data loader basic class.

    Args:
        dataset (object): The dataset.
        batch_size (int): The batch size.
        rank (int): The process rank used in distributed data-parallel
            setting. Default 0.
        world_size (int): The number of total processes used in distributed
            data-parallel settings. Default 1.
        shuffle (bool): A flag indicating whether the dataset should be
            shuffled at each iteration Default False.
    """

    def __init__(
        self,
        dataset: object,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.data = dataset
        self.indices = [*range(rank, len(self.data), self.world_size)]
        self.length = len(self.indices)
        self._counter = 0
        self.batch_size = batch_size
        self.n_batches = self.length // self.batch_size
        self.shuffle = shuffle

    @property
    def start_idx(self):
        return self._counter * self.batch_size

    @property
    def end_idx(self):
        return min(self.length, (1 + self._counter) * self.batch_size)

    def __len__(self):
        return self.n_batches


class SpeechTextLoader(_DataLoader):
    """Build the speech-text iterable data loader

    Args:
        dataset (object): The dataset.
        batch_size (int): The batch size.
        text_padder (IPadder): The text padder.
        speech_padder (IPadder): The speech padder.
        rank (int): The process rank used in distributed data-parallel
            setting. Default 0.
        world_size (int): The number of total processes used in distributed
            data-parallel settings. Default 1.
        shuffle (bool): A flag indicating whether the dataset should be
            shuffled at each iteration Default False.
    """

    def __init__(
        self,
        dataset: object,
        batch_size: int,
        text_padder: IPadder,
        speech_padder: IPadder,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            rank=rank,
            world_size=world_size,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.text_padder = text_padder
        self.speech_padder = speech_padder

    def _stack_padded(self, batch: List[Tuple[Tensor, int]]) -> Tensor:
        return torch.vstack(list(map(lambda x: x[0], batch)))

    def _get_mask(self, batch: List[Tuple[Tensor, int]], max_len_dim: int) -> Tensor:
        def get_mask(x: Tuple[Tensor, int]):
            (example, pad_len) = x
            seq_len = example.shape[max_len_dim]
            return get_pad_mask(seq_len=seq_len - pad_len, pad_len=pad_len)

        masks = list(map(get_mask, batch))
        return torch.vstack(masks)

    def get_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # TODO: Add multi-threading here
        max_speech_len = 0
        max_text_len = 0
        speeches = []
        texts = []
        for idx in self.indices[self.start_idx : self.end_idx]:
            speech, speech_len, text, text_len = self.data[idx]
            max_speech_len = max(max_speech_len, speech_len)
            max_text_len = max(max_text_len, text_len)
            speeches.append(speech)
            texts.append(text)
        speech = [
            self.speech_padder.pad(speech, max_len=max_speech_len)
            for speech in speeches
        ]
        text = [self.text_padder.pad(text, max_text_len) for text in texts]
        speech_mask = self._get_mask(speech, max_len_dim=-2)
        speech = self._stack_padded(speech)
        text_mask = self._get_mask(text, max_len_dim=0)
        text = self._stack_padded(text)
        return speech, speech_mask, text, text_mask

    def __iter__(self):
        self._counter = 0
        if self.shuffle is True:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self._counter >= self.n_batches:
            raise StopIteration
        batch = self.get_batch()
        self._counter += 1
        return batch
