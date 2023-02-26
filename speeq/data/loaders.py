"""
This module contains classes for loading and building data loaders.

Dataset Classes:

- CSVDataset: A base dataset class for handling CSV datasets.

- SpeechTextDataset: A dataset class for speech-text pairs.

Data loader classes

- SpeechTextLoader: An iterable data loader class for speech-text pairs.

The `CSVDataset` class provides a generic base class for handling CSV datasets,
while the `SpeechTextDataset` class is specifically designed for speech-text pairs.
The `SpeechTextLoader` class builds an iterable data loader for speech-text pairs,
which can be used for training speech recognition models.
"""

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
    """A base dataset class for handling CSV datasets.

    Args:
        data_path (Union[str, Path]): The file path of the CSV dataset.

        sep (str): The separator used in the CSV file. Default is ','.

        encoding (str): The encoding of the CSV file. Default is "utf-8".

        sort_key (Optional[str]): The key to sort the data on. Default is an empty string.

        reverse (bool): Used to specify the sorting order. If set to False, data
        will be sorted in ascending order. If set to True, data will be sorted
        in descending order. Default is False.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        sep: str = ",",
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
    """Implements a basic dataset for speech-text pairs to be used in
    speech-recognition.

    Args:
        data_path (Union[str, Path]): The file path for the data in CSV format.

        tokenizer (ITokenizerITokenizer): The tokenizer that will be used to
        process the text data.


        speech_processor (IProcessor): The speech processor, where the `run` method
        returns the speech data with shape [B] or [1, M], or [..., M, F].

        text_processor (IProcessor): The text processor.

        sep (str): The separator used in the CSV file.

        add_sos (bool): A flag that indicates whether to add the Start of
        Sequence (SOS) token to the text sequence. Default is False.

        add_eos (bool): A flag that indicates whether to add the End of Sequence
        (EOS) token to the text sequence. Default is False.

        encoding (Optional[str]): The file encoding. Default "utf-8".

        text_key (Optional[str]): The name of the column that holds the text
        data. Default 'text'.

        speech_key (Optional[str]): The name of the column that holds the audio
        file path. Default 'file_path'

        sort_key (Optional[str]): The key to sort the data on. Default ''.

        reverse (bool): A flag used if a sorting key is passed. If set to False,
        data will be sorted in ascending order. If set to True, data will be
        sorted in descending order. Default is False.

        Example:

        .. code-block:: python

            # Import the module
            from speeq.data.loaders import SpeechTextDataset
            from speeq.data.tokenizers import CharTokenizer
            from speeq.data.processors import OrderedProcessor
            from speeq.data.processes import AudioLoader
            sample_rate = 16000
            sep = ','
            file_path = 'file.csv'

            # creating a dummy tokenizer and processors
            tokenizer = CharTokenizer()
            speech_processor = OrderedProcessor(
                [
                    AudioLoader(sample_rate=sample_rate),
                ]
                )
            text_processor = OrderedProcessor([])
            tokenizer.add_sos_token().add_eos_token()

            # Create an instance of the dataset
            dataset = SpeechTextDataset(
                data_path=file_path,
                tokenizer=tokenizer,
                speech_processor=speech_processor,
                text_processor=text_processor,
                sep=sep,
                add_sos=True
                )

            # to get the first item of the dataset
            speech, speech_len, text, text_len = dataset[0]

            # to get the number of examples in the dataset
            length = len(dataset)

            # to iterate over the dataset
            for speech, speech_len, text, text_len in dataset:
                pass
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
        text_key: Optional[str] = FileKeys.text_key.value,
        speech_key: Optional[str] = FileKeys.speech_key.value,
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
        self.text_key = text_key
        self.speech_key = speech_key

    def _process_text(self, text: str) -> Tuple[Tensor, int]:
        text = self.text_processor.execute(text)
        tokens = self.tokenizer.tokenize(
            text, add_sos=self.add_sos, add_eos=self.add_eos
        )
        return torch.LongTensor(tokens), len(tokens)

    def _process_speech(self, file_path: Union[Path, str]) -> Tuple[Tensor, int]:
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
        text, text_len = self._process_text(item[self.text_key])
        speech, speech_len = self._process_speech(item[self.speech_key])
        return speech, speech_len, text, text_len


class _DataLoader(IDataLoader):
    """
    This class builds an iterable data loader.

    Args:
        dataset (object): The dataset to be loaded.

        batch_size (int): The size of each batch.

        rank (int): The process rank used in distributed data-parallel
        setting. Default is 0.

        world_size (int): The number of total processes used in distributed
        data-parallel settings. Default is 1.

        shuffle (bool): A flag indicating whether the dataset should be
        shuffled at each iteration. Default is False.
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
    """Builds an iterable data loader for speech-text pairs.

    Args:
        dataset (object): The dataset to be loaded, the `__getitem__` method of
        the dataset should return a tuple contains the below in order:

        - The speech tensor of shape [1, M, f]
        - The speech length as integer value equal to M
        - The text tensor of shape [N]
        - The text length as integer value equal to N

        batch_size (int): The size of each batch.

        text_padder (IPadder): The padder for the text data.

        speech_padder (IPadder): The padder for the speech data.

        rank (int): The process rank used in distributed data-parallel
        setting. Default is 0.

        world_size (int): The number of total processes used in distributed
        data-parallel settings. Default is 1.

        shuffle (bool): A flag indicating whether the dataset should be
        shuffled at each iteration. Default is False.

        Example:

        .. code-block:: python

            # Import the module
            from speeq.data.loaders import SpeechTextDataset, SpeechTextLoader
            from speeq.data.padders import DynamicPadder
            from speeq.data.tokenizers import CharTokenizer
            from speeq.data.processors import OrderedProcessor
            from speeq.data.processes import AudioLoader, FeatExtractor
            batch_size = 4
            sample_rate = 16000
            sep = ','
            file_path = 'clean_data.csv'

            # creating a dummy tokenizer, processors, and padders
            tokenizer = CharTokenizer()
            speech_processor = OrderedProcessor(
                [
                    AudioLoader(sample_rate=sample_rate),
                    FeatExtractor(feat_ext_name='mfcc', feat_ext_args={})
                ]
                )
            text_processor = OrderedProcessor([])
            tokenizer.add_sos_token().add_eos_token()
            speech_padder = DynamicPadder(dim=1, pad_val=0.0)
            text_padder = DynamicPadder(dim=0, pad_val=-1)

            # Create an instance of a dataset
            dataset = SpeechTextDataset(
                data_path=file_path,
                tokenizer=tokenizer,
                speech_processor=speech_processor,
                text_processor=text_processor,
                sep=sep,
                add_sos=True
                )

            # Create an instance of the data loader
            loader = SpeechTextLoader(
                dataset=dataset,
                batch_size=batch_size,
                text_padder=text_padder,
                speech_padder=speech_padder
            )

            # to get the number of batches
            n_batches = len(loader)

            # to iterate over the loader
            for batch in dataset:
                speech, speech_len, text, text_len = batch
                break
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
        """Prepares and returns a batch of examples

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: A tuple containing the following tensors
            in order: speech tensor of shape [B, M, d], speech mask tensor of shape [B, M],
            text tensor of shape [B, M], and text mask tensor of shape [B, M].
        """

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
