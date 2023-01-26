import os
from pathlib import Path
from typing import Tuple, Union

from constants import FileKeys
from data.loaders import SpeechTextDataset, SpeechTextLoader
from data.padders import DynamicPadder, StaticPadder
from interfaces import IDataLoader, IDataset, IPadder, ITokenizer
from utils.utils import load_csv

from .tokenizer import CharTokenizer

PADDING_TYPES = {
    'static': StaticPadder,
    'dynamic': DynamicPadder
}


def get_tokenizer(data_config: object) -> ITokenizer:
    """Creates a tokenizer based on the training data,
    or load a pre-built tokenizer from file.

    Args:
        data_config (object): Data configuration object.

    Returns:
        ITokenizer: A tokenizer object.
    """
    tokenizer_path = data_config.tokenizer_path
    tokenizer = CharTokenizer()
    if os.path.exists(tokenizer_path) is True:
        tokenizer.load_tokenizer(tokenizer_path)
        print(f'Tokenizer {tokenizer_path} loadded!')
    else:
        tokenizer.add_pad_token().add_blank_token()
        tokenizer.add_sos_token().add_eos_token()
        data = load_csv(data_config.training_path, sep=data_config.sep)
        data = [item[FileKeys.text_key.value] for item in data]
        tokenizer.set_tokenizer(data)
        tokenizer.save_tokenizer(tokenizer_path)
        print(f'Tokenizer saved to {tokenizer_path}!')
    return tokenizer


def load_tokenizer(tokenizer_path: Union[Path, str]) -> ITokenizer:
    return CharTokenizer().load_tokenizer(tokenizer_path)


def get_asr_datasets(
        data_config: object,
        tokenizer: ITokenizer
) -> Tuple[IDataset, IDataset]:
    """Creates a train and test dataset objects

    Args:
        data_config (object): Data configuration object.
        tokenizer (ITokenizer): The tokenizer to tokenize the test data.

    Returns:
        Tuple[IDataset, IDataset]: The train and test datasets.
    """
    train_dataset = SpeechTextDataset(
        data_path=data_config.training_path,
        tokenizer=tokenizer,
        speech_processor=data_config.speech_processor,
        text_processor=data_config.text_processor,
        sep=data_config.sep,
        add_eos=data_config.add_eos_token,
        add_sos=data_config.add_sos_token
    )
    test_dataset = SpeechTextDataset(
        data_path=data_config.testing_path,
        tokenizer=tokenizer,
        speech_processor=data_config.speech_processor,
        text_processor=data_config.text_processor,
        sep=data_config.sep,
        add_eos=data_config.add_eos_token,
        add_sos=data_config.add_sos_token
    )
    return train_dataset, test_dataset


def get_text_padder(
        data_config: object,
        pad_val: Union[float, int]
) -> IPadder:
    return PADDING_TYPES[data_config.padding_type](
        dim=0,
        pad_val=pad_val,
        max_len=data_config.text_pad_max_len
    )


def get_speech_padder(data_config) -> IPadder:
    return PADDING_TYPES[data_config.padding_type](
        dim=1,
        pad_val=0.0,
        max_len=data_config.speech_pad_max_len
    )


def get_asr_loaders(
        data_config: object,
        tokenizer: ITokenizer,
        batch_size: int,
        world_size: int,
        rank: int
) -> Tuple[IDataLoader, IDataLoader]:
    """Builds training and testing dataloaders.

    Args:
        data_config (object): Data configuration object.
        tokenizer (ITokenizer): the text tokenizer.
        batch_size (int): The batch size.
        world_size (int): The number of nodes/gpus.
        rank (int): the index of the current process/gpu
        will use the data loaders.

    Returns:
        Tuple[IDataLoader, IDataLoader]: The training and testing data
        loaders.
    """
    train_dataset, test_dataset = get_asr_datasets(
        data_config=data_config,
        tokenizer=tokenizer
    )
    if data_config.use_blank_as_pad is True:
        pad_id = tokenizer.special_tokens.blank_id
    else:
        pad_id = tokenizer.special_tokens.pad_id
    text_padder = get_text_padder(data_config, pad_id)
    speech_padder = get_speech_padder(data_config)
    train_loader = SpeechTextLoader(
        dataset=train_dataset,
        rank=rank, world_size=world_size,
        batch_size=batch_size, text_padder=text_padder,
        speech_padder=speech_padder
    )
    test_loader = SpeechTextLoader(
        dataset=test_dataset, rank=0,
        world_size=1, batch_size=batch_size,
        text_padder=text_padder, speech_padder=speech_padder
    )
    return train_loader, test_loader
