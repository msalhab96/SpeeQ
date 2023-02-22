"""
This module serves as a factory for creating objects/instances from other
classes by abstracting the creation process. It contains several functions
that return instances of classes such as tokenizer, ASR datasets, and data
loaders, making it easier to manage dependencies and abstract away the details
of object creation.

Functions:

- get_tokenizer: Returns an instance of a tokenizer object.
- load_tokenizer: Loads and returns a pre-trained tokenizer instance.
- get_asr_datasets: Returns instances of training and testing datasets for ASR tasks.
- get_text_padder: Returns an instance of a text padder object.
- get_speech_padder: Returns an instance of a speech padder object.
- get_asr_loaders: Returns instances of training and testing data loaders for ASR tasks.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

from speeq.constants import CHAR_TOKENIZER_TYPE, TOKENIZER_TYPE_KEY, WORD_TOKENIZER_TYPE
from speeq.interfaces import IDataLoader, IDataset, IPadder, ITokenizer
from speeq.utils.utils import load_json

from .loaders import SpeechTextDataset, SpeechTextLoader
from .padders import DynamicPadder, StaticPadder
from .tokenizers import CharTokenizer, WordTokenizer

PADDING_TYPES = {"static": StaticPadder, "dynamic": DynamicPadder}

TOKENIZERS = {WORD_TOKENIZER_TYPE: WordTokenizer, CHAR_TOKENIZER_TYPE: CharTokenizer}


def get_tokenizer(data_config: object, data: Optional[List[str]] = None) -> ITokenizer:
    """Creates a tokenizer based on the provided data configuration, or loads
    a pre-trained tokenizer from a file. If a pre-trained tokenizer path is not
    provided, the function trains the tokenizer on the provided data.


    Args:
        data_config (object): An object that contains the configuration for the
        data.

        data (List[str], optional): A list of strings to train the tokenizer on.
        Defaults to None.

    Returns:
        ITokenizer: A tokenizer object.
    """
    tokenizer_path = data_config.tokenizer_path
    if data_config.tokenizer_type not in TOKENIZERS:
        raise KeyError(
            f"invalid tokenizer name, please use one of {list(TOKENIZERS.keys())}"
        )
    if os.path.exists(tokenizer_path) is True:
        tokenizer = load_tokenizer(tokenizer_path)
        print(f"Tokenizer {tokenizer_path} loadded!")
        return tokenizer
    assert data is not None
    tokenizer = TOKENIZERS[data_config.tokenizer_type]()
    tokenizer.add_pad_token().add_blank_token()
    tokenizer.add_sos_token().add_eos_token()
    tokenizer.set_tokenizer(data)
    tokenizer.save_tokenizer(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}!")
    return tokenizer


def load_tokenizer(tokenizer_path: Union[Path, str]) -> ITokenizer:
    """Loads a pre-trained tokenizer from the specified path.

    Args:
        tokenizer_path (Union[Path, str]): A path to the pre-trained tokenizer file.

    Returns:
        ITokenizer: An object representing the loaded tokenizer.
    """
    data = load_json(tokenizer_path)
    type = data[TOKENIZER_TYPE_KEY]
    return TOKENIZERS[type]().load_tokenizer_from_dict(data)


def get_asr_datasets(
    data_config: object, tokenizer: ITokenizer
) -> Tuple[IDataset, IDataset]:
    """Creates train and test dataset objects based on the provided data
    configuration and tokenizer.


    Args:
        data_config (object): Data configuration object.

        tokenizer (ITokenizer): The tokenizer to tokenize the test data.

    Returns:
        Tuple[IDataset, IDataset]: A tuple containing the train and test dataset
        objects.
    """
    train_dataset = SpeechTextDataset(
        data_path=data_config.training_path,
        tokenizer=tokenizer,
        speech_processor=data_config.speech_processor,
        text_processor=data_config.text_processor,
        sep=data_config.sep,
        add_eos=data_config.add_eos_token,
        add_sos=data_config.add_sos_token,
        sort_key=data_config.sort_key,
        reverse=data_config.reverse,
    )
    test_dataset = SpeechTextDataset(
        data_path=data_config.testing_path,
        tokenizer=tokenizer,
        speech_processor=data_config.speech_processor,
        text_processor=data_config.text_processor,
        sep=data_config.sep,
        add_eos=data_config.add_eos_token,
        add_sos=data_config.add_sos_token,
        sort_key=data_config.sort_key,
        reverse=data_config.reverse,
    )
    return train_dataset, test_dataset


def get_text_padder(data_config: object, pad_val: Union[float, int]) -> IPadder:
    """Creates a text padding object.

    Args:
        data_config (object): The data configuration object.

        pad_val (Union[float, int]): The value that will be used for padding.

    Returns:
        IPadder: A padder object that can be used to pad sequences of variable
        length to a fixed length, by adding padding values at the beginning or
        end of the sequence. The padding is applied along the zeroth dimension
        of the input tensor.
    """
    return PADDING_TYPES[data_config.padding_type](
        dim=0, pad_val=pad_val, max_len=data_config.text_pad_max_len
    )


def get_speech_padder(data_config) -> IPadder:
    """Creates a speech padding object.

    Args:
        data_config (object): The data configuration object.

        pad_val (Union[float, int]): The value that will be used for padding.

    Returns:
        IPadder: A padder object that can be used to pad sequences of variable
        length to a fixed length, by adding padding values at the beginning or
        end of the sequence. The padding is applied along the first dimension
        of the input tensor.
    """
    return PADDING_TYPES[data_config.padding_type](
        dim=1, pad_val=0.0, max_len=data_config.speech_pad_max_len
    )


def get_asr_loaders(
    data_config: object,
    tokenizer: ITokenizer,
    batch_size: int,
    world_size: int,
    rank: int,
) -> Tuple[IDataLoader, IDataLoader]:
    """Builds training and testing dataloaders.

    Args:
        data_config (object): Data configuration object.

        tokenizer (ITokenizer): the text tokenizer.

        batch_size (int): The batch size.

        world_size (int): The number of nodes/gpus.

        rank (int): the index of the current process/gpu will use the data loaders.

    Returns:
        Tuple[IDataLoader, IDataLoader]: The training and testing data
        loaders.
    """
    train_dataset, test_dataset = get_asr_datasets(
        data_config=data_config, tokenizer=tokenizer
    )
    if data_config.use_blank_as_pad is True:
        pad_id = tokenizer.special_tokens.blank_id
    else:
        pad_id = tokenizer.special_tokens.pad_id
    text_padder = get_text_padder(data_config, pad_id)
    speech_padder = get_speech_padder(data_config)
    train_loader = SpeechTextLoader(
        dataset=train_dataset,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        text_padder=text_padder,
        speech_padder=speech_padder,
        shuffle=data_config.shuffle,
    )
    test_loader = SpeechTextLoader(
        dataset=test_dataset,
        rank=0,
        world_size=1,
        batch_size=batch_size,
        text_padder=text_padder,
        speech_padder=speech_padder,
    )
    return train_loader, test_loader
