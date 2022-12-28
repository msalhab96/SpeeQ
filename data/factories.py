import os
from typing import Union
from constants import FileKeys
from data.loaders import SpeechTextDataset, SpeechTextLoader
from data.padders import DynamicPadder
from .interfaces import IDataset, ITokenizer
from utils.utils import load_csv
from .tokenizer import CharTokenizer


def get_tokenizer(data_config: object) -> ITokenizer:
    tokenizer_path = data_config.tokenizer_path
    tokenizer = CharTokenizer()
    if os.path.exists(tokenizer_path) is True:
        tokenizer.load_tokenizer(tokenizer_path)
        print(f'Tokenizer {tokenizer_path} loadded!')
    else:
        tokenizer.add_pad_token().add_blank_token()
        tokenizer.add_sos_token().add_eos_token()
        data = load_csv(data_config.training_path, sep=data_config.sep)
        print(data[0])
        print(data[0]['text'])
        data = [item[FileKeys.text_key.value] for item in data]
        tokenizer.set_tokenizer(data)
        tokenizer.save_tokenizer(tokenizer_path)
        print(f'Tokenizer saved to {tokenizer_path}!')
    return tokenizer


def get_asr_datasets(
        data_config: object,
        is_ctc: bool,
        tokenizer: ITokenizer
        ) -> IDataset:
    train_dataset = SpeechTextDataset(
        data_path=data_config.training_path,
        tokenizer=tokenizer,
        speech_processor=data_config.speech_processor,
        text_processor=data_config.text_processor,
        sep=data_config.sep,
        add_eos=is_ctc,
        add_sos=is_ctc
    )
    test_dataset = SpeechTextDataset(
        data_path=data_config.testing_path,
        tokenizer=tokenizer,
        speech_processor=data_config.speech_processor,
        text_processor=data_config.text_processor,
        sep=data_config.sep,
        add_eos=is_ctc,
        add_sos=is_ctc
    )
    return train_dataset, test_dataset


def get_text_padder(
        data_config: object,
        pad_val: Union[float, int]
        ):
    # TODO: add static padding here!
    return DynamicPadder(
        dim=0, pad_val=pad_val
    )


def get_speech_padder(data_config):
    return DynamicPadder(
        dim=1, pad_val=0.0
    )


def get_asr_loaders(
        data_config: object,
        is_ctc: bool,
        tokenizer: ITokenizer,
        batch_size: int,
        world_size: int,
        rank: int
        ):
    train_dataset, test_dataset = get_asr_datasets(
        data_config=data_config,
        is_ctc=is_ctc,
        tokenizer=tokenizer
    )
    text_padder = get_text_padder(
            data_config, tokenizer.special_tokens.pad_id
            )
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
