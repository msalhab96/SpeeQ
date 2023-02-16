from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

from speeq.constants import CHAR_TOKENIZER_TYPE, TOKENIZER_TYPE_KEY, WORD_TOKENIZER_TYPE
from speeq.interfaces import ITokenizer
from speeq.utils.utils import load_json, save_json

from .decorators import check_token

PAD = "<PAD>"
OOV = "<OOV>"
SOS = "<SOS>"
EOS = "<EOS>"
BLANK = "<BLANK>"


@dataclass
class _SpecialTokens:
    _pad: Tuple[str, int] = (None, None)
    _blank: Tuple[str, int] = (None, None)
    _sos: Tuple[str, int] = (None, None)
    _eos: Tuple[str, int] = (None, None)
    _oov: Tuple[str, int] = (None, None)

    @property
    def pad_id(self):
        return self._pad[1]

    @property
    def pad_token(self):
        return self._pad[0]

    @property
    def blank_id(self):
        return self._blank[1]

    @property
    def blank_token(self):
        return self._blank[0]

    @property
    def sos_id(self):
        return self._sos[1]

    @property
    def sos_token(self):
        return self._sos[0]

    @property
    def eos_id(self):
        return self._eos[1]

    @property
    def eos_token(self):
        return self._eos[0]

    @property
    def mask_id(self):
        return self._mask[1]

    @property
    def mask_token(self):
        return self._mask[0]

    @property
    def oov_id(self):
        return self._oov[1]

    @property
    def oov_token(self):
        return self._oov[0]


class _BaseTokenizer(ITokenizer):
    _pad_key = "pad"
    _oov_key = "oov"
    _sos_key = "sos"
    _eos_key = "eos"
    _blank_key = "blank"
    _token_to_id_key = "token_to_id"
    _special_tokens_key = "special_tokens"

    def __init__(self) -> None:
        super().__init__()
        self._token_to_id = dict()
        self._id_to_token = dict()
        self.special_tokens = _SpecialTokens()
        self.add_oov_token()

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def add_token(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        token_id = self.vocab_size
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
        return token_id

    @check_token(PAD)
    def add_pad_token(self, token=PAD) -> ITokenizer:
        token_id = self.add_token(token)
        self.special_tokens._pad = (token, token_id)
        return self

    @check_token(BLANK)
    def add_blank_token(self, token=BLANK) -> ITokenizer:
        token_id = self.add_token(token)
        self.special_tokens._blank = (token, token_id)
        return self

    @check_token(SOS)
    def add_sos_token(self, token=SOS) -> ITokenizer:
        token_id = self.add_token(token)
        self.special_tokens._sos = (token, token_id)
        return self

    @check_token(EOS)
    def add_eos_token(self, token=EOS) -> ITokenizer:
        token_id = self.add_token(token)
        self.special_tokens._eos = (token, token_id)
        return self

    @check_token(OOV)
    def add_oov_token(self, token=OOV) -> ITokenizer:
        token_id = self.add_token(token)
        self.special_tokens._oov = (token, token_id)
        return self

    def _reset_id_to_token(self) -> None:
        self._id_to_token = dict(
            zip(self._token_to_id.values(), self._token_to_id.keys())
        )

    def __set_special_tokens_dict(self, data: dict) -> None:
        if self._pad_key in data:
            self.special_tokens._pad = tuple(data[self._pad_key])
        if self._blank_key in data:
            self.special_tokens._blank = tuple(data[self._blank_key])
        if self._sos_key in data:
            self.special_tokens._sos = tuple(data[self._sos_key])
        if self._eos_key in data:
            self.special_tokens._eos = tuple(data[self._eos_key])
        if self._oov_key in data:
            self.special_tokens._oov = tuple(data[self._oov_key])

    def __get_special_tokens_dict(self) -> dict:
        data = {}
        if self.special_tokens.pad_id is not None:
            data[self._pad_key] = list(self.special_tokens._pad)
        if self.special_tokens.blank_id is not None:
            data[self._blank_key] = list(self.special_tokens._blank)
        if self.special_tokens.sos_id is not None:
            data[self._sos_key] = list(self.special_tokens._sos)
        if self.special_tokens.eos_id is not None:
            data[self._eos_key] = list(self.special_tokens._eos)
        if self.special_tokens.oov_id is not None:
            data[self._oov_key] = list(self.special_tokens._oov)
        return data

    def load_tokenizer_from_dict(self, data: dict) -> ITokenizer:
        self._token_to_id = data[self._token_to_id_key]
        self.__set_special_tokens_dict(data[self._special_tokens_key])
        self._reset_id_to_token()
        return self

    def load_tokenizer(
        self, tokenizer_path: Union[str, Path], *args, **kwargs
    ) -> ITokenizer:
        """Loads a pre-trained tokenizer.

        Args:
            tokenizer_path (Union[str, Path]): The pre-trained tokenizer path.

        Returns:
            ITokenizer: The loaded tokenizer.
        """
        if os.path.exists(tokenizer_path) is False:
            raise FileNotFoundError(f"{tokenizer_path} not found!")
        data = load_json(tokenizer_path)
        assert (
            data[TOKENIZER_TYPE_KEY] == self._type
        ), f"""
        The used tokenizer is not matched with the pre-trained tokenizer!
        Given pre-trained tokenizer of type {data[TOKENIZER_TYPE_KEY]} while {self._type}
        is used!
        """
        return self.load_tokenizer_from_dict(data)

    def set_tokenizer(self, data: List[str], *args, **kwargs) -> ITokenizer:
        """Sets/trains the tokenizer on the provided data.

        Args:
            data (List[str]): A list of all text sentences.

        Returns:
            ITokenizer: The trained tokenizer.
        """
        all_tokens = self.get_tokens(data)
        for token in all_tokens:
            self.add_token(token=token)
        self._reset_id_to_token()
        return self

    def save_tokenizer(self, save_path: Union[str, Path], *args, **kwargs) -> None:
        data = {
            TOKENIZER_TYPE_KEY: self._type,
            self._token_to_id_key: self._token_to_id,
            self._special_tokens_key: self.__get_special_tokens_dict(),
        }
        save_json(save_path, data)

    def ids2tokens(self, ids: List[str]) -> List[str]:
        return list(map(lambda x: self._id_to_token[x], ids))

    def tokenize(self, sentence: str, add_sos=False, add_eos=False) -> List[int]:
        """Tokenizes the input sentence.

        Args:
            sentence (str): The sentence to be tokenized.
            add_sos (bool, optional): A flag to whether added SOS token at the
                end of the sequence. Defaults to False.
            add_eos (bool, optional): A flag to whether add EOS token at the
                end of the sequence. Defaults to False.

        Returns:
            List[int]: The tokenized sequence.
        """
        results = []
        if add_sos is True:
            assert self.special_tokens.sos_id is not None
            results.append(self.special_tokens.sos_id)
        tokens = self.preprocess_tokens(sentence)
        results.extend(
            map(lambda x: self._token_to_id.get(x, self.special_tokens.oov_id), tokens)
        )
        if add_eos is True:
            assert self.special_tokens.eos_id is not None
            results.append(self.special_tokens.eos_id)
        return results

    def batch_tokenizer(self, data: List[str], add_sos=False, add_eos=False) -> list:
        def func(sentence):
            return self.tokenize(sentence=sentence, add_sos=add_sos, add_eos=add_eos)

        return list(map(func, data))

    def batch_detokenizer(self, data: List[int]) -> list:
        return list(map(self.ids2tokens, data))


class CharTokenizer(_BaseTokenizer):
    """Implements character based tokenizer."""

    _type = CHAR_TOKENIZER_TYPE

    def __init__(self) -> None:
        super().__init__()

    def get_tokens(self, data: List[str]):
        return set("".join(data))

    def preprocess_tokens(self, sentence: str) -> List[str]:
        return list(sentence)


class WordTokenizer(_BaseTokenizer):
    """Implements white space based tokenizer."""

    _type = WORD_TOKENIZER_TYPE

    def __init__(self, sep=" ") -> None:
        super().__init__()
        self.sep = sep

    def get_tokens(self, data: List[str]):
        result = set()
        for line in data:
            result = result.union(line.split(self.sep))
        return result

    def preprocess_tokens(self, sentence: str) -> List[str]:
        return sentence.split(self.sep)
