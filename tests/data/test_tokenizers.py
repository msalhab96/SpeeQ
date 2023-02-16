import json
import os

import pytest

from speeq.data import tokenizers


class TestCharTokenizer:
    def test_vocab_size(self):
        tokenizer = tokenizers.CharTokenizer()
        assert tokenizer.vocab_size == 1
        assert tokenizer.add_token("a")
        assert tokenizer.add_token("a")
        assert tokenizer.vocab_size == 2
        assert tokenizer.add_token("b")
        assert tokenizer.vocab_size == 3
        assert tokenizer.add_token("c")
        assert tokenizer.vocab_size == 4
        assert tokenizer.add_token("d")

    def test_add_token(self):
        tokenizer = tokenizers.CharTokenizer()
        tokenizer.add_token("a")
        tokenizer.add_token("b")
        assert "a" in tokenizer._token_to_id
        assert "b" in tokenizer._token_to_id

    def test_add_token_id_consistency(self):
        tokenizer = tokenizers.CharTokenizer()
        assert "a" not in tokenizer._token_to_id
        tokenizer.add_token("a")
        id = tokenizer._token_to_id["a"]
        tokenizer.add_token("b")
        tokenizer.add_token("a")
        assert id == tokenizer._token_to_id["a"]

    def test_add_pad_tokens(self):
        tokenizer = tokenizers.CharTokenizer()
        # Testing pad token
        assert tokenizers.PAD != tokenizer.special_tokens.pad_token
        assert tokenizer.special_tokens.pad_token is None
        assert tokenizer.special_tokens.pad_id is None
        tokenizer.add_pad_token()
        assert tokenizers.PAD == tokenizer.special_tokens.pad_token
        assert tokenizer.special_tokens.pad_token is not None
        assert tokenizer.special_tokens.pad_id is not None
        pad_id = tokenizer.special_tokens.pad_id
        tokenizer.add_pad_token()
        assert tokenizer.special_tokens.pad_id == pad_id
        tokenizer.add_blank_token()
        assert tokenizer.special_tokens.pad_id == pad_id

    def test_add_eos_tokens(self):
        tokenizer = tokenizers.CharTokenizer()
        assert tokenizers.EOS != tokenizer.special_tokens.eos_token
        assert tokenizer.special_tokens.eos_token is None
        assert tokenizer.special_tokens.eos_id is None
        tokenizer.add_eos_token()
        assert tokenizers.EOS == tokenizer.special_tokens.eos_token
        assert tokenizer.special_tokens.eos_token is not None
        assert tokenizer.special_tokens.eos_id is not None
        eos_id = tokenizer.special_tokens.eos_id
        tokenizer.add_eos_token()
        assert tokenizer.special_tokens.eos_id == eos_id
        tokenizer.add_blank_token()
        assert tokenizer.special_tokens.eos_id == eos_id

    def test_add_sos_tokens(self):
        tokenizer = tokenizers.CharTokenizer()
        assert tokenizers.SOS != tokenizer.special_tokens.sos_token
        assert tokenizer.special_tokens.sos_token is None
        assert tokenizer.special_tokens.sos_id is None
        tokenizer.add_sos_token()
        assert tokenizers.SOS == tokenizer.special_tokens.sos_token
        assert tokenizer.special_tokens.sos_token is not None
        assert tokenizer.special_tokens.sos_id is not None
        sos_id = tokenizer.special_tokens.sos_id
        tokenizer.add_sos_token()
        assert tokenizer.special_tokens.sos_id == sos_id
        tokenizer.add_blank_token()
        assert tokenizer.special_tokens.sos_id == sos_id

    def test_add_blank_tokens(self):
        tokenizer = tokenizers.CharTokenizer()
        assert tokenizers.BLANK != tokenizer.special_tokens.blank_token
        assert tokenizer.special_tokens.blank_token is None
        assert tokenizer.special_tokens.blank_id is None
        tokenizer.add_blank_token()
        assert tokenizers.BLANK == tokenizer.special_tokens.blank_token
        assert tokenizer.special_tokens.blank_token is not None
        assert tokenizer.special_tokens.blank_id is not None
        blank_id = tokenizer.special_tokens.blank_id
        tokenizer.add_blank_token()
        assert tokenizer.special_tokens.blank_id == blank_id
        tokenizer.add_sos_token()
        assert tokenizer.special_tokens.blank_id == blank_id

    def test_load_tokenizer_from_dict(self, char_tokenizer_dict):
        tokenizer = tokenizers.CharTokenizer()
        tokenizer.load_tokenizer_from_dict(char_tokenizer_dict)
        assert tokenizer._token_to_id == char_tokenizer_dict["token_to_id"]
        assert tokenizer.special_tokens.pad_id is not None
        assert tokenizer.special_tokens.oov_id is not None
        assert tokenizer.special_tokens.sos_id is not None

    def test_load_tokenizer(self, char_tokenizer_dict, tmp_path):
        file_path = os.path.join(tmp_path, "file.json")
        tokenizer = tokenizers.CharTokenizer()
        with open(file_path, "w") as f:
            json.dump(char_tokenizer_dict, f)
        tokenizer.load_tokenizer(file_path)
        assert tokenizer._token_to_id == char_tokenizer_dict["token_to_id"]
        assert tokenizer.special_tokens.pad_id is not None
        assert tokenizer.special_tokens.oov_id is not None
        assert tokenizer.special_tokens.sos_id is not None

    def test_ids2tokens(self, char_tokenizer_dict):
        tokenizer = tokenizers.CharTokenizer()
        tokenizer.load_tokenizer_from_dict(char_tokenizer_dict)
        ids = list(char_tokenizer_dict["token_to_id"].values())
        tokens = list(char_tokenizer_dict["token_to_id"].keys())
        assert tokens == tokenizer.ids2tokens(ids)

    @pytest.mark.parametrize(
        ("sentence", "add_sos", "add_eos", "expected"),
        (
            ("aaba", False, False, [4, 4, 5, 4]),
            (
                "a",
                False,
                False,
                [
                    4,
                ],
            ),
            ("abba", True, False, [2, 4, 5, 5, 4]),
            ("aaba", True, True, [None]),  # assertoin
        ),
    )
    def test_tokenize(self, char_tokenizer_dict, sentence, add_sos, add_eos, expected):
        tokenizer = tokenizers.CharTokenizer()
        tokenizer.load_tokenizer_from_dict(char_tokenizer_dict)
        if add_eos is True:
            with pytest.raises(AssertionError):
                tokenizer.tokenize(sentence=sentence, add_eos=add_eos, add_sos=add_sos)
        else:
            result = tokenizer.tokenize(
                sentence=sentence, add_eos=add_eos, add_sos=add_sos
            )
            assert result == expected

    @pytest.mark.parametrize(
        ("data", "expected"),
        (
            (["dada d v", "132"], {"1", "2", "3", " ", "d", "v", "a"}),
            (["132"], {"1", "2", "3"}),
            (["1"], {"1"}),
            ([""], set()),
            ([" "], {" "}),
        ),
    )
    def test_get_tokens(self, data, expected):
        tokenizer = tokenizers.CharTokenizer()
        result = tokenizer.get_tokens(data=data)
        assert result - expected == set()


class TestWordTokenizer:
    @pytest.mark.parametrize(
        ("data", "expected"),
        (
            (["dada d v", "132"], {"132", "dada", "d", "v"}),
            (["132"], {"132"}),
            (["1"], {"1"}),
            ([""], set()),
        ),
    )
    def test_get_tokens(self, data, expected):
        tokenizer = tokenizers.WordTokenizer()
        result = tokenizer.get_tokens(data=data)
        assert result - expected == set()

    @pytest.mark.parametrize(
        ("sentence", "add_sos", "add_eos", "expected"),
        (
            ("a a b a", False, False, [4, 4, 5, 4]),
            (
                "a",
                False,
                False,
                [
                    4,
                ],
            ),
            ("a b b a", True, False, [2, 4, 5, 5, 4]),
            ("a a b a", True, True, [None]),  # assertoin
        ),
    )
    def test_tokenize(self, word_tokenizer_dict, sentence, add_sos, add_eos, expected):
        tokenizer = tokenizers.WordTokenizer()
        tokenizer.load_tokenizer_from_dict(word_tokenizer_dict)
        if add_eos is True:
            with pytest.raises(AssertionError):
                tokenizer.tokenize(sentence=sentence, add_eos=add_eos, add_sos=add_sos)
        else:
            result = tokenizer.tokenize(
                sentence=sentence, add_eos=add_eos, add_sos=add_sos
            )
            assert result == expected
