from pathlib import Path
from typing import Union

import torch

from config import ModelConfig
from constants import (ENC_OUT_KEY, HIDDEN_STATE_KEY, PREDS_KEY,
                       PREV_HIDDEN_STATE_KEY, SPEECH_IDX_KEY,
                       TERMINATION_STATE_KEY)
from data.registry import load_tokenizer
from interfaces import IProcessor
from models.registry import get_model
from utils.utils import load_state_dict


class _ASRBasePredictor:
    """Implements the base ASR predictor
    """

    def __init__(
            self,
            speech_processor: IProcessor,
            tokenizer_path: Union[str, Path],
            model_config: ModelConfig,
            device: str
    ) -> None:
        self.speech_processor = speech_processor
        self.tokenizer = load_tokenizer(
            tokenizer_path=tokenizer_path
        )
        self.device = device
        self.model = get_model(
            model_config=model_config,
            n_classes=self.tokenizer.vocab_size
        ).to(self.device)
        model, *_ = load_state_dict(model_config.model_path)
        self.model.load_state_dict(model)
        self.model.eval()
        self.blank_id = self.tokenizer.special_tokens.blank_id
        self.sos = self.tokenizer.special_tokens.sos_id
        self.eos = self.tokenizer.special_tokens.eos_id


class CTCPredictor(_ASRBasePredictor):
    """Implements CTC based model predictor

    Args:
        speech_processor (IProcessor): The speech/file pre-processing
            processor.
        tokenizer_path (Union[str, Path]): The trained tokenizer path.
        model_config (ModelConfig): The model configuration.
        device (str): The device to map the operations to.
    """

    def __init__(
            self,
            speech_processor: IProcessor,
            tokenizer_path: Union[str, Path],
            model_config: ModelConfig, device: str,
            *args, **kwargs
    ) -> None:
        super().__init__(
            speech_processor, tokenizer_path, model_config, device
        )

    def predict(self, file_path: Union[Path, str]) -> str:
        speech = self.speech_processor.execute(file_path)
        speech = speech.to(self.device)
        mask = torch.ones(1, speech.shape[1], dtype=torch.bool)
        mask = mask.to(self.device)
        preds, _ = self.model(speech, mask)  # M, 1, C
        preds = torch.argmax(preds, dim=-1)
        preds = preds.squeeze().tolist()
        results = []
        last_idx = -1
        for item in preds:
            if item not in [last_idx, self.blank_id]:
                results.append(item)
                last_idx = item
        if results[0] == self.sos:
            results = results[1:]
        if results[-1] == self.eos:
            results = results[:-1]
        return ''.join(self.tokenizer.ids2tokens(results))


class Seq2SeqPredictor(_ASRBasePredictor):
    """Implements Seq2Seq-Based models predictor

    Args:
        speech_processor (IProcessor): The speech/file pre-processing
            processor.
        tokenizer_path (Union[str, Path]): The trained tokenizer path.
        model_config (ModelConfig): The model configuration.
        device (str): The device to map the operations to.
        max_len (int): The maximum decoding length.
    """

    def __init__(
            self,
            speech_processor: IProcessor,
            tokenizer_path: Union[str, Path],
            model_config: ModelConfig,
            device: str,
            max_len: int,
            *args, **kwargs
    ) -> None:
        super().__init__(
            speech_processor, tokenizer_path, model_config, device
        )
        self.max_len = max_len
        # TODO: Add beam search

    def predict(self, file_path: Union[Path, str]) -> str:
        speech = self.speech_processor.execute(file_path)
        speech = speech.to(self.device)
        mask = torch.ones(1, speech.shape[1], dtype=torch.bool)
        mask = mask.to(self.device)
        counter = 0
        state = {
            PREDS_KEY: torch.LongTensor([[self.sos]]).to(self.device),
            TERMINATION_STATE_KEY: False
        }
        while counter <= self.max_len:
            state = self.model.predict(speech, mask, state)
            last_idx = state[PREDS_KEY][0, -1].item()
            state[TERMINATION_STATE_KEY] = last_idx == self.eos
            if state[TERMINATION_STATE_KEY] is True:
                break
            counter += 1
        results = state[PREDS_KEY][0, 1:-1].tolist()
        return ''.join(self.tokenizer.ids2tokens(results))


class TransducerPredictor(_ASRBasePredictor):
    """Implements transducer-Based models predictor

    Args:
        speech_processor (IProcessor): The speech/file pre-processing
            processor.
        tokenizer_path (Union[str, Path]): The trained tokenizer path.
        model_config (ModelConfig): The model configuration.
        device (str): The device to map the operations to.
    """

    def __init__(
            self,
            speech_processor: IProcessor,
            tokenizer_path: Union[str, Path],
            model_config: ModelConfig,
            device: str
    ) -> None:
        super().__init__(
            speech_processor,
            tokenizer_path,
            model_config,
            device
        )
        # TODO: Add Beam search here

    def predict(self, file_path: Union[Path, str]) -> str:
        speech = self.speech_processor.execute(file_path)
        speech = speech.to(self.device)
        mask = torch.ones(1, speech.shape[1], dtype=torch.bool)
        mask = mask.to(self.device)
        state = {
            PREDS_KEY: torch.LongTensor([[self.sos]]).to(self.device),
            SPEECH_IDX_KEY: 0,
            HIDDEN_STATE_KEY: None
        }
        length = 1
        while state[SPEECH_IDX_KEY] < length:
            state = self.model.predict(speech, mask, state)
            length = state[ENC_OUT_KEY].shape[1]
            last_pred = state[PREDS_KEY][0, -1].item()
            if last_pred == self.blank_id:
                state[SPEECH_IDX_KEY] += 1
                state[HIDDEN_STATE_KEY] = state[PREV_HIDDEN_STATE_KEY]
                # print(state[HIDDEN_STATE_KEY])
                state[PREDS_KEY] = state[PREDS_KEY][:, :-1]
        results = state[PREDS_KEY][0, :].tolist()
        return ''.join(self.tokenizer.ids2tokens(results[1:]))
