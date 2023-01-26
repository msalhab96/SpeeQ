from pathlib import Path
from typing import Union

import torch

from config import ModelConfig
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
        model, *_ = load_state_dict(self.cfg.model_config.model_path)
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
