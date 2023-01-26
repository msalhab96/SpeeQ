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
