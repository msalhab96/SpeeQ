from typing import List
from .layers import (
    PackedGRU,
    PackedLSTM,
    PackedRNN
    )
from .ctc import DeepSpeechV1

RNN_REGISTRY = {
    'rnn': PackedRNN,
    'lstm': PackedLSTM,
    'gru': PackedGRU
}


CTC_MODELS = {
    'deep_speech_v1': DeepSpeechV1
}


def list_ctc_models() -> List[str]:
    """Lists all pre-implemented ctc based
    models.
    """
    return list(CTC_MODELS.values())


def get_model(model_config, n_classes):
    if model_config.template.type == 'ctc':
        return CTC_MODELS[model_config.template.name](
            **model_config.template.get_dict(), n_classes=n_classes
        )
