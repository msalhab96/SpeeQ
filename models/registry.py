from models.seq2seq import (
    LAS, BasicAttSeq2SeqRNN,
    RNNWithLocationAwareAtt, SpeechTransformer
    )
from torch import nn
from typing import List
from .layers import (
    PackedGRU,
    PackedLSTM,
    PackedRNN
    )
from .ctc import (
    BERT, Conformer, DeepSpeechV1, DeepSpeechV2,
    Jasper, QuartzNet, Squeezeformer, Wav2Letter
    )

PACKED_RNN_REGISTRY = {
    'rnn': PackedRNN,
    'lstm': PackedLSTM,
    'gru': PackedGRU
}

RNN_REGISTRY = {
    'rnn': nn.RNN,
    'lstm': nn.LSTM,
    'gru': nn.GRU
}

CTC_MODELS = {
    'deep_speech_v1': DeepSpeechV1,
    'deep_speech_v2': DeepSpeechV2,
    'bert': BERT,
    'conformer': Conformer,
    'jasper': Jasper,
    'wav2letter': Wav2Letter,
    'quartz_net': QuartzNet,
    'squeezeformer': Squeezeformer
}

SEQ2SEQ_MODEL = {
    'las': LAS,
    'rnn_with_location_att': RNNWithLocationAwareAtt,
    'basic_att_rnn': BasicAttSeq2SeqRNN,
    'speech_transformer': SpeechTransformer
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
    if model_config.template.type == 'seq2seq':
        return SEQ2SEQ_MODEL[model_config.template.name](
            **model_config.template.get_dict(), n_classes=n_classes
        )
