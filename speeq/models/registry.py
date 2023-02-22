from typing import List

from torch import nn

from speeq.config import ModelConfig
from speeq.constants import CTC_TYPE, MODEL_BUILDER_TYPE, SEQ2SEQ_TYPE, TRANSDUCER_TYPE

from .ctc import (
    BERT,
    Conformer,
    DeepSpeechV1,
    DeepSpeechV2,
    Jasper,
    QuartzNet,
    Squeezeformer,
    Wav2Letter,
)
from .layers import PackedGRU, PackedLSTM, PackedRNN
from .seq2seq import LAS, BasicAttSeq2SeqRNN, RNNWithLocationAwareAtt, SpeechTransformer
from .skeletons import CTCSkeleton, Seq2SeqSkeleton, TransducerSkeleton
from .transducers import ConformerTransducer, ContextNet, RNNTransducer

PACKED_RNN_REGISTRY = {"rnn": PackedRNN, "lstm": PackedLSTM, "gru": PackedGRU}

RNN_REGISTRY = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

CTC_MODELS = {
    "deep_speech_v1": DeepSpeechV1,
    "deep_speech_v2": DeepSpeechV2,
    "bert": BERT,
    "conformer": Conformer,
    "jasper": Jasper,
    "wav2letter": Wav2Letter,
    "quartz_net": QuartzNet,
    "squeezeformer": Squeezeformer,
}

SEQ2SEQ_MODELS = {
    "las": LAS,
    "rnn_with_location_att": RNNWithLocationAwareAtt,
    "basic_att_rnn": BasicAttSeq2SeqRNN,
    "speech_transformer": SpeechTransformer,
}

TRANSDUCER_MODELS = {
    "rnn-t": RNNTransducer,
    "conformer": ConformerTransducer,
    "context_net": ContextNet,
}

MODELS_BUILDER = {
    CTC_TYPE: CTCSkeleton,
    TRANSDUCER_TYPE: TransducerSkeleton,
    SEQ2SEQ_TYPE: Seq2SeqSkeleton,
}


def list_ctc_models() -> List[str]:
    """Lists all pre-implemented ctc based
    models.
    """
    return list(CTC_MODELS.values())


def list_seq2seq_models() -> List[str]:
    """Lists all pre-implemented seq2seq based
    models.
    """
    return list(SEQ2SEQ_MODELS.values())


def list_transducer_models() -> List[str]:
    """Lists all pre-implemented transducer based
    models.
    """
    return list(TRANSDUCER_MODELS.values())


def get_model(model_config: ModelConfig, n_classes: int) -> nn.Module:
    """Creates and returns a targeted model using the provided configuration
    object `model_config`.

    Args:
        model_config (object): The model configuration object.

        n_classes (int): The number of classes for the model to predict.

    Returns:
        Module: The targeted model created using the configuration object.
    """
    if model_config.template.type == CTC_TYPE:
        return CTC_MODELS[model_config.template.name](
            **model_config.template.get_dict(), n_classes=n_classes
        )
    if model_config.template.type == SEQ2SEQ_TYPE:
        return SEQ2SEQ_MODELS[model_config.template.name](
            **model_config.template.get_dict(), n_classes=n_classes
        )
    if model_config.template.type == TRANSDUCER_TYPE:
        return TRANSDUCER_MODELS[model_config.template.name](
            **model_config.template.get_dict(), n_classes=n_classes
        )
    if model_config.template.type == MODEL_BUILDER_TYPE:
        return MODELS_BUILDER[model_config.template.name](
            **model_config.template.get_dict(), n_classes=n_classes
        )
