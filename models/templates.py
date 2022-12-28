from dataclasses import asdict, dataclass
from interfaces import ITemplate


class BaseTemplate(ITemplate):
    def get_dict(self):
        return asdict(self)

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type


@dataclass
class DeepSpeechV1Temp(BaseTemplate):
    hidden_size: int
    n_linear_layers: int
    bidirectional: int
    max_clip_value: int
    p_dropout: float
    in_features: int
    rnn_type: str = 'rnn'
    _name = 'deep_speech_v1'
    _type = 'ctc'
