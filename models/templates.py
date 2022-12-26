from dataclasses import asdict, dataclass
from abc import ABC, abstractproperty, abstractmethod


class ITemplate(ABC):

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def type(self):
        pass

    @abstractmethod
    def get_dict(self):
        pass


@dataclass
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
