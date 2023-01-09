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


@dataclass
class BERTTemp(BaseTemplate):
    max_len: int
    in_feature: int
    d_model: int
    h: int
    hidden_size: int
    n_layers: int
    p_dropout: float
    _name = 'bert'
    _type = 'ctc'


@dataclass
class DeepSpeechV2Temp(BaseTemplate):
    n_conv: int
    kernel_size: int
    stride: int
    in_features: int
    hidden_size: int
    bidirectional: bool
    n_rnn: int
    n_linear_layers: int
    max_clip_value: int
    tau: int
    p_dropout: float
    rnn_type: str = 'rnn'
    _name = 'deep_speech_v2'
    _type = 'ctc'


@dataclass
class ConformerCTCTemp(BaseTemplate):
    d_model: int
    n_conf_layers: int
    ff_expansion_factor: int
    h: int
    kernel_size: int
    ss_kernel_size: int
    in_features: int
    res_scaling: float
    p_dropout: float
    _name = 'conformer'
    _type = 'ctc'
