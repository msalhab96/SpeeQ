from dataclasses import asdict, dataclass
from typing import List, Optional, Union
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


@dataclass
class JasperTemp(BaseTemplate):
    in_features: int
    num_blocks: int
    num_sub_blocks: int
    channel_inc: int
    epilog_kernel_size: int
    prelog_kernel_size: int
    prelog_stride: int
    prelog_n_channels: int
    blocks_kernel_size: Union[int, List[int]]
    p_dropout: float
    _name = 'jasper'
    _type = 'ctc'


@dataclass
class Wav2LetterTemp(BaseTemplate):
    in_features: int
    n_conv_layers: int
    layers_kernel_size: int
    layers_channels_size: int
    pre_conv_stride: int
    pre_conv_kernel_size: int
    post_conv_channels_size: int
    post_conv_kernel_size: int
    p_dropout: float
    wav_kernel_size: Optional[int] = None
    wav_stride: Optional[int] = None
    _name = 'wav2letter'
    _type = 'ctc'
