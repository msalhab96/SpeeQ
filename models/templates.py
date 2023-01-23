from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple, Union
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
    ss_stride: int
    ss_num_conv_layers: int
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


@dataclass
class LASTemp(BaseTemplate):
    in_features: int
    hidden_size: int
    enc_num_layers: int
    reduction_factor: int
    bidirectional: bool
    dec_num_layers: int
    emb_dim: int
    p_dropout: float
    teacher_forcing_rate: float = 0.0
    rnn_type: str = 'rnn'
    _name = 'las'
    _type = 'seq2seq'


@dataclass
class BasicAttSeq2SeqRNNTemp(BaseTemplate):
    in_features: int
    hidden_size: int
    enc_num_layers: int
    bidirectional: bool
    dec_num_layers: int
    emb_dim: int
    p_dropout: float
    teacher_forcing_rate: float = 0.0
    rnn_type: str = 'rnn'
    _name = 'basic_att_rnn'
    _type = 'seq2seq'


@dataclass
class RNNWithLocationAwareAttTemp(BaseTemplate):
    in_features: int
    hidden_size: int
    enc_num_layers: int
    bidirectional: bool
    dec_num_layers: int
    emb_dim: int
    kernel_size: int
    activation: str
    p_dropout: float
    inv_temperature: Union[float, int] = 1
    teacher_forcing_rate: float = 0.0
    rnn_type: str = 'rnn'
    _name = 'rnn_with_location_att'
    _type = 'seq2seq'


@dataclass
class SpeechTransformerTemp(BaseTemplate):
    in_features: int
    n_conv_layers: int
    kernel_size: int
    stride: int
    d_model: int
    n_enc_layers: int
    n_dec_layers: int
    ff_size: int
    h: int
    att_kernel_size: int
    att_out_channels: int
    masking_value: int = -1e15
    _name = 'speech_transformer'
    _type = 'seq2seq'


@dataclass
class QuartzNetTemp(BaseTemplate):
    in_features: int
    num_blocks: int
    block_repetition: int
    num_sub_blocks: int
    channels_size: List[int]
    epilog_kernel_size: int
    epilog_channel_size: Tuple[int, int]
    prelog_kernel_size: int
    prelog_stride: int
    prelog_n_channels: int
    groups: int
    blocks_kernel_size: Union[int, List[int]]
    p_dropout: float
    _name = 'quartz_net'
    _type = 'ctc'


@dataclass
class SqueezeformerCTCTemp(BaseTemplate):
    in_features: int
    n: int
    d_model: int
    ff_expansion_factor: int
    h: int
    kernel_size: int
    pooling_kernel_size: int
    pooling_stride: int
    ss_kernel_size: Union[int, List[int]]
    ss_stride: Union[int, List[int]]
    ss_n_conv_layers: int
    p_dropout: float
    ss_groups: Union[int, List[int]] = 1
    masking_value: int = -1e15
    _name = 'squeezeformer'
    _type = 'ctc'


@dataclass
class RNNTTemp(BaseTemplate):
    in_features: int
    emb_dim: int
    n_layers: int
    hidden_size: int
    bidirectional: bool
    rnn_type: str
    p_dropout: float
    _name = 'rnn-t'
    _type = 'transducer'
