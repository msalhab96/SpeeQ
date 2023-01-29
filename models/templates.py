from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple, Union

from constants import CTC_TYPE, SEQ2SEQ_TYPE, TRANSDUCER_TYPE
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
    """DeepSpeech 1 model template
    https://arxiv.org/abs/1412.5567

    Args:
        in_features (int): The input feature size.
        hidden_size (int): The layers' hidden size.
        n_linear_layers (int): The number of feed-forward layers.
        bidirectional (bool): if the rnn is bidirectional or not.
        max_clip_value (int): The maximum relu value.
        p_dropout (float): The dropout rate.
        rnn_type (str): rnn, gru or lstm. Default 'rnn'.
    """
    in_features: int
    hidden_size: int
    n_linear_layers: int
    bidirectional: bool
    max_clip_value: int
    p_dropout: float
    rnn_type: str = 'rnn'
    _name = 'deep_speech_v1'
    _type = CTC_TYPE


@dataclass
class BERTTemp(BaseTemplate):
    """BERT model template
    https://arxiv.org/abs/1810.04805

    Args:
        max_len (int): The maximum length for positional encoding.
        in_feature (int): The input/speech feature size.
        d_model (int): The model dimensionality.
        h (int): The number of heads.
        hidden_size (int): The inner size of the feed forward module.
        n_layers (int): The number of transformer encoders.
        p_dropout (float): The dropout rate.
    """
    max_len: int
    in_feature: int
    d_model: int
    h: int
    hidden_size: int
    n_layers: int
    p_dropout: float
    _name = 'bert'
    _type = CTC_TYPE


@dataclass
class DeepSpeechV2Temp(BaseTemplate):
    """deep speech 2 model template
    https://arxiv.org/abs/1512.02595

    Args:
        n_conv (int): The number of convolution layers.
        kernel_size (int): The convolution layers' kernel size.
        stride (int): The convolution layers' stride.
        in_features (int): The input/speech feature size.
        hidden_size (int): The layers' hidden size.
        bidirectional (bool): if the rnn is bidirectional or not.
        n_rnn (int): The number of RNN layers.
        n_linear_layers (int): The number of linear layers.
        max_clip_value (int): The maximum relu value.
        tau (int): The future context size.
        p_dropout (float): The dropout rate.
        rnn_type (str): rnn, gru or lstm. Default 'rnn'.
    """
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
    _type = CTC_TYPE


@dataclass
class ConformerCTCTemp(BaseTemplate):
    """ConformerCTC model template
    https://arxiv.org/abs/2005.08100

    Args:
        d_model (int): The model dimension.
        n_conf_layers (int): The number of conformer blocks.
        ff_expansion_factor (int): The feed-forward expansion factor.
        h (int): The number of heads.
        kernel_size (int): The kernel size of conv module.
        ss_kernel_size (int): The kernel size of the subsampling layer.
        ss_stride (int): The stride of the subsampling layer.
        ss_num_conv_layers (int): The number of subsampling layer.
        in_features (int): The input/speech feature size.
        res_scaling (float): The residual connection multiplier.
        p_dropout (float): The dropout rate.
    """
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
    _type = CTC_TYPE


@dataclass
class JasperTemp(BaseTemplate):
    """Jasper model template
    https://arxiv.org/abs/1904.03288

    Args:
        in_features (int): The input/speech feature size.
        num_blocks (int): The number of jasper blocks, denoted
            as 'B' in the paper.
        num_sub_blocks (int): The number of jasper subblocks, denoted
            as 'R' in the paper.
        channel_inc (int): The rate to increase the number of channels
            across the blocks.
        epilog_kernel_size (int): The epilog block convolution's kernel size.
        prelog_kernel_size (int): The prelog block convolution's kernel size.
        prelog_stride (int): The prelog block convolution's stride.
        prelog_n_channels (int): The prelog block convolution's number of
            output channnels.
        blocks_kernel_size (Union[int, List[int]]): The convolution layer's
            kernel size of each jasper block.
        p_dropout (float): The dropout rate.
    """
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
    _type = CTC_TYPE


@dataclass
class Wav2LetterTemp(BaseTemplate):
    """Wav2Letter model template
    https://arxiv.org/abs/1609.03193

    Args:
        in_features (int): The input/speech feature size.
        n_conv_layers (int): The number of convolution layers.
        layers_kernel_size (int): The convolution layers' kernel size.
        layers_channels_size (int): The convolution layers' channel size.
        pre_conv_stride (int): The prenet convolution stride.
        pre_conv_kernel_size (int): The prenet convolution kernel size.
        post_conv_channels_size (int): The postnet convolution channel size.
        post_conv_kernel_size (int): The postnet convolution kernel size.
        p_dropout (float): The dropout rate.
        wav_kernel_size (Optional[int]): The kernel size of the first
            layer that process the wav samples directly if wav is modeled.
            Default None.
        wav_stride (Optional[int]): The stride size of the first
            layer that process the wav samples directly if wav is modeled.
            Default None.
    """
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
    _type = CTC_TYPE


@dataclass
class LASTemp(BaseTemplate):
    """Listen, Attend and Spell model template
    https://arxiv.org/abs/1508.01211

    Args:
        in_features (int): The encoder's input feature speech size.
        hidden_size (int): The RNNs' hidden size.
        enc_num_layers (int): The number of the encoder's layers.
        reduction_factor (int): The time resolution reduction factor.
        bidirectional (bool): If the encoder's RNNs are bidirectional or not.
        dec_num_layers (int): The number of the decoders' RNN layers.
        emb_dim (int): The embedding size.
        p_dropout (float): The dropout rate.
        teacher_forcing_rate (float): The teacher forcing rate. Default 0.0
        rnn_type (str): The rnn type. Default 'rnn'.
    """
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
    _type = SEQ2SEQ_TYPE


@dataclass
class BasicAttSeq2SeqRNNTemp(BaseTemplate):
    """Basic RNN encoder decoder model template.

    Args:
        in_features (int): The encoder's input feature speech size.
        hidden_size (int): The RNNs' hidden size.
        enc_num_layers (int): The number of the encoder's layers.
        bidirectional (bool): If the encoder's RNNs are bidirectional or not.
        dec_num_layers (int): The number of the decoders' RNN layers.
        emb_dim (int): The embedding size.
        p_dropout (float): The dropout rate.
        teacher_forcing_rate (float): The teacher forcing rate. Default 0.0
        rnn_type (str): The rnn type. Default 'rnn'.
    """
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
    _type = SEQ2SEQ_TYPE


@dataclass
class RNNWithLocationAwareAttTemp(BaseTemplate):
    """RNN seq2seq with location aware attention model tempalte
        in https://arxiv.org/abs/1506.07503

    Args:
        in_features (int): The encoder's input feature speech size.
        hidden_size (int): The RNNs' hidden size.
        enc_num_layers (int): The number of the encoder's layers.
        bidirectional (bool): If the encoder's RNNs are bidirectional or not.
        dec_num_layers (int): The number of the decoders' RNN layers.
        emb_dim (int): The embedding size.
        kernel_size (int): The attention kernel size.
        activation (str): The activation function to use in the
            attention layer. it can be either softmax or sigmax.
        p_dropout (float): The dropout rate.
        inv_temperature (Union[float, int]): The inverse temperature value of
            the attention. Default 1.
        teacher_forcing_rate (float): The teacher forcing rate. Default 0.0
        rnn_type (str): The rnn type. default 'rnn'.
    """
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
    _type = SEQ2SEQ_TYPE


@dataclass
class SpeechTransformerTemp(BaseTemplate):
    """Speech Transformer model template
    https://ieeexplore.ieee.org/document/8462506

    Args:
        in_features (int): The input/speech feature size.
        n_conv_layers (int): The number of down-sampling convolutional layers.
        kernel_size (int): The down-sampling convolutional layers kernel size.
        stride (int): The down-sampling convolutional layers stride.
        d_model (int): The model dimensionality.
        n_enc_layers (int): The number of encoder layers.
        n_dec_layers (int): The number of decoder layers.
        ff_size (int): The feed-forward inner layer dimensionality.
        h (int): The number of attention heads.
        att_kernel_size (int): The attentional convolutional
            layers' kernel size.
        att_out_channels (int): The number of output channels of the
            attentional convolution
        masking_value (int): The attentin masking value. Default -1e15
    """
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
    _type = SEQ2SEQ_TYPE


@dataclass
class QuartzNetTemp(BaseTemplate):
    """QuartzNet model template
    https://arxiv.org/abs/1910.10261

    Args:
        in_features (int): The input/speech feature size.
        num_blocks (int): The number of QuartzNet blocks, denoted
            as 'B' in the paper.
        block_repetition (int): The nubmer of times to repeat each block.
            denoted as S in the paper.
        num_sub_blocks (int): The number of QuartzNet subblocks, denoted
            as 'R' in the paper.
        channels_size (List[int]): The channel size of each block. it has to
            be of length equal to num_blocks
        epilog_kernel_size (int): The epilog block convolution's kernel size.
        epilog_channel_size (Tuple[int, int]): The epilog blocks channels size.
        prelog_kernel_size (int): The prelog block convolution's kernel size.
        prelog_stride (int): The prelog block convolution's stride.
        prelog_n_channels (int): The prelog block convolution's number of
            output channnels.
        groups (int): The groups size.
        blocks_kernel_size (Union[int, List[int]]): The convolution layer's
            kernel size of each jasper block.
        p_dropout (float): The dropout rate.
    """
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
    _type = CTC_TYPE


@dataclass
class SqueezeformerCTCTemp(BaseTemplate):
    """Squeezeformer model template
    https://arxiv.org/abs/2206.00888

    Args:
        in_features (int): The input/speech feature size.
        n (int): The number of layers per block, denoted as N in the paper.
        d_model (int): The model dimension.
        ff_expansion_factor (int): The linear layer's expansion factor.
        h (int): The number of heads.
        kernel_size (int): The depth-wise convolution kernel size.
        pooling_kernel_size (int): The pooling convolution kernel size.
        pooling_stride (int): The pooling convolution stride size.
        ss_kernel_size (Union[int, List[int]]): The kernel size of the
            subsampling layer.
        ss_stride (Union[int, List[int]]): The stride of the subsampling layer.
        ss_n_conv_layers (int): The number of subsampling convolutional layers.
        p_dropout (float): The dropout rate.
        ss_groups (Union[int, List[int]]): The subsampling convolution groups
            size.
        masking_value (int): The masking value. Default -1e15
    """
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
    _type = CTC_TYPE


@dataclass
class RNNTTemp(BaseTemplate):
    """RNN transducer model template
    https://arxiv.org/abs/1211.3711

    Args:
        in_features (int): The input feature size.
        emb_dim (int): The embedding layer's size.
        n_layers (int): The number of the encoder's RNN layers.
        hidden_size (int): The RNN's hidden size.
        bidirectional (bool): If the RNN is bidirectional or not.
        rnn_type (str): The RNN type.
        p_dropout (float): The dropout rate.
    """
    in_features: int
    emb_dim: int
    n_layers: int
    hidden_size: int
    bidirectional: bool
    rnn_type: str
    p_dropout: float
    _name = 'rnn-t'
    _type = TRANSDUCER_TYPE


@dataclass
class ConformerTransducerTemp(BaseTemplate):
    """Conformer transducer model template
    https://arxiv.org/abs/2005.08100

    Args:
        d_model (int): The model dimension.
        n_conf_layers (int): The number of conformer blocks.
        ff_expansion_factor (int): The feed-forward expansion factor.
        h (int): The number of heads.
        kernel_size (int): The kernel size of conv module.
        ss_kernel_size (int): The kernel size of the subsampling layer.
        ss_stride (int): The stride of the subsampling layer.
        ss_num_conv_layers (int): The number of subsampling layers.
        in_features (int): The input/speech feature size.
        res_scaling (float): The residual connection multiplier.
        emb_dim (int): The embedding layer's size.
        rnn_type (str): The RNN type.
        p_dropout (float): The dropout rate.
    """
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
    emb_dim: int
    rnn_type: str
    p_dropout: float
    _name = 'conformer'
    _type = TRANSDUCER_TYPE


@dataclass
class ContextNetTemp(BaseTemplate):
    """ContextNet transducer model template
    https://arxiv.org/abs/2005.03191

    Args:
        in_features (int): The input feature size.
        emb_dim (int): The embedding layer's size.
        n_layers (int): The number of ContextNet blocks.
        n_sub_layers (Union[int, List[int]]): The number of convolutional
            layers per block, if list is passed, it has to be of length equal
            to n_layers.
        stride (Union[int, List[int]]): The stride of the last convolutional
            layers per block, if list is passed, it has to be of length equal
            to n_layers.
        out_channels (Union[int, List[int]]): The channels size of the
            convolutional layers per block, if list is passed, it has to be of
            length equal to n_layers.
        kernel_size (int): The convolutional layers kernel size.
        reduction_factor (int): The feature reduction size of the
            Squeeze-and-excitation module.
        rnn_type (str): The RNN type.
    """
    in_features: int
    emb_dim: int
    n_layers: int
    n_sub_layers: Union[int, List[int]]
    stride: Union[int, List[int]]
    out_channels: Union[int, List[int]]
    kernel_size: int
    reduction_factor: int
    rnn_type: str
    _name = 'context_net'
    _type = TRANSDUCER_TYPE
