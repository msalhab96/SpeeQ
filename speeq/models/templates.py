"""This file contains templates for various pre-implemented models. Each template is a model configuration for a specific pre-implemented model in the framework.

Classes:

- BaseTemplate: Base template that defines common configuration parameters for all models.
- DeepSpeechV1Temp: Template for configuring DeepSpeechV1 model.
- BERTTemp: Template for configuring BERT model.
- DeepSpeechV2Temp: Template for configuring DeepSpeechV2 model.
- ConformerCTCTemp: Template for configuring Conformer CTC model.
- JasperTemp: Template for configuring Jasper model.
- Wav2LetterTemp: Template for configuring Wav2Letter model.
- LASTemp: Template for configuring LAS model.
- BasicAttSeq2SeqRNNTemp: Template for configuring Basic Attention Seq2Seq RNN model.
- RNNWithLocationAwareAttTemp: Template for configuring RNN with Location-Aware Attention model.
- SpeechTransformerTemp: Template for configuring Speech Transformer model.
- QuartzNetTemp: Template for configuring QuartzNet model.
- SqueezeformerCTCTemp: Template for configuring Squeezeformer CTC model.
- RNNTTemp: Template for configuring RNNT model.
- ConformerTransducerTemp: Template for configuring Conformer Transducer model.
- ContextNetTemp: Template for configuring ContextNet model.


Builder:

The below templates can be used to build custome model:

- CTCModelBuilderTemp: Template for building CTC models.
- TransducerBuilderTemp: Template for building Transducer models.
- Seq2SeqBuilderTemp: Template for building Seq2Seq models.



"""
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple, Union

from torch.nn import Module

from speeq.constants import CTC_TYPE, MODEL_BUILDER_TYPE, SEQ2SEQ_TYPE, TRANSDUCER_TYPE
from speeq.interfaces import ITemplate


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

        hidden_size (int): The hidden size of the rnn layers.

        n_linear_layers (int): The number of feed-forward layers.

        bidirectional (bool): A flag indicating if the rnn is bidirectional or not.

        max_clip_value (int): The maximum relu clipping value.

        p_dropout (float): The dropout rate.

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.
    """

    in_features: int
    hidden_size: int
    n_linear_layers: int
    bidirectional: bool
    max_clip_value: int
    p_dropout: float
    rnn_type: str = "rnn"
    _name = "deep_speech_v1"
    _type = CTC_TYPE


@dataclass
class BERTTemp(BaseTemplate):
    """BERT model template
    https://arxiv.org/abs/1810.04805

    Args:
        max_len (int): The maximum length for positional encoding.

        in_features (int): The input/speech feature size.

        d_model (int): The model dimensionality.

        h (int): The number of attention heads.

        ff_size (int): The inner size of the feed forward module.

        n_layers (int): The number of transformer encoders.

        p_dropout (float): The dropout rate.
    """

    max_len: int
    in_features: int
    d_model: int
    h: int
    ff_size: int
    n_layers: int
    p_dropout: float
    _name = "bert"
    _type = CTC_TYPE


@dataclass
class DeepSpeechV2Temp(BaseTemplate):
    """deep speech 2 model template
    https://arxiv.org/abs/1512.02595

    Args:

        n_conv (int): The number of convolution layers.

        kernel_size (int): The kernel size of the convolution layers.

        stride (int): The stride size of the convolution layer.

        in_features (int): The input/speech feature size.

        hidden_size (int): The hidden size of the RNN layers.

        bidirectional (bool): A flag indicating if the rnn is bidirectional or not.

        n_rnn (int): The number of RNN layers.

        n_linear_layers (int): The number of linear layers.

        max_clip_value (int): The maximum relu clipping value.

        tau (int): The future context size.

        p_dropout (float): The dropout rate.

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.
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
    rnn_type: str = "rnn"
    _name = "deep_speech_v2"
    _type = CTC_TYPE


@dataclass
class ConformerCTCTemp(BaseTemplate):
    """ConformerCTC model template
    https://arxiv.org/abs/2005.08100

    Args:

        d_model (int): The model dimension.

        n_conf_layers (int): The number of conformer blocks.

        ff_expansion_factor (int): The feed-forward expansion factor.

        h (int): The number of attention heads.

        kernel_size (int): The convolution module kernel size.

        ss_kernel_size (int): The subsampling layer kernel size.

        ss_stride (int): The subsampling layer stride size.

        ss_num_conv_layers (int): The number of subsampling convolutional layers.

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
    _name = "conformer"
    _type = CTC_TYPE


@dataclass
class JasperTemp(BaseTemplate):
    """Jasper model template
    https://arxiv.org/abs/1904.03288

    Args:

        in_features (int): The input/speech feature size.

        num_blocks (int): The number of Jasper blocks (denoted as 'B' in the paper).

        num_sub_blocks (int): The number of Jasper subblocks (denoted as 'R' in the paper).

        channel_inc (int): The rate to increase the number of channels across the blocks.

        epilog_kernel_size (int): The kernel size of the epilog block convolution layer.

        prelog_kernel_size (int): The kernel size of the prelog block ocnvolution layer.

        prelog_stride (int): The stride size of the prelog block convolution layer.

        prelog_n_channels (int): The output channnels of the prelog block convolution layer.

        blocks_kernel_size (Union[int, List[int]]): The kernel size(s) of the convolution layer for each block.

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
    _name = "jasper"
    _type = CTC_TYPE


@dataclass
class Wav2LetterTemp(BaseTemplate):
    """Wav2Letter model template
    https://arxiv.org/abs/1609.03193

    Args:

        in_features (int): The input/speech feature size.

        n_conv_layers (int): The number of convolution layers.

        layers_kernel_size (int): The kernel size of the convolution layers.

        layers_channels_size (int): The number of output channels of each convolution layer.

        pre_conv_stride (int): The stride of the prenet convolution layer.

        pre_conv_kernel_size (int): The kernel size of the prenet convolution layer.

        post_conv_channels_size (int): The number of output channels of the
        postnet convolution layer.

        post_conv_kernel_size (int): The kernel size of the postnet convolution layer.

        p_dropout (float): The dropout rate.

        wav_kernel_size (Optional[int]): The kernel size of the first layer that
        processes the wav samples directly if wav is modeled. Default None.

        wav_stride (Optional[int]): The stride size of the first layer that
        processes the wav samples directly if wav is modeled. Default None.

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
    _name = "wav2letter"
    _type = CTC_TYPE


@dataclass
class LASTemp(BaseTemplate):
    """Listen, Attend and Spell model template
    https://arxiv.org/abs/1508.01211

    Args:

        in_features (int): The encoder's input feature speech size.

        hidden_size (int): The hidden size of the RNN layers.

        enc_num_layers (int): The number of layers in the encoder.

        reduction_factor (int): The time resolution reduction factor.

        bidirectional (bool): A flag indicating if the rnn is bidirectional or not.

        dec_num_layers (int): The number of the RNN layers in the decoder.

        emb_dim (int): The embedding size.

        p_dropout (float): The dropout rate.

        pred_activation (Module): An instance of an activation function to be
        applied on the last dimension of the predicted logits..

        teacher_forcing_rate (float): The teacher forcing rate. Default 0.0

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.
        Default 'rnn'.
    """

    in_features: int
    hidden_size: int
    enc_num_layers: int
    reduction_factor: int
    bidirectional: bool
    dec_num_layers: int
    emb_dim: int
    p_dropout: float
    pred_activation: Module
    teacher_forcing_rate: float = 0.0
    rnn_type: str = "rnn"
    _name = "las"
    _type = SEQ2SEQ_TYPE


@dataclass
class BasicAttSeq2SeqRNNTemp(BaseTemplate):
    """Basic RNN encoder decoder model template.

    Args:

        in_features (int): The encoder's input feature speech size.

        hidden_size (int): The hidden size of the RNN layers.

        enc_num_layers (int): The number of layers in the encoder.

        bidirectional (bool): A flag indicating if the rnn is bidirectional or not.

        dec_num_layers (int): The number of the RNN layers in the decoder.

        emb_dim (int): The embedding size.

        p_dropout (float): The dropout rate.

        pred_activation (Module): An instance of an activation function.

        teacher_forcing_rate (float): The teacher forcing rate. Default 0.0

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.
        Default 'rnn'.

    """

    in_features: int
    hidden_size: int
    enc_num_layers: int
    bidirectional: bool
    dec_num_layers: int
    emb_dim: int
    p_dropout: float
    pred_activation: Module
    teacher_forcing_rate: float = 0.0
    rnn_type: str = "rnn"
    _name = "basic_att_rnn"
    _type = SEQ2SEQ_TYPE


@dataclass
class RNNWithLocationAwareAttTemp(BaseTemplate):
    """RNN seq2seq with location aware attention model tempalte
        in https://arxiv.org/abs/1506.07503

    Args:

        in_features (int): The encoder's input feature speech size.

        hidden_size (int): The hidden size of the RNN layers.

        enc_num_layers (int): The number of layers in the encoder.

        bidirectional (bool): A flag indicating if the rnn is bidirectional or not.

        dec_num_layers (int): The number of the RNN layers in the decoder.

        emb_dim (int): The embedding size.

        kernel_size (int): The attention kernel size.

        activation (str): The activation function to use in the attention layer.
        it can be either softmax or sigmax.

        p_dropout (float): The dropout rate.

        pred_activation (Module): An instance of an activation function to be
        applied on the last dimension of the predicted logits..

        inv_temperature (Union[float, int]): The inverse temperature value. Default 1.

        teacher_forcing_rate (float): The teacher forcing rate. Default 0.0

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.
        Default 'rnn'.
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
    pred_activation: Module
    inv_temperature: Union[float, int] = 1
    teacher_forcing_rate: float = 0.0
    rnn_type: str = "rnn"
    _name = "rnn_with_location_att"
    _type = SEQ2SEQ_TYPE


@dataclass
class SpeechTransformerTemp(BaseTemplate):
    """Speech Transformer model template
    https://ieeexplore.ieee.org/document/8462506

    Args:

        in_features (int): The input/speech feature size.

        n_conv_layers (int): The number of down-sampling convolutional layers.

        kernel_size (int): The kernel size of the down-sampling convolutional layers.

        stride (int): The stride size of the down-sampling convolutional layers.

        d_model (int): The model dimensionality.

        n_enc_layers (int): The number of encoder layers.

        n_dec_layers (int): The number of decoder layers.

        ff_size (int):  The dimensionality of the inner layer of the feed-forward module.

        h (int): The number of attention heads.

        att_kernel_size (int): The kernel size of the attentional convolutional layers.

        att_out_channels (int): The number of output channels of the attentional convolution layers.

        pred_activation (Module): An activation function instance to be applied on
        the last dimension of the predicted logits.

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
    pred_activation: Module
    masking_value: int = -1e15
    _name = "speech_transformer"
    _type = SEQ2SEQ_TYPE


@dataclass
class QuartzNetTemp(BaseTemplate):
    """QuartzNet model template
    https://arxiv.org/abs/1910.10261

    Args:

        in_features (int): The input/speech feature size.

        num_blocks (int): The number of QuartzNet blocks (denoted as 'B' in the paper).

        block_repetition (int): The number of times to repeat each block (denoted as 'S' in the paper).

        num_sub_blocks (int): The number of QuartzNet subblocks, (denoted as 'R' in the paper).

        channels_size (List[int]): A list of integers representing the number of output channels
        for each block.

        epilog_kernel_size (int): The kernel size of the convolution layer in the epilog block.

        epilog_channel_size (Tuple[int, int]): A tuple for both epilog layers
        of the convolution layer .

        prelog_kernel_size (int): The kernel size pf the convolution layer in the prelog block.

        prelog_stride (int): The stride size of the of the convoltuional layer
        in the prelog block.

        prelog_n_channels (int): The number of output channels of the convolutional
        layer in the prelog block.

        groups (int): The groups size.

        blocks_kernel_size (Union[int, List[int]]): An integer or a list of integers representing the
        kernel size(s) for each block's convolutional layer.

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
    _name = "quartz_net"
    _type = CTC_TYPE


@dataclass
class SqueezeformerCTCTemp(BaseTemplate):
    """Squeezeformer model template
    https://arxiv.org/abs/2206.00888

    Args:

        in_features (int): The input/speech feature size.

        n (int): The number of layers per block, (denoted as N in the paper).

        d_model (int): The model dimension.

        ff_expansion_factor (int): The expansion factor of linear layer in the
        feed forward module.

        h (int): The number of attention heads.

        kernel_size (int): The kernel size of the depth-wise convolution layer.

        pooling_kernel_size (int): The kernel size of the pooling convolution layer.

        pooling_stride (int): The stride size of the pooling convolution layer.

        ss_kernel_size (Union[int, List[int]]): The kernel size of the subsampling layer(s).

        ss_stride (Union[int, List[int]]): The stride of the subsampling layer(s).

        ss_n_conv_layers (int): The number of subsampling convolutional layers.

        p_dropout (float): The dropout rate.

        ss_groups (Union[int, List[int]]): The subsampling convolution groups size(s).

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
    _name = "squeezeformer"
    _type = CTC_TYPE


@dataclass
class RNNTTemp(BaseTemplate):
    """RNN transducer model template
    https://arxiv.org/abs/1211.3711

    Args:

        in_features (int): The input feature size.

        emb_dim (int): The embedding layer's size.

        n_layers (int): The number of the RNN layers in the encoder.

        n_dec_layers (int): The number of RNNs in the decoder (predictor).

        hidden_size (int): The hidden size of the RNN layers.

        bidirectional (bool): A flag indicating if the rnn is bidirectional or not.

        rnn_type (str): The RNN type.

        p_dropout (float): The dropout rate.
    """

    in_features: int
    emb_dim: int
    n_layers: int
    n_dec_layers: int
    hidden_size: int
    bidirectional: bool
    rnn_type: str
    p_dropout: float
    _name = "rnn-t"
    _type = TRANSDUCER_TYPE


@dataclass
class ConformerTransducerTemp(BaseTemplate):
    """Conformer transducer model template
    https://arxiv.org/abs/2005.08100

    Args:

        d_model (int): The model dimension.

        n_conf_layers (int): The number of conformer blocks.

        n_dec_layers (int): The number of RNNs in the decoder (predictor).

        ff_expansion_factor (int): The feed-forward expansion factor.

        h (int): The number of attention heads.

        kernel_size (int): The convolution module kernel size.

        ss_kernel_size (int): The subsampling layer kernel size.

        ss_stride (int): The subsampling layer stride size.

        ss_num_conv_layers (int): The number of subsampling convolutional layers.

        in_features (int): The input/speech feature size.

        res_scaling (float): The residual connection multiplier.

        emb_dim (int): The embedding layer's size.

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.

        p_dropout (float): The dropout rate.
    """

    d_model: int
    n_conf_layers: int
    n_dec_layers: int
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
    _name = "conformer"
    _type = TRANSDUCER_TYPE


@dataclass
class ContextNetTemp(BaseTemplate):
    """ContextNet transducer model template
    https://arxiv.org/abs/2005.03191

    Args:

        in_features (int): The input feature size.

        emb_dim (int): The embedding layer's size.

        n_layers (int): The number of ContextNet blocks.

        n_dec_layers (int): The number of RNNs in the decoder (predictor).

        n_sub_layers (Union[int, List[int]]): The number of convolutional
        layers per block. If list is passed, it has to be of length equal to `n_layers`.

        stride (Union[int, List[int]]): The stride of the last convolutional
        layers per block. If list is passed, it has to be of length equal to
        `n_layers`.

        out_channels (Union[int, List[int]]): The channels size of the
        convolutional layers per block. If list is passed, it has to be of
        length equal to `n_layers`.

        kernel_size (int): The convolutional layers kernel size.

        reduction_factor (int): The feature reduction size of the Squeeze-and-excitation module.

        rnn_type (str): The RNN type it has to be one of rnn, gru or lstm.

    """

    in_features: int
    emb_dim: int
    n_layers: int
    n_dec_layers: int
    n_sub_layers: Union[int, List[int]]
    stride: Union[int, List[int]]
    out_channels: Union[int, List[int]]
    kernel_size: int
    reduction_factor: int
    rnn_type: str
    _name = "context_net"
    _type = TRANSDUCER_TYPE


@dataclass
class CTCModelBuilderTemp(BaseTemplate):
    """CTC-based model builder template

    Args:
        encoder (Module): The speech encoder (acoustic model), such that
        the forward of the encoder returns a tuple of the encoded speech
        tensor and a length tensor of the encoded speech.

        has_bnorm (bool): A flag indicates whether the encoder or the decoder
        has batch normalization.

        pred_net (Union[Module, None]): The prediction network. if provided
        the forward of the prediction network expected to have log softmax
        as an activation function, and the predictions of shape [T, B, C]
        where T is the sequence length, B the batch size, and C the number
        of classes. Default None.

        feat_size (Union[Module, None]): Used if pred_net parameter is not None
        where it's the encoder's output feature size. Default None.
    """

    encoder: Module
    has_bnorm: bool
    pred_net: Union[Module, None] = None
    feat_size: Union[int, None] = None
    _name = CTC_TYPE
    _type = MODEL_BUILDER_TYPE


@dataclass
class TransducerBuilderTemp(BaseTemplate):
    """Transducer-based model builder template

    Args:
        encoder (Module): The speech encoder (acoustic model), such that
        the forward method of the encoder returns a tuple of the encoded
        speech tensor and a length tensor for the encoded speech.

        decoder (Module): The text decoder such that
        the forward method of the decoder returns a tuple of the encoded
        text tensor and a length tensor for the encoded text.

        has_bnorm (bool): A flag indicates whether the encoder, the decoder, or
        the join network has batch normalization.

        join_net (Union[Module, None]): The join network. if provided
        the forward of the join network expected to have no activation
        function, and the results of shape [B, Ts, Tt, C], where B the
        batch size, Ts is the speech sequence length, Tt is the text
        sequence length, and C the number of classes. Default None.

        feat_size (Union[Module, None]): Used if join_net parameter is not None
        where it's the encoder and the decoder's output feature size.
        Default None.
    """

    encoder: Module
    decoder: Module
    has_bnorm: bool
    join_net: Union[Module, None] = None
    feat_size: Union[None, int] = None
    _name = TRANSDUCER_TYPE
    _type = MODEL_BUILDER_TYPE


@dataclass
class Seq2SeqBuilderTemp(BaseTemplate):
    """Seq2Seq-based model builder template

    Args:
        encoder (Module): The speech encoder (acoustic model), such that
        the forward method of the encoder returns a tuple of the encoded
        speech tensor, the last encoder hidden state tensor/tuple if there
        is any, and a length tensor for the encoded speech.

        decoder (Module): The text decoder such that
        the forward method of the decoder takes the encoder's output, the
        last encoder's hidden state (if there is any), the encoder mask,
        the decoder input, and the decoder mask and returns the prediction
        tensor.

        has_bnorm (bool): A flag indicates whether the encoder, the decoder
        has batch normalization.
    """

    encoder: Module
    decoder: Module
    has_bnorm: bool
    _name = SEQ2SEQ_TYPE
    _type = MODEL_BUILDER_TYPE


class VGGTransformerTransducerTemp(BaseTemplate):
    """VGG Transformer transducer model template
    https://arxiv.org/abs/1910.12977

    Args:

        in_features (int): The input feature size.

        emb_dim (int): The embedding layer's size.

        n_layers (int): The number of transformer encoder layers with truncated
        self attention.

        n_dec_layers (int): The number of RNNs in the decoder (predictor).

        rnn_type (str): The RNN type.

        n_vgg_blocks (int): The number of VGG blocks to use.

        n_conv_layers_per_vgg_block (List[int]): A list of integers that specifies the number
        of convolution layers in each block.

        kernel_sizes_per_vgg_block (List[List[int]]): A list of lists that contains the
        kernel size for each layer in each block. The length of the outer list
        should match `n_vgg_blocks`, and each inner list should be the same length
        as the corresponding block's number of layers.

        n_channels_per_vgg_block (List[List[int]]): A list of lists that contains the
        number of channels for each convolution layer in each block. This argument
        should also have length equal to `n_vgg_blocks`, and each sublist should
        have length equal to the number of layers in the corresponding block.

        vgg_pooling_kernel_size (List[int]): A list of integers that specifies the size
        of the max pooling layer in each block. The length of this list should be
        equal to `n_vgg_blocks`.

        d_model (int): The model dimensionality.

        ff_size (int): The feed forward inner layer dimensionality.

        h (int): The number of heads in the attention mechanism.

        joint_size (int): The joint layer feature size (denoted as do in the paper).

        left_size (int): The size of the left window that each time step is
        allowed to look at.

        right_size (int): The size of the right window that each time step is
        allowed to look at.

        p_dropout (float): The dropout rate.

        masking_value (float, optional): The value to use for masking padded
        elements. Defaults to -1e15.
    """

    in_features: int
    emb_dim: int
    n_layers: int
    n_dec_layers: int
    rnn_type: str
    n_vgg_blocks: int
    n_conv_layers_per_vgg_block: List[int]
    kernel_sizes_per_vgg_block: List[List[int]]
    n_channels_per_vgg_block: List[List[int]]
    vgg_pooling_kernel_size: List[int]
    d_model: int
    ff_size: int
    h: int
    joint_size: int
    left_size: int
    right_size: int
    p_dropout: float
    masking_value: int = -1e15
    _name = "vgg_transformer"
    _type = TRANSDUCER_TYPE
