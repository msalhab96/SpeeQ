from typing import Union

from torch.nn import Module

from models.transducers import BaseTransducer

from .ctc import CTCModel


class CTCSkeleton(CTCModel):
    """Builds the CTC-based model skeleton

    Args:
        encoder (Module): The speech encoder (acoustic model), such that
            the forward of the encoder returns a tuple of the encoded speech
            tensor and a length tensor for the encoded speech.
        has_bnorm (bool): A flag indicates whether the encoder or the decoder
            has batch normalization.
        pred_net (Union[Module, None]): The prediction network. if provided
            the forward of the prediction network expected to have log softmax
            as an activation function, and the predictions of shape [B, T, C]
            where T is the sequence length, B the batch size, and C the number
            of classes. Default None.
        feat_size (Union[Module, None]): Used if pred_net parameter is not None
            where it's the encoder's output feature size. Default None.
        n_classes (Union[Module, None]): Used if pred_net parameter is not None
            where it's the number of the classes/characters to be predicted.
    """

    def __init__(
            self,
            encoder: Module,
            has_bnorm: bool,
            pred_net: Union[Module, None] = None,
            feat_size: Union[int, None] = None,
            n_classes: Union[int, None] = None
    ) -> None:
        assert (feat_size is None) ^ (pred_net is None)
        if feat_size is not None:
            assert n_classes is not None
        args = [1, 1]
        if feat_size is not None:
            args[0] = feat_size
        if n_classes is not None:
            args[1] = n_classes
        super().__init__(*args)
        self.has_bnorm = has_bnorm
        self.encoder = encoder
        if pred_net is not None:
            self.pred_net = pred_net


class TransducerSkeleton(BaseTransducer):
    """Builds the Transducer-based model skeleton

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
        n_classes (Union[Module, None]): Used if join_net parameter is not None
            where it's the number of the classes/characters to be predicted.
    """

    def __init__(
            self,
            encoder: Module,
            decoder: Module,
            has_bnorm: bool,
            join_net: Union[Module, None] = None,
            feat_size: Union[int, None] = None,
            n_classes: Union[int, None] = None
    ) -> None:
        assert (feat_size is None) ^ (join_net is None)
        if feat_size is not None:
            assert n_classes is not None
        args = [1, 1]
        if feat_size is not None:
            args[0] = feat_size
        if n_classes is not None:
            args[1] = n_classes
        super().__init__(*args)
        self.has_bnorm = has_bnorm
        self.encoder = encoder
        self.deocder = decoder
        if join_net is not None:
            self.join_net = join_net
