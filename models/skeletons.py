from typing import Union

from torch.nn import Module

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
            as an activation function, and the predictions of shape [T, B, C]
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
