"""Builds the skeleton for CTC, seq2seq, and transducer models, this is used for
building custom models where the user has the ability to combine or build
custom encoder, or decoder.
"""
from typing import Union

from torch import Tensor
from torch.nn import Module

from speeq.utils.utils import get_mask_from_lens

from .ctc import CTCModel
from .transducers import _BaseTransducer


class CTCSkeleton(CTCModel):
    """Builds the CTC-based model skeleton

    Args:

        encoder (Module): The speech encoder (acoustic model), such that
        the forward of the encoder returns a tuple of the encoded speech
        tensor and a length tensor for the encoded speech.

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
        pred_net: Union[Module, None] = None,
        feat_size: Union[int, None] = None,
        n_classes: Union[int, None] = None,
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
        self.encoder = encoder
        if pred_net is not None:
            self.pred_net = pred_net


class TransducerSkeleton(_BaseTransducer):
    """Builds the Transducer-based model skeleton

    Args:
        encoder (Module): The speech encoder (acoustic model), such that
        the forward method of the encoder returns a tuple of the encoded
        speech tensor and a length tensor for the encoded speech.

        decoder (Module): The text decoder such that
        the forward method of the decoder returns a tuple of the encoded
        text tensor and a length tensor for the encoded text.

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
        join_net: Union[Module, None] = None,
        feat_size: Union[int, None] = None,
        n_classes: Union[int, None] = None,
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
        self.encoder = encoder
        self.decoder = decoder
        if join_net is not None:
            self.join_net = join_net


class Seq2SeqSkeleton(Module):
    """Builds the Seq2Seq model skeleton

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
    """

    def __init__(self, encoder: Module, decoder: Module, *args, **kwargs) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _process_encoder_out(self, out: tuple) -> dict:
        if len(out) == 2:
            # not rnn encoder and does not return the last hidden state
            out, lengths = out
            mask = get_mask_from_lens(lengths=lengths, max_len=out.shape[1])
            mask = mask.to(out.device)
            h = None
            return {"enc_out": out, "enc_mask": mask, "h": None}
        if len(out) == 3:
            out, h, lengths = out
            mask = get_mask_from_lens(lengths=lengths, max_len=out.shape[1])
            mask = mask.to(out.device)
            return {"enc_out": out, "enc_mask": mask, "h": h}
        # TODO: raise an error here

    def forward(
        self,
        speech: Tensor,
        speech_mask: Tensor,
        text: Tensor,
        text_mask: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        """Passes the input to the model.

        Args:
            speech (Tensor): The speech of shape [B, M, d]
            speech_mask (Tensor): The speech mask of shape [B, M]
            text (Tensor): The text tensor of shape [B, N]
            text_mask (Tensor): The text mask tensor of shape [B, M]

        Returns:
            Tensor: The result tensor of shape [B, N, C]
        """
        out = self.encoder(x=speech, mask=speech_mask, return_h=True)
        args = self._process_encoder_out(out)
        result = self.decoder(dec_inp=text, dec_mask=text_mask, **args)
        return result
