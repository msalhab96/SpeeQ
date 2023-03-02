import pytest
import torch
from torch.nn import Softmax

from speeq.models import decoders
from tests.helpers import IGNORE_USERWARNING, check_grad, get_mask


class TestGlobAttRNNDecoder:
    @pytest.mark.parametrize(
        (
            "embed_dim",
            "hidden_size",
            "n_layers",
            "n_classes",
            "pred_activation",
            "teacher_forcing_rate",
            "rnn_type",
            "batch_size",
            "enc_len",
            "target_len",
            "enc_pad_lens",
            "pass_h_nan",
        ),
        (
            (16, 8, 3, 6, Softmax(dim=-1), 0.15, "rnn", 3, 4, 5, [0, 1, 2], False),
            (16, 8, 3, 6, Softmax(dim=-1), 0.15, "rnn", 3, 1, 5, [0, 0, 0], False),
            (16, 8, 3, 6, Softmax(dim=-1), 0.15, "lstm", 3, 4, 5, [0, 1, 2], False),
            (16, 8, 3, 6, Softmax(dim=-1), 0.15, "gru", 3, 4, 5, [0, 1, 2], False),
            (16, 8, 1, 6, Softmax(dim=-1), 0.15, "rnn", 3, 4, 5, [0, 1, 2], False),
            (16, 8, 1, 6, Softmax(dim=-1), 0, "rnn", 3, 4, 5, [0, 1, 2], True),
        ),
    )
    def test_forward(
        self,
        batcher,
        int_batcher,
        embed_dim,
        hidden_size,
        n_layers,
        n_classes,
        pred_activation,
        teacher_forcing_rate,
        rnn_type,
        batch_size,
        enc_len,
        target_len,
        enc_pad_lens,
        pass_h_nan,
    ):
        expected_shape = (batch_size, target_len, n_classes)
        target = int_batcher(batch_size, target_len, n_classes)
        enc_out = batcher(batch_size, enc_len, hidden_size)
        if pass_h_nan is True:
            h = None
        else:
            h = batcher(1, batch_size, hidden_size)
            if rnn_type == "lstm":
                h = (
                    batcher(1, batch_size, hidden_size),
                    batcher(1, batch_size, hidden_size),
                )
        model = decoders.GlobAttRNNDecoder(
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_classes=n_classes,
            pred_activation=pred_activation,
            teacher_forcing_rate=teacher_forcing_rate,
            rnn_type=rnn_type,
        )
        enc_mask = get_mask(enc_len, enc_pad_lens)
        result = model(h=h, enc_out=enc_out, enc_mask=enc_mask, dec_inp=target)
        check_grad(result=result, model=model)
        assert result.shape == expected_shape
        assert torch.allclose(result.sum(dim=-1), torch.ones_like(target).float())


class TestLocationAwareAttDecoder:
    @pytest.mark.filterwarnings(IGNORE_USERWARNING)
    @pytest.mark.parametrize(
        (
            "embed_dim",
            "hidden_size",
            "n_layers",
            "n_classes",
            "pred_activation",
            "kernel_size",
            "activation",
            "inv_temperature",
            "teacher_forcing_rate",
            "rnn_type",
            "batch_size",
            "enc_len",
            "target_len",
            "enc_pad_lens",
            "pass_h_nan",
        ),
        (
            (
                16,
                8,
                3,
                6,
                Softmax(dim=-1),
                4,
                "softmax",
                1,
                0.15,
                "rnn",
                3,
                4,
                5,
                [0, 1, 2],
                False,
            ),
            (
                16,
                8,
                3,
                6,
                Softmax(dim=-1),
                4,
                "sigmax",
                1,
                0.15,
                "rnn",
                3,
                4,
                5,
                [0, 1, 2],
                False,
            ),
            (
                16,
                8,
                3,
                6,
                Softmax(dim=-1),
                4,
                "softmax",
                1,
                0.15,
                "rnn",
                3,
                1,
                5,
                [0, 0, 0],
                False,
            ),
            (
                16,
                8,
                3,
                6,
                Softmax(dim=-1),
                4,
                "softmax",
                1,
                0.15,
                "lstm",
                3,
                4,
                5,
                [0, 1, 2],
                False,
            ),
            (
                16,
                8,
                3,
                6,
                Softmax(dim=-1),
                4,
                "softmax",
                1,
                0.15,
                "gru",
                3,
                4,
                5,
                [0, 1, 2],
                False,
            ),
            (
                16,
                8,
                1,
                6,
                Softmax(dim=-1),
                2,
                "softmax",
                1,
                0.15,
                "rnn",
                3,
                4,
                5,
                [0, 1, 2],
                False,
            ),
            (
                16,
                8,
                1,
                6,
                Softmax(dim=-1),
                3,
                "softmax",
                1,
                0,
                "rnn",
                3,
                4,
                5,
                [0, 1, 2],
                False,
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        int_batcher,
        embed_dim,
        hidden_size,
        activation,
        n_layers,
        n_classes,
        pred_activation,
        kernel_size,
        inv_temperature,
        teacher_forcing_rate,
        rnn_type,
        batch_size,
        enc_len,
        target_len,
        enc_pad_lens,
        pass_h_nan,
    ):
        expected_shape = (batch_size, target_len, n_classes)
        target = int_batcher(batch_size, target_len, n_classes)
        enc_out = batcher(batch_size, enc_len, hidden_size)
        if pass_h_nan is True:
            h = None
        else:
            h = batcher(1, batch_size, hidden_size)
            if rnn_type == "lstm":
                h = (
                    batcher(1, batch_size, hidden_size),
                    batcher(1, batch_size, hidden_size),
                )
        model = decoders.LocationAwareAttDecoder(
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_classes=n_classes,
            pred_activation=pred_activation,
            kernel_size=kernel_size,
            activation=activation,
            inv_temperature=inv_temperature,
            teacher_forcing_rate=teacher_forcing_rate,
            rnn_type=rnn_type,
        )
        enc_mask = get_mask(enc_len, enc_pad_lens)
        result = model(h=h, enc_out=enc_out, enc_mask=enc_mask, dec_inp=target)
        check_grad(result=result, model=model)
        assert result.shape == expected_shape
        assert torch.allclose(result.sum(dim=-1), torch.ones_like(target).float())


class TestTransformerDecoder:
    @pytest.mark.filterwarnings(IGNORE_USERWARNING)
    @pytest.mark.parametrize(
        (
            "n_classes",
            "n_layers",
            "d_model",
            "ff_size",
            "h",
            "pred_activation",
            "batch_size",
            "enc_len",
            "target_len",
            "enc_pad_lens",
            "dec_pad_lens",
        ),
        (
            (6, 2, 8, 4, 2, Softmax(dim=-1), 3, 4, 5, [0, 1, 3], [0, 1, 0]),
            (6, 2, 8, 4, 2, Softmax(dim=-1), 3, 4, 1, [0, 1, 3], [0, 0, 0]),
            (6, 2, 8, 4, 2, Softmax(dim=-1), 3, 1, 5, [0, 0, 0], [0, 1, 0]),
            (6, 2, 8, 4, 2, Softmax(dim=-1), 3, 4, 5, [0, 1, 0], [0, 1, 0]),
            (6, 2, 8, 4, 2, Softmax(dim=-1), 3, 4, 5, [0, 1, 3], [0, 1, 3]),
        ),
    )
    def test_forward(
        self,
        batcher,
        int_batcher,
        n_classes,
        n_layers,
        d_model,
        ff_size,
        h,
        pred_activation,
        batch_size,
        enc_len,
        target_len,
        enc_pad_lens,
        dec_pad_lens,
    ):
        expected_shape = (batch_size, target_len, n_classes)
        target = int_batcher(batch_size, target_len, n_classes)
        enc_out = batcher(batch_size, enc_len, d_model)
        model = decoders.TransformerDecoder(
            n_classes=n_classes,
            n_layers=n_layers,
            d_model=d_model,
            ff_size=ff_size,
            h=h,
            pred_activation=pred_activation,
        )
        enc_mask = get_mask(enc_len, enc_pad_lens)
        dec_mask = get_mask(target_len, dec_pad_lens)
        result = model(
            enc_out=enc_out, enc_mask=enc_mask, dec_inp=target, dec_mask=dec_mask
        )
        check_grad(result=result, model=model)
        assert result.shape == expected_shape
        assert torch.allclose(result.sum(dim=-1), torch.ones_like(target).float())


class TestSpeechTransformerDecoder:
    @pytest.mark.filterwarnings(IGNORE_USERWARNING)
    @pytest.mark.parametrize(
        (
            "n_classes",
            "n_layers",
            "d_model",
            "ff_size",
            "h",
            "pred_activation",
            "batch_size",
            "enc_len",
            "target_len",
            "enc_pad_lens",
            "dec_pad_lens",
        ),
        (
            (6, 2, 8, 4, 2, Softmax(dim=-1), 3, 4, 5, [0, 1, 3], [0, 1, 0]),
            (6, 2, 8, 4, 2, Softmax(dim=-1), 3, 4, 1, [0, 1, 3], [0, 0, 0]),
            (6, 2, 8, 4, 2, Softmax(dim=-1), 3, 1, 5, [0, 0, 0], [0, 1, 0]),
            (6, 2, 8, 4, 2, Softmax(dim=-1), 3, 4, 5, [0, 1, 0], [0, 1, 0]),
            (6, 2, 8, 4, 2, Softmax(dim=-1), 3, 4, 5, [0, 1, 3], [0, 1, 3]),
        ),
    )
    def test_forward(
        self,
        batcher,
        int_batcher,
        n_classes,
        n_layers,
        d_model,
        ff_size,
        h,
        pred_activation,
        batch_size,
        enc_len,
        target_len,
        enc_pad_lens,
        dec_pad_lens,
    ):
        expected_shape = (batch_size, target_len, n_classes)
        target = int_batcher(batch_size, target_len, n_classes)
        enc_out = batcher(batch_size, enc_len, d_model)
        model = decoders.SpeechTransformerDecoder(
            n_classes=n_classes,
            n_layers=n_layers,
            d_model=d_model,
            ff_size=ff_size,
            h=h,
            pred_activation=pred_activation,
        )
        enc_mask = get_mask(enc_len, enc_pad_lens)
        dec_mask = get_mask(target_len, dec_pad_lens)
        result = model(
            enc_out=enc_out, enc_mask=enc_mask, dec_inp=target, dec_mask=dec_mask
        )
        check_grad(result=result, model=model)
        assert result.shape == expected_shape
        assert torch.allclose(result.sum(dim=-1), torch.ones_like(target).float())


class TestTransformerTransducerDecoder:
    @pytest.mark.parametrize(
        (
            "vocab_size",
            "n_layers",
            "d_model",
            "ff_size",
            "h",
            "left_size",
            "right_size",
            "p_dropout",
            "batch_size",
            "seq_len",
            "pad_lens",
            "expected_shape",
        ),
        (
            (7, 2, 16, 4, 2, 3, 5, 0.1, 3, 6, [0, 1, 0], (3, 6, 16)),
            (
                7,
                2,
                16,
                4,
                2,
                3,
                5,
                0.1,
                1,
                6,
                [
                    0,
                ],
                (1, 6, 16),
            ),
            (7, 2, 16, 4, 2, 1, 1, 0.1, 3, 6, [0, 1, 0], (3, 6, 16)),
        ),
    )
    def test_forward(
        self,
        int_batcher,
        vocab_size,
        n_layers,
        d_model,
        ff_size,
        h,
        left_size,
        right_size,
        p_dropout,
        batch_size,
        seq_len,
        pad_lens,
        expected_shape,
    ):
        input = int_batcher(batch_size, seq_len, vocab_size)
        print(input.shape)
        model = decoders.TransformerTransducerDecoder(
            vocab_size=vocab_size,
            n_layers=n_layers,
            d_model=d_model,
            ff_size=ff_size,
            h=h,
            left_size=left_size,
            right_size=right_size,
            p_dropout=p_dropout,
        )
        mask = get_mask(seq_len, pad_lens)
        result, _ = model(input, mask)
        print(result.shape)
        check_grad(result=result, model=model)
        assert result.shape == expected_shape
