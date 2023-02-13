import pytest
import torch

from speeq.models import seq2seq
from tests.helpers import check_grad, get_mask


class Seq2SeqBaseTest:
    def check(
        self,
        batcher,
        int_batcher,
        model_args,
        n_classes,
        batch_size,
        enc_feat_size,
        enc_len,
        dec_len,
        enc_pad_lens,
        dec_pad_lens,
        expected_shape,
    ):
        enc_inp = batcher(batch_size, enc_len, enc_feat_size)
        dec_inp = int_batcher(batch_size, dec_len, n_classes)
        enc_mask = get_mask(seq_len=enc_len, pad_lens=enc_pad_lens)
        dec_mask = get_mask(seq_len=dec_len, pad_lens=dec_pad_lens)
        model = self.model(**model_args)
        result = model(
            speech=enc_inp, speech_mask=enc_mask, text=dec_inp, text_mask=dec_mask
        )
        assert result.shape == expected_shape
        check_grad(result=result, model=model)


class TestBasicAttSeq2SeqRNN(Seq2SeqBaseTest):
    model = seq2seq.BasicAttSeq2SeqRNN

    @pytest.mark.parametrize(
        (
            "model_args",
            "n_classes",
            "batch_size",
            "enc_feat_size",
            "enc_len",
            "dec_len",
            "enc_pad_lens",
            "dec_pad_lens",
            "expected_shape",
        ),
        (
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "hidden_size": 4,
                    "enc_num_layers": 2,
                    "bidirectional": False,
                    "dec_num_layers": 2,
                    "emb_dim": 4,
                    "p_dropout": 0.1,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                    "teacher_forcing_rate": 0.1,
                    "rnn_type": "rnn",
                },
                5,
                3,
                8,
                3,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "hidden_size": 4,
                    "enc_num_layers": 2,
                    "bidirectional": False,
                    "dec_num_layers": 2,
                    "emb_dim": 4,
                    "p_dropout": 0.1,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                    "teacher_forcing_rate": 0.1,
                    "rnn_type": "lstm",
                },
                5,
                3,
                8,
                3,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "hidden_size": 4,
                    "enc_num_layers": 2,
                    "bidirectional": False,
                    "dec_num_layers": 2,
                    "emb_dim": 4,
                    "p_dropout": 0.1,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                    "teacher_forcing_rate": 0.1,
                    "rnn_type": "gru",
                },
                5,
                3,
                8,
                3,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "hidden_size": 4,
                    "enc_num_layers": 2,
                    "bidirectional": True,
                    "dec_num_layers": 2,
                    "emb_dim": 4,
                    "p_dropout": 0.1,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                    "teacher_forcing_rate": 0.1,
                    "rnn_type": "rnn",
                },
                5,
                3,
                8,
                3,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        int_batcher,
        model_args,
        n_classes,
        batch_size,
        enc_feat_size,
        enc_len,
        dec_len,
        enc_pad_lens,
        dec_pad_lens,
        expected_shape,
    ):
        self.check(
            batcher,
            int_batcher,
            model_args,
            n_classes,
            batch_size,
            enc_feat_size,
            enc_len,
            dec_len,
            enc_pad_lens,
            dec_pad_lens,
            expected_shape,
        )


class TestLAS(Seq2SeqBaseTest):
    model = seq2seq.LAS

    @pytest.mark.parametrize(
        (
            "model_args",
            "n_classes",
            "batch_size",
            "enc_feat_size",
            "enc_len",
            "dec_len",
            "enc_pad_lens",
            "dec_pad_lens",
            "expected_shape",
        ),
        (
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "hidden_size": 6,
                    "enc_num_layers": 1,
                    "reduction_factor": 1,
                    "bidirectional": False,
                    "dec_num_layers": 2,
                    "emb_dim": 4,
                    "p_dropout": 0.1,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                    "teacher_forcing_rate": 0.1,
                    "rnn_type": "rnn",
                },
                5,
                3,
                8,
                6,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "hidden_size": 6,
                    "enc_num_layers": 1,
                    "reduction_factor": 1,
                    "bidirectional": False,
                    "dec_num_layers": 2,
                    "emb_dim": 4,
                    "p_dropout": 0.1,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                    "teacher_forcing_rate": 0.1,
                    "rnn_type": "lstm",
                },
                5,
                3,
                8,
                6,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "hidden_size": 6,
                    "enc_num_layers": 1,
                    "reduction_factor": 1,
                    "bidirectional": False,
                    "dec_num_layers": 2,
                    "emb_dim": 4,
                    "p_dropout": 0.1,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                    "teacher_forcing_rate": 0.1,
                    "rnn_type": "gru",
                },
                5,
                3,
                8,
                6,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "hidden_size": 6,
                    "enc_num_layers": 1,
                    "reduction_factor": 2,
                    "bidirectional": False,
                    "dec_num_layers": 2,
                    "emb_dim": 4,
                    "p_dropout": 0.1,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                    "teacher_forcing_rate": 0.1,
                    "rnn_type": "rnn",
                },
                5,
                3,
                8,
                6,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "hidden_size": 6,
                    "enc_num_layers": 3,
                    "reduction_factor": 2,
                    "bidirectional": False,
                    "dec_num_layers": 2,
                    "emb_dim": 4,
                    "p_dropout": 0.1,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                    "teacher_forcing_rate": 0.1,
                    "rnn_type": "rnn",
                },
                5,
                3,
                8,
                6,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "hidden_size": 6,
                    "enc_num_layers": 3,
                    "reduction_factor": 2,
                    "bidirectional": True,
                    "dec_num_layers": 2,
                    "emb_dim": 4,
                    "p_dropout": 0.1,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                    "teacher_forcing_rate": 0.1,
                    "rnn_type": "rnn",
                },
                5,
                3,
                8,
                6,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        int_batcher,
        model_args,
        n_classes,
        batch_size,
        enc_feat_size,
        enc_len,
        dec_len,
        enc_pad_lens,
        dec_pad_lens,
        expected_shape,
    ):
        self.check(
            batcher,
            int_batcher,
            model_args,
            n_classes,
            batch_size,
            enc_feat_size,
            enc_len,
            dec_len,
            enc_pad_lens,
            dec_pad_lens,
            expected_shape,
        )


class TestRNNWithLocationAwareAtt(Seq2SeqBaseTest):
    model = seq2seq.RNNWithLocationAwareAtt

    @pytest.mark.parametrize(
        (
            "model_args",
            "n_classes",
            "batch_size",
            "enc_feat_size",
            "enc_len",
            "dec_len",
            "enc_pad_lens",
            "dec_pad_lens",
            "expected_shape",
        ),
        (
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "hidden_size": 6,
                    "enc_num_layers": 2,
                    "bidirectional": False,
                    "dec_num_layers": 1,
                    "emb_dim": 4,
                    "kernel_size": 5,
                    "activation": "softmax",
                    "p_dropout": 0.1,
                    "inv_temperature": 1,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                    "teacher_forcing_rate": 0.1,
                    "rnn_type": "rnn",
                },
                5,
                3,
                8,
                6,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "hidden_size": 6,
                    "enc_num_layers": 2,
                    "bidirectional": False,
                    "dec_num_layers": 1,
                    "emb_dim": 4,
                    "kernel_size": 5,
                    "activation": "softmax",
                    "p_dropout": 0.1,
                    "inv_temperature": 1,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                    "teacher_forcing_rate": 0.1,
                    "rnn_type": "lstm",
                },
                5,
                3,
                8,
                6,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "hidden_size": 6,
                    "enc_num_layers": 2,
                    "bidirectional": False,
                    "dec_num_layers": 1,
                    "emb_dim": 4,
                    "kernel_size": 5,
                    "activation": "softmax",
                    "p_dropout": 0.1,
                    "inv_temperature": 1,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                    "teacher_forcing_rate": 0.1,
                    "rnn_type": "gru",
                },
                5,
                3,
                8,
                6,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "hidden_size": 6,
                    "enc_num_layers": 2,
                    "bidirectional": True,
                    "dec_num_layers": 1,
                    "emb_dim": 4,
                    "kernel_size": 5,
                    "activation": "softmax",
                    "p_dropout": 0.1,
                    "inv_temperature": 1,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                    "teacher_forcing_rate": 0.1,
                    "rnn_type": "rnn",
                },
                5,
                3,
                8,
                6,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "hidden_size": 6,
                    "enc_num_layers": 1,
                    "bidirectional": True,
                    "dec_num_layers": 2,
                    "emb_dim": 4,
                    "kernel_size": 5,
                    "activation": "softmax",
                    "p_dropout": 0.1,
                    "inv_temperature": 1,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                    "teacher_forcing_rate": 0.1,
                    "rnn_type": "rnn",
                },
                5,
                3,
                8,
                6,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        int_batcher,
        model_args,
        n_classes,
        batch_size,
        enc_feat_size,
        enc_len,
        dec_len,
        enc_pad_lens,
        dec_pad_lens,
        expected_shape,
    ):
        self.check(
            batcher,
            int_batcher,
            model_args,
            n_classes,
            batch_size,
            enc_feat_size,
            enc_len,
            dec_len,
            enc_pad_lens,
            dec_pad_lens,
            expected_shape,
        )


class TestSpeechTransformer(Seq2SeqBaseTest):
    model = seq2seq.SpeechTransformer

    @pytest.mark.parametrize(
        (
            "model_args",
            "n_classes",
            "batch_size",
            "enc_feat_size",
            "enc_len",
            "dec_len",
            "enc_pad_lens",
            "dec_pad_lens",
            "expected_shape",
        ),
        (
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "n_conv_layers": 1,
                    "kernel_size": 1,
                    "stride": 1,
                    "d_model": 16,
                    "n_enc_layers": 1,
                    "n_dec_layers": 2,
                    "ff_size": 8,
                    "h": 2,
                    "att_kernel_size": 5,
                    "att_out_channels": 8,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                },
                5,
                3,
                8,
                6,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "n_conv_layers": 1,
                    "kernel_size": 1,
                    "stride": 1,
                    "d_model": 16,
                    "n_enc_layers": 2,
                    "n_dec_layers": 1,
                    "ff_size": 8,
                    "h": 2,
                    "att_kernel_size": 5,
                    "att_out_channels": 8,
                    "pred_activation": torch.nn.Softmax(dim=-1),
                },
                5,
                3,
                8,
                6,
                5,
                [0, 2, 1],
                [1, 0, 2],
                (3, 5, 5),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        int_batcher,
        model_args,
        n_classes,
        batch_size,
        enc_feat_size,
        enc_len,
        dec_len,
        enc_pad_lens,
        dec_pad_lens,
        expected_shape,
    ):
        self.check(
            batcher,
            int_batcher,
            model_args,
            n_classes,
            batch_size,
            enc_feat_size,
            enc_len,
            dec_len,
            enc_pad_lens,
            dec_pad_lens,
            expected_shape,
        )