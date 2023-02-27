import pytest
import torch

from speeq.models import ctc
from tests.helpers import IGNORE_USERWARNING, check_grad, get_mask


class CTCBaseTest:
    def check(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        in_features,
        pad_lens,
        expected_lens,
        expected_shape,
    ):
        input = batcher(batch_size, seq_len, in_features)
        mask = get_mask(seq_len=seq_len, pad_lens=pad_lens)
        model = self.model(**model_args)
        results, lengths = model(input, mask)
        assert results.shape == expected_shape
        assert torch.all(expected_lens == lengths)
        check_grad(result=results, model=model)


class TestCTCModel:
    def test_create_object(self):
        with pytest.raises(NotImplementedError):
            ctc.CTCModel(1, 1)


class TestDeepSpeechV1(CTCBaseTest):
    model = ctc.DeepSpeechV1

    @pytest.mark.parametrize(
        (
            "model_args",
            "batch_size",
            "seq_len",
            "in_features",
            "pad_lens",
            "expected_lens",
            "expected_shape",
        ),
        (
            (
                {
                    "in_features": 8,
                    "hidden_size": 12,
                    "n_linear_layers": 2,
                    "bidirectional": False,
                    "n_classes": 5,
                    "max_clip_value": 10,
                    "rnn_type": "rnn",
                    "p_dropout": 0.01,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 10]),
                (10, 2, 5),
            ),
            (
                {
                    "in_features": 8,
                    "hidden_size": 12,
                    "n_linear_layers": 2,
                    "bidirectional": True,
                    "n_classes": 5,
                    "max_clip_value": 10,
                    "rnn_type": "rnn",
                    "p_dropout": 0.01,
                },
                2,
                10,
                8,
                [0, 0],
                torch.LongTensor([10, 10]),
                (10, 2, 5),
            ),
            (
                {
                    "in_features": 8,
                    "hidden_size": 12,
                    "n_linear_layers": 2,
                    "bidirectional": True,
                    "n_classes": 5,
                    "max_clip_value": 10,
                    "rnn_type": "lstm",
                    "p_dropout": 0.01,
                },
                2,
                10,
                8,
                [0, 0],
                torch.LongTensor([10, 10]),
                (10, 2, 5),
            ),
            (
                {
                    "in_features": 8,
                    "hidden_size": 12,
                    "n_linear_layers": 2,
                    "bidirectional": True,
                    "n_classes": 5,
                    "max_clip_value": 10,
                    "rnn_type": "gru",
                    "p_dropout": 0.01,
                },
                2,
                10,
                8,
                [0, 0],
                torch.LongTensor([10, 10]),
                (10, 2, 5),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        in_features,
        pad_lens,
        expected_lens,
        expected_shape,
    ):
        self.check(
            batcher=batcher,
            model_args=model_args,
            batch_size=batch_size,
            seq_len=seq_len,
            in_features=in_features,
            pad_lens=pad_lens,
            expected_lens=expected_lens,
            expected_shape=expected_shape,
        )


class TestBERT(CTCBaseTest):
    model = ctc.BERT

    @pytest.mark.parametrize(
        (
            "model_args",
            "batch_size",
            "seq_len",
            "in_features",
            "pad_lens",
            "expected_lens",
            "expected_shape",
        ),
        (
            (
                {
                    "max_len": 15,
                    "in_features": 8,
                    "d_model": 16,
                    "h": 2,
                    "ff_size": 4,
                    "n_layers": 2,
                    "n_classes": 5,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 10]),
                (10, 2, 5),
            ),
            (
                {
                    "max_len": 15,
                    "in_features": 8,
                    "d_model": 16,
                    "h": 2,
                    "ff_size": 4,
                    "n_layers": 1,
                    "n_classes": 5,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 10]),
                (10, 2, 5),
            ),
            (
                {
                    "max_len": 15,
                    "in_features": 8,
                    "d_model": 16,
                    "h": 2,
                    "ff_size": 4,
                    "n_layers": 1,
                    "n_classes": 5,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [0, 0],
                torch.LongTensor([10, 10]),
                (10, 2, 5),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        in_features,
        pad_lens,
        expected_lens,
        expected_shape,
    ):
        self.check(
            batcher=batcher,
            model_args=model_args,
            batch_size=batch_size,
            seq_len=seq_len,
            in_features=in_features,
            pad_lens=pad_lens,
            expected_lens=expected_lens,
            expected_shape=expected_shape,
        )


class TestDeepSpeechV2(CTCBaseTest):
    model = ctc.DeepSpeechV2

    @pytest.mark.parametrize(
        (
            "model_args",
            "batch_size",
            "seq_len",
            "in_features",
            "pad_lens",
            "expected_lens",
            "expected_shape",
        ),
        (
            (
                {
                    "n_conv": 4,
                    "kernel_size": 1,
                    "stride": 1,
                    "in_features": 8,
                    "hidden_size": 16,
                    "bidirectional": False,
                    "n_rnn": 2,
                    "n_linear_layers": 2,
                    "n_classes": 5,
                    "max_clip_value": 10,
                    "rnn_type": "rnn",
                    "tau": 5,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 10]),
                (10, 2, 5),
            ),
            (
                {
                    "n_conv": 4,
                    "kernel_size": 1,
                    "stride": 1,
                    "in_features": 8,
                    "hidden_size": 16,
                    "bidirectional": True,
                    "n_rnn": 2,
                    "n_linear_layers": 2,
                    "n_classes": 5,
                    "max_clip_value": 10,
                    "rnn_type": "rnn",
                    "tau": 5,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 10]),
                (10, 2, 5),
            ),
            (
                {
                    "n_conv": 2,
                    "kernel_size": 2,
                    "stride": 1,
                    "in_features": 8,
                    "hidden_size": 16,
                    "bidirectional": False,
                    "n_rnn": 2,
                    "n_linear_layers": 2,
                    "n_classes": 5,
                    "max_clip_value": 10,
                    "rnn_type": "rnn",
                    "tau": 5,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 8]),
                (8, 2, 5),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        in_features,
        pad_lens,
        expected_lens,
        expected_shape,
    ):
        self.check(
            batcher=batcher,
            model_args=model_args,
            batch_size=batch_size,
            seq_len=seq_len,
            in_features=in_features,
            pad_lens=pad_lens,
            expected_lens=expected_lens,
            expected_shape=expected_shape,
        )


class TestConformer(CTCBaseTest):
    model = ctc.Conformer

    @pytest.mark.filterwarnings(IGNORE_USERWARNING)
    @pytest.mark.parametrize(
        (
            "model_args",
            "batch_size",
            "seq_len",
            "in_features",
            "pad_lens",
            "expected_lens",
            "expected_shape",
        ),
        (
            (
                {
                    "n_classes": 5,
                    "d_model": 8,
                    "n_conf_layers": 2,
                    "ff_expansion_factor": 2,
                    "h": 2,
                    "kernel_size": 4,
                    "ss_kernel_size": 1,
                    "ss_stride": 1,
                    "ss_num_conv_layers": 1,
                    "in_features": 8,
                    "res_scaling": 0.5,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 10]),
                (10, 2, 5),
            ),
            (
                {
                    "n_classes": 5,
                    "d_model": 8,
                    "n_conf_layers": 1,
                    "ff_expansion_factor": 2,
                    "h": 2,
                    "kernel_size": 1,
                    "ss_kernel_size": 1,
                    "ss_stride": 1,
                    "ss_num_conv_layers": 1,
                    "in_features": 8,
                    "res_scaling": 0.5,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 10]),
                (10, 2, 5),
            ),
            (
                {
                    "n_classes": 5,
                    "d_model": 8,
                    "n_conf_layers": 2,
                    "ff_expansion_factor": 2,
                    "h": 2,
                    "kernel_size": 4,
                    "ss_kernel_size": 2,
                    "ss_stride": 1,
                    "ss_num_conv_layers": 2,
                    "in_features": 8,
                    "res_scaling": 0.5,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 8]),
                (8, 2, 5),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        in_features,
        pad_lens,
        expected_lens,
        expected_shape,
    ):
        self.check(
            batcher=batcher,
            model_args=model_args,
            batch_size=batch_size,
            seq_len=seq_len,
            in_features=in_features,
            pad_lens=pad_lens,
            expected_lens=expected_lens,
            expected_shape=expected_shape,
        )


class TestJasper(CTCBaseTest):
    model = ctc.Jasper

    @pytest.mark.parametrize(
        (
            "model_args",
            "batch_size",
            "seq_len",
            "in_features",
            "pad_lens",
            "expected_lens",
            "expected_shape",
        ),
        (
            (
                {
                    "n_classes": 5,
                    "in_features": 8,
                    "num_blocks": 2,
                    "num_sub_blocks": 3,
                    "channel_inc": 2,
                    "epilog_kernel_size": 1,
                    "prelog_kernel_size": 1,
                    "prelog_stride": 1,
                    "prelog_n_channels": 4,
                    "blocks_kernel_size": 2,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 10]),
                (10, 2, 5),
            ),
            (
                {
                    "n_classes": 5,
                    "in_features": 8,
                    "num_blocks": 2,
                    "num_sub_blocks": 3,
                    "channel_inc": 2,
                    "epilog_kernel_size": 1,
                    "prelog_kernel_size": 1,
                    "prelog_stride": 1,
                    "prelog_n_channels": 8,
                    "blocks_kernel_size": 2,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 10]),
                (10, 2, 5),
            ),
            (
                {
                    "n_classes": 5,
                    "in_features": 8,
                    "num_blocks": 2,
                    "num_sub_blocks": 3,
                    "channel_inc": 2,
                    "epilog_kernel_size": 4,
                    "prelog_kernel_size": 2,
                    "prelog_stride": 1,
                    "prelog_n_channels": 8,
                    "blocks_kernel_size": 2,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 9]),
                (9, 2, 5),
            ),
            (
                {
                    "n_classes": 5,
                    "in_features": 8,
                    "num_blocks": 2,
                    "num_sub_blocks": 3,
                    "channel_inc": 2,
                    "epilog_kernel_size": 1,
                    "prelog_kernel_size": 2,
                    "prelog_stride": 1,
                    "prelog_n_channels": 8,
                    "blocks_kernel_size": 2,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 9]),
                (9, 2, 5),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        in_features,
        pad_lens,
        expected_lens,
        expected_shape,
    ):
        self.check(
            batcher=batcher,
            model_args=model_args,
            batch_size=batch_size,
            seq_len=seq_len,
            in_features=in_features,
            pad_lens=pad_lens,
            expected_lens=expected_lens,
            expected_shape=expected_shape,
        )


class TestWav2Letter(CTCBaseTest):
    model = ctc.Wav2Letter

    @pytest.mark.parametrize(
        (
            "model_args",
            "batch_size",
            "seq_len",
            "in_features",
            "pad_lens",
            "expected_lens",
            "expected_shape",
        ),
        (
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "n_conv_layers": 3,
                    "layers_kernel_size": 3,
                    "layers_channels_size": 2,
                    "pre_conv_stride": 1,
                    "pre_conv_kernel_size": 1,
                    "post_conv_channels_size": 4,
                    "post_conv_kernel_size": 3,
                    "p_dropout": 0.1,
                    "wav_kernel_size": None,
                    "wav_stride": None,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 10]),
                (10, 2, 5),
            ),
            (
                {
                    "in_features": 8,
                    "n_classes": 5,
                    "n_conv_layers": 3,
                    "layers_kernel_size": 3,
                    "layers_channels_size": 2,
                    "pre_conv_stride": 1,
                    "pre_conv_kernel_size": 2,
                    "post_conv_channels_size": 4,
                    "post_conv_kernel_size": 3,
                    "p_dropout": 0.1,
                    "wav_kernel_size": None,
                    "wav_stride": None,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 9]),
                (9, 2, 5),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        in_features,
        pad_lens,
        expected_lens,
        expected_shape,
    ):
        self.check(
            batcher=batcher,
            model_args=model_args,
            batch_size=batch_size,
            seq_len=seq_len,
            in_features=in_features,
            pad_lens=pad_lens,
            expected_lens=expected_lens,
            expected_shape=expected_shape,
        )


class TestQuartzNet(CTCBaseTest):
    model = ctc.QuartzNet

    @pytest.mark.parametrize(
        (
            "model_args",
            "batch_size",
            "seq_len",
            "in_features",
            "pad_lens",
            "expected_lens",
            "expected_shape",
        ),
        (
            (
                {
                    "n_classes": 5,
                    "in_features": 8,
                    "num_blocks": 2,
                    "block_repetition": 2,
                    "num_sub_blocks": 3,
                    "channels_size": [2, 1],
                    "epilog_kernel_size": 1,
                    "epilog_channel_size": (3, 1),
                    "prelog_kernel_size": 1,
                    "prelog_stride": 1,
                    "prelog_n_channels": 4,
                    "groups": 1,
                    "blocks_kernel_size": 2,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 10]),
                (10, 2, 5),
            ),
            (
                {
                    "n_classes": 5,
                    "in_features": 8,
                    "num_blocks": 2,
                    "block_repetition": 2,
                    "num_sub_blocks": 3,
                    "channels_size": [2, 1],
                    "epilog_kernel_size": 1,
                    "epilog_channel_size": (3, 1),
                    "prelog_kernel_size": 2,
                    "prelog_stride": 1,
                    "prelog_n_channels": 4,
                    "groups": 1,
                    "blocks_kernel_size": 2,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 9]),
                (9, 2, 5),
            ),
            (
                {
                    "n_classes": 5,
                    "in_features": 8,
                    "num_blocks": 2,
                    "block_repetition": 2,
                    "num_sub_blocks": 3,
                    "channels_size": [2, 2],
                    "epilog_kernel_size": 1,
                    "epilog_channel_size": (3, 1),
                    "prelog_kernel_size": 2,
                    "prelog_stride": 1,
                    "prelog_n_channels": 4,
                    "groups": 2,
                    "blocks_kernel_size": 2,
                    "p_dropout": 0.1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 9]),
                (9, 2, 5),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        in_features,
        pad_lens,
        expected_lens,
        expected_shape,
    ):
        self.check(
            batcher=batcher,
            model_args=model_args,
            batch_size=batch_size,
            seq_len=seq_len,
            in_features=in_features,
            pad_lens=pad_lens,
            expected_lens=expected_lens,
            expected_shape=expected_shape,
        )


class TestSqueezeformer(CTCBaseTest):
    model = ctc.Squeezeformer

    @pytest.mark.parametrize(
        (
            "model_args",
            "batch_size",
            "seq_len",
            "in_features",
            "pad_lens",
            "expected_lens",
            "expected_shape",
        ),
        (
            (
                {
                    "n_classes": 5,
                    "in_features": 8,
                    "n": 3,
                    "d_model": 16,
                    "ff_expansion_factor": 1,
                    "h": 2,
                    "kernel_size": 3,
                    "pooling_kernel_size": 3,
                    "pooling_stride": 1,
                    "ss_kernel_size": 1,
                    "ss_stride": 1,
                    "ss_n_conv_layers": 1,
                    "p_dropout": 0.1,
                    "ss_groups": 1,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 10]),
                (10, 2, 5),
            ),
            (
                {
                    "n_classes": 5,
                    "in_features": 8,
                    "n": 3,
                    "d_model": 16,
                    "ff_expansion_factor": 1,
                    "h": 2,
                    "kernel_size": 3,
                    "pooling_kernel_size": 3,
                    "pooling_stride": 1,
                    "ss_kernel_size": 1,
                    "ss_stride": 1,
                    "ss_n_conv_layers": 1,
                    "p_dropout": 0.1,
                    "ss_groups": 4,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 10]),
                (10, 2, 5),
            ),
            (
                {
                    "n_classes": 5,
                    "in_features": 8,
                    "n": 1,
                    "d_model": 16,
                    "ff_expansion_factor": 1,
                    "h": 2,
                    "kernel_size": 3,
                    "pooling_kernel_size": 3,
                    "pooling_stride": 1,
                    "ss_kernel_size": 2,
                    "ss_stride": 1,
                    "ss_n_conv_layers": 2,
                    "p_dropout": 0.1,
                    "ss_groups": 4,
                },
                2,
                10,
                8,
                [2, 0],
                torch.LongTensor([8, 8]),
                (8, 2, 5),
            ),
        ),
    )
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        in_features,
        pad_lens,
        expected_lens,
        expected_shape,
    ):
        self.check(
            batcher=batcher,
            model_args=model_args,
            batch_size=batch_size,
            seq_len=seq_len,
            in_features=in_features,
            pad_lens=pad_lens,
            expected_lens=expected_lens,
            expected_shape=expected_shape,
        )
