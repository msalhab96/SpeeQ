import pytest
import torch

from speeq.models import encoders
from tests.helpers import IGNORE_USERWARNING, check_grad, get_mask


def encoder_paramterizer(test_cases):
    return pytest.mark.parametrize(
        (
            "model_args",
            "batch_size",
            "seq_len",
            "feat_size",
            "pad_lens",
            "expected_shape",
            "expected_lens",
        ),
        test_cases,
    )


class BaseTest:
    def check(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        feat_size,
        pad_lens,
        expected_shape,
        expected_lens,
    ):
        input = batcher(batch_size, seq_len, feat_size)
        mask = get_mask(seq_len=seq_len, pad_lens=pad_lens)
        model = self.model(**model_args)
        result, lengths = model(input, mask)
        assert result.shape == expected_shape
        assert torch.all(expected_lens == lengths).item()
        check_grad(result=result, model=model)


class TestDeepSpeechV1Encoder(BaseTest):
    model = encoders.DeepSpeechV1Encoder
    test_cases = (
        (
            {
                "in_features": 8,
                "hidden_size": 32,
                "n_linear_layers": 3,
                "bidirectional": False,
                "max_clip_value": 10,
                "rnn_type": "rnn",
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 32),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 32,
                "n_linear_layers": 3,
                "bidirectional": False,
                "max_clip_value": 10,
                "rnn_type": "gru",
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 32),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 32,
                "n_linear_layers": 3,
                "bidirectional": False,
                "max_clip_value": 10,
                "rnn_type": "lstm",
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 32),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 32,
                "n_linear_layers": 3,
                "bidirectional": False,
                "max_clip_value": 10,
                "rnn_type": "rnn",
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [1, 6, 3],
            (3, 9, 32),
            torch.LongTensor([9, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 32,
                "n_linear_layers": 3,
                "bidirectional": True,
                "max_clip_value": 10,
                "rnn_type": "rnn",
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [1, 6, 3],
            (3, 9, 32),
            torch.LongTensor([9, 4, 7]),
        ),
    )

    @encoder_paramterizer(test_cases=test_cases)
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        feat_size,
        pad_lens,
        expected_shape,
        expected_lens,
    ):
        self.check(
            batcher,
            model_args,
            batch_size,
            seq_len,
            feat_size,
            pad_lens,
            expected_shape,
            expected_lens,
        )


class TestDeepSpeechV2Encoder(BaseTest):
    model = encoders.DeepSpeechV2Encoder
    test_cases = (
        (
            {
                "n_conv": 1,
                "kernel_size": 1,
                "stride": 1,
                "in_features": 8,
                "hidden_size": 32,
                "bidirectional": False,
                "max_clip_value": 10,
                "n_rnn": 2,
                "n_linear_layers": 2,
                "rnn_type": "rnn",
                "tau": 5,
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 32),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "n_conv": 1,
                "kernel_size": 1,
                "stride": 1,
                "in_features": 8,
                "hidden_size": 32,
                "bidirectional": False,
                "max_clip_value": 10,
                "n_rnn": 2,
                "n_linear_layers": 2,
                "rnn_type": "gru",
                "tau": 5,
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 32),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "n_conv": 1,
                "kernel_size": 1,
                "stride": 1,
                "in_features": 8,
                "hidden_size": 32,
                "bidirectional": False,
                "max_clip_value": 10,
                "n_rnn": 2,
                "n_linear_layers": 2,
                "rnn_type": "lstm",
                "tau": 5,
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 32),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "n_conv": 1,
                "kernel_size": 1,
                "stride": 1,
                "in_features": 8,
                "hidden_size": 32,
                "bidirectional": True,
                "max_clip_value": 10,
                "n_rnn": 2,
                "n_linear_layers": 2,
                "rnn_type": "rnn",
                "tau": 5,
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 32),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "n_conv": 2,
                "kernel_size": 3,
                "stride": 1,
                "in_features": 8,
                "hidden_size": 32,
                "bidirectional": False,
                "max_clip_value": 10,
                "n_rnn": 2,
                "n_linear_layers": 2,
                "rnn_type": "rnn",
                "tau": 5,
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 6, 32),
            torch.LongTensor([6, 4, 6]),
        ),
        (
            {
                "n_conv": 2,
                "kernel_size": 3,
                "stride": 1,
                "in_features": 8,
                "hidden_size": 32,
                "bidirectional": True,
                "max_clip_value": 10,
                "n_rnn": 2,
                "n_linear_layers": 2,
                "rnn_type": "rnn",
                "tau": 5,
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 6, 32),
            torch.LongTensor([6, 4, 6]),
        ),
    )

    @encoder_paramterizer(test_cases=test_cases)
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        feat_size,
        pad_lens,
        expected_shape,
        expected_lens,
    ):
        self.check(
            batcher,
            model_args,
            batch_size,
            seq_len,
            feat_size,
            pad_lens,
            expected_shape,
            expected_lens,
        )


class TestConformerEncoder(BaseTest):
    model = encoders.ConformerEncoder
    test_cases = (
        (
            {
                "d_model": 8,
                "n_conf_layers": 2,
                "ff_expansion_factor": 2,
                "h": 2,
                "kernel_size": 4,
                "ss_kernel_size": 1,
                "ss_stride": 1,
                "ss_num_conv_layers": 2,
                "in_features": 16,
                "res_scaling": 0.5,
                "p_dropout": 0.05,
            },
            3,
            10,
            16,
            [0, 6, 3],
            (3, 10, 8),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "d_model": 8,
                "n_conf_layers": 2,
                "ff_expansion_factor": 2,
                "h": 2,
                "kernel_size": 4,
                "ss_kernel_size": 3,
                "ss_stride": 1,
                "ss_num_conv_layers": 2,
                "in_features": 16,
                "res_scaling": 0.5,
                "p_dropout": 0.05,
            },
            3,
            10,
            16,
            [0, 6, 3],
            (3, 6, 8),
            torch.LongTensor([6, 4, 6]),
        ),
        (
            {
                "d_model": 8,
                "n_conf_layers": 2,
                "ff_expansion_factor": 2,
                "h": 2,
                "kernel_size": 4,
                "ss_kernel_size": 3,
                "ss_stride": 2,
                "ss_num_conv_layers": 1,
                "in_features": 16,
                "res_scaling": 0.5,
                "p_dropout": 0.05,
            },
            3,
            10,
            16,
            [0, 6, 3],
            (3, 4, 8),
            torch.LongTensor([4, 2, 4]),
        ),
    )

    @pytest.mark.filterwarnings(IGNORE_USERWARNING)
    @encoder_paramterizer(test_cases=test_cases)
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        feat_size,
        pad_lens,
        expected_shape,
        expected_lens,
    ):
        self.check(
            batcher,
            model_args,
            batch_size,
            seq_len,
            feat_size,
            pad_lens,
            expected_shape,
            expected_lens,
        )


class TestJasperEncoder(BaseTest):
    model = encoders.JasperEncoder
    test_cases = (
        (
            {
                "in_features": 8,
                "num_blocks": 2,
                "num_sub_blocks": 3,
                "channel_inc": 4,
                "epilog_kernel_size": 2,
                "prelog_kernel_size": 1,
                "prelog_stride": 1,
                "prelog_n_channels": 4,
                "blocks_kernel_size": 3,
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 20),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "num_blocks": 2,
                "num_sub_blocks": 3,
                "channel_inc": 4,
                "epilog_kernel_size": 2,
                "prelog_kernel_size": 3,
                "prelog_stride": 2,
                "prelog_n_channels": 4,
                "blocks_kernel_size": 3,
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 4, 20),
            torch.LongTensor([4, 2, 4]),
        ),
        (
            {
                "in_features": 8,
                "num_blocks": 1,
                "num_sub_blocks": 3,
                "channel_inc": 4,
                "epilog_kernel_size": 2,
                "prelog_kernel_size": 3,
                "prelog_stride": 2,
                "prelog_n_channels": 4,
                "blocks_kernel_size": 3,
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 4, 16),
            torch.LongTensor([4, 2, 4]),
        ),
    )

    @pytest.mark.filterwarnings(IGNORE_USERWARNING)
    @encoder_paramterizer(test_cases=test_cases)
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        feat_size,
        pad_lens,
        expected_shape,
        expected_lens,
    ):
        self.check(
            batcher,
            model_args,
            batch_size,
            seq_len,
            feat_size,
            pad_lens,
            expected_shape,
            expected_lens,
        )


class TestWav2LetterEncoder(BaseTest):
    model = encoders.Wav2LetterEncoder
    test_cases = (
        (
            {
                "in_features": 8,
                "n_conv_layers": 4,
                "layers_kernel_size": 3,
                "layers_channels_size": 16,
                "pre_conv_stride": 1,
                "pre_conv_kernel_size": 3,
                "post_conv_channels_size": 16,
                "post_conv_kernel_size": 3,
                "p_dropout": 0.04,
                "wav_kernel_size": None,
                "wav_stride": None,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 8, 16),
            torch.LongTensor([8, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "n_conv_layers": 4,
                "layers_kernel_size": 3,
                "layers_channels_size": 16,
                "pre_conv_stride": 1,
                "pre_conv_kernel_size": 1,
                "post_conv_channels_size": 16,
                "post_conv_kernel_size": 3,
                "p_dropout": 0.04,
                "wav_kernel_size": None,
                "wav_stride": None,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 1,
                "n_conv_layers": 4,
                "layers_kernel_size": 3,
                "layers_channels_size": 16,
                "pre_conv_stride": 1,
                "pre_conv_kernel_size": 1,
                "post_conv_channels_size": 16,
                "post_conv_kernel_size": 3,
                "p_dropout": 0.04,
                "wav_kernel_size": 4,
                "wav_stride": 1,
            },
            3,
            10,
            1,
            [0, 6, 3],
            (3, 7, 16),
            torch.LongTensor([7, 4, 7]),
        ),
    )

    @encoder_paramterizer(test_cases=test_cases)
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        feat_size,
        pad_lens,
        expected_shape,
        expected_lens,
    ):
        self.check(
            batcher,
            model_args,
            batch_size,
            seq_len,
            feat_size,
            pad_lens,
            expected_shape,
            expected_lens,
        )


class TestQuartzNetEncoder(BaseTest):
    model = encoders.QuartzNetEncoder
    test_cases = (
        (
            {
                "in_features": 8,
                "num_blocks": 2,
                "block_repetition": 2,
                "num_sub_blocks": 3,
                "channels_size": [8, 16],
                "epilog_kernel_size": 2,
                "epilog_channel_size": (8, 8),
                "groups": 4,
                "prelog_kernel_size": 1,
                "prelog_stride": 1,
                "prelog_n_channels": 4,
                "blocks_kernel_size": 3,
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 8),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "num_blocks": 2,
                "block_repetition": 2,
                "num_sub_blocks": 3,
                "channels_size": [8, 16],
                "epilog_kernel_size": 2,
                "epilog_channel_size": (8, 16),
                "groups": 4,
                "prelog_kernel_size": 1,
                "prelog_stride": 1,
                "prelog_n_channels": 4,
                "blocks_kernel_size": 3,
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "num_blocks": 2,
                "block_repetition": 2,
                "num_sub_blocks": 3,
                "channels_size": [8, 16],
                "epilog_kernel_size": 2,
                "epilog_channel_size": (8, 16),
                "groups": 4,
                "prelog_kernel_size": 3,
                "prelog_stride": 1,
                "prelog_n_channels": 4,
                "blocks_kernel_size": 3,
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 8, 16),
            torch.LongTensor([8, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "num_blocks": 2,
                "block_repetition": 2,
                "num_sub_blocks": 3,
                "channels_size": [8, 16],
                "epilog_kernel_size": 2,
                "epilog_channel_size": (8, 16),
                "groups": 4,
                "prelog_kernel_size": 3,
                "prelog_stride": 2,
                "prelog_n_channels": 4,
                "blocks_kernel_size": 3,
                "p_dropout": 0.01,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 4, 16),
            torch.LongTensor([4, 2, 4]),
        ),
    )

    @pytest.mark.filterwarnings(IGNORE_USERWARNING)
    @encoder_paramterizer(test_cases=test_cases)
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        feat_size,
        pad_lens,
        expected_shape,
        expected_lens,
    ):
        self.check(
            batcher,
            model_args,
            batch_size,
            seq_len,
            feat_size,
            pad_lens,
            expected_shape,
            expected_lens,
        )


class TestSqueezeformerEncoder(BaseTest):
    model = encoders.SqueezeformerEncoder
    test_cases = (
        (
            {
                "in_features": 8,
                "n": 3,
                "d_model": 16,
                "ff_expansion_factor": 2,
                "h": 2,
                "kernel_size": 3,
                "pooling_kernel_size": 3,
                "pooling_stride": 1,
                "ss_kernel_size": 1,
                "ss_stride": 1,
                "ss_n_conv_layers": 1,
                "p_dropout": 0.01,
                "ss_groups": 8,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "n": 1,
                "d_model": 16,
                "ff_expansion_factor": 2,
                "h": 2,
                "kernel_size": 3,
                "pooling_kernel_size": 3,
                "pooling_stride": 1,
                "ss_kernel_size": 1,
                "ss_stride": 1,
                "ss_n_conv_layers": 1,
                "p_dropout": 0.01,
                "ss_groups": 8,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "n": 1,
                "d_model": 16,
                "ff_expansion_factor": 2,
                "h": 2,
                "kernel_size": 3,
                "pooling_kernel_size": 3,
                "pooling_stride": 1,
                "ss_kernel_size": 1,
                "ss_stride": 2,
                "ss_n_conv_layers": 1,
                "p_dropout": 0.01,
                "ss_groups": 8,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 5, 16),
            torch.LongTensor([5, 2, 4]),
        ),
        (
            {
                "in_features": 8,
                "n": 1,
                "d_model": 16,
                "ff_expansion_factor": 2,
                "h": 2,
                "kernel_size": 3,
                "pooling_kernel_size": 3,
                "pooling_stride": 1,
                "ss_kernel_size": 3,
                "ss_stride": 2,
                "ss_n_conv_layers": 1,
                "p_dropout": 0.01,
                "ss_groups": 8,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 4, 16),
            torch.LongTensor([4, 2, 4]),
        ),
    )

    @encoder_paramterizer(test_cases=test_cases)
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        feat_size,
        pad_lens,
        expected_shape,
        expected_lens,
    ):
        self.check(
            batcher,
            model_args,
            batch_size,
            seq_len,
            feat_size,
            pad_lens,
            expected_shape,
            expected_lens,
        )


class TestSpeechTransformerEncoder(BaseTest):
    model = encoders.SpeechTransformerEncoder
    test_cases = (
        (
            {
                "in_features": 8,
                "n_conv_layers": 1,
                "kernel_size": 3,
                "stride": 1,
                "d_model": 16,
                "n_layers": 2,
                "ff_size": 8,
                "h": 2,
                "att_kernel_size": 4,
                "att_out_channels": 16,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 8, 16),
            torch.LongTensor([8, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "n_conv_layers": 1,
                "kernel_size": 3,
                "stride": 1,
                "d_model": 16,
                "n_layers": 2,
                "ff_size": 8,
                "h": 2,
                "att_kernel_size": 1,
                "att_out_channels": 16,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 8, 16),
            torch.LongTensor([8, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "n_conv_layers": 2,
                "kernel_size": 3,
                "stride": 1,
                "d_model": 16,
                "n_layers": 2,
                "ff_size": 8,
                "h": 2,
                "att_kernel_size": 4,
                "att_out_channels": 16,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 6, 16),
            torch.LongTensor([6, 4, 6]),
        ),
    )

    @encoder_paramterizer(test_cases=test_cases)
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        feat_size,
        pad_lens,
        expected_shape,
        expected_lens,
    ):
        self.check(
            batcher,
            model_args,
            batch_size,
            seq_len,
            feat_size,
            pad_lens,
            expected_shape,
            expected_lens,
        )


class TestRNNEncoder(BaseTest):
    model = encoders.RNNEncoder
    test_cases = (
        (
            {
                "in_features": 8,
                "hidden_size": 16,
                "bidirectional": False,
                "n_layers": 2,
                "p_dropout": 0.01,
                "rnn_type": "rnn",
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 16,
                "bidirectional": False,
                "n_layers": 2,
                "p_dropout": 0.01,
                "rnn_type": "gru",
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 16,
                "bidirectional": False,
                "n_layers": 2,
                "p_dropout": 0.01,
                "rnn_type": "lstm",
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 16,
                "bidirectional": True,
                "n_layers": 2,
                "p_dropout": 0.01,
                "rnn_type": "rnn",
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 16,
                "bidirectional": True,
                "n_layers": 2,
                "p_dropout": 0.01,
                "rnn_type": "gru",
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 16,
                "bidirectional": True,
                "n_layers": 2,
                "p_dropout": 0.01,
                "rnn_type": "lstm",
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
    )

    @encoder_paramterizer(test_cases=test_cases)
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        feat_size,
        pad_lens,
        expected_shape,
        expected_lens,
    ):
        self.check(
            batcher,
            model_args,
            batch_size,
            seq_len,
            feat_size,
            pad_lens,
            expected_shape,
            expected_lens,
        )


class TestPyramidRNNEncoder(BaseTest):
    model = encoders.PyramidRNNEncoder
    test_cases = (
        (
            {
                "in_features": 8,
                "hidden_size": 16,
                "reduction_factor": 1,
                "bidirectional": False,
                "n_layers": 2,
                "p_dropout": 0.01,
                "rnn_type": "rnn",
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 16,
                "reduction_factor": 2,
                "bidirectional": False,
                "n_layers": 2,
                "p_dropout": 0.01,
                "rnn_type": "rnn",
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 5, 16),
            torch.LongTensor([5, 2, 4]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 16,
                "reduction_factor": 3,
                "bidirectional": False,
                "n_layers": 2,
                "p_dropout": 0.01,
                "rnn_type": "rnn",
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 4, 16),
            torch.LongTensor([4, 2, 3]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 16,
                "reduction_factor": 1,
                "bidirectional": False,
                "n_layers": 2,
                "p_dropout": 0.01,
                "rnn_type": "lstm",
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 16,
                "reduction_factor": 1,
                "bidirectional": False,
                "n_layers": 2,
                "p_dropout": 0.01,
                "rnn_type": "gru",
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 16,
                "reduction_factor": 1,
                "bidirectional": True,
                "n_layers": 2,
                "p_dropout": 0.01,
                "rnn_type": "rnn",
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 16,
                "reduction_factor": 1,
                "bidirectional": True,
                "n_layers": 2,
                "p_dropout": 0.01,
                "rnn_type": "lstm",
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "hidden_size": 16,
                "reduction_factor": 1,
                "bidirectional": True,
                "n_layers": 2,
                "p_dropout": 0.01,
                "rnn_type": "gru",
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
    )

    @encoder_paramterizer(test_cases=test_cases)
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        feat_size,
        pad_lens,
        expected_shape,
        expected_lens,
    ):
        self.check(
            batcher,
            model_args,
            batch_size,
            seq_len,
            feat_size,
            pad_lens,
            expected_shape,
            expected_lens,
        )


class TestContextNetEncoder(BaseTest):
    model = encoders.ContextNetEncoder
    test_cases = (
        (
            {
                "in_features": 8,
                "n_layers": 2,
                "n_sub_layers": 3,
                "stride": 1,
                "out_channels": 16,
                "kernel_size": 2,
                "reduction_factor": 4,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
    )

    @encoder_paramterizer(test_cases=test_cases)
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        feat_size,
        pad_lens,
        expected_shape,
        expected_lens,
    ):
        self.check(
            batcher,
            model_args,
            batch_size,
            seq_len,
            feat_size,
            pad_lens,
            expected_shape,
            expected_lens,
        )


class TestVGGTransformerEncoder(BaseTest):
    model = encoders.VGGTransformerEncoder
    test_cases = (
        (
            {
                "in_features": 8,
                "n_layers": 2,
                "n_vgg_blocks": 1,
                "n_conv_layers_per_vgg_block": [
                    2,
                ],
                "kernel_sizes_per_vgg_block": [[4, 8]],
                "n_channels_per_vgg_block": [[8, 4]],
                "vgg_pooling_kernel_size": [
                    1,
                ],
                "d_model": 16,
                "ff_size": 8,
                "h": 2,
                "left_size": 2,
                "right_size": 2,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "n_layers": 2,
                "n_vgg_blocks": 1,
                "n_conv_layers_per_vgg_block": [
                    2,
                ],
                "kernel_sizes_per_vgg_block": [[4, 8]],
                "n_channels_per_vgg_block": [[8, 4]],
                "vgg_pooling_kernel_size": [
                    2,
                ],
                "d_model": 16,
                "ff_size": 8,
                "h": 2,
                "left_size": 2,
                "right_size": 2,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 5, 16),
            torch.LongTensor([5, 2, 3]),
        ),
    )

    @encoder_paramterizer(test_cases=test_cases)
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        feat_size,
        pad_lens,
        expected_shape,
        expected_lens,
    ):
        self.check(
            batcher,
            model_args,
            batch_size,
            seq_len,
            feat_size,
            pad_lens,
            expected_shape,
            expected_lens,
        )


class TestTransformerTransducerEncoder(BaseTest):
    model = encoders.TransformerTransducerEncoder
    test_cases = (
        (
            {
                "in_features": 8,
                "n_layers": 2,
                "d_model": 16,
                "ff_size": 8,
                "h": 2,
                "left_size": 2,
                "right_size": 2,
                "p_dropout": 0.05,
                "stride": 1,
                "kernel_size": 1,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 10, 16),
            torch.LongTensor([10, 4, 7]),
        ),
        (
            {
                "in_features": 8,
                "n_layers": 2,
                "d_model": 16,
                "ff_size": 8,
                "h": 2,
                "left_size": 2,
                "right_size": 2,
                "p_dropout": 0.05,
                "stride": 2,
                "kernel_size": 1,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 5, 16),
            torch.LongTensor([5, 2, 4]),
        ),
        (
            {
                "in_features": 8,
                "n_layers": 2,
                "d_model": 16,
                "ff_size": 8,
                "h": 2,
                "left_size": 2,
                "right_size": 2,
                "p_dropout": 0.05,
                "stride": 2,
                "kernel_size": 2,
            },
            3,
            10,
            8,
            [0, 6, 3],
            (3, 5, 16),
            torch.LongTensor([5, 2, 4]),
        ),
    )

    @encoder_paramterizer(test_cases=test_cases)
    def test_forward(
        self,
        batcher,
        model_args,
        batch_size,
        seq_len,
        feat_size,
        pad_lens,
        expected_shape,
        expected_lens,
    ):
        self.check(
            batcher,
            model_args,
            batch_size,
            seq_len,
            feat_size,
            pad_lens,
            expected_shape,
            expected_lens,
        )
