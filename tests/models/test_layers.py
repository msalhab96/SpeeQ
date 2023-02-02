import pytest
import torch
from torch import LongTensor

from speeq.models import layers
from tests.helpers import check_grad


class TestPackedRNN:

    @pytest.mark.parametrize(
        (
            'inp_size',
            'seq_len',
            'batch_size',
            'hidden_size',
            'bidirectional',
            'batch_first',
            'expected',
            'lengths'
        ),
        (
            (8, 15, 3, 32, False, True, (3, 15, 32), LongTensor([8, 15, 1])),
            (8, 15, 3, 32, False, True, (3, 7, 32), LongTensor([2, 7, 1])),
            (8, 15, 3, 32, True, True, (3, 15, 64), LongTensor([15, 6, 10])),
            (8, 10, 3, 32, True, True, (3, 10, 64), LongTensor([3, 6, 10])),
            (8, 15, 3, 32, True, False, (15, 3, 64), LongTensor([8, 2, 15])),
            (8, 15, 3, 32, True, False, (9, 3, 64), LongTensor([8, 2, 9])),
            (8, 15, 3, 32, False, False, (1, 3, 32), LongTensor([1, 1, 1])),
        )
    )
    def test_shape(
            self, inp_size, seq_len, batch_size,
            hidden_size, bidirectional, batch_first,
            expected, lengths
            ):
        """Tests the shape and the length returned.
        """
        batch = torch.randn(batch_size, seq_len, inp_size)
        if batch_first is False:
            batch.transpose_(0, 1)
        model = layers.PackedRNN(
            input_size=inp_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
        out, _, result_lengths = model(batch, lengths)
        assert out.shape == expected
        assert torch.all(result_lengths == lengths)


class TestPredModule:
    parameters_mark = pytest.mark.parametrize(
        ('in_features', 'n_classes', 'batch_size', 'seq_len'),
        (
            (10, 20, 1, 5),
            (10, 20, 3, 5),
            (10, 1, 3, 5),
            (1, 20, 3, 5),
        )
    )
    module = layers.PredModule

    @parameters_mark
    def test_model(
            self, in_features, n_classes, batch_size, seq_len
            ):
        """Tests the shape and the gradients of the model
        """
        expected = (batch_size, seq_len, n_classes)
        input = torch.randn(batch_size, seq_len, in_features)
        model = self.module(
            in_features=in_features,
            n_classes=n_classes,
            activation=lambda x: x
        )
        result = model(input)
        shape = result.shape
        assert shape == expected
        check_grad(result, model)


class TestConvPredModule(TestPredModule):
    module = layers.ConvPredModule


class TestFeedForwardModule:
    @pytest.mark.parametrize(
        ('d_model', 'ff_size', 'batch_size', 'seq_len', 'feat_size'),
        (
            (16, 32, 3, 5, 16),
            (32, 32, 3, 5, 32),
            (32, 32, 3, 1, 32),
            (32, 16, 1, 1, 32),
        )
    )
    def test_model(
            self, batcher, d_model, ff_size, batch_size, seq_len, feat_size
            ):
        """Tests the shape and the gradients of the model
        """
        expected = (batch_size, seq_len, d_model)
        batch = batcher(batch_size, seq_len, feat_size)
        layer = layers.FeedForwardModule(d_model=d_model, ff_size=ff_size)
        result = layer(batch)
        assert result.shape == expected
        check_grad(result, layer)


class TestAddAndNorm:

    @pytest.mark.parametrize(
        ('batch_size', 'seq_len', 'd_model'),
        (
            (1, 16, 8),
            (1, 1, 16),
            (3, 8, 4),
        )
    )
    def test_model(self, batcher, batch_size, seq_len, d_model):
        """Tests the shape and the gradients of the model
        """
        model = layers.AddAndNorm(d_model=d_model)
        shape = (batch_size, seq_len, d_model)
        x = batcher(*shape)
        sub_x = batcher(batch_size, seq_len, d_model)
        result = model(x, sub_x)
        assert result.shape == shape
        check_grad(result=result, model=model)


class TestMultiHeadAtt:

    model_args = ('d_model', 'h', 'masking_value')
    masking_input = torch.tensor([
        [
            [
                [0.7, 0.3, 1],
                [1.0, 2.3, 3.01]
            ],
            [
                [0.5, 0.5, 0.005],
                [1, 7.2, 7.2],
            ],
            [
                [0.5, 0.5, 0.1],
                [1, 0.1, 3.2],
            ],
            [
                [0.99, 0.01, 0.045],
                [1, 2.5, 2.5],
            ]
        ],
        [
            [
                [0.01, 0.3, 0.69],
                [8.5, 0.0508, 0.6]
            ],
            [
                [0.5, 0.5, 5],
                [0.25, 0.25, 0.25],
            ],
            [
                [0.3, 0.6, 0.1],
                [3.2, 8.1, 0.995],
            ],
            [
                [0.3, 0.3, 0.4],
                [0.995, 0.995, 0.995],
            ]
        ]
    ])
    masking_expected = torch.tensor([
        [
            [
                [0.7, 0.3, -1e15],
                [1.0, 2.3, -1e15]
            ],
            [
                [0.5, 0.5, -1e15],
                [1, 7.2, -1e15],
            ],
            [
                [0.5, 0.5, -1e15],
                [1, 0.1, -1e15],
            ],
            [
                [0.99, 0.01, -1e15],
                [1, 2.5, -1e15],
            ]
        ],
        [
            [
                [0.01, -1e15, -1e15],
                [-1e15, -1e15, -1e15]
            ],
            [
                [0.5, -1e15, -1e15],
                [-1e15, -1e15, -1e15],
            ],
            [
                [0.3, -1e15, -1e15],
                [-1e15, -1e15, -1e15],
            ],
            [
                [0.3, -1e15, -1e15],
                [-1e15, -1e15, -1e15],
            ]
        ]
    ])
    encoder_mask = torch.BoolTensor([
        [True, True, False],
        [True, False, False],
    ])

    decoder_mask = torch.BoolTensor([
        [True, True],
        [True, False]
    ])

    @pytest.mark.parametrize(
        (*model_args, 'batch_size', 'seq_len'),
        (
            (24, 6, -1e15, 3, 4),
            (32, 8, -1e15, 3, 1),
            (24, 1, -1e15, 3, 2),
        )
    )
    def test_reshape(
            self, batcher, d_model, h, masking_value, batch_size, seq_len
            ):
        """Tests the ._reshape function
        """
        model = layers.MultiHeadAtt(d_model, h, masking_value)
        expected = (batch_size, seq_len, h, d_model // h)
        x = batcher(batch_size, seq_len, d_model)
        result = model._reshape(x)
        assert result.shape == expected

    @pytest.mark.parametrize(
        (*model_args, 'att', 'key_mask', 'query_mask', 'expected'),
        (
            (
                24, 6, -1e15,
                masking_input,
                encoder_mask,
                decoder_mask,
                masking_expected
            ),
        )
    )
    def test_mask(
            self, d_model, h, masking_value, att,
            key_mask, query_mask, expected
            ):
        """Tests ._mask function
        """
        model = layers.MultiHeadAtt(d_model, h, masking_value)
        result = model._mask(att=att, key_mask=key_mask, query_mask=query_mask)
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize(
        (*model_args, 'batch_size', 'max_enc_len', 'max_dec_len'),
        (
            (24, 6, -1e15, 3, 5, 4),
            (24, 6, -1e15, 3, 2, 4),
            (32, 8, -1e15, 3, 2, 4),
            (32, 8, -1e15, 3, 2, 1),
        )
    )
    def test_with_no_mask(
            self, batcher, d_model, h, masking_value,
            batch_size, max_enc_len, max_dec_len
            ):
        """Tests the layer shape and gradients under no masking condition
        """
        expected = (batch_size, max_dec_len, d_model)
        query = batcher(batch_size, max_dec_len, d_model)
        value = batcher(batch_size, max_enc_len, d_model)
        key = batcher(batch_size, max_enc_len, d_model)
        model = layers.MultiHeadAtt(d_model, h, masking_value)
        result = model(key=key, value=value, query=query)
        assert result.shape == expected
        check_grad(result=result, model=model)

    @pytest.mark.parametrize(
        (*model_args, 'encoder_mask', 'decoder_mask'),
        (
            (24, 2, -1e15, encoder_mask, decoder_mask),
        )
    )
    def test_with_masks(
            self, batcher, d_model, h, masking_value,
            encoder_mask, decoder_mask
            ):
        """Tests the layer shape and gradients under masking condition
        """
        expected = (*decoder_mask.shape, d_model)
        key = batcher(*encoder_mask.shape, d_model)
        value = batcher(*encoder_mask.shape, d_model)
        query = batcher(*decoder_mask.shape, d_model)
        model = layers.MultiHeadAtt(d_model, h, masking_value)
        result = model(
            key=key, query=query, value=value,
            key_mask=encoder_mask, query_mask=decoder_mask
            )
        assert result.shape == expected
        check_grad(result=result, model=model)
