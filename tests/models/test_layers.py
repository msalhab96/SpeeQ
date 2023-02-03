import pytest
import torch
from torch import LongTensor

from speeq.models import layers
from tests.helpers import IGNORE_USERWARNING, check_grad, get_mask


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
    def test_forward(
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
    def test_forward(
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
    def test_forward(self, batcher, batch_size, seq_len, d_model):
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


class TestMaskedMultiHeadAtt:

    key_mask1 = torch.BoolTensor([
        [True, True, False],
        [True, True, True],
    ])
    expected1 = torch.BoolTensor([
        [
            [True, False, False],
            [True, True, False],
            [False, False, False]
        ],
        [
            [True, False, False],
            [True, True, False],
            [True, True, True]
        ]
    ])
    key_mask2 = torch.BoolTensor([
        [True, True, False, False]
    ])
    expected2 = torch.BoolTensor([
        [
            [True, False, False, False],
            [True, True, False, False],
            [False, False, False, False],
            [False, False, False, False]
        ]
    ])

    @pytest.mark.parametrize(
        ('d_model', 'h', 'key_mask', 'expected'),
        (
            (12, 2, key_mask1, expected1),
            (12, 2, key_mask2, expected2)
        )
    )
    def test_get_looking_ahead_mask(self, d_model, h, key_mask, expected):
        """Test the functionality of .get_looking_ahead_mask function
        """
        model = layers.MaskedMultiHeadAtt(d_model=d_model, h=h)
        result = model.get_looking_ahead_mask(key_mask)
        print(result)
        print(expected)
        assert torch.all(result == expected).item()


class TestTransformerEncLayer:

    @pytest.mark.parametrize(
        ('d_model', 'ff_size', 'h', 'batch_size', 'seq_len'),
        (
            (24, 26, 2, 3, 8),
            (24, 26, 4, 3, 1),
            (32, 35, 8, 3, 7),
        )
    )
    def test_forward(self, batcher, d_model, ff_size, h, batch_size, seq_len):
        """Tests the returned shape and the gradients of the model's forward
        """
        model = layers.TransformerEncLayer(
            d_model=d_model, ff_size=ff_size, h=h
        )
        shape = (batch_size, seq_len, d_model)
        input = batcher(*shape)
        result = model(input)
        assert result.shape == shape
        check_grad(result=result, model=model)


class TestRowConv1D:
    input_parametrize = pytest.mark.parametrize(
        ('tau', 'feat_size', 'batch_size', 'seq_len'),
        (
            (5, 12, 3, 9),
            (5, 12, 3, 5),
            (5, 12, 3, 2),
        )
    )

    @input_parametrize
    def test_pad(self, batcher, tau, feat_size, batch_size, seq_len):
        """Tests the ._pad function
        """
        model = layers.RowConv1D(tau=tau, feat_size=feat_size)
        # transposing seq_le with feat_size
        expected_shape = (batch_size, feat_size, seq_len + tau)
        batch = batcher(batch_size, feat_size, seq_len)
        result = model._pad(batch)
        assert result.shape == expected_shape

    @input_parametrize
    def test_forward(self, batcher, tau, feat_size, batch_size, seq_len):
        """Tests the returned shape and the gradients of the model's forward
        """
        model = layers.RowConv1D(tau=tau, feat_size=feat_size)
        shape = (batch_size, seq_len, feat_size)
        batch = batcher(*shape)
        result = model(batch)
        assert result.shape == shape
        check_grad(result=result, model=model)


class TestConv1DLayers:

    model_args = (
        'in_size', 'out_size', 'kernel_size', 'stride', 'n_layers', 'p_dropout'
    )

    @pytest.mark.parametrize(
        (
            *model_args, 'batch_size', 'seq_len', 'lengths',
            'expected_len', 'expected_shape'
        ),
        (
            (
                12, 16, 2, 1, 3, 0.0, 3, 10,
                torch.LongTensor([10, 8, 5]),
                torch.LongTensor([7, 7, 5]),
                (3, 7, 16)
            ),
            (
                12, 16, 3, 1, 3, 0.0, 3, 10,
                torch.LongTensor([10, 8, 5]),
                torch.LongTensor([4, 4, 4]),
                (3, 4, 16)
            ),
            (
                12, 16, [1, 2, 3], 1, 3, 0.0, 3, 10,
                torch.LongTensor([10, 8, 5]),
                torch.LongTensor([7, 7, 5]),
                (3, 7, 16)
            ),
            (
                12, [5, 7, 16], [1, 2, 3], 1, 3, 0.0, 3, 10,
                torch.LongTensor([10, 8, 5]),
                torch.LongTensor([7, 7, 5]),
                (3, 7, 16)
            ),
            (
                12, [5, 7, 16], [1, 2, 3], [1, 1, 1], 3, 0.0, 3, 10,
                torch.LongTensor([10, 8, 5]),
                torch.LongTensor([7, 7, 5]),
                (3, 7, 16)
            ),
            (
                12, [5, 7, 16], [1, 2, 1], [1, 2, 1], 3, 0.0, 3, 10,
                torch.LongTensor([10, 8, 5]),
                torch.LongTensor([5, 4, 3]),
                (3, 5, 16)
            )
        )
    )
    def test_forward(
            self, batcher, in_size, out_size, kernel_size, stride,
            n_layers, p_dropout, batch_size, seq_len, lengths,
            expected_len, expected_shape
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        batch = batcher(batch_size, seq_len, in_size)
        model = layers.Conv1DLayers(
            in_size=in_size, out_size=out_size, kernel_size=kernel_size,
            stride=stride, n_layers=n_layers, p_dropout=p_dropout
        )
        result, lengths = model(batch, lengths)
        print(lengths.dtype)
        assert lengths.dtype in [torch.int32, torch.int64, torch.long]
        assert result.shape == expected_shape
        assert torch.all(lengths == expected_len).item()
        check_grad(result=result, model=model)


class TestGlobalMulAttention:
    @pytest.mark.parametrize(
        (
            'enc_size', 'dec_size', 'scaling_factor',
            'batch_size', 'seq_len', 'pad_lens'
        ),
        (
            (8, 16, 1, 2, 7, [0, 3]),
            (8, 16, 1, 2, 7, None),
            (16, 16, 0.1, 1, 9, [4]),
        )
    )
    def test_forward(
            self, batcher, enc_size, dec_size,
            scaling_factor, batch_size, seq_len, pad_lens
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        shape = (batch_size, 1, dec_size)
        key = batcher(batch_size, seq_len, enc_size)
        query = batcher(*shape)
        model = layers.GlobalMulAttention(
            enc_feat_size=enc_size, dec_feat_size=dec_size,
            scaling_factor=scaling_factor
        )
        mask = None
        if pad_lens is not None:
            mask = [[1] * (seq_len - item) + [0] * item for item in pad_lens]
            mask = torch.BoolTensor(mask)
        result = model(key=key, query=query, mask=mask)
        assert result.shape == shape
        check_grad(result=result, model=model)


class TestConformerFeedForward:
    @pytest.mark.parametrize(
        ('d_model', 'expansion_factor', 'p_dropout', 'batch_size', 'seq_len'),
        (
            (16, 2, 0.0, 2, 3),
            (16, 1, 0.1, 2, 3),
            (16, 3, 0.5, 1, 7),
        )
    )
    def test_forward(
            self, batcher, d_model, expansion_factor,
            p_dropout, batch_size, seq_len
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        shape = (batch_size, seq_len, d_model)
        input = batcher(*shape)
        model = layers.ConformerFeedForward(
            d_model=d_model,
            expansion_factor=expansion_factor,
            p_dropout=p_dropout
        )
        result = model(input)
        assert result.shape == shape
        check_grad(result=result, model=model)


class TestConformerConvModule:
    @pytest.mark.filterwarnings(IGNORE_USERWARNING)
    @pytest.mark.parametrize(
        ('d_model', 'kernel_size', 'p_dropout', 'seq_len', 'batch_size'),
        (
            (16, 8, 0.1, 8, 4),
            (8, 3, 0.1, 15, 2),
            (8, 3, 0.0, 15, 1),
        )
    )
    def test_forward(
            self, batcher, d_model, kernel_size, p_dropout, seq_len, batch_size
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        shape = (batch_size, seq_len, d_model)
        input = batcher(*shape)
        model = layers.ConformerConvModule(
            d_model=d_model, kernel_size=kernel_size, p_dropout=p_dropout
        )
        result = model(input)
        assert result.shape == shape
        check_grad(model=model, result=result)


class TestConformerRelativeMHSA:
    @pytest.mark.parametrize(
        ('d_model', 'h', 'p_dropout', 'batch_size', 'seq_len', 'pad_len'),
        (
            (16, 4, 0.0, 3, 10, [4, 5, 10]),
            (16, 4, 0.0, 3, 10, None)
        )
    )
    def test_forward(
            self, batcher, d_model, h, p_dropout,
            batch_size, seq_len, pad_len
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        shape = (batch_size, seq_len, d_model)
        input = batcher(*shape)
        model = layers.ConformerRelativeMHSA(
            d_model=d_model, h=h, p_dropout=p_dropout
        )
        mask = None
        if pad_len is not None:
            mask = [[1] * (seq_len - item) + [0] * item for item in pad_len]
            mask = torch.BoolTensor(mask)
        result = model(input, mask=mask)
        assert result.shape == shape
        check_grad(result=result, model=model)


class TestConformerBlock:
    @pytest.mark.parametrize(
        (
            'd_model', 'ff_expansion_factor',
            'h', 'kernel_size', 'p_dropout',
            'batch_size', 'seq_len', 'pad_len'
        ),
        (
            (16, 2, 4, 5, 0.0, 3, 10, [4, 5, 10]),
            (16, 2, 4, 1, 0.0, 3, 10, [7, 5, 10]),
            (16, 2, 4, 10, 0.0, 3, 10, [7, 5, 10]),
            (32, 2, 2, 10, 0.0, 3, 10, [7, 5, 10]),
            (16, 2, 2, 10, 0.0, 3, 10, None),
            (64, 2, 2, 10, 0.0, 3, 10, None),
        )
    )
    def test_forward(
            self, batcher, d_model, ff_expansion_factor, h, kernel_size,
            p_dropout, batch_size, seq_len, pad_len
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        shape = (batch_size, seq_len, d_model)
        input = batcher(*shape)
        model = layers.ConformerBlock(
            d_model=d_model,
            ff_expansion_factor=ff_expansion_factor,
            h=h,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )
        mask = None
        if pad_len is not None:
            mask = [[1] * (seq_len - item) + [0] * item for item in pad_len]
            mask = torch.BoolTensor(mask)
        result = model(input, mask=mask)
        assert result.shape == shape
        check_grad(result=result, model=model)


class TestConformerPreNet:
    @pytest.mark.parametrize(
        (
            'in_features', 'kernel_size', 'stride',
            'n_conv_layers', 'd_model', 'p_dropout', 'groups',
            'batch_size', 'seq_len', 'lengths', 'expected_lengths',
            'expected_shape'
        ),
        (
            (
                8, 1, 1, 2, 16, 0.1, 1, 3, 10,
                torch.LongTensor([10, 7, 8]),
                torch.LongTensor([10, 7, 8]),
                (3, 10, 16)
            ),
            (
                8, 2, 1, 2, 16, 0.1, 1, 3, 10,
                torch.LongTensor([10, 7, 8]),
                torch.LongTensor([8, 7, 8]),
                (3, 8, 16)
            ),
            (
                8, [1, 2], 1, 2, 16, 0.1, 1, 3, 10,
                torch.LongTensor([10, 7, 8]),
                torch.LongTensor([9, 7, 8]),
                (3, 9, 16)
            ),
            (
                8, [1, 2], [2, 1], 2, 16, 0.1, 1, 3, 10,
                torch.LongTensor([10, 7, 8]),
                torch.LongTensor([4, 4, 4]),
                (3, 4, 16)
            ),
            (
                8, [1, 2], [2, 2], 2, 16, 0.1, 1, 3, 10,
                torch.LongTensor([10, 7, 8]),
                torch.LongTensor([2, 2, 2]),
                (3, 2, 16)
            ),
        )
    )
    def test_forward(
            self, batcher, in_features, kernel_size, stride,
            n_conv_layers, d_model, p_dropout, groups,
            batch_size, seq_len, lengths, expected_lengths,
            expected_shape
    ):
        """Tests the returned shape and the gradients of the model's forward
        and the returened lentghs as well
        """
        input = batcher(batch_size, seq_len, in_features)
        model = layers.ConformerPreNet(
            in_features=in_features,
            kernel_size=kernel_size,
            stride=stride,
            n_conv_layers=n_conv_layers,
            d_model=d_model,
            p_dropout=p_dropout,
            groups=groups
        )
        result, lengths = model(input, lengths)
        assert result.shape == expected_shape
        assert torch.all(lengths == expected_lengths).item()
        check_grad(result=result, model=model)


class TestJasperSubBlock:
    @pytest.mark.parametrize(
        (
            'in_channels', 'out_channels', 'kernel_size',
            'p_dropout', 'stride', 'padding', 'batch_size', 'seq_len',
            'add_residual'
        ),
        (
            (16, 8, 4, 0.01, 1, 'same', 3, 10, True),
            (16, 8, 4, 0.01, 1, 'same', 3, 10, False),
        )
    )
    def test_forward(
            self, batcher, in_channels, out_channels, kernel_size,
            p_dropout, stride, padding, batch_size, seq_len, add_residual
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        expected_shape = (batch_size, out_channels, seq_len)
        input = batcher(batch_size, in_channels, seq_len)
        model = layers.JasperSubBlock(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, p_dropout=p_dropout,
            stride=stride, padding=padding
        )
        residual = None
        if add_residual:
            residual = batcher(batch_size, out_channels, seq_len)
        result = model(input, residual)
        assert result.shape == expected_shape
        check_grad(result=result, model=model)


class TestJasperResidual:
    @pytest.mark.parametrize(
        ('in_channels', 'out_channels', 'seq_len', 'batch_size'),
        (
            (8, 16, 5, 3),
            (8, 16, 5, 1),
            (16, 16, 5, 3),
        )
    )
    def test_forward(
            self, batcher, in_channels, out_channels, seq_len, batch_size
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        input = batcher(batch_size, in_channels, seq_len)
        expected_shape = (batch_size, out_channels, seq_len)
        model = layers.JasperResidual(
            in_channels=in_channels, out_channels=out_channels
        )
        result = model(input)
        assert result.shape == expected_shape
        check_grad(result=result, model=model)


class TestJasperBlock:
    @pytest.mark.parametrize(
        (
            'num_sub_blocks', 'in_channels', 'out_channels',
            'kernel_size', 'p_dropout', 'batch_size', 'seq_len'
        ),
        (
            (3, 8, 16, 4, 0.0, 3, 10),
            (1, 8, 16, 4, 0.0, 3, 10),
            (1, 8, 16, 10, 0.0, 3, 10),
        )
    )
    def test_forward(
            self, batcher, num_sub_blocks, in_channels, out_channels,
            kernel_size, p_dropout, batch_size, seq_len
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        input = batcher(batch_size, in_channels, seq_len)
        expected_shape = (batch_size, out_channels, seq_len)
        model = layers.JasperBlock(
            num_sub_blocks=num_sub_blocks,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )
        result = model(input)
        assert result.shape == expected_shape
        check_grad(result=result, model=model)


class TestJasperBlocks:
    @pytest.mark.parametrize(
        (
            'num_blocks', 'num_sub_blocks', 'in_channels', 'channel_inc',
            'kernel_size', 'p_dropout', 'batch_size', 'seq_len'
        ),
        (
            (1, 3, 8, 16, 4, 0.0, 3, 10),
            (2, 1, 8, 16, 4, 0.0, 3, 10),
            (3, 1, 8, 16, 3, 0.0, 3, 10),
        )
    )
    def test_forward(
            self, batcher, num_blocks, num_sub_blocks, in_channels,
            channel_inc, kernel_size, p_dropout, batch_size, seq_len
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        out_channels = in_channels + channel_inc * num_blocks
        input = batcher(batch_size, in_channels, seq_len)
        expected_shape = (batch_size, out_channels, seq_len)
        model = layers.JasperBlocks(
            num_blocks=num_blocks,
            num_sub_blocks=num_sub_blocks,
            in_channels=in_channels,
            channel_inc=channel_inc,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )
        result = model(input)
        assert result.shape == expected_shape
        check_grad(result=result, model=model)


class TestLocAwareGlobalAddAttention:
    @pytest.mark.parametrize(
        (
            'enc_feat_size', 'dec_feat_size', 'kernel_size',
            'activation', 'inv_temperature', 'batch_size', 'seq_len'
        ),
        (
            (8, 16, 4, 'softmax', 1, 3, 10),
            (8, 16, 4, 'sigmax', 1, 3, 10),
            (16, 16, 8, 'sigmax', 1, 1, 10)
        )
    )
    def model_test(
            self, batcher, enc_feat_size, dec_feat_size,
            kernel_size, activation, inv_temperature,
            batch_size, seq_len
    ):
        """Tests the returned results and alpha shape and the gradients
        of the model's forward
        """
        expected_result_shape = (batch_size, 1, dec_feat_size)
        expected_alpha_shape = (batch_size, 1, seq_len)
        key = batcher(batch_size, seq_len, enc_feat_size)
        query = batcher(batch_size, 1, enc_feat_size)
        alpha = batcher(batch_size, 1, seq_len)
        model = layers.LocAwareGlobalAddAttention(
            enc_feat_size=enc_feat_size, dec_feat_size=dec_feat_size,
            kernel_size=kernel_size, activation=activation,
            inv_temperature=inv_temperature
        )
        result, alpha = model(key=key, query=query, alpha=alpha)
        assert result.shape == expected_result_shape
        assert alpha.shape == expected_alpha_shape
        check_grad(result=result, model=model)
        check_grad(result=alpha, model=model)


class TestMultiHeadAtt2d:
    @pytest.mark.parametrize(
        (
            'd_model', 'h', 'out_channels',
            'kernel_size', 'batch_size', 'seq_len', 'pad_lens'
        ),
        (
            (16, 4, 32, 4, 3, 10, [2, 0, 5]),
            (16, 4, 32, 4, 3, 10, None),
            (16, 2, 16, 4, 3, 10, None),
        )
    )
    def test_forward(
            self, batcher, d_model, h, out_channels,
            kernel_size, batch_size, seq_len, pad_lens
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        shape = (batch_size, seq_len, d_model)
        key = batcher(*shape)
        query = batcher(*shape)
        value = batcher(*shape)
        model = layers.MultiHeadAtt2d(
            d_model=d_model, h=h, out_channels=out_channels,
            kernel_size=kernel_size
        )
        mask = None
        if pad_lens is not None:
            mask = [[1] * (seq_len - item) + [0] * item for item in pad_lens]
            mask = torch.BoolTensor(mask)
        result = model(key=key, query=query, value=value, mask=mask)
        assert shape == result.shape
        check_grad(model=model, result=result)


class TestSpeechTransformerEncLayer:
    @pytest.mark.parametrize(
        (
            'd_model', 'ff_size', 'h', 'out_channels', 'kernel_size',
            'batch_size', 'seq_len', 'pad_lens'
        ),
        (
            (16, 32, 4, 16, 5, 3, 10, None),
            (16, 32, 4, 16, 5, 3, 10, [5, 3, 10]),
        )
    )
    def test_forward(
            self, batcher, d_model, ff_size, h,
            out_channels, kernel_size, batch_size,
            seq_len, pad_lens
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        shape = (batch_size, seq_len, d_model)
        input = batcher(*shape)
        model = layers.SpeechTransformerEncLayer(
            d_model=d_model, ff_size=ff_size,
            h=h, out_channels=out_channels,
            kernel_size=kernel_size
        )
        mask = None
        if pad_lens is not None:
            mask = [[1] * (seq_len - item) + [0] * item for item in pad_lens]
            mask = torch.BoolTensor(mask)
        result = model(input, mask=mask)
        assert shape == result.shape
        check_grad(model=model, result=result)


class TestTransformerDecLayer:
    @pytest.mark.parametrize(
        (
            'd_model', 'ff_size', 'h', 'enc_seq_len', 'dec_seq_len',
            'batch_size', 'enc_pad_lens', 'dec_pad_lens'
        ),
        (
            (16, 32, 2, 8, 3, 3, [3, 5, 8], [3, 2, 1]),
            (16, 32, 2, 8, 3, 3, None, [3, 2, 1]),
            (16, 32, 2, 8, 3, 3, [3, 5, 8], None),
            (16, 32, 2, 8, 3, 3, None, None),
            (16, 32, 2, 8, 3, 1, None, None),
            (16, 32, 2, 8, 3, 1, [8], [3]),
        )
    )
    def test_forward(
            self, batcher, d_model, ff_size, h, enc_seq_len,
            dec_seq_len, batch_size, enc_pad_lens, dec_pad_lens
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        shape = (batch_size, dec_seq_len, d_model)
        enc_mask = None
        dec_mask = None
        model = layers.TransformerDecLayer(
            d_model=d_model, ff_size=ff_size, h=h
        )
        if enc_pad_lens is not None:
            enc_mask = get_mask(enc_seq_len, enc_pad_lens)
        if dec_pad_lens is not None:
            dec_mask = get_mask(dec_seq_len, dec_pad_lens)
        enc = batcher(batch_size, enc_seq_len, d_model)
        dec = batcher(*shape)
        result = model(
            enc_out=enc, enc_mask=enc_mask, dec_inp=dec, dec_mask=dec_mask
        )
        assert result.shape == shape
        check_grad(result=result, model=model)


class TestPositionalEmbedding:
    @pytest.mark.parametrize(
        ('vocab_size', 'batch_size', 'embed_dim', 'seq_len'),
        (
            (10, 2, 8, 4),
            (10, 1, 8, 1),
        )
    )
    def test_forward(
            self, int_batcher, vocab_size, batch_size, embed_dim, seq_len
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        expected_shape = (batch_size, seq_len, embed_dim)
        model = layers.PositionalEmbedding(
            vocab_size=vocab_size, embed_dim=embed_dim
        )
        input = int_batcher(batch_size, seq_len, vocab_size)
        result = model(input)
        assert result.shape == expected_shape
        check_grad(result=result, model=model)


class TestGroupsShuffle:
    @pytest.mark.parametrize(
        ('groups', 'n_channels', 'batch_size', 'seq_len'),
        (
            (8, 16, 3, 5),
            (2, 32, 1, 3)
        )
    )
    def test_forward(self, batcher, groups, n_channels, batch_size, seq_len):
        """Tests the returned shape and the gradients of the model's forward
        """
        shape = (batch_size, n_channels, seq_len)
        input = batcher(*shape)
        model = layers.GroupsShuffle(groups=groups)
        result = model(input)
        assert result.shape == shape


class TestQuartzSubBlock:
    @pytest.mark.parametrize(
        (
            'in_channels', 'out_channels', 'kernel_size',
            'p_dropout', 'groups', 'batch_size', 'seq_len', 'add_residual'
        ),
        (
            (16, 8, 4, 0.01, 2, 3, 5, False),
            (16, 8, 4, 0.01, 2, 3, 5, True),
        )
    )
    def test_forward(
            self, batcher, in_channels, out_channels, kernel_size,
            p_dropout, groups, batch_size, seq_len, add_residual
    ):
        """Tests the returned shape and the gradients of the model's forward 
        """
        input = batcher(batch_size, in_channels, seq_len)
        expected_shape = (batch_size, out_channels, seq_len)
        model = layers.QuartzSubBlock(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, p_dropout=p_dropout, groups=groups
        )
        residual = None
        if add_residual:
            residual = batcher(batch_size, out_channels, seq_len)
        result = model(input, residual)
        assert result.shape == expected_shape
        check_grad(result=result, model=model)


class TestQuartzBlock:
    @pytest.mark.parametrize(
        (
            'num_sub_blocks', 'in_channels', 'out_channels',
            'kernel_size', 'groups', 'p_dropout', 'batch_size', 'seq_len'
        ),
        (
            (3, 8, 16, 4, 2, 0.0, 3, 10),
            (1, 8, 16, 4, 2, 0.0, 3, 10),
            (1, 8, 16, 10, 4, 0.0, 3, 10),
        )
    )
    def test_forward(
            self, batcher, num_sub_blocks, in_channels, out_channels,
            kernel_size, groups, p_dropout, batch_size, seq_len
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        input = batcher(batch_size, in_channels, seq_len)
        expected_shape = (batch_size, out_channels, seq_len)
        model = layers.QuartzBlock(
            num_sub_blocks=num_sub_blocks,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            p_dropout=p_dropout
        )
        result = model(input)
        assert result.shape == expected_shape
        check_grad(result=result, model=model)


class TestQuartzBlocks:
    @pytest.mark.parametrize(
        (
            'num_blocks', 'block_repetition', 'num_sub_blocks',
            'in_channels', 'channels_size', 'kernel_size', 'groups',
            'p_dropout', 'batch_size', 'seq_len'
        ),
        (
            (1, 3, 2, 8, [16], 4, 2, 0.0, 3, 10),
            (1, 1, 1, 8, [16], 4, 2, 0.0, 3, 10),
            (3, 3, 2, 8, [8, 12, 16], 4, 2, 0.0, 3, 10),
            (3, 3, 2, 8, [8, 12, 16], [1, 2, 3], 2, 0.0, 3, 10),
        )
    )
    def test_forward(
            self, batcher, num_blocks, block_repetition, num_sub_blocks,
            in_channels, channels_size, kernel_size, groups, p_dropout,
            batch_size, seq_len
    ):
        """Tests the returned shape and the gradients of the model's forward
        """
        out_channels = channels_size[-1]
        input = batcher(batch_size, in_channels, seq_len)
        expected_shape = (batch_size, out_channels, seq_len)
        model = layers.QuartzBlocks(
            num_blocks=num_blocks,
            block_repetition=block_repetition,
            num_sub_blocks=num_sub_blocks,
            in_channels=in_channels,
            channels_size=channels_size,
            kernel_size=kernel_size,
            groups=groups,
            p_dropout=p_dropout
        )
        result = model(input)
        assert result.shape == expected_shape
        check_grad(result=result, model=model)


class TestScaling1d:
    @pytest.mark.parametrize(
        ('d_model', 'batch_size', 'seq_len'),
        (
            (8, 1, 15),
            (8, 1, 1),
            (1, 1, 1),
        )
    )
    def test_forward(self, batcher, d_model, batch_size, seq_len):
        """Tests the returned shape and the gradients of the model's forward
        """
        shape = (batch_size, seq_len, d_model)
        input = batcher(*shape)
        model = layers.Scaling1d(d_model=d_model)
        result = model(input)
        assert result.shape == shape
        check_grad(result=result, model=model)


class TestSqueezeformerConvModule:
    @pytest.mark.parametrize(
        (
            'd_model', 'kernel_size', 'p_dropout', 'batch_size', 'seq_len'
        ),
        (
            (8, 4, 0.01, 3, 2),
            (8, 4, 0.01, 1, 2),
            (16, 1, 0.01, 1, 2),
        )
    )
    def test_forward(
            self, batcher, d_model, kernel_size,
            p_dropout, batch_size, seq_len
    ):
        shape = (batch_size, seq_len, d_model)
        input = batcher(*shape)
        model = layers.SqueezeformerConvModule(
            d_model=d_model, kernel_size=kernel_size, p_dropout=p_dropout
        )
        result = model(input)
        assert result.shape == shape
        check_grad(result=result, model=model)


class TestSqueezeformerRelativeMHSA:
    @pytest.mark.parametrize(
        (
            'd_model', 'h', 'p_dropout', 'batch_size', 'seq_len', 'pad_lens'
        ),
        (
            (4, 2, 0.1, 2, 10, None),
            (4, 2, 0.1, 2, 10, [8, 10]),
            (4, 2, 0.1, 2, 10, [8, 4])
        )
    )
    def test_forward(
            self, batcher, d_model, h, p_dropout, batch_size, seq_len, pad_lens
    ):
        shape = (batch_size, seq_len, d_model)
        input = batcher(*shape)
        mask = None
        if pad_lens is not None:
            mask = get_mask(seq_len=seq_len, pad_lens=pad_lens)
        model = layers.SqueezeformerRelativeMHSA(
            d_model=d_model, h=h, p_dropout=p_dropout
        )
        result = model(input, mask=mask)
        assert result.shape == shape
        check_grad(result=result, model=model)


class TestSqueezeformerFeedForward:
    @pytest.mark.parametrize(
        (
            'd_model', 'expansion_factor', 'p_dropout', 'batch_size', 'seq_len'
        ),
        (
            (4, 2, 0.1, 2, 10),
            (4, 4, 0.1, 1, 10),
            (4, 4, 0.1, 2, 1),
        )
    )
    def test_forward(
            self, batcher, d_model, expansion_factor, p_dropout,
            batch_size, seq_len
    ):
        shape = (batch_size, seq_len, d_model)
        input = batcher(*shape)
        model = layers.SqueezeformerFeedForward(
            d_model=d_model,
            expansion_factor=expansion_factor,
            p_dropout=p_dropout
        )
        result = model(input)
        assert result.shape == shape
        check_grad(result=result, model=model)


class TestSqueezeformerBlock:
    @pytest.mark.parametrize(
        (
            'd_model', 'ff_expansion_factor', 'h', 'kernel_size',
            'p_dropout', 'batch_size', 'seq_len', 'pad_lens'
        ),
        (
            (4, 2, 2, 3, 0.1, 2, 10, None),
            (4, 2, 2, 3, 0.1, 1, 10, None),
            (4, 2, 2, 3, 0.1, 2, 10, [5, 6]),
        )
    )
    def test_forward(
            self, batcher, d_model, ff_expansion_factor, h, kernel_size,
            p_dropout, batch_size, seq_len, pad_lens
    ):
        mask = None
        if pad_lens is not None:
            mask = get_mask(seq_len=seq_len, pad_lens=pad_lens)
        shape = (batch_size, seq_len, d_model)
        input = batcher(*shape)
        model = layers.SqueezeformerBlock(
            d_model=d_model,
            ff_expansion_factor=ff_expansion_factor,
            h=h, kernel_size=kernel_size, p_dropout=p_dropout
        )
        result = model(input, mask=mask)
        assert result.shape == shape
        check_grad(result=result, model=model)


class TestSqueezeAndExcit1D:
    @pytest.mark.parametrize(
        (
            'in_feature', 'reduction_factor', 'batch_size',
            'seq_len', 'pad_lens'
        ),
        (
            (16, 2, 3, 7, [5, 2, 0]),
            (16, 1, 3, 7, [5, 2, 0]),
            (16, 1, 3, 7, [0, 0, 0]),
        )
    )
    def test_forward(
            self, batcher, in_feature, reduction_factor,
            batch_size, seq_len, pad_lens
    ):
        shape = (batch_size, in_feature, seq_len)
        input = batcher(*shape)
        model = layers.SqueezeAndExcit1D(
            in_feature=in_feature, reduction_factor=reduction_factor
        )
        mask = get_mask(seq_len=seq_len, pad_lens=pad_lens)
        result = model(input, mask)
        assert result.shape == shape
        check_grad(result=result, model=model)


class TestContextNetConvLayer:
    @pytest.mark.parametrize(
        (
            'in_channels', 'out_channels', 'kernel_size', 'stride',
            'batch_size', 'seq_len', 'lengths', 'expected_lens',
            'expected_shape'
        ),
        (
            (
                16, 32, 3, 1, 3, 10,
                torch.LongTensor([10, 8, 5]),
                torch.LongTensor([10, 8, 5]),
                (3, 32, 10)
            ),
            (
                16, 32, 3, 2, 3, 10,
                torch.LongTensor([10, 8, 5]),
                torch.LongTensor([4, 4, 3]),
                (3, 32, 4)
            ),
        )
    )
    def test_forward(
            self, batcher, in_channels, out_channels, kernel_size, stride,
            batch_size, seq_len, lengths, expected_lens, expected_shape
    ):
        input = batcher(batch_size, in_channels, seq_len)
        model = layers.ContextNetConvLayer(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride
        )
        result, lengths = model(input, lengths)
        assert result.shape == expected_shape
        assert torch.all(lengths == expected_lens).item()
        check_grad(result=result, model=model)


class TestContextNetResidual:
    @pytest.mark.parametrize(
        (
            'in_channels', 'out_channels', 'kernel_size', 'stride',
            'batch_size', 'seq_len', 'res_len'
        ),
        (
            (8, 16, 4, 1, 3, 10, 10),
            (8, 16, 4, 2, 3, 4, 10),
            (8, 16, 1, 2, 3, 4, 7),
        )
    )
    def test_forward(
            self, batcher, in_channels,
            out_channels, kernel_size, stride,
            batch_size, seq_len, res_len
    ):
        shape = (batch_size, out_channels, seq_len)
        input = batcher(*shape)
        residual = batcher(batch_size, in_channels, res_len)
        model = layers.ContextNetResidual(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride
        )
        result = model(residual, input)
        assert result.shape == shape


class TestContextNetBlock:
    @pytest.mark.parametrize(
        (
            'n_layers', 'in_channels', 'out_channels', 'kernel_size',
            'reduction_factor', 'add_residual', 'last_layer_stride',
            'seq_len', 'batch_size', 'lengths', 'expected_lengths',
            'expected_shape'
        ),
        (
            (
                3, 8, 16, 3, 2, False, 1, 10, 3,
                torch.LongTensor([10, 8, 5]),
                torch.LongTensor([10, 8, 5]),
                (3, 16, 10)
            ),
            (
                3, 8, 16, 3, 2, False, 2, 10, 3,
                torch.LongTensor([10, 8, 5]),
                torch.LongTensor([4, 4, 3]),
                (3, 16, 4)
            ),
        )
    )
    def test_forward(
            self, batcher, n_layers, in_channels, out_channels, kernel_size,
            reduction_factor, add_residual, last_layer_stride, seq_len,
            batch_size, lengths, expected_lengths, expected_shape
    ):
        input = batcher(batch_size, in_channels, seq_len)
        model = layers.ContextNetBlock(
            n_layers=n_layers, in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size,
            reduction_factor=reduction_factor, add_residual=add_residual,
            last_layer_stride=last_layer_stride
        )
        result, lengths = model(input, lengths)
        assert result.shape == expected_shape
        assert torch.all(lengths == expected_lengths)
        check_grad(result=result, model=model)
