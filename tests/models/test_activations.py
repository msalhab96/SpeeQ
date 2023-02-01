import pytest
import torch

from speeq.models import activations


class TestSigmax:
    parameters_mark = pytest.mark.parametrize(
        ('inp_shape', 'dim'),
        (
            ((1,), 0),
            ((4,), 0),
            ((4, 2), 0),
            ((4, 2), 1),
            ((4, 2, 3), 2),
        )
        )

    @parameters_mark
    def test_shape(self, inp_shape, dim):
        """Tests the output shape.
        """
        act = activations.Sigmax(dim=dim)
        inp = torch.ones(*inp_shape)
        result = act(inp)
        assert result.shape == inp_shape

    @parameters_mark
    def test_sum(self, inp_shape, dim):
        """Tests the summation accross the dim.
        """
        act = activations.Sigmax(dim=dim)
        inp = torch.ones(*inp_shape)
        result = act(inp)
        print(result.sum(dim=dim))
        print(result.shape, inp_shape[dim], result.sum(dim=dim).shape)
        if len(inp_shape) == 1:
            expected = 1
        else:
            expected = torch.ones(*inp_shape[:dim], *inp_shape[1 + dim:])
        mask = result.sum(dim=dim) == expected
        assert torch.all(mask).item()

    @pytest.mark.parametrize(
        ('input', 'target', 'dim'),
        (
            (
                torch.tensor([5.]),
                torch.tensor([1.]),
                0
            ),
            (
                torch.tensor([3., 3.]),
                torch.tensor([0.5, 0.5]),
                0
            ),
            (
                torch.tensor([[3.], [3.]]),
                torch.tensor([1., 1.]),
                1
            ),
            (
                torch.tensor([[3., 3.], [3., 3.]]),
                torch.tensor([[.5, .5], [.5, .5]]),
                1
            ),
            (
                torch.tensor([[0.15, 3.4], [0., 5.]]),
                torch.tensor([[0.3592, 0.6407], [0.3442, 0.6557]]),
                1
            ),
        )
    )
    def test_values(self, input, target, dim):
        """tests the correctness of the values
        """
        act = activations.Sigmax(dim=dim)
        result = act(input)
        assert torch.allclose(
            result, target, rtol=1e-2, atol=1e-2
            )


class TestCReLu:
    parameters_mark = pytest.mark.parametrize(
        ('inputs', 'expected', 'max_val'),
        (
            (
                torch.tensor([
                    [-1.5, 5, 2, 10],
                    [-2, -0.05, 2.0, 0.1],
                    [0.0, 1, 2.0, 0.1],
                ]),
                torch.tensor([
                    [0, 5, 2, 5],
                    [0, 0, 2.0, 0.1],
                    [0.0, 1, 2.0, 0.1],
                ]),
                5
            ),
            (
                torch.tensor([
                    [-1.5],
                ]),
                torch.tensor([
                    [0.0],
                ]),
                1
            )
        )
    )

    @parameters_mark
    def test_shape(self, inputs, max_val, expected):
        """Tests the output shape.
        """
        act = activations.CReLu(max_val)
        result = act(inputs)
        assert result.shape == inputs.shape

    @parameters_mark
    def test_values(self, inputs, max_val, expected):
        """tests the correctness of the values.
        """
        act = activations.CReLu(max_val)
        result = act(inputs)
        print(result)
        assert torch.allclose(
            result, expected, rtol=1e-15, atol=1e-15
            )
