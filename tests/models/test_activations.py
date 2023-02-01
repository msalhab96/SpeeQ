import pytest
import torch

from speeq.models import activations


class TestSigmax:
    mark = pytest.mark.parametrize(
        ('inp_shape', 'dim'),
        (
            ((1,), 0),
            ((4,), 0),
            ((4, 2), 0),
            ((4, 2), 1),
            ((4, 2, 3), 2),
        )
        )

    @mark
    def test_shape(self, inp_shape, dim):
        """Tests the output shape.
        """
        act = activations.Sigmax(dim=dim)
        inp = torch.ones(*inp_shape)
        result = act(inp)
        assert result.shape == inp_shape

    @mark
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
        print(result)
        assert torch.allclose(
            result, target, rtol=1e-2, atol=1e-2
            )
