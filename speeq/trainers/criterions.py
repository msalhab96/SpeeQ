"""
Contains different loss functions used in various speech recognition models.

- CTCLoss: Connectionist Temporal Classification loss function.
- CrossEntropyLoss: Cross-entropy loss function.
- NLLLoss: Negative log-likelihood loss function.
- RNNTLoss: Recurrent Neural Network Transducer loss function.

"""
from typing import Tuple

from torch import Tensor, nn
from torchaudio import transforms


class CTCLoss(nn.CTCLoss):
    """The CTC loss.

    Args:

        blank_id (int): The blank id.

        reduction (str, optional): Specifies the reduction to apply to the
        output. Default to "mean".

        zero_infinity (bool, optional): Whether to zero infinite losses and the
        associated gradients. Default: False Infinite losses mainly occur when
        the inputs are too short to be aligned to the targets.

    """

    def __init__(
        self, blank_id: int, reduction="mean", zero_infinity=False, *args, **kwargs
    ):
        super().__init__(
            blank=blank_id, reduction=reduction, zero_infinity=zero_infinity
        )


def remove_positionals(input: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Removes the SOS from the target and EOS
    prediction from the input

    Args:

        input (Tensor): The input tensor of shape [B, M, C].

        target (Tensor): The target tensor of shape [B, C]

    Returns:

        Tuple[Tensor, Tensor]: The input and target.
    """
    input = input[:, :-1, :]
    input = input.contiguous()
    target = target[:, 1:]
    target = target.contiguous()
    return input, target


def get_flatten_results(input: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Flatten the results by making the input that of shape [B, M, C] to be of shape
    [B * M, C] and the target of shape [B, M] to be of shape [B * M]

    Args:

        input (Tensor): The predictions of shape [B, M, C].

        target (Tensor): The target tensor of shape [B, M]

    Returns:

        Tuple[Tensor, Tensor]: Atuple of the flatten results.
    """
    target = target.view(-1)
    input = input.view(-1, input.shape[-1])
    return input, target


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """computes the cross entropy loss between input logits and target.

    Args:

        pad_id (int): The padding id.

        reduction (str, optional): Specifies the reduction to apply to the
        output. Default to "mean".

        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the
        amount of smoothing when computing the loss. Default 0.0.
    """

    def __init__(
        self, pad_id: int, reduction="mean", label_smoothing=0.0, *args, **kwargs
    ) -> None:
        super().__init__(
            ignore_index=pad_id, reduction=reduction, label_smoothing=label_smoothing
        )

    def forward(self, input, target, *args, **kwargs):
        # input of shape [B, M, C]
        # target of shape [B, M]
        input, target = remove_positionals(input, target)
        input, target = get_flatten_results(input, target)
        return super().forward(input, target)


class NLLLoss(nn.NLLLoss):
    """computes the negative log likelihood loss.

    Args:

        pad_id (int): The padding id.

        reduction (str, optional): Specifies the reduction to apply to the
        output. Default to "mean".
    """

    def __init__(self, pad_id: int, reduction="mean", *args, **kwargs) -> None:
        super().__init__(ignore_index=pad_id, reduction=reduction)

    def forward(self, input, target, *args, **kwargs):
        # input of shape [B, M, C]
        # target of shape [B, M]
        input, target = remove_positionals(input, target)
        input, target = get_flatten_results(input, target)
        return super().forward(input, target)


class RNNTLoss(transforms.RNNTLoss):
    """computes the RNNT loss.

    Args:

        blank_id (int): The blank id.

        reduction (str, optional): Specifies the reduction to apply to the
        output. Default to "mean".
    """

    def __init__(self, blank_id: int, reduction="mean", *args, **kwargs) -> None:
        super().__init__(blank=blank_id, reduction=reduction)

    def forward(
        self, logits: Tensor, logits_len: Tensor, targets: Tensor, target_len: Tensor
    ) -> Tensor:
        # logits of shape [B, Ts, Tt, C]
        # target of shape [B, Tt] and start with SOS
        targets = targets[:, 1:]
        targets = targets.contiguous()
        target_len = target_len - 1
        return super().forward(
            logits=logits,
            logit_lengths=logits_len,
            targets=targets,
            target_lengths=target_len,
        )
