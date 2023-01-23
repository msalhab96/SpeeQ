from typing import Tuple
from torch import Tensor
from torch import nn
from torchaudio import transforms


class CTCLoss(nn.CTCLoss):
    def __init__(
            self,
            blank_id: int,
            reduction='mean',
            zero_infinity=False,
            *args,
            **kwargs
            ):
        super().__init__(
            blank=blank_id,
            reduction=reduction,
            zero_infinity=zero_infinity
            )


def remove_positionals(
        input: Tensor, target: Tensor
        ) -> Tuple[Tensor, Tensor]:
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


def get_flatten_results(
        input: Tensor, target: Tensor
        ) -> Tuple[Tensor, Tensor]:
    target = target.view(-1)
    input = input.view(-1, input.shape[-1])
    return input, target


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
            self,
            pad_id: int,
            reduction='mean',
            label_smoothing=0.0,
            *args, **kwargs
            ) -> None:
        super().__init__(
            ignore_index=pad_id,
            reduction=reduction,
            label_smoothing=label_smoothing
        )

    def forward(self, input, target, *args, **kwargs):
        # input of shape [B, M, C]
        # target of shape [B, M]
        input, target = remove_positionals(input, target)
        input, target = get_flatten_results(input, target)
        return super().forward(input, target)


class NLLLoss(nn.NLLLoss):
    def __init__(
            self,
            pad_id: int,
            reduction='mean',
            *args, **kwargs
            ) -> None:
        super().__init__(
            ignore_index=pad_id,
            reduction=reduction
        )

    def forward(self, input, target, *args, **kwargs):
        # input of shape [B, M, C]
        # target of shape [B, M]
        input, target = remove_positionals(input, target)
        input, target = get_flatten_results(input, target)
        return super().forward(input, target)


class RNNTLoss(transforms.RNNTLoss):
    def __init__(
            self, blank_id: int, reduction='mean', *args, **kwargs
            ) -> None:
        super().__init__(blank=blank_id, reduction=reduction)

    def forward(
            self,
            logits: Tensor,
            logits_len: Tensor,
            targets: Tensor,
            target_len: Tensor
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
            target_lengths=target_len
        )
