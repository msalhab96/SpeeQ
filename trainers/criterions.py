from torch import nn


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
