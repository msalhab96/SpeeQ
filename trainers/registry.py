from torch.optim import (
    Adam, RMSprop, SGD
)
from trainers.criterions import CTCLoss

CRITERIONS = {
    'ctc': CTCLoss
}

OPTIMIZERS = {
    'adam': Adam,
    'rmsprop': RMSprop,
    'sgd': SGD
}


def get_criterion(
        name: str,
        blank_id: int,
        pad_id: int,
        *args,
        **kwargs
        ):
    assert name in CRITERIONS
    return CRITERIONS[name](
        blank_id=blank_id,
        pad_id=pad_id,
        *args,
        **kwargs
    )
