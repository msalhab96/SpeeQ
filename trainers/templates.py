from dataclasses import asdict, dataclass
from numbers import Number

from constants import SCHEDULER_TYPE_KEY
from interfaces import ITemplate


class BaseSchedulerTemplate(ITemplate):
    def get_dict(self) -> dict:
        return asdict(self)

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return SCHEDULER_TYPE_KEY


@dataclass
class NoamSchedulerTemp(BaseSchedulerTemplate):
    """Noam scheduler template

    Args:
        warmup_staps (int): The warmup steps.
        d_model (int): The model dimension.
    """
    warmup_staps: int
    d_model: int
    _name = 'noam'


@dataclass
class SqueezeformerNoamSchedulerTemp(BaseSchedulerTemplate):
    """Noam scheduler with changes proposed in Squeezeformer paper template.

    Args:
        warmup_staps (int): The warmup steps.
        lr_peak (Number): The peak value of the learning rate.
        decay_rate (Number): The decay rate of the learning rate.
        t_peak (Number): The number of steps to keep the peak learning
            rate for.
    """
    warmup_staps: int
    lr_peak: Number
    decay_rate: Number
    t_peak: int
    _name = 'squeezeformer_noam'
