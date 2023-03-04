from functools import wraps
import os
from typing import Callable

from speeq.utils.utils import get_key_tag, save_state_dict


def step_log(key: str, category: str):
    """Logs the result value at each time
    the wrapped function get called.

    Args:
        key (str): The key will be used to log the value
        to.
    """

    def logger(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if self.is_master:
                # Log only for the master node
                self.inline_log(key, category, result)
            return result

        return wrapper

    return logger


def export_ckpt(key: str, category: str) -> Callable:
    """Saves a checkpoint at any steps that the results
    are less than the minimum global loss.
    """
    tag = get_key_tag(key, category)

    def exporter(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(trainer, *args, **kwargs):
            results = func(trainer, *args, **kwargs)
            if trainer.is_master:
                loss = trainer.history[tag][-1]
                if loss < trainer.min_loss:
                    if os.path.exists(trainer.outdir) is False:
                        os.mkdir(trainer.outdir)
                    trainer.min_loss = loss
                    save_state_dict(
                        model_name="checkpoint",
                        outdir=trainer.outdir,
                        model=trainer.model,
                        optimizer=trainer.optimizer,
                        step=trainer.counter,
                        history=trainer.history,
                    )
            return results

        return wrapper

    return exporter
