from typing import Callable
from functools import wraps


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
