from functools import wraps
from typing import Callable


def check_token(token: str) -> Callable:
    """To check if a token exists or not
    Args:
        token ([type]): the token to be checked
    """
    def decorator(func):
        @wraps(func)
        def wrapper(obj, token=token):
            if token in obj._token_to_id:
                return obj._token_to_id[token]
            return func(obj, token)
        return wrapper
    return decorator
