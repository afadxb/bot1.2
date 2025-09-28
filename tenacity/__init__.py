"""Minimal tenacity stub."""

from __future__ import annotations

from typing import Any, Callable


class RetryError(Exception):
    pass


def retry(stop=None, wait=None):
    def decorator(func: Callable):
        def wrapper(*args: Any, **kwargs: Any):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def stop_after_attempt(attempts: int):
    return attempts


def wait_exponential(multiplier: int = 1, min: int = 1, max: int = 10):
    return (multiplier, min, max)
