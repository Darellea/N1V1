"""
utils/retry.py

Small async retry/backoff helper used across the codebase.

Provides:
- async_retry_call(callable) : retry an async zero-arg callable with exponential backoff + jitter
- retry_async(retries, base_backoff, max_backoff) : decorator to wrap async functions
"""

import asyncio
import logging
import random
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


async def async_retry_call(
    call_fn: Callable[[], "Coroutine[Any, Any, Any]"],
    retries: int = 3,
    base_backoff: float = 0.5,
    max_backoff: float = 5.0,
) -> Any:
    """
    Retry an async zero-arg callable with exponential backoff and jitter.

    Args:
        call_fn: zero-arg callable returning a coroutine
        retries: number of retry attempts
        base_backoff: base delay in seconds
        max_backoff: maximum delay cap

    Returns:
        Result of the coroutine or raises last exception after exhausting retries
    """
    attempt = 0
    while True:
        try:
            attempt += 1
            return await call_fn()
        except Exception as exc:
            if attempt > retries:
                logger.error(f"Retry exhausted after {attempt-1} retries: {exc}")
                raise
            backoff = min(max_backoff, base_backoff * (2 ** (attempt - 1)))
            jitter = backoff * 0.1
            sleep_for = max(0.0, backoff + random.uniform(-jitter, jitter))
            logger.warning(f"Async call failed (attempt {attempt}/{retries}), retrying in {sleep_for:.2f}s: {exc}")
            await asyncio.sleep(sleep_for)
            continue


def retry_async(retries: int = 3, base_backoff: float = 0.5, max_backoff: float = 5.0):
    """
    Decorator to apply async_retry_call to an async function.

    Usage:

        @retry_async(retries=2)
        async def unstable(...):
            ...

    The wrapped function's exceptions will be retried according to the policy.
    """
    def decorator(func: Callable[..., "Coroutine[Any, Any, Any]"]):
        async def wrapper(*args, **kwargs):
            return await async_retry_call(lambda: func(*args, **kwargs), retries=retries, base_backoff=base_backoff, max_backoff=max_backoff)
        return wrapper
    return decorator
