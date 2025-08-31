"""utils/retry.py

Small async retry/backoff helper used across the codebase.

Provides:
- async_retry_call(callable) : retry an async zero-arg callable with exponential backoff + optional jitter
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
    Retry an async zero-arg callable with exponential backoff and small jitter.

    Args:
        call_fn: zero-arg callable returning a coroutine
        retries: number of retry attempts (total attempts). The function will try up to `retries` times.
        base_backoff: base delay in seconds
        max_backoff: maximum delay cap

    Returns:
        Result of the coroutine or raises last exception after exhausting retries
    """
    last_exc = None
    # attempts indexed from 0..retries-1
    for attempt in range(retries):
        try:
            return await call_fn()
        except Exception as exc:
            # Let cancellations and keyboard interrupts propagate immediately.
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
                raise

            last_exc = exc
            # If this was the last allowed attempt, log error and re-raise the last exception.
            if attempt >= retries - 1:
                logger.error("Non-recoverable error in async_retry_call; re-raising", exc_info=True)
                raise last_exc

            # Compute exponential backoff (attempt 0 -> base_backoff, attempt 1 -> base_backoff*2, ...)
            backoff = min(base_backoff * (2 ** attempt), max_backoff)
            # Small jitter to avoid thundering herd: +/-10%
            jitter = backoff * 0.1
            sleep_for = max(0.0, backoff + random.uniform(-jitter, jitter))
            logger.warning(
                f"Retry attempt {attempt + 1}/{retries} failed with {exc}; retrying in {sleep_for:.2f}s"
            )
            await asyncio.sleep(sleep_for)

    # Shouldn't reach here, but raise last exception defensively.
    if last_exc:
        raise last_exc
    return None


def retry_async(retries: int = 3, base_backoff: float = 0.5, max_backoff: float = 5.0):
    """
    Decorator to apply async_retry_call to an async function.

    Usage:

        @retry_async(retries=2)
        async def unstable():
            import random
            if random.random() < 0.7:
                raise RuntimeError("transient error")
            return "success"
    """
    def decorator(func: Callable[..., "Coroutine[Any, Any, Any]"]):
        async def wrapper(*args, **kwargs):
            return await async_retry_call(
                lambda: func(*args, **kwargs),
                retries=retries,
                base_backoff=base_backoff,
                max_backoff=max_backoff,
            )
        return wrapper
    return decorator
