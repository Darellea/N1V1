"""
Retry and error handling utilities for the SignalRouter.

Provides centralized retry/backoff functionality and error handling
wrappers for signal processing operations.
"""

import asyncio
import logging
import random
import time
from typing import Any, Callable, Coroutine, Dict, Optional

logger = logging.getLogger(__name__)


class RetryManager:
    """
    Manages retry logic with exponential backoff and jitter for async operations.
    """

    def __init__(self,
                 max_retries: int = 2,
                 base_backoff: float = 0.5,
                 max_backoff: float = 5.0,
                 backoff_multiplier: float = 2.0,
                 jitter_factor: float = 0.1):
        """
        Initialize the retry manager.

        Args:
            max_retries: Maximum number of retry attempts
            base_backoff: Base delay in seconds
            max_backoff: Maximum delay cap
            backoff_multiplier: Exponential backoff multiplier
            jitter_factor: Jitter factor for randomization
        """
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.backoff_multiplier = backoff_multiplier
        self.jitter_factor = jitter_factor

    async def retry_async_call(self,
                              call_fn: Callable[[], Coroutine[Any, Any, Any]],
                              context: str = "operation") -> Any:
        """
        Retry an async callable with exponential backoff and jitter.

        Args:
            call_fn: Zero-arg callable returning coroutine
            context: Context description for logging

        Returns:
            Result of the coroutine or raises last exception
        """
        attempt = 0
        last_exception = None

        while attempt <= self.max_retries:
            try:
                attempt += 1
                return await call_fn()

            except Exception as e:
                last_exception = e

                if attempt > self.max_retries:
                    logger.error(f"Retry exhausted for {context} after {attempt-1} retries: {str(e)}")
                    raise

                # Calculate backoff with jitter
                backoff = min(
                    self.max_backoff,
                    self.base_backoff * (self.backoff_multiplier ** (attempt - 1))
                )

                # Add jitter
                jitter = backoff * self.jitter_factor
                sleep_for = backoff + random.uniform(-jitter, jitter)
                sleep_for = max(0.0, sleep_for)

                logger.warning(
                    f"{context} failed (attempt {attempt}/{self.max_retries + 1}), "
                    f"retrying in {sleep_for:.2f}s: {str(e)}"
                )

                await asyncio.sleep(sleep_for)

        # This should never be reached, but just in case
        raise last_exception

    def retry_sync_call(self,
                       call_fn: Callable[[], Any],
                       context: str = "operation") -> Any:
        """
        Retry a synchronous callable with exponential backoff and jitter.

        Args:
            call_fn: Zero-arg callable
            context: Context description for logging

        Returns:
            Result of the callable or raises last exception
        """
        attempt = 0
        last_exception = None

        while attempt <= self.max_retries:
            try:
                attempt += 1
                return call_fn()

            except Exception as e:
                last_exception = e

                if attempt > self.max_retries:
                    logger.error(f"Retry exhausted for {context} after {attempt-1} retries: {str(e)}")
                    raise

                # Calculate backoff with jitter
                backoff = min(
                    self.max_backoff,
                    self.base_backoff * (self.backoff_multiplier ** (attempt - 1))
                )

                # Add jitter
                jitter = backoff * self.jitter_factor
                sleep_for = backoff + random.uniform(-jitter, jitter)
                sleep_for = max(0.0, sleep_for)

                logger.warning(
                    f"{context} failed (attempt {attempt}/{self.max_retries + 1}), "
                    f"retrying in {sleep_for:.2f}s: {str(e)}"
                )

                time.sleep(sleep_for)

        # This should never be reached, but just in case
        raise last_exception


class ErrorHandler:
    """
    Handles errors and provides safe-mode functionality for signal processing.
    """

    def __init__(self, safe_mode_threshold: int = 10):
        """
        Initialize the error handler.

        Args:
            safe_mode_threshold: Number of critical errors before entering safe mode
        """
        self.critical_errors = 0
        self.safe_mode_threshold = safe_mode_threshold
        self.block_processing = False
        self._lock = asyncio.Lock()

    async def record_critical_error(self,
                                  error: Exception,
                                  context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a critical error and potentially enter safe mode.

        Args:
            error: The exception that occurred
            context: Additional context information
        """
        async with self._lock:
            self.critical_errors += 1

            context_str = f" (context: {context})" if context else ""
            logger.error(f"Critical error #{self.critical_errors}: {str(error)}{context_str}")

            if self.critical_errors >= self.safe_mode_threshold and not self.block_processing:
                self.block_processing = True
                logger.critical(
                    f"Entering safe mode after {self.critical_errors} critical errors. "
                    "Signal processing blocked."
                )

    def is_blocking(self) -> bool:
        """
        Check if processing should be blocked due to errors.

        Returns:
            True if processing should be blocked
        """
        return self.block_processing

    def reset_error_count(self) -> None:
        """
        Reset the error count and exit safe mode.
        """
        self.critical_errors = 0
        self.block_processing = False
        logger.info("Error count reset, exiting safe mode")

    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.

        Returns:
            Dictionary with error statistics
        """
        return {
            "critical_errors": self.critical_errors,
            "safe_mode_threshold": self.safe_mode_threshold,
            "block_processing": self.block_processing
        }


# Global instances for convenience
default_retry_manager = RetryManager()
default_error_handler = ErrorHandler()


async def with_retry_async(call_fn: Callable[[], Coroutine[Any, Any, Any]],
                          context: str = "operation",
                          max_retries: int = 2) -> Any:
    """
    Convenience function to retry an async call with default settings.

    Args:
        call_fn: Zero-arg callable returning coroutine
        context: Context description for logging
        max_retries: Maximum number of retry attempts

    Returns:
        Result of the coroutine
    """
    retry_mgr = RetryManager(max_retries=max_retries)
    return await retry_mgr.retry_async_call(call_fn, context)


def with_retry_sync(call_fn: Callable[[], Any],
                   context: str = "operation",
                   max_retries: int = 2) -> Any:
    """
    Convenience function to retry a sync call with default settings.

    Args:
        call_fn: Zero-arg callable
        context: Context description for logging
        max_retries: Maximum number of retry attempts

    Returns:
        Result of the callable
    """
    retry_mgr = RetryManager(max_retries=max_retries)
    return retry_mgr.retry_sync_call(call_fn, context)


async def safe_async_call(call_fn: Callable[[], Coroutine[Any, Any, Any]],
                         context: str = "operation",
                         error_handler: Optional[ErrorHandler] = None) -> Any:
    """
    Safely execute an async call with error handling.

    Args:
        call_fn: Zero-arg callable returning coroutine
        context: Context description for logging
        error_handler: Error handler instance

    Returns:
        Result of the coroutine or None if failed
    """
    if error_handler and error_handler.is_blocking():
        logger.warning(f"{context} blocked due to safe mode")
        return None

    try:
        return await call_fn()
    except Exception as e:
        logger.exception(f"Error in {context}: {e}")
        if error_handler:
            await error_handler.record_critical_error(e, {"context": context})
        return None


def safe_sync_call(call_fn: Callable[[], Any],
                  context: str = "operation",
                  error_handler: Optional[ErrorHandler] = None) -> Any:
    """
    Safely execute a sync call with error handling.

    Args:
        call_fn: Zero-arg callable
        context: Context description for logging
        error_handler: Error handler instance

    Returns:
        Result of the callable or None if failed
    """
    if error_handler and error_handler.is_blocking():
        logger.warning(f"{context} blocked due to safe mode")
        return None

    try:
        return call_fn()
    except Exception as e:
        logger.exception(f"Error in {context}: {e}")
        if error_handler:
            # For sync calls, we can't await, so we'll log but not record
            logger.error(f"Critical error in {context}: {e}")
        return None
