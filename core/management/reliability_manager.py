"""
core/management/reliability_manager.py

Manages retries, error tracking, and safe mode for order execution reliability.
"""

import asyncio
import logging
import random
from typing import Callable, Any, Optional, Dict

logger = logging.getLogger(__name__)


class ReliabilityManager:
    """Manages retries, error tracking, and safe mode."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ReliabilityManager.

        Args:
            config: Reliability configuration dictionary
        """
        self._reliability = {
            "max_retries": 3,
            "backoff_base": 0.5,
            "max_backoff": 10.0,
            "safe_mode_threshold": 5,
            "close_positions_on_safe": False,
        }
        if config:
            self._reliability.update(config)

        self.safe_mode_active: bool = False
        self.critical_error_count: int = 0

    async def retry_async(
        self,
        coro_factory: Callable[[], "Coroutine[Any, Any, Any]"],
        retries: Optional[int] = None,
        base_backoff: Optional[float] = None,
        max_backoff: Optional[float] = None,
        exceptions: tuple = (Exception,),
    ) -> Any:
        """
        Execute an async operation with exponential backoff and jitter.

        Args:
            coro_factory: A zero-arg callable returning the coroutine to execute.
            retries: Number of retry attempts (not counting the initial try).
            base_backoff: Base backoff in seconds (will be doubled each retry).
            max_backoff: Maximum backoff cap.
            exceptions: Tuple of exception types that should trigger a retry.

        Returns:
            Result of the coroutine if successful.

        Raises:
            The last exception if all retries are exhausted.
        """
        retries = retries if retries is not None else self._reliability.get("max_retries", 3)
        base_backoff = base_backoff if base_backoff is not None else self._reliability.get("backoff_base", 0.5)
        max_backoff = max_backoff if max_backoff is not None else self._reliability.get("max_backoff", 10.0)

        attempt = 0
        while True:
            try:
                attempt += 1
                coro = coro_factory()
                return await coro
            except exceptions as e:
                if attempt > retries:
                    logger.error(f"Retry exhausted after {attempt-1} retries: {str(e)}", exc_info=True)
                    raise
                backoff = min(max_backoff, base_backoff * (2 ** (attempt - 1)))
                jitter = backoff * 0.1
                sleep_for = backoff + random.uniform(-jitter, jitter)
                logger.warning(
                    f"Operation failed (attempt {attempt}/{retries}). Retrying in {sleep_for:.2f}s. Error: {str(e)}",
                    exc_info=False,
                )
                await asyncio.sleep(max(0.0, sleep_for))

    def record_critical_error(self, exc: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a critical error occurrence and activate safe mode if thresholds exceeded.

        Args:
            exc: Exception instance
            context: Optional contextual info (symbol, operation, etc.)
        """
        try:
            self.critical_error_count += 1
            logger.error(f"Critical error #{self.critical_error_count}: {str(exc)}", exc_info=False)

            threshold = int(self._reliability.get("safe_mode_threshold", 5))
            if self.critical_error_count >= threshold and not self.safe_mode_active:
                self.activate_safe_mode(reason=str(exc), context=context)
        except Exception:
            logger.exception("Failed to record critical error")

    def activate_safe_mode(self, reason: str = "", context: Optional[Dict[str, Any]] = None) -> None:
        """
        Enable safe mode which disables opening new positions and optionally closes existing ones.
        """
        try:
            self.safe_mode_active = True
            logger.critical(f"Safe mode activated due to: {reason}")

            close_on_safe = bool(self._reliability.get("close_positions_on_safe", False))
            if close_on_safe:
                logger.info("Safe mode: closing existing positions")
                # Implementation: This would require access to order_manager
                # For now, log the action; integration needed in bot_engine
                logger.warning("Position closing not implemented; requires order_manager integration")
        except Exception:
            logger.exception("Failed to activate safe mode")
