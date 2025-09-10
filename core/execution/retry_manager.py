"""
Retry Manager

Handles retry logic with exponential backoff and fallback mechanisms
for failed execution attempts.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, Callable, Awaitable, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .execution_types import ExecutionPolicy, ExecutionStatus
from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class RetryErrorType(Enum):
    """Types of errors that can trigger retries."""
    NETWORK_ERROR = "network_error"
    EXCHANGE_TIMEOUT = "exchange_timeout"
    RATE_LIMIT = "rate_limit"
    EXCHANGE_ERROR = "exchange_error"
    INSUFFICIENT_LIQUIDITY = "insufficient_liquidity"
    PRICE_SLIPPAGE = "price_slippage"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    timestamp: datetime
    error_type: RetryErrorType
    error_message: str
    delay_used: float
    policy_used: ExecutionPolicy


@dataclass
class RetryResult:
    """Result of retry operation."""
    success: bool
    final_result: Optional[Dict[str, Any]]
    total_attempts: int
    total_delay: float
    attempts: List[RetryAttempt]
    fallback_used: bool
    final_policy: ExecutionPolicy
    error_message: Optional[str] = None


class RetryManager:
    """
    Manages retry logic with exponential backoff and policy fallback
    for failed execution attempts.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the retry manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.enabled = self.config.get('enabled', True)
        self.max_retries = self.config.get('max_retries', 3)
        self.backoff_base = self.config.get('backoff_base', 1.0)
        self.max_backoff = self.config.get('max_backoff', 30.0)
        self.retry_on_errors = set(self.config.get('retry_on_errors', [
            'network', 'exchange_timeout', 'rate_limit'
        ]))

        # Fallback configuration
        self.fallback_enabled = self.config.get('fallback_enabled', True)
        self.fallback_policy = ExecutionPolicy(self.config.get('fallback_policy', 'market'))
        self.fallback_on_attempt = self.config.get('fallback_on_attempt', 2)

        # Kill-switch configuration
        self.kill_switch_enabled = self.config.get('kill_switch_enabled', True)
        self.kill_switch_threshold = self.config.get('kill_switch_threshold', 5)

        # Error tracking
        self.error_counts: Dict[str, int] = {}
        self.kill_switch_activated = False

        self.logger.info(f"RetryManager initialized: max_retries={self.max_retries}, "
                        f"backoff_base={self.backoff_base}s")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'enabled': True,
            'max_retries': 3,
            'backoff_base': 1.0,
            'max_backoff': 30.0,
            'retry_on_errors': ['network', 'exchange_timeout', 'rate_limit'],
            'fallback_enabled': True,
            'fallback_policy': 'market',
            'fallback_on_attempt': 2,
            'kill_switch_enabled': True,
            'kill_switch_threshold': 5
        }

    async def execute_with_retry(
        self,
        execution_func: Callable[..., Awaitable[Dict[str, Any]]],
        signal: Any,
        policy: ExecutionPolicy,
        context: Dict[str, Any],
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a function with retry logic and fallback mechanisms.

        Args:
            execution_func: Function to execute
            signal: Trading signal
            policy: Execution policy
            context: Execution context
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Execution result dictionary
        """
        if not self.enabled:
            return await execution_func(signal, policy, context, *args, **kwargs)

        attempts: List[RetryAttempt] = []
        current_policy = policy
        total_delay = 0.0
        fallback_used = False

        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                self.logger.debug(f"Execution attempt {attempt + 1}/{self.max_retries + 1} "
                                f"with policy {current_policy.value}")

                # Execute the function
                start_time = time.time()
                result = await execution_func(signal, current_policy, context, *args, **kwargs)
                execution_time = time.time() - start_time

                # Check if execution was successful
                if self._is_successful_execution(result):
                    self.logger.info(f"Execution succeeded on attempt {attempt + 1}")
                    # Ensure status is COMPLETED for successful executions
                    result['status'] = ExecutionStatus.COMPLETED
                    return self._enrich_result_with_retry_info(
                        result, attempt, total_delay, attempts, fallback_used, current_policy
                    )

                # Execution failed, determine if we should retry
                error_type = self._classify_error(result)
                should_retry = self._should_retry(error_type, attempt)

                if not should_retry:
                    self.logger.warning(f"Execution failed on attempt {attempt + 1}, not retrying: {error_type.value}")
                    return self._create_failure_result(
                        attempt, total_delay, attempts, current_policy,
                        f"Execution failed: {error_type.value}", fallback_used
                    )

                # Check if we should fallback to different policy
                if self.fallback_enabled and attempt + 1 >= self.fallback_on_attempt and current_policy != self.fallback_policy:
                    current_policy = self.fallback_policy
                    fallback_used = True
                    self.logger.info(f"Falling back to policy {current_policy.value} on attempt {attempt + 1}")

                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt)
                total_delay += delay

                # Record the attempt
                attempt_info = RetryAttempt(
                    attempt_number=attempt + 1,
                    timestamp=datetime.now(),
                    error_type=error_type,
                    error_message=result.get('error_message', 'Unknown error'),
                    delay_used=delay,
                    policy_used=current_policy
                )
                attempts.append(attempt_info)

                # Log retry attempt
                trade_logger.performance("Execution Retry", {
                    'attempt': attempt + 1,
                    'error_type': error_type.value,
                    'delay': delay,
                    'policy': current_policy.value,
                    'symbol': getattr(signal, 'symbol', 'unknown')
                })

                # Wait before retry
                if delay > 0:
                    self.logger.debug(f"Waiting {delay:.1f}s before retry")
                    await asyncio.sleep(delay)

            except Exception as e:
                self.logger.error(f"Unexpected error during execution attempt {attempt + 1}: {e}")

                # Check kill switch
                if self.kill_switch_enabled:
                    await self._check_kill_switch(str(e))

                error_type = RetryErrorType.UNKNOWN_ERROR
                should_retry = attempt < self.max_retries

                if not should_retry:
                    return self._create_failure_result(
                        attempt, total_delay, attempts, current_policy, str(e), fallback_used
                    )

                # Record the attempt
                attempt_info = RetryAttempt(
                    attempt_number=attempt + 1,
                    timestamp=datetime.now(),
                    error_type=error_type,
                    error_message=str(e),
                    delay_used=self._calculate_delay(attempt),
                    policy_used=current_policy
                )
                attempts.append(attempt_info)

                # Wait before retry
                delay = self._calculate_delay(attempt)
                total_delay += delay
                if delay > 0:
                    await asyncio.sleep(delay)

        # All attempts exhausted
        return self._create_failure_result(
            self.max_retries, total_delay, attempts, current_policy,
            f"All {self.max_retries + 1} attempts exhausted", fallback_used
        )

    def _is_successful_execution(self, result: Dict[str, Any]) -> bool:
        """
        Determine if an execution result indicates success.

        Args:
            result: Execution result dictionary

        Returns:
            True if execution was successful
        """
        if not isinstance(result, dict):
            return False

        # Check status
        status = result.get('status')
        if isinstance(status, ExecutionStatus):
            return status == ExecutionStatus.COMPLETED
        elif isinstance(status, str):
            return status.lower() in ['completed', 'success', 'ok']

        # Check for error indicators
        if result.get('error') or result.get('error_message'):
            return False

        # Check if we have executed orders
        orders = result.get('orders', [])
        return len(orders) > 0

    def _classify_error(self, result: Dict[str, Any]) -> RetryErrorType:
        """
        Classify the type of error from execution result.

        Args:
            result: Execution result dictionary

        Returns:
            Classified error type
        """
        error_message = result.get('error_message', '').lower()

        # Network-related errors
        if any(keyword in error_message for keyword in ['network', 'connection', 'timeout']):
            if 'rate_limit' in error_message or 'too many requests' in error_message:
                return RetryErrorType.RATE_LIMIT
            elif 'timeout' in error_message:
                return RetryErrorType.EXCHANGE_TIMEOUT
            else:
                return RetryErrorType.NETWORK_ERROR

        # Exchange errors
        if any(keyword in error_message for keyword in ['exchange', 'api', 'server']):
            return RetryErrorType.EXCHANGE_ERROR

        # Liquidity errors
        if any(keyword in error_message for keyword in ['liquidity', 'volume', 'insufficient']):
            return RetryErrorType.INSUFFICIENT_LIQUIDITY

        # Price slippage
        if any(keyword in error_message for keyword in ['slippage', 'price']):
            return RetryErrorType.PRICE_SLIPPAGE

        return RetryErrorType.UNKNOWN_ERROR

    def _should_retry(self, error_type: RetryErrorType, attempt: int) -> bool:
        """
        Determine if we should retry based on error type and attempt number.

        Args:
            error_type: Type of error that occurred
            attempt: Current attempt number (0-based)

        Returns:
            True if we should retry
        """
        if attempt >= self.max_retries:
            return False

        # Check if the full error type or its category is in retry_on_errors
        error_category = error_type.value.split('_')[0]  # Get first part of error type
        return (error_type.value in self.retry_on_errors or 
                error_category in self.retry_on_errors)

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for next retry attempt using exponential backoff.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: base * 2^attempt
        delay = self.backoff_base * (2 ** attempt)

        # Cap at maximum delay
        delay = min(delay, self.max_backoff)

        # Add some jitter to prevent thundering herd
        jitter = delay * 0.1 * (0.5 - time.time() % 1)  # +/- 10% jitter
        delay += jitter

        return max(0, delay)

    async def _check_kill_switch(self, error_message: str) -> None:
        """
        Check if kill switch should be activated based on error patterns.

        Args:
            error_message: Error message to analyze
        """
        if not self.kill_switch_enabled:
            return

        # Simple error counting - in production, this could be more sophisticated
        error_key = error_message.lower()[:50]  # First 50 chars as key
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        total_errors = sum(self.error_counts.values())

        if total_errors >= self.kill_switch_threshold:
            self.kill_switch_activated = True
            self.logger.critical(f"Kill switch activated! {total_errors} errors in recent executions")

            trade_logger.performance("Kill Switch Activated", {
                'total_errors': total_errors,
                'error_threshold': self.kill_switch_threshold,
                'error_counts': self.error_counts
            })

    def _enrich_result_with_retry_info(
        self,
        result: Dict[str, Any],
        attempts: int,
        total_delay: float,
        attempt_history: List[RetryAttempt],
        fallback_used: bool,
        final_policy: ExecutionPolicy
    ) -> Dict[str, Any]:
        """
        Enrich execution result with retry information.

        Args:
            result: Original execution result
            attempts: Number of attempts made
            total_delay: Total delay time
            attempt_history: History of retry attempts
            fallback_used: Whether fallback was used
            final_policy: Final policy used

        Returns:
            Enriched result dictionary
        """
        result.update({
            'retries': attempts,
            'total_delay': total_delay,
            'attempt_history': [attempt.__dict__ for attempt in attempt_history],
            'fallback_used': fallback_used,
            'final_policy': final_policy.value
        })
        return result

    def _create_failure_result(
        self,
        attempts: int,
        total_delay: float,
        attempt_history: List[RetryAttempt],
        final_policy: ExecutionPolicy,
        error_message: str,
        fallback_used: bool = False
    ) -> Dict[str, Any]:
        """
        Create a failure result dictionary.

        Args:
            attempts: Number of attempts made
            total_delay: Total delay time
            attempt_history: History of retry attempts
            final_policy: Final policy used
            error_message: Error message
            fallback_used: Whether fallback was used

        Returns:
            Failure result dictionary
        """
        return {
            'status': ExecutionStatus.FAILED,
            'orders': [],
            'executed_amount': 0,
            'average_price': None,
            'total_cost': 0,
            'fees': 0,
            'slippage': 0,
            'retries': attempts,
            'total_delay': total_delay,
            'attempt_history': [attempt.__dict__ for attempt in attempt_history],
            'fallback_used': fallback_used,
            'final_policy': final_policy.value,
            'error_message': error_message
        }

    def is_kill_switch_active(self) -> bool:
        """
        Check if kill switch is currently active.

        Returns:
            True if kill switch is active
        """
        return self.kill_switch_activated

    def reset_kill_switch(self) -> None:
        """Reset the kill switch."""
        self.kill_switch_activated = False
        self.error_counts.clear()
        self.logger.info("Kill switch reset")

    def get_retry_statistics(self) -> Dict[str, Any]:
        """
        Get retry statistics.

        Returns:
            Dictionary with retry statistics
        """
        total_attempts = sum(len(attempts) for attempts in [[]])  # Placeholder
        successful_retries = 0  # Would track actual successes
        failed_retries = 0  # Would track actual failures

        return {
            'enabled': self.enabled,
            'max_retries': self.max_retries,
            'backoff_base': self.backoff_base,
            'max_backoff': self.max_backoff,
            'fallback_enabled': self.fallback_enabled,
            'fallback_policy': self.fallback_policy.value,
            'kill_switch_active': self.kill_switch_activated,
            'error_counts': self.error_counts.copy(),
            'retry_on_errors': list(self.retry_on_errors)
        }

    def update_retry_config(self, config: Dict[str, Any]) -> None:
        """
        Update retry configuration dynamically.

        Args:
            config: New configuration values
        """
        self.config.update(config)

        # Update instance variables
        self.enabled = config.get('enabled', self.enabled)
        self.max_retries = config.get('max_retries', self.max_retries)
        self.backoff_base = config.get('backoff_base', self.backoff_base)
        self.max_backoff = config.get('max_backoff', self.max_backoff)
        self.retry_on_errors = set(config.get('retry_on_errors', self.retry_on_errors))
        self.fallback_enabled = config.get('fallback_enabled', self.fallback_enabled)
        self.fallback_policy = ExecutionPolicy(config.get('fallback_policy', self.fallback_policy.value))
        self.fallback_on_attempt = config.get('fallback_on_attempt', self.fallback_on_attempt)
        self.kill_switch_enabled = config.get('kill_switch_enabled', self.kill_switch_enabled)
        self.kill_switch_threshold = config.get('kill_switch_threshold', self.kill_switch_threshold)

        self.logger.info("Retry configuration updated")
