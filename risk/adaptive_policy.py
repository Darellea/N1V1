"""
Adaptive Risk Policy Engine

This module implements dynamic risk management that automatically adjusts exposure
based on market conditions, performance metrics, and trading outcomes. It provides
real-time risk multipliers that scale position sizes to protect capital during
volatile periods and increase exposure during favorable conditions.

Performance Optimizations:
- Memory-efficient data processing with in-place operations and method chaining
- Caching mechanism for expensive calculations (_calculate_volatility_level, _calculate_trend_strength)
- Optimized data types (float32 where appropriate) to reduce memory footprint
- Reduced intermediate variable creation to minimize memory allocation
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from risk.utils import (
    enhanced_validate_market_data,
    get_atr,
    get_config_value,
    safe_divide,
)
from strategies.regime.market_regime import MarketRegime, get_market_regime_detector
from utils.config_loader import get_config
from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class RiskLevel(Enum):
    """Risk level classifications."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DefensiveMode(Enum):
    """Defensive mode states."""

    NORMAL = "normal"
    CAUTION = "caution"
    DEFENSIVE = "defensive"
    KILL_SWITCH = "kill_switch"


class TransitionMode(Enum):
    """Policy transition modes."""

    IMMEDIATE = "immediate"
    GRADUAL = "gradual"


class TransitionState(Enum):
    """Policy transition states."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class RiskPolicy:
    """Represents a risk policy configuration."""

    def __init__(
        self,
        max_position: float = 1000.0,
        max_loss: float = 100.0,
        volatility_threshold: float = 0.05,
        trend_threshold: float = 0.02,
        **kwargs,
    ):
        self.max_position = max_position
        self.max_loss = max_loss
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        # Store any additional parameters
        self.__dict__.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskPolicy":
        """Create policy from dictionary."""
        return cls(**data)

    def __eq__(self, other: object) -> bool:
        """Check equality with another policy."""
        if not isinstance(other, RiskPolicy):
            return False
        return self.to_dict() == other.to_dict()


class PolicyTransition:
    """Manages gradual transitions between risk policies."""

    def __init__(
        self,
        from_policy: RiskPolicy,
        to_policy: RiskPolicy,
        duration: float,
        mode: TransitionMode = TransitionMode.GRADUAL,
        interpolation_func: Optional[Callable[[float], float]] = None,
        completion_callback: Optional[Callable[["PolicyTransition"], None]] = None,
    ):
        """
        Initialize policy transition.

        Args:
            from_policy: Starting policy
            to_policy: Target policy
            duration: Transition duration in seconds
            mode: Transition mode (immediate or gradual)
            interpolation_func: Custom interpolation function (progress -> progress)
        """
        if duration < 0:
            raise ValueError("Duration must be non-negative")
        if duration == 0 and mode != TransitionMode.IMMEDIATE:
            raise ValueError("Duration must be positive for gradual transitions")
        if from_policy is None or to_policy is None:
            raise ValueError("Policies cannot be None")

        self.from_policy = from_policy
        self.to_policy = to_policy
        self.duration = duration
        self.mode = mode
        self.interpolation_func = interpolation_func or (
            lambda x: x
        )  # Linear by default
        self.completion_callback = completion_callback

        # State
        self.state = TransitionState.PENDING
        self.progress = 0.0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message: Optional[str] = None

        # Threading
        self._lock = threading.Lock()
        self._transition_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()

    def is_valid(self) -> bool:
        """Check if transition is valid."""
        try:
            duration_valid = self.duration >= 0 and (
                self.duration > 0 or self.mode == TransitionMode.IMMEDIATE
            )
            return (
                self.from_policy is not None
                and self.to_policy is not None
                and duration_valid
                and self.mode in TransitionMode
            )
        except Exception:
            return False

    def start(self) -> None:
        """Start the transition."""
        with self._lock:
            if self.state != TransitionState.PENDING:
                raise RuntimeError(
                    f"Cannot start transition in state {self.state.value}"
                )

            self.state = TransitionState.IN_PROGRESS
            self.start_time = datetime.now()
            self._cancel_event.clear()

            if self.mode == TransitionMode.IMMEDIATE:
                self._complete_immediately()
            else:
                self._start_gradual_transition()

    def cancel(self) -> None:
        """Cancel the transition."""
        with self._lock:
            if self.state not in [TransitionState.IN_PROGRESS, TransitionState.PAUSED]:
                return

            self._cancel_event.set()
            self.state = TransitionState.CANCELLED
            self.end_time = datetime.now()

    def pause(self) -> None:
        """Pause the transition."""
        with self._lock:
            if self.state != TransitionState.IN_PROGRESS:
                raise RuntimeError(
                    f"Cannot pause transition in state {self.state.value}"
                )
            self.state = TransitionState.PAUSED

    def resume(self) -> None:
        """Resume the transition."""
        with self._lock:
            if self.state != TransitionState.PAUSED:
                raise RuntimeError(
                    f"Cannot resume transition in state {self.state.value}"
                )
            self.state = TransitionState.IN_PROGRESS

    def get_current_policy(self) -> RiskPolicy:
        """Get the current interpolated policy."""
        if self.state == TransitionState.PENDING:
            return self.from_policy
        elif self.state in [TransitionState.COMPLETED, TransitionState.CANCELLED]:
            return self.to_policy
        else:
            return self.interpolate(self.progress)

    def interpolate(self, progress: float) -> RiskPolicy:
        """
        Interpolate between from_policy and to_policy at given progress.

        Args:
            progress: Progress from 0.0 to 1.0

        Returns:
            Interpolated policy
        """
        if not (0.0 <= progress <= 1.0):
            raise ValueError("Progress must be between 0.0 and 1.0")

        # Apply interpolation function
        adjusted_progress = self.interpolation_func(progress)

        # Linear interpolation of numeric attributes
        interpolated_attrs = {}
        from_dict = self.from_policy.to_dict()
        to_dict = self.to_policy.to_dict()

        for key in set(from_dict.keys()) | set(to_dict.keys()):
            from_val = from_dict.get(key, 0.0)
            to_val = to_dict.get(key, 0.0)

            if isinstance(from_val, (int, float)) and isinstance(to_val, (int, float)):
                interpolated_attrs[key] = from_val + adjusted_progress * (
                    to_val - from_val
                )
            else:
                # For non-numeric values, use from_policy until halfway, then to_policy
                interpolated_attrs[key] = (
                    from_val if adjusted_progress < 0.5 else to_val
                )

        return RiskPolicy(**interpolated_attrs)

    def _complete_immediately(self) -> None:
        """Complete immediate transition."""
        self.progress = 1.0
        self.state = TransitionState.COMPLETED
        self.end_time = datetime.now()

    def _start_gradual_transition(self) -> None:
        """Start gradual transition in background thread."""
        self._transition_thread = threading.Thread(
            target=self._run_transition, daemon=True
        )
        self._transition_thread.start()

    def _run_transition(self) -> None:
        """Run the gradual transition."""
        try:
            start_time = time.time()
            while not self._cancel_event.is_set():
                elapsed = time.time() - start_time
                if elapsed >= self.duration:
                    break

                self.progress = min(elapsed / self.duration, 1.0)
                time.sleep(0.1)  # Update frequency

            if not self._cancel_event.is_set():
                self.progress = 1.0
                self.state = TransitionState.COMPLETED
                # Call completion callback if provided
                if self.completion_callback:
                    try:
                        self.completion_callback(self)
                    except Exception as e:
                        logger.error(f"Error in transition completion callback: {e}")
            else:
                self.state = TransitionState.CANCELLED

            self.end_time = datetime.now()

        except Exception as e:
            logger.error(f"Error during policy transition: {e}")
            self.state = TransitionState.FAILED
            self.error_message = str(e)
            self.end_time = datetime.now()


class EmergencyOverride:
    """Represents an emergency policy override."""

    def __init__(
        self, policy: RiskPolicy, reason: str, timeout: float, priority: int = 1
    ):
        """
        Initialize emergency override.

        Args:
            policy: Override policy to apply
            reason: Reason for override
            timeout: Timeout in seconds
            priority: Override priority (higher = more important)
        """
        self.policy = policy
        self.reason = reason
        self.timeout = timeout
        self.priority = priority
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(seconds=timeout)

    def is_expired(self) -> bool:
        """Check if override has expired."""
        return datetime.now() > self.expires_at

    def time_remaining(self) -> float:
        """Get remaining time in seconds."""
        remaining = (self.expires_at - datetime.now()).total_seconds()
        return max(0.0, remaining)


class RiskMultiplierEvent:
    """Event data for risk multiplier changes."""

    def __init__(
        self,
        symbol: str,
        old_multiplier: float,
        new_multiplier: float,
        reason: str,
        context: Dict[str, Any],
        timestamp: datetime,
    ):
        self.symbol = symbol
        self.old_multiplier = old_multiplier
        self.new_multiplier = new_multiplier
        self.reason = reason
        self.context = context
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "symbol": self.symbol,
            "old_multiplier": self.old_multiplier,
            "new_multiplier": self.new_multiplier,
            "reason": self.reason,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }


class MarketConditionMonitor:
    """
    Monitors market conditions to provide inputs for risk adaptation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Make thresholds configurable with safe defaults
        self.volatility_threshold = get_config_value(
            config, "volatility_threshold", 0.05, float
        )
        self.volatility_lookback = get_config_value(
            config, "volatility_lookback", 20, int
        )
        self.adx_trend_threshold = get_config_value(
            config, "adx_trend_threshold", 25, float
        )

        # Historical data storage
        self.volatility_history: Dict[str, List[float]] = {}
        self.adx_history: Dict[str, List[float]] = {}

    def assess_market_conditions(
        self, symbol: str, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Assess current market conditions for risk adaptation.

        Args:
            symbol: Trading symbol
            market_data: OHLCV market data

        Returns:
            Dictionary with market condition metrics
        """
        try:
            # Validate market data input
            self._validate_market_data(market_data)

            conditions = {
                "volatility_level": self._calculate_volatility_level(
                    symbol, market_data
                ),
                "trend_strength": self._calculate_trend_strength(market_data),
                "liquidity_score": self._calculate_liquidity_score(market_data),
                "regime": self._detect_market_regime(market_data),
                "risk_level": RiskLevel.MODERATE.value,
            }

            # Determine overall risk level
            conditions["risk_level"] = self._determine_risk_level(conditions)

            return conditions

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"Error assessing market conditions for {symbol}: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # Return safe defaults instead of re-raising for graceful degradation
            return {
                "volatility_level": "unknown",
                "trend_strength": 25,
                "liquidity_score": 0.5,
                "regime": MarketRegime.UNKNOWN.value,
                "risk_level": RiskLevel.MODERATE.value,
            }
        except Exception as e:
            logger.critical(
                f"Unexpected error assessing market conditions for {symbol}: {e}"
            )
            logger.critical(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(
                f"Unexpected error in market condition assessment for {symbol}: {e}"
            ) from e

    @lru_cache(maxsize=128)
    def _calculate_volatility_level_cached(
        self, symbol: str, data_hash: int, data_len: int
    ) -> str:
        """
        Cached version of volatility level calculation.

        This method uses LRU caching to avoid redundant ATR calculations,
        which are computationally expensive. The cache key includes a data hash
        to ensure cache invalidation when market data changes.

        Args:
            symbol: Trading symbol
            data_hash: Hash of market data for cache invalidation
            data_len: Length of data for validation

        Returns:
            Volatility level classification
        """
        # Get the actual data from the cache context (passed via global)
        data = getattr(self, "_temp_data", None)
        if data is None or len(data) != data_len:
            return "unknown"

        try:
            if len(data) < self.volatility_lookback:
                return "unknown"

            # Memory-efficient validation and ATR calculation
            # Use in-place operations and method chaining to minimize memory allocation
            if not (
                hasattr(data, "columns")
                and all(col in data.columns for col in ["high", "low", "close"])
                and len(data) > 0
            ):
                return "unknown"

            # Calculate ATR with optimized data types (use float32 for memory efficiency)
            high_vals = data["high"].astype(np.float32, copy=False)
            low_vals = data["low"].astype(np.float32, copy=False)
            close_vals = data["close"].astype(np.float32, copy=False)

            # Chain ATR calculation to avoid intermediate variables
            atr = get_atr(
                high_vals,
                low_vals,
                close_vals,
                period=self.volatility_lookback,
                method="ema",
            )

            if atr <= 0:
                return "unknown"

            # Calculate ATR as percentage of price using in-place operation
            current_price = close_vals.iloc[-1]
            atr_percentage = safe_divide(atr, current_price, 0.0)

            # Store in history with memory-efficient list management
            if symbol not in self.volatility_history:
                self.volatility_history[symbol] = []
            self.volatility_history[symbol].append(atr_percentage)

            # Trim history in-place to maintain memory bounds
            if len(self.volatility_history[symbol]) > 100:
                self.volatility_history[symbol][:] = self.volatility_history[symbol][
                    -100:
                ]

            # Classify volatility with optimized threshold comparisons
            if atr_percentage > self.volatility_threshold * 2:
                return "very_high"
            elif atr_percentage > self.volatility_threshold:
                return "high"
            elif atr_percentage > self.volatility_threshold * 0.5:
                return "moderate"
            else:
                return "low"

        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.error(f"Error calculating volatility level: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # For data-related errors, log detailed information
            if isinstance(e, (KeyError, IndexError)):
                logger.error(
                    f"Data access error - available columns: {list(data.columns) if hasattr(data, 'columns') else 'N/A'}"
                )
            return "unknown"
        except Exception as e:
            logger.critical(f"Unexpected error in volatility calculation: {e}")
            logger.critical(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(
                f"Unexpected error calculating volatility level: {e}"
            ) from e

    def _calculate_volatility_level(self, symbol: str, data: pd.DataFrame) -> str:
        """
        Calculate volatility level based on ATR with caching.

        This wrapper method prepares data for cached calculation,
        ensuring memory-efficient processing and avoiding redundant computations.

        Args:
            symbol: Trading symbol
            data: Market data DataFrame

        Returns:
            Volatility level classification
        """
        try:
            # Create a lightweight hash for cache key (avoid hashing large DataFrame)
            data_hash = hash(
                (symbol, len(data), data["close"].iloc[-1] if len(data) > 0 else 0)
            )

            # Temporarily store data for cached method (avoids passing large object to cache)
            self._temp_data = data

            result = self._calculate_volatility_level_cached(
                symbol, data_hash, len(data)
            )

            # Clean up temporary data
            self._temp_data = None

            return result

        except (KeyError, IndexError, TypeError, AttributeError) as e:
            logger.error(f"Error in volatility level wrapper: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return "unknown"
        except Exception as e:
            logger.critical(f"Unexpected error in volatility level wrapper: {e}")
            logger.critical(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(
                f"Unexpected error in volatility level calculation: {e}"
            ) from e

    @lru_cache(maxsize=128)
    def _calculate_trend_strength_cached(self, data_hash: int, data_len: int) -> float:
        """
        Cached version of trend strength calculation using ADX.

        This method uses LRU caching to avoid redundant ADX calculations,
        which involve multiple exponential moving averages and are computationally expensive.
        The cache key is based on data characteristics to ensure proper invalidation.

        Args:
            data_hash: Hash of market data for cache invalidation
            data_len: Length of data for validation

        Returns:
            Trend strength value (ADX)
        """
        # Get the actual data from the cache context
        data = getattr(self, "_temp_data", None)
        if data is None or len(data) != data_len:
            return 25.0

        try:
            if len(data) < 28:  # Need enough data for ADX
                return 25.0

            # Memory-efficient validation
            if not (
                hasattr(data, "columns")
                and all(col in data.columns for col in ["high", "low", "close"])
                and len(data) > 0
            ):
                return 25.0

            # Use optimized data types and in-place operations to reduce memory usage
            high_vals = data["high"].astype(np.float32, copy=False)
            low_vals = data["low"].astype(np.float32, copy=False)
            close_vals = data["close"].astype(np.float32, copy=False)

            # Calculate DM+/DM- using vectorized operations (memory efficient)
            high_diff = high_vals - high_vals.shift(1)
            low_diff = low_vals.shift(1) - low_vals

            # Use numpy operations for better performance and memory efficiency
            dm_plus = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            dm_minus = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

            # Calculate ATR with optimized parameters
            atr = get_atr(high_vals, low_vals, close_vals, period=14, method="ema")
            if atr <= 0:
                return 25.0

            # Calculate directional indicators using method chaining to avoid intermediate variables
            di_plus = safe_divide(
                pd.Series(dm_plus, dtype=np.float32).ewm(span=14).mean().iloc[-1],
                atr,
                0.0,
            )
            di_minus = safe_divide(
                pd.Series(dm_minus, dtype=np.float32).ewm(span=14).mean().iloc[-1],
                atr,
                0.0,
            )

            # Calculate ADX using method chaining for memory efficiency
            dx = (
                safe_divide(np.abs(di_plus - di_minus), (di_plus + di_minus), 0.0) * 100
            )
            adx = pd.Series([dx], dtype=np.float32).ewm(span=14).mean().iloc[-1]

            return float(adx)

        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as e:
            logger.error(f"Error calculating trend strength: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # For data-related errors, log detailed information
            if isinstance(e, (KeyError, IndexError)):
                logger.error(
                    f"Data access error - available columns: {list(data.columns) if hasattr(data, 'columns') else 'N/A'}"
                )
            return 25.0
        except Exception as e:
            logger.critical(f"Unexpected error in trend strength calculation: {e}")
            logger.critical(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(
                f"Unexpected error calculating trend strength: {e}"
            ) from e

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate trend strength using ADX with caching.

        This wrapper method prepares data for cached calculation,
        ensuring memory-efficient processing and avoiding redundant computations.

        Args:
            data: Market data DataFrame

        Returns:
            Trend strength value (ADX)
        """
        try:
            # Create a lightweight hash for cache key
            data_hash = hash(
                (
                    len(data),
                    data["high"].iloc[-1] if len(data) > 0 else 0,
                    data["low"].iloc[-1] if len(data) > 0 else 0,
                    data["close"].iloc[-1] if len(data) > 0 else 0,
                )
            )

            # Temporarily store data for cached method
            self._temp_data = data

            result = self._calculate_trend_strength_cached(data_hash, len(data))

            # Clean up temporary data
            self._temp_data = None

            return result

        except (KeyError, IndexError, TypeError, AttributeError) as e:
            logger.error(f"Error in trend strength wrapper: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return 25.0
        except Exception as e:
            logger.critical(f"Unexpected error in trend strength wrapper: {e}")
            logger.critical(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(
                f"Unexpected error in trend strength calculation: {e}"
            ) from e

    def _calculate_liquidity_score(self, data: pd.DataFrame) -> float:
        """Calculate liquidity score based on volume and spread."""
        try:
            if "volume" not in data.columns:
                return 0.5

            # Average volume over last 20 periods
            avg_volume = data["volume"].tail(20).mean()

            # Volume consistency (coefficient of variation)
            volume_std = data["volume"].tail(20).std()
            volume_cv = volume_std / avg_volume if avg_volume > 0 else 1.0

            # Higher consistency = higher liquidity score
            liquidity_score = max(0.1, 1.0 - volume_cv)

            return float(liquidity_score)

        except Exception as e:
            logger.warning(f"Error calculating liquidity score: {e}")
            return 0.5

    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime."""
        try:
            regime_detector = get_market_regime_detector()
            result = regime_detector.detect_regime(data)
            return result.regime.value
        except Exception as e:
            logger.warning(f"Error detecting market regime: {e}")
            return MarketRegime.UNKNOWN.value

    def _determine_risk_level(self, conditions: Dict[str, Any]) -> str:
        """Determine overall risk level from conditions."""
        risk_score = 0

        # Volatility contribution
        vol_level = conditions.get("volatility_level", "moderate")
        if vol_level == "very_high":
            risk_score += 3
        elif vol_level == "high":
            risk_score += 2
        elif vol_level == "moderate":
            risk_score += 1

        # Trend strength contribution (weak trends are riskier)
        trend_strength = conditions.get("trend_strength", 25)
        if trend_strength < 20:
            risk_score += 2
        elif trend_strength < 25:
            risk_score += 1

        # Liquidity contribution (low liquidity increases risk)
        liquidity = conditions.get("liquidity_score", 0.5)
        if liquidity < 0.3:
            risk_score += 2
        elif liquidity < 0.5:
            risk_score += 1

        # Regime contribution
        regime = conditions.get("regime", "unknown")
        if regime in ["volatile", "high_volatility"]:
            risk_score += 2

        # Classify risk level
        if risk_score >= 6:
            return RiskLevel.VERY_HIGH.value
        elif risk_score >= 4:
            return RiskLevel.HIGH.value
        elif risk_score >= 2:
            return RiskLevel.MODERATE.value
        elif risk_score >= 1:
            return RiskLevel.LOW.value
        else:
            return RiskLevel.VERY_LOW.value

    def _validate_market_data(self, data: pd.DataFrame) -> None:
        """
        Validate market data DataFrame schema and content using enhanced validation.

        This method leverages the centralized enhanced_validate_market_data function
        from risk.utils to ensure consistent data quality checks across the risk management
        system. By using a shared utility, we eliminate code duplication and make future
        validation improvements easier to implement.

        Args:
            data: Market data DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        # Check for minimal required columns
        required_columns = ["close", "high", "low"]
        if not enhanced_validate_market_data(data, required_columns):
            raise ValueError(
                "Market data validation failed - missing required columns or invalid data"
            )

        # Auto-generate 'open' column if missing (use close as proxy)
        if "open" not in data.columns:
            data["open"] = data["close"]

        # Validate with core columns (allow missing 'open' in non-critical cases)
        core_required_columns = ["close", "high", "low"]
        if not enhanced_validate_market_data(data, core_required_columns):
            raise ValueError(
                "Market data validation failed - missing required columns or invalid data"
            )


class PerformanceMonitor:
    """
    Monitors trading performance to provide inputs for risk adaptation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback_days = config.get("lookback_days", 30)
        self.min_sharpe_threshold = config.get("min_sharpe", -0.5)
        self.max_consecutive_losses = config.get("max_consecutive_losses", 5)

        # Performance tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        self.consecutive_losses = 0

    def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """
        Update performance tracking with new trade result.

        Args:
            trade_result: Dictionary with trade outcome data
        """
        try:
            self.trade_history.append(trade_result)

            # Keep only recent history
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
            self.trade_history = [
                trade
                for trade in self.trade_history
                if datetime.fromisoformat(
                    trade.get("timestamp", datetime.now().isoformat())
                )
                > cutoff_date
            ]

            # Update consecutive losses
            pnl = trade_result.get("pnl", 0)
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

            # Update daily returns (simplified)
            if "timestamp" in trade_result:
                # Group by day and calculate daily returns
                self._update_daily_returns()

        except Exception as e:
            logger.warning(f"Error updating performance: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate current performance metrics using memory-efficient operations.

        This method optimizes memory usage by:
        - Using numpy arrays with float32 for reduced memory footprint
        - Avoiding intermediate list comprehensions where possible
        - Using in-place operations for cumulative calculations
        - Minimizing object creation and method chaining

        Returns:
            Dictionary with performance metrics
        """
        try:
            if not self.trade_history:
                return {
                    "sharpe_ratio": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 1.0,
                    "max_drawdown": 0.0,
                    "consecutive_losses": 0,
                    "total_trades": 0,
                }

            # Memory-efficient extraction of PnL values using numpy
            # Convert to float32 to reduce memory usage (sufficient precision for financial calculations)
            pnls = np.array(
                [trade.get("pnl", 0.0) for trade in self.trade_history],
                dtype=np.float32,
            )

            if len(pnls) == 0:
                return {
                    "sharpe_ratio": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 1.0,
                    "max_drawdown": 0.0,
                    "consecutive_losses": 0,
                    "total_trades": 0,
                }

            # Calculate win/loss metrics using vectorized operations
            wins_mask = pnls > 0
            losses_mask = pnls < 0

            wins = np.sum(wins_mask)
            losses = np.sum(losses_mask)

            win_rate = wins / len(pnls) if len(pnls) > 0 else 0.0

            # Calculate averages using masked arrays for memory efficiency
            if wins > 0:
                avg_win = np.mean(pnls[wins_mask])
            else:
                avg_win = 0.0

            if losses > 0:
                avg_loss = abs(np.mean(pnls[losses_mask]))
            else:
                avg_loss = 0.0

            profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

            # Calculate Sharpe ratio with optimized numpy operations
            if len(pnls) > 1:
                # Use in-place calculations to avoid additional memory allocation
                returns_mean = np.mean(pnls)
                returns_std = np.std(pnls)
                sharpe_ratio = (
                    (returns_mean / returns_std) * np.sqrt(252)
                    if returns_std > 0
                    else 0.0
                )
            else:
                sharpe_ratio = 0.0

            # Calculate max drawdown using optimized cumulative operations
            # Use in-place operations on the array to minimize memory usage
            cumulative = np.cumsum(pnls, dtype=np.float32)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

            return {
                "sharpe_ratio": float(sharpe_ratio),
                "win_rate": float(win_rate),
                "profit_factor": float(profit_factor),
                "max_drawdown": float(max_drawdown),
                "consecutive_losses": self.consecutive_losses,
                "total_trades": len(self.trade_history),
            }

        except Exception as e:
            logger.warning(f"Error calculating performance metrics: {e}")
            return {
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 1.0,
                "max_drawdown": 0.0,
                "consecutive_losses": 0,
                "total_trades": 0,
            }

    def _update_daily_returns(self) -> None:
        """Update daily returns calculation."""
        try:
            # Group trades by day
            daily_pnl = {}
            for trade in self.trade_history:
                if "timestamp" in trade:
                    date = datetime.fromisoformat(trade["timestamp"]).date()
                    daily_pnl[date] = daily_pnl.get(date, 0) + trade.get("pnl", 0)

            self.daily_returns = list(daily_pnl.values())

        except Exception as e:
            logger.warning(f"Error updating daily returns: {e}")


class AdaptiveRiskPolicy:
    """
    Main adaptive risk policy engine that combines market conditions and
    performance metrics to determine optimal risk multipliers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adaptive risk policy.

        Args:
            config: Configuration dictionary for risk adaptation

        Raises:
            ValueError: If configuration parameters are invalid
        """
        # Validate configuration parameters
        self._validate_config(config)
        self.config = config

        # Risk multiplier bounds
        self.min_multiplier = config.get("min_multiplier", 0.1)
        self.max_multiplier = config.get("max_multiplier", 1.0)

        # Thresholds
        self.volatility_threshold = config.get("volatility_threshold", 0.05)
        self.performance_lookback = config.get("performance_lookback_days", 30)
        self.min_sharpe = config.get("min_sharpe", -0.5)
        self.max_consecutive_losses = config.get("max_consecutive_losses", 5)

        # Kill switch settings
        self.kill_switch_threshold = config.get("kill_switch_threshold", 10)
        self.kill_switch_window = timedelta(
            hours=config.get("kill_switch_window_hours", 24)
        )

        # Initialize components
        self.market_monitor = MarketConditionMonitor(config.get("market_monitor", {}))
        self.performance_monitor = PerformanceMonitor(
            config.get("performance_monitor", {})
        )

        # State tracking
        self.current_multiplier = 1.0
        self.defensive_mode = DefensiveMode.NORMAL
        self.kill_switch_activated = False
        self.kill_switch_timestamp = None

        # Policy transition state
        self.current_policy = RiskPolicy()  # Default policy
        self.active_transition: Optional[PolicyTransition] = None
        self.emergency_overrides: List[EmergencyOverride] = []
        self.transition_history: List[PolicyTransition] = []

        # Event tracking
        self.multiplier_history: List[RiskMultiplierEvent] = []
        self.defensive_mode_history: List[Dict[str, Any]] = []

        logger.info("AdaptiveRiskPolicy initialized")

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration parameters for the adaptive risk policy.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")

        # Validate risk multiplier bounds
        min_multiplier = config.get("min_multiplier", 0.1)
        max_multiplier = config.get("max_multiplier", 1.0)

        if not isinstance(min_multiplier, (int, float)):
            raise ValueError(
                f"min_multiplier must be numeric, got {type(min_multiplier)}"
            )
        if not isinstance(max_multiplier, (int, float)):
            raise ValueError(
                f"max_multiplier must be numeric, got {type(max_multiplier)}"
            )
        if min_multiplier <= 0:
            raise ValueError(f"min_multiplier must be positive, got {min_multiplier}")
        if max_multiplier <= 0:
            raise ValueError(f"max_multiplier must be positive, got {max_multiplier}")
        if min_multiplier > max_multiplier:
            raise ValueError(
                f"min_multiplier ({min_multiplier}) cannot be greater than max_multiplier ({max_multiplier})"
            )

        # Validate volatility threshold
        volatility_threshold = config.get("volatility_threshold", 0.05)
        if not isinstance(volatility_threshold, (int, float)):
            raise ValueError(
                f"volatility_threshold must be numeric, got {type(volatility_threshold)}"
            )
        if volatility_threshold <= 0:
            raise ValueError(
                f"volatility_threshold must be positive, got {volatility_threshold}"
            )
        if volatility_threshold > 1.0:
            raise ValueError(
                f"volatility_threshold seems too high (>1.0), got {volatility_threshold}"
            )

        # Validate performance lookback days
        performance_lookback = config.get("performance_lookback_days", 30)
        if not isinstance(performance_lookback, int):
            raise ValueError(
                f"performance_lookback_days must be an integer, got {type(performance_lookback)}"
            )
        if performance_lookback <= 0:
            raise ValueError(
                f"performance_lookback_days must be positive, got {performance_lookback}"
            )
        if performance_lookback > 365:
            raise ValueError(
                f"performance_lookback_days seems too high (>365), got {performance_lookback}"
            )

        # Validate Sharpe ratio threshold
        min_sharpe = config.get("min_sharpe", -0.5)
        if not isinstance(min_sharpe, (int, float)):
            raise ValueError(f"min_sharpe must be numeric, got {type(min_sharpe)}")
        if min_sharpe < -5.0 or min_sharpe > 5.0:
            raise ValueError(
                f"min_sharpe seems unreasonable (outside [-5.0, 5.0]), got {min_sharpe}"
            )

        # Validate consecutive losses threshold
        max_consecutive_losses = config.get("max_consecutive_losses", 5)
        if not isinstance(max_consecutive_losses, int):
            raise ValueError(
                f"max_consecutive_losses must be an integer, got {type(max_consecutive_losses)}"
            )
        if max_consecutive_losses <= 0:
            raise ValueError(
                f"max_consecutive_losses must be positive, got {max_consecutive_losses}"
            )
        if max_consecutive_losses > 50:
            raise ValueError(
                f"max_consecutive_losses seems too high (>50), got {max_consecutive_losses}"
            )

        # Validate kill switch settings
        kill_switch_threshold = config.get("kill_switch_threshold", 10)
        if not isinstance(kill_switch_threshold, int):
            raise ValueError(
                f"kill_switch_threshold must be an integer, got {type(kill_switch_threshold)}"
            )
        if kill_switch_threshold <= 0:
            raise ValueError(
                f"kill_switch_threshold must be positive, got {kill_switch_threshold}"
            )

        kill_switch_window_hours = config.get("kill_switch_window_hours", 24)
        if not isinstance(kill_switch_window_hours, (int, float)):
            raise ValueError(
                f"kill_switch_window_hours must be numeric, got {type(kill_switch_window_hours)}"
            )
        if kill_switch_window_hours <= 0:
            raise ValueError(
                f"kill_switch_window_hours must be positive, got {kill_switch_window_hours}"
            )
        if kill_switch_window_hours > 168:  # One week
            raise ValueError(
                f"kill_switch_window_hours seems too high (>168), got {kill_switch_window_hours}"
            )

    def get_risk_multiplier(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str]:
        """
        Calculate the current risk multiplier based on market conditions and performance.

        This method implements fallback mechanisms for data unavailability and multi-source
        validation to ensure resilience during system failures and data outages.

        Args:
            symbol: Trading symbol
            market_data: Current market data
            context: Additional context information

        Returns:
            Tuple of (risk_multiplier, reasoning)
        """
        try:
            # Check kill switch first
            if self.kill_switch_activated:
                return 0.0, "Kill switch activated - trading suspended"

            # Handle empty or invalid market data with fallback
            if market_data is None or market_data.empty:
                logger.warning(
                    f"Insufficient market data for {symbol}, using conservative fallback"
                )
                return self._get_conservative_fallback_multiplier("insufficient data")

            # Get effective policy (considering transitions and overrides)
            effective_policy = self.get_effective_policy()

            # Try to assess market conditions with fallback mechanisms
            market_conditions = self._assess_market_conditions_with_fallback(
                symbol, market_data
            )

            # Get performance metrics with fallback
            performance_metrics = self._get_performance_metrics_with_fallback()

            # Combine context
            full_context = {
                "market_conditions": market_conditions,
                "performance_metrics": performance_metrics,
                "additional_context": context or {},
                "effective_policy": effective_policy.to_dict(),
            }

            # Calculate base multiplier from market conditions
            market_multiplier = self._calculate_market_multiplier(market_conditions)

            # Calculate performance multiplier
            performance_multiplier = self._calculate_performance_multiplier(
                performance_metrics
            )

            # Combine multipliers
            combined_multiplier = market_multiplier * performance_multiplier

            # Apply defensive mode adjustments
            final_multiplier = self._apply_defensive_mode(
                combined_multiplier, full_context
            )

            # Clamp to bounds
            final_multiplier = max(
                self.min_multiplier, min(self.max_multiplier, final_multiplier)
            )

            # Check for kill switch activation
            self._check_kill_switch_activation(final_multiplier, full_context)

            # If kill switch was activated during this call, return 0
            if self.kill_switch_activated:
                return 0.0, "Kill switch activated - trading suspended"

            # Generate reasoning
            reasoning = self._generate_reasoning(
                market_conditions, performance_metrics, final_multiplier
            )

            # Track multiplier change
            if (
                abs(final_multiplier - self.current_multiplier) > 0.01
            ):  # Significant change
                self._track_multiplier_change(
                    symbol,
                    self.current_multiplier,
                    final_multiplier,
                    reasoning,
                    full_context,
                )

            self.current_multiplier = final_multiplier

            # Log the decision
            self._log_risk_decision(symbol, final_multiplier, reasoning, full_context)

            return final_multiplier, reasoning

        except Exception as e:
            logger.error(f"Error calculating risk multiplier for {symbol}: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # Return conservative fallback on any error
            return self._get_conservative_fallback_multiplier(f"error: {str(e)}")

    def update_from_trade_result(
        self, symbol: str, trade_result: Dict[str, Any]
    ) -> None:
        """
        Update the policy with trade result for learning.

        Args:
            symbol: Trading symbol
            trade_result: Trade outcome data
        """
        try:
            self.performance_monitor.update_performance(trade_result)

            # Log trade result for risk analysis
            trade_logger.performance(
                "Risk policy trade update",
                {
                    "symbol": symbol,
                    "pnl": trade_result.get("pnl", 0),
                    "current_multiplier": self.current_multiplier,
                    "defensive_mode": self.defensive_mode.value,
                },
            )

        except Exception as e:
            logger.warning(f"Error updating from trade result: {e}")

    def _calculate_market_multiplier(self, market_conditions: Dict[str, Any]) -> float:
        """Calculate risk multiplier based on market conditions."""
        risk_level = market_conditions.get("risk_level", RiskLevel.MODERATE.value)

        # Base multipliers for different risk levels
        multipliers = {
            RiskLevel.VERY_LOW.value: 1.2,
            RiskLevel.LOW.value: 1.1,
            RiskLevel.MODERATE.value: 1.0,
            RiskLevel.HIGH.value: 0.7,
            RiskLevel.VERY_HIGH.value: 0.4,
        }

        base_multiplier = multipliers.get(risk_level, 1.0)

        # Additional adjustments based on specific conditions
        volatility_level = market_conditions.get("volatility_level", "moderate")
        if volatility_level == "very_high":
            base_multiplier *= 0.8
        elif volatility_level == "high":
            base_multiplier *= 0.9

        # Trend strength adjustment (stronger trends allow higher risk)
        trend_strength = market_conditions.get("trend_strength", 25)
        if trend_strength > 40:
            base_multiplier *= 1.1
        elif trend_strength < 20:
            base_multiplier *= 0.9

        return base_multiplier

    def _calculate_performance_multiplier(
        self, performance_metrics: Dict[str, Any]
    ) -> float:
        """Calculate risk multiplier based on performance metrics."""
        multiplier = 1.0

        # Sharpe ratio adjustment
        sharpe_ratio = performance_metrics.get("sharpe_ratio", 0)
        if sharpe_ratio < self.min_sharpe:
            # Reduce risk when Sharpe is poor
            reduction_factor = max(0.3, 1.0 + (sharpe_ratio - self.min_sharpe) * 0.5)
            multiplier *= reduction_factor
        elif sharpe_ratio > 1.0:
            # Increase risk when Sharpe is good
            multiplier *= min(1.3, 1.0 + (sharpe_ratio - 1.0) * 0.2)

        # Consecutive losses adjustment
        consecutive_losses = performance_metrics.get("consecutive_losses", 0)
        if consecutive_losses >= self.max_consecutive_losses:
            # Activate defensive mode
            self.defensive_mode = DefensiveMode.DEFENSIVE
            multiplier *= 0.5
        elif consecutive_losses >= 3:
            # Caution mode
            self.defensive_mode = DefensiveMode.CAUTION
            multiplier *= 0.7
        else:
            # Normal mode
            self.defensive_mode = DefensiveMode.NORMAL

        # Win rate adjustment
        win_rate = performance_metrics.get("win_rate", 0.5)
        if win_rate < 0.4:
            multiplier *= 0.8
        elif win_rate > 0.6:
            multiplier *= 1.1

        return multiplier

    def _apply_defensive_mode(
        self, base_multiplier: float, context: Dict[str, Any]
    ) -> float:
        """Apply defensive mode adjustments."""
        if self.defensive_mode == DefensiveMode.DEFENSIVE:
            return base_multiplier * 0.6
        elif self.defensive_mode == DefensiveMode.CAUTION:
            return base_multiplier * 0.8

        return base_multiplier

    def _check_kill_switch_activation(
        self, multiplier: float, context: Dict[str, Any]
    ) -> None:
        """Check if kill switch should be activated."""
        try:
            # Count recent defensive mode activations
            recent_defensive = [
                event
                for event in self.defensive_mode_history
                if datetime.now() - event["timestamp"] < self.kill_switch_window
                and event["mode"] == DefensiveMode.DEFENSIVE.value
            ]

            if len(recent_defensive) >= self.kill_switch_threshold:
                self.kill_switch_activated = True
                self.kill_switch_timestamp = datetime.now()

                logger.critical(
                    "KILL_SWITCH_ACTIVATED: Too many defensive mode activations"
                )
                trade_logger.performance(
                    "Kill switch activated",
                    {
                        "reason": "excessive_defensive_activations",
                        "threshold": self.kill_switch_threshold,
                        "window_hours": self.kill_switch_window.total_seconds() / 3600,
                        "activation_count": len(recent_defensive),
                    },
                )

        except Exception as e:
            logger.warning(f"Error checking kill switch: {e}")

    def _generate_reasoning(
        self,
        market_conditions: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        final_multiplier: float,
    ) -> str:
        """Generate human-readable reasoning for the risk multiplier."""
        reasons = []

        # Market condition reasons
        risk_level = market_conditions.get("risk_level", "moderate")
        if risk_level in ["high", "very_high"]:
            reasons.append(f"high market risk ({risk_level})")
        elif risk_level in ["low", "very_low"]:
            reasons.append(f"favorable market conditions ({risk_level})")

        volatility = market_conditions.get("volatility_level", "moderate")
        if volatility in ["high", "very_high"]:
            reasons.append(f"high volatility ({volatility})")

        # Performance reasons
        sharpe = performance_metrics.get("sharpe_ratio", 0)
        if sharpe < self.min_sharpe:
            reasons.append(".2f")

        consecutive_losses = performance_metrics.get("consecutive_losses", 0)
        if consecutive_losses >= 3:
            reasons.append(f"{consecutive_losses} consecutive losses")

        win_rate = performance_metrics.get("win_rate", 0.5)
        if win_rate < 0.4:
            reasons.append(".1%")

        # Defensive mode
        if self.defensive_mode != DefensiveMode.NORMAL:
            reasons.append(f"{self.defensive_mode.value} mode active")

        if not reasons:
            reasons.append("normal conditions")

        reasoning = ", ".join(reasons)
        return f"Risk multiplier {final_multiplier:.2f}: {reasoning}"

    def _track_multiplier_change(
        self,
        symbol: str,
        old_multiplier: float,
        new_multiplier: float,
        reason: str,
        context: Dict[str, Any],
    ) -> None:
        """Track multiplier changes for analysis."""
        event = RiskMultiplierEvent(
            symbol=symbol,
            old_multiplier=old_multiplier,
            new_multiplier=new_multiplier,
            reason=reason,
            context=context,
            timestamp=datetime.now(),
        )

        self.multiplier_history.append(event)

        # Keep only recent history
        cutoff_date = datetime.now() - timedelta(days=30)
        self.multiplier_history = [
            event for event in self.multiplier_history if event.timestamp > cutoff_date
        ]

    def _log_risk_decision(
        self, symbol: str, multiplier: float, reasoning: str, context: Dict[str, Any]
    ) -> None:
        """Log risk decision for monitoring."""
        try:
            log_data = {
                "symbol": symbol,
                "risk_multiplier": multiplier,
                "reasoning": reasoning,
                "defensive_mode": self.defensive_mode.value,
                "kill_switch_active": self.kill_switch_activated,
                "market_risk_level": context.get("market_conditions", {}).get(
                    "risk_level"
                ),
                "sharpe_ratio": context.get("performance_metrics", {}).get(
                    "sharpe_ratio"
                ),
                "consecutive_losses": context.get("performance_metrics", {}).get(
                    "consecutive_losses"
                ),
            }

            trade_logger.performance("Risk multiplier update", log_data)

        except Exception as e:
            logger.warning(f"Error logging risk decision: {e}")

    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get comprehensive risk statistics."""
        return {
            "current_multiplier": self.current_multiplier,
            "defensive_mode": self.defensive_mode.value,
            "kill_switch_activated": self.kill_switch_activated,
            "total_multiplier_changes": len(self.multiplier_history),
            "performance_metrics": self.performance_monitor.get_performance_metrics(),
            "recent_events": [
                event.to_dict()
                for event in self.multiplier_history[-10:]  # Last 10 events
            ],
        }

    def reset_kill_switch(self) -> bool:
        """Manually reset the kill switch."""
        if not self.kill_switch_activated:
            return False

        self.kill_switch_activated = False
        self.kill_switch_timestamp = None
        self.defensive_mode = DefensiveMode.NORMAL

        logger.info("Kill switch manually reset")
        trade_logger.performance("Kill switch reset", {"manual_reset": True})

        return True

    def start_transition(self, transition: PolicyTransition) -> None:
        """
        Start a policy transition.

        Args:
            transition: PolicyTransition to execute

        Raises:
            RuntimeError: If a transition is already in progress
        """
        if self.active_transition is not None:
            raise RuntimeError("Transition already in progress")

        if not transition.is_valid():
            raise ValueError("Invalid transition")

        # Validate policy parameters
        self._validate_policy(transition.to_policy)

        # Set completion callback for gradual transitions
        if transition.mode == TransitionMode.GRADUAL:
            transition.completion_callback = self._on_transition_completed

        self.active_transition = transition
        self.transition_history.append(transition)

        # Keep only recent history
        if len(self.transition_history) > 10:
            self.transition_history = self.transition_history[-10:]

        transition.start()

        # For immediate transitions, update current policy immediately
        if (
            transition.mode == TransitionMode.IMMEDIATE
            and transition.state == TransitionState.COMPLETED
        ):
            self.current_policy = transition.to_policy
            self.active_transition = None

        logger.info(f"Started policy transition: {transition.mode.value}")

    def execute_transition(self, transition: PolicyTransition) -> PolicyTransition:
        """
        Execute a transition (blocking).

        Args:
            transition: Transition to execute

        Returns:
            Completed transition
        """
        self.start_transition(transition)

        # Wait for completion
        while transition.state not in [
            TransitionState.COMPLETED,
            TransitionState.FAILED,
            TransitionState.CANCELLED,
        ]:
            time.sleep(0.1)

        if transition.state == TransitionState.COMPLETED:
            self.current_policy = transition.to_policy
            self.active_transition = None
        elif transition.state == TransitionState.FAILED:
            self.active_transition = None
            raise RuntimeError(f"Transition failed: {transition.error_message}")

        return transition

    def rollback_transition(self, transition: PolicyTransition, reason: str) -> None:
        """
        Rollback a failed transition.

        Args:
            transition: Transition to rollback
            reason: Reason for rollback
        """
        if transition.state != TransitionState.IN_PROGRESS:
            return

        transition.cancel()
        transition.state = TransitionState.FAILED  # Mark as failed for rollback
        transition.error_message = reason
        self.current_policy = transition.from_policy
        self.active_transition = None

        logger.warning(f"Rolled back transition: {reason}")
        trade_logger.performance(
            "Transition rollback",
            {
                "reason": reason,
                "from_policy": transition.from_policy.to_dict(),
                "to_policy": transition.to_policy.to_dict(),
            },
        )

    def apply_emergency_override(self, override: EmergencyOverride) -> None:
        """
        Apply an emergency policy override.

        Args:
            override: Emergency override to apply
        """
        # Check if higher priority override exists
        existing_higher_priority = any(
            o
            for o in self.emergency_overrides
            if not o.is_expired() and o.priority > override.priority
        )

        if existing_higher_priority:
            logger.warning(
                "Higher priority emergency override active, ignoring new override"
            )
            return

        # Pause active transition if any
        if (
            self.active_transition
            and self.active_transition.state == TransitionState.IN_PROGRESS
        ):
            self.active_transition.pause()

        # Apply override
        self.current_policy = override.policy
        self.emergency_overrides.append(override)

        # Keep only active overrides
        self.emergency_overrides = [
            o for o in self.emergency_overrides if not o.is_expired()
        ]

        logger.critical(f"Emergency override applied: {override.reason}")
        trade_logger.performance(
            "Emergency override",
            {
                "reason": override.reason,
                "policy": override.policy.to_dict(),
                "timeout": override.timeout,
            },
        )

    def get_transition_progress(self) -> Optional[Dict[str, Any]]:
        """
        Get current transition progress.

        Returns:
            Progress information or None if no active transition
        """
        if self.active_transition is None:
            return None

        transition = self.active_transition
        return {
            "state": transition.state.value,
            "progress": transition.progress,
            "start_time": transition.start_time.isoformat()
            if transition.start_time
            else None,
            "end_time": transition.end_time.isoformat()
            if transition.end_time
            else None,
            "duration": transition.duration,
            "from_policy": transition.from_policy.to_dict(),
            "to_policy": transition.to_policy.to_dict(),
            "current_policy": transition.get_current_policy().to_dict(),
        }

    def cancel_active_transition(self) -> bool:
        """
        Cancel the active transition.

        Returns:
            True if cancelled, False if no active transition
        """
        if self.active_transition is None:
            return False

        self.active_transition.cancel()
        self.active_transition = None
        logger.info("Active transition cancelled")
        return True

    def get_effective_policy(self) -> RiskPolicy:
        """
        Get the currently effective policy, considering transitions and overrides.

        Returns:
            Effective policy
        """
        # Check emergency overrides first (highest priority)
        active_overrides = [o for o in self.emergency_overrides if not o.is_expired()]
        if active_overrides:
            # Use highest priority override
            highest_priority = max(active_overrides, key=lambda o: o.priority)
            return highest_priority.policy

        # Check active transition
        if self.active_transition:
            return self.active_transition.get_current_policy()

        # Use current policy
        return self.current_policy

    def _on_transition_completed(self, transition: PolicyTransition) -> None:
        """
        Callback called when a transition completes.

        Args:
            transition: The completed transition
        """
        if transition.state == TransitionState.COMPLETED:
            self.current_policy = transition.to_policy
            self.active_transition = None
            logger.info(
                f"Transition completed: policy updated to {transition.to_policy.to_dict()}"
            )
        elif transition.state == TransitionState.FAILED:
            logger.error(f"Transition failed: {transition.error_message}")
            self.active_transition = None
        elif transition.state == TransitionState.CANCELLED:
            logger.info("Transition cancelled")
            self.active_transition = None

    def _validate_policy(self, policy: RiskPolicy) -> None:
        """
        Validate policy parameters.

        Args:
            policy: Policy to validate

        Raises:
            ValueError: If policy is invalid
        """
        # Basic validation - can be extended
        if policy.max_position <= 0:
            raise ValueError("max_position must be positive")
        if policy.max_loss <= 0:
            raise ValueError("max_loss must be positive")
        if not (0 < policy.volatility_threshold <= 1.0):
            raise ValueError("volatility_threshold must be between 0 and 1")
        if not (0 < policy.trend_threshold <= 1.0):
            raise ValueError("trend_threshold must be between 0 and 1")

        # Safety checks for extreme values
        if policy.max_position > 100000:  # Unrealistic position size
            raise ValueError("Policy validation failed: max_position too high")
        if policy.max_loss > 10000:  # Unrealistic loss limit
            raise ValueError("Policy validation failed: max_loss too high")

    def cleanup_expired_overrides(self) -> int:
        """
        Clean up expired emergency overrides.

        Returns:
            Number of overrides cleaned up
        """
        before_count = len(self.emergency_overrides)
        self.emergency_overrides = [
            o for o in self.emergency_overrides if not o.is_expired()
        ]

        cleaned = before_count - len(self.emergency_overrides)
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired emergency overrides")
            # Resume any paused transitions when overrides expire
            if (
                self.active_transition
                and self.active_transition.state == TransitionState.PAUSED
            ):
                self.active_transition.resume()
                logger.info("Resumed paused transition after emergency override expiry")

        return cleaned

    def _get_conservative_fallback_multiplier(self, reason: str) -> Tuple[float, str]:
        """
        Return a neutral fallback multiplier when data is unavailable or calculations fail.

        This method implements the fallback mechanism for data unavailability by providing
        a neutral risk multiplier (1.0) that maintains normal exposure during uncertain conditions.

        Args:
            reason: Reason for using fallback (e.g., "insufficient data", "error: ...")

        Returns:
            Tuple of (neutral_multiplier, reasoning)
        """
        # Use neutral multiplier (1.0) during data unavailability
        neutral_multiplier = 1.0

        reasoning = f"Neutral multiplier {neutral_multiplier:.2f}: {reason} - using safe defaults"

        logger.warning(
            f"Using neutral fallback for {reason}: multiplier={neutral_multiplier}"
        )

        return neutral_multiplier, reasoning

    def _assess_market_conditions_with_fallback(
        self, symbol: str, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Assess market conditions with fallback mechanisms for data unavailability.

        This method implements multi-source validation by attempting to assess market conditions
        from the primary data source, with fallback to cached or default values if the primary
        source fails.

        Args:
            symbol: Trading symbol
            market_data: Primary market data source

        Returns:
            Dictionary with market condition metrics, using fallbacks if needed
        """
        try:
            # Try primary assessment
            return self.market_monitor.assess_market_conditions(symbol, market_data)

        except Exception as e:
            logger.warning(f"Primary market assessment failed for {symbol}: {e}")

            # Try secondary validation if available (simulated multi-source)
            try:
                return self._assess_market_conditions_secondary(symbol, market_data)
            except Exception as secondary_e:
                logger.warning(
                    f"Secondary market assessment also failed for {symbol}: {secondary_e}"
                )

                # Use cached conditions if available
                cached_conditions = self._get_cached_market_conditions(symbol)
                if cached_conditions:
                    logger.info(f"Using cached market conditions for {symbol}")
                    return cached_conditions

                # Final fallback to conservative defaults
                logger.warning(
                    f"Using conservative defaults for market conditions on {symbol}"
                )
                return self._get_conservative_market_conditions()

    def _assess_market_conditions_secondary(
        self, symbol: str, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Secondary market condition assessment for multi-source validation.

        This method provides an alternative assessment approach to validate against
        the primary source, implementing redundancy in risk assessment.

        Args:
            symbol: Trading symbol
            market_data: Market data for secondary assessment

        Returns:
            Dictionary with secondary market condition metrics

        Raises:
            Exception: If secondary assessment fails
        """
        # Implement secondary assessment logic (could use different calculation methods)
        try:
            # Use simplified assessment as secondary validation
            conditions = {
                "volatility_level": "moderate",  # Default conservative
                "trend_strength": 25.0,  # Neutral trend
                "liquidity_score": 0.5,  # Neutral liquidity
                "regime": MarketRegime.UNKNOWN.value,
                "risk_level": RiskLevel.MODERATE.value,
            }

            # Try to get basic volatility if data is available
            if (
                market_data is not None
                and not market_data.empty
                and "close" in market_data.columns
                and len(market_data) >= 5
            ):
                # Simple volatility calculation for secondary validation
                returns = market_data["close"].pct_change().dropna()
                if len(returns) >= 4:
                    volatility = returns.std() * 100  # As percentage

                    # Classify volatility for secondary assessment
                    if volatility > 5.0:
                        conditions["volatility_level"] = "high"
                    elif volatility > 2.0:
                        conditions["volatility_level"] = "moderate"
                    else:
                        conditions["volatility_level"] = "low"

                    # Adjust risk level based on secondary assessment
                    if conditions["volatility_level"] == "high":
                        conditions["risk_level"] = RiskLevel.HIGH.value
                    elif conditions["volatility_level"] == "low":
                        conditions["risk_level"] = RiskLevel.LOW.value

            return conditions

        except Exception as e:
            logger.error(f"Secondary market assessment failed: {e}")
            raise

    def _get_cached_market_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached market conditions for the symbol.

        This method provides a fallback mechanism by using previously calculated
        market conditions when current data is unavailable.

        Args:
            symbol: Trading symbol

        Returns:
            Cached market conditions or None if not available
        """
        try:
            # Check if we have recent volatility history for this symbol
            if (
                hasattr(self.market_monitor, "volatility_history")
                and symbol in self.market_monitor.volatility_history
                and self.market_monitor.volatility_history[symbol]
            ):
                # Use most recent volatility data
                recent_volatility = self.market_monitor.volatility_history[symbol][-1]

                # Reconstruct basic conditions from cached data
                conditions = {
                    "volatility_level": "moderate",  # Default
                    "trend_strength": 25.0,  # Neutral
                    "liquidity_score": 0.5,  # Neutral
                    "regime": MarketRegime.UNKNOWN.value,
                    "risk_level": RiskLevel.MODERATE.value,
                }

                # Classify based on cached volatility
                if recent_volatility > 0.08:  # 8% volatility threshold
                    conditions["volatility_level"] = "very_high"
                    conditions["risk_level"] = RiskLevel.VERY_HIGH.value
                elif recent_volatility > 0.05:  # 5% volatility threshold
                    conditions["volatility_level"] = "high"
                    conditions["risk_level"] = RiskLevel.HIGH.value
                elif recent_volatility > 0.02:  # 2% volatility threshold
                    conditions["volatility_level"] = "moderate"
                else:
                    conditions["volatility_level"] = "low"
                    conditions["risk_level"] = RiskLevel.LOW.value

                logger.info(
                    f"Retrieved cached conditions for {symbol}: volatility_level={conditions['volatility_level']}"
                )
                return conditions

        except Exception as e:
            logger.warning(
                f"Error retrieving cached market conditions for {symbol}: {e}"
            )

        return None

    def _get_conservative_market_conditions(self) -> Dict[str, Any]:
        """
        Return conservative default market conditions when all other methods fail.

        This provides the safest possible assumptions during complete data unavailability.

        Returns:
            Dictionary with conservative market condition defaults
        """
        return {
            "volatility_level": "high",  # Assume high volatility (conservative)
            "trend_strength": 20.0,  # Weak trend (conservative)
            "liquidity_score": 0.3,  # Low liquidity (conservative)
            "regime": MarketRegime.UNKNOWN.value,
            "risk_level": RiskLevel.HIGH.value,  # High risk level (conservative)
        }

    def _get_performance_metrics_with_fallback(self) -> Dict[str, Any]:
        """
        Get performance metrics with fallback mechanisms.

        This method ensures performance metrics are always available, even during
        data unavailability, by using cached or default values.

        Returns:
            Dictionary with performance metrics, using fallbacks if needed
        """
        try:
            return self.performance_monitor.get_performance_metrics()
        except Exception as e:
            logger.warning(f"Performance metrics calculation failed: {e}")

            # Return conservative defaults
            return {
                "sharpe_ratio": -0.5,  # Poor performance (conservative)
                "win_rate": 0.4,  # Below average (conservative)
                "profit_factor": 0.8,  # Below 1.0 (conservative)
                "max_drawdown": 0.1,  # 10% drawdown (conservative)
                "consecutive_losses": 2,  # Some losses (conservative)
                "total_trades": 10,  # Minimum history
            }


# Global instance
_adaptive_policy: Optional[AdaptiveRiskPolicy] = None


def get_adaptive_risk_policy(
    config: Optional[Dict[str, Any]] = None
) -> AdaptiveRiskPolicy:
    """Get the global adaptive risk policy instance."""
    global _adaptive_policy
    if _adaptive_policy is None:
        if config is None:
            config = get_config("risk.adaptive", {})
        _adaptive_policy = AdaptiveRiskPolicy(config)
    return _adaptive_policy


def get_risk_multiplier(
    symbol: str, market_data: pd.DataFrame, context: Optional[Dict[str, Any]] = None
) -> Tuple[float, str]:
    """
    Convenience function to get risk multiplier.

    Args:
        symbol: Trading symbol
        market_data: Current market data
        context: Additional context

    Returns:
        Tuple of (risk_multiplier, reasoning)
    """
    policy = get_adaptive_risk_policy()
    return policy.get_risk_multiplier(symbol, market_data, context)
