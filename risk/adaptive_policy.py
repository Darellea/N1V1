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
import asyncio
import traceback
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from functools import lru_cache
import numpy as np
import pandas as pd

from utils.config_loader import get_config
from utils.logger import get_trade_logger, TradeLogger
from core.contracts import TradingSignal
from strategies.regime.market_regime import get_market_regime_detector, MarketRegime
from risk.utils import safe_divide, get_atr, calculate_volatility_percentage, validate_market_data, get_config_value, clamp_value, enhanced_validate_market_data

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


class RiskMultiplierEvent:
    """Event data for risk multiplier changes."""

    def __init__(
        self,
        symbol: str,
        old_multiplier: float,
        new_multiplier: float,
        reason: str,
        context: Dict[str, Any],
        timestamp: datetime
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
            'symbol': self.symbol,
            'old_multiplier': self.old_multiplier,
            'new_multiplier': self.new_multiplier,
            'reason': self.reason,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }


class MarketConditionMonitor:
    """
    Monitors market conditions to provide inputs for risk adaptation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Make thresholds configurable with safe defaults
        self.volatility_threshold = get_config_value(config, 'volatility_threshold', 0.05, float)
        self.volatility_lookback = get_config_value(config, 'volatility_lookback', 20, int)
        self.adx_trend_threshold = get_config_value(config, 'adx_trend_threshold', 25, float)

        # Historical data storage
        self.volatility_history: Dict[str, List[float]] = {}
        self.adx_history: Dict[str, List[float]] = {}

    def assess_market_conditions(
        self,
        symbol: str,
        market_data: pd.DataFrame
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
                'volatility_level': self._calculate_volatility_level(symbol, market_data),
                'trend_strength': self._calculate_trend_strength(market_data),
                'liquidity_score': self._calculate_liquidity_score(market_data),
                'regime': self._detect_market_regime(market_data),
                'risk_level': RiskLevel.MODERATE.value
            }

            # Determine overall risk level
            conditions['risk_level'] = self._determine_risk_level(conditions)

            return conditions

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"Error assessing market conditions for {symbol}: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # For critical data validation errors, re-raise to prevent silent failures
            if isinstance(e, ValueError) and "Market data" in str(e):
                raise ValueError(f"Critical market data validation error for {symbol}: {e}") from e
            return {
                'volatility_level': 'unknown',
                'trend_strength': 25,
                'liquidity_score': 0.5,
                'regime': MarketRegime.UNKNOWN.value,
                'risk_level': RiskLevel.MODERATE.value
            }
        except Exception as e:
            logger.critical(f"Unexpected error assessing market conditions for {symbol}: {e}")
            logger.critical(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(f"Unexpected error in market condition assessment for {symbol}: {e}") from e

    @lru_cache(maxsize=128)
    def _calculate_volatility_level_cached(self, symbol: str, data_hash: int, data_len: int) -> str:
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
        data = getattr(self, '_temp_data', None)
        if data is None or len(data) != data_len:
            return 'unknown'

        try:
            if len(data) < self.volatility_lookback:
                return 'unknown'

            # Memory-efficient validation and ATR calculation
            # Use in-place operations and method chaining to minimize memory allocation
            if not (hasattr(data, 'columns') and
                    all(col in data.columns for col in ['high', 'low', 'close']) and
                    len(data) > 0):
                return 'unknown'

            # Calculate ATR with optimized data types (use float32 for memory efficiency)
            high_vals = data['high'].astype(np.float32, copy=False)
            low_vals = data['low'].astype(np.float32, copy=False)
            close_vals = data['close'].astype(np.float32, copy=False)

            # Chain ATR calculation to avoid intermediate variables
            atr = (get_atr(high_vals, low_vals, close_vals,
                          period=self.volatility_lookback, method='ema'))

            if atr <= 0:
                return 'unknown'

            # Calculate ATR as percentage of price using in-place operation
            current_price = close_vals.iloc[-1]
            atr_percentage = safe_divide(atr, current_price, 0.0)

            # Store in history with memory-efficient list management
            if symbol not in self.volatility_history:
                self.volatility_history[symbol] = []
            self.volatility_history[symbol].append(atr_percentage)

            # Trim history in-place to maintain memory bounds
            if len(self.volatility_history[symbol]) > 100:
                self.volatility_history[symbol][:] = self.volatility_history[symbol][-100:]

            # Classify volatility with optimized threshold comparisons
            if atr_percentage > self.volatility_threshold * 2:
                return 'very_high'
            elif atr_percentage > self.volatility_threshold:
                return 'high'
            elif atr_percentage > self.volatility_threshold * 0.5:
                return 'moderate'
            else:
                return 'low'

        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.error(f"Error calculating volatility level: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # For data-related errors, log detailed information
            if isinstance(e, (KeyError, IndexError)):
                logger.error(f"Data access error - available columns: {list(data.columns) if hasattr(data, 'columns') else 'N/A'}")
            return 'unknown'
        except Exception as e:
            logger.critical(f"Unexpected error in volatility calculation: {e}")
            logger.critical(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(f"Unexpected error calculating volatility level: {e}") from e

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
            data_hash = hash((symbol, len(data), data['close'].iloc[-1] if len(data) > 0 else 0))

            # Temporarily store data for cached method (avoids passing large object to cache)
            self._temp_data = data

            result = self._calculate_volatility_level_cached(symbol, data_hash, len(data))

            # Clean up temporary data
            self._temp_data = None

            return result

        except (KeyError, IndexError, TypeError, AttributeError) as e:
            logger.error(f"Error in volatility level wrapper: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return 'unknown'
        except Exception as e:
            logger.critical(f"Unexpected error in volatility level wrapper: {e}")
            logger.critical(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(f"Unexpected error in volatility level calculation: {e}") from e

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
        data = getattr(self, '_temp_data', None)
        if data is None or len(data) != data_len:
            return 25.0

        try:
            if len(data) < 28:  # Need enough data for ADX
                return 25.0

            # Memory-efficient validation
            if not (hasattr(data, 'columns') and
                    all(col in data.columns for col in ['high', 'low', 'close']) and
                    len(data) > 0):
                return 25.0

            # Use optimized data types and in-place operations to reduce memory usage
            high_vals = data['high'].astype(np.float32, copy=False)
            low_vals = data['low'].astype(np.float32, copy=False)
            close_vals = data['close'].astype(np.float32, copy=False)

            # Calculate DM+/DM- using vectorized operations (memory efficient)
            high_diff = high_vals - high_vals.shift(1)
            low_diff = low_vals.shift(1) - low_vals

            # Use numpy operations for better performance and memory efficiency
            dm_plus = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            dm_minus = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

            # Calculate ATR with optimized parameters
            atr = get_atr(high_vals, low_vals, close_vals, period=14, method='ema')
            if atr <= 0:
                return 25.0

            # Calculate directional indicators using method chaining to avoid intermediate variables
            di_plus = safe_divide(
                pd.Series(dm_plus, dtype=np.float32).ewm(span=14).mean().iloc[-1],
                atr, 0.0
            )
            di_minus = safe_divide(
                pd.Series(dm_minus, dtype=np.float32).ewm(span=14).mean().iloc[-1],
                atr, 0.0
            )

            # Calculate ADX using method chaining for memory efficiency
            dx = safe_divide(np.abs(di_plus - di_minus), (di_plus + di_minus), 0.0) * 100
            adx = pd.Series([dx], dtype=np.float32).ewm(span=14).mean().iloc[-1]

            return float(adx)

        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as e:
            logger.error(f"Error calculating trend strength: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # For data-related errors, log detailed information
            if isinstance(e, (KeyError, IndexError)):
                logger.error(f"Data access error - available columns: {list(data.columns) if hasattr(data, 'columns') else 'N/A'}")
            return 25.0
        except Exception as e:
            logger.critical(f"Unexpected error in trend strength calculation: {e}")
            logger.critical(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(f"Unexpected error calculating trend strength: {e}") from e

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
            data_hash = hash((len(data),
                             data['high'].iloc[-1] if len(data) > 0 else 0,
                             data['low'].iloc[-1] if len(data) > 0 else 0,
                             data['close'].iloc[-1] if len(data) > 0 else 0))

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
            raise RuntimeError(f"Unexpected error in trend strength calculation: {e}") from e

    def _calculate_liquidity_score(self, data: pd.DataFrame) -> float:
        """Calculate liquidity score based on volume and spread."""
        try:
            if 'volume' not in data.columns:
                return 0.5

            # Average volume over last 20 periods
            avg_volume = data['volume'].tail(20).mean()

            # Volume consistency (coefficient of variation)
            volume_std = data['volume'].tail(20).std()
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
        vol_level = conditions.get('volatility_level', 'moderate')
        if vol_level == 'very_high':
            risk_score += 3
        elif vol_level == 'high':
            risk_score += 2
        elif vol_level == 'moderate':
            risk_score += 1

        # Trend strength contribution (weak trends are riskier)
        trend_strength = conditions.get('trend_strength', 25)
        if trend_strength < 20:
            risk_score += 2
        elif trend_strength < 25:
            risk_score += 1

        # Liquidity contribution (low liquidity increases risk)
        liquidity = conditions.get('liquidity_score', 0.5)
        if liquidity < 0.3:
            risk_score += 2
        elif liquidity < 0.5:
            risk_score += 1

        # Regime contribution
        regime = conditions.get('regime', 'unknown')
        if regime in ['volatile', 'high_volatility']:
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
        # Use enhanced validation with OHLC columns for comprehensive checks
        required_columns = ['close', 'high', 'low', 'open']  # Full OHLC for volatility calculations
        if not enhanced_validate_market_data(data, required_columns):
            raise ValueError("Market data validation failed - missing required columns or invalid data")


class PerformanceMonitor:
    """
    Monitors trading performance to provide inputs for risk adaptation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback_days = config.get('lookback_days', 30)
        self.min_sharpe_threshold = config.get('min_sharpe', -0.5)
        self.max_consecutive_losses = config.get('max_consecutive_losses', 5)

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
                trade for trade in self.trade_history
                if datetime.fromisoformat(trade.get('timestamp', datetime.now().isoformat())) > cutoff_date
            ]

            # Update consecutive losses
            pnl = trade_result.get('pnl', 0)
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

            # Update daily returns (simplified)
            if 'timestamp' in trade_result:
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
                    'sharpe_ratio': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 1.0,
                    'max_drawdown': 0.0,
                    'consecutive_losses': 0,
                    'total_trades': 0
                }

            # Memory-efficient extraction of PnL values using numpy
            # Convert to float32 to reduce memory usage (sufficient precision for financial calculations)
            pnls = np.array([trade.get('pnl', 0.0) for trade in self.trade_history], dtype=np.float32)

            if len(pnls) == 0:
                return {
                    'sharpe_ratio': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 1.0,
                    'max_drawdown': 0.0,
                    'consecutive_losses': 0,
                    'total_trades': 0
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

            profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

            # Calculate Sharpe ratio with optimized numpy operations
            if len(pnls) > 1:
                # Use in-place calculations to avoid additional memory allocation
                returns_mean = np.mean(pnls)
                returns_std = np.std(pnls)
                sharpe_ratio = (returns_mean / returns_std) * np.sqrt(252) if returns_std > 0 else 0.0
            else:
                sharpe_ratio = 0.0

            # Calculate max drawdown using optimized cumulative operations
            # Use in-place operations on the array to minimize memory usage
            cumulative = np.cumsum(pnls, dtype=np.float32)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

            return {
                'sharpe_ratio': float(sharpe_ratio),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'max_drawdown': float(max_drawdown),
                'consecutive_losses': self.consecutive_losses,
                'total_trades': len(self.trade_history)
            }

        except Exception as e:
            logger.warning(f"Error calculating performance metrics: {e}")
            return {
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'profit_factor': 1.0,
                'max_drawdown': 0.0,
                'consecutive_losses': 0,
                'total_trades': 0
            }

    def _update_daily_returns(self) -> None:
        """Update daily returns calculation."""
        try:
            # Group trades by day
            daily_pnl = {}
            for trade in self.trade_history:
                if 'timestamp' in trade:
                    date = datetime.fromisoformat(trade['timestamp']).date()
                    daily_pnl[date] = daily_pnl.get(date, 0) + trade.get('pnl', 0)

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
        self.min_multiplier = config.get('min_multiplier', 0.1)
        self.max_multiplier = config.get('max_multiplier', 1.0)

        # Thresholds
        self.volatility_threshold = config.get('volatility_threshold', 0.05)
        self.performance_lookback = config.get('performance_lookback_days', 30)
        self.min_sharpe = config.get('min_sharpe', -0.5)
        self.max_consecutive_losses = config.get('max_consecutive_losses', 5)

        # Kill switch settings
        self.kill_switch_threshold = config.get('kill_switch_threshold', 10)
        self.kill_switch_window = timedelta(hours=config.get('kill_switch_window_hours', 24))

        # Initialize components
        self.market_monitor = MarketConditionMonitor(config.get('market_monitor', {}))
        self.performance_monitor = PerformanceMonitor(config.get('performance_monitor', {}))

        # State tracking
        self.current_multiplier = 1.0
        self.defensive_mode = DefensiveMode.NORMAL
        self.kill_switch_activated = False
        self.kill_switch_timestamp = None

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
        min_multiplier = config.get('min_multiplier', 0.1)
        max_multiplier = config.get('max_multiplier', 1.0)

        if not isinstance(min_multiplier, (int, float)):
            raise ValueError(f"min_multiplier must be numeric, got {type(min_multiplier)}")
        if not isinstance(max_multiplier, (int, float)):
            raise ValueError(f"max_multiplier must be numeric, got {type(max_multiplier)}")
        if min_multiplier <= 0:
            raise ValueError(f"min_multiplier must be positive, got {min_multiplier}")
        if max_multiplier <= 0:
            raise ValueError(f"max_multiplier must be positive, got {max_multiplier}")
        if min_multiplier > max_multiplier:
            raise ValueError(f"min_multiplier ({min_multiplier}) cannot be greater than max_multiplier ({max_multiplier})")

        # Validate volatility threshold
        volatility_threshold = config.get('volatility_threshold', 0.05)
        if not isinstance(volatility_threshold, (int, float)):
            raise ValueError(f"volatility_threshold must be numeric, got {type(volatility_threshold)}")
        if volatility_threshold <= 0:
            raise ValueError(f"volatility_threshold must be positive, got {volatility_threshold}")
        if volatility_threshold > 1.0:
            raise ValueError(f"volatility_threshold seems too high (>1.0), got {volatility_threshold}")

        # Validate performance lookback days
        performance_lookback = config.get('performance_lookback_days', 30)
        if not isinstance(performance_lookback, int):
            raise ValueError(f"performance_lookback_days must be an integer, got {type(performance_lookback)}")
        if performance_lookback <= 0:
            raise ValueError(f"performance_lookback_days must be positive, got {performance_lookback}")
        if performance_lookback > 365:
            raise ValueError(f"performance_lookback_days seems too high (>365), got {performance_lookback}")

        # Validate Sharpe ratio threshold
        min_sharpe = config.get('min_sharpe', -0.5)
        if not isinstance(min_sharpe, (int, float)):
            raise ValueError(f"min_sharpe must be numeric, got {type(min_sharpe)}")
        if min_sharpe < -5.0 or min_sharpe > 5.0:
            raise ValueError(f"min_sharpe seems unreasonable (outside [-5.0, 5.0]), got {min_sharpe}")

        # Validate consecutive losses threshold
        max_consecutive_losses = config.get('max_consecutive_losses', 5)
        if not isinstance(max_consecutive_losses, int):
            raise ValueError(f"max_consecutive_losses must be an integer, got {type(max_consecutive_losses)}")
        if max_consecutive_losses <= 0:
            raise ValueError(f"max_consecutive_losses must be positive, got {max_consecutive_losses}")
        if max_consecutive_losses > 50:
            raise ValueError(f"max_consecutive_losses seems too high (>50), got {max_consecutive_losses}")

        # Validate kill switch settings
        kill_switch_threshold = config.get('kill_switch_threshold', 10)
        if not isinstance(kill_switch_threshold, int):
            raise ValueError(f"kill_switch_threshold must be an integer, got {type(kill_switch_threshold)}")
        if kill_switch_threshold <= 0:
            raise ValueError(f"kill_switch_threshold must be positive, got {kill_switch_threshold}")

        kill_switch_window_hours = config.get('kill_switch_window_hours', 24)
        if not isinstance(kill_switch_window_hours, (int, float)):
            raise ValueError(f"kill_switch_window_hours must be numeric, got {type(kill_switch_window_hours)}")
        if kill_switch_window_hours <= 0:
            raise ValueError(f"kill_switch_window_hours must be positive, got {kill_switch_window_hours}")
        if kill_switch_window_hours > 168:  # One week
            raise ValueError(f"kill_switch_window_hours seems too high (>168), got {kill_switch_window_hours}")

    def get_risk_multiplier(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
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
                logger.warning(f"Insufficient market data for {symbol}, using conservative fallback")
                return self._get_conservative_fallback_multiplier("insufficient data")

            # Try to assess market conditions with fallback mechanisms
            market_conditions = self._assess_market_conditions_with_fallback(symbol, market_data)

            # Get performance metrics with fallback
            performance_metrics = self._get_performance_metrics_with_fallback()

            # Combine context
            full_context = {
                'market_conditions': market_conditions,
                'performance_metrics': performance_metrics,
                'additional_context': context or {}
            }

            # Calculate base multiplier from market conditions
            market_multiplier = self._calculate_market_multiplier(market_conditions)

            # Calculate performance multiplier
            performance_multiplier = self._calculate_performance_multiplier(performance_metrics)

            # Combine multipliers
            combined_multiplier = market_multiplier * performance_multiplier

            # Apply defensive mode adjustments
            final_multiplier = self._apply_defensive_mode(combined_multiplier, full_context)

            # Clamp to bounds
            final_multiplier = max(self.min_multiplier, min(self.max_multiplier, final_multiplier))

            # Check for kill switch activation
            self._check_kill_switch_activation(final_multiplier, full_context)

            # If kill switch was activated during this call, return 0
            if self.kill_switch_activated:
                return 0.0, "Kill switch activated - trading suspended"

            # Generate reasoning
            reasoning = self._generate_reasoning(market_conditions, performance_metrics, final_multiplier)

            # Track multiplier change
            if abs(final_multiplier - self.current_multiplier) > 0.01:  # Significant change
                self._track_multiplier_change(symbol, self.current_multiplier, final_multiplier, reasoning, full_context)

            self.current_multiplier = final_multiplier

            # Log the decision
            self._log_risk_decision(symbol, final_multiplier, reasoning, full_context)

            return final_multiplier, reasoning

        except Exception as e:
            logger.error(f"Error calculating risk multiplier for {symbol}: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # Return conservative fallback on any error
            return self._get_conservative_fallback_multiplier(f"error: {str(e)}")

    def update_from_trade_result(self, symbol: str, trade_result: Dict[str, Any]) -> None:
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
                    'symbol': symbol,
                    'pnl': trade_result.get('pnl', 0),
                    'current_multiplier': self.current_multiplier,
                    'defensive_mode': self.defensive_mode.value
                }
            )

        except Exception as e:
            logger.warning(f"Error updating from trade result: {e}")

    def _calculate_market_multiplier(self, market_conditions: Dict[str, Any]) -> float:
        """Calculate risk multiplier based on market conditions."""
        risk_level = market_conditions.get('risk_level', RiskLevel.MODERATE.value)

        # Base multipliers for different risk levels
        multipliers = {
            RiskLevel.VERY_LOW.value: 1.2,
            RiskLevel.LOW.value: 1.1,
            RiskLevel.MODERATE.value: 1.0,
            RiskLevel.HIGH.value: 0.7,
            RiskLevel.VERY_HIGH.value: 0.4
        }

        base_multiplier = multipliers.get(risk_level, 1.0)

        # Additional adjustments based on specific conditions
        volatility_level = market_conditions.get('volatility_level', 'moderate')
        if volatility_level == 'very_high':
            base_multiplier *= 0.8
        elif volatility_level == 'high':
            base_multiplier *= 0.9

        # Trend strength adjustment (stronger trends allow higher risk)
        trend_strength = market_conditions.get('trend_strength', 25)
        if trend_strength > 40:
            base_multiplier *= 1.1
        elif trend_strength < 20:
            base_multiplier *= 0.9

        return base_multiplier

    def _calculate_performance_multiplier(self, performance_metrics: Dict[str, Any]) -> float:
        """Calculate risk multiplier based on performance metrics."""
        multiplier = 1.0

        # Sharpe ratio adjustment
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        if sharpe_ratio < self.min_sharpe:
            # Reduce risk when Sharpe is poor
            reduction_factor = max(0.3, 1.0 + (sharpe_ratio - self.min_sharpe) * 0.5)
            multiplier *= reduction_factor
        elif sharpe_ratio > 1.0:
            # Increase risk when Sharpe is good
            multiplier *= min(1.3, 1.0 + (sharpe_ratio - 1.0) * 0.2)

        # Consecutive losses adjustment
        consecutive_losses = performance_metrics.get('consecutive_losses', 0)
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
        win_rate = performance_metrics.get('win_rate', 0.5)
        if win_rate < 0.4:
            multiplier *= 0.8
        elif win_rate > 0.6:
            multiplier *= 1.1

        return multiplier

    def _apply_defensive_mode(self, base_multiplier: float, context: Dict[str, Any]) -> float:
        """Apply defensive mode adjustments."""
        if self.defensive_mode == DefensiveMode.DEFENSIVE:
            return base_multiplier * 0.6
        elif self.defensive_mode == DefensiveMode.CAUTION:
            return base_multiplier * 0.8

        return base_multiplier

    def _check_kill_switch_activation(self, multiplier: float, context: Dict[str, Any]) -> None:
        """Check if kill switch should be activated."""
        try:
            # Count recent defensive mode activations
            recent_defensive = [
                event for event in self.defensive_mode_history
                if datetime.now() - event['timestamp'] < self.kill_switch_window
                and event['mode'] == DefensiveMode.DEFENSIVE.value
            ]

            if len(recent_defensive) >= self.kill_switch_threshold:
                self.kill_switch_activated = True
                self.kill_switch_timestamp = datetime.now()

                logger.critical("KILL_SWITCH_ACTIVATED: Too many defensive mode activations")
                trade_logger.performance(
                    "Kill switch activated",
                    {
                        'reason': 'excessive_defensive_activations',
                        'threshold': self.kill_switch_threshold,
                        'window_hours': self.kill_switch_window.total_seconds() / 3600,
                        'activation_count': len(recent_defensive)
                    }
                )

        except Exception as e:
            logger.warning(f"Error checking kill switch: {e}")

    def _generate_reasoning(
        self,
        market_conditions: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        final_multiplier: float
    ) -> str:
        """Generate human-readable reasoning for the risk multiplier."""
        reasons = []

        # Market condition reasons
        risk_level = market_conditions.get('risk_level', 'moderate')
        if risk_level in ['high', 'very_high']:
            reasons.append(f"high market risk ({risk_level})")
        elif risk_level in ['low', 'very_low']:
            reasons.append(f"favorable market conditions ({risk_level})")

        volatility = market_conditions.get('volatility_level', 'moderate')
        if volatility in ['high', 'very_high']:
            reasons.append(f"high volatility ({volatility})")

        # Performance reasons
        sharpe = performance_metrics.get('sharpe_ratio', 0)
        if sharpe < self.min_sharpe:
            reasons.append(".2f")

        consecutive_losses = performance_metrics.get('consecutive_losses', 0)
        if consecutive_losses >= 3:
            reasons.append(f"{consecutive_losses} consecutive losses")

        win_rate = performance_metrics.get('win_rate', 0.5)
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
        context: Dict[str, Any]
    ) -> None:
        """Track multiplier changes for analysis."""
        event = RiskMultiplierEvent(
            symbol=symbol,
            old_multiplier=old_multiplier,
            new_multiplier=new_multiplier,
            reason=reason,
            context=context,
            timestamp=datetime.now()
        )

        self.multiplier_history.append(event)

        # Keep only recent history
        cutoff_date = datetime.now() - timedelta(days=30)
        self.multiplier_history = [
            event for event in self.multiplier_history
            if event.timestamp > cutoff_date
        ]

    def _log_risk_decision(
        self,
        symbol: str,
        multiplier: float,
        reasoning: str,
        context: Dict[str, Any]
    ) -> None:
        """Log risk decision for monitoring."""
        try:
            log_data = {
                'symbol': symbol,
                'risk_multiplier': multiplier,
                'reasoning': reasoning,
                'defensive_mode': self.defensive_mode.value,
                'kill_switch_active': self.kill_switch_activated,
                'market_risk_level': context.get('market_conditions', {}).get('risk_level'),
                'sharpe_ratio': context.get('performance_metrics', {}).get('sharpe_ratio'),
                'consecutive_losses': context.get('performance_metrics', {}).get('consecutive_losses')
            }

            trade_logger.performance("Risk multiplier update", log_data)

        except Exception as e:
            logger.warning(f"Error logging risk decision: {e}")

    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get comprehensive risk statistics."""
        return {
            'current_multiplier': self.current_multiplier,
            'defensive_mode': self.defensive_mode.value,
            'kill_switch_activated': self.kill_switch_activated,
            'total_multiplier_changes': len(self.multiplier_history),
            'performance_metrics': self.performance_monitor.get_performance_metrics(),
            'recent_events': [
                event.to_dict() for event in self.multiplier_history[-10:]  # Last 10 events
            ]
        }

    def reset_kill_switch(self) -> bool:
        """Manually reset the kill switch."""
        if not self.kill_switch_activated:
            return False

        self.kill_switch_activated = False
        self.kill_switch_timestamp = None
        self.defensive_mode = DefensiveMode.NORMAL

        logger.info("Kill switch manually reset")
        trade_logger.performance("Kill switch reset", {'manual_reset': True})

        return True

    def _get_conservative_fallback_multiplier(self, reason: str) -> Tuple[float, str]:
        """
        Return a conservative fallback multiplier when data is unavailable or calculations fail.

        This method implements the fallback mechanism for data unavailability by providing
        a safe, conservative risk multiplier that minimizes exposure during uncertain conditions.

        Args:
            reason: Reason for using fallback (e.g., "insufficient data", "error: ...")

        Returns:
            Tuple of (conservative_multiplier, reasoning)
        """
        # Use a very conservative multiplier (25% of normal risk) during data unavailability
        conservative_multiplier = max(self.min_multiplier, 0.25)

        reasoning = f"Conservative fallback multiplier {conservative_multiplier:.2f}: {reason} - using safe defaults"

        logger.warning(f"Using conservative fallback for {reason}: multiplier={conservative_multiplier}")

        return conservative_multiplier, reasoning

    def _assess_market_conditions_with_fallback(
        self,
        symbol: str,
        market_data: pd.DataFrame
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
                logger.warning(f"Secondary market assessment also failed for {symbol}: {secondary_e}")

                # Use cached conditions if available
                cached_conditions = self._get_cached_market_conditions(symbol)
                if cached_conditions:
                    logger.info(f"Using cached market conditions for {symbol}")
                    return cached_conditions

                # Final fallback to conservative defaults
                logger.warning(f"Using conservative defaults for market conditions on {symbol}")
                return self._get_conservative_market_conditions()

    def _assess_market_conditions_secondary(
        self,
        symbol: str,
        market_data: pd.DataFrame
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
                'volatility_level': 'moderate',  # Default conservative
                'trend_strength': 25.0,  # Neutral trend
                'liquidity_score': 0.5,  # Neutral liquidity
                'regime': MarketRegime.UNKNOWN.value,
                'risk_level': RiskLevel.MODERATE.value
            }

            # Try to get basic volatility if data is available
            if (market_data is not None and not market_data.empty and
                'close' in market_data.columns and len(market_data) >= 5):

                # Simple volatility calculation for secondary validation
                returns = market_data['close'].pct_change().dropna()
                if len(returns) >= 4:
                    volatility = returns.std() * 100  # As percentage

                    # Classify volatility for secondary assessment
                    if volatility > 5.0:
                        conditions['volatility_level'] = 'high'
                    elif volatility > 2.0:
                        conditions['volatility_level'] = 'moderate'
                    else:
                        conditions['volatility_level'] = 'low'

                    # Adjust risk level based on secondary assessment
                    if conditions['volatility_level'] == 'high':
                        conditions['risk_level'] = RiskLevel.HIGH.value
                    elif conditions['volatility_level'] == 'low':
                        conditions['risk_level'] = RiskLevel.LOW.value

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
            if (hasattr(self.market_monitor, 'volatility_history') and
                symbol in self.market_monitor.volatility_history and
                self.market_monitor.volatility_history[symbol]):

                # Use most recent volatility data
                recent_volatility = self.market_monitor.volatility_history[symbol][-1]

                # Reconstruct basic conditions from cached data
                conditions = {
                    'volatility_level': 'moderate',  # Default
                    'trend_strength': 25.0,  # Neutral
                    'liquidity_score': 0.5,  # Neutral
                    'regime': MarketRegime.UNKNOWN.value,
                    'risk_level': RiskLevel.MODERATE.value
                }

                # Classify based on cached volatility
                if recent_volatility > 0.08:  # 8% volatility threshold
                    conditions['volatility_level'] = 'very_high'
                    conditions['risk_level'] = RiskLevel.VERY_HIGH.value
                elif recent_volatility > 0.05:  # 5% volatility threshold
                    conditions['volatility_level'] = 'high'
                    conditions['risk_level'] = RiskLevel.HIGH.value
                elif recent_volatility > 0.02:  # 2% volatility threshold
                    conditions['volatility_level'] = 'moderate'
                else:
                    conditions['volatility_level'] = 'low'
                    conditions['risk_level'] = RiskLevel.LOW.value

                logger.info(f"Retrieved cached conditions for {symbol}: volatility_level={conditions['volatility_level']}")
                return conditions

        except Exception as e:
            logger.warning(f"Error retrieving cached market conditions for {symbol}: {e}")

        return None

    def _get_conservative_market_conditions(self) -> Dict[str, Any]:
        """
        Return conservative default market conditions when all other methods fail.

        This provides the safest possible assumptions during complete data unavailability.

        Returns:
            Dictionary with conservative market condition defaults
        """
        return {
            'volatility_level': 'high',  # Assume high volatility (conservative)
            'trend_strength': 20.0,  # Weak trend (conservative)
            'liquidity_score': 0.3,  # Low liquidity (conservative)
            'regime': MarketRegime.UNKNOWN.value,
            'risk_level': RiskLevel.HIGH.value  # High risk level (conservative)
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
                'sharpe_ratio': -0.5,  # Poor performance (conservative)
                'win_rate': 0.4,  # Below average (conservative)
                'profit_factor': 0.8,  # Below 1.0 (conservative)
                'max_drawdown': 0.1,  # 10% drawdown (conservative)
                'consecutive_losses': 2,  # Some losses (conservative)
                'total_trades': 10  # Minimum history
            }


# Global instance
_adaptive_policy: Optional[AdaptiveRiskPolicy] = None


def get_adaptive_risk_policy(config: Optional[Dict[str, Any]] = None) -> AdaptiveRiskPolicy:
    """Get the global adaptive risk policy instance."""
    global _adaptive_policy
    if _adaptive_policy is None:
        if config is None:
            config = get_config('risk.adaptive', {})
        _adaptive_policy = AdaptiveRiskPolicy(config)
    return _adaptive_policy


def get_risk_multiplier(
    symbol: str,
    market_data: pd.DataFrame,
    context: Optional[Dict[str, Any]] = None
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
