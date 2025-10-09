"""
Strategy Selector (Meta-Strategy Layer)

This module provides dynamic strategy selection based on market conditions.
It can operate in rule-based or ML-based mode to choose the most suitable
trading strategy for current market conditions.

Supported Strategy Categories:
- Trend-following: EMA crossover, MACD trend detection
- Mean reversion: RSI oversold/overbought, Bollinger Bands reversion
- Breakout: Volatility-based breakouts

Selection Modes:
- Rule-based: Uses technical indicators (ADX, volatility) to determine regime
- ML-based: Uses reinforcement learning or classifier to adapt strategy weights
"""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from backtest.backtester import compute_regime_aware_metrics, export_regime_aware_report
from ml.indicators import calculate_adx, calculate_atr
from strategies.atr_breakout_strategy import ATRBreakoutStrategy
from strategies.base_strategy import BaseStrategy
from strategies.bollinger_reversion_strategy import BollingerReversionStrategy
from strategies.donchian_breakout_strategy import DonchianBreakoutStrategy
from strategies.ema_cross_strategy import EMACrossStrategy
from strategies.keltner_channel_strategy import KeltnerChannelStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.obv_strategy import OBVStrategy
from strategies.regime.market_regime import MarketRegime, get_recommended_strategies
from strategies.regime.regime_forecaster import ForecastingResult
from strategies.rsi_strategy import RSIStrategy
from strategies.stochastic_strategy import StochasticStrategy
from strategies.vwap_pullback_strategy import VWAPPullbackStrategy
from utils.config_loader import get_config

# Knowledge base imports
try:
    from knowledge_base import (
        MarketCondition,
        StrategyCategory,
        StrategyMetadata,
        get_knowledge_manager,
    )
    from knowledge_base import MarketRegime as KBMarketRegime

    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BASE_AVAILABLE = False

logger = logging.getLogger(__name__)

if not KNOWLEDGE_BASE_AVAILABLE:
    logger.warning("Knowledge base not available, running without adaptive learning")


class StrategyCategory(Enum):
    """Strategy category classifications."""

    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"


class StrategyPerformance:
    """Tracks performance metrics for a strategy."""

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.total_returns = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.profit_factor = 0.0
        self.last_updated = datetime.now()

        # Recent performance tracking
        self.recent_trades: List[Dict[str, Any]] = []
        self.max_recent_trades = 100

    def update_trade(self, pnl: float, returns: float, is_win: bool):
        """Update performance with a new trade."""
        self.total_trades += 1
        self.total_pnl += pnl
        self.total_returns += returns

        if is_win:
            self.winning_trades += 1
            self.avg_win = (
                (self.avg_win * (self.winning_trades - 1)) + pnl
            ) / self.winning_trades
        else:
            self.losing_trades += 1
            self.avg_loss = (
                (self.avg_loss * (self.losing_trades - 1)) + abs(pnl)
            ) / self.losing_trades

        # Update win rate
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades

        # Update profit factor
        if self.losing_trades > 0 and self.avg_loss > 0:
            self.profit_factor = (self.avg_win * self.winning_trades) / (
                self.avg_loss * self.losing_trades
            )

        # Store recent trade
        self.recent_trades.append(
            {
                "pnl": pnl,
                "returns": returns,
                "is_win": is_win,
                "timestamp": datetime.now(),
            }
        )

        # Keep only recent trades
        if len(self.recent_trades) > self.max_recent_trades:
            self.recent_trades = self.recent_trades[-self.max_recent_trades :]

        self.last_updated = datetime.now()

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            "strategy_name": self.strategy_name,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "total_returns": self.total_returns,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "last_updated": self.last_updated.isoformat(),
        }

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02):
        """Calculate Sharpe ratio from recent trades."""
        if len(self.recent_trades) < 2:
            return 0.0

        returns = [trade["returns"] for trade in self.recent_trades]
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                self.sharpe_ratio = (avg_return - risk_free_rate) / std_return
        return self.sharpe_ratio


class MarketStateAnalyzer:
    """Analyzes market conditions to determine regime."""

    def __init__(self):
        self.adx_trend_threshold = 25
        self.adx_sideways_threshold = 20
        self.volatility_percentile = 70

    def analyze_market_state(self, data: pd.DataFrame) -> MarketRegime:
        """
        Analyze market data to determine current regime.

        Args:
            data: OHLCV DataFrame

        Returns:
            MarketRegime classification
        """
        if data.empty or len(data) < 20:
            return MarketRegime.SIDEWAYS

        try:
            # Calculate ADX for trend strength
            adx = calculate_adx(data, period=14)
            current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0

            # Calculate ATR for volatility
            atr = calculate_atr(data, period=14)
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0

            # Calculate recent volatility percentile
            returns = data["close"].pct_change().dropna()
            if len(returns) > 20:
                recent_volatility = returns.tail(20).std()
                volatility_threshold = np.percentile(
                    returns.abs(), self.volatility_percentile
                )
                is_high_volatility = recent_volatility > volatility_threshold
            else:
                is_high_volatility = False

            # Determine trend direction by checking if closes are consistently increasing/decreasing
            if len(data) >= 20:
                recent_closes = data["close"].tail(20)
                # Check if price is trending up (more closes above their moving average)
                sma_10 = recent_closes.rolling(10).mean()
                closes_above_ma = (recent_closes > sma_10).sum()
                closes_below_ma = (recent_closes < sma_10).sum()

                # Trend strength based on consistency
                trend_ratio = closes_above_ma / len(recent_closes)
                trending_up = trend_ratio > 0.7  # 70% of closes above MA
                trending_down = (
                    closes_below_ma / len(recent_closes)
                ) > 0.7  # 70% below MA

                # Additional check: overall price movement
                price_change = (
                    recent_closes.iloc[-1] - recent_closes.iloc[0]
                ) / recent_closes.iloc[0]
                if price_change > 0.05:  # 5% up
                    trending_up = True
                elif price_change < -0.05:  # 5% down
                    trending_down = True
            else:
                trending_up = trending_down = False

            # Classify regime
            if current_adx > self.adx_trend_threshold:
                if trending_up:
                    return MarketRegime.TRENDING
                elif trending_down:
                    return MarketRegime.TRENDING
                else:
                    return MarketRegime.SIDEWAYS
            elif current_adx < self.adx_sideways_threshold:
                # Check for very flat price action (true sideways)
                if len(data) >= 20:
                    recent_range = (
                        data["high"].tail(20).max() - data["low"].tail(20).min()
                    ) / data["close"].tail(20).mean()
                    is_very_flat = recent_range < 0.02  # Less than 2% range

                    if is_very_flat and current_atr < (
                        data["close"].tail(20).mean() * 0.005
                    ):  # ATR < 0.5%
                        return MarketRegime.LOW_VOLATILITY
                    elif is_high_volatility:
                        return MarketRegime.HIGH_VOLATILITY
                    else:
                        return MarketRegime.SIDEWAYS
                else:
                    return MarketRegime.SIDEWAYS
            else:
                return MarketRegime.SIDEWAYS

        except Exception as e:
            logger.warning(f"Error analyzing market state: {e}")
            return MarketRegime.SIDEWAYS


class RuleBasedSelector:
    """
    Rule-based strategy selection using technical indicators.

    Rule Hierarchy:
    1. Trending Market: Prefer EMACrossStrategy for trend-following
    2. Ranging Market: Prefer RSIStrategy for mean-reversion
    3. Fallback: First available strategy

    Trending Detection:
    - Price slope > 5% over recent period indicates trending
    - Uses normalized slope: (end_price - start_price) / period / start_price
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Keep old attributes for backward compatibility
        self.adx_trend_threshold = self.config.get("adx_trend_threshold", 25)
        self.adx_sideways_threshold = self.config.get("adx_sideways_threshold", 20)
        # New attributes for simple trending detection
        self.trend_slope_threshold = self.config.get("trend_slope_threshold", 0.05)
        self.lookback_period = self.config.get("lookback_period", 20)

    def select_strategy(
        self, market_data: pd.DataFrame, available_strategies: List[type]
    ) -> Optional[type]:
        """
        Select the best strategy based on current market conditions.

        Uses simple trending detection:
        - If normalized slope > threshold, select EMACrossStrategy (trending)
        - Otherwise, select RSIStrategy (ranging)

        Args:
            market_data: Current market data
            available_strategies: List of available strategy classes

        Returns:
            Selected strategy class or None
        """
        if market_data.empty or len(market_data) < self.lookback_period:
            return available_strategies[0] if available_strategies else None

        # Simple trending detection - use full data for slope calculation
        prices = market_data["close"]
        if len(prices) < 2:
            return available_strategies[0] if available_strategies else None

        slope = (prices.iloc[-1] - prices.iloc[0]) / len(prices)
        slope_normalized = slope / prices.iloc[0] if prices.iloc[0] != 0 else 0

        if slope_normalized > self.trend_slope_threshold:
            # Trending market - prefer EMACrossStrategy
            for strat in available_strategies:
                if strat.__name__ == "EMACrossStrategy":
                    logger.info(
                        f"Selected EMACrossStrategy for trending market (slope: {slope_normalized:.3f})"
                    )
                    return strat
            # If EMACrossStrategy not found, fallback
            logger.warning(
                "EMACrossStrategy not found in available strategies, using fallback"
            )
        else:
            # Ranging market - prefer RSIStrategy
            for strat in available_strategies:
                if strat.__name__ == "RSIStrategy":
                    logger.info(
                        f"Selected RSIStrategy for ranging market (slope: {slope_normalized:.3f})"
                    )
                    return strat
            # If RSIStrategy not found, fallback
            logger.warning(
                "RSIStrategy not found in available strategies, using fallback"
            )

        # Fallback to first available strategy
        if available_strategies:
            fallback = available_strategies[0]
            logger.warning(
                f"Preferred strategy not found, using fallback {fallback.__name__}"
            )
            return fallback

        return None


class MLBasedSelector:
    """ML-based strategy selection using performance history."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.strategy_weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.min_trades_for_learning = self.config.get("min_trades_for_learning", 10)

    def select_strategy(
        self,
        market_data: pd.DataFrame,
        available_strategies: List[type],
        strategy_performances: Dict[str, StrategyPerformance],
    ) -> Optional[type]:
        """
        Select strategy using ML-based approach with performance weighting.

        Args:
            market_data: Current market data
            available_strategies: List of available strategy classes
            strategy_performances: Performance data for each strategy

        Returns:
            Selected strategy class
        """
        if not available_strategies:
            return None

        # Initialize weights if needed
        for strategy_class in available_strategies:
            strategy_name = strategy_class.__name__
            if strategy_name not in self.strategy_weights:
                self.strategy_weights[strategy_name] = 1.0 / len(available_strategies)
                self.performance_history[strategy_name] = []

        # Update weights based on recent performance
        self._update_weights(strategy_performances)

        # Select strategy based on weights
        strategy_names = [s.__name__ for s in available_strategies]
        weights = [self.strategy_weights.get(name, 1.0) for name in strategy_names]

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Select strategy probabilistically
        selected_idx = np.random.choice(len(available_strategies), p=weights)
        selected_strategy = available_strategies[selected_idx]

        logger.info(
            f"ML-based selection: {selected_strategy.__name__} (weight: {weights[selected_idx]:.3f})"
        )
        return selected_strategy

    def _update_weights(self, strategy_performances: Dict[str, StrategyPerformance]):
        """Update strategy weights based on performance."""
        for strategy_name, performance in strategy_performances.items():
            if strategy_name not in self.strategy_weights:
                continue

            # Only update if we have enough trades
            if performance.total_trades >= self.min_trades_for_learning:
                # Use recent win rate as performance metric
                recent_win_rate = performance.win_rate

                # Update weight using exponential moving average
                current_weight = self.strategy_weights[strategy_name]
                new_weight = current_weight + self.learning_rate * (
                    recent_win_rate - 0.5
                )
                self.strategy_weights[strategy_name] = max(
                    0.1, min(2.0, new_weight)
                )  # Clamp weights


class StrategySelector:
    """
    Meta-strategy manager that dynamically selects trading strategies
    based on market conditions and performance history.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        available_strategies: Optional[List[type]] = None,
    ):
        """
        Initialize the strategy selector.

        Args:
            config: Configuration dictionary
            available_strategies: Optional list of available strategies (overrides loading from file system)
        """
        self.config = config or self._get_default_config()
        cfg = get_config("strategy_selector", {})
        self.enabled = cfg.get("enabled", self.config.get("enabled", False))
        self.mode = cfg.get("mode", self.config.get("mode", "rule_based"))
        self.ensemble = cfg.get("ensemble", self.config.get("ensemble", False))

        # Initialize selectors
        self.rule_selector = RuleBasedSelector(self.config.get("rules", {}))
        self.ml_selector = MLBasedSelector(self.config.get("ml_config", {}))

        # Strategy registry - use provided strategies or load from file system
        self.available_strategies = (
            available_strategies or self._load_available_strategies()
        )
        self.strategy_performances: Dict[str, StrategyPerformance] = {}

        # Initialize performance tracking
        for strategy_class in self.available_strategies:
            strategy_name = strategy_class.__name__
            self.strategy_performances[strategy_name] = StrategyPerformance(
                strategy_name
            )

        # Current active strategy
        self.current_strategy: Optional[type] = None
        self.current_strategy_instance: Optional[BaseStrategy] = None

        logger.info(
            f"StrategySelector initialized: mode={self.mode}, ensemble={self.ensemble}"
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "enabled": False,
            "mode": "rule_based",
            "ensemble": False,
            "rules": {"adx_trend_threshold": 25, "adx_sideways_threshold": 20},
            "ml_config": {"learning_rate": 0.1, "min_trades_for_learning": 10},
        }

    def _load_available_strategies(self) -> List[type]:
        """Load all available strategy classes."""
        strategies = []

        # Trend-following strategies
        strategy_classes = [
            (EMACrossStrategy, "EMACrossStrategy"),
            (MACDStrategy, "MACDStrategy"),
            (DonchianBreakoutStrategy, "DonchianBreakoutStrategy"),
            # Mean reversion strategies
            (RSIStrategy, "RSIStrategy"),
            (BollingerReversionStrategy, "BollingerReversionStrategy"),
            (StochasticStrategy, "StochasticStrategy"),
            # Volatility-based strategies
            (ATRBreakoutStrategy, "ATRBreakoutStrategy"),
            (KeltnerChannelStrategy, "KeltnerChannelStrategy"),
            # Volume-based strategies
            (OBVStrategy, "OBVStrategy"),
            (VWAPPullbackStrategy, "VWAPPullbackStrategy"),
        ]

        for strategy_class, strategy_name in strategy_classes:
            try:
                from unittest.mock import MagicMock

                # Check if strategy is a mock (patched by tests)
                if isinstance(strategy_class, MagicMock) or (
                    hasattr(strategy_class, "_mock_name") and strategy_class._mock_name
                ):
                    strategies.append(strategy_class)
                    logger.debug(f"Loaded mock strategy: {strategy_name}")
                else:
                    strategies.append(strategy_class)
                    logger.debug(f"Loaded strategy: {strategy_name}")
            except (ImportError, AttributeError) as e:
                logger.warning(f"{strategy_name} not available: {e}")

        logger.info(
            f"Loaded {len(strategies)} strategies: {[s.__name__ for s in strategies]}"
        )
        return strategies

    def select_strategy(
        self,
        market_data: pd.DataFrame,
        available_strategies: Optional[List[type]] = None,
    ) -> Optional[type]:
        """
        Select the most suitable strategy for current market conditions.

        Args:
            market_data: Current market data
            available_strategies: Optional list of available strategies (overrides self.available_strategies)

        Returns:
            Selected strategy class
        """
        strategies = available_strategies or self.available_strategies

        if not self.enabled or not strategies:
            return strategies[0] if strategies else None

        # Get initial selection from rule-based or ML-based selector
        if self.mode == "rule_based":
            selected = self.rule_selector.select_strategy(market_data, strategies)
        elif self.mode == "ml_based":
            selected = self.ml_selector.select_strategy(
                market_data, strategies, self.strategy_performances
            )
        else:
            logger.warning(f"Unknown selection mode: {self.mode}, using rule_based")
            selected = self.rule_selector.select_strategy(market_data, strategies)

        # Apply knowledge base adaptive weighting if available
        if KNOWLEDGE_BASE_AVAILABLE and selected:
            selected = self._apply_knowledge_base_weighting(
                market_data, strategies, selected
            )

        if selected:
            self.current_strategy = selected
            logger.info(f"Selected strategy: {selected.__name__}")

        return selected

    def select_strategies_ensemble(
        self, market_data: pd.DataFrame, max_strategies: int = 2
    ) -> List[type]:
        """
        Select multiple strategies for ensemble trading.

        Args:
            market_data: Current market data
            max_strategies: Maximum number of strategies to select

        Returns:
            List of selected strategy classes
        """
        if not self.enabled or not self.ensemble:
            return [self.available_strategies[0]] if self.available_strategies else []

        # For ensemble mode, select top strategies
        if self.mode == "ml_based":
            # Sort strategies by performance weights
            strategy_weights = []
            for strategy_class in self.available_strategies:
                strategy_name = strategy_class.__name__
                weight = self.ml_selector.strategy_weights.get(strategy_name, 1.0)
                strategy_weights.append((strategy_class, weight))

            strategy_weights.sort(key=lambda x: x[1], reverse=True)
            selected = [s[0] for s in strategy_weights[:max_strategies]]
        else:
            # For rule-based ensemble, select primary + secondary
            primary = self.rule_selector.select_strategy(
                market_data, self.available_strategies
            )
            remaining = [s for s in self.available_strategies if s != primary]
            secondary = (
                self.rule_selector.select_strategy(market_data, remaining)
                if remaining
                else None
            )

            selected = [s for s in [primary, secondary] if s is not None]

        logger.info(f"Ensemble selection: {[s.__name__ for s in selected]}")
        return selected[:max_strategies]  # Ensure we don't exceed max_strategies

    def _apply_knowledge_base_weighting(
        self,
        market_data: pd.DataFrame,
        available_strategies: List[type],
        initial_selection: type,
    ) -> type:
        """
        Apply knowledge base adaptive weighting to potentially override initial selection.

        Args:
            market_data: Current market data
            available_strategies: List of available strategy classes
            initial_selection: Initially selected strategy class

        Returns:
            Final selected strategy class (may be different from initial)
        """
        try:
            # Get knowledge manager
            knowledge_manager = get_knowledge_manager()

            # Convert market data to market condition
            market_condition = self._extract_market_condition(market_data)

            # Convert available strategies to metadata
            strategy_metadata_list = []
            for strategy_class in available_strategies:
                metadata = self._create_strategy_metadata(strategy_class)
                strategy_metadata_list.append(metadata)

            # Get adaptive weights from knowledge base
            adaptive_weights = knowledge_manager.get_adaptive_weights(
                market_condition, strategy_metadata_list
            )

            # Log knowledge base influence
            logger.info(
                "KNOWLEDGE_QUERY: Retrieved adaptive weights from knowledge base"
            )
            for strategy_name, weight in adaptive_weights.items():
                logger.info(
                    f"KNOWLEDGE_APPLY: Strategy {strategy_name} weight: {weight:.3f}"
                )

            # Check if any strategy has significantly better weight
            initial_strategy_name = initial_selection.__name__
            initial_weight = adaptive_weights.get(initial_strategy_name, 1.0)

            # Find strategy with highest weight
            best_strategy_name = max(adaptive_weights.items(), key=lambda x: x[1])[0]
            best_weight = adaptive_weights[best_strategy_name]

            # Override selection if best strategy has much higher weight
            weight_threshold = 1.3  # 30% better performance
            if best_weight > initial_weight * weight_threshold:
                # Find the strategy class that matches the best strategy name
                for strategy_class in available_strategies:
                    if strategy_class.__name__ == best_strategy_name:
                        logger.info(
                            f"KNOWLEDGE_OVERRIDE: Overriding {initial_strategy_name} "
                            f"with {best_strategy_name} (weight: {best_weight:.3f} vs {initial_weight:.3f})"
                        )
                        return strategy_class

            logger.info(
                f"KNOWLEDGE_MAINTAIN: Keeping initial selection {initial_strategy_name} "
                f"(weight: {initial_weight:.3f})"
            )
            return initial_selection

        except Exception as e:
            logger.warning(
                f"KNOWLEDGE_ERROR: Failed to apply knowledge base weighting: {e}"
            )
            return initial_selection

    def _extract_market_condition(self, market_data: pd.DataFrame) -> Any:
        """Extract market condition from market data."""
        if not KNOWLEDGE_BASE_AVAILABLE:
            # Return a simple dict when knowledge base is not available
            return {
                "regime": "unknown",
                "volatility": 0.0,
                "trend_strength": 0.0,
                "timestamp": datetime.now(),
            }

        if market_data.empty:
            return MarketCondition(
                regime=KBMarketRegime.UNKNOWN, volatility=0.0, trend_strength=0.0
            )

        try:
            # Calculate basic market metrics
            returns = market_data["close"].pct_change().dropna()
            volatility = returns.std() if len(returns) > 0 else 0.0

            # Calculate trend strength using ADX-like calculation
            if len(market_data) >= 14:
                adx = calculate_adx(market_data, period=14)
                trend_strength = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0
            else:
                trend_strength = 0.0

            # Determine regime based on trend strength and volatility
            if trend_strength > 25:
                regime = KBMarketRegime.TRENDING
            elif volatility > returns.quantile(0.7) if len(returns) > 0 else False:
                regime = KBMarketRegime.HIGH_VOLATILITY
            else:
                regime = KBMarketRegime.SIDEWAYS

            return MarketCondition(
                regime=regime,
                volatility=float(volatility),
                trend_strength=float(trend_strength),
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.warning(f"Failed to extract market condition: {e}")
            return MarketCondition(
                regime=KBMarketRegime.UNKNOWN, volatility=0.0, trend_strength=0.0
            )

    def _create_strategy_metadata(self, strategy_class: type) -> Any:
        """Create strategy metadata from strategy class."""
        if not KNOWLEDGE_BASE_AVAILABLE:
            # Return a simple dict when knowledge base is not available
            return {
                "name": strategy_class.__name__,
                "category": "unknown",
                "parameters": {},
                "timeframe": "1h",
                "indicators_used": ["price"],
                "risk_profile": "medium",
            }

        strategy_name = strategy_class.__name__

        # Map strategy names to categories (simplified mapping)
        category_mapping = {
            "EMACrossStrategy": StrategyCategory.TREND_FOLLOWING,
            "MACDStrategy": StrategyCategory.TREND_FOLLOWING,
            "DonchianBreakoutStrategy": StrategyCategory.BREAKOUT,
            "RSIStrategy": StrategyCategory.MEAN_REVERSION,
            "BollingerReversionStrategy": StrategyCategory.MEAN_REVERSION,
            "StochasticStrategy": StrategyCategory.MEAN_REVERSION,
            "ATRBreakoutStrategy": StrategyCategory.BREAKOUT,
            "KeltnerChannelStrategy": StrategyCategory.TREND_FOLLOWING,
            "OBVStrategy": StrategyCategory.VOLUME_BASED,
            "VWAPPullbackStrategy": StrategyCategory.MEAN_REVERSION,
        }

        category = category_mapping.get(strategy_name, StrategyCategory.TREND_FOLLOWING)

        # Get indicators used (simplified)
        indicators_mapping = {
            "EMACrossStrategy": ["ema"],
            "MACDStrategy": ["macd"],
            "DonchianBreakoutStrategy": ["donchian"],
            "RSIStrategy": ["rsi"],
            "BollingerReversionStrategy": ["bollinger_bands"],
            "StochasticStrategy": ["stochastic"],
            "ATRBreakoutStrategy": ["atr"],
            "KeltnerChannelStrategy": ["keltner"],
            "OBVStrategy": ["obv"],
            "VWAPPullbackStrategy": ["vwap"],
        }

        indicators = indicators_mapping.get(strategy_name, ["price"])

        return StrategyMetadata(
            name=strategy_name,
            category=category,
            parameters={},  # Could be populated from strategy config
            timeframe="1h",
            indicators_used=indicators,
            risk_profile="medium",
        )

    def update_performance(
        self, strategy_name: str, pnl: float, returns: float, is_win: bool
    ):
        """
        Update performance metrics for a strategy.

        Args:
            strategy_name: Name of the strategy
            pnl: Profit/loss amount
            returns: Return percentage
            is_win: Whether the trade was profitable
        """
        if strategy_name not in self.strategy_performances:
            self.strategy_performances[strategy_name] = StrategyPerformance(
                strategy_name
            )

        perf = self.strategy_performances[strategy_name]
        perf.update_trade(pnl, returns, is_win)
        logger.debug(
            f"Updated performance for {strategy_name}: win_rate={perf.win_rate:.3f}"
        )

    def get_strategy_performance(
        self, strategy_name: Optional[str] = None
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Get performance metrics for strategies.

        Args:
            strategy_name: Specific strategy name, or None for all

        Returns:
            Performance metrics
        """
        if strategy_name:
            if strategy_name in self.strategy_performances:
                return self.strategy_performances[strategy_name].get_metrics()
            else:
                return {}
        else:
            return {
                name: perf.get_metrics()
                for name, perf in self.strategy_performances.items()
            }

    def save_performance_history(self, path: str):
        """Save performance history to disk."""
        performance_data = {
            "timestamp": datetime.now().isoformat(),
            "strategy_performances": self.get_strategy_performance(),
            "ml_weights": self.ml_selector.strategy_weights
            if hasattr(self.ml_selector, "strategy_weights")
            else {},
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(performance_data, f, indent=2, default=str)

        logger.info(f"Performance history saved to {path}")

    def load_performance_history(self, path: str):
        """Load performance history from disk."""
        if not Path(path).exists():
            logger.warning(f"Performance history file not found: {path}")
            return

        with open(path, "r") as f:
            data = json.load(f)

        # Restore performance data
        if "strategy_performances" in data:
            for strategy_name, perf_data in data["strategy_performances"].items():
                if strategy_name in self.strategy_performances:
                    perf = self.strategy_performances[strategy_name]
                    perf.total_trades = perf_data.get("total_trades", 0)
                    perf.winning_trades = perf_data.get("winning_trades", 0)
                    perf.losing_trades = perf_data.get("losing_trades", 0)
                    perf.total_pnl = perf_data.get("total_pnl", 0.0)
                    perf.win_rate = perf_data.get("win_rate", 0.0)

        # Restore ML weights
        if "ml_weights" in data and hasattr(self.ml_selector, "strategy_weights"):
            self.ml_selector.strategy_weights.update(data["ml_weights"])

        logger.info(f"Performance history loaded from {path}")

    def get_current_strategy(self) -> Optional[type]:
        """Get the currently selected strategy."""
        return self.current_strategy

    def reset(self):
        """Reset the selector state."""
        self.current_strategy = None
        self.current_strategy_instance = None
        logger.info("Strategy selector reset")

    def validate_multi_timeframe(
        self, signal: Any, multi_tf_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate a trading signal across multiple timeframes.

        Args:
            signal: Trading signal to validate
            multi_tf_data: Multi-timeframe data from TimeframeManager

        Returns:
            Validation result with confidence score and details
        """
        try:
            validation_result = {
                "is_valid": False,
                "confidence_score": 0.0,
                "signal_origin_timeframe": None,
                "confirming_timeframes": [],
                "conflicting_timeframes": [],
                "validation_details": {},
                "recommendation": "REJECT",
            }

            if not signal or not multi_tf_data:
                validation_result["validation_details"][
                    "reason"
                ] = "Missing signal or multi-timeframe data"
                return validation_result

            # Extract signal information
            signal_type = getattr(signal, "signal_type", None)
            if signal_type:
                signal_type = (
                    signal_type.value
                    if hasattr(signal_type, "value")
                    else str(signal_type)
                )
            else:
                signal_type = getattr(signal, "type", "UNKNOWN")

            symbol = getattr(signal, "symbol", "UNKNOWN")
            signal_timeframe = getattr(signal, "metadata", {}).get("timeframe", "15m")

            validation_result["signal_origin_timeframe"] = signal_timeframe

            # Define timeframe hierarchy for validation
            timeframe_hierarchy = {
                "15m": ["1h", "4h"],
                "1h": ["4h", "1d"],
                "4h": ["1d"],
                "1d": [],
            }

            # Get timeframes to validate against
            timeframes_to_check = timeframe_hierarchy.get(signal_timeframe, [])

            if not timeframes_to_check:
                validation_result["validation_details"][
                    "reason"
                ] = f"No higher timeframes configured for {signal_timeframe}"
                validation_result[
                    "is_valid"
                ] = True  # Accept if no higher timeframes to check
                validation_result["recommendation"] = "ACCEPT"
                return validation_result

            # Perform multi-timeframe validation
            confirming_count = 0
            total_checked = 0

            for tf in timeframes_to_check:
                if tf not in multi_tf_data.get("data", {}):
                    validation_result["conflicting_timeframes"].append(
                        f"{tf} (missing data)"
                    )
                    continue

                tf_data = multi_tf_data["data"][tf]
                if tf_data.empty:
                    validation_result["conflicting_timeframes"].append(
                        f"{tf} (empty data)"
                    )
                    continue

                total_checked += 1

                # Analyze trend on higher timeframe
                trend = self._analyze_timeframe_trend(tf_data, tf)

                # Check if trend aligns with signal
                if self._does_trend_confirm_signal(trend, signal_type):
                    confirming_count += 1
                    validation_result["confirming_timeframes"].append(tf)
                    validation_result["validation_details"][tf] = f"CONFIRMS ({trend})"
                else:
                    validation_result["conflicting_timeframes"].append(tf)
                    validation_result["validation_details"][tf] = f"CONFLICTS ({trend})"

            # Calculate confidence score
            if total_checked > 0:
                validation_result["confidence_score"] = confirming_count / total_checked

                # Determine if signal is valid based on confidence threshold
                min_confidence = 0.6  # 60% of higher timeframes must confirm
                validation_result["is_valid"] = (
                    validation_result["confidence_score"] >= min_confidence
                )

                if validation_result["is_valid"]:
                    validation_result["recommendation"] = "ACCEPT"
                else:
                    validation_result["recommendation"] = "REJECT"

            # Log validation outcome
            logger.info(
                f"MULTI-TF VALIDATION: {symbol} {signal_type} on {signal_timeframe} - "
                f"Confidence: {validation_result['confidence_score']:.2f}, "
                f"Confirming: {validation_result['confirming_timeframes']}, "
                f"Conflicting: {validation_result['conflicting_timeframes']}, "
                f"Recommendation: {validation_result['recommendation']}"
            )

            return validation_result

        except Exception as e:
            logger.error(f"Failed to validate signal across timeframes: {e}")
            validation_result["validation_details"]["error"] = str(e)
            return validation_result

    def _analyze_timeframe_trend(self, tf_data: pd.DataFrame, timeframe: str) -> str:
        """
        Analyze trend direction for a specific timeframe.

        Args:
            tf_data: Timeframe data
            timeframe: Timeframe string

        Returns:
            Trend direction: 'bullish', 'bearish', or 'sideways'
        """
        try:
            if tf_data.empty or len(tf_data) < 20:
                return "unknown"

            # Use simple moving average crossover for trend determination
            sma_short_period = 10
            sma_long_period = 20

            if len(tf_data) < sma_long_period:
                return "unknown"

            sma_short = tf_data["close"].rolling(sma_short_period).mean()
            sma_long = tf_data["close"].rolling(sma_long_period).mean()

            if len(sma_short) < 2 or len(sma_long) < 2:
                return "unknown"

            # Check recent trend
            recent_short = sma_short.iloc[-1]
            recent_long = sma_long.iloc[-1]
            prev_short = sma_short.iloc[-2]
            prev_long = sma_long.iloc[-2]

            if recent_short > recent_long and prev_short > prev_long:
                return "bullish"
            elif recent_short < recent_long and prev_short < prev_long:
                return "bearish"
            else:
                return "sideways"

        except Exception as e:
            logger.warning(f"Failed to analyze trend for {timeframe}: {e}")
            return "unknown"

    def _does_trend_confirm_signal(self, trend: str, signal_type: str) -> bool:
        """
        Check if a trend confirms a signal type.

        Args:
            trend: Trend direction ('bullish', 'bearish', 'sideways')
            signal_type: Signal type ('BUY', 'SELL', etc.)

        Returns:
            True if trend confirms signal, False otherwise
        """
        if trend == "unknown":
            return False

        buy_signals = ["BUY", "LONG"]
        sell_signals = ["SELL", "SHORT"]

        if signal_type in buy_signals:
            return trend == "bullish"
        elif signal_type in sell_signals:
            return trend == "bearish"
        else:
            return False

    def incorporate_regime_forecast(
        self, strategy_weights: Dict[str, float], forecast_result: ForecastingResult
    ) -> Dict[str, float]:
        """
        Incorporate regime forecast into strategy weighting.

        Args:
            strategy_weights: Current strategy weights
            forecast_result: Forecasting result from RegimeForecaster

        Returns:
            Updated strategy weights incorporating forecast
        """
        try:
            if not forecast_result or not forecast_result.predictions:
                logger.warning("No forecast data available for strategy weighting")
                return strategy_weights

            updated_weights = strategy_weights.copy()

            # Get the most confident forecast (shortest horizon with highest confidence)
            best_horizon = None
            best_confidence = 0.0
            best_predictions = None

            for horizon, confidence in forecast_result.confidence_scores.items():
                if (
                    confidence > best_confidence and confidence > 0.6
                ):  # Minimum confidence threshold
                    best_horizon = horizon
                    best_confidence = confidence
                    best_predictions = forecast_result.predictions.get(horizon, {})

            if not best_predictions:
                logger.info("No sufficiently confident forecast available")
                return strategy_weights

            # Map forecasted regime to recommended strategies
            forecasted_regime = max(best_predictions.items(), key=lambda x: x[1])[0]
            recommended_strategies = get_recommended_strategies_from_forecast(
                forecasted_regime
            )

            logger.info(
                f"Incorporating forecast: {forecasted_regime} (confidence: {best_confidence:.2f})"
            )
            logger.info(
                f"Recommended strategies for forecast: {recommended_strategies}"
            )

            # Adjust weights based on forecast
            forecast_boost = min(best_confidence * 0.5, 0.3)  # Max 30% boost

            for strategy_name in recommended_strategies:
                if strategy_name in updated_weights:
                    # Boost recommended strategies
                    updated_weights[strategy_name] *= 1 + forecast_boost
                else:
                    # Add new strategies with base weight
                    updated_weights[strategy_name] = 0.1

            # Normalize weights
            total_weight = sum(updated_weights.values())
            if total_weight > 0:
                updated_weights = {
                    k: v / total_weight for k, v in updated_weights.items()
                }

            logger.info(f"Updated strategy weights with forecast: {updated_weights}")
            return updated_weights

        except Exception as e:
            logger.error(f"Failed to incorporate regime forecast: {e}")
            return strategy_weights

    def select_strategy_with_forecast(
        self,
        market_data: pd.DataFrame,
        forecast_result: Optional[ForecastingResult] = None,
        available_strategies: Optional[List[type]] = None,
    ) -> Optional[type]:
        """
        Select strategy incorporating regime forecast information.

        Args:
            market_data: Current market data
            forecast_result: Optional forecasting result
            available_strategies: Optional list of available strategies

        Returns:
            Selected strategy class
        """
        strategies = available_strategies or self.available_strategies

        if not self.enabled or not strategies:
            return strategies[0] if strategies else None

        # Get base selection
        base_selection = self.select_strategy(market_data, strategies)

        # If no forecast available, return base selection
        if not forecast_result or not forecast_result.predictions:
            return base_selection

        try:
            # Get strategy weights
            strategy_weights = {s.__name__: 1.0 for s in strategies}

            # Incorporate forecast
            updated_weights = self.incorporate_regime_forecast(
                strategy_weights, forecast_result
            )

            # Select strategy based on updated weights
            strategy_names = list(updated_weights.keys())
            weights = list(updated_weights.values())

            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]

            # Select strategy probabilistically
            selected_idx = np.random.choice(len(strategy_names), p=weights)
            selected_name = strategy_names[selected_idx]

            # Find corresponding strategy class
            for strategy_class in strategies:
                if strategy_class.__name__ == selected_name:
                    if strategy_class != base_selection:
                        logger.info(
                            f"Forecast-influenced selection: {strategy_class.__name__} "
                            f"(base: {base_selection.__name__ if base_selection else 'None'})"
                        )
                    return strategy_class

            # Fallback to base selection
            return base_selection

        except Exception as e:
            logger.error(f"Error in forecast-influenced selection: {e}")
            return base_selection

    def get_forecast_impact_analysis(
        self, forecast_result: ForecastingResult
    ) -> Dict[str, Any]:
        """
        Analyze the potential impact of regime forecast on strategy selection.

        Args:
            forecast_result: Forecasting result to analyze

        Returns:
            Analysis of forecast impact
        """
        try:
            analysis = {
                "forecast_available": False,
                "recommended_regime": None,
                "recommended_strategies": [],
                "confidence_level": 0.0,
                "strategy_changes": [],
                "risk_implications": {},
            }

            if not forecast_result or not forecast_result.predictions:
                return analysis

            analysis["forecast_available"] = True

            # Find best forecast
            best_horizon = None
            best_confidence = 0.0
            best_predictions = None

            for horizon, confidence in forecast_result.confidence_scores.items():
                if confidence > best_confidence:
                    best_horizon = horizon
                    best_confidence = confidence
                    best_predictions = forecast_result.predictions.get(horizon, {})

            if not best_predictions:
                return analysis

            # Get recommended regime and strategies
            forecasted_regime_name = max(best_predictions.items(), key=lambda x: x[1])[
                0
            ]
            analysis["recommended_regime"] = forecasted_regime_name
            analysis["confidence_level"] = best_confidence

            # Map regime name to enum for strategy recommendations
            regime_mapping = {
                "trend_up": MarketRegime.TREND_UP,
                "trend_down": MarketRegime.TREND_DOWN,
                "range_tight": MarketRegime.RANGE_TIGHT,
                "range_wide": MarketRegime.RANGE_WIDE,
                "volatile_spike": MarketRegime.VOLATILE_SPIKE,
                "trending": MarketRegime.TRENDING,
                "sideways": MarketRegime.SIDEWAYS,
                "volatile": MarketRegime.VOLATILE,
            }

            forecasted_regime = regime_mapping.get(
                forecasted_regime_name, MarketRegime.UNKNOWN
            )
            analysis["recommended_strategies"] = get_recommended_strategies(
                forecasted_regime
            )

            # Analyze potential strategy changes
            current_regime = forecast_result.current_regime
            if current_regime and current_regime != forecasted_regime_name:
                analysis["strategy_changes"] = [
                    f"Potential shift from {current_regime} to {forecasted_regime_name}",
                    "Strategy adaptation may be beneficial",
                ]

            # Risk implications
            if best_confidence > 0.8:
                analysis["risk_implications"][
                    "confidence"
                ] = "High confidence - strong signal"
                analysis["risk_implications"][
                    "action"
                ] = "Consider adjusting strategy weights"
            elif best_confidence > 0.6:
                analysis["risk_implications"][
                    "confidence"
                ] = "Medium confidence - moderate signal"
                analysis["risk_implications"][
                    "action"
                ] = "Monitor and consider gradual adjustment"
            else:
                analysis["risk_implications"][
                    "confidence"
                ] = "Low confidence - weak signal"
                analysis["risk_implications"][
                    "action"
                ] = "Maintain current strategy selection"

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing forecast impact: {e}")
            return {"error": str(e)}

    def generate_regime_aware_backtest_report(
        self,
        equity_progression: List[Dict[str, Any]],
        regime_data: List[Dict[str, Any]],
        output_path: str = "results/regime_aware_backtest_report.json",
    ):
        """
        Generate a comprehensive regime-aware backtest report.

        Args:
            equity_progression: List of equity progression records
            regime_data: List of regime detection results
            output_path: Path to save the report

        Returns:
            Path to the generated report
        """
        try:
            # Compute regime-aware metrics
            regime_metrics = compute_regime_aware_metrics(
                equity_progression, regime_data
            )

            # Export comprehensive report
            report_path = export_regime_aware_report(regime_metrics, output_path)

            logger.info(f"Regime-aware backtest report generated: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"Failed to generate regime-aware backtest report: {e}")
            return None


def get_market_regime_detector():
    from .regime_detector import RegimeDetector

    return RegimeDetector()


# Global strategy selector instance
_strategy_selector: Optional[StrategySelector] = None


def get_strategy_selector() -> StrategySelector:
    """Get the global strategy selector instance."""
    global _strategy_selector
    if _strategy_selector is None:
        config = get_config("strategy_selector", {})
        _strategy_selector = StrategySelector(config)
    return _strategy_selector


def select_strategy(market_data: pd.DataFrame) -> Optional[type]:
    """
    Convenience function to select a strategy.

    Args:
        market_data: Current market data

    Returns:
        Selected strategy class
    """
    selector = get_strategy_selector()
    return selector.select_strategy(market_data)


def update_strategy_performance(
    strategy_name: str, pnl: float, returns: float, is_win: bool
):
    """
    Convenience function to update strategy performance.

    Args:
        strategy_name: Name of the strategy
        pnl: Profit/loss amount
        returns: Return percentage
        is_win: Whether the trade was profitable
    """
    selector = get_strategy_selector()
    selector.update_performance(strategy_name, pnl, returns, is_win)


def get_recommended_strategies_from_forecast(forecasted_regime: str) -> List[str]:
    """
    Get recommended strategies based on forecasted regime.

    Args:
        forecasted_regime: Forecasted regime name (string)

    Returns:
        List of recommended strategy names
    """
    # Map forecasted regime strings to MarketRegime enums
    regime_mapping = {
        "trend_up": MarketRegime.TREND_UP,
        "trend_down": MarketRegime.TREND_DOWN,
        "range_tight": MarketRegime.RANGE_TIGHT,
        "range_wide": MarketRegime.RANGE_WIDE,
        "volatile_spike": MarketRegime.VOLATILE_SPIKE,
        "trending": MarketRegime.TRENDING,
        "sideways": MarketRegime.SIDEWAYS,
        "volatile": MarketRegime.VOLATILE,
    }

    regime = regime_mapping.get(forecasted_regime, MarketRegime.UNKNOWN)
    return get_recommended_strategies(regime)
