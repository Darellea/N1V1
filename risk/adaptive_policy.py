"""
Adaptive Risk Policy Engine

This module implements dynamic risk management that automatically adjusts exposure
based on market conditions, performance metrics, and trading outcomes. It provides
real-time risk multipliers that scale position sizes to protect capital during
volatile periods and increase exposure during favorable conditions.
"""

from __future__ import annotations

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import numpy as np
import pandas as pd

from utils.config_loader import get_config
from utils.logger import get_trade_logger, TradeLogger
from core.contracts import TradingSignal
from strategies.regime.market_regime import get_market_regime_detector, MarketRegime

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
        self.volatility_threshold = config.get('volatility_threshold', 0.05)  # 5% ATR threshold
        self.volatility_lookback = config.get('volatility_lookback', 20)  # 20 periods
        self.adx_trend_threshold = config.get('adx_trend_threshold', 25)

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

        except Exception as e:
            logger.warning(f"Error assessing market conditions for {symbol}: {e}")
            return {
                'volatility_level': 'unknown',
                'trend_strength': 25,
                'liquidity_score': 0.5,
                'regime': MarketRegime.UNKNOWN.value,
                'risk_level': RiskLevel.MODERATE.value
            }

    def _calculate_volatility_level(self, symbol: str, data: pd.DataFrame) -> str:
        """Calculate volatility level based on ATR."""
        try:
            if len(data) < self.volatility_lookback:
                return 'unknown'

            # Calculate ATR
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift(1))
            low_close = np.abs(data['low'] - data['close'].shift(1))
            true_range = pd.Series(np.maximum.reduce([high_low.values, high_close.values, low_close.values]))
            atr = true_range.rolling(self.volatility_lookback).mean().iloc[-1]

            # Calculate ATR as percentage of price
            current_price = data['close'].iloc[-1]
            atr_percentage = atr / current_price

            # Store in history
            if symbol not in self.volatility_history:
                self.volatility_history[symbol] = []
            self.volatility_history[symbol].append(atr_percentage)
            if len(self.volatility_history[symbol]) > 100:
                self.volatility_history[symbol] = self.volatility_history[symbol][-100:]

            # Classify volatility
            if atr_percentage > self.volatility_threshold * 2:
                return 'very_high'
            elif atr_percentage > self.volatility_threshold:
                return 'high'
            elif atr_percentage > self.volatility_threshold * 0.5:
                return 'moderate'
            else:
                return 'low'

        except Exception as e:
            logger.warning(f"Error calculating volatility level: {e}")
            return 'unknown'

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX."""
        try:
            if len(data) < 28:  # Need enough data for ADX
                return 25.0

            # Simplified ADX calculation
            high = data['high']
            low = data['low']
            close = data['close']

            # Calculate DM+/DM-
            high_diff = high - high.shift(1)
            low_diff = low.shift(1) - low

            dm_plus = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            dm_minus = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

            # Calculate ATR
            tr = np.maximum.reduce([
                high - low,
                np.abs(high - close.shift(1)),
                np.abs(low - close.shift(1))
            ])

            # Smooth everything with 14-period EMA
            atr = pd.Series(tr).ewm(span=14).mean().iloc[-1]
            di_plus = pd.Series(dm_plus).ewm(span=14).mean().iloc[-1] / atr if atr > 0 else 0
            di_minus = pd.Series(dm_minus).ewm(span=14).mean().iloc[-1] / atr if atr > 0 else 0

            # Calculate ADX
            dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) > 0 else 0
            adx = pd.Series(dx).ewm(span=14).mean().iloc[-1]

            return float(adx)

        except Exception as e:
            logger.warning(f"Error calculating trend strength: {e}")
            return 25.0

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
        Calculate current performance metrics.

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

            # Calculate basic metrics
            pnls = [trade.get('pnl', 0) for trade in self.trade_history]
            wins = sum(1 for pnl in pnls if pnl > 0)
            losses = sum(1 for pnl in pnls if pnl < 0)

            win_rate = wins / len(pnls) if pnls else 0.0
            avg_win = sum(pnl for pnl in pnls if pnl > 0) / wins if wins > 0 else 0
            avg_loss = abs(sum(pnl for pnl in pnls if pnl < 0) / losses) if losses > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

            # Calculate Sharpe ratio
            if len(pnls) > 1:
                returns = np.array(pnls)
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0.0

            # Calculate max drawdown
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

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
        """
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

    def get_risk_multiplier(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, str]:
        """
        Calculate the current risk multiplier based on market conditions and performance.

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

            # Handle empty or invalid market data
            if market_data is None or market_data.empty:
                return 1.0, "Risk multiplier 1.00: neutral conditions (insufficient data)"

            # Assess market conditions
            market_conditions = self.market_monitor.assess_market_conditions(symbol, market_data)

            # Get performance metrics
            performance_metrics = self.performance_monitor.get_performance_metrics()

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
            return 1.0, f"Error: {str(e)}"

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
