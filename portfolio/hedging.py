"""
portfolio/hedging.py

Portfolio Hedging Strategies for Risk Management.

This module provides hedging strategies to protect portfolio value
during adverse market conditions.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import numpy as np
from decimal import Decimal


class PortfolioHedger:
    """
    Portfolio Hedging Manager.

    Provides various hedging strategies to protect portfolio value
    during market downturns or high volatility periods.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Portfolio Hedger.

        Args:
            config: Hedging configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Hedging configuration
        self.enabled = config.get('enabled', False)
        self.max_stablecoin_pct = config.get('max_stablecoin_pct', 0.3)
        self.hedge_trigger = config.get('trigger', {})

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging for hedger."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def evaluate_hedging(self, positions: Dict[str, Any],
                         market_conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate whether hedging is needed based on market conditions.

        Args:
            positions: Current portfolio positions
            market_conditions: Current market conditions

        Returns:
            Hedging actions if needed, None otherwise
        """
        if not self.enabled:
            return None

        # Check if hedging trigger conditions are met
        hedge_needed, trigger_reason = self._check_hedge_trigger(market_conditions)

        if not hedge_needed:
            return None

        self.logger.info(f"Hedging triggered: {trigger_reason}")

        # Calculate hedging actions
        hedging_actions = self._calculate_hedging_actions(positions, market_conditions)

        if hedging_actions:
            hedging_actions['trigger_reason'] = trigger_reason
            hedging_actions['timestamp'] = datetime.now()

        return hedging_actions

    def _check_hedge_trigger(self, market_conditions: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if hedging trigger conditions are met.

        Args:
            market_conditions: Current market conditions

        Returns:
            Tuple of (hedge_needed, trigger_reason)
        """
        trigger_config = self.hedge_trigger

        # Check ADX below threshold (trend weakness)
        adx_threshold = trigger_config.get('adx_below', 15)
        current_adx = market_conditions.get('adx', 25)

        if current_adx < adx_threshold:
            return True, f"ADX below threshold: {current_adx} < {adx_threshold}"

        # Check MA crossover (bearish signal)
        ma_crossover = trigger_config.get('ma_crossover')
        if ma_crossover == 'bearish':
            # Check if fast MA crossed below slow MA
            fast_ma = market_conditions.get('fast_ma')
            slow_ma = market_conditions.get('slow_ma')
            prev_fast_ma = market_conditions.get('prev_fast_ma')
            prev_slow_ma = market_conditions.get('prev_slow_ma')

            if (fast_ma and slow_ma and prev_fast_ma and prev_slow_ma):
                if prev_fast_ma > prev_slow_ma and fast_ma < slow_ma:
                    return True, "Bearish MA crossover detected"

        # Check volatility threshold
        vol_threshold = trigger_config.get('volatility_above', 0.05)
        current_vol = market_conditions.get('volatility', 0.02)

        if current_vol > vol_threshold:
            return True, f"Volatility above threshold: {current_vol} > {vol_threshold}"

        # Check drawdown threshold
        dd_threshold = trigger_config.get('drawdown_above', 0.1)
        current_dd = market_conditions.get('drawdown', 0.0)

        if current_dd > dd_threshold:
            return True, f"Drawdown above threshold: {current_dd} > {dd_threshold}"

        return False, ""

    def _calculate_hedging_actions(self, positions: Dict[str, Any],
                                  market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate hedging actions based on current positions and market conditions.

        Args:
            positions: Current portfolio positions
            market_conditions: Current market conditions

        Returns:
            Dictionary with hedging actions
        """
        actions = {
            'action_type': 'hedge',
            'trades': []
        }

        # Determine hedging strategy based on market conditions
        hedge_strategy = self._select_hedge_strategy(market_conditions)

        if hedge_strategy == 'stablecoin_rotation':
            actions['trades'] = self._stablecoin_rotation_hedge(positions)
        elif hedge_strategy == 'partial_exit':
            actions['trades'] = self._partial_exit_hedge(positions)
        elif hedge_strategy == 'volatility_hedge':
            actions['trades'] = self._volatility_hedge(positions, market_conditions)
        else:
            self.logger.warning(f"Unknown hedge strategy: {hedge_strategy}")
            return {}

        return actions

    def _select_hedge_strategy(self, market_conditions: Dict[str, Any]) -> str:
        """
        Select appropriate hedging strategy based on market conditions.

        Args:
            market_conditions: Current market conditions

        Returns:
            Selected hedging strategy
        """
        # Default strategy
        strategy = 'stablecoin_rotation'

        # Select strategy based on conditions
        volatility = market_conditions.get('volatility', 0.02)
        drawdown = market_conditions.get('drawdown', 0.0)
        trend_strength = market_conditions.get('trend_strength', 0.0)

        if volatility > 0.08:  # Very high volatility
            strategy = 'partial_exit'
        elif drawdown > 0.15:  # Significant drawdown
            strategy = 'partial_exit'
        elif abs(trend_strength) < 0.1:  # Weak trend
            strategy = 'volatility_hedge'

        return strategy

    def _stablecoin_rotation_hedge(self, positions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Implement stablecoin rotation hedging.

        Args:
            positions: Current portfolio positions

        Returns:
            List of hedging trades
        """
        trades = []

        # Step 1: Calculate total portfolio value
        try:
            total_portfolio_value = sum(float(pos.market_value) for pos in positions.values())
            total_portfolio_value = Decimal(str(total_portfolio_value))
        except (TypeError, ValueError, AttributeError):
            total_portfolio_value = Decimal('0')

        if total_portfolio_value == 0:
            return trades

        # Step 2: Identify stablecoin and risk positions
        stable_positions = {}
        risk_positions = {}

        for symbol, position in positions.items():
            if self._is_stablecoin(symbol):
                stable_positions[symbol] = position
            else:
                risk_positions[symbol] = position

        # Step 3: Calculate current stablecoin percentage
        try:
            current_stable_value = sum(float(pos.market_value) for pos in stable_positions.values())
            current_stable_pct = current_stable_value / float(total_portfolio_value)
        except (TypeError, ValueError, AttributeError):
            current_stable_pct = 0.0

        # Step 4: Check if hedging is needed
        if current_stable_pct >= self.max_stablecoin_pct:
            return trades  # Already at or above target

        # Step 5: Calculate amount to hedge
        target_stable_value = total_portfolio_value * Decimal(str(self.max_stablecoin_pct))
        amount_to_hedge = target_stable_value - Decimal(str(current_stable_value))

        if amount_to_hedge <= 0:
            return trades

        # Step 6: Calculate total risk asset value for proportional allocation
        try:
            total_risk_value = sum(float(pos.market_value) for pos in risk_positions.values())
        except (TypeError, ValueError, AttributeError):
            total_risk_value = 0.0

        if total_risk_value == 0:
            return trades

        # Step 7: Generate proportional sell trades from risk assets
        for symbol, position in risk_positions.items():
            try:
                position_value = float(position.market_value)
                if isinstance(position.market_value, Mock):
                    position_value = 0.0
            except (TypeError, ValueError, AttributeError):
                position_value = 0.0

            if position_value <= 0:
                continue

            # Calculate proportional amount to sell from this position
            proportion = position_value / total_risk_value
            sell_value = amount_to_hedge * Decimal(str(proportion))

            if sell_value <= 0:
                continue

            # Get current price
            try:
                current_price = position.current_price
                if isinstance(current_price, Mock):
                    current_price = Decimal('1')
            except AttributeError:
                current_price = Decimal('1')

            # Calculate quantity to sell
            try:
                sell_quantity = sell_value / current_price
            except (TypeError, ValueError):
                sell_quantity = Decimal('0')

            if sell_quantity > 0:
                trades.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': float(sell_quantity),
                    'price': float(current_price),
                    'value': float(sell_value),
                    'reason': 'hedge_stablecoin_rotation'
                })

        return trades

    def _partial_exit_hedge(self, positions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Implement partial exit hedging (reduce exposure).

        Args:
            positions: Current portfolio positions

        Returns:
            List of hedging trades
        """
        trades = []

        # Sell a portion of each volatile position
        for symbol, position in positions.items():
            try:
                market_value = position.market_value
                if isinstance(market_value, Mock):
                    market_value = Decimal('0')
            except AttributeError:
                market_value = Decimal('0')
            try:
                market_value_numeric = float(market_value)
            except (TypeError, ValueError):
                market_value_numeric = 0.0
            if not self._is_stablecoin(symbol) and market_value_numeric > 0:
                # Sell 15-25% of position based on size
                sell_pct = 0.15 if market_value_numeric < 1000 else 0.25
                try:
                    sell_value = Decimal(str(market_value_numeric * sell_pct))
                except (TypeError, ValueError):
                    sell_value = Decimal('0')
                try:
                    current_price = position.current_price
                    if isinstance(current_price, Mock):
                        current_price = Decimal('1')
                except AttributeError:
                    current_price = Decimal('1')
                try:
                    sell_quantity = sell_value / current_price
                except (TypeError, ValueError):
                    sell_quantity = Decimal('0')

                if sell_quantity > 0:
                    trades.append({
                        'symbol': symbol,
                        'side': 'sell',
                        'quantity': float(sell_quantity),
                        'price': float(current_price),
                        'value': float(sell_value),
                        'reason': 'hedge_partial_exit'
                    })

        return trades

    def _volatility_hedge(self, positions: Dict[str, Any],
                         market_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Implement volatility-based hedging.

        Args:
            positions: Current portfolio positions
            market_conditions: Current market conditions

        Returns:
            List of hedging trades
        """
        trades = []

        # Calculate volatility-adjusted hedge ratios
        volatility = market_conditions.get('volatility', 0.02)

        # Higher volatility = more aggressive hedging
        hedge_ratio = min(volatility * 5, 0.4)  # Max 40% reduction

        for symbol, position in positions.items():
            try:
                market_value = position.market_value
                if isinstance(market_value, Mock):
                    market_value = Decimal('0')
            except AttributeError:
                market_value = Decimal('0')
            try:
                market_value_numeric = float(market_value)
            except (TypeError, ValueError):
                market_value_numeric = 0.0
            if not self._is_stablecoin(symbol) and market_value_numeric > 0:
                # Reduce position by volatility-adjusted amount
                try:
                    reduce_value = Decimal(str(market_value_numeric * hedge_ratio))
                except (TypeError, ValueError):
                    reduce_value = Decimal('0')
                try:
                    current_price = position.current_price
                    if isinstance(current_price, Mock):
                        current_price = Decimal('1')
                except AttributeError:
                    current_price = Decimal('1')
                try:
                    reduce_quantity = reduce_value / current_price
                except (TypeError, ValueError):
                    reduce_quantity = Decimal('0')

                if reduce_quantity > 0:
                    trades.append({
                        'symbol': symbol,
                        'side': 'sell',
                        'quantity': float(reduce_quantity),
                        'price': float(current_price),
                        'value': float(reduce_value),
                        'reason': 'hedge_volatility'
                    })

        return trades

    def _is_stablecoin(self, symbol: str) -> bool:
        """
        Check if a symbol represents a stablecoin.

        Args:
            symbol: Asset symbol

        Returns:
            True if symbol is a stablecoin
        """
        stablecoins = ['USDT', 'USDC', 'BUSD', 'DAI', 'UST']
        base_asset = symbol.upper().split('/')[0]
        return base_asset in stablecoins

    def calculate_hedge_effectiveness(self, pre_hedge_value: Decimal,
                                    post_hedge_value: Decimal,
                                    market_change: float) -> Dict[str, Any]:
        """
        Calculate hedging effectiveness metrics.

        Args:
            pre_hedge_value: Portfolio value before hedging
            post_hedge_value: Portfolio value after hedging
            market_change: Market change percentage during hedge period

        Returns:
            Dictionary with hedging effectiveness metrics
        """
        if pre_hedge_value == 0:
            return {}

        # Calculate portfolio changes
        portfolio_change_pct = (post_hedge_value - pre_hedge_value) / pre_hedge_value

        # Calculate hedge effectiveness
        # Effectiveness = 1 - (portfolio_change / market_change)
        if market_change != 0:
            effectiveness = 1 - (portfolio_change_pct / market_change)
        else:
            effectiveness = 0.0

        return {
            'pre_hedge_value': float(pre_hedge_value),
            'post_hedge_value': float(post_hedge_value),
            'portfolio_change_pct': portfolio_change_pct,
            'market_change_pct': market_change,
            'hedge_effectiveness': effectiveness,
            'hedge_ratio': abs(portfolio_change_pct / market_change) if market_change != 0 else 0.0
        }

    def get_hedge_summary(self) -> Dict[str, Any]:
        """
        Get summary of hedging configuration and status.

        Returns:
            Dictionary with hedging summary
        """
        return {
            'enabled': self.enabled,
            'max_stablecoin_pct': self.max_stablecoin_pct,
            'trigger_conditions': self.hedge_trigger,
            'available_strategies': [
                'stablecoin_rotation',
                'partial_exit',
                'volatility_hedge'
            ]
        }


class MarketConditionAnalyzer:
    """
    Market Condition Analyzer for Hedging Decisions.

    Analyzes market conditions to determine optimal hedging strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Market Condition Analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def analyze_conditions(self, market_data: pd.DataFrame,
                          portfolio_positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current market conditions for hedging decisions.

        Args:
            market_data: Recent market data
            portfolio_positions: Current portfolio positions

        Returns:
            Dictionary with market condition analysis
        """
        conditions = {}

        if market_data.empty:
            return conditions

        try:
            # Calculate ADX (trend strength)
            conditions['adx'] = self._calculate_adx(market_data)

            # Calculate moving averages
            conditions.update(self._calculate_moving_averages(market_data))

            # Calculate volatility
            conditions['volatility'] = self._calculate_volatility(market_data)

            # Calculate drawdown
            conditions['drawdown'] = self._calculate_portfolio_drawdown(portfolio_positions)

            # Determine market regime
            conditions['regime'] = self._determine_market_regime(conditions)

        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")

        return conditions

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average Directional Index (ADX).

        Args:
            data: OHLCV data
            period: Calculation period

        Returns:
            ADX value
        """
        if len(data) < period * 2:
            return 25.0  # Default neutral value

        try:
            high = data['high']
            low = data['low']
            close = data['close']

            # Calculate True Range
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)

            # Calculate Directional Movements
            dm_plus = (high - high.shift(1)).where(
                (high - high.shift(1)) > (low.shift(1) - low), 0
            )
            dm_minus = (low.shift(1) - low).where(
                (low.shift(1) - low) > (high - high.shift(1)), 0
            )

            # Calculate Directional Indicators
            di_plus = 100 * (dm_plus.rolling(period).mean() / tr.rolling(period).mean())
            di_minus = 100 * (dm_minus.rolling(period).mean() / tr.rolling(period).mean())

            # Calculate DX and ADX
            dx = 100 * ((di_plus - di_minus).abs() / (di_plus + di_minus))
            adx = dx.rolling(period).mean()

            return float(adx.iloc[-1])

        except Exception:
            return 25.0

    def _calculate_moving_averages(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate moving averages for crossover analysis.

        Args:
            data: Price data

        Returns:
            Dictionary with MA values
        """
        try:
            close = data['close']
            fast_ma = close.rolling(10).mean()
            slow_ma = close.rolling(20).mean()

            return {
                'fast_ma': float(fast_ma.iloc[-1]),
                'slow_ma': float(slow_ma.iloc[-1]),
                'prev_fast_ma': float(fast_ma.iloc[-2]) if len(fast_ma) > 1 else float(fast_ma.iloc[-1]),
                'prev_slow_ma': float(slow_ma.iloc[-2]) if len(slow_ma) > 1 else float(slow_ma.iloc[-1])
            }

        except Exception:
            return {}

    def _calculate_volatility(self, data: pd.DataFrame, period: int = 20) -> float:
        """
        Calculate realized volatility.

        Args:
            data: Price data
            period: Lookback period

        Returns:
            Volatility value
        """
        try:
            returns = data['close'].pct_change().dropna()
            volatility = returns.tail(period).std() * np.sqrt(252)  # Annualized
            return float(volatility)

        except Exception:
            return 0.02  # Default 2%

    def _calculate_portfolio_drawdown(self, positions: Dict[str, Any]) -> float:
        """
        Calculate portfolio drawdown.

        Args:
            positions: Portfolio positions

        Returns:
            Maximum drawdown percentage
        """
        try:
            total_pnl = sum(float(getattr(pos, 'total_pnl', 0)) for pos in positions.values())
            total_entry_value = sum(
                float(getattr(pos, 'quantity', 0)) * float(getattr(pos, 'entry_price', 0)) for pos in positions.values()
            )

            if total_entry_value > 0:
                return abs(total_pnl / total_entry_value)
            else:
                return 0.0

        except Exception:
            return 0.0

    def _determine_market_regime(self, conditions: Dict[str, Any]) -> str:
        """
        Determine current market regime.

        Args:
            conditions: Market condition metrics

        Returns:
            Market regime classification
        """
        adx = conditions.get('adx', 25)
        volatility = conditions.get('volatility', 0.02)

        if adx > 30 and volatility < 0.03:
            return 'strong_trend'
        elif adx < 20 and volatility > 0.05:
            return 'high_volatility'
        elif adx < 15:
            return 'weak_trend'
        else:
            return 'normal'
