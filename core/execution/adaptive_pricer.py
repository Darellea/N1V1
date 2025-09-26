"""
Adaptive Pricer

Dynamically adjusts limit order prices based on market volatility and conditions
to improve execution probability and reduce slippage.
"""

import logging
from typing import Dict, Any, Optional
from decimal import Decimal, ROUND_DOWN

from core.contracts import TradingSignal
from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class AdaptivePricer:
    """
    Dynamically adjusts limit order prices based on market volatility
    and liquidity conditions to improve execution probability.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adaptive pricer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.enabled = self.config.get('enabled', True)
        self.atr_window = self.config.get('atr_window', 14)
        self.price_adjustment_multiplier = Decimal(str(self.config.get('price_adjustment_multiplier', 0.5)))
        self.max_price_adjustment_pct = Decimal(str(self.config.get('max_price_adjustment_pct', 0.05)))

        # Adjustment factors for different market conditions
        self.volatility_factors = self.config.get('volatility_factors', {
            'low': Decimal('0.2'),      # Small adjustment for low volatility
            'medium': Decimal('0.5'),   # Medium adjustment for normal volatility
            'high': Decimal('0.8'),     # Large adjustment for high volatility
            'extreme': Decimal('1.0')   # Maximum adjustment for extreme volatility
        })

        # Spread-based adjustments
        self.spread_factors = self.config.get('spread_factors', {
            'tight': Decimal('0.1'),    # Small spread
            'normal': Decimal('0.3'),   # Normal spread
            'wide': Decimal('0.6'),     # Wide spread
            'very_wide': Decimal('1.0') # Very wide spread
        })

        self.logger.info("AdaptivePricer initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'enabled': True,
            'atr_window': 14,
            'price_adjustment_multiplier': 0.5,
            'max_price_adjustment_pct': 0.05,
            'volatility_factors': {
                'low': 0.2,
                'medium': 0.5,
                'high': 0.8,
                'extreme': 1.0
            },
            'spread_factors': {
                'tight': 0.1,
                'normal': 0.3,
                'wide': 0.6,
                'very_wide': 1.0
            }
        }

    async def adjust_price(self, signal: TradingSignal, context: Dict[str, Any]) -> Optional[Decimal]:
        """
        Adjust the limit price for better execution probability.

        Args:
            signal: Trading signal with price to adjust
            context: Market context information

        Returns:
            Adjusted price or None if no adjustment needed/applicable
        """
        if not self.enabled:
            return None

        if not signal.price:
            # No price to adjust
            return None

        try:
            # Calculate base adjustment based on volatility
            volatility_adjustment = self._calculate_volatility_adjustment(signal, context)

            # Calculate spread-based adjustment
            spread_adjustment = self._calculate_spread_adjustment(signal, context)

            # Combine adjustments
            total_adjustment = volatility_adjustment + spread_adjustment

            # Apply limits
            max_adjustment = signal.price * self.max_price_adjustment_pct
            total_adjustment = max(-max_adjustment, min(max_adjustment, total_adjustment))

            # Calculate final price
            adjusted_price = signal.price + total_adjustment

            # Ensure price is valid (positive and reasonable)
            if adjusted_price <= 0:
                self.logger.warning(f"Adjusted price {adjusted_price} is invalid, keeping original")
                return signal.price

            # Round to appropriate precision
            adjusted_price = self._round_price(adjusted_price, signal.symbol)

            # Log adjustment
            adjustment_pct = (adjusted_price - signal.price) / signal.price * 100
            self.logger.debug(f"Price adjustment for {signal.symbol}: "
                            f"{signal.price} -> {adjusted_price} ({adjustment_pct:.3f}%)")

            trade_logger.performance("Price Adjustment", {
                'symbol': signal.symbol,
                'original_price': float(signal.price),
                'adjusted_price': float(adjusted_price),
                'adjustment_pct': float(adjustment_pct),
                'volatility_adjustment': float(volatility_adjustment),
                'spread_adjustment': float(spread_adjustment)
            })

            return adjusted_price

        except Exception as e:
            self.logger.error(f"Error adjusting price: {e}")
            return signal.price  # Return original price on error

    def _calculate_volatility_adjustment(self, signal: TradingSignal, context: Dict[str, Any]) -> Decimal:
        """
        Calculate price adjustment based on market volatility.

        Args:
            signal: Trading signal
            context: Market context

        Returns:
            Price adjustment amount
        """
        # Get ATR (Average True Range) as volatility measure
        atr_value = context.get('atr', context.get('volatility', 0))
        if atr_value <= 0:
            return Decimal(0)

        # Convert to Decimal if necessary
        atr = Decimal(str(atr_value)) if not isinstance(atr_value, Decimal) else atr_value

        # Get current price for percentage calculation
        current_price = signal.current_price or signal.price or Decimal(100)
        atr_pct = atr / current_price

        # Determine volatility level
        if atr_pct < Decimal('0.005'):  # < 0.5%
            factor = self.volatility_factors['low']
        elif atr_pct < Decimal('0.02'):  # < 2%
            factor = self.volatility_factors['medium']
        elif atr_pct < Decimal('0.05'):  # < 5%
            factor = self.volatility_factors['high']
        else:
            factor = self.volatility_factors['extreme']

        # Calculate adjustment - convert factor to Decimal
        decimal_factor = Decimal(str(factor))
        base_adjustment = atr * self.price_adjustment_multiplier * decimal_factor

        # Direction depends on signal type and order side
        if signal.signal_type and signal.signal_type.value == 'ENTRY_LONG':
            # For buy orders, adjust price down (more aggressive) in high volatility
            return -base_adjustment
        elif signal.signal_type and signal.signal_type.value == 'ENTRY_SHORT':
            # For sell orders, adjust price up (more aggressive) in high volatility
            return base_adjustment
        else:
            # Default: no adjustment
            return Decimal(0)

    def _calculate_spread_adjustment(self, signal: TradingSignal, context: Dict[str, Any]) -> Decimal:
        """
        Calculate price adjustment based on bid-ask spread.

        Args:
            signal: Trading signal
            context: Market context

        Returns:
            Price adjustment amount
        """
        spread_pct = Decimal(str(context.get('spread_pct', 0.002)))  # Default 0.2%, convert to Decimal

        # Determine spread level
        if spread_pct < Decimal('0.001'):  # < 0.1%
            factor = self.spread_factors['tight']
        elif spread_pct < Decimal('0.005'):  # < 0.5%
            factor = self.spread_factors['normal']
        elif spread_pct < Decimal('0.01'):  # < 1%
            factor = self.spread_factors['wide']
        else:
            factor = self.spread_factors['very_wide']

        # Calculate adjustment based on spread
        current_price = signal.current_price or signal.price or Decimal(100)
        spread_adjustment = current_price * spread_pct * factor

        # Direction depends on signal type
        if signal.signal_type and signal.signal_type.value == 'ENTRY_LONG':
            # For buy orders, adjust price down to cross the spread
            return -spread_adjustment
        elif signal.signal_type and signal.signal_type.value == 'ENTRY_SHORT':
            # For sell orders, adjust price up to cross the spread
            return spread_adjustment
        else:
            return Decimal(0)

    def _round_price(self, price: Decimal, symbol: str) -> Decimal:
        """
        Round price to appropriate precision for the symbol.

        Args:
            price: Price to round
            symbol: Trading symbol

        Returns:
            Rounded price
        """
        # Different symbols have different tick sizes
        # This is a simplified implementation
        if 'BTC' in symbol:
            # BTC typically has 2 decimal places for USD pairs
            return price.quantize(Decimal('0.01'), rounding=ROUND_DOWN)
        elif 'ETH' in symbol:
            # ETH typically has 2 decimal places for USD pairs
            return price.quantize(Decimal('0.01'), rounding=ROUND_DOWN)
        else:
            # Default: 4 decimal places
            return price.quantize(Decimal('0.0001'), rounding=ROUND_DOWN)

    def calculate_optimal_limit_price(self, signal: TradingSignal, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate optimal limit price with bounds for the order.

        Args:
            signal: Trading signal
            context: Market context

        Returns:
            Dictionary with optimal price and bounds
        """
        if not signal.price:
            return {
                'optimal_price': None,
                'upper_bound': None,
                'lower_bound': None,
                'confidence': 0.0
            }

        # Get adjusted price
        optimal_price = self.adjust_price(signal, context)
        if not optimal_price:
            optimal_price = signal.price

        # Calculate bounds based on volatility
        atr = context.get('atr', context.get('volatility', 0))
        if atr > 0:
            # Bounds are 1 ATR from optimal price
            upper_bound = optimal_price + atr
            lower_bound = optimal_price - atr
            confidence = 0.8  # High confidence with ATR data
        else:
            # Default bounds: +/- 1% of price
            bound_pct = Decimal('0.01')
            upper_bound = optimal_price * (1 + bound_pct)
            lower_bound = optimal_price * (1 - bound_pct)
            confidence = 0.6  # Medium confidence without ATR

        return {
            'optimal_price': optimal_price,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'confidence': confidence,
            'adjustment_reason': 'volatility_and_spread_based'
        }

    def get_pricing_recommendations(self, signal: TradingSignal, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive pricing recommendations.

        Args:
            signal: Trading signal
            context: Market context

        Returns:
            Dictionary with pricing recommendations
        """
        recommendations = {
            'recommended_price': None,
            'price_range': {},
            'market_conditions': {},
            'risk_assessment': {},
            'execution_probability': 0.0
        }

        if not signal.price:
            return recommendations

        # Get optimal pricing
        optimal_pricing = self.calculate_optimal_limit_price(signal, context)
        recommendations['recommended_price'] = optimal_pricing['optimal_price']
        recommendations['price_range'] = {
            'upper': optimal_pricing['upper_bound'],
            'lower': optimal_pricing['lower_bound']
        }

        # Assess market conditions
        recommendations['market_conditions'] = self._assess_market_conditions(context)

        # Risk assessment
        recommendations['risk_assessment'] = self._assess_pricing_risk(signal, context)

        # Estimate execution probability
        recommendations['execution_probability'] = self._estimate_execution_probability(
            signal, optimal_pricing, context
        )

        return recommendations

    def _assess_market_conditions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess current market conditions for pricing.

        Args:
            context: Market context

        Returns:
            Market condition assessment
        """
        conditions = {}

        # Volatility assessment
        atr = context.get('atr', 0)
        if atr > 0:
            current_price = context.get('market_price', 100)
            volatility_pct = atr / current_price

            if volatility_pct < 0.01:
                conditions['volatility'] = 'low'
            elif volatility_pct < 0.03:
                conditions['volatility'] = 'medium'
            elif volatility_pct < 0.06:
                conditions['volatility'] = 'high'
            else:
                conditions['volatility'] = 'extreme'
        else:
            conditions['volatility'] = 'unknown'

        # Liquidity assessment
        spread_pct = context.get('spread_pct', 0.002)
        if spread_pct < 0.001:
            conditions['liquidity'] = 'excellent'
        elif spread_pct < 0.005:
            conditions['liquidity'] = 'good'
        elif spread_pct < 0.01:
            conditions['liquidity'] = 'moderate'
        else:
            conditions['liquidity'] = 'poor'

        # Volume assessment
        volume_ratio = context.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            conditions['volume'] = 'high'
        elif volume_ratio > 0.8:
            conditions['volume'] = 'normal'
        else:
            conditions['volume'] = 'low'

        return conditions

    def _assess_pricing_risk(self, signal: TradingSignal, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess pricing-related risks.

        Args:
            signal: Trading signal
            context: Market context

        Returns:
            Risk assessment
        """
        risks = {
            'slippage_risk': 'low',
            'execution_risk': 'low',
            'opportunity_risk': 'low'
        }

        # Assess slippage risk
        spread_pct = context.get('spread_pct', 0.002)
        volatility_pct = context.get('volatility_pct', 0.02)

        combined_risk = spread_pct + volatility_pct
        if combined_risk > 0.05:
            risks['slippage_risk'] = 'high'
        elif combined_risk > 0.02:
            risks['slippage_risk'] = 'medium'

        # Assess execution risk (price too far from market)
        if signal.price and context.get('market_price'):
            price_diff_pct = abs(signal.price - context['market_price']) / context['market_price']
            if price_diff_pct > 0.05:
                risks['execution_risk'] = 'high'
            elif price_diff_pct > 0.02:
                risks['execution_risk'] = 'medium'

        # Assess opportunity risk (missing fast market moves)
        if context.get('trend_strength', 0) > 0.7:
            risks['opportunity_risk'] = 'high'
        elif context.get('trend_strength', 0) > 0.4:
            risks['opportunity_risk'] = 'medium'

        return risks

    def _estimate_execution_probability(self, signal: TradingSignal, pricing: Dict[str, Any],
                                      context: Dict[str, Any]) -> float:
        """
        Estimate the probability of order execution.

        Args:
            signal: Trading signal
            pricing: Pricing information
            context: Market context

        Returns:
            Execution probability (0.0 to 1.0)
        """
        base_probability = 0.7  # Base 70% probability

        # Adjust based on price distance from market
        if signal.price and context.get('market_price'):
            price_diff_pct = abs(signal.price - context['market_price']) / context['market_price']

            if price_diff_pct < 0.005:  # Within 0.5%
                base_probability += 0.2
            elif price_diff_pct < 0.02:  # Within 2%
                base_probability += 0.1
            elif price_diff_pct > 0.05:  # More than 5% away
                base_probability -= 0.3

        # Adjust based on spread
        spread_pct = context.get('spread_pct', 0.002)
        if spread_pct < 0.001:
            base_probability += 0.1
        elif spread_pct > 0.01:
            base_probability -= 0.2

        # Adjust based on volatility
        volatility_pct = context.get('volatility_pct', 0.02)
        if volatility_pct > 0.05:
            base_probability -= 0.1

        # Ensure bounds
        return max(0.1, min(0.95, base_probability))

    def update_pricing_config(self, config: Dict[str, Any]) -> None:
        """
        Update pricing configuration dynamically.

        Args:
            config: New configuration values
        """
        self.config.update(config)

        # Update instance variables
        self.enabled = config.get('enabled', self.enabled)
        self.atr_window = config.get('atr_window', self.atr_window)
        self.price_adjustment_multiplier = Decimal(str(config.get('price_adjustment_multiplier',
                                                                 self.price_adjustment_multiplier)))
        self.max_price_adjustment_pct = Decimal(str(config.get('max_price_adjustment_pct',
                                                              self.max_price_adjustment_pct)))

        if 'volatility_factors' in config:
            self.volatility_factors.update(config['volatility_factors'])

        if 'spread_factors' in config:
            self.spread_factors.update(config['spread_factors'])

        self.logger.info("Pricing configuration updated")

    def get_pricing_statistics(self) -> Dict[str, Any]:
        """
        Get pricing statistics and configuration.

        Returns:
            Dictionary with pricing statistics
        """
        return {
            'enabled': self.enabled,
            'atr_window': self.atr_window,
            'price_adjustment_multiplier': float(self.price_adjustment_multiplier),
            'max_price_adjustment_pct': float(self.max_price_adjustment_pct),
            'volatility_factors': self.volatility_factors.copy(),
            'spread_factors': self.spread_factors.copy()
        }
