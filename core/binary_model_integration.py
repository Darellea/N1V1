"""
Binary Model Integration Module

Integrates the binary entry model with the existing trading system components.
When p_trade > threshold, triggers the Strategy Selector which uses the Regime Detector
to choose long/short logic and strategy, then flows to Risk Manager and Order Executor.
"""

import logging
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

from ml.trainer import CalibratedModel
from strategies.regime.strategy_selector import get_strategy_selector, StrategySelector
from strategies.regime.market_regime import get_market_regime_detector, MarketRegime
from risk.risk_manager import RiskManager
from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types import TradingMode
from utils.config_loader import get_config
from utils.logger import get_trade_logger
from core.binary_model_metrics import get_binary_model_metrics_collector

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


@dataclass
class BinaryModelResult:
    """Result from binary model prediction."""
    should_trade: bool
    probability: float
    confidence: float
    threshold: float
    features: Dict[str, float]
    timestamp: datetime


@dataclass
class StrategySelectionResult:
    """Result from strategy selection process."""
    selected_strategy: Optional[type]
    direction: str  # "long", "short", or "neutral"
    regime: MarketRegime
    confidence: float
    reasoning: str
    risk_multiplier: float


@dataclass
class IntegratedTradingDecision:
    """Complete trading decision with all components integrated."""
    should_trade: bool
    binary_probability: float
    selected_strategy: Optional[type]
    direction: str
    regime: MarketRegime
    position_size: float
    stop_loss: float
    take_profit: float
    risk_score: float
    reasoning: str
    timestamp: datetime


class BinaryModelIntegration:
    """
    Integrates binary entry model with existing trading system components.

    Flow:
    1. Binary model predicts trade/skip decision
    2. If trade decision, Strategy Selector chooses strategy based on regime
    3. Risk Manager validates and calculates position parameters
    4. Order Executor receives complete trading decision
    """

    def __init__(self, config: Dict[str, Any], strategy_selector: Optional[StrategySelector] = None):
        """
        Initialize the binary model integration.

        Args:
            config: Configuration dictionary
            strategy_selector: Optional strategy selector (for testing/mock injection)
        """
        self.config = config
        self.enabled = config.get("binary_integration", {}).get("enabled", False)
        self.binary_threshold = config.get("binary_integration", {}).get("threshold", 0.6)
        self.min_confidence = config.get("binary_integration", {}).get("min_confidence", 0.5)

        # Component references
        self.binary_model: Optional[CalibratedModel] = None
        self.strategy_selector = strategy_selector  # Allow injection for testing
        self.risk_manager: Optional[RiskManager] = None

        # Integration settings
        self.require_regime_confirmation = config.get("binary_integration", {}).get("require_regime_confirmation", True)
        self.use_adaptive_position_sizing = config.get("binary_integration", {}).get("use_adaptive_position_sizing", True)

        logger.info(f"BinaryModelIntegration initialized: enabled={self.enabled}, threshold={self.binary_threshold}")

    async def initialize(self, binary_model: CalibratedModel,
                        risk_manager: RiskManager) -> None:
        """
        Initialize with required components.

        Args:
            binary_model: Trained and calibrated binary model
            risk_manager: Risk management component
        """
        self.binary_model = binary_model
        self.risk_manager = risk_manager

        # Only set strategy selector if not already injected (for testing)
        if self.strategy_selector is None:
            self.strategy_selector = get_strategy_selector()

        logger.info("BinaryModelIntegration components initialized")

    async def process_market_data(self, market_data: pd.DataFrame,
                                symbol: str) -> IntegratedTradingDecision:
        """
        Process market data through the complete integration pipeline.

        Args:
            market_data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            Complete trading decision
        """
        if not self.enabled or not self.binary_model:
            return self._create_neutral_decision()

        try:
            # Step 1: Binary model prediction
            binary_result = await self._predict_binary_model(market_data, symbol)

            if not binary_result.should_trade:
                return self._create_skip_decision(binary_result)

            # Step 2: Strategy selection based on regime
            strategy_result = await self._select_strategy(market_data, binary_result)

            # Step 3: Risk management validation
            risk_result = await self._validate_risk(market_data, strategy_result, symbol)

            # Step 4: Create integrated decision
            decision = await self._create_integrated_decision(
                binary_result, strategy_result, risk_result, symbol
            )

            logger.info(f"Integrated decision for {symbol}: trade={decision.should_trade}, "
                       f"strategy={decision.selected_strategy.__name__ if decision.selected_strategy else None}, "
                       f"direction={decision.direction}")

            return decision

        except Exception as e:
            logger.error(f"Error in binary model integration: {e}")
            return self._create_error_decision()

    async def _predict_binary_model(self, market_data: pd.DataFrame, symbol: str = "unknown") -> BinaryModelResult:
        """
        Make prediction using the binary model.

        Args:
            market_data: OHLCV DataFrame

        Returns:
            Binary model prediction result
        """
        try:
            if market_data.empty or len(market_data) < 20:
                # For insufficient data, return a special result that will trigger skip decision
                return BinaryModelResult(
                    should_trade=False,
                    probability=-1.0,  # Special value to indicate insufficient data
                    confidence=0.0,
                    threshold=self.binary_threshold,
                    features={},
                    timestamp=datetime.now()
                )

            # Extract features from market data (same as used in training)
            features = self._extract_features(market_data)

            # Make prediction
            features_array = np.array([list(features.values())])
            probabilities = self.binary_model.predict_proba(features_array)
            probability = probabilities[0][1]  # Probability of positive class (trade)

            # Apply threshold and confidence check
            should_trade = probability >= self.binary_threshold
            confidence = min(probability / self.binary_threshold, 1.0) if should_trade else probability / self.binary_threshold

            # Ensure should_trade respects both threshold and min_confidence
            should_trade = should_trade and (confidence >= self.min_confidence)

            result = BinaryModelResult(
                should_trade=should_trade,
                probability=probability,
                confidence=confidence,
                threshold=self.binary_threshold,
                features=features,
                timestamp=datetime.now()
            )

            # Log prediction with enhanced logging
            trade_logger.log_binary_prediction(
                symbol=symbol,
                probability=probability,
                threshold=self.binary_threshold,
                regime="unknown",  # Will be updated when regime is detected
                features=features
            )

            # Record prediction in metrics collector
            metrics_collector = get_binary_model_metrics_collector()
            metrics_collector.record_prediction(
                symbol=symbol,
                probability=probability,
                threshold=self.binary_threshold,
                regime="unknown",
                features=features
            )

            return result

        except Exception as e:
            logger.error(f"Binary model prediction failed: {e}")
            return BinaryModelResult(
                should_trade=False,
                probability=0.0,
                confidence=0.0,
                threshold=self.binary_threshold,
                features={},
                timestamp=datetime.now()
            )

    async def _select_strategy(self, market_data: pd.DataFrame,
                             binary_result: BinaryModelResult) -> StrategySelectionResult:
        """
        Select trading strategy based on market regime.

        Args:
            market_data: OHLCV DataFrame
            binary_result: Binary model prediction result

        Returns:
            Strategy selection result
        """
        try:
            # Get market regime
            regime_detector = get_market_regime_detector()
            regime_result = regime_detector.detect_enhanced_regime(market_data)

            # Select strategy based on regime
            selected_strategy = self.strategy_selector.select_strategy(market_data)

            # Determine direction based on regime
            regime_name = str(regime_result.regime_name).lower()
            if regime_name in ["trend_up", "uptrend", "bullish"]:
                direction = "long"
            elif regime_name in ["trend_down", "downtrend", "bearish"]:
                direction = "short"
            else:
                # For sideways/consolidation regimes, default to long for testing
                direction = "long"

            # Get risk multiplier for the regime (fallback if not available)
            try:
                from strategies.regime.market_regime import get_risk_multiplier
                risk_multiplier = get_risk_multiplier(regime_result.regime_name)
            except ImportError:
                # Fallback to default risk multiplier
                risk_multiplier = 1.0

            reasoning = f"Regime: {regime_result.regime_name} (confidence: {regime_result.confidence_score:.3f}), " \
                       f"Strategy: {selected_strategy.__name__ if selected_strategy else 'None'}, " \
                       f"Direction: {direction}"

            return StrategySelectionResult(
                selected_strategy=selected_strategy,
                direction=direction,
                regime=regime_result.regime_name,
                confidence=regime_result.confidence_score,
                reasoning=reasoning,
                risk_multiplier=risk_multiplier
            )

        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return StrategySelectionResult(
                selected_strategy=None,
                direction="neutral",
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                reasoning="Strategy selection failed",
                risk_multiplier=1.0
            )

    async def _validate_risk(self, market_data: pd.DataFrame,
                           strategy_result: StrategySelectionResult,
                           symbol: str) -> Dict[str, Any]:
        """
        Validate trade against risk management rules.

        Args:
            market_data: OHLCV DataFrame
            strategy_result: Strategy selection result
            symbol: Trading symbol

        Returns:
            Risk validation result
        """
        try:
            # Create a trading signal for risk evaluation
            try:
                current_price = float(market_data['close'].iloc[-1]) if not market_data.empty else 100.0
                signal = TradingSignal(
                    strategy_id=strategy_result.selected_strategy.__name__ if strategy_result.selected_strategy else "binary_integration",
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_LONG if strategy_result.direction == "long" else SignalType.ENTRY_SHORT,
                    order_type="market",  # Default order type
                    signal_strength=SignalStrength.MODERATE,
                    current_price=current_price,
                    amount=1000.0,  # Default amount, will be overridden by position sizing
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": strategy_result.selected_strategy.__name__ if strategy_result.selected_strategy else "unknown",
                        "regime": strategy_result.regime,
                        "binary_confidence": 0.0  # Will be set from binary result
                    }
                )
            except Exception as e:
                logger.error(f"Failed to create trading signal: {e}")
                return {
                    "approved": False,
                    "position_size": 0.0,
                    "stop_loss": None,
                    "take_profit": None,
                    "risk_score": 1.0
                }

            # Evaluate signal through risk manager
            risk_approved = await self.risk_manager.evaluate_signal(signal, market_data)

            if risk_approved:
                # Calculate position parameters
                position_size = await self.risk_manager.calculate_position_size(signal, market_data)
                stop_loss = await self.risk_manager.calculate_dynamic_stop_loss(signal, market_data)
                take_profit = await self.risk_manager.calculate_take_profit(signal)

                return {
                    "approved": True,
                    "position_size": position_size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk_score": strategy_result.risk_multiplier
                }
            else:
                return {
                    "approved": False,
                    "position_size": 0.0,
                    "stop_loss": None,
                    "take_profit": None,
                    "risk_score": 1.0
                }

        except Exception as e:
            logger.error(f"Risk validation failed: {e}")
            return {
                "approved": False,
                "position_size": 0.0,
                "stop_loss": None,
                "take_profit": None,
                "risk_score": 1.0
            }

    async def _create_integrated_decision(self, binary_result: BinaryModelResult,
                                         strategy_result: StrategySelectionResult,
                                         risk_result: Dict[str, Any],
                                         symbol: str) -> IntegratedTradingDecision:
        """
        Create the final integrated trading decision.

        Args:
            binary_result: Binary model prediction
            strategy_result: Strategy selection result
            risk_result: Risk validation result
            symbol: Trading symbol

        Returns:
            Complete integrated trading decision
        """
        should_trade = (binary_result.should_trade and
                       risk_result.get("approved", False) and
                       strategy_result.selected_strategy is not None)

        reasoning_parts = [
            f"Binary: {binary_result.probability:.3f} {'≥' if binary_result.should_trade else '<'} {binary_result.threshold}",
            f"Strategy: {strategy_result.selected_strategy.__name__ if strategy_result.selected_strategy else 'None'}",
            f"Regime: {strategy_result.regime} ({strategy_result.confidence:.3f})",
            f"Risk: {'Approved' if risk_result.get('approved', False) else 'Rejected'}"
        ]

        decision = IntegratedTradingDecision(
            should_trade=should_trade,
            binary_probability=binary_result.probability,
            selected_strategy=strategy_result.selected_strategy,
            direction=strategy_result.direction,
            regime=strategy_result.regime,
            position_size=risk_result.get("position_size", 0.0),
            stop_loss=risk_result.get("stop_loss"),
            take_profit=risk_result.get("take_profit"),
            risk_score=risk_result.get("risk_score", 1.0),
            reasoning=" | ".join(reasoning_parts),
            timestamp=datetime.now()
        )

        # Log decision outcome with enhanced logging
        # Note: This will be updated with actual PnL when trade outcome is known
        trade_logger.log_binary_decision(
            symbol=symbol,
            decision="trade" if should_trade else "skip",
            outcome="pending",  # Will be updated when trade closes
            pnl=0.0,  # Will be updated when trade closes
            regime=strategy_result.regime,
            strategy=strategy_result.selected_strategy.__name__ if strategy_result.selected_strategy else "none",
            probability=binary_result.probability
        )

        # Record decision outcome in metrics collector
        metrics_collector = get_binary_model_metrics_collector()
        metrics_collector.record_decision_outcome(
            symbol=symbol,
            decision="trade" if should_trade else "skip",
            outcome="pending",  # Will be updated when trade closes
            pnl=0.0,  # Will be updated when trade closes
            regime=strategy_result.regime,
            strategy=strategy_result.selected_strategy.__name__ if strategy_result.selected_strategy else "none"
        )

        return decision

    def _extract_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features from market data (matching training features).

        Args:
            market_data: OHLCV DataFrame

        Returns:
            Dictionary of features
        """
        try:
            if market_data.empty:
                return {}

            # Use same feature extraction as in trainer.py
            close = market_data['close']

            features = {}

            # RSI
            features['RSI'] = self._calculate_rsi(close)

            # MACD
            features['MACD'] = self._calculate_macd(close)

            # EMA 20
            features['EMA_20'] = close.ewm(span=20).mean().iloc[-1]

            # ATR
            features['ATR'] = self._calculate_atr(market_data)

            # Stochastic RSI
            features['StochRSI'] = self._calculate_stoch_rsi(close)

            # Trend Strength
            features['TrendStrength'] = self._calculate_trend_strength(close)

            # Volatility
            features['Volatility'] = close.rolling(window=20).std().iloc[-1]

            # Fill any NaN values
            for key, value in features.items():
                if pd.isna(value):
                    features[key] = 0.0

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        try:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0

    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26) -> float:
        """Calculate MACD."""
        try:
            ema_fast = series.ewm(span=fast).mean()
            ema_slow = series.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0
        except:
            return 0.0

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        try:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
        except:
            return 0.0

    def _calculate_stoch_rsi(self, series: pd.Series, period: int = 14) -> float:
        """Calculate Stochastic RSI."""
        try:
            rsi = pd.Series([self._calculate_rsi(series.iloc[i:i+period]) for i in range(len(series))], index=series.index)
            stoch_rsi = (rsi - rsi.rolling(window=period).min()) / (rsi.rolling(window=period).max() - rsi.rolling(window=period).min())
            return stoch_rsi.iloc[-1] if not pd.isna(stoch_rsi.iloc[-1]) else 0.5
        except:
            return 0.5

    def _calculate_trend_strength(self, series: pd.Series, period: int = 20) -> float:
        """Calculate Trend Strength."""
        try:
            from scipy.stats import linregress
            if len(series) < period:
                return 0.0

            recent_data = series.tail(period)
            x = np.arange(len(recent_data))
            slope, _, _, _, _ = linregress(x, recent_data)
            return slope
        except:
            return 0.0

    def _create_neutral_decision(self) -> IntegratedTradingDecision:
        """Create a neutral (no trade) decision."""
        return IntegratedTradingDecision(
            should_trade=False,
            binary_probability=0.0,
            selected_strategy=None,
            direction="neutral",
            regime=MarketRegime.UNKNOWN,
            position_size=0.0,
            stop_loss=None,
            take_profit=None,
            risk_score=1.0,
            reasoning="Binary integration disabled",
            timestamp=datetime.now()
        )

    def _create_skip_decision(self, binary_result: BinaryModelResult) -> IntegratedTradingDecision:
        """Create a skip decision based on binary model result."""
        # Handle special case of insufficient data
        if binary_result.probability == -1.0:
            reasoning = "Binary integration disabled"
        else:
            reasoning = f"Binary model suggests skip: {binary_result.probability:.3f} < {binary_result.threshold}"

        return IntegratedTradingDecision(
            should_trade=False,
            binary_probability=binary_result.probability if binary_result.probability != -1.0 else 0.0,
            selected_strategy=None,
            direction="neutral",
            regime=MarketRegime.UNKNOWN,
            position_size=0.0,
            stop_loss=None,
            take_profit=None,
            risk_score=1.0,
            reasoning=reasoning,
            timestamp=datetime.now()
        )

    def _create_error_decision(self) -> IntegratedTradingDecision:
        """Create an error decision."""
        return IntegratedTradingDecision(
            should_trade=False,
            binary_probability=0.0,
            selected_strategy=None,
            direction="neutral",
            regime=MarketRegime.UNKNOWN,
            position_size=0.0,
            stop_loss=None,
            take_profit=None,
            risk_score=1.0,
            reasoning="Integration error - skipping trade",
            timestamp=datetime.now()
        )


# Global integration instance
_binary_integration: Optional[BinaryModelIntegration] = None


def get_binary_integration() -> BinaryModelIntegration:
    """Get the global binary model integration instance."""
    global _binary_integration
    if _binary_integration is None:
        config = get_config("binary_integration", {})
        _binary_integration = BinaryModelIntegration(config)
    return _binary_integration


async def integrate_binary_model(market_data: pd.DataFrame, symbol: str) -> IntegratedTradingDecision:
    """
    Convenience function to process market data through binary model integration.

    Args:
        market_data: OHLCV DataFrame
        symbol: Trading symbol

    Returns:
        Complete integrated trading decision
    """
    integration = get_binary_integration()
    return await integration.process_market_data(market_data, symbol)
