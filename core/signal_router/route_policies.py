"""
Routing policies for the SignalRouter.

Encapsulates rules for how signals are routed to specific strategies,
executors, or other components based on signal characteristics.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from core.contracts import SignalStrength, SignalType, TradingSignal

if TYPE_CHECKING:
    from core.ensemble_manager import EnsembleManager
    from predictive_models import PredictiveModelManager

logger = logging.getLogger(__name__)


class RoutePolicy:
    """
    Defines routing policies for trading signals.
    """

    def __init__(
        self,
        ensemble_manager: Optional["EnsembleManager"] = None,
        predictive_manager: Optional["PredictiveModelManager"] = None,
    ):
        """
        Initialize the route policy.

        Args:
            ensemble_manager: Ensemble manager for signal combination
            predictive_manager: Predictive models manager
        """
        self.ensemble_manager = ensemble_manager
        self.predictive_manager = predictive_manager

    def get_routing_decision(
        self, signal: TradingSignal, market_data: Dict = None
    ) -> Dict[str, Any]:
        """
        Determine how a signal should be routed.

        Args:
            signal: The trading signal to route
            market_data: Optional market data context

        Returns:
            Dictionary with routing decisions
        """
        decision = {
            "route_to_risk_manager": True,  # Always check risk
            "route_to_executor": False,
            "route_to_ensemble": False,
            "route_to_predictive": False,
            "priority": "normal",
            "reason": "",
        }

        try:
            # Check ensemble routing
            if self._should_route_to_ensemble(signal):
                decision["route_to_ensemble"] = True
                decision["reason"] += "Ensemble processing required. "

            # Check predictive models routing
            if self._should_route_to_predictive(signal):
                decision["route_to_predictive"] = True
                decision["reason"] += "Predictive models required. "

            # Determine priority
            decision["priority"] = self._determine_priority(signal)

            # Check if signal should proceed to execution
            if self._should_route_to_executor(signal):
                decision["route_to_executor"] = True
                decision["reason"] += "Signal approved for execution. "

            logger.debug(f"Routing decision for {signal.symbol}: {decision}")

        except Exception as e:
            logger.exception(f"Error determining routing for signal: {e}")
            # Default to safe routing
            decision["route_to_executor"] = False
            decision["reason"] = "Error in routing decision, blocking execution"

        return decision

    def _should_route_to_ensemble(self, signal: TradingSignal) -> bool:
        """
        Determine if signal should be routed to ensemble manager.

        Args:
            signal: The trading signal

        Returns:
            True if should route to ensemble
        """
        if not self.ensemble_manager or not self.ensemble_manager.enabled:
            return False

        # Route to ensemble if:
        # 1. Signal is from individual strategy (not already ensemble)
        # 2. Ensemble is configured for this signal type
        if (
            signal.strategy_id != "ensemble"
            and hasattr(signal, "metadata")
            and not signal.metadata.get("ensemble", False)
        ):
            # Check if ensemble supports this signal type
            if hasattr(self.ensemble_manager, "supports_signal_type"):
                return self.ensemble_manager.supports_signal_type(signal.signal_type)

            return True

        return False

    def _should_route_to_predictive(self, signal: TradingSignal) -> bool:
        """
        Determine if signal should be routed to predictive models.

        Args:
            signal: The trading signal

        Returns:
            True if should route to predictive models
        """
        if not self.predictive_manager or not self.predictive_manager.enabled:
            return False

        # Route to predictive models for entry signals
        return signal.signal_type in {SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT}

    def _should_route_to_executor(self, signal: TradingSignal) -> bool:
        """
        Determine if signal should be routed to executor.

        Args:
            signal: The trading signal

        Returns:
            True if should route to executor
        """
        # Basic checks for executor routing
        if signal.signal_type not in {
            SignalType.ENTRY_LONG,
            SignalType.ENTRY_SHORT,
            SignalType.EXIT_LONG,
            SignalType.EXIT_SHORT,
        }:
            return False

        # Check signal strength - very weak signals might not go to executor
        if signal.signal_strength == SignalStrength.WEAK:
            # Only route weak signals if they have special metadata
            if hasattr(signal, "metadata") and signal.metadata:
                if signal.metadata.get("force_execute", False):
                    return True
            return False

        return True

    def _determine_priority(self, signal: TradingSignal) -> str:
        """
        Determine the priority level for signal processing.

        Args:
            signal: The trading signal

        Returns:
            Priority level string
        """
        # High priority for exit signals (to close positions quickly)
        if signal.signal_type in {SignalType.EXIT_LONG, SignalType.EXIT_SHORT}:
            return "high"

        # High priority for strong signals
        if signal.signal_strength == SignalStrength.STRONG:
            return "high"

        # Medium priority for moderate signals
        if signal.signal_strength == SignalStrength.MODERATE:
            return "medium"

        # Low priority for weak signals
        return "low"


class MLRoutePolicy:
    """
    Specialized routing policy for ML-filtered signals.
    """

    def __init__(self, ml_config: Dict[str, Any]):
        """
        Initialize ML routing policy.

        Args:
            ml_config: ML configuration
        """
        self.ml_enabled = ml_config.get("enabled", False)
        self.confidence_threshold = ml_config.get("confidence_threshold", 0.6)
        self.fallback_to_raw = ml_config.get("fallback_to_raw_signals", True)

    def should_apply_ml_filter(self, signal: TradingSignal) -> bool:
        """
        Determine if ML filter should be applied to this signal.

        Args:
            signal: The trading signal

        Returns:
            True if ML filter should be applied
        """
        if not self.ml_enabled:
            return False

        # Apply ML filter to entry signals
        return signal.signal_type in {SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT}

    def handle_ml_result(
        self, signal: TradingSignal, ml_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle ML filtering result and determine next steps.

        Args:
            signal: The trading signal
            ml_result: ML filtering result

        Returns:
            Dictionary with handling decisions
        """
        decision = {
            "approved": True,
            "modified_signal": signal,
            "reason": "",
            "confidence": ml_result.get("confidence", 0.0),
        }

        if not ml_result.get("approved", True):
            decision["approved"] = False
            decision["reason"] = ml_result.get("reason", "ML rejected")
            logger.info(f"ML rejected signal: {decision['reason']}")
            return decision

        # Check confidence threshold
        confidence = ml_result.get("confidence", 0.0)
        if confidence < self.confidence_threshold:
            if not self.fallback_to_raw:
                decision["approved"] = False
                decision[
                    "reason"
                ] = f"Low confidence: {confidence:.3f} < {self.confidence_threshold}"
                return decision
            else:
                logger.info(
                    f"ML confidence low ({confidence:.3f}), proceeding with raw signal"
                )

        # Apply ML-driven signal modifications
        if (
            signal.signal_strength == SignalStrength.WEAK
            and confidence >= self.confidence_threshold
        ):
            # Upgrade weak signal if ML has high confidence
            decision["modified_signal"] = signal._replace(
                signal_strength=SignalStrength.MODERATE
            )
            decision["reason"] = "Signal strength upgraded by ML"

        return decision


class MarketRegimeRoutePolicy:
    """
    Routing policy based on market regime detection.
    """

    def __init__(self, regime_config: Dict[str, Any]):
        """
        Initialize market regime routing policy.

        Args:
            regime_config: Market regime configuration
        """
        self.regime_enabled = regime_config.get("enabled", False)
        self.risk_adjustment = regime_config.get("risk_adjustment", True)

    def adjust_for_regime(self, signal: TradingSignal, regime: str) -> TradingSignal:
        """
        Adjust signal based on current market regime.

        Args:
            signal: The trading signal
            regime: Current market regime

        Returns:
            Adjusted trading signal
        """
        if not self.regime_enabled or not self.risk_adjustment:
            return signal

        # Adjust position size based on regime
        adjustment_factor = self._get_regime_adjustment(regime)

        if adjustment_factor != 1.0:
            new_amount = signal.amount * adjustment_factor
            logger.info(
                f"Adjusted signal amount for {regime} regime: "
                f"{signal.amount} -> {new_amount}"
            )

            # Create adjusted signal
            adjusted_signal = signal._replace(amount=new_amount)

            # Add regime info to metadata
            if (
                not hasattr(adjusted_signal, "metadata")
                or adjusted_signal.metadata is None
            ):
                adjusted_signal = adjusted_signal._replace(metadata={})

            if adjusted_signal.metadata is None:
                adjusted_signal = adjusted_signal._replace(metadata={})

            adjusted_signal.metadata["regime_adjustment"] = {
                "original_amount": signal.amount,
                "adjustment_factor": adjustment_factor,
                "regime": regime,
            }

            return adjusted_signal

        return signal

    def _get_regime_adjustment(self, regime: str) -> float:
        """
        Get position size adjustment factor for regime.

        Args:
            regime: Market regime

        Returns:
            Adjustment factor
        """
        regime_adjustments = {
            "bull": 1.2,  # Increase position in bull markets
            "bear": 0.7,  # Reduce position in bear markets
            "sideways": 0.9,  # Slightly reduce in sideways markets
            "volatile": 0.6,  # Significantly reduce in volatile markets
            "calm": 1.1,  # Slightly increase in calm markets
        }

        return regime_adjustments.get(regime, 1.0)


# Convenience functions
def get_default_route_policy(
    ensemble_manager=None, predictive_manager=None
) -> RoutePolicy:
    """
    Get default routing policy instance.

    Args:
        ensemble_manager: Optional ensemble manager
        predictive_manager: Optional predictive manager

    Returns:
        RoutePolicy instance
    """
    return RoutePolicy(ensemble_manager, predictive_manager)


def get_ml_route_policy(ml_config: Dict[str, Any]) -> MLRoutePolicy:
    """
    Get ML routing policy instance.

    Args:
        ml_config: ML configuration

    Returns:
        MLRoutePolicy instance
    """
    return MLRoutePolicy(ml_config)


def get_regime_route_policy(regime_config: Dict[str, Any]) -> MarketRegimeRoutePolicy:
    """
    Get market regime routing policy instance.

    Args:
        regime_config: Market regime configuration

    Returns:
        MarketRegimeRoutePolicy instance
    """
    return MarketRegimeRoutePolicy(regime_config)
