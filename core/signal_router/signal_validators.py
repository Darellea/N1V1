"""
Signal validation utilities for the SignalRouter.

Handles signal integrity checks, confidence thresholds, duplicates,
and sanity checks for trading signals.
"""

import logging
from typing import TYPE_CHECKING, Optional

from core.contracts import SignalType, TradingSignal

if TYPE_CHECKING:
    from risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class SignalValidator:
    """
    Validates trading signals for integrity and compliance.
    """

    def __init__(self, risk_manager: Optional["RiskManager"] = None):
        """
        Initialize the signal validator.

        Args:
            risk_manager: Risk manager instance for validation rules
        """
        self.risk_manager = risk_manager

    def validate_signal(self, signal: TradingSignal) -> bool:
        """
        Validate a trading signal's basic properties.

        Args:
            signal: The trading signal to validate

        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Basic property validation
            if not self._validate_basic_properties(signal):
                logger.debug("Signal failed basic property validation")
                return False

            # Risk management validation
            if not self._validate_risk_requirements(signal):
                logger.debug("Signal failed risk management validation")
                return False

            # Signal-specific validation
            if not self._validate_signal_specific_rules(signal):
                logger.debug("Signal failed signal-specific validation")
                return False

            logger.debug(f"Signal validation passed for {signal.symbol}")
            return True

        except Exception as e:
            logger.exception(f"Error during signal validation: {e}")
            return False

    def _validate_basic_properties(self, signal: TradingSignal) -> bool:
        """
        Validate basic signal properties.

        Args:
            signal: The trading signal to validate

        Returns:
            True if basic properties are valid
        """
        # Required fields
        if not signal.symbol or not signal.amount or signal.amount <= 0:
            logger.debug("Signal missing required fields or invalid amount")
            return False

        # Strategy ID validation
        if not signal.strategy_id:
            logger.debug("Signal missing strategy ID")
            return False

        # Timestamp validation (should be reasonable)
        if signal.timestamp <= 0:
            logger.debug("Signal has invalid timestamp")
            return False

        return True

    def _validate_risk_requirements(self, signal: TradingSignal) -> bool:
        """
        Validate signal against risk management requirements.

        Args:
            signal: The trading signal to validate

        Returns:
            True if risk requirements are met
        """
        if not self.risk_manager:
            return True  # No risk manager, skip validation

        try:
            # Check stop loss requirements for entry signals
            if signal.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]:
                if (
                    getattr(self.risk_manager, "require_stop_loss", False)
                    and not signal.stop_loss
                ):
                    logger.debug("Signal missing required stop loss")
                    return False

                # Validate stop loss distance if present
                if signal.stop_loss and signal.current_price:
                    stop_distance = (
                        abs(signal.current_price - signal.stop_loss)
                        / signal.current_price
                    )
                    min_stop_distance = getattr(
                        self.risk_manager, "min_stop_distance", 0.005
                    )  # 0.5% default

                    if stop_distance < min_stop_distance:
                        logger.debug(
                            f"Stop loss distance too small: {stop_distance:.4%} < {min_stop_distance:.4%}"
                        )
                        return False

            # Check take profit requirements
            if signal.take_profit and signal.current_price:
                tp_distance = (
                    abs(signal.take_profit - signal.current_price)
                    / signal.current_price
                )
                min_tp_distance = getattr(
                    self.risk_manager, "min_take_profit_distance", 0.01
                )  # 1% default

                if tp_distance < min_tp_distance:
                    logger.debug(
                        f"Take profit distance too small: {tp_distance:.4%} < {min_tp_distance:.4%}"
                    )
                    return False

            return True

        except Exception as e:
            logger.exception(f"Error during risk validation: {e}")
            return False

    def _validate_signal_specific_rules(self, signal: TradingSignal) -> bool:
        """
        Validate signal-specific rules.

        Args:
            signal: The trading signal to validate

        Returns:
            True if signal-specific rules are met
        """
        # Order type validation
        if signal.order_type:
            order_type_value = getattr(signal.order_type, "value", None)
            if order_type_value == "limit" and not signal.price:
                logger.debug("Limit order missing price")
                return False

        # Amount validation based on signal type
        if signal.signal_type in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT]:
            # Exit signals should have reasonable amounts
            if signal.amount > 1e6:  # Sanity check
                logger.debug("Exit signal amount too large")
                return False

        # Price validation
        if signal.price and signal.price <= 0:
            logger.debug("Invalid price")
            return False

        return True

    def validate_signal_confidence(
        self, signal: TradingSignal, min_confidence: float = 0.5
    ) -> bool:
        """
        Validate signal confidence score.

        Args:
            signal: The trading signal to validate
            min_confidence: Minimum required confidence score

        Returns:
            True if confidence meets requirements
        """
        try:
            confidence = getattr(signal, "confidence", None)
            if confidence is None:
                # If no confidence provided, assume it's acceptable
                return True

            if confidence < min_confidence:
                logger.debug(
                    f"Signal confidence too low: {confidence:.3f} < {min_confidence:.3f}"
                )
                return False

            return True

        except Exception as e:
            logger.exception(f"Error validating signal confidence: {e}")
            return False

    def check_for_duplicates(
        self, signal: TradingSignal, recent_signals: list, time_window: int = 300
    ) -> bool:
        """
        Check if signal is a duplicate of recent signals.

        Args:
            signal: The trading signal to check
            recent_signals: List of recent signals
            time_window: Time window in seconds to check for duplicates

        Returns:
            True if signal is not a duplicate
        """
        try:
            current_time = signal.timestamp

            for recent_signal in recent_signals:
                # Check time window
                if abs(current_time - recent_signal.timestamp) > time_window:
                    continue

                # Check if signals are similar
                if (
                    recent_signal.symbol == signal.symbol
                    and recent_signal.strategy_id == signal.strategy_id
                    and recent_signal.signal_type == signal.signal_type
                ):
                    # Check amount similarity (within 10%)
                    amount_diff = (
                        abs(recent_signal.amount - signal.amount) / recent_signal.amount
                    )
                    if amount_diff < 0.1:
                        logger.debug("Duplicate signal detected")
                        return False

            return True

        except Exception as e:
            logger.exception(f"Error checking for duplicates: {e}")
            return True  # Allow signal if check fails

    def perform_sanity_checks(self, signal: TradingSignal) -> bool:
        """
        Perform sanity checks on signal data.

        Args:
            signal: The trading signal to check

        Returns:
            True if all sanity checks pass
        """
        try:
            # Amount sanity checks
            if signal.amount > 1e8:  # $100M position sanity check
                logger.debug("Signal amount exceeds sanity threshold")
                return False

            # Price sanity checks
            if signal.price and signal.price > 1e7:  # $10M per unit sanity check
                logger.debug("Signal price exceeds sanity threshold")
                return False

            # Stop loss sanity checks
            if signal.stop_loss and signal.current_price:
                loss_pct = (
                    abs(signal.stop_loss - signal.current_price) / signal.current_price
                )
                if loss_pct > 0.5:  # 50% stop loss sanity check
                    logger.debug("Stop loss percentage exceeds sanity threshold")
                    return False

            return True

        except Exception as e:
            logger.exception(f"Error during sanity checks: {e}")
            return False
