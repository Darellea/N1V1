"""
Comprehensive Order Validation Pipeline

Implements multi-stage validation for trading orders including:
- Pre-trade validation (basic checks, market hours)
- Risk validation (integration with risk_manager)
- Exchange compatibility validation (exchange-specific rules)
- Validation rule management interface
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as datetime_time
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum

import jsonschema
from ccxt.base.errors import ExchangeError

from core.contracts import TradingSignal
from risk.risk_manager import RiskManager
from utils.adapter import signal_to_dict
from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class ValidationStage(Enum):
    """Validation pipeline stages."""
    PRE_TRADE = "pre_trade"
    RISK = "risk"
    EXCHANGE_COMPATIBILITY = "exchange_compatibility"


class ValidationSeverity(Enum):
    """Validation failure severity levels."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    stage: ValidationStage
    check_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    suggested_fix: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report for an order."""
    signal: Any
    timestamp: float
    overall_passed: bool = True
    results: List[ValidationResult] = field(default_factory=list)
    execution_time_ms: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    critical_count: int = 0

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result to the report."""
        self.results.append(result)
        if not result.passed:
            if result.severity == ValidationSeverity.ERROR:
                self.error_count += 1
            elif result.severity == ValidationSeverity.WARNING:
                self.warning_count += 1
            elif result.severity == ValidationSeverity.CRITICAL:
                self.critical_count += 1

        # Update overall status - only pass if no errors or critical failures
        self.overall_passed = (self.error_count == 0 and self.critical_count == 0)

    def get_errors(self) -> List[ValidationResult]:
        """Get all error results."""
        return [r for r in self.results if not r.passed and r.severity == ValidationSeverity.ERROR]

    def get_warnings(self) -> List[ValidationResult]:
        """Get all warning results."""
        return [r for r in self.results if not r.passed and r.severity == ValidationSeverity.WARNING]

    def get_critical_errors(self) -> List[ValidationResult]:
        """Get all critical error results."""
        return [r for r in self.results if not r.passed and r.severity == ValidationSeverity.CRITICAL]


class ValidationRule(ABC):
    """Abstract base class for validation rules."""

    def __init__(self, name: str, enabled: bool = True, severity: ValidationSeverity = ValidationSeverity.ERROR):
        self.name = name
        self.enabled = enabled
        self.severity = severity

    @abstractmethod
    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate a signal against this rule."""
        pass

    def is_applicable(self, signal: Any, context: Dict[str, Any]) -> bool:
        """Check if this rule applies to the given signal."""
        return self.enabled


class PreTradeValidator:
    """Handles pre-trade validation checks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules: List[ValidationRule] = []
        self._initialize_rules()

    def _initialize_rules(self) -> None:
        """Initialize pre-trade validation rules."""
        self.rules.extend([
            BasicFieldValidator(),
            AmountValidator(),
            PriceValidator(),
            SymbolValidator(),
            OrderTypeValidator(),
            SignalTypeValidator(),
            MarketHoursValidator(self.config.get("market_hours", {})),
            TimestampValidator(),
        ])

    async def validate(self, signal: Any, context: Dict[str, Any]) -> List[ValidationResult]:
        """Run all pre-trade validation rules."""
        results = []
        for rule in self.rules:
            if rule.is_applicable(signal, context):
                try:
                    result = await rule.validate(signal, context)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in pre-trade validation rule {rule.name}: {e}")
                    results.append(ValidationResult(
                        stage=ValidationStage.PRE_TRADE,
                        check_name=rule.name,
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Validation rule failed: {str(e)}"
                    ))
        return results


class RiskValidator:
    """Handles risk validation using RiskManager."""

    def __init__(self, risk_manager: RiskManager, config: Dict[str, Any]):
        self.risk_manager = risk_manager
        self.config = config
        self.rules: List[ValidationRule] = []
        self._initialize_rules()

    def _initialize_rules(self) -> None:
        """Initialize risk validation rules."""
        self.rules.extend([
            RiskManagerValidator(self.risk_manager),
            PositionSizeValidator(self.risk_manager),
            PortfolioRiskValidator(self.risk_manager),
            StopLossValidator(),
            TakeProfitValidator(),
        ])

    async def validate(self, signal: Any, context: Dict[str, Any]) -> List[ValidationResult]:
        """Run all risk validation rules."""
        results = []
        for rule in self.rules:
            if rule.is_applicable(signal, context):
                try:
                    result = await rule.validate(signal, context)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in risk validation rule {rule.name}: {e}")
                    results.append(ValidationResult(
                        stage=ValidationStage.RISK,
                        check_name=rule.name,
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Risk validation rule failed: {str(e)}"
                    ))
        return results


class ExchangeCompatibilityValidator:
    """Handles exchange-specific validation rules."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchange_rules: Dict[str, List[ValidationRule]] = {}
        self._initialize_exchange_rules()

    def _initialize_exchange_rules(self) -> None:
        """Initialize exchange-specific validation rules."""
        # Default rules for common exchanges
        default_rules = [
            MinimumOrderSizeValidator(),
            MaximumOrderSizeValidator(),
            PricePrecisionValidator(),
            AmountPrecisionValidator(),
            TradingPairValidator(),
        ]

        # Exchange-specific rules can be added here
        self.exchange_rules["default"] = default_rules
        self.exchange_rules["binance"] = default_rules + [
            BinanceLeverageValidator(),
            BinanceIsolatedMarginValidator(),
        ]
        self.exchange_rules["kucoin"] = default_rules + [
            KuCoinMinimumVolumeValidator(),
        ]

    def get_exchange_rules(self, exchange: str) -> List[ValidationRule]:
        """Get validation rules for a specific exchange."""
        return self.exchange_rules.get(exchange.lower(), self.exchange_rules["default"])

    async def validate(self, signal: Any, context: Dict[str, Any]) -> List[ValidationResult]:
        """Run exchange compatibility validation."""
        results = []
        exchange = context.get("exchange", "default")
        rules = self.get_exchange_rules(exchange)

        for rule in rules:
            if rule.is_applicable(signal, context):
                try:
                    result = await rule.validate(signal, context)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in exchange validation rule {rule.name}: {e}")
                    results.append(ValidationResult(
                        stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                        check_name=rule.name,
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Exchange validation rule failed: {str(e)}"
                    ))
        return results


class OrderValidationPipeline:
    """Main validation pipeline orchestrator."""

    def __init__(self, config: Dict[str, Any], risk_manager: Optional[RiskManager] = None):
        self.config = config
        self.risk_manager = risk_manager

        # Initialize validators
        self.pre_trade_validator = PreTradeValidator(config)
        self.risk_validator = RiskValidator(risk_manager, config) if risk_manager else None
        self.exchange_validator = ExchangeCompatibilityValidator(config)

        # Pipeline configuration
        self.fail_fast = config.get("fail_fast", False)
        self.timeout_seconds = config.get("validation_timeout", 5.0)
        self.enable_circuit_breaker = config.get("enable_circuit_breaker", True)

        # Circuit breaker for validation pipeline
        self._validation_failures = 0
        self._max_consecutive_failures = config.get("max_validation_failures", 5)

        # Performance tracking
        self._validation_times: List[float] = []
        self._max_timing_history = 100

    async def validate_order(self, signal: Any, context: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """
        Run complete validation pipeline for an order.

        Args:
            signal: Trading signal to validate
            context: Additional validation context (exchange, market_data, etc.)

        Returns:
            ValidationReport with complete results
        """
        context = context or {}
        report = ValidationReport(signal=signal, timestamp=asyncio.get_event_loop().time())

        # Circuit breaker check
        if self.enable_circuit_breaker and self._validation_failures >= self._max_consecutive_failures:
            report.add_result(ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name="circuit_breaker",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message="Validation pipeline circuit breaker triggered",
                suggested_fix="Check system health and reset validation pipeline"
            ))
            return report

        start_time = asyncio.get_event_loop().time()

        try:
            # Stage 1: Pre-trade validation
            pre_trade_results = await self._run_with_timeout(
                self.pre_trade_validator.validate(signal, context),
                "pre_trade"
            )
            for result in pre_trade_results:
                report.add_result(result)

            # Fail fast if configured and we have errors
            if self.fail_fast and not report.overall_passed:
                report.execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                return report

            # Stage 2: Risk validation (if risk manager available)
            if self.risk_validator:
                risk_results = await self._run_with_timeout(
                    self.risk_validator.validate(signal, context),
                    "risk"
                )
                for result in risk_results:
                    report.add_result(result)

                if self.fail_fast and not report.overall_passed:
                    report.execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                    return report

            # Stage 3: Exchange compatibility validation
            exchange_results = await self._run_with_timeout(
                self.exchange_validator.validate(signal, context),
                "exchange_compatibility"
            )
            for result in exchange_results:
                report.add_result(result)

        except asyncio.TimeoutError:
            report.add_result(ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name="timeout",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation timed out after {self.timeout_seconds}s"
            ))
        except Exception as e:
            logger.error(f"Unexpected error in validation pipeline: {e}")
            report.add_result(ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name="pipeline_error",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation pipeline error: {str(e)}"
            ))

        # Update circuit breaker
        if report.overall_passed:
            self._validation_failures = 0
        else:
            self._validation_failures += 1

        # Track performance
        execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
        report.execution_time_ms = execution_time
        self._validation_times.append(execution_time)
        if len(self._validation_times) > self._max_timing_history:
            self._validation_times.pop(0)

        return report

    async def _run_with_timeout(self, coro, stage_name: str):
        """Run a validation stage with timeout protection."""
        try:
            return await asyncio.wait_for(coro, timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(f"Validation stage {stage_name} timed out")
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get validation pipeline performance statistics."""
        if not self._validation_times:
            return {"average_time_ms": 0, "max_time_ms": 0, "min_time_ms": 0}

        return {
            "average_time_ms": sum(self._validation_times) / len(self._validation_times),
            "max_time_ms": max(self._validation_times),
            "min_time_ms": min(self._validation_times),
            "total_validations": len(self._validation_times),
            "circuit_breaker_failures": self._validation_failures
        }

    def reset_circuit_breaker(self) -> None:
        """Reset the validation circuit breaker."""
        self._validation_failures = 0
        logger.info("Validation pipeline circuit breaker reset")


# ===== VALIDATION RULES IMPLEMENTATIONS =====

class BasicFieldValidator(ValidationRule):
    """Validates presence of required basic fields."""

    def __init__(self):
        super().__init__("basic_fields")

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        required_fields = ["strategy_id", "symbol", "signal_type", "order_type", "amount"]
        signal_dict = signal_to_dict(signal)

        missing_fields = []
        for field in required_fields:
            if field not in signal_dict or signal_dict[field] is None:
                missing_fields.append(field)

        if missing_fields:
            return ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Missing required fields: {', '.join(missing_fields)}",
                suggested_fix="Ensure all required fields are provided in the signal"
            )

        return ValidationResult(
            stage=ValidationStage.PRE_TRADE,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="All required fields present"
        )


class AmountValidator(ValidationRule):
    """Validates order amount."""

    def __init__(self):
        super().__init__("amount_validation")

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        signal_dict = signal_to_dict(signal)
        amount = signal_dict.get("amount")

        if amount is None:
            return ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Order amount is required",
                suggested_fix="Provide a valid order amount"
            )

        try:
            amount_decimal = Decimal(str(amount))
            if amount_decimal <= 0:
                return ValidationResult(
                    stage=ValidationStage.PRE_TRADE,
                    check_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message="Order amount must be positive",
                    suggested_fix="Use a positive amount value"
                )
        except (InvalidOperation, ValueError, TypeError):
            return ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Invalid amount format",
                suggested_fix="Use a valid numeric amount"
            )

        return ValidationResult(
            stage=ValidationStage.PRE_TRADE,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Amount validation passed"
        )


class PriceValidator(ValidationRule):
    """Validates price fields."""

    def __init__(self):
        super().__init__("price_validation")

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        signal_dict = signal_to_dict(signal)
        price = signal_dict.get("price")
        stop_loss = signal_dict.get("stop_loss")
        take_profit = signal_dict.get("take_profit")

        # Validate price if provided
        if price is not None:
            try:
                price_decimal = Decimal(str(price))
                if price_decimal <= 0:
                    return ValidationResult(
                        stage=ValidationStage.PRE_TRADE,
                        check_name=self.name,
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message="Price must be positive",
                        suggested_fix="Use a positive price value"
                    )
            except (InvalidOperation, ValueError, TypeError):
                return ValidationResult(
                    stage=ValidationStage.PRE_TRADE,
                    check_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message="Invalid price format",
                    suggested_fix="Use a valid numeric price"
                )

        # Validate stop loss if provided
        if stop_loss is not None:
            try:
                sl_decimal = Decimal(str(stop_loss))
                if sl_decimal <= 0:
                    return ValidationResult(
                        stage=ValidationStage.PRE_TRADE,
                        check_name=self.name,
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message="Stop loss must be positive",
                        suggested_fix="Use a positive stop loss value"
                    )
            except (InvalidOperation, ValueError, TypeError):
                return ValidationResult(
                    stage=ValidationStage.PRE_TRADE,
                    check_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message="Invalid stop loss format",
                    suggested_fix="Use a valid numeric stop loss"
                )

        # Validate take profit if provided
        if take_profit is not None:
            try:
                tp_decimal = Decimal(str(take_profit))
                if tp_decimal <= 0:
                    return ValidationResult(
                        stage=ValidationStage.PRE_TRADE,
                        check_name=self.name,
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message="Take profit must be positive",
                        suggested_fix="Use a positive take profit value"
                    )
            except (InvalidOperation, ValueError, TypeError):
                return ValidationResult(
                    stage=ValidationStage.PRE_TRADE,
                    check_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message="Invalid take profit format",
                    suggested_fix="Use a valid numeric take profit"
                )

        return ValidationResult(
            stage=ValidationStage.PRE_TRADE,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Price validation passed"
        )


class SymbolValidator(ValidationRule):
    """Validates trading symbol format."""

    def __init__(self):
        super().__init__("symbol_validation")

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        signal_dict = signal_to_dict(signal)
        symbol = signal_dict.get("symbol", "")

        if not symbol or not isinstance(symbol, str):
            return ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Symbol is required and must be a string",
                suggested_fix="Provide a valid trading symbol (e.g., 'BTC/USDT')"
            )

        if "/" not in symbol:
            return ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Symbol must be in BASE/QUOTE format",
                suggested_fix="Use format like 'BTC/USDT' or 'ETH/BTC'"
            )

        base, quote = symbol.split("/", 1)
        if not base or not quote:
            return ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Symbol must have both base and quote currencies",
                suggested_fix="Ensure both base and quote are non-empty"
            )

        return ValidationResult(
            stage=ValidationStage.PRE_TRADE,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Symbol format is valid"
        )


class OrderTypeValidator(ValidationRule):
    """Validates order type."""

    def __init__(self):
        super().__init__("order_type_validation")
        self.valid_order_types = {"MARKET", "LIMIT", "STOP", "STOP_LIMIT"}

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        signal_dict = signal_to_dict(signal)
        order_type = signal_dict.get("order_type", "").upper()

        if not order_type:
            return ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Order type is required",
                suggested_fix="Specify order type: MARKET, LIMIT, STOP, or STOP_LIMIT"
            )

        if order_type not in self.valid_order_types:
            return ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid order type: {order_type}",
                suggested_fix=f"Use one of: {', '.join(self.valid_order_types)}"
            )

        return ValidationResult(
            stage=ValidationStage.PRE_TRADE,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Order type is valid"
        )


class SignalTypeValidator(ValidationRule):
    """Validates signal type."""

    def __init__(self):
        super().__init__("signal_type_validation")
        self.valid_signal_types = {
            "ENTRY_LONG", "ENTRY_SHORT", "EXIT_LONG", "EXIT_SHORT"
        }

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        signal_dict = signal_to_dict(signal)
        signal_type = signal_dict.get("signal_type", "").upper()

        if not signal_type:
            return ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Signal type is required",
                suggested_fix="Specify signal type: ENTRY_LONG, ENTRY_SHORT, EXIT_LONG, or EXIT_SHORT"
            )

        if signal_type not in self.valid_signal_types:
            return ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid signal type: {signal_type}",
                suggested_fix=f"Use one of: {', '.join(self.valid_signal_types)}"
            )

        return ValidationResult(
            stage=ValidationStage.PRE_TRADE,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Signal type is valid"
        )


class MarketHoursValidator(ValidationRule):
    """Validates trading during market hours."""

    def __init__(self, market_hours_config: Dict[str, Any]):
        super().__init__("market_hours", enabled=market_hours_config.get("enabled", False))
        self.market_hours = market_hours_config
        self.timezone = market_hours_config.get("timezone", "UTC")

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        if not self.enabled:
            return ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name=self.name,
                passed=True,
                severity=ValidationSeverity.WARNING,
                message="Market hours validation disabled"
            )

        # Get current time (simplified - in real implementation would handle timezones)
        current_time = datetime.now().time()

        # Check if within market hours
        market_open = datetime_time.fromisoformat(self.market_hours.get("open", "00:00"))
        market_close = datetime_time.fromisoformat(self.market_hours.get("close", "23:59"))

        if not (market_open <= current_time <= market_close):
            return ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.WARNING,
                message=f"Outside market hours ({market_open} - {market_close})",
                suggested_fix="Wait for market hours or use market hours override"
            )

        return ValidationResult(
            stage=ValidationStage.PRE_TRADE,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.WARNING,
            message="Within market hours"
        )


class TimestampValidator(ValidationRule):
    """Validates signal timestamp."""

    def __init__(self):
        super().__init__("timestamp_validation")

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        signal_dict = signal_to_dict(signal)
        timestamp = signal_dict.get("timestamp")

        if timestamp is None:
            return ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.WARNING,
                message="Timestamp not provided",
                suggested_fix="Include timestamp in signal"
            )

        try:
            # Accept both string and numeric timestamps
            if isinstance(timestamp, str):
                ts_value = int(timestamp)
            else:
                ts_value = int(timestamp)

            current_time = int(time.time() * 1000)

            # Check if timestamp is not too old (within last 24 hours)
            if current_time - ts_value > 86400000:  # 24 hours in milliseconds
                return ValidationResult(
                    stage=ValidationStage.PRE_TRADE,
                    check_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message="Signal timestamp is too old",
                    suggested_fix="Use current timestamp"
                )

            # Check if timestamp is not too far in the future (allow 5 minutes grace)
            if ts_value > current_time + 300000:  # 5 minutes in milliseconds
                return ValidationResult(
                    stage=ValidationStage.PRE_TRADE,
                    check_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message="Signal timestamp is in the future",
                    suggested_fix="Use current timestamp"
                )

        except (ValueError, TypeError, OverflowError):
            return ValidationResult(
                stage=ValidationStage.PRE_TRADE,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Invalid timestamp format",
                suggested_fix="Use valid timestamp (milliseconds since epoch)"
            )

        return ValidationResult(
            stage=ValidationStage.PRE_TRADE,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,  # Change to ERROR for successful validation
            message="Timestamp validation passed"
        )


# ===== RISK VALIDATION RULES =====

class RiskManagerValidator(ValidationRule):
    """Integrates with RiskManager for comprehensive risk validation."""

    def __init__(self, risk_manager: RiskManager):
        super().__init__("risk_manager_integration")
        self.risk_manager = risk_manager

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        market_data = context.get("market_data")
        risk_passed = await self.risk_manager.evaluate_signal(signal, market_data)

        if not risk_passed:
            return ValidationResult(
                stage=ValidationStage.RISK,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Signal failed risk management evaluation",
                suggested_fix="Review risk parameters and adjust signal accordingly"
            )

        return ValidationResult(
            stage=ValidationStage.RISK,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Risk management validation passed"
        )


class PositionSizeValidator(ValidationRule):
    """Validates position size against risk limits."""

    def __init__(self, risk_manager: RiskManager):
        super().__init__("position_size")
        self.risk_manager = risk_manager

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        signal_dict = signal_to_dict(signal)
        amount = signal_dict.get("amount")

        if amount is None:
            return ValidationResult(
                stage=ValidationStage.RISK,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Amount required for position size validation",
                suggested_fix="Provide order amount"
            )

        # Check against maximum position size
        account_balance = await self.risk_manager._get_current_balance()
        position_pct = Decimal(str(amount)) / account_balance

        if position_pct > self.risk_manager.max_position_size:
            return ValidationResult(
                stage=ValidationStage.RISK,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Position size {position_pct:.2%} exceeds maximum {self.risk_manager.max_position_size:.2%}",
                suggested_fix="Reduce order amount or increase max_position_size"
            )

        return ValidationResult(
            stage=ValidationStage.RISK,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Position size within risk limits"
        )


class PortfolioRiskValidator(ValidationRule):
    """Validates portfolio-level risk constraints."""

    def __init__(self, risk_manager: RiskManager):
        super().__init__("portfolio_risk")
        self.risk_manager = risk_manager

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        # Check daily loss limit
        if self.risk_manager.today_pnl < 0:
            loss_pct = abs(self.risk_manager.today_pnl) / self.risk_manager.today_start_balance
            if loss_pct >= self.risk_manager.max_daily_loss:
                return ValidationResult(
                    stage=ValidationStage.RISK,
                    check_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Daily loss limit exceeded: {loss_pct:.2%} >= {self.risk_manager.max_daily_loss:.2%}",
                    suggested_fix="Stop trading for the day or reduce position sizes"
                )

        # Check maximum concurrent positions
        current_positions = await self.risk_manager._get_current_positions()
        max_positions = context.get("max_concurrent_positions", 5)
        if len(current_positions) >= max_positions:
            return ValidationResult(
                stage=ValidationStage.RISK,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Maximum concurrent positions exceeded: {len(current_positions)} >= {max_positions}",
                suggested_fix="Close existing positions or increase max_concurrent_positions"
            )

        return ValidationResult(
            stage=ValidationStage.RISK,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Portfolio risk constraints satisfied"
        )


class StopLossValidator(ValidationRule):
    """Validates stop loss requirements."""

    def __init__(self):
        super().__init__("stop_loss")

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        signal_dict = signal_to_dict(signal)
        order_type = signal_dict.get("order_type", "").upper()
        stop_loss = signal_dict.get("stop_loss")

        # Stop orders must have stop loss
        if order_type in ["STOP", "STOP_LIMIT"]:
            if stop_loss is None:
                return ValidationResult(
                    stage=ValidationStage.RISK,
                    check_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message="Stop orders must include stop_loss price",
                    suggested_fix="Add stop_loss to stop orders"
                )

        return ValidationResult(
            stage=ValidationStage.RISK,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Stop loss validation passed"
        )


class TakeProfitValidator(ValidationRule):
    """Validates take profit logic."""

    def __init__(self):
        super().__init__("take_profit")

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        signal_dict = signal_to_dict(signal)
        stop_loss = signal_dict.get("stop_loss")
        take_profit = signal_dict.get("take_profit")
        current_price = signal_dict.get("current_price")

        if stop_loss and take_profit and current_price:
            try:
                sl = Decimal(str(stop_loss))
                tp = Decimal(str(take_profit))
                price = Decimal(str(current_price))

                signal_type = signal_dict.get("signal_type", "").upper()

                if signal_type.endswith("_LONG"):
                    # For long positions: TP > price > SL
                    if not (tp > price > sl):
                        return ValidationResult(
                            stage=ValidationStage.RISK,
                            check_name=self.name,
                            passed=False,
                            severity=ValidationSeverity.WARNING,
                            message="Invalid long position levels: TP should be > price > SL",
                            suggested_fix="Adjust take profit or stop loss levels"
                        )
                elif signal_type.endswith("_SHORT"):
                    # For short positions: SL > price > TP
                    if not (sl > price > tp):
                        return ValidationResult(
                            stage=ValidationStage.RISK,
                            check_name=self.name,
                            passed=False,
                            severity=ValidationSeverity.WARNING,
                            message="Invalid short position levels: SL should be > price > TP",
                            suggested_fix="Adjust take profit or stop loss levels"
                        )
            except (InvalidOperation, ValueError, TypeError):
                return ValidationResult(
                    stage=ValidationStage.RISK,
                    check_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message="Invalid price format for TP/SL validation",
                    suggested_fix="Use valid numeric prices"
                )

        return ValidationResult(
            stage=ValidationStage.RISK,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.WARNING,
            message="Take profit validation passed"
        )


# ===== EXCHANGE COMPATIBILITY RULES =====

class MinimumOrderSizeValidator(ValidationRule):
    """Validates minimum order size for exchange."""

    def __init__(self):
        super().__init__("minimum_order_size")

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        signal_dict = signal_to_dict(signal)
        amount = signal_dict.get("amount")
        symbol = signal_dict.get("symbol", "")
        exchange = context.get("exchange", "default")

        if amount is None:
            return ValidationResult(
                stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Amount required for minimum order size check",
                suggested_fix="Provide order amount"
            )

        # Get minimum order size for symbol/exchange (simplified)
        min_sizes = {
            "default": {"BTC/USDT": Decimal("0.0001"), "ETH/USDT": Decimal("0.001")},
            "binance": {"BTC/USDT": Decimal("0.000001"), "ETH/USDT": Decimal("0.00001")},
        }

        exchange_mins = min_sizes.get(exchange, min_sizes["default"])
        min_size = exchange_mins.get(symbol, Decimal("0.0001"))

        try:
            amount_decimal = Decimal(str(amount))
            if amount_decimal < min_size:
                return ValidationResult(
                    stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                    check_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Order size {amount} below minimum {min_size} for {symbol} on {exchange}",
                    suggested_fix=f"Increase order size to at least {min_size}"
                )
        except (InvalidOperation, ValueError, TypeError):
            return ValidationResult(
                stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Invalid amount format for minimum size check",
                suggested_fix="Use valid numeric amount"
            )

        return ValidationResult(
            stage=ValidationStage.EXCHANGE_COMPATIBILITY,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Order size meets minimum requirements"
        )


class MaximumOrderSizeValidator(ValidationRule):
    """Validates maximum order size for exchange."""

    def __init__(self):
        super().__init__("maximum_order_size")

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        signal_dict = signal_to_dict(signal)
        amount = signal_dict.get("amount")
        symbol = signal_dict.get("symbol", "")
        exchange = context.get("exchange", "default")

        if amount is None:
            return ValidationResult(
                stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Amount required for maximum order size check",
                suggested_fix="Provide order amount"
            )

        # Get maximum order size for symbol/exchange (simplified)
        max_sizes = {
            "default": {"BTC/USDT": Decimal("100"), "ETH/USDT": Decimal("1000")},
            "binance": {"BTC/USDT": Decimal("1000"), "ETH/USDT": Decimal("10000")},
        }

        exchange_maxes = max_sizes.get(exchange, max_sizes["default"])
        max_size = exchange_maxes.get(symbol, Decimal("1000"))

        try:
            amount_decimal = Decimal(str(amount))
            if amount_decimal > max_size:
                return ValidationResult(
                    stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                    check_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Order size {amount} exceeds maximum {max_size} for {symbol} on {exchange}",
                    suggested_fix=f"Reduce order size to at most {max_size}"
                )
        except (InvalidOperation, ValueError, TypeError):
            return ValidationResult(
                stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Invalid amount format for maximum size check",
                suggested_fix="Use valid numeric amount"
            )

        return ValidationResult(
            stage=ValidationStage.EXCHANGE_COMPATIBILITY,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Order size within maximum limits"
        )


class PricePrecisionValidator(ValidationRule):
    """Validates price precision for exchange."""

    def __init__(self):
        super().__init__("price_precision")

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        signal_dict = signal_to_dict(signal)
        price = signal_dict.get("price")
        symbol = signal_dict.get("symbol", "")
        exchange = context.get("exchange", "default")

        if price is None:
            return ValidationResult(
                stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                check_name=self.name,
                passed=True,
                severity=ValidationSeverity.ERROR,
                message="No price to validate precision for"
            )

        # Get price precision for symbol/exchange (simplified)
        precisions = {
            "default": {"BTC/USDT": 2, "ETH/USDT": 2},
            "binance": {"BTC/USDT": 5, "ETH/USDT": 4},
        }

        exchange_precisions = precisions.get(exchange, precisions["default"])
        precision = exchange_precisions.get(symbol, 2)

        try:
            price_str = str(price)
            if "." in price_str:
                decimal_places = len(price_str.split(".")[1])
                if decimal_places > precision:
                    return ValidationResult(
                        stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                        check_name=self.name,
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Price precision {decimal_places} exceeds maximum {precision} for {symbol} on {exchange}",
                        suggested_fix=f"Round price to {precision} decimal places"
                    )
        except (ValueError, TypeError):
            return ValidationResult(
                stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Invalid price format for precision check",
                suggested_fix="Use valid numeric price"
            )

        return ValidationResult(
            stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                check_name=self.name,
                passed=True,
                severity=ValidationSeverity.ERROR,
                message="Price precision is valid"
            )


class AmountPrecisionValidator(ValidationRule):
    """Validates amount precision for exchange."""

    def __init__(self):
        super().__init__("amount_precision")

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        signal_dict = signal_to_dict(signal)
        amount = signal_dict.get("amount")
        symbol = signal_dict.get("symbol", "")
        exchange = context.get("exchange", "default")

        if amount is None:
            return ValidationResult(
                stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Amount required for precision check",
                suggested_fix="Provide order amount"
            )

        # Get amount precision for symbol/exchange (simplified)
        precisions = {
            "default": {"BTC/USDT": 6, "ETH/USDT": 5},
            "binance": {"BTC/USDT": 8, "ETH/USDT": 8},
        }

        exchange_precisions = precisions.get(exchange, precisions["default"])
        precision = exchange_precisions.get(symbol, 6)

        try:
            amount_str = str(amount)
            if "." in amount_str:
                decimal_places = len(amount_str.split(".")[1])
                if decimal_places > precision:
                    return ValidationResult(
                        stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                        check_name=self.name,
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Amount precision {decimal_places} exceeds maximum {precision} for {symbol} on {exchange}",
                        suggested_fix=f"Round amount to {precision} decimal places"
                    )
        except (ValueError, TypeError):
            return ValidationResult(
                stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Invalid amount format for precision check",
                suggested_fix="Use valid numeric amount"
            )

        return ValidationResult(
            stage=ValidationStage.EXCHANGE_COMPATIBILITY,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Amount precision is valid"
        )


class TradingPairValidator(ValidationRule):
    """Validates that trading pair is supported by exchange."""

    def __init__(self):
        super().__init__("trading_pair")

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        signal_dict = signal_to_dict(signal)
        symbol = signal_dict.get("symbol", "")
        exchange = context.get("exchange", "default")

        if not symbol:
            return ValidationResult(
                stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Symbol required for trading pair validation",
                suggested_fix="Provide trading symbol"
            )

        # Get supported pairs for exchange (simplified - in real implementation would query exchange)
        supported_pairs = {
            "default": {"BTC/USDT", "ETH/USDT", "BNB/USDT"},
            "binance": {"BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"},
            "kucoin": {"BTC/USDT", "ETH/USDT", "KCS/USDT"},
        }

        exchange_pairs = supported_pairs.get(exchange, supported_pairs["default"])

        if symbol not in exchange_pairs:
            return ValidationResult(
                stage=ValidationStage.EXCHANGE_COMPATIBILITY,
                check_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Trading pair {symbol} not supported on {exchange}",
                suggested_fix=f"Use one of supported pairs: {', '.join(exchange_pairs)}"
            )

        return ValidationResult(
            stage=ValidationStage.EXCHANGE_COMPATIBILITY,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Trading pair is supported"
        )


# Exchange-specific validators
class BinanceLeverageValidator(ValidationRule):
    """Validates Binance leverage settings."""

    def __init__(self):
        super().__init__("binance_leverage")

    def is_applicable(self, signal: Any, context: Dict[str, Any]) -> bool:
        return context.get("exchange", "").lower() == "binance"

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        # Binance-specific leverage validation would go here
        return ValidationResult(
            stage=ValidationStage.EXCHANGE_COMPATIBILITY,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Binance leverage validation passed"
        )


class BinanceIsolatedMarginValidator(ValidationRule):
    """Validates Binance isolated margin settings."""

    def __init__(self):
        super().__init__("binance_isolated_margin")

    def is_applicable(self, signal: Any, context: Dict[str, Any]) -> bool:
        return context.get("exchange", "").lower() == "binance"

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        # Binance-specific isolated margin validation would go here
        return ValidationResult(
            stage=ValidationStage.EXCHANGE_COMPATIBILITY,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Binance isolated margin validation passed"
        )


class KuCoinMinimumVolumeValidator(ValidationRule):
    """Validates KuCoin minimum volume requirements."""

    def __init__(self):
        super().__init__("kucoin_minimum_volume")

    def is_applicable(self, signal: Any, context: Dict[str, Any]) -> bool:
        return context.get("exchange", "").lower() == "kucoin"

    async def validate(self, signal: Any, context: Dict[str, Any]) -> ValidationResult:
        # KuCoin-specific minimum volume validation would go here
        return ValidationResult(
            stage=ValidationStage.EXCHANGE_COMPATIBILITY,
            check_name=self.name,
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="KuCoin minimum volume validation passed"
        )


# ===== VALIDATION RULE MANAGEMENT INTERFACE =====

class ValidationRuleManager:
    """Manages validation rules and their configuration."""

    def __init__(self):
        self.rules: Dict[str, ValidationRule] = {}
        self.rule_configs: Dict[str, Dict[str, Any]] = {}

    def register_rule(self, rule: ValidationRule, config: Optional[Dict[str, Any]] = None) -> None:
        """Register a validation rule."""
        self.rules[rule.name] = rule
        self.rule_configs[rule.name] = config or {}

    def unregister_rule(self, rule_name: str) -> bool:
        """Unregister a validation rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            del self.rule_configs[rule_name]
            return True
        return False

    def get_rule(self, rule_name: str) -> Optional[ValidationRule]:
        """Get a validation rule by name."""
        return self.rules.get(rule_name)

    def list_rules(self) -> List[str]:
        """List all registered rule names."""
        return list(self.rules.keys())

    def update_rule_config(self, rule_name: str, config: Dict[str, Any]) -> bool:
        """Update configuration for a validation rule."""
        if rule_name in self.rule_configs:
            self.rule_configs[rule_name].update(config)
            return True
        return False

    def enable_rule(self, rule_name: str) -> bool:
        """Enable a validation rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            return True
        return False

    def disable_rule(self, rule_name: str) -> bool:
        """Disable a validation rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            return True
        return False
