"""
Comprehensive Order Validation Tests

Tests for the multi-stage validation pipeline including:
- Pre-trade validation (basic checks, market hours)
- Risk validation (integration with risk_manager)
- Exchange compatibility validation (exchange-specific rules)
- Validation rule management interface
"""

import asyncio
import pytest
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, time

from core.order_validation import (
    OrderValidationPipeline,
    ValidationReport,
    ValidationResult,
    ValidationStage,
    ValidationSeverity,
    ValidationRule,
    PreTradeValidator,
    RiskValidator,
    ExchangeCompatibilityValidator,
    BasicFieldValidator,
    AmountValidator,
    SymbolValidator,
    OrderTypeValidator,
    SignalTypeValidator,
    MarketHoursValidator,
    TimestampValidator,
    RiskManagerValidator,
    PositionSizeValidator,
    PortfolioRiskValidator,
    StopLossValidator,
    TakeProfitValidator,
    MinimumOrderSizeValidator,
    MaximumOrderSizeValidator,
    PricePrecisionValidator,
    AmountPrecisionValidator,
    TradingPairValidator,
    ValidationRuleManager,
)
from risk.risk_manager import RiskManager


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(
            stage=ValidationStage.PRE_TRADE,
            check_name="test_check",
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Test message",
            details={"key": "value"},
            suggested_fix="Fix it"
        )

        assert result.stage == ValidationStage.PRE_TRADE
        assert result.check_name == "test_check"
        assert result.passed is True
        assert result.severity == ValidationSeverity.ERROR
        assert result.message == "Test message"
        assert result.details == {"key": "value"}
        assert result.suggested_fix == "Fix it"


class TestValidationReport:
    """Test ValidationReport dataclass."""

    def test_validation_report_creation(self):
        """Test creating a validation report."""
        signal = {"symbol": "BTC/USDT"}
        report = ValidationReport(signal=signal, timestamp=123.45)

        assert report.signal == signal
        assert report.timestamp == 123.45
        assert report.overall_passed is True
        assert report.results == []
        assert report.error_count == 0
        assert report.warning_count == 0

    def test_add_result_updates_counts(self):
        """Test that adding results updates error/warning counts."""
        signal = {"symbol": "BTC/USDT"}
        report = ValidationReport(signal=signal, timestamp=123.45)

        # Add error result
        error_result = ValidationResult(
            stage=ValidationStage.PRE_TRADE,
            check_name="error_check",
            passed=False,
            severity=ValidationSeverity.ERROR,
            message="Error occurred"
        )
        report.add_result(error_result)

        # Add warning result
        warning_result = ValidationResult(
            stage=ValidationStage.RISK,
            check_name="warning_check",
            passed=False,
            severity=ValidationSeverity.WARNING,
            message="Warning occurred"
        )
        report.add_result(warning_result)

        # Add passing result
        pass_result = ValidationResult(
            stage=ValidationStage.EXCHANGE_COMPATIBILITY,
            check_name="pass_check",
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Passed"
        )
        report.add_result(pass_result)

        assert report.error_count == 1
        assert report.warning_count == 1
        assert report.overall_passed is False
        assert len(report.results) == 3

    def test_get_errors_returns_only_errors(self):
        """Test get_errors returns only error results."""
        signal = {"symbol": "BTC/USDT"}
        report = ValidationReport(signal=signal, timestamp=123.45)

        error_result = ValidationResult(
            stage=ValidationStage.PRE_TRADE,
            check_name="error_check",
            passed=False,
            severity=ValidationSeverity.ERROR,
            message="Error"
        )
        warning_result = ValidationResult(
            stage=ValidationStage.RISK,
            check_name="warning_check",
            passed=False,
            severity=ValidationSeverity.WARNING,
            message="Warning"
        )
        pass_result = ValidationResult(
            stage=ValidationStage.EXCHANGE_COMPATIBILITY,
            check_name="pass_check",
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Passed"
        )

        report.add_result(error_result)
        report.add_result(warning_result)
        report.add_result(pass_result)

        errors = report.get_errors()
        assert len(errors) == 1
        assert errors[0].check_name == "error_check"

    def test_get_warnings_returns_only_warnings(self):
        """Test get_warnings returns only warning results."""
        signal = {"symbol": "BTC/USDT"}
        report = ValidationReport(signal=signal, timestamp=123.45)

        error_result = ValidationResult(
            stage=ValidationStage.PRE_TRADE,
            check_name="error_check",
            passed=False,
            severity=ValidationSeverity.ERROR,
            message="Error"
        )
        warning_result = ValidationResult(
            stage=ValidationStage.RISK,
            check_name="warning_check",
            passed=False,
            severity=ValidationSeverity.WARNING,
            message="Warning"
        )
        pass_result = ValidationResult(
            stage=ValidationStage.EXCHANGE_COMPATIBILITY,
            check_name="pass_check",
            passed=True,
            severity=ValidationSeverity.ERROR,
            message="Passed"
        )

        report.add_result(error_result)
        report.add_result(warning_result)
        report.add_result(pass_result)

        warnings = report.get_warnings()
        assert len(warnings) == 1
        assert warnings[0].check_name == "warning_check"


class TestBasicFieldValidator:
    """Test BasicFieldValidator."""

    @pytest.fixture
    def validator(self):
        return BasicFieldValidator()

    def test_valid_signal_passes(self, validator):
        """Test that a valid signal passes validation."""
        signal = {
            "strategy_id": "test_strategy",
            "symbol": "BTC/USDT",
            "signal_type": "ENTRY_LONG",
            "order_type": "MARKET",
            "amount": "0.001"
        }

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is True
        assert "All required fields present" in result.message

    def test_missing_required_field_fails(self, validator):
        """Test that missing required field fails validation."""
        signal = {
            "strategy_id": "test_strategy",
            "symbol": "BTC/USDT",
            "signal_type": "ENTRY_LONG",
            # Missing order_type
            "amount": "0.001"
        }

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Missing required fields" in result.message
        assert "order_type" in result.message

    def test_none_required_field_fails(self, validator):
        """Test that None required field fails validation."""
        signal = {
            "strategy_id": "test_strategy",
            "symbol": "BTC/USDT",
            "signal_type": "ENTRY_LONG",
            "order_type": None,  # None value
            "amount": "0.001"
        }

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Missing required fields" in result.message


class TestAmountValidator:
    """Test AmountValidator."""

    @pytest.fixture
    def validator(self):
        return AmountValidator()

    def test_valid_amount_passes(self, validator):
        """Test that valid amount passes validation."""
        signal = {"amount": "0.001"}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is True
        assert "Amount validation passed" in result.message

    def test_missing_amount_fails(self, validator):
        """Test that missing amount fails validation."""
        signal = {}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Order amount is required" in result.message

    def test_zero_amount_fails(self, validator):
        """Test that zero amount fails validation."""
        signal = {"amount": "0"}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Order amount must be positive" in result.message

    def test_negative_amount_fails(self, validator):
        """Test that negative amount fails validation."""
        signal = {"amount": "-0.001"}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Order amount must be positive" in result.message

    def test_invalid_amount_format_fails(self, validator):
        """Test that invalid amount format fails validation."""
        signal = {"amount": "invalid"}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Invalid amount format" in result.message


class TestSymbolValidator:
    """Test SymbolValidator."""

    @pytest.fixture
    def validator(self):
        return SymbolValidator()

    def test_valid_symbol_passes(self, validator):
        """Test that valid symbol passes validation."""
        signal = {"symbol": "BTC/USDT"}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is True
        assert "Symbol format is valid" in result.message

    def test_missing_symbol_fails(self, validator):
        """Test that missing symbol fails validation."""
        signal = {}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Symbol is required" in result.message

    def test_empty_symbol_fails(self, validator):
        """Test that empty symbol fails validation."""
        signal = {"symbol": ""}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Symbol is required" in result.message

    def test_invalid_symbol_format_fails(self, validator):
        """Test that invalid symbol format fails validation."""
        signal = {"symbol": "BTCUSDT"}  # Missing slash

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Symbol must be in BASE/QUOTE format" in result.message

    def test_symbol_without_quote_fails(self, validator):
        """Test that symbol without quote currency fails validation."""
        signal = {"symbol": "BTC/"}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Symbol must have both base and quote currencies" in result.message


class TestOrderTypeValidator:
    """Test OrderTypeValidator."""

    @pytest.fixture
    def validator(self):
        return OrderTypeValidator()

    def test_valid_order_type_passes(self, validator):
        """Test that valid order type passes validation."""
        signal = {"order_type": "MARKET"}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is True
        assert "Order type is valid" in result.message

    def test_missing_order_type_fails(self, validator):
        """Test that missing order type fails validation."""
        signal = {}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Order type is required" in result.message

    def test_invalid_order_type_fails(self, validator):
        """Test that invalid order type fails validation."""
        signal = {"order_type": "INVALID"}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Invalid order type" in result.message


class TestSignalTypeValidator:
    """Test SignalTypeValidator."""

    @pytest.fixture
    def validator(self):
        return SignalTypeValidator()

    def test_valid_signal_type_passes(self, validator):
        """Test that valid signal type passes validation."""
        signal = {"signal_type": "ENTRY_LONG"}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is True
        assert "Signal type is valid" in result.message

    def test_missing_signal_type_fails(self, validator):
        """Test that missing signal type fails validation."""
        signal = {}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Signal type is required" in result.message

    def test_invalid_signal_type_fails(self, validator):
        """Test that invalid signal type fails validation."""
        signal = {"signal_type": "INVALID"}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Invalid signal type" in result.message


class TestMarketHoursValidator:
    """Test MarketHoursValidator."""

    @pytest.fixture
    def validator(self):
        market_hours_config = {
            "enabled": True,
            "open": "09:00",
            "close": "17:00",
            "timezone": "UTC"
        }
        return MarketHoursValidator(market_hours_config)

    def test_within_market_hours_passes(self, validator):
        """Test that signal within market hours passes validation."""
        # Mock current time to be within market hours
        with patch('core.order_validation.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)  # Noon

            signal = {"symbol": "BTC/USDT"}
            result = asyncio.run(validator.validate(signal, {}))

            assert result.passed is True
            assert "Within market hours" in result.message

    def test_outside_market_hours_fails(self, validator):
        """Test that signal outside market hours fails validation."""
        # Mock current time to be outside market hours
        with patch('core.order_validation.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 20, 0, 0)  # 8 PM

            signal = {"symbol": "BTC/USDT"}
            result = asyncio.run(validator.validate(signal, {}))

            assert result.passed is False
            assert "Outside market hours" in result.message

    def test_disabled_validator_passes(self):
        """Test that disabled market hours validator always passes."""
        market_hours_config = {"enabled": False}
        validator = MarketHoursValidator(market_hours_config)

        signal = {"symbol": "BTC/USDT"}
        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is True
        assert "Market hours validation disabled" in result.message


class TestTimestampValidator:
    """Test TimestampValidator."""

    @pytest.fixture
    def validator(self):
        return TimestampValidator()

    def test_missing_timestamp_warns(self, validator):
        """Test that missing timestamp generates warning."""
        signal = {}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert result.severity == ValidationSeverity.WARNING
        assert "Timestamp not provided" in result.message

    def test_future_timestamp_warns(self, validator):
        """Test that future timestamp generates warning."""
        # Mock current time
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0  # Current time in seconds

            # Timestamp far in the future (more than 5 minutes)
            future_timestamp = 2000000  # 2,000,000 milliseconds (clearly more than 5 min future from 1,000,000)
            signal = {"timestamp": str(future_timestamp)}

            result = asyncio.run(validator.validate(signal, {}))

            assert result.passed is False
            assert result.severity == ValidationSeverity.WARNING
            assert "timestamp is in the future" in result.message

    def test_old_timestamp_warns(self, validator):
        """Test that very old timestamp generates warning."""
        # Mock current time
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0  # Current time in seconds

            # Timestamp from over 24 hours ago (very old)
            old_timestamp = -86400001  # Just over 24 hours ago in milliseconds
            signal = {"timestamp": str(old_timestamp)}

            result = asyncio.run(validator.validate(signal, {}))

            assert result.passed is False
            assert result.severity == ValidationSeverity.WARNING
            assert "timestamp is too old" in result.message

    def test_valid_timestamp_passes(self, validator):
        """Test that valid recent timestamp passes validation."""
        # Mock current time
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0  # Current time in seconds

            # Valid timestamp (close to current time) - should be in milliseconds
            valid_timestamp = 1000000  # 1000 seconds in milliseconds (same as mock time)
            signal = {"timestamp": str(valid_timestamp)}

            result = asyncio.run(validator.validate(signal, {}))

            assert result.passed is True
            assert "Timestamp validation passed" in result.message


class TestRiskManagerValidator:
    """Test RiskManagerValidator."""

    @pytest.fixture
    def mock_risk_manager(self):
        """Create a mock risk manager."""
        risk_manager = Mock(spec=RiskManager)
        risk_manager.evaluate_signal = AsyncMock(return_value=True)
        return risk_manager

    @pytest.fixture
    def validator(self, mock_risk_manager):
        return RiskManagerValidator(mock_risk_manager)

    def test_risk_validation_passes(self, validator, mock_risk_manager):
        """Test that passing risk validation succeeds."""
        mock_risk_manager.evaluate_signal.return_value = True

        signal = {"symbol": "BTC/USDT"}
        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is True
        assert "Risk management validation passed" in result.message

    def test_risk_validation_fails(self, validator, mock_risk_manager):
        """Test that failing risk validation fails."""
        mock_risk_manager.evaluate_signal.return_value = False

        signal = {"symbol": "BTC/USDT"}
        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Signal failed risk management evaluation" in result.message


class TestPositionSizeValidator:
    """Test PositionSizeValidator."""

    @pytest.fixture
    def mock_risk_manager(self):
        """Create a mock risk manager."""
        risk_manager = Mock(spec=RiskManager)
        risk_manager.max_position_size = Decimal("0.1")
        risk_manager._get_current_balance = AsyncMock(return_value=Decimal("10000"))
        return risk_manager

    @pytest.fixture
    def validator(self, mock_risk_manager):
        return PositionSizeValidator(mock_risk_manager)

    def test_position_size_within_limit_passes(self, validator, mock_risk_manager):
        """Test that position size within limit passes."""
        mock_risk_manager._get_current_balance.return_value = Decimal("10000")

        signal = {"amount": "100"}  # 1% of balance
        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is True
        assert "Position size within risk limits" in result.message

    def test_position_size_over_limit_fails(self, validator, mock_risk_manager):
        """Test that position size over limit fails."""
        mock_risk_manager._get_current_balance.return_value = Decimal("10000")

        signal = {"amount": "2000"}  # 20% of balance, over 10% limit
        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "exceeds maximum" in result.message

    def test_missing_amount_fails(self, validator, mock_risk_manager):
        """Test that missing amount fails validation."""
        signal = {}
        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Amount required for position size validation" in result.message


class TestStopLossValidator:
    """Test StopLossValidator."""

    @pytest.fixture
    def validator(self):
        return StopLossValidator()

    def test_stop_order_without_stop_loss_fails(self, validator):
        """Test that stop order without stop loss fails."""
        signal = {"order_type": "STOP"}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Stop orders must include stop_loss" in result.message

    def test_stop_limit_order_without_stop_loss_fails(self, validator):
        """Test that stop limit order without stop loss fails."""
        signal = {"order_type": "STOP_LIMIT"}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is False
        assert "Stop orders must include stop_loss" in result.message

    def test_market_order_without_stop_loss_passes(self, validator):
        """Test that market order without stop loss passes."""
        signal = {"order_type": "MARKET"}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is True
        assert "Stop loss validation passed" in result.message

    def test_stop_order_with_stop_loss_passes(self, validator):
        """Test that stop order with stop loss passes."""
        signal = {"order_type": "STOP", "stop_loss": "50000"}

        result = asyncio.run(validator.validate(signal, {}))

        assert result.passed is True
        assert "Stop loss validation passed" in result.message


class TestMinimumOrderSizeValidator:
    """Test MinimumOrderSizeValidator."""

    @pytest.fixture
    def validator(self):
        return MinimumOrderSizeValidator()

    def test_order_above_minimum_passes(self, validator):
        """Test that order above minimum size passes."""
        signal = {"amount": "0.002"}
        context = {"exchange": "binance"}

        result = asyncio.run(validator.validate(signal, context))

        assert result.passed is True
        assert "Order size meets minimum requirements" in result.message

    def test_order_below_minimum_fails(self, validator):
        """Test that order below minimum size fails."""
        signal = {"amount": "0.00005"}  # Below Binance minimum for BTC
        context = {"exchange": "binance"}

        result = asyncio.run(validator.validate(signal, context))

        assert result.passed is False
        assert "below minimum" in result.message

    def test_missing_amount_fails(self, validator):
        """Test that missing amount fails validation."""
        signal = {}
        context = {"exchange": "binance"}

        result = asyncio.run(validator.validate(signal, context))

        assert result.passed is False
        assert "Amount required for minimum order size check" in result.message


class TestMaximumOrderSizeValidator:
    """Test MaximumOrderSizeValidator."""

    @pytest.fixture
    def validator(self):
        return MaximumOrderSizeValidator()

    def test_order_below_maximum_passes(self, validator):
        """Test that order below maximum size passes."""
        signal = {"amount": "50"}
        context = {"exchange": "binance"}

        result = asyncio.run(validator.validate(signal, context))

        assert result.passed is True
        assert "Order size within maximum limits" in result.message

    def test_order_above_maximum_fails(self, validator):
        """Test that order above maximum size fails."""
        signal = {"amount": "2000"}  # Above Binance maximum for BTC
        context = {"exchange": "binance"}

        result = asyncio.run(validator.validate(signal, context))

        assert result.passed is False
        assert "exceeds maximum" in result.message


class TestPricePrecisionValidator:
    """Test PricePrecisionValidator."""

    @pytest.fixture
    def validator(self):
        return PricePrecisionValidator()

    def test_price_within_precision_passes(self, validator):
        """Test that price within precision passes."""
        signal = {"price": "50000.12"}
        context = {"exchange": "binance"}

        result = asyncio.run(validator.validate(signal, context))

        assert result.passed is True
        assert "Price precision is valid" in result.message

    def test_price_exceeds_precision_fails(self, validator):
        """Test that price exceeding precision fails."""
        signal = {"price": "50000.12345"}  # More than 5 decimal places for Binance BTC
        context = {"exchange": "binance"}

        result = asyncio.run(validator.validate(signal, context))

        assert result.passed is False
        assert "precision" in result.message.lower()

    def test_missing_price_passes(self, validator):
        """Test that missing price passes (no precision to validate)."""
        signal = {}
        context = {"exchange": "binance"}

        result = asyncio.run(validator.validate(signal, context))

        assert result.passed is True
        assert "No price to validate precision for" in result.message


class TestTradingPairValidator:
    """Test TradingPairValidator."""

    @pytest.fixture
    def validator(self):
        return TradingPairValidator()

    def test_supported_pair_passes(self, validator):
        """Test that supported trading pair passes."""
        signal = {"symbol": "BTC/USDT"}
        context = {"exchange": "binance"}

        result = asyncio.run(validator.validate(signal, context))

        assert result.passed is True
        assert "Trading pair is supported" in result.message

    def test_unsupported_pair_fails(self, validator):
        """Test that unsupported trading pair fails."""
        signal = {"symbol": "UNKNOWN/USDT"}
        context = {"exchange": "binance"}

        result = asyncio.run(validator.validate(signal, context))

        assert result.passed is False
        assert "not supported" in result.message

    def test_missing_symbol_fails(self, validator):
        """Test that missing symbol fails validation."""
        signal = {}
        context = {"exchange": "binance"}

        result = asyncio.run(validator.validate(signal, context))

        assert result.passed is False
        assert "Symbol required for trading pair validation" in result.message


class TestOrderValidationPipeline:
    """Test OrderValidationPipeline."""

    @pytest.fixture
    def mock_risk_manager(self):
        """Create a mock risk manager."""
        risk_manager = Mock(spec=RiskManager)
        risk_manager.evaluate_signal = AsyncMock(return_value=True)
        risk_manager.max_position_size = Decimal("0.1")
        risk_manager._get_current_balance = AsyncMock(return_value=Decimal("10000"))
        risk_manager.today_pnl = Decimal("0")
        risk_manager.today_start_balance = Decimal("10000")
        return risk_manager

    @pytest.fixture
    def pipeline(self, mock_risk_manager):
        """Create a validation pipeline."""
        config = {
            "fail_fast": False,
            "validation_timeout": 5.0,
            "enable_circuit_breaker": False,
            "market_hours": {"enabled": False}
        }
        return OrderValidationPipeline(config, mock_risk_manager)

    def test_valid_signal_passes_all_stages(self, pipeline):
        """Test that a valid signal passes all validation stages."""
        # Mock current time for timestamp validation
        with patch('asyncio.AbstractEventLoop.time') as mock_time:
            mock_time.return_value = 1000.0  # Current time in seconds

            signal = {
                "strategy_id": "test_strategy",
                "symbol": "BTC/USDT",
                "signal_type": "ENTRY_LONG",
                "order_type": "MARKET",
                "amount": "0.001",
                "timestamp": "999000"  # Valid timestamp
            }

            report = asyncio.run(pipeline.validate_order(signal))

            assert report.overall_passed is True
            assert report.error_count == 0
            assert len(report.results) > 0

    def test_invalid_signal_fails_validation(self, pipeline):
        """Test that an invalid signal fails validation."""
        signal = {
            "strategy_id": "test_strategy",
            "symbol": "BTC/USDT",
            "signal_type": "INVALID_TYPE",  # Invalid signal type
            "order_type": "MARKET",
            "amount": "0.001"
        }

        report = asyncio.run(pipeline.validate_order(signal))

        assert report.overall_passed is False
        assert report.error_count > 0

    def test_pipeline_timeout_handling(self, pipeline):
        """Test that pipeline handles timeouts properly."""
        # Set very short timeout
        pipeline.timeout_seconds = 0.001

        signal = {
            "strategy_id": "test_strategy",
            "symbol": "BTC/USDT",
            "signal_type": "ENTRY_LONG",
            "order_type": "MARKET",
            "amount": "0.001"
        }

        report = asyncio.run(pipeline.validate_order(signal))

        # Should still complete but may have timeout errors
        assert isinstance(report, ValidationReport)

    def test_pipeline_performance_tracking(self, pipeline):
        """Test that pipeline tracks performance metrics."""
        signal = {
            "strategy_id": "test_strategy",
            "symbol": "BTC/USDT",
            "signal_type": "ENTRY_LONG",
            "order_type": "MARKET",
            "amount": "0.001"
        }

        # Run validation multiple times
        for _ in range(3):
            asyncio.run(pipeline.validate_order(signal))

        stats = pipeline.get_performance_stats()

        assert "average_time_ms" in stats
        assert "total_validations" in stats
        assert stats["total_validations"] == 3

    def test_circuit_breaker_activates(self):
        """Test that circuit breaker activates after consecutive failures."""
        config = {
            "fail_fast": False,
            "validation_timeout": 5.0,
            "enable_circuit_breaker": True,
            "max_validation_failures": 2,
            "market_hours": {"enabled": False}
        }
        pipeline = OrderValidationPipeline(config)

        # Force failures to trigger circuit breaker
        pipeline._validation_failures = 3

        signal = {
            "strategy_id": "test_strategy",
            "symbol": "BTC/USDT",
            "signal_type": "ENTRY_LONG",
            "order_type": "MARKET",
            "amount": "0.001"
        }

        report = asyncio.run(pipeline.validate_order(signal))

        assert report.overall_passed is False
        assert len(report.results) == 1
        assert "circuit_breaker" in report.results[0].check_name


class TestValidationRuleManager:
    """Test ValidationRuleManager."""

    @pytest.fixture
    def rule_manager(self):
        return ValidationRuleManager()

    def test_register_and_get_rule(self, rule_manager):
        """Test registering and getting a validation rule."""
        rule = BasicFieldValidator()
        rule_manager.register_rule(rule)

        retrieved = rule_manager.get_rule("basic_fields")
        assert retrieved is rule

    def test_unregister_rule(self, rule_manager):
        """Test unregistering a validation rule."""
        rule = BasicFieldValidator()
        rule_manager.register_rule(rule)

        assert rule_manager.get_rule("basic_fields") is rule

        result = rule_manager.unregister_rule("basic_fields")
        assert result is True
        assert rule_manager.get_rule("basic_fields") is None

    def test_list_rules(self, rule_manager):
        """Test listing all registered rules."""
        rule1 = BasicFieldValidator()
        rule2 = AmountValidator()

        rule_manager.register_rule(rule1)
        rule_manager.register_rule(rule2)

        rules = rule_manager.list_rules()
        assert "basic_fields" in rules
        assert "amount_validation" in rules
        assert len(rules) == 2

    def test_update_rule_config(self, rule_manager):
        """Test updating rule configuration."""
        rule = BasicFieldValidator()
        rule_manager.register_rule(rule, {"enabled": True})

        result = rule_manager.update_rule_config("basic_fields", {"enabled": False})
        assert result is True
        assert rule_manager.rule_configs["basic_fields"]["enabled"] is False

    def test_enable_disable_rule(self, rule_manager):
        """Test enabling and disabling rules."""
        rule = BasicFieldValidator()
        rule_manager.register_rule(rule)

        # Initially enabled
        assert rule.enabled is True

        # Disable
        result = rule_manager.disable_rule("basic_fields")
        assert result is True
        assert rule.enabled is False

        # Enable
        result = rule_manager.enable_rule("basic_fields")
        assert result is True
        assert rule.enabled is True


# Integration tests
class TestComprehensiveValidationIntegration:
    """Integration tests for the complete validation pipeline."""

    @pytest.fixture
    def full_pipeline(self):
        """Create a fully configured validation pipeline."""
        config = {
            "fail_fast": False,
            "validation_timeout": 10.0,
            "enable_circuit_breaker": False,
            "market_hours": {"enabled": False}
        }

        # Mock risk manager
        risk_manager = Mock(spec=RiskManager)
        risk_manager.evaluate_signal = AsyncMock(return_value=True)
        risk_manager.max_position_size = Decimal("0.1")
        risk_manager._get_current_balance = AsyncMock(return_value=Decimal("10000"))
        risk_manager.today_pnl = Decimal("0")
        risk_manager.today_start_balance = Decimal("10000")

        return OrderValidationPipeline(config, risk_manager)

    def test_complete_valid_order_validation(self, full_pipeline):
        """Test complete validation of a valid order."""
        # Mock current time for timestamp validation
        with patch('asyncio.AbstractEventLoop.time') as mock_time:
            mock_time.return_value = 1000.0  # Current time in seconds

            signal = {
                "strategy_id": "momentum_strategy",
                "symbol": "BTC/USDT",
                "signal_type": "ENTRY_LONG",
                "signal_strength": "STRONG",
                "order_type": "MARKET",
                "amount": "0.001",
                "price": "50000.00",
                "stop_loss": "49000.00",
                "take_profit": "52000.00",
                "timestamp": "999000"  # Valid timestamp
            }

            context = {
                "exchange": "binance",
                "market_data": None,
                "max_concurrent_positions": 5
            }

            report = asyncio.run(full_pipeline.validate_order(signal, context))

            assert report.overall_passed is True
            assert report.error_count == 0

            # Should have results from all validation stages
            stages = {result.stage for result in report.results}
            assert ValidationStage.PRE_TRADE in stages
            assert ValidationStage.RISK in stages
            assert ValidationStage.EXCHANGE_COMPATIBILITY in stages

    def test_validation_with_multiple_errors(self, full_pipeline):
        """Test validation with multiple validation errors."""
        signal = {
            "strategy_id": "test_strategy",
            "symbol": "INVALID",  # Invalid symbol
            "signal_type": "INVALID_TYPE",  # Invalid signal type
            "order_type": "INVALID_ORDER",  # Invalid order type
            "amount": "-0.001",  # Negative amount
        }

        report = asyncio.run(full_pipeline.validate_order(signal))

        assert report.overall_passed is False
        assert report.error_count > 1

    def test_validation_with_warnings(self, full_pipeline):
        """Test validation that passes but generates warnings."""
        signal = {
            "strategy_id": "test_strategy",
            "symbol": "BTC/USDT",
            "signal_type": "ENTRY_LONG",
            "order_type": "MARKET",
            "amount": "0.001"
            # Missing timestamp - should generate warning
        }

        report = asyncio.run(full_pipeline.validate_order(signal))

        assert report.overall_passed is True  # Errors only
        assert report.warning_count > 0

    def test_exchange_specific_validation(self, full_pipeline):
        """Test exchange-specific validation rules."""
        # Mock current time for timestamp validation
        with patch('asyncio.AbstractEventLoop.time') as mock_time:
            mock_time.return_value = 1000.0  # Current time in seconds

            signal = {
                "strategy_id": "test_strategy",
                "symbol": "BTC/USDT",
                "signal_type": "ENTRY_LONG",
                "order_type": "MARKET",
                "amount": "0.001",
                "timestamp": "999000"  # Valid timestamp
            }

            # Test with Binance exchange
            context = {"exchange": "binance"}
            report = asyncio.run(full_pipeline.validate_order(signal, context))

            assert report.overall_passed is True

            # Test with KuCoin exchange
            context = {"exchange": "kucoin"}
            report = asyncio.run(full_pipeline.validate_order(signal, context))

            assert report.overall_passed is True

    def test_validation_pipeline_resets_circuit_breaker(self, full_pipeline):
        """Test that successful validations reset circuit breaker."""
        # Initially no failures
        assert full_pipeline._validation_failures == 0

        # Mock current time for timestamp validation
        with patch('asyncio.AbstractEventLoop.time') as mock_time:
            mock_time.return_value = 1000.0  # Current time in seconds

            # Run a successful validation
            signal = {
                "strategy_id": "test_strategy",
                "symbol": "BTC/USDT",
                "signal_type": "ENTRY_LONG",
                "order_type": "MARKET",
                "amount": "0.001",
                "timestamp": "999000"  # Valid timestamp
            }

            report = asyncio.run(full_pipeline.validate_order(signal))
            assert report.overall_passed is True

            # Failures should still be 0
            assert full_pipeline._validation_failures == 0

    def test_validation_timeout_protection(self):
        """Test that validation is protected against timeouts."""
        config = {
            "fail_fast": False,
            "validation_timeout": 0.001,  # Very short timeout
            "enable_circuit_breaker": False,
            "market_hours": {"enabled": False}
        }

        # Create a slow validator for testing
        class SlowValidator(ValidationRule):
            def __init__(self):
                super().__init__("slow_validator")

            async def validate(self, signal, context):
                await asyncio.sleep(0.01)  # Sleep longer than timeout
                return ValidationResult(
                    stage=ValidationStage.PRE_TRADE,
                    check_name=self.name,
                    passed=True,
                    severity=ValidationSeverity.ERROR,
                    message="Should not reach here"
                )

        # Create pipeline with slow validator
        pipeline = OrderValidationPipeline(config)
        pipeline.pre_trade_validator.rules = [SlowValidator()]

        signal = {
            "strategy_id": "test_strategy",
            "symbol": "BTC/USDT",
            "signal_type": "ENTRY_LONG",
            "order_type": "MARKET",
            "amount": "0.001"
        }

        report = asyncio.run(pipeline.validate_order(signal))

        # Should have timeout error
        assert report.overall_passed is False
        assert any("timeout" in result.check_name for result in report.results)


if __name__ == "__main__":
    pytest.main([__file__])
