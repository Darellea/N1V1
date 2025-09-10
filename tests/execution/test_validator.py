"""
Tests for Execution Validator

Comprehensive tests for pre-trade validation functionality.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import patch

from core.execution.validator import ExecutionValidator
from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types.order_types import OrderType


class TestExecutionValidator:
    """Test cases for ExecutionValidator."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            'enabled': True,
            'check_balance': True,
            'check_slippage': True,
            'max_slippage_pct': 0.02,
            'min_order_size': 0.000001,
            'max_order_size': 1000000,
            'tick_size': 0.00000001,
            'lot_size': 0.00000001,
            'tradable_symbols': ['BTC/USDT', 'ETH/USDT']
        }

    @pytest.fixture
    def validator(self, config):
        """Execution validator instance."""
        return ExecutionValidator(config)

    @pytest.fixture
    def valid_signal(self):
        """Valid trading signal."""
        return TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.LIMIT,
            amount=Decimal("1000"),
            price=Decimal("50000"),
            current_price=Decimal("50000"),
            stop_loss=Decimal("49000"),
            timestamp=datetime.now()
        )

    @pytest.fixture
    def market_context(self):
        """Market context for testing."""
        return {
            'market_price': Decimal("50000"),
            'account_balance': Decimal("100000"),
            'expected_slippage_pct': Decimal("0.001"),
            'current_hour': 12
        }

    def test_initialization(self, validator, config):
        """Test validator initialization."""
        assert validator.enabled == config['enabled']
        assert validator.check_balance == config['check_balance']
        assert validator.check_slippage == config['check_slippage']
        assert validator.max_slippage_pct == Decimal(str(config['max_slippage_pct']))
        assert validator.min_order_size == Decimal(str(config['min_order_size']))
        assert validator.max_order_size == Decimal(str(config['max_order_size']))

    @pytest.mark.asyncio
    async def test_validate_valid_signal(self, validator, valid_signal, market_context):
        """Test validation of a valid signal."""
        result = await validator.validate_signal(valid_signal, market_context)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_disabled_validator(self, validator, valid_signal, market_context):
        """Test validation when validator is disabled."""
        validator.enabled = False
        result = await validator.validate_signal(valid_signal, market_context)
        assert result is True

    def test_validate_basic_signal_missing_symbol(self, validator):
        """Test validation of signal with missing symbol."""
        invalid_signal = TradingSignal(
            strategy_id="test",
            symbol="",  # Missing symbol
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1000"),
            timestamp=datetime.now()
        )

        with patch.object(validator, '_log_validation_error') as mock_log:
            result = validator._validate_basic_signal(invalid_signal)
            assert result is False
            mock_log.assert_called_once()
            assert "missing_symbol" in mock_log.call_args[0]

    def test_validate_basic_signal_invalid_amount(self, validator):
        """Test validation of signal with invalid amount."""
        invalid_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("-100"),  # Invalid amount
            timestamp=datetime.now()
        )

        with patch.object(validator, '_log_validation_error') as mock_log:
            result = validator._validate_basic_signal(invalid_signal)
            assert result is False
            mock_log.assert_called_once()
            assert "invalid_amount" in mock_log.call_args[0]

    def test_validate_basic_signal_missing_signal_type(self, validator):
        """Test validation of signal with missing signal type."""
        invalid_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=None,  # Missing signal type
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1000"),
            timestamp=datetime.now()
        )

        with patch.object(validator, '_log_validation_error') as mock_log:
            result = validator._validate_basic_signal(invalid_signal)
            assert result is False
            mock_log.assert_called_once()
            assert "missing_signal_type" in mock_log.call_args[0]

    def test_validate_order_size_too_small(self, validator):
        """Test validation of order size that's too small."""
        small_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("0.0000001"),  # Below minimum
            timestamp=datetime.now()
        )

        with patch.object(validator, '_log_validation_error') as mock_log:
            result = validator._validate_order_size(small_signal)
            assert result is False
            mock_log.assert_called_once()
            assert "order_too_small" in mock_log.call_args[0]

    def test_validate_order_size_too_large(self, validator):
        """Test validation of order size that's too large."""
        large_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("2000000"),  # Above maximum
            timestamp=datetime.now()
        )

        with patch.object(validator, '_log_validation_error') as mock_log:
            result = validator._validate_order_size(large_signal)
            assert result is False
            mock_log.assert_called_once()
            assert "order_too_large" in mock_log.call_args[0]

    def test_validate_order_size_invalid_lot_size(self, validator):
        """Test validation of order size with invalid lot size."""
        # Set lot size to 0.1
        validator.lot_size = Decimal("0.1")

        invalid_lot_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.55"),  # Not multiple of 0.1
            timestamp=datetime.now()
        )

        with patch.object(validator, '_log_validation_error') as mock_log:
            result = validator._validate_order_size(invalid_lot_signal)
            assert result is False
            mock_log.assert_called_once()
            assert "invalid_lot_size" in mock_log.call_args[0]

    def test_validate_order_size_valid_lot_size(self, validator):
        """Test validation of order size with valid lot size."""
        # Set lot size to 0.1
        validator.lot_size = Decimal("0.1")

        valid_lot_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.5"),  # Multiple of 0.1
            timestamp=datetime.now()
        )

        result = validator._validate_order_size(valid_lot_signal)
        assert result is True

    def test_validate_order_size_zero_lot_size(self, validator):
        """Test validation of order size with zero lot size."""
        # Set lot size to 0
        validator.lot_size = Decimal("0")

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.5"),
            timestamp=datetime.now()
        )

        with patch.object(validator, '_log_validation_error') as mock_log:
            result = validator._validate_order_size(signal)
            assert result is False
            mock_log.assert_called_once()
            assert "invalid_lot_size_config" in mock_log.call_args[0]

    def test_validate_price_limit_without_price(self, validator):
        """Test validation of limit order without price."""
        limit_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.LIMIT,
            amount=Decimal("1000"),
            price=None,  # Missing price for limit order
            timestamp=datetime.now()
        )

        with patch.object(validator, '_log_validation_error') as mock_log:
            result = validator._validate_price(limit_signal, {})
            assert result is False
            mock_log.assert_called_once()
            assert "missing_limit_price" in mock_log.call_args[0]

    def test_validate_price_invalid_tick_size(self, validator):
        """Test validation of price with invalid tick size."""
        # Set tick size to 0.01
        validator.tick_size = Decimal("0.01")

        invalid_tick_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.LIMIT,
            amount=Decimal("1000"),
            price=Decimal("50000.005"),  # Not multiple of 0.01
            timestamp=datetime.now()
        )

        with patch.object(validator, '_log_validation_error') as mock_log:
            result = validator._validate_price(invalid_tick_signal, {})
            assert result is False
            mock_log.assert_called_once()
            assert "invalid_tick_size" in mock_log.call_args[0]

    def test_validate_price_too_far_from_market(self, validator):
        """Test validation of price too far from market price."""
        far_price_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.LIMIT,
            amount=Decimal("1000"),
            price=Decimal("75000"),  # 50% above market
            timestamp=datetime.now()
        )

        context = {'market_price': Decimal("50000")}

        with patch.object(validator, '_log_validation_error') as mock_log:
            result = validator._validate_price(far_price_signal, context)
            assert result is False
            mock_log.assert_called_once()
            assert "price_too_far_from_market" in mock_log.call_args[0]

    @pytest.mark.asyncio
    async def test_validate_balance_insufficient(self, validator, valid_signal):
        """Test validation of insufficient account balance."""
        context = {
            'account_balance': Decimal("1000"),  # Insufficient balance
            'market_price': Decimal("50000")
        }

        with patch.object(validator, '_log_validation_error') as mock_log:
            result = await validator._validate_balance(valid_signal, context)
            assert result is False
            mock_log.assert_called_once()
            assert "insufficient_balance" in mock_log.call_args[0]

    @pytest.mark.asyncio
    async def test_validate_balance_sufficient(self, validator, valid_signal):
        """Test validation of sufficient account balance."""
        context = {
            'account_balance': Decimal("100000"),  # Sufficient balance
            'market_price': Decimal("50000")
        }

        result = await validator._validate_balance(valid_signal, context)
        assert result is True

    def test_validate_slippage_excessive(self, validator):
        """Test validation of excessive slippage."""
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1000"),
            timestamp=datetime.now()
        )

        context = {
            'expected_slippage_pct': Decimal("0.05")  # 5% slippage, above max 2%
        }

        with patch.object(validator, '_log_validation_error') as mock_log:
            result = validator._validate_slippage(signal, context)
            assert result is False
            mock_log.assert_called_once()
            assert "excessive_slippage" in mock_log.call_args[0]

    def test_validate_slippage_acceptable(self, validator):
        """Test validation of acceptable slippage."""
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1000"),
            timestamp=datetime.now()
        )

        context = {
            'expected_slippage_pct': Decimal("0.01")  # 1% slippage, within max 2%
        }

        result = validator._validate_slippage(signal, context)
        assert result is True

    def test_validate_exchange_constraints_tradable_symbol(self, validator):
        """Test validation of tradable symbol."""
        signal = TradingSignal(
            strategy_id="test",
            symbol="ADA/USDT",  # Not in tradable symbols
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1000"),
            timestamp=datetime.now()
        )

        with patch.object(validator, '_log_validation_error') as mock_log:
            result = validator._validate_exchange_constraints(signal)
            assert result is False
            mock_log.assert_called_once()
            assert "symbol_not_tradable" in mock_log.call_args[0]

    def test_validate_exchange_constraints_trading_hours(self, validator):
        """Test validation of trading hours."""
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1000"),
            timestamp=datetime.now()
        )

        context = {'current_hour': 2}  # Outside trading hours

        with patch.object(validator, '_log_validation_error') as mock_log:
            result = validator._validate_exchange_constraints(signal, context)
            assert result is False
            mock_log.assert_called_once()
            assert "outside_trading_hours" in mock_log.call_args[0]

    def test_validate_exchange_constraints_trading_hours_within_range(self, validator):
        """Test validation of trading hours within allowed range."""
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1000"),
            timestamp=datetime.now()
        )

        context = {'current_hour': 12}  # Within trading hours (9-16)

        result = validator._validate_exchange_constraints(signal, context)
        assert result is True

    def test_validate_exchange_constraints_custom_trading_hours(self, validator):
        """Test validation with custom trading hours configuration."""
        # Set custom trading hours (8 AM to 6 PM)
        validator.config['trading_hours'] = (8, 18)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1000"),
            timestamp=datetime.now()
        )

        # Test hour within custom range
        context_within = {'current_hour': 14}
        result = validator._validate_exchange_constraints(signal, context_within)
        assert result is True

        # Test hour outside custom range
        context_outside = {'current_hour': 7}
        with patch.object(validator, '_log_validation_error') as mock_log:
            result = validator._validate_exchange_constraints(signal, context_outside)
            assert result is False
            mock_log.assert_called_once()
            assert "outside_trading_hours" in mock_log.call_args[0]
            assert "8-18" in mock_log.call_args[0][2]  # Check that custom hours are in the message

    def test_get_validation_rules(self, validator):
        """Test getting validation rules."""
        rules = validator.get_validation_rules()

        assert isinstance(rules, dict)
        assert 'enabled' in rules
        assert 'check_balance' in rules
        assert 'check_slippage' in rules
        assert 'max_slippage_pct' in rules
        assert 'min_order_size' in rules
        assert 'max_order_size' in rules
        assert 'tick_size' in rules
        assert 'lot_size' in rules

    def test_update_validation_rules(self, validator):
        """Test updating validation rules."""
        new_rules = {
            'enabled': False,
            'max_slippage_pct': 0.05,
            'min_order_size': 0.001
        }

        validator.update_validation_rules(new_rules)

        assert validator.enabled is False
        assert validator.max_slippage_pct == Decimal("0.05")
        assert validator.min_order_size == Decimal("0.001")

    @pytest.mark.asyncio
    async def test_validation_error_logging(self, validator, valid_signal, market_context):
        """Test that validation errors are properly logged."""
        # Create invalid signal
        invalid_signal = valid_signal.copy()
        invalid_signal.amount = Decimal("-100")

        with patch('core.execution.validator.trade_logger') as mock_trade_logger:
            result = await validator.validate_signal(invalid_signal, market_context)

            assert result is False
            mock_trade_logger.performance.assert_called()

    def test_validation_disabled_checks(self, validator):
        """Test that disabled checks are skipped."""
        validator.check_balance = False
        validator.check_slippage = False

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1000"),
            timestamp=datetime.now()
        )

        # These should not be called when checks are disabled
        with patch.object(validator, '_validate_balance') as mock_balance:
            with patch.object(validator, '_validate_slippage') as mock_slippage:
                result = validator._validate_basic_signal(signal)  # Only basic validation

                # Balance and slippage validation should not be called
                mock_balance.assert_not_called()
                mock_slippage.assert_not_called()


class TestValidationEdgeCases:
    """Test edge cases for validation."""

    @pytest.fixture
    def validator(self):
        """Execution validator instance."""
        return ExecutionValidator()

    def test_zero_amount_validation(self, validator):
        """Test validation of zero amount."""
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("0"),
            timestamp=datetime.now()
        )

        with patch.object(validator, '_log_validation_error') as mock_log:
            result = validator._validate_basic_signal(signal)
            assert result is False
            mock_log.assert_called_once()

    def test_minimum_valid_amount(self, validator):
        """Test validation of minimum valid amount."""
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("0.000001"),  # Exactly minimum
            timestamp=datetime.now()
        )

        result = validator._validate_order_size(signal)
        assert result is True

    def test_maximum_valid_amount(self, validator):
        """Test validation of maximum valid amount."""
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1000000"),  # Exactly maximum
            timestamp=datetime.now()
        )

        result = validator._validate_order_size(signal)
        assert result is True

    def test_price_validation_without_context(self, validator):
        """Test price validation without market context."""
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.LIMIT,
            amount=Decimal("1000"),
            price=Decimal("50000"),
            timestamp=datetime.now()
        )

        # Should pass without market price context
        result = validator._validate_price(signal, {})
        assert result is True

    @pytest.mark.asyncio
    async def test_balance_validation_without_price(self, validator):
        """Test balance validation when no price is available."""
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1000"),
            timestamp=datetime.now()
        )

        context = {
            'account_balance': Decimal("1000"),
            'market_price': Decimal("100")  # Use market price as fallback
        }

        result = await validator._validate_balance(signal, context)
        assert result is True  # Should use market price

    def test_slippage_validation_without_expected_slippage(self, validator):
        """Test slippage validation without expected slippage in context."""
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1000"),
            timestamp=datetime.now()
        )

        # Should use default slippage
        result = validator._validate_slippage(signal, {})
        assert result is True
