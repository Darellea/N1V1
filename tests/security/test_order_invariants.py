"""
Security tests for order flow invariants and validation.

Tests order schema validation, duplicate prevention, state consistency,
and security monitoring for trading operations.
"""

import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from core.order_executor import OrderExecutor
from utils.security import (
    OrderFlowValidator,
    get_order_flow_validator,
    log_security_event,
)


class TestOrderFlowValidator:
    """Test order flow validation and invariants."""

    @pytest.fixture
    def validator(self):
        return OrderFlowValidator()

    def test_validate_order_schema_valid(self, validator):
        """Test validation of a valid order schema."""
        valid_order = {
            "id": "order_12345678901234567890123456789012",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "amount": 0.001,
            "price": 50000.0,
        }

        result = validator.validate_order_schema(valid_order)
        assert result is True

    def test_validate_order_schema_missing_fields(self, validator):
        """Test validation with missing required fields."""
        invalid_order = {
            "id": "order_123",
            "symbol": "BTC/USDT"
            # Missing side, type, amount
        }

        result = validator.validate_order_schema(invalid_order)
        assert result is False

    def test_validate_order_schema_invalid_id(self, validator):
        """Test validation with invalid order ID format."""
        invalid_order = {
            "id": "invalid-order-id!@#",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "amount": 0.001,
        }

        result = validator.validate_order_schema(invalid_order)
        assert result is False

    def test_validate_order_schema_negative_amount(self, validator):
        """Test validation with negative amount."""
        invalid_order = {
            "id": "order_12345678901234567890123456789012",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "amount": -0.001,
        }

        result = validator.validate_order_schema(invalid_order)
        assert result is False

    def test_validate_order_schema_zero_price(self, validator):
        """Test validation with zero price."""
        invalid_order = {
            "id": "order_12345678901234567890123456789012",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "amount": 0.001,
            "price": 0.0,
        }

        result = validator.validate_order_schema(invalid_order)
        assert result is False

    def test_register_order_success(self, validator):
        """Test successful order registration."""
        order = {
            "id": "order_12345678901234567890123456789012",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "amount": 0.001,
        }

        result = validator.register_order(order)
        assert result is True
        assert order["id"] in validator.order_ids
        assert order["id"] in validator.order_states

    def test_register_order_duplicate(self, validator):
        """Test duplicate order registration prevention."""
        order = {
            "id": "order_12345678901234567890123456789012",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "amount": 0.001,
        }

        # Register first time
        result1 = validator.register_order(order)
        assert result1 is True

        # Try to register again (should fail)
        result2 = validator.register_order(order)
        assert result2 is False

    def test_update_order_state_valid_transition(self, validator):
        """Test valid order state transition."""
        order_id = "order_12345678901234567890123456789012"

        # Set up initial state
        validator.order_states[order_id] = {
            "status": "pending",
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
        }

        result = validator.update_order_state(order_id, "filled")
        assert result is True
        assert validator.order_states[order_id]["status"] == "filled"

    def test_update_order_state_invalid_transition(self, validator):
        """Test invalid order state transition."""
        order_id = "order_12345678901234567890123456789012"

        # Set up initial state as filled (terminal state)
        validator.order_states[order_id] = {
            "status": "filled",
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
        }

        # Try to transition from terminal state (should fail)
        result = validator.update_order_state(order_id, "cancelled")
        assert result is False

    def test_update_order_state_unknown_order(self, validator):
        """Test update of unknown order."""
        result = validator.update_order_state("unknown_order", "filled")
        assert result is False

    def test_validate_state_consistency_clean(self, validator):
        """Test state consistency validation with clean state."""
        # Set up some orders
        order_id = "order_12345678901234567890123456789012"
        validator.order_states[order_id] = {
            "status": "pending",
            "created_at": datetime.utcnow() - timedelta(hours=1),
            "last_updated": datetime.utcnow(),
        }

        result = validator.validate_state_consistency()
        assert result["consistent"] is True
        assert len(result["inconsistencies"]) == 0

    def test_validate_state_consistency_stale_order(self, validator):
        """Test state consistency with stale order detection."""
        # Set up a stale order (older than 24 hours)
        order_id = "order_12345678901234567890123456789012"
        validator.order_states[order_id] = {
            "status": "pending",
            "created_at": datetime.utcnow() - timedelta(hours=25),
            "last_updated": datetime.utcnow() - timedelta(hours=25),
        }

        result = validator.validate_state_consistency()
        assert result["consistent"] is False
        assert len(result["inconsistencies"]) == 1
        assert result["inconsistencies"][0]["type"] == "stale_order"

    def test_validate_state_consistency_time_anomaly(self, validator):
        """Test state consistency with time anomaly detection."""
        # Set up order with last_updated before created_at
        order_id = "order_12345678901234567890123456789012"
        validator.order_states[order_id] = {
            "status": "pending",
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow() - timedelta(hours=1),  # Before created_at
        }

        result = validator.validate_state_consistency()
        assert result["consistent"] is False
        assert len(result["inconsistencies"]) == 1
        assert result["inconsistencies"][0]["type"] == "time_anomaly"

    def test_cleanup_completed_orders(self, validator):
        """Test cleanup of old completed orders."""
        # Set up orders with different ages
        old_order_id = "old_order_12345678901234567890123456789012"
        new_order_id = "new_order_12345678901234567890123456789012"

        cutoff_date = datetime.utcnow() - timedelta(days=40)

        validator.order_states = {
            old_order_id: {
                "status": "filled",
                "created_at": cutoff_date - timedelta(days=10),
                "last_updated": cutoff_date - timedelta(days=10),
            },
            new_order_id: {
                "status": "filled",
                "created_at": datetime.utcnow() - timedelta(days=1),
                "last_updated": datetime.utcnow() - timedelta(days=1),
            },
        }
        validator.order_ids = {old_order_id, new_order_id}

        validator.cleanup_completed_orders(max_age_days=30)

        # Old order should be removed
        assert old_order_id not in validator.order_states
        assert old_order_id not in validator.order_ids

        # New order should remain
        assert new_order_id in validator.order_states
        assert new_order_id in validator.order_ids


class TestOrderExecutorSecurity:
    """Test OrderExecutor security features."""

    @pytest.fixture
    def config(self):
        return {
            "max_retries": 3,
            "retry_base_delay": 1.0,
            "retry_max_delay": 30.0,
            "retry_budget": 5,
            "security": {"max_orders_per_minute": 10},
        }

    @pytest.fixture
    def order_executor(self, config):
        return OrderExecutor(config)

    async def test_validate_signal_security_valid(self, order_executor):
        """Test validation of valid signal security."""
        # Create a mock signal
        signal = Mock()
        signal.id = "signal_123"
        signal.symbol = "BTC/USDT"
        signal.side = "buy"
        signal.type = "limit"
        signal.amount = 0.001
        signal.price = 50000.0
        signal.strategy_id = "test_strategy"

        result = await order_executor._validate_signal_security(signal)
        assert result is True

    async def test_validate_signal_security_invalid_schema(self, order_executor):
        """Test validation of signal with invalid schema."""
        # Create a mock signal with missing required fields
        signal = Mock()
        signal.id = "signal_123"
        signal.symbol = "BTC/USDT"
        # Missing side, amount, etc.

        result = await order_executor._validate_signal_security(signal)
        assert result is False

    async def test_validate_signal_security_duplicate_order(self, order_executor):
        """Test validation with duplicate order detection."""
        # First, register an order
        validator = get_order_flow_validator()
        existing_order = {
            "id": "duplicate_order_12345678901234567890123456789012",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "amount": 0.001,
        }
        validator.register_order(existing_order)

        # Create signal with same ID
        signal = Mock()
        signal.id = "duplicate_order_12345678901234567890123456789012"
        signal.symbol = "BTC/USDT"
        signal.side = "buy"
        signal.type = "limit"
        signal.amount = 0.001
        signal.price = 50000.0
        signal.strategy_id = "test_strategy"

        result = await order_executor._validate_signal_security(signal)
        assert result is False

    async def test_check_rate_limits_within_limit(self, order_executor):
        """Test rate limiting within acceptable limits."""
        signal_data = {"symbol": "BTC/USDT", "amount": 0.001}

        # Should allow the request
        result = await order_executor._check_rate_limits(signal_data)
        assert result is True

    async def test_check_rate_limits_exceeded(self, order_executor):
        """Test rate limiting when limit is exceeded."""
        signal_data = {"symbol": "BTC/USDT", "amount": 0.001}

        # Simulate exceeding the rate limit
        order_executor._rate_limit_cache = {
            "BTC/USDT": [time.time()] * 15
        }  # 15 requests

        result = await order_executor._check_rate_limits(signal_data)
        assert result is False

    def test_validate_signal_integrity_valid(self, order_executor):
        """Test signal integrity validation with valid signal."""
        signal = Mock()
        signal.symbol = "BTC/USDT"
        signal.side = "buy"
        signal.amount = 0.001

        result = order_executor._validate_signal_integrity(signal)
        assert result is True

    def test_validate_signal_integrity_missing_attrs(self, order_executor):
        """Test signal integrity validation with missing attributes."""
        signal = Mock()
        signal.symbol = "BTC/USDT"
        # Missing side and amount

        result = order_executor._validate_signal_integrity(signal)
        assert result is False

    def test_validate_signal_integrity_invalid_side(self, order_executor):
        """Test signal integrity validation with invalid side."""
        signal = Mock()
        signal.symbol = "BTC/USDT"
        signal.side = "invalid_side"
        signal.amount = 0.001

        result = order_executor._validate_signal_integrity(signal)
        assert result is False

    def test_validate_signal_integrity_negative_amount(self, order_executor):
        """Test signal integrity validation with negative amount."""
        signal = Mock()
        signal.symbol = "BTC/USDT"
        signal.side = "buy"
        signal.amount = -0.001

        result = order_executor._validate_signal_integrity(signal)
        assert result is False

    async def test_validate_order_flow_invariants(self, order_executor):
        """Test order flow invariants validation."""
        result = await order_executor.validate_order_flow_invariants()

        assert "consistent" in result
        assert "inconsistencies" in result
        assert "total_orders" in result

    def test_get_security_stats(self, order_executor):
        """Test security statistics retrieval."""
        stats = order_executor.get_security_stats()

        assert "execution_stats" in stats
        assert "order_flow_consistency" in stats
        assert "rate_limit_cache_size" in stats
        assert "security_events_logged" in stats

    async def test_perform_security_health_check(self, order_executor):
        """Test security health check."""
        health = await order_executor.perform_security_health_check()

        assert "order_flow_validator" in health
        assert "rate_limiting" in health
        assert "signal_validation" in health
        assert "schema_validation" in health


class TestSecurityEventLogging:
    """Test security event logging functionality."""

    def test_log_security_event_info(self):
        """Test logging security event at INFO level."""
        with patch("utils.security.logging") as mock_logging:
            log_security_event("test_event", {"key": "value"}, "INFO")

            mock_logger = mock_logging.getLogger.return_value
            mock_logger.info.assert_called_once()

    def test_log_security_event_warning(self):
        """Test logging security event at WARNING level."""
        with patch("utils.security.logging") as mock_logging:
            log_security_event("test_event", {"key": "value"}, "WARNING")

            mock_logger = mock_logging.getLogger.return_value
            mock_logger.warning.assert_called_once()

    def test_log_security_event_error(self):
        """Test logging security event at ERROR level."""
        with patch("utils.security.logging") as mock_logging:
            log_security_event("test_event", {"key": "value"}, "ERROR")

            mock_logger = mock_logging.getLogger.return_value
            mock_logger.error.assert_called_once()


class TestOrderFlowIntegration:
    """Test integration between order flow components."""

    def test_get_order_flow_validator_singleton(self):
        """Test singleton pattern for order flow validator."""
        validator1 = get_order_flow_validator()
        validator2 = get_order_flow_validator()

        assert validator1 is validator2

    async def test_end_to_end_order_validation(self):
        """Test end-to-end order validation flow."""
        validator = get_order_flow_validator()

        # Create a valid order
        order = {
            "id": "e2e_test_order_12345678901234567890123456789012",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "amount": 0.001,
            "price": 50000.0,
        }

        # Test schema validation
        schema_valid = validator.validate_order_schema(order)
        assert schema_valid is True

        # Test registration
        registered = validator.register_order(order)
        assert registered is True

        # Test state update
        state_updated = validator.update_order_state(order["id"], "filled")
        assert state_updated is True

        # Test consistency validation
        consistency = validator.validate_state_consistency()
        assert consistency["consistent"] is True

    async def test_malformed_order_handling(self):
        """Test handling of malformed orders."""
        validator = get_order_flow_validator()

        malformed_orders = [
            # Missing required fields
            {"id": "test1", "symbol": "BTC/USDT"},
            # Invalid ID format
            {
                "id": "invalid!@#",
                "symbol": "BTC/USDT",
                "side": "buy",
                "type": "limit",
                "amount": 0.001,
            },
            # Negative amount
            {
                "id": "test3",
                "symbol": "BTC/USDT",
                "side": "buy",
                "type": "limit",
                "amount": -0.001,
            },
            # Zero price
            {
                "id": "test4",
                "symbol": "BTC/USDT",
                "side": "buy",
                "type": "limit",
                "amount": 0.001,
                "price": 0,
            },
        ]

        for order in malformed_orders:
            result = validator.validate_order_schema(order)
            assert result is False, f"Order {order} should have failed validation"


if __name__ == "__main__":
    pytest.main([__file__])
