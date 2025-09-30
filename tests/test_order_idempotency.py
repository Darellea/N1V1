"""
Unit tests for order idempotency functionality.

Tests idempotency key requirements and duplicate order handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from core.exceptions import MissingIdempotencyError
from core.order_manager import OrderManager
from core.types import TradingMode


class TestOrderIdempotency:
    """Test cases for order idempotency functionality."""

    @pytest.fixture
    def order_manager(self):
        """Create a test OrderManager instance."""
        config = {
            "order": {},
            "paper": {"initial_balance": 1000.0},
            "reliability": {}
        }
        return OrderManager(config, TradingMode.PAPER)

    @pytest.fixture
    def mock_signal(self):
        """Create a mock signal with required attributes."""
        signal = MagicMock()
        signal.strategy_id = "test_strategy"
        signal.symbol = "BTC/USDT"
        signal.signal_type = "ENTRY_LONG"
        signal.order_type = "MARKET"
        signal.amount = "0.001"
        signal.idempotency_key = "test_key_123"
        return signal

    @pytest.mark.asyncio
    async def test_duplicate_order_with_idempotency_key_returns_cached_result(self, order_manager, mock_signal):
        """Test that submitting the same order twice with idempotency_key returns only one execution."""
        # Mock the execution strategy to return a successful result
        mock_result = {"id": "order_123", "status": "filled"}
        order_manager._execution_strategies[TradingMode.PAPER].execute_order = AsyncMock(return_value=mock_result)

        # First execution
        result1 = await order_manager.execute_order(mock_signal)
        assert result1 == mock_result

        # Second execution with same key should return cached result
        result2 = await order_manager.execute_order(mock_signal)
        assert result2 == mock_result

        # Verify execute_order was only called once
        order_manager._execution_strategies[TradingMode.PAPER].execute_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_order_without_idempotency_key_raises_error(self, order_manager, mock_signal):
        """Test that submitting an order without idempotency_key raises MissingIdempotencyError."""
        # Remove idempotency key
        mock_signal.idempotency_key = None

        with pytest.raises(MissingIdempotencyError):
            await order_manager.execute_order(mock_signal)

    @pytest.mark.asyncio
    async def test_order_with_empty_idempotency_key_raises_error(self, order_manager, mock_signal):
        """Test that submitting an order with empty idempotency_key raises MissingIdempotencyError."""
        # Set empty idempotency key
        mock_signal.idempotency_key = ""

        with pytest.raises(MissingIdempotencyError):
            await order_manager.execute_order(mock_signal)

    @pytest.mark.asyncio
    async def test_different_idempotency_keys_allow_multiple_executions(self, order_manager, mock_signal):
        """Test that orders with different idempotency keys are executed separately."""
        # Mock the execution strategy
        mock_result1 = {"id": "order_123", "status": "filled"}
        mock_result2 = {"id": "order_456", "status": "filled"}
        order_manager._execution_strategies[TradingMode.PAPER].execute_order = AsyncMock(side_effect=[mock_result1, mock_result2])

        # First execution
        result1 = await order_manager.execute_order(mock_signal)
        assert result1 == mock_result1

        # Second execution with different key
        mock_signal.idempotency_key = "different_key_456"
        result2 = await order_manager.execute_order(mock_signal)
        assert result2 == mock_result2

        # Verify execute_order was called twice
        assert order_manager._execution_strategies[TradingMode.PAPER].execute_order.call_count == 2
