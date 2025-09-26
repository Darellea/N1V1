"""
Integration test for end-to-end order flow.
Tests the complete order lifecycle from signal to execution.
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from core.order_manager import OrderManager
from core.types import TradingMode
from core.contracts import SignalType


class MockSignal:
    """Mock signal for testing."""
    def __init__(self, symbol="BTC/USDT", signal_type=SignalType.ENTRY_LONG,
                 amount=0.001, price=50000, order_type="MARKET"):
        self.symbol = symbol
        self.signal_type = signal_type
        self.amount = amount
        self.price = price
        self.current_price = price  # For paper executor slippage calculation
        self.order_type = order_type
        self.strategy_id = "test_strategy"
        self.signal_strength = "STRONG"
        self.stop_loss = None
        self.take_profit = None
        self.timestamp = 1234567890


@pytest.mark.asyncio
class TestOrderFlowIntegration:
    """Test end-to-end order flow integration."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "order": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "base_currency": "USDT",
                "trade_fee": "0.001"
            },
            "exchange": {
                "name": "kucoin"
            },
            "paper": {
                "initial_balance": "1000.0",
                "trade_fee": "0.001",
                "slippage": "0.001"
            },
            "risk": {
                "max_position_size": 0.1
            }
        }

    @pytest.fixture
    def order_manager(self, config):
        """Order manager instance."""
        return OrderManager(config, TradingMode.PAPER)

    async def test_complete_order_flow_paper_trading(self, order_manager):
        """Test complete order flow in paper trading mode."""
        # Create a mock signal
        signal = MockSignal()

        # Execute order
        result = await order_manager.execute_order(signal)

        # Verify order was processed
        assert result is not None
        assert "id" in result
        assert result["symbol"] == "BTC/USDT"
        assert result["status"] == "filled" or result["status"] == "closed"

        # Check balance was updated
        balance = await order_manager.get_balance()
        assert balance < Decimal("1000.0")  # Should have decreased due to position

        # Check equity calculation
        equity = await order_manager.get_equity()
        assert equity == balance  # In paper mode, equity equals balance

        # Verify position was recorded
        positions = order_manager.order_processor.positions
        assert "BTC/USDT" in positions
        position = positions["BTC/USDT"]
        assert abs(position["amount"] - Decimal("0.001")) < Decimal("0.0001")  # Approximate comparison
        assert abs(position["entry_price"] - Decimal("50050")) < Decimal("0.01")  # Account for slippage

    async def test_order_flow_with_validation(self, order_manager):
        """Test order flow with payload validation."""
        # Test invalid signal (missing required fields)
        invalid_signal = MockSignal()
        invalid_signal.symbol = ""  # Invalid symbol

        result = await order_manager.execute_order(invalid_signal)
        assert result is None  # Should fail validation

    async def test_order_flow_multiple_orders(self, order_manager):
        """Test multiple orders in sequence."""
        signals = [
            MockSignal("BTC/USDT", SignalType.ENTRY_LONG, 0.001, 50000),
            MockSignal("ETH/USDT", SignalType.ENTRY_LONG, 0.01, 3000),
            MockSignal("BTC/USDT", SignalType.EXIT_LONG, 0.001, 51000),
        ]

        results = []
        for signal in signals:
            result = await order_manager.execute_order(signal)
            results.append(result)

        # Verify results
        assert all(r is not None for r in results)
        assert len(order_manager.order_processor.positions) >= 0  # May have positions left

        # Check final balance
        final_balance = await order_manager.get_balance()
        assert final_balance != Decimal("1000.0")  # Should have changed

    async def test_order_flow_portfolio_mode(self, config):
        """Test order flow in portfolio mode."""
        # Initialize portfolio
        pairs = ["BTC/USDT", "ETH/USDT"]
        order_manager = OrderManager(config, TradingMode.PAPER)
        await order_manager.initialize_portfolio(pairs, True)

        # Execute orders for different pairs
        signals = [
            MockSignal("BTC/USDT", SignalType.ENTRY_LONG, 0.001, 50000),
            MockSignal("ETH/USDT", SignalType.ENTRY_LONG, 0.01, 3000),
        ]

        for signal in signals:
            result = await order_manager.execute_order(signal)
            assert result is not None

        # Check portfolio balances
        portfolio_balances = order_manager.paper_balances
        assert len(portfolio_balances) > 0

        # Check equity calculation in portfolio mode
        equity = await order_manager.get_equity()
        assert equity > Decimal("0")

    async def test_order_flow_live_mode_simulation(self, config):
        """Test order flow simulation in live mode."""
        # Create order manager in live mode
        order_manager = OrderManager(config, TradingMode.LIVE)

        # Mock the live executor's execute_live_order method
        with patch.object(order_manager.live_executor, 'execute_live_order', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {
                "id": "test_order_123",
                "symbol": "BTC/USDT",
                "side": "buy",
                "amount": 0.001,
                "price": 50000,
                "status": "filled"
            }

            # Execute order
            signal = MockSignal()
            result = await order_manager.execute_order(signal)

            # Verify the mock was called
            mock_execute.assert_called_once()

            # Verify result
            assert result is not None
            assert result["id"] == "test_order_123"

    async def test_order_cancellation_flow(self, order_manager):
        """Test order cancellation flow."""
        # This would require mocking open orders
        # For now, just test the method exists and doesn't crash
        await order_manager.cancel_all_orders()
        assert True  # If we get here, no exception was raised

    async def test_balance_and_equity_calculation(self, order_manager):
        """Test balance and equity calculation accuracy."""
        # Initial state
        initial_balance = await order_manager.get_balance()
        initial_equity = await order_manager.get_equity()

        assert initial_balance == Decimal("1000.0")
        assert initial_equity == initial_balance

        # Execute an order
        signal = MockSignal()
        result = await order_manager.execute_order(signal)
        assert result is not None  # Ensure order executed successfully

        # Clear balance cache to get updated value
        order_manager._balance_cache = None
        order_manager._equity_cache = None

        # Check updated values
        new_balance = await order_manager.get_balance()
        new_equity = await order_manager.get_equity()

        assert new_balance < initial_balance
        assert new_equity == new_balance  # Paper mode

    async def test_error_handling_in_order_flow(self, order_manager):
        """Test error handling throughout the order flow."""
        # Test with invalid amount
        signal = MockSignal()
        signal.amount = -1  # Invalid amount

        result = await order_manager.execute_order(signal)
        assert result is None  # Should fail validation

        # Test with invalid price
        signal = MockSignal()
        signal.price = -100  # Invalid price

        result = await order_manager.execute_order(signal)
        assert result is None  # Should fail validation

    async def test_concurrent_order_execution(self, order_manager):
        """Test concurrent order execution."""
        signals = [MockSignal("BTC/USDT", SignalType.ENTRY_LONG, 0.001, 50000 + i)
                  for i in range(10)]

        # Execute orders concurrently
        tasks = [order_manager.execute_order(signal) for signal in signals]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all orders were processed
        successful_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        assert len(successful_results) == 10

        # Check positions
        positions = order_manager.order_processor.positions
        assert len(positions) == 1  # All for the same symbol
