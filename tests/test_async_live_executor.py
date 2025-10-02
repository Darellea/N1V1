"""
tests/test_async_live_executor.py

Tests for async-first live order execution.
Verifies that CPU-bound operations are offloaded and timeouts work.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.execution.live_executor import LiveOrderExecutor


class TestAsyncLiveExecutor:
    """Test cases for async-first live order execution."""

    @pytest.fixture
    def executor(self):
        """Create a test live executor instance."""
        config = {
            "exchange": {
                "name": "binance",
                "default_type": "spot",
            }
        }
        return LiveOrderExecutor(config)

    @pytest.mark.asyncio
    async def test_order_execution_with_timeout(self, executor):
        """Test that order execution includes timeout protection."""
        # Mock slow order creation
        mock_exchange = AsyncMock()

        async def slow_order(*args, **kwargs):
            await asyncio.sleep(60)

        mock_exchange.create_order = AsyncMock(side_effect=slow_order)  # Very slow
        executor.exchange = mock_exchange

        # Mock signal
        signal = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.001,
            "price": 50000,
            "order_type": "market",
        }

        start_time = time.time()

        # This should timeout after 30 seconds
        with pytest.raises(RuntimeError, match="Order execution timed out"):
            await executor.execute_live_order(signal)

        elapsed = time.time() - start_time
        assert elapsed >= 30.0 and elapsed < 35.0  # Should timeout after ~30 seconds

    @pytest.mark.asyncio
    async def test_cpu_bound_parameter_processing_offloaded(self, executor):
        """Test that CPU-bound order parameter processing is offloaded to thread pool."""
        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.create_order.return_value = {"id": "test_order_123"}
        executor.exchange = mock_exchange

        # Mock signal with complex processing requirements
        signal = MagicMock()
        signal.symbol = "BTC/USDT"
        signal.side = "buy"
        signal.amount = 0.001
        signal.price = 50000
        signal.order_type.value = "market"
        signal.params = {"test": "param"}
        signal.metadata = {"extra": "data"}

        # Mock the thread pool operation
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = {
                "symbol": "BTC/USDT",
                "side": "buy",
                "amount": 0.001,
                "price": 50000,
                "type": "market",
                "params": {"test": "param", "extra": "data"},
            }

            result = await executor.execute_live_order(signal)

            # Verify that CPU-intensive parameter processing was offloaded
            assert mock_to_thread.called
            args, kwargs = mock_to_thread.call_args
            assert (
                args[0] == executor._prepare_order_params
            )  # Should call prepare function in thread

            # Verify order was executed
            assert result["id"] == "test_order_123"

    @pytest.mark.asyncio
    async def test_event_loop_not_blocked_during_order_processing(self, executor):
        """Test that order processing doesn't block the event loop."""
        # Mock exchange with slow response
        mock_exchange = AsyncMock()

        async def slow_create_order(*args, **kwargs):
            await asyncio.sleep(0.2)
            return {"id": "test_order", "status": "filled"}

        mock_exchange.create_order.side_effect = slow_create_order
        executor.exchange = mock_exchange

        # Simple signal
        signal = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.001,
            "price": 50000,
            "order_type": "market",
        }

        # Mock the thread pool operation to be slow (simulating CPU work)
        async def slow_to_thread(func, *args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow CPU work in thread pool
            return func(*args, **kwargs)

        with patch("asyncio.to_thread", side_effect=slow_to_thread):
            # Start order execution
            order_task = asyncio.create_task(executor.execute_live_order(signal))

            # While order processing is running, verify event loop is still responsive
            start_time = time.time()
            responsive_check = 0

            while not order_task.done() and (time.time() - start_time) < 1.0:
                # This should execute multiple times if event loop is not blocked
                await asyncio.sleep(0.01)
                responsive_check += 1

            # Wait for order to complete
            result = await order_task

            # Verify event loop remained responsive during order processing
            assert (
                responsive_check > 5
            ), "Event loop was blocked during order processing"
            assert result["id"] == "test_order"

    @pytest.mark.asyncio
    async def test_concurrent_orders_dont_block_each_other(self, executor):
        """Test that multiple concurrent order executions don't block each other."""
        # Mock exchange with varying response times
        mock_exchange = AsyncMock()
        executor.exchange = mock_exchange

        # Mock create_order with different delays for different symbols
        async def mock_create_order(*args, **kwargs):
            symbol = args[0] if args else kwargs.get("symbol", "")
            if "BTC" in symbol:
                await asyncio.sleep(0.15)  # BTC orders take longer
            elif "ETH" in symbol:
                await asyncio.sleep(0.1)  # ETH orders medium time
            else:
                await asyncio.sleep(0.05)  # Other orders fast

            return {"id": f"order_{symbol}", "status": "filled"}

        mock_exchange.create_order.side_effect = mock_create_order

        # Create multiple order signals
        signals = [
            {
                "symbol": "BTC/USDT",
                "side": "buy",
                "amount": 0.001,
                "price": 50000,
                "order_type": "market",
            },
            {
                "symbol": "ETH/USDT",
                "side": "sell",
                "amount": 1.0,
                "price": 3000,
                "order_type": "market",
            },
            {
                "symbol": "ADA/USDT",
                "side": "buy",
                "amount": 100.0,
                "price": 1.5,
                "order_type": "market",
            },
        ]

        # Execute orders concurrently
        start_time = time.time()

        tasks = [executor.execute_live_order(signal) for signal in signals]
        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        # All operations should complete, and total time should be less than
        # if they were executed sequentially (0.15 + 0.1 + 0.05 = 0.3)
        # Concurrent execution should take ~0.15 seconds
        assert elapsed < 0.2, f"Concurrent orders took too long: {elapsed}s"
        assert len(results) == 3
        assert all("id" in result for result in results)

    @pytest.mark.asyncio
    async def test_parameter_processing_thread_safety(self, executor):
        """Test that parameter processing in threads is thread-safe."""
        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.create_order.return_value = {"id": "test_order"}
        executor.exchange = mock_exchange

        # Create signal that requires complex processing
        signal = MagicMock()
        signal.symbol = "BTC/USDT"
        signal.side = "buy"
        signal.amount = 0.001
        signal.price = 50000
        signal.order_type.value = "market"
        signal.params = {"leverage": 5, "margin_mode": "isolated"}
        signal.metadata = {"strategy": "test", "timestamp": 1234567890}

        # Execute order
        result = await executor.execute_live_order(signal)

        # Verify order was processed correctly
        assert result["id"] == "test_order"

        # Verify exchange was called with correct parameters
        call_args = mock_exchange.create_order.call_args
        assert call_args[0][0] == "BTC/USDT"  # symbol
        assert call_args[0][3] == 0.001  # amount

    @pytest.mark.asyncio
    async def test_timeout_error_properly_handled(self, executor):
        """Test that timeout errors are properly handled and converted."""
        # Mock exchange that times out
        mock_exchange = AsyncMock()

        async def slow_order(*args, **kwargs):
            await asyncio.sleep(35)  # Longer than timeout

        mock_exchange.create_order = AsyncMock(side_effect=slow_order)
        executor.exchange = mock_exchange

        signal = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.001,
            "price": 50000,
            "order_type": "market",
        }

        # Should raise RuntimeError with timeout message
        with pytest.raises(RuntimeError, match="Order execution timed out"):
            await executor.execute_live_order(signal)

    @pytest.mark.asyncio
    async def test_exchange_errors_properly_propagated(self, executor):
        """Test that exchange errors are properly propagated."""
        from ccxt.base.errors import ExchangeError, NetworkError

        # Mock exchange that raises ExchangeError
        mock_exchange = AsyncMock()
        mock_exchange.create_order.side_effect = ExchangeError("Insufficient balance")
        executor.exchange = mock_exchange

        signal = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 1000,  # Large amount to trigger error
            "price": 50000,
            "order_type": "market",
        }

        # Should propagate the ExchangeError
        with pytest.raises(ExchangeError, match="Insufficient balance"):
            await executor.execute_live_order(signal)

    @pytest.mark.asyncio
    async def test_network_errors_properly_handled(self, executor):
        """Test that network errors are properly handled."""
        from ccxt.base.errors import NetworkError

        # Mock exchange that raises NetworkError
        mock_exchange = AsyncMock()
        mock_exchange.create_order.side_effect = NetworkError("Connection timeout")
        executor.exchange = mock_exchange

        signal = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.001,
            "price": 50000,
            "order_type": "market",
        }

        # Should propagate the NetworkError
        with pytest.raises(NetworkError, match="Connection timeout"):
            await executor.execute_live_order(signal)


class TestAsyncOrderParameterProcessing:
    """Test cases for order parameter processing in threads."""

    @pytest.fixture
    def executor(self):
        """Create a test live executor instance."""
        config = {"exchange": {"name": "binance"}}
        return LiveOrderExecutor(config)

    def test_prepare_order_params_dict_signal(self, executor):
        """Test parameter preparation from dict signal."""
        signal = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.001,
            "price": 50000,
            "order_type": "market",
            "params": {"test": "param"},
        }

        result = executor._prepare_order_params(signal)

        expected = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.001,
            "price": 50000,
            "type": "market",
            "params": {"test": "param"},
        }

        assert result == expected

    def test_prepare_order_params_object_signal(self, executor):
        """Test parameter preparation from object signal."""
        signal = MagicMock()
        signal.symbol = "ETH/USDT"
        signal.side = "sell"
        signal.amount = 1.5
        signal.price = 3000
        signal.order_type.value = "limit"
        signal.params = {"time_in_force": "GTC"}
        signal.metadata = {"strategy_id": "test_strategy"}

        result = executor._prepare_order_params(signal)

        expected = {
            "symbol": "ETH/USDT",
            "side": "sell",
            "amount": 1.5,
            "price": 3000,
            "type": "limit",
            "params": {"time_in_force": "GTC"},  # metadata is not merged into params
        }

        assert result == expected

    def test_prepare_order_params_signal_type_mapping(self, executor):
        """Test signal type to side mapping."""
        from core.contracts import SignalType

        # Test ENTRY_LONG -> buy
        signal = MagicMock()
        signal.symbol = "BTC/USDT"
        signal.side = None  # Explicitly set to None to trigger signal_type mapping
        signal.amount = 0.001
        signal.price = 50000
        signal.order_type.value = "market"
        signal.signal_type = SignalType.ENTRY_LONG

        result = executor._prepare_order_params(signal)
        assert result["side"] == "buy"

        # Test ENTRY_SHORT -> sell
        signal.signal_type = SignalType.ENTRY_SHORT
        result = executor._prepare_order_params(signal)
        assert result["side"] == "sell"

        # Test EXIT_LONG -> sell
        signal.signal_type = SignalType.EXIT_LONG
        result = executor._prepare_order_params(signal)
        assert result["side"] == "sell"

        # Test EXIT_SHORT -> buy
        signal.signal_type = SignalType.EXIT_SHORT
        result = executor._prepare_order_params(signal)
        assert result["side"] == "buy"

    def test_prepare_order_params_none_values(self, executor):
        """Test parameter preparation with None values."""
        signal = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.001,
            "price": None,  # No price for market order
            "order_type": "market",
        }

        result = executor._prepare_order_params(signal)

        assert result["price"] is None
        assert result["type"] == "market"


class TestAsyncResourceManagement:
    """Test cases for proper async resource management in executor."""

    @pytest.fixture
    def executor(self):
        """Create a test live executor instance."""
        config = {"exchange": {"name": "binance"}}
        return LiveOrderExecutor(config)

    @pytest.mark.asyncio
    async def test_exchange_cleanup_on_shutdown(self, executor):
        """Test that exchange connections are properly closed on shutdown."""
        mock_exchange = AsyncMock()
        executor.exchange = mock_exchange

        await executor.shutdown()

        mock_exchange.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_without_exchange(self, executor):
        """Test shutdown when no exchange is initialized."""
        executor.exchange = None

        # Should not raise exception
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_with_exchange_close_error(self, executor):
        """Test shutdown handles exchange close errors gracefully."""
        mock_exchange = AsyncMock()
        mock_exchange.close.side_effect = Exception("Close failed")
        executor.exchange = mock_exchange

        # Should not raise exception
        await executor.shutdown()

        mock_exchange.close.assert_called_once()


class TestAsyncTimeoutScenarios:
    """Test cases for various timeout scenarios."""

    @pytest.fixture
    def executor(self):
        """Create a test live executor instance."""
        config = {"exchange": {"name": "binance"}}
        return LiveOrderExecutor(config)

    @pytest.mark.asyncio
    async def test_very_fast_order_execution(self, executor):
        """Test that fast orders complete successfully."""
        mock_exchange = AsyncMock()
        mock_exchange.create_order.return_value = {
            "id": "fast_order_123",
            "status": "filled",
        }
        executor.exchange = mock_exchange

        signal = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.001,
            "price": 50000,
            "order_type": "market",
        }

        start_time = time.time()
        result = await executor.execute_live_order(signal)
        elapsed = time.time() - start_time

        assert result["id"] == "fast_order_123"
        assert elapsed < 1.0  # Should complete quickly

    @pytest.mark.asyncio
    async def test_order_execution_timeout_boundary(self, executor):
        """Test order execution at timeout boundary."""
        # Mock exchange that takes just under 30 seconds
        mock_exchange = AsyncMock()

        async def slow_create_order(*args, **kwargs):
            await asyncio.sleep(25)  # Just under 30 second timeout
            return {"id": "slow_order_123", "status": "filled"}

        mock_exchange.create_order.side_effect = slow_create_order
        executor.exchange = mock_exchange

        signal = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.001,
            "price": 50000,
            "order_type": "market",
        }

        start_time = time.time()
        result = await executor.execute_live_order(signal)
        elapsed = time.time() - start_time

        assert result["id"] == "slow_order_123"
        assert elapsed >= 25 and elapsed < 35  # Should complete just before timeout
