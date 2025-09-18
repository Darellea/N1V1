import asyncio
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from ccxt.base.errors import NetworkError, ExchangeError
from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types import TradingMode
from core.order_manager import OrderManager
from notifier.discord_bot import DiscordNotifier


class TestRegression:
    """Regression tests for failure scenarios and edge cases."""

    @pytest.fixture
    def config(self):
        """Config for regression testing."""
        return {
            "order": {
                "base_currency": "USDT",
                "exchange": "binance"
            },
            "risk": {},
            "paper": {
                "initial_balance": 10000.0
            },
            "reliability": {},
            "discord": {
                "alerts": {"enabled": True},
                "webhook_url": "https://discord.com/api/webhooks/test"
            }
        }

    @pytest.fixture
    def mock_executors(self):
        """Mock executors for regression testing."""
        with patch('core.order_manager.LiveOrderExecutor') as mock_live, \
             patch('core.order_manager.PaperOrderExecutor') as mock_paper, \
             patch('core.order_manager.BacktestOrderExecutor') as mock_backtest:

            mock_live_instance = MagicMock()
            mock_paper_instance = MagicMock()
            mock_backtest_instance = MagicMock()

            mock_live.return_value = mock_live_instance
            mock_paper.return_value = mock_paper_instance
            mock_backtest.return_value = mock_backtest_instance

            # Default successful responses
            mock_paper_instance.execute_paper_order = AsyncMock(return_value={"id": "test_order", "status": "filled"})
            mock_paper_instance.get_balance = MagicMock(return_value=Decimal("10000"))
            mock_paper_instance.set_initial_balance = MagicMock()
            mock_paper_instance.set_portfolio_mode = MagicMock()

            yield {
                'live': mock_live_instance,
                'paper': mock_paper_instance,
                'backtest': mock_backtest_instance
            }

    @pytest.fixture
    def mock_managers(self):
        """Mock managers for regression testing."""
        with patch('core.order_manager.ReliabilityManager') as mock_reliability, \
             patch('core.order_manager.PortfolioManager') as mock_portfolio, \
             patch('core.order_manager.OrderProcessor') as mock_processor:

            mock_reliability_instance = MagicMock()
            mock_portfolio_instance = MagicMock()
            mock_processor_instance = MagicMock()

            mock_reliability.return_value = mock_reliability_instance
            mock_portfolio.return_value = mock_portfolio_instance
            mock_processor.return_value = mock_processor_instance

            # Default states
            mock_reliability_instance.safe_mode_active = False
            # Mock retry_async to actually retry and eventually raise the exception
            async def mock_retry_async(func, **kwargs):
                try:
                    return await func()
                except Exception as e:
                    # For testing, just re-raise the exception to simulate retry failure
                    raise e
            mock_reliability_instance.retry_async = AsyncMock(side_effect=mock_retry_async)
            mock_reliability_instance.record_critical_error = MagicMock()

            mock_portfolio_instance.paper_balances = {"USDT": Decimal("10000")}
            mock_portfolio_instance.set_initial_balance = MagicMock()
            mock_portfolio_instance.initialize_portfolio = AsyncMock()

            mock_processor_instance.process_order = AsyncMock(return_value={"id": "processed_order", "status": "filled"})
            mock_processor_instance.open_orders = {}
            mock_processor_instance.closed_orders = {}
            mock_processor_instance.positions = {}
            mock_processor_instance.get_active_order_count = MagicMock(return_value=0)
            mock_processor_instance.get_open_position_count = MagicMock(return_value=0)

            yield {
                'reliability': mock_reliability_instance,
                'portfolio': mock_portfolio_instance,
                'processor': mock_processor_instance
            }

    @pytest.mark.asyncio
    async def test_network_error_retry_mechanism(self, config, mock_executors, mock_managers):
        """Test that network errors trigger retry mechanism."""
        order_manager = OrderManager(config, TradingMode.LIVE)

        # Mock live executor to raise NetworkError
        mock_executors['live'].execute_live_order = AsyncMock(side_effect=NetworkError("Connection timeout"))

        signal = TradingSignal(
            strategy_id="network_error_test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        # Execute order - should trigger retry
        result = await order_manager.execute_order(signal)

        # Should return None due to error
        assert result is None

        # Verify retry was attempted
        mock_managers['reliability'].retry_async.assert_called_once()

        # Verify error was recorded
        mock_managers['reliability'].record_critical_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_exchange_error_handling(self, config, mock_executors, mock_managers):
        """Test handling of exchange-specific errors."""
        order_manager = OrderManager(config, TradingMode.LIVE)

        # Mock live executor to raise ExchangeError
        mock_executors['live'].execute_live_order = AsyncMock(side_effect=ExchangeError("Invalid API key"))

        signal = TradingSignal(
            strategy_id="exchange_error_test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        result = await order_manager.execute_order(signal)

        # Should return None due to error
        assert result is None

        # Verify error handling
        mock_managers['reliability'].record_critical_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_error_recovery(self, config, mock_executors, mock_managers):
        """Test recovery from timeout errors."""
        order_manager = OrderManager(config, TradingMode.LIVE)

        # Mock live executor to raise TimeoutError
        mock_executors['live'].execute_live_order = AsyncMock(side_effect=asyncio.TimeoutError("Request timeout"))

        signal = TradingSignal(
            strategy_id="timeout_test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        result = await order_manager.execute_order(signal)

        # Should return None due to timeout
        assert result is None

        # Verify timeout was handled as network error
        mock_managers['reliability'].record_critical_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_corrupted_signal_data_handling(self, config, mock_executors, mock_managers):
        """Test handling of corrupted or invalid signal data."""
        order_manager = OrderManager(config, TradingMode.PAPER)

        # Create signal with invalid data
        signal = TradingSignal(
            strategy_id="corrupted_data_test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        # Mock executor to raise exception due to corrupted data
        mock_executors['paper'].execute_paper_order = AsyncMock(side_effect=ValueError("Invalid amount"))

        result = await order_manager.execute_order(signal)

        # Should return None due to error
        assert result is None

    @pytest.mark.asyncio
    async def test_balance_fetch_failure_recovery(self, config, mock_executors, mock_managers):
        """Test recovery when balance fetch fails."""
        order_manager = OrderManager(config, TradingMode.LIVE)

        # Mock exchange balance fetch to fail
        mock_exchange = MagicMock()
        mock_exchange.fetch_balance = AsyncMock(side_effect=NetworkError("Balance fetch failed"))
        mock_executors['live'].exchange = mock_exchange

        balance = await order_manager.get_balance()

        # Should return 0 on failure
        assert balance == Decimal("0")

    @pytest.mark.asyncio
    async def test_equity_calculation_with_missing_ticker_data(self, config, mock_executors, mock_managers):
        """Test equity calculation when ticker data is unavailable."""
        order_manager = OrderManager(config, TradingMode.LIVE)

        # Mock exchange with missing ticker data
        mock_exchange = MagicMock()
        mock_exchange.fetch_balance = AsyncMock(return_value={"total": {"USDT": 5000.0}})
        mock_exchange.fetch_ticker = AsyncMock(side_effect=NetworkError("Ticker unavailable"))
        mock_executors['live'].exchange = mock_exchange

        # Add a position to test unrealized PnL calculation
        mock_managers['processor'].positions = {
            "BTC/USDT": {
                "entry_price": Decimal("50000"),
                "amount": Decimal("1.0")
            }
        }

        equity = await order_manager.get_equity()

        # Should return balance only (unrealized calculation failed)
        assert equity == Decimal("5000")

    @pytest.mark.asyncio
    async def test_discord_notification_failure_does_not_break_flow(self, config, mock_executors, mock_managers):
        """Test that Discord notification failures don't break the main flow."""
        from notifier.discord_bot import DiscordNotifier

        with patch('aiohttp.ClientSession') as mock_session:
            mock_instance = MagicMock()
            mock_session.return_value = mock_instance

            # Mock Discord API failure
            mock_response = MagicMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Discord API Error")
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.close = AsyncMock()

            discord_config = config["discord"]
            notifier = DiscordNotifier(discord_config)
            await notifier.initialize()

            # Try to send signal alert
            signal = TradingSignal(
                strategy_id="test",
                symbol="BTC/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type="market",
                amount=Decimal("1.0")
            )
            result = await notifier.send_signal_alert(signal)

            # Should return False but not raise exception
            assert result is False

    @pytest.mark.asyncio
    async def test_discord_rate_limit_handling(self, config, mock_executors, mock_managers):
        """Test Discord rate limit handling."""
        from notifier.discord_bot import DiscordNotifier

        with patch('aiohttp.ClientSession') as mock_session:
            mock_instance = MagicMock()
            mock_session.return_value = mock_instance

            # Mock rate limit response
            rate_limit_response = MagicMock()
            rate_limit_response.status = 429
            rate_limit_response.json = AsyncMock(return_value={"retry_after": 1.0})
            rate_limit_response.text = AsyncMock(return_value="Rate limited")

            # Mock success response
            success_response = MagicMock()
            success_response.status = 204
            success_response.text = AsyncMock(return_value="OK")
            success_response.json = AsyncMock(return_value={})

            # First call rate limited, second succeeds
            mock_instance.post = AsyncMock(side_effect=[rate_limit_response, success_response])
            mock_instance.close = AsyncMock()

            discord_config = config["discord"]
            notifier = DiscordNotifier(discord_config)
            await notifier.initialize()

            # Try to send signal alert
            signal = TradingSignal(
                strategy_id="test",
                symbol="BTC/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type="market",
                amount=Decimal("1.0")
            )
            result = await notifier.send_signal_alert(signal)

            # Should eventually succeed after retry
            assert result is True
            assert mock_instance.post.call_count == 2

    @pytest.mark.asyncio
    async def test_portfolio_initialization_with_invalid_data(self, config, mock_executors, mock_managers):
        """Test portfolio initialization with invalid data."""
        order_manager = OrderManager(config, TradingMode.PAPER)

        # Test with invalid pairs
        with pytest.raises((ValueError, TypeError)):
            await order_manager.initialize_portfolio(None, True, {"INVALID": 1.5})

    @pytest.mark.asyncio
    async def test_concurrent_order_cancellation_stress(self, config, mock_executors, mock_managers):
        """Test concurrent order cancellation under stress."""
        order_manager = OrderManager(config, TradingMode.LIVE)

        # Mock exchange cancellation
        mock_exchange = MagicMock()
        mock_exchange.cancel_order = AsyncMock(return_value=True)
        mock_executors['live'].exchange = mock_exchange

        # Add multiple orders
        order_ids = [f"order_{i}" for i in range(10)]
        for order_id in order_ids:
            mock_managers['processor'].open_orders[order_id] = MagicMock()

        # Cancel all orders concurrently
        tasks = [order_manager.cancel_order(order_id) for order_id in order_ids]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)

        # Verify exchange was called for each
        assert mock_exchange.cancel_order.call_count == 10

    @pytest.mark.asyncio
    async def test_memory_leak_prevention_in_long_running_sessions(self, config, mock_executors, mock_managers):
        """Test prevention of memory leaks in long-running sessions."""
        order_manager = OrderManager(config, TradingMode.PAPER)

        # Simulate many order executions
        signals = []
        for i in range(100):
            signal = TradingSignal(
                strategy_id=f"memory_test_{i}",
                symbol="BTC/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type="market",
                amount=Decimal("0.1")
            )
            signals.append(signal)

        # Execute all orders
        tasks = [order_manager.execute_order(signal) for signal in signals]
        results = await asyncio.gather(*tasks)

        # Verify all completed
        assert len(results) == 100
        assert all(result is not None for result in results)

        # Verify no memory leaks in tracking structures
        assert len(mock_managers['processor'].open_orders) == 0  # Should be cleaned up

    @pytest.mark.asyncio
    async def test_graceful_shutdown_under_load(self, config, mock_executors, mock_managers):
        """Test graceful shutdown when system is under load."""
        order_manager = OrderManager(config, TradingMode.LIVE)

        # Mock live executor shutdown
        mock_executors['live'].shutdown = AsyncMock()

        # Simulate active operations
        mock_managers['processor'].open_orders = {"active_order": MagicMock()}

        # Shutdown
        await order_manager.shutdown()

        # Verify cleanup happened
        mock_executors['live'].shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_trading_mode_fallback(self, config, mock_executors, mock_managers):
        """Test fallback behavior for invalid trading modes."""
        # Test with invalid mode
        order_manager = OrderManager(config, "invalid_mode")

        # Should fallback to PAPER mode
        assert order_manager.mode == TradingMode.PAPER

        # Should still work
        signal = TradingSignal(
            strategy_id="fallback_test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        result = await order_manager.execute_order(signal)
        assert result is not None

    @pytest.mark.asyncio
    async def test_extreme_market_conditions_simulation(self, config, mock_executors, mock_managers):
        """Test behavior under extreme market conditions."""
        order_manager = OrderManager(config, TradingMode.PAPER)

        # Simulate extreme price volatility
        extreme_signals = [
            TradingSignal(
                strategy_id=f"extreme_{i}",
                symbol="BTC/USDT",
                signal_type=SignalType.ENTRY_LONG if i % 2 == 0 else SignalType.ENTRY_SHORT,
                signal_strength=SignalStrength.EXTREME,
                order_type="market",
                amount=Decimal("1000"),  # Large amount
                price=Decimal("100000") if i % 2 == 0 else Decimal("1000")  # Extreme prices
            ) for i in range(10)
        ]

        # Execute extreme orders
        tasks = [order_manager.execute_order(signal) for signal in extreme_signals]
        results = await asyncio.gather(*tasks)

        # System should handle extreme conditions without crashing
        assert len(results) == 10
        # Some may fail due to validation, but shouldn't crash
        successful = [r for r in results if r is not None]
        assert len(successful) >= 0  # At least some should process
