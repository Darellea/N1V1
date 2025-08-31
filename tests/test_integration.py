import asyncio
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types import TradingMode
from core.order_manager import OrderManager
from notifier.discord_bot import DiscordNotifier
from core.signal_router import SignalRouter
from core.execution.paper_executor import PaperOrderExecutor


class DummyRiskManager:
    """Simple risk manager stub that approves all signals."""

    require_stop_loss = False

    async def evaluate_signal(self, signal, market_data=None):
        await asyncio.sleep(0)  # yield control to allow concurrency in tests
        return True


class TestIntegration:
    """Integration tests for end-to-end trading workflows."""

    @pytest.fixture
    def config(self):
        """Complete config for integration testing."""
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
        """Mock all executors for integration."""
        with patch('core.order_manager.LiveOrderExecutor') as mock_live, \
             patch('core.order_manager.PaperOrderExecutor') as mock_paper, \
             patch('core.order_manager.BacktestOrderExecutor') as mock_backtest:

            mock_paper_instance = MagicMock()
            mock_paper_instance.execute_paper_order = AsyncMock(return_value={
                "id": "integration_order",
                "symbol": "BTC/USDT",
                "status": "filled",
                "amount": 1.0,
                "price": 50000.0,
                "cost": 50000.0
            })

            # Mock paper_balance attribute and get_balance method
            mock_paper_instance.paper_balance = Decimal("10000")
            mock_paper_instance.get_balance = MagicMock(return_value=Decimal("10000"))
            mock_paper_instance.set_initial_balance = MagicMock()
            mock_paper_instance.set_portfolio_mode = MagicMock()

            # Mock config for fee calculation
            mock_config = MagicMock()
            mock_config.__getitem__ = MagicMock(side_effect=lambda key: {
                "trade_fee": Decimal("0.001"),  # 0.1% fee
                "slippage": Decimal("0.001"),   # 0.1% slippage
                "base_currency": "USDT"
            }[key])
            mock_paper_instance.config = mock_config

            # Mock the execute_paper_order to update balance
            async def mock_execute_with_balance_update(signal):
                # Simulate order execution result
                result = {
                    "id": "paper_0",
                    "symbol": signal.symbol,
                    "type": signal.order_type,
                    "side": "buy",
                    "amount": float(signal.amount),
                    "price": float(signal.price),
                    "status": "filled",
                    "filled": float(signal.amount),
                    "remaining": 0.0,
                    "cost": float(signal.amount) * float(signal.price),
                    "fee": {"cost": 0.00001, "currency": "USDT"},
                    "timestamp": 1234567890
                }
                # Simulate balance update: 10000 - 500 - 0.00001 = 9499.99999
                mock_paper_instance.paper_balance = Decimal("9499.99999")
                mock_paper_instance.get_balance.return_value = Decimal("9499.99999")
                return result

            mock_paper_instance.execute_paper_order.side_effect = mock_execute_with_balance_update

            mock_paper.return_value = mock_paper_instance
            mock_live.return_value = MagicMock()
            mock_backtest.return_value = MagicMock()

            yield {
                'paper': mock_paper_instance
            }

    @pytest.fixture
    def mock_managers(self):
        """Mock managers for integration."""
        with patch('core.order_manager.ReliabilityManager') as mock_reliability, \
             patch('core.order_manager.PortfolioManager') as mock_portfolio, \
             patch('core.order_manager.OrderProcessor') as mock_processor:

            mock_reliability_instance = MagicMock()
            mock_reliability_instance.safe_mode_active = False
            mock_reliability_instance.retry_async = AsyncMock(side_effect=lambda func, **kwargs: func())

            mock_portfolio_instance = MagicMock()
            mock_portfolio_instance.paper_balances = {"USDT": Decimal("9500")}
            mock_portfolio_instance.set_initial_balance = MagicMock()
            mock_portfolio_instance.initialize_portfolio = MagicMock()

            mock_processor_instance = MagicMock()
            mock_processor_instance.process_order = AsyncMock(return_value={
                "id": "processed_integration_order",
                "symbol": "BTC/USDT",
                "status": "filled",
                "amount": 1.0,
                "price": 50000.0,
                "cost": 50000.0,
                "pnl": 0.0
            })
            mock_processor_instance.open_orders = {}
            mock_processor_instance.closed_orders = {}
            mock_processor_instance.positions = {}
            mock_processor_instance.get_active_order_count = MagicMock(return_value=0)
            mock_processor_instance.get_open_position_count = MagicMock(return_value=0)

            mock_reliability.return_value = mock_reliability_instance
            mock_portfolio.return_value = mock_portfolio_instance
            mock_processor.return_value = mock_processor_instance

            yield {
                'reliability': mock_reliability_instance,
                'portfolio': mock_portfolio_instance,
                'processor': mock_processor_instance
            }

    @pytest.fixture
    def mock_discord_session(self):
        """Mock Discord aiohttp session."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_instance = MagicMock()
            mock_session.return_value = mock_instance

            # Mock response with async context manager
            mock_response = MagicMock()
            mock_response.status = 204
            mock_response.text = AsyncMock(return_value="OK")
            mock_response.json = AsyncMock(return_value={})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.close = AsyncMock()

            yield mock_instance

    @pytest.mark.asyncio
    async def test_signal_to_order_flow(self, config, mock_executors, mock_managers):
        """Test complete flow from signal generation to order execution."""
        # Initialize components
        order_manager = OrderManager(config, TradingMode.PAPER)

        # Create a trading signal
        signal = TradingSignal(
            strategy_id="integration_test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            current_price=Decimal("50000")
        )

        # Execute the order
        order_result = await order_manager.execute_order(signal)

        # Verify the order was processed
        assert order_result is not None
        assert order_result["id"] == "processed_integration_order"
        assert order_result["symbol"] == "BTC/USDT"
        assert order_result["status"] == "filled"

        # Verify executor was called
        mock_executors['paper'].execute_paper_order.assert_called_once_with(signal)
        mock_managers['processor'].process_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_signal_to_notification_flow(self, config, mock_executors, mock_managers, mock_discord_session):
        """Test complete flow from signal to Discord notification."""
        # Initialize components
        order_manager = OrderManager(config, TradingMode.PAPER)
        discord_config = config["discord"]
        discord_notifier = DiscordNotifier(discord_config)

        # Create and execute a trading signal
        signal = TradingSignal(
            strategy_id="notification_test_strategy",
            symbol="ETH/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("2.0"),
            price=Decimal("3000"),
            current_price=Decimal("3000")
        )

        # Execute order
        order_result = await order_manager.execute_order(signal)

        # Send signal alert
        signal_alert_result = await discord_notifier.send_signal_alert(signal)

        # Send trade alert
        trade_data = {
            "symbol": order_result["symbol"],
            "type": "market",
            "side": "buy",
            "amount": order_result["amount"],
            "price": order_result["price"],
            "pnl": order_result.get("pnl", 0),
            "status": order_result["status"],
            "mode": "paper"
        }
        trade_alert_result = await discord_notifier.send_trade_alert(trade_data)

        # Verify notifications were sent
        assert signal_alert_result is True
        assert trade_alert_result is True

        # Verify Discord API was called twice
        assert mock_discord_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_signal_router_to_order_manager_integration(self, config, mock_executors, mock_managers):
        """Test integration between SignalRouter and OrderManager."""
        # Initialize components
        order_manager = OrderManager(config, TradingMode.PAPER)
        risk_manager = DummyRiskManager()
        signal_router = SignalRouter(risk_manager=risk_manager)

        # Create a signal
        signal = TradingSignal(
            strategy_id="router_integration_test",
            symbol="ADA/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("100.0"),
            price=Decimal("1.50"),
            current_price=Decimal("1.50")
        )

        # Process signal through router
        router_result = await signal_router.process_signal(signal)

        # Verify signal was approved
        assert router_result is not None

        # Execute order through order manager
        order_result = await order_manager.execute_order(signal)

        # Verify order execution
        assert order_result is not None
        assert order_result["symbol"] == "BTC/USDT"  # Mock returns BTC/USDT
        assert order_result["status"] == "filled"

    @pytest.mark.asyncio
    async def test_portfolio_balance_updates(self, config, mock_executors, mock_managers):
        """Test that portfolio balances are correctly updated after trades."""
        order_manager = OrderManager(config, TradingMode.PAPER)

        # Initial balance check
        initial_balance = await order_manager.get_balance()
        assert initial_balance == Decimal("10000")

        # Execute a trade
        signal = TradingSignal(
            strategy_id="balance_test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("0.01"),  # 0.01 BTC
            price=Decimal("50000")   # at $50000 = $500 cost
        )

        await order_manager.execute_order(signal)

        # Check updated balance with tolerance for decimal precision
        updated_balance = await order_manager.get_balance()
        assert abs(updated_balance - Decimal("9499.99999")) < Decimal("0.0001")

    @pytest.mark.asyncio
    async def test_multiple_signals_concurrent_processing(self, config, mock_executors, mock_managers):
        """Test processing multiple signals concurrently."""
        order_manager = OrderManager(config, TradingMode.PAPER)

        # Create multiple signals
        signals = []
        for i in range(5):
            signal = TradingSignal(
                strategy_id=f"concurrent_test_{i}",
                symbol=f"ALT{i}/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type="market",
                amount=Decimal("1.0"),
                price=Decimal("10.0")
            )
            signals.append(signal)

        # Execute orders concurrently
        tasks = [order_manager.execute_order(signal) for signal in signals]
        results = await asyncio.gather(*tasks)

        # Verify all orders were processed
        assert len(results) == 5
        for result in results:
            assert result is not None
            assert result["status"] == "filled"

        # Verify executor was called for each signal
        assert mock_executors['paper'].execute_paper_order.call_count == 5

    @pytest.mark.asyncio
    async def test_error_handling_in_integration_flow(self, config, mock_executors, mock_managers):
        """Test error handling in the integration flow."""
        order_manager = OrderManager(config, TradingMode.PAPER)

        # Mock executor to raise an exception
        mock_executors['paper'].execute_paper_order.side_effect = Exception("Exchange error")

        signal = TradingSignal(
            strategy_id="error_test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        # Execute order - should handle the error gracefully
        order_result = await order_manager.execute_order(signal)

        # Should return None on error
        assert order_result is None

        # Verify error was recorded
        mock_managers['reliability'].record_critical_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_safe_mode_integration(self, config, mock_executors, mock_managers):
        """Test safe mode integration across components."""
        order_manager = OrderManager(config, TradingMode.PAPER)

        # Activate safe mode
        mock_managers['reliability'].safe_mode_active = True

        signal = TradingSignal(
            strategy_id="safe_mode_test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        # Execute order - should be skipped due to safe mode
        order_result = await order_manager.execute_order(signal)

        # Should return skipped result
        assert order_result is not None
        assert order_result["status"] == "skipped"
        assert order_result["reason"] == "safe_mode_active"

        # Verify no actual execution happened
        mock_executors['paper'].execute_paper_order.assert_not_called()
