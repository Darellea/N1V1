"""
tests/test_live_executor.py

Comprehensive tests for LiveOrderExecutor functionality.
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from ccxt.base.errors import NetworkError, ExchangeError

from core.execution.live_executor import LiveOrderExecutor
from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types.order_types import OrderType


class TestLiveOrderExecutor:
    """Test cases for LiveOrderExecutor functionality."""

    @pytest.fixture
    def config(self):
        """Basic config for testing."""
        return {
            "exchange": {
                "name": "binance",
                "default_type": "spot"
            }
        }

    @pytest.fixture
    def mock_exchange(self):
        """Mock CCXT exchange."""
        mock_exch = AsyncMock()
        mock_exch.create_order = AsyncMock()
        mock_exch.close = AsyncMock()
        mock_exch.load_markets = AsyncMock()  # Prevent network calls during init
        return mock_exch

    @pytest.fixture
    def sample_signal(self):
        """Sample trading signal for testing."""
        return TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            price=Decimal("50000.0"),
            current_price=Decimal("50000.0"),
            stop_loss=Decimal("49000.0"),
            take_profit=Decimal("52000.0"),
            metadata={"test": "data"}
        )

    @pytest.fixture
    def sample_signal_dict(self):
        """Sample signal as dictionary for testing."""
        return {
            "symbol": "BTC/USDT",
            "order_type": "market",
            "side": "buy",
            "amount": 1.0,
            "price": 50000.0,
            "params": {"test": "param"}
        }

    def test_init_with_config(self, config):
        """Test LiveOrderExecutor initialization with config."""
        with patch('ccxt.async_support.binance') as mock_exchange_class:
            mock_exchange_instance = MagicMock()
            mock_exchange_class.return_value = mock_exchange_instance

            executor = LiveOrderExecutor(config)

            assert executor.config == config
            assert executor.exchange is not None
            mock_exchange_class.assert_called_once()

    def test_init_with_env_vars(self, config):
        """Test LiveOrderExecutor initialization with environment variables."""
        config_no_creds = {"exchange": {"name": "binance"}}

        with patch.dict('os.environ', {
            'CRYPTOBOT_EXCHANGE_API_KEY': 'test_key',
            'CRYPTOBOT_EXCHANGE_API_SECRET': 'test_secret',
            'CRYPTOBOT_EXCHANGE_API_PASSPHRASE': 'test_pass'
        }):
            with patch('ccxt.async_support.binance') as mock_exchange_class:
                mock_exchange_instance = MagicMock()
                mock_exchange_class.return_value = mock_exchange_instance

                executor = LiveOrderExecutor(config_no_creds)

                # Verify exchange was initialized with env vars
                call_args = mock_exchange_class.call_args[0][0]
                assert call_args['apiKey'] == 'test_key'
                assert call_args['secret'] == 'test_secret'
                assert call_args['password'] == 'test_pass'

    def test_init_exchange_not_found(self, config):
        """Test initialization when exchange class is not found."""
        config_bad_exchange = {"exchange": {"name": "nonexistent_exchange"}}

        with patch('ccxt.async_support.nonexistent_exchange', side_effect=AttributeError, create=True):
            with pytest.raises(AttributeError):
                LiveOrderExecutor(config_bad_exchange)

    @pytest.mark.asyncio
    async def test_execute_live_order_success(self, config, sample_signal, mock_exchange):
        """Test successful live order execution."""
        with patch('ccxt.async_support.binance', return_value=mock_exchange):
            executor = LiveOrderExecutor(config)

            # Mock successful order response
            mock_exchange.create_order.return_value = {
                "id": "test_order_123",
                "status": "filled",
                "amount": 1.0,
                "price": 50000.0,
                "cost": 50000.0
            }

            result = await executor.execute_live_order(sample_signal)

            assert result["id"] == "test_order_123"
            assert result["status"] == "filled"
            mock_exchange.create_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_live_order_dict_signal(self, config, sample_signal_dict, mock_exchange):
        """Test live order execution with dictionary signal."""
        with patch('ccxt.async_support.binance', return_value=mock_exchange):
            executor = LiveOrderExecutor(config)

            mock_exchange.create_order.return_value = {
                "id": "test_order_456",
                "status": "filled"
            }

            result = await executor.execute_live_order(sample_signal_dict)

            assert result["id"] == "test_order_456"
            mock_exchange.create_order.assert_called_once_with(
                "BTC/USDT", "market", "buy", 1.0, 50000.0, {"test": "param"}
            )

    @pytest.mark.asyncio
    async def test_execute_live_order_no_exchange(self, config, sample_signal):
        """Test order execution when exchange is not initialized."""
        with patch('ccxt.async_support.binance', return_value=None):
            executor = LiveOrderExecutor(config)
            executor.exchange = None

            with pytest.raises(RuntimeError, match="Exchange not initialized"):
                await executor.execute_live_order(sample_signal)

    @pytest.mark.asyncio
    async def test_execute_live_order_network_error(self, config, sample_signal, mock_exchange):
        """Test order execution with network error."""
        with patch('ccxt.async_support.binance', return_value=mock_exchange):
            executor = LiveOrderExecutor(config)

            mock_exchange.create_order.side_effect = NetworkError("Connection failed")

            with pytest.raises(NetworkError):
                await executor.execute_live_order(sample_signal)

    @pytest.mark.asyncio
    async def test_execute_live_order_exchange_error(self, config, sample_signal, mock_exchange):
        """Test order execution with exchange error."""
        with patch('ccxt.async_support.binance', return_value=mock_exchange):
            executor = LiveOrderExecutor(config)

            mock_exchange.create_order.side_effect = ExchangeError("Invalid order")

            with pytest.raises(ExchangeError):
                await executor.execute_live_order(sample_signal)

    @pytest.mark.asyncio
    async def test_execute_live_order_unexpected_error(self, config, sample_signal, mock_exchange):
        """Test order execution with unexpected error."""
        with patch('ccxt.async_support.binance', return_value=mock_exchange):
            executor = LiveOrderExecutor(config)

            mock_exchange.create_order.side_effect = ValueError("Unexpected error")

            with pytest.raises(ValueError):
                await executor.execute_live_order(sample_signal)

    @pytest.mark.asyncio
    async def test_create_order_positional_args(self, config, mock_exchange):
        """Test _create_order_on_exchange with positional arguments."""
        with patch('ccxt.async_support.binance', return_value=mock_exchange):
            executor = LiveOrderExecutor(config)

            order_params = {
                "symbol": "BTC/USDT",
                "type": "market",
                "side": "buy",
                "amount": 1.0,
                "price": 50000.0,
                "params": {}
            }

            mock_exchange.create_order.return_value = {"id": "test_order"}

            result = await executor._create_order_on_exchange(order_params)

            assert result["id"] == "test_order"
            # Should try positional args first
            mock_exchange.create_order.assert_called_with(
                "BTC/USDT", "market", "buy", 1.0, 50000.0, {}
            )

    @pytest.mark.asyncio
    async def test_create_order_kwargs_fallback(self, config, mock_exchange):
        """Test _create_order_on_exchange with kwargs fallback."""
        with patch('ccxt.async_support.binance', return_value=mock_exchange):
            executor = LiveOrderExecutor(config)

            order_params = {
                "symbol": "BTC/USDT",
                "type": "market",
                "side": "buy",
                "amount": 1.0,
                "price": 50000.0,
                "params": {}
            }

            # First call raises TypeError, second succeeds
            mock_exchange.create_order.side_effect = [TypeError("Positional args not supported"), {"id": "test_order"}]

            result = await executor._create_order_on_exchange(order_params)

            assert result["id"] == "test_order"
            # Should have tried kwargs after positional failed
            assert mock_exchange.create_order.call_count == 2

    @pytest.mark.asyncio
    async def test_create_order_both_fail(self, config, mock_exchange):
        """Test _create_order_on_exchange when both positional and kwargs fail."""
        with patch('ccxt.async_support.binance', return_value=mock_exchange):
            executor = LiveOrderExecutor(config)

            order_params = {
                "symbol": "BTC/USDT",
                "type": "market",
                "side": "buy",
                "amount": 1.0,
                "price": 50000.0,
                "params": {}
            }

            # Both calls fail
            mock_exchange.create_order.side_effect = TypeError("Both methods failed")

            with pytest.raises(TypeError):
                await executor._create_order_on_exchange(order_params)

    @pytest.mark.asyncio
    async def test_shutdown_with_exchange(self, config, mock_exchange):
        """Test shutdown with active exchange."""
        with patch('ccxt.async_support.binance', return_value=mock_exchange):
            executor = LiveOrderExecutor(config)

            await executor.shutdown()

            mock_exchange.close.assert_called_once()
            assert executor.exchange is None

    @pytest.mark.asyncio
    async def test_shutdown_no_exchange(self, config):
        """Test shutdown when no exchange is initialized."""
        with patch('ccxt.async_support.binance', return_value=None):
            executor = LiveOrderExecutor(config)
            executor.exchange = None

            # Should not raise any errors
            await executor.shutdown()

            assert executor.exchange is None

    @pytest.mark.asyncio
    async def test_execute_live_order_limit_order(self, config, mock_exchange):
        """Test live order execution with limit order."""
        with patch('ccxt.async_support.binance', return_value=mock_exchange):
            executor = LiveOrderExecutor(config)

            signal = TradingSignal(
                strategy_id="test_strategy",
                symbol="ETH/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type=OrderType.LIMIT,
                amount=Decimal("2.0"),
                price=Decimal("3000.0"),
                current_price=Decimal("3000.0")
            )

            mock_exchange.create_order.return_value = {
                "id": "limit_order_123",
                "status": "open",
                "amount": 2.0,
                "price": 3000.0
            }

            result = await executor.execute_live_order(signal)

            assert result["id"] == "limit_order_123"
            mock_exchange.create_order.assert_called_once_with(
                "ETH/USDT", "limit", "buy", 2.0, 3000.0, {}
            )

    @pytest.mark.asyncio
    async def test_execute_live_order_sell_signal(self, config, mock_exchange):
        """Test live order execution with sell signal."""
        with patch('ccxt.async_support.binance', return_value=mock_exchange):
            executor = LiveOrderExecutor(config)

            signal = TradingSignal(
                strategy_id="test_strategy",
                symbol="BTC/USDT",
                signal_type=SignalType.EXIT_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type=OrderType.MARKET,
                amount=Decimal("1.0"),
                price=Decimal("51000.0"),
                current_price=Decimal("51000.0")
            )

            mock_exchange.create_order.return_value = {
                "id": "sell_order_123",
                "status": "filled",
                "amount": 1.0,
                "price": 51000.0
            }

            result = await executor.execute_live_order(signal)

            assert result["id"] == "sell_order_123"
            mock_exchange.create_order.assert_called_once_with(
                "BTC/USDT", "market", "sell", 1.0, 51000.0, {}
            )

    @pytest.mark.asyncio
    async def test_execute_live_order_with_params(self, config, mock_exchange):
        """Test live order execution with additional parameters."""
        with patch('ccxt.async_support.binance', return_value=mock_exchange):
            executor = LiveOrderExecutor(config)

            signal = TradingSignal(
                strategy_id="test_strategy",
                symbol="BTC/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type=OrderType.MARKET,
                amount=Decimal("1.0"),
                price=Decimal("50000.0"),
                current_price=Decimal("50000.0"),
                metadata={"exchange_specific": {"timeInForce": "GTC"}}
            )

            mock_exchange.create_order.return_value = {"id": "test_order"}

            result = await executor.execute_live_order(signal)

            assert result["id"] == "test_order"
            # Verify params were passed
            call_args = mock_exchange.create_order.call_args[0]
            assert call_args[5] == {"exchange_specific": {"timeInForce": "GTC"}}
