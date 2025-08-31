import asyncio
import os
from dotenv import load_dotenv
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from notifier.discord_bot import DiscordNotifier
from core.contracts import TradingSignal, SignalType, SignalStrength
from decimal import Decimal


class MockAsyncContextManager:
    """Custom async context manager for mocking aiohttp responses."""

    def __init__(self, mock_response):
        self.mock_response = mock_response

    async def __aenter__(self):
        return self.mock_response

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

load_dotenv()


class TestDiscordNotifier:
    """Test cases for DiscordNotifier functionality."""

    @pytest.fixture
    def discord_config(self):
        """Basic Discord config for testing."""
        return {
            "alerts": {
                "enabled": True
            },
            "commands": {
                "enabled": False
            },
            "webhook_url": "https://discord.com/api/webhooks/test",
            "bot_token": "test_bot_token",
            "channel_id": "123456789"
        }

    @pytest.fixture
    def mock_aiohttp_session(self):
        """Mock aiohttp ClientSession."""
        mock_instance = MagicMock()
        mock_instance.close = AsyncMock()

        # Mock response
        mock_response = MagicMock()
        mock_response.status = 204
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.json = AsyncMock(return_value={})

        # Mock post method to return the response directly (not a context manager)
        mock_instance.post = AsyncMock(return_value=mock_response)

        yield mock_instance

    @pytest.fixture
    def mock_discord_imports(self):
        """Mock discord imports to avoid import errors."""
        with patch.dict('sys.modules', {
            'discord': MagicMock(),
            'discord.Webhook': MagicMock(),
            'discord.AsyncWebhookAdapter': MagicMock(),
            'discord.ext': MagicMock(),
            'discord.ext.commands': MagicMock(),
        }):
            yield

    def test_init_webhook_mode(self, discord_config, mock_aiohttp_session, mock_discord_imports):
        """Test DiscordNotifier initialization in webhook mode."""
        with patch.dict(os.environ, {}, clear=True):
            notifier = DiscordNotifier(discord_config)

        assert notifier.alerts_enabled is True
        assert notifier.commands_enabled is False
        assert notifier.webhook_url == "https://discord.com/api/webhooks/test"
        assert notifier.bot_token == "test_bot_token"
        assert notifier.channel_id == "123456789"
        assert notifier.bot is None

    def test_init_bot_mode(self):
        """Test DiscordNotifier initialization in bot mode."""
        config = {
            "alerts": {"enabled": True},
            "commands": {"enabled": True},
            "bot_token": "test_token",
            "channel_id": "123456"
        }

        with patch('notifier.discord_bot.discord.Intents') as mock_intents, \
             patch('notifier.discord_bot.commands.Bot') as mock_bot:

            mock_intents_instance = MagicMock()
            mock_intents.default.return_value = mock_intents_instance
            mock_bot_instance = MagicMock()
            mock_bot.return_value = mock_bot_instance

            notifier = DiscordNotifier(config)

            assert notifier.commands_enabled is True
            assert notifier.bot is not None
            mock_bot.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_notification_webhook_success(self, discord_config, mock_aiohttp_session, mock_discord_imports):
        """Test successful notification sending via webhook."""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            notifier = DiscordNotifier(discord_config)
            await notifier.initialize()

            result = await notifier.send_notification("Test message")

            assert result is True
            mock_aiohttp_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_notification_bot_rest_success(self, mock_aiohttp_session, mock_discord_imports):
        """Test successful notification sending via bot REST API."""
        config = {
            "alerts": {"enabled": True},
            "bot_token": "test_token",
            "channel_id": "123456"
        }
        with patch.dict(os.environ, {
            "CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN": "test_token",
            "CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID": "123456"
        }, clear=False), \
             patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            notifier = DiscordNotifier(config)

            result = await notifier.send_notification("Test message")

            assert result is True
            mock_aiohttp_session.post.assert_called_once()
            # Verify the URL contains the channel endpoint
            call_args = mock_aiohttp_session.post.call_args
            assert "channels/123456/messages" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_send_notification_with_embed(self, discord_config, mock_aiohttp_session, mock_discord_imports):
        """Test notification sending with embed data."""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            notifier = DiscordNotifier(discord_config)
            await notifier.initialize()

            embed_data = {
                "title": "Test Embed",
                "description": "Test description",
                "color": 0x00FF00
            }

            result = await notifier.send_notification("Test message", embed_data)

            assert result is True
            call_args = mock_aiohttp_session.post.call_args
            payload = call_args[1]['json']
            assert payload['content'] == "Test message"
            assert payload['embeds'] == [embed_data]

    @pytest.mark.asyncio
    async def test_send_notification_disabled_alerts(self, discord_config, mock_aiohttp_session, mock_discord_imports):
        """Test that notifications are not sent when alerts are disabled."""
        discord_config["alerts"]["enabled"] = False
        notifier = DiscordNotifier(discord_config)

        result = await notifier.send_notification("Test message")

        assert result is False
        mock_aiohttp_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_notification_rate_limit_retry(self, discord_config, mock_aiohttp_session, mock_discord_imports):
        """Test handling of rate limit with retry."""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            notifier = DiscordNotifier(discord_config)
            await notifier.initialize()

            # Mock rate limit response
            mock_response_429 = MagicMock()
            mock_response_429.status = 429
            mock_response_429.json = AsyncMock(return_value={"retry_after": 1.0})
            mock_response_429.text = AsyncMock(return_value="Rate limited")

            # Mock success response
            mock_response_204 = MagicMock()
            mock_response_204.status = 204
            mock_response_204.text = AsyncMock(return_value="OK")
            mock_response_204.json = AsyncMock(return_value={})

            # First call returns 429, second succeeds
            mock_aiohttp_session.post.side_effect = [mock_response_429, mock_response_204]

            result = await notifier.send_notification("Test message")

            assert result is True
            assert mock_aiohttp_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_send_notification_server_error_retry(self, discord_config, mock_aiohttp_session, mock_discord_imports):
        """Test handling of server errors with retry."""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            notifier = DiscordNotifier(discord_config)
            await notifier.initialize()

            # Mock server error response
            mock_response_500 = MagicMock()
            mock_response_500.status = 500
            mock_response_500.text = AsyncMock(return_value="Internal Server Error")
            mock_response_500.json = AsyncMock(return_value={})

            # Mock success response
            mock_response_204 = MagicMock()
            mock_response_204.status = 204
            mock_response_204.text = AsyncMock(return_value="OK")
            mock_response_204.json = AsyncMock(return_value={})

            # First call returns 500, second succeeds
            mock_aiohttp_session.post.side_effect = [mock_response_500, mock_response_204]

            result = await notifier.send_notification("Test message")

            assert result is True
            assert mock_aiohttp_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_send_notification_max_retries_exceeded(self, discord_config, mock_aiohttp_session, mock_discord_imports):
        """Test that notification fails after max retries."""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            notifier = DiscordNotifier(discord_config)
            await notifier.initialize()

            # Mock persistent failure
            mock_response = MagicMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")
            mock_response.json = AsyncMock(return_value={})
            mock_aiohttp_session.post.return_value = mock_response

            result = await notifier.send_notification("Test message")

            assert result is False
            assert mock_aiohttp_session.post.call_count == 5  # max_retries

    @pytest.mark.asyncio
    async def test_send_trade_alert(self, discord_config, mock_aiohttp_session, mock_discord_imports):
        """Test sending trade alert notification."""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            notifier = DiscordNotifier(discord_config)
            await notifier.initialize()

            trade_data = {
                "symbol": "BTC/USDT",
                "type": "market",
                "side": "buy",
                "amount": 1.0,
                "price": 50000.0,
                "pnl": 100.0,
                "status": "filled",
                "mode": "live"
            }

            result = await notifier.send_trade_alert(trade_data)

            assert result is True
            call_args = mock_aiohttp_session.post.call_args
            payload = call_args[1]['json']
            assert "New Trade Execution" in payload['embeds'][0]['title']
            assert payload['embeds'][0]['fields'][0]['name'] == "Symbol"
            assert payload['embeds'][0]['fields'][0]['value'] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_send_signal_alert(self, discord_config, mock_aiohttp_session, mock_discord_imports):
        """Test sending signal alert notification."""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            notifier = DiscordNotifier(discord_config)
            await notifier.initialize()

            signal = TradingSignal(
                strategy_id="test_strategy",
                symbol="BTC/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type="market",
                amount=Decimal("1.0"),
                price=Decimal("50000")
            )

            result = await notifier.send_signal_alert(signal)

            assert result is True
            call_args = mock_aiohttp_session.post.call_args
            payload = call_args[1]['json']
            assert "New Trading Signal" in payload['embeds'][0]['title']
            assert payload['embeds'][0]['fields'][0]['name'] == "Symbol"
            assert payload['embeds'][0]['fields'][0]['value'] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_send_error_alert(self, discord_config, mock_aiohttp_session, mock_discord_imports):
        """Test sending error alert notification."""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            notifier = DiscordNotifier(discord_config)
            await notifier.initialize()

            error_data = {
                "error": "Test error",
                "component": "OrderManager",
                "timestamp": "2023-01-01T00:00:00Z"
            }

            result = await notifier.send_error_alert(error_data)

            assert result is True
            call_args = mock_aiohttp_session.post.call_args
            payload = call_args[1]['json']
            assert payload['embeds'][0]['title'] == "🚨 System Error"
            assert payload['embeds'][0]['fields'][0]['name'] == "Error"
            assert payload['embeds'][0]['fields'][0]['value'] == "Test error"

    @pytest.mark.asyncio
    async def test_send_performance_report(self, discord_config, mock_aiohttp_session, mock_discord_imports):
        """Test sending performance report notification."""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            notifier = DiscordNotifier(discord_config)
            await notifier.initialize()

            performance_data = {
                "total_trades": 100,
                "win_rate": 0.65,
                "total_pnl": 5000.0,
                "max_win": 1000.0,
                "max_loss": -500.0,
                "sharpe_ratio": 1.2
            }

            result = await notifier.send_performance_report(performance_data)

            assert result is True
            call_args = mock_aiohttp_session.post.call_args
            payload = call_args[1]['json']
            assert payload['embeds'][0]['title'] == "📈 Performance Report"
            assert payload['embeds'][0]['fields'][0]['name'] == "Total Trades"
            assert payload['embeds'][0]['fields'][0]['value'] == 100

    @pytest.mark.asyncio
    async def test_initialize_bot_mode(self):
        """Test bot initialization."""
        config = {
            "alerts": {"enabled": True},
            "commands": {"enabled": True},
            "bot_token": "test_token",
            "channel_id": "123456"
        }

        with patch.dict(os.environ, {
            "CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN": "test_token",
            "CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID": "123456"
        }, clear=False), \
             patch('notifier.discord_bot.discord.Intents') as mock_intents, \
             patch('notifier.discord_bot.commands.Bot') as mock_bot:

            mock_intents_instance = MagicMock()
            mock_intents.default.return_value = mock_intents_instance
            mock_bot_instance = MagicMock()
            mock_bot_instance.start = AsyncMock()
            mock_bot.return_value = mock_bot_instance

            notifier = DiscordNotifier(config)

            await notifier.initialize()

            mock_bot_instance.start.assert_called_once_with("test_token")

    @pytest.mark.asyncio
    async def test_shutdown(self, discord_config, mock_aiohttp_session, mock_discord_imports):
        """Test shutdown functionality."""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            notifier = DiscordNotifier(discord_config)
            # Initialize the notifier to create the session
            await notifier.initialize()
            # Ensure the mock has the closed attribute set to False to trigger close
            notifier.session.closed = False
            # Patch the close method to be an AsyncMock to track calls
            notifier.session.close = AsyncMock()

            await notifier.shutdown()

            notifier.session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_bot(self):
        """Test shutdown with bot."""
        config = {
            "alerts": {"enabled": True},
            "commands": {"enabled": True},
            "bot_token": os.getenv("CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN", "test_token")
        }

        with patch('notifier.discord_bot.discord.Intents') as mock_intents, \
             patch('notifier.discord_bot.commands.Bot') as mock_bot:

            mock_intents_instance = MagicMock()
            mock_intents.default.return_value = mock_intents_instance
            mock_bot_instance = MagicMock()
            mock_bot_instance.logout = AsyncMock()
            mock_bot_instance.close = AsyncMock()
            mock_bot.return_value = mock_bot_instance

            notifier = DiscordNotifier(config)
            notifier._bot_task = AsyncMock()

            await notifier.shutdown()

            mock_bot_instance.logout.assert_called_once()
            mock_bot_instance.close.assert_called_once()
