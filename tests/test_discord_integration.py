import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from notifier.discord_bot import DiscordNotifier
from core.contracts import TradingSignal, SignalType, SignalStrength
from decimal import Decimal


# Skip all tests if aiohttp is not available (for CI environments without discord dependencies)
aiohttp = pytest.importorskip("aiohttp")

"""
Discord Integration Tests

These tests mock aiohttp requests by default to avoid hitting Discord APIs in CI.
To run live tests against real Discord APIs, set DISCORD_LIVE_TEST=true:

    DISCORD_LIVE_TEST=true pytest tests/test_discord_integration.py

Or run only live tests:

    pytest -m discord_live tests/test_discord_integration.py

Note: Live tests require valid Discord tokens and channel IDs in environment variables
or .env file.
"""


@pytest.fixture
def discord_webhook_notifier(discord_webhook_config):
    """Fixture to create DiscordNotifier instances for webhook tests."""
    notifier = DiscordNotifier(discord_webhook_config)
    yield notifier


@pytest.fixture
def discord_bot_integration_notifier(discord_bot_config):
    """Fixture to create DiscordNotifier instances for bot integration tests."""
    with patch.dict(os.environ, {
        "CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN": discord_bot_config["bot_token"],
        "CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID": discord_bot_config["channel_id"]
    }, clear=False), \
         patch('notifier.discord_bot.discord.Intents') as mock_intents, \
         patch('notifier.discord_bot.commands.Bot') as mock_bot:

        mock_intents_instance = MagicMock()
        mock_intents.default.return_value = mock_intents_instance
        mock_bot_instance = MagicMock()
        mock_bot_instance.start = AsyncMock()
        mock_bot_instance.logout = AsyncMock()
        mock_bot_instance.close = AsyncMock()
        mock_bot.return_value = mock_bot_instance

        notifier = DiscordNotifier(discord_bot_config)
        # Initialize synchronously for fixture
        notifier.bot = mock_bot_instance
        yield notifier


@pytest.mark.integration
class TestDiscordIntegration:
    """Integration tests for Discord API interactions."""

    @pytest.fixture
    def discord_webhook_config(self):
        """Discord webhook configuration for integration testing."""
        return {
            "alerts": {
                "enabled": True
            },
            "commands": {
                "enabled": False
            },
            "webhook_url": os.getenv("TEST_DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/test"),
            "bot_token": "test_bot_token",
            "channel_id": "123456789"
        }

    @pytest.fixture
    def discord_bot_config(self):
        """Discord bot configuration for integration testing."""
        return {
            "alerts": {
                "enabled": True
            },
            "commands": {
                "enabled": True
            },
            "bot_token": os.getenv("CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN", "test_bot_token"),
            "channel_id": os.getenv("CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID", "123456789")
        }

    @pytest.fixture
    def sample_trade_data(self):
        """Sample trade data for testing."""
        return {
            "symbol": "BTC/USDT",
            "type": "market",
            "side": "buy",
            "amount": 1.0,
            "price": 50000.0,
            "pnl": 100.0,
            "status": "filled",
            "mode": "live"
        }

    @pytest.fixture
    def sample_signal(self):
        """Sample trading signal for testing."""
        return TradingSignal(
            strategy_id="integration_test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0"),
            price=Decimal("50000")
        )

    @pytest.fixture
    def sample_error_data(self):
        """Sample error data for testing."""
        return {
            "error": "Integration test error",
            "component": "DiscordIntegrationTest",
            "timestamp": "2023-01-01T00:00:00Z"
        }

    @pytest.fixture
    def sample_performance_data(self):
        """Sample performance data for testing."""
        return {
            "total_trades": 100,
            "win_rate": 0.65,
            "total_pnl": 5000.0,
            "max_win": 1000.0,
            "max_loss": -500.0,
            "sharpe_ratio": 1.2
        }

    @pytest.mark.asyncio
    async def test_webhook_integration_basic_notification(self, discord_webhook_notifier):
        """Test basic notification sending via webhook integration."""
        result = await discord_webhook_notifier.send_notification("Integration Test: Basic notification")

        assert result is True

    @pytest.mark.asyncio
    async def test_webhook_integration_with_embed(self, discord_webhook_notifier):
        """Test notification with embed via webhook integration."""
        embed_data = {
            "title": "Integration Test Embed",
            "description": "Testing embed functionality",
            "color": 0x00FF00,
            "fields": [
                {"name": "Test Field", "value": "Test Value", "inline": True}
            ]
        }

        result = await discord_webhook_notifier.send_notification("Integration Test: With embed", embed_data)

        assert result is True

    @pytest.mark.asyncio
    async def test_webhook_integration_trade_alert(self, discord_webhook_notifier, sample_trade_data):
        """Test trade alert via webhook integration."""
        result = await discord_webhook_notifier.send_trade_alert(sample_trade_data)

        assert result is True

    @pytest.mark.asyncio
    async def test_webhook_integration_signal_alert(self, discord_webhook_notifier, sample_signal):
        """Test signal alert via webhook integration."""
        result = await discord_webhook_notifier.send_signal_alert(sample_signal)

        assert result is True

    @pytest.mark.asyncio
    async def test_webhook_integration_error_alert(self, discord_webhook_notifier, sample_error_data):
        """Test error alert via webhook integration."""
        result = await discord_webhook_notifier.send_error_alert(sample_error_data)

        assert result is True

    @pytest.mark.asyncio
    async def test_webhook_integration_performance_report(self, discord_webhook_notifier, sample_performance_data):
        """Test performance report via webhook integration."""
        result = await discord_webhook_notifier.send_performance_report(sample_performance_data)

        assert result is True

    @pytest.mark.asyncio
    async def test_bot_integration_basic_notification(self, discord_bot_integration_notifier):
        """Test basic notification sending via bot integration."""
        result = await discord_bot_integration_notifier.send_notification("Integration Test: Bot notification")

        assert result is True

    @pytest.mark.asyncio
    async def test_bot_integration_trade_alert(self, discord_bot_integration_notifier, sample_trade_data):
        """Test trade alert via bot integration."""
        result = await discord_bot_integration_notifier.send_trade_alert(sample_trade_data)

        assert result is True

    @pytest.mark.asyncio
    async def test_webhook_integration_rate_limit_handling(self, discord_webhook_notifier):
        """Test rate limit handling in webhook integration."""
        # Send multiple notifications quickly to potentially trigger rate limits
        tasks = []
        for i in range(10):
            task = discord_webhook_notifier.send_notification(f"Rate limit test {i}")
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # At least some should succeed
        success_count = sum(1 for result in results if result is True)
        assert success_count > 0

    @pytest.mark.asyncio
    async def test_webhook_integration_error_recovery(self, discord_webhook_config):
        """Test error recovery in webhook integration."""
        # Temporarily patch aiohttp to return error response
        import aiohttp

        async def mock_post_error(self, url, **kwargs):
            """Mock POST method that returns an error response."""
            mock_response = MagicMock()
            mock_response.status = 500  # Server error
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json = AsyncMock(return_value={})
            mock_response.text = AsyncMock(return_value="Internal Server Error")
            mock_response.close = AsyncMock()
            return mock_response

        # Temporarily replace the mock post method
        original_post = aiohttp.ClientSession.post
        aiohttp.ClientSession.post = mock_post_error

        try:
            notifier = DiscordNotifier(discord_webhook_config)
            result = await notifier.send_notification("Error recovery test")
            # Should handle the error gracefully
            assert result is False
        finally:
            # Restore original method
            aiohttp.ClientSession.post = original_post

    @pytest.mark.asyncio
    async def test_webhook_integration_large_payload(self, discord_webhook_notifier):
        """Test handling of large payloads in webhook integration."""
        # Create a large embed with many fields
        large_embed = {
            "title": "Large Payload Test",
            "description": "Testing with a large number of fields",
            "color": 0xFF0000,
            "fields": []
        }

        # Add many fields to create a large payload
        for i in range(25):  # Discord allows up to 25 fields
            large_embed["fields"].append({
                "name": f"Field {i}",
                "value": f"Value {i} - This is a test value for field {i}",
                "inline": True
            })

        result = await discord_webhook_notifier.send_notification("Large payload test", large_embed)

        assert result is True

    @pytest.mark.asyncio
    async def test_webhook_integration_special_characters(self, discord_webhook_notifier):
        """Test handling of special characters in webhook integration."""
        special_message = "Special chars: éñüñ 中文 🚀 💯 🔥 🔥 🔥"
        result = await discord_webhook_notifier.send_notification(special_message)
        assert result is True

    @pytest.mark.asyncio
    async def test_webhook_integration_empty_content(self, discord_webhook_notifier):
        """Test handling of empty content in webhook integration."""
        result = await discord_webhook_notifier.send_notification("")
        assert result is True

    @pytest.mark.asyncio
    async def test_webhook_integration_disabled_alerts(self, discord_webhook_config):
        """Test disabled alerts in webhook integration."""
        discord_webhook_config["alerts"]["enabled"] = False
        notifier = DiscordNotifier(discord_webhook_config)

        result = await notifier.send_notification("Disabled alerts test")

        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.discord_live
    async def test_live_discord_webhook_integration(self, discord_webhook_config):
        """Test actual Discord webhook integration when DISCORD_LIVE_TEST=true."""
        # This test will only run when DISCORD_LIVE_TEST=true is set
        # and will make real HTTP calls to Discord
        notifier = DiscordNotifier(discord_webhook_config)

        result = await notifier.send_notification("Live Discord Integration Test")

        # In live mode, this should either succeed or fail based on actual Discord response
        # We can't assert True here since it depends on valid credentials
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_bot_integration_initialization(self, discord_bot_integration_notifier):
        """Test bot initialization in integration."""
        # The fixture already handles initialization, so we just verify it was set up correctly
        assert discord_bot_integration_notifier.bot is not None

    @pytest.mark.asyncio
    async def test_bot_integration_shutdown(self, discord_bot_integration_notifier):
        """Test bot shutdown in integration."""
        # Create a mock task for the bot
        mock_task = AsyncMock()
        mock_task.done.return_value = False
        mock_task.cancel = MagicMock()
        discord_bot_integration_notifier._bot_task = mock_task

        # Mock the bot's logout and close methods to be AsyncMock
        discord_bot_integration_notifier.bot.logout = AsyncMock()
        discord_bot_integration_notifier.bot.close = AsyncMock()

        await discord_bot_integration_notifier.shutdown()

        # Verify the bot methods were called
        discord_bot_integration_notifier.bot.logout.assert_called_once()
        discord_bot_integration_notifier.bot.close.assert_called_once()
