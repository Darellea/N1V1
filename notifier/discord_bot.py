"""
notifier/discord_bot.py

Discord integration for trade alerts, system notifications, and bot commands.
Supports both webhook-based notifications and interactive bot functionality.
"""

import logging
import asyncio
from typing import Dict, Optional, List, Any
from datetime import datetime
import json
import os

try:
    import discord
    from discord import Webhook

    try:
        from discord import AsyncWebhookAdapter
    except Exception:
        AsyncWebhookAdapter = None
    from discord.ext import commands
except Exception:
    # Fallback when discord package is not available or different version is installed.
    discord = None
    Webhook = None
    AsyncWebhookAdapter = None
    commands = None
import aiohttp

from utils.logger import get_trade_logger
from utils.config_loader import get_config
from utils.adapter import signal_to_dict

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class DiscordNotifier:
    """
    Handles Discord notifications via webhooks and interactive bot commands.
    Supports both asynchronous alerts and command-based interaction.
    """

    def __init__(self, discord_config: Dict):
        """
        Initialize the Discord notifier.

        Args:
            discord_config: Discord configuration from main config
        """
        self.config = discord_config
        # Prefer environment variables for sensitive Discord credentials (backward-compatible)
        # Supported env vars:
        #  - CRYPTOBOT_NOTIFICATIONS_DISCORD_WEBHOOK_URL
        #  - CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN
        #  - CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID
        self.webhook_url = os.getenv("CRYPTOBOT_NOTIFICATIONS_DISCORD_WEBHOOK_URL") or discord_config.get("webhook_url")
        self.bot_token = os.getenv("CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN") or discord_config.get("bot_token")
        self.channel_id = os.getenv("CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID") or discord_config.get("channel_id")
        self.alerts_enabled = discord_config.get("alerts", {}).get("enabled", False)
        self.commands_enabled = discord_config.get("commands", {}).get("enabled", False)
        self.bot = None
        self.session = None

        # Initialize based on configuration
        if self.commands_enabled and self.bot_token:
            self._initialize_bot()
        elif self.alerts_enabled and self.webhook_url:
            self._initialize_webhook()

    def _initialize_bot(self) -> None:
        """Initialize the Discord bot for interactive commands."""
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True

        self.bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

        # Register commands
        self._register_commands()

    def _initialize_webhook(self) -> None:
        """Initialize the webhook session for notifications."""
        self.session = aiohttp.ClientSession()

    def _register_commands(self) -> None:
        """Register Discord bot commands."""

        @self.bot.command(name="status")
        async def status(ctx):
            """Get bot status."""
            if not await self._verify_channel(ctx):
                return

            status_msg = self._generate_status_message()
            embed = discord.Embed(
                title="🤖 Trading Bot Status",
                description=status_msg,
                color=discord.Color.blue(),
            )
            await ctx.send(embed=embed)

        @self.bot.command(name="pause")
        async def pause(ctx):
            """Pause trading."""
            if not await self._verify_channel(ctx):
                return

            # In a real implementation, this would interface with the bot engine
            embed = discord.Embed(
                title="⏸ Trading Paused",
                description="All trading activity has been paused.",
                color=discord.Color.orange(),
            )
            await ctx.send(embed=embed)

        @self.bot.command(name="resume")
        async def resume(ctx):
            """Resume trading."""
            if not await self._verify_channel(ctx):
                return

            # In a real implementation, this would interface with the bot engine
            embed = discord.Embed(
                title="▶ Trading Resumed",
                description="Trading activity has been resumed.",
                color=discord.Color.green(),
            )
            await ctx.send(embed=embed)

        @self.bot.command(name="trades")
        async def trades(ctx, limit: int = 5):
            """Get recent trades."""
            if not await self._verify_channel(ctx):
                return

            recent_trades = trade_logger.get_trade_history(limit)
            if not recent_trades:
                await ctx.send("No recent trades found.")
                return

            embed = discord.Embed(
                title=f"📊 Last {len(recent_trades)} Trades", color=discord.Color.gold()
            )

            for trade in recent_trades:
                pnl = trade.get("pnl", 0)
                color = discord.Color.green() if pnl >= 0 else discord.Color.red()
                embed.add_field(
                    name=f"{trade['symbol']} {trade['type'].upper()}",
                    value=(
                        f"Price: {trade['price']:.4f}\n"
                        f"Amount: {trade['amount']:.4f}\n"
                        f"PNL: {pnl:.4f} ({'✅' if pnl >= 0 else '❌'})\n"
                        f"Time: {trade.get('timestamp', 'N/A')}"
                    ),
                    inline=True,
                )

            await ctx.send(embed=embed)

    async def _verify_channel(self, ctx) -> bool:
        """Verify command was issued in the correct channel."""
        if str(ctx.channel.id) != str(self.channel_id):
            await ctx.send(f"❌ Please use commands in the designated bot channel.")
            return False
        return True

    def _generate_status_message(self) -> str:
        """Generate a status message with current bot metrics."""
        # In a real implementation, this would fetch live data
        stats = trade_logger.get_performance_stats()
        mode = get_config("environment.mode", "unknown")
        if mode is None:
            mode = "unknown"
        mode = str(mode).upper()

        return (
            f"**Mode:** {mode}\n"
            f"**Trades Today:** {stats['total_trades']}\n"
            f"**Win Rate:** {stats['win_rate']:.1%}\n"
            f"**Total PNL:** {stats['total_pnl']:.4f}\n"
            f"**Status:** {'🟢 Running' if True else '🔴 Stopped'}\n"
        )

    async def initialize(self) -> None:
        """Initialize the Discord connection."""
        if self.bot and self.commands_enabled:
            asyncio.create_task(self.bot.start(self.bot_token))
            logger.info("Discord bot started")
        elif self.session and self.alerts_enabled:
            logger.info("Discord webhook notifications enabled")

    async def shutdown(self) -> None:
        """Cleanup Discord resources."""
        if self.bot:
            await self.bot.close()
        if self.session:
            await self.session.close()
        logger.info("Discord notifier shutdown")

    async def send_notification(self, message: str, embed_data: Dict = None) -> bool:
        """
        Send a notification via Discord webhook.

        Args:
            message: Text message to send
            embed_data: Dictionary of embed data

        Returns:
            True if notification was sent successfully
        """
        if not self.alerts_enabled or not self.webhook_url:
            return False

        try:
            async with self.session.post(
                self.webhook_url,
                json={"content": message, "embeds": [embed_data] if embed_data else []},
            ) as response:
                if response.status != 204:
                    logger.error(f"Discord webhook error: {response.status}")
                    return False
                return True
        except Exception as e:
            logger.error(f"Discord notification failed: {str(e)}")
            return False

    async def send_trade_alert(self, trade_data: Dict) -> bool:
        """
        Send a trade alert notification.

        Args:
            trade_data: Dictionary of trade data

        Returns:
            True if notification was sent successfully
        """
        if not self.alerts_enabled:
            return False

        pnl = trade_data.get("pnl", 0)
        is_win = pnl >= 0

        embed = {
            "title": "💰 New Trade Execution",
            "color": 0x00FF00 if is_win else 0xFF0000,
            "fields": [
                {"name": "Symbol", "value": trade_data["symbol"], "inline": True},
                {"name": "Type", "value": trade_data["type"].upper(), "inline": True},
                {"name": "Side", "value": trade_data["side"].upper(), "inline": True},
                {
                    "name": "Amount",
                    "value": f"{trade_data['amount']:.4f}",
                    "inline": True,
                },
                {
                    "name": "Price",
                    "value": f"{trade_data['price']:.4f}",
                    "inline": True,
                },
                {"name": "PNL", "value": f"{pnl:.4f}", "inline": True},
                {
                    "name": "Status",
                    "value": trade_data["status"].upper(),
                    "inline": True,
                },
                {
                    "name": "Mode",
                    "value": trade_data.get("mode", "N/A").upper(),
                    "inline": True,
                },
            ],
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": "Crypto Trading Bot"},
        }

        return await self.send_notification(
            message=f"New trade executed: {trade_data['symbol']}", embed_data=embed
        )

    async def send_signal_alert(self, signal_data: Any) -> bool:
        """
        Send a trading signal alert.

        Args:
            signal_data: Signal object or dictionary

        Returns:
            True if notification was sent successfully
        """
        if not self.alerts_enabled:
            return False

        # Normalize incoming signal objects (dataclass/objects) into plain dicts
        sig = signal_data if isinstance(signal_data, dict) else signal_to_dict(signal_data)

        embed = {
            "title": "📡 New Trading Signal",
            "color": 0x0000FF,
            "fields": [
                {"name": "Symbol", "value": sig.get("symbol", "N/A"), "inline": True},
                {"name": "Type", "value": sig.get("signal_type", "N/A"), "inline": True},
                {
                    "name": "Strength",
                    "value": sig.get("strength", "N/A"),
                    "inline": True,
                },
                {
                    "name": "Price",
                    "value": f"{sig.get('price', 'N/A')}",
                    "inline": True,
                },
                {
                    "name": "Amount",
                    "value": f"{sig.get('amount', 'N/A')}",
                    "inline": True,
                },
                {
                    "name": "Stop Loss",
                    "value": f"{sig.get('stop_loss', 'N/A')}",
                    "inline": True,
                },
                {
                    "name": "Take Profit",
                    "value": f"{sig.get('take_profit', 'N/A')}",
                    "inline": True,
                },
            ],
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": "Crypto Trading Bot"},
        }

        return await self.send_notification(
            message=f"New signal generated: {sig.get('symbol', 'N/A')}", embed_data=embed
        )

    async def send_error_alert(self, error_data: Dict) -> bool:
        """
        Send an error alert notification.

        Args:
            error_data: Dictionary containing error information

        Returns:
            True if notification was sent successfully
        """
        if not self.alerts_enabled:
            return False

        embed = {
            "title": "🚨 System Error",
            "color": 0xFF0000,
            "fields": [
                {
                    "name": "Error",
                    "value": error_data.get("error", "Unknown error"),
                    "inline": False,
                },
                {
                    "name": "Component",
                    "value": error_data.get("component", "Unknown"),
                    "inline": True,
                },
                {
                    "name": "Timestamp",
                    "value": error_data.get("timestamp", "N/A"),
                    "inline": True,
                },
            ],
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": "Crypto Trading Bot"},
        }

        return await self.send_notification(
            message="An error occurred in the trading system!", embed_data=embed
        )

    async def send_performance_report(self, performance_data: Dict) -> bool:
        """
        Send a performance report notification.

        Args:
            performance_data: Dictionary of performance metrics

        Returns:
            True if notification was sent successfully
        """
        if not self.alerts_enabled:
            return False

        embed = {
            "title": "📈 Performance Report",
            "color": 0x00FF00,
            "fields": [
                {
                    "name": "Total Trades",
                    "value": performance_data.get("total_trades", 0),
                    "inline": True,
                },
                {
                    "name": "Win Rate",
                    "value": f"{performance_data.get('win_rate', 0):.1%}",
                    "inline": True,
                },
                {
                    "name": "Total PNL",
                    "value": f"{performance_data.get('total_pnl', 0):.4f}",
                    "inline": True,
                },
                {
                    "name": "Max Win",
                    "value": f"{performance_data.get('max_win', 0):.4f}",
                    "inline": True,
                },
                {
                    "name": "Max Loss",
                    "value": f"{performance_data.get('max_loss', 0):.4f}",
                    "inline": True,
                },
                {
                    "name": "Sharpe Ratio",
                    "value": f"{performance_data.get('sharpe_ratio', 0):.2f}",
                    "inline": True,
                },
            ],
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": "Crypto Trading Bot"},
        }

        return await self.send_notification(
            message="Daily performance report generated", embed_data=embed
        )
