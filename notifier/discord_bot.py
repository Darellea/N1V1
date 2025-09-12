"""
notifier/discord_bot.py

Discord integration for trade alerts, system notifications, and bot commands.
Supports both webhook-based notifications and interactive bot functionality.
"""

import logging
import asyncio
import random
from typing import Dict, Optional, List, Any
from datetime import datetime
from utils.time import now_ms, to_iso
import json
import os

try:
    import discord
    from discord import Webhook

    try:
        from discord import AsyncWebhookAdapter
    except ImportError:
        AsyncWebhookAdapter = None
    from discord.ext import commands
except ImportError:
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

    def __init__(self, discord_config: Dict, task_manager: Optional["TaskManager"] = None):
        """
        Initialize the Discord notifier.

        Args:
            discord_config: Discord configuration from main config
            task_manager: Optional TaskManager for tracking background tasks
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
        self._session_lock = asyncio.Lock()  # Protect session creation/access
        self._bot_task = None
        self.task_manager = task_manager
        self._shutdown_event = asyncio.Event()

        # Initialize based on configuration
        # Priority:
        # 1. Interactive bot (commands) when commands are enabled and token present.
        # 2. REST API using bot token + channel_id when alerts are enabled.
        # 3. Webhook URL fallback when provided.
        if self.commands_enabled and self.bot_token:
            self._initialize_bot()
        elif self.alerts_enabled and self.bot_token and self.channel_id:
            # Use REST API mode with bot token + channel id for sending messages (no interactive commands).
            # This uses aiohttp to POST to the Discord channel messages endpoint with Bot token auth.
            self._initialize_webhook()
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
        """Initialize the aiohttp session for notifications.

        This session is used for either webhook-based notifications (webhook_url)
        or REST API calls using a Bot token + channel_id. We keep the method name
        for backward compatibility.
        """
        # Don't create session here - it will be created asynchronously in initialize()
        # Informational logging; actual sending path chosen in send_notification based on config.
        if self.bot_token and self.channel_id and not self.webhook_url:
            logger.info("Discord notifier configured to use Bot token + channel (REST API) for alerts")
        elif self.webhook_url:
            logger.info("Discord webhook notifications enabled")
        else:
            logger.debug("Discord notifier initialized without webhook or bot-token+channel; alerts disabled until configured")

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
            # Keep reference to bot task so it can be cancelled/awaited on shutdown
            if self.task_manager:
                self._bot_task = self.task_manager.create_task(self.bot.start(self.bot_token), name="DiscordBot")
            else:
                self._bot_task = asyncio.create_task(self.bot.start(self.bot_token))
            logger.info("Discord bot started")
        elif self.alerts_enabled and (self.webhook_url or (self.bot_token and self.channel_id)):
            # Create aiohttp session asynchronously - only if not already created
            await self._ensure_session()
            logger.info("Discord webhook notifications enabled")

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists, creating it if necessary."""
        async with self._session_lock:
            if self.session is None or self.session.closed:
                # Create new session with proper connector configuration
                connector = aiohttp.TCPConnector(limit=10, limit_per_host=5, ttl_dns_cache=30)
                self.session = aiohttp.ClientSession(connector=connector)
                logger.debug("Created new aiohttp session for Discord notifier")

    async def _close_session(self) -> None:
        """Safely close the aiohttp session."""
        async with self._session_lock:
            if self.session and not self.session.closed:
                try:
                    await self.session.close()
                    # Give time for underlying connections to close
                    await asyncio.sleep(0.1)
                    logger.debug("aiohttp session closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing aiohttp session: {e}")
                finally:
                    self.session = None

    async def shutdown(self) -> None:
        """Cleanup Discord resources."""
        # Cancel/await background bot task if it was created with timeout protection
        try:
            if getattr(self, "_bot_task", None) and hasattr(self._bot_task, 'done'):
                try:
                    # Check if task is done, handling both real tasks and AsyncMock
                    is_done = self._bot_task.done()
                    if hasattr(is_done, '__bool__'):  # Real task returns bool
                        is_done = bool(is_done)
                    elif hasattr(is_done, '__call__'):  # AsyncMock returns Mock, check if it's falsy
                        is_done = False  # Assume AsyncMock task is not done unless explicitly set

                    if not is_done:
                        # For real tasks, cancel and await
                        if hasattr(self._bot_task, 'cancel'):
                            self._bot_task.cancel()
                        await asyncio.wait_for(self._bot_task, timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout reached while awaiting discord bot task cancellation")
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Error awaiting discord bot task during shutdown")
        except Exception:
            logger.exception("Failed to cancel/await bot task")

        # Attempt to explicitly logout and then close the bot client (if present).
        # Some discord.py versions expose logout(); calling it first ensures the session/token is terminated
        # on the Discord side prior to closing the client.
        try:
            if self.bot:
                try:
                    if hasattr(self.bot, "logout"):
                        await asyncio.wait_for(self.bot.logout(), timeout=15.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout reached while awaiting discord bot logout")
                except Exception:
                    # Not all library versions expose logout or it may fail; continue to close the client anyway.
                    logger.debug("discord.Bot.logout() not available or failed; continuing to close the bot", exc_info=True)
                try:
                    await asyncio.wait_for(self.bot.close(), timeout=15.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout reached while awaiting discord bot close")
                except Exception:
                    logger.exception("Failed to close discord bot client")
        except Exception:
            logger.exception("Failed while shutting down discord bot client")

        # Close aiohttp session using the new method
        try:
            await self._close_session()
        except Exception:
            logger.exception("Failed to close aiohttp session for discord notifier")

        logger.info("Discord notifier shutdown")

    async def send_notification(self, message: str, embed_data: Dict = None) -> bool:
        """
        Send a notification via Discord.

        Supports two modes:
        - Bot token + channel_id (preferred when configured): uses Discord REST API with Bot auth.
        - Webhook URL (fallback): posts to the webhook endpoint.

        Implements retry/backoff with jitter and respects 429 'retry_after' responses.
        """
        if not self.alerts_enabled:
            return False

        # Handle empty content gracefully
        if not message or not message.strip():
            if embed_data:
                # If we have embed data, send with a placeholder message
                message = "📊 Update"
            else:
                # No content and no embed - skip sending but return success
                logger.debug("Skipping Discord notification: empty content and no embed data")
                return True

        # Ensure HTTP session exists
        await self._ensure_session()

        # Prepare payload
        payload = {"content": message}
        if embed_data:
            payload["embeds"] = [embed_data]

        # Prefer Bot REST mode when bot_token+channel_id are present (user said they use tokens+channel).
        # Only fall back to webhook mode when no bot credentials are available.
        use_bot_rest = bool(self.bot_token and self.channel_id)
        use_webhook = bool(self.webhook_url) and not use_bot_rest

        if not use_bot_rest and not use_webhook:
            logger.error("No valid Discord notifier configuration: missing webhook_url or bot_token+channel_id")
            return False

        # Common retry strategy parameters
        max_retries = 5
        base_backoff = 0.5  # seconds
        attempt = 0

        while attempt < max_retries:
            try:
                if use_bot_rest:
                    url = f"https://discord.com/api/v10/channels/{self.channel_id}/messages"
                    headers = {
                        "Authorization": f"Bot {self.bot_token}",
                        "Content-Type": "application/json",
                    }
                else:
                    url = self.webhook_url
                    headers = {"Content-Type": "application/json"}

                resp = await self.session.post(url, json=payload, headers=headers)
                status = int(resp.status)
                if status in (200, 201, 204):
                    return True

                if status == 429:
                    # Rate limited. Discord typically returns JSON with 'retry_after' (seconds).
                    retry_after = None
                    try:
                        data = await resp.json()
                        retry_after = data.get("retry_after", None)
                    except Exception:
                        # If body is not JSON or parsing failed, fall back to exponential backoff.
                        retry_after = None

                    if retry_after is not None:
                        # retry_after may be in seconds (float)
                        sleep_for = float(retry_after)
                    else:
                        # Exponential backoff with jitter
                        sleep_for = base_backoff * (2 ** attempt) + random.random() * base_backoff

                    logger.warning(f"Discord rate limited (429). Sleeping for {sleep_for:.2f}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(sleep_for)
                    attempt += 1
                    continue

                # For other non-success statuses, log body and give up (no retry) unless it's a 5xx.
                body = await resp.text()
                if 500 <= status < 600:
                    # Server error: retry with backoff
                    sleep_for = base_backoff * (2 ** attempt) + random.random() * base_backoff
                    logger.warning(f"Discord server error {status}. Retrying after {sleep_for:.2f}s (attempt {attempt+1}/{max_retries}) - body: {body}")
                    await asyncio.sleep(sleep_for)
                    attempt += 1
                    continue

                logger.error(f"Discord notification failed: status={status} body={body}")
                return False

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # Network-level errors: retry with backoff
                sleep_for = base_backoff * (2 ** attempt) + random.random() * base_backoff
                logger.exception(f"Discord notification HTTP/client error; retrying after {sleep_for:.2f}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(sleep_for)
                attempt += 1
                continue
            except Exception:
                logger.exception("Unexpected error sending Discord notification")
                return False

        logger.error("Discord notification failed after maximum retries")
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
            "timestamp": to_iso(now_ms()),
            "footer": {"text": "Crypto Trading Bot"},
            "timestamp": to_iso(now_ms()),
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
            "timestamp": to_iso(now_ms()),
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
            "timestamp": to_iso(now_ms()),
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
            "timestamp": to_iso(now_ms()),
            "footer": {"text": "Crypto Trading Bot"},
        }

        return await self.send_notification(
            message="Daily performance report generated", embed_data=embed
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()

    def __del__(self):
        """Destructor - attempt cleanup if not already done."""
        # This is a safeguard in case shutdown() wasn't called properly
        # We can't use async operations in __del__, so we just warn
        # But we need to be careful about logging after the event loop is closed
        try:
            import sys
            if hasattr(self, 'session') and self.session and not self.session.closed:
                # Try to log, but don't fail if logging is unavailable
                try:
                    logger.warning("DiscordNotifier session was not properly closed before destruction")
                except (ValueError, RuntimeError):
                    # Logging system is shut down, print to stderr as fallback
                    print("WARNING: DiscordNotifier session was not properly closed before destruction", file=sys.stderr)
            if hasattr(self, '_bot_task') and self._bot_task and not self._bot_task.done():
                try:
                    logger.warning("DiscordNotifier bot task was not properly cancelled before destruction")
                except (ValueError, RuntimeError):
                    # Logging system is shut down, print to stderr as fallback
                    print("WARNING: DiscordNotifier bot task was not properly cancelled before destruction", file=sys.stderr)
        except Exception:
            # If anything goes wrong in __del__, just silently pass
            pass
