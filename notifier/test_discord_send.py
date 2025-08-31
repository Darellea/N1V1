import os
import sys
import asyncio
import traceback
import logging

# Ensure repository root is on sys.path so 'notifier' package can be imported when running this script directly.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from notifier.discord_bot import DiscordNotifier

# Load .env (simple parser) so environment variables in .env are available to the test.
env_path = ".env"
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                # Do not overwrite existing environment vars, just set defaults from .env
                os.environ.setdefault(k, v)

# Build a minimal discord config. The DiscordNotifier prefers environment variables for tokens,
# but we pass a config dict for other flags.
discord_config = {
    "alerts": {"enabled": True},
    "commands": {"enabled": False},
    # webhook_url, bot_token, channel_id will be read from env if present
    "webhook_url": os.getenv("CRYPTOBOT_NOTIFICATIONS_DISCORD_WEBHOOK_URL"),
    "bot_token": os.getenv("CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN"),
    "channel_id": os.getenv("CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID"),
}

async def main():
    logger = logging.getLogger(__name__)
    logger.info("Using configuration:")
    logger.info("  webhook_url: %s", bool(discord_config.get("webhook_url")))
    logger.info("  bot_token: %s", bool(discord_config.get("bot_token")))
    logger.info("  channel_id: %s", bool(discord_config.get("channel_id")))

    notifier = DiscordNotifier(discord_config)

    try:
        await notifier.initialize()
        logger.info("Notifier initialized.")
    except Exception:
        logger.error("Failed to initialize notifier:")
        traceback.print_exc()
        return

    try:
        ok = await notifier.send_notification(
            "TESTING FROM THE REPO ⚠️⚠️⚠️",
            embed_data={"title": "Testing from N1V1", "description": "If You Read This. The Module Works!"},
        )
        logger.info("send_notification returned: %s", ok)
    except Exception:
        logger.error("send_notification raised an exception:")
        traceback.print_exc()
    finally:
        try:
            await notifier.shutdown()
            logger.info("Notifier shutdown completed.")
        except Exception:
            logger.error("Notifier.shutdown raised an exception:")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        traceback.print_exc()
