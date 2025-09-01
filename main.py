#!/usr/bin/env python3
"""
Main entry point for the crypto trading bot system.
Handles initialization, mode selection, and core system startup.
"""

import asyncio
import logging
import sys
import argparse
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from utils.config_loader import load_config, ConfigLoader
from utils.logger import setup_logging
from core.bot_engine import BotEngine

# Rich console instance for pretty printing
console = Console()


class CryptoTradingBot:
    """
    Main class for the crypto trading bot system.
    Handles initialization and manages the trading modes.
    """

    def __init__(self):
        """Initialize the bot with configuration and logging."""
        self.config: Optional[dict] = None
        self.bot_engine: Optional[BotEngine] = None
        self.logger: logging.Logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """
        Initialize the bot by loading configuration and setting up systems.
        """
        try:
            # Display startup banner
            self._display_banner()

            # Load configuration
            self.config = load_config()
            setup_logging(self.config.get("logging", {}))

            self.logger.info("Initializing CryptoTradingBot")
            self.logger.info("Configuration loaded successfully")

            # Initialize core engine
            self.bot_engine = BotEngine(self.config)
            await self.bot_engine.initialize()

            self.logger.info("Bot engine initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize bot: {str(e)}", exc_info=True)
            self.logger.error(f"Initialization failed: {str(e)}")
            sys.exit(1)

    async def run(self) -> None:
        """
        Main execution method for the trading bot.
        """
        if not self.bot_engine:
            self.logger.error("Bot engine not initialized")
            return

        try:
            self.logger.info("Starting CryptoTradingBot")
            self.logger.info("Starting trading bot...")

            # Start the main bot engine
            await self.bot_engine.run()

        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
            self.logger.warning("Bot stopped by user")
        except Exception as e:
            self.logger.error(f"Bot crashed: {str(e)}", exc_info=True)
            self.logger.error(f"Bot crashed: {str(e)}")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """
        Cleanup and shutdown the bot gracefully.
        """
        self.logger.info("Shutting down bot")
        self.logger.warning("Shutting down bot...")

        if self.bot_engine:
            await self.bot_engine.shutdown()

        self.logger.info("Bot shutdown complete")

    def _display_banner(self) -> None:
        """Display the startup banner."""
        banner_text = Text.assemble(
            ("CryptoTradingBot\n", "bold blue"),
            ("Version: 1.0.0\n", "bold green"),
            ("Mode: ", "bold"),
            (f"{self._get_mode()}\n", "bold cyan"),
            ("Status: ", "bold"),
            ("INITIALIZING", "bold yellow"),
        )

        panel = Panel(
            banner_text,
            title="[bold]Crypto Trading System[/bold]",
            subtitle="[italic]Secure • Reliable • Profitable[/italic]",
            border_style="blue",
            padding=(1, 2),
        )
        self.logger.info("CryptoTradingBot v1.0.0 - Mode: %s", self._get_mode())

    def _print_help(self) -> None:
        """Print help message for CLI usage."""
        help_text = """
Usage: python main.py [OPTIONS]

Options:
  --help, -h       Show this help message and exit
  --status         Show the current trading bot status table and exit
  (no options)     Run the trading bot normally with live updating status
"""
        console.print(help_text)

    def _get_mode(self) -> str:
        """Determine the operating mode from command line arguments."""
        if len(sys.argv) > 1:
            return sys.argv[1].upper()
        return "LIVE"  # Default mode


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run the trading bot normally
  python main.py --help            # Show this help message
  python main.py --status          # Show current status and exit
        """
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show the current trading bot status table and exit"
    )

    return parser.parse_args()


async def main():
    """Main async entry point."""
    # Parse CLI arguments first
    args = parse_arguments()

    # Handle CLI-only commands that should exit immediately
    if args.status:
        try:
            # Load config and print status table once, then exit
            config = load_config()
            setup_logging(config.get("logging", {}))
            # Disable terminal display for CLI status to avoid live panel
            config["monitoring"]["terminal_display"] = False
            bot_engine = BotEngine(config)
            await bot_engine.initialize()
            bot_engine.print_status_table()
            await bot_engine.shutdown()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to show status: {str(e)}")
            sys.exit(1)
        sys.exit(0)

    # Normal execution path
    bot = CryptoTradingBot()
    await bot.initialize()
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.warning("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
