#!/usr/bin/env python3
"""
Main entry point for the crypto trading bot system.
Handles initialization, mode selection, and core system startup.
"""

import sys
import argparse


def print_status() -> None:
    """Print the trading bot status table for --status flag."""
    try:
        mode = "STATUS"
        status = "CHECK"
        balance_str = "0.00 USDT"
        equity_str = "0.00 USDT"
        active_orders = "0"
        open_positions = "0"
        total_pnl = "0.00"
        win_rate = "0.00%"

        print("\n+-----------------+---------------------+")
        print("| Trading Bot Status                  |")
        print("+-----------------+---------------------+")
        print(f"| Mode            | {mode:<19} |")
        print(f"| Status          | {status:<19} |")
        print(f"| Balance         | {balance_str:<19} |")
        print(f"| Equity          | {equity_str:<19} |")
        print(f"| Active Orders   | {active_orders:<19} |")
        print(f"| Open Positions  | {open_positions:<19} |")
        print(f"| Total PnL       | {total_pnl:<19} |")
        print(f"| Win Rate        | {win_rate:<19} |")
        print("+-----------------+---------------------+")
        sys.stdout.flush()

    except Exception as e:
        print(f"Failed to show status: {str(e)}", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run the trading bot normally (CLI mode)
  python main.py --help            # Show this help message
  python main.py --status          # Show current status and exit
  python main.py --api             # Run with FastAPI web interface
  USE_FASTAPI=true python main.py # Run with FastAPI (environment variable)
        """
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show the current trading bot status table and exit"
    )

    parser.add_argument(
        "--api",
        action="store_true",
        help="Run with FastAPI web interface instead of CLI mode"
    )

    return parser.parse_args()


async def main(args=None):
    """Main async entry point."""
    if args is None:
        args = parse_arguments()

    # Check if FastAPI mode is enabled
    use_fastapi = args.api or os.getenv("USE_FASTAPI", "").lower() in ("true", "1", "yes")

    if use_fastapi:
        if not FASTAPI_AVAILABLE:
            logger = logging.getLogger(__name__)
            logger.error("FastAPI mode requested but FastAPI dependencies are not installed")
            logger.error("Install with: pip install fastapi uvicorn")
            sys.exit(1)
        else:
            logger = logging.getLogger(__name__)
            logger.info("Starting in FastAPI mode")

            # Initialize bot
            bot = CryptoTradingBot()
            await bot.initialize()

            # Start FastAPI server
            logger.info("Starting FastAPI server on http://localhost:8000")
            logger.info("API documentation available at http://localhost:8000/docs")

            # Run uvicorn server (this will block)
            uvicorn.run(
                "api.app:app",
                host="0.0.0.0",
                port=8000,
                reload=False,
                log_level="info"
            )
            return

    # Normal CLI execution path
    logger = logging.getLogger(__name__)
    logger.info("Starting in CLI mode")
    bot = CryptoTradingBot()
    await bot.initialize()
    await bot.run()


if __name__ == "__main__":
    args = parse_arguments()

    if args.status:
        try:
            print_status()
        except Exception as e:
            print(f"Failed to show status: {str(e)}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

# Import heavy modules only if not --status
import asyncio
import logging
import os
from typing import Optional

from utils.config_loader import load_config, ConfigLoader
from utils.logger import setup_logging
from core.bot_engine import BotEngine

# FastAPI imports (optional)
try:
    import uvicorn
    from api.app import app, set_bot_engine
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    uvicorn = None
    app = None
    set_bot_engine = None


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

            # Set bot engine reference in FastAPI app if available
            if FASTAPI_AVAILABLE and set_bot_engine:
                set_bot_engine(self.bot_engine)
                self.logger.info("Bot engine registered with FastAPI app")

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
        self.logger.info("=====================================")
        self.logger.info("  Crypto Trading System")
        self.logger.info("  Secure • Reliable • Profitable")
        self.logger.info("=====================================")
        self.logger.info("CryptoTradingBot")
        self.logger.info("Version: 1.0.0")
        self.logger.info(f"Mode: {self._get_mode()}")
        self.logger.info("Status: INITIALIZING")
        self.logger.info("=====================================")
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
        self.logger.info(help_text)

    def _get_mode(self) -> str:
        """Determine the operating mode from command line arguments."""
        if len(sys.argv) > 1:
            return sys.argv[1].upper()
        return "LIVE"  # Default mode


if __name__ == "__main__":
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("Application terminated by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        sys.exit(1)
