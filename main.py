#!/usr/bin/env python3
"""
Main entry point for the crypto trading bot system.
Handles initialization, mode selection, and core system startup.
"""

import asyncio
import logging
import sys
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
            console.print(
                "[bold green]✓ Configuration loaded successfully[/bold green]"
            )

            # Initialize core engine
            self.bot_engine = BotEngine(self.config)
            await self.bot_engine.initialize()

            console.print(
                "[bold green]✓ Bot engine initialized successfully[/bold green]"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize bot: {str(e)}", exc_info=True)
            console.print(f"[bold red]✗ Initialization failed: {str(e)}[/bold red]")
            sys.exit(1)

    async def run(self) -> None:
        """
        Main execution method for the trading bot.
        """
        if not self.bot_engine:
            self.logger.error("Bot engine not initialized")
            console.print("[bold red]✗ Bot engine not initialized[/bold red]")
            return

        try:
            self.logger.info("Starting CryptoTradingBot")
            console.print("[bold green]✓ Starting trading bot...[/bold green]")

            # Start the main bot engine
            await self.bot_engine.run()

        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
            console.print("\n[bold yellow]⚠ Bot stopped by user[/bold yellow]")
        except Exception as e:
            self.logger.error(f"Bot crashed: {str(e)}", exc_info=True)
            console.print(f"[bold red]✗ Bot crashed: {str(e)}[/bold red]")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """
        Cleanup and shutdown the bot gracefully.
        """
        self.logger.info("Shutting down bot")
        console.print("[bold yellow]⚠ Shutting down bot...[/bold yellow]")

        if self.bot_engine:
            await self.bot_engine.shutdown()

        self.logger.info("Bot shutdown complete")
        console.print("[bold green]✓ Bot shutdown complete[/bold green]")

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
        console.print(panel)

    def _get_mode(self) -> str:
        """Determine the operating mode from command line arguments."""
        if len(sys.argv) > 1:
            return sys.argv[1].upper()
        return "LIVE"  # Default mode


async def main():
    """Main async entry point."""
    bot = CryptoTradingBot()
    await bot.initialize()
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠ Application terminated by user[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]✗ Fatal error: {str(e)}[/bold red]")
        sys.exit(1)
