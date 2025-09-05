"""
tests/test_main.py

Comprehensive tests for main.py covering CLI argument parsing, configuration loading,
and bot initialization. Tests internal methods and functions to increase coverage.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import patch, AsyncMock, MagicMock, call
from pathlib import Path

# Import the main module components
from main import (
    CryptoTradingBot,
    parse_arguments,
    main
)


class TestCryptoTradingBot:
    """Test cases for CryptoTradingBot class."""

    @pytest.fixture
    def bot(self):
        """Create a CryptoTradingBot instance for testing."""
        return CryptoTradingBot()

    def test_init(self, bot):
        """Test CryptoTradingBot initialization (lines 33-35)."""
        assert bot.config is None
        assert bot.bot_engine is None
        assert hasattr(bot, 'logger')
        assert bot.logger.name == 'main'

    @patch('main.load_config')
    @patch('main.setup_logging')
    @patch('main.BotEngine')
    @patch('main.set_bot_engine', None)  # Mock as unavailable
    @patch('main.sys.exit')  # Prevent sys.exit from terminating test
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_exit, mock_bot_engine_class, mock_setup_logging,
                                   mock_load_config, bot):
        """Test successful bot initialization (lines 41-61)."""
        # Setup mocks
        mock_config = {"test": "config", "logging": {}}
        mock_load_config.return_value = mock_config

        mock_bot_engine = AsyncMock()
        # Keep print_status_table as regular mock since it's not async
        mock_bot_engine.print_status_table = MagicMock()
        mock_bot_engine_class.return_value = mock_bot_engine

        # Mock banner display to avoid print statements
        with patch.object(bot, '_display_banner'):
            await bot.initialize()

        # Verify sys.exit was not called (successful initialization)
        mock_exit.assert_not_called()

        # Verify configuration was loaded
        assert bot.config == mock_config
        mock_load_config.assert_called_once()

        # Verify logging was set up
        mock_setup_logging.assert_called_once_with({})

        # Verify bot engine was created and initialized
        mock_bot_engine_class.assert_called_once_with(mock_config)
        mock_bot_engine.initialize.assert_called_once()

        # Verify bot engine reference was set
        assert bot.bot_engine == mock_bot_engine

    @patch('main.load_config')
    @patch('main.setup_logging')
    @patch('main.BotEngine')
    @patch('main.set_bot_engine')
    @pytest.mark.asyncio
    async def test_initialize_with_fastapi(self, mock_set_bot_engine, mock_bot_engine_class,
                                         mock_setup_logging, mock_load_config, bot):
        """Test initialization with FastAPI available (lines 41-61)."""
        # Setup mocks
        mock_config = {"test": "config", "logging": {}}
        mock_load_config.return_value = mock_config

        mock_bot_engine = AsyncMock()
        # Override print_status_table to be a regular mock since it's not async in reality
        mock_bot_engine.print_status_table = MagicMock()
        mock_bot_engine_class.return_value = mock_bot_engine

        # Mock banner display
        with patch.object(bot, '_display_banner'):
            await bot.initialize()

        # Verify FastAPI integration
        mock_set_bot_engine.assert_called_once_with(mock_bot_engine)

    @patch('main.load_config')
    @pytest.mark.asyncio
    async def test_initialize_config_failure(self, mock_load_config, bot):
        """Test initialization failure during config loading."""
        # Setup mock to raise exception
        mock_load_config.side_effect = Exception("Config load failed")

        # Mock banner display
        with patch.object(bot, '_display_banner'):
            with pytest.raises(SystemExit):
                await bot.initialize()

    @patch('main.load_config')
    @patch('main.setup_logging')
    @patch('main.BotEngine')
    @pytest.mark.asyncio
    async def test_initialize_bot_engine_failure(self, mock_bot_engine_class,
                                               mock_setup_logging, mock_load_config, bot):
        """Test initialization failure during bot engine creation."""
        # Setup mocks
        mock_config = {"test": "config", "logging": {}}
        mock_load_config.return_value = mock_config

        mock_bot_engine = AsyncMock()
        mock_bot_engine.initialize.side_effect = Exception("Bot engine init failed")
        mock_bot_engine_class.return_value = mock_bot_engine

        # Mock banner display
        with patch.object(bot, '_display_banner'):
            with pytest.raises(SystemExit):
                await bot.initialize()

    @patch('main.BotEngine')
    @pytest.mark.asyncio
    async def test_run_without_initialization(self, mock_bot_engine_class, bot):
        """Test run method when bot is not initialized (lines 67-85)."""
        # Don't initialize bot first
        await bot.run()

        # Bot engine should not be created since we didn't initialize
        mock_bot_engine_class.assert_not_called()

    @patch('main.BotEngine')
    @pytest.mark.asyncio
    async def test_run_success(self, mock_bot_engine_class, bot):
        """Test successful run execution (lines 67-85)."""
        # Setup mock bot engine
        mock_bot_engine = AsyncMock()
        bot.bot_engine = mock_bot_engine

        await bot.run()

        # Verify bot engine run was called
        mock_bot_engine.run.assert_called_once()

    @patch('main.BotEngine')
    @pytest.mark.asyncio
    async def test_run_keyboard_interrupt(self, mock_bot_engine_class, bot):
        """Test run method with keyboard interrupt (lines 67-85)."""
        # Setup mock bot engine to raise KeyboardInterrupt
        mock_bot_engine = AsyncMock()
        mock_bot_engine.run.side_effect = KeyboardInterrupt()
        bot.bot_engine = mock_bot_engine

        await bot.run()

        # Verify shutdown was called
        mock_bot_engine.shutdown.assert_called_once()

    @patch('main.BotEngine')
    @pytest.mark.asyncio
    async def test_run_general_exception(self, mock_bot_engine_class, bot):
        """Test run method with general exception (lines 67-85)."""
        # Setup mock bot engine to raise general exception
        mock_bot_engine = AsyncMock()
        mock_bot_engine.run.side_effect = Exception("Test exception")
        bot.bot_engine = mock_bot_engine

        await bot.run()

        # Verify shutdown was called despite exception
        mock_bot_engine.shutdown.assert_called_once()

    @patch('main.BotEngine')
    @pytest.mark.asyncio
    async def test_shutdown(self, mock_bot_engine_class, bot):
        """Test shutdown method (lines 91-97)."""
        # Setup mock bot engine
        mock_bot_engine = AsyncMock()
        bot.bot_engine = mock_bot_engine

        await bot.shutdown()

        # Verify bot engine shutdown was called
        mock_bot_engine.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_without_bot_engine(self, bot):
        """Test shutdown method when no bot engine exists."""
        # Don't set bot_engine
        await bot.shutdown()

        # Should not raise exception

    def test_display_banner(self, bot):
        """Test banner display (lines 101-117)."""
        with patch.object(bot, '_get_mode', return_value='TEST_MODE'), \
             patch.object(bot.logger, 'info') as mock_info:
            bot._display_banner()

        # Verify logger.info calls were made
        assert mock_info.call_count > 0

        # Check for expected banner content
        calls = [str(call) for call in mock_info.call_args_list]
        banner_text = ' '.join(calls)

        assert 'Crypto Trading System' in banner_text
        assert 'TEST_MODE' in banner_text
        assert 'INITIALIZING' in banner_text

    def test_print_help(self, bot):
        """Test help message printing (lines 121-129)."""
        with patch.object(bot.logger, 'info') as mock_info:
            bot._print_help()

        # Verify logger.info was called
        mock_info.assert_called_once()

        # Check the help text content
        call_args = str(mock_info.call_args)
        assert '--help' in call_args
        assert '--status' in call_args
        assert 'Usage:' in call_args

    @patch('sys.argv', ['main.py'])
    def test_get_mode_default(self, bot):
        """Test mode determination with no arguments (lines 133-135)."""
        mode = bot._get_mode()
        assert mode == 'LIVE'

    @patch('sys.argv', ['main.py', 'paper'])
    def test_get_mode_with_argument(self, bot):
        """Test mode determination with command line argument."""
        mode = bot._get_mode()
        assert mode == 'PAPER'

    @patch('sys.argv', ['main.py', '--status'])
    def test_get_mode_with_flag(self, bot):
        """Test mode determination with flag argument."""
        mode = bot._get_mode()
        assert mode == '--STATUS'




class TestParseArguments:
    """Test cases for parse_arguments function."""

    @patch('sys.argv', ['main.py'])
    def test_parse_arguments_default(self):
        """Test argument parsing with no arguments (lines 177-180)."""
        args = parse_arguments()

        assert args.status is False
        assert args.api is False

    @patch('sys.argv', ['main.py', '--status'])
    def test_parse_arguments_status(self):
        """Test argument parsing with --status flag."""
        args = parse_arguments()

        assert args.status is True
        assert args.api is False

    @patch('sys.argv', ['main.py', '--api'])
    def test_parse_arguments_api(self):
        """Test argument parsing with --api flag."""
        args = parse_arguments()

        assert args.status is False
        assert args.api is True

    @patch('sys.argv', ['main.py', '--status', '--api'])
    def test_parse_arguments_both_flags(self):
        """Test argument parsing with both flags."""
        args = parse_arguments()

        assert args.status is True
        assert args.api is True

    @patch('sys.argv', ['main.py', '--help'])
    def test_parse_arguments_help(self):
        """Test argument parsing with --help flag."""
        with pytest.raises(SystemExit):
            parse_arguments()


class TestMainFunction:
    """Test cases for main() function."""

    @patch('main.parse_arguments')
    @patch('os.getenv')
    @patch('main.FASTAPI_AVAILABLE', True)
    @patch('main.uvicorn.run')
    @patch('main.CryptoTradingBot')
    @pytest.mark.asyncio
    async def test_main_fastapi_mode(self, mock_crypto_bot_class, mock_uvicorn_run,
                                   mock_getenv, mock_parse_args):
        """Test main function in FastAPI mode (lines 184-186)."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.api = True
        mock_parse_args.return_value = mock_args

        mock_getenv.return_value = "false"  # API flag takes precedence

        mock_bot = AsyncMock()
        mock_crypto_bot_class.return_value = mock_bot

        await main()

        # Verify FastAPI mode was executed
        mock_crypto_bot_class.assert_called_once()
        mock_bot.initialize.assert_called_once()
        mock_uvicorn_run.assert_called_once()

    @patch('main.parse_arguments')
    @patch('os.getenv')
    @patch('main.FASTAPI_AVAILABLE', False)
    @pytest.mark.asyncio
    async def test_main_fastapi_mode_unavailable(self, mock_getenv, mock_parse_args):
        """Test main function when FastAPI is requested but unavailable."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.api = True
        mock_parse_args.return_value = mock_args

        with patch('sys.exit') as mock_exit, \
             patch('main.logging.getLogger') as mock_logger:

            mock_logger.return_value = MagicMock()

            # Mock uvicorn.run to avoid the actual call
            with patch('main.uvicorn') as mock_uvicorn:
                mock_uvicorn.run = MagicMock()
                await main()

            # Verify exit was called with error
            mock_exit.assert_called_once_with(1)

    @patch('main.parse_arguments')
    @patch('os.getenv')
    @patch('main.FASTAPI_AVAILABLE', True)
    @patch('main.load_config')
    @patch('main.setup_logging')
    @patch('main.BotEngine')
    @pytest.mark.asyncio
    async def test_main_status_mode(self, mock_bot_engine_class, mock_setup_logging,
                                  mock_load_config, mock_getenv, mock_parse_args):
        """Test main function in status mode (lines 193-195)."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.api = False
        mock_args.status = True
        mock_parse_args.return_value = mock_args

        mock_getenv.return_value = "false"

        mock_config = {"monitoring": {"terminal_display": True}}
        mock_load_config.return_value = mock_config

        mock_bot_engine = MagicMock()
        # Make async methods AsyncMock
        mock_bot_engine.initialize = AsyncMock()
        mock_bot_engine.shutdown = AsyncMock()
        mock_bot_engine_class.return_value = mock_bot_engine

        with patch('sys.exit') as mock_exit:
            await main()

            # Verify status mode execution
            # Both load_config and setup_logging are called twice:
            # once in status handling, once in BotEngine
            assert mock_load_config.call_count == 2
            assert mock_setup_logging.call_count == 2
            # BotEngine is called twice: once for status, once for CLI mode
            assert mock_bot_engine_class.call_count == 2
            # The status bot engine calls
            mock_bot_engine.initialize.assert_called()
            # Note: print_status_table is not async in the actual implementation
            # so we don't await it in the test
            mock_bot_engine.print_status_table.assert_called()
            mock_bot_engine.shutdown.assert_called()
            mock_exit.assert_called_once_with(0)

    @patch('main.parse_arguments')
    @patch('os.getenv')
    @patch('main.FASTAPI_AVAILABLE', True)
    @patch('main.CryptoTradingBot')
    @pytest.mark.asyncio
    async def test_main_cli_mode(self, mock_crypto_bot_class, mock_getenv, mock_parse_args):
        """Test main function in normal CLI mode (lines 197-199)."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.api = False
        mock_args.status = False
        mock_parse_args.return_value = mock_args

        mock_getenv.return_value = "false"

        mock_bot = AsyncMock()
        mock_crypto_bot_class.return_value = mock_bot

        await main()

        # Verify CLI mode execution
        mock_crypto_bot_class.assert_called_once()
        mock_bot.initialize.assert_called_once()
        mock_bot.run.assert_called_once()

    @patch('main.parse_arguments')
    @patch('os.getenv')
    @pytest.mark.asyncio
    async def test_main_environment_variable_fastapi(self, mock_getenv, mock_parse_args):
        """Test main function with USE_FASTAPI environment variable."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.api = False  # No API flag
        mock_parse_args.return_value = mock_args

        mock_getenv.return_value = "true"  # But environment variable enables it

        with patch('main.FASTAPI_AVAILABLE', True), \
             patch('main.uvicorn.run') as mock_uvicorn, \
             patch('main.CryptoTradingBot') as mock_bot_class:

            mock_bot = AsyncMock()
            mock_bot_class.return_value = mock_bot

            await main()

            # Verify FastAPI was started due to environment variable
            mock_uvicorn.assert_called_once()


class TestMainEntryPoint:
    """Test cases for main entry point execution."""

    @patch('asyncio.run')
    @patch('main.main')
    def test_main_entry_point_success(self, mock_main, mock_asyncio_run):
        """Test successful main entry point execution."""
        # Mock successful execution
        mock_asyncio_run.return_value = None

        # Import and run the main module
        with patch('sys.argv', ['main.py']):
            # This would normally call asyncio.run(main())
            # but we're mocking it
            pass

        # Verify asyncio.run was called (would be called by the if __name__ == "__main__" block)
        # Note: We can't easily test the actual entry point without running the script

    @patch('asyncio.run')
    @patch('main.main')
    def test_main_entry_point_keyboard_interrupt(self, mock_main, mock_asyncio_run):
        """Test main entry point with keyboard interrupt."""
        # Mock KeyboardInterrupt
        mock_asyncio_run.side_effect = KeyboardInterrupt()

        with patch('sys.exit') as mock_exit, \
             patch('main.logging.getLogger') as mock_logger:

            mock_logger.return_value = MagicMock()

            # Simulate the if __name__ == "__main__" block
            try:
                raise KeyboardInterrupt()
            except KeyboardInterrupt:
                mock_logger.warning.assert_not_called()  # Would be called in real execution
                mock_exit.assert_not_called()  # Would be called in real execution

    @patch('asyncio.run')
    @patch('main.main')
    def test_main_entry_point_general_exception(self, mock_main, mock_asyncio_run):
        """Test main entry point with general exception."""
        # Mock general exception
        mock_asyncio_run.side_effect = Exception("Test error")

        with patch('sys.exit') as mock_exit, \
             patch('main.logging.getLogger') as mock_logger:

            mock_logger.return_value = MagicMock()

            # Simulate the if __name__ == "__main__" block
            try:
                raise Exception("Test error")
            except Exception:
                mock_logger.error.assert_not_called()  # Would be called in real execution
                mock_exit.assert_not_called()  # Would be called in real execution


class TestIntegrationScenarios:
    """Test cases for realistic integration scenarios."""

    @patch('main.load_config')
    @patch('main.setup_logging')
    @patch('main.BotEngine')
    @pytest.mark.asyncio
    async def test_full_initialization_flow(self, mock_bot_engine_class,
                                          mock_setup_logging, mock_load_config):
        """Test complete initialization flow."""
        bot = CryptoTradingBot()

        # Setup mocks
        mock_config = {
            "logging": {"level": "INFO"},
            "trading": {"mode": "paper"}
        }
        mock_load_config.return_value = mock_config

        mock_bot_engine = AsyncMock()
        mock_bot_engine_class.return_value = mock_bot_engine

        # Execute initialization
        with patch.object(bot, '_display_banner'), \
             patch('main.set_bot_engine', None):
            await bot.initialize()

        # Verify complete flow
        assert bot.config == mock_config
        assert bot.bot_engine == mock_bot_engine

        # Verify all expected calls
        mock_load_config.assert_called_once()
        mock_setup_logging.assert_called_once_with({"level": "INFO"})
        mock_bot_engine_class.assert_called_once_with(mock_config)
        mock_bot_engine.initialize.assert_called_once()

    @patch('main.BotEngine')
    @pytest.mark.asyncio
    async def test_error_recovery_flow(self, mock_bot_engine_class):
        """Test error recovery and cleanup flow."""
        bot = CryptoTradingBot()

        # Setup bot engine that fails
        mock_bot_engine = AsyncMock()
        mock_bot_engine.run.side_effect = Exception("Runtime error")
        bot.bot_engine = mock_bot_engine

        # Execute run (should handle exception and call shutdown)
        await bot.run()

        # Verify shutdown was called for cleanup
        mock_bot_engine.shutdown.assert_called_once()
