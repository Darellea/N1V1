import pytest
import asyncio
from unittest.mock import AsyncMock
from core.bot_engine import BotEngine

@pytest.mark.asyncio
async def test_botengine_respects_order_manager_safe_mode(tmp_path):
    """
    If OrderManager.safe_mode_active is True, BotEngine should skip trading cycles
    and not attempt to execute orders.
    """
    config = {
        "environment": {"mode": "paper"},
        "exchange": {"base_currency": "USDT", "markets": ["BTC/USDT"]},
        "trading": {"initial_balance": 1000.0},
        "risk_management": {},
        "notifications": {"discord": {"enabled": False}},
        "monitoring": {"terminal_display": False, "update_interval": 1},
        "strategies": {"active_strategies": [], "strategy_config": {}},
    }

    bot = BotEngine(config)
    # minimal initialization state
    bot.data_fetcher = AsyncMock()
    bot.strategies = []
    # OrderManager stub with safe_mode_active True
    class OMStub:
        safe_mode_active = True
        async def get_balance(self): return 1000
        async def get_equity(self): return 1000
        async def get_active_order_count(self): return 0
        async def get_open_position_count(self): return 0
    bot.order_manager = OMStub()

    # Risk manager stub
    class RMStub:
        async def evaluate_signal(self, *args, **kwargs): return True
    bot.risk_manager = RMStub()

    # Run one trading cycle; it should return early due to safe mode and not raise
    await bot._trading_cycle()
    # After running, global_safe_mode should be True
    assert bot.global_safe_mode is True
