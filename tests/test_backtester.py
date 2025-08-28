import pytest
import asyncio
from unittest.mock import AsyncMock
from decimal import Decimal
import csv
from pathlib import Path

from core.bot_engine import BotEngine
from backtest.backtester import export_equity_from_botengine


@pytest.mark.asyncio
async def test_record_trade_equity_appends_record():
    """record_trade_equity should append a record with correct fields and values."""
    # Minimal config required by BotEngine.__init__
    config = {
        "environment": {"mode": "paper"},
        "exchange": {"base_currency": "USDT"},
        "trading": {"initial_balance": 1000.0},
        "risk_management": {},
        "notifications": {"discord": {"enabled": False}},
        "monitoring": {"terminal_display": False, "update_interval": 1},
        "strategies": {"active_strategies": [], "strategy_config": {}},
    }

    bot = BotEngine(config)
    # Set starting_balance manually (normally set during initialize)
    bot.starting_balance = 1000.0

    # Mock order_manager.get_equity to return a Decimal value
    mock_order_manager = AsyncMock()
    mock_order_manager.get_equity.return_value = Decimal("1100.50")
    bot.order_manager = mock_order_manager

    # Ensure no prior records
    assert bot.performance_stats.get("equity_progression") == []

    order_result = {"id": "trade_1", "timestamp": 1234567890, "pnl": 100.5}
    await bot.record_trade_equity(order_result)

    eq_prog = bot.performance_stats["equity_progression"]
    assert len(eq_prog) == 1

    record = eq_prog[0]
    assert record["trade_id"] == "trade_1"
    assert record["timestamp"] == 1234567890
    assert pytest.approx(record["equity"], rel=1e-6) == 1100.50
    assert pytest.approx(record["pnl"], rel=1e-6) == 100.5
    # cumulative_return = (1100.50 - 1000.0) / 1000.0 = 0.1005
    assert pytest.approx(record["cumulative_return"], rel=1e-6) == 0.1005


@pytest.mark.asyncio
async def test_record_trade_equity_fallbacks_to_total_pnl_when_order_manager_unavailable():
    """If order_manager.get_equity returns 0 or is unavailable, equity is derived from starting_balance + total_pnl."""
    config = {
        "environment": {"mode": "paper"},
        "exchange": {"base_currency": "USDT"},
        "trading": {"initial_balance": 2000.0},
        "risk_management": {},
        "notifications": {"discord": {"enabled": False}},
        "monitoring": {"terminal_display": False, "update_interval": 1},
        "strategies": {"active_strategies": [], "strategy_config": {}},
    }

    bot = BotEngine(config)
    bot.starting_balance = 2000.0

    # Mock order_manager.get_equity to return 0.0 (not tracking)
    mock_order_manager = AsyncMock()
    mock_order_manager.get_equity.return_value = Decimal("0")
    bot.order_manager = mock_order_manager

    # Simulate total_pnl tracked elsewhere (update_performance_metrics would normally set this)
    bot.performance_stats["total_pnl"] = 150.0

    order_result = {"id": "trade_2", "timestamp": 2222222222, "pnl": 50.0}
    await bot.record_trade_equity(order_result)

    eq_prog = bot.performance_stats["equity_progression"]
    # last appended record
    record = eq_prog[-1]
    # equity should be starting_balance + total_pnl = 2150.0
    assert pytest.approx(record["equity"], rel=1e-6) == 2150.0
    assert (
        pytest.approx(record["cumulative_return"], rel=1e-6)
        == (2150.0 - 2000.0) / 2000.0
    )


@pytest.mark.asyncio
async def test_record_trade_equity_handles_missing_pnl():
    """record_trade_equity should handle order_result without pnl."""
    config = {
        "environment": {"mode": "paper"},
        "exchange": {"base_currency": "USDT"},
        "trading": {"initial_balance": 500.0},
        "risk_management": {},
        "notifications": {"discord": {"enabled": False}},
        "monitoring": {"terminal_display": False, "update_interval": 1},
        "strategies": {"active_strategies": [], "strategy_config": {}},
    }

    bot = BotEngine(config)
    bot.starting_balance = 500.0

    mock_order_manager = AsyncMock()
    mock_order_manager.get_equity.return_value = Decimal("525.0")
    bot.order_manager = mock_order_manager

    order_result = {"id": "trade_3", "timestamp": 3333333333}  # no 'pnl' key
    await bot.record_trade_equity(order_result)

    record = bot.performance_stats["equity_progression"][-1]
    assert record["pnl"] is None
    assert pytest.approx(record["equity"], rel=1e-6) == 525.0
    assert (
        pytest.approx(record["cumulative_return"], rel=1e-6) == (525.0 - 500.0) / 500.0
    )


@pytest.mark.asyncio
async def test_export_equity_csv_written_and_contains_rows(tmp_path):
    """After populating equity_progression, export to CSV and verify contents."""
    config = {
        "environment": {"mode": "paper"},
        "exchange": {"base_currency": "USDT"},
        "trading": {"initial_balance": 1000.0},
        "risk_management": {},
        "notifications": {"discord": {"enabled": False}},
        "monitoring": {"terminal_display": False, "update_interval": 1},
        "strategies": {"active_strategies": [], "strategy_config": {}},
    }

    bot = BotEngine(config)
    bot.starting_balance = 1000.0

    mock_order_manager = AsyncMock()
    # First call returns 1100.5, second 1080.0
    mock_order_manager.get_equity.side_effect = [Decimal("1100.50"), Decimal("1080.00")]
    bot.order_manager = mock_order_manager

    # Create two trade records
    await bot.record_trade_equity({"id": "t1", "timestamp": 1, "pnl": 100.5})
    await bot.record_trade_equity({"id": "t2", "timestamp": 2, "pnl": -20.5})

    out_csv = tmp_path / "equity_curve.csv"
    out_path = export_equity_from_botengine(bot, out_path=str(out_csv))

    # Verify file exists
    p = Path(out_path)
    assert p.exists()

    # Read CSV and verify rows
    with open(p, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert len(rows) == 2
    # Check first row values
    assert rows[0]["trade_id"] == "t1"
    assert rows[0]["timestamp"] == "1"
    assert float(rows[0]["equity"]) == pytest.approx(1100.50)
    assert float(rows[0]["pnl"]) == pytest.approx(100.5)
    # Check second row
    assert rows[1]["trade_id"] == "t2"
    assert rows[1]["timestamp"] == "2"
    assert float(rows[1]["equity"]) == pytest.approx(1080.00)
    assert float(rows[1]["pnl"]) == pytest.approx(-20.5)
