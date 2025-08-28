import pytest
import asyncio
from decimal import Decimal

from core.order_manager import OrderManager
from core.bot_engine import BotEngine
from backtest.backtester import compute_backtest_metrics


@pytest.mark.asyncio
async def test_order_manager_initialize_portfolio_splits_balance_equally():
    """
    When initialize_portfolio is called without allocation, the initial paper_balance
    should be split equally among the provided pairs.
    """
    cfg = {
        "paper": {"initial_balance": 1000.0},
        # keep minimal keys OrderManager expects in nested config
        "order": {"slippage": 0.001, "trade_fee": 0.001, "base_currency": "USDT"},
    }
    om = OrderManager(cfg, mode="paper")
    await om.initialize_portfolio(["BTC/USDT", "ETH/USDT"], True, allocation=None)

    assert om.portfolio_mode is True
    assert "BTC/USDT" in om.paper_balances and "ETH/USDT" in om.paper_balances
    # Each should be approximately half of 1000
    assert pytest.approx(float(om.paper_balances["BTC/USDT"]), rel=1e-6) == 500.0
    assert pytest.approx(float(om.paper_balances["ETH/USDT"]), rel=1e-6) == 500.0


@pytest.mark.asyncio
async def test_botengine_record_trade_equity_includes_symbol_when_provided():
    """
    BotEngine.record_trade_equity should include the trade's symbol in the recorded
    equity progression when the order_result includes a 'symbol' key.
    """
    config = {
        "environment": {"mode": "paper"},
        "exchange": {"base_currency": "USDT", "markets": ["BTC/USDT", "ETH/USDT"]},
        "trading": {"initial_balance": 1000.0, "portfolio_mode": True},
        "risk_management": {},
        "notifications": {"discord": {"enabled": False}},
        "monitoring": {"terminal_display": False, "update_interval": 1},
        "strategies": {"active_strategies": [], "strategy_config": {}},
    }

    bot = BotEngine(config)
    bot.starting_balance = 1000.0

    # Provide a minimal async-compatible order_manager
    class DummyOM:
        async def get_equity(self):
            return Decimal("1100.50")

    bot.order_manager = DummyOM()

    order_result = {"id": "trade_sym_1", "timestamp": 1111, "pnl": 100.5, "symbol": "BTC/USDT"}
    await bot.record_trade_equity(order_result)

    rec = bot.performance_stats["equity_progression"][-1]
    assert rec["trade_id"] == "trade_sym_1"
    assert rec["symbol"] == "BTC/USDT"
    assert pytest.approx(rec["equity"], rel=1e-6) == 1100.50
    assert pytest.approx(rec["pnl"], rel=1e-6) == 100.5


def test_compute_backtest_metrics_returns_per_symbol_groups():
    """
    compute_backtest_metrics should return 'per_symbol' metrics when records include a 'symbol' key.
    """
    eq_prog = [
        {"trade_id": "t1", "timestamp": 1, "symbol": "BTC/USDT", "equity": 1000.0, "pnl": 50.0},
        {"trade_id": "t2", "timestamp": 2, "symbol": "BTC/USDT", "equity": 1050.0, "pnl": -10.0},
        {"trade_id": "t3", "timestamp": 3, "symbol": "ETH/USDT", "equity": 1000.0, "pnl": 20.0},
        {"trade_id": "t4", "timestamp": 4, "symbol": "ETH/USDT", "equity": 1020.0, "pnl": -5.0},
    ]

    metrics = compute_backtest_metrics(eq_prog)
    assert "per_symbol" in metrics
    assert "BTC/USDT" in metrics["per_symbol"]
    assert "ETH/USDT" in metrics["per_symbol"]

    btc_metrics = metrics["per_symbol"]["BTC/USDT"]
    eth_metrics = metrics["per_symbol"]["ETH/USDT"]

    # BTC had two trades (one win, one loss)
    assert btc_metrics["total_trades"] == 2
    assert btc_metrics["wins"] == 1
    assert btc_metrics["losses"] == 1

    # ETH had two trades (one win, one loss)
    assert eth_metrics["total_trades"] == 2
    assert eth_metrics["wins"] == 1
    assert eth_metrics["losses"] == 1

    # Overall metrics should also be present
    assert "total_trades" in metrics
    assert metrics["total_trades"] == 4
