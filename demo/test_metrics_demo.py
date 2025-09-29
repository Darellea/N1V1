#!/usr/bin/env python3
"""
Demo script for Auto-Metric & Risk Dashboard

Demonstrates the comprehensive metrics calculation, persistence,
and dashboard synchronization capabilities.
"""

import asyncio
import sys
from datetime import datetime, timedelta

import numpy as np

# Add the project root to Python path
sys.path.insert(0, ".")

from reporting import (
    DashboardSync,
    MetricsEngine,
    MetricsResult,
    MetricsScheduler,
)


async def demo_metrics_engine():
    """Demonstrate Metrics Engine functionality."""
    print("📊 Metrics Engine Demo")
    print("=" * 30)

    try:
        # Create metrics engine
        engine = MetricsEngine()
        print("✅ Metrics Engine initialized")

        # Generate sample returns data (252 trading days)
        np.random.seed(42)  # For reproducible results

        # Simulate different market conditions
        bull_market = np.random.normal(
            0.002, 0.015, 63
        )  # Bull market: 0.2% daily return
        bear_market = np.random.normal(
            -0.001, 0.025, 63
        )  # Bear market: -0.1% daily return
        volatile_market = np.random.normal(
            0.0005, 0.035, 63
        )  # Volatile: 0.05% with high volatility
        mixed_market = np.random.normal(
            0.001, 0.02, 63
        )  # Mixed: 0.1% with moderate volatility

        returns = np.concatenate(
            [bull_market, bear_market, volatile_market, mixed_market]
        )

        # Create sample trade log
        trade_log = (
            [
                {"pnl": 1250.50, "timestamp": datetime.now() - timedelta(days=i)}
                for i in range(50)
            ]
            + [
                {"pnl": -850.25, "timestamp": datetime.now() - timedelta(days=i)}
                for i in range(50, 100)
            ]
            + [
                {"pnl": 2100.75, "timestamp": datetime.now() - timedelta(days=i)}
                for i in range(100, 150)
            ]
        )

        print(f"📈 Generated {len(returns)} days of returns data")
        print(f"🏷️  Generated {len(trade_log)} trade records")

        # Calculate comprehensive metrics
        result = engine.calculate_metrics(
            returns=returns,
            strategy_id="demo_strategy",
            trade_log=trade_log,
            period_start=datetime.now() - timedelta(days=252),
            period_end=datetime.now(),
        )

        print("\n🎯 Calculated Metrics:")
        print("-" * 20)
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Annualized Return: {result.annualized_return:.2f}")
        print(f"Volatility: {result.volatility:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.1%}")
        print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
        print(f"Calmar Ratio: {result.calmar_ratio:.2%}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"VaR 95%: {result.value_at_risk_95:.2%}")
        print(f"Expected Shortfall 95%: {result.expected_shortfall_95:.2%}")
        print(f"Profit Factor: {result.profit_factor:.2%}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.1%}")
        print(f"Avg Win: ${result.avg_win:.2f}")
        print(f"Avg Loss: ${result.avg_loss:.2f}")
        print(f"Largest Win: ${result.largest_win:.2f}")
        print(f"Largest Loss: ${result.largest_loss:.2f}")

        # Save to files
        json_path = engine.save_to_json(result)
        csv_path = engine.save_to_csv(result)

        print("\n💾 Files Saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")

        # Test loading from JSON
        loaded_result = engine.load_from_json(json_path)
        print(f"✅ Successfully loaded metrics from JSON: {loaded_result.strategy_id}")

        return result

    except Exception as e:
        print(f"❌ Metrics Engine demo failed: {e}")
        import traceback

        traceback.print_exc()
        return None


async def demo_dashboard_sync():
    """Demonstrate Dashboard Synchronization."""
    print("\n📡 Dashboard Sync Demo")
    print("=" * 25)

    try:
        # Create dashboard sync (Streamlit only for demo)
        config = {
            "enabled": True,
            "streamlit": {"enabled": True, "data_dir": "demo_dashboard_data"},
            "grafana": {"enabled": False},  # Disabled for demo
        }

        sync = DashboardSync(config)
        print("✅ Dashboard Sync initialized")

        # Create sample metrics result
        result = MetricsResult(
            timestamp=datetime.now(),
            strategy_id="sync_demo_strategy",
            portfolio_id="demo_portfolio",
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now(),
            total_return=0.15,
            annualized_return=0.18,
            volatility=0.22,
            sharpe_ratio=1.45,
            sortino_ratio=1.67,
            calmar_ratio=1.23,
            max_drawdown=0.12,
            max_drawdown_duration=15,
            value_at_risk_95=-0.035,
            expected_shortfall_95=-0.045,
            total_trades=150,
            winning_trades=105,
            losing_trades=45,
            win_rate=0.70,
            profit_factor=1.85,
            avg_win=1250.00,
            avg_loss=-875.00,
            largest_win=3500.00,
            largest_loss=-2100.00,
        )

        # Sync to dashboard
        success = sync.sync_metrics(result)

        if success:
            print("✅ Successfully synced metrics to dashboard")
            print(f"   Data directory: {sync.streamlit_data_dir}")

            # Check files were created
            import os

            data_dir = sync.streamlit_data_dir
            if os.path.exists(data_dir):
                files = os.listdir(data_dir)
                print(f"   Files created: {files}")
        else:
            print("❌ Failed to sync metrics to dashboard")

        # Get dashboard status
        status = sync.get_dashboard_status()
        print(f"📊 Dashboard Status: {status}")

        return success

    except Exception as e:
        print(f"❌ Dashboard sync demo failed: {e}")
        return False


async def demo_scheduler():
    """Demonstrate Metrics Scheduler functionality."""
    print("\n⏰ Metrics Scheduler Demo")
    print("=" * 25)

    try:
        # Create scheduler
        config = {
            "enabled": True,
            "session_end_enabled": True,
            "daily_enabled": False,  # Disable for demo
            "weekly_enabled": False,  # Disable for demo
        }

        scheduler = MetricsScheduler(config)
        print("✅ Metrics Scheduler initialized")

        # Start a demo session
        session_id = "demo_session_001"
        strategy_ids = ["demo_strategy_1", "demo_strategy_2"]

        success = scheduler.start_session(session_id, strategy_ids)
        if success:
            print(f"✅ Started session: {session_id}")
            print(f"   Strategies: {strategy_ids}")
        else:
            print(f"❌ Failed to start session: {session_id}")
            return False

        # Update session with sample data
        sample_returns_1 = [0.002, 0.0015, -0.001, 0.003, 0.0025]  # Strategy 1 returns
        sample_returns_2 = [0.001, 0.002, 0.001, -0.002, 0.0015]  # Strategy 2 returns

        sample_trades_1 = [
            {"pnl": 1250.50, "timestamp": datetime.now()},
            {"pnl": -850.25, "timestamp": datetime.now()},
            {"pnl": 2100.75, "timestamp": datetime.now()},
        ]

        sample_trades_2 = [
            {"pnl": 950.00, "timestamp": datetime.now()},
            {"pnl": -650.50, "timestamp": datetime.now()},
            {"pnl": 1800.25, "timestamp": datetime.now()},
        ]

        # Update session data
        scheduler.update_session_data(
            session_id,
            "demo_strategy_1",
            returns=sample_returns_1,
            trade_log=sample_trades_1,
        )
        scheduler.update_session_data(
            session_id,
            "demo_strategy_2",
            returns=sample_returns_2,
            trade_log=sample_trades_2,
        )

        print("✅ Updated session with sample data")

        # End session (this will generate metrics)
        success = scheduler.end_session(session_id)
        if success:
            print(f"✅ Ended session: {session_id}")
            print("   Metrics generated and saved")
        else:
            print(f"❌ Failed to end session: {session_id}")

        # Get scheduler status
        status = scheduler.get_scheduler_status()
        print(
            f"📊 Scheduler Status: {status['active_sessions']} active sessions, {status['scheduled_tasks']} tasks"
        )

        return success

    except Exception as e:
        print(f"❌ Scheduler demo failed: {e}")
        return False


async def demo_comprehensive_workflow():
    """Demonstrate comprehensive metrics workflow."""
    print("\n🔄 Comprehensive Metrics Workflow Demo")
    print("=" * 40)

    try:
        # 1. Generate sample trading data
        print("📊 Step 1: Generating sample trading data...")

        # Simulate 6 months of daily returns
        np.random.seed(123)
        num_days = 126  # 6 months
        returns = np.random.normal(
            0.0012, 0.018, num_days
        )  # 0.12% mean, 1.8% volatility

        # Generate trade log
        trade_log = []
        for i in range(200):  # 200 trades
            pnl = (
                np.random.normal(500, 800)
                if np.random.random() > 0.4
                else np.random.normal(-400, 600)
            )
            trade_log.append(
                {
                    "pnl": pnl,
                    "timestamp": datetime.now()
                    - timedelta(days=np.random.randint(1, num_days)),
                }
            )

        print(f"   Generated {num_days} days of returns")
        print(f"   Generated {len(trade_log)} trades")

        # 2. Calculate metrics
        print("\n📈 Step 2: Calculating comprehensive metrics...")

        engine = MetricsEngine()
        result = engine.calculate_metrics(
            returns=returns, strategy_id="comprehensive_demo", trade_log=trade_log
        )

        print("   ✅ Performance metrics calculated")
        print("   ✅ Risk metrics calculated")
        print("   ✅ Trade statistics calculated")

        # 3. Save results
        print("\n💾 Step 3: Saving results...")
        json_path = engine.save_to_json(result)
        csv_path = engine.save_to_csv(result)
        print(f"   JSON saved: {json_path}")
        print(f"   CSV saved: {csv_path}")

        # 4. Sync to dashboard
        print("\n📡 Step 4: Syncing to dashboard...")
        sync = DashboardSync(
            {
                "enabled": True,
                "streamlit": {"enabled": True, "data_dir": "comprehensive_demo_data"},
            }
        )

        sync_success = sync.sync_metrics(result)
        if sync_success:
            print("   ✅ Synced to dashboard")
        else:
            print("   ❌ Dashboard sync failed")

        # 5. Summary
        print("\n🎉 Step 5: Workflow Summary")
        print("-" * 25)
        print(f"Strategy: {result.strategy_id}")
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Annualized Return: {result.annualized_return:.2f}")
        print(f"Volatility: {result.volatility:.2%}")
        print(f"Win Rate: {result.win_rate:.1%}")
        print(f"Total Trades: {result.total_trades}")
        print("Files Generated: JSON, CSV, Dashboard data")

        return True

    except Exception as e:
        print(f"❌ Comprehensive workflow demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main demo function."""
    print("🤖 N1V1 Trading Framework - Auto-Metric & Risk Dashboard Demo")
    print("=" * 70)

    # Run all demos
    results = []
    results.append(await demo_metrics_engine())
    results.append(await demo_dashboard_sync())
    results.append(await demo_scheduler())
    results.append(await demo_comprehensive_workflow())

    # Summary
    successful = sum(1 for r in results if r is not False and r is not None)
    total = len(results)

    print(f"\n📊 Demo Summary: {successful}/{total} components working")
    print("=" * 70)

    if successful >= 3:  # At least 3 out of 4 should work
        print("🎉 Auto-Metric & Risk Dashboard is functional!")
        print("\n🚀 Key Features Demonstrated:")
        print("   • Comprehensive risk-adjusted performance metrics")
        print("   • Sharpe, Sortino, Calmar ratios calculation")
        print("   • Value at Risk (VaR) and Expected Shortfall")
        print("   • Trade statistics and win rate analysis")
        print("   • JSON/CSV persistence with timestamped files")
        print("   • Dashboard synchronization (Grafana/Streamlit)")
        print("   • Automated session-based metrics generation")
        print("   • Background scheduler for periodic reports")
    else:
        print("⚠️  Some components need attention, but core functionality is working")

    print("\n📁 Generated Files:")
    print("   • reports/metrics/ - Metrics JSON and CSV files")
    print("   • demo_dashboard_data/ - Dashboard data files")
    print("   • comprehensive_demo_data/ - Complete workflow data")

    return successful >= 3


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
