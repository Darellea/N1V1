import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.order_manager import (
    Fill,
    OrderManager,
    PartialFillReconciliationManager,
    PartialFillRecord,
)
from core.types import TradingMode


class TestPartialFillReconciliation:
    """Comprehensive tests for partial fill reconciliation system."""

    @pytest.fixture
    def config(self):
        """Basic config with partial fill settings."""
        return {
            "order": {"base_currency": "USDT", "exchange": "binance"},
            "risk": {},
            "paper": {"initial_balance": 10000.0},
            "reliability": {},
            "partial_fill": {
                "fill_timeout": 5.0,  # 5 seconds for tests
                "max_fill_retries": 2,
                "fill_retry_delay": 0.1,  # Fast retry for tests
                "reconciliation_interval": 1.0,  # Fast reconciliation
                "enable_fill_audit": True
            }
        }

    @pytest.fixture
    def mock_executors(self):
        """Mock all executors."""
        with patch("core.order_manager.LiveOrderExecutor") as mock_live, patch(
            "core.order_manager.PaperOrderExecutor"
        ) as mock_paper, patch(
            "core.order_manager.BacktestOrderExecutor"
        ) as mock_backtest:
            # Mock instances
            mock_live_instance = MagicMock()
            mock_paper_instance = MagicMock()
            mock_backtest_instance = MagicMock()

            mock_live.return_value = mock_live_instance
            mock_paper.return_value = mock_paper_instance
            mock_backtest.return_value = mock_backtest_instance

            # Mock paper executor methods
            mock_paper_instance.execute_paper_order = AsyncMock(
                return_value={"id": "test_order", "status": "filled"}
            )
            mock_paper_instance.get_balance = MagicMock(return_value=Decimal("10000"))
            mock_paper_instance.set_initial_balance = MagicMock()
            mock_paper_instance.set_portfolio_mode = MagicMock()

            yield {
                "live": mock_live_instance,
                "paper": mock_paper_instance,
                "backtest": mock_backtest_instance,
            }

    @pytest.fixture
    def mock_managers(self):
        """Mock reliability and portfolio managers."""
        with patch("core.order_manager.ReliabilityManager") as mock_reliability, patch(
            "core.order_manager.PortfolioManager"
        ) as mock_portfolio, patch(
            "core.order_manager.OrderProcessor"
        ) as mock_processor:
            mock_reliability_instance = MagicMock()
            mock_portfolio_instance = MagicMock()
            mock_processor_instance = MagicMock()

            mock_reliability.return_value = mock_reliability_instance
            mock_portfolio.return_value = mock_portfolio_instance
            mock_processor.return_value = mock_processor_instance

            # Mock reliability manager
            mock_reliability_instance.safe_mode_active = False

            async def mock_retry_async(func, **kwargs):
                if callable(func):
                    result = await func()
                else:
                    result = func
                return result

            mock_reliability_instance.retry_async = AsyncMock(
                side_effect=mock_retry_async
            )
            mock_reliability_instance.record_critical_error = MagicMock()

            # Mock portfolio manager
            mock_portfolio_instance.paper_balances = {"USDT": Decimal("10000")}
            mock_portfolio_instance.set_initial_balance = MagicMock()
            mock_portfolio_instance.initialize_portfolio = MagicMock()

            # Mock order processor
            mock_processor_instance.process_order = AsyncMock(
                return_value={"id": "processed_order", "status": "filled"}
            )
            mock_processor_instance.open_orders = {}
            mock_processor_instance.closed_orders = {}
            mock_processor_instance.positions = {}
            mock_processor_instance.get_active_order_count = MagicMock(return_value=0)
            mock_processor_instance.get_open_position_count = MagicMock(return_value=0)

            yield {
                "reliability": mock_reliability_instance,
                "portfolio": mock_portfolio_instance,
                "processor": mock_processor_instance,
            }

    def test_fill_dataclass_creation(self):
        """Test Fill dataclass creation and properties."""
        fill = Fill(
            order_id="test_order_123",
            symbol="BTC/USDT",
            amount=Decimal("0.5"),
            price=Decimal("50000"),
            timestamp=1234567890.0,
            fill_type="partial",
            exchange_order_id="exch_123",
            fees={"trading": Decimal("0.001")}
        )

        assert fill.order_id == "test_order_123"
        assert fill.symbol == "BTC/USDT"
        assert fill.amount == Decimal("0.5")
        assert fill.price == Decimal("50000")
        assert fill.fill_type == "partial"
        assert fill.exchange_order_id == "exch_123"
        assert fill.fees == {"trading": Decimal("0.001")}

    def test_partial_fill_record_creation(self):
        """Test PartialFillRecord creation and initial state."""
        record = PartialFillRecord(
            original_order_id="order_123",
            symbol="BTC/USDT",
            original_amount=Decimal("1.0")
        )

        assert record.original_order_id == "order_123"
        assert record.symbol == "BTC/USDT"
        assert record.original_amount == Decimal("1.0")
        assert record.filled_amount == Decimal("0")
        assert record.remaining_amount == Decimal("1.0")
        assert record.status == "pending"
        assert record.retry_count == 0
        assert record.max_retries == 3
        assert len(record.audit_trail) == 0

    def test_partial_fill_record_add_fill(self):
        """Test adding fills to a partial fill record."""
        record = PartialFillRecord(
            original_order_id="order_123",
            symbol="BTC/USDT",
            original_amount=Decimal("1.0")
        )

        # Add first partial fill
        fill1 = Fill(
            order_id="order_123",
            symbol="BTC/USDT",
            amount=Decimal("0.3"),
            price=Decimal("50000"),
            timestamp=time.time(),
            fill_type="partial"
        )
        record.add_fill(fill1)

        assert record.filled_amount == Decimal("0.3")
        assert record.remaining_amount == Decimal("0.7")
        assert record.status == "partially_filled"
        assert len(record.fills) == 1
        assert len(record.audit_trail) == 1

        # Add second fill to complete the order
        fill2 = Fill(
            order_id="order_123",
            symbol="BTC/USDT",
            amount=Decimal("0.7"),
            price=Decimal("50100"),
            timestamp=time.time(),
            fill_type="final"
        )
        record.add_fill(fill2)

        assert record.filled_amount == Decimal("1.0")
        assert record.remaining_amount == Decimal("0")
        assert record.status == "fully_filled"
        assert len(record.fills) == 2
        assert len(record.audit_trail) == 2

    def test_partial_fill_record_should_retry(self):
        """Test retry logic for partial fill records."""
        record = PartialFillRecord(
            original_order_id="order_123",
            symbol="BTC/USDT",
            original_amount=Decimal("1.0"),
            max_retries=2
        )

        # Initially should retry
        assert record.should_retry() is True

        # After some retries, still should retry
        record.retry_count = 1
        assert record.should_retry() is True

        # After max retries, should not retry
        record.retry_count = 2
        assert record.should_retry() is False

        # Should not retry if expired
        record.retry_count = 0
        record.created_at = time.time() - 1  # Expire immediately
        record.timeout_seconds = 0.5
        assert record.should_retry() is False

        # Should not retry if manual intervention required
        record.created_at = time.time()
        record.timeout_seconds = 300
        record.manual_intervention_required = True
        assert record.should_retry() is False

    def test_partial_fill_record_timeout(self):
        """Test timeout detection for partial fill records."""
        record = PartialFillRecord(
            original_order_id="order_123",
            symbol="BTC/USDT",
            original_amount=Decimal("1.0"),
            timeout_seconds=1.0
        )

        # Initially not expired
        assert not record.is_expired()

        # Mock expired state
        record.created_at = time.time() - 2.0
        assert record.is_expired()

    def test_partial_fill_record_manual_intervention(self):
        """Test manual intervention marking."""
        record = PartialFillRecord(
            original_order_id="order_123",
            symbol="BTC/USDT",
            original_amount=Decimal("1.0")
        )

        record.mark_for_manual_intervention("Test reason")

        assert record.manual_intervention_required is True
        assert record.status == "manual_intervention"
        assert len(record.audit_trail) == 1
        assert "manual_intervention_required" in record.audit_trail[0]["action"]

    def test_partial_fill_reconciliation_manager_creation(self, config):
        """Test PartialFillReconciliationManager initialization."""
        manager = PartialFillReconciliationManager(config["partial_fill"])

        assert manager.fill_timeout == 5.0
        assert manager.max_retries == 2
        assert manager.retry_delay == 0.1
        assert manager.reconciliation_interval == 1.0
        assert manager.audit_enabled is True
        assert len(manager.partial_fills) == 0
        assert not manager._running

    def test_register_partial_fill(self, config):
        """Test registering a new partial fill."""
        manager = PartialFillReconciliationManager(config["partial_fill"])

        record = manager.register_partial_fill("order_123", "BTC/USDT", Decimal("1.0"))

        assert "order_123" in manager.partial_fills
        assert record.original_order_id == "order_123"
        assert record.symbol == "BTC/USDT"
        assert record.original_amount == Decimal("1.0")
        assert manager.metrics["total_partial_fills"] == 1
        assert len(record.audit_trail) == 1

    def test_add_fill_to_record(self, config):
        """Test adding fills to partial fill records."""
        manager = PartialFillReconciliationManager(config["partial_fill"])

        # Register partial fill
        record = manager.register_partial_fill("order_123", "BTC/USDT", Decimal("1.0"))

        # Add fill
        fill = Fill(
            order_id="order_123",
            symbol="BTC/USDT",
            amount=Decimal("0.5"),
            price=Decimal("50000"),
            timestamp=time.time(),
            fill_type="partial"
        )

        result = manager.add_fill_to_record("order_123", fill)
        assert result is True

        # Check record was updated
        updated_record = manager.partial_fills["order_123"]
        assert updated_record.filled_amount == Decimal("0.5")
        assert updated_record.status == "partially_filled"

    def test_add_fill_to_nonexistent_record(self, config):
        """Test adding fill to non-existent record."""
        manager = PartialFillReconciliationManager(config["partial_fill"])

        fill = Fill(
            order_id="order_123",
            symbol="BTC/USDT",
            amount=Decimal("0.5"),
            price=Decimal("50000"),
            timestamp=time.time(),
            fill_type="partial"
        )

        result = manager.add_fill_to_record("order_123", fill)
        assert result is False

    def test_get_fill_metrics(self, config):
        """Test getting fill reconciliation metrics."""
        manager = PartialFillReconciliationManager(config["partial_fill"])

        # Initially all metrics should be 0
        metrics = manager.get_fill_metrics()
        assert all(v == 0 for v in metrics.values())

        # Register a partial fill
        manager.register_partial_fill("order_123", "BTC/USDT", Decimal("1.0"))
        metrics = manager.get_fill_metrics()
        assert metrics["total_partial_fills"] == 1

    def test_get_pending_fills(self, config):
        """Test getting pending partial fills."""
        manager = PartialFillReconciliationManager(config["partial_fill"])

        # Register partial fills with different statuses
        record1 = manager.register_partial_fill("order_123", "BTC/USDT", Decimal("1.0"))
        record2 = manager.register_partial_fill("order_456", "ETH/USDT", Decimal("2.0"))
        record3 = manager.register_partial_fill("order_789", "ADA/USDT", Decimal("3.0"))

        # Mark one as fully filled
        record2.status = "fully_filled"

        pending = manager.get_pending_fills()
        assert len(pending) == 2
        assert all(r.status in ["pending", "partially_filled"] for r in pending)

    def test_get_stuck_fills(self, config):
        """Test getting stuck partial fills."""
        manager = PartialFillReconciliationManager(config["partial_fill"])

        # Register partial fills
        record1 = manager.register_partial_fill("order_123", "BTC/USDT", Decimal("1.0"))
        record2 = manager.register_partial_fill("order_456", "ETH/USDT", Decimal("2.0"))

        # Mark one for manual intervention and expire another
        record1.mark_for_manual_intervention("Test")
        record2.created_at = time.time() - 10.0  # Expire

        stuck = manager.get_stuck_fills()
        assert len(stuck) == 2

    @pytest.mark.asyncio
    async def test_reconciliation_loop_start_stop(self, config):
        """Test starting and stopping the reconciliation loop."""
        manager = PartialFillReconciliationManager(config["partial_fill"])

        assert not manager._running

        # Start the loop
        await manager.start_reconciliation_loop()
        assert manager._running
        assert manager._reconciliation_task is not None

        # Stop the loop
        await manager.stop_reconciliation_loop()
        assert not manager._running

        # Task should be cancelled
        await asyncio.sleep(0.1)  # Allow cancellation to complete
        assert manager._reconciliation_task.cancelled()

    @pytest.mark.asyncio
    async def test_reconciliation_loop_timeout_handling(self, config):
        """Test reconciliation loop handles timeouts correctly."""
        manager = PartialFillReconciliationManager(config["partial_fill"])

        # Register a partial fill and expire it
        record = manager.register_partial_fill("order_123", "BTC/USDT", Decimal("1.0"))
        record.created_at = time.time() - 10.0  # Expire immediately

        # Start reconciliation
        await manager.start_reconciliation_loop()
        await asyncio.sleep(1.1)  # Allow reconciliation to run

        # Stop and check results
        await manager.stop_reconciliation_loop()

        # Record should be marked as timed out
        assert record.status == "timed_out"
        assert manager.metrics["timed_out_fills"] == 1

    @pytest.mark.asyncio
    async def test_order_manager_process_fill_report_full_fill(self, config, mock_executors, mock_managers):
        """Test processing a full fill report."""
        om = OrderManager(config, TradingMode.PAPER)

        fill_data = {
            "order_id": "order_123",
            "symbol": "BTC/USDT",
            "amount": "1.0",
            "price": "50000",
            "fill_type": "full"
        }

        await om.process_fill_report(fill_data)

        # Should update position directly (not create partial fill record)
        assert "BTC/USDT" in om.order_processor.positions
        position = om.order_processor.positions["BTC/USDT"]
        assert position["amount"] == 1.0
        assert position["entry_price"] == 50000.0

    @pytest.mark.asyncio
    async def test_order_manager_process_fill_report_partial_fill(self, config, mock_executors, mock_managers):
        """Test processing a partial fill report."""
        om = OrderManager(config, TradingMode.PAPER)

        # First partial fill
        fill_data1 = {
            "order_id": "order_123",
            "symbol": "BTC/USDT",
            "amount": "0.5",
            "price": "50000",
            "fill_type": "partial",
            "original_amount": "1.0"
        }

        await om.process_fill_report(fill_data1)

        # Should create partial fill record
        assert "order_123" in om.fill_reconciler.partial_fills
        record = om.fill_reconciler.partial_fills["order_123"]
        assert record.filled_amount == Decimal("0.5")
        assert record.remaining_amount == Decimal("0.5")
        assert record.status == "partially_filled"

        # Second partial fill to complete
        fill_data2 = {
            "order_id": "order_123",
            "symbol": "BTC/USDT",
            "amount": "0.5",
            "price": "50100",
            "fill_type": "final"
        }

        await om.process_fill_report(fill_data2)

        # Should complete the fill and update position
        record = om.fill_reconciler.partial_fills["order_123"]
        assert record.filled_amount == Decimal("1.0")
        assert record.status == "fully_filled"

        # Position should be updated with volume-weighted average price
        assert "BTC/USDT" in om.order_processor.positions
        position = om.order_processor.positions["BTC/USDT"]
        assert position["amount"] == 1.0
        # Average price: (0.5 * 50000 + 0.5 * 50100) / 1.0 = 50050.0
        assert position["entry_price"] == 50050.0

    @pytest.mark.asyncio
    async def test_order_manager_process_fill_report_missing_order_id(self, config, mock_executors, mock_managers):
        """Test processing fill report with missing order_id."""
        om = OrderManager(config, TradingMode.PAPER)

        fill_data = {
            "symbol": "BTC/USDT",
            "amount": "1.0",
            "price": "50000"
        }

        # Should not raise exception, just log warning
        await om.process_fill_report(fill_data)

        # No partial fills should be created
        assert len(om.fill_reconciler.partial_fills) == 0

    @pytest.mark.asyncio
    async def test_order_manager_process_fill_report_invalid_data(self, config, mock_executors, mock_managers):
        """Test processing fill report with invalid data."""
        om = OrderManager(config, TradingMode.PAPER)

        fill_data = {
            "order_id": "order_123",
            "symbol": "BTC/USDT",
            "amount": "invalid_amount",
            "price": "50000"
        }

        await om.process_fill_report(fill_data)

        # Should increment exchange discrepancies metric
        assert om.fill_reconciler.metrics["exchange_discrepancies"] == 1

    @pytest.mark.asyncio
    async def test_order_manager_partial_fill_retry(self, config, mock_executors, mock_managers):
        """Test partial fill retry mechanism."""
        om = OrderManager(config, TradingMode.PAPER)

        # Mock execute_order to return a successful result
        original_execute = om.execute_order
        async def mock_execute_order(signal, **kwargs):
            return {
                "id": f"retry_{signal.get('idempotency_key', 'unknown')}",
                "status": "filled",
                "amount": signal.get("amount", 0),
                "price": 50000.0
            }
        om.execute_order = mock_execute_order

        try:
            # Create partial fill record
            record = om.fill_reconciler.register_partial_fill("order_123", "BTC/USDT", Decimal("1.0"))

            # Add partial fill
            fill = Fill(
                order_id="order_123",
                symbol="BTC/USDT",
                amount=Decimal("0.5"),
                price=Decimal("50000"),
                timestamp=time.time(),
                fill_type="partial"
            )
            om.fill_reconciler.add_fill_to_record("order_123", fill)

            # Trigger retry
            await om._retry_partial_fill(record)

            # Should increment retry count
            assert record.retry_count == 1

        finally:
            # Restore original method
            om.execute_order = original_execute

    @pytest.mark.asyncio
    async def test_order_manager_get_fill_reconciliation_metrics(self, config, mock_executors, mock_managers):
        """Test getting fill reconciliation metrics from OrderManager."""
        om = OrderManager(config, TradingMode.PAPER)

        # Add some test data
        om.fill_reconciler.register_partial_fill("order_123", "BTC/USDT", Decimal("1.0"))
        om.fill_reconciler.metrics["successful_reconciliations"] = 5

        metrics = om.get_fill_reconciliation_metrics()

        assert metrics["total_partial_fills"] == 1
        assert metrics["successful_reconciliations"] == 5

    @pytest.mark.asyncio
    async def test_order_manager_get_pending_partial_fills(self, config, mock_executors, mock_managers):
        """Test getting pending partial fills from OrderManager."""
        om = OrderManager(config, TradingMode.PAPER)

        # Create some test records
        record1 = om.fill_reconciler.register_partial_fill("order_123", "BTC/USDT", Decimal("1.0"))
        record2 = om.fill_reconciler.register_partial_fill("order_456", "ETH/USDT", Decimal("2.0"))
        record2.status = "fully_filled"  # Not pending

        pending = om.get_pending_partial_fills()

        assert len(pending) == 1
        assert pending[0].original_order_id == "order_123"

    @pytest.mark.asyncio
    async def test_order_manager_get_stuck_partial_fills(self, config, mock_executors, mock_managers):
        """Test getting stuck partial fills from OrderManager."""
        om = OrderManager(config, TradingMode.PAPER)

        # Create test records
        record1 = om.fill_reconciler.register_partial_fill("order_123", "BTC/USDT", Decimal("1.0"))
        record2 = om.fill_reconciler.register_partial_fill("order_456", "ETH/USDT", Decimal("2.0"))

        # Make them stuck
        record1.mark_for_manual_intervention("Test")
        record2.created_at = time.time() - 10.0  # Expire

        stuck = om.get_stuck_partial_fills()

        assert len(stuck) == 2

    @pytest.mark.asyncio
    async def test_order_manager_start_fill_reconciliation(self, config, mock_executors, mock_managers):
        """Test starting fill reconciliation from OrderManager."""
        om = OrderManager(config, TradingMode.PAPER)

        await om.start_fill_reconciliation()

        assert om.fill_reconciler._running

        # Cleanup
        await om.fill_reconciler.stop_reconciliation_loop()

    @pytest.mark.asyncio
    async def test_order_manager_shutdown_stops_reconciliation(self, config, mock_executors, mock_managers):
        """Test that shutdown stops fill reconciliation."""
        om = OrderManager(config, TradingMode.PAPER)

        # Start reconciliation
        await om.start_fill_reconciliation()
        assert om.fill_reconciler._running

        # Shutdown should stop it
        await om.shutdown()
        assert not om.fill_reconciler._running

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_partial_fill_timeout_handling(self):
        """Test partial fill timeout handling with actual timeout."""
        config = {
            "order": {"base_currency": "USDT"},
            "risk": {},
            "paper": {"initial_balance": 10000.0},
            "reliability": {},
            "partial_fill": {
                "fill_timeout": 2.0,  # Short timeout for test
                "max_fill_retries": 1,
                "fill_retry_delay": 0.1,
                "reconciliation_interval": 0.5,
                "enable_fill_audit": True
            }
        }

        with patch("core.order_manager.LiveOrderExecutor"), \
             patch("core.order_manager.PaperOrderExecutor"), \
             patch("core.order_manager.BacktestOrderExecutor"), \
             patch("core.order_manager.ReliabilityManager"), \
             patch("core.order_manager.PortfolioManager"), \
             patch("core.order_manager.OrderProcessor"):

            om = OrderManager(config, TradingMode.PAPER)

            # Register a partial fill
            record = om.fill_reconciler.register_partial_fill("order_123", "BTC/USDT", Decimal("1.0"))

            # Start reconciliation
            await om.start_fill_reconciliation()

            # Wait for timeout (longer than fill_timeout)
            await asyncio.sleep(3.0)

            # Stop reconciliation
            await om.stop_fill_reconciliation()

            # Record should be timed out
            assert record.status == "timed_out"
            assert om.fill_reconciler.metrics["timed_out_fills"] == 1

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_partial_fill_hanging_reconciliation_timeout(self):
        """Test that hanging reconciliation attempts don't cause infinite hangs."""
        config = {
            "order": {"base_currency": "USDT"},
            "risk": {},
            "paper": {"initial_balance": 10000.0},
            "reliability": {},
            "partial_fill": {
                "fill_timeout": 5.0,
                "max_fill_retries": 1,
                "fill_retry_delay": 0.1,
                "reconciliation_interval": 1.0,
                "enable_fill_audit": True
            }
        }

        with patch("core.order_manager.LiveOrderExecutor"), \
             patch("core.order_manager.PaperOrderExecutor"), \
             patch("core.order_manager.BacktestOrderExecutor"), \
             patch("core.order_manager.ReliabilityManager"), \
             patch("core.order_manager.PortfolioManager"), \
             patch("core.order_manager.OrderProcessor"):

            om = OrderManager(config, TradingMode.PAPER)

            # Mock reconciliation to hang
            original_attempt = om.fill_reconciler._attempt_reconciliation

            async def hanging_reconciliation(record):
                await asyncio.sleep(10)  # Would hang without timeout protection

            om.fill_reconciler._attempt_reconciliation = hanging_reconciliation

            try:
                # Register a partial fill
                record = om.fill_reconciler.register_partial_fill("order_123", "BTC/USDT", Decimal("1.0"))

                # Start reconciliation with timeout protection
                await om.start_fill_reconciliation()

                # Wait a bit then stop - should not hang
                await asyncio.sleep(2.0)
                await om.stop_fill_reconciliation()

                # Should have completed without hanging
                assert True  # If we get here, no hang occurred

            finally:
                om.fill_reconciler._attempt_reconciliation = original_attempt

    @pytest.mark.asyncio
    async def test_exchange_compatibility_different_fill_formats(self, config, mock_executors, mock_managers):
        """Test handling different exchange fill report formats."""
        om = OrderManager(config, TradingMode.PAPER)

        # Test various fill formats that different exchanges might send
        test_cases = [
            # Standard format - full fill
            {
                "order_id": "order_1",
                "symbol": "BTC/USDT",
                "amount": "1.0",
                "price": "50000",
                "fill_type": "full"
            },
            # Format with fees - partial fill (creates record but doesn't update position yet)
            {
                "order_id": "order_2",
                "symbol": "ETH/USDT",
                "amount": "2.0",
                "price": "3000",
                "fill_type": "partial",
                "original_amount": "2.0",  # Same as amount = full fill
                "fees": {"trading": "0.001"}
            },
            # Format with exchange order ID - final fill for a previous partial
            {
                "order_id": "order_3",
                "symbol": "ADA/USDT",
                "amount": "1000",
                "price": "1.50",
                "fill_type": "final",
                "original_amount": "1000",  # Same as amount = full fill
                "exchange_order_id": "exch_12345"
            }
        ]

        for fill_data in test_cases:
            await om.process_fill_report(fill_data)

        # Should handle all formats without errors
        assert om.fill_reconciler.metrics["exchange_discrepancies"] == 0

        # Check that positions were updated appropriately
        # Full fills should update positions immediately
        assert "BTC/USDT" in om.order_processor.positions
        assert "ETH/USDT" in om.order_processor.positions
        assert "ADA/USDT" in om.order_processor.positions

    @pytest.mark.asyncio
    async def test_manual_intervention_workflow(self, config, mock_executors, mock_managers):
        """Test manual intervention workflow for stuck orders."""
        om = OrderManager(config, TradingMode.PAPER)

        # Create a stuck partial fill
        record = om.fill_reconciler.register_partial_fill("stuck_order", "BTC/USDT", Decimal("1.0"))

        # Simulate multiple failed reconciliation attempts
        for i in range(6):  # More than max attempts
            record.reconciliation_attempts = i + 1
            await om.fill_reconciler._attempt_reconciliation(record)

        # Should be marked for manual intervention
        assert record.manual_intervention_required is True
        assert record.status == "manual_intervention"
        assert om.fill_reconciler.metrics["manual_interventions"] == 1

        # Should appear in stuck fills list
        stuck_fills = om.get_stuck_partial_fills()
        assert len(stuck_fills) == 1
        assert stuck_fills[0].original_order_id == "stuck_order"

    @pytest.mark.asyncio
    async def test_position_management_integration(self, config, mock_executors, mock_managers):
        """Test integration with position management."""
        om = OrderManager(config, TradingMode.PAPER)

        # Simulate multiple partial fills for the same symbol
        fills_data = [
            {
                "order_id": "order_1",
                "symbol": "BTC/USDT",
                "amount": "0.5",
                "price": "50000",
                "fill_type": "partial",
                "original_amount": "1.0"
            },
            {
                "order_id": "order_1",
                "symbol": "BTC/USDT",
                "amount": "0.5",
                "price": "50100",
                "fill_type": "final"
            },
            # Different order for same symbol
            {
                "order_id": "order_2",
                "symbol": "BTC/USDT",
                "amount": "0.3",
                "price": "50200",
                "fill_type": "full"
            }
        ]

        for fill_data in fills_data:
            await om.process_fill_report(fill_data)

        # Check final position
        position = om.order_processor.positions["BTC/USDT"]
        # Total amount: 0.5 + 0.5 + 0.3 = 1.3
        assert position["amount"] == 1.3

        # Volume-weighted average price:
        # (0.5 * 50000 + 0.5 * 50100 + 0.3 * 50200) / 1.3
        expected_avg = (0.5 * 50000 + 0.5 * 50100 + 0.3 * 50200) / 1.3
        assert abs(position["entry_price"] - expected_avg) < 0.01  # Small tolerance for float comparison

        # Should have both order IDs in position
        assert "order_1" in position["orders"]
        assert "order_2" in position["orders"]
