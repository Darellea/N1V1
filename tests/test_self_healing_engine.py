"""
Unit and integration tests for Self-Healing Engine feature.

Tests cover:
- Watchdog service monitoring and heartbeat processing
- Failure detection and diagnosis algorithms
- Recovery orchestration and state management
- Emergency procedures and circuit breakers
- Component registry and health monitoring
- Integration with existing N1V1 components
- Performance benchmarking and reliability testing
- Edge cases and error handling
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import psutil
import os

from core.self_healing_engine import (
    SelfHealingEngine, ComponentType, ComponentStatus,
    ComponentRegistry, HealingOrchestrator, EmergencyProcedures,
    MonitoringDashboard
)
from core.watchdog import (
    WatchdogService, HeartbeatProtocol, FailureDetector,
    RecoveryOrchestrator, StateManager, ComponentStatus as WatchdogStatus,
    FailureSeverity, FailureType, HeartbeatMessage
)
from core.diagnostics import HealthStatus
from core.signal_router.event_bus import get_default_enhanced_event_bus


class TestComponentRegistry:
    """Test ComponentRegistry functionality."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = ComponentRegistry()

        assert len(registry.components) == 0
        assert len(registry.component_types) == 0

    def test_register_component(self):
        """Test component registration."""
        registry = ComponentRegistry()

        # Mock component
        mock_component = Mock()
        mock_component.component_id = "test_component"

        info = registry.register_component(
            "test_component",
            ComponentType.BOT_ENGINE,
            mock_component,
            critical=True,
            dependencies=["dep1", "dep2"]
        )

        assert info.component_id == "test_component"
        assert info.component_type == ComponentType.BOT_ENGINE
        assert info.critical is True
        assert info.dependencies == ["dep1", "dep2"]
        assert "test_component" in registry.components

    def test_unregister_component(self):
        """Test component unregistration."""
        registry = ComponentRegistry()

        mock_component = Mock()
        registry.register_component("test_comp", ComponentType.BOT_ENGINE, mock_component)

        # Unregister
        registry.unregister_component("test_comp")

        assert "test_comp" not in registry.components

    def test_get_component(self):
        """Test component retrieval."""
        registry = ComponentRegistry()

        mock_component = Mock()
        registry.register_component("test_comp", ComponentType.BOT_ENGINE, mock_component)

        info = registry.get_component("test_comp")
        assert info is not None
        assert info.component_id == "test_comp"

        # Non-existent component
        info = registry.get_component("nonexistent")
        assert info is None

    def test_get_components_by_type(self):
        """Test getting components by type."""
        registry = ComponentRegistry()

        # Register multiple components
        mock_comp1 = Mock()
        mock_comp2 = Mock()

        registry.register_component("bot1", ComponentType.BOT_ENGINE, mock_comp1)
        registry.register_component("bot2", ComponentType.BOT_ENGINE, mock_comp2)
        registry.register_component("strategy1", ComponentType.STRATEGY, Mock())

        bot_components = registry.get_components_by_type(ComponentType.BOT_ENGINE)
        assert len(bot_components) == 2
        assert all(c.component_type == ComponentType.BOT_ENGINE for c in bot_components)

    def test_get_critical_components(self):
        """Test getting critical components."""
        registry = ComponentRegistry()

        # Register critical and non-critical components
        registry.register_component("critical1", ComponentType.BOT_ENGINE, Mock(), critical=True)
        registry.register_component("critical2", ComponentType.DATABASE, Mock(), critical=True)
        registry.register_component("normal1", ComponentType.STRATEGY, Mock(), critical=False)

        critical_components = registry.get_critical_components()
        assert len(critical_components) == 2
        assert all(c.critical for c in critical_components)

    def test_update_component_status(self):
        """Test component status updates."""
        registry = ComponentRegistry()

        mock_component = Mock()
        registry.register_component("test_comp", ComponentType.BOT_ENGINE, mock_component)

        # Update status
        registry.update_component_status("test_comp", healthy=True)
        info = registry.get_component("test_comp")
        assert info.consecutive_failures == 0

        # Update with failure
        registry.update_component_status("test_comp", healthy=False)
        info = registry.get_component("test_comp")
        assert info.consecutive_failures == 1

    def test_get_registry_stats(self):
        """Test registry statistics."""
        registry = ComponentRegistry()

        # Register components
        registry.register_component("comp1", ComponentType.BOT_ENGINE, Mock(), critical=True)
        registry.register_component("comp2", ComponentType.STRATEGY, Mock(), critical=False)
        registry.register_component("comp3", ComponentType.STRATEGY, Mock(), critical=False)

        stats = registry.get_registry_stats()

        assert stats['total_components'] == 3
        assert stats['critical_components'] == 1
        assert stats['healthy_components'] == 3  # All healthy initially
        assert stats['failing_components'] == 0


class TestHeartbeatProtocol:
    """Test HeartbeatProtocol functionality."""

    def test_protocol_initialization(self):
        """Test heartbeat protocol initialization."""
        protocol = HeartbeatProtocol("test_component", "test_type")

        assert protocol.component_id == "test_component"
        assert protocol.component_type == "test_type"
        assert protocol._heartbeat_interval == 30
        assert protocol._last_heartbeat is None

    def test_heartbeat_creation(self):
        """Test heartbeat message creation."""
        protocol = HeartbeatProtocol("test_comp", "test_type")

        heartbeat = protocol.create_heartbeat(
            status=ComponentStatus.HEALTHY,
            latency_ms=45.2,
            error_count=0
        )

        assert heartbeat.component_id == "test_comp"
        assert heartbeat.component_type == "test_type"
        assert heartbeat.status == ComponentStatus.HEALTHY
        assert heartbeat.latency_ms == 45.2
        assert heartbeat.error_count == 0
        assert isinstance(heartbeat.timestamp, datetime)

        # Should have system metrics
        assert heartbeat.memory_mb is not None
        assert heartbeat.cpu_percent is not None

    def test_heartbeat_overdue_check(self):
        """Test heartbeat overdue detection."""
        protocol = HeartbeatProtocol("test_comp", "test_type")
        protocol.set_heartbeat_interval(1)  # 1 second for testing

        # No heartbeat yet
        assert protocol.is_heartbeat_overdue() is True

        # Create heartbeat
        protocol.create_heartbeat()

        # Should not be overdue immediately
        assert protocol.is_heartbeat_overdue() is False

        # Wait for overdue
        import time
        time.sleep(2)

        assert protocol.is_heartbeat_overdue() is True

    def test_custom_metrics_provider(self):
        """Test custom metrics provider."""
        protocol = HeartbeatProtocol("test_comp", "test_type")

        def custom_metrics():
            return {"custom_metric": 42, "another_metric": "test"}

        protocol.register_metric_provider(custom_metrics)

        heartbeat = protocol.create_heartbeat()

        assert "custom_metric" in heartbeat.custom_metrics
        assert heartbeat.custom_metrics["custom_metric"] == 42


class TestFailureDetector:
    """Test FailureDetector functionality."""

    def test_detector_initialization(self):
        """Test failure detector initialization."""
        config = {"anomaly_threshold": 2.0, "history_window": 50}
        detector = FailureDetector(config)

        assert detector.anomaly_threshold == 2.0
        assert detector.history_window == 50
        assert len(detector.heartbeat_history) == 0

    def test_heartbeat_processing(self):
        """Test heartbeat processing and storage."""
        detector = FailureDetector({})

        # Create test heartbeat
        heartbeat = HeartbeatMessage(
            component_id="test_comp",
            component_type="test_type",
            version="1.0",
            timestamp=datetime.now(),
            status=ComponentStatus.HEALTHY,
            latency_ms=50.0
        )

        diagnosis = detector.process_heartbeat(heartbeat)

        # Should have stored heartbeat
        assert "test_comp" in detector.heartbeat_history
        assert len(detector.heartbeat_history["test_comp"]) == 1

        # Should not detect failure with single heartbeat
        assert diagnosis is None

    def test_anomaly_detection_latency(self):
        """Test latency anomaly detection."""
        detector = FailureDetector({"anomaly_threshold": 2.0})

        # Create normal heartbeats
        for i in range(10):
            heartbeat = HeartbeatMessage(
                component_id="test_comp",
                component_type="test_type",
                version="1.0",
                timestamp=datetime.now(),
                status=ComponentStatus.HEALTHY,
                latency_ms=50.0  # Normal latency
            )
            detector.process_heartbeat(heartbeat)

        # Create anomalous heartbeat
        anomalous_heartbeat = HeartbeatMessage(
            component_id="test_comp",
            component_type="test_type",
            version="1.0",
            timestamp=datetime.now(),
            status=ComponentStatus.HEALTHY,
            latency_ms=200.0  # High latency (anomaly)
        )

        diagnosis = detector.process_heartbeat(anomalous_heartbeat)

        # Should detect anomaly
        assert diagnosis is not None
        assert diagnosis.failure_type == FailureType.PERFORMANCE
        assert diagnosis.severity == FailureSeverity.MEDIUM

    def test_baseline_calculation(self):
        """Test baseline metrics calculation."""
        detector = FailureDetector({"min_samples_for_baseline": 5})

        # Add heartbeats
        for i in range(10):
            heartbeat = HeartbeatMessage(
                component_id="test_comp",
                component_type="test_type",
                version="1.0",
                timestamp=datetime.now(),
                status=ComponentStatus.HEALTHY,
                latency_ms=50.0 + i,  # Varying latency
                memory_mb=100.0
            )
            detector.process_heartbeat(heartbeat)

        # Should have calculated baseline
        assert "test_comp" in detector.baseline_metrics
        baseline = detector.baseline_metrics["test_comp"]

        assert "latency_mean" in baseline
        assert "latency_std" in baseline
        assert baseline["latency_mean"] > 0

    def test_failure_type_classification(self):
        """Test failure type classification."""
        detector = FailureDetector({})

        # Test different anomaly types
        test_cases = [
            ("latency_spike", FailureType.PERFORMANCE),
            ("memory_leak", FailureType.RESOURCE),
            ("error_rate_spike", FailureType.LOGIC),
            ("status_degradation", FailureType.CONNECTIVITY)
        ]

        for anomaly_type, expected_type in test_cases:
            failure_type = detector._classify_failure_type(anomaly_type)
            assert failure_type == expected_type


class TestRecoveryOrchestrator:
    """Test RecoveryOrchestrator functionality."""

    def test_orchestrator_initialization(self):
        """Test recovery orchestrator initialization."""
        config = {"recovery_timeout": 300}
        registry = ComponentRegistry()
        orchestrator = HealingOrchestrator(config, registry)

        assert orchestrator.config == config
        assert orchestrator.registry == registry
        assert len(orchestrator.pending_actions) == 0

    @pytest.mark.asyncio
    async def test_recovery_initiation(self):
        """Test recovery action initiation."""
        registry = ComponentRegistry()
        orchestrator = HealingOrchestrator({}, registry)

        # Mock component
        mock_component = Mock()
        registry.register_component("test_comp", ComponentType.BOT_ENGINE, mock_component)

        # Mock diagnosis
        diagnosis = Mock()
        diagnosis.component_id = "test_comp"
        diagnosis.failure_type = FailureType.CONNECTIVITY
        diagnosis.severity = FailureSeverity.HIGH
        diagnosis.estimated_recovery_time = 60

        action = await orchestrator.initiate_healing("test_comp", FailureType.CONNECTIVITY,
                                                   FailureSeverity.HIGH, diagnosis.to_dict())

        assert action is not None
        assert action.component_id == "test_comp"
        assert action.action_type == "test_recovery_connectivity"
        assert "test_comp" in orchestrator.pending_actions

    def test_priority_calculation(self):
        """Test recovery priority calculation."""
        orchestrator = HealingOrchestrator({}, ComponentRegistry())

        # Test different severity levels
        assert orchestrator._calculate_priority(FailureSeverity.CRITICAL) == 10
        assert orchestrator._calculate_priority(FailureSeverity.HIGH) == 7
        assert orchestrator._calculate_priority(FailureSeverity.MEDIUM) == 5
        assert orchestrator._calculate_priority(FailureSeverity.LOW) == 3

    def test_timeout_calculation(self):
        """Test recovery timeout calculation."""
        orchestrator = HealingOrchestrator({}, ComponentRegistry())

        # Test different combinations
        timeout = orchestrator._calculate_timeout(FailureSeverity.CRITICAL, FailureType.CONNECTIVITY)
        assert timeout == 60  # Reduced for connectivity

        timeout = orchestrator._calculate_timeout(FailureSeverity.HIGH, FailureType.RESOURCE)
        assert timeout == 480  # Increased for resource issues

    def test_recovery_stats(self):
        """Test recovery statistics."""
        orchestrator = HealingOrchestrator({}, ComponentRegistry())

        # Mock actions
        orchestrator.completed_actions = [Mock()] * 5
        orchestrator.failed_actions = [Mock()] * 2

        stats = orchestrator.get_healing_stats()

        assert stats['completed_actions'] == 5
        assert stats['failed_actions'] == 2
        assert stats['success_rate'] == 5 / 7  # 5/7 ≈ 0.714


class TestStateManager:
    """Test StateManager functionality."""

    def test_state_manager_initialization(self):
        """Test state manager initialization."""
        manager = StateManager({})

        assert len(manager.state_snapshots) == 0
        assert len(manager.state_versions) == 0

    def test_snapshot_creation(self):
        """Test state snapshot creation."""
        manager = StateManager({})

        state_data = {"balance": 10000.0, "positions": []}
        snapshot_id = manager.create_snapshot("test_comp", state_data)

        assert snapshot_id.startswith("test_comp_")
        assert snapshot_id in manager.state_snapshots
        assert manager.state_snapshots[snapshot_id]['data'] == state_data

    def test_snapshot_restoration(self):
        """Test state snapshot restoration."""
        manager = StateManager({})

        original_state = {"balance": 10000.0, "positions": ["BTC"]}
        snapshot_id = manager.create_snapshot("test_comp", original_state)

        # Restore snapshot
        restored_state = manager.restore_snapshot("test_comp", snapshot_id)

        assert restored_state == original_state

        # Restore latest snapshot
        restored_state = manager.restore_snapshot("test_comp")

        assert restored_state == original_state

    def test_snapshot_versioning(self):
        """Test snapshot versioning."""
        manager = StateManager({})

        # Create multiple snapshots
        for i in range(3):
            state_data = {"version": i}
            manager.create_snapshot("test_comp", state_data)

        # Should have version history
        assert "test_comp" in manager.state_versions
        assert len(manager.state_versions["test_comp"]) == 3

    def test_snapshot_cleanup(self):
        """Test snapshot cleanup (keeping only recent ones)."""
        manager = StateManager({})

        # Create many snapshots
        for i in range(15):
            state_data = {"version": i}
            manager.create_snapshot("test_comp", state_data)

        # Should only keep 10 most recent
        assert len(manager.state_versions["test_comp"]) == 10


class TestWatchdogService:
    """Test WatchdogService functionality."""

    def test_watchdog_initialization(self):
        """Test watchdog service initialization."""
        config = {"heartbeat_interval": 30}
        watchdog = WatchdogService(config)

        assert len(watchdog.heartbeat_protocols) == 0
        assert watchdog.heartbeats_received == 0
        assert watchdog.failures_detected == 0

    def test_component_registration(self):
        """Test component registration with watchdog."""
        watchdog = WatchdogService({})

        protocol = watchdog.register_component("test_comp", "test_type", 15)

        assert "test_comp" in watchdog.heartbeat_protocols
        assert protocol.component_id == "test_comp"
        assert protocol._heartbeat_interval == 15

    @pytest.mark.asyncio
    async def test_heartbeat_processing(self):
        """Test heartbeat processing."""
        watchdog = WatchdogService({})

        # Register component
        watchdog.register_component("test_comp", "test_type")

        # Create heartbeat
        heartbeat = HeartbeatMessage(
            component_id="test_comp",
            component_type="test_type",
            version="1.0",
            timestamp=datetime.now(),
            status=ComponentStatus.HEALTHY
        )

        # Process heartbeat
        await watchdog.receive_heartbeat(heartbeat)

        assert watchdog.heartbeats_received == 1

    @pytest.mark.asyncio
    async def test_overdue_heartbeat_detection(self):
        """Test overdue heartbeat detection."""
        watchdog = WatchdogService({})

        # Register component with short interval
        protocol = watchdog.register_component("test_comp", "test_type", 1)

        # Wait for overdue
        await asyncio.sleep(2)

        # Check for overdue heartbeats
        await watchdog._check_overdue_heartbeats()

        # Should have detected failure
        assert watchdog.failures_detected > 0

    def test_watchdog_stats(self):
        """Test watchdog statistics."""
        watchdog = WatchdogService({})

        # Mock some activity
        watchdog.heartbeats_received = 100
        watchdog.failures_detected = 5
        watchdog.recoveries_initiated = 3
        watchdog.recoveries_successful = 2

        stats = watchdog.get_watchdog_stats()

        assert stats['heartbeats_received'] == 100
        assert stats['failures_detected'] == 5
        assert stats['recoveries_initiated'] == 3
        assert stats['recoveries_successful'] == 2

    def test_component_status(self):
        """Test component status retrieval."""
        watchdog = WatchdogService({})

        # Register component
        watchdog.register_component("test_comp", "test_type")

        status = watchdog.get_component_status("test_comp")

        assert status is not None
        assert status['component_id'] == "test_comp"
        assert status['component_type'] == "test_type"

        # Non-existent component
        status = watchdog.get_component_status("nonexistent")
        assert status is None


class TestEmergencyProcedures:
    """Test EmergencyProcedures functionality."""

    def test_emergency_initialization(self):
        """Test emergency procedures initialization."""
        registry = ComponentRegistry()
        procedures = EmergencyProcedures({}, registry)

        assert procedures.emergency_mode is False
        assert procedures.emergency_start_time is None

    @pytest.mark.asyncio
    async def test_emergency_activation(self):
        """Test emergency mode activation."""
        registry = ComponentRegistry()
        procedures = EmergencyProcedures({}, registry)

        await procedures.activate_emergency_mode("Test emergency")

        assert procedures.emergency_mode is True
        assert procedures.emergency_start_time is not None

    @pytest.mark.asyncio
    async def test_emergency_deactivation(self):
        """Test emergency mode deactivation."""
        registry = ComponentRegistry()
        procedures = EmergencyProcedures({}, registry)

        await procedures.activate_emergency_mode("Test emergency")
        await procedures.deactivate_emergency_mode()

        assert procedures.emergency_mode is False
        assert procedures.emergency_start_time is None

    def test_emergency_duration(self):
        """Test emergency duration calculation."""
        registry = ComponentRegistry()
        procedures = EmergencyProcedures({}, registry)

        # No emergency active
        duration = procedures.get_emergency_duration()
        assert duration is None

        # Emergency active
        procedures.emergency_mode = True
        procedures.emergency_start_time = datetime.now() - timedelta(minutes=5)

        duration = procedures.get_emergency_duration()
        assert duration is not None
        assert duration.total_seconds() >= 300  # At least 5 minutes


class TestMonitoringDashboard:
    """Test MonitoringDashboard functionality."""

    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        registry = ComponentRegistry()
        watchdog = WatchdogService({})
        dashboard = MonitoringDashboard({}, registry, watchdog)

        assert dashboard.registry == registry
        assert dashboard.watchdog == watchdog

    def test_system_health_calculation(self):
        """Test system health calculation."""
        registry = ComponentRegistry()
        watchdog = WatchdogService({})
        dashboard = MonitoringDashboard({}, registry, watchdog)

        # Mock registry stats
        registry.get_registry_stats = Mock(return_value={
            'total_components': 10,
            'healthy_components': 8,
            'failing_components': 2
        })

        # Mock watchdog stats
        watchdog.get_watchdog_stats = Mock(return_value={
            'heartbeats_received': 100,
            'failures_detected': 2
        })

        data = dashboard.get_dashboard_data()
        system_health = data['system_health']

        assert system_health['overall_health'] == "DEGRADED"  # 20% failing
        assert system_health['health_score'] == 75  # 75% healthy

    def test_component_status_summary(self):
        """Test component status summary."""
        registry = ComponentRegistry()
        watchdog = WatchdogService({})
        dashboard = MonitoringDashboard({}, registry, watchdog)

        # Register components
        registry.register_component("comp1", ComponentType.BOT_ENGINE, Mock(), critical=True)
        registry.register_component("comp2", ComponentType.STRATEGY, Mock(), critical=False)

        data = dashboard.get_dashboard_data()
        component_status = data['component_status']

        assert len(component_status) == 2
        assert component_status[0]['component_id'] == "comp1"
        assert component_status[1]['component_id'] == "comp2"

    def test_failure_statistics(self):
        """Test failure statistics calculation."""
        registry = ComponentRegistry()
        watchdog = WatchdogService({})
        dashboard = MonitoringDashboard({}, registry, watchdog)

        # Mock watchdog stats
        watchdog.get_watchdog_stats = Mock(return_value={
            'failures_detected': 5,
            'recoveries_initiated': 5,
            'recoveries_successful': 4
        })

        data = dashboard.get_dashboard_data()
        failure_stats = data['failure_stats']

        assert failure_stats['total_failures'] == 5
        assert failure_stats['recovery_attempts'] == 5
        assert failure_stats['successful_recoveries'] == 4
        assert failure_stats['recovery_success_rate'] == 80.0


class TestSelfHealingEngine:
    """Test SelfHealingEngine integration."""

    def test_engine_initialization(self):
        """Test self-healing engine initialization."""
        config = {"monitoring": {"heartbeat_interval": 30}}
        engine = SelfHealingEngine(config)

        assert engine.config == config
        assert isinstance(engine.component_registry, ComponentRegistry)
        assert isinstance(engine.watchdog_service, WatchdogService)
        assert isinstance(engine.healing_orchestrator, HealingOrchestrator)
        assert isinstance(engine.emergency_procedures, EmergencyProcedures)
        assert isinstance(engine.monitoring_dashboard, MonitoringDashboard)

    def test_component_registration(self):
        """Test component registration with engine."""
        engine = SelfHealingEngine({})

        mock_component = Mock()
        engine.register_component(
            "test_comp",
            ComponentType.BOT_ENGINE,
            mock_component,
            critical=True
        )

        # Should be registered in component registry
        info = engine.component_registry.get_component("test_comp")
        assert info is not None
        assert info.component_type == ComponentType.BOT_ENGINE
        assert info.critical is True

        # Should be registered with watchdog
        assert "test_comp" in engine.watchdog_service.heartbeat_protocols

    @pytest.mark.asyncio
    async def test_heartbeat_sending(self):
        """Test heartbeat sending."""
        engine = SelfHealingEngine({})

        # Register component
        mock_component = Mock()
        engine.register_component("test_comp", ComponentType.BOT_ENGINE, mock_component)

        # Send heartbeat
        await engine.send_heartbeat(
            component_id="test_comp",
            status=ComponentStatus.HEALTHY,
            latency_ms=45.0,
            error_count=0
        )

        # Should have updated component status
        info = engine.component_registry.get_component("test_comp")
        assert info.consecutive_failures == 0

    def test_engine_stats(self):
        """Test engine statistics."""
        engine = SelfHealingEngine({})

        # Mock some activity
        engine.total_failures_handled = 10
        engine.total_recoveries_successful = 8

        stats = engine.get_engine_stats()

        assert stats['total_failures_handled'] == 10
        assert stats['total_recoveries_successful'] == 8
        assert 'registry_stats' in stats
        assert 'healing_stats' in stats
        assert 'watchdog_stats' in stats

    def test_component_status_retrieval(self):
        """Test component status retrieval."""
        engine = SelfHealingEngine({})

        # Register component
        mock_component = Mock()
        engine.register_component("test_comp", ComponentType.BOT_ENGINE, mock_component)

        status = engine.get_component_status("test_comp")

        assert status is not None
        assert status['component_id'] == "test_comp"
        assert status['component_type'] == "BOT_ENGINE"

        # Non-existent component
        status = engine.get_component_status("nonexistent")
        assert status is None


class TestIntegrationWithN1V1:
    """Test integration with existing N1V1 components."""

    @pytest.mark.asyncio
    async def test_bot_engine_integration(self, mock_bot_engine):
        """Test integration with BotEngine."""
        engine = SelfHealingEngine({})

        # Register bot engine
        engine.register_component(
            "bot_engine_main",
            ComponentType.BOT_ENGINE,
            mock_bot_engine,
            critical=True
        )

        # Send heartbeat
        await engine.send_heartbeat(
            component_id="bot_engine_main",
            status=ComponentStatus.HEALTHY,
            latency_ms=50.0,
            error_count=0,
            custom_metrics={
                'active_orders': 5,
                'open_positions': 2,
                'total_pnl': 150.0
            }
        )

        # Should have processed heartbeat
        assert engine.watchdog_service.heartbeats_received == 1

    @pytest.mark.asyncio
    async def test_diagnostics_integration(self):
        """Test integration with diagnostics system."""
        from core.diagnostics import get_diagnostics_manager

        engine = SelfHealingEngine({})
        diagnostics = get_diagnostics_manager()

        # Register health check
        async def check_self_healing_engine():
            stats = engine.get_engine_stats()
            total_components = stats['registry_stats']['total_components']

            status = HealthStatus.HEALTHY if total_components >= 0 else HealthStatus.DEGRADED

            return {
                'component': 'self_healing_engine',
                'status': status,
                'latency_ms': 10.0,
                'message': f'Engine healthy: {total_components} components monitored',
                'details': {'monitored_components': total_components}
            }

        diagnostics.register_health_check('self_healing_engine', check_self_healing_engine)

        # Run health check
        state = await diagnostics.run_health_check()

        # Should have self-healing engine health data
        assert 'self_healing_engine' in state.component_statuses

    @pytest.mark.asyncio
    async def test_event_bus_integration(self):
        """Test integration with event bus."""
        from core.signal_router.events import get_default_enhanced_event_bus

        engine = SelfHealingEngine({})
        event_bus = get_default_enhanced_event_bus()

        # Register component
        mock_component = Mock()
        engine.register_component("test_comp", ComponentType.BOT_ENGINE, mock_component)

        # Send heartbeat that should trigger failure detection
        await engine.send_heartbeat(
            component_id="test_comp",
            status=ComponentStatus.CRITICAL,
            latency_ms=1000.0,
            error_count=10
        )

        # Should have published failure event
        # (This would be verified by checking event bus publications)


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_heartbeat_processing_performance(self, performance_timer):
        """Test heartbeat processing performance."""
        engine = SelfHealingEngine({})

        # Register component
        mock_component = Mock()
        engine.register_component("test_comp", ComponentType.BOT_ENGINE, mock_component)

        # Measure heartbeat processing time
        performance_timer.start()

        for i in range(100):
            await engine.send_heartbeat(
                component_id="test_comp",
                status=ComponentStatus.HEALTHY,
                latency_ms=50.0 + i * 0.1,
                error_count=0
            )

        performance_timer.stop()

        duration_ms = performance_timer.duration_ms()

        # Should process 100 heartbeats quickly
        assert duration_ms < 1000  # Less than 1 second total

    @pytest.mark.asyncio
    async def test_concurrent_heartbeat_processing(self):
        """Test concurrent heartbeat processing."""
        engine = SelfHealingEngine({})

        # Register multiple components
        for i in range(10):
            mock_component = Mock()
            engine.register_component(f"comp_{i}", ComponentType.STRATEGY, mock_component)

        # Send heartbeats concurrently
        tasks = []
        for i in range(10):
            task = engine.send_heartbeat(
                component_id=f"comp_{i}",
                status=ComponentStatus.HEALTHY,
                latency_ms=50.0,
                error_count=0
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        # All heartbeats should be processed
        assert engine.watchdog_service.heartbeats_received == 10

    @pytest.mark.asyncio
    async def test_memory_usage(self, memory_monitor):
        """Test memory usage during operation."""
        engine = SelfHealingEngine({})

        memory_monitor.start()

        # Register components and send heartbeats
        for i in range(50):
            mock_component = Mock()
            engine.register_component(f"comp_{i}", ComponentType.STRATEGY, mock_component)

            await engine.send_heartbeat(
                component_id=f"comp_{i}",
                status=ComponentStatus.HEALTHY,
                latency_ms=50.0,
                error_count=0
            )

        memory_delta = memory_monitor.get_memory_delta()

        # Memory usage should be reasonable
        assert memory_delta < 50  # Less than 50MB increase


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_component_registration(self):
        """Test handling of invalid component registration."""
        engine = SelfHealingEngine({})

        # Try to register with invalid component type
        with pytest.raises(Exception):
            engine.register_component("test_comp", "invalid_type", Mock())

    @pytest.mark.asyncio
    async def test_heartbeat_for_nonexistent_component(self):
        """Test heartbeat for non-existent component."""
        engine = SelfHealingEngine({})

        # Should handle gracefully
        await engine.send_heartbeat(
            component_id="nonexistent",
            status=ComponentStatus.HEALTHY
        )

        # No errors should occur

    @pytest.mark.asyncio
    async def test_recovery_failure_handling(self):
        """Test handling of recovery failures."""
        engine = SelfHealingEngine({})

        # Register component
        mock_component = Mock()
        engine.register_component("test_comp", ComponentType.BOT_ENGINE, mock_component)

        # Mock recovery orchestrator to fail
        engine.healing_orchestrator.initiate_healing = AsyncMock(return_value=None)

        # Send failing heartbeat
        await engine.send_heartbeat(
            component_id="test_comp",
            status=ComponentStatus.CRITICAL
        )

        # Should handle failure gracefully
        # (Exact behavior depends on implementation)

    @pytest.mark.asyncio
    async def test_emergency_mode_under_load(self):
        """Test emergency mode activation under load."""
        engine = SelfHealingEngine({})

        # Register many critical components
        for i in range(20):
            mock_component = Mock()
            engine.register_component(
                f"critical_{i}",
                ComponentType.BOT_ENGINE,
                mock_component,
                critical=True
            )

        # Fail many critical components
        for i in range(12):  # More than 50%
            await engine.send_heartbeat(
                component_id=f"critical_{i}",
                status=ComponentStatus.CRITICAL
            )

        # Should potentially trigger emergency mode
        # (Depends on emergency threshold configuration)


class TestReliability:
    """Test reliability and stability."""

    @pytest.mark.asyncio
    async def test_long_running_stability(self):
        """Test long-running stability."""
        engine = SelfHealingEngine({})

        # Register component
        mock_component = Mock()
        engine.register_component("test_comp", ComponentType.BOT_ENGINE, mock_component)

        # Send heartbeats for extended period
        start_time = time.time()

        for i in range(100):
            await engine.send_heartbeat(
                component_id="test_comp",
                status=ComponentStatus.HEALTHY,
                latency_ms=50.0,
                error_count=0
            )
            await asyncio.sleep(0.01)  # Small delay

        duration = time.time() - start_time

        # Should complete within reasonable time
        assert duration < 5  # Less than 5 seconds for 100 heartbeats

        # Should have processed all heartbeats
        assert engine.watchdog_service.heartbeats_received == 100

    @pytest.mark.asyncio
    async def test_failure_recovery_cycle(self):
        """Test failure detection and recovery cycle."""
        engine = SelfHealingEngine({})

        # Register component
        mock_component = Mock()
        engine.register_component("test_comp", ComponentType.BOT_ENGINE, mock_component)

        # Normal operation
        await engine.send_heartbeat("test_comp", ComponentStatus.HEALTHY, 50.0, 0)

        # Failure
        await engine.send_heartbeat("test_comp", ComponentStatus.CRITICAL, 1000.0, 5)

        # Recovery
        await engine.send_heartbeat("test_comp", ComponentStatus.HEALTHY, 45.0, 0)

        # Should have tracked the cycle
        info = engine.component_registry.get_component("test_comp")
        assert info.consecutive_failures == 0  # Reset after recovery
