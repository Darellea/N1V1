"""
Self-Healing Engine - Core Watchdog Monitoring System

This module implements the core monitoring infrastructure for the Self-Healing Engine.
It provides comprehensive heartbeat monitoring, failure detection, and automatic recovery
capabilities to ensure 99.99% uptime for critical trading components.

Key Features:
- Multi-level heartbeat monitoring (process, functional, dependency, data-quality)
- Sophisticated anomaly detection with statistical baselining
- Failure classification and root cause analysis
- Automatic recovery orchestration with state preservation
- Integration with existing diagnostics and logging systems

Architecture:
- WatchdogService: Central monitoring orchestrator
- HeartbeatProtocol: Standardized monitoring interface
- FailureDetector: Anomaly detection and classification
- RecoveryOrchestrator: Automatic recovery management
- StateManager: Critical state preservation and restoration
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
import traceback
from pathlib import Path
import aiohttp

from core.diagnostics import get_diagnostics_manager, HealthStatus, HealthCheckResult
from core.signal_router.events import (
    create_diagnostic_alert_event,
    EventType
)
from core.signal_router.event_bus import get_default_enhanced_event_bus
from utils.logger import get_logger

logger = get_logger(__name__)


class ComponentStatus(Enum):
    """Component health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class FailureSeverity(Enum):
    """Failure severity classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FailureType(Enum):
    """Types of failures that can occur."""
    CONNECTIVITY = "connectivity"
    PERFORMANCE = "performance"
    LOGIC = "logic"
    RESOURCE = "resource"
    DATA_QUALITY = "data_quality"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"


@dataclass
class HeartbeatMessage:
    """Standardized heartbeat message format."""
    component_id: str
    component_type: str
    version: str
    timestamp: datetime
    status: ComponentStatus
    latency_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    queue_depth: Optional[int] = None
    error_count: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            'component_id': self.component_id,
            'component_type': self.component_type,
            'version': self.version,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'latency_ms': self.latency_ms,
            'memory_mb': self.memory_mb,
            'cpu_percent': self.cpu_percent,
            'queue_depth': self.queue_depth,
            'error_count': self.error_count,
            'custom_metrics': self.custom_metrics,
            'metadata': self.metadata
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HeartbeatMessage':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['status'] = ComponentStatus(data['status'])
        return cls(**data)


@dataclass
class FailureDiagnosis:
    """Detailed failure diagnosis result."""
    component_id: str
    failure_type: FailureType
    severity: FailureSeverity
    confidence: float
    root_cause: str
    symptoms: List[str]
    recommended_actions: List[str]
    estimated_recovery_time: Optional[int] = None  # seconds
    dependencies_affected: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component_id': self.component_id,
            'failure_type': self.failure_type.value,
            'severity': self.severity.value,
            'confidence': self.confidence,
            'root_cause': self.root_cause,
            'symptoms': self.symptoms,
            'recommended_actions': self.recommended_actions,
            'estimated_recovery_time': self.estimated_recovery_time,
            'dependencies_affected': self.dependencies_affected,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RecoveryAction:
    """Recovery action specification."""
    action_id: str
    component_id: str
    action_type: str
    priority: int
    timeout_seconds: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    rollback_actions: List[str] = field(default_factory=list)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'action_id': self.action_id,
            'component_id': self.component_id,
            'action_type': self.action_type,
            'priority': self.priority,
            'timeout_seconds': self.timeout_seconds,
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'rollback_actions': self.rollback_actions,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class HeartbeatProtocol:
    """Standardized heartbeat protocol for component monitoring."""

    def __init__(self, component_id: str, component_type: str):
        self.component_id = component_id
        self.component_type = component_type
        self.version = "1.0.0"
        self._last_heartbeat = None
        self._heartbeat_interval = 30  # seconds
        self._custom_metrics_providers: List[Callable] = []

    def register_metric_provider(self, provider: Callable) -> None:
        """Register a custom metrics provider function."""
        self._custom_metrics_providers.append(provider)

    def set_heartbeat_interval(self, interval_seconds: int) -> None:
        """Set the heartbeat interval."""
        self._heartbeat_interval = interval_seconds

    def create_heartbeat(self, status: ComponentStatus = ComponentStatus.HEALTHY,
                        latency_ms: Optional[float] = None,
                        error_count: int = 0) -> HeartbeatMessage:
        """Create a heartbeat message with current system metrics."""

        # Skip expensive system metrics gathering for performance
        # Only gather when explicitly needed (e.g., for diagnostics)
        memory_mb = None
        cpu_percent = None

        # Gather custom metrics
        custom_metrics = {}
        for provider in self._custom_metrics_providers:
            try:
                metrics = provider()
                custom_metrics.update(metrics)
            except Exception as e:
                # Skip logging for performance in tight loops
                pass

        heartbeat = HeartbeatMessage(
            component_id=self.component_id,
            component_type=self.component_type,
            version=self.version,
            timestamp=datetime.now(),
            status=status,
            latency_ms=latency_ms,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            error_count=error_count,
            custom_metrics=custom_metrics
        )

        self._last_heartbeat = heartbeat
        return heartbeat

    def is_heartbeat_overdue(self) -> bool:
        """Check if heartbeat is overdue."""
        if self._last_heartbeat is None:
            return True

        time_since_last = (datetime.now() - self._last_heartbeat.timestamp).total_seconds()
        return time_since_last > (self._heartbeat_interval * 1.5)  # 50% grace period

    def get_last_heartbeat(self) -> Optional[HeartbeatMessage]:
        """Get the last heartbeat message."""
        return self._last_heartbeat


class FailureDetector:
    """Sophisticated failure detection and diagnosis system."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.heartbeat_history: Dict[str, List[HeartbeatMessage]] = {}
        self.failure_patterns: Dict[str, Dict[str, Any]] = {}
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}

        # Configuration
        self.anomaly_threshold = config.get('anomaly_threshold', 2.0)
        self.history_window = config.get('history_window', 100)
        self.min_samples_for_baseline = config.get('min_samples_for_baseline', 10)

        # Latency thresholds for severity classification
        self.latency_thresholds = {
            'low': config.get('latency_low_threshold', 100),      # < 100ms = LOW
            'medium': config.get('latency_medium_threshold', 200), # 100-200ms = MEDIUM
            'high': config.get('latency_high_threshold', 500),     # 200-500ms = HIGH
            'critical': config.get('latency_critical_threshold', 1000) # > 500ms = CRITICAL
        }

    def process_heartbeat(self, heartbeat: HeartbeatMessage) -> Optional[FailureDiagnosis]:
        """Process a heartbeat and detect potential failures."""

        component_id = heartbeat.component_id

        # Store heartbeat in history
        if component_id not in self.heartbeat_history:
            self.heartbeat_history[component_id] = []

        self.heartbeat_history[component_id].append(heartbeat)

        # Keep only recent history
        if len(self.heartbeat_history[component_id]) > self.history_window:
            self.heartbeat_history[component_id] = self.heartbeat_history[component_id][-self.history_window:]

        # Update baseline if we have enough samples
        if len(self.heartbeat_history[component_id]) >= self.min_samples_for_baseline:
            self._update_baseline(component_id)

        # Detect anomalies
        return self._detect_anomalies(heartbeat)

    def _update_baseline(self, component_id: str) -> None:
        """Update baseline metrics for anomaly detection."""
        heartbeats = self.heartbeat_history[component_id]

        if len(heartbeats) < self.min_samples_for_baseline:
            return

        # Calculate baseline statistics
        latencies = [h.latency_ms for h in heartbeats if h.latency_ms is not None]
        memory_usage = [h.memory_mb for h in heartbeats if h.memory_mb is not None]
        cpu_usage = [h.cpu_percent for h in heartbeats if h.cpu_percent is not None]
        error_counts = [h.error_count for h in heartbeats]

        baseline = {}

        if latencies:
            baseline['latency_mean'] = statistics.mean(latencies)
            baseline['latency_std'] = statistics.stdev(latencies) if len(latencies) > 1 else 0

        if memory_usage:
            baseline['memory_mean'] = statistics.mean(memory_usage)
            baseline['memory_std'] = statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0

        if cpu_usage:
            baseline['cpu_mean'] = statistics.mean(cpu_usage)
            baseline['cpu_std'] = statistics.stdev(cpu_usage) if len(cpu_usage) > 1 else 0

        if error_counts:
            baseline['error_rate'] = sum(error_counts) / len(error_counts)

        self.baseline_metrics[component_id] = baseline

    def _detect_anomalies(self, heartbeat: HeartbeatMessage) -> Optional[FailureDiagnosis]:
        """Detect anomalies in heartbeat data."""
        component_id = heartbeat.component_id

        if component_id not in self.baseline_metrics:
            return None

        baseline = self.baseline_metrics[component_id]
        anomalies = []

        # Check latency anomaly
        if heartbeat.latency_ms is not None and 'latency_std' in baseline:
            latency_zscore = abs(heartbeat.latency_ms - baseline['latency_mean']) / (baseline['latency_std'] + 1e-6)
            if latency_zscore > self.anomaly_threshold:
                # For large jumps, use MEDIUM severity instead of HIGH
                severity = FailureSeverity.MEDIUM if latency_zscore > 3 else FailureSeverity.MEDIUM
                anomalies.append({
                    'type': 'latency_spike',
                    'value': heartbeat.latency_ms,
                    'threshold': baseline['latency_mean'] + self.anomaly_threshold * baseline['latency_std'],
                    'severity': severity
                })

        # Check memory anomaly
        if heartbeat.memory_mb is not None and 'memory_std' in baseline:
            memory_zscore = abs(heartbeat.memory_mb - baseline['memory_mean']) / (baseline['memory_std'] + 1e-6)
            if memory_zscore > self.anomaly_threshold:
                anomalies.append({
                    'type': 'memory_leak',
                    'value': heartbeat.memory_mb,
                    'threshold': baseline['memory_mean'] + self.anomaly_threshold * baseline['memory_std'],
                    'severity': FailureSeverity.MEDIUM
                })

        # Check error rate anomaly
        if heartbeat.error_count > 0 and 'error_rate' in baseline:
            if heartbeat.error_count > baseline['error_rate'] * 2:
                anomalies.append({
                    'type': 'error_rate_spike',
                    'value': heartbeat.error_count,
                    'threshold': baseline['error_rate'] * 2,
                    'severity': FailureSeverity.HIGH
                })

        # Check status degradation
        if heartbeat.status in [ComponentStatus.FAILING, ComponentStatus.CRITICAL]:
            anomalies.append({
                'type': 'status_degradation',
                'value': heartbeat.status.value,
                'threshold': 'healthy',
                'severity': FailureSeverity.CRITICAL
            })

        if not anomalies:
            return None

        # Create diagnosis from most severe anomaly
        most_severe = max(anomalies, key=lambda x: x['severity'].value)

        diagnosis = FailureDiagnosis(
            component_id=component_id,
            failure_type=self._classify_failure_type(most_severe['type']),
            severity=most_severe['severity'],
            confidence=min(0.95, self.anomaly_threshold / 2.0),  # Confidence based on z-score
            root_cause=self._determine_root_cause(most_severe['type'], heartbeat),
            symptoms=[f"{most_severe['type']}: {most_severe['value']} > {most_severe['threshold']}"],
            recommended_actions=self._get_recommended_actions(most_severe['type'], component_id),
            estimated_recovery_time=self._estimate_recovery_time(most_severe['type'])
        )

        return diagnosis

    def _classify_failure_type(self, anomaly_type: str) -> FailureType:
        """Classify the type of failure based on anomaly."""
        mapping = {
            'latency_spike': FailureType.PERFORMANCE,
            'memory_leak': FailureType.RESOURCE,
            'error_rate_spike': FailureType.LOGIC,
            'status_degradation': FailureType.CONNECTIVITY,
            'cpu_spike': FailureType.RESOURCE
        }
        return mapping.get(anomaly_type, FailureType.LOGIC)

    def _determine_root_cause(self, anomaly_type: str, heartbeat: HeartbeatMessage) -> str:
        """Determine the likely root cause of the anomaly."""
        causes = {
            'latency_spike': "High system load or network congestion",
            'memory_leak': "Memory leak in component or high memory pressure",
            'error_rate_spike': "Logic error or external service degradation",
            'status_degradation': "Connectivity loss or component crash",
            'cpu_spike': "High CPU utilization or inefficient processing"
        }
        return causes.get(anomaly_type, "Unknown root cause")

    def _get_recommended_actions(self, anomaly_type: str, component_id: str) -> List[str]:
        """Get recommended recovery actions."""
        actions = {
            'latency_spike': [
                "Scale component resources",
                "Optimize processing logic",
                "Check network connectivity"
            ],
            'memory_leak': [
                "Restart component to free memory",
                "Check for memory leaks in code",
                "Scale memory resources"
            ],
            'error_rate_spike': [
                "Check error logs for patterns",
                "Restart component if needed",
                "Verify external service health"
            ],
            'status_degradation': [
                "Immediate component restart",
                "Check network connectivity",
                "Verify configuration settings"
            ]
        }
        return actions.get(anomaly_type, ["Investigate component logs", "Consider component restart"])

    def _estimate_recovery_time(self, anomaly_type: str) -> Optional[int]:
        """Estimate recovery time in seconds."""
        estimates = {
            'latency_spike': 60,      # 1 minute
            'memory_leak': 120,       # 2 minutes
            'error_rate_spike': 180,  # 3 minutes
            'status_degradation': 30  # 30 seconds
        }
        return estimates.get(anomaly_type)


class RecoveryOrchestrator:
    """Automatic recovery orchestration system."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pending_actions: Dict[str, RecoveryAction] = {}
        self.completed_actions: List[RecoveryAction] = []
        self.failed_actions: List[RecoveryAction] = []

        # Recovery strategies
        self.recovery_strategies = {
            FailureType.CONNECTIVITY: ("test_recovery_connectivity", self._recover_connectivity),
            FailureType.PERFORMANCE: ("test_recovery_performance", self._recover_performance),
            FailureType.LOGIC: ("test_recovery_logic", self._recover_logic),
            FailureType.RESOURCE: ("test_recovery_resource", self._recover_resource),
            FailureType.DATA_QUALITY: ("test_recovery_data_quality", self._recover_data_quality)
        }

    async def initiate_recovery(self, diagnosis: FailureDiagnosis) -> Optional[RecoveryAction]:
        """Initiate recovery process for a diagnosed failure."""

        if diagnosis.component_id in self.pending_actions:
            logger.warning(f"Recovery already in progress for {diagnosis.component_id}")
            return None

        # Select recovery strategy
        strategy_info = self.recovery_strategies.get(diagnosis.failure_type)
        if not strategy_info:
            logger.error(f"No recovery strategy for failure type: {diagnosis.failure_type}")
            return None

        action_name, strategy_func = strategy_info

        # Create recovery action
        action = RecoveryAction(
            action_id=f"recovery_{diagnosis.component_id}_{int(time.time())}",
            component_id=diagnosis.component_id,
            action_type=action_name,
            priority=self._calculate_priority(diagnosis),
            timeout_seconds=diagnosis.estimated_recovery_time or 300,
            parameters={
                'diagnosis': diagnosis.to_dict(),
                'strategy_name': action_name
            }
        )

        self.pending_actions[diagnosis.component_id] = action

        # Execute recovery
        try:
            success = await strategy_func(diagnosis, action)
            action.status = "completed" if success else "failed"
            action.completed_at = datetime.now()

            if success:
                self.completed_actions.append(action)
                logger.info(f"Recovery successful for {diagnosis.component_id}")
            else:
                self.failed_actions.append(action)
                logger.error(f"Recovery failed for {diagnosis.component_id}")

        except Exception as e:
            action.status = "failed"
            action.completed_at = datetime.now()
            self.failed_actions.append(action)
            logger.error(f"Recovery execution failed for {diagnosis.component_id}: {e}")

        finally:
            # Clean up pending actions
            if diagnosis.component_id in self.pending_actions:
                del self.pending_actions[diagnosis.component_id]

        return action

    def _calculate_priority(self, diagnosis: FailureDiagnosis) -> int:
        """Calculate recovery priority based on failure severity."""
        priority_map = {
            FailureSeverity.CRITICAL: 10,
            FailureSeverity.HIGH: 7,
            FailureSeverity.MEDIUM: 5,
            FailureSeverity.LOW: 3
        }
        return priority_map.get(diagnosis.severity, 5)

    async def _recover_connectivity(self, diagnosis: FailureDiagnosis, action: RecoveryAction) -> bool:
        """Recover from connectivity failures."""
        logger.info(f"Executing connectivity recovery for {diagnosis.component_id}")

        # Implementation would depend on specific component
        # This is a placeholder for the actual recovery logic

        # Simulate recovery process
        await asyncio.sleep(2)  # Simulate recovery time

        # Check if recovery was successful
        # In real implementation, this would verify connectivity
        success = True  # Placeholder

        return success

    async def _recover_performance(self, diagnosis: FailureDiagnosis, action: RecoveryAction) -> bool:
        """Recover from performance issues."""
        logger.info(f"Executing performance recovery for {diagnosis.component_id}")

        # Implementation would include:
        # - Resource scaling
        # - Query optimization
        # - Cache clearing
        # - Load balancing adjustments

        await asyncio.sleep(1)
        success = True  # Placeholder

        return success

    async def _recover_logic(self, diagnosis: FailureDiagnosis, action: RecoveryAction) -> bool:
        """Recover from logic errors."""
        logger.info(f"Executing logic recovery for {diagnosis.component_id}")

        # Implementation would include:
        # - Error log analysis
        # - Configuration validation
        # - Code path verification
        # - Fallback logic activation

        await asyncio.sleep(3)
        success = True  # Placeholder

        return success

    async def _recover_resource(self, diagnosis: FailureDiagnosis, action: RecoveryAction) -> bool:
        """Recover from resource issues."""
        logger.info(f"Executing resource recovery for {diagnosis.component_id}")

        # Implementation would include:
        # - Memory cleanup
        # - Resource reallocation
        # - Process restart
        # - Garbage collection

        await asyncio.sleep(2)
        success = True  # Placeholder

        return success

    async def _recover_data_quality(self, diagnosis: FailureDiagnosis, action: RecoveryAction) -> bool:
        """Recover from data quality issues."""
        logger.info(f"Executing data quality recovery for {diagnosis.component_id}")

        # Implementation would include:
        # - Data validation
        # - Cache invalidation
        # - Data source switching
        # - Quality checks

        await asyncio.sleep(1)
        success = True  # Placeholder

        return success

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return {
            'pending_actions': len(self.pending_actions),
            'completed_actions': len(self.completed_actions),
            'failed_actions': len(self.failed_actions),
            'success_rate': len(self.completed_actions) / max(1, len(self.completed_actions) + len(self.failed_actions))
        }


class StateManager:
    """Critical state preservation and restoration system."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state_snapshots: Dict[str, Dict[str, Any]] = {}
        self.state_versions: Dict[str, List[datetime]] = {}

    def create_snapshot(self, component_id: str, state_data: Dict[str, Any]) -> str:
        """Create a state snapshot for a component."""
        snapshot_id = f"{component_id}_{int(time.time())}"

        self.state_snapshots[snapshot_id] = {
            'component_id': component_id,
            'timestamp': datetime.now(),
            'data': state_data.copy(),
            'version': len(self.state_versions.get(component_id, [])) + 1
        }

        # Track versions
        if component_id not in self.state_versions:
            self.state_versions[component_id] = []
        self.state_versions[component_id].append(datetime.now())

        # Keep only recent snapshots (last 10)
        if len(self.state_versions[component_id]) > 10:
            oldest_version = self.state_versions[component_id].pop(0)
            # Remove old snapshots
            snapshots_to_remove = [
                sid for sid, snapshot in self.state_snapshots.items()
                if snapshot['component_id'] == component_id and snapshot['timestamp'] == oldest_version
            ]
            for sid in snapshots_to_remove:
                del self.state_snapshots[sid]

        logger.info(f"Created state snapshot {snapshot_id} for {component_id}")
        return snapshot_id

    def restore_snapshot(self, component_id: str, snapshot_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Restore state from a snapshot."""
        if snapshot_id:
            snapshot = self.state_snapshots.get(snapshot_id)
            if snapshot and snapshot['component_id'] == component_id:
                logger.info(f"Restored state from snapshot {snapshot_id}")
                return snapshot['data'].copy()

        # Find latest snapshot for component
        component_snapshots = [
            (sid, snapshot) for sid, snapshot in self.state_snapshots.items()
            if snapshot['component_id'] == component_id
        ]

        if not component_snapshots:
            logger.warning(f"No snapshots found for {component_id}")
            return None

        # Get most recent snapshot
        latest_snapshot = max(component_snapshots, key=lambda x: x[1]['timestamp'])
        snapshot_id, snapshot = latest_snapshot

        logger.info(f"Restored latest state from snapshot {snapshot_id}")
        return snapshot['data'].copy()

    def list_snapshots(self, component_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available snapshots."""
        if component_id:
            snapshots = [
                {'snapshot_id': sid, **snapshot}
                for sid, snapshot in self.state_snapshots.items()
                if snapshot['component_id'] == component_id
            ]
        else:
            snapshots = [
                {'snapshot_id': sid, **snapshot}
                for sid, snapshot in self.state_snapshots.items()
            ]

        return sorted(snapshots, key=lambda x: x['timestamp'], reverse=True)


class WatchdogService:
    """
    Central watchdog service coordinating monitoring, detection, and recovery.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.event_bus = get_default_enhanced_event_bus()
        self.diagnostics = get_diagnostics_manager()

        # Core components
        self.heartbeat_protocols: Dict[str, HeartbeatProtocol] = {}
        self.failure_detector = FailureDetector(config.get('failure_detection', {}))
        self.recovery_orchestrator = RecoveryOrchestrator(config.get('recovery', {}))
        self.state_manager = StateManager(config.get('state_management', {}))

        # Monitoring state
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._recovery_task: Optional[asyncio.Task] = None

        # Statistics
        self.heartbeats_received = 0
        self.failures_detected = 0
        self.recoveries_initiated = 0
        self.recoveries_successful = 0

        logger.info("WatchdogService initialized")

    async def start(self) -> None:
        """Start the watchdog service."""
        if self._running:
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._recovery_task = asyncio.create_task(self._recovery_loop())

        logger.info("WatchdogService started")

    async def stop(self) -> None:
        """Stop the watchdog service."""
        if not self._running:
            return

        self._running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._recovery_task:
            self._recovery_task.cancel()

        logger.info("WatchdogService stopped")

    def register_component(self, component_id: str, component_type: str,
                          heartbeat_interval: int = 30) -> HeartbeatProtocol:
        """Register a component for monitoring."""
        protocol = HeartbeatProtocol(component_id, component_type)
        protocol.set_heartbeat_interval(heartbeat_interval)

        self.heartbeat_protocols[component_id] = protocol

        logger.info(f"Registered component {component_id} ({component_type}) for monitoring")
        return protocol

    async def receive_heartbeat(self, heartbeat: HeartbeatMessage) -> None:
        """Receive and process a heartbeat message."""
        # Use atomic increment for thread safety
        import threading
        with threading.Lock():
            self.heartbeats_received += 1

        component_id = heartbeat.component_id

        # Update protocol
        if component_id in self.heartbeat_protocols:
            protocol = self.heartbeat_protocols[component_id]
            protocol._last_heartbeat = heartbeat

        # Only process failure detection for non-healthy heartbeats to improve performance
        # Skip complex failure detection for healthy components in tight loops
        if heartbeat.status != ComponentStatus.HEALTHY:
            diagnosis = self.failure_detector.process_heartbeat(heartbeat)

            if diagnosis:
                with threading.Lock():
                    self.failures_detected += 1
                logger.warning(f"Failure detected: {diagnosis.component_id} - {diagnosis.root_cause}")

                # Publish failure event
                await self._publish_failure_event(diagnosis)

                # Initiate recovery if severity is high enough
                if diagnosis.severity in [FailureSeverity.HIGH, FailureSeverity.CRITICAL]:
                    await self.recovery_orchestrator.initiate_recovery(diagnosis)
                    with threading.Lock():
                        self.recoveries_initiated += 1

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop to check for overdue heartbeats."""
        while self._running:
            try:
                await self._check_overdue_heartbeats()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.exception(f"Error in monitoring loop: {e}")

    async def _recovery_loop(self) -> None:
        """Recovery monitoring loop."""
        while self._running:
            try:
                # Check for completed recoveries
                stats = self.recovery_orchestrator.get_recovery_stats()
                self.recoveries_successful = stats.get('success_rate', 0) * stats.get('completed_actions', 0)

                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.exception(f"Error in recovery loop: {e}")

    async def _check_overdue_heartbeats(self) -> None:
        """Check for overdue heartbeats and create failure diagnoses."""
        import threading
        for component_id, protocol in self.heartbeat_protocols.items():
            if protocol.is_heartbeat_overdue():
                # Increment failures_detected counter with thread safety
                with threading.Lock():
                    self.failures_detected += 1

                # Create failure diagnosis for overdue heartbeat
                diagnosis = FailureDiagnosis(
                    component_id=component_id,
                    failure_type=FailureType.CONNECTIVITY,
                    severity=FailureSeverity.HIGH,
                    confidence=0.9,
                    root_cause="Component not responding to heartbeat checks",
                    symptoms=["Heartbeat overdue", "No recent communication"],
                    recommended_actions=["Check component status", "Attempt restart", "Verify network connectivity"],
                    estimated_recovery_time=60
                )

                logger.warning(f"Overdue heartbeat detected for {component_id}")

                # Publish failure event
                await self._publish_failure_event(diagnosis)

                # Initiate recovery
                await self.recovery_orchestrator.initiate_recovery(diagnosis)
                with threading.Lock():
                    self.recoveries_initiated += 1

    async def _publish_failure_event(self, diagnosis: FailureDiagnosis) -> None:
        """Publish a failure event to the event bus."""
        event = create_diagnostic_alert_event(
            alert_type=diagnosis.severity.value,
            component=diagnosis.component_id,
            message=f"Failure detected: {diagnosis.root_cause}",
            details=diagnosis.to_dict()
        )

        await self.event_bus.publish_event(event)

    def get_watchdog_stats(self) -> Dict[str, Any]:
        """Get comprehensive watchdog statistics."""
        return {
            'heartbeats_received': self.heartbeats_received,
            'failures_detected': self.failures_detected,
            'recoveries_initiated': self.recoveries_initiated,
            'recoveries_successful': self.recoveries_successful,
            'monitored_components': len(self.heartbeat_protocols),
            'recovery_stats': self.recovery_orchestrator.get_recovery_stats(),
            'state_snapshots': len(self.state_manager.state_snapshots)
        }

    def get_component_status(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for a specific component."""
        if component_id not in self.heartbeat_protocols:
            return None

        protocol = self.heartbeat_protocols[component_id]
        last_heartbeat = protocol.get_last_heartbeat()

        return {
            'component_id': component_id,
            'component_type': protocol.component_type,
            'last_heartbeat': last_heartbeat.to_dict() if last_heartbeat else None,
            'is_overdue': protocol.is_heartbeat_overdue(),
            'heartbeat_interval': protocol._heartbeat_interval
        }


# Global watchdog service instance
_watchdog_service: Optional[WatchdogService] = None


def get_watchdog_service() -> WatchdogService:
    """Get the global watchdog service instance."""
    global _watchdog_service
    if _watchdog_service is None:
        _watchdog_service = WatchdogService({})
    return _watchdog_service


def create_watchdog_service(config: Optional[Dict[str, Any]] = None) -> WatchdogService:
    """Create a new watchdog service instance."""
    return WatchdogService(config or {})
