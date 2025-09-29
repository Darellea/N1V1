"""
Alert Rules Manager for N1V1 Trading Framework

This module provides functionality for creating, managing, and evaluating
alert rules based on monitoring metrics. It supports rule creation,
evaluation, deduplication, and notification delivery.

The current implementation is minimal and can be extended for production use.
"""

from typing import Dict, Any, List, Optional
import time
import asyncio
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AlertRule:
    """Represents an alert rule configuration."""
    name: str
    query: str
    duration: str
    severity: str
    description: str = ""
    channels: List[str] = None
    enabled: bool = True
    manager: Optional['AlertRulesManager'] = None

    def __post_init__(self):
        if self.channels is None:
            self.channels = ["log"]

    async def evaluate(self, metrics_data: Dict[str, Any]) -> bool:
        """Evaluate the alert rule against metrics data."""
        # Simple query evaluation for basic > comparison
        # In production, this would be a proper query parser
        logger.debug(f"Evaluating rule {self.name} with query: {self.query}")

        try:
            # Parse simple queries like "metric_name > value"
            parts = self.query.split()
            if len(parts) == 3:
                metric_name, operator, threshold_str = parts
                if metric_name in metrics_data:
                    value = metrics_data[metric_name]

                    # Try numeric comparison first
                    try:
                        left = float(value)
                        right = float(threshold_str)
                        operators = {
                            ">": left > right,
                            ">=": left >= right,
                            "<": left < right,
                            "<=": left <= right,
                            "==": left == right,
                            "!=": left != right
                        }
                        return operators.get(operator, False)
                    except (ValueError, TypeError):
                        # Fall back to string comparison for == and !=
                        if operator == "==":
                            return str(value) == threshold_str
                        elif operator == "!=":
                            return str(value) != threshold_str
                        else:
                            return False
                else:
                    return False
            else:
                return False
        except (KeyError, TypeError):
            return False

    def _is_deduplicated(self) -> bool:
        """Check if an alert should be deduplicated."""
        if not self.manager:
            return False
        current_time = time.time()
        recent_alerts = [
            alert for alert in self.manager.alert_history
            if alert['rule_name'] == self.name and alert['severity'] == self.severity and
            current_time - alert['timestamp'] < self.manager.deduplication_window
        ]
        return len(recent_alerts) > 0

    def _record_alert(self, alert: Dict[str, Any]):
        """Record an alert in the history."""
        if self.manager:
            self.manager.alert_history.append(alert)
            # Maintain max history size
            if len(self.manager.alert_history) > self.manager.max_history_size:
                self.manager.alert_history = self.manager.alert_history[-self.manager.max_history_size:]




class AlertRulesManager:
    """
    Manager for alert rules in the monitoring system.

    This class handles the creation, storage, evaluation, and notification
    of alert rules. It provides deduplication to prevent alert spam and
    supports multiple notification channels.

    The current implementation is minimal and can be extended for production use.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules: Dict[str, AlertRule] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.deduplication_window = config.get('deduplication_window', 300)  # 5 minutes
        self.max_history_size = config.get('max_history_size', 1000)

        logger.info("AlertRulesManager initialized")

    async def create_rule(self, rule_config: Dict[str, Any]) -> AlertRule:
        """Create a new alert rule."""
        rule = AlertRule(
            name=rule_config['name'],
            query=rule_config['query'],
            duration=rule_config.get('duration', '5m'),
            severity=rule_config.get('severity', 'warning'),
            description=rule_config.get('description', ''),
            channels=rule_config.get('channels', ['log']),
            manager=self
        )

        self.rules[rule.name] = rule
        logger.info(f"Created alert rule: {rule.name}")
        return rule

    async def evaluate_rules(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate all enabled rules against the provided metrics data."""
        alerts = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            try:
                triggered = await rule.evaluate(metrics_data)
                if triggered:
                    # Check deduplication per (rule_name, severity)
                    current_time = time.time()
                    recent_alerts = [
                        alert for alert in self.alert_history
                        if alert['rule_name'] == rule.name and alert['severity'] == rule.severity and
                        current_time - alert['timestamp'] < self.deduplication_window
                    ]
                    if not recent_alerts:
                        # Trigger alert
                        alert = {
                            'rule_name': rule.name,
                            'severity': rule.severity,
                            'description': rule.description,
                            'timestamp': current_time,
                            'metrics_data': metrics_data
                        }
                        self._record_alert(alert)
                        await self._deliver_notifications(alert, rule.channels)
                        alerts.append(alert)
            except Exception as e:
                logger.exception(f"Error evaluating rule {rule.name}: {e}")

        return alerts



    def _record_alert(self, alert: Dict[str, Any]):
        """Record an alert in the history."""
        self.alert_history.append(alert)

        # Maintain max history size
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]

    async def _deliver_notifications(self, alert: Dict[str, Any], channels: List[str]):
        """Deliver alert notifications to specified channels."""
        for channel in channels:
            try:
                if channel == "log":
                    logger.warning(f"ALERT: {alert['rule_name']} - {alert['description']}")
                elif channel == "discord":
                    await self._send_discord_notification(alert)
                elif channel == "email":
                    await self._send_email_notification(alert)
                else:
                    logger.warning(f"Unknown notification channel: {channel}")
            except Exception as e:
                logger.exception(f"Error delivering notification via {channel}: {e}")

    async def _send_discord_notification(self, alert: Dict[str, Any]):
        """Send alert notification to Discord (stub implementation)."""
        # Stub - in production, integrate with Discord API
        logger.info(f"Discord notification: {alert['rule_name']}")

    async def _send_email_notification(self, alert: Dict[str, Any]):
        """Send alert notification via email (stub implementation)."""
        # Stub - in production, integrate with email service
        logger.info(f"Email notification: {alert['rule_name']}")

    def get_rules(self) -> Dict[str, AlertRule]:
        """Get all alert rules."""
        return self.rules.copy()

    def get_alert_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        if limit:
            return self.alert_history[-limit:]
        return self.alert_history.copy()

    def enable_rule(self, rule_name: str):
        """Enable an alert rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            logger.info(f"Enabled alert rule: {rule_name}")

    def disable_rule(self, rule_name: str):
        """Disable an alert rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            logger.info(f"Disabled alert rule: {rule_name}")

    def delete_rule(self, rule_name: str):
        """Delete an alert rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Deleted alert rule: {rule_name}")
