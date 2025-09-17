"""
Test suite for Alert Rules Manager.

Tests alert rule creation, evaluation, deduplication, and notification delivery.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from core.alert_rules_manager import AlertRulesManager, AlertRule


class TestAlertRule:
    """Test AlertRule dataclass and methods."""

    def test_alert_rule_creation(self):
        """Test basic AlertRule creation."""
        rule = AlertRule(
            name="test_rule",
            query="cpu_usage > 80",
            duration="5m",
            severity="warning",
            description="High CPU usage",
            channels=["log", "discord"]
        )

        assert rule.name == "test_rule"
        assert rule.query == "cpu_usage > 80"
        assert rule.duration == "5m"
        assert rule.severity == "warning"
        assert rule.description == "High CPU usage"
        assert rule.channels == ["log", "discord"]
        assert rule.enabled is True
        assert rule.manager is None

    def test_alert_rule_default_channels(self):
        """Test AlertRule with default channels."""
        rule = AlertRule(
            name="test_rule",
            query="memory_usage > 90",
            duration="10m",
            severity="critical"
        )

        assert rule.channels == ["log"]

    @pytest.mark.asyncio
    async def test_rule_evaluation_greater_than(self):
        """Test rule evaluation with > operator."""
        rule = AlertRule(
            name="cpu_high",
            query="cpu_usage > 80",
            duration="5m",
            severity="warning"
        )

        # Test triggering
        metrics = {"cpu_usage": 85}
        triggered = await rule.evaluate(metrics)
        assert triggered is True

        # Test not triggering
        metrics = {"cpu_usage": 75}
        triggered = await rule.evaluate(metrics)
        assert triggered is False

    @pytest.mark.asyncio
    async def test_rule_evaluation_less_than(self):
        """Test rule evaluation with < operator."""
        rule = AlertRule(
            name="cpu_low",
            query="cpu_usage < 20",
            duration="5m",
            severity="info"
        )

        # Test triggering
        metrics = {"cpu_usage": 15}
        triggered = await rule.evaluate(metrics)
        assert triggered is True

        # Test not triggering
        metrics = {"cpu_usage": 25}
        triggered = await rule.evaluate(metrics)
        assert triggered is False

    @pytest.mark.asyncio
    async def test_rule_evaluation_greater_equal(self):
        """Test rule evaluation with >= operator."""
        rule = AlertRule(
            name="memory_high",
            query="memory_usage >= 90",
            duration="5m",
            severity="warning"
        )

        # Test triggering (equal)
        metrics = {"memory_usage": 90}
        triggered = await rule.evaluate(metrics)
        assert triggered is True

        # Test triggering (greater)
        metrics = {"memory_usage": 95}
        triggered = await rule.evaluate(metrics)
        assert triggered is True

        # Test not triggering
        metrics = {"memory_usage": 85}
        triggered = await rule.evaluate(metrics)
        assert triggered is False

    @pytest.mark.asyncio
    async def test_rule_evaluation_less_equal(self):
        """Test rule evaluation with <= operator."""
        rule = AlertRule(
            name="disk_low",
            query="disk_usage <= 10",
            duration="5m",
            severity="info"
        )

        # Test triggering (equal)
        metrics = {"disk_usage": 10}
        triggered = await rule.evaluate(metrics)
        assert triggered is True

        # Test triggering (less)
        metrics = {"disk_usage": 5}
        triggered = await rule.evaluate(metrics)
        assert triggered is True

        # Test not triggering
        metrics = {"disk_usage": 15}
        triggered = await rule.evaluate(metrics)
        assert triggered is False

    @pytest.mark.asyncio
    async def test_rule_evaluation_equal(self):
        """Test rule evaluation with == operator."""
        rule = AlertRule(
            name="status_check",
            query="service_status == 1",
            duration="5m",
            severity="critical"
        )

        # Test triggering
        metrics = {"service_status": 1}
        triggered = await rule.evaluate(metrics)
        assert triggered is True

        # Test not triggering
        metrics = {"service_status": 0}
        triggered = await rule.evaluate(metrics)
        assert triggered is False

    @pytest.mark.asyncio
    async def test_rule_evaluation_invalid_operator(self):
        """Test rule evaluation with invalid operator."""
        rule = AlertRule(
            name="invalid_rule",
            query="metric != 5",
            duration="5m",
            severity="warning"
        )

        metrics = {"metric": 5}
        triggered = await rule.evaluate(metrics)
        assert triggered is False

    @pytest.mark.asyncio
    async def test_rule_evaluation_missing_metric(self):
        """Test rule evaluation when metric is missing."""
        rule = AlertRule(
            name="missing_metric",
            query="unknown_metric > 50",
            duration="5m",
            severity="warning"
        )

        metrics = {"other_metric": 60}
        triggered = await rule.evaluate(metrics)
        assert triggered is False

    @pytest.mark.asyncio
    async def test_rule_evaluation_invalid_query_format(self):
        """Test rule evaluation with invalid query format."""
        rule = AlertRule(
            name="invalid_query",
            query="invalid query format",
            duration="5m",
            severity="warning"
        )

        metrics = {"metric": 50}
        triggered = await rule.evaluate(metrics)
        assert triggered is False

    @pytest.mark.asyncio
    async def test_rule_evaluation_with_manager(self):
        """Test rule evaluation with manager."""
        manager = AlertRulesManager({})
        rule = AlertRule(
            name="test_with_manager",
            query="test_metric > 10",
            duration="5m",
            severity="warning",
            manager=manager
        )

        metrics = {"test_metric": 15}

        # Rule evaluation should work with manager
        triggered = await rule.evaluate(metrics)
        assert triggered is True

    def test_rule_deduplication_without_manager(self):
        """Test deduplication check without manager."""
        rule = AlertRule(
            name="no_manager",
            query="metric > 5",
            duration="5m",
            severity="warning"
        )

        # Should not be deduplicated without manager
        assert rule._is_deduplicated() is False

    def test_rule_deduplication_with_manager(self):
        """Test deduplication check with manager."""
        manager = AlertRulesManager({'deduplication_window': 60})
        rule = AlertRule(
            name="test_dedup",
            query="metric > 5",
            duration="5m",
            severity="warning",
            manager=manager
        )

        # Initially not deduplicated
        assert rule._is_deduplicated() is False

        # Add recent alert
        manager.alert_history.append({
            'rule_name': 'test_dedup',
            'timestamp': time.time()
        })

        # Should be deduplicated now
        assert rule._is_deduplicated() is True

    def test_rule_deduplication_expired(self):
        """Test deduplication with expired alert."""
        manager = AlertRulesManager({'deduplication_window': 1})  # 1 second window
        rule = AlertRule(
            name="test_expired",
            query="metric > 5",
            duration="5m",
            severity="warning",
            manager=manager
        )

        # Add old alert
        manager.alert_history.append({
            'rule_name': 'test_expired',
            'timestamp': time.time() - 10  # 10 seconds ago
        })

        # Should not be deduplicated (expired)
        assert rule._is_deduplicated() is False


class TestAlertRulesManager:
    """Test AlertRulesManager class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'deduplication_window': 300,
            'max_history_size': 100
        }
        self.manager = AlertRulesManager(self.config)

    def test_manager_initialization(self):
        """Test AlertRulesManager initialization."""
        assert len(self.manager.rules) == 0
        assert len(self.manager.alert_history) == 0
        assert self.manager.deduplication_window == 300
        assert self.manager.max_history_size == 100

    @pytest.mark.asyncio
    async def test_create_rule(self):
        """Test creating a new alert rule."""
        rule_config = {
            'name': 'test_rule',
            'query': 'cpu_usage > 80',
            'duration': '5m',
            'severity': 'warning',
            'description': 'High CPU usage detected',
            'channels': ['log', 'discord']
        }

        rule = await self.manager.create_rule(rule_config)

        assert rule.name == 'test_rule'
        assert rule.query == 'cpu_usage > 80'
        assert rule.severity == 'warning'
        assert rule.channels == ['log', 'discord']
        assert self.manager.rules['test_rule'] == rule

    @pytest.mark.asyncio
    async def test_create_rule_defaults(self):
        """Test creating a rule with default values."""
        rule_config = {
            'name': 'minimal_rule',
            'query': 'memory > 90',
            'severity': 'critical'
        }

        rule = await self.manager.create_rule(rule_config)

        assert rule.duration == '5m'
        assert rule.channels == ['log']
        assert rule.description == ''

    @pytest.mark.asyncio
    async def test_evaluate_rules(self):
        """Test evaluating all rules."""
        # Create rules
        rule1_config = {
            'name': 'cpu_high',
            'query': 'cpu_usage > 80',
            'severity': 'warning'
        }
        rule2_config = {
            'name': 'memory_high',
            'query': 'memory_usage > 90',
            'severity': 'critical'
        }

        await self.manager.create_rule(rule1_config)
        await self.manager.create_rule(rule2_config)

        # Test metrics that trigger both rules
        metrics = {'cpu_usage': 85, 'memory_usage': 95}

        with patch.object(self.manager, '_deliver_notifications', new_callable=AsyncMock):
            alerts = await self.manager.evaluate_rules(metrics)

        assert len(alerts) == 2
        assert alerts[0]['rule_name'] == 'cpu_high'
        assert alerts[1]['rule_name'] == 'memory_high'

    @pytest.mark.asyncio
    async def test_evaluate_rules_no_triggers(self):
        """Test evaluating rules with no triggers."""
        rule_config = {
            'name': 'cpu_high',
            'query': 'cpu_usage > 80',
            'severity': 'warning'
        }

        await self.manager.create_rule(rule_config)

        # Metrics that don't trigger
        metrics = {'cpu_usage': 70}

        alerts = await self.manager.evaluate_rules(metrics)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_evaluate_rules_disabled(self):
        """Test evaluating disabled rules."""
        rule_config = {
            'name': 'disabled_rule',
            'query': 'cpu_usage > 80',
            'severity': 'warning'
        }

        rule = await self.manager.create_rule(rule_config)
        rule.enabled = False

        metrics = {'cpu_usage': 85}
        alerts = await self.manager.evaluate_rules(metrics)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_notification_delivery_log(self):
        """Test log notification delivery."""
        alert = {
            'rule_name': 'test_alert',
            'severity': 'warning',
            'description': 'Test alert',
            'timestamp': time.time(),
            'metrics_data': {'cpu': 90}
        }

        with patch('core.alert_rules_manager.logger') as mock_logger:
            await self.manager._deliver_notifications(alert, ['log'])
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_notification_delivery_discord(self):
        """Test Discord notification delivery."""
        alert = {
            'rule_name': 'test_alert',
            'severity': 'critical',
            'description': 'Critical alert',
            'timestamp': time.time(),
            'metrics_data': {'cpu': 95}
        }

        with patch.object(self.manager, '_send_discord_notification', new_callable=AsyncMock) as mock_discord:
            await self.manager._deliver_notifications(alert, ['discord'])
            mock_discord.assert_called_once_with(alert)

    @pytest.mark.asyncio
    async def test_notification_delivery_email(self):
        """Test email notification delivery."""
        alert = {
            'rule_name': 'test_alert',
            'severity': 'warning',
            'description': 'Email alert',
            'timestamp': time.time(),
            'metrics_data': {'memory': 90}
        }

        with patch.object(self.manager, '_send_email_notification', new_callable=AsyncMock) as mock_email:
            await self.manager._deliver_notifications(alert, ['email'])
            mock_email.assert_called_once_with(alert)

    @pytest.mark.asyncio
    async def test_notification_delivery_multiple_channels(self):
        """Test notification delivery to multiple channels."""
        alert = {
            'rule_name': 'multi_channel',
            'severity': 'critical',
            'description': 'Multi-channel alert',
            'timestamp': time.time(),
            'metrics_data': {'cpu': 95}
        }

        with patch.object(self.manager, '_send_discord_notification', new_callable=AsyncMock) as mock_discord, \
             patch.object(self.manager, '_send_email_notification', new_callable=AsyncMock) as mock_email, \
             patch('core.alert_rules_manager.logger') as mock_logger:

            await self.manager._deliver_notifications(alert, ['log', 'discord', 'email'])

            mock_logger.warning.assert_called_once()
            mock_discord.assert_called_once_with(alert)
            mock_email.assert_called_once_with(alert)

    @pytest.mark.asyncio
    async def test_notification_delivery_unknown_channel(self):
        """Test notification delivery with unknown channel."""
        alert = {
            'rule_name': 'unknown_channel',
            'severity': 'warning',
            'description': 'Unknown channel alert',
            'timestamp': time.time(),
            'metrics_data': {'cpu': 85}
        }

        with patch('core.alert_rules_manager.logger') as mock_logger:
            await self.manager._deliver_notifications(alert, ['unknown'])

            # Should log warning about unknown channel
            mock_logger.warning.assert_called_with("Unknown notification channel: unknown")

    def test_get_rules(self):
        """Test getting all rules."""
        rule_config = {
            'name': 'test_rule',
            'query': 'cpu > 80',
            'severity': 'warning'
        }

        # Initially empty
        rules = self.manager.get_rules()
        assert len(rules) == 0

        # Add rule
        self.manager.rules['test_rule'] = AlertRule(**rule_config, duration='5m')

        rules = self.manager.get_rules()
        assert len(rules) == 1
        assert 'test_rule' in rules

    def test_get_alert_history(self):
        """Test getting alert history."""
        # Initially empty
        history = self.manager.get_alert_history()
        assert len(history) == 0

        # Add alerts
        self.manager.alert_history = [
            {'rule_name': 'rule1', 'timestamp': 1000},
            {'rule_name': 'rule2', 'timestamp': 1001},
            {'rule_name': 'rule3', 'timestamp': 1002}
        ]

        # Get all history
        history = self.manager.get_alert_history()
        assert len(history) == 3

        # Get limited history
        history = self.manager.get_alert_history(limit=2)
        assert len(history) == 2
        assert history[0]['rule_name'] == 'rule2'
        assert history[1]['rule_name'] == 'rule3'

    def test_enable_rule(self):
        """Test enabling a rule."""
        rule = AlertRule(
            name='test_rule',
            query='cpu > 80',
            duration='5m',
            severity='warning',
            enabled=False
        )
        self.manager.rules['test_rule'] = rule

        self.manager.enable_rule('test_rule')
        assert rule.enabled is True

    def test_enable_nonexistent_rule(self):
        """Test enabling a nonexistent rule."""
        # Should not raise error
        self.manager.enable_rule('nonexistent')
        assert 'nonexistent' not in self.manager.rules

    def test_disable_rule(self):
        """Test disabling a rule."""
        rule = AlertRule(
            name='test_rule',
            query='cpu > 80',
            duration='5m',
            severity='warning',
            enabled=True
        )
        self.manager.rules['test_rule'] = rule

        self.manager.disable_rule('test_rule')
        assert rule.enabled is False

    def test_disable_nonexistent_rule(self):
        """Test disabling a nonexistent rule."""
        # Should not raise error
        self.manager.disable_rule('nonexistent')
        assert 'nonexistent' not in self.manager.rules

    def test_delete_rule(self):
        """Test deleting a rule."""
        rule = AlertRule(
            name='test_rule',
            query='cpu > 80',
            duration='5m',
            severity='warning'
        )
        self.manager.rules['test_rule'] = rule

        self.manager.delete_rule('test_rule')
        assert 'test_rule' not in self.manager.rules

    def test_delete_nonexistent_rule(self):
        """Test deleting a nonexistent rule."""
        # Should not raise error
        self.manager.delete_rule('nonexistent')
        assert len(self.manager.rules) == 0

    def test_deduplication_logic(self):
        """Test deduplication logic in manager."""
        # Initially not deduplicated
        assert self.manager._is_deduplicated('test_rule') is False

        # Add recent alert
        self.manager.alert_history.append({
            'rule_name': 'test_rule',
            'timestamp': time.time()
        })

        # Should be deduplicated
        assert self.manager._is_deduplicated('test_rule') is True

        # Different rule should not be deduplicated
        assert self.manager._is_deduplicated('other_rule') is False

    def test_deduplication_expired(self):
        """Test deduplication with expired alerts."""
        # Short deduplication window
        manager = AlertRulesManager({'deduplication_window': 1})

        # Add old alert
        manager.alert_history.append({
            'rule_name': 'test_rule',
            'timestamp': time.time() - 10
        })

        # Should not be deduplicated (expired)
        assert manager._is_deduplicated('test_rule') is False

    def test_alert_history_size_limit(self):
        """Test alert history size limiting."""
        manager = AlertRulesManager({'max_history_size': 2})

        # Add more alerts than limit using _record_alert
        for i in range(5):
            alert = {
                'rule_name': f'rule_{i}',
                'timestamp': time.time()
            }
            manager._record_alert(alert)

        # Should only keep the most recent alerts
        assert len(manager.alert_history) == 2
        assert manager.alert_history[0]['rule_name'] == 'rule_3'
        assert manager.alert_history[1]['rule_name'] == 'rule_4'

    @pytest.mark.asyncio
    async def test_rule_evaluation_error_handling(self):
        """Test error handling in rule evaluation."""
        rule_config = {
            'name': 'error_rule',
            'query': 'cpu_usage > 80',
            'severity': 'warning'
        }

        await self.manager.create_rule(rule_config)

        # Mock rule.evaluate to raise exception
        with patch.object(self.manager.rules['error_rule'], 'evaluate', side_effect=Exception("Test error")):
            metrics = {'cpu_usage': 85}

            # Should not raise exception, should log error
            with patch('core.alert_rules_manager.logger') as mock_logger:
                alerts = await self.manager.evaluate_rules(metrics)
                mock_logger.exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_notification_delivery_error_handling(self):
        """Test error handling in notification delivery."""
        alert = {
            'rule_name': 'error_alert',
            'severity': 'critical',
            'description': 'Error test',
            'timestamp': time.time(),
            'metrics_data': {'cpu': 95}
        }

        # Mock discord notification to raise exception
        with patch.object(self.manager, '_send_discord_notification', side_effect=Exception("Discord error")), \
             patch('core.alert_rules_manager.logger') as mock_logger:

            await self.manager._deliver_notifications(alert, ['discord'])
            mock_logger.exception.assert_called_once()


class TestIntegrationScenarios:
    """Test integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_alert_workflow(self):
        """Test complete alert workflow from creation to notification."""
        manager = AlertRulesManager({})

        # Create rule
        rule_config = {
            'name': 'cpu_monitor',
            'query': 'cpu_usage > 80',
            'severity': 'warning',
            'channels': ['log']
        }

        rule = await manager.create_rule(rule_config)

        # Evaluate with triggering metrics
        metrics = {'cpu_usage': 85}

        with patch('core.alert_rules_manager.logger') as mock_logger:
            alerts = await manager.evaluate_rules(metrics)

            assert len(alerts) == 1
            assert alerts[0]['rule_name'] == 'cpu_monitor'

            # Verify notification was sent
            mock_logger.warning.assert_called_once()

            # Verify alert was recorded
            assert len(manager.alert_history) == 1
            assert manager.alert_history[0]['rule_name'] == 'cpu_monitor'

    @pytest.mark.asyncio
    async def test_deduplication_workflow(self):
        """Test deduplication in complete workflow."""
        manager = AlertRulesManager({'deduplication_window': 60})

        # Create rule
        rule_config = {
            'name': 'memory_monitor',
            'query': 'memory_usage > 90',
            'severity': 'critical'
        }

        await manager.create_rule(rule_config)

        # First evaluation - should trigger
        metrics = {'memory_usage': 95}

        with patch('core.alert_rules_manager.logger'):
            alerts1 = await manager.evaluate_rules(metrics)
            assert len(alerts1) == 1

        # Second evaluation within window - should be deduplicated
        with patch('core.alert_rules_manager.logger'):
            alerts2 = await manager.evaluate_rules(metrics)
            assert len(alerts2) == 0

        # Verify only one alert in history
        assert len(manager.alert_history) == 1


class TestAlertRuleEdgeCases:
    """Test AlertRule edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_rule_evaluation_with_float_values(self):
        """Test rule evaluation with float values."""
        rule = AlertRule(
            name="float_test",
            query="metric > 50.5",
            duration="5m",
            severity="warning"
        )

        # Test with float values
        metrics = {"metric": 60.7}
        triggered = await rule.evaluate(metrics)
        assert triggered is True

        metrics = {"metric": 40.2}
        triggered = await rule.evaluate(metrics)
        assert triggered is False

    @pytest.mark.asyncio
    async def test_rule_evaluation_with_string_values(self):
        """Test rule evaluation with string threshold (should handle gracefully)."""
        rule = AlertRule(
            name="string_threshold",
            query="metric > abc",
            duration="5m",
            severity="warning"
        )

        metrics = {"metric": 50}
        triggered = await rule.evaluate(metrics)
        assert triggered is False  # Should not crash, just return False

    @pytest.mark.asyncio
    async def test_rule_evaluation_with_non_numeric_metric(self):
        """Test rule evaluation when metric value is non-numeric."""
        rule = AlertRule(
            name="non_numeric",
            query="metric > 50",
            duration="5m",
            severity="warning"
        )

        metrics = {"metric": "not_a_number"}
        triggered = await rule.evaluate(metrics)
        assert triggered is False  # Should handle gracefully

    @pytest.mark.asyncio
    async def test_rule_evaluation_complex_query(self):
        """Test rule evaluation with complex query (unsupported)."""
        rule = AlertRule(
            name="complex_query",
            query="metric1 > 50 AND metric2 < 30",
            duration="5m",
            severity="warning"
        )

        metrics = {"metric1": 60, "metric2": 20}
        triggered = await rule.evaluate(metrics)
        assert triggered is False  # Complex queries not supported

    @pytest.mark.asyncio
    async def test_rule_evaluation_empty_query(self):
        """Test rule evaluation with empty query."""
        rule = AlertRule(
            name="empty_query",
            query="",
            duration="5m",
            severity="warning"
        )

        metrics = {"metric": 50}
        triggered = await rule.evaluate(metrics)
        assert triggered is False

    @pytest.mark.asyncio
    async def test_rule_evaluation_whitespace_query(self):
        """Test rule evaluation with whitespace-only query."""
        rule = AlertRule(
            name="whitespace_query",
            query="   ",
            duration="5m",
            severity="warning"
        )

        metrics = {"metric": 50}
        triggered = await rule.evaluate(metrics)
        assert triggered is False

    def test_rule_deduplication_with_expired_alerts(self):
        """Test deduplication with various time scenarios."""
        manager = AlertRulesManager({'deduplication_window': 10})  # 10 seconds
        rule = AlertRule(
            name="time_test",
            query="metric > 5",
            duration="5m",
            severity="warning",
            manager=manager
        )

        current_time = time.time()

        # Add alert from 15 seconds ago (expired)
        manager.alert_history.append({
            'rule_name': 'time_test',
            'timestamp': current_time - 15
        })

        # Should not be deduplicated
        assert rule._is_deduplicated() is False

        # Add recent alert (within window)
        manager.alert_history.append({
            'rule_name': 'time_test',
            'timestamp': current_time - 5
        })

        # Should be deduplicated
        assert rule._is_deduplicated() is True

    def test_rule_record_alert_without_manager(self):
        """Test _record_alert when no manager is set."""
        rule = AlertRule(
            name="no_manager",
            query="metric > 5",
            duration="5m",
            severity="warning"
        )

        alert = {'rule_name': 'test', 'timestamp': time.time()}
        # Should not crash
        rule._record_alert(alert)


class TestAlertRulesManagerEdgeCases:
    """Test AlertRulesManager edge cases."""

    def test_manager_initialization_with_empty_config(self):
        """Test manager initialization with empty config."""
        manager = AlertRulesManager({})
        assert manager.deduplication_window == 300  # default
        assert manager.max_history_size == 1000  # default

    def test_manager_initialization_with_custom_config(self):
        """Test manager initialization with custom config."""
        config = {
            'deduplication_window': 600,
            'max_history_size': 500
        }
        manager = AlertRulesManager(config)
        assert manager.deduplication_window == 600
        assert manager.max_history_size == 500

    @pytest.mark.asyncio
    async def test_create_rule_with_minimal_config(self):
        """Test creating rule with minimal configuration."""
        manager = AlertRulesManager({})
        rule_config = {
            'name': 'minimal',
            'query': 'test > 1'
        }

        rule = await manager.create_rule(rule_config)
        assert rule.name == 'minimal'
        assert rule.query == 'test > 1'
        assert rule.duration == '5m'  # default
        assert rule.severity == 'warning'  # default
        assert rule.channels == ['log']  # default

    @pytest.mark.asyncio
    async def test_create_rule_duplicate_name(self):
        """Test creating rule with duplicate name."""
        manager = AlertRulesManager({})
        rule_config1 = {
            'name': 'duplicate',
            'query': 'test > 1'
        }
        rule_config2 = {
            'name': 'duplicate',
            'query': 'test > 2'
        }

        await manager.create_rule(rule_config1)
        # Should overwrite existing rule
        rule2 = await manager.create_rule(rule_config2)

        assert len(manager.rules) == 1
        assert manager.rules['duplicate'].query == 'test > 2'

    @pytest.mark.asyncio
    async def test_evaluate_rules_with_empty_rules(self):
        """Test evaluating rules when no rules exist."""
        manager = AlertRulesManager({})
        metrics = {'cpu': 80}

        alerts = await manager.evaluate_rules(metrics)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_evaluate_rules_with_disabled_rules(self):
        """Test evaluating rules with all rules disabled."""
        manager = AlertRulesManager({})

        rule_config = {
            'name': 'disabled',
            'query': 'cpu > 80',
            'severity': 'warning'
        }

        rule = await manager.create_rule(rule_config)
        rule.enabled = False

        metrics = {'cpu': 90}
        alerts = await manager.evaluate_rules(metrics)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_evaluate_rules_with_mixed_enabled_disabled(self):
        """Test evaluating rules with mix of enabled and disabled rules."""
        manager = AlertRulesManager({})

        # Create enabled rule
        enabled_config = {
            'name': 'enabled_rule',
            'query': 'cpu > 80',
            'severity': 'warning'
        }

        # Create disabled rule
        disabled_config = {
            'name': 'disabled_rule',
            'query': 'memory > 90',
            'severity': 'critical'
        }

        enabled_rule = await manager.create_rule(enabled_config)
        disabled_rule = await manager.create_rule(disabled_config)
        disabled_rule.enabled = False

        metrics = {'cpu': 85, 'memory': 95}

        with patch.object(manager, '_deliver_notifications', new_callable=AsyncMock):
            alerts = await manager.evaluate_rules(metrics)

        assert len(alerts) == 1
        assert alerts[0]['rule_name'] == 'enabled_rule'

    def test_is_deduplicated_with_empty_history(self):
        """Test deduplication check with empty history."""
        manager = AlertRulesManager({})
        assert manager._is_deduplicated('any_rule') is False

    def test_is_deduplicated_with_different_rule_names(self):
        """Test deduplication check with different rule names."""
        manager = AlertRulesManager({})

        manager.alert_history.append({
            'rule_name': 'rule1',
            'timestamp': time.time()
        })

        assert manager._is_deduplicated('rule2') is False
        assert manager._is_deduplicated('rule1') is True

    def test_record_alert_with_history_limit(self):
        """Test recording alerts with history size limit."""
        manager = AlertRulesManager({'max_history_size': 2})

        # Add alerts beyond limit
        for i in range(5):
            alert = {
                'rule_name': f'rule_{i}',
                'timestamp': time.time()
            }
            manager._record_alert(alert)

        assert len(manager.alert_history) == 2
        # Should keep most recent
        assert manager.alert_history[0]['rule_name'] == 'rule_3'
        assert manager.alert_history[1]['rule_name'] == 'rule_4'

    @pytest.mark.asyncio
    async def test_deliver_notifications_with_empty_channels(self):
        """Test notification delivery with empty channels."""
        manager = AlertRulesManager({})
        alert = {'rule_name': 'test', 'severity': 'warning'}

        # Should not crash
        await manager._deliver_notifications(alert, [])

    @pytest.mark.asyncio
    async def test_deliver_notifications_with_invalid_channel(self):
        """Test notification delivery with invalid channel type."""
        manager = AlertRulesManager({})
        alert = {'rule_name': 'test', 'severity': 'warning'}

        with patch('core.alert_rules_manager.logger') as mock_logger:
            await manager._deliver_notifications(alert, [None])

            mock_logger.warning.assert_called_with("Unknown notification channel: None")

    @pytest.mark.asyncio
    async def test_send_discord_notification_implementation(self):
        """Test Discord notification implementation."""
        manager = AlertRulesManager({})
        alert = {'rule_name': 'discord_test', 'severity': 'critical'}

        with patch('core.alert_rules_manager.logger') as mock_logger:
            await manager._send_discord_notification(alert)

            mock_logger.info.assert_called_with("Discord notification: discord_test")

    @pytest.mark.asyncio
    async def test_send_email_notification_implementation(self):
        """Test email notification implementation."""
        manager = AlertRulesManager({})
        alert = {'rule_name': 'email_test', 'severity': 'warning'}

        with patch('core.alert_rules_manager.logger') as mock_logger:
            await manager._send_email_notification(alert)

            mock_logger.info.assert_called_with("Email notification: email_test")

    def test_get_rules_returns_copy(self):
        """Test that get_rules returns a copy, not reference."""
        manager = AlertRulesManager({})

        rule = AlertRule(name='test', query='cpu > 80', duration='5m', severity='warning')
        manager.rules['test'] = rule

        rules = manager.get_rules()
        assert rules == manager.rules

        # Modify returned dict
        rules['new_rule'] = AlertRule(name='new', query='mem > 90', duration='5m', severity='warning')

        # Original should be unchanged
        assert 'new_rule' not in manager.rules

    def test_get_alert_history_returns_copy(self):
        """Test that get_alert_history returns a copy."""
        manager = AlertRulesManager({})

        manager.alert_history = [
            {'rule_name': 'rule1', 'timestamp': 1000},
            {'rule_name': 'rule2', 'timestamp': 1001}
        ]

        history = manager.get_alert_history()
        assert history == manager.alert_history

        # Modify returned list
        history.append({'rule_name': 'rule3', 'timestamp': 1002})

        # Original should be unchanged
        assert len(manager.alert_history) == 2

    def test_get_alert_history_with_limit_zero(self):
        """Test get_alert_history with limit of 0."""
        manager = AlertRulesManager({})

        manager.alert_history = [
            {'rule_name': 'rule1', 'timestamp': 1000}
        ]

        history = manager.get_alert_history(limit=0)
        assert history == []

    def test_get_alert_history_with_negative_limit(self):
        """Test get_alert_history with negative limit."""
        manager = AlertRulesManager({})

        manager.alert_history = [
            {'rule_name': 'rule1', 'timestamp': 1000}
        ]

        history = manager.get_alert_history(limit=-1)
        assert history == []

    def test_enable_rule_already_enabled(self):
        """Test enabling a rule that's already enabled."""
        manager = AlertRulesManager({})

        rule = AlertRule(name='test', query='cpu > 80', duration='5m', severity='warning', enabled=True)
        manager.rules['test'] = rule

        manager.enable_rule('test')
        assert rule.enabled is True

    def test_disable_rule_already_disabled(self):
        """Test disabling a rule that's already disabled."""
        manager = AlertRulesManager({})

        rule = AlertRule(name='test', query='cpu > 80', duration='5m', severity='warning', enabled=False)
        manager.rules['test'] = rule

        manager.disable_rule('test')
        assert rule.enabled is False

    def test_delete_rule_from_empty_manager(self):
        """Test deleting rule from empty manager."""
        manager = AlertRulesManager({})

        manager.delete_rule('nonexistent')
        assert len(manager.rules) == 0


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_config_with_invalid_deduplication_window(self):
        """Test config with invalid deduplication window."""
        config = {'deduplication_window': 'invalid'}
        manager = AlertRulesManager(config)
        # Should use default
        assert manager.deduplication_window == 300

    def test_config_with_negative_deduplication_window(self):
        """Test config with negative deduplication window."""
        config = {'deduplication_window': -100}
        manager = AlertRulesManager(config)
        assert manager.deduplication_window == -100  # No validation

    def test_config_with_zero_max_history_size(self):
        """Test config with zero max history size."""
        config = {'max_history_size': 0}
        manager = AlertRulesManager(config)
        assert manager.max_history_size == 0

    def test_config_with_negative_max_history_size(self):
        """Test config with negative max history size."""
        config = {'max_history_size': -50}
        manager = AlertRulesManager(config)
        assert manager.max_history_size == -50


class TestPerformanceScenarios:
    """Test performance-related scenarios."""

    def test_manager_with_many_rules(self):
        """Test manager performance with many rules."""
        manager = AlertRulesManager({})

        # Create many rules
        for i in range(100):
            rule = AlertRule(
                name=f'rule_{i}',
                query=f'metric_{i} > {i}',
                duration='5m',
                severity='warning'
            )
            manager.rules[rule.name] = rule

        assert len(manager.rules) == 100

    def test_manager_with_large_history(self):
        """Test manager with large alert history."""
        manager = AlertRulesManager({'max_history_size': 10000})

        # Add many alerts
        for i in range(200):
            alert = {
                'rule_name': f'rule_{i % 10}',
                'timestamp': time.time()
            }
            manager._record_alert(alert)

        assert len(manager.alert_history) == 10000  # Should be limited

    @pytest.mark.asyncio
    async def test_evaluate_rules_with_many_rules(self):
        """Test evaluating many rules simultaneously."""
        manager = AlertRulesManager({})

        # Create many rules
        for i in range(50):
            rule_config = {
                'name': f'rule_{i}',
                'query': f'metric_{i} > {i * 2}',
                'severity': 'warning'
            }
            await manager.create_rule(rule_config)

        # Create metrics that trigger some rules
        metrics = {}
        for i in range(50):
            metrics[f'metric_{i}'] = i * 3  # Will trigger rules where i*3 > i*2

        with patch.object(manager, '_deliver_notifications', new_callable=AsyncMock):
            alerts = await manager.evaluate_rules(metrics)

        # Should have alerts for rules where condition is met
        triggered_count = sum(1 for i in range(50) if i * 3 > i * 2)
        assert len(alerts) == triggered_count


class TestErrorRecovery:
    """Test error recovery scenarios."""

    @pytest.mark.asyncio
    async def test_rule_evaluation_with_exception_in_manager(self):
        """Test rule evaluation when manager has internal errors."""
        manager = AlertRulesManager({})

        rule_config = {
            'name': 'error_rule',
            'query': 'cpu > 80',
            'severity': 'warning'
        }

        await manager.create_rule(rule_config)

        # Simulate metrics that cause issues
        metrics = {'cpu': 'not_a_number'}

        # Should handle errors gracefully
        with patch('core.alert_rules_manager.logger') as mock_logger:
            alerts = await manager.evaluate_rules(metrics)

            # Should not crash, but may not trigger alerts
            assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_notification_delivery_with_partial_failures(self):
        """Test notification delivery with some channels failing."""
        manager = AlertRulesManager({})

        alert = {
            'rule_name': 'partial_failure',
            'severity': 'critical',
            'description': 'Test partial failure',
            'timestamp': time.time(),
            'metrics_data': {'cpu': 95}
        }

        # Mock discord to fail, email to succeed
        with patch.object(manager, '_send_discord_notification', side_effect=Exception("Discord failed")), \
             patch.object(manager, '_send_email_notification', new_callable=AsyncMock) as mock_email, \
             patch('core.alert_rules_manager.logger') as mock_logger:

            await manager._deliver_notifications(alert, ['log', 'discord', 'email'])

            # Should log the discord error
            mock_logger.exception.assert_called_once()

            # Should still try email
            mock_email.assert_called_once_with(alert)

            # Should still log to console
            mock_logger.warning.assert_called_once()


class TestConcurrency:
    """Test concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_rule_evaluation(self):
        """Test concurrent rule evaluation."""
        import asyncio

        manager = AlertRulesManager({})

        # Create multiple rules
        for i in range(10):
            rule_config = {
                'name': f'concurrent_rule_{i}',
                'query': f'metric_{i} > {i * 10}',
                'severity': 'warning'
            }
            await manager.create_rule(rule_config)

        # Create metrics
        metrics = {f'metric_{i}': i * 15 for i in range(10)}

        # Evaluate concurrently
        with patch.object(manager, '_deliver_notifications', new_callable=AsyncMock):
            alerts = await manager.evaluate_rules(metrics)

        # Should trigger alerts for all rules
        assert len(alerts) == 10

    @pytest.mark.asyncio
    async def test_concurrent_rule_creation(self):
        """Test concurrent rule creation."""
        import asyncio

        manager = AlertRulesManager({})

        async def create_rule_async(i):
            rule_config = {
                'name': f'async_rule_{i}',
                'query': f'metric_{i} > {i}',
                'severity': 'warning'
            }
            return await manager.create_rule(rule_config)

        # Create rules concurrently
        tasks = [create_rule_async(i) for i in range(20)]
        rules = await asyncio.gather(*tasks)

        assert len(manager.rules) == 20
        assert len(rules) == 20


class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_alert_history_integrity(self):
        """Test alert history maintains data integrity."""
        manager = AlertRulesManager({'max_history_size': 5})

        # Add alerts with various data
        test_alerts = [
            {'rule_name': 'rule1', 'timestamp': 1000, 'severity': 'warning', 'data': 'test1'},
            {'rule_name': 'rule2', 'timestamp': 1001, 'severity': 'critical', 'data': 'test2'},
            {'rule_name': 'rule3', 'timestamp': 1002, 'severity': 'info', 'data': 'test3'},
            {'rule_name': 'rule4', 'timestamp': 1003, 'severity': 'warning', 'data': 'test4'},
            {'rule_name': 'rule5', 'timestamp': 1004, 'severity': 'critical', 'data': 'test5'},
            {'rule_name': 'rule6', 'timestamp': 1005, 'severity': 'info', 'data': 'test6'},
        ]

        for alert in test_alerts:
            manager._record_alert(alert)

        # Should only keep last 5
        assert len(manager.alert_history) == 5

        # Verify data integrity
        for alert in manager.alert_history:
            assert 'rule_name' in alert
            assert 'timestamp' in alert
            assert 'severity' in alert
            assert 'data' in alert

    def test_rules_dictionary_integrity(self):
        """Test rules dictionary maintains integrity."""
        manager = AlertRulesManager({})

        # Add rules
        rule1 = AlertRule(name='rule1', query='cpu > 80', duration='5m', severity='warning')
        rule2 = AlertRule(name='rule2', query='mem > 90', duration='5m', severity='critical')

        manager.rules['rule1'] = rule1
        manager.rules['rule2'] = rule2

        # Verify integrity
        assert len(manager.rules) == 2
        assert manager.rules['rule1'].name == 'rule1'
        assert manager.rules['rule2'].name == 'rule2'

        # Test copy behavior
        rules_copy = manager.get_rules()
        rules_copy['rule3'] = AlertRule(name='rule3', query='disk > 95', duration='5m', severity='warning')

        # Original should be unchanged
        assert len(manager.rules) == 2
        assert 'rule3' not in manager.rules


if __name__ == "__main__":
    pytest.main([__file__])
