"""
Comprehensive tests for AlertRulesManager.

Tests alert rule creation, evaluation, deduplication, and notification delivery.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from core.alert_rules_manager import AlertRulesManager, AlertRule


class TestAlertRule:
    """Test AlertRule functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "name": "test_rule",
            "query": "cpu_usage > 80",
            "duration": "5m",
            "severity": "warning",
            "description": "High CPU usage detected"
        }
        self.manager = AlertRulesManager({})
        self.rule = AlertRule(
            name=self.config["name"],
            query=self.config["query"],
            duration=self.config["duration"],
            severity=self.config["severity"],
            description=self.config["description"],
            manager=self.manager
        )

    def test_alert_rule_initialization(self):
        """Test AlertRule initialization."""
        assert self.rule.name == "test_rule"
        assert self.rule.query == "cpu_usage > 80"
        assert self.rule.duration == "5m"
        assert self.rule.severity == "warning"
        assert self.rule.description == "High CPU usage detected"
        assert self.rule.channels == ["log"]
        assert self.rule.enabled is True

    @pytest.mark.asyncio
    async def test_alert_rule_evaluation_greater_than(self):
        """Test alert rule evaluation with > operator."""
        # Test trigger condition
        metrics_data = {"cpu_usage": 85}
        triggered = await self.rule.evaluate(metrics_data)
        assert triggered

        # Test non-trigger condition
        metrics_data = {"cpu_usage": 75}
        triggered = await self.rule.evaluate(metrics_data)
        assert not triggered

    @pytest.mark.asyncio
    async def test_alert_rule_evaluation_less_than(self):
        """Test alert rule evaluation with < operator."""
        rule = AlertRule(
            name="low_memory",
            query="memory_usage < 20",
            duration="5m",
            severity="critical",
            manager=self.manager
        )

        # Test trigger condition
        metrics_data = {"memory_usage": 15}
        triggered = await rule.evaluate(metrics_data)
        assert triggered

        # Test non-trigger condition
        metrics_data = {"memory_usage": 25}
        triggered = await rule.evaluate(metrics_data)
        assert not triggered

    @pytest.mark.asyncio
    async def test_alert_rule_evaluation_greater_equal(self):
        """Test alert rule evaluation with >= operator."""
        rule = AlertRule(
            name="high_load",
            query="load_average >= 5.0",
            duration="1m",
            severity="warning",
            manager=self.manager
        )

        # Test trigger conditions
        assert await rule.evaluate({"load_average": 5.0})
        assert await rule.evaluate({"load_average": 6.0})

        # Test non-trigger condition
        assert not await rule.evaluate({"load_average": 4.0})

    @pytest.mark.asyncio
    async def test_alert_rule_evaluation_less_equal(self):
        """Test alert rule evaluation with <= operator."""
        rule = AlertRule(
            name="low_disk",
            query="disk_usage <= 10",
            duration="1m",
            severity="critical",
            manager=self.manager
        )

        # Test trigger conditions
        assert await rule.evaluate({"disk_usage": 10})
        assert await rule.evaluate({"disk_usage": 5})

        # Test non-trigger condition
        assert not await rule.evaluate({"disk_usage": 15})

    @pytest.mark.asyncio
    async def test_alert_rule_evaluation_equal(self):
        """Test alert rule evaluation with == operator."""
        rule = AlertRule(
            name="status_check",
            query="service_status == 0",
            duration="1m",
            severity="critical",
            manager=self.manager
        )

        # Test trigger condition
        assert await rule.evaluate({"service_status": 0})

        # Test non-trigger conditions
        assert not await rule.evaluate({"service_status": 1})
        assert not await rule.evaluate({"service_status": "down"})

    @pytest.mark.asyncio
    async def test_alert_rule_evaluation_missing_metric(self):
        """Test alert rule evaluation with missing metric."""
        metrics_data = {"other_metric": 50}
        triggered = await self.rule.evaluate(metrics_data)
        assert not triggered

    @pytest.mark.asyncio
    async def test_alert_rule_evaluation_invalid_query(self):
        """Test alert rule evaluation with invalid query format."""
        rule = AlertRule(
            name="invalid_rule",
            query="invalid query format",
            duration="1m",
            severity="warning",
            manager=self.manager
        )

        metrics_data = {"cpu_usage": 90}
        triggered = await rule.evaluate(metrics_data)
        assert not triggered

    @pytest.mark.asyncio
    async def test_alert_rule_evaluation_with_deduplication(self):
        """Test alert rule evaluation with deduplication."""
        # First evaluation should trigger (no deduplication in evaluate method)
        metrics_data = {"cpu_usage": 85}
        triggered1 = await self.rule.evaluate(metrics_data)
        assert triggered1

        # Second evaluation should also trigger (deduplication is handled by manager)
        triggered2 = await self.rule.evaluate(metrics_data)
        assert triggered2

    def test_alert_rule_deduplication_logic(self):
        """Test deduplication logic."""
        # Initially no deduplication
        assert not self.rule._is_deduplicated()

        # After recording an alert, should be deduplicated
        self.rule._record_alert({
            'rule_name': self.rule.name,
            'severity': self.rule.severity,
            'description': self.rule.description,
            'timestamp': time.time(),
            'metrics_data': {"cpu_usage": 85}
        })

        assert self.rule._is_deduplicated()


class TestAlertRulesManager:
    """Test AlertRulesManager functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'deduplication_window': 300,
            'max_history_size': 1000
        }
        self.manager = AlertRulesManager(self.config)

    def test_manager_initialization(self):
        """Test AlertRulesManager initialization."""
        assert len(self.manager.rules) == 0
        assert len(self.manager.alert_history) == 0
        assert self.manager.deduplication_window == 300
        assert self.manager.max_history_size == 1000

    @pytest.mark.asyncio
    async def test_create_rule(self):
        """Test rule creation."""
        rule_config = {
            "name": "test_rule",
            "query": "cpu_usage > 80",
            "duration": "5m",
            "severity": "warning",
            "description": "High CPU usage detected",
            "channels": ["log", "email"]
        }

        rule = await self.manager.create_rule(rule_config)

        assert rule.name == "test_rule"
        assert rule.query == "cpu_usage > 80"
        assert rule.severity == "warning"
        assert rule.channels == ["log", "email"]
        assert rule.manager == self.manager

        # Verify rule is stored
        assert "test_rule" in self.manager.rules
        assert self.manager.rules["test_rule"] == rule

    @pytest.mark.asyncio
    async def test_create_duplicate_rule(self):
        """Test creating a rule with duplicate name."""
        rule_config = {
            "name": "duplicate_rule",
            "query": "cpu_usage > 80",
            "duration": "5m",
            "severity": "warning"
        }

        # Create first rule
        rule1 = await self.manager.create_rule(rule_config)

        # Create duplicate rule
        rule2 = await self.manager.create_rule(rule_config)

        # Should return the same rule instance
        assert rule1 == rule2
        assert len(self.manager.rules) == 1

    @pytest.mark.asyncio
    async def test_evaluate_rules(self):
        """Test evaluating all rules."""
        # Create multiple rules
        rule1_config = {
            "name": "high_cpu",
            "query": "cpu_usage > 80",
            "duration": "5m",
            "severity": "warning"
        }

        rule2_config = {
            "name": "low_memory",
            "query": "memory_usage < 20",
            "duration": "5m",
            "severity": "critical"
        }

        await self.manager.create_rule(rule1_config)
        await self.manager.create_rule(rule2_config)

        # Evaluate with metrics that should trigger both rules
        metrics_data = {"cpu_usage": 85, "memory_usage": 15}
        alerts = await self.manager.evaluate_rules(metrics_data)

        assert len(alerts) == 2
        assert alerts[0]['rule_name'] == 'high_cpu'
        assert alerts[1]['rule_name'] == 'low_memory'

    @pytest.mark.asyncio
    async def test_evaluate_rules_with_disabled_rule(self):
        """Test evaluating rules with disabled rule."""
        rule_config = {
            "name": "disabled_rule",
            "query": "cpu_usage > 80",
            "duration": "5m",
            "severity": "warning"
        }

        rule = await self.manager.create_rule(rule_config)
        rule.enabled = False

        metrics_data = {"cpu_usage": 85}
        alerts = await self.manager.evaluate_rules(metrics_data)

        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_evaluate_rules_with_deduplication(self):
        """Test rule evaluation with deduplication."""
        rule_config = {
            "name": "dedup_test",
            "query": "cpu_usage > 80",
            "duration": "5m",
            "severity": "warning"
        }

        await self.manager.create_rule(rule_config)

        # First evaluation should trigger
        metrics_data = {"cpu_usage": 85}
        alerts1 = await self.manager.evaluate_rules(metrics_data)
        assert len(alerts1) == 1

        # Second evaluation should be deduplicated
        alerts2 = await self.manager.evaluate_rules(metrics_data)
        assert len(alerts2) == 0

    @pytest.mark.asyncio
    async def test_notification_delivery_log(self):
        """Test log notification delivery."""
        with patch('core.alert_rules_manager.logger') as mock_logger:
            alert = {
                'rule_name': 'test_alert',
                'severity': 'warning',
                'description': 'Test alert',
                'timestamp': time.time(),
                'metrics_data': {'cpu_usage': 85}
            }

            await self.manager._deliver_notifications(alert, ['log'])

            mock_logger.warning.assert_called_once_with(
                "ALERT: test_alert - Test alert"
            )

    @pytest.mark.asyncio
    async def test_notification_delivery_discord(self):
        """Test Discord notification delivery."""
        with patch.object(self.manager, '_send_discord_notification') as mock_discord:
            alert = {
                'rule_name': 'test_alert',
                'severity': 'warning',
                'description': 'Test alert'
            }

            await self.manager._deliver_notifications(alert, ['discord'])

            mock_discord.assert_called_once_with(alert)

    @pytest.mark.asyncio
    async def test_notification_delivery_email(self):
        """Test email notification delivery."""
        with patch.object(self.manager, '_send_email_notification') as mock_email:
            alert = {
                'rule_name': 'test_alert',
                'severity': 'warning',
                'description': 'Test alert'
            }

            await self.manager._deliver_notifications(alert, ['email'])

            mock_email.assert_called_once_with(alert)

    @pytest.mark.asyncio
    async def test_notification_delivery_unknown_channel(self):
        """Test notification delivery with unknown channel."""
        with patch('core.alert_rules_manager.logger') as mock_logger:
            alert = {
                'rule_name': 'test_alert',
                'severity': 'warning',
                'description': 'Test alert'
            }

            await self.manager._deliver_notifications(alert, ['unknown'])

            mock_logger.warning.assert_called_once_with(
                "Unknown notification channel: unknown"
            )

    @pytest.mark.asyncio
    async def test_notification_delivery_multiple_channels(self):
        """Test notification delivery to multiple channels."""
        with patch.object(self.manager, '_send_discord_notification') as mock_discord, \
             patch.object(self.manager, '_send_email_notification') as mock_email:

            alert = {
                'rule_name': 'test_alert',
                'severity': 'warning',
                'description': 'Test alert'
            }

            await self.manager._deliver_notifications(alert, ['discord', 'email'])

            mock_discord.assert_called_once_with(alert)
            mock_email.assert_called_once_with(alert)

    def test_get_rules(self):
        """Test getting all rules."""
        # Initially empty
        assert self.manager.get_rules() == {}

        # Add a rule
        rule_config = {
            "name": "test_rule",
            "query": "cpu_usage > 80",
            "duration": "5m",
            "severity": "warning"
        }

        # Create rule synchronously for this test
        rule = AlertRule(**rule_config, manager=self.manager)
        self.manager.rules[rule.name] = rule

        rules = self.manager.get_rules()
        assert len(rules) == 1
        assert 'test_rule' in rules

    def test_get_alert_history(self):
        """Test getting alert history."""
        # Initially empty
        assert self.manager.get_alert_history() == []

        # Add alerts to history
        alert1 = {'rule_name': 'rule1', 'timestamp': time.time()}
        alert2 = {'rule_name': 'rule2', 'timestamp': time.time()}

        self.manager.alert_history = [alert1, alert2]

        # Get all history
        history = self.manager.get_alert_history()
        assert len(history) == 2

        # Get limited history
        limited_history = self.manager.get_alert_history(limit=1)
        assert len(limited_history) == 1
        assert limited_history[0] == alert2  # Most recent first

    def test_enable_disable_rule(self):
        """Test enabling and disabling rules."""
        rule_config = {
            "name": "test_rule",
            "query": "cpu_usage > 80",
            "duration": "5m",
            "severity": "warning"
        }

        # Create rule synchronously
        rule = AlertRule(**rule_config, manager=self.manager)
        self.manager.rules[rule.name] = rule

        # Initially enabled
        assert rule.enabled

        # Disable rule
        self.manager.disable_rule('test_rule')
        assert not rule.enabled

        # Enable rule
        self.manager.enable_rule('test_rule')
        assert rule.enabled

    def test_delete_rule(self):
        """Test deleting rules."""
        rule_config = {
            "name": "test_rule",
            "query": "cpu_usage > 80",
            "duration": "5m",
            "severity": "warning"
        }

        # Create rule synchronously
        rule = AlertRule(**rule_config, manager=self.manager)
        self.manager.rules[rule.name] = rule

        # Verify rule exists
        assert 'test_rule' in self.manager.rules

        # Delete rule
        self.manager.delete_rule('test_rule')

        # Verify rule is deleted
        assert 'test_rule' not in self.manager.rules

    def test_delete_nonexistent_rule(self):
        """Test deleting a non-existent rule."""
        # Should not raise an exception
        self.manager.delete_rule('nonexistent_rule')

    def test_enable_disable_nonexistent_rule(self):
        """Test enabling/disabling a non-existent rule."""
        # Should not raise an exception
        self.manager.enable_rule('nonexistent_rule')
        self.manager.disable_rule('nonexistent_rule')

    def test_alert_history_size_limit(self):
        """Test alert history size limiting."""
        manager = AlertRulesManager({'max_history_size': 3})

        # Add more alerts than the limit
        for i in range(5):
            alert = {'rule_name': f'rule{i}', 'timestamp': time.time()}
            manager._record_alert(alert)

        # Should only keep the most recent alerts
        assert len(manager.alert_history) == 3

    @pytest.mark.asyncio
    async def test_rule_evaluation_error_handling(self):
        """Test error handling during rule evaluation."""
        rule_config = {
            "name": "error_rule",
            "query": "cpu_usage > 80",
            "duration": "5m",
            "severity": "warning"
        }

        rule = await self.manager.create_rule(rule_config)

        # Mock evaluate to raise an exception
        with patch.object(rule, 'evaluate', side_effect=Exception("Test error")):
            with patch('core.alert_rules_manager.logger') as mock_logger:
                metrics_data = {"cpu_usage": 85}
                alerts = await self.manager.evaluate_rules(metrics_data)

                # Should continue processing despite error
                assert len(alerts) == 0
                mock_logger.exception.assert_called_once()


class TestAlertRuleEdgeCases:
    """Test AlertRule edge cases."""

    def setup_method(self):
        """Setup test fixtures."""
        self.manager = AlertRulesManager({})

    @pytest.mark.asyncio
    async def test_rule_with_empty_query(self):
        """Test rule with empty query."""
        rule = AlertRule(
            name="empty_query",
            query="",
            duration="5m",
            severity="warning",
            manager=self.manager
        )

        metrics_data = {"cpu_usage": 85}
        triggered = await rule.evaluate(metrics_data)
        assert not triggered

    @pytest.mark.asyncio
    async def test_rule_with_whitespace_query(self):
        """Test rule with whitespace-only query."""
        rule = AlertRule(
            name="whitespace_query",
            query="   ",
            duration="5m",
            severity="warning",
            manager=self.manager
        )

        metrics_data = {"cpu_usage": 85}
        triggered = await rule.evaluate(metrics_data)
        assert not triggered

    @pytest.mark.asyncio
    async def test_rule_with_malformed_query(self):
        """Test rule with malformed query."""
        malformed_queries = [
            "cpu_usage >",  # Missing value
            "> 80",  # Missing metric
            "cpu_usage 80",  # Missing operator
            "cpu_usage >> 80",  # Invalid operator
            "cpu_usage > abc",  # Non-numeric value
        ]

        for query in malformed_queries:
            rule = AlertRule(
                name=f"malformed_{hash(query)}",
                query=query,
                duration="5m",
                severity="warning",
                manager=self.manager
            )

            metrics_data = {"cpu_usage": 85}
            triggered = await rule.evaluate(metrics_data)
            assert not triggered, f"Query '{query}' should not trigger"

    @pytest.mark.asyncio
    async def test_rule_with_float_values(self):
        """Test rule evaluation with float values."""
        rule = AlertRule(
            name="float_rule",
            query="temperature > 98.6",
            duration="5m",
            severity="warning",
            manager=self.manager
        )

        # Test with float values
        assert await rule.evaluate({"temperature": 99.5})
        assert await rule.evaluate({"temperature": 100.0})
        assert not await rule.evaluate({"temperature": 98.0})

    @pytest.mark.asyncio
    async def test_rule_with_string_values(self):
        """Test rule evaluation with string values."""
        rule = AlertRule(
            name="string_rule",
            query="status == down",
            duration="5m",
            severity="critical",
            manager=self.manager
        )

        # Test with string values
        assert await rule.evaluate({"status": "down"})
        assert not await rule.evaluate({"status": "up"})
        assert not await rule.evaluate({"status": "maintenance"})


class TestAlertRulesManagerIntegration:
    """Test AlertRulesManager integration scenarios."""

    def setup_method(self):
        """Setup test fixtures."""
        self.manager = AlertRulesManager({
            'deduplication_window': 60,  # 1 minute
            'max_history_size': 10
        })

    @pytest.mark.asyncio
    async def test_multiple_rules_same_metric(self):
        """Test multiple rules evaluating the same metric."""
        # Create rules with different thresholds
        rule1_config = {
            "name": "warning_rule",
            "query": "cpu_usage > 70",
            "duration": "5m",
            "severity": "warning"
        }

        rule2_config = {
            "name": "critical_rule",
            "query": "cpu_usage > 90",
            "duration": "5m",
            "severity": "critical"
        }

        await self.manager.create_rule(rule1_config)
        await self.manager.create_rule(rule2_config)

        # Test with value that triggers warning but not critical
        metrics_data = {"cpu_usage": 80}
        alerts = await self.manager.evaluate_rules(metrics_data)

        assert len(alerts) == 1
        assert alerts[0]['rule_name'] == 'warning_rule'
        assert alerts[0]['severity'] == 'warning'

        # Clear alert history to test deduplication doesn't prevent different rules
        self.manager.alert_history.clear()

        # Test with value that triggers both
        metrics_data = {"cpu_usage": 95}
        alerts = await self.manager.evaluate_rules(metrics_data)

        assert len(alerts) == 2
        rule_names = [alert['rule_name'] for alert in alerts]
        assert 'warning_rule' in rule_names
        assert 'critical_rule' in rule_names

    @pytest.mark.asyncio
    async def test_complex_metrics_evaluation(self):
        """Test evaluation with complex metrics data."""
        rule_config = {
            "name": "complex_rule",
            "query": "error_rate > 0.05",
            "duration": "5m",
            "severity": "warning"
        }

        await self.manager.create_rule(rule_config)

        # Test with complex metrics
        metrics_data = {
            "cpu_usage": 85.5,
            "memory_usage": 78.2,
            "disk_usage": 45.1,
            "error_rate": 0.08,
            "response_time": 2.5,
            "throughput": 150.0
        }

        alerts = await self.manager.evaluate_rules(metrics_data)

        assert len(alerts) == 1
        assert alerts[0]['rule_name'] == 'complex_rule'
        assert alerts[0]['severity'] == 'warning'

    @pytest.mark.asyncio
    async def test_deduplication_window_expiration(self):
        """Test deduplication window expiration."""
        rule_config = {
            "name": "dedup_test",
            "query": "cpu_usage > 80",
            "duration": "5m",
            "severity": "warning"
        }

        await self.manager.create_rule(rule_config)

        # First alert
        metrics_data = {"cpu_usage": 85}
        alerts1 = await self.manager.evaluate_rules(metrics_data)
        assert len(alerts1) == 1

        # Modify deduplication window to be very short
        self.manager.deduplication_window = 0.1  # 100ms

        # Wait for deduplication window to expire
        await asyncio.sleep(0.2)

        # Second alert should trigger again
        alerts2 = await self.manager.evaluate_rules(metrics_data)
        assert len(alerts2) == 1
