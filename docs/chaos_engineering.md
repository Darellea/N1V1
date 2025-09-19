# Chaos Engineering for N1V1 Trading System

This document describes the chaos engineering framework implemented for the N1V1 trading system. Chaos engineering is a disciplined approach to identifying failures before they become outages by proactively testing how systems respond to turbulent conditions in production.

## Overview

The N1V1 chaos engineering suite simulates real-world operational hazards to validate system resilience and ensure automatic recovery within SLA requirements. The framework includes both stress testing for high-load scenarios and chaos experiments for failure injection.

## Supported Chaos Scenarios

### Network Partition
- **Description**: Simulates network connectivity issues causing API timeouts
- **Duration**: 60 seconds
- **Expected Impact**: 100% request failure during partition
- **SLA Recovery Time**: ≤30 seconds
- **Failure Type**: `CHAOS_NETWORK_PARTITION`

### Rate Limit Flood
- **Description**: Simulates exchange rate limiting with HTTP 429 responses
- **Duration**: 30 seconds
- **Expected Impact**: 50% request failure rate
- **SLA Recovery Time**: ≤45 seconds
- **Failure Type**: `CHAOS_RATE_LIMIT`

### Exchange Downtime
- **Description**: Simulates complete exchange API unavailability
- **Duration**: 5 minutes (300 seconds)
- **Expected Impact**: 100% request failure during downtime
- **SLA Recovery Time**: ≤60 seconds
- **Failure Type**: `CHAOS_EXCHANGE_DOWNTIME`

### Database Outage
- **Description**: Simulates database connection failures
- **Duration**: 2 minutes (120 seconds)
- **Expected Impact**: 80% request failure rate
- **SLA Recovery Time**: ≤90 seconds
- **Failure Type**: `CHAOS_DATABASE_OUTAGE`

## Load Stress Testing

In addition to chaos scenarios, the framework includes comprehensive load testing:

- **Concurrent Strategies**: 50+ simultaneous trading strategies
- **Order Burst Generation**: 10k+ orders per burst across strategies
- **Latency SLA**: P95 response time ≤150ms
- **Failure Rate SLA**: ≤0.5% error rate
- **Throughput Requirement**: ≥100 orders/second

## Running Chaos Tests Locally

### Prerequisites

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Ensure pytest is available
pip install pytest

# For chaos test orchestration script
chmod +x scripts/chaos_test.sh
```

### Running Individual Scenarios

```bash
# Run all chaos scenarios
python -m pytest tests/stress/chaos_tests.py -v

# Run specific scenario
python -m pytest tests/stress/chaos_tests.py::test_network_partition_chaos -v

# Run with verbose output
python -m pytest tests/stress/chaos_tests.py -v -s
```

### Using the Orchestration Script

The `scripts/chaos_test.sh` script provides advanced orchestration capabilities:

```bash
# Run all scenarios sequentially (default)
./scripts/chaos_test.sh

# Run specific scenarios
./scripts/chaos_test.sh --scenarios network_partition,rate_limit_flood

# Run in random order
./scripts/chaos_test.sh --mode random

# Run scenarios in parallel
./scripts/chaos_test.sh --parallel

# Extended test duration
./scripts/chaos_test.sh --duration 3600

# Custom output directory
./scripts/chaos_test.sh --output-dir ./my_chaos_reports

# Verbose logging
./scripts/chaos_test.sh --verbose

# Show help
./scripts/chaos_test.sh --help
```

### Running Load Stress Tests

```bash
# Run load stress tests
python tests/stress/test_load.py

# Custom configuration
python tests/stress/test_load.py --strategies 100 --orders-per-burst 2000 --bursts 5
```

## Chaos Test Reports

### Report Structure

Chaos test results are saved to the `chaos_reports/` directory with the following structure:

```
chaos_reports/
├── chaos_run_20231201_020000/          # Timestamped run directory
│   ├── network_partition_report.json   # Individual scenario reports
│   ├── rate_limit_flood_report.json
│   ├── exchange_downtime_report.json
│   ├── database_outage_report.json
│   └── chaos_summary.json              # Consolidated summary
```

### Individual Scenario Report

Each scenario generates a detailed JSON report:

```json
{
  "scenario": {
    "name": "network_partition",
    "description": "Network partition causing 60s API timeouts",
    "duration_seconds": 60,
    "expected_failure_rate": 100.0,
    "sla_recovery_time_seconds": 30
  },
  "execution": {
    "start_time": "2023-12-01T02:00:00.000Z",
    "end_time": "2023-12-01T02:01:00.000Z",
    "recovery_time_seconds": 25.3,
    "failure_injected": true,
    "recovery_detected": true
  },
  "metrics": {
    "orders_before_chaos": 150,
    "orders_during_chaos": 0,
    "orders_after_chaos": 148,
    "failures_before_chaos": 1,
    "failures_during_chaos": 50,
    "failures_after_chaos": 1,
    "chaos_failure_rate": 100.0,
    "recovery_successful": true
  },
  "validation": {
    "recovery_detected": true,
    "recovery_time_seconds": 25.3,
    "sla_met": true,
    "failure_rate_acceptable": true,
    "watchdog_alerts_fired": true,
    "recovery_events_logged": true
  },
  "watchdog": {
    "alerts_count": 3,
    "recovery_events_count": 1
  },
  "timestamp": "2023-12-01T02:01:05.000Z"
}
```

### Summary Report

The `chaos_summary.json` provides an overview of all scenarios:

```json
{
  "test_run": {
    "timestamp": "2023-12-01T02:01:05.000Z",
    "duration_seconds": 1850,
    "mode": "sequential",
    "parallel": false,
    "total_scenarios": 4,
    "scenarios_run": ["network_partition", "rate_limit_flood", "exchange_downtime", "database_outage"]
  },
  "results": {
    "passed_scenarios": 4,
    "failed_scenarios": 0,
    "pass_rate_percent": 100.0,
    "average_recovery_time_seconds": 32.5
  }
}
```

## Interpreting Chaos Reports

### Key Metrics to Analyze

1. **Recovery Detection**: Whether the system detected and recovered from chaos
2. **Recovery Time**: Time taken to restore normal operation
3. **SLA Compliance**: Whether recovery met time requirements
4. **Failure Rate**: Actual vs. expected failure rates during chaos
5. **Watchdog Alerts**: Whether monitoring systems detected issues

### Pass/Fail Criteria

A chaos scenario **PASSES** when:
- ✅ Recovery is detected within the scenario duration
- ✅ Recovery time meets SLA requirements (`sla_met: true`)
- ✅ Failure rate during chaos is within expected bounds
- ✅ Watchdog alerts are fired appropriately
- ✅ Recovery events are logged

A chaos scenario **FAILS** when:
- ❌ No recovery is detected
- ❌ Recovery time exceeds SLA limits
- ❌ Unexpected failure rates outside acceptable ranges
- ❌ Monitoring systems fail to detect issues

### SLA Expectations

| Scenario | Max Recovery Time | Acceptable Failure Rate |
|----------|------------------|-------------------------|
| Network Partition | 30 seconds | 90-100% |
| Rate Limit Flood | 45 seconds | 40-60% |
| Exchange Downtime | 60 seconds | 95-100% |
| Database Outage | 90 seconds | 70-90% |

### Load Test SLA

| Metric | Threshold | Status |
|--------|-----------|--------|
| P95 Latency | ≤150ms | ✅ PASS |
| Failure Rate | ≤0.5% | ✅ PASS |
| Throughput | ≥100 orders/sec | ✅ PASS |

## Chaos Event Logging

Chaos engineering events are logged with structured data for analysis:

```
INFO: CHAOS_EVENT: {
  "timestamp": "2023-12-01T02:00:30.000Z",
  "event_type": "chaos_failure_detected",
  "chaos_engineering": true,
  "component_id": "order_executor",
  "failure_type": "chaos_network_partition",
  "error_message": "Simulated network partition",
  "confidence": 0.95,
  "metadata": {"chaos_scenario": "network_partition"}
}
```

## CI/CD Integration

### Nightly Chaos Runs

Chaos tests run automatically every night at 2 AM UTC on the main branch:

```yaml
# .github/workflows/ci.yml
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC

jobs:
  chaos-engineering:
    if: github.ref == 'refs/heads/main' && github.event_name == 'schedule'
    # ... chaos test execution
```

### Chaos Test Artifacts

- **Retention**: 90 days
- **Location**: GitHub Actions artifacts
- **Notification**: Discord alerts for pass/fail status

## Troubleshooting

### Common Issues

1. **Chaos tests not running**: Ensure pytest and dependencies are installed
2. **Permission denied on script**: Run `chmod +x scripts/chaos_test.sh`
3. **Import errors**: Check Python path and virtual environment
4. **Report generation fails**: Verify write permissions on output directory

### Debugging Chaos Tests

```bash
# Run with maximum verbosity
./scripts/chaos_test.sh --verbose

# Run individual scenario for debugging
python -m pytest tests/stress/chaos_tests.py::test_network_partition_chaos -v -s

# Check chaos event logs
grep "CHAOS_EVENT" logs/*.log
```

## Best Practices

1. **Run chaos tests in staging first**: Never run untested chaos scenarios in production
2. **Start with short durations**: Gradually increase chaos duration as confidence grows
3. **Monitor system resources**: Chaos testing can impact system performance
4. **Review reports regularly**: Use chaos reports to identify systemic weaknesses
5. **Automate remediation**: Implement automatic fixes for known chaos patterns

## Integration with Monitoring

Chaos events integrate with the existing watchdog monitoring system:

- **Failure Detection**: Chaos failures are classified and diagnosed
- **Alert Generation**: Structured alerts for chaos events
- **Recovery Tracking**: Automatic recovery orchestration
- **Metrics Collection**: Performance impact measurement

## Future Enhancements

- **Kubernetes Chaos**: Integration with Chaos Mesh for container-level chaos
- **Multi-region Testing**: Cross-region network partition simulation
- **Database Chaos**: More sophisticated database failure scenarios
- **Load Balancer Chaos**: Proxy and load balancer failure injection
- **Automated Remediation**: ML-driven automatic fix suggestions

## Support

For questions about chaos engineering in N1V1:

- **Documentation**: This document and inline code comments
- **Logs**: Check `logs/` directory for chaos event details
- **Reports**: Review `chaos_reports/` for execution details
- **Monitoring**: Watchdog alerts provide real-time chaos status

---

*Chaos engineering ensures N1V1 maintains 99.99% uptime by proactively testing failure scenarios and validating automatic recovery mechanisms.*
