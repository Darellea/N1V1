# Health and Readiness Endpoints - Monitoring Integration Guide

This document describes the implementation of standardized health and readiness endpoints in the N1V1 trading framework and their integration with monitoring systems.

## Overview

The N1V1 framework now provides two critical HTTP endpoints for system monitoring:

- **`/api/v1/health`** - Lightweight health check (always returns 200 if service is running)
- **`/api/v1/ready`** - Comprehensive readiness check (returns 503 if critical dependencies fail)

## Endpoints

### Health Endpoint (`/api/v1/health`)

**Purpose**: Provides a lightweight heartbeat check to verify the service is running.

**HTTP Method**: GET
**Path**: `/api/v1/health`
**Authentication**: None required
**Rate Limiting**: Subject to global rate limits

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-19T09:21:36.123456",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "bot_engine": "available",
  "correlation_id": "health_000001",
  "check_latency_ms": 1.23
}
```

**Response Fields**:
- `status`: Always "healthy" (or "unhealthy" if critical failure)
- `timestamp`: ISO 8601 timestamp of the check
- `version`: Framework version
- `uptime_seconds`: Service uptime in seconds
- `bot_engine`: Bot engine availability status
- `correlation_id`: Unique identifier for log correlation
- `check_latency_ms`: Time taken to perform the check

### Readiness Endpoint (`/api/v1/ready`)

**Purpose**: Validates all critical dependencies before allowing traffic.

**HTTP Method**: GET
**Path**: `/api/v1/ready`
**Authentication**: None required
**Rate Limiting**: Subject to global rate limits

**Response (200 OK - Ready)**:
```json
{
  "ready": true,
  "timestamp": "2025-01-19T09:21:36.123456",
  "correlation_id": "health_000002",
  "total_latency_ms": 45.67,
  "checks": {
    "database": {
      "ready": true,
      "latency_ms": 12.34,
      "message": "Database connection successful",
      "details": {"type": "postgresql"}
    },
    "exchange": {
      "ready": true,
      "latency_ms": 23.45,
      "message": "Exchange API responsive",
      "details": {"status_code": 200, "url": "https://api.binance.com/api/v3/ping"}
    },
    "message_queue": {
      "ready": true,
      "latency_ms": null,
      "message": "Message queue not configured (optional)",
      "details": {"configured": false, "optional": true}
    },
    "cache": {
      "ready": true,
      "latency_ms": 5.67,
      "message": "Cache connection successful",
      "details": {"type": "redis"}
    },
    "bot_engine": {
      "ready": true,
      "latency_ms": 4.21,
      "message": "Bot engine ready",
      "details": {"running": true}
    }
  }
}
```

**Response (503 Service Unavailable - Not Ready)**:
```json
{
  "ready": false,
  "timestamp": "2025-01-19T09:21:36.123456",
  "correlation_id": "health_000003",
  "total_latency_ms": 123.45,
  "checks": {
    "database": {
      "ready": false,
      "latency_ms": 100.12,
      "message": "Database connection failed: Connection timeout",
      "details": {"error": "Connection timeout", "type": "postgresql"}
    },
    "exchange": {
      "ready": true,
      "latency_ms": 23.45,
      "message": "Exchange API responsive",
      "details": {"status_code": 200, "url": "https://api.binance.com/api/v3/ping"}
    }
  }
}
```

## Dependency Checks

The readiness endpoint performs the following checks:

### 1. Database Connectivity
- **Environment Variable**: `DATABASE_URL`
- **Supported Types**: PostgreSQL, MySQL, MongoDB, SQLAlchemy-compatible
- **Timeout**: 5 seconds
- **Optional**: No (critical dependency)

### 2. Exchange API Connectivity
- **Environment Variable**: `EXCHANGE_API_URL` (defaults to Binance ping endpoint)
- **Protocol**: HTTP/HTTPS
- **Timeout**: 5 seconds
- **Optional**: No (critical dependency)

### 3. Message Queue Connectivity
- **Environment Variables**: `MESSAGE_QUEUE_URL` or `RABBITMQ_URL`
- **Supported Types**: RabbitMQ (AMQP), Kafka
- **Timeout**: N/A (URL format validation only)
- **Optional**: Yes (considered ready if not configured)

### 4. Cache Connectivity
- **Environment Variable**: `REDIS_URL` (defaults to `redis://localhost:6379/0`)
- **Type**: Redis
- **Timeout**: 5 seconds
- **Optional**: No (critical dependency)

### 5. Bot Engine Readiness
- **Source**: Global bot engine instance
- **Checks**: Engine initialization, required attributes, running state
- **Timeout**: N/A (fast attribute check)
- **Optional**: No (critical dependency)

## Monitoring Integration

### Prometheus Integration

The existing `/metrics` endpoint provides Prometheus-compatible metrics. The health endpoints complement this by providing application-level health signals.

**Recommended Prometheus Configuration**:
```yaml
scrape_configs:
  - job_name: 'n1v1-trading-bot'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'n1v1-health'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/api/v1/health'
    scrape_interval: 30s
    # Note: Health endpoint returns JSON, not Prometheus format
```

### Kubernetes Integration

**Readiness Probe**:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: n1v1-trading-bot
spec:
  containers:
  - name: trading-bot
    image: n1v1/trading-bot:latest
    ports:
    - containerPort: 8000
    readinessProbe:
      httpGet:
        path: /api/v1/ready
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3
    livenessProbe:
      httpGet:
        path: /api/v1/health
        port: 8000
      initialDelaySeconds: 60
      periodSeconds: 30
      timeoutSeconds: 3
```

**Service Configuration**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: n1v1-trading-bot
spec:
  selector:
    app: n1v1-trading-bot
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

### Grafana Dashboards

**Recommended Dashboard Panels**:

1. **Health Status Panel**:
   - Query: `up{job="n1v1-health"}`
   - Type: Status panel
   - Colors: Green (healthy), Red (unhealthy)

2. **Readiness Status Panel**:
   - Query: Custom JSON API data source pointing to `/api/v1/ready`
   - Display readiness status and individual component statuses

3. **Dependency Latency Panel**:
   - Query: Parse latency metrics from readiness endpoint JSON
   - Type: Graph panel
   - Track performance of each dependency check

### Alerting Rules

**Prometheus Alerting Rules**:
```yaml
groups:
- name: n1v1-health
  rules:
  - alert: N1V1ServiceUnhealthy
    expr: up{job="n1v1-health"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "N1V1 trading bot health check failed"
      description: "N1V1 service is not responding to health checks"

  - alert: N1V1ServiceNotReady
    expr: |
      rate(http_requests_total{job="n1v1", status="503", path="/api/v1/ready"}[5m]) > 0
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "N1V1 trading bot not ready"
      description: "N1V1 service is returning 503 on readiness checks"
```

### ELK Stack Integration

**Logstash Configuration**:
```json
input {
  http {
    host => "0.0.0.0"
    port => 8080
    url_path => "/api/v1/health"
  }
}

filter {
  json {
    source => "message"
  }

  if [correlation_id] {
    mutate {
      add_field => { "request_id" => "%{correlation_id}" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "n1v1-health-%{+YYYY.MM.dd}"
  }
}
```

## Logging

### Health Check Logging

Health checks are logged at DEBUG level with correlation IDs for traceability:

```
DEBUG - Health check completed correlation_id=health_000001 latency_ms=1.23 status=healthy
```

### Readiness Check Logging

Readiness failures are logged with throttling (every 5 minutes per component) to prevent log spam:

```
WARNING - Readiness check failed correlation_id=health_000002 failed_components=['database'] total_checks=5 latency_ms=123.45
ERROR - Readiness check failed for database correlation_id=health_000002 component=database message=Database connection failed: Connection timeout latency_ms=100.12 details={"error": "Connection timeout", "type": "postgresql"}
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | Database connection string | None | Yes |
| `EXCHANGE_API_URL` | Exchange API endpoint | `https://api.binance.com/api/v3/ping` | No |
| `MESSAGE_QUEUE_URL` | Message queue connection string | None | No |
| `RABBITMQ_URL` | RabbitMQ connection string | None | No |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` | No |

### Timeouts

- **Database**: 5 seconds
- **Exchange API**: 5 seconds
- **Cache**: 5 seconds
- **Message Queue**: URL format validation only
- **Bot Engine**: Fast attribute check

## Testing

### Unit Tests

The implementation includes comprehensive unit tests covering:

- Health endpoint response structure and metadata
- Readiness endpoint component checks
- Error handling for missing dependencies
- Correlation ID uniqueness
- Timestamp format validation
- HTTP status code handling

### Integration Tests

For production deployments, consider:

1. **Load Testing**: Verify endpoints perform under high concurrency
2. **Dependency Failure Simulation**: Test behavior when dependencies are unavailable
3. **Network Latency Testing**: Ensure timeouts work correctly
4. **Rate Limiting Testing**: Verify endpoints respect rate limits

## Troubleshooting

### Common Issues

1. **503 on Readiness but 200 on Health**:
   - Check dependency configurations
   - Verify network connectivity to external services
   - Review database/cache connection strings

2. **Slow Response Times**:
   - Check network latency to external services
   - Verify database query performance
   - Monitor system resource usage

3. **Correlation ID Issues**:
   - Ensure proper logging configuration
   - Check log aggregation system setup

### Debug Commands

```bash
# Test health endpoint
curl -v http://localhost:8000/api/v1/health

# Test readiness endpoint
curl -v http://localhost:8000/api/v1/ready

# Check logs for correlation IDs
grep "correlation_id=health_" /var/log/n1v1/application.log

# Test with Prometheus
curl -H "Accept: text/plain" http://localhost:8000/metrics
```

## Security Considerations

- Health and readiness endpoints do not require authentication
- They provide detailed system information - consider network segmentation
- Rate limiting is applied to prevent abuse
- CORS is configured to restrict cross-origin requests
- Sensitive connection details are not exposed in responses

## Performance Impact

- **Health Endpoint**: Minimal impact (< 5ms typical)
- **Readiness Endpoint**: Moderate impact (50-200ms typical)
- Both endpoints are optimized for low latency
- Consider endpoint-specific rate limits for high-traffic deployments

## Future Enhancements

Potential improvements:

1. **Custom Health Checks**: Plugin system for application-specific checks
2. **Metrics Integration**: Expose health check metrics via Prometheus
3. **Circuit Breakers**: Automatic failover for failing dependencies
4. **Health Check Caching**: Cache results to reduce load on dependencies
5. **Advanced Monitoring**: Integration with distributed tracing systems
