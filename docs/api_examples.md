# N1V1 API Examples

Complete guide for using N1V1's REST API with practical examples in Python and curl.

## Overview

N1V1 provides a comprehensive REST API for monitoring, controlling, and integrating with the trading system. The API includes endpoints for:

- **Bot Control**: Start, stop, pause, and resume trading
- **Monitoring**: Real-time status, performance metrics, and health checks
- **Signals & Orders**: Access trading signals and order history
- **Metrics**: Prometheus-compatible metrics endpoint
- **Configuration**: Dynamic configuration management

## Authentication

### API Key Authentication

Most endpoints require an API key for security:

```bash
# Set API key as environment variable
export API_KEY="your_secure_api_key_here"

# Or pass in request headers
curl -H "Authorization: Bearer your_api_key" http://localhost:8000/api/v1/status
```

### Configuration

Set the API key in your environment or configuration:

```bash
# Environment variable
export API_KEY="your_secure_api_key"

# Or in config.json
{
  "api": {
    "enabled": true,
    "key": "your_secure_api_key"
  }
}
```

## Base URL

All API endpoints are prefixed with `/api/v1`:

```bash
BASE_URL="http://localhost:8000/api/v1"
```

## Core Endpoints

### Bot Status

#### Get Bot Status

```bash
curl -H "Authorization: Bearer $API_KEY" \
     http://localhost:8000/api/v1/status
```

**Response:**
```json
{
  "running": true,
  "paused": false,
  "mode": "paper",
  "pairs": ["BTC/USDT", "ETH/USDT"],
  "timestamp": "2025-09-19T12:30:00Z"
}
```

**Python Example:**
```python
import requests

def get_bot_status(api_key: str, base_url: str = "http://localhost:8000"):
    """Get current bot status."""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{base_url}/api/v1/status", headers=headers)
    return response.json()

# Usage
status = get_bot_status("your_api_key")
print(f"Bot running: {status['running']}")
print(f"Mode: {status['mode']}")
```

### Bot Control

#### Pause Bot

```bash
curl -X POST \
     -H "Authorization: Bearer $API_KEY" \
     http://localhost:8000/api/v1/pause
```

**Response:**
```json
{
  "message": "Bot paused successfully"
}
```

#### Resume Bot

```bash
curl -X POST \
     -H "Authorization: Bearer $API_KEY" \
     http://localhost:8000/api/v1/resume
```

**Response:**
```json
{
  "message": "Bot resumed successfully"
}
```

**Python Example:**
```python
import requests

def control_bot(api_key: str, action: str, base_url: str = "http://localhost:8000"):
    """Control bot (pause/resume)."""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(f"{base_url}/api/v1/{action}", headers=headers)
    return response.json()

# Pause bot
result = control_bot("your_api_key", "pause")
print(result["message"])

# Resume bot
result = control_bot("your_api_key", "resume")
print(result["message"])
```

### Trading Signals

#### Get Recent Signals

```bash
curl -H "Authorization: Bearer $API_KEY" \
     http://localhost:8000/api/v1/signals
```

**Response:**
```json
{
  "signals": [
    {
      "id": "ema_cross_1632000000000",
      "symbol": "BTC/USDT",
      "timestamp": "2025-09-19T12:30:00Z",
      "confidence": 0.85,
      "signal_type": "ENTRY_LONG",
      "strategy": "ema_cross"
    },
    {
      "id": "rsi_strategy_1632000000000",
      "symbol": "ETH/USDT",
      "timestamp": "2025-09-19T12:25:00Z",
      "confidence": 0.72,
      "signal_type": "ENTRY_SHORT",
      "strategy": "rsi_strategy"
    }
  ]
}
```

**Python Example:**
```python
import requests
from datetime import datetime

def get_recent_signals(api_key: str, base_url: str = "http://localhost:8000"):
    """Get recent trading signals."""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{base_url}/api/v1/signals", headers=headers)
    signals = response.json()["signals"]

    for signal in signals:
        timestamp = datetime.fromisoformat(signal["timestamp"].replace('Z', '+00:00'))
        print(f"{signal['strategy']}: {signal['signal_type']} {signal['symbol']} "
              f"(confidence: {signal['confidence']:.2f}) at {timestamp}")

    return signals

# Usage
signals = get_recent_signals("your_api_key")
```

### Order Management

#### Get Recent Orders

```bash
curl -H "Authorization: Bearer $API_KEY" \
     http://localhost:8000/api/v1/orders
```

**Response:**
```json
{
  "orders": [
    {
      "id": "order_12345",
      "symbol": "BTC/USDT",
      "timestamp": "2025-09-19T12:30:00Z",
      "side": "BUY",
      "quantity": 0.001,
      "price": 45000.00,
      "pnl": 150.25,
      "equity": 10050.75,
      "cumulative_return": 0.015
    }
  ]
}
```

**Python Example:**
```python
import requests

def get_recent_orders(api_key: str, base_url: str = "http://localhost:8000"):
    """Get recent orders/trades."""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{base_url}/api/v1/orders", headers=headers)
    orders = response.json()["orders"]

    total_pnl = 0
    for order in orders:
        print(f"{order['symbol']} {order['side']} {order['quantity']} @ {order['price']}")
        print(f"  PnL: ${order['pnl']:.2f}, Equity: ${order['equity']:.2f}")
        total_pnl += order['pnl']

    print(f"Total PnL: ${total_pnl:.2f}")
    return orders

# Usage
orders = get_recent_orders("your_api_key")
```

### Performance Metrics

#### Get Performance Data

```bash
curl -H "Authorization: Bearer $API_KEY" \
     http://localhost:8000/api/v1/performance
```

**Response:**
```json
{
  "total_pnl": 1250.75,
  "win_rate": 0.68,
  "wins": 34,
  "losses": 16,
  "sharpe_ratio": 1.45,
  "max_drawdown": 0.08
}
```

**Python Example:**
```python
import requests

def get_performance_metrics(api_key: str, base_url: str = "http://localhost:8000"):
    """Get bot performance metrics."""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{base_url}/api/v1/performance", headers=headers)
    metrics = response.json()

    print("=== Performance Metrics ===")
    print(f"Total PnL: ${metrics['total_pnl']:.2f}")
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Wins/Losses: {metrics['wins']}/{metrics['losses']}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.1%}")

    return metrics

# Usage
metrics = get_performance_metrics("your_api_key")
```

### Equity Data

#### Get Equity Curve

```bash
curl -H "Authorization: Bearer $API_KEY" \
     http://localhost:8000/api/v1/equity
```

**Response:**
```json
{
  "equity_curve": [
    {
      "timestamp": "2025-09-19T10:00:00Z",
      "balance": 10000.00,
      "equity": 10000.00,
      "cumulative_return": 0.0
    },
    {
      "timestamp": "2025-09-19T11:00:00Z",
      "balance": 10150.25,
      "equity": 10150.25,
      "cumulative_return": 0.015
    }
  ]
}
```

**Python Example:**
```python
import requests
import matplotlib.pyplot as plt
from datetime import datetime

def get_equity_curve(api_key: str, base_url: str = "http://localhost:8000"):
    """Get equity curve data."""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{base_url}/api/v1/equity", headers=headers)
    data = response.json()["equity_curve"]

    timestamps = []
    equity_values = []

    for point in data:
        timestamp = datetime.fromisoformat(point["timestamp"].replace('Z', '+00:00'))
        timestamps.append(timestamp)
        equity_values.append(point["equity"])

    return timestamps, equity_values

def plot_equity_curve(api_key: str):
    """Plot equity curve."""
    timestamps, equity = get_equity_curve(api_key)

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, equity)
    plt.title('N1V1 Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.show()

# Usage
timestamps, equity = get_equity_curve("your_api_key")
plot_equity_curve("your_api_key")
```

## Health & Monitoring

### Health Check

#### Basic Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-19T12:30:00Z",
  "version": "1.0.0",
  "uptime": "2h 15m"
}
```

#### Readiness Check

```bash
curl http://localhost:8000/ready
```

**Response:**
```json
{
  "status": "ready",
  "checks": {
    "database": "ok",
    "redis": "ok",
    "strategies": "ok",
    "exchange_connection": "ok"
  },
  "timestamp": "2025-09-19T12:30:00Z"
}
```

**Python Example:**
```python
import requests

def check_system_health(base_url: str = "http://localhost:8000"):
    """Check system health and readiness."""
    # Basic health check
    health_response = requests.get(f"{base_url}/health")
    health_data = health_response.json()

    print("=== Health Check ===")
    print(f"Status: {health_data['status']}")
    print(f"Version: {health_data['version']}")
    print(f"Uptime: {health_data['uptime']}")

    # Readiness check
    ready_response = requests.get(f"{base_url}/ready")
    ready_data = ready_response.json()

    print("\n=== Readiness Check ===")
    print(f"Overall Status: {ready_data['status']}")
    print("Component Status:")
    for component, status in ready_data['checks'].items():
        print(f"  {component}: {status}")

    return health_data, ready_data

# Usage
health, ready = check_system_health()
```

### Prometheus Metrics

#### Get Metrics

```bash
curl http://localhost:8000/metrics
```

**Response:**
```
# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total{method="GET",endpoint="/api/v1/status",status="200"} 42

# HELP trades_total Total number of trades executed
# TYPE trades_total counter
trades_total{symbol="BTC/USDT",side="BUY"} 15

# HELP wins_total Total number of winning trades
# TYPE wins_total counter
wins_total 10

# HELP losses_total Total number of losing trades
# TYPE losses_total counter
losses_total 5
```

**Python Example:**
```python
import requests

def get_prometheus_metrics(base_url: str = "http://localhost:8000"):
    """Get Prometheus metrics."""
    response = requests.get(f"{base_url}/metrics")
    metrics_text = response.text

    # Parse metrics (basic example)
    lines = metrics_text.split('\n')
    metrics = {}

    for line in lines:
        if line.startswith('# HELP'):
            parts = line.split(' ', 2)
            if len(parts) >= 2:
                metric_name = parts[1]
                metrics[metric_name] = {'help': parts[2] if len(parts) > 2 else ''}
        elif not line.startswith('#') and line.strip():
            parts = line.split(' ')
            if len(parts) >= 2:
                metric_name = parts[0].split('{')[0]  # Remove labels
                value = float(parts[1])
                if metric_name in metrics:
                    metrics[metric_name]['value'] = value

    return metrics

# Usage
metrics = get_prometheus_metrics()
for name, data in metrics.items():
    if 'value' in data:
        print(f"{name}: {data['value']}")
```

## Python Client Library

### Basic Client Class

```python
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

class N1V1Client:
    """Python client for N1V1 API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _get(self, endpoint: str) -> Dict[str, Any]:
        """Make GET request to API."""
        url = f"{self.base_url}/api/v1/{endpoint}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make POST request to API."""
        url = f"{self.base_url}/api/v1/{endpoint}"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def get_status(self) -> Dict[str, Any]:
        """Get bot status."""
        return self._get("status")

    def get_signals(self) -> List[Dict[str, Any]]:
        """Get recent signals."""
        return self._get("signals")["signals"]

    def get_orders(self) -> List[Dict[str, Any]]:
        """Get recent orders."""
        return self._get("orders")["orders"]

    def get_performance(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self._get("performance")

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """Get equity curve data."""
        return self._get("equity")["equity_curve"]

    def pause_bot(self) -> Dict[str, Any]:
        """Pause the bot."""
        return self._post("pause")

    def resume_bot(self) -> Dict[str, Any]:
        """Resume the bot."""
        return self._post("resume")

    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_readiness(self) -> Dict[str, Any]:
        """Get readiness status."""
        response = self.session.get(f"{self.base_url}/ready")
        response.raise_for_status()
        return response.json()

# Usage example
client = N1V1Client(api_key="your_api_key")

# Get status
status = client.get_status()
print(f"Bot is {'running' if status['running'] else 'stopped'}")

# Get recent signals
signals = client.get_signals()
for signal in signals[:5]:  # Show first 5
    print(f"{signal['strategy']}: {signal['signal_type']} for {signal['symbol']}")

# Get performance
perf = client.get_performance()
print(f"Win rate: {perf['win_rate']:.1%}")
print(f"Total PnL: ${perf['total_pnl']:.2f}")
```

### Advanced Client Features

```python
import time
from typing import Callable

class N1V1Monitor(N1V1Client):
    """Advanced client with monitoring capabilities."""

    def monitor_signals(self, callback: Callable, interval: int = 60):
        """Monitor signals with callback function."""
        while True:
            try:
                signals = self.get_signals()
                if signals:
                    callback(signals)
            except Exception as e:
                print(f"Error monitoring signals: {e}")

            time.sleep(interval)

    def monitor_performance(self, callback: Callable, interval: int = 300):
        """Monitor performance metrics."""
        last_pnl = 0

        while True:
            try:
                perf = self.get_performance()
                current_pnl = perf['total_pnl']

                if current_pnl != last_pnl:
                    callback(perf, current_pnl - last_pnl)
                    last_pnl = current_pnl

            except Exception as e:
                print(f"Error monitoring performance: {e}")

            time.sleep(interval)

# Usage
def signal_callback(signals):
    """Handle new signals."""
    for signal in signals:
        print(f"New signal: {signal['strategy']} - {signal['signal_type']}")

def performance_callback(perf, pnl_change):
    """Handle performance updates."""
    print(f"PnL change: ${pnl_change:.2f}")
    print(f"Current win rate: {perf['win_rate']:.1%}")

monitor = N1V1Monitor(api_key="your_api_key")

# Start monitoring (run in background threads)
import threading

signal_thread = threading.Thread(
    target=monitor.monitor_signals,
    args=(signal_callback,),
    daemon=True
)
signal_thread.start()

perf_thread = threading.Thread(
    target=monitor.monitor_performance,
    args=(performance_callback,),
    daemon=True
)
perf_thread.start()
```

## Error Handling

### Common Error Responses

```python
# Authentication error
{
  "error": {
    "code": 401,
    "message": "Invalid API key",
    "details": {
      "endpoint": "/api/v1/status"
    }
  }
}

# Bot not available
{
  "error": {
    "code": 503,
    "message": "Bot engine not available",
    "details": {
      "path": "/api/v1/status",
      "method": "GET"
    }
  }
}

# Rate limit exceeded
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded: 60 per 1 minute(s)",
    "details": {
      "limit": 60,
      "window": "1 minute",
      "endpoint": "/api/v1/status"
    }
  }
}
```

### Python Error Handling

```python
import requests
from requests.exceptions import RequestException

def safe_api_call(func, *args, **kwargs):
    """Wrapper for safe API calls with error handling."""
    try:
        return func(*args, **kwargs)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("Authentication failed. Check API key.")
        elif e.response.status_code == 429:
            print("Rate limit exceeded. Waiting...")
            time.sleep(60)  # Wait before retry
        elif e.response.status_code == 503:
            print("Service unavailable. Bot may not be running.")
        else:
            print(f"HTTP error: {e.response.status_code}")
        return None
    except requests.exceptions.ConnectionError:
        print("Connection failed. Check if N1V1 is running.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage
client = N1V1Client(api_key="your_api_key")
status = safe_api_call(client.get_status)
if status:
    print(f"Bot status: {status}")
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Default limit**: 60 requests per minute
- **Headers**: Rate limit information is included in responses
- **Exempt endpoints**: `/health`, `/ready`, `/metrics`, `/`

```bash
# Check rate limit headers
curl -v -H "Authorization: Bearer $API_KEY" \
     http://localhost:8000/api/v1/status

# Response headers include:
# X-RateLimit-Limit: 60
# X-RateLimit-Remaining: 59
# X-RateLimit-Reset: 1632000000
```

## WebSocket Support (Future)

For real-time updates, WebSocket support is planned:

```python
# Future WebSocket example
import websocket
import json

def on_message(ws, message):
    """Handle WebSocket messages."""
    data = json.loads(message)
    if data['type'] == 'signal':
        print(f"New signal: {data['signal']}")
    elif data['type'] == 'order':
        print(f"New order: {data['order']}")

ws = websocket.WebSocketApp(
    "ws://localhost:8000/ws",
    on_message=on_message
)
ws.run_forever()
```

## Integration Examples

### Trading Dashboard

```python
import dash
from dash import html, dcc
import plotly.graph_objects as go
from N1V1Client import N1V1Client

app = dash.Dash(__name__)
client = N1V1Client(api_key="your_api_key")

app.layout = html.Div([
    html.H1('N1V1 Trading Dashboard'),
    dcc.Graph(id='equity-chart'),
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0)
])

@app.callback(
    dash.dependencies.Output('equity-chart', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_equity_chart(n):
    """Update equity chart."""
    timestamps, equity = client.get_equity_curve()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=equity,
        mode='lines',
        name='Equity'
    ))

    fig.update_layout(
        title='Portfolio Equity',
        xaxis_title='Time',
        yaxis_title='Equity ($)'
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
```

### Alert System

```python
import smtplib
from email.mime.text import MIMEText

def send_alert(subject: str, message: str, email_config: dict):
    """Send email alert."""
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = email_config['from']
    msg['To'] = email_config['to']

    with smtplib.SMTP(email_config['smtp_server']) as server:
        server.login(email_config['username'], email_config['password'])
        server.send_message(msg)

def monitor_and_alert(api_key: str, email_config: dict):
    """Monitor bot and send alerts."""
    client = N1V1Client(api_key=api_key)

    while True:
        try:
            status = client.get_status()
            performance = client.get_performance()

            # Alert on high drawdown
            if performance['max_drawdown'] > 0.1:  # 10% drawdown
                send_alert(
                    "High Drawdown Alert",
                    f"Current drawdown: {performance['max_drawdown']:.1%}",
                    email_config
                )

            # Alert on low win rate
            if performance['win_rate'] < 0.5:
                send_alert(
                    "Low Win Rate Alert",
                    f"Current win rate: {performance['win_rate']:.1%}",
                    email_config
                )

        except Exception as e:
            send_alert("Monitoring Error", str(e), email_config)

        time.sleep(300)  # Check every 5 minutes

# Usage
email_config = {
    'smtp_server': 'smtp.gmail.com',
    'username': 'your_email@gmail.com',
    'password': 'your_password',
    'from': 'your_email@gmail.com',
    'to': 'alerts@yourcompany.com'
}

monitor_and_alert("your_api_key", email_config)
```

## Best Practices

### Security
1. **Always use HTTPS** in production
2. **Rotate API keys** regularly
3. **Use environment variables** for sensitive data
4. **Implement IP whitelisting** for production
5. **Monitor API usage** for anomalies

### Performance
1. **Cache responses** when possible
2. **Use WebSocket** for real-time data (when available)
3. **Batch requests** to reduce API calls
4. **Handle rate limits** gracefully
5. **Implement retry logic** with exponential backoff

### Error Handling
1. **Check HTTP status codes** before processing responses
2. **Implement comprehensive error handling** for all API calls
3. **Log errors** for debugging and monitoring
4. **Provide user-friendly error messages**
5. **Handle network timeouts** appropriately

### Monitoring
1. **Monitor API response times** and error rates
2. **Set up alerts** for API failures
3. **Track usage patterns** for capacity planning
4. **Log all API requests** for security auditing
5. **Implement health checks** in your applications

## API Reference

### Endpoints Summary

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/v1/status` | Get bot status | Yes |
| POST | `/api/v1/pause` | Pause bot | Yes |
| POST | `/api/v1/resume` | Resume bot | Yes |
| GET | `/api/v1/signals` | Get recent signals | Yes |
| GET | `/api/v1/orders` | Get recent orders | Yes |
| GET | `/api/v1/performance` | Get performance metrics | Yes |
| GET | `/api/v1/equity` | Get equity curve | Yes |
| GET | `/health` | Health check | No |
| GET | `/ready` | Readiness check | No |
| GET | `/metrics` | Prometheus metrics | No |

### Response Codes

- **200**: Success
- **401**: Authentication failed
- **403**: Forbidden
- **429**: Rate limit exceeded
- **500**: Internal server error
- **503**: Service unavailable

## Support

For API-related issues or questions:

- **API Documentation**: http://localhost:8000/docs (when running)
- **GitHub Issues**: Bug reports and feature requests
- **Community**: Discord channel for API discussions
- **Professional Support**: Enterprise API support available

---

**Note**: All examples assume N1V1 is running on `http://localhost:8000`. Adjust the base URL for your deployment environment.
