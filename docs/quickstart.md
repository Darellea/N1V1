# N1V1 Quickstart Guide

Get started with N1V1's paper-trade demo in under 10 minutes using Docker Compose.

## Prerequisites

- **Docker**: Version 20.10+ with Docker Compose
- **Python**: 3.8+ (optional, for local development)
- **Git**: For cloning the repository
- **4GB RAM**: Minimum for running all services

## Installation

### Option 1: Docker Compose (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Darellea/N1V1.git
   cd N1V1
   ```

2. **Start the demo environment:**
   ```bash
   docker compose -f deploy/docker-compose.dev.yml up -d
   ```

3. **Verify installation:**
   ```bash
   docker compose -f deploy/docker-compose.dev.yml ps
   curl http://localhost:8000/health
   ```

### Option 2: Local Development Setup

1. **Clone and install dependencies:**
   ```bash
   git clone https://github.com/Darellea/N1V1.git
   cd N1V1
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Run the application:**
   ```bash
   python main.py --api --debug
   ```

3. **Verify installation:**
   ```bash
   curl http://localhost:8000/health
   ```

## � Quick Demo Setup

### 1. Clone and Navigate

```bash
git clone https://github.com/Darellea/N1V1.git
cd N1V1
```

### 2. Launch Paper-Trade Demo

```bash
# Start all services (trading engine, API, monitoring)
docker compose -f deploy/docker-compose.dev.yml up -d

# View logs
docker compose -f deploy/docker-compose.dev.yml logs -f
```

### 3. Access the Demo

| Service | URL | Description |
|---------|-----|-------------|
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Grafana** | http://localhost:3000 | Monitoring dashboards (admin/admin) |
| **Health Check** | http://localhost:8000/health | System health status |
| **Metrics** | http://localhost:8000/metrics | Prometheus metrics |

### 4. Verify Everything Works

```bash
# Check service health
curl http://localhost:8000/health

# Get bot status
curl http://localhost:8000/api/v1/status

# View recent signals
curl http://localhost:8000/api/v1/signals
```

## �📊 What You'll See

### Trading Dashboard
- **Real-time signals** from multiple strategies (EMA, RSI, MACD, etc.)
- **Paper-trade execution** with simulated orders
- **Performance metrics** (PnL, win rate, Sharpe ratio)
- **Risk monitoring** with circuit breaker status

### Monitoring Dashboards
- **System metrics**: CPU, memory, API response times
- **Trading metrics**: Signal confidence, order execution latency
- **Risk metrics**: Drawdown, exposure, position sizes
- **Strategy performance**: Individual strategy returns

## 🛠️ Development Setup (Optional)

For contributors who want to modify the code:

### Local Development Environment

```bash
# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests to verify setup
python tests/run_comprehensive_tests.py

# Start development server
python main.py --api --debug
```

### IDE Setup

1. **VS Code** (recommended):
   - Install Python extension
   - Configure interpreter: `Python 3.8+`
   - Enable linting: `ruff` or `flake8`

2. **PyCharm**:
   - Set project interpreter
   - Enable pytest integration
   - Configure code style (Black)

## 🧪 Testing Your Setup

### Run the Test Suite

```bash
# Unit tests only
pytest tests/ -k "unit" -v

# Integration tests
pytest tests/integration/ -v

# Full comprehensive test suite
python tests/run_comprehensive_tests.py

# Performance benchmarks
python scripts/run_profiling.py
```

### Test Coverage Requirements

- **Minimum coverage**: 95%
- **Core modules**: 100% coverage required
- **API endpoints**: Full integration test coverage

## 🔧 Configuration

### Basic Configuration

Create `config.json` for custom settings:

```json
{
  "exchange": {
    "name": "binance",
    "testnet": true
  },
  "risk_management": {
    "max_position_size": 0.02,
    "circuit_breaker_enabled": true
  },
  "strategies": {
    "active_strategies": ["ema_cross", "rsi", "macd"],
    "max_concurrent": 3
  }
}
```

### Environment Variables

```bash
# API Configuration
export API_KEY="your_secure_api_key"

# Monitoring
export PROMETHEUS_ENABLED=true
export GRAFANA_ENABLED=true

# Logging
export LOG_LEVEL=INFO
```

## 📈 Understanding the Demo

### Trading Strategies Included

1. **EMA Cross Strategy**: Trend following with exponential moving averages
2. **RSI Strategy**: Mean reversion using relative strength index
3. **MACD Strategy**: Momentum trading with MACD indicators
4. **Bollinger Bands**: Volatility-based mean reversion
5. **Stochastic Strategy**: Oscillator-based momentum trading

### Risk Management Features

- **Circuit Breaker**: Automatic trading suspension on losses
- **Position Sizing**: Risk-based position calculation
- **Drawdown Limits**: Maximum loss protection
- **Exposure Controls**: Portfolio concentration limits

### Monitoring & Observability

- **Prometheus Metrics**: Real-time performance tracking
- **Grafana Dashboards**: Visual monitoring interface
- **Health Checks**: Automated system health validation
- **Alerting**: Discord/Telegram notification integration

## 🚨 Troubleshooting

### Common Issues

#### Installation Issues
**Docker Not Found**
```bash
# Check if Docker is installed
docker --version

# Install Docker Desktop for your platform
# Windows/Mac: https://www.docker.com/products/docker-desktop
# Linux: https://docs.docker.com/engine/install/
```

**Python Dependencies Missing**
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Verify pytest installation
pytest --version
```

#### Configuration Issues
**Invalid Configuration File**
```bash
# Validate config.json syntax
python -c "import json; json.load(open('config.json'))"

# Check for required fields
python -c "import json; c=json.load(open('config.json')); print('exchange' in c)"
```

**Environment Variables Not Set**
```bash
# Check required environment variables
echo $API_KEY
echo $LOG_LEVEL

# Set missing variables
export API_KEY="your_api_key_here"
export LOG_LEVEL=INFO
```

#### Connection Issues
**Port Already in Use**
```bash
# Check what's using the port
lsof -i :8000

# Stop conflicting service or change port
docker compose -f deploy/docker-compose.dev.yml down
```

**API Returns 503**
```bash
# Check bot engine status
curl http://localhost:8000/api/v1/status

# Restart services
docker compose -f deploy/docker-compose.dev.yml restart
```

#### Permission Issues
**Docker Permission Denied**
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER

# Restart session or run:
newgrp docker
```

**File Permission Errors**
```bash
# Fix log directory permissions
chmod 755 logs/
chmod 644 logs/*.log

# Fix config file permissions
chmod 600 config.json
```

#### Dependency Issues
**Docker Compose Fails**
```bash
# Clean up and retry
docker compose -f deploy/docker-compose.dev.yml down -v
docker system prune -f
docker compose -f deploy/docker-compose.dev.yml up --build
```

**Memory Issues**
```bash
# Check Docker memory settings
docker system info | grep "Total Memory"

# Increase Docker memory allocation in Docker Desktop
# Or run fewer services
```

### Logs and Debugging

```bash
# View all service logs
docker compose -f deploy/docker-compose.dev.yml logs

# View specific service logs
docker compose -f deploy/docker-compose.dev.yml logs api

# Follow logs in real-time
docker compose -f deploy/docker-compose.dev.yml logs -f trading-engine
```

## 🎯 Next Steps

### For Users
1. **Explore Strategies**: Try different strategy combinations
2. **Customize Risk**: Adjust position sizes and risk parameters
3. **Monitor Performance**: Use Grafana dashboards for insights
4. **Paper Trade**: Test strategies with simulated trading

### For Developers
1. **Read Strategy Guide**: `docs/strategy_development.md`
2. **Explore API**: `docs/api_examples.md`
3. **Run Tests**: `python tests/run_comprehensive_tests.py`
4. **Contribute**: See `CONTRIBUTING.md` for guidelines

## 📚 Additional Resources

- **Strategy Development**: `docs/strategy_development.md`
- **ML Pipeline**: `docs/ml_onboarding.md`
- **API Reference**: `docs/api_examples.md`
- **Deployment**: `docs/deployment.md`
- **Security**: `docs/security.md`

## 🆘 Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides in `docs/`
- **Community**: Discord channel for discussions
- **Professional Support**: Enterprise support available

---

**Demo Environment Notes:**
- Uses testnet/paper trading mode
- Pre-configured with safe risk parameters
- Includes sample historical data
- All services run in Docker containers
- Data persists between container restarts
