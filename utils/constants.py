"""
Constants - Application-wide constants and configuration values.

Centralized location for all hardcoded values, magic numbers,
and configuration constants used throughout the application.
"""

from pathlib import Path

# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "historical_data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
TEST_DATA_DIR = PROJECT_ROOT / "test_historical_data"

# File paths
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config.json"
DEFAULT_LOG_FILE = LOGS_DIR / "crypto_bot.log"
TRADES_CSV_FILE = LOGS_DIR / "trades.csv"
ERROR_LOG_FILE = LOGS_DIR / "errors.log"

# ============================================================================
# TRADING CONSTANTS
# ============================================================================

# Timeframes
TIMEFRAMES = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}

# Trading modes
TRADING_MODES = ["paper", "live", "backtest"]

# Order types
ORDER_TYPES = ["market", "limit", "stop", "stop_limit", "trailing_stop"]

# Order sides
ORDER_SIDES = ["buy", "sell"]

# Order statuses
ORDER_STATUSES = ["open", "filled", "cancelled", "rejected", "expired"]

# ============================================================================
# TECHNICAL ANALYSIS CONSTANTS
# ============================================================================

# RSI parameters
RSI_DEFAULT_PERIOD = 14
RSI_OVERBOUGHT_LEVEL = 70
RSI_OVERSOLD_LEVEL = 30

# Moving average periods
MA_PERIODS = [10, 20, 50, 100, 200]

# Bollinger Bands
BB_DEFAULT_PERIOD = 20
BB_DEFAULT_STD_DEV = 2.0

# MACD parameters
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# Stochastic parameters
STOCHASTIC_K_PERIOD = 14
STOCHASTIC_D_PERIOD = 3
STOCHASTIC_OVERBOUGHT = 80
STOCHASTIC_OVERSOLD = 20

# ============================================================================
# RISK MANAGEMENT CONSTANTS
# ============================================================================

# Position sizing
DEFAULT_POSITION_SIZE = 0.1  # 10% of portfolio
MAX_POSITION_SIZE = 0.3  # 30% of portfolio
MIN_POSITION_SIZE = 0.01  # 1% of portfolio

# Stop loss defaults
DEFAULT_STOP_LOSS_PCT = 0.05  # 5%
DEFAULT_TAKE_PROFIT_PCT = 0.10  # 10%

# Risk per trade
DEFAULT_RISK_PER_TRADE = 0.02  # 2% of portfolio per trade
MAX_DAILY_DRAWDOWN = 0.10  # 10% max daily drawdown

# ============================================================================
# PERFORMANCE METRICS CONSTANTS
# ============================================================================

# Sharpe ratio calculation
SHARPE_ANNUALIZATION_FACTOR = 252  # Trading days per year
RISK_FREE_RATE = 0.02  # 2% risk-free rate

# Maximum drawdown calculation
MAX_DRAWDOWN_LOOKBACK = 252  # 1 year lookback

# Win rate calculation
MIN_TRADES_FOR_STATS = 10

# ============================================================================
# DATA FETCHING CONSTANTS
# ============================================================================

# Rate limiting
DEFAULT_RATE_LIMIT = 10  # requests per second
BURST_RATE_LIMIT = 50  # burst requests
RATE_LIMIT_BACKOFF = 60  # seconds to wait after rate limit

# Data caching
CACHE_TTL_SECONDS = 300  # 5 minutes
MAX_CACHE_SIZE = 1000  # maximum cached items
CACHE_CLEANUP_INTERVAL = 600  # 10 minutes

# Historical data limits
DEFAULT_HISTORICAL_LIMIT = 1000
MAX_HISTORICAL_LIMIT = 10000
MIN_HISTORICAL_LIMIT = 100

# ============================================================================
# API AND NETWORK CONSTANTS
# ============================================================================

# HTTP timeouts
DEFAULT_TIMEOUT = 30000  # 30 seconds
LONG_TIMEOUT = 60000  # 1 minute
SHORT_TIMEOUT = 10000  # 10 seconds

# Retry configuration
DEFAULT_MAX_RETRIES = 3
API_MAX_RETRIES = 5
NETWORK_MAX_RETRIES = 10

# Backoff configuration
DEFAULT_BACKOFF_FACTOR = 2.0
MAX_BACKOFF_DELAY = 300  # 5 minutes

# ============================================================================
# LOGGING CONSTANTS
# ============================================================================

# Log levels
LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

# Log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
)

# Log rotation
DEFAULT_MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_LOG_BACKUP_COUNT = 5

# ============================================================================
# SECURITY CONSTANTS
# ============================================================================

# Password requirements
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128

# Session management
SESSION_TIMEOUT = 3600  # 1 hour
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = 900  # 15 minutes

# API key validation
API_KEY_MIN_LENGTH = 20
API_KEY_MAX_LENGTH = 128

# ============================================================================
# MACHINE LEARNING CONSTANTS
# ============================================================================

# Model parameters
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
MIN_TRAINING_SAMPLES = 1000
MAX_TRAINING_SAMPLES = 100000

# Feature engineering
MAX_FEATURES = 100
CORRELATION_THRESHOLD = 0.95

# Model validation
CROSS_VALIDATION_FOLDS = 5
TEST_SIZE_RATIO = 0.2

# ============================================================================
# MONITORING AND ALERTS CONSTANTS
# ============================================================================

# Health check intervals
HEALTH_CHECK_INTERVAL = 60  # seconds
COMPONENT_TIMEOUT = 30  # seconds

# Alert thresholds
CPU_WARNING_THRESHOLD = 80  # percent
CPU_CRITICAL_THRESHOLD = 95  # percent

MEMORY_WARNING_THRESHOLD = 500  # MB
MEMORY_CRITICAL_THRESHOLD = 1000  # MB

DISK_WARNING_THRESHOLD = 80  # percent
DISK_CRITICAL_THRESHOLD = 95  # percent

# ============================================================================
# BACKTESTING CONSTANTS
# ============================================================================

# Commission and fees
DEFAULT_COMMISSION = 0.001  # 0.1%
DEFAULT_SPREAD = 0.0005  # 0.05%

# Slippage models
SLIPPAGE_MODELS = ["fixed", "percentage", "volume_based"]

# Walk-forward optimization
DEFAULT_TRAIN_WINDOW = 90  # days
DEFAULT_TEST_WINDOW = 30  # days
MIN_OBSERVATIONS = 1000

# ============================================================================
# EXCHANGE-SPECIFIC CONSTANTS
# ============================================================================

# Binance
BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_TESTNET_URL = "https://testnet.binance.vision"

# KuCoin
KUCOIN_BASE_URL = "https://api.kucoin.com"
KUCOIN_SANDBOX_URL = "https://api-sandbox.kucoin.com"

# Common exchange settings
DEFAULT_EXCHANGE_TIMEOUT = 30000  # 30 seconds
DEFAULT_EXCHANGE_RATE_LIMIT = 10  # requests per second

# ============================================================================
# CONFIGURATION SCHEMA CONSTANTS
# ============================================================================

# Configuration validation
REQUIRED_CONFIG_KEYS = ["environment", "exchange", "trading", "risk_management"]

# Environment validation
VALID_ENVIRONMENTS = ["development", "staging", "production"]
VALID_MODES = ["paper", "live", "backtest"]

# ============================================================================
# ERROR HANDLING CONSTANTS
# ============================================================================

# Error severity mapping
ERROR_SEVERITY_LEVELS = {"low": 1, "medium": 2, "high": 3, "critical": 4}

# Error categories
ERROR_CATEGORIES = [
    "network",
    "data",
    "configuration",
    "security",
    "performance",
    "business_logic",
    "system",
    "external_api",
]

# ============================================================================
# UTILITY CONSTANTS
# ============================================================================

# Time constants
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
MILLISECONDS_PER_SECOND = 1000

# Size constants
KB = 1024
MB = 1024 * KB
GB = 1024 * MB

# Common limits
MAX_LIST_SIZE = 10000
MAX_DICT_SIZE = 1000
MAX_STRING_LENGTH = 1000000

# ============================================================================
# DEFAULT CONFIGURATION TEMPLATES
# ============================================================================

DEFAULT_TRADING_CONFIG = {
    "initial_balance": 1000.0,
    "max_concurrent_trades": 3,
    "slippage": 0.001,
    "order_timeout": 60,
    "trade_fee": 0.001,
    "portfolio_mode": False,
}

DEFAULT_RISK_CONFIG = {
    "stop_loss": DEFAULT_STOP_LOSS_PCT,
    "take_profit": DEFAULT_TAKE_PROFIT_PCT,
    "position_size": DEFAULT_POSITION_SIZE,
    "max_position_size": MAX_POSITION_SIZE,
    "risk_reward_ratio": 2.0,
    "max_daily_drawdown": MAX_DAILY_DRAWDOWN,
}

DEFAULT_LOGGING_CONFIG = {
    "level": "INFO",
    "file_logging": True,
    "console_logging": True,
    "max_size": DEFAULT_MAX_LOG_SIZE,
    "backup_count": DEFAULT_LOG_BACKUP_COUNT,
}

# ============================================================================
# VALIDATION PATTERNS
# ============================================================================

# Symbol validation patterns
VALID_SYMBOL_PATTERN = r"^[A-Z0-9]+/[A-Z0-9]+$"
VALID_EXCHANGE_PATTERN = r"^[a-zA-Z0-9_-]+$"

# Email validation
EMAIL_PATTERN = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

# URL validation
URL_PATTERN = r"^https?://[^\s/$.?#].[^\s]*$"

# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Experimental features
EXPERIMENTAL_FEATURES = {
    "advanced_ml": False,
    "multi_timeframe_analysis": True,
    "dynamic_position_sizing": False,
    "auto_optimization": False,
}

# Beta features
BETA_FEATURES = {
    "portfolio_optimization": False,
    "alternative_data": False,
    "social_sentiment": False,
}

# ============================================================================
# METRICS AND MONITORING
# ============================================================================

# Metrics collection
METRICS_ENABLED = True
METRICS_INTERVAL = 60  # seconds
METRICS_RETENTION = 7  # days

# Dashboard configuration
DASHBOARD_UPDATE_INTERVAL = 30  # seconds
DASHBOARD_MAX_POINTS = 1000

# ============================================================================
# EXTERNAL SERVICE CONFIGURATION
# ============================================================================

# Database connection
DEFAULT_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_bot",
    "pool_size": 10,
    "connection_timeout": 30,
}

# Redis configuration
DEFAULT_REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": None,
    "socket_timeout": 5,
}

# ============================================================================
# DEPRECATED CONSTANTS (TO BE REMOVED)
# ============================================================================

# Mark deprecated constants for future removal
DEPRECATED_CONSTANTS = [
    "OLD_TIMEFRAME_MAPPING",  # Use TIMEFRAMES instead
    "LEGACY_ORDER_TYPES",  # Use ORDER_TYPES instead
]

# ============================================================================
# TYPE HINTS AND VALIDATION
# ============================================================================

# Type definitions for better code documentation
Symbol = str
Timeframe = str
OrderId = str
TradeId = str

# Validation schemas
CONFIG_SCHEMA_VERSION = "1.0.0"

# ============================================================================
# ENVIRONMENT-SPECIFIC OVERRIDES
# ============================================================================

# Environment-specific configurations
ENVIRONMENT_OVERRIDES = {
    "development": {"debug": True, "log_level": "DEBUG", "cache_enabled": False},
    "staging": {"debug": False, "log_level": "INFO", "cache_enabled": True},
    "production": {"debug": False, "log_level": "WARNING", "cache_enabled": True},
}
