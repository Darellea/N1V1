"""
Configuration Documentation Generator.

Generates comprehensive documentation for configuration files,
validation schemas, and usage examples.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from utils.constants import (
    DEFAULT_LOGGING_CONFIG,
    DEFAULT_RISK_CONFIG,
    DEFAULT_TRADING_CONFIG,
    ORDER_TYPES,
    TIMEFRAMES,
    VALID_MODES,
)


class ConfigDocumentationGenerator:
    """
    Generates comprehensive documentation for configuration files
    and validation schemas.
    """

    def __init__(self):
        self.templates = {
            "trading": DEFAULT_TRADING_CONFIG,
            "risk": DEFAULT_RISK_CONFIG,
            "logging": DEFAULT_LOGGING_CONFIG,
        }

    def generate_full_documentation(self, output_dir: str = "docs") -> None:
        """Generate complete configuration documentation."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate main configuration guide
        self._generate_main_config_guide(output_path)

        # Generate schema documentation
        self._generate_schema_docs(output_path)

        # Generate examples
        self._generate_examples(output_path)

        # Generate validation rules
        self._generate_validation_docs(output_path)

        print(f"Configuration documentation generated in {output_path}")

    def _generate_main_config_guide(self, output_dir: Path) -> None:
        """Generate main configuration guide."""
        content = f"""# N1V1 Trading Framework Configuration Guide

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This document provides comprehensive guidance for configuring the N1V1 Trading Framework.

## Configuration Structure

### Environment Configuration

```json
{{
  "environment": {{
    "mode": "paper|live|backtest",
    "debug": true|false,
    "log_level": "DEBUG|INFO|WARNING|ERROR"
  }}
}}
```

**Fields:**
- `mode`: Trading mode ({', '.join(VALID_MODES)})
- `debug`: Enable debug logging
- `log_level`: Logging verbosity level

### Exchange Configuration

```json
{{
  "exchange": {{
    "name": "binance|kucoin",
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "api_passphrase": "your_passphrase",
    "sandbox": true|false,
    "timeout": 30000,
    "rate_limit": 10
  }}
}}
```

**Security Note:** API credentials should be provided via environment variables in production:
- `CRYPTOBOT_EXCHANGE_API_KEY`
- `CRYPTOBOT_EXCHANGE_API_SECRET`
- `CRYPTOBOT_EXCHANGE_API_PASSPHRASE`

### Trading Configuration

```json
{json.dumps(DEFAULT_TRADING_CONFIG, indent=2)}
```

### Risk Management Configuration

```json
{json.dumps(DEFAULT_RISK_CONFIG, indent=2)}
```

### Logging Configuration

```json
{json.dumps(DEFAULT_LOGGING_CONFIG, indent=2)}
```

## Timeframes

Supported timeframes: {', '.join(TIMEFRAMES.keys())}

## Order Types

Supported order types: {', '.join(ORDER_TYPES)}

## Environment Variables

### Exchange Configuration
- `CRYPTOBOT_EXCHANGE_API_KEY`: Exchange API key
- `CRYPTOBOT_EXCHANGE_API_SECRET`: Exchange API secret
- `CRYPTOBOT_EXCHANGE_API_PASSPHRASE`: Exchange API passphrase (if required)

### Discord Notifications
- `CRYPTOBOT_NOTIFICATIONS_DISCORD_WEBHOOK_URL`: Discord webhook URL
- `CRYPTOBOT_NOTIFICATIONS_DISCORD_BOT_TOKEN`: Discord bot token
- `CRYPTOBOT_NOTIFICATIONS_DISCORD_CHANNEL_ID`: Discord channel ID

### Database Configuration
- `CRYPTOBOT_DATABASE_HOST`: Database host
- `CRYPTOBOT_DATABASE_PORT`: Database port
- `CRYPTOBOT_DATABASE_NAME`: Database name
- `CRYPTOBOT_DATABASE_USER`: Database username
- `CRYPTOBOT_DATABASE_PASSWORD`: Database password

## Configuration Validation

The framework validates configuration against a JSON schema that ensures:
- Required fields are present
- Data types are correct
- Values are within acceptable ranges
- Environment-specific constraints are met

## Best Practices

### Security
1. Never commit API keys to version control
2. Use environment variables for sensitive data
3. Rotate API keys regularly
4. Use sandbox mode for testing

### Performance
1. Adjust rate limits based on exchange requirements
2. Configure appropriate cache settings
3. Set reasonable timeout values
4. Monitor memory usage in production

### Risk Management
1. Start with conservative position sizes
2. Set appropriate stop loss levels
3. Configure maximum drawdown limits
4. Test strategies thoroughly in paper mode

## Troubleshooting

### Common Issues

1. **Invalid API Key**: Check environment variables and API key validity
2. **Rate Limit Exceeded**: Reduce request frequency or increase rate limit
3. **Configuration Validation Error**: Check JSON syntax and required fields
4. **Memory Issues**: Adjust cache settings or increase system memory

### Debug Mode

Enable debug mode for detailed logging:

```json
{{
  "environment": {{
    "debug": true,
    "log_level": "DEBUG"
  }}
}}
```

## Support

For additional help, refer to:
- Framework documentation
- Exchange API documentation
- Community forums
"""
        with open(output_dir / "configuration_guide.md", "w") as f:
            f.write(content)

    def _generate_schema_docs(self, output_dir: Path) -> None:
        """Generate JSON schema documentation."""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "N1V1 Trading Framework Configuration",
            "version": "1.0.0",
            "type": "object",
            "required": ["environment", "exchange", "trading", "risk_management"],
            "properties": {
                "environment": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": VALID_MODES,
                            "description": "Trading mode",
                        },
                        "debug": {
                            "type": "boolean",
                            "description": "Enable debug logging",
                        },
                        "log_level": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                            "description": "Logging level",
                        },
                    },
                    "required": ["mode"],
                },
                "exchange": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Exchange name (binance, kucoin, etc.)",
                        },
                        "api_key": {
                            "type": "string",
                            "description": "Exchange API key",
                        },
                        "api_secret": {
                            "type": "string",
                            "description": "Exchange API secret",
                        },
                        "api_passphrase": {
                            "type": "string",
                            "description": "Exchange API passphrase (if required)",
                        },
                        "sandbox": {
                            "type": "boolean",
                            "description": "Use sandbox/testnet",
                        },
                        "timeout": {
                            "type": "number",
                            "minimum": 1000,
                            "maximum": 120000,
                            "description": "Request timeout in milliseconds",
                        },
                        "rate_limit": {
                            "type": "number",
                            "minimum": 1,
                            "maximum": 100,
                            "description": "Requests per second",
                        },
                    },
                    "required": ["name"],
                },
                "trading": {
                    "type": "object",
                    "properties": {
                        "initial_balance": {
                            "type": "number",
                            "minimum": 0,
                            "description": "Starting balance",
                        },
                        "max_concurrent_trades": {
                            "type": "number",
                            "minimum": 1,
                            "maximum": 50,
                            "description": "Maximum concurrent trades",
                        },
                        "slippage": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 0.1,
                            "description": "Slippage tolerance",
                        },
                        "order_timeout": {
                            "type": "number",
                            "minimum": 10,
                            "maximum": 3600,
                            "description": "Order timeout in seconds",
                        },
                        "trade_fee": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 0.01,
                            "description": "Trading fee",
                        },
                        "portfolio_mode": {
                            "type": "boolean",
                            "description": "Enable portfolio trading",
                        },
                    },
                },
                "risk_management": {
                    "type": "object",
                    "properties": {
                        "stop_loss": {
                            "type": "number",
                            "minimum": 0.001,
                            "maximum": 0.5,
                            "description": "Stop loss percentage",
                        },
                        "take_profit": {
                            "type": "number",
                            "minimum": 0.001,
                            "maximum": 1.0,
                            "description": "Take profit percentage",
                        },
                        "position_size": {
                            "type": "number",
                            "minimum": 0.01,
                            "maximum": 1.0,
                            "description": "Position size percentage",
                        },
                        "max_position_size": {
                            "type": "number",
                            "minimum": 0.01,
                            "maximum": 1.0,
                            "description": "Maximum position size",
                        },
                        "risk_reward_ratio": {
                            "type": "number",
                            "minimum": 1.0,
                            "maximum": 10.0,
                            "description": "Risk-reward ratio",
                        },
                        "max_daily_drawdown": {
                            "type": "number",
                            "minimum": 0.01,
                            "maximum": 1.0,
                            "description": "Maximum daily drawdown",
                        },
                    },
                },
            },
        }

        with open(output_dir / "configuration_schema.json", "w") as f:
            json.dump(schema, f, indent=2)

    def _generate_examples(self, output_dir: Path) -> None:
        """Generate configuration examples."""
        examples = {
            "paper_trading": {
                "environment": {"mode": "paper", "debug": True, "log_level": "INFO"},
                "exchange": {
                    "name": "binance",
                    "sandbox": True,
                    "timeout": 30000,
                    "rate_limit": 10,
                },
                "trading": DEFAULT_TRADING_CONFIG,
                "risk_management": DEFAULT_RISK_CONFIG,
                "logging": DEFAULT_LOGGING_CONFIG,
            },
            "live_trading": {
                "environment": {"mode": "live", "debug": False, "log_level": "WARNING"},
                "exchange": {
                    "name": "binance",
                    "sandbox": False,
                    "timeout": 30000,
                    "rate_limit": 5,
                },
                "trading": {
                    **DEFAULT_TRADING_CONFIG,
                    "max_concurrent_trades": 1,
                    "position_size": 0.05,
                },
                "risk_management": {
                    **DEFAULT_RISK_CONFIG,
                    "stop_loss": 0.03,
                    "take_profit": 0.06,
                },
                "logging": DEFAULT_LOGGING_CONFIG,
            },
            "backtesting": {
                "environment": {
                    "mode": "backtest",
                    "debug": False,
                    "log_level": "INFO",
                },
                "exchange": {
                    "name": "binance",
                    "sandbox": True,
                    "timeout": 30000,
                    "rate_limit": 50,
                },
                "trading": {**DEFAULT_TRADING_CONFIG, "initial_balance": 10000.0},
                "risk_management": DEFAULT_RISK_CONFIG,
                "backtesting": {
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "timeframe": "1h",
                    "commission": 0.001,
                },
                "logging": DEFAULT_LOGGING_CONFIG,
            },
        }

        for name, config in examples.items():
            with open(output_dir / f"config_example_{name}.json", "w") as f:
                json.dump(config, f, indent=2)

    def _generate_validation_docs(self, output_dir: Path) -> None:
        """Generate validation rules documentation."""
        validation_rules = """
# Configuration Validation Rules

## Required Fields

The following fields are required in all configurations:

- `environment.mode`: Trading mode (paper/live/backtest)
- `exchange.name`: Exchange name (binance/kucoin/etc.)

## Data Type Validation

### Numeric Fields
- `trading.initial_balance`: Must be positive number
- `trading.max_concurrent_trades`: Integer between 1-50
- `risk_management.stop_loss`: Number between 0.001-0.5
- `risk_management.take_profit`: Number between 0.001-1.0
- `exchange.timeout`: Integer between 1000-120000 (milliseconds)
- `exchange.rate_limit`: Integer between 1-100 (requests/second)

### String Fields
- `environment.mode`: Must be one of: paper, live, backtest
- `environment.log_level`: Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `exchange.name`: Valid exchange identifier

### Boolean Fields
- `environment.debug`: true/false
- `exchange.sandbox`: true/false
- `trading.portfolio_mode`: true/false

## Environment-Specific Validation

### Live Mode Requirements
- Exchange credentials must be provided (via config or environment variables)
- `exchange.sandbox` should be false
- Rate limits should be conservative (≤10 requests/second)

### Paper Mode Recommendations
- Can use test/sandbox credentials
- Higher rate limits acceptable
- Debug logging can be enabled

### Backtest Mode
- Exchange credentials optional
- Historical data source required
- Commission and slippage settings important

## Security Validation

### API Keys
- Minimum length: 20 characters
- Maximum length: 128 characters
- Must contain valid characters only

### Environment Variables
- Sensitive data should use environment variables
- Config file should not contain real credentials
- Environment variable names follow CRYPTOBOT_* pattern

## Cross-Field Validation

### Risk Management
- `take_profit` should be greater than `stop_loss`
- `max_position_size` should be ≥ `position_size`
- `risk_reward_ratio` should be ≥ 1.0

### Trading Parameters
- `max_concurrent_trades` affects position sizing
- `order_timeout` should be reasonable for exchange
- `slippage` tolerance affects execution quality

## Custom Validation Rules

### Portfolio Mode
When `trading.portfolio_mode` is true:
- `exchange.markets` should be provided
- Multiple symbols should be configured
- Position sizing should account for diversification

### Multi-Timeframe Analysis
When multi-timeframe features are enabled:
- Timeframe hierarchy should be logical
- Data availability should be verified
- Memory usage should be monitored

## Error Messages

Validation errors provide specific guidance:
- Missing required fields
- Invalid data types
- Out-of-range values
- Inconsistent configurations
- Security violations

## Automated Validation

The framework automatically validates configuration on:
- Application startup
- Configuration reload
- Environment changes
- Before critical operations
"""

        with open(output_dir / "validation_rules.md", "w") as f:
            f.write(validation_rules)


def main():
    """Command-line interface for configuration documentation generation."""
    parser = argparse.ArgumentParser(description="Generate configuration documentation")
    parser.add_argument(
        "--output", "-o", default="docs", help="Output directory for documentation"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json", "yaml"],
        default="markdown",
        help="Documentation format",
    )

    args = parser.parse_args()

    generator = ConfigDocumentationGenerator()
    generator.generate_full_documentation(args.output)


if __name__ == "__main__":
    main()
