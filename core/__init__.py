"""
Core module initialization with secure logging configuration.

This module sets up the secure logging system for all core components,
ensuring sensitive information is properly sanitized and structured.
"""

import logging
import logging.config
import os
from typing import Any, Dict, Optional

from .logging_utils import (
    LogSanitizer,
    LogSensitivity,
    StructuredLogger,
    create_secure_logger_config,
    get_structured_logger,
    set_global_log_sensitivity,
)

# Global configuration for core logging
_CORE_LOG_CONFIG = {
    "sensitivity": LogSensitivity.SECURE,  # Default to secure
    "level": "INFO",
    "file_logging": True,
    "console": True,
    "log_file": "logs/core.log",
}


def get_core_logger(name: str) -> Any:
    """
    Get a secure structured logger for core modules.

    Args:
        name: Logger name (will be prefixed with 'core.')

    Returns:
        StructuredLogger instance
    """
    return get_structured_logger(f"core.{name}", _CORE_LOG_CONFIG["sensitivity"])


def configure_core_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Configure secure logging for all core modules.

    Args:
        config: Optional configuration override
    """
    global _CORE_LOG_CONFIG

    if config:
        _CORE_LOG_CONFIG.update(config)

    # Set sensitivity from environment or config
    sensitivity_str = os.getenv(
        "LOG_SENSITIVITY", _CORE_LOG_CONFIG.get("sensitivity", "secure")
    )
    try:
        if isinstance(sensitivity_str, str):
            sensitivity = LogSensitivity[sensitivity_str.upper()]
        else:
            sensitivity = sensitivity_str
    except (ValueError, KeyError):
        sensitivity = LogSensitivity.SECURE

    # Apply global sensitivity
    set_global_log_sensitivity(sensitivity)
    _CORE_LOG_CONFIG["sensitivity"] = sensitivity

    # Configure logging
    log_config = create_secure_logger_config(
        log_level=_CORE_LOG_CONFIG.get("level", "INFO"),
        sensitivity=sensitivity,
        log_file=_CORE_LOG_CONFIG.get("log_file"),
    )

    # Apply configuration
    logging.config.dictConfig(log_config)

    # Log configuration
    logger = get_structured_logger("core.init")
    logger.info(
        "Core logging configured",
        sensitivity=sensitivity.value,
        level=_CORE_LOG_CONFIG.get("level"),
        file_logging=_CORE_LOG_CONFIG.get("file_logging"),
    )


# Initialize logging on import
configure_core_logging()

# Import monitoring components
from .alert_rules_manager import AlertRulesManager
from .dashboard_manager import DashboardManager
from .metrics_collector import (
    MetricSample,
    MetricsCollector,
    MetricSeries,
    collect_exchange_metrics,
    collect_risk_metrics,
    collect_strategy_metrics,
    collect_trading_metrics,
    get_metrics_collector,
)
from .metrics_endpoint import MetricsEndpoint

__all__ = [
    "LogSanitizer",
    "StructuredLogger",
    "LogSensitivity",
    "get_structured_logger",
    "set_global_log_sensitivity",
    "create_secure_logger_config",
    "AlertRulesManager",
    "DashboardManager",
    "MetricsCollector",
    "MetricSample",
    "MetricSeries",
    "get_metrics_collector",
    "collect_trading_metrics",
    "collect_risk_metrics",
    "collect_strategy_metrics",
    "collect_exchange_metrics",
    "MetricsEndpoint",
    "get_core_logger",
    "configure_core_logging",
]
