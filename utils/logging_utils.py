"""
Standardized Logging Utilities for N1V1 Framework.

Provides centralized logging setup with consistent configuration,
structured logging with context propagation, and dynamic log level
management to eliminate duplication across the codebase.
"""

import json
import logging
import logging.handlers
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Union

import colorama
from colorama import Back, Fore, Style

from utils.config_factory import get_logging_config
from utils.constants import PROJECT_ROOT

# Initialize colorama for Windows support
colorama.init(autoreset=True)

logger = logging.getLogger(__name__)


@dataclass
class LogContext:
    """Structured logging context."""

    component: str
    operation: Optional[str] = None
    user_id: Optional[str] = None
    symbol: Optional[str] = None
    trade_id: Optional[str] = None
    correlation_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "component": self.component,
            "operation": self.operation,
            "user_id": self.user_id,
            "symbol": self.symbol,
            "trade_id": self.trade_id,
            "correlation_id": self.correlation_id,
            **self.additional_data,
        }


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter with consistent formatting.
    """

    LEVEL_COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.WHITE + Back.RED,
        "TRADE": Fore.MAGENTA,
        "PERF": Fore.BLUE,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and structured data."""
        # Add color based on level
        levelname = record.levelname
        color = self.LEVEL_COLORS.get(levelname, "")

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Format message
        message = super().format(record)

        # Add structured context if available
        if hasattr(record, "context") and record.context:
            context_str = json.dumps(record.context, default=str, indent=None)
            message = f"{message} | context={context_str}"

        return f"{color}{timestamp} - {record.name} - {levelname} - {message}{Style.RESET_ALL}"


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add structured context if available
        if hasattr(record, "context") and record.context:
            log_entry["context"] = record.context

        # Add any extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "context",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


class ContextAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically adds context to log records.
    """

    def __init__(self, logger: logging.Logger, context: Optional[LogContext] = None):
        super().__init__(logger, {})
        self.context = context or LogContext(component="unknown")

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log record to add context."""
        # Add context to the record
        kwargs.setdefault("extra", {})
        kwargs["extra"]["context"] = self.context.to_dict()

        return msg, kwargs

    def with_context(self, **context_updates) -> "ContextAdapter":
        """Create a new adapter with updated context."""
        new_context = LogContext(
            component=self.context.component,
            operation=context_updates.get("operation", self.context.operation),
            user_id=context_updates.get("user_id", self.context.user_id),
            symbol=context_updates.get("symbol", self.context.symbol),
            trade_id=context_updates.get("trade_id", self.context.trade_id),
            correlation_id=context_updates.get(
                "correlation_id", self.context.correlation_id
            ),
            additional_data={**self.context.additional_data, **context_updates},
        )

        return ContextAdapter(self.logger, new_context)


class LogLevelManager:
    """
    Dynamic log level management for runtime adjustments.
    """

    def __init__(self):
        self.original_levels: Dict[str, int] = {}
        self.lock = threading.Lock()

    def set_level(self, logger_name: str, level: Union[str, int]) -> None:
        """
        Set log level for a specific logger.

        Args:
            logger_name: Name of the logger
            level: New log level (string or int)
        """
        with self.lock:
            target_logger = logging.getLogger(logger_name)

            # Store original level if not already stored
            if logger_name not in self.original_levels:
                self.original_levels[logger_name] = target_logger.level

            # Set new level
            if isinstance(level, str):
                level = getattr(logging, level.upper(), logging.INFO)

            target_logger.setLevel(level)
            logger.info(
                f"Set log level for '{logger_name}' to {logging.getLevelName(level)}"
            )

    def reset_level(self, logger_name: str) -> None:
        """
        Reset log level to original value.

        Args:
            logger_name: Name of the logger
        """
        with self.lock:
            if logger_name in self.original_levels:
                original_level = self.original_levels[logger_name]
                logging.getLogger(logger_name).setLevel(original_level)
                logger.info(
                    f"Reset log level for '{logger_name}' to {logging.getLevelName(original_level)}"
                )
                del self.original_levels[logger_name]

    def reset_all_levels(self) -> None:
        """Reset all log levels to their original values."""
        with self.lock:
            for logger_name in list(self.original_levels.keys()):
                self.reset_level(logger_name)

    def get_current_levels(self) -> Dict[str, str]:
        """Get current log levels for all configured loggers."""
        levels = {}

        # Get all loggers
        for name in sorted(logging.root.manager.loggerDict.keys()):
            logger_instance = logging.getLogger(name)
            levels[name] = logging.getLevelName(logger_instance.level)

        return levels


class LoggingManager:
    """
    Centralized logging manager with consistent setup and configuration.

    Eliminates duplication by providing standardized logging setup
    across the entire framework.
    """

    def __init__(self):
        self.level_manager = LogLevelManager()
        self.config = {}
        self.initialized = False
        self.loggers: Dict[str, ContextAdapter] = {}

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize logging system with configuration.

        Args:
            config: Logging configuration dictionary
        """
        if self.initialized:
            logger.warning("Logging system already initialized")
            return

        # Get configuration
        self.config = config or get_logging_config()

        # Set up root logger
        self._setup_root_logger()

        # Set up component-specific loggers
        self._setup_component_loggers()

        self.initialized = True
        logger.info("Logging system initialized successfully")

    def _setup_root_logger(self) -> None:
        """Set up the root logger with handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(
            getattr(logging, self.config.get("level", "INFO").upper(), logging.INFO)
        )

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add console handler if enabled
        if self.config.get("console", True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                ColoredFormatter("%(name)s - %(levelname)s - %(message)s")
            )
            console_handler.setLevel(root_logger.level)
            root_logger.addHandler(console_handler)

        # Add file handler if enabled
        if self.config.get("file_logging", True):
            self._setup_file_handler(root_logger)

    def _setup_file_handler(self, target_logger: logging.Logger) -> None:
        """Set up rotating file handler."""
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(exist_ok=True)

        log_file = logs_dir / "n1v1.log"
        max_size = self.config.get("max_size", 10 * 1024 * 1024)  # 10MB default
        backup_count = self.config.get("backup_count", 5)

        file_handler = logging.handlers.RotatingFileHandler(
            str(log_file), maxBytes=max_size, backupCount=backup_count, encoding="utf-8"
        )

        file_handler.setFormatter(StructuredFormatter())
        file_handler.setLevel(target_logger.level)
        target_logger.addHandler(file_handler)

    def _setup_component_loggers(self) -> None:
        """Set up loggers for common components."""
        components = [
            "trading",
            "risk",
            "execution",
            "data",
            "api",
            "database",
            "notifications",
            "monitoring",
            "optimization",
            "backtest",
        ]

        for component in components:
            self.get_component_logger(component)

    def get_component_logger(
        self, component: str, operation: Optional[str] = None
    ) -> ContextAdapter:
        """
        Get a configured logger for a specific component.

        Args:
            component: Component name
            operation: Optional operation name

        Returns:
            Configured ContextAdapter logger
        """
        if not self.initialized:
            self.initialize()

        cache_key = f"{component}:{operation or ''}"

        if cache_key not in self.loggers:
            base_logger = logging.getLogger(f"n1v1.{component}")
            context = LogContext(component=component, operation=operation)
            adapter = ContextAdapter(base_logger, context)
            self.loggers[cache_key] = adapter

        return self.loggers[cache_key]

    def create_operation_logger(
        self, component: str, operation: str, **context
    ) -> ContextAdapter:
        """
        Create a logger for a specific operation with additional context.

        Args:
            component: Component name
            operation: Operation name
            **context: Additional context data

        Returns:
            Configured ContextAdapter logger
        """
        base_logger = self.get_component_logger(component, operation)

        # Add additional context
        for key, value in context.items():
            if hasattr(base_logger.context, key):
                setattr(base_logger.context, key, value)
            else:
                base_logger.context.additional_data[key] = value

        return base_logger

    def set_component_level(self, component: str, level: Union[str, int]) -> None:
        """
        Set log level for a specific component.

        Args:
            component: Component name
            level: Log level
        """
        logger_name = f"n1v1.{component}"
        self.level_manager.set_level(logger_name, level)

    def reset_component_level(self, component: str) -> None:
        """
        Reset log level for a specific component.

        Args:
            component: Component name
        """
        logger_name = f"n1v1.{component}"
        self.level_manager.reset_level(logger_name)

    def get_log_levels(self) -> Dict[str, str]:
        """Get current log levels for all components."""
        return self.level_manager.get_current_levels()

    def reload_configuration(self) -> None:
        """Reload logging configuration."""
        self.config = get_logging_config()
        self.initialize(self.config)
        logger.info("Logging configuration reloaded")

    def shutdown(self) -> None:
        """Shutdown logging system and cleanup."""
        self.level_manager.reset_all_levels()

        # Close all handlers
        for logger_instance in logging.root.manager.loggerDict.values():
            if hasattr(logger_instance, "handlers"):
                for handler in logger_instance.handlers:
                    if hasattr(handler, "close"):
                        handler.close()

        logging.shutdown()
        self.initialized = False
        logger.info("Logging system shutdown complete")


# Global logging manager instance
_logging_manager = LoggingManager()


def get_logging_manager() -> LoggingManager:
    """Get the global logging manager instance."""
    return _logging_manager


# Convenience functions for backward compatibility
def initialize_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize logging system (backward compatibility)."""
    _logging_manager.initialize(config)


def get_component_logger(
    component: str, operation: Optional[str] = None
) -> ContextAdapter:
    """Get component logger (backward compatibility)."""
    return _logging_manager.get_component_logger(component, operation)


def create_operation_logger(
    component: str, operation: str, **context
) -> ContextAdapter:
    """Create operation logger (backward compatibility)."""
    return _logging_manager.create_operation_logger(component, operation, **context)


def set_log_level(component: str, level: Union[str, int]) -> None:
    """Set component log level (backward compatibility)."""
    _logging_manager.set_component_level(component, level)


def reset_log_level(component: str) -> None:
    """Reset component log level (backward compatibility)."""
    _logging_manager.reset_component_level(component)


# Utility functions for common logging patterns
def log_trade_execution(logger: ContextAdapter, trade_data: Dict[str, Any]) -> None:
    """
    Log trade execution with structured data.

    Args:
        logger: Logger instance
        trade_data: Trade execution data
    """
    logger.info(
        f"Trade executed: {trade_data.get('symbol', 'UNKNOWN')} "
        f"{trade_data.get('side', 'UNKNOWN')} {trade_data.get('quantity', 0)}",
        extra={"trade_data": trade_data},
    )


def log_error_with_context(
    logger: ContextAdapter, error: Exception, operation: str, **context
) -> None:
    """
    Log error with structured context.

    Args:
        logger: Logger instance
        error: Exception that occurred
        operation: Operation that failed
        **context: Additional context data
    """
    logger.error(
        f"Error in {operation}: {str(error)}",
        extra={"error_type": type(error).__name__, "operation": operation, **context},
        exc_info=True,
    )


def log_performance_metric(
    logger: ContextAdapter, metric_name: str, value: Union[int, float], **context
) -> None:
    """
    Log performance metric.

    Args:
        logger: Logger instance
        metric_name: Name of the metric
        value: Metric value
        **context: Additional context data
    """
    logger.info(
        f"Performance metric: {metric_name} = {value}",
        extra={"metric_name": metric_name, "metric_value": value, **context},
    )


def log_system_status(
    logger: ContextAdapter, status: str, component: str, **context
) -> None:
    """
    Log system status change.

    Args:
        logger: Logger instance
        status: New status
        component: Component name
        **context: Additional context data
    """
    logger.info(
        f"System status: {component} is {status}",
        extra={"status": status, "component": component, **context},
    )


# Quick setup for development
def setup_development_logging() -> None:
    """Set up logging for development environment."""
    config = {
        "level": "DEBUG",
        "console": True,
        "file_logging": True,
        "max_size": 5 * 1024 * 1024,  # 5MB
        "backup_count": 3,
    }

    initialize_logging(config)


def setup_production_logging() -> None:
    """Set up logging for production environment."""
    config = {
        "level": "INFO",
        "console": False,
        "file_logging": True,
        "max_size": 50 * 1024 * 1024,  # 50MB
        "backup_count": 10,
    }

    initialize_logging(config)
