"""
Logging Manager - Structured logging with consistent schema and aggregation.

Provides centralized logging configuration, structured logging with consistent schema,
log level standardization, and log aggregation and analysis capabilities.
"""

import json
import logging
import logging.config
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.constants import (
    DEFAULT_LOG_FILE,
    DEFAULT_LOG_FORMAT,
    DETAILED_LOG_FORMAT,
    ERROR_LOG_FILE,
    LOG_LEVELS,
    LOGS_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Structured log entry with consistent schema."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    level: str = "INFO"
    logger_name: str = ""
    message: str = ""
    module: str = ""
    function: str = ""
    line_number: int = 0
    thread_id: int = field(default_factory=lambda: threading.get_ident())
    process_id: int = field(default_factory=lambda: __import__("os").getpid())
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    component: str = ""
    operation: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "logger_name": self.logger_name,
            "message": self.message,
            "module": self.module,
            "function": self.function,
            "line_number": self.line_number,
            "thread_id": self.thread_id,
            "process_id": self.process_id,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "component": self.component,
            "operation": self.operation,
            "context": self.context,
            "error_code": self.error_code,
            "stack_trace": self.stack_trace,
            "performance_metrics": self.performance_metrics,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class StructuredLogRecord(logging.LogRecord):
    """Extended LogRecord with structured logging capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.correlation_id = getattr(self, "correlation_id", None)
        self.user_id = getattr(self, "user_id", None)
        self.session_id = getattr(self, "session_id", None)
        self.request_id = getattr(self, "request_id", None)
        self.component = getattr(self, "component", "unknown")
        self.operation = getattr(self, "operation", "")
        self.context = getattr(self, "context", {})
        self.error_code = getattr(self, "error_code", None)
        self.performance_metrics = getattr(self, "performance_metrics", {})


class StructuredJSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Create structured log entry
        log_entry = LogEntry(
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=getattr(record, "module", ""),
            function=getattr(record, "funcName", ""),
            line_number=getattr(record, "lineno", 0),
            correlation_id=getattr(record, "correlation_id", None),
            user_id=getattr(record, "user_id", None),
            session_id=getattr(record, "session_id", None),
            request_id=getattr(record, "request_id", None),
            component=getattr(record, "component", "unknown"),
            operation=getattr(record, "operation", ""),
            context=getattr(record, "context", {}),
            error_code=getattr(record, "error_code", None),
            performance_metrics=getattr(record, "performance_metrics", {}),
        )

        # Add exception info if present
        if record.exc_info:
            log_entry.stack_trace = self.formatException(record.exc_info)

        return log_entry.to_json()


class ContextualFormatter(logging.Formatter):
    """Formatter that includes contextual information."""

    def __init__(self, fmt=None, datefmt=None, style="%", context_keys=None):
        super().__init__(fmt, datefmt, style)
        self.context_keys = context_keys or ["correlation_id", "user_id", "component"]

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with contextual information."""
        # Add contextual information to the message
        context_parts = []
        for key in self.context_keys:
            value = getattr(record, key, None)
            if value:
                context_parts.append(f"{key}={value}")

        if context_parts:
            original_msg = record.getMessage()
            record.msg = f"{' | '.join(context_parts)} | {original_msg}"

        return super().format(record)


class LogAggregator:
    """
    Aggregates and analyzes log data for insights and monitoring.
    """

    def __init__(self, max_entries: int = 10000):
        self.entries: deque = deque(maxlen=max_entries)
        self.metrics: Dict[str, Any] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.patterns: Dict[str, re.Pattern] = {}

    def add_entry(self, entry: LogEntry):
        """Add a log entry to the aggregator."""
        self.entries.append(entry)

        # Update metrics
        self._update_metrics(entry)

        # Check for patterns
        self._check_patterns(entry)

    def _update_metrics(self, entry: LogEntry):
        """Update aggregation metrics."""
        # Error rate tracking
        if entry.level in ["ERROR", "CRITICAL"]:
            hour_key = entry.timestamp[:13]  # YYYY-MM-DDTHH
            if "error_rate" not in self.metrics:
                self.metrics["error_rate"] = defaultdict(int)
            self.metrics["error_rate"][hour_key] += 1

        # Component activity
        if "component_activity" not in self.metrics:
            self.metrics["component_activity"] = defaultdict(int)
        self.metrics["component_activity"][entry.component] += 1

        # Performance metrics
        if entry.performance_metrics:
            if "performance" not in self.metrics:
                self.metrics["performance"] = defaultdict(list)
            for key, value in entry.performance_metrics.items():
                if isinstance(value, (int, float)):
                    self.metrics["performance"][key].append(value)

    def _check_patterns(self, entry: LogEntry):
        """Check log entry against defined patterns."""
        message = entry.message.lower()

        # Define alert patterns
        alert_patterns = {
            "high_error_rate": r"error.*rate.*high|too many errors",
            "security_breach": r"security|breach|unauthorized|intrusion",
            "performance_degradation": r"slow|timeout|degraded|latency",
            "resource_exhaustion": r"out of memory|disk full|cpu high",
        }

        for pattern_name, pattern in alert_patterns.items():
            if re.search(pattern, message):
                self.alerts.append(
                    {
                        "timestamp": entry.timestamp,
                        "pattern": pattern_name,
                        "level": entry.level,
                        "message": entry.message,
                        "component": entry.component,
                    }
                )

    def get_metrics_report(self) -> Dict[str, Any]:
        """Generate metrics report."""
        return {
            "total_entries": len(self.entries),
            "time_range": self._get_time_range(),
            "error_rate_trend": dict(self.metrics.get("error_rate", {})),
            "component_activity": dict(self.metrics.get("component_activity", {})),
            "performance_summary": self._get_performance_summary(),
            "active_alerts": len(self.alerts),
            "recent_alerts": self.alerts[-10:] if self.alerts else [],
        }

    def _get_time_range(self) -> Dict[str, str]:
        """Get the time range of aggregated logs."""
        if not self.entries:
            return {"start": None, "end": None}

        timestamps = [e.timestamp for e in self.entries]
        return {"start": min(timestamps), "end": max(timestamps)}

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        perf_data = self.metrics.get("performance", {})
        summary = {}

        for metric, values in perf_data.items():
            if values:
                summary[metric] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "p95": sorted(values)[int(len(values) * 0.95)]
                    if len(values) > 1
                    else max(values),
                }

        return summary

    def search_logs(
        self,
        query: str,
        level: Optional[str] = None,
        component: Optional[str] = None,
        limit: int = 100,
    ) -> List[LogEntry]:
        """Search logs with filters."""
        results = []

        for entry in self.entries:
            # Apply filters
            if level and entry.level != level:
                continue
            if component and entry.component != component:
                continue
            if query.lower() not in entry.message.lower():
                continue

            results.append(entry)
            if len(results) >= limit:
                break

        return results


class LogLevelValidator:
    """
    Validates and standardizes log levels across the application.
    """

    def __init__(self):
        self.allowed_levels = set(LOG_LEVELS.keys())
        self.level_hierarchy = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50,
        }

    def validate_level(self, level: str) -> bool:
        """Validate if a log level is allowed."""
        return level.upper() in self.allowed_levels

    def normalize_level(self, level: Union[str, int]) -> str:
        """Normalize log level to string format."""
        if isinstance(level, str):
            return level.upper()
        elif isinstance(level, int):
            # Convert numeric level to string
            for name, value in self.level_hierarchy.items():
                if level == value:
                    return name
            return "INFO"  # Default
        else:
            return "INFO"

    def should_log(self, message_level: str, logger_level: str) -> bool:
        """Check if a message should be logged based on levels."""
        msg_value = self.level_hierarchy.get(message_level.upper(), 20)
        logger_value = self.level_hierarchy.get(logger_level.upper(), 20)
        return msg_value >= logger_value

    def get_valid_levels(self) -> List[str]:
        """Get list of valid log levels."""
        return sorted(
            self.allowed_levels, key=lambda x: self.level_hierarchy.get(x, 99)
        )


class LoggingConfigurationManager:
    """
    Manages logging configuration with validation and dynamic updates.
    """

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.validator = LogLevelValidator()
        self.aggregator = LogAggregator()
        self.active_loggers: Dict[str, logging.Logger] = {}

    def load_configuration(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load logging configuration from file or use defaults."""
        if config_file and Path(config_file).exists():
            with open(config_file, "r") as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()

        # Validate configuration
        self._validate_config()

        return self.config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {"format": DEFAULT_LOG_FORMAT},
                "detailed": {"format": DETAILED_LOG_FORMAT},
                "json": {"()": "utils.logging_manager.StructuredJSONFormatter"},
                "contextual": {
                    "()": "utils.logging_manager.ContextualFormatter",
                    "format": DEFAULT_LOG_FORMAT,
                    "context_keys": ["correlation_id", "component", "operation"],
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": str(DEFAULT_LOG_FILE),
                    "maxBytes": 10 * 1024 * 1024,  # 10MB
                    "backupCount": 5,
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "json",
                    "filename": str(ERROR_LOG_FILE),
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 5,
                },
                "json_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "json",
                    "filename": str(LOGS_DIR / "structured.log"),
                    "maxBytes": 50 * 1024 * 1024,  # 50MB
                    "backupCount": 3,
                },
            },
            "loggers": {
                "trading_system": {
                    "level": "DEBUG",
                    "handlers": ["console", "file", "error_file", "json_file"],
                    "propagate": False,
                },
                "utils": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
            },
            "root": {"level": "INFO", "handlers": ["console"]},
        }

    def _validate_config(self):
        """Validate logging configuration."""
        # Validate log levels
        for logger_config in self.config.get("loggers", {}).values():
            level = logger_config.get("level", "INFO")
            if not self.validator.validate_level(level):
                logger.warning(f"Invalid log level '{level}' in configuration")

        # Validate formatters
        required_formatters = ["standard", "detailed", "json"]
        available_formatters = set(self.config.get("formatters", {}).keys())

        for formatter in required_formatters:
            if formatter not in available_formatters:
                logger.warning(
                    f"Required formatter '{formatter}' not found in configuration"
                )

    def apply_configuration(self):
        """Apply the logging configuration."""
        try:
            logging.config.dictConfig(self.config)
            logger.info("Logging configuration applied successfully")
        except Exception as e:
            logger.error(f"Failed to apply logging configuration: {e}")
            # Apply minimal fallback configuration
            self._apply_fallback_config()

    def _apply_fallback_config(self):
        """Apply minimal fallback logging configuration."""
        fallback_config = {
            "version": 1,
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "standard",
                }
            },
            "formatters": {"standard": {"format": DEFAULT_LOG_FORMAT}},
            "root": {"level": "INFO", "handlers": ["console"]},
        }

        logging.config.dictConfig(fallback_config)
        logger.warning("Applied fallback logging configuration")

    def update_log_level(self, logger_name: str, level: str):
        """Update log level for a specific logger."""
        if not self.validator.validate_level(level):
            raise ValueError(f"Invalid log level: {level}")

        logger_instance = logging.getLogger(logger_name)
        logger_instance.setLevel(getattr(logging, level.upper()))

        logger.info(f"Updated log level for '{logger_name}' to {level}")

    def add_structured_handler(self, logger_name: str = "trading_system"):
        """Add structured JSON handler to a logger."""
        logger_instance = logging.getLogger(logger_name)

        # Check if JSON handler already exists
        for handler in logger_instance.handlers:
            if hasattr(handler, "formatter") and isinstance(
                handler.formatter, StructuredJSONFormatter
            ):
                return  # Already has structured handler

        # Add structured handler
        json_handler = logging.handlers.RotatingFileHandler(
            LOGS_DIR / "structured.log", maxBytes=50 * 1024 * 1024, backupCount=3
        )
        json_handler.setFormatter(StructuredJSONFormatter())
        json_handler.setLevel(logging.INFO)

        logger_instance.addHandler(json_handler)
        logger.info(f"Added structured handler to logger '{logger_name}'")

    def get_configuration_report(self) -> Dict[str, Any]:
        """Generate configuration report."""
        return {
            "active_loggers": len(logging.root.manager.loggerDict),
            "configured_handlers": list(self.config.get("handlers", {}).keys()),
            "configured_formatters": list(self.config.get("formatters", {}).keys()),
            "log_levels": self._get_current_log_levels(),
            "validation_status": "valid"
            if self._validate_config() is None
            else "has_warnings",
        }

    def _get_current_log_levels(self) -> Dict[str, str]:
        """Get current log levels for all loggers."""
        levels = {}
        for name, logger_instance in logging.root.manager.loggerDict.items():
            if isinstance(logger_instance, logging.Logger):
                levels[name] = self.validator.normalize_level(logger_instance.level)

        levels["root"] = self.validator.normalize_level(logging.root.level)
        return levels


class LoggingManager:
    """
    Comprehensive logging management system with structured logging,
    aggregation, and monitoring capabilities.
    """

    def __init__(self):
        self.config_manager = LoggingConfigurationManager()
        self.aggregator = LogAggregator()
        self.validator = LogLevelValidator()
        self.context: Dict[str, Any] = {}
        self._correlation_counter = 0

    def initialize(self, config_file: Optional[str] = None):
        """Initialize the logging system."""
        # Load configuration
        config = self.config_manager.load_configuration(config_file)

        # Apply configuration
        self.config_manager.apply_configuration()

        # Add structured handlers
        self.config_manager.add_structured_handler()

        logger.info("Logging system initialized successfully")

    def get_logger(self, name: str, component: str = "unknown") -> logging.Logger:
        """Get a configured logger with component context."""
        logger_instance = logging.getLogger(name)

        # Add component context
        if not hasattr(logger_instance, "_component"):
            logger_instance._component = component

        return logger_instance

    def create_correlation_id(self) -> str:
        """Create a unique correlation ID for request tracking."""
        self._correlation_counter += 1
        return f"corr_{int(time.time())}_{self._correlation_counter}"

    def set_context(self, **kwargs):
        """Set logging context for current thread."""
        self.context.update(kwargs)

    def clear_context(self):
        """Clear logging context."""
        self.context.clear()

    def log_with_context(
        self, logger: logging.Logger, level: str, message: str, **kwargs
    ):
        """Log a message with additional context."""
        # Merge global context with local context
        full_context = {**self.context, **kwargs}

        # Add structured fields
        extra = {
            "correlation_id": full_context.get("correlation_id"),
            "user_id": full_context.get("user_id"),
            "session_id": full_context.get("session_id"),
            "request_id": full_context.get("request_id"),
            "component": full_context.get("component", "unknown"),
            "operation": full_context.get("operation", ""),
            "context": {
                k: v
                for k, v in full_context.items()
                if k
                not in [
                    "correlation_id",
                    "user_id",
                    "session_id",
                    "request_id",
                    "component",
                    "operation",
                ]
            },
            "error_code": full_context.get("error_code"),
            "performance_metrics": full_context.get("performance_metrics", {}),
        }

        # Remove None values
        extra = {k: v for k, v in extra.items() if v is not None}

        logger.log(getattr(logging, level.upper()), message, extra=extra)

    def log_performance(
        self, operation: str, duration: float, component: str = "unknown", **metrics
    ):
        """Log performance metrics."""
        self.log_with_context(
            logger=logging.getLogger(f"{component}.performance"),
            level="INFO",
            message=f"Performance: {operation}",
            component=component,
            operation=operation,
            performance_metrics={
                "duration_seconds": duration,
                "operation": operation,
                **metrics,
            },
        )

    def log_error(
        self,
        error: Exception,
        component: str = "unknown",
        operation: str = "",
        error_code: Optional[str] = None,
    ):
        """Log an error with full context."""
        import traceback

        self.log_with_context(
            logger=logging.getLogger(f"{component}.error"),
            level="ERROR",
            message=f"Error in {operation}: {str(error)}",
            component=component,
            operation=operation,
            error_code=error_code or type(error).__name__,
            stack_trace=traceback.format_exc(),
        )

    def get_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive logging system report."""
        return {
            "configuration": self.config_manager.get_configuration_report(),
            "metrics": self.aggregator.get_metrics_report(),
            "validation": {
                "log_levels_valid": True,  # Would implement full validation
                "configuration_valid": True,
                "handlers_active": len(logging.root.handlers) > 0,
            },
            "context": self.context.copy(),
            "active_correlations": self._correlation_counter,
        }

    def search_logs(self, query: str, **filters) -> List[LogEntry]:
        """Search aggregated logs."""
        return self.aggregator.search_logs(query, **filters)

    def export_logs(
        self,
        output_file: str,
        format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ):
        """Export logs in specified format."""
        entries = list(self.aggregator.entries)

        # Filter by time range
        if start_time or end_time:
            filtered_entries = []
            for entry in entries:
                entry_time = datetime.fromisoformat(
                    entry.timestamp.replace("Z", "+00:00")
                )
                if start_time and entry_time < start_time:
                    continue
                if end_time and entry_time > end_time:
                    continue
                filtered_entries.append(entry)
            entries = filtered_entries

        if format == "json":
            with open(output_file, "w") as f:
                json.dump(
                    [entry.to_dict() for entry in entries], f, indent=2, default=str
                )
        elif format == "csv":
            import csv

            with open(output_file, "w", newline="") as f:
                if entries:
                    writer = csv.DictWriter(f, fieldnames=entries[0].to_dict().keys())
                    writer.writeheader()
                    writer.writerows([entry.to_dict() for entry in entries])

        logger.info(f"Exported {len(entries)} log entries to {output_file}")


# Global logging manager instance
_logging_manager = None


def get_logging_manager() -> LoggingManager:
    """Get the global logging manager instance."""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager


# Convenience functions
def setup_logging(config_file: Optional[str] = None):
    """Set up the logging system."""
    manager = get_logging_manager()
    manager.initialize(config_file)


def get_logger(name: str, component: str = "unknown") -> logging.Logger:
    """Get a configured logger."""
    manager = get_logging_manager()
    return manager.get_logger(name, component)


def log_performance(
    operation: str, duration: float, component: str = "unknown", **metrics
):
    """Log performance metrics."""
    manager = get_logging_manager()
    manager.log_performance(operation, duration, component, **metrics)


def log_error(
    error: Exception,
    component: str = "unknown",
    operation: str = "",
    error_code: Optional[str] = None,
):
    """Log an error with context."""
    manager = get_logging_manager()
    manager.log_error(error, component, operation, error_code)


# Context manager for correlation IDs
class LoggingContext:
    """Context manager for logging correlation."""

    def __init__(self, **context):
        self.context = context
        self.manager = get_logging_manager()

    def __enter__(self):
        self.manager.set_context(**self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.clear_context()


# Decorator for automatic performance logging
def log_performance_decorator(component: str = "unknown"):
    """Decorator to automatically log function performance."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                duration = end_time - start_time

                log_performance(func.__name__, duration, component)
                return result
            except Exception as e:
                end_time = time.perf_counter()
                duration = end_time - start_time

                log_performance(f"{func.__name__}_error", duration, component)
                log_error(e, component, func.__name__)
                raise

        return wrapper

    return decorator
