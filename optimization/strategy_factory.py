"""
Strategy Factory Module

This module provides a secure factory pattern for creating trading strategies
from genetic genomes. It replaces dynamic code generation with a safe,
configurable approach that prevents arbitrary code execution.

Key Features:
- Secure strategy instantiation from genomes
- Parameter validation and constraint checking
- Registry-based strategy management
- Comprehensive error handling and logging
- Support for multiple strategy types
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    get_indicator_params,
    get_signal_logic_params,
)


class StrategyGenerationError(Exception):
    """
    Custom exception for strategy generation failures.

    This exception provides detailed information about why a strategy
    generation failed, including the specific error type and context.
    """

    def __init__(
        self,
        message: str,
        error_type: str = "unknown",
        genome_info: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize the exception.

        Args:
            message: Detailed error message
            error_type: Type of error (e.g., 'validation_failed', 'factory_error', 'missing_strategy')
            genome_info: Information about the genome that failed
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.error_type = error_type
        self.genome_info = genome_info or {}
        self.cause = cause

    def __str__(self) -> str:
        """String representation with detailed information."""
        base_msg = f"StrategyGenerationError [{self.error_type}]: {super().__str__()}"
        if self.genome_info:
            base_msg += f" | Genome: {self.genome_info}"
        if self.cause:
            base_msg += f" | Caused by: {type(self.cause).__name__}: {str(self.cause)}"
        return base_msg


class StrategyFactory:
    """
    Secure factory for creating trading strategies from genomes.

    This factory replaces dynamic code generation with a secure mapping approach
    that prevents arbitrary code execution and provides comprehensive validation.

    SECURITY FEATURES:
    - Predefined strategy mappings only
    - Parameter validation against allowed ranges
    - No dynamic code execution (exec/eval/type)
    - Clear audit trail for strategy instantiation
    """

    # Registry of allowed strategy types and their parameter constraints
    # SECURITY: Classes are explicitly defined to prevent dynamic code execution
    # Only pre-approved strategy classes can be instantiated through this factory
    STRATEGY_REGISTRY = {
        "rsi_momentum": {
            "class": None,  # To be registered by calling register_strategy()
            "description": "RSI-based momentum strategy",
            "parameters": {
                "rsi_period": {"min": 2, "max": 50, "type": int},
                "overbought": {"min": 60, "max": 90, "type": int},
                "oversold": {"min": 10, "max": 40, "type": int},
                "momentum_period": {"min": 5, "max": 30, "type": int},
            },
        },
        "macd_crossover": {
            "class": None,  # To be registered by calling register_strategy()
            "description": "MACD crossover strategy",
            "parameters": {
                "fast_period": {"min": 5, "max": 20, "type": int},
                "slow_period": {"min": 20, "max": 50, "type": int},
                "signal_period": {"min": 5, "max": 15, "type": int},
            },
        },
        "bollinger_reversion": {
            "class": None,  # To be registered by calling register_strategy()
            "description": "Bollinger Bands mean reversion strategy",
            "parameters": {
                "period": {"min": 10, "max": 50, "type": int},
                "std_dev": {"min": 1.5, "max": 3.0, "type": float},
            },
        },
        "volume_price": {
            "class": None,  # To be registered by calling register_strategy()
            "description": "Volume-weighted price action strategy",
            "parameters": {
                "volume_threshold": {"min": 0.5, "max": 3.0, "type": float},
                "price_lookback": {"min": 3, "max": 20, "type": int},
            },
        },
    }

    @classmethod
    def register_strategy(
        cls,
        strategy_type: str,
        strategy_class: type,
        description: str,
        parameters: Dict[str, Any],
    ) -> None:
        """
        Register a new strategy type with the factory.

        Args:
            strategy_type: Unique identifier for the strategy
            strategy_class: The actual strategy class
            description: Human-readable description
            parameters: Parameter constraints
        """
        if strategy_type in cls.STRATEGY_REGISTRY:
            logger.warning(
                f"Strategy type '{strategy_type}' already registered, overwriting"
            )

        cls.STRATEGY_REGISTRY[strategy_type] = {
            "class": strategy_class,
            "description": description,
            "parameters": parameters,
        }

        logger.info(f"Registered strategy type: {strategy_type}")

    @classmethod
    def validate_genome(cls, genome) -> Tuple[bool, List[str]]:
        """
        Validate a genome against security constraints.

        Args:
            genome: The genome to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check genome structure
        if not genome.genes:
            errors.append("Genome has no genes")
            return False, errors

        # Validate each gene
        for i, gene in enumerate(genome.genes):
            if not gene.enabled:
                continue  # Skip disabled genes

            # Validate component type
            if gene.component_type not in [
                StrategyComponent.INDICATOR,
                StrategyComponent.SIGNAL_LOGIC,
                StrategyComponent.RISK_MANAGEMENT,
                StrategyComponent.FILTER,
            ]:
                errors.append(f"Gene {i}: Invalid component type {gene.component_type}")

            # Validate parameters based on component type
            param_errors = cls._validate_gene_parameters(gene)
            errors.extend([f"Gene {i}: {err}" for err in param_errors])

        return len(errors) == 0, errors

    @classmethod
    def _validate_gene_parameters(cls, gene) -> List[str]:
        """Validate parameters for a single gene."""
        errors = []

        # Get parameter constraints based on component type
        if gene.component_type == StrategyComponent.INDICATOR and gene.indicator_type:
            constraints = cls._get_indicator_constraints(gene.indicator_type)
        elif (
            gene.component_type == StrategyComponent.SIGNAL_LOGIC and gene.signal_logic
        ):
            constraints = cls._get_signal_constraints(gene.signal_logic)
        else:
            # For other component types, use generic validation
            constraints = cls._get_generic_constraints(gene.component_type)

        # Validate each parameter
        for param_name, param_value in gene.parameters.items():
            if param_name in constraints:
                constraint = constraints[param_name]
                if not cls._validate_parameter(param_value, constraint):
                    errors.append(
                        f"Parameter {param_name}={param_value} violates constraint {constraint}"
                    )
            else:
                errors.append(f"Unknown parameter {param_name}")

        return errors

    @classmethod
    def _validate_parameter(cls, value: Any, constraint: Dict[str, Any]) -> bool:
        """Validate a single parameter against its constraint."""
        try:
            # Type check
            expected_type = constraint["type"]
            if not isinstance(value, expected_type):
                return False

            # Range check
            if "min" in constraint and value < constraint["min"]:
                return False
            if "max" in constraint and value > constraint["max"]:
                return False

            # Enum check
            if (
                "allowed_values" in constraint
                and value not in constraint["allowed_values"]
            ):
                return False

            return True
        except Exception:
            return False

    @classmethod
    def _get_indicator_constraints(cls, indicator_type) -> Dict[str, Any]:
        """Get parameter constraints for an indicator type."""
        # Use configuration from config module
        config_params = get_indicator_params(indicator_type.value)
        if config_params:
            return config_params

        # Fallback to hardcoded constraints
        constraints = {
            IndicatorType.RSI: {
                "period": {"min": 2, "max": 50, "type": int},
                "overbought": {"min": 50, "max": 95, "type": int},
                "oversold": {"min": 5, "max": 50, "type": int},
            },
            IndicatorType.MACD: {
                "fast_period": {"min": 5, "max": 20, "type": int},
                "slow_period": {"min": 20, "max": 50, "type": int},
                "signal_period": {"min": 5, "max": 15, "type": int},
            },
            IndicatorType.BOLLINGER_BANDS: {
                "period": {"min": 10, "max": 50, "type": int},
                "std_dev": {"min": 1.0, "max": 3.0, "type": float},
            },
            IndicatorType.STOCHASTIC: {
                "k_period": {"min": 5, "max": 30, "type": int},
                "d_period": {"min": 3, "max": 10, "type": int},
                "overbought": {"min": 70, "max": 95, "type": int},
                "oversold": {"min": 5, "max": 30, "type": int},
            },
            IndicatorType.MOVING_AVERAGE: {
                "period": {"min": 5, "max": 100, "type": int},
                "type": {"allowed_values": ["sma", "ema", "wma"], "type": str},
            },
            IndicatorType.ATR: {"period": {"min": 5, "max": 30, "type": int}},
            IndicatorType.VOLUME: {"period": {"min": 5, "max": 50, "type": int}},
            IndicatorType.PRICE_ACTION: {
                "lookback": {"min": 3, "max": 20, "type": int}
            },
        }
        return constraints.get(indicator_type, {})

    @classmethod
    def _get_signal_constraints(cls, signal_logic) -> Dict[str, Any]:
        """Get parameter constraints for signal logic."""
        # Use configuration from config module
        config_params = get_signal_logic_params(signal_logic.value)
        if config_params:
            return config_params

        # Fallback to hardcoded constraints
        constraints = {
            SignalLogic.CROSSOVER: {
                "fast_period": {"min": 5, "max": 20, "type": int},
                "slow_period": {"min": 20, "max": 50, "type": int},
            },
            SignalLogic.THRESHOLD: {
                "threshold": {"min": 0.1, "max": 0.9, "type": float},
                "direction": {"allowed_values": ["above", "below"], "type": str},
            },
            SignalLogic.PATTERN: {
                "pattern_type": {
                    "allowed_values": ["double_bottom", "double_top", "head_shoulders"],
                    "type": str,
                },
                "tolerance": {"min": 0.01, "max": 0.1, "type": float},
            },
            SignalLogic.DIVERGENCE: {
                "lookback": {"min": 5, "max": 20, "type": int},
                "threshold": {"min": 0.05, "max": 0.5, "type": float},
            },
            SignalLogic.MOMENTUM: {
                "period": {"min": 5, "max": 30, "type": int},
                "threshold": {"min": 0.01, "max": 0.1, "type": float},
            },
            SignalLogic.MEAN_REVERSION: {
                "mean_period": {"min": 10, "max": 50, "type": int},
                "std_threshold": {"min": 1.0, "max": 3.0, "type": float},
            },
        }
        return constraints.get(signal_logic, {})

    @classmethod
    def _get_generic_constraints(cls, component_type) -> Dict[str, Any]:
        """Get generic parameter constraints for component types."""
        constraints = {
            StrategyComponent.RISK_MANAGEMENT: {
                "stop_loss": {"min": 0.005, "max": 0.1, "type": float},
                "take_profit": {"min": 0.01, "max": 0.2, "type": float},
            },
            StrategyComponent.FILTER: {
                "volume_threshold": {"min": 0.1, "max": 5.0, "type": float}
            },
        }
        return constraints.get(component_type, {})

    @classmethod
    def create_strategy_from_genome(cls, genome) -> Optional[Any]:
        """
        Create a strategy instance from a validated genome.

        Args:
            genome: The genome to convert to a strategy

        Returns:
            Strategy instance or None if creation fails
        """
        try:
            # Validate genome first
            is_valid, errors = cls.validate_genome(genome)
            if not is_valid:
                logger.error(f"Genome validation failed: {errors}")
                return None

            # Determine strategy type from genome characteristics
            strategy_type = cls._infer_strategy_type(genome)
            if not strategy_type or strategy_type not in cls.STRATEGY_REGISTRY:
                logger.error(f"Unknown or unsupported strategy type: {strategy_type}")
                return None

            # Get strategy class
            strategy_info = cls.STRATEGY_REGISTRY[strategy_type]
            strategy_class = strategy_info["class"]

            if strategy_class is None:
                logger.error(f"Strategy class not registered for type: {strategy_type}")
                return None

            # Extract and validate parameters
            strategy_params = cls._extract_strategy_parameters(genome, strategy_type)

            # Create strategy configuration
            strategy_config = {
                "name": f"secure_generated_{strategy_type}_{id(genome)}",
                "symbols": ["BTC/USDT"],  # Default, can be overridden
                "timeframe": "1h",
                "required_history": 100,
                "params": strategy_params,
                "genome_id": id(genome),  # For tracking
                "strategy_type": strategy_type,  # For audit trail
            }

            # Create and return strategy instance
            strategy_instance = strategy_class(strategy_config)

            logger.info(
                f"Successfully created strategy of type '{strategy_type}' from genome"
            )
            return strategy_instance

        except Exception as e:
            logger.error(f"Failed to create strategy from genome: {e}")
            return None

    @classmethod
    def _infer_strategy_type(cls, genome) -> Optional[str]:
        """
        Infer the strategy type from genome characteristics.

        This is a simplified inference - in practice, this could be more sophisticated
        based on the combination of genes and their parameters.
        """
        # Count component types
        component_counts = {}
        for gene in genome.genes:
            if gene.enabled:
                component_type = gene.component_type.value
                component_counts[component_type] = (
                    component_counts.get(component_type, 0) + 1
                )

        # Simple inference based on dominant components
        if component_counts.get("indicator", 0) > 0:
            # Look for specific indicator types
            for gene in genome.genes:
                if gene.enabled and gene.component_type == StrategyComponent.INDICATOR:
                    if gene.indicator_type == IndicatorType.RSI:
                        return "rsi_momentum"
                    elif gene.indicator_type == IndicatorType.MACD:
                        return "macd_crossover"
                    elif gene.indicator_type == IndicatorType.BOLLINGER_BANDS:
                        return "bollinger_reversion"
                    elif gene.indicator_type == IndicatorType.VOLUME:
                        return "volume_price"

        # Default fallback
        return "rsi_momentum"  # Safe default

    @classmethod
    def _extract_strategy_parameters(cls, genome, strategy_type: str) -> Dict[str, Any]:
        """Extract validated parameters for the strategy."""
        strategy_info = cls.STRATEGY_REGISTRY[strategy_type]
        allowed_params = strategy_info["parameters"]

        extracted_params = {}

        # Extract parameters from genome genes
        for gene in genome.genes:
            if not gene.enabled:
                continue

            for param_name, param_value in gene.parameters.items():
                if param_name in allowed_params:
                    # Validate parameter value
                    constraint = allowed_params[param_name]
                    if cls._validate_parameter(param_value, constraint):
                        extracted_params[param_name] = param_value

        # Set defaults for missing parameters
        for param_name, constraint in allowed_params.items():
            if param_name not in extracted_params:
                if "default" in constraint:
                    extracted_params[param_name] = constraint["default"]
                else:
                    # Use midpoint of range for numeric types
                    if "min" in constraint and "max" in constraint:
                        if constraint["type"] == int:
                            extracted_params[param_name] = (
                                constraint["min"] + constraint["max"]
                            ) // 2
                        elif constraint["type"] == float:
                            extracted_params[param_name] = (
                                constraint["min"] + constraint["max"]
                            ) / 2.0

        return extracted_params

    @classmethod
    def get_available_strategies(cls) -> Dict[str, str]:
        """Get list of available strategy types and their descriptions."""
        return {
            name: info["description"]
            for name, info in cls.STRATEGY_REGISTRY.items()
            if info["class"] is not None
        }

    @classmethod
    def get_strategy_info(cls, strategy_type: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific strategy type."""
        if strategy_type not in cls.STRATEGY_REGISTRY:
            return None

        info = cls.STRATEGY_REGISTRY[strategy_type].copy()
        # Remove the class object from the returned info for security
        info.pop("class", None)
        return info


# Import required classes to avoid circular imports
# These will be imported when the module is loaded
try:
    from .genome import IndicatorType, SignalLogic, StrategyComponent

    logger = logging.getLogger(f"{__name__}")
except ImportError:
    # Fallback for when genome module is not available yet
    logger = logging.getLogger(f"{__name__}")

    # Define placeholder enums
    class StrategyComponent:
        INDICATOR = "indicator"
        SIGNAL_LOGIC = "signal_logic"
        RISK_MANAGEMENT = "risk_management"
        FILTER = "filter"

    class IndicatorType:
        RSI = "rsi"
        MACD = "macd"
        BOLLINGER_BANDS = "bollinger_bands"
        STOCHASTIC = "stochastic"
        MOVING_AVERAGE = "moving_average"
        ATR = "atr"
        VOLUME = "volume"
        PRICE_ACTION = "price_action"

    class SignalLogic:
        CROSSOVER = "crossover"
        THRESHOLD = "threshold"
        PATTERN = "pattern"
        DIVERGENCE = "divergence"
        MOMENTUM = "momentum"
        MEAN_REVERSION = "mean_reversion"
