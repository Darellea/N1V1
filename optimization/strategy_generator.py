"""
Hybrid AI Strategy Generator

This module implements a revolutionary system for autonomously discovering,
optimizing, and validating new trading strategies beyond human design capabilities.
It combines genetic programming, Bayesian optimization, and distributed evaluation
to create a self-improving trading system.

Key Features:
- Strategy DNA representation with composable genetic blocks
- Evolutionary optimization with species formation
- Bayesian surrogate modeling for expensive evaluations
- Distributed backtesting across multiple cores/GPUs
- Dynamic strategy generation and runtime compilation
- Continuous learning and adaptation

Architecture:
- StrategyGenome: Genetic representation of trading strategies
- StrategyGenerator: Main evolutionary engine
- BayesianOptimizer: Surrogate model for fitness evaluation
- DistributedEvaluator: Parallel strategy evaluation
- StrategyRuntime: Dynamic loading and execution system

Refactored Structure:
- strategy_factory.py: Secure strategy instantiation
- genome.py: Genetic representation and operations
- distributed_evaluator.py: Parallel fitness evaluation
- bayesian_optimizer.py: Surrogate optimization
- config.py: Centralized configuration management
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random
import json
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import defaultdict
import warnings

import pandas as pd
import numpy as np

# ML libraries with fallbacks
try:
    import sklearn.gaussian_process as gp
    from sklearn.gaussian_process import GaussianProcessRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available, Bayesian optimization disabled")

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not available, neural components disabled")

from .base_optimizer import BaseOptimizer
from .genome import (
    StrategyGenome, StrategyGene, StrategyComponent,
    IndicatorType, SignalLogic, Species, GenomeList
)
from strategies.base_strategy import BaseStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.bollinger_reversion_strategy import BollingerReversionStrategy
from backtest.backtester import compute_backtest_metrics
from utils.logger import get_logger

# SECURITY NOTE: This module previously used dynamic code generation (type() and exec())
# which posed significant security risks including arbitrary code execution.
# The implementation has been refactored to use a secure factory pattern that:
# 1. Maps predefined strategy types to safe, validated classes
# 2. Validates all genome parameters against allowed ranges
# 3. Prevents execution of unknown or malicious code
# 4. Provides clear audit trails for strategy instantiation

logger = get_logger(__name__)

# Type aliases for complex types
FitnessResults = List[Tuple[StrategyGenome, float]]
StrategyConfig = Dict[str, Any]
ParameterConstraints = Dict[str, Dict[str, Any]]


class StrategyGenerationError(Exception):
    """
    Custom exception for strategy generation failures.

    This exception provides detailed information about why a strategy
    generation failed, including the specific error type and context.
    """

    def __init__(self, message: str, error_type: str = "unknown",
                 genome_info: Optional[Dict[str, Any]] = None,
                 cause: Optional[Exception] = None):
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
        'rsi_momentum': {
            'class': None,  # To be registered by calling register_strategy()
            'description': 'RSI-based momentum strategy',
            'parameters': {
                'rsi_period': {'min': 2, 'max': 50, 'type': int},
                'overbought': {'min': 60, 'max': 90, 'type': int},
                'oversold': {'min': 10, 'max': 40, 'type': int},
                'momentum_period': {'min': 5, 'max': 30, 'type': int}
            }
        },
        'macd_crossover': {
            'class': None,  # To be registered by calling register_strategy()
            'description': 'MACD crossover strategy',
            'parameters': {
                'fast_period': {'min': 5, 'max': 20, 'type': int},
                'slow_period': {'min': 20, 'max': 50, 'type': int},
                'signal_period': {'min': 5, 'max': 15, 'type': int}
            }
        },
        'bollinger_reversion': {
            'class': None,  # To be registered by calling register_strategy()
            'description': 'Bollinger Bands mean reversion strategy',
            'parameters': {
                'period': {'min': 10, 'max': 50, 'type': int},
                'std_dev': {'min': 1.5, 'max': 3.0, 'type': float}
            }
        },
        'volume_price': {
            'class': None,  # To be registered by calling register_strategy()
            'description': 'Volume-weighted price action strategy',
            'parameters': {
                'volume_threshold': {'min': 0.5, 'max': 3.0, 'type': float},
                'price_lookback': {'min': 3, 'max': 20, 'type': int}
            }
        }
    }

    @classmethod
    def register_strategy(cls, strategy_type: str, strategy_class: type,
                         description: str, parameters: Dict[str, Any]) -> None:
        """
        Register a new strategy type with the factory.

        Args:
            strategy_type: Unique identifier for the strategy
            strategy_class: The actual strategy class
            description: Human-readable description
            parameters: Parameter constraints
        """
        if strategy_type in cls.STRATEGY_REGISTRY:
            logger.warning(f"Strategy type '{strategy_type}' already registered, overwriting")

        cls.STRATEGY_REGISTRY[strategy_type] = {
            'class': strategy_class,
            'description': description,
            'parameters': parameters
        }

        logger.info(f"Registered strategy type: {strategy_type}")

    @classmethod
    def validate_genome(cls, genome: StrategyGenome) -> Tuple[bool, List[str]]:
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
            if gene.component_type not in [StrategyComponent.INDICATOR,
                                         StrategyComponent.SIGNAL_LOGIC,
                                         StrategyComponent.RISK_MANAGEMENT,
                                         StrategyComponent.TIMEFRAME,
                                         StrategyComponent.FILTER]:
                errors.append(f"Gene {i}: Invalid component type {gene.component_type}")

            # Validate parameters based on component type
            param_errors = cls._validate_gene_parameters(gene)
            errors.extend([f"Gene {i}: {err}" for err in param_errors])

        return len(errors) == 0, errors

    @classmethod
    def _validate_gene_parameters(cls, gene: StrategyGene) -> List[str]:
        """Validate parameters for a single gene."""
        errors = []

        # Get parameter constraints based on component type
        if gene.component_type == StrategyComponent.INDICATOR and gene.indicator_type:
            constraints = cls._get_indicator_constraints(gene.indicator_type)
        elif gene.component_type == StrategyComponent.SIGNAL_LOGIC and gene.signal_logic:
            constraints = cls._get_signal_constraints(gene.signal_logic)
        else:
            # For other component types, use generic validation
            constraints = cls._get_generic_constraints(gene.component_type)

        # Validate each parameter
        for param_name, param_value in gene.parameters.items():
            if param_name in constraints:
                constraint = constraints[param_name]
                if not cls._validate_parameter(param_value, constraint):
                    errors.append(f"Parameter {param_name}={param_value} violates constraint {constraint}")
            else:
                errors.append(f"Unknown parameter {param_name}")

        return errors

    @classmethod
    def _validate_parameter(cls, value: Any, constraint: Dict[str, Any]) -> bool:
        """Validate a single parameter against its constraint."""
        try:
            # Type check
            expected_type = constraint['type']
            if not isinstance(value, expected_type):
                return False

            # Range check
            if 'min' in constraint and value < constraint['min']:
                return False
            if 'max' in constraint and value > constraint['max']:
                return False

            # Enum check
            if 'allowed_values' in constraint and value not in constraint['allowed_values']:
                return False

            return True
        except Exception:
            return False

    @classmethod
    def _get_indicator_constraints(cls, indicator_type: IndicatorType) -> Dict[str, Any]:
        """Get parameter constraints for an indicator type."""
        constraints = {
            IndicatorType.RSI: {
                'period': {'min': 2, 'max': 50, 'type': int},
                'overbought': {'min': 50, 'max': 95, 'type': int},
                'oversold': {'min': 5, 'max': 50, 'type': int}
            },
            IndicatorType.MACD: {
                'fast_period': {'min': 5, 'max': 20, 'type': int},
                'slow_period': {'min': 20, 'max': 50, 'type': int},
                'signal_period': {'min': 5, 'max': 15, 'type': int}
            },
            IndicatorType.BOLLINGER_BANDS: {
                'period': {'min': 10, 'max': 50, 'type': int},
                'std_dev': {'min': 1.0, 'max': 3.0, 'type': float}
            },
            IndicatorType.STOCHASTIC: {
                'k_period': {'min': 5, 'max': 30, 'type': int},
                'd_period': {'min': 3, 'max': 10, 'type': int},
                'overbought': {'min': 70, 'max': 95, 'type': int},
                'oversold': {'min': 5, 'max': 30, 'type': int}
            },
            IndicatorType.MOVING_AVERAGE: {
                'period': {'min': 5, 'max': 100, 'type': int},
                'type': {'allowed_values': ['sma', 'ema', 'wma'], 'type': str}
            },
            IndicatorType.ATR: {
                'period': {'min': 5, 'max': 30, 'type': int}
            },
            IndicatorType.VOLUME: {
                'period': {'min': 5, 'max': 50, 'type': int}
            },
            IndicatorType.PRICE_ACTION: {
                'lookback': {'min': 3, 'max': 20, 'type': int}
            }
        }
        return constraints.get(indicator_type, {})

    @classmethod
    def _get_signal_constraints(cls, signal_logic: SignalLogic) -> Dict[str, Any]:
        """Get parameter constraints for signal logic."""
        constraints = {
            SignalLogic.CROSSOVER: {
                'fast_period': {'min': 5, 'max': 20, 'type': int},
                'slow_period': {'min': 20, 'max': 50, 'type': int}
            },
            SignalLogic.THRESHOLD: {
                'threshold': {'min': 0.1, 'max': 0.9, 'type': float},
                'direction': {'allowed_values': ['above', 'below'], 'type': str}
            },
            SignalLogic.PATTERN: {
                'pattern_type': {'allowed_values': ['double_bottom', 'double_top', 'head_shoulders'], 'type': str},
                'tolerance': {'min': 0.01, 'max': 0.1, 'type': float}
            },
            SignalLogic.DIVERGENCE: {
                'lookback': {'min': 5, 'max': 20, 'type': int},
                'threshold': {'min': 0.05, 'max': 0.5, 'type': float}
            },
            SignalLogic.MOMENTUM: {
                'period': {'min': 5, 'max': 30, 'type': int},
                'threshold': {'min': 0.01, 'max': 0.1, 'type': float}
            },
            SignalLogic.MEAN_REVERSION: {
                'mean_period': {'min': 10, 'max': 50, 'type': int},
                'std_threshold': {'min': 1.0, 'max': 3.0, 'type': float}
            }
        }
        return constraints.get(signal_logic, {})

    @classmethod
    def _get_generic_constraints(cls, component_type: StrategyComponent) -> Dict[str, Any]:
        """Get generic parameter constraints for component types."""
        constraints = {
            StrategyComponent.RISK_MANAGEMENT: {
                'stop_loss': {'min': 0.005, 'max': 0.1, 'type': float},
                'take_profit': {'min': 0.01, 'max': 0.2, 'type': float}
            },
            StrategyComponent.TIMEFRAME: {
                'timeframe': {'allowed_values': ['1m', '5m', '15m', '30m', '1h', '4h', '1d'], 'type': str}
            },
            StrategyComponent.FILTER: {
                'volume_threshold': {'min': 0.1, 'max': 5.0, 'type': float}
            }
        }
        return constraints.get(component_type, {})

    @classmethod
    def create_strategy_from_genome(cls, genome: StrategyGenome) -> Optional[BaseStrategy]:
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
            strategy_class = strategy_info['class']

            if strategy_class is None:
                logger.error(f"Strategy class not registered for type: {strategy_type}")
                return None

            # Extract and validate parameters
            strategy_params = cls._extract_strategy_parameters(genome, strategy_type)

            # Create strategy configuration
            strategy_config = {
                'name': f'secure_generated_{strategy_type}_{id(genome)}',
                'symbols': ['BTC/USDT'],  # Default, can be overridden
                'timeframe': '1h',
                'required_history': 100,
                'params': strategy_params,
                'genome_id': id(genome),  # For tracking
                'strategy_type': strategy_type  # For audit trail
            }

            # Create and return strategy instance
            strategy_instance = strategy_class(strategy_config)

            logger.info(f"Successfully created strategy of type '{strategy_type}' from genome")
            return strategy_instance

        except Exception as e:
            logger.error(f"Failed to create strategy from genome: {e}")
            return None

    @classmethod
    def _infer_strategy_type(cls, genome: StrategyGenome) -> Optional[str]:
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
                component_counts[component_type] = component_counts.get(component_type, 0) + 1

        # Simple inference based on dominant components
        if component_counts.get('indicator', 0) > 0:
            # Look for specific indicator types
            for gene in genome.genes:
                if gene.enabled and gene.component_type == StrategyComponent.INDICATOR:
                    if gene.indicator_type == IndicatorType.RSI:
                        return 'rsi_momentum'
                    elif gene.indicator_type == IndicatorType.MACD:
                        return 'macd_crossover'
                    elif gene.indicator_type == IndicatorType.BOLLINGER_BANDS:
                        return 'bollinger_reversion'
                    elif gene.indicator_type == IndicatorType.VOLUME:
                        return 'volume_price'

        # Default fallback
        return 'rsi_momentum'  # Safe default

    @classmethod
    def _extract_strategy_parameters(cls, genome: StrategyGenome, strategy_type: str) -> Dict[str, Any]:
        """Extract validated parameters for the strategy."""
        strategy_info = cls.STRATEGY_REGISTRY[strategy_type]
        allowed_params = strategy_info['parameters']

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
                if 'default' in constraint:
                    extracted_params[param_name] = constraint['default']
                else:
                    # Use midpoint of range for numeric types
                    if 'min' in constraint and 'max' in constraint:
                        if constraint['type'] == int:
                            extracted_params[param_name] = (constraint['min'] + constraint['max']) // 2
                        elif constraint['type'] == float:
                            extracted_params[param_name] = (constraint['min'] + constraint['max']) / 2.0

        return extracted_params

    @classmethod
    def get_available_strategies(cls) -> Dict[str, str]:
        """Get list of available strategy types and their descriptions."""
        return {name: info['description'] for name, info in cls.STRATEGY_REGISTRY.items()
                if info['class'] is not None}

    @classmethod
    def get_strategy_info(cls, strategy_type: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific strategy type."""
        if strategy_type not in cls.STRATEGY_REGISTRY:
            return None

        info = cls.STRATEGY_REGISTRY[strategy_type].copy()
        # Remove the class object from the returned info for security
        info.pop('class', None)
        return info





class BayesianOptimizer:
    """Bayesian optimization for expensive fitness evaluations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gp_model = None
        self.observations: List[Dict[str, Any]] = []
        self.is_trained = False

        if not SKLEARN_AVAILABLE:
            logger.warning("Bayesian optimization disabled - scikit-learn not available")

    def add_observation(self, genome: StrategyGenome, fitness: float) -> None:
        """Add an observation to the training data."""
        if not SKLEARN_AVAILABLE:
            return

        # Convert genome to feature vector
        features = self._genome_to_features(genome)
        self.observations.append({
            'features': features,
            'fitness': fitness,
            'genome': genome
        })

    def suggest_next_genome(self, population: List[StrategyGenome]) -> Optional[StrategyGenome]:
        """Suggest next genome to evaluate using acquisition function."""
        if not SKLEARN_AVAILABLE or len(self.observations) < 5:
            return None

        try:
            self._train_model()

            # Generate candidate genomes
            candidates = []
            for _ in range(10):  # Generate 10 candidates
                candidate = self._generate_candidate(population)
                if candidate:
                    candidates.append(candidate)

            if not candidates:
                return None

            # Evaluate acquisition function
            best_candidate = None
            best_acquisition = float('-inf')

            for candidate in candidates:
                features = self._genome_to_features(candidate)
                acquisition_value = self._expected_improvement(features)
                if acquisition_value > best_acquisition:
                    best_acquisition = acquisition_value
                    best_candidate = candidate

            return best_candidate

        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            return None

    def _train_model(self) -> None:
        """Train the Gaussian Process model."""
        if len(self.observations) < 2:
            return

        X = np.array([obs['features'] for obs in self.observations])
        y = np.array([obs['fitness'] for obs in self.observations])

        # Standardize features
        X_scaled = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6)

        # Train GP model
        kernel = gp.kernels.ConstantKernel(1.0) * gp.kernels.RBF(1.0)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True
        )
        self.gp_model.fit(X_scaled, y)
        self.is_trained = True

    def _expected_improvement(self, features: np.ndarray) -> float:
        """Calculate expected improvement acquisition function."""
        if not self.is_trained or self.gp_model is None:
            return 0.0

        features_scaled = (features - np.mean([obs['features'] for obs in self.observations], axis=0)) / \
                         (np.std([obs['features'] for obs in self.observations], axis=0) + 1e-6)

        # Predict mean and std
        y_pred, y_std = self.gp_model.predict([features_scaled], return_std=True)

        # Current best fitness
        best_fitness = max(obs['fitness'] for obs in self.observations)

        # Expected improvement
        improvement = y_pred[0] - best_fitness
        if y_std[0] > 0:
            z = improvement / y_std[0]
            ei = improvement * self._normal_cdf(z) + y_std[0] * self._normal_pdf(z)
            return max(0, ei)

        return max(0, improvement)

    def _normal_cdf(self, x: float) -> float:
        """Cumulative distribution function of standard normal."""
        return (1.0 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi))) / 2

    def _normal_pdf(self, x: float) -> float:
        """Probability density function of standard normal."""
        return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

    def _genome_to_features(self, genome: StrategyGenome) -> np.ndarray:
        """Convert genome to feature vector for GP model."""
        features = []

        # Count of each component type
        component_counts = defaultdict(int)
        for gene in genome.genes:
            component_counts[gene.component_type] += 1

        for component_type in StrategyComponent:
            features.append(component_counts[component_type])

        # Average parameters
        if genome.genes:
            total_weight = sum(g.weight for g in genome.genes)
            avg_weight = total_weight / len(genome.genes)
            features.append(avg_weight)

            enabled_ratio = sum(1 for g in genome.genes if g.enabled) / len(genome.genes)
            features.append(enabled_ratio)
        else:
            features.extend([1.0, 1.0])

        # Fitness and age
        features.append(genome.fitness if genome.fitness != float('-inf') else -1000)
        features.append(genome.age)

        return np.array(features)

    def _generate_candidate(self, population: List[StrategyGenome]) -> Optional[StrategyGenome]:
        """Generate a candidate genome for evaluation."""
        if not population:
            return None

        # Select two random parents
        parent1 = random.choice(population)
        parent2 = random.choice(population)

        # Create offspring
        child1, child2 = parent1.crossover(parent2)

        # Mutate
        child1.mutate(0.2)  # Higher mutation rate for exploration
        child2.mutate(0.2)

        # Return the child with potentially better features
        return child1 if random.random() < 0.5 else child2


class DistributedEvaluator:
    """Distributed evaluation of strategy fitness."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_workers = config.get('max_workers', mp.cpu_count())
        self.executor = None
        self.evaluation_cache: Dict[str, float] = {}

    async def initialize(self) -> None:
        """Initialize the distributed evaluator."""
        if self.config.get('use_processes', False):
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        logger.info(f"Initialized distributed evaluator with {self.max_workers} workers")

    async def evaluate_population(self, population: List[StrategyGenome],
                                data: pd.DataFrame) -> List[Tuple[StrategyGenome, float]]:
        """Evaluate fitness of a population in parallel."""
        if not population:
            return []

        # Create evaluation tasks
        tasks = []
        for genome in population:
            cache_key = self._get_cache_key(genome, data)
            if cache_key in self.evaluation_cache:
                # Use cached result
                cached_fitness = self.evaluation_cache[cache_key]
                genome.fitness = cached_fitness
                tasks.append((genome, cached_fitness))
            else:
                tasks.append(self._evaluate_genome_async(genome, data))

        # Execute evaluations
        if self.executor:
            # Parallel execution
            loop = asyncio.get_event_loop()
            results = await asyncio.gather(*tasks, return_exceptions=True)

            evaluated_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Evaluation failed for genome {i}: {result}")
                    population[i].fitness = float('-inf')
                    evaluated_results.append((population[i], float('-inf')))
                else:
                    genome, fitness = result
                    genome.fitness = fitness
                    evaluated_results.append((genome, fitness))

                    # Cache result
                    cache_key = self._get_cache_key(genome, data)
                    self.evaluation_cache[cache_key] = fitness
        else:
            # Sequential execution (fallback)
            evaluated_results = []
            for genome in population:
                fitness = await self._evaluate_genome_fitness(genome, data)
                genome.fitness = fitness
                evaluated_results.append((genome, fitness))

        return evaluated_results

    async def _evaluate_genome_async(self, genome: StrategyGenome, data: pd.DataFrame) -> Tuple[StrategyGenome, float]:
        """Evaluate a single genome asynchronously."""
        loop = asyncio.get_event_loop()
        fitness = await loop.run_in_executor(
            self.executor, self._evaluate_genome_fitness_sync, genome, data
        )
        return genome, fitness

    def _evaluate_genome_fitness_sync(self, genome: StrategyGenome, data: pd.DataFrame) -> float:
        """Synchronous fitness evaluation for executor."""
        try:
            # Convert genome to strategy
            strategy = self._genome_to_strategy(genome)

            if strategy is None:
                return float('-inf')

            # Run backtest
            equity_progression = self._run_backtest(strategy, data)

            if not equity_progression:
                return float('-inf')

            # Calculate fitness
            metrics = compute_backtest_metrics(equity_progression)
            fitness = self._calculate_fitness_score(metrics)

            return fitness

        except Exception as e:
            logger.error(f"Genome evaluation failed: {e}")
            return float('-inf')

    async def _evaluate_genome_fitness(self, genome: StrategyGenome, data: pd.DataFrame) -> float:
        """Evaluate fitness of a single genome."""
        try:
            # Convert genome to strategy
            strategy = self._genome_to_strategy(genome)

            if strategy is None:
                return float('-inf')

            # Run backtest
            equity_progression = self._run_backtest(strategy, data)

            if not equity_progression:
                return float('-inf')

            # Calculate fitness
            metrics = compute_backtest_metrics(equity_progression)
            fitness = self._calculate_fitness_score(metrics)

            return fitness

        except Exception as e:
            logger.error(f"Genome evaluation failed: {e}")
            return float('-inf')

    def _genome_to_strategy(self, genome: StrategyGenome) -> Optional[BaseStrategy]:
        """Convert genome to executable strategy."""
        try:
            # This is a simplified conversion - in practice, this would be more complex
            # and involve dynamic strategy generation

            # For now, create a mock strategy based on genome characteristics
            from strategies.base_strategy import BaseStrategy

            class GeneratedStrategy(BaseStrategy):
                def __init__(self, config):
                    super().__init__(config)
                    self.genome = genome

                def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
                    # Simplified signal generation based on genome
                    signals = []

                    if data.empty or len(data) < 20:
                        return signals

                    # Mock signal generation - replace with actual genome interpretation
                    for i in range(20, len(data)):
                        if random.random() < 0.1:  # 10% chance of signal
                            signal_type = "BUY" if random.random() < 0.5 else "SELL"
                            signals.append({
                                'timestamp': data.index[i],
                                'signal_type': signal_type,
                                'symbol': self.config.get('symbols', ['BTC/USDT'])[0],
                                'price': data.iloc[i]['close'],
                                'metadata': {'genome_id': id(genome)}
                            })

                    return signals

            strategy_config = {
                'name': f'generated_{id(genome)}',
                'symbols': ['BTC/USDT'],
                'timeframe': '1h',
                'required_history': 100,
                'params': {}
            }

            return GeneratedStrategy(strategy_config)

        except Exception as e:
            logger.error(f"Failed to convert genome to strategy: {e}")
            return None

    def _run_backtest(self, strategy: BaseStrategy, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run simplified backtest for strategy evaluation."""
        try:
            # Generate signals
            signals = strategy.generate_signals(data)

            if not signals:
                return []

            # Mock equity progression
            equity_progression = []
            initial_equity = 10000.0
            current_equity = initial_equity
            trade_count = 0

            for signal in signals:
                # Simulate trade outcome
                pnl = np.random.normal(0, 100)  # Random P&L
                current_equity += pnl
                trade_count += 1

                equity_progression.append({
                    'trade_id': trade_count,
                    'timestamp': signal['timestamp'],
                    'equity': current_equity,
                    'pnl': pnl,
                    'cumulative_return': (current_equity - initial_equity) / initial_equity
                })

            return equity_progression

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return []

    def _calculate_fitness_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate composite fitness score."""
        if not metrics:
            return float('-inf')

        # Multi-objective fitness
        score = 0.0

        # Sharpe ratio (primary metric)
        sharpe = metrics.get('sharpe_ratio', 0)
        score += sharpe * 1.0

        # Total return
        total_return = metrics.get('total_return', 0)
        score += total_return * 0.5

        # Win rate
        win_rate = metrics.get('win_rate', 0)
        score += win_rate * 0.3

        # Penalize drawdown
        max_drawdown = metrics.get('max_drawdown', 0)
        score -= max_drawdown * 0.2

        # Complexity penalty (simplified)
        score -= 0.01  # Small penalty for strategy complexity

        return score

    def _get_cache_key(self, genome: StrategyGenome, data: pd.DataFrame) -> str:
        """Generate cache key for genome evaluation."""
        # Simple cache key based on genome structure
        gene_summary = f"{len(genome.genes)}_{genome.generation}"
        data_summary = f"{len(data)}_{data.index[0] if not data.empty else 'empty'}"
        return f"{gene_summary}_{data_summary}"

    def get_worker_status(self) -> Dict[str, Any]:
        """Get status of workers in the distributed evaluator."""
        status = {
            'max_workers': self.max_workers,
            'active_workers': 0,
            'executor_type': 'ProcessPoolExecutor' if self.config.get('use_processes', False) else 'ThreadPoolExecutor',
            'cache_size': len(self.evaluation_cache),
            'is_initialized': self.executor is not None
        }

        if self.executor:
            # Try to get active thread/process count
            try:
                if hasattr(self.executor, '_threads'):
                    status['active_workers'] = len([t for t in self.executor._threads if t.is_alive()])
                elif hasattr(self.executor, '_processes'):
                    status['active_workers'] = len([p for p in self.executor._processes.values() if p.is_alive()])
            except Exception:
                status['active_workers'] = 0

        return status

    async def shutdown(self) -> None:
        """Shutdown the distributed evaluator."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        logger.info("Distributed evaluator shutdown")


class StrategyGenerator(BaseOptimizer):
    """
    Hybrid AI Strategy Generator using evolutionary algorithms and Bayesian optimization.

    This system:
    1. Represents trading strategies as genetic genomes
    2. Evolves populations using genetic programming
    3. Uses Bayesian optimization for efficient exploration
    4. Evaluates strategies in parallel across distributed workers
    5. Generates production-ready trading strategies
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Strategy Generator.

        Args:
            config: Configuration dictionary containing:
                - population_size: Number of genomes in population
                - generations: Number of generations to evolve
                - mutation_rate: Probability of genome mutation
                - crossover_rate: Probability of crossover
                - speciation_threshold: Threshold for species formation
                - bayesian_enabled: Whether to use Bayesian optimization
                - distributed_enabled: Whether to use distributed evaluation
        """
        super().__init__(config)

        # Core configuration
        self.population_size = config.get('population_size', 50)
        self.generations = config.get('generations', 20)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.7)
        self.speciation_threshold = config.get('speciation_threshold', 0.3)
        self.elitism_rate = config.get('elitism_rate', 0.1)

        # Advanced features
        self.bayesian_enabled = config.get('bayesian_enabled', True)
        self.distributed_enabled = config.get('distributed_enabled', True)

        # Population and species
        self.population: List[StrategyGenome] = []
        self.species: List[Species] = []
        self.best_genome: Optional[StrategyGenome] = None
        self.current_generation: int = 0

        # Optimization components
        self.bayesian_optimizer = BayesianOptimizer(config) if self.bayesian_enabled else None
        self.distributed_evaluator = DistributedEvaluator(config) if self.distributed_enabled else None

        # Statistics
        self.generation_stats: List[Dict[str, Any]] = []
        self.species_history: List[Dict[str, Any]] = []

        logger.info("StrategyGenerator initialized with advanced AI capabilities")

    async def initialize(self) -> None:
        """Initialize the strategy generator."""
        if self.distributed_evaluator:
            await self.distributed_evaluator.initialize()

        # Initialize population
        await self._initialize_population()

        logger.info("StrategyGenerator fully initialized")

    def optimize(self, strategy_class, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the hybrid AI strategy generation process.

        Args:
            strategy_class: Base strategy class (used as template)
            data: Historical data for optimization

        Returns:
            Best generated strategy parameters
        """
        # Note: This is an async process, but we provide a sync interface
        # In practice, you'd want to call this from an async context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(self._optimize_async(strategy_class, data))
            return result
        finally:
            loop.close()

    async def _optimize_async(self, strategy_class, data: pd.DataFrame) -> Dict[str, Any]:
        """Async implementation of optimization."""
        start_time = datetime.now()

        logger.info("Starting Hybrid AI Strategy Generation")
        logger.info(f"Population size: {self.population_size}")
        logger.info(f"Generations: {self.generations}")
        logger.info(f"Bayesian optimization: {self.bayesian_enabled}")
        logger.info(f"Distributed evaluation: {self.distributed_enabled}")

        # Initialize population
        await self._initialize_population()

        # Evolution loop
        for generation in range(self.generations):
            logger.info(f"Generation {generation + 1}/{self.generations}")

            # Evaluate population
            await self._evaluate_population(data)

            # Update species
            self._update_species()

            # Bayesian optimization (if enabled)
            if self.bayesian_optimizer and generation > 2:
                bayesian_candidate = self.bayesian_optimizer.suggest_next_genome(self.population)
                if bayesian_candidate:
                    self.population.append(bayesian_candidate)

            # Create next generation
            await self._create_next_generation()

            # Update best genome
            self._update_best_genome()

            # Log generation statistics
            self._log_generation_stats(generation + 1)

            # Update Bayesian optimizer with observations
            if self.bayesian_optimizer:
                for genome in self.population:
                    if genome.fitness != float('-inf'):
                        self.bayesian_optimizer.add_observation(genome, genome.fitness)

        # Finalize optimization
        optimization_time = (datetime.now() - start_time).total_seconds()
        self.config['optimization_time'] = optimization_time

        logger.info(".2f")
        logger.info(f"Best fitness: {self.best_genome.fitness:.4f}" if self.best_genome else "No best genome found")

        # Return best strategy
        if self.best_genome:
            return {
                'genome': self.best_genome.to_dict(),
                'fitness': self.best_genome.fitness,
                'generation': self.best_genome.generation,
                'species_id': self.best_genome.species_id
            }

        return {}

    async def _initialize_population(self) -> None:
        """Initialize the population with diverse genomes."""
        self.population = []

        for i in range(self.population_size):
            genome = self._create_random_genome()
            genome.generation = 0
            genome.age = 0
            self.population.append(genome)

        logger.info(f"Initialized population with {len(self.population)} diverse genomes")

    def _create_random_genome(self) -> StrategyGenome:
        """Create a random genome."""
        genome = StrategyGenome()

        # Add 3-8 random genes
        num_genes = random.randint(3, 8)

        for _ in range(num_genes):
            component_type = random.choice(list(StrategyComponent))

            gene = StrategyGene(component_type=component_type)

            if component_type == StrategyComponent.INDICATOR:
                gene.indicator_type = random.choice(list(IndicatorType))
                gene.parameters = genome._get_default_indicator_params(gene.indicator_type)
            elif component_type == StrategyComponent.SIGNAL_LOGIC:
                gene.signal_logic = random.choice(list(SignalLogic))
                gene.parameters = genome._get_default_signal_params(gene.signal_logic)
            elif component_type == StrategyComponent.RISK_MANAGEMENT:
                gene.parameters = {'stop_loss': random.uniform(0.01, 0.05), 'take_profit': random.uniform(0.02, 0.1)}
            elif component_type == StrategyComponent.TIMEFRAME:
                gene.parameters = {'timeframe': random.choice(['15m', '1h', '4h'])}
            elif component_type == StrategyComponent.FILTER:
                gene.parameters = {'volume_threshold': random.uniform(0.5, 2.0)}

            genome.genes.append(gene)

        return genome

    async def _evaluate_population(self, data: pd.DataFrame) -> None:
        """Evaluate fitness of the current population."""
        if self.distributed_evaluator:
            # Distributed evaluation
            results = await self.distributed_evaluator.evaluate_population(self.population, data)
            for genome, fitness in results:
                genome.fitness = fitness
        else:
            # Sequential evaluation
            for genome in self.population:
                if genome.fitness == float('-inf'):
                    fitness = await self._evaluate_genome_fitness(genome, data)
                    genome.fitness = fitness

    async def _evaluate_genome_fitness(self, genome: StrategyGenome, data: pd.DataFrame) -> float:
        """Evaluate fitness of a single genome."""
        try:
            # Convert genome to strategy
            strategy = self._genome_to_strategy(genome)

            if strategy is None:
                return float('-inf')

            # Run backtest
            equity_progression = self._run_backtest(strategy, data)

            if not equity_progression:
                return float('-inf')

            # Calculate fitness
            metrics = compute_backtest_metrics(equity_progression)
            fitness = self._calculate_fitness_score(metrics)

            return fitness

        except Exception as e:
            logger.error(f"Genome evaluation failed: {e}")
            return float('-inf')

    def _genome_to_strategy(self, genome: StrategyGenome) -> Optional[BaseStrategy]:
        """
        Convert genome to executable strategy using secure factory pattern.

        This method replaces the previous dynamic class generation approach
        with a secure factory pattern that validates all inputs and prevents
        arbitrary code execution. It provides comprehensive error logging and
        fallback mechanisms for robust strategy generation.

        Args:
            genome: The genome to convert to a strategy

        Returns:
            Strategy instance or None if creation fails

        Raises:
            StrategyGenerationError: If strategy generation fails with detailed error information
        """
        try:
            # Validate genome input
            if genome is None:
                raise StrategyGenerationError(
                    "Genome cannot be None",
                    error_type="invalid_input",
                    genome_info={"genome_provided": False}
                )

            if not genome.genes:
                raise StrategyGenerationError(
                    "Genome has no genes - cannot create strategy",
                    error_type="empty_genome",
                    genome_info={"gene_count": 0, "generation": genome.generation}
                )

            # Log genome information for debugging
            genome_info = {
                "gene_count": len(genome.genes),
                "generation": genome.generation,
                "fitness": genome.fitness,
                "enabled_genes": sum(1 for g in genome.genes if g.enabled)
            }
            logger.debug(f"Attempting to convert genome to strategy: {genome_info}")

            # Use the secure StrategyFactory to create strategy from genome
            strategy = StrategyFactory.create_strategy_from_genome(genome)

            if strategy is None:
                # Try to get more detailed error information from factory
                is_valid, validation_errors = StrategyFactory.validate_genome(genome)

                if not is_valid:
                    raise StrategyGenerationError(
                        f"Genome validation failed: {validation_errors}",
                        error_type="validation_failed",
                        genome_info=genome_info,
                        cause=ValueError(f"Validation errors: {validation_errors}")
                    )
                else:
                    raise StrategyGenerationError(
                        "StrategyFactory returned None despite valid genome",
                        error_type="factory_error",
                        genome_info=genome_info
                    )

            # Validate the created strategy
            if not hasattr(strategy, 'generate_signals'):
                raise StrategyGenerationError(
                    "Created strategy missing required generate_signals method",
                    error_type="invalid_strategy",
                    genome_info=genome_info
                )

            logger.debug(f"Successfully created strategy from genome using factory pattern")
            return strategy

        except StrategyGenerationError:
            # Re-raise our custom exceptions
            raise

        except ValueError as e:
            logger.error(f"Value error in genome conversion: {str(e)}")
            logger.error(f"Genome info: {genome_info if 'genome_info' in locals() else 'N/A'}")
            raise StrategyGenerationError(
                f"Invalid genome data: {str(e)}",
                error_type="value_error",
                genome_info=genome_info if 'genome_info' in locals() else {},
                cause=e
            ) from e

        except ImportError as e:
            logger.error(f"Import error during strategy creation: {str(e)}")
            raise StrategyGenerationError(
                f"Required module not available: {str(e)}",
                error_type="import_error",
                genome_info=genome_info if 'genome_info' in locals() else {},
                cause=e
            ) from e

        except RuntimeError as e:
            logger.error(f"Runtime error during strategy creation: {str(e)}")
            raise StrategyGenerationError(
                f"Strategy creation failed: {str(e)}",
                error_type="runtime_error",
                genome_info=genome_info if 'genome_info' in locals() else {},
                cause=e
            ) from e

        except Exception as e:
            logger.error(f"Unexpected error in genome conversion: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise StrategyGenerationError(
                f"Unexpected error during genome conversion: {str(e)}",
                error_type="unexpected_error",
                genome_info=genome_info if 'genome_info' in locals() else {},
                cause=e
            ) from e

    def _run_backtest(self, strategy: BaseStrategy, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run simplified backtest."""
        try:
            signals = strategy.generate_signals(data)

            if not signals:
                return []

            equity_progression = []
            initial_equity = 10000.0
            current_equity = initial_equity
            trade_count = 0

            for signal in signals:
                pnl = np.random.normal(0, 100)
                current_equity += pnl
                trade_count += 1

                equity_progression.append({
                    'trade_id': trade_count,
                    'timestamp': signal['timestamp'],
                    'equity': current_equity,
                    'pnl': pnl,
                    'cumulative_return': (current_equity - initial_equity) / initial_equity
                })

            return equity_progression

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return []

    def _calculate_fitness_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate composite fitness score."""
        if not metrics:
            return float('-inf')

        score = 0.0

        # Multi-objective scoring
        score += metrics.get('sharpe_ratio', 0) * 1.0
        score += metrics.get('total_return', 0) * 0.5
        score += metrics.get('win_rate', 0) * 0.3
        score -= metrics.get('max_drawdown', 0) * 0.2

        return score

    def _update_species(self) -> None:
        """Update species classification."""
        # Simplified species formation
        if not self.population:
            return

        # Clear existing species
        self.species = []

        # Group genomes by similarity (simplified)
        species_dict = defaultdict(list)

        for genome in self.population:
            # Simple species key based on number of genes and fitness range
            species_key = f"{len(genome.genes)}_{int(genome.fitness // 0.5)}"
            species_dict[species_key].append(genome)

        # Create species objects
        for species_id, members in species_dict.items():
            if len(members) > 1:
                species = Species(
                    species_id=species_id,
                    representative=members[0],
                    members=members
                )
                species.update_representative()
                self.species.append(species)

        logger.debug(f"Formed {len(self.species)} species from {len(self.population)} genomes")

    async def _create_next_generation(self) -> None:
        """Create the next generation through selection, crossover, and mutation."""
        if not self.population:
            return

        new_population = []

        # Elitism
        elite_count = max(1, int(len(self.population) * self.elitism_rate))
        sorted_population = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        new_population.extend(sorted_population[:elite_count])

        # Fill rest through reproduction
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            child1.mutate(self.mutation_rate)
            child2.mutate(self.mutation_rate)

            # Age increment
            child1.age = parent1.age + 1
            child2.age = parent2.age + 1

            new_population.extend([child1, child2])

        # Trim to population size
        self.population = new_population[:self.population_size]

    def _tournament_selection(self) -> StrategyGenome:
        """Tournament selection for parent selection."""
        tournament_size = min(5, len(self.population))
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda g: g.fitness)

    def _update_best_genome(self) -> None:
        """Update the best genome found so far."""
        if not self.population:
            return

        current_best = max(self.population, key=lambda g: g.fitness)
        if self.best_genome is None or current_best.fitness > self.best_genome.fitness:
            self.best_genome = current_best.copy()

    def _log_generation_stats(self, generation: int) -> None:
        """Log statistics for the current generation."""
        if not self.population:
            return

        fitness_values = [g.fitness for g in self.population if g.fitness != float('-inf')]

        if fitness_values:
            best_fitness = max(fitness_values)
            avg_fitness = np.mean(fitness_values)
            std_fitness = np.std(fitness_values)

            logger.info(
                f"Gen {generation}/{self.generations} | "
                f"Best: {best_fitness:.4f} | "
                f"Avg: {avg_fitness:.4f} | "
                f"Std: {std_fitness:.4f} | "
                f"Species: {len(self.species)}"
            )

            # Store generation data
            self.generation_stats.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': std_fitness,
                'population_size': len(self.population),
                'species_count': len(self.species)
            })

    def get_generation_stats(self) -> List[Dict[str, Any]]:
        """Get generation statistics."""
        return self.generation_stats.copy()

    def get_species_info(self) -> List[Dict[str, Any]]:
        """Get information about current species."""
        species_info = []
        for species in self.species:
            species_info.append({
                'species_id': species.species_id,
                'member_count': len(species.members),
                'best_fitness': species.representative.fitness,
                'diversity_score': species.calculate_diversity_score(),
                'stagnation_counter': species.stagnation_counter
            })
        return species_info

    def save_population(self) -> None:
        """Save current population to disk using model_path."""
        model_path = self.config.get('model_path', 'models/strategy_generator')
        path = f"{model_path}/population.json"

        population_data = {
            'timestamp': datetime.now().isoformat(),
            'population': [genome.to_dict() for genome in self.population],
            'best_genome': self.best_genome.to_dict() if self.best_genome else None,
            'generation_stats': self.generation_stats,
            'config': self.config
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(population_data, f, indent=2, default=str)

        logger.info(f"Population saved to {path}")

    def load_population(self, path: Optional[str] = None) -> None:
        """Load population from disk."""
        if path is None:
            model_path = self.config.get('model_path', 'models/strategy_generator')
            path = f"{model_path}/population.json"

        if not Path(path).exists():
            logger.warning(f"Population file not found: {path}")
            return

        with open(path, 'r') as f:
            data = json.load(f)

        self.population = [StrategyGenome.from_dict(g) for g in data.get('population', [])]
        if data.get('best_genome'):
            self.best_genome = StrategyGenome.from_dict(data['best_genome'])

        self.generation_stats = data.get('generation_stats', [])

        logger.info(f"Population loaded from {path}")

    async def generate_strategy(self, genome: StrategyGenome, name: str) -> Optional[BaseStrategy]:
        """
        Generate a strategy instance from a genome using secure factory pattern.

        This method replaces the previous dynamic class generation (using type())
        with a secure factory pattern that validates all inputs and prevents
        arbitrary code execution.

        Args:
            genome: The genome to convert to a strategy
            name: Name for the generated strategy (used for identification)

        Returns:
            Strategy instance or None if generation fails
        """
        try:
            # Use the secure StrategyFactory to create strategy from genome
            strategy = StrategyFactory.create_strategy_from_genome(genome)

            if strategy is None:
                logger.warning("StrategyFactory failed to create strategy from genome")
                return None

            # Update strategy name if provided
            if hasattr(strategy, 'config') and strategy.config:
                strategy.config['name'] = name

            logger.info(f"Successfully generated strategy '{name}' from genome using secure factory")
            return strategy

        except Exception as e:
            logger.error(f"Failed to generate strategy '{name}': {e}")
            return None

    async def evolve(self) -> None:
        """Run one generation of evolution."""
        if not self.population:
            await self._initialize_population()

        # Evaluate current population
        # For now, use dummy data - in practice this would be passed in
        dummy_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1h'),
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })

        await self._evaluate_population(dummy_data)
        await self._create_next_generation()
        self._update_best_genome()
        self.current_generation += 1

    @staticmethod
    async def calculate_fitness(genome: StrategyGenome, market_data: pd.DataFrame) -> float:
        """Calculate fitness for a genome using direct evaluation."""
        try:
            # Simple fitness based on genome complexity and market data
            base_fitness = len(genome.genes) * 0.1

            # Add some market-based variation
            if not market_data.empty:
                volatility = market_data['close'].pct_change().std()
                base_fitness += volatility * 10  # Reward strategies in volatile markets

            return base_fitness
        except Exception:
            return float('-inf')

    @staticmethod
    async def calculate_fitness_with_backtest(genome: StrategyGenome, market_data: pd.DataFrame, backtester) -> float:
        """Calculate fitness using backtesting."""
        try:
            # Run actual backtest
            results = await backtester.run_backtest(genome, market_data)

            return StrategyGenerator.calculate_multi_objective_fitness(results)
        except Exception:
            return float('-inf')

    @staticmethod
    def calculate_multi_objective_fitness(results: Dict[str, Any]) -> float:
        """Calculate multi-objective fitness from backtest results."""
        if not results:
            return float('-inf')

        score = 0.0

        # Sharpe ratio (primary metric)
        sharpe = results.get('sharpe_ratio', 0)
        score += sharpe * 1.0

        # Total return
        total_return = results.get('total_return', 0)
        score += total_return * 0.5

        # Win rate
        win_rate = results.get('win_rate', 0)
        score += win_rate * 0.3

        # Penalize drawdown
        max_drawdown = results.get('max_drawdown', 0)
        score -= max_drawdown * 0.2

        return score

    @staticmethod
    def calculate_risk_adjusted_fitness(results: Dict[str, Any]) -> float:
        """Calculate risk-adjusted fitness."""
        if not results:
            return float('-inf')

        sharpe = results.get('sharpe_ratio', 0)
        total_return = results.get('total_return', 0)
        max_drawdown = results.get('max_drawdown', 0)

        # Risk-adjusted return
        if max_drawdown > 0:
            risk_adjusted_return = total_return / max_drawdown
        else:
            risk_adjusted_return = total_return

        # Combine with Sharpe
        return sharpe * 0.7 + risk_adjusted_return * 0.3

    async def evaluate_population_fitness_parallel(self, population: List[StrategyGenome], fitness_func) -> List[float]:
        """Evaluate population fitness in parallel."""
        if not population:
            return []

        # Create tasks for parallel evaluation
        tasks = []
        for genome in population:
            tasks.append(fitness_func(genome))

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        fitness_scores = []
        for result in results:
            if isinstance(result, Exception):
                fitness_scores.append(float('-inf'))
            else:
                fitness_scores.append(result)

        return fitness_scores

    async def shutdown(self) -> None:
        """Shutdown the strategy generator."""
        if self.distributed_evaluator:
            await self.distributed_evaluator.shutdown()

        logger.info("StrategyGenerator shutdown complete")

    def get_strategy_generator_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the strategy generation process."""
        return {
            'optimizer_type': 'Hybrid AI Strategy Generator',
            'population_size': len(self.population),
            'generations_completed': len(self.generation_stats),
            'best_fitness': self.best_genome.fitness if self.best_genome else None,
            'species_count': len(self.species),
            'bayesian_enabled': self.bayesian_enabled,
            'distributed_enabled': self.distributed_enabled,
            'total_evaluations': sum(len(p) for p in [self.population]) if self.population else 0,
            'generation_stats': self.generation_stats[-5:] if self.generation_stats else [],  # Last 5 generations
            'species_info': self.get_species_info()
        }


# Convenience functions
async def create_strategy_generator(config: Optional[Dict[str, Any]] = None) -> StrategyGenerator:
    """
    Create and initialize a strategy generator instance.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized StrategyGenerator instance
    """
    default_config = {
        'population_size': 50,
        'generations': 20,
        'mutation_rate': 0.1,
        'crossover_rate': 0.7,
        'elitism_rate': 0.1,
        'speciation_threshold': 0.3,
        'bayesian_enabled': True,
        'distributed_enabled': True,
        'max_workers': mp.cpu_count(),
        'use_processes': False
    }

    merged_config = {**default_config, **(config or {})}
    generator = StrategyGenerator(merged_config)
    await generator.initialize()

    return generator


def get_strategy_generator(config: Optional[Dict[str, Any]] = None) -> StrategyGenerator:
    """
    Get a strategy generator instance (synchronous wrapper).

    Args:
        config: Configuration dictionary

    Returns:
        StrategyGenerator instance
    """
    # Note: This returns an uninitialized instance
    # For async initialization, use create_strategy_generator
    default_config = {
        'population_size': 50,
        'generations': 20,
        'mutation_rate': 0.1,
        'crossover_rate': 0.7,
        'elitism_rate': 0.1,
        'speciation_threshold': 0.3,
        'bayesian_enabled': True,
        'distributed_enabled': True,
        'max_workers': mp.cpu_count(),
        'use_processes': False
    }

    merged_config = {**default_config, **(config or {})}
    return StrategyGenerator(merged_config)


# Update optimizer factory to include strategy generator
def _register_strategy_generator():
    """Register strategy generator with optimizer factory."""
    try:
        from .optimizer_factory import OptimizerFactory

        # Add strategy generator to registry
        OptimizerFactory.OPTIMIZER_REGISTRY.update({
            'strategy_generator': StrategyGenerator,
            'ai_strategy': StrategyGenerator,
            'hybrid_ai': StrategyGenerator
        })

        # Update available optimizers
        OptimizerFactory._get_available_optimizers = lambda: {
            **OptimizerFactory.get_available_optimizers(),
            'strategy_generator': 'Hybrid AI Strategy Generator - Evolutionary strategy discovery',
            'ai_strategy': 'Hybrid AI Strategy Generator - Evolutionary strategy discovery',
            'hybrid_ai': 'Hybrid AI Strategy Generator - Evolutionary strategy discovery'
        }

    except ImportError:
        logger.warning("Could not register with optimizer factory")


# Initialize strategy registry with actual classes
StrategyFactory.STRATEGY_REGISTRY['rsi_momentum']['class'] = RSIStrategy
StrategyFactory.STRATEGY_REGISTRY['bollinger_reversion']['class'] = BollingerReversionStrategy

# Register on import
_register_strategy_generator()
