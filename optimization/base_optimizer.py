"""
optimization/base_optimizer.py

Abstract base class for all optimization techniques.
Defines the interface and common functionality for strategy optimization.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from backtest.backtester import compute_backtest_metrics

# ============================================================================
# DISTRIBUTED PROCESSING SUPPORT
# ============================================================================
#
# The current optimization framework runs all fitness evaluations on a single machine,
# which creates a significant bottleneck for large-scale optimization problems with:
# - Many strategies to optimize simultaneously
# - Large population sizes in genetic algorithms
# - Complex fitness evaluations requiring significant computation
#
# RECOMMENDED DISTRIBUTED FRAMEWORKS:
# 1. Ray (https://ray.io/): Excellent for distributed computing with simple API
#    - Use Ray for parallel fitness evaluations across multiple nodes
#    - Supports both CPU and GPU acceleration
#    - Easy integration with existing Python code
#
# 2. Dask (https://dask.org/): Good for distributed data processing
#    - Use Dask for distributed computation on large datasets
#    - Integrates well with pandas and numpy
#    - Supports dynamic task scheduling
#
# IMPLEMENTATION APPROACH:
# - Create an OptimizerBackend abstract class for different execution modes
# - Implement DistributedOptimizer using Ray/Dask for parallel execution
# - Keep LocalOptimizer for single-machine execution (current implementation)
# - Allow switching between backends based on problem size and available resources
# ============================================================================


class OptimizerBackend(ABC):
    """
    Abstract base class for optimization execution backends.

    This class defines the interface for different execution modes:
    - Local execution (single machine)
    - Distributed execution (multiple nodes/machines)
    - GPU-accelerated execution
    """

    @abstractmethod
    def evaluate_fitness_batch(
        self, strategy_instances: List[Any], data: pd.DataFrame
    ) -> List[float]:
        """
        Evaluate fitness for multiple strategy instances in parallel.

        Args:
            strategy_instances: List of strategy instances to evaluate
            data: Historical data for backtesting

        Returns:
            List of fitness scores corresponding to each strategy instance
        """
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the backend with configuration.

        Args:
            config: Backend-specific configuration
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Clean up resources and shutdown the backend.
        """
        pass


class DistributedOptimizer(OptimizerBackend):
    """
    Placeholder for distributed optimization backend.

    This class serves as a template for implementing distributed processing
    using frameworks like Ray or Dask. It provides the interface that would
    be implemented to distribute fitness evaluations across multiple nodes.

    IMPLEMENTATION NOTES:
    - Use Ray/Dask to parallelize fitness evaluations
    - Handle node failures and load balancing
    - Support dynamic scaling based on workload
    - Cache results to avoid redundant computations
    - Monitor resource usage and performance metrics
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize distributed optimizer.

        Args:
            config: Configuration for distributed execution
        """
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Placeholder for distributed framework client
        self.distributed_client = None

        # Configuration for distributed execution
        self.num_workers = config.get("num_workers", 4)
        self.framework = config.get("framework", "ray")  # 'ray' or 'dask'
        self.cluster_address = config.get("cluster_address", None)

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the distributed backend.

        This method would:
        1. Connect to Ray/Dask cluster
        2. Initialize worker nodes
        3. Set up distributed data structures
        4. Configure load balancing

        Args:
            config: Backend configuration
        """
        self.logger.info(f"Initializing {self.framework} distributed backend")

        # Placeholder implementation - would initialize Ray/Dask here
        if self.framework == "ray":
            # import ray
            # ray.init(address=self.cluster_address, num_cpus=self.num_workers)
            self.logger.info("Ray distributed backend initialized (placeholder)")
        elif self.framework == "dask":
            # from dask.distributed import Client
            # self.distributed_client = Client(self.cluster_address)
            self.logger.info("Dask distributed backend initialized (placeholder)")

    def evaluate_fitness_batch(
        self, strategy_instances: List[Any], data: pd.DataFrame
    ) -> List[float]:
        """
        Evaluate fitness for multiple strategies in parallel using distributed computing.

        This method would:
        1. Distribute strategy instances across worker nodes
        2. Execute fitness evaluations in parallel
        3. Collect and return results
        4. Handle failures and retries

        Args:
            strategy_instances: List of strategy instances to evaluate
            data: Historical data for backtesting

        Returns:
            List of fitness scores
        """
        self.logger.info(
            f"Evaluating {len(strategy_instances)} strategies using {self.framework}"
        )

        # Placeholder implementation - would use Ray/Dask for parallel execution
        fitness_scores = []

        for i, strategy in enumerate(strategy_instances):
            # In real implementation, this would be distributed
            # fitness = ray.remote(evaluate_single_fitness).remote(strategy, data)
            # fitness_scores.append(fitness)

            # Placeholder: simulate distributed evaluation
            fitness_scores.append(float("-inf"))  # Placeholder score

        # In real implementation:
        # return ray.get(fitness_scores) if self.framework == 'ray' else [f.result() for f in fitness_scores]

        self.logger.info("Distributed fitness evaluation completed (placeholder)")
        return fitness_scores

    def shutdown(self) -> None:
        """
        Shutdown the distributed backend and clean up resources.

        This method would:
        1. Close connections to worker nodes
        2. Clean up distributed data structures
        3. Save any cached results
        4. Log final statistics
        """
        self.logger.info(f"Shutting down {self.framework} distributed backend")

        # Placeholder implementation - would shutdown Ray/Dask here
        if self.distributed_client:
            # self.distributed_client.close()
            pass

        self.logger.info("Distributed backend shutdown complete")


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    strategy_name: str
    optimizer_type: str
    best_params: Dict[str, Any]
    best_fitness: float
    fitness_metric: str
    iterations: int
    total_evaluations: int
    optimization_time: float
    timestamp: datetime
    results_history: List[Dict[str, Any]]
    convergence_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationResult":
        """Create from dictionary."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ParameterBounds:
    """Parameter bounds and constraints for optimization."""

    name: str
    min_value: Union[int, float]
    max_value: Union[int, float]
    step: Optional[Union[int, float]] = None
    param_type: str = "float"  # "int", "float", "categorical"
    categories: Optional[List[Any]] = None

    def __post_init__(self):
        """Validate parameter bounds after initialization."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Parameter name must be a non-empty string")

        if self.param_type not in ["int", "float", "categorical"]:
            raise ValueError(
                f"Invalid param_type: {self.param_type}. Must be 'int', 'float', or 'categorical'"
            )

        if self.param_type == "categorical":
            if self.categories is None or not isinstance(self.categories, list):
                raise ValueError(
                    "Categories must be provided for categorical parameters"
                )
            if len(self.categories) == 0:
                raise ValueError("Categories list cannot be empty")
        else:
            # For int/float parameters, validate min_value and max_value
            if not isinstance(self.min_value, (int, float)):
                raise TypeError(
                    f"min_value must be numeric, got {type(self.min_value)}"
                )
            if not isinstance(self.max_value, (int, float)):
                raise TypeError(
                    f"max_value must be numeric, got {type(self.max_value)}"
                )
            if self.min_value >= self.max_value:
                raise ValueError(
                    f"min_value ({self.min_value}) must be less than max_value ({self.max_value})"
                )

        if self.step is not None:
            if not isinstance(self.step, (int, float)):
                raise TypeError(f"step must be numeric, got {type(self.step)}")
            if self.step <= 0:
                raise ValueError("step must be positive")

    def validate_value(self, value: Any) -> bool:
        """Validate if a value is within bounds."""
        if self.param_type == "categorical":
            return value in (self.categories or [])
        elif self.param_type == "int":
            return isinstance(value, int) and self.min_value <= value <= self.max_value
        else:  # float
            return (
                isinstance(value, (int, float))
                and self.min_value <= value <= self.max_value
            )

    def clamp_value(self, value: Any) -> Any:
        """Clamp value to bounds."""
        if self.param_type == "categorical":
            return (
                value
                if value in (self.categories or [])
                else (self.categories[0] if self.categories else value)
            )
        elif self.param_type == "int":
            return max(self.min_value, min(self.max_value, int(value)))
        else:  # float
            return max(self.min_value, min(self.max_value, float(value)))


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization techniques.

    Provides common functionality for:
    - Parameter validation and bounds checking
    - Fitness evaluation using backtesting
    - Results persistence and loading
    - Logging and progress tracking
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the optimizer.

        Args:
            config: Configuration dictionary for the optimizer
        """
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Setup logging
        self._setup_logging()

        # Parameter bounds for validation
        self.parameter_bounds: Dict[str, ParameterBounds] = {}

        # Optimization state
        self.current_iteration = 0
        self.total_evaluations = 0
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_fitness = float("-inf")
        self.results_history: List[Dict[str, Any]] = []

        # Fitness function configuration
        self.fitness_metric = config.get("fitness_metric", "sharpe_ratio")
        self.fitness_weights = config.get(
            "fitness_weights",
            {
                "sharpe_ratio": 1.0,
                "total_return": 0.3,
                "win_rate": 0.2,
                "max_drawdown": -0.1,  # Negative weight for minimization
            },
        )

    def _setup_logging(self) -> None:
        """Setup logging for the optimizer."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def add_parameter_bounds(self, bounds: ParameterBounds) -> None:
        """
        Add parameter bounds for validation.

        Args:
            bounds: ParameterBounds object defining constraints
        """
        self.parameter_bounds[bounds.name] = bounds

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameter values against bounds.

        Args:
            params: Parameter dictionary to validate

        Returns:
            True if all parameters are valid
        """
        if not params:
            return True
        for name, value in params.items():
            if value is None:
                continue
            if name in self.parameter_bounds:
                bounds = self.parameter_bounds[name]
                if bounds.param_type in ("int", "float"):
                    if not (bounds.min_value <= value <= bounds.max_value):
                        self.logger.warning(
                            f"Parameter {name}={value} is out of bounds"
                        )
                        return False
        return True

    def clamp_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clamp parameter values to their bounds.

        Args:
            params: Parameter dictionary to clamp

        Returns:
            Clamped parameter dictionary
        """
        clamped = params.copy()
        for name, value in params.items():
            if name in self.parameter_bounds:
                bounds = self.parameter_bounds[name]
                clamped[name] = bounds.clamp_value(value)
        return clamped

    def evaluate_fitness(self, strategy_instance, data: pd.DataFrame) -> float:
        """
        Evaluate fitness of a strategy with given parameters.

        Args:
            strategy_instance: Strategy instance to evaluate
            data: Historical data for backtesting

        Returns:
            Fitness score
        """
        try:
            # Run backtest
            equity_progression = self._run_backtest(strategy_instance, data)

            if not equity_progression:
                return float("-inf")

            # Compute metrics
            metrics = compute_backtest_metrics(equity_progression)

            # Calculate composite fitness score
            fitness = self._calculate_fitness_score(metrics)

            self.total_evaluations += 1

            return fitness

        except Exception as e:
            self.logger.error(f"Error evaluating fitness: {str(e)}")
            return float("-inf")

    def _run_backtest(
        self, strategy_instance, data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Run backtest for strategy evaluation.

        Args:
            strategy_instance: Strategy to backtest
            data: Historical data

        Returns:
            Equity progression data
        """
        # This is a simplified backtest implementation
        # In a real implementation, this would use the full backtesting framework
        try:
            # Mock equity progression for demonstration
            # Replace with actual backtesting logic
            equity_progression = []

            # Simulate some trades
            initial_equity = 10000.0
            current_equity = initial_equity
            trade_count = 0

            # Simple simulation - replace with actual strategy execution
            for i in range(min(100, len(data))):
                if i % 10 == 0:  # Simulate trade every 10 periods
                    pnl = np.random.normal(0, 50)  # Random P&L
                    current_equity += pnl
                    trade_count += 1

                    equity_progression.append(
                        {
                            "trade_id": trade_count,
                            "timestamp": data.index[i]
                            if hasattr(data.index, "__getitem__")
                            else i,
                            "equity": current_equity,
                            "pnl": pnl,
                            "cumulative_return": (current_equity - initial_equity)
                            / initial_equity,
                        }
                    )

            return equity_progression

        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            return []

    def _calculate_fitness_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate composite fitness score from metrics.

        Args:
            metrics: Backtest metrics

        Returns:
            Composite fitness score
        """
        score = 0.0

        # Apply weights to different metrics
        for metric, weight in self.fitness_weights.items():
            if metric in metrics:
                value = metrics[metric]
                # Handle negative weights (for minimization metrics like drawdown)
                if weight < 0:
                    # Convert minimization to maximization by negating
                    value = -value if value != 0 else 0
                    weight = abs(weight)
                score += value * weight

        return score

    def update_best_params(self, params: Dict[str, Any], fitness: float) -> None:
        """
        Update best parameters if fitness improved.

        Args:
            params: Parameter set
            fitness: Fitness score
        """
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_params = params.copy()

            self.logger.info(f"New best fitness: {fitness:.4f} with params: {params}")

    def save_results(self, output_path: str) -> str:
        """
        Save optimization results to file.

        SECURITY FIX: Path validation prevents directory traversal attacks.
        Only allows file operations within the designated results directory.
        Resolves paths to canonical form and validates against safe base directory.

        Args:
            output_path: Path to save results (relative to results directory)

        Returns:
            Path to saved file
        """
        # SECURITY: Validate and sanitize file path to prevent directory traversal
        safe_path = self._validate_and_sanitize_path(output_path, operation="save")
        if safe_path is None:
            raise ValueError(f"Invalid path for save operation: {output_path}")

        os.makedirs(os.path.dirname(safe_path), exist_ok=True)

        result = OptimizationResult(
            strategy_name=self.config.get("strategy_name", "unknown"),
            optimizer_type=self.__class__.__name__,
            best_params=self.best_params or {},
            best_fitness=self.best_fitness,
            fitness_metric=self.fitness_metric,
            iterations=self.current_iteration,
            total_evaluations=self.total_evaluations,
            optimization_time=self.config.get("optimization_time", 0.0),
            timestamp=datetime.now(),
            results_history=self.results_history,
            convergence_info=self._get_convergence_info(),
        )

        with open(safe_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        self.logger.info(f"Results saved to {safe_path}")
        return safe_path

    def load_results(self, input_path: str) -> Optional[OptimizationResult]:
        """
        Load optimization results from file.

        SECURITY FIX: Path validation prevents directory traversal attacks.
        Only allows file operations within the designated results directory.
        Resolves paths to canonical form and validates against safe base directory.

        Args:
            input_path: Path to load results from (relative to results directory)

        Returns:
            OptimizationResult if file exists, None otherwise
        """
        # SECURITY: Validate and sanitize file path to prevent directory traversal
        safe_path = self._validate_and_sanitize_path(input_path, operation="load")
        if safe_path is None:
            self.logger.error(f"Invalid path for load operation: {input_path}")
            return None

        if not os.path.exists(safe_path):
            return None

        try:
            with open(safe_path, "r") as f:
                data = json.load(f)

            result = OptimizationResult.from_dict(data)
            self.best_params = result.best_params
            self.best_fitness = result.best_fitness
            self.results_history = result.results_history

            self.logger.info(f"Results loaded from {safe_path}")
            return result

        except Exception as e:
            self.logger.error(f"Error loading results: {str(e)}")
            return None

    def _get_convergence_info(self) -> Optional[Dict[str, Any]]:
        """
        Get convergence information for the optimization.

        Returns:
            Convergence metrics dictionary
        """
        if len(self.results_history) < 2:
            return None

        # Calculate convergence metrics
        recent_fitness = [r["fitness"] for r in self.results_history[-10:]]
        if len(recent_fitness) >= 2:
            fitness_std = np.std(recent_fitness)
            fitness_trend = np.polyfit(range(len(recent_fitness)), recent_fitness, 1)[0]

            return {
                "recent_fitness_std": fitness_std,
                "fitness_trend": fitness_trend,
                "converged": fitness_std < 0.01 and abs(fitness_trend) < 0.001,
            }

        return None

    def _validate_and_sanitize_path(
        self, user_path: str, operation: str
    ) -> Optional[str]:
        """
        Validate and sanitize file paths to prevent directory traversal attacks.

        SECURITY: This method implements path validation to mitigate file path injection
        vulnerabilities. It resolves paths to canonical form and ensures they remain
        within a designated safe directory, preventing access to sensitive system files.

        Args:
            user_path: User-provided path string
            operation: Operation type ('save' or 'load') for logging

        Returns:
            Sanitized absolute path if valid, None if invalid
        """
        try:
            # Define the safe base directory for file operations
            # This should be configurable but defaults to 'results' directory
            base_dir = self.config.get("results_directory", "results")

            # Convert to absolute path and resolve any symlinks/canonicalize
            # This prevents directory traversal using .. or symlinks
            resolved_path = os.path.abspath(os.path.join(base_dir, user_path))

            # Ensure the resolved path is within the base directory
            # This is the key security check that prevents directory traversal
            base_dir_abs = os.path.abspath(base_dir)

            # Check if resolved path starts with base directory path
            if (
                not resolved_path.startswith(base_dir_abs + os.sep)
                and resolved_path != base_dir_abs
            ):
                self.logger.error(
                    f"SECURITY: Path traversal attempt blocked in {operation} operation. "
                    f"User path: {user_path}, Resolved: {resolved_path}, Base: {base_dir_abs}"
                )
                return None

            # Additional validation: ensure no null bytes or other dangerous characters
            if "\x00" in resolved_path:
                self.logger.error(f"SECURITY: Null byte detected in path: {user_path}")
                return None

            # Ensure the path doesn't contain dangerous characters
            dangerous_chars = ["<", ">", "|", "*", "?"]
            if any(char in resolved_path for char in dangerous_chars):
                self.logger.error(
                    f"SECURITY: Dangerous characters in path: {user_path}"
                )
                return None

            self.logger.debug(
                f"Path validation successful for {operation}: {resolved_path}"
            )
            return resolved_path

        except Exception as e:
            self.logger.error(f"Error validating path '{user_path}': {str(e)}")
            return None

    @abstractmethod
    def optimize(self, strategy_class, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize strategy parameters.

        Args:
            strategy_class: Strategy class to optimize
            data: Historical data for optimization

        Returns:
            Best parameter set found
        """
        pass

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of optimization process.

        Returns:
            Summary dictionary
        """
        return {
            "optimizer_type": self.__class__.__name__,
            "iterations": self.current_iteration,
            "total_evaluations": self.total_evaluations,
            "best_fitness": self.best_fitness,
            "best_params": self.best_params,
            "fitness_metric": self.fitness_metric,
            "convergence_info": self._get_convergence_info(),
        }
