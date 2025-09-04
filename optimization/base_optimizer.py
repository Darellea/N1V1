"""
optimization/base_optimizer.py

Abstract base class for all optimization techniques.
Defines the interface and common functionality for strategy optimization.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import logging
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

from backtest.backtester import compute_backtest_metrics


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
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationResult':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
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

    def validate_value(self, value: Any) -> bool:
        """Validate if a value is within bounds."""
        if self.param_type == "categorical":
            return value in (self.categories or [])
        elif self.param_type == "int":
            return isinstance(value, int) and self.min_value <= value <= self.max_value
        else:  # float
            return isinstance(value, (int, float)) and self.min_value <= value <= self.max_value

    def clamp_value(self, value: Any) -> Any:
        """Clamp value to bounds."""
        if self.param_type == "categorical":
            return value if value in (self.categories or []) else (self.categories[0] if self.categories else value)
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
        self.best_fitness = float('-inf')
        self.results_history: List[Dict[str, Any]] = []

        # Fitness function configuration
        self.fitness_metric = config.get('fitness_metric', 'sharpe_ratio')
        self.fitness_weights = config.get('fitness_weights', {
            'sharpe_ratio': 1.0,
            'total_return': 0.3,
            'win_rate': 0.2,
            'max_drawdown': -0.1  # Negative weight for minimization
        })

    def _setup_logging(self) -> None:
        """Setup logging for the optimizer."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        for name, value in params.items():
            if name in self.parameter_bounds:
                bounds = self.parameter_bounds[name]
                if not bounds.validate_value(value):
                    self.logger.warning(f"Parameter {name}={value} is out of bounds")
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
                return float('-inf')

            # Compute metrics
            metrics = compute_backtest_metrics(equity_progression)

            # Calculate composite fitness score
            fitness = self._calculate_fitness_score(metrics)

            self.total_evaluations += 1

            return fitness

        except Exception as e:
            self.logger.error(f"Error evaluating fitness: {str(e)}")
            return float('-inf')

    def _run_backtest(self, strategy_instance, data: pd.DataFrame) -> List[Dict[str, Any]]:
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

                    equity_progression.append({
                        'trade_id': trade_count,
                        'timestamp': data.index[i] if hasattr(data.index, '__getitem__') else i,
                        'equity': current_equity,
                        'pnl': pnl,
                        'cumulative_return': (current_equity - initial_equity) / initial_equity
                    })

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

            self.logger.info(
                f"New best fitness: {fitness:.4f} with params: {params}"
            )

    def save_results(self, output_path: str) -> str:
        """
        Save optimization results to file.

        Args:
            output_path: Path to save results

        Returns:
            Path to saved file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        result = OptimizationResult(
            strategy_name=self.config.get('strategy_name', 'unknown'),
            optimizer_type=self.__class__.__name__,
            best_params=self.best_params or {},
            best_fitness=self.best_fitness,
            fitness_metric=self.fitness_metric,
            iterations=self.current_iteration,
            total_evaluations=self.total_evaluations,
            optimization_time=self.config.get('optimization_time', 0.0),
            timestamp=datetime.now(),
            results_history=self.results_history,
            convergence_info=self._get_convergence_info()
        )

        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        self.logger.info(f"Results saved to {output_path}")
        return output_path

    def load_results(self, input_path: str) -> Optional[OptimizationResult]:
        """
        Load optimization results from file.

        Args:
            input_path: Path to load results from

        Returns:
            OptimizationResult if file exists, None otherwise
        """
        if not os.path.exists(input_path):
            return None

        try:
            with open(input_path, 'r') as f:
                data = json.load(f)

            result = OptimizationResult.from_dict(data)
            self.best_params = result.best_params
            self.best_fitness = result.best_fitness
            self.results_history = result.results_history

            self.logger.info(f"Results loaded from {input_path}")
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
        recent_fitness = [r['fitness'] for r in self.results_history[-10:]]
        if len(recent_fitness) >= 2:
            fitness_std = np.std(recent_fitness)
            fitness_trend = np.polyfit(range(len(recent_fitness)), recent_fitness, 1)[0]

            return {
                'recent_fitness_std': fitness_std,
                'fitness_trend': fitness_trend,
                'converged': fitness_std < 0.01 and abs(fitness_trend) < 0.001
            }

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
            'optimizer_type': self.__class__.__name__,
            'iterations': self.current_iteration,
            'total_evaluations': self.total_evaluations,
            'best_fitness': self.best_fitness,
            'best_params': self.best_params,
            'fitness_metric': self.fitness_metric,
            'convergence_info': self._get_convergence_info()
        }
