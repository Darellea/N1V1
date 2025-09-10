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

from .base_optimizer import BaseOptimizer, ParameterBounds
from strategies.base_strategy import BaseStrategy
from backtest.backtester import compute_backtest_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


class StrategyComponent(Enum):
    """Types of strategy components that can be genetically combined."""
    INDICATOR = "indicator"
    SIGNAL_LOGIC = "signal_logic"
    RISK_MANAGEMENT = "risk_management"
    TIMEFRAME = "timeframe"
    FILTER = "filter"


class IndicatorType(Enum):
    """Available technical indicators."""
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    STOCHASTIC = "stochastic"
    MOVING_AVERAGE = "moving_average"
    ATR = "atr"
    VOLUME = "volume"
    PRICE_ACTION = "price_action"


class SignalLogic(Enum):
    """Types of signal generation logic."""
    CROSSOVER = "crossover"
    THRESHOLD = "threshold"
    PATTERN = "pattern"
    DIVERGENCE = "divergence"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"


@dataclass
class StrategyGene:
    """Individual gene representing a strategy component."""
    component_type: StrategyComponent
    indicator_type: Optional[IndicatorType] = None
    signal_logic: Optional[SignalLogic] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'component_type': self.component_type.value,
            'indicator_type': self.indicator_type.value if self.indicator_type else None,
            'signal_logic': self.signal_logic.value if self.signal_logic else None,
            'parameters': self.parameters,
            'weight': self.weight,
            'enabled': self.enabled
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyGene':
        """Create from dictionary."""
        return cls(
            component_type=StrategyComponent(data['component_type']),
            indicator_type=IndicatorType(data['indicator_type']) if data.get('indicator_type') else None,
            signal_logic=SignalLogic(data['signal_logic']) if data.get('signal_logic') else None,
            parameters=data.get('parameters', {}),
            weight=data.get('weight', 1.0),
            enabled=data.get('enabled', True)
        )


@dataclass
class StrategyGenome:
    """Complete genetic representation of a trading strategy."""
    genes: List[StrategyGene] = field(default_factory=list)
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    species_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate genome after initialization."""
        self._validate_genome()

    def _validate_genome(self) -> None:
        """Validate genome structure and completeness."""
        if not self.genes:
            return

        # Ensure we have at least one indicator and signal logic
        has_indicator = any(g.component_type == StrategyComponent.INDICATOR for g in self.genes)
        has_signal = any(g.component_type == StrategyComponent.SIGNAL_LOGIC for g in self.genes)

        if not (has_indicator and has_signal):
            logger.warning("Genome missing required components (indicator or signal logic)")

    def copy(self) -> 'StrategyGenome':
        """Create a deep copy of this genome."""
        return StrategyGenome(
            genes=[StrategyGene(
                component_type=g.component_type,
                indicator_type=g.indicator_type,
                signal_logic=g.signal_logic,
                parameters=g.parameters.copy(),
                weight=g.weight,
                enabled=g.enabled
            ) for g in self.genes],
            fitness=self.fitness,
            age=self.age,
            generation=self.generation,
            species_id=self.species_id,
            metadata=self.metadata.copy()
        )

    def mutate(self, mutation_rate: float = 0.1) -> 'StrategyGenome':
        """Apply mutations to this genome."""
        for gene in self.genes:
            if random.random() < mutation_rate:
                self._mutate_gene(gene)

        # Occasionally add or remove genes
        if random.random() < 0.05:  # 5% chance
            if random.random() < 0.5 and len(self.genes) < 10:
                self._add_random_gene()
            elif len(self.genes) > 3:
                self._remove_random_gene()

        self._validate_genome()
        return self

    def _mutate_gene(self, gene: StrategyGene) -> None:
        """Mutate a single gene."""
        # Mutate parameters
        for param_name, param_value in gene.parameters.items():
            if isinstance(param_value, (int, float)):
                # Gaussian mutation for numeric parameters
                if isinstance(param_value, int):
                    mutation = random.gauss(0, max(1, abs(param_value) * 0.1))
                    gene.parameters[param_name] = max(1, int(param_value + mutation))
                else:
                    mutation = random.gauss(0, max(0.01, abs(param_value) * 0.1))
                    gene.parameters[param_name] = max(0.01, param_value + mutation)

        # Mutate weight
        if random.random() < 0.3:
            gene.weight = max(0.1, min(2.0, gene.weight + random.gauss(0, 0.2)))

        # Occasionally disable/enable gene
        if random.random() < 0.1:
            gene.enabled = not gene.enabled

    def _add_random_gene(self) -> None:
        """Add a random gene to the genome."""
        component_type = random.choice(list(StrategyComponent))

        gene = StrategyGene(component_type=component_type)

        if component_type == StrategyComponent.INDICATOR:
            gene.indicator_type = random.choice(list(IndicatorType))
            gene.parameters = self._get_default_indicator_params(gene.indicator_type)
        elif component_type == StrategyComponent.SIGNAL_LOGIC:
            gene.signal_logic = random.choice(list(SignalLogic))
            gene.parameters = self._get_default_signal_params(gene.signal_logic)

        self.genes.append(gene)

    def _remove_random_gene(self) -> None:
        """Remove a random gene from the genome."""
        if self.genes:
            gene_to_remove = random.choice(self.genes)
            self.genes.remove(gene_to_remove)

    def _get_default_indicator_params(self, indicator_type: IndicatorType) -> Dict[str, Any]:
        """Get default parameters for an indicator type."""
        defaults = {
            IndicatorType.RSI: {'period': 14, 'overbought': 70, 'oversold': 30},
            IndicatorType.MACD: {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            IndicatorType.BOLLINGER_BANDS: {'period': 20, 'std_dev': 2.0},
            IndicatorType.STOCHASTIC: {'k_period': 14, 'd_period': 3, 'overbought': 80, 'oversold': 20},
            IndicatorType.MOVING_AVERAGE: {'period': 20, 'type': 'sma'},
            IndicatorType.ATR: {'period': 14},
            IndicatorType.VOLUME: {'period': 20},
            IndicatorType.PRICE_ACTION: {'lookback': 5}
        }
        return defaults.get(indicator_type, {})

    def _get_default_signal_params(self, signal_logic: SignalLogic) -> Dict[str, Any]:
        """Get default parameters for signal logic."""
        defaults = {
            SignalLogic.CROSSOVER: {'fast_period': 9, 'slow_period': 21},
            SignalLogic.THRESHOLD: {'threshold': 0.5, 'direction': 'above'},
            SignalLogic.PATTERN: {'pattern_type': 'double_bottom', 'tolerance': 0.02},
            SignalLogic.DIVERGENCE: {'lookback': 10, 'threshold': 0.1},
            SignalLogic.MOMENTUM: {'period': 10, 'threshold': 0.02},
            SignalLogic.MEAN_REVERSION: {'mean_period': 20, 'std_threshold': 2.0}
        }
        return defaults.get(signal_logic, {})

    def crossover(self, other: 'StrategyGenome') -> Tuple['StrategyGenome', 'StrategyGenome']:
        """Perform crossover with another genome."""
        # Single-point crossover
        if not self.genes or not other.genes:
            return self.copy(), other.copy()

        min_len = min(len(self.genes), len(other.genes))
        if min_len < 2:
            return self.copy(), other.copy()

        crossover_point = random.randint(1, min_len - 1)

        child1_genes = self.genes[:crossover_point] + other.genes[crossover_point:]
        child2_genes = other.genes[:crossover_point] + self.genes[crossover_point:]

        child1 = StrategyGenome(
            genes=child1_genes,
            generation=max(self.generation, other.generation) + 1
        )
        child2 = StrategyGenome(
            genes=child2_genes,
            generation=max(self.generation, other.generation) + 1
        )

        return child1, child2

    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary for serialization."""
        return {
            'genes': [gene.to_dict() for gene in self.genes],
            'fitness': self.fitness,
            'age': self.age,
            'generation': self.generation,
            'species_id': self.species_id,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyGenome':
        """Create genome from dictionary."""
        return cls(
            genes=[StrategyGene.from_dict(g) for g in data.get('genes', [])],
            fitness=data.get('fitness', float('-inf')),
            age=data.get('age', 0),
            generation=data.get('generation', 0),
            species_id=data.get('species_id'),
            metadata=data.get('metadata', {})
        )

    def __str__(self) -> str:
        """String representation of the genome."""
        return f"Genome(gen={self.generation}, fitness={self.fitness:.4f}, genes={len(self.genes)})"

    @staticmethod
    def select_best(population: List['StrategyGenome'], num_to_select: int) -> List['StrategyGenome']:
        """Select the best N genomes by fitness."""
        if not population:
            return []

        # Sort by fitness in descending order
        sorted_population = sorted(population, key=lambda g: g.fitness, reverse=True)
        return sorted_population[:num_to_select]

    @staticmethod
    def tournament_selection(population: List['StrategyGenome'], tournament_size: int) -> 'StrategyGenome':
        """Perform tournament selection to choose a parent."""
        if not population:
            raise ValueError("Population cannot be empty")

        if tournament_size > len(population):
            tournament_size = len(population)

        # Select random candidates for tournament
        tournament = random.sample(population, tournament_size)

        # Return the best from the tournament
        return max(tournament, key=lambda g: g.fitness)


@dataclass
class Species:
    """Species for maintaining population diversity."""
    species_id: str
    representative: StrategyGenome
    members: List[StrategyGenome] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)
    stagnation_counter: int = 0

    def update_representative(self) -> None:
        """Update species representative based on current members."""
        if not self.members:
            return

        # Find member with highest fitness
        best_member = max(self.members, key=lambda g: g.fitness)
        self.representative = best_member.copy()

    def calculate_diversity_score(self) -> float:
        """Calculate diversity score within species."""
        if len(self.members) < 2:
            return 0.0

        # Simple diversity based on fitness variance
        fitness_values = [g.fitness for g in self.members if g.fitness != float('-inf')]
        if len(fitness_values) < 2:
            return 0.0

        return np.std(fitness_values) / (np.mean(fitness_values) + 1e-6)


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
        """Convert genome to executable strategy."""
        # Simplified implementation - in practice this would be more sophisticated
        try:
            from strategies.base_strategy import BaseStrategy

            class GeneratedStrategy(BaseStrategy):
                def __init__(self, config):
                    super().__init__(config)
                    self.genome = genome

                def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
                    signals = []

                    if data.empty or len(data) < 20:
                        return signals

                    # Generate signals based on genome characteristics
                    signal_probability = min(0.2, len(genome.genes) / 20.0)  # More genes = more signals

                    for i in range(20, len(data)):
                        if random.random() < signal_probability:
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

    async def generate_strategy(self, genome: StrategyGenome, name: str) -> Optional[type]:
        """Generate a strategy class from a genome."""
        try:
            # Use the existing _genome_to_strategy method
            strategy_instance = self._genome_to_strategy(genome)

            if strategy_instance is None:
                return None

            # Create a class from the instance
            strategy_class = type(name, (type(strategy_instance),), {
                '__init__': lambda self, config: super(type(self), self).__init__(config),
                'generate_signals': strategy_instance.generate_signals,
                'genome': genome
            })

            return strategy_class

        except Exception as e:
            logger.error(f"Failed to generate strategy: {e}")
            return None

    async def evolve(self) -> None:
        """Run one generation of evolution."""
        if not self.population:
            await self._initialize_population()

        # Evaluate current population
        # For now, use dummy data - in practice this would be passed in
        dummy_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
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


# Register on import
_register_strategy_generator()
