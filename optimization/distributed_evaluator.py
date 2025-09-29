"""
Distributed Evaluator Module

This module provides distributed evaluation capabilities for strategy fitness
assessment. It supports both parallel processing within a single machine and
distributed processing across multiple nodes.

Key Features:
- Parallel fitness evaluation using ThreadPoolExecutor or ProcessPoolExecutor
- Distributed processing support (placeholder for Ray/Dask integration)
- Intelligent caching to avoid redundant evaluations
- Resource management and error handling
- Async evaluation support
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import get_distributed_evaluation_config
from .genome import StrategyGenome

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of a single fitness evaluation."""

    genome: StrategyGenome
    fitness: float
    evaluation_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DistributedEvaluator:
    """
    Distributed evaluation of strategy fitness.

    This class provides comprehensive distributed processing capabilities for
    evaluating trading strategy fitness across multiple cores or nodes. It
    supports both synchronous and asynchronous evaluation modes with
    intelligent caching and resource management.

    Key capabilities:
    - Parallel evaluation using multiple threads or processes
    - Distributed processing framework integration (Ray/Dask placeholders)
    - Intelligent result caching to avoid redundant computations
    - Resource monitoring and management
    - Error handling and recovery
    - Async evaluation support
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the distributed evaluator.

        Args:
            config: Configuration dictionary for distributed evaluation.
                   If None, uses default configuration from config module.
        """
        if config is None:
            eval_config = get_distributed_evaluation_config()
            config = {
                "enabled": eval_config.enabled,
                "max_workers": eval_config.max_workers,
                "use_processes": eval_config.use_processes,
                "worker_timeout": eval_config.worker_timeout,
                "max_retries": eval_config.max_retries,
                "cache_enabled": eval_config.cache_enabled,
                "cache_max_size": eval_config.cache_max_size,
            }

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Core configuration
        self.enabled = config.get("enabled", True)
        self.max_workers = config.get("max_workers", mp.cpu_count())
        self.use_processes = config.get("use_processes", False)
        self.worker_timeout = config.get("worker_timeout", 300)
        self.max_retries = config.get("max_retries", 3)

        # Caching
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_max_size = config.get("cache_max_size", 1000)
        self._evaluation_cache: Dict[str, EvaluationResult] = {}

        # Executor management
        self.executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None

        # Statistics
        self.total_evaluations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.failed_evaluations = 0

        self.logger.info(
            f"DistributedEvaluator initialized with {self.max_workers} max workers"
        )

    async def initialize(self) -> None:
        """Initialize the distributed evaluator."""
        if not self.enabled:
            self.logger.info("Distributed evaluation disabled")
            return

        # Initialize executors
        if self.use_processes:
            self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
            self.logger.info(
                f"Initialized ProcessPoolExecutor with {self.max_workers} workers"
            )
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            self.logger.info(
                f"Initialized ThreadPoolExecutor with {self.max_workers} workers"
            )

    async def evaluate_population(
        self,
        population: List[StrategyGenome],
        data: pd.DataFrame,
        fitness_func: Optional[Callable] = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate fitness of a population in parallel.

        Args:
            population: List of genomes to evaluate
            data: Historical data for backtesting
            fitness_func: Custom fitness evaluation function

        Returns:
            List of evaluation results
        """
        if not population:
            return []

        if not self.enabled:
            # Fallback to sequential evaluation
            return await self._evaluate_sequential(population, data, fitness_func)

        # Check cache first
        cached_results = []
        to_evaluate = []

        for genome in population:
            if self.cache_enabled:
                cache_key = self._get_cache_key(genome, data)
                if cache_key in self._evaluation_cache:
                    cached_results.append(self._evaluation_cache[cache_key])
                    self.cache_hits += 1
                    continue

            to_evaluate.append(genome)
            self.cache_misses += 1

        # Evaluate remaining genomes
        if to_evaluate:
            if self.use_processes:
                evaluated_results = await self._evaluate_with_processes(
                    to_evaluate, data, fitness_func
                )
            else:
                evaluated_results = await self._evaluate_with_threads(
                    to_evaluate, data, fitness_func
                )

            # Cache results
            if self.cache_enabled:
                for result in evaluated_results:
                    if result.success:
                        cache_key = self._get_cache_key(result.genome, data)
                        self._evaluation_cache[cache_key] = result

                        # Maintain cache size limit
                        if len(self._evaluation_cache) > self.cache_max_size:
                            # Remove oldest entry (simple LRU approximation)
                            oldest_key = next(iter(self._evaluation_cache))
                            del self._evaluation_cache[oldest_key]

        # Combine cached and newly evaluated results
        all_results = cached_results + evaluated_results
        self.total_evaluations += len(to_evaluate)

        return all_results

    async def _evaluate_with_threads(
        self,
        genomes: List[StrategyGenome],
        data: pd.DataFrame,
        fitness_func: Optional[Callable],
    ) -> List[EvaluationResult]:
        """Evaluate genomes using ThreadPoolExecutor."""
        if not self.executor:
            await self.initialize()

        results = []

        # Submit all evaluation tasks
        future_to_genome = {}
        for genome in genomes:
            future = self.executor.submit(
                self._evaluate_single_genome_sync, genome, data, fitness_func
            )
            future_to_genome[future] = genome

        # Collect results as they complete
        for future in as_completed(future_to_genome):
            genome = future_to_genome[future]
            try:
                result = future.result(timeout=self.worker_timeout)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Evaluation failed for genome: {e}")
                self.failed_evaluations += 1
                # Create error result
                error_result = EvaluationResult(
                    genome=genome,
                    fitness=float("-inf"),
                    evaluation_time=0.0,
                    success=False,
                    error_message=str(e),
                )
                results.append(error_result)

        return results

    async def _evaluate_with_processes(
        self,
        genomes: List[StrategyGenome],
        data: pd.DataFrame,
        fitness_func: Optional[Callable],
    ) -> List[EvaluationResult]:
        """Evaluate genomes using ProcessPoolExecutor."""
        if not self.process_executor:
            await self.initialize()

        results = []

        # Submit all evaluation tasks
        future_to_genome = {}
        for genome in genomes:
            future = self.process_executor.submit(
                self._evaluate_single_genome_sync, genome, data, fitness_func
            )
            future_to_genome[future] = genome

        # Collect results as they complete
        for future in as_completed(future_to_genome):
            genome = future_to_genome[future]
            try:
                result = future.result(timeout=self.worker_timeout)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Evaluation failed for genome: {e}")
                self.failed_evaluations += 1
                # Create error result
                error_result = EvaluationResult(
                    genome=genome,
                    fitness=float("-inf"),
                    evaluation_time=0.0,
                    success=False,
                    error_message=str(e),
                )
                results.append(error_result)

        return results

    async def _evaluate_sequential(
        self,
        genomes: List[StrategyGenome],
        data: pd.DataFrame,
        fitness_func: Optional[Callable],
    ) -> List[EvaluationResult]:
        """Evaluate genomes sequentially (fallback mode)."""
        results = []

        for genome in genomes:
            result = self._evaluate_single_genome_sync(genome, data, fitness_func)
            results.append(result)

        return results

    def _evaluate_single_genome_sync(
        self,
        genome: StrategyGenome,
        data: pd.DataFrame,
        fitness_func: Optional[Callable],
    ) -> EvaluationResult:
        """Synchronous evaluation of a single genome."""
        import time

        start_time = time.time()

        try:
            if fitness_func:
                fitness = fitness_func(genome, data)
            else:
                fitness = self._default_fitness_function(genome, data)

            evaluation_time = time.time() - start_time

            return EvaluationResult(
                genome=genome,
                fitness=fitness,
                evaluation_time=evaluation_time,
                success=True,
            )

        except Exception as e:
            evaluation_time = time.time() - start_time
            self.logger.error(f"Genome evaluation failed: {e}")

            return EvaluationResult(
                genome=genome,
                fitness=float("-inf"),
                evaluation_time=evaluation_time,
                success=False,
                error_message=str(e),
            )

    def _default_fitness_function(
        self, genome: StrategyGenome, data: pd.DataFrame
    ) -> float:
        """Default fitness evaluation function."""
        try:
            # Simple fitness based on genome complexity and data characteristics
            base_fitness = len(genome.genes) * 0.1

            # Add some data-based variation
            if not data.empty:
                volatility = data["close"].pct_change().std()
                base_fitness += volatility * 10  # Reward strategies in volatile markets

            # Add randomness to simulate real evaluation
            base_fitness += np.random.normal(0, 0.1)

            return base_fitness

        except Exception:
            return float("-inf")

    def _get_cache_key(self, genome: StrategyGenome, data: pd.DataFrame) -> str:
        """Generate a cache key for genome evaluation."""
        # Simple cache key based on genome structure and data
        gene_summary = f"{len(genome.genes)}_{genome.generation}"
        data_summary = f"{len(data)}_{data.index[0] if not data.empty else 'empty'}"
        return f"{gene_summary}_{data_summary}"

    async def evaluate_population_async(
        self,
        population: List[StrategyGenome],
        data: pd.DataFrame,
        fitness_func: Optional[Callable] = None,
    ) -> List[EvaluationResult]:
        """
        Async version of population evaluation.

        Args:
            population: List of genomes to evaluate
            data: Historical data for backtesting
            fitness_func: Custom fitness evaluation function

        Returns:
            List of evaluation results
        """
        # For now, delegate to the main evaluate_population method
        # In a full implementation, this would use asyncio.gather for true async evaluation
        return await self.evaluate_population(population, data, fitness_func)

    def get_worker_status(self) -> Dict[str, Any]:
        """Get status of workers in the distributed evaluator."""
        status = {
            "enabled": self.enabled,
            "max_workers": self.max_workers,
            "use_processes": self.use_processes,
            "executor_active": (self.executor is not None)
            or (self.process_executor is not None),
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self._evaluation_cache),
            "cache_max_size": self.cache_max_size,
            "total_evaluations": self.total_evaluations,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "failed_evaluations": self.failed_evaluations,
            "cache_hit_rate": self.cache_hits
            / max(1, self.cache_hits + self.cache_misses),
        }

        # Add executor-specific information
        if self.executor:
            status["executor_type"] = "ThreadPoolExecutor"
            try:
                status["active_threads"] = len(
                    [t for t in self.executor._threads if t.is_alive()]
                )
            except Exception:
                status["active_threads"] = 0
        elif self.process_executor:
            status["executor_type"] = "ProcessPoolExecutor"
            try:
                status["active_processes"] = len(
                    [
                        p
                        for p in self.process_executor._processes.values()
                        if p.is_alive()
                    ]
                )
            except Exception:
                status["active_processes"] = 0

        return status

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._evaluation_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("Evaluation cache cleared")

    def set_cache_size(self, max_size: int) -> None:
        """Set the maximum cache size."""
        self.cache_max_size = max_size
        self.logger.info(f"Cache max size set to {max_size}")

    async def shutdown(self) -> None:
        """Shutdown the distributed evaluator and clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        if self.process_executor:
            self.process_executor.shutdown(wait=True)
            self.process_executor = None

        self.logger.info("DistributedEvaluator shutdown complete")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the evaluator."""
        total_requests = self.cache_hits + self.cache_misses

        return {
            "total_evaluations": self.total_evaluations,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "failed_evaluations": self.failed_evaluations,
            "cache_hit_rate": self.cache_hits / max(1, total_requests),
            "success_rate": (self.total_evaluations - self.failed_evaluations)
            / max(1, self.total_evaluations),
            "average_evaluation_time": 0.0,  # Would need to track this
            "cache_utilization": len(self._evaluation_cache) / self.cache_max_size,
        }

    def __str__(self) -> str:
        """String representation of the distributed evaluator."""
        status = self.get_worker_status()
        return (
            f"DistributedEvaluator(workers={status['max_workers']}, "
            f"cache={status['cache_size']}/{status['cache_max_size']}, "
            f"hit_rate={status.get('cache_hit_rate', 0):.1%})"
        )


# Convenience functions
async def create_distributed_evaluator(
    config: Optional[Dict[str, Any]] = None
) -> DistributedEvaluator:
    """
    Create and initialize a distributed evaluator.

    Args:
        config: Optional configuration overrides

    Returns:
        Initialized DistributedEvaluator instance
    """
    evaluator = DistributedEvaluator(config)
    await evaluator.initialize()
    return evaluator


def get_distributed_evaluator(
    config: Optional[Dict[str, Any]] = None
) -> DistributedEvaluator:
    """
    Get a distributed evaluator instance (synchronous wrapper).

    Args:
        config: Configuration dictionary

    Returns:
        DistributedEvaluator instance
    """
    return DistributedEvaluator(config)
