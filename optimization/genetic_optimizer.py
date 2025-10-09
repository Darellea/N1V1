"""
optimization/genetic_optimizer.py

Genetic Algorithm implementation for strategy parameter optimization.
Uses evolutionary principles to find optimal parameter combinations.
"""

import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_optimizer import BaseOptimizer, ParameterBounds


@dataclass
class Chromosome:
    """Represents a parameter set as a chromosome for GA."""

    genes: Dict[str, Any]  # Parameter name -> value mapping
    fitness: float = float("-inf")

    def __post_init__(self):
        """Validate chromosome after initialization."""
        pass  # Allow empty genes for graceful handling

    def copy(self) -> "Chromosome":
        """Create a copy of this chromosome."""
        return Chromosome(genes=self.genes.copy(), fitness=self.fitness)

    def mutate(
        self, mutation_rate: float, parameter_bounds: Dict[str, ParameterBounds]
    ) -> None:
        """
        Mutate this chromosome's genes.

        Args:
            mutation_rate: Probability of mutation for each gene
            parameter_bounds: Parameter constraints
        """
        for param_name, current_value in self.genes.items():
            if random.random() < mutation_rate:
                bounds = parameter_bounds.get(param_name)
                if bounds:
                    self.genes[param_name] = self._mutate_gene(current_value, bounds)

    def _mutate_gene(self, value: Any, bounds: ParameterBounds) -> Any:
        """
        Mutate a single gene within its bounds.

        Args:
            value: Current gene value
            bounds: Parameter bounds

        Returns:
            Mutated gene value
        """
        if bounds.param_type == "categorical":
            # Random selection from categories (excluding current value if possible)
            categories = bounds.categories or []
            if len(categories) > 1:
                available = [cat for cat in categories if cat != value]
                return random.choice(available) if available else value
            return value

        elif bounds.param_type == "int":
            # Random integer within bounds
            if bounds.step and bounds.step > 1:
                # Use step size for discrete mutation
                step = bounds.step
                min_val = bounds.min_value
                max_val = bounds.max_value
                current_step = int((value - min_val) / step)
                new_step = current_step + random.choice([-1, 1])
                new_step = max(0, min(int((max_val - min_val) / step), new_step))
                return min_val + new_step * step
            else:
                # Random integer mutation
                return random.randint(int(bounds.min_value), int(bounds.max_value))

        else:  # float
            # Gaussian mutation
            if bounds.step and bounds.step > 0:
                # Use step size as mutation scale
                mutation = random.gauss(0, bounds.step)
            else:
                # Use 10% of range as mutation scale
                range_size = bounds.max_value - bounds.min_value
                mutation = random.gauss(0, range_size * 0.1)

            new_value = value + mutation
            return max(bounds.min_value, min(bounds.max_value, new_value))


class GeneticOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer for strategy parameters.

    This optimizer:
    1. Represents parameter sets as chromosomes
    2. Maintains a population of parameter combinations
    3. Uses selection, crossover, and mutation operators
    4. Evolves the population over generations
    5. Returns the fittest parameter set
    """

    # Nested ParameterBounds class for test compatibility
    ParameterBounds = ParameterBounds

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Genetic Algorithm Optimizer with Adaptive Population Sizing.

        Args:
            config: Configuration dictionary containing:
                - initial_population_size: Starting number of chromosomes in population
                - min_population_size: Minimum population size (default: 10)
                - max_population_size: Maximum population size (default: 200)
                - generations: Number of generations to evolve
                - mutation_rate: Probability of gene mutation
                - crossover_rate: Probability of crossover
                - elitism_rate: Fraction of best individuals to preserve
                - tournament_size: Size of tournament for selection
                - adaptation_rate: How aggressively to adjust population size (default: 0.1)
        """
        super().__init__(config)

        # Performance optimizations (must be set early for test mode logic)
        self.enable_fitness_caching = config.get("enable_fitness_caching", True)
        self.enable_early_termination = config.get("enable_early_termination", True)
        self.early_termination_threshold = config.get(
            "early_termination_threshold", 0.001
        )
        self.early_termination_window = config.get("early_termination_window", 5)
        self.test_mode = config.get("test_mode", False)  # Reduce population for tests

        # ============================================================================
        # ADAPTIVE POPULATION SIZING
        # ============================================================================
        #
        # PROBLEM WITH FIXED POPULATION SIZE:
        # - Fixed size can lead to premature convergence (population too small)
        # - Fixed size can be inefficient for complex problems (population too small)
        # - Fixed size wastes resources on simple problems (population too large)
        #
        # BENEFITS OF ADAPTIVE SIZING:
        # - Dynamically adjusts to problem complexity
        # - Prevents premature convergence by increasing size when diversity is low
        # - Improves efficiency by reducing size when convergence is fast
        # - Better exploration vs exploitation balance
        # - Can find better solutions by adapting to the optimization landscape
        #
        # ADAPTATION LOGIC:
        # - Monitor fitness improvement rate (convergence speed)
        # - Monitor population diversity (standard deviation of fitness)
        # - If convergence is too slow AND diversity is low: increase population
        # - If convergence is too fast: decrease population
        # - Stay within min/max bounds to prevent extreme sizes
        # ============================================================================

        # Adaptive population sizing parameters
        # Support both 'population_size' and 'initial_population_size' for backward compatibility
        self.initial_population_size = config.get(
            "initial_population_size", config.get("population_size", 20)
        )
        self.min_population_size = config.get("min_population_size", 10)
        self.max_population_size = config.get("max_population_size", 200)
        self.adaptation_rate = config.get(
            "adaptation_rate", 0.1
        )  # How aggressively to adapt

        # Current population size (starts with initial, changes adaptively)
        self.population_size = self.initial_population_size

        # Apply test mode optimizations
        if self.test_mode:
            # Reduce population size for tests to improve performance
            self.population_size = min(self.population_size, 20)
            self.max_population_size = min(self.max_population_size, 20)
            self.logger.info(
                f"Test mode enabled: reduced population size to {self.population_size}"
            )

        # GA specific configuration
        self.generations = config.get("generations", 10)
        self.mutation_rate = config.get("mutation_rate", 0.1)
        self.crossover_rate = config.get("crossover_rate", 0.7)
        self.elitism_rate = config.get("elitism_rate", 0.1)
        self.tournament_size = config.get("tournament_size", 3)

        # Fitness cache: maps parameter hash to fitness value
        self.fitness_cache: Dict[str, float] = {}

        # Population state
        self.population: List[Chromosome] = []
        self.best_chromosome: Optional[Chromosome] = None

        # Adaptive sizing tracking
        self.fitness_history: List[float] = []  # Track best fitness per generation
        self.diversity_history: List[float] = []  # Track population diversity
        self.population_size_history: List[int] = []  # Track population size changes

        # Elitism count (will be updated when population size changes)
        self.elite_count = max(1, int(self.population_size * self.elitism_rate))

        # Make Chromosome accessible within the class
        self.Chromosome = Chromosome

        # Parameter bounds inherited from BaseOptimizer (dict)

        # ============================================================================
        # DETERMINISTIC EXECUTION AND SEED MANAGEMENT
        # ============================================================================
        #
        # PROBLEM WITH NON-DETERMINISTIC OPTIMIZATION:
        # - Random seed issues make backtests irreproducible
        # - Parallel execution can cause seed conflicts
        # - No way to resume optimization from specific state
        # - Hard to debug optimization issues due to randomness
        #
        # SOLUTION: STRICT SEED CONTROL AND DETERMINISTIC EXECUTION
        # - Support reproducible mode (fixed seed) and exploratory mode (random seed)
        # - Seed isolation for parallel/distributed workers
        # - Random state checkpointing and restoration
        # - Maintain optimization algorithm effectiveness
        # ============================================================================

        # Seed management configuration
        self.random_mode = config.get(
            "random_mode", "reproducible"
        )  # "reproducible" or "exploratory"
        self.base_seed = config.get("base_seed", 42)  # Base seed for reproducible mode
        self.worker_id = config.get(
            "worker_id", 0
        )  # For parallel execution seed isolation
        self.num_workers = config.get(
            "num_workers", 1
        )  # Total number of parallel workers

        # Random state management
        self.random_state: Optional[np.random.RandomState] = None
        self.python_random_state: Optional[tuple] = None
        self.checkpoint_data: Optional[Dict[str, Any]] = None

        # Initialize random state based on mode
        self._initialize_random_state()

    def add_parameter_bounds(self, bounds: ParameterBounds) -> None:
        """
        Add parameter bounds for validation.

        Args:
            bounds: ParameterBounds object defining constraints
        """
        self.parameter_bounds[bounds.name] = bounds

    def _calculate_fitness(
        self, strategy, data: pd.DataFrame, params: Dict[str, Any]
    ) -> float:
        """
        Calculate fitness for a parameter set.

        Args:
            strategy: Strategy instance
            data: Historical data
            params: Parameter set

        Returns:
            Fitness score
        """
        return self.evaluate_fitness(strategy, data)

    def _create_next_generation(self) -> None:
        """
        Create the next generation through selection, crossover, and mutation.
        Includes adaptive population sizing based on convergence metrics.
        """
        # Adaptive population sizing: adjust size based on convergence
        self._adapt_population_size()

        new_population = []

        # Elitism: preserve best individuals (recalculate elite count for new population size)
        self.elite_count = max(1, int(self.population_size * self.elitism_rate))
        elite = self._select_elite()
        new_population.extend(elite)

        # Fill rest of population through selection and reproduction
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if random.random() < self.crossover_rate:
                offspring1, offspring2 = self._crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()

            # Mutation
            offspring1.mutate(self.mutation_rate, self._get_bounds_dict())
            offspring2.mutate(self.mutation_rate, self._get_bounds_dict())

            new_population.extend([offspring1, offspring2])

        # Trim population to correct size
        self.population = new_population[: self.population_size]

    def _adapt_population_size(self) -> None:
        """
        Adapt population size based on convergence metrics.

        This method implements the adaptive population sizing logic:
        - Monitors fitness improvement rate and population diversity
        - Increases population size when convergence is slow and diversity is low
        - Decreases population size when convergence is fast
        - Maintains population size within configured bounds
        """
        if len(self.fitness_history) < 2:
            # Not enough history to make adaptation decisions
            self.population_size_history.append(self.population_size)
            return

        # Get current fitness metrics
        current_fitness = self.fitness_history[-1]
        previous_fitness = self.fitness_history[-2]

        # Calculate fitness improvement rate
        fitness_improvement = current_fitness - previous_fitness

        # Get current population diversity (standard deviation of fitness)
        current_diversity = (
            self.diversity_history[-1] if self.diversity_history else 0.0
        )

        # Adaptive logic
        old_population_size = self.population_size
        adaptation_factor = 0

        # Case 1: Slow convergence with low diversity - increase population
        if fitness_improvement < 0.01 and current_diversity < 0.1:
            # Convergence is slow and population lacks diversity
            adaptation_factor = self.adaptation_rate
            self.logger.debug(".2f")

        # Case 2: Fast convergence - decrease population to improve efficiency
        elif fitness_improvement > 0.05:
            # Convergence is fast, can reduce population size
            adaptation_factor = -self.adaptation_rate * 0.5  # Less aggressive decrease
            self.logger.debug(".2f")

        # Apply adaptation if needed
        if adaptation_factor != 0:
            new_size = int(self.population_size * (1 + adaptation_factor))

            # Ensure within bounds
            new_size = max(
                self.min_population_size, min(self.max_population_size, new_size)
            )

            if new_size != self.population_size:
                self.logger.info(
                    f"Adapting population size: {self.population_size} -> {new_size} "
                    f"(fitness_improvement: {fitness_improvement:.4f}, diversity: {current_diversity:.4f})"
                )
                self.population_size = new_size

        # Track population size history
        self.population_size_history.append(self.population_size)

    def _get_best_solution(self) -> Optional[Chromosome]:
        """
        Get the best solution from the current population.

        Returns:
            Best chromosome or None if population is empty
        """
        if not self.population:
            return None

        return max(
            self.population,
            key=lambda x: x.fitness if x.fitness != float("-inf") else float("-inf"),
        )

    def _get_bounds_dict(self) -> Dict[str, ParameterBounds]:
        """
        Get parameter bounds dictionary.

        Returns:
            Dictionary mapping parameter names to bounds
        """
        return self.parameter_bounds

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics.

        Returns:
            Statistics dictionary
        """
        if not self.population:
            return {
                "best_fitness": 0.0,
                "average_fitness": 0.0,
                "worst_fitness": 0.0,
                "population_size": 0,
                "generations": self.generations,
            }

        fitness_values = [
            chrom.fitness for chrom in self.population if chrom.fitness != float("-inf")
        ]

        if not fitness_values:
            return {
                "best_fitness": 0.0,
                "average_fitness": 0.0,
                "worst_fitness": 0.0,
                "population_size": len(self.population),
                "generations": self.generations,
            }

        return {
            "best_fitness": max(fitness_values),
            "average_fitness": np.mean(fitness_values),
            "worst_fitness": min(fitness_values),
            "population_size": len(self.population),
            "generations": self.generations,
        }

    def optimize(self, strategy_class, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization.

        Args:
            strategy_class: Strategy class to optimize
            data: Historical data for optimization

        Returns:
            Best parameter set found
        """
        # Handle case with no parameter bounds
        if not self.parameter_bounds:
            self.logger.warning("No parameter bounds defined, returning empty result")
            return {}

        start_time = time.time()

        self.logger.info("Starting Genetic Algorithm Optimization")
        self.logger.info(f"Population size: {self.population_size}")
        self.logger.info(f"Generations: {self.generations}")
        self.logger.info(f"Mutation rate: {self.mutation_rate}")
        self.logger.info(f"Crossover rate: {self.crossover_rate}")

        # Initialize population
        self._initialize_population()

        # Evaluate initial population
        self._evaluate_population(strategy_class, data)

        # Evolution loop
        for generation in range(self.generations):
            self.current_iteration = generation + 1

            self.logger.info(f"Generation {generation + 1}/{self.generations}")

            # Create new population
            self._create_next_generation()

            # Evaluate new population
            self._evaluate_population(strategy_class, data)

            # Update best chromosome
            self._update_best_chromosome()

            # Log generation statistics
            self._log_generation_stats(generation + 1)

            # Check for early termination
            if self.enable_early_termination and self._should_terminate_early():
                self.logger.info(
                    f"Early termination at generation {generation + 1} due to convergence"
                )
                break

        # Finalize optimization
        optimization_time = time.time() - start_time
        self.config["optimization_time"] = optimization_time

        self.logger.info(f"Genetic Algorithm completed in {optimization_time:.2f}s")
        self.logger.info(f"Best fitness: {self.best_fitness:.4f}")
        self.logger.info(f"Best parameters: {self.best_params}")

        # Ensure we return the best parameters found
        if self.best_chromosome and self.best_chromosome.genes:
            return self.best_chromosome.genes
        return self.best_params or {}

    def _initialize_population(self) -> None:
        """
        Initialize the population with random chromosomes.
        """
        self.population = []

        # Handle case with no parameter bounds
        if not self.parameter_bounds:
            # Create chromosomes with empty genes for test compatibility
            for _ in range(self.population_size):
                chromosome = Chromosome(genes={})
                self.population.append(chromosome)
            self.logger.info(
                f"Initialized population with {len(self.population)} chromosomes (empty genes)"
            )
            return

        for _ in range(self.population_size):
            genes = {}

            # Generate random values for each parameter
            for bounds in self.parameter_bounds.values():
                param_name = bounds.name
                if bounds.param_type == "categorical":
                    genes[param_name] = random.choice(
                        bounds.categories or [bounds.min_value]
                    )
                elif bounds.param_type == "int":
                    genes[param_name] = random.randint(
                        int(bounds.min_value), int(bounds.max_value)
                    )
                else:  # float
                    genes[param_name] = random.uniform(
                        bounds.min_value, bounds.max_value
                    )

            chromosome = Chromosome(genes=genes)
            self.population.append(chromosome)

        self.logger.info(
            f"Initialized population with {len(self.population)} chromosomes"
        )

    def _evaluate_population(self, strategy_class, data: pd.DataFrame) -> None:
        """
        Evaluate fitness of all chromosomes in population.

        Args:
            strategy_class: Strategy class to evaluate
            data: Historical data for backtesting
        """
        for chromosome in self.population:
            if chromosome.fitness == float("-inf"):  # Not evaluated yet
                # Check fitness cache first
                if self.enable_fitness_caching:
                    cache_key = self._get_cache_key(chromosome.genes)
                    if cache_key in self.fitness_cache:
                        chromosome.fitness = self.fitness_cache[cache_key]
                        self.update_best_params(chromosome.genes, chromosome.fitness)
                        continue

                # Create strategy instance with chromosome genes
                try:
                    strategy_config = {
                        "name": "ga_optimization",
                        "symbols": ["BTC/USDT"],
                        "timeframe": "1h",
                        "required_history": 100,
                        "params": chromosome.genes,
                    }

                    strategy_instance = strategy_class(strategy_config)

                    # Evaluate fitness
                    fitness = self.evaluate_fitness(strategy_instance, data)
                    chromosome.fitness = fitness

                    # Cache the result
                    if self.enable_fitness_caching:
                        self.fitness_cache[cache_key] = fitness

                    # Update best params tracking
                    self.update_best_params(chromosome.genes, fitness)

                except Exception as e:
                    self.logger.debug(f"Chromosome evaluation failed: {str(e)}")
                    chromosome.fitness = float("-inf")

    def _select_elite(self) -> List[Chromosome]:
        """
        Select elite individuals to preserve in next generation.

        Returns:
            List of elite chromosomes
        """
        # Sort population by fitness (descending)
        sorted_population = sorted(
            self.population,
            key=lambda x: x.fitness if x.fitness != float("-inf") else float("-inf"),
            reverse=True,
        )

        return sorted_population[: self.elite_count]

    def _tournament_selection(self) -> Chromosome:
        """
        Perform tournament selection.

        Returns:
            Selected chromosome
        """
        # Select random individuals for tournament
        tournament = random.sample(
            self.population, min(self.tournament_size, len(self.population))
        )

        # Return the best from tournament
        return max(
            tournament,
            key=lambda x: x.fitness if x.fitness != float("-inf") else float("-inf"),
        )

    def _crossover(
        self, parent1: Chromosome, parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Perform crossover between two parent chromosomes.

        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome

        Returns:
            Tuple of two offspring chromosomes
        """
        # Single-point crossover with support for different gene sets
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()

        # Get union of all parameter names from both parents
        all_params = sorted(set(genes1.keys()) | set(genes2.keys()))

        if len(all_params) > 1:
            # Choose crossover point
            crossover_point = random.randint(1, len(all_params) - 1)

            # Swap genes after crossover point, but only for parameters present in both parents
            for param in all_params[crossover_point:]:
                if param in genes1 and param in genes2:
                    genes1[param], genes2[param] = genes2[param], genes1[param]

        offspring1 = Chromosome(genes=genes1)
        offspring2 = Chromosome(genes=genes2)

        return offspring1, offspring2

    def _update_best_chromosome(self) -> None:
        """
        Update the best chromosome found so far.
        """
        if not self.population:
            return

        current_best = max(
            self.population,
            key=lambda x: x.fitness if x.fitness != float("-inf") else float("-inf"),
        )

        if (
            self.best_chromosome is None
            or current_best.fitness > self.best_chromosome.fitness
        ):
            self.best_chromosome = current_best.copy()

    def _log_generation_stats(self, generation: int) -> None:
        """
        Log statistics for the current generation and track adaptive metrics.

        Args:
            generation: Current generation number
        """
        if not self.population:
            return

        fitness_values = [
            chrom.fitness for chrom in self.population if chrom.fitness != float("-inf")
        ]

        if fitness_values:
            best_fitness = max(fitness_values)
            avg_fitness = np.mean(fitness_values)
            std_fitness = np.std(fitness_values)

            self.logger.info(
                f"Gen {generation}/{self.generations} | "
                f"Pop: {self.population_size} | "
                f"Best: {best_fitness:.4f} | "
                f"Avg: {avg_fitness:.4f} | "
                f"Std: {std_fitness:.4f} | "
                f"Params: {self.best_params}"
            )

            # Track metrics for adaptive population sizing
            self.fitness_history.append(best_fitness)
            self.diversity_history.append(std_fitness)

            # Store generation data for analysis
            self.results_history.append(
                {
                    "generation": generation,
                    "population_size": self.population_size,
                    "best_fitness": best_fitness,
                    "avg_fitness": avg_fitness,
                    "std_fitness": std_fitness,
                    "best_params": self.best_params.copy() if self.best_params else {},
                }
            )

    def get_ga_summary(self) -> Dict[str, Any]:
        """
        Get summary of genetic algorithm optimization process.

        Returns:
            Summary dictionary
        """
        summary = self.get_optimization_summary()
        summary.update(
            {
                "population_size": self.population_size,
                "initial_population_size": self.initial_population_size,
                "min_population_size": self.min_population_size,
                "max_population_size": self.max_population_size,
                "adaptation_rate": self.adaptation_rate,
                "population_size_history": self.population_size_history,
                "generations": self.generations,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "elitism_rate": self.elitism_rate,
                "tournament_size": self.tournament_size,
                "elite_count": self.elite_count,
                "final_population_fitness": [
                    chrom.fitness
                    for chrom in self.population
                    if chrom.fitness != float("-inf")
                ],
            }
        )

        return summary

    def get_population_diversity(self) -> Dict[str, Any]:
        """
        Calculate population diversity metrics.

        Returns:
            Diversity metrics dictionary
        """
        if not self.population:
            return {}

        # Calculate diversity for each parameter
        diversity = {}

        for bounds in self.parameter_bounds.values():
            param_name = bounds.name
            values = [
                chrom.genes.get(param_name)
                for chrom in self.population
                if param_name in chrom.genes
            ]

            if values:
                if isinstance(values[0], (int, float)):
                    diversity[param_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": min(values),
                        "max": max(values),
                        "unique_count": len(set(values)),
                    }
                else:  # categorical
                    value_counts = defaultdict(int)
                    for v in values:
                        value_counts[v] += 1
                    diversity[param_name] = dict(value_counts)

        return diversity

    def _initialize_random_state(self) -> None:
        """
        Initialize random state based on configuration.

        This method sets up deterministic random number generation for reproducible results.
        """
        if self.random_mode == "reproducible":
            # Use fixed seed for reproducible results
            worker_seed = self.base_seed + self.worker_id
            self.random_state = np.random.RandomState(worker_seed)
            random.seed(worker_seed)
            self.logger.info(
                f"Initialized reproducible random state with seed: {worker_seed}"
            )
        elif self.random_mode == "exploratory":
            # Use random seed for exploration
            self.random_state = np.random.RandomState()
            # Don't set python random seed to maintain some randomness
            self.logger.info("Initialized exploratory random state")
        else:
            raise ValueError(
                f"Invalid random_mode: {self.random_mode}. Must be 'reproducible' or 'exploratory'"
            )

        # Store initial state for checkpointing
        self._save_random_state()

    def _save_random_state(self) -> None:
        """
        Save current random state for checkpointing.
        """
        if self.random_state is not None:
            self.python_random_state = random.getstate()
            # Note: numpy random state is not directly serializable, so we store the seed

    def _restore_random_state(self) -> None:
        """
        Restore random state from checkpoint.
        """
        if self.python_random_state is not None:
            random.setstate(self.python_random_state)
        if self.random_state is not None and self.random_mode == "reproducible":
            # Reinitialize numpy random state with the same seed
            worker_seed = self.base_seed + self.worker_id
            self.random_state = np.random.RandomState(worker_seed)

    def set_random_seed(self, seed: int) -> None:
        """
        Set random seed for reproducible execution.

        Args:
            seed: Random seed value
        """
        self.base_seed = seed
        self.random_mode = "reproducible"
        self._initialize_random_state()

    def set_exploratory_mode(self) -> None:
        """
        Set exploratory mode for random seed execution.
        """
        self.random_mode = "exploratory"
        self._initialize_random_state()

    def get_random_state_info(self) -> Dict[str, Any]:
        """
        Get information about current random state.

        Returns:
            Dictionary with random state information
        """
        return {
            "random_mode": self.random_mode,
            "base_seed": self.base_seed,
            "worker_id": self.worker_id,
            "num_workers": self.num_workers,
            "has_numpy_state": self.random_state is not None,
            "has_python_state": self.python_random_state is not None,
        }

    def create_checkpoint(self) -> Dict[str, Any]:
        """
        Create a checkpoint of the current optimization state.

        Returns:
            Checkpoint data dictionary
        """
        checkpoint = {
            "random_mode": self.random_mode,
            "base_seed": self.base_seed,
            "worker_id": self.worker_id,
            "num_workers": self.num_workers,
            "population_size": self.population_size,
            "current_generation": getattr(self, "current_iteration", 0),
            "best_params": self.best_params,
            "best_fitness": self.best_fitness,
            "population": [
                {"genes": chrom.genes, "fitness": chrom.fitness}
                for chrom in self.population
            ]
            if self.population
            else [],
            "fitness_history": self.fitness_history.copy(),
            "diversity_history": self.diversity_history.copy(),
            "population_size_history": self.population_size_history.copy(),
            "results_history": self.results_history.copy(),
            "timestamp": time.time(),
        }

        self.checkpoint_data = checkpoint
        return checkpoint

    def restore_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Restore optimization state from checkpoint.

        Args:
            checkpoint: Checkpoint data dictionary
        """
        # Restore configuration
        self.random_mode = checkpoint.get("random_mode", "reproducible")
        self.base_seed = checkpoint.get("base_seed", 42)
        self.worker_id = checkpoint.get("worker_id", 0)
        self.num_workers = checkpoint.get("num_workers", 1)

        # Reinitialize random state
        self._initialize_random_state()

        # Restore optimization state
        self.population_size = checkpoint.get(
            "population_size", self.initial_population_size
        )
        self.current_iteration = checkpoint.get("current_generation", 0)
        self.best_params = checkpoint.get("best_params")
        self.best_fitness = checkpoint.get("best_fitness", float("-inf"))

        # Restore population
        population_data = checkpoint.get("population", [])
        self.population = []
        for chrom_data in population_data:
            chromosome = Chromosome(
                genes=chrom_data["genes"], fitness=chrom_data["fitness"]
            )
            self.population.append(chromosome)

        # Restore history
        self.fitness_history = checkpoint.get("fitness_history", [])
        self.diversity_history = checkpoint.get("diversity_history", [])
        self.population_size_history = checkpoint.get("population_size_history", [])
        self.results_history = checkpoint.get("results_history", [])

        # Update best chromosome
        if self.population:
            self.best_chromosome = max(
                self.population,
                key=lambda x: x.fitness
                if x.fitness != float("-inf")
                else float("-inf"),
            )

        self.checkpoint_data = checkpoint
        self.logger.info(
            f"Restored optimization state from checkpoint (generation {self.current_iteration})"
        )

    def validate_reproducibility(
        self, strategy_class, data: pd.DataFrame, num_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Validate that optimization produces reproducible results.

        Args:
            strategy_class: Strategy class to optimize
            data: Historical data for optimization
            num_runs: Number of runs to validate reproducibility

        Returns:
            Validation results dictionary
        """
        if self.random_mode != "reproducible":
            raise ValueError(
                "Reproducibility validation requires reproducible random mode"
            )

        results = []
        original_seed = self.base_seed

        for run in range(num_runs):
            # Reset random state for each run
            self.set_random_seed(original_seed)

            # Run optimization
            result = self.optimize(strategy_class, data)
            results.append(result)

        # Check reproducibility
        all_same = all(self._params_equal(results[0], result) for result in results[1:])

        return {
            "is_reproducible": all_same,
            "num_runs": num_runs,
            "results": results,
            "validation_passed": all_same,
        }

    def _params_equal(
        self, params1: Dict[str, Any], params2: Dict[str, Any], tolerance: float = 1e-10
    ) -> bool:
        """
        Check if two parameter dictionaries are equal within tolerance.

        Args:
            params1: First parameter set
            params2: Second parameter set
            tolerance: Numerical tolerance for floating point comparison

        Returns:
            True if parameters are equal
        """
        if set(params1.keys()) != set(params2.keys()):
            return False

        for key in params1.keys():
            val1, val2 = params1[key], params2[key]
            if isinstance(val1, float) and isinstance(val2, float):
                if abs(val1 - val2) > tolerance:
                    return False
            elif val1 != val2:
                return False

        return True

    def get_worker_seed(self, generation: Optional[int] = None) -> int:
        """
        Get isolated seed for parallel worker execution.

        Args:
            generation: Current generation (for additional seed variation)

        Returns:
            Worker-specific seed
        """
        seed = self.base_seed + self.worker_id
        if generation is not None:
            seed += generation * self.num_workers
        return seed

    def set_worker_config(self, worker_id: int, num_workers: int) -> None:
        """
        Configure optimizer for parallel execution.

        Args:
            worker_id: Unique ID for this worker (0 to num_workers-1)
            num_workers: Total number of parallel workers
        """
        self.worker_id = worker_id
        self.num_workers = num_workers
        self._initialize_random_state()
        self.logger.info(
            f"Configured for parallel execution: worker {worker_id}/{num_workers}"
        )

    def _get_cache_key(self, params: Dict[str, Any]) -> str:
        """
        Generate a cache key for parameter set.

        Args:
            params: Parameter dictionary

        Returns:
            String key for caching
        """
        # Sort parameters for consistent hashing
        sorted_items = sorted(params.items())
        return str(sorted_items)

    def _should_terminate_early(self) -> bool:
        """
        Check if optimization should terminate early due to convergence.

        Returns:
            True if optimization should terminate early
        """
        if len(self.fitness_history) < self.early_termination_window:
            return False

        # Check if fitness improvement has been minimal over the last few generations
        recent_fitness = self.fitness_history[-self.early_termination_window :]

        # Calculate improvement over the window
        max_improvement = max(recent_fitness) - min(recent_fitness)

        # Terminate if improvement is below threshold
        return max_improvement < self.early_termination_threshold
