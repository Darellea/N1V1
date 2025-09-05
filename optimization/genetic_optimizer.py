"""
optimization/genetic_optimizer.py

Genetic Algorithm implementation for strategy parameter optimization.
Uses evolutionary principles to find optimal parameter combinations.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import random
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
import numpy as np

from .base_optimizer import BaseOptimizer, ParameterBounds


@dataclass
class Chromosome:
    """Represents a parameter set as a chromosome for GA."""

    genes: Dict[str, Any]  # Parameter name -> value mapping
    fitness: float = float('-inf')

    def __post_init__(self):
        """Validate chromosome after initialization."""
        pass  # Allow empty genes for graceful handling

    def copy(self) -> 'Chromosome':
        """Create a copy of this chromosome."""
        return Chromosome(
            genes=self.genes.copy(),
            fitness=self.fitness
        )

    def mutate(self, mutation_rate: float, parameter_bounds: Dict[str, ParameterBounds]) -> None:
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

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Genetic Algorithm Optimizer.

        Args:
            config: Configuration dictionary containing:
                - population_size: Number of chromosomes in population
                - generations: Number of generations to evolve
                - mutation_rate: Probability of gene mutation
                - crossover_rate: Probability of crossover
                - elitism_rate: Fraction of best individuals to preserve
                - tournament_size: Size of tournament for selection
        """
        super().__init__(config)

        # GA specific configuration
        self.population_size = config.get('population_size', 20)
        self.generations = config.get('generations', 10)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.7)
        self.elitism_rate = config.get('elitism_rate', 0.1)
        self.tournament_size = config.get('tournament_size', 3)

        # Population state
        self.population: List[Chromosome] = []
        self.best_chromosome: Optional[Chromosome] = None

        # Elitism count
        self.elite_count = max(1, int(self.population_size * self.elitism_rate))

        # Make Chromosome accessible within the class
        self.Chromosome = Chromosome

        # Override parameter bounds to use list for index access
        self.parameter_bounds: List[ParameterBounds] = []

    def add_parameter_bounds(self, bounds: ParameterBounds) -> None:
        """
        Add parameter bounds for validation.

        Args:
            bounds: ParameterBounds object defining constraints
        """
        self.parameter_bounds.append(bounds)

    def _calculate_fitness(self, strategy, data: pd.DataFrame, params: Dict[str, Any]) -> float:
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
        """
        new_population = []

        # Elitism: preserve best individuals
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
        self.population = new_population[:self.population_size]

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
            key=lambda x: x.fitness if x.fitness != float('-inf') else float('-inf')
        )

    def _get_bounds_dict(self) -> Dict[str, ParameterBounds]:
        """
        Convert parameter bounds list to dictionary for compatibility.

        Returns:
            Dictionary mapping parameter names to bounds
        """
        return {bounds.name: bounds for bounds in self.parameter_bounds}

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics.

        Returns:
            Statistics dictionary
        """
        if not self.population:
            return {
                'best_fitness': 0.0,
                'average_fitness': 0.0,
                'worst_fitness': 0.0,
                'population_size': 0,
                'generations': self.generations
            }

        fitness_values = [
            chrom.fitness for chrom in self.population
            if chrom.fitness != float('-inf')
        ]

        if not fitness_values:
            return {
                'best_fitness': 0.0,
                'average_fitness': 0.0,
                'worst_fitness': 0.0,
                'population_size': len(self.population),
                'generations': self.generations
            }

        return {
            'best_fitness': max(fitness_values),
            'average_fitness': np.mean(fitness_values),
            'worst_fitness': min(fitness_values),
            'population_size': len(self.population),
            'generations': self.generations
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

        # Finalize optimization
        optimization_time = time.time() - start_time
        self.config['optimization_time'] = optimization_time

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
            self.logger.info(f"Initialized population with {len(self.population)} chromosomes (empty genes)")
            return

        for _ in range(self.population_size):
            genes = {}

            # Generate random values for each parameter
            for bounds in self.parameter_bounds:
                param_name = bounds.name
                if bounds.param_type == "categorical":
                    genes[param_name] = random.choice(bounds.categories or [bounds.min_value])
                elif bounds.param_type == "int":
                    genes[param_name] = random.randint(int(bounds.min_value), int(bounds.max_value))
                else:  # float
                    genes[param_name] = random.uniform(bounds.min_value, bounds.max_value)

            chromosome = Chromosome(genes=genes)
            self.population.append(chromosome)

        self.logger.info(f"Initialized population with {len(self.population)} chromosomes")

    def _evaluate_population(self, strategy_class, data: pd.DataFrame) -> None:
        """
        Evaluate fitness of all chromosomes in population.

        Args:
            strategy_class: Strategy class to evaluate
            data: Historical data for backtesting
        """
        for chromosome in self.population:
            if chromosome.fitness == float('-inf'):  # Not evaluated yet
                # Create strategy instance with chromosome genes
                try:
                    strategy_config = {
                        'name': 'ga_optimization',
                        'symbols': ['BTC/USDT'],
                        'timeframe': '1h',
                        'required_history': 100,
                        'params': chromosome.genes
                    }

                    strategy_instance = strategy_class(strategy_config)

                    # Evaluate fitness
                    fitness = self.evaluate_fitness(strategy_instance, data)
                    chromosome.fitness = fitness

                    # Update best params tracking
                    self.update_best_params(chromosome.genes, fitness)

                except Exception as e:
                    self.logger.debug(f"Chromosome evaluation failed: {str(e)}")
                    chromosome.fitness = float('-inf')

    def _select_elite(self) -> List[Chromosome]:
        """
        Select elite individuals to preserve in next generation.

        Returns:
            List of elite chromosomes
        """
        # Sort population by fitness (descending)
        sorted_population = sorted(
            self.population,
            key=lambda x: x.fitness if x.fitness != float('-inf') else float('-inf'),
            reverse=True
        )

        return sorted_population[:self.elite_count]

    def _tournament_selection(self) -> Chromosome:
        """
        Perform tournament selection.

        Returns:
            Selected chromosome
        """
        # Select random individuals for tournament
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))

        # Return the best from tournament
        return max(tournament, key=lambda x: x.fitness if x.fitness != float('-inf') else float('-inf'))

    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform crossover between two parent chromosomes.

        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome

        Returns:
            Tuple of two offspring chromosomes
        """
        # Single-point crossover
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()

        # Get all parameter names
        param_names = list(genes1.keys())

        if len(param_names) > 1:
            # Choose crossover point
            crossover_point = random.randint(1, len(param_names) - 1)

            # Swap genes after crossover point
            for i in range(crossover_point, len(param_names)):
                param = param_names[i]
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
            key=lambda x: x.fitness if x.fitness != float('-inf') else float('-inf')
        )

        if self.best_chromosome is None or current_best.fitness > self.best_chromosome.fitness:
            self.best_chromosome = current_best.copy()

    def _log_generation_stats(self, generation: int) -> None:
        """
        Log statistics for the current generation.

        Args:
            generation: Current generation number
        """
        if not self.population:
            return

        fitness_values = [
            chrom.fitness for chrom in self.population
            if chrom.fitness != float('-inf')
        ]

        if fitness_values:
            best_fitness = max(fitness_values)
            avg_fitness = np.mean(fitness_values)
            std_fitness = np.std(fitness_values)

            self.logger.info(
                f"Gen {generation}/{self.generations} | "
                f"Best: {best_fitness:.4f} | "
                f"Avg: {avg_fitness:.4f} | "
                f"Std: {std_fitness:.4f} | "
                f"Params: {self.best_params}"
            )

            # Store generation data for analysis
            self.results_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': std_fitness,
                'best_params': self.best_params.copy() if self.best_params else {}
            })

    def get_ga_summary(self) -> Dict[str, Any]:
        """
        Get summary of genetic algorithm optimization process.

        Returns:
            Summary dictionary
        """
        summary = self.get_optimization_summary()
        summary.update({
            'population_size': self.population_size,
            'generations': self.generations,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'elitism_rate': self.elitism_rate,
            'tournament_size': self.tournament_size,
            'elite_count': self.elite_count,
            'final_population_fitness': [
                chrom.fitness for chrom in self.population
                if chrom.fitness != float('-inf')
            ]
        })

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

        for bounds in self.parameter_bounds:
            param_name = bounds.name
            values = [
                chrom.genes.get(param_name) for chrom in self.population
                if param_name in chrom.genes
            ]

            if values:
                if isinstance(values[0], (int, float)):
                    diversity[param_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': min(values),
                        'max': max(values),
                        'unique_count': len(set(values))
                    }
                else:  # categorical
                    value_counts = defaultdict(int)
                    for v in values:
                        value_counts[v] += 1
                    diversity[param_name] = dict(value_counts)

        return diversity
