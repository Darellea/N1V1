"""
Genome module for Strategy Generator

This module contains the core genetic representation classes for trading strategies.
It provides the building blocks for evolutionary optimization of trading strategies.

Classes:
- StrategyComponent: Enum of strategy component types
- IndicatorType: Enum of technical indicators
- SignalLogic: Enum of signal generation logic
- StrategyGene: Individual gene representing a strategy component
- StrategyGenome: Complete genetic representation of a trading strategy
- Species: Species for maintaining population diversity
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import json
from collections import defaultdict

import numpy as np

# Type aliases for better code readability
GeneList = List['StrategyGene']
GenomeList = List['StrategyGenome']


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
    """
    Individual gene representing a strategy component.

    This class encapsulates a single component of a trading strategy,
    including its type, parameters, and configuration.
    """
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
    """
    Complete genetic representation of a trading strategy.

    This class represents an entire trading strategy as a collection of genes,
    providing methods for genetic operations like mutation and crossover.
    """
    genes: GeneList = field(default_factory=list)
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
            # Log warning instead of raising exception for flexibility
            pass

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
        """
        Apply mutations to this genome.

        Args:
            mutation_rate: Probability of mutating each gene

        Returns:
            Self for method chaining
        """
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
        """
        Perform crossover with another genome.

        Args:
            other: Another genome to crossover with

        Returns:
            Tuple of two child genomes
        """
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
    def select_best(population: GenomeList, num_to_select: int) -> GenomeList:
        """
        Select the best N genomes by fitness.

        Args:
            population: List of genomes to select from
            num_to_select: Number of genomes to select

        Returns:
            List of best genomes
        """
        if not population:
            return []

        # Sort by fitness in descending order
        sorted_population = sorted(population, key=lambda g: g.fitness, reverse=True)
        return sorted_population[:num_to_select]

    @staticmethod
    def tournament_selection(population: GenomeList, tournament_size: int) -> 'StrategyGenome':
        """
        Perform tournament selection to choose a parent.

        Args:
            population: Population to select from
            tournament_size: Size of tournament

        Returns:
            Selected genome
        """
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
    """
    Species for maintaining population diversity.

    This class groups similar genomes together to maintain diversity
    in the population during evolution.
    """
    species_id: str
    representative: StrategyGenome
    members: GenomeList = field(default_factory=list)
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
        """
        Calculate diversity score within species.

        Returns:
            Diversity score (higher = more diverse)
        """
        if len(self.members) < 2:
            return 0.0

        # Simple diversity based on fitness variance
        fitness_values = [g.fitness for g in self.members if g.fitness != float('-inf')]
        if len(fitness_values) < 2:
            return 0.0

        return np.std(fitness_values) / (np.mean(fitness_values) + 1e-6)
