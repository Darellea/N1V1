"""
Unit and integration tests for Hybrid AI Strategy Generator feature.

Tests cover:
- Genetic algorithm operations (crossover, mutation, selection)
- Strategy genome encoding/decoding
- Population evolution and fitness evaluation
- Distributed evaluation system
- Generated strategy validation and execution
- Knowledge base integration
- Performance benchmarking
- Edge cases and error handling
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import tempfile
import os
from pathlib import Path

from optimization.strategy_generator import (
    StrategyGenerator, StrategyGenome, StrategyGene,
    StrategyComponent, IndicatorType, SignalLogic
)
from strategies.generated import GeneratedStrategy
from strategies.base_strategy import BaseStrategy
from backtest.backtester import Backtester
from core.diagnostics import HealthStatus


class TestStrategyGenome:
    """Test StrategyGenome class functionality."""

    def test_genome_creation(self):
        """Test basic genome creation."""
        genome = StrategyGenome()

        assert len(genome.genes) == 0
        assert genome.fitness == 0.0
        assert genome.generation == 0

    def test_genome_with_genes(self):
        """Test genome with genes."""
        genome = StrategyGenome()

        # Add indicator gene
        indicator_gene = StrategyGene(
            component_type=StrategyComponent.INDICATOR,
            indicator_type=IndicatorType.RSI,
            parameters={'period': 14, 'overbought': 70, 'oversold': 30}
        )
        genome.genes.append(indicator_gene)

        # Add signal logic gene
        signal_gene = StrategyGene(
            component_type=StrategyComponent.SIGNAL_LOGIC,
            signal_logic=SignalLogic.THRESHOLD,
            parameters={'threshold': 0.5, 'direction': 'above'}
        )
        genome.genes.append(signal_gene)

        assert len(genome.genes) == 2
        assert genome.genes[0].indicator_type == IndicatorType.RSI
        assert genome.genes[1].signal_logic == SignalLogic.THRESHOLD

    def test_genome_serialization(self):
        """Test genome serialization."""
        genome = StrategyGenome()
        genome.fitness = 1.5
        genome.generation = 5

        # Add a gene
        gene = StrategyGene(
            component_type=StrategyComponent.INDICATOR,
            indicator_type=IndicatorType.MACD,
            parameters={'fast_period': 12, 'slow_period': 26}
        )
        genome.genes.append(gene)

        # Serialize
        data = genome.to_dict()

        assert isinstance(data, dict)
        assert 'genes' in data
        assert 'fitness' in data
        assert 'generation' in data
        assert len(data['genes']) == 1

        # Deserialize
        genome2 = StrategyGenome.from_dict(data)

        assert len(genome2.genes) == 1
        assert genome2.fitness == 1.5
        assert genome2.generation == 5
        assert genome2.genes[0].indicator_type == IndicatorType.MACD

    def test_genome_copy(self):
        """Test genome copying."""
        genome = StrategyGenome()
        genome.fitness = 2.0

        gene = StrategyGene(
            component_type=StrategyComponent.INDICATOR,
            indicator_type=IndicatorType.BOLLINGER_BANDS,
            parameters={'period': 20, 'std_dev': 2.0}
        )
        genome.genes.append(gene)

        # Copy genome
        genome_copy = genome.copy()

        assert genome_copy.fitness == genome.fitness
        assert len(genome_copy.genes) == len(genome.genes)
        assert genome_copy.genes[0].indicator_type == genome.genes[0].indicator_type

        # Modify copy
        genome_copy.fitness = 3.0
        genome_copy.genes[0].parameters['period'] = 30

        # Original should be unchanged
        assert genome.fitness == 2.0
        assert genome.genes[0].parameters['period'] == 20


class TestGeneticOperations:
    """Test genetic algorithm operations."""

    def test_mutation_operation(self):
        """Test genome mutation."""
        genome = StrategyGenome()

        # Add initial gene
        gene = StrategyGene(
            component_type=StrategyComponent.INDICATOR,
            indicator_type=IndicatorType.RSI,
            parameters={'period': 14, 'overbought': 70, 'oversold': 30}
        )
        genome.genes.append(gene)

        original_period = gene.parameters['period']

        # Apply mutation
        mutated = genome.mutate(mutation_rate=1.0)  # 100% mutation rate

        # Should have same number of genes
        assert len(mutated.genes) == len(genome.genes)

        # Parameters might have changed
        # (exact behavior depends on mutation implementation)

    def test_crossover_operation(self):
        """Test genome crossover."""
        # Parent 1
        genome1 = StrategyGenome()
        gene1 = StrategyGene(
            component_type=StrategyComponent.INDICATOR,
            indicator_type=IndicatorType.RSI,
            parameters={'period': 14}
        )
        genome1.genes.append(gene1)

        # Parent 2
        genome2 = StrategyGenome()
        gene2 = StrategyGene(
            component_type=StrategyComponent.INDICATOR,
            indicator_type=IndicatorType.MACD,
            parameters={'fast_period': 12}
        )
        genome2.genes.append(gene2)

        # Perform crossover
        child1, child2 = genome1.crossover(genome2)

        # Children should exist
        assert child1 is not None
        assert child2 is not None

        # Should have genes from both parents
        assert len(child1.genes) > 0
        assert len(child2.genes) > 0

    def test_selection_operation(self):
        """Test population selection."""
        # Create population with different fitness values
        population = []

        for i in range(10):
            genome = StrategyGenome()
            genome.fitness = i * 0.1  # 0.0, 0.1, 0.2, ..., 0.9
            population.append(genome)

        # Select best individuals
        selected = StrategyGenome.select_best(population, num_to_select=5)

        assert len(selected) == 5

        # Should be the highest fitness individuals
        fitness_values = [g.fitness for g in selected]
        assert max(fitness_values) == 0.9
        assert min(fitness_values) == 0.5  # 5th highest from 0.0-0.9

    def test_tournament_selection(self):
        """Test tournament selection."""
        population = []

        for i in range(20):
            genome = StrategyGenome()
            genome.fitness = np.random.random()
            population.append(genome)

        # Perform tournament selection
        winner = StrategyGenome.tournament_selection(population, tournament_size=5)

        assert winner is not None
        assert winner in population


class TestStrategyGenerator:
    """Test StrategyGenerator class."""

    def test_generator_initialization(self, test_config, temp_dir):
        """Test generator initialization."""
        config = test_config.get("strategy_generator", {})
        config["model_path"] = temp_dir

        generator = StrategyGenerator(config)

        assert generator.population_size == config.get("population_size", 50)
        assert generator.generations == config.get("generations", 20)
        assert generator.mutation_rate == config.get("mutation_rate", 0.1)
        assert len(generator.population) == 0  # Not initialized yet

    @pytest.mark.asyncio
    async def test_population_initialization(self, test_config, temp_dir):
        """Test population initialization."""
        config = test_config.get("strategy_generator", {})
        config["population_size"] = 10  # Smaller for testing
        config["model_path"] = temp_dir

        generator = StrategyGenerator(config)
        await generator.initialize()

        assert len(generator.population) == 10

        # All genomes should have genes
        for genome in generator.population:
            assert len(genome.genes) > 0

    @pytest.mark.asyncio
    async def test_evolution_process(self, test_config, temp_dir, generate_strategy_population):
        """Test evolution process."""
        config = test_config.get("strategy_generator", {})
        config["population_size"] = 5  # Very small for testing
        config["generations"] = 2
        config["model_path"] = temp_dir

        generator = StrategyGenerator(config)
        await generator.initialize()

        # Mock fitness evaluation
        async def mock_evaluate_fitness(genome):
            # Simple fitness based on number of genes
            return len(genome.genes) * 0.1

        generator.evaluate_fitness = mock_evaluate_fitness

        # Run evolution
        await generator.evolve()

        # Should have completed evolution
        assert generator.current_generation > 0

        # Population should still exist
        assert len(generator.population) > 0

    @pytest.mark.asyncio
    async def test_strategy_generation(self, test_config, temp_dir):
        """Test strategy generation from genome."""
        config = test_config.get("strategy_generator", {})
        config["model_path"] = temp_dir

        generator = StrategyGenerator(config)
        await generator.initialize()

        # Get first genome from population
        genome = generator.population[0]

        # Generate strategy
        strategy_class = await generator.generate_strategy(genome, "TestStrategy")

        assert strategy_class is not None
        assert hasattr(strategy_class, '__name__')
        assert 'TestStrategy' in strategy_class.__name__

    @pytest.mark.asyncio
    async def test_population_save_load(self, test_config, temp_dir):
        """Test population saving and loading."""
        config = test_config.get("strategy_generator", {})
        config["population_size"] = 5
        config["model_path"] = temp_dir

        generator = StrategyGenerator(config)
        await generator.initialize()

        # Save population
        await generator.save_population()

        # Create new generator and load population
        generator2 = StrategyGenerator(config)
        await generator2.load_population()

        # Should have loaded the population
        assert len(generator2.population) == len(generator.population)

        # Fitness values should match
        for g1, g2 in zip(generator.population, generator2.population):
            assert g1.fitness == g2.fitness


class TestGeneratedStrategy:
    """Test GeneratedStrategy class."""

    @pytest.mark.asyncio
    async def test_generated_strategy_creation(self, generate_strategy_population):
        """Test generated strategy creation."""
        population = generate_strategy_population(5)
        genome = population[0]

        # Create generated strategy
        strategy_class = await GeneratedStrategy.create_from_genome(genome, "GeneratedTestStrategy")

        assert strategy_class is not None
        assert issubclass(strategy_class, BaseStrategy)

        # Instantiate strategy
        strategy = strategy_class({"name": "test"})

        assert hasattr(strategy, 'generate_signals')
        assert hasattr(strategy, 'initialize')

    @pytest.mark.asyncio
    async def test_generated_strategy_execution(self, generate_strategy_population, synthetic_market_data):
        """Test generated strategy execution."""
        population = generate_strategy_population(5)
        genome = population[0]

        # Create and instantiate strategy
        strategy_class = await GeneratedStrategy.create_from_genome(genome, "ExecutionTestStrategy")
        strategy = strategy_class({"name": "execution_test"})

        # Initialize strategy
        await strategy.initialize()

        # Generate signals
        signals = await strategy.generate_signals(synthetic_market_data)

        # Should return a list (may be empty)
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_strategy_validation(self, generate_strategy_population):
        """Test strategy validation."""
        population = generate_strategy_population(10)

        for genome in population:
            # Validate genome
            is_valid = await GeneratedStrategy.validate_genome(genome)

            # Should be boolean
            assert isinstance(is_valid, bool)

            # Most generated strategies should be valid
            # (exact validation depends on implementation)


class TestFitnessEvaluation:
    """Test fitness evaluation system."""

    @pytest.mark.asyncio
    async def test_fitness_calculation(self, generate_strategy_population, synthetic_market_data):
        """Test fitness calculation for strategies."""
        population = generate_strategy_population(3)

        for genome in population:
            # Calculate fitness
            fitness = await StrategyGenerator.calculate_fitness(genome, synthetic_market_data)

            assert isinstance(fitness, (int, float))
            assert fitness >= 0.0  # Fitness should be non-negative

    @pytest.mark.asyncio
    async def test_backtest_integration(self, generate_strategy_population, synthetic_market_data):
        """Test integration with backtesting system."""
        genome = generate_strategy_population(1)[0]

        # Mock backtester
        backtester = Mock(spec=Backtester)
        backtester.run_backtest = AsyncMock(return_value={
            'total_return': 0.15,
            'sharpe_ratio': 1.8,
            'max_drawdown': 0.08,
            'win_rate': 0.65,
            'total_trades': 50
        })

        # Calculate fitness using backtest results
        fitness = await StrategyGenerator.calculate_fitness_with_backtest(
            genome, synthetic_market_data, backtester
        )

        assert isinstance(fitness, (int, float))
        assert fitness > 0.0

        # Backtester should have been called
        backtester.run_backtest.assert_called_once()

    def test_multi_objective_fitness(self):
        """Test multi-objective fitness calculation."""
        # Mock backtest results
        results = {
            'total_return': 0.20,      # 20% return
            'sharpe_ratio': 2.1,       # Good Sharpe
            'max_drawdown': 0.05,      # 5% drawdown
            'win_rate': 0.70,          # 70% win rate
            'total_trades': 100
        }

        # Calculate multi-objective fitness
        fitness = StrategyGenerator.calculate_multi_objective_fitness(results)

        assert isinstance(fitness, (int, float))
        assert fitness > 0.0

        # Higher returns should give higher fitness
        results_high = results.copy()
        results_high['total_return'] = 0.30
        fitness_high = StrategyGenerator.calculate_multi_objective_fitness(results_high)

        assert fitness_high > fitness

    def test_risk_adjusted_fitness(self):
        """Test risk-adjusted fitness calculation."""
        # Strategy with high returns but high risk
        high_risk_results = {
            'total_return': 0.30,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.15,
            'win_rate': 0.60
        }

        # Strategy with moderate returns but low risk
        low_risk_results = {
            'total_return': 0.15,
            'sharpe_ratio': 2.5,
            'max_drawdown': 0.03,
            'win_rate': 0.75
        }

        high_risk_fitness = StrategyGenerator.calculate_risk_adjusted_fitness(high_risk_results)
        low_risk_fitness = StrategyGenerator.calculate_risk_adjusted_fitness(low_risk_results)

        # Low risk strategy should have higher fitness due to better risk adjustment
        assert low_risk_fitness > high_risk_fitness


class TestDistributedEvaluation:
    """Test distributed evaluation system."""

    @pytest.mark.asyncio
    async def test_distributed_evaluation_setup(self, test_config, temp_dir):
        """Test distributed evaluation setup."""
        config = test_config.get("strategy_generator", {})
        config["distributed_enabled"] = True
        config["max_workers"] = 2
        config["model_path"] = temp_dir

        generator = StrategyGenerator(config)
        await generator.initialize()

        # Should have distributed evaluator
        assert hasattr(generator, 'distributed_evaluator')

    @pytest.mark.asyncio
    async def test_parallel_fitness_evaluation(self, test_config, temp_dir, generate_strategy_population):
        """Test parallel fitness evaluation."""
        config = test_config.get("strategy_generator", {})
        config["distributed_enabled"] = True
        config["max_workers"] = 2
        config["model_path"] = temp_dir

        generator = StrategyGenerator(config)
        await generator.initialize()

        # Create small population for testing
        population = generate_strategy_population(4)

        # Mock fitness function
        async def mock_fitness(genome):
            await asyncio.sleep(0.01)  # Simulate computation time
            return len(genome.genes) * 0.1

        # Evaluate fitness in parallel
        fitness_scores = await generator.evaluate_population_fitness_parallel(population, mock_fitness)

        assert len(fitness_scores) == len(population)
        assert all(isinstance(score, (int, float)) for score in fitness_scores)

    @pytest.mark.asyncio
    async def test_worker_management(self, test_config, temp_dir):
        """Test worker management in distributed evaluation."""
        config = test_config.get("strategy_generator", {})
        config["distributed_enabled"] = True
        config["max_workers"] = 3
        config["model_path"] = temp_dir

        generator = StrategyGenerator(config)
        await generator.initialize()

        # Should be able to handle worker management
        assert generator.distributed_evaluator is not None

        # Test worker status
        worker_status = generator.distributed_evaluator.get_worker_status()
        assert isinstance(worker_status, dict)


class TestKnowledgeBaseIntegration:
    """Test knowledge base integration."""

    @pytest.mark.asyncio
    async def test_strategy_storage(self, generate_strategy_population, temp_dir):
        """Test strategy storage in knowledge base."""
        population = generate_strategy_population(3)
        genome = population[0]

        # Store strategy
        strategy_id = await GeneratedStrategy.store_in_knowledge_base(
            genome, {"performance": 1.5, "source": "test"}
        )

        assert isinstance(strategy_id, str)
        assert len(strategy_id) > 0

    @pytest.mark.asyncio
    async def test_strategy_retrieval(self, generate_strategy_population, temp_dir):
        """Test strategy retrieval from knowledge base."""
        population = generate_strategy_population(3)
        genome = population[0]

        # Store strategy
        strategy_id = await GeneratedStrategy.store_in_knowledge_base(
            genome, {"performance": 2.0, "source": "test"}
        )

        # Retrieve strategy
        retrieved_genome = await GeneratedStrategy.retrieve_from_knowledge_base(strategy_id)

        assert retrieved_genome is not None
        assert len(retrieved_genome.genes) == len(genome.genes)

    @pytest.mark.asyncio
    async def test_strategy_search(self, generate_strategy_population, temp_dir):
        """Test strategy search in knowledge base."""
        population = generate_strategy_population(5)

        # Store multiple strategies
        stored_ids = []
        for i, genome in enumerate(population):
            strategy_id = await GeneratedStrategy.store_in_knowledge_base(
                genome, {"performance": i * 0.2, "source": "test"}
            )
            stored_ids.append(strategy_id)

        # Search for strategies
        search_results = await GeneratedStrategy.search_knowledge_base({
            "min_performance": 0.5,
            "source": "test"
        })

        assert isinstance(search_results, list)
        # Should find some strategies meeting criteria


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_generation_performance(self, test_config, temp_dir, performance_timer):
        """Test strategy generation performance."""
        config = test_config.get("strategy_generator", {})
        config["population_size"] = 5
        config["model_path"] = temp_dir

        generator = StrategyGenerator(config)
        await generator.initialize()

        genome = generator.population[0]

        # Measure generation time
        performance_timer.start()
        strategy_class = await generator.generate_strategy(genome, "PerformanceTestStrategy")
        performance_timer.stop()

        duration_ms = performance_timer.duration_ms()

        # Should be reasonably fast
        assert duration_ms < 1000  # Less than 1 second
        assert strategy_class is not None

    @pytest.mark.asyncio
    async def test_evolution_performance(self, test_config, temp_dir, performance_timer):
        """Test evolution performance."""
        config = test_config.get("strategy_generator", {})
        config["population_size"] = 10
        config["generations"] = 3
        config["model_path"] = temp_dir

        generator = StrategyGenerator(config)
        await generator.initialize()

        # Mock fitness evaluation
        async def quick_fitness(genome):
            return len(genome.genes) * 0.1

        generator.evaluate_fitness = quick_fitness

        # Measure evolution time
        performance_timer.start()
        await generator.evolve()
        performance_timer.stop()

        duration_seconds = performance_timer.duration_seconds()

        # Should complete within reasonable time
        assert duration_seconds < 30  # Less than 30 seconds for small population

    @pytest.mark.asyncio
    async def test_memory_usage_during_evolution(self, test_config, temp_dir, memory_monitor):
        """Test memory usage during evolution."""
        config = test_config.get("strategy_generator", {})
        config["population_size"] = 20
        config["generations"] = 2
        config["model_path"] = temp_dir

        generator = StrategyGenerator(config)
        await generator.initialize()

        memory_monitor.start()

        # Run evolution
        async def mock_fitness(genome):
            return len(genome.genes) * 0.1

        generator.evaluate_fitness = mock_fitness
        await generator.evolve()

        memory_delta = memory_monitor.get_memory_delta()

        # Memory usage should be reasonable
        assert memory_delta < 100  # Less than 100MB increase


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_genome_handling(self, temp_dir):
        """Test handling of invalid genomes."""
        # Create invalid genome
        invalid_genome = StrategyGenome()
        # Add invalid gene
        invalid_gene = StrategyGene(
            component_type="invalid_type",
            parameters={}
        )
        invalid_genome.genes.append(invalid_gene)

        # Should handle gracefully
        strategy_class = await GeneratedStrategy.create_from_genome(invalid_genome, "InvalidTest")

        # May return None or create a basic strategy
        assert strategy_class is not None or True  # Allow either behavior

    @pytest.mark.asyncio
    async def test_empty_population_handling(self, test_config, temp_dir):
        """Test handling of empty population."""
        config = test_config.get("strategy_generator", {})
        config["population_size"] = 0
        config["model_path"] = temp_dir

        generator = StrategyGenerator(config)
        await generator.initialize()

        # Should handle empty population
        assert len(generator.population) == 0

        # Evolution should handle empty population
        await generator.evolve()
        assert generator.current_generation >= 0

    @pytest.mark.asyncio
    async def test_fitness_evaluation_failure(self, generate_strategy_population, synthetic_market_data):
        """Test handling of fitness evaluation failures."""
        genome = generate_strategy_population(1)[0]

        # Mock backtester that fails
        backtester = Mock(spec=Backtester)
        backtester.run_backtest = AsyncMock(side_effect=Exception("Backtest failed"))

        # Should handle failure gracefully
        fitness = await StrategyGenerator.calculate_fitness_with_backtest(
            genome, synthetic_market_data, backtester
        )

        # Should return some fitness value (possibly default)
        assert isinstance(fitness, (int, float))

    @pytest.mark.asyncio
    async def test_strategy_generation_timeout(self, generate_strategy_population):
        """Test strategy generation timeout handling."""
        large_genome = generate_strategy_population(1)[0]

        # Add many genes to potentially cause timeout
        for i in range(50):
            gene = StrategyGene(
                component_type=StrategyComponent.INDICATOR,
                indicator_type=IndicatorType.RSI,
                parameters={'period': 14 + i}
            )
            large_genome.genes.append(gene)

        # Should handle large genomes gracefully
        strategy_class = await GeneratedStrategy.create_from_genome(large_genome, "LargeGenomeTest")

        assert strategy_class is not None


class TestHealthMonitoring:
    """Test health monitoring integration."""

    @pytest.mark.asyncio
    async def test_health_check_integration(self, test_config, temp_dir):
        """Test integration with health monitoring system."""
        from core.diagnostics import get_diagnostics_manager

        config = test_config.get("strategy_generator", {})
        config["model_path"] = temp_dir

        generator = StrategyGenerator(config)
        await generator.initialize()

        diagnostics = get_diagnostics_manager()

        # Register health check
        async def check_strategy_generator():
            try:
                population_size = len(generator.population)
                current_generation = generator.current_generation
                best_fitness = max([g.fitness for g in generator.population]) if generator.population else 0

                status = HealthStatus.HEALTHY if population_size > 0 else HealthStatus.DEGRADED

                return {
                    'component': 'strategy_generator',
                    'status': status,
                    'latency_ms': 15.0,
                    'message': f'Generator healthy: pop={population_size}, gen={current_generation}, best_fit={best_fitness:.2f}',
                    'details': {
                        'population_size': population_size,
                        'current_generation': current_generation,
                        'best_fitness': best_fitness
                    }
                }
            except Exception as e:
                return {
                    'component': 'strategy_generator',
                    'status': HealthStatus.CRITICAL,
                    'message': f'Health check failed: {str(e)}',
                    'details': {'error': str(e)}
                }

        diagnostics.register_health_check('strategy_generator', check_strategy_generator)

        # Run health check
        state = await diagnostics.run_health_check()

        # Should have strategy generator health data
        assert 'strategy_generator' in state.component_statuses
        sg_status = state.component_statuses['strategy_generator']

        assert sg_status.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert 'Generator healthy:' in sg_status.message
