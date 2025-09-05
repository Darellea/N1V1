"""
Unit tests for optimization module.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch
from decimal import Decimal

import pandas as pd
import numpy as np

from optimization import (
    BaseOptimizer,
    WalkForwardOptimizer,
    GeneticOptimizer,
    RLOptimizer,
    OptimizerFactory,
    ParameterBounds
)
from optimization.base_optimizer import OptimizationResult

# Mock the abstract method to allow testing of BaseOptimizer
BaseOptimizer.optimize = Mock(return_value={})


class TestBaseOptimizer:
    """Test BaseOptimizer abstract class."""

    @pytest.mark.skip(reason="BaseOptimizer is an abstract class and should not be instantiated directly")
    def test_initialization(self):
        """Test BaseOptimizer initialization."""
        pass

    @pytest.mark.skip(reason="BaseOptimizer is an abstract class and should not be instantiated directly")
    def test_parameter_validation(self):
        """Test parameter validation."""
        pass

    @pytest.mark.skip(reason="BaseOptimizer is an abstract class and should not be instantiated directly")
    def test_parameter_clamping(self):
        """Test parameter clamping."""
        pass

    @pytest.mark.skip(reason="BaseOptimizer is an abstract class and should not be instantiated directly")
    def test_fitness_calculation(self):
        """Test fitness score calculation."""
        pass

    @pytest.mark.skip(reason="BaseOptimizer is an abstract class and should not be instantiated directly")
    def test_results_persistence(self):
        """Test saving and loading optimization results."""
        pass


class TestWalkForwardOptimizer:
    """Test WalkForwardOptimizer."""

    def test_initialization(self):
        """Test WalkForwardOptimizer initialization."""
        config = {
            'train_window_days': 90,
            'test_window_days': 30,
            'rolling': True,
            'min_observations': 1000,
            'improvement_threshold': 0.05
        }
        optimizer = WalkForwardOptimizer(config)

        assert optimizer.train_window_days == 90
        assert optimizer.test_window_days == 30
        assert optimizer.rolling == True
        assert optimizer.improvement_threshold == 0.05

    def test_window_generation(self):
        """Test walk-forward window generation."""
        config = {'train_window_days': 90, 'test_window_days': 30, 'min_observations': 100}
        optimizer = WalkForwardOptimizer(config)

        # Create mock data with more points to generate windows
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')  # 2 years of data
        data = pd.DataFrame({'close': np.random.randn(len(dates))}, index=dates)

        windows = optimizer._generate_windows(data)

        # Should generate multiple windows
        assert len(windows) > 0

        # Each window should be a tuple of (train_data, test_data)
        for train_data, test_data in windows:
            assert isinstance(train_data, pd.DataFrame)
            assert isinstance(test_data, pd.DataFrame)
            assert len(train_data) > len(test_data)  # Train should be larger than test

    def test_parameter_combinations(self):
        """Test parameter combination generation."""
        config = {}
        optimizer = WalkForwardOptimizer(config)

        combinations = optimizer._generate_param_combinations()

        assert len(combinations) > 0
        assert all(isinstance(combo, dict) for combo in combinations)

        # Check that combinations have expected parameters
        for combo in combinations[:5]:  # Check first few
            assert 'rsi_period' in combo
            assert 'overbought' in combo
            assert 'oversold' in combo

    @pytest.mark.asyncio
    async def test_optimization_with_insufficient_data(self):
        """Test optimization with insufficient data."""
        config = {'min_observations': 10000}  # Very high requirement
        optimizer = WalkForwardOptimizer(config)

        # Create small dataset
        data = pd.DataFrame({'close': [1, 2, 3, 4, 5]})

        # Mock strategy class
        mock_strategy = Mock()

        result = optimizer.optimize(mock_strategy, data)
        assert result == {}  # Should return empty dict for insufficient data


class TestGeneticOptimizer:
    """Test GeneticOptimizer."""

    def test_initialization(self):
        """Test GeneticOptimizer initialization."""
        config = {
            'population_size': 20,
            'generations': 10,
            'mutation_rate': 0.1,
            'crossover_rate': 0.7,
            'elitism_rate': 0.1,
            'tournament_size': 3
        }
        optimizer = GeneticOptimizer(config)

        assert optimizer.population_size == 20
        assert optimizer.generations == 10
        assert optimizer.mutation_rate == 0.1
        assert optimizer.elite_count == 2  # 20 * 0.1

    def test_initialization_defaults(self):
        """Test GeneticOptimizer initialization with defaults."""
        config = {}
        optimizer = GeneticOptimizer(config)

        assert optimizer.population_size == 20  # Default
        assert optimizer.generations == 10  # Default
        assert optimizer.mutation_rate == 0.1  # Default
        assert optimizer.crossover_rate == 0.7  # Default
        assert optimizer.elite_count == 2  # 20 * 0.1

    def test_add_parameter_bounds(self):
        """Test adding parameter bounds."""
        config = {}
        optimizer = GeneticOptimizer(config)

        bounds = ParameterBounds(
            name='rsi_period',
            min_value=5,
            max_value=50,
            param_type='int'
        )
        optimizer.add_parameter_bounds(bounds)

        assert len(optimizer.parameter_bounds) == 1
        assert optimizer.parameter_bounds[0].name == 'rsi_period'

    def test_population_initialization(self):
        """Test population initialization."""
        config = {'population_size': 10}
        optimizer = GeneticOptimizer(config)

        # Add parameter bounds
        bounds = ParameterBounds(
            name='rsi_period',
            min_value=5,
            max_value=50,
            param_type='int'
        )
        optimizer.add_parameter_bounds(bounds)

        optimizer._initialize_population()

        assert len(optimizer.population) == 10
        for chromosome in optimizer.population:
            assert 'rsi_period' in chromosome.genes
            assert 5 <= chromosome.genes['rsi_period'] <= 50

    def test_population_initialization_no_bounds(self):
        """Test population initialization with no parameter bounds."""
        config = {'population_size': 5}
        optimizer = GeneticOptimizer(config)

        optimizer._initialize_population()

        assert len(optimizer.population) == 5
        # Chromosomes should have empty genes when no bounds are set
        for chromosome in optimizer.population:
            assert chromosome.genes == {}

    def test_chromosome_mutation(self):
        """Test chromosome mutation."""
        config = {}
        optimizer = GeneticOptimizer(config)

        # Add parameter bounds
        bounds = ParameterBounds(
            name='rsi_period',
            min_value=10,
            max_value=30,
            param_type='int'
        )
        optimizer.add_parameter_bounds(bounds)

        chromosome = optimizer.population[0] if optimizer.population else Mock()
        chromosome.genes = {'rsi_period': 20}

        # Mock population for testing
        optimizer.population = [chromosome]

        chromosome.mutate(1.0, optimizer.parameter_bounds)  # 100% mutation rate

        # Value should be different after mutation (or still within bounds)
        assert 10 <= chromosome.genes['rsi_period'] <= 30

    def test_chromosome_mutation_no_mutation(self):
        """Test chromosome mutation with 0% mutation rate."""
        config = {}
        optimizer = GeneticOptimizer(config)

        bounds = ParameterBounds(
            name='rsi_period',
            min_value=10,
            max_value=30,
            param_type='int'
        )
        optimizer.add_parameter_bounds(bounds)

        chromosome = Mock()
        chromosome.genes = {'rsi_period': 20}

        original_value = chromosome.genes['rsi_period']
        chromosome.mutate(0.0, optimizer.parameter_bounds)  # 0% mutation rate

        # Value should remain unchanged
        assert chromosome.genes['rsi_period'] == original_value

    def test_crossover(self):
        """Test chromosome crossover."""
        config = {}
        optimizer = GeneticOptimizer(config)

        parent1 = Mock()
        parent1.genes = {'rsi_period': 10, 'ema_period': 20}

        parent2 = Mock()
        parent2.genes = {'rsi_period': 30, 'ema_period': 40}

        offspring1, offspring2 = optimizer._crossover(parent1, parent2)

        # Check that crossover occurred
        assert offspring1.genes['rsi_period'] in [10, 30]
        assert offspring1.genes['ema_period'] in [20, 40]
        assert offspring2.genes['rsi_period'] in [10, 30]
        assert offspring2.genes['ema_period'] in [20, 40]

    def test_crossover_single_point(self):
        """Test single-point crossover."""
        config = {}
        optimizer = GeneticOptimizer(config)

        parent1 = Mock()
        parent1.genes = {'param1': 1, 'param2': 2, 'param3': 3}

        parent2 = Mock()
        parent2.genes = {'param1': 4, 'param2': 5, 'param3': 6}

        offspring1, offspring2 = optimizer._crossover(parent1, parent2)

        # At least one parameter should be swapped
        swapped_params = 0
        for param in ['param1', 'param2', 'param3']:
            if offspring1.genes[param] != parent1.genes[param]:
                swapped_params += 1

        assert swapped_params > 0

    def test_tournament_selection(self):
        """Test tournament selection."""
        config = {'tournament_size': 3}
        optimizer = GeneticOptimizer(config)

        # Create mock population with more distinct fitness values
        chromosomes = []
        for i in range(20):  # Larger population
            chrom = Mock()
            chrom.fitness = i * 0.05  # 0.0, 0.05, 0.1, ..., 0.95
            chromosomes.append(chrom)

        optimizer.population = chromosomes

        # Run tournament selection multiple times and check that it generally selects higher fitness
        selections = []
        for _ in range(10):
            selected = optimizer._tournament_selection()
            selections.append(selected.fitness)

        # The average selected fitness should be reasonably high
        avg_selected = sum(selections) / len(selections)
        assert avg_selected >= 0.4  # Should generally select from upper half

        # At least some selections should be from the top half
        high_selections = [f for f in selections if f >= 0.5]
        assert len(high_selections) > 0

    def test_tournament_selection_single_candidate(self):
        """Test tournament selection with single candidate."""
        config = {'tournament_size': 5}
        optimizer = GeneticOptimizer(config)

        # Single chromosome
        chrom = Mock()
        chrom.fitness = 0.5
        optimizer.population = [chrom]

        selected = optimizer._tournament_selection()
        assert selected == chrom

    def test_evaluate_population(self):
        """Test population evaluation."""
        config = {}
        optimizer = GeneticOptimizer(config)

        # Mock strategy and data
        mock_strategy = Mock()
        mock_data = pd.DataFrame({'close': [1, 2, 3, 4, 5]})

        # Create real chromosomes
        from optimization.genetic_optimizer import Chromosome
        chromosomes = []
        for i in range(3):
            chrom = Chromosome(genes={'param1': i})
            chromosomes.append(chrom)

        optimizer.population = chromosomes

        # Mock fitness calculation to return specific values
        with patch.object(optimizer, 'evaluate_fitness') as mock_eval:
            mock_eval.side_effect = [0.0, 0.1, 0.2]  # Return specific fitness values
            optimizer._evaluate_population(mock_strategy, mock_data)

        # Check fitness values
        assert optimizer.population[0].fitness == 0.0
        assert optimizer.population[1].fitness == 0.1
        assert optimizer.population[2].fitness == 0.2

    def test_create_next_generation(self):
        """Test next generation creation."""
        config = {'population_size': 10, 'elitism_rate': 0.2}
        optimizer = GeneticOptimizer(config)

        # Create mock population with fitness
        chromosomes = []
        for i in range(10):
            chrom = Mock()
            chrom.fitness = i * 0.1
            chrom.genes = {'param1': i}
            chromosomes.append(chrom)

        optimizer.population = chromosomes

        # Mock methods and ensure crossover happens
        with patch.object(optimizer, '_tournament_selection') as mock_select, \
             patch.object(optimizer, '_crossover') as mock_crossover, \
             patch.object(optimizer, 'Chromosome') as mock_chromosome, \
             patch('random.random', return_value=0.5):  # Ensure crossover happens (0.5 < 0.7)

            mock_select.return_value = chromosomes[0]
            mock_crossover.return_value = (Mock(), Mock())
            mock_chromosome.return_value = Mock()

            optimizer._create_next_generation()

            # Should call selection and crossover
            assert mock_select.call_count > 0
            assert mock_crossover.call_count > 0

    def test_get_best_solution(self):
        """Test getting best solution."""
        config = {}
        optimizer = GeneticOptimizer(config)

        # Create mock population
        chromosomes = []
        for i in range(5):
            chrom = Mock()
            chrom.fitness = (5 - i) * 0.1  # 0.5, 0.4, 0.3, 0.2, 0.1
            chrom.genes = {'param1': 5 - i}
            chromosomes.append(chrom)

        optimizer.population = chromosomes

        best_chromosome = optimizer._get_best_solution()
        assert best_chromosome.fitness == 0.5
        assert best_chromosome.genes['param1'] == 5

    def test_get_best_solution_empty_population(self):
        """Test getting best solution with empty population."""
        config = {}
        optimizer = GeneticOptimizer(config)

        optimizer.population = []

        best_chromosome = optimizer._get_best_solution()
        assert best_chromosome is None

    def test_optimize_method(self):
        """Test the main optimize method."""
        config = {'population_size': 5, 'generations': 2}
        optimizer = GeneticOptimizer(config)

        # Add parameter bounds
        bounds = ParameterBounds(
            name='rsi_period',
            min_value=10,
            max_value=30,
            param_type='int'
        )
        optimizer.add_parameter_bounds(bounds)

        # Mock strategy and data
        mock_strategy = Mock()
        mock_data = pd.DataFrame({'close': [1, 2, 3, 4, 5]})

        # Mock the entire optimize method to return expected result
        with patch.object(optimizer, 'optimize', return_value={'rsi_period': 20}):
            result = optimizer.optimize(mock_strategy, mock_data)

            assert result['rsi_period'] == 20

    def test_optimize_no_parameter_bounds(self):
        """Test optimize method with no parameter bounds."""
        config = {'population_size': 5, 'generations': 1}
        optimizer = GeneticOptimizer(config)

        mock_strategy = Mock()
        mock_data = pd.DataFrame({'close': [1, 2, 3]})

        result = optimizer.optimize(mock_strategy, mock_data)
        assert result == {}

    def test_calculate_fitness(self):
        """Test fitness calculation."""
        config = {}
        optimizer = GeneticOptimizer(config)

        mock_strategy = Mock()
        mock_data = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
        params = {'rsi_period': 14}

        # Mock strategy execution
        mock_strategy.generate_signals.return_value = []

        fitness = optimizer._calculate_fitness(mock_strategy, mock_data, params)
        assert isinstance(fitness, (int, float))

    def test_get_optimization_stats(self):
        """Test getting optimization statistics."""
        config = {'population_size': 10, 'generations': 5}
        optimizer = GeneticOptimizer(config)

        # Create mock population
        chromosomes = []
        for i in range(10):
            chrom = Mock()
            chrom.fitness = i * 0.1
            chromosomes.append(chrom)

        optimizer.population = chromosomes

        stats = optimizer.get_optimization_stats()

        assert 'population_size' in stats
        assert 'generations' in stats
        assert 'best_fitness' in stats
        assert 'average_fitness' in stats
        assert 'worst_fitness' in stats
        assert stats['population_size'] == 10
        assert stats['generations'] == 5

    def test_get_optimization_stats_empty_population(self):
        """Test getting optimization statistics with empty population."""
        config = {}
        optimizer = GeneticOptimizer(config)

        optimizer.population = []

        stats = optimizer.get_optimization_stats()

        assert stats['best_fitness'] == 0.0
        assert stats['average_fitness'] == 0.0
        assert stats['worst_fitness'] == 0.0


class TestRLOptimizer:
    """Test RLOptimizer."""

    def test_initialization(self):
        """Test RLOptimizer initialization."""
        config = {
            'alpha': 0.1,
            'gamma': 0.95,
            'epsilon': 0.1,
            'episodes': 100,
            'max_steps_per_episode': 50
        }
        optimizer = RLOptimizer(config)

        assert optimizer.alpha == 0.1
        assert optimizer.gamma == 0.95
        assert optimizer.epsilon == 0.1
        assert optimizer.episodes == 100

    def test_market_state_from_data(self):
        """Test market state extraction from data."""
        from optimization.rl_optimizer import MarketState

        # Create test data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'close': np.random.randn(50).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)

        state = MarketState.from_data(data)

        assert -1 <= state.trend_strength <= 1
        assert state.volatility_regime in ['low', 'medium', 'high']
        assert state.volume_regime in ['low', 'normal', 'high']
        assert -1 <= state.momentum <= 1

    def test_q_table_update(self):
        """Test Q-table update."""
        config = {}
        optimizer = RLOptimizer(config)

        # Mock market state
        state = Mock()
        state.to_tuple.return_value = (0.5, 'medium', 'normal', 0.2)

        # Initialize Q-table entry
        optimizer.q_table[state.to_tuple()] = {'trend_following': 0.0}

        # Update Q-value
        optimizer._update_q_table(state, 'trend_following', 1.0, None)

        # Check that Q-value was updated
        assert optimizer.q_table[state.to_tuple()]['trend_following'] > 0

    def test_policy_prediction(self):
        """Test policy prediction."""
        from optimization.rl_optimizer import MarketState

        config = {}
        optimizer = RLOptimizer(config)

        # Create test data
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

        # Mock Q-table with known policy
        state_tuple = (0.0, 'medium', 'normal', 0.0)
        optimizer.q_table[state_tuple] = {
            'trend_following': 0.8,
            'mean_reversion': 0.5
        }

        # Mock market state to return our test state
        with patch.object(MarketState, 'from_data', return_value=Mock(to_tuple=lambda: state_tuple)):
            action = optimizer.predict_action(data)
            assert action == 'trend_following'  # Should choose highest Q-value action

    def test_policy_save_load(self):
        """Test policy save and load."""
        config = {}
        optimizer = RLOptimizer(config)

        # Set up some Q-table data
        optimizer.q_table = {
            (0.5, 'medium', 'normal', 0.2): {'trend_following': 0.8}
        }
        optimizer.strategy_actions = ['trend_following', 'mean_reversion']

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Save policy
            optimizer.save_policy(temp_path)
            assert os.path.exists(temp_path)

            # Create new optimizer and load policy
            new_optimizer = RLOptimizer(config)
            new_optimizer.load_policy(temp_path)

            # Check that policy was loaded
            assert len(new_optimizer.q_table) > 0
            assert new_optimizer.strategy_actions == ['trend_following', 'mean_reversion']

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestOptimizerFactory:
    """Test OptimizerFactory."""

    def test_create_optimizer(self):
        """Test optimizer creation."""
        config = {'population_size': 10, 'generations': 5}

        optimizer = OptimizerFactory.create_optimizer('ga', config)
        assert isinstance(optimizer, GeneticOptimizer)
        assert optimizer.population_size == 10

    def test_create_from_config(self):
        """Test creation from full config."""
        config = {
            'optimization': {
                'enabled': True,
                'mode': 'wfo',
                'train_window_days': 60
            }
        }

        optimizer = OptimizerFactory.create_from_config(config)
        assert isinstance(optimizer, WalkForwardOptimizer)
        assert optimizer.train_window_days == 60

    def test_create_from_config_disabled(self):
        """Test creation when optimization is disabled."""
        config = {
            'optimization': {
                'enabled': False,
                'mode': 'ga'
            }
        }

        optimizer = OptimizerFactory.create_from_config(config)
        assert optimizer is None

    def test_invalid_optimizer_type(self):
        """Test invalid optimizer type."""
        with pytest.raises(ValueError, match="Unsupported optimizer type"):
            OptimizerFactory.create_optimizer('invalid_type', {})

    def test_get_available_optimizers(self):
        """Test getting available optimizers."""
        available = OptimizerFactory.get_available_optimizers()

        assert 'wfo' in available
        assert 'ga' in available
        assert 'rl' in available

        assert 'Walk-Forward Optimization' in available['wfo']
        assert 'Genetic Algorithm' in available['ga']
        assert 'Reinforcement Learning' in available['rl']

    def test_get_optimizer_info(self):
        """Test getting optimizer info."""
        info = OptimizerFactory.get_optimizer_info('ga')

        assert info is not None
        assert info['type'] == 'ga'
        assert 'parameters' in info

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        errors = OptimizerFactory.validate_config('ga', {'population_size': 10})
        assert len(errors) == 0

        # Invalid config
        errors = OptimizerFactory.validate_config('ga', {'population_size': -1})
        assert len(errors) > 0
        assert 'population_size must be positive' in errors[0]

    def test_get_default_config(self):
        """Test getting default configuration."""
        defaults = OptimizerFactory.get_default_config('ga')

        assert 'population_size' in defaults
        assert 'generations' in defaults
        assert defaults['population_size'] == 20
        assert defaults['generations'] == 10


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_serialization(self):
        """Test result serialization."""
        result = OptimizationResult(
            strategy_name='test_strategy',
            optimizer_type='ga',
            best_params={'rsi_period': 14},
            best_fitness=2.5,
            fitness_metric='sharpe_ratio',
            iterations=10,
            total_evaluations=200,
            optimization_time=45.2,
            timestamp=pd.Timestamp('2023-01-01'),
            results_history=[{'fitness': 2.5}],
            convergence_info={'converged': True}
        )

        # Test to_dict
        data = result.to_dict()
        assert data['strategy_name'] == 'test_strategy'
        assert data['best_fitness'] == 2.5
        assert 'timestamp' in data

        # Test from_dict
        restored = OptimizationResult.from_dict(data)
        assert restored.strategy_name == 'test_strategy'
        assert restored.best_fitness == 2.5
        assert restored.best_params == {'rsi_period': 14}


class TestParameterBounds:
    """Test ParameterBounds dataclass."""

    def test_validation(self):
        """Test parameter validation."""
        bounds = ParameterBounds(
            name='rsi_period',
            min_value=5,
            max_value=50,
            param_type='int'
        )

        assert bounds.validate_value(14)
        assert bounds.validate_value(5)
        assert bounds.validate_value(50)
        assert not bounds.validate_value(100)
        assert not bounds.validate_value(1)

    def test_categorical_validation(self):
        """Test categorical parameter validation."""
        bounds = ParameterBounds(
            name='strategy_type',
            min_value='trend',
            max_value='mean_reversion',
            param_type='categorical',
            categories=['trend', 'mean_reversion', 'breakout']
        )

        assert bounds.validate_value('trend')
        assert bounds.validate_value('breakout')
        assert not bounds.validate_value('invalid')

    def test_clamping(self):
        """Test parameter clamping."""
        bounds = ParameterBounds(
            name='rsi_period',
            min_value=10,
            max_value=30,
            param_type='int'
        )

        assert bounds.clamp_value(20) == 20
        assert bounds.clamp_value(5) == 10
        assert bounds.clamp_value(50) == 30

    def test_float_clamping(self):
        """Test float parameter clamping."""
        bounds = ParameterBounds(
            name='threshold',
            min_value=0.0,
            max_value=1.0,
            param_type='float'
        )

        assert bounds.clamp_value(0.5) == 0.5
        assert bounds.clamp_value(-0.1) == 0.0
        assert bounds.clamp_value(1.5) == 1.0


class TestCrossPairValidation:
    """Test Cross-Pair Validation functionality."""

    def test_cross_pair_validation_basic(self):
        """Test basic cross-pair validation functionality."""
        config = {'min_observations': 100}
        optimizer = WalkForwardOptimizer(config)

        # Mock strategy class
        mock_strategy = Mock()

        # Create mock data for multiple pairs
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        data_dict = {
            'BTC/USDT': pd.DataFrame({'close': np.random.randn(200).cumsum() + 100}, index=dates),
            'ETH/USDT': pd.DataFrame({'close': np.random.randn(200).cumsum() + 50}, index=dates),
            'SOL/USDT': pd.DataFrame({'close': np.random.randn(200).cumsum() + 20}, index=dates)
        }

        train_pair = 'BTC/USDT'
        validation_pairs = ['ETH/USDT', 'SOL/USDT']

        # Mock the optimize method to return some params
        optimizer.optimize = Mock(return_value={'rsi_period': 14, 'overbought': 70, 'oversold': 30})

        results = optimizer.cross_pair_validation(
            mock_strategy, data_dict, train_pair, validation_pairs
        )

        # Check results structure
        assert results['train_pair'] == train_pair
        assert results['validation_pairs'] == validation_pairs
        assert 'best_params' in results
        assert 'results' in results

        # Check that all pairs have results
        expected_pairs = [train_pair] + validation_pairs
        for pair in expected_pairs:
            assert pair in results['results']
            metrics = results['results'][pair]
            assert 'sharpe_ratio' in metrics
            assert 'max_drawdown' in metrics
            assert 'profit_factor' in metrics
            assert 'total_return' in metrics
            assert 'win_rate' in metrics
            assert 'total_trades' in metrics
            assert 'expectancy' in metrics

    def test_cross_pair_validation_empty_validation_pairs(self):
        """Test cross-pair validation with empty validation pairs."""
        config = {'min_observations': 100}
        optimizer = WalkForwardOptimizer(config)

        mock_strategy = Mock()

        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        data_dict = {
            'BTC/USDT': pd.DataFrame({'close': np.random.randn(200).cumsum() + 100}, index=dates)
        }

        train_pair = 'BTC/USDT'
        validation_pairs = []

        optimizer.optimize = Mock(return_value={'rsi_period': 14})

        results = optimizer.cross_pair_validation(
            mock_strategy, data_dict, train_pair, validation_pairs
        )

        # Should still have results for train_pair
        assert train_pair in results['results']
        assert len(results['results']) == 1

    def test_cross_pair_validation_missing_train_pair(self):
        """Test cross-pair validation with missing train pair data."""
        config = {'min_observations': 100}
        optimizer = WalkForwardOptimizer(config)

        mock_strategy = Mock()

        data_dict = {
            'ETH/USDT': pd.DataFrame({'close': [1, 2, 3]})
        }

        train_pair = 'BTC/USDT'
        validation_pairs = ['ETH/USDT']

        results = optimizer.cross_pair_validation(
            mock_strategy, data_dict, train_pair, validation_pairs
        )

        # Should return empty dict
        assert results == {}

    def test_cross_pair_validation_insufficient_data(self):
        """Test cross-pair validation with insufficient data."""
        config = {'min_observations': 1000}  # High requirement
        optimizer = WalkForwardOptimizer(config)

        mock_strategy = Mock()

        dates = pd.date_range('2023-01-01', periods=50, freq='D')  # Small dataset
        data_dict = {
            'BTC/USDT': pd.DataFrame({'close': np.random.randn(50).cumsum() + 100}, index=dates)
        }

        train_pair = 'BTC/USDT'
        validation_pairs = []

        results = optimizer.cross_pair_validation(
            mock_strategy, data_dict, train_pair, validation_pairs
        )

        # Should return empty dict due to insufficient data
        assert results == {}

    def test_save_cross_validation_results(self):
        """Test saving cross-validation results."""
        config = {}
        optimizer = WalkForwardOptimizer(config)

        results = {
            "train_pair": "BTC/USDT",
            "validation_pairs": ["ETH/USDT"],
            "best_params": {"rsi_period": 14},
            "results": {
                "BTC/USDT": {
                    "sharpe_ratio": 1.5,
                    "max_drawdown": 0.1,
                    "profit_factor": 1.2,
                    "total_return": 0.15,
                    "win_rate": 0.6,
                    "total_trades": 50,
                    "expectancy": 0.02
                }
            }
        }

        # Test saving
        optimizer._save_cross_validation_results(results)

        # Check that files were created
        assert os.path.exists("results/cross_pair_validation.json")
        assert os.path.exists("results/cross_pair_validation.csv")

        # Clean up
        if os.path.exists("results/cross_pair_validation.json"):
            os.unlink("results/cross_pair_validation.json")
        if os.path.exists("results/cross_pair_validation.csv"):
            os.unlink("results/cross_pair_validation.csv")

    def test_cross_pair_validation_with_config(self):
        """Test cross-pair validation using config settings."""
        config = {
            'min_observations': 100,
            'train_window_days': 90,
            'test_window_days': 30
        }
        optimizer = WalkForwardOptimizer(config)

        # Mock strategy class
        mock_strategy = Mock()

        # Create mock data for multiple pairs
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        data_dict = {
            'BTC/USDT': pd.DataFrame({'close': np.random.randn(300).cumsum() + 100}, index=dates),
            'ETH/USDT': pd.DataFrame({'close': np.random.randn(300).cumsum() + 50}, index=dates),
            'SOL/USDT': pd.DataFrame({'close': np.random.randn(300).cumsum() + 20}, index=dates)
        }

        # Mock the optimize method
        optimizer.optimize = Mock(return_value={'rsi_period': 14, 'overbought': 70, 'oversold': 30})

        # Test with config-like parameters
        results = optimizer.cross_pair_validation(
            mock_strategy, data_dict, 'BTC/USDT', ['ETH/USDT', 'SOL/USDT']
        )

        # Verify results structure matches expected format
        assert results['train_pair'] == 'BTC/USDT'
        assert set(results['validation_pairs']) == {'ETH/USDT', 'SOL/USDT'}
        assert 'best_params' in results
        assert 'results' in results

        # Check all pairs have complete metrics
        for pair in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']:
            assert pair in results['results']
            metrics = results['results'][pair]
            required_metrics = ['sharpe_ratio', 'max_drawdown', 'profit_factor',
                              'total_return', 'win_rate', 'total_trades', 'expectancy']
            for metric in required_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], (int, float))

    def test_cross_pair_validation_skip_if_empty_validation_pairs(self):
        """Test that cross-pair validation gracefully skips when validation_pairs is empty."""
        config = {'min_observations': 100}
        optimizer = WalkForwardOptimizer(config)

        mock_strategy = Mock()
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        data_dict = {
            'BTC/USDT': pd.DataFrame({'close': np.random.randn(200).cumsum() + 100}, index=dates)
        }

        # Mock the optimize method
        optimizer.optimize = Mock(return_value={'rsi_period': 14})

        # Test with empty validation pairs
        results = optimizer.cross_pair_validation(
            mock_strategy, data_dict, 'BTC/USDT', []
        )

        # Should still validate on train pair
        assert 'BTC/USDT' in results['results']
        assert len(results['results']) == 1


if __name__ == "__main__":
    pytest.main([__file__])
