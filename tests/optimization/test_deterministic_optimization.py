"""
Deterministic optimization tests for GeneticOptimizer.
Tests reproducibility, seed isolation, and deterministic execution.
"""

import json
import time
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from optimization.base_optimizer import ParameterBounds
from optimization.genetic_optimizer import GeneticOptimizer


class TestDeterministicOptimization:
    """Test deterministic execution and reproducibility."""

    @pytest.mark.timeout(60)
    def test_reproducible_mode_produces_identical_results(self):
        """Test that reproducible mode produces identical results across runs."""
        config = {
            "population_size": 10,
            "generations": 3,
            "random_mode": "reproducible",
            "base_seed": 12345,
        }

        # Create mock strategy and data
        mock_strategy = Mock()
        data = pd.DataFrame({"close": np.random.randn(100).cumsum() + 100})

        # Mock fitness evaluation to return deterministic results
        with patch.object(GeneticOptimizer, "evaluate_fitness") as mock_eval:
            mock_eval.return_value = 0.5

            # Run optimization twice with same seed
            optimizer1 = GeneticOptimizer(config.copy())
            result1 = optimizer1.optimize(mock_strategy, data)

            optimizer2 = GeneticOptimizer(config.copy())
            result2 = optimizer2.optimize(mock_strategy, data)

            # Results should be identical
            assert result1 == result2
            assert optimizer1.best_params == optimizer2.best_params
            assert optimizer1.best_fitness == optimizer2.best_fitness

    @pytest.mark.timeout(60)
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        # Create mock strategy and data
        mock_strategy = Mock()
        data = pd.DataFrame({"close": np.random.randn(100).cumsum() + 100})

        # Mock fitness evaluation
        with patch.object(GeneticOptimizer, "evaluate_fitness") as mock_eval:
            mock_eval.return_value = 0.5

            # Run with different seeds
            config1 = {
                "population_size": 10,
                "generations": 3,
                "random_mode": "reproducible",
                "base_seed": 12345,
            }
            optimizer1 = GeneticOptimizer(config1)

            # Add parameter bounds
            bounds = ParameterBounds(
                name="test_param", min_value=0.0, max_value=1.0, param_type="float"
            )
            optimizer1.add_parameter_bounds(bounds)

            result1 = optimizer1.optimize(mock_strategy, data)

            config2 = {
                "population_size": 10,
                "generations": 3,
                "random_mode": "reproducible",
                "base_seed": 54321,
            }
            optimizer2 = GeneticOptimizer(config2)
            optimizer2.add_parameter_bounds(bounds)

            result2 = optimizer2.optimize(mock_strategy, data)

            # Results should be different (with high probability)
            # Note: In rare cases they might be the same, but that's extremely unlikely
            assert result1 != result2 or optimizer1.best_params != optimizer2.best_params

    @pytest.mark.timeout(60)
    def test_exploratory_mode_produces_varied_results(self):
        """Test that exploratory mode produces varied results."""
        # Create mock strategy and data
        mock_strategy = Mock()
        data = pd.DataFrame({"close": np.random.randn(100).cumsum() + 100})

        # Mock fitness evaluation
        with patch.object(GeneticOptimizer, "evaluate_fitness") as mock_eval:
            mock_eval.return_value = 0.5

            results = []
            for _ in range(3):
                config = {
                    "population_size": 10,
                    "generations": 3,
                    "random_mode": "exploratory",
                }
                optimizer = GeneticOptimizer(config)

                # Add parameter bounds
                bounds = ParameterBounds(
                    name="test_param", min_value=0.0, max_value=1.0, param_type="float"
                )
                optimizer.add_parameter_bounds(bounds)

                result = optimizer.optimize(mock_strategy, data)
                results.append(result)

            # At least some results should be different
            # (though theoretically they could all be the same)
            different_results = len(set(str(r) for r in results)) > 1
            assert different_results or len(results) == 1

    @pytest.mark.timeout(60)
    def test_seed_isolation_for_parallel_workers(self):
        """Test that different workers get different seeds."""
        base_config = {
            "population_size": 5,
            "generations": 2,
            "random_mode": "reproducible",
            "base_seed": 1000,
            "num_workers": 4,
        }

        # Create mock strategy and data
        mock_strategy = Mock()
        data = pd.DataFrame({"close": np.random.randn(50).cumsum() + 100})

        with patch.object(GeneticOptimizer, "evaluate_fitness") as mock_eval:
            mock_eval.return_value = 0.5

            results = []
            for worker_id in range(4):
                config = base_config.copy()
                config["worker_id"] = worker_id
                optimizer = GeneticOptimizer(config)
                result = optimizer.optimize(mock_strategy, data)
                results.append(result)

            # Check that worker seeds are different
            seeds = [opt.get_worker_seed() for opt in [
                GeneticOptimizer({**base_config, "worker_id": i}) for i in range(4)
            ]]
            assert len(set(seeds)) == 4  # All seeds should be unique

    @pytest.mark.timeout(60)
    def test_checkpoint_save_and_restore(self):
        """Test checkpointing functionality."""
        config = {
            "population_size": 8,
            "generations": 2,
            "random_mode": "reproducible",
            "base_seed": 777,
        }

        optimizer = GeneticOptimizer(config)

        # Add parameter bounds
        bounds = ParameterBounds(
            name="test_param", min_value=0.0, max_value=1.0, param_type="float"
        )
        optimizer.add_parameter_bounds(bounds)

        # Initialize population
        optimizer._initialize_population()

        # Create checkpoint
        checkpoint = optimizer.create_checkpoint()

        # Modify optimizer state
        optimizer.best_fitness = 0.9
        optimizer.best_params = {"test_param": 0.8}

        # Restore from checkpoint
        optimizer.restore_from_checkpoint(checkpoint)

        # Check that state was restored
        assert optimizer.best_fitness != 0.9  # Should be restored to original
        assert optimizer.random_mode == "reproducible"
        assert optimizer.base_seed == 777

    @pytest.mark.timeout(60)
    def test_reproducibility_validation_method(self):
        """Test the validate_reproducibility method."""
        config = {
            "population_size": 5,
            "generations": 2,
            "random_mode": "reproducible",
            "base_seed": 999,
        }

        optimizer = GeneticOptimizer(config)

        # Add parameter bounds
        bounds = ParameterBounds(
            name="test_param", min_value=0.0, max_value=1.0, param_type="float"
        )
        optimizer.add_parameter_bounds(bounds)

        # Create mock strategy and data
        mock_strategy = Mock()
        data = pd.DataFrame({"close": np.random.randn(50).cumsum() + 100})

        with patch.object(GeneticOptimizer, "evaluate_fitness") as mock_eval:
            mock_eval.return_value = 0.5

            # Run reproducibility validation
            validation_result = optimizer.validate_reproducibility(mock_strategy, data, num_runs=3)

            # Should pass validation
            assert validation_result["is_reproducible"] is True
            assert validation_result["num_runs"] == 3
            assert len(validation_result["results"]) == 3
            assert validation_result["validation_passed"] is True

    @pytest.mark.timeout(60)
    def test_random_state_info(self):
        """Test random state information retrieval."""
        config = {
            "random_mode": "reproducible",
            "base_seed": 456,
            "worker_id": 2,
            "num_workers": 8,
        }

        optimizer = GeneticOptimizer(config)
        info = optimizer.get_random_state_info()

        assert info["random_mode"] == "reproducible"
        assert info["base_seed"] == 456
        assert info["worker_id"] == 2
        assert info["num_workers"] == 8
        assert info["has_numpy_state"] is True
        assert info["has_python_state"] is not None

    @pytest.mark.timeout(60)
    def test_worker_seed_generation(self):
        """Test worker seed generation for parallel execution."""
        config = {
            "random_mode": "reproducible",
            "base_seed": 100,
            "worker_id": 3,
            "num_workers": 10,
        }

        optimizer = GeneticOptimizer(config)

        # Test base worker seed
        base_seed = optimizer.get_worker_seed()
        assert base_seed == 103  # 100 + 3

        # Test generation-specific seed
        gen_seed = optimizer.get_worker_seed(generation=5)
        assert gen_seed == 103 + 5 * 10  # base + generation * num_workers

    @pytest.mark.timeout(60)
    def test_set_random_seed_method(self):
        """Test setting random seed through method."""
        config = {"population_size": 5, "generations": 1}
        optimizer = GeneticOptimizer(config)

        # Initially reproducible (default)
        assert optimizer.random_mode == "reproducible"

        # Set different reproducible seed
        optimizer.set_random_seed(123)
        assert optimizer.random_mode == "reproducible"
        assert optimizer.base_seed == 123

        # Check that random state was reinitialized
        info = optimizer.get_random_state_info()
        assert info["random_mode"] == "reproducible"
        assert info["base_seed"] == 123

    @pytest.mark.timeout(60)
    def test_set_exploratory_mode_method(self):
        """Test setting exploratory mode through method."""
        config = {
            "random_mode": "reproducible",
            "base_seed": 123,
            "population_size": 5,
            "generations": 1,
        }
        optimizer = GeneticOptimizer(config)

        # Initially reproducible
        assert optimizer.random_mode == "reproducible"

        # Set exploratory
        optimizer.set_exploratory_mode()
        assert optimizer.random_mode == "exploratory"

    @pytest.mark.timeout(60)
    def test_deterministic_population_initialization(self):
        """Test that population initialization is deterministic with fixed seed."""
        config1 = {
            "population_size": 10,
            "random_mode": "reproducible",
            "base_seed": 42,
        }
        config2 = {
            "population_size": 10,
            "random_mode": "reproducible",
            "base_seed": 42,
        }

        # Add same parameter bounds to both
        bounds = ParameterBounds(
            name="param1", min_value=0.0, max_value=1.0, param_type="float"
        )

        optimizer1 = GeneticOptimizer(config1)
        optimizer1.add_parameter_bounds(bounds)
        optimizer1._initialize_population()

        optimizer2 = GeneticOptimizer(config2)
        optimizer2.add_parameter_bounds(bounds)
        optimizer2._initialize_population()

        # Populations should be identical
        assert len(optimizer1.population) == len(optimizer2.population)
        for chrom1, chrom2 in zip(optimizer1.population, optimizer2.population):
            assert chrom1.genes == chrom2.genes

    @pytest.mark.timeout(30)
    def test_optimization_timeout_safety(self):
        """Test that optimization completes within timeout or fails gracefully."""
        config = {
            "population_size": 50,
            "generations": 50,  # Reasonable size for timeout test
            "random_mode": "reproducible",
            "base_seed": 1,
            "test_mode": True,  # Enable test optimizations
        }

        optimizer = GeneticOptimizer(config)

        # Add parameter bounds
        bounds = ParameterBounds(
            name="param1", min_value=0.0, max_value=1.0, param_type="float"
        )
        optimizer.add_parameter_bounds(bounds)

        # Create mock strategy and data
        mock_strategy = Mock()
        data = pd.DataFrame({"close": np.random.randn(100).cumsum() + 100})

        # Mock fitness evaluation with small delay
        def delayed_fitness(*args, **kwargs):
            time.sleep(0.005)  # Small delay per evaluation
            return 0.5

        with patch.object(GeneticOptimizer, "evaluate_fitness", side_effect=delayed_fitness):
            start_time = time.time()
            result = optimizer.optimize(mock_strategy, data)
            elapsed = time.time() - start_time
            # Should complete within reasonable time
            assert elapsed < 30  # 30 seconds max for this test
            assert result is not None

    @pytest.mark.timeout(60)
    def test_deterministic_benchmark_test(self):
        """Benchmark test to ensure deterministic performance."""
        config = {
            "population_size": 20,
            "generations": 5,
            "random_mode": "reproducible",
            "base_seed": 12345,
        }

        optimizer = GeneticOptimizer(config)

        # Add parameter bounds
        bounds = ParameterBounds(
            name="param1", min_value=0.0, max_value=1.0, param_type="float"
        )
        optimizer.add_parameter_bounds(bounds)

        # Create mock strategy and data
        mock_strategy = Mock()
        data = pd.DataFrame({"close": np.random.randn(200).cumsum() + 100})

        # Mock fitness evaluation
        fitness_calls = []
        def track_fitness(*args, **kwargs):
            fitness_calls.append(time.time())
            return np.random.random()  # Random fitness for benchmarking

        with patch.object(GeneticOptimizer, "evaluate_fitness", side_effect=track_fitness):
            start_time = time.time()
            result = optimizer.optimize(mock_strategy, data)
            end_time = time.time()

            # Should complete in reasonable time
            assert end_time - start_time < 30  # 30 seconds max

            # Should have made fitness evaluations
            assert len(fitness_calls) > 0

            # Result should be valid
            assert isinstance(result, dict)
            assert "param1" in result

    @pytest.mark.timeout(60)
    def test_seed_isolation_comprehensive(self):
        """Comprehensive test of seed isolation across multiple workers."""
        num_workers = 5
        base_seed = 1000

        # Create mock strategy and data
        mock_strategy = Mock()
        data = pd.DataFrame({"close": np.random.randn(50).cumsum() + 100})

        with patch.object(GeneticOptimizer, "evaluate_fitness") as mock_eval:
            mock_eval.return_value = 0.5

            optimizers = []
            results = []

            # Create multiple workers
            for worker_id in range(num_workers):
                config = {
                    "population_size": 5,
                    "generations": 2,
                    "random_mode": "reproducible",
                    "base_seed": base_seed,
                    "worker_id": worker_id,
                    "num_workers": num_workers,
                }
                optimizer = GeneticOptimizer(config)
                optimizers.append(optimizer)

                # Add parameter bounds
                bounds = ParameterBounds(
                    name="param1", min_value=0.0, max_value=1.0, param_type="float"
                )
                optimizer.add_parameter_bounds(bounds)

                result = optimizer.optimize(mock_strategy, data)
                results.append(result)

            # Check that all workers have different seeds
            seeds = [opt.get_worker_seed() for opt in optimizers]
            assert len(set(seeds)) == num_workers

            # Check that seeds are properly offset
            expected_seeds = [base_seed + i for i in range(num_workers)]
            assert set(seeds) == set(expected_seeds)

    @pytest.mark.timeout(60)
    def test_checkpoint_restoration_preserves_state(self):
        """Test that checkpoint restoration preserves optimization state."""
        config = {
            "population_size": 6,
            "generations": 3,
            "random_mode": "reproducible",
            "base_seed": 555,
        }

        optimizer = GeneticOptimizer(config)

        # Add parameter bounds
        bounds = ParameterBounds(
            name="param1", min_value=0.0, max_value=1.0, param_type="float"
        )
        optimizer.add_parameter_bounds(bounds)

        # Create mock strategy and data
        mock_strategy = Mock()
        data = pd.DataFrame({"close": np.random.randn(50).cumsum() + 100})

        with patch.object(GeneticOptimizer, "evaluate_fitness") as mock_eval:
            mock_eval.return_value = 0.5

            # Run partial optimization
            optimizer._initialize_population()
            optimizer._evaluate_population(mock_strategy, data)
            optimizer._update_best_chromosome()

            # Create checkpoint
            checkpoint = optimizer.create_checkpoint()

            # Modify state
            original_best = optimizer.best_fitness
            optimizer.best_fitness = 999.0

            # Restore from checkpoint
            optimizer.restore_from_checkpoint(checkpoint)

            # State should be restored
            assert optimizer.best_fitness == original_best
            assert optimizer.random_mode == "reproducible"
            assert optimizer.base_seed == 555

    @pytest.mark.timeout(60)
    def test_reproducibility_with_complex_parameters(self):
        """Test reproducibility with complex parameter sets."""
        config = {
            "population_size": 8,
            "generations": 2,
            "random_mode": "reproducible",
            "base_seed": 7777,
        }

        optimizer = GeneticOptimizer(config)

        # Add multiple parameter bounds
        bounds = [
            ParameterBounds(name="float_param", min_value=0.0, max_value=1.0, param_type="float"),
            ParameterBounds(name="int_param", min_value=1, max_value=100, param_type="int"),
            ParameterBounds(name="cat_param", min_value=0, max_value=2, param_type="categorical",
                          categories=["option_a", "option_b", "option_c"]),
        ]

        for bound in bounds:
            optimizer.add_parameter_bounds(bound)

        # Create mock strategy and data
        mock_strategy = Mock()
        data = pd.DataFrame({"close": np.random.randn(50).cumsum() + 100})

        with patch.object(GeneticOptimizer, "evaluate_fitness") as mock_eval:
            mock_eval.return_value = 0.5

            # Run multiple times and check reproducibility
            results = []
            for _ in range(3):
                opt = GeneticOptimizer(config.copy())
                for bound in bounds:
                    opt.add_parameter_bounds(bound)
                result = opt.optimize(mock_strategy, data)
                results.append(result)

            # All results should be identical
            for result in results[1:]:
                assert optimizer._params_equal(results[0], result)


if __name__ == "__main__":
    pytest.main([__file__])
