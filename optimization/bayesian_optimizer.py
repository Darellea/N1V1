"""
Bayesian Optimizer Module

This module provides Bayesian optimization capabilities for efficient exploration
of the strategy parameter space. It uses Gaussian Processes to model the fitness
landscape and acquisition functions to suggest optimal parameter combinations.

Key Features:
- Gaussian Process regression for fitness landscape modeling
- Multiple acquisition functions (Expected Improvement, Upper Confidence Bound)
- Intelligent exploration-exploitation trade-off
- Integration with distributed evaluation
- Automatic hyperparameter tuning
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import get_bayesian_optimization_config
from .genome import StrategyGenome

# Check for scikit-learn availability
try:
    import sklearn.gaussian_process as gp
    from sklearn.exceptions import NotFittedError
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available, Bayesian optimization disabled")

logger = logging.getLogger(__name__)


@dataclass
class BayesianOptimizationResult:
    """Result of Bayesian optimization iteration."""

    suggested_genome: StrategyGenome
    expected_improvement: float
    uncertainty: float
    acquisition_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BayesianOptimizer:
    """
    Bayesian optimization for efficient parameter space exploration.

    This class implements Bayesian optimization using Gaussian Processes to model
    the relationship between strategy parameters and fitness. It uses acquisition
    functions to balance exploration and exploitation in the search for optimal
    strategies.

    Key capabilities:
    - Gaussian Process regression for fitness modeling
    - Multiple acquisition functions (EI, UCB, PI)
    - Automatic feature engineering from genomes
    - Hyperparameter optimization
    - Integration with distributed evaluation
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Bayesian optimizer.

        Args:
            config: Configuration dictionary for Bayesian optimization.
                   If None, uses default configuration from config module.
        """
        if not SKLEARN_AVAILABLE:
            logger.warning(
                "Scikit-learn not available, Bayesian optimization will be disabled"
            )
            self.enabled = False
            return

        if config is None:
            bayes_config = get_bayesian_optimization_config()
            config = {
                "enabled": bayes_config.enabled,
                "min_observations_for_training": bayes_config.min_observations_for_training,
                "acquisition_function": bayes_config.acquisition_function,
                "gp_kernel": bayes_config.gp_kernel,
                "gp_alpha": bayes_config.gp_alpha,
                "gp_normalize_y": bayes_config.gp_normalize_y,
                "ei_kappa": bayes_config.ei_kappa,
                "ei_xi": bayes_config.ei_xi,
            }

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Core configuration
        self.enabled = config.get("enabled", True) and SKLEARN_AVAILABLE
        self.min_observations_for_training = config.get(
            "min_observations_for_training", 5
        )
        self.acquisition_function = config.get(
            "acquisition_function", "expected_improvement"
        )

        # Gaussian Process configuration
        self.gp_kernel_type = config.get("gp_kernel", "rbf")
        self.gp_alpha = config.get("gp_alpha", 1e-6)
        self.gp_normalize_y = config.get("gp_normalize_y", True)

        # Acquisition function parameters
        self.ei_kappa = config.get("ei_kappa", 1.96)  # For UCB
        self.ei_xi = config.get("ei_xi", 0.01)  # For EI

        # Internal state
        self.gp_model: Optional[GaussianProcessRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.observations: List[Dict[str, Any]] = []
        self.is_trained = False

        # Performance tracking
        self.total_suggestions = 0
        self.successful_suggestions = 0

        if self.enabled:
            self._initialize_gp_model()
            self.logger.info(
                "BayesianOptimizer initialized with Gaussian Process model"
            )
        else:
            self.logger.info(
                "BayesianOptimizer initialized but disabled (scikit-learn not available)"
            )

    def _initialize_gp_model(self) -> None:
        """Initialize the Gaussian Process model."""
        if not self.enabled:
            return

        # Select kernel based on configuration
        if self.gp_kernel_type == "rbf":
            kernel = gp.kernels.ConstantKernel(1.0) * gp.kernels.RBF(1.0)
        elif self.gp_kernel_type == "matern":
            kernel = gp.kernels.ConstantKernel(1.0) * gp.kernels.Matern(nu=1.5)
        else:
            kernel = gp.kernels.ConstantKernel(1.0) * gp.kernels.RBF(1.0)
            self.logger.warning(
                f"Unknown kernel type '{self.gp_kernel_type}', using RBF"
            )

        # Initialize Gaussian Process
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.gp_alpha,
            normalize_y=self.gp_normalize_y,
            n_restarts_optimizer=10,
            random_state=42,
        )

        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()

    def add_observation(self, genome: StrategyGenome, fitness: float) -> None:
        """
        Add an observation to the training data.

        Args:
            genome: The genome that was evaluated
            fitness: The fitness value obtained
        """
        if not self.enabled:
            return

        # Convert genome to feature vector
        features = self._genome_to_features(genome)

        # Store observation
        observation = {
            "genome": genome,
            "features": features,
            "fitness": fitness,
            "timestamp": pd.Timestamp.now(),
        }

        self.observations.append(observation)

        # Mark as not trained since we have new data
        self.is_trained = False

        self.logger.debug(
            f"Added observation: fitness={fitness:.4f}, features_shape={features.shape}"
        )

    def suggest_next_genome(
        self, population: List[StrategyGenome]
    ) -> Optional[BayesianOptimizationResult]:
        """
        Suggest the next genome to evaluate using Bayesian optimization.

        Args:
            population: Current population of genomes

        Returns:
            BayesianOptimizationResult with suggested genome and metadata
        """
        if (
            not self.enabled
            or len(self.observations) < self.min_observations_for_training
        ):
            return None

        try:
            # Train model if needed
            if not self.is_trained:
                self._train_model()

            if not self.is_trained:
                return None

            # Generate candidate genomes
            candidates = self._generate_candidates(population)

            if not candidates:
                return None

            # Evaluate acquisition function for each candidate
            best_candidate = None
            best_acquisition_value = float("-inf")
            best_metadata = {}

            for candidate in candidates:
                acquisition_value, metadata = self._evaluate_acquisition_function(
                    candidate
                )

                if acquisition_value > best_acquisition_value:
                    best_acquisition_value = acquisition_value
                    best_candidate = candidate
                    best_metadata = metadata

            if best_candidate:
                self.total_suggestions += 1

                result = BayesianOptimizationResult(
                    suggested_genome=best_candidate,
                    expected_improvement=best_metadata.get("expected_improvement", 0),
                    uncertainty=best_metadata.get("uncertainty", 0),
                    acquisition_value=best_acquisition_value,
                    metadata=best_metadata,
                )

                self.logger.info(
                    f"Bayesian optimization suggested genome with acquisition value: {best_acquisition_value:.4f}"
                )
                return result

        except Exception as e:
            self.logger.error(f"Bayesian optimization failed: {str(e)}")

        return None

    def _train_model(self) -> None:
        """Train the Gaussian Process model with current observations."""
        if not self.enabled or len(self.observations) < 2:
            return

        try:
            # Extract features and fitness values
            X = np.array([obs["features"] for obs in self.observations])
            y = np.array([obs["fitness"] for obs in self.observations])

            # Scale features
            X_scaled = self.scaler.fit_transform(X) if self.scaler else X

            # Train GP model
            self.gp_model.fit(X_scaled, y)
            self.is_trained = True

            self.logger.info(
                f"Trained GP model with {len(self.observations)} observations"
            )

        except Exception as e:
            self.logger.error(f"Failed to train GP model: {str(e)}")
            self.is_trained = False

    def _generate_candidates(
        self, population: List[StrategyGenome]
    ) -> List[StrategyGenome]:
        """Generate candidate genomes for evaluation."""
        candidates = []

        # Generate candidates through mutation and crossover
        for _ in range(10):  # Generate 10 candidates
            if len(population) >= 2:
                # Select two random parents
                parent1, parent2 = np.random.choice(population, 2, replace=False)

                # Create offspring through crossover
                child1, child2 = parent1.crossover(parent2)

                # Apply mutation
                child1.mutate(0.2)
                child2.mutate(0.2)

                candidates.extend([child1, child2])
            else:
                # If small population, just mutate existing genomes
                parent = np.random.choice(population)
                child = parent.copy()
                child.mutate(0.3)
                candidates.append(child)

        return candidates

    def _evaluate_acquisition_function(
        self, genome: StrategyGenome
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the acquisition function for a genome.

        Args:
            genome: Genome to evaluate

        Returns:
            Tuple of (acquisition_value, metadata)
        """
        if not self.is_trained or self.gp_model is None:
            return 0.0, {}

        try:
            # Convert genome to features
            features = self._genome_to_features(genome)
            features_scaled = (
                self.scaler.transform([features]) if self.scaler else [features]
            )

            # Predict mean and uncertainty
            y_pred, y_std = self.gp_model.predict(features_scaled, return_std=True)

            # Current best fitness
            best_fitness = max(obs["fitness"] for obs in self.observations)

            # Calculate acquisition function
            if self.acquisition_function == "expected_improvement":
                acquisition_value, metadata = self._expected_improvement(
                    y_pred[0], y_std[0], best_fitness
                )
            elif self.acquisition_function == "upper_confidence_bound":
                acquisition_value, metadata = self._upper_confidence_bound(
                    y_pred[0], y_std[0]
                )
            elif self.acquisition_function == "probability_of_improvement":
                acquisition_value, metadata = self._probability_of_improvement(
                    y_pred[0], y_std[0], best_fitness
                )
            else:
                # Default to expected improvement
                acquisition_value, metadata = self._expected_improvement(
                    y_pred[0], y_std[0], best_fitness
                )

            return acquisition_value, metadata

        except Exception as e:
            self.logger.error(f"Acquisition function evaluation failed: {str(e)}")
            return 0.0, {}

    def _expected_improvement(
        self, mean: float, std: float, best_fitness: float
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate Expected Improvement acquisition function."""
        improvement = mean - best_fitness

        if std > 0:
            z = improvement / std
            ei = (
                improvement * self._normal_cdf(z)
                + std * self._normal_pdf(z) * self.ei_xi
            )
            ei = max(0, ei)
        else:
            ei = max(0, improvement)

        metadata = {
            "expected_improvement": ei,
            "uncertainty": std,
            "improvement": improvement,
            "acquisition_function": "expected_improvement",
        }

        return ei, metadata

    def _upper_confidence_bound(
        self, mean: float, std: float
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate Upper Confidence Bound acquisition function."""
        ucb = mean + self.ei_kappa * std

        metadata = {
            "upper_confidence_bound": ucb,
            "uncertainty": std,
            "mean": mean,
            "kappa": self.ei_kappa,
            "acquisition_function": "upper_confidence_bound",
        }

        return ucb, metadata

    def _probability_of_improvement(
        self, mean: float, std: float, best_fitness: float
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate Probability of Improvement acquisition function."""
        improvement = mean - best_fitness

        if std > 0:
            z = improvement / std
            pi = self._normal_cdf(z)
        else:
            pi = 1.0 if improvement > 0 else 0.0

        metadata = {
            "probability_of_improvement": pi,
            "uncertainty": std,
            "improvement": improvement,
            "acquisition_function": "probability_of_improvement",
        }

        return pi, metadata

    def _normal_cdf(self, x: float) -> float:
        """Cumulative distribution function of standard normal."""
        return (1.0 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi))) / 2

    def _normal_pdf(self, x: float) -> float:
        """Probability density function of standard normal."""
        return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)

    def _genome_to_features(self, genome: StrategyGenome) -> np.ndarray:
        """
        Convert a genome to a feature vector for GP regression.

        Args:
            genome: Genome to convert

        Returns:
            Feature vector as numpy array
        """
        features = []

        # Count of each component type
        component_counts = {}
        for gene in genome.genes:
            if gene.enabled:
                component_type = gene.component_type.value
                component_counts[component_type] = (
                    component_counts.get(component_type, 0) + 1
                )

        # Add component type counts as features
        from .genome import StrategyComponent

        for component_type in StrategyComponent:
            features.append(component_counts.get(component_type.value, 0))

        # Average gene parameters
        if genome.genes:
            total_weight = sum(g.weight for g in genome.genes)
            avg_weight = total_weight / len(genome.genes)
            features.append(avg_weight)

            enabled_ratio = sum(1 for g in genome.genes if g.enabled) / len(
                genome.genes
            )
            features.append(enabled_ratio)
        else:
            features.extend([1.0, 1.0])

        # Genome metadata
        features.append(genome.fitness if genome.fitness != float("-inf") else -1000)
        features.append(genome.age)
        features.append(genome.generation)

        # Gene-level features (simplified)
        total_params = sum(len(g.parameters) for g in genome.genes if g.enabled)
        features.append(total_params)

        return np.array(features)

    def get_model_status(self) -> Dict[str, Any]:
        """Get the current status of the Bayesian optimization model."""
        if not self.enabled:
            return {"enabled": False, "reason": "Scikit-learn not available"}

        return {
            "enabled": self.enabled,
            "is_trained": self.is_trained,
            "num_observations": len(self.observations),
            "min_observations_required": self.min_observations_for_training,
            "acquisition_function": self.acquisition_function,
            "gp_kernel": self.gp_kernel_type,
            "total_suggestions": self.total_suggestions,
            "successful_suggestions": self.successful_suggestions,
        }

    def clear_observations(self) -> None:
        """Clear all observations and reset the model."""
        self.observations.clear()
        self.is_trained = False
        self.total_suggestions = 0
        self.successful_suggestions = 0
        self.logger.info("Bayesian optimization observations cleared")

    def get_observations_summary(self) -> Dict[str, Any]:
        """Get summary statistics of observations."""
        if not self.observations:
            return {"num_observations": 0}

        fitness_values = [obs["fitness"] for obs in self.observations]

        return {
            "num_observations": len(self.observations),
            "best_fitness": max(fitness_values),
            "worst_fitness": min(fitness_values),
            "avg_fitness": np.mean(fitness_values),
            "std_fitness": np.std(fitness_values),
            "fitness_range": max(fitness_values) - min(fitness_values),
        }

    def __str__(self) -> str:
        """String representation of the Bayesian optimizer."""
        status = self.get_model_status()
        if not status["enabled"]:
            return "BayesianOptimizer(disabled)"

        return (
            f"BayesianOptimizer(observations={status['num_observations']}, "
            f"trained={status['is_trained']}, "
            f"acquisition={status['acquisition_function']})"
        )


# Convenience functions
def create_bayesian_optimizer(
    config: Optional[Dict[str, Any]] = None
) -> BayesianOptimizer:
    """
    Create a Bayesian optimizer instance.

    Args:
        config: Optional configuration overrides

    Returns:
        BayesianOptimizer instance
    """
    return BayesianOptimizer(config)


def get_bayesian_optimizer(
    config: Optional[Dict[str, Any]] = None
) -> BayesianOptimizer:
    """
    Get a Bayesian optimizer instance (alias for create_bayesian_optimizer).

    Args:
        config: Configuration dictionary

    Returns:
        BayesianOptimizer instance
    """
    return create_bayesian_optimizer(config)
