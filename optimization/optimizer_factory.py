"""
optimization/optimizer_factory.py

Factory class for creating optimizer instances based on configuration.
"""

from typing import Dict, Any, Optional, List
import logging

from .base_optimizer import BaseOptimizer
from .walk_forward import WalkForwardOptimizer
from .genetic_optimizer import GeneticOptimizer
from .rl_optimizer import RLOptimizer


class OptimizerFactory:
    """
    Factory class for creating optimizer instances.

    Provides a centralized way to create different types of optimizers
    based on configuration parameters.
    """

    # Registry of available optimizers
    OPTIMIZER_REGISTRY = {
        'wfo': WalkForwardOptimizer,
        'walk_forward': WalkForwardOptimizer,
        'ga': GeneticOptimizer,
        'genetic': GeneticOptimizer,
        'rl': RLOptimizer,
        'reinforcement_learning': RLOptimizer
    }

    @classmethod
    def create_optimizer(cls, optimizer_type: str, config: Dict[str, Any]) -> BaseOptimizer:
        """
        Create an optimizer instance based on type and configuration.

        Args:
            optimizer_type: Type of optimizer to create ('wfo', 'ga', 'rl', etc.)
            config: Configuration dictionary for the optimizer

        Returns:
            Configured optimizer instance

        Raises:
            ValueError: If optimizer type is not supported
        """
        optimizer_type = optimizer_type.lower()

        if optimizer_type not in cls.OPTIMIZER_REGISTRY:
            available_types = list(cls.OPTIMIZER_REGISTRY.keys())
            raise ValueError(
                f"Unsupported optimizer type: {optimizer_type}. "
                f"Available types: {available_types}"
            )

        optimizer_class = cls.OPTIMIZER_REGISTRY[optimizer_type]

        # Create and return optimizer instance
        optimizer = optimizer_class(config)

        logger = logging.getLogger(__name__)
        logger.info(f"Created {optimizer_type} optimizer: {optimizer_class.__name__}")

        return optimizer

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> Optional[BaseOptimizer]:
        """
        Create optimizer from full configuration dictionary.

        Args:
            config: Full configuration dictionary containing optimization settings

        Returns:
            Optimizer instance if optimization is enabled, None otherwise
        """
        optimization_config = config.get('optimization', {})

        if not optimization_config.get('enabled', False):
            return None

        optimizer_type = optimization_config.get('mode', 'wfo')

        # Remove 'enabled' and 'mode' from optimizer config
        optimizer_config = {k: v for k, v in optimization_config.items()
                          if k not in ['enabled', 'mode']}

        # Add strategy name if available
        if 'strategy_name' in config:
            optimizer_config['strategy_name'] = config['strategy_name']

        return cls.create_optimizer(optimizer_type, optimizer_config)

    @classmethod
    def get_available_optimizers(cls) -> Dict[str, str]:
        """
        Get dictionary of available optimizer types and their descriptions.

        Returns:
            Dictionary mapping optimizer types to descriptions
        """
        return {
            'wfo': 'Walk-Forward Optimization - Rolling window validation',
            'walk_forward': 'Walk-Forward Optimization - Rolling window validation',
            'ga': 'Genetic Algorithm - Evolutionary parameter optimization',
            'genetic': 'Genetic Algorithm - Evolutionary parameter optimization',
            'rl': 'Reinforcement Learning - Strategy selection optimization',
            'reinforcement_learning': 'Reinforcement Learning - Strategy selection optimization'
        }

    @classmethod
    def get_optimizer_info(cls, optimizer_type: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific optimizer type.

        Args:
            optimizer_type: Type of optimizer to get info for

        Returns:
            Dictionary with optimizer information, or None if not found
        """
        optimizer_type = optimizer_type.lower()

        if optimizer_type not in cls.OPTIMIZER_REGISTRY:
            return None

        optimizer_class = cls.OPTIMIZER_REGISTRY[optimizer_type]

        # Get basic information
        info = {
            'type': optimizer_type,
            'class_name': optimizer_class.__name__,
            'description': cls.get_available_optimizers().get(optimizer_type, ''),
            'parameters': {}
        }

        # Get default parameters from class
        if hasattr(optimizer_class, '__init__'):
            import inspect
            sig = inspect.signature(optimizer_class.__init__)
            for param_name, param in sig.parameters.items():
                if param_name != 'self' and param_name != 'config':
                    if param.default != inspect.Parameter.empty:
                        info['parameters'][param_name] = {
                            'default': param.default,
                            'description': cls._get_parameter_description(param_name)
                        }

        return info

    @classmethod
    def _get_parameter_description(cls, param_name: str) -> str:
        """
        Get description for a parameter name.

        Args:
            param_name: Parameter name

        Returns:
            Parameter description
        """
        descriptions = {
            # Walk-Forward parameters
            'train_window_days': 'Size of training window in days',
            'test_window_days': 'Size of testing window in days',
            'rolling': 'Whether to use rolling windows (True) or non-overlapping (False)',
            'min_observations': 'Minimum number of observations required for optimization',
            'improvement_threshold': 'Minimum improvement percentage to update parameters',

            # Genetic Algorithm parameters
            'population_size': 'Number of chromosomes in the population',
            'generations': 'Number of generations to evolve',
            'mutation_rate': 'Probability of gene mutation (0.0 to 1.0)',
            'crossover_rate': 'Probability of crossover between parents (0.0 to 1.0)',
            'elitism_rate': 'Fraction of best individuals to preserve (0.0 to 1.0)',
            'tournament_size': 'Size of tournament for parent selection',

            # Reinforcement Learning parameters
            'alpha': 'Learning rate for Q-learning (0.0 to 1.0)',
            'gamma': 'Discount factor for future rewards (0.0 to 1.0)',
            'epsilon': 'Exploration rate for epsilon-greedy policy (0.0 to 1.0)',
            'episodes': 'Number of training episodes',
            'max_steps_per_episode': 'Maximum steps per training episode',
            'reward_function': 'Function used to calculate rewards',

            # Common parameters
            'fitness_metric': 'Primary metric for fitness evaluation',
            'fitness_weights': 'Weights for different fitness components'
        }

        return descriptions.get(param_name, f'Parameter: {param_name}')

    @classmethod
    def validate_config(cls, optimizer_type: str, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration for a specific optimizer type.

        Args:
            optimizer_type: Type of optimizer
            config: Configuration dictionary to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        optimizer_type = optimizer_type.lower()

        if optimizer_type not in cls.OPTIMIZER_REGISTRY:
            errors.append(f"Unsupported optimizer type: {optimizer_type}")
            return errors

        # Merge provided config with defaults to ensure all required parameters are present
        defaults = cls.get_default_config(optimizer_type)
        merged_config = {**defaults, **config}

        # Type-specific validation
        if optimizer_type in ['wfo', 'walk_forward']:
            cls._validate_walk_forward_config(merged_config, errors)
        elif optimizer_type in ['ga', 'genetic']:
            cls._validate_genetic_config(merged_config, errors)
        elif optimizer_type in ['rl', 'reinforcement_learning']:
            cls._validate_rl_config(merged_config, errors)

        return errors

    @classmethod
    def _validate_walk_forward_config(cls, config: Dict[str, Any], errors: List[str]) -> None:
        """Validate Walk-Forward optimizer configuration."""
        if config.get('train_window_days', 0) <= 0:
            errors.append("train_window_days must be positive")
        if config.get('test_window_days', 0) <= 0:
            errors.append("test_window_days must be positive")
        if config.get('min_observations', 0) <= 0:
            errors.append("min_observations must be positive")
        if not (0 < config.get('improvement_threshold', 0) <= 1):
            errors.append("improvement_threshold must be between 0 and 1")

    @classmethod
    def _validate_genetic_config(cls, config: Dict[str, Any], errors: List[str]) -> None:
        """Validate Genetic Algorithm optimizer configuration."""
        if config.get('population_size', 0) <= 0:
            errors.append("population_size must be positive")
        if config.get('generations', 0) <= 0:
            errors.append("generations must be positive")
        if not (0 <= config.get('mutation_rate', -1) <= 1):
            errors.append("mutation_rate must be between 0 and 1")
        if not (0 <= config.get('crossover_rate', -1) <= 1):
            errors.append("crossover_rate must be between 0 and 1")
        if not (0 <= config.get('elitism_rate', -1) <= 1):
            errors.append("elitism_rate must be between 0 and 1")
        if config.get('tournament_size', 0) <= 0:
            errors.append("tournament_size must be positive")

    @classmethod
    def _validate_rl_config(cls, config: Dict[str, Any], errors: List[str]) -> None:
        """Validate Reinforcement Learning optimizer configuration."""
        if not (0 < config.get('alpha', 0) <= 1):
            errors.append("alpha must be between 0 and 1")
        if not (0 <= config.get('gamma', -1) <= 1):
            errors.append("gamma must be between 0 and 1")
        if not (0 <= config.get('epsilon', -1) <= 1):
            errors.append("epsilon must be between 0 and 1")
        if config.get('episodes', 0) <= 0:
            errors.append("episodes must be positive")
        if config.get('max_steps_per_episode', 0) <= 0:
            errors.append("max_steps_per_episode must be positive")

    @classmethod
    def get_default_config(cls, optimizer_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific optimizer type.

        Args:
            optimizer_type: Type of optimizer

        Returns:
            Default configuration dictionary
        """
        optimizer_type = optimizer_type.lower()

        defaults = {
            'wfo': {
                'train_window_days': 90,
                'test_window_days': 30,
                'rolling': True,
                'min_observations': 1000,
                'improvement_threshold': 0.05,
                'fitness_metric': 'sharpe_ratio'
            },
            'walk_forward': {
                'train_window_days': 90,
                'test_window_days': 30,
                'rolling': True,
                'min_observations': 1000,
                'improvement_threshold': 0.05,
                'fitness_metric': 'sharpe_ratio'
            },
            'ga': {
                'population_size': 20,
                'generations': 10,
                'mutation_rate': 0.1,
                'crossover_rate': 0.7,
                'elitism_rate': 0.1,
                'tournament_size': 3,
                'fitness_metric': 'sharpe_ratio'
            },
            'genetic': {
                'population_size': 20,
                'generations': 10,
                'mutation_rate': 0.1,
                'crossover_rate': 0.7,
                'elitism_rate': 0.1,
                'tournament_size': 3,
                'fitness_metric': 'sharpe_ratio'
            },
            'rl': {
                'alpha': 0.1,
                'gamma': 0.95,
                'epsilon': 0.1,
                'episodes': 100,
                'max_steps_per_episode': 50,
                'reward_function': 'sharpe_ratio',
                'fitness_metric': 'total_return'
            },
            'reinforcement_learning': {
                'alpha': 0.1,
                'gamma': 0.95,
                'epsilon': 0.1,
                'episodes': 100,
                'max_steps_per_episode': 50,
                'reward_function': 'sharpe_ratio',
                'fitness_metric': 'total_return'
            }
        }

        return defaults.get(optimizer_type, {})
