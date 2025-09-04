"""
optimization/__init__.py

Self-Optimization Layer for Trading Strategies.

This module provides advanced optimization techniques to automatically
adapt trading strategies over time without manual retuning.
"""

from .base_optimizer import BaseOptimizer, ParameterBounds, OptimizationResult
from .walk_forward import WalkForwardOptimizer
from .genetic_optimizer import GeneticOptimizer
from .rl_optimizer import RLOptimizer
from .optimizer_factory import OptimizerFactory

__all__ = [
    'BaseOptimizer',
    'ParameterBounds',
    'OptimizationResult',
    'WalkForwardOptimizer',
    'GeneticOptimizer',
    'RLOptimizer',
    'OptimizerFactory'
]
