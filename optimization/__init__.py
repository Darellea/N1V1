"""
optimization/__init__.py

Self-Optimization Layer for Trading Strategies.

This module provides advanced optimization techniques to automatically
adapt trading strategies over time without manual retuning.
"""

from .base_optimizer import BaseOptimizer, OptimizationResult, ParameterBounds
from .cross_asset_validation import (
    AssetSelector,
    AssetValidationResult,
    CrossAssetValidationResult,
    CrossAssetValidator,
    ValidationAsset,
    ValidationCriteria,
    create_cross_asset_validator,
    run_cross_asset_validation,
)
from .genetic_optimizer import GeneticOptimizer
from .optimizer_factory import OptimizerFactory
from .rl_optimizer import RLOptimizer
from .walk_forward import WalkForwardOptimizer

__all__ = [
    "BaseOptimizer",
    "ParameterBounds",
    "OptimizationResult",
    "WalkForwardOptimizer",
    "GeneticOptimizer",
    "RLOptimizer",
    "OptimizerFactory",
    "CrossAssetValidator",
    "ValidationCriteria",
    "AssetSelector",
    "ValidationAsset",
    "AssetValidationResult",
    "CrossAssetValidationResult",
    "create_cross_asset_validator",
    "run_cross_asset_validation",
]
