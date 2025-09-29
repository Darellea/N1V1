"""
Predictive Models Module

This module provides predictive models for trading signals including:
- Price direction classification
- Volatility forecasting
- Volume surge detection

All models are designed to work with time series data and provide
predictions that can be used as filters in trading strategies.
"""

from .predictive_model_manager import PredictiveModelManager
from .price_predictor import PricePredictor
from .types import PredictionContext
from .volatility_predictor import VolatilityPredictor
from .volume_predictor import VolumePredictor

__all__ = [
    "PricePredictor",
    "VolatilityPredictor",
    "VolumePredictor",
    "PredictiveModelManager",
    "PredictionContext",
]
