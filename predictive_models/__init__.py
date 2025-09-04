"""
Predictive Models Module

This module provides predictive models for trading signals including:
- Price direction classification
- Volatility forecasting
- Volume surge detection

All models are designed to work with time series data and provide
predictions that can be used as filters in trading strategies.
"""

from .price_predictor import PricePredictor
from .volatility_predictor import VolatilityPredictor
from .volume_predictor import VolumePredictor
from .predictive_model_manager import PredictiveModelManager
from .types import PredictionContext

__all__ = [
    'PricePredictor',
    'VolatilityPredictor',
    'VolumePredictor',
    'PredictiveModelManager',
    'PredictionContext'
]
