#!/usr/bin/env python3
"""
Predictive Regime Forecasting Demo

This script demonstrates the Predictive Regime Forecasting system integrated
with the N1V1 trading framework. It shows how to:

1. Train regime forecasting models on historical data
2. Generate real-time regime forecasts
3. Integrate forecasts with strategy selection
4. Monitor forecast performance

Usage:
    python demo_regime_forecasting.py
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import N1V1 components
from strategies.regime.regime_forecaster import (
    get_regime_forecaster, RegimeForecaster, ForecastingResult
)
from strategies.regime.market_regime import detect_enhanced_market_regime
from strategies.regime.strategy_selector import (
    get_strategy_selector, get_recommended_strategies_from_forecast
)
from core.timeframe_manager import TimeframeManager
from data.data_fetcher import DataFetcher
from utils.logger import setup_logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeForecastingDemo:
    """Demonstration of the Predictive Regime Forecasting system."""

    def __init__(self):
        self.forecaster = None
        self.selector = None
        self.timeframe_manager = None
        self.data_fetcher = None

    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Predictive Regime Forecasting Demo")

        # Initialize data fetcher
        self.data_fetcher = DataFetcher({
            "name": "kucoin",
            "sandbox": True,
            "markets": ["BTC/USDT"]
        })
        await self.data_fetcher.initialize()

        # Initialize timeframe manager
        tf_config = {
            "cache_ttl_seconds": 300,
            "max_concurrent_fetches": 2,
            "timestamp_alignment_tolerance_ms": 60000,
            "missing_data_threshold": 0.8
        }
        self.timeframe_manager = TimeframeManager(self.data_fetcher, tf_config)
        await self.timeframe_manager.initialize()

        # Register symbol with timeframes
        self.timeframe_manager.add_symbol("BTC/USDT", ["15m", "1h", "4h"])

        # Initialize forecaster
        forecast_config = {
            "sequence_length": 30,  # Shorter for demo
            "forecasting_horizons": [5, 10],
            "models_enabled": {"xgboost": True, "lstm": False},
            "training": {
                "epochs": 50,  # Fewer epochs for demo
                "batch_size": 16
            }
        }
        self.forecaster = RegimeForecaster(forecast_config)

        # Initialize strategy selector
        self.selector = get_strategy_selector()

        logger.info("Demo initialization complete")

    async def generate_synthetic_data(self, symbol: str = "BTC/USDT",
                                    days: int = 30) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for demonstration.

        Args:
            symbol: Trading symbol
            days: Number of days of data to generate

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Generating synthetic data for {symbol} ({days} days)")

        # Generate timestamps (1-hour intervals)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1h')

        np.random.seed(42)  # For reproducible results

        # Generate synthetic price data with trends and volatility
        n_points = len(timestamps)

        # Base price around $50,000
        base_price = 50000

        # Generate random walk with drift
        drift = 0.0001  # Small upward drift
        volatility = 0.02  # 2% daily volatility

        # Create price series
        price_changes = np.random.normal(drift, volatility/np.sqrt(24), n_points)
        prices = base_price * np.exp(np.cumsum(price_changes))

        # Add some regime-like behavior (trends and ranges)
        for i in range(0, n_points, 120):  # Every 5 days
            if np.random.random() > 0.5:  # 50% chance of trend
                trend_length = np.random.randint(24, 72)  # 1-3 days
                trend_strength = np.random.normal(0.001, 0.0005)
                end_idx = min(i + trend_length, n_points)
                trend_changes = np.linspace(0, trend_strength * trend_length, end_idx - i)
                prices[i:end_idx] *= np.exp(trend_changes)

        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            # Add some intrabar volatility
            high_mult = 1 + np.random.uniform(0, 0.005)
            low_mult = 1 - np.random.uniform(0, 0.005)
            open_price = price * (1 + np.random.normal(0, 0.002))
            close_price = price
            high_price = max(open_price, close_price) * high_mult
            low_price = min(open_price, close_price) * low_mult

            # Volume (simulated)
            volume = np.random.uniform(100, 1000)

            data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        logger.info(f"Generated {len(df)} data points")
        return df

    async def create_training_labels(self, data: pd.DataFrame) -> List[str]:
        """
        Create regime labels for training data.

        Args:
            data: OHLCV DataFrame

        Returns:
            List of regime labels
        """
        logger.info("Creating training labels from historical data")

        labels = []
        window_size = 50  # Same as sequence length

        for i in range(window_size, len(data), 10):  # Sample every 10 points
            window = data.iloc[i-window_size:i]

            try:
                regime_result = detect_enhanced_market_regime(window)
                labels.append(regime_result.regime_name)
            except Exception as e:
                logger.debug(f"Failed to detect regime at index {i}: {e}")
                labels.append("unknown")

        logger.info(f"Created {len(labels)} training labels")
        return labels

    async def train_models(self):
        """Train the forecasting models."""
        logger.info("Starting model training")

        # Generate training data
        training_data = await self.generate_synthetic_data(days=60)
        training_labels = await self.create_training_labels(training_data)

        # Train the models
        training_metrics = await self.forecaster.train_models(training_data, training_labels)

        if training_metrics:
            logger.info("Training completed successfully!")
            for metric in training_metrics:
                logger.info(f"Model: {metric.model_name}")
                logger.info(".3f")
                logger.info(".3f")
                logger.info(".3f")
                logger.info(".3f")
                logger.info(".1f")
        else:
            logger.warning("Training failed or returned no metrics")

    async def demonstrate_forecasting(self):
        """Demonstrate real-time forecasting."""
        logger.info("Starting forecasting demonstration")

        # Generate recent data for forecasting
        recent_data = await self.generate_synthetic_data(days=2)

        # Get multi-timeframe data
        multi_tf_data = await self.timeframe_manager.fetch_multi_timeframe_data("BTC/USDT")

        if multi_tf_data:
            logger.info("Multi-timeframe data fetched successfully")
        else:
            logger.warning("Failed to fetch multi-timeframe data, using single timeframe")

        # Generate forecast
        forecast_result = await self.forecaster.forecast_next_regime(recent_data)

        if forecast_result.predictions:
            logger.info("🎯 FORECAST GENERATED SUCCESSFULLY!")
            logger.info(f"Current Regime: {forecast_result.current_regime}")
            logger.info("Predictions:")

            for horizon, predictions in forecast_result.predictions.items():
                logger.info(f"  {horizon}-step ahead:")
                for regime, probability in predictions.items():
                    logger.info(".3f")

                # Get confidence for this horizon
                confidence = forecast_result.confidence_scores.get(horizon, 0)
                uncertainty = forecast_result.uncertainty_estimates.get(horizon, 1)
                logger.info(".3f")
                logger.info(".3f")

            # Demonstrate strategy integration
            await self.demonstrate_strategy_integration(forecast_result)

        else:
            logger.warning("No forecast generated")

    async def demonstrate_strategy_integration(self, forecast_result: ForecastingResult):
        """Demonstrate how forecasts integrate with strategy selection."""
        logger.info("🔄 Demonstrating Strategy Integration")

        # Generate sample market data
        market_data = await self.generate_synthetic_data(days=1)

        # Get base strategy selection
        base_strategy = self.selector.select_strategy(market_data)
        logger.info(f"Base Strategy Selection: {base_strategy.__name__ if base_strategy else 'None'}")

        # Get forecast-influenced selection
        forecast_strategy = self.selector.select_strategy_with_forecast(market_data, forecast_result)
        logger.info(f"Forecast-Influenced Selection: {forecast_strategy.__name__ if forecast_strategy else 'None'}")

        # Analyze forecast impact
        impact_analysis = self.selector.get_forecast_impact_analysis(forecast_result)

        logger.info("📊 Forecast Impact Analysis:")
        logger.info(f"Forecast Available: {impact_analysis.get('forecast_available', False)}")
        logger.info(f"Recommended Regime: {impact_analysis.get('recommended_regime', 'N/A')}")
        logger.info(f"Confidence Level: {impact_analysis.get('confidence_level', 0):.3f}")
        logger.info(f"Recommended Strategies: {impact_analysis.get('recommended_strategies', [])}")

        risk_implications = impact_analysis.get('risk_implications', {})
        if risk_implications:
            logger.info(f"Risk Assessment: {risk_implications.get('confidence', 'Unknown')}")
            logger.info(f"Recommended Action: {risk_implications.get('action', 'Monitor')}")

    async def run_performance_analysis(self):
        """Run performance analysis of the forecasting system."""
        logger.info("📈 Running Performance Analysis")

        # Get model performance metrics
        performance = self.forecaster.get_model_performance()

        logger.info("Model Performance Summary:")
        logger.info(f"Trained Models: {performance.get('models_trained', [])}")
        logger.info(f"Is Trained: {performance.get('is_trained', False)}")

        training_history = performance.get('training_history', [])
        if training_history:
            logger.info("Training History:")
            for i, metrics in enumerate(training_history):
                logger.info(f"  Model {i+1}: {metrics.get('model_name', 'Unknown')}")
                logger.info(".3f")
                logger.info(".3f")

    async def save_models(self):
        """Save trained models."""
        logger.info("💾 Saving trained models")

        model_path = "models/demo_regime_forecasting"
        self.forecaster.save_models(model_path)

        logger.info(f"Models saved to: {model_path}")

    async def run_demo(self):
        """Run the complete demonstration."""
        logger.info("🚀 Starting Predictive Regime Forecasting Demo")
        logger.info("=" * 60)

        try:
            # Initialize components
            await self.initialize()

            # Train models
            await self.train_models()

            # Demonstrate forecasting
            await self.demonstrate_forecasting()

            # Run performance analysis
            await self.run_performance_analysis()

            # Save models
            await self.save_models()

            logger.info("✅ Demo completed successfully!")

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        finally:
            # Cleanup
            if self.timeframe_manager:
                await self.timeframe_manager.shutdown()
            if self.data_fetcher:
                await self.data_fetcher.shutdown()


async def main():
    """Main entry point."""
    demo = RegimeForecastingDemo()
    await demo.run_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
