"""
Predictive Model Manager

Manages all predictive models and coordinates their predictions.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional
from .price_predictor import PricePredictor
from .volatility_predictor import VolatilityPredictor
from .volume_predictor import VolumePredictor
from .types import PredictionContext

logger = logging.getLogger(__name__)


class PredictiveModelManager:
    """
    Manages predictive models and coordinates predictions across all models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the predictive model manager.

        Args:
            config: Configuration dictionary for predictive models
        """
        self.config = config
        self.enabled = config.get("enabled", False)

        if not self.enabled:
            logger.info("Predictive models are disabled")
            self.price_predictor = None
            self.volatility_predictor = None
            self.volume_predictor = None
            return

        # Initialize individual predictors
        models_config = config.get("models", {})

        self.price_predictor = PricePredictor(models_config.get("price_direction", {}))
        self.volatility_predictor = VolatilityPredictor(models_config.get("volatility", {}))
        self.volume_predictor = VolumePredictor(models_config.get("volume_surge", {}))

        logger.info("PredictiveModelManager initialized")

    def load_models(self) -> bool:
        """
        Load all trained models from disk.

        Returns:
            True if all models loaded successfully
        """
        if not self.enabled:
            return True

        success = True

        if self.price_predictor:
            if not self.price_predictor.load_model():
                logger.warning("Failed to load price prediction model")
                success = False

        if self.volatility_predictor:
            if not self.volatility_predictor.load_model():
                logger.warning("Failed to load volatility prediction model")
                success = False

        if self.volume_predictor:
            if not self.volume_predictor.load_model():
                logger.warning("Failed to load volume prediction model")
                success = False

        if success:
            logger.info("All predictive models loaded successfully")
        else:
            logger.warning("Some predictive models failed to load")

        return success

    def predict(self, market_data: pd.DataFrame) -> PredictionContext:
        """
        Generate predictions using all enabled models.

        Args:
            market_data: DataFrame with OHLCV data

        Returns:
            PredictionContext with all predictions
        """
        if not self.enabled:
            return PredictionContext()

        predictions = PredictionContext()
        confidence_scores = []

        try:
            # Price direction prediction
            if self.price_predictor:
                direction, price_conf = self.price_predictor.predict(market_data)
                predictions.price_direction = direction
                predictions.price_confidence = price_conf
                confidence_scores.append(price_conf)
                logger.debug(f"Price direction: {direction} (confidence: {price_conf:.3f})")

            # Volatility prediction
            if self.volatility_predictor:
                regime, vol_conf = self.volatility_predictor.predict(market_data)
                predictions.volatility = regime
                predictions.volatility_confidence = vol_conf
                confidence_scores.append(vol_conf)
                logger.debug(f"Volatility regime: {regime} (confidence: {vol_conf:.3f})")

            # Volume surge detection
            if self.volume_predictor:
                is_surge, volume_conf = self.volume_predictor.predict(market_data)
                predictions.volume_surge = is_surge
                predictions.volume_confidence = volume_conf
                confidence_scores.append(volume_conf)
                logger.debug(f"Volume surge: {is_surge} (confidence: {volume_conf:.3f})")

            # Calculate overall confidence
            if confidence_scores:
                predictions.confidence = sum(confidence_scores) / len(confidence_scores)
            else:
                predictions.confidence = 0.5

            logger.debug(f"Overall prediction confidence: {predictions.confidence:.3f}")

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            # Return neutral predictions on error
            return PredictionContext()

        return predictions

    def train_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all predictive models.

        Args:
            training_data: DataFrame with OHLCV training data

        Returns:
            Dictionary with training metrics for each model
        """
        if not self.enabled:
            return {"status": "disabled"}

        training_results = {
            "price_direction": {},
            "volatility": {},
            "volume_surge": {}
        }

        try:
            # Train price predictor
            if self.price_predictor:
                logger.info("Training price direction model...")
                training_results["price_direction"] = self.price_predictor.train(training_data)

            # Train volatility predictor
            if self.volatility_predictor:
                logger.info("Training volatility model...")
                training_results["volatility"] = self.volatility_predictor.train(training_data)

            # Train volume predictor
            if self.volume_predictor:
                logger.info("Training volume surge model...")
                training_results["volume_surge"] = self.volume_predictor.train(training_data)

            logger.info("All models trained successfully")
            training_results["status"] = "success"

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            training_results["status"] = "error"
            training_results["error"] = str(e)

        return training_results

    def get_model_status(self) -> Dict[str, Any]:
        """
        Get the status of all models.

        Returns:
            Dictionary with model status information
        """
        status = {
            "enabled": self.enabled,
            "models": {}
        }

        if not self.enabled:
            return status

        # Check price predictor
        if self.price_predictor:
            status["models"]["price_direction"] = {
                "type": self.price_predictor.model_type,
                "loaded": self.price_predictor.model is not None,
                "model_path": self.price_predictor.model_path
            }

        # Check volatility predictor
        if self.volatility_predictor:
            status["models"]["volatility"] = {
                "type": self.volatility_predictor.model_type,
                "loaded": self.volatility_predictor.model is not None,
                "model_path": self.volatility_predictor.model_path
            }

        # Check volume predictor
        if self.volume_predictor:
            status["models"]["volume_surge"] = {
                "type": self.volume_predictor.model_type,
                "loaded": self.volume_predictor.model is not None or self.volume_predictor.model_type == "zscore",
                "model_path": self.volume_predictor.model_path
            }

        return status

    def should_allow_signal(self, signal_type: str, predictions: PredictionContext) -> bool:
        """
        Determine if a trading signal should be allowed based on predictions.

        Args:
            signal_type: Type of signal (e.g., "BUY", "SELL")
            predictions: PredictionContext with current predictions

        Returns:
            True if signal should be allowed
        """
        if not self.enabled:
            return True  # Allow all signals if predictive models are disabled

        # Get confidence threshold from config
        confidence_threshold = self.config.get("confidence_threshold", 0.5)

        # Check overall confidence
        if predictions.confidence < confidence_threshold:
            logger.debug(f"Signal blocked: overall confidence {predictions.confidence:.3f} < threshold {confidence_threshold}")
            return False

        # Apply model-specific filters
        models_config = self.config.get("models", {})

        # Price direction filter
        price_config = models_config.get("price_direction", {})
        if price_config.get("enabled", True):
            price_threshold = price_config.get("confidence_threshold", 0.6)
            if predictions.price_confidence and predictions.price_confidence < price_threshold:
                logger.debug(f"Signal blocked: price confidence {predictions.price_confidence:.3f} < threshold {price_threshold}")
                return False

            # Check if signal direction matches prediction
            if signal_type.upper() in ["BUY", "ENTRY_LONG"] and predictions.price_direction == "down":
                logger.debug("BUY signal blocked: price direction predicts down")
                return False
            elif signal_type.upper() in ["SELL", "ENTRY_SHORT"] and predictions.price_direction == "up":
                logger.debug("SELL signal blocked: price direction predicts up")
                return False

        # Volatility filter
        vol_config = models_config.get("volatility", {})
        if vol_config.get("enabled", True):
            vol_threshold = vol_config.get("confidence_threshold", 0.6)
            if predictions.volatility_confidence and predictions.volatility_confidence < vol_threshold:
                logger.debug(f"Signal blocked: volatility confidence {predictions.volatility_confidence:.3f} < threshold {vol_threshold}")
                return False

            # Optionally block signals in high volatility
            if vol_config.get("block_high_volatility", False) and predictions.volatility == "high":
                logger.debug("Signal blocked: high volatility regime")
                return False

        # Volume surge filter
        volume_config = models_config.get("volume_surge", {})
        if volume_config.get("enabled", True):
            volume_threshold = volume_config.get("confidence_threshold", 0.6)
            if predictions.volume_confidence and predictions.volume_confidence < volume_threshold:
                logger.debug(f"Signal blocked: volume confidence {predictions.volume_confidence:.3f} < threshold {volume_threshold}")
                return False

            # Optionally require volume surge for signals
            if volume_config.get("require_surge", False) and not predictions.volume_surge:
                logger.debug("Signal blocked: volume surge required but not detected")
                return False

        return True
