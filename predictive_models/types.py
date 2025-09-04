"""
Type definitions for predictive models.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PredictionContext:
    """
    Container for predictions from all predictive models.

    Attributes:
        price_direction: Predicted price direction ("up", "down", "neutral")
        volatility: Predicted volatility regime ("high", "low")
        volume_surge: Whether volume surge is detected (True/False)
        confidence: Overall confidence score (0.0 to 1.0)
        price_confidence: Confidence for price direction prediction
        volatility_confidence: Confidence for volatility prediction
        volume_confidence: Confidence for volume surge detection
    """
    price_direction: str = "neutral"
    volatility: str = "low"
    volume_surge: bool = False
    confidence: float = 0.5
    price_confidence: Optional[float] = None
    volatility_confidence: Optional[float] = None
    volume_confidence: Optional[float] = None

    def __post_init__(self):
        """Validate prediction values."""
        if self.price_direction not in ["up", "down", "neutral"]:
            raise ValueError(f"Invalid price_direction: {self.price_direction}")
        if self.volatility not in ["high", "low"]:
            raise ValueError(f"Invalid volatility: {self.volatility}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Invalid confidence: {self.confidence}")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "price_direction": self.price_direction,
            "volatility": self.volatility,
            "volume_surge": self.volume_surge,
            "confidence": self.confidence,
            "price_confidence": self.price_confidence,
            "volatility_confidence": self.volatility_confidence,
            "volume_confidence": self.volume_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PredictionContext":
        """Create from dictionary."""
        return cls(
            price_direction=data.get("price_direction", "neutral"),
            volatility=data.get("volatility", "low"),
            volume_surge=data.get("volume_surge", False),
            confidence=data.get("confidence", 0.5),
            price_confidence=data.get("price_confidence"),
            volatility_confidence=data.get("volatility_confidence"),
            volume_confidence=data.get("volume_confidence"),
        )
