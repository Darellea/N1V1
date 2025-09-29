"""
portfolio/__init__.py

Capital & Portfolio Management Layer for Multi-Asset Trading.

This module provides comprehensive portfolio management capabilities including:
- Multi-asset position tracking and capital allocation
- Dynamic asset rotation based on momentum and performance
- Adaptive rebalancing with multiple allocation schemes
- Risk management through hedging strategies
- Portfolio metrics and performance reporting
"""

from .allocator import (
    CapitalAllocator,
    EqualWeightAllocator,
    MomentumWeightAllocator,
    RiskParityAllocator,
)
from .hedging import PortfolioHedger
from .portfolio_manager import PortfolioManager

__all__ = [
    "PortfolioManager",
    "CapitalAllocator",
    "EqualWeightAllocator",
    "RiskParityAllocator",
    "MomentumWeightAllocator",
    "PortfolioHedger",
]
