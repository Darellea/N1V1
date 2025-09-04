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

from .portfolio_manager import PortfolioManager
from .allocator import (
    CapitalAllocator,
    EqualWeightAllocator,
    RiskParityAllocator,
    MomentumWeightAllocator
)
from .hedging import PortfolioHedger

__all__ = [
    'PortfolioManager',
    'CapitalAllocator',
    'EqualWeightAllocator',
    'RiskParityAllocator',
    'MomentumWeightAllocator',
    'PortfolioHedger'
]
