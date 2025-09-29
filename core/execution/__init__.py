"""
Advanced Execution Module

This module provides sophisticated order execution strategies including:
- Smart Order Execution (large order splitting)
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- DCA (Dollar-Cost Averaging)

All executors inherit from BaseExecutor and provide consistent interfaces.
"""

from .base_executor import BaseExecutor
from .dca_executor import DCAExecutor
from .smart_order_executor import SmartOrderExecutor
from .twap_executor import TWAPExecutor
from .vwap_executor import VWAPExecutor

__all__ = [
    "BaseExecutor",
    "SmartOrderExecutor",
    "TWAPExecutor",
    "VWAPExecutor",
    "DCAExecutor",
]
