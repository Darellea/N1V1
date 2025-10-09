"""
core/signal_router.py

Facade for the modular signal routing system.

This file provides backward compatibility while delegating to the new
modular components in the signal_router/ package.
"""

# Import main classes from the package
from .signal_router.router import SignalRouter, JournalWriter

# Re-export for backward compatibility
__all__ = ["SignalRouter", "JournalWriter"]
