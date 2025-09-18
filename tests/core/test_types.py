"""
Tests for core/types.py module.

Tests the TradingMode enum and other shared types.
"""

import pytest
from core.types import TradingMode


class TestTradingMode:
    """Test TradingMode enum functionality."""

    def test_trading_mode_values(self):
        """Test that TradingMode has the expected values."""
        assert TradingMode.LIVE.value == "live"
        assert TradingMode.PAPER.value == "paper"
        assert TradingMode.BACKTEST.value == "backtest"

    def test_trading_mode_names(self):
        """Test that TradingMode has the expected names."""
        assert TradingMode.LIVE.name == "LIVE"
        assert TradingMode.PAPER.name == "PAPER"
        assert TradingMode.BACKTEST.name == "BACKTEST"

    def test_trading_mode_members(self):
        """Test that all expected members exist."""
        members = list(TradingMode)
        assert len(members) == 3
        assert TradingMode.LIVE in members
        assert TradingMode.PAPER in members
        assert TradingMode.BACKTEST in members

    def test_trading_mode_string_representation(self):
        """Test string representation of TradingMode."""
        assert str(TradingMode.LIVE) == "TradingMode.LIVE"
        assert repr(TradingMode.LIVE) == "<TradingMode.LIVE: 'live'>"

    def test_trading_mode_equality(self):
        """Test equality comparisons."""
        assert TradingMode.LIVE == TradingMode.LIVE
        assert TradingMode.LIVE != TradingMode.PAPER

    def test_trading_mode_hashable(self):
        """Test that TradingMode instances are hashable."""
        mode_set = {TradingMode.LIVE, TradingMode.PAPER, TradingMode.BACKTEST}
        assert len(mode_set) == 3

    def test_trading_mode_iteration(self):
        """Test iteration over TradingMode."""
        modes = []
        for mode in TradingMode:
            modes.append(mode)

        assert len(modes) == 3
        assert TradingMode.LIVE in modes
        assert TradingMode.PAPER in modes
        assert TradingMode.BACKTEST in modes

    def test_trading_mode_by_value(self):
        """Test accessing TradingMode by value."""
        assert TradingMode("live") == TradingMode.LIVE
        assert TradingMode("paper") == TradingMode.PAPER
        assert TradingMode("backtest") == TradingMode.BACKTEST

    def test_trading_mode_invalid_value(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            TradingMode("invalid")

    def test_trading_mode_by_name(self):
        """Test accessing TradingMode by name."""
        assert TradingMode["LIVE"] == TradingMode.LIVE
        assert TradingMode["PAPER"] == TradingMode.PAPER
        assert TradingMode["BACKTEST"] == TradingMode.BACKTEST

    def test_trading_mode_invalid_name(self):
        """Test that invalid names raise KeyError."""
        with pytest.raises(KeyError):
            TradingMode["INVALID"]


class TestTypesModule:
    """Test the types module structure."""

    def test_module_imports(self):
        """Test that the module can be imported successfully."""
        import core.types
        assert hasattr(core.types, 'TradingMode')
        assert core.types.TradingMode is TradingMode

    def test_module_docstring(self):
        """Test that the module has a docstring."""
        import core.types
        assert core.types.__doc__ is not None
        assert "package initializer" in core.types.__doc__

    def test_module_attributes(self):
        """Test that the module has expected attributes."""
        import core.types
        assert hasattr(core.types, '__file__')
        assert hasattr(core.types, '__name__')
        assert core.types.__name__ == 'core.types'
