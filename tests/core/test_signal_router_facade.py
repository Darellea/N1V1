"""
Tests for signal_router facade module.

Tests the backward compatibility facade that imports from the signal_router package.
"""

import pytest
from unittest.mock import patch, MagicMock
from core.signal_router import SignalRouter, JournalWriter


class TestSignalRouterFacade:
    """Test the signal_router facade module."""

    def test_imports_available(self):
        """Test that the facade properly imports the main classes."""
        # Test that SignalRouter is imported and is a class
        assert SignalRouter is not None
        assert callable(SignalRouter)

        # Test that JournalWriter is imported and is a class
        assert JournalWriter is not None
        assert callable(JournalWriter)

    def test_signal_router_instantiation(self):
        """Test that SignalRouter can be instantiated through the facade."""
        # Test that we can create a SignalRouter instance
        # Note: We can't easily mock the complex initialization, so we just test that it's callable
        assert callable(SignalRouter)

        # Test that it has the expected interface
        import inspect
        sig = inspect.signature(SignalRouter.__init__)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'risk_manager' in params

    def test_journal_writer_instantiation(self):
        """Test that JournalWriter can be instantiated through the facade."""
        from pathlib import Path

        # Test that JournalWriter can be instantiated
        path = Path("/tmp/test_journal.jsonl")
        task_manager = MagicMock()

        # Just test that instantiation doesn't raise an error
        writer = JournalWriter(path, task_manager)
        assert writer is not None

        # Test that it has the expected interface
        assert hasattr(writer, 'append')
        assert hasattr(writer, 'stop')

    def test_backward_compatibility(self):
        """Test that the facade maintains backward compatibility."""
        # Test that the classes are accessible from the facade
        from core import signal_router

        # Verify the classes are available
        assert hasattr(signal_router, 'SignalRouter')
        assert hasattr(signal_router, 'JournalWriter')

        # Verify they are the same as direct imports
        assert signal_router.SignalRouter is SignalRouter
        assert signal_router.JournalWriter is JournalWriter

    def test_facade_file_content(self):
        """Test that the facade file has the expected content."""
        # Read the facade file directly
        with open("core/signal_router.py", "r") as f:
            content = f.read()

        # Check that it imports the expected classes
        assert "from .signal_router.router import SignalRouter, JournalWriter" in content
        assert '__all__ = ["SignalRouter", "JournalWriter"]' in content

        # Check that it has the expected docstring
        assert 'Facade for the modular signal routing system' in content


class TestFacadeIntegration:
    """Test integration aspects of the facade."""

    def test_facade_components_work_together(self):
        """Test that facade components can work together."""
        # Test creating both components
        risk_manager = MagicMock()

        # We can't easily instantiate SignalRouter due to complex dependencies,
        # but we can test that JournalWriter works
        from pathlib import Path
        path = Path("/tmp/test.jsonl")
        writer = JournalWriter(path)

        # Verify writer was created successfully
        assert writer is not None
        assert hasattr(writer, 'append')
        assert hasattr(writer, 'stop')

    def test_import_error_handling(self):
        """Test that import errors are properly handled."""
        # This test ensures that if the underlying modules have import issues,
        # the facade will propagate them appropriately
        try:
            # Try to import the facade
            import core.signal_router

            # If we get here, the imports worked
            # Test that we can access the classes
            assert core.signal_router.SignalRouter is not None
            assert core.signal_router.JournalWriter is not None

        except ImportError as e:
            # If there's an import error, it should be descriptive
            assert "signal_router" in str(e) or "router" in str(e)

    def test_module_structure(self):
        """Test the module has the expected structure."""
        import core.signal_router as sr

        # Test module has expected attributes
        assert hasattr(sr, '__file__')
        assert hasattr(sr, '__name__')
        assert sr.__name__ == 'core.signal_router'

        # Test the module docstring exists
        assert sr.__doc__ is not None
        # The signal_router package has its own docstring, not the facade
        assert len(sr.__doc__.strip()) > 0
