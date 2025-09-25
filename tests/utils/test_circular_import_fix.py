"""
Test to verify that circular import issues have been resolved.

This test ensures that importing modules that previously had circular
dependencies works correctly without ImportError.
"""

import sys
import importlib
import time
import os
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_import_chain():
    """Test importing modules individually to check for basic import issues."""
    modules_to_test = [
        'utils.error_handler',
        'utils.security',
        'utils.config_loader',
        'utils.logging_manager',
        'core.async_optimizer'
    ]

    imported_modules = {}
    import_times = {}

    print("Testing individual module imports...")

    for module_name in modules_to_test:
        try:
            start_time = time.time()
            module = importlib.import_module(module_name)
            import_time = time.time() - start_time

            imported_modules[module_name] = module
            import_times[module_name] = import_time

            print(".3f")
        except ImportError as e:
            print(f"❌ FAILED to import {module_name}: {e}")
            assert False
        except Exception as e:
            print(f"❌ ERROR importing {module_name}: {e}")
            assert False

    print("\n✅ All modules imported successfully!")
    print("\nImport performance:")
    for module_name, import_time in import_times.items():
        print(".3f")

    # Test that we can access key functions from imported modules
    try:
        # Test error_handler functions
        from utils.error_handler import get_error_handler, ErrorHandler
        error_handler = get_error_handler()
        assert isinstance(error_handler, ErrorHandler)
        print("✅ Error handler functions accessible")

        # Test security functions
        from utils.security import sanitize_error_message, get_credential_manager
        test_message = "Test message with API_KEY=secret123"
        sanitized = sanitize_error_message(test_message)
        assert "secret123" not in sanitized
        print("✅ Security functions accessible")

        # Test config loader functions
        from utils.config_loader import get_config
        config = get_config("nonexistent_key", "default_value")
        assert config == "default_value"
        print("✅ Config loader functions accessible")

        # Test async optimizer functions
        from core.async_optimizer import get_async_optimizer
        optimizer = get_async_optimizer()
        assert optimizer is not None
        print("✅ Async optimizer functions accessible")

    except Exception as e:
        print(f"❌ ERROR testing module functions: {e}")
        assert False

    print("\n🎉 Basic import verification PASSED!")
    assert True


def test_lazy_import_functionality():
    """Test that lazy import functions work correctly."""
    print("\nTesting lazy import functionality...")

    try:
        # Test the lazy import function
        from utils.error_handler import _get_sanitize_error_message
        sanitize_func = _get_sanitize_error_message()
        assert callable(sanitize_func)

        # Test that it actually works
        test_input = "Error with API_KEY=secret123"
        result = sanitize_func(test_input)
        assert "secret123" not in result
        assert "API_KEY" in result  # Should still contain the key name but mask the value

        print("✅ Lazy import functionality working correctly")
        assert True

    except Exception as e:
        print(f"❌ ERROR testing lazy import: {e}")
        assert False


def test_module_interactions():
    """Test that modules can interact without circular import issues."""
    print("\nTesting module interactions...")

    try:
        # Import all modules
        import utils.error_handler as eh
        import utils.security as sec
        import utils.logger as log
        import utils.config_loader as cl

        # Test that error handler can use security functions through lazy import
        error_handler = eh.get_error_handler()

        # Create a test error context
        context = error_handler.create_error_context(
            component="test",
            operation="circular_import_test"
        )

        # This should work without circular import issues
        print("✅ Module interactions working correctly")
        assert True

    except Exception as e:
        print(f"❌ ERROR testing module interactions: {e}")
        assert False


if __name__ == "__main__":
    print("=" * 60)
    print("CIRCULAR IMPORT FIX VERIFICATION TEST")
    print("=" * 60)

    success = True

    # Run all tests
    try:
        test_import_chain()
    except AssertionError:
        success = False

    try:
        test_lazy_import_functionality()
    except AssertionError:
        success = False

    try:
        test_module_interactions()
    except AssertionError:
        success = False

    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - Circular import issues resolved!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - Circular import issues may still exist")
        sys.exit(1)
