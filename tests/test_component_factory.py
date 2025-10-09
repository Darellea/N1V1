"""
tests/test_component_factory.py

Tests for the Dependency Injection ComponentFactory system.
"""


import pytest

from core.component_factory import ComponentFactory


class TestComponentFactoryRegistration:
    """Test component registration functionality."""

    def test_register_component(self):
        """Test registering a component with default singleton behavior."""
        ComponentFactory.reset()

        def mock_builder():
            return {"test": "value"}

        ComponentFactory.register("test_component", mock_builder)

        assert ComponentFactory.is_registered("test_component")
        assert "test_component" in ComponentFactory.list_registered()

    def test_register_component_singleton_true(self):
        """Test registering a component as singleton."""
        ComponentFactory.reset()

        def mock_builder():
            return {"singleton": True}

        ComponentFactory.register("singleton_comp", mock_builder, singleton=True)

        # Should be registered
        assert ComponentFactory.is_registered("singleton_comp")

        # Get stats
        stats = ComponentFactory.get_stats()
        assert stats["registered_components"] == 1
        assert stats["component_names"] == ["singleton_comp"]

    def test_register_component_singleton_false(self):
        """Test registering a component as non-singleton."""
        ComponentFactory.reset()

        call_count = 0

        def mock_builder():
            nonlocal call_count
            call_count += 1
            return {"instance": call_count}

        ComponentFactory.register("non_singleton_comp", mock_builder, singleton=False)

        # Get multiple instances
        instance1 = ComponentFactory.get("non_singleton_comp")
        instance2 = ComponentFactory.get("non_singleton_comp")

        # Should be different instances
        assert instance1 != instance2
        assert instance1["instance"] == 1
        assert instance2["instance"] == 2


class TestComponentFactoryResolution:
    """Test component resolution functionality."""

    def test_get_registered_component(self):
        """Test getting a registered component."""
        ComponentFactory.reset()

        expected_value = {"data": "test"}
        ComponentFactory.register("test_service", lambda: expected_value)

        result = ComponentFactory.get("test_service")
        assert result == expected_value

    def test_get_unregistered_component_raises_keyerror(self):
        """Test that getting an unregistered component raises KeyError."""
        ComponentFactory.reset()

        with pytest.raises(KeyError, match="Component 'unknown' not registered"):
            ComponentFactory.get("unknown")

    def test_singleton_behavior(self):
        """Test that singleton components return the same instance."""
        ComponentFactory.reset()

        call_count = 0

        def mock_builder():
            nonlocal call_count
            call_count += 1
            return {"call": call_count}

        ComponentFactory.register("singleton_test", mock_builder, singleton=True)

        # Get multiple times
        instance1 = ComponentFactory.get("singleton_test")
        instance2 = ComponentFactory.get("singleton_test")

        # Should be the same instance
        assert instance1 is instance2
        assert instance1["call"] == 1  # Builder called only once

    def test_lazy_instantiation(self):
        """Test that components are created lazily."""
        ComponentFactory.reset()

        builder_called = False

        def mock_builder():
            nonlocal builder_called
            builder_called = True
            return {"created": True}

        ComponentFactory.register("lazy_test", mock_builder)

        # Builder should not be called yet
        assert not builder_called

        # Get the component
        result = ComponentFactory.get("lazy_test")

        # Now builder should have been called
        assert builder_called
        assert result["created"] is True


class TestComponentFactoryOverrides:
    """Test environment override functionality."""

    def test_override_component(self):
        """Test overriding a component for testing."""
        ComponentFactory.reset()

        # Register original component
        original = {"type": "original"}
        ComponentFactory.register("override_test", lambda: original)

        # Override with mock
        mock_component = {"type": "mock"}
        ComponentFactory.override("override_test", mock_component)

        # Should return the override
        result = ComponentFactory.get("override_test")
        assert result == mock_component
        assert result["type"] == "mock"

    def test_clear_overrides(self):
        """Test clearing component overrides."""
        ComponentFactory.reset()

        # Register and override
        ComponentFactory.register("clear_test", lambda: {"original": True})
        ComponentFactory.override("clear_test", {"mock": True})

        # Clear overrides
        ComponentFactory.clear_overrides()

        # Should get original component
        result = ComponentFactory.get("clear_test")
        assert result["original"] is True

    def test_override_nonexistent_component(self):
        """Test overriding a component that isn't registered."""
        ComponentFactory.reset()

        mock_component = {"mock": True}
        ComponentFactory.override("nonexistent", mock_component)

        # Should return the override even if not registered
        result = ComponentFactory.get("nonexistent")
        assert result == mock_component


class TestComponentFactoryCacheManagement:
    """Test cache management functionality."""

    def test_clear_cache(self):
        """Test clearing the singleton cache."""
        ComponentFactory.reset()

        call_count = 0

        def mock_builder():
            nonlocal call_count
            call_count += 1
            return {"instance": call_count}

        ComponentFactory.register("cache_test", mock_builder, singleton=True)

        # Get instance (creates it)
        instance1 = ComponentFactory.get("cache_test")
        assert instance1["instance"] == 1

        # Get again (should be cached)
        instance2 = ComponentFactory.get("cache_test")
        assert instance2 is instance1

        # Clear cache
        ComponentFactory.clear_cache()

        # Get again (should create new instance)
        instance3 = ComponentFactory.get("cache_test")
        assert instance3["instance"] == 2
        assert instance3 is not instance1

    def test_reset_factory(self):
        """Test resetting the entire factory."""
        ComponentFactory.reset()

        # Register some components
        ComponentFactory.register("reset_test1", lambda: {"comp1": True})
        ComponentFactory.register("reset_test2", lambda: {"comp2": True})
        ComponentFactory.override("reset_test1", {"override": True})

        # Reset
        ComponentFactory.reset()

        # Should be empty
        assert not ComponentFactory.list_registered()
        stats = ComponentFactory.get_stats()
        assert stats["registered_components"] == 0
        assert stats["cached_singletons"] == 0
        assert stats["active_overrides"] == 0


class TestComponentFactoryStats:
    """Test factory statistics functionality."""

    def test_get_stats_empty_factory(self):
        """Test getting stats for empty factory."""
        ComponentFactory.reset()

        stats = ComponentFactory.get_stats()

        expected_stats = {
            "registered_components": 0,
            "cached_singletons": 0,
            "active_overrides": 0,
            "component_names": [],
        }

        assert stats == expected_stats

    def test_get_stats_with_components(self):
        """Test getting stats with registered components."""
        ComponentFactory.reset()

        # Register components
        ComponentFactory.register("stat_test1", lambda: {"comp1": True}, singleton=True)
        ComponentFactory.register(
            "stat_test2", lambda: {"comp2": True}, singleton=False
        )
        ComponentFactory.override("stat_test1", {"override": True})

        # Get one singleton (creates it) - but overridden components don't get cached
        ComponentFactory.get("stat_test1")

        # Get the non-singleton one (shouldn't be cached)
        ComponentFactory.get("stat_test2")

        stats = ComponentFactory.get_stats()

        assert stats["registered_components"] == 2
        assert stats["cached_singletons"] == 0  # No singletons cached due to override
        assert stats["active_overrides"] == 1
        assert set(stats["component_names"]) == {"stat_test1", "stat_test2"}


class TestComponentFactoryIntegration:
    """Integration tests for component factory usage patterns."""

    def test_factory_as_dependency_injection_container(self):
        """Test using factory as a full DI container."""
        ComponentFactory.reset()

        # Register a service that depends on another service
        def config_service():
            return {"database_url": "sqlite:///:memory:"}

        def user_repository():
            config = ComponentFactory.get("config")
            return {"config": config, "type": "repository"}

        def user_service():
            repo = ComponentFactory.get("user_repository")
            return {"repository": repo, "type": "service"}

        # Register components
        ComponentFactory.register("config", config_service)
        ComponentFactory.register("user_repository", user_repository)
        ComponentFactory.register("user_service", user_service)

        # Resolve the top-level service
        service = ComponentFactory.get("user_service")

        # Verify dependency injection worked
        assert service["type"] == "service"
        assert service["repository"]["type"] == "repository"
        assert service["repository"]["config"]["database_url"] == "sqlite:///:memory:"

    def test_test_override_pattern(self):
        """Test the common pattern of overriding components in tests."""
        ComponentFactory.reset()

        # Register real service
        def real_database():
            return {"type": "real", "connection": "production_db"}

        ComponentFactory.register("database", real_database)

        # In test, override with mock
        mock_db = {"type": "mock", "connection": "memory"}
        ComponentFactory.override("database", mock_db)

        # Application code gets the mock
        db = ComponentFactory.get("database")
        assert db["type"] == "mock"
        assert db["connection"] == "memory"

        # Clean up after test
        ComponentFactory.clear_overrides()

        # Back to real service
        real_db = ComponentFactory.get("database")
        assert real_db["type"] == "real"
        assert real_db["connection"] == "production_db"


class TestComponentFactoryErrorHandling:
    """Test error handling in component factory."""

    def test_builder_exception_handling(self):
        """Test handling exceptions in component builders."""
        ComponentFactory.reset()

        def failing_builder():
            raise ValueError("Builder failed")

        ComponentFactory.register("failing_comp", failing_builder)

        with pytest.raises(ValueError, match="Builder failed"):
            ComponentFactory.get("failing_comp")

    def test_builder_exception_with_singleton(self):
        """Test that singleton cache doesn't cache failed builds."""
        ComponentFactory.reset()

        call_count = 0

        def failing_builder():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return {"success": True}

        ComponentFactory.register("retry_builder", failing_builder, singleton=True)

        # First call should fail
        with pytest.raises(ValueError, match="First call fails"):
            ComponentFactory.get("retry_builder")

        # Second call should succeed (not cached)
        result = ComponentFactory.get("retry_builder")
        assert result["success"] is True
        assert call_count == 2
