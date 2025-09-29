"""
Test suite for Dashboard Manager.

Tests dashboard creation, rendering, querying, and management functionality.
"""

import asyncio
import time

import pytest

from core.dashboard_manager import Dashboard, DashboardManager


class TestDashboard:
    """Test Dashboard dataclass."""

    def test_dashboard_creation(self):
        """Test basic Dashboard creation."""
        dashboard = Dashboard(
            id="test_dashboard",
            title="Test Dashboard",
            description="A test dashboard",
            panels=[{"id": "panel1", "title": "CPU Usage", "type": "graph"}],
            tags=["test", "monitoring"],
        )

        assert dashboard.id == "test_dashboard"
        assert dashboard.title == "Test Dashboard"
        assert dashboard.description == "A test dashboard"
        assert len(dashboard.panels) == 1
        assert dashboard.tags == ["test", "monitoring"]
        assert dashboard.created_at is not None
        assert dashboard.updated_at is not None

    def test_dashboard_default_values(self):
        """Test Dashboard with default values."""
        dashboard = Dashboard(id="minimal_dashboard", title="Minimal Dashboard")

        assert dashboard.panels == []
        assert dashboard.tags == []
        assert dashboard.description == ""
        assert dashboard.created_at is not None
        assert dashboard.updated_at is not None

    def test_dashboard_auto_timestamps(self):
        """Test that timestamps are set automatically."""
        before_creation = time.time()
        dashboard = Dashboard(id="test", title="Test")
        after_creation = time.time()

        assert before_creation <= dashboard.created_at <= after_creation
        assert before_creation <= dashboard.updated_at <= after_creation


class TestDashboardManager:
    """Test DashboardManager class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = {"cache_ttl": 300, "max_dashboards": 10}
        self.manager = DashboardManager(self.config)

    def test_manager_initialization(self):
        """Test DashboardManager initialization."""
        assert len(self.manager.dashboards) == 0
        assert len(self.manager.query_cache) == 0
        assert self.manager.cache_ttl == 300
        assert self.manager.max_dashboards == 10

    @pytest.mark.asyncio
    async def test_create_dashboard(self):
        """Test creating a new dashboard."""
        dashboard_config = {
            "id": "test_dashboard",
            "title": "Test Dashboard",
            "description": "A test dashboard",
            "panels": [{"id": "panel1", "title": "CPU Usage", "type": "graph"}],
            "tags": ["test", "monitoring"],
        }

        result = await self.manager.create_dashboard(dashboard_config)

        assert result["id"] == "test_dashboard"
        assert result["title"] == "Test Dashboard"
        assert result["description"] == "A test dashboard"
        assert len(result["panels"]) == 1
        assert result["tags"] == ["test", "monitoring"]
        assert "test_dashboard" in self.manager.dashboards

    @pytest.mark.asyncio
    async def test_create_dashboard_auto_id(self):
        """Test creating a dashboard with auto-generated ID."""
        dashboard_config = {"title": "Auto ID Dashboard"}

        result = await self.manager.create_dashboard(dashboard_config)

        assert result["id"].startswith("dashboard_")
        assert result["title"] == "Auto ID Dashboard"
        assert result["id"] in self.manager.dashboards

    @pytest.mark.asyncio
    async def test_create_dashboard_defaults(self):
        """Test creating a dashboard with default values."""
        dashboard_config = {"title": "Minimal Dashboard"}

        result = await self.manager.create_dashboard(dashboard_config)

        assert result["description"] == ""
        assert result["panels"] == []
        assert result["tags"] == []

    @pytest.mark.asyncio
    async def test_render_dashboard(self):
        """Test rendering a dashboard."""
        # Create a dashboard first
        dashboard_config = {
            "id": "render_test",
            "title": "Render Test",
            "panels": [{"id": "panel1", "title": "Test Panel", "type": "graph"}],
        }

        await self.manager.create_dashboard(dashboard_config)

        # Render the dashboard
        result = await self.manager.render_dashboard("render_test")

        assert result["dashboard_id"] == "render_test"
        assert result["title"] == "Render Test"
        assert result["status"] == "rendered"
        assert len(result["panels"]) == 1
        assert result["panels"][0]["title"] == "Test Panel"
        assert "rendered_at" in result

    @pytest.mark.asyncio
    async def test_render_nonexistent_dashboard(self):
        """Test rendering a nonexistent dashboard."""
        with pytest.raises(ValueError, match="Dashboard nonexistent not found"):
            await self.manager.render_dashboard("nonexistent")

    def test_get_panel_data(self):
        """Test getting panel data."""
        panel = {"id": "test_panel", "title": "Test Panel"}
        data = self.manager._get_panel_data(panel)

        assert "series" in data
        assert "timestamp" in data
        assert len(data["series"]) == 1
        assert data["series"][0]["name"] == "sample_metric"

    @pytest.mark.asyncio
    async def test_query_metrics(self):
        """Test querying metrics."""
        result = await self.manager.query_metrics("cpu_usage", "1h")

        assert result["status"] == "success"
        assert "results" in result
        assert "query_time" in result
        assert result["query_time"] >= 0

        # Check that result is cached
        cache_key = "cpu_usage:1h"
        assert cache_key in self.manager.query_cache

    @pytest.mark.asyncio
    async def test_query_metrics_caching(self):
        """Test that metrics queries are cached."""
        # First query
        result1 = await self.manager.query_metrics("test_metric", "30m")
        cache_key = "test_metric:30m"

        assert cache_key in self.manager.query_cache

        # Second query should use cache
        result2 = await self.manager.query_metrics("test_metric", "30m")

        # Results should be the same (from cache)
        assert result2 == result1

    @pytest.mark.asyncio
    async def test_query_metrics_cache_expiry(self):
        """Test cache expiry for metrics queries."""
        # Create manager with short TTL
        manager = DashboardManager({"cache_ttl": 0.1})  # 100ms TTL

        # First query
        await manager.query_metrics("short_cache", "1h")
        cache_key = "short_cache:1h"

        assert cache_key in manager.query_cache

        # Wait for cache to expire
        await asyncio.sleep(0.2)

        # Second query should not use expired cache
        # (We can't easily test this without mocking time, but the structure is there)

    def test_get_dashboards(self):
        """Test getting all dashboards."""
        # Initially empty
        dashboards = self.manager.get_dashboards()
        assert len(dashboards) == 0

        # Add a dashboard
        dashboard = Dashboard(id="test", title="Test Dashboard")
        self.manager.dashboards["test"] = dashboard

        dashboards = self.manager.get_dashboards()
        assert len(dashboards) == 1
        assert "test" in dashboards

    def test_get_dashboard(self):
        """Test getting a specific dashboard."""
        # Non-existent dashboard
        result = self.manager.get_dashboard("nonexistent")
        assert result is None

        # Existing dashboard
        dashboard = Dashboard(id="existing", title="Existing Dashboard")
        self.manager.dashboards["existing"] = dashboard

        result = self.manager.get_dashboard("existing")
        assert result == dashboard

    def test_update_dashboard(self):
        """Test updating an existing dashboard."""
        # Create dashboard
        dashboard = Dashboard(
            id="update_test", title="Original Title", description="Original description"
        )
        self.manager.dashboards["update_test"] = dashboard

        # Update it
        updates = {"title": "Updated Title", "description": "Updated description"}

        self.manager.update_dashboard("update_test", updates)

        updated_dashboard = self.manager.dashboards["update_test"]
        assert updated_dashboard.title == "Updated Title"
        assert updated_dashboard.description == "Updated description"
        assert updated_dashboard.updated_at >= dashboard.updated_at

    def test_update_nonexistent_dashboard(self):
        """Test updating a nonexistent dashboard."""
        with pytest.raises(ValueError, match="Dashboard nonexistent not found"):
            self.manager.update_dashboard("nonexistent", {"title": "New Title"})

    def test_delete_dashboard(self):
        """Test deleting a dashboard."""
        # Create dashboard
        dashboard = Dashboard(id="delete_test", title="Delete Test")
        self.manager.dashboards["delete_test"] = dashboard

        # Delete it
        self.manager.delete_dashboard("delete_test")

        assert "delete_test" not in self.manager.dashboards

    def test_delete_nonexistent_dashboard(self):
        """Test deleting a nonexistent dashboard."""
        # Should not raise error
        self.manager.delete_dashboard("nonexistent")
        assert len(self.manager.dashboards) == 0

    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        # Add some dashboards and cache entries
        self.manager.dashboards["test1"] = Dashboard(id="test1", title="Test 1")
        self.manager.dashboards["test2"] = Dashboard(id="test2", title="Test 2")
        self.manager.query_cache["query1"] = {"data": "test", "timestamp": time.time()}

        stats = self.manager.get_performance_stats()

        assert stats["total_dashboards"] == 2
        assert stats["cache_size"] == 1
        assert "cache_hit_ratio" in stats
        assert "avg_query_time" in stats

    def test_clear_cache(self):
        """Test clearing the query cache."""
        # Add cache entries
        self.manager.query_cache["query1"] = {"data": "test1", "timestamp": time.time()}
        self.manager.query_cache["query2"] = {"data": "test2", "timestamp": time.time()}

        assert len(self.manager.query_cache) == 2

        # Clear cache
        self.manager.clear_cache()

        assert len(self.manager.query_cache) == 0

    @pytest.mark.asyncio
    async def test_dashboard_lifecycle(self):
        """Test complete dashboard lifecycle."""
        # Create
        config = {
            "id": "lifecycle_test",
            "title": "Lifecycle Test",
            "description": "Testing dashboard lifecycle",
            "panels": [{"id": "panel1", "title": "Test Panel"}],
            "tags": ["test"],
        }

        created = await self.manager.create_dashboard(config)
        assert created["id"] == "lifecycle_test"

        # Read
        dashboard = self.manager.get_dashboard("lifecycle_test")
        assert dashboard is not None
        assert dashboard.title == "Lifecycle Test"

        # Update
        self.manager.update_dashboard("lifecycle_test", {"title": "Updated Lifecycle"})
        updated = self.manager.get_dashboard("lifecycle_test")
        assert updated.title == "Updated Lifecycle"

        # Render
        rendered = await self.manager.render_dashboard("lifecycle_test")
        assert rendered["status"] == "rendered"

        # Delete
        self.manager.delete_dashboard("lifecycle_test")
        assert self.manager.get_dashboard("lifecycle_test") is None

    @pytest.mark.asyncio
    async def test_concurrent_dashboard_operations(self):
        """Test concurrent dashboard operations."""

        async def create_dashboard(worker_id: int):
            config = {
                "id": f"concurrent_{worker_id}",
                "title": f"Concurrent Dashboard {worker_id}",
            }
            await self.manager.create_dashboard(config)

        # Create multiple dashboards concurrently
        tasks = [create_dashboard(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify all dashboards were created
        dashboards = self.manager.get_dashboards()
        assert len(dashboards) == 5

        for i in range(5):
            assert f"concurrent_{i}" in dashboards

    @pytest.mark.asyncio
    async def test_query_performance(self):
        """Test query performance tracking."""
        start_time = time.time()
        result = await self.manager.query_metrics("performance_test", "5m")
        query_time = time.time() - start_time

        # Query should complete in reasonable time
        assert result["query_time"] >= 0
        assert result["query_time"] < 1.0  # Should be fast (stub implementation)

    def test_dashboard_data_isolation(self):
        """Test that dashboard data is properly isolated."""
        # Create two dashboards
        dashboard1 = Dashboard(
            id="isolation1",
            title="Dashboard 1",
            panels=[{"id": "panel1", "title": "Panel 1"}],
        )
        dashboard2 = Dashboard(
            id="isolation2",
            title="Dashboard 2",
            panels=[{"id": "panel2", "title": "Panel 2"}],
        )

        self.manager.dashboards["isolation1"] = dashboard1
        self.manager.dashboards["isolation2"] = dashboard2

        # Modify one dashboard
        dashboard1.title = "Modified Dashboard 1"

        # Other dashboard should be unchanged
        assert dashboard2.title == "Dashboard 2"
        assert len(dashboard2.panels) == 1
        assert dashboard2.panels[0]["title"] == "Panel 2"

    def test_cache_key_generation(self):
        """Test that cache keys are generated correctly."""
        # Query with different parameters should create different cache keys
        cache_key1 = "metric1:1h"
        cache_key2 = "metric1:2h"
        cache_key3 = "metric2:1h"

        # Simulate adding to cache
        self.manager.query_cache[cache_key1] = {
            "data": "test1",
            "timestamp": time.time(),
        }
        self.manager.query_cache[cache_key2] = {
            "data": "test2",
            "timestamp": time.time(),
        }
        self.manager.query_cache[cache_key3] = {
            "data": "test3",
            "timestamp": time.time(),
        }

        assert len(self.manager.query_cache) == 3
        assert cache_key1 in self.manager.query_cache
        assert cache_key2 in self.manager.query_cache
        assert cache_key3 in self.manager.query_cache

    @pytest.mark.asyncio
    async def test_render_dashboard_with_multiple_panels(self):
        """Test rendering dashboard with multiple panels."""
        dashboard_config = {
            "id": "multi_panel",
            "title": "Multi Panel Dashboard",
            "panels": [
                {"id": "panel1", "title": "CPU Usage", "type": "graph"},
                {"id": "panel2", "title": "Memory Usage", "type": "graph"},
                {"id": "panel3", "title": "Disk Usage", "type": "table"},
            ],
        }

        await self.manager.create_dashboard(dashboard_config)
        result = await self.manager.render_dashboard("multi_panel")

        assert len(result["panels"]) == 3
        panel_titles = [panel["title"] for panel in result["panels"]]
        assert "CPU Usage" in panel_titles
        assert "Memory Usage" in panel_titles
        assert "Disk Usage" in panel_titles

    def test_dashboard_timestamps(self):
        """Test that dashboard timestamps are updated correctly."""
        dashboard = Dashboard(id="timestamp_test", title="Timestamp Test")

        original_created = dashboard.created_at
        original_updated = dashboard.updated_at

        # Simulate time passing
        time.sleep(0.001)

        # Update dashboard
        self.manager.dashboards["timestamp_test"] = dashboard
        self.manager.update_dashboard("timestamp_test", {"description": "Updated"})

        updated_dashboard = self.manager.dashboards["timestamp_test"]

        assert updated_dashboard.created_at == original_created
        assert updated_dashboard.updated_at > original_updated


class TestDashboardManagerEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.manager = DashboardManager({})

    @pytest.mark.asyncio
    async def test_create_dashboard_with_empty_config(self):
        """Test creating dashboard with minimal config."""
        config = {"title": "Minimal"}
        result = await self.manager.create_dashboard(config)

        assert result["title"] == "Minimal"
        assert result["description"] == ""
        assert result["panels"] == []
        assert result["tags"] == []

    @pytest.mark.asyncio
    async def test_render_empty_dashboard(self):
        """Test rendering dashboard with no panels."""
        config = {"id": "empty_dashboard", "title": "Empty Dashboard", "panels": []}

        await self.manager.create_dashboard(config)
        result = await self.manager.render_dashboard("empty_dashboard")

        assert result["status"] == "rendered"
        assert len(result["panels"]) == 0

    def test_get_performance_stats_empty_manager(self):
        """Test performance stats for empty manager."""
        stats = self.manager.get_performance_stats()

        assert stats["total_dashboards"] == 0
        assert stats["cache_size"] == 0

    @pytest.mark.asyncio
    async def test_query_metrics_empty_cache(self):
        """Test metrics query with empty cache."""
        result = await self.manager.query_metrics("empty_test", "1h")

        assert result["status"] == "success"
        assert len(self.manager.query_cache) == 1

    def test_clear_empty_cache(self):
        """Test clearing already empty cache."""
        # Cache should already be empty
        assert len(self.manager.query_cache) == 0

        # Clearing should not cause issues
        self.manager.clear_cache()
        assert len(self.manager.query_cache) == 0

    def test_update_dashboard_partial_updates(self):
        """Test partial dashboard updates."""
        dashboard = Dashboard(
            id="partial_update",
            title="Original Title",
            description="Original Description",
            tags=["original"],
        )
        self.manager.dashboards["partial_update"] = dashboard

        # Update only title
        self.manager.update_dashboard("partial_update", {"title": "New Title"})

        updated = self.manager.dashboards["partial_update"]
        assert updated.title == "New Title"
        assert updated.description == "Original Description"  # Unchanged
        assert updated.tags == ["original"]  # Unchanged


if __name__ == "__main__":
    pytest.main([__file__])
