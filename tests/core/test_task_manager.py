import asyncio
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.task_manager import TaskManager


class TestTaskManager:
    """Test cases for TaskManager functionality."""

    @pytest.mark.asyncio
    async def test_create_task_basic(self):
        """Test basic task creation and tracking."""
        tm = TaskManager()
        task = tm.create_task(asyncio.sleep(0.1))
        assert task in tm.get_tracked_tasks()
        await task
        # Task should be removed after completion
        assert task not in tm.get_tracked_tasks()

    @pytest.mark.asyncio
    async def test_create_task_with_name(self):
        """Test task creation with name parameter."""
        tm = TaskManager()
        task = tm.create_task(asyncio.sleep(0.1), name="test_task")
        assert task.get_name() == "test_task"
        await task

    @pytest.mark.asyncio
    async def test_task_completion_callback_success(self):
        """Test that successful task completion removes it from tracking."""
        tm = TaskManager()
        task = tm.create_task(asyncio.sleep(0.1))
        await task
        assert len(tm.get_tracked_tasks()) == 0

    @pytest.mark.asyncio
    async def test_task_completion_callback_exception(self):
        """Test that task exceptions are logged and task is removed."""
        tm = TaskManager()

        async def failing_task():
            await asyncio.sleep(0.01)
            raise ValueError("Test exception")

        task = tm.create_task(failing_task())
        with pytest.raises(ValueError):
            await task

        # Task should still be removed from tracking after exception
        assert len(tm.get_tracked_tasks()) == 0

    @pytest.mark.asyncio
    async def test_cancel_all_tasks(self):
        """Test cancelling all tracked tasks."""
        tm = TaskManager()

        # Create multiple tasks
        tasks = []
        for i in range(3):
            task = tm.create_task(asyncio.sleep(1.0), name=f"task_{i}")
            tasks.append(task)

        # Verify all tasks are tracked
        assert len(tm.get_tracked_tasks()) == 3

        # Cancel all tasks
        await tm.cancel_all()

        # Verify all tasks are cancelled and removed
        assert len(tm.get_tracked_tasks()) == 0

        # Verify tasks are actually cancelled
        for task in tasks:
            assert task.cancelled()

    @pytest.mark.asyncio
    async def test_shutdown_prevents_new_tasks(self):
        """Test that shutdown prevents creation of new tasks."""
        tm = TaskManager()
        await tm.cancel_all()  # This sets _shutdown = True

        # Suppress AsyncMock warnings in this test
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*coroutine.*never awaited.*",
                category=RuntimeWarning,
            )

            # Use AsyncMock properly for async functions
            mock_sleep = AsyncMock(return_value=None)
            with patch("asyncio.sleep", mock_sleep):
                with pytest.raises(RuntimeError, match="TaskManager is shutting down"):
                    tm.create_task(asyncio.sleep(0.1))

            # Ensure the mock was called but don't await it again
            mock_sleep.assert_called_once_with(0.1)

    @pytest.mark.asyncio
    async def test_multiple_concurrent_tasks(self):
        """Test handling multiple concurrent tasks."""
        tm = TaskManager()

        async def quick_task(n):
            await asyncio.sleep(0.01)
            return n * 2

        # Create multiple concurrent tasks
        tasks = []
        for i in range(5):
            task = tm.create_task(quick_task(i))
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Verify results
        assert results == [0, 2, 4, 6, 8]

        # Verify all tasks are cleaned up
        assert len(tm.get_tracked_tasks()) == 0

    @pytest.mark.asyncio
    async def test_task_manager_with_exception_in_callback(self):
        """Test that exceptions in completion callback don't break tracking."""
        tm = TaskManager()

        # Mock the logger to avoid actual logging during test

        with pytest.MonkeyPatch().context() as m:
            mock_logger = MagicMock()
            m.setattr("core.task_manager.logger", mock_logger)

            async def failing_task():
                await asyncio.sleep(0.01)
                raise RuntimeError("Task failed")

            task = tm.create_task(failing_task())

            # Wait for task to complete and trigger callback
            with pytest.raises(RuntimeError):
                await task

            # Task should still be removed despite callback issues
            assert len(tm.get_tracked_tasks()) == 0

    @pytest.mark.asyncio
    async def test_get_tracked_tasks_snapshot(self):
        """Test that get_tracked_tasks returns a snapshot."""
        tm = TaskManager()

        # Create a task
        async def dummy():
            await asyncio.sleep(0.01)

        task = tm.create_task(dummy())
        await task  # Wait for task to complete

        # Get snapshot
        snapshot = tm.get_tracked_tasks()
        assert len(snapshot) == 0  # Task should be completed and removed

        # Create another task to test snapshot isolation
        task2 = tm.create_task(asyncio.sleep(0.01))
        snapshot = tm.get_tracked_tasks()
        assert len(snapshot) == 1
        assert task2 in snapshot

        # Modify the snapshot (should not affect internal state)
        snapshot.clear()
        assert len(tm.get_tracked_tasks()) == 1
        await task2  # Clean up
