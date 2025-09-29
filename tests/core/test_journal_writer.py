import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from core.signal_router import JournalWriter


@pytest.mark.asyncio
async def test_journal_writer_initialization():
    """Test JournalWriter initialization and basic properties."""
    with tempfile.TemporaryDirectory() as temp_dir:
        journal_path = Path(temp_dir) / "test_journal.jsonl"
        writer = JournalWriter(journal_path)

        assert writer.path == journal_path
        assert writer._queue is not None
        assert writer._task is None
        assert writer.task_manager is None


@pytest.mark.asyncio
async def test_journal_writer_append_with_running_loop():
    """Test JournalWriter append method with a running event loop."""
    with tempfile.TemporaryDirectory() as temp_dir:
        journal_path = Path(temp_dir) / "test_journal.jsonl"
        writer = JournalWriter(journal_path)

        test_entry = {"action": "test", "data": "sample"}

        # Append entry - this should trigger lazy task creation
        writer.append(test_entry)

        # Give some time for async operations
        await asyncio.sleep(0.1)

        # Check that file was created and contains the entry
        assert journal_path.exists()
        with open(journal_path, "r") as f:
            content = f.read()
            assert "test" in content
            assert "sample" in content

        # Stop the writer to clean up
        await writer.stop()


@pytest.mark.asyncio
async def test_journal_writer_append_without_loop():
    """Test JournalWriter append method when no event loop is running."""
    with tempfile.TemporaryDirectory() as temp_dir:
        journal_path = Path(temp_dir) / "test_journal.jsonl"
        writer = JournalWriter(journal_path)

        test_entry = {"action": "test_no_loop", "data": "fallback"}

        # Simulate no event loop by patching get_event_loop to raise RuntimeError
        with patch("asyncio.get_event_loop", side_effect=RuntimeError("No event loop")):
            writer.append(test_entry)

        # Should have written synchronously
        assert journal_path.exists()
        with open(journal_path, "r") as f:
            content = f.read()
            assert "test_no_loop" in content
            assert "fallback" in content


@pytest.mark.asyncio
async def test_journal_writer_worker_loop():
    """Test the JournalWriter worker processes entries correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        journal_path = Path(temp_dir) / "test_journal.jsonl"
        writer = JournalWriter(journal_path)

        # Manually start the worker task
        writer._task = asyncio.create_task(writer._worker())

        # Put entries in queue
        test_entries = [
            {"action": "store", "id": "1"},
            {"action": "cancel", "id": "2"},
            None,  # Sentinel to stop
        ]

        for entry in test_entries:
            await writer._queue.put(entry)

        # Wait for worker to finish
        await writer._task

        # Check file contents
        assert journal_path.exists()
        with open(journal_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2  # Two entries before sentinel
            assert "store" in lines[0]
            assert "cancel" in lines[1]


@pytest.mark.asyncio
async def test_journal_writer_stop():
    """Test JournalWriter stop method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        journal_path = Path(temp_dir) / "test_journal.jsonl"
        writer = JournalWriter(journal_path)

        # Start a worker task
        writer._task = asyncio.create_task(writer._worker())

        # Stop the writer
        await writer.stop()

        # Task should be done
        assert writer._task.done()
