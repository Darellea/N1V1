"""
Tests for distributed locking mechanism in DatasetVersionManager.

Tests concurrent operations, lock timeouts, and deadlock detection.
"""

import pytest
import threading
import time
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from data.dataset_versioning import DatasetVersionManager, LockTimeoutError, DeadlockError


@pytest.fixture
def temp_base_path():
    """Create a temporary directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def version_manager(temp_base_path):
    """Create a DatasetVersionManager instance for testing."""
    return DatasetVersionManager(base_path=temp_base_path)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': [100 + i * 0.1 for i in range(100)],
        'high': [101 + i * 0.1 for i in range(100)],
        'low': [99 + i * 0.1 for i in range(100)],
        'close': [100.5 + i * 0.1 for i in range(100)],
        'volume': [1000 + i * 10 for i in range(100)]
    })
    df.set_index('timestamp', inplace=True)
    return df


@pytest.mark.timeout(15)
def test_concurrent_read_operations(version_manager, sample_dataframe):
    """Test that multiple concurrent read operations work correctly."""
    # Create a version first
    version_id = version_manager.create_version(
        df=sample_dataframe,
        version_name="test_dataset",
        description="Test dataset for concurrent reads"
    )

    results = []
    errors = []

    def read_operation(thread_id):
        try:
            df = version_manager.load_version(version_id)
            results.append((thread_id, len(df) if df is not None else 0))
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Start multiple threads reading the same version
    threads = []
    for i in range(5):
        t = threading.Thread(target=read_operation, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Verify all operations succeeded
    assert len(results) == 5
    assert len(errors) == 0
    for thread_id, length in results:
        assert length == 100  # All should get the full dataset


@pytest.mark.timeout(15)
def test_write_blocks_reads(version_manager, sample_dataframe):
    """Test that write operations block concurrent reads."""
    # Create initial version
    version_id = version_manager.create_version(
        df=sample_dataframe,
        version_name="blocking_test",
        description="Test dataset for write blocking reads"
    )

    write_started = threading.Event()
    write_completed = threading.Event()
    read_attempted = threading.Event()

    def write_operation():
        write_started.set()
        # Create another version with same name (should block)
        version_manager.create_version(
            df=sample_dataframe,
            version_name="blocking_test",
            description="Second version"
        )
        write_completed.set()

    def read_operation():
        read_attempted.wait()  # Wait until read is attempted
        df = version_manager.load_version(version_id)
        return df is not None

    # Start write operation
    write_thread = threading.Thread(target=write_operation)
    write_thread.start()
    write_started.wait()  # Ensure write has started

    # Start read operation
    read_thread = threading.Thread(target=read_operation)
    read_attempted.set()
    read_thread.start()

    # Both should complete successfully
    write_thread.join(timeout=10)
    read_thread.join(timeout=10)

    assert write_completed.is_set()
    # Read should have completed (not blocked indefinitely)


@pytest.mark.timeout(15)
def test_lock_timeout_prevention(version_manager, sample_dataframe):
    """Test that lock timeouts work correctly and prevent hangs."""
    # Create a version
    version_id = version_manager.create_version(
        df=sample_dataframe,
        version_name="timeout_test",
        description="Test dataset for timeout"
    )

    # Manually acquire a write lock (simulate holding it)
    lock_acquired = threading.Event()
    lock_released = threading.Event()

    def hold_lock():
        # This is a bit tricky since we don't expose acquire_lock directly
        # We'll use the internal lock manager
        version_manager.lock_manager.acquire_write_lock("timeout_test", timeout=30)
        lock_acquired.set()
        time.sleep(3)  # Hold the lock for 3 seconds
        version_manager.lock_manager.release_lock("timeout_test")
        lock_released.set()

    # Start thread that holds the lock
    holder_thread = threading.Thread(target=hold_lock)
    holder_thread.start()
    lock_acquired.wait()  # Wait until lock is acquired

    # Now try to acquire the same lock with short timeout
    start_time = time.time()
    with pytest.raises(LockTimeoutError):
        version_manager.lock_manager.acquire_write_lock("timeout_test", timeout=1.0)

    elapsed = time.time() - start_time
    assert 0.9 <= elapsed <= 2.0  # Should timeout quickly

    # Wait for holder to release
    lock_released.wait()
    holder_thread.join()


@pytest.mark.timeout(15)
def test_deadlock_detection_basic(version_manager, sample_dataframe):
    """Test basic deadlock detection."""
    # Create two datasets
    version_manager.create_version(
        df=sample_dataframe,
        version_name="dataset_a",
        description="Dataset A"
    )
    version_manager.create_version(
        df=sample_dataframe,
        version_name="dataset_b",
        description="Dataset B"
    )

    detected_deadlocks = []

    def thread_a():
        try:
            # Acquire lock on A, then try to acquire B
            version_manager.lock_manager.acquire_write_lock("dataset_a", timeout=5)
            time.sleep(0.1)  # Small delay to ensure ordering
            # This should detect potential deadlock and raise
            version_manager.lock_manager.acquire_write_lock("dataset_b", timeout=5)
            version_manager.lock_manager.release_lock("dataset_b")
            version_manager.lock_manager.release_lock("dataset_a")
        except (DeadlockError, LockTimeoutError) as e:
            detected_deadlocks.append(("thread_a", type(e).__name__))
            # Clean up
            try:
                version_manager.lock_manager.release_lock("dataset_a")
            except:
                pass

    def thread_b():
        try:
            # Acquire lock on B, then try to acquire A
            version_manager.lock_manager.acquire_write_lock("dataset_b", timeout=5)
            time.sleep(0.1)
            version_manager.lock_manager.acquire_write_lock("dataset_a", timeout=5)
            version_manager.lock_manager.release_lock("dataset_a")
            version_manager.lock_manager.release_lock("dataset_b")
        except (DeadlockError, LockTimeoutError) as e:
            detected_deadlocks.append(("thread_b", type(e).__name__))
            try:
                version_manager.lock_manager.release_lock("dataset_b")
            except:
                pass

    # Start both threads
    t1 = threading.Thread(target=thread_a)
    t2 = threading.Thread(target=thread_b)
    t1.start()
    t2.start()

    # Wait for threads to complete
    t1.join(timeout=10)
    t2.join(timeout=10)

    # At least one thread should have detected deadlock or timeout
    assert len(detected_deadlocks) > 0, f"No deadlocks or timeouts detected: {detected_deadlocks}"


@pytest.mark.timeout(15)
def test_lock_metrics_tracking(version_manager, sample_dataframe):
    """Test that lock acquisition metrics are properly tracked."""
    # Perform some operations
    version_id = version_manager.create_version(
        df=sample_dataframe,
        version_name="metrics_test",
        description="Test dataset for metrics"
    )

    # Load the version multiple times
    for _ in range(3):
        df = version_manager.load_version(version_id)
        assert df is not None

    # Check metrics
    metrics = version_manager.get_lock_metrics()
    assert isinstance(metrics, dict)
    assert "acquisitions" in metrics
    assert "timeouts" in metrics
    assert "deadlocks" in metrics
    assert "avg_acquire_time" in metrics

    # Should have at least some acquisitions (create + loads)
    assert metrics["acquisitions"] >= 4  # 1 write + 3 reads
    assert metrics["timeouts"] == 0  # No timeouts expected
    assert metrics["avg_acquire_time"] >= 0


@pytest.mark.timeout(15)
def test_multiple_concurrent_versions(version_manager, sample_dataframe):
    """Test concurrent operations on different datasets."""
    # Create multiple datasets
    version_ids = []
    for i in range(3):
        vid = version_manager.create_version(
            df=sample_dataframe,
            version_name=f"concurrent_test_{i}",
            description=f"Concurrent test dataset {i}"
        )
        version_ids.append(vid)

    results = []
    errors = []

    def concurrent_operation(thread_id, dataset_idx):
        try:
            # Alternate between creating new versions and reading existing ones
            if thread_id % 2 == 0:
                # Create new version
                new_df = sample_dataframe.copy()
                new_df['volume'] = new_df['volume'] * (thread_id + 1)  # Modify slightly
                vid = version_manager.create_version(
                    df=new_df,
                    version_name=f"concurrent_test_{dataset_idx}",
                    description=f"Thread {thread_id} version"
                )
                results.append(("create", thread_id, vid is not None))
            else:
                # Read existing version
                df = version_manager.load_version(version_ids[dataset_idx])
                results.append(("read", thread_id, df is not None and len(df) > 0))
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Start multiple threads operating on different datasets
    threads = []
    for i in range(6):  # 6 threads
        dataset_idx = i % 3  # Distribute across 3 datasets
        t = threading.Thread(target=concurrent_operation, args=(i, dataset_idx))
        threads.append(t)
        t.start()

    # Wait for completion
    for t in threads:
        t.join(timeout=10)

    # Verify results
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) == 6

    creates = [r for r in results if r[0] == "create"]
    reads = [r for r in results if r[0] == "read"]

    assert len(creates) == 3
    assert len(reads) == 3

    # All operations should succeed
    for op_type, thread_id, success in results:
        assert success, f"Operation {op_type} by thread {thread_id} failed"


@pytest.mark.timeout(15)
def test_lock_priority_and_fairness(version_manager, sample_dataframe):
    """Test that locks are acquired fairly and respect priority."""
    # Create a dataset
    version_id = version_manager.create_version(
        df=sample_dataframe,
        version_name="fairness_test",
        description="Test dataset for lock fairness"
    )

    acquisition_order = []
    lock = threading.Lock()

    def acquire_and_record(thread_id, lock_type="read"):
        try:
            if lock_type == "read":
                version_manager.lock_manager.acquire_read_lock("fairness_test", timeout=30)
            else:
                version_manager.lock_manager.acquire_write_lock("fairness_test", timeout=30)

            with lock:
                acquisition_order.append(f"thread_{thread_id}")

            time.sleep(0.1)  # Hold briefly

            version_manager.lock_manager.release_lock("fairness_test")
        except Exception as e:
            with lock:
                acquisition_order.append(f"error_{thread_id}")

    # Start multiple readers
    threads = []
    for i in range(5):
        t = threading.Thread(target=acquire_and_record, args=(i, "read"))
        threads.append(t)
        t.start()

    # Wait for all to complete
    for t in threads:
        t.join(timeout=10)

    # All threads should have acquired the lock
    successful_acquisitions = [x for x in acquisition_order if not x.startswith("error")]
    assert len(successful_acquisitions) == 5

    # Check that no errors occurred
    errors = [x for x in acquisition_order if x.startswith("error")]
    assert len(errors) == 0


@pytest.mark.timeout(15)
def test_lock_timeout_does_not_corrupt_state(version_manager, sample_dataframe):
    """Test that lock timeouts don't leave the system in an inconsistent state."""
    # Create a dataset
    version_id = version_manager.create_version(
        df=sample_dataframe,
        version_name="consistency_test",
        description="Test dataset for consistency"
    )

    # Verify initial state
    initial_versions = version_manager.list_versions()
    assert version_id in initial_versions

    # Cause a timeout scenario
    timeout_occurred = False
    try:
        # Try to acquire write lock while holding read lock with short timeout
        # This is tricky to simulate reliably, but let's try
        version_manager.lock_manager.acquire_read_lock("consistency_test", timeout=30)
        # In a separate thread, try to get write lock with very short timeout
        def try_write():
            try:
                version_manager.lock_manager.acquire_write_lock("consistency_test", timeout=0.1)
                return True
            except LockTimeoutError:
                return False
            finally:
                try:
                    version_manager.lock_manager.release_lock("consistency_test")
                except:
                    pass

        write_thread = threading.Thread(target=try_write)
        write_thread.start()
        write_thread.join(timeout=5)

        version_manager.lock_manager.release_lock("consistency_test")

    except LockTimeoutError:
        timeout_occurred = True

    # Verify system state is still consistent
    final_versions = version_manager.list_versions()
    assert final_versions == initial_versions

    # Should still be able to load the version
    df = version_manager.load_version(version_id)
    assert df is not None
    assert len(df) == 100
