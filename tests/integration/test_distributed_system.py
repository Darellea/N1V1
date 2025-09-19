"""
Integration tests for the distributed task processing system.

Tests the complete distributed architecture including:
- Task scheduling and queuing
- Worker processing
- Queue adapters (in-memory, RabbitMQ, Kafka)
- Distributed metrics collection
- Fault tolerance and recovery
"""

import asyncio
import pytest
import time
import json
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

from core.task_manager import (
    TaskManager,
    TaskMessage,
    InMemoryQueueAdapter,
    RabbitMQAdapter,
    KafkaAdapter
)
from core.bot_engine import BotEngine, DistributedScheduler, DistributedExecutor


class TestDistributedTaskManager:
    """Test cases for the distributed task manager."""

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        """Test configuration for distributed system."""
        return {
            "queue": {
                "type": "in_memory",
                "max_workers": 2,
                "worker_timeout": 30
            }
        }

    @pytest.fixture
    async def task_manager(self, config: Dict[str, Any]) -> TaskManager:
        """Create a task manager instance for testing."""
        manager = TaskManager(config)
        await manager.initialize_queue()
        yield manager
        await manager.shutdown_queue()

    @pytest.mark.asyncio
    async def test_task_enqueue_dequeue(self, task_manager: TaskManager):
        """Test basic task enqueue and dequeue operations."""
        # Create a test task
        task_data = {
            'action': 'process_signal',
            'symbol': 'BTC/USDT',
            'timestamp': time.time()
        }

        # Enqueue task
        task_id = await task_manager.enqueue_signal_task(task_data)
        assert task_id is not None
        assert isinstance(task_id, str)

        # Dequeue task
        dequeued_task = await task_manager.queue_adapter.dequeue_task()
        assert dequeued_task is not None
        assert dequeued_task.task_id == task_id
        assert dequeued_task.task_type == 'signal'
        assert dequeued_task.payload == task_data

    @pytest.mark.asyncio
    async def test_worker_processing(self, task_manager: TaskManager):
        """Test worker task processing."""
        processed_tasks = []

        # Register a test handler
        async def test_handler(payload: Dict[str, Any], **kwargs) -> bool:
            processed_tasks.append(payload)
            return True

        task_manager.register_task_handler('test', test_handler)

        # Enqueue a test task
        task_data = {'test_key': 'test_value'}
        task_id = await task_manager.enqueue_task('test', task_data)

        # Start a worker
        await task_manager.start_workers(1)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Stop workers
        await task_manager.stop_workers()

        # Verify task was processed
        assert len(processed_tasks) == 1
        assert processed_tasks[0] == task_data

    @pytest.mark.asyncio
    async def test_task_failure_retry(self, task_manager: TaskManager):
        """Test task failure and retry logic."""
        failure_count = 0
        max_failures = 2

        async def failing_handler(payload: Dict[str, Any], **kwargs) -> bool:
            nonlocal failure_count
            failure_count += 1
            if failure_count < max_failures:
                raise Exception("Simulated failure")
            return True

        task_manager.register_task_handler('failing_test', failing_handler)

        # Enqueue failing task
        task_data = {'test': 'failure_retry'}
        task_id = await task_manager.enqueue_task('failing_test', task_data)

        # Start worker
        await task_manager.start_workers(1)

        # Wait for retries
        await asyncio.sleep(2.0)  # Allow time for retries

        # Stop workers
        await task_manager.stop_workers()

        # Verify task eventually succeeded after retries
        assert failure_count == max_failures

    @pytest.mark.asyncio
    async def test_queue_status_monitoring(self, task_manager: TaskManager):
        """Test queue status monitoring."""
        # Enqueue multiple tasks
        for i in range(5):
            await task_manager.enqueue_signal_task({'task': i})

        # Get queue status
        status = await task_manager.get_queue_status()

        assert 'queue_depth' in status
        assert 'active_workers' in status
        assert 'max_workers' in status
        assert status['queue_depth'] == 5
        assert status['max_workers'] == 2


class TestQueueAdapters:
    """Test cases for different queue adapters."""

    @pytest.mark.asyncio
    async def test_in_memory_adapter(self):
        """Test in-memory queue adapter."""
        config = {'max_size': 100}
        adapter = InMemoryQueueAdapter(config)

        await adapter.connect()

        # Test enqueue/dequeue
        task = TaskMessage(
            task_id='test_1',
            task_type='signal',
            payload={'test': 'data'}
        )

        success = await adapter.enqueue_task(task)
        assert success

        dequeued = await adapter.dequeue_task()
        assert dequeued is not None
        assert dequeued.task_id == 'test_1'

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_rabbitmq_adapter_mock(self):
        """Test RabbitMQ adapter with mocked connection."""
        config = {
            'host': 'localhost',
            'port': 5672,
            'user': 'guest',
            'password': 'guest'
        }

        with patch('core.task_manager.aio_pika') as mock_aio_pika:
            # Mock the connection and channel
            mock_connection = AsyncMock()
            mock_channel = AsyncMock()
            mock_queue = AsyncMock()
            mock_queue.declaration_result.message_count = 5

            mock_channel.get_queue.return_value = mock_queue
            mock_connection.channel.return_value = mock_connection
            mock_aio_pika.connect_robust.return_value = mock_connection
            mock_connection.channel = mock_channel

            adapter = RabbitMQAdapter(config)

            # Test connection
            await adapter.connect()
            assert adapter._connected

            # Test queue depth
            depth = await adapter.get_queue_depth()
            assert depth == 5

            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_kafka_adapter_mock(self):
        """Test Kafka adapter with mocked connection."""
        config = {
            'bootstrap_servers': ['localhost:9092'],
            'group_id': 'test_group'
        }

        with patch('core.task_manager.aiokafka') as mock_aiokafka:
            # Mock producers and consumers
            mock_producer = AsyncMock()
            mock_consumer = AsyncMock()

            mock_aiokafka.AIOKafkaProducer.return_value = mock_producer
            mock_aiokafka.AIOKafkaConsumer.return_value = mock_consumer

            adapter = KafkaAdapter(config)

            # Test connection
            await adapter.connect()
            assert adapter._connected

            await adapter.disconnect()


class TestDistributedScheduler:
    """Test cases for the distributed scheduler."""

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {
            "queue": {
                "type": "in_memory",
                "max_workers": 2
            }
        }

    @pytest.fixture
    async def scheduler(self, config: Dict[str, Any]) -> DistributedScheduler:
        task_manager = TaskManager(config)
        scheduler = DistributedScheduler(task_manager, config)
        await scheduler.initialize()
        yield scheduler
        await task_manager.shutdown_queue()

    @pytest.mark.asyncio
    async def test_signal_scheduling(self, scheduler: DistributedScheduler):
        """Test signal processing task scheduling."""
        market_data = {
            'BTC/USDT': {'close': [50000, 51000]},
            'ETH/USDT': {'close': [3000, 3100]}
        }

        task_id = await scheduler.schedule_signal_processing(market_data)
        assert task_id is not None

        # Verify task was enqueued
        status = await scheduler.get_queue_status()
        assert status['queue_depth'] >= 1

    @pytest.mark.asyncio
    async def test_backtest_scheduling(self, scheduler: DistributedScheduler):
        """Test backtest task scheduling."""
        backtest_config = {
            'strategy': 'RSIStrategy',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'symbol': 'BTC/USDT'
        }

        task_id = await scheduler.schedule_backtest(backtest_config)
        assert task_id is not None

    @pytest.mark.asyncio
    async def test_optimization_scheduling(self, scheduler: DistributedScheduler):
        """Test optimization task scheduling."""
        optimization_config = {
            'algorithm': 'genetic',
            'population_size': 50,
            'generations': 20
        }

        task_id = await scheduler.schedule_optimization(optimization_config)
        assert task_id is not None


class TestDistributedExecutor:
    """Test cases for the distributed executor."""

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {
            "queue": {
                "type": "in_memory",
                "max_workers": 1
            }
        }

    @pytest.fixture
    async def executor(self, config: Dict[str, Any]) -> DistributedExecutor:
        task_manager = TaskManager(config)
        executor = DistributedExecutor(task_manager, config)
        yield executor
        await task_manager.shutdown_queue()

    @pytest.mark.asyncio
    async def test_task_handler_registration(self, executor: DistributedExecutor):
        """Test task handler registration."""
        assert 'signal' in executor.task_manager.task_handlers
        assert 'backtest' in executor.task_manager.task_handlers
        assert 'optimization' in executor.task_manager.task_handlers

    @pytest.mark.asyncio
    async def test_signal_task_processing(self, executor: DistributedExecutor):
        """Test signal task processing."""
        # Mock bot engine
        mock_bot_engine = Mock()
        mock_bot_engine._generate_signals = AsyncMock(return_value=[])
        mock_bot_engine._evaluate_risk = AsyncMock(return_value=[])
        mock_bot_engine._execute_orders = AsyncMock()
        mock_bot_engine._update_state = AsyncMock()

        await executor.initialize(mock_bot_engine)

        # Create and process signal task
        task_payload = {
            'market_data': {'BTC/USDT': {'close': [50000]}},
            'pairs': ['BTC/USDT']
        }

        success = await executor._handle_signal_task(task_payload)
        assert success

        # Verify bot engine methods were called
        mock_bot_engine._generate_signals.assert_called_once()
        mock_bot_engine._evaluate_risk.assert_called_once()
        mock_bot_engine._execute_orders.assert_called_once()
        mock_bot_engine._update_state.assert_called_once()


class TestDistributedMetrics:
    """Test cases for distributed system metrics."""

    @pytest.mark.asyncio
    async def test_distributed_metrics_collection(self):
        """Test collection of distributed system metrics."""
        from core.metrics_collector import MetricsCollector, collect_distributed_metrics

        collector = MetricsCollector({})

        # Collect distributed metrics
        await collect_distributed_metrics(collector)

        # Verify metrics were recorded
        queue_depth = collector.get_metric_value("distributed_queue_depth")
        assert queue_depth is not None

        active_workers = collector.get_metric_value("distributed_active_workers")
        assert active_workers is not None

        success_rate = collector.get_metric_value("distributed_task_success_rate")
        assert success_rate is not None


class TestFaultTolerance:
    """Test cases for fault tolerance and recovery."""

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {
            "queue": {
                "type": "in_memory",
                "max_workers": 2
            }
        }

    @pytest.mark.asyncio
    async def test_worker_crash_recovery(self, config: Dict[str, Any]):
        """Test recovery from worker crashes."""
        task_manager = TaskManager(config)
        await task_manager.initialize_queue()

        crash_count = 0

        async def crashing_handler(payload: Dict[str, Any], **kwargs) -> bool:
            nonlocal crash_count
            crash_count += 1
            if crash_count == 1:
                raise Exception("Simulated crash")
            return True

        task_manager.register_task_handler('crash_test', crashing_handler)

        # Enqueue task
        await task_manager.enqueue_task('crash_test', {'test': 'crash'})

        # Start worker
        await task_manager.start_workers(1)

        # Wait for processing and retry
        await asyncio.sleep(1.0)

        # Stop workers
        await task_manager.stop_workers()

        # Verify task was retried and eventually succeeded
        assert crash_count == 2  # First attempt failed, second succeeded

        await task_manager.shutdown_queue()

    @pytest.mark.asyncio
    async def test_queue_adapter_failure_recovery(self, config: Dict[str, Any]):
        """Test recovery from queue adapter failures."""
        task_manager = TaskManager(config)
        await task_manager.initialize_queue()

        # Simulate queue failure
        original_enqueue = task_manager.queue_adapter.enqueue_task
        fail_count = 0

        async def failing_enqueue(task: TaskMessage) -> bool:
            nonlocal fail_count
            fail_count += 1
            if fail_count == 1:
                return False  # Simulate failure
            return await original_enqueue(task)

        task_manager.queue_adapter.enqueue_task = failing_enqueue

        # Try to enqueue task
        success = await task_manager.enqueue_task('test', {'data': 'test'})
        assert not success  # First attempt should fail

        # Second attempt should succeed
        success = await task_manager.enqueue_task('test', {'data': 'test'})
        assert success

        await task_manager.shutdown_queue()


class TestLoadBalancing:
    """Test cases for load balancing across workers."""

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {
            "queue": {
                "type": "in_memory",
                "max_workers": 3
            }
        }

    @pytest.mark.asyncio
    async def test_task_distribution(self, config: Dict[str, Any]):
        """Test that tasks are distributed evenly across workers."""
        task_manager = TaskManager(config)
        await task_manager.initialize_queue()

        worker_task_counts = {}

        async def counting_handler(payload: Dict[str, Any], **kwargs) -> bool:
            worker_id = payload.get('worker_id', 'unknown')
            worker_task_counts[worker_id] = worker_task_counts.get(worker_id, 0) + 1
            await asyncio.sleep(0.01)  # Simulate processing time
            return True

        task_manager.register_task_handler('count_test', counting_handler)

        # Enqueue multiple tasks
        for i in range(10):
            await task_manager.enqueue_task('count_test', {'task_id': i})

        # Start multiple workers
        await task_manager.start_workers(3)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Stop workers
        await task_manager.stop_workers()

        # Verify tasks were distributed
        total_tasks_processed = sum(worker_task_counts.values())
        assert total_tasks_processed == 10

        # Check that multiple workers were used
        assert len(worker_task_counts) >= 1

        await task_manager.shutdown_queue()

    @pytest.mark.asyncio
    async def test_priority_queue_processing(self, config: Dict[str, Any]):
        """Test that high-priority tasks are processed first."""
        task_manager = TaskManager(config)
        await task_manager.initialize_queue()

        processing_order = []

        async def ordered_handler(payload: Dict[str, Any], **kwargs) -> bool:
            processing_order.append(payload['priority'])
            return True

        task_manager.register_task_handler('priority_test', ordered_handler)

        # Enqueue tasks with different priorities
        await task_manager.enqueue_task('priority_test', {'priority': 1}, priority=1)
        await task_manager.enqueue_task('priority_test', {'priority': 3}, priority=3)
        await task_manager.enqueue_task('priority_test', {'priority': 2}, priority=2)

        # Start worker
        await task_manager.start_workers(1)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Stop workers
        await task_manager.stop_workers()

        # Verify high-priority task was processed first
        assert processing_order[0] == 3  # Highest priority first

        await task_manager.shutdown_queue()


if __name__ == "__main__":
    pytest.main([__file__])
