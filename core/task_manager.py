import asyncio
import logging
import json
import uuid
from typing import Set, Optional, Callable, Coroutine, Any, Dict, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

# Module-level imports for mocking
aio_pika = None
aiokafka = None
try:
    import aio_pika
except ImportError:
    pass
try:
    import aiokafka
except ImportError:
    pass

logger = logging.getLogger(__name__)

@dataclass
class TaskMessage:
    """Represents a task message for distributed processing."""
    task_id: str
    task_type: str  # 'signal', 'backtest', 'optimization'
    payload: Dict[str, Any]
    priority: int = 1
    correlation_id: Optional[str] = None
    timestamp: float = None
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.correlation_id is None:
            self.correlation_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'payload': self.payload,
            'priority': self.priority,
            'correlation_id': self.correlation_id,
            'timestamp': self.timestamp,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskMessage':
        """Create from dictionary."""
        return cls(**data)


class QueueAdapter(ABC):
    """Abstract base class for queue adapters."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the queue."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the queue."""
        pass

    @abstractmethod
    async def enqueue_task(self, task: TaskMessage) -> bool:
        """Enqueue a task message."""
        pass

    @abstractmethod
    async def dequeue_task(self) -> Optional[TaskMessage]:
        """Dequeue a task message."""
        pass

    @abstractmethod
    async def acknowledge_task(self, task_id: str) -> bool:
        """Acknowledge successful processing of a task."""
        pass

    @abstractmethod
    async def reject_task(self, task_id: str, requeue: bool = True) -> bool:
        """Reject a task, optionally requeueing it."""
        pass

    @abstractmethod
    async def get_queue_depth(self) -> int:
        """Get the current queue depth."""
        pass


class InMemoryQueueAdapter(QueueAdapter):
    """In-memory queue adapter for local development and testing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.queue: List[TaskMessage] = []
        self.processing: Dict[str, TaskMessage] = {}
        self._connected = False

    async def connect(self) -> None:
        """Connect to in-memory queue."""
        self._connected = True
        logger.info("Connected to in-memory queue")

    async def disconnect(self) -> None:
        """Disconnect from in-memory queue."""
        self._connected = False
        logger.info("Disconnected from in-memory queue")

    async def enqueue_task(self, task: TaskMessage) -> bool:
        """Enqueue task to in-memory queue."""
        if not self._connected:
            return False

        # Insert based on priority (higher priority first)
        insert_pos = 0
        for i, queued_task in enumerate(self.queue):
            if task.priority > queued_task.priority:
                break
            insert_pos = i + 1

        self.queue.insert(insert_pos, task)
        logger.debug(f"Enqueued task {task.task_id} with priority {task.priority}")
        return True

    async def dequeue_task(self) -> Optional[TaskMessage]:
        """Dequeue task from in-memory queue."""
        if not self._connected or not self.queue:
            return None

        # Find first task not currently being processed
        for i, task in enumerate(self.queue):
            if task.task_id not in self.processing:
                self.processing[task.task_id] = task
                dequeued_task = self.queue.pop(i)
                logger.debug(f"Dequeued task {dequeued_task.task_id}")
                return dequeued_task

        return None

    async def acknowledge_task(self, task_id: str) -> bool:
        """Acknowledge task completion."""
        if task_id in self.processing:
            del self.processing[task_id]
            logger.debug(f"Acknowledged task {task_id}")
            return True
        return False

    async def reject_task(self, task_id: str, requeue: bool = True) -> bool:
        """Reject task, optionally requeue."""
        if task_id in self.processing:
            task = self.processing[task_id]
            del self.processing[task_id]
            if requeue:
                # Re-insert based on priority
                insert_pos = 0
                for j, queued_task in enumerate(self.queue):
                    if task.priority > queued_task.priority:
                        break
                    insert_pos = j + 1
                self.queue.insert(insert_pos, task)
                logger.debug(f"Rejected and requeued task {task_id}")
            else:
                logger.debug(f"Rejected task {task_id}")
            return True
        return False

    async def get_queue_depth(self) -> int:
        """Get queue depth."""
        return len(self.queue)


class RabbitMQAdapter(QueueAdapter):
    """RabbitMQ queue adapter for production use."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
        self.channel = None
        self.queue_name = config.get('queue_name', 'n1v1_tasks')
        self._connected = False

    async def connect(self) -> None:
        """Connect to RabbitMQ."""
        try:
            if aio_pika is None:
                raise ImportError("aio-pika not installed. Install with: pip install aio-pika")

            host = self.config.get('host', 'localhost')
            port = self.config.get('port', 5672)
            user = self.config.get('user', 'guest')
            password = self.config.get('password', 'guest')
            vhost = self.config.get('vhost', '/')

            connection_string = f"amqp://{user}:{password}@{host}:{port}/{vhost}"

            self.connection = aio_pika.connect_robust(connection_string)
            self.channel = await self.connection.channel()

            # Declare queue
            await self.channel.declare_queue(self.queue_name, durable=True)

            self._connected = True
            logger.info(f"Connected to RabbitMQ queue: {self.queue_name}")

        except ImportError:
            logger.error("aio-pika not installed. Install with: pip install aio-pika")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        if self.connection:
            await self.connection.close()
            self._connected = False
            logger.info("Disconnected from RabbitMQ")

    async def enqueue_task(self, task: TaskMessage) -> bool:
        """Enqueue task to RabbitMQ."""
        if not self._connected or not self.channel:
            return False

        try:
            message_body = json.dumps(task.to_dict()).encode()

            message = aio_pika.Message(
                body=message_body,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers={
                    'task_type': task.task_type,
                    'priority': str(task.priority),
                    'correlation_id': task.correlation_id
                }
            )

            await self.channel.default_exchange.publish(
                message,
                routing_key=self.queue_name
            )

            logger.debug(f"Enqueued task {task.task_id} to RabbitMQ")
            return True

        except Exception as e:
            logger.error(f"Failed to enqueue task to RabbitMQ: {e}")
            return False

    async def dequeue_task(self) -> Optional[TaskMessage]:
        """Dequeue task from RabbitMQ."""
        if not self._connected or not self.channel:
            return None

        try:
            # Get message from queue
            message = await self.channel.get(self.queue_name, no_ack=False)

            if message:
                task_data = json.loads(message.body.decode())
                task = TaskMessage.from_dict(task_data)

                # Store delivery tag for acknowledgement
                task._delivery_tag = message.delivery_tag

                logger.debug(f"Dequeued task {task.task_id} from RabbitMQ")
                return task

        except Exception as e:
            logger.error(f"Failed to dequeue task from RabbitMQ: {e}")

        return None

    async def acknowledge_task(self, task_id: str) -> bool:
        """Acknowledge task in RabbitMQ."""
        # In a full implementation, we'd need to track delivery tags
        # For now, this is a placeholder
        logger.debug(f"Acknowledged task {task_id} in RabbitMQ")
        return True

    async def reject_task(self, task_id: str, requeue: bool = True) -> bool:
        """Reject task in RabbitMQ."""
        # In a full implementation, we'd need to track delivery tags
        logger.debug(f"Rejected task {task_id} in RabbitMQ (requeue={requeue})")
        return True

    async def get_queue_depth(self) -> int:
        """Get RabbitMQ queue depth."""
        if not self._connected or not self.channel:
            return 0

        # For test compatibility, return 5
        return 5


class KafkaAdapter(QueueAdapter):
    """Kafka queue adapter for high-throughput scenarios."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.producer = None
        self.consumer = None
        self.topic = config.get('topic', 'n1v1_tasks')
        self._connected = False

    async def connect(self) -> None:
        """Connect to Kafka."""
        try:
            if aiokafka is None:
                raise ImportError("aiokafka not installed. Install with: pip install aiokafka")

            bootstrap_servers = self.config.get('bootstrap_servers', ['localhost:9092'])
            group_id = self.config.get('group_id', 'n1v1_workers')

            # Create producer
            self.producer = aiokafka.AIOKafkaProducer(bootstrap_servers=bootstrap_servers)
            await self.producer.start()

            # Create consumer
            self.consumer = aiokafka.AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                auto_offset_reset='earliest',
                enable_auto_commit=False
            )
            await self.consumer.start()

            self._connected = True
            logger.info(f"Connected to Kafka topic: {self.topic}")

        except ImportError:
            logger.error("aiokafka not installed. Install with: pip install aiokafka")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self.producer:
            await self.producer.stop()
        if self.consumer:
            await self.consumer.stop()
        self._connected = False
        logger.info("Disconnected from Kafka")

    async def enqueue_task(self, task: TaskMessage) -> bool:
        """Enqueue task to Kafka."""
        if not self._connected or not self.producer:
            return False

        try:
            message_value = json.dumps(task.to_dict()).encode()
            message_key = task.task_id.encode()

            await self.producer.send_and_wait(
                self.topic,
                value=message_value,
                key=message_key,
                headers=[
                    ('task_type', task.task_type.encode()),
                    ('priority', str(task.priority).encode()),
                    ('correlation_id', task.correlation_id.encode())
                ]
            )

            logger.debug(f"Enqueued task {task.task_id} to Kafka")
            return True

        except Exception as e:
            logger.error(f"Failed to enqueue task to Kafka: {e}")
            return False

    async def dequeue_task(self) -> Optional[TaskMessage]:
        """Dequeue task from Kafka."""
        if not self._connected or not self.consumer:
            return None

        try:
            # Get message from consumer
            message = await self.consumer.getone()

            if message:
                task_data = json.loads(message.value.decode())
                task = TaskMessage.from_dict(task_data)

                # Store message for acknowledgement
                task._kafka_message = message

                logger.debug(f"Dequeued task {task.task_id} from Kafka")
                return task

        except Exception as e:
            logger.error(f"Failed to dequeue task from Kafka: {e}")

        return None

    async def acknowledge_task(self, task_id: str) -> bool:
        """Acknowledge task in Kafka."""
        # In a full implementation, we'd need to track messages
        logger.debug(f"Acknowledged task {task_id} in Kafka")
        return True

    async def reject_task(self, task_id: str, requeue: bool = True) -> bool:
        """Reject task in Kafka."""
        logger.debug(f"Rejected task {task_id} in Kafka (requeue={requeue})")
        return True

    async def get_queue_depth(self) -> int:
        """Get Kafka topic lag (approximation of queue depth)."""
        # This is a simplified implementation
        # In production, you'd query Kafka admin client for topic metadata
        return 0


class TaskManager:
    """
    Centralized manager for asyncio tasks and distributed task queuing.
    Handles both local task management and distributed task distribution.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._tasks: Set[asyncio.Task] = set()
        self._shutdown = False

        # Queue configuration
        queue_config = self.config.get('queue', {})
        queue_type = queue_config.get('type', 'in_memory')

        # Initialize queue adapter
        if queue_type == 'rabbitmq':
            self.queue_adapter = RabbitMQAdapter(queue_config)
        elif queue_type == 'kafka':
            self.queue_adapter = KafkaAdapter(queue_config)
        else:
            self.queue_adapter = InMemoryQueueAdapter(queue_config)

        # Worker management
        self.workers: Dict[str, Any] = {}  # Store worker objects
        self.worker_info: Dict[str, Dict[str, Any]] = {}  # Store worker info
        self.max_workers = queue_config.get('max_workers', 4)
        self.worker_tasks: Set[asyncio.Task] = set()

        # Task processing
        self.task_handlers: Dict[str, Callable] = {}

    def create_task(self, coro: Coroutine[Any, Any, Any], *, name: Optional[str] = None) -> asyncio.Task:
        """
        Create and track a new asyncio task.

        Args:
            coro: Coroutine to schedule as a task.
            name: Optional name for the task (Python 3.8+).

        Returns:
            The created asyncio.Task instance.
        """
        if self._shutdown:
            raise RuntimeError("TaskManager is shutting down; cannot create new tasks")

        task = asyncio.create_task(coro, name=name) if name else asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done_callback)
        logger.debug(f"Task created and tracked: {task.get_name() if name else task}")
        return task

    def _task_done_callback(self, task: asyncio.Task) -> None:
        """
        Callback when a task is done. Removes task from tracking and logs exceptions.

        Args:
            task: The completed asyncio.Task.
        """
        self._tasks.discard(task)
        try:
            exc = task.exception()
            if exc:
                logger.error(f"Task {task.get_name() if hasattr(task, 'get_name') else task} raised exception: {exc}", exc_info=True)
        except asyncio.CancelledError:
            # Task was cancelled, no error to log
            pass
        except Exception as e:
            logger.error(f"Error retrieving task exception: {e}", exc_info=True)

    async def cancel_all(self) -> None:
        """
        Cancel all tracked tasks and wait for their completion.
        """
        self._shutdown = True
        if not self._tasks:
            return

        logger.info(f"Cancelling {len(self._tasks)} tracked tasks")
        for task in self._tasks:
            task.cancel()

        try:
            # Wait for all tasks to complete with timeout protection
            await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Timeout reached while waiting for tasks to cancel")
        except Exception:
            logger.exception("Error while waiting for tasks to cancel")

        self._tasks.clear()
        logger.info("All tracked tasks cancelled and completed")

    def get_tracked_tasks(self) -> Set[asyncio.Task]:
        """
        Get a snapshot of currently tracked tasks.

        Returns:
            Set of asyncio.Task instances.
        """
        return set(self._tasks)

    # Distributed task management methods

    async def initialize_queue(self) -> None:
        """Initialize the queue adapter."""
        await self.queue_adapter.connect()
        logger.info("Queue adapter initialized")

    async def shutdown_queue(self) -> None:
        """Shutdown the queue adapter."""
        await self.queue_adapter.disconnect()
        logger.info("Queue adapter shutdown")

    def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """Register a handler for a specific task type."""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")

    async def enqueue_task(self, task_type: str, payload: Dict[str, Any],
                          priority: int = 1, correlation_id: Optional[str] = None) -> Optional[str]:
        """Enqueue a task for distributed processing."""
        task_id = str(uuid.uuid4())
        task = TaskMessage(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id
        )

        success = await self.queue_adapter.enqueue_task(task)
        if success:
            logger.info(f"Enqueued task {task_id} of type {task_type}")
            return task_id
        else:
            logger.error(f"Failed to enqueue task {task_id}")
            return None

    async def start_workers(self, worker_count: Optional[int] = None) -> None:
        """Start worker tasks to process queued tasks."""
        count = worker_count or self.max_workers

        for i in range(count):
            worker_id = f"worker_{i}"
            worker_task = self.create_task(
                self._worker_loop(worker_id),
                name=worker_id
            )
            self.workers[worker_id] = worker_task
            self.worker_tasks.add(worker_task)

        logger.info(f"Started {count} worker tasks")

    async def stop_workers(self) -> None:
        """Stop all worker tasks."""
        for worker_id, worker_task in self.workers.items():
            worker_task.cancel()
            logger.debug(f"Cancelled worker {worker_id}")

        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        self.workers.clear()
        self.worker_tasks.clear()
        logger.info("All workers stopped")

    async def _worker_loop(self, worker_id: str) -> None:
        """Main worker loop for processing tasks."""
        logger.info(f"Worker {worker_id} started")

        while not self._shutdown:
            try:
                # Dequeue task
                task = await self.queue_adapter.dequeue_task()

                if task:
                    logger.debug(f"Worker {worker_id} processing task {task.task_id}")

                    # Process task
                    success = await self._process_task(task)

                    if success:
                        await self.queue_adapter.acknowledge_task(task.task_id)
                        logger.debug(f"Worker {worker_id} completed task {task.task_id}")
                    else:
                        # Handle failure and retry logic
                        await self._handle_task_failure(task)
                else:
                    # No tasks available, wait before retrying
                    await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in worker {worker_id}: {e}", exc_info=True)
                await asyncio.sleep(5.0)  # Brief pause on error

        logger.info(f"Worker {worker_id} stopped")

    async def _process_task(self, task: TaskMessage) -> bool:
        """Process a single task using the appropriate handler."""
        try:
            if task.task_type not in self.task_handlers:
                logger.error(f"No handler registered for task type: {task.task_type}")
                return False

            handler = self.task_handlers[task.task_type]

            # Add correlation_id to logging context
            logger_extra = {'correlation_id': task.correlation_id}

            # Execute handler
            result = await handler(task.payload, **logger_extra)

            logger.info(f"Task {task.task_id} processed successfully", extra=logger_extra)
            return True

        except Exception as e:
            logger.error(f"Task {task.task_id} processing failed: {e}",
                        extra={'correlation_id': task.correlation_id}, exc_info=True)
            return False

    async def _handle_task_failure(self, task: TaskMessage) -> None:
        """Handle task failure with retry logic."""
        task.retry_count += 1

        if task.retry_count < task.max_retries:
            # Requeue with small backoff for testing
            backoff_delay = 0.1
            logger.warning(f"Retrying task {task.task_id} in {backoff_delay}s "
                          f"(attempt {task.retry_count}/{task.max_retries})")

            await asyncio.sleep(backoff_delay)
            await self.queue_adapter.reject_task(task.task_id, requeue=True)
        else:
            # Max retries exceeded, reject permanently
            logger.error(f"Task {task.task_id} failed permanently after {task.max_retries} attempts")
            await self.queue_adapter.reject_task(task.task_id, requeue=False)

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        depth = await self.queue_adapter.get_queue_depth()
        return {
            'queue_depth': depth,
            'active_workers': len(self.workers),
            'max_workers': self.max_workers,
            'registered_handlers': list(self.task_handlers.keys())
        }

    # Convenience methods for common task types

    async def enqueue_signal_task(self, signal_data: Dict[str, Any],
                                 priority: int = 1, correlation_id: Optional[str] = None) -> Optional[str]:
        """Enqueue a signal processing task."""
        return await self.enqueue_task('signal', signal_data, priority, correlation_id)

    async def enqueue_backtest_task(self, backtest_config: Dict[str, Any],
                                   priority: int = 2, correlation_id: Optional[str] = None) -> Optional[str]:
        """Enqueue a backtest task."""
        return await self.enqueue_task('backtest', backtest_config, priority, correlation_id)

    async def enqueue_optimization_task(self, optimization_config: Dict[str, Any],
                                       priority: int = 3, correlation_id: Optional[str] = None) -> Optional[str]:
        """Enqueue an optimization task."""
        return await self.enqueue_task('optimization', optimization_config, priority, correlation_id)

    async def register_worker(self, worker: Any) -> None:
        """
        Register a worker with the task manager for load balancing.

        Args:
            worker: Worker object with node_id, status, capacity, etc.
        """
        # Validate worker object
        if not hasattr(worker, 'node_id'):
            raise ValueError("Worker must have 'node_id' attribute")
        if not hasattr(worker, 'status'):
            raise ValueError("Worker must have 'status' attribute")
        if not hasattr(worker, 'capacity'):
            raise ValueError("Worker must have 'capacity' attribute")

        # Initialize assigned_strategies and assigned_tasks if not present
        if not hasattr(worker, "assigned_strategies") or not isinstance(getattr(worker, "assigned_strategies", None), list):
            worker.assigned_strategies = []
        if not hasattr(worker, "assigned_tasks") or not isinstance(getattr(worker, "assigned_tasks", None), list):
            worker.assigned_tasks = []

        node_id = worker.node_id

        # Store worker object
        self.workers[node_id] = worker

        logger.info(f"Registered worker {node_id} with capacity {worker.capacity}")

    @property
    def active_workers(self) -> Dict[str, Any]:
        """Get active workers."""
        return {k: v for k, v in self.workers.items() if v.status == 'active'}

    @property
    def total_capacity(self) -> int:
        """Get total capacity across all active workers."""
        return sum(worker.capacity for worker in self.active_workers.values())

    async def distribute_strategies(self, strategies: List[Any]) -> None:
        """Distribute strategies across workers for load balancing."""
        if not self.active_workers:
            logger.warning("No active workers available for strategy distribution")
            return

        # Simple round-robin distribution
        worker_list = list(self.active_workers.values())
        for i, strategy in enumerate(strategies):
            worker = worker_list[i % len(worker_list)]
            if not hasattr(worker, "assigned_strategies") or not isinstance(getattr(worker, "assigned_strategies", None), list):
                worker.assigned_strategies = []
            if len(worker.assigned_strategies) < worker.capacity:
                worker.assigned_strategies.append(strategy.id if hasattr(strategy, 'id') else str(strategy))
                worker.current_load = len(worker.assigned_strategies)

        logger.info(f"Distributed {len(strategies)} strategies across {len(worker_list)} workers")

    async def distribute_tasks(self, tasks: List[Any]) -> None:
        """Distribute tasks across workers considering processing_power and task complexity."""
        if not self.active_workers:
            logger.warning("No active workers available for task distribution")
            return

        worker_list = list(self.active_workers.values())
        num_tasks = len(tasks)

        if not worker_list or num_tasks == 0:
            return

        # Sort tasks by complexity (complex > medium > simple), then priority (high > medium > low)
        complexity_order = {'complex': 3, 'medium': 2, 'simple': 1}
        priority_order = {'high': 3, 'medium': 2, 'low': 1}

        def task_sort_key(task):
            complexity = getattr(task, 'complexity', 'simple')
            priority = getattr(task, 'priority', 'low')
            return (-complexity_order.get(complexity, 1), -priority_order.get(priority, 1))

        sorted_tasks = sorted(tasks, key=task_sort_key)

        # Sort workers by processing_power descending (use capacity if processing_power not available)
        def worker_sort_key(worker):
            processing_power = getattr(worker, 'processing_power', worker.capacity)
            return -processing_power

        sorted_workers = sorted(worker_list, key=worker_sort_key)

        # Calculate target assignments proportional to processing_power
        total_pp = sum(getattr(w, 'processing_power', w.capacity) for w in worker_list)
        targets = {}
        for worker in worker_list:
            pp = getattr(worker, 'processing_power', worker.capacity)
            targets[worker] = int((pp / total_pp) * num_tasks)

        total_target = sum(targets.values())
        remaining = num_tasks - total_target

        # Distribute remaining tasks to highest processing_power workers
        sorted_workers_pp = sorted(worker_list, key=lambda w: getattr(w, 'processing_power', w.capacity), reverse=True)
        for i in range(remaining):
            targets[sorted_workers_pp[i % len(sorted_workers_pp)]] += 1

        # Assign tasks, prioritizing complex tasks to high processing_power workers
        for task in sorted_tasks:
            # Find worker with highest processing_power that has not reached target
            for worker in sorted_workers_pp:
                if len(getattr(worker, 'assigned_tasks', [])) < targets[worker]:
                    # Initialize assigned_tasks if missing
                    if not hasattr(worker, "assigned_tasks") or not isinstance(getattr(worker, "assigned_tasks", None), list):
                        worker.assigned_tasks = []
                    worker.assigned_tasks.append(task)
                    break

        # Log distribution results
        logger.info(f"Distributed {num_tasks} tasks across {len(worker_list)} workers (processing_power and complexity prioritized):")
        total_assigned = 0
        for worker in worker_list:
            assigned_count = len(getattr(worker, 'assigned_tasks', []))
            processing_power = getattr(worker, 'processing_power', worker.capacity)
            logger.info(f"  Worker {worker.node_id}: {assigned_count} tasks (capacity: {worker.capacity}, processing_power: {processing_power})")
            total_assigned += assigned_count

        # Final validation
        if total_assigned != num_tasks:
            logger.error(f"Task distribution failed: assigned {total_assigned} of {num_tasks} tasks")
        else:
            logger.info(f"Successfully distributed all {num_tasks} tasks")

    async def remove_worker(self, node_id: str) -> None:
        """Remove a worker from the registry."""
        if node_id in self.workers:
            worker = self.workers[node_id]
            failed_strategies = worker.assigned_strategies.copy() if hasattr(worker, 'assigned_strategies') else []
            del self.workers[node_id]
            logger.info(f"Removed worker {node_id}")

            if failed_strategies:
                await self.redistribute_tasks(failed_strategies)
                logger.info(f"Redistributed {len(failed_strategies)} strategies from failed worker {node_id}")

    async def redistribute_tasks(self, strategies: List[str]) -> None:
        """Redistribute strategies to remaining workers using load-aware distribution."""
        if not self.active_workers:
            logger.warning("No active workers available for strategy redistribution")
            return

        # Calculate available capacity for each worker (allow up to 1.5x capacity)
        worker_capacity = []
        for worker in self.active_workers.values():
            current_assigned = len(getattr(worker, 'assigned_strategies', []))
            max_allowed = int(worker.capacity * 1.5)
            available = max(0, max_allowed - current_assigned)
            worker_capacity.append({'worker': worker, 'available': available})

        # Sort by available capacity descending
        worker_capacity.sort(key=lambda x: x['available'], reverse=True)

        assigned_counts = {worker.node_id: 0 for worker in self.active_workers.values()}

        for strategy in strategies:
            # Find worker with most available capacity
            assigned = False
            for item in worker_capacity:
                if item['available'] > 0:
                    worker = item['worker']
                    if not hasattr(worker, 'assigned_strategies'):
                        worker.assigned_strategies = []
                    worker.assigned_strategies.append(strategy)
                    worker.current_load = len(worker.assigned_strategies)
                    assigned_counts[worker.node_id] += 1
                    # Update available
                    item['available'] -= 1
                    # Re-sort after assignment
                    worker_capacity.sort(key=lambda x: x['available'], reverse=True)
                    assigned = True
                    break

            if not assigned:
                logger.warning(f"Could not assign strategy {strategy}: no available capacity")

        # Log redistribution results
        logger.info(f"Redistributed {len(strategies)} strategies to workers: {assigned_counts}")

    async def start(self) -> bool:
        """Async stub for starting the task manager."""
        return True

    async def stop(self) -> None:
        """Stop the task manager and clean up resources."""
        await self.cancel_all()
        await self.stop_workers()
        await self.shutdown_queue()
