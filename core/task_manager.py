import asyncio
import logging
from typing import Set, Optional, Callable, Coroutine, Any

logger = logging.getLogger(__name__)

class TaskManager:
    """
    Centralized manager for asyncio tasks to ensure tracking, cancellation,
    and error handling for all background tasks.
    """

    def __init__(self) -> None:
        self._tasks: Set[asyncio.Task] = set()
        self._shutdown = False

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

        # Wait for all tasks to complete, ignoring exceptions
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("All tracked tasks cancelled and completed")

    def get_tracked_tasks(self) -> Set[asyncio.Task]:
        """
        Get a snapshot of currently tracked tasks.

        Returns:
            Set of asyncio.Task instances.
        """
        return set(self._tasks)
