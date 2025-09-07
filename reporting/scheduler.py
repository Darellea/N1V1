"""
Metrics Scheduler

Handles automatic scheduling of metrics generation and reporting.
Integrates with the framework's event system and cron-like scheduling.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import threading
import schedule

from .metrics import MetricsEngine, MetricsResult, get_metrics_engine
from .sync import get_dashboard_sync, sync_metrics_to_dashboards
from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


@dataclass
class ScheduledTask:
    """Represents a scheduled metrics task."""
    task_id: str
    name: str
    schedule_type: str  # 'session_end', 'daily', 'weekly', 'monthly'
    strategy_ids: List[str]
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True
    config: Optional[Dict[str, Any]] = None


class MetricsScheduler:
    """
    Scheduler for automatic metrics generation and reporting.

    Handles session-end metrics, daily/weekly aggregation, and
    integrates with dashboard synchronization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the metrics scheduler.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Components
        self.metrics_engine = get_metrics_engine()
        self.dashboard_sync = get_dashboard_sync()

        # Configuration
        self.enabled = self.config.get('enabled', True)
        self.session_end_enabled = self.config.get('session_end_enabled', True)
        self.daily_enabled = self.config.get('daily_enabled', True)
        self.weekly_enabled = self.config.get('weekly_enabled', True)

        # Task management
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # Scheduler thread
        self.scheduler_thread: Optional[threading.Thread] = None
        self.running = False

        # Initialize default tasks
        self._initialize_default_tasks()

        self.logger.info("MetricsScheduler initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'enabled': True,
            'session_end_enabled': True,
            'daily_enabled': True,
            'weekly_enabled': True,
            'daily_time': '02:00',  # 2 AM
            'weekly_day': 'monday',
            'weekly_time': '03:00',  # 3 AM on Mondays
            'session_timeout_minutes': 30,
            'max_concurrent_sessions': 10
        }

    def _initialize_default_tasks(self) -> None:
        """Initialize default scheduled tasks."""
        # Daily metrics aggregation
        if self.daily_enabled:
            self.add_task(
                task_id="daily_aggregation",
                name="Daily Metrics Aggregation",
                schedule_type="daily",
                strategy_ids=["all"],
                config={
                    'time': self.config.get('daily_time', '02:00'),
                    'period_days': 1
                }
            )

        # Weekly metrics aggregation
        if self.weekly_enabled:
            self.add_task(
                task_id="weekly_aggregation",
                name="Weekly Metrics Aggregation",
                schedule_type="weekly",
                strategy_ids=["all"],
                config={
                    'day': self.config.get('weekly_day', 'monday'),
                    'time': self.config.get('weekly_time', '03:00'),
                    'period_days': 7
                }
            )

    def add_task(self, task_id: str, name: str, schedule_type: str,
                strategy_ids: List[str], config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a scheduled task.

        Args:
            task_id: Unique task identifier
            name: Human-readable task name
            schedule_type: Type of schedule ('session_end', 'daily', 'weekly', 'monthly')
            strategy_ids: List of strategy IDs to process
            config: Task-specific configuration

        Returns:
            True if task was added successfully
        """
        if task_id in self.scheduled_tasks:
            self.logger.warning(f"Task {task_id} already exists")
            return False

        task = ScheduledTask(
            task_id=task_id,
            name=name,
            schedule_type=schedule_type,
            strategy_ids=strategy_ids,
            config=config or {}
        )

        self.scheduled_tasks[task_id] = task
        self.logger.info(f"Added scheduled task: {name} ({task_id})")
        return True

    def remove_task(self, task_id: str) -> bool:
        """
        Remove a scheduled task.

        Args:
            task_id: Task identifier to remove

        Returns:
            True if task was removed
        """
        if task_id in self.scheduled_tasks:
            del self.scheduled_tasks[task_id]
            self.logger.info(f"Removed scheduled task: {task_id}")
            return True
        return False

    def start_session(self, session_id: str, strategy_ids: List[str],
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start a trading session for metrics tracking.

        Args:
            session_id: Unique session identifier
            strategy_ids: List of strategy IDs in this session
            metadata: Optional session metadata

        Returns:
            True if session was started
        """
        if not self.enabled or not self.session_end_enabled:
            return True

        if session_id in self.active_sessions:
            self.logger.warning(f"Session {session_id} already active")
            return False

        # Check session limit
        max_sessions = self.config.get('max_concurrent_sessions', 10)
        if len(self.active_sessions) >= max_sessions:
            self.logger.warning(f"Maximum concurrent sessions ({max_sessions}) reached")
            return False

        session_data = {
            'session_id': session_id,
            'strategy_ids': strategy_ids,
            'start_time': datetime.now(),
            'metadata': metadata or {},
            'returns': {strategy_id: [] for strategy_id in strategy_ids},
            'trade_logs': {strategy_id: [] for strategy_id in strategy_ids}
        }

        self.active_sessions[session_id] = session_data
        self.logger.info(f"Started metrics session: {session_id} with strategies: {strategy_ids}")
        return True

    def update_session_data(self, session_id: str, strategy_id: str,
                           returns: Optional[List[float]] = None,
                           trade_log: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Update session data with new returns or trades.

        Args:
            session_id: Session identifier
            strategy_id: Strategy identifier
            returns: New returns data
            trade_log: New trade log data

        Returns:
            True if update was successful
        """
        if session_id not in self.active_sessions:
            self.logger.warning(f"Session {session_id} not found")
            return False

        session = self.active_sessions[session_id]

        if strategy_id not in session['strategy_ids']:
            self.logger.warning(f"Strategy {strategy_id} not in session {session_id}")
            return False

        if returns:
            session['returns'][strategy_id].extend(returns)

        if trade_log:
            session['trade_logs'][strategy_id].extend(trade_log)

        return True

    def end_session(self, session_id: str) -> bool:
        """
        End a trading session and generate metrics report.

        Args:
            session_id: Session identifier

        Returns:
            True if session ended successfully
        """
        if session_id not in self.active_sessions:
            self.logger.warning(f"Session {session_id} not found")
            return False

        session = self.active_sessions[session_id]
        session['end_time'] = datetime.now()

        try:
            # Generate metrics for each strategy in the session
            results = []
            for strategy_id in session['strategy_ids']:
                returns = session['returns'][strategy_id]
                trade_log = session['trade_logs'][strategy_id]

                if not returns:
                    self.logger.warning(f"No returns data for strategy {strategy_id} in session {session_id}")
                    continue

                # Generate session metrics
                result = self.metrics_engine.generate_session_report(
                    returns=returns,
                    strategy_id=strategy_id,
                    trade_log=trade_log
                )

                results.append(result)

                # Sync to dashboards
                if self.dashboard_sync.enabled:
                    sync_success = self.dashboard_sync.sync_metrics(result)
                    if not sync_success:
                        self.logger.warning(f"Failed to sync metrics for {strategy_id} to dashboards")

            # Publish session summary event
            self._publish_session_summary_event(session, results)

            # Clean up session
            del self.active_sessions[session_id]

            self.logger.info(f"Ended metrics session: {session_id} with {len(results)} strategy reports")
            return True

        except Exception as e:
            self.logger.error(f"Error ending session {session_id}: {e}", exc_info=True)
            return False

    def start_scheduler(self) -> bool:
        """
        Start the background scheduler for periodic tasks.

        Returns:
            True if scheduler started successfully
        """
        if not self.enabled or self.running:
            return False

        def scheduler_worker():
            """Background scheduler worker."""
            self.running = True
            self.logger.info("Metrics scheduler started")

            # Schedule daily tasks
            if self.daily_enabled:
                schedule.every().day.at(self.config.get('daily_time', '02:00')).do(
                    self._run_daily_tasks
                )

            # Schedule weekly tasks
            if self.weekly_enabled:
                weekly_day = self.config.get('weekly_day', 'monday')
                weekly_time = self.config.get('weekly_time', '03:00')

                # Map day names to schedule attributes
                day_map = {
                    'monday': schedule.every().monday,
                    'tuesday': schedule.every().tuesday,
                    'wednesday': schedule.every().wednesday,
                    'thursday': schedule.every().thursday,
                    'friday': schedule.every().friday,
                    'saturday': schedule.every().saturday,
                    'sunday': schedule.every().sunday
                }

                if weekly_day in day_map:
                    day_map[weekly_day].at(weekly_time).do(self._run_weekly_tasks)

            # Run scheduler loop
            while self.running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    self.logger.error(f"Scheduler error: {e}")
                    time.sleep(60)

            self.logger.info("Metrics scheduler stopped")

        # Start scheduler in background thread
        self.scheduler_thread = threading.Thread(target=scheduler_worker, daemon=True)
        self.scheduler_thread.start()

        self.logger.info("Metrics scheduler thread started")
        return True

    def stop_scheduler(self) -> bool:
        """
        Stop the background scheduler.

        Returns:
            True if scheduler stopped successfully
        """
        if not self.running:
            return True

        self.running = False

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)

        self.logger.info("Metrics scheduler stopped")
        return True

    def _run_daily_tasks(self) -> None:
        """Run daily scheduled tasks."""
        self.logger.info("Running daily metrics tasks")

        try:
            # Find daily tasks
            daily_tasks = [task for task in self.scheduled_tasks.values()
                          if task.schedule_type == 'daily' and task.enabled]

            for task in daily_tasks:
                self._execute_task(task)

        except Exception as e:
            self.logger.error(f"Error running daily tasks: {e}")

    def _run_weekly_tasks(self) -> None:
        """Run weekly scheduled tasks."""
        self.logger.info("Running weekly metrics tasks")

        try:
            # Find weekly tasks
            weekly_tasks = [task for task in self.scheduled_tasks.values()
                           if task.schedule_type == 'weekly' and task.enabled]

            for task in weekly_tasks:
                self._execute_task(task)

        except Exception as e:
            self.logger.error(f"Error running weekly tasks: {e}")

    def _execute_task(self, task: ScheduledTask) -> None:
        """
        Execute a scheduled task.

        Args:
            task: Task to execute
        """
        try:
            self.logger.info(f"Executing scheduled task: {task.name}")

            task.last_run = datetime.now()

            # Determine strategies to process
            if "all" in task.strategy_ids:
                # Get all available strategies from metrics history
                strategy_ids = self._get_all_strategies()
            else:
                strategy_ids = task.strategy_ids

            # Generate metrics for each strategy
            period_days = task.config.get('period_days', 1)

            for strategy_id in strategy_ids:
                try:
                    # Get recent returns data (this would need to be implemented based on your data storage)
                    returns = self._get_strategy_returns(strategy_id, period_days)

                    if not returns:
                        self.logger.warning(f"No returns data for strategy {strategy_id}")
                        continue

                    # Generate periodic report
                    result = self.metrics_engine.generate_periodic_report(
                        returns=returns,
                        strategy_id=strategy_id,
                        period_days=period_days
                    )

                    # Sync to dashboards
                    if self.dashboard_sync.enabled:
                        self.dashboard_sync.sync_metrics(result)

                except Exception as e:
                    self.logger.error(f"Error processing strategy {strategy_id} in task {task.task_id}: {e}")

            task.next_run = self._calculate_next_run(task)

        except Exception as e:
            self.logger.error(f"Error executing task {task.task_id}: {e}")

    def _get_all_strategies(self) -> List[str]:
        """
        Get all available strategy IDs from metrics history.

        Returns:
            List of strategy IDs
        """
        try:
            # This would query your metrics storage to get all strategy IDs
            # For now, return a placeholder
            return ["default_strategy"]
        except Exception as e:
            self.logger.error(f"Error getting strategies: {e}")
            return []

    def _get_strategy_returns(self, strategy_id: str, period_days: int) -> Optional[List[float]]:
        """
        Get returns data for a strategy over the specified period.

        Args:
            strategy_id: Strategy identifier
            period_days: Number of days of data to retrieve

        Returns:
            List of returns or None if not available
        """
        try:
            # This would query your trading data storage
            # For now, return mock data
            import numpy as np
            np.random.seed(42)  # For reproducible results

            # Generate mock daily returns
            num_periods = period_days
            returns = np.random.normal(0.001, 0.02, num_periods)  # Mean 0.1%, std 2%

            return returns.tolist()

        except Exception as e:
            self.logger.error(f"Error getting returns for {strategy_id}: {e}")
            return None

    def _calculate_next_run(self, task: ScheduledTask) -> datetime:
        """
        Calculate the next run time for a task.

        Args:
            task: Scheduled task

        Returns:
            Next run datetime
        """
        now = datetime.now()

        if task.schedule_type == 'daily':
            # Next day at the specified time
            time_str = task.config.get('time', '02:00')
            hour, minute = map(int, time_str.split(':'))

            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)

        elif task.schedule_type == 'weekly':
            # Next occurrence of the specified day and time
            day_name = task.config.get('day', 'monday')
            time_str = task.config.get('time', '03:00')

            # Map day names to weekday numbers (0=Monday, 6=Sunday)
            day_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }

            target_weekday = day_map.get(day_name, 0)
            hour, minute = map(int, time_str.split(':'))

            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # Find next occurrence of target weekday
            days_ahead = (target_weekday - now.weekday()) % 7
            if days_ahead == 0 and next_run <= now:
                days_ahead = 7

            next_run += timedelta(days=days_ahead)

        else:
            # Default to tomorrow
            next_run = now + timedelta(days=1)

        return next_run

    def _publish_session_summary_event(self, session: Dict[str, Any], results: List[MetricsResult]) -> None:
        """
        Publish session summary event.

        Args:
            session: Session data
            results: List of metrics results
        """
        event_data = {
            'event_type': 'METRICS_SESSION_SUMMARY',
            'session_id': session['session_id'],
            'start_time': session['start_time'].isoformat(),
            'end_time': session.get('end_time', datetime.now()).isoformat(),
            'duration_minutes': (session.get('end_time', datetime.now()) - session['start_time']).total_seconds() / 60,
            'strategies_processed': len(session['strategy_ids']),
            'metrics_generated': len(results),
            'strategies': session['strategy_ids']
        }

        self.logger.info(f"Published session summary event: {event_data}")

    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get scheduler status and statistics.

        Returns:
            Status information
        """
        return {
            'enabled': self.enabled,
            'running': self.running,
            'active_sessions': len(self.active_sessions),
            'scheduled_tasks': len(self.scheduled_tasks),
            'session_end_enabled': self.session_end_enabled,
            'daily_enabled': self.daily_enabled,
            'weekly_enabled': self.weekly_enabled,
            'tasks': [
                {
                    'task_id': task.task_id,
                    'name': task.name,
                    'schedule_type': task.schedule_type,
                    'enabled': task.enabled,
                    'last_run': task.last_run.isoformat() if task.last_run else None,
                    'next_run': task.next_run.isoformat() if task.next_run else None
                }
                for task in self.scheduled_tasks.values()
            ]
        }

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """
        Get information about active sessions.

        Returns:
            List of active session information
        """
        sessions = []
        for session_id, session_data in self.active_sessions.items():
            sessions.append({
                'session_id': session_id,
                'strategy_ids': session_data['strategy_ids'],
                'start_time': session_data['start_time'].isoformat(),
                'duration_minutes': (datetime.now() - session_data['start_time']).total_seconds() / 60,
                'returns_count': {sid: len(returns) for sid, returns in session_data['returns'].items()},
                'trades_count': {sid: len(trades) for sid, trades in session_data['trade_logs'].items()}
            })

        return sessions

    def cleanup_expired_sessions(self, timeout_minutes: Optional[int] = None) -> int:
        """
        Clean up expired sessions.

        Args:
            timeout_minutes: Session timeout in minutes

        Returns:
            Number of sessions cleaned up
        """
        if timeout_minutes is None:
            timeout_minutes = self.config.get('session_timeout_minutes', 30)

        cutoff_time = datetime.now() - timedelta(minutes=timeout_minutes)
        expired_sessions = []

        for session_id, session_data in self.active_sessions.items():
            if session_data['start_time'] < cutoff_time:
                expired_sessions.append(session_id)

        # End expired sessions
        for session_id in expired_sessions:
            self.logger.info(f"Cleaning up expired session: {session_id}")
            self.end_session(session_id)

        return len(expired_sessions)


# Global instance
_metrics_scheduler: Optional[MetricsScheduler] = None


def get_metrics_scheduler() -> MetricsScheduler:
    """Get the global metrics scheduler instance."""
    global _metrics_scheduler
    if _metrics_scheduler is None:
        from utils.config_loader import get_config
        config = get_config('metrics_scheduler', {})
        _metrics_scheduler = MetricsScheduler(config)
    return _metrics_scheduler


def start_session(session_id: str, strategy_ids: List[str],
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Convenience function to start a metrics session.

    Args:
        session_id: Unique session identifier
        strategy_ids: List of strategy IDs
        metadata: Optional session metadata

    Returns:
        True if session started successfully
    """
    scheduler = get_metrics_scheduler()
    return scheduler.start_session(session_id, strategy_ids, metadata)


def end_session(session_id: str) -> bool:
    """
    Convenience function to end a metrics session.

    Args:
        session_id: Session identifier

    Returns:
        True if session ended successfully
    """
    scheduler = get_metrics_scheduler()
    return scheduler.end_session(session_id)


def update_session_data(session_id: str, strategy_id: str,
                       returns: Optional[List[float]] = None,
                       trade_log: Optional[List[Dict[str, Any]]] = None) -> bool:
    """
    Convenience function to update session data.

    Args:
        session_id: Session identifier
        strategy_id: Strategy identifier
        returns: New returns data
        trade_log: New trade log data

    Returns:
        True if update was successful
    """
    scheduler = get_metrics_scheduler()
    return scheduler.update_session_data(session_id, strategy_id, returns, trade_log)
