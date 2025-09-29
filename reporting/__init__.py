"""
Reporting Module

Provides comprehensive performance and risk reporting capabilities
for the trading framework, including automatic metrics calculation,
persistence, and dashboard synchronization.
"""

from .metrics import (
    MetricsEngine,
    MetricsResult,
    calculate_session_metrics,
    get_metrics_engine,
)
from .scheduler import (
    MetricsScheduler,
    end_session,
    get_metrics_scheduler,
    start_session,
    update_session_data,
)
from .sync import (
    DashboardSync,
    StreamlitDashboard,
    get_dashboard_sync,
    get_streamlit_dashboard,
    sync_metrics_to_dashboards,
)

__all__ = [
    # Metrics Engine
    "MetricsEngine",
    "MetricsResult",
    "get_metrics_engine",
    "calculate_session_metrics",
    # Dashboard Sync
    "DashboardSync",
    "StreamlitDashboard",
    "get_dashboard_sync",
    "get_streamlit_dashboard",
    "sync_metrics_to_dashboards",
    # Scheduler
    "MetricsScheduler",
    "get_metrics_scheduler",
    "start_session",
    "end_session",
    "update_session_data",
]
