"""
Reporting Module

Provides comprehensive performance and risk reporting capabilities
for the trading framework, including automatic metrics calculation,
persistence, and dashboard synchronization.
"""

from .metrics import (
    MetricsEngine,
    MetricsResult,
    get_metrics_engine,
    calculate_session_metrics
)

from .sync import (
    DashboardSync,
    StreamlitDashboard,
    get_dashboard_sync,
    get_streamlit_dashboard,
    sync_metrics_to_dashboards
)

from .scheduler import (
    MetricsScheduler,
    get_metrics_scheduler,
    start_session,
    end_session,
    update_session_data
)

__all__ = [
    # Metrics Engine
    'MetricsEngine',
    'MetricsResult',
    'get_metrics_engine',
    'calculate_session_metrics',

    # Dashboard Sync
    'DashboardSync',
    'StreamlitDashboard',
    'get_dashboard_sync',
    'get_streamlit_dashboard',
    'sync_metrics_to_dashboards',

    # Scheduler
    'MetricsScheduler',
    'get_metrics_scheduler',
    'start_session',
    'end_session',
    'update_session_data'
]
