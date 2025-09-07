"""
Dashboard Synchronization

Handles synchronization of metrics with external dashboard systems
like Grafana and Streamlit.
"""

import logging
import os
import json
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
import shutil

from .metrics import MetricsResult

logger = logging.getLogger(__name__)


class DashboardSync:
    """
    Handles synchronization of metrics with dashboard systems.

    Supports Grafana (via Prometheus/JSON endpoints) and Streamlit
    (via file-based data sharing).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize dashboard synchronization.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Grafana configuration
        self.grafana_enabled = self.config.get('grafana', {}).get('enabled', False)
        self.grafana_url = self.config.get('grafana', {}).get('url', '')
        self.grafana_api_key = self.config.get('grafana', {}).get('api_key', '')

        # Streamlit configuration
        self.streamlit_enabled = self.config.get('streamlit', {}).get('enabled', False)
        self.streamlit_data_dir = self.config.get('streamlit', {}).get('data_dir', 'dashboard/data')

        # General settings
        self.enabled = self.config.get('enabled', True)
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout = self.config.get('timeout', 10)

        # Create directories
        if self.streamlit_enabled:
            os.makedirs(self.streamlit_data_dir, exist_ok=True)

        self.logger.info("DashboardSync initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'enabled': True,
            'max_retries': 3,
            'timeout': 10,
            'grafana': {
                'enabled': False,
                'url': 'http://localhost:3000',
                'api_key': ''
            },
            'streamlit': {
                'enabled': True,
                'data_dir': 'dashboard/data'
            }
        }

    def sync_metrics(self, result: MetricsResult) -> bool:
        """
        Synchronize metrics with all configured dashboards.

        Args:
            result: MetricsResult to sync

        Returns:
            True if sync was successful
        """
        if not self.enabled:
            return True

        success = True

        # Sync with Grafana
        if self.grafana_enabled:
            try:
                grafana_success = self._sync_to_grafana(result)
                if not grafana_success:
                    success = False
            except Exception as e:
                self.logger.error(f"Grafana sync failed: {e}")
                success = False

        # Sync with Streamlit
        if self.streamlit_enabled:
            try:
                streamlit_success = self._sync_to_streamlit(result)
                if not streamlit_success:
                    success = False
            except Exception as e:
                self.logger.error(f"Streamlit sync failed: {e}")
                success = False

        return success

    def _sync_to_grafana(self, result: MetricsResult) -> bool:
        """
        Sync metrics to Grafana via API.

        Args:
            result: MetricsResult to sync

        Returns:
            True if sync was successful
        """
        if not self.grafana_enabled or not self.grafana_url:
            return True

        try:
            # Convert metrics to Prometheus format
            prometheus_data = self._convert_to_prometheus(result)

            # Send to Grafana
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.grafana_api_key}' if self.grafana_api_key else ''
            }

            url = f"{self.grafana_url}/api/v1/push"
            response = requests.post(
                url,
                json=prometheus_data,
                headers=headers,
                timeout=self.timeout
            )

            if response.status_code in [200, 202]:
                self.logger.info(f"Successfully synced metrics to Grafana for {result.strategy_id}")
                return True
            else:
                self.logger.error(f"Grafana sync failed with status {response.status_code}: {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"Error syncing to Grafana: {e}")
            return False

    def _sync_to_streamlit(self, result: MetricsResult) -> bool:
        """
        Sync metrics to Streamlit via file system.

        Args:
            result: MetricsResult to sync

        Returns:
            True if sync was successful
        """
        if not self.streamlit_enabled:
            return True

        try:
            # Save latest metrics
            latest_file = os.path.join(self.streamlit_data_dir, 'latest_metrics.json')
            with open(latest_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)

            # Save historical metrics
            history_file = os.path.join(self.streamlit_data_dir, 'metrics_history.json')
            self._append_to_history(history_file, result)

            # Save strategy-specific metrics
            strategy_file = os.path.join(self.streamlit_data_dir, f'metrics_{result.strategy_id}.json')
            with open(strategy_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)

            self.logger.info(f"Successfully synced metrics to Streamlit for {result.strategy_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error syncing to Streamlit: {e}")
            return False

    def _convert_to_prometheus(self, result: MetricsResult) -> Dict[str, Any]:
        """
        Convert MetricsResult to Prometheus format.

        Args:
            result: MetricsResult to convert

        Returns:
            Prometheus-formatted data
        """
        timestamp_ms = int(result.timestamp.timestamp() * 1000)

        # Create metric data points
        metrics_data = []

        # Performance metrics
        metrics_data.extend([
            {
                'name': 'trading_total_return',
                'value': result.total_return,
                'timestamp': timestamp_ms,
                'labels': {'strategy': result.strategy_id}
            },
            {
                'name': 'trading_sharpe_ratio',
                'value': result.sharpe_ratio,
                'timestamp': timestamp_ms,
                'labels': {'strategy': result.strategy_id}
            },
            {
                'name': 'trading_max_drawdown',
                'value': result.max_drawdown,
                'timestamp': timestamp_ms,
                'labels': {'strategy': result.strategy_id}
            },
            {
                'name': 'trading_win_rate',
                'value': result.win_rate,
                'timestamp': timestamp_ms,
                'labels': {'strategy': result.strategy_id}
            }
        ])

        # Risk metrics
        metrics_data.extend([
            {
                'name': 'trading_volatility',
                'value': result.volatility,
                'timestamp': timestamp_ms,
                'labels': {'strategy': result.strategy_id}
            },
            {
                'name': 'trading_value_at_risk_95',
                'value': result.value_at_risk_95,
                'timestamp': timestamp_ms,
                'labels': {'strategy': result.strategy_id}
            }
        ])

        # Trade statistics
        metrics_data.extend([
            {
                'name': 'trading_total_trades',
                'value': result.total_trades,
                'timestamp': timestamp_ms,
                'labels': {'strategy': result.strategy_id}
            },
            {
                'name': 'trading_winning_trades',
                'value': result.winning_trades,
                'timestamp': timestamp_ms,
                'labels': {'strategy': result.strategy_id}
            }
        ])

        return {
            'metrics': metrics_data
        }

    def _append_to_history(self, history_file: str, result: MetricsResult) -> None:
        """
        Append metrics to historical data file.

        Args:
            history_file: Path to history file
            result: MetricsResult to append
        """
        try:
            # Load existing history
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []

            # Add new metrics
            history.append(result.to_dict())

            # Keep only last 1000 entries to prevent file from growing too large
            if len(history) > 1000:
                history = history[-1000:]

            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error appending to history: {e}")

    def get_dashboard_status(self) -> Dict[str, Any]:
        """
        Get status of dashboard synchronization.

        Returns:
            Status information
        """
        return {
            'enabled': self.enabled,
            'grafana': {
                'enabled': self.grafana_enabled,
                'connected': self._test_grafana_connection() if self.grafana_enabled else False
            },
            'streamlit': {
                'enabled': self.streamlit_enabled,
                'data_dir_exists': os.path.exists(self.streamlit_data_dir) if self.streamlit_enabled else False
            }
        }

    def _test_grafana_connection(self) -> bool:
        """
        Test connection to Grafana.

        Returns:
            True if connection is successful
        """
        if not self.grafana_enabled or not self.grafana_url:
            return False

        try:
            headers = {
                'Authorization': f'Bearer {self.grafana_api_key}' if self.grafana_api_key else ''
            }

            response = requests.get(
                f"{self.grafana_url}/api/health",
                headers=headers,
                timeout=self.timeout
            )

            return response.status_code == 200
        except:
            return False

    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        Clean up old dashboard data files.

        Args:
            days_to_keep: Number of days of data to keep

        Returns:
            Number of files cleaned up
        """
        if not self.streamlit_enabled:
            return 0

        try:
            import time
            cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
            cleaned_count = 0

            for filename in os.listdir(self.streamlit_data_dir):
                filepath = os.path.join(self.streamlit_data_dir, filename)
                if os.path.isfile(filepath):
                    file_time = os.path.getmtime(filepath)
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        cleaned_count += 1

            self.logger.info(f"Cleaned up {cleaned_count} old dashboard files")
            return cleaned_count

        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            return 0


class StreamlitDashboard:
    """
    Streamlit-specific dashboard utilities.
    """

    def __init__(self, data_dir: str = 'dashboard/data'):
        """
        Initialize Streamlit dashboard.

        Args:
            data_dir: Directory for dashboard data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def get_latest_metrics(self, strategy_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get latest metrics for dashboard display.

        Args:
            strategy_id: Optional strategy filter

        Returns:
            Latest metrics data
        """
        try:
            if strategy_id:
                filepath = os.path.join(self.data_dir, f'metrics_{strategy_id}.json')
            else:
                filepath = os.path.join(self.data_dir, 'latest_metrics.json')

            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            return None

        except Exception as e:
            logger.error(f"Error getting latest metrics: {e}")
            return None

    def get_metrics_history(self, strategy_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical metrics for dashboard display.

        Args:
            strategy_id: Optional strategy filter
            limit: Maximum number of records

        Returns:
            Historical metrics data
        """
        try:
            filepath = os.path.join(self.data_dir, 'metrics_history.json')

            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    history = json.load(f)

                # Filter by strategy if specified
                if strategy_id:
                    history = [h for h in history if h.get('strategy_id') == strategy_id]

                # Return most recent records
                return history[-limit:] if history else []
            return []

        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            return []

    def get_all_strategies(self) -> List[str]:
        """
        Get list of all strategies with metrics.

        Returns:
            List of strategy IDs
        """
        try:
            strategies = set()

            # Check individual strategy files
            for filename in os.listdir(self.data_dir):
                if filename.startswith('metrics_') and filename.endswith('.json') and not filename.startswith('metrics_history'):
                    strategy_id = filename.replace('metrics_', '').replace('.json', '')
                    strategies.add(strategy_id)

            # Check history file
            history_file = os.path.join(self.data_dir, 'metrics_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    for record in history:
                        if 'strategy_id' in record:
                            strategies.add(record['strategy_id'])

            return sorted(list(strategies))

        except Exception as e:
            logger.error(f"Error getting strategies: {e}")
            return []


# Global instances
_dashboard_sync: Optional[DashboardSync] = None
_streamlit_dashboard: Optional[StreamlitDashboard] = None


def get_dashboard_sync() -> DashboardSync:
    """Get the global dashboard sync instance."""
    global _dashboard_sync
    if _dashboard_sync is None:
        from utils.config_loader import get_config
        config = get_config('dashboard_sync', {})
        _dashboard_sync = DashboardSync(config)
    return _dashboard_sync


def get_streamlit_dashboard() -> StreamlitDashboard:
    """Get the global Streamlit dashboard instance."""
    global _streamlit_dashboard
    if _streamlit_dashboard is None:
        from utils.config_loader import get_config
        config = get_config('streamlit', {})
        data_dir = config.get('data_dir', 'dashboard/data')
        _streamlit_dashboard = StreamlitDashboard(data_dir)
    return _streamlit_dashboard


def sync_metrics_to_dashboards(result: MetricsResult) -> bool:
    """
    Convenience function to sync metrics to all configured dashboards.

    Args:
        result: MetricsResult to sync

    Returns:
        True if sync was successful
    """
    sync = get_dashboard_sync()
    return sync.sync_metrics(result)
