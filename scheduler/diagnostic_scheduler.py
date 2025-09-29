"""
Diagnostic scheduler for periodic health monitoring.

Provides scheduling capabilities for running diagnostics at regular intervals
and managing diagnostic reports.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.diagnostics import (
    DiagnosticsManager,
    get_diagnostics_manager,
)
from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticReport:
    """Daily diagnostic report summary."""

    date: str
    total_checks: int = 0
    healthy_checks: int = 0
    degraded_checks: int = 0
    critical_checks: int = 0
    anomalies_detected: int = 0
    average_latency_ms: float = 0.0
    uptime_percentage: float = 100.0
    top_issues: List[str] = field(default_factory=list)
    component_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class DiagnosticScheduler:
    """
    Scheduler for periodic diagnostic execution and reporting.
    """

    def __init__(self, diagnostics_manager: Optional[DiagnosticsManager] = None):
        """
        Initialize the diagnostic scheduler.

        Args:
            diagnostics_manager: Optional diagnostics manager instance
        """
        self.diagnostics = diagnostics_manager or get_diagnostics_manager()
        self.logger = get_trade_logger()

        # Scheduling configuration
        self.check_interval_sec = 60  # Run diagnostics every minute
        self.report_interval_hours = 24  # Generate daily report

        # State tracking
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._reporter_task: Optional[asyncio.Task] = None

        # Daily report tracking
        self.daily_report = DiagnosticReport(date=datetime.now().strftime("%Y-%m-%d"))
        self.last_report_time = datetime.now()

        # Reports directory
        self.reports_dir = Path("reports/diagnostics")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("DiagnosticScheduler initialized")

    async def start(self) -> None:
        """Start the diagnostic scheduler."""
        if self._running:
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduling_loop())
        self._reporter_task = asyncio.create_task(self._reporting_loop())

        logger.info("Diagnostic scheduler started")

    async def stop(self) -> None:
        """Stop the diagnostic scheduler."""
        if not self._running:
            return

        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        if self._reporter_task:
            self._reporter_task.cancel()
            try:
                await self._reporter_task
            except asyncio.CancelledError:
                pass

        # Generate final report
        await self._generate_daily_report()

        logger.info("Diagnostic scheduler stopped")

    async def _scheduling_loop(self) -> None:
        """Main scheduling loop for running diagnostics."""
        while self._running:
            try:
                # Run diagnostics
                await self.diagnostics.run_health_check()

                # Update daily report
                await self._update_daily_report()

                # Wait for next check
                await asyncio.sleep(self.check_interval_sec)

            except Exception as e:
                logger.exception(f"Error in diagnostic scheduling loop: {e}")
                await asyncio.sleep(10)  # Brief pause before retry

    async def _reporting_loop(self) -> None:
        """Reporting loop for generating periodic reports."""
        while self._running:
            try:
                now = datetime.now()
                next_report = self.last_report_time + timedelta(
                    hours=self.report_interval_hours
                )

                if now >= next_report:
                    await self._generate_daily_report()
                    self.last_report_time = now
                    # Reset daily report for new day
                    self.daily_report = DiagnosticReport(date=now.strftime("%Y-%m-%d"))

                # Sleep until next report check (every hour)
                await asyncio.sleep(3600)

            except Exception as e:
                logger.exception(f"Error in diagnostic reporting loop: {e}")
                await asyncio.sleep(60)  # Brief pause before retry

    async def _update_daily_report(self) -> None:
        """Update the daily report with latest diagnostic results."""
        try:
            # Get current health status
            status = self.diagnostics.get_detailed_status()

            # Update counters
            self.daily_report.total_checks += 1

            # Count component statuses
            healthy_count = 0
            degraded_count = 0
            critical_count = 0
            total_latency = 0.0
            latency_count = 0

            for component_name, component_data in status.get("components", {}).items():
                comp_status = component_data.get("status", "unknown")

                if comp_status == "healthy":
                    healthy_count += 1
                elif comp_status == "degraded":
                    degraded_count += 1
                elif comp_status == "critical":
                    critical_count += 1

                # Track latency
                latency = component_data.get("latency_ms")
                if latency is not None:
                    total_latency += latency
                    latency_count += 1

                # Update component summary
                if component_name not in self.daily_report.component_summaries:
                    self.daily_report.component_summaries[component_name] = {
                        "total_checks": 0,
                        "healthy": 0,
                        "degraded": 0,
                        "critical": 0,
                        "avg_latency_ms": 0.0,
                        "last_status": "unknown",
                    }

                comp_summary = self.daily_report.component_summaries[component_name]
                comp_summary["total_checks"] += 1
                comp_summary["last_status"] = comp_status

                if comp_status == "healthy":
                    comp_summary["healthy"] += 1
                elif comp_status == "degraded":
                    comp_summary["degraded"] += 1
                elif comp_status == "critical":
                    comp_summary["critical"] += 1

                # Update average latency
                if latency is not None:
                    current_avg = comp_summary["avg_latency_ms"]
                    comp_summary["avg_latency_ms"] = (
                        (current_avg * (comp_summary["total_checks"] - 1)) + latency
                    ) / comp_summary["total_checks"]

            # Update daily report counters
            self.daily_report.healthy_checks = healthy_count
            self.daily_report.degraded_checks = degraded_count
            self.daily_report.critical_checks = critical_count

            # Update average latency
            if latency_count > 0:
                current_avg = self.daily_report.average_latency_ms
                self.daily_report.average_latency_ms = (
                    (current_avg * (self.daily_report.total_checks - 1))
                    + (total_latency / latency_count)
                ) / self.daily_report.total_checks

            # Update anomalies count
            self.daily_report.anomalies_detected = status.get("anomaly_count", 0)

            # Calculate uptime percentage (based on healthy checks)
            total_components = len(status.get("components", {}))
            if total_components > 0:
                healthy_percentage = (
                    healthy_count / (healthy_count + degraded_count + critical_count)
                ) * 100
                self.daily_report.uptime_percentage = healthy_percentage

            # Update top issues
            if critical_count > 0:
                critical_components = [
                    name
                    for name, data in status.get("components", {}).items()
                    if data.get("status") == "critical"
                ]
                self.daily_report.top_issues = critical_components[:5]  # Top 5 issues

        except Exception as e:
            logger.exception(f"Error updating daily report: {e}")

    async def _generate_daily_report(self) -> None:
        """Generate and save the daily diagnostic report."""
        try:
            # Convert report to dictionary
            report_dict = {
                "date": self.daily_report.date,
                "total_checks": self.daily_report.total_checks,
                "healthy_checks": self.daily_report.healthy_checks,
                "degraded_checks": self.daily_report.degraded_checks,
                "critical_checks": self.daily_report.critical_checks,
                "anomalies_detected": self.daily_report.anomalies_detected,
                "average_latency_ms": round(self.daily_report.average_latency_ms, 2),
                "uptime_percentage": round(self.daily_report.uptime_percentage, 2),
                "top_issues": self.daily_report.top_issues,
                "component_summaries": self.daily_report.component_summaries,
                "created_at": self.daily_report.created_at.isoformat(),
                "generated_at": datetime.now().isoformat(),
            }

            # Save to file
            report_file = (
                self.reports_dir / f"diagnostic_report_{self.daily_report.date}.json"
            )
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=2, default=str)

            # Log summary
            self.logger.info(
                f"Daily diagnostic report generated: {self.daily_report.date}",
                extra={
                    "report_summary": {
                        "total_checks": self.daily_report.total_checks,
                        "uptime_percentage": f"{self.daily_report.uptime_percentage:.1f}%",
                        "critical_issues": len(self.daily_report.top_issues),
                        "anomalies": self.daily_report.anomalies_detected,
                    }
                },
            )

            logger.info(f"Diagnostic report saved to {report_file}")

        except Exception as e:
            logger.exception(f"Error generating daily report: {e}")

    def get_current_report(self) -> Dict[str, Any]:
        """Get the current daily report data."""
        return {
            "date": self.daily_report.date,
            "total_checks": self.daily_report.total_checks,
            "healthy_checks": self.daily_report.healthy_checks,
            "degraded_checks": self.daily_report.degraded_checks,
            "critical_checks": self.daily_report.critical_checks,
            "anomalies_detected": self.daily_report.anomalies_detected,
            "average_latency_ms": round(self.daily_report.average_latency_ms, 2),
            "uptime_percentage": round(self.daily_report.uptime_percentage, 2),
            "top_issues": self.daily_report.top_issues,
            "component_summaries": self.daily_report.component_summaries,
            "last_updated": datetime.now().isoformat(),
        }

    def get_recent_reports(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent diagnostic reports."""
        reports = []

        try:
            for report_file in self.reports_dir.glob("diagnostic_report_*.json"):
                try:
                    with open(report_file, "r", encoding="utf-8") as f:
                        report_data = json.load(f)
                        reports.append(report_data)
                except Exception as e:
                    logger.warning(f"Error reading report file {report_file}: {e}")

            # Sort by date (newest first) and limit to requested days
            reports.sort(key=lambda x: x.get("date", ""), reverse=True)
            return reports[:days]

        except Exception as e:
            logger.exception(f"Error getting recent reports: {e}")
            return []

    async def run_manual_check(self) -> Dict[str, Any]:
        """Run a manual health check and return results."""
        try:
            await self.diagnostics.run_health_check()
            return self.diagnostics.get_detailed_status()
        except Exception as e:
            logger.exception(f"Error running manual health check: {e}")
            return {
                "error": str(e),
                "overall_status": "critical",
                "last_check": datetime.now().isoformat(),
            }

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status information."""
        return {
            "running": self._running,
            "check_interval_sec": self.check_interval_sec,
            "report_interval_hours": self.report_interval_hours,
            "last_report_time": self.last_report_time.isoformat(),
            "reports_directory": str(self.reports_dir),
            "current_report": self.get_current_report(),
        }


# Global scheduler instance
_global_scheduler: Optional[DiagnosticScheduler] = None


def get_diagnostic_scheduler() -> DiagnosticScheduler:
    """Get the global diagnostic scheduler instance."""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = DiagnosticScheduler()
    return _global_scheduler


def create_diagnostic_scheduler(
    diagnostics_manager: Optional[DiagnosticsManager] = None,
) -> DiagnosticScheduler:
    """Create a new diagnostic scheduler instance."""
    return DiagnosticScheduler(diagnostics_manager)
