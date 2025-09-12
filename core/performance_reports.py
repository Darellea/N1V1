"""
Comprehensive Performance Reports
=================================

This module provides comprehensive performance reporting capabilities for the trading framework,
including hierarchical performance breakdowns, hotspot identification, comparative analysis,
and trend analysis for long-term performance tracking.

Key Features:
- Hierarchical performance breakdowns
- Hotspot identification with code context
- Comparative analysis between runs
- Trend analysis for long-term performance tracking
- Export capabilities (JSON, CSV, HTML)
- Interactive performance dashboards
"""

import asyncio
import time
import json
import csv
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import statistics
import logging
from collections import defaultdict, OrderedDict
import pandas as pd
import numpy as np
from jinja2 import Template
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from core.performance_profiler import get_profiler, PerformanceProfiler
from core.performance_monitor import get_performance_monitor, RealTimePerformanceMonitor
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceReport:
    """Comprehensive performance report data structure."""
    report_id: str
    timestamp: float
    duration: float
    summary: Dict[str, Any] = field(default_factory=dict)
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    trends: Dict[str, Any] = field(default_factory=dict)
    comparisons: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HotspotAnalysis:
    """Detailed analysis of performance hotspots."""
    function_name: str
    total_time: float
    avg_time: float
    call_count: int
    percentage: float
    code_context: Optional[str] = None
    optimization_potential: str = "unknown"
    severity: str = "low"


class PerformanceReportGenerator:
    """
    Comprehensive performance report generator for the trading framework.

    Generates detailed reports with hierarchical breakdowns, hotspot analysis,
    trend analysis, and comparative performance insights.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Report configuration
        self.max_hotspots = config.get('max_hotspots', 20)
        self.trend_window = config.get('trend_window', 3600.0)  # 1 hour
        self.comparison_window = config.get('comparison_window', 86400.0)  # 24 hours

        # Dependencies
        self.profiler = get_profiler()
        self.monitor = get_performance_monitor()

        # Report storage
        self.reports: Dict[str, PerformanceReport] = {}
        self.report_history: List[PerformanceReport] = []

        # Templates
        self.html_template = self._load_html_template()

        logger.info("PerformanceReportGenerator initialized")

    async def generate_comprehensive_report(self, session_id: Optional[str] = None,
                                          include_trends: bool = True,
                                          include_comparisons: bool = True) -> PerformanceReport:
        """
        Generate a comprehensive performance report.

        Args:
            session_id: Optional session ID to focus on
            include_trends: Whether to include trend analysis
            include_comparisons: Whether to include comparative analysis

        Returns:
            Comprehensive performance report
        """
        start_time = time.time()
        report_id = f"report_{int(start_time)}"

        # Generate base report data
        profiler_report = self.profiler.generate_report(session_id)

        # Get monitor status, handling existing event loop
        monitor_status = {}
        if self.monitor:
            try:
                # Check if we're already in an event loop
                loop = asyncio.get_running_loop()
                # If we get here, there's already a running loop
                # Use asyncio.create_task to run the coroutine
                task = asyncio.create_task(self.monitor.get_performance_status())
                monitor_status = await task
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                monitor_status = asyncio.run(self.monitor.get_performance_status())

        # Create report structure
        report = PerformanceReport(
            report_id=report_id,
            timestamp=start_time,
            duration=0.0,  # Will be set at end
            metadata={
                "session_id": session_id,
                "generator_version": "1.0.0",
                "framework_version": "N1V1"
            }
        )

        # Generate summary
        report.summary = self._generate_summary(profiler_report, monitor_status)

        # Identify hotspots
        report.hotspots = self._identify_hotspots(profiler_report)

        # Analyze trends
        if include_trends:
            report.trends = self._analyze_trends()

        # Generate comparisons
        if include_comparisons:
            report.comparisons = self._generate_comparisons()

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        # Set duration
        report.duration = time.time() - start_time

        # Store report
        self.reports[report_id] = report
        self.report_history.append(report)

        logger.info(f"Generated comprehensive performance report: {report_id}")
        return report

    def _generate_summary(self, profiler_report: Dict, monitor_status: Dict) -> Dict[str, Any]:
        """Generate report summary."""
        summary = {
            "total_functions": 0,
            "total_measurements": 0,
            "time_range": {"start": None, "end": None},
            "performance_score": 50.0,
            "system_health": monitor_status.get("system_health", 50.0),
            "anomaly_count": monitor_status.get("recent_anomalies", 0),
            "alert_count": monitor_status.get("active_alerts", 0)
        }

        if "summary" in profiler_report:
            prof_summary = profiler_report["summary"]
            summary.update({
                "total_functions": prof_summary.get("total_functions", 0),
                "total_measurements": prof_summary.get("total_measurements", 0),
                "time_range": prof_summary.get("time_range", {"start": None, "end": None})
            })

        # Calculate performance score
        if monitor_status.get("system_health"):
            summary["performance_score"] = monitor_status["system_health"]

        return summary

    def _identify_hotspots(self, profiler_report: Dict) -> List[Dict[str, Any]]:
        """Identify performance hotspots."""
        hotspots = []

        if "functions" not in profiler_report:
            return hotspots

        # Get profiler hotspots
        profiler_hotspots = self.profiler.get_hotspots(self.max_hotspots)

        for hotspot in profiler_hotspots:
            analysis = HotspotAnalysis(
                function_name=hotspot["function"],
                total_time=hotspot["total_time"],
                avg_time=hotspot["avg_time"],
                call_count=hotspot["call_count"],
                percentage=0.0,  # Will be calculated
                optimization_potential=self._assess_optimization_potential(hotspot),
                severity=self._calculate_severity(hotspot)
            )

            hotspots.append({
                "function_name": analysis.function_name,
                "total_time": analysis.total_time,
                "avg_time": analysis.avg_time,
                "call_count": analysis.call_count,
                "optimization_potential": analysis.optimization_potential,
                "severity": analysis.severity,
                "recommendations": self._get_hotspot_recommendations(analysis)
            })

        # Calculate percentages
        if hotspots:
            total_time = sum(h["total_time"] for h in hotspots)
            for hotspot in hotspots:
                hotspot["percentage"] = (hotspot["total_time"] / total_time) * 100 if total_time > 0 else 0

        return hotspots

    def _assess_optimization_potential(self, hotspot: Dict) -> str:
        """Assess optimization potential for a hotspot."""
        avg_time = hotspot.get("avg_time", 0)

        if avg_time > 1.0:  # > 1 second
            return "high"
        elif avg_time > 0.1:  # > 100ms
            return "medium"
        elif avg_time > 0.01:  # > 10ms
            return "low"
        else:
            return "minimal"

    def _calculate_severity(self, hotspot: Dict) -> str:
        """Calculate severity level for a hotspot."""
        total_time = hotspot.get("total_time", 0)
        call_count = hotspot.get("call_count", 0)

        # High severity if significant time consumption or frequent calls
        if total_time > 10.0 or call_count > 1000:
            return "high"
        elif total_time > 1.0 or call_count > 100:
            return "medium"
        else:
            return "low"

    def _get_hotspot_recommendations(self, analysis: HotspotAnalysis) -> List[str]:
        """Get recommendations for optimizing a hotspot."""
        recommendations = []

        if analysis.avg_time > 0.1:
            recommendations.append("Consider vectorization or NumPy operations")
        if analysis.call_count > 100:
            recommendations.append("Review algorithm complexity - potential O(n²) or worse")
        if "loop" in analysis.function_name.lower():
            recommendations.append("Consider replacing Python loops with NumPy operations")
        if analysis.severity == "high":
            recommendations.append("High priority: Implement caching or precomputation")

        return recommendations

    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends."""
        trends = {
            "overall_trend": "stable",
            "improving_metrics": [],
            "degrading_metrics": [],
            "trend_analysis": {}
        }

        if not self.monitor:
            return trends

        # Analyze baselines for trends
        for metric_name, baseline in self.monitor.baselines.items():
            trend_info = {
                "slope": baseline.trend_slope,
                "direction": "improving" if baseline.trend_slope < 0 else "degrading",
                "magnitude": abs(baseline.trend_slope),
                "is_significant": abs(baseline.trend_slope) > 0.01
            }

            trends["trend_analysis"][metric_name] = trend_info

            if trend_info["is_significant"]:
                if trend_info["direction"] == "improving":
                    trends["improving_metrics"].append(metric_name)
                else:
                    trends["degrading_metrics"].append(metric_name)

        # Overall trend assessment
        improving_count = len(trends["improving_metrics"])
        degrading_count = len(trends["degrading_metrics"])

        if improving_count > degrading_count:
            trends["overall_trend"] = "improving"
        elif degrading_count > improving_count:
            trends["overall_trend"] = "degrading"

        return trends

    def _generate_comparisons(self) -> Dict[str, Any]:
        """Generate comparative analysis."""
        comparisons = {
            "baseline_comparison": {},
            "historical_comparison": {},
            "benchmark_comparison": {}
        }

        if len(self.report_history) < 2:
            return comparisons

        # Compare with previous report
        current = self.report_history[-1]
        previous = self.report_history[-2] if len(self.report_history) > 1 else None

        if previous:
            comparisons["historical_comparison"] = {
                "duration_change": current.duration - previous.duration,
                "function_count_change": (
                    current.summary.get("total_functions", 0) -
                    previous.summary.get("total_functions", 0)
                ),
                "performance_score_change": (
                    current.summary.get("performance_score", 50.0) -
                    previous.summary.get("performance_score", 50.0)
                )
            }

        return comparisons

    def _generate_recommendations(self, report: PerformanceReport) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Based on hotspots
        high_severity_hotspots = [h for h in report.hotspots if h.get("severity") == "high"]
        if high_severity_hotspots:
            recommendations.append(
                f"Focus on optimizing {len(high_severity_hotspots)} high-severity hotspots"
            )

        # Based on trends
        if report.trends.get("overall_trend") == "degrading":
            recommendations.append("Performance is degrading - investigate recent changes")

        # Based on system health
        health_score = report.summary.get("system_health", 50.0)
        if health_score < 70:
            recommendations.append("System health is below optimal - review resource usage")

        # Based on anomalies
        anomaly_count = report.summary.get("anomaly_count", 0)
        if anomaly_count > 5:
            recommendations.append("High anomaly rate detected - investigate performance instability")

        return recommendations

    def export_report(self, report: PerformanceReport, format: str = "json",
                     output_path: Optional[str] = None) -> str:
        """
        Export performance report in various formats.

        Args:
            report: Performance report to export
            format: Export format ('json', 'csv', 'html')
            output_path: Optional output path

        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.fromtimestamp(report.timestamp).strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/performance_report_{timestamp}.{format}"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            self._export_json(report, output_path)
        elif format == "csv":
            self._export_csv(report, output_path)
        elif format == "html":
            self._export_html(report, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported performance report to: {output_path}")
        return output_path

    def _export_json(self, report: PerformanceReport, output_path: str) -> None:
        """Export report as JSON."""
        data = {
            "report_id": report.report_id,
            "timestamp": report.timestamp,
            "duration": report.duration,
            "summary": report.summary,
            "hotspots": report.hotspots,
            "trends": report.trends,
            "comparisons": report.comparisons,
            "recommendations": report.recommendations,
            "metadata": report.metadata
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _export_csv(self, report: PerformanceReport, output_path: str) -> None:
        """Export report as CSV."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Summary section
            writer.writerow(["Section", "Metric", "Value"])
            writer.writerow(["Summary", "Total Functions", report.summary.get("total_functions", 0)])
            writer.writerow(["Summary", "Total Measurements", report.summary.get("total_measurements", 0)])
            writer.writerow(["Summary", "Performance Score", report.summary.get("performance_score", 0)])
            writer.writerow([])

            # Hotspots section
            writer.writerow(["Hotspots"])
            writer.writerow(["Function", "Total Time", "Avg Time", "Call Count", "Severity"])
            for hotspot in report.hotspots:
                writer.writerow([
                    hotspot["function_name"],
                    hotspot["total_time"],
                    hotspot["avg_time"],
                    hotspot["call_count"],
                    hotspot["severity"]
                ])
            writer.writerow([])

            # Recommendations section
            writer.writerow(["Recommendations"])
            for rec in report.recommendations:
                writer.writerow([rec])

    def _export_html(self, report: PerformanceReport, output_path: str) -> None:
        """Export report as HTML."""
        template_vars = {
            "report": report,
            "timestamp": datetime.fromtimestamp(report.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "summary": report.summary,
            "hotspots": report.hotspots,
            "trends": report.trends,
            "recommendations": report.recommendations
        }

        html_content = self.html_template.render(**template_vars)

        with open(output_path, 'w') as f:
            f.write(html_content)

    def _load_html_template(self) -> Template:
        """Load HTML template for reports."""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Report - {{ report.report_id }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .hotspot { background: #fff; border: 1px solid #ddd; padding: 10px; margin: 5px 0; }
                .high-severity { border-left: 5px solid #ff4444; }
                .medium-severity { border-left: 5px solid #ffaa44; }
                .low-severity { border-left: 5px solid #44ff44; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Report</h1>
                <p><strong>Report ID:</strong> {{ report.report_id }}</p>
                <p><strong>Generated:</strong> {{ timestamp }}</p>
                <p><strong>Duration:</strong> {{ "%.2f"|format(report.duration) }} seconds</p>
            </div>

            <div class="section">
                <h2>Summary</h2>
                <table>
                    <tr><td>Total Functions</td><td>{{ summary.total_functions }}</td></tr>
                    <tr><td>Total Measurements</td><td>{{ summary.total_measurements }}</td></tr>
                    <tr><td>Performance Score</td><td>{{ "%.1f"|format(summary.performance_score) }}</td></tr>
                    <tr><td>System Health</td><td>{{ "%.1f"|format(summary.system_health) }}</td></tr>
                    <tr><td>Anomaly Count</td><td>{{ summary.anomaly_count }}</td></tr>
                </table>
            </div>

            <div class="section">
                <h2>Performance Hotspots</h2>
                {% for hotspot in hotspots %}
                <div class="hotspot {{ hotspot.severity }}-severity">
                    <h3>{{ hotspot.function_name }}</h3>
                    <p><strong>Total Time:</strong> {{ "%.4f"|format(hotspot.total_time) }}s</p>
                    <p><strong>Average Time:</strong> {{ "%.6f"|format(hotspot.avg_time) }}s</p>
                    <p><strong>Call Count:</strong> {{ hotspot.call_count }}</p>
                    <p><strong>Severity:</strong> {{ hotspot.severity|upper }}</p>
                    <p><strong>Optimization Potential:</strong> {{ hotspot.optimization_potential|upper }}</p>
                    {% if hotspot.recommendations %}
                    <p><strong>Recommendations:</strong></p>
                    <ul>
                        {% for rec in hotspot.recommendations %}
                        <li>{{ rec }}</li>
                        {% endfor %}
                    </ul>
                    {% endif %}
                </div>
                {% endfor %}
            </div>

            {% if recommendations %}
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {% for rec in recommendations %}
                    <li>{{ rec }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </body>
        </html>
        """

        return Template(template_str)

    def get_report_history(self, limit: int = 10) -> List[PerformanceReport]:
        """Get recent performance reports."""
        return self.report_history[-limit:] if self.report_history else []

    def cleanup_old_reports(self, max_age_days: int = 30) -> int:
        """Clean up old performance reports."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        # Remove from history
        old_count = 0
        self.report_history = [
            report for report in self.report_history
            if report.timestamp >= cutoff_time or (old_count := old_count + 1)
        ]

        # Remove from reports dict
        to_remove = [
            report_id for report_id, report in self.reports.items()
            if report.timestamp < cutoff_time
        ]

        for report_id in to_remove:
            del self.reports[report_id]

        logger.info(f"Cleaned up {len(to_remove)} old performance reports")
        return len(to_remove)


# Global report generator instance
_report_generator: Optional[PerformanceReportGenerator] = None


def get_performance_report_generator() -> PerformanceReportGenerator:
    """Get the global performance report generator instance."""
    global _report_generator
    if _report_generator is None:
        _report_generator = PerformanceReportGenerator({})
    return _report_generator


def create_performance_report_generator(config: Optional[Dict[str, Any]] = None) -> PerformanceReportGenerator:
    """Create a new performance report generator instance."""
    return PerformanceReportGenerator(config or {})
