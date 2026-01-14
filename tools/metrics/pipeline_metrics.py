"""Real-Time Pipeline Metrics & Observability.

Provides real-time metrics collection for pipeline monitoring using
Cloud Monitoring (Google Cloud) or Prometheus, complementing the
existing success_metrics.py which tracks business outcomes.

This module focuses on:
- Real-time counters (pipeline runs, API calls, objects processed)
- Timing histograms (job duration, API latency)
- Quality score distributions
- Error rates and failure tracking
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class MetricsBackend(str, Enum):
    """Supported metrics backends."""
    CLOUD_MONITORING = "cloud_monitoring"
    PROMETHEUS = "prometheus"
    IN_MEMORY = "in_memory"  # For testing/local development


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"


# Protocol for metrics implementations
class MetricProtocol(Protocol):
    """Protocol that all metric implementations must follow."""

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        ...

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value for histogram metric."""
        ...

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        ...


@dataclass
class InMemoryMetric:
    """In-memory metric for testing/local development."""
    name: str
    description: str
    metric_type: MetricType
    unit: str = ""

    # Storage
    values: List[float] = field(default_factory=list)
    labels_history: List[Dict[str, str]] = field(default_factory=list)

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter."""
        self.values.append(value)
        self.labels_history.append(labels or {})

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe histogram value."""
        self.values.append(value)
        self.labels_history.append(labels or {})

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge value."""
        self.values.append(value)
        self.labels_history.append(labels or {})

    def get_total(self) -> float:
        """Get total (for counters)."""
        return sum(self.values)

    def get_latest(self) -> Optional[float]:
        """Get latest value (for gauges)."""
        return self.values[-1] if self.values else None

    def get_stats(self) -> Dict[str, float]:
        """Get statistics (for histograms)."""
        if not self.values:
            return {}

        sorted_vals = sorted(self.values)
        return {
            "count": len(self.values),
            "sum": sum(self.values),
            "min": min(self.values),
            "max": max(self.values),
            "mean": sum(self.values) / len(self.values),
            "p50": sorted_vals[int(len(sorted_vals) * 0.5)],
            "p95": sorted_vals[int(len(sorted_vals) * 0.95)],
            "p99": sorted_vals[int(len(sorted_vals) * 0.99)],
        }


class CloudMonitoringMetric:
    """Cloud Monitoring (Google Cloud) metric."""

    def __init__(
        self,
        name: str,
        description: str,
        metric_type: MetricType,
        unit: str = "",
        project_id: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.metric_type = metric_type
        self.unit = unit

        # Initialize Cloud Monitoring client if available
        try:
            from google.cloud import monitoring_v3

            self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
            self.client = monitoring_v3.MetricServiceClient()
            self.project_name = f"projects/{self.project_id}"

            # Create custom metric descriptor
            self._create_metric_descriptor()

            self.enabled = True
        except ImportError:
            logger.warning(
                "google-cloud-monitoring not installed. "
                "Install with: pip install google-cloud-monitoring"
            )
            self.enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize Cloud Monitoring: {e}")
            self.enabled = False

    def _create_metric_descriptor(self) -> None:
        """Create custom metric descriptor."""
        if not self.enabled:
            return

        try:
            from google.cloud import monitoring_v3
            from google.api import metric_pb2

            descriptor = metric_pb2.MetricDescriptor()
            descriptor.type = f"custom.googleapis.com/blueprint_pipeline/{self.name}"
            descriptor.display_name = self.name
            descriptor.description = self.description

            # Map metric type
            if self.metric_type == MetricType.COUNTER:
                descriptor.metric_kind = metric_pb2.MetricDescriptor.MetricKind.CUMULATIVE
                descriptor.value_type = metric_pb2.MetricDescriptor.ValueType.INT64
            elif self.metric_type == MetricType.GAUGE:
                descriptor.metric_kind = metric_pb2.MetricDescriptor.MetricKind.GAUGE
                descriptor.value_type = metric_pb2.MetricDescriptor.ValueType.DOUBLE
            else:  # HISTOGRAM
                descriptor.metric_kind = metric_pb2.MetricDescriptor.MetricKind.CUMULATIVE
                descriptor.value_type = metric_pb2.MetricDescriptor.ValueType.DISTRIBUTION

            if self.unit:
                descriptor.unit = self.unit

            # Create or update
            self.client.create_metric_descriptor(
                name=self.project_name,
                metric_descriptor=descriptor
            )
        except Exception as e:
            # Descriptor might already exist
            logger.debug(f"Metric descriptor creation: {e}")

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter."""
        if not self.enabled:
            return

        self._write_point(value, labels)

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe histogram value."""
        if not self.enabled:
            return

        self._write_point(value, labels)

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge value."""
        if not self.enabled:
            return

        self._write_point(value, labels)

    def _write_point(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Write time series point."""
        try:
            from google.cloud import monitoring_v3
            from google.api import metric_pb2
            import google.protobuf.timestamp_pb2

            series = monitoring_v3.TimeSeries()
            series.metric.type = f"custom.googleapis.com/blueprint_pipeline/{self.name}"

            # Add labels
            if labels:
                for key, val in labels.items():
                    series.metric.labels[key] = str(val)

            # Create point
            point = monitoring_v3.Point()
            point.value.double_value = value

            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 10**9)

            point.interval.end_time.seconds = seconds
            point.interval.end_time.nanos = nanos

            series.points = [point]

            # Write
            self.client.create_time_series(
                name=self.project_name,
                time_series=[series]
            )
        except Exception as e:
            logger.warning(f"Failed to write metric {self.name}: {e}")


class PrometheusMetric:
    """Prometheus metric."""

    def __init__(
        self,
        name: str,
        description: str,
        metric_type: MetricType,
        unit: str = "",
    ):
        self.name = name
        self.description = description
        self.metric_type = metric_type
        self.unit = unit

        # Initialize Prometheus client if available
        try:
            from prometheus_client import Counter, Histogram, Gauge

            # Create metric based on type
            if metric_type == MetricType.COUNTER:
                self.metric = Counter(
                    name,
                    description,
                    labelnames=['job', 'scene_id', 'status']
                )
            elif metric_type == MetricType.HISTOGRAM:
                self.metric = Histogram(
                    name,
                    description,
                    labelnames=['job', 'scene_id']
                )
            else:  # GAUGE
                self.metric = Gauge(
                    name,
                    description,
                    labelnames=['job', 'scene_id']
                )

            self.enabled = True
        except ImportError:
            logger.warning(
                "prometheus_client not installed. "
                "Install with: pip install prometheus_client"
            )
            self.enabled = False

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter."""
        if not self.enabled:
            return

        if labels:
            self.metric.labels(**labels).inc(value)
        else:
            self.metric.inc(value)

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe histogram value."""
        if not self.enabled:
            return

        if labels:
            self.metric.labels(**labels).observe(value)
        else:
            self.metric.observe(value)

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge value."""
        if not self.enabled:
            return

        if labels:
            self.metric.labels(**labels).set(value)
        else:
            self.metric.set(value)


class PipelineMetricsCollector:
    """Centralized metrics for pipeline monitoring.

    Provides real-time observability for the BlueprintPipeline with support
    for Cloud Monitoring, Prometheus, or in-memory storage.

    Example:
        # Initialize (auto-detects backend)
        metrics = PipelineMetricsCollector()

        # Track a job
        with metrics.track_job("regen3d-job", "scene_001"):
            # Process scene
            metrics.objects_processed.inc(15, labels={"scene_id": "scene_001"})

            # Track API call
            metrics.gemini_calls.inc(labels={"scene_id": "scene_001"})
    """

    def __init__(
        self,
        backend: Optional[MetricsBackend] = None,
        project_id: Optional[str] = None,
        enable_logging: bool = True,
    ):
        """Initialize metrics collector.

        Args:
            backend: Metrics backend to use (auto-detected if None)
            project_id: GCP project ID (for Cloud Monitoring)
            enable_logging: Whether to log metrics to stdout
        """
        self.backend = backend or self._detect_backend()
        self.project_id = project_id
        self.enable_logging = enable_logging

        # Initialize metrics
        self._init_metrics()

        logger.info(f"Initialized metrics collector with backend: {self.backend}")

    def _detect_backend(self) -> MetricsBackend:
        """Auto-detect metrics backend."""
        # Check for Cloud Monitoring
        if os.getenv("GCP_PROJECT_ID"):
            try:
                import google.cloud.monitoring_v3
                return MetricsBackend.CLOUD_MONITORING
            except ImportError:
                pass

        # Check for Prometheus
        try:
            import prometheus_client
            return MetricsBackend.PROMETHEUS
        except ImportError:
            pass

        # Fallback to in-memory
        logger.warning("No metrics backend detected, using in-memory storage")
        return MetricsBackend.IN_MEMORY

    def _create_metric(
        self,
        name: str,
        description: str,
        metric_type: MetricType,
        unit: str = "",
    ) -> MetricProtocol:
        """Create a metric based on backend."""
        if self.backend == MetricsBackend.CLOUD_MONITORING:
            return CloudMonitoringMetric(name, description, metric_type, unit, self.project_id)
        elif self.backend == MetricsBackend.PROMETHEUS:
            return PrometheusMetric(name, description, metric_type, unit)
        else:
            return InMemoryMetric(name, description, metric_type, unit)

    def _init_metrics(self) -> None:
        """Initialize all pipeline metrics."""
        # Pipeline execution metrics
        self.pipeline_runs_total = self._create_metric(
            "pipeline_runs_total",
            "Total number of pipeline runs",
            MetricType.COUNTER,
        )

        self.pipeline_duration_seconds = self._create_metric(
            "pipeline_duration_seconds",
            "Pipeline job duration in seconds",
            MetricType.HISTOGRAM,
            "s",
        )

        # Object processing metrics
        self.objects_processed = self._create_metric(
            "objects_processed_total",
            "Total number of objects processed",
            MetricType.COUNTER,
        )

        self.objects_failed = self._create_metric(
            "objects_failed_total",
            "Total number of objects that failed processing",
            MetricType.COUNTER,
        )

        # API call metrics
        self.gemini_calls = self._create_metric(
            "gemini_api_calls_total",
            "Total number of Gemini API calls",
            MetricType.COUNTER,
        )

        self.gemini_tokens_input = self._create_metric(
            "gemini_tokens_input_total",
            "Total Gemini input tokens",
            MetricType.COUNTER,
        )

        self.gemini_tokens_output = self._create_metric(
            "gemini_tokens_output_total",
            "Total Gemini output tokens",
            MetricType.COUNTER,
        )

        self.gemini_latency_seconds = self._create_metric(
            "gemini_api_latency_seconds",
            "Gemini API call latency",
            MetricType.HISTOGRAM,
            "s",
        )

        # Genie Sim metrics
        self.geniesim_jobs = self._create_metric(
            "geniesim_jobs_submitted_total",
            "Total number of Genie Sim jobs submitted",
            MetricType.COUNTER,
        )

        self.geniesim_episodes_generated = self._create_metric(
            "geniesim_episodes_generated_total",
            "Total number of episodes generated by Genie Sim",
            MetricType.COUNTER,
        )

        self.geniesim_job_duration_seconds = self._create_metric(
            "geniesim_job_duration_seconds",
            "Genie Sim job duration",
            MetricType.HISTOGRAM,
            "s",
        )

        self.geniesim_api_latency_seconds = self._create_metric(
            "geniesim_api_latency_seconds",
            "Genie Sim API call latency",
            MetricType.HISTOGRAM,
            "s",
        )

        # Quality metrics
        self.episode_quality_score = self._create_metric(
            "episode_quality_score",
            "Episode quality scores",
            MetricType.HISTOGRAM,
        )

        self.physics_validation_score = self._create_metric(
            "physics_validation_score",
            "Physics validation scores",
            MetricType.HISTOGRAM,
        )

        self.sensor_capture_source_total = self._create_metric(
            "sensor_capture_source_total",
            "Sensor capture source counts",
            MetricType.COUNTER,
        )

        # Resource metrics
        self.scenes_in_progress = self._create_metric(
            "scenes_in_progress",
            "Number of scenes currently being processed",
            MetricType.GAUGE,
        )

        self.storage_bytes_written = self._create_metric(
            "storage_bytes_written_total",
            "Total bytes written to storage",
            MetricType.COUNTER,
            "By",
        )

        # Error metrics
        self.errors_total = self._create_metric(
            "errors_total",
            "Total number of errors",
            MetricType.COUNTER,
        )

        self.retries_total = self._create_metric(
            "retries_total",
            "Total number of retries",
            MetricType.COUNTER,
        )

    @contextmanager
    def track_job(self, job_name: str, scene_id: str):
        """Context manager to track job execution.

        Args:
            job_name: Name of the job (e.g., "regen3d-job")
            scene_id: Scene identifier

        Example:
            with metrics.track_job("regen3d-job", "scene_001"):
                # Process scene
                pass
        """
        start = time.time()
        labels = {"job": job_name, "scene_id": scene_id, "status": "success"}

        # Increment in-progress gauge
        self.scenes_in_progress.set(1, labels={"job": job_name})

        try:
            yield

            # Success
            self.pipeline_runs_total.inc(labels=labels)

            if self.enable_logging:
                logger.info(f"[METRICS] Job {job_name} completed for scene {scene_id}")

        except Exception as e:
            # Failure
            labels["status"] = "failure"
            self.pipeline_runs_total.inc(labels=labels)
            self.errors_total.inc(labels={
                "job": job_name,
                "scene_id": scene_id,
                "error_type": type(e).__name__
            })

            if self.enable_logging:
                logger.error(f"[METRICS] Job {job_name} failed for scene {scene_id}: {e}")

            raise

        finally:
            # Record duration
            duration = time.time() - start
            self.pipeline_duration_seconds.observe(
                duration,
                labels={"job": job_name, "scene_id": scene_id}
            )

            # Decrement in-progress gauge
            self.scenes_in_progress.set(0, labels={"job": job_name})

            if self.enable_logging:
                logger.info(f"[METRICS] Job {job_name} took {duration:.2f}s")

    @contextmanager
    def track_api_call(self, api_name: str, operation: str, scene_id: str = ""):
        """Context manager to track API call latency.

        Args:
            api_name: API name (e.g., "gemini", "geniesim")
            operation: Operation name (e.g., "generate_physics", "submit_job")
            scene_id: Optional scene identifier

        Example:
            with metrics.track_api_call("gemini", "generate_physics", "scene_001"):
                # Make API call
                pass
        """
        start = time.time()
        labels = {"api": api_name, "operation": operation}
        if scene_id:
            labels["scene_id"] = scene_id

        try:
            yield
        finally:
            latency = time.time() - start

            if api_name == "gemini":
                self.gemini_latency_seconds.observe(latency, labels)
            elif api_name == "geniesim":
                self.geniesim_api_latency_seconds.observe(latency, labels)

    def track_gemini_call(
        self,
        scene_id: str,
        tokens_input: int,
        tokens_output: int,
        operation: str = "",
    ) -> None:
        """Track a Gemini API call.

        Args:
            scene_id: Scene identifier
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            operation: Optional operation name
        """
        labels = {"scene_id": scene_id}
        if operation:
            labels["operation"] = operation

        self.gemini_calls.inc(labels=labels)
        self.gemini_tokens_input.inc(tokens_input, labels=labels)
        self.gemini_tokens_output.inc(tokens_output, labels=labels)

        if self.enable_logging:
            logger.info(
                f"[METRICS] Gemini call: {tokens_input} in, {tokens_output} out "
                f"(scene: {scene_id})"
            )

    def track_geniesim_job(
        self,
        scene_id: str,
        job_id: str,
        duration_seconds: float,
        episode_count: int,
    ) -> None:
        """Track a Genie Sim job.

        Args:
            scene_id: Scene identifier
            job_id: Genie Sim job ID
            duration_seconds: Job duration
            episode_count: Number of episodes generated
        """
        labels = {"scene_id": scene_id, "job_id": job_id}

        self.geniesim_jobs.inc(labels=labels)
        self.geniesim_episodes_generated.inc(episode_count, labels=labels)
        self.geniesim_job_duration_seconds.observe(duration_seconds, labels=labels)

        if self.enable_logging:
            logger.info(
                f"[METRICS] Genie Sim job {job_id}: {episode_count} episodes "
                f"in {duration_seconds:.2f}s (scene: {scene_id})"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics (only for in-memory backend).

        Returns:
            Dictionary of metric statistics
        """
        if self.backend != MetricsBackend.IN_MEMORY:
            return {"error": "Stats only available for in-memory backend"}

        stats = {}

        # Collect stats from all metrics
        for name, metric in self.__dict__.items():
            if isinstance(metric, InMemoryMetric):
                if metric.metric_type == MetricType.COUNTER:
                    stats[name] = metric.get_total()
                elif metric.metric_type == MetricType.GAUGE:
                    stats[name] = metric.get_latest()
                elif metric.metric_type == MetricType.HISTOGRAM:
                    stats[name] = metric.get_stats()

        return stats

    def reset(self) -> None:
        """Reset all metrics (only for in-memory backend)."""
        if self.backend != MetricsBackend.IN_MEMORY:
            logger.warning("Reset only available for in-memory backend")
            return

        for metric in self.__dict__.values():
            if isinstance(metric, InMemoryMetric):
                metric.values.clear()
                metric.labels_history.clear()


# Global singleton instance
_metrics: Optional[PipelineMetricsCollector] = None


def get_metrics() -> PipelineMetricsCollector:
    """Get global metrics instance.

    Returns:
        Global PipelineMetricsCollector instance

    Example:
        from tools.metrics.pipeline_metrics import get_metrics

        metrics = get_metrics()
        with metrics.track_job("regen3d-job", "scene_001"):
            # Process scene
            pass
    """
    global _metrics
    if _metrics is None:
        _metrics = PipelineMetricsCollector()
    return _metrics


def reset_metrics() -> None:
    """Reset global metrics instance (for testing)."""
    global _metrics
    if _metrics:
        _metrics.reset()
    _metrics = None
