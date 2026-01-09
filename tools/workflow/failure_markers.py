"""
Enhanced failure markers for workflow jobs.

Provides rich error context for debugging failed pipeline jobs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FailureContext:
    """Rich context for a pipeline failure."""

    # Basic info
    scene_id: str
    job_name: str
    status: str = "failed"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Error details
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None

    # Job context
    failed_step: Optional[str] = None
    execution_id: Optional[str] = None
    workflow_execution_id: Optional[str] = None
    attempt_number: int = 1

    # Input parameters
    input_params: Dict[str, Any] = field(default_factory=dict)

    # Partial results (if any)
    partial_results: Dict[str, Any] = field(default_factory=dict)

    # Environment info
    environment: Dict[str, Any] = field(default_factory=dict)

    # Logs and debugging
    logs_url: Optional[str] = None
    debug_artifacts: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scene_id": self.scene_id,
            "job_name": self.job_name,
            "status": self.status,
            "timestamp": self.timestamp,
            "error": {
                "code": self.error_code,
                "message": self.error_message,
                "type": self.error_type,
                "stack_trace": self.stack_trace,
            },
            "context": {
                "failed_step": self.failed_step,
                "execution_id": self.execution_id,
                "workflow_execution_id": self.workflow_execution_id,
                "attempt_number": self.attempt_number,
            },
            "input_params": self.input_params,
            "partial_results": self.partial_results,
            "environment": self.environment,
            "debugging": {
                "logs_url": self.logs_url,
                "debug_artifacts": self.debug_artifacts,
            },
            "recommendations": self.recommendations,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class FailureMarkerWriter:
    """
    Writes enhanced failure markers to GCS.

    Example:
        writer = FailureMarkerWriter(
            bucket="my-bucket",
            scene_id="kitchen_001",
            job_name="genie-sim-export-job",
        )

        try:
            process_scene()
        except Exception as e:
            writer.write_failure(
                exception=e,
                failed_step="convert_scene_graph",
                input_params={"robot_type": "franka"},
                partial_results={"objects_processed": 50},
            )
            raise
    """

    def __init__(
        self,
        bucket: str,
        scene_id: str,
        job_name: str,
        base_path: Optional[str] = None,
    ):
        self.bucket = bucket
        self.scene_id = scene_id
        self.job_name = job_name
        self.base_path = base_path or f"scenes/{scene_id}/geniesim"

    def write_failure(
        self,
        exception: Exception,
        failed_step: Optional[str] = None,
        input_params: Optional[Dict[str, Any]] = None,
        partial_results: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[str]] = None,
    ) -> Path:
        """
        Write enhanced failure marker.

        Args:
            exception: The exception that caused the failure
            failed_step: Name of the step that failed
            input_params: Input parameters for the job
            partial_results: Any partial results before failure
            recommendations: Recommendations for fixing the issue

        Returns:
            Path to the written failure marker

        Example:
            try:
                result = process_objects(objects)
            except ValueError as e:
                writer.write_failure(
                    exception=e,
                    failed_step="validate_objects",
                    input_params={"num_objects": len(objects)},
                    recommendations=[
                        "Check that all objects have valid dimensions",
                        "Verify object categories are supported",
                    ],
                )
                raise
        """
        # Build failure context
        context = self._build_context(
            exception=exception,
            failed_step=failed_step,
            input_params=input_params or {},
            partial_results=partial_results or {},
            recommendations=recommendations or [],
        )

        # Write to GCS
        marker_path = self._write_to_gcs(context)

        logger.info(
            f"Wrote enhanced failure marker to {marker_path}",
            extra=context.to_dict(),
        )

        return marker_path

    def _build_context(
        self,
        exception: Exception,
        failed_step: Optional[str],
        input_params: Dict[str, Any],
        partial_results: Dict[str, Any],
        recommendations: List[str],
    ) -> FailureContext:
        """Build failure context from exception and parameters."""
        # Extract error details
        error_message = str(exception)
        error_type = type(exception).__name__
        stack_trace = ''.join(traceback.format_exception(
            type(exception),
            exception,
            exception.__traceback__,
        ))

        # Get environment info
        environment = {
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": str(Path.cwd()),
        }

        # Add Cloud Run specific env vars
        cloud_run_vars = [
            "K_SERVICE",
            "K_REVISION",
            "K_CONFIGURATION",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_REGION",
        ]
        for var in cloud_run_vars:
            if var in os.environ:
                environment[var.lower()] = os.environ[var]

        # Generate logs URL if in GCP
        logs_url = None
        if "GOOGLE_CLOUD_PROJECT" in os.environ:
            project = os.environ["GOOGLE_CLOUD_PROJECT"]
            logs_url = (
                f"https://console.cloud.google.com/logs/query"
                f"?project={project}"
                f"&query=resource.labels.job_name%3D%22{self.job_name}%22"
            )

        # Add recommendations based on error type
        auto_recommendations = self._generate_recommendations(exception)
        all_recommendations = recommendations + auto_recommendations

        return FailureContext(
            scene_id=self.scene_id,
            job_name=self.job_name,
            error_message=error_message,
            error_type=error_type,
            stack_trace=stack_trace,
            failed_step=failed_step,
            execution_id=os.getenv("CLOUD_RUN_EXECUTION", "unknown"),
            workflow_execution_id=os.getenv("GOOGLE_CLOUD_WORKFLOW_EXECUTION_ID"),
            attempt_number=int(os.getenv("CLOUD_RUN_TASK_ATTEMPT", "1")),
            input_params=input_params,
            partial_results=partial_results,
            environment=environment,
            logs_url=logs_url,
            recommendations=all_recommendations,
        )

    def _generate_recommendations(self, exception: Exception) -> List[str]:
        """Generate automatic recommendations based on error type."""
        recommendations = []

        error_msg = str(exception).lower()

        if "timeout" in error_msg:
            recommendations.append(
                "Increase timeout values for long-running operations"
            )
            recommendations.append(
                "Check for hanging processes or infinite loops"
            )

        if "memory" in error_msg or "oom" in error_msg:
            recommendations.append(
                "Increase memory limits for this job"
            )
            recommendations.append(
                "Use streaming/batching for large datasets"
            )

        if "connection" in error_msg or "network" in error_msg:
            recommendations.append(
                "Check network connectivity and firewall rules"
            )
            recommendations.append(
                "Verify API endpoints are accessible"
            )

        if "permission" in error_msg or "forbidden" in error_msg:
            recommendations.append(
                "Check IAM permissions for service account"
            )
            recommendations.append(
                "Verify resource access (GCS buckets, secrets, etc.)"
            )

        if "not found" in error_msg:
            recommendations.append(
                "Verify input files/resources exist"
            )
            recommendations.append(
                "Check file paths and GCS URIs"
            )

        if "validation" in error_msg or "invalid" in error_msg:
            recommendations.append(
                "Review input data format and schema"
            )
            recommendations.append(
                "Check for missing required fields"
            )

        return recommendations

    def _write_to_gcs(self, context: FailureContext) -> Path:
        """Write failure context to GCS."""
        try:
            from google.cloud import storage
        except ImportError:
            # Fallback to local file if GCS not available
            return self._write_to_local(context)

        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket)

            # Write detailed failure marker
            blob_path = f"{self.base_path}/.geniesim_failed"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(
                context.to_json(),
                content_type="application/json",
            )

            # Also write a simple marker for backward compatibility
            simple_marker_path = f"{self.base_path}/.failed"
            simple_blob = bucket.blob(simple_marker_path)
            simple_blob.upload_from_string(
                json.dumps({
                    "scene_id": self.scene_id,
                    "job_name": self.job_name,
                    "status": "failed",
                    "timestamp": context.timestamp,
                    "error": context.error_message,
                }),
                content_type="application/json",
            )

            return Path(f"gs://{self.bucket}/{blob_path}")

        except Exception as e:
            logger.error(f"Failed to write to GCS: {e}")
            return self._write_to_local(context)

    def _write_to_local(self, context: FailureContext) -> Path:
        """Fallback: write failure context to local file."""
        local_path = Path("/tmp") / f"{self.scene_id}_failure.json"

        try:
            with open(local_path, 'w') as f:
                f.write(context.to_json())

            logger.info(f"Wrote failure marker to local file: {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to write local failure marker: {e}")
            raise


def write_failure_marker(
    bucket: str,
    scene_id: str,
    job_name: str,
    exception: Exception,
    **kwargs,
) -> None:
    """
    Convenience function to write a failure marker.

    Args:
        bucket: GCS bucket name
        scene_id: Scene ID
        job_name: Name of the job that failed
        exception: The exception that occurred
        **kwargs: Additional arguments passed to FailureMarkerWriter.write_failure

    Example:
        try:
            process_scene()
        except Exception as e:
            write_failure_marker(
                bucket="my-bucket",
                scene_id="kitchen_001",
                job_name="genie-sim-export-job",
                exception=e,
                failed_step="export_scene_graph",
                input_params={"robot_type": "franka"},
            )
            raise
    """
    writer = FailureMarkerWriter(bucket, scene_id, job_name)
    writer.write_failure(exception, **kwargs)
