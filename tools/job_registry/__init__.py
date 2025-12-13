"""
BlueprintPipeline Job Registry.

Provides centralized tracking of all pipeline jobs for the ZeroScene-based pipeline.

This registry is the source of truth for:
- Which jobs are active in the ZeroScene pipeline
- Job dependencies and execution order
- Pipeline routing and job metadata

Usage:
    from tools.job_registry import JobRegistry, JobStatus, PipelineMode

    registry = JobRegistry()

    # Get active jobs
    jobs = registry.get_active_jobs()

    # Check pipeline readiness
    if registry.is_zeroscene_ready():
        print("ZeroScene pipeline is ready")

    # Get job sequence
    sequence = registry.get_job_sequence()
"""

from .registry import (
    JobStatus,
    JobCategory,
    PipelineMode,
    JobInfo,
    JobRegistry,
    get_registry,
)

__all__ = [
    "JobStatus",
    "JobCategory",
    "PipelineMode",
    "JobInfo",
    "JobRegistry",
    "get_registry",
]
