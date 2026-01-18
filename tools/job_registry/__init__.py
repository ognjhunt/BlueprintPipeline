"""
BlueprintPipeline Job Registry.

Provides centralized tracking of all pipeline jobs for the 3D-RE-GEN-based pipeline.

This registry is the source of truth for:
- Which jobs are active in the 3D-RE-GEN pipeline
- Job dependencies and execution order
- Pipeline routing and job metadata

Usage:
    from tools.job_registry import JobRegistry, JobStatus, PipelineMode
    from tools.job_registry import JOBS, JOBS_BY_NAME

    registry = JobRegistry()

    # Get active jobs
    jobs = registry.get_active_jobs()

    # Check pipeline readiness
    if registry.is_regen3d_ready():
        print("3D-RE-GEN pipeline is ready")

    # Get job sequence
    sequence = registry.get_job_sequence()

    # Or access a stable list of jobs without instantiating JobRegistry
    jobs = JOBS
    job = JOBS_BY_NAME["regen3d-job"]
"""

from .registry import (
    JobStatus,
    JobCategory,
    PipelineMode,
    JobInfo,
    JobRegistry,
    get_registry,
    JOBS,
    JOBS_BY_NAME,
)

__all__ = [
    "JobStatus",
    "JobCategory",
    "PipelineMode",
    "JobInfo",
    "JobRegistry",
    "get_registry",
    "JOBS",
    "JOBS_BY_NAME",
]
