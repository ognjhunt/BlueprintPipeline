"""
BlueprintPipeline Job Registry.

Provides centralized tracking of all pipeline jobs, their deprecation status,
and ZeroScene transition state.

This registry is the source of truth for:
- Which jobs are deprecated (replaced by ZeroScene)
- Which jobs are kept (still required for SimReady output)
- Which jobs serve as fallbacks during the transition
- Pipeline routing based on available capabilities

Usage:
    from tools.job_registry import JobRegistry, JobStatus, PipelineMode

    registry = JobRegistry()

    # Check if a job is deprecated
    if registry.is_deprecated("seg-job"):
        print("seg-job is deprecated, use zeroscene-job instead")

    # Get active jobs for current pipeline mode
    jobs = registry.get_active_jobs(PipelineMode.ZEROSCENE_FIRST)

    # Check pipeline readiness
    if registry.is_zeroscene_ready():
        print("ZeroScene pipeline is ready")
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
