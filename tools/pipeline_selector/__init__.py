"""
Pipeline Selector for BlueprintPipeline.

Provides routing logic for choosing between 3D-RE-GEN and Gemini pipelines,
with automatic fallback handling.

Usage:
    from tools.pipeline_selector import PipelineSelector, select_pipeline

    # Auto-select pipeline based on environment and availability
    selector = PipelineSelector()
    pipeline = selector.select()

    # Get job sequence for selected pipeline
    jobs = selector.get_job_sequence()

    # Check if 3D-RE-GEN output exists
    if selector.has_regen3d_output(scene_dir):
        jobs = selector.get_regen3d_jobs()
    else:
        jobs = selector.get_gemini_jobs()
"""

from .selector import (
    PipelineSelector,
    select_pipeline,
    should_skip_deprecated_job,
    get_active_pipeline_mode,
)

__all__ = [
    "PipelineSelector",
    "select_pipeline",
    "should_skip_deprecated_job",
    "get_active_pipeline_mode",
]
