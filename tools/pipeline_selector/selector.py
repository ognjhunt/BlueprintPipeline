"""
Pipeline Selector Implementation.

Handles routing between ZeroScene and Gemini pipelines with automatic fallback.

Environment Variables:
    PIPELINE_MODE: "zeroscene_first" | "gemini_only" | "hybrid" (default: zeroscene_first)
    ZEROSCENE_AVAILABLE: "true" | "false" (override auto-detection)
    FORCE_DEPRECATED_JOB: "true" to force run deprecated jobs
    AUTO_FALLBACK: "true" to auto-fallback to Gemini if ZeroScene fails (default: true)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any


class PipelineMode(str, Enum):
    """Pipeline execution mode."""
    ZEROSCENE_FIRST = "zeroscene_first"  # Use ZeroScene when available, fallback to Gemini
    GEMINI_ONLY = "gemini_only"          # Use only the Gemini reconstruction pipeline
    HYBRID = "hybrid"                     # Use both and compare (development mode)


@dataclass
class PipelineDecision:
    """Result of pipeline selection."""
    mode: PipelineMode
    use_zeroscene: bool
    use_gemini_fallback: bool
    reason: str
    job_sequence: List[str]


class PipelineSelector:
    """Selects and routes between ZeroScene and Gemini pipelines.

    The selector determines which pipeline to use based on:
    1. Environment configuration (PIPELINE_MODE)
    2. ZeroScene output availability
    3. Fallback preferences
    """

    # ZeroScene output markers
    ZEROSCENE_MARKERS = [
        "zeroscene/scene_info.json",
        "zeroscene/objects",
    ]

    # Gemini pipeline output markers
    GEMINI_MARKERS = [
        "seg/inventory.json",
        "layout/scene_layout.json",
    ]

    def __init__(self, scene_root: Optional[Path] = None):
        """Initialize the pipeline selector.

        Args:
            scene_root: Root directory for scene data (for detecting outputs)
        """
        self.scene_root = Path(scene_root) if scene_root else None

    def get_mode(self) -> PipelineMode:
        """Get the configured pipeline mode."""
        mode_str = os.getenv("PIPELINE_MODE", "zeroscene_first").lower()

        if mode_str == "gemini_only":
            return PipelineMode.GEMINI_ONLY
        elif mode_str == "hybrid":
            return PipelineMode.HYBRID
        else:
            return PipelineMode.ZEROSCENE_FIRST

    def is_zeroscene_available(self) -> bool:
        """Check if ZeroScene is available (external service or local)."""
        # Check environment override
        override = os.getenv("ZEROSCENE_AVAILABLE", "").lower()
        if override == "true":
            return True
        if override == "false":
            return False

        # ZeroScene paper is published but code may not be released yet
        # https://arxiv.org/html/2509.23607v1
        # For now, check if zeroscene output exists
        if self.scene_root:
            return self.has_zeroscene_output(self.scene_root)

        return False

    def has_zeroscene_output(self, scene_dir: Path) -> bool:
        """Check if ZeroScene output exists for a scene."""
        for marker in self.ZEROSCENE_MARKERS:
            marker_path = scene_dir / marker
            if marker_path.exists():
                return True
        return False

    def has_gemini_output(self, scene_dir: Path) -> bool:
        """Check if Gemini pipeline output exists for a scene."""
        for marker in self.GEMINI_MARKERS:
            marker_path = scene_dir / marker
            if marker_path.exists():
                return True
        return False

    def select(self, scene_dir: Optional[Path] = None) -> PipelineDecision:
        """Select the appropriate pipeline for processing.

        Args:
            scene_dir: Scene directory to check for existing outputs

        Returns:
            PipelineDecision with selected mode and job sequence
        """
        mode = self.get_mode()
        scene_dir = scene_dir or self.scene_root
        auto_fallback = os.getenv("AUTO_FALLBACK", "true").lower() == "true"

        # Gemini-only mode
        if mode == PipelineMode.GEMINI_ONLY:
            return PipelineDecision(
                mode=mode,
                use_zeroscene=False,
                use_gemini_fallback=False,
                reason="PIPELINE_MODE=gemini_only",
                job_sequence=self._get_gemini_jobs(),
            )

        # Hybrid mode (development)
        if mode == PipelineMode.HYBRID:
            return PipelineDecision(
                mode=mode,
                use_zeroscene=True,
                use_gemini_fallback=True,
                reason="PIPELINE_MODE=hybrid (running both pipelines)",
                job_sequence=self._get_hybrid_jobs(),
            )

        # ZeroScene-first mode
        has_zeroscene = scene_dir and self.has_zeroscene_output(scene_dir)
        zeroscene_available = self.is_zeroscene_available()

        if has_zeroscene:
            return PipelineDecision(
                mode=mode,
                use_zeroscene=True,
                use_gemini_fallback=False,
                reason="ZeroScene output exists",
                job_sequence=self._get_zeroscene_jobs(),
            )

        if zeroscene_available:
            return PipelineDecision(
                mode=mode,
                use_zeroscene=True,
                use_gemini_fallback=auto_fallback,
                reason="ZeroScene available, will run reconstruction",
                job_sequence=self._get_zeroscene_jobs(),
            )

        # ZeroScene not available, check if we should fallback
        has_gemini = scene_dir and self.has_gemini_output(scene_dir)

        if has_gemini:
            return PipelineDecision(
                mode=mode,
                use_zeroscene=False,
                use_gemini_fallback=True,
                reason="Using existing Gemini output (ZeroScene not available)",
                job_sequence=self._get_downstream_jobs(),
            )

        if auto_fallback:
            return PipelineDecision(
                mode=mode,
                use_zeroscene=False,
                use_gemini_fallback=True,
                reason="Falling back to Gemini pipeline (ZeroScene not available)",
                job_sequence=self._get_gemini_jobs(),
            )

        # No fallback, ZeroScene required but not available
        return PipelineDecision(
            mode=mode,
            use_zeroscene=False,
            use_gemini_fallback=False,
            reason="ZeroScene required but not available (set AUTO_FALLBACK=true for Gemini)",
            job_sequence=[],
        )

    def _get_zeroscene_jobs(self) -> List[str]:
        """Get job sequence for ZeroScene pipeline."""
        return [
            # ZeroScene reconstruction runs externally
            "zeroscene-job",       # Adapter: converts ZeroScene → BlueprintPipeline
            "scale-job",           # Optional: calibrate scale if needed
            "interactive-job",     # Articulation for doors/drawers
            "simready-job",        # Physics properties
            "usd-assembly-job",    # Assemble scene.usda
            "replicator-job",      # Domain randomization
            "variation-gen-job",   # Variation assets
            "isaac-lab-job",       # Isaac Lab tasks
        ]

    def _get_gemini_jobs(self) -> List[str]:
        """Get job sequence for Gemini pipeline (fallback)."""
        return [
            "seg-job",             # Segmentation + inventory
            "scene-da3-job",       # Depth extraction
            "layout-job",          # Layout reconstruction
            "scale-job",           # Scale calibration
            "multiview-job",       # Object isolation
            "sam3d-job",           # 3D mesh generation
            "hunyuan-job",         # Texture refinement
            "interactive-job",     # Articulation
            "simready-job",        # Physics properties
            "usd-assembly-job",    # Assemble scene.usda
            "replicator-job",      # Domain randomization
            "variation-gen-job",   # Variation assets
            "isaac-lab-job",       # Isaac Lab tasks
        ]

    def _get_downstream_jobs(self) -> List[str]:
        """Get downstream jobs (after reconstruction)."""
        return [
            "interactive-job",
            "simready-job",
            "usd-assembly-job",
            "replicator-job",
            "variation-gen-job",
            "isaac-lab-job",
        ]

    def _get_hybrid_jobs(self) -> List[str]:
        """Get job sequence for hybrid mode."""
        # In hybrid mode, we run ZeroScene and use Gemini for comparison
        return self._get_zeroscene_jobs()

    def should_skip_job(self, job_name: str) -> bool:
        """Check if a job should be skipped based on current pipeline mode."""
        mode = self.get_mode()

        # Jobs deprecated in ZeroScene-first mode
        deprecated_in_zeroscene = {
            "seg-job",
            "multiview-job",
            "scene-da3-job",
            "layout-job",
            "sam3d-job",
            "hunyuan-job",
        }

        if mode == PipelineMode.ZEROSCENE_FIRST:
            if job_name in deprecated_in_zeroscene:
                force = os.getenv("FORCE_DEPRECATED_JOB", "").lower() in ("1", "true", "yes")
                return not force

        return False

    def get_job_env_overrides(self, job_name: str) -> Dict[str, str]:
        """Get environment variable overrides for a job based on pipeline mode."""
        mode = self.get_mode()
        overrides = {}

        # Pass pipeline mode to all jobs
        overrides["PIPELINE_MODE"] = mode.value

        # For ZeroScene mode, set prefixes to ZeroScene output locations
        if mode == PipelineMode.ZEROSCENE_FIRST:
            overrides["USE_ZEROSCENE_OUTPUT"] = "true"

        return overrides


# =============================================================================
# Convenience Functions
# =============================================================================


def select_pipeline(scene_dir: Optional[Path] = None) -> PipelineDecision:
    """Select the appropriate pipeline for a scene.

    Args:
        scene_dir: Optional scene directory to check for existing outputs

    Returns:
        PipelineDecision with mode, job sequence, and reasoning
    """
    selector = PipelineSelector(scene_root=scene_dir)
    return selector.select(scene_dir)


def should_skip_deprecated_job(job_name: str) -> bool:
    """Check if a deprecated job should be skipped.

    Args:
        job_name: Name of the job to check

    Returns:
        True if the job should be skipped
    """
    selector = PipelineSelector()
    return selector.should_skip_job(job_name)


def get_active_pipeline_mode() -> PipelineMode:
    """Get the currently active pipeline mode."""
    selector = PipelineSelector()
    return selector.get_mode()


def create_deprecation_check(job_name: str, tag: str = "JOB"):
    """Create a deprecation check function for a specific job.

    Returns a function that can be called at the start of main() to check
    if the job should run.

    Usage:
        check_deprecation = create_deprecation_check("seg-job", "GEMINI-INV")

        def main():
            if not check_deprecation():
                sys.exit(0)
            # ... rest of job
    """
    import sys

    def check():
        pipeline_mode = os.getenv("PIPELINE_MODE", "zeroscene_first").lower()
        force_run = os.getenv("FORCE_DEPRECATED_JOB", "").lower() in ("1", "true", "yes")

        if pipeline_mode == "zeroscene_first" and not force_run:
            print(f"[{tag}] WARNING: {job_name} is DEPRECATED (ZeroScene transition)", file=sys.stderr)
            print(f"[{tag}] Set PIPELINE_MODE=gemini_only or FORCE_DEPRECATED_JOB=true to run", file=sys.stderr)
            print(f"[{tag}] See docs/ZEROSCENE_TRANSITION.md for details", file=sys.stderr)
            return False

        if pipeline_mode != "gemini_only":
            print(f"[{tag}] NOTE: Running deprecated job as fallback", file=sys.stderr)

        return True

    return check
