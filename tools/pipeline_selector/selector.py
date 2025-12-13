"""
Pipeline Selector Implementation.

Handles ZeroScene pipeline execution and job routing.

Environment Variables:
    ZEROSCENE_AVAILABLE: "true" | "false" (override auto-detection)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict


class PipelineMode(str, Enum):
    """Pipeline execution mode."""
    ZEROSCENE_FIRST = "zeroscene_first"  # Use ZeroScene pipeline (default)


@dataclass
class PipelineDecision:
    """Result of pipeline selection."""
    mode: PipelineMode
    use_zeroscene: bool
    reason: str
    job_sequence: List[str]


class PipelineSelector:
    """Selects and routes pipeline jobs.

    The selector determines job sequence based on:
    1. ZeroScene output availability
    2. Scene directory contents
    """

    # ZeroScene output markers
    ZEROSCENE_MARKERS = [
        "zeroscene/scene_info.json",
        "zeroscene/objects",
    ]

    def __init__(self, scene_root: Optional[Path] = None):
        """Initialize the pipeline selector.

        Args:
            scene_root: Root directory for scene data (for detecting outputs)
        """
        self.scene_root = Path(scene_root) if scene_root else None

    def get_mode(self) -> PipelineMode:
        """Get the configured pipeline mode."""
        return PipelineMode.ZEROSCENE_FIRST

    def is_zeroscene_available(self) -> bool:
        """Check if ZeroScene is available (external service or local)."""
        # Check environment override
        override = os.getenv("ZEROSCENE_AVAILABLE", "").lower()
        if override == "true":
            return True
        if override == "false":
            return False

        # Check if zeroscene output exists
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

    def select(self, scene_dir: Optional[Path] = None) -> PipelineDecision:
        """Select the appropriate pipeline for processing.

        Args:
            scene_dir: Scene directory to check for existing outputs

        Returns:
            PipelineDecision with selected mode and job sequence
        """
        mode = self.get_mode()
        scene_dir = scene_dir or self.scene_root

        has_zeroscene = scene_dir and self.has_zeroscene_output(scene_dir)
        zeroscene_available = self.is_zeroscene_available()

        if has_zeroscene:
            return PipelineDecision(
                mode=mode,
                use_zeroscene=True,
                reason="ZeroScene output exists",
                job_sequence=self._get_zeroscene_jobs(),
            )

        if zeroscene_available:
            return PipelineDecision(
                mode=mode,
                use_zeroscene=True,
                reason="ZeroScene available, will run reconstruction",
                job_sequence=self._get_zeroscene_jobs(),
            )

        # ZeroScene not available
        return PipelineDecision(
            mode=mode,
            use_zeroscene=False,
            reason="Waiting for ZeroScene reconstruction (set ZEROSCENE_AVAILABLE=true when ready)",
            job_sequence=self._get_zeroscene_jobs(),
        )

    def _get_zeroscene_jobs(self) -> List[str]:
        """Get job sequence for ZeroScene pipeline."""
        return [
            # ZeroScene reconstruction runs externally
            "zeroscene-job",       # Adapter: converts ZeroScene -> BlueprintPipeline
            "scale-job",           # Optional: calibrate scale if needed
            "interactive-job",     # Articulation for doors/drawers
            "simready-job",        # Physics properties
            "usd-assembly-job",    # Assemble scene.usda
            "replicator-job",      # Domain randomization
            "variation-gen-job",   # Variation assets
            "isaac-lab-job",       # Isaac Lab tasks
        ]

    def get_job_env_overrides(self, job_name: str) -> Dict[str, str]:
        """Get environment variable overrides for a job."""
        mode = self.get_mode()
        overrides = {}

        # Pass pipeline mode to all jobs
        overrides["PIPELINE_MODE"] = mode.value
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


def get_active_pipeline_mode() -> PipelineMode:
    """Get the currently active pipeline mode."""
    return PipelineMode.ZEROSCENE_FIRST
