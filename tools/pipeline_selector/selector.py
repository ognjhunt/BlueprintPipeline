"""
Pipeline Selector Implementation.

Handles 3D-RE-GEN pipeline execution and job routing.

3D-RE-GEN (arXiv:2512.17459) is a modular, compositional pipeline for
"image → sim-ready 3D reconstruction" with explicit physical constraints.

Reference:
- Paper: https://arxiv.org/abs/2512.17459
- Project: https://3dregen.jdihlmann.com/
- GitHub: https://github.com/cgtuebingen/3D-RE-GEN

Environment Variables:
    REGEN3D_AVAILABLE: "true" | "false" (override auto-detection)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict


class PipelineMode(str, Enum):
    """Pipeline execution mode."""
    REGEN3D_FIRST = "regen3d_first"  # Use 3D-RE-GEN pipeline (default)


@dataclass
class PipelineDecision:
    """Result of pipeline selection."""
    mode: PipelineMode
    use_regen3d: bool
    reason: str
    job_sequence: List[str]


class PipelineSelector:
    """Selects and routes pipeline jobs.

    The selector determines job sequence based on:
    1. 3D-RE-GEN output availability
    2. Scene directory contents
    """

    # 3D-RE-GEN output markers
    REGEN3D_MARKERS = [
        "regen3d/scene_info.json",
        "regen3d/objects",
    ]

    def __init__(self, scene_root: Optional[Path] = None):
        """Initialize the pipeline selector.

        Args:
            scene_root: Root directory for scene data (for detecting outputs)
        """
        self.scene_root = Path(scene_root) if scene_root else None

    def get_mode(self) -> PipelineMode:
        """Get the configured pipeline mode."""
        return PipelineMode.REGEN3D_FIRST

    def is_regen3d_available(self) -> bool:
        """Check if 3D-RE-GEN is available (external service or local)."""
        # Check environment override
        override = os.getenv("REGEN3D_AVAILABLE", "").lower()
        if override == "true":
            return True
        if override == "false":
            return False

        # Check if regen3d output exists
        if self.scene_root:
            return self.has_regen3d_output(self.scene_root)

        return False

    def has_regen3d_output(self, scene_dir: Path) -> bool:
        """Check if 3D-RE-GEN output exists for a scene."""
        for marker in self.REGEN3D_MARKERS:
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

        has_regen3d = scene_dir and self.has_regen3d_output(scene_dir)
        regen3d_available = self.is_regen3d_available()

        if has_regen3d:
            return PipelineDecision(
                mode=mode,
                use_regen3d=True,
                reason="3D-RE-GEN output exists",
                job_sequence=self._get_regen3d_jobs(),
            )

        if regen3d_available:
            return PipelineDecision(
                mode=mode,
                use_regen3d=True,
                reason="3D-RE-GEN available, will run reconstruction",
                job_sequence=self._get_regen3d_jobs(),
            )

        # 3D-RE-GEN not available
        return PipelineDecision(
            mode=mode,
            use_regen3d=False,
            reason="Waiting for 3D-RE-GEN reconstruction (set REGEN3D_AVAILABLE=true when ready)",
            job_sequence=self._get_regen3d_jobs(),
        )

    def _get_regen3d_jobs(self) -> List[str]:
        """Get job sequence for 3D-RE-GEN pipeline."""
        return [
            # 3D-RE-GEN reconstruction runs externally
            "regen3d-job",         # Adapter: converts 3D-RE-GEN -> BlueprintPipeline
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
        overrides["USE_REGEN3D_OUTPUT"] = "true"

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
    return PipelineMode.REGEN3D_FIRST
