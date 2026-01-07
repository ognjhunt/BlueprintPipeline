"""
Pipeline Selector Implementation.

Handles 3D-RE-GEN pipeline execution and job routing, with Genie Sim 3.0
as the default backend for data generation.

3D-RE-GEN (arXiv:2512.17459) is a modular, compositional pipeline for
"image → sim-ready 3D reconstruction" with explicit physical constraints.

Genie Sim 3.0 Integration (DEFAULT):
By default, the pipeline routes to Genie Sim 3.0 for:
- Task generation (LLM)
- Trajectory planning (cuRobo)
- Data collection (automated + teleop)
- Evaluation (VLM)
- LeRobot v0.3.3 dataset export

To use BlueprintPipeline's own episode generation instead, set USE_GENIESIM=false.

Reference:
- 3D-RE-GEN Paper: https://arxiv.org/abs/2512.17459
- Genie Sim 3.0 Paper: https://arxiv.org/html/2601.02078v1
- Genie Sim GitHub: https://github.com/AgibotTech/genie_sim

Environment Variables:
    REGEN3D_AVAILABLE: "true" | "false" (override auto-detection)
    USE_GENIESIM: "true" (default) | "false" (use BlueprintPipeline episode generation)
    GENIESIM_ROBOT_TYPE: Robot type for Genie Sim (franka, g2, ur10)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict


class PipelineMode(str, Enum):
    """Pipeline execution mode."""
    REGEN3D_FIRST = "regen3d_first"  # Use 3D-RE-GEN with BlueprintPipeline episode generation
    GENIESIM = "geniesim"  # Use Genie Sim 3.0 for data generation (default)


class DataGenerationBackend(str, Enum):
    """Backend for episode/data generation."""
    BLUEPRINTPIPELINE = "blueprintpipeline"  # Our own episode-generation-job (USE_GENIESIM=false)
    GENIESIM = "geniesim"  # AGIBOT Genie Sim 3.0 (default)


@dataclass
class PipelineDecision:
    """Result of pipeline selection."""
    mode: PipelineMode
    use_regen3d: bool
    reason: str
    job_sequence: List[str]
    data_backend: DataGenerationBackend = DataGenerationBackend.BLUEPRINTPIPELINE


class PipelineSelector:
    """Selects and routes pipeline jobs.

    The selector determines job sequence based on:
    1. 3D-RE-GEN output availability
    2. Scene directory contents
    3. Data generation backend (BlueprintPipeline or Genie Sim)
    """

    # 3D-RE-GEN output markers
    REGEN3D_MARKERS = [
        "regen3d/scene_info.json",
        "regen3d/objects",
    ]

    # Genie Sim export markers
    GENIESIM_MARKERS = [
        "geniesim/scene_graph.json",
        "geniesim/export_manifest.json",
    ]

    def __init__(self, scene_root: Optional[Path] = None):
        """Initialize the pipeline selector.

        Args:
            scene_root: Root directory for scene data (for detecting outputs)
        """
        self.scene_root = Path(scene_root) if scene_root else None

    def get_mode(self) -> PipelineMode:
        """Get the configured pipeline mode.

        Returns GENIESIM by default. Set USE_GENIESIM=false to use
        BlueprintPipeline's own episode generation instead.
        """
        use_geniesim = os.getenv("USE_GENIESIM", "true").lower() == "true"
        if use_geniesim:
            return PipelineMode.GENIESIM
        return PipelineMode.REGEN3D_FIRST

    def get_data_backend(self) -> DataGenerationBackend:
        """Get the configured data generation backend.

        Returns GENIESIM by default. Set USE_GENIESIM=false to use
        BlueprintPipeline's own episode generation instead.
        """
        use_geniesim = os.getenv("USE_GENIESIM", "true").lower() == "true"
        if use_geniesim:
            return DataGenerationBackend.GENIESIM
        return DataGenerationBackend.BLUEPRINTPIPELINE

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

    def has_geniesim_export(self, scene_dir: Path) -> bool:
        """Check if Genie Sim export exists for a scene."""
        for marker in self.GENIESIM_MARKERS:
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
        data_backend = self.get_data_backend()
        scene_dir = scene_dir or self.scene_root

        has_regen3d = scene_dir and self.has_regen3d_output(scene_dir)
        regen3d_available = self.is_regen3d_available()

        # Determine job sequence based on data backend
        if data_backend == DataGenerationBackend.GENIESIM:
            job_sequence = self._get_geniesim_jobs()
            reason_suffix = " → Genie Sim data generation"
        else:
            job_sequence = self._get_regen3d_jobs()
            reason_suffix = " → BlueprintPipeline episode generation"

        if has_regen3d:
            return PipelineDecision(
                mode=mode,
                use_regen3d=True,
                reason=f"3D-RE-GEN output exists{reason_suffix}",
                job_sequence=job_sequence,
                data_backend=data_backend,
            )

        if regen3d_available:
            return PipelineDecision(
                mode=mode,
                use_regen3d=True,
                reason=f"3D-RE-GEN available, will run reconstruction{reason_suffix}",
                job_sequence=job_sequence,
                data_backend=data_backend,
            )

        # 3D-RE-GEN not available
        return PipelineDecision(
            mode=mode,
            use_regen3d=False,
            reason=f"Waiting for 3D-RE-GEN reconstruction (set REGEN3D_AVAILABLE=true when ready){reason_suffix}",
            job_sequence=job_sequence,
            data_backend=data_backend,
        )

    def _get_regen3d_jobs(self) -> List[str]:
        """Get job sequence for 3D-RE-GEN pipeline with BlueprintPipeline data generation."""
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
            "episode-generation-job",  # BlueprintPipeline episode generation
        ]

    def _get_geniesim_jobs(self) -> List[str]:
        """Get job sequence for 3D-RE-GEN pipeline with Genie Sim data generation.

        This sequence replaces episode-generation-job and isaac-lab-job with
        genie-sim-export-job, which outputs files ready for Genie Sim to consume.

        Data generation (tasks, trajectories, episodes, evaluation) happens
        in Genie Sim 3.0, not in BlueprintPipeline.
        """
        return [
            # 3D-RE-GEN reconstruction runs externally
            "regen3d-job",         # Adapter: converts 3D-RE-GEN -> BlueprintPipeline
            "scale-job",           # Optional: calibrate scale if needed
            "interactive-job",     # Articulation for doors/drawers
            "simready-job",        # Physics properties
            "usd-assembly-job",    # Assemble scene.usda
            "replicator-job",      # Domain randomization (still useful for scene variants)
            # NOTE: variation-gen-job removed - Genie Sim handles asset variations
            # NOTE: isaac-lab-job removed - Genie Sim handles task generation
            # NOTE: episode-generation-job removed - Genie Sim handles data collection
            "genie-sim-export-job",  # Export to Genie Sim format
            # After this, Genie Sim takes over for:
            # - LLM task generation
            # - cuRobo trajectory planning
            # - Automated/teleop data collection
            # - VLM evaluation
            # - LeRobot export
        ]

    def get_job_env_overrides(self, job_name: str) -> Dict[str, str]:
        """Get environment variable overrides for a job."""
        mode = self.get_mode()
        data_backend = self.get_data_backend()
        overrides = {}

        # Pass pipeline mode to all jobs
        overrides["PIPELINE_MODE"] = mode.value
        overrides["USE_REGEN3D_OUTPUT"] = "true"
        overrides["DATA_BACKEND"] = data_backend.value

        # Genie Sim specific overrides
        if data_backend == DataGenerationBackend.GENIESIM:
            overrides["USE_GENIESIM"] = "true"
            robot_type = os.getenv("GENIESIM_ROBOT_TYPE", "franka")
            overrides["GENIESIM_ROBOT_TYPE"] = robot_type

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
    selector = PipelineSelector()
    return selector.get_mode()


def get_data_generation_backend() -> DataGenerationBackend:
    """Get the currently active data generation backend."""
    selector = PipelineSelector()
    return selector.get_data_backend()


def is_geniesim_enabled() -> bool:
    """Check if Genie Sim integration is enabled.

    Returns True by default. Set USE_GENIESIM=false to disable.
    """
    return os.getenv("USE_GENIESIM", "true").lower() == "true"
