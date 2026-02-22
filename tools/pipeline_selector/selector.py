"""Pipeline selection for text-first Stage 1 (SceneSmith/SAGE)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from tools.config.env import parse_bool_env


class PipelineMode(str, Enum):
    """Pipeline execution mode."""

    STANDARD = "standard"
    GENIESIM = "geniesim"


class DataGenerationBackend(str, Enum):
    """Backend for episode/data generation."""

    BLUEPRINTPIPELINE = "blueprintpipeline"
    GENIESIM = "geniesim"


@dataclass
class PipelineDecision:
    """Result of pipeline selection."""

    mode: PipelineMode
    use_stage1_output: bool
    reason: str
    job_sequence: List[str]
    data_backend: DataGenerationBackend = DataGenerationBackend.BLUEPRINTPIPELINE


class PipelineSelector:
    """Select and route pipeline jobs for a scene."""

    STAGE1_MARKERS = [
        "assets/.stage1_complete",
        "assets/scene_manifest.json",
        "layout/scene_layout_scaled.json",
        "seg/inventory.json",
    ]

    GENIESIM_MARKERS = [
        "geniesim/scene_graph.json",
        "geniesim/export_manifest.json",
    ]

    VARIATION_ASSET_MARKERS = [
        "variation_assets/.variation_pipeline_complete",
        "variation_assets/simready_assets.json",
    ]

    def __init__(self, scene_root: Optional[Path] = None):
        self.scene_root = Path(scene_root) if scene_root else None

    def get_mode(self) -> PipelineMode:
        use_geniesim = parse_bool_env(os.getenv("USE_GENIESIM"), default=True)
        return PipelineMode.GENIESIM if use_geniesim else PipelineMode.STANDARD

    def get_data_backend(self) -> DataGenerationBackend:
        use_geniesim = parse_bool_env(os.getenv("USE_GENIESIM"), default=True)
        if use_geniesim:
            return DataGenerationBackend.GENIESIM
        return DataGenerationBackend.BLUEPRINTPIPELINE

    def has_stage1_output(self, scene_dir: Path) -> bool:
        return all((scene_dir / marker).exists() for marker in self.STAGE1_MARKERS)

    def has_geniesim_export(self, scene_dir: Path) -> bool:
        return any((scene_dir / marker).exists() for marker in self.GENIESIM_MARKERS)

    def has_variation_assets_output(self, scene_dir: Path) -> bool:
        return any((scene_dir / marker).exists() for marker in self.VARIATION_ASSET_MARKERS)

    def select(self, scene_dir: Optional[Path] = None) -> PipelineDecision:
        mode = self.get_mode()
        data_backend = self.get_data_backend()
        scene_dir = scene_dir or self.scene_root

        has_stage1 = bool(scene_dir and self.has_stage1_output(scene_dir))

        if data_backend == DataGenerationBackend.GENIESIM:
            job_sequence = self._get_geniesim_jobs()
            reason_suffix = " -> Genie Sim data generation"
        else:
            job_sequence = self._get_standard_jobs()
            reason_suffix = " -> BlueprintPipeline episode generation"

        if has_stage1:
            decision = PipelineDecision(
                mode=mode,
                use_stage1_output=True,
                reason=f"Stage 1 outputs already available{reason_suffix}",
                job_sequence=job_sequence,
                data_backend=data_backend,
            )
        else:
            decision = PipelineDecision(
                mode=mode,
                use_stage1_output=False,
                reason=f"Stage 1 outputs missing; text Stage 1 jobs will run{reason_suffix}",
                job_sequence=job_sequence,
                data_backend=data_backend,
            )

        return self._enforce_variation_assets(decision, scene_dir)

    def _get_standard_jobs(self) -> List[str]:
        return [
            "text-scene-gen-job",
            "text-scene-adapter-job",
            "scale-job",
            "interactive-job",
            "simready-job",
            "usd-assembly-job",
            "replicator-job",
            "variation-gen-job",
            "isaac-lab-job",
            "episode-generation-job",
        ]

    def _get_geniesim_jobs(self) -> List[str]:
        return [
            "text-scene-gen-job",
            "text-scene-adapter-job",
            "scale-job",
            "interactive-job",
            "simready-job",
            "usd-assembly-job",
            "replicator-job",
            "variation-gen-job",
            "genie-sim-export-job",
            "genie-sim-submit-job",
            "genie-sim-import-job",
        ]

    def _enforce_variation_assets(
        self,
        decision: PipelineDecision,
        scene_dir: Optional[Path],
    ) -> PipelineDecision:
        if decision.data_backend != DataGenerationBackend.GENIESIM:
            return decision

        allow_missing = parse_bool_env(
            os.getenv("ALLOW_MISSING_VARIATION_ASSETS"),
            default=False,
        )
        if allow_missing or not scene_dir:
            return decision

        if self.has_variation_assets_output(scene_dir):
            return decision

        filtered_jobs = [
            job
            for job in decision.job_sequence
            if job
            not in {
                "genie-sim-export-job",
                "genie-sim-submit-job",
                "genie-sim-import-job",
            }
        ]
        compliance_reason = (
            "Compliance check failed: variation assets are required for commercial "
            "Genie Sim use. Missing variation_assets/.variation_pipeline_complete or "
            "variation_assets/simready_assets.json under the scene root. "
            "Set ALLOW_MISSING_VARIATION_ASSETS=true to override."
        )
        return PipelineDecision(
            mode=decision.mode,
            use_stage1_output=decision.use_stage1_output,
            reason=f"{compliance_reason} {decision.reason}",
            job_sequence=filtered_jobs,
            data_backend=decision.data_backend,
        )

    def get_job_env_overrides(self, job_name: str) -> Dict[str, str]:
        mode = self.get_mode()
        data_backend = self.get_data_backend()
        overrides = {
            "PIPELINE_MODE": mode.value,
            "USE_STAGE1_OUTPUT": "true",
            "DATA_BACKEND": data_backend.value,
        }

        if data_backend == DataGenerationBackend.GENIESIM:
            overrides["USE_GENIESIM"] = "true"
            overrides["GENIESIM_ROBOT_TYPE"] = os.getenv("GENIESIM_ROBOT_TYPE", "franka")

        return overrides


def select_pipeline(scene_dir: Optional[Path] = None) -> PipelineDecision:
    selector = PipelineSelector(scene_root=scene_dir)
    return selector.select(scene_dir)


def get_active_pipeline_mode() -> PipelineMode:
    selector = PipelineSelector()
    return selector.get_mode()


def get_data_generation_backend() -> DataGenerationBackend:
    selector = PipelineSelector()
    return selector.get_data_backend()


def is_geniesim_enabled() -> bool:
    return parse_bool_env(os.getenv("USE_GENIESIM"), default=True)


def should_skip_deprecated_job(job_name: str) -> bool:
    return False
