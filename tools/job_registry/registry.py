"""
Job Registry Implementation.

Tracks all pipeline jobs for the 3D-RE-GEN-based BlueprintPipeline.

3D-RE-GEN (arXiv:2512.17459) is a modular, compositional pipeline for
"image → sim-ready 3D reconstruction" with explicit physical constraints.

Pipeline Jobs (3D-RE-GEN-first approach):
    - regen3d-job: Adapter for 3D-RE-GEN outputs
    - interactive-job: Articulation detection using Particulate
    - simready-job: Physics + manipulation hints
    - usd-assembly-job: USD scene assembly
    - replicator-job: Domain randomization scripts
    - variation-gen-job: Variation asset generation
    - isaac-lab-job: Isaac Lab task generation
    - scale-job: Optional scale calibration

Reference:
- Paper: https://arxiv.org/abs/2512.17459
- Project: https://3dregen.jdihlmann.com/
- GitHub: https://github.com/cgtuebingen/3D-RE-GEN

Module exports:
    - JOBS: A stable, read-only tuple of all JobInfo entries.
    - JOBS_BY_NAME: A name → JobInfo mapping for quick access.
    Import these for callers that don't need to instantiate JobRegistry directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class JobStatus(str, Enum):
    """Status of a job in the pipeline."""
    ACTIVE = "active"              # Fully operational, primary path
    NEW = "new"                    # Newly added for 3D-RE-GEN pipeline
    EXPERIMENTAL = "experimental"  # Under development


class JobCategory(str, Enum):
    """Category of job in the pipeline."""
    ENRICHMENT = "enrichment"      # Asset enrichment
    ASSEMBLY = "assembly"          # Scene assembly
    GENERATION = "generation"      # Synthetic data generation
    TRAINING = "training"          # Training preparation
    ADAPTER = "adapter"            # Pipeline adapters


class PipelineMode(str, Enum):
    """Pipeline execution mode."""
    REGEN3D_FIRST = "regen3d_first"  # Use 3D-RE-GEN pipeline (default)


@dataclass
class JobInfo:
    """Information about a pipeline job."""
    name: str
    description: str
    status: JobStatus
    category: JobCategory

    # Entry point information
    entry_script: str
    docker_image: Optional[str] = None

    # Environment variables
    required_env_vars: List[str] = field(default_factory=list)
    optional_env_vars: List[str] = field(default_factory=list)

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    # Notes
    migration_notes: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.status in (JobStatus.ACTIVE, JobStatus.NEW)


class JobRegistry:
    """Central registry for all pipeline jobs.

    Tracks job status for the 3D-RE-GEN-based pipeline.
    """

    def __init__(self):
        self._jobs: Dict[str, JobInfo] = {}
        self._initialize_jobs()

    def _initialize_jobs(self):
        """Initialize the job registry with all known jobs."""

        # =====================================================================
        # 3D-RE-GEN PIPELINE JOBS
        # =====================================================================

        self._jobs["regen3d-job"] = JobInfo(
            name="regen3d-job",
            description="Adapter converting 3D-RE-GEN outputs to BlueprintPipeline format",
            status=JobStatus.NEW,
            category=JobCategory.ADAPTER,
            entry_script="regen3d-job/regen3d_adapter_job.py",
            docker_image="regen3d-job",
            required_env_vars=["SCENE_ID", "REGEN3D_PREFIX", "ASSETS_PREFIX", "LAYOUT_PREFIX"],
            optional_env_vars=["GEMINI_API_KEY", "OPENAI_API_KEY", "TRUST_REGEN3D_SCALE"],
            depends_on=["regen3d-reconstruction"],
            outputs=[
                "assets/scene_manifest.json",
                "layout/scene_layout_scaled.json",
                "seg/inventory.json",
                "assets/obj_*/asset.glb",
            ],
            migration_notes=(
                "This adapter is the critical integration layer between 3D-RE-GEN "
                "and the rest of the BlueprintPipeline. 3D-RE-GEN uses 4-DoF ground "
                "constraints and background bounding for sim-ready placement."
            ),
        )

        self._jobs["interactive-job"] = JobInfo(
            name="interactive-job",
            description="Articulation bridge using Particulate (fast mesh-based, ~10s)",
            status=JobStatus.ACTIVE,
            category=JobCategory.ENRICHMENT,
            entry_script="interactive-job/run_interactive_assets.py",
            docker_image="interactive-job",
            required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX"],
            optional_env_vars=[
                "PARTICULATE_ENDPOINT",    # Particulate service URL
                "DISALLOW_PLACEHOLDER_URDF",
                "TIMEOUT_SECONDS",
            ],
            depends_on=["regen3d-job"],
            outputs=["assets/interactive/obj_*/articulated.usda"],
            migration_notes=(
                "Uses Particulate (arXiv:2512.11798) for fast feed-forward "
                "mesh articulation detection."
            ),
        )

        self._jobs["simready-job"] = JobInfo(
            name="simready-job",
            description="Physics and manipulation hints generation",
            status=JobStatus.ACTIVE,
            category=JobCategory.ENRICHMENT,
            entry_script="simready-job/prepare_simready_assets.py",
            docker_image="simready-job",
            required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX"],
            optional_env_vars=["GEMINI_API_KEY", "OPENAI_API_KEY", "SIMREADY_ADD_PROXY_COLLIDER"],
            depends_on=["regen3d-job", "interactive-job"],
            outputs=[
                "assets/obj_*/simready.usda",
                "assets/obj_*/metadata.json",
            ],
            migration_notes=(
                "Supports both Gemini and OpenAI for physics estimation. "
                "3D-RE-GEN materials provide additional hints for friction/roughness."
            ),
        )

        self._jobs["usd-assembly-job"] = JobInfo(
            name="usd-assembly-job",
            description="USD scene assembly and conversion",
            status=JobStatus.ACTIVE,
            category=JobCategory.ASSEMBLY,
            entry_script="usd-assembly-job/assemble_scene.py",
            docker_image="usd-assembly-job",
            required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX", "LAYOUT_PREFIX", "USD_PREFIX"],
            depends_on=["simready-job"],
            outputs=["usd/scene.usda"],
            migration_notes=(
                "Reads layout from 3D-RE-GEN adapter output."
            ),
        )

        self._jobs["replicator-job"] = JobInfo(
            name="replicator-job",
            description="Replicator bundle generation for domain randomization",
            status=JobStatus.ACTIVE,
            category=JobCategory.GENERATION,
            entry_script="replicator-job/generate_replicator_bundle.py",
            docker_image="replicator-job",
            required_env_vars=["BUCKET", "SCENE_ID", "SEG_PREFIX", "ASSETS_PREFIX", "REPLICATOR_PREFIX"],
            optional_env_vars=["GEMINI_API_KEY", "OPENAI_API_KEY", "REQUESTED_POLICIES"],
            depends_on=["usd-assembly-job"],
            outputs=[
                "replicator/placement_regions.usda",
                "replicator/policies/*.py",
                "replicator/variation_assets/manifest.json",
            ],
            migration_notes=(
                "Reads inventory from seg/inventory.json. "
                "Supports OpenAI as alternative to Gemini for scene analysis."
            ),
        )

        self._jobs["variation-gen-job"] = JobInfo(
            name="variation-gen-job",
            description="Variation asset generation for domain randomization",
            status=JobStatus.ACTIVE,
            category=JobCategory.GENERATION,
            entry_script="variation-gen-job/generate_variation_assets.py",
            docker_image="variation-gen-job",
            required_env_vars=["BUCKET", "SCENE_ID", "REPLICATOR_PREFIX"],
            optional_env_vars=["GEMINI_API_KEY", "OPENAI_API_KEY", "MAX_ASSETS"],
            depends_on=["replicator-job"],
            outputs=["variation_assets/*/reference.png", "variation_assets/variation_assets.json"],
            migration_notes=(
                "Supports OpenAI DALL-E as alternative to Gemini for image generation."
            ),
        )

        self._jobs["isaac-lab-job"] = JobInfo(
            name="isaac-lab-job",
            description="Isaac Lab task generation for RL training",
            status=JobStatus.NEW,
            category=JobCategory.TRAINING,
            entry_script="isaac-lab-job/generate_tasks.py",
            docker_image="isaac-lab-job",
            required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX", "REPLICATOR_PREFIX"],
            optional_env_vars=["POLICIES", "ENVIRONMENT_TYPE"],
            depends_on=["replicator-job"],
            outputs=[
                "isaac_lab/env_cfg.py",
                "isaac_lab/task_*.py",
                "isaac_lab/train_cfg.yaml",
            ],
            migration_notes=(
                "Generates Isaac Lab training configurations. "
                "Uses the same PolicyTarget concepts as replicator-job."
            ),
        )

        self._jobs["episode-generation-job"] = JobInfo(
            name="episode-generation-job",
            description="Generates training-ready robotic episodes in LeRobot format",
            status=JobStatus.NEW,
            category=JobCategory.TRAINING,
            entry_script="episode-generation-job/generate_episodes.py",
            docker_image="episode-generation-job",
            required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX", "EPISODES_PREFIX"],
            optional_env_vars=[
                "ROBOT_TYPE",            # franka, ur10, fetch (default: franka)
                "EPISODES_PER_VARIATION",  # default: 10
                "MAX_VARIATIONS",        # default: all
                "FPS",                   # default: 30
                "USE_LLM",               # default: true
                "GEMINI_API_KEY",
            ],
            depends_on=["replicator-job", "isaac-lab-job"],
            outputs=[
                "episodes/lerobot/meta/info.json",
                "episodes/lerobot/data/chunk-*/episode_*.parquet",
                "episodes/manifests/generation_manifest.json",
            ],
            migration_notes=(
                "Generates manipulation episodes for each scene variation. "
                "Uses AI (Gemini) for motion planning, outputs LeRobot v2.0 format. "
                "Sellable alongside scenes for plug-and-play RL training."
            ),
        )

        self._jobs["quality-gate-job"] = JobInfo(
            name="quality-gate-job",
            description="Evaluates production SLIs and enforces quality gates on episode outputs",
            status=JobStatus.NEW,
            category=JobCategory.TRAINING,
            entry_script="tools/quality_gates/sli_gate_runner.py",
            docker_image="quality-gate-job",
            required_env_vars=["BUCKET", "SCENE_ID", "EPISODES_PREFIX"],
            optional_env_vars=["DATA_ROOT", "QUALITY_GATE_REPORT_PATH"],
            depends_on=["episode-generation-job"],
            outputs=["episodes/quality/quality_gate_report.json"],
            migration_notes=(
                "Runs quality gates using dataset_quality_manifest.json and "
                "generation_manifest.json to block production runs when SLIs fall below thresholds."
            ),
        )

        self._jobs["scale-job"] = JobInfo(
            name="scale-job",
            description="Scale calibration and reference object detection",
            status=JobStatus.ACTIVE,
            category=JobCategory.ENRICHMENT,
            entry_script="scale-job/calibrate_scale.py",
            docker_image="scale-job",
            required_env_vars=["BUCKET", "SCENE_ID", "LAYOUT_PREFIX"],
            depends_on=["regen3d-job"],
            outputs=["layout/scene_layout_scaled.json"],
            migration_notes=(
                "Optional if TRUST_REGEN3D_SCALE=true. "
                "Otherwise runs after regen3d-job to calibrate metric scale."
            ),
        )

        # =====================================================================
        # ISAAC LAB-ARENA INTEGRATION JOBS
        # =====================================================================

        self._jobs["arena-export-job"] = JobInfo(
            name="arena-export-job",
            description="Export scene to Isaac Lab-Arena format with affordances",
            status=JobStatus.NEW,
            category=JobCategory.TRAINING,
            entry_script="arena-export-job/arena_export_job.py",
            docker_image="arena-export-job",
            required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX"],
            optional_env_vars=[
                "USE_LLM_AFFORDANCES",    # Enable Gemini for affordance detection
                "ENABLE_HUB_REGISTRATION",  # Register with LeRobot Hub
                "HUB_NAMESPACE",          # Hugging Face namespace
                "GEMINI_API_KEY",
                "HF_TOKEN",               # For Hub registration
            ],
            depends_on=["isaac-lab-job", "usd-assembly-job"],
            outputs=[
                "arena/arena_manifest.json",
                "arena/scene_module.py",
                "arena/tasks/*.py",
                "arena/asset_registry.json",
                "arena/hub_config.yaml",
            ],
            migration_notes=(
                "Isaac Lab-Arena integration for standardized policy evaluation. "
                "Detects affordances (Openable, Graspable, Turnable, etc.) and "
                "generates Arena-compatible tasks. Supports LeRobot Hub registration. "
                "Reference: https://developer.nvidia.com/isaac/lab-arena"
            ),
        )

        # =====================================================================
        # DREAM2FLOW PIPELINE JOBS (arXiv:2512.24766)
        # =====================================================================

        self._jobs["dream2flow-preparation-job"] = JobInfo(
            name="dream2flow-preparation-job",
            description="Dream2Flow bundle preparation (video generation, 3D flow extraction)",
            status=JobStatus.NEW,
            category=JobCategory.TRAINING,
            entry_script="dream2flow-preparation-job/entrypoint.py",
            docker_image="dream2flow-preparation-job",
            required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX", "DREAM2FLOW_PREFIX"],
            optional_env_vars=[
                "USD_PREFIX",
                "NUM_TASKS",
                "RESOLUTION_WIDTH",
                "RESOLUTION_HEIGHT",
                "NUM_FRAMES",
                "FPS",
                "ROBOT",
                "VIDEO_API_ENDPOINT",
            ],
            depends_on=["regen3d-job"],
            outputs=[
                "dream2flow/*/manifest.json",
                "dream2flow/*/video/*.mp4",
                "dream2flow/*/flow/*.json",
                "dream2flow/*/trajectory/*.json",
                "dream2flow/dream2flow_bundles_manifest.json",
            ],
            migration_notes=(
                "Dream2Flow (arXiv:2512.24766) uses video diffusion to imagine task "
                "execution, then extracts 3D object flow as embodiment-agnostic goal/reward. "
                "Scaffolding ready - full implementation pending model release."
            ),
        )

        self._jobs["dream2flow-inference-job"] = JobInfo(
            name="dream2flow-inference-job",
            description="Dream2Flow model inference (video generation + flow extraction)",
            status=JobStatus.NEW,
            category=JobCategory.TRAINING,
            entry_script="dream2flow-preparation-job/dream2flow_inference_job.py",
            docker_image="dream2flow-preparation-job",
            required_env_vars=["BUCKET", "SCENE_ID", "DREAM2FLOW_PREFIX"],
            optional_env_vars=[
                "VIDEO_API_ENDPOINT",
                "VIDEO_CHECKPOINT_PATH",
                "OVERWRITE",
            ],
            depends_on=["dream2flow-preparation-job"],
            outputs=[
                "dream2flow/*/inference_video.mp4",
                "dream2flow/.dream2flow_inference_complete",
            ],
            migration_notes=(
                "Runs Dream2Flow model inference on prepared bundles. "
                "Currently uses placeholder generation - will be updated when "
                "Dream2Flow model is publicly released."
            ),
        )

    def get_job(self, name: str) -> Optional[JobInfo]:
        """Get job information by name."""
        return self._jobs.get(name)

    def get_all_jobs(self) -> List[JobInfo]:
        """Get all registered jobs."""
        return list(self._jobs.values())

    def get_jobs_by_status(self, status: JobStatus) -> List[JobInfo]:
        """Get jobs with a specific status."""
        return [j for j in self._jobs.values() if j.status == status]

    def get_jobs_by_category(self, category: JobCategory) -> List[JobInfo]:
        """Get jobs in a specific category."""
        return [j for j in self._jobs.values() if j.category == category]

    def get_active_jobs(self) -> List[JobInfo]:
        """Get all active jobs."""
        return [j for j in self._jobs.values() if j.is_active]

    def is_regen3d_ready(self) -> bool:
        """Check if 3D-RE-GEN pipeline is ready (all required jobs exist)."""
        required_jobs = ["regen3d-job"]
        for name in required_jobs:
            job = self.get_job(name)
            if not job:
                return False
        return True

    def get_pipeline_mode(self) -> PipelineMode:
        """Get current pipeline mode from environment."""
        return PipelineMode.REGEN3D_FIRST

    def get_job_sequence(self) -> List[str]:
        """Get the recommended job execution sequence."""
        return [
            "regen3d-reconstruction",  # External 3D-RE-GEN
            "regen3d-job",
            "scale-job",  # Optional
            "interactive-job",
            "simready-job",
            "usd-assembly-job",
            "replicator-job",
            "variation-gen-job",
            "isaac-lab-job",
            "arena-export-job",  # Isaac Lab-Arena integration (affordances + tasks)
            "episode-generation-job",  # Generates training episodes
        ]

    def print_status_report(self):
        """Print a status report of all jobs."""
        print("\n" + "=" * 70)
        print("BlueprintPipeline Job Registry Status Report")
        print("=" * 70)

        print(f"\nPipeline Mode: {self.get_pipeline_mode().value}")
        print(f"3D-RE-GEN Ready: {self.is_regen3d_ready()}")

        print("\n--- ACTIVE JOBS ---")
        for job in self.get_jobs_by_status(JobStatus.ACTIVE):
            print(f"  [{job.status.value:12}] {job.name:20} - {job.description[:40]}...")

        print("\n--- NEW JOBS (3D-RE-GEN Pipeline) ---")
        for job in self.get_jobs_by_status(JobStatus.NEW):
            print(f"  [{job.status.value:12}] {job.name:20} - {job.description[:40]}...")

        print("\n" + "=" * 70)


# Singleton instance
_registry: Optional[JobRegistry] = None


def get_registry() -> JobRegistry:
    """Get the global job registry instance."""
    global _registry
    if _registry is None:
        _registry = JobRegistry()
    return _registry


JOBS = tuple(get_registry().get_all_jobs())
JOBS_BY_NAME = {job.name: job for job in JOBS}
