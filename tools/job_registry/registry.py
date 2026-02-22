"""Job registry for text-first Stage 1 (SceneSmith/SAGE)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from tools.config.env import parse_bool_env


class JobStatus(str, Enum):
    """Status of a job in the pipeline."""

    ACTIVE = "active"
    NEW = "new"
    EXPERIMENTAL = "experimental"


class JobCategory(str, Enum):
    """Category of job in the pipeline."""

    ENRICHMENT = "enrichment"
    ASSEMBLY = "assembly"
    GENERATION = "generation"
    TRAINING = "training"
    ADAPTER = "adapter"


class PipelineMode(str, Enum):
    """Pipeline execution mode."""

    STANDARD = "standard"


@dataclass
class JobInfo:
    """Information about a pipeline job."""

    name: str
    description: str
    status: JobStatus
    category: JobCategory

    entry_script: str
    docker_image: Optional[str] = None

    required_env_vars: List[str] = field(default_factory=list)
    optional_env_vars: List[str] = field(default_factory=list)

    depends_on: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    migration_notes: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.status in (JobStatus.ACTIVE, JobStatus.NEW)


class JobRegistry:
    """Central registry for all pipeline jobs."""

    def __init__(self):
        self._jobs: Dict[str, JobInfo] = {}
        self._initialize_jobs()

    def _initialize_jobs(self):
        enable_experimental = parse_bool_env(os.getenv("ENABLE_EXPERIMENTAL_PIPELINE"), default=False) is True
        enable_dwm = enable_experimental or (
            parse_bool_env(os.getenv("ENABLE_DWM"), default=False) is True
        )
        enable_dream2flow = enable_experimental or (
            parse_bool_env(os.getenv("ENABLE_DREAM2FLOW"), default=False) is True
        )

        self._jobs["text-scene-gen-job"] = JobInfo(
            name="text-scene-gen-job",
            description="Generate text Stage 1 package from scene request",
            status=JobStatus.NEW,
            category=JobCategory.GENERATION,
            entry_script="text-scene-gen-job/generate_text_scene.py",
            docker_image="text-scene-gen-job",
            required_env_vars=["BUCKET", "SCENE_ID"],
            optional_env_vars=["TEXT_BACKEND_DEFAULT", "TEXT_BACKEND_ALLOWLIST", "TEXT_GEN_MAX_SEEDS"],
            outputs=[
                "textgen/package.json",
                "textgen/request.normalized.json",
                "textgen/.textgen_complete",
            ],
            migration_notes="Stage 1 source generation using SceneSmith/SAGE backends.",
        )

        self._jobs["text-scene-adapter-job"] = JobInfo(
            name="text-scene-adapter-job",
            description="Build canonical Stage 1 assets/layout/seg artifacts",
            status=JobStatus.NEW,
            category=JobCategory.ADAPTER,
            entry_script="text-scene-adapter-job/adapt_text_scene.py",
            docker_image="text-scene-adapter-job",
            required_env_vars=["BUCKET", "SCENE_ID"],
            optional_env_vars=["ASSETS_PREFIX", "LAYOUT_PREFIX", "SEG_PREFIX"],
            depends_on=["text-scene-gen-job"],
            outputs=[
                "assets/scene_manifest.json",
                "layout/scene_layout_scaled.json",
                "seg/inventory.json",
                "assets/.stage1_complete",
            ],
            migration_notes="Canonical Stage 1 completion marker is assets/.stage1_complete.",
        )

        self._jobs["interactive-job"] = JobInfo(
            name="interactive-job",
            description="Articulation bridge using Particulate",
            status=JobStatus.ACTIVE,
            category=JobCategory.ENRICHMENT,
            entry_script="interactive-job/run_interactive_assets.py",
            docker_image="interactive-job",
            required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX"],
            optional_env_vars=["PARTICULATE_ENDPOINT", "DISALLOW_PLACEHOLDER_URDF", "TIMEOUT_SECONDS"],
            depends_on=["text-scene-adapter-job"],
            outputs=["assets/interactive/obj_*/articulated.usda"],
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
            depends_on=["text-scene-adapter-job", "interactive-job"],
            outputs=["assets/obj_*/simready.usda", "assets/obj_*/metadata.json"],
        )

        self._jobs["scale-job"] = JobInfo(
            name="scale-job",
            description="Scale calibration and reference object detection",
            status=JobStatus.ACTIVE,
            category=JobCategory.ENRICHMENT,
            entry_script="scale-job/calibrate_scale.py",
            docker_image="scale-job",
            required_env_vars=["BUCKET", "SCENE_ID", "LAYOUT_PREFIX"],
            depends_on=["text-scene-adapter-job"],
            outputs=["layout/scene_layout_scaled.json"],
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
            outputs=["isaac_lab/env_cfg.py", "isaac_lab/task_*.py", "isaac_lab/train_cfg.yaml"],
        )

        self._jobs["episode-generation-job"] = JobInfo(
            name="episode-generation-job",
            description="Generates training-ready robotic episodes in LeRobot format",
            status=JobStatus.NEW,
            category=JobCategory.TRAINING,
            entry_script="episode-generation-job/generate_episodes.py",
            docker_image="episode-generation-job",
            required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX", "EPISODES_PREFIX"],
            optional_env_vars=["ROBOT_TYPE", "EPISODES_PER_VARIATION", "MAX_VARIATIONS", "FPS", "USE_LLM", "GEMINI_API_KEY"],
            depends_on=["replicator-job", "isaac-lab-job"],
            outputs=[
                "episodes/lerobot/meta/info.json",
                "episodes/lerobot/data/chunk-*/episode_*.parquet",
                "episodes/manifests/generation_manifest.json",
            ],
        )

        self._jobs["genie-sim-export-job"] = JobInfo(
            name="genie-sim-export-job",
            description="Exports pipeline outputs to Genie Sim format",
            status=JobStatus.ACTIVE,
            category=JobCategory.TRAINING,
            entry_script="genie-sim-export-job/export_to_geniesim.py",
            docker_image="genie-sim-export-job",
            required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX"],
            depends_on=["variation-gen-job"],
            outputs=["geniesim/export_manifest.json", "geniesim/scene_graph.json"],
        )

        self._jobs["genie-sim-submit-job"] = JobInfo(
            name="genie-sim-submit-job",
            description="Submits Genie Sim jobs and tracks execution",
            status=JobStatus.ACTIVE,
            category=JobCategory.TRAINING,
            entry_script="genie-sim-submit-job/submit_job.py",
            docker_image="genie-sim-submit-job",
            required_env_vars=["BUCKET", "SCENE_ID", "GENIESIM_PREFIX"],
            depends_on=["genie-sim-export-job"],
            outputs=["geniesim/job.json", "geniesim/.geniesim_submitted"],
        )

        self._jobs["genie-sim-import-job"] = JobInfo(
            name="genie-sim-import-job",
            description="Imports Genie Sim generated artifacts back into pipeline",
            status=JobStatus.ACTIVE,
            category=JobCategory.TRAINING,
            entry_script="genie-sim-import-job/import_results.py",
            docker_image="genie-sim-import-job",
            required_env_vars=["BUCKET", "SCENE_ID", "GENIESIM_PREFIX"],
            depends_on=["genie-sim-submit-job"],
            outputs=["geniesim/.geniesim_import_complete"],
        )

        self._jobs["dataset-delivery-job"] = JobInfo(
            name="dataset-delivery-job",
            description="Delivers curated dataset bundles to configured storage targets",
            status=JobStatus.ACTIVE,
            category=JobCategory.TRAINING,
            entry_script="dataset-delivery-job/deliver_dataset.py",
            docker_image="dataset-delivery-job",
            required_env_vars=["BUCKET", "SCENE_ID"],
            depends_on=["genie-sim-import-job"],
            outputs=["geniesim/.dataset_delivery_complete"],
        )

        self._jobs["arena-export-job"] = JobInfo(
            name="arena-export-job",
            description="Export scene to Isaac Lab-Arena format with affordances",
            status=JobStatus.NEW,
            category=JobCategory.TRAINING,
            entry_script="arena-export-job/arena_export_job.py",
            docker_image="arena-export-job",
            required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX"],
            optional_env_vars=["USE_LLM_AFFORDANCES", "ENABLE_HUB_REGISTRATION", "HUB_NAMESPACE", "GEMINI_API_KEY", "HF_TOKEN"],
            depends_on=["isaac-lab-job", "usd-assembly-job"],
            outputs=[
                "arena/arena_manifest.json",
                "arena/scene_module.py",
                "arena/tasks/*.py",
                "arena/asset_registry.json",
                "arena/hub_config.yaml",
            ],
        )

        if enable_dwm:
            self._jobs["dwm-preparation-job"] = JobInfo(
                name="dwm-preparation-job",
                description="DWM bundle preparation (egocentric videos + hand meshes)",
                status=JobStatus.NEW,
                category=JobCategory.TRAINING,
                entry_script="dwm-preparation-job/entrypoint.py",
                docker_image="dwm-preparation-job",
                required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX", "DWM_PREFIX"],
                optional_env_vars=["USD_PREFIX", "REPLICATOR_PREFIX", "DWM_MODEL_PATH", "SKIP_DWM", "RENDER_BACKEND"],
                depends_on=["usd-assembly-job"],
                outputs=["dwm/*/manifest.json", "dwm/*/video/*.mp4", "dwm/dwm_bundles_manifest.json"],
            )

            self._jobs["dwm-inference-job"] = JobInfo(
                name="dwm-inference-job",
                description="DWM model inference (interaction video generation)",
                status=JobStatus.NEW,
                category=JobCategory.TRAINING,
                entry_script="dwm-preparation-job/inference_entrypoint.py",
                docker_image="dwm-preparation-job",
                required_env_vars=["BUCKET", "SCENE_ID", "DWM_PREFIX"],
                optional_env_vars=["DWM_CHECKPOINT_PATH", "OVERWRITE"],
                depends_on=["dwm-preparation-job"],
                outputs=["dwm/*/inference_video.mp4", "dwm/.dwm_inference_complete"],
            )

        if enable_dream2flow:
            self._jobs["dream2flow-preparation-job"] = JobInfo(
                name="dream2flow-preparation-job",
                description="Dream2Flow bundle preparation (video generation, 3D flow extraction)",
                status=JobStatus.NEW,
                category=JobCategory.TRAINING,
                entry_script="dream2flow-preparation-job/entrypoint.py",
                docker_image="dream2flow-preparation-job",
                required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX", "DREAM2FLOW_PREFIX"],
                optional_env_vars=["USD_PREFIX", "NUM_TASKS", "RESOLUTION_WIDTH", "RESOLUTION_HEIGHT", "NUM_FRAMES", "FPS", "ROBOT", "VIDEO_API_ENDPOINT"],
                depends_on=["usd-assembly-job"],
                outputs=[
                    "dream2flow/*/manifest.json",
                    "dream2flow/*/video/*.mp4",
                    "dream2flow/*/flow/*.json",
                    "dream2flow/*/trajectory/*.json",
                    "dream2flow/dream2flow_bundles_manifest.json",
                ],
            )

            self._jobs["dream2flow-inference-job"] = JobInfo(
                name="dream2flow-inference-job",
                description="Dream2Flow model inference (video generation + flow extraction)",
                status=JobStatus.NEW,
                category=JobCategory.TRAINING,
                entry_script="dream2flow-preparation-job/dream2flow_inference_job.py",
                docker_image="dream2flow-preparation-job",
                required_env_vars=["BUCKET", "SCENE_ID", "DREAM2FLOW_PREFIX"],
                optional_env_vars=["VIDEO_API_ENDPOINT", "VIDEO_CHECKPOINT_PATH", "OVERWRITE"],
                depends_on=["dream2flow-preparation-job"],
                outputs=["dream2flow/*/inference_video.mp4", "dream2flow/.dream2flow_inference_complete"],
            )

    def get_job(self, name: str) -> Optional[JobInfo]:
        return self._jobs.get(name)

    def get_all_jobs(self) -> List[JobInfo]:
        return list(self._jobs.values())

    def get_jobs_by_status(self, status: JobStatus) -> List[JobInfo]:
        return [j for j in self._jobs.values() if j.status == status]

    def get_jobs_by_category(self, category: JobCategory) -> List[JobInfo]:
        return [j for j in self._jobs.values() if j.category == category]

    def get_active_jobs(self) -> List[JobInfo]:
        return [j for j in self._jobs.values() if j.is_active]

    def is_stage1_ready(self) -> bool:
        required_jobs = ["text-scene-gen-job", "text-scene-adapter-job"]
        return all(self.get_job(name) is not None for name in required_jobs)

    def get_pipeline_mode(self) -> PipelineMode:
        return PipelineMode.STANDARD

    def get_job_sequence(self) -> List[str]:
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
            "arena-export-job",
            "episode-generation-job",
        ]

    def print_status_report(self):
        print("\n" + "=" * 70)
        print("BlueprintPipeline Job Registry Status Report")
        print("=" * 70)

        print(f"\nPipeline Mode: {self.get_pipeline_mode().value}")
        print(f"Text Stage 1 Ready: {self.is_stage1_ready()}")

        print("\n--- ACTIVE JOBS ---")
        for job in self.get_jobs_by_status(JobStatus.ACTIVE):
            print(f"  [{job.status.value:12}] {job.name:24} - {job.description[:40]}...")

        print("\n--- NEW JOBS ---")
        for job in self.get_jobs_by_status(JobStatus.NEW):
            print(f"  [{job.status.value:12}] {job.name:24} - {job.description[:40]}...")

        print("\n" + "=" * 70)


_registry: Optional[JobRegistry] = None


def get_registry() -> JobRegistry:
    global _registry
    if _registry is None:
        _registry = JobRegistry()
    return _registry


JOBS = tuple(get_registry().get_all_jobs())
JOBS_BY_NAME = {job.name: job for job in JOBS}
