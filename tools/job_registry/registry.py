"""
Job Registry Implementation.

Tracks all pipeline jobs, their deprecation status, and ZeroScene transition state.

ZeroScene Transition Status (as of 2025-12):
    DEPRECATED (replaced by ZeroScene):
        - seg-job: Segmentation + inventory → ZeroScene segmentation
        - multiview-job: Object isolation/views → ZeroScene foreground/background
        - scene-da3-job: Depth/point cloud → ZeroScene depth extraction
        - layout-job: Layout reconstruction → ZeroScene pose optimization
        - sam3d-job: 3D mesh generation → ZeroScene mesh reconstruction
        - hunyuan-job: Texture/refinement → ZeroScene PBR materials

    KEPT (still required for SimReady output):
        - zeroscene-job: NEW adapter for ZeroScene outputs
        - interactive-job: Articulation bridge (PhysX-Anything)
        - simready-job: Physics + manipulation hints
        - usd-assembly-job: USD scene assembly
        - replicator-job: Domain randomization scripts
        - variation-gen-job: Variation asset generation
        - isaac-lab-job: NEW Isaac Lab task generation

    FALLBACK (kept as fallback while ZeroScene stabilizes):
        - All deprecated jobs remain available as fallback
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class JobStatus(str, Enum):
    """Status of a job in the pipeline."""
    ACTIVE = "active"              # Fully operational, primary path
    DEPRECATED = "deprecated"      # Replaced by ZeroScene, use as fallback only
    FALLBACK = "fallback"          # Available as fallback during transition
    NEW = "new"                    # Newly added for ZeroScene pipeline
    EXPERIMENTAL = "experimental"  # Under development


class JobCategory(str, Enum):
    """Category of job in the pipeline."""
    RECONSTRUCTION = "reconstruction"    # Scene reconstruction (deprecated with ZeroScene)
    ENRICHMENT = "enrichment"            # Asset enrichment (kept)
    ASSEMBLY = "assembly"                # Scene assembly (kept)
    GENERATION = "generation"            # Synthetic data generation (kept)
    TRAINING = "training"                # Training preparation (kept)
    ADAPTER = "adapter"                  # Pipeline adapters (new)


class PipelineMode(str, Enum):
    """Pipeline execution mode."""
    ZEROSCENE_FIRST = "zeroscene_first"  # Use ZeroScene when available, fallback to Gemini
    GEMINI_ONLY = "gemini_only"          # Use only the Gemini reconstruction pipeline
    HYBRID = "hybrid"                     # Use both and compare (development mode)


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

    # Transition information
    replaced_by: Optional[str] = None
    fallback_for: Optional[str] = None
    zeroscene_replacement: Optional[str] = None

    # Environment variables
    required_env_vars: List[str] = field(default_factory=list)
    optional_env_vars: List[str] = field(default_factory=list)

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    # Notes
    deprecation_reason: Optional[str] = None
    migration_notes: Optional[str] = None

    @property
    def is_deprecated(self) -> bool:
        return self.status == JobStatus.DEPRECATED

    @property
    def is_active(self) -> bool:
        return self.status in (JobStatus.ACTIVE, JobStatus.NEW)


class JobRegistry:
    """Central registry for all pipeline jobs.

    Tracks job status, deprecation, and ZeroScene transition state.
    """

    def __init__(self):
        self._jobs: Dict[str, JobInfo] = {}
        self._initialize_jobs()

    def _initialize_jobs(self):
        """Initialize the job registry with all known jobs."""

        # =====================================================================
        # DEPRECATED JOBS (Replaced by ZeroScene)
        # Keep as fallback while ZeroScene is not fully operational
        # =====================================================================

        self._jobs["seg-job"] = JobInfo(
            name="seg-job",
            description="Gemini-based scene segmentation and inventory generation",
            status=JobStatus.DEPRECATED,
            category=JobCategory.RECONSTRUCTION,
            entry_script="seg-job/run_gemini_inventory.py",
            docker_image="seg-job",
            replaced_by="zeroscene-job",
            zeroscene_replacement="ZeroScene instance segmentation + depth extraction",
            required_env_vars=["BUCKET", "IMAGES_PREFIX", "SEG_PREFIX", "GEMINI_API_KEY"],
            depends_on=["image-upload"],
            outputs=["seg/inventory.json", "seg/dataset/data.yaml"],
            deprecation_reason=(
                "ZeroScene provides superior segmentation with 3D-aware instance "
                "segmentation and depth extraction. Gemini inventory generation "
                "is kept as a lightweight semantic enrichment step."
            ),
            migration_notes=(
                "For ZeroScene path: Use zeroscene-job to produce inventory.json. "
                "For fallback: This job remains fully functional."
            ),
        )

        self._jobs["multiview-job"] = JobInfo(
            name="multiview-job",
            description="Object isolation and generative view synthesis",
            status=JobStatus.DEPRECATED,
            category=JobCategory.RECONSTRUCTION,
            entry_script="multiview-job/run_multiview_gemini_generative.py",
            docker_image="multiview-job",
            replaced_by="zeroscene-job",
            zeroscene_replacement="ZeroScene foreground/background mesh separation",
            required_env_vars=["BUCKET", "SCENE_ID", "LAYOUT_PREFIX", "MULTIVIEW_PREFIX"],
            optional_env_vars=["ENABLE_GEMINI_VIEWS", "VIEWS_PER_OBJECT"],
            depends_on=["seg-job", "layout-job"],
            outputs=["multiview/obj_*/view_*.png"],
            deprecation_reason=(
                "ZeroScene directly produces foreground object meshes and background "
                "mesh, eliminating the need for 2D view synthesis."
            ),
            migration_notes=(
                "ZeroScene outputs per-object GLB meshes directly. "
                "Multiview crops are no longer needed for 3D reconstruction."
            ),
        )

        self._jobs["scene-da3-job"] = JobInfo(
            name="scene-da3-job",
            description="Depth-Anything 3 depth and point cloud extraction",
            status=JobStatus.DEPRECATED,
            category=JobCategory.RECONSTRUCTION,
            entry_script="scene-da3-job/run_da3.py",
            docker_image="scene-da3-job",
            replaced_by="zeroscene-job",
            zeroscene_replacement="ZeroScene depth extraction with 3D/2D projection losses",
            required_env_vars=["BUCKET", "SCENE_ID", "DATASET_PREFIX", "OUT_PREFIX"],
            depends_on=["seg-job"],
            outputs=["da3/da3_geom.npz", "da3/depth.png"],
            deprecation_reason=(
                "ZeroScene extracts depth with optimized 3D/2D projection losses, "
                "producing more accurate depth maps for scene reconstruction."
            ),
            migration_notes=(
                "ZeroScene depth is output to zeroscene/depth/depth.exr. "
                "Layout-job can consume either source."
            ),
        )

        self._jobs["layout-job"] = JobInfo(
            name="layout-job",
            description="Layout reconstruction from depth data",
            status=JobStatus.DEPRECATED,
            category=JobCategory.RECONSTRUCTION,
            entry_script="layout-job/run_layout.py",
            docker_image="layout-job",
            replaced_by="zeroscene-job",
            zeroscene_replacement="ZeroScene pose optimization using 3D/2D projection losses",
            required_env_vars=["BUCKET", "SCENE_ID", "DA3_PREFIX", "LAYOUT_PREFIX"],
            depends_on=["scene-da3-job"],
            outputs=["layout/scene_layout.json"],
            deprecation_reason=(
                "ZeroScene optimizes object poses using 3D and 2D projection losses, "
                "producing more accurate layout with proper scale."
            ),
            migration_notes=(
                "zeroscene-job adapter produces scene_layout_scaled.json "
                "compatible with downstream jobs."
            ),
        )

        self._jobs["sam3d-job"] = JobInfo(
            name="sam3d-job",
            description="SAM3D 3D mesh generation from multiview images",
            status=JobStatus.DEPRECATED,
            category=JobCategory.RECONSTRUCTION,
            entry_script="sam3d-job/run_sam3d_from_assets.py",
            docker_image="sam3d-job",
            replaced_by="zeroscene-job",
            zeroscene_replacement="ZeroScene triangle mesh reconstruction",
            required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX"],
            depends_on=["multiview-job"],
            outputs=["assets/obj_*/model.glb"],
            deprecation_reason=(
                "ZeroScene reconstructs complete scene with foreground objects "
                "and background mesh as explicit triangle meshes."
            ),
            migration_notes=(
                "ZeroScene outputs meshes to zeroscene/objects/obj_*/mesh.glb. "
                "zeroscene-job adapter copies them to standard assets/ location."
            ),
        )

        self._jobs["hunyuan-job"] = JobInfo(
            name="hunyuan-job",
            description="Hunyuan texture and material refinement",
            status=JobStatus.DEPRECATED,
            category=JobCategory.RECONSTRUCTION,
            entry_script="hunyuan-job/run_hunyuan.py",
            docker_image="hunyuan-job",
            replaced_by="zeroscene-job",
            zeroscene_replacement="ZeroScene PBR material estimation",
            required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX"],
            depends_on=["sam3d-job"],
            outputs=["assets/obj_*/textures/*"],
            deprecation_reason=(
                "ZeroScene includes PBR material estimation for better rendering "
                "realism, reducing need for post-process texture refinement."
            ),
            migration_notes=(
                "ZeroScene outputs material.json with PBR parameters. "
                "simready-job can use these for physics material hints."
            ),
        )

        # =====================================================================
        # KEPT JOBS (Still Required for SimReady Output)
        # These jobs are still necessary because ZeroScene does not provide
        # a fully SimReady Isaac Sim training package end-to-end.
        # =====================================================================

        self._jobs["zeroscene-job"] = JobInfo(
            name="zeroscene-job",
            description="Adapter converting ZeroScene outputs to BlueprintPipeline format",
            status=JobStatus.NEW,
            category=JobCategory.ADAPTER,
            entry_script="zeroscene-job/zeroscene_adapter_job.py",
            docker_image="zeroscene-job",
            fallback_for="seg-job,multiview-job,scene-da3-job,layout-job,sam3d-job,hunyuan-job",
            required_env_vars=["SCENE_ID", "ZEROSCENE_PREFIX", "ASSETS_PREFIX", "LAYOUT_PREFIX"],
            optional_env_vars=["GEMINI_API_KEY", "OPENAI_API_KEY", "TRUST_ZEROSCENE_SCALE"],
            depends_on=["zeroscene-reconstruction"],
            outputs=[
                "assets/scene_manifest.json",
                "layout/scene_layout_scaled.json",
                "seg/inventory.json",
                "assets/obj_*/asset.glb",
            ],
            migration_notes=(
                "This adapter is the critical integration layer between ZeroScene "
                "and the rest of the BlueprintPipeline."
            ),
        )

        self._jobs["interactive-job"] = JobInfo(
            name="interactive-job",
            description="Articulation bridge using PhysX-Anything",
            status=JobStatus.ACTIVE,
            category=JobCategory.ENRICHMENT,
            entry_script="interactive-job/run_interactive_assets.py",
            docker_image="interactive-job",
            required_env_vars=["BUCKET", "SCENE_ID", "ASSETS_PREFIX"],
            optional_env_vars=["PHYSX_SERVICE_URL", "TIMEOUT_SECONDS"],
            depends_on=["zeroscene-job"],  # or seg-job in fallback mode
            outputs=["assets/interactive/obj_*/articulated.usda"],
            migration_notes=(
                "Works with both ZeroScene and Gemini pipelines. "
                "Requires articulated object candidates from inventory."
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
            depends_on=["zeroscene-job", "interactive-job"],
            outputs=[
                "assets/obj_*/simready.usda",
                "assets/obj_*/metadata.json",
            ],
            migration_notes=(
                "Now supports both Gemini and OpenAI for physics estimation. "
                "ZeroScene materials provide additional hints for friction/roughness."
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
                "Works with both pipeline paths. Reads layout from either "
                "ZeroScene adapter or layout-job output."
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
                "Reads inventory from seg/inventory.json (produced by either pipeline). "
                "Now supports OpenAI as alternative to Gemini for scene analysis."
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
                "Now supports OpenAI DALL-E as alternative to Gemini for image generation."
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
                "New job for generating Isaac Lab training configurations. "
                "Uses the same PolicyTarget concepts as replicator-job."
            ),
        )

        # =====================================================================
        # SUPPORTING JOBS
        # =====================================================================

        self._jobs["scale-job"] = JobInfo(
            name="scale-job",
            description="Scale calibration and reference object detection",
            status=JobStatus.ACTIVE,
            category=JobCategory.ENRICHMENT,
            entry_script="scale-job/calibrate_scale.py",
            docker_image="scale-job",
            required_env_vars=["BUCKET", "SCENE_ID", "LAYOUT_PREFIX"],
            depends_on=["layout-job"],  # or zeroscene-job
            outputs=["layout/scene_layout_scaled.json"],
            migration_notes=(
                "May be optional with ZeroScene if TRUST_ZEROSCENE_SCALE=true. "
                "Otherwise runs after zeroscene-job to calibrate metric scale."
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

    def get_deprecated_jobs(self) -> List[JobInfo]:
        """Get all deprecated jobs."""
        return self.get_jobs_by_status(JobStatus.DEPRECATED)

    def get_active_jobs(self, mode: PipelineMode = PipelineMode.ZEROSCENE_FIRST) -> List[JobInfo]:
        """Get active jobs for a pipeline mode."""
        if mode == PipelineMode.GEMINI_ONLY:
            # Include deprecated jobs, exclude zeroscene-specific
            return [
                j for j in self._jobs.values()
                if j.status != JobStatus.NEW or j.name not in ("zeroscene-job",)
            ]
        elif mode == PipelineMode.ZEROSCENE_FIRST:
            # Exclude deprecated jobs, include new
            return [
                j for j in self._jobs.values()
                if j.status != JobStatus.DEPRECATED
            ]
        else:  # HYBRID
            return list(self._jobs.values())

    def is_deprecated(self, job_name: str) -> bool:
        """Check if a job is deprecated."""
        job = self.get_job(job_name)
        return job.is_deprecated if job else False

    def get_replacement(self, job_name: str) -> Optional[str]:
        """Get the replacement job for a deprecated job."""
        job = self.get_job(job_name)
        return job.replaced_by if job else None

    def get_fallback(self, job_name: str) -> Optional[List[str]]:
        """Get fallback jobs for a given job."""
        job = self.get_job(job_name)
        if job and job.fallback_for:
            return job.fallback_for.split(",")
        return None

    def is_zeroscene_ready(self) -> bool:
        """Check if ZeroScene pipeline is ready (all required jobs exist)."""
        required_new_jobs = ["zeroscene-job"]
        for name in required_new_jobs:
            job = self.get_job(name)
            if not job or job.status != JobStatus.NEW:
                return False
        return True

    def get_pipeline_mode(self) -> PipelineMode:
        """Get current pipeline mode from environment."""
        mode_str = os.getenv("PIPELINE_MODE", "zeroscene_first").lower()

        if mode_str == "gemini_only":
            return PipelineMode.GEMINI_ONLY
        elif mode_str == "hybrid":
            return PipelineMode.HYBRID
        else:
            return PipelineMode.ZEROSCENE_FIRST

    def get_job_sequence(self, mode: Optional[PipelineMode] = None) -> List[str]:
        """Get the recommended job execution sequence for a pipeline mode."""
        if mode is None:
            mode = self.get_pipeline_mode()

        if mode == PipelineMode.ZEROSCENE_FIRST:
            return [
                "zeroscene-reconstruction",  # External ZeroScene
                "zeroscene-job",
                "scale-job",  # Optional
                "interactive-job",
                "simready-job",
                "usd-assembly-job",
                "replicator-job",
                "variation-gen-job",
                "isaac-lab-job",
            ]
        elif mode == PipelineMode.GEMINI_ONLY:
            return [
                "seg-job",
                "scene-da3-job",
                "layout-job",
                "scale-job",
                "multiview-job",
                "sam3d-job",
                "hunyuan-job",
                "interactive-job",
                "simready-job",
                "usd-assembly-job",
                "replicator-job",
                "variation-gen-job",
                "isaac-lab-job",
            ]
        else:  # HYBRID
            # Run both and compare
            return self.get_job_sequence(PipelineMode.ZEROSCENE_FIRST)

    def print_status_report(self):
        """Print a status report of all jobs."""
        print("\n" + "=" * 70)
        print("BlueprintPipeline Job Registry Status Report")
        print("=" * 70)

        print(f"\nPipeline Mode: {self.get_pipeline_mode().value}")
        print(f"ZeroScene Ready: {self.is_zeroscene_ready()}")

        print("\n--- DEPRECATED JOBS (Replaced by ZeroScene) ---")
        for job in self.get_deprecated_jobs():
            print(f"  [{job.status.value:12}] {job.name:20} → {job.replaced_by or 'N/A'}")
            if job.deprecation_reason:
                print(f"                  Reason: {job.deprecation_reason[:60]}...")

        print("\n--- ACTIVE JOBS (Still Required) ---")
        for job in self.get_jobs_by_status(JobStatus.ACTIVE):
            print(f"  [{job.status.value:12}] {job.name:20} - {job.description[:40]}...")

        print("\n--- NEW JOBS (ZeroScene Pipeline) ---")
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
