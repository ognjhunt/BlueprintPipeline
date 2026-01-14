#!/usr/bin/env python3
"""
Local Pipeline Runner for BlueprintPipeline.

Runs the full pipeline locally without requiring GCS or Cloud Run.
Useful for development, testing, and debugging.

Usage:
    # Generate mock scene and run pipeline
    python fixtures/generate_mock_regen3d.py --scene-id test_kitchen --output-dir ./test_scenes
    python tools/run_local_pipeline.py --scene-dir ./test_scenes/scenes/test_kitchen

    # Run with specific steps
    python tools/run_local_pipeline.py --scene-dir ./scene --steps regen3d,simready,usd

    # Run with validation
    python tools/run_local_pipeline.py --scene-dir ./scene --validate

Pipeline Steps:
    1. regen3d   - Adapt 3D-RE-GEN outputs to BlueprintPipeline format
    2. scale     - (Optional) Scale calibration
    3. interactive - Particulate articulation (requires PARTICULATE_ENDPOINT)
    4. simready  - Prepare physics-ready assets
    5. usd       - Assemble scene.usda
    6. replicator - Generate Replicator bundle
    7. variation-gen - Generate variation assets for Genie Sim export
    8. isaac-lab - Generate Isaac Lab task package
    9. genie-sim-export - Export scene bundle for Genie Sim
    10. genie-sim-submit - Submit/run Genie Sim generation (API or local)
    11. dwm       - Generate DWM conditioning data (egocentric videos + hand meshes)
    12. dwm-inference - Run DWM model to generate interaction videos for each bundle
    13. validate  - QA validation

Note: DWM steps are optional and only included by default when --enable-dwm is set.

References:
- 3D-RE-GEN (arXiv:2512.17459): "image → sim-ready 3D reconstruction"
  Paper: https://arxiv.org/abs/2512.17459
  Project: https://3dregen.jdihlmann.com/

- DWM (arXiv:2512.17907): Dexterous World Models for egocentric interaction
  Paper: https://arxiv.org/abs/2512.17907
  Project: https://snuvclab.github.io/dwm/
"""

import argparse
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class PipelineStep(str, Enum):
    """Pipeline steps in execution order."""
    REGEN3D = "regen3d"
    SCALE = "scale"
    INTERACTIVE = "interactive"
    SIMREADY = "simready"
    USD = "usd"
    REPLICATOR = "replicator"
    VARIATION_GEN = "variation-gen"
    ISAAC_LAB = "isaac-lab"
    GENIESIM_EXPORT = "genie-sim-export"
    GENIESIM_SUBMIT = "genie-sim-submit"
    DWM = "dwm"  # Dexterous World Model preparation
    DWM_INFERENCE = "dwm-inference"  # Run DWM model on prepared bundles
    DREAM2FLOW = "dream2flow"  # Dream2Flow preparation (arXiv:2512.24766)
    DREAM2FLOW_INFERENCE = "dream2flow-inference"  # Run Dream2Flow inference
    VALIDATE = "validate"


@dataclass
class StepResult:
    """Result of a pipeline step."""
    step: PipelineStep
    success: bool
    duration_seconds: float
    message: str = ""
    outputs: Dict[str, Any] = None

    def __post_init__(self):
        if self.outputs is None:
            self.outputs = {}


class LocalPipelineRunner:
    """Run BlueprintPipeline locally for testing."""

    # Default step order
    DEFAULT_STEPS = [
        PipelineStep.REGEN3D,
        PipelineStep.SIMREADY,
        PipelineStep.USD,
        PipelineStep.REPLICATOR,
        PipelineStep.ISAAC_LAB,
        PipelineStep.DREAM2FLOW,  # Dream2Flow conditioning data (arXiv:2512.24766)
        PipelineStep.DREAM2FLOW_INFERENCE,  # Run Dream2Flow inference
    ]

    DWM_STEPS = [
        PipelineStep.DWM,  # DWM conditioning data generation
        PipelineStep.DWM_INFERENCE,  # Run DWM model on bundles
    ]

    def __init__(
        self,
        scene_dir: Path,
        verbose: bool = True,
        skip_interactive: bool = True,
        environment_type: str = "kitchen",
        enable_dwm: bool = False,
    ):
        """Initialize the local pipeline runner.

        Args:
            scene_dir: Path to scene directory (contains regen3d/, etc.)
            verbose: Print detailed progress
            skip_interactive: Skip interactive-job (requires external service)
            environment_type: Environment type for policy selection
        """
        self.scene_dir = Path(scene_dir).resolve()
        self.verbose = verbose
        self.skip_interactive = skip_interactive
        self.environment_type = environment_type
        self.enable_dwm = enable_dwm
        self.environment = os.getenv("BP_ENV", "development").lower()

        # Derive scene ID from directory name
        self.scene_id = self.scene_dir.name

        # Setup paths
        self.regen3d_dir = self.scene_dir / "regen3d"
        self.assets_dir = self.scene_dir / "assets"
        self.layout_dir = self.scene_dir / "layout"
        self.seg_dir = self.scene_dir / "seg"
        self.usd_dir = self.scene_dir / "usd"
        self.replicator_dir = self.scene_dir / "replicator"
        self.isaac_lab_dir = self.scene_dir / "isaac_lab"
        self.geniesim_dir = self.scene_dir / "geniesim"
        self.episodes_dir = self.scene_dir / "episodes"
        self.dwm_dir = self.scene_dir / "dwm"  # DWM conditioning data
        self.dream2flow_dir = self.scene_dir / "dream2flow"  # Dream2Flow conditioning data

        self.results: List[StepResult] = []

    def log(self, msg: str, level: str = "INFO") -> None:
        """Log a message."""
        if self.verbose:
            print(f"[LOCAL-PIPELINE] [{level}] {msg}")

    def _write_marker(self, marker_path: Path, status: str) -> None:
        """Write a simple JSON marker file."""
        marker_path.write_text(json.dumps({
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "scene_id": self.scene_id,
        }, indent=2))

    def run(
        self,
        steps: Optional[List[PipelineStep]] = None,
        run_validation: bool = False,
    ) -> bool:
        """Run the pipeline.

        Args:
            steps: Specific steps to run (default: all applicable)
            run_validation: Run QA validation at the end

        Returns:
            True if all steps succeeded
        """
        if steps is None:
            steps = self._resolve_default_steps()

        if run_validation and PipelineStep.VALIDATE not in steps:
            steps.append(PipelineStep.VALIDATE)

        self._apply_labs_flags(run_validation=run_validation)

        self.log("=" * 60)
        self.log("BlueprintPipeline Local Runner")
        self.log("=" * 60)
        self.log(f"Scene directory: {self.scene_dir}")
        self.log(f"Scene ID: {self.scene_id}")
        self.log(f"Steps: {[s.value for s in steps]}")
        self.log("=" * 60)

        # Check prerequisites
        if not self.regen3d_dir.is_dir():
            self.log(f"ERROR: 3D-RE-GEN output not found at {self.regen3d_dir}", "ERROR")
            self.log("Run: python fixtures/generate_mock_regen3d.py first", "ERROR")
            return False

        # Create output directories
        for d in [self.assets_dir, self.layout_dir, self.seg_dir,
                  self.usd_dir, self.replicator_dir, self.isaac_lab_dir,
                  self.geniesim_dir, self.episodes_dir,
                  self.dwm_dir, self.dream2flow_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Run each step
        all_success = True
        for step in steps:
            result = self._run_step(step)
            self.results.append(result)

            if not result.success:
                all_success = False
                self.log(f"Step {step.value} failed: {result.message}", "ERROR")
                # Continue with remaining steps for partial results

        # Print summary
        self._print_summary()

        return all_success

    def _apply_labs_flags(self, run_validation: bool) -> None:
        """Apply production/labs flags for staging or lab validation runs."""
        staging_e2e = os.environ.get("RUN_STAGING_E2E") == "1"
        labs_staging = os.environ.get("LABS_STAGING", "").lower() in {"1", "true", "yes"}
        labs_validation = os.environ.get("LABS_VALIDATION", "").lower() in {"1", "true", "yes"}

        if not (staging_e2e or labs_staging or (labs_validation and run_validation)):
            return

        os.environ.setdefault("LABS_STAGING", "1")
        os.environ.setdefault("DATA_QUALITY_LEVEL", "production")
        os.environ.setdefault("ISAAC_SIM_REQUIRED", "true")
        os.environ.setdefault("PRODUCTION_MODE", "true")
        self.log("Applied labs/production flags for staging or lab validation runs")

    def _resolve_default_steps(self) -> List[PipelineStep]:
        """Resolve default steps, using pipeline selector for Genie Sim mode."""
        try:
            from tools.pipeline_selector.selector import PipelineSelector, DataGenerationBackend
        except ImportError:
            self.log("Pipeline selector not available; falling back to default steps", "WARNING")
            steps = self.DEFAULT_STEPS.copy()
            if self.enable_dwm:
                insert_at = self.DEFAULT_STEPS.index(PipelineStep.DREAM2FLOW)
                steps[insert_at:insert_at] = self.DWM_STEPS
            return steps

        selector = PipelineSelector(scene_root=self.scene_dir)
        decision = selector.select(self.scene_dir)
        if decision.data_backend == DataGenerationBackend.GENIESIM:
            steps = self._map_jobs_to_steps(decision.job_sequence)
        else:
            steps = self.DEFAULT_STEPS.copy()

        if self.enable_dwm:
            insert_at = steps.index(PipelineStep.DREAM2FLOW) if PipelineStep.DREAM2FLOW in steps else len(steps)
            steps[insert_at:insert_at] = self.DWM_STEPS

        return steps

    def _map_jobs_to_steps(self, job_sequence: List[str]) -> List[PipelineStep]:
        """Map pipeline selector job names to local runner steps."""
        mapping = {
            "regen3d-job": PipelineStep.REGEN3D,
            "scale-job": PipelineStep.SCALE,
            "interactive-job": PipelineStep.INTERACTIVE,
            "simready-job": PipelineStep.SIMREADY,
            "usd-assembly-job": PipelineStep.USD,
            "replicator-job": PipelineStep.REPLICATOR,
            "variation-gen-job": PipelineStep.VARIATION_GEN,
            "isaac-lab-job": PipelineStep.ISAAC_LAB,
            "genie-sim-export-job": PipelineStep.GENIESIM_EXPORT,
            "genie-sim-submit-job": PipelineStep.GENIESIM_SUBMIT,
        }
        steps: List[PipelineStep] = []
        for job_name in job_sequence:
            step = mapping.get(job_name)
            if step is None:
                self.log(f"Skipping unsupported local job: {job_name}", "WARNING")
                continue
            steps.append(step)
        return steps

    def _run_step(self, step: PipelineStep) -> StepResult:
        """Run a single pipeline step."""
        import time
        start_time = time.time()

        self.log(f"\n--- Running step: {step.value} ---")

        try:
            if step == PipelineStep.REGEN3D:
                result = self._run_regen3d_adapter()
            elif step == PipelineStep.SCALE:
                result = self._run_scale()
            elif step == PipelineStep.INTERACTIVE:
                result = self._run_interactive()
            elif step == PipelineStep.SIMREADY:
                result = self._run_simready()
            elif step == PipelineStep.USD:
                result = self._run_usd_assembly()
            elif step == PipelineStep.REPLICATOR:
                result = self._run_replicator()
            elif step == PipelineStep.VARIATION_GEN:
                result = self._run_variation_gen()
            elif step == PipelineStep.ISAAC_LAB:
                result = self._run_isaac_lab()
            elif step == PipelineStep.GENIESIM_EXPORT:
                result = self._run_geniesim_export()
            elif step == PipelineStep.GENIESIM_SUBMIT:
                result = self._run_geniesim_submit()
            elif step == PipelineStep.DWM:
                result = self._run_dwm()
            elif step == PipelineStep.DWM_INFERENCE:
                result = self._run_dwm_inference()
            elif step == PipelineStep.DREAM2FLOW:
                result = self._run_dream2flow()
            elif step == PipelineStep.DREAM2FLOW_INFERENCE:
                result = self._run_dream2flow_inference()
            elif step == PipelineStep.VALIDATE:
                result = self._run_validation()
            else:
                result = StepResult(
                    step=step,
                    success=False,
                    duration_seconds=0,
                    message=f"Unknown step: {step.value}",
                )
        except Exception as e:
            result = StepResult(
                step=step,
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Exception: {e}\n{traceback.format_exc()}",
            )

        result.duration_seconds = time.time() - start_time
        status = "OK" if result.success else "FAILED"
        self.log(f"Step {step.value}: {status} ({result.duration_seconds:.2f}s)")

        return result

    def _run_regen3d_adapter(self) -> StepResult:
        """Run the 3D-RE-GEN adapter."""
        from tools.regen3d_adapter import Regen3DAdapter

        adapter = Regen3DAdapter(verbose=self.verbose)

        # Load 3D-RE-GEN outputs
        regen3d_output = adapter.load_regen3d_output(self.regen3d_dir)
        self.log(f"Loaded {len(regen3d_output.objects)} objects from 3D-RE-GEN")

        # Generate manifest
        manifest = adapter.create_manifest(
            regen3d_output,
            scene_id=self.scene_id,
            environment_type=self.environment_type,
        )
        manifest_path = self.assets_dir / "scene_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        manifest = self._annotate_articulation_requirements(manifest, manifest_path)
        self.log(f"Wrote manifest: {manifest_path}")

        # Generate layout
        layout = adapter.create_layout(regen3d_output, apply_scale_factor=1.0)
        layout_path = self.layout_dir / "scene_layout_scaled.json"
        layout_path.write_text(json.dumps(layout, indent=2))
        self.log(f"Wrote layout: {layout_path}")

        # Copy assets
        asset_paths = adapter.copy_assets(
            regen3d_output,
            self.scene_dir,
            assets_prefix="assets",
        )
        self.log(f"Copied {len(asset_paths)} assets")

        # Generate inventory
        inventory = self._generate_inventory(regen3d_output)
        inventory_path = self.seg_dir / "inventory.json"
        inventory_path.write_text(json.dumps(inventory, indent=2))
        self.log(f"Wrote inventory: {inventory_path}")

        # Write completion marker
        marker_path = self.assets_dir / ".regen3d_complete"
        marker_content = {
            "status": "complete",
            "scene_id": self.scene_id,
            "objects_count": len(regen3d_output.objects),
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }
        marker_path.write_text(json.dumps(marker_content, indent=2))

        return StepResult(
            step=PipelineStep.REGEN3D,
            success=True,
            duration_seconds=0,
            message="3D-RE-GEN adapter completed",
            outputs={
                "manifest": str(manifest_path),
                "layout": str(layout_path),
                "objects_count": len(regen3d_output.objects),
            },
        )

    def _annotate_articulation_requirements(
        self,
        manifest: Dict[str, Any],
        manifest_path: Path,
    ) -> Dict[str, Any]:
        """Detect articulated objects and annotate the manifest."""
        from tools.articulation import detect_scene_articulations

        results = detect_scene_articulations(
            manifest,
            use_llm=False,
            verbose=self.verbose,
        )

        required_ids = []
        for obj in manifest.get("objects", []):
            obj_id = obj.get("id")
            if not obj_id or obj.get("sim_role") in {"background", "scene_shell"}:
                continue

            result = results.get(obj_id)
            if not result or not result.has_articulation:
                continue

            required_ids.append(obj_id)
            articulation = obj.get("articulation") or {}
            detection_payload = {
                "type": result.articulation_type.value,
                "confidence": result.confidence,
                "method": result.detection_method,
            }
            if result.joint_axis is not None:
                detection_payload["axis"] = [float(v) for v in result.joint_axis.tolist()]
            if result.joint_range is not None:
                detection_payload["range"] = [float(result.joint_range[0]), float(result.joint_range[1])]

            articulation.update({
                "required": True,
                "detection": detection_payload,
            })
            obj["articulation"] = articulation

            if obj.get("sim_role") in {"unknown", "static", None, ""}:
                obj["sim_role"] = self._infer_articulation_role(obj)

        metadata = manifest.get("metadata") or {}
        metadata["articulation_detection"] = {
            "required_count": len(required_ids),
            "required_objects": required_ids,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source": "heuristic",
        }
        manifest["metadata"] = metadata
        manifest_path.write_text(json.dumps(manifest, indent=2))

        if required_ids:
            self.log(f"Detected {len(required_ids)} articulated objects requiring interactive processing")

        return manifest

    def _infer_articulation_role(self, obj: Dict[str, Any]) -> str:
        """Infer an articulated sim_role based on object category."""
        category = (obj.get("category")
                    or (obj.get("semantics") or {}).get("category")
                    or obj.get("id", ""))
        category = category.lower()

        appliance_keywords = {
            "refrigerator",
            "fridge",
            "oven",
            "microwave",
            "dishwasher",
            "washing_machine",
            "washer",
            "dryer",
            "stove",
            "range",
        }
        if any(keyword in category for keyword in appliance_keywords):
            return "articulated_appliance"
        return "articulated_furniture"

    def _required_articulation_ids(self, manifest: Dict[str, Any]) -> List[str]:
        """Return object IDs that require articulation."""
        required = []
        for obj in manifest.get("objects", []):
            obj_id = obj.get("id")
            if not obj_id:
                continue
            if obj.get("sim_role") in {"articulated_furniture", "articulated_appliance"}:
                required.append(obj_id)
                continue
            articulation = obj.get("articulation") or {}
            if articulation.get("required"):
                required.append(obj_id)
        return required

    def _generate_inventory(self, regen3d_output) -> Dict[str, Any]:
        """Generate semantic inventory from 3D-RE-GEN output."""
        objects = []
        for obj in regen3d_output.objects:
            inv_obj = {
                "id": obj.id,
                "category": obj.category or "object",
                "short_description": obj.description or f"Object {obj.id}",
                "sim_role": obj.sim_role if obj.sim_role != "unknown" else "static",
                "bounds": obj.bounds,
                "is_floor_contact": obj.pose.is_floor_contact,
            }
            objects.append(inv_obj)

        return {
            "scene_id": self.scene_id,
            "environment_type": self.environment_type,
            "total_objects": len(objects),
            "objects": objects,
            "metadata": {
                "source": "regen3d_adapter",
                "generated_at": datetime.utcnow().isoformat() + "Z",
            },
        }

    def _run_scale(self) -> StepResult:
        """Run scale calibration (optional)."""
        # For local testing, we trust the 3D-RE-GEN scale
        self.log("Scale calibration: using 3D-RE-GEN scale (trusted)")

        return StepResult(
            step=PipelineStep.SCALE,
            success=True,
            duration_seconds=0,
            message="Using 3D-RE-GEN scale",
        )

    def _run_interactive(self) -> StepResult:
        """Run interactive asset processing."""
        production_mode = self.environment == "production"
        manifest_path = self.assets_dir / "scene_manifest.json"
        required_ids: List[str] = []
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            required_ids = self._required_articulation_ids(manifest)

        if self.skip_interactive:
            if production_mode:
                return StepResult(
                    step=PipelineStep.INTERACTIVE,
                    success=False,
                    duration_seconds=0,
                    message=(
                        "Interactive job is required in production. "
                        "Disable skip_interactive and provide PARTICULATE_ENDPOINT."
                    ),
                    outputs={"required_articulations": required_ids},
                )
            if required_ids:
                return StepResult(
                    step=PipelineStep.INTERACTIVE,
                    success=False,
                    duration_seconds=0,
                    message=(
                        "Articulation required but interactive job is disabled. "
                        "Run with --with-interactive and set PARTICULATE_ENDPOINT."
                    ),
                    outputs={"required_articulations": required_ids},
                )
            self.log("Interactive job skipped (requires PARTICULATE_ENDPOINT)")
            return StepResult(
                step=PipelineStep.INTERACTIVE,
                success=True,
                duration_seconds=0,
                message="Skipped (no PARTICULATE_ENDPOINT)",
            )

        # Would need PARTICULATE_ENDPOINT to run
        endpoint = os.getenv("PARTICULATE_ENDPOINT")
        if not endpoint:
            if production_mode:
                return StepResult(
                    step=PipelineStep.INTERACTIVE,
                    success=False,
                    duration_seconds=0,
                    message="PARTICULATE_ENDPOINT is required in production",
                    outputs={"required_articulations": required_ids},
                )
            if required_ids:
                return StepResult(
                    step=PipelineStep.INTERACTIVE,
                    success=False,
                    duration_seconds=0,
                    message="Articulation required but PARTICULATE_ENDPOINT not set",
                    outputs={"required_articulations": required_ids},
                )
            return StepResult(
                step=PipelineStep.INTERACTIVE,
                success=True,
                duration_seconds=0,
                message="Skipped (PARTICULATE_ENDPOINT not set)",
            )

        interactive_script = REPO_ROOT / "interactive-job" / "run_interactive_assets.py"
        env = os.environ.copy()
        env.update({
            "ASSETS_PREFIX": str(self.assets_dir),
            "REGEN3D_PREFIX": str(self.regen3d_dir),
            "SCENE_ID": self.scene_id,
            "PARTICULATE_ENDPOINT": endpoint,
            "PRODUCTION_MODE": "false",
            "DISALLOW_PLACEHOLDER_URDF": os.getenv("DISALLOW_PLACEHOLDER_URDF", "false"),
        })

        self.log("Running interactive-job entrypoint locally")
        proc = subprocess.run(
            [sys.executable, str(interactive_script)],
            cwd=str(REPO_ROOT),
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            return StepResult(
                step=PipelineStep.INTERACTIVE,
                success=False,
                duration_seconds=0,
                message=f"interactive-job failed with exit code {proc.returncode}",
            )

        failure_marker = self.assets_dir / ".interactive_failed"
        if failure_marker.is_file():
            failure_payload = json.loads(failure_marker.read_text())
            return StepResult(
                step=PipelineStep.INTERACTIVE,
                success=False,
                duration_seconds=0,
                message=f"interactive-job reported failure: {failure_payload.get('reason')}",
                outputs={"failure_payload": failure_payload},
            )

        results_path = self.assets_dir / "interactive" / "interactive_results.json"
        if not results_path.is_file():
            return StepResult(
                step=PipelineStep.INTERACTIVE,
                success=False,
                duration_seconds=0,
                message="interactive-job results not found",
            )

        results_data = json.loads(results_path.read_text())
        if required_ids:
            result_by_id = {
                str(r.get("id")): r for r in results_data.get("objects", [])
            }
            missing = [
                obj_id for obj_id in required_ids
                if not result_by_id.get(str(obj_id), {}).get("is_articulated")
            ]
            if missing:
                return StepResult(
                    step=PipelineStep.INTERACTIVE,
                    success=False,
                    duration_seconds=0,
                    message="Articulation required but not available for all objects",
                    outputs={"missing_articulations": missing},
                )

        return StepResult(
            step=PipelineStep.INTERACTIVE,
            success=True,
            duration_seconds=0,
            message="Interactive job completed",
            outputs={
                "interactive_results": str(results_path),
                "articulated_count": results_data.get("articulated_count", 0),
            },
        )

    def _run_simready(self) -> StepResult:
        """Run simready preparation."""
        # Load manifest
        manifest_path = self.assets_dir / "scene_manifest.json"
        if not manifest_path.is_file():
            return StepResult(
                step=PipelineStep.SIMREADY,
                success=False,
                duration_seconds=0,
                message="Manifest not found - run regen3d step first",
            )

        manifest = json.loads(manifest_path.read_text())

        # For local testing, add basic physics properties to each object
        for obj in manifest.get("objects", []):
            if not obj.get("physics"):
                mass = self._estimate_mass(obj)
                friction_static = 0.5
                friction_dynamic = 0.4
                restitution = 0.1

                obj["physics"] = {
                    "dynamic": obj.get("sim_role") in ["manipulable_object", "clutter"],
                    "mass_kg": mass,
                    "friction_static": friction_static,
                    "friction_dynamic": friction_dynamic,
                    "restitution": restitution,
                    "center_of_mass_offset": [0.0, 0.0, 0.0],
                    # Sim2Real distribution ranges for domain randomization
                    "mass_kg_range": [mass * 0.8, mass * 1.2],
                    "friction_static_range": [friction_static * 0.85, min(1.5, friction_static * 1.15)],
                    "friction_dynamic_range": [friction_dynamic * 0.85, min(1.2, friction_dynamic * 1.15)],
                    "restitution_range": [max(0.0, restitution * 0.7), min(1.0, restitution * 1.3)],
                    "center_of_mass_noise": [0.005, 0.005, 0.005],
                }

        # Write updated manifest
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Write simready completion marker
        marker_path = self.assets_dir / ".simready_complete"
        marker_path.write_text(f"completed at {datetime.utcnow().isoformat()}Z\n")

        self.log(f"Added physics to {len(manifest.get('objects', []))} objects")

        return StepResult(
            step=PipelineStep.SIMREADY,
            success=True,
            duration_seconds=0,
            message="SimReady preparation completed",
            outputs={"objects_processed": len(manifest.get("objects", []))},
        )

    def _estimate_mass(self, obj: Dict[str, Any]) -> float:
        """Estimate mass from dimensions and material hints."""
        dims = obj.get("dimensions_est", {})
        volume = (
            dims.get("width", 0.1) *
            dims.get("height", 0.1) *
            dims.get("depth", 0.1)
        )

        # Default density (like plastic)
        density = 1000  # kg/m^3

        material = obj.get("physics_hints", {}).get("material_type", "")
        if material in ["metal", "steel"]:
            density = 7800
        elif material in ["wood"]:
            density = 600
        elif material in ["ceramic", "stone"]:
            density = 2500
        elif material in ["fabric", "plastic"]:
            density = 200

        return max(0.01, volume * density)  # Minimum 10g

    def _run_usd_assembly(self) -> StepResult:
        """Run USD assembly."""
        # Check prerequisites
        manifest_path = self.assets_dir / "scene_manifest.json"
        layout_path = self.layout_dir / "scene_layout_scaled.json"

        if not manifest_path.is_file():
            return StepResult(
                step=PipelineStep.USD,
                success=False,
                duration_seconds=0,
                message="Manifest not found",
            )

        if not layout_path.is_file():
            return StepResult(
                step=PipelineStep.USD,
                success=False,
                duration_seconds=0,
                message="Layout not found",
            )

        manifest = json.loads(manifest_path.read_text())
        layout = json.loads(layout_path.read_text())

        # Generate a minimal scene.usda
        usda_content = self._generate_usda(manifest, layout)
        usda_path = self.usd_dir / "scene.usda"
        usda_path.write_text(usda_content)
        self._write_marker(self.assets_dir / ".usd_assembly_complete", status="completed")

        self.log(f"Generated USD: {usda_path}")

        return StepResult(
            step=PipelineStep.USD,
            success=True,
            duration_seconds=0,
            message="USD assembly completed",
            outputs={"scene_usda": str(usda_path)},
        )

    def _generate_usda(self, manifest: Dict, layout: Dict) -> str:
        """Generate a minimal scene.usda file."""
        meters_per_unit = manifest.get("scene", {}).get("meters_per_unit", 1.0)
        coord_frame = manifest.get("scene", {}).get("coordinate_frame", "Y")

        lines = [
            '#usda 1.0',
            '(',
            f'    metersPerUnit = {meters_per_unit}',
            f'    upAxis = "{coord_frame.upper()}"',
            '    defaultPrim = "World"',
            ')',
            '',
            'def Xform "World" {',
            '    def Xform "Scene" {',
        ]

        # Add objects as references
        for obj in manifest.get("objects", []):
            obj_id = obj.get("id", "unknown")
            asset_path = obj.get("asset", {}).get("path", f"assets/obj_{obj_id}/asset.glb")

            # Get transform
            transform = obj.get("transform", {})
            pos = transform.get("position", {})
            px = pos.get("x", 0)
            py = pos.get("y", 0)
            pz = pos.get("z", 0)

            lines.append(f'        def Xform "obj_{obj_id}" (')
            lines.append(f'            # Reference to: {asset_path}')
            lines.append(f'        ) {{')
            lines.append(f'            double3 xformOp:translate = ({px}, {py}, {pz})')
            lines.append(f'            uniform token[] xformOpOrder = ["xformOp:translate"]')
            lines.append(f'        }}')
            lines.append('')

        lines.append('    }')
        lines.append('}')

        return '\n'.join(lines)

    def _run_replicator(self) -> StepResult:
        """Run replicator bundle generation."""
        # Load manifest and inventory
        manifest_path = self.assets_dir / "scene_manifest.json"
        inventory_path = self.seg_dir / "inventory.json"

        if not manifest_path.is_file():
            return StepResult(
                step=PipelineStep.REPLICATOR,
                success=False,
                duration_seconds=0,
                message="Manifest not found",
            )

        manifest = json.loads(manifest_path.read_text())
        inventory = {}
        if inventory_path.is_file():
            inventory = json.loads(inventory_path.read_text())

        # Create replicator directories
        policies_dir = self.replicator_dir / "policies"
        policies_dir.mkdir(parents=True, exist_ok=True)

        # Generate bundle metadata
        bundle_metadata = {
            "scene_id": self.scene_id,
            "environment_type": self.environment_type,
            "policies": ["manipulation", "navigation"],
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        (self.replicator_dir / "bundle_metadata.json").write_text(
            json.dumps(bundle_metadata, indent=2)
        )

        # Generate minimal placement regions
        placement_regions = self._generate_placement_regions(manifest)
        (self.replicator_dir / "placement_regions.usda").write_text(placement_regions)

        # Generate variation asset manifest
        variation_dir = self.replicator_dir / "variation_assets"
        variation_dir.mkdir(parents=True, exist_ok=True)
        variation_manifest = self._generate_variation_manifest(manifest, inventory)
        (variation_dir / "manifest.json").write_text(
            json.dumps(variation_manifest, indent=2)
        )
        self._write_marker(self.replicator_dir / ".replicator_complete", status="completed")

        self.log("Generated replicator bundle")

        return StepResult(
            step=PipelineStep.REPLICATOR,
            success=True,
            duration_seconds=0,
            message="Replicator bundle generated",
            outputs={
                "placement_regions": str(self.replicator_dir / "placement_regions.usda"),
                "bundle_metadata": str(self.replicator_dir / "bundle_metadata.json"),
            },
        )

    def _run_variation_gen(self) -> StepResult:
        """Run variation asset generation for local testing."""
        manifest_path = self.replicator_dir / "variation_assets" / "manifest.json"
        if not manifest_path.is_file():
            return StepResult(
                step=PipelineStep.VARIATION_GEN,
                success=False,
                duration_seconds=0,
                message="Variation manifest not found - run replicator step first",
            )

        variation_assets_dir = self.scene_dir / "variation_assets"
        variation_assets_dir.mkdir(parents=True, exist_ok=True)

        variation_assets_prefix = f"{self.scene_id}/variation_assets"
        os.environ.setdefault("VARIATION_ASSETS_PREFIX", variation_assets_prefix)

        manifest = json.loads(manifest_path.read_text())
        assets = manifest.get("assets", [])
        if not assets:
            return StepResult(
                step=PipelineStep.VARIATION_GEN,
                success=False,
                duration_seconds=0,
                message="Variation manifest contains no assets",
            )

        self._generate_mock_variation_assets(assets, variation_assets_dir)
        marker_path = variation_assets_dir / ".variation_pipeline_complete"
        self._write_marker(marker_path, status="completed")

        return StepResult(
            step=PipelineStep.VARIATION_GEN,
            success=True,
            duration_seconds=0,
            message="Variation assets generated (mock)",
            outputs={
                "variation_assets": str(variation_assets_dir / "variation_assets.json"),
                "variation_marker": str(marker_path),
            },
        )

    def _generate_mock_variation_assets(
        self,
        assets: List[Dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """Generate mock variation assets for local runs."""
        objects = []
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
            b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01"
            b"\xe2!\xbc\x33\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        for asset in assets:
            name = asset.get("name") or asset.get("id") or "variation_asset"
            category = asset.get("category", "object")
            description = asset.get("description") or f"Variation of {category}"
            asset_dir = output_dir / name
            asset_dir.mkdir(parents=True, exist_ok=True)
            reference_path = asset_dir / "reference.png"
            if not reference_path.exists():
                reference_path.write_bytes(png_bytes)

            objects.append({
                "id": name,
                "name": name,
                "category": category,
                "short_description": description,
                "sim_role": "manipulable_object",
                "must_be_separate_asset": True,
                "preferred_view": f"variation_assets/{name}/reference.png",
                "multiview_dir": None,
                "crop_path": None,
                "physics_hints": asset.get("physics_hints", {}),
                "semantic_class": asset.get("semantic_class", category),
                "asset": {
                    "license": "CC0",
                    "commercial_ok": True,
                },
            })

        payload = {
            "scene_id": self.scene_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source": "variation-gen-job-mock",
            "objects": objects,
            "metadata": {
                "total_objects": len(objects),
                "generation_type": "variation_assets",
            },
        }
        (output_dir / "variation_assets.json").write_text(json.dumps(payload, indent=2))

    def _generate_placement_regions(self, manifest: Dict) -> str:
        """Generate minimal placement_regions.usda."""
        lines = [
            '#usda 1.0',
            '(',
            '    defaultPrim = "PlacementRegions"',
            ')',
            '',
            'def Xform "PlacementRegions" {',
        ]

        # Find surfaces for placement
        for obj in manifest.get("objects", []):
            category = (obj.get("category") or "").lower()
            if category in ["counter", "table", "shelf", "desk"]:
                obj_id = obj.get("id", "unknown")
                pos = obj.get("transform", {}).get("position", {})
                dims = obj.get("dimensions_est", {})

                # Create a placement region on top of this surface
                height = dims.get("height", 0.9)
                width = dims.get("width", 0.6)
                depth = dims.get("depth", 0.6)

                lines.append(f'    def Xform "region_{obj_id}" {{')
                lines.append(f'        double3 xformOp:translate = ({pos.get("x", 0)}, {pos.get("y", 0) + height}, {pos.get("z", 0)})')
                lines.append(f'        uniform token[] xformOpOrder = ["xformOp:translate"]')
                lines.append(f'        custom float3 extent = ({width * 0.8}, 0.02, {depth * 0.8})')
                lines.append(f'        custom string region_type = "placement_surface"')
                lines.append(f'    }}')

        lines.append('}')
        return '\n'.join(lines)

    def _generate_variation_manifest(self, manifest: Dict, inventory: Dict) -> Dict:
        """Generate variation asset manifest."""
        assets = []

        # Identify what variation assets might be needed based on scene
        for obj in inventory.get("objects", []):
            if obj.get("sim_role") in ["manipulable_object", "clutter"]:
                category = obj.get("category", "object")
                assets.append({
                    "name": f"variation_{category}",
                    "category": category,
                    "description": f"Variation of {category}",
                    "priority": "recommended",
                    "source_hint": "generate",
                })

        if not assets:
            for obj in manifest.get("objects", []):
                if obj.get("sim_role") in {"background", "scene_shell"}:
                    continue
                category = (obj.get("category")
                            or (obj.get("semantics") or {}).get("category")
                            or "object")
                assets.append({
                    "name": f"variation_{category}",
                    "category": category,
                    "description": f"Variation of {category}",
                    "priority": "recommended",
                    "source_hint": "generate",
                })
                if len(assets) >= 3:
                    break

        return {
            "scene_id": self.scene_id,
            "assets": assets,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    def _run_isaac_lab(self) -> StepResult:
        """Run Isaac Lab task generation."""
        try:
            from tools.isaac_lab_tasks.task_generator import IsaacLabTaskGenerator, TaskConfig

            # Load policy config
            policy_config_path = REPO_ROOT / "policy_configs" / "environment_policies.json"
            if policy_config_path.is_file():
                policy_config = json.loads(policy_config_path.read_text())
            else:
                policy_config = {"policies": {}, "environments": {}}

            # Load manifest as recipe
            manifest_path = self.assets_dir / "scene_manifest.json"
            if not manifest_path.is_file():
                return StepResult(
                    step=PipelineStep.ISAAC_LAB,
                    success=False,
                    duration_seconds=0,
                    message="Manifest not found",
                )

            manifest = json.loads(manifest_path.read_text())

            # Build recipe from manifest
            recipe = {
                "metadata": {
                    "environment_type": self.environment_type,
                    "scene_path": str(self.usd_dir / "scene.usda"),
                },
                "room": manifest.get("scene", {}).get("room", {}),
                "objects": manifest.get("objects", []),
            }

            # Generate task
            generator = IsaacLabTaskGenerator(policy_config)

            # Select policy based on environment
            policy_id = "manipulation"  # Default
            if self.environment_type == "kitchen":
                policy_id = "dish_loading"
            elif self.environment_type == "warehouse":
                policy_id = "pick_place"

            task = generator.generate(
                recipe=recipe,
                policy_id=policy_id,
                robot_type="franka",
                num_envs=1024,
            )

            # Save task files
            saved_files = generator.save(task, str(self.isaac_lab_dir))

            self.log(f"Generated Isaac Lab task: {task.task_name}")
            self.log(f"Files: {list(saved_files.keys())}")

            return StepResult(
                step=PipelineStep.ISAAC_LAB,
                success=True,
                duration_seconds=0,
                message=f"Generated task: {task.task_name}",
                outputs={"files": list(saved_files.keys())},
            )

        except ImportError as e:
            return StepResult(
                step=PipelineStep.ISAAC_LAB,
                success=False,
                duration_seconds=0,
                message=f"Import error: {e}",
            )
        except Exception as e:
            return StepResult(
                step=PipelineStep.ISAAC_LAB,
                success=False,
                duration_seconds=0,
                message=f"Error: {e}\n{traceback.format_exc()}",
            )

    def _run_geniesim_export(self) -> StepResult:
        """Run Genie Sim export job locally."""
        try:
            sys.path.insert(0, str(REPO_ROOT / "genie-sim-export-job"))
            from export_to_geniesim import run_geniesim_export_job
        except ImportError as e:
            return StepResult(
                step=PipelineStep.GENIESIM_EXPORT,
                success=False,
                duration_seconds=0,
                message=f"Import error (Genie Sim export job not found): {e}",
            )

        root = self.scene_dir.parent
        assets_prefix = f"{self.scene_id}/assets"
        geniesim_prefix = f"{self.scene_id}/geniesim"
        variation_assets_prefix = f"{self.scene_id}/variation_assets"
        replicator_prefix = f"{self.scene_id}/replicator"
        robot_type = os.getenv("GENIESIM_ROBOT_TYPE", "franka")
        os.environ.setdefault("VARIATION_ASSETS_PREFIX", variation_assets_prefix)

        exit_code = run_geniesim_export_job(
            root=root,
            scene_id=self.scene_id,
            assets_prefix=assets_prefix,
            geniesim_prefix=geniesim_prefix,
            robot_type=robot_type,
            variation_assets_prefix=variation_assets_prefix,
            replicator_prefix=replicator_prefix,
            copy_usd=True,
        )
        if exit_code != 0:
            return StepResult(
                step=PipelineStep.GENIESIM_EXPORT,
                success=False,
                duration_seconds=0,
                message=f"Genie Sim export failed with exit code {exit_code}",
            )

        return StepResult(
            step=PipelineStep.GENIESIM_EXPORT,
            success=True,
            duration_seconds=0,
            message="Genie Sim export completed",
            outputs={
                "geniesim_dir": str(self.geniesim_dir),
                "scene_graph": str(self.geniesim_dir / "scene_graph.json"),
                "task_config": str(self.geniesim_dir / "task_config.json"),
            },
        )

    @staticmethod
    def _resolve_geniesim_submission_mode(
        api_key: Optional[str],
        force_local: bool,
        mock_mode: bool,
    ) -> str:
        """Resolve Genie Sim submission mode with local default."""
        if mock_mode:
            return "mock"
        if force_local:
            return "local"
        return "local"

    def _run_geniesim_submit(self) -> StepResult:
        """Submit Genie Sim generation (API or local framework)."""
        try:
            from tools.geniesim_adapter.local_framework import (
                check_geniesim_availability,
                run_local_data_collection,
            )
        except ImportError as e:
            return StepResult(
                step=PipelineStep.GENIESIM_SUBMIT,
                success=False,
                duration_seconds=0,
                message=f"Import error (Genie Sim submit dependencies not found): {e}",
            )

        scene_graph_path = self.geniesim_dir / "scene_graph.json"
        asset_index_path = self.geniesim_dir / "asset_index.json"
        task_config_path = self.geniesim_dir / "task_config.json"
        if not scene_graph_path.is_file() or not asset_index_path.is_file() or not task_config_path.is_file():
            return StepResult(
                step=PipelineStep.GENIESIM_SUBMIT,
                success=False,
                duration_seconds=0,
                message="Genie Sim export outputs missing - run genie-sim-export first",
            )

        scene_graph = json.loads(scene_graph_path.read_text())
        asset_index = json.loads(asset_index_path.read_text())
        task_config = json.loads(task_config_path.read_text())

        robot_type = os.getenv("GENIESIM_ROBOT_TYPE", "franka")
        episodes_per_task = int(os.getenv("EPISODES_PER_TASK", "10"))
        num_variations = int(os.getenv("NUM_VARIATIONS", "5"))
        min_quality_score = float(os.getenv("MIN_QUALITY_SCORE", "0.85"))

        force_local = os.getenv("GENIESIM_FORCE_LOCAL", "false").lower() == "true"
        mock_mode = os.getenv("GENIESIM_MOCK_MODE", "false").lower() == "true"
        production_mode = self.environment == "production"
        submission_mode = self._resolve_geniesim_submission_mode(
            api_key=None,
            force_local=force_local,
            mock_mode=mock_mode,
        )
        if production_mode and submission_mode == "mock":
            return StepResult(
                step=PipelineStep.GENIESIM_SUBMIT,
                success=False,
                duration_seconds=0,
                message="Mock Genie Sim submission is not allowed in production",
            )
        job_id = None
        submission_message = None
        local_run_result = None
        preflight_status = None
        missing_components: List[str] = []
        remediation_guidance = None

        if submission_mode == "mock":
            try:
                sys.path.insert(0, str(REPO_ROOT / "genie-sim-export-job"))
                from geniesim_client import GenieSimClient, GenerationParams
            except ImportError as e:
                return StepResult(
                    step=PipelineStep.GENIESIM_SUBMIT,
                    success=False,
                    duration_seconds=0,
                    message=f"Import error (Genie Sim mock client not found): {e}",
                )

            generation_params = GenerationParams(
                episodes_per_task=episodes_per_task,
                num_variations=num_variations,
                robot_type=robot_type,
                min_quality_score=min_quality_score,
            )
            client = GenieSimClient(mock_mode=True, validate_on_init=False)
            try:
                result = client.submit_generation_job(
                    scene_graph=scene_graph,
                    asset_index=asset_index,
                    task_config=task_config,
                    generation_params=generation_params,
                    job_name=f"{self.scene_id}-geniesim-mock",
                )
                if not result.success or not result.job_id:
                    raise RuntimeError(result.message or "Genie Sim mock submission failed")
                job_id = result.job_id
                submission_message = result.message
            finally:
                client.close()
        else:
            import uuid

            job_id = f"local-{uuid.uuid4()}"
            submission_message = "Local Genie Sim execution started."
            preflight_status = check_geniesim_availability()
            if not preflight_status.get("isaac_sim_available", False):
                missing_components.append("Isaac Sim path")
            if not preflight_status.get("grpc_available", False):
                missing_components.append("gRPC stubs")
            if not preflight_status.get("server_running", False):
                missing_components.append("server running")
            remediation_guidance = (
                "Set ISAAC_SIM_PATH to your Isaac Sim install, ensure gRPC stubs are installed, "
                "and start or expose the Genie Sim server at the configured host/port."
            )
            if production_mode and not preflight_status.get("available", False):
                return StepResult(
                    step=PipelineStep.GENIESIM_SUBMIT,
                    success=False,
                    duration_seconds=0,
                    message=(
                        "Production Genie Sim submission requires real dependencies. "
                        f"Missing: {', '.join(missing_components) or 'unknown'}."
                    ),
                    outputs={"missing_components": missing_components},
                )

            output_dir = self.episodes_dir / f"geniesim_{job_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            config_dir = output_dir / "config"
            config_dir.mkdir(parents=True, exist_ok=True)

            merged_manifest_path = self.geniesim_dir / "merged_scene_manifest.json"
            if merged_manifest_path.is_file():
                scene_manifest = json.loads(merged_manifest_path.read_text())
            else:
                scene_manifest = {"scene_graph": scene_graph}

            scene_manifest_path = config_dir / "scene_manifest.json"
            task_config_local_path = config_dir / "task_config.json"
            scene_manifest_path.write_text(json.dumps(scene_manifest, indent=2))
            task_config_local_path.write_text(json.dumps(task_config, indent=2))

            if not preflight_status.get("available", False):
                submission_message = (
                    "Local Genie Sim preflight failed; missing: "
                    f"{', '.join(missing_components) or 'unknown components'}."
                )
            else:
                local_run_result = run_local_data_collection(
                    scene_manifest_path=scene_manifest_path,
                    task_config_path=task_config_local_path,
                    output_dir=output_dir,
                    robot_type=robot_type,
                    episodes_per_task=episodes_per_task,
                    verbose=True,
                )
                submission_message = (
                    "Local Genie Sim execution completed."
                    if local_run_result and local_run_result.success
                    else "Local Genie Sim execution failed."
                )

        if submission_mode == "mock":
            job_status = "completed"
        else:
            job_status = (
                "completed"
                if local_run_result and local_run_result.success
                else ("failed" if submission_mode == "local" else "submitted")
            )
            if submission_mode == "local" and preflight_status and not preflight_status.get("available", False):
                job_status = "failed"

        job_payload = {
            "job_id": job_id,
            "scene_id": self.scene_id,
            "status": job_status,
            "submission_mode": submission_mode,
            "submitted_at": datetime.utcnow().isoformat() + "Z",
            "message": submission_message,
            "bundle": {
                "scene_graph": str(scene_graph_path),
                "asset_index": str(asset_index_path),
                "task_config": str(task_config_path),
            },
            "generation_params": {
                "robot_type": robot_type,
                "episodes_per_task": episodes_per_task,
                "num_variations": num_variations,
                "min_quality_score": min_quality_score,
            },
        }

        if submission_mode == "local":
            episodes_path = str(self.episodes_dir / f"geniesim_{job_id}")
            job_payload["artifacts"] = {
                "episodes_path": episodes_path,
                "episodes_prefix": episodes_path,
                "lerobot_path": str(Path(episodes_path) / "lerobot"),
                "lerobot_prefix": str(Path(episodes_path) / "lerobot"),
            }
            job_payload["local_execution"] = {
                "success": bool(local_run_result and local_run_result.success),
                "episodes_collected": getattr(local_run_result, "episodes_collected", 0) if local_run_result else 0,
                "episodes_passed": getattr(local_run_result, "episodes_passed", 0) if local_run_result else 0,
                "preflight": {
                    "available": preflight_status.get("available", False) if preflight_status else False,
                    "isaac_sim_available": (
                        preflight_status.get("isaac_sim_available", False) if preflight_status else False
                    ),
                    "grpc_available": (
                        preflight_status.get("grpc_available", False) if preflight_status else False
                    ),
                    "server_running": (
                        preflight_status.get("server_running", False) if preflight_status else False
                    ),
                    "missing_components": missing_components,
                    "details": preflight_status.get("details", {}) if preflight_status else {},
                },
                "remediation": (
                    remediation_guidance
                    if preflight_status and not preflight_status.get("available", False)
                    else None
                ),
            }

        job_path = self.geniesim_dir / "job.json"
        job_path.write_text(json.dumps(job_payload, indent=2))

        return StepResult(
            step=PipelineStep.GENIESIM_SUBMIT,
            success=job_status != "failed",
            duration_seconds=0,
            message=submission_message or "Genie Sim submission completed",
            outputs={
                "job_id": job_id,
                "job_payload": str(job_path),
            },
        )

    def _run_dwm(self) -> StepResult:
        """
        Run DWM (Dexterous World Model) preparation.

        Generates conditioning data for DWM video diffusion model:
        - Egocentric camera trajectories
        - Static scene video renders
        - Hand mesh video renders
        - Text prompts

        Reference: DWM paper (arXiv:2512.17907)
        """
        try:
            # Import DWM preparation job
            sys.path.insert(0, str(REPO_ROOT / "dwm-preparation-job"))
            from prepare_dwm_bundle import run_dwm_preparation

            # Check prerequisites
            manifest_path = self.assets_dir / "scene_manifest.json"
            if not manifest_path.is_file():
                return StepResult(
                    step=PipelineStep.DWM,
                    success=False,
                    duration_seconds=0,
                    message="Manifest not found - run regen3d step first",
                )

            # Run DWM preparation
            output = run_dwm_preparation(
                scene_dir=self.scene_dir,
                output_dir=self.dwm_dir,
                num_trajectories=5,  # Default number of trajectories
                verbose=self.verbose,
            )

            if output.success:
                marker_path = self.dwm_dir / ".dwm_complete"
                self._write_marker(marker_path, status="completed")

                return StepResult(
                    step=PipelineStep.DWM,
                    success=True,
                    duration_seconds=output.generation_time_seconds,
                    message=f"Generated {output.num_bundles} DWM bundles",
                    outputs={
                        "bundles_count": output.num_bundles,
                        "total_frames": output.total_frames,
                        "output_dir": str(output.output_dir),
                        "manifest": str(output.manifest_path) if output.manifest_path else None,
                        "completion_marker": str(marker_path),
                    },
                )
            else:
                return StepResult(
                    step=PipelineStep.DWM,
                    success=False,
                    duration_seconds=output.generation_time_seconds,
                    message=f"DWM generation failed: {output.errors[:1]}",
                    outputs={"errors": output.errors},
                )

        except ImportError as e:
            return StepResult(
                step=PipelineStep.DWM,
                success=False,
                duration_seconds=0,
                message=f"Import error (DWM job not found): {e}",
            )
        except Exception as e:
            return StepResult(
                step=PipelineStep.DWM,
                success=False,
                duration_seconds=0,
                message=f"Error: {e}\n{traceback.format_exc()}",
            )

    def _run_dwm_inference(self) -> StepResult:
        """Run the DWM inference job on prepared bundles."""
        try:
            sys.path.insert(0, str(REPO_ROOT / "dwm-preparation-job"))
            from dwm_inference_job import run_dwm_inference

            manifest_path = self.dwm_dir / "dwm_bundles_manifest.json"
            if not manifest_path.is_file():
                return StepResult(
                    step=PipelineStep.DWM_INFERENCE,
                    success=False,
                    duration_seconds=0,
                    message="DWM bundles manifest not found - run dwm step first",
                )

            output = run_dwm_inference(
                bundles_dir=self.dwm_dir,
                api_endpoint=os.environ.get("DWM_API_ENDPOINT"),
                checkpoint_path=os.environ.get("DWM_CHECKPOINT_PATH"),
                verbose=self.verbose,
            )

            if output.success:
                marker_path = self.dwm_dir / ".dwm_inference_complete"
                self._write_marker(marker_path, status="completed")
                return StepResult(
                    step=PipelineStep.DWM_INFERENCE,
                    success=True,
                    duration_seconds=0,
                    message=f"Generated interaction videos for {len(output.bundles_processed)} bundles",
                    outputs={
                        "bundles": len(output.bundles_processed),
                        "manifest": str(output.manifest_path) if output.manifest_path else None,
                        "completion_marker": str(marker_path),
                    },
                )

            return StepResult(
                step=PipelineStep.DWM_INFERENCE,
                success=False,
                duration_seconds=0,
                message=f"DWM inference failed: {output.errors[:1]}",
                outputs={"errors": output.errors},
            )

        except ImportError as e:
            return StepResult(
                step=PipelineStep.DWM_INFERENCE,
                success=False,
                duration_seconds=0,
                message=f"Import error (DWM inference job not found): {e}",
            )
        except Exception as e:
            return StepResult(
                step=PipelineStep.DWM_INFERENCE,
                success=False,
                duration_seconds=0,
                message=f"Error: {e}\n{traceback.format_exc()}",
            )

    def _run_dream2flow(self) -> StepResult:
        """
        Run Dream2Flow preparation.

        Generates Dream2Flow bundles from scene for video-to-flow robot control:
        - Task instructions from scene manifest
        - Initial RGB-D observations
        - Generated task videos (via video diffusion)
        - 3D object flow extraction
        - Robot tracking targets

        Reference: Dream2Flow paper (arXiv:2512.24766)
        """
        try:
            sys.path.insert(0, str(REPO_ROOT / "dream2flow-preparation-job"))
            from prepare_dream2flow_bundle import run_dream2flow_preparation

            # Check prerequisites
            manifest_path = self.assets_dir / "scene_manifest.json"
            if not manifest_path.is_file():
                return StepResult(
                    step=PipelineStep.DREAM2FLOW,
                    success=False,
                    duration_seconds=0,
                    message="Manifest not found - run regen3d step first",
                )

            # Run Dream2Flow preparation
            output = run_dream2flow_preparation(
                scene_dir=self.scene_dir,
                output_dir=self.dream2flow_dir,
                num_tasks=5,  # Default number of tasks
                verbose=self.verbose,
            )

            if output.success:
                marker_path = self.dream2flow_dir / ".dream2flow_complete"
                self._write_marker(marker_path, status="completed")

                return StepResult(
                    step=PipelineStep.DREAM2FLOW,
                    success=True,
                    duration_seconds=output.generation_time_seconds,
                    message=f"Generated {len(output.bundles)} Dream2Flow bundles",
                    outputs={
                        "bundles_count": len(output.bundles),
                        "output_dir": str(output.output_dir),
                        "manifest": str(output.manifest_path) if output.manifest_path else None,
                        "completion_marker": str(marker_path),
                    },
                )
            else:
                return StepResult(
                    step=PipelineStep.DREAM2FLOW,
                    success=False,
                    duration_seconds=output.generation_time_seconds,
                    message=f"Dream2Flow generation failed: {output.errors[:1] if output.errors else 'Unknown error'}",
                    outputs={"errors": output.errors},
                )

        except ImportError as e:
            return StepResult(
                step=PipelineStep.DREAM2FLOW,
                success=False,
                duration_seconds=0,
                message=f"Import error (Dream2Flow job not found): {e}",
            )
        except Exception as e:
            return StepResult(
                step=PipelineStep.DREAM2FLOW,
                success=False,
                duration_seconds=0,
                message=f"Error: {e}\n{traceback.format_exc()}",
            )

    def _run_dream2flow_inference(self) -> StepResult:
        """Run the Dream2Flow inference job on prepared bundles."""
        try:
            sys.path.insert(0, str(REPO_ROOT / "dream2flow-preparation-job"))
            from dream2flow_inference_job import run_dream2flow_inference

            manifest_path = self.dream2flow_dir / "dream2flow_bundles_manifest.json"
            if not manifest_path.is_file():
                return StepResult(
                    step=PipelineStep.DREAM2FLOW_INFERENCE,
                    success=False,
                    duration_seconds=0,
                    message="Dream2Flow bundles manifest not found - run dream2flow step first",
                )

            output = run_dream2flow_inference(
                bundles_dir=self.dream2flow_dir,
                api_endpoint=os.environ.get("DREAM2FLOW_API_ENDPOINT"),
                checkpoint_path=os.environ.get("DREAM2FLOW_CHECKPOINT_PATH"),
                verbose=self.verbose,
            )

            if output.success:
                marker_path = self.dream2flow_dir / ".dream2flow_inference_complete"
                self._write_marker(marker_path, status="completed")
                return StepResult(
                    step=PipelineStep.DREAM2FLOW_INFERENCE,
                    success=True,
                    duration_seconds=0,
                    message=f"Generated flow extractions for {len(output.bundles_processed)} bundles",
                    outputs={
                        "bundles": len(output.bundles_processed),
                        "manifest": str(output.manifest_path) if output.manifest_path else None,
                        "completion_marker": str(marker_path),
                    },
                )

            return StepResult(
                step=PipelineStep.DREAM2FLOW_INFERENCE,
                success=False,
                duration_seconds=0,
                message=f"Dream2Flow inference failed: {output.errors[:1] if output.errors else 'Unknown error'}",
                outputs={"errors": output.errors},
            )

        except ImportError as e:
            return StepResult(
                step=PipelineStep.DREAM2FLOW_INFERENCE,
                success=False,
                duration_seconds=0,
                message=f"Import error (Dream2Flow inference job not found): {e}",
            )
        except Exception as e:
            return StepResult(
                step=PipelineStep.DREAM2FLOW_INFERENCE,
                success=False,
                duration_seconds=0,
                message=f"Error: {e}\n{traceback.format_exc()}",
            )

    def _run_validation(self) -> StepResult:
        """Run QA validation."""
        try:
            from tools.qa_validation.validator import run_qa_validation

            report_path = self.scene_dir / "validation_report.json"
            report = run_qa_validation(
                scene_dir=self.scene_dir,
                scene_id=self.scene_id,
                output_report=report_path,
                verbose=self.verbose,
            )

            if report.passed:
                return StepResult(
                    step=PipelineStep.VALIDATE,
                    success=True,
                    duration_seconds=0,
                    message=f"Validation passed: {report.summary.get('passed', 0)}/{report.summary.get('total_checks', 0)} checks",
                    outputs={"report": str(report_path), "summary": report.summary},
                )
            else:
                return StepResult(
                    step=PipelineStep.VALIDATE,
                    success=False,
                    duration_seconds=0,
                    message=f"Validation failed: {len(report.issues)} errors",
                    outputs={"report": str(report_path), "issues": report.issues[:5]},
                )

        except ImportError as e:
            return StepResult(
                step=PipelineStep.VALIDATE,
                success=False,
                duration_seconds=0,
                message=f"Import error: {e}",
            )

    def _print_summary(self) -> None:
        """Print pipeline execution summary."""
        self.log("\n" + "=" * 60)
        self.log("PIPELINE SUMMARY")
        self.log("=" * 60)

        total_time = sum(r.duration_seconds for r in self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed

        for result in self.results:
            status = "PASS" if result.success else "FAIL"
            self.log(f"  {result.step.value:15} [{status}] {result.duration_seconds:.2f}s - {result.message[:50]}")

        self.log("-" * 60)
        self.log(f"Total: {passed} passed, {failed} failed in {total_time:.2f}s")

        # Print key outputs
        self.log("\nKey Outputs:")
        for result in self.results:
            for key, value in result.outputs.items():
                if key in ["manifest", "layout", "scene_usda", "report"]:
                    self.log(f"  {key}: {value}")

        self.log("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run BlueprintPipeline locally for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Path to scene directory (contains regen3d/, etc.)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        help="Comma-separated list of steps to run (default: all)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run QA validation at the end",
    )
    parser.add_argument(
        "--environment",
        default="kitchen",
        choices=["kitchen", "office", "warehouse", "laundry"],
        help="Environment type for policy selection",
    )
    parser.add_argument(
        "--with-interactive",
        action="store_true",
        help="Run interactive job (requires PARTICULATE_ENDPOINT)",
    )
    parser.add_argument(
        "--enable-dwm",
        action="store_true",
        help="Include optional DWM preparation/inference steps in the default pipeline",
    )
    parser.add_argument(
        "--use-geniesim",
        action="store_true",
        help="Use Genie Sim execution mode (overrides USE_GENIESIM for this run)",
    )
    parser.add_argument(
        "--mock-geniesim",
        action="store_true",
        help="Run Genie Sim steps in mock mode (no external services required)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    if args.use_geniesim:
        os.environ["USE_GENIESIM"] = "true"
    if args.mock_geniesim:
        os.environ["GENIESIM_MOCK_MODE"] = "true"
        os.environ["USE_GENIESIM"] = "true"

    # Parse steps
    steps = None
    if args.steps:
        step_names = [s.strip().lower() for s in args.steps.split(",")]
        steps = []
        for name in step_names:
            try:
                steps.append(PipelineStep(name))
            except ValueError:
                print(f"Unknown step: {name}")
                print(f"Available: {[s.value for s in PipelineStep]}")
                sys.exit(1)

    # Create and run pipeline
    runner = LocalPipelineRunner(
        scene_dir=args.scene_dir,
        verbose=not args.quiet,
        skip_interactive=not args.with_interactive,
        environment_type=args.environment,
        enable_dwm=args.enable_dwm,
    )

    success = runner.run(steps=steps, run_validation=args.validate)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
