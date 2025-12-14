#!/usr/bin/env python3
"""
Local Pipeline Runner for BlueprintPipeline.

Runs the full pipeline locally without requiring GCS or Cloud Run.
Useful for development, testing, and debugging.

Usage:
    # Generate mock scene and run pipeline
    python fixtures/generate_mock_zeroscene.py --scene-id test_kitchen --output-dir ./test_scenes
    python tools/run_local_pipeline.py --scene-dir ./test_scenes/scenes/test_kitchen

    # Run with specific steps
    python tools/run_local_pipeline.py --scene-dir ./scene --steps zeroscene,simready,usd

    # Run with validation
    python tools/run_local_pipeline.py --scene-dir ./scene --validate

Pipeline Steps:
    1. zeroscene  - Adapt ZeroScene outputs to BlueprintPipeline format
    2. scale      - (Optional) Scale calibration
    3. interactive - PhysX-Anything articulation (requires PHYSX_ENDPOINT)
    4. simready   - Prepare physics-ready assets
    5. usd        - Assemble scene.usda
    6. replicator - Generate Replicator bundle
    7. isaac-lab  - Generate Isaac Lab task package
    8. validate   - QA validation
"""

import argparse
import json
import os
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
    ZEROSCENE = "zeroscene"
    SCALE = "scale"
    INTERACTIVE = "interactive"
    SIMREADY = "simready"
    USD = "usd"
    REPLICATOR = "replicator"
    ISAAC_LAB = "isaac-lab"
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
        PipelineStep.ZEROSCENE,
        PipelineStep.SIMREADY,
        PipelineStep.USD,
        PipelineStep.REPLICATOR,
        PipelineStep.ISAAC_LAB,
    ]

    def __init__(
        self,
        scene_dir: Path,
        verbose: bool = True,
        skip_interactive: bool = True,
        environment_type: str = "kitchen",
    ):
        """Initialize the local pipeline runner.

        Args:
            scene_dir: Path to scene directory (contains zeroscene/, etc.)
            verbose: Print detailed progress
            skip_interactive: Skip interactive-job (requires external service)
            environment_type: Environment type for policy selection
        """
        self.scene_dir = Path(scene_dir).resolve()
        self.verbose = verbose
        self.skip_interactive = skip_interactive
        self.environment_type = environment_type

        # Derive scene ID from directory name
        self.scene_id = self.scene_dir.name

        # Setup paths
        self.zeroscene_dir = self.scene_dir / "zeroscene"
        self.assets_dir = self.scene_dir / "assets"
        self.layout_dir = self.scene_dir / "layout"
        self.seg_dir = self.scene_dir / "seg"
        self.usd_dir = self.scene_dir / "usd"
        self.replicator_dir = self.scene_dir / "replicator"
        self.isaac_lab_dir = self.scene_dir / "isaac_lab"

        self.results: List[StepResult] = []

    def log(self, msg: str, level: str = "INFO") -> None:
        """Log a message."""
        if self.verbose:
            print(f"[LOCAL-PIPELINE] [{level}] {msg}")

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
            steps = self.DEFAULT_STEPS.copy()

        if run_validation and PipelineStep.VALIDATE not in steps:
            steps.append(PipelineStep.VALIDATE)

        self.log("=" * 60)
        self.log("BlueprintPipeline Local Runner")
        self.log("=" * 60)
        self.log(f"Scene directory: {self.scene_dir}")
        self.log(f"Scene ID: {self.scene_id}")
        self.log(f"Steps: {[s.value for s in steps]}")
        self.log("=" * 60)

        # Check prerequisites
        if not self.zeroscene_dir.is_dir():
            self.log(f"ERROR: ZeroScene output not found at {self.zeroscene_dir}", "ERROR")
            self.log("Run: python fixtures/generate_mock_zeroscene.py first", "ERROR")
            return False

        # Create output directories
        for d in [self.assets_dir, self.layout_dir, self.seg_dir,
                  self.usd_dir, self.replicator_dir, self.isaac_lab_dir]:
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

    def _run_step(self, step: PipelineStep) -> StepResult:
        """Run a single pipeline step."""
        import time
        start_time = time.time()

        self.log(f"\n--- Running step: {step.value} ---")

        try:
            if step == PipelineStep.ZEROSCENE:
                result = self._run_zeroscene_adapter()
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
            elif step == PipelineStep.ISAAC_LAB:
                result = self._run_isaac_lab()
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

    def _run_zeroscene_adapter(self) -> StepResult:
        """Run the zeroscene adapter."""
        from tools.zeroscene_adapter import ZeroSceneAdapter

        adapter = ZeroSceneAdapter(verbose=self.verbose)

        # Load ZeroScene outputs
        zeroscene_output = adapter.load_zeroscene_output(self.zeroscene_dir)
        self.log(f"Loaded {len(zeroscene_output.objects)} objects from ZeroScene")

        # Generate manifest
        manifest = adapter.create_manifest(
            zeroscene_output,
            scene_id=self.scene_id,
            environment_type=self.environment_type,
        )
        manifest_path = self.assets_dir / "scene_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        self.log(f"Wrote manifest: {manifest_path}")

        # Generate layout
        layout = adapter.create_layout(zeroscene_output, apply_scale_factor=1.0)
        layout_path = self.layout_dir / "scene_layout_scaled.json"
        layout_path.write_text(json.dumps(layout, indent=2))
        self.log(f"Wrote layout: {layout_path}")

        # Copy assets
        asset_paths = adapter.copy_assets(
            zeroscene_output,
            self.scene_dir,
            assets_prefix="assets",
        )
        self.log(f"Copied {len(asset_paths)} assets")

        # Generate inventory
        inventory = self._generate_inventory(zeroscene_output)
        inventory_path = self.seg_dir / "inventory.json"
        inventory_path.write_text(json.dumps(inventory, indent=2))
        self.log(f"Wrote inventory: {inventory_path}")

        # Write completion marker
        marker_path = self.assets_dir / ".zeroscene_complete"
        marker_content = {
            "status": "complete",
            "scene_id": self.scene_id,
            "objects_count": len(zeroscene_output.objects),
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }
        marker_path.write_text(json.dumps(marker_content, indent=2))

        return StepResult(
            step=PipelineStep.ZEROSCENE,
            success=True,
            duration_seconds=0,
            message="ZeroScene adapter completed",
            outputs={
                "manifest": str(manifest_path),
                "layout": str(layout_path),
                "objects_count": len(zeroscene_output.objects),
            },
        )

    def _generate_inventory(self, zeroscene_output) -> Dict[str, Any]:
        """Generate semantic inventory from ZeroScene output."""
        objects = []
        for obj in zeroscene_output.objects:
            inv_obj = {
                "id": obj.id,
                "category": obj.category or "object",
                "short_description": obj.description or f"Object {obj.id}",
                "sim_role": obj.sim_role if obj.sim_role != "unknown" else "static",
                "bounds": obj.bounds,
            }
            objects.append(inv_obj)

        return {
            "scene_id": self.scene_id,
            "environment_type": self.environment_type,
            "total_objects": len(objects),
            "objects": objects,
            "metadata": {
                "source": "zeroscene_adapter",
                "generated_at": datetime.utcnow().isoformat() + "Z",
            },
        }

    def _run_scale(self) -> StepResult:
        """Run scale calibration (optional)."""
        # For local testing, we trust the ZeroScene scale
        self.log("Scale calibration: using ZeroScene scale (trusted)")

        return StepResult(
            step=PipelineStep.SCALE,
            success=True,
            duration_seconds=0,
            message="Using ZeroScene scale",
        )

    def _run_interactive(self) -> StepResult:
        """Run interactive asset processing."""
        if self.skip_interactive:
            self.log("Interactive job skipped (requires PHYSX_ENDPOINT)")
            return StepResult(
                step=PipelineStep.INTERACTIVE,
                success=True,
                duration_seconds=0,
                message="Skipped (no PHYSX_ENDPOINT)",
            )

        # Would need PHYSX_ENDPOINT to run
        endpoint = os.getenv("PHYSX_ENDPOINT")
        if not endpoint:
            return StepResult(
                step=PipelineStep.INTERACTIVE,
                success=True,
                duration_seconds=0,
                message="Skipped (PHYSX_ENDPOINT not set)",
            )

        # TODO: Import and run interactive job
        return StepResult(
            step=PipelineStep.INTERACTIVE,
            success=True,
            duration_seconds=0,
            message="Interactive processing not implemented in local mode",
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
                message="Manifest not found - run zeroscene step first",
            )

        manifest = json.loads(manifest_path.read_text())

        # For local testing, add basic physics properties to each object
        for obj in manifest.get("objects", []):
            if not obj.get("physics"):
                obj["physics"] = {
                    "dynamic": obj.get("sim_role") in ["manipulable_object", "clutter"],
                    "mass": self._estimate_mass(obj),
                    "friction": 0.5,
                    "restitution": 0.1,
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
        help="Path to scene directory (contains zeroscene/, etc.)",
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
        help="Run interactive job (requires PHYSX_ENDPOINT)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

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
    )

    success = runner.run(steps=steps, run_validation=args.validate)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
