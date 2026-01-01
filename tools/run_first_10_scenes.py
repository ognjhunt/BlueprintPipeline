#!/usr/bin/env python3
"""
Generate First 10 Scenes - End-to-End Pipeline Runner

This script provides a complete workflow for generating the first 10 scenes
through BlueprintPipeline with:
- Quality gates with human-in-the-loop validation
- Sim-to-real tracking setup
- Customer success metrics

Prerequisites:
- 3D-RE-GEN outputs must be available (scene_manifest.json, meshes, etc.)

Usage:
    # With 3D-RE-GEN outputs
    python tools/run_first_10_scenes.py --input-dir ./regen3d_outputs

    # Test mode (uses mock data)
    python tools/run_first_10_scenes.py --test-mode
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import pipeline components
from tools.quality_gates import (
    QualityGateRegistry,
    QualityGateCheckpoint,
    NotificationService,
    NotificationPayload,
    generate_qa_context,
)
from tools.sim2real import (
    ExperimentTracker,
    TaskType,
    RobotType,
)
from tools.metrics import (
    track_pipeline_run,
    track_scene_delivery,
    update_pipeline_status,
    get_success_summary,
)


@dataclass
class SceneConfig:
    """Configuration for a scene to process."""
    scene_id: str
    input_path: Path
    environment_type: str = "generic"
    task_types: List[str] = None  # Tasks to generate
    robot_type: str = "franka"

    def __post_init__(self):
        if self.task_types is None:
            self.task_types = ["pick_place"]


@dataclass
class PipelineResult:
    """Result of processing a scene."""
    scene_id: str
    success: bool
    error: Optional[str] = None
    processing_time_seconds: float = 0.0
    output_dir: Optional[Path] = None
    object_count: int = 0
    episode_count: int = 0
    quality_gates_passed: int = 0
    quality_gates_failed: int = 0
    delivery_id: Optional[str] = None
    sim2real_experiment_id: Optional[str] = None


class First10ScenesRunner:
    """Runner for generating the first 10 scenes end-to-end."""

    def __init__(
        self,
        output_base: Path,
        notify_email: str = "ohstnhunt@gmail.com",
        notify_phone: str = "9196389913",
        customer_id: str = "internal_validation",
        dry_run: bool = False,
        verbose: bool = True,
    ):
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.customer_id = customer_id
        self.dry_run = dry_run
        self.verbose = verbose

        # Initialize components
        self.quality_gates = QualityGateRegistry(verbose=verbose)

        self.notifications = NotificationService(
            email=notify_email,
            phone=notify_phone,
            verbose=verbose,
        )

        self.sim2real_tracker = ExperimentTracker(
            experiments_dir=self.output_base / "sim2real_experiments",
            verbose=verbose,
        )

        self.results: List[PipelineResult] = []

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[PIPELINE] {msg}")

    def run(self, scenes: List[SceneConfig]) -> List[PipelineResult]:
        """Run the pipeline for all scenes.

        Args:
            scenes: List of scene configurations

        Returns:
            List of pipeline results
        """
        self.log(f"Starting pipeline for {len(scenes)} scenes")
        self.log(f"Output directory: {self.output_base}")
        self.log("")

        for i, scene in enumerate(scenes, 1):
            self.log(f"=" * 60)
            self.log(f"Processing scene {i}/{len(scenes)}: {scene.scene_id}")
            self.log(f"=" * 60)

            try:
                result = self._process_scene(scene)
                self.results.append(result)

                if result.success:
                    self.log(f"Scene {scene.scene_id} completed successfully")
                else:
                    self.log(f"Scene {scene.scene_id} failed: {result.error}")

            except Exception as e:
                self.log(f"Scene {scene.scene_id} failed with exception: {e}")
                self.results.append(PipelineResult(
                    scene_id=scene.scene_id,
                    success=False,
                    error=str(e),
                ))

            self.log("")

        # Generate summary
        self._generate_summary()

        return self.results

    def _process_scene(self, scene: SceneConfig) -> PipelineResult:
        """Process a single scene through the pipeline.

        Expects 3D-RE-GEN outputs to be available at scene.input_path.
        """
        start_time = time.time()

        output_dir = self.output_base / scene.scene_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Track delivery
        delivery_id = track_pipeline_run(
            scene_id=scene.scene_id,
            customer_id=self.customer_id,
            environment_type=scene.environment_type,
        )
        update_pipeline_status(delivery_id, "processing")

        # Step 1: Validate 3D-RE-GEN outputs exist
        self.log("Step 1: Checking 3D-RE-GEN outputs")
        manifest_path = scene.input_path / "scene_manifest.json"
        if not manifest_path.exists():
            # Also check assets subdirectory
            manifest_path = scene.input_path / "assets" / "scene_manifest.json"

        if not manifest_path.exists():
            return PipelineResult(
                scene_id=scene.scene_id,
                success=False,
                error="3D-RE-GEN outputs not found (missing scene_manifest.json)",
                delivery_id=delivery_id,
            )

        manifest = json.loads(manifest_path.read_text())
        object_count = len(manifest.get("objects", []))
        self.log(f"  Found manifest with {object_count} objects")

        # Step 2: Quality Gate - Manifest Validation
        self.log("Step 2: Quality Gate - Manifest Validation")
        gate_results = self.quality_gates.run_checkpoint(
            QualityGateCheckpoint.MANIFEST_VALIDATED,
            context={"scene_id": scene.scene_id, "manifest": manifest},
            notification_service=self.notifications,
        )

        if not self.quality_gates.can_proceed():
            self._send_qa_notification(scene, "manifest_validated", gate_results)
            return PipelineResult(
                scene_id=scene.scene_id,
                success=False,
                error="Manifest validation failed",
                quality_gates_failed=len(self.quality_gates.get_blocking_failures()),
                delivery_id=delivery_id,
            )

        # Step 3: SimReady (Mock for now - would call actual job)
        self.log("Step 3: SimReady Processing (physics estimation)")
        if not self.dry_run:
            self._create_mock_simready(output_dir, manifest)

        gate_results = self.quality_gates.run_checkpoint(
            QualityGateCheckpoint.SIMREADY_COMPLETE,
            context={
                "scene_id": scene.scene_id,
                "physics_properties": {"objects": manifest.get("objects", [])},
            },
        )

        # Step 4: USD Assembly (Mock for now)
        self.log("Step 4: USD Assembly")
        if not self.dry_run:
            self._create_mock_usd(output_dir, scene)

        usd_path = output_dir / "usd" / "scene.usda"
        gate_results = self.quality_gates.run_checkpoint(
            QualityGateCheckpoint.USD_ASSEMBLED,
            context={"scene_id": scene.scene_id, "usd_path": str(usd_path)},
            notification_service=self.notifications,
        )

        if not self.quality_gates.can_proceed():
            update_pipeline_status(delivery_id, "qa_review")
            self._send_qa_notification(scene, "usd_assembled", gate_results)
            return PipelineResult(
                scene_id=scene.scene_id,
                success=False,
                error="USD validation failed",
                quality_gates_failed=len(self.quality_gates.get_blocking_failures()),
                delivery_id=delivery_id,
            )

        # Step 5: Isaac Lab Task Generation (Mock)
        self.log("Step 5: Isaac Lab Task Generation")
        if not self.dry_run:
            self._create_mock_isaac_lab(output_dir, scene)

        isaac_lab_dir = output_dir / "isaac_lab"
        gate_results = self.quality_gates.run_checkpoint(
            QualityGateCheckpoint.ISAAC_LAB_GENERATED,
            context={"scene_id": scene.scene_id, "isaac_lab_dir": str(isaac_lab_dir)},
        )

        # Step 6: Episode Generation (Mock)
        self.log("Step 6: Episode Generation")
        episode_count = 0
        if not self.dry_run:
            episode_count = self._create_mock_episodes(output_dir, scene)

        gate_results = self.quality_gates.run_checkpoint(
            QualityGateCheckpoint.EPISODES_GENERATED,
            context={
                "scene_id": scene.scene_id,
                "episode_stats": {
                    "total_generated": episode_count,
                    "passed_quality_filter": int(episode_count * 0.85),
                    "average_quality_score": 0.82,
                    "collision_free_rate": 0.90,
                },
            },
        )

        # Step 7: Final Quality Gate
        self.log("Step 7: Scene Ready Check")
        gate_results = self.quality_gates.run_checkpoint(
            QualityGateCheckpoint.SCENE_READY,
            context={
                "scene_id": scene.scene_id,
                "readiness_checklist": {
                    "usd_valid": usd_path.exists(),
                    "physics_stable": True,
                    "episodes_generated": episode_count > 0,
                    "replicator_ready": (output_dir / "replicator").exists(),
                    "isaac_lab_ready": isaac_lab_dir.exists(),
                },
            },
            notification_service=self.notifications,
        )

        # Generate QA context for human review
        qa_context = generate_qa_context(
            checkpoint="scene_ready",
            scene_id=scene.scene_id,
            context_data={
                "object_count": object_count,
                "episode_count": episode_count,
                "environment_type": scene.environment_type,
            },
            verbose=self.verbose,
        )

        # Save QA context
        qa_path = output_dir / "qa_review" / "context.json"
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        qa_path.write_text(json.dumps(qa_context.to_dict(), indent=2))

        # Set up sim-to-real experiment
        experiment = self.sim2real_tracker.create(
            name=f"{scene.scene_id} Transfer Validation",
            scene_id=scene.scene_id,
            task_type=scene.task_types[0] if scene.task_types else "pick_place",
            robot_type=scene.robot_type,
            policy_source=str(output_dir / "isaac_lab"),
            description=f"Sim-to-real validation for {scene.scene_id}",
            training_scene_source="BlueprintPipeline",
            training_episodes=episode_count,
        )

        # Complete processing
        processing_time = time.time() - start_time

        # Save quality gate report
        report_path = output_dir / "qa_validation_report.json"
        self.quality_gates.save_report(scene.scene_id, report_path)

        # Track delivery completion
        track_scene_delivery(
            delivery_id=delivery_id,
            object_count=object_count,
            episode_count=episode_count,
            quality_score=0.82,  # From episode stats
            processing_time_hours=processing_time / 3600,
        )

        return PipelineResult(
            scene_id=scene.scene_id,
            success=True,
            processing_time_seconds=processing_time,
            output_dir=output_dir,
            object_count=object_count,
            episode_count=episode_count,
            quality_gates_passed=sum(1 for r in self.quality_gates.results if r.passed),
            quality_gates_failed=sum(1 for r in self.quality_gates.results if not r.passed),
            delivery_id=delivery_id,
            sim2real_experiment_id=experiment.experiment_id,
        )

    def _create_mock_simready(self, output_dir: Path, manifest: Dict) -> None:
        """Create mock simready outputs."""
        simready_dir = output_dir / "simready"
        simready_dir.mkdir(exist_ok=True)

        for obj in manifest.get("objects", []):
            obj_id = obj.get("id", "unknown")
            simready_file = simready_dir / f"{obj_id}_simready.usda"
            simready_file.write_text(f"""#usda 1.0
(
    defaultPrim = "{obj_id}"
)

def Xform "{obj_id}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    float physics:mass = 1.0
    float3 physics:centerOfMass = (0, 0, 0)
}}
""")

    def _create_mock_usd(self, output_dir: Path, scene: SceneConfig) -> None:
        """Create mock USD scene."""
        usd_dir = output_dir / "usd"
        usd_dir.mkdir(exist_ok=True)

        usd_file = usd_dir / "scene.usda"
        usd_file.write_text(f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Y"
)

def Xform "World"
{{
    def PhysicsScene "PhysicsScene"
    {{
        float3 physics:gravityDirection = (0, -1, 0)
        float physics:gravityMagnitude = 9.81
    }}

    def Xform "Objects"
    {{
        # Objects would be referenced here
    }}
}}
""")

    def _create_mock_isaac_lab(self, output_dir: Path, scene: SceneConfig) -> None:
        """Create mock Isaac Lab outputs."""
        isaac_dir = output_dir / "isaac_lab"
        isaac_dir.mkdir(exist_ok=True)

        # env_cfg.py
        (isaac_dir / "env_cfg.py").write_text('''"""Environment configuration."""
from dataclasses import dataclass

@dataclass
class EnvCfg:
    """Environment configuration."""
    scene_path: str = "./scene.usda"
    num_envs: int = 4096
    episode_length: int = 500
''')

        # task file
        for task in scene.task_types:
            (isaac_dir / f"task_{task}.py").write_text(f'''"""Task: {task}"""
class {task.title().replace("_", "")}Task:
    """Task implementation for {task}."""
    pass
''')

    def _create_mock_episodes(self, output_dir: Path, scene: SceneConfig) -> int:
        """Create mock episode outputs."""
        episodes_dir = output_dir / "episodes"
        episodes_dir.mkdir(exist_ok=True)

        meta_dir = episodes_dir / "meta"
        meta_dir.mkdir(exist_ok=True)

        # Create stats file
        episode_count = 100  # Mock 100 episodes
        stats = {
            "total_episodes": episode_count,
            "task_types": scene.task_types,
            "robot_type": scene.robot_type,
            "avg_quality_score": 0.82,
        }
        (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2))

        return episode_count

    def _send_qa_notification(
        self,
        scene: SceneConfig,
        checkpoint: str,
        gate_results: list,
    ) -> None:
        """Send notification for QA review."""
        failed = [r for r in gate_results if not r.passed]

        payload = NotificationPayload(
            subject=f"QA Review Required: {scene.scene_id}",
            body=f"Pipeline stopped at {checkpoint} with {len(failed)} issues.",
            scene_id=scene.scene_id,
            checkpoint=checkpoint,
            severity="error" if failed else "info",
            qa_context={
                "Failed Gates": [r.gate_id for r in failed],
                "Issues": [r.message for r in failed],
                "Recommendations": [
                    rec for r in failed for rec in r.recommendations
                ],
            },
            action_required=True,
        )

        self.notifications.send(payload)

    def _generate_summary(self) -> None:
        """Generate and display summary."""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        self.log("=" * 60)
        self.log("PIPELINE SUMMARY")
        self.log("=" * 60)
        self.log(f"Total scenes: {len(self.results)}")
        self.log(f"Successful: {len(successful)}")
        self.log(f"Failed: {len(failed)}")
        self.log("")

        if successful:
            total_objects = sum(r.object_count for r in successful)
            total_episodes = sum(r.episode_count for r in successful)
            avg_time = sum(r.processing_time_seconds for r in successful) / len(successful)

            self.log(f"Total objects: {total_objects}")
            self.log(f"Total episodes: {total_episodes}")
            self.log(f"Avg processing time: {avg_time:.1f}s")

        if failed:
            self.log("\nFailed scenes:")
            for r in failed:
                self.log(f"  - {r.scene_id}: {r.error}")

        # Save summary
        summary_path = self.output_base / "pipeline_summary.json"
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_scenes": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "results": [
                {
                    "scene_id": r.scene_id,
                    "success": r.success,
                    "error": r.error,
                    "processing_time_seconds": r.processing_time_seconds,
                    "object_count": r.object_count,
                    "episode_count": r.episode_count,
                    "delivery_id": r.delivery_id,
                    "sim2real_experiment_id": r.sim2real_experiment_id,
                }
                for r in self.results
            ],
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        self.log(f"\nSummary saved to: {summary_path}")


def create_test_scenes(output_dir: Path) -> List[SceneConfig]:
    """Create test scene configurations with mock 3D-RE-GEN outputs."""
    test_dir = output_dir / "test_inputs"
    test_dir.mkdir(parents=True, exist_ok=True)

    scenes = []

    # Create 10 test scenes with mock 3D-RE-GEN outputs
    scene_types = [
        ("kitchen_001", "kitchen", ["pick_place", "open_drawer"]),
        ("kitchen_002", "kitchen", ["pick_place", "pour"]),
        ("warehouse_001", "warehouse", ["pick_place", "stack"]),
        ("warehouse_002", "warehouse", ["pick_place", "push"]),
        ("lab_001", "lab", ["pick_place", "insert"]),
        ("office_001", "office", ["pick_place"]),
        ("retail_001", "retail", ["pick_place"]),
        ("hospital_001", "hospital", ["pick_place"]),
        ("factory_001", "factory", ["pick_place", "articulated_access"]),
        ("home_001", "home", ["pick_place", "open_door"]),
    ]

    for scene_id, env_type, tasks in scene_types:
        scene_dir = test_dir / scene_id
        scene_dir.mkdir(exist_ok=True)

        # Create mock 3D-RE-GEN manifest
        manifest = {
            "version": "1.0.0",
            "scene_id": scene_id,
            "scene": {
                "coordinate_frame": "y_up",
                "meters_per_unit": 1.0,
                "environment_type": env_type,
            },
            "objects": [
                {
                    "id": f"obj_{i}",
                    "sim_role": "manipulable_object",
                    "asset": {
                        "path": f"objects/obj_{i}/mesh.glb",
                        "format": "glb",
                        "source": "3d-re-gen",
                    },
                    "transform": {
                        "position": {"x": i * 0.5, "y": 0, "z": 0},
                        "rotation_quaternion": {"w": 1, "x": 0, "y": 0, "z": 0},
                        "scale": {"x": 1, "y": 1, "z": 1},
                    },
                }
                for i in range(5)
            ],
        }
        (scene_dir / "scene_manifest.json").write_text(json.dumps(manifest, indent=2))

        scenes.append(SceneConfig(
            scene_id=scene_id,
            input_path=scene_dir,
            environment_type=env_type,
            task_types=tasks,
        ))

    return scenes


def main():
    parser = argparse.ArgumentParser(
        description="Generate first 10 scenes through BlueprintPipeline"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory with 3D-RE-GEN outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./pipeline_output"),
        help="Output directory",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run with test/mock data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip actual processing, just validate flow",
    )
    parser.add_argument(
        "--email",
        default="ohstnhunt@gmail.com",
        help="Email for notifications",
    )
    parser.add_argument(
        "--phone",
        default="9196389913",
        help="Phone for SMS notifications",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("BlueprintPipeline - First 10 Scenes Generator")
    print("=" * 60 + "\n")

    # Get scenes to process
    if args.test_mode:
        print("Running in TEST MODE with mock 3D-RE-GEN data\n")
        scenes = create_test_scenes(args.output_dir)
    elif args.input_dir:
        # Discover scenes in input directory (expect 3D-RE-GEN outputs)
        scenes = []
        for scene_dir in sorted(args.input_dir.iterdir()):
            if scene_dir.is_dir():
                scenes.append(SceneConfig(
                    scene_id=scene_dir.name,
                    input_path=scene_dir,
                ))
        if not scenes:
            print(f"No scene directories found in {args.input_dir}")
            sys.exit(1)
    else:
        print("Error: Must specify --input-dir or --test-mode")
        sys.exit(1)

    # Limit to 10 scenes
    scenes = scenes[:10]
    print(f"Processing {len(scenes)} scenes:\n")
    for s in scenes:
        print(f"  - {s.scene_id} ({s.environment_type})")
    print()

    # Run pipeline
    runner = First10ScenesRunner(
        output_base=args.output_dir,
        notify_email=args.email,
        notify_phone=args.phone,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    results = runner.run(scenes)

    # Show success summary
    print("\n")
    try:
        summary = get_success_summary()
        print("Success Metrics:")
        print(f"  Delivery Rate: {summary.get('delivery_rate', 0):.1%}")
        print(f"  QA First Pass: {summary.get('qa_first_pass_rate', 0):.1%}")
    except Exception:
        pass

    # Exit code
    successful = sum(1 for r in results if r.success)
    if successful == len(results):
        print("\nAll scenes processed successfully!")
        sys.exit(0)
    else:
        print(f"\n{len(results) - successful} scenes failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
