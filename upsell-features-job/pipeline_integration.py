#!/usr/bin/env python3
"""
Pipeline Integration for Upsell Features.

Integrates all upsell features into the main BlueprintPipeline workflow.
This is the main entry point for running the enhanced pipeline.

Usage:
    python pipeline_integration.py --scene-dir ./scenes/kitchen_001 --tier pro
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import upsell feature modules
from bundle_config import (
    BundleConfigManager,
    BundleTier,
    BundleFeatures,
    BUNDLE_CONFIGS,
)
from vla_finetuning_generator import VLAFinetuningGenerator, VLAModel
from language_annotator import LanguageAnnotator, integrate_with_lerobot_export
from sim2real_service import Sim2RealService, ValidationTier
from contact_rich_tasks import (
    ContactRichTaskGenerator,
    ContactRichTaskType,
    ToleranceClass,
)
from tactile_sensor_sim import (
    TactileSensorSimulator,
    DualGripperTactileSimulator,
    TactileSensorType,
    SENSOR_CONFIGS,
)
from advanced_capabilities import (
    AdvancedCapabilities,
    MultiRobotCoordinator,
    DeformableObjectGenerator,
    BimanualTaskGenerator,
)


@dataclass
class PipelineResult:
    """Result of pipeline run."""
    scene_id: str
    bundle_tier: str
    success: bool
    outputs: Dict[str, Path]
    metrics: Dict[str, Any]
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "bundle_tier": self.bundle_tier,
            "success": self.success,
            "outputs": {k: str(v) for k, v in self.outputs.items()},
            "metrics": self.metrics,
            "errors": self.errors,
        }


class EnhancedPipeline:
    """
    Enhanced BlueprintPipeline with all upsell features integrated.
    """

    def __init__(
        self,
        scene_dir: Path,
        bundle_tier: BundleTier = BundleTier.STANDARD,
        robot_type: str = "franka",
        verbose: bool = True,
    ):
        self.scene_dir = Path(scene_dir)
        self.bundle_tier = bundle_tier
        self.robot_type = robot_type
        self.verbose = verbose

        # Get bundle configuration
        self.config_manager = BundleConfigManager(verbose=verbose)
        self.features = BUNDLE_CONFIGS[bundle_tier]

        # Initialize output directories
        self.outputs_dir = self.scene_dir / "upsell_outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Track results
        self.outputs: Dict[str, Path] = {}
        self.metrics: Dict[str, Any] = {}
        self.errors: List[str] = []

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[PIPELINE] {msg}")

    def run(self) -> PipelineResult:
        """Run the complete enhanced pipeline."""
        self.log(f"Starting {self.bundle_tier.value.upper()} pipeline for {self.scene_dir.name}")
        self.log(f"Features: {self._summarize_features()}")

        scene_id = self.scene_dir.name

        try:
            # 1. Run base episode generation (would call existing pipeline)
            self._run_base_pipeline()

            # 2. Add language annotations
            if self.features.language_annotations:
                self._add_language_annotations()

            # 3. Generate VLA fine-tuning packages
            if self.features.vla_finetuning_package:
                self._generate_vla_packages()

            # 4. Run sim2real validation
            if self.features.sim2real_validation:
                self._run_sim2real_validation()

            # 5. Generate contact-rich tasks
            if self.features.contact_rich_tasks:
                self._generate_contact_rich_tasks()

            # 6. Add tactile sensor data
            if self.features.tactile_simulation:
                self._add_tactile_data()

            # 7. Generate advanced capabilities
            self._generate_advanced_capabilities()

            # 8. Generate bundle manifest
            self._generate_manifest()

            success = True

        except Exception as e:
            self.log(f"Pipeline error: {e}")
            self.errors.append(str(e))
            success = False

        return PipelineResult(
            scene_id=scene_id,
            bundle_tier=self.bundle_tier.value,
            success=success,
            outputs=self.outputs,
            metrics=self.metrics,
            errors=self.errors,
        )

    def _summarize_features(self) -> str:
        """Summarize enabled features."""
        enabled = []
        if self.features.language_annotations:
            enabled.append("language")
        if self.features.vla_finetuning_package:
            enabled.append(f"vla({len(self.features.vla_models)} models)")
        if self.features.sim2real_validation:
            enabled.append(f"sim2real({self.features.sim2real_tier})")
        if self.features.contact_rich_tasks:
            enabled.append("contact-rich")
        if self.features.tactile_simulation:
            enabled.append("tactile")
        if self.features.multi_robot:
            enabled.append("multi-robot")
        if self.features.bimanual:
            enabled.append("bimanual")
        return ", ".join(enabled) if enabled else "base only"

    def _run_base_pipeline(self) -> None:
        """Run the base episode generation pipeline."""
        self.log("Running base episode generation...")

        # This would call the existing episode generation job
        # For now, we check if episodes exist or simulate
        episodes_dir = self.scene_dir / "episodes"

        if episodes_dir.exists():
            self.log("Found existing episodes")
        else:
            self.log("Episodes directory not found - would run generation")
            # In production:
            # from episode_generation_job.generate_episodes import main as generate_episodes
            # generate_episodes(scene_dir=self.scene_dir, ...)

        self.outputs["episodes"] = episodes_dir
        self.metrics["base_pipeline"] = {
            "episodes_generated": self.features.episodes,
            "variations": self.features.variations,
        }

    def _add_language_annotations(self) -> None:
        """Add language annotations to episodes."""
        self.log("Adding language annotations...")

        episodes_dir = self.scene_dir / "episodes"
        tasks_path = episodes_dir / "meta" / "tasks.jsonl"

        if not tasks_path.exists():
            self.log("No tasks.jsonl found - creating sample")
            # Create sample tasks for demo
            (episodes_dir / "meta").mkdir(parents=True, exist_ok=True)
            sample_tasks = [
                {"task_index": 0, "task": "Pick up the cup and place it on the counter"},
                {"task_index": 1, "task": "Open the drawer"},
            ]
            with open(tasks_path, "w") as f:
                for task in sample_tasks:
                    f.write(json.dumps(task) + "\n")

        output_path = self.outputs_dir / "language_annotations.json"

        annotator = LanguageAnnotator(
            use_llm=True,
            num_variations=self.features.num_language_variations,
            verbose=self.verbose,
        )

        annotations = annotator.annotate_episodes(
            episodes_meta_path=tasks_path,
            output_path=output_path,
        )

        self.outputs["language_annotations"] = output_path
        self.metrics["language"] = {
            "total_annotations": sum(len(v) for v in annotations.values()),
            "tasks_annotated": len(annotations),
            "variations_per_task": self.features.num_language_variations,
        }

    def _generate_vla_packages(self) -> None:
        """Generate VLA fine-tuning packages."""
        self.log("Generating VLA fine-tuning packages...")

        episodes_dir = self.scene_dir / "episodes"
        vla_dir = self.outputs_dir / "vla_finetuning"

        # Determine which models to generate for
        models = [VLAModel(m) for m in self.features.vla_models]

        generator = VLAFinetuningGenerator(
            episodes_dir=episodes_dir,
            output_dir=vla_dir,
            scene_id=self.scene_dir.name,
            models=models,
            verbose=self.verbose,
        )

        results = generator.generate_all()

        self.outputs["vla_finetuning"] = vla_dir
        self.metrics["vla"] = {
            "models_generated": [m.value for m in models],
            "packages": list(results.keys()),
        }

    def _run_sim2real_validation(self) -> None:
        """Run sim2real validation."""
        self.log(f"Running sim2real validation ({self.features.sim2real_tier})...")

        sim2real_dir = self.outputs_dir / "sim2real"

        # Map tier
        tier_map = {
            "basic": ValidationTier.BASIC,
            "comprehensive": ValidationTier.COMPREHENSIVE,
            "certification": ValidationTier.CERTIFICATION,
        }
        tier = tier_map.get(self.features.sim2real_tier, ValidationTier.BASIC)

        service = Sim2RealService(
            experiments_dir=sim2real_dir,
            verbose=self.verbose,
        )

        # Generate demo policy path
        policy_path = str(self.scene_dir / "isaac_lab" / "checkpoints" / "policy.pt")

        report = service.run_full_validation(
            scene_id=self.scene_dir.name,
            task_type=service.validator.experiments.get("task_type", "pick_place"),
            robot_type=service.validator.experiments.get("robot_type", "franka"),
            policy_path=policy_path,
            tier=tier,
            output_dir=sim2real_dir / "reports",
        )

        self.outputs["sim2real"] = sim2real_dir
        self.metrics["sim2real"] = {
            "tier": tier.value,
            "sim_success_rate": report.sim_success_rate,
            "real_success_rate": report.real_success_rate,
            "transfer_gap": report.transfer_gap,
            "transfer_quality": report.transfer_quality,
            "production_ready": report.production_ready,
        }

    def _generate_contact_rich_tasks(self) -> None:
        """Generate contact-rich task episodes."""
        self.log("Generating contact-rich tasks...")

        contact_dir = self.outputs_dir / "contact_rich"
        contact_dir.mkdir(parents=True, exist_ok=True)

        generator = ContactRichTaskGenerator(verbose=self.verbose)

        # Generate peg-in-hole tasks
        tasks = []
        for tolerance in [ToleranceClass.MEDIUM, ToleranceClass.TIGHT]:
            spec = generator.create_task_spec(
                task_type=ContactRichTaskType.PEG_IN_HOLE,
                tolerance_class=tolerance,
            )
            tasks.append(spec)

        # Save task specs
        specs_path = contact_dir / "contact_rich_tasks.json"
        with open(specs_path, "w") as f:
            json.dump([t.to_dict() for t in tasks], f, indent=2)

        self.outputs["contact_rich"] = contact_dir
        self.metrics["contact_rich"] = {
            "task_types": ["peg_in_hole"],
            "tolerances": ["medium", "tight"],
            "num_specs": len(tasks),
        }

    def _add_tactile_data(self) -> None:
        """Add tactile sensor simulation data."""
        self.log(f"Adding tactile simulation ({self.features.tactile_sensor_type})...")

        tactile_dir = self.outputs_dir / "tactile"
        tactile_dir.mkdir(parents=True, exist_ok=True)

        sensor_type = TactileSensorType(self.features.tactile_sensor_type)
        sensor_config = SENSOR_CONFIGS[sensor_type]

        simulator = DualGripperTactileSimulator(
            sensor_type=sensor_type,
            verbose=self.verbose,
        )

        # Generate sample tactile data
        import numpy as np
        left_frame, right_frame = simulator.simulate_grasp(
            object_position=np.array([0.5, 0, 0.1]),
            gripper_width=0.04,
            grasp_force=10.0,
        )

        # Save sample data
        np.save(tactile_dir / "sample_left_tactile.npy", left_frame.tactile_image)
        np.save(tactile_dir / "sample_right_tactile.npy", right_frame.tactile_image)

        # Save config
        with open(tactile_dir / "sensor_config.json", "w") as f:
            json.dump(sensor_config.to_dict(), f, indent=2)

        self.outputs["tactile"] = tactile_dir
        self.metrics["tactile"] = {
            "sensor_type": sensor_type.value,
            "resolution": list(sensor_config.resolution),
        }

    def _generate_advanced_capabilities(self) -> None:
        """Generate advanced capability data."""
        capabilities_to_generate = []

        if self.features.multi_robot:
            capabilities_to_generate.append("multi_robot")
        if self.features.deformable_objects:
            capabilities_to_generate.append("deformable")
        if self.features.bimanual:
            capabilities_to_generate.append("bimanual")

        if not capabilities_to_generate:
            return

        self.log(f"Generating advanced capabilities: {capabilities_to_generate}")

        advanced_dir = self.outputs_dir / "advanced"

        caps = AdvancedCapabilities(
            output_dir=advanced_dir,
            verbose=self.verbose,
        )

        bundle = caps.generate_advanced_bundle(
            scene_id=self.scene_dir.name,
            capabilities=capabilities_to_generate,
        )

        self.outputs["advanced"] = advanced_dir
        self.metrics["advanced"] = {
            "capabilities": capabilities_to_generate,
        }

    def _generate_manifest(self) -> None:
        """Generate bundle manifest."""
        self.log("Generating bundle manifest...")

        manifest = {
            "scene_id": self.scene_dir.name,
            "bundle_tier": self.bundle_tier.value,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "features": self.features.to_dict(),
            "outputs": {k: str(v) for k, v in self.outputs.items()},
            "metrics": self.metrics,
        }

        manifest_path = self.outputs_dir / "bundle_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self.outputs["manifest"] = manifest_path


def run_enhanced_pipeline(
    scene_dir: Path,
    tier: str = "standard",
    robot_type: str = "franka",
) -> Dict[str, Any]:
    """
    Main entry point for running the enhanced pipeline.

    This function is called from the main pipeline workflow.
    """
    pipeline = EnhancedPipeline(
        scene_dir=scene_dir,
        bundle_tier=BundleTier(tier),
        robot_type=robot_type,
    )

    result = pipeline.run()
    return result.to_dict()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run enhanced BlueprintPipeline with upsell features"
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        required=True,
        help="Path to scene directory",
    )
    parser.add_argument(
        "--tier",
        choices=["standard", "pro", "enterprise", "foundation"],
        default="standard",
        help="Bundle tier",
    )
    parser.add_argument(
        "--robot-type",
        default="franka",
        help="Robot type",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running",
    )

    args = parser.parse_args()

    if args.dry_run:
        manager = BundleConfigManager()
        config = manager.generate_pipeline_config(
            tier=BundleTier(args.tier),
            scene_id=args.scene_dir.name,
            robot_type=args.robot_type,
        )
        print("\nPipeline Configuration:")
        print(json.dumps(config, indent=2))
        return

    result = run_enhanced_pipeline(
        scene_dir=args.scene_dir,
        tier=args.tier,
        robot_type=args.robot_type,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nScene: {result['scene_id']}")
    print(f"Tier: {result['bundle_tier']}")
    print(f"Success: {result['success']}")

    print("\nOutputs:")
    for name, path in result['outputs'].items():
        print(f"  - {name}: {path}")

    print("\nMetrics:")
    print(json.dumps(result['metrics'], indent=2))

    if result['errors']:
        print("\nErrors:")
        for error in result['errors']:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
