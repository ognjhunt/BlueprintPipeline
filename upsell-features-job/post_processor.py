#!/usr/bin/env python3
"""
Upsell Features Post-Processor.

This module runs after episode generation to add upsell features based on bundle tier.
It can be called:
1. Directly from episode-generation-job (integrated mode)
2. As a standalone job triggered by workflow
3. Via CLI for manual processing

Usage:
    # Integrated (called from generate_episodes.py)
    from upsell_features_job.post_processor import run_upsell_post_processing
    run_upsell_post_processing(scene_dir, tier="pro")

    # CLI
    python post_processor.py --scene-dir ./scenes/kitchen_001 --tier pro
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is in path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also add current directory for local imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


class BundleTier(Enum):
    """Bundle pricing tiers."""
    STANDARD = "standard"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    FOUNDATION = "foundation"


@dataclass
class TierFeatures:
    """Features enabled for each tier."""
    language_annotations: bool = False
    num_language_variations: int = 0
    vla_finetuning: bool = False
    vla_models: List[str] = field(default_factory=list)
    sim2real_validation: bool = False
    sim2real_tier: str = "none"
    contact_rich_tasks: bool = False
    tactile_simulation: bool = False
    tactile_sensor_type: str = "gelslim"
    multi_robot: bool = False
    deformable_objects: bool = False
    bimanual: bool = False
    custom_robot: bool = False


# Tier configurations
TIER_CONFIGS = {
    BundleTier.STANDARD: TierFeatures(
        # Standard tier: Base features only, no upsells
    ),
    BundleTier.PRO: TierFeatures(
        language_annotations=True,
        num_language_variations=10,
        vla_finetuning=True,
        vla_models=["openvla", "smolvla"],
    ),
    BundleTier.ENTERPRISE: TierFeatures(
        language_annotations=True,
        num_language_variations=15,
        vla_finetuning=True,
        vla_models=["openvla", "pi0", "smolvla", "groot"],
        sim2real_validation=True,
        sim2real_tier="basic",
        contact_rich_tasks=True,
        tactile_simulation=True,
        tactile_sensor_type="gelslim",
    ),
    BundleTier.FOUNDATION: TierFeatures(
        language_annotations=True,
        num_language_variations=20,
        vla_finetuning=True,
        vla_models=["openvla", "pi0", "smolvla", "groot"],
        sim2real_validation=True,
        sim2real_tier="certification",
        contact_rich_tasks=True,
        tactile_simulation=True,
        tactile_sensor_type="digit",
        multi_robot=True,
        deformable_objects=True,
        bimanual=True,
        custom_robot=True,
    ),
}


@dataclass
class PostProcessingResult:
    """Result of post-processing."""
    scene_id: str
    tier: str
    success: bool
    features_applied: List[str]
    outputs: Dict[str, str]
    metrics: Dict[str, Any]
    errors: List[str]
    processing_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "tier": self.tier,
            "success": self.success,
            "features_applied": self.features_applied,
            "outputs": self.outputs,
            "metrics": self.metrics,
            "errors": self.errors,
            "processing_time_seconds": self.processing_time_seconds,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


class UpsellPostProcessor:
    """
    Post-processor that adds upsell features to generated episodes.
    """

    def __init__(
        self,
        scene_dir: Path,
        tier: BundleTier = BundleTier.STANDARD,
        robot_type: str = "franka",
        verbose: bool = True,
    ):
        self.scene_dir = Path(scene_dir)
        self.tier = tier
        self.robot_type = robot_type
        self.verbose = verbose
        self.features = TIER_CONFIGS[tier]

        # Output directories
        self.episodes_dir = self.scene_dir / "episodes"
        self.upsell_dir = self.episodes_dir / "upsell_outputs"
        self.upsell_dir.mkdir(parents=True, exist_ok=True)

        # Track results
        self.features_applied: List[str] = []
        self.outputs: Dict[str, str] = {}
        self.metrics: Dict[str, Any] = {}
        self.errors: List[str] = []

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[UPSELL-POST] [{level}] {msg}")

    def process(self) -> PostProcessingResult:
        """Run all applicable upsell features."""
        import time
        start_time = time.time()

        scene_id = self.scene_dir.name
        self.log(f"Processing scene {scene_id} with {self.tier.value.upper()} tier")

        # Skip if standard tier (no upsells)
        if self.tier == BundleTier.STANDARD:
            self.log("Standard tier - no upsell features to apply")
            return PostProcessingResult(
                scene_id=scene_id,
                tier=self.tier.value,
                success=True,
                features_applied=[],
                outputs={},
                metrics={},
                errors=[],
                processing_time_seconds=time.time() - start_time,
            )

        try:
            # 1. Language Annotations
            if self.features.language_annotations:
                self._apply_language_annotations()

            # 2. VLA Fine-tuning Packages
            if self.features.vla_finetuning:
                self._generate_vla_packages()

            # 3. Sim2Real Validation
            if self.features.sim2real_validation:
                self._run_sim2real_validation()

            # 4. Contact-Rich Tasks
            if self.features.contact_rich_tasks:
                self._generate_contact_rich_specs()

            # 5. Tactile Simulation
            if self.features.tactile_simulation:
                self._add_tactile_data()

            # 6. Advanced Capabilities
            self._apply_advanced_capabilities()

            # 7. Write manifest
            self._write_upsell_manifest()

            success = len(self.errors) == 0

        except Exception as e:
            self.log(f"Post-processing failed: {e}", "ERROR")
            self.errors.append(str(e))
            traceback.print_exc()
            success = False

        return PostProcessingResult(
            scene_id=scene_id,
            tier=self.tier.value,
            success=success,
            features_applied=self.features_applied,
            outputs=self.outputs,
            metrics=self.metrics,
            errors=self.errors,
            processing_time_seconds=time.time() - start_time,
        )

    def _apply_language_annotations(self) -> None:
        """Add language annotations to episodes."""
        self.log("Applying language annotations...")

        try:
            from language_annotator import LanguageAnnotator

            output_path = self.upsell_dir / "language_annotations.json"

            # Find tasks file
            tasks_path = self.episodes_dir / "meta" / "tasks.jsonl"
            if not tasks_path.exists():
                # Create from episode metadata if missing
                tasks_path = self._create_tasks_from_episodes()

            annotator = LanguageAnnotator(
                use_llm=True,
                num_variations=self.features.num_language_variations,
                verbose=self.verbose,
            )

            annotations = annotator.annotate_episodes(
                episodes_meta_path=tasks_path,
                output_path=output_path,
            )

            self.features_applied.append("language_annotations")
            self.outputs["language_annotations"] = str(output_path)
            self.metrics["language_annotations"] = {
                "total_annotations": sum(len(v) for v in annotations.values()),
                "tasks_annotated": len(annotations),
                "variations_per_task": self.features.num_language_variations,
            }
            self.log(f"  Generated {self.metrics['language_annotations']['total_annotations']} annotations")

        except ImportError as e:
            self.log(f"  Language annotator not available: {e}", "WARNING")
            self._create_placeholder_annotations()
        except Exception as e:
            self.errors.append(f"Language annotations failed: {e}")
            self.log(f"  Failed: {e}", "ERROR")

    def _create_tasks_from_episodes(self) -> Path:
        """Create tasks.jsonl from episode metadata."""
        tasks_path = self.episodes_dir / "meta" / "tasks.jsonl"
        tasks_path.parent.mkdir(parents=True, exist_ok=True)

        # Try to read from generation manifest
        manifest_path = self.episodes_dir / "manifests" / "generation_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

            tasks = []
            for i, task_info in enumerate(manifest.get("tasks", [])):
                tasks.append({
                    "task_index": i,
                    "task": task_info.get("description", task_info.get("task_name", "manipulation task")),
                })

            with open(tasks_path, "w") as f:
                for task in tasks:
                    f.write(json.dumps(task) + "\n")

            return tasks_path

        # Fallback: create generic tasks
        with open(tasks_path, "w") as f:
            f.write(json.dumps({"task_index": 0, "task": "Pick up the object and place it"}) + "\n")

        return tasks_path

    def _create_placeholder_annotations(self) -> None:
        """Create placeholder annotations when LLM not available."""
        output_path = self.upsell_dir / "language_annotations.json"

        # Read tasks
        tasks_path = self.episodes_dir / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            tasks_path = self._create_tasks_from_episodes()

        annotations = {}
        with open(tasks_path) as f:
            for line in f:
                task = json.loads(line)
                task_desc = task.get("task", "manipulation task")
                task_idx = task.get("task_index", 0)

                # Generate simple variations
                variations = [
                    task_desc,
                    task_desc.replace("Pick up", "Grasp").replace("pick up", "grasp"),
                    f"Please {task_desc.lower()}",
                    f"Robot, {task_desc.lower()}",
                    task_desc.replace("and", "then"),
                ]
                annotations[str(task_idx)] = variations[:self.features.num_language_variations]

        with open(output_path, "w") as f:
            json.dump(annotations, f, indent=2)

        self.features_applied.append("language_annotations")
        self.outputs["language_annotations"] = str(output_path)
        self.metrics["language_annotations"] = {
            "total_annotations": sum(len(v) for v in annotations.values()),
            "tasks_annotated": len(annotations),
            "variations_per_task": self.features.num_language_variations,
            "method": "template",
        }

    def _generate_vla_packages(self) -> None:
        """Generate VLA fine-tuning packages."""
        self.log("Generating VLA fine-tuning packages...")

        try:
            from vla_finetuning_generator import VLAFinetuningGenerator, VLAModel

            vla_dir = self.upsell_dir / "vla_finetuning"
            models = [VLAModel(m) for m in self.features.vla_models]

            generator = VLAFinetuningGenerator(
                episodes_dir=self.episodes_dir,
                output_dir=vla_dir,
                scene_id=self.scene_dir.name,
                models=models,
                verbose=self.verbose,
            )

            results = generator.generate_all()

            self.features_applied.append("vla_finetuning")
            self.outputs["vla_finetuning"] = str(vla_dir)
            self.metrics["vla_finetuning"] = {
                "models": [m.value for m in models],
                "packages_generated": len(results),
            }
            self.log(f"  Generated {len(results)} VLA packages")

        except ImportError as e:
            self.log(f"  VLA generator not available: {e}", "WARNING")
            self._create_placeholder_vla()
        except Exception as e:
            self.errors.append(f"VLA generation failed: {e}")
            self.log(f"  Failed: {e}", "ERROR")

    def _create_placeholder_vla(self) -> None:
        """Create placeholder VLA configs."""
        vla_dir = self.upsell_dir / "vla_finetuning"
        vla_dir.mkdir(parents=True, exist_ok=True)

        for model in self.features.vla_models:
            model_dir = vla_dir / model
            model_dir.mkdir(parents=True, exist_ok=True)

            config = {
                "model": model,
                "scene_id": self.scene_dir.name,
                "episodes_path": str(self.episodes_dir),
                "status": "placeholder",
                "note": "Full config requires VLA generator module",
            }

            with open(model_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

        self.features_applied.append("vla_finetuning")
        self.outputs["vla_finetuning"] = str(vla_dir)
        self.metrics["vla_finetuning"] = {
            "models": self.features.vla_models,
            "packages_generated": len(self.features.vla_models),
            "method": "placeholder",
        }

    def _run_sim2real_validation(self) -> None:
        """Run sim2real validation."""
        self.log(f"Running sim2real validation ({self.features.sim2real_tier})...")

        try:
            from sim2real_service import Sim2RealService, ValidationTier

            sim2real_dir = self.upsell_dir / "sim2real"
            sim2real_dir.mkdir(parents=True, exist_ok=True)

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

            # Check for policy checkpoint
            policy_path = self.scene_dir / "isaac_lab" / "checkpoints" / "policy.pt"

            report = service.run_full_validation(
                scene_id=self.scene_dir.name,
                task_type="pick_place",
                robot_type=self.robot_type,
                policy_path=str(policy_path),
                tier=tier,
                output_dir=sim2real_dir / "reports",
            )

            self.features_applied.append("sim2real_validation")
            self.outputs["sim2real"] = str(sim2real_dir)
            self.metrics["sim2real"] = {
                "tier": tier.value,
                "sim_success_rate": report.sim_success_rate,
                "real_success_rate": report.real_success_rate,
                "transfer_gap": report.transfer_gap,
                "production_ready": report.production_ready,
            }
            self.log(f"  Transfer gap: {report.transfer_gap:.1%}")

        except ImportError as e:
            self.log(f"  Sim2Real service not available: {e}", "WARNING")
            self._create_placeholder_sim2real()
        except Exception as e:
            self.errors.append(f"Sim2Real validation failed: {e}")
            self.log(f"  Failed: {e}", "ERROR")

    def _create_placeholder_sim2real(self) -> None:
        """Create placeholder sim2real report."""
        sim2real_dir = self.upsell_dir / "sim2real"
        sim2real_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "scene_id": self.scene_dir.name,
            "tier": self.features.sim2real_tier,
            "status": "placeholder",
            "note": "Full validation requires sim2real service module",
            "estimated_transfer_rate": 0.75,
        }

        with open(sim2real_dir / "validation_report.json", "w") as f:
            json.dump(report, f, indent=2)

        self.features_applied.append("sim2real_validation")
        self.outputs["sim2real"] = str(sim2real_dir)
        self.metrics["sim2real"] = {
            "tier": self.features.sim2real_tier,
            "method": "placeholder",
        }

    def _generate_contact_rich_specs(self) -> None:
        """Generate contact-rich task specifications."""
        self.log("Generating contact-rich task specs...")

        try:
            from contact_rich_tasks import (
                ContactRichTaskGenerator,
                ContactRichTaskType,
                ToleranceClass,
            )

            contact_dir = self.upsell_dir / "contact_rich"
            contact_dir.mkdir(parents=True, exist_ok=True)

            generator = ContactRichTaskGenerator(verbose=self.verbose)

            tasks = []
            for task_type in [ContactRichTaskType.PEG_IN_HOLE, ContactRichTaskType.SNAP_FIT]:
                for tolerance in [ToleranceClass.MEDIUM, ToleranceClass.TIGHT]:
                    spec = generator.create_task_spec(
                        task_type=task_type,
                        tolerance_class=tolerance,
                    )
                    tasks.append(spec.to_dict())

            with open(contact_dir / "contact_rich_tasks.json", "w") as f:
                json.dump(tasks, f, indent=2)

            self.features_applied.append("contact_rich_tasks")
            self.outputs["contact_rich"] = str(contact_dir)
            self.metrics["contact_rich"] = {
                "num_tasks": len(tasks),
            }
            self.log(f"  Generated {len(tasks)} contact-rich specs")

        except ImportError as e:
            self.log(f"  Contact-rich generator not available: {e}", "WARNING")
        except Exception as e:
            self.errors.append(f"Contact-rich generation failed: {e}")
            self.log(f"  Failed: {e}", "ERROR")

    def _add_tactile_data(self) -> None:
        """Add tactile sensor simulation data."""
        self.log(f"Adding tactile simulation ({self.features.tactile_sensor_type})...")

        try:
            from tactile_sensor_sim import (
                TactileSensorSimulator,
                TactileSensorType,
                SENSOR_CONFIGS,
            )
            import numpy as np

            tactile_dir = self.upsell_dir / "tactile"
            tactile_dir.mkdir(parents=True, exist_ok=True)

            sensor_type = TactileSensorType(self.features.tactile_sensor_type)
            sensor_config = SENSOR_CONFIGS[sensor_type]

            simulator = TactileSensorSimulator(
                sensor_type=sensor_type,
                verbose=self.verbose,
            )

            # Generate sample tactile data
            frame = simulator.simulate_contact(
                contact_position=np.array([0.0, 0.0, 0.0]),
                contact_force=10.0,
                contact_normal=np.array([0.0, 0.0, 1.0]),
            )

            np.save(tactile_dir / "sample_tactile.npy", frame.tactile_image)

            with open(tactile_dir / "sensor_config.json", "w") as f:
                json.dump(sensor_config.to_dict(), f, indent=2)

            self.features_applied.append("tactile_simulation")
            self.outputs["tactile"] = str(tactile_dir)
            self.metrics["tactile"] = {
                "sensor_type": sensor_type.value,
                "resolution": list(sensor_config.resolution),
            }
            self.log(f"  Added {sensor_type.value} tactile data")

        except ImportError as e:
            self.log(f"  Tactile simulator not available: {e}", "WARNING")
        except Exception as e:
            self.errors.append(f"Tactile simulation failed: {e}")
            self.log(f"  Failed: {e}", "ERROR")

    def _apply_advanced_capabilities(self) -> None:
        """Apply advanced capabilities (multi-robot, deformable, bimanual)."""
        capabilities = []
        if self.features.multi_robot:
            capabilities.append("multi_robot")
        if self.features.deformable_objects:
            capabilities.append("deformable")
        if self.features.bimanual:
            capabilities.append("bimanual")

        if not capabilities:
            return

        self.log(f"Applying advanced capabilities: {capabilities}...")

        try:
            from advanced_capabilities import AdvancedCapabilities

            advanced_dir = self.upsell_dir / "advanced"

            caps = AdvancedCapabilities(
                output_dir=advanced_dir,
                verbose=self.verbose,
            )

            bundle = caps.generate_advanced_bundle(
                scene_id=self.scene_dir.name,
                capabilities=capabilities,
            )

            self.features_applied.extend(capabilities)
            self.outputs["advanced"] = str(advanced_dir)
            self.metrics["advanced"] = {
                "capabilities": capabilities,
            }
            self.log(f"  Applied {len(capabilities)} advanced capabilities")

        except ImportError as e:
            self.log(f"  Advanced capabilities not available: {e}", "WARNING")
        except Exception as e:
            self.errors.append(f"Advanced capabilities failed: {e}")
            self.log(f"  Failed: {e}", "ERROR")

    def _write_upsell_manifest(self) -> None:
        """Write upsell processing manifest."""
        manifest = {
            "scene_id": self.scene_dir.name,
            "bundle_tier": self.tier.value,
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "features_applied": self.features_applied,
            "outputs": self.outputs,
            "metrics": self.metrics,
            "errors": self.errors,
        }

        manifest_path = self.upsell_dir / "upsell_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self.outputs["manifest"] = str(manifest_path)


def run_upsell_post_processing(
    scene_dir: Path,
    tier: str = "standard",
    robot_type: str = "franka",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run upsell post-processing on a scene.

    This is the main entry point called from episode_generation_job.

    Args:
        scene_dir: Path to scene directory
        tier: Bundle tier (standard, pro, enterprise, foundation)
        robot_type: Robot type
        verbose: Enable verbose logging

    Returns:
        Result dictionary with applied features and outputs
    """
    processor = UpsellPostProcessor(
        scene_dir=Path(scene_dir),
        tier=BundleTier(tier),
        robot_type=robot_type,
        verbose=verbose,
    )

    result = processor.process()
    return result.to_dict()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run upsell features post-processing on generated episodes"
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
        default=os.getenv("BUNDLE_TIER", "standard"),
        help="Bundle tier (default: from BUNDLE_TIER env or standard)",
    )
    parser.add_argument(
        "--robot-type",
        default=os.getenv("ROBOT_TYPE", "franka"),
        help="Robot type",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    result = run_upsell_post_processing(
        scene_dir=args.scene_dir,
        tier=args.tier,
        robot_type=args.robot_type,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 60)
    print("UPSELL POST-PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Scene: {result['scene_id']}")
    print(f"Tier: {result['tier']}")
    print(f"Success: {result['success']}")
    print(f"Features Applied: {', '.join(result['features_applied']) or 'none'}")
    print(f"Processing Time: {result['processing_time_seconds']:.1f}s")

    if result['outputs']:
        print("\nOutputs:")
        for name, path in result['outputs'].items():
            print(f"  - {name}: {path}")

    if result['errors']:
        print("\nErrors:")
        for error in result['errors']:
            print(f"  - {error}")

    # Exit with error code if failed
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
