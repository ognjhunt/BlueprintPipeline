"""
Bundle Packager for Dream2Flow.

Packages Dream2Flow pipeline outputs (video, flow, trajectory) into
self-contained bundles with manifests for downstream consumption.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    Dream2FlowBundle,
    Dream2FlowPipelineOutput,
)


class Dream2FlowBundlePackager:
    """Packages Dream2Flow outputs into bundles."""

    def __init__(self, output_dir: Path, verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.verbose = verbose

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[D2F-PACKAGER] [{level}] {msg}")

    def package_bundle(
        self,
        bundle: Dream2FlowBundle,
        include_intermediate: bool = True,
    ) -> Path:
        """
        Package a single bundle to disk.

        Returns path to bundle directory.
        """
        bundle_dir = self.output_dir / bundle.bundle_id
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # Create manifest
        manifest = self._create_manifest(bundle)

        # Write manifest
        manifest_path = bundle_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

        bundle.bundle_dir = bundle_dir
        bundle.manifest_path = manifest_path

        self.log(f"Packaged bundle: {bundle.bundle_id}")
        return bundle_dir

    def package_all(
        self,
        bundles: list[Dream2FlowBundle],
        scene_id: str,
        include_intermediate: bool = True,
    ) -> Dream2FlowPipelineOutput:
        """
        Package all bundles and create master manifest.

        Returns pipeline output with all bundles and manifest path.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Package each bundle
        for bundle in bundles:
            self.package_bundle(bundle, include_intermediate)

        # Count successes
        num_successful_videos = sum(1 for b in bundles if b.video_generation_success)
        num_successful_flows = sum(1 for b in bundles if b.flow_extraction_success)
        num_successful_trajectories = sum(1 for b in bundles if b.robot_execution_success)

        # Create master manifest
        master_manifest = {
            "scene_id": scene_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "num_bundles": len(bundles),
            "success_rates": {
                "video_generation": num_successful_videos / len(bundles) if bundles else 0,
                "flow_extraction": num_successful_flows / len(bundles) if bundles else 0,
                "robot_execution": num_successful_trajectories / len(bundles) if bundles else 0,
            },
            "bundles": [
                {
                    "bundle_id": b.bundle_id,
                    "instruction": b.instruction.text if b.instruction else None,
                    "success": b.success,
                    "video_path": str(b.generated_video.video_path) if b.generated_video and b.generated_video.video_path else None,
                    "num_frames": b.num_frames,
                }
                for b in bundles
            ],
        }

        master_manifest_path = self.output_dir / "dream2flow_bundles_manifest.json"
        master_manifest_path.write_text(json.dumps(master_manifest, indent=2))

        self.log(f"Created master manifest with {len(bundles)} bundles")

        return Dream2FlowPipelineOutput(
            scene_id=scene_id,
            bundles=bundles,
            output_dir=self.output_dir,
            num_tasks=len(bundles),
            num_successful_videos=num_successful_videos,
            num_successful_flows=num_successful_flows,
            num_successful_trajectories=num_successful_trajectories,
            manifest_path=master_manifest_path,
            video_generation_failures=len(bundles) - num_successful_videos,
            flow_extraction_failures=num_successful_videos - num_successful_flows,
            robot_execution_failures=num_successful_flows - num_successful_trajectories,
        )

    def _create_manifest(self, bundle: Dream2FlowBundle) -> dict[str, Any]:
        """Create manifest dict for a bundle."""
        manifest = {
            "bundle_id": bundle.bundle_id,
            "scene_id": bundle.scene_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "resolution": list(bundle.resolution),
            "num_frames": bundle.num_frames,
            "fps": bundle.fps,
            "success": bundle.success,
            "video_generation_success": bundle.video_generation_success,
            "flow_extraction_success": bundle.flow_extraction_success,
            "robot_execution_success": bundle.robot_execution_success,
        }

        # Add instruction
        if bundle.instruction:
            manifest["instruction"] = {
                "text": bundle.instruction.text,
                "task_type": bundle.instruction.task_type.value,
                "target_object": bundle.instruction.target_object,
            }

        # Add video info
        if bundle.generated_video:
            manifest["generated_video"] = {
                "video_id": bundle.generated_video.video_id,
                "video_path": bundle.generated_video.video_path.name if bundle.generated_video.video_path else None,
                "frames_dir": bundle.generated_video.frames_dir.name if bundle.generated_video.frames_dir else None,
                "model_name": bundle.generated_video.model_name,
                "quality_score": bundle.generated_video.quality_score,
            }

        # Add flow info
        if bundle.flow_extraction and bundle.flow_extraction.success:
            manifest["flow_extraction"] = {
                "num_objects": bundle.flow_extraction.num_objects,
                "method": bundle.flow_extraction.method.value,
                "confidence": bundle.flow_extraction.extraction_confidence,
            }

        # Add trajectory info
        if bundle.robot_trajectory:
            manifest["robot_trajectory"] = {
                "trajectory_id": bundle.robot_trajectory.trajectory_id,
                "robot": bundle.robot_trajectory.robot.value,
                "num_frames": bundle.robot_trajectory.num_frames,
                "mean_tracking_error": bundle.robot_trajectory.mean_tracking_error,
            }

        # Add metadata
        manifest["metadata"] = bundle.metadata

        return manifest


def package_dream2flow_bundle(
    bundle: Dream2FlowBundle,
    output_dir: Path,
) -> Path:
    """Convenience function to package a single bundle."""
    packager = Dream2FlowBundlePackager(output_dir)
    return packager.package_bundle(bundle)
