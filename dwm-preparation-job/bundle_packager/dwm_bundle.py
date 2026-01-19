"""
DWM bundle packager.

Packages all DWM conditioning inputs into a structured format
ready for DWM inference or downstream processing.

Bundle Structure:
    {bundle_id}/
    ├── manifest.json           # Bundle metadata
    ├── static_scene_video.mp4  # Rendered static scene
    ├── hand_mesh_video.mp4     # Rendered hand meshes
    ├── camera_trajectory.json  # Camera poses
    ├── hand_trajectory.json    # Hand poses
    ├── frames/
    │   ├── static_scene/       # Individual frames (optional)
    │   └── hand_mesh/          # Individual frames (optional)
    └── metadata/
        ├── scene_info.json     # Source scene info
        └── prompt.txt          # Text prompt
"""

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    CameraTrajectory,
    DWMConditioningBundle,
    DWMPipelineOutput,
    DWMSceneConfig,
    HandActionType,
    HandTrajectory,
)


@dataclass
class BundleManifest:
    """Manifest for a DWM conditioning bundle."""

    # Bundle info
    bundle_id: str
    scene_id: str
    created_at: str

    # Content info
    num_frames: int
    fps: float
    resolution: tuple[int, int]

    # Action info
    action_type: str
    target_object_id: Optional[str]
    text_prompt: str
    hand_model: Optional[str] = None
    hand_model_requested: Optional[str] = None

    # File paths (relative to bundle root)
    static_scene_video: Optional[str] = None
    hand_mesh_video: Optional[str] = None
    camera_trajectory_file: str = "camera_trajectory.json"
    hand_trajectory_file: Optional[str] = None
    robot_actions_file: Optional[str] = None
    interaction_video: Optional[str] = None

    # Optional frame directories
    static_scene_frames_dir: Optional[str] = None
    hand_mesh_frames_dir: Optional[str] = None
    static_scene_depth_dir: Optional[str] = None
    static_scene_seg_dir: Optional[str] = None
    interaction_frames_dir: Optional[str] = None
    physics_rollout_file: Optional[str] = None

    # Source info
    source_manifest: Optional[str] = None
    source_usd: Optional[str] = None
    scene_state_file: Optional[str] = None

    # DWM compatibility
    dwm_compatible: bool = True
    dwm_version: str = "1.0"


def generate_text_prompt(
    action_type: HandActionType,
    target_category: Optional[str] = None,
    target_description: Optional[str] = None,
) -> str:
    """
    Generate a text prompt for DWM conditioning.

    Based on DWM paper: text prompts improve generation quality
    by providing semantic guidance.

    Args:
        action_type: Type of manipulation action
        target_category: Category of target object (e.g., "drawer", "door")
        target_description: Description of target

    Returns:
        Text prompt string
    """
    action_verbs = {
        HandActionType.REACH: "reaching toward",
        HandActionType.GRASP: "grasping",
        HandActionType.PULL: "pulling",
        HandActionType.PUSH: "pushing",
        HandActionType.ROTATE: "rotating",
        HandActionType.LIFT: "lifting",
        HandActionType.PLACE: "placing",
        HandActionType.SLIDE: "sliding",
    }

    verb = action_verbs.get(action_type, "interacting with")

    if target_description:
        target = target_description
    elif target_category:
        target = f"the {target_category}"
    else:
        target = "an object"

    # Build natural language prompt
    prompts = {
        HandActionType.REACH: f"A hand {verb} {target}.",
        HandActionType.GRASP: f"A hand {verb} and picking up {target}.",
        HandActionType.PULL: f"A hand {verb} {target} toward the viewer.",
        HandActionType.PUSH: f"A hand {verb} {target} away.",
        HandActionType.ROTATE: f"A hand {verb} {target}.",
        HandActionType.LIFT: f"A hand {verb} {target} off the surface.",
        HandActionType.PLACE: f"A hand {verb} {target} down.",
        HandActionType.SLIDE: f"A hand {verb} {target} across the surface.",
    }

    return prompts.get(action_type, f"A hand {verb} {target}.")


def create_bundle_manifest(
    bundle: DWMConditioningBundle,
    output_dir: Path,
) -> BundleManifest:
    """
    Create a manifest for a DWM bundle.

    Args:
        bundle: DWM conditioning bundle
        output_dir: Bundle output directory

    Returns:
        BundleManifest with all metadata
    """
    manifest = BundleManifest(
        bundle_id=bundle.bundle_id,
        scene_id=bundle.scene_id,
        created_at=datetime.utcnow().isoformat() + "Z",
        num_frames=bundle.num_frames,
        fps=bundle.fps,
        resolution=bundle.resolution,
        action_type=(
            bundle.hand_trajectory.action_type.value
            if bundle.hand_trajectory else "unknown"
        ),
        target_object_id=(
            bundle.camera_trajectory.target_object_id
            if bundle.camera_trajectory else None
        ),
        text_prompt=bundle.text_prompt,
    )
    if bundle.metadata:
        manifest.hand_model = bundle.metadata.get("hand_model")
        manifest.hand_model_requested = bundle.metadata.get("hand_model_requested")

    # Set file paths if they exist
    if bundle.static_scene_video_path:
        manifest.static_scene_video = bundle.static_scene_video_path.name
    if bundle.hand_mesh_video_path:
        manifest.hand_mesh_video = bundle.hand_mesh_video_path.name
    if bundle.static_scene_frames_dir:
        manifest.static_scene_frames_dir = "frames/static_scene"
    if bundle.hand_mesh_frames_dir:
        manifest.hand_mesh_frames_dir = "frames/hand_mesh"
    if bundle.interaction_video_path:
        manifest.interaction_video = bundle.interaction_video_path.name
    if bundle.interaction_frames_dir:
        manifest.interaction_frames_dir = "frames/interaction"
    if bundle.hand_trajectory:
        manifest.hand_trajectory_file = "hand_trajectory.json"
        if bundle.hand_trajectory.robot_actions:
            manifest.robot_actions_file = "robot_actions.json"
    if bundle.static_scene_depth_dir:
        manifest.static_scene_depth_dir = "frames/static_scene_depth"
    if bundle.static_scene_seg_dir:
        manifest.static_scene_seg_dir = "frames/static_scene_seg"
    if bundle.scene_state_path:
        manifest.scene_state_file = "metadata/scene_state.json"
    if bundle.physics_log_path:
        manifest.physics_rollout_file = "metadata/physics_rollout.jsonl"

    return manifest


class DWMBundlePackager:
    """
    Packages DWM conditioning bundles.

    Takes generated trajectories and rendered outputs and packages
    them into a structured format for DWM inference.
    """

    def __init__(self, output_base_dir: Path):
        """
        Initialize bundle packager.

        Args:
            output_base_dir: Base directory for bundle outputs
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def package_bundle(
        self,
        bundle: DWMConditioningBundle,
        include_frames: bool = True,
        compress: bool = False,
    ) -> Path:
        """
        Package a single DWM conditioning bundle.

        Args:
            bundle: DWM conditioning bundle to package
            include_frames: Include individual frames
            compress: Create compressed archive

        Returns:
            Path to packaged bundle directory
        """
        bundle_dir = self.output_base_dir / bundle.bundle_id
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        frames_dir = bundle_dir / "frames"
        metadata_dir = bundle_dir / "metadata"
        frames_dir.mkdir(exist_ok=True)
        metadata_dir.mkdir(exist_ok=True)

        # Export camera trajectory
        if bundle.camera_trajectory:
            self._export_camera_trajectory(
                bundle.camera_trajectory,
                bundle_dir / "camera_trajectory.json",
            )

        # Export hand trajectory
        if bundle.hand_trajectory:
            self._export_hand_trajectory(
                bundle.hand_trajectory,
                bundle_dir / "hand_trajectory.json",
            )

        # Copy or link videos
        if bundle.static_scene_video_path and bundle.static_scene_video_path.exists():
            shutil.copy2(
                bundle.static_scene_video_path,
                bundle_dir / "static_scene_video.mp4",
            )

        if bundle.hand_mesh_video_path and bundle.hand_mesh_video_path.exists():
            shutil.copy2(
                bundle.hand_mesh_video_path,
                bundle_dir / "hand_mesh_video.mp4",
            )

        # Copy frames if requested
        if include_frames:
            if bundle.static_scene_frames_dir and bundle.static_scene_frames_dir.exists():
                shutil.copytree(
                    bundle.static_scene_frames_dir,
                    frames_dir / "static_scene",
                    dirs_exist_ok=True,
                )
            if bundle.static_scene_depth_dir and bundle.static_scene_depth_dir.exists():
                shutil.copytree(
                    bundle.static_scene_depth_dir,
                    frames_dir / "static_scene_depth",
                    dirs_exist_ok=True,
                )
            if bundle.static_scene_seg_dir and bundle.static_scene_seg_dir.exists():
                shutil.copytree(
                    bundle.static_scene_seg_dir,
                    frames_dir / "static_scene_seg",
                    dirs_exist_ok=True,
                )
        if bundle.hand_mesh_frames_dir and bundle.hand_mesh_frames_dir.exists():
            shutil.copytree(
                bundle.hand_mesh_frames_dir,
                frames_dir / "hand_mesh",
                dirs_exist_ok=True,
            )

        # Write text prompt
        (metadata_dir / "prompt.txt").write_text(bundle.text_prompt)

        scene_info = {
            "bundle_id": bundle.bundle_id,
            "scene_id": bundle.scene_id,
            "action_type": (
                bundle.hand_trajectory.action_type.value
                if bundle.hand_trajectory else "unknown"
            ),
            "fps": bundle.fps,
            "resolution": list(bundle.resolution),
            "num_frames": bundle.num_frames,
            "target_object_id": (
                bundle.camera_trajectory.target_object_id
                if bundle.camera_trajectory else None
            ),
            "metadata": bundle.metadata,
        }
        (metadata_dir / "scene_info.json").write_text(json.dumps(scene_info, indent=2))

        # Write bundle metadata
        (metadata_dir / "bundle_info.json").write_text(json.dumps(
            bundle.metadata,
            indent=2,
        ))

        # Robot fine-tune manifest scaffold
        self._write_robot_finetune_manifest(bundle, metadata_dir)
        if bundle.scene_state_path and bundle.scene_state_path.exists():
            shutil.copy2(bundle.scene_state_path, metadata_dir / "scene_state.json")
        if bundle.physics_log_path and bundle.physics_log_path.exists():
            shutil.copy2(bundle.physics_log_path, metadata_dir / "physics_rollout.jsonl")

        # Create and write manifest
        manifest = create_bundle_manifest(bundle, bundle_dir)
        (bundle_dir / "manifest.json").write_text(json.dumps(
            {
                "bundle_id": manifest.bundle_id,
                "scene_id": manifest.scene_id,
                "created_at": manifest.created_at,
                "num_frames": manifest.num_frames,
                "fps": manifest.fps,
                "resolution": list(manifest.resolution),
                "action_type": manifest.action_type,
                "target_object_id": manifest.target_object_id,
                "text_prompt": manifest.text_prompt,
                "hand_model": manifest.hand_model,
                "hand_model_requested": manifest.hand_model_requested,
                "static_scene_video": manifest.static_scene_video,
                "hand_mesh_video": manifest.hand_mesh_video,
                "interaction_video": manifest.interaction_video,
                "camera_trajectory_file": manifest.camera_trajectory_file,
                "hand_trajectory_file": manifest.hand_trajectory_file,
                "robot_actions_file": manifest.robot_actions_file,
                "static_scene_frames_dir": manifest.static_scene_frames_dir,
                "hand_mesh_frames_dir": manifest.hand_mesh_frames_dir,
                "interaction_frames_dir": manifest.interaction_frames_dir,
                "dwm_compatible": manifest.dwm_compatible,
                "dwm_version": manifest.dwm_version,
                "static_scene_depth_dir": manifest.static_scene_depth_dir,
                "static_scene_seg_dir": manifest.static_scene_seg_dir,
                "scene_state_file": manifest.scene_state_file,
                "physics_rollout_file": manifest.physics_rollout_file,
            },
            indent=2,
        ))

        # Compress if requested
        if compress:
            archive_path = self.output_base_dir / f"{bundle.bundle_id}.tar.gz"
            shutil.make_archive(
                str(archive_path.with_suffix("")),
                "gztar",
                bundle_dir.parent,
                bundle_dir.name,
            )
            return archive_path

        return bundle_dir

    def _export_camera_trajectory(
        self,
        trajectory: CameraTrajectory,
        output_path: Path,
    ) -> None:
        """Export camera trajectory to JSON."""
        import numpy as np

        data = {
            "trajectory_id": trajectory.trajectory_id,
            "trajectory_type": trajectory.trajectory_type.value,
            "num_frames": trajectory.num_frames,
            "fps": trajectory.fps,
            "duration_seconds": trajectory.duration,
            "resolution": list(trajectory.resolution),
            "focal_length": trajectory.focal_length,
            "target_object_id": trajectory.target_object_id,
            "action_type": (
                trajectory.action_type.value
                if trajectory.action_type else None
            ),
            "description": trajectory.description,
            "poses": [],
        }

        for pose in trajectory.poses:
            data["poses"].append({
                "frame_idx": pose.frame_idx,
                "timestamp": pose.timestamp,
                "transform": pose.transform.tolist(),
                "position": pose.position.tolist(),
            })

        output_path.write_text(json.dumps(data, indent=2))

    def _export_hand_trajectory(
        self,
        trajectory: HandTrajectory,
        output_path: Path,
    ) -> None:
        """Export hand trajectory to JSON."""
        data = {
            "trajectory_id": trajectory.trajectory_id,
            "action_type": trajectory.action_type.value,
            "num_frames": trajectory.num_frames,
            "fps": trajectory.fps,
            "description": trajectory.description,
            "target_object_id": trajectory.target_object_id,
            "camera_trajectory_id": trajectory.camera_trajectory_id,
            "poses": [],
            "robot_actions_file": "robot_actions.json"
            if trajectory.robot_actions
            else None,
        }

        for pose in trajectory.poses:
            pose_data = {
                "frame_idx": pose.frame_idx,
                "timestamp": pose.timestamp,
                "hand_side": pose.hand_side,
                "position": pose.position.tolist(),
                "rotation": pose.rotation.tolist(),
                "contact_fingertips": pose.contact_fingertips,
            }

            if pose.pose_params is not None:
                pose_data["pose_params"] = pose.pose_params.tolist()
            if pose.shape_params is not None:
                pose_data["shape_params"] = pose.shape_params.tolist()

            data["poses"].append(pose_data)

        output_path.write_text(json.dumps(data, indent=2))

        if trajectory.robot_actions:
            self._export_robot_actions(
                trajectory.robot_actions,
                output_path.parent / "robot_actions.json",
            )

    def _export_robot_actions(
        self,
        actions: list,
        output_path: Path,
    ) -> None:
        """Export robot actions aligned with the hand trajectory."""
        data = {
            "actions": {
                str(action.frame_idx): action.to_json()
                for action in actions
            }
        }
        output_path.write_text(json.dumps(data, indent=2))

    def package_all(
        self,
        bundles: list[DWMConditioningBundle],
        scene_id: str,
        include_frames: bool = True,
    ) -> DWMPipelineOutput:
        """
        Package all bundles for a scene.

        Args:
            bundles: List of bundles to package
            scene_id: Scene identifier
            include_frames: Include individual frames

        Returns:
            DWMPipelineOutput with all packaged bundles
        """
        import time
        start_time = time.time()

        packaged_bundles = []
        errors = []

        for bundle in bundles:
            try:
                bundle_path = self.package_bundle(
                    bundle,
                    include_frames=include_frames,
                )
                packaged_bundles.append(bundle)
            except Exception as e:
                errors.append(f"Failed to package {bundle.bundle_id}: {e}")

        # Write overall manifest
        manifest_path = self.output_base_dir / "dwm_bundles_manifest.json"
        manifest_data = {
            "scene_id": scene_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "num_bundles": len(packaged_bundles),
            "bundles": [
                {
                    "bundle_id": b.bundle_id,
                    "action_type": (
                        b.hand_trajectory.action_type.value
                        if b.hand_trajectory else "unknown"
                    ),
                    "target_object_id": (
                        b.camera_trajectory.target_object_id
                        if b.camera_trajectory else None
                    ),
                    "text_prompt": b.text_prompt,
                    "hand_model": (b.metadata or {}).get("hand_model"),
                    "hand_model_requested": (b.metadata or {}).get("hand_model_requested"),
                    "physics_rollout_file": "metadata/physics_rollout.jsonl"
                    if b.physics_log_path else None,
                    "interaction_video": (
                        b.interaction_video_path.name
                        if b.interaction_video_path else None
                    ),
                    "interaction_frames_dir": (
                        "frames/interaction"
                        if b.interaction_frames_dir else None
                    ),
                }
                for b in packaged_bundles
            ],
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2))

        return DWMPipelineOutput(
            scene_id=scene_id,
            bundles=packaged_bundles,
            output_dir=self.output_base_dir,
            num_trajectories=len(packaged_bundles),
            num_bundles=len(packaged_bundles),
            total_frames=sum(b.num_frames for b in packaged_bundles),
            manifest_path=manifest_path,
            errors=errors,
            generation_time_seconds=time.time() - start_time,
        )

    def _write_robot_finetune_manifest(
        self,
        bundle: DWMConditioningBundle,
        metadata_dir: Path,
    ) -> None:
        """Write scaffold manifest pointing to real-robot fine-tune assets."""
        robot_info = (bundle.metadata or {}).get("robot_retargeting", {}) if bundle.metadata else {}
        manifest = {
            "robot_model": robot_info.get("robot_model"),
            "real_robot_demos": robot_info.get("demo_roots", []),
            "notes": (
                "Add references to real robot demonstration sequences aligned with "
                "robot_actions.json for fine-tuning."
            ),
        }
        (metadata_dir / "robot_finetune_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )


def package_dwm_bundle(
    camera_trajectory: CameraTrajectory,
    hand_trajectory: HandTrajectory,
    scene_id: str,
    output_dir: Path,
    static_scene_video_path: Optional[Path] = None,
    hand_mesh_video_path: Optional[Path] = None,
    static_scene_frames_dir: Optional[Path] = None,
    hand_mesh_frames_dir: Optional[Path] = None,
    static_scene_depth_dir: Optional[Path] = None,
    static_scene_seg_dir: Optional[Path] = None,
    scene_state_path: Optional[Path] = None,
    physics_log_path: Optional[Path] = None,
    target_category: Optional[str] = None,
    target_description: Optional[str] = None,
) -> Path:
    """
    Convenience function to package a single DWM bundle.

    Args:
        camera_trajectory: Camera trajectory
        hand_trajectory: Hand trajectory
        scene_id: Scene identifier
        output_dir: Output directory
        static_scene_video_path: Path to static scene video
        hand_mesh_video_path: Path to hand mesh video
        static_scene_frames_dir: Path to static scene frames
        hand_mesh_frames_dir: Path to hand mesh frames
        static_scene_depth_dir: Path to static scene depth frames
        static_scene_seg_dir: Path to static scene segmentation frames
        scene_state_path: Path to per-frame scene state JSON
        target_category: Category of target object
        target_description: Description of target

    Returns:
        Path to packaged bundle
    """
    # Generate text prompt
    text_prompt = generate_text_prompt(
        hand_trajectory.action_type,
        target_category,
        target_description,
    )

    # Create bundle
    bundle = DWMConditioningBundle(
        bundle_id=f"{scene_id}_{camera_trajectory.trajectory_id}",
        scene_id=scene_id,
        camera_trajectory=camera_trajectory,
        hand_trajectory=hand_trajectory,
        static_scene_video_path=static_scene_video_path,
        hand_mesh_video_path=hand_mesh_video_path,
        static_scene_frames_dir=static_scene_frames_dir,
        hand_mesh_frames_dir=hand_mesh_frames_dir,
        static_scene_depth_dir=static_scene_depth_dir,
        static_scene_seg_dir=static_scene_seg_dir,
        scene_state_path=scene_state_path,
        physics_log_path=physics_log_path,
        text_prompt=text_prompt,
        resolution=camera_trajectory.resolution,
        num_frames=camera_trajectory.num_frames,
        fps=camera_trajectory.fps,
        metadata={
            "camera_trajectory_id": camera_trajectory.trajectory_id,
            "hand_trajectory_id": hand_trajectory.trajectory_id,
            "action_type": hand_trajectory.action_type.value,
        },
    )

    # Link hand trajectory to camera trajectory
    hand_trajectory.camera_trajectory_id = camera_trajectory.trajectory_id

    # Package
    packager = DWMBundlePackager(output_dir)
    return packager.package_bundle(bundle)
