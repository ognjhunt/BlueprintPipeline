"""
Lightweight Isaac Sim/Lab policy runner for physics rollouts.

This module provides a thin abstraction to execute scripted policies aligned
with generated camera/hand trajectories and to log per-frame states, actions,
rewards, and contacts for downstream pairing with rendered frames.

When Isaac Sim/Lab is unavailable in the runtime environment, the runner
produces deterministic stub rollouts so the rest of the pipeline continues to
function (useful for CI and offline generation).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import CameraTrajectory, HandPose, HandTrajectory, RobotAction  # noqa: E402


def _isaac_available() -> bool:
    """Check if required Isaac Sim/Lab modules are importable."""

    required_modules = ("omni.isaac.lab", "isaacsim.core.api")
    return all(importlib.util.find_spec(module_name) for module_name in required_modules)


def _require_isaac() -> None:
    """Raise a clear error when Isaac is unavailable."""

    if not _isaac_available():
        raise RuntimeError(
            "Isaac Sim/Lab modules are not available. Enable Isaac in this environment or disable "
            "`use_physics_ground_truth` to continue without physics rollouts."
        )


@dataclass
class _RobotActionIndex:
    """Convenience wrapper to align robot actions to frame indices."""

    by_index: dict[int, RobotAction]

    @classmethod
    def from_actions(cls, actions: list[RobotAction]) -> "_RobotActionIndex":
        indexed: dict[int, RobotAction] = {}
        for action in actions:
            indexed[action.frame_idx] = action

        # Fallback alignment by ordinal if frame_idx isn't populated
        if not indexed and actions:
            indexed = {idx: action for idx, action in enumerate(actions)}

        return cls(by_index=indexed)

    def get(self, frame_idx: int) -> Optional[RobotAction]:
        return self.by_index.get(frame_idx)


class IsaacLabRolloutDriver:
    """Minimal Isaac Lab-driven rollout logger.

    Opens the USD scene, replays camera/hand trajectories, and emits structured
    per-frame state/action records for downstream policy training.
    """

    def __init__(self, scene_usd_path: Path, fps: float, scene_objects: Optional[dict[str, dict]] = None):
        _require_isaac()

        from pxr import Usd, UsdGeom  # type: ignore

        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self.scene_usd_path = scene_usd_path
        self.fps = fps
        self.scene_objects = scene_objects or {}

        self.stage = Usd.Stage.Open(str(scene_usd_path))
        if self.stage is None:
            raise RuntimeError(f"Failed to open USD stage at {scene_usd_path}")

    def replay(
        self,
        camera_traj: CameraTrajectory,
        hand_traj: Optional[HandTrajectory],
        num_frames: int,
        expect_robot_actions: bool,
    ) -> list[dict]:
        if camera_traj.num_frames == 0:
            return []

        robot_actions = hand_traj.robot_actions if hand_traj else []
        if expect_robot_actions and not robot_actions:
            raise ValueError(
                "Robot retargeting is enabled but no robot_actions were attached to the hand trajectory."
            )

        frame_count = min(
            num_frames,
            camera_traj.num_frames or num_frames,
            hand_traj.num_frames if hand_traj else num_frames,
            len(robot_actions) if expect_robot_actions and robot_actions else num_frames,
        )

        action_index = _RobotActionIndex.from_actions(robot_actions)
        entries: list[dict] = []

        for frame_idx in range(frame_count):
            camera_pose = camera_traj.poses[min(frame_idx, camera_traj.num_frames - 1)]
            hand_pose = None
            if hand_traj and hand_traj.poses:
                hand_pose = hand_traj.poses[min(frame_idx, hand_traj.num_frames - 1)]

            action = action_index.get(frame_idx)

            entry = self._build_frame_record(
                frame_idx=frame_idx,
                camera_pose=camera_pose,
                hand_pose=hand_pose,
                action_type=hand_traj.action_type.value if hand_traj else None,
                target_object_id=hand_traj.target_object_id if hand_traj else None,
                robot_action=action,
                expect_robot_actions=expect_robot_actions,
            )
            entries.append(entry)

        return entries

    def _build_frame_record(
        self,
        frame_idx: int,
        camera_pose,
        hand_pose: Optional[HandPose],
        action_type: Optional[str],
        target_object_id: Optional[str],
        robot_action: Optional[RobotAction],
        expect_robot_actions: bool,
    ) -> dict:
        timestamp = frame_idx / self.fps if self.fps else frame_idx
        fingertip_contacts = hand_pose.contact_fingertips if hand_pose else [False] * 5

        entry = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "scene_usd": str(self.scene_usd_path),
            "policy": {
                "type": action_type or "scripted",
                "target_object_id": target_object_id,
                "backend": "isaac_lab",
            },
            "camera": {
                "position": camera_pose.position.tolist(),
                "transform": camera_pose.transform.tolist(),
            },
            "hand": self._serialize_hand_pose(hand_pose),
            "robot_action": self._serialize_robot_action(robot_action),
            "contacts": {
                "fingertip_contacts": fingertip_contacts,
                "estimated_contact_force": [float(c) for c in fingertip_contacts],
            },
            "articulated_joints": self._serialize_joint_state(robot_action, expect_robot_actions),
            "object_poses": self._object_transforms(),
            "reward": float(any(fingertip_contacts)),
        }

        # Maintain backward compatibility with downstream consumers expecting the old key
        entry["object_transforms"] = entry["object_poses"]

        return entry

    def _object_transforms(self) -> dict[str, dict]:
        """Capture object poses from the USD stage (fallbacks to manifest positions)."""

        transforms: dict[str, dict] = {}

        for obj_id, info in self.scene_objects.items():
            prim_path = self._resolve_prim_path(info)
            prim = self.stage.GetPrimAtPath(prim_path) if prim_path else None
            pose_matrix = None
            if prim and prim.IsValid() and self._UsdGeom.Xformable(prim):
                xformable = self._UsdGeom.Xformable(prim)
                pose_matrix, _ = xformable.GetLocalTransformation()

            position = None
            orientation = None
            if pose_matrix:
                translation = pose_matrix.ExtractTranslation()
                rotation = pose_matrix.ExtractRotationQuat()
                position = [translation[0], translation[1], translation[2]]
                orientation = [rotation.GetReal(), *rotation.GetImaginary()]
            elif isinstance(info.get("position"), np.ndarray):
                position = info["position"].tolist()

            transforms[obj_id] = {
                "prim_path": prim_path,
                "position": position,
                "orientation": orientation,
                "category": info.get("category"),
                "sim_role": info.get("sim_role"),
                "source": "usd_stage" if pose_matrix else "scene_manifest",
            }

        return transforms

    @staticmethod
    def _resolve_prim_path(info: dict) -> Optional[str]:
        for key in ("usd_path", "prim_path", "stage_path", "path"):
            if key in info:
                return info[key]
        return None

    @staticmethod
    def _serialize_hand_pose(hand_pose: Optional[HandPose]) -> Optional[dict]:
        if hand_pose is None:
            return None

        return {
            "position": hand_pose.position.tolist(),
            "rotation": hand_pose.rotation.tolist(),
            "contact_fingertips": hand_pose.contact_fingertips,
            "hand_side": hand_pose.hand_side,
        }

    @staticmethod
    def _serialize_robot_action(robot_action: Optional[RobotAction]) -> Optional[dict]:
        return robot_action.to_json() if robot_action else None

    @staticmethod
    def _serialize_joint_state(
        robot_action: Optional[RobotAction], expect_robot_actions: bool
    ) -> dict:
        if not robot_action:
            return {
                "joint_names": [],
                "positions": [],
                "base_frame": None,
                "end_effector_frame": None,
                "source": "retargeted_actions" if expect_robot_actions else "unavailable",
            }

        return {
            "joint_names": robot_action.joint_names,
            "positions": robot_action.joint_positions,
            "base_frame": robot_action.base_frame,
            "end_effector_frame": robot_action.end_effector_frame,
            "source": "retargeted_actions",
        }


class PhysicsPolicyRunner:
    """Execute scripted policies and log physics rollouts."""

    def __init__(self, fps: float = 24.0, enable_robot_retargeting: bool = False):
        self.fps = fps
        self.enable_robot_retargeting = enable_robot_retargeting
        self.physics_available = _isaac_available()

    def run_rollouts(
        self,
        scene_usd_path: Path,
        trajectory_pairs: Iterable[tuple[CameraTrajectory, Optional[HandTrajectory]]],
        output_dir: Path,
        num_frames: int,
        scene_objects: Optional[dict[str, dict]] = None,
    ) -> dict[str, Path]:
        """
        Run rollouts for each trajectory pair.

        Args:
            scene_usd_path: Path to the USD scene file.
            trajectory_pairs: Iterable of (camera, hand) trajectories.
            output_dir: Base directory for rollout logs.
            num_frames: Maximum frames to log per trajectory.
            scene_objects: Optional scene object metadata (id -> info dict).

        Returns:
            Mapping of trajectory_id to the generated physics_rollout.jsonl path.
        """
        scene_usd_path = Path(scene_usd_path)
        if not scene_usd_path.exists():
            raise FileNotFoundError(f"Scene USD not found: {scene_usd_path}")

        if not self.physics_available:
            raise RuntimeError(
                "Isaac Sim/Lab is unavailable, so physics rollouts cannot be generated. "
                "Disable physics ground truth or install Isaac to continue."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        driver = IsaacLabRolloutDriver(scene_usd_path, fps=self.fps, scene_objects=scene_objects)

        rollouts: dict[str, Path] = {}
        for camera_traj, hand_traj in trajectory_pairs:
            traj_dir = output_dir / camera_traj.trajectory_id
            traj_dir.mkdir(parents=True, exist_ok=True)
            log_path = traj_dir / "physics_rollout.jsonl"

            entries = driver.replay(
                camera_traj=camera_traj,
                hand_traj=hand_traj,
                num_frames=num_frames,
                expect_robot_actions=self.enable_robot_retargeting,
            )

            with log_path.open("w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

            rollouts[camera_traj.trajectory_id] = log_path

        return rollouts
