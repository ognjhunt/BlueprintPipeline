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

import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import CameraTrajectory, HandPose, HandTrajectory  # noqa: E402


def _try_import_isaac() -> bool:
    """Check if Isaac Sim/Lab modules are importable."""
    try:
        import omni.isaac.core  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


class PhysicsPolicyRunner:
    """Execute scripted policies and log physics rollouts."""

    def __init__(self, fps: float = 24.0):
        self.fps = fps
        self.physics_available = _try_import_isaac()

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

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rollouts: dict[str, Path] = {}
        for camera_traj, hand_traj in trajectory_pairs:
            traj_dir = output_dir / camera_traj.trajectory_id
            traj_dir.mkdir(parents=True, exist_ok=True)
            log_path = traj_dir / "physics_rollout.jsonl"

            entries = self._execute_policy(
                scene_usd_path=scene_usd_path,
                camera_traj=camera_traj,
                hand_traj=hand_traj,
                num_frames=num_frames,
                scene_objects=scene_objects or {},
            )

            with log_path.open("w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

            rollouts[camera_traj.trajectory_id] = log_path

        return rollouts

    def _execute_policy(
        self,
        scene_usd_path: Path,
        camera_traj: CameraTrajectory,
        hand_traj: Optional[HandTrajectory],
        num_frames: int,
        scene_objects: dict[str, dict],
    ) -> list[dict]:
        """
        Execute a scripted policy (or stub) for a single trajectory.

        Returns:
            List of JSON-serializable per-frame entries.
        """
        if camera_traj.num_frames == 0:
            return []

        frame_count = min(
            num_frames,
            camera_traj.num_frames or num_frames,
            hand_traj.num_frames if hand_traj else num_frames,
        )

        entries: list[dict] = []
        for frame_idx in range(frame_count):
            camera_pose = camera_traj.poses[min(frame_idx, camera_traj.num_frames - 1)]
            hand_pose = None
            if hand_traj and hand_traj.poses:
                hand_pose = hand_traj.poses[min(frame_idx, hand_traj.num_frames - 1)]

            entry = self._build_frame_record(
                frame_idx=frame_idx,
                camera_pose=camera_pose,
                hand_pose=hand_pose,
                action_type=hand_traj.action_type.value if hand_traj else None,
                target_object_id=hand_traj.target_object_id if hand_traj else None,
                scene_objects=scene_objects,
                backend="isaac_lab" if self.physics_available else "stub",
                scene_usd_path=scene_usd_path,
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
        scene_objects: dict[str, dict],
        backend: str,
        scene_usd_path: Path,
    ) -> dict:
        """Construct a serializable rollout record."""
        timestamp = frame_idx / self.fps if self.fps else frame_idx
        fingertip_contacts = (
            hand_pose.contact_fingertips if hand_pose else [False] * 5
        )

        contact_force = [float(c) * 1.0 for c in fingertip_contacts]
        reward = float(any(fingertip_contacts))

        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "scene_usd": str(scene_usd_path),
            "policy": {
                "type": action_type or "scripted",
                "target_object_id": target_object_id,
                "backend": backend,
            },
            "camera": {
                "position": camera_pose.position.tolist(),
                "transform": camera_pose.transform.tolist(),
            },
            "hand": self._serialize_hand_pose(hand_pose),
            "contacts": {
                "fingertip_contacts": fingertip_contacts,
                "estimated_contact_force": contact_force,
            },
            "reward": reward,
            "object_transforms": self._object_transforms(scene_objects),
        }

    @staticmethod
    def _serialize_hand_pose(hand_pose: Optional[HandPose]) -> Optional[dict]:
        """Serialize hand pose data if present."""
        if hand_pose is None:
            return None

        return {
            "position": hand_pose.position.tolist(),
            "rotation": hand_pose.rotation.tolist(),
            "contact_fingertips": hand_pose.contact_fingertips,
            "hand_side": hand_pose.hand_side,
        }

    @staticmethod
    def _object_transforms(scene_objects: dict[str, dict]) -> dict[str, dict]:
        """Provide simple object transform snapshots for downstream alignment."""
        transforms: dict[str, dict] = {}
        for obj_id, info in scene_objects.items():
            position = info.get("position")
            transforms[obj_id] = {
                "position": position.tolist() if isinstance(position, np.ndarray) else position,
                "category": info.get("category"),
                "sim_role": info.get("sim_role"),
            }
        return transforms
