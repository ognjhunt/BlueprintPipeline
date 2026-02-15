#!/usr/bin/env python3
"""
Bridge SAGE HDF5 demo output → LeRobot v2.0 format.

Reads dataset.hdf5 produced by SAGE stages 6-7, converts each demo into
JointTrajectory objects, and feeds them into the existing LeRobotExporter
with 80/10/10 train/val/test splits.

Usage:
    python sage_to_lerobot.py \
        --hdf5 /workspace/SAGE/server/results/layout_XYZ/demos/dataset.hdf5 \
        --output /workspace/outputs/kitchen_lerobot \
        --dataset-name sage-franka-kitchen-v1 \
        --room-type kitchen \
        --task "Pick up the mug from the counter and place it on the dining table"
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import h5py
import numpy as np

# Add project roots to path
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "episode-generation-job"))

from trajectory_solver import JointTrajectory, JointState, RobotConfig, FRANKA_CONFIG
from lerobot_exporter import LeRobotExporter, LeRobotDatasetConfig


def load_sage_hdf5(hdf5_path: Path) -> list:
    """Load demos from a SAGE HDF5 file and return list of (trajectory, task_desc)."""
    trajectories = []

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            print(f"WARNING: No 'data' group in {hdf5_path}")
            return trajectories

        data = f["data"]
        for demo_key in sorted(data.keys()):
            demo = data[demo_key]

            # Extract joint states
            if "states" in demo:
                positions = np.array(demo["states"])
            elif "obs" in demo and "joint_positions" in demo["obs"]:
                positions = np.array(demo["obs"]["joint_positions"])
            else:
                print(f"  Skipping {demo_key}: no joint position data found")
                continue

            # Extract actions (if available)
            actions = None
            if "actions" in demo:
                actions = np.array(demo["actions"])

            # Extract task description from metadata
            task_desc = "manipulation task"
            if demo.attrs.get("task_description"):
                task_desc = demo.attrs["task_description"]
                if isinstance(task_desc, bytes):
                    task_desc = task_desc.decode("utf-8")

            num_frames = positions.shape[0]
            num_joints = min(positions.shape[1], 7) if positions.ndim > 1 else 7
            fps = 30.0

            # Build JointState list
            states = []
            for frame_idx in range(num_frames):
                pos = positions[frame_idx, :num_joints] if positions.ndim > 1 else positions[frame_idx]
                state = JointState(
                    frame_idx=frame_idx,
                    timestamp=frame_idx / fps,
                    joint_positions=np.array(pos, dtype=np.float32),
                )
                states.append(state)

            trajectory = JointTrajectory(
                trajectory_id=f"sage_{demo_key}_{uuid.uuid4().hex[:8]}",
                robot_type="franka",
                robot_config=FRANKA_CONFIG,
                states=states,
                fps=fps,
                total_duration=num_frames / fps,
                provenance={
                    "action_source": "sage_curobo_rrt",
                    "velocity_source": "finite_difference",
                    "scene_source": "sage",
                },
            )

            trajectories.append((trajectory, task_desc))
            print(f"  Loaded {demo_key}: {num_frames} frames, {num_joints} joints")

    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Convert SAGE HDF5 demos to LeRobot format")
    parser.add_argument("--hdf5", required=True, help="Path to SAGE dataset.hdf5")
    parser.add_argument("--output", required=True, help="Output directory for LeRobot dataset")
    parser.add_argument("--dataset-name", default="sage-franka-v1", help="Dataset name")
    parser.add_argument("--room-type", default="kitchen", help="Room type for metadata")
    parser.add_argument("--task", default=None, help="Override task description for all episodes")
    parser.add_argument("--robot-type", default="franka", help="Robot type")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    parser.add_argument("--include-images", action="store_true", help="Include RGB images if available")
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5)
    output_dir = Path(args.output)

    if not hdf5_path.exists():
        print(f"ERROR: HDF5 file not found: {hdf5_path}")
        sys.exit(1)

    print(f"Loading SAGE demos from {hdf5_path}...")
    trajectories = load_sage_hdf5(hdf5_path)

    if not trajectories:
        print("ERROR: No trajectories loaded.")
        sys.exit(1)

    # Override task description if provided
    if args.task:
        trajectories = [(traj, args.task) for traj, _ in trajectories]

    print(f"Loaded {len(trajectories)} trajectories. Exporting to LeRobot format...")

    config = LeRobotDatasetConfig(
        dataset_name=args.dataset_name,
        robot_type=args.robot_type,
        fps=args.fps,
        state_dim=7,
        action_dim=8,
        include_images=args.include_images,
        data_pack_tier="core",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    exporter = LeRobotExporter(config)
    for trajectory, task_desc in trajectories:
        exporter.add_episode(trajectory, task_desc)

    result = exporter.finalize(output_dir)

    # Write additional metadata
    meta = {
        "pipeline": "sage",
        "room_type": args.room_type,
        "robot_type": args.robot_type,
        "num_episodes": len(trajectories),
        "fps": args.fps,
        "dataset_name": args.dataset_name,
        "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
    }
    meta_path = output_dir / "sage_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Export complete: {output_dir}")
    print(f"  Episodes: {len(trajectories)}")
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
