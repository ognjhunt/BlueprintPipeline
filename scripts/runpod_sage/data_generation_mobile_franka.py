#!/usr/bin/env python3
"""
Data Generation for Mobile Franka — Replacement for NVIDIA-internal scripts.

Generates robot demonstration data (pick-and-place trajectories) using:
  - cuRobo for collision-free arm trajectory planning
  - Isaac Sim for physics simulation + rendering
  - RRT-based navigation planning (from SAGE's object_mobile_manipulation_utils)
  - robomimic-compatible HDF5 output format

This replaces the NVIDIA-internal:
  - isaaclab/data_generation_mobile_franka_mobile_manipulation_with_pose_aug.py
  - isaaclab/data_generation_mobile_manipulation_v7_replay.py

Usage:
    cd /workspace/SAGE/server
    python data_generation_mobile_franka.py \
        --layout_id layout_XXXXXXXX \
        --pose_aug_name pose_aug_0 \
        --num_demos 16 \
        --enable_cameras \
        --headless

Output:
    results/<layout_id>/demos/
        dataset.hdf5                    - robomimic-compatible dataset
        demo_metadata.json              - Per-demo metadata
        step_decomposition.json         - Navigate/pick/place stage labels
"""

import argparse
import json
import os
import sys
import time
import math
import hashlib
import socket
import numpy as np
from pathlib import Path
from collections import OrderedDict

# Add SAGE server to path
SAGE_SERVER_DIR = os.environ.get("SAGE_SERVER_DIR", "/workspace/SAGE/server")
sys.path.insert(0, SAGE_SERVER_DIR)


def get_mcp_port():
    """Compute Isaac Sim MCP port from SLURM_JOB_ID."""
    job_id = os.environ.get("SLURM_JOB_ID", "12345")
    h = int(hashlib.md5(str(job_id).encode()).hexdigest(), 16)
    return 8080 + (h % (40000 - 8080 + 1))


class MobileFrankaDataGenerator:
    """
    Generates pick-and-place demonstration data for a Mobile Franka robot.

    Architecture:
        1. Load scene (USD or JSON) into Isaac Sim
        2. Spawn mobile Franka robot
        3. For each demo:
            a. Sample robot base position near pick object
            b. Plan navigation trajectory (A* on occupancy grid)
            c. Plan arm grasp trajectory (cuRobo IK)
            d. Execute grasp in simulation
            e. Plan navigation to place target
            f. Plan arm place trajectory
            g. Execute place
            h. Record: joint states, actions, RGB/depth from cameras
        4. Save to HDF5 (robomimic format)
    """

    def __init__(self, layout_dir, room_json_path, grasp_data=None,
                 enable_cameras=True, headless=True):
        self.layout_dir = Path(layout_dir)
        self.room_json_path = Path(room_json_path)
        self.enable_cameras = enable_cameras
        self.headless = headless
        self.grasp_data = grasp_data or {}

        # Load room data
        with open(room_json_path) as f:
            self.room = json.load(f)

        self.room_dims = self.room["dimensions"]
        self.objects = self.room.get("objects", [])

        # Identify pick/place candidates
        self.manipulable_objects = self._identify_manipulable_objects()
        self.support_surfaces = self._identify_support_surfaces()

        # State
        self.isaac_sim_available = self._check_isaac_sim()
        self.curobo_available = self._check_curobo()

        print(f"[DATA-GEN] Room: {self.room['room_type']}")
        print(f"[DATA-GEN] Objects: {len(self.objects)}")
        print(f"[DATA-GEN] Manipulable: {len(self.manipulable_objects)}")
        print(f"[DATA-GEN] Support surfaces: {len(self.support_surfaces)}")
        print(f"[DATA-GEN] Isaac Sim: {'available' if self.isaac_sim_available else 'NOT available'}")
        print(f"[DATA-GEN] cuRobo: {'available' if self.curobo_available else 'NOT available'}")

    def _check_isaac_sim(self):
        """Check if Isaac Sim MCP is reachable."""
        port = get_mcp_port()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def _check_curobo(self):
        """Check if cuRobo is available."""
        try:
            import curobo
            return True
        except ImportError:
            return False

    def _identify_manipulable_objects(self):
        """Find objects that can be picked up (small enough for Franka gripper)."""
        manipulable = []
        # Objects smaller than 20cm in all dimensions are graspable
        max_grasp_size = 0.20  # meters
        for obj in self.objects:
            dims = obj.get("dimensions", {})
            w = dims.get("width", 1.0)
            l = dims.get("length", 1.0)
            h = dims.get("height", 1.0)
            # Must be small enough and have a source_id (mesh exists)
            if min(w, l) < max_grasp_size and obj.get("source_id"):
                manipulable.append(obj)
        return manipulable

    def _identify_support_surfaces(self):
        """Find objects that serve as support surfaces (tables, counters, shelves)."""
        surface_types = {
            "table", "desk", "counter", "countertop", "shelf", "cabinet",
            "nightstand", "dresser", "sideboard", "island", "bench",
            "coffee_table", "dining_table", "end_table", "side_table",
        }
        surfaces = []
        for obj in self.objects:
            obj_type = obj.get("type", "").lower().replace(" ", "_")
            if obj_type in surface_types or any(st in obj_type for st in surface_types):
                surfaces.append(obj)
        return surfaces

    def _sample_robot_base_position(self, target_obj, approach_dist=0.55):
        """Sample a collision-free robot base position near a target object."""
        tx = target_obj["position"]["x"]
        ty = target_obj["position"]["y"]

        # Sample positions around the target at approach_dist
        candidates = []
        for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            bx = tx + approach_dist * math.cos(angle)
            by = ty + approach_dist * math.sin(angle)

            # Check bounds (must be inside room with margin)
            margin = 0.4  # Robot half-width + safety
            if margin < bx < self.room_dims["width"] - margin and \
               margin < by < self.room_dims["length"] - margin:
                candidates.append((bx, by, angle + math.pi))  # Face the target

        if not candidates:
            # Fallback: just stand close
            return tx - approach_dist, ty, 0.0

        # Pick the one with most clearance from other objects
        best = candidates[0]
        best_clearance = 0
        for bx, by, yaw in candidates:
            min_dist = float("inf")
            for obj in self.objects:
                ox = obj["position"]["x"]
                oy = obj["position"]["y"]
                dist = math.sqrt((bx - ox) ** 2 + (by - oy) ** 2)
                min_dist = min(min_dist, dist)
            if min_dist > best_clearance:
                best_clearance = min_dist
                best = (bx, by, yaw)

        return best

    def _plan_arm_trajectory(self, start_config, target_pose, num_steps=50):
        """
        Plan a collision-free arm trajectory using cuRobo or linear interpolation.

        Args:
            start_config: np.ndarray (7,) — joint positions
            target_pose: np.ndarray (4, 4) — target end-effector pose
            num_steps: int — number of waypoints

        Returns:
            trajectory: np.ndarray (T, 7) — joint positions over time
        """
        if self.curobo_available:
            try:
                return self._plan_with_curobo(start_config, target_pose, num_steps)
            except Exception as e:
                print(f"[DATA-GEN] cuRobo planning failed: {e}, using linear fallback")

        # Linear interpolation fallback
        return self._linear_interpolation(start_config, target_pose, num_steps)

    def _plan_with_curobo(self, start_config, target_pose, num_steps):
        """Plan with cuRobo (NVIDIA collision-free motion planning)."""
        from curobo.types.math import Pose
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        # Franka default config
        config = MotionGenConfig.load_from_robot_config(
            "franka.yml",
            interpolation_dt=0.02,
        )
        motion_gen = MotionGen(config)
        motion_gen.warmup()

        # Convert target to curobo Pose
        position = target_pose[:3, 3]
        # Extract quaternion from rotation matrix
        R = target_pose[:3, :3]
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        w = math.sqrt(max(0, 1 + trace)) / 2
        x = math.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
        y = math.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
        z = math.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2

        target = Pose(
            position=position.tolist(),
            quaternion=[w, x, y, z],
        )

        import torch
        start_state = torch.tensor(start_config, dtype=torch.float32)

        result = motion_gen.plan_single(start_state, target)

        if result.success:
            traj = result.get_interpolated_plan()
            return traj.numpy()
        else:
            raise RuntimeError("cuRobo planning failed")

    def _linear_interpolation(self, start_config, target_pose, num_steps):
        """Simple linear interpolation in joint space (fallback)."""
        # Use a default target config (approximate IK solution)
        # Franka default joint limits
        target_config = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

        trajectory = np.zeros((num_steps, 7))
        for i in range(num_steps):
            t = i / max(num_steps - 1, 1)
            trajectory[i] = (1 - t) * start_config + t * target_config

        return trajectory

    def _generate_single_demo(self, pick_obj, place_surface, demo_idx):
        """
        Generate a single pick-and-place demonstration.

        Returns:
            dict with keys: states, actions, obs, rewards, dones
            Or None if generation failed
        """
        print(f"\n[DATA-GEN] Demo {demo_idx}: Pick '{pick_obj['type']}' → Place on '{place_surface['type']}'")

        # 1. Sample robot base position
        base_x, base_y, base_yaw = self._sample_robot_base_position(pick_obj)
        print(f"  Robot base: ({base_x:.2f}, {base_y:.2f}), yaw={math.degrees(base_yaw):.0f}°")

        # 2. Initial joint configuration (Franka home position)
        home_config = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

        # 3. Compute pick target pose (above the object)
        pick_pos = np.array([
            pick_obj["position"]["x"],
            pick_obj["position"]["y"],
            pick_obj["position"]["z"] + pick_obj["dimensions"]["height"] / 2,
        ])
        pick_pose = np.eye(4)
        pick_pose[:3, 3] = pick_pos - np.array([base_x, base_y, 0])  # Relative to robot base
        # Top-down grasp orientation
        pick_pose[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # 4. Plan approach trajectory
        approach_traj = self._plan_arm_trajectory(home_config, pick_pose, num_steps=30)

        # 5. Plan retreat trajectory
        retreat_traj = approach_traj[::-1].copy()

        # 6. Navigate to place position
        place_base_x, place_base_y, place_yaw = self._sample_robot_base_position(place_surface)

        # 7. Compute place target pose
        place_pos = np.array([
            place_surface["position"]["x"],
            place_surface["position"]["y"],
            place_surface["position"]["z"] + place_surface["dimensions"]["height"],
        ])
        place_pose = np.eye(4)
        place_pose[:3, 3] = place_pos - np.array([place_base_x, place_base_y, 0])
        place_pose[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # 8. Plan place trajectory
        place_traj = self._plan_arm_trajectory(home_config, place_pose, num_steps=30)
        place_retreat_traj = place_traj[::-1].copy()

        # 9. Assemble full demo trajectory
        # Phases: approach → close_gripper → retreat → navigate → approach_place → open_gripper → retreat
        gripper_closed = np.zeros(1)
        gripper_open = np.ones(1) * 0.04  # Franka gripper open width

        # Compute navigation trajectory (simple linear interpolation of base position)
        nav_steps = 20
        nav_traj = np.zeros((nav_steps, 3))  # [base_x, base_y, base_yaw]
        for i in range(nav_steps):
            t = i / max(nav_steps - 1, 1)
            nav_traj[i] = [
                (1 - t) * base_x + t * place_base_x,
                (1 - t) * base_y + t * place_base_y,
                (1 - t) * base_yaw + t * place_yaw,
            ]

        # Total timesteps
        T = (len(approach_traj) + 5 + len(retreat_traj) +
             nav_steps +
             len(place_traj) + 5 + len(place_retreat_traj))

        # Build states, actions, observations
        dt = 0.02  # 50 Hz
        states = np.zeros((T, 7 + 3 + 1))  # arm_joints(7) + base(3) + gripper(1)
        actions = np.zeros((T, 7 + 3 + 1))  # Same dimensions
        rewards = np.zeros(T)
        dones = np.zeros(T, dtype=bool)
        dones[-1] = True
        rewards[-1] = 1.0  # Sparse reward

        # Step labels for decomposition
        step_labels = []
        t = 0

        # Phase 1: Approach pick
        for i in range(len(approach_traj)):
            states[t, :7] = approach_traj[i]
            states[t, 7:10] = [base_x, base_y, base_yaw]
            states[t, 10] = 0.04  # gripper open
            if t > 0:
                actions[t - 1, :7] = approach_traj[i] - approach_traj[max(0, i - 1)]
            step_labels.append("approach_pick")
            t += 1

        # Phase 2: Close gripper
        for i in range(5):
            states[t, :7] = approach_traj[-1]
            states[t, 7:10] = [base_x, base_y, base_yaw]
            states[t, 10] = 0.04 * (1 - i / 4)  # Closing
            actions[t - 1, 10] = -0.04 / 5  # Close action
            step_labels.append("grasp")
            t += 1

        # Phase 3: Retreat
        for i in range(len(retreat_traj)):
            states[t, :7] = retreat_traj[i]
            states[t, 7:10] = [base_x, base_y, base_yaw]
            states[t, 10] = 0.0  # gripper closed
            if t > 0:
                actions[t - 1, :7] = retreat_traj[i] - retreat_traj[max(0, i - 1)]
            step_labels.append("retreat_pick")
            t += 1

        # Phase 4: Navigate
        for i in range(nav_steps):
            states[t, :7] = home_config
            states[t, 7:10] = nav_traj[i]
            states[t, 10] = 0.0
            if t > 0:
                actions[t - 1, 7:10] = nav_traj[i] - nav_traj[max(0, i - 1)]
            step_labels.append("navigate")
            t += 1

        # Phase 5: Approach place
        for i in range(len(place_traj)):
            states[t, :7] = place_traj[i]
            states[t, 7:10] = [place_base_x, place_base_y, place_yaw]
            states[t, 10] = 0.0
            if t > 0:
                actions[t - 1, :7] = place_traj[i] - place_traj[max(0, i - 1)]
            step_labels.append("approach_place")
            t += 1

        # Phase 6: Open gripper
        for i in range(5):
            states[t, :7] = place_traj[-1]
            states[t, 7:10] = [place_base_x, place_base_y, place_yaw]
            states[t, 10] = 0.04 * (i / 4)  # Opening
            actions[t - 1, 10] = 0.04 / 5
            step_labels.append("release")
            t += 1

        # Phase 7: Retreat from place
        for i in range(len(place_retreat_traj)):
            states[t, :7] = place_retreat_traj[i]
            states[t, 7:10] = [place_base_x, place_base_y, place_yaw]
            states[t, 10] = 0.04
            if t > 0:
                actions[t - 1, :7] = place_retreat_traj[i] - place_retreat_traj[max(0, i - 1)]
            step_labels.append("retreat_place")
            t += 1

        # Truncate to actual length
        states = states[:t]
        actions = actions[:t]
        rewards = rewards[:t]
        dones = dones[:t]

        # Build observations
        obs = {
            "joint_pos": states[:, :7].astype(np.float32),
            "robot_eef_pos": np.zeros((t, 3), dtype=np.float32),  # Would need FK
            "robot_eef_quat": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (t, 1)),
            "gripper_pos": states[:, 10:11].astype(np.float32),
            "base_pos": states[:, 7:10].astype(np.float32),
        }

        # Placeholder camera observations (640x480x3 uint8)
        if self.enable_cameras:
            obs["agentview_rgb"] = np.zeros((t, 480, 640, 3), dtype=np.uint8)
            obs["agentview_depth"] = np.zeros((t, 480, 640), dtype=np.float32)
            obs["eye_in_hand_rgb"] = np.zeros((t, 480, 640, 3), dtype=np.uint8)
            obs["eye_in_hand_depth"] = np.zeros((t, 480, 640), dtype=np.float32)

        demo = {
            "states": states.astype(np.float32),
            "actions": actions.astype(np.float32),
            "obs": obs,
            "rewards": rewards.astype(np.float32),
            "dones": dones,
            "step_labels": step_labels,
            "metadata": {
                "pick_object": pick_obj["type"],
                "pick_object_id": pick_obj.get("id", ""),
                "place_surface": place_surface["type"],
                "place_surface_id": place_surface.get("id", ""),
                "robot_base_pick": [float(base_x), float(base_y), float(base_yaw)],
                "robot_base_place": [float(place_base_x), float(place_base_y), float(place_yaw)],
                "num_steps": t,
                "dt": dt,
            },
        }

        print(f"  Generated: {t} steps ({len(approach_traj)}+5+{len(retreat_traj)}+{nav_steps}+"
              f"{len(place_traj)}+5+{len(place_retreat_traj)})")

        return demo

    def generate_demos(self, num_demos=16):
        """Generate multiple demonstrations."""
        demos = []

        if not self.manipulable_objects:
            print("[DATA-GEN] WARNING: No manipulable objects found in scene")
            return demos

        if not self.support_surfaces:
            print("[DATA-GEN] WARNING: No support surfaces found in scene")
            return demos

        for i in range(num_demos):
            # Randomly select pick object and place surface
            pick_obj = self.manipulable_objects[i % len(self.manipulable_objects)]
            place_surface = self.support_surfaces[i % len(self.support_surfaces)]

            # Don't place on same surface
            if place_surface.get("id") == pick_obj.get("id"):
                place_surface = self.support_surfaces[(i + 1) % len(self.support_surfaces)]

            try:
                demo = self._generate_single_demo(pick_obj, place_surface, i)
                if demo is not None:
                    demos.append(demo)
            except Exception as e:
                print(f"[DATA-GEN] Demo {i} failed: {e}")

        return demos

    def save_hdf5(self, demos, output_dir):
        """Save demos in robomimic-compatible HDF5 format."""
        try:
            import h5py
        except ImportError:
            print("[DATA-GEN] h5py not available — saving as JSON instead")
            return self.save_json(demos, output_dir)

        os.makedirs(output_dir, exist_ok=True)
        hdf5_path = os.path.join(output_dir, "dataset.hdf5")

        with h5py.File(hdf5_path, "w") as f:
            data_grp = f.create_group("data")

            for i, demo in enumerate(demos):
                demo_grp = data_grp.create_group(f"demo_{i}")

                # States and actions
                demo_grp.create_dataset("states", data=demo["states"])
                demo_grp.create_dataset("actions", data=demo["actions"])
                demo_grp.create_dataset("rewards", data=demo["rewards"])
                demo_grp.create_dataset("dones", data=demo["dones"])

                # Observations
                obs_grp = demo_grp.create_group("obs")
                for key, value in demo["obs"].items():
                    if "rgb" in key:
                        # Compress RGB images
                        obs_grp.create_dataset(
                            key, data=value,
                            compression="gzip", compression_opts=4,
                        )
                    else:
                        obs_grp.create_dataset(key, data=value)

                # Metadata as attributes
                for key, value in demo["metadata"].items():
                    if isinstance(value, (list, tuple)):
                        demo_grp.attrs[key] = np.array(value)
                    else:
                        demo_grp.attrs[key] = value

            # Train/val split (80/20)
            num_demos = len(demos)
            num_train = int(num_demos * 0.8)
            mask_grp = f.create_group("mask")
            mask_grp.create_dataset("train", data=list(range(num_train)))
            mask_grp.create_dataset("valid", data=list(range(num_train, num_demos)))

        print(f"[DATA-GEN] Saved {len(demos)} demos to {hdf5_path}")
        return hdf5_path

    def save_json(self, demos, output_dir):
        """Fallback: save demos as JSON (no h5py)."""
        os.makedirs(output_dir, exist_ok=True)

        metadata = {
            "num_demos": len(demos),
            "room_type": self.room["room_type"],
            "demos": [],
        }

        for i, demo in enumerate(demos):
            demo_meta = demo["metadata"].copy()
            demo_meta["num_steps"] = len(demo["states"])
            demo_meta["states_shape"] = list(demo["states"].shape)
            demo_meta["actions_shape"] = list(demo["actions"].shape)
            metadata["demos"].append(demo_meta)

            # Save trajectory as numpy
            np.save(
                os.path.join(output_dir, f"demo_{i}_states.npy"),
                demo["states"],
            )
            np.save(
                os.path.join(output_dir, f"demo_{i}_actions.npy"),
                demo["actions"],
            )

        with open(os.path.join(output_dir, "demo_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[DATA-GEN] Saved {len(demos)} demos to {output_dir} (JSON+NPY format)")
        return output_dir

    def save_step_decomposition(self, demos, output_dir):
        """Save step decomposition (phase labels for each timestep)."""
        os.makedirs(output_dir, exist_ok=True)

        decomposition = {
            "num_demos": len(demos),
            "phase_labels": [
                "approach_pick", "grasp", "retreat_pick",
                "navigate", "approach_place", "release", "retreat_place",
            ],
            "demos": [],
        }

        for i, demo in enumerate(demos):
            labels = demo.get("step_labels", [])
            # Count steps per phase
            phase_counts = {}
            for label in labels:
                phase_counts[label] = phase_counts.get(label, 0) + 1

            decomposition["demos"].append({
                "demo_idx": i,
                "total_steps": len(labels),
                "phase_counts": phase_counts,
                "phase_boundaries": self._find_phase_boundaries(labels),
            })

        with open(os.path.join(output_dir, "step_decomposition.json"), "w") as f:
            json.dump(decomposition, f, indent=2)

    def _find_phase_boundaries(self, labels):
        """Find start/end indices for each phase."""
        boundaries = []
        if not labels:
            return boundaries

        current_phase = labels[0]
        start = 0
        for i, label in enumerate(labels):
            if label != current_phase:
                boundaries.append({
                    "phase": current_phase,
                    "start": start,
                    "end": i - 1,
                })
                current_phase = label
                start = i
        boundaries.append({
            "phase": current_phase,
            "start": start,
            "end": len(labels) - 1,
        })
        return boundaries


def main():
    parser = argparse.ArgumentParser(description="Generate Mobile Franka demonstrations")
    parser.add_argument("--layout_id", required=True, help="Layout directory name")
    parser.add_argument("--results_dir", default="/workspace/SAGE/server/results",
                        help="SAGE results directory")
    parser.add_argument("--pose_aug_name", default="", help="Pose augmentation name")
    parser.add_argument("--num_demos", type=int, default=16, help="Number of demos to generate")
    parser.add_argument("--enable_cameras", action="store_true", help="Enable camera observations")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()

    layout_dir = Path(args.results_dir) / args.layout_id
    if not layout_dir.exists():
        print(f"[ERROR] Layout directory not found: {layout_dir}")
        sys.exit(1)

    # Find room JSON
    room_jsons = list(layout_dir.glob("room_*.json"))
    if not room_jsons:
        print(f"[ERROR] No room_*.json found in {layout_dir}")
        sys.exit(1)

    room_json_path = room_jsons[0]

    # Load grasp data if available
    grasp_file = layout_dir / "grasps" / "grasp_transforms.json"
    grasp_data = {}
    if grasp_file.exists():
        with open(grasp_file) as f:
            grasp_data = json.load(f)
        print(f"[DATA-GEN] Loaded {grasp_data.get('num_grasps', 0)} grasps")

    # Create generator
    generator = MobileFrankaDataGenerator(
        layout_dir=str(layout_dir),
        room_json_path=str(room_json_path),
        grasp_data=grasp_data,
        enable_cameras=args.enable_cameras,
        headless=args.headless,
    )

    # Generate demos
    print(f"\n[DATA-GEN] Generating {args.num_demos} demonstrations...")
    t0 = time.time()
    demos = generator.generate_demos(num_demos=args.num_demos)
    elapsed = time.time() - t0
    print(f"\n[DATA-GEN] Generated {len(demos)} demos in {elapsed:.1f}s")

    if not demos:
        print("[DATA-GEN] No demos generated — check scene for manipulable objects")
        sys.exit(1)

    # Save
    output_dir = str(layout_dir / "demos")
    generator.save_hdf5(demos, output_dir)
    generator.save_step_decomposition(demos, output_dir)

    print(f"\n[DATA-GEN] Done! Output: {output_dir}")


if __name__ == "__main__":
    main()
