#!/usr/bin/env python3
"""
DWM Preparation Job - Main entry point.

Generates DWM conditioning bundles from BlueprintPipeline scenes:
1. Generates egocentric camera trajectories
2. Renders static scene videos along trajectories
3. Generates hand manipulation trajectories
4. Renders hand mesh videos
5. Packages everything into DWM-ready bundles

Based on DWM paper (arXiv:2512.17907):
- DWM requires: static scene video + hand mesh video + text prompt
- Output resolution: 720x480, 49 frames at 24fps
- Uses video diffusion to generate plausible interaction dynamics

Production bundles must be rendered against the real USD scene (Isaac Sim backend).
The mock renderer is reserved for CI smoke tests where GPU rendering is unavailable.

Usage:
    python prepare_dwm_bundle.py \\
        --manifest-path ./assets/scene_manifest.json \\
        --scene-usd-path ./usd/scene.usda \\
        --output-dir ./dwm_output \\
        --num-trajectories 5
"""

import argparse
import json
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import (
    CameraTrajectory,
    DWMConditioningBundle,
    DWMPipelineOutput,
    DWMSceneConfig,
    HandActionType,
    HandTrajectory,
    TrajectoryType,
)
from trajectory_generator import EgocentricTrajectoryGenerator
from trajectory_generator.physics_policy_runner import PhysicsPolicyRunner
from scene_renderer import RenderBackend, RenderConfig, SceneRenderer
from hand_motion import (
    HandMeshRenderer,
    HandRenderConfig,
    HandTrajectoryGenerator,
    HandRetargeter,
    RobotConfig,
)
from bundle_packager import DWMBundlePackager, generate_text_prompt


@dataclass
class DWMJobConfig:
    """Configuration for DWM preparation job."""

    # Input paths
    manifest_path: Path
    scene_usd_path: Optional[Path] = None
    scene_glb_path: Optional[Path] = None

    # Output
    output_dir: Path = Path("./dwm_output")

    # Trajectory generation
    num_trajectories: int = 5
    trajectory_types: list[TrajectoryType] = None
    action_types: list[HandActionType] = None
    target_object_ids: Optional[list[str]] = None

    # Video parameters (DWM defaults)
    resolution: tuple[int, int] = (720, 480)
    num_frames: int = 49
    fps: float = 24.0

    # Rendering
    render_backend: Optional[RenderBackend] = None  # Isaac Sim for production; mock only for CI
    render_static_scene: bool = True
    render_hand_mesh: bool = True
    export_depth: bool = True
    export_segmentation: bool = True
    export_scene_state: bool = True

    # Output options
    encode_videos: bool = True
    keep_frames: bool = True
    compress_bundles: bool = False

    # Processing
    verbose: bool = True
    enable_robot_retargeting: bool = False
    robot_config_name: str = "ur5e_parallel_gripper"
    robot_demo_roots: Optional[list[Path]] = None
    use_physics_ground_truth: bool = False

    def __post_init__(self):
        if self.trajectory_types is None:
            self.trajectory_types = [
                TrajectoryType.APPROACH,
                TrajectoryType.REACH_MANIPULATE,
            ]
        if self.action_types is None:
            self.action_types = [
                HandActionType.GRASP,
                HandActionType.PULL,
                HandActionType.PUSH,
            ]
        if self.robot_demo_roots is None:
            self.robot_demo_roots = []


class DWMPreparationJob:
    """
    Main job class for DWM conditioning data preparation.

    Orchestrates the full pipeline:
    1. Load scene manifest and extract object info
    2. Generate egocentric camera trajectories
    3. Generate aligned hand manipulation trajectories
    4. Render static scene videos
    5. Render hand mesh videos
    6. Package into DWM-ready bundles
    """

    def __init__(self, config: DWMJobConfig):
        """
        Initialize DWM preparation job.

        Args:
            config: Job configuration
        """
        self.config = config
        self.manifest = None
        self.scene_objects = {}

        # Initialize generators
        self.trajectory_generator = None
        self.hand_generator = HandTrajectoryGenerator()
        self.hand_retargeter = (
            HandRetargeter(RobotConfig(name=config.robot_config_name))
            if config.enable_robot_retargeting
            else None
        )

        # Initialize renderers
        render_config = RenderConfig(
            width=config.resolution[0],
            height=config.resolution[1],
        )
        self.scene_renderer = SceneRenderer(
            backend=config.render_backend,
            config=render_config,
        )

        hand_render_config = HandRenderConfig(
            width=config.resolution[0],
            height=config.resolution[1],
        )
        self.hand_renderer = HandMeshRenderer(config=hand_render_config)

        # Initialize packager
        self.packager = DWMBundlePackager(config.output_dir)
        self.physics_runner = PhysicsPolicyRunner(fps=config.fps)

    def log(self, msg: str, level: str = "INFO") -> None:
        """Log a message."""
        if self.config.verbose:
            print(f"[DWM-JOB] [{level}] {msg}")

    def load_manifest(self) -> bool:
        """Load and parse scene manifest."""
        try:
            self.manifest = json.loads(self.config.manifest_path.read_text())
            self.scene_id = self.manifest.get("scene_id", "unknown")

            # Extract objects
            for obj in self.manifest.get("objects", []):
                obj_id = obj.get("id", f"obj_{len(self.scene_objects)}")
                pos = obj.get("transform", {}).get("position", {})
                position = np.array([
                    pos.get("x", 0),
                    pos.get("y", 0),
                    pos.get("z", 0),
                ])

                self.scene_objects[obj_id] = {
                    "position": position,
                    "category": obj.get("category", "object"),
                    "sim_role": obj.get("sim_role", "static"),
                    "bounds": obj.get("bounds", {}),
                    "description": obj.get("semantics", {}).get("description", ""),
                }

            self.log(f"Loaded manifest: {self.scene_id} ({len(self.scene_objects)} objects)")
            return True

        except Exception as e:
            self.log(f"Failed to load manifest: {e}", "ERROR")
            return False

    def setup_trajectory_generator(self) -> bool:
        """Setup trajectory generator from manifest."""
        try:
            self.trajectory_generator = EgocentricTrajectoryGenerator.from_manifest(
                self.config.manifest_path
            )
            self.trajectory_generator.fps = self.config.fps
            self.trajectory_generator.num_frames = self.config.num_frames
            return True
        except Exception as e:
            self.log(f"Failed to setup trajectory generator: {e}", "ERROR")
            return False

    def generate_trajectories(self) -> list[CameraTrajectory]:
        """Generate camera trajectories for the scene."""
        self.log(f"Generating {self.config.num_trajectories} trajectories...")

        trajectories = self.trajectory_generator.generate_for_scene(
            num_trajectories=self.config.num_trajectories,
            trajectory_types=self.config.trajectory_types,
            action_types=self.config.action_types,
            target_object_ids=self.config.target_object_ids,
        )

        self.log(f"Generated {len(trajectories)} camera trajectories")
        return trajectories

    def generate_hand_trajectories(
        self,
        camera_trajectories: list[CameraTrajectory],
    ) -> list[tuple[CameraTrajectory, HandTrajectory]]:
        """
        Generate hand trajectories aligned with camera trajectories.

        Args:
            camera_trajectories: List of camera trajectories

        Returns:
            List of (camera_trajectory, hand_trajectory) pairs
        """
        self.log("Generating hand trajectories...")

        pairs = []
        for cam_traj in camera_trajectories:
            # Determine target position
            target_obj_id = cam_traj.target_object_id
            if target_obj_id and target_obj_id in self.scene_objects:
                target_pos = self.scene_objects[target_obj_id]["position"]
            else:
                # Use last camera position as target (approximation)
                last_pose = cam_traj.poses[-1]
                target_pos = last_pose.position + last_pose.forward * 0.5

            # Determine action type
            if cam_traj.action_type:
                action_type = cam_traj.action_type
            elif cam_traj.trajectory_type == TrajectoryType.REACH_MANIPULATE:
                action_type = HandActionType.GRASP
            else:
                action_type = HandActionType.REACH

            # Generate hand trajectory
            hand_traj = self.hand_generator.generate_for_camera_trajectory(
                camera_trajectory=cam_traj,
                target_position=target_pos,
                action_type=action_type,
            )

            # Link trajectories
            hand_traj.camera_trajectory_id = cam_traj.trajectory_id
            hand_traj.target_object_id = target_obj_id

            if self.hand_retargeter:
                hand_traj.robot_actions = self.hand_retargeter.retarget(
                    hand_traj, camera_traj=cam_traj
                )

            pairs.append((cam_traj, hand_traj))

        self.log(f"Generated {len(pairs)} hand trajectories")
        return pairs

    def render_scene_videos(
        self,
        trajectory_pairs: list[tuple[CameraTrajectory, HandTrajectory]],
    ) -> dict[str, dict]:
        """
        Render static scene videos for all trajectories.

        Args:
            trajectory_pairs: List of (camera, hand) trajectory pairs

        Returns:
            Dict mapping trajectory_id to render output paths
        """
        if not self.config.render_static_scene:
            self.log("Skipping static scene rendering (disabled)")
            return {}

        self.log("Rendering static scene videos...")

        # Determine scene file to render
        scene_path = None
        if self.config.scene_glb_path and self.config.scene_glb_path.exists():
            scene_path = self.config.scene_glb_path
        elif self.config.scene_usd_path and self.config.scene_usd_path.exists():
            scene_path = self.config.scene_usd_path
        else:
            if self.scene_renderer.backend == RenderBackend.MOCK:
                self.log(
                    "No renderable scene file found - mock renderer active (intended for CI smoke tests only)",
                    "WARN",
                )
            else:
                raise FileNotFoundError(
                    "No renderable scene file found. Provide a USD/GLB scene path for production rendering "
                    "or disable rendering explicitly for CI runs."
                )

        # Load scene
        if scene_path:
            if self.scene_renderer.backend == RenderBackend.ISAAC_SIM and scene_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
                raise ValueError(
                    f"Isaac Sim rendering requires a USD input. Found {scene_path.suffix} for {scene_path}."
                )
            self.scene_renderer.load_scene(scene_path)

        outputs = {}
        for cam_traj, hand_traj in trajectory_pairs:
            traj_id = cam_traj.trajectory_id
            frames_dir = self.config.output_dir / "frames" / traj_id / "static_scene"
            depth_dir = (
                self.config.output_dir / "frames" / traj_id / "static_scene_depth"
                if self.config.export_depth else None
            )
            seg_dir = (
                self.config.output_dir / "frames" / traj_id / "static_scene_seg"
                if self.config.export_segmentation else None
            )

            self.log(f"  Rendering {traj_id}...")

            render_outputs = self.scene_renderer.render_trajectory_outputs(
                trajectory=cam_traj,
                output_dir=frames_dir,
                frame_prefix="scene",
                depth_dir=depth_dir,
                segmentation_dir=seg_dir,
            )
            frame_paths = render_outputs.get("color", [])

            video_path = None
            if self.config.encode_videos and frame_paths:
                video_path = self.config.output_dir / "videos" / f"{traj_id}_scene.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                self.scene_renderer.frames_to_video(
                    frames_dir=frames_dir,
                    output_path=video_path,
                    fps=cam_traj.fps,
                    frame_pattern="scene_*.png",
                )

            scene_state_path = None
            if self.config.export_scene_state:
                state_dir = self.config.output_dir / "metadata" / traj_id
                scene_state_path = self._write_scene_state(
                    cam_traj,
                    state_dir / "scene_state.json",
                )

            outputs[traj_id] = {
                "frames_dir": frames_dir,
                "video_path": video_path,
                "num_frames": len(frame_paths),
                "depth_dir": depth_dir if render_outputs.get("depth") else None,
                "seg_dir": seg_dir if render_outputs.get("segmentation") else None,
                "scene_state_path": scene_state_path,
            }

        self.log(f"Rendered {len(outputs)} static scene videos")
        return outputs

    def _build_object_states(self) -> list[dict[str, Any]]:
        """Build object state list from the loaded manifest."""
        object_states = []
        for obj in self.manifest.get("objects", []):
            articulation_state = obj.get("articulation_state") or obj.get("articulation") or {}
            object_states.append({
                "id": obj.get("id"),
                "name": obj.get("name"),
                "category": obj.get("category"),
                "sim_role": obj.get("sim_role"),
                "transform": obj.get("transform", {}),
                "articulation": articulation_state,
            })
        return object_states

    def _write_scene_state(
        self,
        trajectory: CameraTrajectory,
        output_path: Path,
    ) -> Path:
        """Write per-frame scene state entries for a trajectory."""
        object_states = self._build_object_states()
        scene_info = self.manifest.get("scene", {})

        state_entries = []
        for pose in trajectory.poses:
            timestamp = pose.frame_idx / trajectory.fps if trajectory.fps else pose.frame_idx
            state_entries.append({
                "frame_idx": pose.frame_idx,
                "timestamp": timestamp,
                "camera_pose": pose.transform.tolist(),
                "camera_position": pose.position.tolist(),
                "objects": object_states,
                "scene": {
                    "coordinate_frame": scene_info.get("coordinate_frame"),
                    "meters_per_unit": scene_info.get("meters_per_unit"),
                },
            })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(state_entries, indent=2))
        return output_path

    def render_hand_videos(
        self,
        trajectory_pairs: list[tuple[CameraTrajectory, HandTrajectory]],
    ) -> dict[str, dict]:
        """
        Render hand mesh videos for all trajectories.

        Args:
            trajectory_pairs: List of (camera, hand) trajectory pairs

        Returns:
            Dict mapping trajectory_id to render output paths
        """
        if not self.config.render_hand_mesh:
            self.log("Skipping hand mesh rendering (disabled)")
            return {}

        self.log("Rendering hand mesh videos...")

        outputs = {}
        for cam_traj, hand_traj in trajectory_pairs:
            traj_id = cam_traj.trajectory_id
            frames_dir = self.config.output_dir / "frames" / traj_id / "hand_mesh"

            self.log(f"  Rendering {traj_id}...")

            frame_paths = self.hand_renderer.render_trajectory(
                hand_trajectory=hand_traj,
                camera_trajectory=cam_traj,
                output_dir=frames_dir,
                frame_prefix="hand",
            )

            video_path = None
            if self.config.encode_videos and frame_paths:
                video_path = self.config.output_dir / "videos" / f"{traj_id}_hand.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                self.hand_renderer.frames_to_video(
                    frames_dir=frames_dir,
                    output_path=video_path,
                    fps=hand_traj.fps,
                    frame_pattern="hand_*.png",
                )

            outputs[traj_id] = {
                "frames_dir": frames_dir,
                "video_path": video_path,
                "num_frames": len(frame_paths),
            }

        self.log(f"Rendered {len(outputs)} hand mesh videos")
        return outputs

    def run_physics_rollouts(
        self,
        trajectory_pairs: list[tuple[CameraTrajectory, HandTrajectory]],
    ) -> dict[str, Path]:
        """
        Execute optional physics-based rollouts for each trajectory.

        Returns:
            Dict mapping trajectory_id to physics rollout log path
        """
        if not self.config.use_physics_ground_truth:
            self.log("Skipping physics rollouts (disabled)")
            return {}

        if not self.config.scene_usd_path or not self.config.scene_usd_path.exists():
            self.log(
                "Physics rollouts requested but USD scene is missing",
                "WARN",
            )
            return {}

        self.log("Running physics rollouts with scripted policies...")
        output_dir = self.config.output_dir / "metadata"

        return self.physics_runner.run_rollouts(
            scene_usd_path=self.config.scene_usd_path,
            trajectory_pairs=trajectory_pairs,
            output_dir=output_dir,
            num_frames=self.config.num_frames,
            scene_objects=self.scene_objects,
        )

    def create_bundles(
        self,
        trajectory_pairs: list[tuple[CameraTrajectory, HandTrajectory]],
        scene_renders: dict[str, dict],
        hand_renders: dict[str, dict],
        physics_rollouts: Optional[dict[str, Path]] = None,
    ) -> list[DWMConditioningBundle]:
        """
        Create DWM conditioning bundles from all components.

        Args:
            trajectory_pairs: List of (camera, hand) trajectory pairs
            scene_renders: Scene video render outputs
            hand_renders: Hand video render outputs

        Returns:
            List of DWM conditioning bundles
        """
        self.log("Creating DWM conditioning bundles...")

        bundles = []
        for cam_traj, hand_traj in trajectory_pairs:
            traj_id = cam_traj.trajectory_id
            physics_log_path = (physics_rollouts or {}).get(traj_id)

            # Get target object info for prompt
            target_obj_id = cam_traj.target_object_id
            target_category = None
            target_description = None
            if target_obj_id and target_obj_id in self.scene_objects:
                obj_info = self.scene_objects[target_obj_id]
                target_category = obj_info.get("category")
                target_description = obj_info.get("description")

            # Generate text prompt
            text_prompt = generate_text_prompt(
                action_type=hand_traj.action_type,
                target_category=target_category,
                target_description=target_description,
            )

            # Get render paths
            scene_render = scene_renders.get(traj_id, {})
            hand_render = hand_renders.get(traj_id, {})

            bundle = DWMConditioningBundle(
                bundle_id=f"{self.scene_id}_{traj_id}",
                scene_id=self.scene_id,
                camera_trajectory=cam_traj,
                hand_trajectory=hand_traj,
                static_scene_video_path=scene_render.get("video_path"),
                hand_mesh_video_path=hand_render.get("video_path"),
                static_scene_frames_dir=scene_render.get("frames_dir"),
                static_scene_depth_dir=scene_render.get("depth_dir"),
                static_scene_seg_dir=scene_render.get("seg_dir"),
                scene_state_path=scene_render.get("scene_state_path"),
                hand_mesh_frames_dir=hand_render.get("frames_dir"),
                physics_log_path=physics_log_path,
                text_prompt=text_prompt,
                resolution=self.config.resolution,
                num_frames=self.config.num_frames,
                fps=self.config.fps,
                metadata={
                    "target_object_id": target_obj_id,
                    "target_category": target_category,
                    "action_type": hand_traj.action_type.value,
                    "trajectory_type": cam_traj.trajectory_type.value,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "robot_retargeting": {
                        "enabled": self.config.enable_robot_retargeting,
                        "robot_model": self.config.robot_config_name,
                        "demo_roots": [str(p) for p in self.config.robot_demo_roots],
                    },
                    "physics_rollout": {
                        "enabled": self.config.use_physics_ground_truth,
                        "log_relative_path": "metadata/physics_rollout.jsonl"
                        if physics_log_path else None,
                    },
                },
            )

            bundles.append(bundle)

        self.log(f"Created {len(bundles)} bundles")
        return bundles

    def run(self) -> DWMPipelineOutput:
        """
        Run the full DWM preparation pipeline.

        Returns:
            DWMPipelineOutput with all results
        """
        start_time = time.time()
        errors = []

        self.log("=" * 60)
        self.log("DWM Preparation Job")
        self.log("=" * 60)
        self.log(f"Manifest: {self.config.manifest_path}")
        self.log(f"Output: {self.config.output_dir}")
        self.log(f"Trajectories: {self.config.num_trajectories}")
        self.log(f"Resolution: {self.config.resolution}")
        self.log(f"Frames: {self.config.num_frames} @ {self.config.fps}fps")
        self.log("=" * 60)

        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Load manifest
        if not self.load_manifest():
            return DWMPipelineOutput(
                scene_id="unknown",
                errors=["Failed to load manifest"],
                generation_time_seconds=time.time() - start_time,
            )

        # Step 2: Setup trajectory generator
        if not self.setup_trajectory_generator():
            return DWMPipelineOutput(
                scene_id=self.scene_id,
                errors=["Failed to setup trajectory generator"],
                generation_time_seconds=time.time() - start_time,
            )

        # Step 3: Generate camera trajectories
        try:
            camera_trajectories = self.generate_trajectories()
        except Exception as e:
            errors.append(f"Trajectory generation failed: {e}")
            camera_trajectories = []

        if not camera_trajectories:
            return DWMPipelineOutput(
                scene_id=self.scene_id,
                errors=errors or ["No trajectories generated"],
                generation_time_seconds=time.time() - start_time,
            )

        # Step 4: Generate hand trajectories
        try:
            trajectory_pairs = self.generate_hand_trajectories(camera_trajectories)
        except Exception as e:
            errors.append(f"Hand trajectory generation failed: {e}")
            trajectory_pairs = [(t, None) for t in camera_trajectories]

        # Step 5: Render static scene videos
        try:
            scene_renders = self.render_scene_videos(trajectory_pairs)
        except Exception as e:
            errors.append(f"Scene rendering failed: {e}")
            scene_renders = {}

        # Step 6: Render hand mesh videos
        try:
            hand_renders = self.render_hand_videos(trajectory_pairs)
        except Exception as e:
            errors.append(f"Hand rendering failed: {e}")
            hand_renders = {}

        # Optional: Physics rollouts
        try:
            physics_rollouts = self.run_physics_rollouts(trajectory_pairs)
        except Exception as e:
            errors.append(f"Physics rollout failed: {e}")
            physics_rollouts = {}

        # Step 7: Create bundles
        try:
            bundles = self.create_bundles(
                trajectory_pairs,
                scene_renders,
                hand_renders,
                physics_rollouts,
            )
        except Exception as e:
            errors.append(f"Bundle creation failed: {e}")
            bundles = []

        # Step 8: Package bundles
        try:
            output = self.packager.package_all(
                bundles=bundles,
                scene_id=self.scene_id,
                include_frames=self.config.keep_frames,
            )
            output.errors.extend(errors)
            output.generation_time_seconds = time.time() - start_time
        except Exception as e:
            output = DWMPipelineOutput(
                scene_id=self.scene_id,
                bundles=bundles,
                output_dir=self.config.output_dir,
                errors=errors + [f"Packaging failed: {e}"],
                generation_time_seconds=time.time() - start_time,
            )

        # Summary
        self.log("=" * 60)
        self.log("DWM PREPARATION SUMMARY")
        self.log("=" * 60)
        self.log(f"Bundles generated: {len(output.bundles)}")
        self.log(f"Total frames: {output.total_frames}")
        self.log(f"Time: {output.generation_time_seconds:.2f}s")
        if output.errors:
            self.log(f"Errors: {len(output.errors)}", "WARN")
            for err in output.errors[:3]:
                self.log(f"  - {err}", "WARN")
        self.log(f"Output: {output.output_dir}")
        self.log("=" * 60)

        return output


def prepare_dwm_bundles(
    manifest_path: Path,
    scene_usd_path: Optional[Path] = None,
    output_dir: Path = Path("./dwm_output"),
    num_trajectories: int = 5,
    trajectory_types: Optional[list[TrajectoryType]] = None,
    action_types: Optional[list[HandActionType]] = None,
    resolution: tuple[int, int] = (720, 480),
    num_frames: int = 49,
    fps: float = 24.0,
    verbose: bool = True,
    use_physics_ground_truth: bool = False,
) -> DWMPipelineOutput:
    """
    Convenience function to prepare DWM bundles.

    Args:
        manifest_path: Path to scene manifest
        scene_usd_path: Path to scene USD file
        output_dir: Output directory
        num_trajectories: Number of trajectories to generate
        trajectory_types: Types of trajectories
        action_types: Types of hand actions
        resolution: Video resolution
        num_frames: Number of frames per video
        fps: Frames per second
        verbose: Print progress
        use_physics_ground_truth: Run Isaac Sim/Lab rollouts for aligned physics logs

    Returns:
        DWMPipelineOutput with results
    """
    config = DWMJobConfig(
        manifest_path=Path(manifest_path),
        scene_usd_path=Path(scene_usd_path) if scene_usd_path else None,
        output_dir=Path(output_dir),
        num_trajectories=num_trajectories,
        trajectory_types=trajectory_types,
        action_types=action_types,
        resolution=resolution,
        num_frames=num_frames,
        fps=fps,
        verbose=verbose,
        use_physics_ground_truth=use_physics_ground_truth,
    )

    job = DWMPreparationJob(config)
    return job.run()


def run_dwm_preparation(
    scene_dir: Path,
    output_dir: Optional[Path] = None,
    num_trajectories: int = 5,
    verbose: bool = True,
    use_physics_ground_truth: bool = False,
) -> DWMPipelineOutput:
    """
    Run DWM preparation on a scene directory.

    Expects standard BlueprintPipeline scene structure:
        scene_dir/
        ├── assets/
        │   └── scene_manifest.json
        └── usd/
            └── scene.usda

    Args:
        scene_dir: Path to scene directory
        output_dir: Output directory (default: scene_dir/dwm)
        num_trajectories: Number of trajectories
        verbose: Print progress

    Returns:
        DWMPipelineOutput with results
    """
    scene_dir = Path(scene_dir)

    manifest_path = scene_dir / "assets" / "scene_manifest.json"
    scene_usd_path = scene_dir / "usd" / "scene.usda"

    if output_dir is None:
        output_dir = scene_dir / "dwm"

    return prepare_dwm_bundles(
        manifest_path=manifest_path,
        scene_usd_path=scene_usd_path,
        output_dir=output_dir,
        num_trajectories=num_trajectories,
        verbose=verbose,
        use_physics_ground_truth=use_physics_ground_truth,
    )


# =============================================================================
# Episode-Based DWM Generation (Enhanced Pipeline)
# =============================================================================


def prepare_dwm_episodes(
    manifest_path: Path,
    output_dir: Path = Path("./dwm_output"),
    scene_usd_path: Optional[Path] = None,
    max_episodes: int = 10,
    max_clips_per_episode: int = 10,
    prioritize_by: str = "dwm_relevance",
    use_llm_analysis: bool = True,
    use_grounded_search: bool = True,
    resolution: tuple[int, int] = (720, 480),
    frames_per_clip: int = 49,
    fps: float = 24.0,
    render_videos: bool = False,
    verbose: bool = True,
    enable_robot_retargeting: bool = False,
    robot_config_name: str = "ur5e_parallel_gripper",
    robot_demo_roots: Optional[list[Path]] = None,
):
    """
    Generate episode-based DWM bundles with scene analysis and task planning.

    This is the enhanced pipeline that uses:
    1. Scene Analyzer - Extract semantic meaning from manifest using Gemini
    2. Task Planner - Generate episode action sequences
    3. Episode Bundler - Chain clips into full manipulation tasks

    Args:
        manifest_path: Path to scene_manifest.json
        output_dir: Output directory for bundles
        scene_usd_path: Optional path to scene USD for rendering
        max_episodes: Maximum episodes to generate
        max_clips_per_episode: Maximum clips per episode
        prioritize_by: How to prioritize tasks ("dwm_relevance", "difficulty", "variety")
        use_llm_analysis: Use Gemini for scene analysis
        use_grounded_search: Enable Google Search grounding for analysis
        resolution: Video resolution (default: 720x480)
        frames_per_clip: Frames per DWM clip (default: 49)
        fps: Frames per second (default: 24.0)
        render_videos: Whether to render videos (requires Isaac Sim)
        verbose: Print progress

    Returns:
        EpisodeBundlerOutput with all generated episodes

    Example:
        output = prepare_dwm_episodes(
            manifest_path="./assets/scene_manifest.json",
            output_dir="./dwm_episodes",
            max_episodes=5,
        )

        for episode in output.episodes:
            print(f"Episode: {episode.task_name}")
            for clip in episode.clips:
                print(f"  Clip {clip.clip_index}: {clip.text_prompt}")
    """
    if robot_demo_roots is None:
        robot_demo_roots = []
    from episode_bundler import EpisodeBundler

    bundler = EpisodeBundler(
        output_dir=Path(output_dir),
        frames_per_clip=frames_per_clip,
        fps=fps,
        resolution=resolution,
        generate_trajectories=True,
        render_videos=render_videos,
        verbose=verbose,
        enable_robot_retargeting=enable_robot_retargeting,
        robot_config_name=robot_config_name,
        robot_demo_roots=robot_demo_roots,
    )

    return bundler.bundle_from_manifest(
        manifest_path=Path(manifest_path),
        scene_usd_path=Path(scene_usd_path) if scene_usd_path else None,
        max_episodes=max_episodes,
        max_clips_per_episode=max_clips_per_episode,
    )


def run_dwm_episode_preparation(
    scene_dir: Path,
    output_dir: Optional[Path] = None,
    max_episodes: int = 10,
    verbose: bool = True,
    enable_robot_retargeting: bool = False,
    robot_config_name: str = "ur5e_parallel_gripper",
    robot_demo_roots: Optional[list[Path]] = None,
):
    """
    Run episode-based DWM preparation on a scene directory.

    This is the enhanced version that generates meaningful task episodes
    based on scene analysis.

    Expects standard BlueprintPipeline scene structure:
        scene_dir/
        ├── assets/
        │   └── scene_manifest.json
        └── usd/
            └── scene.usda

    Args:
        scene_dir: Path to scene directory
        output_dir: Output directory (default: scene_dir/dwm_episodes)
        max_episodes: Maximum episodes to generate
        verbose: Print progress

    Returns:
        EpisodeBundlerOutput with results
    """
    scene_dir = Path(scene_dir)

    manifest_path = scene_dir / "assets" / "scene_manifest.json"
    scene_usd_path = scene_dir / "usd" / "scene.usda"

    if output_dir is None:
        output_dir = scene_dir / "dwm_episodes"

    return prepare_dwm_episodes(
        manifest_path=manifest_path,
        scene_usd_path=scene_usd_path,
        output_dir=output_dir,
        max_episodes=max_episodes,
        verbose=verbose,
        enable_robot_retargeting=enable_robot_retargeting,
        robot_config_name=robot_config_name,
        robot_demo_roots=robot_demo_roots,
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate DWM conditioning bundles from BlueprintPipeline scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate from manifest (legacy mode)
    python prepare_dwm_bundle.py \\
        --manifest-path ./assets/scene_manifest.json \\
        --output-dir ./dwm_output

    # Generate from scene directory (legacy mode)
    python prepare_dwm_bundle.py \\
        --scene-dir ./scenes/kitchen_001 \\
        --num-trajectories 10

    # ENHANCED: Generate episode-based bundles with scene analysis
    python prepare_dwm_bundle.py \\
        --scene-dir ./scenes/kitchen_001 \\
        --episodes \\
        --max-episodes 5

    # ENHANCED: Episode-based with custom settings
    python prepare_dwm_bundle.py \\
        --manifest-path ./manifest.json \\
        --episodes \\
        --max-episodes 10 \\
        --max-clips 8 \\
        --prioritize difficulty
""",
    )

    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="Path to scene manifest JSON",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        help="Path to scene directory (alternative to manifest-path)",
    )
    parser.add_argument(
        "--scene-usd-path",
        type=Path,
        help="Path to scene USD file (required for production Isaac Sim rendering)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./dwm_output"),
        help="Output directory for bundles",
    )

    # Episode-based mode (enhanced)
    parser.add_argument(
        "--episodes",
        action="store_true",
        help="Use episode-based generation with scene analysis (enhanced mode)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=10,
        help="Maximum episodes to generate (episode mode only)",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=10,
        help="Maximum clips per episode (episode mode only)",
    )
    parser.add_argument(
        "--prioritize",
        choices=["dwm_relevance", "difficulty", "variety"],
        default="dwm_relevance",
        help="Task prioritization strategy (episode mode only)",
    )

    # Legacy mode options
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=5,
        help="Number of trajectories to generate (legacy mode)",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        choices=["reach", "grasp", "pull", "push", "rotate", "lift", "place"],
        default=["grasp", "pull", "push"],
        help="Hand action types to generate (legacy mode)",
    )

    # Common options
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[720, 480],
        metavar=("WIDTH", "HEIGHT"),
        help="Video resolution",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=49,
        help="Number of frames per video/clip",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frames per second",
    )
    parser.add_argument(
        "--enable-robot-retargeting",
        action="store_true",
        help="Enable hand-to-robot retargeting and robot fine-tune metadata",
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        default="ur5e_parallel_gripper",
        help="Robot configuration name to use for retargeting",
    )
    parser.add_argument(
        "--robot-demo-root",
        type=Path,
        action="append",
        default=[],
        help="Root path to real-robot demonstration data (can be specified multiple times)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip rendering (CI/smoke tests only; production bundles require rendered frames)",
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable depth map export during static scene rendering",
    )
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="Disable segmentation mask export during static scene rendering",
    )
    parser.add_argument(
        "--no-scene-state",
        action="store_true",
        help="Disable scene state export",
    )
    parser.add_argument(
        "--use-physics-ground-truth",
        action="store_true",
        help="Run scripted physics rollouts in Isaac Sim/Lab and log per-frame states/actions/contacts",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    # Determine manifest path
    if args.scene_dir:
        manifest_path = args.scene_dir / "assets" / "scene_manifest.json"
        scene_usd_path = args.scene_usd_path or (args.scene_dir / "usd" / "scene.usda")
        if args.episodes:
            output_dir = args.output_dir if args.output_dir != Path("./dwm_output") else (args.scene_dir / "dwm_episodes")
        else:
            output_dir = args.output_dir if args.output_dir != Path("./dwm_output") else (args.scene_dir / "dwm")
    elif args.manifest_path:
        manifest_path = args.manifest_path
        scene_usd_path = args.scene_usd_path
        output_dir = args.output_dir
    else:
        parser.error("Either --manifest-path or --scene-dir is required")

    # Use episode-based mode if requested
    if args.episodes:
        print("[DWM] Using EPISODE-BASED mode with scene analysis")
        print(f"[DWM] Max episodes: {args.max_episodes}, Max clips/episode: {args.max_clips}")
        print(f"[DWM] Prioritize by: {args.prioritize}")

        output = prepare_dwm_episodes(
            manifest_path=manifest_path,
            output_dir=output_dir,
            scene_usd_path=scene_usd_path,
            max_episodes=args.max_episodes,
            max_clips_per_episode=args.max_clips,
            prioritize_by=args.prioritize,
            resolution=tuple(args.resolution),
            frames_per_clip=args.num_frames,
            fps=args.fps,
            render_videos=not args.no_render,
            verbose=not args.quiet,
            enable_robot_retargeting=args.enable_robot_retargeting,
            robot_config_name=args.robot_config,
            robot_demo_roots=args.robot_demo_root,
        )

        # Exit with appropriate code
        sys.exit(0 if output.success else 1)

    # Legacy mode: trajectory-based generation
    print("[DWM] Using LEGACY mode (trajectory-based)")

    # Map action names to enums
    action_map = {
        "reach": HandActionType.REACH,
        "grasp": HandActionType.GRASP,
        "pull": HandActionType.PULL,
        "push": HandActionType.PUSH,
        "rotate": HandActionType.ROTATE,
        "lift": HandActionType.LIFT,
        "place": HandActionType.PLACE,
    }
    action_types = [action_map[a] for a in args.actions]

    # Create config
    config = DWMJobConfig(
        manifest_path=manifest_path,
        scene_usd_path=scene_usd_path,
        output_dir=output_dir,
        num_trajectories=args.num_trajectories,
        action_types=action_types,
        resolution=tuple(args.resolution),
        num_frames=args.num_frames,
        fps=args.fps,
        render_static_scene=not args.no_render,
        render_hand_mesh=not args.no_render,
        export_depth=not args.no_depth,
        export_segmentation=not args.no_segmentation,
        export_scene_state=not args.no_scene_state,
        verbose=not args.quiet,
        enable_robot_retargeting=args.enable_robot_retargeting,
        robot_config_name=args.robot_config,
        robot_demo_roots=args.robot_demo_root,
        use_physics_ground_truth=args.use_physics_ground_truth,
    )

    # Run job
    job = DWMPreparationJob(config)
    output = job.run()

    # Exit with appropriate code
    sys.exit(0 if output.success else 1)


if __name__ == "__main__":
    main()
