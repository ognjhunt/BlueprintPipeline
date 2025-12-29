#!/usr/bin/env python3
"""
Episode Bundler for DWM - Chains clips into full task episodes.

This module combines scene analysis, task planning, and trajectory generation
to produce complete DWM conditioning bundles organized by episodes.

Key features:
1. Generates camera trajectories for each clip
2. Generates hand motion trajectories aligned with camera
3. Creates episode manifests with clip sequences
4. Packages everything for DWM inference

Output structure:
    dwm/
    ├── episodes_manifest.json           # Master manifest of all episodes
    ├── episode_{task_id}/
    │   ├── episode_manifest.json        # Episode metadata
    │   ├── clip_000/
    │   │   ├── manifest.json            # Clip metadata
    │   │   ├── static_scene_video.mp4   # Scene conditioning video
    │   │   ├── hand_mesh_video.mp4      # Hand conditioning video
    │   │   ├── camera_trajectory.json   # Camera poses
    │   │   ├── hand_trajectory.json     # Hand poses (MANO-compatible)
    │   │   └── metadata/
    │   │       ├── prompt.txt           # Text prompt
    │   │       └── clip_info.json       # Additional metadata
    │   ├── clip_001/
    │   └── ...
    └── ...
"""

import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_analyzer import (
    EnvironmentType,
    ObjectAffordance,
    SceneAnalysisResult,
    SceneAnalyzer,
)
from task_planner import (
    ActionStep,
    EpisodeClip,
    ManipulationEpisode,
    TaskPlanner,
    TaskPlannerOutput,
)
from models import (
    CameraPose,
    CameraTrajectory,
    DWMConditioningBundle,
    HandActionType,
    HandPose,
    HandTrajectory,
    RobotAction,
    TrajectoryType,
)
from trajectory_generator import EgocentricTrajectoryGenerator
from hand_motion import HandTrajectoryGenerator
from hand_motion import HandRetargeter, RobotConfig


# =============================================================================
# Constants
# =============================================================================

DWM_FRAMES = 49
DWM_FPS = 24.0
DWM_RESOLUTION = (720, 480)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ClipBundle:
    """Complete bundle for a single DWM clip."""
    clip_id: str
    clip_index: int
    episode_id: str

    # Trajectories
    camera_trajectory: Optional[CameraTrajectory] = None
    hand_trajectory: Optional[HandTrajectory] = None

    # Text prompt
    text_prompt: str = ""

    # Output paths (filled after rendering)
    static_scene_video_path: Optional[Path] = None
    hand_mesh_video_path: Optional[Path] = None
    camera_trajectory_path: Optional[Path] = None
    hand_trajectory_path: Optional[Path] = None

    # Frame directories
    static_scene_frames_dir: Optional[Path] = None
    hand_mesh_frames_dir: Optional[Path] = None
    robot_actions_path: Optional[Path] = None

    # Metadata
    primary_action: Optional[str] = None
    target_object: Optional[str] = None
    frame_range: Tuple[int, int] = (0, 49)
    state_init: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    state_end: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    step_goals: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Check if all components are generated."""
        return (
            self.camera_trajectory is not None and
            self.hand_trajectory is not None and
            bool(self.text_prompt)
        )


@dataclass
class EpisodeBundle:
    """Complete bundle for a manipulation episode."""
    episode_id: str
    task_id: str
    task_name: str
    description: str

    # Scene context
    scene_id: str = ""
    environment_type: str = "generic"

    # Clips
    clips: List[ClipBundle] = field(default_factory=list)

    # Objects
    source_objects: List[str] = field(default_factory=list)
    target_objects: List[str] = field(default_factory=list)

    # Timing
    total_frames: int = 0
    total_duration_seconds: float = 0.0

    # Output directory
    output_dir: Optional[Path] = None
    state_init: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    state_end: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def clip_count(self) -> int:
        return len(self.clips)

    def get_all_prompts(self) -> List[str]:
        """Get all text prompts in order."""
        return [clip.text_prompt for clip in self.clips]


@dataclass
class EpisodeBundlerOutput:
    """Complete output from episode bundler."""
    scene_id: str
    environment_type: str

    # Episodes
    episodes: List[EpisodeBundle] = field(default_factory=list)

    # Statistics
    total_episodes: int = 0
    total_clips: int = 0
    total_frames: int = 0
    total_duration_seconds: float = 0.0

    # Output
    output_dir: Optional[Path] = None
    manifest_path: Optional[Path] = None

    # Timing
    generation_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.episodes) > 0 and len(self.errors) == 0


# =============================================================================
# Episode Bundler
# =============================================================================


class EpisodeBundler:
    """
    Bundles DWM clips into complete manipulation episodes.

    This is the main entry point for episode-based DWM generation.

    Usage:
        bundler = EpisodeBundler(output_dir="./dwm_output")
        output = bundler.bundle_from_manifest(manifest_path)

        for episode in output.episodes:
            print(f"Episode: {episode.task_name}")
            for clip in episode.clips:
                print(f"  Clip {clip.clip_index}: {clip.text_prompt}")
    """

    def __init__(
        self,
        output_dir: Path = Path("./dwm_output"),
        frames_per_clip: int = DWM_FRAMES,
        fps: float = DWM_FPS,
        resolution: Tuple[int, int] = DWM_RESOLUTION,
        generate_trajectories: bool = True,
        render_videos: bool = True,
        verbose: bool = True,
        enable_robot_retargeting: bool = False,
        robot_config_name: str = "ur5e_parallel_gripper",
        robot_demo_roots: Optional[List[Path]] = None,
    ):
        """
        Initialize the episode bundler.

        Args:
            output_dir: Output directory for bundles
            frames_per_clip: Frames per DWM clip (default: 49)
            fps: Frames per second (default: 24.0)
            resolution: Video resolution (default: 720x480)
            generate_trajectories: Whether to generate trajectories
            render_videos: Whether to render videos (requires renderer)
            verbose: Print progress
        """
        self.output_dir = Path(output_dir)
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.resolution = resolution
        self.generate_trajectories = generate_trajectories
        self.render_videos = render_videos
        self.verbose = verbose
        self.robot_demo_roots = robot_demo_roots or []

        # Initialize components
        self.scene_analyzer = SceneAnalyzer(verbose=verbose)
        self.task_planner = TaskPlanner(
            frames_per_clip=frames_per_clip,
            fps=fps,
            verbose=verbose,
        )
        self.trajectory_generator = None
        self.hand_generator = HandTrajectoryGenerator()
        self.hand_retargeter = (
            HandRetargeter(RobotConfig(name=robot_config_name))
            if enable_robot_retargeting
            else None
        )

    def log(self, msg: str, level: str = "INFO") -> None:
        """Log a message."""
        if self.verbose:
            print(f"[EPISODE-BUNDLER] [{level}] {msg}")

    def bundle_from_manifest(
        self,
        manifest_path: Path,
        scene_usd_path: Optional[Path] = None,
        max_episodes: int = 10,
        max_clips_per_episode: int = 10,
    ) -> EpisodeBundlerOutput:
        """
        Generate complete episode bundles from a scene manifest.

        Args:
            manifest_path: Path to scene_manifest.json
            scene_usd_path: Optional path to scene USD for rendering
            max_episodes: Maximum episodes to generate
            max_clips_per_episode: Maximum clips per episode

        Returns:
            EpisodeBundlerOutput with all generated bundles
        """
        start_time = time.time()
        manifest_path = Path(manifest_path)
        errors = []

        self.log("=" * 60)
        self.log("Episode-Based DWM Bundle Generation")
        self.log("=" * 60)
        self.log(f"Manifest: {manifest_path}")
        self.log(f"Output: {self.output_dir}")
        self.log(f"Frames/clip: {self.frames_per_clip}")
        self.log(f"FPS: {self.fps}")
        self.log("=" * 60)

        # Step 1: Analyze scene
        self.log("Step 1: Analyzing scene...")
        try:
            analysis = self.scene_analyzer.analyze(manifest_path)
            self.log(f"  Found {len(analysis.object_semantics)} objects, "
                     f"{len(analysis.task_templates)} tasks")
        except Exception as e:
            errors.append(f"Scene analysis failed: {e}")
            return EpisodeBundlerOutput(
                scene_id="unknown",
                environment_type="generic",
                errors=errors,
                generation_time_seconds=time.time() - start_time,
            )

        # Step 2: Plan episodes
        self.log("Step 2: Planning episodes...")
        plan_episodes: List[ManipulationEpisode] = []
        try:
            plan = self.task_planner.plan_episodes(analysis, max_episodes=max_episodes)
            self.log(f"  Planned {len(plan.episodes)} episodes, "
                     f"{plan.total_clips} clips")
            plan_episodes, length_errors = self._validate_episode_lengths(plan.episodes)
            errors.extend(length_errors)
            if length_errors:
                self.log(f"  Validation warnings: {len(length_errors)}", "WARNING")
            if not plan_episodes:
                return EpisodeBundlerOutput(
                    scene_id=analysis.scene_id,
                    environment_type=analysis.environment_type.value,
                    errors=errors or ["No episodes passed validation"],
                    generation_time_seconds=time.time() - start_time,
                )
        except Exception as e:
            errors.append(f"Task planning failed: {e}")
            return EpisodeBundlerOutput(
                scene_id=analysis.scene_id,
                environment_type=analysis.environment_type.value,
                errors=errors,
                generation_time_seconds=time.time() - start_time,
            )

        # Step 3: Initialize trajectory generator
        if self.generate_trajectories:
            self.log("Step 3: Setting up trajectory generator...")
            try:
                self.trajectory_generator = EgocentricTrajectoryGenerator.from_manifest(
                    manifest_path
                )
                self.trajectory_generator.fps = self.fps
                self.trajectory_generator.num_frames = self.frames_per_clip
            except Exception as e:
                self.log(f"  Trajectory generator setup failed: {e}", "WARNING")
                self.trajectory_generator = None

        # Step 4: Generate bundles for each episode
        self.log("Step 4: Generating episode bundles...")
        episode_bundles = []

        for i, episode in enumerate(plan_episodes):
            self.log(f"  Episode {i+1}/{len(plan_episodes)}: {episode.task_name}")

            try:
                bundle = self._create_episode_bundle(
                    episode=episode,
                    analysis=analysis,
                    max_clips=max_clips_per_episode,
                )
                episode_bundles.append(bundle)
            except Exception as e:
                self.log(f"    Failed: {e}", "ERROR")
                errors.append(f"Episode {episode.task_id} failed: {e}")

        # Step 5: Write bundles to disk
        self.log("Step 5: Writing bundles to disk...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for bundle in episode_bundles:
            try:
                self._write_episode_bundle(bundle)
            except Exception as e:
                self.log(f"  Failed to write {bundle.episode_id}: {e}", "ERROR")
                errors.append(f"Write failed for {bundle.episode_id}: {e}")

        # Step 6: Write master manifest
        self.log("Step 6: Writing master manifest...")
        manifest_path = self._write_master_manifest(episode_bundles, analysis)

        # Calculate totals
        total_clips = sum(b.clip_count for b in episode_bundles)
        total_frames = sum(b.total_frames for b in episode_bundles)
        total_duration = sum(b.total_duration_seconds for b in episode_bundles)
        generation_time = time.time() - start_time

        # Summary
        self.log("=" * 60)
        self.log("SUMMARY")
        self.log("=" * 60)
        self.log(f"Episodes: {len(episode_bundles)}")
        self.log(f"Clips: {total_clips}")
        self.log(f"Frames: {total_frames}")
        self.log(f"Duration: {total_duration:.1f}s")
        self.log(f"Generation time: {generation_time:.1f}s")
        if errors:
            self.log(f"Errors: {len(errors)}", "WARNING")
        self.log(f"Output: {self.output_dir}")
        self.log("=" * 60)

        return EpisodeBundlerOutput(
            scene_id=analysis.scene_id,
            environment_type=analysis.environment_type.value,
            episodes=episode_bundles,
            total_episodes=len(episode_bundles),
            total_clips=total_clips,
            total_frames=total_frames,
            total_duration_seconds=total_duration,
            output_dir=self.output_dir,
            manifest_path=manifest_path,
            generation_time_seconds=generation_time,
            errors=errors,
        )

    def _create_episode_bundle(
        self,
        episode: ManipulationEpisode,
        analysis: SceneAnalysisResult,
        max_clips: int = 10,
    ) -> EpisodeBundle:
        """Create a complete bundle for an episode."""

        clips_to_process = episode.clips[:max_clips]
        clip_bundles = []
        prev_cam: Optional[CameraTrajectory] = None
        prev_hand: Optional[HandTrajectory] = None

        for clip in clips_to_process:
            clip_bundle = self._create_clip_bundle(
                clip=clip,
                episode=episode,
                analysis=analysis,
                previous_camera=prev_cam,
                previous_hand=prev_hand,
            )
            prev_cam = clip_bundle.camera_trajectory or prev_cam
            prev_hand = clip_bundle.hand_trajectory or prev_hand
            clip_bundles.append(clip_bundle)

        total_frames = sum(c.frame_range[1] - c.frame_range[0] for c in clip_bundles)

        return EpisodeBundle(
            episode_id=episode.episode_id,
            task_id=episode.task_id,
            task_name=episode.task_name,
            description=episode.description,
            scene_id=analysis.scene_id,
            environment_type=analysis.environment_type.value,
            clips=clip_bundles,
            source_objects=episode.source_objects,
            target_objects=episode.target_objects,
            total_frames=total_frames,
            total_duration_seconds=total_frames / self.fps,
            state_init=clips_to_process[0].object_states_start if clips_to_process else {},
            state_end=clips_to_process[-1].object_states_end if clips_to_process else {},
        )

    def _create_clip_bundle(
        self,
        clip: EpisodeClip,
        episode: ManipulationEpisode,
        analysis: SceneAnalysisResult,
        previous_camera: Optional[CameraTrajectory] = None,
        previous_hand: Optional[HandTrajectory] = None,
    ) -> ClipBundle:
        """Create a bundle for a single clip."""

        # Generate camera trajectory
        camera_trajectory = None
        if self.generate_trajectories and self.trajectory_generator:
            camera_trajectory = self._generate_camera_trajectory(
                clip, episode, analysis, previous_camera
            )

        # Generate hand trajectory
        hand_trajectory = None
        if self.generate_trajectories and camera_trajectory:
            hand_trajectory = self._generate_hand_trajectory(
                clip, camera_trajectory, episode, analysis, previous_hand
            )

        return ClipBundle(
            clip_id=clip.clip_id,
            clip_index=clip.clip_index,
            episode_id=episode.episode_id,
            camera_trajectory=camera_trajectory,
            hand_trajectory=hand_trajectory,
            text_prompt=clip.text_prompt,
            primary_action=clip.primary_action.value if clip.primary_action else None,
            target_object=clip.primary_target,
            frame_range=(clip.start_frame, clip.end_frame),
            state_init=clip.object_states_start,
            state_end=clip.object_states_end,
            step_goals=clip.step_goals,
        )

    def _generate_camera_trajectory(
        self,
        clip: EpisodeClip,
        episode: ManipulationEpisode,
        analysis: SceneAnalysisResult,
        previous_camera: Optional[CameraTrajectory] = None,
    ) -> Optional[CameraTrajectory]:
        """Generate camera trajectory for a clip."""

        if not self.trajectory_generator:
            return None

        # Get target object position
        target_id = clip.primary_target
        target_position = None

        if target_id:
            # Try to find object position from manifest
            for obj in analysis.object_semantics:
                if obj.object_id == target_id and obj.typical_height_m:
                    # Use typical height as rough position
                    target_position = np.array([0, obj.typical_height_m, 1.0])
                    break

        if target_position is None:
            # Default target position
            target_position = np.array([0, 1.0, 1.0])

        # Generate trajectory based on primary action
        action_type = clip.primary_action if clip.primary_action else HandActionType.REACH

        try:
            from trajectory_generator import generate_reach_manipulate_trajectory

            trajectory = generate_reach_manipulate_trajectory(
                object_position=target_position,
                action_type=action_type,
                num_frames=self.frames_per_clip,
                fps=self.fps,
                trajectory_id=f"{clip.clip_id}_cam",
            )
            trajectory.target_object_id = target_id
            if previous_camera and previous_camera.poses:
                trajectory = self._align_camera_trajectory(
                    trajectory, previous_camera.poses[-1]
                )
            return trajectory

        except Exception as e:
            self.log(f"    Camera trajectory generation failed: {e}", "WARNING")
            return None

    def _align_camera_trajectory(
        self, trajectory: CameraTrajectory, previous_pose: CameraPose
    ) -> CameraTrajectory:
        """Align a generated trajectory to start from previous pose."""
        if not trajectory.poses:
            return trajectory

        base_start = trajectory.poses[0]
        try:
            transform_offset = previous_pose.transform @ np.linalg.inv(base_start.transform)
        except Exception:
            transform_offset = None

        aligned_poses = []
        prev_idx = previous_pose.frame_idx if previous_pose.frame_idx is not None else self.frames_per_clip - 1
        prev_ts = previous_pose.timestamp if previous_pose.timestamp is not None else (
            prev_idx / self.fps
        )
        time_offset = prev_ts + (1.0 / self.fps) - base_start.timestamp

        for pose in trajectory.poses:
            new_transform = pose.transform
            if transform_offset is not None:
                new_transform = transform_offset @ pose.transform

            aligned_poses.append(
                CameraPose(
                    frame_idx=pose.frame_idx,
                    transform=new_transform,
                    timestamp=pose.timestamp + time_offset,
                    focal_length=pose.focal_length,
                )
            )

        return CameraTrajectory(
            trajectory_id=trajectory.trajectory_id,
            trajectory_type=trajectory.trajectory_type,
            poses=aligned_poses,
            fps=trajectory.fps,
            focal_length=trajectory.focal_length,
            sensor_width=trajectory.sensor_width,
            sensor_height=trajectory.sensor_height,
            resolution=trajectory.resolution,
            target_object_id=trajectory.target_object_id,
            action_type=trajectory.action_type,
            description=trajectory.description,
        )

    def _generate_hand_trajectory(
        self,
        clip: EpisodeClip,
        camera_trajectory: CameraTrajectory,
        episode: ManipulationEpisode,
        analysis: SceneAnalysisResult,
        previous_hand: Optional[HandTrajectory] = None,
    ) -> Optional[HandTrajectory]:
        """Generate hand trajectory aligned with camera trajectory."""

        try:
            # Get target position
            target_id = clip.primary_target
            target_position = None

            if camera_trajectory.poses:
                # Use camera's look-at point as target
                last_pose = camera_trajectory.poses[-1]
                target_position = last_pose.position + last_pose.forward * 0.5

            if target_position is None:
                target_position = np.array([0, 1.0, 1.0])

            # Determine action type
            action_type = clip.primary_action if clip.primary_action else HandActionType.REACH

            # Generate hand trajectory
            hand_trajectory = self.hand_generator.generate_for_camera_trajectory(
                camera_trajectory=camera_trajectory,
                target_position=target_position,
                action_type=action_type,
            )

            hand_trajectory.camera_trajectory_id = camera_trajectory.trajectory_id
            hand_trajectory.target_object_id = target_id

            if previous_hand and previous_hand.poses:
                hand_trajectory = self._align_hand_trajectory(
                    hand_trajectory, previous_hand.poses[-1]
                )

            if self.hand_retargeter:
                hand_trajectory.robot_actions = self.hand_retargeter.retarget(
                    hand_trajectory, camera_traj=camera_trajectory
                )

            return hand_trajectory

        except Exception as e:
            self.log(f"    Hand trajectory generation failed: {e}", "WARNING")
            return None

    def _align_hand_trajectory(
        self, trajectory: HandTrajectory, previous_pose: HandPose
    ) -> HandTrajectory:
        """Offset hand trajectory so it continues smoothly from previous pose."""
        if not trajectory.poses:
            return trajectory

        base_start = trajectory.poses[0]
        translation_offset = previous_pose.position - base_start.position

        # rotation alignment
        try:
            rotation_offset = previous_pose.rotation @ np.linalg.inv(base_start.rotation)
        except Exception:
            rotation_offset = None

        prev_idx = previous_pose.frame_idx if previous_pose.frame_idx is not None else self.frames_per_clip - 1
        prev_ts = previous_pose.timestamp if previous_pose.timestamp is not None else (
            prev_idx / self.fps
        )
        time_offset = prev_ts + (1.0 / self.fps) - base_start.timestamp

        aligned_poses: List[HandPose] = []
        for pose in trajectory.poses:
            new_position = pose.position + translation_offset
            new_rotation = pose.rotation
            if rotation_offset is not None:
                new_rotation = rotation_offset @ pose.rotation

            aligned_pose = HandPose(
                frame_idx=pose.frame_idx,
                hand_side=pose.hand_side,
                position=new_position,
                rotation=new_rotation,
                pose_params=pose.pose_params,
                shape_params=pose.shape_params,
                joint_positions=pose.joint_positions,
                timestamp=pose.timestamp + time_offset,
                contact_fingertips=pose.contact_fingertips,
            )
            aligned_poses.append(aligned_pose)

        return HandTrajectory(
            trajectory_id=trajectory.trajectory_id,
            action_type=trajectory.action_type,
            poses=aligned_poses,
            camera_trajectory_id=trajectory.camera_trajectory_id,
            target_object_id=trajectory.target_object_id,
            description=trajectory.description,
            fps=trajectory.fps,
            robot_actions=trajectory.robot_actions,
        )

    def _write_episode_bundle(self, bundle: EpisodeBundle) -> None:
        """Write an episode bundle to disk."""

        episode_dir = self.output_dir / f"episode_{bundle.task_id}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        bundle.output_dir = episode_dir

        # Write episode manifest
        episode_manifest = {
            "episode_id": bundle.episode_id,
            "task_id": bundle.task_id,
            "task_name": bundle.task_name,
            "description": bundle.description,
            "scene_id": bundle.scene_id,
            "environment_type": bundle.environment_type,
            "clip_count": bundle.clip_count,
            "total_frames": bundle.total_frames,
            "total_duration_seconds": bundle.total_duration_seconds,
            "source_objects": bundle.source_objects,
            "target_objects": bundle.target_objects,
            "state_init": bundle.state_init,
            "state_end": bundle.state_end,
            "clips": [
                {
                    "clip_id": clip.clip_id,
                    "clip_index": clip.clip_index,
                    "frame_range": list(clip.frame_range),
                    "primary_action": clip.primary_action,
                    "target_object": clip.target_object,
                    "text_prompt": clip.text_prompt,
                    "is_complete": clip.is_complete,
                    "state_init": clip.state_init,
                    "state_end": clip.state_end,
                    "step_goals": clip.step_goals,
                }
                for clip in bundle.clips
            ],
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

        manifest_path = episode_dir / "episode_manifest.json"
        manifest_path.write_text(json.dumps(episode_manifest, indent=2))

        # Write each clip
        for clip in bundle.clips:
            self._write_clip_bundle(clip, episode_dir)

    def _write_clip_bundle(self, clip: ClipBundle, episode_dir: Path) -> None:
        """Write a clip bundle to disk."""

        clip_dir = episode_dir / f"clip_{clip.clip_index:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir = clip_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Write clip manifest
        clip_manifest = {
            "clip_id": clip.clip_id,
            "clip_index": clip.clip_index,
            "episode_id": clip.episode_id,
            "frame_range": list(clip.frame_range),
            "frame_count": clip.frame_range[1] - clip.frame_range[0],
            "primary_action": clip.primary_action,
            "target_object": clip.target_object,
            "text_prompt": clip.text_prompt,
            "fps": self.fps,
            "resolution": list(self.resolution),
            "has_camera_trajectory": clip.camera_trajectory is not None,
            "has_hand_trajectory": clip.hand_trajectory is not None,
            "has_robot_actions": bool(
                clip.hand_trajectory and clip.hand_trajectory.robot_actions
            ),
            "state_init": clip.state_init,
            "state_end": clip.state_end,
            "step_goals": clip.step_goals,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

        (clip_dir / "manifest.json").write_text(json.dumps(clip_manifest, indent=2))

        # Write text prompt
        (metadata_dir / "prompt.txt").write_text(clip.text_prompt)

        # Write camera trajectory
        if clip.camera_trajectory:
            cam_data = self._serialize_camera_trajectory(clip.camera_trajectory)
            cam_path = clip_dir / "camera_trajectory.json"
            cam_path.write_text(json.dumps(cam_data, indent=2))
            clip.camera_trajectory_path = cam_path

        # Write hand trajectory
        if clip.hand_trajectory:
            hand_data = self._serialize_hand_trajectory(clip.hand_trajectory)
            hand_path = clip_dir / "hand_trajectory.json"
            hand_path.write_text(json.dumps(hand_data, indent=2))
            clip.hand_trajectory_path = hand_path

            if clip.hand_trajectory.robot_actions:
                robot_actions_data = self._serialize_robot_actions(
                    clip.hand_trajectory.robot_actions
                )
                robot_actions_path = clip_dir / "robot_actions.json"
                robot_actions_path.write_text(json.dumps(robot_actions_data, indent=2))
                clip.robot_actions_path = robot_actions_path

        # Write additional metadata
        clip_info = {
            "clip_id": clip.clip_id,
            "text_prompt": clip.text_prompt,
            "primary_action": clip.primary_action,
            "target_object": clip.target_object,
            "dwm_compatible": True,
            "resolution": list(self.resolution),
            "fps": self.fps,
            "frame_count": self.frames_per_clip,
            "has_robot_actions": bool(clip.robot_actions_path),
            "state_init": clip.state_init,
            "state_end": clip.state_end,
        }
        (metadata_dir / "clip_info.json").write_text(json.dumps(clip_info, indent=2))

        self._write_robot_finetune_manifest(metadata_dir)

    def _serialize_camera_trajectory(self, trajectory: CameraTrajectory) -> dict:
        """Serialize camera trajectory to JSON-compatible dict."""
        return {
            "trajectory_id": trajectory.trajectory_id,
            "trajectory_type": trajectory.trajectory_type.value,
            "fps": trajectory.fps,
            "num_frames": trajectory.num_frames,
            "duration_seconds": trajectory.duration,
            "resolution": list(trajectory.resolution),
            "focal_length": trajectory.focal_length,
            "target_object_id": trajectory.target_object_id,
            "action_type": trajectory.action_type.value if trajectory.action_type else None,
            "poses": [
                {
                    "frame_idx": pose.frame_idx,
                    "timestamp": pose.timestamp,
                    "transform": pose.transform.tolist(),
                    "position": pose.position.tolist(),
                }
                for pose in trajectory.poses
            ],
        }

    def _serialize_hand_trajectory(self, trajectory: HandTrajectory) -> dict:
        """Serialize hand trajectory to JSON-compatible dict."""
        return {
            "trajectory_id": trajectory.trajectory_id,
            "action_type": trajectory.action_type.value,
            "fps": trajectory.fps,
            "num_frames": trajectory.num_frames,
            "camera_trajectory_id": trajectory.camera_trajectory_id,
            "target_object_id": trajectory.target_object_id,
            "description": trajectory.description,
            "poses": [
                {
                    "frame_idx": pose.frame_idx,
                    "timestamp": pose.timestamp,
                    "hand_side": pose.hand_side,
                    "position": pose.position.tolist() if isinstance(pose.position, np.ndarray) else list(pose.position),
                    "rotation": pose.rotation.tolist() if isinstance(pose.rotation, np.ndarray) else list(pose.rotation),
                    "pose_params": pose.pose_params.tolist() if pose.pose_params is not None else None,
                    "contact_fingertips": pose.contact_fingertips,
                }
                for pose in trajectory.poses
            ],
            "robot_actions_file": "robot_actions.json"
            if trajectory.robot_actions
            else None,
        }

    def _serialize_robot_actions(self, actions: List[RobotAction]) -> dict:
        """Serialize robot actions for export."""
        return {
            "actions": [action.to_json() for action in actions],
        }

    def _write_robot_finetune_manifest(self, metadata_dir: Path) -> None:
        """Write a scaffold manifest for real-robot fine-tuning assets."""
        robot_model = (
            self.hand_retargeter.robot_config.name if self.hand_retargeter else None
        )
        manifest = {
            "robot_model": robot_model,
            "real_robot_demos": [str(p) for p in self.robot_demo_roots],
            "notes": (
                "Add paths to synchronized robot demonstrations for fine-tuning. "
                "Robot actions are recorded in robot_actions.json."
            ),
        }
        (metadata_dir / "robot_finetune_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

    def _write_master_manifest(
        self,
        episodes: List[EpisodeBundle],
        analysis: SceneAnalysisResult,
    ) -> Path:
        """Write master manifest for all episodes."""

        manifest = {
            "scene_id": analysis.scene_id,
            "environment_type": analysis.environment_type.value,
            "total_episodes": len(episodes),
            "total_clips": sum(e.clip_count for e in episodes),
            "total_frames": sum(e.total_frames for e in episodes),
            "total_duration_seconds": sum(e.total_duration_seconds for e in episodes),
            "dwm_config": {
                "frames_per_clip": self.frames_per_clip,
                "fps": self.fps,
                "resolution": list(self.resolution),
            },
            "episodes": [
                {
                    "episode_id": e.episode_id,
                    "task_id": e.task_id,
                    "task_name": e.task_name,
                    "clip_count": e.clip_count,
                    "total_frames": e.total_frames,
                    "output_dir": str(e.output_dir) if e.output_dir else None,
                    "state_init": e.state_init,
                    "state_end": e.state_end,
                }
                for e in episodes
            ],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "scene_summary": analysis.scene_summary,
            "recommended_policies": analysis.recommended_policies,
        }

        manifest_path = self.output_dir / "episodes_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        return manifest_path

    def _validate_episode_lengths(
        self, episodes: List[ManipulationEpisode]
    ) -> Tuple[List[ManipulationEpisode], List[str]]:
        """Validate clip counts against requested sequence lengths."""
        valid: List[ManipulationEpisode] = []
        errors: List[str] = []

        for episode in episodes:
            expected = None
            if episode.source_template:
                expected = episode.source_template.dwm_clip_count

            if expected and episode.clip_count != expected:
                msg = (
                    f"Episode {episode.task_id} expected {expected} clips "
                    f"but planned {episode.clip_count}"
                )
                errors.append(msg)
                continue

            valid.append(episode)

        return valid, errors


# =============================================================================
# Convenience Functions
# =============================================================================


def bundle_episodes(
    manifest_path: Path,
    output_dir: Path = Path("./dwm_output"),
    max_episodes: int = 10,
    verbose: bool = True,
) -> EpisodeBundlerOutput:
    """Convenience function to bundle episodes from manifest."""
    bundler = EpisodeBundler(output_dir=output_dir, verbose=verbose)
    return bundler.bundle_from_manifest(manifest_path, max_episodes=max_episodes)


def bundle_from_analysis(
    analysis: SceneAnalysisResult,
    output_dir: Path = Path("./dwm_output"),
    max_episodes: int = 10,
    verbose: bool = True,
) -> EpisodeBundlerOutput:
    """Bundle episodes from pre-computed analysis."""
    bundler = EpisodeBundler(output_dir=output_dir, verbose=verbose)

    # Plan episodes
    plan = bundler.task_planner.plan_episodes(analysis, max_episodes=max_episodes)

    # Create bundles
    valid_episodes, _ = bundler._validate_episode_lengths(plan.episodes)

    episode_bundles = []
    for episode in valid_episodes:
        try:
            bundle = bundler._create_episode_bundle(episode, analysis)
            episode_bundles.append(bundle)
        except Exception as e:
            if verbose:
                print(f"Failed to create bundle for {episode.task_id}: {e}")

    # Write to disk
    bundler.output_dir.mkdir(parents=True, exist_ok=True)
    for bundle in episode_bundles:
        bundler._write_episode_bundle(bundle)

    manifest_path = bundler._write_master_manifest(episode_bundles, analysis)

    return EpisodeBundlerOutput(
        scene_id=analysis.scene_id,
        environment_type=analysis.environment_type.value,
        episodes=episode_bundles,
        total_episodes=len(episode_bundles),
        total_clips=sum(b.clip_count for b in episode_bundles),
        total_frames=sum(b.total_frames for b in episode_bundles),
        total_duration_seconds=sum(b.total_duration_seconds for b in episode_bundles),
        output_dir=bundler.output_dir,
        manifest_path=manifest_path,
    )


# =============================================================================
# CLI Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate episode-based DWM bundles from scene manifest"
    )
    parser.add_argument("manifest_path", type=Path, help="Path to scene_manifest.json")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("./dwm_output"),
                        help="Output directory for bundles")
    parser.add_argument("--max-episodes", type=int, default=5,
                        help="Maximum episodes to generate")
    parser.add_argument("--max-clips", type=int, default=10,
                        help="Maximum clips per episode")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output")

    args = parser.parse_args()

    output = bundle_episodes(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Episodes: {output.total_episodes}")
    print(f"Clips: {output.total_clips}")
    print(f"Total frames: {output.total_frames}")
    print(f"Duration: {output.total_duration_seconds:.1f}s")
    print(f"Output: {output.output_dir}")
    if output.errors:
        print(f"Errors: {len(output.errors)}")
        for err in output.errors[:3]:
            print(f"  - {err}")
