#!/usr/bin/env python3
"""
Dream2Flow Preparation Job - Main entry point.

Generates Dream2Flow bundles from BlueprintPipeline scenes:
1. Generates task instructions from scene manifest
2. Renders initial RGB-D observations
3. Generates "dreamed" task videos using video diffusion
4. Extracts 3D object flow from generated videos
5. Creates robot tracking targets
6. Packages everything into Dream2Flow-ready bundles

Based on Dream2Flow paper (arXiv:2512.24766):
- Input: RGB-D observation + language instruction
- Video generation: Generate imagined task execution video
- Flow extraction: Extract 3D object flow (masks, depth, point tracking)
- Robot control: Use flow as goal/reward for trajectory optimization or RL

Note: The Dream2Flow model is not yet publicly released.
This module provides scaffolding that will be updated when available.

Usage:
    python prepare_dream2flow_bundle.py \\
        --manifest-path ./assets/scene_manifest.json \\
        --output-dir ./dream2flow_output \\
        --num-tasks 5
"""

import argparse
import importlib.util
import json
import os
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
    Dream2FlowBundle,
    Dream2FlowJobConfig,
    Dream2FlowPipelineOutput,
    FlowExtractionMethod,
    GeneratedVideo,
    RGBDObservation,
    RobotEmbodiment,
    RobotTrackingMethod,
    RobotTrackingTarget,
    TaskInstruction,
    TaskType,
)
from video_generator import VideoGenerator, VideoGeneratorConfig
from flow_extractor import FlowExtractor, FlowExtractorConfig
from robot_tracker import RobotTracker, RobotTrackerConfig
from bundle_packager import Dream2FlowBundlePackager


# Task templates for different object categories
TASK_TEMPLATES = {
    "drawer": [
        ("Open the {object}", TaskType.OPEN),
        ("Close the {object}", TaskType.CLOSE),
        ("Pull the {object}", TaskType.PULL),
    ],
    "door": [
        ("Open the {object}", TaskType.OPEN),
        ("Close the {object}", TaskType.CLOSE),
    ],
    "cabinet": [
        ("Open the {object}", TaskType.OPEN),
        ("Close the {object}", TaskType.CLOSE),
    ],
    "refrigerator": [
        ("Open the {object} door", TaskType.OPEN),
        ("Close the {object} door", TaskType.CLOSE),
    ],
    "microwave": [
        ("Open the {object} door", TaskType.OPEN),
        ("Close the {object} door", TaskType.CLOSE),
    ],
    "oven": [
        ("Open the {object}", TaskType.OPEN),
        ("Close the {object}", TaskType.CLOSE),
    ],
    "chair": [
        ("Pull the {object}", TaskType.PULL_CHAIR),
        ("Push the {object}", TaskType.PUSH),
    ],
    "bowl": [
        ("Cover the {object}", TaskType.COVER),
        ("Pick up the {object}", TaskType.GRASP),
    ],
    "cup": [
        ("Pick up the {object}", TaskType.GRASP),
        ("Place the {object}", TaskType.PLACE),
    ],
    "bottle": [
        ("Pick up the {object}", TaskType.GRASP),
        ("Place the {object}", TaskType.PLACE),
    ],
    "can": [
        ("Pick up the {object}", TaskType.GRASP),
        ("Recycle the {object}", TaskType.RECYCLE),
    ],
    "default": [
        ("Pick up the {object}", TaskType.GRASP),
        ("Push the {object}", TaskType.PUSH),
        ("Move the {object}", TaskType.PICK_AND_PLACE),
    ],
}


class Dream2FlowPreparationJob:
    """
    Main job class for Dream2Flow preparation.

    Orchestrates the full pipeline:
    1. Load scene manifest and generate task instructions
    2. Render initial RGB-D observations
    3. Generate "dreamed" task videos
    4. Extract 3D object flow from videos
    5. Create robot tracking targets
    6. Package into Dream2Flow bundles
    """

    def __init__(self, config: Dream2FlowJobConfig):
        self.config = config
        self.manifest = None
        self.scene_objects = {}
        allow_placeholder = config.allow_placeholder and not config.require_real_backends

        # Initialize pipeline components
        video_config = VideoGeneratorConfig(
            resolution=config.resolution,
            num_frames=config.num_frames,
            fps=config.fps,
            model_name=config.video_model,
            api_endpoint=config.video_api_endpoint,
            checkpoint_path=config.video_checkpoint_path,
            enabled=config.enable_video_generation,
            allow_placeholder=allow_placeholder,
            require_real_backend=config.require_real_backends,
        )
        self.video_generator = VideoGenerator(video_config)

        flow_config = FlowExtractorConfig(
            method=config.flow_method,
            num_tracking_points=config.num_tracking_points,
            segmentation_api=config.segmentation_api,
            depth_api=config.depth_api,
            tracking_api=config.tracking_api,
            enabled=config.enable_flow_extraction,
            allow_placeholder=allow_placeholder,
        )
        self.flow_extractor = FlowExtractor(flow_config)

        tracker_config = RobotTrackerConfig(
            method=config.tracking_method,
            robot=config.robot_embodiment,
            tracking_api=config.robot_tracking_api,
            enabled=config.enable_robot_tracking,
            allow_placeholder=allow_placeholder,
            require_real_backend=config.require_real_backends,
        )
        self.robot_tracker = RobotTracker(tracker_config)

        self.packager = Dream2FlowBundlePackager(config.output_dir)

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.config.verbose:
            print(f"[D2F-JOB] [{level}] {msg}")

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
                    "name": obj.get("name", obj_id),
                    "sim_role": obj.get("sim_role", "static"),
                    "bounds": obj.get("bounds", {}),
                    "description": obj.get("semantics", {}).get("description", ""),
                    "affordances": (obj.get("semantics") or {}).get("affordances", []),
                    "articulation": obj.get("articulation_state") or obj.get("articulation") or {},
                }

            self.log(f"Loaded manifest: {self.scene_id} ({len(self.scene_objects)} objects)")
            return True

        except Exception as e:
            self.log(f"Failed to load manifest: {e}", "ERROR")
            return False

    def generate_task_instructions(self) -> list[TaskInstruction]:
        """Generate task instructions based on scene objects."""
        instructions = []

        for obj_id, obj_info in self.scene_objects.items():
            category = obj_info["category"].lower()
            name = obj_info.get("name", category)

            # Skip background/static objects
            if obj_info.get("sim_role") == "background":
                continue

            # Get templates for this category
            templates = TASK_TEMPLATES.get(category, TASK_TEMPLATES["default"])

            for template_text, task_type in templates:
                # Filter by configured task types
                if self.config.task_types and task_type not in self.config.task_types:
                    continue

                # Filter by target object IDs
                if self.config.target_object_ids and obj_id not in self.config.target_object_ids:
                    continue

                instruction_text = template_text.format(object=name)

                instruction = TaskInstruction(
                    text=instruction_text,
                    action_verb=template_text.split()[0].lower(),
                    target_object=name,
                    task_type=task_type,
                    parameters={"object_id": obj_id},
                )
                instructions.append(instruction)

        # Limit to configured number of tasks
        if len(instructions) > self.config.num_tasks:
            # Prioritize variety of task types
            selected = []
            task_type_counts = {}
            for instr in instructions:
                tt = instr.task_type
                if task_type_counts.get(tt, 0) < 2:
                    selected.append(instr)
                    task_type_counts[tt] = task_type_counts.get(tt, 0) + 1
                if len(selected) >= self.config.num_tasks:
                    break
            instructions = selected

        self.log(f"Generated {len(instructions)} task instructions")
        return instructions

    def render_initial_observation(
        self,
        instruction: TaskInstruction,
        output_dir: Path,
    ) -> RGBDObservation:
        """
        Render initial RGB-D observation for a task.

        Render from Isaac Sim/Omniverse when available, using USD scene + camera calibration.
        Falls back to placeholder data only when explicitly allowed.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._should_enforce_real_render():
            if not self.config.scene_usd_path:
                raise RuntimeError("Scene USD path is required for production Dream2Flow rendering.")

        if self.config.scene_usd_path:
            try:
                return self._render_with_isaac_sim(output_dir)
            except Exception as exc:
                if self._should_enforce_real_render() or not self.config.allow_placeholder:
                    raise
                self.log(f"Isaac Sim render failed, falling back to placeholder: {exc}", "WARNING")

        if not self.config.allow_placeholder:
            raise RuntimeError("Placeholder rendering disabled but Isaac Sim render is unavailable.")

        return self._render_placeholder_observation(output_dir)

    def _should_enforce_real_render(self) -> bool:
        """Check if production flags require real rendering."""
        return (
            os.getenv("PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}
            or os.getenv("DATA_QUALITY_LEVEL", "").lower() == "production"
            or os.getenv("DREAM2FLOW_PRODUCTION", "").lower() in {"1", "true", "yes"}
        )

    def _render_placeholder_observation(self, output_dir: Path) -> RGBDObservation:
        """Generate a placeholder RGB-D observation for development."""
        width, height = self.config.resolution

        rgb = np.full((height, width, 3), 60, dtype=np.uint8)
        depth = np.linspace(0.5, 3.0, height)[:, np.newaxis]
        depth = np.tile(depth, (1, width)).astype(np.float32)

        from PIL import Image

        rgb_path = output_dir / "initial_rgb.png"
        depth_path = output_dir / "initial_depth.png"

        Image.fromarray(rgb).save(rgb_path)
        depth_mm = (depth * 1000).astype(np.uint16)
        Image.fromarray(depth_mm).save(depth_path)

        fx = fy = 500.0
        cx, cy = width / 2, height / 2
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])

        return RGBDObservation(
            frame_idx=0,
            rgb=rgb,
            depth=depth,
            rgb_path=rgb_path,
            depth_path=depth_path,
            camera_intrinsics=intrinsics,
            camera_extrinsics=np.eye(4),
            timestamp=time.time(),
        )

    def _render_with_isaac_sim(self, output_dir: Path) -> RGBDObservation:
        """Render RGB-D observation with Isaac Sim/Omniverse."""
        if not self.config.scene_usd_path:
            raise RuntimeError("Scene USD path is required for Isaac Sim rendering.")

        if importlib.util.find_spec("omni") is None:
            raise RuntimeError("Isaac Sim modules are not available in this environment.")

        from omni.isaac.kit import SimulationApp
        import omni
        from omni.isaac.core.utils.stage import open_stage
        import omni.kit.viewport.utility as vp_utils
        from omni.kit.capture.viewport import CaptureExtension
        from omni.syntheticdata import SyntheticDataHelper
        from PIL import Image

        width, height = self.config.resolution
        rgb_path = output_dir / "initial_rgb.png"
        depth_path = output_dir / "initial_depth.png"

        simulation_app = SimulationApp({"headless": True})
        try:
            open_stage(str(self.config.scene_usd_path))
            stage = omni.usd.get_context().get_stage()

            intrinsics, extrinsics = self._load_camera_calibration(stage, width, height)
            camera_path = "/World/Dream2FlowCamera"
            self._configure_render_camera(stage, camera_path, intrinsics, extrinsics, width, height)

            viewport = vp_utils.get_active_viewport()
            viewport.set_texture_resolution((width, height))
            viewport.set_active_camera(camera_path)

            capture_ext = CaptureExtension()
            capture_ext.capture_frame(str(rgb_path), width, height)
            simulation_app.update()

            sd_helper = SyntheticDataHelper()
            render_product = viewport.get_render_product_path()
            gt = sd_helper.get_groundtruth(render_product, ["depth"], wait_for_servers=True)
            if "depth" not in gt:
                raise RuntimeError("Isaac Sim depth capture missing 'depth' output.")

            depth = np.asarray(gt["depth"], dtype=np.float32)
            depth_mm = np.clip(depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            Image.fromarray(depth_mm).save(depth_path)

            rgb = np.array(Image.open(rgb_path))

            return RGBDObservation(
                frame_idx=0,
                rgb=rgb,
                depth=depth,
                rgb_path=rgb_path,
                depth_path=depth_path,
                camera_intrinsics=intrinsics,
                camera_extrinsics=extrinsics,
                timestamp=time.time(),
            )
        finally:
            simulation_app.close()

    def _load_camera_calibration(
        self,
        stage,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load camera intrinsics/extrinsics from the USD stage."""
        intrinsics = None
        extrinsics = None

        cameras_scope = stage.GetPrimAtPath("/World/Cameras")
        camera_prims = []
        if cameras_scope and cameras_scope.IsValid():
            camera_prims.extend(cameras_scope.GetChildren())
        if not camera_prims:
            camera_prims = [prim for prim in stage.Traverse() if prim.IsValid()]

        for prim in camera_prims:
            intr_attr = prim.GetAttribute("intrinsics")
            extr_attr = prim.GetAttribute("cameraExtrinsics")
            if intr_attr and intr_attr.HasAuthoredValueOpinion():
                intr_val = intr_attr.Get()
                intr_vals = list(intr_val)
                if len(intr_vals) >= 3:
                    fx, fy, cx = intr_vals[:3]
                    cy = height / 2
                    intrinsics = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1],
                    ], dtype=np.float32)
            if extr_attr and extr_attr.HasAuthoredValueOpinion():
                extr_val = extr_attr.Get()
                extrinsics = np.array(extr_val, dtype=np.float32)
            if intrinsics is not None and extrinsics is not None:
                break

        if intrinsics is None:
            raise RuntimeError("No camera intrinsics found in USD scene.")
        if extrinsics is None:
            extrinsics = np.eye(4, dtype=np.float32)

        return intrinsics, extrinsics

    def _configure_render_camera(
        self,
        stage,
        camera_path: str,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        """Create/update a USD camera prim with the desired calibration."""
        from pxr import Gf, UsdGeom

        camera = UsdGeom.Camera.Define(stage, camera_path)
        xform = UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()
        matrix = Gf.Matrix4d(*extrinsics.astype(float).flatten().tolist())
        xform.AddTransformOp().Set(matrix)

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        horizontal_aperture = 36.0
        focal_length = fx * horizontal_aperture / width
        vertical_aperture = height * focal_length / fy

        camera.GetFocalLengthAttr().Set(float(focal_length))
        camera.GetHorizontalApertureAttr().Set(float(horizontal_aperture))
        camera.GetVerticalApertureAttr().Set(float(vertical_aperture))

        horiz_offset = (cx - (width / 2.0)) * horizontal_aperture / width
        vert_offset = (cy - (height / 2.0)) * vertical_aperture / height
        camera.GetHorizontalApertureOffsetAttr().Set(float(horiz_offset))
        camera.GetVerticalApertureOffsetAttr().Set(float(vert_offset))
    def process_task(
        self,
        instruction: TaskInstruction,
        task_idx: int,
    ) -> Dream2FlowBundle:
        """Process a single task through the Dream2Flow pipeline."""
        bundle_id = f"{self.scene_id}_task_{task_idx:03d}"
        task_dir = self.config.output_dir / bundle_id

        self.log(f"Processing task {task_idx + 1}: '{instruction.text}'")

        # Step 1: Render initial observation
        observation = self.render_initial_observation(
            instruction=instruction,
            output_dir=task_dir / "observation",
        )

        # Step 2: Generate dreamed video
        video = self.video_generator.generate(
            observation=observation,
            instruction=instruction,
            output_dir=task_dir / "video",
            video_id=f"{bundle_id}_video",
        )

        video_success = video.num_frames > 0

        # Step 3: Extract 3D object flow
        flow_result = None
        if video_success:
            flow_result = self.flow_extractor.extract(
                video=video,
                output_dir=task_dir / "flow",
                target_object=instruction.target_object,
            )

        flow_success = flow_result is not None and flow_result.success

        # Step 4: Create tracking target and generate trajectory
        tracking_target = None
        robot_trajectory = None
        execution_success = False

        if flow_success and flow_result.object_flows:
            primary_flow = flow_result.get_primary_flow()
            if primary_flow:
                tracking_target = RobotTrackingTarget(
                    target_id=f"{bundle_id}_target",
                    object_flow=primary_flow,
                    tracking_method=self.config.tracking_method,
                    robot=self.config.robot_embodiment,
                )

                tracking_result = self.robot_tracker.track(
                    object_flow=primary_flow,
                    output_dir=task_dir / "trajectory",
                )

                if tracking_result.success:
                    robot_trajectory = tracking_result.trajectory
                    execution_success = True

        # Create bundle
        bundle = Dream2FlowBundle(
            bundle_id=bundle_id,
            scene_id=self.scene_id,
            instruction=instruction,
            initial_observation=observation,
            generated_video=video,
            flow_extraction=flow_result,
            tracking_target=tracking_target,
            robot_trajectory=robot_trajectory,
            resolution=self.config.resolution,
            num_frames=self.config.num_frames,
            fps=self.config.fps,
            bundle_dir=task_dir,
            video_generation_success=video_success,
            flow_extraction_success=flow_success,
            robot_execution_success=execution_success,
            metadata={
                "task_idx": task_idx,
                "generated_at": datetime.utcnow().isoformat() + "Z",
            },
        )

        return bundle

    def run(self) -> Dream2FlowPipelineOutput:
        """Run the full Dream2Flow preparation pipeline."""
        start_time = time.time()
        errors = []

        self.log("=" * 60)
        self.log("Dream2Flow Preparation Job")
        self.log("=" * 60)
        self.log(f"Manifest: {self.config.manifest_path}")
        self.log(f"Output: {self.config.output_dir}")
        self.log(f"Tasks: {self.config.num_tasks}")
        self.log(f"Resolution: {self.config.resolution}")
        self.log(f"Frames: {self.config.num_frames} @ {self.config.fps}fps")
        self.log("=" * 60)

        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Load manifest
        if not self.load_manifest():
            return Dream2FlowPipelineOutput(
                scene_id="unknown",
                errors=["Failed to load manifest"],
                generation_time_seconds=time.time() - start_time,
            )

        # Step 2: Generate task instructions
        instructions = self.generate_task_instructions()
        if not instructions:
            return Dream2FlowPipelineOutput(
                scene_id=self.scene_id,
                errors=["No tasks generated from manifest"],
                generation_time_seconds=time.time() - start_time,
            )

        # Step 3: Process each task
        bundles = []
        for i, instruction in enumerate(instructions):
            try:
                bundle = self.process_task(instruction, i)
                bundles.append(bundle)
            except Exception as e:
                error_msg = f"Task {i} failed: {e}"
                errors.append(error_msg)
                self.log(error_msg, "ERROR")
                traceback.print_exc()

        # Step 4: Package bundles
        output = self.packager.package_all(
            bundles=bundles,
            scene_id=self.scene_id,
        )
        output.errors.extend(errors)
        output.generation_time_seconds = time.time() - start_time

        # Summary
        self.log("=" * 60)
        self.log("DREAM2FLOW PREPARATION SUMMARY")
        self.log("=" * 60)
        self.log(f"Bundles generated: {len(output.bundles)}")
        self.log(f"Video generation: {output.num_successful_videos}/{len(bundles)}")
        self.log(f"Flow extraction: {output.num_successful_flows}/{len(bundles)}")
        self.log(f"Robot execution: {output.num_successful_trajectories}/{len(bundles)}")
        self.log(f"Time: {output.generation_time_seconds:.2f}s")
        if output.errors:
            self.log(f"Errors: {len(output.errors)}", "WARN")
        self.log(f"Output: {output.output_dir}")
        self.log("=" * 60)

        return output


def prepare_dream2flow_bundles(
    manifest_path: Path,
    output_dir: Path = Path("./dream2flow_output"),
    num_tasks: int = 5,
    task_types: Optional[list[TaskType]] = None,
    resolution: tuple[int, int] = (720, 480),
    num_frames: int = 49,
    fps: float = 24.0,
    verbose: bool = True,
) -> Dream2FlowPipelineOutput:
    """
    Convenience function to prepare Dream2Flow bundles.

    Args:
        manifest_path: Path to scene manifest
        output_dir: Output directory
        num_tasks: Number of tasks to generate
        task_types: Types of tasks to generate
        resolution: Video resolution
        num_frames: Frames per video
        fps: Frames per second
        verbose: Print progress

    Returns:
        Dream2FlowPipelineOutput with results
    """
    config = Dream2FlowJobConfig(
        manifest_path=Path(manifest_path),
        output_dir=Path(output_dir),
        num_tasks=num_tasks,
        task_types=task_types,
        resolution=resolution,
        num_frames=num_frames,
        fps=fps,
        verbose=verbose,
    )

    job = Dream2FlowPreparationJob(config)
    return job.run()


def run_dream2flow_preparation(
    scene_dir: Path,
    output_dir: Optional[Path] = None,
    num_tasks: int = 5,
    verbose: bool = True,
) -> Dream2FlowPipelineOutput:
    """
    Run Dream2Flow preparation on a scene directory.

    Expects standard BlueprintPipeline scene structure:
        scene_dir/
        ├── assets/
        │   └── scene_manifest.json
        └── usd/
            └── scene.usda

    Args:
        scene_dir: Path to scene directory
        output_dir: Output directory (default: scene_dir/dream2flow)
        num_tasks: Number of tasks
        verbose: Print progress

    Returns:
        Dream2FlowPipelineOutput with results
    """
    scene_dir = Path(scene_dir)

    manifest_path = scene_dir / "assets" / "scene_manifest.json"
    scene_usd_path = scene_dir / "usd" / "scene.usda"

    if output_dir is None:
        output_dir = scene_dir / "dream2flow"

    config = Dream2FlowJobConfig(
        manifest_path=manifest_path,
        scene_usd_path=scene_usd_path if scene_usd_path.exists() else None,
        output_dir=output_dir,
        num_tasks=num_tasks,
        verbose=verbose,
    )

    job = Dream2FlowPreparationJob(config)
    return job.run()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Dream2Flow bundles from BlueprintPipeline scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate from manifest
    python prepare_dream2flow_bundle.py \\
        --manifest-path ./assets/scene_manifest.json \\
        --output-dir ./dream2flow_output

    # Generate from scene directory
    python prepare_dream2flow_bundle.py \\
        --scene-dir ./scenes/kitchen_001 \\
        --num-tasks 10

    # Specify task types
    python prepare_dream2flow_bundle.py \\
        --manifest-path ./manifest.json \\
        --tasks open close push grasp
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
        help="Path to scene USD file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./dream2flow_output"),
        help="Output directory for bundles",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=5,
        help="Number of tasks to generate",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["push", "grasp", "open", "close", "pull", "place", "pick_and_place"],
        default=None,
        help="Task types to generate",
    )
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
        help="Number of frames per video",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frames per second",
    )
    parser.add_argument(
        "--robot",
        choices=["franka_panda", "ur5e", "spot", "gr1"],
        default="franka_panda",
        help="Robot embodiment for tracking",
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
        output_dir = args.output_dir if args.output_dir != Path("./dream2flow_output") else (args.scene_dir / "dream2flow")
    elif args.manifest_path:
        manifest_path = args.manifest_path
        scene_usd_path = args.scene_usd_path
        output_dir = args.output_dir
    else:
        parser.error("Either --manifest-path or --scene-dir is required")

    # Map task names to enums
    task_types = None
    if args.tasks:
        task_map = {
            "push": TaskType.PUSH,
            "grasp": TaskType.GRASP,
            "open": TaskType.OPEN,
            "close": TaskType.CLOSE,
            "pull": TaskType.PULL,
            "place": TaskType.PLACE,
            "pick_and_place": TaskType.PICK_AND_PLACE,
        }
        task_types = [task_map[t] for t in args.tasks]

    # Map robot name to enum
    robot_map = {
        "franka_panda": RobotEmbodiment.FRANKA_PANDA,
        "ur5e": RobotEmbodiment.UR5E,
        "spot": RobotEmbodiment.BOSTON_DYNAMICS_SPOT,
        "gr1": RobotEmbodiment.FOURIER_GR1,
    }
    robot = robot_map[args.robot]

    # Create config
    config = Dream2FlowJobConfig(
        manifest_path=manifest_path,
        scene_usd_path=scene_usd_path if scene_usd_path and scene_usd_path.exists() else None,
        output_dir=output_dir,
        num_tasks=args.num_tasks,
        task_types=task_types,
        resolution=tuple(args.resolution),
        num_frames=args.num_frames,
        fps=args.fps,
        robot_embodiment=robot,
        verbose=not args.quiet,
    )

    # Run job
    job = Dream2FlowPreparationJob(config)
    output = job.run()

    # Exit with appropriate code
    sys.exit(0 if output.success else 1)


if __name__ == "__main__":
    main()
