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

        # Initialize pipeline components
        video_config = VideoGeneratorConfig(
            resolution=config.resolution,
            num_frames=config.num_frames,
            fps=config.fps,
            model_name=config.video_model,
            api_endpoint=config.video_api_endpoint,
        )
        self.video_generator = VideoGenerator(video_config)

        flow_config = FlowExtractorConfig(
            method=config.flow_method,
            num_tracking_points=config.num_tracking_points,
        )
        self.flow_extractor = FlowExtractor(flow_config)

        tracker_config = RobotTrackerConfig(
            method=config.tracking_method,
            robot=config.robot_embodiment,
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

        In production, this would render from the USD scene.
        Currently creates placeholder observation.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Integrate with Isaac Sim / Omniverse for real rendering
        # Placeholder: create simple observation

        width, height = self.config.resolution

        # Placeholder RGB (gray with text)
        rgb = np.full((height, width, 3), 60, dtype=np.uint8)

        # Placeholder depth (gradient)
        depth = np.linspace(0.5, 3.0, height)[:, np.newaxis]
        depth = np.tile(depth, (1, width)).astype(np.float32)

        # Save to files
        from PIL import Image
        rgb_path = output_dir / "initial_rgb.png"
        depth_path = output_dir / "initial_depth.png"

        Image.fromarray(rgb).save(rgb_path)
        depth_mm = (depth * 1000).astype(np.uint16)
        Image.fromarray(depth_mm).save(depth_path)

        # Default camera intrinsics
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
        )

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
