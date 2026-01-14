#!/usr/bin/env python3
"""
Full Isaac Sim Pipeline Orchestrator.

This script runs the complete BlueprintPipeline within Isaac Sim environment,
orchestrating all steps from scene loading to episode export.

MUST be run with Isaac Sim's Python:
    /isaac-sim/python.sh tools/run_full_isaacsim_pipeline.py --scene-id kitchen_001

Features:
- Real physics simulation via PhysX
- Actual sensor capture via Replicator
- Validated episode generation
- Multi-camera RGB/depth/segmentation capture
- Quality-filtered exports

Author: BlueprintPipeline Team
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Isaac Sim Initialization (MUST happen before other imports)
# =============================================================================

def initialize_isaac_sim(headless: bool = True) -> bool:
    """
    Initialize Isaac Sim application.

    This must be called before importing any omni modules.

    Args:
        headless: Run without GUI

    Returns:
        True if initialization successful
    """
    print("[PIPELINE] Initializing Isaac Sim...")

    try:
        # Check if we're already in Isaac Sim context
        import omni
        print("[PIPELINE] Isaac Sim already initialized")
        return True
    except ImportError:
        pass

    try:
        # Try to initialize via Isaac Lab's AppLauncher
        from omni.isaac.lab.app import AppLauncher

        # Create launcher with headless config
        launcher = AppLauncher(headless=headless)
        launcher.start()

        print("[PIPELINE] Isaac Sim initialized via AppLauncher")
        return True

    except ImportError:
        print("[PIPELINE] AppLauncher not available, trying direct initialization...")

    try:
        # Direct Isaac Sim initialization
        from omni.isaac.kit import SimulationApp

        config = {
            "headless": headless,
            "width": 1280,
            "height": 720,
            "renderer": "RayTracedLighting",
            "anti_aliasing": 3,
        }

        simulation_app = SimulationApp(config)
        print("[PIPELINE] Isaac Sim initialized via SimulationApp")
        return True

    except ImportError as e:
        print(f"[PIPELINE] ERROR: Failed to initialize Isaac Sim: {e}")
        print("[PIPELINE] Make sure to run with: /isaac-sim/python.sh")
        return False


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for full Isaac Sim pipeline."""

    # Scene
    scene_id: str
    scene_path: Optional[Path] = None

    # Data location
    data_root: Path = Path("/mnt/gcs")
    output_root: Path = Path("/output")

    # Robot
    robot_type: str = "franka"
    robot_usd: str = "/Isaac/Robots/Franka/franka.usd"

    # Episode generation
    episodes_per_variation: int = 10
    max_variations: Optional[int] = None
    fps: float = 30.0

    # Data pack
    data_pack_tier: str = "core"  # core, plus, full
    num_cameras: int = 1
    image_resolution: tuple = (640, 480)

    # Quality
    min_quality_score: float = 0.7
    use_cpgen: bool = True
    use_llm: bool = True

    # Physics
    physics_dt: float = 1.0 / 120.0  # 120Hz physics
    rendering_dt: float = 1.0 / 30.0  # 30Hz rendering

    # Validation
    validate_episodes: bool = True
    validate_physics: bool = True

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create config from environment variables."""
        scene_id = os.environ.get("SCENE_ID", "")
        if not scene_id:
            raise ValueError("SCENE_ID environment variable required")

        return cls(
            scene_id=scene_id,
            data_root=Path(os.environ.get("DATA_ROOT", "/mnt/gcs")),
            output_root=Path(os.environ.get("OUTPUT_ROOT", "/output")),
            robot_type=os.environ.get("ROBOT_TYPE", "franka"),
            episodes_per_variation=int(os.environ.get("EPISODES_PER_VARIATION", "10")),
            max_variations=int(os.environ["MAX_VARIATIONS"]) if os.environ.get("MAX_VARIATIONS") else None,
            fps=float(os.environ.get("FPS", "30")),
            data_pack_tier=os.environ.get("DATA_PACK_TIER", "core"),
            num_cameras=int(os.environ.get("NUM_CAMERAS", "1")),
            image_resolution=tuple(map(int, os.environ.get("IMAGE_RESOLUTION", "640,480").split(","))),
            min_quality_score=float(os.environ.get("MIN_QUALITY_SCORE", "0.7")),
            use_cpgen=os.environ.get("USE_CPGEN", "true").lower() == "true",
            use_llm=os.environ.get("USE_LLM", "true").lower() == "true",
        )


# =============================================================================
# Pipeline Stages
# =============================================================================

class IsaacSimPipeline:
    """Full Isaac Sim pipeline orchestrator."""

    def __init__(self, config: PipelineConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.start_time = time.time()

        # Will be set after Isaac Sim initialization
        self.world = None
        self.physics_context = None
        self.stage = None

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            elapsed = time.time() - self.start_time
            print(f"[PIPELINE] [{elapsed:>7.1f}s] [{level}] {msg}")

    def run(self) -> Dict[str, Any]:
        """Run the full pipeline."""
        results = {
            "scene_id": self.config.scene_id,
            "status": "started",
            "stages": {},
            "start_time": datetime.utcnow().isoformat(),
        }

        try:
            # Stage 1: Load scene
            self.log("=" * 70)
            self.log("STAGE 1: Loading Scene")
            self.log("=" * 70)
            results["stages"]["load_scene"] = self._load_scene()

            # Stage 2: Setup robot
            self.log("=" * 70)
            self.log("STAGE 2: Setting Up Robot")
            self.log("=" * 70)
            results["stages"]["setup_robot"] = self._setup_robot()

            # Stage 3: Configure cameras
            self.log("=" * 70)
            self.log("STAGE 3: Configuring Cameras")
            self.log("=" * 70)
            results["stages"]["setup_cameras"] = self._setup_cameras()

            # Stage 4: Initialize physics
            self.log("=" * 70)
            self.log("STAGE 4: Initializing Physics")
            self.log("=" * 70)
            results["stages"]["init_physics"] = self._init_physics()

            # Stage 5: Generate episodes
            self.log("=" * 70)
            self.log("STAGE 5: Generating Episodes")
            self.log("=" * 70)
            results["stages"]["generate_episodes"] = self._generate_episodes()

            # Stage 6: Export dataset
            self.log("=" * 70)
            self.log("STAGE 6: Exporting Dataset")
            self.log("=" * 70)
            results["stages"]["export"] = self._export_dataset()

            # Stage 7: Validate output
            if self.config.validate_episodes:
                self.log("=" * 70)
                self.log("STAGE 7: Validating Output")
                self.log("=" * 70)
                results["stages"]["validate"] = self._validate_output()

            results["status"] = "completed"
            results["end_time"] = datetime.utcnow().isoformat()
            results["total_time_seconds"] = time.time() - self.start_time

            self.log("=" * 70)
            self.log("PIPELINE COMPLETED SUCCESSFULLY")
            self.log(f"Total time: {results['total_time_seconds']:.1f}s")
            self.log("=" * 70)

        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            self.log(f"Pipeline failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()

        return results

    def _load_scene(self) -> Dict[str, Any]:
        """Load the USD scene."""
        import omni
        from omni.isaac.core import World
        from pxr import Usd, UsdGeom

        # Determine scene path
        if self.config.scene_path:
            scene_path = self.config.scene_path
        else:
            scene_path = self.config.data_root / f"scenes/{self.config.scene_id}/usd/scene.usda"

        self.log(f"Loading scene: {scene_path}")

        if not scene_path.exists():
            raise FileNotFoundError(f"Scene not found: {scene_path}")

        # Create World
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.physics_dt,
            rendering_dt=self.config.rendering_dt,
        )

        # Open stage
        omni.usd.get_context().open_stage(str(scene_path))
        self.stage = omni.usd.get_context().get_stage()

        # Count objects
        prim_count = 0
        mesh_count = 0
        for prim in self.stage.Traverse():
            prim_count += 1
            if prim.IsA(UsdGeom.Mesh):
                mesh_count += 1

        self.log(f"Scene loaded: {prim_count} prims, {mesh_count} meshes")

        return {
            "status": "success",
            "scene_path": str(scene_path),
            "prim_count": prim_count,
            "mesh_count": mesh_count,
        }

    def _setup_robot(self) -> Dict[str, Any]:
        """Set up the robot in the scene."""
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from pxr import UsdGeom, Gf

        robot_prim_path = "/World/Robot"

        # Check if robot already in scene
        robot_prim = self.stage.GetPrimAtPath(robot_prim_path)

        if not robot_prim.IsValid():
            self.log(f"Adding robot from: {self.config.robot_usd}")

            # Add robot reference
            add_reference_to_stage(
                usd_path=self.config.robot_usd,
                prim_path=robot_prim_path,
            )

            # Position robot
            robot_xform = UsdGeom.Xformable(self.stage.GetPrimAtPath(robot_prim_path))
            robot_xform.ClearXformOpOrder()
            robot_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        else:
            self.log("Robot already in scene")

        # Initialize robot in World
        self.robot = self.world.scene.add(
            Robot(
                prim_path=robot_prim_path,
                name=self.config.robot_type,
            )
        )

        self.log(f"Robot initialized: {self.config.robot_type}")

        return {
            "status": "success",
            "robot_type": self.config.robot_type,
            "prim_path": robot_prim_path,
        }

    def _setup_cameras(self) -> Dict[str, Any]:
        """Configure cameras for sensor capture."""
        import omni.replicator.core as rep
        from pxr import UsdGeom, Gf

        cameras = []

        # Camera configurations based on data pack tier
        camera_configs = [
            {
                "name": "wrist",
                "prim_path": "/World/Robot/panda_hand/wrist_camera",
                "resolution": self.config.image_resolution,
                "fov": 60.0,
            },
        ]

        if self.config.num_cameras >= 2:
            camera_configs.append({
                "name": "overhead",
                "prim_path": "/World/Cameras/overhead",
                "position": (0.5, 0.0, 1.5),
                "target": (0.5, 0.0, 0.85),
                "resolution": self.config.image_resolution,
                "fov": 75.0,
            })

        if self.config.num_cameras >= 3:
            camera_configs.append({
                "name": "side",
                "prim_path": "/World/Cameras/side",
                "position": (0.0, 1.0, 1.0),
                "target": (0.5, 0.0, 0.85),
                "resolution": self.config.image_resolution,
                "fov": 60.0,
            })

        if self.config.num_cameras >= 4:
            camera_configs.append({
                "name": "front",
                "prim_path": "/World/Cameras/front",
                "position": (1.5, 0.0, 1.0),
                "target": (0.5, 0.0, 0.85),
                "resolution": self.config.image_resolution,
                "fov": 60.0,
            })

        for cam_config in camera_configs:
            cam_prim = self.stage.GetPrimAtPath(cam_config["prim_path"])

            if not cam_prim.IsValid():
                # Create camera
                self.log(f"Creating camera: {cam_config['name']}")

                cam_prim = UsdGeom.Camera.Define(self.stage, cam_config["prim_path"])

                if "position" in cam_config:
                    xform = UsdGeom.Xformable(cam_prim.GetPrim())
                    xform.AddTranslateOp().Set(Gf.Vec3d(*cam_config["position"]))

            cameras.append({
                "name": cam_config["name"],
                "prim_path": cam_config["prim_path"],
                "resolution": cam_config["resolution"],
            })

        self.cameras = cameras
        self.log(f"Configured {len(cameras)} cameras")

        # Setup Replicator render products
        self.render_products = []
        for cam in cameras:
            rp = rep.create.render_product(
                cam["prim_path"],
                cam["resolution"],
            )
            self.render_products.append(rp)

        self.log("Replicator render products created")

        return {
            "status": "success",
            "cameras": cameras,
            "data_pack_tier": self.config.data_pack_tier,
        }

    def _init_physics(self) -> Dict[str, Any]:
        """Initialize physics simulation."""
        import omni.physx

        # Reset world
        self.world.reset()

        # Get physics context
        self.physics_context = self.world.get_physics_context()

        # Configure physics
        self.physics_context.set_solver_type("TGS")  # Temporal Gauss-Seidel
        self.physics_context.set_gravity(value=(0.0, 0.0, -9.81))

        # Warm up physics
        self.log("Warming up physics simulation...")
        for _ in range(10):
            self.world.step(render=False)

        self.log("Physics initialized")

        return {
            "status": "success",
            "physics_dt": self.config.physics_dt,
            "solver": "TGS",
        }

    def _generate_episodes(self) -> Dict[str, Any]:
        """Generate episodes with real physics and sensor capture."""
        # Import episode generation modules
        from episode_generation_job.generate_episodes import (
            EpisodeGenerator,
            EpisodeGenerationConfig,
        )
        from episode_generation_job.sensor_data_capture import (
            DataPackTier,
            SensorDataCaptureMode,
            create_sensor_capture,
        )

        # Load scene manifest
        manifest_path = self.config.data_root / f"scenes/{self.config.scene_id}/assets/scene_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.log(f"Loaded manifest: {len(manifest.get('objects', []))} objects")

        # Create sensor capture with real Isaac Sim integration
        tier = DataPackTier[self.config.data_pack_tier.upper()]
        camera_specs = [
            {
                "prim_path": camera["prim_path"],
                "camera_type": camera["name"],
                "camera_id": camera["name"],
            }
            for camera in self.cameras
        ]

        sensor_capture = create_sensor_capture(
            data_pack=tier,
            num_cameras=self.config.num_cameras,
            resolution=self.config.image_resolution,
            fps=self.config.fps,
            camera_specs=camera_specs,
            scene_usd_path=str(self.config.data_root / f"scenes/{self.config.scene_id}/scene.usd"),
            capture_mode=SensorDataCaptureMode.ISAAC_SIM,
            allow_mock_capture=False,
            verbose=self.verbose,
        )

        # Create episode generator config
        output_dir = self.config.output_root / f"scenes/{self.config.scene_id}/episodes"
        output_dir.mkdir(parents=True, exist_ok=True)

        gen_config = EpisodeGenerationConfig(
            scene_id=self.config.scene_id,
            manifest_path=manifest_path,
            robot_type=self.config.robot_type,
            episodes_per_variation=self.config.episodes_per_variation,
            max_variations=self.config.max_variations,
            fps=self.config.fps,
            use_llm=self.config.use_llm,
            use_cpgen=self.config.use_cpgen,
            min_quality_score=self.config.min_quality_score,
            data_pack_tier=self.config.data_pack_tier,
            num_cameras=self.config.num_cameras,
            image_resolution=self.config.image_resolution,
            capture_sensor_data=True,
            use_mock_capture=False,  # Use real capture!
            output_dir=output_dir,
        )

        # Run generation
        generator = EpisodeGenerator(gen_config, verbose=self.verbose)
        generator.sensor_capture = sensor_capture  # Inject real sensor capture
        generator.world = self.world  # Inject World for physics

        output = generator.generate(manifest)

        self.generation_output = output

        return {
            "status": "success" if output.success else "failed",
            "total_episodes": output.total_episodes,
            "valid_episodes": output.valid_episodes,
            "pass_rate": output.pass_rate,
            "average_quality_score": output.average_quality_score,
            "total_frames": output.total_frames,
            "output_dir": str(output_dir),
            "errors": output.errors,
        }

    def _export_dataset(self) -> Dict[str, Any]:
        """Export the generated dataset."""
        if not hasattr(self, "generation_output"):
            return {"status": "skipped", "reason": "No generation output"}

        output = self.generation_output

        self.log(f"Dataset exported to: {output.lerobot_dataset_path}")
        self.log(f"Manifest: {output.manifest_path}")

        if output.validation_report_path:
            self.log(f"Validation report: {output.validation_report_path}")

        return {
            "status": "success",
            "lerobot_dataset": str(output.lerobot_dataset_path),
            "manifest": str(output.manifest_path),
            "validation_report": str(output.validation_report_path) if output.validation_report_path else None,
        }

    def _validate_output(self) -> Dict[str, Any]:
        """Validate the generated output."""
        output = self.generation_output

        checks = []

        # Check episode count
        if output.valid_episodes > 0:
            checks.append(("episode_count", True, f"{output.valid_episodes} valid episodes"))
        else:
            checks.append(("episode_count", False, "No valid episodes"))

        # Check quality score
        if output.average_quality_score >= self.config.min_quality_score:
            checks.append(("quality_score", True, f"Avg score: {output.average_quality_score:.2f}"))
        else:
            checks.append(("quality_score", False, f"Low quality: {output.average_quality_score:.2f}"))

        # Check pass rate
        if output.pass_rate >= 0.5:
            checks.append(("pass_rate", True, f"Pass rate: {output.pass_rate:.1%}"))
        else:
            checks.append(("pass_rate", False, f"Low pass rate: {output.pass_rate:.1%}"))

        # Check output files
        if output.lerobot_dataset_path and output.lerobot_dataset_path.exists():
            checks.append(("output_files", True, "Dataset files exist"))
        else:
            checks.append(("output_files", False, "Missing output files"))

        all_passed = all(check[1] for check in checks)

        for name, passed, msg in checks:
            status = "PASS" if passed else "FAIL"
            self.log(f"  [{status}] {name}: {msg}")

        return {
            "status": "success" if all_passed else "failed",
            "checks": [{"name": c[0], "passed": c[1], "message": c[2]} for c in checks],
            "all_passed": all_passed,
        }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run full Isaac Sim pipeline for episode generation"
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default=os.environ.get("SCENE_ID", ""),
        help="Scene identifier",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "/mnt/gcs")),
        help="Root path for scene data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("OUTPUT_ROOT", "/output")),
        help="Root path for output",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default=os.environ.get("ROBOT_TYPE", "franka"),
        choices=["franka", "ur10", "fetch"],
        help="Robot type",
    )
    parser.add_argument(
        "--data-pack",
        type=str,
        default=os.environ.get("DATA_PACK_TIER", "core"),
        choices=["core", "plus", "full"],
        help="Data pack tier",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(os.environ.get("EPISODES_PER_VARIATION", "10")),
        help="Episodes per variation",
    )
    parser.add_argument(
        "--cameras",
        type=int,
        default=int(os.environ.get("NUM_CAMERAS", "1")),
        help="Number of cameras",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=os.environ.get("HEADLESS", "1") == "1",
        help="Run in headless mode",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )

    args = parser.parse_args()

    if not args.scene_id:
        print("ERROR: --scene-id is required")
        sys.exit(1)

    # Initialize Isaac Sim FIRST
    if not initialize_isaac_sim(headless=args.headless):
        print("ERROR: Failed to initialize Isaac Sim")
        print("Make sure to run with: /isaac-sim/python.sh")
        sys.exit(1)

    # Create config
    config = PipelineConfig(
        scene_id=args.scene_id,
        data_root=args.data_root,
        output_root=args.output_root,
        robot_type=args.robot_type,
        data_pack_tier=args.data_pack,
        episodes_per_variation=args.episodes,
        num_cameras=args.cameras,
    )

    # Run pipeline
    pipeline = IsaacSimPipeline(config, verbose=args.verbose)
    results = pipeline.run()

    # Write results
    results_path = args.output_root / f"scenes/{args.scene_id}/pipeline_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults written to: {results_path}")

    # Exit with appropriate code
    if results["status"] == "completed":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
