"""
Replicator Generator - Creates Replicator YAML configurations for SDG.

This module generates Omniverse Replicator configurations for synthetic
data generation, including:
- Camera setups
- Domain randomization
- Annotation writers
- Batch generation configs

Outputs YAML files compatible with Replicator's portable workflow.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from .randomizers import RandomizerConfig, get_randomizers_for_policy
from .writers import WriterConfig, get_writers_for_policy


@dataclass
class CameraConfig:
    """Configuration for a render camera."""

    name: str
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    focal_length: float = 24.0
    aperture: tuple[float, float] = (36.0, 24.0)
    clipping_range: tuple[float, float] = (0.1, 1000.0)
    resolution: tuple[int, int] = (1280, 720)


@dataclass
class ReplicatorConfig:
    """Complete Replicator configuration."""

    name: str
    scene_path: str
    num_frames: int
    output_dir: str
    cameras: list[CameraConfig] = field(default_factory=list)
    randomizers: list[RandomizerConfig] = field(default_factory=list)
    writers: list[WriterConfig] = field(default_factory=list)
    seed: Optional[int] = None
    headless: bool = True
    use_gpu: bool = True


class ReplicatorGenerator:
    """
    Generates Replicator YAML configurations from recipes.

    Usage:
        generator = ReplicatorGenerator(policy_config)
        config = generator.generate(recipe, policy_id="dexterous_pick_place")
        generator.save(config, "replicator/")
    """

    def __init__(self, policy_config: dict[str, Any]):
        """
        Initialize generator with policy configuration.

        Args:
            policy_config: Policy configuration from environment_policies.json
        """
        self.policy_config = policy_config
        self.policies = policy_config.get("policies", {})
        self.environments = policy_config.get("environments", {})

    def generate(
        self,
        recipe: dict[str, Any],
        policy_id: str,
        num_frames: int = 1000,
        output_dir: str = "./replicator_output",
        cameras: Optional[list[CameraConfig]] = None,
    ) -> ReplicatorConfig:
        """
        Generate Replicator configuration for a recipe and policy.

        Args:
            recipe: Scene recipe (recipe.json content)
            policy_id: ID of the training policy
            num_frames: Number of frames to generate
            output_dir: Output directory for generated data
            cameras: Custom camera configurations (auto-generated if None)

        Returns:
            ReplicatorConfig ready for export
        """
        env_type = recipe.get("metadata", {}).get("environment_type", "unknown")
        scene_path = recipe.get("metadata", {}).get("scene_path", "scene.usda")

        policy = self.policies.get(policy_id, {})

        if cameras is None:
            cameras = self._generate_cameras(recipe, env_type)

        randomizers = get_randomizers_for_policy(
            policy, recipe, self.environments.get(env_type, {})
        )

        capture_config = policy.get("capture_config", {})
        writers = get_writers_for_policy(capture_config, output_dir)

        return ReplicatorConfig(
            name=f"{env_type}_{policy_id}",
            scene_path=scene_path,
            num_frames=num_frames,
            output_dir=output_dir,
            cameras=cameras,
            randomizers=randomizers,
            writers=writers,
            seed=recipe.get("metadata", {}).get("deterministic_seed"),
            headless=True,
            use_gpu=True,
        )

    def _generate_cameras(
        self, recipe: dict[str, Any], env_type: str
    ) -> list[CameraConfig]:
        """Generate default camera configurations based on scene."""
        room = recipe.get("room", {})
        dims = room.get("dimensions", {})

        width = dims.get("width", 5.0)
        depth = dims.get("depth", 5.0)
        height = dims.get("height", 3.0)

        cameras = []

        cameras.append(
            CameraConfig(
                name="overview",
                position=(width * 0.8, height * 0.7, depth * 0.8),
                target=(0, height * 0.3, 0),
                focal_length=24.0,
            )
        )

        corners = [
            ("corner_front_left", (-width * 0.4, height * 0.5, -depth * 0.4)),
            ("corner_front_right", (width * 0.4, height * 0.5, -depth * 0.4)),
            ("corner_back_left", (-width * 0.4, height * 0.5, depth * 0.4)),
            ("corner_back_right", (width * 0.4, height * 0.5, depth * 0.4)),
        ]

        for name, pos in corners:
            cameras.append(
                CameraConfig(
                    name=name,
                    position=pos,
                    target=(0, height * 0.3, 0),
                    focal_length=35.0,
                )
            )

        cameras.append(
            CameraConfig(
                name="robot_eye",
                position=(0, 1.2, -depth * 0.3),
                target=(0, 0.9, depth * 0.2),
                focal_length=50.0,
                resolution=(1920, 1080),
            )
        )

        return cameras

    def save(self, config: ReplicatorConfig, output_dir: str) -> dict[str, str]:
        """
        Save Replicator configuration to YAML files.

        Args:
            config: ReplicatorConfig to save
            output_dir: Directory to save files

        Returns:
            Dictionary mapping config names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        dataset_config = self._build_dataset_yaml(config)
        dataset_path = output_path / "dataset.yaml"
        with open(dataset_path, "w") as f:
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
        saved_files["dataset"] = str(dataset_path)

        cameras_config = self._build_cameras_yaml(config.cameras)
        cameras_path = output_path / "cameras.yaml"
        with open(cameras_path, "w") as f:
            yaml.dump(cameras_config, f, default_flow_style=False, sort_keys=False)
        saved_files["cameras"] = str(cameras_path)

        randomizers_config = self._build_randomizers_yaml(config.randomizers)
        randomizers_path = output_path / "randomizations.yaml"
        with open(randomizers_path, "w") as f:
            yaml.dump(randomizers_config, f, default_flow_style=False, sort_keys=False)
        saved_files["randomizations"] = str(randomizers_path)

        writers_config = self._build_writers_yaml(config.writers, config.output_dir)
        writers_path = output_path / "writers.yaml"
        with open(writers_path, "w") as f:
            yaml.dump(writers_config, f, default_flow_style=False, sort_keys=False)
        saved_files["writers"] = str(writers_path)

        script_content = self._build_generation_script(config, output_path)
        script_path = output_path / "run_generation.py"
        with open(script_path, "w") as f:
            f.write(script_content)
        saved_files["script"] = str(script_path)

        return saved_files

    def _build_dataset_yaml(self, config: ReplicatorConfig) -> dict[str, Any]:
        """Build main dataset YAML configuration."""
        return {
            "version": "1.0",
            "metadata": {
                "name": config.name,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "generator": "BlueprintRecipe",
            },
            "scene": {
                "path": config.scene_path,
                "up_axis": "Y",
                "meters_per_unit": 1.0,
            },
            "generation": {
                "num_frames": config.num_frames,
                "seed": config.seed,
                "headless": config.headless,
                "use_gpu": config.use_gpu,
            },
            "output": {
                "directory": config.output_dir,
                "format": "replicator_default",
            },
        }

    def _build_cameras_yaml(self, cameras: list[CameraConfig]) -> dict[str, Any]:
        """Build cameras YAML configuration."""
        camera_configs = []
        for cam in cameras:
            camera_configs.append(
                {
                    "name": cam.name,
                    "type": "perspective",
                    "position": list(cam.position),
                    "look_at": list(cam.target),
                    "focal_length": cam.focal_length,
                    "horizontal_aperture": cam.aperture[0],
                    "vertical_aperture": cam.aperture[1],
                    "clipping_range": list(cam.clipping_range),
                    "resolution": list(cam.resolution),
                }
            )

        return {
            "version": "1.0",
            "cameras": camera_configs,
            "render_products": [
                {"camera": cam.name, "resolution": list(cam.resolution)}
                for cam in cameras
            ],
        }

    def _build_randomizers_yaml(self, randomizers: list[RandomizerConfig]) -> dict[str, Any]:
        """Build randomizers YAML configuration."""
        randomizer_configs = []
        for rand in randomizers:
            randomizer_configs.append(
                {
                    "name": rand.name,
                    "type": rand.randomizer_type,
                    "enabled": rand.enabled,
                    "frequency": rand.frequency,
                    "targets": rand.targets,
                    "parameters": rand.parameters,
                }
            )

        return {
            "version": "1.0",
            "randomizers": randomizer_configs,
        }

    def _build_writers_yaml(
        self, writers: list[WriterConfig], output_dir: str
    ) -> dict[str, Any]:
        """Build writers YAML configuration."""
        writer_configs = []
        for writer in writers:
            writer_configs.append(
                {
                    "name": writer.name,
                    "type": writer.writer_type,
                    "enabled": writer.enabled,
                    "output_dir": f"{output_dir}/{writer.name}",
                    "file_format": writer.file_format,
                    "parameters": writer.parameters,
                }
            )

        return {
            "version": "1.0",
            "writers": writer_configs,
            "annotators": self._get_required_annotators(writers),
        }

    def _get_required_annotators(self, writers: list[WriterConfig]) -> list[dict[str, Any]]:
        """Determine required annotators from writers."""
        annotators = []
        seen = set()

        for writer in writers:
            if writer.annotator and writer.annotator not in seen:
                annotators.append({"name": writer.annotator, "type": writer.annotator})
                seen.add(writer.annotator)

        return annotators

    def _build_generation_script(
        self, config: ReplicatorConfig, output_path: Path
    ) -> str:
        """Build Python script for running generation."""
        return f'''#!/usr/bin/env python3
"""
Replicator Generation Script - {config.name}
Generated by BlueprintRecipe

Run with Isaac Sim:
    ./python.sh run_generation.py
"""

import omni.replicator.core as rep
import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent
DATASET_CONFIG = CONFIG_DIR / "dataset.yaml"
CAMERAS_CONFIG = CONFIG_DIR / "cameras.yaml"
RANDOMIZERS_CONFIG = CONFIG_DIR / "randomizations.yaml"
WRITERS_CONFIG = CONFIG_DIR / "writers.yaml"


def load_config(path):
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def setup_scene(scene_config):
    """Load and setup the scene."""
    rep.open_stage(scene_config["path"])


def setup_cameras(cameras_config):
    """Setup render cameras."""
    cameras = []
    for cam_cfg in cameras_config["cameras"]:
        with rep.new_layer(layer_name=f"Camera_{cam_cfg['name']}"):
            camera = rep.create.camera(
                position=cam_cfg["position"],
                look_at=cam_cfg["look_at"],
                focal_length=cam_cfg["focal_length"],
            )
            cameras.append(camera)
    return cameras


def setup_randomizers(randomizers_config, scene_objects):
    """Setup domain randomizers."""
    for rand_cfg in randomizers_config["randomizers"]:
        if not rand_cfg.get("enabled", True):
            continue

        rand_type = rand_cfg["type"]

        if rand_type == "object_scatter":
            with rep.trigger.on_frame(num_frames={config.num_frames}):
                rep.randomizer.scatter(
                    scene_objects,
                    scatter_plane=rep.get.prims(path_pattern="/Objects/*"),
                    **rand_cfg.get("parameters", {{}}),
                )

        elif rand_type == "material_variation":
            with rep.trigger.on_frame():
                rep.randomizer.materials(
                    scene_objects,
                    **rand_cfg.get("parameters", {{}}),
                )

        elif rand_type == "lighting_variation":
            with rep.trigger.on_frame():
                lights = rep.get.prims(path_pattern="/Lights/*")
                rep.modify.pose(
                    lights,
                    rotation=rep.distribution.uniform((0, 0, 0), (0, 360, 0)),
                )


def setup_writers(writers_config):
    """Setup annotation writers."""
    writers = []
    for writer_cfg in writers_config["writers"]:
        if not writer_cfg.get("enabled", True):
            continue

        writer_type = writer_cfg["type"]
        output_dir = writer_cfg["output_dir"]

        if writer_type == "rgb":
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=output_dir, rgb=True)
        elif writer_type == "semantic_segmentation":
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=output_dir, semantic_segmentation=True)
        elif writer_type == "instance_segmentation":
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=output_dir, instance_segmentation=True)
        elif writer_type == "depth":
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=output_dir, distance_to_camera=True)
        elif writer_type == "bounding_box_2d":
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=output_dir, bounding_box_2d_tight=True)
        elif writer_type == "bounding_box_3d":
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=output_dir, bounding_box_3d=True)

        writers.append(writer)

    return writers


def main():
    """Main generation loop."""
    dataset_cfg = load_config(DATASET_CONFIG)
    cameras_cfg = load_config(CAMERAS_CONFIG)
    randomizers_cfg = load_config(RANDOMIZERS_CONFIG)
    writers_cfg = load_config(WRITERS_CONFIG)

    setup_scene(dataset_cfg["scene"])

    scene_objects = rep.get.prims(path_pattern="/Objects/*")

    cameras = setup_cameras(cameras_cfg)
    setup_randomizers(randomizers_cfg, scene_objects)
    writers = setup_writers(writers_cfg)

    for camera in cameras:
        render_product = rep.create.render_product(
            camera, resolution={config.cameras[0].resolution if config.cameras else (1280, 720)}
        )
        for writer in writers:
            writer.attach([render_product])

    num_frames = dataset_cfg["generation"]["num_frames"]
    rep.orchestrator.run_until_complete(num_frames=num_frames)

    print(f"Generation complete: {num_frames} frames generated")


if __name__ == "__main__":
    main()
'''

