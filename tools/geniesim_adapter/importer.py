"""
Genie Sim 3.0 Importer for BlueprintPipeline.

Imports Genie Sim output (episodes, datasets, trajectories) back into BlueprintPipeline
for further processing, fine-tuning, or archival.

Usage:
    from tools.geniesim_adapter import GenieSimImporter, GenieSimImportConfig

    config = GenieSimImportConfig(
        output_format="lerobot",
        validate_schema=True,
    )
    importer = GenieSimImporter(config)
    result = importer.import_dataset(geniesim_dir, blueprint_output_dir)
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenieSimImportConfig:
    """Configuration for Genie Sim import."""

    # Input validation
    validate_schema: bool = True
    validate_trajectories: bool = True

    # Output format
    output_format: str = "lerobot"  # lerobot, huggingface, custom
    include_metadata: bool = True
    include_raw_episodes: bool = False

    # Processing options
    merge_robot_variants: bool = False  # Merge data from multiple robot types
    sample_episodes: Optional[int] = None  # Limit to N episodes for testing

    # Storage
    copy_raw_data: bool = True
    compress_output: bool = False

    # Validation
    check_episode_counts: bool = True
    check_trajectory_validity: bool = True


@dataclass
class GenieSimImportResult:
    """Result of Genie Sim import."""

    success: bool
    scene_id: str
    output_dir: Path

    # Output files
    dataset_manifest_path: Optional[Path] = None
    episode_index_path: Optional[Path] = None
    metadata_path: Optional[Path] = None

    # Statistics
    num_episodes: int = 0
    num_trajectories: int = 0
    total_frames: int = 0
    robots_processed: List[str] = field(default_factory=list)

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "scene_id": self.scene_id,
            "output_dir": str(self.output_dir),
            "outputs": {
                "dataset_manifest": str(self.dataset_manifest_path) if self.dataset_manifest_path else None,
                "episode_index": str(self.episode_index_path) if self.episode_index_path else None,
                "metadata": str(self.metadata_path) if self.metadata_path else None,
            },
            "statistics": {
                "episodes": self.num_episodes,
                "trajectories": self.num_trajectories,
                "total_frames": self.total_frames,
                "robots": self.robots_processed,
            },
            "errors": self.errors,
            "warnings": self.warnings,
        }


class GenieSimImporter:
    """
    Main importer for converting Genie Sim output back to BlueprintPipeline format.

    Handles:
    1. Episode data from Genie Sim data collection
    2. Trajectory data from cuRobo planning
    3. LeRobot-formatted datasets
    4. Scene metadata and annotations
    """

    def __init__(self, config: Optional[GenieSimImportConfig] = None, verbose: bool = True):
        """
        Initialize importer.

        Args:
            config: Import configuration
            verbose: Print progress
        """
        self.config = config or GenieSimImportConfig()
        self.verbose = verbose

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[GENIESIM-IMPORTER] [{level}] {msg}")

    def import_dataset(
        self,
        geniesim_dir: Path,
        output_dir: Path,
    ) -> GenieSimImportResult:
        """
        Import Genie Sim dataset back to BlueprintPipeline format.

        Args:
            geniesim_dir: Directory containing Genie Sim output
            output_dir: Output directory for BlueprintPipeline format

        Returns:
            GenieSimImportResult with paths and statistics
        """
        self.log("=" * 70)
        self.log("Genie Sim 3.0 Import")
        self.log("=" * 70)
        self.log(f"Input: {geniesim_dir}")
        self.log(f"Output: {output_dir}")

        result = GenieSimImportResult(
            success=False,
            scene_id="",
            output_dir=output_dir,
        )

        try:
            # Validate input directory
            geniesim_dir = Path(geniesim_dir)
            if not geniesim_dir.exists():
                raise ValueError(f"Input directory not found: {geniesim_dir}")

            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load metadata
            self.log("Loading metadata...")
            metadata = self._load_metadata(geniesim_dir)
            result.scene_id = metadata.get("scene_id", "imported_scene")

            # Import episodes
            self.log("Importing episodes...")
            episodes = self._import_episodes(geniesim_dir, output_dir, result)

            # Import trajectories
            self.log("Importing trajectories...")
            trajectories = self._import_trajectories(geniesim_dir, output_dir, result)

            # Create dataset manifest
            self.log("Creating dataset manifest...")
            manifest_path = self._create_dataset_manifest(
                output_dir,
                result.scene_id,
                metadata,
                episodes,
                trajectories,
            )
            result.dataset_manifest_path = manifest_path

            # Create episode index
            self.log("Creating episode index...")
            index_path = self._create_episode_index(output_dir, episodes)
            result.episode_index_path = index_path

            # Copy metadata
            if self.config.include_metadata:
                self.log("Saving metadata...")
                metadata_path = output_dir / "import_metadata.json"
                self._save_metadata(metadata_path, metadata, result)
                result.metadata_path = metadata_path

            # Validate output
            if self.config.validate_schema:
                self.log("Validating output...")
                self._validate_output(output_dir, result)

            result.success = True
            self.log("Import complete!")
            self.log(f"Imported {result.num_episodes} episodes with {result.total_frames} frames")

        except Exception as e:
            error_msg = f"Import failed: {str(e)}"
            self.log(error_msg, "ERROR")
            result.errors.append(error_msg)

        return result

    def _load_metadata(self, geniesim_dir: Path) -> Dict[str, Any]:
        """Load metadata from Genie Sim output."""
        metadata_path = geniesim_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {
            "scene_id": geniesim_dir.name,
            "import_timestamp": datetime.utcnow().isoformat() + "Z",
            "geniesim_version": "3.0",
        }

    def _import_episodes(
        self,
        geniesim_dir: Path,
        output_dir: Path,
        result: GenieSimImportResult,
    ) -> List[Dict[str, Any]]:
        """Import episode data from Genie Sim output."""
        episodes = []
        episodes_dir = geniesim_dir / "episodes"

        if not episodes_dir.exists():
            self.log(f"No episodes directory found: {episodes_dir}", "WARNING")
            result.warnings.append(f"Missing episodes directory: {episodes_dir}")
            return episodes

        # Create output episodes directory
        output_episodes_dir = output_dir / "episodes"
        output_episodes_dir.mkdir(parents=True, exist_ok=True)

        # Process each robot variant
        for robot_dir in sorted(episodes_dir.iterdir()):
            if not robot_dir.is_dir():
                continue

            robot_type = robot_dir.name
            self.log(f"Processing episodes for robot: {robot_type}")

            # Process episodes
            for episode_file in sorted(robot_dir.glob("episode_*.json")):
                if self.config.sample_episodes and len(episodes) >= self.config.sample_episodes:
                    break

                try:
                    episode_data = self._process_episode(episode_file, robot_type)
                    episodes.append(episode_data)
                    result.num_trajectories += episode_data.get("trajectory_count", 0)
                    result.total_frames += episode_data.get("frame_count", 0)

                    # Copy episode data
                    if self.config.copy_raw_data:
                        output_robot_dir = output_episodes_dir / robot_type
                        output_robot_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy(episode_file, output_robot_dir / episode_file.name)

                except Exception as e:
                    error_msg = f"Failed to process {episode_file.name}: {str(e)}"
                    self.log(error_msg, "ERROR")
                    result.errors.append(error_msg)

            result.robots_processed.append(robot_type)

        result.num_episodes = len(episodes)
        self.log(f"Imported {result.num_episodes} episodes")
        return episodes

    def _import_trajectories(
        self,
        geniesim_dir: Path,
        output_dir: Path,
        result: GenieSimImportResult,
    ) -> List[Dict[str, Any]]:
        """Import trajectory data from Genie Sim output."""
        trajectories = []
        trajectories_dir = geniesim_dir / "trajectories"

        if not trajectories_dir.exists():
            self.log(f"No trajectories directory found", "WARNING")
            return trajectories

        # Create output trajectories directory
        output_trajectories_dir = output_dir / "trajectories"
        output_trajectories_dir.mkdir(parents=True, exist_ok=True)

        # Process trajectory files
        for traj_file in sorted(trajectories_dir.glob("trajectory_*.json")):
            try:
                traj_data = self._process_trajectory(traj_file)
                trajectories.append(traj_data)

                # Copy trajectory data
                if self.config.copy_raw_data:
                    shutil.copy(traj_file, output_trajectories_dir / traj_file.name)

            except Exception as e:
                error_msg = f"Failed to process {traj_file.name}: {str(e)}"
                self.log(error_msg, "ERROR")
                result.errors.append(error_msg)

        return trajectories

    def _process_episode(self, episode_file: Path, robot_type: str) -> Dict[str, Any]:
        """Process a single episode file."""
        with open(episode_file) as f:
            episode_data = json.load(f)

        # Validate and normalize episode structure
        return {
            "episode_id": episode_data.get("episode_id", episode_file.stem),
            "robot_type": robot_type,
            "trajectory_count": len(episode_data.get("trajectories", [])),
            "frame_count": sum(
                len(t.get("frames", [])) for t in episode_data.get("trajectories", [])
            ),
            "task_type": episode_data.get("task_type", "unknown"),
            "success": episode_data.get("success", False),
            "metadata": episode_data.get("metadata", {}),
        }

    def _process_trajectory(self, trajectory_file: Path) -> Dict[str, Any]:
        """Process a single trajectory file."""
        with open(trajectory_file) as f:
            traj_data = json.load(f)

        return {
            "trajectory_id": traj_data.get("trajectory_id", trajectory_file.stem),
            "frame_count": len(traj_data.get("frames", [])),
            "duration": traj_data.get("duration", 0),
            "planner": traj_data.get("planner", "unknown"),
            "success": traj_data.get("success", False),
        }

    def _create_dataset_manifest(
        self,
        output_dir: Path,
        scene_id: str,
        metadata: Dict[str, Any],
        episodes: List[Dict[str, Any]],
        trajectories: List[Dict[str, Any]],
    ) -> Path:
        """Create dataset manifest file."""
        manifest = {
            "scene_id": scene_id,
            "import_timestamp": datetime.utcnow().isoformat() + "Z",
            "format": "blueprint_pipeline",
            "source": "geniesim_3.0",
            "statistics": {
                "episodes": len(episodes),
                "trajectories": len(trajectories),
            },
            "episodes": episodes,
            "trajectories": trajectories,
            "metadata": metadata,
        }

        manifest_path = output_dir / "dataset_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest_path

    def _create_episode_index(
        self,
        output_dir: Path,
        episodes: List[Dict[str, Any]],
    ) -> Path:
        """Create episode index for quick lookups."""
        index = {
            "total_episodes": len(episodes),
            "by_robot": {},
            "by_task": {},
            "episodes": {},
        }

        # Organize by robot and task
        for episode in episodes:
            robot = episode.get("robot_type", "unknown")
            task = episode.get("task_type", "unknown")
            episode_id = episode.get("episode_id", "unknown")

            if robot not in index["by_robot"]:
                index["by_robot"][robot] = []
            index["by_robot"][robot].append(episode_id)

            if task not in index["by_task"]:
                index["by_task"][task] = []
            index["by_task"][task].append(episode_id)

            index["episodes"][episode_id] = {
                "robot": robot,
                "task": task,
                "frames": episode.get("frame_count", 0),
            }

        index_path = output_dir / "episode_index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        return index_path

    def _save_metadata(
        self,
        metadata_path: Path,
        metadata: Dict[str, Any],
        result: GenieSimImportResult,
    ) -> None:
        """Save import metadata."""
        full_metadata = {
            **metadata,
            "import_config": {
                "output_format": self.config.output_format,
                "validate_schema": self.config.validate_schema,
                "copy_raw_data": self.config.copy_raw_data,
            },
            "import_result": result.to_dict(),
        }

        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)

    def _validate_output(self, output_dir: Path, result: GenieSimImportResult) -> None:
        """Validate imported output."""
        # Check required files
        required_files = ["dataset_manifest.json", "episode_index.json"]
        for filename in required_files:
            if not (output_dir / filename).exists():
                msg = f"Missing required file: {filename}"
                result.warnings.append(msg)
                self.log(msg, "WARNING")

        # Validate JSON files
        for json_file in output_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                msg = f"Invalid JSON in {json_file.name}: {str(e)}"
                result.errors.append(msg)
                self.log(msg, "ERROR")
