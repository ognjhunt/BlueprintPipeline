"""
Genie Sim 3.0 Exporter for BlueprintPipeline.

Main entry point for exporting BlueprintPipeline scenes to Genie Sim format.
Coordinates scene graph conversion, asset registration, and task configuration.

Usage:
    from tools.geniesim_adapter import GenieSimExporter, GenieSimExportConfig

    config = GenieSimExportConfig(
        robot_type="franka",
        generate_embeddings=False,
    )
    exporter = GenieSimExporter(config)
    result = exporter.export(manifest_path, output_dir)

References:
    - Genie Sim 3.0 Paper: https://arxiv.org/html/2601.02078v1
    - Integration Spec: docs/GENIESIM_INTEGRATION.md
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .scene_graph import SceneGraphConverter, GenieSimSceneGraph
from .asset_index import AssetIndexBuilder, GenieSimAssetIndex
from .task_config import TaskConfigGenerator, GenieSimTaskConfig


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GenieSimExportConfig:
    """Configuration for Genie Sim export."""

    # Robot configuration
    robot_type: str = "franka"  # franka, g2, ur10
    urdf_path: Optional[str] = None

    # Export options
    generate_embeddings: bool = False
    embedding_model: str = "qwen-text-embedding-v4"
    max_tasks: int = 50

    # USD handling
    copy_usd_files: bool = True  # Copy USD files to output directory
    usd_relative_paths: bool = True  # Use relative paths in output

    # Commercial filtering
    filter_commercial_only: bool = False  # Only include commercially-usable assets

    # Output options
    pretty_json: bool = True
    include_metadata: bool = True


@dataclass
class GenieSimExportResult:
    """Result of Genie Sim export."""

    success: bool
    scene_id: str
    output_dir: Path

    # Output files
    scene_graph_path: Optional[Path] = None
    asset_index_path: Optional[Path] = None
    task_config_path: Optional[Path] = None
    scene_config_path: Optional[Path] = None

    # Statistics
    num_nodes: int = 0
    num_edges: int = 0
    num_assets: int = 0
    num_tasks: int = 0

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "scene_id": self.scene_id,
            "output_dir": str(self.output_dir),
            "outputs": {
                "scene_graph": str(self.scene_graph_path) if self.scene_graph_path else None,
                "asset_index": str(self.asset_index_path) if self.asset_index_path else None,
                "task_config": str(self.task_config_path) if self.task_config_path else None,
                "scene_config": str(self.scene_config_path) if self.scene_config_path else None,
            },
            "statistics": {
                "nodes": self.num_nodes,
                "edges": self.num_edges,
                "assets": self.num_assets,
                "tasks": self.num_tasks,
            },
            "errors": self.errors,
            "warnings": self.warnings,
        }


# =============================================================================
# Main Exporter
# =============================================================================


class GenieSimExporter:
    """
    Main exporter for converting BlueprintPipeline scenes to Genie Sim format.

    This exporter creates:
    1. scene_graph.json - Scene graph with nodes and edges
    2. asset_index.json - Asset metadata for RAG retrieval
    3. task_config.json - Task generation hints
    4. scene_config.yaml - Genie Sim scene configuration

    Usage:
        config = GenieSimExportConfig(robot_type="franka")
        exporter = GenieSimExporter(config)
        result = exporter.export(manifest_path, output_dir)
    """

    def __init__(self, config: Optional[GenieSimExportConfig] = None, verbose: bool = True):
        """
        Initialize exporter.

        Args:
            config: Export configuration
            verbose: Print progress
        """
        self.config = config or GenieSimExportConfig()
        self.verbose = verbose

        # Initialize converters
        self.scene_graph_converter = SceneGraphConverter(verbose=verbose)
        self.asset_index_builder = AssetIndexBuilder(
            generate_embeddings=self.config.generate_embeddings,
            embedding_model=self.config.embedding_model,
            verbose=verbose,
        )
        self.task_config_generator = TaskConfigGenerator(verbose=verbose)

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[GENIESIM-EXPORTER] [{level}] {msg}")

    def export(
        self,
        manifest_path: Path,
        output_dir: Path,
        usd_source_dir: Optional[Path] = None,
    ) -> GenieSimExportResult:
        """
        Export BlueprintPipeline scene to Genie Sim format.

        Args:
            manifest_path: Path to scene_manifest.json
            output_dir: Output directory for Genie Sim files
            usd_source_dir: Optional source directory for USD files

        Returns:
            GenieSimExportResult with paths to generated files
        """
        self.log("=" * 70)
        self.log("Genie Sim 3.0 Export")
        self.log("=" * 70)
        self.log(f"Input: {manifest_path}")
        self.log(f"Output: {output_dir}")

        result = GenieSimExportResult(
            success=False,
            scene_id="",
            output_dir=output_dir,
        )

        try:
            # Load manifest
            self.log("Loading manifest...")
            with open(manifest_path) as f:
                manifest = json.load(f)

            scene_id = manifest.get("scene_id", "unknown")
            result.scene_id = scene_id
            self.log(f"Scene: {scene_id}")

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Determine USD base path
            if usd_source_dir:
                usd_base_path = str(usd_source_dir)
            elif self.config.usd_relative_paths:
                usd_base_path = "../usd"  # Relative to geniesim output
            else:
                usd_base_path = str(manifest_path.parent / "usd")

            # Step 1: Convert scene graph
            self.log("\nStep 1: Converting scene graph...")
            scene_graph = self.scene_graph_converter.convert(manifest, usd_base_path)
            result.num_nodes = len(scene_graph.nodes)
            result.num_edges = len(scene_graph.edges)
            self.log(f"  Nodes: {result.num_nodes}, Edges: {result.num_edges}")

            scene_graph_path = output_dir / "scene_graph.json"
            scene_graph.save(scene_graph_path)
            result.scene_graph_path = scene_graph_path

            # Step 2: Build asset index
            self.log("\nStep 2: Building asset index...")
            asset_index = self.asset_index_builder.build(manifest, usd_base_path)

            if self.config.filter_commercial_only:
                asset_index = asset_index.filter_commercial()
                self.log(f"  Filtered to commercial-only: {len(asset_index.assets)} assets")

            result.num_assets = len(asset_index.assets)
            self.log(f"  Assets: {result.num_assets}")

            asset_index_path = output_dir / "asset_index.json"
            asset_index.save(asset_index_path)
            result.asset_index_path = asset_index_path

            # Step 3: Generate task config
            self.log("\nStep 3: Generating task configuration...")
            task_config = self.task_config_generator.generate(
                manifest=manifest,
                robot_type=self.config.robot_type,
                urdf_path=self.config.urdf_path,
                max_tasks=self.config.max_tasks,
            )
            result.num_tasks = len(task_config.suggested_tasks)
            self.log(f"  Tasks: {result.num_tasks}")

            task_config_path = output_dir / "task_config.json"
            task_config.save(task_config_path)
            result.task_config_path = task_config_path

            # Step 4: Generate scene config (YAML for Genie Sim)
            self.log("\nStep 4: Generating scene configuration...")
            scene_config_path = output_dir / "scene_config.yaml"
            self._write_scene_config(
                scene_id=scene_id,
                manifest=manifest,
                scene_graph=scene_graph,
                asset_index=asset_index,
                task_config=task_config,
                output_path=scene_config_path,
            )
            result.scene_config_path = scene_config_path

            # Step 5: Copy USD files if requested
            if self.config.copy_usd_files and usd_source_dir:
                self.log("\nStep 5: Copying USD files...")
                usd_output_dir = output_dir / "usd"
                self._copy_usd_files(usd_source_dir, usd_output_dir)

            # Step 6: Write export manifest
            self.log("\nStep 6: Writing export manifest...")
            export_manifest_path = output_dir / "export_manifest.json"
            self._write_export_manifest(
                result=result,
                config=self.config,
                output_path=export_manifest_path,
            )

            result.success = True
            self.log("\n" + "=" * 70)
            self.log("Export Complete!")
            self.log("=" * 70)
            self.log(f"  Scene Graph: {result.scene_graph_path}")
            self.log(f"  Asset Index: {result.asset_index_path}")
            self.log(f"  Task Config: {result.task_config_path}")
            self.log(f"  Scene Config: {result.scene_config_path}")

        except Exception as e:
            result.errors.append(str(e))
            self.log(f"Export failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()

        return result

    def _write_scene_config(
        self,
        scene_id: str,
        manifest: Dict[str, Any],
        scene_graph: GenieSimSceneGraph,
        asset_index: GenieSimAssetIndex,
        task_config: GenieSimTaskConfig,
        output_path: Path,
    ) -> None:
        """Write Genie Sim scene configuration YAML."""
        scene_config = manifest.get("scene", {})
        room = scene_config.get("room", {})
        bounds = room.get("bounds", {})

        # Build YAML content
        yaml_content = f"""# Genie Sim 3.0 Scene Configuration
# Auto-generated from BlueprintPipeline
# Scene: {scene_id}
# Generated: {datetime.utcnow().isoformat()}Z

scene:
  id: "{scene_id}"
  environment_type: "{scene_config.get('environment_type', 'general')}"
  coordinate_system: "{scene_config.get('coordinate_frame', 'y_up')}"
  meters_per_unit: {scene_config.get('meters_per_unit', 1.0)}

room:
  width: {bounds.get('width', 5.0)}
  depth: {bounds.get('depth', 5.0)}
  height: {bounds.get('height', 3.0)}

robot:
  type: "{task_config.robot_config.robot_type}"
  base_position: {task_config.robot_config.base_position}
  workspace:
    min: {task_config.robot_config.workspace_bounds[0]}
    max: {task_config.robot_config.workspace_bounds[1]}

assets:
  index_path: "asset_index.json"
  total_count: {len(asset_index.assets)}
  commercial_only: {self.config.filter_commercial_only}

scene_graph:
  path: "scene_graph.json"
  node_count: {len(scene_graph.nodes)}
  edge_count: {len(scene_graph.edges)}

tasks:
  config_path: "task_config.json"
  suggested_count: {len(task_config.suggested_tasks)}
  max_tasks: {self.config.max_tasks}

# Genie Sim integration settings
geniesim:
  # Data collection settings
  collection:
    mode: "automated"  # automated or teleop
    episodes_per_task: 100
    use_curobo: true

  # Evaluation settings
  evaluation:
    enabled: true
    vlm_scoring: true
    scenarios: 1000

  # Export settings
  export:
    format: "lerobot_v0.3.3"
    include_visual_obs: true
    include_depth: true

# Source information
source:
  pipeline: "blueprintpipeline"
  version: "2.0.0"
  manifest_path: "scene_manifest.json"
"""

        with open(output_path, "w") as f:
            f.write(yaml_content)

    def _copy_usd_files(
        self,
        source_dir: Path,
        output_dir: Path,
    ) -> None:
        """Copy USD files to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        usd_extensions = {".usd", ".usda", ".usdc", ".usdz"}
        copied = 0

        for usd_file in source_dir.glob("**/*"):
            if usd_file.is_file() and usd_file.suffix.lower() in usd_extensions:
                relative_path = usd_file.relative_to(source_dir)
                dest_path = output_dir / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(usd_file, dest_path)
                copied += 1

        self.log(f"  Copied {copied} USD files")

    def _write_export_manifest(
        self,
        result: GenieSimExportResult,
        config: GenieSimExportConfig,
        output_path: Path,
    ) -> None:
        """Write export manifest JSON."""
        manifest = {
            "export_info": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "exporter_version": "1.0.0",
                "source_pipeline": "blueprintpipeline",
            },
            "config": {
                "robot_type": config.robot_type,
                "generate_embeddings": config.generate_embeddings,
                "embedding_model": config.embedding_model if config.generate_embeddings else None,
                "filter_commercial_only": config.filter_commercial_only,
                "max_tasks": config.max_tasks,
            },
            "result": result.to_dict(),
            "geniesim_compatibility": {
                "version": "3.0",
                "isaac_sim_version": "5.1.0",
                "formats": {
                    "scene_graph": "json",
                    "asset_index": "json",
                    "task_config": "json",
                    "scene_config": "yaml",
                },
            },
        }

        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)


# =============================================================================
# Convenience Functions
# =============================================================================


def export_to_geniesim(
    manifest_path: Path,
    output_dir: Path,
    robot_type: str = "franka",
    verbose: bool = True,
) -> GenieSimExportResult:
    """
    Convenience function to export a scene to Genie Sim format.

    Args:
        manifest_path: Path to scene_manifest.json
        output_dir: Output directory
        robot_type: Robot type (franka, g2, ur10)
        verbose: Print progress

    Returns:
        GenieSimExportResult
    """
    config = GenieSimExportConfig(robot_type=robot_type)
    exporter = GenieSimExporter(config, verbose=verbose)
    return exporter.export(manifest_path, output_dir)
