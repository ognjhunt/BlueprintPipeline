#!/usr/bin/env python3
"""
Genie Sim Export Job for BlueprintPipeline.

Cloud Run job that converts BlueprintPipeline scenes to Genie Sim 3.0 format,
enabling data generation using AGIBOT's simulation platform.

This job:
1. Loads the BlueprintPipeline scene manifest
2. Converts to Genie Sim scene graph format
3. Builds asset index for RAG retrieval
4. Generates task configuration hints
5. Outputs files ready for Genie Sim data generation

Pipeline Position:
    3D-RE-GEN → simready → usd-assembly → replicator → [THIS JOB] → Genie Sim

Environment Variables:
    BUCKET: GCS bucket name
    SCENE_ID: Scene identifier
    ASSETS_PREFIX: Path to scene assets (scene_manifest.json)
    GENIESIM_PREFIX: Output path for Genie Sim files
    ROBOT_TYPE: Robot type (franka, g2, ur10) - default: franka
    MAX_TASKS: Maximum suggested tasks - default: 50
    GENERATE_EMBEDDINGS: Generate semantic embeddings - default: false
    FILTER_COMMERCIAL: Only include commercial-use assets - default: false
    COPY_USD: Copy USD files to output - default: true
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.geniesim_adapter import (
    GenieSimExporter,
    GenieSimExportConfig,
    GenieSimExportResult,
)


def run_geniesim_export_job(
    root: Path,
    scene_id: str,
    assets_prefix: str,
    geniesim_prefix: str,
    robot_type: str = "franka",
    urdf_path: Optional[str] = None,
    max_tasks: int = 50,
    generate_embeddings: bool = False,
    filter_commercial: bool = False,
    copy_usd: bool = True,
) -> int:
    """
    Run the Genie Sim export job.

    Args:
        root: Root path (e.g., /mnt/gcs)
        scene_id: Scene identifier
        assets_prefix: Path to scene assets
        geniesim_prefix: Output path for Genie Sim files
        robot_type: Robot type (franka, g2, ur10)
        urdf_path: Custom URDF path for robot
        max_tasks: Maximum suggested tasks
        generate_embeddings: Generate semantic embeddings
        filter_commercial: Only include commercial-use assets
        copy_usd: Copy USD files to output

    Returns:
        0 on success, 1 on failure
    """
    print(f"[GENIESIM-EXPORT-JOB] Starting Genie Sim export for scene: {scene_id}")
    print(f"[GENIESIM-EXPORT-JOB] Assets prefix: {assets_prefix}")
    print(f"[GENIESIM-EXPORT-JOB] Output prefix: {geniesim_prefix}")
    print(f"[GENIESIM-EXPORT-JOB] Robot type: {robot_type}")
    print(f"[GENIESIM-EXPORT-JOB] Max tasks: {max_tasks}")
    print(f"[GENIESIM-EXPORT-JOB] Generate embeddings: {generate_embeddings}")
    print(f"[GENIESIM-EXPORT-JOB] Filter commercial: {filter_commercial}")
    print(f"[GENIESIM-EXPORT-JOB] Copy USD: {copy_usd}")

    assets_dir = root / assets_prefix
    output_dir = root / geniesim_prefix

    # Load manifest
    manifest_path = assets_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        print(f"[GENIESIM-EXPORT-JOB] ERROR: Manifest not found: {manifest_path}")
        return 1

    # Find USD source directory
    usd_source_dir = None
    for possible_usd_dir in [
        assets_dir.parent / "usd",
        assets_dir / "usd",
        root / f"scenes/{scene_id}/usd",
    ]:
        if possible_usd_dir.is_dir():
            usd_source_dir = possible_usd_dir
            print(f"[GENIESIM-EXPORT-JOB] Found USD directory: {usd_source_dir}")
            break

    # Configure exporter
    config = GenieSimExportConfig(
        robot_type=robot_type,
        urdf_path=urdf_path,
        generate_embeddings=generate_embeddings,
        max_tasks=max_tasks,
        copy_usd_files=copy_usd,
        filter_commercial_only=filter_commercial,
    )

    try:
        exporter = GenieSimExporter(config, verbose=True)
        result = exporter.export(
            manifest_path=manifest_path,
            output_dir=output_dir,
            usd_source_dir=usd_source_dir if copy_usd else None,
        )

        if result.success:
            print("\n[GENIESIM-EXPORT-JOB] Export completed successfully")
            print(f"[GENIESIM-EXPORT-JOB]   Scene Graph: {result.scene_graph_path}")
            print(f"[GENIESIM-EXPORT-JOB]   Asset Index: {result.asset_index_path}")
            print(f"[GENIESIM-EXPORT-JOB]   Task Config: {result.task_config_path}")
            print(f"[GENIESIM-EXPORT-JOB]   Nodes: {result.num_nodes}")
            print(f"[GENIESIM-EXPORT-JOB]   Edges: {result.num_edges}")
            print(f"[GENIESIM-EXPORT-JOB]   Assets: {result.num_assets}")
            print(f"[GENIESIM-EXPORT-JOB]   Tasks: {result.num_tasks}")

            # Write completion marker
            marker_path = output_dir / "_GENIESIM_EXPORT_COMPLETE"
            marker_path.write_text(json.dumps({
                "scene_id": scene_id,
                "robot_type": robot_type,
                "success": True,
                "stats": {
                    "nodes": result.num_nodes,
                    "edges": result.num_edges,
                    "assets": result.num_assets,
                    "tasks": result.num_tasks,
                },
            }, indent=2))

            return 0
        else:
            print(f"[GENIESIM-EXPORT-JOB] ERROR: Export failed")
            for error in result.errors:
                print(f"[GENIESIM-EXPORT-JOB]   - {error}")
            return 1

    except Exception as e:
        print(f"[GENIESIM-EXPORT-JOB] ERROR: {e}")
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    # Get configuration from environment
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")

    if not scene_id:
        print("[GENIESIM-EXPORT-JOB] ERROR: SCENE_ID is required")
        sys.exit(1)

    # Prefixes with defaults
    assets_prefix = os.getenv("ASSETS_PREFIX", f"scenes/{scene_id}/assets")
    geniesim_prefix = os.getenv("GENIESIM_PREFIX", f"scenes/{scene_id}/geniesim")

    # Configuration
    robot_type = os.getenv("ROBOT_TYPE", "franka")
    urdf_path = os.getenv("URDF_PATH")  # Optional custom URDF
    max_tasks = int(os.getenv("MAX_TASKS", "50"))
    generate_embeddings = os.getenv("GENERATE_EMBEDDINGS", "false").lower() == "true"
    filter_commercial = os.getenv("FILTER_COMMERCIAL", "false").lower() == "true"
    copy_usd = os.getenv("COPY_USD", "true").lower() == "true"

    print("[GENIESIM-EXPORT-JOB] Configuration:")
    print(f"[GENIESIM-EXPORT-JOB]   Bucket: {bucket}")
    print(f"[GENIESIM-EXPORT-JOB]   Scene ID: {scene_id}")
    print(f"[GENIESIM-EXPORT-JOB]   Robot Type: {robot_type}")
    print(f"[GENIESIM-EXPORT-JOB]   Max Tasks: {max_tasks}")

    GCS_ROOT = Path("/mnt/gcs")

    exit_code = run_geniesim_export_job(
        root=GCS_ROOT,
        scene_id=scene_id,
        assets_prefix=assets_prefix,
        geniesim_prefix=geniesim_prefix,
        robot_type=robot_type,
        urdf_path=urdf_path,
        max_tasks=max_tasks,
        generate_embeddings=generate_embeddings,
        filter_commercial=filter_commercial,
        copy_usd=copy_usd,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
