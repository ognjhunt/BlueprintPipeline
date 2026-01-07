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
5. Generates MULTI-ROBOT configuration (DEFAULT: ENABLED)
6. Generates enhanced features config (VLA, annotations, bimanual)
7. Outputs files ready for Genie Sim data generation

Pipeline Position:
    3D-RE-GEN → simready → usd-assembly → replicator → [THIS JOB] → Genie Sim

Enhanced Features (DEFAULT: ENABLED):
    - Multi-robot embodiment data (franka, g2, ur10, gr1, fetch, etc.)
    - Bimanual manipulation tasks
    - Multi-robot coordination scenarios
    - Rich ground truth annotations
    - VLA fine-tuning package configs

Environment Variables:
    BUCKET: GCS bucket name
    SCENE_ID: Scene identifier
    ASSETS_PREFIX: Path to scene assets (scene_manifest.json)
    GENIESIM_PREFIX: Output path for Genie Sim files
    ROBOT_TYPE: Primary robot type (franka, g2, ur10) - default: franka
    MAX_TASKS: Maximum suggested tasks - default: 50
    GENERATE_EMBEDDINGS: Generate semantic embeddings - default: false
    FILTER_COMMERCIAL: Only include commercial-use assets - default: true
    COPY_USD: Copy USD files to output - default: true
    ENABLE_MULTI_ROBOT: Generate for multiple robot types - default: true
    ENABLE_BIMANUAL: Generate bimanual tasks - default: true
    ENABLE_VLA_PACKAGES: Generate VLA fine-tuning configs - default: true
    ENABLE_RICH_ANNOTATIONS: Generate rich annotation configs - default: true
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
    filter_commercial: bool = True,  # Default TRUE for commercial use
    copy_usd: bool = True,
    enable_multi_robot: bool = True,  # DEFAULT: ENABLED
    enable_bimanual: bool = True,  # DEFAULT: ENABLED
    enable_vla_packages: bool = True,  # DEFAULT: ENABLED
    enable_rich_annotations: bool = True,  # DEFAULT: ENABLED
    variation_assets_prefix: Optional[str] = None,  # Path to variation assets
    replicator_prefix: Optional[str] = None,  # Path to replicator bundle
) -> int:
    """
    Run the Genie Sim export job.

    Args:
        root: Root path (e.g., /mnt/gcs)
        scene_id: Scene identifier
        assets_prefix: Path to scene assets
        geniesim_prefix: Output path for Genie Sim files
        robot_type: Primary robot type (franka, g2, ur10)
        urdf_path: Custom URDF path for robot
        max_tasks: Maximum suggested tasks
        generate_embeddings: Generate semantic embeddings
        filter_commercial: Only include commercial-use assets (DEFAULT: True)
        copy_usd: Copy USD files to output
        enable_multi_robot: Generate for multiple robot types (DEFAULT: True)
        enable_bimanual: Generate bimanual task configs (DEFAULT: True)
        enable_vla_packages: Generate VLA fine-tuning configs (DEFAULT: True)
        enable_rich_annotations: Generate rich annotation configs (DEFAULT: True)
        variation_assets_prefix: Path to variation assets (YOUR commercial assets)
        replicator_prefix: Path to replicator bundle

    Returns:
        0 on success, 1 on failure
    """
    print(f"[GENIESIM-EXPORT-JOB] Starting Genie Sim export for scene: {scene_id}")
    print(f"[GENIESIM-EXPORT-JOB] Assets prefix: {assets_prefix}")
    print(f"[GENIESIM-EXPORT-JOB] Output prefix: {geniesim_prefix}")
    print(f"[GENIESIM-EXPORT-JOB] Variation assets prefix: {variation_assets_prefix}")
    print(f"[GENIESIM-EXPORT-JOB] Replicator prefix: {replicator_prefix}")
    print(f"[GENIESIM-EXPORT-JOB] Primary robot type: {robot_type}")
    print(f"[GENIESIM-EXPORT-JOB] Max tasks: {max_tasks}")
    print(f"[GENIESIM-EXPORT-JOB] Generate embeddings: {generate_embeddings}")
    print(f"[GENIESIM-EXPORT-JOB] Filter commercial: {filter_commercial}")
    print(f"[GENIESIM-EXPORT-JOB] Copy USD: {copy_usd}")
    print(f"[GENIESIM-EXPORT-JOB] Multi-robot enabled: {enable_multi_robot}")
    print(f"[GENIESIM-EXPORT-JOB] Bimanual enabled: {enable_bimanual}")
    print(f"[GENIESIM-EXPORT-JOB] VLA packages enabled: {enable_vla_packages}")
    print(f"[GENIESIM-EXPORT-JOB] Rich annotations enabled: {enable_rich_annotations}")

    assets_dir = root / assets_prefix
    output_dir = root / geniesim_prefix

    # Load manifest
    manifest_path = assets_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        print(f"[GENIESIM-EXPORT-JOB] ERROR: Manifest not found: {manifest_path}")
        return 1

    # Load and merge variation assets (YOUR commercial assets for domain randomization)
    # This is CRITICAL for commercial use - Genie Sim's assets are CC BY-NC-SA 4.0
    variation_assets_dir = None
    if variation_assets_prefix:
        variation_assets_dir = root / variation_assets_prefix
        variation_assets_json = variation_assets_dir / "variation_assets.json"
        if variation_assets_json.is_file():
            print(f"[GENIESIM-EXPORT-JOB] Loading variation assets from: {variation_assets_json}")
            try:
                with open(variation_assets_json) as f:
                    variation_data = json.load(f)
                variation_objects = variation_data.get("objects", [])
                print(f"[GENIESIM-EXPORT-JOB] Found {len(variation_objects)} variation assets")

                # Mark these as YOUR commercial assets
                for obj in variation_objects:
                    if "asset" not in obj:
                        obj["asset"] = {}
                    obj["asset"]["source"] = "blueprintpipeline_generated"
                    obj["asset"]["commercial_ok"] = True
                    obj["is_variation_asset"] = True

                # We'll merge these with the manifest later
            except Exception as e:
                print(f"[GENIESIM-EXPORT-JOB] WARNING: Failed to load variation assets: {e}")
                variation_objects = []
        else:
            print(f"[GENIESIM-EXPORT-JOB] No variation assets found at: {variation_assets_json}")
            variation_objects = []
    else:
        variation_objects = []
        print("[GENIESIM-EXPORT-JOB] WARNING: No variation_assets_prefix specified")
        print("[GENIESIM-EXPORT-JOB] WARNING: Without YOUR variation assets, you cannot sell the data commercially!")

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

    # Load manifest and merge with variation assets
    print(f"[GENIESIM-EXPORT-JOB] Loading manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    original_object_count = len(manifest.get("objects", []))
    print(f"[GENIESIM-EXPORT-JOB] Original manifest has {original_object_count} objects")

    # Merge variation assets into manifest
    if variation_objects:
        print(f"[GENIESIM-EXPORT-JOB] Merging {len(variation_objects)} variation assets into manifest")
        if "objects" not in manifest:
            manifest["objects"] = []
        manifest["objects"].extend(variation_objects)

        # Also add to a separate key for reference
        manifest["variation_assets"] = {
            "count": len(variation_objects),
            "source": "variation-gen-job",
            "commercial_ok": True,
        }
        print(f"[GENIESIM-EXPORT-JOB] Merged manifest now has {len(manifest['objects'])} objects")
    else:
        print("[GENIESIM-EXPORT-JOB] WARNING: No variation assets to merge")
        print("[GENIESIM-EXPORT-JOB] WARNING: Scene will only have original objects")
        print("[GENIESIM-EXPORT-JOB] WARNING: For domain randomization in commercial use, you need variation assets!")

    # Write merged manifest to output directory for the exporter
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_manifest_path = output_dir / "merged_scene_manifest.json"
    with open(merged_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[GENIESIM-EXPORT-JOB] Wrote merged manifest to: {merged_manifest_path}")

    # Configure exporter with enhanced features
    config = GenieSimExportConfig(
        robot_type=robot_type,
        urdf_path=urdf_path,
        generate_embeddings=generate_embeddings,
        max_tasks=max_tasks,
        copy_usd_files=copy_usd,
        filter_commercial_only=filter_commercial,
        # Enhanced features (DEFAULT: ENABLED)
        enable_multi_robot=enable_multi_robot,
        enable_bimanual=enable_bimanual,
        enable_vla_packages=enable_vla_packages,
        enable_rich_annotations=enable_rich_annotations,
        enable_multi_robot_coordination=enable_multi_robot,  # Tied to multi_robot
    )

    try:
        exporter = GenieSimExporter(config, verbose=True)
        # Use merged manifest that includes YOUR variation assets
        result = exporter.export(
            manifest_path=merged_manifest_path,
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
                "commercial_data": filter_commercial,
                "stats": {
                    "nodes": result.num_nodes,
                    "edges": result.num_edges,
                    "assets": result.num_assets,
                    "tasks": result.num_tasks,
                    "original_objects": original_object_count,
                    "variation_assets": len(variation_objects),
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
    # IMPORTANT: variation_assets_prefix contains YOUR commercial assets
    variation_assets_prefix = os.getenv("VARIATION_ASSETS_PREFIX", f"scenes/{scene_id}/variation_assets")
    replicator_prefix = os.getenv("REPLICATOR_PREFIX", f"scenes/{scene_id}/replicator")

    # Configuration
    robot_type = os.getenv("ROBOT_TYPE", "franka")
    urdf_path = os.getenv("URDF_PATH")  # Optional custom URDF
    max_tasks = int(os.getenv("MAX_TASKS", "50"))
    generate_embeddings = os.getenv("GENERATE_EMBEDDINGS", "false").lower() == "true"
    # Default to TRUE for commercial use - only use your own assets
    filter_commercial = os.getenv("FILTER_COMMERCIAL", "true").lower() == "true"
    copy_usd = os.getenv("COPY_USD", "true").lower() == "true"

    # Enhanced features (DEFAULT: ENABLED)
    enable_multi_robot = os.getenv("ENABLE_MULTI_ROBOT", "true").lower() == "true"
    enable_bimanual = os.getenv("ENABLE_BIMANUAL", "true").lower() == "true"
    enable_vla_packages = os.getenv("ENABLE_VLA_PACKAGES", "true").lower() == "true"
    enable_rich_annotations = os.getenv("ENABLE_RICH_ANNOTATIONS", "true").lower() == "true"

    print("[GENIESIM-EXPORT-JOB] Configuration:")
    print(f"[GENIESIM-EXPORT-JOB]   Bucket: {bucket}")
    print(f"[GENIESIM-EXPORT-JOB]   Scene ID: {scene_id}")
    print(f"[GENIESIM-EXPORT-JOB]   Variation Assets: {variation_assets_prefix}")
    print(f"[GENIESIM-EXPORT-JOB]   Replicator Bundle: {replicator_prefix}")
    print(f"[GENIESIM-EXPORT-JOB]   Primary Robot Type: {robot_type}")
    print(f"[GENIESIM-EXPORT-JOB]   Max Tasks: {max_tasks}")
    print(f"[GENIESIM-EXPORT-JOB]   Multi-Robot: {enable_multi_robot}")
    print(f"[GENIESIM-EXPORT-JOB]   Bimanual: {enable_bimanual}")
    print(f"[GENIESIM-EXPORT-JOB]   VLA Packages: {enable_vla_packages}")
    print(f"[GENIESIM-EXPORT-JOB]   Rich Annotations: {enable_rich_annotations}")
    print(f"[GENIESIM-EXPORT-JOB]   Commercial Filter: {filter_commercial}")

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
        # Enhanced features (DEFAULT: ENABLED)
        enable_multi_robot=enable_multi_robot,
        enable_bimanual=enable_bimanual,
        enable_vla_packages=enable_vla_packages,
        enable_rich_annotations=enable_rich_annotations,
        # YOUR commercial assets for domain randomization
        variation_assets_prefix=variation_assets_prefix,
        replicator_prefix=replicator_prefix,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
