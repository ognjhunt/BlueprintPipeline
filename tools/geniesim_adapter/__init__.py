"""
Genie Sim 3.0 Adapter Module for BlueprintPipeline.

This module provides the integration layer between BlueprintPipeline's scene
representation and AGIBOT's Genie Sim 3.0 data generation platform.

Architecture:
    BlueprintPipeline handles:
        - Scene image generation (Gemini)
        - 3D reconstruction (3D-RE-GEN)
        - SimReady USD assembly
        - Your own assets (no NC restriction)
        - DWM/Dream2Flow data (unique)

    Genie Sim 3.0 handles:
        - Task generation (LLM)
        - Trajectory planning (cuRobo)
        - Data collection (automated + teleop)
        - Evaluation (VLM)
        - LeRobot export

Usage:
    from tools.geniesim_adapter import GenieSimExporter

    exporter = GenieSimExporter(config)
    result = exporter.export(manifest_path, output_dir)
"""

from .scene_graph import (
    SceneGraphConverter,
    GenieSimNode,
    GenieSimEdge,
    GenieSimSceneGraph,
)
from .asset_index import (
    AssetIndexBuilder,
    GenieSimAsset,
    GenieSimAssetIndex,
)
from .task_config import (
    TaskConfigGenerator,
    GenieSimTaskConfig,
    SuggestedTask,
)
from .exporter import (
    GenieSimExporter,
    GenieSimExportConfig,
    GenieSimExportResult,
)

__all__ = [
    # Scene Graph
    "SceneGraphConverter",
    "GenieSimNode",
    "GenieSimEdge",
    "GenieSimSceneGraph",
    # Asset Index
    "AssetIndexBuilder",
    "GenieSimAsset",
    "GenieSimAssetIndex",
    # Task Config
    "TaskConfigGenerator",
    "GenieSimTaskConfig",
    "SuggestedTask",
    # Exporter
    "GenieSimExporter",
    "GenieSimExportConfig",
    "GenieSimExportResult",
]
