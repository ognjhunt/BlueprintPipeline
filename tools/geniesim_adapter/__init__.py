"""
Genie Sim 3.0 Adapter Module for BlueprintPipeline.

This module provides the integration layer between BlueprintPipeline's scene
representation and AGIBOT's Genie Sim 3.0 data generation platform.

**LOCAL FRAMEWORK (Recommended for Default/Free Pipeline)**

Genie Sim 3.0 is an open-source LOCAL simulation framework that runs on Isaac Sim.
Use the local framework for running data collection without a hosted API:

    from tools.geniesim_adapter import (
        GenieSimLocalFramework,
        run_local_data_collection,
        check_geniesim_availability,
    )

    # Check if Genie Sim is available locally
    status = check_geniesim_availability()
    if status['available']:
        # Run data collection locally
        result = run_local_data_collection(
            scene_manifest_path=Path("scene_manifest.json"),
            task_config_path=Path("task_config.json"),
            output_dir=Path("./output"),
        )

Architecture:
    BlueprintPipeline handles:
        - Scene image generation (Gemini)
        - 3D reconstruction (3D-RE-GEN)
        - SimReady USD assembly
        - Your own assets (no NC restriction)
        - DWM/Dream2Flow data (unique)
        - Multi-robot embodiment data (arms, humanoids, mobile)
        - VLA fine-tuning packages
        - Rich ground truth annotations

    Genie Sim 3.0 handles (via local framework or API):
        - Task generation (LLM)
        - Trajectory planning (cuRobo)
        - Data collection (automated + teleop)
        - Evaluation (VLM)
        - LeRobot export

Enhanced Features (DEFAULT: ENABLED):
    - Multi-robot data generation (franka, g2, ur10, gr1, fetch, etc.)
    - Bimanual manipulation data
    - Multi-robot coordination scenarios
    - Rich annotations (2D/3D boxes, segmentation, depth GT)
    - VLA fine-tuning packages (OpenVLA, Pi0, SmolVLA, GR00T)

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
from .multi_robot_config import (
    MultiRobotConfig,
    RobotType,
    RobotCategory,
    RobotSpec,
    ROBOT_SPECS,
    DEFAULT_MULTI_ROBOT_CONFIG,
    FULL_ROBOT_CONFIG,
    get_robot_spec,
    get_geniesim_robot_config,
)
from .exporter import (
    GenieSimExporter,
    GenieSimExportConfig,
    GenieSimExportResult,
)
from .importer import (
    GenieSimImporter,
    GenieSimImportConfig,
    GenieSimImportResult,
)
from .local_framework import (
    GenieSimLocalFramework,
    GenieSimConfig,
    GenieSimServerStatus,
    DataCollectionResult,
    check_geniesim_availability,
    run_local_data_collection,
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
    # Multi-Robot Config
    "MultiRobotConfig",
    "RobotType",
    "RobotCategory",
    "RobotSpec",
    "ROBOT_SPECS",
    "DEFAULT_MULTI_ROBOT_CONFIG",
    "FULL_ROBOT_CONFIG",
    "get_robot_spec",
    "get_geniesim_robot_config",
    # Exporter
    "GenieSimExporter",
    "GenieSimExportConfig",
    "GenieSimExportResult",
    # Importer
    "GenieSimImporter",
    "GenieSimImportConfig",
    "GenieSimImportResult",
    # Local Framework
    "GenieSimLocalFramework",
    "GenieSimConfig",
    "GenieSimServerStatus",
    "DataCollectionResult",
    "check_geniesim_availability",
    "run_local_data_collection",
]
