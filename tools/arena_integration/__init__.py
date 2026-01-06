"""
Isaac Lab-Arena Integration for Blueprint Pipeline.

This module provides comprehensive integration with NVIDIA Isaac Lab-Arena,
enabling Blueprint scenes to be evaluated with standardized benchmarks.

Components:
- affordances: Standardized object interaction detection and tagging
- arena_exporter: Convert Blueprint scenes to Arena-compatible format
- task_mapping: Map Blueprint policies to Arena tasks via affordances
- evaluation_runner: Run Arena evaluations on Blueprint scenes
- hub_registration: Auto-register environments with LeRobot Hub
"""

from .affordances import (
    AffordanceType,
    AffordanceDetector,
    AffordanceParams,
    detect_affordances,
    AFFORDANCE_REGISTRY,
)

from .arena_exporter import (
    ArenaSceneExporter,
    ArenaExportConfig,
    export_scene_to_arena,
)

from .task_mapping import (
    TaskAffordanceMapper,
    get_arena_tasks_for_affordances,
    BLUEPRINT_TO_ARENA_MAPPING,
)

from .evaluation_runner import (
    ArenaEvaluationRunner,
    EvaluationConfig,
    EvaluationResult,
    run_arena_evaluation,
)

from .hub_registration import (
    LeRobotHubRegistrar,
    HubConfig,
    register_with_hub,
)

__all__ = [
    # Affordances
    "AffordanceType",
    "AffordanceDetector",
    "AffordanceParams",
    "detect_affordances",
    "AFFORDANCE_REGISTRY",
    # Arena Export
    "ArenaSceneExporter",
    "ArenaExportConfig",
    "export_scene_to_arena",
    # Task Mapping
    "TaskAffordanceMapper",
    "get_arena_tasks_for_affordances",
    "BLUEPRINT_TO_ARENA_MAPPING",
    # Evaluation
    "ArenaEvaluationRunner",
    "EvaluationConfig",
    "EvaluationResult",
    "run_arena_evaluation",
    # Hub Registration
    "LeRobotHubRegistrar",
    "HubConfig",
    "register_with_hub",
]
