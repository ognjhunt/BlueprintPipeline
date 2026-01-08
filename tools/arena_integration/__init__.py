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

NEW - Isaac Lab-Arena Architecture Upgrade:
- components: Modular Lego architecture (Objects, Scenes, Embodiments, Tasks)
- parallel_evaluation: GPU-accelerated evaluation (1000+ parallel envs)
- mimic_integration: Isaac Lab-Mimic demo scaling (complements Genie Sim)
- groot_integration: GR00T N VLM evaluation support
- lerobot_hub: Updated LeRobot Environment Hub v2.0 format
- composite_tasks: Dynamic task chaining for long-horizon evaluation
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

# NEW: Modular Components (Lego Architecture)
from .components import (
    ArenaObject,
    ArenaScene,
    ArenaEmbodiment,
    ArenaTask,
    ArenaEnvironmentSpec,
    ArenaEnvironmentBuilder,
    ArenaComponentRegistry,
    ObjectCategory,
    EnvironmentType,
    EmbodimentType,
    TaskDifficulty,
    get_registry,
)

# NEW: GPU-Accelerated Parallel Evaluation
from .parallel_evaluation import (
    ParallelEvaluator,
    ParallelEvalConfig,
    ParallelEvalResult,
    MultiPolicyEvaluator,
    run_parallel_evaluation,
    estimate_evaluation_time,
)

# NEW: Isaac Lab-Mimic Integration (Demo Scaling)
from .mimic_integration import (
    MimicAugmenter,
    MimicConfig,
    AugmentationStrategy,
    Demonstration,
    AugmentationResult,
    augment_genie_sim_episodes,
)

# NEW: GR00T N VLM Integration
from .groot_integration import (
    GR00TPolicy,
    GR00TConfig,
    GR00TModelType,
    GR00TEvaluator,
    GR00TEvaluationConfig,
    GR00TEvaluationResult,
    GR00TBenchmarkGenerator,
    evaluate_groot_on_arena,
    load_groot_for_arena,
)

# NEW: LeRobot Hub v2.0 Integration
from .lerobot_hub import (
    LeRobotHubPublisher,
    HubPublishConfig,
    HubEnvironmentSpec,
    HubEnvironmentBuilder,
    HubPublishResult,
    BenchmarkCategory,
    publish_to_hub,
    generate_hub_spec,
)

# NEW: Composite Task Chaining
from .composite_tasks import (
    CompositeTask,
    CompositeTaskBuilder,
    CompositeTaskExecutor,
    TaskChain,
    TaskNode,
    TaskTransition,
    TransitionType,
    HandoffState,
    build_composite_task,
    evaluate_composite_task,
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
    # Evaluation (Legacy)
    "ArenaEvaluationRunner",
    "EvaluationConfig",
    "EvaluationResult",
    "run_arena_evaluation",
    # Hub Registration (Legacy)
    "LeRobotHubRegistrar",
    "HubConfig",
    "register_with_hub",
    # NEW: Modular Components
    "ArenaObject",
    "ArenaScene",
    "ArenaEmbodiment",
    "ArenaTask",
    "ArenaEnvironmentSpec",
    "ArenaEnvironmentBuilder",
    "ArenaComponentRegistry",
    "ObjectCategory",
    "EnvironmentType",
    "EmbodimentType",
    "TaskDifficulty",
    "get_registry",
    # NEW: Parallel Evaluation
    "ParallelEvaluator",
    "ParallelEvalConfig",
    "ParallelEvalResult",
    "MultiPolicyEvaluator",
    "run_parallel_evaluation",
    "estimate_evaluation_time",
    # NEW: Mimic Integration
    "MimicAugmenter",
    "MimicConfig",
    "AugmentationStrategy",
    "Demonstration",
    "AugmentationResult",
    "augment_genie_sim_episodes",
    # NEW: GR00T Integration
    "GR00TPolicy",
    "GR00TConfig",
    "GR00TModelType",
    "GR00TEvaluator",
    "GR00TEvaluationConfig",
    "GR00TEvaluationResult",
    "GR00TBenchmarkGenerator",
    "evaluate_groot_on_arena",
    "load_groot_for_arena",
    # NEW: LeRobot Hub v2.0
    "LeRobotHubPublisher",
    "HubPublishConfig",
    "HubEnvironmentSpec",
    "HubEnvironmentBuilder",
    "HubPublishResult",
    "BenchmarkCategory",
    "publish_to_hub",
    "generate_hub_spec",
    # NEW: Composite Tasks
    "CompositeTask",
    "CompositeTaskBuilder",
    "CompositeTaskExecutor",
    "TaskChain",
    "TaskNode",
    "TaskTransition",
    "TransitionType",
    "HandoffState",
    "build_composite_task",
    "evaluate_composite_task",
]
