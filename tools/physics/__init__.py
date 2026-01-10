"""Physics Extensions for BlueprintPipeline.

Extends rigid body physics with:
- Soft body physics (deformable objects)
- Multi-agent coordination physics
- Dynamic scene physics (moving obstacles, human interaction)
"""

from .soft_body import (
    SoftBodyPhysics,
    SoftBodyType,
    SoftBodyProperties,
    DeformableMaterial,
)

from .multi_agent import (
    MultiAgentCoordinator,
    AgentConfiguration,
    CollisionAvoidanceStrategy,
)

from .dynamic_scene import (
    DynamicSceneManager,
    DynamicObstacle,
    HumanModel,
    SceneDynamicsConfig,
)

__all__ = [
    # Soft body
    "SoftBodyPhysics",
    "SoftBodyType",
    "SoftBodyProperties",
    "DeformableMaterial",

    # Multi-agent
    "MultiAgentCoordinator",
    "AgentConfiguration",
    "CollisionAvoidanceStrategy",

    # Dynamic scenes
    "DynamicSceneManager",
    "DynamicObstacle",
    "HumanModel",
    "SceneDynamicsConfig",
]
