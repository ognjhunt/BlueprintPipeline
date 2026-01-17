"""
Hand motion generation and rendering for DWM conditioning.

Generates and renders hand mesh trajectories for use as DWM
conditioning inputs alongside static scene videos.

Developer note:
- MANO rendering requires the official MANO assets (download from https://mano.is.tue.mpg.de/)
  and the smplx Python package. Set MANO_MODEL_PATH to the directory containing the MANO
  model files (e.g., ~/.mano/models).
- When HandRenderConfig.require_mano=True, MANO requests raise clear exceptions if assets
  or dependencies are missing instead of falling back to the simple mesh.
"""

from .hand_trajectory_handler import (
    HandTrajectoryGenerator,
    generate_grasp_trajectory,
    generate_pull_trajectory,
    generate_push_trajectory,
    generate_reach_trajectory,
)
from .hand_mesh_renderer import (
    HandModel,
    HandMeshRenderer,
    render_hand_trajectory_to_video,
)
from .retargeting import HandRetargeter, RobotConfig

__all__ = [
    "HandTrajectoryGenerator",
    "generate_grasp_trajectory",
    "generate_pull_trajectory",
    "generate_push_trajectory",
    "generate_reach_trajectory",
    "HandMeshRenderer",
    "HandModel",
    "render_hand_trajectory_to_video",
    "HandRetargeter",
    "RobotConfig",
]
