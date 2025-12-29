"""
Trajectory generation for DWM conditioning.

Generates egocentric camera trajectories through 3D scenes
for rendering static scene videos as DWM inputs.
"""

from .egocentric_trajectories import (
    EgocentricTrajectoryGenerator,
    generate_approach_trajectory,
    generate_orbit_trajectory,
    generate_reach_manipulate_trajectory,
    generate_walkthrough_trajectory,
)
from .physics_policy_runner import PhysicsPolicyRunner

__all__ = [
    "EgocentricTrajectoryGenerator",
    "generate_approach_trajectory",
    "generate_orbit_trajectory",
    "generate_reach_manipulate_trajectory",
    "generate_walkthrough_trajectory",
    "PhysicsPolicyRunner",
]
