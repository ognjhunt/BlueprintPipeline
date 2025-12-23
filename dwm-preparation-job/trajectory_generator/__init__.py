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

__all__ = [
    "EgocentricTrajectoryGenerator",
    "generate_approach_trajectory",
    "generate_orbit_trajectory",
    "generate_reach_manipulate_trajectory",
    "generate_walkthrough_trajectory",
]
