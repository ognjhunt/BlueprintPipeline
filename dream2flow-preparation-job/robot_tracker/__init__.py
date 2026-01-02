"""
Robot Tracker module for Dream2Flow.

This module uses 3D object flow as a goal/reward signal for robot control,
implementing the final stage of the Dream2Flow pipeline.

Methods supported:
1. Trajectory Optimization - MPC, iLQR for tracking flow
2. Reinforcement Learning - Use flow as embodiment-agnostic reward
3. Behavior Cloning - Learn from flow demonstrations

Reference: Dream2Flow (arXiv:2512.24766)
"""

from .robot_tracker import (
    RobotTracker,
    RobotTrackerConfig,
    MockRobotTracker,
    track_object_flow,
)

__all__ = [
    "RobotTracker",
    "RobotTrackerConfig",
    "MockRobotTracker",
    "track_object_flow",
]
