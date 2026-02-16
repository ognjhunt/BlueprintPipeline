"""
Fallback replacement for isaaclab.omron_franka_occupancy.

The original module is NVIDIA-internal and not included in the public SAGE repo.
This provides the three functions that object_mobile_manipulation_utils.py imports:
  - occupancy_map(forward, side, yaw, offset) -> callable
  - support_point (namedtuple/class)
  - get_forward_side_from_support_point_and_yaw(pos_2d, yaw) -> (forward, side)

The mobile Franka (Omron LD-250 base + Franka Emika arm) has approximate footprint:
  - Base: 0.707m x 0.530m (Omron LD-250 specs)
  - Arm reach: ~0.855m from base center
  - Total footprint for occupancy: ~0.80m x 0.60m with safety margin
"""

import math
import os
from collections import namedtuple
import numpy as np


# ── support_point type ───────────────────────────────────────────────────────
# Simple named type for robot support point (where the robot stands)
support_point = namedtuple("support_point", ["x", "y", "yaw"])


# ── Robot Footprint Constants ────────────────────────────────────────────────
# Omron LD-250 base dimensions (meters)
# Configurable to avoid hard-coding conservative assumptions in all environments.
ROBOT_BASE_LENGTH = float(os.getenv("OMRON_BASE_LENGTH_M", "0.707"))  # Along forward direction
ROBOT_BASE_WIDTH = float(os.getenv("OMRON_BASE_WIDTH_M", "0.530"))    # Along side direction
# Safety margin for collision checking.
# The old fixed 0.10m often over-constrained kitchen corridors; use a more
# realistic default while still allowing explicit override.
SAFETY_MARGIN = float(os.getenv("OMRON_OCCUPANCY_SAFETY_MARGIN_M", "0.04"))
# Effective footprint with margin
FOOTPRINT_HALF_LENGTH = (ROBOT_BASE_LENGTH / 2) + SAFETY_MARGIN  # ~0.45m
FOOTPRINT_HALF_WIDTH = (ROBOT_BASE_WIDTH / 2) + SAFETY_MARGIN    # ~0.37m


def get_forward_side_from_support_point_and_yaw(pos_2d, yaw):
    """
    Compute the forward and side positions given a 2D support position and yaw angle.

    The robot stands at pos_2d facing direction yaw (radians).
    'forward' = the position along the robot's facing direction
    'side' = the position along the robot's lateral axis

    Args:
        pos_2d: [x, y] position of the robot base support point
        yaw: float, yaw angle in radians (0 = +X direction)

    Returns:
        forward: float, position along forward axis (world frame projection)
        side: float, position along lateral axis (world frame projection)
    """
    x, y = float(pos_2d[0]), float(pos_2d[1])
    # Forward direction: the component along the yaw direction
    # Side direction: the component perpendicular to yaw
    forward = x * math.cos(yaw) + y * math.sin(yaw)
    side = -x * math.sin(yaw) + y * math.cos(yaw)
    return forward, side


def occupancy_map(forward, side, yaw, offset=0.05):
    """
    Create a robot occupancy checker function.

    Returns a callable that takes 2D points and returns True if they fall
    within the robot's footprint (including offset margin).

    The robot's center is at (side, forward) in the rotated frame,
    with orientation given by yaw.

    Args:
        forward: float, forward position component
        side: float, side position component
        yaw: float, yaw angle in radians
        offset: float, additional safety offset (meters)

    Returns:
        callable: fn(points) -> np.ndarray[bool]
            points: np.ndarray of shape (N, 2) in world frame
            returns: boolean array, True = inside robot footprint
    """
    # Robot center in world frame
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)

    # Reconstruct world position from forward/side
    center_x = forward * cos_y - side * sin_y
    center_y = forward * sin_y + side * cos_y

    # Effective half-extents with offset
    half_l = FOOTPRINT_HALF_LENGTH + offset
    half_w = FOOTPRINT_HALF_WIDTH + offset

    def _check_occupancy(points):
        """Check if 2D points fall within robot footprint.

        Args:
            points: np.ndarray of shape (N, 2) — world-frame XY positions

        Returns:
            np.ndarray[bool] of shape (N,) — True if inside robot footprint
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        # Transform points to robot-local frame
        dx = pts[:, 0] - center_x
        dy = pts[:, 1] - center_y

        # Rotate to robot frame
        local_x = dx * cos_y + dy * sin_y      # along forward
        local_y = -dx * sin_y + dy * cos_y     # along side

        # Check AABB in robot frame
        inside = (np.abs(local_x) <= half_l) & (np.abs(local_y) <= half_w)
        return inside

    return _check_occupancy


def get_robot_footprint(yaw=0.0, center=(0.0, 0.0), offset=0.05):
    """
    Get the 4 corner points of the robot footprint in world frame.

    Useful for visualization and debugging.

    Args:
        yaw: float, robot orientation in radians
        center: tuple (x, y), robot center in world frame
        offset: float, safety margin

    Returns:
        np.ndarray of shape (4, 2) — corner points in world frame
    """
    half_l = FOOTPRINT_HALF_LENGTH + offset
    half_w = FOOTPRINT_HALF_WIDTH + offset

    # Local corners (forward, side)
    corners_local = np.array([
        [half_l, half_w],
        [half_l, -half_w],
        [-half_l, -half_w],
        [-half_l, half_w],
    ])

    # Rotate to world frame
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])

    corners_world = (R @ corners_local.T).T + np.array(center)
    return corners_world
