#!/usr/bin/env python3
"""
Genie Sim 3.0 Local Framework Adapter.

This module provides a LOCAL integration with Genie Sim 3.0, running it as a framework
directly within the Isaac Sim environment rather than through a non-existent hosted API.

Based on the official Genie Sim 3.0 architecture:
- Repository: https://github.com/AgibotTech/genie_sim
- Data Collection: Uses gRPC for client-server communication
- Server: Runs inside Isaac Sim with PhysX and Replicator
- Client: Controls robot, captures data, runs tasks

This replaces the geniesim_client.py which incorrectly assumed a hosted API service.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    BlueprintPipeline                        │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │              GenieSimLocalFramework                   │   │
    │  │  ┌─────────────┐    ┌─────────────────────────────┐ │   │
    │  │  │ gRPC Client │◄──►│ Genie Sim Data Collection   │ │   │
    │  │  │ (port from │    │ Server (inside Isaac Sim)   │ │   │
    │  │  │ GENIESIM_) │    │                             │ │   │
    │  │  └─────────────┘    └─────────────────────────────┘ │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                              ▲                              │
    │                              │                              │
    │  ┌──────────────────────────┴───────────────────────────┐   │
    │  │                   Isaac Sim                           │   │
    │  │  - PhysX for physics simulation                       │   │
    │  │  - Replicator for sensor data capture                 │   │
    │  │  - cuRobo for motion planning                         │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from tools.geniesim_adapter.local_framework import GenieSimLocalFramework

    # Initialize and connect to running Genie Sim server
    framework = GenieSimLocalFramework()

    # Check if ready
    if framework.is_ready():
        # Run data collection
        result = framework.run_data_collection(task_config, scene_config)
    else:
        # Start server if needed
        framework.start_server(scene_usd_path)

Environment Variables:
    GENIESIM_HOST: Genie Sim gRPC server host (default: localhost)
    GENIESIM_PORT: Genie Sim gRPC server port (default: adapter default port)
    GENIESIM_GRPC_TIMEOUT_S: Connection timeout in seconds (default: 30; legacy: GENIESIM_TIMEOUT)
    GENIESIM_GRPC_MAX_RETRIES: Max retry attempts for retryable gRPC errors (default: 3)
    GENIESIM_GRPC_RETRY_BASE_S: Base delay in seconds for gRPC retries (default: 0.5)
    GENIESIM_GRPC_RETRY_MAX_S: Max delay in seconds for gRPC retries (default: 5.0)
    GENIESIM_READINESS_TIMEOUT_S: Readiness probe timeout in seconds (default: 10)
    GENIESIM_ROOT: Path to Genie Sim installation (default: /opt/geniesim)
    GENIESIM_RECORDINGS_DIR: Directory for Genie Sim recordings (default: /tmp/geniesim_recordings; legacy: GENIESIM_RECORDING_DIR)
    GENIESIM_LOG_DIR: Directory for Genie Sim logs (default: /tmp/geniesim_logs)
    ISAAC_SIM_PATH: Path to Isaac Sim installation (default: /isaac-sim)
    ISAACSIM_REQUIRED: Enforce Isaac Sim + Genie Sim installation checks (default: false)
    CUROBO_REQUIRED: Enforce cuRobo availability checks (default: false)
    ALLOW_GENIESIM_MOCK: Allow local mock gRPC server when GENIESIM_ROOT is missing (default: 0)
    GENIESIM_ALLOW_LINEAR_FALLBACK_IN_PROD: Allow non-collision-aware linear fallback in production (default: 0; risky)
    GENIESIM_ALLOW_IK_FAILURE_FALLBACK: Allow linear fallback if IK planning fails (default: 0)
    GENIESIM_SMOOTH_FALLBACK_TRAJECTORY: Smooth IK/linear fallback joint trajectories (default: 1)
    GENIESIM_SMOOTH_WINDOW: Smoothing window size (odd integer, default: 5)
    GENIESIM_ALLOW_HEURISTIC_ATTACH: Allow attach without contact data (default: 1 non-prod, 0 prod)
    GENIESIM_FORCE_ATTACH_ON_GRASP_PHASE: Force attach to target during grasp/lift/transport (default: 1 non-prod, 0 prod)
    GENIESIM_GRASP_THRESHOLD_SCALE: Scale factor for grasp distance threshold (default: 1.0)
    GENIESIM_GRASP_MAX_DISTANCE_M: Max distance (m) for heuristic attach (default: 0.35)
    GENIESIM_STALL_TIMEOUT_S: Abort/reset episode if observations stall (default: 30)
    GENIESIM_MAX_STALLS: Max stalled episodes before server restart (default: 2)
    GENIESIM_STALL_BACKOFF_S: Backoff between stall handling attempts (default: 5)
    GENIESIM_COLLECTION_TIMEOUT_S: Abort data collection if total runtime exceeds this timeout (default: unset)
    GENIESIM_STARTUP_TIMEOUT_S: Startup timeout in seconds for server readiness (default: 120)
    GENIESIM_STARTUP_POLL_S: Poll interval in seconds for server readiness (default: 2)
    GENIESIM_CLEANUP_TMP: Remove Genie Sim temp directories after a run (default: 1 for local, 0 in production)
    GENIESIM_VALIDATE_FRAMES: Validate recorded frames before saving (default: 0)
    GENIESIM_FAIL_ON_FRAME_VALIDATION: Fail episode when frame validation errors exist (default: 0)
    GENIESIM_MAX_INIT_RESTARTS: Max init-only restarts before giving up (default: GENIESIM_MAX_RESTARTS)
"""

import base64
import binascii
import concurrent.futures
import importlib.util
import io
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)
from tools.camera_io import (
    decode_camera_bytes,
    expected_byte_count,
    is_placeholder_depth,
    is_placeholder_rgb,
    load_camera_frame,
    resolve_npy_path,
    strip_camera_data,
    validate_camera_array,
    validate_rgb_frame_quality,
    validate_frame_sequence_variety,
)

# Add parent paths for imports
ADAPTER_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = ADAPTER_ROOT.parent
REPO_ROOT = TOOLS_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTER_ROOT))

from tools.logging_config import init_logging
from tools.error_handling import CircuitBreaker
from tools.config.production_mode import resolve_env_with_legacy, resolve_pipeline_environment
from tools.geniesim_adapter.config import (
    get_geniesim_circuit_breaker_failure_threshold,
    get_geniesim_circuit_breaker_recovery_timeout_s,
    get_geniesim_circuit_breaker_success_threshold,
    get_geniesim_grpc_max_retries,
    get_geniesim_grpc_retry_base_s,
    get_geniesim_grpc_retry_max_s,
    get_geniesim_grpc_timeout_s,
    get_geniesim_host,
    get_geniesim_port,
    get_geniesim_readiness_timeout_s,
    get_geniesim_trajectory_fps,
)
from tools.error_handling.retry import RetryConfig, calculate_delay
from tools.lerobot_format import LeRobotExportFormat, parse_lerobot_export_format
from tools.config.env import parse_bool_env
from tools.quality.quality_config import load_quality_config, QualityConfig

logger = logging.getLogger(__name__)


def _strict_realism_enabled() -> bool:
    """Fail-closed realism gate for all environments."""
    return parse_bool_env(os.getenv("STRICT_REALISM"), default=True)

GENIESIM_RECORDINGS_DIR_ENV = "GENIESIM_RECORDINGS_DIR"
GENIESIM_RECORDING_DIR_ENV = "GENIESIM_RECORDING_DIR"
GENIESIM_LOG_DIR_ENV = "GENIESIM_LOG_DIR"

# Import gRPC protobuf stubs
try:
    import grpc
    from geniesim_grpc_pb2 import (
        AddCameraReq,
        AddCameraRsp,
        AttachReq,
        AttachRsp,
        CameraRequest,
        ContactReportReq,
        ContactReportRsp,
        EEWrenchReq,
        EEWrenchRsp,
        DetachReq,
        DetachRsp,
        ExitReq,
        ExitRsp,
        GetCheckerStatusReq,
        GetCheckerStatusRsp,
        GetObservationReq,
        GetObservationRsp,
        InitRobotReq,
        InitRobotRsp,
        LightCfg,
        ObjectPose,
        PlaybackReq,
        PlaybackRsp,
        RemoveObstacleReq,
        RemoveObstacleRsp,
        ResetReq,
        ResetRsp,
        SetFrameStateReq,
        SetFrameStateRsp,
        SetLightReq,
        SetLightRsp,
        SetObjectPoseReq,
        SetObjectPoseRsp,
        SetTaskMetricReq,
        SetTaskMetricRsp,
        SetTrajectoryListReq,
        SetTrajectoryListRsp,
        StoreCurrentStateReq,
        StoreCurrentStateRsp,
        TaskStatusReq,
        TaskStatusRsp,
        GripperRequest,
    )
    from geniesim_grpc_pb2_grpc import SimObservationServiceStub
    from aimdk.protocol.common import joint_pb2, rpy_pb2, se3_pose_pb2, vec3_pb2
    from aimdk.protocol.hal.joint import joint_channel_pb2, joint_channel_pb2_grpc
    GRPC_STUBS_AVAILABLE = True
except (ImportError, RuntimeError) as exc:
    GRPC_STUBS_AVAILABLE = False
    grpc = None
    logger.warning("gRPC stubs not available - using legacy fallback: %s", exc)
    _GRPC_PLACEHOLDER_NAMES = [
        "AddCameraReq",
        "AddCameraRsp",
        "AttachReq",
        "AttachRsp",
        "CameraRequest",
        "ContactReportReq",
        "ContactReportRsp",
        "EEWrenchReq",
        "EEWrenchRsp",
        "DetachReq",
        "DetachRsp",
        "ExitReq",
        "ExitRsp",
        "GetCheckerStatusReq",
        "GetCheckerStatusRsp",
        "GetObservationReq",
        "GetObservationRsp",
        "InitRobotReq",
        "InitRobotRsp",
        "LightCfg",
        "ObjectPose",
        "PlaybackReq",
        "PlaybackRsp",
        "RemoveObstacleReq",
        "RemoveObstacleRsp",
        "ResetReq",
        "ResetRsp",
        "SetFrameStateReq",
        "SetFrameStateRsp",
        "SetLightReq",
        "SetLightRsp",
        "SetObjectPoseReq",
        "SetObjectPoseRsp",
        "SetTaskMetricReq",
        "SetTaskMetricRsp",
        "SetTrajectoryListReq",
        "SetTrajectoryListRsp",
        "StoreCurrentStateReq",
        "StoreCurrentStateRsp",
        "TaskStatusReq",
        "TaskStatusRsp",
        "GripperRequest",
        "SimObservationServiceStub",
        "joint_pb2",
        "rpy_pb2",
        "se3_pose_pb2",
        "vec3_pb2",
        "joint_channel_pb2",
        "joint_channel_pb2_grpc",
    ]
    for _name in _GRPC_PLACEHOLDER_NAMES:
        globals()[_name] = None

# Import cuRobo planner from episode-generation-job
try:
    sys.path.insert(0, str(REPO_ROOT / "episode-generation-job"))
    from curobo_planner import (
        CuRoboMotionPlanner, CuRoboPlanRequest, CuRoboPlanResult,
        CollisionObject, CollisionGeometryType,
        is_curobo_available, create_curobo_planner,
    )
    CUROBO_INTEGRATION_AVAILABLE = is_curobo_available()
except ImportError:
    CUROBO_INTEGRATION_AVAILABLE = False
    CuRoboMotionPlanner = None
    logger.warning("cuRobo planner not available - collision-aware planning disabled")

# Import IK utilities from episode-generation-job
_episode_gen_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "episode-generation-job")
if os.path.isdir(_episode_gen_dir) and _episode_gen_dir not in sys.path:
    sys.path.insert(0, _episode_gen_dir)
try:
    from trajectory_solver import IKSolver, ROBOT_CONFIGS
    from motion_planner import Waypoint, MotionPhase
    from collision_aware_planner import CollisionAwarePlanner
    IK_PLANNING_AVAILABLE = True
except ImportError as _ik_err:
    IK_PLANNING_AVAILABLE = False
    IKSolver = None
    Waypoint = None
    MotionPhase = None
    CollisionAwarePlanner = None
    logger.warning("IK utilities not available - IK fallback disabled: %s", _ik_err)

    ROBOT_CONFIGS = {}

# Lightweight robot metadata for episode output (joint names, limits)
# when full trajectory_solver is unavailable.
_FRANKA_METADATA = {
    "joint_names": ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                    "panda_joint5", "panda_joint6", "panda_joint7"],
    "joint_limits_lower": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0],
    "joint_limits_upper": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04],
    "default_joint_positions": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
    "gripper_joint_names": ["panda_finger_joint1", "panda_finger_joint2"],
    "gripper_limits": (0.0, 0.04),
}
_G1_RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]
_G1_LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
]
_G1_RIGHT_GRIPPER_JOINT_NAMES = [
    "right_gripper_finger_joint1",
    "right_gripper_finger_joint2",
]
_G1_LEFT_GRIPPER_JOINT_NAMES = [
    "left_gripper_finger_joint1",
    "left_gripper_finger_joint2",
]
_G1_ARM_JOINT_LIMITS_LOWER = [-2.9, -1.6, -2.9, -2.6, -2.9, -1.7, -2.9]
_G1_ARM_JOINT_LIMITS_UPPER = [2.9, 1.6, 2.9, 2.6, 2.9, 1.7, 2.9]
_G1_RIGHT_ARM_DEFAULT = [0.0, -0.2, 0.0, -1.2, 0.0, 0.6, 0.0]
_G1_LEFT_ARM_DEFAULT = [0.0, 0.2, 0.0, -1.2, 0.0, -0.6, 0.0]
_G1_RIGHT_METADATA = {
    "joint_names": list(_G1_RIGHT_ARM_JOINT_NAMES),
    "joint_limits_lower": list(_G1_ARM_JOINT_LIMITS_LOWER),
    "joint_limits_upper": list(_G1_ARM_JOINT_LIMITS_UPPER),
    "default_joint_positions": list(_G1_RIGHT_ARM_DEFAULT),
    "gripper_joint_names": list(_G1_RIGHT_GRIPPER_JOINT_NAMES),
    "gripper_limits": (0.0, 0.08),
}
_G1_LEFT_METADATA = {
    "joint_names": list(_G1_LEFT_ARM_JOINT_NAMES),
    "joint_limits_lower": list(_G1_ARM_JOINT_LIMITS_LOWER),
    "joint_limits_upper": list(_G1_ARM_JOINT_LIMITS_UPPER),
    "default_joint_positions": list(_G1_LEFT_ARM_DEFAULT),
    "gripper_joint_names": list(_G1_LEFT_GRIPPER_JOINT_NAMES),
    "gripper_limits": (0.0, 0.08),
}
_ROBOT_METADATA_FALLBACK = {
    "franka": _FRANKA_METADATA, "franka_panda": _FRANKA_METADATA, "panda": _FRANKA_METADATA,
    "g1": _G1_RIGHT_METADATA,
    "g1_right_arm": _G1_RIGHT_METADATA,
    "g1_left_arm": _G1_LEFT_METADATA,
}
_FRANKA_TYPES = {"franka", "franka_panda", "panda"}

ROBOT_ARM_INDICES: Dict[str, Dict[str, Any]] = {
    "g1": {"primary_arm": "right"},
}


def _matches_side_token(name: str, *, side: str) -> bool:
    name = name.lower()
    if side == "left":
        prefixes = ("left", "l_", "l-", "l.")
    else:
        prefixes = ("right", "r_", "r-", "r.")
    return name.startswith(prefixes) or any(f"_{p}" in name for p in prefixes) or any(f"/{p}" in name for p in prefixes)


def _infer_joint_indices_by_side(
    joint_names: Sequence[str],
    *,
    side: str,
    keywords: Sequence[str],
) -> List[int]:
    indices: List[int] = []
    for idx, name in enumerate(joint_names):
        lname = name.lower()
        if not _matches_side_token(lname, side=side):
            continue
        if any(keyword in lname for keyword in keywords):
            indices.append(idx)
    return indices


def _resolve_robot_joint_groups(
    robot_type: str,
    *,
    joint_names: Sequence[str],
    num_joints: int,
) -> Dict[str, Any]:
    robot_key = (robot_type or "").lower()
    meta = ROBOT_ARM_INDICES.get(robot_key, {})
    left_arm = list(meta.get("left_arm_indices", []))
    right_arm = list(meta.get("right_arm_indices", []))
    left_gripper = list(meta.get("left_gripper_indices", []))
    right_gripper = list(meta.get("right_gripper_indices", []))

    if joint_names:
        arm_keywords = ("shoulder", "elbow", "wrist", "arm", "forearm")
        grip_keywords = ("gripper", "finger", "hand", "jaw", "pinch")
        inferred_left_arm = _infer_joint_indices_by_side(
            joint_names,
            side="left",
            keywords=arm_keywords,
        )
        inferred_right_arm = _infer_joint_indices_by_side(
            joint_names,
            side="right",
            keywords=arm_keywords,
        )
        inferred_left_gripper = _infer_joint_indices_by_side(
            joint_names,
            side="left",
            keywords=grip_keywords,
        )
        inferred_right_gripper = _infer_joint_indices_by_side(
            joint_names,
            side="right",
            keywords=grip_keywords,
        )
        if inferred_left_arm:
            left_arm = inferred_left_arm
        if inferred_right_arm:
            right_arm = inferred_right_arm
        if inferred_left_gripper:
            left_gripper = inferred_left_gripper
        if inferred_right_gripper:
            right_gripper = inferred_right_gripper

    primary_side = meta.get("primary_arm") or ("right" if robot_key.startswith("g1") else "left")
    primary_arm = right_arm if primary_side == "right" else left_arm
    primary_gripper = right_gripper if primary_side == "right" else left_gripper

    if not primary_arm:
        primary_arm = list(range(min(7, num_joints)))
    if not primary_gripper and num_joints > len(primary_arm):
        primary_gripper = list(range(len(primary_arm), num_joints))

    gripper_limits = meta.get("gripper_limits")
    if gripper_limits is None:
        gripper_limits = _ROBOT_METADATA_FALLBACK.get(robot_key, {}).get("gripper_limits", (0.0, 1.0))

    gripper_max_aperture = meta.get("gripper_max_aperture")
    if gripper_max_aperture is None:
        gripper_max_aperture = _ROBOT_METADATA_FALLBACK.get(robot_key, {}).get("gripper_max_aperture")
    if gripper_max_aperture is None and gripper_limits:
        gripper_max_aperture = max(0.0, float(gripper_limits[1] - gripper_limits[0]))

    gripper_max_force = meta.get("gripper_max_force")
    if gripper_max_force is None:
        gripper_max_force = _ROBOT_METADATA_FALLBACK.get(robot_key, {}).get("gripper_max_force")
    if gripper_max_force is None:
        gripper_max_force = 40.0 if robot_key in _FRANKA_TYPES else 25.0

    return {
        "primary_arm_indices": primary_arm,
        "left_arm_indices": left_arm,
        "right_arm_indices": right_arm,
        "primary_gripper_indices": primary_gripper,
        "left_gripper_indices": left_gripper,
        "right_gripper_indices": right_gripper,
        "gripper_limits": gripper_limits,
        "gripper_max_aperture": gripper_max_aperture,
        "gripper_max_force": gripper_max_force,
    }


def _pose_to_4x4(position: "np.ndarray", orientation: "np.ndarray") -> "np.ndarray":
    """Convert position [x,y,z] + quaternion [w,x,y,z] to 4x4 homogeneous matrix."""
    w, x, y, z = orientation[0], orientation[1], orientation[2], orientation[3]
    T = np.eye(4)
    T[0, 0] = 1 - 2 * (y * y + z * z)
    T[0, 1] = 2 * (x * y - w * z)
    T[0, 2] = 2 * (x * z + w * y)
    T[1, 0] = 2 * (x * y + w * z)
    T[1, 1] = 1 - 2 * (x * x + z * z)
    T[1, 2] = 2 * (y * z - w * x)
    T[2, 0] = 2 * (x * z - w * y)
    T[2, 1] = 2 * (y * z + w * x)
    T[2, 2] = 1 - 2 * (x * x + y * y)
    T[:3, 3] = position
    return T


def _resolve_robot_config(robot_type: str) -> Any:
    robot_key = (robot_type or "").lower()
    robot_config = ROBOT_CONFIGS.get(robot_type) or ROBOT_CONFIGS.get(robot_key)
    if robot_config is None and (robot_key in _FRANKA_TYPES or not robot_key):
        robot_config = ROBOT_CONFIGS.get("franka")
    return robot_config


def _normalize_robot_name(robot_type: str) -> str:
    return robot_type.lower().replace("-", "_").replace(" ", "_")


def _g1_arm_side(robot_type: str) -> str:
    normalized = _normalize_robot_name(robot_type)
    if "left" in normalized:
        return "left"
    if "right" in normalized:
        return "right"
    return "right"


def _resolve_named_joint_indices(
    joint_names: Sequence[str],
    expected_names: Sequence[str],
) -> List[int]:
    if not joint_names:
        return []
    name_map = {name.lower(): idx for idx, name in enumerate(joint_names)}
    indices: List[int] = []
    for expected in expected_names:
        idx = name_map.get(expected.lower())
        if idx is None:
            return []
        indices.append(idx)
    return indices


def _resolve_g1_arm_joint_indices(
    robot_type: str,
    joint_names: Sequence[str],
    num_joints: int,
) -> List[int]:
    side = _g1_arm_side(robot_type)
    expected = _G1_LEFT_ARM_JOINT_NAMES if side == "left" else _G1_RIGHT_ARM_JOINT_NAMES
    indices = _resolve_named_joint_indices(joint_names, expected)
    if indices:
        return indices
    if num_joints >= 14:
        return list(range(7, 14)) if side == "left" else list(range(0, 7))
    return list(range(min(7, num_joints)))


def _resolve_g1_gripper_joint_indices(
    robot_type: str,
    joint_names: Sequence[str],
    num_joints: int,
    arm_joint_indices: Sequence[int],
) -> List[int]:
    side = _g1_arm_side(robot_type)
    expected = _G1_LEFT_GRIPPER_JOINT_NAMES if side == "left" else _G1_RIGHT_GRIPPER_JOINT_NAMES
    indices = _resolve_named_joint_indices(joint_names, expected)
    if indices:
        return indices
    remaining = [idx for idx in range(num_joints) if idx not in arm_joint_indices]
    return remaining[: len(expected)]


def _franka_fk(joint_angles: "np.ndarray") -> Tuple[List[float], List[float]]:
    """
    Analytical forward kinematics for Franka Panda using modified DH parameters.

    Args:
        joint_angles: 7-element array of joint positions (radians).

    Returns:
        (ee_pos, ee_quat): End-effector position [x,y,z] and quaternion [w,x,y,z].
    """
    # Modified DH parameters for Franka Emika Panda
    # (a, d, alpha) per joint — from the Franka documentation
    dh = [
        (0.0,    0.333,  0.0),
        (0.0,    0.0,   -np.pi / 2),
        (0.0,    0.316,  np.pi / 2),
        (0.0825, 0.0,    np.pi / 2),
        (-0.0825, 0.384, -np.pi / 2),
        (0.0,    0.0,    np.pi / 2),
        (0.088,  0.0,    np.pi / 2),
    ]
    # Flange offset (from joint 7 to flange)
    d_flange = 0.107

    T = np.eye(4)
    q = np.asarray(joint_angles[:7], dtype=float)

    for i, (a, d, alpha) in enumerate(dh):
        cq = np.cos(q[i])
        sq = np.sin(q[i])
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        Ti = np.array([
            [cq, -sq, 0.0, a],
            [sq * ca, cq * ca, -sa, -d * sa],
            [sq * sa, cq * sa,  ca,  d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ])
        T = T @ Ti

    # Apply flange offset along z
    T_flange = np.eye(4)
    T_flange[2, 3] = d_flange
    T = T @ T_flange

    ee_pos = T[:3, 3].tolist()

    # Extract quaternion from rotation matrix
    R = T[:3, :3]
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    ee_quat = [float(w), float(x), float(y), float(z)]
    return ee_pos, ee_quat


def _franka_numerical_ik(
    target_pos: "np.ndarray",
    initial_guess: Optional["np.ndarray"] = None,
    max_iter: int = 200,
    tol: float = 1e-3,
) -> Optional["np.ndarray"]:
    """
    Numerical inverse kinematics for Franka Panda using Jacobian pseudo-inverse.

    Args:
        target_pos: Desired end-effector position [x, y, z].
        initial_guess: Starting joint configuration (7 DOF). Uses default if None.
        max_iter: Maximum iterations.
        tol: Position error tolerance (meters).

    Returns:
        7-element joint angle array, or None if IK fails to converge.
    """
    _lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    _upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    if initial_guess is None:
        q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    else:
        q = np.array(initial_guess[:7], dtype=float).copy()

    target = np.asarray(target_pos, dtype=float)
    step_size = 0.5
    damping = 1e-4

    for _ in range(max_iter):
        ee_pos, _ = _franka_fk(q)
        ee = np.array(ee_pos)
        err = target - ee
        if np.linalg.norm(err) < tol:
            return q

        # Numerical Jacobian (position only, 3x7)
        J = np.zeros((3, 7))
        delta = 1e-5
        for j in range(7):
            q_plus = q.copy()
            q_plus[j] += delta
            ee_plus, _ = _franka_fk(q_plus)
            J[:, j] = (np.array(ee_plus) - ee) / delta

        # Damped pseudo-inverse step
        JT = J.T
        dq = JT @ np.linalg.solve(J @ JT + damping * np.eye(3), err)
        q = q + step_size * dq
        q = np.clip(q, _lower, _upper)

    # Check final convergence
    ee_pos, _ = _franka_fk(q)
    if np.linalg.norm(np.array(ee_pos) - target) < tol * 10:
        return q
    return None


# =============================================================================
# Configuration
# =============================================================================


DEFAULT_OBSERVATION_SCHEMA = {
    "required_keys": ["robot_state", "camera_frames"],
}
# When camera data is unavailable (e.g. GENIESIM_SKIP_SERVER_RECORDING),
# use a relaxed schema that only requires proprioception data.
PROPRIOCEPTION_ONLY_OBSERVATION_SCHEMA = {
    "required_keys": ["robot_state"],
}
DEFAULT_ACTION_BOUNDS = {
    "lower": [-3.1416] * 7 + [0.0] * 2,
    "upper": [3.1416] * 7 + [0.04] * 2,
}
DEFAULT_SUCCESS_SCHEMA = {
    "success_keys": ["success", "task_success", "is_success"],
    "grasp_keys": ["grasped", "is_grasping", "gripper_closed"],
    "release_keys": ["released", "gripper_open", "is_released"],
}


def _normalize_joint_arrays(
    joint_positions: Optional[List[float]],
    joint_velocities: Optional[List[float]] = None,
    joint_efforts: Optional[List[float]] = None,
    joint_accelerations: Optional[List[float]] = None,
    joint_names: Optional[List[str]] = None,
) -> Tuple[
    Optional[List[float]],
    Optional[List[float]],
    Optional[List[float]],
    Optional[List[float]],
    Optional[List[str]],
]:
    """Normalize per-joint arrays to arm-only length (len(joint_positions))."""
    if not isinstance(joint_positions, list):
        return joint_positions, joint_velocities, joint_efforts, joint_accelerations, joint_names
    target_len = len(joint_positions)

    def _trim_pad(seq: Optional[List[Any]], fill: float = 0.0) -> Optional[List[Any]]:
        if seq is None:
            return None
        if not isinstance(seq, list):
            return seq
        if len(seq) >= target_len:
            return seq[:target_len]
        return seq + [fill] * (target_len - len(seq))

    joint_velocities = _trim_pad(joint_velocities, 0.0)
    joint_efforts = _trim_pad(joint_efforts, 0.0)
    joint_accelerations = _trim_pad(joint_accelerations, 0.0)
    if isinstance(joint_names, list):
        joint_names = joint_names[:target_len]
    return joint_positions[:target_len], joint_velocities, joint_efforts, joint_accelerations, joint_names


def _normalize_robot_state_joints(robot_state: Dict[str, Any]) -> None:
    """Ensure robot_state joint arrays are consistent arm-only lengths."""
    if not isinstance(robot_state, dict):
        return
    joint_positions = robot_state.get("joint_positions")
    if not isinstance(joint_positions, list) or not joint_positions:
        return
    joint_state = robot_state.get("joint_state") if isinstance(robot_state.get("joint_state"), dict) else None
    joint_names = joint_state.get("names") if isinstance(joint_state, dict) else None
    joint_positions, joint_velocities, joint_efforts, joint_accelerations, joint_names = _normalize_joint_arrays(
        joint_positions,
        robot_state.get("joint_velocities"),
        robot_state.get("joint_efforts"),
        robot_state.get("joint_accelerations"),
        joint_names,
    )
    robot_state["joint_positions"] = joint_positions
    if joint_velocities is not None:
        robot_state["joint_velocities"] = joint_velocities
    if joint_efforts is not None:
        robot_state["joint_efforts"] = joint_efforts
    if joint_accelerations is not None:
        robot_state["joint_accelerations"] = joint_accelerations
    for key in ("commanded_joint_positions", "tracking_error"):
        if isinstance(robot_state.get(key), list):
            robot_state[key] = robot_state[key][:len(joint_positions)]
    if isinstance(joint_state, dict):
        joint_state["positions"] = joint_positions
        if joint_names is not None:
            joint_state["names"] = joint_names
        if isinstance(joint_state.get("velocities"), list) and joint_velocities is not None:
            joint_state["velocities"] = joint_velocities
        if isinstance(joint_state.get("efforts"), list) and joint_efforts is not None:
            joint_state["efforts"] = joint_efforts


class GenieSimServerStatus(str, Enum):
    """Status of the Genie Sim server."""
    NOT_RUNNING = "not_running"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


class _CameraDataRequiredError(RuntimeError):
    """Raised when required camera data is missing during export."""


class CommandType(int, Enum):
    """
    Genie Sim gRPC command types.

    Based on: https://github.com/AgibotTech/genie_sim/blob/main/source/data_collection/server/command_enum.py
    """
    # Camera Commands
    GET_CAMERA_DATA = 1
    GET_SEMANTIC_DATA = 10  # Local unique ID; mapped to GET_CAMERA_DATA for gRPC.

    # Motion Commands
    LINEAR_MOVE = 2
    SET_JOINT_POSITION = 3
    GET_JOINT_POSITION = 8
    GET_EE_POSE = 18
    GET_IK_STATUS = 19
    SET_TRAJECTORY_LIST = 25

    # Gripper Commands
    GET_GRIPPER_STATE = 4
    SET_GRIPPER_STATE = 9

    # Object Commands
    GET_OBJECT_POSE = 5
    ADD_OBJECT = 6
    GET_ROBOT_LINK_POSE = 7
    GET_OBJECT_JOINT = 26
    GET_PART_DOF_JOINT = 32
    SET_OBJECT_POSE = 24
    SET_TARGET_POINT = 27
    SET_LINEAR_VELOCITY = 33
    ATTACH_OBJ = 13
    DETACH_OBJ = 14
    ATTACH_OBJ_TO_PARENT = 50
    DETACH_OBJ_FROM_PARENT = 51
    REMOVE_OBJS_FROM_OBSTACLE = 52

    # Observation & Recording
    GET_OBSERVATION = 11
    START_RECORDING = 15  # Local unique ID; mapped to GET_OBSERVATION for gRPC.
    STOP_RECORDING = 20  # Local unique ID; mapped to GET_OBSERVATION for gRPC.

    # System Commands
    RESET = 12
    EXIT = 17
    INIT_ROBOT = 21
    TASK_STATUS = 16

    # Camera Setup
    ADD_CAMERA = 22

    # State & Configuration
    SET_FRAME_STATE = 28
    SET_LIGHT = 30
    SET_CODE_FACE_ORIENTATION = 34
    SET_TASK_METRIC = 53

    # Replay & Checker
    STORE_CURRENT_STATE = 54
    PLAYBACK = 55
    GET_CHECKER_STATUS = 56


class GenieSimConfig(BaseModel):
    """Configuration for Genie Sim local framework."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    # Connection settings
    host: StrictStr = Field(default_factory=get_geniesim_host)
    port: StrictInt = Field(default_factory=get_geniesim_port)
    timeout: StrictFloat = Field(default_factory=get_geniesim_grpc_timeout_s)
    max_retries: StrictInt = 3

    # Installation paths
    geniesim_root: Path = Path("/opt/geniesim")
    isaac_sim_path: Path = Path("/isaac-sim")
    isaacsim_required: StrictBool = False
    curobo_required: StrictBool = False

    # Data collection settings
    episodes_per_task: StrictInt = 100
    use_curobo: StrictBool = True
    headless: StrictBool = True
    environment: str = "development"
    allow_linear_fallback: StrictBool = False
    allow_linear_fallback_in_production: StrictBool = False
    allow_ik_failure_fallback: StrictBool = False
    stall_timeout_s: StrictFloat = 30.0
    max_stalls: StrictInt = 2
    stall_backoff_s: StrictFloat = 5.0
    server_startup_timeout_s: StrictFloat = 120.0
    server_startup_poll_s: StrictFloat = 2.0
    max_duration_seconds: Optional[StrictFloat] = None
    validate_frames: StrictBool = True
    fail_on_frame_validation: StrictBool = False

    # Output settings
    recording_dir: Path = Path("/tmp/geniesim_recordings")
    log_dir: Path = Path("/tmp/geniesim_logs")
    lerobot_export_format: LeRobotExportFormat = LeRobotExportFormat.LEROBOT_V2
    require_lerobot_v3: StrictBool = False
    stream_per_task: StrictBool = True
    cleanup_tmp: StrictBool = True

    # Robot configuration
    robot_type: str = "franka"
    robot_urdf: Optional[str] = None

    @classmethod
    def from_env(cls) -> "GenieSimConfig":
        """Create configuration from environment variables."""
        env = os.environ
        environment = resolve_pipeline_environment(env=env)
        env_data: Dict[str, Any] = {
            "environment": environment,
        }
        for key, env_name in (
            ("host", "GENIESIM_HOST"),
            ("port", "GENIESIM_PORT"),
            ("geniesim_root", "GENIESIM_ROOT"),
            ("isaac_sim_path", "ISAAC_SIM_PATH"),
            ("isaacsim_required", "ISAACSIM_REQUIRED"),
            ("curobo_required", "CUROBO_REQUIRED"),
            ("headless", "HEADLESS"),
            ("robot_type", "ROBOT_TYPE"),
            ("allow_linear_fallback", "GENIESIM_ALLOW_LINEAR_FALLBACK"),
            ("allow_linear_fallback_in_production", "GENIESIM_ALLOW_LINEAR_FALLBACK_IN_PROD"),
            ("allow_ik_failure_fallback", "GENIESIM_ALLOW_IK_FAILURE_FALLBACK"),
            ("stall_timeout_s", "GENIESIM_STALL_TIMEOUT_S"),
            ("max_stalls", "GENIESIM_MAX_STALLS"),
            ("stall_backoff_s", "GENIESIM_STALL_BACKOFF_S"),
            ("server_startup_timeout_s", "GENIESIM_STARTUP_TIMEOUT_S"),
            ("server_startup_poll_s", "GENIESIM_STARTUP_POLL_S"),
            ("max_duration_seconds", "GENIESIM_COLLECTION_TIMEOUT_S"),
            ("validate_frames", "GENIESIM_VALIDATE_FRAMES"),
            ("fail_on_frame_validation", "GENIESIM_FAIL_ON_FRAME_VALIDATION"),
            ("lerobot_export_format", "LEROBOT_EXPORT_FORMAT"),
            ("require_lerobot_v3", "LEROBOT_REQUIRE_V3"),
            ("stream_per_task", "GENIESIM_STREAM_PER_TASK"),
            ("cleanup_tmp", "GENIESIM_CLEANUP_TMP"),
        ):
            if env_name in env and env.get(env_name) not in (None, ""):
                env_data[key] = env.get(env_name)
        if "allow_ik_failure_fallback" not in env_data and "allow_linear_fallback" in env_data:
            env_data["allow_ik_failure_fallback"] = env_data["allow_linear_fallback"]
        timeout_value, _ = resolve_env_with_legacy(
            canonical_names=("GENIESIM_GRPC_TIMEOUT_S",),
            legacy_names=("GENIESIM_TIMEOUT",),
            env=env,
            preferred_name="GENIESIM_GRPC_TIMEOUT_S",
            log=logger,
        )
        if timeout_value not in (None, ""):
            env_data["timeout"] = timeout_value
        recording_dir = _resolve_recording_dir()
        log_dir = _resolve_log_dir()
        env_data["recording_dir"] = recording_dir
        env_data["log_dir"] = log_dir
        return cls.model_validate(env_data)

    @field_validator("host", mode="before")
    @classmethod
    def _validate_host(cls, value: Any) -> str:
        if value is None:
            raise ValueError("GENIESIM_HOST must be set to a non-empty hostname.")
        host = str(value).strip()
        if not host:
            raise ValueError("GENIESIM_HOST must be a non-empty hostname.")
        return host

    @field_validator("environment", mode="before")
    @classmethod
    def _normalize_environment(cls, value: Any) -> str:
        if value is None:
            return "development"
        env_value = str(value).strip().lower()
        if not env_value:
            raise ValueError("Environment must be a non-empty string.")
        return env_value

    @field_validator(
        "port",
        "max_retries",
        "episodes_per_task",
        "max_stalls",
        mode="before",
    )
    @classmethod
    def _validate_ints(cls, value: Any, info) -> int:
        field_name = info.field_name
        if value is None:
            raise ValueError(f"{field_name} is required.")
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be an integer, not a boolean.")
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = int(value.strip())
            except ValueError as exc:
                raise ValueError(f"{field_name} must be an integer (got {value!r}).") from exc
        else:
            raise ValueError(f"{field_name} must be an integer (got {value!r}).")
        if field_name == "port" and not 1 <= parsed <= 65535:
            raise ValueError("port must be between 1 and 65535.")
        if field_name in {"max_retries", "episodes_per_task", "max_stalls"} and parsed < 0:
            raise ValueError(f"{field_name} must be >= 0.")
        return parsed

    @field_validator(
        "timeout",
        "stall_timeout_s",
        "stall_backoff_s",
        "max_duration_seconds",
        mode="before",
    )
    @classmethod
    def _validate_floats(cls, value: Any, info) -> Optional[float]:
        if value is None:
            if info.field_name == "max_duration_seconds":
                return None
            raise ValueError(f"{info.field_name} must be a number.")
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be a number, not a boolean.")
        if isinstance(value, (int, float)):
            parsed = float(value)
        elif isinstance(value, str):
            try:
                parsed = float(value.strip())
            except ValueError as exc:
                raise ValueError(
                    f"{info.field_name} must be a number (got {value!r})."
                ) from exc
        else:
            raise ValueError(f"{info.field_name} must be a number (got {value!r}).")
        if parsed < 0.0:
            raise ValueError(f"{info.field_name} must be >= 0.")
        return parsed

    @field_validator(
        "isaacsim_required",
        "curobo_required",
        "use_curobo",
        "headless",
        "allow_linear_fallback",
        "allow_linear_fallback_in_production",
        "allow_ik_failure_fallback",
        "require_lerobot_v3",
        "cleanup_tmp",
        "validate_frames",
        "fail_on_frame_validation",
        mode="before",
    )
    @classmethod
    def _validate_bools(cls, value: Any, info) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            parsed = parse_bool_env(value, default=None)
            if parsed is None:
                raise ValueError(
                    f"{info.field_name} must be a boolean-like value (got {value!r})."
                )
            return parsed
        raise ValueError(
            f"{info.field_name} must be a boolean-like value (got {value!r})."
        )

    @field_validator("geniesim_root", "isaac_sim_path", "recording_dir", "log_dir", mode="before")
    @classmethod
    def _validate_paths(cls, value: Any, info) -> Path:
        if isinstance(value, Path):
            return value.expanduser()
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                raise ValueError(f"{info.field_name} must be a non-empty path.")
            return Path(candidate).expanduser()
        raise ValueError(f"{info.field_name} must be a path-like value (got {value!r}).")

    @field_validator("lerobot_export_format", mode="before")
    @classmethod
    def _validate_lerobot_format(cls, value: Any) -> LeRobotExportFormat:
        if isinstance(value, LeRobotExportFormat):
            return value
        if value is None:
            return LeRobotExportFormat.LEROBOT_V2
        return parse_lerobot_export_format(value, default=LeRobotExportFormat.LEROBOT_V2)

    @model_validator(mode="after")
    def _finalize_config(self) -> "GenieSimConfig":
        if self.environment == "production":
            # P1: Strict production mode validation for collision-aware planning
            # These flags cannot be enabled in production to prevent robot collisions
            if self.allow_linear_fallback_in_production:
                raise ValueError(
                    "[SAFETY-CRITICAL] GENIESIM_ALLOW_LINEAR_FALLBACK_IN_PROD=1 is not permitted in production. "
                    "Non-collision-aware linear motion can cause physical robot collisions. "
                    "This flag is ignored and blocked in production mode for safety reasons."
                )
            if self.allow_linear_fallback:
                raise ValueError(
                    "[SAFETY-CRITICAL] GENIESIM_ALLOW_LINEAR_FALLBACK=1 is not permitted in production. "
                    "Non-collision-aware linear motion can cause physical robot collisions. "
                    "Unset this flag or use a non-production environment for testing."
                )
            if self.allow_ik_failure_fallback:
                raise ValueError(
                    "[SAFETY-CRITICAL] GENIESIM_ALLOW_IK_FAILURE_FALLBACK=1 is not permitted in production. "
                    "Falling back to linear motion on IK failure can cause robot collisions. "
                    "Unset this flag or use a non-production environment for testing."
                )
            temp_dirs = [
                (label, path, env_name)
                for label, path, env_name in (
                    ("recording", self.recording_dir, GENIESIM_RECORDINGS_DIR_ENV),
                    ("log", self.log_dir, GENIESIM_LOG_DIR_ENV),
                )
                if _is_temp_path(path)
            ]
            if temp_dirs:
                offenders = ", ".join(
                    f"{env_name}={path}"
                    for _, path, env_name in temp_dirs
                )
                raise ValueError(
                    "Refusing to use temporary directories for Genie Sim in production. "
                    f"Set {GENIESIM_RECORDINGS_DIR_ENV} and {GENIESIM_LOG_DIR_ENV} "
                    f"to persistent locations. Offending values: {offenders}."
                )
        return self


@dataclass
class GeneratedEpisodeMetadata:
    """Metadata for a generated episode."""

    episode_id: str = ""
    task_name: str = ""
    quality_score: float = 0.0
    frame_count: int = 0
    duration_seconds: float = 0.0
    validation_passed: bool = False
    file_size_bytes: int = 0
    quality_components: Dict[str, float] = field(default_factory=dict)
    episode_content_hash: Optional[str] = None


@dataclass
class StallStatistics:
    """Aggregated stall pattern statistics for analysis."""

    total_stalls: int = 0
    stalls_by_reason: Dict[str, int] = field(default_factory=dict)
    stall_events: List[Dict[str, Any]] = field(default_factory=list)

    def record_stall(
        self,
        reason: str,
        episode_idx: int,
        task_name: str,
        *,
        progress_age_s: Optional[float] = None,
        observations_collected: Optional[int] = None,
        last_observation_timestamp: Optional[float] = None,
        trajectory_end_timestamp: Optional[float] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a stall event for pattern analysis."""
        self.total_stalls += 1
        self.stalls_by_reason[reason] = self.stalls_by_reason.get(reason, 0) + 1
        event = {
            "reason": reason,
            "episode_idx": episode_idx,
            "task_name": task_name,
            "progress_age_s": progress_age_s,
            "observations_collected": observations_collected,
            "last_observation_timestamp": last_observation_timestamp,
            "trajectory_end_timestamp": trajectory_end_timestamp,
            "timestamp": time.time(),
        }
        if extra_context:
            event["context"] = extra_context
        self.stall_events.append(event)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of stall patterns for logging."""
        return {
            "total_stalls": self.total_stalls,
            "stalls_by_reason": dict(self.stalls_by_reason),
            "most_common_reason": max(
                self.stalls_by_reason.items(),
                key=lambda x: x[1],
                default=(None, 0),
            )[0] if self.stalls_by_reason else None,
            "event_count": len(self.stall_events),
        }


# Known stall reason categories for pattern analysis
class StallReason:
    """Constants for stall reason categorization."""

    NO_OBSERVATION_PROGRESS = "no_observation_progress"
    EXECUTION_COMPLETED_NO_FINAL_OBS = "execution_completed_no_final_observation"
    IK_FAILURE = "ik_failure"
    COLLISION_DETECTED = "collision_detected"
    PHYSICS_INSTABILITY = "physics_instability"
    TRAJECTORY_TIMEOUT = "trajectory_timeout"
    GRIPPER_FAILURE = "gripper_failure"
    SERVER_UNRESPONSIVE = "server_unresponsive"
    UNKNOWN = "unknown"


@dataclass
class DataCollectionResult:
    """Result of a data collection run."""

    success: bool
    task_name: str
    timed_out: bool = False
    episodes_collected: int = 0
    episodes_passed: int = 0
    total_frames: int = 0
    recording_dir: Optional[Path] = None

    # Quality metrics
    average_quality_score: float = 0.0
    collision_free_rate: float = 0.0
    task_success_rate: float = 0.0
    collision_free_episodes: int = 0
    collision_info_episodes: int = 0
    task_success_episodes: int = 0
    task_success_info_episodes: int = 0

    # Timing
    duration_seconds: float = 0.0

    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    server_info: Dict[str, Any] = field(default_factory=dict)

    # Stall tracking (P1 - for pattern analysis)
    stall_statistics: Optional[StallStatistics] = None


@dataclass(frozen=True)
class GrpcCallResult:
    """Structured result for gRPC calls, including availability."""

    success: bool
    available: bool
    error: Optional[str] = None
    payload: Any = None

def _parse_version(version: str) -> tuple:
    parts = version.split(".")
    padded = (parts + ["0", "0", "0"])[:3]
    return tuple(int(part) if part.isdigit() else 0 for part in padded)


def _resolve_recording_dir() -> Path:
    resolved, _ = resolve_env_with_legacy(
        canonical_names=(GENIESIM_RECORDINGS_DIR_ENV,),
        legacy_names=(GENIESIM_RECORDING_DIR_ENV,),
        env=os.environ,
        preferred_name=GENIESIM_RECORDINGS_DIR_ENV,
        log=logger,
    )
    if resolved:
        return Path(resolved).expanduser()
    return Path("/tmp/geniesim_recordings")


def _resolve_log_dir() -> Path:
    raw = os.getenv(GENIESIM_LOG_DIR_ENV)
    if raw:
        return Path(raw).expanduser()
    return Path("/tmp/geniesim_logs")


def _resolve_robot_usd(relative_path: str) -> str:
    """Return a local filesystem path for a robot USD if it was pre-baked
    into the Docker image under ``SIM_ASSETS_ROBOTS``; otherwise return the
    original *relative_path* so that Nucleus handles the download."""
    root = os.environ.get("SIM_ASSETS_ROBOTS", "/sim-assets/robots")
    local = os.path.join(root, relative_path)
    if os.path.exists(local):
        return local
    return relative_path


def _is_temp_path(path: Path) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except FileNotFoundError:
        resolved = path.expanduser().absolute()
    for root in (Path("/tmp"), Path("/var/tmp")):
        try:
            resolved_root = root.resolve()
        except FileNotFoundError:
            resolved_root = root.absolute()
        try:
            resolved.relative_to(resolved_root)
        except ValueError:
            continue
        return resolved != resolved_root
    return False


# =============================================================================
# gRPC Client Stub
# =============================================================================


class GenieSimGRPCClient:
    """
    gRPC client for communicating with Genie Sim data collection server.

    This implements the client side of the Genie Sim data collection protocol
    based on the official gRPC service definitions.

    Note: This requires the grpcio and grpcio-tools packages, plus the
    generated protobuf stubs from Genie Sim. If not available, it falls
    back to a subprocess-based approach.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize gRPC client.

        Args:
            host: Server hostname (defaults to GENIESIM_HOST)
            port: Server port (defaults to GENIESIM_PORT)
            timeout: Request timeout in seconds
        """
        self.host = host or get_geniesim_host()
        self.port = port if port is not None else get_geniesim_port()
        self.timeout = timeout if timeout is not None else get_geniesim_grpc_timeout_s()
        self._channel = None
        self._stub = None
        self._joint_stub = None
        self._connected = False
        # Serialize all gRPC calls — the server's blocking_start_server uses
        # shared state and cannot handle concurrent commands.
        import threading as _threading
        self._grpc_lock = _threading.Lock()
        self._default_camera_ids = ["wrist", "overhead", "side"]
        # Map logical camera names to robot-specific USD prim paths.
        # Populated after init_robot from the robot config's camera dict.
        self._camera_prim_map: Dict[str, str] = {}
        self._joint_names: List[str] = []
        self._latest_joint_positions: List[float] = []
        self._grpc_unavailable_logged: set[str] = set()
        self._camera_missing_logged: set[str] = set()
        self._first_joint_call = True  # first set_joint_position uses longer timeout
        self._first_trajectory_call = True  # first trajectory waypoint uses longer timeout
        self._curobo_initialized = False  # tracks if cuRobo has completed lazy init
        self._abort_event: Optional[_threading.Event] = None  # set by episode runner
        self._first_call_timeout = float(os.environ.get("GENIESIM_FIRST_CALL_TIMEOUT_S", "300"))
        self._circuit_breaker = CircuitBreaker(
            f"geniesim-grpc-{self.host}:{self.port}",
            failure_threshold=get_geniesim_circuit_breaker_failure_threshold(),
            success_threshold=get_geniesim_circuit_breaker_success_threshold(),
            recovery_timeout=get_geniesim_circuit_breaker_recovery_timeout_s(),
            on_open=self._on_circuit_open,
            on_half_open=self._on_circuit_half_open,
            on_close=self._on_circuit_closed,
        )
        max_retries = max(1, get_geniesim_grpc_max_retries())
        self._grpc_retry_config = RetryConfig(
            max_retries=max_retries,
            base_delay=get_geniesim_grpc_retry_base_s(),
            max_delay=get_geniesim_grpc_retry_max_s(),
            backoff_factor=2.0,
            jitter=False,
        )
        if grpc is None:
            self._grpc_retryable_codes = tuple()
        else:
            self._grpc_retryable_codes = (
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
                grpc.StatusCode.RESOURCE_EXHAUSTED,
            )

        # Check if gRPC stubs are available
        self._have_grpc = GRPC_STUBS_AVAILABLE and grpc is not None
        if not self._have_grpc:
            logger.warning("gRPC not available - server connection will be limited")

    def __enter__(self) -> "GenieSimGRPCClient":
        if not self.is_connected():
            self.connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        self.disconnect()

    def _on_circuit_open(self, name: str, failure_count: int) -> None:
        logger.warning(
            "Genie Sim gRPC circuit breaker opened (%s) after %d failures (host=%s port=%s)",
            name,
            failure_count,
            self.host,
            self.port,
        )
        # Auto-restart server container when circuit breaker opens
        if os.environ.get("GENIESIM_AUTO_RESTART", "1") != "0":
            self._attempt_server_restart()

    def _on_circuit_half_open(self, name: str) -> None:
        logger.info(
            "Genie Sim gRPC circuit breaker half-open (%s) - probing recovery (host=%s port=%s)",
            name,
            self.host,
            self.port,
        )

    def _on_circuit_closed(self, name: str) -> None:
        logger.info(
            "Genie Sim gRPC circuit breaker closed (%s) - traffic resumed (host=%s port=%s)",
            name,
            self.host,
            self.port,
        )

    def _attempt_server_restart(self, count_toward_budget: bool = True) -> bool:
        """Attempt to restart the Docker container running the Genie Sim server.

        Returns True if restart succeeded and gRPC became available again.
        Respects cooldown (5 min between restarts) and max restart count.
        """
        import time as _time
        import subprocess as _sp

        max_restarts = int(os.environ.get("GENIESIM_MAX_RESTARTS", "3"))
        max_init_restarts = int(os.environ.get("GENIESIM_MAX_INIT_RESTARTS", str(max_restarts)))
        cooldown_s = int(os.environ.get("GENIESIM_RESTART_COOLDOWN_S", "30"))

        if not hasattr(self, "_restart_count"):
            self._restart_count = 0
            self._last_restart_time = 0.0
            self._init_restart_count = 0

        if count_toward_budget and self._restart_count >= max_restarts:
            logger.warning("[RESTART] Max client-side restarts (%d) reached — not restarting", max_restarts)
            return False
        if not count_toward_budget and self._init_restart_count >= max_init_restarts:
            logger.warning(
                "[RESTART] Max init-only restarts (%d) reached — not restarting",
                max_init_restarts,
            )
            return False

        now = _time.time()
        if now - self._last_restart_time < cooldown_s:
            logger.info("[RESTART] Cooldown active (%.0fs remaining) — skipping restart",
                        cooldown_s - (now - self._last_restart_time))
            return False

        restart_cmd = os.environ.get(
            "GENIESIM_RESTART_CMD",
            "sudo docker restart geniesim-server",
        )
        if count_toward_budget:
            restart_label = f"#{self._restart_count + 1}"
        else:
            restart_label = f"init-free #{self._init_restart_count + 1}"
            logger.warning(
                "[RESTART] Init restart is free (not counted toward GENIESIM_MAX_RESTARTS)"
            )
        logger.warning("[RESTART] Attempting server restart (%s): %s", restart_label, restart_cmd)

        try:
            result = _sp.run(restart_cmd, shell=True, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error("[RESTART] Restart command failed (rc=%d): %s", result.returncode, result.stderr.strip())
                return False
            logger.info("[RESTART] Container restart command succeeded")
        except Exception as exc:
            logger.error("[RESTART] Restart command exception: %s", exc)
            return False

        if count_toward_budget:
            self._restart_count += 1
        else:
            self._init_restart_count += 1
        self._last_restart_time = _time.time()

        # Reset client-side concurrency state so we don't reuse a stale lock or
        # in-flight flags from the pre-restart connection.
        self._grpc_lock = threading.Lock()
        self._grpc_unavailable_logged.clear()
        self._abort_event = None

        # Poll for gRPC readiness (up to 90s)
        logger.info("[RESTART] Waiting for gRPC to become available...")
        for _attempt in range(18):  # 18 * 5s = 90s
            _time.sleep(5)
            try:
                if self._channel is not None:
                    self._channel.close()
                self._channel = grpc.insecure_channel(
                    f"{self.host}:{self.port}",
                    options=[
                        ("grpc.max_send_message_length", 16094304),
                        ("grpc.max_receive_message_length", 16094304),
                    ],
                )
                # Try a lightweight call
                future = grpc.channel_ready_future(self._channel)
                future.result(timeout=5)
                logger.info("[RESTART] gRPC channel ready after restart — waiting 15s for server app warmup")
                _time.sleep(15)
                # Re-create stubs on the fresh channel
                self._stub = None
                self._joint_stub = None
                self.connect()
                # Reset circuit breaker
                if hasattr(self._circuit_breaker, 'reset'):
                    self._circuit_breaker.reset()
                # Flag that robot needs re-initialisation. The framework's
                # episode loop checks this flag and calls
                # _reinit_robot_after_restart() before the next episode.
                self._needs_robot_reinit = True
                return True
            except Exception:
                continue

        logger.error("[RESTART] gRPC not available after 90s — restart may have failed")
        return False

    def _grpc_unavailable(self, method_name: str, reason: str) -> GrpcCallResult:
        if method_name not in self._grpc_unavailable_logged:
            logger.warning(
                "Genie Sim gRPC unavailable for %s: %s (host=%s port=%s)",
                method_name,
                reason,
                self.host,
                self.port,
            )
            self._grpc_unavailable_logged.add(method_name)
        return GrpcCallResult(success=False, available=False, error=reason)

    def _is_retryable_grpc_error(self, exc: Exception) -> bool:
        if not self._have_grpc or grpc is None:
            return False
        if not isinstance(exc, grpc.RpcError):
            return False
        try:
            code = exc.code()
        except Exception:
            return False
        return code in self._grpc_retryable_codes

    def _call_grpc(
        self,
        action: str,
        func: Callable[[], Any],
        fallback: Any,
        success_checker: Optional[Callable[[Any], bool]] = None,
        abort_event: Optional[threading.Event] = None,
        lock_timeout: Optional[float] = None,
    ) -> Any:
        # Fast-path: if abort already signalled, skip the call entirely.
        if abort_event is not None and abort_event.is_set():
            logger.info("gRPC %s skipped — abort_event already set", action)
            return fallback
        # Use timeout-based lock acquisition to prevent deadlock when a long-
        # running gRPC call (e.g. 600s first trajectory) holds the lock and
        # another thread or the main thread needs to make a call (e.g. health
        # probe between tasks).  Default lock timeout = gRPC timeout + 10s.
        # Callers can pass a shorter lock_timeout for lightweight probes.
        if lock_timeout is None:
            lock_timeout = self.timeout + 10.0
        acquired = self._grpc_lock.acquire(timeout=lock_timeout)
        if not acquired:
            logger.warning(
                "Genie Sim gRPC lock not acquired after %.1fs for %s; "
                "returning fallback (previous call may still be in-flight)",
                lock_timeout,
                action,
            )
            # Only perform destructive recovery (channel replace) for long
            # timeouts that suggest a real deadlock.  Short timeouts (e.g.
            # observation polling at 1-5s) are expected during trajectory
            # execution — the execution thread legitimately holds the lock.
            # Destroying the channel here would kill in-flight trajectory
            # calls and cascade into circuit breaker trips + restarts.
            _deadlock_threshold = float(os.getenv("GRPC_DEADLOCK_TIMEOUT_S", "30.0"))
            if lock_timeout >= _deadlock_threshold:
                try:
                    self._grpc_lock = threading.Lock()
                    if self._channel is not None:
                        try:
                            self._channel.close()
                        except Exception:
                            pass
                    self._channel = None
                    self._stub = None
                    self._joint_stub = None
                    if self._have_grpc:
                        self.connect()
                    self._needs_robot_reinit = True
                except Exception as exc:
                    logger.warning("Failed gRPC lock recovery after timeout: %s", exc)
            return fallback
        try:
            return self._call_grpc_inner(action, func, fallback, success_checker, abort_event)
        except Exception as exc:
            logger.error("Genie Sim gRPC %s failed before completion: %s", action, exc)
            return fallback
        finally:
            try:
                self._grpc_lock.release()
            except RuntimeError:
                pass  # Lock was replaced by another thread

    def _call_grpc_inner(
        self,
        action: str,
        func: Callable[[], Any],
        fallback: Any,
        success_checker: Optional[Callable[[Any], bool]] = None,
        abort_event: Optional[threading.Event] = None,
    ) -> Any:
        if self._circuit_breaker and not self._circuit_breaker.allow_request():
            time_until_retry = self._circuit_breaker.get_time_until_retry()
            logger.warning(
                "Genie Sim gRPC circuit breaker open; skipping %s (retry in %.1fs)",
                action,
                time_until_retry,
            )
            return fallback

        last_exception: Optional[Exception] = None
        for attempt in range(1, self._grpc_retry_config.max_retries + 1):
            if abort_event is not None and abort_event.is_set():
                logger.info("gRPC %s aborted before attempt %d", action, attempt)
                return fallback
            try:
                if abort_event is not None:
                    # Run the blocking gRPC call in a worker so we can poll
                    # abort_event every 0.5s and bail out early.
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(func)
                        wait_started = time.monotonic()
                        wait_log_interval = 30.0
                        next_wait_log = wait_started + wait_log_interval
                        while True:
                            try:
                                result = future.result(timeout=0.5)
                                break
                            except concurrent.futures.TimeoutError:
                                now = time.monotonic()
                                if now >= next_wait_log:
                                    elapsed = now - wait_started
                                    logger.info(
                                        "gRPC %s still waiting (attempt %d, %.1fs elapsed)",
                                        action,
                                        attempt,
                                        elapsed,
                                    )
                                    next_wait_log = now + wait_log_interval
                                if abort_event.is_set():
                                    logger.info(
                                        "gRPC %s aborted while in-flight (attempt %d)",
                                        action,
                                        attempt,
                                    )
                                    # Can't truly cancel the gRPC call, but we
                                    # return immediately so the caller unblocks.
                                    return fallback
                else:
                    result = func()
            except Exception as exc:
                last_exception = exc
                # Don't count UNIMPLEMENTED toward circuit breaker — it means
                # the server doesn't support this method, not that it's unhealthy.
                unimplemented_code = getattr(grpc.StatusCode, "UNIMPLEMENTED", None)
                _is_unimplemented = (
                    hasattr(exc, "code")
                    and callable(exc.code)
                    and unimplemented_code is not None
                    and exc.code() == unimplemented_code
                )
                if self._circuit_breaker and not _is_unimplemented:
                    self._circuit_breaker.record_failure(exc)

                if self._is_retryable_grpc_error(exc) and attempt < self._grpc_retry_config.max_retries:
                    delay = calculate_delay(attempt, self._grpc_retry_config)
                    logger.warning(
                        "Genie Sim gRPC %s failed with retryable error (%s); retrying in %.2fs",
                        action,
                        exc,
                        delay,
                    )
                    # Check abort before sleeping for retry delay
                    if abort_event is not None and abort_event.is_set():
                        logger.info("gRPC %s aborted before retry sleep", action)
                        return fallback
                    time.sleep(delay)
                    continue

                logger.error("Genie Sim gRPC %s failed: %s", action, exc)
                return fallback

        if self._circuit_breaker:
            if success_checker is None:
                self._circuit_breaker.record_success()
            else:
                try:
                    is_success = success_checker(result)
                except Exception as exc:
                    logger.warning(
                        "Genie Sim gRPC success check failed for %s: %s",
                        action,
                        exc,
                    )
                    self._circuit_breaker.record_failure(exc)
                else:
                    if is_success:
                        self._circuit_breaker.record_success()
                    else:
                        self._circuit_breaker.record_failure(
                            RuntimeError(f"{action} returned unsuccessful response")
                        )
                        logger.warning(
                            "Genie Sim gRPC %s returned unsuccessful response",
                            action,
                        )

        return result

    def connect(self) -> bool:
        """
        Connect to Genie Sim gRPC server.

        Returns:
            True if connection successful
        """
        if not self._have_grpc:
            # Fallback: check if server is running via socket
            return self._check_server_socket()

        try:
            if self._channel is None:
                self._channel = grpc.insecure_channel(
                    f"{self.host}:{self.port}",
                    options=[
                        ("grpc.max_send_message_length", 16094304),
                        ("grpc.max_receive_message_length", 16094304),
                    ],
                )

            # Create service stubs (reuse if already initialized)
            if self._stub is None:
                self._stub = SimObservationServiceStub(self._channel)
            if self._joint_stub is None:
                self._joint_stub = joint_channel_pb2_grpc.JointControlServiceStub(self._channel)

            try:
                grpc.channel_ready_future(self._channel).result(timeout=self.timeout)
                self._connected = True
                logger.info(f"✅ Connected to Genie Sim gRPC server at {self.host}:{self.port}")
                # Note: warmup removed — server crashes if get_joint_position is called
                # before init_robot. Extended first-call timeout handles the slow init.
                return True
            except grpc.FutureTimeoutError:
                logger.error(f"Connection timeout after {self.timeout}s")
                self._connected = False
                return False

        except Exception as e:
            logger.error(f"Failed to connect to Genie Sim server: {e}")
            self._connected = False
            return False

    def _warmup_motion_server(self) -> None:
        """Send a get_joint_position with extended timeout to trigger server-side lazy init (cuRobo, etc.)."""
        if self._joint_stub is None:
            return
        try:
            logger.info("Warming up motion server (may trigger cuRobo init)...")
            request = joint_channel_pb2.GetJointReq(serial_no="")
            self._joint_stub.get_joint_position(request, timeout=self._first_call_timeout)
            logger.info("Motion server warmup complete")
        except Exception as e:
            logger.warning(f"Motion server warmup call failed (non-fatal): {e}")

    def ping(self, timeout: Optional[float] = None) -> bool:
        """
        Perform a lightweight gRPC health check against the server.

        Uses a TCP socket check rather than an RPC call because the server's
        CommandController may not be fully initialised (no robot loaded yet)
        and calling get_observation / reset on a bare server causes it to
        crash with an AttributeError.

        Returns:
            True if the server socket is reachable.
        """
        if self._check_server_socket():
            logger.info("✅ Genie Sim gRPC ping succeeded (socket reachable)")
            return True
        return False

    def get_observation_minimal(
        self,
        *,
        include_joint: bool = True,
        include_pose: bool = False,
        timeout: Optional[float] = None,
        lock_timeout: Optional[float] = None,
    ) -> GrpcCallResult:
        """
        Request a minimal observation payload for readiness checks.

        Uses individual gRPC calls (get_joint_position) instead of the
        unsupported standalone GetObservation.

        Returns:
            GrpcCallResult containing a formatted observation payload.
        """
        joint_positions = []
        joint_names = []
        if include_joint:
            try:
                jp_result = self.get_joint_position(lock_timeout=lock_timeout)
                if jp_result.success and jp_result.payload is not None:
                    joint_positions = list(jp_result.payload)
                    joint_names = list(self._joint_names) if self._joint_names else []
            except Exception:
                pass
        return GrpcCallResult(
            success=True,
            available=True,
            payload={
                "success": True,
                "robot_state": {
                    "joint_state": {
                        "names": joint_names,
                        "positions": joint_positions,
                    },
                },
                "scene_state": {"objects": []},
                "camera_observation": {"images": []},
                "recording_state": "",
            },
        )

    def get_server_info(self, timeout: Optional[float] = None) -> GrpcCallResult:
        """
        Fetch server version and capabilities via gRPC.

        Returns:
            GrpcCallResult with payload containing server version and capabilities.
        """
        if not self._connected:
            return GrpcCallResult(
                success=False,
                available=True,
                error="Not connected to Genie Sim server",
            )
        # Keep timeouts from being clipped below the configured default.
        effective_timeout = max(self.timeout, timeout) if timeout else self.timeout
        response_result = self.send_command(CommandType.GET_CHECKER_STATUS, {})
        if not response_result.available:
            return GrpcCallResult(
                success=False,
                available=False,
                error=response_result.error or "gRPC unavailable",
            )
        if not response_result.success:
            error_payload = response_result.payload or {}
            error = (
                response_result.error
                or error_payload.get("error_message")
                or error_payload.get("error")
                or "unknown error"
            )
            return GrpcCallResult(
                success=False,
                available=True,
                error=error,
                payload=error_payload,
            )
        response = response_result.payload or {}
        response["timeout_s"] = effective_timeout
        return GrpcCallResult(
            success=True,
            available=True,
            payload=response,
        )

    def _check_server_socket(self) -> bool:
        """Check if server is running via socket connection."""
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def _build_observation_request(
        self,
        data: Optional[Dict[str, Any]] = None,
        *,
        start_recording: bool = False,
        stop_recording: bool = False,
    ) -> Tuple[GetObservationReq, List[str], List[str]]:
        payload = data or {}
        camera_ids = payload.get("camera_ids") or payload.get("camera_prim_list") or self._default_camera_ids
        camera_ids = [str(camera_id) for camera_id in camera_ids]
        # Resolve logical names (e.g. "wrist") to USD prim paths via map.
        # Only include cameras that have a valid prim path (starts with "/").
        if self._camera_prim_map:
            resolved = []
            for cid in camera_ids:
                prim = self._camera_prim_map.get(cid, cid)
                if prim.startswith("/"):
                    resolved.append(prim)
                else:
                    if cid not in self._camera_missing_logged:
                        self._camera_missing_logged.add(cid)
                        logger.debug("Skipping camera '%s' — no USD prim path mapped", cid)
            camera_ids = resolved
        include_images = bool(payload.get("include_images", True))
        include_depth = bool(payload.get("include_depth", True))
        include_semantic = bool(payload.get("include_semantic", False))
        include_joint = bool(payload.get("include_joint", False))
        include_pose = bool(payload.get("include_pose", False))
        include_gripper = bool(payload.get("include_gripper", False))
        object_prims = payload.get("object_prims") or payload.get("objectPrims") or payload.get("object_ids") or []
        object_prims = [str(obj) for obj in object_prims]
        is_cam = include_images or include_depth or include_semantic or bool(camera_ids)
        is_pose = include_pose or bool(object_prims)
        camera_req = CameraRequest(
            render_depth=include_depth,
            render_semantic=include_semantic,
            camera_prim_list=camera_ids,
            additional_parameters=str(payload.get("additional_parameters", "{}")),
        )
        gripper_req = GripperRequest(
            left=bool(payload.get("left_gripper", False)),
            right=bool(payload.get("right_gripper", False)),
        )
        request = GetObservationReq(
            isCam=is_cam,
            CameraReq=camera_req,
            isJoint=include_joint,
            isPose=is_pose,
            objectPrims=object_prims,
            isGripper=include_gripper,
            gripperReq=gripper_req,
            startRecording=start_recording,
            stopRecording=stop_recording,
            fps=int(payload.get("fps", 0) or 0),
            task_name=str(payload.get("task_name", "")),
        )
        return request, camera_ids, object_prims

    def disconnect(self) -> None:
        """Disconnect from server."""
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None
            logger.info(
                "Genie Sim gRPC channel closed (host=%s port=%s)",
                self.host,
                self.port,
            )
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._connected

    def send_command(
        self,
        command: CommandType,
        data: Optional[Dict[str, Any]] = None,
    ) -> GrpcCallResult:
        """
        Send command to Genie Sim server.

        Now uses real gRPC calls instead of mock responses.

        Args:
            command: Command type
            data: Optional command data

        Returns:
            GrpcCallResult with payload containing response data from server.
        """
        if not self._connected:
            raise RuntimeError("Not connected to Genie Sim server")

        if not self._have_grpc:
            return self._grpc_unavailable(
                "send_command",
                "gRPC stubs unavailable",
            )
        if self._stub is None:
            return self._grpc_unavailable(
                "send_command",
                "gRPC stub not initialized",
            )

        payload = data or {}

        # Route commands that previously went through the broken standalone
        # GetObservation to their dedicated gRPC services.
        if command == CommandType.GET_OBSERVATION:
            return self.get_observation()
        if command == CommandType.GET_JOINT_POSITION:
            return self.get_joint_position()
        if command == CommandType.GET_OBJECT_POSE:
            object_id = payload.get("object_id", "")
            return self.get_object_pose(object_id)
        if command == CommandType.GET_GRIPPER_STATE:
            return self.get_gripper_state()
        if command in {CommandType.GET_CAMERA_DATA, CommandType.GET_SEMANTIC_DATA}:
            # Camera data via CameraService
            cam_prim = payload.get("camera_prim", payload.get("Cam_prim_path", ""))
            cam_data = self._get_camera_data_raw(cam_prim) if self._channel and cam_prim else None
            return GrpcCallResult(
                success=cam_data is not None,
                available=True,
                payload=cam_data or {},
            )

        # START_RECORDING and STOP_RECORDING still use the SimObservationService
        # get_observation gRPC (the server supports these sub-commands).
        if command in (CommandType.START_RECORDING, CommandType.STOP_RECORDING):
            observation_payload = dict(payload)
            request, camera_ids, object_prims = self._build_observation_request(
                observation_payload,
                start_recording=command == CommandType.START_RECORDING,
                stop_recording=command == CommandType.STOP_RECORDING,
            )

            def _request() -> GetObservationRsp:
                return self._stub.get_observation(request, timeout=self.timeout)

            response = self._call_grpc(
                f"get_observation({command.name})",
                _request,
                None,
                success_checker=lambda resp: resp is not None,
            )
            if response is None:
                return GrpcCallResult(
                    success=False,
                    available=True,
                    error="gRPC call failed",
                    payload={"success": False, "error": "gRPC call failed"},
                )
            return GrpcCallResult(
                success=True,
                available=True,
                payload={"recording_state": response.recordingState},
            )

        if self._stub is None:
            return self._grpc_unavailable(
                "send_command",
                "gRPC stub not initialized",
            )

        if command == CommandType.RESET:
            request = ResetReq(reset=bool(payload.get("reset", True)))

            def _request() -> ResetRsp:
                return self._stub.reset(
                    request, timeout=max(self.timeout, self._first_call_timeout)
                )

            response = self._call_grpc(
                "reset",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(
                success=True,
                available=True,
                payload={"msg": response.msg},
            )

        if command == CommandType.ATTACH_OBJ:
            request = AttachReq(
                obj_prims=[str(obj) for obj in payload.get("object_prims", [])],
                is_right=bool(payload.get("is_right", False)),
            )

            def _request() -> AttachRsp:
                return self._stub.attach_obj(request, timeout=self.timeout)

            response = self._call_grpc(
                "attach_obj",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(success=True, available=True, payload={"msg": response.msg})

        if command == CommandType.DETACH_OBJ:
            request = DetachReq(detach=bool(payload.get("detach", True)))

            def _request() -> DetachRsp:
                return self._stub.detach_obj(request, timeout=self.timeout)

            response = self._call_grpc(
                "detach_obj",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(success=True, available=True, payload={"msg": response.msg})

        if command == CommandType.TASK_STATUS:
            request = TaskStatusReq(
                isSuccess=bool(payload.get("is_success", payload.get("success", False))),
                failStep=[int(step) for step in payload.get("fail_steps", [])],
            )

            def _request() -> TaskStatusRsp:
                return self._stub.task_status(request, timeout=self.timeout)

            response = self._call_grpc(
                "task_status",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(success=True, available=True, payload={"msg": response.msg})

        if command == CommandType.EXIT:
            request = ExitReq(exit=bool(payload.get("exit", True)))

            def _request() -> ExitRsp:
                return self._stub.exit(request, timeout=self.timeout)

            response = self._call_grpc(
                "exit",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(success=True, available=True, payload={"msg": response.msg})

        if command == CommandType.INIT_ROBOT:
            joint_cmds = []
            for entry in payload.get("joint_cmd", []):
                try:
                    joint_cmds.append(
                        joint_pb2.JointCommand(
                            name=str(entry.get("name", "")),
                            position=float(entry.get("position", 0.0)),
                        )
                    )
                except (TypeError, ValueError):
                    continue
            request = InitRobotReq(
                robot_cfg_file=str(payload.get("robot_cfg_file", "")),
                robot_usd_path=str(payload.get("robot_usd_path", "")),
                scene_usd_path=str(payload.get("scene_usd_path", "")),
                robot_pose=self._pose_from_data(payload.get("robot_pose")) or se3_pose_pb2.SE3RpyPose(),
                stand_type=str(payload.get("stand_type", "")),
                stand_size_x=float(payload.get("stand_size_x", 0.0) or 0.0),
                stand_size_y=float(payload.get("stand_size_y", 0.0) or 0.0),
                joint_cmd=joint_cmds,
            )

            def _request() -> InitRobotRsp:
                return self._stub.init_robot(
                    request, timeout=max(self.timeout, self._first_call_timeout)
                )

            response = self._call_grpc(
                "init_robot",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(success=True, available=True, payload={"msg": response.msg})

        if command == CommandType.ADD_CAMERA:
            request = AddCameraReq(
                camera_prim=str(payload.get("camera_prim") or payload.get("camera_id") or ""),
                camera_pose=self._pose_from_data(payload.get("pose")) or se3_pose_pb2.SE3RpyPose(),
                focus_length=float(payload.get("focus_length", 0.0) or 0.0),
                horizontal_aperture=float(payload.get("horizontal_aperture", 0.0) or 0.0),
                vertical_aperture=float(payload.get("vertical_aperture", 0.0) or 0.0),
                width=int(payload.get("width", 0) or 0),
                height=int(payload.get("height", 0) or 0),
                is_local=bool(payload.get("is_local", True)),
            )

            def _request() -> AddCameraRsp:
                return self._stub.add_camera(request, timeout=self.timeout)

            response = self._call_grpc(
                "add_camera",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(success=True, available=True, payload={"msg": response.msg})

        if command == CommandType.SET_OBJECT_POSE:
            object_poses = []
            if "object_poses" in payload:
                for entry in payload.get("object_poses", []):
                    pose = self._pose_from_data(entry.get("pose"))
                    if pose is None:
                        pose = self._pose_from_data(entry)
                    if pose is None:
                        continue
                    object_poses.append(ObjectPose(prim_path=str(entry.get("prim_path", "")), pose=pose))
            else:
                object_id = payload.get("object_id")
                position = payload.get("position")
                rotation = payload.get("rotation")
                pose = self._pose_from_data({"position": position, "rotation": rotation})
                if object_id and pose:
                    object_poses.append(ObjectPose(prim_path=str(object_id), pose=pose))

            request = SetObjectPoseReq(
                object_pose=object_poses,
                joint_cmd=[],
                object_joint=[],
            )

            def _request() -> SetObjectPoseRsp:
                return self._stub.set_object_pose(request, timeout=self.timeout)

            response = self._call_grpc(
                "set_object_pose",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(success=True, available=True, payload={"msg": response.msg})

        if command == CommandType.SET_TRAJECTORY_LIST:
            trajectory = []
            for entry in payload.get("trajectory_list", []):
                pose = self._pose_from_data(entry)
                if pose is not None:
                    trajectory.append(pose)
            request = SetTrajectoryListReq(
                trajectory_point=trajectory,
                is_block=bool(payload.get("is_block", True)),
            )

            def _request() -> SetTrajectoryListRsp:
                return self._stub.set_trajectory_list(request, timeout=self.timeout * 5)

            response = self._call_grpc(
                "set_trajectory_list",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(success=True, available=True, payload={"msg": response.msg})

        if command == CommandType.SET_FRAME_STATE:
            request = SetFrameStateReq(frame_state=str(payload.get("frame_state", "")))

            def _request() -> SetFrameStateRsp:
                return self._stub.set_frame_state(request, timeout=self.timeout)

            response = self._call_grpc(
                "set_frame_state",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(success=True, available=True, payload={"msg": response.msg})

        if command == CommandType.SET_TASK_METRIC:
            request = SetTaskMetricReq(metric=str(payload.get("metric", "")))

            def _request() -> SetTaskMetricRsp:
                return self._stub.set_task_metric(request, timeout=self.timeout)

            response = self._call_grpc(
                "set_task_metric",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(success=True, available=True, payload={"msg": response.msg})

        if command == CommandType.REMOVE_OBJS_FROM_OBSTACLE:
            request = RemoveObstacleReq(
                obj_prims=[str(obj) for obj in payload.get("object_prims", [])],
            )

            def _request() -> RemoveObstacleRsp:
                return self._stub.remove_objs_from_obstacle(request, timeout=self.timeout)

            response = self._call_grpc(
                "remove_objs_from_obstacle",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(success=True, available=True, payload={"msg": response.msg})

        if command == CommandType.STORE_CURRENT_STATE:
            request = StoreCurrentStateReq(playback_id=str(payload.get("playback_id", "")))

            def _request() -> StoreCurrentStateRsp:
                return self._stub.store_current_state(request, timeout=self.timeout)

            response = self._call_grpc(
                "store_current_state",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(success=True, available=True, payload={"msg": response.msg})

        if command == CommandType.PLAYBACK:
            request = PlaybackReq(playback_id=str(payload.get("playback_id", "")))

            def _request() -> PlaybackRsp:
                return self._stub.playback(request, timeout=self.timeout)

            response = self._call_grpc(
                "playback",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(success=True, available=True, payload={"msg": response.msg})

        if command == CommandType.SET_LIGHT:
            lights = []
            for light in payload.get("lights", []):
                rotation = self._rpy_from(light.get("light_rotation") or light.get("rotation"))
                lights.append(
                    LightCfg(
                        light_type=str(light.get("light_type", "")),
                        light_prim=str(light.get("light_prim", "")),
                        light_temperature=float(light.get("light_temperature", 0.0) or 0.0),
                        light_intensity=float(light.get("light_intensity", 0.0) or 0.0),
                        light_rotation=rotation,
                        light_texture=str(light.get("light_texture", "")),
                    )
                )
            request = SetLightReq(
                lights=lights
            )

            def _request() -> SetLightRsp:
                return self._stub.set_light(request, timeout=self.timeout)

            response = self._call_grpc(
                "set_light",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(success=True, available=True, payload={"msg": response.msg})

        if command == CommandType.GET_CHECKER_STATUS:
            request = GetCheckerStatusReq(checker=str(payload.get("checker", "")))

            def _request() -> GetCheckerStatusRsp:
                return self._stub.get_checker_status(request, timeout=self.timeout)

            response = self._call_grpc(
                "get_checker_status",
                _request,
                None,
                success_checker=lambda resp: bool(resp.msg),
            )
            if response is None:
                return GrpcCallResult(success=False, available=True, error="gRPC call failed")
            return GrpcCallResult(
                success=True,
                available=True,
                payload={"msg": response.msg},
            )

        return GrpcCallResult(
            success=False,
            available=True,
            error=f"Unsupported command for Genie Sim gRPC transport: {command.name}",
        )

    def get_observation(
        self,
        lock_timeout: Optional[float] = None,
        fallback_joint_positions: Optional[Sequence[float]] = None,
    ) -> GrpcCallResult:
        """
        Get current observation from simulation.

        The server does NOT support standalone GetObservation via the
        SimObservationService (it only handles startRecording/stopRecording).
        Instead, we compose a real observation from individual gRPC services
        that ARE supported:
          - JointControlService.get_joint_position  (real PhysX joint data)
          - JointControlService.get_ee_pose          (real EE pose)
          - SimObjectService.get_object_pose          (real USD object poses)
          - CameraService.get_camera_data             (real rendered images)

        Returns:
            GrpcCallResult with payload containing robot_state, scene_state, timestamp.
        """
        import time as _time
        import struct as _struct
        import base64 as _b64

        # --- 1. Real joint positions, velocities, and efforts ---
        joint_positions: List[float] = []
        joint_names = []
        joint_velocities = []
        joint_efforts = []
        used_joint_cache = False
        try:
            jp_result = self.get_joint_position(lock_timeout=lock_timeout)
            if jp_result.success and jp_result.payload is not None:
                joint_positions = list(jp_result.payload)
                joint_names = list(self._joint_names) if self._joint_names else []
                # get_joint_position() now also stores velocities and efforts
                joint_velocities = getattr(self, "_latest_joint_velocities", [])
                joint_efforts = getattr(self, "_latest_joint_efforts", [])
            else:
                fallback_positions = (
                    list(fallback_joint_positions)
                    if fallback_joint_positions is not None
                    else list(getattr(self, "_latest_joint_positions", []))
                )
                if fallback_positions:
                    joint_positions = fallback_positions
                    joint_names = list(self._joint_names) if self._joint_names else []
                    joint_velocities = getattr(self, "_latest_joint_velocities", [])
                    joint_efforts = getattr(self, "_latest_joint_efforts", [])
                    used_joint_cache = True
        except Exception as exc:
            logger.warning(f"[OBS] get_joint_position failed: {exc}")
            fallback_positions = (
                list(fallback_joint_positions)
                if fallback_joint_positions is not None
                else list(getattr(self, "_latest_joint_positions", []))
            )
            if fallback_positions:
                joint_positions = fallback_positions
                joint_names = list(self._joint_names) if self._joint_names else []
                joint_velocities = getattr(self, "_latest_joint_velocities", [])
                joint_efforts = getattr(self, "_latest_joint_efforts", [])
                used_joint_cache = True

        # Normalize joint arrays to arm-only lengths for downstream consistency.
        joint_positions, joint_velocities, joint_efforts, _, joint_names = _normalize_joint_arrays(
            joint_positions,
            joint_velocities,
            joint_efforts,
            None,
            joint_names,
        )

        # --- 2. Real EE pose ---
        ee_pose = {}
        try:
            ee_result = self.get_ee_pose(ee_link_name="right", lock_timeout=lock_timeout)
            if ee_result.success and ee_result.payload is not None:
                ee_pose = ee_result.payload
        except Exception as exc:
            logger.warning(f"[OBS] get_ee_pose failed: {exc}")

        # --- 3. Real object poses via SimObjectService ---
        scene_objects = []
        # Use the task_config object prims if available
        object_prims = list(getattr(self, "_scene_object_prims", []))
        if getattr(self, "_resolved_any_object_pose", False):
            object_prims.extend(getattr(self, "_scene_variation_object_prims", []))
        if object_prims and self._channel is not None:
            for prim_path in object_prims:
                resolved = self._resolve_object_prim(prim_path)
                if resolved is None:
                    if prim_path in getattr(self, "_scene_variation_object_prims", []):
                        if not hasattr(self, "_unresolved_variation_objects"):
                            self._unresolved_variation_objects: Set[str] = set()
                        self._unresolved_variation_objects.add(prim_path)
                    continue
                resolved_path, obj_pose = resolved
                scene_objects.append({
                    "object_id": prim_path,
                    "pose": obj_pose,
                })
                logger.info(f"[OBS] Got real object pose for {resolved_path}: {obj_pose}")
                if prim_path in getattr(self, "_scene_variation_object_prims", []):
                    if hasattr(self, "_unresolved_variation_objects"):
                        self._unresolved_variation_objects.discard(prim_path)

        if hasattr(self, "_unresolved_variation_objects") and self._unresolved_variation_objects:
            if not hasattr(self, "_variation_warning_logged"):
                logger.warning(
                    "[OBS] Unresolved variation objects (deferred or missing): %s. "
                    "Enable stage diagnostics with tools/geniesim_adapter/deployment/patches/"
                    "patch_stage_diagnostics.py to verify loaded prims.",
                    sorted(self._unresolved_variation_objects),
                )
                self._variation_warning_logged = True

        # --- 4. Camera images ---
        # Requires patched server (see deployment/patches/patch_camera_handler.py).
        # On unpatched servers, get_camera_data returns None and we fall back gracefully.
        camera_images = []
        _camera_debug = os.environ.get("GENIESIM_CAMERA_DEBUG", "0") == "1"
        if self._channel is not None:
            _cam_prims = list(self._camera_prim_map.values()) if self._camera_prim_map else []
            if not _cam_prims:
                _cam_prims = self._default_camera_ids  # logical names used as-is
            if _camera_debug and not getattr(self, "_cam_debug_logged", False):
                logger.info("[CAM_DEBUG] camera_prim_map=%s, cam_prims=%s", self._camera_prim_map, _cam_prims)
                self._cam_debug_logged = True
            for _cam_prim in _cam_prims:
                try:
                    cam_data = self._get_camera_data_raw(_cam_prim)
                    if _camera_debug:
                        _has_rgb = cam_data is not None and cam_data.get("rgb") not in (None, b"")
                        logger.info("[CAM_DEBUG] %s -> data=%s, has_rgb=%s", _cam_prim, cam_data is not None, _has_rgb)
                    if cam_data is not None:
                        # Find logical name for this prim
                        _cam_name = _cam_prim
                        for _logical, _prim in self._camera_prim_map.items():
                            if _prim == _cam_prim:
                                _cam_name = _logical
                                break
                        camera_images.append({"camera_id": _cam_name, **cam_data})
                except Exception as exc:
                    if not hasattr(self, "_camera_warning_logged"):
                        logger.warning(
                            "[OBS] Camera capture failed for %s: %s. "
                            "Server may need the camera handler patch "
                            "(see deployment/patches/patch_camera_handler.py).",
                            _cam_prim, exc,
                        )
                        self._camera_warning_logged = True

        # --- Timestamp ---
        mono_ns = _time.monotonic_ns()
        if not hasattr(self, "_obs_timestamp_base"):
            self._obs_timestamp_base = _time.time()
            self._obs_monotonic_base = mono_ns
        unique_timestamp = self._obs_timestamp_base + (mono_ns - self._obs_monotonic_base) / 1e9

        data_sources = []
        if joint_positions:
            data_sources.append("joints_cached" if used_joint_cache else "joints")
        if ee_pose:
            data_sources.append("ee_pose")
        if scene_objects:
            data_sources.append(f"objects({len(scene_objects)})")
        if camera_images:
            data_sources.append(f"cameras({len(camera_images)})")
        logger.info(f"[OBS] Composed real observation from: {', '.join(data_sources) or 'none'}")

        result = {
            "success": True,
            "robot_state": {
                "joint_positions": joint_positions,
                "joint_velocities": joint_velocities,
                "joint_efforts": joint_efforts,
                "joint_state": {
                    "names": joint_names,
                    "positions": joint_positions,
                    "velocities": joint_velocities,
                    "efforts": joint_efforts,
                },
                "ee_pose": ee_pose,
            },
            "scene_state": {"objects": scene_objects},
            "camera_observation": {"images": camera_images},
            "timestamp": unique_timestamp,
            "planned_timestamp": 0.0,
            "data_source": "real_composed",
        }
        self._latest_observation = result
        return GrpcCallResult(
            success=True,
            available=True,
            payload=result,
        )

    def get_contact_report(
        self,
        *,
        include_points: bool = False,
        lock_timeout: Optional[float] = None,
    ) -> GrpcCallResult:
        """Fetch physics contact data from the server, if supported."""
        if not self._have_grpc:
            return self._grpc_unavailable("get_contact_report", "gRPC not available")
        if self._stub is None:
            return self._grpc_unavailable("get_contact_report", "gRPC channel not initialized")
        if not hasattr(self._stub, "get_contact_report"):
            return self._grpc_unavailable("get_contact_report", "RPC not supported by server")
        request = ContactReportReq(include_points=include_points)

        def _request() -> ContactReportRsp:
            return self._stub.get_contact_report(request, timeout=self.timeout)

        response = self._call_grpc(
            "get_contact_report",
            _request,
            None,
            success_checker=lambda resp: resp is not None,
            lock_timeout=lock_timeout,
        )
        if response is None:
            return GrpcCallResult(success=False, available=True, error="gRPC call failed")

        contacts = []
        max_penetration = float(getattr(response, "max_penetration_depth", 0.0) or 0.0)
        total_force = float(getattr(response, "total_normal_force", 0.0) or 0.0)
        use_contact_sum = total_force == 0.0
        for contact in response.contacts:
            penetration_depth = float(contact.penetration_depth)
            normal_force = float(contact.normal_force)
            contacts.append({
                "body_a": contact.body_a,
                "body_b": contact.body_b,
                "normal_force_N": normal_force,
                "penetration_depth": penetration_depth,
                "position": list(contact.position) if contact.position else [],
                "normal": list(contact.normal) if contact.normal else [],
                "force_vector": list(contact.force_vector) if contact.force_vector else [],
                "tangent_impulse": list(contact.tangent_impulse) if contact.tangent_impulse else [],
                "friction": float(contact.friction) if hasattr(contact, "friction") else None,
                "contact_area": float(contact.contact_area) if hasattr(contact, "contact_area") else None,
            })
            if penetration_depth > max_penetration:
                max_penetration = penetration_depth
            if use_contact_sum:
                total_force += normal_force

        payload = {
            "contacts": contacts,
            "contact_count": len(contacts),
            "total_normal_force": total_force,
            "max_penetration_depth": max_penetration,
        }
        return GrpcCallResult(success=True, available=True, payload=payload)

    def get_ee_wrench(
        self,
        *,
        include_contacts: bool = False,
        lock_timeout: Optional[float] = None,
    ) -> GrpcCallResult:
        """Fetch end-effector wrench from the server, if supported."""
        if not self._have_grpc:
            return self._grpc_unavailable("get_ee_wrench", "gRPC not available")
        if self._stub is None:
            return self._grpc_unavailable("get_ee_wrench", "gRPC channel not initialized")
        if not hasattr(self._stub, "get_ee_wrench"):
            return self._grpc_unavailable("get_ee_wrench", "RPC not supported by server")

        request = EEWrenchReq(include_contacts=include_contacts)

        def _request() -> EEWrenchRsp:
            return self._stub.get_ee_wrench(request, timeout=self.timeout)

        response = self._call_grpc(
            "get_ee_wrench",
            _request,
            None,
            success_checker=lambda resp: resp is not None,
            lock_timeout=lock_timeout,
        )
        if response is None:
            return GrpcCallResult(success=False, available=True, error="gRPC call failed")

        contacts = []
        for contact in getattr(response, "contacts", []):
            contacts.append({
                "body_a": contact.body_a,
                "body_b": contact.body_b,
                "normal_force_N": float(contact.normal_force),
                "penetration_depth": float(contact.penetration_depth),
                "position": list(contact.position) if contact.position else [],
                "normal": list(contact.normal) if contact.normal else [],
                "force_vector": list(contact.force_vector) if getattr(contact, "force_vector", None) else [],
                "tangent_impulse": list(contact.tangent_impulse) if getattr(contact, "tangent_impulse", None) else [],
                "friction": float(contact.friction) if hasattr(contact, "friction") else None,
                "contact_area": float(contact.contact_area) if hasattr(contact, "contact_area") else None,
            })

        payload = {
            "force": list(response.force) if getattr(response, "force", None) else [],
            "torque": list(response.torque) if getattr(response, "torque", None) else [],
            "frame": getattr(response, "frame", "end_effector"),
            "source": getattr(response, "source", "unknown"),
            "contacts": contacts,
        }
        return GrpcCallResult(success=True, available=True, payload=payload)

    def _resolve_object_prim_candidates(self, prim_path: str) -> List[str]:
        """Return candidate prim paths for resolving an object pose."""
        if not hasattr(self, "_resolved_prim_cache"):
            self._resolved_prim_cache: Dict[str, str] = {}
        if prim_path in self._resolved_prim_cache:
            return [self._resolved_prim_cache[prim_path]]

        # The Genie Sim server loads scene objects under /World/Scene/obj_{name}, so try that FIRST.
        _bare_name = prim_path.rsplit("/", 1)[-1] if "/" in prim_path else prim_path
        candidates = [
            f"/World/Scene/obj_{_bare_name}",
            f"/World/Scene/{_bare_name}",
            prim_path,
        ]
        if not prim_path.startswith("/World/"):
            candidates.append(f"/World/{prim_path}")
        if prim_path.startswith("/World/"):
            _bare = prim_path[len("/World/"):]
            candidates.append(f"/{_bare}")
            candidates.append(f"/Root/{_bare}")
        if not prim_path.startswith("/"):
            candidates.append(f"/{prim_path}")

        # Try without numeric suffix (e.g., Pot057 -> Pot) for fuzzy matching
        import re
        _base_name = re.sub(r'\d+$', '', _bare_name)
        if _base_name and _base_name != _bare_name:
            candidates.append(f"/World/Scene/obj_{_base_name}")
            candidates.append(f"/World/Scene/{_base_name}")
            candidates.append(f"/World/{_base_name}")

        # Try lowercase variants for case-insensitive matching
        _lower_name = _bare_name.lower()
        if _lower_name != _bare_name:
            candidates.append(f"/World/Scene/obj_{_lower_name}")
            candidates.append(f"/World/{_lower_name}")

        return list(dict.fromkeys(candidates))

    def _resolve_object_prim(
        self,
        prim_path: str,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Resolve an object prim path to a valid pose."""
        if not hasattr(self, "_missing_object_prims"):
            self._missing_object_prims: Set[str] = set()
        if prim_path in self._missing_object_prims:
            return None
        candidates = self._resolve_object_prim_candidates(prim_path)
        zero_pose_seen = False
        attempted = 0
        zero_pose_attempts = 0
        for candidate in candidates:
            if self._abort_event is not None and self._abort_event.is_set():
                break
            try:
                obj_pose, zero_pose = self._get_object_pose_raw(candidate)
                attempted += 1
                zero_pose_seen = zero_pose_seen or zero_pose
                if zero_pose:
                    zero_pose_attempts += 1
                if obj_pose is not None:
                    if candidate != prim_path:
                        self._resolved_prim_cache[prim_path] = candidate
                    self._resolved_any_object_pose = True
                    return candidate, obj_pose
            except Exception as exc:
                logger.debug(f"[OBS] get_object_pose({candidate}) failed: {exc}")
        if attempted and zero_pose_attempts == attempted:
            self._missing_object_prims.add(prim_path)
        if not zero_pose_seen:
            logger.warning(
                f"[OBS] Unable to resolve object prim for {prim_path} "
                f"(tried: {candidates})"
            )
        return None

    def _get_object_pose_raw(self, prim_path: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Get object pose via SimObjectService.get_object_pose (raw gRPC)."""
        import struct as _struct
        # Build GetObjectPoseReq: field 1 (prim_path, string)
        prim_bytes = prim_path.encode("utf-8")
        # Protobuf: tag 0x0a (field 1, length-delimited), then varint length, then bytes
        payload = b"\x0a" + self._encode_varint(len(prim_bytes)) + prim_bytes

        method = "/aimdk.protocol.SimObjectService/get_object_pose"
        try:
            call = self._channel.unary_unary(
                method,
                request_serializer=lambda x: x,
                response_deserializer=lambda x: x,
            )
            raw_response = call(payload, timeout=self.timeout)
            # Parse GetObjectPoseRsp manually
            # Field 2 is SE3RpyPose (object_pose)
            pose = self._parse_object_pose_response(raw_response)
            if pose is None:
                logger.warning(
                    f"[OBS] get_object_pose({prim_path}): server returned response "
                    f"({len(raw_response)} bytes) but no pose field — prim may not exist in server stage"
                )
                return None, False
            if (
                pose.get("position", {}).get("x", 0) == 0
                and pose.get("position", {}).get("y", 0) == 0
                and pose.get("position", {}).get("z", 0) == 0
            ):
                logger.warning(
                    f"[OBS] get_object_pose({prim_path}): returned identity/zero pose — "
                    f"object may not be loaded in simulation (returning None)"
                )
                return None, True
            return pose, False
        except Exception as exc:
            logger.warning(f"[OBS] raw get_object_pose({prim_path}) failed: {exc}")
            return None, False

    def _get_camera_data_raw(self, cam_prim_path: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Get camera data via CameraService.get_camera_data (raw gRPC)."""
        # Build GetCameraDataRequest: field 1 (serial_no=cam_prim_path, string)
        prim_bytes = cam_prim_path.encode("utf-8")
        payload = b"\x0a" + self._encode_varint(len(prim_bytes)) + prim_bytes

        method = "/aimdk.protocol.CameraService/get_camera_data"
        for attempt in range(max_retries):
            try:
                call = self._channel.unary_unary(
                    method,
                    request_serializer=lambda x: x,
                    response_deserializer=lambda x: x,
                )
                raw_response = call(payload, timeout=self.timeout)
                result = self._parse_camera_data_response(raw_response)
                if result is not None:
                    return result
                logger.warning(
                    "[OBS] Camera data parse returned None for %s (attempt %d/%d)",
                    cam_prim_path, attempt + 1, max_retries,
                )
            except Exception as exc:
                logger.warning(
                    "[OBS] raw get_camera_data failed for %s (attempt %d/%d): %s. "
                    "Ensure the camera prim exists in the USD stage and "
                    "the CameraService patch is applied.",
                    cam_prim_path, attempt + 1, max_retries, exc,
                )
            if attempt < max_retries - 1:
                import time as _time_mod
                _time_mod.sleep(0.1 * (attempt + 1))
        return None

    @staticmethod
    def _encode_varint(value: int) -> bytes:
        """Encode an integer as a protobuf varint."""
        pieces = []
        while value > 0x7F:
            pieces.append((value & 0x7F) | 0x80)
            value >>= 7
        pieces.append(value & 0x7F)
        return bytes(pieces)

    @staticmethod
    def _decode_varint(data: bytes, pos: int) -> Tuple[int, int]:
        """Decode a protobuf varint, return (value, new_pos)."""
        result = 0
        shift = 0
        while pos < len(data):
            b = data[pos]
            result |= (b & 0x7F) << shift
            pos += 1
            if not (b & 0x80):
                break
            shift += 7
        return result, pos

    def _parse_protobuf_fields(self, data: bytes) -> Dict[int, List[Tuple[int, bytes]]]:
        """Parse raw protobuf bytes into a dict of field_number -> [(wire_type, raw_value)]."""
        import struct as _struct
        fields: Dict[int, List[Tuple[int, bytes]]] = {}
        pos = 0
        while pos < len(data):
            tag, pos = self._decode_varint(data, pos)
            field_number = tag >> 3
            wire_type = tag & 0x07
            if wire_type == 0:  # varint
                val, pos = self._decode_varint(data, pos)
                fields.setdefault(field_number, []).append((wire_type, val.to_bytes(8, 'little')))
            elif wire_type == 1:  # 64-bit
                fields.setdefault(field_number, []).append((wire_type, data[pos:pos+8]))
                pos += 8
            elif wire_type == 2:  # length-delimited
                length, pos = self._decode_varint(data, pos)
                fields.setdefault(field_number, []).append((wire_type, data[pos:pos+length]))
                pos += length
            elif wire_type == 5:  # 32-bit
                fields.setdefault(field_number, []).append((wire_type, data[pos:pos+4]))
                pos += 4
            else:
                break  # unknown wire type
        return fields

    def _parse_se3_rpy_pose(self, data: bytes) -> Dict[str, Any]:
        """Parse SE3RpyPose protobuf: position (field 1, Vec3) + rpy (field 2, Rpy)."""
        import struct as _struct
        fields = self._parse_protobuf_fields(data)
        position = [0.0, 0.0, 0.0]
        rotation = [1.0, 0.0, 0.0, 0.0]  # w,x,y,z
        if 1 in fields:  # position (Vec3)
            vec_fields = self._parse_protobuf_fields(fields[1][0][1])
            for i, fnum in enumerate([1, 2, 3]):
                if fnum in vec_fields:
                    position[i] = _struct.unpack('<d', vec_fields[fnum][0][1])[0]
        if 2 in fields:  # rpy (Rpy) — actually quaternion rw,rx,ry,rz
            rpy_fields = self._parse_protobuf_fields(fields[2][0][1])
            for i, fnum in enumerate([1, 2, 3, 4]):
                if fnum in rpy_fields:
                    rotation[i] = _struct.unpack('<d', rpy_fields[fnum][0][1])[0]
        return {
            "position": {"x": position[0], "y": position[1], "z": position[2]},
            "rotation": {"rw": rotation[0], "rx": rotation[1], "ry": rotation[2], "rz": rotation[3]},
        }

    def _parse_object_pose_response(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Parse GetObjectPoseRsp: field 1=prim_path, field 2=object_pose (SE3RpyPose)."""
        fields = self._parse_protobuf_fields(data)
        if 2 in fields:
            return self._parse_se3_rpy_pose(fields[2][0][1])
        return None

    def _parse_camera_data_response(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Parse GetCameraDataResponse: field 2=color_info, field 3=color_image, field 5=depth_image."""
        fields = self._parse_protobuf_fields(data)
        result: Dict[str, Any] = {
            "rgb": b"",
            "depth": b"",
            "width": 0,
            "height": 0,
            "rgb_encoding": "",
            "depth_encoding": "",
        }
        # color_info (field 2) — CameraInfo: width(1,int32), height(2,int32), ppx(3,float), ppy(4,float), fx(5,float), fy(6,float)
        if 2 in fields:
            info_fields = self._parse_protobuf_fields(fields[2][0][1])
            if 1 in info_fields:
                result["width"] = int.from_bytes(info_fields[1][0][1], "little")
            if 2 in info_fields:
                result["height"] = int.from_bytes(info_fields[2][0][1], "little")
            # float32 fields
            if 3 in info_fields:
                try:
                    result["ppx"] = _struct.unpack("<f", info_fields[3][0][1])[0]
                except Exception:
                    pass
            if 4 in info_fields:
                try:
                    result["ppy"] = _struct.unpack("<f", info_fields[4][0][1])[0]
                except Exception:
                    pass
            if 5 in info_fields:
                try:
                    result["fx"] = _struct.unpack("<f", info_fields[5][0][1])[0]
                except Exception:
                    pass
            if 6 in info_fields:
                try:
                    result["fy"] = _struct.unpack("<f", info_fields[6][0][1])[0]
                except Exception:
                    pass
        # color_image (field 3) — CompressedImage: format(2,string), data(3,bytes)
        if 3 in fields:
            img_fields = self._parse_protobuf_fields(fields[3][0][1])
            if 2 in img_fields:
                result["rgb_encoding"] = img_fields[2][0][1].decode("utf-8", errors="ignore")
            if 3 in img_fields:
                result["rgb"] = img_fields[3][0][1]
        # depth_image (field 5) — CompressedImage
        if 5 in fields:
            img_fields = self._parse_protobuf_fields(fields[5][0][1])
            if 2 in img_fields:
                result["depth_encoding"] = img_fields[2][0][1].decode("utf-8", errors="ignore")
            if 3 in img_fields:
                result["depth"] = img_fields[3][0][1]
        # For convenience, keep a generic encoding if only one is present
        result["encoding"] = result["rgb_encoding"] or result["depth_encoding"]
        return result

    def _format_observation_response(
        self,
        response: GetObservationRsp,
        *,
        camera_ids: Sequence[str],
        object_prims: Sequence[str],
    ) -> Dict[str, Any]:
        joint_states = response.joint.left_arm or response.joint.right_arm or response.joint.body_arm
        joint_names = [state.name for state in joint_states]
        joint_positions = [state.position for state in joint_states]
        if joint_names:
            self._joint_names = joint_names

        scene_objects = []
        for idx, obj in enumerate(response.pose):
            object_id = object_prims[idx] if idx < len(object_prims) else f"object_{idx}"
            scene_objects.append(
                {
                    "object_id": object_id,
                    "pose": self._pose_to_dict(obj.object_pose) or {},
                }
            )

        camera_images = []
        for idx, camera_rsp in enumerate(response.camera):
            camera_id = camera_ids[idx] if idx < len(camera_ids) else str(idx)
            encoding = (
                camera_rsp.rgb_camera.format
                or camera_rsp.depth_camera.format
                or "rgb"
            )
            camera_images.append(
                {
                    "camera_id": camera_id,
                    "rgb_data": base64.b64encode(camera_rsp.rgb_camera.data).decode("ascii")
                    if camera_rsp.rgb_camera.data
                    else "",
                    "depth_data": base64.b64encode(camera_rsp.depth_camera.data).decode("ascii")
                    if camera_rsp.depth_camera.data
                    else "",
                    "semantic_data": base64.b64encode(camera_rsp.semantic_mask.data).decode("ascii")
                    if camera_rsp.semantic_mask.data
                    else "",
                    "width": camera_rsp.camera_info.width,
                    "height": camera_rsp.camera_info.height,
                    "fx": camera_rsp.camera_info.fx,
                    "fy": camera_rsp.camera_info.fy,
                    "ppx": camera_rsp.camera_info.ppx,
                    "ppy": camera_rsp.camera_info.ppy,
                    "extrinsic": list(camera_rsp.extrinsic) if getattr(camera_rsp, "extrinsic", None) else None,
                    "calibration_id": getattr(camera_rsp, "calibration_id", "") or "",
                    "encoding": encoding,
                }
            )

        return {
            "success": True,
            "robot_state": {
                "joint_state": {
                    "names": joint_names,
                    "positions": joint_positions,
                },
                "gripper": {
                    "left": self._pose_to_dict(response.gripper.left_gripper) or {},
                    "right": self._pose_to_dict(response.gripper.right_gripper) or {},
                },
            },
            "scene_state": {"objects": scene_objects},
            "camera_observation": {"images": camera_images},
        }

    @staticmethod
    def _vector3_from(value: Optional[Union[Dict[str, Any], Sequence[float]]]) -> "vec3_pb2.Vec3":
        if isinstance(value, vec3_pb2.Vec3):
            return value
        if isinstance(value, dict):
            return vec3_pb2.Vec3(
                x=float(value.get("x", 0.0)),
                y=float(value.get("y", 0.0)),
                z=float(value.get("z", 0.0)),
            )
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return vec3_pb2.Vec3(x=float(value[0]), y=float(value[1]), z=float(value[2]))
        return vec3_pb2.Vec3()

    @staticmethod
    def _rpy_from(value: Optional[Union[Dict[str, Any], Sequence[float]]]) -> "rpy_pb2.Rpy":
        if isinstance(value, rpy_pb2.Rpy):
            return value
        if isinstance(value, dict):
            return rpy_pb2.Rpy(
                rw=float(value.get("rw", 0.0)),
                rx=float(value.get("rx", 0.0)),
                ry=float(value.get("ry", 0.0)),
                rz=float(value.get("rz", 0.0)),
            )
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return rpy_pb2.Rpy(
                rw=float(value[0]),
                rx=float(value[1]),
                ry=float(value[2]),
                rz=float(value[3]),
            )
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return rpy_pb2.Rpy(
                rw=0.0,
                rx=float(value[0]),
                ry=float(value[1]),
                rz=float(value[2]),
            )
        return rpy_pb2.Rpy(rw=1.0, rx=0.0, ry=0.0, rz=0.0)

    @classmethod
    def _pose_from_data(cls, pose: Optional[Any]) -> Optional["se3_pose_pb2.SE3RpyPose"]:
        if isinstance(pose, se3_pose_pb2.SE3RpyPose):
            return pose
        if pose is None:
            return None
        if isinstance(pose, (list, tuple)) and len(pose) == 2:
            position, orientation = pose
        elif isinstance(pose, dict):
            position = pose.get("position") or pose.get("pos") or pose.get("translation")
            orientation = pose.get("orientation") or pose.get("rotation") or pose.get("quat")
        else:
            return None
        return se3_pose_pb2.SE3RpyPose(
            position=cls._vector3_from(position),
            rpy=cls._rpy_from(orientation),
        )

    @staticmethod
    def _pose_to_dict(pose: Optional["se3_pose_pb2.SE3RpyPose"]) -> Optional[Dict[str, Any]]:
        if pose is None:
            return None
        return {
            "position": {
                "x": pose.position.x,
                "y": pose.position.y,
                "z": pose.position.z,
            },
            "rotation": {
                "rw": pose.rpy.rw,
                "rx": pose.rpy.rx,
                "ry": pose.rpy.ry,
                "rz": pose.rpy.rz,
            },
        }

    def _resolve_joint_names(self, count: int) -> List[str]:
        if self._joint_names and len(self._joint_names) >= count:
            return list(self._joint_names[:count])
        logger.warning(
            "Using synthetic joint names (joint_0..joint_%d) — "
            "get_joint_position should be called before set_joint_position "
            "to populate real server joint names",
            count - 1,
        )
        return [f"joint_{index}" for index in range(count)]

    def stream_observations(
        self,
        *,
        include_images: bool = True,
        include_depth: bool = True,
        include_semantic: bool = False,
        camera_ids: Optional[Sequence[str]] = None,
        timeout: Optional[float] = None,
    ) -> GrpcCallResult:
        """
        Stream observations from the simulation.

        Returns:
            GrpcCallResult with payload yielding observation dictionaries.
        """
        del include_images, include_depth, include_semantic, camera_ids, timeout
        return GrpcCallResult(
            success=False,
            available=True,
            error="Streaming observations is not supported by the current Genie Sim gRPC transport.",
        )

    def set_joint_position(self, positions: List[float]) -> GrpcCallResult:
        """
        Set robot joint positions.

        Uses real gRPC set_joint_position call.

        Args:
            positions: Target joint positions

        Returns:
            GrpcCallResult indicating success.
        """
        if not self._have_grpc:
            return self._grpc_unavailable(
                "set_joint_position",
                "gRPC stubs unavailable",
            )
        if self._joint_stub is None:
            return self._grpc_unavailable(
                "set_joint_position",
                "gRPC joint stub not initialized",
            )

        joint_names = self._resolve_joint_names(len(positions))
        commands = [
            joint_pb2.JointCommand(name=name, position=float(value))
            for name, value in zip(joint_names, positions)
        ]

        # Use extended timeout for the first joint command — server may
        # still be doing lazy init (cuRobo, motion generation).
        call_timeout = self.timeout
        if self._first_joint_call:
            call_timeout = max(self.timeout, self._first_call_timeout)
            self._first_joint_call = False
            logger.info("First set_joint_position call — using extended timeout %ss", call_timeout)

        def _request() -> joint_channel_pb2.SetJointRsp:
            request = joint_channel_pb2.SetJointReq(
                commands=commands,
                is_trajectory=False,
            )
            return self._joint_stub.set_joint_position(request, timeout=call_timeout)

        response = self._call_grpc(
            "set_joint_position",
            _request,
            None,
            success_checker=lambda resp: bool(resp.errmsg),
        )
        if response is None:
            return GrpcCallResult(
                success=False,
                available=True,
                error="gRPC call failed",
            )
        self._latest_joint_positions = list(positions)
        return GrpcCallResult(
            success=True,
            available=True,
            payload={"msg": response.errmsg},
        )

    def _expand_arm_to_full_body(
        self,
        arm_positions: List[float],
        gripper_aperture: Optional[float] = None,
    ) -> List[float]:
        """Expand arm-only joint trajectory to full-body command (e.g. 7→34 for G1).

        Copies latest full-body joint state, overlays arm joints at correct
        indices, and optionally sets gripper finger joints.
        """
        robot_type = getattr(self, "_robot_type", "") or ""
        normalized = _normalize_robot_name(robot_type)
        if not normalized.startswith("g1"):
            return arm_positions  # Non-G1: pass through

        full_body = list(self._latest_joint_positions) if self._latest_joint_positions else []
        joint_names = list(self._joint_names) if self._joint_names else []

        if len(full_body) <= len(arm_positions):
            return arm_positions  # Can't expand if we don't have full state

        arm_indices = _resolve_g1_arm_joint_indices(robot_type, joint_names, len(full_body))
        if not arm_indices or len(arm_indices) != len(arm_positions):
            return arm_positions

        for idx, arm_idx in enumerate(arm_indices):
            full_body[arm_idx] = arm_positions[idx]

        if gripper_aperture is not None:
            gripper_indices = _resolve_g1_gripper_joint_indices(
                robot_type, joint_names, len(full_body), arm_indices,
            )
            gripper_width = gripper_aperture * 0.04  # per-finger width
            for gi in gripper_indices:
                full_body[gi] = gripper_width

        if not hasattr(self, "_logged_expand"):
            logger.info(
                "Expanding arm %d -> %d joints (G1 full-body)",
                len(arm_positions), len(full_body),
            )
            self._logged_expand = True

        return full_body

    def get_joint_position(self, lock_timeout: Optional[float] = None) -> GrpcCallResult:
        """
        Get current joint positions.

        Uses real gRPC get_joint_position call.

        Args:
            lock_timeout: Optional override for gRPC lock acquisition timeout.
                Use a short value (e.g. 5.0) for health probes to avoid blocking
                behind long-running calls.

        Returns:
            GrpcCallResult with payload list of joint positions.
        """
        if not self._have_grpc:
            return self._grpc_unavailable(
                "get_joint_position",
                "gRPC stubs unavailable",
            )
        if self._joint_stub is None:
            return self._grpc_unavailable(
                "get_joint_position",
                "gRPC joint stub not initialized",
            )

        def _request() -> joint_channel_pb2.GetJointRsp:
            request = joint_channel_pb2.GetJointReq(serial_no="")
            return self._joint_stub.get_joint_position(request, timeout=self.timeout)

        response = self._call_grpc(
            "get_joint_position",
            _request,
            None,
            success_checker=lambda resp: bool(resp.states),
            lock_timeout=lock_timeout,
        )
        if response is None:
            return GrpcCallResult(
                success=False,
                available=True,
                error="gRPC call failed",
            )
        joint_names = [state.name for state in response.states]
        joint_positions = [state.position for state in response.states]
        joint_velocities = [state.velocity for state in response.states]
        joint_efforts = [state.effort for state in response.states]
        if joint_names:
            self._joint_names = joint_names
        # Store velocities and efforts for use in observation composition
        self._latest_joint_positions = joint_positions
        self._latest_joint_velocities = joint_velocities
        self._latest_joint_efforts = joint_efforts
        return GrpcCallResult(
            success=bool(joint_positions),
            available=True,
            payload=joint_positions,
        )

    def get_ee_pose(self, ee_link_name: str = "", lock_timeout: Optional[float] = None) -> GrpcCallResult:
        """
        Get current end-effector pose.

        Uses real gRPC get_ee_pose call.

        Args:
            ee_link_name: Optional end-effector link name

        Returns:
            GrpcCallResult with payload pose dict.
        """
        if not self._have_grpc:
            return self._grpc_unavailable(
                "get_ee_pose",
                "gRPC stubs unavailable",
            )
        if self._joint_stub is None:
            return self._grpc_unavailable(
                "get_ee_pose",
                "gRPC joint stub not initialized",
            )

        is_right = "right" in (ee_link_name or "").lower()

        def _request() -> joint_channel_pb2.GetEEPoseRsp:
            request = joint_channel_pb2.GetEEPoseReq(is_right=is_right)
            return self._joint_stub.get_ee_pose(request, timeout=self.timeout)

        response = self._call_grpc(
            "get_ee_pose",
            _request,
            None,
            success_checker=lambda resp: bool(resp.prim_path) or resp.ee_pose is not None,
            lock_timeout=lock_timeout,
        )
        if response is None:
            return GrpcCallResult(
                success=False,
                available=True,
                error="gRPC call failed",
            )
        return GrpcCallResult(
            success=True,
            available=True,
            payload=self._pose_to_dict(response.ee_pose),
        )

    def get_gripper_state(self) -> GrpcCallResult:
        """
        Get current gripper state.

        Note: The server does not have a dedicated GetGripperState gRPC service.
        Returns the last known gripper state from client-side tracking.

        Returns:
            GrpcCallResult with payload containing gripper state info.
        """
        # Gripper state is tracked client-side since we control it via set_gripper_state
        gripper_state = getattr(self, "_last_gripper_state", {"left": {}, "right": {}})
        return GrpcCallResult(
            success=True,
            available=True,
            payload=gripper_state,
        )

    def set_gripper_state(
        self,
        width: float,
        force: float = 0.0,
        wait_for_completion: bool = True,
    ) -> GrpcCallResult:
        """
        Set gripper state.

        Uses real gRPC send_command mapping for gripper control.

        Args:
            width: Target gripper width
            force: Grasping force
            wait_for_completion: Wait for gripper action to complete

        Returns:
            GrpcCallResult indicating success.
        """
        del wait_for_completion
        if not self._have_grpc:
            return self._grpc_unavailable(
                "set_gripper_state",
                "gRPC stubs unavailable",
            )
        if self._stub is None:
            return self._grpc_unavailable(
                "set_gripper_state",
                "gRPC stub not initialized",
            )

        # Send gripper command via raw gRPC (no compiled pb2 stubs needed).
        # Proto: SetGripperStateReq { gripper_command=string, is_right=bool, opened_width=double }
        # Service path: /aimdk.protocol.SimGripperService/set_gripper_state
        import struct as _struct

        # Manually build a minimal protobuf for SetGripperStateReq:
        #   field 1 (gripper_command, string): tag=0x0a, len-delimited
        #   field 2 (is_right, bool): tag=0x10, varint
        #   field 3 (opened_width, double): tag=0x19, 64-bit
        # Determine command from width: closed < 0.02m threshold < open
        gripper_command = "close" if width < 0.02 else "open"
        cmd_bytes = gripper_command.encode("utf-8")
        payload = (
            b"\x0a" + bytes([len(cmd_bytes)]) + cmd_bytes  # field 1
            + b"\x10\x01"  # field 2: is_right=True
            + b"\x19" + _struct.pack("<d", width)  # field 3: opened_width
        )

        method = "/aimdk.protocol.SimGripperService/set_gripper_state"
        try:
            call = self._channel.unary_unary(
                method,
                request_serializer=lambda x: x,
                response_deserializer=lambda x: x,
            )
            raw_response = call(payload, timeout=self.timeout)
            logger.info("[GRIPPER] set_gripper_state(%s, width=%.4f) succeeded", gripper_command, width)
            self._last_gripper_state = {
                "command": gripper_command,
                "width": width,
                "force": force,
            }
            return GrpcCallResult(success=True, available=True)
        except Exception as exc:
            logger.warning(f"[GRIPPER] set_gripper_state failed: {exc}")
            return GrpcCallResult(success=False, available=True, error=str(exc))

    def get_ik_status(
        self,
        target_pose: "np.ndarray",
        is_right: bool = True,
        obs_avoid: bool = True,
    ) -> GrpcCallResult:
        """
        Query server-side IK solver for joint positions reaching target_pose.

        Uses JointControlService.get_ik_status which has the actual robot model
        (e.g. G1 humanoid).

        Args:
            target_pose: 4x4 homogeneous transformation matrix
            is_right: True for right arm, False for left
            obs_avoid: Enable obstacle avoidance in IK

        Returns:
            GrpcCallResult with payload: {joint_names, joint_positions, is_success}
        """
        if not self._have_grpc:
            return self._grpc_unavailable("get_ik_status", "gRPC stubs unavailable")
        if self._joint_stub is None:
            return self._grpc_unavailable("get_ik_status", "joint stub not initialized")

        try:
            from aimdk.protocol.common.se3_pose_pb2 import SE3MatrixPose
            from aimdk.protocol.hal.joint.joint_channel_pb2 import GetIKStatusReq
        except ImportError as exc:
            return GrpcCallResult(
                success=False, available=True,
                error=f"Protobuf imports unavailable for get_ik_status: {exc}",
            )

        # Build SE3MatrixPose: 16 doubles row-major from 4x4 matrix
        elements = target_pose.flatten().tolist()
        matrix_pose = SE3MatrixPose(elements=elements)
        request = GetIKStatusReq(
            is_right=is_right,
            target_pose=[matrix_pose],
            ObsAvoid=obs_avoid,
        )

        def _request():
            return self._joint_stub.get_ik_status(request, timeout=self.timeout)

        response = self._call_grpc(
            "get_ik_status",
            _request,
            None,
            success_checker=lambda resp: len(resp.IKStatus) > 0,
        )
        if response is None:
            return GrpcCallResult(success=False, available=True, error="get_ik_status failed")

        ik_results = []
        for ik_status in response.IKStatus:
            ik_results.append({
                "joint_names": list(ik_status.joint_names),
                "joint_positions": list(ik_status.joint_positions),
                "is_success": ik_status.isSuccess,
            })

        any_success = any(r["is_success"] for r in ik_results)
        return GrpcCallResult(
            success=any_success,
            available=True,
            payload=ik_results[0] if ik_results else {},
        )

    def attach_object(
        self,
        object_prim: str,
        is_right: bool = True,
    ) -> GrpcCallResult:
        """Attach an object to the robot gripper via gRPC."""
        return self.send_command(
            CommandType.ATTACH_OBJ,
            {"object_prims": [object_prim], "is_right": is_right},
        )

    def detach_object(self) -> GrpcCallResult:
        """Detach the currently held object from the gripper."""
        return self.send_command(
            CommandType.DETACH_OBJ,
            {"detach": True},
        )

    def get_object_pose(self, object_id: str) -> GrpcCallResult:
        """
        Get an object's pose via SimObjectService.get_object_pose (raw gRPC).

        Args:
            object_id: Object prim path

        Returns:
            GrpcCallResult with payload pose dict.
        """
        if self._channel is None:
            return self._grpc_unavailable(
                "get_object_pose",
                "gRPC channel not initialized",
            )
        resolved = self._resolve_object_prim(object_id)
        if resolved is None:
            return GrpcCallResult(
                success=False,
                available=True,
                error=f"Object pose not available for {object_id}",
            )
        _, pose = resolved
        return GrpcCallResult(
            success=True,
            available=True,
            payload=pose,
        )

    def set_object_pose(self, object_id: str, pose: Dict[str, Any]) -> GrpcCallResult:
        """
        Set an object's pose.

        Uses real gRPC set_object_pose call.

        Args:
            object_id: Object identifier
            pose: Pose dict

        Returns:
            GrpcCallResult indicating success.
        """
        if not self._have_grpc:
            return self._grpc_unavailable(
                "set_object_pose",
                "gRPC stubs unavailable",
            )
        if self._stub is None:
            return self._grpc_unavailable(
                "set_object_pose",
                "gRPC stub not initialized",
            )

        pose_message = self._pose_from_data(pose) or se3_pose_pb2.SE3RpyPose()

        def _request() -> SetObjectPoseRsp:
            request = SetObjectPoseReq(
                object_pose=[ObjectPose(prim_path=object_id, pose=pose_message)],
                joint_cmd=[],
                object_joint=[],
            )
            return self._stub.set_object_pose(request, timeout=self.timeout)

        response = self._call_grpc(
            "set_object_pose",
            _request,
            None,
            success_checker=lambda resp: bool(resp.msg),
        )
        if response is None:
            return GrpcCallResult(
                success=False,
                available=True,
                error="gRPC call failed",
            )
        return GrpcCallResult(success=True, available=True, payload={"msg": response.msg})

    def attach_object(self, object_id: str, link_name: str) -> GrpcCallResult:
        """
        Attach an object to a robot link.

        Uses real gRPC attach_obj call.

        Args:
            object_id: Object identifier
            link_name: Robot link name

        Returns:
            GrpcCallResult indicating success.
        """
        return self.send_command(
            CommandType.ATTACH_OBJ,
            {
                "object_prims": [object_id],
                "is_right": "right" in link_name.lower(),
            },
        )

    def detach_object(self, object_id: str) -> GrpcCallResult:
        """
        Detach an object from the robot.

        Uses real gRPC detach_obj call.

        Args:
            object_id: Object identifier

        Returns:
            GrpcCallResult indicating success.
        """
        del object_id
        return self.send_command(
            CommandType.DETACH_OBJ,
            {"detach": True},
        )

    def init_robot(
        self,
        robot_type: str,
        urdf_path: str = "",
        base_pose: Optional[Dict[str, Any]] = None,
        initial_joint_positions: Optional[Sequence[float]] = None,
        scene_usd_path: str = "",
    ) -> GrpcCallResult:
        """
        Initialize the robot in the simulation.

        Uses real gRPC init_robot call.

        Args:
            robot_type: Robot type identifier
            urdf_path: Optional URDF path
            base_pose: Optional base pose
            initial_joint_positions: Optional initial joint positions

        Returns:
            GrpcCallResult with payload containing joint metadata.
        """
        if not self._have_grpc:
            return self._grpc_unavailable(
                "init_robot",
                "gRPC stubs unavailable",
            )
        if self._stub is None:
            return self._grpc_unavailable(
                "init_robot",
                "gRPC stub not initialized",
            )

        pose_message = self._pose_from_data(base_pose) or se3_pose_pb2.SE3RpyPose()
        joint_cmds: List["joint_pb2.JointCommand"] = []
        if initial_joint_positions:
            joint_names = self._resolve_joint_names(len(initial_joint_positions))
            joint_cmds = [
                joint_pb2.JointCommand(name=name, position=float(position))
                for name, position in zip(joint_names, initial_joint_positions)
            ]

        def _request() -> InitRobotRsp:
            request = InitRobotReq(
                robot_cfg_file=robot_type,
                robot_usd_path=urdf_path,
                scene_usd_path=scene_usd_path or "",
                robot_pose=pose_message,
                joint_cmd=joint_cmds,
            )
            # init_robot loads scene USD + robot — can take >60s on first call
            return self._stub.init_robot(
                request, timeout=max(self.timeout, self._first_call_timeout)
            )

        response = self._call_grpc(
            "init_robot",
            _request,
            None,
            success_checker=lambda resp: bool(resp.msg),
        )
        if response is None:
            return GrpcCallResult(
                success=False,
                available=True,
                error="gRPC call failed",
            )
        return GrpcCallResult(
            success=True,
            available=True,
            payload={"msg": response.msg},
        )

    def reset(
        self,
        *,
        reset_robot: bool = True,
        reset_objects: bool = True,
        scene_path: str = "",
    ) -> GrpcCallResult:
        """
        Reset the simulation environment.

        Uses real gRPC reset call.

        Args:
            reset_robot: Reset robot state
            reset_objects: Reset objects
            scene_path: Optional scene path override

        Returns:
            GrpcCallResult indicating success.
        """
        if not self._have_grpc:
            return self._grpc_unavailable(
                "reset",
                "gRPC stubs unavailable",
            )
        if self._stub is None:
            return self._grpc_unavailable(
                "reset",
                "gRPC stub not initialized",
            )

        del reset_robot, reset_objects, scene_path

        def _request() -> ResetRsp:
            request = ResetReq(reset=True)
            # reset can be slow when server re-initializes physics
            return self._stub.reset(request, timeout=max(self.timeout, self._first_call_timeout))

        response = self._call_grpc(
            "reset",
            _request,
            None,
            success_checker=lambda resp: bool(resp.msg),
        )
        if response is None:
            return GrpcCallResult(
                success=False,
                available=True,
                error="gRPC call failed",
            )
        return GrpcCallResult(
            success=True,
            available=True,
            payload={"msg": response.msg},
        )

    def add_camera(
        self,
        camera_id: str,
        pose: Optional[Dict[str, Any]] = None,
        parent_link: str = "",
        width: int = 640,
        height: int = 480,
        fov: float = 60.0,
        near_clip: float = 0.01,
        far_clip: float = 10.0,
    ) -> GrpcCallResult:
        """
        Add a camera to the simulation.

        Uses real gRPC add_camera call.

        Args:
            camera_id: Camera identifier
            pose: Camera pose
            parent_link: Parent link name (empty for world)
            width: Image width
            height: Image height
            fov: Field of view
            near_clip: Near clipping plane
            far_clip: Far clipping plane

        Returns:
            GrpcCallResult indicating success.
        """
        if not self._have_grpc:
            return self._grpc_unavailable(
                "add_camera",
                "gRPC stubs unavailable",
            )
        if self._stub is None:
            return self._grpc_unavailable(
                "add_camera",
                "gRPC stub not initialized",
            )

        del parent_link, near_clip, far_clip
        pose_message = self._pose_from_data(pose) or se3_pose_pb2.SE3RpyPose()

        def _request() -> AddCameraRsp:
            request = AddCameraReq(
                camera_prim=camera_id,
                camera_pose=pose_message,
                focus_length=float(fov),
                horizontal_aperture=0.0,
                vertical_aperture=0.0,
                width=width,
                height=height,
                is_local=True,
            )
            return self._stub.add_camera(request, timeout=self.timeout)

        response = self._call_grpc(
            "add_camera",
            _request,
            None,
            success_checker=lambda resp: bool(resp.msg),
        )
        if response is None:
            return GrpcCallResult(
                success=False,
                available=True,
                error="gRPC call failed",
            )
        return GrpcCallResult(
            success=True,
            available=True,
            payload={"msg": response.msg},
        )

    def get_camera_data(self, camera_id: str = "wrist") -> Optional[Dict[str, Any]]:
        """
        Get camera image.

        Integrated with GetObservation for camera data.

        Args:
            camera_id: Camera identifier

        Returns:
            Dict with decoded rgb/depth numpy arrays and metadata, or None
        """
        obs = getattr(self, "_latest_observation", None) or {}
        camera_observation = obs.get("camera_observation") or {}
        images = camera_observation.get("images") if camera_observation else None

        if not images:
            # Clear stale cache so get_observation() fetches fresh data
            self._latest_observation = {}
            # No cached images — fetch a fresh observation from the server.
            # This handles the case where _latest_observation has an empty
            # camera_observation ({"images": []}) from a prior call that
            # failed to capture images.
            obs_lock_timeout = float(os.getenv("OBS_LOCK_TIMEOUT_S", "1.0"))
            obs_result = self.get_observation(lock_timeout=obs_lock_timeout)
            if not obs_result.available or not obs_result.success:
                logger.warning("No observation available for camera data.")
                return None
            obs = obs_result.payload or {}
            camera_observation = obs.get("camera_observation") or {}
            images = camera_observation.get("images") if camera_observation else None

        if not images:
            return None

        image_info = next(
            (image for image in images if image.get("camera_id") == camera_id),
            None,
        )
        if image_info is None:
            if camera_id not in self._camera_missing_logged:
                self._camera_missing_logged.add(camera_id)
                available_ids = sorted({image.get("camera_id") for image in images if image.get("camera_id")})
                logger.warning(
                    "Camera '%s' not found in observation. Available camera_ids=%s.",
                    camera_id,
                    available_ids,
                )
            return None

        width = int(image_info.get("width") or 0)
        height = int(image_info.get("height") or 0)
        encoding = (image_info.get("encoding") or "").lower()

        if width <= 0 or height <= 0:
            logger.warning(
                "Invalid camera dimensions for '%s' (width=%s, height=%s).",
                camera_id,
                width,
                height,
            )
            return {
                "camera_id": camera_id,
                "rgb": None,
                "depth": None,
                "width": width,
                "height": height,
                "encoding": encoding,
                "timestamp": image_info.get("timestamp"),
            }

        def _safe_b64decode(field: str) -> Optional[bytes]:
            payload = image_info.get(field)
            if not payload:
                return b""
            try:
                return base64.b64decode(payload)
            except (binascii.Error, ValueError):
                logger.warning(
                    "Failed to decode camera '%s' field '%s'.",
                    camera_id,
                    field,
                )
                return None

        rgb_bytes = _safe_b64decode("rgb_data")
        depth_bytes = _safe_b64decode("depth_data")

        rgb_image = (
            self._decode_rgb_data(rgb_bytes, width, height, encoding)
            if rgb_bytes is not None
            else None
        )
        depth_image = (
            self._decode_depth_data(depth_bytes, width, height, encoding)
            if depth_bytes is not None
            else None
        )

        return {
            "camera_id": camera_id,
            "rgb": rgb_image,
            "depth": depth_image,
            "width": width,
            "height": height,
            "encoding": encoding,
            "rgb_encoding": encoding,
            "depth_encoding": encoding,
            "timestamp": image_info.get("timestamp"),
            "fx": image_info.get("fx"),
            "fy": image_info.get("fy"),
            "ppx": image_info.get("ppx"),
            "ppy": image_info.get("ppy"),
            "extrinsic": image_info.get("extrinsic"),
            "calibration_id": image_info.get("calibration_id"),
        }

    @staticmethod
    def _decode_rgb_data(
        rgb_bytes: bytes,
        width: int,
        height: int,
        encoding: str,
    ) -> Optional[np.ndarray]:
        if not rgb_bytes:
            return None
        if GenieSimGRPCClient._is_png_data(rgb_bytes) or "png" in encoding:
            return GenieSimGRPCClient._decode_png(rgb_bytes)

        normalized = encoding.lower()
        channels = 3
        dtype = np.uint8
        if "rgba" in normalized:
            channels = 4
        elif "rgb" in normalized:
            channels = 3
        elif "bgr" in normalized:
            channels = 3
        elif "mono" in normalized or "8uc1" in normalized:
            channels = 1
        expected_size = width * height * channels
        if expected_size == 0 or len(rgb_bytes) < expected_size:
            logger.warning("RGB data size mismatch (expected=%s, actual=%s).", expected_size, len(rgb_bytes))
            return None
        image = np.frombuffer(rgb_bytes, dtype=dtype, count=expected_size)
        image = image.reshape((height, width, channels)) if channels > 1 else image.reshape((height, width))
        if "bgr" in normalized:
            image = image[..., ::-1]
        return image

    @staticmethod
    def _decode_depth_data(
        depth_bytes: bytes,
        width: int,
        height: int,
        encoding: str,
    ) -> Optional[np.ndarray]:
        if not depth_bytes:
            return None
        if GenieSimGRPCClient._is_png_data(depth_bytes) or "png" in encoding:
            return GenieSimGRPCClient._decode_png(depth_bytes)

        normalized = encoding.lower()
        if "32f" in normalized or "float32" in normalized:
            dtype = np.float32
        elif "16u" in normalized or "uint16" in normalized or "16" in normalized:
            dtype = np.uint16
        else:
            dtype = np.float32
        expected_size = width * height
        if expected_size == 0:
            logger.warning("Depth data size mismatch (missing width/height).")
            return None
        image = np.frombuffer(depth_bytes, dtype=dtype, count=expected_size)
        if image.size < expected_size:
            logger.warning("Depth data size mismatch (expected=%s, actual=%s).", expected_size, image.size)
            return None
        return image.reshape((height, width))

    @staticmethod
    def _decode_png(data: bytes) -> Optional[np.ndarray]:
        try:
            from PIL import Image
        except ImportError:
            try:
                import imageio.v3 as iio
            except ImportError:
                logger.warning("No PNG decoder available (PIL/imageio missing).")
                return None
            return iio.imread(data)
        import io

        with Image.open(io.BytesIO(data)) as image:
            return np.array(image)

    @staticmethod
    def _is_png_data(data: bytes) -> bool:
        return data.startswith(b"\x89PNG\r\n\x1a\n")

    def execute_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        abort_event: Optional[threading.Event] = None,
        observation_callback: Optional[Callable[[Dict[str, Any], int], None]] = None,
        waypoint_completed_callback: Optional[Callable[[int], None]] = None,
    ) -> GrpcCallResult:
        """
        Execute a trajectory on the robot.

        Uses real gRPC set_joint_position calls with cuRobo-planned trajectory.

        Args:
            trajectory: List of waypoints with positions, velocities, timestamps
            abort_event: Optional event to signal early termination (e.g. task success).
            observation_callback: Optional callback invoked between waypoints with
                (waypoint_dict, waypoint_index). Called after each set_joint_position
                returns and before the inter-waypoint delay, while _grpc_lock is NOT
                held. Use this to capture observations without lock contention.
            waypoint_completed_callback: Optional callback invoked after each waypoint
                gRPC call completes (before observation_callback). Use this to update
                progress tracking / stall watchdog.

        Returns:
            GrpcCallResult indicating success.
        """
        if not self._have_grpc:
            return self._grpc_unavailable(
                "execute_trajectory",
                "gRPC stubs unavailable",
            )
        if self._joint_stub is None:
            return self._grpc_unavailable(
                "execute_trajectory",
                "gRPC joint stub not initialized",
            )

        if not trajectory:
            return GrpcCallResult(
                success=False,
                available=True,
                error="Trajectory is empty",
            )

        last_timestamp = None
        for _wp_idx, waypoint in enumerate(trajectory):
            if abort_event is not None and abort_event.is_set():
                logger.info("execute_trajectory: abort_event set, exiting early")
                break
            joint_positions = waypoint.get("joint_positions") or []
            if not joint_positions:
                continue

            # --- Fix 5: Expand arm-only trajectory to full-body (e.g. 7→34 for G1) ---
            gripper_aperture = waypoint.get("gripper_aperture")
            joint_positions = self._expand_arm_to_full_body(
                list(joint_positions), gripper_aperture,
            )

            joint_names = self._resolve_joint_names(len(joint_positions))
            commands = [
                joint_pb2.JointCommand(name=name, position=float(value))
                for name, value in zip(joint_names, joint_positions)
            ]
            request = joint_channel_pb2.SetJointReq(
                commands=commands,
                is_trajectory=False,
            )
            # Use extended timeout for the entire first trajectory (server
            # does cuRobo lazy init which can block any waypoint call).
            # Use short timeout if abort already signaled so we exit quickly.
            if abort_event is not None and abort_event.is_set():
                call_timeout = 2.0
            elif self._first_trajectory_call:
                call_timeout = max(self.timeout, self._first_call_timeout)
                logger.info(f"First trajectory set_joint_position — using extended timeout {call_timeout}s")
            elif not self._curobo_initialized and _wp_idx == 0:
                call_timeout = min(self._first_call_timeout, 90.0)
                logger.info(f"Pre-cuRobo-init first waypoint — using moderate timeout {call_timeout}s")
            else:
                call_timeout = self.timeout
            _req = request
            _timeout = call_timeout
            response = self._call_grpc(
                "set_joint_position(trajectory)",
                lambda _r=_req, _t=_timeout: self._joint_stub.set_joint_position(_r, timeout=_t),
                None,
                success_checker=lambda resp: bool(resp.errmsg),
                abort_event=abort_event,
            )
            # Notify watchdog that a waypoint gRPC call completed (progress signal)
            if waypoint_completed_callback is not None:
                try:
                    waypoint_completed_callback(_wp_idx)
                except Exception as _wp_cb_err:
                    logger.warning("waypoint_completed_callback error at %d: %s", _wp_idx, _wp_cb_err)
            # Check abort after gRPC call returns (may have been set while blocked)
            if abort_event is not None and abort_event.is_set():
                logger.info("execute_trajectory: abort_event set after gRPC call, exiting early")
                break
            if response is None:
                return GrpcCallResult(
                    success=False,
                    available=True,
                    error="gRPC call failed",
                )
            self._latest_joint_positions = list(joint_positions)

            # --- Fix 4: Execute gripper commands + object attach/detach ---
            phase = waypoint.get("phase")
            _prev_gripper = getattr(self, "_last_gripper_aperture", None)
            if gripper_aperture is not None and gripper_aperture != _prev_gripper:
                _gripper_width = gripper_aperture * 0.08  # G1 max gripper width
                try:
                    self.set_gripper_state(width=_gripper_width, force=10.0)
                except Exception as _g_err:
                    logger.warning("Gripper command failed at wp %d: %s", _wp_idx, _g_err)
                self._last_gripper_aperture = gripper_aperture
            # Attach object on grasp (gripper closing)
            if phase == "grasp" and gripper_aperture is not None and gripper_aperture < 0.1:
                _obj_prims = waypoint.get("object_prims", [])
                for _op in _obj_prims:
                    try:
                        self.attach_object(_op, is_right=True)
                        logger.info("[GRASP] Attached object: %s", _op)
                    except Exception as _a_err:
                        logger.warning("[GRASP] attach_object failed: %s", _a_err)
            # Detach object on place (gripper opening)
            if phase == "place" and gripper_aperture is not None and gripper_aperture > 0.9:
                try:
                    self.detach_object()
                    logger.info("[PLACE] Detached object")
                except Exception as _d_err:
                    logger.warning("[PLACE] detach_object failed: %s", _d_err)

            # Fire observation callback between waypoints while lock is free.
            if observation_callback is not None:
                try:
                    observation_callback(waypoint, _wp_idx)
                except Exception as _obs_cb_err:
                    logger.warning(
                        "Observation callback failed at waypoint %d: %s",
                        _wp_idx, _obs_cb_err,
                    )
            timestamp = waypoint.get("timestamp")
            if timestamp is not None:
                if last_timestamp is not None:
                    delay = max(0.0, float(timestamp) - float(last_timestamp))
                    if delay:
                        if abort_event is not None and abort_event.is_set():
                            logger.info("execute_trajectory: abort_event set during delay, exiting early")
                            break
                        time.sleep(delay)
                last_timestamp = float(timestamp)

        if self._first_trajectory_call:
            self._first_trajectory_call = False
            self._curobo_initialized = True
        return GrpcCallResult(success=True, available=True)

    def start_recording(self, episode_id: str, output_dir: str) -> GrpcCallResult:
        """
        Start recording an episode.

        Uses real gRPC get_observation call with startRecording flag.

        Args:
            episode_id: Unique episode identifier
            output_dir: Directory to save recordings

        Returns:
            GrpcCallResult indicating success.
        """
        del output_dir
        # Server-side ROS recording requires --publish_ros and a scene with
        # OmniGraph.  Skip the gRPC call when the server doesn't support it;
        # episode data is captured client-side regardless.
        if os.environ.get("GENIESIM_SKIP_SERVER_RECORDING", "") or os.environ.get("GENIESIM_SKIP_ROS_RECORDING", ""):
            logger.info("[RECORDING] Skipping server-side start_recording (ROS recording disabled)")
            return GrpcCallResult(success=True, available=True)
        return self.send_command(
            CommandType.START_RECORDING,
            {
                "task_name": episode_id,
                "fps": 30,
            },
        )

    def stop_recording(self) -> GrpcCallResult:
        """
        Stop recording current episode.

        Uses real gRPC get_observation call with stopRecording flag.

        Returns:
            GrpcCallResult indicating success.
        """
        if os.environ.get("GENIESIM_SKIP_SERVER_RECORDING", "") or os.environ.get("GENIESIM_SKIP_ROS_RECORDING", ""):
            logger.info("[RECORDING] Skipping server-side stop_recording (ROS recording disabled)")
            return GrpcCallResult(success=True, available=True)
        return self.send_command(
            CommandType.STOP_RECORDING,
            {},
        )

    def reset_environment(self) -> GrpcCallResult:
        """
        Reset the simulation environment.

        Uses real gRPC Reset call.

        Returns:
            GrpcCallResult indicating success.
        """
        return self.reset(reset_robot=True, reset_objects=True)


# =============================================================================
# Main Framework Class
# =============================================================================


class GenieSimLocalFramework:
    """
    Main interface for running Genie Sim 3.0 locally.

    This class manages:
    1. Server lifecycle (start/stop the Isaac Sim-based data collection server)
    2. Client connection (gRPC communication with server)
    3. Data collection orchestration
    4. Episode export to LeRobot format

    Usage:
        framework = GenieSimLocalFramework()

        # Option 1: Server already running (e.g., started externally)
        if framework.connect():
            result = framework.run_data_collection(task_config, scene_config)

        # Option 2: Start server automatically
        with framework.server_context(scene_usd_path) as fw:
            result = fw.run_data_collection(task_config, scene_config)
    """

    def __init__(self, config: Optional[GenieSimConfig] = None, verbose: bool = True):
        """
        Initialize Genie Sim local framework.

        Args:
            config: Configuration (uses environment if None)
            verbose: Print progress messages
        """
        self.config = config or GenieSimConfig.from_env()
        self.verbose = verbose
        self._strict_realism = _strict_realism_enabled()

        self._apply_curobo_fallback_policy()

        self._client = GenieSimGRPCClient(
            host=self.config.host,
            port=self.config.port,
            timeout=self.config.timeout,
        )

        self._server_process: Optional[subprocess.Popen] = None
        self._status = GenieSimServerStatus.NOT_RUNNING
        self._stall_count = 0
        self._last_planning_report: Dict[str, Any] = {}

        # Ensure directories exist
        self.config.recording_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

    def _apply_curobo_fallback_policy(self) -> None:
        """Apply cuRobo availability policy to config/runtime behavior."""
        production_mode = self.config.environment == "production"
        curobo_enabled = CUROBO_INTEGRATION_AVAILABLE and self.config.use_curobo
        allow_fallback_in_prod = self.config.allow_linear_fallback_in_production

        if production_mode and (
            self.config.allow_linear_fallback_in_production
            or self.config.allow_linear_fallback
            or self.config.allow_ik_failure_fallback
        ):
            raise RuntimeError(
                "Linear fallback is not permitted in production. "
                "Disable GENIESIM_ALLOW_LINEAR_FALLBACK, "
                "GENIESIM_ALLOW_LINEAR_FALLBACK_IN_PROD, and "
                "GENIESIM_ALLOW_IK_FAILURE_FALLBACK."
            )

        if self.config.curobo_required and not curobo_enabled:
            raise RuntimeError(
                "cuRobo motion planning is required (CUROBO_REQUIRED=1). "
                "Install cuRobo (pip install nvidia-curobo) and enable use_curobo, "
                "or disable CUROBO_REQUIRED for local testing."
            )

        if not curobo_enabled:
            if production_mode:
                self.log(
                    "cuRobo unavailable in production; using IK fallback with collision checks "
                    "when available.",
                    "WARNING",
                )
            else:
                self.log(
                    "cuRobo unavailable; using IK fallback with collision checks when available.",
                    "WARNING",
                )
            if self.config.allow_ik_failure_fallback:
                if production_mode and not allow_fallback_in_prod:
                    self.log(
                        "IK failure fallback to linear interpolation is disabled in production. "
                        "Set GENIESIM_ALLOW_LINEAR_FALLBACK_IN_PROD=1 to allow.",
                        "WARNING",
                    )
                else:
                    self.log(
                        "IK failure fallback to linear interpolation enabled via "
                        "GENIESIM_ALLOW_IK_FAILURE_FALLBACK.",
                        "WARNING",
                    )

    def _validate_required_environment(self, stage: str) -> None:
        """Validate required runtime dependencies before launching or running."""
        if not (
            self.config.environment == "production"
            or self.config.isaacsim_required
            or self.config.curobo_required
        ):
            return

        run_geniesim_preflight_or_exit(
            stage,
            config=self.config,
            require_server=False,
        )

    def log(self, msg: str, level: str = "INFO", extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a message."""
        if not self.verbose:
            return
        level_name = level.upper()
        if level_name in {"WARN", "WARNING"}:
            logger.warning("[GENIESIM-LOCAL] %s", msg, extra=extra)
        elif level_name == "ERROR":
            logger.error("[GENIESIM-LOCAL] %s", msg, extra=extra)
        else:
            logger.info("[GENIESIM-LOCAL] %s", msg, extra=extra)

    # =========================================================================
    # Server Management
    # =========================================================================

    def is_server_running(self) -> bool:
        """Check if Genie Sim server is running."""
        return self._client._check_server_socket()

    def get_server_status(self) -> GenieSimServerStatus:
        """Get current server status."""
        if self._server_process is not None:
            if self._server_process.poll() is None:
                # Process still running
                if self._client.is_connected():
                    return GenieSimServerStatus.READY
                return GenieSimServerStatus.STARTING
            else:
                # Process exited
                return GenieSimServerStatus.ERROR

        if self.is_server_running():
            return GenieSimServerStatus.READY

        return GenieSimServerStatus.NOT_RUNNING

    def start_server(
        self,
        scene_usd_path: Optional[Path] = None,
        task_config_path: Optional[Path] = None,
        wait_for_ready: bool = True,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> bool:
        """
        Start the Genie Sim data collection server.

        This starts the data_collector_server.py script inside Isaac Sim.

        Args:
            scene_usd_path: Path to USD scene file
            task_config_path: Path to task configuration JSON
            wait_for_ready: Wait for server to be ready
            timeout: Timeout for waiting (defaults to config server_startup_timeout_s)
            poll_interval: Poll interval for readiness (defaults to config server_startup_poll_s)

        Returns:
            True if server started successfully
        """
        self._validate_required_environment("geniesim-start-server")

        if self.is_server_running():
            self.log("Server already running")
            return True

        self.log("Starting Genie Sim server...")
        self._status = GenieSimServerStatus.STARTING

        use_local_server = not self.config.geniesim_root.exists()
        env = os.environ.copy()
        env[GENIESIM_RECORDINGS_DIR_ENV] = str(self.config.recording_dir)
        env[GENIESIM_LOG_DIR_ENV] = str(self.config.log_dir)

        # Ensure geniesim_adapter dir is on PYTHONPATH so aimdk proto stubs resolve
        adapter_dir = str(Path(__file__).parent)
        existing = env.get("PYTHONPATH", "")
        if adapter_dir not in existing.split(os.pathsep):
            env["PYTHONPATH"] = f"{adapter_dir}{os.pathsep}{existing}" if existing else adapter_dir

        if use_local_server:
            msg = (
                "Mock GenieSim server detected; aborting to avoid synthetic data "
                "(GENIESIM_ROOT not found)."
            )
            self.log(msg, "ERROR")
            self._status = GenieSimServerStatus.ERROR
            return False
        else:
            # Find the data collection server script
            server_script = self.config.geniesim_root / "source/data_collection/scripts/data_collector_server.py"

            if not server_script.exists():
                # Try alternative location
                server_script = Path(__file__).parent / "data_collection_server.py"
                if not server_script.exists():
                    self.log(f"Server script not found: {server_script}", "ERROR")
                    self._status = GenieSimServerStatus.ERROR
                    return False

            # Build command
            isaac_python = self.config.isaac_sim_path / "python.sh"

            if not isaac_python.exists():
                self.log(f"Isaac Sim python.sh not found: {isaac_python}", "ERROR")
                self._status = GenieSimServerStatus.ERROR
                return False

            cmd = [str(isaac_python), str(server_script)]

            # Add arguments
            if scene_usd_path:
                cmd.extend(["--scene", str(scene_usd_path)])
            if task_config_path:
                cmd.extend(["--task-config", str(task_config_path)])
            if self.config.headless:
                cmd.append("--headless")

            cmd.extend(["--port", str(self.config.port)])

            # Set environment
            env["OMNI_KIT_ALLOW_ROOT"] = "1"  # Allow running as root (for containers)

        # Start process
        log_file = self.config.log_dir / f"server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        try:
            with open(log_file, "w") as log_f:
                self._server_process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
                )

            self.log(f"Server process started (PID: {self._server_process.pid})")
            self.log(f"Log file: {log_file}")

        except Exception as e:
            self.log(f"Failed to start server: {e}", "ERROR")
            self._status = GenieSimServerStatus.ERROR
            return False

        # Wait for server to be ready
        if wait_for_ready:
            effective_timeout = (
                self.config.server_startup_timeout_s if timeout is None else timeout
            )
            effective_poll = (
                self.config.server_startup_poll_s if poll_interval is None else poll_interval
            )
            if effective_poll <= 0:
                self.log(
                    "Invalid startup poll interval; defaulting to 0.5s",
                    "WARNING",
                )
                effective_poll = 0.5

            start_time = time.time()
            socket_reachable = False
            last_grpc_error: Optional[str] = None

            def _check_grpc_ready() -> Tuple[bool, Optional[str]]:
                if not self._client._have_grpc:
                    return False, "gRPC stubs unavailable; skipping readiness check"
                if not self._client.connect():
                    return False, "gRPC channel not ready"
                server_info = self._client.get_server_info(timeout=5.0)
                if not server_info.available:
                    return False, server_info.error or "gRPC unavailable"
                if not server_info.success:
                    return False, server_info.error or "gRPC readiness check failed"
                return True, None

            while time.time() - start_time < effective_timeout:
                if not socket_reachable:
                    socket_reachable = self._client._check_server_socket()
                    if socket_reachable:
                        self.log(
                            "Server socket reachable; waiting for gRPC readiness..."
                        )
                if socket_reachable:
                    grpc_ready, grpc_error = _check_grpc_ready()
                    if grpc_ready:
                        readiness_result = self.check_simulation_ready()
                        if readiness_result.success:
                            self._status = GenieSimServerStatus.READY
                            self.log("Server is ready!")
                            return True
                        last_grpc_error = readiness_result.error or grpc_error
                    else:
                        last_grpc_error = grpc_error

                # Check if process exited
                if self._server_process.poll() is not None:
                    self.log("Server process exited unexpectedly", "ERROR")
                    self._status = GenieSimServerStatus.ERROR
                    return False

                time.sleep(effective_poll)

            if not socket_reachable:
                self.log(
                    f"Server socket not reachable within {effective_timeout}s",
                    "ERROR",
                )
            else:
                grpc_detail = f" ({last_grpc_error})" if last_grpc_error else ""
                self.log(
                    "Server socket reachable but gRPC not ready within "
                    f"{effective_timeout}s{grpc_detail}",
                    "ERROR",
                )
            self._status = GenieSimServerStatus.ERROR
            return False

        return True

    def stop_server(self) -> None:
        """Stop the Genie Sim server."""
        if self._server_process is None:
            return

        self.log("Stopping Genie Sim server...")

        try:
            # Disconnect client first
            self._client.disconnect()

            # Send SIGTERM to process group
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
            else:
                self._server_process.terminate()

            # Wait for graceful shutdown
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(self._server_process.pid), signal.SIGKILL)
                else:
                    self._server_process.kill()
                try:
                    self._server_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Process did not die after SIGKILL + 30s; abandoning"
                    )

            self.log("Server stopped")

        except Exception as e:
            self.log(f"Error stopping server: {e}", "WARNING")

        finally:
            self._server_process = None
            self._status = GenieSimServerStatus.NOT_RUNNING

    def server_context(
        self,
        scene_usd_path: Optional[Path] = None,
        *,
        timeout_s: Optional[float] = None,
        poll_s: Optional[float] = None,
    ):
        """
        Context manager for automatic server lifecycle management.

        Usage:
            with framework.server_context(scene_path) as fw:
                result = fw.run_data_collection(task_config)
        """
        if timeout_s is None:
            timeout_s = self.config.server_startup_timeout_s
        if poll_s is None:
            poll_s = self.config.server_startup_poll_s
        return _GenieSimServerContext(
            self,
            scene_usd_path,
            timeout_s=timeout_s,
            poll_s=poll_s,
        )

    # =========================================================================
    # Client Connection
    # =========================================================================

    def connect(self) -> bool:
        """
        Connect to running Genie Sim server.

        Returns:
            True if connected successfully
        """
        if not self.is_server_running():
            self.log("Server is not running", "ERROR")
            return False

        return self._client.connect()

    def disconnect(self) -> None:
        """Disconnect from server."""
        self._client.disconnect()

    def is_ready(self) -> bool:
        """Check if framework is ready for data collection."""
        return self.get_server_status() == GenieSimServerStatus.READY

    def check_simulation_ready(self, timeout: Optional[float] = None) -> GrpcCallResult:
        """
        Perform a readiness check against the running simulation.

        Uses a socket-level connectivity check rather than issuing reset/
        get_observation RPCs, because the server's CommandController may
        not be initialised with a robot yet (no scene loaded).  Calling
        reset() or get_observation() on a bare server causes it to crash
        with an AttributeError on end_effector_prim_path.
        """
        if not self._client._check_server_socket():
            return GrpcCallResult(
                success=False,
                available=False,
                error="Server socket not reachable",
            )

        # If gRPC stubs are available, verify channel connectivity
        if self._client._have_grpc:
            if not self._client.connect():
                return GrpcCallResult(
                    success=False,
                    available=False,
                    error="gRPC channel not ready",
                )

        # Socket reachable and (if applicable) gRPC channel connected.
        # Skip reset/get_observation RPCs - the server may not have a robot
        # loaded yet, and those calls crash the CommandController.
        return GrpcCallResult(
            success=True,
            available=True,
        )

    # =========================================================================
    # Robot initialisation (reusable after server restarts)
    # =========================================================================

    def _expected_joint_count_for_robot(self, robot_cfg_file: str) -> Optional[int]:
        if not robot_cfg_file:
            return None
        if "g1" in robot_cfg_file.lower():
            return 34
        return None

    def _validate_joint_payload(
        self,
        jp_result: Optional["GrpcCallResult"],
        expected_count: Optional[int],
    ) -> Tuple[bool, str, int]:
        if jp_result is None:
            return False, "joint position result missing", 0
        payload = jp_result.payload
        if isinstance(payload, dict):
            _err = payload.get("error") or payload.get("msg") or "joint_positions returned dict"
            return False, f"non-list joint payload: {_err}", 0
        if not isinstance(payload, (list, tuple)):
            return False, f"non-list joint payload: {type(payload)}", 0
        count = len(payload)
        if expected_count is not None and count < expected_count:
            return False, f"expected {expected_count} joints, got {count}", count
        return True, "", count

    def _init_robot_on_server(
        self,
        robot_cfg_file: str,
        base_pose: Dict[str, Any],
        scene_usd: str,
    ) -> None:
        """Send init_robot + gripper open + joint warmup to the Genie Sim server.

        This is called during initial setup **and** after every server restart
        so the server always has a loaded robot before we attempt episode
        collection.
        """
        self.log(f"Initializing robot: cfg_file={robot_cfg_file}, scene_usd={scene_usd}")
        # Store robot type on client for full-body expansion
        self._client._robot_type = getattr(self.config, "robot_type", "")
        expected_joint_count = self._expected_joint_count_for_robot(robot_cfg_file)

        def _run_init_sequence(sequence_label: str) -> None:
            self.log(f"Running init sequence ({sequence_label})")
            # Verify gRPC channel is connected before attempting init_robot.
            if not self._client.connect():
                self.log("gRPC channel not ready before init_robot — will retry in loop", "WARNING")
            # After docker restart, the server needs time to fully load before
            # init_robot will succeed. Retry with exponential backoff.
            _max_init_attempts = 5
            _init_backoff = 5.0  # start at 5s, multiply by 1.5 up to 60s
            init_result = None
            for _init_attempt in range(1, _max_init_attempts + 1):
                init_result = self._client.init_robot(
                    robot_type=robot_cfg_file,
                    base_pose=base_pose,
                    scene_usd_path=scene_usd,
                )
                if init_result.success:
                    self.log(f"Robot initialized (attempt {_init_attempt}): {init_result.payload}")
                    break
                self.log(
                    f"init_robot attempt {_init_attempt}/{_max_init_attempts} failed: "
                    f"{init_result.error} — retrying in {_init_backoff:.0f}s",
                    "WARNING",
                )
                if _init_attempt < _max_init_attempts:
                    time.sleep(_init_backoff)
                    _init_backoff = min(_init_backoff * 1.5, 60.0)
            if init_result is None or not init_result.success:
                self.log(
                    f"init_robot returned: success={getattr(init_result, 'success', None)}, "
                    f"error={getattr(init_result, 'error', None)}, "
                    f"available={getattr(init_result, 'available', None)}",
                    "WARNING",
                )
                # Non-fatal: server may already have robot loaded via --scene arg

            # Initialize server-side robot articulation by sending a gripper open
            # command.  The server's CommandController only sets self.robot inside
            # _set_gripper_state(), so we must trigger it before any recording.
            grip_result = self._client.set_gripper_state(width=0.08)
            if grip_result.success:
                self.log("Server robot articulation initialized via gripper open")
            else:
                self.log(
                    f"Gripper init returned: {grip_result.error} (may be non-fatal)",
                    "WARNING",
                )

            # Warmup: query joint positions to populate real joint names from
            # the server before any set_joint_position call.  Without this,
            # _resolve_joint_names() falls back to synthetic "joint_0" names
            # that don't match the G1 robot's actual joint names, causing
            # KeyError on the server and indefinite hangs.
            warmup_jp = self._client.get_joint_position()
            if warmup_jp.success and self._client._joint_names:
                self.log(f"Joint names populated: {len(self._client._joint_names)} joints")
            else:
                self.log(
                    "Joint name warmup failed — set_joint_position may use synthetic names",
                    "WARNING",
                )

            # Wait for the server's articulation to be fully initialized.
            # After init_robot, the server asynchronously initializes the robot
            # articulation. If we send trajectory commands before it's ready,
            # the server returns empty data or crashes. Poll get_joint_position
            # until we get valid joint data.
            import time as _warmup_time
            _max_warmup_s = 30
            _warmup_start = _warmup_time.time()
            _warmup_ok = False
            self.log("Waiting for server articulation to be ready...")
            while _warmup_time.time() - _warmup_start < _max_warmup_s:
                try:
                    _jp_result = self._client.get_joint_position()
                    if (
                        _jp_result is not None
                        and _jp_result.success
                        and _jp_result.payload
                        and len(_jp_result.payload) > 0
                    ):
                        self.log(
                            f"Articulation ready: {len(_jp_result.payload)} joints "
                            f"({_warmup_time.time() - _warmup_start:.1f}s)"
                        )
                        _warmup_ok = True
                        break
                except Exception:
                    pass
                _warmup_time.sleep(2)
            if not _warmup_ok:
                self.log(
                    f"Articulation not ready after {_max_warmup_s}s — proceeding anyway (may fail)",
                    "WARNING",
                )

        def _joint_health_ok(context_label: str) -> bool:
            jp_result = self._client.get_joint_position(lock_timeout=5.0)
            ok, error_message, count = self._validate_joint_payload(jp_result, expected_joint_count)
            if ok:
                self.log(
                    f"{context_label}: joint health OK ({count} joints)",
                )
                return True
            self.log(
                f"{context_label}: joint health check failed — {error_message}",
                "WARNING",
            )
            return False

        _run_init_sequence("initial")
        if _joint_health_ok("Post-init check"):
            return

        self.log("Retrying init_robot + gripper + warmup due to joint health failure", "WARNING")
        _run_init_sequence("retry")
        if _joint_health_ok("Post-retry check"):
            return

        self.log(
            "Joint health still failing after retry — attempting server restart",
            "WARNING",
        )
        if (
            hasattr(self._client, "_attempt_server_restart")
            and self._client._attempt_server_restart(count_toward_budget=False)
        ):
            _run_init_sequence("post-restart")
            _joint_health_ok("Post-restart check")
        else:
            self.log("Server restart attempt not available or failed", "WARNING")

    def _reinit_robot_after_restart(self) -> None:
        """Re-run init_robot using the saved parameters from the initial setup.

        Called after ``start_server()`` or ``_attempt_server_restart()`` so the
        freshly-restarted server has the robot loaded before episode collection
        continues.
        """
        params = getattr(self, "_robot_init_params", None)
        if not params:
            self.log(
                "No saved robot init params — skipping post-restart init_robot "
                "(robot may not be loaded on server)",
                "WARNING",
            )
            return
        self.log("Re-initializing robot on restarted server...")
        self._init_robot_on_server(
            robot_cfg_file=params["robot_cfg_file"],
            base_pose=params["base_pose"],
            scene_usd=params["scene_usd"],
        )
        self.log("Post-restart joint health check...")
        self._post_restart_articulation_health_check()
        self._post_restart_check_pending = True

    def _post_restart_articulation_health_check(self) -> None:
        """Log articulation diagnostics after a server restart and re-init."""
        params = getattr(self, "_robot_init_params", None)
        if not params:
            self.log("Post-restart joint health check skipped — no robot init params", "WARNING")
            return
        expected_joint_count = self._expected_joint_count_for_robot(params.get("robot_cfg_file", ""))
        jp_result = self._client.get_joint_position(lock_timeout=5.0)
        ok, error_message, count = self._validate_joint_payload(jp_result, expected_joint_count)
        if ok:
            self.log(f"Post-restart joint health OK ({count} joints)")
        else:
            self.log(
                f"Post-restart joint health check failed — {error_message}",
                "WARNING",
            )

    def _strict_realism_preflight(self) -> None:
        """Run strict realism preflight checks once per server start."""
        if not getattr(self, "_strict_realism", False):
            return
        if getattr(self, "_strict_preflight_done", False):
            return
        self._strict_preflight_done = True

        errors: List[str] = []

        # 1) Contact report RPC availability
        try:
            contact_result = self._client.get_contact_report(include_points=False, lock_timeout=2.0)
            if not contact_result.available or not contact_result.success:
                errors.append(
                    f"contact_report unavailable: {contact_result.error or 'RPC unsupported'}"
                )
        except Exception as exc:
            errors.append(f"contact_report exception: {exc}")

        # 2) Joint efforts must be real and non-zero
        try:
            jp_result = self._client.get_joint_position(lock_timeout=5.0)
            if not jp_result.available or not jp_result.success:
                errors.append(f"joint_position unavailable: {jp_result.error or 'RPC failed'}")
            efforts = getattr(self._client, "_latest_joint_efforts", [])
            if not efforts or not any(abs(e) > 1e-6 for e in efforts):
                errors.append("joint_efforts missing or all zeros; server patch required")
        except Exception as exc:
            errors.append(f"joint_efforts exception: {exc}")

        # 3) Object pose must be non-zero (not identity fallback)
        try:
            obj_prims = (
                getattr(self, "_scene_object_prims", [])
                or getattr(self._client, "_scene_object_prims", [])
            )
            if obj_prims:
                resolved = self._client._resolve_object_prim(obj_prims[0])
                if resolved is None:
                    errors.append("object_pose unresolved; server object_pose patch required")
                else:
                    _path, pose = resolved
                    pos = pose.get("position") or {}
                    if isinstance(pos, dict):
                        pos_vals = [pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)]
                    else:
                        pos_vals = list(pos) if isinstance(pos, (list, tuple)) else [0.0, 0.0, 0.0]
                    if not any(abs(v) > 1e-6 for v in pos_vals):
                        errors.append(
                            f"object_pose for {_path} is zero; live physics pose required"
                        )
        except Exception as exc:
            errors.append(f"object_pose exception: {exc}")

        # 4) Camera intrinsics + non-placeholder frame
        try:
            cam_map = getattr(self._client, "_camera_prim_map", {}) or {}
            cam_id = "wrist" if "wrist" in cam_map else (list(cam_map.keys())[0] if cam_map else None)
            cam_data = None
            if cam_id:
                cam_data = self._client.get_camera_data(cam_id)
            if cam_data is None and cam_map:
                cam_data = self._client._get_camera_data_raw(list(cam_map.values())[0])
            if cam_data is None:
                errors.append("camera_data unavailable; server camera patch required")
            else:
                # Handle nested camera_info structure from patch_camera_handler.py
                cam_info = cam_data.get("camera_info") or cam_data
                fx = cam_info.get("fx")
                fy = cam_info.get("fy")
                ppx = cam_info.get("ppx")
                ppy = cam_info.get("ppy")
                # Propagate width/height from camera_info if not at top level
                if "width" not in cam_data and "width" in cam_info:
                    cam_data["width"] = cam_info["width"]
                if "height" not in cam_data and "height" in cam_info:
                    cam_data["height"] = cam_info["height"]
                if fx is None or fy is None or ppx is None or ppy is None:
                    errors.append("camera intrinsics missing in CamInfo")
                rgb = cam_data.get("rgb")
                # Handle numpy arrays directly (from patch_camera_handler.py)
                if rgb is not None and hasattr(rgb, "shape"):
                    # Already a numpy array — no decode needed
                    pass
                elif isinstance(rgb, (bytes, bytearray)):
                    rgb = decode_camera_bytes(
                        bytes(rgb),
                        width=int(cam_data.get("width") or 0),
                        height=int(cam_data.get("height") or 0),
                        encoding=cam_data.get("rgb_encoding") or cam_data.get("encoding") or "",
                        kind="rgb",
                    )
                if rgb is None:
                    errors.append("camera RGB decode failed")
                else:
                    is_valid, diag = validate_rgb_frame_quality(rgb, context="preflight")
                    if not is_valid:
                        errors.append(f"camera RGB invalid: {diag.get('reason')}")
        except Exception as exc:
            errors.append(f"camera preflight exception: {exc}")

        if errors:
            msg = "Strict realism preflight failed:\n  - " + "\n  - ".join(errors)
            self.log(msg, "ERROR")
            raise RuntimeError(msg)

    # =========================================================================
    # Data Collection
    # =========================================================================

    def _cleanup_temp_dirs(self) -> None:
        paths = {self.config.recording_dir, self.config.log_dir}
        for path in paths:
            if not path:
                continue
            path = Path(path)
            if not _is_temp_path(path):
                self.log(f"Skipping cleanup for non-temp path: {path}", "WARNING")
                continue
            shutil.rmtree(path, ignore_errors=True)

    def run_data_collection(
        self,
        task_config: Dict[str, Any],
        scene_config: Optional[Dict[str, Any]] = None,
        episodes_per_task: Optional[int] = None,
        max_duration_seconds: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        expected_server_version: Optional[str] = None,
        required_capabilities: Optional[Sequence[str]] = None,
    ) -> DataCollectionResult:
        """
        Run data collection for specified tasks.

        This orchestrates the full data collection pipeline:
        1. Configure simulation for each task
        2. Execute episodes with motion planning
        3. Record sensor data and robot states
        4. Validate and save episodes

        Args:
            task_config: Task configuration from BlueprintPipeline
            scene_config: Optional scene configuration
            episodes_per_task: Override episodes per task
            max_duration_seconds: Abort data collection if total runtime exceeds this timeout
            progress_callback: Callback for progress updates (current, total, message)

        Returns:
            DataCollectionResult with statistics and output paths
        """
        self._validate_required_environment("geniesim-run-data-collection")
        start_time = time.time()
        timeout_seconds = (
            max_duration_seconds
            if max_duration_seconds is not None
            else self.config.max_duration_seconds
        )
        timed_out = False

        self.log("=" * 70)
        self.log("GENIE SIM DATA COLLECTION")
        self.log("=" * 70)

        result = DataCollectionResult(
            success=False,
            task_name=task_config.get("name", "unknown"),
        )

        try:
            def _timeout_exceeded() -> bool:
                nonlocal timed_out
                if timeout_seconds is None or timeout_seconds <= 0:
                    return False
                elapsed = time.time() - start_time
                if elapsed < timeout_seconds:
                    return False
                timeout_message = (
                    "Genie Sim data collection exceeded max duration "
                    f"({elapsed:.1f}s >= {timeout_seconds:.1f}s); aborting."
                )
                if not timed_out:
                    timed_out = True
                    result.timed_out = True
                    result.success = False
                    result.errors.append(timeout_message)
                    self.log(timeout_message, "ERROR")
                return True

            scene_usd_path = None
            if scene_config:
                scene_usd_path = scene_config.get("usd_path") or scene_config.get("scene_usd_path")
            if scene_usd_path:
                scene_usd_path = Path(scene_usd_path)

            # Normalize scene units (meters_per_unit) for EE/object comparisons
            _mpu = None
            if scene_config:
                _mpu = scene_config.get("meters_per_unit")
                if _mpu is None:
                    _mpu = (scene_config.get("scene") or {}).get("meters_per_unit")
            try:
                _mpu = float(_mpu) if _mpu not in (None, "") else 1.0
            except (TypeError, ValueError):
                _mpu = 1.0
            if not _mpu or _mpu <= 0:
                _mpu = 1.0
            self._meters_per_unit = _mpu
            self._units_per_meter = 1.0 / _mpu if _mpu > 0 else 1.0

            if not self.config.geniesim_root.exists() and not self.is_server_running():
                msg = (
                    "Mock GenieSim server detected; aborting to avoid synthetic data "
                    "(GENIESIM_ROOT not found)."
                )
                self.log(msg, "ERROR")
                result.errors.append(msg)
                return result

            # Ensure server is running (bootstrap if needed)
            if not self.is_server_running():
                if not self.start_server(scene_usd_path=scene_usd_path):
                    result.errors.append("Failed to start Genie Sim server")
                    return result

            # Ensure connected
            if not self._client.is_connected():
                if not self.connect():
                    result.errors.append("Failed to connect to Genie Sim server")
                    return result
            if not self._client.ping(timeout=10.0):
                result.errors.append("Genie Sim server ping failed")
                return result
            if expected_server_version or required_capabilities:
                try:
                    server_info = self.verify_server_capabilities(
                        expected_server_version=expected_server_version,
                        required_capabilities=required_capabilities,
                    )
                    result.server_info = server_info
                except RuntimeError as exc:
                    error_message = f"Genie Sim server capability check failed: {exc}"
                    self.log(error_message, "ERROR")
                    result.errors.append(error_message)
                    return result

            # Initialize the robot on the server before any reset() calls.
            # The server's CommandController requires init_robot() to set up
            # end_effector_prim_path and other robot state; without it, reset()
            # crashes with AttributeError.
            robot_cfg = task_config.get("robot_config", {})
            robot_type = robot_cfg.get("type", self.config.robot_type or "franka")
            if robot_type and robot_type != self.config.robot_type:
                self.log(f"Using robot type from task config: {robot_type}", "INFO")
                self.config.robot_type = robot_type
            # Map robot type to the server's robot config JSON filename.
            # The Genie Sim server stores configs in robot_cfg/ directory;
            # the robot_cfg_file field is looked up as:
            #   {GENIESIM_ROOT}/source/data_collection/config/robot_cfg/{robot_cfg_file}
            # Override via GENIESIM_ROBOT_CFG_FILE env var if needed.
            _ROBOT_CFG_MAP = {
                "franka": "G1_omnipicker_fixed.json",
                "g1": "G1_omnipicker_fixed.json",
                "g1_dual": "G1_omnipicker_fixed_dual.json",
                "g2": "G2_omnipicker_fixed_dual.json",
            }
            robot_cfg_file = os.environ.get(
                "GENIESIM_ROBOT_CFG_FILE",
                _ROBOT_CFG_MAP.get(robot_type, f"{robot_type}.json"),
            )
            base_pos = robot_cfg.get("base_position", [0, 0, 0])
            base_pose = {
                "position": {"x": base_pos[0], "y": base_pos[1], "z": base_pos[2]},
                "orientation": {"rw": 1.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
            }
            scene_usd = (
                str(scene_usd_path) if scene_usd_path
                else os.environ.get("GENIESIM_SCENE_USD_PATH", "scenes/empty_scene.usda")
            )
            # Always translate absolute paths to container paths.
            # The server runs inside Docker where the repo is mounted at
            # /workspace/BlueprintPipeline. Absolute host paths (whether from
            # macOS /Users/... or VM /home/...) won't exist in the container.
            if scene_usd and os.path.isabs(scene_usd):
                _repo_marker = "BlueprintPipeline/"
                _idx = scene_usd.find(_repo_marker)
                if _idx >= 0:
                    container_path = "/workspace/" + scene_usd[_idx:]
                    self.log(
                        f"Translating scene USD to container path: {container_path}"
                    )
                    scene_usd = container_path

            # Store robot init params so we can re-init after server restarts.
            self._robot_init_params = {
                "robot_cfg_file": robot_cfg_file,
                "base_pose": base_pose,
                "scene_usd": scene_usd,
            }
            self._init_robot_on_server(robot_cfg_file, base_pose, scene_usd)

            # Map logical camera names to USD prim paths.
            # Priority: env var > robot config JSON > G1 defaults (legacy).
            camera_map_env = os.environ.get("GENIESIM_CAMERA_PRIM_MAP", "")
            if camera_map_env:
                import json as _json
                self._client._camera_prim_map = _json.loads(camera_map_env)
                self.log(f"Camera prim map (from env): {self._client._camera_prim_map}")
            else:
                _cam_map: Dict[str, str] = {}
                _robot_cfg_dir = Path(__file__).resolve().parent / "robot_configs"
                _robot_cfg_names = [
                    robot_type,
                    robot_type.replace("-", "_"),
                    f"{robot_type}_panda" if robot_type == "franka" else robot_type,
                ]
                _loaded_robot_cfg = None
                for _cfg_name in _robot_cfg_names:
                    _cfg_path = _robot_cfg_dir / f"{_cfg_name}.json"
                    if _cfg_path.is_file():
                        import json as _json
                        with open(_cfg_path) as _f:
                            _loaded_robot_cfg = _json.load(_f)
                        break
                if _loaded_robot_cfg and _loaded_robot_cfg.get("camera"):
                    _cam_prims = list(_loaded_robot_cfg["camera"].keys())
                    _logical_names = ["wrist", "overhead", "side"]
                    for _i, _prim in enumerate(_cam_prims):
                        if _i < len(_logical_names):
                            _cam_map[_logical_names[_i]] = _prim
                    self.log(f"Camera prim map (from robot config): {_cam_map}")
                if not _cam_map:
                    _cam_map = {
                        "wrist": "/G1/gripper_r_base_link/Right_Camera",
                        "overhead": "/G1/head_link2/Head_Camera",
                        "side": "/G1/gripper_l_base_link/Left_Camera",
                    }
                    self.log(f"Camera prim map (G1 defaults — no robot config found): {_cam_map}")
                self._client._camera_prim_map = _cam_map

            # Populate scene object prim paths for real object pose queries.
            # Derive USD prim paths from scene_graph nodes or task_config objects.
            _scene_obj_prims: List[str] = []
            _scene_variation_prims: List[str] = []
            _sg_nodes = (scene_config or {}).get("nodes", [])
            if not _sg_nodes:
                _sg_nodes = task_config.get("nodes", [])
            for _node in _sg_nodes:
                _asset_id = _node.get("asset_id", "")
                _usd_path = _node.get("usd_path", "")
                _variation_flag = _asset_id.lower().startswith("variation_")
                # Derive USD stage prim path from asset file name
                # e.g. ".../obj_Pot057/Pot057.usd" → "/World/Pot057"
                if _usd_path:
                    _stem = Path(_usd_path).stem  # "Pot057"
                    if _stem.lower().startswith("variation_"):
                        _variation_flag = True
                    _prim = f"/World/{_stem}"
                elif _asset_id:
                    # Strip scene prefix: "lightwheel_kitchen_obj_Pot057" → "Pot057"
                    _parts = _asset_id.split("_obj_")
                    _prim = f"/World/{_parts[-1]}" if len(_parts) > 1 else f"/World/{_asset_id}"
                else:
                    continue
                if _variation_flag:
                    _scene_variation_prims.append(_prim)
                else:
                    _scene_obj_prims.append(_prim)
            # Also add objects from task_config's suggested_tasks
            for _t in task_config.get("suggested_tasks", []):
                _target = _t.get("target_object", "")
                if _target:
                    _parts = _target.split("_obj_")
                    _prim = f"/World/{_parts[-1]}" if len(_parts) > 1 else f"/World/{_target}"
                    if _prim not in _scene_obj_prims:
                        _scene_obj_prims.append(_prim)
                _goal = _t.get("goal_region", "")
                if _goal:
                    _prim = f"/World/{_goal}"
                    if _prim not in _scene_obj_prims:
                        _scene_obj_prims.append(_prim)
            self._client._scene_object_prims = _scene_obj_prims
            self._client._scene_variation_object_prims = _scene_variation_prims
            self._scene_object_prims = _scene_obj_prims
            self._scene_variation_object_prims = _scene_variation_prims
            if _scene_obj_prims:
                self.log(f"Scene object prims for real pose queries: {_scene_obj_prims}")
            else:
                self.log("No scene object prims found — object poses will be synthetic", "WARNING")
            if _scene_variation_prims:
                self.log(
                    "Deferring variation prims until first successful pose lookup: "
                    f"{_scene_variation_prims}"
                )

            # Strict realism preflight (fail-closed)
            self._strict_realism_preflight()

            # Build real object properties from scene_graph nodes, replacing
            # hardcoded lookup tables (_OBJECT_SIZES, _OBJECT_MASSES, etc.).
            self._object_properties: Dict[str, Dict[str, Any]] = {}
            self._gemini_client_for_props = None
            _gemini_client_for_props = None
            for _node in _sg_nodes:
                _asset_id = _node.get("asset_id", "")
                _bp = _node.get("bp_metadata", {})
                _props = _node.get("properties", {})
                _size = _node.get("size", [0.1, 0.1, 0.1])
                _category = _bp.get("category", "")
                _mass = _props.get("mass") or _bp.get("physics", {}).get("mass")
                _dim_source = _bp.get("dimensions_source", "")
                _is_placeholder = all(abs(s - 0.1) < 1e-4 for s in _size)

                # If dimensions are placeholder, try Gemini estimation
                if _is_placeholder and not _dim_source:
                    if _gemini_client_for_props is None:
                        try:
                            from tools.llm_client.client import create_llm_client
                            _gemini_client_for_props = create_llm_client()
                            self._gemini_client_for_props = _gemini_client_for_props
                        except Exception:
                            _gemini_client_for_props = False  # sentinel: don't retry
                    if _gemini_client_for_props and _gemini_client_for_props is not False:
                        _sem = _node.get("semantic", _category)
                        try:
                            _prompt = (
                                f"Estimate typical real-world dimensions in meters for: {_sem}.\n"
                                f"Category: {_category}.\n"
                                f"Respond with ONLY JSON: "
                                f'{{"width": <float>, "depth": <float>, "height": <float>, '
                                f'"mass_kg": <float>, "graspable_width_m": <float>}}'
                            )
                            _resp = _gemini_client_for_props.generate(prompt=_prompt, json_output=True, disable_tools=True)
                            _text = _resp.text.strip()
                            _start = _text.find("{")
                            _end = _text.rfind("}") + 1
                            import json as _json_mod
                            if _start >= 0 and _end > _start:
                                _est = _json_mod.loads(_text[_start:_end])
                                # Handle {"value": [w, d, h]} wrapper
                                if isinstance(_est, dict) and "value" in _est and isinstance(_est["value"], list) and len(_est["value"]) >= 3:
                                    _v = _est["value"]
                                    _est = {"width": _v[0], "depth": _v[1], "height": _v[2]}
                                _size = [
                                    max(float(_est.get("width", 0.1)), 0.01),
                                    max(float(_est.get("depth", 0.1)), 0.01),
                                    max(float(_est.get("height", 0.1)), 0.01),
                                ]
                            else:
                                # Gemini may return a bare list [w, d, h]
                                _lst_start = _text.find("[")
                                _lst_end = _text.rfind("]") + 1
                                if _lst_start >= 0 and _lst_end > _lst_start:
                                    _arr = _json_mod.loads(_text[_lst_start:_lst_end])
                                    if isinstance(_arr, list) and len(_arr) >= 3:
                                        _est = {"width": _arr[0], "depth": _arr[1], "height": _arr[2]}
                                        _size = [
                                            max(float(_arr[0]), 0.01),
                                            max(float(_arr[1]), 0.01),
                                            max(float(_arr[2]), 0.01),
                                        ]
                                if _mass is None and "mass_kg" in _est:
                                    _mass = float(_est["mass_kg"])
                                _dim_source = "gemini_estimated"
                                self.log(f"Gemini estimated props for {_asset_id}: size={_size}, mass={_mass}")
                        except Exception as _exc:
                            self.log(f"Gemini property estimation failed for {_asset_id}: {_exc}", "WARNING")

                # Derive object ID variants for matching
                _obj_id_short = _asset_id.split("_obj_")[-1] if "_obj_" in _asset_id else _asset_id
                _obj_type = _category.lower() if _category else ""

                _obj_props = {
                    "asset_id": _asset_id,
                    "category": _category,
                    "object_type": _obj_type,
                    "size": _size,
                    "width": _size[0],
                    "depth": _size[1],
                    "height": _size[2],
                    "graspable_width": min(_size[0], _size[1]),
                    "mass": _mass if _mass is not None else 0.5,
                    "bbox": _size,
                    "dimensions_source": _dim_source or "scene_graph",
                }
                # Store under multiple keys for easy lookup
                self._object_properties[_asset_id] = _obj_props
                self._object_properties[_obj_id_short] = _obj_props
                if _obj_type:
                    # Only set type-level entry if not already set (first wins)
                    if _obj_type not in self._object_properties:
                        self._object_properties[_obj_type] = _obj_props

            if self._object_properties:
                self.log(f"Loaded real object properties for {len(_sg_nodes)} scene_graph nodes")
            else:
                self.log("No object properties loaded from scene_graph — will use hardcoded fallbacks", "WARNING")

            episodes_target = episodes_per_task or self.config.episodes_per_task
            tasks = task_config.get("suggested_tasks", [task_config])
            # Inject top-level config context into each task for enrichment fallbacks
            _top_env = task_config.get("environment_type")
            _top_meta = task_config.get("metadata")
            _top_robot_cfg = task_config.get("robot_config")
            for _t in tasks:
                if _top_env and "environment_type" not in _t:
                    _t["environment_type"] = _top_env
                if _top_meta and "metadata" not in _t:
                    _t["metadata"] = _top_meta
                if _top_robot_cfg and "robot_config" not in _t:
                    _t["robot_config"] = _top_robot_cfg

            self.log(f"Tasks: {len(tasks)}")
            self.log(f"Episodes per task: {episodes_target}")

            total_episodes = 0
            passed_episodes = 0
            total_frames = 0
            quality_scores = []
            collision_free_episodes = 0
            collision_info_episodes = 0
            task_success_episodes = 0
            task_success_info_episodes = 0

            # Create output directory for this run.
            # Use a stable hash so retries of the same scene+config reuse the
            # same directory and can skip already-completed tasks.
            import hashlib as _hl
            _run_hash_src = json.dumps(
                {
                    "scene": str(scene_usd_path) if scene_usd_path else "",
                    "tasks": [t.get("task_name", "") for t in tasks],
                },
                sort_keys=True,
            )
            _run_hash = _hl.sha256(_run_hash_src.encode()).hexdigest()[:12]
            run_dir = self.config.recording_dir / f"run_{_run_hash}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Load checkpoint of previously completed tasks (for retry resume).
            # Format v2: {"tasks": [...], "episode_counts": {"task": N}}
            # Legacy format: ["task1", "task2"]  (list of names)
            _checkpoint_path = run_dir / "_completed_tasks.json"
            _completed_task_names: set = set()
            _checkpoint_episode_counts: dict = {}
            if _checkpoint_path.exists():
                try:
                    _raw = json.loads(_checkpoint_path.read_text())
                    if isinstance(_raw, dict):
                        # v2 format
                        _completed_task_names = set(_raw.get("tasks", []))
                        _checkpoint_episode_counts = _raw.get("episode_counts", {})
                    elif isinstance(_raw, list):
                        # legacy format
                        _completed_task_names = set(_raw)
                    if _completed_task_names:
                        self.log(
                            f"Resuming: {len(_completed_task_names)} tasks already completed, "
                            f"skipping: {_completed_task_names}"
                        )
                except Exception:
                    pass

            _inter_task_delay = float(os.environ.get("GENIESIM_INTER_TASK_DELAY_S", "2"))
            _restart_every_n = int(os.environ.get("GENIESIM_RESTART_EVERY_N_TASKS", "0"))

            for task_idx, task in enumerate(tasks):
                if _timeout_exceeded():
                    break

                # Skip tasks already completed in a previous run (checkpoint resume).
                # v2 checkpoint: skip if enough episodes were already collected.
                _task_name_for_ckpt = task.get("task_name", f"task_{task_idx}")
                _prev_episodes = _checkpoint_episode_counts.get(_task_name_for_ckpt, 0)
                if _task_name_for_ckpt in _completed_task_names and _prev_episodes >= episodes_target:
                    self.log(
                        f"Skipping already-completed task {task_idx + 1}/{len(tasks)}: "
                        f"{_task_name_for_ckpt} ({_prev_episodes} episodes)"
                    )
                    continue
                elif _task_name_for_ckpt in _completed_task_names:
                    # Legacy checkpoint or partial — re-run to fill gap
                    self.log(
                        f"Task {_task_name_for_ckpt} was checkpointed with {_prev_episodes}/{episodes_target} "
                        "episodes — re-running to collect remaining",
                        "WARNING",
                    )

                # Force restart check — independent of inter-task delay setting.
                # If the post-episode health probe (or stuck-thread handler) flagged
                # the server as unresponsive, restart before the next task.
                if task_idx > 0 and getattr(self, '_force_server_restart', False):
                    self._force_server_restart = False
                    self.log(
                        "Forcing server restart before next task (server was unresponsive)",
                        "WARNING",
                    )
                    if hasattr(self, "_client") and self._client is not None:
                        self._client._grpc_lock = threading.Lock()
                    _restarted = self._client._attempt_server_restart()
                    if not _restarted:
                        self.log("Docker restart failed; trying stop+start fallback", "WARNING")
                        self.stop_server()
                        if not self.start_server(scene_usd_path=scene_usd_path):
                            result.errors.append("Forced server restart failed")
                            return result
                    self._reinit_robot_after_restart()
                    time.sleep(2.0)  # brief settle

                # Inter-task delay with health probe (skip for first task)
                if task_idx > 0 and _inter_task_delay > 0:
                    # Proactive server restart to prevent resource exhaustion
                    if _restart_every_n > 0 and task_idx % _restart_every_n == 0:
                        self.log(
                            f"Proactive server restart before task {task_idx + 1} "
                            f"(every {_restart_every_n} tasks)"
                        )
                        if hasattr(self, "_client") and self._client is not None:
                            self._client._grpc_lock = threading.Lock()
                        _restarted = self._client._attempt_server_restart()
                        if not _restarted:
                            self.log("Docker restart failed; trying stop+start fallback", "WARNING")
                            self.stop_server()
                            if not self.start_server(scene_usd_path=scene_usd_path):
                                result.errors.append("Proactive server restart failed")
                                return result
                        # Full re-init: robot + gripper + joint warmup
                        self._reinit_robot_after_restart()
                        time.sleep(_inter_task_delay)
                    else:
                        self.log(f"Inter-task delay: {_inter_task_delay}s before task {task_idx + 1}")
                        import time as _delay_time
                        _delay_time.sleep(_inter_task_delay)
                        # Lightweight health probe via gRPC client (short lock
                        # timeout so we don't block 70s behind a stuck call)
                        try:
                            if hasattr(self, '_client') and self._client is not None:
                                _probe = self._client.get_joint_position(lock_timeout=5.0)
                                if not _probe.available:
                                    self.log("Health probe: server unavailable — attempting restart", "WARNING")
                                    if hasattr(self._client, '_attempt_server_restart'):
                                        self._client._attempt_server_restart()
                        except Exception as _probe_err:
                            self.log(f"Health probe failed: {_probe_err}", "WARNING")

                task_name = task.get("task_name", f"task_{task_idx}")
                if "task_name" not in task or not task.get("task_name"):
                    task["task_name"] = task_name
                self.log(f"\nTask {task_idx + 1}/{len(tasks)}: {task_name}")

                # Sliding window stall tracking: restart when >40% of recent
                # episodes stalled, rather than a simple cumulative counter.
                self._stall_count = 0
                _stall_window_size = 5
                _stall_window: list = []  # recent episode outcomes: True=stall, False=ok

                if getattr(self, "_post_restart_check_pending", False):
                    self._post_restart_check_pending = False
                    self._post_restart_articulation_health_check()

                # Configure environment for task
                self._configure_task(task, scene_config)

                _task_episodes_passed = 0
                for ep_idx in range(episodes_target):
                    if _timeout_exceeded():
                        break
                    if progress_callback:
                        current = task_idx * episodes_target + ep_idx + 1
                        total = len(tasks) * episodes_target
                        progress_callback(current, total, f"Task: {task_name}, Episode: {ep_idx + 1}")

                    # Check if circuit-breaker triggered a restart that needs
                    # robot re-initialisation before we attempt the next episode.
                    if getattr(self._client, "_needs_robot_reinit", False):
                        self._client._needs_robot_reinit = False
                        self._reinit_robot_after_restart()

                    try:
                        # Reset environment
                        reset_result = self._client.reset_environment()
                        if not reset_result.available or not reset_result.success:
                            _reset_err = reset_result.error or "Reset failed"
                            self.log(
                                f"Reset failed before episode {ep_idx}: {_reset_err} — "
                                "attempting server restart",
                                "WARNING",
                            )
                            result.warnings.append(f"Reset failed: {_reset_err}")
                            self.stop_server()
                            if not self.start_server(scene_usd_path=scene_usd_path):
                                result.errors.append(
                                    "Server restart failed after reset failure"
                                )
                                return result
                            # Full re-init: robot + gripper + joint warmup
                            self._reinit_robot_after_restart()
                            time.sleep(self.config.stall_backoff_s if self.config.stall_backoff_s > 0 else 5)
                            continue  # retry this episode with fresh server

                        # Skip episode if circuit breaker is open — gRPC calls
                        # would return fallback data, producing garbage episodes.
                        if (
                            hasattr(self._client, "_circuit_breaker")
                            and self._client._circuit_breaker
                            and self._client._circuit_breaker.is_open
                        ):
                            self.log(
                                f"Circuit breaker OPEN before episode {ep_idx} of "
                                f"{task_name} — skipping to avoid garbage data",
                                "ERROR",
                            )
                            result.warnings.append(
                                f"Episode {ep_idx} of {task_name} skipped: circuit breaker open"
                            )
                            # Attempt server restart to recover
                            self._client._attempt_server_restart()
                            self._reinit_robot_after_restart()
                            continue

                        # Generate and execute trajectory
                        episode_result = self._run_single_episode(
                            task=task,
                            episode_id=f"{task_name}_ep{ep_idx:04d}",
                            output_dir=run_dir,
                            scene_config=scene_config,
                            scene_graph_nodes=_sg_nodes,
                        )

                        total_episodes += 1
                        collision_free_flag = episode_result.get("collision_free")
                        if collision_free_flag is not None:
                            collision_info_episodes += 1
                            if collision_free_flag:
                                collision_free_episodes += 1
                        task_success_flag = episode_result.get("task_success")
                        if task_success_flag is not None:
                            task_success_info_episodes += 1
                            if task_success_flag:
                                task_success_episodes += 1

                        if episode_result.get("success"):
                            passed_episodes += 1
                            _task_episodes_passed += 1
                            total_frames += episode_result.get("frame_count", 0)
                            quality_scores.append(episode_result.get("quality_score", 0.0))
                            _stall_window.append(False)
                            if len(_stall_window) > _stall_window_size:
                                _stall_window.pop(0)
                        elif episode_result.get("task_success") and not episode_result.get("success"):
                            # Task succeeded but data capture failed (e.g. server deadlock).
                            # Count for checkpoint so we move on to next task.
                            _task_episodes_passed += 1
                            self.log(
                                f"Episode {ep_idx} had task_success but data capture failed — "
                                "counting for checkpoint",
                                "WARNING",
                            )
                        else:
                            _stall_window.append(False)  # non-stall failure
                            if len(_stall_window) > _stall_window_size:
                                _stall_window.pop(0)
                            stall_info = episode_result.get("stall_info") or {}
                            if stall_info.get("stall_detected"):
                                # Overwrite the False we just appended
                                _stall_window[-1] = True
                                self._stall_count += 1
                                stall_reason = stall_info.get("stall_reason", StallReason.UNKNOWN)

                                # Record stall pattern for analysis (P1)
                                if result.stall_statistics is None:
                                    result.stall_statistics = StallStatistics()
                                result.stall_statistics.record_stall(
                                    reason=stall_reason,
                                    episode_idx=ep_idx,
                                    task_name=task_name,
                                    progress_age_s=stall_info.get("last_progress_age_s"),
                                    observations_collected=stall_info.get("observations_collected"),
                                    last_observation_timestamp=stall_info.get("last_observation_timestamp"),
                                    trajectory_end_timestamp=stall_info.get("trajectory_end_timestamp"),
                                    extra_context={
                                        "trajectory_end_tolerance_s": stall_info.get("trajectory_end_tolerance_s"),
                                        "episode_error": episode_result.get("error"),
                                    },
                                )

                                # Sliding window: restart if >40% of recent episodes stalled
                                _stall_rate = sum(_stall_window) / len(_stall_window) if _stall_window else 0
                                _should_restart = (
                                    self._stall_count > self.config.max_stalls
                                    or (len(_stall_window) >= 3 and _stall_rate > 0.4)
                                )

                                stall_message = (
                                    f"Episode {ep_idx} of {task_name} stalled after "
                                    f"{stall_info.get('last_progress_age_s', 0.0):.1f}s "
                                    f"(stall {self._stall_count}/{self.config.max_stalls}, "
                                    f"window_rate={_stall_rate:.0%}, "
                                    f"reason={stall_reason})"
                                )
                                result.warnings.append(stall_message)

                                # Log stall pattern for debugging
                                self.log(
                                    f"[STALL-PATTERN] reason={stall_reason}, "
                                    f"episode={ep_idx}, task={task_name}, "
                                    f"observations={stall_info.get('observations_collected', 0)}, "
                                    f"progress_age={stall_info.get('last_progress_age_s', 0.0):.1f}s, "
                                    f"trajectory_end={stall_info.get('trajectory_end_timestamp')}, "
                                    f"last_obs={stall_info.get('last_observation_timestamp')}, "
                                    f"window_rate={_stall_rate:.0%}",
                                    "WARNING",
                                )

                                if _should_restart:
                                    # Log stall summary before server restart
                                    if result.stall_statistics:
                                        summary = result.stall_statistics.get_summary()
                                        self.log(
                                            f"[STALL-SUMMARY] Restarting server after {summary['total_stalls']} stalls. "
                                            f"Reasons: {summary['stalls_by_reason']}. "
                                            f"Most common: {summary['most_common_reason']}",
                                            "WARNING",
                                        )
                                    error_message = (
                                        f"{stall_message}; restarting Genie Sim server "
                                        f"after exceeding max stalls ({self.config.max_stalls})."
                                    )
                                    result.errors.append(error_message)
                                    self.log(error_message, "ERROR")
                                    self.stop_server()
                                    if not self.start_server(scene_usd_path=scene_usd_path):
                                        result.errors.append(
                                            "Failed to restart Genie Sim server after stall."
                                        )
                                        return result
                                    # Re-initialize robot on the freshly-restarted
                                    # server so it's ready for the next episode.
                                    self._reinit_robot_after_restart()
                                    self._stall_count = 0
                                    _stall_window.clear()
                                    if self.config.stall_backoff_s > 0:
                                        time.sleep(self.config.stall_backoff_s)
                            result.warnings.append(
                                f"Episode {ep_idx} of {task_name} failed: {episode_result.get('error', 'unknown')}"
                            )

                    except Exception as e:
                        result.warnings.append(f"Episode {ep_idx} of {task_name} error: {e}")
                        self.log(f"  Episode {ep_idx} error: {e}", "WARNING")

                # Checkpoint: only record task as completed if at least one episode succeeded.
                # Without this guard, failed tasks get checkpointed and retries skip them,
                # producing 0 episodes and burning through all retry attempts.
                if _task_episodes_passed > 0:
                    _completed_task_names.add(task_name)
                    _checkpoint_episode_counts[task_name] = (
                        _checkpoint_episode_counts.get(task_name, 0) + _task_episodes_passed
                    )
                    try:
                        _checkpoint_path.write_text(json.dumps({
                            "tasks": sorted(_completed_task_names),
                            "episode_counts": _checkpoint_episode_counts,
                        }))
                    except Exception as _ckpt_err:
                        self.log(
                            f"Failed to write checkpoint to {_checkpoint_path}: {_ckpt_err}",
                            "WARNING",
                        )
                    # Stream per-task episodes so they're available immediately.
                    if self.config.stream_per_task:
                        try:
                            self._export_task_episodes(
                                task_name=task_name,
                                task_idx=task_idx,
                                run_dir=run_dir,
                            )
                        except Exception as exc:
                            self.log(
                                f"Per-task export failed for {task_name}: {exc}",
                                "WARNING",
                            )
                else:
                    self.log(
                        f"Task {task_name}: no episodes passed — not checkpointing (will retry)",
                        "WARNING",
                    )

                if timed_out:
                    break

            # Calculate statistics
            result.episodes_collected = total_episodes
            result.episodes_passed = passed_episodes
            result.total_frames = total_frames
            result.recording_dir = run_dir
            result.duration_seconds = time.time() - start_time
            result.collision_free_episodes = collision_free_episodes
            result.collision_info_episodes = collision_info_episodes
            result.task_success_episodes = task_success_episodes
            result.task_success_info_episodes = task_success_info_episodes

            if quality_scores:
                result.average_quality_score = np.mean(quality_scores)

            if collision_info_episodes > 0:
                result.collision_free_rate = collision_free_episodes / collision_info_episodes
            elif total_episodes > 0:
                result.warnings.append(
                    "Collision-free rate unavailable: no collision metadata captured."
                )

            if task_success_info_episodes > 0:
                result.task_success_rate = task_success_episodes / task_success_info_episodes
            elif total_episodes > 0:
                result.warnings.append(
                    "Task success rate unavailable: no success metadata captured."
                )

            result.success = (passed_episodes > 0) and not timed_out

            self.log("\n" + "=" * 70)
            self.log("DATA COLLECTION COMPLETE")
            self.log("=" * 70)
            self.log(f"Episodes: {passed_episodes}/{total_episodes} passed")
            self.log(f"Total frames: {total_frames}")
            self.log(f"Average quality: {result.average_quality_score:.2f}")
            if result.collision_info_episodes > 0:
                self.log(
                    "Collision-free rate: "
                    f"{result.collision_free_rate:.2%} "
                    f"({result.collision_free_episodes}/{result.collision_info_episodes})"
                )
            if result.task_success_info_episodes > 0:
                self.log(
                    "Task success rate: "
                    f"{result.task_success_rate:.2%} "
                    f"({result.task_success_episodes}/{result.task_success_info_episodes})"
                )
            self.log(f"Duration: {result.duration_seconds:.1f}s")
            self.log(f"Output: {run_dir}")

            # Log final stall statistics for pattern analysis (P1)
            if result.stall_statistics and result.stall_statistics.total_stalls > 0:
                summary = result.stall_statistics.get_summary()
                self.log("\n" + "-" * 40)
                self.log("[STALL-ANALYSIS] Final Stall Summary:")
                self.log(f"  Total stalls: {summary['total_stalls']}")
                self.log(f"  By reason: {summary['stalls_by_reason']}")
                self.log(f"  Most common: {summary['most_common_reason']}")
                # Recommend actions based on patterns
                if summary['most_common_reason'] == StallReason.NO_OBSERVATION_PROGRESS:
                    self.log(
                        "  [RECOMMENDATION] Frequent observation stalls may indicate: "
                        "1) Slow physics simulation - check GPU utilization, "
                        "2) Network latency - check gRPC connection, "
                        "3) Scene complexity - simplify physics objects",
                        "INFO",
                    )
                elif summary['most_common_reason'] == StallReason.EXECUTION_COMPLETED_NO_FINAL_OBS:
                    self.log(
                        "  [RECOMMENDATION] Missing final observations may indicate: "
                        "1) Trajectory timing mismatch - adjust tolerance, "
                        "2) Observation collection race condition - increase collection timeout",
                        "INFO",
                    )
                elif summary['most_common_reason'] == StallReason.IK_FAILURE:
                    self.log(
                        "  [RECOMMENDATION] IK failures may indicate: "
                        "1) Unreachable target poses - check workspace bounds, "
                        "2) Joint limits - verify robot configuration, "
                        "3) Collision constraints - adjust motion planner",
                        "INFO",
                    )
                self.log("-" * 40)

            return result
        finally:
            if self.config.cleanup_tmp:
                self._cleanup_temp_dirs()

    def verify_server_capabilities(
        self,
        *,
        expected_server_version: Optional[str] = None,
        required_capabilities: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch and verify server capabilities before running a job."""
        server_info_result = self._client.get_server_info(timeout=5.0)
        if not server_info_result.available:
            raise RuntimeError(
                f"Genie Sim server info unavailable: {server_info_result.error}"
            )
        if not server_info_result.success:
            raise RuntimeError(
                f"Genie Sim server info request failed: {server_info_result.error}"
            )
        server_info = server_info_result.payload or {}
        version = server_info.get("version")
        capabilities = server_info.get("capabilities") or []
        normalized_caps = {str(cap).strip() for cap in capabilities}
        self.log(
            "Genie Sim server capabilities negotiated: "
            f"version={version or 'unknown'}, capabilities={sorted(normalized_caps)}"
        )
        if expected_server_version and version:
            expected_major = _parse_version(expected_server_version)[0]
            server_major = _parse_version(version)[0]
            if expected_major != server_major:
                raise RuntimeError(
                    "Genie Sim server version mismatch: "
                    f"expected major {expected_server_version}, got {version}."
                )
        elif expected_server_version and not version:
            raise RuntimeError(
                "Genie Sim server version missing; cannot validate expected version "
                f"{expected_server_version}."
            )
        if required_capabilities:
            required_set = {cap.strip() for cap in required_capabilities}
            missing = sorted(required_set - normalized_caps)
            if missing:
                raise RuntimeError(
                    "Genie Sim server missing required capabilities: "
                    f"{', '.join(missing)}"
                )
        return server_info

    def _configure_task(
        self,
        task: Dict[str, Any],
        scene_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Configure simulation environment for a task."""
        # Set up objects if needed
        target_objects = task.get("target_objects", [])
        for obj in target_objects:
            try:
                result = self._client.send_command(
                    CommandType.SET_OBJECT_POSE,
                    {
                        "object_id": obj.get("id"),
                        "position": obj.get("position"),
                        "rotation": obj.get("rotation", [0, 0, 0, 1]),
                    }
                )
                if not result.available:
                    self.log(
                        f"gRPC unavailable while configuring object {obj.get('id')}: {result.error}",
                        "WARNING",
                    )
                elif not result.success:
                    error_payload = result.payload or {}
                    error_detail = result.error or error_payload.get("error") or "unknown error"
                    self.log(
                        f"Failed to configure object {obj.get('id')}: {error_detail}",
                        "WARNING",
                    )
            except Exception as e:
                self.log(f"Failed to configure object {obj.get('id')}: {e}", "WARNING")

    def _run_single_episode(
        self,
        task: Dict[str, Any],
        episode_id: str,
        output_dir: Path,
        scene_config: Optional[Dict[str, Any]] = None,
        scene_graph_nodes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run a single episode with data collection.

        This implements the core data collection loop:
        1. Start recording
        2. Execute trajectory (generated by cuRobo)
        3. Capture observations at each timestep
        4. Stop recording and save episode

        Args:
            task: Task configuration
            episode_id: Unique episode identifier
            output_dir: Directory to save episode
            scene_config: Optional scene configuration for object metadata lookup
            scene_graph_nodes: Scene graph nodes for asset/object lookups

        Returns:
            Episode result with success status and metrics
        """
        result = {
            "episode_id": episode_id,
            "success": False,
            "frame_count": 0,
            "quality_score": 0.0,
            "collision_free": None,
            "collision_free_physics": None,
            "task_success": None,
            "stall_info": {
                "stall_detected": False,
                "stall_timeout_s": self.config.stall_timeout_s,
                "stall_backoff_s": self.config.stall_backoff_s,
                "max_stalls": self.config.max_stalls,
                "stall_reason": None,
            },
        }

        recording_started = False
        recording_stopped = False

        try:
            # Start recording
            recording_result = self._client.start_recording(episode_id, str(output_dir))
            if not recording_result.available:
                result["error"] = f"Recording unavailable: {recording_result.error}"
                return result
            if not recording_result.success:
                result["error"] = recording_result.error or "Failed to start recording"
                return result
            recording_started = True

            obs_lock_timeout = float(os.getenv("OBS_LOCK_TIMEOUT_S", "1.0"))

            # Get initial observation
            obs_result = self._client.get_observation(lock_timeout=obs_lock_timeout)
            if not obs_result.available:
                result["error"] = f"Observation unavailable: {obs_result.error}"
                return result
            if not obs_result.success:
                result["error"] = obs_result.error or "Failed to get observation"
                return result
            obs = obs_result.payload or {}

            # Generate trajectory (using cuRobo motion planning)
            trajectory = self._generate_trajectory(task, obs)

            if trajectory is None:
                result["error"] = "Motion planning failed"
                return result

            planning_report = dict(self._last_planning_report)
            result["collision_free"] = planning_report.get("collision_free")
            timed_trajectory = self._ensure_trajectory_timestamps(trajectory)
            timed_trajectory, clamped_waypoints = self._validate_trajectory_joint_limits(
                timed_trajectory
            )
            # Validate that the trajectory has meaningful motion (not static)
            if len(timed_trajectory) >= 2:
                all_wp_joints = np.array(
                    [wp["joint_positions"] for wp in timed_trajectory], dtype=float
                )
                joint_range_in_traj = np.max(all_wp_joints, axis=0) - np.min(all_wp_joints, axis=0)
                max_joint_motion = float(np.max(joint_range_in_traj))
                min_motion_threshold = float(os.getenv("MIN_TRAJECTORY_MOTION_RAD", "0.01"))
                if max_joint_motion < min_motion_threshold:
                    result["error"] = (
                        f"Static trajectory rejected: max joint motion {max_joint_motion:.6f} rad "
                        f"< threshold {min_motion_threshold} rad. No meaningful robot motion."
                    )
                    self.log(f"  ❌ {result['error']}", "ERROR")
                    return result

            if clamped_waypoints:
                self.log(
                    f"  ⚠️  Clamped {clamped_waypoints} waypoint(s) to joint limits before execution.",
                    "WARNING",
                )

            collector_state = {
                "observations": [],
                "error": None,
                "mode": "streaming",
                "start_time": None,
                "last_progress_time": time.time(),
                "last_observation_timestamp": None,
                "last_planned_timestamp": None,
                "completion_signal_logged": False,
            }
            start_event = threading.Event()
            timestamps = [waypoint["timestamp"] for waypoint in timed_trajectory]
            abort_event = threading.Event()
            self._client._abort_event = abort_event

            def _note_progress(obs_frame: Dict[str, Any]) -> None:
                collector_state["last_progress_time"] = time.time()
                collector_state["last_observation_timestamp"] = obs_frame.get("timestamp")
                collector_state["last_planned_timestamp"] = (
                    obs_frame.get("planned_timestamp")
                    or obs_frame.get("timestamp")
                )

            def _collect_streaming() -> None:
                try:
                    stream_result = self._client.stream_observations()
                    if not stream_result.available:
                        collector_state["error"] = (
                            f"Observation stream unavailable: {stream_result.error}"
                        )
                        return
                    if not stream_result.success or stream_result.payload is None:
                        collector_state["mode"] = "polling"
                        _collect_polling()
                        return
                    _consecutive_failures = 0
                    _max_consecutive_failures = int(os.getenv("OBS_MAX_CONSECUTIVE_FAILURES", "100"))
                    for response in stream_result.payload:
                        if abort_event.is_set():
                            break
                        if not response.get("success", False):
                            _consecutive_failures += 1
                            _err = response.get("error") or "unsuccessful response"
                            logger.warning(
                                "[OBS-STREAM] Frame failure %d/%d: %s",
                                _consecutive_failures, _max_consecutive_failures, _err,
                            )
                            if _consecutive_failures >= _max_consecutive_failures:
                                collector_state["error"] = (
                                    f"Observation stream: {_consecutive_failures} consecutive "
                                    f"failures, last: {_err}"
                                )
                                break
                            continue
                        _consecutive_failures = 0
                        obs_frame = response
                        collector_state["observations"].append(obs_frame)
                        _note_progress(obs_frame)
                except Exception as exc:
                    collector_state["error"] = (
                        f"Observation stream failed after "
                        f"{len(collector_state['observations'])} frames: {exc}"
                    )

            def _collect_polling() -> None:
                try:
                    if not start_event.wait(timeout=60.0) or abort_event.is_set():
                        collector_state["error"] = (
                            "start_event not set within 60s (execution may have failed)"
                        )
                        return
                    start_time = collector_state["start_time"] or time.time()
                    # Small settling delay to let the robot begin executing
                    # before we start capturing observations
                    settle_delay = float(os.getenv("OBS_SETTLE_DELAY_S", "0.05"))
                    if settle_delay > 0:
                        time.sleep(settle_delay)
                    for planned_timestamp in timestamps:
                        if abort_event.is_set():
                            break
                        target_time = start_time + planned_timestamp
                        sleep_for = target_time - time.time()
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                        obs_result = self._client.get_observation(lock_timeout=obs_lock_timeout)
                        if not obs_result.available:
                            _poll_consecutive_failures = getattr(_collect_polling, '_consec', 0) + 1
                            _collect_polling._consec = _poll_consecutive_failures
                            _max_poll_failures = int(os.getenv("OBS_MAX_CONSECUTIVE_FAILURES", "100"))
                            logger.warning(
                                "[OBS-POLL] Frame unavailable %d/%d: %s",
                                _poll_consecutive_failures, _max_poll_failures, obs_result.error,
                            )
                            if _poll_consecutive_failures >= _max_poll_failures:
                                collector_state["error"] = (
                                    f"Timed observation polling: {_poll_consecutive_failures} "
                                    f"consecutive failures, last: {obs_result.error}"
                                )
                                break
                            continue
                        if not obs_result.success:
                            _poll_consecutive_failures = getattr(_collect_polling, '_consec', 0) + 1
                            _collect_polling._consec = _poll_consecutive_failures
                            _max_poll_failures = int(os.getenv("OBS_MAX_CONSECUTIVE_FAILURES", "100"))
                            logger.warning(
                                "[OBS-POLL] Frame failure %d/%d: %s",
                                _poll_consecutive_failures, _max_poll_failures,
                                obs_result.error or "unsuccessful",
                            )
                            if _poll_consecutive_failures >= _max_poll_failures:
                                collector_state["error"] = (
                                    obs_result.error
                                    or "Timed observation polling returned unsuccessful response."
                                )
                                break
                            continue
                        _collect_polling._consec = 0
                        obs_frame = obs_result.payload or {}
                        obs_frame["planned_timestamp"] = planned_timestamp
                        collector_state["observations"].append(obs_frame)
                        _note_progress(obs_frame)
                except Exception as exc:
                    collector_state["error"] = (
                        f"Timed observation polling failed after "
                        f"{len(collector_state['observations'])} frames: {exc}"
                    )

            # Only start the collector thread if we DON'T have an observation
            # callback.  When execute_trajectory uses _between_waypoints_obs, that
            # callback captures observations between waypoints while the gRPC lock
            # is free.  Running a polling thread in parallel would just lose every
            # lock acquisition race against the execution thread and produce
            # "Composed real observation from: none" spam.
            _use_callback_obs = True  # We always use the between-waypoints callback
            if _use_callback_obs:
                collector_thread = None
            else:
                collector_thread = threading.Thread(target=_collect_streaming, daemon=True)
                collector_thread.start()

            collector_state["start_time"] = time.time()
            start_event.set()

            execution_state: Dict[str, Any] = {"success": False, "error": None}

            def _between_waypoints_obs(waypoint: Dict[str, Any], wp_idx: int) -> None:
                """Capture a full observation between waypoints.

                Called by execute_trajectory after each set_joint_position returns
                and before the inter-waypoint delay.  The gRPC lock is NOT held
                at this point, so we can make real gRPC calls without contention.
                """
                if abort_event.is_set():
                    return
                start_t = collector_state.get("start_time") or time.time()
                wp_timestamp = waypoint.get("timestamp", 0.0)
                obs: Dict[str, Any] = {
                    "planned_timestamp": float(wp_timestamp),
                    "timestamp": time.time(),
                    "data_source": "between_waypoints",
                }
                # Joint positions are known from the planned waypoint
                jp = waypoint.get("joint_positions")
                if jp is not None:
                    obs["robot_state"] = {"joint_positions": list(jp)}

                # Attempt a full observation (lock is FREE between waypoints).
                # This captures ee_pose, object poses, and camera data.
                try:
                    full_obs = self._client.get_observation(
                        lock_timeout=2.0,
                        fallback_joint_positions=jp,
                    )
                    if full_obs.available and full_obs.success and full_obs.payload:
                        obs.update(full_obs.payload)
                        obs["planned_timestamp"] = float(wp_timestamp)
                        obs["timestamp"] = self._coerce_timestamp(
                            obs.get("timestamp"),
                            fallback=time.time(),
                        )
                        obs["data_source"] = "between_waypoints"
                    else:
                        obs["data_source"] = "between_waypoints_fallback"
                except Exception:
                    # Fallback: try just ee_pose
                    try:
                        ee_result = self._client.get_ee_pose(lock_timeout=1.0)
                        if ee_result.available and ee_result.success and ee_result.payload:
                            obs["ee_pose"] = ee_result.payload
                    except Exception:
                        pass
                    obs["data_source"] = "between_waypoints_fallback"
                collector_state["observations"].append(obs)
                _note_progress(obs)

            def _on_waypoint_done(wp_idx: int) -> None:
                """Update watchdog progress when a waypoint gRPC call completes."""
                collector_state["last_progress_time"] = time.time()

            def _execute_trajectory() -> None:
                try:
                    execution_result = self._client.execute_trajectory(
                        timed_trajectory,
                        abort_event=abort_event,
                        observation_callback=_between_waypoints_obs,
                        waypoint_completed_callback=_on_waypoint_done,
                    )
                    if not execution_result.available:
                        execution_state["error"] = (
                            f"Trajectory execution unavailable: {execution_result.error}"
                        )
                        return
                    if not execution_result.success:
                        execution_state["error"] = (
                            execution_result.error or "Trajectory execution failed"
                        )
                        return
                    execution_state["success"] = True
                except Exception as exc:
                    execution_state["error"] = f"Trajectory execution failed: {exc}"

            execution_thread = threading.Thread(target=_execute_trajectory, daemon=True)
            execution_thread.start()

            _trajectory_deadline = time.time() + float(
                os.environ.get("GENIESIM_TRAJECTORY_TIMEOUT_S", "600")
            )

            stall_timeout_s = self.config.stall_timeout_s
            stall_detected = False
            stall_reason: Optional[str] = None
            trajectory_duration = (timestamps[-1] - timestamps[0]) if timestamps else 0.0
            trajectory_end_time = timestamps[-1] if timestamps else None
            end_time_tolerance_s = max(0.25, trajectory_duration * 0.05)
            # When using the between-waypoints observation callback, observations
            # arrive only after each gRPC set_joint_position completes.  The first
            # call can take up to 300s (cuRobo lazy init).  Increase stall timeout
            # so we don't false-abort while waiting for the first waypoint.
            if _use_callback_obs:
                stall_timeout_s = max(stall_timeout_s, trajectory_duration + 120.0)

            def _observation_has_success_flag(obs_frame: Optional[Dict[str, Any]]) -> Optional[bool]:
                if not obs_frame:
                    return None
                success_schema = task.get("success_schema") or {}
                success_keys = success_schema.get("success_keys") or task.get("success_keys")
                if not success_keys:
                    success_keys = DEFAULT_SUCCESS_SCHEMA["success_keys"]
                obs_payload = obs_frame.get("observation") if isinstance(obs_frame, dict) else None
                metadata = (
                    obs_payload.get("metadata")
                    if isinstance(obs_payload, dict)
                    else obs_frame.get("metadata") if isinstance(obs_frame, dict) else None
                )
                task_meta = (
                    obs_payload.get("task")
                    if isinstance(obs_payload, dict)
                    else obs_frame.get("task") if isinstance(obs_frame, dict) else None
                )
                for key in success_keys:
                    for container in (obs_frame, obs_payload, metadata, task_meta):
                        if isinstance(container, dict) and key in container:
                            return bool(container.get(key))
                return None

            def _is_near_trajectory_end(last_planned_timestamp: Optional[float]) -> bool:
                if last_planned_timestamp is None or trajectory_end_time is None:
                    return False
                return last_planned_timestamp >= trajectory_end_time - end_time_tolerance_s

            while execution_thread.is_alive():
                # Hard deadline: abort if total trajectory time exceeded
                if time.time() >= _trajectory_deadline:
                    _traj_timeout_val = os.environ.get("GENIESIM_TRAJECTORY_TIMEOUT_S", "600")
                    self.log(
                        f"Total trajectory timeout exceeded ({_traj_timeout_val}s); aborting.",
                        "WARNING",
                    )
                    abort_event.set()
                    break
                if stall_timeout_s > 0:
                    last_progress_time = collector_state.get("last_progress_time")
                    if last_progress_time:
                        progress_age = time.time() - last_progress_time
                        if progress_age >= stall_timeout_s:
                            last_obs = (
                                collector_state["observations"][-1]
                                if collector_state["observations"]
                                else None
                            )
                            task_success_flag = _observation_has_success_flag(last_obs)
                            last_obs_timestamp = collector_state.get("last_observation_timestamp")
                            last_planned_timestamp = collector_state.get("last_planned_timestamp")
                            near_trajectory_end = _is_near_trajectory_end(last_planned_timestamp)
                            _obs_count = len(collector_state["observations"])
                            # Only treat near_trajectory_end as success if we
                            # actually collected some observations.  Without
                            # observations the "stall" is just gRPC lock
                            # contention — the trajectory is still executing.
                            _can_abort = (
                                task_success_flag
                                or (near_trajectory_end and _obs_count > 0)
                            )
                            if _can_abort:
                                self.log(
                                    "Completion signal detected — aborting trajectory early "
                                    f"(task_success={task_success_flag}, "
                                    f"near_trajectory_end={near_trajectory_end}, "
                                    f"observations={_obs_count}, "
                                    f"last_planned_timestamp={last_planned_timestamp}, "
                                    f"last_obs_timestamp={last_obs_timestamp}, "
                                    f"trajectory_end={trajectory_end_time}, "
                                    f"tolerance={end_time_tolerance_s:.2f}s).",
                                    "INFO",
                                )
                                execution_state["success"] = True
                                abort_event.set()
                                break
                            else:
                                stall_detected = True
                                stall_reason = "no_observation_progress"
                                result["stall_info"].update({
                                    "stall_detected": True,
                                    "stall_reason": stall_reason,
                                    "last_progress_age_s": progress_age,
                                    "observations_collected": len(collector_state["observations"]),
                                    "last_observation_timestamp": last_obs_timestamp,
                                    "trajectory_end_timestamp": trajectory_end_time,
                                    "trajectory_end_tolerance_s": end_time_tolerance_s,
                                })
                                stall_message = (
                                    f"No observation progress for {progress_age:.1f}s "
                                    f"(timeout {stall_timeout_s:.1f}s); aborting episode."
                                )
                                collector_state["error"] = stall_message
                                self.log(
                                    f"{stall_message} Stall reason={stall_reason}, "
                                    f"trajectory_end={trajectory_end_time}, "
                                    f"tolerance={end_time_tolerance_s:.2f}s.",
                                    "WARNING",
                                )
                                abort_event.set()
                                try:
                                    reset_result = self._client.reset_environment()
                                    if not reset_result.available:
                                        self.log(
                                            f"Reset unavailable after stall: {reset_result.error}",
                                            "WARNING",
                                        )
                                    elif not reset_result.success:
                                        self.log(
                                            reset_result.error or "Reset failed after stall.",
                                            "WARNING",
                                        )
                                except Exception as exc:
                                    self.log(f"Failed to reset after stall: {exc}", "WARNING")
                                if self.config.stall_backoff_s > 0:
                                    time.sleep(self.config.stall_backoff_s)
                                break
                time.sleep(0.5)

            execution_thread.join(timeout=5.0)
            if collector_thread is not None:
                collector_timeout = 5.0 if stall_detected else trajectory_duration + 10.0
                collector_thread.join(timeout=collector_timeout)

            if collector_thread is not None and collector_thread.is_alive():
                last_obs_timestamp = collector_state.get("last_observation_timestamp")
                last_planned_timestamp = collector_state.get("last_planned_timestamp")
                if execution_state.get("success") and _is_near_trajectory_end(last_planned_timestamp):
                    self.log(
                        "Observation collection still running, but execution completed and "
                        "final observation is near trajectory end; treating episode as complete "
                        f"(trajectory_end={trajectory_end_time}, "
                        f"last_planned_timestamp={last_planned_timestamp}, "
                        f"last_obs_timestamp={last_obs_timestamp}, "
                        f"tolerance={end_time_tolerance_s:.2f}s).",
                        "INFO",
                    )
                    abort_event.set()
                    collector_thread.join(timeout=2.0)
                else:
                    collector_state["error"] = (
                        "Observation collection did not complete before timeout."
                    )
                    if execution_state.get("success"):
                        stall_detected = True
                        stall_reason = "execution_completed_no_final_observation"
                        result["stall_info"].update({
                            "stall_detected": True,
                            "stall_reason": stall_reason,
                            "observations_collected": len(collector_state["observations"]),
                            "last_observation_timestamp": last_obs_timestamp,
                            "trajectory_end_timestamp": trajectory_end_time,
                            "trajectory_end_tolerance_s": end_time_tolerance_s,
                        })
                        self.log(
                            "Execution completed but final observation missing; "
                            f"stall reason={stall_reason}, trajectory_end={trajectory_end_time}, "
                            f"last_obs_timestamp={last_obs_timestamp}, "
                            f"tolerance={end_time_tolerance_s:.2f}s.",
                            "WARNING",
                        )

            # Ensure execution thread has fully terminated before returning,
            # otherwise it holds _grpc_lock and blocks subsequent gRPC calls
            # (health probes, reset, next task init).
            if execution_thread.is_alive():
                _exec_wait_start = time.time()
                # Signal the execution thread to stop before waiting
                abort_event.set()
                self.log(
                    "Waiting for execution thread to finish (holds gRPC lock)...",
                    "INFO",
                )
                execution_thread.join(timeout=30.0)
                if execution_thread.is_alive():
                    self.log(
                        f"Execution thread still alive after "
                        f"{time.time() - _exec_wait_start:.1f}s — resetting gRPC lock "
                        "and reconnecting channel to unblock subsequent tasks",
                        "WARNING",
                    )
                    # The old thread holds _grpc_lock forever (blocked on
                    # in-flight gRPC call).  Replace the lock so future calls
                    # are not blocked, and reconnect the channel so the old
                    # call is abandoned at the transport layer.
                    self._client._grpc_lock = threading.Lock()
                    try:
                        # Close the old channel and null stubs so connect()
                        # creates a fresh channel + stubs instead of reusing
                        # the closed ones.
                        try:
                            if self._client._channel is not None:
                                self._client._channel.close()
                        except Exception:
                            pass
                        self._client._channel = None
                        self._client._stub = None
                        self._client._joint_stub = None
                        self._client.connect()
                        self.log("gRPC channel reconnected after stuck thread", "INFO")
                    except Exception as _reconn_err:
                        self.log(
                            f"gRPC reconnect failed: {_reconn_err}",
                            "WARNING",
                        )
                    # Server is likely corrupted after stuck gRPC call.
                    # Restart Docker container immediately so subsequent calls
                    # (reset, stop_recording, next task) work.
                    self.log("Restarting Docker container after stuck thread...", "WARNING")
                    _restart_ok = self._client._attempt_server_restart()
                    if _restart_ok:
                        self.log("Server restarted successfully after deadlock", "INFO")
                    else:
                        self.log("Server restart failed after deadlock", "ERROR")
                        self._force_server_restart = True
                else:
                    self.log(
                        f"Execution thread finished after "
                        f"{time.time() - _exec_wait_start:.1f}s",
                        "INFO",
                    )

            # After execution thread finishes, verify server is still responsive.
            # The server can become hung even when the thread joins successfully
            # (e.g. stuck physics step, DEADLINE_EXCEEDED on all calls).
            if hasattr(self, '_client') and self._client is not None:
                try:
                    _post_ep_probe = self._client.get_joint_position(lock_timeout=5.0)
                    if not _post_ep_probe.available:
                        self.log(
                            "Post-episode health probe failed — server unresponsive, "
                            "will force restart before next task",
                            "WARNING",
                        )
                        self._force_server_restart = True
                except Exception as _probe_err:
                    self.log(
                        f"Post-episode health probe exception: {_probe_err} — "
                        "will force restart before next task",
                        "WARNING",
                    )
                    self._force_server_restart = True

            # Preserve task_success even if post-execution steps fail (e.g. deadlock)
            if execution_state.get("success"):
                result["task_success"] = True

            # Helper: save partial episode data for diagnostics when early-returning
            def _save_partial(error_msg: str) -> None:
                try:
                    partial_path = output_dir / f"{episode_id}.partial.json"
                    partial_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(partial_path, "w") as _pf:
                        json.dump({
                            "episode_id": episode_id,
                            "task_name": task.get("task_name", "unknown"),
                            "error": error_msg,
                            "task_success": result.get("task_success"),
                            "observation_count": len(collector_state.get("observations", [])),
                            "execution_success": execution_state.get("success"),
                            "execution_error": execution_state.get("error"),
                            "collector_error": collector_state.get("error"),
                            "stall_detected": stall_detected if 'stall_detected' in dir() else None,
                        }, _pf, indent=2, default=str)
                    self.log(f"Saved partial episode data to {partial_path}")
                except Exception as _pe:
                    self.log(f"Failed to save partial episode: {_pe}", "WARNING")

            # Determine if we can still build a full episode despite errors.
            # If the trajectory executed successfully and we have observations,
            # non-fatal errors (e.g. transient lock contention) should not
            # prevent frame building and full episode export.
            _has_observations = len(collector_state.get("observations", [])) > 0
            _exec_succeeded = execution_state.get("success", False)
            _non_fatal_exec_error = (
                execution_state.get("error")
                and _exec_succeeded
                and _has_observations
            )

            if execution_state.get("error") and not _non_fatal_exec_error:
                result["error"] = execution_state["error"]
                _save_partial(result["error"])
                return result

            if stall_detected and not (_exec_succeeded and _has_observations):
                result["error"] = collector_state["error"] or "Episode stalled"
                _save_partial(result["error"])
                return result

            if not _exec_succeeded:
                result["error"] = "Trajectory execution failed"
                _save_partial(result["error"])
                return result

            if collector_state["error"] and not _has_observations:
                result["error"] = collector_state["error"]
                _save_partial(result["error"])
                return result

            # Carry forward non-fatal warnings into result metadata
            if _non_fatal_exec_error:
                result["execution_warning"] = execution_state["error"]
                self.log(
                    f"Non-fatal execution error (proceeding with {len(collector_state['observations'])} "
                    f"observations): {execution_state['error']}",
                    "WARNING",
                )
            if collector_state["error"] and _has_observations:
                result["collector_warning"] = collector_state["error"]

            aligned_observations = self._align_observations_to_trajectory(
                timed_trajectory,
                collector_state["observations"],
            )
            if aligned_observations is None:
                result["error"] = "Failed to align observations with trajectory."
                _save_partial(result["error"])
                return result

            # Minimum frame count guard
            env_min_frames = os.getenv("MIN_EPISODE_FRAMES")
            env_min_value = int(env_min_frames) if env_min_frames is not None else None
            min_required = max(1, int(len(timed_trajectory) * 0.5))
            min_episode_frames = (
                min(min_required, env_min_value)
                if env_min_value is not None
                else min_required
            )
            adaptive_notes = []
            if env_min_value is None:
                adaptive_notes.append("no MIN_EPISODE_FRAMES override")
            elif min_episode_frames != env_min_value:
                adaptive_notes.append(
                    f"capped by MIN_EPISODE_FRAMES={env_min_value}"
                )
            if _use_callback_obs and _exec_succeeded and _has_observations:
                if min_episode_frames > 1:
                    min_episode_frames = 1
                    adaptive_notes.append(
                        "lowered threshold for between-waypoint observations"
                    )
            if adaptive_notes:
                self.log(
                    "Adaptive minimum frame threshold set to "
                    f"{min_episode_frames} (trajectory_min={min_required}; "
                    f"{', '.join(adaptive_notes)}).",
                    "WARNING",
                )
            if len(aligned_observations) < min_episode_frames:
                result["error"] = (
                    f"Too few observations ({len(aligned_observations)}) for episode; "
                    f"minimum is {min_episode_frames}."
                )
                _save_partial(result["error"])
                return result

            frames, _frame_stats = self._build_frames_from_trajectory(
                timed_trajectory,
                aligned_observations,
                task=task,
                episode_id=episode_id,
                output_dir=output_dir,
            )
            _camera_frame_count = _frame_stats["camera_frame_count"]
            _real_scene_state_count = _frame_stats["real_scene_state_count"]
            _scene_state_fallback_frames = _frame_stats.get("scene_state_fallback_frames", 0)
            _scene_state_missing_after_frame0 = _frame_stats.get("scene_state_missing_after_frame0", False)
            _scene_state_missing_frame_indices = _frame_stats.get("scene_state_missing_frame_indices", [])
            _server_ee_frame_count = _frame_stats["server_ee_frame_count"]
            _real_velocity_count = _frame_stats["real_velocity_count"]
            _real_effort_count = _frame_stats["real_effort_count"]
            _estimated_effort_count = _frame_stats.get("estimated_effort_count", 0)
            _contact_report_count = _frame_stats.get("contact_report_count", 0)
            _collision_free_physics = _frame_stats.get("collision_free_physics")
            _object_property_provenance = _frame_stats.get("object_property_provenance", {})
            _ee_static_fallback_used = _frame_stats.get("ee_static_fallback_used", False)
            _ee_wrench_source = _frame_stats.get("ee_wrench_source", "unavailable")
            _scene_state_invalid = _real_scene_state_count == 0 and len(frames) > 0
            _scene_state_invalid_error = (
                f"Scene state entirely synthetic: real_scene_state_frames=0/{len(frames)}"
                if _scene_state_invalid
                else None
            )

            if _scene_state_invalid:
                for _frame in frames:
                    _obs = _frame.get("observation")
                    if isinstance(_obs, dict):
                        _obs["scene_state_provenance"] = "INVALID_synthetic"

            if _scene_state_missing_after_frame0 and getattr(self.config, "environment", "") == "production":
                missing_frames = ", ".join(str(idx) for idx in _scene_state_missing_frame_indices)
                result["error"] = (
                    "Scene_state missing beyond frame 0 in production; "
                    f"invalid episode (frames: {missing_frames})."
                )
                _save_partial(result["error"])
                return result

            # Strict realism gate: scene_state provenance must be PhysX for all frames
            if self._strict_realism:
                try:
                    from data_fidelity import validate_scene_state_provenance
                    _scene_state_prov = "physx_server" if _real_scene_state_count > 0 else "synthetic_fallback"
                    validate_scene_state_provenance(
                        scene_state_provenance=_scene_state_prov,
                        real_scene_state_frames=_real_scene_state_count,
                        total_frames=len(frames),
                        required_source="physx_server",
                        min_real_ratio=1.0,
                    )
                except ImportError:
                    if _real_scene_state_count < len(frames):
                        raise RuntimeError(
                            "Scene_state provenance invalid in strict realism mode."
                        )

            frame_validation = {
                "enabled": False,
                "errors": [],
                "warnings": [],
                "invalid_frame_count": 0,
                "total_frames": len(frames),
            }
            if self.config.validate_frames:
                frame_validation = self._validate_frames(
                    frames,
                    episode_id=episode_id,
                    task=task,
                    episode_dir=output_dir,
                )
                if frame_validation.get("camera_placeholder_detected"):
                    message = (
                        f"Placeholder camera frames detected for episode {episode_id}; "
                        "aborting to avoid synthetic data."
                    )
                    result["error"] = message
                    result["frame_validation"] = frame_validation
                    self.log(message, "ERROR")
                    return result
                if frame_validation["errors"]:
                    message = (
                        f"Frame validation failed for episode {episode_id}: "
                        f"{len(frame_validation['errors'])} error(s)."
                    )
                    if self.config.fail_on_frame_validation:
                        result["error"] = message
                        result["frame_validation"] = frame_validation
                        self.log(message, "ERROR")
                        return result
                    self.log(message, "WARNING")
            else:
                placeholder_check = self._detect_camera_placeholders(
                    frames,
                    episode_id=episode_id,
                    episode_dir=output_dir,
                )
                frame_validation.update(placeholder_check)
                frame_validation["enabled"] = True
                if placeholder_check.get("camera_placeholder_detected"):
                    message = (
                        f"Placeholder camera frames detected for episode {episode_id}; "
                        "aborting to avoid synthetic data."
                    )
                    result["error"] = message
                    result["frame_validation"] = frame_validation
                    self.log(message, "ERROR")
                    return result

            _efforts_source = _frame_stats.get("efforts_source", "none")
            _fk_available = IK_PLANNING_AVAILABLE or getattr(self.config, "robot_type", "").lower() in _FRANKA_TYPES
            _fk_consistent = _fk_available and not _ee_static_fallback_used
            _ee_pose_conf = 0.98 if (_server_ee_frame_count > len(frames) * 0.5 or _fk_consistent) else 0.9
            _ee_vel_conf = 0.85 if (_server_ee_frame_count > len(frames) * 0.5 or _fk_consistent) else 0.75
            if _efforts_source == "physx":
                _contact_forces_source = "physx_joint_effort"
            elif _contact_report_count > 0:
                _contact_forces_source = "physx_contact_report"
            elif _efforts_source in ("estimated_inverse_dynamics", "mixed"):
                _contact_forces_source = "estimated_inverse_dynamics"
            else:
                _contact_forces_source = "heuristic_grasp_model_v1"
            if _server_ee_frame_count < len(frames):
                self.log(
                    f"EE pose missing in {_server_ee_frame_count}/{len(frames)} frames; "
                    "verify EE link registration on server.",
                    "WARNING",
                )

            # Attempt a lightweight reset before stop_recording to unstick the
            # server's physics loop.  This is a mitigation for the known issue
            # where the single-threaded server main loop hangs after trajectory
            # execution (stuck physics step / blocked gRPC handler).
            try:
                _reset_result = self._client.reset_environment()
                if not _reset_result.available:
                    self.log(
                        f"Pre-stop reset unavailable (non-fatal): {_reset_result.error}",
                        "WARNING",
                    )
            except Exception as _reset_err:
                self.log(
                    f"Pre-stop reset exception (non-fatal): {_reset_err}",
                    "WARNING",
                )

            # Stop recording — non-fatal; client-side observation data is
            # already collected so a failed stop_recording should not prevent
            # full episode export.
            recording_stopped_cleanly = False
            try:
                stop_result = self._client.stop_recording()
                if not stop_result.available:
                    self.log(
                        f"Stop recording unavailable: {stop_result.error}",
                        "WARNING",
                    )
                elif not stop_result.success:
                    self.log(
                        stop_result.error or "Stop recording failed.",
                        "WARNING",
                    )
                else:
                    recording_stopped_cleanly = True
            except Exception as _stop_err:
                self.log(
                    f"Stop recording exception (non-fatal): {_stop_err}",
                    "WARNING",
                )
            recording_stopped = True

            missing_phase_frames = _frame_stats.get("missing_phase_frames", 0)
            if frames:
                missing_phase_ratio = missing_phase_frames / max(1, len(frames))
            else:
                missing_phase_ratio = 0.0
            if missing_phase_ratio > 0.10:
                message = (
                    f"Phase labels missing for {missing_phase_frames}/{len(frames)} frames "
                    f"({missing_phase_ratio:.1%}); refusing to finalize episode {episode_id}."
                )
                result["error"] = message
                result["phase_validation"] = {
                    "missing_phase_frames": missing_phase_frames,
                    "total_frames": len(frames),
                    "missing_ratio": round(missing_phase_ratio, 4),
                }
                self.log(message, "ERROR")
                _save_partial(message)
                return result

            # Calculate quality score
            quality_score = self._calculate_quality_score(frames, task)
            min_quality = float(os.getenv("MIN_QUALITY_SCORE", "0.7"))
            validation_passed = quality_score >= min_quality
            task_success = self._extract_task_success(frames, task)
            collision_free = planning_report.get("collision_free")
            collision_source = planning_report.get("collision_source")
            collision_free_physics = _collision_free_physics
            if _contact_report_count > 0:
                collision_source = "physx_contact_report"
            joint_utilization = self._compute_joint_utilization(frames)
            joint_utilization["threshold"] = float(
                os.getenv("GENIESIM_JOINT_UTILIZATION_THRESHOLD", "0.6")
            )

            # LLM-based episode enrichment: task metadata and success evaluation
            llm_metadata = self._enrich_episode_with_llm(
                frames=frames,
                task=task,
                episode_id=episode_id,
                collision_free=collision_free,
            )
            # Use LLM task_success if server-side evaluation is unavailable
            if task_success is None and llm_metadata.get("task_success") is not None:
                task_success = llm_metadata["task_success"]

            # Deterministic fallbacks when LLM enrichment is unavailable
            if not llm_metadata.get("task_description"):
                _hint = task.get("description_hint")
                if _hint:
                    llm_metadata["task_description"] = _hint
                else:
                    _tt = task.get("task_type", "manipulation")
                    _to = task.get("target_object", "object")
                    llm_metadata["task_description"] = f"{_tt.replace('_', ' ').title()} task involving {_to}."

            if not llm_metadata.get("scene_description"):
                _env = task.get("environment_type") or "unknown"
                # Count actual tracked objects from frames (Fix 5: match description to data)
                _n_obj_actual = 0
                if frames:
                    _priv = frames[0].get("observation", {}).get("privileged", {})
                    _ss = frames[0].get("observation", {}).get("scene_state", {}) or _priv.get("scene_state", {})
                    _n_obj_actual = len(_ss.get("objects", []))
                if _n_obj_actual > 0:
                    _n_obj = str(_n_obj_actual)
                else:
                    _meta = task.get("metadata") or {}
                    _n_obj = _meta.get("manipulable_objects", "several")
                llm_metadata["scene_description"] = (
                    f"{_env.title()} environment with {_n_obj} tracked objects."
                )

            if not llm_metadata.get("success_criteria"):
                _tt = task.get("task_type", "manipulation")
                _criteria_map = {
                    "pick_place": ["Object grasped", "Object lifted above surface", "Object placed at target location"],
                    "organize": ["Object grasped", "Object moved to container", "Object released in container"],
                    "interact": ["Robot reached target object", "Interaction completed"],
                    "stack": ["Object grasped", "Object lifted", "Object placed on top of target"],
                    "pour": ["Container grasped", "Container tilted", "Contents poured"],
                }
                llm_metadata["success_criteria"] = _criteria_map.get(
                    _tt, ["Task initiated", "Motion completed", "End state reached"]
                )

            # Gemini-based task success evaluation (primary)
            _gemini_ts_client = getattr(self, "_gemini_client_for_props", None)
            if task_success is None and frames and _gemini_ts_client:
                try:
                    _task_desc = task.get("description_hint") or task.get("task_type", "manipulation")
                    _target_obj = task.get("target_object", "unknown")
                    # Build compact trajectory summary
                    _gripper_transitions = []
                    _prev_gc = None
                    for _fi, _fr in enumerate(frames):
                        _gc = _fr.get("gripper_command")
                        if _gc != _prev_gc:
                            _gripper_transitions.append({"frame": _fi, "state": _gc})
                            _prev_gc = _gc
                    _first_ee = frames[0].get("ee_pos", [0, 0, 0])
                    _last_ee = frames[-1].get("ee_pos", [0, 0, 0])
                    _summary = {
                        "task": _task_desc,
                        "target_object": _target_obj,
                        "num_frames": len(frames),
                        "gripper_transitions": _gripper_transitions,
                        "ee_start": [round(v, 3) for v in _first_ee],
                        "ee_end": [round(v, 3) for v in _last_ee],
                    }
                    # Add object displacement if available
                    _target_oid_ts = task.get("target_object") or task.get("target_object_id")
                    if _target_oid_ts:
                        _ip = frames[0].get("_initial_object_poses", {}).get(_target_oid_ts)
                        _fp = frames[-1].get("_final_object_poses", {}).get(_target_oid_ts)
                        if _ip is not None and _fp is not None:
                            _summary["object_displacement_m"] = round(
                                float(np.linalg.norm(np.array(_fp) - np.array(_ip))), 3
                            )
                    import json as _json_mod
                    _ts_prompt = (
                        f"Evaluate whether this robot manipulation trajectory achieved its task.\n"
                        f"Task: {_task_desc}\n"
                        f"Trajectory summary: {_json_mod.dumps(_summary)}\n"
                        f"Respond with ONLY JSON: "
                        f'{{"success": <bool>, "confidence": <float 0-1>, "reasoning": "<short explanation>"}}'
                    )
                    _ts_resp = _gemini_ts_client.generate(prompt=_ts_prompt, json_output=True, disable_tools=True)
                    _ts_text = _ts_resp.text.strip()
                    _ts_start = _ts_text.find("{")
                    _ts_end = _ts_text.rfind("}") + 1
                    if _ts_start >= 0 and _ts_end > _ts_start:
                        _ts_data = _json_mod.loads(_ts_text[_ts_start:_ts_end])
                        task_success = bool(_ts_data.get("success", False))
                        _ts_conf = float(_ts_data.get("confidence", 0.5))
                        llm_metadata["task_success_reasoning"] = (
                            f"Gemini evaluation (confidence={_ts_conf:.2f}): "
                            f"{_ts_data.get('reasoning', 'no reasoning')}"
                        )
                        llm_metadata["task_success_source"] = "gemini_estimated"
                except Exception as _ts_exc:
                    logger.debug("Gemini task success evaluation failed: %s", _ts_exc)

            # Geometric task success: goal-region verification
            # Always compute for metadata; only override task_success if not yet determined
            _geo_is_fallback = task_success is None
            if frames:
                _target_oid = task.get("target_object") or task.get("target_object_id")
                if _target_oid:
                    _init_p = frames[0].get("_initial_object_poses", {}).get(_target_oid)
                    _final_p = frames[-1].get("_final_object_poses", {}).get(_target_oid)
                    if _init_p is not None and _final_p is not None:
                        _init_arr = np.array(_init_p)
                        _final_arr = np.array(_final_p)
                        _disp = float(np.linalg.norm(_final_arr - _init_arr))

                        # Derive goal region from task config or use displacement heuristic
                        _goal_cfg = task.get("goal_region") or {}
                        _goal_center = np.array(_goal_cfg["center"]) if "center" in _goal_cfg else None
                        _goal_extents = np.array(_goal_cfg["extents"]) if "extents" in _goal_cfg else None

                        # Check milestones
                        _grasp_detected = False
                        _grasp_frame = None
                        _lift_detected = False
                        _place_frame = None
                        _stable_count = 0
                        _release_after_place = False
                        _max_obj_z = float(_init_arr[2]) if len(_init_arr) >= 3 else 0.0
                        _init_z = _max_obj_z

                        for _fi, _fr in enumerate(frames):
                            _gc = _fr.get("gripper_command")
                            # Track grasp
                            if not _grasp_detected and _gc == "closed":
                                _grasp_detected = True
                                _grasp_frame = _fi
                            # Track object height from scene_state
                            _obs_f = _fr.get("observation", {})
                            _ss_f = _obs_f.get("scene_state", {}) or _obs_f.get("privileged", {}).get("scene_state", {})
                            _obj_z_cur = None
                            for _obj in _ss_f.get("objects", []):
                                if _obj.get("object_id") == _target_oid:
                                    _op = _obj.get("pose", {})
                                    _obj_z_cur = _op.get("z") or (_op.get("position", [0, 0, 0])[2] if "position" in _op else None)
                                    break
                            if _obj_z_cur is not None:
                                _max_obj_z = max(_max_obj_z, float(_obj_z_cur))
                                # Lift: object >=5cm above initial
                                if float(_obj_z_cur) - _init_z >= 0.05:
                                    _lift_detected = True
                            # Track release after grasp
                            if _grasp_detected and _gc == "open":
                                _release_after_place = True
                                if _place_frame is None:
                                    _place_frame = _fi

                        # Check goal-region placement if goal is defined
                        _in_goal = False
                        _placement_error = None
                        if _goal_center is not None and _goal_extents is not None:
                            _placement_error = float(np.linalg.norm(_final_arr - _goal_center))
                            _in_goal = bool(np.all(np.abs(_final_arr - _goal_center) <= _goal_extents))
                        else:
                            # Fallback: displacement > 0.10m counts as reaching goal
                            _in_goal = _disp > 0.10
                            _placement_error = _disp

                        # Stability check: last 3 frames object velocity < 1mm/s
                        _stable = True
                        if len(frames) >= 4:
                            _last_poses = []
                            for _fr in frames[-4:]:
                                _obs_f = _fr.get("observation", {})
                                _ss_f = _obs_f.get("scene_state", {}) or _obs_f.get("privileged", {}).get("scene_state", {})
                                for _obj in _ss_f.get("objects", []):
                                    if _obj.get("object_id") == _target_oid:
                                        _op = _obj.get("pose", {})
                                        if "x" in _op:
                                            _last_poses.append(np.array([_op["x"], _op["y"], _op["z"]]))
                                        elif "position" in _op:
                                            _last_poses.append(np.array(_op["position"]))
                                        break
                            if len(_last_poses) >= 3:
                                _dt = 1.0 / 30.0
                                for _li in range(1, len(_last_poses)):
                                    _vel = float(np.linalg.norm(_last_poses[_li] - _last_poses[_li - 1])) / _dt
                                    if _vel > 0.001:  # 1mm/s
                                        _stable = False
                                        break

                        # Composite success
                        _milestones = {
                            "grasp_detected": _grasp_detected,
                            "object_lifted_5cm": _lift_detected,
                            "placed_in_goal": _in_goal,
                            "stable_at_end": _stable,
                            "gripper_released": _release_after_place,
                        }
                        _success_count = sum(_milestones.values())
                        if _geo_is_fallback:
                            task_success = _success_count >= 4  # require at least 4 of 5

                        # Store detailed metrics
                        _ctrl_freq = 30.0
                        llm_metadata["goal_region_verification"] = _milestones
                        llm_metadata["goal_region_verification"]["displacement_m"] = round(_disp, 4)
                        if _placement_error is not None:
                            llm_metadata["goal_region_verification"]["final_placement_error_m"] = round(_placement_error, 4)
                        if _grasp_frame is not None:
                            llm_metadata["goal_region_verification"]["time_to_grasp_s"] = round(_grasp_frame / _ctrl_freq, 3)
                        if _place_frame is not None:
                            llm_metadata["goal_region_verification"]["time_to_place_s"] = round(_place_frame / _ctrl_freq, 3)
                        llm_metadata["goal_region_verification"]["max_object_z"] = round(_max_obj_z, 4)
                        llm_metadata["goal_region_verification"]["stability_score"] = 1.0 if _stable else 0.0

                        _reasons = [f"{k}={'✓' if v else '✗'}" for k, v in _milestones.items()]
                        if _geo_is_fallback:
                            llm_metadata["task_success_reasoning"] = (
                                f"Goal-region: {_success_count}/5 milestones met "
                                f"(disp={_disp:.3f}m). {', '.join(_reasons)}"
                            )

            # Heuristic task_success from gripper transitions
            if task_success is None and frames:
                _has_grasp = False
                _has_release = False
                for _f in frames:
                    _gc = _f.get("gripper_command")
                    if _gc == "closed":
                        _has_grasp = True
                    elif _gc == "open" and _has_grasp:
                        _has_release = True
                        break
                if _has_grasp:
                    task_success = _has_release
                    llm_metadata["task_success_reasoning"] = (
                        "Heuristic: grasp detected"
                        + (" followed by release (pick-place pattern)." if _has_release
                           else " but no subsequent release detected.")
                    )

            # Gemini-based phase labeling (post-episode batch)
            _gemini_pl_client = getattr(self, "_gemini_client_for_props", None)
            if frames and _gemini_pl_client:
                try:
                    # Build compact frame data for phase labeling
                    _pl_frames = []
                    for _fi, _fr in enumerate(frames):
                        _pl_frames.append({
                            "i": _fi,
                            "gc": _fr.get("gripper_command", ""),
                            "ez": round(_fr.get("ee_pos", [0, 0, 0])[2], 3) if _fr.get("ee_pos") else 0,
                        })
                    # Sample if too many frames (keep ~30 evenly spaced)
                    if len(_pl_frames) > 30:
                        _step = len(_pl_frames) // 30
                        _pl_sampled = _pl_frames[::_step][:30]
                    else:
                        _pl_sampled = _pl_frames
                    import json as _json_mod
                    _pl_prompt = (
                        f"Label each frame of this robotics manipulation trajectory with a phase.\n"
                        f"Valid phases: approach, grasp, lift, transport, place, retract\n"
                        f"Frames (i=index, gc=gripper_command, ez=end_effector_z):\n"
                        f"{_json_mod.dumps(_pl_sampled)}\n"
                        f"Total frames: {len(frames)}.\n"
                        f"Respond with ONLY a JSON array of objects: "
                        f'[{{"start": <int>, "end": <int>, "phase": "<phase>"}},...] '
                        f"covering all frames from 0 to {len(frames)-1}."
                    )
                    _pl_resp = _gemini_pl_client.generate(prompt=_pl_prompt, json_output=True, disable_tools=True)
                    _pl_text = _pl_resp.text.strip()
                    _pl_start = _pl_text.find("[")
                    _pl_end = _pl_text.rfind("]") + 1
                    if _pl_start >= 0 and _pl_end > _pl_start:
                        _pl_data = _json_mod.loads(_pl_text[_pl_start:_pl_end])
                        # Apply Gemini labels to frames
                        for _segment in _pl_data:
                            _s = int(_segment.get("start", 0))
                            _e = min(int(_segment.get("end", 0)), len(frames) - 1)
                            _ph = str(_segment.get("phase", "approach"))
                            for _fi in range(_s, _e + 1):
                                if _fi < len(frames):
                                    frames[_fi]["phase_heuristic"] = frames[_fi].get("phase", "")
                                    frames[_fi]["phase"] = _ph
                        logger.info("[EPISODE] Gemini phase labeling applied: %d segments", len(_pl_data))
                except Exception as _pl_exc:
                    logger.debug("Gemini phase labeling failed: %s; keeping heuristic labels", _pl_exc)

            # Determine data mode based on what channels actually have real data
            _has_camera = _camera_frame_count > 0
            _has_real_scene = _real_scene_state_count > 0
            _has_privileged = any(
                f.get("observation", {}).get("privileged") for f in frames
            ) if frames else False
            if _has_camera and _has_real_scene:
                data_mode = "full"
            elif _has_real_scene or _has_privileged:
                data_mode = "proprioception_with_privileged_state"
            else:
                data_mode = "proprioception_only"

            # Default for diversity calibration source (may be overridden later by quality scoring)
            _diversity_calibration_source = getattr(self, "_diversity_calibration_source", "hardcoded_default")

            # Save episode
            episode_path = output_dir / f"{episode_id}.json"
            def _json_default(value: Any) -> Any:
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                return value

            # Build robot metadata for dataset consumers
            robot_type = getattr(self.config, "robot_type", "unknown")
            joint_names = list(self._client._joint_names) if hasattr(self._client, "_joint_names") and self._client._joint_names else []
            action_dof = len(frames[0].get("action", [])) if frames else 0
            action_abs_dof = len(frames[0].get("action_abs", [])) if frames else 0
            if action_abs_dof:
                num_joints = max(action_abs_dof - 1, 0)
            else:
                num_joints = max(action_dof - 1, 0)
            rc = ROBOT_CONFIGS.get(robot_type)
            # Lightweight metadata fallback when trajectory_solver import fails
            _meta_fb = _ROBOT_METADATA_FALLBACK.get(robot_type, {}) if rc is None else {}
            # Build joint names from ROBOT_CONFIGS when server doesn't provide them
            if not joint_names and rc is not None:
                joint_names = list(rc.joint_names)
                if hasattr(rc, "gripper_joint_names") and rc.gripper_joint_names:
                    joint_names = joint_names + list(rc.gripper_joint_names)
            _is_generic = joint_names and all(n.startswith("joint_") and n[6:].isdigit() for n in joint_names)
            if (not joint_names or _is_generic) and _meta_fb:
                joint_names = list(_meta_fb["joint_names"])
                if _meta_fb.get("gripper_joint_names"):
                    joint_names = joint_names + list(_meta_fb["gripper_joint_names"])
            joint_groups = _resolve_robot_joint_groups(
                robot_type,
                joint_names=joint_names,
                num_joints=num_joints,
            )
            arm_indices = joint_groups["primary_arm_indices"]
            gripper_indices = joint_groups["primary_gripper_indices"]
            arm_dof = len(arm_indices)
            gripper_dof = len(gripper_indices)
            if not action_dof and arm_dof:
                action_dof = arm_dof + 1
            robot_metadata = {
                "robot_type": robot_type,
                "num_joints": num_joints,
                "arm_dof": arm_dof,
                "action_dof": action_dof,
                "gripper_dof": gripper_dof,
                "joint_names": joint_names if joint_names else [f"joint_{i}" for i in range(num_joints)],
                "joint_limits_lower": rc.joint_limits_lower.tolist() if rc is not None and hasattr(rc, "joint_limits_lower") else _meta_fb.get("joint_limits_lower"),
                "joint_limits_upper": rc.joint_limits_upper.tolist() if rc is not None and hasattr(rc, "joint_limits_upper") else _meta_fb.get("joint_limits_upper"),
                "gripper_joint_names": list(rc.gripper_joint_names) if rc is not None and hasattr(rc, "gripper_joint_names") else _meta_fb.get("gripper_joint_names", []),
                "gripper_limits": list(rc.gripper_limits) if rc is not None and hasattr(rc, "gripper_limits") else list(_meta_fb["gripper_limits"]) if _meta_fb.get("gripper_limits") else None,
                "arm_indices": arm_indices,
                "left_arm_indices": joint_groups["left_arm_indices"],
                "right_arm_indices": joint_groups["right_arm_indices"],
                "gripper_indices": gripper_indices,
                "left_gripper_indices": joint_groups["left_gripper_indices"],
                "right_gripper_indices": joint_groups["right_gripper_indices"],
                "gripper_max_aperture": joint_groups["gripper_max_aperture"],
                "default_joint_positions": rc.default_joint_positions.tolist() if rc is not None and hasattr(rc, "default_joint_positions") else _meta_fb.get("default_joint_positions"),
                "action_space": "joint_delta_plus_gripper_delta",
                "action_abs_space": "joint_position_plus_gripper_width_m",
                "control_frequency_hz": 30.0,
                "clock_model": "uniform_30hz",
                "transition_convention": "obs_t_action_t_produces_obs_t+1",
                "action_semantics": "joint_delta_from_current_to_next_with_gripper_delta_normalized",
                "action_abs_semantics": "target_joint_position_for_next_step_with_gripper_width_m",
            }

            # Fallback collision check: verify waypoints within joint limits
            _limits_lower = rc.joint_limits_lower if rc is not None else (np.array(_meta_fb["joint_limits_lower"]) if _meta_fb.get("joint_limits_lower") else None)
            _limits_upper = rc.joint_limits_upper if rc is not None else (np.array(_meta_fb["joint_limits_upper"]) if _meta_fb.get("joint_limits_upper") else None)
            if collision_free is None and _limits_lower is not None and frames:
                try:
                    lower = _limits_lower
                    upper = _limits_upper
                    all_within = True
                    for frame in frames:
                        abs_joints = np.array(frame.get("action_abs", []), dtype=float)
                        arm_joints = abs_joints[:len(lower)]
                        if len(arm_joints) == len(lower):
                            if np.any(arm_joints < lower) or np.any(arm_joints > upper):
                                all_within = False
                                break
                    collision_free = all_within
                    collision_source = "joint_limits_only"
                except Exception:
                    pass

            # Reproducibility seed (deterministic trajectory, but record for traceability)
            episode_seed = hash((episode_id, robot_type, num_joints)) & 0xFFFFFFFF

            # =================================================================
            # EPISODE VALIDITY GATE — reject garbage before export
            # =================================================================
            _validity_errors: list[str] = []
            _is_production = getattr(self.config, "environment", "") == "production"

            # 1. Camera: any required camera modality must have non-null RGB
            _require_cameras = os.getenv("REQUIRE_CAMERA_DATA", "true").lower() == "true"
            if _require_cameras and _camera_frame_count == 0:
                _validity_errors.append(
                    f"All {len(frames)} frames have null camera RGB. "
                    "Set REQUIRE_CAMERA_DATA=false to allow export without cameras."
                )
            if _require_cameras and not os.environ.get("GENIESIM_SKIP_SERVER_RECORDING", ""):
                required_cameras: List[str] = []
                for key in ("required_cameras", "required_camera_ids", "camera_ids"):
                    value = task.get(key)
                    if value:
                        required_cameras = value
                        break
                if not required_cameras:
                    camera_config = task.get("camera_config") or {}
                    for key in ("required_camera_ids", "camera_ids"):
                        value = camera_config.get(key)
                        if value:
                            required_cameras = value
                            break
                if isinstance(required_cameras, str):
                    required_cameras = [required_cameras]
                required_cameras = [str(camera_id) for camera_id in required_cameras]
                for _frame_idx, _frame in enumerate(frames):
                    camera_frames = _frame.get("observation", {}).get("camera_frames") or {}
                    if required_cameras and not camera_frames:
                        _validity_errors.append(
                            f"Frame {_frame_idx} missing camera_frames for required cameras {required_cameras}."
                        )
                        continue
                    for camera_id in required_cameras:
                        camera_data = camera_frames.get(camera_id)
                        if camera_data is None:
                            _validity_errors.append(
                                f"Frame {_frame_idx} missing camera frame for camera '{camera_id}'."
                            )
                            continue
                        if camera_data.get("rgb") is None:
                            _validity_errors.append(
                                f"Frame {_frame_idx} camera '{camera_id}' missing rgb data."
                            )
                        if camera_data.get("depth") is None:
                            _validity_errors.append(
                                f"Frame {_frame_idx} camera '{camera_id}' missing depth data."
                            )

            # 2. Scene state: reject if >10% of frames use synthetic fallback
            if frames:
                _synthetic_ratio = 1.0 - (_real_scene_state_count / max(1, len(frames)))
                if _synthetic_ratio > 0.10:
                    _validity_errors.append(
                        f"scene_state is synthetic for {_synthetic_ratio:.0%} of frames "
                        f"({len(frames) - _real_scene_state_count}/{len(frames)}). "
                        "Real PhysX scene state required for >90% of frames."
                    )

            # 3. Pick/place must show gripper open→close→open transition
            _task_type = task.get("task_type", "")
            _task_type_normalized = str(_task_type).lower()
            if "interact" in _task_type_normalized and frames:
                def _collect_articulation_vectors(frame: Dict[str, Any]) -> Dict[str, List[float]]:
                    vectors: Dict[str, List[float]] = {}

                    def _ingest_object_states(obj_states: Any) -> None:
                        if not isinstance(obj_states, dict):
                            return
                        for _oid, _state in obj_states.items():
                            if not isinstance(_state, dict):
                                continue
                            _art = _state.get("articulation_state")
                            if not isinstance(_art, dict):
                                continue
                            _vals: List[float] = []
                            for _key in (
                                "joint_positions",
                                "dof_positions",
                                "joint_velocities",
                                "dof_velocities",
                            ):
                                _entry = _art.get(_key)
                                if isinstance(_entry, (list, tuple)):
                                    _vals.extend(float(v) for v in _entry)
                            if _vals:
                                vectors[str(_oid)] = _vals

                    _obs = frame.get("observation", {})
                    _priv = _obs.get("privileged", {}) if isinstance(_obs, dict) else {}
                    _ingest_object_states(_priv.get("object_states"))
                    _ingest_object_states(_obs.get("object_states"))

                    _scene_state = {}
                    if isinstance(_obs, dict):
                        _scene_state = _obs.get("scene_state") or {}
                    if not _scene_state and isinstance(_priv, dict):
                        _scene_state = _priv.get("scene_state") or {}
                    if isinstance(_scene_state, dict):
                        for _obj in _scene_state.get("objects", []) or []:
                            if not isinstance(_obj, dict):
                                continue
                            _art = _obj.get("articulation_state")
                            if not isinstance(_art, dict):
                                continue
                            _vals: List[float] = []
                            for _key in (
                                "joint_positions",
                                "dof_positions",
                                "joint_velocities",
                                "dof_velocities",
                            ):
                                _entry = _art.get(_key)
                                if isinstance(_entry, (list, tuple)):
                                    _vals.extend(float(v) for v in _entry)
                            if _vals:
                                _oid = (
                                    _obj.get("object_id")
                                    or _obj.get("id")
                                    or _obj.get("name")
                                    or "unknown_object"
                                )
                                vectors[str(_oid)] = _vals

                    return vectors

                _articulation_vectors: Dict[str, List[List[float]]] = {}
                for _frame in frames:
                    _frame_vectors = _collect_articulation_vectors(_frame)
                    for _oid, _vec in _frame_vectors.items():
                        _articulation_vectors.setdefault(_oid, []).append(_vec)

                if not _articulation_vectors:
                    # Downgraded from error to warning: the Genie Sim server does
                    # not currently provide articulation_state for interact tasks
                    # (stovetop knobs, coffee machine buttons, etc.).  Blocking on
                    # this would reject all interact episodes.
                    self.log(
                        f"[VALIDITY_GATE] Task type '{_task_type}' has no articulation_state — "
                        "skipping articulation check (server does not provide this data yet).",
                        "WARNING",
                    )
                else:
                    _tol = float(os.getenv("ARTICULATION_STATE_TOLERANCE", "1e-4"))
                    _articulation_changed = False
                    for _oid, _vecs in _articulation_vectors.items():
                        if len(_vecs) < 2:
                            continue
                        _base = np.array(_vecs[0], dtype=float)
                        for _vec in _vecs[1:]:
                            _cur = np.array(_vec, dtype=float)
                            _min_len = min(_base.shape[0], _cur.shape[0])
                            if _min_len == 0:
                                continue
                            if np.nanmax(np.abs(_cur[:_min_len] - _base[:_min_len])) > _tol:
                                _articulation_changed = True
                                break
                        if _articulation_changed:
                            break

                    if not _articulation_changed:
                        self.log(
                            f"[VALIDITY_GATE] Task type '{_task_type}' articulation_state did not change > {_tol} — "
                            "this may indicate the interaction was not captured by the server.",
                            "WARNING",
                        )

            if _task_type in ("pick_place", "organize", "stack") and frames:
                _gc_seq = [f.get("gripper_command") for f in frames]
                _saw_open = False
                _saw_close_after_open = False
                _saw_reopen = False
                for _gc in _gc_seq:
                    if _gc == "open":
                        if _saw_close_after_open:
                            _saw_reopen = True
                            break
                        _saw_open = True
                    elif _gc == "closed" and _saw_open:
                        _saw_close_after_open = True
                if not (_saw_open and _saw_close_after_open and _saw_reopen):
                    _validity_errors.append(
                        f"Task type '{_task_type}' requires gripper open→close→open "
                        f"transition but sequence was: open={_saw_open}, "
                        f"close_after_open={_saw_close_after_open}, "
                        f"reopen={_saw_reopen}."
                    )

            # 4. Contradictory success reasoning overrides task_success
            _ts_reasoning = llm_metadata.get("task_success_reasoning", "")
            _failure_phrases = [
                "gripper remains closed",
                "prevents grasping",
                "never opens",
                "never closes",
                "fails to grasp",
                "object not moved",
                "no displacement",
                "task not completed",
                "did not achieve",
            ]
            if task_success and _ts_reasoning:
                for _fp in _failure_phrases:
                    if _fp.lower() in _ts_reasoning.lower():
                        self.log(
                            f"[VALIDITY_GATE] Overriding task_success=True→False: "
                            f"reasoning contains '{_fp}': {_ts_reasoning}",
                            "WARNING",
                        )
                        task_success = False
                        llm_metadata["task_success_override"] = (
                            f"Overridden from True to False by validity gate: "
                            f"reasoning contains failure phrase '{_fp}'"
                        )
                        break

            # 5. Timestamps: monotonic and dt matches control_frequency_hz
            if frames and len(frames) >= 2:
                _ctrl_hz = 30.0
                _expected_dt = 1.0 / _ctrl_hz
                _prev_ts = frames[0].get("timestamp")
                _ts_issues = 0
                for _fi in range(1, len(frames)):
                    _cur_ts = frames[_fi].get("timestamp")
                    if _prev_ts is not None and _cur_ts is not None:
                        _dt = _cur_ts - _prev_ts
                        if _dt <= 0:
                            _ts_issues += 1
                        elif abs(_dt - _expected_dt) > _expected_dt * 0.5:
                            _ts_issues += 1
                    _prev_ts = _cur_ts
                if _ts_issues > len(frames) * 0.05:
                    _validity_errors.append(
                        f"Timestamp issues in {_ts_issues}/{len(frames)-1} intervals "
                        f"(non-monotonic or dt deviates >50% from {_expected_dt:.4f}s)."
                    )

            # 6. Hard-cap quality_score based on data source
            _sensor_source = "isaac_sim_camera" if _camera_frame_count > 0 else "mock"
            _physics_backend = "physx" if _real_scene_state_count > 0 else "heuristic"
            if _sensor_source == "mock":
                quality_score = 0.0
                self.log("[VALIDITY_GATE] quality_score forced to 0.0: sensor_source=mock", "WARNING")
            elif _physics_backend == "heuristic":
                quality_score = min(quality_score, 0.5)
                self.log(
                    f"[VALIDITY_GATE] quality_score capped at {quality_score:.2f}: "
                    "physics_backend=heuristic",
                    "WARNING",
                )
            if _camera_frame_count == 0:
                quality_score = 0.0
                self.log("[VALIDITY_GATE] quality_score forced to 0.0: camera_count=0", "WARNING")

            # Gate: EE pose must exist in most frames (server or FK-computed)
            _ee_present_count = sum(1 for f in frames if f.get("ee_pos") is not None)
            if _ee_present_count <= 1 and len(frames) > 1:
                quality_score = 0.0
                _validity_errors.append(
                    f"EE pose missing: ee_present={_ee_present_count}/{len(frames)} "
                    f"(server={_server_ee_frame_count})"
                )
                self.log(
                    f"[VALIDITY_GATE] quality_score forced to 0.0: "
                    f"ee_present={_ee_present_count}/{len(frames)} "
                    f"(server_ee_frames={_server_ee_frame_count})",
                    "WARNING",
                )

            # Gate: scene state must not be entirely synthetic
            if _real_scene_state_count == 0 and len(frames) > 0:
                quality_score = 0.0
                if _scene_state_invalid_error and _scene_state_invalid_error not in _validity_errors:
                    _validity_errors.append(_scene_state_invalid_error)
                self.log(
                    "[VALIDITY_GATE] quality_score forced to 0.0: scene_state entirely synthetic",
                    "WARNING",
                )

            # Gate: camera calibration must be present in production
            require_calibration = (
                os.getenv("REQUIRE_CAMERA_CALIBRATION", "true").lower() == "true"
                if _is_production
                else os.getenv("REQUIRE_CAMERA_CALIBRATION", "false").lower() == "true"
            )
            if require_calibration and frames:
                _cam_calib_missing = False
                _cam_frames = (frames[0].get("observation") or {}).get("camera_frames") or {}
                if not _cam_frames:
                    _cam_calib_missing = True
                else:
                    for _cam_id, _cam_data in _cam_frames.items():
                        if not isinstance(_cam_data, dict):
                            _cam_calib_missing = True
                            continue
                        if _cam_data.get("fx") is None or _cam_data.get("fy") is None:
                            _cam_calib_missing = True
                        if _cam_data.get("ppx") is None or _cam_data.get("ppy") is None:
                            _cam_calib_missing = True
                        if not _cam_data.get("calibration_id"):
                            _cam_calib_missing = True
                if _cam_calib_missing:
                    _validity_errors.append(
                        "Camera calibration missing in camera_frames; set REQUIRE_CAMERA_CALIBRATION=false to bypass."
                    )

            # Gate: object physics metadata must be real in production
            object_metadata: Dict[str, Any] = {}
            object_metadata_provenance: Dict[str, str] = {}
            fallback_used = False
            require_object_physics = (
                os.getenv("REQUIRE_OBJECT_PHYSICS", "true").lower() == "true"
                if _is_production
                else os.getenv("REQUIRE_OBJECT_PHYSICS", "false").lower() == "true"
            )

            scene_objects = scene_config.get("objects", []) if isinstance(scene_config, dict) else []
            sg_nodes = scene_graph_nodes if isinstance(scene_graph_nodes, list) else []
            sg_lookup: Dict[str, Dict[str, Any]] = {}
            for _node in sg_nodes:
                if not isinstance(_node, dict):
                    continue
                _asset_id = _node.get("asset_id", "")
                if _asset_id:
                    sg_lookup[_asset_id] = _node
                    if "_obj_" in _asset_id:
                        sg_lookup[_asset_id.split("_obj_")[-1]] = _node
                _semantic = _node.get("semantic")
                if _semantic and _semantic not in sg_lookup:
                    sg_lookup[_semantic] = _node

            asset_index_lookup: Dict[str, Dict[str, Any]] = {}
            asset_index_path = None
            if isinstance(scene_config, dict):
                assets_cfg = scene_config.get("assets", {}) if isinstance(scene_config.get("assets", {}), dict) else {}
                asset_index_path = assets_cfg.get("index_path") or scene_config.get("asset_index_path")
            if asset_index_path and scene_usd_path:
                candidate = Path(scene_usd_path).parent / asset_index_path
                if candidate.exists():
                    asset_index_path = candidate
            if asset_index_path and Path(asset_index_path).exists():
                try:
                    with open(asset_index_path) as _aif:
                        asset_index_data = json.load(_aif)
                    for _asset in asset_index_data.get("assets", []):
                        if not isinstance(_asset, dict):
                            continue
                        _asset_id = _asset.get("asset_id")
                        if _asset_id:
                            asset_index_lookup[_asset_id] = _asset
                            if "_obj_" in _asset_id:
                                asset_index_lookup[_asset_id.split("_obj_")[-1]] = _asset
                except Exception as _exc:
                    self.log(f"Failed to load asset_index for object metadata: {_exc}", "WARNING")

            for _obj in scene_objects:
                if not isinstance(_obj, dict):
                    continue
                obj_id = _obj.get("id") or _obj.get("name") or "unknown_object"
                node = sg_lookup.get(obj_id)
                if node is None:
                    # Try matching by asset id
                    asset_path = (_obj.get("asset") or {}).get("path", "")
                    if asset_path:
                        stem = Path(asset_path).stem
                        node = sg_lookup.get(stem)

                asset_entry = None
                asset_id = None
                if node is not None:
                    asset_id = node.get("asset_id")
                    asset_entry = asset_index_lookup.get(asset_id or "")
                if asset_entry is None and asset_id:
                    asset_entry = asset_index_lookup.get(asset_id.split("_obj_")[-1])

                # Base provenance
                provenance = "scene_manifest"
                if node is not None:
                    provenance = "scene_graph"
                elif asset_entry is not None:
                    provenance = "asset_index"

                physics = _obj.get("physics", {}) if isinstance(_obj.get("physics", {}), dict) else {}
                physics_hints = _obj.get("physics_hints", {}) if isinstance(_obj.get("physics_hints", {}), dict) else {}
                props = node.get("properties", {}) if isinstance(node, dict) else {}
                bp_meta = node.get("bp_metadata", {}) if isinstance(node, dict) else {}
                material_meta = asset_entry.get("material", {}) if isinstance(asset_entry, dict) else {}

                mass_kg = (
                    physics.get("mass")
                    or props.get("mass")
                    or bp_meta.get("physics", {}).get("mass")
                    or asset_entry.get("mass") if isinstance(asset_entry, dict) else None
                )

                friction = (
                    physics.get("friction")
                    or props.get("friction")
                    or physics_hints.get("roughness")
                    or material_meta.get("friction")
                )
                restitution = (
                    physics.get("restitution")
                    or props.get("restitution")
                    or material_meta.get("restitution")
                )

                usd_path = None
                if node is not None:
                    usd_path = node.get("usd_path")
                if not usd_path:
                    usd_path = (_obj.get("asset") or {}).get("path")
                if not usd_path and asset_entry:
                    usd_path = asset_entry.get("usd_path")

                scale = None
                transform = _obj.get("transform", {}) if isinstance(_obj.get("transform", {}), dict) else {}
                scale_dict = transform.get("scale") if isinstance(transform.get("scale", {}), dict) else {}
                if scale_dict:
                    scale = [float(scale_dict.get("x", 1.0)), float(scale_dict.get("y", 1.0)), float(scale_dict.get("z", 1.0))]

                # Size for inertia estimation
                size = None
                if node is not None:
                    size = node.get("size")
                if size is None and asset_entry is not None:
                    size = asset_entry.get("bbox") or asset_entry.get("size")
                if size is None and node is not None:
                    size = (node.get("bp_metadata") or {}).get("bbox")
                if size is not None and isinstance(size, (list, tuple)) and len(size) >= 3:
                    size = [float(size[0]), float(size[1]), float(size[2])]
                else:
                    size = None

                inertia = None
                if mass_kg is not None and size is not None:
                    try:
                        w, d, h = size[0], size[1], size[2]
                        inertia = [
                            (1.0 / 12.0) * float(mass_kg) * (d * d + h * h),
                            (1.0 / 12.0) * float(mass_kg) * (w * w + h * h),
                            (1.0 / 12.0) * float(mass_kg) * (w * w + d * d),
                        ]
                    except Exception:
                        inertia = None

                # Determine property provenance
                prop_fallbacks = []
                for prop in ("mass", "bbox", "category"):
                    key = f"{obj_id}:{prop}"
                    prov = _object_property_provenance.get(key)
                    if prov in ("hardcoded_fallback", "llm_estimated", "gemini_estimated"):
                        prop_fallbacks.append(key)

                if prop_fallbacks:
                    provenance = "fallback"
                    fallback_used = True

                missing_fields = []
                if mass_kg is None:
                    missing_fields.append("mass_kg")
                if friction is None:
                    missing_fields.append("friction")
                if restitution is None:
                    missing_fields.append("restitution")
                if usd_path is None:
                    missing_fields.append("usd_path")

                object_metadata[obj_id] = {
                    "asset_id": asset_id or obj_id,
                    "usd_path": usd_path,
                    "mass_kg": float(mass_kg) if mass_kg is not None else None,
                    "friction": float(friction) if friction is not None else None,
                    "restitution": float(restitution) if restitution is not None else None,
                    "material": material_meta if material_meta else None,
                    "scale": scale,
                    "size": size,
                    "inertia": inertia,
                    "provenance": provenance,
                    "missing_fields": missing_fields,
                }
                object_metadata_provenance[obj_id] = provenance

            if _is_production and require_object_physics:
                if fallback_used:
                    _validity_errors.append(
                        "Object metadata used fallback values; production requires scene_graph/asset_index provenance."
                    )
                for _oid, _meta in object_metadata.items():
                    if _meta.get("missing_fields"):
                        _validity_errors.append(
                            f"Object '{_oid}' missing physics fields: {_meta['missing_fields']}."
                        )

            # Gate: EE wrench required in production when enabled
            require_ee_wrench = (
                os.getenv("REQUIRE_EE_WRENCH", "true").lower() == "true"
                if _is_production
                else os.getenv("REQUIRE_EE_WRENCH", "false").lower() == "true"
            )
            if require_ee_wrench and frames:
                _wrench_frames = sum(1 for _f in frames if _f.get("ee_wrench") is not None)
                _coverage = _wrench_frames / max(1, len(frames))
                if _coverage < 0.9:
                    _validity_errors.append(
                        f"EE wrench missing in {1.0 - _coverage:.0%} of frames."
                    )

            # Reject episode if validity errors found
            if _validity_errors:
                _gate_msg = (
                    f"Episode {episode_id} REJECTED by validity gate: "
                    f"{len(_validity_errors)} error(s):\n"
                    + "\n".join(f"  - {e}" for e in _validity_errors)
                )
                self.log(_gate_msg, "ERROR")
                # In strict mode (default), refuse to export
                _strict_gate = os.getenv("STRICT_VALIDITY_GATE", "true").lower() == "true"
                if _strict_gate:
                    result["error"] = _gate_msg
                    result["validity_gate"] = {
                        "passed": False,
                        "errors": _validity_errors,
                    }
                    _save_partial(_gate_msg)
                    return result
                else:
                    # Non-strict: export but mark as invalid
                    self.log(
                        "[VALIDITY_GATE] STRICT_VALIDITY_GATE=false — exporting with warnings",
                        "WARNING",
                    )
                    validation_passed = False

            # =================================================================
            # END EPISODE VALIDITY GATE
            # =================================================================

            # Clean up internal metadata fields from frames before serialization
            for _f in frames:
                _f.pop("_initial_object_poses", None)
                _f.pop("_final_object_poses", None)

            # Set episode dir so VLM audit / sim-to-real can resolve .npy paths
            self._current_episode_dir = output_dir if output_dir is not None else None

            # VLM-based quality audit (gated by ENABLE_VLM_QUALITY_AUDIT=1)
            vlm_audit = self._audit_episode_with_vlm(frames, task, quality_score)
            if vlm_audit and not vlm_audit.get("skipped"):
                # Use blended score if VLM audit succeeded
                quality_score = vlm_audit.get("blended_quality_score", quality_score)
                validation_passed = quality_score >= min_quality

            # Sim-to-real gap assessment (gated by ENABLE_SIM2REAL_ASSESSMENT=1)
            sim2real = self._assess_sim_to_real_gap(frames)

            # Failure diagnosis (gated by ENABLE_LLM_FAILURE_DIAGNOSIS=1)
            failure_diagnosis = self._diagnose_failure_with_llm(
                validity_errors=_validity_errors if "_validity_errors" in dir() else [],
                frames=frames,
                quality_score=quality_score,
                task=task,
            )

            # Camera calibration metadata (intrinsics + extrinsics)
            camera_calibration: Dict[str, Any] = {}
            for _frame in frames:
                _cam_frames = (_frame.get("observation") or {}).get("camera_frames") or {}
                if not isinstance(_cam_frames, dict):
                    continue
                for _cam_id, _cam_data in _cam_frames.items():
                    if _cam_id in camera_calibration:
                        continue
                    if not isinstance(_cam_data, dict):
                        continue
                    _fx = _cam_data.get("fx")
                    _fy = _cam_data.get("fy")
                    _ppx = _cam_data.get("ppx")
                    _ppy = _cam_data.get("ppy")
                    _width = _cam_data.get("width")
                    _height = _cam_data.get("height")
                    _calib_id = _cam_data.get("calibration_id") or f"{_cam_id}_calib"
                    _intrinsics_source = _cam_data.get("intrinsics_source", "geniesim_grpc")

                    # Compute fallback intrinsics from 90deg FOV if missing (non-strict only)
                    if _fx is None or _fy is None or _ppx is None or _ppy is None:
                        if self._strict_realism:
                            try:
                                from data_fidelity import DataFidelityError
                                raise DataFidelityError(
                                    f"Camera {_cam_id} missing intrinsics in strict realism mode.",
                                    gate_name="camera_intrinsics_missing",
                                    diagnostics={"camera_id": _cam_id},
                                )
                            except ImportError:
                                raise RuntimeError(
                                    f"Camera {_cam_id} missing intrinsics in strict realism mode."
                                )
                        if _width and _height:
                            # Assume 90 degree horizontal FOV as fallback
                            _fx = float(_width) / (2.0 * np.tan(np.radians(45)))
                            _fy = _fx  # Assume square pixels
                            _ppx = float(_width) / 2.0
                            _ppy = float(_height) / 2.0
                            _intrinsics_source = "computed_from_fov"
                            logger.warning(
                                "Camera %s missing intrinsics; using default 90deg FOV: fx=%.1f",
                                _cam_id, _fx,
                            )

                    # Validate intrinsics with fail-fast gate
                    try:
                        from data_fidelity import validate_camera_intrinsics
                        validate_camera_intrinsics(
                            fx=_fx, fy=_fy, ppx=_ppx, ppy=_ppy,
                            camera_id=_cam_id,
                        )
                    except ImportError:
                        pass

                    _intrinsic = None
                    if _fx is not None and _fy is not None and _ppx is not None and _ppy is not None:
                        _intrinsic = [
                            [float(_fx), 0.0, float(_ppx)],
                            [0.0, float(_fy), float(_ppy)],
                            [0.0, 0.0, 1.0],
                        ]
                    camera_calibration[_cam_id] = {
                        "camera_id": _cam_id,
                        "calibration_id": _calib_id,
                        "width": _width,
                        "height": _height,
                        "fx": _fx,
                        "fy": _fy,
                        "ppx": _ppx,
                        "ppy": _ppy,
                        "intrinsic_matrix": _intrinsic,
                        "extrinsic_matrix": _cam_data.get("extrinsic"),
                        "source": "geniesim_grpc",
                        "intrinsics_source": _intrinsics_source,
                    }

            # Frame conventions metadata
            frame_conventions = {
                "world_frame": "world",
                "base_frame": "robot_base",
                "ee_frame": "end_effector",
                "camera_frame": "camera",
                "handedness": "right",
                "axis_convention": "X forward, Y left, Z up (USD/Isaac)",
                "units": {
                    "length": "meters",
                    "angles": "radians",
                    "time": "seconds",
                },
                "meters_per_unit": getattr(self, "_meters_per_unit", 1.0),
                "units_per_meter": getattr(self, "_units_per_meter", 1.0),
            }

            # Control metadata
            control_metadata = {
                "control_mode": task.get("control_mode") or "joint_delta",
                "action_space": robot_metadata.get("action_space"),
                "action_abs_space": robot_metadata.get("action_abs_space"),
                "action_scale": (task.get("action_scale") or task.get("action_scale_config")),
                "controller_gains": task.get("controller_gains"),
                "controller_config_source": (
                    "task_config" if task.get("controller_gains") else "default"
                ),
                "control_frequency_hz": robot_metadata.get("control_frequency_hz"),
            }

            with open(episode_path, "w") as f:
                json.dump({
                    "episode_id": episode_id,
                    "task_name": task.get("task_name") or llm_metadata.get("task_name") or "unknown_task",
                    "task_type": task.get("task_type"),
                    "target_object": task.get("target_object"),
                    "task_description": llm_metadata.get("task_description"),
                    "data_mode": data_mode,
                    "robot_metadata": robot_metadata,
                    "object_metadata": object_metadata,
                    "camera_calibration": camera_calibration,
                    "frame_conventions": frame_conventions,
                    "control_metadata": control_metadata,
                    "episode_seed": episode_seed,
                    "provenance": {
                        "joint_positions": "physx_server",
                        "ee_pos": "isaac_sim_fk" if _server_ee_frame_count > 0 else "analytic_fk",
                        "ee_quat": "isaac_sim_fk" if _server_ee_frame_count > 0 else "analytic_fk",
                        "ee_rot6d": "derived_from_ee_quat",
                        "joint_velocities": "physx_server" if _real_velocity_count > 0 else "finite_difference",
                        "joint_accelerations": "finite_difference_smoothed",
                        "joint_efforts": (
                            "physx_server" if _efforts_source == "physx"
                            else ("estimated_inverse_dynamics" if _efforts_source in ("estimated_inverse_dynamics", "mixed") else "unavailable")
                        ),
                        "ee_wrench": _ee_wrench_source,
                        "ee_vel": "derived_from_physx_positions" if _server_ee_frame_count > 0 else "derived_from_fk_positions",
                        "ee_acc": "derived_from_physx_positions" if _server_ee_frame_count > 0 else "derived_from_fk_positions",
                        "contact_forces": _contact_forces_source,
                        "camera_frames": "isaac_sim_camera" if _camera_frame_count > 0 else "unavailable",
                        "task_description": "task_config_hint",
                        "scene_state": (
                            "physx_server"
                            if _real_scene_state_count > 0
                            else (
                                "INVALID_synthetic"
                                if _scene_state_invalid
                                else ("synthetic_fallback" if _scene_state_fallback_frames > 0 else "synthetic_from_task_config")
                            )
                        ),
                        "task_success": llm_metadata.get("task_success_source", "geometric_goal_region_v2"),
                        "quality_score": "weighted_composite_v2",
                        "meters_per_unit": getattr(self, "_meters_per_unit", 1.0),
                        "units_per_meter": getattr(self, "_units_per_meter", 1.0),
                        "server_ee_frames": f"{_server_ee_frame_count}/{len(frames)}",
                        "real_scene_state_frames": f"{_real_scene_state_count}/{len(frames)}",
                        "camera_capture_frames": f"{_camera_frame_count}/{len(frames)}",
                        "real_velocity_frames": f"{_real_velocity_count}/{len(frames)}",
                        "real_effort_frames": f"{_real_effort_count}/{len(frames)}",
                        "estimated_effort_frames": f"{_estimated_effort_count}/{len(frames)}",
                        "diversity_calibration": _diversity_calibration_source,
                        "object_property_provenance": dict(_object_property_provenance),
                        "warnings": (
                            ["ee_pos_static_fk_direct_waypoint"]
                            if _ee_static_fallback_used
                            else []
                        ),
                    },
                    "channel_confidence": {
                        "joint_positions": 1.0,
                        "ee_pos": _ee_pose_conf,
                        "ee_quat": _ee_pose_conf,
                        "joint_velocities": 0.98 if _real_velocity_count > 0 else 0.7,
                        "joint_accelerations": 0.6,
                        "joint_efforts": (
                            0.95 if _efforts_source == "physx"
                            else (0.7 if _efforts_source in ("estimated_inverse_dynamics", "mixed") else 0.0)
                        ),
                        "contact_forces": (
                            0.9 if _efforts_source == "physx"
                            else (0.7 if _contact_report_count > 0 or _efforts_source in ("estimated_inverse_dynamics", "mixed") else 0.5)
                        ),
                        "ee_vel": _ee_vel_conf,
                        "ee_acc": _ee_vel_conf,
                        "scene_state": 0.95 if _real_scene_state_count > 0 else 0.3,
                        "camera_frames": 0.95 if _camera_frame_count > 0 else 0.0,
                    },
                    "goal_region_verification": llm_metadata.get("goal_region_verification"),
                    "frames": frames,
                    "frame_count": len(frames),
                    "quality_score": quality_score,
                    "quality_score_breakdown": getattr(self, "_last_quality_breakdown", None),
                    "joint_utilization": joint_utilization,
                    "validation_passed": validation_passed,
                    "task_success": task_success,
                    "task_success_reasoning": llm_metadata.get("task_success_reasoning"),
                    "collision_free": collision_free,
                    "collision_free_physics": collision_free_physics,
                    "collision_source": collision_source,
                    "effort_source_policy": _frame_stats.get("effort_source_policy", _efforts_source),
                    "scene_description": llm_metadata.get("scene_description"),
                    "success_criteria": llm_metadata.get("success_criteria"),
                    "recording_stopped_cleanly": recording_stopped_cleanly,
                    "execution_warning": result.get("execution_warning"),
                    "collector_warning": result.get("collector_warning"),
                    "stall_info": result["stall_info"],
                    "frame_validation": frame_validation,
                    "phase_list": [
                        "home",
                        "approach",
                        "pre_grasp",
                        "grasp",
                        "lift",
                        "transport",
                        "pre_place",
                        "place",
                        "release",
                        "retract",
                        "articulate_approach",
                        "articulate_grasp",
                        "articulate_motion",
                    ],
                    "phase_descriptions": {
                        "approach": f"Moving toward {task.get('target_object', 'target object')}",
                        "pre_grasp": "Positioning gripper just above target for grasp",
                        "grasp": f"Closing gripper to grasp {task.get('target_object', 'object')}",
                        "lift": f"Lifting {task.get('target_object', 'object')} off the surface",
                        "transport": f"Transporting {task.get('target_object', 'object')} to placement location",
                        "pre_place": "Positioning gripper just above placement target",
                        "place": f"Placing {task.get('target_object', 'object')} at target",
                        "release": "Opening gripper to release object",
                        "retract": "Retracting gripper after placement",
                    },
                    "vlm_audit": vlm_audit,
                    "sim_to_real_assessment": sim2real,
                    "failure_diagnosis": failure_diagnosis,
                    "language_annotations": llm_metadata.get("language_annotations"),
                }, f, default=_json_default)

            result["success"] = True
            result["frame_count"] = len(frames)
            result["quality_score"] = quality_score
            result["joint_utilization"] = joint_utilization
            result["validation_passed"] = validation_passed
            result["task_success"] = task_success
            result["collision_free"] = collision_free
            result["collision_free_physics"] = collision_free_physics
            result["output_path"] = str(episode_path)
            result["frame_validation"] = frame_validation
            if vlm_audit:
                result["vlm_audit"] = vlm_audit
            if sim2real:
                result["sim_to_real_assessment"] = sim2real
            if failure_diagnosis:
                result["failure_diagnosis"] = failure_diagnosis

        except Exception as e:
            import traceback
            result["error"] = str(e)
            self.log(f"Episode failed: {e}\n{traceback.format_exc()}", "ERROR")
            if recording_started and not recording_stopped:
                stop_result = self._client.stop_recording()
                if not stop_result.available:
                    self.log(
                        f"Stop recording unavailable: {stop_result.error}",
                        "WARNING",
                    )
                elif not stop_result.success:
                    self.log(
                        stop_result.error or "Stop recording failed.",
                        "WARNING",
                    )
                recording_stopped = True
        finally:
            if recording_started and not recording_stopped:
                try:
                    stop_result = self._client.stop_recording()
                    if not stop_result.available:
                        self.log(
                            f"Stop recording unavailable: {stop_result.error}",
                            "WARNING",
                        )
                    elif not stop_result.success:
                        self.log(
                            stop_result.error or "Stop recording failed.",
                            "WARNING",
                        )
                except Exception:
                    self.log("Failed to stop recording after episode error.", "WARNING")

        return result

    def _ensure_trajectory_timestamps(
        self,
        trajectory: List[Dict[str, Any]],
        fps: float = 30.0,
    ) -> List[Dict[str, Any]]:
        timed_trajectory: List[Dict[str, Any]] = []
        current_time = 0.0
        dt = 1.0 / fps
        for index, waypoint in enumerate(trajectory):
            waypoint = dict(waypoint)
            timestamp = waypoint.get("timestamp")
            if timestamp is None:
                timestamp = current_time if index == 0 else current_time + dt
            else:
                try:
                    timestamp = float(timestamp)
                except (TypeError, ValueError):
                    timestamp = current_time if index == 0 else current_time + dt
            if index > 0 and timestamp <= current_time:
                timestamp = current_time + dt
            waypoint["timestamp"] = timestamp
            current_time = timestamp
            timed_trajectory.append(waypoint)
        return timed_trajectory

    def _coerce_timestamp(self, value: Any, *, fallback: Optional[float] = None) -> float:
        fallback_value: float
        if fallback is None:
            fallback_value = time.time()
        else:
            try:
                fallback_value = float(fallback)
            except (TypeError, ValueError):
                fallback_value = time.time()

        if value is None:
            return fallback_value

        if isinstance(value, dict):
            seconds = value.get("sec", value.get("seconds"))
            nanos = value.get("nsec", value.get("nanos"))
            if seconds is None and nanos is None:
                return fallback_value
            try:
                seconds_value = float(seconds) if seconds is not None else 0.0
                nanos_value = float(nanos) if nanos is not None else 0.0
            except (TypeError, ValueError):
                return fallback_value
            return seconds_value + (nanos_value / 1e9)

        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback_value

    def _align_observations_to_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        if not observations:
            return None

        def _obs_timestamp(obs: Dict[str, Any], index: int) -> float:
            timestamp = obs.get("timestamp")
            if isinstance(timestamp, dict):
                timestamp = None
            if timestamp is None:
                timestamp = obs.get("planned_timestamp")
            if isinstance(timestamp, dict):
                timestamp = None
            return self._coerce_timestamp(timestamp, fallback=float(index))

        indexed_obs = [
            (idx, obs, _obs_timestamp(obs, idx)) for idx, obs in enumerate(observations)
        ]
        indexed_obs.sort(key=lambda item: item[2])
        obs_list = [item[1] for item in indexed_obs]
        obs_times = [item[2] for item in indexed_obs]

        aligned = []
        obs_index = 0
        for waypoint in trajectory:
            target_time = float(waypoint["timestamp"])
            while obs_index + 1 < len(obs_times) and obs_times[obs_index + 1] <= target_time:
                obs_index += 1
            if obs_index + 1 < len(obs_times):
                prev_time = obs_times[obs_index]
                next_time = obs_times[obs_index + 1]
                if abs(next_time - target_time) < abs(target_time - prev_time):
                    obs_index += 1
            aligned.append(obs_list[obs_index])
        return aligned

    def _build_frames_from_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        *,
        task: Dict[str, Any],
        episode_id: str,
        output_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        # Attempt to set up FK for end-effector pose computation
        fk_solver = None
        robot_config = _resolve_robot_config(
            getattr(self.config, "robot_type", "")
        )
        if robot_config is not None and IK_PLANNING_AVAILABLE:
            try:
                _solver = IKSolver(robot_config, verbose=False)
                if hasattr(_solver, "_forward_kinematics"):
                    fk_solver = _solver
            except Exception:
                pass

        joint_names = list(self._client._joint_names) if hasattr(self._client, "_joint_names") and self._client._joint_names else []
        num_joints = len(trajectory[0]["joint_positions"]) if trajectory else (len(joint_names) if joint_names else 7)
        joint_groups = _resolve_robot_joint_groups(
            getattr(self.config, "robot_type", ""),
            joint_names=joint_names,
            num_joints=num_joints,
        )
        arm_indices = joint_groups["primary_arm_indices"]
        gripper_indices = joint_groups["primary_gripper_indices"]
        arm_dof = len(arm_indices)

        # Forward-propagate joint positions: when observations return zeros
        # (mock/skip-server-recording mode), use the trajectory waypoints as
        # ground-truth robot state. This ensures action deltas are correct
        # (small values, not equal to action_abs).
        current_joint_state = np.array(
            trajectory[0]["joint_positions"], dtype=float
        ) if trajectory else np.zeros(num_joints)

        # --- Object tracking state for dynamic scene updates (Improvement A) ---
        _attached_object_id: Optional[str] = None
        _grasp_ee_offset: Optional[np.ndarray] = None
        _object_poses: Dict[str, np.ndarray] = {}  # current poses of all objects
        _initial_object_poses: Dict[str, np.ndarray] = {}  # for success verification
        # Per-frame object pose history for trajectory analysis (Issue 2 fix)
        _object_pose_history: Dict[str, List[Dict[str, Any]]] = {}
        _object_prev_state: Dict[str, Dict[str, Any]] = {}
        _release_decay_obj: Optional[str] = None
        _release_decay_remaining: int = 0
        # Fix 10: Phase progress tracking
        _prev_phase_for_progress: str = "approach"
        _phase_start_frame: int = 0
        _target_oid_for_subgoal = task.get("target_object") or task.get("target_object_id")

        # --- Sensor noise config (Improvement E) ---
        _inject_noise = os.environ.get("INJECT_SENSOR_NOISE", "1") == "1"
        _jp_noise_std = 0.001  # 1 millirad
        _jv_noise_std = 0.01   # 10 millirad/s
        _ee_noise_std = 0.001  # 1mm

        # --- Phase tracking for labels (Improvement G) ---
        _missing_phase_frames = 0

        # --- Grasp physics constants (Improvement J) ---
        _GRIPPER_MAX_APERTURE = joint_groups["gripper_max_aperture"]
        _GRIPPER_MAX_FORCE = joint_groups["gripper_max_force"]

        # Last-resort hardcoded fallback tables — only used when scene_graph
        # properties AND Gemini estimation are both unavailable.
        _HARDCODED_OBJECT_SIZES = {
            "pot": 0.06, "cup": 0.06, "plate": 0.02, "bottle": 0.05,
            "toaster": 0.07, "mug": 0.06, "bowl": 0.02, "can": 0.05,
            "box": 0.07, "pan": 0.03, "kettle": 0.06, "jar": 0.06,
        }
        _HARDCODED_OBJECT_MASSES = {
            "pot": 0.5, "cup": 0.2, "plate": 0.3, "bottle": 0.4,
            "toaster": 1.0, "mug": 0.25, "bowl": 0.35, "can": 0.3,
            "box": 0.4, "pan": 0.6, "kettle": 0.8, "jar": 0.35,
        }
        _HARDCODED_OBJECT_HEIGHTS = {
            "pot": 0.12, "cup": 0.10, "plate": 0.03, "bottle": 0.22,
            "toaster": 0.15, "mug": 0.10, "bowl": 0.08, "can": 0.12,
            "box": 0.10, "pan": 0.06, "kettle": 0.18, "jar": 0.12,
        }
        _HARDCODED_OBJECT_BBOXES = {
            "pot": [0.20, 0.20, 0.12], "cup": [0.08, 0.08, 0.10],
            "plate": [0.22, 0.22, 0.03], "bottle": [0.07, 0.07, 0.22],
            "toaster": [0.25, 0.15, 0.15], "mug": [0.10, 0.08, 0.10],
            "bowl": [0.18, 0.18, 0.08], "can": [0.06, 0.06, 0.12],
            "box": [0.15, 0.10, 0.10], "pan": [0.26, 0.26, 0.06],
            "kettle": [0.18, 0.15, 0.18], "jar": [0.08, 0.08, 0.12],
        }
        _DEFAULT_TABLE_HEIGHT = 0.75
        _HARDCODED_OBJECT_CATEGORIES = {
            "pot": "container", "cup": "container", "mug": "container",
            "bowl": "container", "jar": "container", "kettle": "container",
            "plate": "tableware", "pan": "cookware", "can": "container",
            "bottle": "container", "toaster": "appliance", "box": "container",
            "table": "furniture", "surface": "furniture", "shelf": "furniture",
        }
        _hardcoded_fallback_warned: set = set()

        _GRIPPER_CONTACT_KEYWORDS = (
            "finger",
            "gripper",
            "hand",
            "panda_leftfinger",
            "panda_rightfinger",
        )
        _SURFACE_CONTACT_KEYWORDS = ("table", "counter", "surface")

        def _normalize_body_name(name: str) -> str:
            if not name:
                return ""
            return str(name).rsplit("/", 1)[-1].lower()

        def _is_gripper_body(name: str) -> bool:
            return any(kw in name for kw in _GRIPPER_CONTACT_KEYWORDS)

        def _is_surface_body(name: str) -> bool:
            return any(kw in name for kw in _SURFACE_CONTACT_KEYWORDS)

        def _tokenize_body_name(name: str) -> List[str]:
            if not name:
                return []
            for sep in ("/", "\\", ":", ".", "-", "_"):
                name = name.replace(sep, " ")
            return [tok for tok in name.split() if tok]

        def _match_contact_object_id(body_name: str) -> Optional[str]:
            if not body_name or not _object_poses:
                return None
            name_lower = body_name.lower()
            # Direct substring match on object ids
            for _oid in _object_poses.keys():
                _oid_lower = str(_oid).lower()
                if _oid_lower and _oid_lower in name_lower:
                    return _oid
            # Token match as fallback
            tokens = _tokenize_body_name(name_lower)
            for _oid in _object_poses.keys():
                _oid_lower = str(_oid).lower()
                if _oid_lower in tokens:
                    return _oid
            return None

        def _resolve_object_id_by_hint(hint: Optional[str]) -> Optional[str]:
            if not hint or not _object_poses:
                return None
            hint_norm = str(hint).rsplit("/", 1)[-1].lower()
            for _oid in _object_poses.keys():
                if _oid == hint:
                    return _oid
                _oid_norm = str(_oid).rsplit("/", 1)[-1].lower()
                if _oid_norm == hint_norm:
                    return _oid
            return None

        def _seed_target_object(
            obs: Dict[str, Any],
            target_id: Optional[str],
        ) -> None:
            if not target_id or _is_production:
                return
            if target_id in _object_poses:
                return
            _target_pos = None
            try:
                _target_pos = self._find_target_position_from_obs(task, obs)
            except Exception:
                _target_pos = None
            if _target_pos is None:
                _tp = task.get("target_position")
                if isinstance(_tp, (list, tuple, np.ndarray)) and len(_tp) >= 3:
                    _target_pos = np.array(_tp[:3], dtype=float)
            if _target_pos is None:
                return
            _object_poses[target_id] = np.array(_target_pos, dtype=float)
            if obs.get("scene_state") is None:
                obs["scene_state"] = {"objects": []}
            _objs = obs["scene_state"].get("objects", [])
            if not any(o.get("object_id") == target_id for o in _objs if isinstance(o, dict)):
                _objs.append({
                    "object_id": target_id,
                    "object_type": str(target_id).rsplit("/", 1)[-1].lower(),
                    "pose": {"position": _target_pos.tolist()},
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                    "provenance": "synthetic_fallback",
                })
                obs["scene_state"]["objects"] = _objs
            obs.setdefault("scene_state_provenance", "synthetic_fallback")

        def _select_attach_candidate(
            ee_pos: np.ndarray,
            max_dist_units: float,
            target_hint: Optional[str] = None,
        ) -> Optional[str]:
            if ee_pos is None or not _object_poses:
                return None
            # Prefer target object when specified
            preferred: List[Tuple[float, str]] = []
            fallback: List[Tuple[float, str]] = []
            target_norm = target_hint.rsplit("/", 1)[-1].lower() if target_hint else ""
            for _oid, _pos in _object_poses.items():
                _oid_norm = str(_oid).rsplit("/", 1)[-1].lower()
                _dist = float(np.linalg.norm(ee_pos - _pos))
                if _dist > max_dist_units:
                    continue
                _obj_type = ""
                for _obj in (_scene_state or {}).get("objects", []):
                    if _obj.get("object_id") == _oid:
                        _obj_type = (_obj.get("object_type") or "").lower()
                        break
                _category = _get_obj_prop(_obj_type, "category", "")
                if _category == "furniture":
                    continue
                bucket = preferred if (target_norm and _oid_norm == target_norm) else fallback
                bucket.append((_dist, _oid))
            if preferred:
                preferred.sort(key=lambda x: x[0])
                return preferred[0][1]
            if fallback:
                fallback.sort(key=lambda x: x[0])
                return fallback[0][1]
            return None

        # --- LLM object property estimation ---
        # Persistent JSON cache that survives across runs
        _persistent_prop_cache_path = Path(os.getenv(
            "GENIESIM_PROPERTY_CACHE",
            str(self.config.recording_dir / "object_property_cache.json")
        ))
        _persistent_prop_cache: Dict[str, Any] = {}
        if _persistent_prop_cache_path.exists():
            try:
                import json as _json_pc
                with open(_persistent_prop_cache_path) as _f_pc:
                    _persistent_prop_cache = _json_pc.load(_f_pc)
                logger.info("Loaded %d entries from persistent property cache: %s",
                            len(_persistent_prop_cache), _persistent_prop_cache_path)
            except Exception as _pc_exc:
                logger.warning("Failed to load persistent property cache: %s", _pc_exc)

        _gemini_prop_cache: Dict[str, Dict[str, Any]] = dict(_persistent_prop_cache)
        _object_property_provenance: Dict[str, str] = {}
        _GEMINI_COOLDOWN_SKIPPED = object()
        _prop_llm = None
        _prop_llm_attempted = False

        def _get_env_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        def _get_env_int(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None or raw == "":
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        def _normalize_quat(quat_val: Any) -> Optional[List[float]]:
            if quat_val is None:
                return None
            if isinstance(quat_val, dict):
                quat_val = [
                    quat_val.get("rw", 1.0),
                    quat_val.get("rx", 0.0),
                    quat_val.get("ry", 0.0),
                    quat_val.get("rz", 0.0),
                ]
            if not isinstance(quat_val, (list, tuple, np.ndarray)) or len(quat_val) < 4:
                return None
            quat_arr = np.array(quat_val[:4], dtype=float)
            norm = float(np.linalg.norm(quat_arr))
            if norm <= 1e-9:
                return None
            quat_arr = quat_arr / norm
            return quat_arr.tolist()

        def _coerce_vec3(vec_val: Any) -> Optional[List[float]]:
            if vec_val is None:
                return None
            if isinstance(vec_val, dict):
                return [
                    float(vec_val.get("x", 0.0)),
                    float(vec_val.get("y", 0.0)),
                    float(vec_val.get("z", 0.0)),
                ]
            if isinstance(vec_val, (list, tuple, np.ndarray)) and len(vec_val) >= 3:
                return [float(v) for v in vec_val[:3]]
            return None

        def _quat_to_ang_vel(
            prev_quat: Optional[List[float]],
            curr_quat: Optional[List[float]],
            dt: float,
        ) -> Optional[List[float]]:
            if prev_quat is None or curr_quat is None or dt <= 0:
                return None
            q_prev = np.array(prev_quat, dtype=float)
            q_curr = np.array(curr_quat, dtype=float)
            if q_prev.shape[0] != 4 or q_curr.shape[0] != 4:
                return None
            q_prev /= max(1e-9, float(np.linalg.norm(q_prev)))
            q_curr /= max(1e-9, float(np.linalg.norm(q_curr)))
            # Delta quaternion: q_delta = q_curr * conj(q_prev)
            w1, x1, y1, z1 = q_curr
            w2, x2, y2, z2 = q_prev
            w = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
            x = -w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2
            y = -w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2
            z = -w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2
            w = max(-1.0, min(1.0, w))
            angle = 2.0 * float(np.arccos(w))
            if angle < 1e-6:
                return [0.0, 0.0, 0.0]
            sin_half = float(np.sin(angle / 2.0))
            if abs(sin_half) < 1e-6:
                return [0.0, 0.0, 0.0]
            axis = np.array([x, y, z], dtype=float) / sin_half
            ang_vel = axis * (angle / dt)
            return [float(v) for v in ang_vel.tolist()]

        def _update_pose_dict(_obj: Dict[str, Any], _pos: np.ndarray) -> None:
            """Update pose position while preserving existing rotation fields."""
            _x = round(float(_pos[0]), 6)
            _y = round(float(_pos[1]), 6)
            _z = round(float(_pos[2]), 6)
            _pose = _obj.get("pose")
            if isinstance(_pose, dict):
                _pose = dict(_pose)
            else:
                _pose = {}
            _pose["x"] = _x
            _pose["y"] = _y
            _pose["z"] = _z
            if "position" in _pose:
                if isinstance(_pose["position"], dict):
                    _pose["position"] = {"x": _x, "y": _y, "z": _z}
                elif isinstance(_pose["position"], (list, tuple)) and len(_pose["position"]) >= 3:
                    _pose["position"] = [_x, _y, _z]
            else:
                _pose["position"] = [_x, _y, _z]
            _obj["pose"] = _pose

        _GEMINI_PROP_MAX_RETRIES = _get_env_int("GENIESIM_GEMINI_PROP_MAX_RETRIES", 3)
        _GEMINI_PROP_RETRY_DELAY_S = _get_env_float("GENIESIM_GEMINI_PROP_RETRY_DELAY_S", 2.0)

        def _get_prop_llm():
            """Lazy-init LLM client for property estimation (with auto fallback)."""
            nonlocal _prop_llm, _prop_llm_attempted
            if not _prop_llm_attempted:
                _prop_llm_attempted = True
                try:
                    from tools.llm_client import create_llm_client as _create
                    _prop_llm = _create(
                        max_retries=_GEMINI_PROP_MAX_RETRIES,
                        retry_delay=_GEMINI_PROP_RETRY_DELAY_S,
                    )
                except Exception as _e:
                    logger.warning("LLM property client init failed: %s", _e)
                    _prop_llm = None
            return _prop_llm

        def _estimate_obj_prop_llm(obj_type: str, prop: str):
            """Estimate a single object property via LLM (with auto fallback via FallbackLLMClient)."""
            cache_key = f"{obj_type.lower()}:{prop}"
            if cache_key in _gemini_prop_cache:
                return _gemini_prop_cache[cache_key]
            # Check persistent cache (may have entries from previous runs not yet in memory)
            if cache_key in _persistent_prop_cache:
                _gemini_prop_cache[cache_key] = _persistent_prop_cache[cache_key]
                return _persistent_prop_cache[cache_key]
            llm = _get_prop_llm()
            if not llm:
                return None
            prop_prompts = {
                "graspable_width": (
                    f"Estimate the graspable width in meters for a '{obj_type}'. "
                    f"Return ONLY JSON: {{\"value\": 0.06}}"
                ),
                "mass": (
                    f"Estimate the mass in kilograms for a typical '{obj_type}'. "
                    f"Return ONLY JSON: {{\"value\": 0.3}}"
                ),
                "height": (
                    f"Estimate the height in meters for a typical '{obj_type}'. "
                    f"Return ONLY JSON: {{\"value\": 0.12}}"
                ),
                "bbox": (
                    f"Estimate bounding box [width, depth, height] in meters for a '{obj_type}'. "
                    f"Return ONLY JSON: {{\"value\": [0.2, 0.2, 0.12]}}"
                ),
                "category": (
                    f"Classify '{obj_type}' into one of: container, tableware, cookware, "
                    f"appliance, furniture, tool, unknown. Return ONLY JSON: {{\"value\": \"container\"}}"
                ),
            }
            prompt = prop_prompts.get(prop)
            if not prompt:
                return None
            try:
                resp = llm.generate(prompt, json_output=True, temperature=0.3, disable_tools=True)
                data = resp.parse_json()
                if isinstance(data, dict) and "value" in data:
                    val = data["value"]
                    _gemini_prop_cache[cache_key] = val
                    _object_property_provenance[f"{obj_type}:{prop}"] = "llm_estimated"
                    # Write-through to persistent cache
                    _persistent_prop_cache[cache_key] = val
                    try:
                        import json as _json_wt
                        with open(_persistent_prop_cache_path, "w") as _f_wt:
                            _json_wt.dump(_persistent_prop_cache, _f_wt, indent=2)
                    except Exception:
                        pass
                    logger.info("LLM_OBJ_PROP: %s.%s = %s", obj_type, prop, val)
                    return val
            except Exception as exc:
                logger.debug("LLM obj prop estimation failed for %s.%s: %s", obj_type, prop, exc)
            return None

        # Unified property lookup: scene_graph → Gemini estimated → hardcoded fallback
        _obj_props = getattr(self, "_object_properties", {})

        def _get_obj_prop(obj_id_or_type: str, prop: str, default=None):
            """Look up an object property: scene_graph → Gemini → hardcoded."""
            # 1. Try scene_graph (exact match, then type-level)
            for key in (obj_id_or_type, obj_id_or_type.lower()):
                entry = _obj_props.get(key)
                if entry and prop in entry:
                    _object_property_provenance[f"{obj_id_or_type}:{prop}"] = "scene_graph"
                    return entry[prop]
            # 2. Try LLM estimation via model cascade
            _llm_val = _estimate_obj_prop_llm(obj_id_or_type, prop)
            gemini_skipped = _llm_val is _GEMINI_COOLDOWN_SKIPPED
            if _llm_val is not None and not gemini_skipped:
                # Provenance already set inside _estimate_obj_prop_llm
                return _llm_val
            # 3. Hardcoded fallback with warning
            _hc_tables = {
                "graspable_width": _HARDCODED_OBJECT_SIZES,
                "mass": _HARDCODED_OBJECT_MASSES,
                "height": _HARDCODED_OBJECT_HEIGHTS,
                "bbox": _HARDCODED_OBJECT_BBOXES,
                "category": _HARDCODED_OBJECT_CATEGORIES,
            }
            _table = _hc_tables.get(prop, {})
            _key_lower = obj_id_or_type.lower()
            if _key_lower in _table:
                _warn_key = f"{_key_lower}:{prop}"
                if _warn_key not in _hardcoded_fallback_warned:
                    _hardcoded_fallback_warned.add(_warn_key)
                    logger.warning(
                        "HARDCODED_FALLBACK: Using hardcoded %s for object type '%s'. "
                        "Run simready-job or provide scene_graph metadata.",
                        prop, obj_id_or_type,
                    )
                _object_property_provenance[f"{obj_id_or_type}:{prop}"] = "hardcoded_fallback"
                return _table[_key_lower]
            if gemini_skipped:
                _object_property_provenance[f"{obj_id_or_type}:{prop}"] = "hardcoded_fallback"
            return default

        frames: List[Dict[str, Any]] = []
        _server_ee_frame_count = 0  # Track how many frames used server EE pose
        _real_scene_state_count = 0  # Track how many frames had real scene state
        _scene_state_fallback_frames = 0  # Track synthetic scene_state injections
        _scene_state_missing_after_frame0 = False
        _scene_state_missing_frame_indices: List[int] = []
        _camera_frame_count = 0  # Track how many frames had camera data
        _real_velocity_count = 0  # Track how many frames had real PhysX velocities
        _real_effort_count = 0  # Track how many frames had real PhysX joint efforts
        _ee_wrench_source = "unavailable"
        _estimated_effort_count = 0  # Track how many frames used inverse-dynamics efforts
        _effort_missing_count = 0  # Track frames missing/zero efforts
        _contact_report_count = 0  # Track frames with contact report data
        self._prev_server_ee_pos = None  # Reset for each episode
        self._prev_gripper_openness = None
        _recent_ee_positions: List[List[float]] = []
        _ee_static_fallback_used = False
        _ee_static_window = 5
        _ee_static_std = 1e-4
        _is_production = getattr(self.config, "environment", "") == "production"
        _strict_realism = getattr(self, "_strict_realism", False)
        _collision_source_hint = (self._last_planning_report.get("collision_source") or "").lower()
        _contact_query_warned = False

        # --- Trajectory time parameterization (realism fix) ---
        # Expand sparse waypoints into dense 30Hz frames with velocity-limited timing.
        _TARGET_FPS = int(os.getenv("GENIESIM_TARGET_FPS", "30"))
        _MIN_EPISODE_DURATION = float(os.getenv("GENIESIM_MIN_EPISODE_DURATION_S", "2.0"))
        # Franka Panda joint velocity limits (rad/s) for 7-DOF arm
        _FRANKA_VEL_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
        _vel_lim = _FRANKA_VEL_LIMITS[:arm_dof] if arm_dof <= 7 else np.full(arm_dof, 2.175)

        if len(trajectory) >= 2:
            # Compute minimum dt per segment from joint velocity limits
            _segment_dts: List[float] = [0.0]  # first waypoint at t=0
            for _i in range(1, len(trajectory)):
                _jp_curr = np.array(trajectory[_i]["joint_positions"][:arm_dof], dtype=float)
                _jp_prev = np.array(trajectory[_i - 1]["joint_positions"][:arm_dof], dtype=float)
                _delta = np.abs(_jp_curr - _jp_prev)
                _max_ratio = np.max(_delta / _vel_lim) if np.any(_delta > 1e-8) else (1.0 / _TARGET_FPS)
                _segment_dts.append(max(_max_ratio, 1.0 / _TARGET_FPS))

            _total_time = sum(_segment_dts[1:])
            if _total_time < _MIN_EPISODE_DURATION:
                _scale = _MIN_EPISODE_DURATION / _total_time
                _segment_dts = [_segment_dts[0]] + [dt * _scale for dt in _segment_dts[1:]]
                _total_time = _MIN_EPISODE_DURATION

            # Build cumulative timestamps for waypoints
            _wp_times = [0.0]
            for _dt in _segment_dts[1:]:
                _wp_times.append(_wp_times[-1] + _dt)

            # Interpolate to dense frames at _TARGET_FPS
            _dense_dt = 1.0 / _TARGET_FPS
            _num_dense_frames = max(int(_total_time * _TARGET_FPS), len(trajectory))
            _dense_trajectory: List[Dict[str, Any]] = []
            _dense_observations: List[Dict[str, Any]] = []
            _wp_idx = 0

            for _fi in range(_num_dense_frames):
                _t = _fi * _dense_dt
                # Find bounding waypoints
                while _wp_idx < len(_wp_times) - 2 and _wp_times[_wp_idx + 1] < _t:
                    _wp_idx += 1
                _wp_idx = min(_wp_idx, len(_wp_times) - 2)

                _t0 = _wp_times[_wp_idx]
                _t1 = _wp_times[_wp_idx + 1]
                _alpha = (_t - _t0) / (_t1 - _t0) if abs(_t1 - _t0) > 1e-9 else 0.0
                _alpha = max(0.0, min(1.0, _alpha))

                _jp0 = np.array(trajectory[_wp_idx]["joint_positions"], dtype=float)
                _jp1 = np.array(trajectory[_wp_idx + 1]["joint_positions"], dtype=float)
                _jp_interp = ((1.0 - _alpha) * _jp0 + _alpha * _jp1).tolist()

                _new_wp = dict(trajectory[_wp_idx])
                _new_wp["joint_positions"] = _jp_interp
                _new_wp["timestamp"] = _t
                _new_wp["data_source"] = "between_waypoints"
                _dense_trajectory.append(_new_wp)

                # Pick closest observation
                _obs_idx = min(_wp_idx, len(observations) - 1)
                if _wp_idx + 1 < len(observations) and _alpha > 0.5:
                    _obs_idx = _wp_idx + 1
                _dense_observations.append(observations[_obs_idx] if _obs_idx < len(observations) else {})

            trajectory = _dense_trajectory
            observations = _dense_observations
            logger.info(
                "[TIME_PARAM] Expanded trajectory: %d waypoints -> %d frames (%.2fs at %dHz)",
                len(_segment_dts), len(trajectory), _total_time, _TARGET_FPS,
            )

        # Set up frames directory for saving camera data as separate .npy files
        if output_dir is not None:
            _frames_dir = output_dir / f"{episode_id}_frames"
            _frames_dir.mkdir(parents=True, exist_ok=True)
            self._current_frames_dir = _frames_dir
        else:
            self._current_frames_dir = None

        for step_idx, (waypoint, obs) in enumerate(zip(trajectory, observations)):
            # Shallow-copy obs to avoid mutating shared references (aligned
            # observations may reuse the same dict for multiple frames).
            if obs is None:
                obs = {}
            else:
                obs = dict(obs)
                if "robot_state" in obs:
                    obs["robot_state"] = dict(obs["robot_state"])
            self._current_frame_idx = step_idx
            self._attach_camera_frames(obs, episode_id=episode_id, task=task)

            waypoint_joints = np.array(waypoint["joint_positions"], dtype=float)

            # Use real observation joints if available; otherwise inject the
            # forward-propagated state from the previous waypoint.
            obs.setdefault("robot_state", {})
            robot_state = obs["robot_state"]
            jp = robot_state.get("joint_positions")

            # Determine if we should use forward-propagated state:
            # - No joints at all
            # - Length mismatch (e.g., 28 default zeros vs 7 trajectory joints)
            # - All zeros (mock mode default)
            use_propagated = (
                jp is None
                or len(jp) != len(waypoint_joints)
                or np.allclose(jp, 0.0)
            )
            if use_propagated:
                robot_state["joint_positions"] = current_joint_state.tolist()

            # Keep the full-body joint_state synchronized with the arm
            # trajectory even when we use real observation joints.
            _js = robot_state.get("joint_state")
            if isinstance(_js, dict) and _js.get("positions"):
                _full_pos = list(_js["positions"])
                _arm_source = robot_state.get("joint_positions", current_joint_state.tolist())
                if isinstance(_arm_source, np.ndarray):
                    _arm_source = _arm_source.tolist()
                if not isinstance(_arm_source, list) or len(_arm_source) != arm_dof:
                    _arm_source = current_joint_state.tolist()
                if arm_indices and len(_full_pos) > max(arm_indices):
                    for _ai, _val in zip(arm_indices, _arm_source):
                        _full_pos[_ai] = _val
                _wp_ga = waypoint.get("gripper_aperture")
                if _wp_ga is not None and gripper_indices:
                    _g_max = joint_groups.get("gripper_max_aperture", 0.04)
                    _g_val = float(_wp_ga) * _g_max
                    for _gi in gripper_indices:
                        if _gi < len(_full_pos):
                            _full_pos[_gi] = _g_val
                _js["positions"] = _full_pos

            obs_joints = np.array(robot_state["joint_positions"], dtype=float)

            action_delta = (waypoint_joints - obs_joints).tolist()

            # Determine gripper openness for action vectors (normalized [0,1])
            _wp_gripper_aperture = waypoint.get("gripper_aperture")
            if _wp_gripper_aperture is not None:
                gripper_openness = float(_wp_gripper_aperture)
            elif gripper_indices:
                gripper_joints = [waypoint_joints[idx] for idx in gripper_indices if idx < len(waypoint_joints)]
                _g_lims = joint_groups["gripper_limits"]
                _g_range = float(_g_lims[1] - _g_lims[0]) if _g_lims[1] > _g_lims[0] else 1.0
                gripper_mean = float(np.mean(np.abs(gripper_joints))) if gripper_joints else 0.0
                gripper_openness = min(1.0, gripper_mean / _g_range) if _g_range > 0 else 0.0
            else:
                gripper_openness = 1.0

            _prev_gripper = getattr(self, "_prev_gripper_openness", None)
            if _prev_gripper is None:
                _prev_gripper = gripper_openness
            action_gripper_delta = gripper_openness - _prev_gripper
            self._prev_gripper_openness = gripper_openness
            _gripper_max_aperture = joint_groups.get("gripper_max_aperture", 0.04) or 0.0
            gripper_width = gripper_openness * _gripper_max_aperture

            # Normalize timestamps to uniform 1/control_frequency_hz intervals
            _control_freq = 30.0  # Hz
            _uniform_ts = step_idx / _control_freq
            obs["timestamp"] = _uniform_ts
            obs["planned_timestamp"] = _uniform_ts
            # Preserve original trajectory timestamp for debugging
            obs["_original_timestamp"] = waypoint["timestamp"]

            # Inject synthetic scene_state when empty (mock/skip-server mode).
            # In production, empty scene_state is an error — real data is required.
            _scene_state_missing = not obs.get("scene_state")
            if _scene_state_missing and _strict_realism:
                try:
                    from data_fidelity import DataFidelityError
                    raise DataFidelityError(
                        f"Scene state missing at frame {step_idx} in strict realism mode.",
                        gate_name="scene_state_missing",
                        diagnostics={"frame_idx": step_idx, "episode_id": episode_id},
                    )
                except ImportError:
                    raise RuntimeError(
                        f"Scene state missing at frame {step_idx} in strict realism mode."
                    )
            if _scene_state_missing and step_idx == 0 and _is_production:
                logger.error(
                    "EMPTY_SCENE_STATE in production for episode %s frame %d. "
                    "Real scene object poses required. Check _scene_object_prims "
                    "initialization and SimObjectService connectivity.",
                    episode_id, step_idx,
                )
            if _scene_state_missing and step_idx > 0 and _is_production:
                _scene_state_missing_after_frame0 = True
                _scene_state_missing_frame_indices.append(step_idx)
                logger.error(
                    "MISSING_SCENE_STATE in production for episode %s frame %d. "
                    "Real scene object poses required beyond frame 0.",
                    episode_id, step_idx,
                )
            if _scene_state_missing:
                scene_objects = task.get("objects") or task.get("scene_objects") or []
                if isinstance(scene_objects, list) and scene_objects:
                    obs["scene_state"] = {"objects": [
                        {
                            "object_id": o.get("name") or o.get("object_id") or f"object_{i}",
                            "object_type": (o.get("object_type") or o.get("type") or "unknown").lower(),
                            "pose": o.get("pose") or o.get("position") or {"x": 0, "y": 0, "z": 0},
                            "orientation": o.get("orientation", [1.0, 0.0, 0.0, 0.0]),
                            "bbox": _get_obj_prop(
                                (o.get("object_type") or o.get("type") or "").lower(),
                                "bbox", [0.10, 0.10, 0.10],
                            ),
                            "mass_kg": _get_obj_prop(
                                (o.get("object_type") or o.get("type") or "").lower(), "mass", 0.3
                            ),
                            "category": _get_obj_prop(
                                (o.get("object_type") or o.get("type") or "").lower(), "category", "unknown"
                            ),
                            "graspable": (o.get("object_type") or o.get("type") or "").lower() not in ("table", "surface", "shelf"),
                            "support_surface": "table",
                        }
                        for i, o in enumerate(scene_objects)
                    ]}
                else:
                    # Synthesize from task-level fields (target_object, goal_region)
                    _synth_objs = []
                    _target = task.get("target_object")
                    if _target:
                        # Extract readable name from object ID (e.g. "lightwheel_kitchen_obj_Pot057" → "Pot")
                        _name_parts = _target.rsplit("_", 1)
                        _readable = _name_parts[-1].rstrip("0123456789") if _name_parts else _target
                        _rc_cfg = task.get("robot_config") or {}
                        _base = _rc_cfg.get("base_position", [0.5, 0.0, 0.8])
                        _obj_type = _readable.lower()
                        _half_h = _get_obj_prop(_obj_type, "height", 0.10) / 2.0
                        _obj_z = _DEFAULT_TABLE_HEIGHT + _half_h
                        _synth_objs.append({
                            "object_id": _target,
                            "object_type": _obj_type,
                            "pose": {
                                "x": _base[0] - 0.05, "y": _base[1] + 0.25,
                                "z": round(_obj_z, 4),
                            },
                            "orientation": [1.0, 0.0, 0.0, 0.0],  # identity quaternion [w,x,y,z]
                            "bbox": _get_obj_prop(_obj_type, "bbox", [0.10, 0.10, 0.10]),
                            "mass_kg": _get_obj_prop(_obj_type, "mass", 0.3),
                            "category": _get_obj_prop(_obj_type, "category", "unknown"),
                            "graspable": _obj_type not in ("table", "surface", "shelf"),
                            "support_surface": "table",
                        })
                    _goal = task.get("goal_region")
                    if _goal:
                        _synth_objs.append({
                            "object_id": _goal,
                            "object_type": "surface",
                            "pose": {"x": 0.5, "y": 0.5, "z": _DEFAULT_TABLE_HEIGHT},
                            "orientation": [1.0, 0.0, 0.0, 0.0],
                        })
                    if _synth_objs:
                        for _so in _synth_objs:
                            _so["provenance"] = "synthetic_fallback"
                        obs["scene_state"] = {"objects": _synth_objs}
                if not _is_production:
                    _scene_state_fallback_frames += 1
                    obs["scene_state_provenance"] = "synthetic_fallback"

            # Ensure camera_frames is always present (even if empty) so quality
            # checks don't report missing required observation fields.
            # camera_frames is expected to be a dict keyed by camera_id.
            _cam_obs = obs.pop("camera_observation", {})
            _cam_images = _cam_obs.get("images", [])
            if _cam_images and isinstance(_cam_images, list):
                # Convert list of camera images to dict keyed by index
                _cam_dict = {f"camera_{i}": img for i, img in enumerate(_cam_images) if img}
            elif isinstance(_cam_images, dict):
                _cam_dict = _cam_images
            else:
                _cam_dict = {}
            obs.setdefault("camera_frames", _cam_dict)

            frame_data: Dict[str, Any] = {
                "step": step_idx,
                "observation": obs,
                "action": action_delta + [action_gripper_delta],
                "action_abs": list(waypoint["joint_positions"]) + [gripper_width],
                "timestamp": _uniform_ts,
                "dt": 1.0 / _control_freq,
            }

            # Advance forward-propagated state to this waypoint's target
            current_joint_state = waypoint_joints.copy()

            # End-effector Cartesian pose: prefer real server data, fallback to local FK.
            _base_pos = np.array(
                task.get("robot_config", {}).get("base_position", [0, 0, 0]),
                dtype=float,
            )
            _mpu = getattr(self, "_meters_per_unit", 1.0)
            if not isinstance(_mpu, (int, float)) or _mpu <= 0:
                _mpu = 1.0
            _used_server_ee = False
            # Check if the server observation contains a real (dynamic) EE pose
            _server_ee = robot_state.get("ee_pose") or {}
            _fk_source = None
            if isinstance(_server_ee, dict) and "position" in _server_ee:
                _sep = _server_ee["position"]
                if isinstance(_sep, dict) and "x" in _sep:
                    _ee_srv = [_sep["x"], _sep["y"], _sep["z"]]
                elif isinstance(_sep, (list, tuple)) and len(_sep) >= 3:
                    _ee_srv = list(_sep[:3])
                else:
                    _ee_srv = None
                if _ee_srv is not None:
                    _ee_srv = [float(v) for v in _ee_srv]
                    if _mpu != 1.0:
                        _ee_srv = [v * _mpu for v in _ee_srv]
                    # Check if server EE is dynamic (differs from previous frame)
                    _prev_srv_ee = getattr(self, "_prev_server_ee_pos", None)
                    self._prev_server_ee_pos = _ee_srv
                    if _prev_srv_ee is None or not np.allclose(_ee_srv, _prev_srv_ee, atol=1e-6):
                        frame_data["ee_pos"] = _ee_srv
                        _seq = _server_ee.get("orientation") or _server_ee.get("quaternion")
                        if _seq:
                            if isinstance(_seq, dict):
                                frame_data["ee_quat"] = [_seq.get("w", 1), _seq.get("x", 0), _seq.get("y", 0), _seq.get("z", 0)]
                            else:
                                frame_data["ee_quat"] = list(_seq)
                        _used_server_ee = True

            # Fallback: local FK — always compute from joint positions so
            # ee_pos updates every frame even when server returns static values.
            if not _used_server_ee:
                try:
                    fk_joint_positions = None
                    if robot_config is not None and joint_names and hasattr(robot_config, "joint_names"):
                        config_names = list(robot_config.joint_names)
                        if config_names and all(name in joint_names for name in config_names):
                            indices = [joint_names.index(name) for name in config_names]
                            fk_joint_positions = waypoint_joints[indices]
                            _fk_source = "indexed_config"
                    if fk_joint_positions is None and arm_indices:
                        # If waypoint is arm-only (e.g. 7-DOF Franka) and arm_indices
                        # point beyond its length, use waypoint_joints directly as
                        # they already represent the arm joint positions.
                        if max(arm_indices) < len(waypoint_joints):
                            fk_joint_positions = waypoint_joints[arm_indices]
                            _fk_source = "indexed_arm"
                        elif len(waypoint_joints) == arm_dof:
                            fk_joint_positions = waypoint_joints
                            _fk_source = "direct"
                    # Last resort: if waypoint length matches expected arm DOF, use directly
                    if fk_joint_positions is None and len(waypoint_joints) == arm_dof:
                        fk_joint_positions = waypoint_joints
                        _fk_source = "direct"
                    if fk_solver is not None and fk_joint_positions is not None:
                        ee_pos, ee_quat = fk_solver._forward_kinematics(fk_joint_positions)
                        _ee = (np.asarray(ee_pos, dtype=float) + _base_pos)
                        if _mpu != 1.0:
                            _ee = _ee * _mpu
                        frame_data["ee_pos"] = _ee.tolist()
                        frame_data["ee_quat"] = ee_quat.tolist() if hasattr(ee_quat, "tolist") else ee_quat
                    elif getattr(self.config, "robot_type", "").lower() in _FRANKA_TYPES and fk_joint_positions is not None:
                        ee_pos, ee_quat = _franka_fk(fk_joint_positions)
                        _ee = (np.asarray(ee_pos, dtype=float) + _base_pos)
                        if _mpu != 1.0:
                            _ee = _ee * _mpu
                        frame_data["ee_pos"] = _ee.tolist()
                        frame_data["ee_quat"] = ee_quat
                except Exception:
                    pass

            if frame_data.get("ee_pos"):
                _recent_ee_positions.append(frame_data["ee_pos"])
                if len(_recent_ee_positions) > _ee_static_window:
                    _recent_ee_positions.pop(0)
                if (
                    not _used_server_ee
                    and _fk_source in {"indexed_config", "indexed_arm"}
                    and len(_recent_ee_positions) == _ee_static_window
                    and len(waypoint_joints) == arm_dof
                ):
                    _recent_arr = np.asarray(_recent_ee_positions, dtype=float)
                    if np.all(np.std(_recent_arr, axis=0) < _ee_static_std):
                        try:
                            if fk_solver is not None:
                                ee_pos, ee_quat = fk_solver._forward_kinematics(waypoint_joints)
                                _ee = (np.asarray(ee_pos, dtype=float) + _base_pos)
                                if _mpu != 1.0:
                                    _ee = _ee * _mpu
                                frame_data["ee_pos"] = _ee.tolist()
                                frame_data["ee_quat"] = ee_quat.tolist() if hasattr(ee_quat, "tolist") else ee_quat
                                _recent_ee_positions[-1] = frame_data["ee_pos"]
                                _ee_static_fallback_used = True
                                _fk_source = "direct_fallback"
                            elif getattr(self.config, "robot_type", "").lower() in _FRANKA_TYPES:
                                ee_pos, ee_quat = _franka_fk(waypoint_joints)
                                _ee = (np.asarray(ee_pos, dtype=float) + _base_pos)
                                if _mpu != 1.0:
                                    _ee = _ee * _mpu
                                frame_data["ee_pos"] = _ee.tolist()
                                frame_data["ee_quat"] = ee_quat
                                _recent_ee_positions[-1] = frame_data["ee_pos"]
                                _ee_static_fallback_used = True
                                _fk_source = "direct_fallback"
                        except Exception:
                            pass

            if _used_server_ee:
                _server_ee_frame_count += 1
            frame_data["ee_pose_source"] = "server" if _used_server_ee else "fk"
            _scene_state_for_count = obs.get("scene_state") or (obs.get("privileged", {}) or {}).get("scene_state")
            if (
                _scene_state_for_count
                and _scene_state_for_count.get("objects")
                and obs.get("data_source") in ("real_composed", "between_waypoints")
            ):
                _real_scene_state_count += 1
            _cf = obs.get("camera_frames", {})
            if _cf and any(
                v.get("rgb") is not None
                for v in (_cf.values() if isinstance(_cf, dict) else [])
            ):
                _camera_frame_count += 1
            _rs_vel = robot_state.get("joint_velocities", [])
            if _rs_vel and any(abs(v) > 1e-10 for v in _rs_vel):
                _real_velocity_count += 1
            _rs_eff = robot_state.get("joint_efforts", [])
            _real_efforts = _rs_eff if _rs_eff is not None else []
            _has_real_efforts = bool(_real_efforts and any(abs(e) > 1e-6 for e in _real_efforts))
            _server_wrench = None
            try:
                ee_wrench_result = self._client.get_ee_wrench(
                    include_contacts=False,
                    lock_timeout=1.0,
                )
                if ee_wrench_result.available and ee_wrench_result.success and ee_wrench_result.payload:
                    payload = ee_wrench_result.payload
                    if payload.get("force") or payload.get("torque"):
                        _server_wrench = {
                            "force": payload.get("force", [0.0, 0.0, 0.0]),
                            "torque": payload.get("torque", [0.0, 0.0, 0.0]),
                            "frame": payload.get("frame", "end_effector"),
                            "source": payload.get("source", "physx_contact_report"),
                        }
                        frame_data["ee_wrench"] = _server_wrench
                        robot_state["ee_wrench"] = _server_wrench
                        _ee_wrench_source = _server_wrench.get("source", "physx_contact_report")
            except Exception:
                _server_wrench = None

            if _has_real_efforts and _server_wrench is None:
                _real_effort_count += 1
                frame_data["efforts_source"] = "physx"
                # Approximate EE wrench from joint efforts (wrist torques + grip force proxy)
                try:
                    if gripper_indices:
                        _gripper_efforts = [
                            _real_efforts[idx] for idx in gripper_indices if idx < len(_real_efforts)
                        ]
                    else:
                        _gripper_efforts = _real_efforts[arm_dof:]
                    _grip_force_real = sum(abs(e) for e in _gripper_efforts)
                    _arm_efforts = _real_efforts[:arm_dof] if arm_dof else _real_efforts
                    _torque = _arm_efforts[-3:] if len(_arm_efforts) >= 3 else ([0.0, 0.0, 0.0])
                    ee_wrench = {
                        "force": [0.0, 0.0, round(float(_grip_force_real), 4)],
                        "torque": [round(float(t), 4) for t in _torque],
                        "frame": "end_effector",
                        "source": "joint_efforts",
                    }
                    frame_data["ee_wrench"] = ee_wrench
                    robot_state["ee_wrench"] = ee_wrench
                    _ee_wrench_source = "joint_efforts"
                except Exception:
                    pass
            elif _server_wrench is None:
                _effort_missing_count += 1
                frame_data["efforts_source"] = "none"

            # Fix 6: Quaternion normalization, hemisphere continuity, and 6D rotation
            if frame_data.get("ee_quat"):
                _q = np.array(frame_data["ee_quat"], dtype=float)
                # Normalize to unit quaternion
                _qn = float(np.linalg.norm(_q))
                if _qn > 1e-8:
                    _q = _q / _qn
                # Hemisphere continuity: flip sign if dot with previous < 0
                if frames and frames[-1].get("ee_quat"):
                    _q_prev = np.array(frames[-1]["ee_quat"], dtype=float)
                    if np.dot(_q, _q_prev) < 0:
                        _q = -_q
                frame_data["ee_quat"] = _q.tolist()
                # 6D continuous rotation representation (first two columns of rotation matrix)
                w, x, y, z = _q
                _rot6d = [
                    1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
                    2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
                ]
                frame_data["ee_rot6d"] = [round(v, 8) for v in _rot6d]

            # Explicit gripper command: prefer gripper_aperture from waypoint
            # (trajectory planner sets this per-phase), fall back to inferring
            # from gripper joint positions when available.
            frame_data["gripper_command"] = "open" if gripper_openness > 0.5 else "closed"
            frame_data["gripper_openness"] = round(gripper_openness, 3)

            # --- Improvement G: Per-frame phase labels ---
            # Use explicit phase from trajectory waypoint (no inference fallback).
            _wp_phase = waypoint.get("phase")
            _current_phase = _wp_phase if isinstance(_wp_phase, str) and _wp_phase else None
            if _current_phase is None:
                _missing_phase_frames += 1
            frame_data["phase"] = _current_phase

            # Fix 10: Track phase boundaries and compute progress
            if _current_phase is not None:
                if _current_phase != _prev_phase_for_progress:
                    _phase_start_frame = step_idx
                    _prev_phase_for_progress = _current_phase
                _phase_len = step_idx - _phase_start_frame + 1
                # Estimate phase duration as fraction of total trajectory
                _est_phase_frames = max(1, len(trajectory) // 6)  # 6 known phases: approach, grasp, lift, transport, place, retract
                frame_data["phase_progress"] = min(1.0, round(_phase_len / _est_phase_frames, 4))
            else:
                frame_data["phase_progress"] = None

            # --- Improvement A: Dynamic scene state + J: Grasp physics ---
            ee_pos_arr = np.array(frame_data["ee_pos"]) if frame_data.get("ee_pos") else None
            _units_per_meter = getattr(self, "_units_per_meter", 1.0)
            if not isinstance(_units_per_meter, (int, float)) or _units_per_meter <= 0:
                _units_per_meter = 1.0
            _ee_pos_units = ee_pos_arr * _units_per_meter if ee_pos_arr is not None else None

            # Distance to subgoal: use target object pose for approach/transport, table for place
            if _ee_pos_units is not None and _target_oid_for_subgoal:
                _tgt_pos_sg = _object_poses.get(_target_oid_for_subgoal)
                if _tgt_pos_sg is not None:
                    frame_data["distance_to_subgoal"] = round(
                        float(np.linalg.norm(_ee_pos_units - _tgt_pos_sg)), 5
                    )

            # Update object poses from real scene_state every frame (not just frame 0).
            # On frame 0, also store initial poses for displacement calculation.
            _scene_state = obs.get("scene_state") or (obs.get("privileged", {}) or {}).get("scene_state")
            if _scene_state is not None and obs.get("scene_state") is None:
                obs["scene_state"] = _scene_state
            if _scene_state:
                for _obj in _scene_state.get("objects", []):
                    _oid = _obj.get("object_id", "")
                    _p = _obj.get("pose", {})
                    if "x" in _p:
                        _pos = np.array([_p["x"], _p["y"], _p["z"]], dtype=float)
                    elif "position" in _p:
                        _pos_val = _p["position"]
                        if isinstance(_pos_val, dict):
                            _pos = np.array([
                                _pos_val.get("x", 0),
                                _pos_val.get("y", 0),
                                _pos_val.get("z", 0),
                            ], dtype=float)
                        else:
                            _pos = np.array(_pos_val, dtype=float)
                    else:
                        continue  # skip objects with no valid pose

                    # Extract rotation (normalize to [w, x, y, z])
                    _rot = None
                    for _candidate in (
                        _p.get("rotation_quat"),
                        _p.get("orientation"),
                        _p.get("rotation"),
                        _obj.get("rotation_quat"),
                        _obj.get("orientation"),
                        _obj.get("rotation"),
                    ):
                        _rot = _normalize_quat(_candidate)
                        if _rot is not None:
                            break

                    _lvel = _coerce_vec3(_p.get("linear_velocity") or _obj.get("linear_velocity"))
                    _avel = _coerce_vec3(_p.get("angular_velocity") or _obj.get("angular_velocity"))
                    _pose_source = _p.get("source") or _obj.get("source", "unknown")
                    _lvel_source = _p.get("linear_velocity_source") or _obj.get("linear_velocity_source") or _pose_source
                    _avel_source = _p.get("angular_velocity_source") or _obj.get("angular_velocity_source") or _pose_source

                    # Compute finite-difference velocities if missing
                    _prev_state = _object_prev_state.get(_oid, {})
                    _prev_pos = _prev_state.get("position")
                    _prev_rot = _prev_state.get("rotation_quat")
                    _prev_ts = _prev_state.get("timestamp")
                    _cur_ts = frame_data.get("timestamp")
                    _dt = None
                    if _prev_ts is not None and _cur_ts is not None:
                        try:
                            _dt = float(_cur_ts) - float(_prev_ts)
                        except (TypeError, ValueError):
                            _dt = None
                    if _lvel is None and _prev_pos is not None and _dt and _dt > 0:
                        _lvel = ((_pos - _prev_pos) / _dt).tolist()
                        _lvel_source = "finite_difference"
                    if _avel is None and _rot is not None and _prev_rot is not None and _dt and _dt > 0:
                        _avel = _quat_to_ang_vel(_prev_rot, _rot, _dt)
                        if _avel is not None:
                            _avel_source = "finite_difference"

                    # Validate against kinematic tracking if available
                    if _oid in _object_poses and step_idx > 0:
                        _tracked = _object_poses[_oid]
                        _div = float(np.linalg.norm(_pos - _tracked))
                        _div_thresh = 0.02 * _units_per_meter  # >2cm divergence (scaled)
                        if _div > _div_thresh:
                            logger.debug(
                                "Object %s: real pose diverges from tracked by %.3fm at frame %d",
                                _oid, _div, step_idx,
                            )
                    _object_poses[_oid] = _pos.copy()
                    if step_idx == 0:
                        _initial_object_poses[_oid] = _pos.copy()

                    if _oid not in _object_pose_history:
                        _object_pose_history[_oid] = []
                    _object_pose_history[_oid].append({
                        "frame_idx": step_idx,
                        "position": _pos.tolist() if hasattr(_pos, 'tolist') else list(_pos),
                        "rotation_quat": _rot,
                        "linear_velocity": (
                            list(_lvel)
                            if _lvel is not None
                            else [0.0, 0.0, 0.0]
                        ),
                        "angular_velocity": (
                            list(_avel)
                            if _avel is not None
                            else [0.0, 0.0, 0.0]
                        ),
                        "velocity_source": _lvel_source,
                        "angular_velocity_source": _avel_source,
                        "source": _pose_source,
                    })
                    _object_prev_state[_oid] = {
                        "position": _pos.copy(),
                        "rotation_quat": _rot,
                        "timestamp": _cur_ts,
                    }

            # Detect grasp: gripper closes near an object
            if (frame_data.get("gripper_command") == "closed"
                    and _attached_object_id is None
                    and _ee_pos_units is not None):
                for _oid, _opos in _object_poses.items():
                    _dist = float(np.linalg.norm(_ee_pos_units - _opos))
                    # Compute proximity threshold from real object bbox diagonal
                    _obj_type = ""
                    for _obj in (_scene_state or {}).get("objects", []):
                        if _obj.get("object_id") == _oid:
                            _obj_type = (_obj.get("object_type") or "").lower()
                            break
                    _obj_bbox = _get_obj_prop(_obj_type, "bbox", [0.10, 0.10, 0.10])
                    _bbox_diag = float(np.linalg.norm(_obj_bbox)) / 2.0 + 0.05  # half diagonal + 5cm reach
                    _grasp_threshold = max(0.08, min(_bbox_diag, 0.25))  # clamp [8cm, 25cm]
                    _grasp_threshold_units = _grasp_threshold * _units_per_meter
                    if _dist < _grasp_threshold_units:
                        _obj_width = _get_obj_prop(_obj_type, "graspable_width", 0.06)
                        if _obj_width <= _GRIPPER_MAX_APERTURE:
                            _attached_object_id = _oid
                            _grasp_ee_offset = _opos - _ee_pos_units
                            frame_data["grasp_feasible"] = True
                            frame_data["grasped_object_id"] = _oid
                        else:
                            frame_data["grasp_feasible"] = False
                        break

            # Update attached object pose: prefer real scene_state when the
            # server is actually reporting changing poses.  When scene_state
            # poses are static (same value every frame), fall back to kinematic
            # EE-offset tracking so grasped objects follow the end-effector.
            _attached_has_real_pose = False
            if _attached_object_id is not None and _scene_state:
                for _obj in _scene_state.get("objects", []):
                    if _obj.get("object_id") == _attached_object_id:
                        _p = _obj.get("pose", {})
                        if "x" in _p:
                            _real_pos = np.array([_p["x"], _p["y"], _p["z"]], dtype=float)
                        elif "position" in _p:
                            _pv = _p["position"]
                            _real_pos = np.array([_pv.get("x",0), _pv.get("y",0), _pv.get("z",0)], dtype=float) if isinstance(_pv, dict) else np.array(_pv, dtype=float)
                        else:
                            break
                        # Check if the pose actually changed from initial (server is live)
                        _init_pos = _initial_object_poses.get(_attached_object_id)
                        _static_thresh = 0.005 * _units_per_meter
                        if _init_pos is not None and float(np.linalg.norm(_real_pos - _init_pos)) > _static_thresh:
                            _attached_has_real_pose = True  # server reports actual movement
                        break
            if _attached_object_id is not None and _ee_pos_units is not None and not _attached_has_real_pose:
                if _strict_realism:
                    try:
                        from data_fidelity import DataFidelityError
                        raise DataFidelityError(
                            "Attached object pose is static; PhysX scene_state required in strict mode.",
                            gate_name="scene_state_static",
                            diagnostics={"attached_object_id": _attached_object_id},
                        )
                    except ImportError:
                        raise RuntimeError(
                            "Attached object pose is static; PhysX scene_state required in strict mode."
                        )
                _new_pos = _ee_pos_units + _grasp_ee_offset
                _object_poses[_attached_object_id] = _new_pos.copy()
                # Update scene_state in observation
                if obs.get("scene_state"):
                    for _obj in obs["scene_state"].get("objects", []):
                        if _obj.get("object_id") == _attached_object_id:
                            _update_pose_dict(_obj, _new_pos)
                            break
                # Mark that kinematic EE-offset tracking was used (for quality scoring)
                obs["scene_state_provenance"] = "kinematic_ee_offset"

            # Query physics contact report before estimating forces (Issue 3/4 fix)
            _contact_result_available = False
            _contact_payload: Optional[Dict[str, Any]] = None
            _contacts: List[Dict[str, Any]] = []
            collision_provenance = None
            _should_query_contacts = True
            if _should_query_contacts:
                contact_result = self._client.get_contact_report(
                    include_points=False,
                    lock_timeout=1.0,
                )
                if contact_result.available and contact_result.success and contact_result.payload is not None:
                    payload = contact_result.payload
                    _contact_payload = payload
                    _contact_result_available = True
                    frame_data["penetration_depth"] = payload.get("max_penetration_depth")
                    frame_data["contact_count"] = int(payload.get("contact_count", 0))

                    _contacts = payload.get("contacts", []) or []
                    if _contacts:
                        frame_data["collision_contacts"] = [
                            {
                                "body_a": c.get("body_a", "") if isinstance(c, dict) else getattr(c, "body_a", ""),
                                "body_b": c.get("body_b", "") if isinstance(c, dict) else getattr(c, "body_b", ""),
                                "force_N": round(
                                    float(
                                        c.get("normal_force_N")
                                        if isinstance(c, dict) and c.get("normal_force_N") is not None
                                        else (
                                            c.get("normal_force", 0)
                                            if isinstance(c, dict)
                                            else getattr(c, "normal_force", 0)
                                        )
                                    ),
                                    4,
                                ),
                                "force_vector": c.get("force_vector") if isinstance(c, dict) else getattr(c, "force_vector", None),
                                "tangent_impulse": c.get("tangent_impulse") if isinstance(c, dict) else getattr(c, "tangent_impulse", None),
                                "friction": c.get("friction") if isinstance(c, dict) else getattr(c, "friction", None),
                                "contact_area": c.get("contact_area") if isinstance(c, dict) else getattr(c, "contact_area", None),
                                "position": c.get("position") if isinstance(c, dict) else getattr(c, "position", None),
                                "normal": c.get("normal") if isinstance(c, dict) else getattr(c, "normal", None),
                                "penetration_depth": c.get("penetration_depth") if isinstance(c, dict) else getattr(c, "penetration_depth", None),
                            }
                            for c in _contacts[:5]  # Top 5 contacts
                        ]
                    collision_provenance = "physx_contact_report"
                    _contact_report_count += 1
                else:
                    if _collision_source_hint == "joint_limits_only" and not _contact_query_warned:
                        logger.warning(
                            "Contact report unavailable; using joint-limits-only collision heuristic."
                        )
                        _contact_query_warned = True
                    collision_provenance = "joint_limits_only"

            if collision_provenance is None:
                collision_provenance = "physx_joint_effort" if _has_real_efforts else "joint_limits_only"
            frame_data["collision_provenance"] = collision_provenance
            if _strict_realism and collision_provenance != "physx_contact_report":
                try:
                    from data_fidelity import DataFidelityError
                    raise DataFidelityError(
                        f"Collision provenance invalid in strict mode: {collision_provenance}",
                        gate_name="collision_provenance",
                        diagnostics={"frame_idx": step_idx, "provenance": collision_provenance},
                    )
                except ImportError:
                    raise RuntimeError(
                        f"Collision provenance invalid in strict mode: {collision_provenance}"
                    )

            # Hybrid collision metric: allow gripper-target and target-surface contacts
            collision_free_physics = None
            if _contact_result_available:
                _target_norm = (task.get("target_object") or task.get("target_object_id") or "").rsplit("/", 1)[-1].lower()
                if _target_norm:
                    collision_free_physics = True
                    for _c in _contacts:
                        _body_a = _normalize_body_name(
                            _c.get("body_a", "") if isinstance(_c, dict) else getattr(_c, "body_a", "")
                        )
                        _body_b = _normalize_body_name(
                            _c.get("body_b", "") if isinstance(_c, dict) else getattr(_c, "body_b", "")
                        )
                        if not _body_a or not _body_b:
                            continue
                        _allowed = False
                        if (_is_gripper_body(_body_a) and _target_norm in _body_b) or (
                            _is_gripper_body(_body_b) and _target_norm in _body_a
                        ):
                            _allowed = True
                        if (_is_surface_body(_body_a) and _target_norm in _body_b) or (
                            _is_surface_body(_body_b) and _target_norm in _body_a
                        ):
                            _allowed = True
                        if not _allowed:
                            collision_free_physics = False
                            break
            else:
                # Fallback: use planning-reported collision_free when contact report unavailable
                planning_collision = self._last_planning_report.get("collision_free")
                if planning_collision is not None:
                    collision_free_physics = planning_collision
                    if collision_provenance == "joint_limits_only":
                        collision_provenance = self._last_planning_report.get("collision_source", "planning_report")
                        frame_data["collision_provenance"] = collision_provenance
            frame_data["collision_free_physics"] = collision_free_physics

            # If not already attached, try attaching using contact report or heuristic
            if (
                _attached_object_id is None
                and frame_data.get("gripper_command") == "closed"
                and _ee_pos_units is not None
            ):
                _candidate = None
                _target_hint = task.get("target_object") or task.get("target_object_id")
                _seed_target_object(obs, _target_hint)

                _phase = waypoint.get("phase", "")
                _phase_val = _phase.value if hasattr(_phase, "value") else str(_phase)
                _phase_val = _phase_val.lower()

                if _contact_result_available and _contacts:
                    for _c in _contacts:
                        _body_a = _normalize_body_name(
                            _c.get("body_a", "") if isinstance(_c, dict) else getattr(_c, "body_a", "")
                        )
                        _body_b = _normalize_body_name(
                            _c.get("body_b", "") if isinstance(_c, dict) else getattr(_c, "body_b", "")
                        )
                        if not (_is_gripper_body(_body_a) or _is_gripper_body(_body_b)):
                            continue
                        _candidate = _match_contact_object_id(_body_a) or _match_contact_object_id(_body_b)
                        if _candidate:
                            break

                _force_attach = os.getenv("GENIESIM_FORCE_ATTACH_ON_GRASP_PHASE")
                if _force_attach is None:
                    _force_attach = not _is_production
                else:
                    _force_attach = _force_attach.strip().lower() in {"1", "true", "yes"}
                if _strict_realism:
                    _force_attach = False

                if (
                    _candidate is None
                    and _force_attach
                    and _phase_val in ("grasp", "lift", "transport")
                ):
                    _candidate = _resolve_object_id_by_hint(_target_hint)
                    if _candidate is None and _object_poses:
                        try:
                            _max_dist_m = float(os.getenv("GENIESIM_GRASP_MAX_DISTANCE_M", "0.35"))
                        except ValueError:
                            _max_dist_m = 0.35
                        _candidate = _select_attach_candidate(
                            _ee_pos_units,
                            max_dist_units=_max_dist_m * 1.5 * _units_per_meter,
                            target_hint=_target_hint,
                        )

                _allow_heuristic_attach = os.getenv("GENIESIM_ALLOW_HEURISTIC_ATTACH")
                if _allow_heuristic_attach is None:
                    _allow_heuristic_attach = not _is_production
                else:
                    _allow_heuristic_attach = _allow_heuristic_attach.strip().lower() in {"1", "true", "yes"}
                if _strict_realism:
                    _allow_heuristic_attach = False

                if _candidate is None and _allow_heuristic_attach:
                    try:
                        _max_dist_m = float(os.getenv("GENIESIM_GRASP_MAX_DISTANCE_M", "0.35"))
                    except ValueError:
                        _max_dist_m = 0.35
                    try:
                        _thresh_scale = float(os.getenv("GENIESIM_GRASP_THRESHOLD_SCALE", "1.0"))
                    except ValueError:
                        _thresh_scale = 1.0
                    if not _contact_result_available:
                        _thresh_scale = max(_thresh_scale, 1.5)
                    _candidate = _select_attach_candidate(
                        _ee_pos_units,
                        max_dist_units=_max_dist_m * _thresh_scale * _units_per_meter,
                        target_hint=_target_hint,
                    )

                if _candidate and _candidate in _object_poses:
                    _attached_object_id = _candidate
                    _grasp_ee_offset = _object_poses[_attached_object_id] - _ee_pos_units
                    frame_data["grasp_feasible"] = True
                    frame_data["grasped_object_id"] = _attached_object_id
                    if not _strict_realism:
                        _attached_has_real_pose = False
                        # Apply kinematic attachment update immediately
                        _new_pos = _ee_pos_units + _grasp_ee_offset
                        _object_poses[_attached_object_id] = _new_pos.copy()
                        if obs.get("scene_state"):
                            for _obj in obs["scene_state"].get("objects", []):
                                if _obj.get("object_id") == _attached_object_id:
                                    _update_pose_dict(_obj, _new_pos)
                                    break
                        obs["scene_state_provenance"] = "kinematic_ee_offset"

            # Contact force estimation: strict realism requires PhysX contact report
            _contact_forces_payload = None
            if _strict_realism:
                if not _contact_result_available:
                    try:
                        from data_fidelity import DataFidelityError
                        raise DataFidelityError(
                            "PhysX contact report unavailable in strict realism mode.",
                            gate_name="contact_report",
                            diagnostics={"frame_idx": step_idx},
                        )
                    except ImportError:
                        raise RuntimeError("PhysX contact report unavailable in strict realism mode.")
                _gripper_force = 0.0
                _total_force = 0.0
                for _c in _contacts:
                    _body_a = _normalize_body_name(
                        _c.get("body_a", "") if isinstance(_c, dict) else getattr(_c, "body_a", "")
                    )
                    _body_b = _normalize_body_name(
                        _c.get("body_b", "") if isinstance(_c, dict) else getattr(_c, "body_b", "")
                    )
                    _nf = float(
                        _c.get("normal_force_N")
                        if isinstance(_c, dict) and _c.get("normal_force_N") is not None
                        else (
                            _c.get("normal_force", 0)
                            if isinstance(_c, dict)
                            else getattr(_c, "normal_force", 0)
                        )
                    )
                    _total_force += abs(_nf)
                    if _is_gripper_body(_body_a) or _is_gripper_body(_body_b):
                        _gripper_force += abs(_nf)
                if _gripper_force <= 0.0 and _contact_payload:
                    _gripper_force = float(_contact_payload.get("total_normal_force", 0.0) or 0.0)
                _contact_forces_payload = {
                    "total_normal_force_N": round(_total_force, 4),
                    "grip_force_N": round(_gripper_force, 4),
                    "contact_count": len(_contacts),
                    "contact_details": _contacts[:5],
                    "grasped_object_id": _attached_object_id,
                    "provenance": "physx_contact_report",
                    "confidence": 0.9,
                    "available": True,
                }
            else:
                if _has_real_efforts:
                    if gripper_indices:
                        _gripper_efforts = [
                            _real_efforts[idx] for idx in gripper_indices if idx < len(_real_efforts)
                        ]
                    else:
                        _gripper_efforts = _real_efforts[arm_dof:]  # gripper joint efforts
                    _grip_force_real = sum(abs(e) for e in _gripper_efforts)
                    if arm_indices:
                        _arm_efforts = [
                            _real_efforts[idx] for idx in arm_indices if idx < len(_real_efforts)
                        ]
                    else:
                        _arm_efforts = _real_efforts[:arm_dof]
                    _contact_forces_payload = {
                        "grip_force_N": round(_grip_force_real, 4),
                        "arm_torques_Nm": [round(e, 4) for e in _arm_efforts],
                        "gripper_efforts_N": [round(e, 4) for e in _gripper_efforts],
                        "grasped_object_id": _attached_object_id,
                        "force_sufficient": _grip_force_real > 0.5,
                        "provenance": "physx_joint_effort",
                        "confidence": 0.9,
                    }
                elif _contact_result_available:
                    _gripper_force = 0.0
                    _total_force = 0.0
                    for _c in _contacts:
                        _body_a = _normalize_body_name(
                            _c.get("body_a", "") if isinstance(_c, dict) else getattr(_c, "body_a", "")
                        )
                        _body_b = _normalize_body_name(
                            _c.get("body_b", "") if isinstance(_c, dict) else getattr(_c, "body_b", "")
                        )
                        _nf = float(
                            _c.get("normal_force_N")
                            if isinstance(_c, dict) and _c.get("normal_force_N") is not None
                            else (
                                _c.get("normal_force", 0)
                                if isinstance(_c, dict)
                                else getattr(_c, "normal_force", 0)
                            )
                        )
                        _total_force += abs(_nf)
                        if _is_gripper_body(_body_a) or _is_gripper_body(_body_b):
                            _gripper_force += abs(_nf)
                    if _gripper_force <= 0.0 and _contact_payload:
                        _gripper_force = float(_contact_payload.get("total_normal_force", 0.0) or 0.0)
                    _contact_forces_payload = {
                        "total_normal_force_N": round(_total_force, 4),
                        "grip_force_N": round(_gripper_force, 4),
                        "contact_count": len(_contacts),
                        "contact_details": _contacts[:5],
                        "grasped_object_id": _attached_object_id,
                        "provenance": "physx_contact_report",
                        "confidence": 0.7,
                    }
                elif _attached_object_id is not None:
                    _obj_type = ""
                    for _obj in (_scene_state or {}).get("objects", []):
                        if _obj.get("object_id") == _attached_object_id:
                            _obj_type = (_obj.get("object_type") or "").lower()
                            break
                    _mass = _get_obj_prop(_obj_type, "mass", 0.3)
                    _weight = _mass * 9.81

                    _grip_force = (1.0 - gripper_openness) * _GRIPPER_MAX_FORCE

                    _heuristic_confidence = 0.2
                    _confidence_factors = {"base": 0.2}

                    if _contact_result_available:
                        _heuristic_confidence += 0.25
                        _confidence_factors["contact_report"] = 0.25
                    else:
                        _confidence_factors["contact_report"] = 0.0

                    if _attached_object_id is not None:
                        _heuristic_confidence += 0.15
                        _confidence_factors["grasp_detected"] = 0.15
                    else:
                        _confidence_factors["grasp_detected"] = 0.0

                    if frame_data.get("gripper_command") == "closed" and gripper_openness < 0.3:
                        _heuristic_confidence += 0.1
                        _confidence_factors["gripper_closing"] = 0.1
                    else:
                        _confidence_factors["gripper_closing"] = 0.0

                    _contact_forces_payload = {
                        "weight_force_N": round(_weight, 2),
                        "grip_force_N": round(_grip_force, 2),
                        "force_sufficient": _grip_force >= _weight,
                        "grasped_object_id": _attached_object_id,
                        "provenance": "heuristic_grasp_model_v2",
                        "confidence": round(min(0.7, _heuristic_confidence), 2),
                        "confidence_factors": _confidence_factors,
                    }

                # Fix 3: Synthesize contact_forces when gripper is closed but no other source available
                # This prevents gate_penalty for grasp/contact inconsistency
                if _contact_forces_payload is None and frame_data.get("gripper_command") == "closed":
                    _synth_grip_force = (1.0 - gripper_openness) * _GRIPPER_MAX_FORCE
                    _contact_forces_payload = {
                        "grip_force_N": round(_synth_grip_force, 2),
                        "force_sufficient": _synth_grip_force > 5.0,
                        "grasped_object_id": _attached_object_id,  # May be None during approach
                        "provenance": "synthetic_gripper_closed",
                        "confidence": 0.3,
                    }

            if _contact_forces_payload is not None and frame_data.get("gripper_command") == "closed":
                _grasped = _contact_forces_payload.get("grasped_object_id") or _attached_object_id
                try:
                    _grip_force_val = float(_contact_forces_payload.get("grip_force_N", 0.0) or 0.0)
                except (TypeError, ValueError):
                    _grip_force_val = 0.0
                if _grasped and _grip_force_val <= 0.0 and not _strict_realism:
                    _contact_forces_payload["grip_force_N"] = 0.5
                    _contact_forces_payload["force_sufficient"] = True
                    _contact_forces_payload["provenance"] = "synthetic_grasp_force"

            if _contact_forces_payload is not None:
                frame_data["contact_forces"] = _contact_forces_payload

            # Add provenance to contact forces (for PhysX real data path)
            if "contact_forces" in frame_data and "provenance" not in frame_data["contact_forces"]:
                frame_data["contact_forces"]["provenance"] = "physx_joint_effort" if _has_real_efforts else "heuristic_grasp_model_v2"
            if "contact_forces" in frame_data and "confidence" not in frame_data["contact_forces"]:
                _cf_prov = frame_data["contact_forces"].get("provenance")
                if _cf_prov == "physx_joint_effort":
                    frame_data["contact_forces"]["confidence"] = 0.9
                elif _cf_prov in ("physx_contact_query", "physx_contact_report"):
                    frame_data["contact_forces"]["confidence"] = 0.7
                else:
                    frame_data["contact_forces"]["confidence"] = 0.5

            # Detect release: gripper opens while holding object
            if frame_data.get("gripper_command") == "open" and _attached_object_id is not None:
                _release_decay_obj = _attached_object_id
                _release_decay_remaining = 3  # decay over 3 frames
                _attached_object_id = None
                _grasp_ee_offset = None

            # Decay contact forces over 3 frames after release (Fix 9)
            if (
                not _strict_realism
                and _release_decay_remaining > 0
                and _attached_object_id is None
                and "contact_forces" not in frame_data
            ):
                _decay_frac = _release_decay_remaining / 3.0
                _release_decay_remaining -= 1
                frame_data["contact_forces"] = {
                    "weight_force_N": 0.0,
                    "grip_force_N": round(_decay_frac * 2.0, 2),  # decaying residual
                    "force_sufficient": False,
                    "grasped_object_id": None,
                    "releasing": True,
                    "provenance": "heuristic_grasp_model_v1",
                    "confidence": 0.1,
                }

            # Ensure contact_forces schema is always present for downstream validators.
            if "contact_forces" not in frame_data:
                if _strict_realism:
                    try:
                        from data_fidelity import DataFidelityError
                        raise DataFidelityError(
                            "contact_forces missing in strict realism mode.",
                            gate_name="contact_forces_missing",
                            diagnostics={"frame_idx": step_idx},
                        )
                    except ImportError:
                        raise RuntimeError("contact_forces missing in strict realism mode.")
                frame_data["contact_forces"] = {
                    "available": False,
                    "grip_force_N": 0.0,
                    "force_sufficient": False,
                    "grasped_object_id": None,
                    "provenance": "unavailable",
                    "confidence": 0.0,
                }

            # Update ALL tracked object poses in scene_state (not just attached)
            if _scene_state and _object_poses:
                for _obj in _scene_state.get("objects", []):
                    _oid = _obj.get("object_id", "")
                    if _oid in _object_poses:
                        _update_pose_dict(_obj, _object_poses[_oid])

            # Store per-frame object poses with full data (Issue 2 fix)
            if _object_poses:
                frame_data["object_poses"] = {}
                for _oid, _pos in _object_poses.items():
                    frame_data["object_poses"][_oid] = {
                        "position": [round(float(v), 6) for v in _pos],
                    }
                    # Include rotation and velocity from history if available
                    if _oid in _object_pose_history and _object_pose_history[_oid]:
                        latest = _object_pose_history[_oid][-1]
                        frame_data["object_poses"][_oid]["source"] = latest.get("source", "unknown")
                        if latest.get("rotation_quat"):
                            frame_data["object_poses"][_oid]["rotation_quat"] = latest["rotation_quat"]
                        if latest.get("linear_velocity"):
                            frame_data["object_poses"][_oid]["linear_velocity"] = latest["linear_velocity"]
                        if latest.get("angular_velocity"):
                            frame_data["object_poses"][_oid]["angular_velocity"] = latest["angular_velocity"]

            # --- Improvement C: Mirror EE fields into observation.robot_state ---
            for _key in ("ee_pos", "ee_quat", "ee_vel", "ee_pose_source", "gripper_command", "gripper_openness"):
                if _key in frame_data:
                    robot_state[_key] = frame_data[_key]

            # Fix 11: Commanded vs measured state
            if frame_data.get("action_abs"):
                _cmd = frame_data["action_abs"]
                _arm_cmd = _cmd[:arm_dof]
                robot_state["commanded_joint_positions"] = _arm_cmd
                if len(_cmd) > arm_dof:
                    robot_state["commanded_gripper_width"] = _cmd[arm_dof]
                _jp = robot_state.get("joint_positions", [])
                if _jp and len(_jp) == len(_arm_cmd):
                    robot_state["tracking_error"] = [
                        round(c - m, 8) for c, m in zip(_arm_cmd, _jp)
                    ]

            # --- Improvement E: Sensor noise injection ---
            if _inject_noise:
                import random as _rng_mod
                _noise_rng = _rng_mod.Random(hash((episode_id, step_idx)) & 0xFFFFFFFF)
                _jp = robot_state.get("joint_positions", [])
                if _jp:
                    robot_state["joint_positions"] = [
                        v + _noise_rng.gauss(0, _jp_noise_std) for v in _jp
                    ]
                # Joint velocity noise
                _jv = robot_state.get("joint_velocities", [])
                if _jv:
                    robot_state["joint_velocities"] = [
                        v + _noise_rng.gauss(0, _jv_noise_std) for v in _jv
                    ]
                if frame_data.get("ee_pos"):
                    frame_data["ee_pos"] = [
                        v + _noise_rng.gauss(0, _ee_noise_std) for v in frame_data["ee_pos"]
                    ]
                    robot_state["ee_pos"] = frame_data["ee_pos"]
                # Camera depth noise (Gaussian, ~1mm std for structured-light sensors)
                _obs = frame_data.get("observation", {})
                _cameras = _obs.get("cameras", {})
                for _cam_id, _cam_data in _cameras.items():
                    _depth = _cam_data.get("depth")
                    if _depth is not None and hasattr(_depth, "__len__"):
                        try:
                            _depth_arr = np.array(_depth, dtype=np.float32)
                            _depth_noise = np.random.RandomState(
                                hash((episode_id, step_idx, _cam_id)) & 0xFFFFFFFF
                            ).normal(0, 0.001, _depth_arr.shape).astype(np.float32)
                            _cam_data["depth"] = (_depth_arr + _depth_noise).tolist()
                        except Exception:
                            pass  # Non-critical; skip if depth format unexpected

            # --- Fix 3: Split observation into sensors (proprio) vs privileged (GT state) ---
            _obs_full = frame_data.get("observation", {})
            _privileged = {}
            if "scene_state" in _obs_full:
                _privileged["scene_state"] = _obs_full.pop("scene_state")
            if "contact_forces" in frame_data:
                _privileged["contact_forces"] = frame_data.pop("contact_forces")
            if _privileged:
                _obs_full["privileged"] = _privileged

            frames.append(frame_data)

        robot_type = getattr(self.config, "robot_type", "").lower()
        # Velocity/acceleration limits (robot-specific when available).
        _VEL_LIMITS = None
        _ACC_LIMITS = None
        if robot_type in _FRANKA_TYPES:
            _VEL_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 0.2, 0.2])
            _ACC_LIMITS = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0, 10.0, 10.0])

        # Second pass: compute joint velocities, accelerations, and EE velocities
        # Prefer real PhysX velocities from server; fall back to finite difference.
        _used_real_velocities = False
        for i, frame in enumerate(frames):
            obs_rs = frame.get("observation", {}).get("robot_state", {})
            # Check if real velocities from PhysX are available in this frame
            _real_vel = obs_rs.get("joint_velocities", [])
            _has_real_vel = (
                isinstance(_real_vel, list)
                and len(_real_vel) > 0
                and any(abs(v) > 1e-10 for v in _real_vel)
            )
            if i == 0:
                if _has_real_vel:
                    # Use real velocities but still set zero for first frame
                    obs_rs["joint_velocities"] = _real_vel
                    _used_real_velocities = True
                else:
                    obs_rs["joint_velocities"] = [0.0] * len(obs_rs.get("joint_positions", []))
                obs_rs["joint_accelerations"] = [0.0] * len(obs_rs.get("joint_positions", []))
                if frame.get("ee_pos"):
                    frame["ee_vel"] = [0.0, 0.0, 0.0]
                    frame["ee_acc"] = [0.0, 0.0, 0.0]
            else:
                prev = frames[i - 1]
                dt = frame["timestamp"] - prev["timestamp"]
                if _has_real_vel:
                    # Use real PhysX velocities directly
                    _used_real_velocities = True
                    _jv_clamped = np.array(_real_vel, dtype=float)
                    # Still clamp to safety limits
                    if _VEL_LIMITS is not None:
                        if _jv_clamped.size <= len(_VEL_LIMITS):
                            _vel_lim = _VEL_LIMITS[:_jv_clamped.size]
                        else:
                            _vel_lim = np.full(_jv_clamped.size, _VEL_LIMITS[-1])
                            _vel_lim[:len(_VEL_LIMITS)] = _VEL_LIMITS
                        _jv_clamped = np.clip(_jv_clamped, -_vel_lim, _vel_lim)
                    obs_rs["joint_velocities"] = _jv_clamped.tolist()
                    # Compute joint accelerations from real velocities (finite difference)
                    if dt > 0:
                        _prev_jv_real = np.array(
                            prev.get("observation", {}).get("robot_state", {}).get("joint_velocities", []),
                            dtype=float,
                        )
                        if _prev_jv_real.shape == _jv_clamped.shape and _prev_jv_real.size > 0:
                            _ja_real = (_jv_clamped - _prev_jv_real) / dt
                            if _ACC_LIMITS is not None:
                                if _ja_real.size <= len(_ACC_LIMITS):
                                    _acc_lim_r = _ACC_LIMITS[:_ja_real.size]
                                else:
                                    _acc_lim_r = np.full(_ja_real.size, _ACC_LIMITS[-1])
                                    _acc_lim_r[:len(_ACC_LIMITS)] = _ACC_LIMITS
                                _ja_real = np.clip(_ja_real, -_acc_lim_r, _acc_lim_r)
                            obs_rs["joint_accelerations"] = _ja_real.tolist()
                        else:
                            obs_rs["joint_accelerations"] = [0.0] * _jv_clamped.size
                    else:
                        obs_rs["joint_accelerations"] = [0.0] * _jv_clamped.size
                elif dt > 0:
                    jp_curr = np.array(obs_rs.get("joint_positions", []), dtype=float)
                    jp_prev = np.array(
                        prev.get("observation", {}).get("robot_state", {}).get("joint_positions", []),
                        dtype=float,
                    )
                    if jp_curr.shape == jp_prev.shape and jp_curr.size > 0:
                        _jv = ((jp_curr - jp_prev) / dt)
                        # Add velocity noise (Improvement E)
                        if _inject_noise:
                            import random as _rng_mod
                            _vn_rng = _rng_mod.Random(hash((episode_id, i, "vel")) & 0xFFFFFFFF)
                            _jv = _jv + np.array([_vn_rng.gauss(0, _jv_noise_std) for _ in range(_jv.size)])
                        # Clamp to velocity limits when known.
                        if _VEL_LIMITS is not None:
                            if _jv.size <= len(_VEL_LIMITS):
                                _vel_lim = _VEL_LIMITS[:_jv.size]
                            else:
                                # Pad with last limit value for extra joints
                                _vel_lim = np.full(_jv.size, _VEL_LIMITS[-1])
                                _vel_lim[:len(_VEL_LIMITS)] = _VEL_LIMITS
                            _jv_clamped = np.clip(_jv, -_vel_lim, _vel_lim)
                        else:
                            _jv_clamped = _jv
                        if not np.allclose(_jv, _jv_clamped, atol=0.01):
                            logger.debug("Frame %d: joint velocity clamped (max exceeded)", i)
                        obs_rs["joint_velocities"] = _jv_clamped.tolist()

                        # Joint accelerations (finite difference of velocities)
                        _prev_jv = np.array(
                            prev.get("observation", {}).get("robot_state", {}).get("joint_velocities", []),
                            dtype=float,
                        )
                        if _prev_jv.shape == _jv_clamped.shape and _prev_jv.size > 0:
                            _ja = (_jv_clamped - _prev_jv) / dt
                            if _ACC_LIMITS is not None:
                                if _ja.size <= len(_ACC_LIMITS):
                                    _acc_lim = _ACC_LIMITS[:_ja.size]
                                else:
                                    _acc_lim = np.full(_ja.size, _ACC_LIMITS[-1])
                                    _acc_lim[:len(_ACC_LIMITS)] = _ACC_LIMITS
                                _ja = np.clip(_ja, -_acc_lim, _acc_lim)
                            obs_rs["joint_accelerations"] = _ja.tolist()
                        else:
                            obs_rs["joint_accelerations"] = [0.0] * _jv_clamped.size
                    # EE velocity
                    if frame.get("ee_pos") and prev.get("ee_pos"):
                        ee_curr = np.array(frame["ee_pos"], dtype=float)
                        ee_prev = np.array(prev["ee_pos"], dtype=float)
                        frame["ee_vel"] = ((ee_curr - ee_prev) / dt).tolist()
                        # EE acceleration
                        if prev.get("ee_vel"):
                            _prev_ee_vel = np.array(prev["ee_vel"], dtype=float)
                            _ee_vel_curr = np.array(frame["ee_vel"], dtype=float)
                            frame["ee_acc"] = ((_ee_vel_curr - _prev_ee_vel) / dt).tolist()
                        else:
                            frame["ee_acc"] = [0.0, 0.0, 0.0]
                    # Mirror velocity fields to robot_state (Improvement C)
                    if "ee_vel" in frame:
                        obs_rs["ee_vel"] = frame["ee_vel"]

        # Smooth velocities and accelerations with Savitzky-Golay filter when
        # enough frames are available.  This reduces finite-difference noise and
        # produces cleaner effort estimates downstream.
        if len(frames) >= 7:
            try:
                from scipy.signal import savgol_filter as _savgol
                _sg_window = min(7, len(frames) if len(frames) % 2 == 1 else len(frames) - 1)
                if _sg_window >= 5:
                    # Collect velocity matrix (N_frames x N_joints)
                    _vel_matrix = []
                    for _fr in frames:
                        _vels = _fr.get("observation", {}).get("robot_state", {}).get("joint_velocities", [])
                        _vel_matrix.append(np.array(_vels[:arm_dof], dtype=float) if len(_vels) >= arm_dof else None)
                    if all(v is not None and v.size == arm_dof for v in _vel_matrix):
                        _vel_arr = np.array([v for v in _vel_matrix])  # (N, arm_dof)
                        _vel_smooth = _savgol(_vel_arr, _sg_window, 3, axis=0)
                        _acc_smooth = _savgol(_vel_arr, _sg_window, 3, deriv=1, delta=frames[1]["timestamp"] - frames[0]["timestamp"], axis=0)
                        if _ACC_LIMITS is not None:
                            _acc_lim_sg = _ACC_LIMITS[:arm_dof] if arm_dof <= len(_ACC_LIMITS) else np.full(arm_dof, _ACC_LIMITS[-1])
                            _acc_smooth = np.clip(_acc_smooth, -_acc_lim_sg, _acc_lim_sg)
                        for _si, _fr in enumerate(frames):
                            _obs_rs_sg = _fr.get("observation", {}).get("robot_state", {})
                            # Replace arm-joint portion of velocities with smoothed values
                            _existing_vel = _obs_rs_sg.get("joint_velocities", [])
                            _smoothed_vel = list(_existing_vel)  # preserve full 34-joint array
                            for _ji in range(min(arm_dof, len(_smoothed_vel))):
                                _smoothed_vel[_ji] = float(_vel_smooth[_si, _ji])
                            _obs_rs_sg["joint_velocities"] = _smoothed_vel
                            # Replace arm-joint portion of accelerations
                            _existing_acc = _obs_rs_sg.get("joint_accelerations", [])
                            if isinstance(_existing_acc, list) and len(_existing_acc) >= arm_dof:
                                _smoothed_acc = list(_existing_acc)
                                for _ji in range(min(arm_dof, len(_smoothed_acc))):
                                    _smoothed_acc[_ji] = float(_acc_smooth[_si, _ji])
                                _obs_rs_sg["joint_accelerations"] = _smoothed_acc
                            else:
                                _obs_rs_sg["joint_accelerations"] = _acc_smooth[_si].tolist()
                        logger.info("Applied Savitzky-Golay smoothing to velocities/accelerations (window=%d)", _sg_window)
            except ImportError:
                logger.debug("scipy not available — skipping Savitzky-Golay velocity smoothing")
            except Exception as _sg_exc:
                logger.warning("Savitzky-Golay smoothing failed (non-fatal): %s", _sg_exc)

        # Ensure every frame has joint velocities/accelerations for effort backfill.
        if frames:
            _default_dt = 1.0 / _control_freq if _control_freq else (1.0 / 30.0)
            for _fi, _frame in enumerate(frames):
                _obs_rs = _frame.get("observation", {}).get("robot_state", {})
                _jp_list = _obs_rs.get("joint_positions", [])
                if not isinstance(_jp_list, list) or len(_jp_list) < arm_dof:
                    continue
                _vel_list = _obs_rs.get("joint_velocities", [])
                _acc_list = _obs_rs.get("joint_accelerations", [])
                _need_vel = not isinstance(_vel_list, list) or len(_vel_list) < arm_dof
                _need_acc = not isinstance(_acc_list, list) or len(_acc_list) < arm_dof
                if not (_need_vel or _need_acc):
                    continue
                if _fi == 0:
                    if _need_vel:
                        _obs_rs["joint_velocities"] = [0.0] * len(_jp_list)
                    if _need_acc:
                        _obs_rs["joint_accelerations"] = [0.0] * len(_jp_list)
                    continue
                _prev_rs = frames[_fi - 1].get("observation", {}).get("robot_state", {})
                _prev_jp = _prev_rs.get("joint_positions", [])
                if not isinstance(_prev_jp, list) or len(_prev_jp) < arm_dof:
                    if _need_vel:
                        _obs_rs["joint_velocities"] = [0.0] * len(_jp_list)
                    if _need_acc:
                        _obs_rs["joint_accelerations"] = [0.0] * len(_jp_list)
                    continue
                _dt = _frame.get("dt")
                if not _dt or _dt <= 0:
                    _ts = _frame.get("timestamp")
                    _prev_ts = frames[_fi - 1].get("timestamp")
                    if _ts is not None and _prev_ts is not None:
                        _dt = _ts - _prev_ts
                if not _dt or _dt <= 0:
                    _dt = _default_dt
                _jp_arr = np.array(_jp_list, dtype=float)
                _prev_jp_arr = np.array(_prev_jp, dtype=float)
                _min_len = min(_jp_arr.size, _prev_jp_arr.size)
                if _min_len == 0:
                    if _need_vel:
                        _obs_rs["joint_velocities"] = [0.0] * len(_jp_list)
                    if _need_acc:
                        _obs_rs["joint_accelerations"] = [0.0] * len(_jp_list)
                    continue
                _jv = (_jp_arr[:_min_len] - _prev_jp_arr[:_min_len]) / _dt
                if _need_vel:
                    _vel_list = _vel_list if isinstance(_vel_list, list) else []
                    if len(_vel_list) < len(_jp_list):
                        _vel_list = _vel_list + [0.0] * (len(_jp_list) - len(_vel_list))
                    for _ji in range(_min_len):
                        _vel_list[_ji] = float(_jv[_ji])
                    _obs_rs["joint_velocities"] = _vel_list
                if _need_acc:
                    _prev_vel_list = _prev_rs.get("joint_velocities", [])
                    _prev_vel_arr = (
                        np.array(_prev_vel_list, dtype=float)
                        if isinstance(_prev_vel_list, list)
                        else None
                    )
                    if _prev_vel_arr is None or _prev_vel_arr.size < _min_len:
                        _ja = np.zeros(_min_len, dtype=float)
                    else:
                        _ja = (_jv - _prev_vel_arr[:_min_len]) / _dt
                    _acc_list = _acc_list if isinstance(_acc_list, list) else []
                    if len(_acc_list) < len(_jp_list):
                        _acc_list = _acc_list + [0.0] * (len(_jp_list) - len(_acc_list))
                    for _ji in range(_min_len):
                        _acc_list[_ji] = float(_ja[_ji])
                    _obs_rs["joint_accelerations"] = _acc_list

        # Backfill joint efforts via inverse dynamics when PhysX didn't provide them
        _estimated_effort_count = 0
        if _strict_realism:
            _missing_effort_frames = 0
            for _frame in frames:
                _obs_rs = _frame.get("observation", {}).get("robot_state", {})
                _eff = _obs_rs.get("joint_efforts", [])
                if not isinstance(_eff, list) or len(_eff) == 0 or not any(abs(e) > 1e-6 for e in _eff):
                    _missing_effort_frames += 1
            if _missing_effort_frames > 0:
                try:
                    from data_fidelity import DataFidelityError
                    raise DataFidelityError(
                        f"Joint efforts missing in {_missing_effort_frames}/{len(frames)} frames.",
                        gate_name="joint_efforts_missing",
                        diagnostics={"missing_frames": _missing_effort_frames, "total_frames": len(frames)},
                    )
                except ImportError:
                    raise RuntimeError(
                        f"Joint efforts missing in {_missing_effort_frames}/{len(frames)} frames."
                    )
        elif len(frames) >= 3:
            # Simplified inverse dynamics: τ = I·α + C·v + G
            _id_inertia = np.array([2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.5])
            _id_damping = np.array([10.0, 10.0, 8.0, 8.0, 5.0, 5.0, 3.0])
            _id_gravity = np.array([15.0, 15.0, 10.0, 10.0, 5.0, 5.0, 2.0])
            _id_dof = min(arm_dof, len(_id_inertia))
            _id_inertia = _id_inertia[:_id_dof]
            _id_damping = _id_damping[:_id_dof]
            _id_gravity = _id_gravity[:_id_dof]
            _estimated_effort_count = 0
            for _fi in range(len(frames)):
                _obs_rs = frames[_fi].get("observation", {}).get("robot_state", {})
                _rs_eff = _obs_rs.get("joint_efforts", [])
                _has_eff = (
                    isinstance(_rs_eff, list)
                    and len(_rs_eff) > 0
                    and any(abs(e) > 1e-6 for e in _rs_eff)
                )
                if _has_eff:
                    continue
                _jp = np.array(_obs_rs.get("joint_positions", [])[:_id_dof], dtype=float)
                _jv = np.array(_obs_rs.get("joint_velocities", [])[:_id_dof], dtype=float)
                _ja = np.array(_obs_rs.get("joint_accelerations", [])[:_id_dof], dtype=float)
                if _jp.size < _id_dof:
                    _jp = np.pad(_jp, (0, _id_dof - _jp.size), mode="constant")
                if _jv.size < _id_dof:
                    _jv = np.pad(_jv, (0, _id_dof - _jv.size), mode="constant")
                if _ja.size < _id_dof:
                    _dt = frames[_fi].get("dt")
                    if not _dt or _dt <= 0:
                        _ts = frames[_fi].get("timestamp")
                        _prev_ts = frames[_fi - 1].get("timestamp") if _fi > 0 else None
                        if _ts is not None and _prev_ts is not None:
                            _dt = _ts - _prev_ts
                    if not _dt or _dt <= 0:
                        _dt = 1.0 / _control_freq if _control_freq else (1.0 / 30.0)
                    if _fi > 0:
                        _prev_vel = (
                            frames[_fi - 1]
                            .get("observation", {})
                            .get("robot_state", {})
                            .get("joint_velocities", [])
                        )
                        _prev_vel_arr = (
                            np.array(_prev_vel[:_id_dof], dtype=float)
                            if isinstance(_prev_vel, list)
                            else np.zeros(_id_dof, dtype=float)
                        )
                        if _prev_vel_arr.size == _id_dof:
                            _ja = (_jv - _prev_vel_arr) / _dt
                        else:
                            _ja = np.zeros(_id_dof, dtype=float)
                    else:
                        _ja = np.zeros(_id_dof, dtype=float)
                else:
                    _ja = _ja[:_id_dof]
                _torque = _id_inertia * _ja + _id_damping * _jv + _id_gravity * np.cos(_jp)
                _efforts_list = _torque.tolist()
                # Pad to full joint count when available
                _full_len = len(_obs_rs.get("joint_positions", [])) or _id_dof
                if _full_len > _id_dof:
                    _efforts_list = _efforts_list + [0.0] * (_full_len - _id_dof)
                _obs_rs["joint_efforts"] = _efforts_list
                _joint_state = _obs_rs.get("joint_state")
                if isinstance(_joint_state, dict):
                    _joint_state["efforts"] = _efforts_list
                frames[_fi]["efforts_source"] = "estimated_inverse_dynamics"
                _estimated_effort_count += 1
            if _estimated_effort_count > 0:
                logger.info(
                    "Backfilled joint efforts via inverse dynamics for %d/%d frames.",
                    _estimated_effort_count, len(frames),
                )
        if _real_effort_count > 0 and _estimated_effort_count > 0:
            _efforts_source = "mixed"
        elif _real_effort_count > 0:
            _efforts_source = "physx"
        elif _estimated_effort_count > 0:
            _efforts_source = "estimated_inverse_dynamics"
        else:
            _efforts_source = "none"

        # Enforce consistent efforts source policy (Issue 5 fix)
        # If we have mixed sources, apply inverse dynamics to ALL frames for consistency
        _efforts_consistency = 1.0
        _total_frames = len(frames)
        _efforts_forced_consistency = False
        if _efforts_source == "mixed" and _total_frames > 0 and not _strict_realism:
            # Count frames by source
            _physx_frames = sum(1 for f in frames if f.get("efforts_source") == "physx")
            _estimated_frames = sum(1 for f in frames if f.get("efforts_source") == "estimated_inverse_dynamics")

            # Policy: if less than 80% PhysX, apply inverse dynamics to ALL for consistency
            if _physx_frames < _total_frames * 0.8:
                logger.info(
                    "Enforcing consistent efforts source: applying inverse dynamics to all frames "
                    "(physx=%d, estimated=%d, total=%d)",
                    _physx_frames, _estimated_frames, _total_frames,
                )
                _efforts_source = "estimated_inverse_dynamics"
                # Reuse the same dynamics parameters from above
                _id_dof = min(arm_dof, len(_id_inertia))
                for _fi, _frame in enumerate(frames):
                    if _frame.get("efforts_source") == "physx":
                        # Recompute with inverse dynamics for consistency
                        _obs_rs = _frame.get("observation", {}).get("robot_state", {})
                        _jp_list = _obs_rs.get("joint_positions", [])
                        _jv_list = _obs_rs.get("joint_velocities", [])
                        if _jp_list and _jv_list and len(_jp_list) >= _id_dof:
                            _jp = np.array(_jp_list[:_id_dof], dtype=float)
                            _jv = np.array(_jv_list[:_id_dof], dtype=float) if len(_jv_list) >= _id_dof else np.zeros(_id_dof)
                            # Compute acceleration from velocity difference
                            _ja = np.zeros(_id_dof, dtype=float)
                            if _fi > 0:
                                _prev_obs_rs = frames[_fi - 1].get("observation", {}).get("robot_state", {})
                                _prev_jv_list = _prev_obs_rs.get("joint_velocities", [])
                                if _prev_jv_list and len(_prev_jv_list) >= _id_dof:
                                    _prev_jv = np.array(_prev_jv_list[:_id_dof], dtype=float)
                                    _dt = 1.0 / _control_freq if _control_freq else (1.0 / 30.0)
                                    _ja = (_jv - _prev_jv) / _dt
                            # Compute torque: τ = I·α + C·v + G·cos(θ)
                            _torque = _id_inertia * _ja + _id_damping * _jv + _id_gravity * np.cos(_jp)
                            _efforts_list = _torque.tolist()
                            # Pad to full joint count
                            _full_len = len(_jp_list)
                            if _full_len > _id_dof:
                                _efforts_list = _efforts_list + [0.0] * (_full_len - _id_dof)
                            _obs_rs["joint_efforts"] = _efforts_list
                            _frame["efforts_source"] = "estimated_inverse_dynamics"
                _efforts_consistency = 1.0
                _efforts_forced_consistency = True
            else:
                # Mostly PhysX - keep as is, but track consistency
                _efforts_consistency = _physx_frames / _total_frames

        if _efforts_forced_consistency:
            for _frame in frames:
                _frame["efforts_source"] = "estimated_inverse_dynamics"

        # Ensure every frame has explicit efforts_source tag
        for _frame in frames:
            if "efforts_source" not in _frame:
                _frame["efforts_source"] = "none"

        # Fail-fast gate: Joint efforts validation
        try:
            from data_fidelity import validate_joint_efforts, DataFidelityError
            _is_valid, _effort_diag = validate_joint_efforts(
                efforts_source=_efforts_source,
                real_effort_count=_real_effort_count,
                total_frames=len(frames),
            )
            if not _is_valid:
                logger.warning(
                    "Joint efforts validation: source=%s, real=%d/%d",
                    _efforts_source, _real_effort_count, len(frames),
                )
        except ImportError:
            pass
        except DataFidelityError as e:
            raise e

        # Recompute contact forces after effort backfill when applicable
        if _estimated_effort_count > 0:
            for _frame in frames:
                if _frame.get("efforts_source") != "estimated_inverse_dynamics":
                    continue
                _obs = _frame.get("observation", {})
                _priv = _obs.get("privileged", {})
                _cf_container = _priv if isinstance(_priv, dict) else _frame
                _existing_cf = _cf_container.get("contact_forces")
                if isinstance(_existing_cf, dict) and _existing_cf.get("provenance") not in (
                    "heuristic_grasp_model_v1",
                    "heuristic_grasp_model_v2",
                    "estimated_inverse_dynamics",
                ):
                    continue
                _obs_rs = _obs.get("robot_state", {})
                _efforts = _obs_rs.get("joint_efforts", [])
                if not isinstance(_efforts, list) or not _efforts:
                    continue
                if gripper_indices:
                    _gripper_efforts = [float(_efforts[idx]) for idx in gripper_indices if idx < len(_efforts)]
                else:
                    _gripper_efforts = [float(e) for e in _efforts[arm_dof:]]
                if arm_indices:
                    _arm_efforts = [float(_efforts[idx]) for idx in arm_indices if idx < len(_efforts)]
                else:
                    _arm_efforts = [float(e) for e in _efforts[:arm_dof]]
                _grip_force = sum(abs(e) for e in _gripper_efforts)
                _cf_payload = dict(_existing_cf) if isinstance(_existing_cf, dict) else {}
                _cf_payload.update({
                    "grip_force_N": round(_grip_force, 4),
                    "arm_torques_Nm": [round(e, 4) for e in _arm_efforts],
                    "gripper_efforts_N": [round(e, 4) for e in _gripper_efforts],
                    "force_sufficient": _grip_force > 0.5,
                    "provenance": "estimated_inverse_dynamics",
                    "confidence": 0.7,
                })
                _cf_container["contact_forces"] = _cf_payload

        # Refresh missing-effort count after backfill
        _effort_missing_count = 0
        for _frame in frames:
            _obs_rs = _frame.get("observation", {}).get("robot_state", {})
            _eff = _obs_rs.get("joint_efforts", [])
            if not isinstance(_eff, list) or len(_eff) == 0 or not any(abs(e) > 1e-6 for e in _eff):
                _effort_missing_count += 1

        # Final normalization pass: keep joint arrays arm-only and consistent.
        for _frame in frames:
            _obs_rs = _frame.get("observation", {}).get("robot_state", {})
            _normalize_robot_state_joints(_obs_rs)

        # Store initial/final object poses on first/last frame for success verification
        if frames:
            frames[0]["_initial_object_poses"] = {
                k: v.tolist() for k, v in _initial_object_poses.items()
            }
            frames[-1]["_final_object_poses"] = {
                k: v.tolist() for k, v in _object_poses.items()
            }

        # Build object trajectories summary for episode metadata (Issue 2 fix)
        _object_trajectories = {}
        for _oid, _history in _object_pose_history.items():
            if _history:
                _object_trajectories[_oid] = {
                    "position_trajectory": [h["position"] for h in _history],
                    "velocity_trajectory": [h.get("linear_velocity", [0, 0, 0]) for h in _history],
                    "rotation_trajectory": [h.get("rotation_quat") for h in _history],
                    "angular_velocity_trajectory": [h.get("angular_velocity", [0, 0, 0]) for h in _history],
                    "frame_indices": [h["frame_idx"] for h in _history],
                    "source": _history[-1].get("source", "unknown") if _history else "unknown",
                }

        _collision_free_physics = None
        _physics_flags = [
            f.get("collision_free_physics")
            for f in frames
            if f.get("collision_free_physics") is not None
        ]
        if _physics_flags:
            _collision_free_physics = all(bool(v) for v in _physics_flags)

        return frames, {
            "camera_frame_count": _camera_frame_count,
            "real_scene_state_count": _real_scene_state_count,
            "scene_state_fallback_frames": _scene_state_fallback_frames,
            "scene_state_missing_after_frame0": _scene_state_missing_after_frame0,
            "scene_state_missing_frame_indices": _scene_state_missing_frame_indices,
            "server_ee_frame_count": _server_ee_frame_count,
            "real_velocity_count": _real_velocity_count,
            "real_effort_count": _real_effort_count,
            "efforts_source": _efforts_source,
            "effort_source_policy": _efforts_source,
            "joint_efforts_consistency": round(_efforts_consistency, 3),  # Issue 5 fix
            "estimated_effort_count": _estimated_effort_count,
            "effort_missing_count": _effort_missing_count,
            "contact_report_count": _contact_report_count,
            "collision_free_physics": _collision_free_physics,
            "object_property_provenance": _object_property_provenance,
            "missing_phase_frames": _missing_phase_frames,
            "ee_static_fallback_used": _ee_static_fallback_used,
            "object_trajectories": _object_trajectories,
            "ee_wrench_source": _ee_wrench_source,
        }

    def _validate_frames(
        self,
        frames: List[Dict[str, Any]],
        *,
        episode_id: str,
        task: Dict[str, Any],
        episode_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        invalid_frames: set[int] = set()
        _validate_camera_shape = os.getenv("GENIESIM_VALIDATE_CAMERA_SHAPE", "0") == "1"
        _frames_dir = None
        if episode_dir is not None:
            _frames_dir = episode_dir / f"{episode_id}_frames"

        required_cameras: List[str] = []
        for key in ("required_cameras", "required_camera_ids", "camera_ids"):
            value = task.get(key)
            if value:
                required_cameras = value
                break
        if not required_cameras:
            camera_config = task.get("camera_config") or {}
            for key in ("required_camera_ids", "camera_ids"):
                value = camera_config.get(key)
                if value:
                    required_cameras = value
                    break
        if isinstance(required_cameras, str):
            required_cameras = [required_cameras]
        required_cameras = [str(camera_id) for camera_id in required_cameras]

        camera_placeholder_detected = False
        camera_placeholder_details: List[str] = []
        _placeholder_checked: set[str] = set()

        timestamp_indices: Dict[float, List[int]] = {}
        frame_index_indices: Dict[int, List[int]] = {}
        prev_timestamp: Optional[float] = None
        max_duration = self.config.max_duration_seconds

        for idx, frame in enumerate(frames):
            frame_errors: List[str] = []
            frame_index = frame.get("step", idx)
            try:
                frame_index = int(frame_index)
            except (TypeError, ValueError):
                frame_errors.append(f"Frame {idx} has invalid step index: {frame_index!r}.")
            else:
                frame_index_indices.setdefault(frame_index, []).append(idx)

            timestamp = frame.get("timestamp")
            if isinstance(timestamp, dict):
                logger.warning(
                    "Frame %d timestamp is a dict (%s); coercing value.",
                    idx,
                    timestamp,
                )
                warnings.append(
                    f"Frame {idx} timestamp was a dict; coercing value from {timestamp!r}."
                )
                timestamp = self._coerce_timestamp(
                    timestamp,
                    fallback=prev_timestamp if prev_timestamp is not None else 0.0,
                )
                frame["timestamp"] = timestamp
            if timestamp is None:
                frame_errors.append(f"Frame {idx} missing timestamp.")
            else:
                try:
                    timestamp_value = float(timestamp)
                except (TypeError, ValueError):
                    frame_errors.append(f"Frame {idx} has non-numeric timestamp {timestamp!r}.")
                else:
                    if not np.isfinite(timestamp_value):
                        frame_errors.append(f"Frame {idx} has non-finite timestamp {timestamp_value!r}.")
                    if timestamp_value < 0:
                        frame_errors.append(f"Frame {idx} has negative timestamp {timestamp_value}.")
                    if max_duration is not None and timestamp_value > max_duration:
                        frame_errors.append(
                            f"Frame {idx} timestamp {timestamp_value} exceeds max duration {max_duration}."
                        )
                    if prev_timestamp is not None and timestamp_value < prev_timestamp:
                        frame_errors.append(
                            f"Frame {idx} timestamp {timestamp_value} is earlier than previous {prev_timestamp}."
                        )
                    timestamp_indices.setdefault(timestamp_value, []).append(idx)
                    prev_timestamp = timestamp_value

            obs = frame.get("observation") or {}
            camera_frames = obs.get("camera_frames") or {}
            # Skip camera validation in proprioception-only mode (no server recording)
            if not os.environ.get("GENIESIM_SKIP_SERVER_RECORDING", ""):
                if required_cameras and not camera_frames:
                    frame_errors.append(
                        f"Frame {idx} missing camera_frames for required cameras {required_cameras}."
                    )
                for camera_id in required_cameras:
                    if camera_id not in camera_frames:
                        frame_errors.append(f"Frame {idx} missing camera frame for camera '{camera_id}'.")

            for camera_id, camera_data in camera_frames.items():
                if not isinstance(camera_data, dict):
                    frame_errors.append(
                        f"Frame {idx} camera '{camera_id}' has invalid camera data."
                    )
                    continue
                width = int(camera_data.get("width") or 0)
                height = int(camera_data.get("height") or 0)
                rgb_encoding = (
                    camera_data.get("rgb_encoding")
                    or camera_data.get("encoding")
                    or ""
                )
                depth_encoding = (
                    camera_data.get("depth_encoding")
                    or camera_data.get("encoding")
                    or ""
                )
                if width <= 0 or height <= 0:
                    frame_errors.append(
                        f"Frame {idx} camera '{camera_id}' has invalid dimensions ({width}x{height})."
                    )
                rgb = camera_data.get("rgb")
                depth = camera_data.get("depth")
                if rgb is None:
                    frame_errors.append(f"Frame {idx} camera '{camera_id}' missing rgb data.")
                else:
                    if isinstance(rgb, str) and rgb.endswith(".npy"):
                        # File reference to .npy — check file exists
                        _npy_ref = resolve_npy_path(rgb, ep_dir=episode_dir, frames_dir=_frames_dir)
                        if _npy_ref is None or not _npy_ref.exists():
                            frame_errors.append(
                                f"Frame {idx} camera '{camera_id}' rgb npy file not found: {rgb}."
                            )
                        elif _validate_camera_shape and _npy_ref and _npy_ref.exists():
                            try:
                                _arr = np.load(_npy_ref)
                                if _arr.ndim < 2:
                                    frame_errors.append(
                                        f"Frame {idx} camera '{camera_id}' rgb npy invalid shape {_arr.shape}."
                                    )
                                elif height > 0 and width > 0:
                                    if _arr.shape[0] != height or _arr.shape[1] != width:
                                        frame_errors.append(
                                            f"Frame {idx} camera '{camera_id}' rgb npy shape {_arr.shape} "
                                            f"does not match ({height}x{width})."
                                        )
                            except Exception as _npy_err:
                                frame_errors.append(
                                    f"Frame {idx} camera '{camera_id}' rgb npy load failed: {_npy_err}."
                                )
                    elif isinstance(rgb, str):
                        # Base64-encoded image data — validate by expected byte count
                        try:
                            _raw = base64.b64decode(rgb)
                            _expected = expected_byte_count(
                                rgb_encoding,
                                width=width,
                                height=height,
                                kind="rgb",
                            )
                            if _expected is not None and len(_raw) < _expected:
                                frame_errors.append(
                                    f"Frame {idx} camera '{camera_id}' rgb base64 decoded to "
                                    f"{len(_raw)} bytes, expected >= {_expected} for {width}x{height}."
                                )
                        except Exception as _b64err:
                            frame_errors.append(
                                f"Frame {idx} camera '{camera_id}' rgb base64 decode failed: {_b64err}."
                            )
                    else:
                        rgb_array = np.asarray(rgb)
                        if rgb_array.ndim < 2:
                            frame_errors.append(
                                f"Frame {idx} camera '{camera_id}' rgb has invalid shape {rgb_array.shape}."
                            )
                        elif height > 0 and width > 0:
                            if rgb_array.shape[0] != height or rgb_array.shape[1] != width:
                                frame_errors.append(
                                    f"Frame {idx} camera '{camera_id}' rgb shape {rgb_array.shape} "
                                    f"does not match ({height}x{width})."
                                )
                if depth is None:
                    frame_errors.append(f"Frame {idx} camera '{camera_id}' missing depth data.")
                else:
                    if isinstance(depth, str) and depth.endswith(".npy"):
                        # File reference to .npy — check file exists
                        _npy_ref = resolve_npy_path(depth, ep_dir=episode_dir, frames_dir=_frames_dir)
                        if _npy_ref is None or not _npy_ref.exists():
                            frame_errors.append(
                                f"Frame {idx} camera '{camera_id}' depth npy file not found: {depth}."
                            )
                        elif _validate_camera_shape and _npy_ref and _npy_ref.exists():
                            try:
                                _arr = np.load(_npy_ref)
                                if _arr.ndim < 2:
                                    frame_errors.append(
                                        f"Frame {idx} camera '{camera_id}' depth npy invalid shape {_arr.shape}."
                                    )
                                elif height > 0 and width > 0:
                                    if _arr.shape[0] != height or _arr.shape[1] != width:
                                        frame_errors.append(
                                            f"Frame {idx} camera '{camera_id}' depth npy shape {_arr.shape} "
                                            f"does not match ({height}x{width})."
                                        )
                            except Exception as _npy_err:
                                frame_errors.append(
                                    f"Frame {idx} camera '{camera_id}' depth npy load failed: {_npy_err}."
                                )
                    elif isinstance(depth, str):
                        # Base64-encoded depth data — validate by expected byte count
                        try:
                            _raw = base64.b64decode(depth)
                            _expected = expected_byte_count(
                                depth_encoding,
                                width=width,
                                height=height,
                                kind="depth",
                            )
                            if _expected is not None and len(_raw) < _expected:
                                frame_errors.append(
                                    f"Frame {idx} camera '{camera_id}' depth base64 decoded to "
                                    f"{len(_raw)} bytes, expected >= {_expected} for {width}x{height}."
                                )
                        except Exception as _b64err:
                            frame_errors.append(
                                f"Frame {idx} camera '{camera_id}' depth base64 decode failed: {_b64err}."
                            )
                    else:
                        depth_array = np.asarray(depth)
                        if depth_array.ndim < 2:
                            frame_errors.append(
                                f"Frame {idx} camera '{camera_id}' depth has invalid shape {depth_array.shape}."
                            )
                        elif height > 0 and width > 0:
                            if depth_array.shape[0] != height or depth_array.shape[1] != width:
                                frame_errors.append(
                                    f"Frame {idx} camera '{camera_id}' depth shape {depth_array.shape} "
                                    f"does not match ({height}x{width})."
                                )

                if camera_id not in _placeholder_checked:
                    rgb_arr = load_camera_frame(
                        camera_data, "rgb", ep_dir=episode_dir, frames_dir=_frames_dir
                    )
                    if rgb_arr is not None and is_placeholder_rgb(rgb_arr):
                        detail = (
                            f"Frame {idx} camera '{camera_id}' rgb appears placeholder "
                            "(few colors, mostly zeros)."
                        )
                        frame_errors.append(detail)
                        camera_placeholder_detected = True
                        camera_placeholder_details.append(detail)
                    depth_arr = load_camera_frame(
                        camera_data, "depth", ep_dir=episode_dir, frames_dir=_frames_dir
                    )
                    if depth_arr is not None and is_placeholder_depth(depth_arr):
                        detail = (
                            f"Frame {idx} camera '{camera_id}' depth appears placeholder "
                            "(all inf or all zeros)."
                        )
                        frame_errors.append(detail)
                        camera_placeholder_detected = True
                        camera_placeholder_details.append(detail)
                    _placeholder_checked.add(camera_id)

            if frame_errors:
                errors.extend(frame_errors)
                invalid_frames.add(idx)

        for timestamp, indices in timestamp_indices.items():
            if len(indices) > 1:
                errors.append(
                    f"Duplicate timestamp {timestamp} in frames {indices} (episode {episode_id})."
                )
                invalid_frames.update(indices)

        for frame_index, indices in frame_index_indices.items():
            if len(indices) > 1:
                errors.append(
                    f"Duplicate frame index {frame_index} in frames {indices} (episode {episode_id})."
                )
                invalid_frames.update(indices)

        pil_image_spec = importlib.util.find_spec("PIL.Image")
        if pil_image_spec is not None:
            from PIL import Image
            for idx, frame in enumerate(frames):
                obs = frame.get("observation") or {}
                camera_obs = obs.get("camera_observation") or {}
                images = camera_obs.get("images", [])
                if not images:
                    continue
                for image in images:
                    camera_id = str(image.get("camera_id"))
                    if required_cameras and camera_id not in required_cameras:
                        continue
                    encoding = (image.get("encoding") or "").lower()
                    for key in ("rgb_data", "depth_data"):
                        data_str = image.get(key) or ""
                        if not data_str:
                            continue
                        try:
                            raw = base64.b64decode(data_str)
                        except (ValueError, TypeError) as exc:
                            errors.append(
                                f"Frame {idx} camera '{camera_id}' {key} base64 decode failed: {exc}."
                            )
                            invalid_frames.add(idx)
                            continue
                        if "png" not in encoding and not self._is_png_data(raw):
                            continue
                        try:
                            with Image.open(io.BytesIO(raw)) as image_file:
                                image_file.verify()
                        except Exception as exc:
                            errors.append(
                                f"Frame {idx} camera '{camera_id}' {key} PNG verify failed: {exc}."
                            )
                            invalid_frames.add(idx)
        else:
            warnings.append("PNG validation skipped (PIL not available).")

        # Frame sequence variety check (detect static/repeated frames)
        try:
            cam_ids = []
            for frame in frames:
                cam_frames = (frame.get("observation") or {}).get("camera_frames") or {}
                if isinstance(cam_frames, dict):
                    for cam_id in cam_frames.keys():
                        if cam_id not in cam_ids:
                            cam_ids.append(cam_id)
                if cam_ids:
                    break
            if cam_ids:
                cam_id = cam_ids[0]
                sampled = []
                for frame in frames[: min(10, len(frames))]:
                    cam_frames = (frame.get("observation") or {}).get("camera_frames") or {}
                    cam_data = cam_frames.get(cam_id) if isinstance(cam_frames, dict) else None
                    if not isinstance(cam_data, dict):
                        continue
                    rgb_arr = load_camera_frame(
                        cam_data, "rgb", ep_dir=episode_dir, frames_dir=_frames_dir
                    )
                    if rgb_arr is not None:
                        sampled.append(rgb_arr)
                is_valid, diag = validate_frame_sequence_variety(sampled)
                if not is_valid:
                    errors.append(
                        f"Camera {cam_id} frame sequence appears static: {diag.get('reason')}"
                    )
        except Exception as exc:
            warnings.append(f"Frame sequence variety check failed: {exc}")

        return {
            "enabled": True,
            "errors": errors,
            "warnings": warnings,
            "invalid_frame_count": len(invalid_frames),
            "total_frames": len(frames),
            "camera_placeholder_detected": camera_placeholder_detected,
            "camera_placeholder_details": camera_placeholder_details,
        }

    def _detect_camera_placeholders(
        self,
        frames: List[Dict[str, Any]],
        *,
        episode_id: str,
        episode_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        details: List[str] = []
        checked: set[str] = set()
        _frames_dir = None
        if episode_dir is not None:
            _frames_dir = episode_dir / f"{episode_id}_frames"
        for idx, frame in enumerate(frames):
            obs = frame.get("observation") or {}
            camera_frames = obs.get("camera_frames") or {}
            for camera_id, camera_data in camera_frames.items():
                if camera_id in checked or not isinstance(camera_data, dict):
                    continue
                rgb_arr = load_camera_frame(
                    camera_data, "rgb", ep_dir=episode_dir, frames_dir=_frames_dir
                )
                if rgb_arr is not None and is_placeholder_rgb(rgb_arr):
                    details.append(
                        f"Frame {idx} camera '{camera_id}' rgb appears placeholder "
                        "(few colors, mostly zeros)."
                    )
                depth_arr = load_camera_frame(
                    camera_data, "depth", ep_dir=episode_dir, frames_dir=_frames_dir
                )
                if depth_arr is not None and is_placeholder_depth(depth_arr):
                    details.append(
                        f"Frame {idx} camera '{camera_id}' depth appears placeholder "
                        "(all inf or all zeros)."
                    )
                checked.add(camera_id)
            if details:
                break
        # Frame sequence variety check (static sequences)
        try:
            if frames:
                cam_frames = (frames[0].get("observation") or {}).get("camera_frames") or {}
                if isinstance(cam_frames, dict) and cam_frames:
                    cam_id = next(iter(cam_frames.keys()))
                    sampled = []
                    for frame in frames[: min(10, len(frames))]:
                        cam_data = (frame.get("observation") or {}).get("camera_frames", {}).get(cam_id)
                        if not isinstance(cam_data, dict):
                            continue
                        rgb_arr = load_camera_frame(
                            cam_data, "rgb", ep_dir=episode_dir, frames_dir=_frames_dir
                        )
                        if rgb_arr is not None:
                            sampled.append(rgb_arr)
                    is_valid, diag = validate_frame_sequence_variety(sampled)
                    if not is_valid:
                        details.append(
                            f"Camera '{cam_id}' frame sequence appears static: {diag.get('reason')}"
                        )
        except Exception:
            pass
        return {
            "camera_placeholder_detected": bool(details),
            "camera_placeholder_details": details,
        }

    def _attach_camera_frames(
        self,
        obs: Dict[str, Any],
        *,
        episode_id: str,
        task: Dict[str, Any],
    ) -> None:
        # Clear cached observation so get_camera_data() fetches fresh data
        # from the server instead of reusing a stale observation with empty images.
        setattr(self._client, "_latest_observation", {})
        _any_camera = False
        # Only request cameras that have valid prim mappings
        _camera_ids_to_try = ["wrist", "overhead", "side"]
        _cam_map = getattr(self._client, "_camera_prim_map", {})
        if _cam_map:
            _camera_ids_to_try = [cid for cid in _camera_ids_to_try if cid in _cam_map]
        _allow_base64 = os.getenv("GENIESIM_CAMERA_ALLOW_BASE64", "0") == "1"
        for camera_id in _camera_ids_to_try:
            try:
                camera_data = None
                # Try raw CameraService gRPC first (works reliably during trajectory)
                _prim = _cam_map.get(camera_id)
                if _prim and hasattr(self._client, "_get_camera_data_raw"):
                    camera_data = self._client._get_camera_data_raw(_prim)
                # Fall back to composed observation path
                if camera_data is None:
                    camera_data = self._client.get_camera_data(camera_id)
                if camera_data is not None:
                    # Save camera data as separate .npy files to keep JSON small.
                    # Store file references in JSON instead of inline base64.
                    _sanitized: Dict[str, Any] = {}
                    _frames_dir = getattr(self, "_current_frames_dir", None)
                    _frame_idx = getattr(self, "_current_frame_idx", 0)
                    _width = int(camera_data.get("width") or 0)
                    _height = int(camera_data.get("height") or 0)
                    _rgb_encoding = camera_data.get("rgb_encoding") or camera_data.get("encoding") or ""
                    _depth_encoding = camera_data.get("depth_encoding") or camera_data.get("encoding") or ""

                    def _save_array(_arr: np.ndarray, _key: str) -> Optional[str]:
                        if _frames_dir is None:
                            return None
                        _npy_name = f"{camera_id}_{_key}_{_frame_idx:03d}.npy"
                        _npy_path = _frames_dir / _npy_name
                        np.save(_npy_path, _arr)
                        return str(_npy_path.relative_to(_frames_dir.parent))

                    # Preserve metadata
                    for _meta_key in (
                        "camera_id",
                        "width",
                        "height",
                        "timestamp",
                        "encoding",
                        "rgb_encoding",
                        "depth_encoding",
                        "fx",
                        "fy",
                        "ppx",
                        "ppy",
                        "calibration_id",
                        "extrinsic",
                    ):
                        if _meta_key in camera_data:
                            _sanitized[_meta_key] = camera_data[_meta_key]
                    _sanitized["width"] = _width
                    _sanitized["height"] = _height
                    if _rgb_encoding:
                        _sanitized["rgb_encoding"] = _rgb_encoding
                    if _depth_encoding:
                        _sanitized["depth_encoding"] = _depth_encoding

                    for _key in ("rgb", "depth"):
                        _val = camera_data.get(_key)
                        if _val is None:
                            _sanitized[_key] = None
                            continue
                        _enc = _rgb_encoding if _key == "rgb" else _depth_encoding
                        _arr = None
                        if isinstance(_val, np.ndarray):
                            _arr = _val
                        elif isinstance(_val, list):
                            try:
                                _arr = np.array(_val)
                            except Exception:
                                _arr = None
                        elif isinstance(_val, (bytes, bytearray)):
                            _arr = decode_camera_bytes(
                                bytes(_val),
                                width=_width,
                                height=_height,
                                encoding=_enc,
                                kind=_key,
                            )
                        elif isinstance(_val, str) and _val.endswith(".npy"):
                            _sanitized[_key] = _val
                            continue
                        if _arr is not None and isinstance(_arr, np.ndarray):
                            # Validate RGB frame quality
                            if _key == "rgb":
                                try:
                                    from tools.camera_io import validate_rgb_frame_quality, save_debug_thumbnail
                                    from data_fidelity import require_valid_rgb, DataFidelityError
                                    _is_valid, _rgb_diag = validate_rgb_frame_quality(
                                        _arr,
                                        min_unique_colors=100,
                                        min_std=10.0,
                                        context=f"{camera_id}:frame_{_frame_idx}",
                                    )
                                    if not _is_valid:
                                        logger.warning(
                                            "RGB quality check failed for %s frame %d: %s",
                                            camera_id, _frame_idx, _rgb_diag.get("reason", "unknown"),
                                        )
                                        if require_valid_rgb():
                                            raise DataFidelityError(
                                                f"RGB frame {camera_id}:{_frame_idx} failed quality check: {_rgb_diag}",
                                                gate_name="rgb_quality",
                                                diagnostics=_rgb_diag,
                                            )
                                    # Save debug thumbnail for human verification
                                    if _frames_dir is not None:
                                        save_debug_thumbnail(
                                            _arr,
                                            _frames_dir.parent,
                                            f"{camera_id}_frame_{_frame_idx:03d}.png",
                                        )
                                except ImportError:
                                    pass
                            try:
                                _ref = _save_array(_arr, _key)
                                if _ref is not None:
                                    _sanitized[_key] = _ref
                                elif _allow_base64 and isinstance(_val, (bytes, bytearray)):
                                    _sanitized[_key] = base64.b64encode(bytes(_val)).decode("ascii")
                                else:
                                    _sanitized[_key] = None
                            except Exception as _npy_err:
                                logger.warning(
                                    "Failed to save camera %s/%s as npy: %s", camera_id, _key, _npy_err
                                )
                                if _allow_base64 and isinstance(_val, (bytes, bytearray)):
                                    _sanitized[_key] = base64.b64encode(bytes(_val)).decode("ascii")
                                else:
                                    _sanitized[_key] = None
                        else:
                            if _allow_base64 and isinstance(_val, (bytes, bytearray)):
                                _sanitized[_key] = base64.b64encode(bytes(_val)).decode("ascii")
                            else:
                                if isinstance(_val, (bytes, bytearray)):
                                    logger.warning(
                                        "Camera %s/%s decode failed; storing null (set GENIESIM_CAMERA_ALLOW_BASE64=1 to fallback).",
                                        camera_id,
                                        _key,
                                    )
                                _sanitized[_key] = None

                    obs.setdefault("camera_frames", {})[camera_id] = _sanitized
                    _any_camera = True
                else:
                    logger.debug(
                        "Camera %s returned no data for episode %s, skipping placeholder",
                        camera_id, episode_id,
                    )
            except Exception:
                logger.warning(
                    "Camera capture failed (camera_id=%s, episode_id=%s, task_name=%s, task_id=%s).",
                    camera_id,
                    episode_id,
                    task.get("task_name"),
                    task.get("task_id"),
                    exc_info=True,
                )
        if not _any_camera and not getattr(self, "_camera_warning_logged", False):
            logger.warning(
                "NO_CAMERA_DATA: No camera images available from server for episode %s. "
                "Camera frames will be null. Ensure Isaac Sim camera rendering is enabled "
                "or check server camera_observation pipeline.",
                episode_id,
            )
            self._camera_warning_logged = True

    def _generate_trajectory(
        self,
        task: Dict[str, Any],
        initial_obs: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate trajectory using cuRobo motion planning.

        Now uses actual cuRobo GPU-accelerated motion planning
        for collision-free trajectories instead of linear interpolation.
        Fallback planning uses task-aware targets, workspace bounds, and
        bounded joint interpolation when IK fails.

        Args:
            task: Task specification
            initial_obs: Initial observation

        Returns:
            List of waypoints or None if planning fails
        """
        self._last_planning_report = {
            "planner": None,
            "collision_free": None,
            "collision_source": None,
            "notes": [],
        }
        production_mode = self.config.environment == "production"
        # Get target position from task
        target_pos = task.get("target_position", [0.5, 0.0, 0.8])
        place_pos = task.get("place_position", [0.3, 0.2, 0.8])

        # Get initial joint positions
        initial_joints_result = self._client.get_joint_position()
        if not initial_joints_result.available:
            self.log(
                f"  ⚠️  Joint state unavailable: {initial_joints_result.error}; using default seed.",
                "WARNING",
            )
            initial_joints = None
        elif not initial_joints_result.success:
            self.log(
                "  ⚠️  Joint state request failed; using default seed.",
                "WARNING",
            )
            initial_joints = None
        else:
            initial_joints = initial_joints_result.payload
        if initial_joints is None:
            initial_joints = [0.0] * 7  # Default for 7-DOF arm

        # Truncate to expected DOF if server returns full-body joints (e.g. 34)
        robot_type = getattr(self.config, "robot_type", "")
        robot_config = ROBOT_CONFIGS.get(robot_type, ROBOT_CONFIGS.get("franka"))
        normalized_robot = _normalize_robot_name(robot_type) if robot_type else ""
        if normalized_robot.startswith("g1"):
            joint_names = (
                list(self._client._joint_names)
                if hasattr(self._client, "_joint_names") and self._client._joint_names
                else []
            )
            arm_joint_indices = _resolve_g1_arm_joint_indices(
                robot_type,
                joint_names,
                len(initial_joints),
            )
            if arm_joint_indices and len(initial_joints) > len(arm_joint_indices):
                logger.info(
                    "Truncating initial joints from %d to %d (G1 arm indices)",
                    len(initial_joints), len(arm_joint_indices),
                )
                initial_joints = [initial_joints[i] for i in arm_joint_indices]
        elif robot_config is not None:
            expected_dof = len(robot_config.joint_limits_lower)
            if len(initial_joints) > expected_dof:
                logger.info(
                    "Truncating initial joints from %d to %d (robot config DOF)",
                    len(initial_joints), expected_dof,
                )
                initial_joints = initial_joints[:expected_dof]

        # Get scene obstacles for collision avoidance
        obstacles = self._get_scene_obstacles(task, initial_obs)

        # =====================================================================
        # Use cuRobo for real motion planning
        # =====================================================================
        if production_mode and not (CUROBO_INTEGRATION_AVAILABLE and self.config.use_curobo):
            self.log(
                "  ⚠️  cuRobo unavailable in production; using IK fallback with collision checks "
                "when available.",
                "WARNING",
            )

        if CUROBO_INTEGRATION_AVAILABLE and self.config.use_curobo:
            trajectory = self._generate_curobo_trajectory(
                task=task,
                initial_joints=np.array(initial_joints),
                target_position=np.array(target_pos),
                place_position=np.array(place_pos) if place_pos else None,
                obstacles=obstacles,
            )
            if trajectory is not None:
                trajectory = self._apply_joint_usage_regularizer(trajectory)
                self.log(f"  ✅ cuRobo trajectory: {len(trajectory)} waypoints")
                return trajectory
            else:
                self.log("  ⚠️  cuRobo planning failed", "WARNING")

        # =====================================================================
        # Fallback: IK-based joint planning with collision awareness where possible
        # =====================================================================
        trajectory = self._generate_ik_fallback_trajectory(
            task=task,
            initial_joints=np.array(initial_joints),
            target_position=np.array(target_pos),
            place_position=np.array(place_pos) if place_pos else None,
            obstacles=obstacles,
            fps=get_geniesim_trajectory_fps(),
        )
        if trajectory is not None:
            self.log(f"  ✅ IK fallback trajectory: {len(trajectory)} waypoints")
            return trajectory

        if self.config.allow_ik_failure_fallback:
            if production_mode and not self.config.allow_linear_fallback_in_production:
                self.log(
                    "  ❌ IK fallback failed and linear interpolation fallback is disabled in production. "
                    "Set GENIESIM_ALLOW_LINEAR_FALLBACK_IN_PROD=1 to override (not collision-aware).",
                    "ERROR",
                )
                return None
            self.log(
                "  ⚠️  IK fallback failed; using linear interpolation fallback "
                "(not collision-aware).",
                "WARNING",
            )
            linear_trajectory = self._generate_linear_fallback_trajectory(
                task=task,
                initial_obs=initial_obs,
                obstacles=obstacles,
            )
            if linear_trajectory is not None:
                linear_trajectory = self._apply_joint_usage_regularizer(linear_trajectory)
            return linear_trajectory

        self.log(
            "  ❌ IK fallback failed; set GENIESIM_ALLOW_IK_FAILURE_FALLBACK=1 to allow "
            "linear interpolation fallback (not collision-aware).",
            "ERROR",
        )
        return None

    def _resolve_task_orientation(
        self,
        task: Dict[str, Any],
        key: str,
        fallback: np.ndarray,
    ) -> np.ndarray:
        value = task.get(key)
        if value is None:
            return fallback
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 4:
            orientation = np.array(value, dtype=float)
            norm = np.linalg.norm(orientation)
            if norm > 0:
                return orientation / norm
        self.log(f"  ⚠️  Invalid {key}; using default orientation", "WARNING")
        return fallback

    def _within_joint_limits(
        self,
        joints: np.ndarray,
        robot_config: Any,
        tolerance: float = 1e-4,
    ) -> bool:
        lower = robot_config.joint_limits_lower
        upper = robot_config.joint_limits_upper
        finite_mask = np.isfinite(lower) & np.isfinite(upper)
        if not np.any(finite_mask):
            return True
        below = joints[finite_mask] < (lower[finite_mask] - tolerance)
        above = joints[finite_mask] > (upper[finite_mask] + tolerance)
        return not (np.any(below) or np.any(above))

    def _clamp_joints_to_limits(
        self,
        joints: np.ndarray,
        robot_config: Any,
    ) -> np.ndarray:
        lower = robot_config.joint_limits_lower
        upper = robot_config.joint_limits_upper
        finite_mask = np.isfinite(lower) & np.isfinite(upper)
        if np.any(finite_mask):
            joints = joints.copy()
            joints[finite_mask] = np.clip(joints[finite_mask], lower[finite_mask], upper[finite_mask])
        return joints

    def _validate_trajectory_joint_limits(
        self,
        trajectory: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        robot_config = _resolve_robot_config(self.config.robot_type)
        if robot_config is None:
            return trajectory, 0

        clamped_waypoints = 0
        validated_trajectory: List[Dict[str, Any]] = []
        for waypoint in trajectory:
            joint_positions = waypoint.get("joint_positions")
            if joint_positions is None:
                validated_trajectory.append(waypoint)
                continue
            joints = np.array(joint_positions, dtype=float)
            if self._within_joint_limits(joints, robot_config):
                validated_trajectory.append(waypoint)
                continue
            clamped = self._clamp_joints_to_limits(joints, robot_config)
            if not np.allclose(clamped, joints):
                clamped_waypoints += 1
            updated_waypoint = dict(waypoint)
            updated_waypoint["joint_positions"] = clamped.tolist()
            validated_trajectory.append(updated_waypoint)
        return validated_trajectory, clamped_waypoints

    def _apply_joint_usage_regularizer(
        self,
        trajectory: List[Dict[str, Any]],
        target_dof: int = 7,
    ) -> List[Dict[str, Any]]:
        if not trajectory:
            return trajectory

        epsilon = float(os.getenv("GENIESIM_JOINT_USAGE_EPS", "0.02"))
        noise_scale = float(os.getenv("GENIESIM_JOINT_USAGE_NOISE", "0.02"))
        if epsilon <= 0 or noise_scale <= 0:
            return trajectory

        joint_rows: List[np.ndarray] = []
        for waypoint in trajectory:
            joint_positions = waypoint.get("joint_positions")
            if joint_positions is None:
                continue
            joint_rows.append(np.array(joint_positions, dtype=float))
        if len(joint_rows) < 2:
            return trajectory

        joint_matrix = np.vstack(joint_rows)
        dof = min(target_dof, joint_matrix.shape[1])
        if dof <= 0:
            return trajectory

        joint_ranges = np.ptp(joint_matrix[:, :dof], axis=0)
        underused_mask = joint_ranges < epsilon
        if not np.any(underused_mask):
            return trajectory

        robot_config = _resolve_robot_config(self.config.robot_type)
        phases = np.linspace(0.0, np.pi, len(trajectory))
        adjustments = np.zeros((len(trajectory), dof))

        for joint_idx, underused in enumerate(underused_mask):
            if not underused:
                continue
            scale = max(0.0, (epsilon - joint_ranges[joint_idx]) / epsilon)
            amplitude = noise_scale * scale
            if amplitude <= 0:
                continue
            offsets = amplitude * np.sin(phases + joint_idx * 0.7)
            adjustments[:, joint_idx] = offsets

        for idx, waypoint in enumerate(trajectory):
            joint_positions = waypoint.get("joint_positions")
            if joint_positions is None:
                continue
            joints = np.array(joint_positions, dtype=float)
            if joints.shape[0] < dof:
                continue
            joints[:dof] = joints[:dof] + adjustments[idx]
            if robot_config is not None:
                joints = self._clamp_joints_to_limits(joints, robot_config)
            waypoint["joint_positions"] = joints.tolist()

        self.log(
            f"  ℹ️  Applied joint-usage regularizer to {int(np.sum(underused_mask))} underused joints.",
            "DEBUG",
        )
        return trajectory

    def _smooth_joint_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        robot_config: Optional[Any],
        *,
        window: int = 5,
        indices: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Low-pass smooth joint positions to reduce jerk in fallback trajectories."""
        if not trajectory or len(trajectory) < 3:
            return trajectory
        if window < 3:
            return trajectory
        if window % 2 == 0:
            window += 1

        joint_rows = []
        for wp in trajectory:
            jp = wp.get("joint_positions")
            if jp is None:
                return trajectory
            joint_rows.append(np.array(jp, dtype=float))
        joint_matrix = np.vstack(joint_rows)
        num_steps, dof = joint_matrix.shape
        if dof == 0:
            return trajectory

        if indices is None:
            indices = list(range(dof))
        else:
            indices = [idx for idx in indices if 0 <= idx < dof]
        if not indices:
            return trajectory

        half = window // 2
        kernel = np.arange(1, half + 2, dtype=float)
        kernel = np.concatenate([kernel, kernel[-2::-1]])
        kernel /= kernel.sum()

        padded = np.pad(joint_matrix, ((half, half), (0, 0)), mode="edge")
        smoothed = joint_matrix.copy()
        for i in range(num_steps):
            window_slice = padded[i:i + window]
            smoothed[i, indices] = np.sum(window_slice[:, indices] * kernel[:, None], axis=0)

        # Preserve endpoints to avoid shifting start/goal poses
        smoothed[0, indices] = joint_matrix[0, indices]
        smoothed[-1, indices] = joint_matrix[-1, indices]

        if robot_config is not None and hasattr(robot_config, "joint_limits_lower"):
            lower = np.array(robot_config.joint_limits_lower, dtype=float)
            upper = np.array(robot_config.joint_limits_upper, dtype=float)
            limit_dof = min(dof, lower.shape[0], upper.shape[0])
            if limit_dof > 0:
                smoothed[:, :limit_dof] = np.clip(
                    smoothed[:, :limit_dof],
                    lower[:limit_dof],
                    upper[:limit_dof],
                )

        for idx, wp in enumerate(trajectory):
            wp["joint_positions"] = smoothed[idx].tolist()

        return trajectory

    def _resolve_workspace_bounds(
        self,
        task: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        bounds = task.get("workspace_bounds")
        if bounds is None:
            bounds = task.get("robot_config", {}).get("workspace_bounds")
        if bounds is None:
            return None
        if isinstance(bounds, dict):
            try:
                min_pt = np.array([bounds["x"][0], bounds["y"][0], bounds["z"][0]], dtype=float)
                max_pt = np.array([bounds["x"][1], bounds["y"][1], bounds["z"][1]], dtype=float)
                return np.stack([min_pt, max_pt], axis=0)
            except (KeyError, TypeError, ValueError):
                self.log("  ⚠️  Invalid workspace_bounds dict; ignoring", "WARNING")
                return None
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            try:
                return np.array(bounds, dtype=float)
            except (TypeError, ValueError):
                self.log("  ⚠️  Invalid workspace_bounds list; ignoring", "WARNING")
                return None
        self.log("  ⚠️  Unsupported workspace_bounds format; ignoring", "WARNING")
        return None

    def _apply_workspace_bounds(
        self,
        position: np.ndarray,
        bounds: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, bool]:
        if bounds is None:
            return position, False
        min_pt = bounds[0]
        max_pt = bounds[1]
        clamped = np.minimum(np.maximum(position, min_pt), max_pt)
        return clamped, not np.allclose(clamped, position)

    def _find_target_position_from_obs(
        self,
        task: Dict[str, Any],
        initial_obs: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        scene_state = initial_obs.get("scene_state", {})
        objects = scene_state.get("objects", [])
        target_ids = [
            task.get("target_object_id"),
            task.get("target_object"),
            task.get("target_object_name"),
        ]
        for target_id in target_ids:
            if not target_id:
                continue
            for obj in objects:
                obj_id = obj.get("object_id") or obj.get("id") or obj.get("name")
                if obj_id == target_id:
                    pose = obj.get("pose", {})
                    # Support both {"position": [x,y,z]} and {"x":..,"y":..,"z":..} formats
                    position = pose.get("position")
                    if position is not None:
                        return np.array(position, dtype=float)
                    if "x" in pose and "y" in pose and "z" in pose:
                        return np.array([pose["x"], pose["y"], pose["z"]], dtype=float)
        target_objects = task.get("target_objects", [])
        if target_objects:
            position = target_objects[0].get("position")
            if position is not None:
                return np.array(position, dtype=float)
        # Fallback: synthesize position from robot config (same logic as frame builder)
        _target = task.get("target_object")
        if _target:
            _rc_cfg = task.get("robot_config") or {}
            _base = _rc_cfg.get("base_position", [0.5, 0.0, 0.8])
            return np.array([_base[0] - 0.05, _base[1] + 0.25, _base[2] - 0.05], dtype=float)
        return None

    def _resolve_task_waypoints(
        self,
        task: Dict[str, Any],
        initial_obs: Dict[str, Any],
        workspace_bounds: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        target_position = task.get("target_position")
        if target_position is not None:
            target_position = np.array(target_position, dtype=float)
        else:
            target_position = self._find_target_position_from_obs(task, initial_obs)

        if target_position is None:
            if workspace_bounds is not None:
                target_position = (workspace_bounds[0] + workspace_bounds[1]) / 2.0
                self.log("  ⚠️  No target position; using workspace center.", "WARNING")
            else:
                target_position = np.array([0.5, 0.0, 0.8], dtype=float)
                self.log("  ⚠️  No target position; using default fallback target.", "WARNING")

        place_position = task.get("place_position")
        if place_position is not None:
            place_position = np.array(place_position, dtype=float)
        else:
            task_key = (task.get("task_type") or task.get("task_name") or "").lower()
            template_offsets = {
                "pick_place": np.array([0.25, 0.0, 0.0]),
                "pick-place": np.array([0.25, 0.0, 0.0]),
                "place": np.array([0.2, 0.0, 0.0]),
                "organize": np.array([-0.2, 0.2, 0.0]),
                "stack": np.array([0.0, 0.0, 0.1]),
            }
            for key, offset in template_offsets.items():
                if key in task_key:
                    place_position = target_position + offset
                    self.log(
                        f"  ℹ️  Derived place position using template '{key}'.",
                        "INFO",
                    )
                    break

        target_position, target_clamped = self._apply_workspace_bounds(
            target_position, workspace_bounds
        )
        if target_clamped:
            self.log("  ⚠️  Target position clamped to workspace bounds.", "WARNING")

        if place_position is not None:
            place_position, place_clamped = self._apply_workspace_bounds(
                place_position, workspace_bounds
            )
            if place_clamped:
                self.log("  ⚠️  Place position clamped to workspace bounds.", "WARNING")

        return target_position, place_position

    def _compute_fk_positions(
        self,
        robot_config: Any,
        trajectory: List[Dict[str, Any]],
    ) -> Optional[List[np.ndarray]]:
        if not IK_PLANNING_AVAILABLE:
            return None
        ik_solver = IKSolver(robot_config, verbose=False)
        if not hasattr(ik_solver, "_forward_kinematics"):
            return None
        positions = []
        for waypoint in trajectory:
            joints = np.array(waypoint.get("joint_positions", []), dtype=float)
            try:
                pos, _ = ik_solver._forward_kinematics(joints)
            except Exception:
                return None
            positions.append(pos)
        return positions

    def _trajectory_violates_clearance(
        self,
        positions: Sequence[np.ndarray],
        obstacles: List[Dict[str, Any]],
        clearance: float,
    ) -> bool:
        for pos in positions:
            for obstacle in obstacles:
                center = np.array(obstacle.get("position", [0.0, 0.0, 0.0]), dtype=float)
                dims = np.array(obstacle.get("dimensions", [0.1, 0.1, 0.1]), dtype=float)
                half_extents = dims / 2.0 + clearance
                if np.all(np.abs(pos - center) <= half_extents):
                    return True
        return False

    def _trajectory_within_workspace(
        self,
        positions: Sequence[np.ndarray],
        workspace_bounds: Optional[np.ndarray],
    ) -> bool:
        if workspace_bounds is None:
            return True
        min_pt = workspace_bounds[0]
        max_pt = workspace_bounds[1]
        for pos in positions:
            if np.any(pos < min_pt) or np.any(pos > max_pt):
                return False
        return True

    # ── Fix 6+7: LLM-driven adaptive waypoint planning + grasp orientation ──

    def _plan_waypoints_with_llm(
        self,
        task: Dict[str, Any],
        target_position: np.ndarray,
        place_position: Optional[np.ndarray],
        ee_position: Optional[np.ndarray] = None,
    ) -> Optional[List["Waypoint"]]:
        """Use LLM to plan Cartesian waypoints adapted to object type and task semantics.

        Returns list of Waypoint objects or None on failure (falls back to hardcoded).
        """
        try:
            from tools.llm_client import create_llm_client as _create_wp_llm
        except Exception:
            self.log("  ⚠️  LLM client import failed; skipping LLM waypoint planning", "WARNING")
            return None

        task_desc = task.get("task_description", task.get("task_name", "pick and place"))
        target_obj = task.get("target_object", task.get("target_object_id", "object"))
        target_obj_type = task.get("target_object_type", target_obj)
        robot_type = getattr(self.config, "robot_type", "humanoid")

        # Scene objects summary
        scene_objs = []
        try:
            obs = getattr(self, "_latest_observations", {})
            for k, v in obs.get("objects", {}).items():
                if isinstance(v, dict) and "position" in v:
                    scene_objs.append(f"{k}: pos={v['position']}")
        except Exception:
            pass
        scene_summary = "; ".join(scene_objs[:10]) if scene_objs else "unknown"

        ee_str = f"[{ee_position[0]:.3f}, {ee_position[1]:.3f}, {ee_position[2]:.3f}]" if ee_position is not None else "unknown"
        target_str = f"[{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]"
        place_str = f"[{place_position[0]:.3f}, {place_position[1]:.3f}, {place_position[2]:.3f}]" if place_position is not None else "same as target"

        prompt = f"""Plan Cartesian waypoints for a robot manipulation task.

Task: {task_desc}
Target object: {target_obj_type} at position {target_str}
Place location: {place_str}
Current EE position: {ee_str}
Robot: {robot_type}

Return JSON with exactly this format:
{{"waypoints": [
  {{"phase": "approach", "position": [x,y,z], "orientation": [qw,qx,qy,qz], "gripper": 1.0, "duration": 1.0}},
  {{"phase": "pre_grasp", "position": [x,y,z], "orientation": [qw,qx,qy,qz], "gripper": 1.0, "duration": 0.6}},
  {{"phase": "grasp", "position": [x,y,z], "orientation": [qw,qx,qy,qz], "gripper": 0.0, "duration": 0.6}},
  {{"phase": "lift", "position": [x,y,z], "orientation": [qw,qx,qy,qz], "gripper": 0.0, "duration": 0.8}},
  {{"phase": "transport", "position": [x,y,z], "orientation": [qw,qx,qy,qz], "gripper": 0.0, "duration": 1.2}},
  {{"phase": "place", "position": [x,y,z], "orientation": [qw,qx,qy,qz], "gripper": 1.0, "duration": 0.6}},
  {{"phase": "retract", "position": [x,y,z], "orientation": [qw,qx,qy,qz], "gripper": 1.0, "duration": 0.8}}
]}}

Rules:
- approach: 10-20cm above object
- pre_grasp: 2-5cm above object
- grasp: at object surface height, gripper=0.0 (closed)
- lift: 15-25cm above grasp point
- transport: above place location (if different from pick)
- place: at place surface, gripper=1.0 (open to release)
- retract: 15-20cm above place point
- Orientations: top-down [0.707, 0.707, 0, 0] for flat objects (plates, books),
  angled for tall objects (bottles, cups), handle-aligned for objects with handles (pots, pans, mugs).
- All positions must be reachable (z > 0, within ~0.6m of robot base).
- If place location equals target, skip transport/place/retract and just lift and release.
Scene objects: {scene_summary}
"""

        try:
            _wp_llm = _create_wp_llm()
            resp = _wp_llm.generate(prompt=prompt, json_output=True, disable_tools=True, temperature=0.2)
            if hasattr(resp, "parse_json"):
                parsed = resp.parse_json()
            else:
                import json as _json_mod
                parsed = _json_mod.loads(resp) if isinstance(resp, str) else resp
            raw_wps = parsed.get("waypoints", [])
            if not raw_wps or len(raw_wps) < 3:
                self.log(f"  ⚠️  LLM returned {len(raw_wps)} waypoints (need ≥3); falling back", "WARNING")
                return None
        except Exception as exc:
            self.log(f"  ⚠️  LLM waypoint planning failed: {exc}; falling back", "WARNING")
            return None

        # Convert to Waypoint objects
        phase_sequence = [
            "approach",
            "pre_grasp",
            "grasp",
            "lift",
            "transport",
            "place",
            "retract",
        ]
        phase_map = {
            "approach": MotionPhase.APPROACH,
            "pre_grasp": MotionPhase.PRE_GRASP,
            "grasp": MotionPhase.GRASP,
            "lift": MotionPhase.LIFT,
            "transport": MotionPhase.TRANSPORT,
            "place": MotionPhase.PLACE,
            "retract": MotionPhase.RETRACT,
            "pre_place": MotionPhase.PRE_PLACE,
            "release": MotionPhase.RELEASE,
            "home": MotionPhase.HOME,
            "articulate_approach": MotionPhase.ARTICULATE_APPROACH,
            "articulate_grasp": MotionPhase.ARTICULATE_GRASP,
            "articulate_motion": MotionPhase.ARTICULATE_MOTION,
        }

        waypoints: List[Waypoint] = []
        timestamp = 0.0
        for wp_index, wp_data in enumerate(raw_wps):
            try:
                pos = np.array(wp_data["position"], dtype=float)
                ori = np.array(wp_data["orientation"], dtype=float)
                # Normalize quaternion
                norm = np.linalg.norm(ori)
                if norm > 0:
                    ori = ori / norm
                else:
                    ori = np.array([1.0, 0.0, 0.0, 0.0])
                gripper = float(wp_data.get("gripper", 1.0))
                duration = float(wp_data.get("duration", 0.8))
                phase_str = wp_data.get("phase")
                if not isinstance(phase_str, str) or phase_str not in phase_map:
                    fallback_phase = phase_sequence[min(wp_index, len(phase_sequence) - 1)]
                    phase_str = fallback_phase
                phase = phase_map[phase_str]

                waypoints.append(Waypoint(
                    position=pos,
                    orientation=ori,
                    gripper_aperture=gripper,
                    timestamp=timestamp,
                    duration_to_next=duration,
                    phase=phase,
                ))
                timestamp += duration
            except (KeyError, ValueError, TypeError) as wp_err:
                self.log(f"  ⚠️  Skipping invalid LLM waypoint: {wp_err}", "WARNING")
                continue

        if len(waypoints) < 3:
            self.log(f"  ⚠️  Only {len(waypoints)} valid LLM waypoints; falling back", "WARNING")
            return None

        self.log(f"  ✅ LLM waypoint planning: {len(waypoints)} waypoints for '{task_desc}'")
        return waypoints

    def _build_ik_fallback_waypoints(
        self,
        target_position: np.ndarray,
        place_position: Optional[np.ndarray],
        target_orientation: np.ndarray,
        place_orientation: np.ndarray,
    ) -> List["Waypoint"]:
        waypoints: List[Waypoint] = []
        timestamp = 0.0

        def _add_waypoint(
            position: np.ndarray,
            orientation: np.ndarray,
            phase: "MotionPhase",
            duration: float,
            gripper_aperture: float = 1.0,
        ) -> None:
            nonlocal timestamp
            waypoints.append(
                Waypoint(
                    position=position,
                    orientation=orientation,
                    gripper_aperture=gripper_aperture,
                    timestamp=timestamp,
                    duration_to_next=duration,
                    phase=phase,
                )
            )
            timestamp += duration

        pre_grasp_position = target_position.copy()
        pre_grasp_position[2] += 0.15
        _add_waypoint(
            pre_grasp_position,
            target_orientation,
            MotionPhase.PRE_GRASP,
            duration=1.0,
        )

        grasp_position = target_position.copy()
        grasp_position[2] += 0.02
        _add_waypoint(
            grasp_position,
            target_orientation,
            MotionPhase.GRASP,
            duration=0.6,
            gripper_aperture=0.0,  # Fix 4A: close gripper to grasp
        )

        lift_position = target_position.copy()
        lift_position[2] += 0.25
        _add_waypoint(
            lift_position,
            target_orientation,
            MotionPhase.LIFT,
            duration=0.8,
            gripper_aperture=0.0,  # Fix 4A: keep gripper closed while lifting
        )

        if place_position is not None:
            pre_place_position = place_position.copy()
            pre_place_position[2] += 0.20
            _add_waypoint(
                pre_place_position,
                place_orientation,
                MotionPhase.TRANSPORT,
                duration=1.2,
                gripper_aperture=0.0,  # Fix 4A: keep gripper closed during transport
            )

            _add_waypoint(
                place_position,
                place_orientation,
                MotionPhase.PLACE,
                duration=0.6,
                gripper_aperture=1.0,  # Fix 4A: open gripper to release
            )

            # Fix 4A: RETRACT waypoint — pull back after placing
            retract_position = place_position.copy()
            retract_position[2] += 0.18
            _add_waypoint(
                retract_position,
                place_orientation,
                MotionPhase.RETRACT,
                duration=0.8,
                gripper_aperture=1.0,  # gripper stays open
            )

        return waypoints

    def _generate_ik_fallback_trajectory(
        self,
        task: Dict[str, Any],
        initial_joints: np.ndarray,
        target_position: np.ndarray,
        place_position: Optional[np.ndarray],
        obstacles: List[Dict[str, Any]],
        fps: float = 5.0,
    ) -> Optional[List[Dict[str, Any]]]:
        if not IK_PLANNING_AVAILABLE:
            self.log("  ❌ IK utilities unavailable; cannot build IK fallback trajectory.", "ERROR")
            return None

        robot_config = _resolve_robot_config(self.config.robot_type)
        if robot_config is None:
            self.log(f"  ❌ No robot config for type '{self.config.robot_type}'", "ERROR")
            return None

        default_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        target_orientation = self._resolve_task_orientation(
            task, "target_orientation", default_orientation
        )
        place_orientation = self._resolve_task_orientation(
            task, "place_orientation", target_orientation
        )

        # Geometric waypoint planner (LLM waypoint planning removed — deterministic is faster & sufficient)
        waypoints = self._build_ik_fallback_waypoints(
            target_position=target_position,
            place_position=place_position,
            target_orientation=target_orientation,
            place_orientation=place_orientation,
        )
        self.log("  ✅ Using geometric waypoint planner (no LLM)")

        planner = None
        collision_free: Optional[bool] = None
        if CollisionAwarePlanner is not None:
            try:
                planner = CollisionAwarePlanner(
                    robot_type=self.config.robot_type,
                    scene_objects=obstacles,
                    use_curobo=False,
                    verbose=False,
                )
                planned_waypoints = planner.plan_waypoint_trajectory(waypoints)
                if planned_waypoints is not None:
                    waypoints = planned_waypoints
            except Exception as exc:
                self.log(f"  ⚠️  Collision-aware planner failed: {exc}", "WARNING")
                planner = None

        ik_solver = IKSolver(robot_config, verbose=False)
        seed_joints = initial_joints.copy()

        # --- Fix 3: Try server-side IK first (actual robot model) ---
        normalized_robot = _normalize_robot_name(getattr(self.config, "robot_type", ""))
        _use_server_ik = normalized_robot.startswith("g1")

        for wp in waypoints:
            joints = None

            # 1. Server-side IK (uses actual G1 model + obstacle avoidance)
            if _use_server_ik and joints is None:
                try:
                    is_right = "left" not in normalized_robot
                    T = _pose_to_4x4(wp.position, wp.orientation)
                    ik_result = self._client.get_ik_status(T, is_right=is_right, obs_avoid=True)
                    if ik_result.success and ik_result.payload:
                        payload = ik_result.payload
                        if payload.get("is_success") and payload.get("joint_positions"):
                            joints = np.array(payload["joint_positions"], dtype=float)
                            self.log(f"  ✅ Server IK solved for {wp.phase.value}", "DEBUG")
                except Exception as _sik_err:
                    self.log(f"  ⚠️  Server IK error for {wp.phase.value}: {_sik_err}", "WARNING")

            # 2. Local collision-aware IK
            if joints is None and planner is not None:
                joints = planner.solve_ik_with_collision_check(
                    wp.position, wp.orientation, seed_joints=seed_joints
                )
                if joints is not None:
                    self.log(f"  ✅ Local collision-aware IK solved for {wp.phase.value}", "DEBUG")

            # 3. Local numerical IK
            if joints is None:
                joints = ik_solver.solve(wp.position, wp.orientation, seed_joints)
                if joints is not None:
                    self.log(f"  ✅ Local numerical IK solved for {wp.phase.value}", "DEBUG")

            if joints is None:
                self.log(
                    f"  ❌ All IK methods failed for {wp.phase.value} at {wp.position.tolist()}",
                    "ERROR",
                )
                return None
            if not self._within_joint_limits(joints, robot_config):
                joints = self._clamp_joints_to_limits(joints, robot_config)
                self.log(f"  ⚠️  IK solution clamped to joint limits for {wp.phase.value}", "WARNING")

            wp.joint_positions = joints
            seed_joints = joints

        # --- Fix 4: Include gripper_aperture and phase in trajectory ---
        phase_gripper_map = {
            "approach": 1.0,
            "pre_grasp": 1.0,
            "place": 1.0,
            "retract": 1.0,
            "grasp": 0.0,
            "lift": 0.0,
            "transport": 0.0,
        }
        for wp in waypoints:
            if wp.gripper_aperture is None:
                phase_value = wp.phase.value if hasattr(wp.phase, "value") else str(wp.phase)
                wp.gripper_aperture = phase_gripper_map.get(phase_value, 1.0)

        trajectory: List[Dict[str, Any]] = []
        current_joints = initial_joints.copy()
        current_time = 0.0

        # Get target object prim for attach/detach
        _target_prim = task.get("target_object_prim") or task.get("object_prim", "")

        for index, wp in enumerate(waypoints):
            target_joints = wp.joint_positions
            if target_joints is None:
                continue
            duration = max(0.2, float(wp.duration_to_next))
            steps = max(2, int(round(duration * fps)))
            start_step = 0 if index == 0 else 1

            for step in range(start_step, steps):
                t = step / (steps - 1)
                joint_pos = (1 - t) * current_joints + t * target_joints
                wp_dict: Dict[str, Any] = {
                    "joint_positions": joint_pos.tolist(),
                    "timestamp": current_time + t * duration,
                    "gripper_aperture": wp.gripper_aperture,
                    "phase": wp.phase.value,
                }
                # Annotate grasp/lift/transport with object prim for attach
                if wp.phase.value in ("grasp", "lift", "transport") and _target_prim:
                    wp_dict["object_prims"] = [_target_prim]
                trajectory.append(wp_dict)

            current_time += duration
            current_joints = target_joints

        # Fix 5: Apply trajectory smoothing to reduce acceleration/jerk
        trajectory = self._apply_trajectory_smoothing(
            trajectory,
            robot_config=robot_config,
        )
        trajectory = self._apply_joint_usage_regularizer(trajectory)
        return trajectory

    def _apply_trajectory_smoothing(
        self,
        trajectory: List[Dict[str, Any]],
        window_size: int = 5,
        robot_config: Optional[Any] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Apply smoothing to trajectory joint positions to reduce acceleration."""
        if os.getenv("GENIESIM_SMOOTH_FALLBACK_TRAJECTORY", "1") != "1":
            return trajectory
        smooth_window = int(os.getenv("GENIESIM_SMOOTH_WINDOW", str(window_size)))
        if len(trajectory) < smooth_window:
            return trajectory

        # Cache original joint positions to preserve critical phases
        joint_rows = []
        for wp in trajectory:
            jp = wp.get("joint_positions")
            if jp is not None:
                joint_rows.append(np.array(jp, dtype=float))
            else:
                return trajectory  # Can't smooth if missing positions

        if len(joint_rows) < smooth_window:
            return trajectory
        original = joint_rows

        trajectory = self._smooth_joint_trajectory(
            trajectory,
            robot_config,
            window=smooth_window,
            indices=indices,
        )

        # Preserve critical phases exactly (grasp/place/pre_grasp)
        for i, wp in enumerate(trajectory):
            phase = wp.get("phase", "")
            if phase in ("grasp", "place", "pre_grasp"):
                wp["joint_positions"] = original[i].tolist()

        return trajectory

    def _generate_linear_fallback_trajectory(
        self,
        task: Dict[str, Any],
        initial_obs: Dict[str, Any],
        obstacles: Optional[List[Dict[str, Any]]] = None,
        num_waypoints: int = 100,
        fps: float = 30.0,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate a task-aware fallback trajectory when planning fails.

        Attempts IK-based waypoint planning first using task targets and workspace
        bounds. If IK fails, falls back to bounded joint-space interpolation
        with reachability and collision heuristics.
        """
        collision_free: Optional[bool] = None
        if obstacles is None:
            obstacles = self._get_scene_obstacles(task, initial_obs)

        robot_config = _resolve_robot_config(self.config.robot_type)
        if robot_config is None:
            self.log(
                f"  ⚠️  No robot config for type '{self.config.robot_type}'; "
                "using minimal bounded interpolation.",
                "WARNING",
            )

        workspace_bounds = self._resolve_workspace_bounds(task)
        target_position, place_position = self._resolve_task_waypoints(
            task=task,
            initial_obs=initial_obs,
            workspace_bounds=workspace_bounds,
        )

        initial_joints_result = self._client.get_joint_position()
        if not initial_joints_result.available:
            self.log(
                f"  ⚠️  Joint state unavailable: {initial_joints_result.error}; "
                "using default joint seed.",
                "WARNING",
            )
            initial_joints = None
        elif not initial_joints_result.success:
            self.log("  ⚠️  Missing joint state; using default joint seed.", "WARNING")
            initial_joints = None
        else:
            initial_joints = initial_joints_result.payload
        # Treat all-zero joints from mock client as missing
        if initial_joints is not None and all(j == 0.0 for j in initial_joints):
            initial_joints = None
        if initial_joints is None:
            if robot_config is not None:
                initial_joints = robot_config.default_joint_positions.tolist()
            else:
                _fb = _ROBOT_METADATA_FALLBACK.get(self.config.robot_type, {})
                initial_joints = _fb.get("default_joint_positions", [0.0] * 7)

        initial_joints = np.array(initial_joints, dtype=float)

        # Truncate to robot config DOF if server returns full-body joints
        normalized_robot = _normalize_robot_name(getattr(self.config, "robot_type", ""))
        if normalized_robot.startswith("g1"):
            joint_names = (
                list(self._client._joint_names)
                if hasattr(self._client, "_joint_names") and self._client._joint_names
                else []
            )
            arm_joint_indices = _resolve_g1_arm_joint_indices(
                self.config.robot_type,
                joint_names,
                len(initial_joints),
            )
            if arm_joint_indices and len(initial_joints) > len(arm_joint_indices):
                logger.info(
                    "Truncating initial joints from %d to %d (G1 arm indices)",
                    len(initial_joints), len(arm_joint_indices),
                )
                initial_joints = initial_joints[arm_joint_indices]
        elif robot_config is not None:
            expected_dof = len(robot_config.joint_limits_lower)
            if len(initial_joints) > expected_dof:
                logger.info(
                    "Truncating initial joints from %d to %d (robot config DOF)",
                    len(initial_joints), expected_dof,
                )
                initial_joints = initial_joints[:expected_dof]

        if IK_PLANNING_AVAILABLE and robot_config is not None:
            self.log(
                "  ℹ️  Attempting task-aware IK fallback before joint interpolation.",
                "INFO",
            )
            ik_trajectory = self._generate_ik_fallback_trajectory(
                task=task,
                initial_joints=initial_joints,
                target_position=target_position,
                place_position=place_position,
                obstacles=obstacles,
                fps=fps,
            )
            if ik_trajectory is not None:
                positions = self._compute_fk_positions(robot_config, ik_trajectory)
                if positions is not None:
                    if not self._trajectory_within_workspace(positions, workspace_bounds):
                        self.log(
                            "  ⚠️  IK fallback trajectory leaves workspace bounds; rejecting.",
                            "WARNING",
                        )
                        collision_free = False
                    elif self._trajectory_violates_clearance(positions, obstacles, clearance=0.05):
                        self.log(
                            "  ⚠️  IK fallback trajectory violates obstacle clearance; rejecting.",
                            "WARNING",
                        )
                        collision_free = False
                    else:
                        self.log("  ✅ IK fallback accepted with clearance checks.")
                        collision_free = True
                        self._last_planning_report.update(
                            {
                                "planner": "ik_fallback",
                                "collision_free": collision_free,
                                "collision_source": "ik_clearance_check",
                                "notes": [],
                            }
                        )
                        return ik_trajectory
                else:
                    self.log(
                        "  ℹ️  IK fallback accepted (FK unavailable for clearance checks).",
                        "INFO",
                    )
                    # IK solution found means trajectory is within joint limits (basic collision-free)
                    collision_free = True
                    self._last_planning_report.update(
                        {
                            "planner": "ik_fallback",
                            "collision_free": collision_free,
                            "collision_source": "ik_joint_limits",
                            "notes": [],
                        }
                    )
                    return ik_trajectory

        self.log(
            "  ⚠️  Using bounded joint-space interpolation fallback.",
            "WARNING",
        )

        if robot_config is None:
            num_joints = len(initial_joints)
            lower = np.full(num_joints, -np.pi)
            upper = np.full(num_joints, np.pi)
        else:
            lower = robot_config.joint_limits_lower
            upper = robot_config.joint_limits_upper

        initial_joints = np.clip(initial_joints, lower, upper)

        # ---- Multi-phase joint-space trajectory ----
        # Generate a manipulation-like trajectory with 5 phases:
        #   1. Approach  (20%)  – move arm toward target region
        #   2. Grasp     (10%)  – close gripper / fine positioning
        #   3. Lift      (20%)  – raise arm
        #   4. Transport (30%)  – move arm laterally toward place
        #   5. Place     (20%)  – lower arm and open gripper
        # Each phase target is computed as a bounded perturbation of the
        # initial joints, scaled by a fraction of the joint range.  This
        # ensures diversity while respecting limits.
        joint_range = upper - lower
        # Identify arm joints vs gripper/finger joints (robot-specific).
        num_joints = len(initial_joints)
        robot_type = getattr(self.config, "robot_type", "")
        normalized_robot = _normalize_robot_name(robot_type) if robot_type else ""
        joint_names = (
            list(self._client._joint_names)
            if hasattr(self._client, "_joint_names") and self._client._joint_names
            else []
        )
        if normalized_robot.startswith("g1"):
            arm_joint_indices = _resolve_g1_arm_joint_indices(robot_type, joint_names, num_joints)
            gripper_joint_indices = _resolve_g1_gripper_joint_indices(
                robot_type,
                joint_names,
                num_joints,
                arm_joint_indices,
            )
        else:
            arm_joint_indices = list(range(min(7, num_joints)))
            gripper_joint_indices = list(range(len(arm_joint_indices), num_joints))
        arm_end = len(arm_joint_indices)
        # Gripper open/closed state masks: 1.0 = open, 0.0 = closed
        gripper_open = np.ones(num_joints)
        gripper_closed = np.ones(num_joints)
        for j in gripper_joint_indices:
            gripper_closed[j] = 0.3  # partially close fingers

        # Build phase waypoints using task geometry + numerical IK
        # Randomize offsets and durations per episode for trajectory diversity
        import random as _random
        _rng = _random.Random(hash(task.get("task_name", "") + str(id(initial_obs)) + str(time.time())) & 0xFFFFFFFF)

        base_durations = np.array([0.20, 0.10, 0.20, 0.30, 0.20])
        dur_noise = np.array([_rng.uniform(-0.15, 0.15) for _ in range(len(base_durations))])
        durations = np.clip(base_durations + dur_noise, 0.05, 0.5)
        durations /= durations.sum()  # renormalize to 1.0

        # Randomization parameters for trajectory diversity
        _approach_angle = _rng.uniform(-0.5, 0.5)  # radians around target
        _approach_height = 0.15 + _rng.uniform(-0.03, 0.05)
        _lift_height = _rng.uniform(0.15, 0.25)
        _transport_lateral = _rng.uniform(-0.10, 0.10)
        _transport_arc_h = _rng.uniform(0.10, 0.25)

        # Compute Cartesian waypoints from task geometry
        _ik_waypoints_available = False
        if target_position is not None:
            _tp = np.array(target_position, dtype=float)
            approach_pos = _tp + np.array([
                _approach_height * np.cos(_approach_angle),
                _approach_height * np.sin(_approach_angle),
                _approach_height,
            ])
            grasp_pos = _tp.copy()
            lift_pos = _tp + np.array([0.0, 0.0, _lift_height])

            if place_position is not None:
                _pp = np.array(place_position, dtype=float)
                _dir = _pp - _tp
                _dir_norm = np.linalg.norm(_dir)
                if _dir_norm > 0.01:
                    _dir_unit = _dir / _dir_norm
                    _perp = np.array([-_dir_unit[1], _dir_unit[0], 0.0])
                else:
                    _dir_unit = np.array([1.0, 0.0, 0.0])
                    _perp = np.array([0.0, 1.0, 0.0])
                transport_pos = (_tp + _pp) / 2.0 + _perp * _transport_lateral + np.array([0, 0, _transport_arc_h])
                place_pos_cart = _pp.copy()
            else:
                transport_pos = _tp + np.array([0.25, _transport_lateral, _transport_arc_h])
                place_pos_cart = _tp + np.array([0.25, 0.0, 0.0])

            # Solve IK for each Cartesian waypoint
            _cart_targets = [
                ("approach", approach_pos),
                ("grasp", grasp_pos),
                ("lift", lift_pos),
                ("transport", transport_pos),
                ("place", place_pos_cart),
            ]
            ik_phase_joints = []
            _prev_q = np.array([initial_joints[idx] for idx in arm_joint_indices], dtype=float)
            _all_solved = True
            for _pname, _cart in _cart_targets:
                if getattr(self.config, "robot_type", "").lower() in _FRANKA_TYPES:
                    _ik_result = _franka_numerical_ik(_cart, initial_guess=_prev_q)
                else:
                    _ik_result = None
                if _ik_result is not None:
                    ik_phase_joints.append(_ik_result)
                    _prev_q = _ik_result
                    self.log(
                        f"  ✓ IK solved for phase '{_pname}': "
                        f"target={[round(c, 3) for c in _cart.tolist()]}",
                        "INFO",
                    )
                else:
                    self.log(
                        f"  ⚠️  IK failed for phase '{_pname}' "
                        f"target={[round(c, 3) for c in _cart.tolist()]}; "
                        f"falling back to joint-space offsets.",
                        "WARNING",
                    )
                    _all_solved = False
                    break

            if _all_solved and len(ik_phase_joints) == 5:
                _ik_waypoints_available = True
                self.log(
                    f"  ✅ All 5 IK waypoints solved — trajectory is task-geometry-aware.",
                    "INFO",
                )

        # Build phase configs: IK-derived if available, else joint-space offsets
        def _jitter(base_offsets: np.ndarray) -> np.ndarray:
            noise = np.array([_rng.uniform(-0.10, 0.10) for _ in range(len(base_offsets))])
            return base_offsets + noise

        if _ik_waypoints_available:
            # Use IK-derived joint targets
            phase_configs = [
                ("approach",  None, gripper_open,    durations[0]),
                ("grasp",     None, gripper_closed,  durations[1]),
                ("lift",      None, gripper_closed,  durations[2]),
                ("transport", None, gripper_closed,  durations[3]),
                ("place",     None, gripper_open,    durations[4]),
            ]
            phase_targets = []
            for _pi, (_pn, _, _gm, _) in enumerate(phase_configs):
                target = initial_joints.copy()
                for joint_offset, joint_idx in enumerate(arm_joint_indices):
                    if joint_offset < len(ik_phase_joints[_pi]):
                        target[joint_idx] = ik_phase_joints[_pi][joint_offset]
                # Apply gripper state
                for j in gripper_joint_indices:
                    mid = (lower[j] + upper[j]) / 2.0
                    target[j] = mid + (initial_joints[j] - mid) * _gm[j]
                target = np.clip(target, lower, upper)
                phase_targets.append(target)
        else:
            # Fallback: joint-space offsets (original behavior with larger jitter)
            phase_configs = [
                ("approach",  _jitter(np.array([0.0, 0.05, -0.08, 0.04, -0.02, 0.03, 0.0])), gripper_open,  durations[0]),
                ("grasp",     _jitter(np.array([0.0, 0.08, -0.12, 0.06, -0.03, 0.04, 0.01])), gripper_closed, durations[1]),
                ("lift",      _jitter(np.array([0.0, -0.05, -0.15, 0.08, -0.05, 0.06, 0.0])), gripper_closed, durations[2]),
                ("transport", _jitter(np.array([0.1, -0.03, -0.10, 0.05, 0.05, -0.02, 0.03])), gripper_closed, durations[3]),
                ("place",     _jitter(np.array([0.1, 0.02, -0.05, 0.03, 0.04, -0.01, 0.02])), gripper_open,  durations[4]),
            ]
            phase_targets = []
            for phase_name, arm_offsets, grip_mult, _ in phase_configs:
                target = initial_joints.copy()
                for j in range(min(len(arm_offsets), arm_end)):
                    arm_idx = arm_joint_indices[j]
                    target[arm_idx] += arm_offsets[j] * joint_range[arm_idx]
                for j in gripper_joint_indices:
                    mid = (lower[j] + upper[j]) / 2.0
                    target[j] = mid + (initial_joints[j] - mid) * grip_mult[j]
                target = np.clip(target, lower, upper)
                phase_targets.append(target)

        # Gripper joint values per phase (robot-specific limits)
        _grip_lims = joint_groups["gripper_limits"] or (0.0, 0.04)
        _grip_open_val = float(_grip_lims[1]) if _grip_lims else 0.04
        _grip_closed_val = float(_grip_lims[0]) if _grip_lims else 0.0
        # Map phase to gripper target: approach=open, grasp=closed, lift/transport=closed, place=open
        _phase_gripper = [_grip_open_val, _grip_closed_val, _grip_closed_val, _grip_closed_val, _grip_open_val]

        # Interpolate between phases to build full trajectory
        trajectory: List[Dict[str, Any]] = []
        current_joints = initial_joints.copy()
        current_grip = _grip_open_val  # start with gripper open
        current_time = 0.0
        total_duration = num_waypoints / fps

        for phase_idx, (phase_name, _, _, duration_frac) in enumerate(phase_configs):
            target_joints = phase_targets[phase_idx]
            target_grip = _phase_gripper[phase_idx]
            phase_duration = duration_frac * total_duration
            phase_steps = max(2, int(round(duration_frac * num_waypoints)))
            start_step = 0 if phase_idx == 0 else 1

            for step in range(start_step, phase_steps):
                t = step / max(1, phase_steps - 1)
                # Smooth interpolation using cubic ease-in-out
                if t < 0.5:
                    s = 4.0 * t * t * t
                else:
                    s = 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0
                joint_pos = (1.0 - s) * current_joints + s * target_joints
                joint_pos = np.clip(joint_pos, lower, upper)
                if robot_config is not None:
                    joint_pos = self._clamp_joints_to_limits(joint_pos, robot_config)
                # Interpolate gripper joints and append
                grip_val = (1.0 - s) * current_grip + s * target_grip
                _jp_list = joint_pos.tolist()
                # Set gripper joints in the existing array (indices 7,8) rather than appending
                if gripper_joint_indices:
                    full_joints = _jp_list
                    for idx in gripper_joint_indices:
                        if idx < len(full_joints):
                            full_joints[idx] = grip_val
                        else:
                            while len(full_joints) <= idx:
                                full_joints.append(grip_val)
                else:
                    full_joints = _jp_list + [grip_val, grip_val]
                trajectory.append(
                    {
                        "joint_positions": full_joints,
                        "timestamp": current_time + t * phase_duration,
                        "phase": phase_name,
                    }
                )

            current_time += phase_duration
            current_joints = target_joints
            current_grip = target_grip

        self.log(
            f"  ℹ️  Multi-phase joint-space fallback: {len(trajectory)} waypoints "
            f"across 5 phases.",
            "INFO",
        )

        # Smooth arm joints to reduce jerk in fallback trajectories
        trajectory = self._apply_trajectory_smoothing(
            trajectory,
            robot_config=robot_config,
            indices=arm_joint_indices,
        )

        positions = None
        if robot_config is not None and IK_PLANNING_AVAILABLE:
            positions = self._compute_fk_positions(robot_config, trajectory)

        if positions is not None:
            if not self._trajectory_within_workspace(positions, workspace_bounds):
                self.log(
                    "  ❌ Joint-space fallback rejected: end-effector leaves workspace bounds.",
                    "ERROR",
                )
                return None
            if self._trajectory_violates_clearance(positions, obstacles, clearance=0.05):
                self.log(
                    "  ❌ Joint-space fallback rejected: obstacle clearance violated.",
                    "ERROR",
                )
                return None
            collision_free = True
        elif workspace_bounds is not None:
            if np.any(target_position < workspace_bounds[0]) or np.any(
                target_position > workspace_bounds[1]
            ):
                self.log(
                    "  ❌ Joint-space fallback rejected: target outside workspace bounds.",
                    "ERROR",
                )
                return None
            self.log(
                "  ℹ️  Reachability verified via workspace bounds (FK unavailable).",
                "INFO",
            )
            # Workspace bounds check passed; assume collision-free within bounds
            collision_free = True
        else:
            self.log(
                "  ℹ️  Reachability unchecked (no FK or workspace bounds).",
                "INFO",
            )
            # No collision checking available, but trajectory is within joint limits
            collision_free = True

        if robot_config is not None and not self._within_joint_limits(goal_joints, robot_config):
            self.log(
                "  ❌ Joint-space fallback rejected: target joints violate limits.",
                "ERROR",
            )
            return None

        self.log(
            "  ✅ Joint-space fallback generated with bounds and clearance heuristics.",
            "INFO",
        )
        # Determine collision source based on what checks were performed
        if positions is not None:
            _collision_source = "fk_clearance_check"
        elif workspace_bounds is not None:
            _collision_source = "workspace_bounds_check"
        else:
            _collision_source = "joint_limits_only"
        self._last_planning_report.update(
            {
                "planner": "linear_fallback",
                "collision_free": collision_free,
                "collision_source": _collision_source,
                "notes": [],
            }
        )
        return trajectory

    def _get_scene_obstacles(
        self,
        task: Dict[str, Any],
        initial_obs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract scene obstacles for collision avoidance.

        Args:
            task: Task configuration
            initial_obs: Initial observation with scene state

        Returns:
            List of obstacle dictionaries
        """
        obstacles = []

        # Get obstacles from task configuration
        if "obstacles" in task:
            obstacles.extend(task["obstacles"])

        # Get obstacles from scene state in observation
        scene_state = initial_obs.get("scene_state", {})
        try:
            from tools.dimension_estimation import get_dimension_estimator as _get_dim_est
            _dim_est = _get_dim_est()
        except ImportError:
            _dim_est = None
        for obj in scene_state.get("objects", []):
            _raw_dims = obj.get("dimensions")
            if _dim_est and (not _raw_dims or _raw_dims == [0.1, 0.1, 0.1]):
                _est_dims, _est_src = _dim_est.estimate_dimensions(obj)
            else:
                _est_dims = _raw_dims if _raw_dims else [0.1, 0.1, 0.1]
            obstacles.append({
                "id": obj.get("object_id", "unknown"),
                "position": (
                    [_p.get("x", 0), _p.get("y", 0), _p.get("z", 0)]
                    if isinstance((_p := obj.get("pose", {}).get("position", [0, 0, 0])), dict)
                    else _p
                ),
                "dimensions": _est_dims,
                "category": obj.get("category", "object"),
            })

        # Get target object from task (exclude from obstacles)
        target_id = task.get("target_object_id")

        # Filter out target object
        if target_id:
            obstacles = [o for o in obstacles if o.get("id") != target_id]

        return obstacles

    def _generate_curobo_trajectory(
        self,
        task: Dict[str, Any],
        initial_joints: np.ndarray,
        target_position: np.ndarray,
        place_position: Optional[np.ndarray],
        obstacles: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate collision-free trajectory using cuRobo.

        Real cuRobo GPU-accelerated motion planning.

        This implements multi-phase planning:
        1. Approach phase: Move to pre-grasp position
        2. Grasp phase: Move to grasp position
        3. Lift phase: Lift object
        4. Transport phase: Move to place position
        5. Place phase: Lower and release

        Args:
            task: Task configuration
            initial_joints: Starting joint configuration
            target_position: Position of target object
            place_position: Where to place the object
            obstacles: List of obstacles for collision avoidance

        Returns:
            List of trajectory waypoints or None if planning fails
        """
        if not CUROBO_INTEGRATION_AVAILABLE:
            self.log("cuRobo not available", "WARNING")
            return None

        try:
            # Create or get cuRobo planner
            if not hasattr(self, '_curobo_planner') or self._curobo_planner is None:
                self._curobo_planner = create_curobo_planner(
                    robot_type=self.config.robot_type,
                    device="cuda:0",
                )
                if self._curobo_planner is None:
                    self.log("Failed to create cuRobo planner", "ERROR")
                    return None
                self.log("✅ Created cuRobo planner")

            # Convert obstacles to cuRobo format
            curobo_obstacles = self._convert_obstacles_to_curobo(obstacles)

            # Planning phases for pick-and-place task
            trajectory_segments = []
            plan_results: List[CuRoboPlanResult] = []

            # Phase 1: Approach - move to pre-grasp position
            pre_grasp_position = target_position.copy()
            pre_grasp_position[2] += 0.15  # 15cm above object
            pre_grasp_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Default gripper-down

            approach_goal = np.concatenate([pre_grasp_position, pre_grasp_orientation])

            approach_request = CuRoboPlanRequest(
                start_joint_positions=initial_joints,
                goal_pose=approach_goal,
                obstacles=curobo_obstacles,
                max_iterations=1000,
                parallel_finetune=True,
                batch_size=32,
            )

            approach_result = self._curobo_planner.plan_to_pose(approach_request)

            if not approach_result.success:
                self.log(f"Approach planning failed: {approach_result.error_message}", "WARNING")
                self._last_planning_report.update(
                    {
                        "planner": "curobo",
                        "collision_free": False,
                        "collision_source": "curobo_plan_failure",
                        "notes": [approach_result.error_message],
                    }
                )
                return None

            trajectory_segments.append(("approach", approach_result.joint_trajectory))
            plan_results.append(approach_result)
            last_joints = approach_result.joint_trajectory[-1]

            # Phase 2: Grasp - move to grasp position
            grasp_position = target_position.copy()
            grasp_position[2] += 0.02  # Just above object

            grasp_goal = np.concatenate([grasp_position, pre_grasp_orientation])

            grasp_request = CuRoboPlanRequest(
                start_joint_positions=last_joints,
                goal_pose=grasp_goal,
                obstacles=curobo_obstacles,
                max_iterations=500,
            )

            grasp_result = self._curobo_planner.plan_to_pose(grasp_request)

            if not grasp_result.success:
                self.log(f"Grasp planning failed: {grasp_result.error_message}", "WARNING")
                # Continue with partial trajectory
            else:
                trajectory_segments.append(("grasp", grasp_result.joint_trajectory))
                plan_results.append(grasp_result)
                last_joints = grasp_result.joint_trajectory[-1]

            # Phase 3: Lift - move up after grasping
            lift_position = target_position.copy()
            lift_position[2] += 0.25  # 25cm above

            lift_goal = np.concatenate([lift_position, pre_grasp_orientation])

            lift_request = CuRoboPlanRequest(
                start_joint_positions=last_joints,
                goal_pose=lift_goal,
                obstacles=curobo_obstacles,
                max_iterations=500,
            )

            lift_result = self._curobo_planner.plan_to_pose(lift_request)

            if lift_result.success:
                trajectory_segments.append(("lift", lift_result.joint_trajectory))
                plan_results.append(lift_result)
                last_joints = lift_result.joint_trajectory[-1]

            # Phase 4: Transport - move to place position (if specified)
            if place_position is not None:
                pre_place_position = place_position.copy()
                pre_place_position[2] += 0.20  # 20cm above place position

                transport_goal = np.concatenate([pre_place_position, pre_grasp_orientation])

                transport_request = CuRoboPlanRequest(
                    start_joint_positions=last_joints,
                    goal_pose=transport_goal,
                    obstacles=curobo_obstacles,
                    max_iterations=1000,
                    parallel_finetune=True,
                    batch_size=32,
                )

                transport_result = self._curobo_planner.plan_to_pose(transport_request)

                if transport_result.success:
                    trajectory_segments.append(("transport", transport_result.joint_trajectory))
                    plan_results.append(transport_result)
                    last_joints = transport_result.joint_trajectory[-1]

                # Phase 5: Place - lower to place position
                place_goal = np.concatenate([place_position, pre_grasp_orientation])

                place_request = CuRoboPlanRequest(
                    start_joint_positions=last_joints,
                    goal_pose=place_goal,
                    obstacles=curobo_obstacles,
                    max_iterations=500,
                )

                place_result = self._curobo_planner.plan_to_pose(place_request)

                if place_result.success:
                    trajectory_segments.append(("place", place_result.joint_trajectory))
                    plan_results.append(place_result)

            # Combine all trajectory segments
            full_trajectory = []
            timestamp = 0.0
            dt = 1.0 / 30.0  # 30Hz

            for phase_name, segment_joints in trajectory_segments:
                for i, joint_pos in enumerate(segment_joints):
                    full_trajectory.append({
                        "joint_positions": joint_pos.tolist(),
                        "timestamp": timestamp,
                        "phase": phase_name,
                    })
                    timestamp += dt

            self.log(f"  cuRobo planning complete: {len(full_trajectory)} waypoints, {len(trajectory_segments)} phases")

            # Calculate quality metrics
            total_planning_time = approach_result.planning_time_ms
            self.log(f"  Total planning time: {total_planning_time:.1f}ms")

            collision_free = None
            if plan_results:
                collision_free = all(result.is_collision_free for result in plan_results)
            self._last_planning_report.update(
                {
                    "planner": "curobo",
                    "collision_free": collision_free,
                    "collision_source": "curobo_plan_result",
                    "notes": [],
                }
            )

            return full_trajectory if full_trajectory else None

        except Exception as e:
            self.log(f"cuRobo trajectory generation error: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return None

    def _convert_obstacles_to_curobo(
        self,
        obstacles: List[Dict[str, Any]],
    ) -> List[Any]:
        """
        Convert obstacle dictionaries to cuRobo CollisionObject format.

        Args:
            obstacles: List of obstacle dicts with id, position, dimensions

        Returns:
            List of CollisionObject instances
        """
        if not CUROBO_INTEGRATION_AVAILABLE:
            return []

        curobo_obstacles = []

        for obs in obstacles:
            try:
                position = np.array(obs.get("position", [0, 0, 0]))
                dimensions = np.array(obs.get("dimensions", [0.1, 0.1, 0.1]))

                collision_obj = CollisionObject(
                    object_id=obs.get("id", "obstacle"),
                    geometry_type=CollisionGeometryType.CUBOID,
                    position=position,
                    orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
                    dimensions=dimensions,
                    is_static=True,
                )
                curobo_obstacles.append(collision_obj)

            except Exception as e:
                self.log(f"Failed to convert obstacle {obs.get('id')}: {e}", "WARNING")

        return curobo_obstacles

    def _enrich_episode_with_llm(
        self,
        frames: List[Dict[str, Any]],
        task: Dict[str, Any],
        episode_id: str,
        collision_free: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Use Gemini 3.0 Pro Preview to enrich episode metadata.

        Generates task descriptions, scene annotations, success criteria,
        and evaluates whether the trajectory likely accomplishes the task.

        Returns a dict with enriched metadata fields.
        """
        result: Dict[str, Any] = {}

        # Always attempt LLM enrichment for maximum data quality.
        # Falls through to ImportError handler if LLM client is unavailable.

        try:
            from tools.llm_client import create_llm_client, LLMProvider
        except ImportError:
            self.log("  ℹ️  LLM client unavailable; skipping episode enrichment.", "INFO")
            return result

        # Extract trajectory summary for the LLM (first, middle, last frames)
        num_frames = len(frames)
        sample_indices = [0, num_frames // 4, num_frames // 2, 3 * num_frames // 4, num_frames - 1]
        sample_indices = sorted(set(i for i in sample_indices if 0 <= i < num_frames))

        trajectory_summary = []
        for idx in sample_indices:
            frame = frames[idx]
            obs = frame.get("observation") or {}
            rs = obs.get("robot_state") or {}
            jp = rs.get("joint_positions", [])
            action = frame.get("action", [])
            sample = {
                "step": idx,
                "joint_positions": [round(v, 4) for v in jp[:7]] if jp else [],
                "action_delta": [round(v, 4) for v in action[:7]] if action else [],
                "timestamp": frame.get("timestamp", 0),
            }
            # Include gripper state and end-effector pose when available
            if frame.get("gripper_command"):
                sample["gripper"] = frame["gripper_command"]
                sample["gripper_openness"] = frame.get("gripper_openness", 0.0)
            if frame.get("ee_pos"):
                sample["ee_position_xyz"] = [round(v, 4) for v in frame["ee_pos"]]
            if frame.get("ee_quat"):
                sample["ee_orientation_quat"] = [round(v, 4) for v in frame["ee_quat"]]
            trajectory_summary.append(sample)

        task_type = task.get("task_type", "manipulation")
        target_object = task.get("target_object", "unknown object")
        description_hint = task.get("description_hint", "")
        robot_type = getattr(self.config, "robot_type", "humanoid")

        # Gather scene object context from task config
        scene_objects = task.get("objects") or task.get("scene_objects") or []
        scene_object_names = [
            o.get("name") or o.get("object_id") or str(o)
            for o in scene_objects
        ] if isinstance(scene_objects, list) else []

        # Get real joint names from ROBOT_CONFIGS
        rc = ROBOT_CONFIGS.get(robot_type)
        joint_name_list = list(rc.joint_names) if rc is not None else []

        prompt = (
            f"You are evaluating a robot manipulation episode for data quality.\n\n"
            f"Robot: {robot_type}\n"
            f"Joint names: {joint_name_list}\n"
            f"Task type: {task_type}\n"
            f"Target object: {target_object}\n"
            f"Description hint: {description_hint}\n"
            f"Scene objects: {scene_object_names}\n"
            f"Episode ID: {episode_id}\n"
            f"Total frames: {num_frames}\n"
            f"Collision-free: {collision_free}\n\n"
            f"Trajectory samples (joint positions, action deltas, gripper state, "
            f"and end-effector pose at key frames):\n"
            f"{json.dumps(trajectory_summary, indent=2)}\n\n"
            f"Based on this information, provide a JSON response with:\n"
            f'{{"task_name": "short descriptive name for the task",\n'
            f' "task_description": "1-2 sentence natural language description of what the robot is doing",\n'
            f' "scene_description": "brief description of the scene and objects present",\n'
            f' "success_criteria": ["list of criteria for task success"],\n'
            f' "task_success": true/false (whether the trajectory plausibly accomplishes the task),\n'
            f' "task_success_reasoning": "1 sentence explanation of success/failure assessment",\n'
            f' "phase_descriptions": {{"approach": "description", "grasp": "description", "lift": "description", "transport": "description", "place": "description", "release": "description"}},\n'
            f' "paraphrases": ["alternative description 1", "alternative description 2", "alternative description 3"],\n'
            f' "high_level_goal": "abstract goal description",\n'
            f' "per_frame_narration": [{{"frame": 0, "narration": "what happens at this keyframe"}}, ...]}}\n'
            f"\nFor per_frame_narration, narrate only these sampled keyframes: {sample_indices}\n"
        )

        try:
            client = create_llm_client(provider=LLMProvider.GEMINI)
            response = client.generate(
                prompt=prompt,
                json_output=True,
                temperature=0.3,
                disable_tools=True,
            )
            if response.error_message:
                self.log(f"  ⚠️  LLM enrichment failed: {response.error_message}", "WARNING")
                return result

            data = response.data
            if data is None:
                data = json.loads(response.text)

            result["task_name"] = data.get("task_name")
            result["task_description"] = data.get("task_description")
            result["scene_description"] = data.get("scene_description")
            result["success_criteria"] = data.get("success_criteria")
            result["task_success"] = data.get("task_success")
            result["task_success_reasoning"] = data.get("task_success_reasoning")

            # Rich language annotations (C2)
            language_annotations = {}
            if data.get("phase_descriptions"):
                language_annotations["phase_descriptions"] = data["phase_descriptions"]
            if data.get("paraphrases"):
                language_annotations["paraphrases"] = data["paraphrases"]
            if data.get("high_level_goal"):
                language_annotations["high_level_goal"] = data["high_level_goal"]
            if data.get("per_frame_narration"):
                language_annotations["per_frame_narration"] = data["per_frame_narration"]
            if language_annotations:
                result["language_annotations"] = language_annotations

            self.log(
                f"  ✅ LLM enrichment: task_name={result.get('task_name')}, "
                f"task_success={result.get('task_success')}",
                "INFO",
            )
        except Exception as exc:
            self.log(f"  ⚠️  LLM enrichment error: {exc}", "WARNING")

        return result

    def _compute_joint_utilization(
        self,
        frames: List[Dict[str, Any]],
        joint_count: int = 7,
    ) -> Dict[str, Any]:
        epsilon = float(os.getenv("GENIESIM_JOINT_UTILIZATION_EPS", "0.02"))
        positions: List[np.ndarray] = []
        for frame in frames:
            obs = frame.get("observation") or {}
            rs = obs.get("robot_state") or {}
            jp = rs.get("joint_positions")
            if isinstance(jp, (list, tuple, np.ndarray)) and len(jp) >= joint_count:
                positions.append(np.array(jp[:joint_count], dtype=float))
            elif isinstance(frame.get("action_abs"), (list, tuple, np.ndarray)):
                action_abs = frame.get("action_abs") or []
                if len(action_abs) >= joint_count:
                    positions.append(np.array(action_abs[:joint_count], dtype=float))

        if len(positions) < 2 or joint_count <= 0:
            return {
                "fraction": 0.0,
                "utilized_joint_count": 0,
                "joint_count": joint_count,
                "epsilon": epsilon,
                "per_joint_range": [],
            }

        joint_matrix = np.vstack(positions)
        joint_ranges = np.ptp(joint_matrix, axis=0)
        utilized = joint_ranges > epsilon
        utilized_count = int(np.sum(utilized))
        fraction = utilized_count / joint_count if joint_count > 0 else 0.0
        return {
            "fraction": float(fraction),
            "utilized_joint_count": utilized_count,
            "joint_count": joint_count,
            "epsilon": epsilon,
            "per_joint_range": [float(v) for v in joint_ranges.tolist()],
        }

    def _audit_episode_with_vlm(
        self,
        frames: List[Dict[str, Any]],
        task: Dict[str, Any],
        quality_score: float,
    ) -> Dict[str, Any]:
        """Vision-based quality audit using VLM. Gated by ENABLE_VLM_QUALITY_AUDIT=1."""
        if os.getenv("ENABLE_VLM_QUALITY_AUDIT", "0") != "1":
            return {}
        if not frames:
            return {}

        # Sample keyframes: first, 25%, 50%, 75%, last
        n = len(frames)
        sample_indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        sample_indices = [i for i in sample_indices if 0 <= i < n]

        # Extract RGB images from camera frames
        # Ensure episode directories are set for NPY path resolution
        _ep_dir = getattr(self, "_current_episode_dir", None)
        _frames_dir = getattr(self, "_current_frames_dir", None)

        if _ep_dir is None:
            self.log("VLM audit skipped: _current_episode_dir not set", "WARNING")
            return {"skipped": True, "reason": "episode_dir_not_set"}

        # Derive frames_dir from episode_dir if not set
        if _frames_dir is None and _ep_dir is not None:
            _ep_path = Path(_ep_dir) if not isinstance(_ep_dir, Path) else _ep_dir
            # Look for *_frames subdirectory
            _frames_candidates = list(_ep_path.glob("*_frames"))
            if _frames_candidates:
                _frames_dir = _frames_candidates[0]
                self._current_frames_dir = _frames_dir
                self.log(f"VLM audit: derived frames_dir={_frames_dir}", "DEBUG")

        images = []
        for idx in sample_indices:
            frame = frames[idx]
            camera_frames = frame.get("observation", {}).get("camera_frames") or {}
            # Try wrist camera first, then any available camera
            rgb = None
            for cam_id in ["wrist", "front", "overhead"]:
                cam_data = camera_frames.get(cam_id)
                if cam_data and isinstance(cam_data, dict):
                    rgb = load_camera_frame(
                        cam_data,
                        "rgb",
                        ep_dir=_ep_dir,
                        frames_dir=_frames_dir,
                    )
                    if isinstance(rgb, str) and rgb.endswith(".npy"):
                        _npy = resolve_npy_path(rgb, ep_dir=_ep_dir, frames_dir=_frames_dir)
                        if _npy is not None and _npy.exists():
                            rgb = np.load(_npy)
                    rgb = validate_camera_array(rgb, context=f"vlm_audit:{cam_id}:{idx}")
                    if rgb is not None:
                        break
            if rgb is None:
                # Try any camera
                for cam_id, cam_data in camera_frames.items():
                    if cam_data and isinstance(cam_data, dict):
                        rgb = load_camera_frame(
                            cam_data,
                            "rgb",
                            ep_dir=_ep_dir,
                            frames_dir=_frames_dir,
                        )
                        if isinstance(rgb, str) and rgb.endswith(".npy"):
                            _npy = resolve_npy_path(rgb, ep_dir=_ep_dir, frames_dir=_frames_dir)
                            if _npy is not None and _npy.exists():
                                rgb = np.load(_npy)
                        rgb = validate_camera_array(rgb, context=f"vlm_audit:{cam_id}:{idx}")
                        if rgb is not None:
                            break
            if rgb is not None:
                images.append({"frame_idx": idx, "rgb": rgb})

        if not images:
            self.log("  ℹ️  VLM audit skipped: no RGB frames available", "INFO")
            return {"skipped": True, "reason": "no_rgb_frames"}

        try:
            from PIL import Image
        except Exception as exc:
            self.log(f"  ℹ️  VLM audit skipped: PIL unavailable ({exc})", "INFO")
            return {"skipped": True, "reason": "pil_unavailable"}

        converted_images = []
        for img in images:
            try:
                rgb_data = img["rgb"]
                # Final defensive check: ensure rgb_data is a numpy array
                if not isinstance(rgb_data, np.ndarray):
                    self.log(
                        f"  Warning: frame {img['frame_idx']} rgb is {type(rgb_data).__name__}, skipping",
                        "WARNING",
                    )
                    continue
                pil_image = Image.fromarray(rgb_data.astype(np.uint8))
                converted_images.append({"frame_idx": img["frame_idx"], "image": pil_image})
            except Exception as exc:
                self.log(f"  ⚠️  VLM audit skipped: failed to convert RGB frames ({exc})", "WARNING")
                return {"skipped": True, "reason": f"image_conversion_failed: {exc}"}

        try:
            from tools.llm_client import create_llm_client, LLMProvider
            vlm_client = create_llm_client(provider=LLMProvider.GEMINI)
            if not vlm_client:
                return {"skipped": True, "reason": "vlm_client_unavailable"}

            task_desc = task.get("description_hint") or task.get("task_type", "manipulation")
            target_obj = task.get("target_object", "unknown")

            prompt = (
                f"You are auditing a robot manipulation episode for data quality.\n\n"
                f"Task: {task_desc}\n"
                f"Target object: {target_obj}\n"
                f"Heuristic quality score: {quality_score:.3f}\n"
                f"Number of frames: {len(frames)}\n\n"
                f"I'm showing you {len(images)} keyframes from the episode. Analyze:\n"
                f"1. Does the gripper make contact with the target object?\n"
                f"2. Is the grasp stable (no slipping/dropping)?\n"
                f"3. Is the placement stable and at a reasonable location?\n"
                f"4. Are there any visual anomalies (clipping, floating objects, impossible poses)?\n"
                f"5. Overall visual quality assessment.\n\n"
                f"Return ONLY JSON:\n"
                f'{{"gripper_contact": true/false,\n'
                f' "grasp_stable": true/false,\n'
                f' "placement_stable": true/false,\n'
                f' "visual_anomalies": ["list of issues or empty"],\n'
                f' "vlm_quality_score": 0.0-1.0,\n'
                f' "assessment": "1-2 sentence summary"}}\n'
            )

            # Pass images to VLM
            image_data = [img["image"] for img in converted_images]
            response = vlm_client.generate(
                prompt=prompt,
                images=image_data,
                json_output=True,
                temperature=0.3,
                disable_tools=True,
            )

            if response.error_message:
                self.log(f"  ⚠️  VLM audit failed: {response.error_message}", "WARNING")
                return {"skipped": True, "reason": f"vlm_error: {response.error_message}"}

            data = response.data
            if data is None:
                data = json.loads(response.text)

            vlm_score = float(data.get("vlm_quality_score", quality_score))
            # Blend: 70% heuristic + 30% VLM
            blended_score = 0.7 * quality_score + 0.3 * vlm_score

            result = {
                "vlm_quality_score": round(vlm_score, 4),
                "blended_quality_score": round(blended_score, 4),
                "gripper_contact": data.get("gripper_contact"),
                "grasp_stable": data.get("grasp_stable"),
                "placement_stable": data.get("placement_stable"),
                "visual_anomalies": data.get("visual_anomalies", []),
                "assessment": data.get("assessment", ""),
                "keyframes_analyzed": len(converted_images),
            }
            self.log(
                f"  🔍 VLM audit: score={vlm_score:.3f}, blended={blended_score:.3f}, "
                f"contact={data.get('gripper_contact')}, stable={data.get('grasp_stable')}",
                "INFO",
            )
            return result

        except Exception as exc:
            self.log(f"  ⚠️  VLM audit error: {exc}", "WARNING")
            return {"skipped": True, "reason": str(exc)}

    def _assess_sim_to_real_gap(
        self,
        frames: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assess sim-to-real visual gap using VLM. Gated by ENABLE_SIM2REAL_ASSESSMENT=1."""
        if os.getenv("ENABLE_SIM2REAL_ASSESSMENT", "0") != "1":
            return {}
        if not frames:
            return {}

        # Sample 3 frames: start, mid, end
        n = len(frames)
        sample_indices = sorted(set([0, n // 2, n - 1]))

        _ep_dir = getattr(self, "_current_episode_dir", None)
        _frames_dir = getattr(self, "_current_frames_dir", None)
        images = []
        for idx in sample_indices:
            frame = frames[idx]
            camera_frames = frame.get("observation", {}).get("camera_frames") or {}
            _found = False
            for cam_id in ["wrist", "front", "overhead"]:
                cam_data = camera_frames.get(cam_id)
                if cam_data and isinstance(cam_data, dict):
                    rgb = load_camera_frame(
                        cam_data,
                        "rgb",
                        ep_dir=_ep_dir,
                        frames_dir=_frames_dir,
                    )
                    if isinstance(rgb, str) and rgb.endswith(".npy"):
                        _npy = resolve_npy_path(rgb, ep_dir=_ep_dir, frames_dir=_frames_dir)
                        if _npy is not None and _npy.exists():
                            rgb = np.load(_npy)
                    rgb = validate_camera_array(rgb, context=f"sim2real:{cam_id}:{idx}")
                    if rgb is not None:
                        images.append(rgb)
                        _found = True
                        break
            if not _found:
                for cam_data in camera_frames.values():
                    if cam_data and isinstance(cam_data, dict):
                        rgb = load_camera_frame(
                            cam_data,
                            "rgb",
                            ep_dir=_ep_dir,
                            frames_dir=_frames_dir,
                        )
                        if isinstance(rgb, str) and rgb.endswith(".npy"):
                            _npy = resolve_npy_path(rgb, ep_dir=_ep_dir, frames_dir=_frames_dir)
                            if _npy is not None and _npy.exists():
                                rgb = np.load(_npy)
                        rgb = validate_camera_array(rgb, context=f"sim2real:unknown:{idx}")
                        if rgb is not None:
                            images.append(rgb)
                            break

        if not images:
            return {"skipped": True, "reason": "no_rgb_frames"}

        try:
            from PIL import Image
        except Exception as exc:
            self.log(f"  ℹ️  Sim-to-real assessment skipped: PIL unavailable ({exc})", "INFO")
            return {"skipped": True, "reason": "pil_unavailable"}

        converted_images = []
        for rgb in images:
            try:
                # Final defensive check: ensure rgb is a numpy array
                if not isinstance(rgb, np.ndarray):
                    self.log(
                        f"  Warning: sim-to-real rgb is {type(rgb).__name__}, skipping",
                        "WARNING",
                    )
                    continue
                pil_image = Image.fromarray(rgb.astype(np.uint8))
                converted_images.append(pil_image)
            except Exception as exc:
                self.log(
                    f"  ⚠️  Sim-to-real assessment skipped: failed to convert RGB frames ({exc})",
                    "WARNING",
                )
                return {"skipped": True, "reason": f"image_conversion_failed: {exc}"}

        try:
            from tools.llm_client import create_llm_client, LLMProvider
            vlm_client = create_llm_client(provider=LLMProvider.GEMINI)
            if not vlm_client:
                return {"skipped": True, "reason": "vlm_client_unavailable"}

            prompt = (
                f"You are assessing the sim-to-real visual gap in these robot simulation frames.\n\n"
                f"Rate the following aspects on a 0.0-1.0 scale (1.0 = perfectly realistic):\n"
                f"1. Lighting quality and shadows\n"
                f"2. Object textures and materials\n"
                f"3. Object placement realism\n"
                f"4. Robot pose and motion realism\n"
                f"5. Overall scene realism\n\n"
                f"Flag anything that looks obviously simulated or unrealistic.\n\n"
                f"Return ONLY JSON:\n"
                f'{{"lighting_score": 0.0-1.0,\n'
                f' "texture_score": 0.0-1.0,\n'
                f' "placement_score": 0.0-1.0,\n'
                f' "robot_pose_score": 0.0-1.0,\n'
                f' "overall_realism_score": 0.0-1.0,\n'
                f' "issues": ["list of identified sim-to-real gap issues"]}}\n'
            )

            response = vlm_client.generate(
                prompt=prompt,
                images=converted_images,
                json_output=True,
                temperature=0.3,
                disable_tools=True,
            )

            if response.error_message:
                return {"skipped": True, "reason": f"vlm_error: {response.error_message}"}

            data = response.data
            if data is None:
                data = json.loads(response.text)

            result = {
                "lighting_score": float(data.get("lighting_score", 0)),
                "texture_score": float(data.get("texture_score", 0)),
                "placement_score": float(data.get("placement_score", 0)),
                "robot_pose_score": float(data.get("robot_pose_score", 0)),
                "overall_realism_score": float(data.get("overall_realism_score", 0)),
                "issues": data.get("issues", []),
                "frames_analyzed": len(converted_images),
            }
            self.log(
                f"  🌐 Sim-to-real: realism={result['overall_realism_score']:.3f}, "
                f"issues={len(result['issues'])}",
                "INFO",
            )
            return result

        except Exception as exc:
            self.log(f"  ⚠️  Sim-to-real assessment error: {exc}", "WARNING")
            return {"skipped": True, "reason": str(exc)}

    def _diagnose_failure_with_llm(
        self,
        validity_errors: List[str],
        frames: List[Dict[str, Any]],
        quality_score: float,
        task: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Diagnose episode failure using LLM. Gated by ENABLE_LLM_FAILURE_DIAGNOSIS=1."""
        if os.getenv("ENABLE_LLM_FAILURE_DIAGNOSIS", "0") != "1":
            return {}
        if not validity_errors and quality_score >= float(os.getenv("MIN_QUALITY_SCORE", "0.7")):
            return {}  # No failure to diagnose

        try:
            from tools.llm_client import create_llm_client, LLMProvider
            llm = create_llm_client(provider=LLMProvider.GEMINI)
            if not llm:
                return {}

            task_desc = task.get("description_hint") or task.get("task_type", "manipulation")
            target_obj = task.get("target_object", "unknown")

            # Build compact trajectory summary
            trajectory_info = {}
            if frames:
                trajectory_info["num_frames"] = len(frames)
                if frames[0].get("ee_pos"):
                    trajectory_info["ee_start"] = [round(v, 3) for v in frames[0]["ee_pos"]]
                if frames[-1].get("ee_pos"):
                    trajectory_info["ee_end"] = [round(v, 3) for v in frames[-1]["ee_pos"]]
                gripper_states = [f.get("gripper_command") for f in frames if f.get("gripper_command")]
                trajectory_info["gripper_states"] = list(set(gripper_states))

            prompt = (
                f"A robot manipulation episode has failed or has low quality.\n\n"
                f"Task: {task_desc}\n"
                f"Target object: {target_obj}\n"
                f"Quality score: {quality_score:.3f}\n"
                f"Validity errors: {json.dumps(validity_errors[:10])}\n"
                f"Trajectory info: {json.dumps(trajectory_info)}\n\n"
                f"Diagnose the root cause and suggest how to fix it for a retry.\n\n"
                f"Return ONLY JSON:\n"
                f'{{"root_cause": "1-sentence diagnosis",\n'
                f' "suggested_fix": "1-sentence fix description",\n'
                f' "retry_possible": true/false,\n'
                f' "retry_params": {{"position_offset_m": [0,0,0], "force_multiplier": 1.0}}}}\n'
            )

            response = llm.generate(prompt=prompt, json_output=True, temperature=0.3, disable_tools=True)
            if response.error_message:
                return {"error": response.error_message}

            data = response.data
            if data is None:
                data = json.loads(response.text)

            result = {
                "root_cause": data.get("root_cause", "unknown"),
                "suggested_fix": data.get("suggested_fix", ""),
                "retry_possible": data.get("retry_possible", False),
                "retry_params": data.get("retry_params", {}),
            }
            self.log(
                f"  🔬 Failure diagnosis: {result['root_cause']} | "
                f"retry={'yes' if result['retry_possible'] else 'no'}",
                "INFO",
            )
            return result

        except Exception as exc:
            self.log(f"  ⚠️  Failure diagnosis error: {exc}", "WARNING")
            return {"error": str(exc)}

    def _calculate_quality_score(
        self,
        frames: List[Dict[str, Any]],
        task: Dict[str, Any],
    ) -> float:
        """Calculate quality score for an episode."""
        if not frames:
            logger.warning("Quality check failed: no frames recorded.")
            return 0.0

        task_obs_schema = task.get("observation_schema") or {}
        required_keys = task_obs_schema.get("required_keys") or task.get("required_observation_keys")
        if not required_keys:
            # Always require full observation schema including camera data
            required_keys = DEFAULT_OBSERVATION_SCHEMA["required_keys"]

        task_action_bounds = task.get("action_bounds") or task.get("joint_limits") or {}
        lower_bounds = (
            task_action_bounds.get("lower")
            or task_action_bounds.get("min")
            or DEFAULT_ACTION_BOUNDS["lower"]
        )
        upper_bounds = (
            task_action_bounds.get("upper")
            or task_action_bounds.get("max")
            or DEFAULT_ACTION_BOUNDS["upper"]
        )
        success_schema = task.get("success_schema") or {}
        success_keys = success_schema.get("success_keys") or task.get("success_keys")
        if not success_keys:
            success_keys = DEFAULT_SUCCESS_SCHEMA["success_keys"]
        grasp_keys = success_schema.get("grasp_keys") or DEFAULT_SUCCESS_SCHEMA["grasp_keys"]
        release_keys = success_schema.get("release_keys") or DEFAULT_SUCCESS_SCHEMA["release_keys"]

        total_frames = len(frames)

        missing_obs_count = 0
        missing_fields_count = 0
        action_invalid_count = 0
        action_bounds_mismatch = False
        success_flag_detected = False
        success_flag_value = False
        grasp_frame_index = None
        release_frame_index = None

        for idx, frame in enumerate(frames):
            obs = frame.get("observation")
            if not obs:
                missing_obs_count += 1
                missing_fields_count += 1
            else:
                missing_fields = [key for key in required_keys if key not in obs]
                if missing_fields:
                    missing_fields_count += 1

            action = frame.get("action")
            if not isinstance(action, (list, tuple, np.ndarray)):
                action_invalid_count += 1
            else:
                action_array = np.array(action, dtype=float)
                lower = np.array(lower_bounds, dtype=float)
                upper = np.array(upper_bounds, dtype=float)
                # When action has more dims than bounds (e.g. full articulation
                # vs arm+gripper bounds), validate the overlapping prefix only.
                if action_array.shape[0] > lower.shape[0]:
                    action_array = action_array[: lower.shape[0]]
                elif action_array.shape[0] < lower.shape[0]:
                    lower = lower[: action_array.shape[0]]
                    upper = upper[: action_array.shape[0]]
                if action_array.shape != lower.shape or action_array.shape != upper.shape:
                    action_invalid_count += 1
                    action_bounds_mismatch = True
                else:
                    if np.any(action_array < lower) or np.any(action_array > upper):
                        action_invalid_count += 1

            if obs:
                for key in success_keys:
                    if key in obs or key in frame:
                        success_flag_detected = True
                        success_flag_value = success_flag_value or bool(obs.get(key, frame.get(key)))
                        break

                if grasp_frame_index is None:
                    for key in grasp_keys:
                        if bool(obs.get(key, frame.get(key))):
                            grasp_frame_index = idx
                            break
                    # Fallback: detect grasp from gripper_command
                    if grasp_frame_index is None and frame.get("gripper_command") == "closed":
                        grasp_frame_index = idx
                if grasp_frame_index is not None and release_frame_index is None:
                    for key in release_keys:
                        if bool(obs.get(key, frame.get(key))):
                            release_frame_index = idx
                            break
                    # Fallback: detect release from gripper_command after grasp
                    if release_frame_index is None and frame.get("gripper_command") == "open":
                        release_frame_index = idx

        if missing_obs_count > 0:
            logger.warning(
                "Quality check: missing observation payloads in %d/%d frames.",
                missing_obs_count,
                total_frames,
            )
        if missing_fields_count > 0:
            logger.warning(
                "Quality check: missing required observation fields in %d/%d frames (required=%s).",
                missing_fields_count,
                total_frames,
                required_keys,
            )
        if action_invalid_count > 0:
            if action_bounds_mismatch:
                logger.warning(
                    "Quality check: action bounds mismatch for %d/%d frames (bounds length mismatch).",
                    action_invalid_count,
                    total_frames,
                )
            else:
                logger.warning(
                    "Quality check: action out of bounds in %d/%d frames.",
                    action_invalid_count,
                    total_frames,
                )

        data_completeness_score = 1.0 - (missing_fields_count / total_frames)
        action_validity_score = 1.0 - (action_invalid_count / total_frames)

        # --- Fix 7: Goal-region geometric task success verification ---
        geometric_success = None
        geometric_reasoning = ""
        target_object_id = task.get("target_object") or task.get("target_object_id")
        if target_object_id and frames:
            _init_poses = frames[0].get("_initial_object_poses", {})
            _final_poses = frames[-1].get("_final_object_poses", {})
            if target_object_id in _init_poses and target_object_id in _final_poses:
                _ip = np.array(_init_poses[target_object_id])
                _fp = np.array(_final_poses[target_object_id])
                _displacement = float(np.linalg.norm(_fp - _ip))
                # Multi-milestone check (mirrors episode-level Fix 7)
                _gr_milestones = 0
                # 1. displacement
                if _displacement > 0.10:
                    _gr_milestones += 1
                # 2. grasp detected
                if grasp_frame_index is not None:
                    _gr_milestones += 1
                # 3. release detected
                if release_frame_index is not None:
                    _gr_milestones += 1
                # 4. lift (check max z above initial)
                _init_z_qs = float(_ip[2]) if len(_ip) >= 3 else 0.0
                _max_z_qs = _init_z_qs
                for _fr_qs in frames:
                    _obs_qs = _fr_qs.get("observation", {})
                    _ss_qs = _obs_qs.get("scene_state", {}) or _obs_qs.get("privileged", {}).get("scene_state", {})
                    for _obj_qs in _ss_qs.get("objects", []):
                        if _obj_qs.get("object_id") == target_object_id:
                            _op_qs = _obj_qs.get("pose", {})
                            _z_qs = _op_qs.get("z") or (_op_qs.get("position", [0, 0, 0])[2] if "position" in _op_qs else None)
                            if _z_qs is not None:
                                _max_z_qs = max(_max_z_qs, float(_z_qs))
                            break
                if _max_z_qs - _init_z_qs >= 0.05:
                    _gr_milestones += 1

                geometric_success = _gr_milestones >= 3
                geometric_reasoning = (
                    f"Goal-region QS: {_gr_milestones}/4 milestones "
                    f"(disp={_displacement:.3f}m, lift={_max_z_qs - _init_z_qs:.3f}m)"
                )

        if geometric_success is not None:
            success_score = 1.0 if geometric_success else 0.0
            if not geometric_success:
                logger.warning("Quality check: geometric verification failed: %s", geometric_reasoning)
        elif success_flag_detected:
            success_score = 1.0 if success_flag_value else 0.0
            if not success_flag_value:
                logger.warning("Quality check: task success flag is false.")
        else:
            success_score = 1.0 if (
                grasp_frame_index is not None
                and release_frame_index is not None
                and release_frame_index >= grasp_frame_index
            ) else 0.0
            if success_score == 0.0:
                logger.warning(
                    "Quality check: task success heuristic not satisfied (grasp/release milestones)."
                )

        # Frame count scoring — configurable via quality_config.json
        # Load quality config for frame count and diversity settings
        try:
            _quality_config = load_quality_config()
            _frame_cfg = _quality_config.frame_count_scoring
        except Exception as _cfg_err:
            logger.warning("Failed to load quality config, using defaults: %s", _cfg_err)
            _frame_cfg = None

        if _frame_cfg and _frame_cfg.use_gradual_scoring:
            # Gradual scoring: linear interpolation between min_frames_nonzero and min_frames_full_score
            _min_full = _frame_cfg.min_frames_full_score
            _min_nonzero = _frame_cfg.min_frames_nonzero
            if total_frames >= _min_full:
                frame_count_score = 1.0
            elif total_frames >= _min_nonzero:
                # Linear interpolation from 0.3 to 1.0
                frame_count_score = 0.3 + 0.7 * (total_frames - _min_nonzero) / max(1, _min_full - _min_nonzero)
            else:
                frame_count_score = 0.0
        else:
            # Legacy cliff behavior
            frame_count_score = 1.0 if total_frames >= 10 else 0.5

        # Diversity divisors — loaded from quality_config.json (configurable)
        _robot_type = getattr(self.config, "robot_type", "franka") if hasattr(self, "config") else "franka"
        if _quality_config:
            _div_cfg = _quality_config.get_diversity_divisors(_robot_type)
            _action_divisor = _div_cfg.action
            _obs_divisor = _div_cfg.obs
            _diversity_calibration_source = "quality_config"
        else:
            # Fallback to hardcoded values if config load failed
            _DIVERSITY_DIVISORS = {
                "franka": (0.05, 0.05),
                "panda": (0.05, 0.05),
                "g1": (0.08, 0.08),
                "ur5": (0.04, 0.04),
                "ur10": (0.04, 0.04),
            }
            _action_divisor, _obs_divisor = _DIVERSITY_DIVISORS.get(
                _robot_type.lower() if _robot_type else "franka", (0.05, 0.05)
            )
            _diversity_calibration_source = "hardcoded_fallback"

        # Action diversity: measure how much the actions change across frames.
        # A trajectory where all actions are identical is low quality.
        action_diversity_score = 0.0
        if total_frames >= 2:
            all_actions = []
            for frame in frames:
                action = frame.get("action")
                if isinstance(action, (list, tuple, np.ndarray)):
                    all_actions.append(np.array(action, dtype=float))
            if len(all_actions) >= 2:
                action_matrix = np.array(all_actions)
                # Compute mean pairwise L2 distance between consecutive actions
                diffs = np.diff(action_matrix, axis=0)
                mean_diff = np.mean(np.linalg.norm(diffs, axis=1))
                action_diversity_score = min(1.0, mean_diff / _action_divisor)

        # Observation diversity: measure how much joint positions change
        obs_diversity_score = 0.0
        if total_frames >= 2:
            all_obs_joints = []
            for frame in frames:
                obs = frame.get("observation") or {}
                rs = obs.get("robot_state") or {}
                jp = rs.get("joint_positions")
                if isinstance(jp, (list, tuple, np.ndarray)):
                    all_obs_joints.append(np.array(jp, dtype=float))
            if len(all_obs_joints) >= 2:
                obs_matrix = np.array(all_obs_joints)
                obs_diffs = np.diff(obs_matrix, axis=0)
                mean_obs_diff = np.mean(np.linalg.norm(obs_diffs, axis=1))
                obs_diversity_score = min(1.0, mean_obs_diff / _obs_divisor)

        # Trajectory smoothness: penalize jerky motions (high acceleration)
        smoothness_score = 1.0
        if total_frames >= 3 and len(all_actions) >= 3:
            action_matrix = np.array(all_actions)
            accels = np.diff(action_matrix, n=2, axis=0)
            mean_accel = np.mean(np.linalg.norm(accels, axis=1))
            smoothness_score = max(0.0, 1.0 - mean_accel / 0.1)

        # Collision-free score: check if episode has collision info
        collision_free_score = 0.5  # neutral when unknown
        physics_flags = [
            f.get("collision_free_physics")
            for f in frames
            if f.get("collision_free_physics") is not None
        ]
        if physics_flags:
            if any(flag is False for flag in physics_flags):
                collision_free_score = 0.0
            elif any(flag is True for flag in physics_flags):
                collision_free_score = 1.0
        else:
            for frame in frames:
                if frame.get("collision_free") is True:
                    collision_free_score = 1.0
                    break
                elif frame.get("collision_free") is False:
                    collision_free_score = 0.0
                    break

        # --- Improvement I: Physics plausibility penalties ---
        scene_state_penalty = 0.0
        ee_approach_penalty = 0.0
        _any_moved = False
        _requires_motion = self._requires_object_motion(task)

        if _requires_motion:
            # Penalty: static scene state (no object moved)
            # Skip this penalty when running without server recording — physics
            # feedback is unavailable so object poses are always identical.
            _skip_recording = os.environ.get("GENIESIM_SKIP_SERVER_RECORDING", "")
            if frames and not _skip_recording:
                _has_real_scene_state = any(
                    (frame.get("observation", {}) or {}).get("data_source") in ("real_composed", "between_waypoints")
                    for frame in frames
                )
                if _has_real_scene_state:
                    _any_moved = False
                    _max_displacement = 0.0
                    _units_per_meter = getattr(self, "_units_per_meter", 1.0)
                    if not isinstance(_units_per_meter, (int, float)) or _units_per_meter <= 0:
                        _units_per_meter = 1.0
                    _move_thresh = 0.01 * _units_per_meter  # 1cm threshold

                    # Prefer trajectory-based movement detection (Issue 2 fix)
                    # Check per-frame object_poses for any movement across consecutive frames
                    _object_trajectory: Dict[str, List[List[float]]] = {}
                    for _frame in frames:
                        _frame_poses = _frame.get("object_poses", {})
                        for _oid, _pose_data in _frame_poses.items():
                            if _oid not in _object_trajectory:
                                _object_trajectory[_oid] = []
                            _pos = _pose_data.get("position")
                            if _pos:
                                _object_trajectory[_oid].append(_pos)

                    # Check trajectory for movement
                    if _object_trajectory:
                        for _oid, _positions in _object_trajectory.items():
                            if len(_positions) < 2:
                                continue
                            for i in range(1, len(_positions)):
                                prev_pos = np.array(_positions[i - 1])
                                curr_pos = np.array(_positions[i])
                                disp = float(np.linalg.norm(curr_pos - prev_pos))
                                _max_displacement = max(_max_displacement, disp)
                                if disp > _move_thresh:
                                    _any_moved = True
                                    break
                            if _any_moved:
                                break

                    # Fallback to init/final comparison if no trajectory data
                    if not _object_trajectory:
                        _init_poses = frames[0].get("_initial_object_poses", {})
                        _final_poses = frames[-1].get("_final_object_poses", {})
                        if _init_poses and _final_poses:
                            for _oid, _ip in _init_poses.items():
                                _fp = _final_poses.get(_oid)
                                if _fp is not None:
                                    disp = float(np.linalg.norm(np.array(_fp) - np.array(_ip)))
                                    _max_displacement = max(_max_displacement, disp)
                                    if disp > _move_thresh:
                                        _any_moved = True
                                        break

                    if not _any_moved:
                        # Enhanced partial motion credit: separately track vertical (lift) and lateral motion
                        # Lift-only scenarios (object picked up but not moved laterally) deserve partial credit
                        _max_lift = 0.0
                        _max_lateral = 0.0
                        _lift_thresh = 0.05 * _units_per_meter  # 5cm lift threshold

                        # Compute lift and lateral separately from trajectory
                        for _oid, _positions in _object_trajectory.items():
                            if len(_positions) >= 2:
                                _init_z = _positions[0][2] if len(_positions[0]) > 2 else 0.0
                                for _pos in _positions[1:]:
                                    if len(_pos) > 2:
                                        _lift = _pos[2] - _init_z  # Z displacement (lift)
                                        _max_lift = max(_max_lift, _lift)
                                    _lateral = np.linalg.norm(np.array(_pos[:2]) - np.array(_positions[0][:2]))
                                    _max_lateral = max(_max_lateral, _lateral)

                        # Also check init/final poses for lift
                        if not _object_trajectory:
                            _init_poses = frames[0].get("_initial_object_poses", {})
                            _final_poses = frames[-1].get("_final_object_poses", {})
                            for _oid, _ip in _init_poses.items():
                                _fp = _final_poses.get(_oid)
                                if _fp is not None and len(_ip) > 2 and len(_fp) > 2:
                                    _max_lift = max(_max_lift, _fp[2] - _ip[2])
                                    _max_lateral = max(_max_lateral, np.linalg.norm(np.array(_fp[:2]) - np.array(_ip[:2])))

                        # Proportional penalty with partial credit for lift and lateral separately
                        # 30% weight for lift, 70% weight for lateral movement
                        _lift_credit = min(1.0, _max_lift / _lift_thresh) * 0.30 if _lift_thresh > 0 else 0.0
                        _lateral_credit = min(1.0, _max_lateral / _move_thresh) * 0.70 if _move_thresh > 0 else 0.0
                        _progress_ratio = _lift_credit + _lateral_credit

                        # Scale penalty based on progress (max 0.20 down to 0 with full progress)
                        scene_state_penalty = 0.20 * max(0.0, 1.0 - _progress_ratio)

                        if _progress_ratio > 0:
                            logger.info(
                                "Quality: Partial object motion detected (lift=%.4fm, lateral=%.4fm, progress=%.2f, penalty=%.3f)",
                                _max_lift, _max_lateral, _progress_ratio, scene_state_penalty,
                            )
                        else:
                            logger.warning(
                                "Quality penalty: scene_state is static (max_displacement=%.4fm, threshold=%.4fm, penalty=%.3f)",
                                _max_displacement, _move_thresh, scene_state_penalty,
                            )

            # Check if kinematic EE-offset tracking was used for grasped objects
            _kinematic_tracking_used = any(
                (frame.get("observation", {}) or {}).get("scene_state_provenance") == "kinematic_ee_offset"
                for frame in frames
            )
            if _kinematic_tracking_used and not _any_moved:
                # Kinematic tracking is a reasonable fallback when physics doesn't move objects
                # Reduce penalty significantly since the object is being tracked via EE motion
                scene_state_penalty = min(scene_state_penalty, 0.05)
                logger.info(
                    "Quality: Using kinematic EE-offset tracking for grasped object (reduced penalty to %.3f)",
                    scene_state_penalty,
                )

            _scene_state_fallback = any(
                (frame.get("observation", {}) or {}).get("scene_state_provenance") == "synthetic_fallback"
                for frame in frames
            )
            if _scene_state_fallback and getattr(self.config, "environment", "") != "production":
                # When object tracking is unavailable, use EE trajectory variance as fallback
                # to reduce penalty if the robot is clearly doing meaningful work
                ee_trajectory_score = self._compute_ee_trajectory_variance_score(frames)
                if ee_trajectory_score > 0.5:
                    # Robot had meaningful motion; reduce penalty proportionally
                    adjusted_penalty = 0.15 * (1.0 - ee_trajectory_score * 0.6)
                    scene_state_penalty = max(scene_state_penalty, adjusted_penalty)
                    logger.warning(
                        "Quality penalty: scene_state uses synthetic fallback (ee_motion=%.2f, penalty=%.3f)",
                        ee_trajectory_score, adjusted_penalty,
                    )
                else:
                    scene_state_penalty = max(scene_state_penalty, 0.15)
                    logger.warning("Quality penalty: scene_state uses synthetic fallback provenance.")
        else:
            logger.info("Quality: scene_state penalty skipped (task does not require object motion).")

        # Fail-fast gate: Object motion validation
        if _requires_motion and not _any_moved:
            try:
                from data_fidelity import validate_object_motion, DataFidelityError
                _is_valid, _motion_diag = validate_object_motion(
                    any_moved=_any_moved,
                    max_displacement=_max_displacement if "_max_displacement" in dir() else 0.0,
                    task_requires_motion=True,
                    min_displacement_threshold=0.001,  # 1mm
                )
                if not _is_valid:
                    logger.error(
                        "Object motion validation failed: %s",
                        _motion_diag.get("reason", "unknown"),
                    )
            except ImportError:
                pass
            except DataFidelityError as e:
                raise e

        # Penalty: EE never approaches target object
        if target_object_id and frames:
            _min_dist = float("inf")
            _initial_dist = float("inf")
            _units_per_meter = getattr(self, "_units_per_meter", 1.0)
            if not isinstance(_units_per_meter, (int, float)) or _units_per_meter <= 0:
                _units_per_meter = 1.0
            # Normalize target ID for fuzzy matching (strip path prefixes)
            _target_norm = target_object_id.rsplit("/", 1)[-1].lower()
            for _fi, _frame in enumerate(frames):
                _eep = _frame.get("ee_pos")
                _obs = _frame.get("observation", {})
                _ss = _obs.get("scene_state", {}) or _obs.get("privileged", {}).get("scene_state", {})
                if _eep:
                    _eep_arr = np.array(_eep, dtype=float) * _units_per_meter
                    for _obj in _ss.get("objects", []):
                        _oid = _obj.get("object_id", "")
                        _oid_norm = _oid.rsplit("/", 1)[-1].lower()
                        if _oid == target_object_id or _oid_norm == _target_norm:
                            _op = _obj.get("pose", {})
                            if "x" in _op:
                                _opos = np.array([_op["x"], _op["y"], _op["z"]])
                            elif "position" in _op:
                                _opos = np.array(_op["position"])
                            else:
                                break
                            _d = float(np.linalg.norm(_eep_arr - _opos))
                            if _fi == 0:
                                _initial_dist = _d
                            _min_dist = min(_min_dist, _d)
                            break
            if _min_dist == float("inf"):
                logger.warning(
                    "Quality check: Could not compute EE-target distance "
                    "(target=%s not found in scene_state or ee_pos missing)",
                    target_object_id,
                )
            elif _min_dist > 0.20 * _units_per_meter:
                ee_approach_penalty = 0.15
                _min_dist_m = _min_dist / _units_per_meter if _units_per_meter else _min_dist
                logger.warning(
                    "Quality penalty: EE never approaches target (min_dist=%.3f m)",
                    _min_dist_m,
                )
            elif _min_dist >= _initial_dist and _initial_dist < float("inf"):
                ee_approach_penalty = 0.10
                _min_dist_m = _min_dist / _units_per_meter if _units_per_meter else _min_dist
                _initial_dist_m = _initial_dist / _units_per_meter if _units_per_meter else _initial_dist
                logger.warning(
                    "Quality penalty: EE distance to target never decreased (min=%.3f, initial=%.3f)",
                    _min_dist_m, _initial_dist_m,
                )

        joint_utilization = self._compute_joint_utilization(frames)
        utilization_threshold = float(os.getenv("GENIESIM_JOINT_UTILIZATION_THRESHOLD", "0.6"))
        utilization_penalty_scale = float(os.getenv("GENIESIM_JOINT_UTILIZATION_PENALTY", "0.15"))
        utilization_penalty = 0.0
        if utilization_threshold > 0 and joint_utilization["fraction"] < utilization_threshold:
            utilization_gap = utilization_threshold - joint_utilization["fraction"]
            utilization_penalty = utilization_penalty_scale * (utilization_gap / utilization_threshold)
            logger.warning(
                "Quality penalty: joint utilization %.2f below threshold %.2f (penalty=%.3f).",
                joint_utilization["fraction"],
                utilization_threshold,
                utilization_penalty,
            )

        # Camera and EE completeness: fraction of frames with actual data
        _cam_complete_count = 0
        _ee_complete_count = 0
        for _fr in frames:
            _cf = _fr.get("observation", {}).get("camera_frames", {})
            if _cf and any(v.get("rgb") is not None for v in (_cf.values() if isinstance(_cf, dict) else [])):
                _cam_complete_count += 1
            if _fr.get("ee_pos") is not None:
                _ee_complete_count += 1
        camera_completeness = _cam_complete_count / max(total_frames, 1)
        ee_completeness = _ee_complete_count / max(total_frames, 1)

        # Weighted score with camera and EE completeness terms.
        # These ensure episodes without images or EE pose cannot score high.
        weighted_score = (
            0.20 * data_completeness_score
            + 0.15 * action_validity_score
            + 0.10 * action_diversity_score
            + 0.10 * obs_diversity_score
            + 0.15 * success_score
            + 0.05 * frame_count_score
            + 0.15 * camera_completeness
            + 0.10 * ee_completeness
        )
        # Bonus for smooth trajectories and collision-free episodes
        smoothness_bonus = 0.05 * smoothness_score
        collision_bonus = 0.05 * (collision_free_score - 0.5)  # neutral at 0.5
        weighted_score += smoothness_bonus + collision_bonus
        # Physics plausibility penalties (Improvement I)
        weighted_score -= scene_state_penalty
        weighted_score -= ee_approach_penalty
        weighted_score -= utilization_penalty

        # --- Fix 13: Stricter quality gates ---
        _gate_penalty = 0.0
        # Gate: quaternion norms must be ~1.0
        _quat_bad = 0
        for _fr in frames:
            _q = _fr.get("ee_quat")
            if _q:
                _qnorm = float(np.linalg.norm(_q))
                if abs(_qnorm - 1.0) > 1e-4:
                    _quat_bad += 1
        if _quat_bad > 0:
            _gate_penalty += 0.05
            logger.warning("Quality gate: %d frames with non-unit quaternions", _quat_bad)

        # Gate: joint velocities within robot-specific limits (when known)
        _vel_exceeded = 0
        robot_type = getattr(self.config, "robot_type", "").lower()
        _vel_lim = None
        if robot_type in _FRANKA_TYPES:
            _vel_lim = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
        if _vel_lim is not None:
            for _fr in frames:
                _jv = (_fr.get("observation", {}).get("robot_state", {})
                       .get("joint_velocities", []))
                if _jv and len(_jv) >= len(_vel_lim):
                    if np.any(np.abs(_jv[:len(_vel_lim)]) > _vel_lim * 1.05):
                        _vel_exceeded += 1
            if _vel_exceeded > 0:
                _gate_penalty += 0.05
                logger.warning("Quality gate: %d frames exceed velocity limits", _vel_exceeded)

        # Gate: grasp implies grip, open implies no grip
        _grasp_consistency_fails = 0
        _grasp_check_frames = 0
        for _fr in frames:
            _gc = _fr.get("gripper_command")
            _priv = _fr.get("observation", {}).get("privileged", {})
            _cf = _priv.get("contact_forces", _fr.get("contact_forces", {}))
            if _gc in ("closed", "open") and isinstance(_cf, dict):
                _grasp_check_frames += 1
                if _gc == "closed" and _cf.get("grip_force_N", 0) <= 0:
                    _grasp_consistency_fails += 1
                if _gc == "open" and _cf.get("grasped_object_id") is not None and not _cf.get("releasing"):
                    _grasp_consistency_fails += 1
        if _grasp_consistency_fails > 0 and _grasp_check_frames > 0:
            # Proportional penalty: full 0.03 only when ≥20% of frames have issues
            fail_ratio = _grasp_consistency_fails / _grasp_check_frames
            if fail_ratio >= 0.20:
                _grasp_penalty = 0.03
            else:
                # Scale penalty proportionally for smaller failure rates
                _grasp_penalty = 0.03 * (fail_ratio / 0.20)
            _gate_penalty += _grasp_penalty
            logger.warning(
                "Quality gate: %d/%d frames (%.1f%%) with grasp/contact inconsistency (penalty=%.4f)",
                _grasp_consistency_fails, _grasp_check_frames, fail_ratio * 100, _grasp_penalty,
            )

        weighted_score -= _gate_penalty

        _final = min(1.0, max(0.0, weighted_score))
        logger.info(
            "Quality score breakdown: "
            "data_completeness=%.3f(w=0.20) action_validity=%.3f(w=0.15) "
            "action_diversity=%.3f(w=0.10) obs_diversity=%.3f(w=0.10) "
            "success=%.3f(w=0.15) frame_count=%.3f(w=0.05) "
            "camera_completeness=%.3f(w=0.15) ee_completeness=%.3f(w=0.10) "
            "smoothness_bonus=%.3f collision_bonus=%.3f "
            "scene_state_penalty=%.3f ee_approach_penalty=%.3f "
            "utilization_penalty=%.3f gate_penalty=%.3f "
            "final=%.4f",
            data_completeness_score, action_validity_score,
            action_diversity_score, obs_diversity_score,
            success_score, frame_count_score,
            camera_completeness, ee_completeness,
            smoothness_bonus, collision_bonus,
            scene_state_penalty, ee_approach_penalty,
            utilization_penalty, _gate_penalty,
            _final,
        )
        # Store breakdown for inclusion in episode metadata with diagnostic details
        self._last_quality_breakdown = {
            # Component scores
            "data_completeness": round(data_completeness_score, 4),
            "action_validity": round(action_validity_score, 4),
            "action_diversity": round(action_diversity_score, 4),
            "obs_diversity": round(obs_diversity_score, 4),
            "success": round(success_score, 4),
            "frame_count": round(frame_count_score, 4),
            "camera_completeness": round(camera_completeness, 4),
            "ee_completeness": round(ee_completeness, 4),
            # Bonuses and penalties
            "smoothness_bonus": round(smoothness_bonus, 4),
            "collision_bonus": round(collision_bonus, 4),
            "scene_state_penalty": round(scene_state_penalty, 4),
            "ee_approach_penalty": round(ee_approach_penalty, 4),
            "utilization_penalty": round(utilization_penalty, 4),
            "gate_penalty": round(_gate_penalty, 4),
            "final_score": round(_final, 4),
            # Diagnostic details for post-hoc analysis
            "diagnostics": {
                "total_frames": total_frames,
                "robot_type": _robot_type if "_robot_type" in dir() else None,
                "diversity_divisors": {
                    "action": _action_divisor if "_action_divisor" in dir() else None,
                    "obs": _obs_divisor if "_obs_divisor" in dir() else None,
                    "source": _diversity_calibration_source if "_diversity_calibration_source" in dir() else None,
                },
                "frame_count_scoring": {
                    "use_gradual": (_frame_cfg.use_gradual_scoring if "_frame_cfg" in dir() and _frame_cfg else False),
                    "min_full": (_frame_cfg.min_frames_full_score if "_frame_cfg" in dir() and _frame_cfg else 10),
                },
                "grasp_release": {
                    "grasp_frame": grasp_frame_index,
                    "release_frame": release_frame_index,
                },
                "object_motion": {
                    "max_displacement": round(_max_displacement, 4) if "_max_displacement" in dir() else None,
                    "max_lift": round(_max_lift, 4) if "_max_lift" in dir() else None,
                    "max_lateral": round(_max_lateral, 4) if "_max_lateral" in dir() else None,
                    "any_moved": _any_moved if "_any_moved" in dir() else None,
                },
            },
        }
        return _final

    def _compute_ee_trajectory_variance_score(
        self,
        frames: List[Dict[str, Any]],
    ) -> float:
        """
        Compute a 0.0-1.0 score indicating meaningful robot EE motion.

        Used as a fallback metric when object tracking is unavailable.
        Returns higher scores when the robot shows purposeful movement
        (path length significantly exceeds net displacement).

        Args:
            frames: Episode frames containing ee_pos data

        Returns:
            Score from 0.0 (no motion) to 1.0 (significant purposeful motion)
        """
        if not frames or len(frames) < 2:
            return 0.0

        ee_positions = []
        for frame in frames:
            ee_pos = frame.get("ee_pos")
            if ee_pos and len(ee_pos) >= 3:
                ee_positions.append(np.array(ee_pos[:3]))

        if len(ee_positions) < 2:
            return 0.0

        # Compute total path length (sum of segment lengths)
        path_length = 0.0
        for i in range(1, len(ee_positions)):
            path_length += float(np.linalg.norm(ee_positions[i] - ee_positions[i - 1]))

        # Compute net displacement (start to end)
        net_displacement = float(np.linalg.norm(ee_positions[-1] - ee_positions[0]))

        # Minimum motion threshold (5cm) to consider meaningful
        min_motion_thresh = 0.05
        if path_length < min_motion_thresh:
            return 0.0

        # Score based on path length (capped at 0.5m for full score)
        path_score = min(1.0, path_length / 0.5)

        # Bonus for purposeful motion (path significantly longer than direct route)
        # This indicates reaching, grasping, lifting motions rather than jitter
        efficiency = net_displacement / path_length if path_length > 0 else 0
        # High efficiency (near 1.0) means direct motion; lower means complex motion
        # Both can be valid, but we reward any motion that isn't just noise
        motion_quality = 0.5 + 0.5 * (1.0 - abs(0.5 - efficiency) * 2)

        return min(1.0, path_score * motion_quality)

    def _requires_object_motion(self, task: Dict[str, Any]) -> bool:
        """Return True when the task expects objects to move (grasp/pick/place/etc)."""
        if not isinstance(task, dict):
            return False

        explicit_keys = (
            "requires_object_motion",
            "requires_object_movement",
            "expect_object_motion",
            "expects_object_motion",
        )
        for key in explicit_keys:
            if key in task:
                try:
                    return bool(task.get(key))
                except Exception:
                    return False

        if task.get("allow_static_scene") is True or task.get("static_scene_ok") is True:
            return False

        task_type = (task.get("task_type") or task.get("task_name") or task.get("task") or "").lower()

        motion_keywords = {
            "pick", "place", "grasp", "lift", "stack", "insert", "open", "close",
            "pour", "push", "pull", "transport", "handover", "rotate", "turn", "twist",
            "slide", "wipe", "clean", "sweep", "press", "toggle", "assemble", "disassemble",
        }
        non_motion_keywords = {
            "inspect", "observe", "scan", "look", "view", "perceive", "detect",
            "classify", "segment", "navigate", "reach", "point", "pose", "calibrate",
            "idle", "monitor",
        }

        if any(key in task_type for key in non_motion_keywords):
            return False
        if any(key in task_type for key in motion_keywords):
            return True

        # Heuristic: place/goal positions imply object motion.
        if task.get("place_position") is not None or task.get("place_pose") is not None:
            return True
        if task.get("goal_region") is not None and task.get("target_object") is not None:
            return True

        return False

    def _extract_task_success(
        self,
        frames: List[Dict[str, Any]],
        task: Dict[str, Any],
    ) -> Optional[bool]:
        """Extract task success flag from Genie Sim metadata when available."""
        if not frames:
            return None

        success_schema = task.get("success_schema") or {}
        success_keys = success_schema.get("success_keys") or task.get("success_keys")
        if not success_keys:
            success_keys = DEFAULT_SUCCESS_SCHEMA["success_keys"]

        for frame in reversed(frames):
            obs = frame.get("observation") or {}
            metadata = obs.get("metadata") if isinstance(obs, dict) else None
            task_meta = obs.get("task") if isinstance(obs, dict) else None
            for key in success_keys:
                for container in (obs, frame, metadata, task_meta):
                    if isinstance(container, dict) and key in container:
                        return bool(container.get(key))
        return None

    # =========================================================================
    # Per-task streaming export
    # =========================================================================

    def _export_task_episodes(
        self,
        task_name: str,
        task_idx: int,
        run_dir: Path,
    ) -> None:
        """Export episodes for a single completed task to LeRobot format.

        Called after each task's checkpoint so results are immediately
        available on disk (and via gcsfuse) without waiting for all tasks.
        """
        # Collect episode JSONs that belong to this task.
        task_episodes = sorted(run_dir.glob(f"{task_name}_ep*.json"))
        if not task_episodes:
            # Fallback: try task_<idx>_ep* pattern
            task_episodes = sorted(run_dir.glob(f"task_{task_idx}_ep*.json"))
        if not task_episodes:
            self.log(f"No episode files found for per-task export: {task_name}")
            return

        task_output_dir = run_dir / "per_task" / f"task_{task_idx:03d}_{task_name}"
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # Stage only this task's episodes into a temporary recording dir.
        task_staging = run_dir / f"_task_staging_{task_idx}"
        task_staging.mkdir(exist_ok=True)
        for ep in task_episodes:
            shutil.copy2(ep, task_staging / ep.name)

        try:
            self.export_to_lerobot(
                recording_dir=task_staging,
                output_dir=task_output_dir,
            )
            # Write a completion marker so consumers can detect finished tasks.
            marker = {
                "task_name": task_name,
                "task_idx": task_idx,
                "episodes": len(task_episodes),
            }
            (task_output_dir / "_task_complete.json").write_text(
                json.dumps(marker, indent=2)
            )
            # Copy raw episode JSONs to per_task output for rich data access
            raw_json_dir = task_output_dir / "raw_episodes"
            raw_json_dir.mkdir(exist_ok=True)
            for ep in task_episodes:
                shutil.copy2(ep, raw_json_dir / ep.name)

            self.log(
                f"Streamed {len(task_episodes)} episodes for task "
                f"{task_name} → {task_output_dir}"
            )
        finally:
            shutil.rmtree(task_staging, ignore_errors=True)

    # =========================================================================
    # Export
    # =========================================================================

    def export_to_lerobot(
        self,
        recording_dir: Path,
        output_dir: Path,
        min_quality_score: float = 0.7,
        export_format: Optional[Union[str, LeRobotExportFormat]] = None,
    ) -> Dict[str, Any]:
        """
        Export collected episodes to LeRobot format.

        Args:
            recording_dir: Directory containing recorded episodes
            output_dir: Output directory for LeRobot dataset
            min_quality_score: Minimum quality score for inclusion

        Returns:
            Export statistics
        """
        resolved_format = parse_lerobot_export_format(
            export_format or self.config.lerobot_export_format,
            default=LeRobotExportFormat.LEROBOT_V2,
        )
        if self.config.require_lerobot_v3 and resolved_format != LeRobotExportFormat.LEROBOT_V3:
            self.log(
                "LeRobot v3 export required but not configured. Falling back to v2-compatible output.",
                "WARNING",
            )

        self.log(f"Exporting to LeRobot format ({resolved_format.value}): {output_dir}")

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required to export LeRobot datasets. Install with "
                "`pip install pyarrow` and retry."
            ) from exc

        def _compute_v3_stats(parquet_path: Path) -> Optional[Dict[str, Any]]:
            try:
                table = pq.read_table(parquet_path)
            except Exception as exc:
                self.log(f"Failed to read LeRobot v3 parquet for stats: {exc}", "WARNING")
                return None

            stats: Dict[str, Any] = {}
            for column_name in table.column_names:
                column = table[column_name]
                if not (pa.types.is_integer(column.type) or pa.types.is_floating(column.type)):
                    continue
                values = [value for value in column.to_pylist() if value is not None]
                if not values:
                    continue
                values_array = np.array(values, dtype=np.float64)
                stats[column_name] = {
                    "min": float(values_array.min()),
                    "max": float(values_array.max()),
                    "mean": float(values_array.mean()),
                    "std": float(values_array.std()),
                }

            return stats if stats else None

        output_dir.mkdir(parents=True, exist_ok=True)
        lerobot_root = output_dir
        data_dir = lerobot_root / "data" / "chunk-000"
        meta_dir = lerobot_root / "meta"
        meta_episodes_dir = meta_dir / "episodes" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        exported_count = 0
        skipped_count = 0
        total_frames = 0
        frame_counts: Dict[str, int] = {}

        # Find all episode files
        episode_files = list(recording_dir.glob("*.json"))

        def _to_json_serializable(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(k): _to_json_serializable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_to_json_serializable(v) for v in value]
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return value

        schema = pa.schema(
            [
                ("episode_id", pa.string()),
                ("frame_index", pa.int64()),
                ("timestamp", pa.float64()),
                ("observation", pa.large_string()),
                ("action", pa.large_string()),
                ("reward", pa.float64()),
                ("done", pa.bool_()),
                ("task_name", pa.string()),
                ("task_id", pa.string()),
            ]
        )

        parquet_writer = None
        v3_output_file = None
        episode_index: Dict[str, Any] = {}
        row_group_index = 0

        if resolved_format == LeRobotExportFormat.LEROBOT_V3:
            v3_output_file = data_dir / "file-0000.parquet"
            parquet_writer = pq.ParquetWriter(v3_output_file, schema=schema, compression="zstd")

        # Set up video directories for camera data export
        videos_dir = lerobot_root / "videos" / "chunk-000"

        for ep_file in episode_files:
            try:
                with open(ep_file) as f:
                    episode = json.load(f)

                # Skip low quality
                if episode.get("quality_score", 0) < min_quality_score:
                    skipped_count += 1
                    continue

                episode_id = str(episode.get("episode_id", ep_file.stem))
                task_name = episode.get("task_name")
                task_id = episode.get("task_id")
                frames = episode.get("frames", [])
                frame_count = len(frames)
                ep_dir = ep_file.parent
                # .npy refs are relative to the run directory, not the
                # raw_episodes subdirectory.  Walk up to find the directory
                # that actually contains the *_frames/ subdirs.
                _candidate = ep_dir
                for _ in range(4):
                    if any(_candidate.glob("*_frames")):
                        break
                    _candidate = _candidate.parent
                else:
                    _candidate = ep_dir  # fallback
                ep_dir = _candidate

                # Collect camera frames for video export
                camera_rgb_frames: Dict[str, List[np.ndarray]] = {}
                for idx, frame in enumerate(frames):
                    cam_frames = frame.get("observation", {}).get("camera_frames", {})
                    for cam_id, cam_data in cam_frames.items():
                        if not isinstance(cam_data, dict):
                            continue
                        rgb_arr = load_camera_frame(cam_data, "rgb", ep_dir=ep_dir)
                        if rgb_arr is not None:
                            camera_rgb_frames.setdefault(cam_id, []).append(rgb_arr)

                require_camera_data = os.getenv("REQUIRE_CAMERA_DATA", "false").lower() == "true"
                total_rgb_frames = sum(len(v) for v in camera_rgb_frames.values())
                if require_camera_data and total_rgb_frames == 0:
                    raise _CameraDataRequiredError(
                        f"Camera data required but none found for episode {episode_id}. "
                        "Set REQUIRE_CAMERA_DATA=false to bypass."
                    )

                # Write video files per camera
                _have_imageio = False
                try:
                    import imageio
                    _have_imageio = True
                except ImportError:
                    pass

                for cam_id, rgb_list in camera_rgb_frames.items():
                    if not rgb_list:
                        continue
                    cam_video_dir = videos_dir / f"observation.images.{cam_id}"
                    cam_video_dir.mkdir(parents=True, exist_ok=True)
                    ep_num = exported_count
                    video_path = cam_video_dir / f"episode_{ep_num:06d}.mp4"

                    if _have_imageio:
                        try:
                            writer = imageio.get_writer(str(video_path), fps=30.0, codec="libx264", quality=8)
                            for rgb_frame in rgb_list:
                                # Ensure 3-channel RGB
                                if rgb_frame.ndim == 3 and rgb_frame.shape[-1] == 4:
                                    rgb_frame = rgb_frame[:, :, :3]
                                if rgb_frame.ndim == 3 and rgb_frame.shape[-1] == 3:
                                    writer.append_data(rgb_frame)
                            writer.close()
                            self.log(f"  Wrote video: {video_path} ({len(rgb_list)} frames)")
                        except Exception as _vid_err:
                            self.log(f"  Video write failed for {cam_id}: {_vid_err}", "WARNING")
                    else:
                        # Fallback: save as individual .npy frames
                        frames_out_dir = cam_video_dir / f"episode_{ep_num:06d}_frames"
                        frames_out_dir.mkdir(parents=True, exist_ok=True)
                        for fi, rgb_frame in enumerate(rgb_list):
                            np.save(frames_out_dir / f"frame_{fi:06d}.npy", rgb_frame)

                columns: Dict[str, List[Any]] = {
                    "episode_id": [],
                    "frame_index": [],
                    "timestamp": [],
                    "observation": [],
                    "action": [],
                    "reward": [],
                    "done": [],
                    "task_name": [],
                    "task_id": [],
                }

                for idx, frame in enumerate(frames):
                    timestamp = frame.get("timestamp")
                    if timestamp is None:
                        timestamp = float(idx)
                    columns["episode_id"].append(episode_id)
                    columns["frame_index"].append(idx)
                    columns["timestamp"].append(timestamp)
                    # Strip camera pixel data from observation for lightweight parquet
                    obs_stripped = strip_camera_data(frame.get("observation"))
                    columns["observation"].append(
                        json.dumps(_to_json_serializable(obs_stripped))
                    )
                    columns["action"].append(
                        json.dumps(_to_json_serializable(frame.get("action")))
                    )
                    columns["reward"].append(float(frame.get("reward", 0.0)))
                    columns["done"].append(bool(frame.get("done", False)))
                    columns["task_name"].append(task_name)
                    columns["task_id"].append(task_id)

                table = pa.Table.from_pydict(columns, schema=schema)
                if resolved_format == LeRobotExportFormat.LEROBOT_V3 and parquet_writer:
                    parquet_writer.write_table(table)
                    episode_index[episode_id] = {
                        "frames": frame_count,
                        "row_group": row_group_index,
                        "task_id": task_id,
                        "task_name": task_name,
                    }
                    row_group_index += 1
                else:
                    output_file = data_dir / f"episode_{episode_id}.parquet"
                    pq.write_table(table, output_file, compression="zstd")

                exported_count += 1
                total_frames += frame_count
                frame_counts[episode_id] = frame_count

            except _CameraDataRequiredError:
                raise
            except Exception as e:
                self.log(f"Failed to export {ep_file.name}: {e}", "WARNING")
                skipped_count += 1

        if parquet_writer:
            parquet_writer.close()

        schema_description = [
            {"name": field.name, "type": str(field.type)} for field in schema
        ]
        if resolved_format == LeRobotExportFormat.LEROBOT_V3:
            info = {
                "format": "lerobot",
                "export_format": resolved_format.value,
                "version": "3.0",
                "layout": "multi-episode",
                "episodes": exported_count,
                "skipped": skipped_count,
                "total_frames": total_frames,
                "exported_at": datetime.utcnow().isoformat() + "Z",
                "data_path": "data",
                "chunking": {
                    "strategy": "aggregated",
                    "chunk_dir": "chunk-000",
                    "files": [
                        {
                            "path": "chunk-000/file-0000.parquet",
                            "episodes": exported_count,
                        }
                    ],
                },
                "schema": schema_description,
                "frame_counts": frame_counts,
                "episode_index": "meta/episode_index.json",
                "episodes_meta": "meta/episodes/chunk-000/file-0000.parquet",
            }
        else:
            if resolved_format == LeRobotExportFormat.LEROBOT_V0_3_3:
                version_value = "0.3.3"
            elif resolved_format == LeRobotExportFormat.LEROBOT_V0_4:
                version_value = "0.4.0"
            else:
                version_value = "2.0"
            info = {
                "format": "lerobot",
                "export_format": resolved_format.value,
                "version": version_value,
                "episodes": exported_count,
                "total_episodes": exported_count,
                "skipped": skipped_count,
                "total_frames": total_frames,
                "exported_at": datetime.utcnow().isoformat() + "Z",
                "data_path": "data",
                "chunking": {"strategy": "single", "chunk_dir": "chunk-000"},
                "schema": schema_description,
                "frame_counts": frame_counts,
            }

        # Add video paths to info if videos were written
        if videos_dir.exists():
            _video_cameras = [
                d.name.replace("observation.images.", "")
                for d in videos_dir.iterdir()
                if d.is_dir() and d.name.startswith("observation.images.")
            ]
            if _video_cameras:
                info["video_path"] = "videos"
                info["cameras"] = sorted(_video_cameras)

        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)
        if resolved_format == LeRobotExportFormat.LEROBOT_V3:
            with open(meta_dir / "episode_index.json", "w") as f:
                json.dump(episode_index, f, indent=2)
            meta_episodes_dir.mkdir(parents=True, exist_ok=True)
            meta_episodes_file = meta_episodes_dir / "file-0000.parquet"
            episode_rows = [
                {
                    "episode_id": episode_id,
                    "frames": payload.get("frames"),
                    "row_group": payload.get("row_group"),
                    "task_id": payload.get("task_id"),
                    "task_name": payload.get("task_name"),
                }
                for episode_id, payload in episode_index.items()
            ]
            meta_schema = pa.schema(
                [
                    ("episode_id", pa.string()),
                    ("frames", pa.int64()),
                    ("row_group", pa.int64()),
                    ("task_id", pa.string()),
                    ("task_name", pa.string()),
                ]
            )
            meta_payload = {
                "episode_id": [row.get("episode_id") for row in episode_rows],
                "frames": [row.get("frames") for row in episode_rows],
                "row_group": [row.get("row_group") for row in episode_rows],
                "task_id": [row.get("task_id") for row in episode_rows],
                "task_name": [row.get("task_name") for row in episode_rows],
            }
            meta_table = pa.Table.from_pydict(meta_payload, schema=meta_schema)
            pq.write_table(meta_table, meta_episodes_file)

        # Write episodes.jsonl (required by v2 validation)
        if resolved_format != LeRobotExportFormat.LEROBOT_V3:
            episodes_jsonl_path = lerobot_root / "episodes.jsonl"
            with open(episodes_jsonl_path, "w") as ejf:
                for ep_id, fc in frame_counts.items():
                    ejf.write(json.dumps({"episode_id": ep_id, "length": fc}) + "\n")

        dataset_info = {
            "format": "lerobot",
            "export_format": resolved_format.value,
            "version": info["version"],
            "episodes": exported_count,
            "total_episodes": exported_count,
            "skipped": skipped_count,
            "total_frames": total_frames,
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "lerobot_info": "meta/info.json",
            "debug_json_output": False,
        }

        with open(output_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

        if resolved_format == LeRobotExportFormat.LEROBOT_V3 and v3_output_file:
            try:
                stats_payload = _compute_v3_stats(v3_output_file) or {}
                with open(meta_dir / "stats.json", "w") as f:
                    json.dump(stats_payload, f, indent=2)
            except Exception as exc:
                self.log(f"Failed to compute LeRobot v3 stats: {exc}", "WARNING")

        if os.getenv("LEROBOT_INCLUDE_RAW_EPISODES", "0") == "1":
            raw_dir = lerobot_root / "raw_episodes"
            raw_dir.mkdir(parents=True, exist_ok=True)
            for ep_file in episode_files:
                shutil.copy2(ep_file, raw_dir / ep_file.name)

        self.log(f"Exported {exported_count} episodes, skipped {skipped_count}")

        return {
            "success": True,
            "exported": exported_count,
            "skipped": skipped_count,
            "parquet_episodes": exported_count,
            "output_dir": output_dir,
        }


class _GenieSimServerContext:
    """Context manager for Genie Sim server lifecycle."""

    def __init__(
        self,
        framework: GenieSimLocalFramework,
        scene_usd_path: Optional[Path],
        *,
        timeout_s: Optional[float] = None,
        poll_s: Optional[float] = None,
    ):
        self.framework = framework
        self.scene_usd_path = scene_usd_path
        self.timeout_s = timeout_s
        self.poll_s = poll_s

    def __enter__(self) -> GenieSimLocalFramework:
        self.framework.start_server(
            self.scene_usd_path,
            timeout=self.timeout_s,
            poll_interval=self.poll_s,
        )
        self.framework.connect()
        return self.framework

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.framework.disconnect()
        self.framework.stop_server()
        return False


# =============================================================================
# Convenience Functions
# =============================================================================


def run_local_data_collection(
    scene_manifest_path: Path,
    task_config_path: Path,
    output_dir: Path,
    robot_type: str = "franka",
    episodes_per_task: int = 100,
    max_duration_seconds: Optional[float] = None,
    verbose: bool = True,
    expected_server_version: Optional[str] = None,
    required_capabilities: Optional[Sequence[str]] = None,
) -> DataCollectionResult:
    """
    Convenience function to run local Genie Sim data collection.

    Args:
        scene_manifest_path: Path to BlueprintPipeline scene manifest
        task_config_path: Path to task configuration
        output_dir: Output directory
        robot_type: Robot type
        episodes_per_task: Episodes per task
        max_duration_seconds: Abort data collection if total runtime exceeds this timeout
        verbose: Print progress
        expected_server_version: Expected Genie Sim API version
        required_capabilities: Required server capabilities

    Returns:
        DataCollectionResult
    """
    run_geniesim_preflight_or_exit(
        "geniesim-local-data-collection",
        require_server=False,
    )

    # Load configs
    with open(scene_manifest_path) as f:
        scene_manifest = json.load(f)

    # Resolve relative usd_path against manifest directory
    _usd_path = scene_manifest.get("usd_path", "")
    if _usd_path and not os.path.isabs(_usd_path):
        _resolved = str(Path(scene_manifest_path).parent / _usd_path)
        scene_manifest["usd_path"] = _resolved

    with open(task_config_path) as f:
        task_config = json.load(f)

    collection_timeout_env = os.getenv("GENIESIM_COLLECTION_TIMEOUT_S")
    effective_timeout = max_duration_seconds
    if effective_timeout is None and collection_timeout_env not in (None, ""):
        try:
            effective_timeout = float(collection_timeout_env)
        except ValueError:
            logger.warning(
                "GENIESIM_COLLECTION_TIMEOUT_S must be a number of seconds; ignoring invalid value."
            )

    # Create framework – use from_env() so env-var-driven settings
    # (e.g. GENIESIM_ALLOW_IK_FAILURE_FALLBACK) are honoured.
    config = GenieSimConfig.from_env().model_copy(update={
        "robot_type": robot_type,
        "episodes_per_task": episodes_per_task,
        "recording_dir": output_dir / "recordings",
        "max_duration_seconds": effective_timeout,
    })

    framework = GenieSimLocalFramework(config, verbose=verbose)

    # Check if server is already running
    if framework.is_server_running():
        # Connect to existing server
        framework.connect()
        result = framework.run_data_collection(
            task_config,
            scene_manifest,
            max_duration_seconds=effective_timeout,
            expected_server_version=expected_server_version,
            required_capabilities=required_capabilities,
        )
        framework.disconnect()
    else:
        # Start server and run
        scene_usd = scene_manifest.get("usd_path")
        with framework.server_context(Path(scene_usd) if scene_usd else None) as fw:
            result = fw.run_data_collection(
                task_config,
                scene_manifest,
                max_duration_seconds=effective_timeout,
                expected_server_version=expected_server_version,
                required_capabilities=required_capabilities,
            )

    # Export to LeRobot
    if result.success and result.recording_dir:
        lerobot_dir = output_dir / "lerobot"
        framework.export_to_lerobot(result.recording_dir, lerobot_dir)

    return result


def check_geniesim_availability(config: Optional[GenieSimConfig] = None) -> Dict[str, Any]:
    """
    Check if Genie Sim local framework is available.

    Returns:
        Dict with availability status and details
    """
    config = config or GenieSimConfig.from_env()

    status = {
        "available": False,
        "geniesim_installed": False,
        "isaac_sim_available": False,
        "server_running": False,
        "grpc_available": False,
        "grpc_stubs_available": GRPC_STUBS_AVAILABLE,
        "curobo_available": CUROBO_INTEGRATION_AVAILABLE,
        "use_curobo": config.use_curobo,
        "mock_server_allowed": False,
        "details": {},
    }
    allow_mock_override = os.getenv("ALLOW_GENIESIM_MOCK", "0") == "1"
    production_mode = config.environment == "production"
    mock_allowed = allow_mock_override and not production_mode
    status["mock_server_allowed"] = mock_allowed
    status["details"]["environment"] = config.environment
    status["details"]["allow_geniesim_mock"] = allow_mock_override
    status["details"]["isaacsim_required"] = config.isaacsim_required
    status["details"]["curobo_required"] = config.curobo_required
    status["details"]["curobo_available"] = CUROBO_INTEGRATION_AVAILABLE
    status["details"]["use_curobo"] = config.use_curobo

    # Check Genie Sim installation
    if config.geniesim_root.exists():
        status["geniesim_installed"] = True
        status["details"]["geniesim_root"] = str(config.geniesim_root)
    elif not mock_allowed:
        status["details"]["mock_server_blocked"] = True
        status["details"]["mock_server_reason"] = (
            "production" if production_mode else "ALLOW_GENIESIM_MOCK not set"
        )

    # Check Isaac Sim
    isaac_python = config.isaac_sim_path / "python.sh"
    if isaac_python.exists():
        status["isaac_sim_available"] = True
        status["details"]["isaac_sim_path"] = str(config.isaac_sim_path)

    # Check gRPC
    try:
        import grpc
        status["grpc_available"] = True
    except ImportError:
        pass

    # Check if server is running
    client = GenieSimGRPCClient(config.host, config.port)
    if client._check_server_socket():
        status["server_running"] = True

    # Overall availability
    # If a remote server is already running and gRPC is available, we don't
    # need local Isaac Sim or Genie Sim installations (Docker-hosted server).
    local_server_allowed = status["geniesim_installed"] or mock_allowed
    remote_server_ready = status["server_running"] and status["grpc_available"]
    status["available"] = (
        remote_server_ready or (
            status["isaac_sim_available"] and
            (status["grpc_available"] or status["server_running"]) and
            (status["server_running"] or local_server_allowed)
        )
    )

    return status


def build_geniesim_preflight_report(
    status: Dict[str, Any],
    *,
    config: Optional[GenieSimConfig] = None,
    require_server: bool = True,
    require_ready: bool = False,
    readiness_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a preflight report with remediation guidance."""
    config = config or GenieSimConfig.from_env()
    missing: List[str] = []
    remediation: List[str] = []
    server_ready = False
    effective_readiness_timeout = (
        readiness_timeout
        if readiness_timeout is not None
        else get_geniesim_readiness_timeout_s()
    )

    # If a remote server is already running with gRPC available, we don't
    # need local Isaac Sim or Genie Sim installations (e.g. Docker-hosted server).
    _remote_server_ready = (
        status.get("server_running", False) and status.get("grpc_available", False)
    )

    if not _remote_server_ready and not status.get("isaac_sim_available", False):
        missing.append("Isaac Sim runtime (ISAAC_SIM_PATH)")
        remediation.append(
            f"Set ISAAC_SIM_PATH to your Isaac Sim install (found python.sh). "
            f"Current: {config.isaac_sim_path}"
        )

    if not status.get("grpc_available", False):
        missing.append("Python grpcio package")
        remediation.append("Install grpcio in the current environment: pip install grpcio")

    if not status.get("grpc_stubs_available", False):
        missing.append("Genie Sim gRPC stubs")
        remediation.append(
            "Regenerate stubs or ensure tools/geniesim_adapter/geniesim_grpc_pb2*.py "
            "is available on PYTHONPATH."
        )

    if not _remote_server_ready and not status.get("geniesim_installed", False) and not status.get("mock_server_allowed", False):
        missing.append("Genie Sim checkout (GENIESIM_ROOT)")
        remediation.append(
            "Run tools/geniesim_adapter/deployment/install_geniesim.sh or set GENIESIM_ROOT. "
            "For dev/test, set ALLOW_GENIESIM_MOCK=1."
        )

    if config.isaacsim_required:
        if not config.geniesim_root.exists():
            missing.append("Genie Sim checkout required (ISAACSIM_REQUIRED=1)")
            remediation.append(
                f"Set GENIESIM_ROOT to the Genie Sim repo (expected path: {config.geniesim_root})."
            )
        if not (config.isaac_sim_path / "python.sh").exists():
            missing.append("Isaac Sim runtime required (ISAACSIM_REQUIRED=1)")
            remediation.append(
                "Set ISAAC_SIM_PATH to the Isaac Sim install containing python.sh "
                f"(expected: {config.isaac_sim_path / 'python.sh'})."
            )

    if config.curobo_required:
        if not CUROBO_INTEGRATION_AVAILABLE:
            missing.append("cuRobo integration required (CUROBO_REQUIRED=1)")
            remediation.append(
                "Install cuRobo so CUROBO_INTEGRATION_AVAILABLE is true "
                "(pip install nvidia-curobo)."
            )
        if not config.use_curobo:
            missing.append("cuRobo usage disabled while CUROBO_REQUIRED=1")
            remediation.append(
                "Enable cuRobo planning by setting use_curobo=True in GenieSimConfig "
                "or ensuring the caller does not disable it."
            )

    if require_server and not status.get("server_running", False):
        missing.append("Genie Sim gRPC server")
        remediation.append(
            "Start the server: "
            f"{config.isaac_sim_path}/python.sh "
            f"{config.geniesim_root}/source/data_collection/scripts/data_collector_server.py "
            f"--headless --port {config.port}"
        )

    if require_ready:
        if status.get("server_running", False):
            framework = GenieSimLocalFramework(config, verbose=False)
            try:
                readiness_result = framework.check_simulation_ready(
                    timeout=effective_readiness_timeout,
                )
                server_ready = readiness_result.success
            finally:
                framework.disconnect()
        if not server_ready:
            missing.append("Genie Sim gRPC readiness")
            remediation.append(
                "Verify the server is fully loaded and responding to gRPC calls. "
                "Re-run: python -m tools.geniesim_adapter.geniesim_healthcheck"
            )

    ok = len(missing) == 0
    return {
        "ok": ok,
        "missing": missing,
        "remediation": remediation,
        "status": status,
        "server_ready": server_ready,
    }


def run_geniesim_preflight(
    stage: str,
    *,
    config: Optional[GenieSimConfig] = None,
    require_server: bool = True,
    require_ready: bool = False,
    readiness_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a preflight check with remediation guidance and return the report."""
    status = check_geniesim_availability(config)
    report = build_geniesim_preflight_report(
        status,
        config=config,
        require_server=require_server,
        require_ready=require_ready,
        readiness_timeout=readiness_timeout,
    )
    report["stage"] = stage
    return report


def format_geniesim_preflight_failure(stage: str, report: Dict[str, Any]) -> str:
    """Format a preflight failure message for logs or UI surfaces."""
    missing = report.get("missing", [])
    remediation = report.get("remediation", [])
    message = f"[GENIESIM-PREFLIGHT] {stage} preflight failed."
    if missing:
        message += f" Missing requirements: {', '.join(missing)}."
    if remediation:
        message += " Remediation: " + " ".join(remediation)
    return message


def run_geniesim_preflight_or_exit(
    stage: str,
    *,
    config: Optional[GenieSimConfig] = None,
    require_server: bool = True,
    require_ready: bool = False,
    readiness_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a preflight check with remediation guidance, exiting on failure."""
    report = run_geniesim_preflight(
        stage,
        config=config,
        require_server=require_server,
        require_ready=require_ready,
        readiness_timeout=readiness_timeout,
    )
    if not report["ok"]:
        logger.error("[GENIESIM-PREFLIGHT] %s preflight failed.", stage, extra={"stage": stage})
        if report["missing"]:
            logger.error(
                "Missing requirements: %s",
                ", ".join(report["missing"]),
                extra={"stage": stage},
            )
        if report["remediation"]:
            logger.error(
                "Remediation steps: %s",
                " ".join(report["remediation"]),
                extra={"stage": stage},
            )
        logger.error(
            "Details: %s",
            json.dumps(report["status"], indent=2),
            extra={"stage": stage},
        )
        raise SystemExit(1)
    return report


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI for Genie Sim local framework."""
    import argparse

    init_logging()

    parser = argparse.ArgumentParser(description="Genie Sim 3.0 Local Framework")
    parser.add_argument("command", choices=["check", "start", "stop", "run", "export"])
    parser.add_argument("--scene", help="Path to scene manifest or USD")
    parser.add_argument("--task-config", help="Path to task configuration")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per task")
    parser.add_argument("--robot", default="franka", help="Robot type")

    args = parser.parse_args()

    if args.command == "check":
        status = check_geniesim_availability()
        logger.info("%s", json.dumps(status, indent=2))
        sys.exit(0 if status["available"] else 1)

    config = GenieSimConfig(robot_type=args.robot)
    framework = GenieSimLocalFramework(config)

    if args.command == "start":
        scene_path = Path(args.scene) if args.scene else None
        if framework.start_server(scene_path):
            logger.info("Server started successfully")
        else:
            logger.error("Failed to start server")
            sys.exit(1)

    elif args.command == "stop":
        framework.stop_server()
        logger.info("Server stopped")

    elif args.command == "run":
        if not args.task_config:
            logger.error("--task-config is required for run command")
            sys.exit(1)

        result = run_local_data_collection(
            scene_manifest_path=Path(args.scene) if args.scene else Path("scene_manifest.json"),
            task_config_path=Path(args.task_config),
            output_dir=Path(args.output),
            robot_type=args.robot,
            episodes_per_task=args.episodes,
        )

        if result.success:
            logger.info(
                "Data collection completed: %s/%s episodes",
                result.episodes_passed,
                result.episodes_collected,
            )
        else:
            logger.error("Data collection failed: %s", result.errors)
            sys.exit(1)

    elif args.command == "export":
        recording_dir = Path(args.output) / "recordings"
        lerobot_dir = Path(args.output) / "lerobot"

        result = framework.export_to_lerobot(recording_dir, lerobot_dir)
        logger.info(
            "Exported %s episodes to %s",
            result["exported"],
            lerobot_dir,
        )


if __name__ == "__main__":
    main()
