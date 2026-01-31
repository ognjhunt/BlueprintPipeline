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
    GENIESIM_STALL_TIMEOUT_S: Abort/reset episode if observations stall (default: 30)
    GENIESIM_MAX_STALLS: Max stalled episodes before server restart (default: 2)
    GENIESIM_STALL_BACKOFF_S: Backoff between stall handling attempts (default: 5)
    GENIESIM_COLLECTION_TIMEOUT_S: Abort data collection if total runtime exceeds this timeout (default: unset)
    GENIESIM_STARTUP_TIMEOUT_S: Startup timeout in seconds for server readiness (default: 120)
    GENIESIM_STARTUP_POLL_S: Poll interval in seconds for server readiness (default: 2)
    GENIESIM_CLEANUP_TMP: Remove Genie Sim temp directories after a run (default: 1 for local, 0 in production)
    GENIESIM_VALIDATE_FRAMES: Validate recorded frames before saving (default: 0)
    GENIESIM_FAIL_ON_FRAME_VALIDATION: Fail episode when frame validation errors exist (default: 0)
"""

import base64
import binascii
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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

logger = logging.getLogger(__name__)

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
except ImportError:
    GRPC_STUBS_AVAILABLE = False
    grpc = None
    logger.warning("gRPC stubs not available - using legacy fallback")
    _GRPC_PLACEHOLDER_NAMES = [
        "AddCameraReq",
        "AddCameraRsp",
        "AttachReq",
        "AttachRsp",
        "CameraRequest",
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
_ROBOT_METADATA_FALLBACK = {
    "franka": _FRANKA_METADATA, "franka_panda": _FRANKA_METADATA, "panda": _FRANKA_METADATA,
}


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


class GenieSimServerStatus(str, Enum):
    """Status of the Genie Sim server."""
    NOT_RUNNING = "not_running"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


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
    stall_timeout_s: StrictFloat = 90.0
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
        self._grpc_unavailable_logged: set[str] = set()
        self._camera_missing_logged: set[str] = set()
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
    ) -> Any:
        with self._grpc_lock:
            return self._call_grpc_inner(action, func, fallback, success_checker)

    def _call_grpc_inner(
        self,
        action: str,
        func: Callable[[], Any],
        fallback: Any,
        success_checker: Optional[Callable[[Any], bool]] = None,
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
            try:
                result = func()
            except Exception as exc:
                last_exception = exc
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure(exc)

                if self._is_retryable_grpc_error(exc) and attempt < self._grpc_retry_config.max_retries:
                    delay = calculate_delay(attempt, self._grpc_retry_config)
                    logger.warning(
                        "Genie Sim gRPC %s failed with retryable error (%s); retrying in %.2fs",
                        action,
                        exc,
                        delay,
                    )
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
                return True
            except grpc.FutureTimeoutError:
                logger.error(f"Connection timeout after {self.timeout}s")
                self._connected = False
                return False

        except Exception as e:
            logger.error(f"Failed to connect to Genie Sim server: {e}")
            self._connected = False
            return False

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
                jp_result = self.get_joint_position()
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
        # Resolve logical names (e.g. "wrist") to USD prim paths via map
        if self._camera_prim_map:
            camera_ids = [self._camera_prim_map.get(cid, cid) for cid in camera_ids]
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
                return self._stub.reset(request, timeout=self.timeout)

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
                return self._stub.init_robot(request, timeout=self.timeout)

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

    def get_observation(self) -> GrpcCallResult:
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
        joint_positions = []
        joint_names = []
        joint_velocities = []
        joint_efforts = []
        try:
            jp_result = self.get_joint_position()
            if jp_result.success and jp_result.payload is not None:
                joint_positions = list(jp_result.payload)
                joint_names = list(self._joint_names) if self._joint_names else []
                # get_joint_position() now also stores velocities and efforts
                joint_velocities = getattr(self, "_latest_joint_velocities", [])
                joint_efforts = getattr(self, "_latest_joint_efforts", [])
        except Exception as exc:
            logger.warning(f"[OBS] get_joint_position failed: {exc}")

        # --- 2. Real EE pose ---
        ee_pose = {}
        try:
            ee_result = self.get_ee_pose(ee_link_name="right")
            if ee_result.success and ee_result.payload is not None:
                ee_pose = ee_result.payload
        except Exception as exc:
            logger.warning(f"[OBS] get_ee_pose failed: {exc}")

        # --- 3. Real object poses via SimObjectService ---
        scene_objects = []
        # Use the task_config object prims if available
        object_prims = getattr(self, "_scene_object_prims", [])
        if object_prims and self._channel is not None:
            for prim_path in object_prims:
                # Try the prim path as-is and with common prefixes
                _candidates = [prim_path]
                if not prim_path.startswith("/World/"):
                    _candidates.append(f"/World/{prim_path}")
                if not prim_path.startswith("/"):
                    _candidates.append(f"/{prim_path}")
                _found = False
                for _candidate in _candidates:
                    try:
                        obj_pose = self._get_object_pose_raw(_candidate)
                        if obj_pose is not None:
                            scene_objects.append({
                                "object_id": prim_path,
                                "pose": obj_pose,
                            })
                            _found = True
                            logger.info(f"[OBS] Got real object pose for {_candidate}: {obj_pose}")
                            break
                    except Exception as exc:
                        logger.debug(f"[OBS] get_object_pose({_candidate}) failed: {exc}")
                if not _found:
                    logger.warning(f"[OBS] No object pose for any variant of {prim_path} (tried: {_candidates})")

        # --- 4. Camera images ---
        # Requires patched server (see deployment/patches/patch_camera_handler.py).
        # On unpatched servers, get_camera_data returns None and we fall back gracefully.
        camera_images = []
        if self._channel is not None:
            _cam_prims = list(self._camera_prim_map.values()) if self._camera_prim_map else []
            if not _cam_prims:
                _cam_prims = self._default_camera_ids  # logical names used as-is
            for _cam_prim in _cam_prims:
                try:
                    cam_data = self._get_camera_data_raw(_cam_prim)
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
            data_sources.append("joints")
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

    def _get_object_pose_raw(self, prim_path: str) -> Optional[Dict[str, Any]]:
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
            return self._parse_object_pose_response(raw_response)
        except Exception as exc:
            logger.debug(f"[OBS] raw get_object_pose failed: {exc}")
            return None

    def _get_camera_data_raw(self, cam_prim_path: str) -> Optional[Dict[str, Any]]:
        """Get camera data via CameraService.get_camera_data (raw gRPC)."""
        # Build GetCameraDataRequest: field 1 (serial_no=cam_prim_path, string)
        prim_bytes = cam_prim_path.encode("utf-8")
        payload = b"\x0a" + self._encode_varint(len(prim_bytes)) + prim_bytes

        method = "/aimdk.protocol.CameraService/get_camera_data"
        try:
            call = self._channel.unary_unary(
                method,
                request_serializer=lambda x: x,
                response_deserializer=lambda x: x,
            )
            raw_response = call(payload, timeout=self.timeout)
            return self._parse_camera_data_response(raw_response)
        except Exception as exc:
            logger.debug(f"[OBS] raw get_camera_data failed: {exc}")
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
        import struct as _struct
        fields = self._parse_protobuf_fields(data)
        result: Dict[str, Any] = {"rgb": b"", "depth": b"", "width": 0, "height": 0}
        # color_info (field 2) — CameraInfo: width(1,int32), height(2,int32)
        if 2 in fields:
            info_fields = self._parse_protobuf_fields(fields[2][0][1])
            if 1 in info_fields:
                result["width"] = int.from_bytes(info_fields[1][0][1], 'little')
            if 2 in info_fields:
                result["height"] = int.from_bytes(info_fields[2][0][1], 'little')
        # color_image (field 3) — CompressedImage: format(2,string), data(3,bytes)
        if 3 in fields:
            img_fields = self._parse_protobuf_fields(fields[3][0][1])
            if 3 in img_fields:
                result["rgb"] = img_fields[3][0][1]
        # depth_image (field 5) — CompressedImage
        if 5 in fields:
            img_fields = self._parse_protobuf_fields(fields[5][0][1])
            if 3 in img_fields:
                result["depth"] = img_fields[3][0][1]
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

        def _request() -> joint_channel_pb2.SetJointRsp:
            request = joint_channel_pb2.SetJointReq(
                commands=commands,
                is_trajectory=False,
            )
            return self._joint_stub.set_joint_position(request, timeout=self.timeout)

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
        return GrpcCallResult(
            success=True,
            available=True,
            payload={"msg": response.errmsg},
        )

    def get_joint_position(self) -> GrpcCallResult:
        """
        Get current joint positions.

        Uses real gRPC get_joint_position call.

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
        self._latest_joint_velocities = joint_velocities
        self._latest_joint_efforts = joint_efforts
        return GrpcCallResult(
            success=bool(joint_positions),
            available=True,
            payload=joint_positions,
        )

    def get_ee_pose(self, ee_link_name: str = "") -> GrpcCallResult:
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
        gripper_command = "open"
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
            logger.info("[GRIPPER] set_gripper_state(open) succeeded")
            return GrpcCallResult(success=True, available=True)
        except Exception as exc:
            logger.warning(f"[GRIPPER] set_gripper_state failed: {exc}")
            return GrpcCallResult(success=False, available=True, error=str(exc))

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
        pose = self._get_object_pose_raw(object_id)
        if pose is None:
            return GrpcCallResult(
                success=False,
                available=True,
                error=f"Object pose not available for {object_id}",
            )
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
            return self._stub.init_robot(request, timeout=self.timeout)

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
            return self._stub.reset(request, timeout=self.timeout)

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
        obs = getattr(self, "_latest_observation", None)
        if obs is None or not obs.get("success"):
            obs_result = self.get_observation()
            if not obs_result.available or not obs_result.success:
                return None
            obs = obs_result.payload or {}
        if not obs.get("success"):
            return None

        camera_observation = obs.get("camera_observation") or {}
        images = camera_observation.get("images", [])
        if not images:
            logger.warning("No camera images available in observation.")
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
            "timestamp": image_info.get("timestamp"),
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

    def execute_trajectory(self, trajectory: List[Dict[str, Any]]) -> GrpcCallResult:
        """
        Execute a trajectory on the robot.

        Uses real gRPC set_joint_position calls with cuRobo-planned trajectory.

        Args:
            trajectory: List of waypoints with positions, velocities, timestamps

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
        for waypoint in trajectory:
            joint_positions = waypoint.get("joint_positions") or []
            if not joint_positions:
                continue
            joint_names = self._resolve_joint_names(len(joint_positions))
            commands = [
                joint_pb2.JointCommand(name=name, position=float(value))
                for name, value in zip(joint_names, joint_positions)
            ]
            request = joint_channel_pb2.SetJointReq(
                commands=commands,
                is_trajectory=False,
            )
            response = self._call_grpc(
                "set_joint_position(trajectory)",
                lambda: self._joint_stub.set_joint_position(request, timeout=self.timeout),
                None,
                success_checker=lambda resp: bool(resp.errmsg),
            )
            if response is None:
                return GrpcCallResult(
                    success=False,
                    available=True,
                    error="gRPC call failed",
                )
            timestamp = waypoint.get("timestamp")
            if timestamp is not None:
                if last_timestamp is not None:
                    delay = max(0.0, float(timestamp) - float(last_timestamp))
                    if delay:
                        time.sleep(delay)
                last_timestamp = float(timestamp)

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
        allow_mock_override = os.getenv("ALLOW_GENIESIM_MOCK", "0") == "1"
        production_mode = self.config.environment == "production"
        env = os.environ.copy()
        env[GENIESIM_RECORDINGS_DIR_ENV] = str(self.config.recording_dir)
        env[GENIESIM_LOG_DIR_ENV] = str(self.config.log_dir)

        # Ensure geniesim_adapter dir is on PYTHONPATH so aimdk proto stubs resolve
        adapter_dir = str(Path(__file__).parent)
        existing = env.get("PYTHONPATH", "")
        if adapter_dir not in existing.split(os.pathsep):
            env["PYTHONPATH"] = f"{adapter_dir}{os.pathsep}{existing}" if existing else adapter_dir

        if use_local_server:
            if production_mode:
                self.log(
                    "Mock Genie Sim gRPC server is disabled in production. "
                    "Install GENIESIM_ROOT or run a real server.",
                    "ERROR",
                )
                self._status = GenieSimServerStatus.ERROR
                return False
            if not allow_mock_override:
                self.log(
                    "GENIESIM_ROOT not found and mock server disabled. "
                    "Set ALLOW_GENIESIM_MOCK=1 for dev/test usage.",
                    "ERROR",
                )
                self._status = GenieSimServerStatus.ERROR
                return False
            self.log("GENIESIM_ROOT not found; using local gRPC server module", "INFO")
            cmd = [
                sys.executable,
                "-m",
                "tools.geniesim_adapter.geniesim_server",
                "--host",
                "0.0.0.0",
                "--port",
                str(self.config.port),
            ]
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
                self._server_process.wait()

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
            self.log(f"Initializing robot: cfg_file={robot_cfg_file}, base_position={base_pos}")
            scene_usd = os.environ.get("GENIESIM_SCENE_USD_PATH", "scenes/empty_scene.usda")
            init_result = self._client.init_robot(
                robot_type=robot_cfg_file,
                base_pose=base_pose,
                scene_usd_path=scene_usd,
            )
            if not init_result.success:
                self.log(
                    f"init_robot returned: success={init_result.success}, "
                    f"error={init_result.error}, available={init_result.available}",
                    "WARNING",
                )
                # Non-fatal: server may already have robot loaded via --scene arg
            else:
                self.log(f"Robot initialized: {init_result.payload}")

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

            # Map logical camera names to the G1 robot's USD prim paths.
            _G1_CAMERA_MAP = {
                "wrist": "/G1/gripper_r_base_link/Right_Camera",
                "overhead": "/G1/head_link2/Head_Camera",
                "side": "/G1/gripper_l_base_link/Left_Camera",
            }
            camera_map_env = os.environ.get("GENIESIM_CAMERA_PRIM_MAP", "")
            if camera_map_env:
                import json as _json
                self._client._camera_prim_map = _json.loads(camera_map_env)
            else:
                self._client._camera_prim_map = _G1_CAMERA_MAP
            self.log(f"Camera prim map: {self._client._camera_prim_map}")

            # Populate scene object prim paths for real object pose queries.
            # Derive USD prim paths from scene_graph nodes or task_config objects.
            _scene_obj_prims: List[str] = []
            _sg_nodes = (scene_config or {}).get("nodes", [])
            if not _sg_nodes:
                _sg_nodes = task_config.get("nodes", [])
            for _node in _sg_nodes:
                _asset_id = _node.get("asset_id", "")
                _usd_path = _node.get("usd_path", "")
                # Derive USD stage prim path from asset file name
                # e.g. ".../obj_Pot057/Pot057.usd" → "/World/Pot057"
                if _usd_path:
                    _stem = Path(_usd_path).stem  # "Pot057"
                    _prim = f"/World/{_stem}"
                elif _asset_id:
                    # Strip scene prefix: "lightwheel_kitchen_obj_Pot057" → "Pot057"
                    _parts = _asset_id.split("_obj_")
                    _prim = f"/World/{_parts[-1]}" if len(_parts) > 1 else f"/World/{_asset_id}"
                else:
                    continue
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
            if _scene_obj_prims:
                self.log(f"Scene object prims for real pose queries: {_scene_obj_prims}")
            else:
                self.log("No scene object prims found — object poses will be synthetic", "WARNING")

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
                            _resp = _gemini_client_for_props.generate(prompt=_prompt, json_output=True)
                            _text = _resp.text.strip()
                            _start = _text.find("{")
                            _end = _text.rfind("}") + 1
                            if _start >= 0 and _end > _start:
                                import json as _json_mod
                                _est = _json_mod.loads(_text[_start:_end])
                                _size = [
                                    max(float(_est.get("width", 0.1)), 0.01),
                                    max(float(_est.get("depth", 0.1)), 0.01),
                                    max(float(_est.get("height", 0.1)), 0.01),
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

            # Create output directory for this run
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self.config.recording_dir / f"run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)

            for task_idx, task in enumerate(tasks):
                if _timeout_exceeded():
                    break
                task_name = task.get("task_name", f"task_{task_idx}")
                if "task_name" not in task or not task.get("task_name"):
                    task["task_name"] = task_name
                self.log(f"\nTask {task_idx + 1}/{len(tasks)}: {task_name}")

                # Configure environment for task
                self._configure_task(task, scene_config)

                for ep_idx in range(episodes_target):
                    if _timeout_exceeded():
                        break
                    if progress_callback:
                        current = task_idx * episodes_target + ep_idx + 1
                        total = len(tasks) * episodes_target
                        progress_callback(current, total, f"Task: {task_name}, Episode: {ep_idx + 1}")

                    try:
                        # Reset environment
                        reset_result = self._client.reset_environment()
                        if not reset_result.available:
                            error_message = (
                                f"Reset unavailable before episode: {reset_result.error}"
                            )
                            result.errors.append(error_message)
                            self.log(error_message, "ERROR")
                            return result
                        if not reset_result.success:
                            error_message = reset_result.error or "Reset failed before episode."
                            result.errors.append(error_message)
                            self.log(error_message, "ERROR")
                            return result

                        # Generate and execute trajectory
                        episode_result = self._run_single_episode(
                            task=task,
                            episode_id=f"{task_name}_ep{ep_idx:04d}",
                            output_dir=run_dir,
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
                            total_frames += episode_result.get("frame_count", 0)
                            quality_scores.append(episode_result.get("quality_score", 0.0))
                        else:
                            stall_info = episode_result.get("stall_info") or {}
                            if stall_info.get("stall_detected"):
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

                                stall_message = (
                                    f"Episode {ep_idx} of {task_name} stalled after "
                                    f"{stall_info.get('last_progress_age_s', 0.0):.1f}s "
                                    f"(stall {self._stall_count}/{self.config.max_stalls}, "
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
                                    f"last_obs={stall_info.get('last_observation_timestamp')}",
                                    "WARNING",
                                )

                                if self._stall_count > self.config.max_stalls:
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
                                    if self.config.stall_backoff_s > 0:
                                        time.sleep(self.config.stall_backoff_s)
                            result.warnings.append(
                                f"Episode {ep_idx} of {task_name} failed: {episode_result.get('error', 'unknown')}"
                            )

                    except Exception as e:
                        result.warnings.append(f"Episode {ep_idx} of {task_name} error: {e}")
                        self.log(f"  Episode {ep_idx} error: {e}", "WARNING")

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

        Returns:
            Episode result with success status and metrics
        """
        result = {
            "episode_id": episode_id,
            "success": False,
            "frame_count": 0,
            "quality_score": 0.0,
            "collision_free": None,
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

            # Get initial observation
            obs_result = self._client.get_observation()
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
                "completion_signal_logged": False,
            }
            start_event = threading.Event()
            timestamps = [waypoint["timestamp"] for waypoint in timed_trajectory]
            abort_event = threading.Event()

            def _note_progress(obs_frame: Dict[str, Any]) -> None:
                collector_state["last_progress_time"] = time.time()
                collector_state["last_observation_timestamp"] = (
                    obs_frame.get("timestamp")
                    or obs_frame.get("planned_timestamp")
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
                    for response in stream_result.payload:
                        if abort_event.is_set():
                            break
                        if not response.get("success", False):
                            collector_state["error"] = (
                                response.get("error")
                                or "Observation stream returned unsuccessful response."
                            )
                            break
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
                    start_event.wait()
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
                        obs_result = self._client.get_observation()
                        if not obs_result.available:
                            collector_state["error"] = (
                                f"Timed observation polling unavailable: {obs_result.error}"
                            )
                            break
                        if not obs_result.success:
                            collector_state["error"] = (
                                obs_result.error
                                or "Timed observation polling returned unsuccessful response."
                            )
                            break
                        obs_frame = obs_result.payload or {}
                        obs_frame["planned_timestamp"] = planned_timestamp
                        collector_state["observations"].append(obs_frame)
                        _note_progress(obs_frame)
                except Exception as exc:
                    collector_state["error"] = (
                        f"Timed observation polling failed after "
                        f"{len(collector_state['observations'])} frames: {exc}"
                    )

            collector_thread = threading.Thread(target=_collect_streaming, daemon=True)
            collector_thread.start()

            collector_state["start_time"] = time.time()
            start_event.set()

            execution_state: Dict[str, Any] = {"success": False, "error": None}

            def _execute_trajectory() -> None:
                try:
                    execution_result = self._client.execute_trajectory(timed_trajectory)
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

            stall_timeout_s = self.config.stall_timeout_s
            stall_detected = False
            stall_reason: Optional[str] = None
            trajectory_duration = (timestamps[-1] - timestamps[0]) if timestamps else 0.0
            trajectory_end_time = timestamps[-1] if timestamps else None
            end_time_tolerance_s = max(0.25, trajectory_duration * 0.05)

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

            def _is_near_trajectory_end(last_timestamp: Optional[float]) -> bool:
                if last_timestamp is None or trajectory_end_time is None:
                    return False
                return last_timestamp >= trajectory_end_time - end_time_tolerance_s

            while execution_thread.is_alive():
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
                            near_trajectory_end = _is_near_trajectory_end(last_obs_timestamp)
                            if task_success_flag or near_trajectory_end:
                                if not collector_state.get("completion_signal_logged"):
                                    self.log(
                                        "Stall check skipped due to completion signal "
                                        f"(task_success={task_success_flag}, "
                                        f"near_trajectory_end={near_trajectory_end}, "
                                        f"last_obs_timestamp={last_obs_timestamp}, "
                                        f"trajectory_end={trajectory_end_time}, "
                                        f"tolerance={end_time_tolerance_s:.2f}s).",
                                        "INFO",
                                    )
                                    collector_state["completion_signal_logged"] = True
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
            collector_timeout = 5.0 if stall_detected else trajectory_duration + 10.0
            collector_thread.join(timeout=collector_timeout)

            if collector_thread.is_alive():
                last_obs_timestamp = collector_state.get("last_observation_timestamp")
                if execution_state.get("success") and _is_near_trajectory_end(last_obs_timestamp):
                    self.log(
                        "Observation collection still running, but execution completed and "
                        "final observation is near trajectory end; treating episode as complete "
                        f"(trajectory_end={trajectory_end_time}, "
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

            if execution_state.get("error"):
                result["error"] = execution_state["error"]
                return result

            if stall_detected:
                result["error"] = collector_state["error"] or "Episode stalled"
                return result

            if not execution_state.get("success"):
                result["error"] = "Trajectory execution failed"
                return result

            if collector_state["error"]:
                result["error"] = collector_state["error"]
                return result

            aligned_observations = self._align_observations_to_trajectory(
                timed_trajectory,
                collector_state["observations"],
            )
            if aligned_observations is None:
                result["error"] = "Failed to align observations with trajectory."
                return result

            # Minimum frame count guard
            min_episode_frames = int(os.getenv("MIN_EPISODE_FRAMES", "20"))
            if len(aligned_observations) < min_episode_frames:
                result["error"] = (
                    f"Too few observations ({len(aligned_observations)}) for episode; "
                    f"minimum is {min_episode_frames}."
                )
                return result

            frames = self._build_frames_from_trajectory(
                timed_trajectory,
                aligned_observations,
                task=task,
                episode_id=episode_id,
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
                )
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

            # Stop recording
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

            # Calculate quality score
            quality_score = self._calculate_quality_score(frames, task)
            min_quality = float(os.getenv("MIN_QUALITY_SCORE", "0.7"))
            validation_passed = quality_score >= min_quality
            task_success = self._extract_task_success(frames, task)
            collision_free = planning_report.get("collision_free")
            collision_source = planning_report.get("collision_source")

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
                    _ts_resp = _gemini_ts_client.generate(prompt=_ts_prompt, json_output=True)
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

            # Geometric task success: goal-region verification (fallback validation)
            if task_success is None and frames:
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
                    _pl_resp = _gemini_pl_client.generate(prompt=_pl_prompt, json_output=True)
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
            num_joints = len(frames[0]["action"]) if frames else 0
            arm_dof = min(7, num_joints)
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
            robot_metadata = {
                "robot_type": robot_type,
                "num_joints": num_joints,
                "arm_dof": arm_dof,
                "gripper_dof": max(0, num_joints - arm_dof),
                "joint_names": joint_names if joint_names else [f"joint_{i}" for i in range(num_joints)],
                "joint_limits_lower": rc.joint_limits_lower.tolist() if rc is not None and hasattr(rc, "joint_limits_lower") else _meta_fb.get("joint_limits_lower"),
                "joint_limits_upper": rc.joint_limits_upper.tolist() if rc is not None and hasattr(rc, "joint_limits_upper") else _meta_fb.get("joint_limits_upper"),
                "gripper_joint_names": list(rc.gripper_joint_names) if rc is not None and hasattr(rc, "gripper_joint_names") else _meta_fb.get("gripper_joint_names", []),
                "gripper_limits": list(rc.gripper_limits) if rc is not None and hasattr(rc, "gripper_limits") else list(_meta_fb["gripper_limits"]) if _meta_fb.get("gripper_limits") else None,
                "default_joint_positions": rc.default_joint_positions.tolist() if rc is not None and hasattr(rc, "default_joint_positions") else _meta_fb.get("default_joint_positions"),
                "action_space": "joint_delta",
                "action_abs_space": "joint_position",
                "control_frequency_hz": 30.0,
                "clock_model": "uniform_30hz",
                "transition_convention": "obs_t_action_t_produces_obs_t+1",
                "action_semantics": "joint_delta_from_current_to_next",
                "action_abs_semantics": "target_joint_position_for_next_step",
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

            # Clean up internal metadata fields from frames before serialization
            for _f in frames:
                _f.pop("_initial_object_poses", None)
                _f.pop("_final_object_poses", None)

            with open(episode_path, "w") as f:
                json.dump({
                    "episode_id": episode_id,
                    "task_name": task.get("task_name") or llm_metadata.get("task_name") or "unknown_task",
                    "task_type": task.get("task_type"),
                    "target_object": task.get("target_object"),
                    "task_description": llm_metadata.get("task_description"),
                    "data_mode": data_mode,
                    "robot_metadata": robot_metadata,
                    "episode_seed": episode_seed,
                    "provenance": {
                        "joint_positions": "physx_server",
                        "ee_pos": "isaac_sim_fk" if _server_ee_frame_count > 0 else "analytic_fk",
                        "ee_quat": "isaac_sim_fk" if _server_ee_frame_count > 0 else "analytic_fk",
                        "ee_rot6d": "derived_from_ee_quat",
                        "joint_velocities": "physx_server" if _real_velocity_count > 0 else "finite_difference",
                        "joint_accelerations": "finite_difference_smoothed",
                        "joint_efforts": "physx_server" if _real_effort_count > 0 else "unavailable",
                        "ee_vel": "derived_from_physx_positions" if _server_ee_frame_count > 0 else "derived_from_fk_positions",
                        "ee_acc": "derived_from_physx_positions" if _server_ee_frame_count > 0 else "derived_from_fk_positions",
                        "contact_forces": (
                            "physx_joint_effort" if _real_effort_count > 0
                            else ("gemini_estimated" if _contact_force_cache else "heuristic_grasp_model_v1")
                        ),
                        "camera_frames": "isaac_sim_camera" if _camera_frame_count > 0 else "unavailable",
                        "task_description": "task_config_hint",
                        "scene_state": "physx_server" if _real_scene_state_count > 0 else "synthetic_from_task_config",
                        "task_success": llm_metadata.get("task_success_source", "geometric_goal_region_v2"),
                        "quality_score": "weighted_composite_v2",
                        "server_ee_frames": f"{_server_ee_frame_count}/{len(frames)}",
                        "real_scene_state_frames": f"{_real_scene_state_count}/{len(frames)}",
                        "camera_capture_frames": f"{_camera_frame_count}/{len(frames)}",
                        "real_velocity_frames": f"{_real_velocity_count}/{len(frames)}",
                        "real_effort_frames": f"{_real_effort_count}/{len(frames)}",
                        "diversity_calibration": _diversity_calibration_source,
                        "object_property_provenance": dict(_object_property_provenance),
                    },
                    "channel_confidence": {
                        "joint_positions": 1.0,
                        "ee_pos": 0.98 if _server_ee_frame_count > len(frames) * 0.5 else 0.7,
                        "ee_quat": 0.98 if _server_ee_frame_count > len(frames) * 0.5 else 0.7,
                        "joint_velocities": 0.98 if _real_velocity_count > 0 else 0.7,
                        "joint_accelerations": 0.6,
                        "joint_efforts": 0.95 if _real_effort_count > 0 else 0.0,
                        "contact_forces": (
                            0.9 if _real_effort_count > 0
                            else (0.6 if _contact_force_cache else 0.2)
                        ),
                        "ee_vel": 0.85 if _server_ee_frame_count > len(frames) * 0.5 else 0.6,
                        "ee_acc": 0.85 if _server_ee_frame_count > len(frames) * 0.5 else 0.6,
                        "scene_state": 0.95 if _real_scene_state_count > 0 else 0.3,
                        "camera_frames": 0.95 if _camera_frame_count > 0 else 0.0,
                    },
                    "goal_region_verification": llm_metadata.get("goal_region_verification"),
                    "frames": frames,
                    "frame_count": len(frames),
                    "quality_score": quality_score,
                    "validation_passed": validation_passed,
                    "task_success": task_success,
                    "task_success_reasoning": llm_metadata.get("task_success_reasoning"),
                    "collision_free": collision_free,
                    "collision_source": collision_source,
                    "scene_description": llm_metadata.get("scene_description"),
                    "success_criteria": llm_metadata.get("success_criteria"),
                    "stall_info": result["stall_info"],
                    "frame_validation": frame_validation,
                    "phase_descriptions": {
                        "approach": f"Moving toward {task.get('target_object', 'target object')}",
                        "grasp": f"Closing gripper to grasp {task.get('target_object', 'object')}",
                        "lift": f"Lifting {task.get('target_object', 'object')} off the surface",
                        "transport": f"Transporting {task.get('target_object', 'object')} to placement location",
                        "place": f"Placing {task.get('target_object', 'object')} at target",
                        "retract": "Retracting gripper after placement",
                    },
                }, f, default=_json_default)

            result["success"] = True
            result["frame_count"] = len(frames)
            result["quality_score"] = quality_score
            result["validation_passed"] = validation_passed
            result["task_success"] = task_success
            result["collision_free"] = collision_free
            result["output_path"] = str(episode_path)
            result["frame_validation"] = frame_validation

        except Exception as e:
            import traceback
            result["error"] = str(e)
            self.log(f"Episode failed: {e}", "ERROR")
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

    def _align_observations_to_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        if not observations:
            return None

        def _obs_timestamp(obs: Dict[str, Any], index: int) -> float:
            timestamp = obs.get("timestamp")
            if timestamp is None:
                timestamp = obs.get("planned_timestamp")
            if timestamp is None:
                return float(index)
            try:
                return float(timestamp)
            except (TypeError, ValueError):
                return float(index)

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
    ) -> List[Dict[str, Any]]:
        # Attempt to set up FK for end-effector pose computation
        fk_solver = None
        robot_config = ROBOT_CONFIGS.get(
            getattr(self.config, "robot_type", ""), None
        )
        if robot_config is not None and IK_PLANNING_AVAILABLE:
            try:
                _solver = IKSolver(robot_config, verbose=False)
                if hasattr(_solver, "_forward_kinematics"):
                    fk_solver = _solver
            except Exception:
                pass

        # Determine gripper joint indices (joints beyond the arm DOFs)
        arm_dof = min(7, len(trajectory[0]["joint_positions"])) if trajectory else 7

        # Forward-propagate joint positions: when observations return zeros
        # (mock/skip-server-recording mode), use the trajectory waypoints as
        # ground-truth robot state. This ensures action deltas are correct
        # (small values, not equal to action_abs).
        current_joint_state = np.array(
            trajectory[0]["joint_positions"], dtype=float
        ) if trajectory else np.zeros(7)

        # --- Object tracking state for dynamic scene updates (Improvement A) ---
        _attached_object_id: Optional[str] = None
        _grasp_ee_offset: Optional[np.ndarray] = None
        _object_poses: Dict[str, np.ndarray] = {}  # current poses of all objects
        _initial_object_poses: Dict[str, np.ndarray] = {}  # for success verification
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
        _grasp_detected = False
        _release_detected = False
        _lift_phase_started = False

        # --- Grasp physics constants (Improvement J) ---
        _GRIPPER_MAX_APERTURE = 0.08  # meters (Franka gripper max opening)
        _GRIPPER_MAX_FORCE = 40.0     # Newtons

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

        # --- Gemini object property estimation (GAP 1) ---
        _gemini_prop_cache: Dict[str, Dict[str, Any]] = {}
        _gemini_prop_llm = None
        _gemini_prop_llm_attempted = False
        _object_property_provenance: Dict[str, str] = {}

        def _get_gemini_prop_client():
            nonlocal _gemini_prop_llm, _gemini_prop_llm_attempted
            if not _gemini_prop_llm_attempted:
                _gemini_prop_llm_attempted = True
                try:
                    from tools.llm_client import create_llm_client as _create
                    _gemini_prop_llm = _create()
                except Exception:
                    _gemini_prop_llm = None
            return _gemini_prop_llm

        def _estimate_obj_prop_gemini(obj_type: str, prop: str):
            """Estimate a single object property via Gemini."""
            cache_key = f"{obj_type.lower()}:{prop}"
            if cache_key in _gemini_prop_cache:
                return _gemini_prop_cache[cache_key]
            llm = _get_gemini_prop_client()
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
                resp = llm.generate(prompt, json_output=True, temperature=0.3)
                data = resp.parse_json()
                if isinstance(data, dict) and "value" in data:
                    val = data["value"]
                    _gemini_prop_cache[cache_key] = val
                    logger.info("GEMINI_OBJ_PROP: %s.%s = %s", obj_type, prop, val)
                    return val
            except Exception as exc:
                logger.debug("Gemini obj prop estimation failed for %s.%s: %s", obj_type, prop, exc)
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
            # 2. Try Gemini estimation (GAP 1 fix)
            _gemini_val = _estimate_obj_prop_gemini(obj_id_or_type, prop)
            if _gemini_val is not None:
                _object_property_provenance[f"{obj_id_or_type}:{prop}"] = "gemini_estimated"
                return _gemini_val
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
            return default

        frames: List[Dict[str, Any]] = []
        _server_ee_frame_count = 0  # Track how many frames used server EE pose
        _real_scene_state_count = 0  # Track how many frames had real scene state
        _camera_frame_count = 0  # Track how many frames had camera data
        _real_velocity_count = 0  # Track how many frames had real PhysX velocities
        _real_effort_count = 0  # Track how many frames had real PhysX joint efforts
        _contact_force_cache: Dict[str, Dict] = {}  # cache Gemini contact force estimates
        self._prev_server_ee_pos = None  # Reset for each episode
        for step_idx, (waypoint, obs) in enumerate(zip(trajectory, observations)):
            # Shallow-copy obs to avoid mutating shared references (aligned
            # observations may reuse the same dict for multiple frames).
            if obs is None:
                obs = {}
            else:
                obs = dict(obs)
                if "robot_state" in obs:
                    obs["robot_state"] = dict(obs["robot_state"])
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

            obs_joints = np.array(robot_state["joint_positions"], dtype=float)

            action_delta = (waypoint_joints - obs_joints).tolist()

            # Normalize timestamps to uniform 1/control_frequency_hz intervals
            _control_freq = 30.0  # Hz
            _uniform_ts = step_idx / _control_freq
            obs["timestamp"] = _uniform_ts
            obs["planned_timestamp"] = _uniform_ts
            # Preserve original trajectory timestamp for debugging
            obs["_original_timestamp"] = waypoint["timestamp"]

            # Inject synthetic scene_state when empty (mock/skip-server mode).
            # In production, empty scene_state is an error — real data is required.
            if not obs.get("scene_state") and step_idx == 0:
                _is_production = getattr(self.config, "environment", "") == "production"
                if _is_production:
                    logger.error(
                        "EMPTY_SCENE_STATE in production for episode %s frame %d. "
                        "Real scene object poses required. Check _scene_object_prims "
                        "initialization and SimObjectService connectivity.",
                        episode_id, step_idx,
                    )
            if not obs.get("scene_state"):
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
                "action": action_delta,
                "action_abs": waypoint["joint_positions"],
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
            _used_server_ee = False
            # Check if the server observation contains a real (dynamic) EE pose
            _server_ee = robot_state.get("ee_pose") or {}
            if isinstance(_server_ee, dict) and "position" in _server_ee:
                _sep = _server_ee["position"]
                if isinstance(_sep, dict) and "x" in _sep:
                    _ee_srv = [_sep["x"], _sep["y"], _sep["z"]]
                elif isinstance(_sep, (list, tuple)) and len(_sep) >= 3:
                    _ee_srv = list(_sep[:3])
                else:
                    _ee_srv = None
                if _ee_srv is not None:
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

            # Fallback: local FK
            if not _used_server_ee:
                try:
                    if fk_solver is not None:
                        ee_pos, ee_quat = fk_solver._forward_kinematics(waypoint_joints[:arm_dof])
                        _ee = (np.asarray(ee_pos, dtype=float) + _base_pos).tolist()
                        frame_data["ee_pos"] = _ee
                        frame_data["ee_quat"] = ee_quat.tolist() if hasattr(ee_quat, 'tolist') else ee_quat
                    else:
                        ee_pos, ee_quat = _franka_fk(waypoint_joints[:arm_dof])
                        _ee = (np.asarray(ee_pos, dtype=float) + _base_pos).tolist()
                        frame_data["ee_pos"] = _ee
                        frame_data["ee_quat"] = ee_quat
                except Exception:
                    pass

            if _used_server_ee:
                _server_ee_frame_count += 1
            if obs.get("scene_state", {}).get("objects") and obs.get("data_source") == "real_composed":
                _real_scene_state_count += 1
            if obs.get("camera_frames"):
                _camera_frame_count += 1
            _rs_vel = robot_state.get("joint_velocities", [])
            if _rs_vel and any(abs(v) > 1e-10 for v in _rs_vel):
                _real_velocity_count += 1
            _rs_eff = robot_state.get("joint_efforts", [])
            if _rs_eff and any(abs(e) > 1e-6 for e in _rs_eff):
                _real_effort_count += 1

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

            # Explicit gripper command: infer from gripper joint positions
            # Normalize by gripper range (e.g. Franka: [0.0, 0.04])
            gripper_openness = 1.0
            if len(waypoint_joints) > arm_dof:
                gripper_joints = waypoint_joints[arm_dof:]
                _g_fb = _ROBOT_METADATA_FALLBACK.get(
                    getattr(self.config, "robot_type", "franka"), {}
                )
                _g_lims = _g_fb.get("gripper_limits", (0.0, 1.0))
                _g_range = float(_g_lims[1] - _g_lims[0]) if _g_lims[1] > _g_lims[0] else 1.0
                gripper_mean = float(np.mean(np.abs(gripper_joints)))
                gripper_openness = min(1.0, gripper_mean / _g_range) if _g_range > 0 else 0.0
                frame_data["gripper_command"] = "open" if gripper_openness > 0.5 else "closed"
                frame_data["gripper_openness"] = round(gripper_openness, 3)

            # --- Improvement G: Per-frame phase labels ---
            _current_phase = "approach"
            if frame_data.get("gripper_command") == "closed" and not _grasp_detected:
                _grasp_detected = True
                _current_phase = "grasp"
            elif _grasp_detected and not _release_detected:
                if frame_data.get("gripper_command") == "open":
                    _release_detected = True
                    _current_phase = "place"
                elif frame_data.get("ee_pos") and step_idx > 0 and frames:
                    _prev_ee_z = frames[-1].get("ee_pos", [0, 0, 0])[2] if frames[-1].get("ee_pos") else 0
                    _curr_ee_z = frame_data.get("ee_pos", [0, 0, 0])[2]
                    # GAP 7: Derive lift threshold from target object height
                    _lift_thresh = 0.005  # default
                    if _target_oid_for_subgoal and _object_poses:
                        _tgt_h = _object_dims.get(_target_oid_for_subgoal, {}).get("height", 0.1) if hasattr(locals().get("_object_dims", None) or {}, "get") else 0.1
                        _lift_thresh = max(0.002, _tgt_h * 0.05)
                    if not _lift_phase_started and _curr_ee_z > _prev_ee_z + _lift_thresh:
                        _lift_phase_started = True
                        _current_phase = "lift"
                    elif _lift_phase_started:
                        _current_phase = "transport"
                    else:
                        _current_phase = "grasp"
            elif _release_detected:
                _current_phase = "retract"
            frame_data["phase"] = _current_phase

            # Fix 10: Track phase boundaries and compute progress
            if _current_phase != _prev_phase_for_progress:
                _phase_start_frame = step_idx
                _prev_phase_for_progress = _current_phase
            _phase_len = step_idx - _phase_start_frame + 1
            # Estimate phase duration as fraction of total trajectory
            _est_phase_frames = max(1, len(trajectory) // 6)  # 6 known phases: approach, grasp, lift, transport, place, retract
            frame_data["phase_progress"] = min(1.0, round(_phase_len / _est_phase_frames, 4))

            # Distance to subgoal: use target object pose for approach/transport, table for place
            if frame_data.get("ee_pos") and _target_oid_for_subgoal:
                _ee_arr_sg = np.array(frame_data["ee_pos"])
                _tgt_pos_sg = _object_poses.get(_target_oid_for_subgoal)
                if _tgt_pos_sg is not None:
                    frame_data["distance_to_subgoal"] = round(float(np.linalg.norm(_ee_arr_sg - _tgt_pos_sg)), 5)

            # --- Improvement A: Dynamic scene state + J: Grasp physics ---
            ee_pos_arr = np.array(frame_data["ee_pos"]) if frame_data.get("ee_pos") else None

            # Update object poses from real scene_state every frame (not just frame 0).
            # On frame 0, also store initial poses for displacement calculation.
            if obs.get("scene_state"):
                for _obj in obs["scene_state"].get("objects", []):
                    _oid = _obj.get("object_id", "")
                    _p = _obj.get("pose", {})
                    if "x" in _p:
                        _pos = np.array([_p["x"], _p["y"], _p["z"]], dtype=float)
                    elif "position" in _p:
                        _pos = np.array(_p["position"], dtype=float)
                    else:
                        continue  # skip objects with no valid pose
                    # Validate against kinematic tracking if available
                    if _oid in _object_poses and step_idx > 0:
                        _tracked = _object_poses[_oid]
                        _div = float(np.linalg.norm(_pos - _tracked))
                        if _div > 0.02:  # >2cm divergence
                            logger.debug(
                                "Object %s: real pose diverges from tracked by %.3fm at frame %d",
                                _oid, _div, step_idx,
                            )
                    _object_poses[_oid] = _pos.copy()
                    if step_idx == 0:
                        _initial_object_poses[_oid] = _pos.copy()

            # Detect grasp: gripper closes near an object
            if (frame_data.get("gripper_command") == "closed"
                    and _attached_object_id is None
                    and ee_pos_arr is not None):
                for _oid, _opos in _object_poses.items():
                    _dist = float(np.linalg.norm(ee_pos_arr - _opos))
                    # Compute proximity threshold from real object bbox diagonal
                    _obj_type = ""
                    for _obj in obs.get("scene_state", {}).get("objects", []):
                        if _obj.get("object_id") == _oid:
                            _obj_type = (_obj.get("object_type") or "").lower()
                            break
                    _obj_bbox = _get_obj_prop(_obj_type, "bbox", [0.10, 0.10, 0.10])
                    _bbox_diag = float(np.linalg.norm(_obj_bbox)) / 2.0 + 0.05  # half diagonal + 5cm reach
                    _grasp_threshold = max(0.08, min(_bbox_diag, 0.25))  # clamp [8cm, 25cm]
                    if _dist < _grasp_threshold:
                        _obj_width = _get_obj_prop(_obj_type, "graspable_width", 0.06)
                        if _obj_width <= _GRIPPER_MAX_APERTURE:
                            _attached_object_id = _oid
                            _grasp_ee_offset = _opos - ee_pos_arr
                            frame_data["grasp_feasible"] = True
                            frame_data["grasped_object_id"] = _oid
                        else:
                            frame_data["grasp_feasible"] = False
                        break

            # Update attached object pose: prefer real scene_state (already updated
            # above), fall back to kinematic EE-offset tracking if no real pose.
            _attached_has_real_pose = (
                _attached_object_id is not None
                and obs.get("scene_state")
                and any(
                    _obj.get("object_id") == _attached_object_id
                    for _obj in obs["scene_state"].get("objects", [])
                )
            )
            if _attached_object_id is not None and ee_pos_arr is not None and not _attached_has_real_pose:
                _new_pos = ee_pos_arr + _grasp_ee_offset
                _object_poses[_attached_object_id] = _new_pos.copy()
                # Update scene_state in observation
                if obs.get("scene_state"):
                    for _obj in obs["scene_state"].get("objects", []):
                        if _obj.get("object_id") == _attached_object_id:
                            _obj["pose"] = {
                                "x": round(float(_new_pos[0]), 6),
                                "y": round(float(_new_pos[1]), 6),
                                "z": round(float(_new_pos[2]), 6),
                            }
                            break

                # Contact force estimation: prefer real joint efforts from PhysX
                _real_efforts = robot_state.get("joint_efforts", [])
                _has_real_efforts = (
                    _real_efforts
                    and len(_real_efforts) > arm_dof
                    and any(abs(e) > 1e-6 for e in _real_efforts)
                )
                if _has_real_efforts:
                    # Real PhysX joint torques available
                    _gripper_efforts = _real_efforts[arm_dof:]  # gripper joint efforts
                    _grip_force_real = sum(abs(e) for e in _gripper_efforts)
                    _arm_efforts = _real_efforts[:arm_dof]
                    frame_data["contact_forces"] = {
                        "grip_force_N": round(_grip_force_real, 4),
                        "arm_torques_Nm": [round(e, 4) for e in _arm_efforts],
                        "gripper_efforts_N": [round(e, 4) for e in _gripper_efforts],
                        "grasped_object_id": _attached_object_id,
                        "force_sufficient": _grip_force_real > 0.5,
                    }
                else:
                    # Fallback: try Gemini estimation, then heuristic
                    _obj_type = ""
                    for _obj in obs.get("scene_state", {}).get("objects", []):
                        if _obj.get("object_id") == _attached_object_id:
                            _obj_type = (_obj.get("object_type") or "").lower()
                            break
                    _mass = _get_obj_prop(_obj_type, "mass", 0.3)
                    _weight = _mass * 9.81
                    _openness_bucket = round(gripper_openness, 1)
                    _cf_cache_key = f"{_obj_type}:{_openness_bucket}"
                    _gemini_force = _contact_force_cache.get(_cf_cache_key)

                    _gemini_cf_client = getattr(self, "_gemini_client_for_props", None)
                    if _gemini_force is None and _gemini_cf_client:
                        try:
                            _obj_width_cf = _get_obj_prop(_obj_type, "graspable_width", 0.06)
                            _cf_prompt = (
                                f"A Franka Panda parallel-jaw gripper (max aperture 80mm, max force 40N) "
                                f"is at {_openness_bucket*100:.0f}% openness grasping a {_obj_type} "
                                f"(mass={_mass:.2f}kg, width={_obj_width_cf:.3f}m). "
                                f"Estimate the grip force in Newtons and whether the grasp is stable. "
                                f'Respond with ONLY JSON: {{"grip_force_N": <float>, "stable": <bool>}}'
                            )
                            _cf_resp = _gemini_cf_client.generate(prompt=_cf_prompt, json_output=True)
                            _cf_text = _cf_resp.text.strip()
                            _cf_start = _cf_text.find("{")
                            _cf_end = _cf_text.rfind("}") + 1
                            if _cf_start >= 0 and _cf_end > _cf_start:
                                import json as _json_mod
                                _cf_data = _json_mod.loads(_cf_text[_cf_start:_cf_end])
                                _gemini_force = {
                                    "grip_force_N": round(float(_cf_data.get("grip_force_N", 0)), 2),
                                    "stable": bool(_cf_data.get("stable", False)),
                                }
                                _contact_force_cache[_cf_cache_key] = _gemini_force
                        except Exception as _cf_exc:
                            logger.debug("Gemini contact force estimation failed: %s", _cf_exc)

                    if _gemini_force is not None:
                        frame_data["contact_forces"] = {
                            "weight_force_N": round(_weight, 2),
                            "grip_force_N": _gemini_force["grip_force_N"],
                            "force_sufficient": _gemini_force["stable"],
                            "grasped_object_id": _attached_object_id,
                            "provenance": "gemini_estimated",
                            "confidence": 0.6,
                        }
                    else:
                        # Last-resort heuristic
                        _grip_force = (1.0 - gripper_openness) * _GRIPPER_MAX_FORCE
                        frame_data["contact_forces"] = {
                            "weight_force_N": round(_weight, 2),
                            "grip_force_N": round(_grip_force, 2),
                            "force_sufficient": _grip_force >= _weight,
                            "grasped_object_id": _attached_object_id,
                            "provenance": "heuristic_grasp_model_v1",
                            "confidence": 0.2,
                        }

            # Add provenance to contact forces (for PhysX real data path)
            if "contact_forces" in frame_data and "provenance" not in frame_data["contact_forces"]:
                _real_efforts = robot_state.get("joint_efforts", [])
                _has_real = _real_efforts and any(abs(e) > 1e-6 for e in _real_efforts)
                frame_data["contact_forces"]["provenance"] = "physx_joint_effort" if _has_real else "heuristic_grasp_model_v1"
                frame_data["contact_forces"]["confidence"] = 0.9 if _has_real else 0.2

            # Detect release: gripper opens while holding object
            if frame_data.get("gripper_command") == "open" and _attached_object_id is not None:
                _release_decay_obj = _attached_object_id
                _release_decay_remaining = 3  # decay over 3 frames
                _attached_object_id = None
                _grasp_ee_offset = None

            # Decay contact forces over 3 frames after release (Fix 9)
            if _release_decay_remaining > 0 and _attached_object_id is None and "contact_forces" not in frame_data:
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

            # Update ALL tracked object poses in scene_state (not just attached)
            if obs.get("scene_state") and _object_poses:
                for _obj in obs["scene_state"].get("objects", []):
                    _oid = _obj.get("object_id", "")
                    if _oid in _object_poses:
                        _p = _object_poses[_oid]
                        _obj["pose"] = {
                            "x": round(float(_p[0]), 6),
                            "y": round(float(_p[1]), 6),
                            "z": round(float(_p[2]), 6),
                        }

            # --- Improvement C: Mirror EE fields into observation.robot_state ---
            for _key in ("ee_pos", "ee_quat", "ee_vel", "gripper_command", "gripper_openness"):
                if _key in frame_data:
                    robot_state[_key] = frame_data[_key]

            # Fix 11: Commanded vs measured state
            if frame_data.get("action_abs"):
                robot_state["commanded_joint_positions"] = frame_data["action_abs"]
                _jp = robot_state.get("joint_positions", [])
                _cmd = frame_data["action_abs"]
                if _jp and len(_jp) == len(_cmd):
                    robot_state["tracking_error"] = [
                        round(c - m, 8) for c, m in zip(_cmd, _jp)
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
                if frame_data.get("ee_pos"):
                    frame_data["ee_pos"] = [
                        v + _noise_rng.gauss(0, _ee_noise_std) for v in frame_data["ee_pos"]
                    ]
                    robot_state["ee_pos"] = frame_data["ee_pos"]

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

        # Franka Panda velocity limits (rad/s) for 7 arm joints + 2 gripper
        _FRANKA_VEL_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 0.2, 0.2])
        _FRANKA_ACC_LIMITS = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0, 10.0, 10.0])

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
                    if _jv_clamped.size <= len(_FRANKA_VEL_LIMITS):
                        _vel_lim = _FRANKA_VEL_LIMITS[:_jv_clamped.size]
                    else:
                        _vel_lim = np.full(_jv_clamped.size, _FRANKA_VEL_LIMITS[-1])
                        _vel_lim[:len(_FRANKA_VEL_LIMITS)] = _FRANKA_VEL_LIMITS
                    _jv_clamped = np.clip(_jv_clamped, -_vel_lim, _vel_lim)
                    obs_rs["joint_velocities"] = _jv_clamped.tolist()
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
                        # Clamp to Franka velocity limits (Fix 4)
                        if _jv.size <= len(_FRANKA_VEL_LIMITS):
                            _vel_lim = _FRANKA_VEL_LIMITS[:_jv.size]
                        else:
                            # Pad with last limit value for extra joints
                            _vel_lim = np.full(_jv.size, _FRANKA_VEL_LIMITS[-1])
                            _vel_lim[:len(_FRANKA_VEL_LIMITS)] = _FRANKA_VEL_LIMITS
                        _jv_clamped = np.clip(_jv, -_vel_lim, _vel_lim)
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
                            if _ja.size <= len(_FRANKA_ACC_LIMITS):
                                _acc_lim = _FRANKA_ACC_LIMITS[:_ja.size]
                            else:
                                _acc_lim = np.full(_ja.size, _FRANKA_ACC_LIMITS[-1])
                                _acc_lim[:len(_FRANKA_ACC_LIMITS)] = _FRANKA_ACC_LIMITS
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

        # Store initial/final object poses on first/last frame for success verification
        if frames:
            frames[0]["_initial_object_poses"] = {
                k: v.tolist() for k, v in _initial_object_poses.items()
            }
            frames[-1]["_final_object_poses"] = {
                k: v.tolist() for k, v in _object_poses.items()
            }

        return frames

    def _validate_frames(
        self,
        frames: List[Dict[str, Any]],
        *,
        episode_id: str,
        task: Dict[str, Any],
    ) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        invalid_frames: set[int] = set()

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
                if width <= 0 or height <= 0:
                    frame_errors.append(
                        f"Frame {idx} camera '{camera_id}' has invalid dimensions ({width}x{height})."
                    )
                rgb = camera_data.get("rgb")
                depth = camera_data.get("depth")
                if rgb is None:
                    frame_errors.append(f"Frame {idx} camera '{camera_id}' missing rgb data.")
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

        return {
            "enabled": True,
            "errors": errors,
            "warnings": warnings,
            "invalid_frame_count": len(invalid_frames),
            "total_frames": len(frames),
        }

    def _attach_camera_frames(
        self,
        obs: Dict[str, Any],
        *,
        episode_id: str,
        task: Dict[str, Any],
    ) -> None:
        setattr(self._client, "_latest_observation", obs)
        for camera_id in ["wrist", "overhead", "side"]:
            try:
                camera_data = self._client.get_camera_data(camera_id)
                if camera_data is not None:
                    obs.setdefault("camera_frames", {})[camera_id] = camera_data
            except Exception:
                logger.warning(
                    "Camera capture failed (camera_id=%s, episode_id=%s, task_name=%s, task_id=%s).",
                    camera_id,
                    episode_id,
                    task.get("task_name"),
                    task.get("task_id"),
                    exc_info=True,
                )

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
        robot_config = ROBOT_CONFIGS.get(self.config.robot_type, ROBOT_CONFIGS.get("franka"))
        if robot_config is not None:
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
            return self._generate_linear_fallback_trajectory(
                task=task,
                initial_obs=initial_obs,
                obstacles=obstacles,
            )

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
        robot_config = ROBOT_CONFIGS.get(self.config.robot_type, ROBOT_CONFIGS.get("franka"))
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
            MotionPhase.APPROACH,
            duration=1.0,
        )

        grasp_position = target_position.copy()
        grasp_position[2] += 0.02
        _add_waypoint(
            grasp_position,
            target_orientation,
            MotionPhase.GRASP,
            duration=0.6,
        )

        lift_position = target_position.copy()
        lift_position[2] += 0.25
        _add_waypoint(
            lift_position,
            target_orientation,
            MotionPhase.LIFT,
            duration=0.8,
        )

        if place_position is not None:
            pre_place_position = place_position.copy()
            pre_place_position[2] += 0.20
            _add_waypoint(
                pre_place_position,
                place_orientation,
                MotionPhase.TRANSPORT,
                duration=1.2,
            )

            _add_waypoint(
                place_position,
                place_orientation,
                MotionPhase.PLACE,
                duration=0.6,
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

        robot_config = ROBOT_CONFIGS.get(self.config.robot_type, ROBOT_CONFIGS.get("franka"))
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

        waypoints = self._build_ik_fallback_waypoints(
            target_position=target_position,
            place_position=place_position,
            target_orientation=target_orientation,
            place_orientation=place_orientation,
        )

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

        for wp in waypoints:
            if planner is not None:
                joints = planner.solve_ik_with_collision_check(
                    wp.position, wp.orientation, seed_joints=seed_joints
                )
            else:
                joints = ik_solver.solve(wp.position, wp.orientation, seed_joints)

            if joints is None:
                self.log(
                    f"  ❌ IK failed for waypoint {wp.phase.value} at position {wp.position.tolist()}",
                    "ERROR",
                )
                return None
            if not self._within_joint_limits(joints, robot_config):
                self.log(
                    f"  ❌ IK solution violates joint limits for waypoint {wp.phase.value}",
                    "ERROR",
                )
                return None

            wp.joint_positions = joints
            seed_joints = joints

        trajectory: List[Dict[str, Any]] = []
        current_joints = initial_joints.copy()
        current_time = 0.0

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
                trajectory.append(
                    {
                        "joint_positions": joint_pos.tolist(),
                        "timestamp": current_time + t * duration,
                    }
                )

            current_time += duration
            current_joints = target_joints

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

        robot_config = ROBOT_CONFIGS.get(self.config.robot_type, ROBOT_CONFIGS.get("franka"))
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
        if robot_config is not None:
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
                    if planner is not None:
                        collision_free = True
                    self._last_planning_report.update(
                        {
                            "planner": "ik_fallback",
                            "collision_free": collision_free,
                            "collision_source": "ik_planner" if planner is not None else None,
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
        # Identify arm joints (first ~6-7) vs gripper/finger joints (remaining)
        num_joints = len(initial_joints)
        arm_end = min(7, num_joints)  # first 7 joints are typically arm
        # Gripper open/closed state masks: 1.0 = open, 0.0 = closed
        gripper_open = np.ones(num_joints)
        gripper_closed = np.ones(num_joints)
        for j in range(arm_end, num_joints):
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
            _prev_q = initial_joints[:arm_end]
            _all_solved = True
            for _pname, _cart in _cart_targets:
                _ik_result = _franka_numerical_ik(_cart, initial_guess=_prev_q)
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
                target[:arm_end] = ik_phase_joints[_pi]
                # Apply gripper state
                for j in range(arm_end, num_joints):
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
                    target[j] += arm_offsets[j] * joint_range[j]
                for j in range(arm_end, num_joints):
                    mid = (lower[j] + upper[j]) / 2.0
                    target[j] = mid + (initial_joints[j] - mid) * grip_mult[j]
                target = np.clip(target, lower, upper)
                phase_targets.append(target)

        # Gripper joint values per phase (Franka finger joints: 0.04 = open, 0.0 = closed)
        _grip_lims = _ROBOT_METADATA_FALLBACK.get(self.config.robot_type, {}).get("gripper_limits", (0.0, 0.04))
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
                if len(_jp_list) >= 9:
                    _jp_list[7] = grip_val
                    _jp_list[8] = grip_val
                    full_joints = _jp_list
                else:
                    full_joints = _jp_list + [grip_val, grip_val]
                trajectory.append(
                    {
                        "joint_positions": full_joints,
                        "timestamp": current_time + t * phase_duration,
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
        else:
            self.log(
                "  ℹ️  Reachability unchecked (no FK or workspace bounds).",
                "INFO",
            )

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
        self._last_planning_report.update(
            {
                "planner": "linear_fallback",
                "collision_free": collision_free,
                "collision_source": "joint_space_clearance" if collision_free is not None else None,
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
            f' "task_success_reasoning": "1 sentence explanation of success/failure assessment"}}\n'
        )

        try:
            client = create_llm_client(provider=LLMProvider.GEMINI)
            response = client.generate(
                prompt=prompt,
                json_output=True,
                temperature=0.3,
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

            self.log(
                f"  ✅ LLM enrichment: task_name={result.get('task_name')}, "
                f"task_success={result.get('task_success')}",
                "INFO",
            )
        except Exception as exc:
            self.log(f"  ⚠️  LLM enrichment error: {exc}", "WARNING")

        return result

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
            # Use relaxed schema when camera data is unavailable
            if os.environ.get("GENIESIM_SKIP_SERVER_RECORDING", ""):
                required_keys = PROPRIOCEPTION_ONLY_OBSERVATION_SCHEMA["required_keys"]
            else:
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

        frame_count_score = 1.0 if total_frames >= 10 else 0.5

        # GAP 2: Diversity divisors — calibrate via Gemini per robot type
        _diversity_calibration_source = "hardcoded_default"
        _action_divisor = 0.05
        _obs_divisor = 0.05
        _robot_type = getattr(self.config, "robot_type", "franka") if hasattr(self, "config") else "franka"
        try:
            from tools.llm_client import create_llm_client as _create_div_llm
            _div_llm = _create_div_llm()
            if _div_llm:
                _div_prompt = (
                    f"For a {_robot_type} robot arm, estimate the typical per-step joint velocity "
                    f"magnitude (in radians) that represents high action diversity during a "
                    f"pick-and-place task. Return ONLY JSON: "
                    f"{{\"action_divisor\": 0.05, \"obs_divisor\": 0.05}}"
                )
                _div_resp = _div_llm.generate(_div_prompt, json_output=True, temperature=0.3)
                _div_data = _div_resp.parse_json()
                if isinstance(_div_data, dict):
                    if "action_divisor" in _div_data and "obs_divisor" in _div_data:
                        _action_divisor = float(_div_data["action_divisor"])
                        _obs_divisor = float(_div_data["obs_divisor"])
                        _diversity_calibration_source = "gemini_calibrated"
                        logger.info(
                            "GEMINI_DIVERSITY: robot=%s action_div=%.4f obs_div=%.4f",
                            _robot_type, _action_divisor, _obs_divisor,
                        )
        except Exception:
            pass  # Keep defaults

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

        # Penalty: static scene state (no object moved)
        # Skip this penalty when running without server recording — physics
        # feedback is unavailable so object poses are always identical.
        _skip_recording = os.environ.get("GENIESIM_SKIP_SERVER_RECORDING", "")
        if frames and not _skip_recording:
            _init_poses = frames[0].get("_initial_object_poses", {})
            _final_poses = frames[-1].get("_final_object_poses", {})
            if _init_poses and _final_poses:
                _any_moved = False
                for _oid, _ip in _init_poses.items():
                    _fp = _final_poses.get(_oid)
                    if _fp is not None:
                        if np.linalg.norm(np.array(_fp) - np.array(_ip)) > 0.01:
                            _any_moved = True
                            break
                if not _any_moved:
                    scene_state_penalty = 0.20
                    logger.warning("Quality penalty: scene_state is static (no object moved >1cm)")

        # Penalty: EE never approaches target object
        if target_object_id and frames:
            _min_dist = float("inf")
            _initial_dist = float("inf")
            # Normalize target ID for fuzzy matching (strip path prefixes)
            _target_norm = target_object_id.rsplit("/", 1)[-1].lower()
            for _fi, _frame in enumerate(frames):
                _eep = _frame.get("ee_pos")
                _obs = _frame.get("observation", {})
                _ss = _obs.get("scene_state", {}) or _obs.get("privileged", {}).get("scene_state", {})
                if _eep:
                    _eep_arr = np.array(_eep)
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
            if _min_dist > 0.20:
                ee_approach_penalty = 0.15
                logger.warning(
                    "Quality penalty: EE never approaches target (min_dist=%.3f m)",
                    _min_dist,
                )
            elif _min_dist >= _initial_dist and _initial_dist < float("inf"):
                ee_approach_penalty = 0.10
                logger.warning(
                    "Quality penalty: EE distance to target never decreased (min=%.3f, initial=%.3f)",
                    _min_dist, _initial_dist,
                )

        # Base score uses original weights for backward compatibility;
        # smoothness and collision are additive bonuses (up to +0.05 each).
        weighted_score = (
            0.30 * data_completeness_score
            + 0.25 * action_validity_score
            + 0.15 * action_diversity_score
            + 0.10 * obs_diversity_score
            + 0.15 * success_score
            + 0.05 * frame_count_score
        )
        # Bonus for smooth trajectories and collision-free episodes
        smoothness_bonus = 0.05 * smoothness_score
        collision_bonus = 0.05 * (collision_free_score - 0.5)  # neutral at 0.5
        weighted_score += smoothness_bonus + collision_bonus
        # Physics plausibility penalties (Improvement I)
        weighted_score -= scene_state_penalty
        weighted_score -= ee_approach_penalty

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

        # Gate: joint velocities within Franka limits
        _vel_exceeded = 0
        _vel_lim = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
        for _fr in frames:
            _jv = (_fr.get("observation", {}).get("robot_state", {})
                   .get("joint_velocities", []))
            if _jv and len(_jv) >= 7:
                if np.any(np.abs(_jv[:7]) > _vel_lim * 1.05):
                    _vel_exceeded += 1
        if _vel_exceeded > 0:
            _gate_penalty += 0.05
            logger.warning("Quality gate: %d frames exceed Franka velocity limits", _vel_exceeded)

        # Gate: grasp implies grip, open implies no grip
        _grasp_consistency_fails = 0
        for _fr in frames:
            _gc = _fr.get("gripper_command")
            _priv = _fr.get("observation", {}).get("privileged", {})
            _cf = _priv.get("contact_forces", _fr.get("contact_forces", {}))
            if _gc == "closed" and isinstance(_cf, dict) and _cf.get("grip_force_N", 0) <= 0:
                _grasp_consistency_fails += 1
            if _gc == "open" and isinstance(_cf, dict) and _cf.get("grasped_object_id") is not None and not _cf.get("releasing"):
                _grasp_consistency_fails += 1
        if _grasp_consistency_fails > 0:
            _gate_penalty += 0.03
            logger.warning("Quality gate: %d frames with grasp/contact inconsistency", _grasp_consistency_fails)

        weighted_score -= _gate_penalty

        return min(1.0, max(0.0, weighted_score))

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
                    columns["observation"].append(
                        json.dumps(_to_json_serializable(frame.get("observation")))
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
    local_server_allowed = status["geniesim_installed"] or mock_allowed
    status["available"] = (
        status["isaac_sim_available"] and
        (status["grpc_available"] or status["server_running"]) and
        (status["server_running"] or local_server_allowed)
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

    if not status.get("isaac_sim_available", False):
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

    if not status.get("geniesim_installed", False) and not status.get("mock_server_allowed", False):
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
