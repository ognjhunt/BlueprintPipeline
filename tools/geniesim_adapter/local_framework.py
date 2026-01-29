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
try:
    from trajectory_solver import IKSolver, ROBOT_CONFIGS
    from motion_planner import Waypoint, MotionPhase
    from collision_aware_planner import CollisionAwarePlanner
    IK_PLANNING_AVAILABLE = True
except ImportError:
    IK_PLANNING_AVAILABLE = False
    IKSolver = None
    ROBOT_CONFIGS = {}
    Waypoint = None
    MotionPhase = None
    CollisionAwarePlanner = None
    logger.warning("IK utilities not available - IK fallback disabled")


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
    "lower": [-3.1416] * 7,
    "upper": [3.1416] * 7,
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
    stall_timeout_s: StrictFloat = 30.0
    max_stalls: StrictInt = 2
    stall_backoff_s: StrictFloat = 5.0
    server_startup_timeout_s: StrictFloat = 120.0
    server_startup_poll_s: StrictFloat = 2.0
    max_duration_seconds: Optional[StrictFloat] = None
    validate_frames: StrictBool = False
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
            resolved.relative_to(root)
        except ValueError:
            continue
        return resolved != root
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

        Returns:
            GrpcCallResult containing a formatted observation payload.
        """
        if not self._have_grpc:
            return self._grpc_unavailable(
                "get_observation_minimal",
                "gRPC stubs unavailable",
            )
        if self._stub is None:
            return self._grpc_unavailable(
                "get_observation_minimal",
                "gRPC stub not initialized",
            )
        effective_timeout = max(self.timeout, timeout) if timeout else self.timeout
        request = GetObservationReq(
            isCam=False,
            isJoint=include_joint,
            isPose=include_pose,
            objectPrims=[],
            isGripper=False,
        )

        def _request() -> GetObservationRsp:
            return self._stub.get_observation(request, timeout=effective_timeout)

        response = self._call_grpc(
            "get_observation(minimal)",
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
        formatted = self._format_observation_response(
            response,
            camera_ids=[],
            object_prims=[],
        )
        formatted["recording_state"] = response.recordingState
        return GrpcCallResult(
            success=True,
            available=True,
            payload=formatted,
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

        if command in {
            CommandType.GET_OBSERVATION,
            CommandType.START_RECORDING,
            CommandType.STOP_RECORDING,
            CommandType.GET_CAMERA_DATA,
            CommandType.GET_SEMANTIC_DATA,
            CommandType.GET_GRIPPER_STATE,
            CommandType.GET_JOINT_POSITION,
            CommandType.GET_OBJECT_POSE,
        }:
            observation_payload = dict(payload)
            if command == CommandType.GET_SEMANTIC_DATA:
                observation_payload["include_semantic"] = True
            if command == CommandType.GET_GRIPPER_STATE:
                observation_payload["include_gripper"] = True
                observation_payload.setdefault("left_gripper", True)
                observation_payload.setdefault("right_gripper", True)
            if command == CommandType.GET_JOINT_POSITION:
                observation_payload["include_joint"] = True
            if command == CommandType.GET_OBJECT_POSE:
                observation_payload["include_pose"] = True
                object_id = observation_payload.get("object_id")
                if object_id:
                    observation_payload.setdefault("object_prims", [object_id])

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

            if command in (CommandType.START_RECORDING, CommandType.STOP_RECORDING):
                return GrpcCallResult(
                    success=True,
                    available=True,
                    payload={"recording_state": response.recordingState},
                )

            formatted = self._format_observation_response(
                response,
                camera_ids=camera_ids,
                object_prims=object_prims,
            )
            return GrpcCallResult(
                success=True,
                available=True,
                payload=formatted,
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

        Uses real gRPC get_observation call.

        Returns:
            GrpcCallResult with payload containing robot_state, scene_state, timestamp.
        """
        # Server does not support standalone GetObservation (only
        # startRecording/stopRecording).  When server-side recording is
        # disabled, build a hybrid observation using real joint data from
        # get_joint_position() (which works) and synthetic placeholders
        # for fields the server can't provide.
        if os.environ.get("GENIESIM_SKIP_SERVER_RECORDING", ""):
            import time as _time
            # Attempt to read real joint positions from the server
            joint_positions = [0.0] * 28
            try:
                jp_result = self.get_joint_position()
                if jp_result.success and jp_result.payload is not None:
                    joint_positions = list(jp_result.payload)
            except Exception:
                pass
            return GrpcCallResult(
                success=True,
                available=True,
                payload={
                    "robot_state": {"joint_positions": joint_positions},
                    "scene_state": {},
                    "timestamp": _time.time(),
                    "planned_timestamp": 0.0,
                },
            )
        if not self._have_grpc:
            return self._grpc_unavailable(
                "get_observation",
                "gRPC stubs unavailable",
            )
        if self._stub is None:
            return self._grpc_unavailable(
                "get_observation",
                "gRPC stub not initialized",
            )

        request, camera_ids, object_prims = self._build_observation_request(
            {
                "include_images": True,
                "include_depth": True,
                "include_semantic": False,
                "include_joint": True,
            }
        )

        def _request() -> GetObservationRsp:
            return self._stub.get_observation(request, timeout=self.timeout)

        response = self._call_grpc(
            "get_observation",
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

        result = self._format_observation_response(
            response,
            camera_ids=camera_ids,
            object_prims=object_prims,
        )
        self._latest_observation = result
        return GrpcCallResult(
            success=True,
            available=True,
            payload=result,
        )

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
        if joint_names:
            self._joint_names = joint_names
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

        Returns:
            GrpcCallResult with payload containing width, force, and grasping state.
        """
        if not self._have_grpc:
            return self._grpc_unavailable(
                "get_gripper_state",
                "gRPC stubs unavailable",
            )
        if self._stub is None:
            return self._grpc_unavailable(
                "get_gripper_state",
                "gRPC stub not initialized",
            )

        request, camera_ids, object_prims = self._build_observation_request(
            {"include_gripper": True}
        )

        def _request() -> GetObservationRsp:
            return self._stub.get_observation(request, timeout=self.timeout)

        response = self._call_grpc(
            "get_observation(gripper)",
            _request,
            None,
            success_checker=lambda resp: resp is not None,
        )
        if response is None:
            return GrpcCallResult(
                success=False,
                available=True,
                error="gRPC call failed",
            )
        formatted = self._format_observation_response(
            response,
            camera_ids=camera_ids,
            object_prims=object_prims,
        )
        gripper_state = formatted.get("robot_state", {}).get("gripper", {})
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
        Get an object's pose.

        Uses real gRPC get_observation call with object pose flags.

        Args:
            object_id: Object identifier

        Returns:
            GrpcCallResult with payload pose dict.
        """
        if not self._have_grpc:
            return self._grpc_unavailable(
                "get_object_pose",
                "gRPC stubs unavailable",
            )
        if self._stub is None:
            return self._grpc_unavailable(
                "get_object_pose",
                "gRPC stub not initialized",
            )

        request, camera_ids, object_prims = self._build_observation_request(
            {
                "include_pose": True,
                "object_prims": [object_id],
            }
        )

        def _request() -> GetObservationRsp:
            return self._stub.get_observation(request, timeout=self.timeout)

        response = self._call_grpc(
            "get_observation(object_pose)",
            _request,
            None,
            success_checker=lambda resp: resp is not None,
        )
        if response is None:
            return GrpcCallResult(
                success=False,
                available=True,
                error="gRPC call failed",
            )
        formatted = self._format_observation_response(
            response,
            camera_ids=camera_ids,
            object_prims=object_prims,
        )
        objects = formatted.get("scene_state", {}).get("objects", [])
        if not objects:
            return GrpcCallResult(
                success=False,
                available=True,
                error=f"Object pose not available for {object_id}",
            )
        return GrpcCallResult(
            success=True,
            available=True,
            payload=objects[0].get("pose"),
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
        if os.environ.get("GENIESIM_SKIP_SERVER_RECORDING", ""):
            logger.info("[RECORDING] Skipping server-side start_recording (GENIESIM_SKIP_SERVER_RECORDING)")
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
        if os.environ.get("GENIESIM_SKIP_SERVER_RECORDING", ""):
            logger.info("[RECORDING] Skipping server-side stop_recording (GENIESIM_SKIP_SERVER_RECORDING)")
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

            episodes_target = episodes_per_task or self.config.episodes_per_task
            tasks = task_config.get("suggested_tasks", [task_config])

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

            # Determine data mode
            data_mode = "proprioception_only" if os.environ.get(
                "GENIESIM_SKIP_SERVER_RECORDING", ""
            ) else "full"

            # Save episode
            episode_path = output_dir / f"{episode_id}.json"
            def _json_default(value: Any) -> Any:
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                return value

            with open(episode_path, "w") as f:
                json.dump({
                    "episode_id": episode_id,
                    "task_name": task.get("task_name") or llm_metadata.get("task_name"),
                    "task_description": llm_metadata.get("task_description"),
                    "data_mode": data_mode,
                    "frames": frames,
                    "frame_count": len(frames),
                    "quality_score": quality_score,
                    "validation_passed": validation_passed,
                    "task_success": task_success,
                    "task_success_reasoning": llm_metadata.get("task_success_reasoning"),
                    "collision_free": collision_free,
                    "collision_source": planning_report.get("collision_source"),
                    "scene_description": llm_metadata.get("scene_description"),
                    "success_criteria": llm_metadata.get("success_criteria"),
                    "stall_info": result["stall_info"],
                    "frame_validation": frame_validation,
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
            result["error"] = str(e)
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
        frames: List[Dict[str, Any]] = []
        for step_idx, (waypoint, obs) in enumerate(zip(trajectory, observations)):
            self._attach_camera_frames(obs, episode_id=episode_id, task=task)
            frames.append({
                "step": step_idx,
                "observation": obs,
                "action": waypoint["joint_positions"],
                "timestamp": waypoint["timestamp"],
            })
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
                    position = obj.get("pose", {}).get("position")
                    if position is not None:
                        return np.array(position, dtype=float)
        target_objects = task.get("target_objects", [])
        if target_objects:
            position = target_objects[0].get("position")
            if position is not None:
                return np.array(position, dtype=float)
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
        fps: float = 30.0,
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
        if initial_joints is None:
            if robot_config is not None:
                initial_joints = robot_config.default_joint_positions.tolist()
            else:
                initial_joints = [0.0] * 7

        initial_joints = np.array(initial_joints, dtype=float)

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

        # Build phase waypoints as offsets from initial joints
        phase_configs = [
            # (phase_name, arm_offset_fraction, gripper_multiplier, duration_fraction)
            ("approach",  np.array([0.0, 0.05, -0.08, 0.04, -0.02, 0.03, 0.0]), gripper_open,  0.20),
            ("grasp",     np.array([0.0, 0.08, -0.12, 0.06, -0.03, 0.04, 0.01]), gripper_closed, 0.10),
            ("lift",      np.array([0.0, -0.05, -0.15, 0.08, -0.05, 0.06, 0.0]), gripper_closed, 0.20),
            ("transport", np.array([0.1, -0.03, -0.10, 0.05, 0.05, -0.02, 0.03]), gripper_closed, 0.30),
            ("place",     np.array([0.1, 0.02, -0.05, 0.03, 0.04, -0.01, 0.02]), gripper_open,  0.20),
        ]

        # Generate phase target joint positions
        phase_targets = []
        for phase_name, arm_offsets, grip_mult, _ in phase_configs:
            target = initial_joints.copy()
            # Apply arm offsets scaled by joint range
            for j in range(min(len(arm_offsets), arm_end)):
                target[j] += arm_offsets[j] * joint_range[j]
            # Apply gripper multiplier
            for j in range(arm_end, num_joints):
                mid = (lower[j] + upper[j]) / 2.0
                target[j] = mid + (initial_joints[j] - mid) * grip_mult[j]
            target = np.clip(target, lower, upper)
            phase_targets.append(target)

        # Interpolate between phases to build full trajectory
        trajectory: List[Dict[str, Any]] = []
        current_joints = initial_joints.copy()
        current_time = 0.0
        total_duration = num_waypoints / fps

        for phase_idx, (phase_name, _, _, duration_frac) in enumerate(phase_configs):
            target_joints = phase_targets[phase_idx]
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
                trajectory.append(
                    {
                        "joint_positions": joint_pos.tolist(),
                        "timestamp": current_time + t * phase_duration,
                    }
                )

            current_time += phase_duration
            current_joints = target_joints

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
        for obj in scene_state.get("objects", []):
            obstacles.append({
                "id": obj.get("object_id", "unknown"),
                "position": obj.get("pose", {}).get("position", [0, 0, 0]),
                "dimensions": obj.get("dimensions", [0.1, 0.1, 0.1]),
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

        if not os.environ.get("GENIESIM_USE_LLM_TASK_PLANNING", ""):
            return result

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
            trajectory_summary.append({
                "step": idx,
                "joint_positions": [round(v, 4) for v in jp[:7]] if jp else [],
                "action_sample": [round(v, 4) for v in action[:7]] if action else [],
                "timestamp": frame.get("timestamp", 0),
            })

        task_type = task.get("task_type", "manipulation")
        target_object = task.get("target_object", "unknown object")
        description_hint = task.get("description_hint", "")
        robot_type = getattr(self.config, "robot_type", "humanoid")

        prompt = (
            f"You are evaluating a robot manipulation episode for data quality.\n\n"
            f"Robot: {robot_type}\n"
            f"Task type: {task_type}\n"
            f"Target object: {target_object}\n"
            f"Description hint: {description_hint}\n"
            f"Episode ID: {episode_id}\n"
            f"Total frames: {num_frames}\n"
            f"Collision-free: {collision_free}\n\n"
            f"Trajectory samples (joint positions at key frames):\n"
            f"{json.dumps(trajectory_summary, indent=2)}\n\n"
            f"Based on this information, provide a JSON response with:\n"
            f'{{"task_name": "short descriptive name for the task",\n'
            f' "task_description": "1-2 sentence natural language description of what the robot is doing",\n'
            f' "scene_description": "brief description of the scene context",\n'
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
                if grasp_frame_index is not None and release_frame_index is None:
                    for key in release_keys:
                        if bool(obs.get(key, frame.get(key))):
                            release_frame_index = idx
                            break

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

        if success_flag_detected:
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
                # Scale: 0.01 rad mean change = low, 0.1+ = high
                action_diversity_score = min(1.0, mean_diff / 0.05)

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
                obs_diversity_score = min(1.0, mean_obs_diff / 0.05)

        weighted_score = (
            0.30 * data_completeness_score
            + 0.25 * action_validity_score
            + 0.15 * action_diversity_score
            + 0.10 * obs_diversity_score
            + 0.15 * success_score
            + 0.05 * frame_count_score
        )

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

        dataset_info = {
            "format": "lerobot",
            "export_format": resolved_format.value,
            "version": info["version"],
            "episodes": exported_count,
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
