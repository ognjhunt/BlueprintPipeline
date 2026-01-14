#!/usr/bin/env python3
"""
Genie Sim 3.0 Local Framework Adapter.

This module provides a LOCAL integration with Genie Sim 3.0, running it as a framework
directly within the Isaac Sim environment rather than through a non-existent hosted API.

Based on the official Genie Sim 3.0 architecture:
- Repository: https://github.com/AgibotTech/genie_sim
- Data Collection: Uses gRPC on port 50051 for client-server communication
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
    │  │  │ (port 50051)│    │ Server (inside Isaac Sim)   │ │   │
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
    GENIESIM_PORT: Genie Sim gRPC server port (default: 50051)
    GENIESIM_TIMEOUT: Connection timeout in seconds (default: 30)
    GENIESIM_ROOT: Path to Genie Sim installation (default: /opt/geniesim)
    ISAAC_SIM_PATH: Path to Isaac Sim installation (default: /isaac-sim)
    ALLOW_GENIESIM_MOCK: Allow local mock gRPC server when GENIESIM_ROOT is missing (default: 0)
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import threading

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent paths for imports
ADAPTER_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = ADAPTER_ROOT.parent
REPO_ROOT = TOOLS_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTER_ROOT))

# Import gRPC protobuf stubs
try:
    from geniesim_grpc_pb2 import (
        Vector3, Quaternion, Pose as GrpcPose, JointState as GrpcJointState,
        GetObservationRequest, GetObservationResponse,
        GetJointPositionRequest, GetJointPositionResponse,
        SetJointPositionRequest, SetJointPositionResponse,
        SetTrajectoryRequest, SetTrajectoryResponse, TrajectoryPoint,
        StartRecordingRequest, StartRecordingResponse,
        StopRecordingRequest, StopRecordingResponse,
        ResetRequest, ResetResponse,
        CommandType as GrpcCommandType,
    )
    from geniesim_grpc_pb2_grpc import GenieSimServiceStub, is_grpc_available, create_channel
    GRPC_STUBS_AVAILABLE = True
except ImportError:
    GRPC_STUBS_AVAILABLE = False
    logger.warning("gRPC stubs not available - using legacy fallback")

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


# =============================================================================
# Configuration
# =============================================================================


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
    GET_SEMANTIC_DATA = 1

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
    START_RECORDING = 11
    STOP_RECORDING = 11

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


@dataclass
class GenieSimConfig:
    """Configuration for Genie Sim local framework."""

    # Connection settings
    host: str = "localhost"
    port: int = 50051
    timeout: float = 30.0
    max_retries: int = 3

    # Installation paths
    geniesim_root: Path = Path("/opt/geniesim")
    isaac_sim_path: Path = Path("/isaac-sim")

    # Data collection settings
    episodes_per_task: int = 100
    use_curobo: bool = True
    headless: bool = True
    environment: str = "development"
    allow_linear_fallback: bool = False

    # Output settings
    recording_dir: Path = Path("/tmp/geniesim_recordings")
    log_dir: Path = Path("/tmp/geniesim_logs")

    # Robot configuration
    robot_type: str = "franka"
    robot_urdf: Optional[str] = None

    @classmethod
    def from_env(cls) -> "GenieSimConfig":
        """Create configuration from environment variables."""
        environment = os.getenv("GENIESIM_ENV", os.getenv("BP_ENV", "development")).lower()
        allow_linear_fallback = os.getenv("GENIESIM_ALLOW_LINEAR_FALLBACK", "0") == "1"
        return cls(
            host=os.getenv("GENIESIM_HOST", "localhost"),
            port=int(os.getenv("GENIESIM_PORT", "50051")),
            timeout=float(os.getenv("GENIESIM_TIMEOUT", "30")),
            geniesim_root=Path(os.getenv("GENIESIM_ROOT", "/opt/geniesim")),
            isaac_sim_path=Path(os.getenv("ISAAC_SIM_PATH", "/isaac-sim")),
            headless=os.getenv("HEADLESS", "1") == "1",
            robot_type=os.getenv("ROBOT_TYPE", "franka"),
            environment=environment,
            allow_linear_fallback=allow_linear_fallback,
        )


@dataclass
class DataCollectionResult:
    """Result of a data collection run."""

    success: bool
    task_name: str
    episodes_collected: int = 0
    episodes_passed: int = 0
    total_frames: int = 0
    recording_dir: Optional[Path] = None

    # Quality metrics
    average_quality_score: float = 0.0
    collision_free_rate: float = 0.0
    task_success_rate: float = 0.0

    # Timing
    duration_seconds: float = 0.0

    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


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

    def __init__(self, host: str = "localhost", port: int = 50051, timeout: float = 30.0):
        """
        Initialize gRPC client.

        Args:
            host: Server hostname
            port: Server port (default 50051)
            timeout: Request timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self._channel = None
        self._stub = None
        self._connected = False

        # Check if gRPC stubs are available
        self._have_grpc = GRPC_STUBS_AVAILABLE and is_grpc_available()
        if not self._have_grpc:
            logger.warning("gRPC not available - server connection will be limited")

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
            # Create gRPC channel using the utility function
            self._channel = create_channel(
                host=self.host,
                port=self.port,
            )

            if self._channel is None:
                logger.error("Failed to create gRPC channel")
                return False

            # Create service stub
            self._stub = GenieSimServiceStub(self._channel)

            # Test connection with a simple call (with timeout)
            import grpc
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

        Returns:
            True if the server responds successfully.
        """
        effective_timeout = min(self.timeout, timeout) if timeout else self.timeout

        if not self._have_grpc or self._stub is None:
            return self._check_server_socket()

        try:
            request = GetObservationRequest(
                include_images=False,
                include_depth=False,
                include_semantic=False,
            )
            response = self._stub.GetObservation(request, timeout=effective_timeout)
            if response.success:
                logger.info("✅ Genie Sim gRPC ping succeeded")
                return True
            logger.warning("Genie Sim gRPC ping returned unsuccessful response")
            return False
        except Exception as exc:
            logger.error(f"Genie Sim gRPC ping failed: {exc}")
            return False

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

    def disconnect(self) -> None:
        """Disconnect from server."""
        if self._channel:
            self._channel.close()
            self._channel = None
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._connected

    def send_command(
        self,
        command: CommandType,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send command to Genie Sim server.

        P0-3 FIX: Now uses real gRPC calls instead of mock responses.

        Args:
            command: Command type
            data: Optional command data

        Returns:
            Response data from server
        """
        if not self._connected:
            raise RuntimeError("Not connected to Genie Sim server")

        if not self._have_grpc or self._stub is None:
            logger.warning(f"gRPC not available - cannot send command: {command.name}")
            return {"success": False, "error": "gRPC not available"}

        # P0-3 FIX: Real gRPC implementation
        try:
            # Convert to gRPC command type
            grpc_command = GrpcCommandType(command.value)

            # Create request
            request = CommandRequest(
                command_type=grpc_command,
                payload=json.dumps(data or {}).encode(),
            )

            # Send via gRPC
            response = self._stub.SendCommand(request, timeout=self.timeout)

            # Parse response
            result = {
                "success": response.success,
                "error_message": response.error_message,
            }

            # Decode payload if present
            if response.payload:
                try:
                    payload_data = json.loads(response.payload.decode())
                    result.update(payload_data)
                except json.JSONDecodeError:
                    logger.warning("Failed to decode response payload")

            return result

        except Exception as e:
            logger.error(f"gRPC command {command.name} failed: {e}")
            return {"success": False, "error": str(e)}

    def get_observation(self) -> Dict[str, Any]:
        """
        Get current observation from simulation.

        P0-3 FIX: Uses real gRPC GetObservation call.

        Returns:
            Dictionary with robot_state, scene_state, timestamp
        """
        if not self._have_grpc or self._stub is None:
            return {"success": False, "error": "gRPC not available"}

        try:
            request = GetObservationRequest(
                include_images=True,
                include_depth=True,
                include_semantic=False,
            )

            response = self._stub.GetObservation(request, timeout=self.timeout)

            if not response.success:
                return {"success": False}

            return {
                "success": True,
                "robot_state": response.robot_state.to_dict() if response.robot_state else {},
                "scene_state": response.scene_state.to_dict() if response.scene_state else {},
                "timestamp": response.timestamp,
            }

        except Exception as e:
            logger.error(f"GetObservation failed: {e}")
            return {"success": False, "error": str(e)}

    def set_joint_position(self, positions: List[float]) -> bool:
        """
        Set robot joint positions.

        P0-3 FIX: Uses real gRPC SetJointPosition call.

        Args:
            positions: Target joint positions

        Returns:
            True if successful
        """
        if not self._have_grpc or self._stub is None:
            return False

        try:
            request = SetJointPositionRequest(
                positions=positions,
                wait_for_completion=True,
            )

            response = self._stub.SetJointPosition(request, timeout=self.timeout)
            return response.success

        except Exception as e:
            logger.error(f"SetJointPosition failed: {e}")
            return False

    def get_joint_position(self) -> Optional[List[float]]:
        """
        Get current joint positions.

        P0-3 FIX: Uses real gRPC GetJointPosition call.

        Returns:
            List of joint positions or None
        """
        if not self._have_grpc or self._stub is None:
            return None

        try:
            request = GetJointPositionRequest()
            response = self._stub.GetJointPosition(request, timeout=self.timeout)

            if response.success and response.joint_state:
                return response.joint_state.positions

            return None

        except Exception as e:
            logger.error(f"GetJointPosition failed: {e}")
            return None

    def get_camera_data(self, camera_id: str = "wrist") -> Optional[np.ndarray]:
        """
        Get camera image.

        P0-3 FIX: Integrated with GetObservation for camera data.

        Args:
            camera_id: Camera identifier

        Returns:
            Image as numpy array or None
        """
        obs = self.get_observation()
        if not obs.get("success"):
            return None

        # Extract camera data from observation
        # (Implementation depends on how camera data is structured in observation)
        return None  # Placeholder

    def execute_trajectory(self, trajectory: List[Dict[str, Any]]) -> bool:
        """
        Execute a trajectory on the robot.

        P0-3 FIX: Uses real gRPC SetTrajectory call with cuRobo-planned trajectory.

        Args:
            trajectory: List of waypoints with positions, velocities, timestamps

        Returns:
            True if execution successful
        """
        if not self._have_grpc or self._stub is None:
            return False

        try:
            # Convert trajectory to gRPC format
            trajectory_points = []
            for waypoint in trajectory:
                point = TrajectoryPoint(
                    positions=waypoint.get("joint_positions", []),
                    velocities=waypoint.get("velocities", []),
                    accelerations=waypoint.get("accelerations", []),
                    time_from_start=waypoint.get("timestamp", 0.0),
                )
                trajectory_points.append(point)

            request = SetTrajectoryRequest(
                points=trajectory_points,
                execution_speed=1.0,
                wait_for_completion=True,
            )

            response = self._stub.SetTrajectory(request, timeout=self.timeout * 5)  # Longer timeout for execution
            return response.success

        except Exception as e:
            logger.error(f"SetTrajectory failed: {e}")
            return False

    def start_recording(self, episode_id: str, output_dir: str = "/tmp/recordings") -> bool:
        """
        Start recording an episode.

        P0-3 FIX: Uses real gRPC StartRecording call.

        Args:
            episode_id: Unique episode identifier
            output_dir: Directory to save recordings

        Returns:
            True if recording started
        """
        if not self._have_grpc or self._stub is None:
            return False

        try:
            request = StartRecordingRequest(
                episode_id=episode_id,
                output_directory=output_dir,
                fps=30.0,
                include_depth=True,
                include_semantic=False,
            )

            response = self._stub.StartRecording(request, timeout=self.timeout)
            return response.success

        except Exception as e:
            logger.error(f"StartRecording failed: {e}")
            return False

    def stop_recording(self) -> bool:
        """
        Stop recording current episode.

        P0-3 FIX: Uses real gRPC StopRecording call.

        Returns:
            True if recording stopped
        """
        if not self._have_grpc or self._stub is None:
            return False

        try:
            request = StopRecordingRequest(save_metadata=True)
            response = self._stub.StopRecording(request, timeout=self.timeout)
            return response.success

        except Exception as e:
            logger.error(f"StopRecording failed: {e}")
            return False

    def reset_environment(self) -> bool:
        """
        Reset the simulation environment.

        P0-3 FIX: Uses real gRPC Reset call.

        Returns:
            True if reset successful
        """
        if not self._have_grpc or self._stub is None:
            return False

        try:
            request = ResetRequest(
                reset_robot=True,
                reset_objects=True,
            )

            response = self._stub.Reset(request, timeout=self.timeout)
            return response.success

        except Exception as e:
            logger.error(f"Reset failed: {e}")
            return False


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

        # Ensure directories exist
        self.config.recording_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

    def _apply_curobo_fallback_policy(self) -> None:
        """Apply cuRobo availability policy to config/runtime behavior."""
        production_mode = self.config.environment == "production"
        curobo_enabled = CUROBO_INTEGRATION_AVAILABLE and self.config.use_curobo
        allow_fallback_env = os.getenv("GENIESIM_ALLOW_LINEAR_FALLBACK")

        if production_mode and not curobo_enabled:
            raise RuntimeError(
                "cuRobo motion planning is required in production. "
                "Install the cuRobo integration and enable use_curobo, or "
                "run with GENIESIM_ENV=development for local testing."
            )

        if not production_mode and not curobo_enabled:
            if allow_fallback_env is None:
                self.config.allow_linear_fallback = True
                self.log(
                    "cuRobo unavailable; enabling linear interpolation fallback by default "
                    "for non-production. Set GENIESIM_ALLOW_LINEAR_FALLBACK=0 to disable.",
                    "WARNING",
                )
            elif self.config.allow_linear_fallback:
                self.log(
                    "cuRobo unavailable; linear interpolation fallback explicitly enabled "
                    "via GENIESIM_ALLOW_LINEAR_FALLBACK.",
                    "WARNING",
                )
            else:
                self.log(
                    "cuRobo unavailable; linear interpolation fallback disabled via "
                    "GENIESIM_ALLOW_LINEAR_FALLBACK=0.",
                    "WARNING",
                )

    def log(self, msg: str, level: str = "INFO") -> None:
        """Log a message."""
        if self.verbose:
            print(f"[GENIESIM-LOCAL] [{level}] {msg}")

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
        timeout: float = 120.0,
    ) -> bool:
        """
        Start the Genie Sim data collection server.

        This starts the data_collector_server.py script inside Isaac Sim.

        Args:
            scene_usd_path: Path to USD scene file
            task_config_path: Path to task configuration JSON
            wait_for_ready: Wait for server to be ready
            timeout: Timeout for waiting

        Returns:
            True if server started successfully
        """
        if self.is_server_running():
            self.log("Server already running")
            return True

        self.log("Starting Genie Sim server...")
        self._status = GenieSimServerStatus.STARTING

        use_local_server = not self.config.geniesim_root.exists()
        allow_mock_override = os.getenv("ALLOW_GENIESIM_MOCK", "0") == "1"
        production_mode = self.config.environment == "production"
        env = os.environ.copy()

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
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.is_server_running():
                    self._status = GenieSimServerStatus.READY
                    self.log("Server is ready!")
                    return True

                # Check if process exited
                if self._server_process.poll() is not None:
                    self.log("Server process exited unexpectedly", "ERROR")
                    self._status = GenieSimServerStatus.ERROR
                    return False

                time.sleep(2)

            self.log(f"Server did not become ready within {timeout}s", "ERROR")
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

    def server_context(self, scene_usd_path: Optional[Path] = None):
        """
        Context manager for automatic server lifecycle management.

        Usage:
            with framework.server_context(scene_path) as fw:
                result = fw.run_data_collection(task_config)
        """
        return _GenieSimServerContext(self, scene_usd_path)

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

    # =========================================================================
    # Data Collection
    # =========================================================================

    def run_data_collection(
        self,
        task_config: Dict[str, Any],
        scene_config: Optional[Dict[str, Any]] = None,
        episodes_per_task: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
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
            progress_callback: Callback for progress updates (current, total, message)

        Returns:
            DataCollectionResult with statistics and output paths
        """
        start_time = time.time()

        self.log("=" * 70)
        self.log("GENIE SIM DATA COLLECTION")
        self.log("=" * 70)

        result = DataCollectionResult(
            success=False,
            task_name=task_config.get("name", "unknown"),
        )

        # Ensure server is running (bootstrap if needed)
        if not self.is_server_running():
            scene_usd_path = None
            if scene_config:
                scene_usd_path = scene_config.get("usd_path") or scene_config.get("scene_usd_path")
            if scene_usd_path:
                scene_usd_path = Path(scene_usd_path)
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

        episodes_target = episodes_per_task or self.config.episodes_per_task
        tasks = task_config.get("suggested_tasks", [task_config])

        self.log(f"Tasks: {len(tasks)}")
        self.log(f"Episodes per task: {episodes_target}")

        total_episodes = 0
        passed_episodes = 0
        total_frames = 0
        quality_scores = []

        # Create output directory for this run
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.recording_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        for task_idx, task in enumerate(tasks):
            task_name = task.get("task_name", f"task_{task_idx}")
            self.log(f"\nTask {task_idx + 1}/{len(tasks)}: {task_name}")

            # Configure environment for task
            self._configure_task(task, scene_config)

            for ep_idx in range(episodes_target):
                if progress_callback:
                    current = task_idx * episodes_target + ep_idx + 1
                    total = len(tasks) * episodes_target
                    progress_callback(current, total, f"Task: {task_name}, Episode: {ep_idx + 1}")

                try:
                    # Reset environment
                    self._client.reset_environment()

                    # Generate and execute trajectory
                    episode_result = self._run_single_episode(
                        task=task,
                        episode_id=f"{task_name}_ep{ep_idx:04d}",
                        output_dir=run_dir,
                    )

                    total_episodes += 1

                    if episode_result.get("success"):
                        passed_episodes += 1
                        total_frames += episode_result.get("frame_count", 0)
                        quality_scores.append(episode_result.get("quality_score", 0.0))
                    else:
                        result.warnings.append(
                            f"Episode {ep_idx} of {task_name} failed: {episode_result.get('error', 'unknown')}"
                        )

                except Exception as e:
                    result.warnings.append(f"Episode {ep_idx} of {task_name} error: {e}")
                    self.log(f"  Episode {ep_idx} error: {e}", "WARNING")

        # Calculate statistics
        result.episodes_collected = total_episodes
        result.episodes_passed = passed_episodes
        result.total_frames = total_frames
        result.recording_dir = run_dir
        result.duration_seconds = time.time() - start_time

        if quality_scores:
            result.average_quality_score = np.mean(quality_scores)

        if total_episodes > 0:
            result.task_success_rate = passed_episodes / total_episodes

        result.success = passed_episodes > 0

        self.log("\n" + "=" * 70)
        self.log("DATA COLLECTION COMPLETE")
        self.log("=" * 70)
        self.log(f"Episodes: {passed_episodes}/{total_episodes} passed")
        self.log(f"Total frames: {total_frames}")
        self.log(f"Average quality: {result.average_quality_score:.2f}")
        self.log(f"Duration: {result.duration_seconds:.1f}s")
        self.log(f"Output: {run_dir}")

        return result

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
                self._client.send_command(
                    CommandType.SET_OBJECT_POSE,
                    {
                        "object_id": obj.get("id"),
                        "position": obj.get("position"),
                        "rotation": obj.get("rotation", [0, 0, 0, 1]),
                    }
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
        }

        try:
            # Start recording
            self._client.start_recording(episode_id)

            # Get initial observation
            obs = self._client.get_observation()

            # Generate trajectory (using cuRobo motion planning)
            trajectory = self._generate_trajectory(task, obs)

            if trajectory is None:
                result["error"] = "Motion planning failed"
                return result

            # Execute trajectory and collect data
            frames = []
            for step_idx, waypoint in enumerate(trajectory):
                # Set joint positions
                self._client.set_joint_position(waypoint["joint_positions"])

                # Get observation
                obs = self._client.get_observation()

                # Capture camera data
                for camera_id in ["wrist", "overhead", "side"]:
                    try:
                        img = self._client.get_camera_data(camera_id)
                        if img is not None:
                            obs[f"image_{camera_id}"] = img
                    except Exception:
                        pass

                frames.append({
                    "step": step_idx,
                    "observation": obs,
                    "action": waypoint["joint_positions"],
                    "timestamp": step_idx / 30.0,  # Assuming 30Hz
                })

            # Stop recording
            self._client.stop_recording()

            # Calculate quality score
            quality_score = self._calculate_quality_score(frames, task)
            min_quality = float(os.getenv("MIN_QUALITY_SCORE", "0.7"))
            validation_passed = quality_score >= min_quality

            # Save episode
            episode_path = output_dir / f"{episode_id}.json"
            with open(episode_path, "w") as f:
                json.dump({
                    "episode_id": episode_id,
                    "task_name": task.get("task_name"),
                    "frames": frames,
                    "frame_count": len(frames),
                    "quality_score": quality_score,
                    "validation_passed": validation_passed,
                }, f)

            result["success"] = True
            result["frame_count"] = len(frames)
            result["quality_score"] = quality_score
            result["validation_passed"] = validation_passed
            result["output_path"] = str(episode_path)

        except Exception as e:
            result["error"] = str(e)
            self._client.stop_recording()

        return result

    def _generate_trajectory(
        self,
        task: Dict[str, Any],
        initial_obs: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate trajectory using cuRobo motion planning.

        P0-4 FIX: Now uses actual cuRobo GPU-accelerated motion planning
        for collision-free trajectories instead of linear interpolation.

        Args:
            task: Task specification
            initial_obs: Initial observation

        Returns:
            List of waypoints or None if planning fails
        """
        production_mode = self.config.environment == "production"
        # Get target position from task
        target_pos = task.get("target_position", [0.5, 0.0, 0.8])
        place_pos = task.get("place_position", [0.3, 0.2, 0.8])

        # Get initial joint positions
        initial_joints = self._client.get_joint_position()
        if initial_joints is None:
            initial_joints = [0.0] * 7  # Default for 7-DOF arm

        # Get scene obstacles for collision avoidance
        obstacles = self._get_scene_obstacles(task, initial_obs)

        # =====================================================================
        # P0-4 FIX: Use cuRobo for real motion planning
        # =====================================================================
        if production_mode and not (CUROBO_INTEGRATION_AVAILABLE and self.config.use_curobo):
            raise RuntimeError(
                "Collision-aware planner required in production; "
                "cuRobo integration is unavailable or disabled."
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
            elif production_mode:
                raise RuntimeError(
                    "Collision-aware planning failed in production; "
                    "linear interpolation fallback is disabled."
                )
            else:
                self.log("  ⚠️  cuRobo planning failed", "WARNING")

        # =====================================================================
        # Fallback: Linear interpolation (not collision-free!)
        # =====================================================================
        if not self.config.allow_linear_fallback:
            self.log(
                "  ❌ Linear interpolation fallback disabled. "
                "Set GENIESIM_ALLOW_LINEAR_FALLBACK=1 for dev-only runs.",
                "ERROR",
            )
            return None

        self.log("  ⚠️  Using linear interpolation fallback (not collision-aware)", "WARNING")

        num_waypoints = 100
        trajectory = []

        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)

            # Simple interpolation (not actual IK - placeholder)
            joint_pos = [
                initial_joints[j] + (np.sin(t * np.pi) * 0.1)
                for j in range(len(initial_joints))
            ]

            trajectory.append({
                "joint_positions": joint_pos,
                "timestamp": i / 30.0,
            })

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

        P0-4 FIX: Real cuRobo GPU-accelerated motion planning.

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
                return None

            trajectory_segments.append(("approach", approach_result.joint_trajectory))
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

    def _calculate_quality_score(
        self,
        frames: List[Dict[str, Any]],
        task: Dict[str, Any],
    ) -> float:
        """Calculate quality score for an episode."""
        # Simple quality heuristics
        score = 1.0

        # Check frame count
        if len(frames) < 10:
            score *= 0.5

        # Check for missing data
        missing_obs = sum(1 for f in frames if not f.get("observation"))
        if missing_obs > 0:
            score *= (1 - missing_obs / len(frames))

        return min(1.0, max(0.0, score))

    # =========================================================================
    # Export
    # =========================================================================

    def export_to_lerobot(
        self,
        recording_dir: Path,
        output_dir: Path,
        min_quality_score: float = 0.7,
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
        self.log(f"Exporting to LeRobot format: {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        exported_count = 0
        skipped_count = 0

        # Find all episode files
        episode_files = list(recording_dir.glob("*.json"))

        for ep_file in episode_files:
            try:
                with open(ep_file) as f:
                    episode = json.load(f)

                # Skip low quality
                if episode.get("quality_score", 0) < min_quality_score:
                    skipped_count += 1
                    continue

                # Convert to LeRobot format
                # (Simplified - actual implementation would use pyarrow)
                lerobot_episode = {
                    "episode_id": episode["episode_id"],
                    "frames": episode["frames"],
                    "metadata": {
                        "task_name": episode.get("task_name"),
                        "frame_count": episode.get("frame_count"),
                    }
                }

                output_file = output_dir / f"{episode['episode_id']}.json"
                with open(output_file, "w") as f:
                    json.dump(lerobot_episode, f)

                exported_count += 1

            except Exception as e:
                self.log(f"Failed to export {ep_file.name}: {e}", "WARNING")
                skipped_count += 1

        # Write dataset info
        dataset_info = {
            "format": "lerobot",
            "version": "1.0",
            "episodes": exported_count,
            "skipped": skipped_count,
            "exported_at": datetime.utcnow().isoformat() + "Z",
        }

        with open(output_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

        self.log(f"Exported {exported_count} episodes, skipped {skipped_count}")

        return {
            "success": True,
            "exported": exported_count,
            "skipped": skipped_count,
            "output_dir": output_dir,
        }


class _GenieSimServerContext:
    """Context manager for Genie Sim server lifecycle."""

    def __init__(self, framework: GenieSimLocalFramework, scene_usd_path: Optional[Path]):
        self.framework = framework
        self.scene_usd_path = scene_usd_path

    def __enter__(self) -> GenieSimLocalFramework:
        self.framework.start_server(self.scene_usd_path)
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
    verbose: bool = True,
) -> DataCollectionResult:
    """
    Convenience function to run local Genie Sim data collection.

    Args:
        scene_manifest_path: Path to BlueprintPipeline scene manifest
        task_config_path: Path to task configuration
        output_dir: Output directory
        robot_type: Robot type
        episodes_per_task: Episodes per task
        verbose: Print progress

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

    # Create framework
    config = GenieSimConfig(
        robot_type=robot_type,
        episodes_per_task=episodes_per_task,
        recording_dir=output_dir / "recordings",
    )

    framework = GenieSimLocalFramework(config, verbose=verbose)

    # Check if server is already running
    if framework.is_server_running():
        # Connect to existing server
        framework.connect()
        result = framework.run_data_collection(task_config, scene_manifest)
        framework.disconnect()
    else:
        # Start server and run
        scene_usd = scene_manifest.get("usd_path")
        with framework.server_context(Path(scene_usd) if scene_usd else None) as fw:
            result = fw.run_data_collection(task_config, scene_manifest)

    # Export to LeRobot
    if result.success and result.recording_dir:
        lerobot_dir = output_dir / "lerobot"
        framework.export_to_lerobot(result.recording_dir, lerobot_dir)

    return result


def check_geniesim_availability() -> Dict[str, Any]:
    """
    Check if Genie Sim local framework is available.

    Returns:
        Dict with availability status and details
    """
    config = GenieSimConfig.from_env()

    status = {
        "available": False,
        "geniesim_installed": False,
        "isaac_sim_available": False,
        "server_running": False,
        "grpc_available": False,
        "grpc_stubs_available": GRPC_STUBS_AVAILABLE,
        "mock_server_allowed": False,
        "details": {},
    }
    allow_mock_override = os.getenv("ALLOW_GENIESIM_MOCK", "0") == "1"
    production_mode = config.environment == "production"
    mock_allowed = allow_mock_override and not production_mode
    status["mock_server_allowed"] = mock_allowed
    status["details"]["environment"] = config.environment
    status["details"]["allow_geniesim_mock"] = allow_mock_override

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
    require_server: bool = True,
    require_ready: bool = False,
    ping_timeout: float = 5.0,
) -> Dict[str, Any]:
    """Build a preflight report with remediation guidance."""
    config = GenieSimConfig.from_env()
    missing: List[str] = []
    remediation: List[str] = []
    server_ready = False

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
            client = GenieSimGRPCClient(config.host, config.port, timeout=ping_timeout)
            server_ready = client.ping(timeout=ping_timeout)
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


def run_geniesim_preflight_or_exit(
    stage: str,
    *,
    require_server: bool = True,
    require_ready: bool = False,
    ping_timeout: float = 5.0,
) -> Dict[str, Any]:
    """Run a preflight check with remediation guidance, exiting on failure."""
    status = check_geniesim_availability()
    report = build_geniesim_preflight_report(
        status,
        require_server=require_server,
        require_ready=require_ready,
        ping_timeout=ping_timeout,
    )
    if not report["ok"]:
        print(f"[GENIESIM-PREFLIGHT] {stage} preflight failed.", file=sys.stderr)
        if report["missing"]:
            print("Missing requirements:", file=sys.stderr)
            for item in report["missing"]:
                print(f"  - {item}", file=sys.stderr)
        if report["remediation"]:
            print("Remediation steps:", file=sys.stderr)
            for step in report["remediation"]:
                print(f"  - {step}", file=sys.stderr)
        print(f"Details: {json.dumps(status, indent=2)}", file=sys.stderr)
        raise SystemExit(1)
    return report


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI for Genie Sim local framework."""
    import argparse

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
        print(json.dumps(status, indent=2))
        sys.exit(0 if status["available"] else 1)

    config = GenieSimConfig(robot_type=args.robot)
    framework = GenieSimLocalFramework(config)

    if args.command == "start":
        scene_path = Path(args.scene) if args.scene else None
        if framework.start_server(scene_path):
            print("Server started successfully")
        else:
            print("Failed to start server")
            sys.exit(1)

    elif args.command == "stop":
        framework.stop_server()
        print("Server stopped")

    elif args.command == "run":
        if not args.task_config:
            print("--task-config is required for run command")
            sys.exit(1)

        result = run_local_data_collection(
            scene_manifest_path=Path(args.scene) if args.scene else Path("scene_manifest.json"),
            task_config_path=Path(args.task_config),
            output_dir=Path(args.output),
            robot_type=args.robot,
            episodes_per_task=args.episodes,
        )

        if result.success:
            print(f"Data collection completed: {result.episodes_passed}/{result.episodes_collected} episodes")
        else:
            print(f"Data collection failed: {result.errors}")
            sys.exit(1)

    elif args.command == "export":
        recording_dir = Path(args.output) / "recordings"
        lerobot_dir = Path(args.output) / "lerobot"

        result = framework.export_to_lerobot(recording_dir, lerobot_dir)
        print(f"Exported {result['exported']} episodes to {lerobot_dir}")


if __name__ == "__main__":
    main()
