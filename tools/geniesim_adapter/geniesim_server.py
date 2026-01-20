#!/usr/bin/env python3
"""Local GenieSim gRPC server runner and in-process servicer."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import grpc

from geniesim_grpc_pb2 import (
    CameraObservation,
    CommandResponse,
    CommandType,
    GetIKStatusRequest,
    GetIKStatusResponse,
    GetJointPositionRequest,
    GetJointPositionResponse,
    GetObservationRequest,
    GetObservationResponse,
    JointState,
    Pose,
    Quaternion,
    ResetRequest,
    ResetResponse,
    RobotState,
    SceneState,
    SetJointPositionRequest,
    SetJointPositionResponse,
    SetTrajectoryRequest,
    SetTrajectoryResponse,
    StartRecordingRequest,
    StartRecordingResponse,
    StopRecordingRequest,
    StopRecordingResponse,
    TaskStatusRequest,
    TaskStatusResponse,
    Vector3,
)
from geniesim_grpc_pb2_grpc import (
    GenieSimServiceServicer,
    add_GenieSimServiceServicer_to_server,
)
from tools.logging_config import init_logging
from tools.geniesim_adapter.config import DEFAULT_GENIESIM_PORT, GENIESIM_PORT_ENV
from tools.config.env import parse_int_env

LOGGER = logging.getLogger("geniesim.server")

GENIESIM_RECORDINGS_DIR_ENV = "GENIESIM_RECORDINGS_DIR"
GENIESIM_DEFAULT_JOINT_COUNT_ENV = "GENIESIM_DEFAULT_JOINT_COUNT"

DEFAULT_SERVER_VERSION = os.getenv("GENIESIM_SERVER_VERSION", "3.0.0")
DEFAULT_SERVER_CAPABILITIES = [
    "data_collection",
    "recording",
    "observation",
    "observation_stream",
    "environment_reset",
    "ik_status",
    "object_manipulation",
    "task_status",
]


@dataclass
class RecordingSession:
    """Track a mock recording session."""

    episode_id: str
    output_directory: Path
    started_at: float
    frames_recorded: int = 0
    include_depth: bool = False
    include_semantic: bool = False
    camera_ids: Sequence[str] = field(default_factory=list)


class GenieSimLocalServicer(GenieSimServiceServicer):
    """Concrete servicer implementation for local testing."""

    def __init__(self, joint_count: int = 7) -> None:
        super().__init__(joint_count=joint_count)
        self._lock = threading.Lock()
        self._joint_positions = [0.0] * joint_count
        self._joint_velocities = [0.0] * joint_count
        self._joint_efforts = [0.0] * joint_count
        self._joint_names = [f"joint_{idx}" for idx in range(joint_count)]
        self._recording: Optional[RecordingSession] = None
        self._server_version = DEFAULT_SERVER_VERSION
        self._capabilities = list(DEFAULT_SERVER_CAPABILITIES)
        self._default_recordings_dir = _resolve_default_recordings_dir()

    def GetObservation(
        self,
        request: GetObservationRequest,
        context,
    ) -> GetObservationResponse:
        LOGGER.debug("GetObservation request: %s", request)
        with self._lock:
            joint_state = self._build_joint_state()
        robot_state = RobotState(
            joint_state=joint_state,
            end_effector_pose=self._default_pose(),
            gripper_width=0.0,
            gripper_is_grasping=False,
            link_poses=[],
            link_names=[],
        )
        scene_state = SceneState(objects=[], simulation_time=time.time(), step_count=0)
        camera_observation = CameraObservation(images=[])
        return GetObservationResponse(
            success=True,
            robot_state=robot_state,
            scene_state=scene_state,
            camera_observation=camera_observation,
            timestamp=time.time(),
        )

    def StreamObservations(
        self,
        request: GetObservationRequest,
        context,
    ):
        LOGGER.debug("StreamObservations request")
        for step in range(3):
            with self._lock:
                joint_state = self._build_joint_state()
            robot_state = RobotState(
                joint_state=joint_state,
                end_effector_pose=self._default_pose(),
                gripper_width=0.0,
                gripper_is_grasping=False,
                link_poses=[],
                link_names=[],
            )
            scene_state = SceneState(objects=[], simulation_time=time.time(), step_count=step)
            camera_observation = CameraObservation(images=[])
            yield GetObservationResponse(
                success=True,
                robot_state=robot_state,
                scene_state=scene_state,
                camera_observation=camera_observation,
                timestamp=time.time(),
            )
            time.sleep(0.05)

    def GetJointPosition(
        self,
        request: GetJointPositionRequest,
        context,
    ) -> GetJointPositionResponse:
        LOGGER.debug("GetJointPosition request")
        with self._lock:
            joint_state = self._build_joint_state()
        return GetJointPositionResponse(success=True, joint_state=joint_state)

    def GetIKStatus(
        self,
        request: GetIKStatusRequest,
        context,
    ) -> GetIKStatusResponse:
        LOGGER.debug("GetIKStatus request")
        with self._lock:
            current_positions = list(self._joint_positions)
        solution = list(request.seed_positions) or current_positions
        return GetIKStatusResponse(
            success=True,
            ik_solvable=True,
            solution=solution,
        )

    def GetTaskStatus(
        self,
        request: TaskStatusRequest,
        context,
    ) -> TaskStatusResponse:
        LOGGER.debug("GetTaskStatus request: %s", request.task_id)
        with self._lock:
            has_recording = self._recording is not None
        if request.task_id:
            status = "running"
            progress = 0.5
        else:
            status = "recording" if has_recording else "idle"
            progress = 0.1 if has_recording else 0.0
        return TaskStatusResponse(success=True, status=status, progress=progress)

    def SetJointPosition(
        self,
        request: SetJointPositionRequest,
        context,
    ) -> SetJointPositionResponse:
        LOGGER.info("SetJointPosition: %s", request.positions)
        if not request.positions:
            return SetJointPositionResponse(success=False, error_message="No joint positions provided")
        with self._lock:
            self._joint_positions = list(request.positions)
            self._joint_velocities = list(request.velocities) or [0.0] * len(self._joint_positions)
            self._joint_efforts = [0.0] * len(self._joint_positions)
            joint_state = self._build_joint_state()
        return SetJointPositionResponse(success=True, current_state=joint_state)

    def SetTrajectory(
        self,
        request: SetTrajectoryRequest,
        context,
    ) -> SetTrajectoryResponse:
        LOGGER.info("SetTrajectory with %d points", len(request.points))
        if not request.points:
            return SetTrajectoryResponse(success=False, error_message="Trajectory points are required")
        last_point = request.points[-1]
        with self._lock:
            if last_point.positions:
                self._joint_positions = list(last_point.positions)
                self._joint_velocities = list(last_point.velocities) or [0.0] * len(self._joint_positions)
                self._joint_efforts = [0.0] * len(self._joint_positions)
        execution_time = max(last_point.time_from_start, 0.0)
        return SetTrajectoryResponse(
            success=True,
            points_executed=len(request.points),
            execution_time=execution_time,
        )

    def StartRecording(
        self,
        request: StartRecordingRequest,
        context,
    ) -> StartRecordingResponse:
        episode_id = request.episode_id or f"episode-{int(time.time())}"
        if request.output_directory:
            output_dir = Path(request.output_directory)
        else:
            output_dir = self._default_recordings_dir
            LOGGER.info(
                "Recording output directory not provided; using %s (override with %s)",
                output_dir,
                GENIESIM_RECORDINGS_DIR_ENV,
            )
        recording_path = output_dir / episode_id
        recording_path.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._recording = RecordingSession(
                episode_id=episode_id,
                output_directory=recording_path,
                started_at=time.time(),
                include_depth=request.include_depth,
                include_semantic=request.include_semantic,
                camera_ids=request.camera_ids,
            )
        LOGGER.info("Recording started at %s", recording_path)
        return StartRecordingResponse(success=True, recording_path=str(recording_path))

    def StopRecording(
        self,
        request: StopRecordingRequest,
        context,
    ) -> StopRecordingResponse:
        with self._lock:
            recording = self._recording
            self._recording = None
        if recording is None:
            return StopRecordingResponse(success=False, error_message="No active recording")
        duration = max(time.time() - recording.started_at, 0.0)
        LOGGER.info("Recording stopped after %.2fs", duration)
        return StopRecordingResponse(
            success=True,
            frames_recorded=recording.frames_recorded,
            duration_seconds=duration,
            recording_path=str(recording.output_directory),
        )

    def Reset(
        self,
        request: ResetRequest,
        context,
    ) -> ResetResponse:
        LOGGER.info("Reset requested (robot=%s, objects=%s)", request.reset_robot, request.reset_objects)
        with self._lock:
            self._joint_positions = [0.0] * len(self._joint_positions)
            self._joint_velocities = [0.0] * len(self._joint_positions)
            self._joint_efforts = [0.0] * len(self._joint_positions)
            self._recording = None
        return ResetResponse(success=True)

    def SendCommand(
        self,
        request,
        context,
    ) -> CommandResponse:
        if request.command_type == CommandType.GET_CHECKER_STATUS:
            payload = json.dumps(
                {
                    "version": self._server_version,
                    "capabilities": self._capabilities,
                }
            ).encode()
            return CommandResponse(success=True, payload=payload)
        return super().SendCommand(request, context)

    def _build_joint_state(self) -> JointState:
        return JointState(
            positions=list(self._joint_positions),
            velocities=list(self._joint_velocities),
            efforts=list(self._joint_efforts),
            names=list(self._joint_names),
        )

    @staticmethod
    def _default_pose() -> Pose:
        return Pose(
            position=Vector3(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
        )


def _configure_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    init_logging(level=log_level)


def _health_service_available() -> bool:
    return importlib.util.find_spec("grpc_health.v1.health") is not None


def _add_health_service(server: grpc.Server) -> None:
    if not _health_service_available():
        LOGGER.warning("grpc_health not available; health service disabled")
        return
    health = importlib.import_module("grpc_health.v1.health")
    health_pb2_grpc = importlib.import_module("grpc_health.v1.health_pb2_grpc")
    health_pb2 = importlib.import_module("grpc_health.v1.health_pb2")
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    LOGGER.info("Health service registered")


def _ensure_concrete_servicer(servicer: GenieSimServiceServicer) -> None:
    if type(servicer) is GenieSimServiceServicer:
        raise TypeError(
            "GenieSimServiceServicer is a base class; "
            "register a concrete implementation such as GenieSimLocalServicer."
        )


def run_health_check(host: str, port: int, timeout: float = 5.0) -> bool:
    target = f"{host}:{port}"
    channel = grpc.insecure_channel(target)
    grpc.channel_ready_future(channel).result(timeout=timeout)
    if _health_service_available():
        health_pb2 = importlib.import_module("grpc_health.v1.health_pb2")
        health_pb2_grpc = importlib.import_module("grpc_health.v1.health_pb2_grpc")
        stub = health_pb2_grpc.HealthStub(channel)
        response = stub.Check(health_pb2.HealthCheckRequest(service=""), timeout=timeout)
        return response.status == health_pb2.HealthCheckResponse.SERVING
    return True


def _resolve_default_recordings_dir() -> Path:
    raw = os.getenv(GENIESIM_RECORDINGS_DIR_ENV)
    if raw:
        return Path(raw).expanduser()
    return Path("/tmp/geniesim_recordings")


def _resolve_default_joint_count() -> int:
    raw = os.getenv(GENIESIM_DEFAULT_JOINT_COUNT_ENV)
    if raw is None or raw == "":
        return 7
    try:
        return parse_int_env(
            raw,
            default=7,
            min_value=1,
            name=GENIESIM_DEFAULT_JOINT_COUNT_ENV,
        ) or 7
    except ValueError as exc:
        LOGGER.warning("Invalid %s: %s Defaulting to 7.", GENIESIM_DEFAULT_JOINT_COUNT_ENV, exc)
        return 7


def serve(args: argparse.Namespace) -> None:
    _configure_logging(args.log_level)
    server = grpc.server(thread_pool=ThreadPoolExecutor(max_workers=args.max_workers))
    servicer = GenieSimLocalServicer(joint_count=args.joint_count)
    _ensure_concrete_servicer(servicer)
    add_GenieSimServiceServicer_to_server(servicer, server)
    _add_health_service(server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    LOGGER.info("GenieSim local gRPC server listening on %s:%s", args.host, args.port)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        LOGGER.info("Shutdown requested")
        server.stop(grace=2)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local GenieSim gRPC server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_GENIESIM_PORT,
        help=f"gRPC port (defaults to ${GENIESIM_PORT_ENV} or adapter default)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--max-workers", type=int, default=10, help="gRPC worker threads")
    parser.add_argument(
        "--joint-count",
        type=int,
        default=_resolve_default_joint_count(),
        help=(
            "Number of mock joints (defaults to $GENIESIM_DEFAULT_JOINT_COUNT or 7)"
        ),
    )
    parser.add_argument("--health-check", action="store_true", help="Run health check and exit")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.health_check:
        _configure_logging(args.log_level)
        try:
            healthy = run_health_check(args.host, args.port)
        except Exception as exc:
            LOGGER.error("Health check failed: %s", exc)
            return 1
        LOGGER.info("Health check status: %s", "SERVING" if healthy else "NOT_SERVING")
        return 0 if healthy else 1
    serve(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
