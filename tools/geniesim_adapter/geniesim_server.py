#!/usr/bin/env python3
"""Local GenieSim gRPC server runner and in-process servicer."""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import struct

import grpc

from tools.geniesim_adapter.geniesim_grpc_pb2 import (
    AddCameraReq,
    AddCameraRsp,
    AttachReq,
    AttachRsp,
    CamImage,
    CamInfo,
    CameraRsp,
    DetachReq,
    DetachRsp,
    ExitReq,
    ExitRsp,
    GetCheckerStatusReq,
    GetCheckerStatusRsp,
    GetObservationReq,
    GetObservationRsp,
    GripperRsp,
    InitRobotReq,
    InitRobotRsp,
    JointRsp,
    ObjectRsp,
    PlaybackReq,
    PlaybackRsp,
    RemoveObstacleReq,
    RemoveObstacleRsp,
    ResetReq,
    ResetRsp,
    SemanticData,
    SetFrameStateReq,
    SetFrameStateRsp,
    SetLightReq,
    SetLightRsp,
    SetMaterailReq,
    SetMaterialRsp,
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
)
from tools.geniesim_adapter.geniesim_grpc_pb2_grpc import (
    SimObservationServiceServicer,
    add_SimObservationServiceServicer_to_server,
)
from aimdk.protocol.common.se3_pose_pb2 import SE3RpyPose
from aimdk.protocol.common.vec3_pb2 import Vec3
from aimdk.protocol.common import rpy_pb2
from aimdk.protocol.hal.joint.joint_channel_pb2 import (
    GetJointRsp,
    SetJointRsp,
    GetIKStatusRsp,
    GetEEPoseRsp,
)
from aimdk.protocol.hal.joint.joint_channel_pb2_grpc import (
    JointControlServiceServicer,
    add_JointControlServiceServicer_to_server,
)
from aimdk.protocol.common.joint_pb2 import JointState
from tools.logging_config import init_logging
from tools.geniesim_adapter.config import (
    DEFAULT_GENIESIM_PORT,
    GENIESIM_PORT_ENV,
    GENIESIM_TLS_CA_ENV,
    GENIESIM_TLS_CERT_ENV,
    GENIESIM_TLS_KEY_ENV,
)
from tools.config.env import parse_int_env

LOGGER = logging.getLogger("geniesim.server")


def _make_synthetic_png(width: int = 64, height: int = 64) -> bytes:
    """Create a minimal valid PNG image (solid black) without external deps."""
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    header = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw_rows = b"".join(b"\x00" + b"\x00" * (width * 3) for _ in range(height))
    compressed = zlib.compress(raw_rows)
    return header + _chunk(b"IHDR", ihdr_data) + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b"")


def _load_tls_credentials() -> Optional[grpc.ServerCredentials]:
    cert_path = os.getenv(GENIESIM_TLS_CERT_ENV)
    key_path = os.getenv(GENIESIM_TLS_KEY_ENV)
    ca_path = os.getenv(GENIESIM_TLS_CA_ENV)
    if not cert_path or not key_path:
        return None
    cert_bytes = Path(cert_path).read_bytes()
    key_bytes = Path(key_path).read_bytes()
    if ca_path:
        ca_bytes = Path(ca_path).read_bytes()
        return grpc.ssl_server_credentials(((key_bytes, cert_bytes),), root_certificates=ca_bytes)
    return grpc.ssl_server_credentials(((key_bytes, cert_bytes),))


class GenieSimLocalServicer(SimObservationServiceServicer):
    """Minimal servicer implementation for local testing."""

    def __init__(self) -> None:
        self._recording_state = "stopped"

    def get_observation(self, req: GetObservationReq, context) -> GetObservationRsp:
        rsp = GetObservationRsp()
        if req.startRecording:
            self._recording_state = "recording"
        if req.stopRecording:
            self._recording_state = "stopped"
        rsp.recordingState = self._recording_state
        if req.startRecording or req.stopRecording:
            return rsp

        # --- Synthetic camera data (4x4 black RGB PNG) ---
        if req.isCam:
            synthetic_rgb = _make_synthetic_png(64, 64)
            synthetic_depth = _make_synthetic_png(64, 64)
            cam_rsp = CameraRsp(
                camera_info=CamInfo(width=64, height=64, ppx=32.0, ppy=32.0, fx=60.0, fy=60.0),
                rgb_camera=CamImage(format="png", data=synthetic_rgb),
                depth_camera=CamImage(format="png", data=synthetic_depth),
                semantic_mask=SemanticData(name="semantic", data=synthetic_depth),
            )
            rsp.camera.append(cam_rsp)

        # --- Synthetic joint state (7-DOF arm) ---
        if req.isJoint:
            for i in range(7):
                rsp.joint.left_arm.append(JointState(name=f"joint_{i}", position=0.0))

        # --- Synthetic gripper pose ---
        if req.isGripper:
            zero_pose = SE3RpyPose(
                position=Vec3(x=0.0, y=0.0, z=0.5),
                rpy=rpy_pb2.Rpy(rw=1.0, rx=0.0, ry=0.0, rz=0.0),
            )
            rsp.gripper.CopyFrom(GripperRsp(left_gripper=zero_pose, right_gripper=zero_pose))

        # --- Synthetic object poses ---
        if req.isPose:
            obj_pose = SE3RpyPose(
                position=Vec3(x=0.3, y=0.0, z=0.8),
                rpy=rpy_pb2.Rpy(rw=1.0, rx=0.0, ry=0.0, rz=0.0),
            )
            rsp.pose.append(ObjectRsp(object_pose=obj_pose))

        return rsp

    def reset(self, req: ResetReq, context) -> ResetRsp:
        del req
        return ResetRsp(msg="reset")

    def attach_obj(self, req: AttachReq, context) -> AttachRsp:
        del req
        return AttachRsp(msg="attached")

    def detach_obj(self, req: DetachReq, context) -> DetachRsp:
        del req
        return DetachRsp(msg="detached")

    def task_status(self, req: TaskStatusReq, context) -> TaskStatusRsp:
        del req
        return TaskStatusRsp(msg="ok")

    def exit(self, req: ExitReq, context) -> ExitRsp:
        del req
        return ExitRsp(msg="exiting")

    def init_robot(self, req: InitRobotReq, context) -> InitRobotRsp:
        del req
        return InitRobotRsp(msg="initialized")

    def add_camera(self, req: AddCameraReq, context) -> AddCameraRsp:
        del req
        return AddCameraRsp(msg="camera added")

    def set_object_pose(self, req: SetObjectPoseReq, context) -> SetObjectPoseRsp:
        del req
        return SetObjectPoseRsp(msg="object updated")

    def set_trajectory_list(self, req: SetTrajectoryListReq, context) -> SetTrajectoryListRsp:
        del req
        return SetTrajectoryListRsp(msg="trajectory queued")

    def set_frame_state(self, req: SetFrameStateReq, context) -> SetFrameStateRsp:
        del req
        return SetFrameStateRsp(msg="frame state set")

    def set_task_metric(self, req: SetTaskMetricReq, context) -> SetTaskMetricRsp:
        del req
        return SetTaskMetricRsp(msg="metric set")

    def set_material(self, req: SetMaterailReq, context) -> SetMaterialRsp:
        del req
        return SetMaterialRsp(msg="material set")

    def set_light(self, req: SetLightReq, context) -> SetLightRsp:
        del req
        return SetLightRsp(msg="light set")

    def remove_objs_from_obstacle(self, req: RemoveObstacleReq, context) -> RemoveObstacleRsp:
        del req
        return RemoveObstacleRsp(msg="removed")

    def store_current_state(self, req: StoreCurrentStateReq, context) -> StoreCurrentStateRsp:
        del req
        return StoreCurrentStateRsp(msg="stored")

    def playback(self, req: PlaybackReq, context) -> PlaybackRsp:
        del req
        return PlaybackRsp(msg="playback started")

    def get_checker_status(self, req: GetCheckerStatusReq, context) -> GetCheckerStatusRsp:
        checker = req.checker or "status"
        return GetCheckerStatusRsp(msg=f"{checker}: ok")


class MockJointControlServicer(JointControlServiceServicer):
    """Mock joint control service returning zero joint positions."""

    def get_joint_position(self, request, context):
        rsp = GetJointRsp()
        # Return 7 zero-position joints (typical robot arm)
        for i in range(7):
            rsp.states.append(JointState(name=f"joint_{i}", position=0.0))
        return rsp

    def set_joint_position(self, request, context):
        return SetJointRsp(errmsg="ok")

    def get_ik_status(self, request, context):
        return GetIKStatusRsp(isSuccess=True)

    def get_ee_pose(self, request, context):
        return GetEEPoseRsp()


def _parse_port() -> int:
    return parse_int_env(
        os.getenv(GENIESIM_PORT_ENV),
        default=DEFAULT_GENIESIM_PORT,
        min_value=1,
        max_value=65535,
        name=GENIESIM_PORT_ENV,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local GenieSim gRPC mock server.")
    parser.add_argument("--port", type=int, default=_parse_port())
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    init_logging()
    server = grpc.server(ThreadPoolExecutor(max_workers=8))
    add_SimObservationServiceServicer_to_server(GenieSimLocalServicer(), server)
    add_JointControlServiceServicer_to_server(MockJointControlServicer(), server)

    bind_addr = f"{args.host}:{args.port}"
    credentials = _load_tls_credentials()
    if credentials:
        server.add_secure_port(bind_addr, credentials)
    else:
        server.add_insecure_port(bind_addr)

    LOGGER.info("Starting GenieSim mock server on port %s", args.port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
