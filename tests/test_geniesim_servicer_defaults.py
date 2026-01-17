from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

for path in (REPO_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.geniesim_adapter import geniesim_grpc_pb2 as geniesim_pb2
from tools.geniesim_adapter import geniesim_grpc_pb2_grpc as geniesim_pb2_grpc


@dataclass
class DummyContext:
    code: object | None = None
    details: str | None = None

    def set_code(self, code: object) -> None:
        self.code = code

    def set_details(self, details: str) -> None:
        self.details = details


def test_default_servicer_handles_core_methods(tmp_path: Path) -> None:
    servicer = geniesim_pb2_grpc.GenieSimServiceServicer()
    context = DummyContext()

    observation = servicer.GetObservation(geniesim_pb2.GetObservationRequest(), context)
    assert observation.success is True

    joint_response = servicer.SetJointPosition(
        geniesim_pb2.SetJointPositionRequest(positions=[0.0, 0.1, 0.2]),
        context,
    )
    assert joint_response.success is True

    joint_state = servicer.GetJointPosition(geniesim_pb2.GetJointPositionRequest(), context)
    assert joint_state.success is True

    ee_pose = servicer.GetEEPose(geniesim_pb2.GetEEPoseRequest(), context)
    assert ee_pose.success is True

    gripper = servicer.SetGripperState(
        geniesim_pb2.SetGripperStateRequest(width=0.05, force=5.0),
        context,
    )
    assert gripper.success is True

    object_pose = geniesim_pb2.Pose(
        position=geniesim_pb2.Vector3(x=1.0, y=2.0, z=3.0),
        orientation=geniesim_pb2.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
    )
    set_object = servicer.SetObjectPose(
        geniesim_pb2.SetObjectPoseRequest(object_id="cube", pose=object_pose),
        context,
    )
    assert set_object.success is True

    get_object = servicer.GetObjectPose(
        geniesim_pb2.GetObjectPoseRequest(object_id="cube"),
        context,
    )
    assert get_object.success is True

    trajectory = servicer.SetTrajectory(
        geniesim_pb2.SetTrajectoryRequest(
            points=[geniesim_pb2.TrajectoryPoint(positions=[0.3, 0.4, 0.5], time_from_start=1.0)]
        ),
        context,
    )
    assert trajectory.success is True

    recording = servicer.StartRecording(
        geniesim_pb2.StartRecordingRequest(output_directory=str(tmp_path)),
        context,
    )
    assert recording.success is True

    stopped = servicer.StopRecording(geniesim_pb2.StopRecordingRequest(), context)
    assert stopped.success is True

    reset = servicer.Reset(geniesim_pb2.ResetRequest(), context)
    assert reset.success is True

    command = servicer.SendCommand(
        geniesim_pb2.CommandRequest(command_type=geniesim_pb2.CommandType.GET_OBSERVATION),
        context,
    )
    assert command.success is True
