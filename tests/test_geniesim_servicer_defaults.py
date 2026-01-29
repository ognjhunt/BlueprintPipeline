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
    servicer = geniesim_pb2_grpc.SimObservationServiceServicer()
    context = DummyContext()

    # get_observation sets UNIMPLEMENTED and raises NotImplementedError
    with pytest.raises(NotImplementedError):
        servicer.get_observation(geniesim_pb2.GetObservationReq(), context)

    with pytest.raises(NotImplementedError):
        servicer.reset(geniesim_pb2.ResetReq(), context)

    with pytest.raises(NotImplementedError):
        servicer.set_object_pose(geniesim_pb2.SetObjectPoseReq(), context)

    with pytest.raises(NotImplementedError):
        servicer.set_trajectory_list(geniesim_pb2.SetTrajectoryListReq(), context)

    with pytest.raises(NotImplementedError):
        servicer.init_robot(geniesim_pb2.InitRobotReq(), context)

    with pytest.raises(NotImplementedError):
        servicer.add_camera(geniesim_pb2.AddCameraReq(), context)

    with pytest.raises(NotImplementedError):
        servicer.store_current_state(geniesim_pb2.StoreCurrentStateReq(), context)

    with pytest.raises(NotImplementedError):
        servicer.playback(geniesim_pb2.PlaybackReq(), context)

    with pytest.raises(NotImplementedError):
        servicer.get_checker_status(geniesim_pb2.GetCheckerStatusReq(), context)

    # All calls should have set UNIMPLEMENTED
    assert context.code is not None
    assert context.details == "Method not implemented!"
