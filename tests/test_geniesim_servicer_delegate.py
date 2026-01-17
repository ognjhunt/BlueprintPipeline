from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


class Delegate:
    def __init__(self) -> None:
        self.called = False

    def GetObservation(
        self,
        request: geniesim_pb2.GetObservationRequest,
        context: DummyContext,
    ) -> geniesim_pb2.GetObservationResponse:
        self.called = True
        return geniesim_pb2.GetObservationResponse(success=True, timestamp=123.0)

    def GetCameraData(self, *_args, **_kwargs) -> None:
        return None

    def GetSemanticData(self, *_args, **_kwargs) -> None:
        return None

    def LinearMove(self, *_args, **_kwargs) -> None:
        return None


def test_servicer_uses_delegate_for_get_observation() -> None:
    delegate = Delegate()
    servicer = geniesim_pb2_grpc.GenieSimServiceServicer(delegate=delegate)
    context = DummyContext()

    response = servicer.GetObservation(geniesim_pb2.GetObservationRequest(), context)

    assert delegate.called is True
    assert response.success is True
