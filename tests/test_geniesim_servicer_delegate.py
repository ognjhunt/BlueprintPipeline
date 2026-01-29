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


def test_servicer_base_class_exists() -> None:
    """Verify that SimObservationServiceServicer is importable and instantiable."""
    servicer = geniesim_pb2_grpc.SimObservationServiceServicer()
    assert servicer is not None


def test_servicer_has_expected_methods() -> None:
    """Verify the base servicer exposes the expected RPC method stubs."""
    servicer = geniesim_pb2_grpc.SimObservationServiceServicer()
    expected_methods = [
        "get_observation",
        "reset",
        "attach_obj",
        "detach_obj",
        "task_status",
        "exit",
        "init_robot",
        "add_camera",
        "set_object_pose",
        "set_trajectory_list",
        "set_frame_state",
        "set_task_metric",
        "set_material",
        "set_light",
        "remove_objs_from_obstacle",
        "store_current_state",
        "playback",
        "get_checker_status",
    ]
    for method_name in expected_methods:
        assert hasattr(servicer, method_name), f"Missing method: {method_name}"
