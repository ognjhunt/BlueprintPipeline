from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

for path in (REPO_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import importlib.util
import types


def _ensure_grpc_module() -> types.ModuleType:
    try:
        import grpc  # type: ignore

        return grpc
    except Exception:
        grpc_stub = types.ModuleType("grpc")
        grpc_stub.__version__ = "1.66.1"

        def _noop(*args, **kwargs):
            return None

        grpc_stub.insecure_channel = _noop
        grpc_stub.secure_channel = _noop
        grpc_stub.intercept_channel = _noop
        grpc_stub.ssl_channel_credentials = _noop
        grpc_stub.metadata_call_credentials = _noop
        grpc_stub.composite_channel_credentials = _noop
        grpc_stub.experimental = types.SimpleNamespace(unary_unary=_noop)
        grpc_stub.StatusCode = types.SimpleNamespace(UNIMPLEMENTED="UNIMPLEMENTED")

        utilities_stub = types.ModuleType("grpc._utilities")

        def first_version_is_lower(current: str, required: str) -> bool:
            return False

        utilities_stub.first_version_is_lower = first_version_is_lower
        grpc_stub._utilities = utilities_stub

        sys.modules.setdefault("grpc", grpc_stub)
        sys.modules.setdefault("grpc._utilities", utilities_stub)
        return grpc_stub


def _load_geniesim_grpc_module() -> types.ModuleType:
    _ensure_grpc_module()

    adapter_root = REPO_ROOT / "tools" / "geniesim_adapter"
    if str(adapter_root) not in sys.path:
        sys.path.insert(0, str(adapter_root))

    tools_module = types.ModuleType("tools")
    tools_module.__path__ = [str(REPO_ROOT / "tools")]
    sys.modules.setdefault("tools", tools_module)

    adapter_module = types.ModuleType("tools.geniesim_adapter")
    adapter_module.__path__ = [str(REPO_ROOT / "tools" / "geniesim_adapter")]
    sys.modules.setdefault("tools.geniesim_adapter", adapter_module)

    module_name = "tools.geniesim_adapter.geniesim_grpc_pb2_grpc"
    module_path = REPO_ROOT / "tools" / "geniesim_adapter" / "geniesim_grpc_pb2_grpc.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load geniesim_grpc_pb2_grpc module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


grpc_module = _load_geniesim_grpc_module()


def test_grpc_module_has_stub_class() -> None:
    assert hasattr(grpc_module, "SimObservationServiceStub")


def test_grpc_module_has_servicer_class() -> None:
    assert hasattr(grpc_module, "SimObservationServiceServicer")


def test_grpc_module_has_add_to_server() -> None:
    assert hasattr(grpc_module, "add_SimObservationServiceServicer_to_server")
