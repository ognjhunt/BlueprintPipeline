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


def _install_grpc_stub() -> types.ModuleType:
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


def _ensure_grpc_module() -> types.ModuleType:
    try:
        import grpc  # type: ignore

        return grpc
    except Exception:
        return _install_grpc_stub()


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


def _clear_geniesim_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "GENIESIM_HOST",
        "GENIESIM_PORT",
        "GENIESIM_TLS_CERT",
        "GENIESIM_TLS_KEY",
        "GENIESIM_TLS_CA",
        "GENIESIM_AUTH_TOKEN",
        "GENIESIM_AUTH_TOKEN_PATH",
        "GENIESIM_AUTH_CERT",
        "GENIESIM_AUTH_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_create_channel_insecure(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_geniesim_env(monkeypatch)
    monkeypatch.setenv("GENIESIM_HOST", "example.com")
    monkeypatch.setenv("GENIESIM_PORT", "1234")

    calls: dict[str, object] = {}

    def fake_insecure_channel(target, options=None):
        calls["target"] = target
        calls["options"] = options
        return "insecure"

    def fake_secure_channel(*args, **kwargs):
        calls["secure"] = True
        return "secure"

    def fake_intercept_channel(channel, *interceptors):
        calls["intercepted"] = True
        return channel

    monkeypatch.setattr(grpc_module.grpc, "insecure_channel", fake_insecure_channel)
    monkeypatch.setattr(grpc_module.grpc, "secure_channel", fake_secure_channel)
    monkeypatch.setattr(grpc_module.grpc, "intercept_channel", fake_intercept_channel)

    channel = grpc_module.create_channel()

    assert channel == "insecure"
    assert calls["target"] == "example.com:1234"
    assert "secure" not in calls
    assert "intercepted" not in calls


def test_create_channel_tls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_geniesim_env(monkeypatch)
    cert_path = tmp_path / "client.pem"
    key_path = tmp_path / "client.key"
    ca_path = tmp_path / "ca.pem"
    cert_path.write_text("CERT")
    key_path.write_text("KEY")
    ca_path.write_text("CA")

    monkeypatch.setenv("GENIESIM_TLS_CERT", str(cert_path))
    monkeypatch.setenv("GENIESIM_TLS_KEY", str(key_path))
    monkeypatch.setenv("GENIESIM_TLS_CA", str(ca_path))

    calls: dict[str, object] = {}

    def fake_ssl_channel_credentials(root_certificates=None, private_key=None, certificate_chain=None):
        calls["root"] = root_certificates
        calls["key"] = private_key
        calls["cert"] = certificate_chain
        return "creds"

    def fake_secure_channel(target, credentials, options=None):
        calls["target"] = target
        calls["creds"] = credentials
        return "secure"

    def fake_intercept_channel(channel, *interceptors):
        calls["intercepted"] = True
        return channel

    monkeypatch.setattr(grpc_module.grpc, "ssl_channel_credentials", fake_ssl_channel_credentials)
    monkeypatch.setattr(grpc_module.grpc, "secure_channel", fake_secure_channel)
    monkeypatch.setattr(grpc_module.grpc, "intercept_channel", fake_intercept_channel)

    channel = grpc_module.create_channel(host="secure.example.com", port=443)

    assert channel == "secure"
    assert calls["target"] == "secure.example.com:443"
    assert calls["creds"] == "creds"
    assert calls["root"] == b"CA"
    assert calls["key"] == b"KEY"
    assert calls["cert"] == b"CERT"
    assert "intercepted" not in calls


def test_create_channel_auth_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_geniesim_env(monkeypatch)
    monkeypatch.setenv("GENIESIM_AUTH_TOKEN", "token-value")

    calls: dict[str, object] = {}

    def fake_insecure_channel(*args, **kwargs):
        return "insecure"

    def fake_intercept_channel(channel, *interceptors):
        calls["interceptors"] = interceptors
        return "intercepted"

    monkeypatch.setattr(grpc_module.grpc, "insecure_channel", fake_insecure_channel)
    monkeypatch.setattr(grpc_module.grpc, "intercept_channel", fake_intercept_channel)

    channel = grpc_module.create_channel(host="auth.example.com", port=5000)

    assert channel == "intercepted"
    assert calls["interceptors"]
    interceptor = calls["interceptors"][0]
    assert ("authorization", "Bearer token-value") in interceptor._metadata
