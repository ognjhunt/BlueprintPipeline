from pathlib import Path

import pytest

from tools.geniesim_adapter import regenerate_stubs


def _write_proto(tmp_path: Path) -> Path:
    proto_path = tmp_path / "geniesim_grpc.proto"
    proto_path.write_text("syntax = 'proto3';\n")
    return proto_path


def test_regenerate_stubs_missing_proto(monkeypatch, tmp_path):
    monkeypatch.setattr(regenerate_stubs, "_ensure_grpc_tools", lambda: None)
    monkeypatch.setattr(regenerate_stubs, "_run_protoc", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(regenerate_stubs, "Path", lambda *_args, **_kwargs: tmp_path / "missing.proto")

    with pytest.raises(SystemExit, match="Proto file not found"):
        regenerate_stubs.main()


def test_regenerate_stubs_run_failure(monkeypatch, tmp_path):
    proto_path = _write_proto(tmp_path)

    monkeypatch.setattr(regenerate_stubs, "_ensure_grpc_tools", lambda: None)
    monkeypatch.setattr(regenerate_stubs, "Path", lambda *_args, **_kwargs: proto_path)
    monkeypatch.setattr(regenerate_stubs.subprocess, "run", lambda *_args, **_kwargs: type("Result", (), {"returncode": 1})())

    with pytest.raises(SystemExit, match="Failed to generate gRPC stubs"):
        regenerate_stubs.main()


def test_regenerate_stubs_missing_outputs(monkeypatch, tmp_path):
    proto_path = _write_proto(tmp_path)

    monkeypatch.setattr(regenerate_stubs, "_ensure_grpc_tools", lambda: None)
    monkeypatch.setattr(regenerate_stubs, "Path", lambda *_args, **_kwargs: proto_path)
    monkeypatch.setattr(regenerate_stubs.subprocess, "run", lambda *_args, **_kwargs: type("Result", (), {"returncode": 0})())

    with pytest.raises(SystemExit, match="Stub generation completed but files missing"):
        regenerate_stubs.main()


def test_regenerate_stubs_success(monkeypatch, tmp_path, capsys):
    proto_path = _write_proto(tmp_path)
    (tmp_path / "geniesim_grpc_pb2.py").write_text("# stub")
    (tmp_path / "geniesim_grpc_pb2_grpc.py").write_text("# stub")

    monkeypatch.setattr(regenerate_stubs, "_ensure_grpc_tools", lambda: None)
    monkeypatch.setattr(regenerate_stubs, "Path", lambda *_args, **_kwargs: proto_path)
    monkeypatch.setattr(regenerate_stubs.subprocess, "run", lambda *_args, **_kwargs: type("Result", (), {"returncode": 0})())

    regenerate_stubs.main()

    assert "Generated Genie Sim gRPC stubs." in capsys.readouterr().out


def test_ensure_grpc_tools_import_error(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def _raise_import_error(name, *args, **kwargs):
        if name.startswith("grpc_tools"):
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raise_import_error)

    with pytest.raises(SystemExit, match="grpcio-tools is required"):
        regenerate_stubs._ensure_grpc_tools()
