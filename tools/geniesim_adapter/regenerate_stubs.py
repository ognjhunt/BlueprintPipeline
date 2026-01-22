#!/usr/bin/env python3
"""Regenerate Genie Sim gRPC Python stubs from geniesim_grpc.proto."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _ensure_grpc_tools() -> None:
    try:
        import grpc_tools.protoc  # noqa: F401
    except ImportError as exc:
        message = (
            "grpcio-tools is required to regenerate Genie Sim gRPC stubs. "
            "Install dependencies with: pip install grpcio grpcio-tools"
        )
        raise SystemExit(message) from exc


def _run_protoc(proto_root: Path, proto_paths: list[Path]) -> None:
    command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_root}",
        f"--python_out={proto_root}",
        f"--grpc_python_out={proto_root}",
        *[str(path) for path in proto_paths],
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(
            "Failed to generate gRPC stubs. Ensure grpcio-tools is installed "
            "and the proto files are valid."
        )


def main() -> None:
    _ensure_grpc_tools()
    proto_root = Path(__file__).resolve().parent
    proto_path = proto_root / "geniesim_grpc.proto"
    joint_proto = proto_root / "aimdk" / "protocol" / "hal" / "joint" / "joint_channel.proto"
    for required in (proto_path, joint_proto):
        if not required.exists():
            raise SystemExit(f"Proto file not found: {required}")

    _run_protoc(proto_root, [proto_path, joint_proto])

    expected_files = [
        proto_root / "geniesim_grpc_pb2.py",
        proto_root / "geniesim_grpc_pb2_grpc.py",
        proto_root / "aimdk" / "protocol" / "hal" / "joint" / "joint_channel_pb2.py",
        proto_root / "aimdk" / "protocol" / "hal" / "joint" / "joint_channel_pb2_grpc.py",
    ]
    missing = [path for path in expected_files if not path.exists()]
    if missing:
        missing_display = ", ".join(str(path) for path in missing)
        raise SystemExit(f"Stub generation completed but files missing: {missing_display}")

    print("Generated Genie Sim gRPC stubs.")


if __name__ == "__main__":
    main()
