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


def _run_protoc(proto_path: Path) -> None:
    output_dir = proto_path.parent
    command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{output_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        str(proto_path),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(
            "Failed to generate gRPC stubs. Ensure grpcio-tools is installed "
            "and the proto file is valid."
        )


def main() -> None:
    _ensure_grpc_tools()
    proto_path = Path(__file__).with_name("geniesim_grpc.proto")
    if not proto_path.exists():
        raise SystemExit(f"Proto file not found: {proto_path}")

    _run_protoc(proto_path)

    expected_files = [
        proto_path.with_name("geniesim_grpc_pb2.py"),
        proto_path.with_name("geniesim_grpc_pb2_grpc.py"),
    ]
    missing = [path for path in expected_files if not path.exists()]
    if missing:
        missing_display = ", ".join(str(path) for path in missing)
        raise SystemExit(f"Stub generation completed but files missing: {missing_display}")

    print("Generated Genie Sim gRPC stubs.")


if __name__ == "__main__":
    main()
