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
    aimdk_common = proto_root / "aimdk" / "protocol" / "common"
    aimdk_hal_joint = proto_root / "aimdk" / "protocol" / "hal" / "joint"

    # Collect all proto files that need compilation
    proto_paths: list[Path] = [proto_root / "geniesim_grpc.proto"]
    for proto_dir in (aimdk_common, aimdk_hal_joint):
        proto_paths.extend(sorted(proto_dir.glob("*.proto")))

    for required in proto_paths:
        if not required.exists():
            raise SystemExit(f"Proto file not found: {required}")

    _run_protoc(proto_root, proto_paths)

    expected_files = [
        proto_root / "geniesim_grpc_pb2.py",
        proto_root / "geniesim_grpc_pb2_grpc.py",
        aimdk_common / "se3_pose_pb2.py",
        aimdk_common / "joint_pb2.py",
        aimdk_common / "vec3_pb2.py",
        aimdk_hal_joint / "joint_channel_pb2.py",
        aimdk_hal_joint / "joint_channel_pb2_grpc.py",
    ]
    missing = [path for path in expected_files if not path.exists()]
    if missing:
        missing_display = ", ".join(str(path) for path in missing)
        raise SystemExit(f"Stub generation completed but files missing: {missing_display}")

    print("Generated Genie Sim gRPC stubs.")


if __name__ == "__main__":
    main()
