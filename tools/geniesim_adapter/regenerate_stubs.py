#!/usr/bin/env python3
"""
Regenerate gRPC stubs for Genie Sim adapter.

Usage:
    python tools/geniesim_adapter/regenerate_stubs.py
"""

import os
import subprocess
import sys
from pathlib import Path

ADAPTER_ROOT = Path(__file__).resolve().parent
PROTO_FILE = ADAPTER_ROOT / "geniesim_grpc.proto"

def regenerate_stubs():
    print(f"Regenerating gRPC stubs from {PROTO_FILE}...")

    if not PROTO_FILE.exists():
        print(f"Error: {PROTO_FILE} not found.")
        sys.exit(1)

    try:
        import grpc_tools.protoc
    except ImportError:
        print("Error: grpcio-tools is not installed. Please run 'pip install grpcio-tools'.")
        sys.exit(1)

    # Command to regenerate stubs
    # python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. geniesim_grpc.proto
    cmd = [
        sys.executable,
        "-m", "grpc_tools.protoc",
        f"-I{ADAPTER_ROOT}",
        f"--python_out={ADAPTER_ROOT}",
        f"--grpc_python_out={ADAPTER_ROOT}",
        str(PROTO_FILE)
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error regenerating stubs:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    print("✅ gRPC stubs regenerated successfully.")

    # Fix imports in generated files if needed
    # (By default, protoc might generate 'import geniesim_grpc_pb2 as ...'
    # which works if ADAPTER_ROOT is on PYTHONPATH)

if __name__ == "__main__":
    regenerate_stubs()
