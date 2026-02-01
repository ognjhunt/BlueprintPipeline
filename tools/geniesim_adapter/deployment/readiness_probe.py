#!/usr/bin/env python3
"""Lightweight gRPC readiness probe for the Genie Sim server.

Exit codes:
  0 — server is ready (gRPC channel connected)
  1 — server is not ready
"""
import os
import sys

try:
    import grpc
except ImportError:
    sys.exit(1)

host = os.environ.get("GENIESIM_HOST", "localhost")
port = os.environ.get("GENIESIM_PORT", "50051")
channel = grpc.insecure_channel(f"{host}:{port}")
try:
    grpc.channel_ready_future(channel).result(timeout=5)
except grpc.FutureTimeoutError:
    sys.exit(1)
finally:
    channel.close()
