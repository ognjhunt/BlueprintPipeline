"""Shim module so aimdk.protocol.geniesim_grpc_pb2 resolves to repo stubs.

This lets Genie Sim server imports pick up the updated RPC definitions
from tools/geniesim_adapter/geniesim_grpc_pb2.py without requiring a
rebuild of the runtime image.
"""
from importlib import import_module as _import_module

_root = _import_module("geniesim_grpc_pb2")
__all__ = [name for name in dir(_root) if not name.startswith("_")]
globals().update({name: getattr(_root, name) for name in __all__})
