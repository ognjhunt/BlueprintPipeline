"""Shim module so aimdk.protocol.geniesim_grpc_pb2_grpc resolves to repo stubs."""
from importlib import import_module as _import_module

_root = _import_module("geniesim_grpc_pb2_grpc")
__all__ = [name for name in dir(_root) if not name.startswith("_")]
globals().update({name: getattr(_root, name) for name in __all__})
