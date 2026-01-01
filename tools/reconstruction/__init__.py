"""3D Reconstruction Backends.

Alternative reconstruction backends to 3D-RE-GEN.
Supports:
- Manual CAD upload (GLB/GLTF/USD files)
- MASt3R depth-based reconstruction
- InstantMesh single-image reconstruction
"""

from .base import (
    ReconstructionBackend,
    ReconstructionResult,
    ReconstructedObject,
)
from .manual_cad import ManualCADBackend
from .mast3r_backend import MASt3RBackend
from .reconstruction_manager import ReconstructionManager

__all__ = [
    "ReconstructionBackend",
    "ReconstructionResult",
    "ReconstructedObject",
    "ManualCADBackend",
    "MASt3RBackend",
    "ReconstructionManager",
]
