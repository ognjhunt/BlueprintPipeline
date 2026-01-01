"""Base Classes for Reconstruction Backends.

Defines the interface that all reconstruction backends must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ReconstructionBackendType(str, Enum):
    """Available reconstruction backends."""
    REGEN3D = "3d-re-gen"  # Original (not yet released)
    MANUAL_CAD = "manual-cad"  # Manual GLB upload
    MAST3R = "mast3r"  # MASt3R depth-based
    INSTANTMESH = "instantmesh"  # Single-image reconstruction
    DUST3R = "dust3r"  # DUSt3R depth-based


@dataclass
class ReconstructedObject:
    """A single reconstructed 3D object."""
    id: str
    mesh_path: Path  # Path to GLB/GLTF file
    category: str  # Object category (chair, table, mug, etc.)

    # Transform (4x4 matrix or decomposed)
    transform_matrix: Optional[np.ndarray] = None
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Bounds (axis-aligned bounding box)
    bounds_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Material properties
    base_color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    metallic: float = 0.0
    roughness: float = 0.5

    # Metadata
    is_floor_contact: bool = False
    sim_role: str = "unknown"  # static, interactive, manipulable_object, etc.
    source_backend: str = "unknown"
    confidence: float = 1.0

    # Optional reference image
    reference_image: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for manifest."""
        return {
            "id": self.id,
            "category": self.category,
            "transform": {
                "position": {"x": self.position[0], "y": self.position[1], "z": self.position[2]},
                "rotation_quaternion": {
                    "w": self.rotation_quaternion[0],
                    "x": self.rotation_quaternion[1],
                    "y": self.rotation_quaternion[2],
                    "z": self.rotation_quaternion[3],
                },
                "scale": {"x": self.scale[0], "y": self.scale[1], "z": self.scale[2]},
            },
            "asset": {
                "path": str(self.mesh_path),
                "format": self.mesh_path.suffix.lstrip(".").lower(),
                "source": self.source_backend,
            },
            "dimensions_est": {
                "width": self.bounds_max[0] - self.bounds_min[0],
                "height": self.bounds_max[1] - self.bounds_min[1],
                "depth": self.bounds_max[2] - self.bounds_min[2],
            },
            "physics_hints": {
                "is_floor_contact": self.is_floor_contact,
                "material_type": "default",
                "roughness": self.roughness,
                "metallic": self.metallic,
            },
            "sim_role": self.sim_role,
            "confidence": self.confidence,
        }


@dataclass
class ReconstructionResult:
    """Result of a reconstruction operation."""
    success: bool
    scene_id: str
    backend: ReconstructionBackendType
    objects: List[ReconstructedObject] = field(default_factory=list)
    background_mesh: Optional[Path] = None
    error: Optional[str] = None

    # Scene metadata
    coordinate_frame: str = "y_up"
    meters_per_unit: float = 1.0
    environment_type: str = "generic"

    # Camera info (if available)
    camera_intrinsics: Optional[np.ndarray] = None
    camera_extrinsics: Optional[np.ndarray] = None

    # Processing stats
    processing_time_seconds: float = 0.0

    def to_manifest(self) -> Dict[str, Any]:
        """Convert to scene manifest format."""
        return {
            "version": "1.0.0",
            "scene_id": self.scene_id,
            "scene": {
                "coordinate_frame": self.coordinate_frame,
                "meters_per_unit": self.meters_per_unit,
                "environment_type": self.environment_type,
                "physics_defaults": {
                    "gravity": [0, -9.81, 0],
                    "solver": "TGS",
                    "time_steps_per_second": 60,
                },
            },
            "objects": [obj.to_dict() for obj in self.objects],
            "metadata": {
                "reconstruction_backend": self.backend.value,
                "processing_time_seconds": self.processing_time_seconds,
                "object_count": len(self.objects),
            },
        }


class ReconstructionBackend(ABC):
    """Abstract base class for reconstruction backends."""

    backend_type: ReconstructionBackendType

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.backend_type.value.upper()}] {msg}")

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available (dependencies installed, etc.)."""
        pass

    @abstractmethod
    def reconstruct(
        self,
        input_path: Path,
        output_dir: Path,
        scene_id: str,
        **kwargs
    ) -> ReconstructionResult:
        """Run reconstruction on input.

        Args:
            input_path: Path to input (image, directory, or manifest)
            output_dir: Directory to write outputs
            scene_id: Scene identifier
            **kwargs: Backend-specific options

        Returns:
            ReconstructionResult with reconstructed objects
        """
        pass

    @abstractmethod
    def get_requirements(self) -> Dict[str, str]:
        """Get backend requirements (packages, models, etc.)."""
        pass

    def validate_input(self, input_path: Path) -> bool:
        """Validate input is appropriate for this backend."""
        return input_path.exists()
