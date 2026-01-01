"""Manual CAD Upload Backend.

Allows users to upload their own GLB/GLTF/USD files directly,
bypassing automatic 3D reconstruction.

This is the simplest and most reliable path to getting scenes
into the pipeline for testing and validation.

Expected Input Structure:
    input_dir/
    ├── manifest.json           # Scene configuration
    └── objects/
        ├── obj_0/
        │   ├── mesh.glb       # 3D mesh
        │   └── metadata.json  # Optional: position, category, etc.
        └── obj_1/
            └── mesh.glb

Or simpler:
    input_dir/
    ├── manifest.json
    └── *.glb                   # All GLB files in root
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    ReconstructionBackend,
    ReconstructionBackendType,
    ReconstructionResult,
    ReconstructedObject,
)


@dataclass
class CADManifestObject:
    """Object definition from manifest."""
    id: str
    mesh_file: str  # Relative path to mesh
    category: str = "unknown"
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # quaternion wxyz
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    is_floor_contact: bool = False
    sim_role: str = "unknown"


class ManualCADBackend(ReconstructionBackend):
    """Backend for manually uploaded CAD files."""

    backend_type = ReconstructionBackendType.MANUAL_CAD

    # Supported mesh formats
    SUPPORTED_FORMATS = {".glb", ".gltf", ".usd", ".usda", ".usdz", ".obj", ".fbx"}

    def __init__(
        self,
        auto_detect_floor_contact: bool = True,
        auto_compute_bounds: bool = True,
        verbose: bool = True,
    ):
        super().__init__(verbose=verbose)
        self.auto_detect_floor_contact = auto_detect_floor_contact
        self.auto_compute_bounds = auto_compute_bounds

    def is_available(self) -> bool:
        """Always available - no external dependencies."""
        return True

    def get_requirements(self) -> Dict[str, str]:
        """No external requirements."""
        return {
            "description": "Manual CAD upload - no external dependencies",
            "supported_formats": ", ".join(self.SUPPORTED_FORMATS),
        }

    def validate_input(self, input_path: Path) -> bool:
        """Check input is a directory with meshes or manifest."""
        if not input_path.exists():
            return False

        if input_path.is_file():
            # Single file - must be supported format
            return input_path.suffix.lower() in self.SUPPORTED_FORMATS

        if input_path.is_dir():
            # Directory - must have manifest or mesh files
            has_manifest = (input_path / "manifest.json").is_file()
            has_meshes = any(
                f.suffix.lower() in self.SUPPORTED_FORMATS
                for f in input_path.rglob("*")
            )
            return has_manifest or has_meshes

        return False

    def reconstruct(
        self,
        input_path: Path,
        output_dir: Path,
        scene_id: str,
        environment_type: str = "generic",
        **kwargs
    ) -> ReconstructionResult:
        """Process uploaded CAD files.

        Args:
            input_path: Directory with mesh files and optional manifest
            output_dir: Where to write processed outputs
            scene_id: Scene identifier
            environment_type: Type of environment (kitchen, warehouse, etc.)
            **kwargs: Additional options

        Returns:
            ReconstructionResult with processed objects
        """
        start_time = time.time()
        self.log(f"Processing CAD files from: {input_path}")

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output structure
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(exist_ok=True)

        try:
            # Load manifest or discover files
            manifest_path = input_path / "manifest.json"
            if manifest_path.is_file():
                objects = self._load_from_manifest(input_path, manifest_path)
            else:
                objects = self._discover_meshes(input_path)

            if not objects:
                return ReconstructionResult(
                    success=False,
                    scene_id=scene_id,
                    backend=self.backend_type,
                    error="No mesh files found in input",
                )

            # Process each object
            reconstructed_objects = []
            for i, obj in enumerate(objects):
                try:
                    processed = self._process_object(
                        obj, input_path, assets_dir, i
                    )
                    reconstructed_objects.append(processed)
                    self.log(f"  Processed: {processed.id} ({processed.category})")
                except Exception as e:
                    self.log(f"  Error processing {obj.id}: {e}")

            processing_time = time.time() - start_time

            result = ReconstructionResult(
                success=True,
                scene_id=scene_id,
                backend=self.backend_type,
                objects=reconstructed_objects,
                environment_type=environment_type,
                processing_time_seconds=processing_time,
            )

            # Save manifest
            manifest_output = output_dir / "scene_manifest.json"
            manifest_output.write_text(json.dumps(result.to_manifest(), indent=2))
            self.log(f"Saved manifest: {manifest_output}")

            self.log(f"Processed {len(reconstructed_objects)} objects in {processing_time:.1f}s")

            return result

        except Exception as e:
            return ReconstructionResult(
                success=False,
                scene_id=scene_id,
                backend=self.backend_type,
                error=str(e),
            )

    def _load_from_manifest(
        self, input_dir: Path, manifest_path: Path
    ) -> List[CADManifestObject]:
        """Load objects from manifest.json."""
        manifest = json.loads(manifest_path.read_text())
        objects = []

        for obj_data in manifest.get("objects", []):
            # Parse position
            pos = obj_data.get("position", {})
            if isinstance(pos, dict):
                position = (pos.get("x", 0), pos.get("y", 0), pos.get("z", 0))
            elif isinstance(pos, list):
                position = tuple(pos[:3]) if len(pos) >= 3 else (0, 0, 0)
            else:
                position = (0, 0, 0)

            # Parse rotation
            rot = obj_data.get("rotation", obj_data.get("rotation_quaternion", {}))
            if isinstance(rot, dict):
                rotation = (rot.get("w", 1), rot.get("x", 0), rot.get("y", 0), rot.get("z", 0))
            elif isinstance(rot, list):
                rotation = tuple(rot[:4]) if len(rot) >= 4 else (1, 0, 0, 0)
            else:
                rotation = (1, 0, 0, 0)

            # Parse scale
            scale = obj_data.get("scale", {})
            if isinstance(scale, dict):
                scale = (scale.get("x", 1), scale.get("y", 1), scale.get("z", 1))
            elif isinstance(scale, (int, float)):
                scale = (scale, scale, scale)
            elif isinstance(scale, list):
                scale = tuple(scale[:3]) if len(scale) >= 3 else (1, 1, 1)
            else:
                scale = (1, 1, 1)

            objects.append(CADManifestObject(
                id=obj_data.get("id", f"obj_{len(objects)}"),
                mesh_file=obj_data.get("mesh_file", obj_data.get("mesh", "")),
                category=obj_data.get("category", "unknown"),
                position=position,
                rotation=rotation,
                scale=scale,
                is_floor_contact=obj_data.get("is_floor_contact", False),
                sim_role=obj_data.get("sim_role", self._infer_sim_role(obj_data.get("category", ""))),
            ))

        return objects

    def _discover_meshes(self, input_dir: Path) -> List[CADManifestObject]:
        """Discover mesh files in directory."""
        objects = []

        # First check for structured directories (objects/obj_X/)
        objects_dir = input_dir / "objects"
        if objects_dir.is_dir():
            for obj_subdir in sorted(objects_dir.iterdir()):
                if obj_subdir.is_dir():
                    mesh_files = [
                        f for f in obj_subdir.iterdir()
                        if f.suffix.lower() in self.SUPPORTED_FORMATS
                    ]
                    if mesh_files:
                        mesh_file = mesh_files[0]
                        # Load metadata if available
                        metadata_file = obj_subdir / "metadata.json"
                        if metadata_file.is_file():
                            metadata = json.loads(metadata_file.read_text())
                        else:
                            metadata = {}

                        objects.append(CADManifestObject(
                            id=obj_subdir.name,
                            mesh_file=str(mesh_file.relative_to(input_dir)),
                            category=metadata.get("category", self._infer_category(mesh_file.stem)),
                            is_floor_contact=metadata.get("is_floor_contact", False),
                            sim_role=metadata.get("sim_role", "unknown"),
                        ))

        # Fallback: find meshes in root directory
        if not objects:
            for mesh_file in sorted(input_dir.iterdir()):
                if mesh_file.suffix.lower() in self.SUPPORTED_FORMATS:
                    objects.append(CADManifestObject(
                        id=mesh_file.stem,
                        mesh_file=mesh_file.name,
                        category=self._infer_category(mesh_file.stem),
                    ))

        return objects

    def _process_object(
        self,
        obj: CADManifestObject,
        input_dir: Path,
        assets_dir: Path,
        index: int,
    ) -> ReconstructedObject:
        """Process a single object."""
        # Resolve mesh path
        mesh_path = input_dir / obj.mesh_file
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        # Copy mesh to output
        obj_dir = assets_dir / obj.id
        obj_dir.mkdir(exist_ok=True)
        output_mesh = obj_dir / f"mesh{mesh_path.suffix.lower()}"
        shutil.copy2(mesh_path, output_mesh)

        # Compute bounds (if possible)
        bounds = self._compute_bounds(output_mesh)

        # Infer floor contact from position and bounds
        is_floor_contact = obj.is_floor_contact
        if self.auto_detect_floor_contact and not is_floor_contact:
            # If object's bottom is near y=0, assume floor contact
            bottom_y = obj.position[1] + bounds[0][1] * obj.scale[1]
            is_floor_contact = abs(bottom_y) < 0.05

        # Infer sim_role
        sim_role = obj.sim_role
        if sim_role == "unknown":
            sim_role = self._infer_sim_role(obj.category)

        return ReconstructedObject(
            id=obj.id,
            mesh_path=output_mesh,
            category=obj.category,
            position=obj.position,
            rotation_quaternion=obj.rotation,
            scale=obj.scale,
            bounds_min=bounds[0],
            bounds_max=bounds[1],
            is_floor_contact=is_floor_contact,
            sim_role=sim_role,
            source_backend=self.backend_type.value,
            confidence=1.0,  # Manual uploads have full confidence
        )

    def _compute_bounds(self, mesh_path: Path) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Compute bounding box from mesh."""
        # Try to load mesh and compute bounds
        try:
            import trimesh
            mesh = trimesh.load(str(mesh_path), force="mesh")
            if hasattr(mesh, "bounds"):
                return (tuple(mesh.bounds[0]), tuple(mesh.bounds[1]))
        except ImportError:
            pass
        except Exception:
            pass

        # Default bounds
        return ((-0.5, 0.0, -0.5), (0.5, 1.0, 0.5))

    def _infer_category(self, name: str) -> str:
        """Infer object category from filename."""
        name = name.lower()

        # Kitchen
        if any(x in name for x in ["mug", "cup"]):
            return "mug"
        if any(x in name for x in ["plate", "dish"]):
            return "plate"
        if any(x in name for x in ["bowl"]):
            return "bowl"
        if any(x in name for x in ["pot", "pan"]):
            return "cookware"
        if any(x in name for x in ["fridge", "refrigerator"]):
            return "refrigerator"
        if any(x in name for x in ["microwave"]):
            return "microwave"
        if any(x in name for x in ["cabinet", "cupboard"]):
            return "cabinet"

        # Furniture
        if any(x in name for x in ["chair"]):
            return "chair"
        if any(x in name for x in ["table", "desk"]):
            return "table"
        if any(x in name for x in ["shelf", "shelving", "bookshelf"]):
            return "shelf"
        if any(x in name for x in ["drawer"]):
            return "drawer"
        if any(x in name for x in ["door"]):
            return "door"

        # Warehouse
        if any(x in name for x in ["box", "carton"]):
            return "box"
        if any(x in name for x in ["pallet"]):
            return "pallet"
        if any(x in name for x in ["crate"]):
            return "crate"

        # Generic
        if any(x in name for x in ["floor", "ground"]):
            return "floor"
        if any(x in name for x in ["wall"]):
            return "wall"
        if any(x in name for x in ["counter", "countertop"]):
            return "countertop"

        return "object"

    def _infer_sim_role(self, category: str) -> str:
        """Infer simulation role from category."""
        category = category.lower()

        # Static (don't move)
        if category in ["floor", "wall", "ceiling", "background"]:
            return "static"

        # Articulated furniture (doors, drawers)
        if category in ["cabinet", "door", "drawer", "refrigerator", "microwave", "oven", "dishwasher"]:
            return "articulated_furniture"

        # Manipulable objects (can be picked up)
        if category in ["mug", "cup", "plate", "bowl", "bottle", "can", "box", "carton", "utensil", "tool"]:
            return "manipulable_object"

        # Interactive (can be interacted with but not picked up)
        if category in ["button", "switch", "lever", "handle", "knob"]:
            return "interactive"

        # Clutter (background objects)
        if category in ["plant", "decoration", "picture"]:
            return "clutter"

        return "unknown"


def create_example_manifest(output_path: Path) -> None:
    """Create an example manifest file for manual CAD upload."""
    example = {
        "scene_id": "my_scene",
        "environment_type": "kitchen",
        "objects": [
            {
                "id": "countertop_0",
                "mesh_file": "objects/countertop/mesh.glb",
                "category": "countertop",
                "position": {"x": 0, "y": 0, "z": 0},
                "rotation_quaternion": {"w": 1, "x": 0, "y": 0, "z": 0},
                "scale": {"x": 1, "y": 1, "z": 1},
                "is_floor_contact": True,
                "sim_role": "static"
            },
            {
                "id": "mug_0",
                "mesh_file": "objects/mug/mesh.glb",
                "category": "mug",
                "position": {"x": 0.3, "y": 0.9, "z": 0.1},
                "is_floor_contact": False,
                "sim_role": "manipulable_object"
            },
            {
                "id": "cabinet_0",
                "mesh_file": "objects/cabinet/mesh.glb",
                "category": "cabinet",
                "position": {"x": -1.0, "y": 0, "z": 0},
                "is_floor_contact": True,
                "sim_role": "articulated_furniture"
            }
        ]
    }

    output_path.write_text(json.dumps(example, indent=2))
    print(f"Example manifest written to: {output_path}")
