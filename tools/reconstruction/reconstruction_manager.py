"""Reconstruction Manager.

Unified interface for all reconstruction backends.
Handles backend selection, fallback, and integration with the pipeline.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base import (
    ReconstructionBackend,
    ReconstructionBackendType,
    ReconstructionResult,
    ReconstructedObject,
)
from .manual_cad import ManualCADBackend
from .mast3r_backend import MASt3RBackend


class ReconstructionManager:
    """Manages reconstruction backends and pipeline integration."""

    # Registered backends in priority order
    BACKEND_CLASSES: Dict[ReconstructionBackendType, Type[ReconstructionBackend]] = {
        ReconstructionBackendType.MANUAL_CAD: ManualCADBackend,
        ReconstructionBackendType.MAST3R: MASt3RBackend,
    }

    def __init__(
        self,
        preferred_backend: Optional[ReconstructionBackendType] = None,
        fallback_enabled: bool = True,
        verbose: bool = True,
    ):
        self.preferred_backend = preferred_backend
        self.fallback_enabled = fallback_enabled
        self.verbose = verbose
        self._backends: Dict[ReconstructionBackendType, ReconstructionBackend] = {}

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[RECONSTRUCTION] {msg}")

    def get_backend(
        self, backend_type: ReconstructionBackendType
    ) -> ReconstructionBackend:
        """Get or create a backend instance."""
        if backend_type not in self._backends:
            if backend_type not in self.BACKEND_CLASSES:
                raise ValueError(f"Unknown backend type: {backend_type}")

            backend_class = self.BACKEND_CLASSES[backend_type]
            self._backends[backend_type] = backend_class(verbose=self.verbose)

        return self._backends[backend_type]

    def get_available_backends(self) -> List[ReconstructionBackendType]:
        """Get list of available backends."""
        available = []
        for backend_type in self.BACKEND_CLASSES:
            try:
                backend = self.get_backend(backend_type)
                if backend.is_available():
                    available.append(backend_type)
            except Exception:
                pass
        return available

    def auto_detect_backend(self, input_path: Path) -> Optional[ReconstructionBackendType]:
        """Auto-detect the best backend for given input."""
        input_path = Path(input_path)

        # Check for manual CAD structure
        if (input_path / "manifest.json").is_file():
            return ReconstructionBackendType.MANUAL_CAD

        # Check for GLB files
        glb_files = list(input_path.glob("*.glb")) + list(input_path.rglob("*/mesh.glb"))
        if glb_files:
            return ReconstructionBackendType.MANUAL_CAD

        # Check for images (for MASt3R)
        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in image_exts]
        if len(images) >= 2:
            # Need MASt3R or similar
            mast3r = self.get_backend(ReconstructionBackendType.MAST3R)
            if mast3r.is_available():
                return ReconstructionBackendType.MAST3R

        return None

    def reconstruct(
        self,
        input_path: Path,
        output_dir: Path,
        scene_id: str,
        backend_type: Optional[ReconstructionBackendType] = None,
        **kwargs
    ) -> ReconstructionResult:
        """Run reconstruction with automatic backend selection.

        Args:
            input_path: Path to input (directory with meshes or images)
            output_dir: Where to write outputs
            scene_id: Scene identifier
            backend_type: Specific backend to use (auto-detect if None)
            **kwargs: Backend-specific options

        Returns:
            ReconstructionResult
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)

        # Select backend
        if backend_type is None:
            backend_type = self.preferred_backend

        if backend_type is None:
            backend_type = self.auto_detect_backend(input_path)

        if backend_type is None:
            return ReconstructionResult(
                success=False,
                scene_id=scene_id,
                backend=ReconstructionBackendType.MANUAL_CAD,
                error="Could not auto-detect reconstruction backend. "
                      "Provide GLB files or multi-view images.",
            )

        self.log(f"Using backend: {backend_type.value}")

        # Get backend
        backend = self.get_backend(backend_type)

        if not backend.is_available():
            if self.fallback_enabled:
                # Try fallback to manual CAD
                self.log(f"Backend {backend_type.value} not available, trying fallback...")
                backend = self.get_backend(ReconstructionBackendType.MANUAL_CAD)
                backend_type = ReconstructionBackendType.MANUAL_CAD
            else:
                return ReconstructionResult(
                    success=False,
                    scene_id=scene_id,
                    backend=backend_type,
                    error=f"Backend {backend_type.value} not available",
                )

        # Run reconstruction
        result = backend.reconstruct(
            input_path=input_path,
            output_dir=output_dir,
            scene_id=scene_id,
            **kwargs
        )

        # Write completion marker
        if result.success:
            self._write_completion_marker(output_dir, result)

        return result

    def _write_completion_marker(
        self, output_dir: Path, result: ReconstructionResult
    ) -> None:
        """Write .regen3d_complete marker for pipeline continuation."""
        marker_path = output_dir / ".regen3d_complete"
        marker_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "scene_id": result.scene_id,
            "backend": result.backend.value,
            "object_count": len(result.objects),
            "processing_time_seconds": result.processing_time_seconds,
        }
        marker_path.write_text(json.dumps(marker_data, indent=2))
        self.log(f"Wrote completion marker: {marker_path}")

    def create_pipeline_inputs(
        self,
        result: ReconstructionResult,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """Create all files needed for downstream pipeline jobs.

        This generates the files that would normally come from 3D-RE-GEN:
        - scene_manifest.json
        - scene_layout_scaled.json
        - inventory.json

        Args:
            result: Reconstruction result
            output_dir: Output directory

        Returns:
            Dict mapping file type to path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Scene manifest
        manifest = result.to_manifest()
        manifest_path = output_dir / "assets" / "scene_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Scene layout (simplified OBB format)
        layout = self._generate_layout(result)
        layout_path = output_dir / "layout" / "scene_layout_scaled.json"
        layout_path.parent.mkdir(parents=True, exist_ok=True)
        layout_path.write_text(json.dumps(layout, indent=2))

        # Inventory (for replicator)
        inventory = self._generate_inventory(result)
        inventory_path = output_dir / "seg" / "inventory.json"
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(json.dumps(inventory, indent=2))

        self.log("Generated pipeline input files:")
        self.log(f"  Manifest: {manifest_path}")
        self.log(f"  Layout: {layout_path}")
        self.log(f"  Inventory: {inventory_path}")

        return {
            "manifest": manifest_path,
            "layout": layout_path,
            "inventory": inventory_path,
        }

    def _generate_layout(self, result: ReconstructionResult) -> Dict[str, Any]:
        """Generate scene_layout_scaled.json from reconstruction result."""
        objects = []

        for obj in result.objects:
            # Compute OBB from bounds
            center = (
                (obj.bounds_min[0] + obj.bounds_max[0]) / 2,
                (obj.bounds_min[1] + obj.bounds_max[1]) / 2,
                (obj.bounds_min[2] + obj.bounds_max[2]) / 2,
            )
            extents = (
                obj.bounds_max[0] - obj.bounds_min[0],
                obj.bounds_max[1] - obj.bounds_min[1],
                obj.bounds_max[2] - obj.bounds_min[2],
            )

            objects.append({
                "id": obj.id,
                "category": obj.category,
                "obb": {
                    "center": list(center),
                    "extents": list(extents),
                    "rotation": list(obj.rotation_quaternion),
                },
                "world_position": list(obj.position),
                "is_floor_contact": obj.is_floor_contact,
            })

        # Compute room bounds
        all_bounds = [
            (obj.bounds_min, obj.bounds_max)
            for obj in result.objects
        ]
        if all_bounds:
            room_min = (
                min(b[0][0] for b in all_bounds),
                min(b[0][1] for b in all_bounds),
                min(b[0][2] for b in all_bounds),
            )
            room_max = (
                max(b[1][0] for b in all_bounds),
                max(b[1][1] for b in all_bounds),
                max(b[1][2] for b in all_bounds),
            )
        else:
            room_min = (-5, 0, -5)
            room_max = (5, 3, 5)

        return {
            "version": "1.0.0",
            "scene_id": result.scene_id,
            "coordinate_frame": result.coordinate_frame,
            "meters_per_unit": result.meters_per_unit,
            "objects": objects,
            "room": {
                "bounds_min": list(room_min),
                "bounds_max": list(room_max),
                "floor_y": 0.0,
                "ceiling_y": room_max[1],
            },
        }

    def _generate_inventory(self, result: ReconstructionResult) -> Dict[str, Any]:
        """Generate inventory.json for replicator job."""
        # Group objects by category
        by_category: Dict[str, List[str]] = {}
        for obj in result.objects:
            if obj.category not in by_category:
                by_category[obj.category] = []
            by_category[obj.category].append(obj.id)

        # Group by sim role
        by_role: Dict[str, List[str]] = {}
        for obj in result.objects:
            if obj.sim_role not in by_role:
                by_role[obj.sim_role] = []
            by_role[obj.sim_role].append(obj.id)

        return {
            "scene_id": result.scene_id,
            "environment_type": result.environment_type,
            "object_count": len(result.objects),
            "by_category": by_category,
            "by_role": by_role,
            "manipulable_objects": by_role.get("manipulable_object", []),
            "articulated_furniture": by_role.get("articulated_furniture", []),
            "static_objects": by_role.get("static", []),
        }


def reconstruct_scene(
    input_path: str,
    output_dir: str,
    scene_id: str = "scene",
    backend: Optional[str] = None,
    **kwargs
) -> ReconstructionResult:
    """Convenience function for reconstruction.

    Args:
        input_path: Path to input
        output_dir: Output directory
        scene_id: Scene identifier
        backend: Backend name (manual-cad, mast3r, etc.)
        **kwargs: Additional options

    Returns:
        ReconstructionResult
    """
    manager = ReconstructionManager()

    backend_type = None
    if backend:
        try:
            backend_type = ReconstructionBackendType(backend)
        except ValueError:
            print(f"Unknown backend: {backend}")
            print(f"Available: {[b.value for b in ReconstructionBackendType]}")

    result = manager.reconstruct(
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        scene_id=scene_id,
        backend_type=backend_type,
        **kwargs
    )

    if result.success:
        # Generate pipeline inputs
        manager.create_pipeline_inputs(result, Path(output_dir))

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BlueprintPipeline 3D Reconstruction")
    parser.add_argument("--input", required=True, help="Input path (directory)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--scene-id", default="scene", help="Scene identifier")
    parser.add_argument(
        "--backend",
        choices=[b.value for b in ReconstructionBackendType],
        help="Reconstruction backend"
    )
    parser.add_argument(
        "--environment-type",
        default="generic",
        help="Environment type (kitchen, warehouse, etc.)"
    )

    args = parser.parse_args()

    result = reconstruct_scene(
        input_path=args.input,
        output_dir=args.output,
        scene_id=args.scene_id,
        backend=args.backend,
        environment_type=args.environment_type,
    )

    if result.success:
        print(f"\nReconstruction successful!")
        print(f"  Objects: {len(result.objects)}")
        print(f"  Backend: {result.backend.value}")
        print(f"  Time: {result.processing_time_seconds:.1f}s")
        print(f"\nOutput: {args.output}")
    else:
        print(f"\nReconstruction failed: {result.error}")
        exit(1)
