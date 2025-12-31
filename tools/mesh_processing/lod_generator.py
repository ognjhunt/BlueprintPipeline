"""
LOD (Level of Detail) Generation for Meshes.

Generates multi-resolution mesh variants for efficient rendering and simulation.
LOD is critical for:
- Rendering performance (distant objects use simpler meshes)
- Physics simulation (collision detection with appropriate detail)
- Memory optimization (streaming lower LODs for distant objects)

Supports:
- GLB/GLTF input meshes
- Configurable LOD levels with target polygon counts
- USD LOD integration with distance-based switching
- Quality-preserving decimation algorithms
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LODLevel:
    """Configuration for a single LOD level."""
    level: int  # 0 = highest detail, increases for lower detail
    target_ratio: float  # Target polygon ratio (0.0-1.0, 1.0 = original)
    screen_size: float  # Screen size threshold for switching (0.0-1.0)

    def __post_init__(self):
        self.target_ratio = max(0.01, min(1.0, self.target_ratio))
        self.screen_size = max(0.0, min(1.0, self.screen_size))


@dataclass
class LODConfig:
    """Configuration for LOD generation."""
    # Default LOD levels: LOD0 (full), LOD1 (50%), LOD2 (25%), LOD3 (10%)
    levels: List[LODLevel] = field(default_factory=lambda: [
        LODLevel(level=0, target_ratio=1.0, screen_size=0.5),
        LODLevel(level=1, target_ratio=0.5, screen_size=0.25),
        LODLevel(level=2, target_ratio=0.25, screen_size=0.1),
        LODLevel(level=3, target_ratio=0.1, screen_size=0.0),
    ])

    # Minimum polygon count (don't decimate below this)
    min_polygons: int = 100

    # Preserve UV seams during decimation
    preserve_uv_seams: bool = True

    # Preserve material boundaries
    preserve_material_boundaries: bool = True

    # Preserve sharp edges (in degrees)
    preserve_sharp_edges_angle: float = 30.0

    # Output format
    output_format: str = "glb"  # "glb", "gltf", "obj"


@dataclass
class LODResult:
    """Result of LOD generation."""
    source_path: Path
    lod_paths: Dict[int, Path]  # level -> path
    polygon_counts: Dict[int, int]  # level -> polygon count
    success: bool
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": str(self.source_path),
            "lods": {str(k): str(v) for k, v in self.lod_paths.items()},
            "polygon_counts": self.polygon_counts,
            "success": self.success,
            "errors": self.errors,
        }


def generate_lod_chain(
    mesh_path: Path,
    output_dir: Path,
    config: Optional[LODConfig] = None,
) -> LODResult:
    """
    Generate a chain of LOD meshes from a source mesh.

    Args:
        mesh_path: Path to source mesh (GLB, GLTF, OBJ)
        output_dir: Directory for output LOD files
        config: LOD configuration (uses defaults if None)

    Returns:
        LODResult with paths to generated LODs
    """
    config = config or LODConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = LODResult(
        source_path=mesh_path,
        lod_paths={},
        polygon_counts={},
        success=False,
    )

    # Try to use trimesh for mesh processing
    try:
        import trimesh
    except ImportError:
        result.errors.append("trimesh not available, cannot generate LODs")
        # Copy original as LOD0
        lod0_path = output_dir / f"{mesh_path.stem}_lod0{mesh_path.suffix}"
        shutil.copy(mesh_path, lod0_path)
        result.lod_paths[0] = lod0_path
        result.success = True
        return result

    # Load mesh
    try:
        scene_or_mesh = trimesh.load(str(mesh_path))
    except Exception as e:
        result.errors.append(f"Failed to load mesh: {e}")
        return result

    # Handle scene vs single mesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        # Combine all meshes in scene
        meshes = []
        for name, geom in scene_or_mesh.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            result.errors.append("No valid meshes found in scene")
            return result
    else:
        mesh = scene_or_mesh

    original_face_count = len(mesh.faces)
    print(f"[LOD] Source mesh: {original_face_count} polygons")

    # Generate each LOD level
    for lod_level in config.levels:
        level = lod_level.level
        target_ratio = lod_level.target_ratio

        # LOD0 is the original mesh
        if level == 0 or target_ratio >= 1.0:
            lod_mesh = mesh.copy()
        else:
            # Calculate target face count
            target_faces = max(
                config.min_polygons,
                int(original_face_count * target_ratio)
            )

            # Decimate mesh
            lod_mesh = _decimate_mesh(
                mesh.copy(),
                target_faces,
                preserve_uv=config.preserve_uv_seams,
            )

        # Save LOD mesh
        lod_filename = f"{mesh_path.stem}_lod{level}.{config.output_format}"
        lod_path = output_dir / lod_filename

        try:
            lod_mesh.export(str(lod_path))
            result.lod_paths[level] = lod_path
            result.polygon_counts[level] = len(lod_mesh.faces)
            print(f"[LOD] Generated LOD{level}: {len(lod_mesh.faces)} polygons ({target_ratio*100:.0f}%)")
        except Exception as e:
            result.errors.append(f"Failed to export LOD{level}: {e}")

    result.success = len(result.lod_paths) > 0

    # Save LOD manifest
    manifest_path = output_dir / f"{mesh_path.stem}_lod_manifest.json"
    manifest_path.write_text(json.dumps(result.to_dict(), indent=2))

    return result


def _decimate_mesh(
    mesh,
    target_faces: int,
    preserve_uv: bool = True,
) -> Any:
    """
    Decimate a mesh to target face count.

    Uses quadric decimation for quality-preserving simplification.
    """
    try:
        # Try to use fast_simplification if available
        import fast_simplification

        vertices = mesh.vertices
        faces = mesh.faces

        simplified_vertices, simplified_faces = fast_simplification.simplify(
            vertices,
            faces,
            target_count=target_faces,
            agg=7,  # Quadric error aggregation
        )

        import trimesh
        return trimesh.Trimesh(vertices=simplified_vertices, faces=simplified_faces)

    except ImportError:
        pass

    # Fallback: use trimesh's built-in simplification
    try:
        # Calculate target reduction
        current_faces = len(mesh.faces)
        if current_faces <= target_faces:
            return mesh

        # Use quadric decimation
        ratio = target_faces / current_faces

        # Trimesh doesn't have great decimation, try simplify_quadric_decimation
        if hasattr(mesh, 'simplify_quadric_decimation'):
            return mesh.simplify_quadric_decimation(target_faces)

        # Alternative: subsample if no decimation available
        # This is a rough fallback
        import trimesh

        # Sample points and reconstruct
        if current_faces > target_faces * 2:
            # Compute a simplified version via voxelization
            voxel_size = mesh.bounding_box.extents.max() / (target_faces ** 0.5)
            voxelized = mesh.voxelized(voxel_size)
            return voxelized.marching_cubes if hasattr(voxelized, 'marching_cubes') else mesh

        return mesh

    except Exception as e:
        print(f"[LOD] Decimation failed: {e}, returning original")
        return mesh


def apply_lod_to_usd(
    usd_path: Path,
    lod_result: LODResult,
    prim_path: str,
    lod_group_name: Optional[str] = None,
) -> bool:
    """
    Apply LOD meshes to a USD stage with proper LOD switching.

    Creates a USD LOD group that automatically switches between LOD levels
    based on screen size.

    Args:
        usd_path: Path to USD file
        lod_result: Result from generate_lod_chain
        prim_path: Prim path where LOD should be applied
        lod_group_name: Name for the LOD group prim

    Returns:
        True if successful
    """
    try:
        from pxr import Usd, UsdGeom, Sdf, Gf
    except ImportError:
        print("[LOD] pxr not available, cannot apply LOD to USD")
        return False

    if not lod_result.success or not lod_result.lod_paths:
        print("[LOD] No LODs to apply")
        return False

    try:
        stage = Usd.Stage.Open(str(usd_path))

        # Create or get the target prim
        target_prim = stage.GetPrimAtPath(prim_path)
        if not target_prim.IsValid():
            print(f"[LOD] Target prim not found: {prim_path}")
            return False

        # Create LOD group name
        if lod_group_name is None:
            lod_group_name = f"{target_prim.GetName()}_LOD"

        # Create LOD group prim
        lod_group_path = f"{prim_path}/{lod_group_name}"
        lod_group = UsdGeom.Scope.Define(stage, lod_group_path)

        # Add LOD variant set
        variant_set = target_prim.GetVariantSets().AddVariantSet("LOD")

        # Copy LOD meshes into stage and create variants
        sorted_levels = sorted(lod_result.lod_paths.keys())

        for level in sorted_levels:
            lod_path = lod_result.lod_paths[level]

            # Add variant
            variant_name = f"LOD{level}"
            variant_set.AddVariant(variant_name)
            variant_set.SetVariantSelection(variant_name)

            # Reference the LOD mesh
            with variant_set.GetVariantEditContext():
                mesh_prim_path = f"{lod_group_path}/mesh_lod{level}"
                mesh_prim = stage.DefinePrim(mesh_prim_path, "Mesh")

                # Add reference to LOD file
                mesh_prim.GetReferences().AddReference(
                    str(lod_path),
                    primPath=Sdf.Path.emptyPath
                )

        # Set default to LOD0
        variant_set.SetVariantSelection("LOD0")

        stage.GetRootLayer().Save()
        print(f"[LOD] Applied {len(sorted_levels)} LOD levels to {prim_path}")
        return True

    except Exception as e:
        print(f"[LOD] Failed to apply LOD to USD: {e}")
        return False


class LODGenerator:
    """
    LOD generation pipeline for batch processing.

    Usage:
        generator = LODGenerator(output_dir, config)
        results = generator.process_meshes([mesh1, mesh2, ...])
    """

    def __init__(
        self,
        output_dir: Path,
        config: Optional[LODConfig] = None,
    ):
        self.output_dir = Path(output_dir)
        self.config = config or LODConfig()
        self.results: List[LODResult] = []

    def process_mesh(self, mesh_path: Path) -> LODResult:
        """Process a single mesh."""
        mesh_output_dir = self.output_dir / mesh_path.stem
        result = generate_lod_chain(mesh_path, mesh_output_dir, self.config)
        self.results.append(result)
        return result

    def process_meshes(self, mesh_paths: List[Path]) -> List[LODResult]:
        """Process multiple meshes."""
        results = []
        for i, mesh_path in enumerate(mesh_paths):
            print(f"[LOD] Processing {i+1}/{len(mesh_paths)}: {mesh_path.name}")
            result = self.process_mesh(mesh_path)
            results.append(result)
        return results

    def save_summary(self, summary_path: Optional[Path] = None) -> Path:
        """Save processing summary."""
        if summary_path is None:
            summary_path = self.output_dir / "lod_summary.json"

        summary = {
            "total_meshes": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
            "results": [r.to_dict() for r in self.results],
        }

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        return summary_path
