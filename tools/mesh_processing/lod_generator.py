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
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LODLevel:
    """Configuration for a single LOD level."""
    level: int  # 0 = highest detail
    target_ratio: float  # Target polygon ratio (0.0-1.0) relative to original
    screen_coverage: float  # Approximate screen coverage threshold for switching

    def __post_init__(self):
        if not 0.0 <= self.target_ratio <= 1.0:
            raise ValueError(f"target_ratio must be 0.0-1.0, got {self.target_ratio}")


@dataclass
class LODConfig:
    """Configuration for LOD generation."""
    levels: List[LODLevel] = field(default_factory=lambda: [
        LODLevel(level=0, target_ratio=1.0, screen_coverage=0.5),
        LODLevel(level=1, target_ratio=0.5, screen_coverage=0.25),
        LODLevel(level=2, target_ratio=0.25, screen_coverage=0.1),
        LODLevel(level=3, target_ratio=0.1, screen_coverage=0.01),
    ])
    preserve_uv: bool = True
    preserve_normals: bool = True
    preserve_boundaries: bool = True

    @classmethod
    def for_simulation(cls) -> "LODConfig":
        """LOD config optimized for physics simulation."""
        return cls(levels=[
            LODLevel(level=0, target_ratio=1.0, screen_coverage=0.5),
            LODLevel(level=1, target_ratio=0.3, screen_coverage=0.1),
            LODLevel(level=2, target_ratio=0.1, screen_coverage=0.01),
        ])

    @classmethod
    def for_visualization(cls) -> "LODConfig":
        """LOD config optimized for rendering quality."""
        return cls(levels=[
            LODLevel(level=0, target_ratio=1.0, screen_coverage=0.4),
            LODLevel(level=1, target_ratio=0.6, screen_coverage=0.2),
            LODLevel(level=2, target_ratio=0.35, screen_coverage=0.1),
            LODLevel(level=3, target_ratio=0.15, screen_coverage=0.05),
        ])


@dataclass
class LODResult:
    """Result of LOD generation for a mesh."""
    source_path: Path
    lod_paths: Dict[int, Path]  # level -> path
    original_face_count: int
    lod_face_counts: Dict[int, int]  # level -> face count
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": str(self.source_path),
            "lods": {k: str(v) for k, v in self.lod_paths.items()},
            "original_faces": self.original_face_count,
            "lod_faces": self.lod_face_counts,
            "success": self.success,
            "error": self.error,
        }


def generate_lod_chain(
    mesh_path: Path,
    output_dir: Path,
    config: Optional[LODConfig] = None,
) -> LODResult:
    """
    Generate LOD chain for a mesh.

    Args:
        mesh_path: Path to input mesh (GLB, GLTF, OBJ, etc.)
        output_dir: Directory for output LOD meshes
        config: LOD configuration (uses default if None)

    Returns:
        LODResult with paths to generated LODs
    """
    config = config or LODConfig()
    mesh_path = Path(mesh_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = LODResult(
        source_path=mesh_path,
        lod_paths={},
        original_face_count=0,
        lod_face_counts={},
        success=False,
    )

    try:
        import trimesh
    except ImportError:
        result.error = "trimesh not installed"
        return result

    try:
        # Load the mesh
        mesh = trimesh.load(mesh_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            # Combine all meshes in scene
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
            else:
                result.error = "No valid meshes in scene"
                return result

        original_faces = len(mesh.faces)
        result.original_face_count = original_faces
        print(f"[LOD] Loaded {mesh_path.name}: {original_faces} faces")

        # Generate each LOD level
        for lod_level in config.levels:
            level = lod_level.level
            target_faces = int(original_faces * lod_level.target_ratio)

            if level == 0:
                # LOD0 is the original mesh
                lod_mesh = mesh.copy()
            else:
                # Decimate mesh
                lod_mesh = _decimate_mesh(
                    mesh,
                    target_faces,
                    preserve_uv=config.preserve_uv,
                    preserve_boundaries=config.preserve_boundaries,
                )

            # Save LOD mesh
            lod_filename = f"{mesh_path.stem}_lod{level}.glb"
            lod_path = output_dir / lod_filename
            lod_mesh.export(lod_path)

            result.lod_paths[level] = lod_path
            result.lod_face_counts[level] = len(lod_mesh.faces)

            reduction = (1 - len(lod_mesh.faces) / original_faces) * 100
            print(f"[LOD] Generated LOD{level}: {len(lod_mesh.faces)} faces "
                  f"({reduction:.1f}% reduction)")

        result.success = True

    except Exception as e:
        result.error = str(e)
        print(f"[LOD] Error processing {mesh_path.name}: {e}")

    return result


def _decimate_mesh(
    mesh: "trimesh.Trimesh",
    target_faces: int,
    preserve_uv: bool = True,
    preserve_boundaries: bool = True,
) -> "trimesh.Trimesh":
    """
    Decimate a mesh to target face count.

    Uses fast_simplification if available, falls back to trimesh.
    """
    import trimesh

    current_faces = len(mesh.faces)
    if target_faces >= current_faces:
        return mesh.copy()

    ratio = target_faces / current_faces

    # Try fast_simplification first (better quality)
    try:
        import fast_simplification

        vertices, faces = fast_simplification.simplify(
            mesh.vertices,
            mesh.faces,
            target_reduction=1.0 - ratio,
            agg=5,  # Aggressiveness (1-10)
        )

        return trimesh.Trimesh(vertices=vertices, faces=faces)

    except ImportError:
        pass

    # Fallback to trimesh's built-in simplification
    try:
        simplified = mesh.simplify_quadric_decimation(target_faces)
        return simplified
    except Exception:
        # Last resort: just return the original
        return mesh.copy()


def apply_lod_to_usd(
    usd_path: Path,
    lod_result: LODResult,
    lod_config: Optional[LODConfig] = None,
) -> bool:
    """
    Apply LOD meshes to a USD file using variant sets.

    Args:
        usd_path: Path to USD file
        lod_result: Result from generate_lod_chain
        lod_config: Configuration with screen coverage values

    Returns:
        True if successful
    """
    try:
        from pxr import Usd, UsdGeom, Sdf
    except ImportError:
        print("[LOD] pxr (OpenUSD) not available, skipping USD LOD integration")
        return False

    config = lod_config or LODConfig()

    try:
        stage = Usd.Stage.Open(str(usd_path))
        if not stage:
            print(f"[LOD] Failed to open USD stage: {usd_path}")
            return False

        # Find the root mesh prim
        root = stage.GetDefaultPrim()
        if not root:
            root = stage.GetPrimAtPath("/")

        # Create LOD variant set
        vset = root.GetVariantSets().AddVariantSet("LOD")

        for lod_level in config.levels:
            level = lod_level.level
            if level not in lod_result.lod_paths:
                continue

            variant_name = f"LOD{level}"
            vset.AddVariant(variant_name)
            vset.SetVariantSelection(variant_name)

            with vset.GetVariantEditContext():
                # Add reference to LOD mesh
                lod_path = lod_result.lod_paths[level]
                # Add as sublayer reference
                root.GetReferences().AddReference(str(lod_path))

        # Set default to LOD0
        vset.SetVariantSelection("LOD0")

        stage.GetRootLayer().Save()
        print(f"[LOD] Applied LOD variants to {usd_path}")
        return True

    except Exception as e:
        print(f"[LOD] Error applying LOD to USD: {e}")
        return False


class LODGenerator:
    """
    LOD generation pipeline for batch processing.

    Usage:
        generator = LODGenerator(output_dir, config=LODConfig.for_simulation())
        results = generator.process_all(mesh_paths)
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

    def process_all(self, mesh_paths: List[Path]) -> List[LODResult]:
        """Process all meshes."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for i, mesh_path in enumerate(mesh_paths):
            print(f"[LOD] Processing {i+1}/{len(mesh_paths)}: {mesh_path.name}")
            self.process_mesh(mesh_path)

        return self.results

    def save_summary(self, summary_path: Optional[Path] = None) -> Path:
        """Save LOD generation summary."""
        if summary_path is None:
            summary_path = self.output_dir / "lod_summary.json"

        summary = {
            "total_meshes": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
            "config": {
                "levels": [
                    {"level": l.level, "ratio": l.target_ratio, "coverage": l.screen_coverage}
                    for l in self.config.levels
                ],
                "preserve_uv": self.config.preserve_uv,
                "preserve_normals": self.config.preserve_normals,
            },
            "results": [r.to_dict() for r in self.results],
        }

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        return summary_path
