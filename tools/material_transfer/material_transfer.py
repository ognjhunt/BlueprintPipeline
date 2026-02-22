"""
Material and Texture Transfer System.

Transfers materials and textures from source assets (GLB, OBJ) to USD output.
Handles:
- PBR material extraction (baseColor, metallic, roughness, normal, etc.)
- Texture file embedding/referencing
- Material binding in USD
- Fallback material generation

This module bridges the gap between Stage 1 text generation outputs and simulation-ready USD.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MaterialType(str, Enum):
    """Standard material types for simulation."""
    PLASTIC = "plastic"
    METAL = "metal"
    WOOD = "wood"
    FABRIC = "fabric"
    GLASS = "glass"
    CERAMIC = "ceramic"
    RUBBER = "rubber"
    LEATHER = "leather"
    PAPER = "paper"
    CARDBOARD = "cardboard"
    STONE = "stone"
    CONCRETE = "concrete"
    FOAM = "foam"
    LIQUID = "liquid"
    FOOD = "food"
    ELECTRONICS = "electronics"
    COMPOSITE = "composite"
    TILE = "tile"
    ORGANIC = "organic"
    UNKNOWN = "unknown"


@dataclass
class TextureInfo:
    """Information about a texture map."""
    path: Path
    texture_type: str  # "baseColor", "metallic", "roughness", "normal", "emissive", "occlusion"
    format: str  # "png", "jpg", "exr"
    width: int = 0
    height: int = 0
    channels: int = 3
    is_embedded: bool = False  # True if texture data is embedded in GLB

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "texture_type": self.texture_type,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "is_embedded": self.is_embedded,
        }


@dataclass
class PBRMaterial:
    """Physically-Based Rendering material definition."""
    name: str
    material_type: MaterialType = MaterialType.UNKNOWN

    # Base color (albedo)
    base_color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    base_color_texture: Optional[TextureInfo] = None

    # Metallic-Roughness
    metallic: float = 0.0
    roughness: float = 0.5
    metallic_roughness_texture: Optional[TextureInfo] = None

    # Normal map
    normal_texture: Optional[TextureInfo] = None
    normal_scale: float = 1.0

    # Emissive
    emissive_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    emissive_texture: Optional[TextureInfo] = None
    emissive_intensity: float = 1.0

    # Occlusion
    occlusion_texture: Optional[TextureInfo] = None
    occlusion_strength: float = 1.0

    # Physics properties (for simulation)
    density: float = 1000.0  # kg/m³
    friction_static: float = 0.5
    friction_dynamic: float = 0.4
    restitution: float = 0.3

    # Additional properties
    double_sided: bool = False
    alpha_mode: str = "OPAQUE"  # "OPAQUE", "MASK", "BLEND"
    alpha_cutoff: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "material_type": self.material_type.value,
            "base_color": list(self.base_color),
            "base_color_texture": self.base_color_texture.to_dict() if self.base_color_texture else None,
            "metallic": self.metallic,
            "roughness": self.roughness,
            "metallic_roughness_texture": self.metallic_roughness_texture.to_dict() if self.metallic_roughness_texture else None,
            "normal_texture": self.normal_texture.to_dict() if self.normal_texture else None,
            "normal_scale": self.normal_scale,
            "emissive_color": list(self.emissive_color),
            "emissive_texture": self.emissive_texture.to_dict() if self.emissive_texture else None,
            "emissive_intensity": self.emissive_intensity,
            "occlusion_texture": self.occlusion_texture.to_dict() if self.occlusion_texture else None,
            "occlusion_strength": self.occlusion_strength,
            "density": self.density,
            "friction_static": self.friction_static,
            "friction_dynamic": self.friction_dynamic,
            "restitution": self.restitution,
            "double_sided": self.double_sided,
            "alpha_mode": self.alpha_mode,
            "alpha_cutoff": self.alpha_cutoff,
        }


# Material type inference from name/color
MATERIAL_KEYWORDS = {
    MaterialType.METAL: ["metal", "steel", "iron", "aluminum", "chrome", "brass", "copper", "silver", "gold"],
    MaterialType.WOOD: ["wood", "oak", "pine", "walnut", "bamboo", "plywood", "timber"],
    MaterialType.PLASTIC: ["plastic", "pvc", "acrylic", "polycarbonate", "abs", "nylon"],
    MaterialType.FABRIC: ["fabric", "cloth", "cotton", "polyester", "linen", "velvet", "silk"],
    MaterialType.GLASS: ["glass", "window", "mirror", "crystal"],
    MaterialType.CERAMIC: ["ceramic", "porcelain", "tile", "pottery"],
    MaterialType.RUBBER: ["rubber", "silicone", "latex", "foam"],
    MaterialType.LEATHER: ["leather", "suede", "hide"],
    MaterialType.PAPER: ["paper", "cardboard", "carton"],
    MaterialType.CARDBOARD: ["cardboard", "carton", "corrugated"],
    MaterialType.STONE: ["stone", "marble", "granite", "slate", "rock"],
    MaterialType.CONCRETE: ["concrete", "cement", "morite"],
    MaterialType.FOAM: ["foam", "sponge", "expanded"],
    MaterialType.LIQUID: ["water", "juice", "liquid", "oil"],
    MaterialType.FOOD: ["fruit", "vegetable", "bread", "meat", "food"],
    MaterialType.ELECTRONICS: ["electronic", "circuit", "pcb", "phone", "laptop"],
    MaterialType.COMPOSITE: ["composite", "laminate", "fiber"],
    MaterialType.TILE: ["tile", "ceramic_tile"],
    MaterialType.ORGANIC: ["organic", "plant", "wood_pulp", "biomass"],
}

# Default physics properties by material type
MATERIAL_PHYSICS = {
    MaterialType.METAL: {"density": 7800.0, "friction_static": 0.6, "friction_dynamic": 0.5, "restitution": 0.2},
    MaterialType.WOOD: {"density": 700.0, "friction_static": 0.5, "friction_dynamic": 0.4, "restitution": 0.3},
    MaterialType.PLASTIC: {"density": 1200.0, "friction_static": 0.4, "friction_dynamic": 0.3, "restitution": 0.4},
    MaterialType.FABRIC: {"density": 300.0, "friction_static": 0.7, "friction_dynamic": 0.6, "restitution": 0.1},
    MaterialType.GLASS: {"density": 2500.0, "friction_static": 0.4, "friction_dynamic": 0.3, "restitution": 0.5},
    MaterialType.CERAMIC: {"density": 2400.0, "friction_static": 0.5, "friction_dynamic": 0.4, "restitution": 0.2},
    MaterialType.RUBBER: {"density": 1100.0, "friction_static": 0.9, "friction_dynamic": 0.8, "restitution": 0.7},
    MaterialType.LEATHER: {"density": 900.0, "friction_static": 0.6, "friction_dynamic": 0.5, "restitution": 0.2},
    MaterialType.PAPER: {"density": 700.0, "friction_static": 0.4, "friction_dynamic": 0.3, "restitution": 0.1},
    MaterialType.CARDBOARD: {"density": 650.0, "friction_static": 0.45, "friction_dynamic": 0.35, "restitution": 0.12},
    MaterialType.STONE: {"density": 2700.0, "friction_static": 0.6, "friction_dynamic": 0.5, "restitution": 0.2},
    MaterialType.CONCRETE: {"density": 2400.0, "friction_static": 0.7, "friction_dynamic": 0.6, "restitution": 0.1},
    MaterialType.FOAM: {"density": 120.0, "friction_static": 0.6, "friction_dynamic": 0.5, "restitution": 0.05},
    MaterialType.LIQUID: {"density": 1000.0, "friction_static": 0.05, "friction_dynamic": 0.03, "restitution": 0.0},
    MaterialType.FOOD: {"density": 850.0, "friction_static": 0.55, "friction_dynamic": 0.45, "restitution": 0.08},
    MaterialType.ELECTRONICS: {"density": 1800.0, "friction_static": 0.45, "friction_dynamic": 0.35, "restitution": 0.1},
    MaterialType.COMPOSITE: {"density": 1400.0, "friction_static": 0.5, "friction_dynamic": 0.4, "restitution": 0.12},
    MaterialType.TILE: {"density": 2200.0, "friction_static": 0.65, "friction_dynamic": 0.55, "restitution": 0.06},
    MaterialType.ORGANIC: {"density": 700.0, "friction_static": 0.52, "friction_dynamic": 0.42, "restitution": 0.1},
    MaterialType.UNKNOWN: {"density": 1000.0, "friction_static": 0.5, "friction_dynamic": 0.4, "restitution": 0.3},
}


def infer_material_type(name: str, base_color: Optional[Tuple[float, ...]] = None) -> MaterialType:
    """
    Infer material type from name and color.

    Args:
        name: Material name
        base_color: Optional RGBA base color

    Returns:
        Inferred MaterialType
    """
    name_lower = name.lower()

    for mat_type, keywords in MATERIAL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in name_lower:
                return mat_type

    # Infer from color if name doesn't match
    if base_color and len(base_color) >= 3:
        r, g, b = base_color[:3]

        # Metallic look (high brightness, low saturation)
        brightness = (r + g + b) / 3
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        saturation = (max_c - min_c) / (max_c + 0.001)

        if brightness > 0.7 and saturation < 0.2:
            return MaterialType.METAL

        # Wood-like (warm brown tones)
        if r > g > b and 0.2 < brightness < 0.6:
            return MaterialType.WOOD

    return MaterialType.UNKNOWN


def extract_materials_from_glb(glb_path: Path, output_dir: Path) -> List[PBRMaterial]:
    """
    Extract materials and textures from a GLB file.

    Args:
        glb_path: Path to GLB file
        output_dir: Directory to save extracted textures

    Returns:
        List of PBRMaterial objects
    """
    materials = []

    try:
        import pygltflib
    except ImportError:
        print("[MATERIAL] WARNING: pygltflib not available, using fallback material extraction")
        return _extract_materials_trimesh(glb_path, output_dir)

    try:
        gltf = pygltflib.GLTF2().load(str(glb_path))
    except Exception as e:
        print(f"[MATERIAL] Failed to load GLB: {e}")
        return materials

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract textures
    texture_paths: Dict[int, Path] = {}
    if gltf.images:
        for idx, image in enumerate(gltf.images):
            # Get image data
            if image.bufferView is not None:
                buffer_view = gltf.bufferViews[image.bufferView]
                buffer = gltf.buffers[buffer_view.buffer]

                # Get binary data
                if hasattr(gltf, '_glb_data') and gltf._glb_data:
                    data = gltf._glb_data[buffer_view.byteOffset:buffer_view.byteOffset + buffer_view.byteLength]
                else:
                    # Try to read from buffer uri
                    data = gltf.get_data_from_buffer_uri(buffer.uri) if buffer.uri else b''
                    if data:
                        data = data[buffer_view.byteOffset:buffer_view.byteOffset + buffer_view.byteLength]

                if data:
                    # Determine extension
                    ext = "png"
                    if image.mimeType == "image/jpeg":
                        ext = "jpg"
                    elif image.mimeType == "image/png":
                        ext = "png"

                    # Save texture
                    tex_path = output_dir / f"texture_{idx}.{ext}"
                    tex_path.write_bytes(data)
                    texture_paths[idx] = tex_path

    # Extract materials
    if gltf.materials:
        for idx, mat in enumerate(gltf.materials):
            name = mat.name or f"material_{idx}"

            pbr_mat = PBRMaterial(name=name)

            # PBR Metallic Roughness
            if mat.pbrMetallicRoughness:
                pbr = mat.pbrMetallicRoughness

                # Base color
                if pbr.baseColorFactor:
                    pbr_mat.base_color = tuple(pbr.baseColorFactor)

                if pbr.baseColorTexture and pbr.baseColorTexture.index is not None:
                    tex_idx = gltf.textures[pbr.baseColorTexture.index].source
                    if tex_idx in texture_paths:
                        pbr_mat.base_color_texture = TextureInfo(
                            path=texture_paths[tex_idx],
                            texture_type="baseColor",
                            format=texture_paths[tex_idx].suffix.lstrip("."),
                            is_embedded=True,
                        )

                # Metallic & Roughness
                pbr_mat.metallic = pbr.metallicFactor if pbr.metallicFactor is not None else 0.0
                pbr_mat.roughness = pbr.roughnessFactor if pbr.roughnessFactor is not None else 0.5

                if pbr.metallicRoughnessTexture and pbr.metallicRoughnessTexture.index is not None:
                    tex_idx = gltf.textures[pbr.metallicRoughnessTexture.index].source
                    if tex_idx in texture_paths:
                        pbr_mat.metallic_roughness_texture = TextureInfo(
                            path=texture_paths[tex_idx],
                            texture_type="metallicRoughness",
                            format=texture_paths[tex_idx].suffix.lstrip("."),
                            is_embedded=True,
                        )

            # Normal map
            if mat.normalTexture and mat.normalTexture.index is not None:
                tex_idx = gltf.textures[mat.normalTexture.index].source
                if tex_idx in texture_paths:
                    pbr_mat.normal_texture = TextureInfo(
                        path=texture_paths[tex_idx],
                        texture_type="normal",
                        format=texture_paths[tex_idx].suffix.lstrip("."),
                        is_embedded=True,
                    )
                pbr_mat.normal_scale = mat.normalTexture.scale if mat.normalTexture.scale else 1.0

            # Emissive
            if mat.emissiveFactor:
                pbr_mat.emissive_color = tuple(mat.emissiveFactor)

            if mat.emissiveTexture and mat.emissiveTexture.index is not None:
                tex_idx = gltf.textures[mat.emissiveTexture.index].source
                if tex_idx in texture_paths:
                    pbr_mat.emissive_texture = TextureInfo(
                        path=texture_paths[tex_idx],
                        texture_type="emissive",
                        format=texture_paths[tex_idx].suffix.lstrip("."),
                        is_embedded=True,
                    )

            # Occlusion
            if mat.occlusionTexture and mat.occlusionTexture.index is not None:
                tex_idx = gltf.textures[mat.occlusionTexture.index].source
                if tex_idx in texture_paths:
                    pbr_mat.occlusion_texture = TextureInfo(
                        path=texture_paths[tex_idx],
                        texture_type="occlusion",
                        format=texture_paths[tex_idx].suffix.lstrip("."),
                        is_embedded=True,
                    )
                pbr_mat.occlusion_strength = mat.occlusionTexture.strength if mat.occlusionTexture.strength else 1.0

            # Alpha mode
            pbr_mat.alpha_mode = mat.alphaMode or "OPAQUE"
            pbr_mat.alpha_cutoff = mat.alphaCutoff if mat.alphaCutoff else 0.5
            pbr_mat.double_sided = mat.doubleSided or False

            # Infer material type and physics
            pbr_mat.material_type = infer_material_type(name, pbr_mat.base_color)
            physics = MATERIAL_PHYSICS.get(pbr_mat.material_type, MATERIAL_PHYSICS[MaterialType.UNKNOWN])
            pbr_mat.density = physics["density"]
            pbr_mat.friction_static = physics["friction_static"]
            pbr_mat.friction_dynamic = physics["friction_dynamic"]
            pbr_mat.restitution = physics["restitution"]

            materials.append(pbr_mat)

    return materials


def _extract_materials_trimesh(glb_path: Path, output_dir: Path) -> List[PBRMaterial]:
    """Fallback material extraction using trimesh."""
    materials = []

    try:
        import trimesh
    except ImportError:
        print("[MATERIAL] WARNING: trimesh not available")
        return materials

    try:
        scene = trimesh.load(str(glb_path))
    except Exception as e:
        print(f"[MATERIAL] Failed to load GLB with trimesh: {e}")
        return materials

    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(scene, trimesh.Scene):
        for name, geometry in scene.geometry.items():
            if hasattr(geometry, 'visual') and geometry.visual:
                visual = geometry.visual
                mat_name = f"material_{name}"

                pbr_mat = PBRMaterial(name=mat_name)

                # Try to get material properties
                if hasattr(visual, 'material'):
                    material = visual.material
                    if hasattr(material, 'baseColorFactor'):
                        pbr_mat.base_color = tuple(material.baseColorFactor)
                    if hasattr(material, 'metallicFactor'):
                        pbr_mat.metallic = material.metallicFactor
                    if hasattr(material, 'roughnessFactor'):
                        pbr_mat.roughness = material.roughnessFactor

                # Check for texture
                if hasattr(visual, 'uv') and visual.uv is not None:
                    if hasattr(visual, 'material') and hasattr(visual.material, 'image'):
                        img = visual.material.image
                        if img is not None:
                            tex_path = output_dir / f"texture_{name}.png"
                            try:
                                img.save(tex_path)
                                pbr_mat.base_color_texture = TextureInfo(
                                    path=tex_path,
                                    texture_type="baseColor",
                                    format="png",
                                    width=img.width,
                                    height=img.height,
                                    is_embedded=True,
                                )
                            except Exception:
                                logger.warning(
                                    "Failed to save texture for material %s to %s.",
                                    name,
                                    tex_path,
                                    exc_info=True,
                                )

                # Infer material type
                pbr_mat.material_type = infer_material_type(mat_name, pbr_mat.base_color)
                physics = MATERIAL_PHYSICS.get(pbr_mat.material_type, MATERIAL_PHYSICS[MaterialType.UNKNOWN])
                pbr_mat.density = physics["density"]
                pbr_mat.friction_static = physics["friction_static"]
                pbr_mat.friction_dynamic = physics["friction_dynamic"]
                pbr_mat.restitution = physics["restitution"]

                materials.append(pbr_mat)

    return materials


def apply_materials_to_usd(
    usd_path: Path,
    materials: List[PBRMaterial],
    textures_dir: Optional[Path] = None,
) -> bool:
    """
    Apply materials to a USD file.

    Args:
        usd_path: Path to USD file
        materials: List of PBRMaterial objects
        textures_dir: Directory containing texture files

    Returns:
        True if successful
    """
    try:
        from pxr import Usd, UsdShade, Sdf, Gf
    except ImportError:
        print("[MATERIAL] WARNING: pxr not available, cannot apply materials to USD")
        return False

    try:
        stage = Usd.Stage.Open(str(usd_path))
    except Exception as e:
        print(f"[MATERIAL] Failed to open USD: {e}")
        return False

    # Create materials scope
    materials_path = "/World/Materials"
    if not stage.GetPrimAtPath(materials_path):
        stage.DefinePrim(materials_path, "Scope")

    for pbr_mat in materials:
        mat_path = f"{materials_path}/{pbr_mat.name}"

        # Create material
        material = UsdShade.Material.Define(stage, mat_path)

        # Create PBR shader
        shader_path = f"{mat_path}/PBRShader"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")

        # Base color
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*pbr_mat.base_color[:3])
        )

        # Metallic
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(pbr_mat.metallic)

        # Roughness
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(pbr_mat.roughness)

        # Connect shader to material
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # Add texture readers if textures exist
        if pbr_mat.base_color_texture and pbr_mat.base_color_texture.path.is_file():
            tex_reader_path = f"{mat_path}/BaseColorTexture"
            tex_reader = UsdShade.Shader.Define(stage, tex_reader_path)
            tex_reader.CreateIdAttr("UsdUVTexture")

            # Copy texture to USD directory if needed
            if textures_dir:
                tex_dest = textures_dir / pbr_mat.base_color_texture.path.name
                if not tex_dest.exists():
                    shutil.copy(pbr_mat.base_color_texture.path, tex_dest)
                tex_path_str = f"./{textures_dir.name}/{tex_dest.name}"
            else:
                tex_path_str = str(pbr_mat.base_color_texture.path)

            tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_path_str)
            tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex_reader.ConnectableAPI(), "rgb"
            )

    stage.GetRootLayer().Save()
    return True


def create_material_manifest(
    materials: List[PBRMaterial],
    output_path: Path,
) -> None:
    """
    Create a JSON manifest of extracted materials.

    Args:
        materials: List of PBRMaterial objects
        output_path: Path to output JSON file
    """
    manifest = {
        "version": "1.0",
        "material_count": len(materials),
        "materials": [mat.to_dict() for mat in materials],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))


class MaterialTransferPipeline:
    """
    Orchestrates material and texture transfer from source to USD.

    Usage:
        pipeline = MaterialTransferPipeline(source_glb, output_dir)
        materials = pipeline.extract()
        pipeline.apply_to_usd(usd_path)
    """

    def __init__(self, source_path: Path, output_dir: Path):
        self.source_path = Path(source_path)
        self.output_dir = Path(output_dir)
        self.textures_dir = self.output_dir / "textures"
        self.materials: List[PBRMaterial] = []

    def extract(self) -> List[PBRMaterial]:
        """Extract materials from source asset."""
        suffix = self.source_path.suffix.lower()

        if suffix in (".glb", ".gltf"):
            self.materials = extract_materials_from_glb(self.source_path, self.textures_dir)
        else:
            print(f"[MATERIAL] Unsupported source format: {suffix}")
            self.materials = []

        print(f"[MATERIAL] Extracted {len(self.materials)} materials from {self.source_path.name}")
        return self.materials

    def apply_to_usd(self, usd_path: Path) -> bool:
        """Apply extracted materials to USD."""
        if not self.materials:
            print("[MATERIAL] No materials to apply")
            return False

        return apply_materials_to_usd(usd_path, self.materials, self.textures_dir)

    def save_manifest(self, manifest_path: Optional[Path] = None) -> Path:
        """Save material manifest."""
        if manifest_path is None:
            manifest_path = self.output_dir / "materials_manifest.json"

        create_material_manifest(self.materials, manifest_path)
        return manifest_path


def transfer_materials(
    source_glb: Path,
    target_usd: Path,
    output_dir: Optional[Path] = None,
) -> Tuple[bool, List[PBRMaterial]]:
    """
    Convenience function to transfer materials from GLB to USD.

    Args:
        source_glb: Path to source GLB file
        target_usd: Path to target USD file
        output_dir: Optional output directory for textures

    Returns:
        Tuple of (success, list of materials)
    """
    if output_dir is None:
        output_dir = target_usd.parent

    pipeline = MaterialTransferPipeline(source_glb, output_dir)
    materials = pipeline.extract()

    if materials:
        success = pipeline.apply_to_usd(target_usd)
        pipeline.save_manifest()
        return success, materials

    return False, []
