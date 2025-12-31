"""
Material and Texture Transfer System.

Transfers materials and textures from source assets (GLB, OBJ) to USD output.
Handles:
- PBR material extraction (baseColor, metallic, roughness, normal, etc.)
- Texture file embedding/referencing
- Material binding in USD
- Fallback material generation

This module bridges the gap between 3D-RE-GEN outputs and simulation-ready USD.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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
    STONE = "stone"
    CONCRETE = "concrete"
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
    MaterialType.STONE: ["stone", "marble", "granite", "slate", "rock"],
    MaterialType.CONCRETE: ["concrete", "cement", "morite"],
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
    MaterialType.STONE: {"density": 2700.0, "friction_static": 0.6, "friction_dynamic": 0.5, "restitution": 0.2},
    MaterialType.CONCRETE: {"density": 2400.0, "friction_static": 0.7, "friction_dynamic": 0.6, "restitution": 0.1},
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
                                pass

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


# =============================================================================
# Enhanced Material Processing
# =============================================================================


def optimize_texture(
    texture_path: Path,
    output_path: Path,
    max_resolution: int = 2048,
    generate_mipmaps: bool = True,
    target_format: str = "png",
) -> Optional[Path]:
    """
    Optimize a texture for simulation use.

    Args:
        texture_path: Input texture path
        output_path: Output texture path
        max_resolution: Maximum resolution (textures larger than this are downscaled)
        generate_mipmaps: Whether to generate mipmaps
        target_format: Target format (png, jpg)

    Returns:
        Path to optimized texture, or None on failure
    """
    try:
        from PIL import Image
    except ImportError:
        print("[MATERIAL] PIL not available, skipping texture optimization")
        shutil.copy(texture_path, output_path)
        return output_path

    try:
        img = Image.open(texture_path)

        # Resize if needed
        if img.width > max_resolution or img.height > max_resolution:
            ratio = min(max_resolution / img.width, max_resolution / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            # Ensure power of 2 for better GPU performance
            new_size = (
                _nearest_power_of_2(new_size[0]),
                _nearest_power_of_2(new_size[1])
            )
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"[MATERIAL] Resized texture to {new_size}")

        # Convert mode if needed
        if target_format == "jpg" and img.mode == "RGBA":
            img = img.convert("RGB")

        # Save optimized texture
        img.save(output_path, format=target_format.upper())

        # Generate mipmaps if requested (save additional files)
        if generate_mipmaps:
            _generate_mipmap_chain(img, output_path)

        return output_path

    except Exception as e:
        print(f"[MATERIAL] Texture optimization failed: {e}")
        shutil.copy(texture_path, output_path)
        return output_path


def _nearest_power_of_2(n: int) -> int:
    """Round to nearest power of 2."""
    if n <= 0:
        return 1
    power = 1
    while power < n:
        power *= 2
    if power - n > n - power // 2:
        return power // 2
    return power


def _generate_mipmap_chain(img, base_path: Path, min_size: int = 16) -> List[Path]:
    """Generate a chain of mipmap textures."""
    try:
        from PIL import Image
    except ImportError:
        return []

    mipmaps = []
    current_size = (img.width, img.height)
    level = 1

    while current_size[0] > min_size and current_size[1] > min_size:
        current_size = (current_size[0] // 2, current_size[1] // 2)
        mip_img = img.resize(current_size, Image.Resampling.LANCZOS)

        mip_path = base_path.parent / f"{base_path.stem}_mip{level}{base_path.suffix}"
        mip_img.save(mip_path)
        mipmaps.append(mip_path)
        level += 1

    return mipmaps


def process_normal_map(
    normal_texture_path: Path,
    output_path: Path,
    flip_y: bool = False,
    normalize: bool = True,
) -> Optional[Path]:
    """
    Process a normal map for correct simulation rendering.

    Different 3D software uses different conventions for normal maps:
    - OpenGL style: Y+ is up (green channel points up)
    - DirectX style: Y- is up (green channel points down)

    Isaac Sim uses OpenGL-style normal maps.

    Args:
        normal_texture_path: Input normal map path
        output_path: Output normal map path
        flip_y: If True, flip the Y (green) channel for DirectX->OpenGL conversion
        normalize: If True, re-normalize the normal vectors

    Returns:
        Path to processed normal map, or None on failure
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("[MATERIAL] PIL/numpy not available, copying normal map as-is")
        shutil.copy(normal_texture_path, output_path)
        return output_path

    try:
        img = Image.open(normal_texture_path)
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Handle different channel counts
        if len(img_array.shape) == 2:
            # Grayscale - probably not a valid normal map
            print("[MATERIAL] Warning: Normal map appears to be grayscale")
            shutil.copy(normal_texture_path, output_path)
            return output_path

        if flip_y:
            # Flip green channel
            img_array[:, :, 1] = 1.0 - img_array[:, :, 1]

        if normalize:
            # Convert from [0,1] to [-1,1]
            normals = img_array[:, :, :3] * 2.0 - 1.0

            # Normalize vectors
            length = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True))
            length = np.maximum(length, 1e-6)  # Avoid division by zero
            normals = normals / length

            # Convert back to [0,1]
            img_array[:, :, :3] = (normals + 1.0) / 2.0

        # Convert back to uint8
        img_array = (img_array * 255).astype(np.uint8)
        result_img = Image.fromarray(img_array)
        result_img.save(output_path)

        return output_path

    except Exception as e:
        print(f"[MATERIAL] Normal map processing failed: {e}")
        shutil.copy(normal_texture_path, output_path)
        return output_path


def bind_material_to_mesh(
    usd_path: Path,
    mesh_prim_path: str,
    material_name: str,
    materials_scope: str = "/World/Materials",
) -> bool:
    """
    Bind a material to a specific mesh prim in USD.

    Args:
        usd_path: Path to USD file
        mesh_prim_path: Prim path of the mesh
        material_name: Name of the material to bind
        materials_scope: Scope where materials are defined

    Returns:
        True if successful
    """
    try:
        from pxr import Usd, UsdShade
    except ImportError:
        print("[MATERIAL] pxr not available")
        return False

    try:
        stage = Usd.Stage.Open(str(usd_path))
        mesh_prim = stage.GetPrimAtPath(mesh_prim_path)

        if not mesh_prim.IsValid():
            print(f"[MATERIAL] Mesh prim not found: {mesh_prim_path}")
            return False

        material_path = f"{materials_scope}/{material_name}"
        material_prim = stage.GetPrimAtPath(material_path)

        if not material_prim.IsValid():
            print(f"[MATERIAL] Material not found: {material_path}")
            return False

        material = UsdShade.Material(material_prim)
        binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        binding_api.Bind(material)

        stage.GetRootLayer().Save()
        return True

    except Exception as e:
        print(f"[MATERIAL] Material binding failed: {e}")
        return False


def create_usd_material_with_textures(
    stage,
    material_name: str,
    material: PBRMaterial,
    materials_scope: str = "/World/Materials",
    textures_relative_path: str = "./textures",
) -> bool:
    """
    Create a complete USD material with all texture bindings.

    This creates a UsdPreviewSurface material with proper texture
    reader connections for all available texture maps.

    Args:
        stage: USD stage
        material_name: Name for the material
        material: PBRMaterial definition
        materials_scope: USD path for materials
        textures_relative_path: Relative path to textures directory

    Returns:
        True if successful
    """
    try:
        from pxr import UsdShade, Sdf, Gf
    except ImportError:
        return False

    mat_path = f"{materials_scope}/{material_name}"

    # Create material
    usd_material = UsdShade.Material.Define(stage, mat_path)

    # Create PBR shader
    shader_path = f"{mat_path}/PBRShader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")

    # Create ST reader for UV coordinates
    st_reader_path = f"{mat_path}/STReader"
    st_reader = UsdShade.Shader.Define(stage, st_reader_path)
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_output = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    # Base color
    if material.base_color_texture and material.base_color_texture.path.is_file():
        tex_path = f"{textures_relative_path}/{material.base_color_texture.path.name}"
        _create_texture_reader(
            stage, f"{mat_path}/BaseColorTex", tex_path, st_output,
            shader, "diffuseColor", Sdf.ValueTypeNames.Color3f
        )
    else:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*material.base_color[:3])
        )

    # Metallic-Roughness
    if material.metallic_roughness_texture and material.metallic_roughness_texture.path.is_file():
        tex_path = f"{textures_relative_path}/{material.metallic_roughness_texture.path.name}"
        # Metallic from blue channel
        _create_texture_reader(
            stage, f"{mat_path}/MetallicTex", tex_path, st_output,
            shader, "metallic", Sdf.ValueTypeNames.Float, channel="b"
        )
        # Roughness from green channel
        _create_texture_reader(
            stage, f"{mat_path}/RoughnessTex", tex_path, st_output,
            shader, "roughness", Sdf.ValueTypeNames.Float, channel="g"
        )
    else:
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(material.metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(material.roughness)

    # Normal map
    if material.normal_texture and material.normal_texture.path.is_file():
        tex_path = f"{textures_relative_path}/{material.normal_texture.path.name}"
        normal_tex_path = f"{mat_path}/NormalTex"
        normal_tex = UsdShade.Shader.Define(stage, normal_tex_path)
        normal_tex.CreateIdAttr("UsdUVTexture")
        normal_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_path)
        normal_tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_output)
        normal_tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(2.0, 2.0, 2.0, 1.0)
        )
        normal_tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(-1.0, -1.0, -1.0, 0.0)
        )
        normal_output = normal_tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            normal_output
        )

    # Occlusion
    if material.occlusion_texture and material.occlusion_texture.path.is_file():
        tex_path = f"{textures_relative_path}/{material.occlusion_texture.path.name}"
        _create_texture_reader(
            stage, f"{mat_path}/OcclusionTex", tex_path, st_output,
            shader, "occlusion", Sdf.ValueTypeNames.Float, channel="r"
        )
    else:
        shader.CreateInput("occlusion", Sdf.ValueTypeNames.Float).Set(
            material.occlusion_strength
        )

    # Emissive
    if material.emissive_texture and material.emissive_texture.path.is_file():
        tex_path = f"{textures_relative_path}/{material.emissive_texture.path.name}"
        _create_texture_reader(
            stage, f"{mat_path}/EmissiveTex", tex_path, st_output,
            shader, "emissiveColor", Sdf.ValueTypeNames.Color3f
        )
    elif any(c > 0 for c in material.emissive_color):
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*material.emissive_color)
        )

    # Opacity for transparent materials
    if material.alpha_mode != "OPAQUE":
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
            material.base_color[3] if len(material.base_color) > 3 else 1.0
        )
        if material.alpha_mode == "MASK":
            shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(
                material.alpha_cutoff
            )

    # Connect shader to material surface
    usd_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    return True


def _create_texture_reader(
    stage,
    tex_reader_path: str,
    texture_file: str,
    st_output,
    target_shader,
    input_name: str,
    input_type,
    channel: str = "rgb"
):
    """Helper to create a texture reader and connect it to a shader input."""
    try:
        from pxr import UsdShade, Sdf
    except ImportError:
        return

    tex_reader = UsdShade.Shader.Define(stage, tex_reader_path)
    tex_reader.CreateIdAttr("UsdUVTexture")
    tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_file)
    tex_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_output)

    if channel == "rgb":
        output = tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    elif channel in ("r", "g", "b", "a"):
        output = tex_reader.CreateOutput(channel, Sdf.ValueTypeNames.Float)
    else:
        output = tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    target_shader.CreateInput(input_name, input_type).ConnectToSource(output)


class EnhancedMaterialTransferPipeline(MaterialTransferPipeline):
    """
    Enhanced material transfer with texture optimization and advanced binding.

    Usage:
        pipeline = EnhancedMaterialTransferPipeline(source_glb, output_dir)
        pipeline.extract()
        pipeline.optimize_textures(max_resolution=2048)
        pipeline.apply_to_usd_enhanced(usd_path)
    """

    def __init__(
        self,
        source_path: Path,
        output_dir: Path,
        optimize_textures: bool = True,
        max_texture_resolution: int = 2048,
    ):
        super().__init__(source_path, output_dir)
        self.optimize = optimize_textures
        self.max_resolution = max_texture_resolution
        self.optimized_textures: Dict[str, Path] = {}

    def optimize_all_textures(self) -> Dict[str, Path]:
        """Optimize all extracted textures."""
        if not self.materials:
            return {}

        optimized_dir = self.output_dir / "textures_optimized"
        optimized_dir.mkdir(parents=True, exist_ok=True)

        for mat in self.materials:
            # Base color texture
            if mat.base_color_texture and mat.base_color_texture.path.is_file():
                opt_path = optimized_dir / mat.base_color_texture.path.name
                optimize_texture(
                    mat.base_color_texture.path,
                    opt_path,
                    max_resolution=self.max_resolution,
                )
                self.optimized_textures[str(mat.base_color_texture.path)] = opt_path
                mat.base_color_texture.path = opt_path

            # Normal map (with special processing)
            if mat.normal_texture and mat.normal_texture.path.is_file():
                opt_path = optimized_dir / mat.normal_texture.path.name
                process_normal_map(mat.normal_texture.path, opt_path)
                self.optimized_textures[str(mat.normal_texture.path)] = opt_path
                mat.normal_texture.path = opt_path

            # Other textures
            for tex_attr in ['metallic_roughness_texture', 'emissive_texture', 'occlusion_texture']:
                tex_info = getattr(mat, tex_attr, None)
                if tex_info and tex_info.path.is_file():
                    opt_path = optimized_dir / tex_info.path.name
                    optimize_texture(tex_info.path, opt_path, max_resolution=self.max_resolution)
                    self.optimized_textures[str(tex_info.path)] = opt_path
                    tex_info.path = opt_path

        print(f"[MATERIAL] Optimized {len(self.optimized_textures)} textures")
        return self.optimized_textures

    def apply_to_usd_enhanced(self, usd_path: Path) -> bool:
        """Apply materials with full texture support."""
        try:
            from pxr import Usd
        except ImportError:
            print("[MATERIAL] pxr not available")
            return super().apply_to_usd(usd_path)

        try:
            stage = Usd.Stage.Open(str(usd_path))

            # Create materials scope
            materials_scope = "/World/Materials"
            if not stage.GetPrimAtPath(materials_scope):
                stage.DefinePrim(materials_scope, "Scope")

            # Create textures directory next to USD
            textures_dir = usd_path.parent / "textures"
            textures_dir.mkdir(parents=True, exist_ok=True)

            for mat in self.materials:
                # Copy textures to USD directory
                self._copy_material_textures(mat, textures_dir)

                # Create full material with textures
                create_usd_material_with_textures(
                    stage,
                    mat.name,
                    mat,
                    materials_scope=materials_scope,
                    textures_relative_path="./textures",
                )

            stage.GetRootLayer().Save()
            print(f"[MATERIAL] Applied {len(self.materials)} enhanced materials to {usd_path}")
            return True

        except Exception as e:
            print(f"[MATERIAL] Enhanced material application failed: {e}")
            return super().apply_to_usd(usd_path)

    def _copy_material_textures(self, mat: PBRMaterial, dest_dir: Path) -> None:
        """Copy all material textures to destination directory."""
        for tex_attr in [
            'base_color_texture', 'metallic_roughness_texture',
            'normal_texture', 'emissive_texture', 'occlusion_texture'
        ]:
            tex_info = getattr(mat, tex_attr, None)
            if tex_info and tex_info.path.is_file():
                dest_path = dest_dir / tex_info.path.name
                if not dest_path.exists():
                    shutil.copy(tex_info.path, dest_path)
