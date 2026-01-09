#!/usr/bin/env python3
"""
Procedural Asset Generator for Domain Randomization.

Generates unlimited variations of 3D assets by varying:
- Color (HSV color space)
- Scale (uniform and non-uniform)
- Texture (procedural, style transfer, atlas)
- Material properties (PBR: roughness, metallic, specular)
- Physical properties (mass, friction, restitution)

This enables infinite domain randomization for robust robot learning.

Usage:
    from procedural_generator import ProceduralAssetGenerator, VariationParams

    generator = ProceduralAssetGenerator(seed=42)
    variations = generator.generate_variations(
        base_mesh, "apple", num_variations=100,
        params=VariationParams(hue_range=30, scale_range=(0.85, 1.15))
    )
"""

import colorsys
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import trimesh for 3D mesh operations
try:
    import trimesh
    HAVE_TRIMESH = True
except ImportError:
    HAVE_TRIMESH = False
    print("[PROCEDURAL] WARNING: trimesh not available. Install with: pip install trimesh")


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class VariationParams:
    """
    Parameters controlling procedural asset variation.

    These parameters define the ranges and modes for generating variations
    from a base asset.
    """

    # Color variation (HSV color space)
    vary_color: bool = True
    hue_range: float = 30.0  # ±degrees (0-360)
    sat_range: float = 0.2   # ±saturation multiplier (0-1)
    val_range: float = 0.15  # ±value/brightness multiplier (0-1)

    # Scale variation
    vary_scale: bool = True
    scale_mode: str = "uniform"  # "uniform" or "non_uniform"
    scale_range: Tuple[float, float] = (0.8, 1.2)  # For uniform scaling

    # Non-uniform scale ranges (used if scale_mode = "non_uniform")
    scale_x_range: Tuple[float, float] = (0.8, 1.2)
    scale_y_range: Tuple[float, float] = (0.8, 1.2)
    scale_z_range: Tuple[float, float] = (0.8, 1.2)

    # Texture variation
    vary_texture: bool = False  # Disabled by default (requires additional deps)
    texture_mode: str = "procedural"  # "procedural", "style_transfer", "atlas"
    texture_size: int = 512  # Texture resolution

    # Material variation (PBR properties)
    vary_material: bool = True
    roughness_std: float = 0.15  # Standard deviation for roughness variation
    specular_range: float = 0.2  # ±specular range
    vary_metallic: bool = False  # Usually keep metallic fixed per material type

    # Physical properties
    update_physics: bool = True
    base_density: float = 600.0  # kg/m³ (default: plastic)
    friction_range: Tuple[float, float] = (0.3, 0.7)
    restitution_range: Tuple[float, float] = (0.1, 0.5)


@dataclass
class PBRMaterial:
    """
    Physically Based Rendering (PBR) material properties.

    Standard PBR material model used in modern 3D rendering (USD, glTF, etc.).
    """

    # Base properties
    base_color: np.ndarray  # RGB in [0, 1]

    # PBR properties
    metallic: float = 0.0      # 0 = dielectric (plastic, wood), 1 = metal
    roughness: float = 0.5     # 0 = mirror smooth, 1 = rough/matte
    specular: float = 0.5      # Specular reflection intensity

    # Advanced properties (optional)
    anisotropy: float = 0.0    # Directional roughness (e.g., brushed metal)
    clearcoat: float = 0.0     # Additional glossy layer (e.g., car paint)
    sheen: float = 0.0         # Soft reflection at grazing angles (e.g., fabric)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "base_color": self.base_color.tolist() if isinstance(self.base_color, np.ndarray) else self.base_color,
            "metallic": float(self.metallic),
            "roughness": float(self.roughness),
            "specular": float(self.specular),
            "anisotropy": float(self.anisotropy),
            "clearcoat": float(self.clearcoat),
            "sheen": float(self.sheen),
        }


@dataclass
class AssetVariation:
    """
    A single procedurally generated asset variation.

    Contains all information about a variation: geometry, appearance,
    material, and physical properties.
    """

    variation_id: str
    base_asset_id: str

    # Geometry
    mesh: Optional[Any] = None  # trimesh.Trimesh if available
    scale: float = 1.0
    scale_vector: Optional[np.ndarray] = None  # For non-uniform scaling

    # Appearance
    base_color: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8]))
    material: Optional[PBRMaterial] = None
    texture: Optional[np.ndarray] = None  # [H, W, 3] if generated

    # Physics properties
    mass: float = 1.0            # kg
    friction: float = 0.5         # Coefficient of friction
    restitution: float = 0.3      # Bounciness (0 = no bounce, 1 = perfect bounce)
    volume: float = 0.001        # m³

    # Metadata
    variation_params: Optional[VariationParams] = None
    generation_time: float = 0.0  # seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "variation_id": self.variation_id,
            "base_asset_id": self.base_asset_id,
            "scale": float(self.scale),
            "scale_vector": self.scale_vector.tolist() if self.scale_vector is not None else None,
            "base_color": self.base_color.tolist(),
            "material": self.material.to_dict() if self.material else None,
            "physics": {
                "mass": float(self.mass),
                "friction": float(self.friction),
                "restitution": float(self.restitution),
                "volume": float(self.volume),
            },
            "generation_time": float(self.generation_time),
        }


# =============================================================================
# Procedural Asset Generator
# =============================================================================


class ProceduralAssetGenerator:
    """
    Main procedural asset generation engine.

    Generates unlimited variations of 3D assets for domain randomization
    by varying color, scale, texture, material, and physical properties.

    Example:
        generator = ProceduralAssetGenerator(seed=42)

        # Generate 100 apple variations
        variations = generator.generate_variations(
            base_apple_mesh,
            "apple",
            num_variations=100,
            params=VariationParams(hue_range=30, scale_range=(0.85, 1.15))
        )

        # Save variations
        for var in variations:
            generator.save_variation(var, Path("output/apples"))
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.

        Args:
            seed: Random seed for reproducibility. If None, non-deterministic.
        """
        if seed is not None:
            np.random.seed(seed)

        self.generated_count = 0

    def generate_variations(
        self,
        base_asset: Any,  # trimesh.Trimesh
        base_asset_id: str,
        num_variations: int,
        params: VariationParams,
    ) -> List[AssetVariation]:
        """
        Generate multiple variations of a base asset.

        Args:
            base_asset: Base 3D mesh (trimesh.Trimesh)
            base_asset_id: Identifier for base asset (e.g., "apple", "bowl")
            num_variations: Number of variations to generate
            params: Variation parameters

        Returns:
            List of AssetVariation objects
        """
        if not HAVE_TRIMESH and base_asset is not None:
            raise RuntimeError("trimesh required for mesh operations. Install with: pip install trimesh")

        variations = []

        for i in range(num_variations):
            variation_id = f"{base_asset_id}_var_{self.generated_count:04d}"
            self.generated_count += 1

            import time
            start_time = time.time()

            # Generate single variation
            variation = self._generate_single_variation(
                base_asset,
                base_asset_id,
                variation_id,
                params,
            )

            variation.generation_time = time.time() - start_time
            variations.append(variation)

        return variations

    def _generate_single_variation(
        self,
        base_asset: Any,
        base_asset_id: str,
        variation_id: str,
        params: VariationParams,
    ) -> AssetVariation:
        """
        Generate a single asset variation.

        This is where the actual variation logic happens.
        """
        # Initialize variation
        variation = AssetVariation(
            variation_id=variation_id,
            base_asset_id=base_asset_id,
            variation_params=params,
        )

        # Copy base mesh if available
        if base_asset is not None and HAVE_TRIMESH:
            variation.mesh = base_asset.copy()
        else:
            variation.mesh = None

        # 1. SCALE VARIATION
        if params.vary_scale:
            if params.scale_mode == "uniform":
                # Uniform scaling (preserves aspect ratio)
                scale = np.random.uniform(*params.scale_range)
                variation.scale = scale

                if variation.mesh is not None:
                    variation.mesh.vertices *= scale

            else:  # non_uniform
                # Non-uniform scaling (changes proportions)
                scale_x = np.random.uniform(*params.scale_x_range)
                scale_y = np.random.uniform(*params.scale_y_range)
                scale_z = np.random.uniform(*params.scale_z_range)

                variation.scale_vector = np.array([scale_x, scale_y, scale_z])

                if variation.mesh is not None:
                    variation.mesh.vertices *= variation.scale_vector

                # Equivalent uniform scale (for volume calculation)
                variation.scale = np.cbrt(scale_x * scale_y * scale_z)

        # 2. COLOR VARIATION
        base_color = np.array([0.8, 0.8, 0.8])  # Default gray

        if params.vary_color:
            # Extract base color from mesh if available
            if base_asset is not None and hasattr(base_asset.visual, 'material'):
                try:
                    base_color_raw = base_asset.visual.material.baseColorFactor
                    if base_color_raw is not None:
                        base_color = np.array(base_color_raw[:3])
                except (AttributeError, IndexError):
                    pass

            # Vary color in HSV space
            variation.base_color = self._vary_color(base_color, params)
        else:
            variation.base_color = base_color

        # 3. MATERIAL VARIATION
        variation.material = self._vary_material(params)
        variation.material.base_color = variation.base_color

        # 4. TEXTURE VARIATION
        if params.vary_texture:
            variation.texture = self._generate_texture(params)

        # 5. PHYSICS PROPERTIES
        if params.update_physics:
            # Compute volume
            if variation.mesh is not None and hasattr(variation.mesh, 'volume'):
                try:
                    volume = variation.mesh.volume
                    if volume <= 0 or not variation.mesh.is_watertight:
                        # Use convex hull volume as fallback
                        volume = variation.mesh.convex_hull.volume
                except:
                    volume = 0.001  # Default 1 liter

                variation.volume = volume
            else:
                # Estimate from scale (assuming 1L base volume)
                variation.volume = 0.001 * (variation.scale ** 3)

            # Mass = density × volume
            variation.mass = params.base_density * variation.volume

            # Friction correlates with roughness
            # Smooth surfaces (low roughness) → low friction
            # Rough surfaces (high roughness) → high friction
            friction_base = (variation.material.roughness * 0.4 + 0.3)  # [0.3, 0.7]
            friction_noise = np.random.uniform(-0.1, 0.1)
            variation.friction = np.clip(friction_base + friction_noise, *params.friction_range)

            # Restitution (bounciness) inversely correlates with roughness
            # Smooth → more bouncy, Rough → less bouncy
            restitution_base = 0.5 * (1.0 - variation.material.roughness)  # [0, 0.5]
            restitution_noise = np.random.uniform(-0.05, 0.05)
            variation.restitution = np.clip(restitution_base + restitution_noise, *params.restitution_range)

        return variation

    def _vary_color(
        self,
        base_color: np.ndarray,
        params: VariationParams
    ) -> np.ndarray:
        """
        Vary color in HSV color space for perceptually realistic variations.

        HSV is more intuitive than RGB:
        - Hue: The actual color (0-360 degrees)
        - Saturation: Color intensity (0-100%)
        - Value: Brightness (0-100%)

        Args:
            base_color: Base RGB color in [0, 1]
            params: Variation parameters

        Returns:
            Varied RGB color in [0, 1]
        """
        # Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(*base_color)

        # 1. Vary HUE (the actual color)
        # Hue is circular: 0° = 360° (red)
        hue_shift = np.random.uniform(-params.hue_range / 360, params.hue_range / 360)
        h_new = (h + hue_shift) % 1.0  # Wrap around [0, 1]

        # 2. Vary SATURATION (color intensity)
        sat_factor = np.random.uniform(1 - params.sat_range, 1 + params.sat_range)
        s_new = np.clip(s * sat_factor, 0.0, 1.0)

        # 3. Vary VALUE (brightness)
        val_factor = np.random.uniform(1 - params.val_range, 1 + params.val_range)
        v_new = np.clip(v * val_factor, 0.0, 1.0)

        # Convert back to RGB
        r, g, b = colorsys.hsv_to_rgb(h_new, s_new, v_new)

        return np.array([r, g, b])

    def _vary_material(self, params: VariationParams) -> PBRMaterial:
        """
        Generate varied PBR material properties.

        Args:
            params: Variation parameters

        Returns:
            PBRMaterial with varied properties
        """
        # Start with default material
        material = PBRMaterial(
            base_color=np.array([0.8, 0.8, 0.8]),
            metallic=0.0,
            roughness=0.5,
            specular=0.5,
        )

        if params.vary_material:
            # Vary roughness (Gaussian distribution around 0.5)
            roughness_delta = np.random.normal(0, params.roughness_std)
            material.roughness = np.clip(0.5 + roughness_delta, 0.0, 1.0)

            # Vary specular
            specular_delta = np.random.uniform(-params.specular_range, params.specular_range)
            material.specular = np.clip(0.5 + specular_delta, 0.0, 1.0)

            # Vary metallic (if enabled)
            if params.vary_metallic:
                # Metallic is usually binary: 0 (dielectric) or 1 (metal)
                # Not realistic to have in-between values
                material.metallic = 1.0 if np.random.random() < 0.5 else 0.0

        return material

    def _generate_texture(self, params: VariationParams) -> np.ndarray:
        """
        Generate procedural texture.

        This is a simple placeholder. Can be enhanced with:
        - Perlin/Simplex noise
        - Neural style transfer
        - Texture atlas sampling

        Args:
            params: Variation parameters

        Returns:
            Texture image [H, W, 3] in [0, 1]
        """
        size = params.texture_size

        if params.texture_mode == "procedural":
            # Simple random noise texture (placeholder)
            # TODO: Implement Perlin/Simplex noise
            texture = np.random.rand(size, size, 3)

        else:
            # Placeholder for other modes
            texture = np.ones((size, size, 3)) * 0.5

        return texture

    def save_variation(
        self,
        variation: AssetVariation,
        output_dir: Path,
        save_mesh: bool = True,
        save_metadata: bool = True,
    ):
        """
        Save asset variation to disk.

        Args:
            variation: Asset variation to save
            output_dir: Output directory
            save_mesh: Whether to save mesh (.glb file)
            save_metadata: Whether to save metadata (.json file)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save mesh
        if save_mesh and variation.mesh is not None and HAVE_TRIMESH:
            mesh_path = output_dir / f"{variation.variation_id}.glb"

            # Apply material to mesh
            if variation.material is not None:
                try:
                    import trimesh.visual
                    material = trimesh.visual.material.PBRMaterial(
                        baseColorFactor=list(variation.base_color) + [1.0],
                        metallicFactor=variation.material.metallic,
                        roughnessFactor=variation.material.roughness,
                    )
                    variation.mesh.visual.material = material
                except Exception as e:
                    print(f"Warning: Could not apply material to mesh: {e}")

            # Export mesh
            variation.mesh.export(mesh_path)

        # Save metadata
        if save_metadata:
            metadata_path = output_dir / f"{variation.variation_id}.json"

            metadata = variation.to_dict()

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    def load_variation(self, variation_dir: Path, variation_id: str) -> AssetVariation:
        """
        Load previously saved variation.

        Args:
            variation_dir: Directory containing variation files
            variation_id: Variation identifier

        Returns:
            Loaded AssetVariation
        """
        # Load metadata
        metadata_path = variation_dir / f"{variation_id}.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load mesh if available
        mesh_path = variation_dir / f"{variation_id}.glb"
        mesh = None
        if mesh_path.exists() and HAVE_TRIMESH:
            mesh = trimesh.load(mesh_path)

        # Reconstruct variation
        variation = AssetVariation(
            variation_id=variation_id,
            base_asset_id=metadata["base_asset_id"],
            mesh=mesh,
            scale=metadata["scale"],
            scale_vector=np.array(metadata["scale_vector"]) if metadata.get("scale_vector") else None,
            base_color=np.array(metadata["base_color"]),
            mass=metadata["physics"]["mass"],
            friction=metadata["physics"]["friction"],
            restitution=metadata["physics"]["restitution"],
            volume=metadata["physics"]["volume"],
        )

        # Reconstruct material
        if metadata.get("material"):
            mat_data = metadata["material"]
            variation.material = PBRMaterial(
                base_color=np.array(mat_data["base_color"]),
                metallic=mat_data["metallic"],
                roughness=mat_data["roughness"],
                specular=mat_data["specular"],
            )

        return variation


# =============================================================================
# Utility Functions
# =============================================================================


def create_default_params(asset_type: str = "generic") -> VariationParams:
    """
    Create default variation parameters for common asset types.

    Args:
        asset_type: Type of asset ("food", "container", "tool", "generic")

    Returns:
        VariationParams configured for asset type
    """
    if asset_type == "food":
        # Food items: vary color significantly, moderate scale
        return VariationParams(
            vary_color=True,
            hue_range=45,  # Wide color variation
            sat_range=0.3,
            val_range=0.2,
            scale_range=(0.85, 1.15),
            roughness_std=0.1,  # Food is usually somewhat consistent
        )

    elif asset_type == "container":
        # Containers: moderate color, larger scale variation
        return VariationParams(
            vary_color=True,
            hue_range=30,
            sat_range=0.2,
            val_range=0.15,
            scale_range=(0.7, 1.3),  # Wider scale range
            roughness_std=0.2,
        )

    elif asset_type == "tool":
        # Tools: limited color (metal/plastic), small scale variation
        return VariationParams(
            vary_color=True,
            hue_range=15,  # Limited color variation
            sat_range=0.1,
            val_range=0.1,
            scale_range=(0.9, 1.1),  # Small scale variation
            roughness_std=0.15,
        )

    else:  # generic
        # Default balanced parameters
        return VariationParams()


# =============================================================================
# Example Usage
# =============================================================================


def main():
    """Example usage of procedural asset generator."""
    print("Procedural Asset Generator - Example\n")

    if not HAVE_TRIMESH:
        print("ERROR: trimesh not installed. Install with: pip install trimesh")
        print("Running in metadata-only mode...\n")

        # Can still generate metadata without meshes
        generator = ProceduralAssetGenerator(seed=42)

        params = VariationParams(
            vary_color=True,
            hue_range=30,
            scale_range=(0.85, 1.15),
        )

        # Generate variations (no mesh)
        variations = generator.generate_variations(
            None,  # No mesh
            "example_asset",
            num_variations=5,
            params=params,
        )

        print(f"Generated {len(variations)} variations (metadata only):\n")
        for var in variations:
            print(f"{var.variation_id}:")
            print(f"  Scale: {var.scale:.2f}")
            print(f"  Color: RGB({var.base_color[0]:.2f}, {var.base_color[1]:.2f}, {var.base_color[2]:.2f})")
            print(f"  Roughness: {var.material.roughness:.2f}")
            print(f"  Mass: {var.mass:.3f} kg")
            print()

        return

    # Full example with trimesh
    print("Creating example sphere mesh...")
    base_mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.05)  # 5cm radius sphere

    print(f"Base mesh: {len(base_mesh.vertices)} vertices, volume={base_mesh.volume:.6f} m³\n")

    # Create generator
    generator = ProceduralAssetGenerator(seed=42)

    # Generate variations
    params = create_default_params("food")  # Food-like variations
    print(f"Generating 10 variations with params:")
    print(f"  Hue range: ±{params.hue_range}°")
    print(f"  Scale range: {params.scale_range}")
    print()

    variations = generator.generate_variations(
        base_mesh,
        "sphere",
        num_variations=10,
        params=params,
    )

    print(f"Generated {len(variations)} variations:\n")
    for i, var in enumerate(variations):
        print(f"{i+1}. {var.variation_id}:")
        print(f"   Scale: {var.scale:.2f}×")
        print(f"   Color: RGB({var.base_color[0]:.2f}, {var.base_color[1]:.2f}, {var.base_color[2]:.2f})")
        print(f"   Material: roughness={var.material.roughness:.2f}, metallic={var.material.metallic:.2f}")
        print(f"   Physics: mass={var.mass:.3f}kg, friction={var.friction:.2f}, bounce={var.restitution:.2f}")
        print(f"   Generated in: {var.generation_time*1000:.1f}ms")
        print()

    # Save variations
    output_dir = Path("./variation_output")
    print(f"Saving variations to {output_dir}...")

    for var in variations:
        generator.save_variation(var, output_dir)

    print(f"✅ Saved {len(variations)} variations")
    print(f"\nOutput directory contains:")
    print(f"  - {len(variations)} .glb mesh files")
    print(f"  - {len(variations)} .json metadata files")


if __name__ == "__main__":
    main()
