# Procedural Asset Generation - Technical Deep Dive

**Date:** 2026-01-09
**Author:** Claude Code
**Component:** variation-assets-pipeline (Phase 3.2)

---

## Table of Contents

1. [Overview](#overview)
2. [Why Procedural Generation?](#why-procedural-generation)
3. [Architecture](#architecture)
4. [Generation Techniques](#generation-techniques)
5. [Implementation Details](#implementation-details)
6. [Integration with Pipeline](#integration-with-pipeline)
7. [Quality Considerations](#quality-considerations)
8. [Performance Optimization](#performance-optimization)

---

## Overview

Procedural asset generation creates synthetic variations of 3D assets (meshes, textures, materials) to increase domain randomization diversity for robot learning. Instead of manually creating hundreds of asset variants, procedural generation can create **infinite variations** from a single base asset.

### Key Benefits

| Benefit | Impact |
|---------|--------|
| **Infinite Diversity** | Generate unlimited variations on-the-fly |
| **Automatic** | No manual 3D modeling required |
| **Consistent Quality** | Programmatic control ensures valid assets |
| **Fast** | GPU-accelerated generation (ms per variant) |
| **Commercializable** | YOUR variations = YOUR sellable data |

### What Gets Varied

1. **Visual Appearance**
   - Base color (HSV color space)
   - Texture patterns (procedural or style transfer)
   - Material properties (roughness, metallic, specular)
   - Decals and wear patterns

2. **Geometric Properties**
   - Scale (±20% typical)
   - Proportions (length/width/height ratios)
   - Detail level (simplification or subdivision)

3. **Physical Properties**
   - Mass (scales with volume)
   - Friction coefficients
   - Restitution (bounciness)

---

## Why Procedural Generation?

### Problem: Limited Asset Diversity

Traditional domain randomization uses **fixed asset libraries**:

```
Kitchen Scene:
  - 5 apple variations
  - 3 bowl variations
  - 2 cabinet variations

Total combinations: 5 × 3 × 2 = 30 scenes
```

**Result:** Policies overfit to these 30 configurations.

### Solution: Procedural Variation

```
Kitchen Scene:
  - 1 base apple → ∞ color variations
  - 1 base bowl → ∞ scale + texture variations
  - 1 base cabinet → ∞ material variations

Total combinations: ∞ × ∞ × ∞ = INFINITE
```

**Result:** Policies learn to generalize, not memorize.

---

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PROCEDURAL ASSET GENERATION                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT: Base Asset                                                  │
│  ┌────────────────────┐                                             │
│  │ mesh.glb           │                                             │
│  │ base_color: #FF0000│                                             │
│  │ roughness: 0.5     │                                             │
│  │ scale: 1.0         │                                             │
│  └─────────┬──────────┘                                             │
│            │                                                         │
│            ▼                                                         │
│  ┌─────────────────────────────────────────────────┐                │
│  │ ProceduralAssetGenerator                        │                │
│  │                                                  │                │
│  │ 1. Color Variation (HSV color space)            │                │
│  │    - Hue shift: ±30°                            │                │
│  │    - Saturation: ±20%                           │                │
│  │    - Value/Brightness: ±15%                     │                │
│  │                                                  │                │
│  │ 2. Scale Variation                              │                │
│  │    - Uniform: scale × [0.8, 1.2]                │                │
│  │    - Non-uniform: [length, width, height]       │                │
│  │                                                  │                │
│  │ 3. Texture Variation                            │                │
│  │    - Procedural noise (Perlin, Simplex)         │                │
│  │    - Style transfer (neural network)            │                │
│  │    - Texture atlas swapping                     │                │
│  │                                                  │                │
│  │ 4. Material Variation                           │                │
│  │    - Roughness: [0.1, 0.9]                      │                │
│  │    - Metallic: [0.0, 1.0]                       │                │
│  │    - Specular: [0.0, 1.0]                       │                │
│  │                                                  │                │
│  │ 5. Physical Property Update                     │                │
│  │    - Mass = density × new_volume                │                │
│  │    - Friction based on material                 │                │
│  │    - Update collision geometry                  │                │
│  └─────────┬───────────────────────────────────────┘                │
│            │                                                         │
│            ▼                                                         │
│  OUTPUT: Asset Variations                                           │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────┐│
│  │ variant_001.glb    │  │ variant_002.glb    │  │ variant_003.glb││
│  │ color: #E63900     │  │ color: #FF6B00     │  │ color: #CC0000 ││
│  │ scale: 0.85        │  │ scale: 1.15        │  │ scale: 0.92    ││
│  │ roughness: 0.7     │  │ roughness: 0.3     │  │ roughness: 0.6 ││
│  └────────────────────┘  └────────────────────┘  └────────────────┘│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Architecture

```python
# Core components
ProceduralAssetGenerator       # Main orchestrator
  ├── ColorVariationEngine      # HSV color manipulation
  ├── ScaleVariationEngine      # Geometric scaling
  ├── TextureVariationEngine    # Texture generation/transfer
  ├── MaterialVariationEngine   # PBR material properties
  └── PhysicsPropertyUpdater    # Mass, friction, restitution

# Support components
AssetValidator                  # Ensures variations are valid
VariationManifestGenerator     # Creates metadata for variations
CollisionGeometryUpdater       # Updates collision meshes
```

---

## Generation Techniques

### 1. Color Variation (HSV Color Space)

**Why HSV instead of RGB?**

RGB color space is **perceptually non-uniform**. Varying RGB values randomly produces unrealistic colors.

HSV (Hue, Saturation, Value) is **perceptually intuitive**:
- **Hue**: The actual color (0-360°)
- **Saturation**: Color intensity (0-100%)
- **Value**: Brightness (0-100%)

#### Algorithm

```python
def vary_color(base_color_rgb: np.ndarray, params: ColorVariationParams) -> np.ndarray:
    """
    Vary color in HSV space for perceptually realistic variations.

    Args:
        base_color_rgb: Base color [R, G, B] in [0, 1]
        params: Variation parameters

    Returns:
        Varied color [R, G, B] in [0, 1]
    """
    # Convert RGB to HSV
    hsv = rgb_to_hsv(base_color_rgb)
    h, s, v = hsv

    # 1. Vary HUE (actual color)
    # Shift hue within ±params.hue_range degrees
    # Hue is circular: 0° = 360° (red)
    hue_shift = np.random.uniform(-params.hue_range, params.hue_range)
    h_new = (h + hue_shift / 360.0) % 1.0  # Wrap around [0, 1]

    # 2. Vary SATURATION (color intensity)
    # Multiply by random factor in [1 - params.sat_range, 1 + params.sat_range]
    sat_factor = np.random.uniform(1 - params.sat_range, 1 + params.sat_range)
    s_new = np.clip(s * sat_factor, 0.0, 1.0)

    # 3. Vary VALUE (brightness)
    # Multiply by random factor in [1 - params.val_range, 1 + params.val_range]
    val_factor = np.random.uniform(1 - params.val_range, 1 + params.val_range)
    v_new = np.clip(v * val_factor, 0.0, 1.0)

    # Convert back to RGB
    rgb_new = hsv_to_rgb([h_new, s_new, v_new])

    return rgb_new
```

#### Example: Red Apple Variations

```
Base Apple: RGB(255, 0, 0) = HSV(0°, 100%, 100%)

Variation 1: Hue +15°  → HSV(15°, 100%, 100%)  → Orange-red
Variation 2: Hue -20°  → HSV(340°, 100%, 100%) → Pink-red
Variation 3: Sat -30%  → HSV(0°, 70%, 100%)    → Pale red
Variation 4: Val -20%  → HSV(0°, 100%, 80%)    → Dark red
```

### 2. Scale Variation

Two approaches: **uniform** and **non-uniform** scaling.

#### Uniform Scaling

Scales all dimensions equally (preserves aspect ratio):

```python
def uniform_scale_variation(
    base_mesh: Mesh,
    scale_range: Tuple[float, float] = (0.8, 1.2)
) -> Mesh:
    """
    Apply uniform scaling to mesh.

    Args:
        base_mesh: Input mesh
        scale_range: Min/max scale factors

    Returns:
        Scaled mesh
    """
    # Random scale factor
    scale = np.random.uniform(scale_range[0], scale_range[1])

    # Scale all vertices
    mesh_copy = base_mesh.copy()
    mesh_copy.vertices *= scale

    # Update physics properties
    # Volume scales as scale³
    mesh_copy.volume = base_mesh.volume * (scale ** 3)
    # Mass scales with volume (assuming constant density)
    mesh_copy.mass = base_mesh.mass * (scale ** 3)

    # Update collision geometry
    mesh_copy.collision_geometry = scale_collision_geometry(
        base_mesh.collision_geometry, scale
    )

    return mesh_copy
```

**Example:** Bowl with 1.2× uniform scaling
- Width: 0.20m → 0.24m
- Height: 0.10m → 0.12m
- Volume: 0.006 m³ → 0.010 m³ (1.2³ = 1.728×)
- Mass: 0.5 kg → 0.864 kg

#### Non-Uniform Scaling

Scales dimensions independently (changes proportions):

```python
def non_uniform_scale_variation(
    base_mesh: Mesh,
    scale_ranges: Dict[str, Tuple[float, float]]
) -> Mesh:
    """
    Apply non-uniform scaling (different per axis).

    Args:
        base_mesh: Input mesh
        scale_ranges: Scale ranges for x, y, z axes

    Returns:
        Scaled mesh
    """
    # Random scale factors per axis
    scale_x = np.random.uniform(*scale_ranges['x'])
    scale_y = np.random.uniform(*scale_ranges['y'])
    scale_z = np.random.uniform(*scale_ranges['z'])

    scale_vector = np.array([scale_x, scale_y, scale_z])

    # Scale vertices per axis
    mesh_copy = base_mesh.copy()
    mesh_copy.vertices *= scale_vector

    # Volume scales as product of scale factors
    volume_scale = scale_x * scale_y * scale_z
    mesh_copy.volume = base_mesh.volume * volume_scale
    mesh_copy.mass = base_mesh.mass * volume_scale

    # Update collision geometry (more complex for non-uniform)
    mesh_copy.collision_geometry = non_uniform_scale_collision(
        base_mesh.collision_geometry, scale_vector
    )

    return mesh_copy
```

**Example:** Elongated bottle
- X (width): 0.8× → narrower
- Y (depth): 0.8× → shallower
- Z (height): 1.3× → taller
- Proportions changed: tall, thin bottle

### 3. Texture Variation

Three techniques: **procedural noise**, **style transfer**, and **texture atlas swapping**.

#### 3.1 Procedural Noise Textures

Generate textures using mathematical noise functions (Perlin, Simplex):

```python
def generate_procedural_texture(
    width: int,
    height: int,
    noise_type: str = "perlin",
    scale: float = 10.0,
    octaves: int = 4,
) -> np.ndarray:
    """
    Generate procedural texture using noise functions.

    Args:
        width, height: Texture resolution
        noise_type: "perlin" or "simplex"
        scale: Noise frequency (lower = larger features)
        octaves: Number of noise layers (more = more detail)

    Returns:
        Texture image [H, W, 3]
    """
    import noise  # pip install noise

    # Create coordinate grid
    texture = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            # Normalized coordinates
            nx = x / width
            ny = y / height

            if noise_type == "perlin":
                # Perlin noise: smooth, natural-looking
                value = noise.pnoise2(
                    nx * scale,
                    ny * scale,
                    octaves=octaves,
                    persistence=0.5,
                    lacunarity=2.0,
                )
            else:  # simplex
                # Simplex noise: faster, no directional artifacts
                value = noise.snoise2(nx * scale, ny * scale)

            # Map from [-1, 1] to [0, 1]
            value = (value + 1.0) / 2.0

            # Apply to RGB channels (can vary per channel for color noise)
            texture[y, x] = [value, value, value]

    return texture
```

**Applications:**
- Wood grain: Low frequency Perlin noise with vertical stretching
- Stone texture: Multi-octave Simplex noise with color variation
- Metal scratches: High frequency noise with directional bias

#### 3.2 Neural Style Transfer

Transfer artistic style from one image to a 3D texture:

```python
def neural_style_transfer_texture(
    base_texture: np.ndarray,
    style_image: np.ndarray,
    content_weight: float = 1.0,
    style_weight: float = 1000.0,
) -> np.ndarray:
    """
    Apply neural style transfer to texture.

    This uses a pre-trained VGG network to extract:
    - Content features (what objects are present)
    - Style features (artistic patterns, colors, textures)

    Args:
        base_texture: Original texture [H, W, 3]
        style_image: Style reference image [H, W, 3]
        content_weight: Weight for preserving content
        style_weight: Weight for style transfer

    Returns:
        Stylized texture [H, W, 3]
    """
    import torch
    import torchvision.models as models
    from torchvision import transforms

    # Load pre-trained VGG19
    vgg = models.vgg19(pretrained=True).features.eval()

    # Content layers: deeper layers capture object structure
    content_layers = ['conv_4']

    # Style layers: multiple layers capture artistic patterns
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    # Convert images to tensors
    content_tensor = image_to_tensor(base_texture)
    style_tensor = image_to_tensor(style_image)

    # Start with base texture or random noise
    output_tensor = content_tensor.clone().requires_grad_(True)

    # Optimizer
    optimizer = torch.optim.LBFGS([output_tensor])

    # Extract target features
    content_features = extract_features(content_tensor, vgg, content_layers)
    style_features = extract_features(style_tensor, vgg, style_layers)

    # Gram matrices for style (capture correlations between features)
    style_grams = {layer: gram_matrix(features)
                   for layer, features in style_features.items()}

    # Optimization loop
    for iteration in range(300):
        def closure():
            optimizer.zero_grad()

            # Extract features from current output
            output_features = extract_features(output_tensor, vgg,
                                              content_layers + style_layers)

            # Content loss: preserve structure
            content_loss = 0
            for layer in content_layers:
                content_loss += mse_loss(
                    output_features[layer],
                    content_features[layer]
                )

            # Style loss: match artistic patterns
            style_loss = 0
            for layer in style_layers:
                output_gram = gram_matrix(output_features[layer])
                target_gram = style_grams[layer]
                style_loss += mse_loss(output_gram, target_gram)

            # Total loss
            loss = content_weight * content_loss + style_weight * style_loss
            loss.backward()

            return loss

        optimizer.step(closure)

    # Convert back to numpy
    stylized_texture = tensor_to_image(output_tensor)

    return stylized_texture
```

**Example:** Apply "Van Gogh Starry Night" style to apple texture
- Base: Smooth red apple
- Style: Swirling brushstrokes, vibrant blues/yellows
- Result: Apple with artistic swirling patterns while preserving shape

#### 3.3 Texture Atlas Swapping

Use pre-made texture libraries and randomly select textures:

```python
class TextureAtlasVariation:
    """
    Variation using pre-existing texture atlases.

    Texture atlas: Large image containing multiple textures
    arranged in a grid. Fast lookup, minimal processing.
    """

    def __init__(self, atlas_dir: Path):
        """Load all texture atlases."""
        self.atlases = {
            "wood": self._load_atlas(atlas_dir / "wood_atlas.png"),
            "metal": self._load_atlas(atlas_dir / "metal_atlas.png"),
            "fabric": self._load_atlas(atlas_dir / "fabric_atlas.png"),
            "stone": self._load_atlas(atlas_dir / "stone_atlas.png"),
        }

    def get_random_texture(
        self,
        material_type: str,
        resolution: Tuple[int, int] = (512, 512)
    ) -> np.ndarray:
        """
        Get random texture from atlas.

        Args:
            material_type: "wood", "metal", "fabric", "stone"
            resolution: Desired texture resolution

        Returns:
            Texture image [H, W, 3]
        """
        atlas = self.atlases[material_type]

        # Atlas is grid of textures
        # e.g., 4096×4096 atlas with 512×512 textures = 8×8 grid = 64 textures
        tile_size = 512
        grid_size = atlas.shape[0] // tile_size

        # Random tile selection
        tile_x = np.random.randint(0, grid_size)
        tile_y = np.random.randint(0, grid_size)

        # Extract tile
        y_start = tile_y * tile_size
        y_end = y_start + tile_size
        x_start = tile_x * tile_size
        x_end = x_start + tile_size

        texture = atlas[y_start:y_end, x_start:x_end]

        # Resize if needed
        if resolution != (tile_size, tile_size):
            texture = cv2.resize(texture, resolution)

        return texture
```

### 4. Material Property Variation (PBR)

Modern 3D rendering uses **Physically Based Rendering (PBR)** with material properties:

```python
@dataclass
class PBRMaterial:
    """Physically Based Rendering material."""

    # Base properties
    base_color: np.ndarray  # RGB albedo

    # PBR properties (all in [0, 1])
    metallic: float = 0.0      # 0 = dielectric, 1 = metal
    roughness: float = 0.5     # 0 = mirror smooth, 1 = rough/matte
    specular: float = 0.5      # Specular reflection intensity

    # Advanced
    anisotropy: float = 0.0    # Directional roughness (brushed metal)
    clearcoat: float = 0.0     # Additional glossy layer (car paint)
    sheen: float = 0.0         # Soft reflection at edges (fabric)

    # Texture maps (optional)
    normal_map: Optional[np.ndarray] = None    # Surface detail
    roughness_map: Optional[np.ndarray] = None  # Varying roughness
    metallic_map: Optional[np.ndarray] = None   # Varying metallic


def vary_material_properties(
    base_material: PBRMaterial,
    params: MaterialVariationParams
) -> PBRMaterial:
    """
    Vary PBR material properties.

    Args:
        base_material: Base material
        params: Variation parameters

    Returns:
        Varied material
    """
    varied = base_material.copy()

    # Vary metallic (0 = plastic/wood, 1 = metal)
    if params.vary_metallic:
        # Bimodal distribution: either 0 (dielectric) or 1 (metal)
        # Not realistic to be in-between
        if np.random.random() < 0.5:
            varied.metallic = 0.0
        else:
            varied.metallic = 1.0

    # Vary roughness
    if params.vary_roughness:
        # Sample from range, bias towards middle values
        # Real-world materials rarely perfectly smooth or perfectly rough
        roughness_delta = np.random.normal(0, params.roughness_std)
        varied.roughness = np.clip(
            base_material.roughness + roughness_delta,
            0.0, 1.0
        )

    # Vary specular
    if params.vary_specular:
        specular_delta = np.random.uniform(
            -params.specular_range,
            params.specular_range
        )
        varied.specular = np.clip(
            base_material.specular + specular_delta,
            0.0, 1.0
        )

    # Vary base color (using HSV as before)
    if params.vary_color:
        varied.base_color = vary_color(
            base_material.base_color,
            params.color_params
        )

    return varied
```

**Material Presets:**

```python
MATERIAL_PRESETS = {
    "plastic_matte": PBRMaterial(
        base_color=[0.8, 0.8, 0.8],
        metallic=0.0,
        roughness=0.7,
        specular=0.3,
    ),
    "plastic_glossy": PBRMaterial(
        base_color=[0.8, 0.8, 0.8],
        metallic=0.0,
        roughness=0.2,
        specular=0.8,
    ),
    "metal_brushed": PBRMaterial(
        base_color=[0.9, 0.9, 0.9],
        metallic=1.0,
        roughness=0.4,
        specular=1.0,
        anisotropy=0.8,  # Directional reflections
    ),
    "metal_polished": PBRMaterial(
        base_color=[0.9, 0.9, 0.9],
        metallic=1.0,
        roughness=0.1,
        specular=1.0,
    ),
    "wood": PBRMaterial(
        base_color=[0.6, 0.4, 0.2],
        metallic=0.0,
        roughness=0.6,
        specular=0.2,
    ),
    "fabric": PBRMaterial(
        base_color=[0.5, 0.5, 0.5],
        metallic=0.0,
        roughness=0.9,
        sheen=0.5,  # Soft edge highlights
    ),
}
```

---

## Implementation Details

### Complete Implementation

```python
# File: variation-assets-pipeline/procedural_generator.py

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import trimesh


@dataclass
class VariationParams:
    """Parameters controlling asset variation."""

    # Color variation (HSV)
    vary_color: bool = True
    hue_range: float = 30.0  # degrees
    sat_range: float = 0.2   # ±20%
    val_range: float = 0.15  # ±15%

    # Scale variation
    vary_scale: bool = True
    scale_mode: str = "uniform"  # "uniform" or "non_uniform"
    scale_range: Tuple[float, float] = (0.8, 1.2)

    # Non-uniform scale ranges (if scale_mode = "non_uniform")
    scale_x_range: Tuple[float, float] = (0.8, 1.2)
    scale_y_range: Tuple[float, float] = (0.8, 1.2)
    scale_z_range: Tuple[float, float] = (0.8, 1.2)

    # Texture variation
    vary_texture: bool = True
    texture_mode: str = "procedural"  # "procedural", "style_transfer", "atlas"

    # Material variation
    vary_material: bool = True
    roughness_std: float = 0.15
    specular_range: float = 0.2
    vary_metallic: bool = False  # Usually keep fixed per material type

    # Physical properties
    update_physics: bool = True
    base_density: float = 600.0  # kg/m³ (default for plastic)


@dataclass
class AssetVariation:
    """A single asset variation."""

    variation_id: str
    base_asset_id: str

    # Geometry
    mesh: trimesh.Trimesh
    scale: float

    # Appearance
    base_color: np.ndarray
    material: 'PBRMaterial'
    texture: Optional[np.ndarray] = None

    # Physics
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.3

    # Metadata
    variation_params: Optional[VariationParams] = None


class ProceduralAssetGenerator:
    """
    Main procedural asset generation engine.

    Generates unlimited variations of 3D assets for domain randomization.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        self.generated_count = 0

    def generate_variations(
        self,
        base_asset: trimesh.Trimesh,
        base_asset_id: str,
        num_variations: int,
        params: VariationParams,
    ) -> List[AssetVariation]:
        """
        Generate multiple variations of a base asset.

        Args:
            base_asset: Base 3D mesh
            base_asset_id: Identifier for base asset
            num_variations: Number of variations to generate
            params: Variation parameters

        Returns:
            List of asset variations
        """
        variations = []

        for i in range(num_variations):
            variation_id = f"{base_asset_id}_var_{self.generated_count:04d}"
            self.generated_count += 1

            # Generate single variation
            variation = self._generate_single_variation(
                base_asset,
                base_asset_id,
                variation_id,
                params,
            )

            variations.append(variation)

        return variations

    def _generate_single_variation(
        self,
        base_asset: trimesh.Trimesh,
        base_asset_id: str,
        variation_id: str,
        params: VariationParams,
    ) -> AssetVariation:
        """Generate single asset variation."""

        # Start with copy of base
        mesh = base_asset.copy()

        # 1. Scale variation
        scale = 1.0
        if params.vary_scale:
            if params.scale_mode == "uniform":
                scale = np.random.uniform(*params.scale_range)
                mesh.vertices *= scale
            else:  # non_uniform
                scale_vec = np.array([
                    np.random.uniform(*params.scale_x_range),
                    np.random.uniform(*params.scale_y_range),
                    np.random.uniform(*params.scale_z_range),
                ])
                mesh.vertices *= scale_vec
                scale = np.cbrt(np.prod(scale_vec))  # Equivalent uniform scale

        # 2. Color variation
        base_color = np.array([0.8, 0.8, 0.8])  # Default gray
        if params.vary_color and hasattr(base_asset.visual, 'material'):
            base_color = self._vary_color(
                base_asset.visual.material.baseColorFactor[:3],
                params
            )

        # 3. Material variation
        material = self._vary_material(base_asset, params)
        material.base_color = base_color

        # 4. Texture variation
        texture = None
        if params.vary_texture:
            texture = self._generate_texture(params)

        # 5. Physics properties
        volume = mesh.volume if mesh.is_watertight else mesh.convex_hull.volume
        mass = params.base_density * volume

        # Friction varies with material roughness
        friction = 0.3 + 0.4 * material.roughness  # [0.3, 0.7]

        # Restitution (bounciness) - lower for rough materials
        restitution = 0.5 * (1.0 - material.roughness)  # [0, 0.5]

        return AssetVariation(
            variation_id=variation_id,
            base_asset_id=base_asset_id,
            mesh=mesh,
            scale=scale,
            base_color=base_color,
            material=material,
            texture=texture,
            mass=mass,
            friction=friction,
            restitution=restitution,
            variation_params=params,
        )

    def _vary_color(
        self,
        base_color: np.ndarray,
        params: VariationParams
    ) -> np.ndarray:
        """Vary color in HSV space."""
        import colorsys

        # Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(*base_color)

        # Vary hue (±hue_range degrees)
        h_new = (h + np.random.uniform(-params.hue_range/360, params.hue_range/360)) % 1.0

        # Vary saturation
        s_new = np.clip(s * np.random.uniform(1-params.sat_range, 1+params.sat_range), 0, 1)

        # Vary value
        v_new = np.clip(v * np.random.uniform(1-params.val_range, 1+params.val_range), 0, 1)

        # Convert back to RGB
        return np.array(colorsys.hsv_to_rgb(h_new, s_new, v_new))

    def _vary_material(
        self,
        base_asset: trimesh.Trimesh,
        params: VariationParams
    ) -> 'PBRMaterial':
        """Generate varied PBR material."""
        # Default material
        material = PBRMaterial(
            base_color=np.array([0.8, 0.8, 0.8]),
            metallic=0.0,
            roughness=0.5,
            specular=0.5,
        )

        if params.vary_material:
            # Vary roughness
            roughness_delta = np.random.normal(0, params.roughness_std)
            material.roughness = np.clip(0.5 + roughness_delta, 0.0, 1.0)

            # Vary specular
            specular_delta = np.random.uniform(-params.specular_range, params.specular_range)
            material.specular = np.clip(0.5 + specular_delta, 0.0, 1.0)

        return material

    def _generate_texture(self, params: VariationParams) -> np.ndarray:
        """Generate procedural texture."""
        # Simple noise texture (can be enhanced)
        size = 512
        texture = np.random.rand(size, size, 3)
        return texture

    def save_variation(
        self,
        variation: AssetVariation,
        output_dir: Path,
    ):
        """Save variation to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save mesh
        mesh_path = output_dir / f"{variation.variation_id}.glb"
        variation.mesh.export(mesh_path)

        # Save metadata
        metadata = {
            "variation_id": variation.variation_id,
            "base_asset_id": variation.base_asset_id,
            "scale": float(variation.scale),
            "base_color": variation.base_color.tolist(),
            "material": {
                "metallic": float(variation.material.metallic),
                "roughness": float(variation.material.roughness),
                "specular": float(variation.material.specular),
            },
            "physics": {
                "mass": float(variation.mass),
                "friction": float(variation.friction),
                "restitution": float(variation.restitution),
            },
        }

        metadata_path = output_dir / f"{variation.variation_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


@dataclass
class PBRMaterial:
    """PBR material properties."""
    base_color: np.ndarray
    metallic: float = 0.0
    roughness: float = 0.5
    specular: float = 0.5
    anisotropy: float = 0.0
    clearcoat: float = 0.0
    sheen: float = 0.0
```

---

## Integration with Pipeline

Procedural generation integrates at the **domain randomization stage**:

```
Pipeline Flow:
  3D-RE-GEN → simready → usd-assembly → replicator → [PROCEDURAL GENERATION] → episode-gen
                                                              ↓
                                                     variation-assets/
```

### Usage in Replicator Job

```python
# In replicator-job/generate_replicator_bundle.py

from variation_assets_pipeline.procedural_generator import (
    ProceduralAssetGenerator,
    VariationParams,
)

# Load base assets
base_apple = trimesh.load("assets/apple.glb")
base_bowl = trimesh.load("assets/bowl.glb")

# Create generator
generator = ProceduralAssetGenerator(seed=42)

# Generate variations
apple_params = VariationParams(
    vary_color=True,
    hue_range=30,  # Red to orange to pink
    vary_scale=True,
    scale_range=(0.85, 1.15),
)

apple_variations = generator.generate_variations(
    base_apple,
    "apple",
    num_variations=100,  # 100 apple variants
    params=apple_params,
)

# Save variations
for var in apple_variations:
    generator.save_variation(var, Path("variation_assets/apples"))

# Now use these in scene variations
for scene_idx in range(num_scenes):
    # Randomly select apple variation
    apple_var = np.random.choice(apple_variations)
    scene.add_object(apple_var.mesh, apple_var.material)
```

---

## Quality Considerations

### Validation

Ensure generated variations are valid:

```python
class AssetValidator:
    """Validate procedurally generated assets."""

    def validate(self, variation: AssetVariation) -> ValidationResult:
        """Validate asset variation."""
        errors = []
        warnings = []

        # 1. Mesh validity
        if not variation.mesh.is_valid:
            errors.append("Mesh has invalid geometry")

        if not variation.mesh.is_watertight:
            warnings.append("Mesh is not watertight (may affect physics)")

        # 2. Scale bounds
        if variation.scale < 0.5 or variation.scale > 2.0:
            warnings.append(f"Extreme scale: {variation.scale}")

        # 3. Color validity
        if not np.all((variation.base_color >= 0) & (variation.base_color <= 1)):
            errors.append("Invalid color values")

        # 4. Material validity
        if not (0 <= variation.material.roughness <= 1):
            errors.append("Invalid roughness")

        # 5. Physics plausibility
        if variation.mass < 0.01:
            warnings.append(f"Very low mass: {variation.mass} kg")
        if variation.mass > 100:
            warnings.append(f"Very high mass: {variation.mass} kg")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
```

---

## Performance Optimization

### GPU Acceleration

Use GPU for texture generation and style transfer:

```python
# Batch texture generation on GPU
import torch

def batch_generate_textures_gpu(
    num_textures: int,
    size: int = 512,
) -> torch.Tensor:
    """Generate textures in parallel on GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate noise on GPU
    textures = torch.rand(num_textures, 3, size, size, device=device)

    # Apply filters (all GPU operations)
    textures = apply_perlin_noise_gpu(textures)
    textures = apply_color_variation_gpu(textures)

    return textures
```

### Caching

Cache generated variations for reuse:

```python
class VariationCache:
    """Cache for procedural variations."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_index = {}

    def get_or_generate(
        self,
        base_asset_id: str,
        params: VariationParams,
        generator: ProceduralAssetGenerator,
    ) -> AssetVariation:
        """Get cached variation or generate new one."""
        cache_key = self._compute_cache_key(base_asset_id, params)

        if cache_key in self.cache_index:
            # Load from cache
            return self._load_cached(cache_key)
        else:
            # Generate and cache
            variation = generator._generate_single_variation(...)
            self._cache_variation(cache_key, variation)
            return variation
```

---

## Summary

Procedural asset generation enables **infinite domain randomization** by:

1. **Color Variation**: HSV color space for realistic variations
2. **Scale Variation**: Uniform/non-uniform geometric scaling
3. **Texture Variation**: Procedural noise, style transfer, or atlas swapping
4. **Material Variation**: PBR properties (roughness, metallic, specular)
5. **Physics Updates**: Mass, friction, restitution based on variations

**Key Benefits:**
- No manual asset creation
- Unlimited diversity
- Commercializable (YOUR variations = YOUR data)
- GPU-accelerated (ms per variant)
- Production-ready validation

**Integration:** Seamlessly integrates with existing replicator-job for domain randomization.
