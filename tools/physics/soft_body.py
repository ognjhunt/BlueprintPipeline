"""Soft Body Physics for Deformable Objects.

Enables simulation of cloth, ropes, soft containers, and other deformable materials
in the BlueprintPipeline. Extends the rigid body physics system to support:

- Cloth simulation (shirts, towels, tablecloths)
- Rope/cable physics (cables, strings, chains)
- Soft containers (bags, pouches, balloons)
- Deformable objects (soft toys, cushions, sponges)

Uses position-based dynamics (PBD) or finite element method (FEM) depending on
the object type and quality requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SoftBodyType(str, Enum):
    """Types of soft body objects."""
    CLOTH = "cloth"  # 2D deformable surface (shirts, towels, flags)
    ROPE = "rope"  # 1D deformable curve (cables, chains, strings)
    SOFT_CONTAINER = "soft_container"  # Volumetric soft body (bags, balloons)
    DEFORMABLE = "deformable"  # Generic deformable (soft toys, cushions)
    FLUID = "fluid"  # Fluid simulation (water, oil) - advanced


class DeformableMaterial(str, Enum):
    """Deformable material types."""
    COTTON = "cotton"  # Towels, t-shirts
    SILK = "silk"  # Smooth cloth
    DENIM = "denim"  # Jeans, thick fabric
    RUBBER = "rubber"  # Elastic materials
    FOAM = "foam"  # Cushions, sponges
    LEATHER = "leather"  # Leather materials
    PLASTIC_SOFT = "plastic_soft"  # Soft plastics
    ROPE_NATURAL = "rope_natural"  # Natural fiber ropes
    ROPE_SYNTHETIC = "rope_synthetic"  # Synthetic ropes
    METAL_CHAIN = "metal_chain"  # Metal chains


@dataclass
class SoftBodyProperties:
    """Physics properties for soft body objects."""

    # Basic properties
    soft_body_type: SoftBodyType = SoftBodyType.DEFORMABLE
    material: DeformableMaterial = DeformableMaterial.COTTON

    # Mechanical properties
    stiffness: float = 1.0  # 0-10, higher = more rigid
    damping: float = 0.5  # 0-1, energy dissipation
    elasticity: float = 0.8  # 0-1, elastic recovery

    # Structural properties
    bend_resistance: float = 0.5  # 0-1, resistance to bending
    stretch_resistance: float = 0.7  # 0-1, resistance to stretching
    compression_resistance: float = 0.5  # 0-1, resistance to compression

    # Collision properties
    thickness: float = 0.002  # meters, collision thickness
    self_collision_enabled: bool = True  # Enable self-collision
    collision_margin: float = 0.001  # meters

    # Simulation quality
    particle_resolution: int = 10  # Particles per meter
    solver_iterations: int = 5  # PBD solver iterations
    substeps: int = 2  # Simulation substeps per frame

    # Mass properties
    mass_per_area: Optional[float] = None  # kg/m^2 for cloth
    mass_per_length: Optional[float] = None  # kg/m for rope
    total_mass: Optional[float] = None  # kg for volumetric

    # Constraints
    fixed_vertices: List[int] = field(default_factory=list)  # Fixed vertex indices
    attachment_points: List[Tuple[int, Tuple[float, float, float]]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "soft_body_type": self.soft_body_type.value,
            "material": self.material.value,
            "stiffness": self.stiffness,
            "damping": self.damping,
            "elasticity": self.elasticity,
            "bend_resistance": self.bend_resistance,
            "stretch_resistance": self.stretch_resistance,
            "compression_resistance": self.compression_resistance,
            "thickness": self.thickness,
            "self_collision_enabled": self.self_collision_enabled,
            "collision_margin": self.collision_margin,
            "particle_resolution": self.particle_resolution,
            "solver_iterations": self.solver_iterations,
            "substeps": self.substeps,
            "mass_per_area": self.mass_per_area,
            "mass_per_length": self.mass_per_length,
            "total_mass": self.total_mass,
        }


class SoftBodyPhysics:
    """Soft body physics manager.

    Detects deformable objects and assigns appropriate soft body physics properties.

    Example:
        physics = SoftBodyPhysics()

        # Detect if object should be soft body
        obj_data = {
            "category": "towel",
            "material_name": "cotton",
        }

        if physics.is_soft_body(obj_data):
            props = physics.generate_soft_body_properties(obj_data)
            print(f"Soft body type: {props.soft_body_type}")
            print(f"Stiffness: {props.stiffness}")
    """

    # Material property mappings
    MATERIAL_PROPERTIES = {
        DeformableMaterial.COTTON: {
            "stiffness": 2.0,
            "damping": 0.4,
            "elasticity": 0.6,
            "bend_resistance": 0.3,
            "stretch_resistance": 0.5,
            "mass_per_area": 0.2,  # kg/m^2
        },
        DeformableMaterial.SILK: {
            "stiffness": 1.0,
            "damping": 0.2,
            "elasticity": 0.8,
            "bend_resistance": 0.1,
            "stretch_resistance": 0.4,
            "mass_per_area": 0.05,
        },
        DeformableMaterial.DENIM: {
            "stiffness": 4.0,
            "damping": 0.6,
            "elasticity": 0.4,
            "bend_resistance": 0.7,
            "stretch_resistance": 0.8,
            "mass_per_area": 0.5,
        },
        DeformableMaterial.RUBBER: {
            "stiffness": 3.0,
            "damping": 0.8,
            "elasticity": 0.95,
            "bend_resistance": 0.5,
            "stretch_resistance": 0.9,
            "compression_resistance": 0.7,
        },
        DeformableMaterial.FOAM: {
            "stiffness": 1.5,
            "damping": 0.9,
            "elasticity": 0.7,
            "compression_resistance": 0.3,
        },
        DeformableMaterial.ROPE_NATURAL: {
            "stiffness": 5.0,
            "damping": 0.5,
            "elasticity": 0.3,
            "stretch_resistance": 0.9,
            "mass_per_length": 0.05,  # kg/m
        },
        DeformableMaterial.ROPE_SYNTHETIC: {
            "stiffness": 6.0,
            "damping": 0.4,
            "elasticity": 0.5,
            "stretch_resistance": 0.95,
            "mass_per_length": 0.03,
        },
    }

    # Object category → soft body type mapping
    SOFT_BODY_CATEGORIES = {
        SoftBodyType.CLOTH: [
            "towel", "cloth", "fabric", "shirt", "t-shirt", "tshirt",
            "pants", "jeans", "jacket", "coat", "blanket", "sheet",
            "tablecloth", "napkin", "curtain", "flag", "banner",
        ],
        SoftBodyType.ROPE: [
            "rope", "cable", "cord", "string", "wire", "chain",
            "belt", "strap", "ribbon", "hose", "tube",
        ],
        SoftBodyType.SOFT_CONTAINER: [
            "bag", "pouch", "sack", "backpack", "purse", "tote",
            "balloon", "inflatable", "pillow_case",
        ],
        SoftBodyType.DEFORMABLE: [
            "pillow", "cushion", "stuffed_toy", "plush", "sponge",
            "foam_block", "stress_ball", "soft_toy", "bean_bag",
        ],
    }

    def __init__(self, enable_logging: bool = True):
        """Initialize soft body physics.

        Args:
            enable_logging: Whether to log detections
        """
        self.enable_logging = enable_logging

    def log(self, msg: str) -> None:
        """Log if enabled."""
        if self.enable_logging:
            logger.info(f"[SOFT_BODY] {msg}")

    def is_soft_body(self, obj_data: Dict[str, Any]) -> bool:
        """Detect if object should use soft body physics.

        Args:
            obj_data: Object data with category, material_name, etc.

        Returns:
            True if object should be soft body
        """
        category = (obj_data.get("category") or "").lower()
        material = (obj_data.get("material_name") or "").lower()

        # Check category
        for soft_type, categories in self.SOFT_BODY_CATEGORIES.items():
            for cat in categories:
                if cat in category:
                    return True

        # Check material keywords
        soft_materials = ["fabric", "cloth", "textile", "rope", "cable", "foam", "sponge"]
        for mat in soft_materials:
            if mat in material:
                return True

        return False

    def detect_soft_body_type(self, obj_data: Dict[str, Any]) -> Optional[SoftBodyType]:
        """Detect soft body type from object data.

        Args:
            obj_data: Object data

        Returns:
            SoftBodyType if detected, else None
        """
        if not self.is_soft_body(obj_data):
            return None

        category = (obj_data.get("category") or "").lower()

        # Match category to soft body type
        for soft_type, categories in self.SOFT_BODY_CATEGORIES.items():
            for cat in categories:
                if cat in category:
                    return soft_type

        # Default to deformable
        return SoftBodyType.DEFORMABLE

    def detect_material(self, obj_data: Dict[str, Any]) -> DeformableMaterial:
        """Detect deformable material from object data.

        Args:
            obj_data: Object data

        Returns:
            DeformableMaterial
        """
        material = (obj_data.get("material_name") or "").lower()
        category = (obj_data.get("category") or "").lower()

        # Material keyword matching
        if "cotton" in material or "towel" in category:
            return DeformableMaterial.COTTON
        elif "silk" in material:
            return DeformableMaterial.SILK
        elif "denim" in material or "jeans" in category:
            return DeformableMaterial.DENIM
        elif "rubber" in material or "elastic" in material:
            return DeformableMaterial.RUBBER
        elif "foam" in material or "cushion" in category or "pillow" in category:
            return DeformableMaterial.FOAM
        elif "leather" in material:
            return DeformableMaterial.LEATHER
        elif "rope" in category or "cord" in category:
            if "synthetic" in material or "nylon" in material:
                return DeformableMaterial.ROPE_SYNTHETIC
            else:
                return DeformableMaterial.ROPE_NATURAL
        elif "chain" in category or "metal" in material:
            return DeformableMaterial.METAL_CHAIN

        # Default to cotton
        return DeformableMaterial.COTTON

    def generate_soft_body_properties(
        self,
        obj_data: Dict[str, Any],
        bounds: Optional[Dict[str, Any]] = None,
    ) -> SoftBodyProperties:
        """Generate soft body properties for an object.

        Args:
            obj_data: Object data with category, material_name, etc.
            bounds: Optional object bounds (size_m, volume_m3)

        Returns:
            SoftBodyProperties
        """
        # Detect type and material
        soft_type = self.detect_soft_body_type(obj_data) or SoftBodyType.DEFORMABLE
        material = self.detect_material(obj_data)

        # Get base material properties
        mat_props = self.MATERIAL_PROPERTIES.get(material, {})

        # Create properties
        props = SoftBodyProperties(
            soft_body_type=soft_type,
            material=material,
            stiffness=mat_props.get("stiffness", 1.0),
            damping=mat_props.get("damping", 0.5),
            elasticity=mat_props.get("elasticity", 0.8),
            bend_resistance=mat_props.get("bend_resistance", 0.5),
            stretch_resistance=mat_props.get("stretch_resistance", 0.7),
            compression_resistance=mat_props.get("compression_resistance", 0.5),
        )

        # Set mass based on type
        if bounds:
            size = bounds.get("size_m", [0.5, 0.5, 0.5])

            if soft_type == SoftBodyType.CLOTH:
                # Estimate surface area (simplified)
                area = size[0] * size[1] * 2  # Two sides
                props.mass_per_area = mat_props.get("mass_per_area", 0.2)
                props.total_mass = props.mass_per_area * area

            elif soft_type == SoftBodyType.ROPE:
                # Estimate length
                length = max(size)
                props.mass_per_length = mat_props.get("mass_per_length", 0.05)
                props.total_mass = props.mass_per_length * length

            else:
                # Volumetric
                volume = bounds.get("volume_m3", size[0] * size[1] * size[2])
                # Use typical density for soft materials
                density = 200.0  # kg/m^3 (typical for soft foam/stuffing)
                props.total_mass = density * volume

        # Adjust simulation quality based on object size
        if bounds:
            max_dim = max(bounds.get("size_m", [0.5, 0.5, 0.5]))

            if max_dim < 0.1:  # Small objects
                props.particle_resolution = 5
                props.solver_iterations = 3
            elif max_dim < 0.5:  # Medium objects
                props.particle_resolution = 10
                props.solver_iterations = 5
            else:  # Large objects
                props.particle_resolution = 15
                props.solver_iterations = 8

        self.log(
            f"Generated soft body properties: type={soft_type.value}, "
            f"material={material.value}, stiffness={props.stiffness:.2f}"
        )

        return props

    def export_to_usd_schema(
        self,
        props: SoftBodyProperties,
        mesh_path: str,
    ) -> Dict[str, Any]:
        """Export soft body properties to USD schema format.

        Args:
            props: Soft body properties
            mesh_path: Path to mesh file

        Returns:
            USD schema dict for PhysxDeformableBodyAPI
        """
        schema = {
            "type": "PhysxDeformableBodyAPI",
            "enabled": True,

            # Solver settings
            "solverPositionIterationCount": props.solver_iterations,
            "collisionRestOffset": props.collision_margin,
            "collisionContactOffset": props.collision_margin * 2,

            # Deformable properties
            "deformableRestOffset": props.thickness / 2,
            "selfCollision": props.self_collision_enabled,
            "selfCollisionFilterDistance": props.collision_margin * 3,

            # Material properties
            "youngsModulus": props.stiffness * 1e6,  # Convert to Pascals
            "poissonsRatio": 0.45,  # Typical for soft materials
            "dampingScale": props.damping,

            # Simulation quality
            "simulationOwner": "World",
            "sleepThreshold": 0.001,
        }

        # Type-specific settings
        if props.soft_body_type == SoftBodyType.CLOTH:
            schema.update({
                "type": "PhysxDeformableSurfaceAPI",
                "bendingStiffnessScale": props.bend_resistance,
                "stretchStiffnessScale": props.stretch_resistance,
            })

        elif props.soft_body_type == SoftBodyType.ROPE:
            schema.update({
                "type": "PhysxDeformableSurfaceAPI",  # Use surface for rope
                "stretchStiffnessScale": props.stretch_resistance * 1.5,
                "bendingStiffnessScale": 0.1,  # Low bending for ropes
            })

        return schema

    def generate_attachment_constraints(
        self,
        obj_data: Dict[str, Any],
        mesh_vertices: np.ndarray,
    ) -> List[Tuple[int, Tuple[float, float, float]]]:
        """Generate attachment points for soft body.

        Args:
            obj_data: Object data
            mesh_vertices: Mesh vertex positions (Nx3 array)

        Returns:
            List of (vertex_index, attachment_position) tuples
        """
        attachments = []

        soft_type = self.detect_soft_body_type(obj_data)

        if soft_type == SoftBodyType.CLOTH:
            # Attach top corners for hanging cloth
            # Find vertices at max Y (top)
            max_y = np.max(mesh_vertices[:, 1])
            top_vertices = np.where(mesh_vertices[:, 1] > max_y - 0.01)[0]

            # Get corners (min/max X and Z)
            if len(top_vertices) > 0:
                top_verts = mesh_vertices[top_vertices]

                # Find 4 corners
                corners = []
                for x_side in [np.min, np.max]:
                    for z_side in [np.min, np.max]:
                        x_val = x_side(top_verts[:, 0])
                        z_val = z_side(top_verts[:, 2])

                        # Find closest vertex
                        dists = np.sum((top_verts - [x_val, max_y, z_val])**2, axis=1)
                        closest_idx = top_vertices[np.argmin(dists)]

                        corners.append(closest_idx)

                # Attach corners
                for idx in corners:
                    pos = tuple(mesh_vertices[idx])
                    attachments.append((int(idx), pos))

        elif soft_type == SoftBodyType.ROPE:
            # Attach one end of rope
            # Find vertex at max Y or max Z (depending on orientation)
            max_y_idx = np.argmax(mesh_vertices[:, 1])
            pos = tuple(mesh_vertices[max_y_idx])
            attachments.append((int(max_y_idx), pos))

        return attachments
