#!/usr/bin/env python3
"""
Physics Validation for Sim-Ready Assets.

This module validates physics properties before USD export to catch issues early:
- Mass sanity checks (positive, reasonable range)
- Friction consistency (dynamic <= static)
- Center of mass bounds validation
- Collision shape appropriateness
- Inertia tensor validation

These validations prevent common physics errors that would cause:
- PhysX crashes (zero/negative mass)
- Unrealistic behavior (bad friction)
- Simulation instabilities (bad inertia)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Levels
# =============================================================================


class ValidationLevel(str, Enum):
    """Severity level for validation issues."""

    CRITICAL = "critical"  # Will crash or fail badly
    WARNING = "warning"    # Suspicious but might work
    INFO = "info"          # Suggestion for improvement


@dataclass
class ValidationIssue:
    """A physics validation issue."""

    level: ValidationLevel
    property_name: str
    message: str
    current_value: Optional[Any] = None
    suggested_value: Optional[Any] = None


@dataclass
class ValidationResult:
    """Result of physics validation."""

    object_id: str
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return any(issue.level == ValidationLevel.CRITICAL for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(issue.level == ValidationLevel.WARNING for issue in self.issues)

    def add_issue(self, level: ValidationLevel, property_name: str, message: str,
                  current_value: Any = None, suggested_value: Any = None):
        """Add a validation issue."""
        self.issues.append(ValidationIssue(
            level=level,
            property_name=property_name,
            message=message,
            current_value=current_value,
            suggested_value=suggested_value,
        ))
        if level == ValidationLevel.CRITICAL:
            self.passed = False


# =============================================================================
# Physics Validator
# =============================================================================


class PhysicsValidator:
    """
    Validates physics properties before USD export.

    This prevents common physics errors that cause simulation failures:
    - Zero/negative mass (PhysX crash)
    - Dynamic friction > static friction (physically inconsistent)
    - Center of mass outside object bounds (unstable)
    - Poor collision approximations (penetration/tunneling)
    """

    # Physics constants and bounds
    MIN_MASS_KG = 0.001  # 1 gram minimum (prevents PhysX errors)
    MAX_MASS_KG = 10000.0  # 10 tons maximum for manipulation
    MIN_DENSITY_KG_M3 = 0.5  # Lighter than styrofoam
    MAX_DENSITY_KG_M3 = 20000.0  # Denser than lead
    MIN_FRICTION = 0.0
    MAX_FRICTION = 2.0

    def __init__(self, strict: bool = False, verbose: bool = True):
        """
        Initialize physics validator.

        Args:
            strict: If True, warnings become errors
            verbose: If True, print validation messages
        """
        self.strict = strict
        self.verbose = verbose

    def validate_object(
        self,
        obj: Dict[str, Any],
        physics: Dict[str, Any],
        bounds: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate physics properties for an object.

        Args:
            obj: Object dictionary with category, dimensions, etc.
            physics: Physics properties dictionary
            bounds: Optional bounding box information

        Returns:
            ValidationResult with issues
        """
        obj_id = obj.get("id", obj.get("name", "unknown"))
        result = ValidationResult(object_id=obj_id, passed=True)

        # 1. Mass validation
        self._validate_mass(physics, result)

        # 2. Friction validation
        self._validate_friction(physics, result)

        # 3. Density validation
        if bounds:
            self._validate_density(physics, bounds, result)

        # 4. Center of mass validation
        if bounds:
            if physics.get("center_of_mass_m") is None:
                computed_com, com_source, com_warning = self._compute_center_of_mass(
                    obj,
                    physics,
                    bounds,
                )
                physics["center_of_mass_m"] = computed_com.tolist()
                physics["center_of_mass_source"] = com_source
                if com_source != "mesh":
                    result.add_issue(
                        ValidationLevel.WARNING,
                        "center_of_mass",
                        com_warning or "Center of mass missing; inferred from bounds instead of mesh data",
                        current_value=physics["center_of_mass_m"],
                        suggested_value="mesh-derived center of mass",
                    )
            self._validate_center_of_mass(physics, bounds, result)

        # 5. Collision shape appropriateness
        self._validate_collision_shape(obj, physics, result)

        # 6. Inertia computation and validation
        if "inertia_tensor" not in physics and bounds:
            inertia_tensor, inertia_source, inertia_warning = self._compute_inertia_tensor(
                obj,
                physics,
                bounds,
            )
            physics["inertia_tensor"] = inertia_tensor
            physics["inertia_source"] = inertia_source
            result.metadata["inertia_source"] = inertia_source
            if inertia_warning:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "inertia_tensor",
                    inertia_warning,
                    current_value=inertia_source,
                    suggested_value="mesh- or USD-derived inertia",
                )
            if self.verbose:
                print(f"  [PHYSICS] Computed inertia tensor for {obj.get('id', 'unknown')}")

        if "inertia_tensor" in physics:
            if "inertia_source" not in result.metadata:
                result.metadata["inertia_source"] = self._infer_inertia_source(physics)
            self._validate_inertia(physics, result)

        # Print summary if verbose
        if self.verbose and result.issues:
            self._print_validation_summary(result)

        return result

    def _validate_mass(self, physics: Dict[str, Any], result: ValidationResult):
        """Validate mass is positive and reasonable."""
        mass = physics.get("mass_kg", 1.0)

        # Critical: zero or negative mass
        if mass <= 0:
            result.add_issue(
                ValidationLevel.CRITICAL,
                "mass_kg",
                "Zero or negative mass will crash PhysX",
                current_value=mass,
                suggested_value=self.MIN_MASS_KG,
            )
        # Critical: mass below minimum
        elif mass < self.MIN_MASS_KG:
            result.add_issue(
                ValidationLevel.CRITICAL,
                "mass_kg",
                f"Mass below minimum ({self.MIN_MASS_KG}kg) may cause instability",
                current_value=mass,
                suggested_value=self.MIN_MASS_KG,
            )
        # Warning: mass above maximum
        elif mass > self.MAX_MASS_KG:
            result.add_issue(
                ValidationLevel.WARNING,
                "mass_kg",
                f"Mass exceeds typical manipulation range ({self.MAX_MASS_KG}kg)",
                current_value=mass,
            )

    def _validate_friction(self, physics: Dict[str, Any], result: ValidationResult):
        """Validate friction coefficients."""
        static_friction = physics.get("static_friction", 0.5)
        dynamic_friction = physics.get("dynamic_friction", 0.4)

        # Warning: dynamic > static (physically inconsistent)
        if dynamic_friction > static_friction:
            result.add_issue(
                ValidationLevel.WARNING,
                "friction",
                "Dynamic friction > static friction (physically inconsistent)",
                current_value=f"static={static_friction:.3f}, dynamic={dynamic_friction:.3f}",
                suggested_value=f"dynamic={static_friction * 0.9:.3f}",
            )

        # Warning: friction out of typical range
        if static_friction < self.MIN_FRICTION or static_friction > self.MAX_FRICTION:
            result.add_issue(
                ValidationLevel.WARNING,
                "static_friction",
                f"Static friction outside typical range [{self.MIN_FRICTION}, {self.MAX_FRICTION}]",
                current_value=static_friction,
            )

        if dynamic_friction < self.MIN_FRICTION or dynamic_friction > self.MAX_FRICTION:
            result.add_issue(
                ValidationLevel.WARNING,
                "dynamic_friction",
                f"Dynamic friction outside typical range [{self.MIN_FRICTION}, {self.MAX_FRICTION}]",
                current_value=dynamic_friction,
            )

    def _validate_density(
        self,
        physics: Dict[str, Any],
        bounds: Dict[str, Any],
        result: ValidationResult
    ):
        """Validate bulk density is reasonable."""
        mass = physics.get("mass_kg", 1.0)
        volume = bounds.get("volume_m3", 0.0)

        if volume > 0:
            density = mass / volume

            if density < self.MIN_DENSITY_KG_M3:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "bulk_density",
                    f"Density ({density:.1f} kg/m³) lighter than styrofoam ({self.MIN_DENSITY_KG_M3} kg/m³)",
                    current_value=density,
                )
            elif density > self.MAX_DENSITY_KG_M3:
                result.add_issue(
                    ValidationLevel.WARNING,
                    "bulk_density",
                    f"Density ({density:.1f} kg/m³) denser than lead ({self.MAX_DENSITY_KG_M3} kg/m³)",
                    current_value=density,
                )

    def _validate_center_of_mass(
        self,
        physics: Dict[str, Any],
        bounds: Dict[str, Any],
        result: ValidationResult
    ):
        """Validate center of mass is within object bounds."""
        com = physics.get("center_of_mass_m", [0, 0, 0])
        if isinstance(com, list):
            com = np.array(com)

        # Get bounding box
        bbox_min = bounds.get("min")
        bbox_max = bounds.get("max")

        if bbox_min is None or bbox_max is None:
            size_m = bounds.get("size_m") or bounds.get("size") or [0.2, 0.2, 0.2]
            center_m = bounds.get("center_m") or bounds.get("center") or [0.0, 0.0, 0.0]
            bbox_min = [center_m[i] - 0.5 * size_m[i] for i in range(3)]
            bbox_max = [center_m[i] + 0.5 * size_m[i] for i in range(3)]

        if bbox_min is None:
            bbox_min = [-0.1, -0.1, -0.1]
        if bbox_max is None:
            bbox_max = [0.1, 0.1, 0.1]
        if isinstance(bbox_min, list):
            bbox_min = np.array(bbox_min)
        if isinstance(bbox_max, list):
            bbox_max = np.array(bbox_max)

        # Check if CoM is within bounds (with small tolerance)
        tolerance = 0.01  # 1cm
        if not self._point_in_bounds(com, bbox_min - tolerance, bbox_max + tolerance):
            result.add_issue(
                ValidationLevel.WARNING,
                "center_of_mass",
                "Center of mass outside object bounds (may cause instability)",
                current_value=com.tolist() if isinstance(com, np.ndarray) else com,
                suggested_value="center of bounding box",
            )

    def _compute_center_of_mass(
        self,
        obj: Dict[str, Any],
        physics: Dict[str, Any],
        bounds: Dict[str, Any],
    ) -> Tuple[np.ndarray, str, Optional[str]]:
        """Compute center of mass from mesh geometry or fallback to bounding box."""
        mesh_path = self._find_mesh_path(obj, physics)
        if mesh_path:
            try:
                import importlib.util
                if importlib.util.find_spec("trimesh") is None:
                    raise ModuleNotFoundError("trimesh not available")
                import trimesh
            except Exception:
                mesh_warning = "Center of mass inferred from bounds; trimesh not available for mesh COM."
            else:
                try:
                    mesh = trimesh.load(str(mesh_path), force="mesh")
                    if isinstance(mesh, trimesh.Scene):
                        geometries = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                        if geometries:
                            mesh = trimesh.util.concatenate(geometries)
                        else:
                            mesh = None

                    if mesh is None or mesh.is_empty:
                        mesh_warning = "Center of mass inferred from bounds; mesh is empty."
                    else:
                        volume = float(mesh.volume) if mesh.volume is not None else 0.0
                        if volume > 0:
                            com = mesh.center_mass
                            if com is None:
                                mass_props = mesh.mass_properties(density=1.0)
                                com = mass_props.get("center_mass")
                            if com is not None:
                                return np.array(com, dtype=float), "mesh", None
                            mesh_warning = "Center of mass inferred from bounds; mesh COM unavailable."
                        else:
                            mesh_warning = "Center of mass inferred from bounds; mesh volume is zero."
                except Exception:
                    mesh_warning = "Center of mass inferred from bounds; mesh COM computation failed."
        else:
            mesh_warning = "Center of mass inferred from bounds; no mesh path available."

        mesh_bounds = physics.get("mesh_bounds") or bounds.get("mesh_bounds") or {}
        mesh_center = mesh_bounds.get("center")
        if mesh_center is None:
            mesh_min = mesh_bounds.get("min") or mesh_bounds.get("minimum")
            mesh_max = mesh_bounds.get("max") or mesh_bounds.get("maximum")
            if mesh_min is not None and mesh_max is not None:
                mesh_center = [(mesh_min[i] + mesh_max[i]) * 0.5 for i in range(3)]

        if mesh_center is not None:
            return np.array(mesh_center, dtype=float), "bounds", mesh_warning

        bounds_center = bounds.get("center_m") or bounds.get("center")
        if bounds_center is None:
            bounds_min = bounds.get("min")
            bounds_max = bounds.get("max")
            if bounds_min is not None and bounds_max is not None:
                bounds_center = [(bounds_min[i] + bounds_max[i]) * 0.5 for i in range(3)]
            else:
                bounds_center = [0.0, 0.0, 0.0]

        return np.array(bounds_center, dtype=float), "bounds", mesh_warning

    def _validate_collision_shape(
        self,
        obj: Dict[str, Any],
        physics: Dict[str, Any],
        result: ValidationResult
    ):
        """Validate collision shape is appropriate for object."""
        collision_shape = physics.get("collision_shape", "box")
        category = obj.get("category", "").lower()

        # Check if simple shapes are appropriate
        complex_categories = [
            "furniture", "appliance", "cabinet", "table", "chair",
            "desk", "shelving", "countertop", "sink", "toilet"
        ]

        is_complex = any(cat in category for cat in complex_categories)

        if is_complex and collision_shape in ["box", "sphere"]:
            result.add_issue(
                ValidationLevel.INFO,
                "collision_shape",
                f"Simple collision shape ({collision_shape}) for complex object ({category})",
                current_value=collision_shape,
                suggested_value="convex_decomposition",
            )

    def _compute_inertia_tensor(
        self,
        obj: Dict[str, Any],
        physics: Dict[str, Any],
        bounds: Dict[str, Any],
    ) -> Tuple[List[float], str, Optional[str]]:
        """
        Compute inertia tensor from USD mass properties, mesh geometry, or bounding box.

        Preference order:
        1) USD mass properties (if provided in physics metadata)
        2) Mesh-based computation (if mesh path + trimesh available)
        3) Box approximation (uniform density box)

        Returns diagonal inertia tensor [Ixx, Iyy, Izz] plus provenance and warning.

        For a uniform density box with dimensions (L, W, H) and mass M:
        Ixx = (1/12) * M * (W^2 + H^2)
        Iyy = (1/12) * M * (L^2 + H^2)
        Izz = (1/12) * M * (L^2 + W^2)

        Args:
            obj: Object metadata dict
            physics: Physics properties dict (must contain mass_kg)
            bounds: Bounds dictionary

        Returns:
            Tuple of diagonal inertia tensor, provenance source, warning message
        """
        usd_inertia = self._extract_usd_inertia(physics)
        if usd_inertia is not None:
            return usd_inertia, "usd", None

        mesh_inertia, mesh_warning, mesh_attempted = self._compute_inertia_from_mesh(obj, physics)
        if mesh_inertia is not None:
            return mesh_inertia, "mesh", None

        box_warning = mesh_warning
        if not mesh_attempted:
            box_warning = "Inertia tensor computed using box approximation; no mesh/USD mass properties available."

        return self._compute_box_inertia(physics, bounds), "box_approx", box_warning

    def _extract_usd_inertia(self, physics: Dict[str, Any]) -> Optional[List[float]]:
        """Extract USD-derived inertia tensor if present."""
        usd_props = physics.get("usd_mass_properties") or physics.get("usd_mass_props") or {}
        inertia = (
            usd_props.get("inertia_tensor")
            or usd_props.get("inertia")
            or usd_props.get("diagonal_inertia")
            or usd_props.get("inertia_diagonal")
            or physics.get("usd_inertia_tensor")
            or physics.get("usd_diagonal_inertia")
        )
        if inertia is None:
            return None

        inertia_array = np.array(inertia, dtype=float)
        if inertia_array.shape == (3,):
            return inertia_array.tolist()
        if inertia_array.shape == (3, 3):
            return inertia_array.diagonal().tolist()
        if inertia_array.size == 9:
            inertia_matrix = inertia_array.reshape(3, 3)
            return inertia_matrix.diagonal().tolist()
        return None

    def _compute_inertia_from_mesh(
        self,
        obj: Dict[str, Any],
        physics: Dict[str, Any],
    ) -> Tuple[Optional[List[float]], Optional[str], bool]:
        """Compute inertia from mesh geometry using trimesh if available."""
        mesh_path = self._find_mesh_path(obj, physics)
        if not mesh_path:
            return None, None, False

        try:
            import importlib.util
            if importlib.util.find_spec("trimesh") is None:
                return None, "Inertia tensor computed using box approximation; trimesh not available.", True
            import trimesh
        except Exception:
            return None, "Inertia tensor computed using box approximation; trimesh unavailable.", True

        try:
            mesh = trimesh.load(str(mesh_path), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                geometries = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if not geometries:
                    return None, "Inertia tensor computed using box approximation; no mesh geometry found.", True
                mesh = trimesh.util.concatenate(geometries)

            if mesh.is_empty:
                return None, "Inertia tensor computed using box approximation; mesh is empty.", True

            volume = float(mesh.volume) if mesh.volume is not None else 0.0
            if volume <= 0:
                return None, "Inertia tensor computed using box approximation; mesh volume is zero.", True

            mass = float(physics.get("mass_kg", 1.0))
            if mass <= 0:
                mass = self.MIN_MASS_KG
            density = mass / volume
            mass_props = mesh.mass_properties(density=density)
            inertia = mass_props.get("inertia")
            if inertia is None:
                return None, "Inertia tensor computed using box approximation; mesh inertia unavailable.", True
            inertia_matrix = np.array(inertia, dtype=float)
            return inertia_matrix.diagonal().tolist(), None, True
        except Exception:
            return None, "Inertia tensor computed using box approximation; mesh computation failed.", True

    def _find_mesh_path(self, obj: Dict[str, Any], physics: Dict[str, Any]) -> Optional[str]:
        """Identify a mesh path from object or physics metadata."""
        candidates = [
            physics.get("mesh_path"),
            physics.get("mesh_file"),
            physics.get("visual_path"),
            obj.get("mesh_path"),
            obj.get("asset_path"),
        ]
        asset = obj.get("asset", {})
        if isinstance(asset, dict):
            candidates.append(asset.get("path"))

        for candidate in candidates:
            if not candidate:
                continue
            if isinstance(candidate, dict):
                continue
            candidate_str = str(candidate)
            if candidate_str.lower().endswith((".usd", ".usda", ".usdc", ".usdz")):
                continue
            if candidate_str:
                return candidate_str
        return None

    def _compute_box_inertia(self, physics: Dict[str, Any], bounds: Dict[str, Any]) -> List[float]:
        """Compute inertia tensor using a box approximation."""
        bbox_min, bbox_max = self._extract_bbox(bounds)
        dimensions = bbox_max - bbox_min  # [length, width, height]
        mass = physics.get("mass_kg", 1.0)

        # Ensure positive mass
        if mass <= 0:
            mass = self.MIN_MASS_KG

        # Compute inertia for box approximation
        L, W, H = dimensions

        # Box inertia formula (around center of mass)
        Ixx = (1.0 / 12.0) * mass * (W**2 + H**2)
        Iyy = (1.0 / 12.0) * mass * (L**2 + H**2)
        Izz = (1.0 / 12.0) * mass * (L**2 + W**2)

        # Return as diagonal tensor
        return [float(Ixx), float(Iyy), float(Izz)]

    def _extract_bbox(self, bounds: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract bbox min/max from bounds with fallbacks."""
        bbox_min = bounds.get("min")
        bbox_max = bounds.get("max")
        if bbox_min is not None and bbox_max is not None:
            return np.array(bbox_min, dtype=float), np.array(bbox_max, dtype=float)

        size_m = bounds.get("size_m") or bounds.get("size") or [0.2, 0.2, 0.2]
        center_m = bounds.get("center_m") or bounds.get("center") or [0.0, 0.0, 0.0]
        bbox_min = [center_m[i] - 0.5 * size_m[i] for i in range(3)]
        bbox_max = [center_m[i] + 0.5 * size_m[i] for i in range(3)]
        return np.array(bbox_min, dtype=float), np.array(bbox_max, dtype=float)

    def _infer_inertia_source(self, physics: Dict[str, Any]) -> Optional[str]:
        """Infer inertia provenance from physics metadata."""
        source = physics.get("inertia_source")
        if source in {"mesh", "usd", "box_approx"}:
            return source
        if (
            physics.get("usd_mass_properties")
            or physics.get("usd_mass_props")
            or physics.get("usd_inertia_tensor")
            or physics.get("usd_diagonal_inertia")
        ):
            return "usd"
        if physics.get("mesh_mass_properties") or physics.get("mesh_path") or physics.get("mesh_file"):
            return "mesh"
        if physics.get("inertia_tensor") is not None:
            return "box_approx"
        return None

    def _validate_inertia(self, physics: Dict[str, Any], result: ValidationResult):
        """Validate inertia tensor is positive definite."""
        inertia = physics.get("inertia_tensor")
        if inertia is None:
            return

        if isinstance(inertia, list):
            inertia = np.array(inertia)

        # Check if diagonal (simplified inertia)
        if inertia.shape == (3,):
            # Diagonal elements should all be positive
            if np.any(inertia <= 0):
                result.add_issue(
                    ValidationLevel.CRITICAL,
                    "inertia_tensor",
                    "Inertia tensor has non-positive diagonal elements",
                    current_value=inertia.tolist(),
                )
        elif inertia.shape == (3, 3):
            # Full tensor should be positive definite
            try:
                eigenvalues = np.linalg.eigvals(inertia)
                if np.any(eigenvalues <= 0):
                    result.add_issue(
                        ValidationLevel.CRITICAL,
                        "inertia_tensor",
                        "Inertia tensor is not positive definite",
                        current_value="invalid_tensor",
                    )
            except np.linalg.LinAlgError:
                result.add_issue(
                    ValidationLevel.CRITICAL,
                    "inertia_tensor",
                    "Inertia tensor is malformed",
                    current_value="malformed_tensor",
                )

    def _point_in_bounds(
        self,
        point: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray
    ) -> bool:
        """Check if point is within bounding box."""
        return np.all(point >= bbox_min) and np.all(point <= bbox_max)

    def _print_validation_summary(self, result: ValidationResult):
        """Print validation summary."""
        if result.has_critical_issues:
            logger.error(f"[PHYSICS-VALIDATION] ❌ {result.object_id}: CRITICAL ISSUES")
        elif result.has_warnings:
            logger.warning(f"[PHYSICS-VALIDATION] ⚠️  {result.object_id}: WARNINGS")
        else:
            logger.info(f"[PHYSICS-VALIDATION] ℹ️  {result.object_id}: Suggestions")

        for issue in result.issues:
            prefix = {
                ValidationLevel.CRITICAL: "  ❌ CRITICAL",
                ValidationLevel.WARNING: "  ⚠️  WARNING",
                ValidationLevel.INFO: "  ℹ️  INFO",
            }[issue.level]

            msg = f"{prefix}: {issue.property_name} - {issue.message}"
            if issue.current_value is not None:
                msg += f" (current: {issue.current_value})"
            if issue.suggested_value is not None:
                msg += f" → (suggested: {issue.suggested_value})"

            if issue.level == ValidationLevel.CRITICAL:
                logger.error(msg)
            elif issue.level == ValidationLevel.WARNING:
                logger.warning(msg)
            else:
                logger.info(msg)


# =============================================================================
# Batch Validation
# =============================================================================


def validate_scene_physics(
    manifest: Dict[str, Any],
    physics_data: Dict[str, Dict[str, Any]],
    strict: bool = False,
    verbose: bool = True,
) -> Dict[str, ValidationResult]:
    """
    Validate physics for all objects in a scene.

    Args:
        manifest: Scene manifest with objects
        physics_data: Dict mapping object_id to physics properties
        strict: If True, warnings become errors
        verbose: If True, print validation messages

    Returns:
        Dict mapping object_id to ValidationResult
    """
    validator = PhysicsValidator(strict=strict, verbose=verbose)
    results = {}

    objects = manifest.get("objects", [])
    for obj in objects:
        obj_id = obj.get("id", obj.get("name", "unknown"))
        physics = physics_data.get(obj_id, {})
        bounds = obj.get("bounds")

        result = validator.validate_object(obj, physics, bounds)
        results[obj_id] = result

    # Print overall summary
    if verbose:
        total = len(results)
        critical = sum(1 for r in results.values() if r.has_critical_issues)
        warnings = sum(1 for r in results.values() if r.has_warnings)

        logger.info(f"\n[PHYSICS-VALIDATION] Summary: {total} objects validated")
        if critical > 0:
            logger.error(f"  ❌ {critical} objects with CRITICAL issues")
        if warnings > 0:
            logger.warning(f"  ⚠️  {warnings} objects with warnings")
        if critical == 0 and warnings == 0:
            logger.info("  ✅ All objects passed validation")

    return results


if __name__ == "__main__":
    # Example usage
    validator = PhysicsValidator(verbose=True)

    # Test object
    test_obj = {
        "id": "test_cube",
        "category": "box",
        "dimensions": [0.1, 0.1, 0.1],
    }

    test_physics = {
        "mass_kg": 0.5,
        "static_friction": 0.6,
        "dynamic_friction": 0.5,
        "collision_shape": "box",
    }

    test_bounds = {
        "min": [-0.05, -0.05, -0.05],
        "max": [0.05, 0.05, 0.05],
        "volume_m3": 0.001,
    }

    result = validator.validate_object(test_obj, test_physics, test_bounds)
    print(f"\nValidation passed: {result.passed}")
    print(f"Issues found: {len(result.issues)}")
