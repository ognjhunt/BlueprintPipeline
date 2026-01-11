"""Tests for physics modules."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import dataclass

from tools.physics.physics_validation import (
    PhysicsValidator,
    PhysicsValidationResult,
    InertiaValidator,
)
from tools.physics.physics_defaults import (
    MATERIAL_DATABASE,
    get_physics_defaults,
    get_material_properties,
)


class TestPhysicsValidationResult:
    """Test PhysicsValidationResult class."""

    def test_result_creation(self):
        """Test creating validation result."""
        result = PhysicsValidationResult(
            is_valid=True,
            object_id="obj_001",
            checks_performed=[],
            errors=[],
        )
        assert result.is_valid is True
        assert result.object_id == "obj_001"

    def test_result_with_errors(self):
        """Test validation result with errors."""
        result = PhysicsValidationResult(
            is_valid=False,
            object_id="obj_001",
            errors=["Invalid mass", "Missing collision"],
        )
        assert result.is_valid is False
        assert len(result.errors) == 2

    def test_result_to_dict(self):
        """Test serializing result to dict."""
        result = PhysicsValidationResult(
            is_valid=True,
            object_id="obj_001",
            warnings=["High mass"],
        )
        result_dict = result.to_dict()
        assert result_dict["is_valid"] is True
        assert result_dict["object_id"] == "obj_001"


class TestPhysicsValidator:
    """Test PhysicsValidator class."""

    def test_validator_init(self):
        """Test initializing physics validator."""
        validator = PhysicsValidator()
        assert validator is not None

    def test_validate_mass_valid(self):
        """Test validating valid mass."""
        validator = PhysicsValidator()

        result = validator.validate_mass(
            object_id="obj_001",
            mass=1.5,
            min_mass=0.1,
            max_mass=100.0,
        )
        assert result.is_valid is True

    def test_validate_mass_too_light(self):
        """Test validating mass that's too light."""
        validator = PhysicsValidator()

        result = validator.validate_mass(
            object_id="obj_001",
            mass=0.01,
            min_mass=0.1,
            max_mass=100.0,
        )
        assert result.is_valid is False
        assert any("mass" in e.lower() for e in result.errors)

    def test_validate_mass_too_heavy(self):
        """Test validating mass that's too heavy."""
        validator = PhysicsValidator()

        result = validator.validate_mass(
            object_id="obj_001",
            mass=500.0,
            min_mass=0.1,
            max_mass=100.0,
        )
        assert result.is_valid is False

    def test_validate_collision_shape_valid(self):
        """Test validating valid collision shape."""
        validator = PhysicsValidator()

        result = validator.validate_collision_shape(
            object_id="obj_001",
            shape="box",
            dimensions=[1.0, 1.0, 1.0],
        )
        assert result.is_valid is True

    def test_validate_collision_shape_invalid_dimensions(self):
        """Test validating collision shape with invalid dimensions."""
        validator = PhysicsValidator()

        result = validator.validate_collision_shape(
            object_id="obj_001",
            shape="box",
            dimensions=[0.0, 1.0, 1.0],  # Zero dimension invalid
        )
        assert result.is_valid is False

    def test_validate_collision_shape_unsupported(self):
        """Test validating unsupported collision shape."""
        validator = PhysicsValidator()

        result = validator.validate_collision_shape(
            object_id="obj_001",
            shape="unsupported_shape",
            dimensions=[1.0],
        )
        assert result.is_valid is False

    def test_validate_com_valid(self):
        """Test validating valid center of mass."""
        validator = PhysicsValidator()

        result = validator.validate_center_of_mass(
            object_id="obj_001",
            com=[0.0, 0.0, 0.5],
            bounds=[-1.0, 1.0],
        )
        assert result.is_valid is True

    def test_validate_com_outside_bounds(self):
        """Test validating COM outside object bounds."""
        validator = PhysicsValidator()

        result = validator.validate_center_of_mass(
            object_id="obj_001",
            com=[10.0, 10.0, 10.0],
            bounds=[-1.0, 1.0],
        )
        assert result.is_valid is False

    def test_validate_friction_valid(self):
        """Test validating valid friction."""
        validator = PhysicsValidator()

        result = validator.validate_friction(
            object_id="obj_001",
            static_friction=0.5,
            dynamic_friction=0.4,
        )
        assert result.is_valid is True

    def test_validate_friction_invalid(self):
        """Test validating invalid friction."""
        validator = PhysicsValidator()

        # Static friction less than dynamic is invalid
        result = validator.validate_friction(
            object_id="obj_001",
            static_friction=0.2,
            dynamic_friction=0.5,
        )
        assert result.is_valid is False


class TestInertiaValidator:
    """Test InertiaValidator class."""

    def test_inertia_validator_init(self):
        """Test initializing inertia validator."""
        validator = InertiaValidator()
        assert validator is not None

    def test_validate_inertia_tensor_valid(self):
        """Test validating valid inertia tensor."""
        validator = InertiaValidator()

        # Symmetric positive definite tensor
        inertia_tensor = [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]

        result = validator.validate_inertia_tensor(
            object_id="obj_001",
            inertia_tensor=inertia_tensor,
        )
        assert result.is_valid is True

    def test_validate_inertia_tensor_not_symmetric(self):
        """Test validating non-symmetric inertia tensor."""
        validator = InertiaValidator()

        # Non-symmetric tensor
        inertia_tensor = [
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],  # Should be 1.0 to match above
            [0.0, 0.0, 2.0],
        ]

        result = validator.validate_inertia_tensor(
            object_id="obj_001",
            inertia_tensor=inertia_tensor,
        )
        assert result.is_valid is False

    def test_compute_inertia_box(self):
        """Test computing inertia for box."""
        validator = InertiaValidator()

        inertia = validator.compute_inertia_box(
            mass=1.0,
            dimensions=[1.0, 2.0, 3.0],
        )

        assert inertia is not None
        assert len(inertia) == 3
        # Check that all inertia components are positive
        assert all(i > 0 for i in inertia)

    def test_compute_inertia_sphere(self):
        """Test computing inertia for sphere."""
        validator = InertiaValidator()

        inertia = validator.compute_inertia_sphere(
            mass=1.0,
            radius=0.5,
        )

        assert inertia is not None
        assert len(inertia) == 3
        # For sphere, all inertia components should be equal
        assert abs(inertia[0] - inertia[1]) < 1e-6
        assert abs(inertia[1] - inertia[2]) < 1e-6

    def test_compute_inertia_cylinder(self):
        """Test computing inertia for cylinder."""
        validator = InertiaValidator()

        inertia = validator.compute_inertia_cylinder(
            mass=1.0,
            radius=0.5,
            height=2.0,
        )

        assert inertia is not None
        assert len(inertia) == 3


class TestPhysicsDefaults:
    """Test physics defaults module."""

    def test_material_database_loaded(self):
        """Test that material database is loaded."""
        assert MATERIAL_DATABASE is not None
        assert len(MATERIAL_DATABASE) > 0

    def test_get_material_properties(self):
        """Test getting material properties."""
        # Test with a known material
        props = get_material_properties("steel")
        assert props is not None
        assert "density" in props or "friction" in props

    def test_get_material_properties_unknown(self):
        """Test getting properties for unknown material."""
        props = get_material_properties("unknown_material_xyz")
        # Should return None or default properties
        assert props is None or isinstance(props, dict)

    def test_get_physics_defaults(self):
        """Test getting physics defaults."""
        defaults = get_physics_defaults()
        assert defaults is not None
        assert isinstance(defaults, dict)
        # Check for expected default properties
        assert "gravity" in defaults or "default_friction" in defaults or "default_density" in defaults

    def test_physics_defaults_structure(self):
        """Test structure of physics defaults."""
        defaults = get_physics_defaults()

        # Should have reasonable default values
        if "gravity" in defaults:
            assert defaults["gravity"] > 0

        if "default_friction" in defaults:
            assert 0 <= defaults["default_friction"] <= 2.0


class TestPhysicsValidationIntegration:
    """Integration tests for physics validation."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        validator = PhysicsValidator()

        # Validate multiple aspects of an object
        object_id = "test_cube"

        # Validate mass
        mass_result = validator.validate_mass(object_id, mass=2.0)
        assert mass_result.is_valid

        # Validate collision
        collision_result = validator.validate_collision_shape(
            object_id,
            shape="box",
            dimensions=[1.0, 1.0, 1.0],
        )
        assert collision_result.is_valid

        # Validate friction
        friction_result = validator.validate_friction(
            object_id,
            static_friction=0.5,
            dynamic_friction=0.4,
        )
        assert friction_result.is_valid

        # All should be valid
        all_valid = all([
            mass_result.is_valid,
            collision_result.is_valid,
            friction_result.is_valid,
        ])
        assert all_valid

    def test_physics_defaults_consistency(self):
        """Test consistency of physics defaults."""
        defaults = get_physics_defaults()

        # If defaults contain material properties, verify they're consistent
        if "materials" in defaults:
            materials = defaults["materials"]
            for material_name, props in materials.items():
                # Each material should have required properties
                required_keys = ["density", "friction"]
                for key in required_keys:
                    # Either in material directly or in database
                    assert key in props or key in MATERIAL_DATABASE.get(material_name, {})


class TestPhysicsEdgeCases:
    """Test edge cases in physics validation."""

    def test_very_small_mass(self):
        """Test validation of very small mass."""
        validator = PhysicsValidator()

        result = validator.validate_mass(
            object_id="tiny_obj",
            mass=0.001,
            min_mass=0.0001,
            max_mass=100.0,
        )
        assert result.is_valid is True

    def test_very_large_mass(self):
        """Test validation of very large mass."""
        validator = PhysicsValidator()

        result = validator.validate_mass(
            object_id="heavy_obj",
            mass=10000.0,
            min_mass=0.1,
            max_mass=100000.0,
        )
        assert result.is_valid is True

    def test_zero_friction(self):
        """Test validation of zero friction (frictionless)."""
        validator = PhysicsValidator()

        result = validator.validate_friction(
            object_id="frictionless",
            static_friction=0.0,
            dynamic_friction=0.0,
        )
        # Should be valid - frictionless surfaces are valid physics
        assert result.is_valid is True

    def test_high_friction(self):
        """Test validation of high friction."""
        validator = PhysicsValidator()

        result = validator.validate_friction(
            object_id="high_friction",
            static_friction=1.5,
            dynamic_friction=1.0,
        )
        assert result.is_valid is True

    def test_negative_mass_invalid(self):
        """Test that negative mass is invalid."""
        validator = PhysicsValidator()

        result = validator.validate_mass(
            object_id="invalid_obj",
            mass=-1.0,
        )
        assert result.is_valid is False

    def test_negative_dimensions_invalid(self):
        """Test that negative dimensions are invalid."""
        validator = PhysicsValidator()

        result = validator.validate_collision_shape(
            object_id="invalid_obj",
            shape="box",
            dimensions=[-1.0, 1.0, 1.0],
        )
        assert result.is_valid is False
