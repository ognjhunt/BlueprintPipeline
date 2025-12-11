"""
Randomizer configurations for Replicator synthetic data generation.

This module defines domain randomization strategies for different
training policies and environment types.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RandomizerConfig:
    """Configuration for a single randomizer."""

    name: str
    randomizer_type: str
    enabled: bool = True
    frequency: str = "per_frame"  # per_frame, per_episode, once
    targets: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)


def get_randomizers_for_policy(
    policy: dict[str, Any],
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> list[RandomizerConfig]:
    """
    Get randomizer configurations for a given policy.

    Args:
        policy: Policy configuration
        recipe: Scene recipe
        environment: Environment configuration

    Returns:
        List of RandomizerConfig objects
    """
    randomizers = []

    policy_randomizers = policy.get("randomizers", [])

    for rand_spec in policy_randomizers:
        if not rand_spec.get("enabled", True):
            continue

        rand_name = rand_spec.get("name", "unknown")
        rand_config = create_randomizer(
            rand_name,
            rand_spec.get("parameters", {}),
            rand_spec.get("frequency", "per_frame"),
            recipe,
            environment,
        )

        if rand_config:
            randomizers.append(rand_config)

    return randomizers


def create_randomizer(
    name: str,
    parameters: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> Optional[RandomizerConfig]:
    """Create a randomizer configuration by name."""
    creators = {
        "object_scatter": _create_object_scatter,
        "material_variation": _create_material_variation,
        "lighting_variation": _create_lighting_variation,
        "camera_variation": _create_camera_variation,
        "articulation_state": _create_articulation_state,
        "object_placement": _create_object_placement,
        "drawer_state": _create_drawer_state,
        "drawer_contents": _create_drawer_contents,
        "door_state": _create_door_state,
        "knob_state": _create_knob_state,
        "cloth_scatter": _create_cloth_scatter,
        "cloth_deformation": _create_cloth_deformation,
        "shelf_population": _create_shelf_population,
        "table_setup": _create_table_setup,
        "dirty_state": _create_dirty_state,
        "dishwasher_state": _create_dishwasher_state,
        "switch_states": _create_switch_states,
        "label_variation": _create_label_variation,
        "pallet_placement": _create_pallet_placement,
        "load_variation": _create_load_variation,
    }

    creator = creators.get(name)
    if creator:
        return creator(parameters, frequency, recipe, environment)

    return RandomizerConfig(
        name=name,
        randomizer_type=name,
        enabled=True,
        frequency=frequency,
        parameters=parameters,
    )


def _create_object_scatter(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create object scatter randomizer."""
    regions = recipe.get("placement_regions", [])
    region_paths = [f"/PlacementRegions/{r['id']}" for r in regions]

    variation_templates = environment.get("variation_asset_templates", [])
    asset_categories = [t["category"] for t in variation_templates if t.get("priority") != "optional"]

    return RandomizerConfig(
        name="object_scatter",
        randomizer_type="scatter",
        enabled=True,
        frequency=frequency,
        targets=region_paths or ["/Objects"],
        parameters={
            "min_objects": params.get("min_objects", 5),
            "max_objects": params.get("max_objects", 20),
            "asset_categories": asset_categories,
            "regions": params.get("regions", []),
            "collision_check": True,
            "orientation_distribution": {
                "type": "uniform",
                "rotation_axis": "Y",
                "range": [0, 360],
            },
        },
    )


def _create_material_variation(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create material variation randomizer."""
    return RandomizerConfig(
        name="material_variation",
        randomizer_type="material",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/*"],
        parameters={
            "variation_type": params.get("variation_type", "texture_swap"),
            "color_variation": {
                "enabled": params.get("color_variation", True),
                "hue_range": [-0.1, 0.1],
                "saturation_range": [-0.2, 0.2],
                "value_range": [-0.1, 0.1],
            },
            "roughness_variation": {
                "enabled": True,
                "range": [-0.1, 0.1],
            },
        },
    )


def _create_lighting_variation(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create lighting variation randomizer."""
    return RandomizerConfig(
        name="lighting_variation",
        randomizer_type="lighting",
        enabled=True,
        frequency=frequency,
        targets=["/Lights/*"],
        parameters={
            "intensity_range": params.get("intensity_range", [0.5, 2.0]),
            "color_temperature_range": params.get("color_temperature_range", [3000, 6500]),
            "position_noise": params.get("position_noise", 0.5),
            "rotation_range": params.get("rotation_range", [-15, 15]),
        },
    )


def _create_camera_variation(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create camera variation randomizer."""
    return RandomizerConfig(
        name="camera_variation",
        randomizer_type="camera",
        enabled=True,
        frequency=frequency,
        targets=["/Cameras/*"],
        parameters={
            "position_noise": params.get("position_noise", 0.1),
            "rotation_noise": params.get("rotation_noise", 5),
            "focal_length_range": params.get("focal_length_range", [20, 50]),
        },
    )


def _create_articulation_state(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create articulation state randomizer."""
    articulated = [obj["id"] for obj in recipe.get("objects", []) if obj.get("articulation")]

    return RandomizerConfig(
        name="articulation_state",
        randomizer_type="joint_state",
        enabled=True,
        frequency=frequency,
        targets=[f"/Objects/{obj_id}" for obj_id in articulated],
        parameters={
            "open_probability": params.get("open_probability", 0.5),
            "state_distribution": "uniform",
            "normalize_range": True,
        },
    )


def _create_object_placement(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create object placement randomizer."""
    return RandomizerConfig(
        name="object_placement",
        randomizer_type="pose",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/*"],
        parameters={
            "position_noise": params.get("position_noise", 0.05),
            "rotation_noise": params.get("rotation_noise", 5),
            "maintain_surface_contact": True,
        },
    )


def _create_drawer_state(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create drawer state randomizer."""
    return RandomizerConfig(
        name="drawer_state",
        randomizer_type="joint_state",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/*/drawer*"],
        parameters={
            "open_range": params.get("open_range", [0.0, 1.0]),
            "distribution": "uniform",
        },
    )


def _create_drawer_contents(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create drawer contents randomizer."""
    return RandomizerConfig(
        name="drawer_contents",
        randomizer_type="scatter",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/*/drawer*/interior"],
        parameters={
            "fill_ratio": params.get("fill_ratio", [0.2, 0.8]),
            "asset_categories": ["office_supplies", "utensils", "accessories"],
            "collision_check": True,
        },
    )


def _create_door_state(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create door state randomizer."""
    return RandomizerConfig(
        name="door_state",
        randomizer_type="joint_state",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/*/door*"],
        parameters={
            "open_range": params.get("open_range", [0.0, 1.57]),
            "distribution": "uniform",
        },
    )


def _create_knob_state(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create knob state randomizer."""
    return RandomizerConfig(
        name="knob_state",
        randomizer_type="joint_state",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/*/knob*", "/Objects/*/dial*"],
        parameters={
            "rotation_range": params.get("rotation_range", [0.0, 6.28]),
            "distribution": "uniform",
        },
    )


def _create_cloth_scatter(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create cloth scatter randomizer."""
    return RandomizerConfig(
        name="cloth_scatter",
        randomizer_type="cloth_scatter",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/hamper*", "/Objects/basket*", "/Objects/bed*"],
        parameters={
            "min_items": params.get("min_items", 5),
            "max_items": params.get("max_items", 20),
            "asset_categories": ["clothing", "linens"],
            "deformation_enabled": True,
        },
    )


def _create_cloth_deformation(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create cloth deformation randomizer."""
    return RandomizerConfig(
        name="cloth_deformation",
        randomizer_type="cloth_simulation",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/*/cloth*"],
        parameters={
            "simulation_steps": params.get("simulation_steps", 10),
            "gravity_variation": [-0.2, 0.2],
            "wind_enabled": params.get("wind_enabled", False),
        },
    )


def _create_shelf_population(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create shelf population randomizer."""
    return RandomizerConfig(
        name="shelf_population",
        randomizer_type="planogram",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/shelf*", "/Objects/rack*"],
        parameters={
            "fill_ratio_range": params.get("fill_ratio_range", [0.3, 0.9]),
            "facing_probability": params.get("facing_probability", 0.9),
            "gap_probability": params.get("gap_probability", 0.1),
            "asset_categories": ["packaged_goods", "bottles", "cans"],
        },
    )


def _create_table_setup(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create table setup randomizer."""
    return RandomizerConfig(
        name="table_setup",
        randomizer_type="table_setting",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/table*", "/Objects/dining_table*"],
        parameters={
            "place_settings": params.get("place_settings", [1, 6]),
            "include_centerpiece": params.get("include_centerpiece", True),
            "setting_style": params.get("setting_style", "casual"),
            "asset_categories": ["dishes", "utensils", "glasses"],
        },
    )


def _create_dirty_state(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create dirty state randomizer for dishes."""
    return RandomizerConfig(
        name="dirty_state",
        randomizer_type="material_overlay",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/*/dish*", "/Objects/*/plate*", "/Objects/*/bowl*"],
        parameters={
            "dirty_probability": params.get("dirty_probability", 0.7),
            "overlay_types": ["food_residue", "water_spots", "grease"],
            "intensity_range": [0.1, 0.8],
        },
    )


def _create_dishwasher_state(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create dishwasher state randomizer."""
    return RandomizerConfig(
        name="dishwasher_state",
        randomizer_type="container_state",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/dishwasher*"],
        parameters={
            "door_state": params.get("door_state", "variable"),
            "loaded_probability": params.get("loaded_probability", 0.3),
            "rack_positions": ["upper", "lower"],
            "asset_categories": ["dishes", "utensils"],
        },
    )


def _create_switch_states(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create switch state randomizer."""
    return RandomizerConfig(
        name="switch_states",
        randomizer_type="binary_state",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/*/switch*", "/Objects/*/button*", "/Objects/*/breaker*"],
        parameters={
            "on_probability": params.get("on_probability", 0.5),
            "per_switch": True,
        },
    )


def _create_label_variation(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create label variation randomizer."""
    return RandomizerConfig(
        name="label_variation",
        randomizer_type="texture_swap",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/*/box*", "/Objects/*/package*"],
        parameters={
            "texture_library": params.get("texture_library", "shipping_labels"),
            "variation_per_instance": True,
        },
    )


def _create_pallet_placement(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create pallet placement randomizer."""
    return RandomizerConfig(
        name="pallet_placement",
        randomizer_type="pose",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/pallet*"],
        parameters={
            "position_noise": params.get("position_noise", 0.1),
            "rotation_noise": params.get("rotation_noise", 10),
            "snap_to_floor": True,
        },
    )


def _create_load_variation(
    params: dict[str, Any],
    frequency: str,
    recipe: dict[str, Any],
    environment: dict[str, Any],
) -> RandomizerConfig:
    """Create load variation randomizer."""
    return RandomizerConfig(
        name="load_variation",
        randomizer_type="stacking",
        enabled=True,
        frequency=frequency,
        targets=["/Objects/pallet*/load"],
        parameters={
            "stack_height_range": params.get("stack_height_range", [1, 4]),
            "box_types": ["small", "medium", "large"],
            "arrangement": params.get("arrangement", "aligned"),
        },
    )

