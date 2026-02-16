"""
Minimal, dependency-light physics presets for "SimReady Lite".

These are intentionally simple defaults to improve realism before data capture:
- mass (kg)
- friction (static/dynamic)
- restitution
- collision approximation hints (convexHull vs convexDecomposition)

Used by:
- Stage 7 Isaac Sim collector (pre-collection realism boost)
- BP postprocess (reporting + consistency checks)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PhysicsPreset:
    mass_kg: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    is_dynamic: bool
    collision_approximation: str
    collision_max_hulls: int


_DEFAULT_DYNAMIC = PhysicsPreset(
    mass_kg=1.0,
    static_friction=0.6,
    dynamic_friction=0.5,
    restitution=0.1,
    is_dynamic=True,
    collision_approximation="convexDecomposition",
    collision_max_hulls=64,
)

_DEFAULT_STATIC = PhysicsPreset(
    mass_kg=25.0,
    static_friction=0.7,
    dynamic_friction=0.6,
    restitution=0.05,
    is_dynamic=False,
    collision_approximation="convexHull",
    collision_max_hulls=128,
)

# Basic category presets (string contains match on obj type/category).
_CATEGORY_PRESETS: Dict[str, PhysicsPreset] = {
    # Surfaces / furniture (kinematic)
    "table": PhysicsPreset(25.0, 0.7, 0.6, 0.05, False, "convexHull", 128),
    "counter": PhysicsPreset(40.0, 0.8, 0.7, 0.05, False, "convexHull", 128),
    "desk": PhysicsPreset(25.0, 0.7, 0.6, 0.05, False, "convexHull", 128),
    "cabinet": PhysicsPreset(35.0, 0.8, 0.7, 0.05, False, "convexHull", 128),
    "shelf": PhysicsPreset(15.0, 0.7, 0.6, 0.05, False, "convexHull", 128),
    "bookshelf": PhysicsPreset(30.0, 0.7, 0.6, 0.05, False, "convexHull", 128),
    "sofa": PhysicsPreset(45.0, 0.8, 0.7, 0.05, False, "convexHull", 128),
    "chair": PhysicsPreset(8.0, 0.7, 0.6, 0.05, False, "convexHull", 128),
    "bed": PhysicsPreset(60.0, 0.8, 0.7, 0.05, False, "convexHull", 128),
    # Appliances (static)
    "stove": PhysicsPreset(60.0, 0.7, 0.6, 0.05, False, "convexHull", 128),
    "oven": PhysicsPreset(60.0, 0.7, 0.6, 0.05, False, "convexHull", 128),
    "refrigerator": PhysicsPreset(65.0, 0.7, 0.6, 0.05, False, "convexHull", 128),
    "fridge": PhysicsPreset(65.0, 0.7, 0.6, 0.05, False, "convexHull", 128),
    "microwave": PhysicsPreset(15.0, 0.7, 0.6, 0.05, False, "convexHull", 128),
    "dishwasher": PhysicsPreset(40.0, 0.7, 0.6, 0.05, False, "convexHull", 128),
    # Manipulables (dynamic)
    "mug": PhysicsPreset(0.3, 0.6, 0.5, 0.2, True, "convexDecomposition", 64),
    "cup": PhysicsPreset(0.25, 0.6, 0.5, 0.2, True, "convexDecomposition", 64),
    "bottle": PhysicsPreset(0.6, 0.6, 0.5, 0.1, True, "convexDecomposition", 64),
    "plate": PhysicsPreset(0.5, 0.5, 0.45, 0.1, True, "convexDecomposition", 64),
    "bowl": PhysicsPreset(0.4, 0.5, 0.45, 0.1, True, "convexDecomposition", 64),
    "glass": PhysicsPreset(0.3, 0.5, 0.4, 0.15, True, "convexDecomposition", 64),
    "jar": PhysicsPreset(0.4, 0.5, 0.4, 0.1, True, "convexDecomposition", 64),
    "salt": PhysicsPreset(0.3, 0.5, 0.4, 0.1, True, "convexDecomposition", 64),
    "pepper": PhysicsPreset(0.3, 0.5, 0.4, 0.1, True, "convexDecomposition", 64),
    "shaker": PhysicsPreset(0.3, 0.5, 0.4, 0.1, True, "convexDecomposition", 64),
    "kettle": PhysicsPreset(0.8, 0.6, 0.5, 0.1, True, "convexDecomposition", 64),
    "knife": PhysicsPreset(0.15, 0.5, 0.4, 0.05, True, "convexDecomposition", 64),
    "utensil": PhysicsPreset(0.1, 0.5, 0.4, 0.05, True, "convexDecomposition", 64),
    "holder": PhysicsPreset(0.35, 0.5, 0.4, 0.1, True, "convexDecomposition", 64),
    "book": PhysicsPreset(0.5, 0.6, 0.55, 0.05, True, "convexDecomposition", 64),
    "phone": PhysicsPreset(0.2, 0.6, 0.55, 0.05, True, "convexDecomposition", 64),
    "remote": PhysicsPreset(0.25, 0.6, 0.55, 0.05, True, "convexDecomposition", 64),
    "plant": PhysicsPreset(3.0, 0.7, 0.6, 0.05, True, "convexDecomposition", 64),
    "lamp": PhysicsPreset(2.0, 0.6, 0.5, 0.1, True, "convexDecomposition", 64),
    "clock": PhysicsPreset(0.7, 0.5, 0.4, 0.1, False, "convexHull", 64),
    "vase": PhysicsPreset(1.0, 0.5, 0.4, 0.15, True, "convexDecomposition", 64),
    "candle": PhysicsPreset(0.3, 0.5, 0.4, 0.05, True, "convexDecomposition", 64),
    "picture": PhysicsPreset(1.5, 0.5, 0.4, 0.05, False, "convexHull", 64),
    "frame": PhysicsPreset(1.0, 0.5, 0.4, 0.05, False, "convexHull", 64),
}


def _as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _volume_m3(dims: Dict[str, Any]) -> float:
    w = _as_float(dims.get("width"), 0.0)
    l = _as_float(dims.get("length"), 0.0) or _as_float(dims.get("depth"), 0.0)
    h = _as_float(dims.get("height"), 0.0)
    if w <= 0.0 or l <= 0.0 or h <= 0.0:
        return 0.0
    return w * l * h


def recommend_simready_lite(
    *,
    category: str,
    dimensions: Optional[Dict[str, Any]] = None,
) -> PhysicsPreset:
    """
    Return a stable, simple physics preset.

    Args:
        category: object category/type string
        dimensions: dict with width/length/height (meters) when available
    """
    cat = (category or "").strip().lower().replace(" ", "_")
    preset = None
    for key, value in _CATEGORY_PRESETS.items():
        if key in cat:
            preset = value
            break

    dims = dimensions or {}
    vol = _volume_m3(dims)

    if preset is not None:
        # If the category suggests static but it's tiny, treat as dynamic.
        if (not preset.is_dynamic) and vol > 0.0 and vol < 0.01:
            return _DEFAULT_DYNAMIC
        # Sanity clamp: small dynamic objects should not exceed a reasonable mass.
        if preset.is_dynamic and vol > 0.0:
            # Density sanity: water is ~1000 kg/m³. Cap at 2000 kg/m³ for solid objects.
            max_mass = max(0.5, vol * 2000.0)
            if preset.mass_kg > max_mass:
                return PhysicsPreset(
                    mass_kg=max_mass,
                    static_friction=preset.static_friction,
                    dynamic_friction=preset.dynamic_friction,
                    restitution=preset.restitution,
                    is_dynamic=preset.is_dynamic,
                    collision_approximation=preset.collision_approximation,
                    collision_max_hulls=preset.collision_max_hulls,
                )
        return preset

    # Heuristic fallback.
    if vol > 0.0 and vol < 0.01:
        return _DEFAULT_DYNAMIC
    return _DEFAULT_STATIC


def preset_to_dict(preset: PhysicsPreset) -> Dict[str, Any]:
    return {
        "mass_kg": preset.mass_kg,
        "static_friction": preset.static_friction,
        "dynamic_friction": preset.dynamic_friction,
        "restitution": preset.restitution,
        "is_dynamic": preset.is_dynamic,
        "collision_approximation": preset.collision_approximation,
        "collision_max_hulls": preset.collision_max_hulls,
    }

