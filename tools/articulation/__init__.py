"""
Articulation Detection Tools.

Enhanced detection of articulated joints in furniture and objects.
"""

from .detector import (
    ArticulationDetector,
    ArticulationType,
    ArticulationResult,
    detect_scene_articulations,
)
from .label_hints import (
    infer_primary_joint_type,
    parse_label_articulation_hint,
    parse_label_articulation_hints,
)

__all__ = [
    "ArticulationDetector",
    "ArticulationType",
    "ArticulationResult",
    "detect_scene_articulations",
    "infer_primary_joint_type",
    "parse_label_articulation_hint",
    "parse_label_articulation_hints",
]
