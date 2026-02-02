"""
Scene Graph Converter for Genie Sim 3.0.

Converts BlueprintPipeline's scene_manifest.json to Genie Sim's hierarchical
scene graph format with nodes (objects) and edges (spatial relations).

Genie Sim Scene Graph Structure:
    - Nodes: Objects encoded with asset_id, semantic, size, pose, task_tag
    - Edges: Spatial relations: on, in, adjacent, aligned, stacked

References:
    - Genie Sim 3.0 Paper: https://arxiv.org/html/2601.02078v1
"""

from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import logging
import math
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.logging_config import init_logging
from tools.config.production_mode import resolve_production_mode

init_logging()
logger = logging.getLogger(__name__)

# Import validation utilities
try:
    from tools.validation import (
        sanitize_string,
        validate_object_id,
        validate_category,
        validate_description,
        validate_dimensions,
        validate_quaternion,
        ValidationError,
        ALLOWED_ASSET_CATEGORIES,
    )
    HAVE_VALIDATION_TOOLS = True
except ImportError:
    HAVE_VALIDATION_TOOLS = False
    ValidationError = ValueError
    logger.warning("Input validation tools not available - XSS/injection protection disabled")

# Import streaming JSON parser
try:
    from tools.performance import StreamingManifestParser, stream_manifest_objects
    HAVE_STREAMING_PARSER = True
except ImportError:
    HAVE_STREAMING_PARSER = False
    logger.warning("Streaming JSON parser not available - may OOM on large manifests (>1000 objects)")

try:
    from tools.config import load_pipeline_config
    HAVE_PIPELINE_CONFIG = True
except ImportError:
    HAVE_PIPELINE_CONFIG = False
    logger.warning("Pipeline config loader not available - scene graph config will use defaults only")


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class Pose:
    """6-DOF pose in Genie Sim format."""

    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # wxyz quaternion

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "position": self.position,
            "orientation": self.orientation,
        }


@dataclass
class GenieSimNode:
    """A node in the Genie Sim scene graph representing an object."""

    asset_id: str
    semantic: str
    size: List[float]  # [width, depth, height]
    pose: Pose
    task_tag: List[str]
    usd_path: str
    properties: Dict[str, Any] = field(default_factory=dict)

    # BlueprintPipeline-specific metadata (preserved for traceability)
    bp_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "semantic": self.semantic,
            "size": self.size,
            "pose": self.pose.to_dict(),
            "task_tag": self.task_tag,
            "usd_path": self.usd_path,
            "properties": self.properties,
            "bp_metadata": self.bp_metadata,
        }


@dataclass
class GenieSimEdge:
    """An edge in the Genie Sim scene graph representing a spatial relation."""

    source: str  # source object asset_id
    target: str  # target object asset_id
    relation: str  # on, in, adjacent, aligned, stacked
    confidence: float = 1.0  # Confidence of inferred relation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class GenieSimSceneGraph:
    """Complete Genie Sim scene graph."""

    scene_id: str
    coordinate_system: str  # y_up or z_up
    meters_per_unit: float
    nodes: List[GenieSimNode]
    edges: List[GenieSimEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "3.0",
            "scene_id": self.scene_id,
            "coordinate_system": self.coordinate_system,
            "meters_per_unit": self.meters_per_unit,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        """Save scene graph to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class SceneGraphRuntimeConfig:
    vertical_proximity_threshold: float
    horizontal_proximity_threshold: float
    alignment_angle_threshold: float
    streaming_batch_size: int
    enable_physics_validation: bool
    physics_contact_depth_threshold: float
    physics_containment_ratio_threshold: float
    heuristic_confidence_scale: float
    allow_unvalidated_input: bool
    require_physics_validation: bool


_SCENE_GRAPH_CONFIG: Optional[SceneGraphRuntimeConfig] = None
# Threshold ranges follow Genie Sim 3.0 scene graph relation heuristics
# (see "Scene Graph Relations" section in the Genie Sim 3.0 paper).
_SCENE_GRAPH_THRESHOLD_RANGES = {
    "vertical_proximity_threshold": (0.0, 0.2),  # meters
    "horizontal_proximity_threshold": (0.0, 0.5),  # meters
    "alignment_angle_threshold": (0.0, 45.0),  # degrees
}
_FALLBACK_ALLOWED_CATEGORIES = frozenset(
    {
        "object",
        "unknown",
        "cup",
        "mug",
        "plate",
        "bowl",
        "utensil",
        "bottle",
        "pot",
        "pan",
        "microwave",
        "refrigerator",
        "oven",
        "dishwasher",
        "sink",
        "faucet",
        "countertop",
        "cabinet",
        "box",
        "package",
        "carton",
        "tote",
        "pallet",
        "shelf",
        "rack",
        "conveyor",
        "desk",
        "chair",
        "monitor",
        "keyboard",
        "mouse",
        "phone",
        "book",
        "pen",
        "drawer",
        "filing_cabinet",
    }
)
_FALLBACK_OBJECT_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_\-]+")


def _sanitize_fallback_text(value: Any, max_length: int) -> str:
    if value is None:
        return ""
    sanitized = html.escape(str(value), quote=True)
    sanitized = "".join(ch for ch in sanitized if ch.isprintable())
    return sanitized.strip()[:max_length]


def _sanitize_fallback_object_id(value: str, max_length: int = 128) -> str:
    sanitized = _FALLBACK_OBJECT_ID_PATTERN.sub("", value).strip()[:max_length]
    if sanitized:
        return sanitized
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"obj_{digest}"


def _sanitize_fallback_category(value: Any) -> Tuple[str, bool]:
    sanitized = _sanitize_fallback_text(value, max_length=64).lower()
    sanitized = _FALLBACK_OBJECT_ID_PATTERN.sub("", sanitized)
    if sanitized in _FALLBACK_ALLOWED_CATEGORIES:
        return sanitized, False
    return "object", True


def _resolve_scene_id_from_manifest(manifest: Dict[str, Any]) -> str:
    metadata = manifest.get("metadata")
    if isinstance(metadata, dict):
        scene_id = metadata.get("scene_id")
        if scene_id:
            return str(scene_id)
    return str(manifest.get("scene_id", "unknown"))


def _namespaced_asset_id(scene_id: Optional[str], obj_id: str) -> str:
    if not scene_id or scene_id == "unknown":
        return obj_id
    scene_prefix = f"{scene_id}_obj_"
    if obj_id.startswith(scene_prefix) or obj_id.startswith(f"{scene_id}:"):
        return obj_id
    return f"{scene_id}_obj_{obj_id}"


def _strip_asset_namespace(asset_id: str) -> str:
    if "_obj_" in asset_id:
        return asset_id.split("_obj_", 1)[1]
    if ":" in asset_id:
        return asset_id.split(":", 1)[1]
    return asset_id


def _resolve_env_float(key: str, default: float, default_source: str) -> Tuple[float, str]:
    raw = os.getenv(key)
    if raw is None:
        return default, default_source
    try:
        return float(raw), f"env:{key}"
    except ValueError:
        logger.warning("Invalid %s=%s; using default %.3f", key, raw, default)
        return default, default_source


def _resolve_env_int(key: str, default: int, default_source: str) -> Tuple[int, str]:
    raw = os.getenv(key)
    if raw is None:
        return default, default_source
    try:
        return int(raw), f"env:{key}"
    except ValueError:
        logger.warning("Invalid %s=%s; using default %d", key, raw, default)
        return default, default_source


def _resolve_env_bool(key: str, default: bool, default_source: str) -> Tuple[bool, str]:
    raw = os.getenv(key)
    if raw is None:
        return default, default_source
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True, f"env:{key}"
    if normalized in {"0", "false", "no", "off"}:
        return False, f"env:{key}"
    logger.warning("Invalid %s=%s; using default %s", key, raw, default)
    return default, default_source


def _validate_scene_graph_threshold(
    name: str,
    value: float,
    source: str,
) -> float:
    min_value, max_value = _SCENE_GRAPH_THRESHOLD_RANGES[name]
    if min_value <= value <= max_value:
        return value
    message = (
        f"{name}={value} out of range [{min_value}, {max_value}] per Genie Sim 3.0 "
        f"scene graph relation heuristics. Source: {source}."
    )
    if resolve_production_mode():
        raise ValueError(message)
    logger.warning(message)
    return value


def _is_isaac_sim_available() -> bool:
    """Check if Isaac Sim is available in the environment.

    Returns True if either:
    1. ISAAC_SIM_PATH environment variable is set to a valid path
    2. isaacsim.core.api can be imported (Isaac Sim Python environment)
    """
    # Check environment variable first (fastest check)
    isaac_sim_path = os.getenv("ISAAC_SIM_PATH")
    if isaac_sim_path:
        isaac_sim_path = Path(isaac_sim_path)
        if isaac_sim_path.exists() and isaac_sim_path.is_dir():
            logger.debug("Isaac Sim detected via ISAAC_SIM_PATH: %s", isaac_sim_path)
            return True

    # Check for Isaac Sim Python module
    try:
        import importlib.util
        spec = importlib.util.find_spec("isaacsim.core.api")
        if spec is not None:
            logger.debug("Isaac Sim detected via isaacsim.core.api module")
            return True
    except (ImportError, ModuleNotFoundError):
        pass

    return False


def _resolve_physics_validation_default(
    *,
    require_physics_validation: bool,
    allow_heuristics_in_prod: bool,
) -> bool:
    """P0 FIX: Resolve physics validation default based on environment.

    Physics validation ensures spatial relations (on, in) are physically accurate
    by using actual physics simulation data rather than heuristics.

    In production mode:
    - If Isaac Sim is available: ENABLE physics validation (more accurate)
    - If Isaac Sim is not available: ERROR unless heuristics are explicitly allowed

    In development mode:
    - Default to False (faster iteration)

    The BP_SCENE_GRAPH_ENABLE_PHYSICS_VALIDATION env var can override this default.
    """
    production_mode = resolve_production_mode()

    if not production_mode:
        logger.debug(
            "Physics validation disabled by default in development mode. "
            "Set BP_SCENE_GRAPH_ENABLE_PHYSICS_VALIDATION=true to enable."
        )
        return False

    # Production mode: enable if Isaac Sim is available
    isaac_available = _is_isaac_sim_available()

    if isaac_available:
        logger.info(
            "P0 FIX: Physics validation ENABLED in production (Isaac Sim detected). "
            "Spatial relations will be validated against physics simulation data."
        )
        return True
    if allow_heuristics_in_prod or not require_physics_validation:
        logger.warning(
            "P0 WARNING: Physics validation explicitly disabled in production while Isaac Sim is unavailable. "
            "Spatial relations will use heuristics only. "
            "Set ISAAC_SIM_PATH to enable physics validation."
        )
        return False

    raise RuntimeError(
        "Physics validation is required in production, but Isaac Sim is unavailable. "
        "Set ISAAC_SIM_PATH to the Isaac Sim installation directory or "
        "explicitly disable strict mode by setting "
        "scene_graph.validation.require_physics_validation=false or "
        "BP_SCENE_GRAPH_ALLOW_HEURISTICS_IN_PROD=true."
    )


def _load_scene_graph_runtime_config() -> SceneGraphRuntimeConfig:
    global _SCENE_GRAPH_CONFIG
    if _SCENE_GRAPH_CONFIG is not None:
        return _SCENE_GRAPH_CONFIG

    if HAVE_PIPELINE_CONFIG:
        pipeline_config = load_pipeline_config()
        relation_config = pipeline_config.scene_graph.relation_inference
        streaming_config = pipeline_config.scene_graph.streaming
        validation_config = pipeline_config.scene_graph.validation
        vertical_default = relation_config.vertical_proximity_threshold
        horizontal_default = relation_config.horizontal_proximity_threshold
        alignment_default = relation_config.alignment_angle_threshold
        physics_validation_default = relation_config.enable_physics_validation
        contact_depth_default = relation_config.physics_contact_depth_threshold
        containment_ratio_default = relation_config.physics_containment_ratio_threshold
        heuristic_confidence_default = relation_config.heuristic_confidence_scale
        batch_default = streaming_config.batch_size
        allow_unvalidated_default = validation_config.allow_unvalidated_input
        require_physics_validation_default = validation_config.require_physics_validation
        vertical_source = (
            "pipeline_config.scene_graph.relation_inference.vertical_proximity_threshold"
        )
        horizontal_source = (
            "pipeline_config.scene_graph.relation_inference.horizontal_proximity_threshold"
        )
        alignment_source = (
            "pipeline_config.scene_graph.relation_inference.alignment_angle_threshold"
        )
        batch_source = "pipeline_config.scene_graph.streaming.batch_size"
        physics_validation_source = (
            "pipeline_config.scene_graph.relation_inference.enable_physics_validation"
        )
        contact_depth_source = (
            "pipeline_config.scene_graph.relation_inference.physics_contact_depth_threshold"
        )
        containment_ratio_source = (
            "pipeline_config.scene_graph.relation_inference.physics_containment_ratio_threshold"
        )
        heuristic_confidence_source = (
            "pipeline_config.scene_graph.relation_inference.heuristic_confidence_scale"
        )
        allow_unvalidated_source = "pipeline_config.scene_graph.validation.allow_unvalidated_input"
        require_physics_validation_source = (
            "pipeline_config.scene_graph.validation.require_physics_validation"
        )
    else:
        require_physics_validation_default = resolve_production_mode()
        vertical_default = 0.05
        horizontal_default = 0.15
        alignment_default = 5.0
        # P0 FIX: Enable physics validation by default when Isaac Sim is available
        # Physics validation ensures spatial relations (on, in) are physically accurate
        # In production, heuristics-only validation can produce invalid relations
        physics_validation_default = _resolve_physics_validation_default(
            require_physics_validation=require_physics_validation_default,
            allow_heuristics_in_prod=os.getenv("BP_SCENE_GRAPH_ALLOW_HEURISTICS_IN_PROD", "").strip().lower()
            in {"1", "true", "yes", "on"},
        )
        contact_depth_default = 0.001
        containment_ratio_default = 0.8
        heuristic_confidence_default = 0.6
        batch_default = 100
        allow_unvalidated_default = True
        vertical_source = "scene_graph.defaults.vertical_proximity_threshold"
        horizontal_source = "scene_graph.defaults.horizontal_proximity_threshold"
        alignment_source = "scene_graph.defaults.alignment_angle_threshold"
        batch_source = "scene_graph.defaults.streaming_batch_size"
        physics_validation_source = "scene_graph.defaults.enable_physics_validation (auto-detected)"
        contact_depth_source = "scene_graph.defaults.physics_contact_depth_threshold"
        containment_ratio_source = "scene_graph.defaults.physics_containment_ratio_threshold"
        heuristic_confidence_source = "scene_graph.defaults.heuristic_confidence_scale"
        allow_unvalidated_source = "scene_graph.defaults.allow_unvalidated_input"
        require_physics_validation_source = "scene_graph.defaults.require_physics_validation"

    vertical_value, vertical_value_source = _resolve_env_float(
        "BP_SCENE_GRAPH_VERTICAL_PROXIMITY_THRESHOLD",
        vertical_default,
        vertical_source,
    )
    horizontal_value, horizontal_value_source = _resolve_env_float(
        "BP_SCENE_GRAPH_HORIZONTAL_PROXIMITY_THRESHOLD",
        horizontal_default,
        horizontal_source,
    )
    alignment_value, alignment_value_source = _resolve_env_float(
        "BP_SCENE_GRAPH_ALIGNMENT_ANGLE_THRESHOLD",
        alignment_default,
        alignment_source,
    )
    streaming_batch_value, streaming_batch_source = _resolve_env_int(
        "BP_SCENE_GRAPH_STREAMING_BATCH_SIZE",
        batch_default,
        batch_source,
    )
    physics_validation_value, physics_validation_source = _resolve_env_bool(
        "BP_SCENE_GRAPH_ENABLE_PHYSICS_VALIDATION",
        physics_validation_default,
        physics_validation_source,
    )
    contact_depth_value, contact_depth_source = _resolve_env_float(
        "BP_SCENE_GRAPH_PHYSICS_CONTACT_DEPTH_THRESHOLD",
        contact_depth_default,
        contact_depth_source,
    )
    containment_ratio_value, containment_ratio_source = _resolve_env_float(
        "BP_SCENE_GRAPH_PHYSICS_CONTAINMENT_RATIO_THRESHOLD",
        containment_ratio_default,
        containment_ratio_source,
    )
    heuristic_confidence_value, heuristic_confidence_source = _resolve_env_float(
        "BP_SCENE_GRAPH_HEURISTIC_CONFIDENCE_SCALE",
        heuristic_confidence_default,
        heuristic_confidence_source,
    )
    allow_unvalidated_value, allow_unvalidated_source = _resolve_env_bool(
        "BP_SCENE_GRAPH_ALLOW_UNVALIDATED_INPUT",
        allow_unvalidated_default,
        allow_unvalidated_source,
    )
    require_physics_validation_value, require_physics_validation_source = _resolve_env_bool(
        "BP_SCENE_GRAPH_REQUIRE_PHYSICS_VALIDATION",
        require_physics_validation_default,
        require_physics_validation_source,
    )
    allow_heuristics_in_prod_value, allow_heuristics_in_prod_source = _resolve_env_bool(
        "BP_SCENE_GRAPH_ALLOW_HEURISTICS_IN_PROD",
        False,
        "scene_graph.defaults.allow_heuristics_in_prod",
    )

    production_mode = resolve_production_mode()
    isaac_available = _is_isaac_sim_available()
    explicit_disable_reason: Optional[str] = None
    if production_mode and not isaac_available:
        if allow_heuristics_in_prod_value:
            explicit_disable_reason = f"{allow_heuristics_in_prod_source}=true"
            physics_validation_value = False
        elif not require_physics_validation_value:
            explicit_disable_reason = f"{require_physics_validation_source}=false"
            physics_validation_value = False
        else:
            raise RuntimeError(
                "Physics validation is required in production, but Isaac Sim is unavailable. "
                "Set ISAAC_SIM_PATH to the Isaac Sim installation directory or "
                "explicitly disable strict mode by setting "
                "scene_graph.validation.require_physics_validation=false or "
                "BP_SCENE_GRAPH_ALLOW_HEURISTICS_IN_PROD=true."
            )

    config = SceneGraphRuntimeConfig(
        vertical_proximity_threshold=_validate_scene_graph_threshold(
            "vertical_proximity_threshold",
            vertical_value,
            vertical_value_source,
        ),
        horizontal_proximity_threshold=_validate_scene_graph_threshold(
            "horizontal_proximity_threshold",
            horizontal_value,
            horizontal_value_source,
        ),
        alignment_angle_threshold=_validate_scene_graph_threshold(
            "alignment_angle_threshold",
            alignment_value,
            alignment_value_source,
        ),
        streaming_batch_size=streaming_batch_value,
        enable_physics_validation=physics_validation_value,
        physics_contact_depth_threshold=contact_depth_value,
        physics_containment_ratio_threshold=containment_ratio_value,
        heuristic_confidence_scale=heuristic_confidence_value,
        allow_unvalidated_input=allow_unvalidated_value,
        require_physics_validation=require_physics_validation_value,
    )

    if not config.enable_physics_validation:
        if explicit_disable_reason:
            logger.warning(
                "Physics validation explicitly disabled (%s). Heuristic relation inference will be used.",
                explicit_disable_reason,
            )
        elif physics_validation_source.startswith(("env:", "pipeline_config")):
            logger.info(
                "Physics validation explicitly disabled via %s.",
                physics_validation_source,
            )
        elif not isaac_available:
            logger.warning(
                "Physics validation implicitly disabled because Isaac Sim is not available. "
                "Set ISAAC_SIM_PATH to enable physics validation.",
            )

    logger.info(
        "Scene graph config: vertical_proximity_threshold=%.3f, horizontal_proximity_threshold=%.3f, "
        "alignment_angle_threshold=%.3f, streaming_batch_size=%d, "
        "enable_physics_validation=%s, physics_contact_depth_threshold=%.4f, "
        "physics_containment_ratio_threshold=%.3f, heuristic_confidence_scale=%.2f, "
        "allow_unvalidated_input=%s, require_physics_validation=%s",
        config.vertical_proximity_threshold,
        config.horizontal_proximity_threshold,
        config.alignment_angle_threshold,
        config.streaming_batch_size,
        config.enable_physics_validation,
        config.physics_contact_depth_threshold,
        config.physics_containment_ratio_threshold,
        config.heuristic_confidence_scale,
        config.allow_unvalidated_input,
        config.require_physics_validation,
    )

    _SCENE_GRAPH_CONFIG = config
    return config


# =============================================================================
# Task Tag Mapping
# =============================================================================

# Maps (sim_role, affordance) -> task_tags
TASK_TAG_MAPPING = {
    # Manipulable objects
    ("manipulable_object", "Graspable"): ["pick", "place"],
    ("manipulable_object", "Stackable"): ["pick", "place", "stack"],
    ("manipulable_object", "Pourable"): ["pick", "pour"],
    ("manipulable_object", "Insertable"): ["pick", "insert"],
    ("manipulable_object", "Hangable"): ["pick", "hang"],

    # Articulated furniture
    ("articulated_furniture", "Openable"): ["open", "close"],
    ("articulated_furniture", "Slidable"): ["pull", "push"],

    # Articulated appliances
    ("articulated_appliance", "Turnable"): ["turn"],
    ("articulated_appliance", "Pressable"): ["press"],
    ("articulated_appliance", "Openable"): ["open", "close"],

    # Interactive objects
    ("interactive", None): ["interact"],

    # Static/surfaces
    ("static", "Supportable"): ["place_on"],
    ("static", "Placeable"): ["place_on"],

    # Containers
    ("manipulable_object", "Containable"): ["pick", "place", "fill"],
    ("manipulable_object", "Fillable"): ["pour_into", "fill"],
}

# Default task tags by sim_role
DEFAULT_TASK_TAGS = {
    "manipulable_object": ["pick", "place"],
    "articulated_furniture": ["open", "close"],
    "articulated_appliance": ["interact"],
    "interactive": ["interact"],
    "static": [],
    "clutter": ["pick", "place"],
    "background": [],
    "scene_shell": [],
    "unknown": [],
}


# =============================================================================
# Relation Inference
# =============================================================================

# Relationship type mapping from BlueprintPipeline to Genie Sim
RELATION_TYPE_MAPPING = {
    "on_top_of": "on",
    "on": "on",
    "inside": "in",
    "in": "in",
    "contains": "in",  # inverse
    "next_to": "adjacent",
    "beside": "adjacent",
    "near": "adjacent",
    "adjacent": "adjacent",
    "aligned_with": "aligned",
    "aligned": "aligned",
    "stacked_on": "stacked",
    "stacked": "stacked",
    "under": "on",  # inverse (target is on source)
    "above": "on",  # (source is on target)
}


class RelationInferencer:
    """Infers spatial relations between objects when not explicitly provided."""

    CONTAINMENT_MARGIN = 0.02  # 2cm margin for containment check
    CACHE_MAX_ENTRIES = 32
    HASH_DECIMAL_PLACES = 6

    def __init__(self, verbose: bool = False, config: Optional[SceneGraphRuntimeConfig] = None):
        self.verbose = verbose
        self.config = config or _load_scene_graph_runtime_config()
        self.vertical_proximity_threshold = self.config.vertical_proximity_threshold
        self.horizontal_proximity_threshold = self.config.horizontal_proximity_threshold
        self.alignment_angle_threshold = self.config.alignment_angle_threshold
        self.enable_physics_validation = self.config.enable_physics_validation
        self.physics_contact_depth_threshold = self.config.physics_contact_depth_threshold
        self.physics_containment_ratio_threshold = self.config.physics_containment_ratio_threshold
        self.heuristic_confidence_scale = self.config.heuristic_confidence_scale
        self._cache: "OrderedDict[Tuple[str, str], List[GenieSimEdge]]" = OrderedDict()
        self.last_cache_hit = False
        self._cache_hits = 0
        self._cache_misses = 0

    def log(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if self.verbose:
            logger.info("[RELATION-INFERENCER] %s", msg, extra=extra)

    def infer_relations(
        self,
        nodes: List[GenieSimNode],
        scene_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, float], None]] = None,
        progress_interval_s: float = 5.0,
        progress_every_pairs: Optional[int] = None,
    ) -> List[GenieSimEdge]:
        """
        Infer spatial relations between nodes.

        Inference rules:
        1. Vertical proximity + not floor contact → "on" edge
        2. Bounds containment → "in" edge
        3. Horizontal proximity (< threshold) → "adjacent" edge
        4. Similar rotation (< angle threshold) → "aligned" edge
        """
        cache_key = self._build_cache_key(scene_id or "unknown", nodes)
        cached_edges = self._cache.get(cache_key)
        if cached_edges is not None:
            self._cache.move_to_end(cache_key)
            self.last_cache_hit = True
            self._cache_hits += 1
            self.log(
                f"Cache hit for scene_id={scene_id or 'unknown'} "
                f"({len(nodes)} nodes).",
                extra={"scene_id": scene_id or "unknown"},
            )
            return list(cached_edges)

        self.last_cache_hit = False
        self._cache_misses += 1
        edges: List[GenieSimEdge] = []
        start_time = time.monotonic()
        last_report_time = start_time
        processed_pairs = 0
        total_pairs = (len(nodes) * (len(nodes) - 1)) // 2

        def report_progress(force: bool = False) -> None:
            nonlocal last_report_time
            if processed_pairs == 0:
                return
            now = time.monotonic()
            should_report = force
            if progress_every_pairs and processed_pairs % progress_every_pairs == 0:
                should_report = True
            if progress_interval_s and now - last_report_time >= progress_interval_s:
                should_report = True
            if not should_report:
                return
            elapsed = now - start_time
            if progress_callback:
                progress_callback(processed_pairs, elapsed)
            if self.verbose:
                self.log(
                    f"Relation inference progress: {processed_pairs}/{total_pairs} "
                    f"pairs in {elapsed:.1f}s",
                    extra={"scene_id": scene_id or "unknown"},
                )
            last_report_time = now

        for i, node_a in enumerate(nodes):
            for j, node_b in enumerate(nodes):
                if i >= j:
                    continue

                processed_pairs += 1
                report_progress()

                # Check for "on" relation
                on_edge = self._check_on_relation(node_a, node_b)
                if on_edge:
                    edges.append(on_edge)
                    continue

                # Check for "in" relation
                in_edge = self._check_in_relation(node_a, node_b)
                if in_edge:
                    edges.append(in_edge)
                    continue

                # Check for "adjacent" relation
                adj_edge = self._check_adjacent_relation(node_a, node_b)
                if adj_edge:
                    edges.append(adj_edge)

                # Check for "aligned" relation (can coexist with others)
                aligned_edge = self._check_aligned_relation(node_a, node_b)
                if aligned_edge:
                    edges.append(aligned_edge)

        report_progress(force=True)
        self.log(f"Inferred {len(edges)} relations from {len(nodes)} nodes")
        self._store_cache_entry(cache_key, edges)
        return edges

    def _build_cache_key(self, scene_id: str, nodes: List[GenieSimNode]) -> Tuple[str, str]:
        node_hashes = [self._hash_node(node) for node in nodes]
        payload = json.dumps(
            {
                "scene_id": scene_id,
                "node_hashes": node_hashes,
                "node_count": len(nodes),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = self._sha256(payload)
        return scene_id, digest

    def _hash_node(self, node: GenieSimNode) -> str:
        payload = json.dumps(
            {
                "asset_id": node.asset_id,
                "position": self._round_floats(node.pose.position),
                "orientation": self._round_floats(node.pose.orientation),
                "size": self._round_floats(node.size),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return self._sha256(payload)

    def _round_floats(self, values: List[float]) -> List[float]:
        return [round(float(value), self.HASH_DECIMAL_PLACES) for value in values]

    def _sha256(self, payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _store_cache_entry(self, cache_key: Tuple[str, str], edges: List[GenieSimEdge]) -> None:
        self._cache[cache_key] = list(edges)
        self._cache.move_to_end(cache_key)
        if len(self._cache) > self.CACHE_MAX_ENTRIES:
            self._cache.popitem(last=False)

    def _check_on_relation(
        self,
        node_a: GenieSimNode,
        node_b: GenieSimNode,
    ) -> Optional[GenieSimEdge]:
        """Check if node_a is ON node_b (or vice versa)."""
        pos_a = np.array(node_a.pose.position)
        pos_b = np.array(node_b.pose.position)
        size_a = np.array(node_a.size)
        size_b = np.array(node_b.size)

        # Vertical separation
        vertical_diff = pos_a[2] - pos_b[2]

        # Check if A is above B
        if vertical_diff > 0:
            # A's bottom should be near B's top
            a_bottom = pos_a[2] - size_a[2] / 2
            b_top = pos_b[2] + size_b[2] / 2

            if abs(a_bottom - b_top) < self.vertical_proximity_threshold:
                # Check horizontal overlap
                if self._has_horizontal_overlap(node_a, node_b):
                    edge = GenieSimEdge(
                        source=node_a.asset_id,
                        target=node_b.asset_id,
                        relation="on",
                        confidence=0.8,
                    )
                    return self._apply_physics_validation(edge, node_a, node_b)

        # Check if B is above A
        elif vertical_diff < 0:
            b_bottom = pos_b[2] - size_b[2] / 2
            a_top = pos_a[2] + size_a[2] / 2

            if abs(b_bottom - a_top) < self.vertical_proximity_threshold:
                if self._has_horizontal_overlap(node_a, node_b):
                    edge = GenieSimEdge(
                        source=node_b.asset_id,
                        target=node_a.asset_id,
                        relation="on",
                        confidence=0.8,
                    )
                    return self._apply_physics_validation(edge, node_b, node_a)

        return None

    def _check_in_relation(
        self,
        node_a: GenieSimNode,
        node_b: GenieSimNode,
    ) -> Optional[GenieSimEdge]:
        """Check if node_a is IN node_b (or vice versa)."""
        # Get bounds
        bounds_a = self._get_bounds(node_a)
        bounds_b = self._get_bounds(node_b)

        # Check if A is inside B (with margin)
        if self._is_inside(bounds_a, bounds_b, self.CONTAINMENT_MARGIN):
            edge = GenieSimEdge(
                source=node_a.asset_id,
                target=node_b.asset_id,
                relation="in",
                confidence=0.9,
            )
            return self._apply_physics_validation(edge, node_a, node_b)

        # Check if B is inside A
        if self._is_inside(bounds_b, bounds_a, self.CONTAINMENT_MARGIN):
            edge = GenieSimEdge(
                source=node_b.asset_id,
                target=node_a.asset_id,
                relation="in",
                confidence=0.9,
            )
            return self._apply_physics_validation(edge, node_b, node_a)

        return None

    def _check_adjacent_relation(
        self,
        node_a: GenieSimNode,
        node_b: GenieSimNode,
    ) -> Optional[GenieSimEdge]:
        """Check if node_a and node_b are adjacent."""
        pos_a = np.array(node_a.pose.position)
        pos_b = np.array(node_b.pose.position)
        size_a = np.array(node_a.size)
        size_b = np.array(node_b.size)

        # Horizontal distance (ignoring vertical)
        horiz_dist = np.linalg.norm(pos_a[:2] - pos_b[:2])

        # Expected separation if adjacent
        expected_sep = (size_a[0] + size_b[0]) / 2  # Use width as proxy

        # Check if close but not overlapping
        if horiz_dist < expected_sep + self.horizontal_proximity_threshold:
            if horiz_dist > expected_sep * 0.5:  # Not overlapping too much
                return GenieSimEdge(
                    source=node_a.asset_id,
                    target=node_b.asset_id,
                    relation="adjacent",
                    confidence=0.7,
                )

        return None

    def _check_aligned_relation(
        self,
        node_a: GenieSimNode,
        node_b: GenieSimNode,
    ) -> Optional[GenieSimEdge]:
        """Check if node_a and node_b are aligned (similar orientation)."""
        quat_a = np.array(node_a.pose.orientation)
        quat_b = np.array(node_b.pose.orientation)

        # Calculate angle between quaternions
        dot = np.abs(np.dot(quat_a, quat_b))
        dot = min(1.0, max(-1.0, dot))
        angle_rad = 2 * math.acos(dot)
        angle_deg = math.degrees(angle_rad)

        if angle_deg < self.alignment_angle_threshold:
            return GenieSimEdge(
                source=node_a.asset_id,
                target=node_b.asset_id,
                relation="aligned",
                confidence=0.6,
            )

        return None

    def _has_horizontal_overlap(
        self,
        node_a: GenieSimNode,
        node_b: GenieSimNode,
    ) -> bool:
        """Check if two nodes overlap horizontally."""
        pos_a = np.array(node_a.pose.position)
        pos_b = np.array(node_b.pose.position)
        size_a = np.array(node_a.size)
        size_b = np.array(node_b.size)

        # Check X overlap
        x_overlap = (
            pos_a[0] - size_a[0] / 2 < pos_b[0] + size_b[0] / 2 and
            pos_a[0] + size_a[0] / 2 > pos_b[0] - size_b[0] / 2
        )

        # Check Y overlap
        y_overlap = (
            pos_a[1] - size_a[1] / 2 < pos_b[1] + size_b[1] / 2 and
            pos_a[1] + size_a[1] / 2 > pos_b[1] - size_b[1] / 2
        )

        return x_overlap and y_overlap

    def _get_bounds(
        self,
        node: GenieSimNode,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get (min, max) bounds for a node."""
        pos = np.array(node.pose.position)
        half_size = np.array(node.size) / 2
        return pos - half_size, pos + half_size

    def _is_inside(
        self,
        inner_bounds: Tuple[np.ndarray, np.ndarray],
        outer_bounds: Tuple[np.ndarray, np.ndarray],
        margin: float,
    ) -> bool:
        """Check if inner bounds are inside outer bounds."""
        inner_min, inner_max = inner_bounds
        outer_min, outer_max = outer_bounds

        return (
            np.all(inner_min >= outer_min - margin) and
            np.all(inner_max <= outer_max + margin)
        )

    def _apply_physics_validation(
        self,
        edge: GenieSimEdge,
        source_node: GenieSimNode,
        target_node: GenieSimNode,
    ) -> Optional[GenieSimEdge]:
        if edge.relation not in {"on", "in"}:
            return edge
        if not self.enable_physics_validation:
            return self._mark_heuristic(edge)

        validation = self._validate_relation_with_physics(edge, source_node, target_node)
        if validation is None:
            return self._mark_heuristic(edge)
        if not validation:
            return None
        edge.metadata["inference_method"] = "physics"
        return edge

    def _mark_heuristic(self, edge: GenieSimEdge) -> GenieSimEdge:
        edge.confidence *= self.heuristic_confidence_scale
        edge.metadata["inference_method"] = "heuristic"
        return edge

    def _validate_relation_with_physics(
        self,
        edge: GenieSimEdge,
        source_node: GenieSimNode,
        target_node: GenieSimNode,
    ) -> Optional[bool]:
        if edge.relation == "on":
            return self._validate_on_with_physics(source_node, target_node)
        if edge.relation == "in":
            return self._validate_in_with_physics(source_node, target_node)
        return None

    def _validate_on_with_physics(
        self,
        source_node: GenieSimNode,
        target_node: GenieSimNode,
    ) -> Optional[bool]:
        entries_source = self._extract_contact_entries(source_node)
        entries_target = self._extract_contact_entries(target_node)
        if not entries_source and not entries_target:
            return None

        depth_source = self._contact_depth(entries_source, target_node.asset_id)
        depth_target = self._contact_depth(entries_target, source_node.asset_id)

        if depth_source is None and depth_target is None:
            return False

        depth_values = [
            depth for depth in [depth_source, depth_target]
            if depth is not None
        ]
        if not depth_values:
            return False
        max_depth = max(depth_values)
        return max_depth >= self.physics_contact_depth_threshold

    def _validate_in_with_physics(
        self,
        source_node: GenieSimNode,
        target_node: GenieSimNode,
    ) -> Optional[bool]:
        contained_entries = self._extract_containment_entries(
            source_node,
            include_contained_in=True,
            include_contains=False,
        )
        contains_entries = self._extract_containment_entries(
            target_node,
            include_contained_in=False,
            include_contains=True,
        )
        if not contained_entries and not contains_entries:
            return None

        if self._match_containment_entry(
            contained_entries,
            target_node.asset_id,
            self.physics_containment_ratio_threshold,
        ):
            return True
        if self._match_containment_entry(
            contains_entries,
            source_node.asset_id,
            self.physics_containment_ratio_threshold,
        ):
            return True
        return False

    def _extract_contact_entries(self, node: GenieSimNode) -> List[Any]:
        metadata = node.bp_metadata or {}
        entries: List[Any] = []
        entries += self._normalize_list(metadata.get("contacts"))
        entries += self._normalize_list(metadata.get("collisions"))

        physics = metadata.get("physics", {})
        if isinstance(physics, dict):
            entries += self._normalize_list(physics.get("contacts"))
            entries += self._normalize_list(physics.get("collisions"))
        return entries

    def _contact_depth(self, entries: List[Any], other_id: str) -> Optional[float]:
        raw_other_id = _strip_asset_namespace(other_id)
        for entry in entries:
            if isinstance(entry, str):
                if entry == other_id or entry == raw_other_id:
                    return self.physics_contact_depth_threshold
                continue
            if not isinstance(entry, dict):
                continue
            other = (
                entry.get("other_id")
                or entry.get("object_id")
                or entry.get("id")
                or entry.get("target_id")
            )
            if other != other_id and other != raw_other_id:
                continue
            depth = (
                entry.get("depth")
                or entry.get("penetration_depth")
                or entry.get("contact_depth")
            )
            if depth is None:
                return self.physics_contact_depth_threshold
            try:
                return float(depth)
            except (TypeError, ValueError):
                return self.physics_contact_depth_threshold
        return None

    def _extract_containment_entries(
        self,
        node: GenieSimNode,
        include_contained_in: bool,
        include_contains: bool,
    ) -> List[Any]:
        metadata = node.bp_metadata or {}
        entries: List[Any] = []

        if include_contained_in:
            entries += self._normalize_list(metadata.get("contained_in"))
        if include_contains:
            entries += self._normalize_list(metadata.get("contains"))

        entries += self._normalize_list(metadata.get("containment"))
        physics = metadata.get("physics", {})
        if isinstance(physics, dict):
            entries += self._normalize_list(physics.get("containment"))
        return entries

    def _match_containment_entry(
        self,
        entries: List[Any],
        other_id: str,
        ratio_threshold: float,
    ) -> bool:
        raw_other_id = _strip_asset_namespace(other_id)
        for entry in entries:
            if isinstance(entry, str):
                if entry == other_id or entry == raw_other_id:
                    return True
                continue
            if not isinstance(entry, dict):
                continue
            other = (
                entry.get("container_id")
                or entry.get("contained_id")
                or entry.get("other_id")
                or entry.get("object_id")
                or entry.get("id")
            )
            if other != other_id and other != raw_other_id:
                continue
            ratio = (
                entry.get("ratio")
                or entry.get("containment_ratio")
                or entry.get("overlap_ratio")
            )
            if ratio is None:
                return True
            try:
                return float(ratio) >= ratio_threshold
            except (TypeError, ValueError):
                return True
        return False

    def _normalize_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]


# =============================================================================
# Scene Graph Converter
# =============================================================================


class SceneGraphConverter:
    """
    Converts BlueprintPipeline scene manifests to Genie Sim scene graphs.

    Usage:
        converter = SceneGraphConverter()
        scene_graph = converter.convert(manifest_dict)
        scene_graph.save(Path("output/scene_graph.json"))
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.runtime_config = _load_scene_graph_runtime_config()
        self.production_mode = resolve_production_mode()
        self._ensure_validation_availability()
        self.relation_inferencer = RelationInferencer(verbose=verbose)
        self._llm_client = None  # lazy-init for Gemini dimension estimation
        self._llm_init_attempted = False
        self._dimension_cache: Dict[str, List[float]] = {}  # category -> [w, d, h]

    def _get_llm_client(self):
        """Lazy-init LLM client for Gemini dimension estimation."""
        if self._llm_init_attempted:
            return self._llm_client
        self._llm_init_attempted = True
        try:
            from tools.llm_client.client import create_llm_client
            self._llm_client = create_llm_client()
            logger.info("[SCENE_GRAPH] LLM client initialized for dimension estimation")
        except Exception as exc:
            logger.warning(
                "[SCENE_GRAPH] LLM client unavailable for dimension estimation: %s", exc
            )
        return self._llm_client

    def _estimate_dimensions_gemini(
        self, category: str, semantic: str
    ) -> Optional[List[float]]:
        """Estimate object dimensions [w, d, h] in meters via Gemini."""
        if category in self._dimension_cache:
            return self._dimension_cache[category]
        client = self._get_llm_client()
        if client is None:
            return None
        prompt = (
            f"Estimate the typical real-world dimensions in meters for: {semantic}.\n"
            f"Category: {category}.\n"
            f"Respond with ONLY a JSON object: "
            f'{{"width": <float>, "depth": <float>, "height": <float>}}\n'
            f"Use meters. Be accurate to real-world objects."
        )
        try:
            response = client.generate(prompt=prompt, json_output=True, disable_tools=True)
            text = response.text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                dims = [
                    max(float(data.get("width", 0.1)), 0.01),
                    max(float(data.get("depth", 0.1)), 0.01),
                    max(float(data.get("height", 0.1)), 0.01),
                ]
                self._dimension_cache[category] = dims
                logger.info(
                    "[SCENE_GRAPH] Gemini estimated dimensions for %s: %s",
                    category, dims,
                )
                return dims
        except Exception as exc:
            logger.warning(
                "[SCENE_GRAPH] Gemini dimension estimation failed for %s: %s",
                category, exc,
            )
        return None

    def _ensure_validation_availability(self) -> None:
        if HAVE_VALIDATION_TOOLS:
            return
        message = (
            "Input validation tools unavailable for scene graph conversion. "
            "Install tools.validation dependencies or enable validation."
        )
        if self.production_mode:
            raise RuntimeError(f"{message} Production mode requires validation.")
        if not self.runtime_config.allow_unvalidated_input:
            raise RuntimeError(f"{message} Unvalidated input fallback disabled.")

    def log(self, msg: str, level: str = "INFO", extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.verbose:
            return
        level_name = level.upper()
        if level_name in {"WARN", "WARNING"}:
            logger.warning("[SCENE-GRAPH-CONVERTER] %s", msg, extra=extra)
        elif level_name == "ERROR":
            logger.error("[SCENE-GRAPH-CONVERTER] %s", msg, extra=extra)
        else:
            logger.info("[SCENE-GRAPH-CONVERTER] %s", msg, extra=extra)

    def convert(
        self,
        manifest: Dict[str, Any],
        usd_base_path: Optional[str] = None,
    ) -> GenieSimSceneGraph:
        """
        Convert BlueprintPipeline manifest to Genie Sim scene graph.

        Args:
            manifest: BlueprintPipeline scene_manifest.json as dict
            usd_base_path: Base path for USD files (if relative paths needed)

        Returns:
            GenieSimSceneGraph ready for Genie Sim
        """
        start_time = time.monotonic()
        self.log("Starting non-streaming manifest conversion")

        scene_id = _resolve_scene_id_from_manifest(manifest)
        scene_config = manifest.get("scene", {})
        objects = manifest.get("objects", [])

        # Extract coordinate system
        coord_frame = scene_config.get("coordinate_frame", "y_up")
        meters_per_unit = scene_config.get("meters_per_unit", 1.0)

        total_objects = len(objects)
        self.log(
            f"Scene: {scene_id}, {total_objects} objects, coord={coord_frame}",
            extra={"scene_id": scene_id},
        )

        # Convert nodes
        nodes = []
        progress_every = None
        if self.verbose and total_objects >= 200:
            progress_every = max(50, total_objects // 10)

        for index, obj in enumerate(objects, start=1):
            node = self._convert_object_to_node(obj, usd_base_path, scene_id=scene_id)
            if node:
                nodes.append(node)
            if progress_every and index % progress_every == 0:
                self.log(
                    f"Converted {index}/{total_objects} objects to nodes",
                    extra={"scene_id": scene_id},
                )

        self.log(f"Converted {len(nodes)} nodes", extra={"scene_id": scene_id})

        # Extract explicit edges from relationships
        explicit_edges = self._extract_explicit_edges(objects, scene_id=scene_id)
        self.log(
            f"Found {len(explicit_edges)} explicit relationships",
            extra={"scene_id": scene_id},
        )

        # Infer additional edges
        inferred_edges = self.relation_inferencer.infer_relations(
            nodes,
            scene_id=scene_id,
            progress_callback=self._relation_inference_progress,
        )

        # Merge edges (explicit edges take priority)
        edges = self._merge_edges(explicit_edges, inferred_edges)
        self.log(f"Total edges: {len(edges)}", extra={"scene_id": scene_id})

        # Create scene graph
        scene_graph = GenieSimSceneGraph(
            scene_id=scene_id,
            coordinate_system=coord_frame,
            meters_per_unit=meters_per_unit,
            nodes=nodes,
            edges=edges,
            metadata={
                "source": "blueprintpipeline",
                "environment_type": scene_config.get("environment_type"),
                "room_bounds": scene_config.get("room", {}).get("bounds"),
            },
        )

        duration = time.monotonic() - start_time
        self.log(
            f"Finished non-streaming conversion in {duration:.2f}s",
            extra={"scene_id": scene_id},
        )
        return scene_graph

    def _relation_inference_progress(self, processed_pairs: int, elapsed: float) -> None:
        """Progress callback for relation inference."""
        if self.verbose:
            self.log(
                f"Relation inference processed {processed_pairs} pairs in {elapsed:.1f}s",
                extra={"processed_pairs": processed_pairs},
            )

    def _convert_object_to_node(
        self,
        obj: Dict[str, Any],
        usd_base_path: Optional[str],
        scene_id: Optional[str] = None,
    ) -> Optional[GenieSimNode]:
        """Convert a BlueprintPipeline object to a Genie Sim node."""
        try:
            obj_id = str(obj.get("id", ""))
            if not obj_id:
                return None
            production_mode = resolve_production_mode()

            # Skip background/shell objects
            sim_role = obj.get("sim_role", "unknown")
            if sim_role in ["background", "scene_shell"]:
                return None

            # GAP-SEC-002 FIX: Validate and sanitize all user inputs to prevent XSS/injection
            if HAVE_VALIDATION_TOOLS:
                try:
                    # Validate object ID
                    obj_id = validate_object_id(obj_id)

                    # Validate and sanitize category
                    category_raw = obj.get("category", "object")
                    category = validate_category(
                        category_raw,
                        allowed_categories=ALLOWED_ASSET_CATEGORIES,
                        strict=False,
                    )

                    # Validate and sanitize description
                    description_raw = obj.get("description", "")
                    description = validate_description(description_raw) if description_raw else ""

                    # Validate and sanitize name
                    name_raw = obj.get("name", obj_id)
                    name = sanitize_string(name_raw, max_length=128)

                except ValidationError as e:
                    logger.warning(
                        "Object validation failed: %s. Using defaults.",
                        e,
                        extra={"scene_id": scene_id, "object_id": obj_id},
                    )
                    category = "object"
                    description = ""
                    name = obj_id[:128]  # Truncate if needed
            else:
                obj_id_raw = obj_id
                obj_id = _sanitize_fallback_object_id(obj_id_raw)
                category_raw = obj.get("category", "object")
                category, category_fallback = _sanitize_fallback_category(category_raw)
                description_raw = obj.get("description", "")
                description = _sanitize_fallback_text(description_raw, max_length=1024)
                name_raw = obj.get("name", obj_id)
                name = _sanitize_fallback_text(name_raw, max_length=128)
                logger.warning(
                    "Validation tools unavailable; using fallback sanitization.",
                    extra={"scene_id": scene_id, "object_id": obj_id},
                )
                if obj_id != obj_id_raw or category_fallback or name != str(name_raw) or description != str(description_raw):
                    logger.info(
                        "Fallback sanitization applied to object fields.",
                        extra={"scene_id": scene_id, "object_id": obj_id},
                    )

            # Build semantic description
            semantic = f"{category}: {name}"
            if description:
                semantic = f"{category}: {description}"

            # Extract and validate size
            dimensions = obj.get("dimensions_est", {})
            if HAVE_VALIDATION_TOOLS and isinstance(dimensions, dict):
                try:
                    # Validate dimensions are positive numbers
                    validated_dims = validate_dimensions(dimensions)
                    size = [
                        validated_dims.get("width", 0.1),
                        validated_dims.get("depth", 0.1),
                        validated_dims.get("height", 0.1),
                    ]
                except ValidationError as e:
                    logger.warning(
                        "Object dimensions invalid: %s. Using defaults.",
                        e,
                        extra={"scene_id": scene_id, "object_id": obj_id},
                    )
                    size = [0.1, 0.1, 0.1]
            elif isinstance(dimensions, dict):
                size = [
                    dimensions.get("width", 0.1),
                    dimensions.get("depth", 0.1),
                    dimensions.get("height", 0.1),
                ]
            elif isinstance(dimensions, list):
                size = dimensions[:3] if len(dimensions) >= 3 else [0.1, 0.1, 0.1]
            else:
                size = [0.1, 0.1, 0.1]

            # Track dimension provenance
            _dim_source = obj.get("dimensions_source", "")

            # If size is still the default placeholder, try Gemini estimation
            _is_placeholder = all(abs(s - 0.1) < 1e-4 for s in size)
            if _is_placeholder and not _dim_source:
                gemini_dims = self._estimate_dimensions_gemini(category, semantic)
                if gemini_dims is not None:
                    size = gemini_dims
                    _dim_source = "gemini_estimated"
                else:
                    _dim_source = "default_placeholder"
                    logger.warning(
                        "[SCENE_GRAPH] Object %s using placeholder dimensions [0.1, 0.1, 0.1]; "
                        "run simready-job or provide dimensions_est in scene_manifest.",
                        obj_id,
                    )

            # Extract pose
            transform = obj.get("transform", {})
            position = transform.get("position", {})
            pos = [
                position.get("x", 0.0),
                position.get("y", 0.0),
                position.get("z", 0.0),
            ]

            # Get orientation (quaternion preferred)
            if "rotation_quaternion" in transform and transform["rotation_quaternion"]:
                rot_q = transform["rotation_quaternion"]
                orientation = [
                    rot_q.get("w", 1.0),
                    rot_q.get("x", 0.0),
                    rot_q.get("y", 0.0),
                    rot_q.get("z", 0.0),
                ]
            elif "rotation_euler" in transform and transform["rotation_euler"]:
                rot_e = transform["rotation_euler"]
                orientation = self._euler_to_quaternion(
                    rot_e.get("roll", 0.0),
                    rot_e.get("pitch", 0.0),
                    rot_e.get("yaw", 0.0),
                )
            else:
                orientation = [1.0, 0.0, 0.0, 0.0]

            if HAVE_VALIDATION_TOOLS:
                try:
                    orientation = validate_quaternion(
                        orientation,
                        field_name=f"{obj_id}.rotation_quaternion",
                        auto_normalize=not production_mode,
                    )
                except ValidationError as e:
                    logger.warning(
                        "Object quaternion invalid: %s. Using identity.",
                        e,
                        extra={"scene_id": scene_id, "object_id": obj_id},
                    )
                    orientation = [1.0, 0.0, 0.0, 0.0]

            pose = Pose(position=pos, orientation=orientation)

            # Get task tags from sim_role and affordances
            task_tags = self._get_task_tags(obj)

            # Get USD path
            asset = obj.get("asset", {})
            usd_path = asset.get("path", "")
            if usd_base_path and usd_path and not usd_path.startswith("/"):
                if usd_path.startswith("assets/"):
                    # assets/ is a sibling of usd/, not inside it — go up
                    # one level from usd_base_path before joining.
                    usd_path = f"../{usd_path}"
                else:
                    usd_path = f"{usd_base_path}/{usd_path}"

            # Extract physics properties
            physics = obj.get("physics", {})
            physics_hints = obj.get("physics_hints", {})
            properties = {
                "mass": physics.get("mass", 0.5),
                "friction": physics.get("friction", physics_hints.get("roughness", 0.5)),
                "restitution": physics.get("restitution", 0.1),
            }

            # Preserve BlueprintPipeline metadata
            bp_metadata = {
                "sim_role": sim_role,
                "category": category,
                "affordances": obj.get("semantics", {}).get("affordances", []),
                "articulation": obj.get("articulation", {}),
                "placement_region": obj.get("placement_region"),
                "physics": obj.get("physics", {}),
                "physics_hints": obj.get("physics_hints", {}),
                "contacts": obj.get("contacts", obj.get("physics", {}).get("contacts", [])),
                "collisions": obj.get("collisions", obj.get("physics", {}).get("collisions", [])),
                "containment": obj.get("containment", obj.get("physics", {}).get("containment", [])),
                "contained_in": obj.get("contained_in"),
                "contains": obj.get("contains"),
                "dimensions_source": _dim_source or obj.get("dimensions_source", ""),
            }

            asset_id = _namespaced_asset_id(scene_id, obj_id)

            return GenieSimNode(
                asset_id=asset_id,
                semantic=semantic,
                size=size,
                pose=pose,
                task_tag=task_tags,
                usd_path=usd_path,
                properties=properties,
                bp_metadata=bp_metadata,
            )

        except Exception as e:
            self.log(f"Failed to convert object {obj.get('id', 'unknown')}: {e}", "WARNING")
            return None

    def _get_task_tags(self, obj: Dict[str, Any]) -> List[str]:
        """Get task tags from object's sim_role and affordances."""
        sim_role = obj.get("sim_role", "unknown")
        semantics = obj.get("semantics", {})
        affordances = semantics.get("affordances", [])

        # Handle affordances that are strings or dicts
        affordance_names = []
        for aff in affordances:
            if isinstance(aff, str):
                affordance_names.append(aff)
            elif isinstance(aff, dict):
                affordance_names.append(aff.get("type", ""))

        # Collect task tags
        task_tags = set()

        for aff_name in affordance_names:
            key = (sim_role, aff_name)
            if key in TASK_TAG_MAPPING:
                task_tags.update(TASK_TAG_MAPPING[key])

        # Add default tags if no affordance matches
        if not task_tags:
            task_tags.update(DEFAULT_TASK_TAGS.get(sim_role, []))

        return list(task_tags)

    def _euler_to_quaternion(
        self,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> List[float]:
        """Convert Euler angles (radians) to quaternion [w, x, y, z]."""
        cr = math.cos(roll / 2)
        sr = math.sin(roll / 2)
        cp = math.cos(pitch / 2)
        sp = math.sin(pitch / 2)
        cy = math.cos(yaw / 2)
        sy = math.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return [w, x, y, z]

    def _extract_explicit_edges(
        self,
        objects: List[Dict[str, Any]],
        scene_id: Optional[str] = None,
    ) -> List[GenieSimEdge]:
        """Extract explicit relationships from object definitions."""
        edges = []

        for obj in objects:
            obj_id = str(obj.get("id", ""))
            relationships = obj.get("relationships", [])

            for rel in relationships:
                rel_type = rel.get("type", "")
                subject_id = rel.get("subject_id", obj_id)
                object_id = rel.get("object_id", "")

                if not object_id:
                    continue

                # Map to Genie Sim relation type
                geniesim_relation = RELATION_TYPE_MAPPING.get(rel_type.lower())
                if not geniesim_relation:
                    continue

                edges.append(GenieSimEdge(
                    source=_namespaced_asset_id(scene_id, str(subject_id)),
                    target=_namespaced_asset_id(scene_id, str(object_id)),
                    relation=geniesim_relation,
                    confidence=1.0,  # Explicit = high confidence
                ))

        return edges

    def _merge_edges(
        self,
        explicit: List[GenieSimEdge],
        inferred: List[GenieSimEdge],
    ) -> List[GenieSimEdge]:
        """Merge explicit and inferred edges, preferring explicit."""
        # Create set of explicit edge keys
        explicit_keys = set()
        for e in explicit:
            explicit_keys.add((e.source, e.target, e.relation))
            explicit_keys.add((e.target, e.source, e.relation))  # Bidirectional check

        # Add explicit edges
        merged = list(explicit)

        # Add inferred edges that don't conflict
        for e in inferred:
            key = (e.source, e.target, e.relation)
            reverse_key = (e.target, e.source, e.relation)
            if key not in explicit_keys and reverse_key not in explicit_keys:
                merged.append(e)

        return merged

    def _infer_relations(
        self,
        nodes: List[GenieSimNode],
        scene_id: Optional[str] = None,
    ) -> List[GenieSimEdge]:
        """Infer spatial relations between nodes."""
        return self.relation_inferencer.infer_relations(
            nodes,
            scene_id=scene_id,
            progress_callback=self._relation_inference_progress,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def convert_manifest_to_scene_graph(
    manifest_path: Path,
    output_path: Optional[Path] = None,
    verbose: bool = True,
    use_streaming: Optional[bool] = None,
) -> GenieSimSceneGraph:
    """
    Convenience function to convert a manifest file to scene graph.

    GAP-PERF-001 FIX: Use streaming JSON parser for large manifests to prevent OOM.
    Streaming uses StreamingManifestParser metadata reads; large files require ijson.
    Production raises if ijson is missing to avoid silent full-file loads.

    Args:
        manifest_path: Path to scene_manifest.json
        output_path: Optional path to save scene_graph.json
        verbose: Print progress
        use_streaming: Force streaming mode (auto-detect if None based on file size)

    Returns:
        GenieSimSceneGraph
    """
    runtime_config = _load_scene_graph_runtime_config()

    file_size_mb = manifest_path.stat().st_size / (1024 * 1024)
    production_mode = resolve_production_mode()
    streaming_threshold_mb = 1.0

    if production_mode:
        use_streaming = True

    # Auto-detect if streaming is needed based on file size
    if use_streaming is None:
        use_streaming = file_size_mb > streaming_threshold_mb  # Use streaming for files > 1MB

    ijson_available = importlib.util.find_spec("ijson") is not None
    if production_mode and not ijson_available:
        raise RuntimeError(
            "Production mode requires ijson for streaming manifest parsing. "
            "Install ijson (pip install ijson) to avoid full-file JSON loads."
        )
    if use_streaming and not ijson_available:
        message = (
            "Streaming requested for manifest but ijson is not installed. "
            "Install ijson (pip install ijson) to avoid full-file JSON loads."
        )
        logger.error(message, extra={"manifest_path": str(manifest_path)})

    # GAP-PERF-001 FIX: Use streaming parser for large manifests
    if use_streaming and HAVE_STREAMING_PARSER:
        if verbose:
            logger.info(
                "[SCENE-GRAPH-CONVERTER] Using streaming parser for large manifest (%.1f MB)",
                file_size_mb,
                extra={"manifest_path": str(manifest_path)},
            )

        def _inflate_metadata(flat_metadata: Dict[str, Any], prefix: str) -> Dict[str, Any]:
            nested: Dict[str, Any] = {}
            for key, value in flat_metadata.items():
                if not key.startswith(prefix):
                    continue
                parts = key[len(prefix):].split(".")
                cursor = nested
                for part in parts[:-1]:
                    cursor = cursor.setdefault(part, {})
                cursor[parts[-1]] = value
            return nested

        # Create converter
        converter = SceneGraphConverter(verbose=verbose)
        parser = StreamingManifestParser(str(manifest_path))
        manifest_metadata = parser.get_metadata()

        metadata_payload = manifest_metadata.get("metadata")
        if metadata_payload is None:
            metadata_payload = _inflate_metadata(manifest_metadata, "metadata.")

        missing_keys = [
            key for key in ("scene_id", "coordinate_system", "meters_per_unit")
            if key not in manifest_metadata
        ]
        if metadata_payload is None or metadata_payload == {}:
            if "metadata" not in manifest_metadata:
                missing_keys.append("metadata")
            metadata_payload = metadata_payload or {}

        if missing_keys:
            logger.warning(
                "Manifest metadata missing keys in streaming mode: %s",
                ", ".join(missing_keys),
                extra={"manifest_path": str(manifest_path)},
            )

        # Start building scene graph with metadata
        scene_graph = GenieSimSceneGraph(
            scene_id=manifest_metadata.get("scene_id", "unknown"),
            coordinate_system=manifest_metadata.get("coordinate_system", "y_up"),
            meters_per_unit=manifest_metadata.get("meters_per_unit", 1.0),
            nodes=[],
            edges=[],
            metadata=metadata_payload,
        )

        # Stream process objects in batches
        usd_base_path = manifest_metadata.get("usd_file")

        batch_size = runtime_config.streaming_batch_size

        def stream_progress(processed_objects: int, elapsed: float) -> None:
            if verbose:
                logger.info(
                    "[SCENE-GRAPH-CONVERTER] Streamed %d objects in %.1fs",
                    processed_objects,
                    elapsed,
                    extra={"manifest_path": str(manifest_path)},
                )

        for batch in parser.stream_objects(
            batch_size=batch_size,
            progress_callback=stream_progress,
            progress_interval_s=5.0,
        ):

            # Convert batch of objects to nodes
            for obj in batch:
                node = converter._convert_object_to_node(
                    obj,
                    usd_base_path,
                    scene_id=scene_graph.scene_id,
                )
                if node:
                    scene_graph.nodes.append(node)

        # Infer relations from node positions
        if verbose:
            logger.info(
                "[SCENE-GRAPH-CONVERTER] Inferring spatial relations for %d nodes...",
                len(scene_graph.nodes),
                extra={"scene_id": scene_graph.scene_id},
            )
        scene_graph.edges = converter._infer_relations(
            scene_graph.nodes,
            scene_id=scene_graph.scene_id,
        )

        if verbose:
            logger.info(
                "[SCENE-GRAPH-CONVERTER] Streaming conversion complete: %d nodes, %d edges",
                len(scene_graph.nodes),
                len(scene_graph.edges),
                extra={"scene_id": scene_graph.scene_id},
            )

    else:
        # Standard mode for small manifests
        with open(manifest_path) as f:
            manifest = json.load(f)
        converter = SceneGraphConverter(verbose=verbose)
        scene_graph = converter.convert(manifest)

    if output_path:
        scene_graph.save(output_path)

    return scene_graph
