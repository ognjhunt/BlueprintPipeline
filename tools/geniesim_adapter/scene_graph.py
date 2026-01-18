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
import json
import logging
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "confidence": self.confidence,
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


_SCENE_GRAPH_CONFIG: Optional[SceneGraphRuntimeConfig] = None


def _parse_env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%s; using default %.3f", key, raw, default)
        return default


def _parse_env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%s; using default %d", key, raw, default)
        return default


def _load_scene_graph_runtime_config() -> SceneGraphRuntimeConfig:
    global _SCENE_GRAPH_CONFIG
    if _SCENE_GRAPH_CONFIG is not None:
        return _SCENE_GRAPH_CONFIG

    if HAVE_PIPELINE_CONFIG:
        pipeline_config = load_pipeline_config()
        relation_config = pipeline_config.scene_graph.relation_inference
        streaming_config = pipeline_config.scene_graph.streaming
        vertical_default = relation_config.vertical_proximity_threshold
        horizontal_default = relation_config.horizontal_proximity_threshold
        alignment_default = relation_config.alignment_angle_threshold
        batch_default = streaming_config.batch_size
    else:
        vertical_default = 0.05
        horizontal_default = 0.15
        alignment_default = 5.0
        batch_default = 100

    config = SceneGraphRuntimeConfig(
        vertical_proximity_threshold=_parse_env_float(
            "BP_SCENE_GRAPH_VERTICAL_PROXIMITY_THRESHOLD",
            vertical_default,
        ),
        horizontal_proximity_threshold=_parse_env_float(
            "BP_SCENE_GRAPH_HORIZONTAL_PROXIMITY_THRESHOLD",
            horizontal_default,
        ),
        alignment_angle_threshold=_parse_env_float(
            "BP_SCENE_GRAPH_ALIGNMENT_ANGLE_THRESHOLD",
            alignment_default,
        ),
        streaming_batch_size=_parse_env_int(
            "BP_SCENE_GRAPH_STREAMING_BATCH_SIZE",
            batch_default,
        ),
    )

    logger.info(
        "Scene graph config: vertical_proximity_threshold=%.3f, horizontal_proximity_threshold=%.3f, "
        "alignment_angle_threshold=%.3f, streaming_batch_size=%d",
        config.vertical_proximity_threshold,
        config.horizontal_proximity_threshold,
        config.alignment_angle_threshold,
        config.streaming_batch_size,
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
        self._cache: "OrderedDict[Tuple[str, str], List[GenieSimEdge]]" = OrderedDict()
        self.last_cache_hit = False
        self._cache_hits = 0
        self._cache_misses = 0

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[RELATION-INFERENCER] {msg}")

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
                f"({len(nodes)} nodes)."
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
                    f"pairs in {elapsed:.1f}s"
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
                    return GenieSimEdge(
                        source=node_a.asset_id,
                        target=node_b.asset_id,
                        relation="on",
                        confidence=0.8,
                    )

        # Check if B is above A
        elif vertical_diff < 0:
            b_bottom = pos_b[2] - size_b[2] / 2
            a_top = pos_a[2] + size_a[2] / 2

            if abs(b_bottom - a_top) < self.vertical_proximity_threshold:
                if self._has_horizontal_overlap(node_a, node_b):
                    return GenieSimEdge(
                        source=node_b.asset_id,
                        target=node_a.asset_id,
                        relation="on",
                        confidence=0.8,
                    )

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
            return GenieSimEdge(
                source=node_a.asset_id,
                target=node_b.asset_id,
                relation="in",
                confidence=0.9,
            )

        # Check if B is inside A
        if self._is_inside(bounds_b, bounds_a, self.CONTAINMENT_MARGIN):
            return GenieSimEdge(
                source=node_b.asset_id,
                target=node_a.asset_id,
                relation="in",
                confidence=0.9,
            )

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
        self.relation_inferencer = RelationInferencer(verbose=verbose)

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[SCENE-GRAPH-CONVERTER] [{level}] {msg}")

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

        scene_id = manifest.get("scene_id", "unknown")
        scene_config = manifest.get("scene", {})
        objects = manifest.get("objects", [])

        # Extract coordinate system
        coord_frame = scene_config.get("coordinate_frame", "y_up")
        meters_per_unit = scene_config.get("meters_per_unit", 1.0)

        total_objects = len(objects)
        self.log(f"Scene: {scene_id}, {total_objects} objects, coord={coord_frame}")

        # Convert nodes
        nodes = []
        progress_every = None
        if self.verbose and total_objects >= 200:
            progress_every = max(50, total_objects // 10)

        for index, obj in enumerate(objects, start=1):
            node = self._convert_object_to_node(obj, usd_base_path)
            if node:
                nodes.append(node)
            if progress_every and index % progress_every == 0:
                self.log(f"Converted {index}/{total_objects} objects to nodes")

        self.log(f"Converted {len(nodes)} nodes")

        # Extract explicit edges from relationships
        explicit_edges = self._extract_explicit_edges(objects)
        self.log(f"Found {len(explicit_edges)} explicit relationships")

        # Infer additional edges
        inferred_edges = self.relation_inferencer.infer_relations(
            nodes,
            scene_id=scene_id,
            progress_callback=self._relation_inference_progress,
        )

        # Merge edges (explicit edges take priority)
        edges = self._merge_edges(explicit_edges, inferred_edges)
        self.log(f"Total edges: {len(edges)}")

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
        self.log(f"Finished non-streaming conversion in {duration:.2f}s")
        return scene_graph

    def _relation_inference_progress(self, processed_pairs: int, elapsed: float) -> None:
        """Progress callback for relation inference."""
        if self.verbose:
            self.log(f"Relation inference processed {processed_pairs} pairs in {elapsed:.1f}s")

    def _convert_object_to_node(
        self,
        obj: Dict[str, Any],
        usd_base_path: Optional[str],
    ) -> Optional[GenieSimNode]:
        """Convert a BlueprintPipeline object to a Genie Sim node."""
        try:
            obj_id = str(obj.get("id", ""))
            if not obj_id:
                return None

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
                    logger.warning(f"Object {obj_id} validation failed: {e}. Using defaults.")
                    category = "object"
                    description = ""
                    name = obj_id[:128]  # Truncate if needed
            else:
                # No validation available - use raw values (insecure)
                category = obj.get("category", "object")
                description = obj.get("description", "")
                name = obj.get("name", obj_id)

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
                    logger.warning(f"Object {obj_id} dimensions invalid: {e}. Using defaults.")
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
                    )
                except ValidationError as e:
                    logger.warning("Object %s quaternion invalid: %s. Using identity.", obj_id, e)
                    orientation = [1.0, 0.0, 0.0, 0.0]

            pose = Pose(position=pos, orientation=orientation)

            # Get task tags from sim_role and affordances
            task_tags = self._get_task_tags(obj)

            # Get USD path
            asset = obj.get("asset", {})
            usd_path = asset.get("path", "")
            if usd_base_path and usd_path and not usd_path.startswith("/"):
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
            }

            return GenieSimNode(
                asset_id=obj_id,
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
                    source=subject_id,
                    target=object_id,
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

    Args:
        manifest_path: Path to scene_manifest.json
        output_path: Optional path to save scene_graph.json
        verbose: Print progress
        use_streaming: Force streaming mode (auto-detect if None based on file size)

    Returns:
        GenieSimSceneGraph
    """
    runtime_config = _load_scene_graph_runtime_config()

    # Auto-detect if streaming is needed based on file size
    if use_streaming is None:
        file_size_mb = manifest_path.stat().st_size / (1024 * 1024)
        use_streaming = file_size_mb > 10  # Use streaming for files > 10MB

    # GAP-PERF-001 FIX: Use streaming parser for large manifests
    if use_streaming and HAVE_STREAMING_PARSER:
        if verbose:
            print(f"[SCENE-GRAPH-CONVERTER] Using streaming parser for large manifest ({manifest_path.stat().st_size / (1024*1024):.1f} MB)")

        # Load metadata without objects (streaming handles objects separately)
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Remove objects array (will be streamed)
        manifest.pop("objects", None)

        # Create converter
        converter = SceneGraphConverter(verbose=verbose)

        # Start building scene graph with metadata
        scene_graph = GenieSimSceneGraph(
            scene_id=manifest.get("scene_id", "unknown"),
            coordinate_system=manifest.get("coordinate_system", "y_up"),
            meters_per_unit=manifest.get("meters_per_unit", 1.0),
            nodes=[],
            edges=[],
            metadata=manifest.get("metadata", {}),
        )

        # Stream process objects in batches
        parser = StreamingManifestParser(str(manifest_path))
        usd_base_path = manifest.get("usd_file")

        batch_count = 0
        batch_size = runtime_config.streaming_batch_size
        for batch in parser.stream_objects(batch_size=batch_size):
            batch_count += 1
            if verbose and batch_count % 10 == 0:
                print(f"[SCENE-GRAPH-CONVERTER] Processed {batch_count * batch_size} objects...")
        def stream_progress(processed_objects: int, elapsed: float) -> None:
            if verbose:
                print(
                    "[SCENE-GRAPH-CONVERTER] Streamed "
                    f"{processed_objects} objects in {elapsed:.1f}s"
                )

        for batch in parser.stream_objects(
            batch_size=100,
            progress_callback=stream_progress,
            progress_interval_s=5.0,
        ):

            # Convert batch of objects to nodes
            for obj in batch:
                node = converter._convert_object_to_node(obj, usd_base_path)
                if node:
                    scene_graph.nodes.append(node)

        # Infer relations from node positions
        if verbose:
            print(f"[SCENE-GRAPH-CONVERTER] Inferring spatial relations for {len(scene_graph.nodes)} nodes...")
        scene_graph.edges = converter._infer_relations(
            scene_graph.nodes,
            scene_id=scene_graph.scene_id,
        )

        if verbose:
            print(f"[SCENE-GRAPH-CONVERTER] Streaming conversion complete: {len(scene_graph.nodes)} nodes, {len(scene_graph.edges)} edges")

    else:
        # Standard mode for small manifests
        with open(manifest_path) as f:
            manifest = json.load(f)
        converter = SceneGraphConverter(verbose=verbose)
        scene_graph = converter.convert(manifest)

    if output_path:
        scene_graph.save(output_path)

    return scene_graph
