#!/usr/bin/env python3
"""
Enhanced Articulation Detection with LLM Fallback.

This module provides intelligent detection of articulated joints in furniture and objects.
It uses a multi-stage approach:
1. Keyword matching (fast, high confidence)
2. LLM-based detection with vision (Gemini) (medium speed, medium-high confidence)
3. Geometric heuristics (fast, low-medium confidence)

Examples of articulated objects:
- Drawers (prismatic joint, linear motion)
- Doors (revolute joint, rotation)
- Cabinets (revolute, may have multiple doors)
- Laptop lids (revolute)
- Trash can lids (revolute)
- Tool chests (multiple prismatic drawers)
"""

import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import LLM client for fallback
try:
    from tools.llm_client import create_llm_client, LLMProvider
    HAVE_LLM_CLIENT = True
except ImportError:
    HAVE_LLM_CLIENT = False
    create_llm_client = None
    LLMProvider = None

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class ArticulationType(str, Enum):
    """Type of articulated joint."""

    PRISMATIC = "prismatic"  # Linear sliding motion (drawers)
    REVOLUTE = "revolute"    # Rotational motion (doors, lids)
    FIXED = "fixed"          # No articulation


@dataclass
class ArticulationResult:
    """Result of articulation detection."""

    object_id: str
    articulation_type: ArticulationType
    confidence: float  # 0-1
    detection_method: str  # "keyword", "llm", "geometric", "none"

    # Joint parameters (if detected)
    joint_axis: Optional[np.ndarray] = None  # [x, y, z] unit vector
    joint_range: Optional[Tuple[float, float]] = None  # (min, max) in meters or radians
    handle_position: Optional[np.ndarray] = None  # [x, y, z] relative to object center

    # Explanation
    reasoning: str = ""

    # Multi-joint support
    additional_joints: List["ArticulationResult"] = field(default_factory=list)

    @property
    def has_articulation(self) -> bool:
        """Check if object has articulation."""
        return self.articulation_type != ArticulationType.FIXED

    @property
    def is_high_confidence(self) -> bool:
        """Check if detection confidence is high."""
        return self.confidence >= 0.8


# =============================================================================
# Articulation Detector
# =============================================================================


class ArticulationDetector:
    """
    Enhanced articulation detector with LLM fallback.

    Detects articulated joints using multiple methods:
    1. Keyword-based matching (fast, high confidence for common objects)
    2. LLM vision analysis (Gemini) for unknown objects (requires image)
    3. Geometric heuristics based on shape and dimensions
    """

    # Keyword-based articulation database
    # Format: category -> (type, confidence, joint_axis, range)
    KNOWN_ARTICULATIONS = {
        # === Furniture with drawers (prismatic) ===
        "drawer": ("prismatic", 0.95, np.array([1, 0, 0]), (0.0, 0.4)),
        "drawers": ("prismatic", 0.95, np.array([1, 0, 0]), (0.0, 0.4)),
        "filing_cabinet": ("prismatic", 0.90, np.array([1, 0, 0]), (0.0, 0.5)),
        "dresser": ("prismatic", 0.85, np.array([1, 0, 0]), (0.0, 0.4)),
        "chest_of_drawers": ("prismatic", 0.90, np.array([1, 0, 0]), (0.0, 0.4)),
        "nightstand": ("prismatic", 0.80, np.array([1, 0, 0]), (0.0, 0.3)),
        "bedside_table": ("prismatic", 0.80, np.array([1, 0, 0]), (0.0, 0.3)),
        "desk_drawer": ("prismatic", 0.90, np.array([1, 0, 0]), (0.0, 0.5)),
        "toolbox": ("prismatic", 0.85, np.array([1, 0, 0]), (0.0, 0.3)),

        # === Furniture with doors (revolute) ===
        "door": ("revolute", 0.95, np.array([0, 0, 1]), (0.0, np.pi / 2)),
        "cabinet": ("revolute", 0.85, np.array([0, 0, 1]), (0.0, np.pi / 2)),
        "cabinet_door": ("revolute", 0.90, np.array([0, 0, 1]), (0.0, np.pi / 2)),
        "cupboard": ("revolute", 0.85, np.array([0, 0, 1]), (0.0, np.pi / 2)),
        "wardrobe": ("revolute", 0.85, np.array([0, 0, 1]), (0.0, np.pi)),
        "armoire": ("revolute", 0.80, np.array([0, 0, 1]), (0.0, np.pi)),
        "hutch": ("revolute", 0.80, np.array([0, 0, 1]), (0.0, np.pi / 2)),
        "china_cabinet": ("revolute", 0.80, np.array([0, 0, 1]), (0.0, np.pi / 2)),
        "safe": ("revolute", 0.90, np.array([0, 0, 1]), (0.0, np.pi)),

        # === Appliances (revolute) ===
        "refrigerator": ("revolute", 0.90, np.array([0, 0, 1]), (0.0, 2 * np.pi / 3)),
        "fridge": ("revolute", 0.90, np.array([0, 0, 1]), (0.0, 2 * np.pi / 3)),
        "oven": ("revolute", 0.85, np.array([1, 0, 0]), (0.0, np.pi / 2)),
        "microwave": ("revolute", 0.85, np.array([0, 0, 1]), (0.0, np.pi / 2)),
        "dishwasher": ("revolute", 0.85, np.array([1, 0, 0]), (0.0, np.pi / 3)),
        "washing_machine": ("revolute", 0.80, np.array([0, 0, 1]), (0.0, 2 * np.pi / 3)),
        "dryer": ("revolute", 0.80, np.array([0, 0, 1]), (0.0, 2 * np.pi / 3)),

        # === Containers with lids (revolute) ===
        "trash_can": ("revolute", 0.70, np.array([1, 0, 0]), (0.0, np.pi / 2)),
        "trash_bin": ("revolute", 0.70, np.array([1, 0, 0]), (0.0, np.pi / 2)),
        "bin": ("revolute", 0.60, np.array([1, 0, 0]), (0.0, np.pi / 2)),
        "chest": ("revolute", 0.75, np.array([1, 0, 0]), (0.0, np.pi / 2)),
        "trunk": ("revolute", 0.75, np.array([1, 0, 0]), (0.0, np.pi / 2)),
        "toy_chest": ("revolute", 0.80, np.array([1, 0, 0]), (0.0, np.pi / 2)),

        # === Electronics (revolute) ===
        "laptop": ("revolute", 0.85, np.array([1, 0, 0]), (0.0, 2 * np.pi / 3)),
        "laptop_lid": ("revolute", 0.90, np.array([1, 0, 0]), (0.0, 2 * np.pi / 3)),
    }

    def __init__(
        self,
        use_llm: bool = True,
        llm_provider: str = "gemini",
        verbose: bool = True,
    ):
        """
        Initialize articulation detector.

        Args:
            use_llm: Enable LLM-based detection fallback
            llm_provider: LLM provider to use ("gemini" or "openai")
            verbose: Print detection progress
        """
        self.use_llm = use_llm and HAVE_LLM_CLIENT
        self.verbose = verbose

        # Initialize LLM client if available
        self.llm_client = None
        if self.use_llm:
            try:
                if llm_provider == "gemini":
                    self.llm_client = create_llm_client(LLMProvider.GEMINI)
                elif llm_provider == "openai":
                    self.llm_client = create_llm_client(LLMProvider.OPENAI)

                if self.verbose:
                    logger.info(f"[ARTICULATION] LLM client initialized ({llm_provider})")
            except Exception as e:
                logger.warning(f"[ARTICULATION] LLM client initialization failed: {e}")
                self.llm_client = None

    def detect(
        self,
        obj: Dict[str, Any],
        image_path: Optional[Path] = None,
    ) -> ArticulationResult:
        """
        Detect articulation for an object.

        Args:
            obj: Object dictionary with id, category, dimensions, etc.
            image_path: Optional path to object image for vision-based detection

        Returns:
            ArticulationResult with type and confidence
        """
        obj_id = obj.get("id", obj.get("name", "unknown"))
        category = (obj.get("category", "") or "").lower().strip()

        if self.verbose:
            logger.info(f"[ARTICULATION] Detecting for {obj_id} (category: {category or 'unknown'})")

        # Stage 1: Keyword matching (fast, high confidence)
        result = self._keyword_match(obj_id, category)
        if result.is_high_confidence:
            if self.verbose:
                logger.info(f"  ✅ Keyword match: {result.articulation_type.value} (confidence: {result.confidence:.2f})")
            return result

        # Stage 2: LLM-based detection (requires image)
        if self.llm_client and image_path and image_path.exists():
            llm_result = self._llm_detect(obj, image_path)
            if llm_result.confidence > result.confidence:
                if self.verbose:
                    logger.info(f"  🤖 LLM detection: {llm_result.articulation_type.value} (confidence: {llm_result.confidence:.2f})")
                return llm_result

        # Stage 3: Geometric heuristics (fallback)
        if result.confidence < 0.5:
            geometric_result = self._geometric_heuristics(obj)
            if geometric_result.confidence > result.confidence:
                if self.verbose:
                    logger.info(f"  📐 Geometric heuristic: {geometric_result.articulation_type.value} (confidence: {geometric_result.confidence:.2f})")
                return geometric_result

        # Return best result
        if self.verbose:
            if result.confidence > 0:
                logger.info(f"  ℹ️  Best match: {result.articulation_type.value} (confidence: {result.confidence:.2f})")
            else:
                logger.info(f"  ❌ No articulation detected")
        return result

    def _keyword_match(self, obj_id: str, category: str) -> ArticulationResult:
        """Match object against known articulation keywords."""
        # Try exact category match
        if category in self.KNOWN_ARTICULATIONS:
            art_type, confidence, axis, joint_range = self.KNOWN_ARTICULATIONS[category]
            return ArticulationResult(
                object_id=obj_id,
                articulation_type=ArticulationType(art_type),
                confidence=confidence,
                detection_method="keyword",
                joint_axis=axis.copy() if isinstance(axis, np.ndarray) else np.array(axis),
                joint_range=joint_range,
                reasoning=f"Exact keyword match: '{category}'",
            )

        # Try substring match
        best_match = None
        best_confidence = 0.0

        for keyword, (art_type, base_confidence, axis, joint_range) in self.KNOWN_ARTICULATIONS.items():
            if keyword in category or keyword in obj_id.lower():
                # Reduce confidence slightly for substring match
                confidence = base_confidence * 0.9
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = ArticulationResult(
                        object_id=obj_id,
                        articulation_type=ArticulationType(art_type),
                        confidence=confidence,
                        detection_method="keyword",
                        joint_axis=axis.copy() if isinstance(axis, np.ndarray) else np.array(axis),
                        joint_range=joint_range,
                        reasoning=f"Substring match: '{keyword}' in '{category or obj_id}'",
                    )

        # No match found
        if best_match is None:
            return ArticulationResult(
                object_id=obj_id,
                articulation_type=ArticulationType.FIXED,
                confidence=0.0,
                detection_method="none",
                reasoning="No keyword match found",
            )

        return best_match

    def _llm_detect(
        self,
        obj: Dict[str, Any],
        image_path: Path,
    ) -> ArticulationResult:
        """Use LLM vision to detect articulation."""
        obj_id = obj.get("id", obj.get("name", "unknown"))
        category = obj.get("category", "object")

        # Construct prompt for LLM
        prompt = f"""Analyze this image of a {category} object (ID: {obj_id}).

Determine if this object has any articulated parts (moving parts like drawers, doors, lids, etc.).

If articulated, identify:
1. Type of joint: "prismatic" (linear sliding, like drawers) or "revolute" (rotation, like doors)
2. Axis of motion: which direction it moves/rotates
3. Approximate range of motion

Respond in JSON format:
{{
    "has_articulation": true/false,
    "articulation_type": "prismatic" or "revolute" or "fixed",
    "confidence": 0-1,
    "joint_axis": [x, y, z] or null,
    "reasoning": "brief explanation"
}}

Examples:
- Drawer: {{"has_articulation": true, "articulation_type": "prismatic", "confidence": 0.9, "joint_axis": [1, 0, 0]}}
- Door: {{"has_articulation": true, "articulation_type": "revolute", "confidence": 0.9, "joint_axis": [0, 0, 1]}}
- Box: {{"has_articulation": false, "articulation_type": "fixed", "confidence": 0.8, "joint_axis": null}}
"""

        try:
            # Call LLM with image
            response = self.llm_client.generate_with_image(
                prompt=prompt,
                image_path=str(image_path),
            )

            # Parse response
            import json
            result_data = json.loads(response.text)

            articulation_type = ArticulationType(result_data.get("articulation_type", "fixed"))
            confidence = float(result_data.get("confidence", 0.5))
            joint_axis = result_data.get("joint_axis")
            reasoning = result_data.get("reasoning", "LLM vision analysis")

            if joint_axis:
                joint_axis = np.array(joint_axis, dtype=float)
                # Normalize axis
                norm = np.linalg.norm(joint_axis)
                if norm > 0:
                    joint_axis = joint_axis / norm

            return ArticulationResult(
                object_id=obj_id,
                articulation_type=articulation_type,
                confidence=confidence,
                detection_method="llm",
                joint_axis=joint_axis,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.warning(f"[ARTICULATION] LLM detection failed: {e}")
            return ArticulationResult(
                object_id=obj_id,
                articulation_type=ArticulationType.FIXED,
                confidence=0.0,
                detection_method="llm_failed",
                reasoning=f"LLM detection failed: {str(e)}",
            )

    def _geometric_heuristics(self, obj: Dict[str, Any]) -> ArticulationResult:
        """Use geometric heuristics to guess articulation."""
        obj_id = obj.get("id", obj.get("name", "unknown"))
        dimensions = obj.get("dimensions", obj.get("size_m", [0.1, 0.1, 0.1]))

        if not dimensions or len(dimensions) < 3:
            return ArticulationResult(
                object_id=obj_id,
                articulation_type=ArticulationType.FIXED,
                confidence=0.0,
                detection_method="geometric",
                reasoning="Insufficient dimension information",
            )

        dims = np.array(dimensions, dtype=float)
        dims = np.sort(dims)  # [smallest, medium, largest]

        # Heuristic 1: Tall narrow objects might be doors/cabinets
        if dims[2] > 1.0 and dims[2] / dims[0] > 5.0:
            return ArticulationResult(
                object_id=obj_id,
                articulation_type=ArticulationType.REVOLUTE,
                confidence=0.4,
                detection_method="geometric",
                joint_axis=np.array([0, 0, 1]),  # Vertical axis
                reasoning="Tall narrow object (likely door/cabinet)",
            )

        # Heuristic 2: Wide shallow objects might have drawers
        if dims[1] / dims[0] > 3.0 and dims[2] / dims[0] < 2.0:
            return ArticulationResult(
                object_id=obj_id,
                articulation_type=ArticulationType.PRISMATIC,
                confidence=0.3,
                detection_method="geometric",
                joint_axis=np.array([1, 0, 0]),  # Horizontal pull
                reasoning="Wide shallow object (might have drawers)",
            )

        # Default: assume fixed
        return ArticulationResult(
            object_id=obj_id,
            articulation_type=ArticulationType.FIXED,
            confidence=0.5,
            detection_method="geometric",
            reasoning="No geometric indicators of articulation",
        )


# =============================================================================
# Batch Detection
# =============================================================================


def detect_scene_articulations(
    manifest: Dict[str, Any],
    image_dir: Optional[Path] = None,
    use_llm: bool = True,
    verbose: bool = True,
) -> Dict[str, ArticulationResult]:
    """
    Detect articulations for all objects in a scene.

    Args:
        manifest: Scene manifest with objects
        image_dir: Directory containing object images
        use_llm: Enable LLM-based detection
        verbose: Print detection progress

    Returns:
        Dict mapping object_id to ArticulationResult
    """
    detector = ArticulationDetector(use_llm=use_llm, verbose=verbose)
    results = {}

    objects = manifest.get("objects", [])
    for obj in objects:
        obj_id = obj.get("id", obj.get("name", "unknown"))

        # Find image if available
        image_path = None
        if image_dir:
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = image_dir / f"{obj_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

        result = detector.detect(obj, image_path)
        results[obj_id] = result

    # Print summary
    if verbose:
        total = len(results)
        articulated = sum(1 for r in results.values() if r.has_articulation)
        high_conf = sum(1 for r in results.values() if r.is_high_confidence and r.has_articulation)

        logger.info(f"\n[ARTICULATION] Detection summary:")
        logger.info(f"  Total objects: {total}")
        logger.info(f"  Articulated: {articulated} ({articulated / total * 100:.1f}%)")
        logger.info(f"  High confidence: {high_conf}")

    return results


if __name__ == "__main__":
    # Example usage
    detector = ArticulationDetector(use_llm=False, verbose=True)

    test_objects = [
        {"id": "drawer_1", "category": "drawer"},
        {"id": "cabinet_1", "category": "cabinet"},
        {"id": "table_1", "category": "table"},
        {"id": "dresser_1", "category": "dresser"},
        {"id": "laptop_1", "category": "laptop"},
    ]

    print("\nArticulation Detection Examples:\n")
    for obj in test_objects:
        result = detector.detect(obj)
        print(f"{obj['id']}: {result.articulation_type.value} (confidence: {result.confidence:.2f})")
        print(f"  {result.reasoning}\n")
