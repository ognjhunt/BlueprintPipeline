"""Deterministic articulation hint parsing from label/category text."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Sequence


_JOINT_PRIORITY = {
    "prismatic": 0,
    "revolute": 1,
    "continuous": 2,
}

_HINT_PATTERNS: List[Dict[str, Any]] = [
    {
        "part": "drawer",
        "joint_type": "prismatic",
        "confidence": 0.95,
        "patterns": (
            r"\bdrawer\b",
            r"\bdrawers\b",
            r"\bdesk\s+with\s+drawers\b",
            r"\bconsole\s+table\s+with\s+drawers\b",
            r"\bchest\s+of\s+drawers\b",
        ),
    },
    {
        "part": "door",
        "joint_type": "revolute",
        "confidence": 0.9,
        "patterns": (
            r"\bdoor\b",
            r"\bdoors\b",
            r"\bcabinet\s+door\b",
            r"\bcabinet\s+with\s+doors\b",
        ),
    },
    {
        "part": "lid",
        "joint_type": "revolute",
        "confidence": 0.85,
        "patterns": (
            r"\blid\b",
            r"\blids\b",
            r"\bhinged\s+lid\b",
        ),
    },
    {
        "part": "hinge",
        "joint_type": "revolute",
        "confidence": 0.8,
        "patterns": (
            r"\bhinge\b",
            r"\bhinges\b",
            r"\bhinged\b",
        ),
    },
    {
        "part": "knob",
        "joint_type": "continuous",
        "confidence": 0.7,
        "patterns": (
            r"\bknob\b",
            r"\bknobs\b",
            r"\bdial\b",
            r"\bdials\b",
            r"\bvalve\b",
            r"\bvalves\b",
        ),
    },
    {
        "part": "handle",
        "joint_type": "revolute",
        "confidence": 0.6,
        "patterns": (
            r"\bhandle\b",
            r"\bhandles\b",
        ),
    },
]


def _normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def infer_primary_joint_type(hint: Dict[str, Any]) -> str | None:
    """Infer the primary joint type from a hint payload."""
    joint_types = hint.get("joint_types") or []
    if not joint_types:
        return None

    counts = Counter(joint_types)
    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], _JOINT_PRIORITY.get(item[0], 999), item[0]),
    )
    return ranked[0][0]


def parse_label_articulation_hint(label: str) -> Dict[str, Any]:
    """Parse articulation hints from one label/category string."""
    if not label:
        return {
            "is_articulated": False,
            "confidence": 0.0,
            "joint_types": [],
            "parts": [],
            "matched_keywords": [],
            "source_text": "",
        }

    normalized = _normalize_label(str(label))
    if not normalized:
        return {
            "is_articulated": False,
            "confidence": 0.0,
            "joint_types": [],
            "parts": [],
            "matched_keywords": [],
            "source_text": "",
        }

    matched_parts: List[str] = []
    matched_joint_types: List[str] = []
    matched_keywords: List[str] = []
    confidences: List[float] = []

    for entry in _HINT_PATTERNS:
        matched_pattern = None
        for pattern in entry["patterns"]:
            if re.search(pattern, normalized):
                matched_pattern = pattern
                break
        if matched_pattern is None:
            continue
        matched_parts.append(entry["part"])
        matched_joint_types.append(entry["joint_type"])
        matched_keywords.append(matched_pattern)
        confidences.append(float(entry["confidence"]))

    if not matched_parts:
        return {
            "is_articulated": False,
            "confidence": 0.0,
            "joint_types": [],
            "parts": [],
            "matched_keywords": [],
            "source_text": normalized,
        }

    unique_parts = list(dict.fromkeys(matched_parts))
    unique_joint_types = list(dict.fromkeys(matched_joint_types))
    unique_keywords = list(dict.fromkeys(matched_keywords))
    base_confidence = max(confidences)
    confidence_bonus = max(0, len(unique_parts) - 1) * 0.04
    confidence = min(0.99, round(base_confidence + confidence_bonus, 4))

    return {
        "is_articulated": True,
        "confidence": confidence,
        "joint_types": unique_joint_types,
        "parts": unique_parts,
        "matched_keywords": unique_keywords,
        "source_text": normalized,
    }


def parse_label_articulation_hints(labels: Sequence[str]) -> List[Dict[str, Any]]:
    """Parse articulation hints for a sequence of labels."""
    return [parse_label_articulation_hint(label) for label in labels]


__all__ = [
    "infer_primary_joint_type",
    "parse_label_articulation_hint",
    "parse_label_articulation_hints",
]
