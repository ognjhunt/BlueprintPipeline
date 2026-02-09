"""Gemini-powered auto-labeling for SAM3 segmentation.

Analyzes an input image using Gemini to determine what objects are present,
producing a dynamic label list for SAM3 text-prompted segmentation.
This makes the pipeline scene-agnostic — it works on kitchens, workshops,
offices, bedrooms, etc. without hardcoded label lists.

Runs locally (Mac) before uploading to the remote VM.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

# Prompt instructs Gemini to return a JSON array of short noun phrases
_LABEL_PROMPT = """\
List all distinct objects and furniture visible in this image.
Return a JSON array of short noun phrases, one per object type.
Include structural elements like "floor", "wall", "ceiling" if visible.
Be specific: prefer "office chair" over just "chair", "potted plant" over "plant".
Do NOT include people.
Return at most 20 labels.

Example output: ["office chair", "wooden desk", "monitor", "potted plant", "floor"]
"""


def generate_labels_from_image(
    image_path: str,
    fallback_labels: Optional[List[str]] = None,
) -> List[str]:
    """Call Gemini to analyze an image and return object labels.

    Args:
        image_path: Path to the input image.
        fallback_labels: Labels to use if Gemini fails.

    Returns:
        List of object label strings for SAM3 segmentation.
    """
    _default_fallback = [
        "chair", "table", "sofa", "cabinet", "shelf", "counter",
        "sink", "refrigerator", "oven", "microwave", "lamp",
        "plant in pot", "bed", "desk", "stool", "floor",
    ]
    fallback = fallback_labels or _default_fallback

    if not Path(image_path).is_file():
        log.warning("Image not found: %s — using fallback labels", image_path)
        return fallback

    try:
        from tools.llm_client.client import create_llm_client, LLMProvider

        client = create_llm_client(provider=LLMProvider.GEMINI)
        response = client.generate(
            prompt=_LABEL_PROMPT,
            image=image_path,
            json_output=True,
            temperature=0.3,
            disable_tools=True,
        )

        labels = _parse_label_response(response.parse_json())

        if not labels:
            log.warning("Gemini returned no labels — using fallback")
            return fallback

        log.info("Gemini detected %d labels: %s", len(labels), labels)
        return labels

    except Exception as exc:
        log.warning(
            "Gemini label generation failed (%s) — using fallback labels",
            exc,
        )
        return fallback


def _parse_label_response(data) -> List[str]:
    """Extract a list of string labels from various JSON response shapes."""
    # Direct array: ["chair", "table", ...]
    if isinstance(data, list):
        return [str(l) for l in data if isinstance(l, str) and l.strip()]

    # Wrapped in a dict: {"labels": [...]} or {"objects": [...]}
    if isinstance(data, dict):
        for key in ("labels", "objects", "items", "object_labels"):
            if key in data and isinstance(data[key], list):
                return [str(l) for l in data[key] if isinstance(l, str) and l.strip()]
        # Single key with array value
        for val in data.values():
            if isinstance(val, list) and val and isinstance(val[0], str):
                return [str(l) for l in val if isinstance(l, str) and l.strip()]

    return []
