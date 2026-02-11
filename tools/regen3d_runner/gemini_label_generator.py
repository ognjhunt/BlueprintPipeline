"""Gemini-powered auto-labeling for SAM3 segmentation.

Analyzes an input image using Gemini to determine what objects are present,
producing a dynamic label list for SAM3 text-prompted segmentation.
This makes the pipeline scene-agnostic — it works on kitchens, workshops,
offices, bedrooms, etc. without any hardcoded label lists.

Uses Agentic Vision (Gemini 3 Flash code_execution) to zoom and crop into
regions of the image for better small-object detection. The model can write
Python code to manipulate the image (zoom, crop, annotate) and re-examine
regions before producing the final label list.

Runs locally (Mac) before uploading to the remote VM.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

# Prompt instructs Gemini to return a JSON array of short noun phrases.
# Encourages agentic vision: the model can zoom/crop into image regions
# using code execution to find small or partially occluded objects.
_LABEL_PROMPT = """\
Carefully analyze this image to identify ALL distinct objects and furniture.

First, examine the full image. Then zoom into different regions (corners, shelves,
countertops, background areas) to find small or partially hidden objects.
Use code execution to crop and inspect areas that might contain objects you missed.

Return a JSON array of short noun phrases, one per object type.
Do not include pure structural/background surfaces like "floor", "wall", "ceiling", or "window".
Be specific: prefer "office chair" over just "chair", "potted plant" over "plant".
Prefer whole, room-level objects over tiny parts.
Do not list tiny components like screws, knobs, handles, hinges, or display sub-parts.
If an object has likely articulated parts, encode that cue naturally in the label.
Examples: "desk with drawers", "cabinet with doors", "box with hinged lid".
Look for articulated object cues (drawers/doors/lids), but return the parent object label.
Do NOT include people.
Return at most 25 labels.

Example output: ["office chair", "wooden desk", "monitor", "potted plant", "floor"]
"""

_TINY_PART_KEYWORDS = {
    "knob", "handle", "hinge", "screw", "button", "switch", "dial", "display panel",
    "control panel", "baseboard", "window frame", "trim",
}

_STRUCTURAL_LABEL_KEYWORDS = {
    "floor",
    "wall",
    "ceiling",
    "window",
    "window curtain",
    "curtain",
    "countertop",
    "tile",
    "tiles",
    "baseboard",
    "trim",
    "column",
    "pillar",
}

_NORMALIZATION_STOPWORDS = {
    "a", "an", "the", "with", "and", "of", "in", "on", "for", "to",
}


def generate_labels_from_image(
    image_path: str,
    *,
    max_labels: int = 12,
    quality_mode: str = "quality",
) -> List[str]:
    """Call Gemini to analyze an image and return object labels.

    Args:
        image_path: Path to the input image.

    Returns:
        List of object label strings for SAM3 segmentation.

    Raises:
        FileNotFoundError: If the image doesn't exist.
        RuntimeError: If Gemini fails to produce labels.
    """
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    from tools.llm_client.client import (
        GeminiClient,
        FallbackLLMClient,
    )

    # Fallback chain: Gemini 3 Flash → Gemini 3 Pro → Gemini 2.5 Flash
    # All support image input + JSON output.
    primary = GeminiClient(model="gemini-3-flash-preview")
    fallbacks = [
        GeminiClient(model="gemini-3-pro-preview"),
        GeminiClient(model="gemini-2.5-flash"),
    ]
    client = FallbackLLMClient(primary=primary, fallbacks=fallbacks)

    response = client.generate(
        prompt=_LABEL_PROMPT,
        image=image_path,
        json_output=True,
        temperature=0.3,
    )

    # With code_execution (agentic vision), the model may run Python code
    # to zoom/crop the image before producing labels. The final JSON output
    # is extracted from response.text / parse_json() as usual.
    try:
        parsed = response.parse_json()
    except (json.JSONDecodeError, ValueError) as e:
        # Agentic vision code execution can sometimes produce mixed output.
        # Try to extract JSON array from the raw text as a fallback.
        log.warning("JSON parse failed (%s), attempting regex extraction from response", e)
        parsed = _extract_json_from_text(response.text)

    labels = _parse_label_response(parsed)
    labels = _post_process_labels(
        labels,
        max_labels=max_labels,
        quality_mode=quality_mode,
    )

    if not labels:
        raise RuntimeError(
            f"Gemini returned no labels for image: {image_path}. "
            f"Raw response: {response.text[:500]}"
        )

    log.info("Gemini detected %d labels: %s", len(labels), labels)
    return labels


def _extract_json_from_text(text: str):
    """Try to extract a JSON array from mixed text (code execution output)."""
    # Look for [...] pattern in the text
    match = re.search(r'\[[\s\S]*?\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Look for {...} pattern
    match = re.search(r'\{[\s\S]*?\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def _parse_label_response(data) -> List[str]:
    """Extract a list of string labels from various JSON response shapes."""
    # Direct array: ["office chair", "wooden desk", ...]
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


def _normalize_label_text(label: str) -> str:
    text = label.strip().lower()
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(label: str) -> List[str]:
    norm = _normalize_label_text(label)
    tokens = []
    for token in norm.split():
        if not token or token in _NORMALIZATION_STOPWORDS:
            continue
        if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


def _is_tiny_part_label(label: str) -> bool:
    norm = _normalize_label_text(label)
    return any(keyword in norm for keyword in _TINY_PART_KEYWORDS)


def _is_structural_label(label: str) -> bool:
    norm = _normalize_label_text(label)
    return any(keyword == norm or keyword in norm for keyword in _STRUCTURAL_LABEL_KEYWORDS)


def _is_semantic_duplicate(candidate_tokens: List[str], kept_tokens: List[str]) -> bool:
    if not candidate_tokens or not kept_tokens:
        return False
    cset = set(candidate_tokens)
    kset = set(kept_tokens)
    if cset == kset:
        return True
    # Treat subset/superset as duplicates for near-identical labels.
    if cset.issubset(kset) or kset.issubset(cset):
        return True
    overlap = len(cset & kset)
    return overlap > 0 and overlap >= min(len(cset), len(kset))


def _post_process_labels(
    raw_labels: List[str],
    *,
    max_labels: int,
    quality_mode: str,
) -> List[str]:
    if max_labels < 1:
        max_labels = 1
    mode = (quality_mode or "quality").strip().lower()
    filtered: List[str] = []
    kept_tokens: List[List[str]] = []

    for raw in raw_labels:
        label = " ".join(raw.strip().split())
        if not label:
            continue
        tokens = _tokenize(label)
        if not tokens:
            continue
        if mode == "quality" and _is_tiny_part_label(label):
            continue
        if mode == "quality" and _is_structural_label(label):
            continue
        if any(_is_semantic_duplicate(tokens, existing) for existing in kept_tokens):
            continue
        filtered.append(label)
        kept_tokens.append(tokens)
        if len(filtered) >= max_labels:
            break

    return filtered
