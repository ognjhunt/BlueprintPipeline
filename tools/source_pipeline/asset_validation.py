from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if len(tok) > 2}


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dims_score(obj: Mapping[str, Any]) -> float:
    dims = obj.get("dimensions_est") if isinstance(obj.get("dimensions_est"), Mapping) else {}
    w = max(0.01, _safe_float(dims.get("width"), 0.25))
    h = max(0.01, _safe_float(dims.get("height"), 0.25))
    d = max(0.01, _safe_float(dims.get("depth"), 0.25))
    ratios = sorted([w / h, h / d, w / d])
    if ratios[-1] > 12.0:
        return 0.2
    if ratios[-1] > 8.0:
        return 0.45
    return 0.9


def validate_asset_candidate(
    *,
    obj: Mapping[str, Any],
    source_kind: str,
    source_path: str,
    room_type: str,
) -> Dict[str, Any]:
    """
    Lightweight validator stub used before scene insertion.

    This is deliberately deterministic and cheap so it can run in CI/prod without
    mandatory VLM dependencies. It still exposes a schema compatible with future
    VLM-backed validation.
    """
    category = str(obj.get("category") or obj.get("name") or "object").lower()
    name = str(obj.get("name") or category).lower()
    source_tokens = _tokenize(source_path)
    semantic_tokens = _tokenize(f"{category} {name}")

    type_match = 1.0 if (semantic_tokens & source_tokens) else 0.55
    style_consistency = 0.8 if room_type else 0.7
    single_object = 0.95 if "+" not in source_path and "," not in source_path else 0.45
    completeness = 0.9 if source_kind not in {"placeholder_usd"} else 0.6
    proportions = _dims_score(obj)
    closed_state = 0.9

    score = (
        0.30 * type_match
        + 0.15 * style_consistency
        + 0.15 * single_object
        + 0.15 * completeness
        + 0.15 * proportions
        + 0.10 * closed_state
    )
    passed = score >= 0.70

    return {
        "schema_version": "v1",
        "passed": passed,
        "status": "pass" if passed else "warn",
        "score": round(score, 4),
        "checks": {
            "type_match": round(type_match, 4),
            "style_consistency": round(style_consistency, 4),
            "single_object": round(single_object, 4),
            "completeness": round(completeness, 4),
            "proportions": round(proportions, 4),
            "closed_state": round(closed_state, 4),
        },
        "source_kind": source_kind,
        "source_path": source_path,
    }


# ---------------------------------------------------------------------------
# VLM-backed asset validation (SceneSmith-style quality gate)
# ---------------------------------------------------------------------------
_VLM_VALIDATION_PROMPT = (
    "You are a 3D asset quality inspector. Evaluate this image of a 3D asset.\n"
    "Expected object: {category} ({description}).\n"
    "Score each criterion 1-5:\n"
    "  type_match: Is this actually a {category}? (1=wrong object, 5=perfect)\n"
    "  proportions: Are dimensions realistic? (1=absurd, 5=lifelike)\n"
    "  completeness: Is the object whole, not truncated? (1=fragment, 5=complete)\n"
    "  single_object: Is there exactly one object, no background clutter? (1=multiple, 5=single clean)\n"
    "Return ONLY JSON: {{\"type_match\":N,\"proportions\":N,\"completeness\":N,\"single_object\":N}}"
)

_VLM_PASS_THRESHOLD = 3  # minimum per-criterion score to pass


def _vlm_validation_enabled() -> bool:
    raw = str(os.environ.get("TEXT_ASSET_VLM_VALIDATION_ENABLED", "")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _find_reference_image(asset_dir: Path) -> Optional[Path]:
    """Look for a reference image in the asset directory."""
    for candidate in ("reference.png", "reference.jpg", "thumbnail.png", "render.png"):
        path = asset_dir / candidate
        if path.exists():
            return path
    return None


def validate_asset_with_vlm(
    *,
    obj: Mapping[str, Any],
    asset_dir: Path,
    source_kind: str,
    source_path: str,
    room_type: str,
    retry_budget: int = 1,
) -> Dict[str, Any]:
    """Validate an asset using VLM (Gemini) if enabled and a reference image exists.

    Falls back to deterministic ``validate_asset_candidate`` when VLM is
    unavailable or the image is missing.
    """
    # Always compute deterministic baseline
    deterministic = validate_asset_candidate(
        obj=obj,
        source_kind=source_kind,
        source_path=source_path,
        room_type=room_type,
    )

    if not _vlm_validation_enabled():
        return deterministic

    image_path = _find_reference_image(asset_dir)
    if image_path is None:
        return deterministic

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return deterministic

    category = str(obj.get("category") or obj.get("name") or "object").lower()
    description = str(obj.get("description") or category)
    prompt = _VLM_VALIDATION_PROMPT.format(category=category, description=description)

    try:
        import google.generativeai as genai  # type: ignore[import-untyped]

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        image_bytes = image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode()
        mime = "image/png" if image_path.suffix == ".png" else "image/jpeg"

        response = model.generate_content(
            [
                {"inline_data": {"mime_type": mime, "data": image_b64}},
                prompt,
            ],
            generation_config={"temperature": 0.1},
        )
        text = (response.text or "").strip()
        # Extract JSON from response (may be wrapped in markdown fences)
        json_match = re.search(r"\{[^}]+\}", text)
        if json_match:
            vlm_scores = json.loads(json_match.group())
        else:
            logger.warning("[VLM-VALIDATE] could not parse JSON from response: %s", text[:200])
            return deterministic

        checks = {}
        all_pass = True
        for key in ("type_match", "proportions", "completeness", "single_object"):
            raw_score = _safe_float(vlm_scores.get(key), 3.0)
            clamped = max(1.0, min(5.0, raw_score))
            checks[key] = round(clamped, 2)
            if clamped < _VLM_PASS_THRESHOLD:
                all_pass = False

        vlm_score = sum(checks.values()) / (5.0 * len(checks))  # normalize to 0-1

        return {
            "schema_version": "v1",
            "passed": all_pass,
            "status": "pass" if all_pass else "fail",
            "score": round(vlm_score, 4),
            "checks": checks,
            "source_kind": source_kind,
            "source_path": source_path,
            "validation_method": "vlm",
            "vlm_model": "gemini-2.0-flash",
            "deterministic_fallback": deterministic,
        }
    except Exception as exc:
        logger.warning("[VLM-VALIDATE] VLM validation failed (%s), using deterministic", exc)
        return deterministic


def is_validation_enabled() -> bool:
    raw = str(os.environ.get("TEXT_ASSET_VALIDATION_ENABLED", "")).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    pipeline_v2 = str(os.environ.get("TEXT_PIPELINE_V2_ENABLED", "")).strip().lower()
    return pipeline_v2 in {"1", "true", "yes", "y", "on"}


def should_soft_repair(validation: Mapping[str, Any]) -> bool:
    if not is_validation_enabled():
        return False
    return not bool(validation.get("passed"))
