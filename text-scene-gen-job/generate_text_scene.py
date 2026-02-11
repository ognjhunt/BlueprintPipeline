#!/usr/bin/env python3
"""Generate text-driven Stage 1 scene package.

Outputs intermediate artifacts at scenes/<scene_id>/textgen/* for adapter consumption.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.source_pipeline.generator import generate_text_scene_package
from tools.source_pipeline.request import (  # noqa: E402
    PipelineSourceMode,
    QualityTier,
    normalize_scene_request,
    scene_request_to_dict,
)
from tools.validation.entrypoint_checks import validate_required_env_vars
from tools.workflow.failure_markers import FailureMarkerWriter

JOB_NAME = "text-scene-gen-job"
GCS_ROOT = Path("/mnt/gcs")
logger = logging.getLogger(__name__)


def _simplify_prompt(prompt: str) -> str:
    text = re.sub(r"\s+", " ", prompt).strip()
    text = re.sub(r"\b(very|highly|extremely|cinematic|ultra|hyper)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_quality_tier(request_tier: QualityTier) -> QualityTier:
    override = (os.getenv("TEXT_GEN_QUALITY_TIER") or "").strip().lower()
    if not override:
        return request_tier
    try:
        return QualityTier(override)
    except ValueError:
        logger.warning("[TEXT-GEN] Invalid TEXT_GEN_QUALITY_TIER=%s, using request tier %s", override, request_tier.value)
        return request_tier


def _profile_for_tier(tier: QualityTier) -> str:
    if tier == QualityTier.PREMIUM:
        return os.getenv("TEXT_GEN_PREMIUM_PROFILE", "premium_v1")
    return os.getenv("TEXT_GEN_STANDARD_PROFILE", "standard_v1")


def _evaluate_package_quality(package: Dict[str, Any]) -> bool:
    """Return True when package is acceptable for Stage 1 completion."""
    quality = package.get("quality_gate_report") or {}
    metrics = quality.get("metrics") or {}
    object_count_raw = metrics.get("object_count")
    collision_rate_raw = metrics.get("collision_rate_pct")
    stability_raw = metrics.get("stability_pct")

    object_count = int(0 if object_count_raw is None else object_count_raw)
    if collision_rate_raw is None:
        logger.warning("[TEXT-GEN] collision_rate_pct missing from quality gate report, treating as fail (100.0)")
    if stability_raw is None:
        logger.warning("[TEXT-GEN] stability_pct missing from quality gate report, treating as fail (0.0)")
    collision_rate_pct = float(100.0 if collision_rate_raw is None else collision_rate_raw)
    stability_pct = float(0.0 if stability_raw is None else stability_raw)

    if object_count < 5:
        logger.info(
            "[TEXT-GEN] quality-gate fail scene=%s object_count=%s collision_rate_pct=%s stability_pct=%s",
            package.get("scene_id"),
            object_count,
            collision_rate_pct,
            stability_pct,
        )
        return False
    if collision_rate_pct > 6.0:
        logger.info(
            "[TEXT-GEN] quality-gate fail scene=%s object_count=%s collision_rate_pct=%s stability_pct=%s",
            package.get("scene_id"),
            object_count,
            collision_rate_pct,
            stability_pct,
        )
        return False
    if stability_pct < 85.0:
        logger.info(
            "[TEXT-GEN] quality-gate fail scene=%s object_count=%s collision_rate_pct=%s stability_pct=%s",
            package.get("scene_id"),
            object_count,
            collision_rate_pct,
            stability_pct,
        )
        return False
    logger.info(
        "[TEXT-GEN] quality-gate pass scene=%s object_count=%s collision_rate_pct=%s stability_pct=%s",
        package.get("scene_id"),
        object_count,
        collision_rate_pct,
        stability_pct,
    )
    return True


def main() -> int:
    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
        },
        label="[TEXT-GEN]",
    )

    bucket = os.environ["BUCKET"]
    scene_id = os.environ["SCENE_ID"]

    request_object = os.getenv("REQUEST_OBJECT", f"scenes/{scene_id}/prompts/scene_request.json")
    textgen_prefix = os.getenv("TEXTGEN_PREFIX", f"scenes/{scene_id}/textgen")
    default_source_mode = os.getenv("DEFAULT_SOURCE_MODE", "text").strip().lower()
    max_seeds = int(os.getenv("TEXT_GEN_MAX_SEEDS", "16"))
    seed = int(os.getenv("TEXT_SEED", "1"))

    request_path = GCS_ROOT / request_object
    if not request_path.is_file():
        raise FileNotFoundError(f"scene request file not found: {request_path}")

    request_payload = _load_json(request_path)
    request = normalize_scene_request(
        request_payload,
        default_source_mode=PipelineSourceMode(default_source_mode),
        max_seeds=max_seeds,
    )

    if request.source_mode == PipelineSourceMode.IMAGE:
        raise ValueError("text-scene-gen-job received source_mode=image; route to image pipeline instead")

    if seed < 1:
        raise ValueError(f"TEXT_SEED must be >= 1, got {seed}")

    quality_tier = _resolve_quality_tier(request.quality_tier)
    profile = _profile_for_tier(quality_tier)

    prompt = request.prompt or "Generate a realistic robotic manipulation scene"

    package = generate_text_scene_package(
        scene_id=scene_id,
        prompt=prompt,
        quality_tier=quality_tier,
        seed=seed,
        provider_policy=request.provider_policy,
        constraints=request.constraints,
    )

    accepted = _evaluate_package_quality(package)

    retry_used = False
    if not accepted and quality_tier == QualityTier.STANDARD:
        simplified_prompt = _simplify_prompt(prompt)
        if simplified_prompt and simplified_prompt != prompt:
            retry_used = True
            package = generate_text_scene_package(
                scene_id=scene_id,
                prompt=simplified_prompt,
                quality_tier=quality_tier,
                seed=seed,
                provider_policy=request.provider_policy,
                constraints=request.constraints,
            )
            package["retry"] = {
                "used": True,
                "reason": "initial_quality_failed",
                "original_prompt": prompt,
                "simplified_prompt": simplified_prompt,
            }
            accepted = _evaluate_package_quality(package)

    if not accepted:
        raise RuntimeError("text scene generation failed quality gate checks")

    output_root = GCS_ROOT / textgen_prefix
    output_root.mkdir(parents=True, exist_ok=True)

    normalized_request = scene_request_to_dict(request)

    _write_json(output_root / "request.normalized.json", normalized_request)
    _write_json(output_root / "package.json", package)
    _write_json(output_root / "placement_graph.json", package.get("placement_graph") or {"relations": []})
    _write_json(output_root / "quality_gate_report.json", package.get("quality_gate_report") or {})

    quality_report = package.get("quality_gate_report") or {}
    quality_metrics = quality_report.get("metrics") or {}
    generation_mode = "llm" if package.get("used_llm") else "deterministic_fallback"
    llm_attempts = int(package.get("llm_attempts") or 0)
    llm_fallback_used = generation_mode == "deterministic_fallback" and llm_attempts > 0
    llm_failure_reason = package.get("llm_failure_reason")

    logger.info(
        "[TEXT-GEN] quality-eval scene=%s decision=%s metrics=%s generation_mode=%s llm_attempts=%s llm_fallback_used=%s llm_failure_reason=%s",
        scene_id,
        "pass",
        json.dumps(quality_metrics, sort_keys=True),
        generation_mode,
        llm_attempts,
        llm_fallback_used,
        llm_failure_reason,
    )

    completion_payload = {
        "scene_id": scene_id,
        "status": "completed",
        "source_mode": request.source_mode.value,
        "quality_tier": quality_tier.value,
        "provider_policy": request.provider_policy,
        "provider_used": package.get("provider_used"),
        "seed": seed,
        "profile": profile,
        "retry_used": retry_used,
        "generation_mode": generation_mode,
        "llm_attempts": llm_attempts,
        "llm_fallback_used": llm_fallback_used,
        "llm_failure_reason": llm_failure_reason,
        "request_object": request_object,
        "textgen_prefix": textgen_prefix,
    }
    _write_json(output_root / ".textgen_complete", completion_payload)

    logger.info("[TEXT-GEN] scene=%s status=completed objects=%s seed=%s tier=%s", scene_id, len(package.get("objects") or []), seed, quality_tier.value)
    return 0


if __name__ == "__main__":
    from tools.logging_config import init_logging
    from tools.startup_validation import validate_and_fail_fast

    init_logging()
    validate_and_fail_fast(job_name="TEXT-SCENE-GEN", validate_gcs=True)

    try:
        sys.exit(main())
    except Exception as exc:
        bucket = os.getenv("BUCKET")
        scene_id = os.getenv("SCENE_ID")
        if bucket and scene_id:
            FailureMarkerWriter(bucket, scene_id, JOB_NAME).write_failure(
                exception=exc,
                failed_step="generate_text_scene",
                input_params={
                    "request_object": os.getenv("REQUEST_OBJECT", ""),
                    "textgen_prefix": os.getenv("TEXTGEN_PREFIX", ""),
                    "text_seed": os.getenv("TEXT_SEED", "1"),
                    "quality_tier": os.getenv("TEXT_GEN_QUALITY_TIER", ""),
                },
                recommendations=[
                    "Verify scene_request.json payload is valid (schema_version=v1).",
                    "Check provider credentials for configured provider policy.",
                ],
            )
        raise
