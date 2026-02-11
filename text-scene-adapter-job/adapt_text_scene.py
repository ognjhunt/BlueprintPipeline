#!/usr/bin/env python3
"""Adapt text-scene generation outputs into canonical Stage 1 artifacts."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.source_pipeline.adapter import build_manifest_layout_inventory  # noqa: E402
from tools.validation.entrypoint_checks import validate_required_env_vars  # noqa: E402
from tools.workflow.failure_markers import FailureMarkerWriter  # noqa: E402

JOB_NAME = "text-scene-adapter-job"
GCS_ROOT = Path("/mnt/gcs")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_source_request(request_path: Path, normalized_path: Path) -> Dict[str, Any]:
    if normalized_path.is_file():
        return _load_json(normalized_path)
    return _load_json(request_path)


def main() -> int:
    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
        },
        label="[TEXT-ADAPTER]",
    )

    bucket = os.environ["BUCKET"]
    scene_id = os.environ["SCENE_ID"]

    request_object = os.getenv("REQUEST_OBJECT", f"scenes/{scene_id}/prompts/scene_request.json")
    textgen_prefix = os.getenv("TEXTGEN_PREFIX", f"scenes/{scene_id}/textgen")
    assets_prefix = os.getenv("ASSETS_PREFIX", f"scenes/{scene_id}/assets")
    layout_prefix = os.getenv("LAYOUT_PREFIX", f"scenes/{scene_id}/layout")
    seg_prefix = os.getenv("SEG_PREFIX", f"scenes/{scene_id}/seg")

    request_path = GCS_ROOT / request_object
    package_path = GCS_ROOT / textgen_prefix / "package.json"
    normalized_request_path = GCS_ROOT / textgen_prefix / "request.normalized.json"

    if not request_path.is_file():
        raise FileNotFoundError(f"request payload not found: {request_path}")
    if not package_path.is_file():
        raise FileNotFoundError(f"text generation package not found: {package_path}")

    source_request = _load_source_request(request_path, normalized_request_path)
    textgen_payload = _load_json(package_path)

    result = build_manifest_layout_inventory(
        root=GCS_ROOT,
        scene_id=scene_id,
        assets_prefix=assets_prefix,
        layout_prefix=layout_prefix,
        seg_prefix=seg_prefix,
        textgen_payload=textgen_payload,
        source_request=source_request,
    )

    adapter_summary = {
        "scene_id": scene_id,
        "status": "completed",
        "bucket": bucket,
        "request_object": request_object,
        "textgen_prefix": textgen_prefix,
        "assets_prefix": assets_prefix,
        "layout_prefix": layout_prefix,
        "seg_prefix": seg_prefix,
        "outputs": result,
    }

    _write_json(GCS_ROOT / textgen_prefix / "adapter_summary.json", adapter_summary)
    _write_json(GCS_ROOT / textgen_prefix / ".text_adapter_complete", adapter_summary)

    logger.info(
        "[TEXT-ADAPTER] scene=%s status=completed manifest=%s layout=%s",
        scene_id,
        result.get("manifest_path"),
        result.get("layout_path"),
    )
    return 0


if __name__ == "__main__":
    from tools.logging_config import init_logging
    from tools.startup_validation import validate_and_fail_fast

    init_logging()
    validate_and_fail_fast(job_name="TEXT-SCENE-ADAPTER", validate_gcs=True)

    try:
        sys.exit(main())
    except Exception as exc:
        bucket = os.getenv("BUCKET")
        scene_id = os.getenv("SCENE_ID")
        if bucket and scene_id:
            FailureMarkerWriter(bucket, scene_id, JOB_NAME).write_failure(
                exception=exc,
                failed_step="adapt_text_scene",
                input_params={
                    "request_object": os.getenv("REQUEST_OBJECT", ""),
                    "textgen_prefix": os.getenv("TEXTGEN_PREFIX", ""),
                    "assets_prefix": os.getenv("ASSETS_PREFIX", ""),
                    "layout_prefix": os.getenv("LAYOUT_PREFIX", ""),
                    "seg_prefix": os.getenv("SEG_PREFIX", ""),
                },
                recommendations=[
                    "Verify text-scene-gen-job completed and wrote textgen/package.json.",
                    "Confirm scene_request.json uses schema_version=v1.",
                ],
            )
        raise
