#!/usr/bin/env python3
"""Cloud Run entrypoint for the scene batch job.

Reads a JSON scene list payload and forwards execution to tools/run_scene_batch.py.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _parse_int(value: str | None, label: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _parse_float(value: str | None, label: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number") from exc


def _parse_scene_payload(payload: str) -> list[Any]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("SCENE_LIST_JSON must be valid JSON") from exc

    scenes: Iterable[Any]
    if isinstance(data, dict) and "scenes" in data:
        scenes = data["scenes"]
    else:
        scenes = data

    if not isinstance(scenes, list) or not scenes:
        raise ValueError("SCENE_LIST_JSON must contain a non-empty scene list")

    if any(not isinstance(entry, str) or not entry.strip() for entry in scenes):
        raise ValueError("SCENE_LIST_JSON scenes must be non-empty strings")

    return list(scenes)


def _default_path(env_name: str, fallback_suffix: str) -> Path:
    override = os.getenv(env_name)
    if override:
        return Path(override)
    if Path("/mnt/gcs").exists():
        return Path("/mnt/gcs") / fallback_suffix
    return Path(fallback_suffix)


def _append_optional_arg(command: list[str], flag: str, value: str | None) -> None:
    if value is not None and value != "":
        command.extend([flag, value])


def main() -> int:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

    payload = os.getenv("SCENE_LIST_JSON")
    if not payload:
        print("[SCENE-BATCH] ERROR: SCENE_LIST_JSON is required", file=sys.stderr)
        return 1

    try:
        scenes = _parse_scene_payload(payload)
        max_concurrent = _parse_int(os.getenv("MAX_CONCURRENT"), "MAX_CONCURRENT")
        retry_attempts = _parse_int(os.getenv("RETRY_ATTEMPTS"), "RETRY_ATTEMPTS")
        retry_delay = _parse_float(os.getenv("RETRY_DELAY"), "RETRY_DELAY")
        rate_limit = _parse_float(os.getenv("RATE_LIMIT"), "RATE_LIMIT")
    except ValueError as exc:
        print(f"[SCENE-BATCH] ERROR: {exc}", file=sys.stderr)
        return 1

    scene_root = _default_path("SCENE_ROOT", "scenes")
    reports_dir = _default_path("REPORTS_DIR", "batch_reports")

    with tempfile.TemporaryDirectory(prefix="scene-batch-") as temp_dir:
        scene_list_path = Path(temp_dir) / "scene_list.json"
        scene_list_path.write_text(json.dumps({"scenes": scenes}, indent=2))

        command = [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_scene_batch.py"),
            "--scene-root",
            str(scene_root),
            "--scene-list",
            str(scene_list_path),
            "--reports-dir",
            str(reports_dir),
        ]

        _append_optional_arg(command, "--max-concurrent", str(max_concurrent) if max_concurrent else None)
        _append_optional_arg(command, "--retry-attempts", str(retry_attempts) if retry_attempts else None)
        _append_optional_arg(command, "--retry-delay", str(retry_delay) if retry_delay else None)
        _append_optional_arg(command, "--rate-limit", str(rate_limit) if rate_limit else None)
        _append_optional_arg(command, "--steps", os.getenv("STEPS"))
        _append_optional_arg(command, "--resume-from", os.getenv("RESUME_FROM"))
        _append_optional_arg(command, "--checkpoint-step", os.getenv("CHECKPOINT_STEP"))
        _append_optional_arg(command, "--dlq-path", os.getenv("DLQ_PATH"))
        _append_optional_arg(command, "--log-level", log_level)

        if _is_truthy(os.getenv("VALIDATE")):
            command.append("--validate")
        if _is_truthy(os.getenv("SKIP_INTERACTIVE")):
            command.append("--skip-interactive")
        if _is_truthy(os.getenv("ENABLE_DWM")):
            command.append("--enable-dwm")
        if _is_truthy(os.getenv("ENABLE_DREAM2FLOW")):
            command.append("--enable-dream2flow")
        if _is_truthy(os.getenv("ENABLE_EXPERIMENTAL")):
            command.append("--enable-experimental")
        if _is_truthy(os.getenv("ENABLE_INVENTORY_ENRICHMENT")):
            command.append("--enable-inventory-enrichment")
        if _is_truthy(os.getenv("DISABLE_ARTICULATED_ASSETS")):
            command.append("--disable-articulated-assets")
        if _is_truthy(os.getenv("SKIP_COMPLETED")):
            command.append("--skip-completed")

        print(f"[SCENE-BATCH] Running: {' '.join(command)}")
        result = subprocess.run(command, check=False)

    if result.returncode == 0:
        print(f"[SCENE-BATCH] Batch reports directory: {reports_dir}")
        print(f"[SCENE-BATCH] Batch report: {reports_dir / 'batch_report.json'}")
        print(f"[SCENE-BATCH] Dead-letter queue: {reports_dir / 'dead_letter_queue.json'}")

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
