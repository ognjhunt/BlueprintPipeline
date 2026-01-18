"""Retention cleanup for pipeline artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .store import CheckpointRecord
from tools.config.env import parse_bool_env


DEFAULT_PIPELINE_RETENTION_DAYS = 30
DEFAULT_INPUT_RETENTION_DAYS = 90
DEFAULT_OUTPUT_RETENTION_DAYS = 365
DEFAULT_LOG_RETENTION_DAYS = 180


@dataclass(frozen=True)
class RetentionPolicy:
    """Retention window per artifact class."""

    inputs_days: int
    intermediate_days: int
    outputs_days: int
    logs_days: int
    fallback_days: int

    def days_for_class(self, class_name: str) -> int:
        if class_name == "inputs":
            return self.inputs_days
        if class_name == "intermediate":
            return self.intermediate_days
        if class_name == "outputs":
            return self.outputs_days
        if class_name == "logs":
            return self.logs_days
        return self.fallback_days


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Invalid value for {name}: {value}")


def load_retention_policy(base_days: Optional[int]) -> RetentionPolicy:
    pipeline_days = base_days or _int_env(
        "PIPELINE_RETENTION_DAYS", DEFAULT_PIPELINE_RETENTION_DAYS
    )
    return RetentionPolicy(
        inputs_days=_int_env("PIPELINE_INPUT_RETENTION_DAYS", DEFAULT_INPUT_RETENTION_DAYS),
        intermediate_days=_int_env("PIPELINE_INTERMEDIATE_RETENTION_DAYS", pipeline_days),
        outputs_days=_int_env("PIPELINE_OUTPUT_RETENTION_DAYS", DEFAULT_OUTPUT_RETENTION_DAYS),
        logs_days=_int_env("PIPELINE_LOG_RETENTION_DAYS", DEFAULT_LOG_RETENTION_DAYS),
        fallback_days=pipeline_days,
    )


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (path for path in root.rglob("*") if path.is_file())


def _load_checkpoint_outputs(checkpoint_file: Path) -> List[Path]:
    try:
        payload = json.loads(checkpoint_file.read_text())
    except json.JSONDecodeError:
        return []
    record = CheckpointRecord.from_dict(payload)
    outputs: List[Path] = []
    for output_path in record.output_paths:
        output = Path(output_path)
        outputs.append(output)
    return outputs


def _classify_path(path: Path, scene_root: Path) -> Optional[str]:
    try:
        relative = path.relative_to(scene_root)
    except ValueError:
        return None
    if not relative.parts:
        return None
    first = relative.parts[0]
    if first == "input":
        return "inputs"
    if first in {".checkpoints", "layout", "seg"}:
        return "intermediate"
    if first in {"assets", "usd", "replicator", "variation_assets", "isaac_lab", "episodes"}:
        return "outputs"
    if first in {"logs", "log"}:
        return "logs"
    return None


def _gather_candidates(scene_root: Path) -> Dict[Path, str]:
    candidates: Dict[Path, str] = {}
    class_map = {
        "inputs": ["input"],
        "intermediate": [".checkpoints", "layout", "seg"],
        "outputs": ["assets", "usd", "replicator", "variation_assets", "isaac_lab", "episodes"],
        "logs": ["logs", "log"],
    }
    for class_name, subdirs in class_map.items():
        for subdir in subdirs:
            for file_path in _iter_files(scene_root / subdir):
                candidates[file_path] = class_name
    checkpoints_dir = scene_root / ".checkpoints"
    for checkpoint_file in _iter_files(checkpoints_dir):
        for output_path in _load_checkpoint_outputs(checkpoint_file):
            resolved = output_path
            if not resolved.is_absolute():
                resolved = scene_root / output_path
            if resolved.exists():
                candidates.setdefault(
                    resolved, _classify_path(resolved, scene_root) or "intermediate"
                )
    return candidates


def _remove_empty_dirs(start: Path, stop: Path) -> None:
    current = start
    while current != stop and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _log_action(action: str, payload: Dict[str, object]) -> None:
    logging.info(json.dumps({"action": action, **payload}, sort_keys=True))


def cleanup_scene(
    scene_root: Path,
    policy: RetentionPolicy,
    *,
    now: datetime,
    dry_run: bool,
) -> Tuple[int, int]:
    deleted = 0
    considered = 0
    candidates = _gather_candidates(scene_root)
    for file_path, class_name in candidates.items():
        considered += 1
        retention_days = policy.days_for_class(class_name)
        cutoff = now - timedelta(days=retention_days)
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
        if mtime >= cutoff:
            continue
        payload = {
            "class": class_name,
            "path": str(file_path),
            "retention_days": retention_days,
            "last_modified": mtime.isoformat(),
            "scene": scene_root.name,
            "deleted_at": now.isoformat(),
        }
        if dry_run:
            _log_action("dry_run", payload)
            continue
        try:
            file_path.unlink()
            _remove_empty_dirs(file_path.parent, scene_root)
            deleted += 1
            _log_action("deleted", payload)
        except OSError as exc:
            _log_action("delete_failed", {**payload, "error": str(exc)})
    return deleted, considered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup pipeline artifacts by retention policy.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.getenv("PIPELINE_STORAGE_ROOT", "/mnt/gcs")),
        help="Storage root containing scenes/",
    )
    parser.add_argument(
        "--scene-prefix",
        type=str,
        default="scenes",
        help="Prefix under root for scene data.",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="Override PIPELINE_RETENTION_DAYS for intermediate retention.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=parse_bool_env(os.getenv("PIPELINE_RETENTION_DRY_RUN"), default=False),
        help="Log deletions without removing files.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    policy = load_retention_policy(args.retention_days)
    root = args.root / args.scene_prefix
    now = datetime.now(timezone.utc)
    if not root.exists():
        _log_action(
            "skip",
            {
                "reason": "root_missing",
                "root": str(root),
                "retention": policy.__dict__,
                "checked_at": now.isoformat(),
            },
        )
        return 0
    total_deleted = 0
    total_considered = 0
    for scene_root in root.iterdir():
        if not scene_root.is_dir():
            continue
        deleted, considered = cleanup_scene(scene_root, policy, now=now, dry_run=args.dry_run)
        total_deleted += deleted
        total_considered += considered
    _log_action(
        "summary",
        {
            "root": str(root),
            "retention": policy.__dict__,
            "dry_run": args.dry_run,
            "deleted": total_deleted,
            "considered": total_considered,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
