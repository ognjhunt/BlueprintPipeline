#!/usr/bin/env python3
"""Cleanup Genie Sim recordings/logs older than a retention window."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from tools.config.env import parse_int_env

GENIESIM_RECORDINGS_DIR_ENV = "GENIESIM_RECORDINGS_DIR"
GENIESIM_LOG_DIR_ENV = "GENIESIM_LOG_DIR"
GENIESIM_CLEANUP_RETENTION_HOURS_ENV = "GENIESIM_CLEANUP_RETENTION_HOURS"

DEFAULT_RECORDINGS_DIR = Path("/tmp/geniesim_recordings")
DEFAULT_LOG_DIR = Path("/tmp/geniesim_logs")
DEFAULT_RETENTION_HOURS = 168


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete Genie Sim recordings/logs older than a retention window.",
    )
    parser.add_argument(
        "--retention-hours",
        type=int,
        default=None,
        help="Retention window in hours (overrides GENIESIM_CLEANUP_RETENTION_HOURS).",
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=None,
        help="Recordings directory (overrides GENIESIM_RECORDINGS_DIR).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Log directory (overrides GENIESIM_LOG_DIR).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log deletions without removing files/directories.",
    )
    return parser.parse_args()


def _resolve_retention_hours(value: Optional[int]) -> int:
    if value is not None:
        return value
    return parse_int_env(
        os.getenv(GENIESIM_CLEANUP_RETENTION_HOURS_ENV),
        default=DEFAULT_RETENTION_HOURS,
        min_value=1,
        name=GENIESIM_CLEANUP_RETENTION_HOURS_ENV,
    )


def _resolve_dir(arg_value: Optional[Path], env_key: str, default: Path) -> Path:
    if arg_value is not None:
        return arg_value
    env_value = os.getenv(env_key)
    return Path(env_value) if env_value else default


def _iter_targets(base_dir: Path) -> Iterable[Path]:
    if not base_dir.exists():
        return []
    if not base_dir.is_dir():
        return []
    return list(base_dir.iterdir())


def _cleanup_entry(path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would remove {path}")
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            return


def cleanup_paths(
    base_dirs: Iterable[Path],
    retention_hours: int,
    dry_run: bool = False,
) -> int:
    cutoff = time.time() - (retention_hours * 3600)
    deleted = 0
    cutoff_dt = datetime.fromtimestamp(cutoff, tz=timezone.utc)
    for base_dir in base_dirs:
        base_dir = base_dir.expanduser()
        if not base_dir.exists():
            print(f"Skipping missing directory: {base_dir}")
            continue
        print(f"Scanning {base_dir} for entries older than {cutoff_dt.isoformat()}")
        for entry in _iter_targets(base_dir):
            try:
                mtime = entry.stat().st_mtime
            except FileNotFoundError:
                continue
            if mtime >= cutoff:
                continue
            _cleanup_entry(entry, dry_run)
            deleted += 1
    return deleted


def main() -> int:
    args = _parse_args()
    retention_hours = _resolve_retention_hours(args.retention_hours)
    if retention_hours <= 0:
        print("Retention window must be positive; skipping cleanup.")
        return 1
    recordings_dir = _resolve_dir(args.recordings_dir, GENIESIM_RECORDINGS_DIR_ENV, DEFAULT_RECORDINGS_DIR)
    log_dir = _resolve_dir(args.log_dir, GENIESIM_LOG_DIR_ENV, DEFAULT_LOG_DIR)

    deleted = cleanup_paths([recordings_dir, log_dir], retention_hours, dry_run=args.dry_run)
    print(f"Cleanup complete. Removed {deleted} entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
