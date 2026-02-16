#!/usr/bin/env python3
"""
Resolve and optionally migrate the SAGE Isaac Sim MCP extension path.

Canonical path:
  <SAGE_DIR>/server/isaacsim_mcp_ext/isaac.sim.mcp_extension

Legacy path:
  <SAGE_DIR>/server/isaacsim/isaac.sim.mcp_extension
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MCPPathResolution:
    resolved_path: Optional[Path]
    state: str
    message: str


def _is_writable_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and os.access(path, os.W_OK | os.X_OK)


def _is_writable_path(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    parent = path.parent
    return _is_writable_dir(parent)


def resolve_mcp_extension_path(sage_dir: Path, *, migrate_legacy: bool = True) -> MCPPathResolution:
    new_path = sage_dir / "server" / "isaacsim_mcp_ext" / "isaac.sim.mcp_extension"
    legacy_path = sage_dir / "server" / "isaacsim" / "isaac.sim.mcp_extension"

    new_exists = new_path.is_dir()
    legacy_exists = legacy_path.is_dir()

    if new_exists:
        if legacy_exists and legacy_path.resolve() != new_path.resolve():
            return MCPPathResolution(
                resolved_path=new_path,
                state="new_with_legacy",
                message=(
                    "Legacy MCP extension path is deprecated: "
                    f"{legacy_path}. Using canonical path: {new_path}"
                ),
            )
        return MCPPathResolution(resolved_path=new_path, state="new", message=f"Using MCP extension: {new_path}")

    if not legacy_exists:
        return MCPPathResolution(
            resolved_path=None,
            state="missing",
            message=(
                "MCP extension missing in both canonical and legacy paths: "
                f"{new_path} ; {legacy_path}"
            ),
        )

    if not migrate_legacy:
        return MCPPathResolution(
            resolved_path=legacy_path,
            state="legacy",
            message=(
                "Using deprecated legacy MCP extension path (migration disabled): "
                f"{legacy_path}"
            ),
        )

    # Attempt best-effort migration from legacy path to canonical path.
    try:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if not _is_writable_path(new_path) or not _is_writable_path(legacy_path):
            raise PermissionError("insufficient permissions to migrate legacy MCP extension path")
        shutil.move(str(legacy_path), str(new_path))
        # Preserve old references via symlink if possible.
        try:
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.symlink_to(new_path)
        except Exception:
            # Symlink is best-effort; migration still succeeded.
            pass
        return MCPPathResolution(
            resolved_path=new_path,
            state="migrated",
            message=(
                "Migrated legacy MCP extension path to canonical location: "
                f"{legacy_path} -> {new_path}"
            ),
        )
    except Exception as exc:
        return MCPPathResolution(
            resolved_path=legacy_path,
            state="legacy_unmigrated",
            message=(
                "Legacy MCP extension path detected but migration failed; using legacy path. "
                f"error={type(exc).__name__}: {exc}; path={legacy_path}"
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve and migrate SAGE Isaac Sim MCP extension path")
    parser.add_argument("--sage-dir", default=os.environ.get("SAGE_DIR", "/workspace/SAGE"))
    parser.add_argument("--migrate-legacy", dest="migrate_legacy", action="store_true", default=True)
    parser.add_argument("--no-migrate-legacy", dest="migrate_legacy", action="store_false")
    parser.add_argument("--quiet", action="store_true", default=False)
    args = parser.parse_args()

    result = resolve_mcp_extension_path(Path(args.sage_dir), migrate_legacy=bool(args.migrate_legacy))
    if not args.quiet:
        level = "INFO" if result.state in {"new", "migrated"} else "WARNING"
        print(f"[mcp-ext-path {level}] {result.message}", file=sys.stderr)
    if result.resolved_path is None:
        return 1
    print(str(result.resolved_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

