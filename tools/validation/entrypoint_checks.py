from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Mapping

from pydantic import ValidationError

from tools.config.production_mode import (
    ensure_config_audit_for_production,
    is_config_audit_enabled,
    resolve_production_mode,
)
from tools.validation.config_schemas import (
    load_and_validate_env_config,
    load_and_validate_manifest,
)


def _format_validation_errors(errors: Iterable[dict]) -> list[str]:
    formatted = []
    for err in errors:
        loc = ".".join(str(part) for part in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        formatted.append(f"{loc}: {msg}" if loc else msg)
    return formatted


def validate_required_env_vars(required_vars: Mapping[str, str], label: str) -> None:
    missing = [name for name in required_vars if not os.getenv(name)]
    if missing:
        print(f"{label} ERROR: missing required environment variables:", file=sys.stderr)
        for name in missing:
            hint = required_vars.get(name)
            if hint:
                print(f"{label}   - {name}: {hint}", file=sys.stderr)
            else:
                print(f"{label}   - {name}", file=sys.stderr)
        sys.exit(1)

    try:
        load_and_validate_env_config()
    except ValidationError as exc:
        print(f"{label} ERROR: invalid environment configuration:", file=sys.stderr)
        for message in _format_validation_errors(exc.errors()):
            print(f"{label}   - {message}", file=sys.stderr)
        sys.exit(1)


def validate_scene_manifest(manifest_path: Path, label: str) -> None:
    try:
        load_and_validate_manifest(manifest_path)
    except FileNotFoundError:
        print(
            f"{label} ERROR: scene manifest not found at {manifest_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    except ValidationError as exc:
        print(
            f"{label} ERROR: invalid scene manifest at {manifest_path}:",
            file=sys.stderr,
        )
        for message in _format_validation_errors(exc.errors()):
            print(f"{label}   - {message}", file=sys.stderr)
        sys.exit(1)


def warn_if_production_audit_disabled(label: str) -> None:
    ensure_config_audit_for_production()
    if resolve_production_mode() and not is_config_audit_enabled():
        print(
            f"{label} WARNING: production mode detected but BP_ENABLE_CONFIG_AUDIT=0; "
            "config audit trail is disabled.",
            file=sys.stderr,
        )
