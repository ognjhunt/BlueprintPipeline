from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Iterable, Mapping

from pydantic import ValidationError

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


def _log_validation_errors(label: str, header: str, messages: Iterable[str]) -> None:
    logger = logging.getLogger(__name__)
    logger.error("%s %s", label, header)
    for message in messages:
        logger.error("%s   - %s", label, message)


def validate_required_env_vars(required_vars: Mapping[str, str], label: str) -> None:
    missing = [name for name in required_vars if not os.getenv(name)]
    if missing:
        _log_validation_errors(label, "ERROR: missing required environment variables:", missing)
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
        formatted_errors = _format_validation_errors(exc.errors())
        _log_validation_errors(label, "ERROR: invalid environment configuration:", formatted_errors)
        print(f"{label} ERROR: invalid environment configuration:", file=sys.stderr)
        for message in formatted_errors:
            print(f"{label}   - {message}", file=sys.stderr)
        sys.exit(1)


def validate_scene_manifest(manifest_path: Path, label: str) -> None:
    try:
        load_and_validate_manifest(manifest_path)
    except FileNotFoundError:
        _log_validation_errors(label, f"ERROR: scene manifest not found at {manifest_path}", [])
        print(
            f"{label} ERROR: scene manifest not found at {manifest_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    except ValidationError as exc:
        formatted_errors = _format_validation_errors(exc.errors())
        _log_validation_errors(label, "ERROR: invalid scene manifest:", formatted_errors)
        print(
            f"{label} ERROR: invalid scene manifest at {manifest_path}:",
            file=sys.stderr,
        )
        for message in formatted_errors:
            print(f"{label}   - {message}", file=sys.stderr)
        sys.exit(1)
