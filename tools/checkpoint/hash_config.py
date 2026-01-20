"""Checkpoint hashing configuration helpers."""

from __future__ import annotations

import os

from tools.config.env import parse_bool_env
from tools.config.production_mode import resolve_production_mode


def resolve_checkpoint_hash_setting() -> bool:
    """Return whether checkpoint output hashes should be stored/validated."""
    raw_setting = os.getenv("BP_CHECKPOINT_HASHES")
    parsed_setting = parse_bool_env(raw_setting, default=None)
    if parsed_setting is None:
        return resolve_production_mode()
    return parsed_setting
