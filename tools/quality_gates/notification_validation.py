"""Validation helpers for quality gate notification configuration."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

from tools.config.production_mode import resolve_production_mode
from tools.config import QualityConfig


def _normalize_channels(raw_channels: Optional[object]) -> list[str]:
    if raw_channels is None:
        return []
    if isinstance(raw_channels, str):
        return [item.strip() for item in raw_channels.split(",") if item.strip()]
    if isinstance(raw_channels, Sequence):
        return [str(item).strip() for item in raw_channels if str(item).strip()]
    if isinstance(raw_channels, Iterable):
        return [str(item).strip() for item in raw_channels if str(item).strip()]
    return []


def ensure_production_notification_channels(
    config: Optional[QualityConfig],
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Ensure production runs have notification channels configured."""
    if not resolve_production_mode(env):
        return

    channels: list[str] = []
    if config and getattr(config, "human_approval", None):
        channels = _normalize_channels(config.human_approval.notification_channels)

    if not channels:
        raise ValueError(
            "Production mode requires notification channels for quality gate approvals. "
            "Configure human_approval.notification_channels in tools/quality_gates/quality_config.json "
            "or set BP_QUALITY_HUMAN_APPROVAL_NOTIFICATION_CHANNELS (comma-separated)."
        )
