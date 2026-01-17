"""Helpers for Genie Sim mock-mode gating."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

from tools.config.production_mode import resolve_production_mode


@dataclass(frozen=True)
class MockModeDecision:
    enabled: bool
    requested: bool
    allow_override: bool
    production_mode: bool
    reason: str


def resolve_geniesim_mock_mode(env: Optional[Mapping[str, str]] = None) -> MockModeDecision:
    """Resolve whether Genie Sim mock mode is allowed."""
    env = env or os.environ
    requested = (env.get("GENIESIM_MOCK_MODE", "false") or "").strip().lower() == "true"
    allow_override = (env.get("ALLOW_GENIESIM_MOCK", "0") or "").strip() == "1"
    production_mode = resolve_production_mode(env)

    if not requested:
        return MockModeDecision(
            enabled=False,
            requested=False,
            allow_override=allow_override,
            production_mode=production_mode,
            reason="GENIESIM_MOCK_MODE not enabled",
        )
    if production_mode:
        return MockModeDecision(
            enabled=False,
            requested=True,
            allow_override=allow_override,
            production_mode=True,
            reason="Mock mode ignored because production mode is enabled",
        )
    if not allow_override:
        return MockModeDecision(
            enabled=False,
            requested=True,
            allow_override=False,
            production_mode=False,
            reason="Mock mode requires ALLOW_GENIESIM_MOCK=1",
        )

    return MockModeDecision(
        enabled=True,
        requested=True,
        allow_override=True,
        production_mode=False,
        reason="Mock mode enabled",
    )
