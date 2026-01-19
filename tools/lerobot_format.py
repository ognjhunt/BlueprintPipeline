"""Shared LeRobot export format helpers."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union


class LeRobotExportFormat(str, Enum):
    """Supported LeRobot export formats."""

    LEROBOT_V2 = "lerobot_v2"
    LEROBOT_V3 = "lerobot_v3"
    LEROBOT_V0_3_3 = "lerobot_v0.3.3"


def parse_lerobot_export_format(
    value: Optional[Union[str, LeRobotExportFormat]],
    default: LeRobotExportFormat = LeRobotExportFormat.LEROBOT_V2,
) -> LeRobotExportFormat:
    """Parse a LeRobot export format value into the enum."""
    if value is None:
        return default
    if isinstance(value, LeRobotExportFormat):
        return value
    normalized = value.strip().lower()
    aliases = {
        "lerobot_v2": LeRobotExportFormat.LEROBOT_V2,
        "lerobot_v3": LeRobotExportFormat.LEROBOT_V3,
        "lerobot_v0.3.3": LeRobotExportFormat.LEROBOT_V0_3_3,
        "v2": LeRobotExportFormat.LEROBOT_V2,
        "v3": LeRobotExportFormat.LEROBOT_V3,
        "0.3.3": LeRobotExportFormat.LEROBOT_V0_3_3,
    }
    if normalized in aliases:
        return aliases[normalized]
    raise ValueError(f"Unknown LeRobot export format: {value}")
