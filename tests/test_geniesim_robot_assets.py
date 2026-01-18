"""Validation tests for Genie Sim robot asset paths."""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.geniesim_adapter.multi_robot_config import (
    DEFAULT_MULTI_ROBOT_CONFIG,
    ROBOT_ASSETS_DIR,
    ROBOT_SPECS,
    resolve_robot_asset_path,
)


def test_default_robot_assets_exist() -> None:
    """Ensure default multi-robot config resolves to existing assets."""
    assert ROBOT_ASSETS_DIR.is_dir()

    for robot_type in DEFAULT_MULTI_ROBOT_CONFIG.get_all_robots():
        spec = ROBOT_SPECS[robot_type]
        for asset_path in (spec.urdf_path, spec.usd_path):
            resolved = resolve_robot_asset_path(asset_path)
            if resolved is None:
                continue
            assert resolved.is_file(), f"Missing asset for {robot_type.value}: {resolved}"
            assert resolved.is_relative_to(Path(ROBOT_ASSETS_DIR).resolve())
