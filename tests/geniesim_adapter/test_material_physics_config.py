from pathlib import Path

import pytest

from tools.geniesim_adapter import asset_index


def test_material_physics_config_warns_when_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(asset_index, "resolve_production_mode", lambda: False)
    missing_path = tmp_path / "material_physics.json"

    with pytest.warns(UserWarning, match=str(missing_path)):
        assert asset_index._load_material_physics_overrides(missing_path) == {}


def test_material_physics_config_errors_in_production(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(asset_index, "resolve_production_mode", lambda: True)
    missing_path = tmp_path / "material_physics.json"

    with pytest.raises(FileNotFoundError, match=str(missing_path)):
        asset_index._load_material_physics_overrides(missing_path)
