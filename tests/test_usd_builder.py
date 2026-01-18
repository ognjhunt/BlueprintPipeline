import pytest

from blueprint_sim.recipe_compiler.usd_builder import USDSceneBuilder


def test_validate_catalog_path_raises_when_catalog_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(USDSceneBuilder, "_load_asset_catalog", lambda self: {})

    builder = USDSceneBuilder()

    with pytest.raises(RuntimeError, match="Asset catalog is not loaded; cannot validate asset paths"):
        builder._validate_catalog_path("missing.usd")


def test_validate_catalog_path_raises_when_asset_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = {"pack_info": {"name": "starter"}, "assets": [{"relative_path": "exists.usd"}]}
    monkeypatch.setattr(USDSceneBuilder, "_load_asset_catalog", lambda self: catalog)

    builder = USDSceneBuilder()

    with pytest.raises(ValueError, match="Asset path 'missing.usd' not found in catalog"):
        builder._validate_catalog_path("missing.usd")
