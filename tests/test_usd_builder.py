import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from blueprint_sim.recipe_compiler.layer_manager import LayerManager
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


def test_articulation_metadata_written_to_usd(tmp_path) -> None:
    usd_physics = pytest.importorskip("pxr.UsdPhysics")

    builder = USDSceneBuilder()
    manager = LayerManager(str(tmp_path))
    layer = manager.create_layer("physics_overrides", tmp_path / "physics_overrides.usda")

    objects = [
        {
            "id": "cabinet_001",
            "physics": {"enabled": True},
            "articulation": {
                "type": "revolute",
                "axis": "z",
                "limits": {"lower": 0.0, "upper": 1.2},
                "damping": 0.05,
                "stiffness": 25.0,
                "friction": 0.2,
                "velocity_limit": 4.5,
                "effort_limit": 18.0,
            },
        }
    ]

    builder.build_physics_overrides(layer, objects)

    stage = layer.stage
    joint_path = "/Objects/cabinet_001/joint"
    joint_prim = stage.GetPrimAtPath(joint_path)
    assert joint_prim.IsValid()

    drive = usd_physics.DriveAPI.Get(stage, joint_path, "angular")
    assert drive.GetStiffnessAttr().Get() == pytest.approx(25.0)
    assert drive.GetMaxForceAttr().Get() == pytest.approx(18.0)

    friction_attr = joint_prim.GetAttribute("physxJoint:friction")
    assert friction_attr.Get() == pytest.approx(0.2)

    velocity_attr = joint_prim.GetAttribute("physxJoint:velocityLimit")
    assert velocity_attr.Get() == pytest.approx(4.5)
