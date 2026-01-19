"""Tests for smart placement engine job modules."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def smart_placement_modules(repo_root: Path):
    """Load smart placement engine modules with a package alias."""
    job_dir = repo_root / "smart-placement-engine-job"
    package_name = "smart_placement_engine_job"

    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            job_dir / "__init__.py",
            submodule_search_locations=[str(job_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load smart placement package")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)

    compatibility = importlib.import_module(f"{package_name}.compatibility_matrix")
    region_detector = importlib.import_module(
        f"{package_name}.intelligent_region_detector"
    )
    placement_engine = importlib.import_module(f"{package_name}.placement_engine")
    physics_validator = importlib.import_module(f"{package_name}.physics_validator")

    sys.modules.setdefault("compatibility_matrix", compatibility)
    sys.modules.setdefault("intelligent_region_detector", region_detector)
    sys.modules.setdefault("placement_engine", placement_engine)
    sys.modules.setdefault("physics_validator", physics_validator)

    run_spec = importlib.util.spec_from_file_location(
        f"{package_name}.run_smart_placement",
        job_dir / "run_smart_placement.py",
    )
    if run_spec is None or run_spec.loader is None:
        raise ImportError("Unable to load run_smart_placement module")
    run_module = importlib.util.module_from_spec(run_spec)
    sys.modules[run_spec.name] = run_module
    run_spec.loader.exec_module(run_module)

    return {
        "compatibility": compatibility,
        "region_detector": region_detector,
        "placement_engine": placement_engine,
        "physics_validator": physics_validator,
        "run_smart_placement": run_module,
    }


@pytest.mark.unit
def test_load_assets_from_minimal_manifest(smart_placement_modules):
    """Ensure asset loading handles minimal manifests."""
    run_module = smart_placement_modules["run_smart_placement"]
    compatibility = smart_placement_modules["compatibility"]

    manifest = {
        "assets": [
            {
                "id": "asset-1",
                "name": "Simple Asset",
            }
        ]
    }

    assets = run_module.load_assets_from_manifest(manifest)

    assert len(assets) == 1
    asset = assets[0]
    assert asset.asset_id == "asset-1"
    assert asset.asset_name == "Simple Asset"
    assert asset.category == compatibility.AssetCategory.MISC_OBJECTS
    assert asset.bounding_box.size == [0.1, 0.1, 0.1]
    assert asset.mass_kg == 0.5


@pytest.mark.unit
def test_plan_placements_is_deterministic(smart_placement_modules):
    """Verify placement plans are deterministic with fixed inputs."""
    compatibility = smart_placement_modules["compatibility"]
    region_detector = smart_placement_modules["region_detector"]
    placement_engine = smart_placement_modules["placement_engine"]

    engine = placement_engine.SmartPlacementEngine(
        randomize_placement=False,
        enable_stacking=False,
    )

    asset = placement_engine.AssetInstance(
        asset_id="plate-1",
        asset_name="Plate",
        category=compatibility.AssetCategory.DISHES,
        bounding_box=placement_engine.BoundingBox(
            center=[0.0, 0.0, 0.0],
            size=[0.2, 0.2, 0.1],
        ),
    )

    region = region_detector.DetectedRegion(
        id="counter-1",
        name="Counter",
        region_type=compatibility.RegionType.COUNTER,
        position=[0.0, 0.0, 0.0],
        size=[1.0, 1.0, 0.2],
    )

    plan = engine.plan_placements(
        assets=[asset],
        regions=[region],
        scene_id="scene-1",
        scene_archetype=compatibility.SceneArchetype.KITCHEN,
    )

    assert plan.scene_id == "scene-1"
    assert plan.total_assets_placed == 1
    assert plan.placements[0].status == placement_engine.PlacementStatus.SUCCESS
    assert plan.placements[0].final_position == pytest.approx([0.04, 0.04, 0.15])


@pytest.mark.unit
def test_physics_validator_valid_and_invalid(smart_placement_modules):
    """Check physics validator outputs for valid vs. invalid placements."""
    compatibility = smart_placement_modules["compatibility"]
    region_detector = smart_placement_modules["region_detector"]
    placement_engine = smart_placement_modules["placement_engine"]
    physics_validator = smart_placement_modules["physics_validator"]

    validator = physics_validator.PhysicsValidator()

    region = region_detector.DetectedRegion(
        id="table-1",
        name="Table",
        region_type=compatibility.RegionType.TABLE,
        position=[0.0, 0.0, 0.0],
        size=[1.0, 1.0, 0.2],
    )

    stable_asset = placement_engine.AssetInstance(
        asset_id="stable-1",
        asset_name="Stable Asset",
        category=compatibility.AssetCategory.DISHES,
        bounding_box=placement_engine.BoundingBox(
            center=[0.0, 0.0, 0.0],
            size=[0.6, 0.6, 0.2],
        ),
    )
    stable_result = placement_engine.PlacementResult(
        asset=stable_asset,
        status=placement_engine.PlacementStatus.SUCCESS,
        final_position=[0.0, 0.0, 0.1],
        final_rotation=[0.0, 0.0, 0.0],
        region_id=region.id,
    )

    unstable_asset = placement_engine.AssetInstance(
        asset_id="unstable-1",
        asset_name="Unstable Asset",
        category=compatibility.AssetCategory.DISHES,
        bounding_box=placement_engine.BoundingBox(
            center=[0.0, 0.0, 0.0],
            size=[0.1, 0.1, 1.0],
        ),
    )
    unstable_result = placement_engine.PlacementResult(
        asset=unstable_asset,
        status=placement_engine.PlacementStatus.SUCCESS,
        final_position=[0.0, 0.0, 0.5],
        final_rotation=[0.0, 0.0, 0.0],
        region_id=region.id,
    )

    stable_validation = validator.validate_placement(stable_result, region)
    unstable_validation = validator.validate_placement(unstable_result, region)

    assert stable_validation.status == physics_validator.ValidationStatus.VALID
    assert unstable_validation.status == physics_validator.ValidationStatus.WARNING
