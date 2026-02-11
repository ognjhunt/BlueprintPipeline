from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def test_replicator_bundle_job_end_to_end(
    tmp_path: Path,
    load_job_module,
    repo_root: Path,
    monkeypatch,
) -> None:
    module = load_job_module("replicator", "generate_replicator_bundle.py")

    inventory = {
        "scene_id": "test_scene",
        "scene_type": "kitchen",
        "objects": [
            {"id": "counter_01", "category": "counter", "sim_role": "static"},
            {"id": "sink_01", "category": "sink", "sim_role": "static"},
            {"id": "dishwasher_01", "category": "dishwasher", "sim_role": "static"},
            {"id": "plate_01", "category": "dish", "sim_role": "manipulable_object"},
            {"id": "mug_01", "category": "mug", "sim_role": "manipulable_object"},
        ],
    }
    _write_json(tmp_path / "seg" / "inventory.json", inventory)

    manifest = {
        "version": "1.0.0",
        "scene_id": "test_scene",
        "scene": {
            "environment_type": "kitchen",
            "coordinate_frame": "y_up",
            "meters_per_unit": 1.0,
            "room": {
                "bounds": {"width": 4.0, "depth": 5.0, "height": 2.8},
            },
        },
        "objects": [
            {"id": "counter_01", "category": "counter", "sim_role": "static",
             "asset": {"path": "objects/counter_01/asset.usdz", "source": "blueprintpipeline"},
             "transform": {"position": {"x": 2.5, "y": 0.0, "z": 0.9},
                           "rotation_euler": {"roll": 0, "pitch": 0, "yaw": 0},
                           "scale": {"x": 1, "y": 1, "z": 1}}},
            {"id": "sink_01", "category": "sink", "sim_role": "static",
             "asset": {"path": "objects/sink_01/asset.usdz", "source": "blueprintpipeline"},
             "transform": {"position": {"x": 2.0, "y": 0.5, "z": 0.9},
                           "rotation_euler": {"roll": 0, "pitch": 0, "yaw": 0},
                           "scale": {"x": 1, "y": 1, "z": 1}}},
            {"id": "dishwasher_01", "category": "dishwasher", "sim_role": "articulated_appliance",
             "articulation": {"required": True, "type": "revolute"},
             "asset": {"path": "objects/dishwasher_01/asset.usdz", "source": "blueprintpipeline"},
             "transform": {"position": {"x": 2.6, "y": 0.2, "z": 0.9},
                           "rotation_euler": {"roll": 0, "pitch": 0, "yaw": 0},
                           "scale": {"x": 1, "y": 1, "z": 1}}},
            {"id": "plate_01", "category": "dish", "sim_role": "manipulable_object",
             "asset": {"path": "objects/plate_01/asset.usdz", "source": "blueprintpipeline"},
             "transform": {"position": {"x": 2.3, "y": 0.1, "z": 0.9},
                           "rotation_euler": {"roll": 0, "pitch": 0, "yaw": 0},
                           "scale": {"x": 1, "y": 1, "z": 1}}},
            {"id": "mug_01", "category": "mug", "sim_role": "manipulable_object",
             "asset": {"path": "objects/mug_01/asset.usdz", "source": "blueprintpipeline"},
             "transform": {"position": {"x": 2.4, "y": -0.1, "z": 0.9},
                           "rotation_euler": {"roll": 0, "pitch": 0, "yaw": 0},
                           "scale": {"x": 1, "y": 1, "z": 1}}},
        ],
    }
    _write_json(tmp_path / "assets" / "scene_manifest.json", manifest)

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv(
        "LLM_MOCK_RESPONSE_PATH",
        str(repo_root / "tests" / "fixtures" / "replicator" / "mock_analysis.json"),
    )

    module.generate_replicator_bundle_job(
        bucket="local",
        scene_id="test_scene",
        seg_prefix="seg",
        assets_prefix="assets",
        usd_prefix="usd",
        replicator_prefix="replicator",
        root=tmp_path,
    )

    output_dir = tmp_path / "replicator"
    placement_regions = list((output_dir / "placement_regions").glob("*.usda"))
    assert placement_regions, "Expected placement_regions/*.usda output"

    policies_dir = output_dir / "policies"
    configs_dir = output_dir / "configs"
    for policy_id in ("dish_loading", "table_clearing"):
        assert (policies_dir / f"{policy_id}.py").is_file()
        assert (configs_dir / f"{policy_id}.json").is_file()

    manifest_path = output_dir / "variation_assets" / "manifest.json"
    assert manifest_path.is_file()

    manifest_data = json.loads(manifest_path.read_text())
    asset_names = {asset["name"] for asset in manifest_data["assets"]}
    assert {"dirty_plate", "dirty_mug"}.issubset(asset_names)

    affordance_graph_path = output_dir / "affordance_graph.json"
    assert affordance_graph_path.is_file()
    affordance_graph = json.loads(affordance_graph_path.read_text())
    assert affordance_graph["scene_id"] == "test_scene"
    assert affordance_graph["regions"]
    assert "dish_loading" in affordance_graph["policy_region_map"]
    articulation_target_ids = {
        target["object_id"] for target in affordance_graph.get("articulation_targets", [])
    }
    assert "dishwasher_01" in articulation_target_ids

    marker_path = output_dir / ".replicator_complete"
    assert marker_path.is_file()
    marker = json.loads(marker_path.read_text())
    assert marker["status"] == "completed"
    assert marker["scene_id"] == "test_scene"

    placement_text = (output_dir / "placement_regions" / "placement_regions.usda").read_text()
    assert "countertop_region" in placement_text
    assert "sink_region" in placement_text
