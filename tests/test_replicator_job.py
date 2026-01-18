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
            {"id": "counter_01", "category": "counter", "sim_role": "static_object"},
            {"id": "sink_01", "category": "sink", "sim_role": "static_object"},
            {"id": "dishwasher_01", "category": "dishwasher", "sim_role": "static_object"},
            {"id": "plate_01", "category": "dish", "sim_role": "manipulable_object"},
            {"id": "mug_01", "category": "mug", "sim_role": "manipulable_object"},
        ],
    }
    _write_json(tmp_path / "seg" / "inventory.json", inventory)

    manifest = {
        "version": "1.0.0",
        "scene_id": "test_scene",
        "objects": [
            {"id": "counter_01", "category": "counter", "sim_role": "static_object"},
            {"id": "sink_01", "category": "sink", "sim_role": "static_object"},
            {"id": "plate_01", "category": "dish", "sim_role": "manipulable_object"},
            {"id": "mug_01", "category": "mug", "sim_role": "manipulable_object"},
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

    placement_text = (output_dir / "placement_regions" / "placement_regions.usda").read_text()
    assert "countertop_region" in placement_text
    assert "sink_region" in placement_text
