from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_variation_gen_module():
    repo_root = Path(__file__).resolve().parents[1]
    job_dir = repo_root / "variation-gen-job"
    if str(job_dir) not in sys.path:
        sys.path.insert(0, str(job_dir))
    import generate_variation_assets

    return generate_variation_assets


def test_variation_gen_job_dry_run(tmp_path, monkeypatch):
    variation_gen = _load_variation_gen_module()

    repo_root = Path(__file__).resolve().parents[1]
    fixture_path = repo_root / "tests" / "fixtures" / "variation_gen" / "manifest.json"
    manifest_payload = json.loads(fixture_path.read_text())

    replicator_manifest_dir = tmp_path / "replicator" / "variation_assets"
    replicator_manifest_dir.mkdir(parents=True, exist_ok=True)
    (replicator_manifest_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2)
    )

    monkeypatch.setenv("VARIATION_GEN_MODE", "mock")

    outcome = variation_gen.process_variation_assets(
        root=tmp_path,
        scene_id="test_scene",
        replicator_prefix="replicator",
        variation_assets_prefix="variation_assets",
        dry_run=True,
    )

    assert outcome.success is True

    asset_dir = tmp_path / "variation_assets" / "spatula_01"
    reference_image = asset_dir / "reference.png"
    assert reference_image.is_file()

    manifest_with_results = tmp_path / "variation_assets" / "manifest_with_results.json"
    generation_report = tmp_path / "variation_assets" / "generation_report.json"
    completion_marker = tmp_path / "variation_assets" / ".variation_pipeline_complete"

    assert manifest_with_results.is_file()
    assert generation_report.is_file()
    assert completion_marker.is_file()

    updated_manifest = json.loads(manifest_with_results.read_text())
    asset_entry = next(
        asset for asset in updated_manifest["assets"] if asset["name"] == "spatula_01"
    )

    assert asset_entry["generation_status"] == "success"
    assert asset_entry["reference_image_path"] == str(reference_image)
