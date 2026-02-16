import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@pytest.mark.unit
def test_validate_stage7_contract_pass(tmp_path, monkeypatch):
    from scripts.runpod_sage import validate_stage7_contract as v
    h5py = pytest.importorskip("h5py")

    layout_dir = tmp_path / "layout_1"
    (layout_dir / "generation").mkdir(parents=True, exist_ok=True)
    (layout_dir / "usd_cache").mkdir(parents=True, exist_ok=True)
    demos_dir = layout_dir / "demos"
    (demos_dir / "videos").mkdir(parents=True, exist_ok=True)
    plans_dir = layout_dir / "plans"
    quality_dir = layout_dir / "quality"
    plans_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(demos_dir / "dataset.hdf5"), "w") as f:
        data = f.create_group("data")
        data.create_group("demo_0")

    (demos_dir / "scene_variant_000.usd").write_text("#usda 1.0\n", encoding="utf-8")
    (demos_dir / "videos" / "demo_0.mp4").write_bytes(b"mp4")

    run_id = "run_abc"
    _write_json(plans_dir / "plan_bundle.json", {"run_id": run_id})
    _write_json(demos_dir / "demo_metadata.json", {"run_id": run_id})
    _write_json(demos_dir / "quality_report.json", {"run_id": run_id, "status": "pass"})
    _write_json(demos_dir / "artifact_manifest.json", {"run_id": run_id, "status": "ok"})

    report_path = layout_dir / "quality" / "contract_report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_stage7_contract.py",
            "--layout-dir",
            str(layout_dir),
            "--run-id",
            run_id,
            "--expected-demos",
            "1",
            "--strict-artifact-contract",
            "1",
            "--strict-provenance",
            "1",
            "--report-path",
            str(report_path),
        ],
    )
    assert v.main() == 0
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "pass"


@pytest.mark.unit
def test_aggregate_run_quality_fail_on_missing_reports(tmp_path, monkeypatch):
    from scripts.runpod_sage import aggregate_run_quality as agg

    layout_dir = tmp_path / "layout_2"
    layout_dir.mkdir(parents=True, exist_ok=True)
    out_path = layout_dir / "quality" / "run_quality_summary.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "aggregate_run_quality.py",
            "--layout-dir",
            str(layout_dir),
            "--run-id",
            "run_xyz",
            "--output-path",
            str(out_path),
        ],
    )
    assert agg.main() == 3
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
