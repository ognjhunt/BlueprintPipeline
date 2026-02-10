import json

from tools.quality_gates.quality_gate import QualityGateCheckpoint, QualityGateRegistry


def test_quality_gate_report_includes_checkpoint_coverage() -> None:
    registry = QualityGateRegistry(verbose=False)
    registry.register_required_checkpoint(QualityGateCheckpoint.SCALE_COMPLETE)
    registry.register_skipped_checkpoint(
        QualityGateCheckpoint.SCALE_COMPLETE,
        "scale report missing during checkpoint replay",
    )

    report = registry.to_report("scene-1")

    assert report["report_version"] == "2.0"
    assert "scale_complete" in report["required_checkpoints"]
    assert "scale_complete" in report["skipped_checkpoints"]
    assert report["skip_reasons"]["scale_complete"] == "scale report missing during checkpoint replay"
    assert report["summary"]["required_checkpoints"] == len(report["required_checkpoints"])
    assert report["summary"]["skipped_checkpoints"] == len(report["skipped_checkpoints"])


def test_run_checkpoint_marks_executed_checkpoint(tmp_path) -> None:
    registry = QualityGateRegistry(verbose=False)
    assets_path = tmp_path / "variation_assets.json"
    assets_path.write_text(
        json.dumps(
            {
                "objects": [
                    {"id": "obj-1", "generated_3d": {"status": "success"}},
                ]
            }
        )
    )

    registry.run_checkpoint(
        checkpoint=QualityGateCheckpoint.VARIATION_GEN_COMPLETE,
        context={"scene_id": "scene-1", "variation_assets_path": str(assets_path)},
    )
    report = registry.to_report("scene-1")

    assert "variation_gen_complete" in report["required_checkpoints"]
    assert "variation_gen_complete" in report["executed_checkpoints"]
    assert "variation_gen_complete" not in report["skipped_checkpoints"]


def test_variation_gate_fails_when_generated_3d_entries_missing(tmp_path) -> None:
    registry = QualityGateRegistry(verbose=False)
    assets_path = tmp_path / "variation_assets.json"
    assets_path.write_text(json.dumps({"objects": [{"id": "obj-1"}]}))

    results = registry.run_checkpoint(
        checkpoint=QualityGateCheckpoint.VARIATION_GEN_COMPLETE,
        context={"scene_id": "scene-1", "variation_assets_path": str(assets_path)},
    )
    variation_gate = next(result for result in results if result.gate_id == "qg-15-variation-assets")

    assert variation_gate.passed is False
    assert "No generated_3d entries found in variation assets" in variation_gate.details["issues"]
