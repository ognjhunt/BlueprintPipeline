from __future__ import annotations

import json
import logging
from pathlib import Path

from tools.cost_tracking.tracker import CostBreakdown


def test_cost_summary_included_when_cost_tracking_available(
    tmp_path: Path,
    load_job_module,
    monkeypatch,
) -> None:
    module = load_job_module("geniesim_import", "import_from_geniesim.py")

    breakdown = CostBreakdown()
    breakdown.total = 12.34
    breakdown.geniesim = 5.0
    breakdown.cloud_run = 2.5
    breakdown.cloud_build = 1.0
    breakdown.gcs_storage = 0.75
    breakdown.gcs_operations = 0.25
    breakdown.gemini = 2.0
    breakdown.other_apis = 0.84
    breakdown.by_job = {"genie-sim-import-job": 2.5}

    class FakeTracker:
        pricing_source = "COST_TRACKING_PRICING_JSON"

        def get_scene_cost(self, scene_id: str) -> CostBreakdown:
            assert scene_id == "scene-123"
            return breakdown

    monkeypatch.setattr(module, "get_cost_tracker", lambda: FakeTracker())

    log = logging.LoggerAdapter(module.logger, {"job_id": "test", "scene_id": "scene-123"})
    cost_summary = module._build_cost_summary("scene-123", log)

    assert cost_summary is not None
    assert cost_summary["total_usd"] == breakdown.total
    assert cost_summary["categories"]["geniesim"] == breakdown.geniesim
    assert cost_summary["categories"]["cloud_run"] == breakdown.cloud_run
    assert cost_summary["categories"]["gcs"] == breakdown.gcs_storage + breakdown.gcs_operations
    assert cost_summary["pricing_source"] == "COST_TRACKING_PRICING_JSON"

    dataset_info_path = tmp_path / "dataset_info.json"
    dataset_info_path.write_text(json.dumps({"dataset_type": "lerobot"}))
    dataset_info_payload = json.loads(dataset_info_path.read_text())

    module._update_dataset_info_cost_summary(
        cost_summary=cost_summary,
        dataset_info_payload=dataset_info_payload,
        dataset_info_path=dataset_info_path,
    )

    updated_dataset_info = json.loads(dataset_info_path.read_text())
    assert updated_dataset_info["cost_summary"] == cost_summary

    import_manifest = {"job_id": "job-1"}
    module._attach_cost_summary(import_manifest, cost_summary)
    assert import_manifest["cost_summary"] == cost_summary
