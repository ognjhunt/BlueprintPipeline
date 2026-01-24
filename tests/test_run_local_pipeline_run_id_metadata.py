import json
import sys
import types
import uuid


def test_run_id_in_geniesim_job_metadata(tmp_path, monkeypatch) -> None:
    dummy_adapter = types.ModuleType("tools.geniesim_adapter")
    dummy_local_framework = types.ModuleType("tools.geniesim_adapter.local_framework")

    class DummyResult:
        success = True
        episodes_collected = 1
        episodes_passed = 1
        timed_out = False

    def run_geniesim_preflight(*args, **kwargs) -> dict:
        return {"ok": True, "status": {"server_running": True}}

    def run_local_data_collection(*args, **kwargs) -> DummyResult:
        return DummyResult()

    def format_geniesim_preflight_failure(*args, **kwargs) -> str:
        return "preflight failed"

    dummy_local_framework.run_geniesim_preflight = run_geniesim_preflight
    dummy_local_framework.run_local_data_collection = run_local_data_collection
    dummy_local_framework.format_geniesim_preflight_failure = format_geniesim_preflight_failure
    dummy_adapter.local_framework = dummy_local_framework

    monkeypatch.setitem(sys.modules, "tools.geniesim_adapter", dummy_adapter)
    monkeypatch.setitem(sys.modules, "tools.geniesim_adapter.local_framework", dummy_local_framework)

    monkeypatch.delenv("BP_RUN_ID", raising=False)

    from tools.run_local_pipeline import LocalPipelineRunner

    scene_dir = tmp_path / "scene"
    geniesim_dir = scene_dir / "geniesim"
    geniesim_dir.mkdir(parents=True, exist_ok=True)

    (scene_dir / "episodes").mkdir(parents=True, exist_ok=True)

    (geniesim_dir / "scene_graph.json").write_text(json.dumps({"nodes": []}))
    (geniesim_dir / "asset_index.json").write_text(json.dumps({"assets": []}))
    (geniesim_dir / "task_config.json").write_text(json.dumps({"tasks": []}))

    runner = LocalPipelineRunner(scene_dir=scene_dir, verbose=False, json_logging=True)
    result = runner._run_geniesim_submit()

    assert result.outputs
    job_payload = json.loads((geniesim_dir / "job.json").read_text())
    assert job_payload["run_id"] == runner.run_id
    uuid.UUID(job_payload["run_id"])
