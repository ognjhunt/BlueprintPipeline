import json
import sys
import types


def test_log_outputs_json_with_run_id(tmp_path, capsys) -> None:
    dummy_checkpoint = types.ModuleType("tools.checkpoint")
    dummy_checkpoint.__path__ = []
    dummy_checkpoint.get_checkpoint_store = lambda *args, **kwargs: None
    dummy_hash_config = types.ModuleType("tools.checkpoint.hash_config")
    dummy_hash_config.resolve_checkpoint_hash_setting = lambda: False
    sys.modules["tools.checkpoint"] = dummy_checkpoint
    sys.modules["tools.checkpoint.hash_config"] = dummy_hash_config

    from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep

    runner = LocalPipelineRunner(
        scene_dir=tmp_path,
        verbose=True,
        json_logging=True,
    )
    runner._current_step = PipelineStep.REGEN3D

    runner.log("hello world", "INFO")

    captured = capsys.readouterr().out.strip().splitlines()
    json_line = next(line for line in reversed(captured) if line.strip().startswith("{"))
    payload = json.loads(json_line)
    assert payload["run_id"] == runner.run_id


def test_log_outputs_text_with_run_id(tmp_path, capsys) -> None:
    dummy_checkpoint = types.ModuleType("tools.checkpoint")
    dummy_checkpoint.__path__ = []
    dummy_checkpoint.get_checkpoint_store = lambda *args, **kwargs: None
    dummy_hash_config = types.ModuleType("tools.checkpoint.hash_config")
    dummy_hash_config.resolve_checkpoint_hash_setting = lambda: False
    sys.modules["tools.checkpoint"] = dummy_checkpoint
    sys.modules["tools.checkpoint.hash_config"] = dummy_hash_config

    from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep

    runner = LocalPipelineRunner(
        scene_dir=tmp_path,
        verbose=True,
        json_logging=False,
    )
    runner._current_step = PipelineStep.REGEN3D

    runner.log("hello world", "INFO")

    captured = capsys.readouterr().out.strip()
    assert f"[run_id={runner.run_id}]" in captured
