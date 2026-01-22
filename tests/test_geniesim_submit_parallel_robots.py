import importlib.util
from pathlib import Path


def _load_submit_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "genie-sim-submit-job" / "submit_to_geniesim.py"
    spec = importlib.util.spec_from_file_location("submit_to_geniesim", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load submit_to_geniesim module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parallel_robot_runs_create_unique_outputs_and_statuses(tmp_path, monkeypatch):
    submit_module = _load_submit_module()

    monkeypatch.setenv("GENIESIM_PARALLEL_ROBOTS", "true")
    monkeypatch.setenv("GENIESIM_MAX_PARALLEL_ROBOTS", "2")
    monkeypatch.setenv("GENIESIM_PARALLEL_USE_PROCESSES", "false")

    calls = []

    def fake_run(
        *,
        scene_manifest,
        task_config,
        output_dir,
        robot_type,
        episodes_per_task,
        expected_server_version,
        required_capabilities,
        verbose=False,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{robot_type}.txt").write_text("ok")
        calls.append((robot_type, output_dir))
        result = submit_module.DataCollectionResult(
            success=True,
            task_name=task_config.get("name", "task"),
        )
        result.episodes_collected = episodes_per_task
        result.episodes_passed = episodes_per_task
        result.server_info = {"version": expected_server_version}
        return result

    monkeypatch.setattr(
        submit_module,
        "_run_local_data_collection_with_handshake",
        fake_run,
    )

    robot_types = ["franka", "panda"]
    scene_manifest = {"scene_graph": {}}
    task_config = {"name": "test-task", "tasks": []}

    (
        local_run_results,
        local_run_ends,
        server_info_by_robot,
        output_dirs,
    ) = submit_module._run_robot_local_collections(
        robot_types=robot_types,
        scene_manifest=scene_manifest,
        task_config=task_config,
        episodes_per_task=2,
        local_root=tmp_path,
        episodes_output_prefix="scenes/demo/episodes",
        job_id="job-123",
        debug_mode=False,
        expected_server_version=submit_module.EXPECTED_GENIESIM_SERVER_VERSION,
        required_capabilities=submit_module.REQUIRED_GENIESIM_CAPABILITIES,
        parallel_enabled=True,
        max_parallel=2,
        use_processes=False,
    )

    assert set(local_run_results.keys()) == set(robot_types)
    assert all(local_run_results[robot].success for robot in robot_types)
    assert all(local_run_ends[robot] is not None for robot in robot_types)
    assert len(set(output_dirs.values())) == len(robot_types)
    assert set(output_dirs.keys()) == set(robot_types)
    assert all((output_dirs[robot] / f"{robot}.txt").exists() for robot in robot_types)
    assert set(server_info_by_robot.keys()) == set(robot_types)
    assert len(calls) == len(robot_types)

    local_execution_by_robot = submit_module._build_local_execution_by_robot(
        robot_types=robot_types,
        local_run_results=local_run_results,
        output_dirs=output_dirs,
        server_info_by_robot=server_info_by_robot,
    )

    assert list(local_execution_by_robot.keys()) == robot_types
    for robot in robot_types:
        assert local_execution_by_robot[robot]["success"] is True
        assert local_execution_by_robot[robot]["output_dir"] == str(output_dirs[robot])
