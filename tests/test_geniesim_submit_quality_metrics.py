import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_submit_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "genie-sim-submit-job" / "submit_to_geniesim.py"
    spec = importlib.util.spec_from_file_location("submit_to_geniesim", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load submit_to_geniesim module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aggregate_quality_metrics_handles_invalid_types():
    submit_module = _load_submit_module()
    results = {
        "franka": SimpleNamespace(
            episodes_collected="5",
            collision_free_episodes="3",
            collision_info_episodes="4",
            task_success_episodes=None,
            task_success_info_episodes="2",
            collision_free_rate="0.75",
            task_success_rate="not-a-number",
        ),
        "panda": SimpleNamespace(
            episodes_collected={"bad": 1},
            collision_free_episodes=2.4,
            collision_info_episodes="",
            task_success_episodes="3.5",
            task_success_info_episodes="0",
            collision_free_rate={"bad": True},
            task_success_rate=0.5,
        ),
    }

    metrics = submit_module._aggregate_quality_metrics(results)

    assert metrics["episodes_collected"] == 5
    assert metrics["collision_free_episodes"] == 3
    assert metrics["collision_info_episodes"] == 4
    assert metrics["task_success_episodes"] == 0
    assert metrics["task_success_info_episodes"] == 2
    assert metrics["collision_free_rate"] == 0.75
    assert metrics["task_success_rate"] == 0.0

    by_robot = metrics["by_robot"]
    assert by_robot["franka"]["collision_free_rate"] == 0.75
    assert by_robot["franka"]["task_success_rate"] is None
    assert by_robot["panda"]["collision_free_rate"] is None
    assert by_robot["panda"]["task_success_rate"] == 0.5
