import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "episode-generation-job"))

from generate_episodes import ManipulationTaskGenerator


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "episode-generation-job" / "sample_manifests" / "multi_object_stack_manifest.json"


def test_multi_object_task_generation_orders_steps():
    manifest = json.loads(FIXTURE_PATH.read_text())
    generator = ManipulationTaskGenerator(use_llm=False, verbose=False)

    tasks_with_specs = generator.generate_tasks_with_specs(
        manifest=manifest,
        manifest_path=None,
        robot_type="franka",
    )

    multi_object_tasks = [task for task, _ in tasks_with_specs if task.get("task_id") == "stack_blocks"]
    assert multi_object_tasks, "Expected stack_blocks task from sample manifest"

    task, spec = next((t, s) for t, s in tasks_with_specs if t.get("task_id") == "stack_blocks")
    steps = task.get("task_steps", [])
    assert [step["step_id"] for step in steps] == [
        "stack_red_on_blue",
        "stack_green_on_red",
    ]

    assert spec.success_criteria.get("ordered_execution_required") is True
    assert spec.success_criteria.get("step_dependencies") == {
        "stack_red_on_blue": [],
        "stack_green_on_red": ["stack_red_on_blue"],
    }
    assert len(spec.segments) == len(steps)
    assert spec.segments[0].start_time == 0.0
    assert spec.segments[1].start_time > spec.segments[0].start_time


def test_multi_object_dependency_cycle_detected():
    generator = ManipulationTaskGenerator(use_llm=False, verbose=False)
    task = {
        "task_id": "cycle_task",
        "task_name": "stack_objects",
        "description": "Cyclic dependency task",
        "task_steps": [
            {"step_id": "a", "target_object_id": "obj1", "depends_on": ["b"]},
            {"step_id": "b", "target_object_id": "obj2", "depends_on": ["a"]},
        ],
    }
    object_lookup = {
        "obj1": {"id": "obj1"},
        "obj2": {"id": "obj2"},
    }

    is_valid, ordered_steps, errors = generator._validate_task_steps(task, object_lookup)

    assert not is_valid
    assert ordered_steps is None
    assert any("cyclic" in error for error in errors)
