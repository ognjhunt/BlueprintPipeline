import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "episode-generation-job"))

from generate_episodes import ManipulationTaskGenerator, _normalize_manifest_objects  # noqa: E402


def test_task_generator_reads_transform_position_and_dimensions_est():
    manifest = {
        "scene_id": "scene_canonical_001",
        "environment_type": "hospital",
        "robot_config": {"workspace_center": [0.5, 0.0, 0.85]},
        "objects": [
            {
                "id": "obj_001",
                "name": "pill_bottle_001",
                "category": "bottle",
                "transform": {"position": {"x": 0.1, "y": -0.2, "z": 0.9}},
                "dimensions_est": {"width": 0.08, "height": 0.12, "depth": 0.08},
            }
        ],
    }

    generator = ManipulationTaskGenerator(use_llm=False, verbose=False)
    tasks_with_specs = generator.generate_tasks_with_specs(manifest=manifest, manifest_path=None, robot_type="franka")

    assert tasks_with_specs, "Expected at least one task to be generated"
    task, _spec = tasks_with_specs[0]
    assert task["target_object_id"] == "obj_001"
    assert task["target_position"] == [0.1, -0.2, 0.9]
    assert task["target_dimensions"] == [0.08, 0.08, 0.12]


def test_normalize_manifest_objects_adds_flat_position_and_dimensions():
    manifest = {
        "scene_id": "scene_norm_001",
        "objects": [
            {
                "id": "obj_001",
                "name": "box_001",
                "category": "box",
                "transform": {"position": {"x": 1.0, "y": 2.0, "z": 3.0}},
                "dimensions_est": {"width": 0.2, "height": 0.3, "depth": 0.4},
            }
        ],
    }

    _normalize_manifest_objects(manifest)
    obj = manifest["objects"][0]
    assert obj["position"] == [1.0, 2.0, 3.0]
    assert obj["dimensions"] == [0.2, 0.4, 0.3]

