from __future__ import annotations

from pathlib import Path

import pytest

from tools.validation import geniesim_export as ge


pytestmark = pytest.mark.usefixtures("add_repo_to_path")


@pytest.mark.parametrize(
    "bounds, expected_min, expected_max",
    [
        ({"width": 2.0, "depth": 4.0, "height": 3.0}, [-1.0, -2.0, 0.0], [1.0, 2.0, 3.0]),
        (
            {"x": [-1.0, 1.0], "y": [-2.0, 2.0], "z": [0.0, 3.0]},
            [-1.0, -2.0, 0.0],
            [1.0, 2.0, 3.0],
        ),
        ([[-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]], [-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]),
    ],
)
def test_parse_workspace_bounds_formats(bounds, expected_min, expected_max) -> None:
    min_pt, max_pt = ge._parse_workspace_bounds(
        bounds,
        context="task config",
        path=Path("task.json"),
    )
    assert min_pt == expected_min
    assert max_pt == expected_max


def test_parse_workspace_bounds_invalid_format_raises() -> None:
    with pytest.raises(ge.ExportConsistencyError, match="Invalid workspace_bounds"):
        ge._parse_workspace_bounds(
            {"x": [0.0, 1.0]},
            context="task config",
            path=Path("task.json"),
        )


def test_extract_helpers() -> None:
    entry = {
        "target_object_id": "obj_1",
        "goal_object": "obj_2",
        "objects": ["obj_3", {"object_id": "obj_4"}, {"id": "obj_5"}],
        "target_objects": [{"asset_id": "obj_6"}],
    }
    assert ge._extract_object_ids_from_entry(entry) == {
        "obj_1",
        "obj_2",
        "obj_3",
        "obj_4",
        "obj_5",
        "obj_6",
    }

    task = {
        "target_position": [0.0, 1.0, 2.0],
        "goal_positions": [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        "target_objects": [{"position": [3.0, 3.0, 3.0]}, {"position": "bad"}],
    }
    positions = ge._extract_positions(task)
    assert ("target_position", [0.0, 1.0, 2.0]) in positions
    assert ("goal_positions[0]", [1.0, 1.0, 1.0]) in positions
    assert ("target_objects[0].position", [3.0, 3.0, 3.0]) in positions

    assert ge._namespaced_asset_id("scene_1", "asset") == "scene_1_obj_asset"
    assert ge._namespaced_asset_id(None, "asset") == "asset"


def test_validate_export_consistency_missing_scene_assets() -> None:
    scene_graph = {"scene_id": "scene_1", "nodes": [{"asset_id": "scene_1_obj_missing"}]}
    asset_index = {"assets": [{"asset_id": "scene_1_obj_present"}]}
    task_config = {}

    with pytest.raises(ge.ExportConsistencyError, match="missing asset_ids"):
        ge.validate_export_consistency_data(
            scene_graph=scene_graph,
            asset_index=asset_index,
            task_config=task_config,
        )


def test_validate_export_consistency_missing_task_assets() -> None:
    scene_graph = {"scene_id": "scene_1", "nodes": [{"asset_id": "scene_1_obj_asset"}]}
    asset_index = {"assets": [{"asset_id": "scene_1_obj_asset"}]}
    task_config = {"tasks": [{"target_object_id": "missing"}]}

    with pytest.raises(ge.ExportConsistencyError, match="missing object ids"):
        ge.validate_export_consistency_data(
            scene_graph=scene_graph,
            asset_index=asset_index,
            task_config=task_config,
        )


def test_validate_export_consistency_out_of_bounds() -> None:
    scene_graph = {"scene_id": "scene_1", "nodes": [{"asset_id": "scene_1_obj_asset"}]}
    asset_index = {"assets": [{"asset_id": "scene_1_obj_asset"}]}
    task_config = {
        "tasks": [{"task_type": "pick", "target_object_id": "asset", "target_position": [5.0, 0.0, 0.0]}],
        "workspace_bounds": {"width": 2.0, "depth": 2.0, "height": 2.0},
    }

    with pytest.raises(ge.ExportConsistencyError, match="outside workspace bounds"):
        ge.validate_export_consistency_data(
            scene_graph=scene_graph,
            asset_index=asset_index,
            task_config=task_config,
        )


def test_validate_export_consistency_success() -> None:
    scene_graph = {"scene_id": "scene_1", "nodes": [{"asset_id": "scene_1_obj_asset"}]}
    asset_index = {"assets": [{"asset_id": "scene_1_obj_asset"}]}
    task_config = {
        "tasks": [
            {"task_type": "pick", "target_object_id": "asset", "target_position": [0.0, 0.0, 0.0]}
        ],
        "workspace_bounds": {"width": 2.0, "depth": 2.0, "height": 2.0},
    }

    ge.validate_export_consistency_data(
        scene_graph=scene_graph,
        asset_index=asset_index,
        task_config=task_config,
    )


def test_validate_export_consistency_file_errors(tmp_path: Path) -> None:
    scene_graph_path = tmp_path / "scene_graph.json"
    asset_index_path = tmp_path / "asset_index.json"
    task_config_path = tmp_path / "task_config.json"

    scene_graph_path.write_text("{}")
    asset_index_path.write_text("{}")

    with pytest.raises(ge.ExportConsistencyError, match="missing at"):
        ge.validate_export_consistency(
            scene_graph_path=scene_graph_path,
            asset_index_path=asset_index_path,
            task_config_path=task_config_path,
        )

    task_config_path.write_text("{not-json}")
    with pytest.raises(ge.ExportConsistencyError, match="Failed to parse"):
        ge.validate_export_consistency(
            scene_graph_path=scene_graph_path,
            asset_index_path=asset_index_path,
            task_config_path=task_config_path,
        )
