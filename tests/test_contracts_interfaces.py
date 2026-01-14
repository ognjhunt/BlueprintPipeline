#!/usr/bin/env python3
"""
Contract tests for 3D-RE-GEN outputs and Genie Sim local outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fixtures.generate_mock_geniesim_local import generate_mock_geniesim_local
from fixtures.generate_mock_regen3d import generate_mock_regen3d
from tests.contract_utils import load_schema, validate_json_schema


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_regen3d_contracts(tmp_path: Path) -> None:
    regen3d_dir = generate_mock_regen3d(
        output_dir=tmp_path,
        scene_id="contract_scene",
        environment_type="kitchen",
    )

    scene_info = _read_json(regen3d_dir / "scene_info.json")
    validate_json_schema(scene_info, load_schema("regen3d_scene_info.schema.json"))

    pose_schema = load_schema("regen3d_object_pose.schema.json")
    bounds_schema = load_schema("regen3d_object_bounds.schema.json")
    material_schema = load_schema("regen3d_object_material.schema.json")
    intrinsics_schema = load_schema("regen3d_camera_intrinsics.schema.json")
    extrinsics_schema = load_schema("regen3d_camera_extrinsics.schema.json")

    objects_dir = regen3d_dir / "objects"
    for obj_dir in objects_dir.iterdir():
        if not obj_dir.is_dir():
            continue
        validate_json_schema(_read_json(obj_dir / "pose.json"), pose_schema)
        validate_json_schema(_read_json(obj_dir / "bounds.json"), bounds_schema)
        validate_json_schema(_read_json(obj_dir / "material.json"), material_schema)

    validate_json_schema(_read_json(regen3d_dir / "background" / "pose.json"), pose_schema)
    validate_json_schema(_read_json(regen3d_dir / "background" / "bounds.json"), bounds_schema)
    validate_json_schema(_read_json(regen3d_dir / "camera" / "intrinsics.json"), intrinsics_schema)
    validate_json_schema(_read_json(regen3d_dir / "camera" / "extrinsics.json"), extrinsics_schema)


def test_regen3d_contracts_fail_on_violation(tmp_path: Path) -> None:
    regen3d_dir = generate_mock_regen3d(
        output_dir=tmp_path,
        scene_id="invalid_scene",
        environment_type="kitchen",
    )
    scene_info = _read_json(regen3d_dir / "scene_info.json")
    scene_info.pop("scene_id", None)

    with pytest.raises(ValueError):
        validate_json_schema(scene_info, load_schema("regen3d_scene_info.schema.json"))


def test_geniesim_local_contracts(tmp_path: Path) -> None:
    output_dir = generate_mock_geniesim_local(
        output_dir=tmp_path,
        run_id="contract_run",
        episodes=2,
    )

    episode_schema = load_schema("geniesim_local_episode.schema.json")
    dataset_schema = load_schema("geniesim_local_dataset_info.schema.json")
    index_schema = load_schema("geniesim_local_episodes_index.schema.json")

    recordings_dir = output_dir / "recordings"
    for episode_file in recordings_dir.glob("*.json"):
        validate_json_schema(_read_json(episode_file), episode_schema)

    metadata_dir = output_dir / "metadata"
    validate_json_schema(_read_json(metadata_dir / "dataset_info.json"), dataset_schema)

    episodes_index_path = metadata_dir / "episodes.jsonl"
    for line in episodes_index_path.read_text().splitlines():
        validate_json_schema(json.loads(line), index_schema)


def test_geniesim_local_contracts_fail_on_violation(tmp_path: Path) -> None:
    output_dir = generate_mock_geniesim_local(
        output_dir=tmp_path,
        run_id="invalid_run",
        episodes=1,
    )
    dataset_info = _read_json(output_dir / "metadata" / "dataset_info.json")
    dataset_info["episodes"] = "one"

    with pytest.raises(ValueError):
        validate_json_schema(dataset_info, load_schema("geniesim_local_dataset_info.schema.json"))
