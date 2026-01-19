import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_episodes() -> list[dict]:
    return [
        {
            "episode_id": "episode_000000",
            "task": "Move to target",
            "resolution": [320, 240],
            "frames": [
                {
                    "timestamp": 0.0,
                    "joint_positions": [0.0] * 7,
                    "gripper_position": 0.0,
                    "action": [0.0] * 8,
                    "reward": 0.1,
                },
                {
                    "timestamp": 0.1,
                    "joint_positions": [0.05] * 7,
                    "gripper_position": 0.01,
                    "action": [0.05] * 8,
                    "reward": 0.2,
                },
            ],
        }
    ]


@pytest.mark.unit
def test_rlds_exporter_writes_split_files(load_job_module, sample_episodes: list[dict], tmp_path: Path) -> None:
    exporters = load_job_module("episode_generation", "multi_format_exporters.py")
    exporter = exporters.RLDSExporter(output_dir=tmp_path / "rlds", verbose=False)

    output_dir = exporter.export_episodes(
        episodes=sample_episodes,
        splits={"train": ["episode_000000"]},
        robot_type="franka",
        camera_names=["wrist"],
    )

    assert (output_dir / "dataset_info.json").exists()
    assert (output_dir / "features.json").exists()
    dataset_info = json.loads((output_dir / "dataset_info.json").read_text())
    split_dir = output_dir / "train"

    files = list(split_dir.iterdir())
    assert files

    if exporter._tf_available:
        assert any(file.suffix == ".tfrecord" for file in files)
        assert dataset_info["export_status"]["used_json_fallback"] is False
    else:
        json_files = [file for file in files if file.suffix == ".json"]
        assert json_files
        payload = json.loads(json_files[0].read_text())
        assert len(payload["steps"]) == len(sample_episodes[0]["frames"])
        assert payload["steps"][0]["is_first"] is True
        assert payload["steps"][1]["is_last"] is True
        assert dataset_info["export_status"]["used_json_fallback"] is True


@pytest.mark.unit
def test_hdf5_exporter_handles_dependency(load_job_module, sample_episodes: list[dict], tmp_path: Path) -> None:
    exporters = load_job_module("episode_generation", "multi_format_exporters.py")
    exporter = exporters.HDF5Exporter(output_path=tmp_path / "dataset.hdf5", verbose=False)

    if not exporter._h5py_available:
        with pytest.raises(RuntimeError):
            exporter.export_episodes(
                episodes=sample_episodes,
                splits={"train": ["episode_000000"]},
                robot_type="franka",
            )
        return

    output_path = exporter.export_episodes(
        episodes=sample_episodes,
        splits={"train": ["episode_000000"]},
        robot_type="franka",
    )
    assert output_path.exists()

    h5py = exporter._h5py
    with h5py.File(output_path, "r") as handle:
        assert "data" in handle
        assert "mask" in handle
        assert handle.attrs["robot_type"] == "franka"


@pytest.mark.unit
def test_rosbag_exporter_fallback(load_job_module, sample_episodes: list[dict], tmp_path: Path) -> None:
    exporters = load_job_module("episode_generation", "multi_format_exporters.py")
    exporter = exporters.ROSBagExporter(output_dir=tmp_path / "rosbag", verbose=False)

    output_dir = exporter.export_episodes(sample_episodes, robot_type="franka")
    assert (output_dir / "metadata.json").exists()

    if exporter._rosbag_available:
        assert (output_dir / "episode_000000.bag").exists()
    else:
        assert (output_dir / "episode_000000_rosbag.json").exists()
