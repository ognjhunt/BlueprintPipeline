from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import pytest

from fixtures.generate_mock_geniesim_local import generate_mock_geniesim_local


@dataclass
class MockEpisodeMetadata:
    episode_id: str
    task_name: str
    quality_score: float
    quality_components: Dict[str, float] = field(default_factory=dict)
    frame_count: int = 0
    duration_seconds: float = 0.0
    validation_passed: bool = True
    file_size_bytes: int = 0
    episode_content_hash: str | None = None


def _load_episode(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_parquet_episode(frames: list[dict], output_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    data = {
        "timestamp": [frame["timestamp"] for frame in frames],
        "action": [frame["action"] for frame in frames],
        "joint_positions": [
            frame["observation"]["joint_positions"] for frame in frames
        ],
        "robot_state": [frame["robot_state"] for frame in frames],
        "rgb_image": [frame.get("rgb_image") for frame in frames],
    }
    table = pa.Table.from_pydict(data)
    pq.write_table(table, output_path)


def test_geniesim_local_mock_deterministic_seed(tmp_path: Path) -> None:
    run_a = generate_mock_geniesim_local(
        output_dir=tmp_path / "run_a",
        run_id="seeded",
        episodes=1,
        seed=123,
    )
    run_b = generate_mock_geniesim_local(
        output_dir=tmp_path / "run_b",
        run_id="seeded",
        episodes=1,
        seed=123,
    )

    episode_a = _load_episode(run_a / "recordings" / "episode_000000.json")
    episode_b = _load_episode(run_b / "recordings" / "episode_000000.json")

    assert episode_a["frames"] == episode_b["frames"]


def test_geniesim_local_mock_converts_without_warnings(
    tmp_path: Path,
    load_job_module,
    monkeypatch,
) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("imageio")
    module = load_job_module("geniesim_import", "import_from_geniesim.py")

    monkeypatch.setenv("BP_PIPELINE_VIDEO_RESOLUTION_WIDTH", "8")
    monkeypatch.setenv("BP_PIPELINE_VIDEO_RESOLUTION_HEIGHT", "8")
    monkeypatch.setenv("BP_PIPELINE_VIDEO_FPS", "30")
    monkeypatch.setenv("BP_PIPELINE_VIDEO_CODEC", "h264")

    output_dir = generate_mock_geniesim_local(
        output_dir=tmp_path,
        run_id="conversion_run",
        episodes=1,
        seed=42,
    )
    recordings_dir = output_dir / "recordings"
    episode_path = recordings_dir / "episode_000000.json"
    episode = _load_episode(episode_path)
    frames = episode["frames"]

    parquet_path = recordings_dir / "episode_000000.parquet"
    _write_parquet_episode(frames, parquet_path)

    metadata = MockEpisodeMetadata(
        episode_id=episode["episode_id"],
        task_name=episode["task_name"],
        quality_score=float(episode["quality_score"]),
        quality_components=episode.get("quality_components", {}),
        frame_count=int(episode["frame_count"]),
        duration_seconds=float(frames[-1]["timestamp"]),
        validation_passed=bool(episode["validation_passed"]),
        file_size_bytes=parquet_path.stat().st_size,
    )

    lerobot_dir = tmp_path / "lerobot_output"
    conversion = module.convert_to_lerobot(
        episodes_dir=recordings_dir,
        output_dir=lerobot_dir,
        episode_metadata_list=[metadata],
        min_quality_score=0.0,
    )

    assert conversion["conversion_failures"] == []
    output_parquet = lerobot_dir / "episode_000000.parquet"
    assert output_parquet.exists()

    validation = module._stream_parquet_validation(
        output_parquet,
        require_parquet_validation=True,
    )
    filtered_warnings = [
        warning
        for warning in validation["warnings"]
        if not warning.startswith("Parquet validation used")
    ]
    assert validation["errors"] == []
    assert filtered_warnings == []


def test_geniesim_local_mock_writes_videos_and_metadata(
    tmp_path: Path,
    load_job_module,
    monkeypatch,
) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("imageio")
    module = load_job_module("geniesim_import", "import_from_geniesim.py")

    monkeypatch.setenv("BP_PIPELINE_VIDEO_RESOLUTION_WIDTH", "8")
    monkeypatch.setenv("BP_PIPELINE_VIDEO_RESOLUTION_HEIGHT", "8")
    monkeypatch.setenv("BP_PIPELINE_VIDEO_FPS", "30")
    monkeypatch.setenv("BP_PIPELINE_VIDEO_CODEC", "h264")

    output_dir = generate_mock_geniesim_local(
        output_dir=tmp_path,
        run_id="video_run",
        episodes=1,
        seed=7,
    )
    recordings_dir = output_dir / "recordings"
    episode_path = recordings_dir / "episode_000000.json"
    episode = _load_episode(episode_path)
    frames = episode["frames"]

    parquet_path = recordings_dir / "episode_000000.parquet"
    _write_parquet_episode(frames, parquet_path)

    metadata = MockEpisodeMetadata(
        episode_id=episode["episode_id"],
        task_name=episode["task_name"],
        quality_score=float(episode["quality_score"]),
        quality_components=episode.get("quality_components", {}),
        frame_count=int(episode["frame_count"]),
        duration_seconds=float(frames[-1]["timestamp"]),
        validation_passed=bool(episode["validation_passed"]),
        file_size_bytes=parquet_path.stat().st_size,
    )

    lerobot_dir = tmp_path / "lerobot_video_output"
    conversion = module.convert_to_lerobot(
        episodes_dir=recordings_dir,
        output_dir=lerobot_dir,
        episode_metadata_list=[metadata],
        min_quality_score=0.0,
    )
    assert conversion["conversion_failures"] == []

    expected_video = (
        lerobot_dir
        / "videos"
        / "camera"
        / "chunk-000"
        / "file-0000.mp4"
    )
    assert expected_video.exists()

    dataset_info = json.loads((lerobot_dir / "dataset_info.json").read_text())
    assert dataset_info["episodes"][0]["video_paths"]["camera"] == (
        "videos/camera/chunk-000/file-0000.mp4"
    )

    episodes_meta_path = (
        lerobot_dir
        / "meta"
        / "episodes"
        / "chunk-000"
        / "file-0000.parquet"
    )
    assert episodes_meta_path.exists()
    pq = pytest.importorskip("pyarrow.parquet")
    table = pq.read_table(episodes_meta_path)
    video_paths = json.loads(table.column("video_paths")[0].as_py())
    assert video_paths["camera"] == "videos/camera/chunk-000/file-0000.mp4"
