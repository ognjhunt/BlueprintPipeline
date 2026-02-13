from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tools.geniesim_adapter.local_framework import GenieSimConfig, GenieSimLocalFramework
from tools.rlds.proto import example_pb2


def _read_tfrecord_records(path: Path) -> list[bytes]:
    records: list[bytes] = []
    with open(path, "rb") as handle:
        while True:
            length_bytes = handle.read(8)
            if not length_bytes:
                break
            (length,) = struct.unpack("<Q", length_bytes)
            handle.read(4)  # length crc
            payload = handle.read(length)
            handle.read(4)  # data crc
            records.append(payload)
    return records


@pytest.mark.unit
def test_geniesim_rlds_export_writes_tfrecord_and_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recording_dir = tmp_path / "recordings"
    recording_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = recording_dir / "task_0_ep0000_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Create tiny RGB frames for 3 canonical cameras.
    def _write_rgb(cam_id: str, frame_idx: int, rgb: np.ndarray) -> str:
        rel = f"task_0_ep0000_frames/{cam_id}_rgb_{frame_idx:06d}.npy"
        np.save(recording_dir / rel, rgb)
        return rel

    rgb0 = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb1 = np.full((8, 8, 3), 255, dtype=np.uint8)

    episode = {
        "episode_id": "task_0_ep0000",
        "task_name": "task_0",
        "task_description": "Pick up the red mug and place it in the dish rack.",
        "task_id": "pick_place::mug",
        "required_camera_ids": ["left", "right", "wrist"],
        "dataset_tier": "physics_certified",
        "certification": {"passed": True, "critical_failures": []},
        "quality_score": 0.99,
        "frames": [
            {
                "timestamp": 0.0,
                "observation": {
                    "robot_state": {"joint_positions": [0.0] * 7},
                    "camera_frames": {
                        "left": {"rgb": _write_rgb("left", 0, rgb0), "width": 8, "height": 8},
                        "right": {"rgb": _write_rgb("right", 0, rgb0), "width": 8, "height": 8},
                        "wrist": {"rgb": _write_rgb("wrist", 0, rgb0), "width": 8, "height": 8},
                    },
                },
                "action": [0.0] * 8,
                "reward": 0.0,
            },
            {
                "timestamp": 0.1,
                "observation": {
                    "robot_state": {"joint_positions": [0.1] * 7},
                    "camera_frames": {
                        "left": {"rgb": _write_rgb("left", 1, rgb1), "width": 8, "height": 8},
                        "right": {"rgb": _write_rgb("right", 1, rgb1), "width": 8, "height": 8},
                        "wrist": {"rgb": _write_rgb("wrist", 1, rgb1), "width": 8, "height": 8},
                    },
                },
                "action": [0.1] * 8,
                "reward": 1.0,
                "done": True,
            },
        ],
    }
    (recording_dir / "task_0_ep0000.json").write_text(json.dumps(episode))

    cfg = GenieSimConfig(
        recording_dir=recording_dir,
        log_dir=tmp_path / "logs",
    )
    framework = GenieSimLocalFramework(cfg, verbose=False)

    monkeypatch.setenv("REQUIRE_CAMERA_DATA", "true")
    monkeypatch.setenv("RLDS_REQUIRE_ENCODED_IMAGES", "true")

    rlds_dir = tmp_path / "rlds"
    export = framework.export_to_rlds(recording_dir=recording_dir, output_dir=rlds_dir)
    assert export["success"] is True
    assert export["exported"] == 1

    tfrecord_path = rlds_dir / "train" / "episode_000000.tfrecord"
    assert tfrecord_path.is_file()

    records = _read_tfrecord_records(tfrecord_path)
    assert len(records) == 2

    ex = example_pb2.Example.FromString(records[0])
    feats = ex.features.feature
    assert feats["language_instruction"].bytes_list.value[0].decode("utf-8") == episode["task_description"]
    assert feats["observation/image_wrist"].bytes_list.value[0].startswith(b"\xff\xd8\xff")
