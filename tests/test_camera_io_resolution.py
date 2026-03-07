from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.camera_io import load_camera_frame


def test_load_camera_frame_resolves_frames_dir(tmp_path: Path) -> None:
    frames_dir = tmp_path / "episode_000_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    np.save(frames_dir / "wrist_rgb_000.npy", arr)

    cam_data = {
        "rgb": "wrist_rgb_000.npy",
        "width": 2,
        "height": 2,
    }

    loaded = load_camera_frame(cam_data, "rgb", ep_dir=tmp_path, frames_dir=frames_dir)
    assert loaded is not None
    assert loaded.shape == arr.shape

    loaded_scan = load_camera_frame(cam_data, "rgb", ep_dir=tmp_path)
    assert loaded_scan is not None
    assert loaded_scan.shape == arr.shape


def test_load_camera_frame_rejects_absolute_path_outside_roots(tmp_path: Path) -> None:
    secret = tmp_path / "secret.npy"
    arr = np.ones((2, 2, 3), dtype=np.uint8)
    np.save(secret, arr)

    cam_data = {
        "rgb": str(secret.resolve()),
        "width": 2,
        "height": 2,
    }

    # No trusted roots passed in (as in physics certification); absolute paths are rejected.
    loaded = load_camera_frame(cam_data, "rgb")
    assert loaded is None


def test_load_camera_frame_handles_malformed_npy(tmp_path: Path) -> None:
    bad = tmp_path / "bad.npy"
    bad.write_bytes(b"not-a-valid-npy")

    cam_data = {
        "rgb": str(bad.resolve()),
        "width": 2,
        "height": 2,
    }

    loaded = load_camera_frame(cam_data, "rgb", ep_dir=tmp_path)
    assert loaded is None
