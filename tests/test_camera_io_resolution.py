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
