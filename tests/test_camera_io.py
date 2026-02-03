from tools.camera_io import strip_camera_data


def test_strip_camera_data_retains_npy_refs() -> None:
    obs = {
        "camera_frames": {
            "wrist": {
                "rgb": "task_0_ep0000_frames/wrist_rgb_000.npy",
                "depth": "task_0_ep0000_frames/wrist_depth_000.npy",
                "width": 640,
                "height": 480,
            },
            "side": {
                "rgb": "inline-base64",
                "depth": None,
                "width": 640,
                "height": 480,
            },
        }
    }

    stripped = strip_camera_data(obs)
    wrist = stripped["camera_frames"]["wrist"]
    side = stripped["camera_frames"]["side"]

    assert wrist["rgb"].endswith(".npy")
    assert wrist["depth"].endswith(".npy")
    assert "rgb" not in side
    assert "depth" not in side
