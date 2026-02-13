import numpy as np

from tools.point_cloud import CameraIntrinsics, depth_to_points


def test_depth_to_points_unprojects_camera_frame() -> None:
    depth = np.ones((2, 2), dtype=np.float32)  # 1m everywhere
    intr = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)

    out = depth_to_points(depth, intr, max_points=None, max_depth_m=10.0)
    pts = out["points"]

    assert pts.shape == (4, 3)

    # np.where returns row-major order: (y,x) = (0,0),(0,1),(1,0),(1,1)
    expected = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(pts, expected)


def test_depth_to_points_applies_extrinsic_cam_to_world() -> None:
    depth = np.ones((1, 2), dtype=np.float32)  # pixels (0,0),(1,0)
    intr = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    extr = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    out = depth_to_points(depth, intr, extrinsic_cam_to_world=extr, max_points=None)
    pts = out["points"]

    expected = np.array(
        [
            [1.0, 2.0, 4.0],
            [2.0, 2.0, 4.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(pts, expected)


def test_depth_to_points_samples_colors() -> None:
    depth = np.ones((2, 2), dtype=np.float32)
    rgb = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    intr = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)

    out = depth_to_points(depth, intr, rgb=rgb, max_points=None)
    colors = out["colors"]

    assert colors.shape == (4, 3)
    expected = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
            [100, 110, 120],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(colors, expected)

