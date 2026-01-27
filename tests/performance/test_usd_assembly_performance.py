import json
import time

import pytest

from tools.scene_manifest.loader import load_manifest_or_scene_assets


def _assert_time_scales(
    small_elapsed: float,
    large_elapsed: float,
    small_size: int,
    large_size: int,
    *,
    tolerance: float = 6.0,
) -> None:
    per_small = max(small_elapsed / small_size, 1e-6)
    per_large = large_elapsed / large_size
    assert per_large <= per_small * tolerance


def _write_manifest(manifest_path, object_count: int) -> None:
    objects = []
    for idx in range(object_count):
        objects.append(
            {
                "id": f"obj-{idx}",
                "sim_role": "static",
                "category": "object",
                "dimensions_est": {"width": 0.1, "depth": 0.1, "height": 0.1},
                "transform": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "asset": {
                    "path": f"objects/obj-{idx}/asset.usdz",
                    "source": "blueprintpipeline",
                },
            }
        )

    manifest = {
        "version": "1.0.0",
        "scene_id": "perf_scene",
        "scene": {
            "coordinate_frame": "y_up",
            "meters_per_unit": 1.0,
        },
        "objects": objects,
    }
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.slow
def test_usd_manifest_loading_scales_with_object_count(tmp_path):
    small_root = tmp_path / "small"
    large_root = tmp_path / "large"
    small_root.mkdir()
    large_root.mkdir()

    _write_manifest(small_root / "scene_manifest.json", 20)
    _write_manifest(large_root / "scene_manifest.json", 100)

    def run_load(root):
        start = time.perf_counter()
        loaded = load_manifest_or_scene_assets(root)
        elapsed = time.perf_counter() - start
        assert loaded is not None
        return elapsed

    small_elapsed = run_load(small_root)
    large_elapsed = run_load(large_root)

    _assert_time_scales(
        small_elapsed,
        large_elapsed,
        20,
        100,
    )
