import json
import time

import pytest

from tools.geniesim_adapter import GenieSimExportConfig, GenieSimExporter
import tools.geniesim_adapter.scene_graph as scene_graph


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
                "name": f"Object {idx}",
                "category": "object",
                "description": "Minimal object",
                "sim_role": "static",
                "dimensions_est": {"width": 0.1, "depth": 0.1, "height": 0.1},
                "transform": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "asset": {
                    "path": f"objects/obj-{idx}/asset.usdz",
                    "source": "blueprintpipeline",
                },
                "semantics": {
                    "affordances": [],
                },
            }
        )

    manifest = {
        "version": "1.0.0",
        "scene_id": "geniesim_perf_scene",
        "scene": {
            "environment_type": "test",
            "coordinate_frame": "y_up",
            "meters_per_unit": 1.0,
        },
        "objects": objects,
    }
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.slow
def test_geniesim_export_scales_with_object_count(tmp_path, monkeypatch):
    small_manifest = tmp_path / "small_manifest.json"
    large_manifest = tmp_path / "large_manifest.json"

    _write_manifest(small_manifest, 10)
    _write_manifest(large_manifest, 40)

    monkeypatch.setattr(scene_graph, "HAVE_PIPELINE_CONFIG", False)
    monkeypatch.setattr(scene_graph, "_SCENE_GRAPH_CONFIG", None)

    config = GenieSimExportConfig(
        generate_embeddings=False,
        require_embeddings=False,
        filter_commercial_only=True,
        dry_run=True,
    )
    exporter = GenieSimExporter(config=config, verbose=False)

    def run_export(manifest_path, output_dir):
        start = time.perf_counter()
        result = exporter.export(manifest_path, output_dir)
        elapsed = time.perf_counter() - start
        assert result.success
        return elapsed

    small_elapsed = run_export(small_manifest, tmp_path / "small_out")
    large_elapsed = run_export(large_manifest, tmp_path / "large_out")

    _assert_time_scales(
        small_elapsed,
        large_elapsed,
        10,
        40,
    )
