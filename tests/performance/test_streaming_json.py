import json

import pytest


@pytest.mark.unit
def test_fallback_stream_json_progress_callback(tmp_path, add_repo_to_path):
    from tools.performance import streaming_json
    manifest_path = tmp_path / "scene_manifest.json"
    manifest = {
        "scene_id": "scene-1",
        "objects": [
            {"id": "obj-1"},
            {"id": "obj-2"},
            {"id": "obj-3"},
        ],
    }
    manifest_path.write_text(json.dumps(manifest))

    calls = []

    def progress_callback(processed_count, elapsed_seconds):
        calls.append((processed_count, elapsed_seconds))

    batches = list(
        streaming_json._fallback_json_array(
            manifest_path,
            "objects",
            batch_size=1,
            progress_callback=progress_callback,
            progress_interval_s=0.0,
            progress_every=1,
        )
    )

    assert len(batches) == 3
    assert len(calls) >= 3
    assert calls[-1][0] == 3
    assert calls[-1][1] >= 0.0
