from __future__ import annotations

import os
import sys
import time

import pytest

from tools.run_local_pipeline import LocalPipelineRunner


def test_main_forwards_gcs_input_args(monkeypatch, tmp_path):
    from tools import run_local_pipeline

    captured = {}

    def fake_configure(
        self,
        bucket_name,
        download_inputs=False,
        upload_outputs=False,
        input_object=None,
        input_generation=None,
        upload_concurrency=None,
    ):
        captured.update(
            {
                "bucket_name": bucket_name,
                "download_inputs": download_inputs,
                "upload_outputs": upload_outputs,
                "input_object": input_object,
                "input_generation": input_generation,
                "upload_concurrency": upload_concurrency,
            }
        )

    def fake_run(self, steps=None, **kwargs):
        return True

    monkeypatch.setattr(run_local_pipeline.LocalPipelineRunner, "configure_gcs_sync", fake_configure)
    monkeypatch.setattr(run_local_pipeline.LocalPipelineRunner, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_local_pipeline.py",
            "--scene-dir",
            str(tmp_path),
            "--gcs-bucket",
            "blueprint-8c1ca.appspot.com",
            "--gcs-download-inputs",
            "--gcs-upload-outputs",
            "--gcs-input-object",
            "scenes/test_scene/images/kitchen_v2.jpeg",
            "--gcs-input-generation",
            "1739220750732768",
            "--gcs-upload-concurrency",
            "8",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        run_local_pipeline.main()

    assert excinfo.value.code == 0
    assert captured == {
        "bucket_name": "blueprint-8c1ca.appspot.com",
        "download_inputs": True,
        "upload_outputs": True,
        "input_object": "scenes/test_scene/images/kitchen_v2.jpeg",
        "input_generation": "1739220750732768",
        "upload_concurrency": 8,
    }


def test_gcs_input_object_without_bucket_fails(monkeypatch, tmp_path):
    from tools import run_local_pipeline

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_local_pipeline.py",
            "--scene-dir",
            str(tmp_path),
            "--gcs-input-object",
            "scenes/test_scene/images/kitchen_v2.jpeg",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        run_local_pipeline.main()

    assert excinfo.value.code == 1


def test_find_input_image_accepts_any_jpeg_name(tmp_path):
    scene_dir = tmp_path / "scene"
    input_dir = scene_dir / "input"
    input_dir.mkdir(parents=True)

    older = input_dir / "old_name.jpg"
    newer = input_dir / "kitchen_v2.jpeg"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    now = time.time()
    older_ts = now - 120
    newer_ts = now - 60
    os.utime(older, (older_ts, older_ts))
    os.utime(newer, (newer_ts, newer_ts))

    runner = LocalPipelineRunner(
        scene_dir=scene_dir,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )

    selected = runner._find_input_image()

    assert selected is not None
    assert selected.name == "kitchen_v2.jpeg"
