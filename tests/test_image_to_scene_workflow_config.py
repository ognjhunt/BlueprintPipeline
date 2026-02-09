from __future__ import annotations

from pathlib import Path


def test_image_to_scene_workflow_uses_cloud_build_executor():
    workflow = Path("workflows/image-to-scene-pipeline.yaml").read_text()

    assert "googleapis.cloudbuild.v1.projects.builds.create" in workflow
    assert "gcloud compute ssh" in workflow
    assert "run_pipeline_gcs.sh" in workflow


def test_image_to_scene_workflow_accepts_any_image_name_under_images_prefix():
    workflow = Path("workflows/image-to-scene-pipeline.yaml").read_text()

    assert "^scenes/[^/]+/images/[^/]+\\\\.([Pp][Nn][Gg]|[Jj][Pp][Ee]?[Gg])$" in workflow


def test_setup_image_trigger_uses_wildcard_path_pattern():
    setup_script = Path("workflows/setup-image-trigger.sh").read_text()

    assert '--event-filters-path-pattern="name=scenes/*/images/*"' in setup_script
    assert "cloudbuild.googleapis.com" in setup_script
