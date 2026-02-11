from __future__ import annotations

from pathlib import Path


def test_run_pipeline_gcs_defaults_to_stage1_only_steps() -> None:
    script = Path("run_pipeline_gcs.sh").read_text()
    assert 'STEPS="${5:-regen3d-reconstruct,regen3d}"' in script


def test_run_pipeline_gcs_has_runner_dependency_preflight() -> None:
    script = Path("run_pipeline_gcs.sh").read_text()
    assert "preflight_runner_dependencies" in script
    assert "python3 -m pip install numpy PyYAML" in script
    assert '"numpy"' in script
    assert '"yaml"' in script
