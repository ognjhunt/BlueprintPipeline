from __future__ import annotations

from pathlib import Path


def test_run_pipeline_supports_backend_selection_and_auto_mode() -> None:
    script = Path("run_pipeline.sh").read_text(encoding="utf-8")

    assert 'DATA_BACKEND="auto"' in script
    assert "--backend" in script
    assert 'if [[ "${DATA_BACKEND}" == "auto" ]]; then' in script
    assert "_use_geniesim_raw" in script
    assert 'DATA_BACKEND="episode"' in script
    assert 'DATA_BACKEND="geniesim"' in script


def test_run_pipeline_routes_steps_by_backend_and_scopes_geniesim_checks() -> None:
    script = Path("run_pipeline.sh").read_text(encoding="utf-8")

    assert 'STEP_ARGS=(--steps genie-sim-submit --force-rerun genie-sim-submit --use-geniesim)' in script
    assert 'STEP_ARGS=(--steps isaac-lab --force-rerun isaac-lab)' in script
    assert 'if [[ "${DATA_BACKEND}" == "geniesim" ]]; then' in script
    assert "Checking gRPC readiness" in script
