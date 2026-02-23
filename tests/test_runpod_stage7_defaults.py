from __future__ import annotations

from pathlib import Path


def test_run_full_pipeline_defaults_to_rgb_required() -> None:
    script = Path("scripts/runpod_sage/run_full_pipeline.sh").read_text()
    assert 'SAGE_REQUIRE_VALID_RGB="${SAGE_REQUIRE_VALID_RGB:-1}"' in script


def test_run_full_pipeline_defaults_to_probe_policy() -> None:
    script = Path("scripts/runpod_sage/run_full_pipeline.sh").read_text()
    assert 'SAGE_STAGE7_RGB_POLICY="${SAGE_STAGE7_RGB_POLICY:-auto_probe_fail}"' in script
    assert 'SAGE_STAGE7_MODE_ORDER="${SAGE_STAGE7_MODE_ORDER:-auto}"' in script
