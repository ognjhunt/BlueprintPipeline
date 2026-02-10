from __future__ import annotations

import logging
import sys

import pytest


def _reset_google_cloud_modules() -> None:
    for module_name in list(sys.modules):
        if (
            module_name == "google"
            or module_name.startswith("google.")
            or module_name == "firebase_admin"
            or module_name.startswith("firebase_admin.")
        ):
            sys.modules.pop(module_name, None)


def test_production_forces_filter_low_quality(
    caplog: pytest.LogCaptureFixture,
    load_job_module,
) -> None:
    _reset_google_cloud_modules()
    module = load_job_module("geniesim_import", "import_from_geniesim.py")

    quality_settings = module.ResolvedQualitySettings(
        min_quality_score=module.QUALITY_CONFIG.default_min_quality_score,
        filter_low_quality=False,
        dimension_thresholds=module.QUALITY_CONFIG.dimension_thresholds,
        config=module.QUALITY_CONFIG,
    )

    caplog.set_level(logging.WARNING)
    log = logging.getLogger("geniesim-import-test")
    updated = module._apply_production_filter_override(
        quality_settings,
        production_mode=True,
        log=log,
        env={"FILTER_LOW_QUALITY": "false"},
    )

    assert updated.filter_low_quality is True
    assert any(
        "Production mode forces FILTER_LOW_QUALITY=true" in record.message
        for record in caplog.records
    )
