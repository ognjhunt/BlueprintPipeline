from __future__ import annotations

import pytest


def test_require_lerobot_defaults_in_production(monkeypatch, load_job_module):
    monkeypatch.delenv("REQUIRE_LEROBOT", raising=False)
    monkeypatch.setenv("PRODUCTION_MODE", "1")
    monkeypatch.delenv("SERVICE_MODE", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)

    module = load_job_module("geniesim_import", "import_from_geniesim.py")
    production_mode = module.resolve_production_mode()
    service_mode = module._is_service_mode()

    resolution = module._resolve_require_lerobot(
        None,
        production_mode=production_mode,
        service_mode=service_mode,
    )

    assert resolution.value is True
    assert resolution.default is True
    assert resolution.source == "default"
