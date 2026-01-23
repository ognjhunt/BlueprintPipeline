import os

from tools.config.production_mode import ensure_config_audit_for_production, is_config_audit_enabled


def test_production_mode_enables_config_audit(monkeypatch):
    monkeypatch.delenv("BP_ENABLE_CONFIG_AUDIT", raising=False)
    monkeypatch.setenv("PIPELINE_ENV", "production")

    assert ensure_config_audit_for_production() is True
    assert os.environ.get("BP_ENABLE_CONFIG_AUDIT") == "1"
    assert is_config_audit_enabled() is True
