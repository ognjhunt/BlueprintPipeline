import logging

import pytest

from tools.config import production_mode


def test_legacy_flags_warn_before_cutoff(monkeypatch, caplog):
    monkeypatch.setattr(
        production_mode,
        "_LEGACY_PRODUCTION_FLAG_REMOVAL_DATE",
        "2999-01-01",
    )
    env = {"BP_ENV": "production"}

    with caplog.at_level(logging.WARNING):
        result = production_mode.resolve_production_mode_detail(env=env)

    assert result == (False, None, None)
    assert any("Legacy production flags" in record.message for record in caplog.records)


def test_legacy_flags_raise_after_cutoff(monkeypatch):
    monkeypatch.setattr(
        production_mode,
        "_LEGACY_PRODUCTION_FLAG_REMOVAL_DATE",
        "2000-01-01",
    )
    env = {"PRODUCTION_MODE": "1"}

    with pytest.raises(RuntimeError, match="Legacy production flags"):
        production_mode.resolve_production_mode_detail(env=env)
