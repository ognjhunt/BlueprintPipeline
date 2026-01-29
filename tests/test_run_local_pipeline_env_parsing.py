import logging

import pytest

from tools.error_handling.retry import NonRetryableError
from tools.run_local_pipeline import LocalPipelineRunner


def _make_runner(tmp_path) -> LocalPipelineRunner:
    return LocalPipelineRunner(
        scene_dir=tmp_path,
        verbose=True,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )


def test_parse_env_int_invalid_production_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.setenv("BP_QUALITY_HUMAN_APPROVAL_NOTIFICATION_CHANNELS", "#test-approvals")
    monkeypatch.setenv("PIPELINE_RETRY_MAX", "nope")

    with pytest.raises(NonRetryableError, match="PIPELINE_RETRY_MAX"):
        _make_runner(tmp_path)


def test_parse_env_float_invalid_production_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.setenv("BP_QUALITY_HUMAN_APPROVAL_NOTIFICATION_CHANNELS", "#test-approvals")
    monkeypatch.setenv("PIPELINE_RETRY_BASE_DELAY", "n/a")

    with pytest.raises(NonRetryableError, match="PIPELINE_RETRY_BASE_DELAY"):
        _make_runner(tmp_path)


def test_parse_resolution_env_invalid_production_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.setenv("BP_QUALITY_HUMAN_APPROVAL_NOTIFICATION_CHANNELS", "#test-approvals")
    monkeypatch.setenv("DEFAULT_CAMERA_RESOLUTION", "1024xNaN")
    runner = _make_runner(tmp_path)

    with pytest.raises(NonRetryableError, match="DEFAULT_CAMERA_RESOLUTION"):
        runner._parse_resolution_env("DEFAULT_CAMERA_RESOLUTION", default=[320, 240])


def test_parse_resolution_env_invalid_non_production_warns(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PIPELINE_ENV", "development")
    monkeypatch.setenv("DEFAULT_CAMERA_RESOLUTION", "badvalue")
    runner = _make_runner(tmp_path)

    result = runner._parse_resolution_env(
        "DEFAULT_CAMERA_RESOLUTION",
        default=[320, 240],
    )
    captured = capsys.readouterr()

    assert result == [320, 240]
    assert "Invalid DEFAULT_CAMERA_RESOLUTION value 'badvalue'" in captured.out


def test_pipeline_env_canonical_has_no_warnings(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.setenv("BP_QUALITY_HUMAN_APPROVAL_NOTIFICATION_CHANNELS", "#test-approvals")

    with caplog.at_level(logging.WARNING):
        runner = _make_runner(tmp_path)

    assert runner.environment == "production"
    assert not any("deprecated" in record.message for record in caplog.records)
