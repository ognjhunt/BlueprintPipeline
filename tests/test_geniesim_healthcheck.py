import json
import logging

import pytest

from tools.geniesim_adapter import geniesim_healthcheck


@pytest.fixture(autouse=True)
def _restore_logging():
    """Restore root logger state after healthcheck tests (which call init_logging)."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_disable = logging.root.manager.disable
    yield
    root.handlers = saved_handlers
    root.setLevel(saved_level)
    logging.disable(logging.NOTSET)
    logging.root.manager.disable = saved_disable


def test_geniesim_healthcheck_json_ok(monkeypatch, capsys):
    status_payload = {"isaac_sim_available": True}
    report_payload = {"ok": True, "status": status_payload}

    monkeypatch.setattr(
        "tools.geniesim_adapter.local_framework.check_geniesim_availability",
        lambda: status_payload,
    )
    monkeypatch.setattr(
        "tools.geniesim_adapter.local_framework.build_geniesim_preflight_report",
        lambda *_args, **_kwargs: report_payload,
    )
    monkeypatch.setattr("sys.argv", ["geniesim_healthcheck", "--json"])

    assert geniesim_healthcheck.main() == 0

    output = capsys.readouterr().out.strip()
    assert json.loads(output) == report_payload


def test_geniesim_healthcheck_human_non_ok(monkeypatch, capsys):
    status_payload = {"isaac_sim_available": False}
    report_payload = {
        "ok": False,
        "status": status_payload,
        "missing": ["grpc"],
    }

    monkeypatch.setattr(
        "tools.geniesim_adapter.local_framework.check_geniesim_availability",
        lambda: status_payload,
    )
    monkeypatch.setattr(
        "tools.geniesim_adapter.local_framework.build_geniesim_preflight_report",
        lambda *_args, **_kwargs: report_payload,
    )
    monkeypatch.setattr("sys.argv", ["geniesim_healthcheck"])

    assert geniesim_healthcheck.main() == 1

    output = capsys.readouterr().out
    assert "Genie Sim Health Check" in output
    assert "Missing requirements: grpc" in output
