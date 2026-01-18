import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.geniesim_adapter import local_framework as lf


def test_curobo_missing_enables_fallback_non_production(monkeypatch):
    monkeypatch.setenv("GENIESIM_ENV", "development")
    monkeypatch.delenv("GENIESIM_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.setattr(lf, "CUROBO_INTEGRATION_AVAILABLE", False)

    config = lf.GenieSimConfig.from_env()
    framework = lf.GenieSimLocalFramework(config=config, verbose=False)

    assert framework.config.allow_linear_fallback is True


def test_curobo_missing_production_fails_fast(monkeypatch):
    monkeypatch.setenv("GENIESIM_ENV", "production")
    monkeypatch.delenv("GENIESIM_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.setattr(lf, "CUROBO_INTEGRATION_AVAILABLE", False)

    config = lf.GenieSimConfig.from_env()

    with pytest.raises(RuntimeError, match="pip install nvidia-curobo"):
        lf.GenieSimLocalFramework(config=config, verbose=False)


@pytest.mark.unit
def test_check_geniesim_availability_allows_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("GENIESIM_ENV", "development")
    monkeypatch.setenv("ALLOW_GENIESIM_MOCK", "1")
    monkeypatch.setenv("GENIESIM_ROOT", str(tmp_path / "missing_geniesim"))
    isaac_path = tmp_path / "isaac"
    isaac_path.mkdir()
    (isaac_path / "python.sh").write_text("#!/bin/sh\necho isaac\n")
    monkeypatch.setenv("ISAAC_SIM_PATH", str(isaac_path))

    monkeypatch.setattr(lf.GenieSimGRPCClient, "_check_server_socket", lambda self: False)
    monkeypatch.setitem(sys.modules, "grpc", type("GrpcStub", (), {})())

    status = lf.check_geniesim_availability()

    assert status["mock_server_allowed"] is True
    assert status["isaac_sim_available"] is True
    assert status["available"] is True
