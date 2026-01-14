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
