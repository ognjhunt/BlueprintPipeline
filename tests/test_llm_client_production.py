import pytest

from tools.llm_client import client as llm_client


def test_production_rejects_env_fallback(monkeypatch):
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.setenv("GEMINI_API_KEY", "env-only")

    def _raise_secret(*args, **kwargs):
        raise RuntimeError("secret missing")

    monkeypatch.setattr(llm_client, "HAVE_SECRET_MANAGER", True)
    monkeypatch.setattr(llm_client, "get_secret_or_env", _raise_secret)

    with pytest.raises(ValueError, match="Secret Manager"):
        llm_client._get_secret_or_env_with_log(
            "gemini-api-key",
            "GEMINI_API_KEY",
            "Gemini",
        )
