import pytest

from tools.checkpoint.hash_config import resolve_checkpoint_hash_setting


pytestmark = pytest.mark.usefixtures("add_repo_to_path")


@pytest.mark.unit
def test_resolve_checkpoint_hashes_defaults_to_production(monkeypatch):
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.delenv("BP_CHECKPOINT_HASHES", raising=False)

    assert resolve_checkpoint_hash_setting() is True
