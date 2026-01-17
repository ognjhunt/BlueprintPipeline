import importlib.util
from pathlib import Path

import pytest


def _load_data_pack_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "episode-generation-job" / "data_pack_config.py"
    spec = importlib.util.spec_from_file_location("data_pack_config", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_data_pack_from_string_defaults_to_core() -> None:
    data_pack_config = _load_data_pack_module()

    assert (
        data_pack_config.data_pack_from_string("unknown-tier")
        == data_pack_config.DataPackTier.CORE
    )


def test_dataset_split_config_env_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    data_pack_config = _load_data_pack_module()

    monkeypatch.setenv("BP_SPLIT_TRAIN_RATIO", "0.7")
    monkeypatch.setenv("BP_SPLIT_VAL_RATIO", "0.2")
    monkeypatch.setenv("BP_SPLIT_TEST_RATIO", "0.2")

    with pytest.raises(ValueError):
        data_pack_config.DatasetSplitConfig()


def test_dataset_split_config_success(monkeypatch: pytest.MonkeyPatch) -> None:
    data_pack_config = _load_data_pack_module()

    monkeypatch.delenv("BP_SPLIT_TRAIN_RATIO", raising=False)
    monkeypatch.delenv("BP_SPLIT_VAL_RATIO", raising=False)
    monkeypatch.delenv("BP_SPLIT_TEST_RATIO", raising=False)

    config = data_pack_config.DatasetSplitConfig(
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, split_seed=123
    )

    assert config.split_seed == 123
