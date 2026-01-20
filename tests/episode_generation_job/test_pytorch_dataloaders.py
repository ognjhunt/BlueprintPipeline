from pathlib import Path

import pytest


@pytest.mark.unit
def test_blueprint_episode_dataset_builds_samples(load_job_module, monkeypatch, tmp_path) -> None:
    dataloaders = load_job_module("episode_generation", "pytorch_dataloaders.py")

    monkeypatch.setattr(dataloaders, "TORCH_AVAILABLE", True)

    meta_dir = tmp_path / "episodes" / "meta"
    meta_dir.mkdir(parents=True)

    (meta_dir / "info.json").write_text("{\"state_dim\": 7, \"action_dim\": 4}")
    (meta_dir / "episodes.jsonl").write_text(
        "{\"length\": 5, \"chunk_index\": 0}\n"
    )
    (meta_dir / "stats.json").write_text(
        "{\"action\": {\"mean\": 0}, \"state\": {\"mean\": 1}}"
    )

    dataset = dataloaders.BlueprintEpisodeDataset(
        dataset_path=meta_dir.parent,
        chunk_size=2,
        history_length=1,
        verbose=False,
    )

    assert len(dataset.episodes) == 1
    assert len(dataset) == 2
    assert dataset.stats["action"]["mean"] == 0


@pytest.mark.unit
def test_data_loading_config_defaults(load_job_module) -> None:
    dataloaders = load_job_module("episode_generation", "pytorch_dataloaders.py")

    config = dataloaders.DataLoadingConfig()

    assert config.normalize_actions is True
    assert config.normalize_states is True


@pytest.mark.unit
def test_create_blueprint_dataloader_overrides_config(load_job_module, monkeypatch, tmp_path) -> None:
    dataloaders = load_job_module("episode_generation", "pytorch_dataloaders.py")

    monkeypatch.setattr(dataloaders, "TORCH_AVAILABLE", True)

    captured = {}

    class DummyDataset:
        def __init__(self, dataset_path, **kwargs):
            captured["dataset_path"] = dataset_path
            captured.update(kwargs)

    class DummyDataLoader:
        def __init__(self, dataset, **kwargs):
            captured["loader_dataset"] = dataset
            captured["loader_kwargs"] = kwargs

    monkeypatch.setattr(dataloaders, "BlueprintEpisodeDataset", DummyDataset)
    monkeypatch.setattr(dataloaders, "DataLoader", DummyDataLoader)

    config = dataloaders.DataLoadingConfig(batch_size=8, sample_by_episode=False)
    dataset_path = tmp_path / "episodes"

    loader = dataloaders.create_blueprint_dataloader(
        dataset_path=dataset_path,
        config=config,
        history_length=3,
    )

    assert isinstance(loader, DummyDataLoader)
    assert captured["dataset_path"] == dataset_path
    assert captured["history_length"] == 3
    assert captured["loader_kwargs"]["batch_size"] == 8
