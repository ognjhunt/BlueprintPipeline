import importlib.util
from pathlib import Path

import pytest


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_delivery_missing_bucket_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("dataset_delivery_missing_bucket", "dataset-delivery-job/dataset_delivery.py")
    monkeypatch.delenv("BUCKET", raising=False)
    monkeypatch.setenv("SCENE_ID", "scene_1")

    with pytest.raises(SystemExit):
        module.main()


def test_dataset_delivery_missing_scene_id_and_manifest_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("dataset_delivery_missing_scene", "dataset-delivery-job/dataset_delivery.py")
    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.delenv("SCENE_ID", raising=False)
    monkeypatch.delenv("IMPORT_MANIFEST_PATH", raising=False)

    with pytest.raises(SystemExit):
        module.main()
