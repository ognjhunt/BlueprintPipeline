import importlib.util
from pathlib import Path

import pytest


def _load_prepare_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "dwm-preparation-job" / "prepare_dwm_bundle.py"
    spec = importlib.util.spec_from_file_location("dwm_prepare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_production_requires_mano_assets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prepare_module = _load_prepare_module()
    from hand_motion.hand_mesh_renderer import MANOUnavailableError

    monkeypatch.delenv("REQUIRE_MANO", raising=False)
    monkeypatch.setenv("MANO_MODEL_PATH", str(tmp_path / "missing_mano"))

    config = prepare_module.DWMJobConfig(
        manifest_path=tmp_path / "scene_manifest.json",
        data_quality_level="production",
        allow_mock_rendering=False,
    )

    with pytest.raises(MANOUnavailableError):
        prepare_module.DWMPreparationJob(config)
