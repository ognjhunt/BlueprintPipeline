from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tests.contract_utils import load_schema, validate_json_schema


REPO_ROOT = Path(__file__).resolve().parents[1]
SCALE_SCRIPT = REPO_ROOT / "scale-job" / "run_scale_from_layout.py"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "scale"


def _load_scale_module():
    spec = importlib.util.spec_from_file_location("run_scale_from_layout", SCALE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_layout(layout_dir: Path, layout_fixture: Path, metadata_fixture: Path) -> None:
    layout_dir.mkdir(parents=True, exist_ok=True)
    layout = json.loads(layout_fixture.read_text())
    metadata = json.loads(metadata_fixture.read_text())
    (layout_dir / "scene_layout.json").write_text(json.dumps(layout, indent=2))
    (layout_dir / "metric_metadata.json").write_text(json.dumps(metadata, indent=2))


def _run_scale(layout_prefix: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BUCKET", "unit-test-bucket")
    monkeypatch.setenv("SCENE_ID", "unit-test-scene")
    monkeypatch.setenv("LAYOUT_PREFIX", layout_prefix)
    module = _load_scale_module()
    module.main()
    return Path("/mnt/gcs") / layout_prefix / "scene_layout_scaled.json"


@pytest.mark.parametrize(
    "layout_fixture,metadata_fixture,expected_scale",
    [
        (
            FIXTURES_DIR / "reference_scene_layout.json",
            FIXTURES_DIR / "reference_metric_metadata.json",
            2.0,
        ),
        (
            FIXTURES_DIR / "scene_metric_layout.json",
            FIXTURES_DIR / "scene_metric_metadata.json",
            2.0,
        ),
    ],
)
def test_scale_job_applies_reference_and_scene_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout_fixture: Path,
    metadata_fixture: Path,
    expected_scale: float,
) -> None:
    prefix = f"scale-job-tests/{layout_fixture.stem}"
    layout_dir = Path("/mnt/gcs") / prefix
    _write_layout(layout_dir, layout_fixture, metadata_fixture)

    schema = load_schema("metric_metadata.schema.json")
    validate_json_schema(json.loads(metadata_fixture.read_text()), schema)

    scaled_path = _run_scale(prefix, monkeypatch)
    scaled_layout = json.loads(scaled_path.read_text())
    assert scaled_layout["scale"]["factor"] == pytest.approx(expected_scale)
