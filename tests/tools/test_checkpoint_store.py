import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.checkpoint import store

pytestmark = pytest.mark.usefixtures("add_repo_to_path")


@pytest.mark.unit
def test_write_and_load_checkpoint_round_trip(tmp_path: Path) -> None:
    output_path = tmp_path / "outputs" / "result.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("{}")

    path = store.write_checkpoint(
        tmp_path,
        "step-a",
        status="completed",
        started_at="2024-01-01T00:00:00Z",
        completed_at="2024-01-01T00:10:00Z",
        outputs={"foo": "bar"},
        output_paths=[output_path],
        scene_id="scene-1",
    )

    payload = json.loads(path.read_text())
    assert payload["step"] == "step-a"
    assert payload["output_paths"] == [str(output_path)]
    assert payload["output_hashes"] == {}

    loaded = store.load_checkpoint(tmp_path, "step-a")
    assert loaded is not None
    assert loaded.step == "step-a"
    assert loaded.status == "completed"
    assert loaded.outputs == {"foo": "bar"}
    assert loaded.output_paths == [str(output_path)]
    assert loaded.scene_id == "scene-1"


@pytest.mark.unit
def test_should_skip_step_requires_outputs(tmp_path: Path) -> None:
    missing_output = tmp_path / "outputs" / "missing.txt"
    store.write_checkpoint(
        tmp_path,
        "step-b",
        status="completed",
        started_at="2024-01-01T00:00:00Z",
        completed_at="2024-01-01T00:02:00Z",
        output_paths=[missing_output],
    )

    assert store.should_skip_step(tmp_path, "step-b") is False

    missing_output.parent.mkdir(parents=True)
    missing_output.write_text("ok")
    assert store.should_skip_step(tmp_path, "step-b") is True

    expected_output = tmp_path / "outputs" / "expected.txt"
    assert store.should_skip_step(
        tmp_path,
        "step-b",
        expected_outputs=[expected_output],
    ) is False


@pytest.mark.unit
def test_should_skip_step_validates_hashes(tmp_path: Path) -> None:
    output_path = tmp_path / "outputs" / "artifact.txt"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("initial")

    store.write_checkpoint(
        tmp_path,
        "step-hash",
        status="completed",
        started_at="2024-01-01T00:00:00Z",
        completed_at="2024-01-01T00:02:00Z",
        output_paths=[output_path],
        store_output_hashes=True,
    )

    assert store.should_skip_step(tmp_path, "step-hash") is True

    output_path.write_text("modified")
    assert store.should_skip_step(tmp_path, "step-hash") is False


@pytest.mark.unit
def test_should_skip_step_validates_freshness_and_metadata(tmp_path: Path) -> None:
    output_path = tmp_path / "outputs" / "artifact.bin"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("data")

    sidecar = Path(f"{output_path}.metadata.json")
    sidecar.write_text("{}")

    future_time = time.time() + 3600
    completed_at = datetime.fromtimestamp(future_time, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    store.write_checkpoint(
        tmp_path,
        "step-fresh",
        status="completed",
        started_at="2024-01-01T00:00:00Z",
        completed_at=completed_at,
        output_paths=[output_path],
    )

    assert store.should_skip_step(
        tmp_path,
        "step-fresh",
        require_nonempty=True,
        require_fresh_outputs=True,
        validate_sidecar_metadata=True,
    ) is False

    os.utime(output_path, (future_time + 10, future_time + 10))
    assert store.should_skip_step(
        tmp_path,
        "step-fresh",
        require_nonempty=True,
        require_fresh_outputs=True,
        validate_sidecar_metadata=True,
    ) is True


@pytest.mark.unit
def test_load_checkpoint_missing_file_returns_none(tmp_path: Path) -> None:
    assert store.load_checkpoint(tmp_path, "missing-step") is None
