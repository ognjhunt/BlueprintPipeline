from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.checkpoint import retention_cleanup, store

pytestmark = pytest.mark.usefixtures("add_repo_to_path")


def _touch(path: Path, *, mtime: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("data")
    timestamp = mtime.timestamp()
    path.utime((timestamp, timestamp))


@pytest.mark.unit
def test_cleanup_scene_prunes_old_files(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene-1"
    now = datetime(2024, 1, 10, tzinfo=timezone.utc)
    old = now - timedelta(days=10)
    recent = now - timedelta(days=1)

    input_file = scene_root / "input" / "payload.txt"
    output_file = scene_root / "assets" / "asset.usd"
    log_file = scene_root / "logs" / "run.log"

    _touch(input_file, mtime=old)
    _touch(output_file, mtime=recent)
    _touch(log_file, mtime=old)

    store.write_checkpoint(
        scene_root,
        "assets-step",
        status="completed",
        started_at="2024-01-01T00:00:00Z",
        completed_at="2024-01-01T00:10:00Z",
        output_paths=[output_file],
    )

    policy = retention_cleanup.RetentionPolicy(
        inputs_days=5,
        intermediate_days=5,
        outputs_days=5,
        logs_days=5,
        fallback_days=5,
    )

    deleted, considered = retention_cleanup.cleanup_scene(
        scene_root,
        policy,
        now=now,
        dry_run=False,
    )

    assert considered >= 3
    assert deleted == 2
    assert not input_file.exists()
    assert output_file.exists()
    assert not log_file.exists()


@pytest.mark.unit
def test_cleanup_scene_dry_run_keeps_files(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene-2"
    now = datetime(2024, 2, 1, tzinfo=timezone.utc)
    old = now - timedelta(days=20)

    input_file = scene_root / "input" / "payload.txt"
    _touch(input_file, mtime=old)

    policy = retention_cleanup.RetentionPolicy(
        inputs_days=5,
        intermediate_days=5,
        outputs_days=5,
        logs_days=5,
        fallback_days=5,
    )

    deleted, considered = retention_cleanup.cleanup_scene(
        scene_root,
        policy,
        now=now,
        dry_run=True,
    )

    assert considered == 1
    assert deleted == 0
    assert input_file.exists()
