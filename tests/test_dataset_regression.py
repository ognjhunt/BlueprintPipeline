from __future__ import annotations

import json
from pathlib import Path


FIXTURE_ROOT = Path("tests/fixtures/golden/dataset_regression")
BASELINE_DATASET = FIXTURE_ROOT / "baseline_dataset"
BASELINE_METRICS = FIXTURE_ROOT / "baseline_metrics.json"
THRESHOLDS = FIXTURE_ROOT / "thresholds.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _assert_within(value: float, baseline: float, delta: float, label: str) -> None:
    assert abs(value - baseline) <= delta, (
        f"{label} drifted beyond threshold: "
        f"value={value}, baseline={baseline}, delta={delta}"
    )


def test_dataset_regression_metrics(add_repo_to_path) -> None:
    from tools.dataset_regression.metrics import compute_regression_metrics

    baseline = _load_json(BASELINE_METRICS)
    thresholds = _load_json(THRESHOLDS)
    metrics = compute_regression_metrics(BASELINE_DATASET)

    _assert_within(
        metrics["episode_count"],
        baseline["episode_count"],
        thresholds["episode_count_delta"],
        "episode_count",
    )

    for bucket, baseline_count in baseline["quality_distribution"].items():
        actual = metrics["quality_distribution"].get(bucket, 0)
        _assert_within(
            actual,
            baseline_count,
            thresholds["quality_distribution_delta"],
            f"quality_distribution[{bucket}]",
        )

    _assert_within(
        metrics["collision_rate"],
        baseline["collision_rate"],
        thresholds["collision_rate_delta"],
        "collision_rate",
    )

    _assert_within(
        metrics["duration_seconds"]["average_seconds"],
        baseline["duration_seconds"]["average_seconds"],
        thresholds["duration_average_delta"],
        "duration_seconds.average_seconds",
    )

    checksums = metrics["checksums"]
    assert len(checksums["missing_files"]) <= thresholds["checksum_missing_allowed"], (
        "Checksum missing files detected: "
        f"{checksums['missing_files']}"
    )
    assert len(checksums["mismatched_files"]) <= thresholds["checksum_mismatch_allowed"], (
        "Checksum mismatches detected: "
        f"{checksums['mismatched_files']}"
    )
    assert checksums["total_files"] >= 1
