#!/usr/bin/env python3
"""
Validate Stage 7 artifact/provenance contract for runpod SAGE pipeline outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return data


def _record(results: List[Dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    results.append({"name": name, "ok": bool(ok), "detail": detail})


def _exists(path: Path) -> bool:
    return path.exists()


def _hdf5_demo_count(path: Path) -> int:
    import h5py

    with h5py.File(str(path), "r") as f:
        data = f.get("data")
        if data is None:
            return 0
        return int(len(data.keys()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stage 7 artifact and provenance contract")
    parser.add_argument("--layout-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-demos", type=int, default=0)
    parser.add_argument("--strict-artifact-contract", type=int, default=1)
    parser.add_argument("--strict-provenance", type=int, default=1)
    parser.add_argument("--report-path", default="")
    args = parser.parse_args()

    layout_dir = Path(args.layout_dir).expanduser().resolve()
    run_id = str(args.run_id).strip()
    strict_artifacts = bool(int(args.strict_artifact_contract))
    strict_provenance = bool(int(args.strict_provenance))
    expected_demos = max(0, int(args.expected_demos))
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if str(args.report_path).strip()
        else (layout_dir / "quality" / "stage7_contract_report.json")
    )

    demos_dir = layout_dir / "demos"
    plans_dir = layout_dir / "plans"
    quality_dir = layout_dir / "quality"

    checks: List[Dict[str, Any]] = []
    errors: List[str] = []

    required_paths = {
        "generation_dir": layout_dir / "generation",
        "usd_cache_dir": layout_dir / "usd_cache",
        "dataset_hdf5": demos_dir / "dataset.hdf5",
        "demo_metadata": demos_dir / "demo_metadata.json",
        "quality_report": demos_dir / "quality_report.json",
        "artifact_manifest": demos_dir / "artifact_manifest.json",
        "plan_bundle": plans_dir / "plan_bundle.json",
    }
    for name, path in required_paths.items():
        ok = _exists(path)
        _record(checks, name, ok, str(path))
        if strict_artifacts and not ok:
            errors.append(f"missing required path: {path}")

    scene_usds = sorted(demos_dir.glob("scene_*.usd"))
    _record(checks, "scene_usd_count", len(scene_usds) >= 1, f"count={len(scene_usds)}")
    if strict_artifacts and len(scene_usds) < 1:
        errors.append("missing scene_*.usd in demos output")

    video_files = sorted((demos_dir / "videos").glob("demo_*.mp4")) if (demos_dir / "videos").exists() else []
    videos_ok = len(video_files) >= expected_demos
    _record(checks, "video_count", videos_ok, f"expected>={expected_demos}, actual={len(video_files)}")
    if strict_artifacts and not videos_ok:
        errors.append(
            f"video count below expectation (expected>={expected_demos}, actual={len(video_files)})"
        )

    dataset_path = required_paths["dataset_hdf5"]
    if dataset_path.exists():
        try:
            demo_count = _hdf5_demo_count(dataset_path)
        except Exception as exc:
            demo_count = -1
            errors.append(f"failed reading {dataset_path}: {exc}")
        _record(checks, "hdf5_demo_count", demo_count >= expected_demos, f"expected>={expected_demos}, actual={demo_count}")
        if strict_artifacts and demo_count < expected_demos:
            errors.append(f"hdf5 demo count below expectation (expected>={expected_demos}, actual={demo_count})")

    if strict_provenance:
        provenance_sources = [
            required_paths["plan_bundle"],
            required_paths["demo_metadata"],
            required_paths["quality_report"],
            required_paths["artifact_manifest"],
        ]
        for path in provenance_sources:
            if not path.exists():
                continue
            try:
                payload = _load_json(path)
            except Exception as exc:
                errors.append(f"failed to parse JSON {path}: {exc}")
                continue
            payload_run_id = str(payload.get("run_id", "")).strip()
            ok = payload_run_id == run_id
            _record(checks, f"run_id_match:{path.name}", ok, f"expected={run_id} actual={payload_run_id}")
            if not ok:
                errors.append(f"run_id mismatch in {path}: expected={run_id} actual={payload_run_id}")

    report: Dict[str, Any] = {
        "layout_dir": str(layout_dir),
        "run_id": run_id,
        "expected_demos": expected_demos,
        "strict_artifact_contract": strict_artifacts,
        "strict_provenance": strict_provenance,
        "checks": checks,
        "errors": errors,
        "status": "pass" if not errors else "fail",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(str(report_path))
    return 0 if not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
