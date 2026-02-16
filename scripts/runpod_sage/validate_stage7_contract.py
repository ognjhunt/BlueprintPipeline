#!/usr/bin/env python3
"""
Validate Stage 7 artifact/provenance contract for runpod SAGE pipeline outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _hdf5_run_id(path: Path) -> str:
    import h5py

    with h5py.File(str(path), "r") as f:
        metadata = f.get("metadata")
        if metadata is None:
            return ""
        provenance = metadata.get("provenance")
        if provenance is None:
            return ""
        attr_run_id = str(provenance.attrs.get("run_id", "")).strip()
        if attr_run_id:
            return attr_run_id
        ds = provenance.get("run_id")
        if ds is None:
            return ""
        try:
            value = ds[()]
        except Exception:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore").strip()
        return str(value).strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _manifest_file_entries(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    files = payload.get("files")
    if not isinstance(files, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in files:
        if isinstance(item, dict):
            out.append(item)
    return out


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
    expected_video_names = {f"demo_{i}.mp4" for i in range(expected_demos)}
    actual_video_names = {p.name for p in video_files}
    missing_videos = sorted(expected_video_names - actual_video_names)
    unexpected_videos = sorted(actual_video_names - expected_video_names)
    videos_exact = (not missing_videos) and (not unexpected_videos)
    _record(
        checks,
        "video_set_exact",
        videos_exact,
        f"expected={sorted(expected_video_names)} actual={sorted(actual_video_names)}",
    )
    if strict_artifacts and not videos_exact:
        errors.append(
            "video set mismatch "
            f"(missing={missing_videos} unexpected={unexpected_videos})"
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
        try:
            h5_run_id = _hdf5_run_id(dataset_path)
        except Exception as exc:
            h5_run_id = ""
            errors.append(f"failed reading hdf5 run_id from {dataset_path}: {exc}")
        h5_run_id_ok = str(h5_run_id).strip() == run_id
        _record(checks, "hdf5_run_id_match", h5_run_id_ok, f"expected={run_id} actual={h5_run_id}")
        if strict_provenance and not h5_run_id_ok:
            errors.append(f"hdf5 run_id mismatch: expected={run_id} actual={h5_run_id}")

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

        manifest_payload: Optional[Dict[str, Any]] = None
        manifest_path = required_paths["artifact_manifest"]
        if manifest_path.exists():
            try:
                manifest_payload = _load_json(manifest_path)
            except Exception as exc:
                errors.append(f"failed to parse JSON {manifest_path}: {exc}")
        if manifest_payload is not None:
            entries = _manifest_file_entries(manifest_payload)
            if not entries:
                errors.append(f"artifact manifest contains no files: {manifest_path}")
            manifest_video_rel = set()
            manifest_scene_rel = set()
            for item in entries:
                rel = str(item.get("path", "")).strip()
                if not rel:
                    continue
                abs_path = demos_dir / rel
                ok_exists = abs_path.exists() and abs_path.is_file()
                _record(checks, f"manifest_exists:{rel}", ok_exists, str(abs_path))
                if not ok_exists:
                    errors.append(f"manifest references missing file: {abs_path}")
                    continue
                expected_size = item.get("size_bytes")
                if isinstance(expected_size, int):
                    size_ok = int(abs_path.stat().st_size) == int(expected_size)
                    _record(
                        checks,
                        f"manifest_size:{rel}",
                        size_ok,
                        f"expected={expected_size} actual={int(abs_path.stat().st_size)}",
                    )
                    if not size_ok:
                        errors.append(f"size mismatch for {abs_path}: expected={expected_size} actual={int(abs_path.stat().st_size)}")
                expected_sha = str(item.get("sha256", "")).strip().lower()
                if expected_sha:
                    actual_sha = _sha256(abs_path)
                    sha_ok = actual_sha.lower() == expected_sha
                    _record(checks, f"manifest_sha256:{rel}", sha_ok, f"expected={expected_sha} actual={actual_sha}")
                    if not sha_ok:
                        errors.append(f"sha256 mismatch for {abs_path}: expected={expected_sha} actual={actual_sha}")
                rel_norm = rel.replace("\\", "/")
                if rel_norm.startswith("videos/") and rel_norm.endswith(".mp4"):
                    manifest_video_rel.add(rel_norm)
                if rel_norm.startswith("scene_") and rel_norm.endswith(".usd"):
                    manifest_scene_rel.add(rel_norm)

            actual_video_rel = {f"videos/{name}" for name in actual_video_names}
            actual_scene_rel = {p.name for p in scene_usds}
            video_manifest_ok = actual_video_rel == manifest_video_rel
            scene_manifest_ok = actual_scene_rel == manifest_scene_rel
            _record(
                checks,
                "manifest_video_set_exact",
                video_manifest_ok,
                f"manifest={sorted(manifest_video_rel)} actual={sorted(actual_video_rel)}",
            )
            _record(
                checks,
                "manifest_scene_set_exact",
                scene_manifest_ok,
                f"manifest={sorted(manifest_scene_rel)} actual={sorted(actual_scene_rel)}",
            )
            if strict_artifacts and not video_manifest_ok:
                errors.append(
                    "artifact manifest video set mismatch "
                    f"(manifest={sorted(manifest_video_rel)} actual={sorted(actual_video_rel)})"
                )
            if strict_artifacts and not scene_manifest_ok:
                errors.append(
                    "artifact manifest scene set mismatch "
                    f"(manifest={sorted(manifest_scene_rel)} actual={sorted(actual_scene_rel)})"
                )

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
