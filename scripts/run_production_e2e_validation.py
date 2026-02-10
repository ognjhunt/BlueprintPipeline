#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def run_command(command: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=capture_output,
    )


def run_command_json(command: List[str]) -> Dict[str, Any]:
    result = run_command(command, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse JSON output from: {' '.join(command)}\n{result.stdout}"
        ) from exc


def gcs_path(bucket: str, object_name: str) -> str:
    return f"gs://{bucket}/{object_name}"


def gcs_object_exists(bucket: str, object_name: str) -> bool:
    result = run_command(["gsutil", "-q", "stat", gcs_path(bucket, object_name)])
    return result.returncode == 0


def read_gcs_json(bucket: str, object_name: str) -> Dict[str, Any]:
    result = run_command(["gsutil", "cat", gcs_path(bucket, object_name)])
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to read {gcs_path(bucket, object_name)}: {result.stderr}"
        )
    return json.loads(result.stdout)


def run_workflow(
    workflow: str,
    region: str,
    payload: Dict[str, Any],
    timeout_seconds: int,
) -> Dict[str, Any]:
    run_response = run_command_json(
        [
            "gcloud",
            "workflows",
            "run",
            workflow,
            "--location",
            region,
            "--data",
            json.dumps(payload),
            "--format=json",
        ]
    )
    execution_name = run_response.get("name")
    if not execution_name:
        raise RuntimeError(f"Workflow run returned no execution name: {run_response}")

    execution_id = execution_name.split("/")[-1]
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        execution = run_command_json(
            [
                "gcloud",
                "workflows",
                "executions",
                "describe",
                execution_id,
                "--workflow",
                workflow,
                "--location",
                region,
                "--format=json",
            ]
        )
        state = execution.get("state")
        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return execution
        time.sleep(10)

    raise TimeoutError(f"Timed out waiting for workflow {workflow} execution {execution_id}")


def evaluate_validation_report(
    report: Dict[str, Any],
    min_pass_rate: float,
    min_average_score: float,
    allowed_backends: List[str],
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    mock_indicators: List[str] = []

    summary = report.get("summary", {})
    pass_rate = float(summary.get("pass_rate", 0.0) or 0.0)
    average_score = float(summary.get("average_score", 0.0) or 0.0)

    if pass_rate < min_pass_rate:
        errors.append(f"pass_rate {pass_rate:.2%} < {min_pass_rate:.2%}")
    if average_score < min_average_score:
        errors.append(f"average_score {average_score:.2f} < {min_average_score:.2f}")

    physics_validation = report.get("physics_validation", {})
    physx_used = bool(physics_validation.get("physx_used", False))
    non_physx_count = int(physics_validation.get("non_physx_episode_count", 0) or 0)
    backends = physics_validation.get("physics_backends", [])

    if not physx_used:
        errors.append("physx_used is false")
        mock_indicators.append("physics_validation.physx_used=false")

    if non_physx_count > 0:
        errors.append(f"non_physx_episode_count {non_physx_count} > 0")
        mock_indicators.append(f"non_physx_episode_count={non_physx_count}")

    unexpected_backends = [b for b in backends if b not in allowed_backends]
    if unexpected_backends:
        errors.append(f"unexpected physics backends: {', '.join(unexpected_backends)}")
        mock_indicators.append(
            f"unexpected physics backends: {', '.join(unexpected_backends)}"
        )

    episodes = report.get("episodes", [])
    dev_fallbacks = [ep for ep in episodes if ep.get("dev_only_fallback")]
    if dev_fallbacks:
        errors.append(f"dev_only_fallback episodes: {len(dev_fallbacks)}")
        mock_indicators.append(f"dev_only_fallback episodes: {len(dev_fallbacks)}")

    return errors, mock_indicators


def evaluate_quality_gate(report: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    summary = report.get("summary", {})
    if not summary.get("can_proceed", False):
        errors.append("quality gates blocked (summary.can_proceed=false)")
    blocking_failures = int(summary.get("blocking_failures", 0) or 0)
    if blocking_failures > 0:
        errors.append(f"blocking_failures {blocking_failures} > 0")
    return errors


def evaluate_import_manifest(report: Dict[str, Any], min_average_score: float) -> List[str]:
    errors: List[str] = []
    episodes = report.get("episodes", {})
    quality = report.get("quality", {})
    passed_validation = int(episodes.get("passed_validation", 0) or 0)
    average_score = float(quality.get("average_score", 0.0) or 0.0)
    scene_id = str(report.get("scene_id") or "").strip().lower()

    if passed_validation <= 0:
        errors.append("import_manifest episodes.passed_validation <= 0")
    if average_score < min_average_score:
        errors.append(
            f"import_manifest quality.average_score {average_score:.2f} < {min_average_score:.2f}"
        )
    if scene_id in {"", "unknown", "none", "null", "n/a"}:
        errors.append("import_manifest scene_id is null/unknown")

    return errors


def evaluate_production_validation(report: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not bool(report.get("production_mode", False)):
        errors.append("production_validation.production_mode is false")
    if not bool(report.get("ok", False)):
        errors.append("production_validation.ok is false")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Run production E2E validation workflows.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--min-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-average-score", type=float, default=0.85)
    parser.add_argument(
        "--allowed-physics-backends",
        default="isaac_sim,isaac_lab",
        help="Comma-separated list of allowed physics backends.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument(
        "--output-json",
        default="analysis_outputs/production_e2e_validation.json",
        help="Local output path for the harness summary JSON.",
    )
    args = parser.parse_args()

    allowed_backends = [b.strip() for b in args.allowed_physics_backends.split(",") if b.strip()]

    markers = {
        "variation_assets": f"scenes/{args.scene_id}/variation_assets/.variation_pipeline_complete",
        "usd_complete": f"scenes/{args.scene_id}/usd/.usd_complete",
        "regen3d_complete": f"scenes/{args.scene_id}/assets/.regen3d_complete",
    }

    for name, marker in markers.items():
        if not gcs_object_exists(args.bucket, marker):
            raise RuntimeError(f"Missing required marker for {name}: {gcs_path(args.bucket, marker)}")

    print("[E2E] Triggering Genie Sim export workflow...")
    run_workflow(
        "genie-sim-export-pipeline",
        args.region,
        {"data": {"bucket": args.bucket, "name": markers["variation_assets"]}},
        args.timeout_seconds,
    )

    geniesim_submitted = f"scenes/{args.scene_id}/geniesim/.geniesim_submitted"
    geniesim_complete = f"scenes/{args.scene_id}/geniesim/.geniesim_complete"
    geniesim_job_metadata = f"scenes/{args.scene_id}/geniesim/job.json"

    if not gcs_object_exists(args.bucket, geniesim_submitted):
        raise RuntimeError("Missing Genie Sim submitted marker.")
    if not gcs_object_exists(args.bucket, geniesim_complete):
        raise RuntimeError("Missing Genie Sim complete marker.")
    if not gcs_object_exists(args.bucket, geniesim_job_metadata):
        raise RuntimeError("Missing Genie Sim job metadata.")

    job_metadata = read_gcs_json(args.bucket, geniesim_job_metadata)
    job_id = job_metadata.get("job_id")
    if not job_id:
        raise RuntimeError("Genie Sim job metadata did not include job_id.")

    print("[E2E] Triggering episode generation workflow...")
    run_workflow(
        "episode-generation-pipeline",
        args.region,
        {"data": {"bucket": args.bucket, "name": markers["usd_complete"]}},
        args.timeout_seconds,
    )

    episodes_complete = f"scenes/{args.scene_id}/episodes/.episodes_complete"
    if not gcs_object_exists(args.bucket, episodes_complete):
        raise RuntimeError("Missing episodes completion marker.")

    validation_report_path = f"scenes/{args.scene_id}/episodes/quality/validation_report.json"
    quality_gate_report_path = (
        f"scenes/{args.scene_id}/episode-generation-job/quality_gate_report.json"
    )

    if not gcs_object_exists(args.bucket, validation_report_path):
        raise RuntimeError("Missing validation_report.json.")
    if not gcs_object_exists(args.bucket, quality_gate_report_path):
        raise RuntimeError("Missing quality_gate_report.json.")

    validation_report = read_gcs_json(args.bucket, validation_report_path)
    quality_gate_report = read_gcs_json(args.bucket, quality_gate_report_path)

    validation_errors, mock_indicators = evaluate_validation_report(
        validation_report,
        args.min_pass_rate,
        args.min_average_score,
        allowed_backends,
    )
    quality_gate_errors = evaluate_quality_gate(quality_gate_report)

    print("[E2E] Triggering Genie Sim import workflow...")
    run_workflow(
        "genie-sim-import-pipeline",
        args.region,
        {
            "job_id": job_id,
            "scene_id": args.scene_id,
            "min_quality_score": str(args.min_average_score),
            "enable_validation": "true",
            "filter_low_quality": "true",
            "wait_for_completion": "true",
        },
        args.timeout_seconds,
    )

    geniesim_import_complete = f"scenes/{args.scene_id}/geniesim/.geniesim_import_complete"
    if not gcs_object_exists(args.bucket, geniesim_import_complete):
        raise RuntimeError("Missing Genie Sim import completion marker.")
    import_manifest_path = f"scenes/{args.scene_id}/episodes/import_manifest.json"
    if not gcs_object_exists(args.bucket, import_manifest_path):
        raise RuntimeError("Missing import_manifest.json after Genie Sim import.")
    import_manifest = read_gcs_json(args.bucket, import_manifest_path)
    import_manifest_errors = evaluate_import_manifest(
        import_manifest,
        args.min_average_score,
    )

    print("[E2E] Triggering DWM preparation workflow...")
    run_workflow(
        "dwm-preparation-pipeline",
        args.region,
        {"data": {"bucket": args.bucket, "name": markers["regen3d_complete"]}},
        args.timeout_seconds,
    )

    dwm_complete = f"scenes/{args.scene_id}/dwm/.dwm_complete"
    if not gcs_object_exists(args.bucket, dwm_complete):
        raise RuntimeError("Missing DWM completion marker.")

    production_validation_path = f"scenes/{args.scene_id}/production_validation.json"
    production_validation = {}
    production_validation_errors: List[str] = []
    if gcs_object_exists(args.bucket, production_validation_path):
        production_validation = read_gcs_json(args.bucket, production_validation_path)
        production_validation_errors = evaluate_production_validation(production_validation)
    else:
        production_validation_errors = ["Missing production_validation.json"]

    summary = {
        "scene_id": args.scene_id,
        "outputs": {
            "geniesim_prefix": f"scenes/{args.scene_id}/geniesim",
            "episodes_prefix": f"scenes/{args.scene_id}/episodes",
            "import_manifest": import_manifest_path,
            "validation_report": validation_report_path,
            "quality_gate_report": quality_gate_report_path,
            "production_validation": production_validation_path,
            "dwm_prefix": f"scenes/{args.scene_id}/dwm",
        },
        "quality": {
            "pass_rate": validation_report.get("summary", {}).get("pass_rate"),
            "average_score": validation_report.get("summary", {}).get("average_score"),
            "physics_validation": validation_report.get("physics_validation", {}),
            "quality_gate_summary": quality_gate_report.get("summary", {}),
            "import_manifest": {
                "episodes_passed_validation": import_manifest.get("episodes", {}).get("passed_validation"),
                "average_score": import_manifest.get("quality", {}).get("average_score"),
                "scene_id": import_manifest.get("scene_id"),
            },
        },
        "production_validation": production_validation,
        "mock_fallbacks": mock_indicators,
    }

    failures = (
        validation_errors
        + quality_gate_errors
        + import_manifest_errors
        + production_validation_errors
    )
    status = "PASS" if not failures else "FAIL"

    print("\n[E2E] ===============================")
    print(f"[E2E] Final Status: {status}")
    print("[E2E] Summary:")
    print(json.dumps(summary, indent=2))
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(
            {
                "status": status,
                "summary": summary,
                "failures": failures,
            },
            handle,
            indent=2,
        )
    print(f"[E2E] Wrote summary JSON to {output_path}")

    if failures:
        print("\n[E2E] Failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    if mock_indicators:
        print("\n[E2E] Mock fallback indicators detected:")
        for indicator in mock_indicators:
            print(f"  - {indicator}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[E2E] ERROR: {exc}")
        sys.exit(1)
