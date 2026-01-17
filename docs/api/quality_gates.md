# `tools/quality_gates` API

## Purpose

`tools/quality_gates` implements quality gates across pipeline checkpoints, supports human approvals for blocking failures, and emits notifications. It is used to enforce data quality SLIs/SLOs and block pipeline progression when critical metrics fall below thresholds.【F:tools/quality_gates/__init__.py†L1-L24】【F:tools/quality_gates/quality_gate.py†L1-L40】

## Public entrypoints

Exported via `tools.quality_gates`:

- Core gate types:
  - `QualityGate`, `QualityGateResult`, `QualityGateSeverity`, `QualityGateCheckpoint`
  - `QualityGateRegistry`
- Human-in-the-loop helpers:
  - `NotificationService`, `NotificationChannel`
  - `send_email_notification`, `send_sms_notification`
- AI QA context:
  - `QAContextGenerator`, `generate_qa_context`【F:tools/quality_gates/__init__.py†L7-L30】

CLI entrypoint:

- `tools/quality_gates/sli_gate_runner.py` provides the production SLI gate runner (`main`) for episode pipelines.【F:tools/quality_gates/sli_gate_runner.py†L1-L39】【F:tools/quality_gates/sli_gate_runner.py†L122-L210】

## Configuration / environment variables

Quality gate thresholds can be configured through `tools/quality_gates/quality_config.json` and overridden via environment variables prefixed with `BP_QUALITY_*` in the quality gate implementation. Examples include:

- `BP_QUALITY_PHYSICS_MASS_MIN_KG`, `BP_QUALITY_PHYSICS_MASS_MAX_KG`
- `BP_QUALITY_EPISODES_COLLISION_FREE_RATE_MIN`
- `BP_QUALITY_EPISODES_QUALITY_PASS_RATE_MIN`
- `BP_QUALITY_EPISODES_QUALITY_SCORE_MIN`
- `BP_QUALITY_EPISODES_MIN_EPISODES_REQUIRED`
- `BP_QUALITY_AVG_SCORE_MIN`
- `BP_QUALITY_SENSOR_CAPTURE_RATE_MIN`
- `BP_QUALITY_PHYSICS_VALIDATION_RATE_MIN`
- `BP_QUALITY_EPISODES_TIER_THRESHOLDS` (JSON override)

The production SLI runner also supports runtime env configuration:

- `SCENE_ID`, `BUCKET`, `EPISODES_PREFIX`, `DATA_ROOT`, `SCENE_DIR`
- `QUALITY_GATE_REPORT_PATH` for saving gate reports to disk【F:tools/quality_gates/quality_gate.py†L1-L41】【F:tools/quality_gates/quality_gate.py†L1026-L1517】【F:tools/quality_gates/sli_gate_runner.py†L153-L176】

## Request/response payloads & data models

### Gate registration and results

- **Request**: `QualityGateRegistry.run_checkpoint(checkpoint, context)` accepts a `QualityGateCheckpoint` and a `context` dictionary containing metrics (e.g., episode stats, physics rates).【F:tools/quality_gates/quality_gate.py†L1155-L1209】
- **Response**: each `QualityGate` returns a `QualityGateResult` with pass/fail flags, severity, details, and recommendations; blocking failures set `requires_human_approval=True`.【F:tools/quality_gates/quality_gate.py†L68-L99】

### Human approval workflow

- `HumanApprovalManager` creates `ApprovalRequest` records for failed gates and persists them under `/tmp/blueprintpipeline/approvals` for external review. Approvals or overrides update the request status and allow the pipeline to proceed if authorized.【F:tools/quality_gates/quality_gate.py†L102-L190】

### SLI gate runner payloads

The SLI runner loads:

- `dataset_quality_manifest.json`
- `manifests/generation_manifest.json`

It then constructs a `quality_context` payload with `episode_stats` and `data_quality` sections, runs gates for `QualityGateCheckpoint.EPISODES_GENERATED`, and optionally writes a report file to `QUALITY_GATE_REPORT_PATH`.【F:tools/quality_gates/sli_gate_runner.py†L70-L139】【F:tools/quality_gates/sli_gate_runner.py†L178-L209】

## Example usage

```python
from tools.quality_gates import QualityGateCheckpoint, QualityGateRegistry

registry = QualityGateRegistry(verbose=True)
registry.run_checkpoint(
    checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
    context={
        "episode_stats": {
            "total_generated": 100,
            "passed_quality_filter": 90,
            "average_quality_score": 0.92,
            "collision_free_rate": 0.88,
        },
        "data_quality": {
            "sensor_capture_rate": 0.95,
            "sensor_sources": ["replicator"],
            "physics_validation_rate": 0.93,
            "physics_backends": ["physx"],
        },
    },
)

if not registry.can_proceed():
    registry.save_report(scene_id="kitchen_001", output_path="/tmp/gate_report.json")
```

