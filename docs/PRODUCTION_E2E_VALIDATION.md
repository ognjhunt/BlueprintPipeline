# Production E2E Validation Runbook

## Purpose
Run a full production end-to-end validation for a scene, including workflow execution, artifact checks, and quality thresholds. This runbook is designed for production-only runs and explicitly prohibits mock capture fallbacks.

## Required Inputs
| Input | Description | Example |
| --- | --- | --- |
| `scene_id` | Scene identifier | `kitchen_001` |
| `bucket` | GCS bucket hosting scene artifacts | `blueprintpipeline-data` |
| `project_id` | Google Cloud project | `my-prod-project` |
| `region` | Workflow region | `us-central1` |
| `variation_assets_marker` | GCS marker for Genie Sim export trigger | `scenes/<scene_id>/variation_assets/.variation_pipeline_complete` |
| `usd_marker` | GCS marker for episode generation trigger | `scenes/<scene_id>/usd/.usd_complete` |
| `regen3d_marker` | GCS marker for DWM prep trigger | `scenes/<scene_id>/assets/.regen3d_complete` |

## Required Production Flags (No Mock Fallbacks)
These flags must be enforced for production validation runs:

### Episode Generation (Isaac Sim)
- `DATA_QUALITY_LEVEL=production`
- `REQUIRE_REAL_PHYSICS=true`
- `ISAAC_SIM_REQUIRED=true`
- `SENSOR_CAPTURE_MODE=isaac_sim`
- `USE_MOCK_CAPTURE=false`
- `ALLOW_MOCK_DATA=false`
- `ALLOW_MOCK_CAPTURE=false`

### Genie Sim Import Validation
- `MIN_QUALITY_SCORE=0.85` (or stricter)
- `ENABLE_VALIDATION=true`
- `FILTER_LOW_QUALITY=true`

### Genie Sim Local Execution
- `GENIESIM_VALIDATE_FRAMES=1`
- `GENIESIM_FAIL_ON_FRAME_VALIDATION=1`

## Expected Artifacts & Metrics by Stage

### 1) Genie Sim Export
**Workflow:** `genie-sim-export-pipeline`

**Inputs**
- Marker: `scenes/<scene_id>/variation_assets/.variation_pipeline_complete`

**Expected GCS Artifacts**
- `scenes/<scene_id>/geniesim/.geniesim_submitted`
- `scenes/<scene_id>/geniesim/.geniesim_complete`
- `scenes/<scene_id>/geniesim/job.json`

**Notes**
- If `use_geniesim=false` in `scenes/<scene_id>/config.json`, the workflow will skip.

### 2) Episode Generation (Isaac Sim)
**Workflow:** `episode-generation-pipeline`

**Inputs**
- Marker: `scenes/<scene_id>/usd/.usd_complete`

**Expected GCS Artifacts**
- `scenes/<scene_id>/episodes/.episodes_complete`
- `scenes/<scene_id>/episodes/quality/validation_report.json`
- `scenes/<scene_id>/episode-generation-job/quality_gate_report.json`

**Validation Metrics**
- `validation_report.json`:
  - `summary.pass_rate >= 0.90`
  - `summary.average_score >= 0.85`
  - `physics_validation.physx_used == true`
  - `physics_validation.non_physx_episode_count == 0`
- `quality_gate_report.json`:
  - `summary.can_proceed == true`
  - `summary.blocking_failures == 0`

### Quality Gate Preflight Criteria (Downstream Consumers)
Downstream workflows that consume episode bundles (training, import, upsell) must pass the quality gate preflight before proceeding. The preflight checks:

1. **Primary (recommended):** `quality_gate_report.json`
   - `summary.can_proceed` must be `true`.
   - `summary.blocking_failures` must be `0`.
2. **Fallback (legacy runs without a gate report):** per-episode `quality_certificate.json`
   - `validation_passed` must be `true`.
   - `recommended_use` must **not** be `testing` (accepted: `production_training`, `fine_tuning`).

If the report or certificate fails these criteria, the downstream workflow writes a `.failed` marker and exits with an error that includes the report path.

#### Expected `quality_gate_report.json` Schema (summary)
```json
{
  "scene_id": "kitchen_001",
  "timestamp": "2024-05-08T12:34:56Z",
  "summary": {
    "total_gates": 12,
    "passed": 11,
    "failed": 1,
    "blocking_failures": 1,
    "can_proceed": false
  },
  "results": [
    {
      "gate_id": "QG-EPISODES-GENERATED",
      "checkpoint": "episodes_generated",
      "passed": false,
      "severity": "error",
      "message": "Blocking failure details...",
      "details": {},
      "recommendations": [],
      "requires_human_approval": true,
      "timestamp": "2024-05-08T12:34:56Z"
    }
  ]
}
```

#### Expected `quality_certificate.json` Fields (fallback)
```json
{
  "episode_id": "episode_000001",
  "scene_id": "kitchen_001",
  "validation_passed": true,
  "recommended_use": "production_training",
  "overall_quality_score": 0.92
}
```

### Golden Dataset Regression (Quality Drift Guard)
Use the pinned golden dataset fixture to detect regressions in episode counts, quality distributions, and checksum
consistency. The regression test enforces drift thresholds so CI fails when metrics deviate from the baseline.

**How to run:**
- `pytest tests/test_dataset_regression.py`
- Metrics and thresholds live under `tests/fixtures/golden/dataset_regression/`:
  - `baseline_metrics.json` (pinned expected metrics)
  - `thresholds.json` (allowed drift)

**Manual metric generation (optional):**
```bash
python -m tools.dataset_regression.metrics --dataset-dir <bundle_dir>
```
This uses the dataset recordings and checksums (or `import_manifest.json`) to summarize collision rates, quality
scores, and episode durations.

### 3) Sim Validation (Genie Sim Import)
**Workflow:** `genie-sim-import-pipeline`

**Inputs**
- Genie Sim `job_id` from `scenes/<scene_id>/geniesim/job.json`

**Expected GCS Artifacts**
- `scenes/<scene_id>/geniesim/.geniesim_import_complete`
- `scenes/<scene_id>/episodes/import_manifest.json`

**Validation Metrics** (from `import_manifest.json`)
- `quality.average_score >= 0.85`
- `quality.threshold == min_quality_score`
- `episodes.passed_validation > 0`

### 4) DWM Preparation
**Workflow:** `dwm-preparation-pipeline`

**Inputs**
- Marker: `scenes/<scene_id>/assets/.regen3d_complete`

**Expected GCS Artifacts**
- `scenes/<scene_id>/dwm/.dwm_complete`

## Execution (Harness Script)
Run the production harness to execute workflows in sequence and validate outputs:

```bash
python scripts/run_production_e2e_validation.py \
  --project-id <project_id> \
  --region us-central1 \
  --bucket <bucket> \
  --scene-id <scene_id>
```

## Pass/Fail Criteria
A run is **PASS** when all of the following are true:
- All required GCS markers exist after each stage.
- `quality_gate_report.json` shows no blocking failures.
- `validation_report.json` meets pass-rate and average-score thresholds.
- No mock capture fallbacks detected (non-PhysX backend, dev-only fallback, or mock indicators).

A run is **FAIL** if any marker is missing, a threshold is violated, or mock fallback indicators are found.
