# Workflow Timeout and Retry Policy

## Overview

This document defines the standardized timeout and retry policies for all BlueprintPipeline workflows.

## Retry Strategy

All critical API calls that start Cloud Run jobs use exponential backoff retry logic:

```yaml
retry:
  predicate: ${http.default_retry_predicate}
  max_retries: 5  # For Cloud Run job launches
  backoff:
    initial_delay: 1 second
    max_delay: 60 seconds
    multiplier: 2
```

### When Retries Apply

Retries are applied to transient failures:
- HTTP 408 (Request Timeout)
- HTTP 429 (Too Many Requests / Rate Limiting)
- HTTP 500, 502, 503, 504 (Server Errors)

### Retryable vs Non-Retryable Failures

**RETRYABLE:**
- GCS/API transient timeouts
- Cloud Run rate limiting
- Service unavailability (temporary)
- Network errors

**NON-RETRYABLE:**
- Invalid parameters (400, 422)
- Unauthorized (401, 403)
- Job logic errors (application-level failures)

## Timeout Policies by Pipeline

### Adaptive Timeout Alignment
- Workflows that accept `bundle_tier` or `scene_complexity` inputs should use the defaults in `policy_configs/adaptive_timeouts.yaml` unless overridden by the pipeline-specific values below.
- When a workflow does not receive adaptive inputs, we set explicit per-workflow defaults (often 3600s) to keep metrics consistent and document any overrides here.

### Episode Generation Pipeline
- **GKE Job Timeout:** 6 hours (21600s) - Isaac Sim GPU processing
- **Cloud Build Timeout:** 6 hours (21600s)
- **Workflow Execution:** No explicit timeout (use Cloud Run default)
- **Reason:** GPU-intensive Isaac Sim runs need extended time

### USD Assembly Pipeline

| Component | Timeout | Retry Strategy |
|-----------|---------|-----------------|
| Convert Job (GLB→USDZ) | 3600s (1 hour) | 5 retries, exponential backoff |
| Simready Job | 3600s (1 hour) | 5 retries, exponential backoff |
| USD Assembly Job | 3600s (1 hour) | 5 retries, exponential backoff |
| Replicator Job | 1800s (30 min) | 5 retries, exponential backoff |
| Isaac Lab Job | 1800s (30 min) | 5 retries, exponential backoff |

### Genie Sim Export Pipeline
- **Job Timeout:** 2700s (45 minutes)
- **Retry:** 5 retries with exponential backoff
- **Reason:** Requires asset processing and cuRobo trajectory planning

### Arena Export Pipeline
- **Job Timeout:** 1800s (30 minutes)
- **Retry:** 5 retries with exponential backoff
- **Reason:** Affordance detection and policy scoring

### Genie Sim Import Pipeline
- **Job Timeout:** 1800s (30 minutes) - explicit
- **Retry:** 5 retries with exponential backoff
- **Reason:** Episode validation and manifests are I/O intensive

### Objects Pipeline
- **Job Timeout:** 3600s (explicit Cloud Run default)
- **Retry:** 5 retries with exponential backoff
- **Reason:** Object detection is typically fast, non-critical, but explicit timeout supports metrics

### Scale Pipeline
- **Job Timeout:** 3600s (explicit Cloud Run default)
- **Retry:** 5 retries with exponential backoff
- **Reason:** Layout scaling is typically fast, non-critical, but explicit timeout supports metrics

### Dream2Flow Preparation Pipeline
- **Preparation Job Timeout:** 3600s (explicit Cloud Run default)
- **Inference Job Timeout:** 3600s (explicit Cloud Run default)
- **Retry:** 5 retries with exponential backoff (both jobs)
- **Reason:** 3D flow extraction and inference can be I/O intensive; explicit timeout supports metrics

### DWM Preparation Pipeline
- **Preparation Job Timeout:** 3600s (explicit Cloud Run default)
- **Inference Job Timeout:** 3600s (explicit Cloud Run default)
- **Retry:** 5 retries with exponential backoff (both jobs)
- **Reason:** Hand mesh video rendering and inference is compute-intensive; explicit timeout supports metrics

### Variation Assets Pipeline
- **Variation-Gen Job Timeout:** 1800s (explicit override)
- **Simready Job Timeout:** 3600s (explicit override)
- **Retry:** 5 retries with exponential backoff (both jobs)
- **Reason:** Domain randomization asset generation is I/O intensive

### Scene Generation Pipeline
- **Job Timeout:** 3600s (explicit Cloud Run default)
- **Retry:** 5 retries with exponential backoff
- **Reason:** Daily batch generation benefits from default timeout + metrics

### Interactive Pipeline
- **Job Timeout:** 1800s (explicit override)
- **Retry:** 5 retries with exponential backoff
- **Reason:** Interactive articulation extraction has bounded runtime

### Training Pipeline
- **Job Timeout:** 21600s (6 hours)
- **Retry:** 5 retries with exponential backoff
- **Reason:** Model training is long-running and CPU/GPU intensive

## Maintenance workflows

Maintenance workflows (for example, retention cleanup) should run in off-peak windows, use
Cloud Scheduler for invocation, and emit explicit start/complete metrics in Cloud Logging so
auditors can trace deletions back to workflow executions.

### Regen3D Pipeline
- **Job Timeout:** 3600s (1 hour) - explicit
- **Retry:** 5 retries with exponential backoff (job start); marker polling has exponential backoff
- **Marker Verification:** Exponential backoff polling, max 6 attempts, capped at 30s between attempts
- **Reason:** 3D reconstruction is compute-intensive, marker polling must handle eventual consistency

## Cloud Run Job Default Timeout

If not explicitly specified in a workflow, Cloud Run uses:
- **Default:** 3600s (1 hour)
- **Maximum:** 3600s (1 hour) for synchronous executions

## Workflow Execution Timeout

Google Cloud Workflows executions have a hard limit:
- **Maximum:** 24 hours per execution
- **Soft Timeout:** 6 hours (epic GPU jobs)

## Storage Operations

For GCS read/write operations (markers, configs):
- **No explicit timeout:** Uses Google API client library defaults (typically 60s per operation)
- **Implicit Retries:** HTTP client libraries automatically retry transient failures

## Rollback/Cleanup Pattern for Output Prefixes

Workflows that write outputs to a prefix **must** wrap critical job launches in `try`/`except` blocks and perform cleanup in the `except` path. The standard rollback pattern is:

1. **List outputs for the step prefix** (for example, `scenes/<scene_id>/episodes/`).
2. **Delete or quarantine partial outputs** using `googleapis.storage.v1.objects.delete` for each object under that prefix.
3. **Write a `.failed` marker** under the same output prefix (for example, `scenes/<scene_id>/episodes/.failed`) so downstream systems stop or alert.

Example pattern:

```yaml
- run_step_job:
    try:
      call: googleapis.run.v2.projects.locations.jobs.run
      args:
        name: ${jobName}
      result: jobExec
    except:
      as: e
      steps:
        - list_outputs_for_cleanup:
            call: googleapis.storage.v1.objects.list
            args:
              bucket: ${bucket}
              prefix: ${outputPrefix + "/"}
              maxResults: 1000
            result: cleanupList
        - delete_outputs_for_cleanup:
            for:
              value: cleanupItem
              in: ${default(cleanupList.items, [])}
              steps:
                - delete_output:
                    call: googleapis.storage.v1.objects.delete
                    args:
                      bucket: ${bucket}
                      object: ${cleanupItem.name}
        - write_failed_marker:
            call: googleapis.storage.v1.objects.insert
            args:
              bucket: ${bucket}
              name: ${outputPrefix + "/.failed"}
              uploadType: "media"
              body: ${json.encode({
                "status": "failed",
                "timestamp": time.format(sys.now()),
                "error": {"message": e.message}
              })}
        - raise_step_error:
            raise: ${e}
```

## Workflow Status Polling

When waiting for Cloud Run job completion:
- **Poll Interval:** 10 seconds (most pipelines)
- **Max Attempts:** Limited by workflow 24-hour hard timeout
- **Logic:** Fixed 10-second intervals for most job polling

**Special Cases with Exponential Backoff:**
- **Regen3D Marker Polling:** Exponential backoff 1s→2s→4s→8s→16s→30s (max 6 attempts)
  - Reason: Handles eventual consistency of GCS writes, avoids hammering storage
- **Simready Completion Polling:** Exponential backoff 5s→10s→20s→40s→60s (max 12 attempts, ~700s total)
  - Reason: Simready can be slow on large scenes, reduced polling frequency after initial attempts

## Timeout Calculation Examples

### Example 1: USD Assembly Pipeline Total Time
```
Convert Job (CONVERT_ONLY):
  - Execution: 5 minutes (typical)
  - Timeout: 3600s (1 hour) ✓ Safe

Simready Job:
  - Execution: 2 minutes (typical)
  - Timeout: 3600s (1 hour) ✓ Safe

USD Assembly Job:
  - Execution: 8 minutes (typical)
  - Timeout: 3600s (1 hour) ✓ Safe

Replicator Job:
  - Execution: 15 minutes (typical)
  - Timeout: 1800s (30 min) ⚠ TIGHT - May timeout on large scenes

Isaac Lab Job:
  - Execution: 8 minutes (typical)
  - Timeout: 1800s (30 min) ✓ Safe

Total Pipeline Runtime: ~30 minutes typical
Total Timeout Budget: ~4 hours (accounting for retries)
```

### Example 2: Retry Overhead
```
Max Retries: 5
Backoff Pattern:
  Attempt 1: Immediate failure
  Attempt 2: Wait 1s + retry
  Attempt 3: Wait 2s + retry
  Attempt 4: Wait 4s + retry
  Attempt 5: Wait 8s + retry
  Attempt 6: Wait 16s + retry (capped at 60s)

Total Retry Overhead: ~31 seconds (best case)
```

## Recommended Adjustments for Production

### For Large Scenes
- Increase Replicator timeout: 1800s → 3600s
- Increase Isaac Lab timeout: 1800s → 3600s

### For High Load
- Reduce max_retries: 5 → 3 (to fail faster and surface issues)
- Increase backoff multiplier: 2 → 1.5 (slower exponential growth)

### For Cost Optimization
- Increase initial_delay: 1s → 2s (longer waits between retries)
- Reduce max_retries: 5 → 3 (fewer wasted attempts)

## Monitoring and Alerts

### Recommended Metrics
1. **Workflow Execution Time:** Track actual runtime vs timeout
2. **Retry Success Rate:** Monitor how many retries are successful vs failures
3. **Timeout Occurrences:** Alert if any job hits timeout
4. **Error Categories:** Track retryable vs non-retryable failures
5. **Timeout Usage Ratio:** Duration / configured timeout per job invocation
6. **Retry Exhaustion:** Count of retry-exhausted job starts

### Alert Thresholds
- Execution time > 80% of timeout: WARNING
- Execution time > 95% of timeout: CRITICAL
- 3+ consecutive retries: WARNING

### Alert Configuration (Cloud Monitoring)
Alert policies and log-based metrics are defined in:
- `infrastructure/monitoring/alerts/alert-policies.yaml`
- `infrastructure/monitoring/metrics/*.yaml`

Log-based metrics are emitted from workflows with structured logging fields:
- `bp_metric: "job_invocation"` with `timeout_seconds`, `duration_seconds`, `timeout_usage_ratio`
- `bp_metric: "job_retry_exhausted"` for retry spikes

### Adaptive Timeout Overrides
Adaptive timeout defaults live in `policy_configs/adaptive_timeouts.yaml`.
Workflows can override defaults by passing `timeout_override_seconds` in the
workflow event payload. Current usage:
- `upsell-features-pipeline.yaml` scales timeout by `bundle_tier` and accepts
  `timeout_override_seconds` for manual overrides.

## Policy Updates

This policy was last updated: **2026-01-11**

Future updates should consider:
- Cloud Run performance improvements (may allow shorter timeouts)
- New job types being added to the pipeline
- Historical execution time metrics from production data
- Cost optimization requirements

## Implementation Status

✓ **Implemented (2026-01-11):**
- Exponential backoff retry logic on critical job launches (5 retries: 1s initial, 60s max)
- Standardized timeout values for each job
- Retry policy documentation with detailed specifications
- Added retry logic to: dream2flow (both jobs), dwm (both jobs), variation-assets (both jobs), objects, scale, regen3d-import
- Exponential backoff marker polling: regen3d (1s→30s) and simready (5s→60s)
- Environment variable standardization (GOOGLE_CLOUD_PROJECT_ID)
- .assets_ready marker producer (scale-pipeline)
- Race condition documentation for parallel pipeline execution
- Non-fatal failure clarification in arena-export-pipeline
- Structured workflow job invocation metrics (timeout usage + retry exhaustion)
- Log-based Cloud Monitoring metrics + alert policy definitions
- Adaptive timeout defaults with bundle-tier override support

⚠️ **Partial:**
- Replicator/Isaac Lab jobs may need timeout adjustment for large scenes (future monitoring needed)
- Historical performance tracking and analysis system (metrics collected but analysis dashboard not complete)

✅ **Completed (Post-2026-01-11):**
- Automatic timeout adjustment based on scene complexity (implemented in `policy_configs/adaptive_timeouts.yaml`)
- Support for different retry policies per customer tier (implemented in `upsell-features-job/customer_config.py` with BundleTier system)
