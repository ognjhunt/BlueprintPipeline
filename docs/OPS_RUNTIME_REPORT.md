# Ops Report: Pipeline Runtime & Timeout Audit

## Objective

Validate end-to-end pipeline runtime behavior for large, multi-robot scenes, capture per-stage runtime distributions/timeout usage from `bp_metric` logs, and tune workflow timeouts if stages regularly exceed 80–95% of configured limits.

## Environment Constraints

- This repo does not include production `bp_metric` log exports or scene catalogs with object counts/robot configuration metadata.
- The current environment does not provide access to the production execution platform to run full pipeline workloads or to query Cloud Logging for `bp_metric` distributions.

## Scene Selection Criteria (Pending)

Target scenes should satisfy **both** conditions:

- High object count (e.g., 150+ objects in `scene_manifest.json`).
- Multi-robot configuration (two or more robots in simulation/episode configs).

## Candidate Scene Inventory (Pending)

A production scene inventory is required to select representative scenes. Capture these inputs before re-running the audit:

- `scene_manifest.json` metadata with object counts.
- Robot configuration per scene (episode/agent configs).

## Pipeline Execution (Not Run)

No production-like runs were executed in this environment. Once scene inventory is available, run each selected scene through the full pipeline and capture:

- Per-stage runtime distributions from `bp_metric: job_invocation` logs (duration, timeout usage ratio).
- Timeout usage ratios and retry exhaustion events from `bp_metric: job_retry_exhausted`.

## Runtime Metrics Summary (Not Available)

| Stage | Timeout (s) | P50 Duration (s) | P90 Duration (s) | P95 Duration (s) | P99 Duration (s) | P95 Timeout Usage | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | Awaiting production log export |

## Timeout Adjustments (Not Applicable)

No timeout changes were made because runtime distributions and timeout usage ratios were unavailable.

## Next Actions

1. Export `bp_metric` logs for the chosen scenes covering `job_invocation` and `job_retry_exhausted` events.
2. Populate the runtime distribution table above and identify any stages exceeding 80–95% of configured timeouts.
3. Update the affected workflow YAML timeouts and `workflows/TIMEOUT_AND_RETRY_POLICY.md` accordingly.
4. Re-run this report with before/after metrics.
