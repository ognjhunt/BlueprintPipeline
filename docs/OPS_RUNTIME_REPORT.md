# Ops Report: Pipeline Runtime & Timeout Audit

## Objective

Validate end-to-end pipeline runtime behavior for production-like scenes, capture
per-stage runtime distributions/timeout usage from `bp_metric` logs, and tune
timeouts when stages consistently approach limits.

## Inputs

- Exported `bp_metric` JSONL logs containing at minimum:
  - `job_invocation` events with stage/job + duration + configured timeout.
  - `job_retry_exhausted` (or equivalent timeout exhaustion) events.

## Generation Command

```bash
python tools/quality_gates/runtime_slo_summary.py \
  --inputs /path/to/bp_metric_export.jsonl \
  --output analysis_outputs/runtime_slo_summary.json \
  --update-ops-doc
```

If `bp_metric` export is unavailable (for example during active runpod debugging),
capture a coarse health signal directly from the run log:

```bash
python tools/quality_gates/runpod_log_health_summary.py \
  --log-path /path/to/pipeline_run.log \
  --output analysis_outputs/triage/run_log_health_summary.json
```

## Runtime Metrics Summary

<!-- RUNTIME_TABLE_START -->
| Stage | Timeout (s) | P50 Duration (s) | P90 Duration (s) | P95 Duration (s) | P99 Duration (s) | P95 Timeout Usage | Timeout Exhausted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
<!-- RUNTIME_TABLE_END -->

## Interpretation Rules

- `P95 timeout usage >= 0.80`: candidate for timeout tuning or optimization.
- `P95 timeout usage >= 0.95`: high risk of timeout failures.
- Any non-zero timeout exhaustion count requires root cause review.

## Next Actions

1. Export `bp_metric` logs for representative release scenes.
2. Re-run `runtime_slo_summary.py` and update this document table.
3. Tune workflow timeout settings and rerun to capture before/after evidence.
