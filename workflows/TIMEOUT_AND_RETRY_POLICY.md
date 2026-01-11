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

### Episode Generation Pipeline
- **GKE Job Timeout:** 6 hours (21600s) - Isaac Sim GPU processing
- **Cloud Build Timeout:** 6 hours (21600s)
- **Workflow Execution:** No explicit timeout (use Cloud Run default)
- **Reason:** GPU-intensive Isaac Sim runs need extended time

### USD Assembly Pipeline

| Component | Timeout | Retry Strategy |
|-----------|---------|-----------------|
| Convert Job (GLB→USDZ) | 3600s (1 hour) | 5 retries, exponential backoff |
| Simready Job | 3600s (1 hour) | 3 retries, exponential backoff |
| USD Assembly Job | 3600s (1 hour) | 3 retries, exponential backoff |
| Replicator Job | 1800s (30 min) | 3 retries, exponential backoff |
| Isaac Lab Job | 1800s (30 min) | 3 retries, exponential backoff |

### Genie Sim Export Pipeline
- **Job Timeout:** 2700s (45 minutes)
- **Retry:** 5 retries with exponential backoff
- **Reason:** Requires asset processing and cuRobo trajectory planning

### Arena Export Pipeline
- **Job Timeout:** 1800s (30 minutes)
- **Retry:** 3 retries with exponential backoff
- **Reason:** Affordance detection and policy scoring

### Genie Sim Import Pipeline
- **Job Timeout:** 1800s (30 minutes) - explicit
- **Retry:** 3 retries with exponential backoff
- **Reason:** Episode validation and manifests are I/O intensive

### Objects Pipeline
- **Job Timeout:** No explicit timeout (uses Cloud Run default)
- **Retry:** 3 retries with exponential backoff
- **Reason:** Object detection is typically fast, non-critical

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

## Workflow Status Polling

When waiting for Cloud Run job completion:
- **Poll Interval:** 10 seconds
- **Max Attempts:** Limited by workflow 24-hour hard timeout
- **Logic:** Exponential backoff NOT used for polling (fixed 10-second intervals)

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

### Alert Thresholds
- Execution time > 80% of timeout: WARNING
- Execution time > 95% of timeout: CRITICAL
- 3+ consecutive retries: WARNING

## Policy Updates

This policy was last updated: **2026-01-11**

Future updates should consider:
- Cloud Run performance improvements (may allow shorter timeouts)
- New job types being added to the pipeline
- Historical execution time metrics from production data
- Cost optimization requirements

## Implementation Status

✓ **Implemented:**
- Exponential backoff retry logic on critical job launches
- Standardized timeout values for each job
- Retry policy documentation

⚠️ **Partial:**
- Replicator/Isaac Lab jobs may need timeout adjustment for large scenes
- DWM and Dream2Flow pipelines still need timeout review

📋 **TODO:**
- Add monitoring/alerting for timeout events
- Historical performance tracking
- Automatic timeout adjustment based on scene complexity
