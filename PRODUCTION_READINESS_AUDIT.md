# BlueprintPipeline Production Readiness Audit

**Document Version:** 1.0
**Date:** January 10, 2026
**Scope:** Genie Sim 3.0 Default Pipeline - End-to-End Analysis

---

## Executive Summary

This document provides a comprehensive audit of the BlueprintPipeline codebase, identifying gaps and areas for improvement needed before going to production or testing with labs. The analysis covers the default Genie Sim 3.0 pipeline from image input through episode generation and LeRobot export.

### Overall Assessment

| Category | Status | Critical Gaps | Notes |
|----------|--------|---------------|-------|
| **Core Pipeline** | 🟡 Mostly Ready | 3 | Genie Sim client integration needs real endpoint testing |
| **Error Handling** | 🟢 Production Ready | 0 | Retry, circuit breaker, timeouts implemented |
| **Quality Gates** | 🟡 Mostly Ready | 2 | Missing Genie Sim-specific validation |
| **Testing** | 🔴 Needs Work | 5 | E2E tests exist but no integration tests with real services |
| **Monitoring/Observability** | 🔴 Needs Work | 4 | Metrics infrastructure exists but not wired |
| **Security** | 🟢 Production Ready | 0 | Secret Manager, input validation implemented |
| **Documentation** | 🟡 Mostly Ready | 2 | API docs incomplete |

### Priority Summary

- **P0 (Critical/Blockers):** 8 items - Must fix before any lab testing
- **P1 (High Priority):** 12 items - Should fix before production
- **P2 (Medium Priority):** 15 items - Important for robustness
- **P3 (Low Priority):** 10 items - Nice to have

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [P0 Critical Gaps](#2-p0-critical-gaps---blockers)
3. [P1 High Priority Gaps](#3-p1-high-priority-gaps)
4. [P2 Medium Priority Gaps](#4-p2-medium-priority-gaps)
5. [P3 Low Priority Gaps](#5-p3-low-priority-gaps)
6. [Testing Gaps Analysis](#6-testing-gaps-analysis)
7. [Operational Readiness](#7-operational-readiness)
8. [Integration Completeness](#8-integration-completeness)
9. [Recommendations](#9-recommendations)

---

## 1. Architecture Overview

### Current Pipeline Flow (Genie Sim 3.0 Mode)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BLUEPRINT PIPELINE - GENIE SIM 3.0 MODE               │
└─────────────────────────────────────────────────────────────────────────────┘

    INPUT IMAGE
         │
         ▼
┌─────────────────┐
│   3D-RE-GEN     │ ◄── External service (not released)
│   (External)    │     Produces: GLB meshes + poses + scene_info.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  regen3d-job    │     Adapter: Converts 3D-RE-GEN output to canonical manifest
│                 │     Output: scene_manifest.json, inventory.json
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────────┐
│simready│ │interactive │     Parallel: Physics prep + Articulation detection
│  job   │ │    job     │
└───┬────┘ └─────┬──────┘
    │            │
    └─────┬──────┘
          ▼
┌─────────────────┐
│usd-assembly-job │     Combines physics prims + articulations into scene.usda
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ replicator-job  │     Domain randomization: placement regions, variation policies
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│genie-sim-export │     DEFAULT PATH: Export to Genie Sim 3.0 format
│      job        │     Output: scene_graph.json, asset_index.json, task_config.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   GENIE SIM 3.0 │ ◄── External service (NVIDIA)
│   (External)    │     Generates: Physics-validated trajectories, sensor data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│genie-sim-import │     Import: Validates and converts Genie Sim episodes
│      job        │     Output: LeRobot v0.3.3 format episodes
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│arena-export-job │     Optional: Export to Arena/LeRobot Hub
│   (Optional)    │     Output: Hub config, scene_module.py, task definitions
└────────┬────────┘
         │
         ▼
    OUTPUT: LeRobot Episodes + Arena Scene
```

### Key Files Analyzed

| Component | File | Status |
|-----------|------|--------|
| Export to Genie Sim | `genie-sim-export-job/export_to_geniesim.py` | ✅ Complete |
| Genie Sim Client | `genie-sim-export-job/geniesim_client.py` | 🟡 Needs real testing |
| Import from Genie Sim | `genie-sim-import-job/import_from_geniesim.py` | ✅ Complete |
| Episode Generation | `episode-generation-job/generate_episodes.py` | ✅ Complete (fallback) |
| Quality Gates | `tools/quality_gates/quality_gate.py` | 🟡 Missing Genie gates |
| Error Handling | `tools/error_handling/retry.py` | ✅ Production ready |
| Arena Export | `tools/arena_integration/arena_exporter.py` | ✅ Complete |
| Local Pipeline | `tools/run_local_pipeline.py` | ✅ Complete |
| E2E Tests | `tests/test_pipeline_e2e.py` | 🟡 Mock-only |

---

## 2. P0 Critical Gaps - Blockers

These must be fixed before any testing with labs.

### P0-1: Genie Sim API Endpoint Configuration is Hardcoded Placeholder

**Location:** `genie-sim-export-job/geniesim_client.py:27-29`

**Current State:**
```python
GENIESIM_API_URL = os.environ.get(
    "GENIESIM_API_URL",
    "https://api.geniesim.nvidia.com/v1"  # Placeholder
)
```

**Problem:** The API URL is a placeholder. There's no validation that the real Genie Sim 3.0 API endpoint is configured, and no health check before submitting jobs.

**Required Fix:**
1. Add startup validation that `GENIESIM_API_URL` is set and reachable
2. Implement health check endpoint call before job submission
3. Add configuration for sandbox vs production environments
4. Document required environment variables

**Impact:** Pipeline will fail silently or hang when trying to connect to Genie Sim

---

### P0-2: Missing Genie Sim Job Polling Timeout

**Location:** `genie-sim-export-job/geniesim_client.py:148-180`

**Current State:**
```python
def wait_for_completion(self, job_id: str, ...) -> GenieSim JobResult:
    while True:
        status = self.get_job_status(job_id)
        if status.state in ["COMPLETED", "FAILED", "CANCELLED"]:
            break
        time.sleep(poll_interval)
```

**Problem:** The polling loop has no maximum timeout. If Genie Sim hangs or returns unexpected states, the pipeline will poll forever.

**Required Fix:**
1. Add `max_wait_time` parameter (e.g., 4 hours default for large jobs)
2. Implement timeout exception with clear error message
3. Add exponential backoff to polling interval
4. Track total elapsed time and abort if exceeded

**Impact:** Pipeline can hang indefinitely, blocking resources

---

### P0-3: No Genie Sim Episode Validation Before Import

**Location:** `genie-sim-import-job/import_from_geniesim.py:89-150`

**Current State:** Episodes are downloaded and converted but there's no validation that:
- Episode data is complete (all required fields present)
- Trajectory data is physics-plausible
- Sensor data dimensions match expectations
- No NaN/Inf values in observations or actions

**Required Fix:**
1. Add `validate_episode()` function with checks:
   - Required fields: `observations`, `actions`, `rewards`, `dones`
   - Observation shapes match robot config
   - Action bounds within robot limits
   - No NaN/Inf values
   - Timestamps are monotonic
2. Implement episode rejection with detailed error reports
3. Add quality certificate generation for each episode

**Impact:** Corrupted or invalid episodes will propagate to training, causing silent failures

---

### P0-4: Missing Isaac Sim Availability Check in Episode Generation Fallback

**Location:** `episode-generation-job/generate_episodes.py:245-280`

**Current State:** The fallback episode generation path (when not using Genie Sim) claims to use Isaac Sim for physics validation but doesn't verify Isaac Sim is actually available.

```python
def _validate_physics_isaacsim(self, trajectory: list) -> bool:
    """Validate trajectory using Isaac Sim physics."""
    # This requires Isaac Sim to be available
    try:
        from omni.isaac.core import World
        ...
```

**Problem:** Import errors are caught but the function returns `True` on failure, allowing invalid trajectories through.

**Required Fix:**
1. Add explicit Isaac Sim availability check at job startup
2. If Isaac Sim unavailable and physics validation required, fail loudly
3. Add `REQUIRE_ISAACSIM_PHYSICS` config option
4. Log warnings when falling back to heuristic validation

**Impact:** Episodes may have physically impossible trajectories

---

### P0-5: Quality Gate Bypass for Genie Sim Export

**Location:** `genie-sim-export-job/export_to_geniesim.py:380-420`

**Current State:** The Genie Sim export job doesn't run quality gates before export. Quality gates exist (`tools/quality_gates/quality_gate.py`) but aren't integrated.

**Required Fix:**
1. Add quality gate checkpoint `GENIESIM_EXPORT_READY`
2. Run gates before starting export:
   - Manifest completeness check
   - Asset file existence check
   - Physics properties validation
   - Scale sanity check
3. Block export if ERROR-severity gates fail
4. Include gate results in export metadata

**Impact:** Invalid scenes exported to Genie Sim, wasting compute and producing bad data

---

### P0-6: No Rate Limiting for Genie Sim API Calls

**Location:** `genie-sim-export-job/geniesim_client.py`

**Current State:** The client makes API calls without rate limiting. While retry logic exists for failures, there's no proactive rate limiting to prevent hitting API quotas.

**Required Fix:**
1. Implement token bucket rate limiter (already exists in `tools/external_services/client.py`)
2. Wire rate limiter into GenieSim client
3. Configure based on Genie Sim API tier limits
4. Add queue for batch operations

**Impact:** API quota exhaustion, 429 errors, job failures

---

### P0-7: Missing Credential Validation at Startup

**Location:** Multiple job entry points

**Current State:** Jobs assume credentials are valid but don't verify until first API call fails.

**Required Fix:**
1. Add startup validation in each job's `main()`:
   - Verify `GENIESIM_API_KEY` is set
   - Verify GCS credentials work (test bucket access)
   - Verify Gemini API key works (if LLM features enabled)
2. Fail fast with clear error messages
3. Add `--validate-only` CLI flag for pre-flight checks

**Impact:** Jobs fail mid-execution after partial work completed

---

### P0-8: Incomplete Error Propagation from Genie Sim

**Location:** `genie-sim-import-job/import_from_geniesim.py:200-250`

**Current State:** When Genie Sim returns partial failures (some episodes failed, others succeeded), the error handling is incomplete:

```python
if job_result.state == "COMPLETED_WITH_ERRORS":
    # Log warning but continue
    self.log(f"Warning: {len(job_result.errors)} episodes failed")
```

**Problem:** Failed episode details aren't preserved, making debugging impossible.

**Required Fix:**
1. Capture all error details from Genie Sim response
2. Write `failed_episodes.json` with:
   - Episode IDs that failed
   - Error messages from Genie Sim
   - Suggested remediation
3. Include failure summary in import result
4. Support `--fail-on-partial-error` mode for strict pipelines

**Impact:** No visibility into why specific episodes failed

---

## 3. P1 High Priority Gaps

Important for production but won't block initial lab testing.

### P1-1: Missing Genie Sim Quality Gates

**Location:** `tools/quality_gates/quality_gate.py`

**Current State:** Quality gates exist for USD assembly and episode generation but not for Genie Sim-specific checkpoints.

**Missing Gates:**
1. `GENIESIM_SCENE_GRAPH_VALID` - Validate scene graph JSON structure
2. `GENIESIM_TASK_CONFIG_VALID` - Validate task configuration
3. `GENIESIM_JOB_SUBMITTED` - Verify job was accepted
4. `GENIESIM_EPISODES_IMPORTED` - Validate imported episode quality

---

### P1-2: No Idempotency for Genie Sim Job Submission

**Location:** `genie-sim-export-job/geniesim_client.py:80-120`

**Problem:** If a job is submitted twice (due to retry or restart), duplicate jobs run on Genie Sim side.

**Required Fix:**
1. Generate deterministic job ID from scene_id + config hash
2. Check for existing job with same ID before submitting
3. Support "resume" mode to continue monitoring existing job
4. Add `force_resubmit` flag to override

---

### P1-3: Missing Progress Tracking for Long-Running Genie Sim Jobs

**Location:** `genie-sim-export-job/geniesim_client.py:148-180`

**Problem:** No progress visibility while waiting for Genie Sim jobs. Users don't know if job is 10% or 90% complete.

**Required Fix:**
1. Parse progress from Genie Sim status response
2. Emit progress events/logs
3. Update GCS marker files with progress
4. Support webhook callbacks for status updates

---

### P1-4: Incomplete Robot Configuration in Genie Sim Export

**Location:** `genie-sim-export-job/export_to_geniesim.py:150-200`

**Current State:** Robot configurations are included but action/observation space definitions are incomplete.

**Missing:**
1. Full joint limits per robot type
2. Gripper action space definition
3. Sensor configurations (cameras, force-torque)
4. Control mode specifications (position, velocity, torque)

---

### P1-5: No Checkpointing for Pipeline Stages

**Location:** `tools/run_local_pipeline.py`

**Problem:** If pipeline fails mid-execution, it must restart from the beginning. No checkpoint/resume capability.

**Required Fix:**
1. Write checkpoint files after each stage completes
2. Add `--resume` flag to continue from last checkpoint
3. Track stage completion in metadata
4. Support `--from-stage X` to start from specific stage

---

### P1-6: Missing Batch Processing for Multiple Scenes

**Location:** Pipeline runners

**Problem:** Current pipeline processes one scene at a time. No batch orchestration for processing multiple scenes in parallel.

**Required Fix:**
1. Add batch mode to `run_local_pipeline.py`
2. Implement parallel scene processing with resource limits
3. Add batch progress tracking
4. Support batch failure handling (continue vs abort)

---

### P1-7: Incomplete Asset Validation Before Export

**Location:** `genie-sim-export-job/export_to_geniesim.py:250-300`

**Problem:** Assets are referenced in scene graph but not validated:
- GLB files might be corrupt
- Textures might be missing
- USD references might be broken

**Required Fix:**
1. Validate all referenced files exist
2. Check GLB file integrity (trimesh load test)
3. Verify USD references resolve
4. Add asset validation gate

---

### P1-8: No Dry-Run Mode for Genie Sim Export

**Location:** `genie-sim-export-job/export_to_geniesim.py`

**Problem:** Can't preview what would be sent to Genie Sim without actually submitting.

**Required Fix:**
1. Add `--dry-run` flag
2. Generate all export files but don't submit
3. Validate against Genie Sim schema
4. Estimate job cost/time

---

### P1-9: Missing Episode Deduplication

**Location:** `genie-sim-import-job/import_from_geniesim.py`

**Problem:** If import runs twice, duplicate episodes may be created.

**Required Fix:**
1. Generate deterministic episode IDs
2. Check for existing episodes before writing
3. Support `--overwrite` vs `--skip-existing` modes

---

### P1-10: Incomplete Task Configuration Generation

**Location:** `genie-sim-export-job/export_to_geniesim.py:320-380`

**Current State:** Task configs are generated but missing key parameters:
- Success criteria definitions
- Reward shaping hints
- Reset conditions
- Curriculum specifications

---

### P1-11: No Structured Logging

**Location:** All jobs

**Problem:** Logging is print-based, not structured. Hard to aggregate and analyze in production.

**Required Fix:**
1. Use Python `logging` with JSON formatter
2. Include correlation IDs (scene_id, job_id)
3. Add log levels appropriately
4. Configure log shipping to Cloud Logging

---

### P1-12: Missing Webhook/Notification Integration

**Location:** Pipeline orchestration

**Problem:** No way to notify external systems when pipeline completes or fails.

**Required Fix:**
1. Add webhook support for pipeline events
2. Support Slack/email notifications
3. Integrate with notification service in `tools/quality_gates/notification_service.py`

---

## 4. P2 Medium Priority Gaps

Important for robustness and maintainability.

### P2-1: Hardcoded Robot Configurations

**Location:** `genie-sim-export-job/export_to_geniesim.py:50-100`

**Problem:** Robot specs are hardcoded in Python. Should be externalized to config files.

### P2-2: Missing Schema Validation for Genie Sim Outputs

**Location:** `genie-sim-export-job/`

**Problem:** No JSON Schema validation for generated scene_graph.json, asset_index.json, task_config.json.

### P2-3: Incomplete Error Messages

**Location:** Multiple files

**Problem:** Many exceptions lack actionable error messages.

### P2-4: No Metrics Collection

**Location:** Pipeline jobs

**Problem:** No metrics for job duration, success rates, episode quality scores.

### P2-5: Missing Configuration Validation

**Location:** Job entry points

**Problem:** Config files aren't validated against schemas at startup.

### P2-6: Inconsistent Timeout Handling

**Location:** Multiple files

**Problem:** Some operations have timeouts, others don't. Inconsistent enforcement.

### P2-7: No Cost Tracking Integration

**Location:** Genie Sim client

**Problem:** No tracking of Genie Sim API costs per scene/job.

### P2-8: Missing Asset Caching

**Location:** Asset upload/download

**Problem:** Assets re-uploaded even if unchanged. No content-addressable caching.

### P2-9: Incomplete Type Hints

**Location:** Multiple files

**Problem:** Some functions missing type hints, reducing IDE support and static analysis.

### P2-10: No API Versioning Strategy

**Location:** Genie Sim client

**Problem:** No handling for Genie Sim API version changes.

### P2-11: Missing Cleanup for Failed Jobs

**Location:** Pipeline runners

**Problem:** Partial outputs from failed jobs not cleaned up.

### P2-12: No Scene Version Tracking

**Location:** Export/Import jobs

**Problem:** No tracking of which pipeline version produced a scene.

### P2-13: Incomplete Audit Trail

**Location:** Pipeline execution

**Problem:** No comprehensive audit log of who/what/when for each operation.

### P2-14: Missing Health Endpoints for Jobs

**Location:** Job containers

**Problem:** No health/ready endpoints for Kubernetes liveness probes.

### P2-15: No Graceful Shutdown Handling

**Location:** Long-running jobs

**Problem:** SIGTERM not handled gracefully - jobs may leave partial state.

---

## 5. P3 Low Priority Gaps

Nice to have improvements.

### P3-1: No Web UI for Pipeline Monitoring

### P3-2: Missing CLI Autocomplete

### P3-3: No Plugin Architecture for Custom Jobs

### P3-4: Incomplete Internationalization

### P3-5: No Performance Profiling Integration

### P3-6: Missing Compression for Large Transfers

### P3-7: No A/B Testing Framework for Configs

### P3-8: Incomplete Caching Strategy Documentation

### P3-9: No Built-in Backup/Restore

### P3-10: Missing Load Testing Framework

---

## 6. Testing Gaps Analysis

### Current Test Coverage

| Test File | Scope | Status |
|-----------|-------|--------|
| `test_pipeline_e2e.py` | Full pipeline with mocks | ✅ Good |
| `test_geniesim_adapter.py` | Genie Sim adapter unit tests | 🟡 Needs expansion |
| `test_cloud_integration.py` | GCP integration | 🟡 Partial |
| `test_tarfile_security.py` | Security validation | ✅ Good |
| `test_integration_geniesim.py` | Genie Sim integration | ❌ Missing real tests |

### Critical Testing Gaps

#### T-1: No Real Genie Sim Integration Tests

**Problem:** All Genie Sim tests use mocks. No tests against real API.

**Required:**
1. Create test fixtures with small scenes
2. Add integration test that submits real job to Genie Sim sandbox
3. Validate response parsing with real data
4. Run as part of CI with sandbox credentials

#### T-2: No Episode Quality Tests

**Problem:** No tests validating episode data quality.

**Required:**
1. Test episode observation shapes
2. Test action bounds
3. Test reward distributions
4. Test for NaN/Inf values

#### T-3: No Performance/Load Tests

**Problem:** Unknown how pipeline performs under load.

**Required:**
1. Test with 100+ objects per scene
2. Test with 1000+ episode batch
3. Measure memory usage
4. Profile CPU bottlenecks

#### T-4: No Failure Injection Tests

**Problem:** Unknown how pipeline handles various failures.

**Required:**
1. Test network failures to Genie Sim
2. Test invalid API responses
3. Test quota exhaustion
4. Test partial job failures

#### T-5: No Cross-Environment Tests

**Problem:** Tests only run locally, not in cloud environment.

**Required:**
1. Test in GKE environment
2. Test with real GCS buckets
3. Test with real Secret Manager

---

## 7. Operational Readiness

### 7.1 Monitoring & Alerting

**Current State:** Prometheus/Grafana infrastructure exists in `infrastructure/monitoring/` but isn't wired to application code.

**Gaps:**
1. No metrics exported from pipeline jobs
2. No alerting rules defined
3. No dashboards for pipeline health
4. No SLO/SLI definitions

**Required:**
1. Add Prometheus metrics to each job:
   - `pipeline_job_duration_seconds`
   - `pipeline_job_success_total`
   - `pipeline_episodes_generated_total`
   - `geniesim_api_latency_seconds`
2. Create Grafana dashboards
3. Define alerts for:
   - Job failure rate > 10%
   - Job duration > 2x baseline
   - API error rate spike

### 7.2 Deployment Readiness

**Current State:** Terraform exists for GKE but deployment automation incomplete.

**Gaps:**
1. No CI/CD pipeline for container builds
2. No staging environment
3. No blue/green deployment
4. No rollback procedure

### 7.3 Documentation

**Current State:** README exists, inline docs good, but operational docs missing.

**Gaps:**
1. No runbook for common failures
2. No architecture diagram (current)
3. No API documentation for internal services
4. No onboarding guide for new developers

---

## 8. Integration Completeness

### External Services Matrix

| Service | Integration Status | Tested Against Real API | Production Ready |
|---------|-------------------|------------------------|------------------|
| 3D-RE-GEN | 🔴 Not available | N/A | ❌ Blocked |
| Genie Sim 3.0 | 🟡 Code complete | ❌ No | 🟡 Needs testing |
| Isaac Sim | 🟢 Complete | ✅ Yes | ✅ Ready |
| Gemini API | 🟢 Complete | ✅ Yes | ✅ Ready |
| GCS | 🟢 Complete | ✅ Yes | ✅ Ready |
| Firestore | 🟢 Complete | ✅ Yes | ✅ Ready |
| Secret Manager | 🟢 Complete | ✅ Yes | ✅ Ready |
| LeRobot Hub | 🟡 Code complete | ❌ No | 🟡 Needs testing |

### Internal Integration Matrix

| From Job | To Job | Integration | Status |
|----------|--------|-------------|--------|
| regen3d-job | simready-job | Manifest handoff | ✅ |
| regen3d-job | interactive-job | Manifest handoff | ✅ |
| simready-job | usd-assembly-job | Physics USD | ✅ |
| interactive-job | usd-assembly-job | URDF/articulations | ✅ |
| usd-assembly-job | replicator-job | scene.usda | ✅ |
| replicator-job | genie-sim-export | Manifest + USD | ✅ |
| genie-sim-export | Genie Sim API | API call | 🟡 |
| Genie Sim API | genie-sim-import | Episode download | 🟡 |
| genie-sim-import | arena-export-job | LeRobot format | ✅ |

---

## 9. Recommendations

### Immediate Actions (Before Lab Testing)

1. **Fix P0-1:** Configure real Genie Sim API endpoint with validation
2. **Fix P0-2:** Add polling timeout (4 hour max)
3. **Fix P0-3:** Implement episode validation before import
4. **Fix P0-5:** Wire quality gates to Genie Sim export
5. **Fix P0-7:** Add credential validation at job startup
6. **Create:** Real integration test with Genie Sim sandbox

### Short-Term (1-2 weeks)

1. Add structured logging across all jobs
2. Implement checkpointing for pipeline stages
3. Add dry-run mode for Genie Sim export
4. Create monitoring dashboards
5. Write operational runbook

### Medium-Term (2-4 weeks)

1. Add batch processing capability
2. Implement cost tracking
3. Add webhook notifications
4. Performance optimization
5. Complete test coverage

### Lab Testing Prerequisites Checklist

Before testing with labs, ensure:

- [ ] Real Genie Sim API endpoint configured and reachable
- [ ] API credentials validated at startup
- [ ] Quality gates integrated with export
- [ ] Episode validation implemented
- [ ] At least one successful end-to-end test with real Genie Sim
- [ ] Monitoring/alerting in place
- [ ] Runbook for common failures documented
- [ ] Cost estimation available
- [ ] Rollback procedure defined

---

## Appendix A: File Reference

| Path | Purpose |
|------|---------|
| `genie-sim-export-job/export_to_geniesim.py` | Main export logic |
| `genie-sim-export-job/geniesim_client.py` | Genie Sim API client |
| `genie-sim-import-job/import_from_geniesim.py` | Episode import logic |
| `episode-generation-job/generate_episodes.py` | Fallback episode generation |
| `tools/quality_gates/quality_gate.py` | Quality gate framework |
| `tools/error_handling/retry.py` | Retry utilities |
| `tools/arena_integration/arena_exporter.py` | Arena/LeRobot export |
| `tools/run_local_pipeline.py` | Local pipeline runner |
| `tests/test_pipeline_e2e.py` | E2E tests |
| `workflows/scene-generation-pipeline.yaml` | Cloud workflow |

---

## Appendix B: Configuration Requirements

### Required Environment Variables

```bash
# Genie Sim (Required)
GENIESIM_API_URL=https://api.geniesim.nvidia.com/v1
GENIESIM_API_KEY=<from-secret-manager>

# GCP (Required)
GOOGLE_CLOUD_PROJECT=<project-id>
GCS_BUCKET=blueprint-scenes

# Optional but Recommended
GEMINI_API_KEY=<from-secret-manager>
ISAAC_SIM_PATH=/opt/nvidia/isaac-sim
LOG_LEVEL=INFO
```

### Required Secrets (Secret Manager)

```
projects/<project>/secrets/geniesim-api-key/versions/latest
projects/<project>/secrets/gemini-api-key/versions/latest
```

---

**Document prepared by:** Claude Code Audit
**Review requested from:** Engineering Lead, Platform Team
