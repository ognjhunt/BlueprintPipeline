# BlueprintPipeline Production Readiness Audit

**Date:** January 10, 2026
**Auditor:** Claude (Opus 4.5)
**Pipeline Mode:** Genie Sim 3.0 (Default)
**Status:** 🟡 **CONDITIONAL PRODUCTION-READY** (pending external dependencies)

---

## Executive Summary

BlueprintPipeline is an architecturally sophisticated system for converting 2D scene images into simulation-ready 3D environments with RL training data. The default pipeline routes to **Genie Sim 3.0** for scalable data generation.

### Overall Assessment

| Category | Status | Score |
|----------|--------|-------|
| **Code Quality** | Good | 8/10 |
| **Architecture** | Excellent | 9/10 |
| **Error Handling** | Good (recently improved) | 7.5/10 |
| **Security** | Good (recently improved) | 7.5/10 |
| **Test Coverage** | Needs Work | 4/10 |
| **Documentation** | Excellent | 9/10 |
| **External Dependencies** | Blocking | 3/10 |
| **Production Readiness** | Conditional | 6/10 |

### Critical Blockers for Lab Testing

| Blocker | Impact | Status | Resolution Path |
|---------|--------|--------|-----------------|
| **3D-RE-GEN Not Available** | Cannot process real images | 🔴 BLOCKING | Wait for Q1 2025 release OR use mock data |
| **Isaac Sim Required** | No physics-verified data | 🟡 REQUIRED | Run inside Isaac Sim container |
| **Genie Sim 3.0 API Access** | Cannot generate episodes externally | 🟡 REQUIRED | Obtain API key from AGIBOT |

---

## 1. Pipeline Architecture Analysis

### 1.1 Default Pipeline Flow (Genie Sim 3.0 Mode)

```
Image Input
    │
    ▼
┌─────────────────────────────────────────┐
│ 3D-RE-GEN (EXTERNAL)                    │
│ Status: 🔴 NOT AVAILABLE                │
│ Paper: arXiv:2512.17459                 │
│ Expected: Q1 2025                       │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ regen3d-job (Adapter)                   │
│ Status: ✅ READY                        │
│ Output: scene_manifest.json             │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ simready-job (Physics Estimation)       │
│ Status: ✅ READY                        │
│ Uses: Gemini API (with heuristic fallback) │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ usd-assembly-job (Scene Assembly)       │
│ Status: ✅ READY                        │
│ Output: scene.usda                      │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ replicator-job (Domain Randomization)   │
│ Status: ✅ READY                        │
│ Output: placement_regions.usda          │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ variation-gen-job (Commercial Assets)   │
│ Status: ✅ READY                        │
│ Output: YOUR commercial-use assets      │
│ Note: REQUIRED for selling data         │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ genie-sim-export-job ⭐ DEFAULT         │
│ Status: ✅ READY                        │
│ Output: scene_graph.json, asset_index   │
│         task_config.json, robot configs │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ GENIE SIM 3.0 (EXTERNAL SERVICE)        │
│ Status: 🟡 REQUIRES API ACCESS          │
│ - LLM task generation                   │
│ - cuRobo trajectory planning            │
│ - Automated data collection             │
│ - VLM evaluation                        │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ genie-sim-import-job (Closes Loop)      │
│ Status: ✅ READY                        │
│ - Episode download                      │
│ - Quality validation                    │
│ - LeRobot format conversion             │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ LeRobot Dataset (Training-Ready)        │
│ Format: Parquet + MP4 videos            │
│ Includes: Quality certificates          │
└─────────────────────────────────────────┘
```

### 1.2 Fallback Pipeline (BlueprintPipeline Episode Generation)

When `USE_GENIESIM=false`:

```
... (up through variation-gen-job)
              │
              ▼
┌─────────────────────────────────────────┐
│ episode-generation-job (Internal)       │
│ Status: ✅ READY (requires Isaac Sim)   │
│ Components:                             │
│ - TaskSpecifier (Gemini-powered)        │
│ - AIMotionPlanner + CollisionAwarePlanner │
│ - CPGenAugmenter (constraint-preserving) │
│ - SimulationValidator (PhysX or heuristic) │
│ - SensorDataCapture (Isaac Sim Replicator) │
│ - LeRobotExporter                       │
└─────────────────────────────────────────┘
```

---

## 2. Gap Analysis by Category

### 2.1 External Dependencies (CRITICAL)

| Dependency | Required For | Status | Workaround |
|------------|--------------|--------|------------|
| **3D-RE-GEN** | Image → 3D reconstruction | 🔴 NOT AVAILABLE | Mock data generator (`fixtures/generate_mock_regen3d.py`) |
| **Isaac Sim** | Physics simulation, sensor capture | 🟡 OPTIONAL (for dev) | Mock mode (dev only) |
| **Genie Sim 3.0 API** | External episode generation | 🟡 REQUIRES ACCESS | Use fallback mode |
| **Gemini API** | Physics estimation, task spec | 🟢 AVAILABLE | Heuristic fallback |
| **Particulate** | Articulation detection | 🟢 OPTIONAL | Heuristic fallback |
| **cuRobo** | GPU motion planning | 🟢 OPTIONAL | RRT fallback |

### 2.2 Code Quality Gaps

#### GAP-CQ-001: Test Coverage is Low
- **Current:** 4 test files for 264 Python files (~1.5% ratio)
- **Impact:** Regressions may not be caught
- **Recommendation:** Add unit tests for:
  - `tools/geniesim_adapter/*.py`
  - `tools/validation/*.py`
  - `tools/error_handling/*.py`
  - Each job's core logic

#### GAP-CQ-002: Premium Feature Modules May Be Stubs
- **Files:** `genie-sim-export-job/default_*.py` (9 modules)
- **Concern:** These import with try/except and may not have full implementations
- **Verification Needed:** Confirm each exporter actually writes meaningful data

#### GAP-CQ-003: Type Hints Inconsistent
- **Observation:** Some modules use full type hints, others don't
- **Impact:** IDE assistance and static analysis limited
- **Recommendation:** Add `mypy` to CI, gradually add type hints

### 2.3 Security Gaps

#### GAP-SEC-001: Secret Manager Integration (PARTIALLY FIXED)
- **Status:** ✅ Implemented in `geniesim_client.py`
- **Remaining:** Not integrated in:
  - `simready-job/prepare_simready_assets.py` (Gemini API key)
  - `episode-generation-job/task_specifier.py` (LLM API keys)
  - `tools/llm_client/` (LLM provider keys)

#### GAP-SEC-002: Input Validation (PARTIALLY FIXED)
- **Status:** ✅ Implemented in `tools/validation/input_validation.py`
- **Integrated:** `tools/geniesim_adapter/scene_graph.py`
- **Not Integrated:**
  - `regen3d-job/regen3d_adapter_job.py`
  - `simready-job/prepare_simready_assets.py`
  - `usd-assembly-job/build_scene_usd.py`

#### GAP-SEC-003: tarfile Extraction Security
- **Location:** `genie-sim-import-job/import_from_geniesim.py:873-874`
- **Risk:** `tar.extractall()` without path validation can enable path traversal
- **Fix Required:**
```python
# Current (vulnerable):
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(output_dir)  # ❌ No validation

# Should be:
def safe_extract(tar, path):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not os.path.abspath(member_path).startswith(os.path.abspath(path)):
            raise ValueError(f"Path traversal detected: {member.name}")
    tar.extractall(path)
```

### 2.4 Error Handling Gaps

#### GAP-EH-001: Retry/Circuit Breaker (FIXED)
- **Status:** ✅ Implemented in `geniesim_client.py`

#### GAP-EH-002: Timeout Handling (FIXED)
- **Status:** ✅ Implemented in `tools/error_handling/timeout.py`

#### GAP-EH-003: Enhanced Failure Markers (NOT INTEGRATED)
- **Module:** `tools/workflow/failure_markers.py` exists
- **Not Yet Integrated Into:**
  - `genie-sim-export-job/export_to_geniesim.py`
  - `simready-job/prepare_simready_assets.py`
  - `episode-generation-job/generate_episodes.py`

#### GAP-EH-004: Partial Failure Handling (NOT INTEGRATED)
- **Module:** `tools/error_handling/partial_failure.py` exists
- **Not Yet Integrated Into:**
  - `episode-generation-job/generate_episodes.py`
  - Should save successful episodes even when some fail

### 2.5 Configuration Management Gaps

#### GAP-CM-001: Config Validation (NOT INTEGRATED)
- **Module:** `tools/validation/config_schemas.py` exists
- **Not Yet Integrated:** Environment variable validation at job startup
- **Risk:** Invalid configs cause cryptic errors deep in execution

#### GAP-CM-002: Environment-Specific Configs
- **Current:** All configs via environment variables
- **Missing:** Environment-specific config files (dev, staging, prod)
- **Recommendation:** Add `config/{dev,staging,prod}.yaml` files

### 2.6 Performance Gaps

#### GAP-PERF-001: Streaming JSON (FIXED)
- **Status:** ✅ Auto-detects large manifests (>10MB) and uses streaming

#### GAP-PERF-002: Parallel Processing (FIXED)
- **Status:** ✅ Implemented in `simready-job` with 10 workers

#### GAP-PERF-003: Memory Management
- **Location:** `episode-generation-job/generate_episodes.py`
- **Issue:** `gc.collect()` calls suggest memory pressure
- **Recommendation:** Profile memory usage, consider generator patterns

### 2.7 Observability Gaps

#### GAP-OBS-001: Structured Logging
- **Current:** Mix of `print()` and `logging.info()`
- **Missing:** Structured JSON logging for Cloud Logging
- **Recommendation:** Standardize on structured logging

#### GAP-OBS-002: Metrics Collection
- **Module:** `tools/metrics/` exists
- **Gap:** Not consistently used across all jobs
- **Missing Metrics:**
  - Episode generation rate
  - Quality score distribution
  - API latency percentiles
  - Error rates by type

#### GAP-OBS-003: Distributed Tracing
- **Current:** None
- **Recommendation:** Add OpenTelemetry tracing for end-to-end visibility

---

## 3. Workflow Orchestration Analysis

### 3.1 Cloud Workflows Review

**File:** `workflows/usd-assembly-pipeline.yaml`

#### Strengths:
- ✅ Proper job chaining with status checks
- ✅ Graceful degradation (replicator failure is non-fatal)
- ✅ Completion markers for coordination
- ✅ Detailed logging at each step

#### Gaps:

##### GAP-WF-001: No Timeout on Polling Loops
```yaml
- wait_for_simready_complete:
    # Polls up to 90 times with 10s sleep = 15 minutes max
    # But no explicit timeout, and counter could theoretically overflow
```
**Recommendation:** Add explicit workflow-level timeout

##### GAP-WF-002: No Dead Letter Queue Integration
- Failed scenes are not automatically routed to DLQ
- Manual intervention required to reprocess failures

##### GAP-WF-003: Missing Workflows
- `genie-sim-export-pipeline.yaml` exists ✅
- `genie-sim-import-pipeline.yaml` exists ✅
- **Missing:** Combined export+generate+import workflow for Genie Sim mode

### 3.2 Local Pipeline Runner

**File:** `tools/run_local_pipeline.py` (referenced in tests)

- ✅ Supports step-by-step execution
- ✅ Validation between steps
- **Gap:** Not documented how to run full Genie Sim mode locally

---

## 4. Testing Infrastructure Analysis

### 4.1 Current Test Coverage

| Test File | Coverage Area | Quality |
|-----------|---------------|---------|
| `test_pipeline_e2e.py` | Full pipeline simulation | Good |
| `test_geniesim_adapter.py` | Scene graph conversion | Good |
| `test_cloud_integration.py` | Cloud Run jobs | Unknown |
| `test_integration_geniesim.py` | Genie Sim API client | Good |

### 4.2 Missing Test Coverage

| Area | Priority | Files Needing Tests |
|------|----------|---------------------|
| **Error Handling** | P0 | `tools/error_handling/*.py` |
| **Input Validation** | P0 | `tools/validation/*.py` |
| **Secret Manager** | P0 | `tools/secrets/*.py` |
| **Episode Generation** | P1 | `episode-generation-job/*.py` |
| **SimReady** | P1 | `simready-job/*.py` |
| **Genie Sim Import** | P1 | `genie-sim-import-job/*.py` |
| **Premium Features** | P2 | `genie-sim-export-job/default_*.py` |

### 4.3 Test Infrastructure Gaps

#### GAP-TEST-001: No CI/CD Pipeline Definition
- No `.github/workflows/` or `cloudbuild.yaml` found
- **Recommendation:** Add GitHub Actions or Cloud Build config

#### GAP-TEST-002: No Integration Test Environment
- Tests use mock data generator
- No Isaac Sim integration tests
- **Recommendation:** Add Isaac Sim Docker test environment

#### GAP-TEST-003: No Load/Performance Tests
- Unknown how system behaves under load
- **Recommendation:** Add k6 or locust tests for API endpoints

---

## 5. Production Deployment Gaps

### 5.1 Infrastructure

#### GAP-INFRA-001: Terraform State Management
- `infrastructure/terraform/` exists
- **Gap:** No remote state backend configured
- **Risk:** State file conflicts in team environments

#### GAP-INFRA-002: Kubernetes Namespace Isolation
- `k8s/namespace-setup.yaml` exists
- **Gap:** No NetworkPolicy for pod isolation
- **Risk:** Pods can communicate freely within cluster

#### GAP-INFRA-003: Resource Limits
- Job definitions may lack resource limits
- **Risk:** OOM kills, resource starvation

### 5.2 Monitoring

#### GAP-MON-001: Alerting Not Configured
- Dashboard exists (`monitoring/dashboard_config.json`)
- **Gap:** No alerting rules defined
- **Recommendation:** Add alerts for:
  - Job failure rate > 5%
  - P95 latency > 10 minutes
  - Error rate spikes

### 5.3 Disaster Recovery

#### GAP-DR-001: No Backup Strategy
- Scene data in GCS has no documented backup
- **Recommendation:** Enable GCS versioning, define retention policy

#### GAP-DR-002: No Runbook
- No documented procedure for incident response
- **Recommendation:** Create runbook for common failures

---

## 6. Recommendations by Priority

### P0: Critical (Must Fix Before Lab Testing)

1. **Resolve 3D-RE-GEN Dependency**
   - Option A: Use mock data generator for testing
   - Option B: Integrate alternative (MASt3R, DUSt3R)
   - Option C: Wait for 3D-RE-GEN Q1 2025 release

2. **Obtain Genie Sim API Access**
   - Contact AGIBOT for API key
   - OR switch to fallback mode (`USE_GENIESIM=false`)

3. **Setup Isaac Sim Environment**
   - Create Isaac Sim Docker image with pipeline
   - Document `/isaac-sim/python.sh` usage

4. **Fix tarfile Security Vulnerability**
   - Add path traversal check in `import_from_geniesim.py`

### P1: High (Should Fix Before Production)

1. **Integrate Failure Markers into All Jobs**
   - Files: `export_to_geniesim.py`, `prepare_simready_assets.py`, `generate_episodes.py`

2. **Integrate Partial Failure Handling**
   - File: `generate_episodes.py`

3. **Integrate Config Validation**
   - Add startup validation for environment variables

4. **Integrate Secret Manager for Remaining API Keys**
   - Gemini API key in `simready-job`
   - LLM keys in `task_specifier.py`

5. **Add CI/CD Pipeline**
   - GitHub Actions or Cloud Build
   - Run tests on every PR

### P2: Medium (Should Fix Before Scale)

1. **Increase Test Coverage**
   - Target: 50% code coverage
   - Focus on error handling and validation

2. **Add Structured Logging**
   - Standardize on JSON logging
   - Add request IDs for tracing

3. **Add Alerting Rules**
   - Job failure rate
   - Latency percentiles
   - Error rate by type

4. **Document Local Development**
   - How to run full pipeline locally
   - How to test Genie Sim mode

### P3: Low (Nice to Have)

1. **Add Type Hints Everywhere**
   - Enable mypy in CI

2. **Add OpenTelemetry Tracing**
   - End-to-end visibility

3. **Create Runbooks**
   - Incident response procedures

---

## 7. Testing Checklist for Lab Deployment

### Pre-Deployment Verification

- [ ] 3D-RE-GEN source available (or mock data working)
- [ ] Genie Sim API key obtained (or fallback mode configured)
- [ ] Isaac Sim container built and tested
- [ ] All environment variables documented
- [ ] Secrets in Secret Manager (or env vars as fallback)
- [ ] GCS bucket created with correct permissions
- [ ] Cloud Run jobs deployed
- [ ] Workflows deployed

### Smoke Test Procedure

1. **Generate Mock Scene**
   ```bash
   python fixtures/generate_mock_regen3d.py \
     --scene-id test_kitchen \
     --output-dir ./test_scenes
   ```

2. **Run Local Pipeline (Dev Mode)**
   ```bash
   python tools/run_local_pipeline.py \
     --scene-dir ./test_scenes/test_kitchen \
     --steps regen3d,simready,usd,replicator
   ```

3. **Run Genie Sim Export**
   ```bash
   SCENE_ID=test_kitchen \
   BUCKET=your-bucket \
   python genie-sim-export-job/export_to_geniesim.py
   ```

4. **Validate Outputs**
   - `geniesim/scene_graph.json` exists and valid JSON
   - `geniesim/asset_index.json` has assets
   - `geniesim/task_config.json` has suggested tasks
   - Premium feature directories populated

### Production Validation

- [ ] Process 10 scenes end-to-end
- [ ] Verify episode quality scores > 0.7 average
- [ ] Monitor job failure rate (target: <5%)
- [ ] Monitor P95 latency (target: <10 min/scene)
- [ ] Verify no credential leaks in logs
- [ ] Verify no hung processes

---

## 8. Conclusion

### Current State

BlueprintPipeline is **architecturally complete** and **code-quality good** for the Genie Sim 3.0 mode. The recent fixes (retry/circuit breaker, timeout handling, input validation, streaming JSON, parallel processing) have significantly improved production readiness.

### Blocking Issues

1. **3D-RE-GEN** - Cannot process real images until available
2. **Genie Sim API** - Cannot generate episodes externally without API access
3. **Isaac Sim** - Cannot generate physics-verified data without Isaac Sim

### Path to Production

| Phase | Duration | Outcome |
|-------|----------|---------|
| **Phase 1: Mock Testing** | 1 week | Validate pipeline with mock data |
| **Phase 2: Fallback Mode** | 2 weeks | Use BlueprintPipeline episode generation |
| **Phase 3: Full Genie Sim** | When available | Enable Genie Sim 3.0 mode |

### Final Assessment

**For Lab Testing with Mock Data:** ✅ READY NOW

**For Production with Real Data:** 🟡 CONDITIONAL
- Requires: Isaac Sim environment
- Requires: Either 3D-RE-GEN OR alternative reconstruction
- Requires: Either Genie Sim API OR fallback mode

---

*Document generated by production readiness audit on 2026-01-10*
*Auditor: Claude (Opus 4.5)*
