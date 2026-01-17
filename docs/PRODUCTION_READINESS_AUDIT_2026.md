# Production Readiness Audit - BlueprintPipeline

**Date:** 2026-01-17
**Auditor:** Claude (Automated Audit)
**Context:** Internal pipeline producing datasets sold to labs

---

## Executive Summary

This audit identified **~300+ issues** across 10 categories. Issues are ranked by their impact on **data quality** (labs pay for this), **pipeline reliability**, and **operational stability**.

**Key Finding:** The most critical issues relate to data quality and integrity - labs are paying for reliable datasets, and issues like missing checksums, partial writes, and weak quality gates could seriously impact customer trust.

---

## Critical Priority (Fix Before Production)

### 1. Data Quality & Integrity Issues

**Impact: Labs receive corrupted/incomplete datasets they paid for**

| Issue | File | Line | Description |
|-------|------|------|-------------|
| **Missing checksums** | `episode-generation-job/lerobot_exporter.py` | 1296+ | No SHA256/MD5 for Parquet, JSON, video files - labs cannot verify integrity |
| **Partial writes without rollback** | `episode-generation-job/lerobot_exporter.py` | 1219-1243 | If export fails mid-write, orphaned files remain with no cleanup |
| **Frame count mismatches exported** | `episode-generation-job/lerobot_exporter.py` | 1188-1199 | Episodes with mismatched trajectory/sensor frames exported (warning only) |
| **Incomplete episodes exported** | `episode-generation-job/lerobot_exporter.py` | 1146-1158 | Missing trajectories/task descriptions produce warnings but still export |
| **Data tier mismatch** | `episode-generation-job/lerobot_exporter.py` | 1160-1172 | "Full" pack episodes could have Core pack data - no blocking validation |
| **Silent frame drops** | `episode-generation-job/lerobot_exporter.py` | 1800-1809 | Invalid RGB frames skipped silently; dataset says "X frames" but has fewer |
| **No data lineage** | `episode-generation-job/lerobot_exporter.py` | 1669-1707 | Labs cannot trace which sim backend, physics params, or quality gates were used |
| **Quality gates bypassable** | `tools/quality_gates/quality_gate.py` | 412-453 | Override requires only 10-char reason - no audit trail |
| **Auto-approve on timeout** | `tools/quality_gates/quality_gate.py` | 266-270 | Failed gates auto-approved after 24h with no human review |
| **Weak thresholds** | `tools/quality_gates/quality_gate.py` | 1130-1184 | 50% quality_pass_rate_min, 20% collision allowed - marginal data accepted |
| **Parquet corruption risk** | `episode-generation-job/lerobot_exporter.py` | 1281-1305 | Non-atomic writes; crash mid-write leaves corrupted files |
| **Reward signal corruption** | `episode-generation-job/lerobot_exporter.py` | 436-473 | Fallback uses quality_score without NaN/Inf validation |
| **Schema validation gaps** | `episode-generation-job/multi_format_exporters.py` | 518-600 | HDF5 export silently passes on conversion failures |
| **Race conditions** | `tools/batch_processing/parallel_runner.py` | Various | Concurrent episode writing without file locks |
| **Sensor data loss** | `episode-generation-job/sensor_data_capture.py` | 1787-1812 | Capture failures set sensor_data to None, episodes export without visual obs |

**Recommended Fix Priority: P0 - Block production release**

---

### 2. Critical Security Issues

| Issue | File | Line | Severity |
|-------|------|------|----------|
| **Unsafe pickle loading** | `objects-job/run_objects_from_layout.py` | 17 | CRITICAL - `allow_pickle=True` allows arbitrary code execution |
| **Exposed debug endpoint** | `particulate-service/particulate_service.py` | 563-581 | HIGH - `/debug` returns internal paths/state without auth |
| **Dynamic code execution** | `tests/test_reward_functions.py` | 25 | CRITICAL - `exec(source, namespace)` on dynamic code |
| **Traceback exposure** | 17+ files | Various | MEDIUM - Stack traces in responses leak internal paths |
| **Table name injection risk** | `tools/asset_catalog/vector_store.py` | 181-191 | MEDIUM - Table names in f-strings (not parameterized) |

---

### 3. Incomplete/Placeholder Features

| Feature | Files | Status | Impact |
|---------|-------|--------|--------|
| **Dream2Flow model** | `dream2flow-preparation-job/*` | NOT RELEASED | "Placeholder video - model not yet available" |
| **3D-RE-GEN code** | `tools/regen3d_adapter/__init__.py` | PENDING (~Q1 2025) | Public release pending |
| **MANO hand model** | `dwm-preparation-job/hand_motion/` | PLACEHOLDER | Uses SimpleHandMesh fallback |
| **NotImplementedError** | 12 files | BLOCKING | Various methods raise NotImplementedError |
| **50+ P0/P1/P2 FIX comments** | Throughout codebase | KNOWN ISSUES | Marked but unfixed |
| **LABS-BLOCKER issues** | `genie-sim-import-job/` | BLOCKING | Quality threshold workarounds |
| **Stub implementations** | `tools/inventory_enrichment/enricher.py` | STUB MODE | Returns pass-through data |
| **Mock capture in production** | `episode-generation-job/` | RISK | Mock data could reach production |

---

## High Priority (Fix Soon After Launch)

### 4. Error Handling Gaps (138+ issues)

| Pattern | Count | Example Files |
|---------|-------|---------------|
| **Bare except with pass** | 12+ | `intelligent_region_detector.py:104`, `build_scene_usd.py:1011` |
| **Missing file I/O error handling** | 25+ | `import_from_geniesim.py:288`, all JSON loads |
| **Missing API error handling** | 15+ | `run_interactive_assets.py:540`, `meshy-job/*` |
| **Silent fallbacks masking failures** | 20+ | `generate_isaac_lab_task.py:492` defaults to `{}` |
| **Subprocess without timeout** | 10+ | `run_variation_asset_pipeline.py:1225` |
| **Missing null checks** | 30+ | Array access without bounds checking |
| **Unhandled JSON parsing** | 18+ | `json.loads()` without try/catch |

**Key Files to Fix:**
- `genie-sim-import-webhook/main.py:112-117` - Webhook endpoint missing comprehensive error handling
- `interactive-job/run_interactive_assets.py:540-570` - API health check with unhandled JSON errors
- `variation-asset-pipeline-job/run_variation_asset_pipeline.py:1256-1268` - Mesh loading without validation

---

### 5. Testing Coverage Gaps

| Category | Status | Impact |
|----------|--------|--------|
| **22/23 job modules** | NO TESTS | episode-generation-job (21 files, 0 tests) - CRITICAL |
| **19/38 tool modules** | NO TESTS | checkpoint, cost_tracking, secrets - all untested |
| **Error handling** | 0% coverage | No tests for error paths |
| **Circuit breaker** | 0% coverage | State transitions untested |
| **387+ public functions** | Untested | Error scenarios not exercised |
| **11 error classes** | 0% coverage | No serialization tests |

**Critical Untested Modules:**
- `episode-generation-job/` - 21 files including reward computation, trajectory solving, quality certificates
- `upsell-features-job/` - 25 files including policy evaluation, generalization analysis
- `tools/checkpoint/` - Checkpoint system (critical for resumable pipelines)
- `tools/cost_tracking/` - Cost calculations (business-critical)
- `tools/secrets/` - Secret management (security-critical)
- `tools/error_handling/circuit_breaker.py` - Circuit breaker state machine

---

### 6. Logging & Monitoring Gaps

| Issue | Count | Impact |
|-------|-------|--------|
| **print() instead of logging** | 126+ instances | No log levels, no filtering, no structured output |
| **Files without logging imports** | 34 files | Critical operations completely unlogged |
| **Missing health checks** | 10+ services | No deep checks for Isaac Sim, GPU, Gemini connectivity |
| **No structured logging** | 0 files | All logs are unstructured strings - hard to parse |
| **No alerting system** | 0 configured | Pipeline failures go unnoticed |
| **Sensitive data risk** | 15+ files | API keys could appear in error logs |

**Files with Most print() Statements:**
- `regen3d-job/regen3d_adapter_job.py` - 60+ print statements
- `isaac-lab-job/generate_isaac_lab_task.py` - 45+ print statements
- `variation-gen-job/generate_variation_assets.py` - 40+ print statements
- `episode-generation-job/isaac_sim_enforcement.py` - 35+ print statements

---

### 7. Environment Configuration Issues

| Issue | Count | Impact |
|-------|-------|--------|
| **BUCKET/SCENE_ID default to ""** | 40+ files | Pipeline runs with empty bucket - fails late |
| **Production mode detection fragmented** | 14 files | Inconsistent SIMREADY_PRODUCTION_MODE vs PRODUCTION_MODE |
| **3 different boolean parsers** | 3 implementations | `parse_env_flag()`, `_env_flag()`, `env_flag()` duplicated |
| **Undocumented env vars** | 11+ variables | PARTICULATE_MODE, GENIESIM_MOCK_MODE, etc. |
| **Hardcoded localhost** | 10+ locations | Breaks remote deployment |
| **No .env.example** | Missing | New developers cannot quickly see required vars |
| **Inconsistent defaults** | 6+ files | DATA_QUALITY_LEVEL has different defaults across jobs |

**Undocumented Environment Variables:**
- `PARTICULATE_MODE`, `PARTICULATE_LOCAL_ENDPOINT`
- `INTERACTIVE_MODE`, `ARTICULATION_BACKEND`
- `APPROVED_PARTICULATE_MODELS`, `DISALLOW_PLACEHOLDER_URDF`
- `LABS_MODE`, `GENIESIM_MOCK_MODE`, `SERVICE_MODE`
- `GENERATE_EMBEDDINGS`, `FILTER_COMMERCIAL`, `COPY_USD`, `ENABLE_BIMANUAL`

---

## Medium Priority (Plan for Future Sprints)

### 8. Performance Issues

| Issue | Location | Impact |
|-------|----------|--------|
| **N+1 database pattern** | `tools/asset_catalog/vector_store.py:207` | Individual INSERTs in loop instead of batch |
| **Unbounded queries** | `tools/asset_catalog/vector_store.py:430` | SELECT * without LIMIT loads all records |
| **Blocking polling loops** | `tools/quality_gates/quality_gate.py:300` | `while True` with 30s sleep, no max iterations |
| **Memory accumulation** | `tools/asset_catalog/vector_store.py:541` | `_metadata_store` dict grows forever, no cleanup |
| **No streaming for large files** | `genie-sim-export-job/geniesim_client.py:1332` | Loads all episodes into memory |
| **Blocking in async code** | `tools/external_services/service_client.py:339` | `time.sleep()` in rate limiter blocks event loop |
| **Quadratic complexity** | `tools/asset_catalog/vector_store.py:240-243` | `list.insert(-1, x)` in loop is O(n) |

---

### 9. Dependency & Versioning Issues

| Issue | Impact |
|-------|--------|
| **Unpinned dependencies in Isaac Sim files** | 33+ packages with `>=` only - builds not reproducible |
| **Missing lock files** | 15/21 requirements files lack locks |
| **Python 3.10 vs 3.11 mix** | 5 jobs use 3.10, 14 use 3.11 - inconsistent |
| **Isaac Sim requirements not synced** | `sync_requirements_pins.py` skips `-isaacsim.txt` files |
| **Version conflicts** | pytest: `==8.3.4` vs `>=7.4.0` across files |

**Files Lacking Lock Files:**
- `arena-export-job`, `dream2flow-preparation-job`, `dwm-preparation-job`
- `episode-generation-job`, `scene-generation-job`, `smart-placement-engine-job`
- `tools/tracing`, `ultrashape`, `variation-asset-pipeline-job`

---

### 10. Rate Limiting & Cost Control

| Issue | Location | Risk |
|-------|----------|------|
| **Hardcoded max_workers=10** | `simready-job/prepare_simready_assets.py:2308` | May exceed Gemini rate limits |
| **No rate limiting on LLM calls** | `tools/articulation/detector.py:174` | Direct client without ServiceClient wrapper |
| **Concurrent API calls uncoordinated** | `tools/performance/parallel_processing.py:91` | All items submitted immediately |
| **Cost tracking not integrated** | All API callers | Gemini/OpenAI costs not tracked |
| **No quota monitoring** | Missing | No alerts when approaching API limits |
| **Missing retry logic** | `tools/asset_catalog/image_captioning.py:43-58` | Immediate fallback, no retry |

---

## Lower Priority (Good to Have)

### 11. Documentation Gaps

| Category | Missing Items |
|----------|---------------|
| **README files** | 52 directories lack READMEs (22 job dirs, 37 tools subdirs) |
| **API documentation** | 5+ major APIs undocumented (geniesim_adapter, llm_client, etc.) |
| **Architecture diagrams** | All data flow diagrams missing |
| **Changelog** | No CHANGELOG file exists |
| **Operations runbook** | 14+ procedures undocumented (incident response, backup, etc.) |
| **Config file docs** | 10+ config files undocumented (policy_configs/, etc.) |

---

## Summary Statistics

| Category | Issues Found | Severity | Effort to Fix |
|----------|--------------|----------|---------------|
| **Data Quality/Integrity** | 15+ critical | CRITICAL | HIGH |
| **Security** | 4 critical, 17 medium | HIGH | MEDIUM |
| **Incomplete Features** | 50+ placeholders | HIGH | DEPENDS |
| **Error Handling** | 138+ gaps | HIGH | HIGH |
| **Testing** | 22 modules untested | HIGH | HIGH |
| **Logging/Monitoring** | 126+ issues | HIGH | MEDIUM |
| **Environment Config** | 70+ issues | MEDIUM | LOW |
| **Performance** | 20+ issues | MEDIUM | MEDIUM |
| **Dependencies** | 15+ conflicts | MEDIUM | LOW |
| **Rate Limiting** | 10+ issues | MEDIUM | LOW |
| **Documentation** | 100+ gaps | LOW | LOW |

---

## Recommended Action Plan

### Phase 1: Data Quality (BEFORE SELLING MORE DATASETS)

1. **Add SHA256 checksums** to all output files (Parquet, JSON, video)
2. **Implement atomic write-then-move** for all exports - prevents partial files
3. **Block export of episodes with frame mismatches** (change from warning to error)
4. **Add data lineage metadata** to info.json (sim backend, physics params, quality gates used)
5. **Enforce tier compliance validation** - Core vs Plus vs Full must match actual content
6. **Remove quality gate bypass** via environment variables in production mode
7. **Add validation for NaN/Inf** in reward signals before export

### Phase 2: Reliability

1. **Remove `allow_pickle=True`** from `objects-job/run_objects_from_layout.py:17`
2. **Disable/secure `/debug` endpoint** in particulate-service
3. **Add tests for episode-generation-job** (21 untested files)
4. **Replace print() with proper logging** (126+ instances)
5. **Add validation for BUCKET/SCENE_ID** (fail fast if empty)
6. **Add try/catch to all file I/O** operations

### Phase 3: Observability

1. **Implement structured logging** (JSON format with job_id, scene_id context)
2. **Add health check endpoints** for all services (deep checks)
3. **Set up alerting** for pipeline failures
4. **Integrate cost tracking** with API calls
5. **Add request/response logging** with sanitization for API calls

### Phase 4: Technical Debt

1. **Add lock files** to all 15 requirements files lacking them
2. **Standardize boolean env var parsing** - create single utility function
3. **Document all environment variables** in ENVIRONMENT_VARIABLES.md
4. **Add tests** for remaining 22 job modules
5. **Pin all Isaac Sim dependencies** with upper bounds

---

## Files Requiring Immediate Attention

### Critical Security Fixes
```
objects-job/run_objects_from_layout.py:17       # allow_pickle=True
particulate-service/particulate_service.py:563  # /debug endpoint
tests/test_reward_functions.py:25               # exec() usage
```

### Critical Data Quality Fixes
```
episode-generation-job/lerobot_exporter.py      # Checksums, atomic writes, validation
tools/quality_gates/quality_gate.py             # Override controls, thresholds
episode-generation-job/multi_format_exporters.py # Schema validation
```

### High Priority Error Handling
```
genie-sim-import-webhook/main.py:112-117        # Webhook error handling
interactive-job/run_interactive_assets.py:540   # API error handling
tools/asset_catalog/vector_store.py             # Database error handling
```

---

## Appendix: Complete Issue Counts by File

| File | Issues |
|------|--------|
| `episode-generation-job/lerobot_exporter.py` | 25+ |
| `tools/quality_gates/quality_gate.py` | 10+ |
| `episode-generation-job/multi_format_exporters.py` | 8+ |
| `tools/asset_catalog/vector_store.py` | 8+ |
| `tools/external_services/service_client.py` | 6+ |
| `genie-sim-export-job/geniesim_client.py` | 6+ |
| `particulate-service/particulate_service.py` | 5+ |
| `interactive-job/run_interactive_assets.py` | 5+ |
| `genie-sim-import-webhook/main.py` | 4+ |
| `variation-gen-job/generate_variation_assets.py` | 4+ |

---

*This audit was generated automatically. Please review findings and prioritize based on your specific business needs and risk tolerance.*
