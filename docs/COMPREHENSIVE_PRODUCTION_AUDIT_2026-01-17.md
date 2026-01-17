# Comprehensive Production Readiness Audit - BlueprintPipeline

**Date:** 2026-01-17
**Auditor:** Claude (Automated Deep Audit)
**Context:** Internal pipeline producing datasets sold to robotics labs
**Branch:** `claude/audit-production-readiness-59Ha5`

---

## Executive Summary

This comprehensive audit identified **~400+ issues** across 13 categories. Since BlueprintPipeline sells datasets to labs, **data quality and integrity issues are the most critical** - they directly impact customer trust and revenue.

**Key Finding:** The most urgent issues relate to data quality, security vulnerabilities, and incomplete integrations with Genie Sim 3.0 and Isaac Lab Arena. These could result in labs receiving corrupted/incomplete datasets, security breaches, or meaningless evaluation results.

---

## Priority Classification

| Priority | Definition | Count |
|----------|------------|-------|
| **P0 (Critical)** | Fix before production release - customer/security impact | 50+ |
| **P1 (High)** | Fix soon after launch - reliability/quality impact | 100+ |
| **P2 (Medium)** | Plan for future sprints - maintainability impact | 150+ |
| **P3 (Lower)** | Good to have - nice-to-have improvements | 100+ |

---

## 🔴 CRITICAL (P0) - Fix Before Production Release

### 1. Data Quality & Integrity Issues

**Impact: Labs receive corrupted/incomplete datasets they paid for**

| Issue | File | Line | Description |
|-------|------|------|-------------|
| **Missing checksums** | `episode-generation-job/lerobot_exporter.py` | 1296+ | No SHA256/MD5 for Parquet, JSON, video files - labs cannot verify integrity |
| **Partial writes without rollback** | `lerobot_exporter.py` | 1219-1243 | If export fails mid-write, orphaned files remain with no cleanup |
| **Mock data can reach production** | `sensor_data_capture.py` | 2014-2118 | `allow_mock_capture` can override production enforcement; WARNING not ERROR |
| **Frame count mismatches exported** | `lerobot_exporter.py` | 1188-1199 | Episodes with mismatched trajectory/sensor frames exported (warning only) |
| **Incomplete episodes exported** | `lerobot_exporter.py` | 1146-1158 | Missing trajectories/task descriptions produce warnings but still export |
| **Data tier mismatch** | `lerobot_exporter.py` | 1160-1172 | "Full" pack episodes could have Core pack data - no blocking validation |
| **Silent frame drops** | `lerobot_exporter.py` | 1800-1809 | Invalid RGB frames skipped silently; dataset says "X frames" but has fewer |
| **No data lineage** | `lerobot_exporter.py` | 1669-1707 | Labs cannot trace which sim backend, physics params, or quality gates were used |
| **Quality gates bypassable** | `tools/quality_gates/quality_gate.py` | 412-453 | Override requires only 10-char reason - no audit trail |
| **Auto-approve on timeout** | `quality_gate.py` | 266-270 | Failed gates auto-approved after 24h with no human review |
| **Weak thresholds** | `quality_gate.py` | 1130-1184 | 50% quality_pass_rate_min, 20% collision allowed - marginal data accepted |
| **Parquet corruption risk** | `lerobot_exporter.py` | 1281-1305 | Non-atomic writes; crash mid-write leaves corrupted files |
| **Reward signal corruption** | `lerobot_exporter.py` | 436-473 | Fallback uses quality_score without NaN/Inf validation |
| **Schema validation gaps** | `multi_format_exporters.py` | 518-600 | HDF5 export silently passes on conversion failures |
| **Sensor data loss** | `sensor_data_capture.py` | 1787-1812 | Capture failures set sensor_data to None, episodes export without visual obs |

**Recommended Fix Priority: P0 - Block production release**

---

### 2. Critical Security Vulnerabilities

| Issue | File | Line | Severity |
|-------|------|------|----------|
| **exec() on dynamic code** | `tools/isaac_lab_tasks/reward_functions.py` | 738 | HIGH - Code injection risk even with AST validation |
| **Debug endpoint requires explicit enablement** | `particulate-service/particulate_service.py` | 69-70 | HIGH - `PARTICULATE_DEBUG` defaults to `"0"` and requires a token |
| **Hardcoded mock API key** | `genie-sim-export-job/geniesim_client.py` | 561 | HIGH - `"mock-api-key"` in production code |
| **No size limit on base64 decode** | `particulate_service.py` | 829-838 | HIGH - DoS via massive payload |
| **SSRF in health check URLs** | `genie-sim-import-webhook/main.py` | 17-21 | MEDIUM - User-controllable URLs |
| **API key exposure in HTTP headers** | `geniesim_client.py` | 588-591 | MEDIUM - Credentials in headers |
| **Traceback exposure** | 17+ files | Various | MEDIUM - Stack traces leak internal paths |

---

### 3. Genie Sim 3.0 Integration Gaps

Based on [Genie Sim 3.0 official repository](https://github.com/AgibotTech/genie_sim):

| Issue | File | Line | Description |
|-------|------|------|-------------|
| **11 gRPC methods not implemented** | `tools/geniesim_adapter/geniesim_grpc_pb2_grpc.py` | 363-473 | All stub methods raise `NotImplementedError("Method not implemented!")` |
| **Circuit breaker non-functional** | `genie-sim-export-job/geniesim_client.py` | 574-576 | `_circuit_breaker` initialized to `None`, never instantiated |
| **Quality gate bypass possible** | `export_to_geniesim.py` | 737-744 | Continues if gates unavailable despite `require_quality_gates=True` |
| **Mock mode via env var** | `geniesim_client.py` | 548-552 | `GENIESIM_MOCK_MODE` can enable mock in production |
| **No manifest checksum validation** | `import_manifest_utils.py` | 148-155 | Checksum computed but never verified downstream |
| **Missing version negotiation** | `submit_to_geniesim.py` | 44-45 | No capability negotiation before using Genie Sim 3.0 features |
| **Hardcoded gRPC host/port** | `local_framework.py` | 208-214 | `localhost:50051` hardcoded, breaks cloud/container deployment |
| **No connection pooling** | `local_framework.py` | 311-351 | New channel created for each connection |
| **Timeout logic backwards** | `local_framework.py` | 360 | Uses `min()` instead of `max()` for timeout |
| **gRPC channel never closed** | `local_framework.py` | 396-403 | No context manager, connection leak risk |
| **Commercial asset filtering bypass** | `export_to_geniesim.py` | 389-421 | `commercial_ok=True` flag can be set arbitrarily |
| **Rate limit not respected** | `geniesim_client.py` | 926-931 | Caps wait at 5 min even if server says longer |

---

### 4. Isaac Lab Arena Integration Gaps

Based on [NVIDIA Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena):

| Issue | File | Line | Description |
|-------|------|------|-------------|
| **Mock GR00T returns random actions** | `groot_integration.py` | 390-450 | `_MockGR00TModel.forward()` returns `np.random.randn()` - meaningless |
| **Policy loading silently fails** | `evaluation_runner.py` | 361 | `except ImportError: pass` - no logging |
| **LLM goal decomposition not implemented** | `composite_tasks.py` | 586-623 | Only 3 hardcoded templates, no actual LLM |
| **Arena API compatibility not validated** | `arena_exporter.py` | 308-330 | Shadow `AffordanceType` hardcodes 11 types |
| **Placeholder evaluation results** | `parallel_evaluation.py` | 424-450 | Mock uses 50% random success rate |
| **Missing benchmark compliance** | All Arena files | - | No validation against official Arena format |
| **Hardcoded Omniverse paths** | `components.py` | 370-374 | `localhost` default breaks cloud |
| **No policy compatibility checking** | All evaluation files | - | No validation policy matches embodiment |
| **Task chain validation not called** | `composite_tasks.py` | 248-284 | `validate()` exists but never invoked |
| **Incomplete LeRobot Hub metadata** | `lerobot_hub.py` | 140-176 | Required fields may be empty |

---

### 5. Workflow Orchestration Critical Issues

| Issue | File | Line | Description |
|-------|------|------|-------------|
| **Syntax error** | `workflows/training-pipeline.yaml` | 74-75 | Duplicate `next:` clauses - step never executes |
| **No error handling** | `workflows/retention-cleanup.yaml` | 32-48 | Job failures not captured |
| **No timeout** | `workflows/retention-cleanup.yaml` | 32-48 | Job could hang indefinitely |
| **No retry policy** | `workflows/retention-cleanup.yaml` | 32-48 | Transient failures fail entire run |
| **Race conditions** | `workflows/usd-assembly-pipeline.yaml` | 12-16 | 3 pipelines trigger on same marker |
| **No rollback mechanisms** | All 17 workflows | - | Partial outputs not cleaned on failure |
| **Extremely long timeout** | `workflows/episode-generation-pipeline.yaml` | 80 | 21600s (6 hours) masks hanging jobs |

---

## 🟠 HIGH (P1) - Fix Soon After Launch

### 6. Testing Coverage Gaps

| Module | Tests | Lines of Code | Status |
|--------|-------|---------------|--------|
| **upsell-features-job** | 0 | ~8,000+ | CRITICAL - Customer output |
| **arena-export-job** | 0 | ~2,500+ | HIGH - Benchmarking |
| **usd-assembly-job** | 0 | ~2,200+ | CRITICAL - Scene generation |
| **regen3d-job** | 0 | ~600+ | CRITICAL - First pipeline stage |
| **genie-sim-export-job** | 0 | ~3,500+ | HIGH - External integration |
| **episode-generation-job** | 5/21 | ~15,000+ | 24% coverage only |
| **dwm-preparation-job** | 0 | ~2,500+ | MEDIUM |
| **smart-placement-engine-job** | 0 | ~1,500+ | MEDIUM |

**Total: 18/23 job directories have ZERO test coverage**

---

### 7. Error Handling Gaps (138+ instances)

| Pattern | Count | Example Files |
|---------|-------|---------------|
| Bare `except Exception: pass` | 51+ | `generate_episodes.py:2923-2935` |
| Missing file I/O error handling | 25+ | `import_from_geniesim.py:288` |
| Missing API error handling | 15+ | `run_interactive_assets.py:540` |
| Silent fallbacks masking failures | 20+ | `generate_isaac_lab_task.py:492` |
| Subprocess without timeout | 10+ | `run_variation_asset_pipeline.py:1225` |
| Missing null checks | 30+ | Array access without bounds checking |
| Unhandled JSON parsing | 18+ | `json.loads()` without try/catch |

---

### 8. Incomplete Features/Placeholders (50+)

| Feature | Files | Status |
|---------|-------|--------|
| **Dream2Flow model** | `dream2flow-preparation-job/*` | NOT RELEASED - placeholder video |
| **3D-RE-GEN code** | `tools/regen3d_adapter/__init__.py` | PENDING (~Q1 2025 release) |
| **MANO hand model** | `dwm-preparation-job/hand_motion/` | PLACEHOLDER - SimpleHandMesh fallback |
| **Tactile sensor simulation** | `data_pack_config.py:273-301` | CONFIGURED BUT NOT IMPLEMENTED |
| **Inventory enrichment** | `tools/inventory_enrichment/enricher.py` | STUB MODE - pass-through |
| **Cost tracking prices** | `tools/cost_tracking/tracker.py:44-46` | PLACEHOLDER - $0.10/job fake |
| **GR00T policy evaluation** | `groot_integration.py:252-253` | Falls back to mock silently |
| **50+ P0/P1/P2 FIX comments** | Throughout codebase | Marked but unfixed |

---

### 9. Sensor Data Capture Issues

| Issue | File | Line | Risk |
|-------|------|------|------|
| Mock warnings only shown once | `sensor_data_capture.py` | 1553-1558 | Silent after first call |
| Velocity data silently dropped | `sensor_data_capture.py` | 1155 | `except Exception: pass` |
| No camera existence validation | Throughout | - | Assumes cameras work |
| Frame timestamps not verified | Multiple | - | Temporal misalignment |
| Mock sensor label as Isaac Sim | `sensor_data_capture.py` | 2014+ | Data mislabeled |

---

## 🟡 MEDIUM (P2) - Plan for Future Sprints

### 10. Hardcoded Values (100+ locations)

| Category | Count | Examples |
|----------|-------|----------|
| gRPC ports | 7 | `50051` in multiple files |
| File paths | 15+ | `/app/*`, `/opt/*`, `/tmp/*` |
| Timeouts | 30+ | 10s to 21600s inconsistent |
| Quality thresholds | 20+ | `0.85` duplicated everywhere |
| Model names | 15+ | `gemini-3-pro-preview` hardcoded |
| GPU specs | 5 | `nvidia-tesla-t4` hardcoded |
| Retry counts | 15+ | 3-5 inconsistent |
| Batch sizes | 10+ | 16, 32, 48, 64 scattered |

---

### 11. Logging & Observability Gaps

| Issue | Count | Impact |
|-------|-------|--------|
| `print()` instead of logging | 126+ | No log levels, filtering, structure |
| Files without logging imports | 34 | Critical operations unlogged |
| Missing health checks | 10+ services | No deep checks for dependencies |
| No structured logging | 0 files | All logs unstructured strings |
| No alerting system | 0 configured | Pipeline failures unnoticed |

**Files with Most print() Statements:**
- `regen3d-job/regen3d_adapter_job.py` - 60+
- `isaac-lab-job/generate_isaac_lab_task.py` - 45+
- `variation-gen-job/generate_variation_assets.py` - 40+

---

### 12. Performance Issues

| Issue | Location | Impact |
|-------|----------|--------|
| N+1 database pattern | `vector_store.py:207` | Individual INSERTs in loop |
| Unbounded queries | `vector_store.py:430` | `SELECT *` without LIMIT |
| Blocking polling loops | `quality_gate.py:300` | `while True` with 30s sleep |
| Memory accumulation | `vector_store.py:541` | `_metadata_store` grows forever |
| No streaming for large files | `geniesim_client.py:1332` | Loads all episodes into memory |
| Blocking in async code | `service_client.py:339` | `time.sleep()` blocks event loop |

---

### 13. Environment Configuration Issues

| Issue | Count | Impact |
|-------|-------|--------|
| BUCKET/SCENE_ID default to "" | 40+ files | Pipeline runs with empty bucket |
| Production mode fragmented | 14 files | Inconsistent SIMREADY_PRODUCTION_MODE vs PRODUCTION_MODE |
| 3 boolean parsers | 3 implementations | Duplicated code |
| Undocumented env vars | 11+ | PARTICULATE_MODE, GENIESIM_MOCK_MODE, etc. |
| Hardcoded localhost | 10+ locations | Breaks remote deployment |
| Inconsistent defaults | 6+ files | DATA_QUALITY_LEVEL varies |

---

### 14. Dependency & Versioning Issues

| Issue | Impact |
|-------|--------|
| Unpinned Isaac Sim dependencies | 33+ packages with `>=` only |
| Missing lock files | 15/21 requirements files lack locks |
| Python version mismatch | 5 jobs use 3.10, 14 use 3.11 |
| Version conflicts | pytest `==8.3.4` vs `>=7.4.0` |

---

## 🟢 LOWER (P3) - Good to Have

### 15. Documentation Gaps

| Category | Missing |
|----------|---------|
| README files | 52 directories lack READMEs |
| API documentation | 5+ major APIs undocumented |
| Architecture diagrams | Data flow diagrams missing |
| Changelog | No CHANGELOG file |
| Operations runbook | 14+ procedures undocumented |

---

## Summary Statistics

| Category | Issues Found | Priority | Effort |
|----------|--------------|----------|--------|
| **Data Quality/Integrity** | 15+ critical | P0 | HIGH |
| **Security** | 7 critical | P0 | MEDIUM |
| **Genie Sim Integration** | 12 critical | P0 | HIGH |
| **Isaac Lab Arena Integration** | 10 critical | P0 | HIGH |
| **Workflow Issues** | 8 critical | P0 | MEDIUM |
| **Testing Coverage** | 22 modules untested | P1 | HIGH |
| **Error Handling** | 138+ gaps | P1 | HIGH |
| **Incomplete Features** | 50+ placeholders | P1 | VARIES |
| **Sensor Data Capture** | 10+ issues | P1 | MEDIUM |
| **Hardcoded Values** | 100+ locations | P2 | LOW |
| **Logging/Monitoring** | 126+ issues | P2 | MEDIUM |
| **Performance** | 20+ issues | P2 | MEDIUM |
| **Environment Config** | 70+ issues | P2 | LOW |
| **Dependencies** | 15+ conflicts | P2 | LOW |
| **Documentation** | 100+ gaps | P3 | LOW |
| **TOTAL** | **~400+ issues** | - | - |

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
8. **Fail hard if mock sensor data in production** - not just warning

### Phase 2: Security & Reliability

1. **Disable debug endpoint by default** in particulate-service
2. **Remove hardcoded mock API key** from geniesim_client.py
3. **Implement size limits** on base64 payload decoding
4. **Fix workflow syntax error** in training-pipeline.yaml
5. **Implement all 11 gRPC stub methods** in geniesim_grpc_pb2_grpc.py
6. **Add tests for episode-generation-job** (21 untested files)
7. **Replace print() with proper logging** (126+ instances)

### Phase 3: Integrations

1. **Implement Genie Sim circuit breaker** (currently None)
2. **Replace mock GR00T** with proper SDK integration or fail explicitly
3. **Implement actual LLM goal decomposition** for Arena composite tasks
4. **Add Arena API version compatibility checking**
5. **Fix workflow race conditions** with distributed locking
6. **Implement manifest checksum verification** for Genie Sim imports

### Phase 4: Technical Debt

1. **Add lock files** to all 15 requirements files lacking them
2. **Standardize boolean env var parsing** - create single utility function
3. **Document all environment variables** in ENVIRONMENT_VARIABLES.md
4. **Add tests** for remaining 22 job modules
5. **Externalize 100+ hardcoded values** to configuration
6. **Pin all Isaac Sim dependencies** with upper bounds

---

## Files Requiring Immediate Attention

### Critical Security Fixes
```
particulate-service/particulate_service.py:69-70    # Debug default enabled
genie-sim-export-job/geniesim_client.py:561         # Hardcoded mock API key
tools/isaac_lab_tasks/reward_functions.py:738       # exec() usage
```

### Critical Data Quality Fixes
```
episode-generation-job/lerobot_exporter.py          # Checksums, atomic writes, validation
episode-generation-job/sensor_data_capture.py       # Mock data enforcement
tools/quality_gates/quality_gate.py                 # Override controls, thresholds
```

### Critical Integration Fixes
```
tools/geniesim_adapter/geniesim_grpc_pb2_grpc.py    # 11 unimplemented methods
tools/arena_integration/groot_integration.py        # Mock returns random actions
workflows/training-pipeline.yaml                    # Syntax error
```

---

## External References

- [NVIDIA Isaac Lab-Arena](https://developer.nvidia.com/isaac/lab-arena) - Official framework
- [Isaac Lab-Arena GitHub](https://github.com/isaac-sim/IsaacLab-Arena) - Source repository
- [Genie Sim 3.0 GitHub](https://github.com/AgibotTech/genie_sim) - AGIBOT repository
- [Genie Sim 3.0 ArXiv Paper](https://arxiv.org/html/2601.02078v1) - Technical details

---

*This audit was generated automatically on 2026-01-17. Please review findings and prioritize based on your specific business needs and risk tolerance.*
