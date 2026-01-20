# Production Readiness Audit: BlueprintPipeline (Genie Sim Mode)

**Date:** 2026-01-20
**Overall Readiness:** ~82%
**Target:** 100% for Genie Sim data production pipeline

---

## Executive Summary

The BlueprintPipeline's Genie Sim mode is **substantially complete** for producing robot manipulation datasets that auto-upload to Firebase Storage. The core flow from `scene.usda` → episode generation → Firebase is fully implemented with proper LeRobot format output compatible with Genie Sim 3.0.

However, there are critical gaps in error handling, test coverage, and security that must be addressed before production deployment.

---

## Readiness by Category

| Category | Status | Readiness |
|----------|--------|-----------|
| **Core Pipeline Flow** | ✅ Complete | 95% |
| **Genie Sim Export** | ✅ Complete | 95% |
| **Genie Sim Submit/Collection** | ✅ Complete | 90% |
| **Genie Sim Import** | ✅ Complete | 95% |
| **Firebase Auto-Upload** | ✅ Complete | 85% |
| **Data Format (LeRobot)** | ✅ Complete | 95% |
| **Error Handling** | ⚠️ Gaps | 70% |
| **Test Coverage** | ⚠️ Gaps | 55% |
| **Security** | ⚠️ Gaps | 75% |
| **Configuration** | ⚠️ Gaps | 80% |

---

## Critical Gaps (P0 - Must Fix Before Production)

### 1. Firebase Upload Null Reference Risk

**File:** `genie-sim-submit-job/submit_to_geniesim.py:1348-1365`

**Issue:** Firebase result summary accessed without null checks:
```python
firebase_result.summary.get("uploaded", 0)  # What if summary is None?
```

**Impact:** Job crash during Firebase upload reporting
**Fix:** Add defensive null checks before accessing `.summary`

---

### 2. JSON Parsing Without Schema Validation

**Files:** `submit_to_geniesim.py:115, 383, 390, 617`

**Issue:** JSON downloaded from GCS is parsed without validating structure. Corrupted/malformed JSON could crash the pipeline silently.

**Impact:** Silent data corruption or cryptic failures
**Fix:** Add JSON schema validation after parsing

---

## High Priority Gaps (P1)

### 3. Test Coverage for Premium Export Modules: ZERO

**Files:** All 10 `default_*.py` modules in `genie-sim-export-job/`

| Module | Size | Tests |
|--------|------|-------|
| `default_premium_analytics.py` | 22KB | 0 |
| `default_sim2real_fidelity.py` | 14KB | 0 |
| `default_embodiment_transfer.py` | 17KB | 0 |
| `default_trajectory_optimality.py` | 10KB | 0 |
| `default_policy_leaderboard.py` | 11KB | 0 |
| `default_tactile_sensor_sim.py` | 2.4KB | 0 |
| `default_language_annotations.py` | 2.5KB | 0 |
| `default_generalization_analyzer.py` | 3KB | 0 |
| `default_sim2real_validation.py` | 19KB | 0 |
| `default_audio_narration.py` | 21KB | 0 |

**Impact:** Bugs in export features won't be caught until production
**Fix:** Add unit tests for each module's factory function

---

### 4. E2E Tests Use Heavy Mocks That Don't Match Reality

**File:** `tests/test_geniesim_full_flow_e2e.py`

**Issue:** Current E2E tests use fake implementations:
- `fake_run_local_data_collection()` - creates 1 placeholder episode
- `fake_upload_episodes_to_firebase()` - returns success without validation

**Impact:** Tests pass but production fails
**Fix:** Add integration tests with real gRPC stubs

---

### 5. Insecure gRPC Channels in Development

**Files:** `tools/geniesim_adapter/geniesim_server.py:359, 400`

**Issue:**
```python
channel = grpc.insecure_channel(target)  # No TLS
server.add_insecure_port(...)  # Unencrypted
```

**Impact:** Credentials transmitted in plaintext during development
**Fix:** Enforce TLS even in dev with self-signed certs

---

## Medium Priority Gaps (P2)

### 6. Placeholder Embeddings Fallback

**File:** `genie-sim-export-job/export_to_geniesim.py:975-982`

When no embedding provider available, falls back to placeholder embeddings without failing.

**Fix:** Set `REQUIRE_EMBEDDINGS=true` in production environment

---

### 7. DEBUG_TOKEN Stored in Environment Variable

**File:** `particulate-service/particulate_service.py:133`

Debug authentication token stored in plaintext environment variable.

**Fix:** Use Google Secret Manager for DEBUG_TOKEN

---

### 8. Firebase Credentials in JSON Environment Variable

**File:** `tools/firebase_upload/uploader.py:40-52`

`FIREBASE_SERVICE_ACCOUNT_JSON` allows full credentials as env var (can be logged).

**Fix:** Require file path only (`FIREBASE_SERVICE_ACCOUNT_PATH`) in production

---

### 9. Firestore Schema Migration Logic Not Implemented

**File:** `docs/firestore_schema_migration.md:54`

```python
# TODO: apply transformation logic when bumping schema versions.
data["schema_version"] = to_version
```

**Impact:** Schema version upgrades won't migrate data correctly
**Fix:** Implement migration transformation functions

---

### 10. Missing Null Checks on Dict Access (12+ instances)

**Files:** Multiple in submit/import jobs

**Pattern:** `getattr(result, "episodes_collected", 0)` could return non-integer.

**Fix:** Add type validation for all result object attributes

---

## Lower Priority Gaps

### 11. No Tests for Core Episode Generation Modules

| Module | Lines | Tests |
|--------|-------|-------|
| `collision_aware_planner.py` | 1,095 | 0 |
| `curobo_planner.py` | 878 | 0 |
| `isaac_sim_integration.py` | ~500 | 0 |
| `pytorch_dataloaders.py` | 779 | 0 |

---

### 12. Hardcoded Meshy API URL

**File:** `meshy-job/run_meshy_from_assets.py:20`
```python
MESHY_BASE = "https://api.meshy.ai"  # No env var override
```

---

### 13. Multiple Production Mode Flags Cause Confusion

**File:** `tools/config/production_mode.py`

8+ different ways to enable production mode creates confusion:
- `PIPELINE_ENV=production`
- `PRODUCTION_MODE=1`
- `SIMREADY_PRODUCTION_MODE=1`
- `ISAAC_SIM_REQUIRED=1`
- And more...

**Fix:** Consolidate to single canonical `PIPELINE_ENV` variable

---

### 14. Silent Exception Swallowing in Cleanup

**File:** `firebase_upload_orchestrator.py:188-222`

Cleanup failures only logged at WARNING level, not propagated.

**Impact:** Failed cleanup operations invisible in monitoring

---

### 15. Dream2Flow Model Not Released (Placeholder Only)

**File:** `dream2flow-preparation-job/flow_extractor/flow_extractor.py:139`

Dream2Flow pipeline is scaffolding only - model from arXiv:2512.24766 not yet released.

**Impact:** If `--enable-dream2flow` is used, you get placeholder outputs
**Fix:** Set `DREAM2FLOW_REQUIRE_REAL_BACKENDS=1` to fail if enabled without model

---

## What's Working Well

1. **Complete Scene → Episodes → Firebase Flow**
   - `scene.usda` → genie-sim-export → genie-sim-submit → genie-sim-import → Firebase ✅

2. **LeRobot Format Export** (Genie Sim 3.0 compatible)
   - Parquet episodes with proper schema ✅
   - Video recordings per camera ✅
   - `meta/info.json`, `tasks.jsonl`, `episodes.jsonl` ✅

3. **Multi-Robot Support**
   - Franka, UR10, G2, GR1, Fetch all supported ✅
   - Per-robot metrics tracking ✅

4. **Quality Gates Framework**
   - Collision-free rate validation ✅
   - Task success rate tracking ✅
   - Quality score thresholds ✅

5. **Firebase Upload with Retry**
   - MD5 + SHA256 verification ✅
   - Automatic retry with backoff ✅
   - Cleanup on failure ✅

6. **Circuit Breaker Pattern for gRPC**
   - Failure threshold tracking ✅
   - Automatic recovery ✅

7. **Commercial Asset Filtering**
   - CC0/CC-BY/MIT/Apache-2.0 enforcement ✅
   - Asset provenance tracking ✅

---

## Required Production Environment Variables

```bash
# Core
PIPELINE_ENV=production
BUCKET=your-gcs-bucket

# Genie Sim
USE_GENIESIM=true
GENIESIM_HOST=localhost  # or your server
GENIESIM_PORT=50051
ISAAC_SIM_PATH=/isaac-sim
GENIESIM_ROOT=/opt/geniesim
ISAACSIM_REQUIRED=true
CUROBO_REQUIRED=true

# Firebase
FIREBASE_STORAGE_BUCKET=your-bucket.appspot.com
FIREBASE_SERVICE_ACCOUNT_PATH=/secrets/firebase.json
ENABLE_FIREBASE_UPLOAD=true
FIREBASE_UPLOAD_PREFIX=datasets

# Quality Enforcement
REQUIRE_EMBEDDINGS=true
FILTER_COMMERCIAL=true
DISALLOW_PLACEHOLDER_URDF=true
DREAM2FLOW_REQUIRE_REAL_BACKENDS=true

# Quality Thresholds
BP_QUALITY_EPISODES_QUALITY_SCORE_MIN=0.90
BP_QUALITY_EPISODES_COLLISION_FREE_RATE_MIN=0.90
```

---

## Recommended Fix Order

| Order | Item | Effort | Impact |
|-------|------|--------|--------|
| 1 | Firebase null checks | 1 hour | Prevents crashes |
| 2 | JSON schema validation | 2 hours | Data integrity |
| 3 | Set production env vars | 30 min | Enable safeguards |
| 4 | Add premium module tests | 4 hours | Catch bugs |
| 5 | Fix insecure gRPC | 2 hours | Security |
| 6 | Move secrets to Secret Manager | 2 hours | Security |
| 7 | Consolidate production flags | 1 hour | Configuration clarity |
| 8 | Add integration tests | 1-2 days | Long-term stability |

---

## Path to 100%

**Current: ~82%**

| Milestone | Fixes | New % |
|-----------|-------|-------|
| Fix P0 Critical (1-2) | Firebase nulls + JSON validation | 87% |
| Fix P1 High (3-5) | Tests + gRPC TLS | 92% |
| Fix P2 (6-10) | Config + secrets | 96% |
| Fix Medium (11-15) | Cleanup | 100% |

**Estimated time to 100%:** 2-3 focused development days for critical + high priority fixes.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GENIE SIM PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. INPUT                                                           │
│     scenes/{scene_id}/                                              │
│     ├── scene_manifest.json                                         │
│     └── usd/scene.usda                                              │
│                          │                                          │
│                          ▼                                          │
│  2. GENIE-SIM-EXPORT-JOB                                            │
│     - Validates manifest + USD                                      │
│     - Generates scene_graph.json                                    │
│     - Generates asset_index.json                                    │
│     - Generates task_config.json                                    │
│     - Premium feature configs                                       │
│     - Asset provenance check                                        │
│                          │                                          │
│                          ▼                                          │
│  3. GENIE-SIM-SUBMIT-JOB                                            │
│     - Server handshake (version check)                              │
│     - GenieSimLocalFramework                                        │
│     - gRPC data collection                                          │
│     - Multi-robot execution                                         │
│     - Quality metrics tracking                                      │
│                          │                                          │
│                          ▼                                          │
│  4. GENIE-SIM-IMPORT-JOB                                            │
│     - Episode schema validation                                     │
│     - Quality score filtering                                       │
│     - LeRobot format conversion                                     │
│     - Import manifest generation                                    │
│                          │                                          │
│                          ▼                                          │
│  5. FIREBASE UPLOAD                                                 │
│     - Concurrent uploads (8 workers)                                │
│     - MD5 + SHA256 verification                                     │
│     - Retry with exponential backoff                                │
│     - Path: datasets/{scene_id}/{robot_type}/                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Output Data Structure (LeRobot Format)

```
datasets/{scene_id}/
├── meta/
│   ├── info.json                # Dataset metadata
│   ├── stats.json               # Per-feature statistics
│   ├── tasks.jsonl              # Task descriptions
│   └── episodes.jsonl           # Episode metadata
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── episode_000001.parquet
├── videos/
│   └── chunk-000/
│       ├── observation.images.wrist/
│       │   └── episode_000000.mp4
│       └── observation.images.overhead/
│           └── episode_000000.mp4
└── ground_truth/                # Plus/Full packs
    └── chunk-000/
        ├── depth/
        ├── segmentation/
        └── bboxes/
```

---

## References

- [AgibotTech/genie_sim GitHub](https://github.com/AgibotTech/genie_sim)
- [Genie Sim 3.0 arXiv Paper](https://arxiv.org/html/2601.02078v1)
- [Hugging Face GenieSimAssets](https://huggingface.co/datasets/agibot-world/GenieSimAssets)
