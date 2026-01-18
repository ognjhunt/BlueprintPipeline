# Production Readiness Audit - BlueprintPipeline
**Date:** 2026-01-18
**Auditor:** Claude Code
**Scope:** Complete production readiness for Genie Sim data generation pipeline

---

## Executive Summary

This audit identifies **47 gaps** across the BlueprintPipeline codebase that need attention before full production release. Gaps are categorized by severity:

| Severity | Count | Description |
|----------|-------|-------------|
| **CRITICAL** | 8 | Blocks production data generation |
| **HIGH** | 12 | Significant functionality gaps |
| **MEDIUM** | 15 | Quality/reliability concerns |
| **LOW** | 12 | Nice-to-have improvements |

**Key Finding:** The pipeline CAN generate Genie Sim data and upload to Firebase Storage (GCS bucket `blueprint-8c1ca.appspot.com`). However, several gaps affect data quality, completeness, and operational reliability.

---

## CRITICAL GAPS (Must Fix Before Production)

### 1. Genie Sim Local Framework Requires Isaac Sim + cuRobo
**Location:** `tools/geniesim_adapter/local_framework.py:50-80`
**Impact:** Episode generation will fail without Isaac Sim environment
**Status:** BLOCKING

The local framework has hard dependencies:
```python
# Required but may not be installed
try:
    from omni.isaac.core import World
    from curobo.wrap.reacher.motion_gen import MotionGen
    ISAACSIM_AVAILABLE = True
except ImportError:
    ISAACSIM_AVAILABLE = False
```

**Resolution:**
- Ensure production GKE cluster has Isaac Sim containers deployed
- Verify `genie-sim-gpu-job` has access to GPU nodes with Isaac Sim
- Set `ISAACSIM_REQUIRED=true` in production environment

---

### 2. No Automatic Dataset Delivery to Labs
**Location:** Pipeline-wide
**Impact:** Generated data sits in GCS but isn't packaged for lab delivery
**Status:** BLOCKING for business model

**Current State:**
- Data uploads to `gs://blueprint-8c1ca.appspot.com/scenes/{SCENE_ID}/`
- Episodes land in `geniesim/` subdirectory
- No automated packaging/notification to labs

**Required:**
1. Create `dataset-delivery-job` that:
   - Packages LeRobot bundles from `import_manifest.json`
   - Generates dataset cards with provenance
   - Uploads to lab-specific delivery buckets
   - Sends notification webhooks

2. Add delivery workflow trigger after `.geniesim_import_complete`

---

### 3. Cost Tracking Uses Placeholder Values
**Location:** `tools/cost_tracking/tracker.py:44-46`
**Impact:** Cannot accurately bill labs or track margins
**Status:** BLOCKING for business

```python
COST_ESTIMATES = {
    "geniesim_job": 0.10,  # $0.10 per job (PLACEHOLDER)
    "geniesim_episode": 0.002,  # $0.002 per episode (PLACEHOLDER)
}
```

**Resolution:**
- Profile actual GPU costs on GKE
- Calculate true per-episode costs including:
  - GPU time (T4/A100 rates)
  - Storage costs
  - API calls (Gemini, etc.)
- Update with real values before first invoice

---

### 4. Dream2Flow Model Placeholder
**Location:** `dream2flow-preparation-job/flow_extractor/flow_extractor.py:139`
**Impact:** Optical flow computation uses fallback, not production model
**Status:** BLOCKING for DWM pipeline

```python
# PLACEHOLDER: Awaiting Dream2Flow model release (arXiv:2512.24766)
```

**Resolution:**
- Monitor Dream2Flow model release
- Implement proper model loading when available
- Until then, document that DWM pipeline produces lower-fidelity flow

---

### 5. MANO Hand Model Fallback in Production
**Location:** `dwm-preparation-job/hand_motion/hand_mesh_renderer.py`
**Impact:** Hand tracking videos use geometric boxes instead of anatomical hands
**Status:** HIGH (degraded quality)

**Current Behavior:**
- If MANO files unavailable → falls back to `SimpleHandMesh`
- Production should enforce MANO for quality

**Resolution:**
1. Obtain MANO license from MPI
2. Upload MANO files to `gs://${PROJECT_ID}-model-files/mano/`
3. Set `REQUIRE_MANO=true` in production
4. Add quality gate for hand model validation

---

### 6. Several NotImplementedError in Critical Paths
**Locations:**
| File | Line | Method |
|------|------|--------|
| `tools/asset_catalog/vector_store.py` | 66-86 | `search()`, `add()`, `remove()`, `count()` |
| `tools/arena_integration/mimic_integration.py` | 179 | `convert_to_mimic_format()` |
| `episode-generation-job/motion_planner.py` | 284 | `plan_bimanual()` |

**Resolution:**
- Vector store: Implement with Pinecone or FAISS for production
- Mimic integration: Complete if Arena export is needed
- Bimanual planning: Implement for humanoid robots (GR1, G1)

---

### 7. Particulate Service Placeholder URDFs in Production
**Location:** `interactive-job/run_interactive_assets.py:90`
**Impact:** Articulated objects may have fake/placeholder physics
**Status:** HIGH

**Current Safeguard:**
```python
if is_production and disallow_placeholder:
    raise ProductionModeError(
        "Set DISALLOW_PLACEHOLDER_URDF=false and disable production mode to allow placeholders."
    )
```

**Resolution:**
- Ensure `DISALLOW_PLACEHOLDER_URDF=true` in production
- Monitor Particulate service availability
- Implement retry logic for transient failures

---

### 8. Missing .geniesim_complete → Firebase Upload Automation
**Location:** `workflows/genie-sim-import-pipeline.yaml`
**Impact:** Data generated but not automatically delivered
**Status:** HIGH

**Current Flow:**
```
genie-sim-export → genie-sim-submit → genie-sim-import → .geniesim_import_complete
```

**Missing Step:**
```
.geniesim_import_complete → dataset-delivery → lab notification
```

---

## HIGH PRIORITY GAPS

### 9. Episode Quality Gate SLI Thresholds May Be Too Relaxed
**Location:** `tools/quality_gates/quality_gate.py:1340-1450`
**Impact:** Low-quality episodes may ship to labs

Default thresholds:
```python
"collision_free_rate_min": 0.8,  # 80% collision-free
"quality_pass_rate_min": 0.5,    # 50% pass rate
"quality_score_min": 0.85,       # 0.85 average score
```

**Resolution:**
- Review with labs what quality they expect
- Consider tier-based thresholds (Core: 0.8, Plus: 0.9, Full: 0.95)

---

### 10. No Automated Regression Testing for Generated Datasets
**Location:** N/A (missing)
**Impact:** Cannot verify dataset quality over time

**Required:**
- Golden dataset validation suite
- Automated comparison of new vs. baseline episodes
- Alert on quality degradation

---

### 11. LeRobot Conversion May Skip Episodes Silently
**Location:** `genie-sim-import-job/import_from_geniesim.py:846-849`
**Impact:** Episodes that fail conversion are logged but not alerted

```python
except Exception as e:
    print(f"[LEROBOT] Warning: Failed to convert episode {ep_metadata.episode_id}: {e}")
    skipped_count += 1
    continue  # Silently continues
```

**Resolution:**
- Add alerting when skip rate > threshold
- Include conversion failures in import manifest
- Consider `REQUIRE_LEROBOT=true` for production

---

### 12. Genie Sim gRPC Server Not Production Hardened
**Location:** `tools/geniesim_adapter/geniesim_grpc_pb2_grpc.py`
**Impact:** No TLS, no auth, basic error handling

**Current:**
```python
grpc.insecure_channel(f"{host}:{port}", options=options)
```

**Resolution:**
- Enable TLS for production
- Add authentication (JWT or mTLS)
- Implement circuit breaker pattern

---

### 13. Human Approval Workflow Uses Local Disk Storage
**Location:** `tools/quality_gates/quality_gate.py:193`
**Impact:** Approvals lost if container restarts

```python
APPROVALS_DIR = Path("/tmp/blueprintpipeline/approvals")
```

**Resolution:**
- Move to Firestore for approval persistence
- Or use GCS for approval requests

---

### 14. No Rate Limiting on LLM Calls
**Location:** `tools/llm_client/`
**Impact:** May hit API rate limits, causing job failures

**Resolution:**
- Implement exponential backoff (partially exists)
- Add request queuing for burst traffic
- Consider caching for repeated prompts

---

### 15. Asset Index Embedding Generation Can Fail Silently
**Location:** `tools/geniesim_adapter/asset_index.py:560-625`
**Impact:** Assets may have missing embeddings

```python
except Exception as e:
    if self.production_mode:
        raise RuntimeError(f"Embedding generation failed in production: {e}") from e
    # Non-production: returns fallback
```

**Resolution:**
- Ensure `PIPELINE_ENV=production` is set
- Add monitoring for embedding failures

---

### 16. No Checksums for Source Assets
**Location:** Pipeline-wide
**Impact:** Cannot verify input data integrity

**Resolution:**
- Add SHA256 checksums for input images
- Verify checksums before processing
- Include in provenance chain

---

### 17. Variation Asset Pipeline May Timeout
**Location:** `workflows/variation-assets-pipeline.yaml`
**Impact:** Large scenes may fail before completion

**Resolution:**
- Review timeout settings (currently 45 min)
- Implement checkpointing for resume
- Add progress tracking

---

### 18. No Multi-Region Redundancy
**Location:** Infrastructure
**Impact:** Single region failure = complete outage

**Resolution:**
- Deploy GKE cluster to multiple regions
- Use multi-region GCS bucket
- Implement failover workflows

---

### 19. Scene Generation Uses Gemini 3.0 Pro Image (Beta)
**Location:** `scene-generation-job/generate_scene_images.py`
**Impact:** Beta API may have breaking changes

**Resolution:**
- Monitor Gemini API stability
- Implement version pinning when GA
- Have fallback model option

---

### 20. Missing Embodiment Transfer for New Robots
**Location:** `tools/geniesim_adapter/multi_robot_config.py`
**Impact:** Only Franka, UR10, G2 fully supported

Currently supported:
- Franka Panda (default)
- Universal Robots UR10
- AGIBOT G2 (humanoid)

Not fully implemented:
- Fourier GR1
- Figure AI Figure 01
- Unitree H1

---

## MEDIUM PRIORITY GAPS

### 21. Test Coverage Below Target
**Location:** `pytest.ini`, `.coveragerc`
**Current:** ~70% coverage
**Target:** 85%+

Key gaps:
- `tools/geniesim_adapter/` - 65%
- `episode-generation-job/` - 60%
- `dwm-preparation-job/` - 55%

---

### 22. Documentation for Lab Integration Missing
**Location:** `docs/`
**Impact:** Labs don't know how to consume datasets

**Required docs:**
- Dataset format specification
- LeRobot integration guide
- Quality metrics explanation
- Provenance chain documentation

---

### 23. No Monitoring Dashboard
**Location:** `monitoring/`
**Impact:** Ops team cannot see pipeline health

**Required:**
- Grafana dashboard for:
  - Jobs per hour
  - Success rate by stage
  - Average quality scores
  - Cost per scene

---

### 24. Alerting Configuration Incomplete
**Location:** `monitoring/alerting.py`
**Impact:** Failures may go unnoticed

**Required alerts:**
- Job failure rate > 10%
- Quality score < threshold
- Cost anomalies
- Storage quota warnings

---

### 25. No Data Retention Policy Enforcement
**Location:** `workflows/retention-cleanup.yaml`
**Impact:** Storage costs grow unbounded

**Resolution:**
- Implement lifecycle policies on GCS
- Auto-archive scenes older than 90 days
- Delete raw assets after delivery

---

### 26. Secret Rotation Not Automated
**Location:** Infrastructure
**Impact:** Security risk from stale credentials

**Resolution:**
- Implement secret rotation for API keys
- Use Workload Identity where possible
- Audit secret access logs

---

### 27. No Canary Deployment Strategy
**Location:** Infrastructure
**Impact:** Bad deployments affect all scenes

**Resolution:**
- Deploy to 10% traffic first
- Monitor metrics before full rollout
- Implement rollback automation

---

### 28. Incomplete Error Categorization
**Location:** `tools/error_handling/`
**Impact:** Cannot distinguish retriable vs. permanent failures

**Required:**
- Categorize all error types
- Implement appropriate retry policies
- Document error codes

---

### 29. No A/B Testing Framework
**Location:** N/A (missing)
**Impact:** Cannot compare pipeline variations

---

### 30. USD Assembly Performance Not Optimized
**Location:** `usd-assembly-job/build_scene_usd.py`
**Impact:** Large scenes take excessive time

**Resolution:**
- Profile USD assembly bottlenecks
- Implement parallel asset loading
- Consider USD caching

---

### 31. Firestore Schema Not Versioned
**Location:** `tools/config/FIRESTORE_SCHEMA.md`
**Impact:** Schema changes may break clients

**Resolution:**
- Add schema version field
- Implement migration scripts
- Document breaking changes

---

### 32. No Circuit Breaker for External Services
**Location:** `tools/external_services/`
**Impact:** Single service failure cascades

**Resolution:**
- Implement circuit breaker pattern
- Add fallback behaviors
- Monitor external service health

---

### 33. Batch Processing Limited to 10 Scenes
**Location:** `tools/run_first_10_scenes.py`
**Impact:** Cannot efficiently process large backlogs

**Resolution:**
- Implement proper batch orchestration
- Add priority queuing
- Support thousands of scenes

---

### 34. No Idempotency Keys for Submissions
**Location:** `genie-sim-submit-job/`
**Impact:** Retries may create duplicate jobs

**Resolution:**
- Generate idempotency keys
- Check for existing submissions
- Deduplicate in import

---

### 35. Quality Config Not Validated on Load
**Location:** `tools/config/`
**Impact:** Invalid config may cause runtime failures

---

## LOW PRIORITY GAPS

### 36. TODOs in Codebase
**Count:** 15+ TODO comments
**Impact:** Technical debt

Notable:
- `workflows/TIMEOUT_AND_RETRY_POLICY.md:332` - Future retry improvements
- `docs/PRODUCTION_READINESS_AUDIT.md:503` - Reward function TODOs

---

### 37. Logging Not Structured
**Location:** Various
**Impact:** Log analysis is difficult

**Resolution:**
- Migrate to structured JSON logging
- Add correlation IDs
- Use Cloud Logging properly

---

### 38. No Performance Benchmarks
**Location:** N/A
**Impact:** Cannot track performance regressions

---

### 39. CLI Help Messages Incomplete
**Location:** Various entrypoints
**Impact:** Poor developer experience

---

### 40. No Changelog Automation
**Location:** `CHANGELOG.md`
**Impact:** Manual changelog maintenance

---

### 41. Docker Images Not Size-Optimized
**Location:** `*/Dockerfile`
**Impact:** Slower deployments, higher costs

---

### 42. No Pre-commit Hooks
**Location:** N/A
**Impact:** Code quality varies

---

### 43. Type Hints Incomplete
**Location:** Various
**Impact:** IDE support reduced

---

### 44. No API Versioning Strategy
**Location:** Various
**Impact:** Breaking changes affect clients

---

### 45. No Graceful Shutdown Handling
**Location:** Job entrypoints
**Impact:** Jobs may lose progress on termination

---

### 46. No Health Check Endpoints
**Location:** Services
**Impact:** Load balancer cannot verify health

---

### 47. No Chaos Engineering Tests
**Location:** N/A
**Impact:** Unknown failure modes

---

## Immediate Action Items for Production

### Phase 1: Critical Blockers (1-2 weeks)

1. **Verify Isaac Sim GKE Setup**
   - [ ] Confirm GPU nodes available
   - [ ] Test `genie-sim-gpu-job` end-to-end
   - [ ] Document deployment requirements

2. **Implement Dataset Delivery Workflow**
   - [ ] Create `dataset-delivery-job`
   - [ ] Add workflow trigger
   - [ ] Test with sample lab

3. **Update Cost Tracking**
   - [ ] Profile actual GPU costs
   - [ ] Update `COST_ESTIMATES`
   - [ ] Validate with finance

4. **Enforce Production Settings**
   ```bash
   PIPELINE_ENV=production
   REQUIRE_MANO=true
   DISALLOW_PLACEHOLDER_URDF=true
   ISAACSIM_REQUIRED=true
   ```

### Phase 2: High Priority (2-4 weeks)

5. Implement vector store for asset catalog
6. Add TLS to gRPC channels
7. Move approval workflow to Firestore
8. Set up monitoring dashboard
9. Configure production alerts

### Phase 3: Medium Priority (1-2 months)

10. Achieve 85% test coverage
11. Complete lab integration docs
12. Implement secret rotation
13. Add canary deployments
14. Optimize USD assembly

---

## Firebase Storage Upload Confirmation

**The pipeline DOES upload to Firebase Storage.**

Firebase Storage bucket: `gs://blueprint-8c1ca.appspot.com`

Data locations after pipeline completion:
```
gs://blueprint-8c1ca.appspot.com/scenes/{SCENE_ID}/
├── geniesim/
│   ├── export_manifest.json
│   ├── job.json
│   ├── recordings/
│   │   └── *.json (episode recordings)
│   ├── lerobot/
│   │   ├── dataset_info.json
│   │   ├── episodes.jsonl
│   │   └── episode_*.parquet
│   ├── import_manifest.json
│   ├── checksums.json
│   └── lerobot_bundle_{job_id}.tar.gz
├── .geniesim_complete (marker)
└── .geniesim_import_complete (marker)
```

**What's Missing:** Automated delivery/notification to labs after upload.

---

## Conclusion

The BlueprintPipeline is **architecturally sound** and has most components needed for production. The critical gaps are primarily:

1. **Operational**: Need proper deployment verification
2. **Business**: Need dataset delivery automation
3. **Quality**: Need MANO and Dream2Flow models

With the Phase 1 items addressed, you can begin production data generation. Phases 2-3 improve reliability and scale.
