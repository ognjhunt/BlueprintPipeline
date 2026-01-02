# Upsell Features Integration Analysis

**Date:** 2026-01-02
**Status:** CRITICAL GAPS IDENTIFIED

---

## Executive Summary

Deep analysis of the BlueprintPipeline reveals that **upsell features are completely disconnected from the automated pipeline**. While the modules exist and compile correctly, they are never invoked during normal pipeline execution.

---

## Critical Findings

### 1. Upsell Features Are Orphaned (CRITICAL)

**Problem:** The `upsell-features-job/` directory contains 9 Python modules with 6,172 lines of code, but:
- NO workflow triggers these modules
- NO job calls these modules
- NO integration point exists in `episode-generation-job/generate_episodes.py`

**Evidence:**
```bash
# Search for any reference to upsell features in workflows
grep -r "upsell" workflows/
# Result: NOTHING

# Search for any import of upsell modules in episode generation
grep -r "upsell\|language_annot\|vla_finetuning\|sim2real_service" episode-generation-job/
# Result: NOTHING
```

### 2. Data Pack Config Flag Unused (HIGH)

**Problem:** `data_pack_config.py` has a `include_language_annotations` flag that is set to `True` for Full pack, but it's NEVER connected to the `language_annotator.py` module.

**Location:** `episode-generation-job/data_pack_config.py:140`
```python
include_language_annotations: bool = False  # For Full pack, this is True
```

But in `generate_episodes.py`, this flag is never checked or acted upon.

### 3. Import Path Issues (MEDIUM)

**Problem:** Upsell modules use relative imports that won't work from other directories:

```python
# In upsell-features-job/language_annotator.py
from tools.llm_client import create_llm_client  # Requires REPO_ROOT in path

# In upsell-features-job/sim2real_service.py
from tools.sim2real.validation import ...  # Requires REPO_ROOT in path
```

These imports only work if running from REPO_ROOT or if PYTHONPATH is set correctly.

### 4. No Cloud Workflow for Upsell Features (HIGH)

**Problem:** There is no `workflows/upsell-features-pipeline.yaml` to:
- Trigger after `.episodes_complete` marker
- Run upsell features based on bundle tier
- Write `.upsell_complete` marker when done

### 5. Bundle Tier Not Propagated (HIGH)

**Problem:** The bundle tier configuration exists in `upsell-features-job/bundle_config.py` but:
- Episode generation workflow doesn't accept bundle tier parameter
- No environment variable controls which tier to use
- No way to select Pro/Enterprise features during automated runs

---

## Current Data Flow (Broken)

```
image → 3D-RE-GEN → regen3d-job → simready → usd-assembly → replicator
                                                              ↓
                                                        isaac-lab-job
                                                              ↓
                                                    episode-generation-job
                                                              ↓
                                                     .episodes_complete
                                                              ↓
                                                           [END]

                    upsell-features-job ← [NEVER CALLED]
```

---

## Required Fixes

### Fix 1: Create Upsell Features Workflow

Create `workflows/upsell-features-pipeline.yaml` that:
- Triggers on `.episodes_complete` marker
- Reads `BUNDLE_TIER` from scene config or environment
- Runs appropriate upsell features
- Writes `.upsell_complete` marker

### Fix 2: Add Post-Processing Hook to Episode Generation

Modify `episode-generation-job/generate_episodes.py` to:
- Accept `BUNDLE_TIER` environment variable
- After episode export, call upsell features based on tier
- Integrate language annotations directly into LeRobot export

### Fix 3: Create Unified Post-Processor

Create `upsell-features-job/post_processor.py` that:
- Reads bundle tier from environment/config
- Runs all applicable upsell features
- Can be called from episode generation or as standalone job

### Fix 4: Fix Import Paths

Update all upsell modules to:
- Use proper REPO_ROOT path setup
- Work when called from any location
- Handle missing dependencies gracefully

---

## Implementation Priority

| Fix | Priority | Effort | Impact |
|-----|----------|--------|--------|
| Post-processor integration | P0 | Medium | Enables all upsell features |
| Language annotations integration | P0 | Low | Immediate VLA value |
| Bundle tier propagation | P1 | Low | Enables tiered pricing |
| Upsell workflow | P1 | Medium | Cloud automation |
| Import path fixes | P2 | Low | Reliability |

---

## Verification Commands

After fixes, these should work:

```bash
# Run episode generation with Pro tier
BUNDLE_TIER=pro python episode-generation-job/generate_episodes.py

# Verify upsell outputs exist
ls scenes/test/episodes/upsell_outputs/
# Expected: language_annotations.json, vla_finetuning/, etc.

# Run standalone upsell processing
python upsell-features-job/post_processor.py --scene-dir ./scenes/test --tier enterprise
```

---

## Files Requiring Changes

1. `episode-generation-job/generate_episodes.py` - Add upsell integration
2. `upsell-features-job/post_processor.py` - NEW: Standalone post-processor
3. `workflows/upsell-features-pipeline.yaml` - NEW: Cloud workflow
4. `upsell-features-job/__init__.py` - Fix exports
5. `episode-generation-job/lerobot_exporter.py` - Add language annotation support

