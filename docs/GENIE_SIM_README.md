# GenieSim Server Patches Documentation

Complete technical documentation of the BlueprintPipeline patches for the upstream GenieSim server.

## Overview

The GenieSim server has three major issues that are fixed by BlueprintPipeline patches:

1. **EE Pose Handler** - Crashes when robot returns >2 values
2. **Object Pose Handler** - Fails to find prims due to path mismatches
3. **GRPC Server** - Type mismatches and unsafe unpacking throughout

This documentation package provides complete analysis, diagrams, code references, and debugging guides.

---

## Documentation Index

### 1. Quick Reference (START HERE)
**File:** `genie_sim_quick_ref.md` (12 KB, 30-second reads)

Best for: Getting started, troubleshooting, at-a-glance summaries

Contains:
- Problem/Solution summary table
- 30-second explanations of each fix
- Patch application order
- Testing checklist
- Troubleshooting guide
- Marker strings for verification

### 2. Technical Analysis
**File:** `genie_sim_analysis.md` (12 KB, comprehensive)

Best for: Understanding the "why" behind each patch

Contains:
- Upstream bug descriptions with code examples
- Detailed explanation of each BlueprintPipeline patch
- Safe unpacking patterns and monkey-patch strategies
- Fuzzy path resolution algorithm
- Deployment flow (Docker build process)
- Key insights and design philosophy
- Idempotency & safety mechanisms

### 3. Flow Diagrams
**File:** `genie_sim_patches_flow.md` (20 KB, visual reference)

Best for: Understanding code execution flow and architecture

Contains:
- EE Pose handler flow diagram (upstream → patched)
- Object Pose handler flow diagram (request → resolution → response)
- Patch application timeline (Docker build → runtime)
- Patch marker detection (idempotency)
- Error handling philosophy
- Path resolution scoring example

### 4. Code Reference
**File:** `genie_sim_code_reference.md` (16 KB, implementation details)

Best for: Developers implementing or debugging patches

Contains:
- File locations and line counts
- EE Pose patch code sections:
  - Monkey-patch wrapper injection
  - Safe unpacking patterns
  - Auto-wiring in robot init
- Object Pose patch code sections:
  - Fuzzy prim path resolver (full implementation)
  - Scoring weights and algorithm
  - Integration points
- GRPC Server patch code sections:
  - Safe float helper
  - Object pose unpacking
  - String wrapping for blocking_start_server
  - Joint position guards
- Patch marker verification
- Testing patterns for each component
- Debugging tips and common issues

---

## Quick Navigation

### For Different Roles

**DevOps/Infrastructure:**
1. Start with `genie_sim_quick_ref.md` - Understand what's patched
2. Read Docker Integration section
3. Check Patch Marker Strings for verification

**Backend Developers:**
1. Read `genie_sim_quick_ref.md` - High-level overview
2. Study `genie_sim_analysis.md` - Understand design decisions
3. Reference `genie_sim_code_reference.md` - Implementation details

**Debuggers/Support:**
1. Check `genie_sim_quick_ref.md` - Troubleshooting section first
2. Look at `genie_sim_patches_flow.md` - Understand execution flow
3. Search `genie_sim_code_reference.md` - Find specific code patterns

**System Architects:**
1. Review `genie_sim_analysis.md` - Full technical scope
2. Study `genie_sim_patches_flow.md` - Architecture diagrams
3. Understand error handling in `genie_sim_code_reference.md`

---

## Key Files Being Patched

### Source Code (Local)
```
tools/geniesim_adapter/
├── deployment/patches/
│   ├── patch_ee_pose_handler.py        (124 lines)
│   ├── patch_object_pose_handler.py    (164 lines)
│   ├── patch_grpc_server.py            (346 lines)
│   └── README
└── geniesim_server.py                   (297 lines - local mock)
```

### Upstream Code (Docker container)
```
/opt/geniesim/source/data_collection/server/
├── command_controller.py                (patched by: ee_pose, object_pose)
└── grpc_server.py                       (patched by: grpc_server)
```

---

## Core Concepts

### 1. EE Pose: Monkey-Patch Wrapper
**Problem:** Variable return values cause unpacking crashes
```python
pos, rot = robot.get_ee_pose()  # ❌ Crashes if >2 values
```

**Solution:** Wrap robot method to always return 2 values
```python
robot.get_ee_pose = _bp_wrap_ee_pose_impl(robot.get_ee_pose)
# Now always returns (pos, rot) or (None, None)
```

**Why This Works:**
- Applied after robot initialization
- Catches exceptions gracefully
- Normalizes all implementations to same interface

### 2. Object Pose: Fuzzy Path Resolution
**Problem:** Requested paths don't exist in USD stage
```python
prim = stage.GetPrimAtPath("/obj/model/cube")  # ❌ None
```

**Solution:** Search for prims by name with intelligent scoring
```python
best_path = self._bp_resolve_prim_path("/obj/model/cube")
# Returns "/World/cube" (found by name + scoring)

prim = stage.GetPrimAtPath(best_path)  # ✓ Found!
```

**Scoring Algorithm:**
- +100 if Xformable (geometric prim)
- +50 if under /World/ (standard location)
- -1 per "/" (prefer simpler hierarchies)

### 3. GRPC Server: Defensive Programming
**Problem:** Type mismatches and unsafe destructuring
```python
position, rotation = object_pose  # ❌ May not unpack correctly
rsp.msg = blocking_start_server(...)  # ❌ Returns dict, expects string
```

**Solution:** Type-check and provide sensible defaults
```python
_safe_float = float(val) or (parse_units(val) if isinstance(val, str) else 0.0)
rsp.msg = str(blocking_start_server(...) or "")
```

---

## Patch Application

### Docker Build
```dockerfile
# All patched during image build (idempotent):
COPY patches/ /tmp/patches/
RUN python3 /tmp/patches/patch_ee_pose_handler.py
RUN python3 /tmp/patches/patch_object_pose_handler.py
RUN python3 /tmp/patches/patch_grpc_server.py
```

### Safety Features
- **Marker Detection:** Each patch checks for its marker string and skips if found
- **Exception Handling:** All patches wrap critical operations in try/except
- **Logging:** All decisions logged with `[PATCH]` prefix
- **Graceful Fallback:** Never crashes, always returns sensible default

### Verification
```bash
# Check what's patched:
grep -c "BlueprintPipeline" command_controller.py  # Should be ≥2
grep -c "BlueprintPipeline" grpc_server.py         # Should be ≥1

# Check logs:
docker logs geniesim-server | grep "\[PATCH\]"
```

---

## Troubleshooting Guide

| Symptom | Likely Cause | Resolution |
|---------|--------------|-----------|
| EE pose still crashes | Monkey-patch not applied | Check for wrapper call after robot init |
| Object poses empty/identity | Path resolution not working | Verify _bp_resolve_prim_path called |
| GRPC errors on type mismatch | Patches not applied to grpc_server.py | Rebuild image, check logs |
| "Already patched" message | Expected - idempotent behavior | This is correct, no action needed |
| Prim path returns original path | Path doesn't exist in stage | Check logs for "WARNING" - this is fallback |

---

## Reading Path by Use Case

### "I need to understand the patches"
1. `genie_sim_quick_ref.md` - Overview (5 min)
2. `genie_sim_analysis.md` - Deep dive (15 min)
3. `genie_sim_code_reference.md` - Code details (10 min)

### "I need to debug a problem"
1. `genie_sim_quick_ref.md` - Troubleshooting section (2 min)
2. `genie_sim_patches_flow.md` - Execution flow (5 min)
3. `genie_sim_code_reference.md` - Relevant code section (5 min)

### "I need to implement similar patches"
1. `genie_sim_analysis.md` - Design philosophy (10 min)
2. `genie_sim_patches_flow.md` - Architecture (10 min)
3. `genie_sim_code_reference.md` - Copy/adapt patterns (10 min)

### "I need to verify patches in production"
1. `genie_sim_quick_ref.md` - Marker strings section (2 min)
2. Search for patch markers in code
3. Check logs for `[PATCH]` output

---

## Document Statistics

| File | Size | Lines | Focus |
|------|------|-------|-------|
| genie_sim_quick_ref.md | 12 KB | ~250 | Quick starts & troubleshooting |
| genie_sim_analysis.md | 12 KB | ~270 | Technical deep-dive |
| genie_sim_patches_flow.md | 20 KB | ~400 | Visual flows & diagrams |
| genie_sim_code_reference.md | 16 KB | ~420 | Code snippets & patterns |
| **Total** | **60 KB** | **~1340** | **Complete reference** |

---

## Key Takeaways

1. **Three patches address three distinct problems:**
   - EE Pose: Normalizes return values via monkey-patching
   - Object Pose: Fuzzy path resolution with intelligent scoring
   - GRPC Server: Safe type conversions with defensive unpacking

2. **All patches are idempotent:**
   - Marker strings prevent re-patching
   - Safe to include in CI/CD pipelines
   - Can be re-run without side effects

3. **All patches include comprehensive error handling:**
   - Type checking before operations
   - Sensible defaults on failures
   - Full logging for debugging

4. **Patches are production-ready:**
   - Extensively documented
   - Validated in BlueprintPipeline
   - Follow defensive programming principles

---

## See Also

- **Upstream GenieSim:** `/opt/geniesim/source/data_collection/server/`
- **Patch Scripts:** `/tools/geniesim_adapter/deployment/patches/`
- **Mock Server:** `/tools/geniesim_adapter/geniesim_server.py`
- **Docker Build:** `Dockerfile.geniesim-server`

---

## Document Generation Date

Generated: 2025-01-31  
Last Updated: 2025-01-31  
Version: 1.0

---

## Questions?

Refer to the relevant document for your question type:
- **"What does this patch do?"** → `genie_sim_analysis.md`
- **"How does this work?"** → `genie_sim_patches_flow.md`
- **"Show me the code"** → `genie_sim_code_reference.md`
- **"How do I...?"** → `genie_sim_quick_ref.md`

