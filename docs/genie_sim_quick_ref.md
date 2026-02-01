# GenieSim Patches: Quick Reference

## One-Pager Summary

| Component | Problem | Solution | Key Feature |
|-----------|---------|----------|-------------|
| **EE Pose** | `pos, rot = robot.get_ee_pose()` crashes with >2 values | Safe unpacking + monkey-patch wrapper | Always normalizes to 2-tuple |
| **Object Pose** | Requested prim paths don't exist in USD stage | Fuzzy path resolution with scoring | Finds "cube" even if path differs |
| **GRPC Server** | Various unpacking and type issues | Safe float parsing + type guards | Graceful fallback to defaults |

---

## File Quick Links

```
patches/
├── patch_ee_pose_handler.py          (124 lines) - EE pose fix
├── patch_object_pose_handler.py      (164 lines) - Object pose fix
├── patch_grpc_server.py              (346 lines) - GRPC server fix
└── README: tools/geniesim_adapter/deployment/patches/

Source:
└── geniesim_server.py                (297 lines) - Local mock server

Docs:
├── genie_sim_analysis.md             (267 lines) - Full technical analysis
├── genie_sim_patches_flow.md         (415 lines) - Flow diagrams
├── genie_sim_code_reference.md       (291 lines) - Code snippets
└── genie_sim_quick_ref.md            (this file)
```

---

## EE Pose in 30 Seconds

**Problem:**
```python
pos, rot = self.robot.get_ee_pose()  # ❌ Fails if returns >2 values
```

**Solution:**
```python
# Auto-injected monkey-patch at robot init:
self._bp_wrap_ee_pose(self.robot)

# Robot.get_ee_pose always returns exactly (pos, rot) or (None, None)
```

**Also applied:**
```python
# Safe unpacking fallback in code:
_ee_result = self.robot.get_ee_pose(...)
if isinstance(_ee_result, (list, tuple)) and len(_ee_result) >= 2:
    pos, rot = _ee_result[0], _ee_result[1]
else:
    pos, rot = _ee_result, None
```

---

## Object Pose in 30 Seconds

**Problem:**
```python
# Requested path doesn't exist in stage
prim = stage.GetPrimAtPath("/obj/model/cube")  # ❌ None
```

**Solution:**
```python
# Auto-injected fuzzy resolution:
prim_path = self._bp_resolve_prim_path(prim_path)
# Now finds "/World/cube" by name matching + scoring

prim = stage.GetPrimAtPath(prim_path)  # ✓ Found!
```

**Scoring Logic:**
```
+100: Is Xformable (geometric prim)
+50:  Under /World/ (standard location)
-1:   Per "/" in path (prefer simpler hierarchies)

Winner: /World/cube (148 pts) > /Env/fixtures/cube (97 pts)
```

---

## GRPC Server Fixes in 30 Seconds

**Three Main Fixes:**

1. **Safe Float Helper**
   ```python
   _bp_safe_float("1.5m")  →  1.5  (parses units)
   _bp_safe_float(None)    →  0.0  (safe default)
   ```

2. **Wrap blocking_start_server Calls**
   ```python
   rsp.msg = str(blocking_start_server(...) or "")
   ```

3. **Guard Joint Position Access**
   ```python
   if not isinstance(joint_positions, dict):
       return rsp  # Early exit if wrong type
   ```

---

## Patch Application Order

```bash
# All idempotent - safe to re-run:
1. python3 patch_ee_pose_handler.py       # Check marker, apply if missing
2. python3 patch_object_pose_handler.py   # Check marker, apply if missing
3. python3 patch_grpc_server.py           # Check marker, apply if missing

# Result: All three markers now in code
# Re-running: All detect marker, skip silently
```

---

## Testing Checklist

```
[ ] EE Pose:
    - Robot returns 2 values  → pos, rot captured
    - Robot returns 3+ values → pos, rot captured, extra ignored
    - Robot raises exception  → Returns (None, None)

[ ] Object Pose:
    - Exact path exists       → Returned immediately
    - Name match exists       → Found via fuzzy search
    - No match               → Original path returned (logged warning)

[ ] GRPC Server:
    - All blocking_start_server wrapped with str()
    - joint_positions type-guarded
    - Float parsing handles "1.5m" format
```

---

## Logging Output to Watch For

```bash
# Successful EE Pose:
[PATCH] Wrapped robot.get_ee_pose for safe 2-value unpacking

# Successful Object Pose:
[PATCH] Resolved prim path (exact): /obj/model/cube -> /World/cube
[PATCH] Prim path exact candidates for 'cube': ['/World/cube', '/Env/fixtures/cube']

# Already Patched (idempotent):
[PATCH] EE pose handler already patched — skipping
[PATCH] Object pose handler already patched — skipping
[PATCH] grpc_server.py already patched — skipping

# Warnings (non-fatal):
[PATCH] WARNING: Could not resolve prim path /unknown/path in stage
[PATCH] Prim path resolution failed: [exception details]
```

---

## Integration Points

### In command_controller.py

```python
# After robot initialization:
self.robot = controller.init_robot(...)
self._bp_wrap_ee_pose(self.robot)  # ← Added by patch

# In handle_get_object_pose:
prim_path = request.prim_path
prim_path = self._bp_resolve_prim_path(prim_path)  # ← Added by patch
```

### In grpc_server.py

```python
# Top of file (after imports):
def _bp_safe_float(val, default=0.0):  # ← Added by patch
    # ... implementation

# Throughout file:
rsp.msg = str(blocking_start_server(...) or "")  # ← Wrapped by patch
if not isinstance(joint_positions, dict):  # ← Added by patch
    return rsp
```

---

## Key Insights

### Why Monkey-Patching for EE Pose?
- Works even if primary pattern not matched
- Catches runtime exceptions gracefully
- Can be applied after robot initialization
- Doesn't require code inspection

### Why Fuzzy Resolution for Object Pose?
- Different scenes have different USD hierarchies
- Asset names stay consistent (robustness)
- Scoring balances multiple criteria
- Substring fallback handles partial paths

### Why Defensive Unpacking?
- Never silently fail (all operations logged)
- Always provide sensible defaults
- Type-check before destructuring
- Graceful degradation

---

## Troubleshooting

### "Already patched" message on rebuild
**Expected behavior** - Markers prevent re-patching. This is correct.

### Prim path resolution returning original path
**Expected** - Path doesn't exist in stage, graceful fallback. Check logs for warnings.

### EE pose still returning >2 values
**Should not happen** - Monkey-patch wrapper applied after robot init. Check logs for wrapper message.

### GRPC server string fields still containing dicts
**Should not happen** - All blocking_start_server calls wrapped. Verify patches applied by checking for markers in grpc_server.py.

---

## Marker Strings (for grep)

```bash
# Check what's been patched:
grep "BlueprintPipeline ee_pose patch" command_controller.py
grep "BlueprintPipeline object_pose patch" command_controller.py
grep "BlueprintPipeline grpc_server patch" grpc_server.py

# Count occurrences (should be 1 per marker):
grep -c "BlueprintPipeline" command_controller.py  # Should be ≥2
grep -c "BlueprintPipeline" grpc_server.py  # Should be ≥1
```

---

## Docker Integration

```dockerfile
# In Dockerfile:
COPY patches/ /tmp/patches/
RUN python3 /tmp/patches/patch_ee_pose_handler.py
RUN python3 /tmp/patches/patch_object_pose_handler.py
RUN python3 /tmp/patches/patch_grpc_server.py

# All idempotent - safe in CI/CD pipelines
```

---

## Further Reading

| Document | Focus |
|----------|-------|
| **genie_sim_analysis.md** | Full technical deep-dive |
| **genie_sim_patches_flow.md** | Visual flow diagrams |
| **genie_sim_code_reference.md** | Code snippets & details |
| **genie_sim_quick_ref.md** | This file - quick start |

---

## At a Glance

```
┌─ EE POSE ──────────────────┐
│ Problem:  >2 return values │
│ Fix:      Monkey-patch     │
│ Result:   Always (pos,rot) │
└────────────────────────────┘

┌─ OBJECT POSE ──────────────┐
│ Problem:  Path not found   │
│ Fix:      Fuzzy resolution │
│ Result:   Finds by name    │
└────────────────────────────┘

┌─ GRPC SERVER ──────────────┐
│ Problem:  Type mismatches  │
│ Fix:      Safe unpacking   │
│ Result:   Graceful fallback│
└────────────────────────────┘

All patches: Idempotent + Logged + Defensive
```

