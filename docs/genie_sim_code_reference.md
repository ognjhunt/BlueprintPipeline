# GenieSim Patches: Code Reference Guide

Quick access to key code snippets from the patch files.

## File Locations

| File | Location | Lines | Purpose |
|------|----------|-------|---------|
| patch_ee_pose_handler.py | `/tools/geniesim_adapter/deployment/patches/` | 124 | Fix EE pose unpacking |
| patch_object_pose_handler.py | `/tools/geniesim_adapter/deployment/patches/` | 164 | Add fuzzy prim path resolution |
| patch_grpc_server.py | `/tools/geniesim_adapter/deployment/patches/` | 346 | Fix GRPC server bugs |
| geniesim_server.py | `/tools/geniesim_adapter/` | 297 | Mock local server |

---

## EE Pose Patch: Key Code Sections

### 1. Monkey-Patch Wrapper Injection

```python
@staticmethod
def _bp_wrap_ee_pose(robot):
    """Wrap robot.get_ee_pose to always return exactly (pos, rot)."""
    _orig_fn = getattr(robot, 'get_ee_pose', None)
    if _orig_fn is None or getattr(_orig_fn, '_bp_wrapped', False):
        return
    
    def _safe_get_ee_pose(*args, **kwargs):
        try:
            result = _orig_fn(*args, **kwargs)
        except Exception as _e:
            print(f'[PATCH] get_ee_pose call failed: {_e}')
            return None, None
        
        if isinstance(result, (list, tuple)):
            if len(result) >= 2:
                return result[0], result[1]
            elif len(result) == 1:
                return result[0], None
            else:
                return None, None
        return result, None
    
    _safe_get_ee_pose._bp_wrapped = True
    robot.get_ee_pose = _safe_get_ee_pose
    print('[PATCH] Wrapped robot.get_ee_pose for safe 2-value unpacking')
```

**Key Features:**
- `_bp_wrapped` flag prevents double-wrapping
- Exception handling catches runtime errors
- Normalizes all returns to 2-tuple
- Preserves original function reference

### 2. Safe Unpacking Pattern

```python
# BEFORE (Broken):
pos, rot = self.robot.get_ee_pose(...)

# AFTER (Safe):
_ee_result = self.robot.get_ee_pose(...)
if isinstance(_ee_result, (list, tuple)) and len(_ee_result) >= 2:
    pos, rot = _ee_result[0], _ee_result[1]
elif isinstance(_ee_result, (list, tuple)) and len(_ee_result) == 1:
    pos, rot = _ee_result[0], None
else:
    pos, rot = _ee_result, None
```

**Error Cases Handled:**
- Extra values: `(x, y, z, extra)` → `(x, y)`
- Single value: `(x,)` → `(x, None)`
- Non-sequence: `value` → `(value, None)`
- None: `None` → `(None, None)`

### 3. Auto-Wiring in Robot Init

```python
# In __init__ or robot initialization handler:
self.robot = controller.init_robot(...)
self._bp_wrap_ee_pose(self.robot)  # BlueprintPipeline ee_pose patch
```

---

## Object Pose Patch: Key Code Sections

### 1. Fuzzy Prim Path Resolver

```python
def _bp_resolve_prim_path(self, requested_path):
    """Resolve a prim path by fuzzy matching against the USD stage.

    If the exact path exists, return it.  Otherwise, search for prims
    whose name (last path component) matches the requested path's name,
    then score candidates: prefer Xformable prims under /World/ with
    shorter paths.  Falls back to substring matching if no exact name
    match is found.
    """
    try:
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return requested_path

        # Exact match (fastest path)
        prim = stage.GetPrimAtPath(requested_path)
        if prim and prim.IsValid():
            return requested_path

        # Extract target name (last path component)
        target_name = requested_path.rstrip("/").rsplit("/", 1)[-1]
        if not target_name:
            return requested_path

        target_lower = target_name.lower()

        # Collect candidates: exact name match and substring match
        exact_candidates = []
        substring_candidates = []

        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            prim_name = prim_path.rstrip("/").rsplit("/", 1)[-1]

            if prim_name == target_name:
                exact_candidates.append((prim_path, prim))
            elif target_lower in prim_name.lower():
                substring_candidates.append((prim_path, prim))

        # Score function - higher is better
        def _score(path, p):
            s = 0
            # Prefer geometric prims (Xformable includes Mesh, Xform, etc.)
            try:
                if p.IsA(UsdGeom.Xformable):
                    s += 100
            except Exception:
                pass
            # Prefer prims under /World/
            if path.startswith("/World/"):
                s += 50
            # Prefer shorter paths (less deeply nested)
            s -= path.count("/")
            return s

        # Try exact matches first, then substring
        for candidates, label in [(exact_candidates, "exact"), 
                                   (substring_candidates, "substring")]:
            if not candidates:
                continue
            scored = sorted(candidates, key=lambda c: _score(c[0], c[1]), 
                           reverse=True)
            best_path = scored[0][0]
            if len(scored) > 1:
                print(f"[PATCH] Prim path {label} candidates for '{target_name}': "
                      f"{[c[0] for c in scored[:5]]}")
            print(f"[PATCH] Resolved prim path ({label}): {requested_path} -> {best_path}")
            return best_path

        print(f"[PATCH] WARNING: Could not resolve prim path {requested_path} in stage")
    except Exception as e:
        print(f"[PATCH] Prim path resolution failed: {e}")

    return requested_path
```

### 2. Scoring Weights

```python
SCORING WEIGHTS:
├─ Xformable (geometric prim): +100
├─ Under /World/ (standard location): +50
└─ Path depth penalty: -1 per "/" character

EXAMPLE:
  /World/cube           = 100 + 50 - 2 = 148  ✓ Best
  /Env/fixtures/cube    = 100 + 0  - 3 = 97
  /Obstacles/cuboid     = 0   + 0  - 2 = -2   (substring only)
```

### 3. Integration Point

```python
# In handle_get_object_pose handler:
def handle_get_object_pose(self, request):
    prim_path = request.prim_path
    
    # PATCHED: Fuzzy resolution
    prim_path = self._bp_resolve_prim_path(prim_path)  
    
    # Now proceed with resolved path
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        position, rotation = prim.get_world_pose()
        # ... return pose
```

---

## GRPC Server Patch: Key Code Sections

### 1. Safe Float Helper

```python
def _bp_safe_float(val, default=0.0):
    """Safely convert value to float, handling unit-suffixed strings."""
    try:
        return float(val)
    except (ValueError, TypeError):
        if isinstance(val, str):
            import re as _re
            _m = _re.match(r'([+-]?\d*\.?\d+)', val.strip())
            if _m:
                return float(_m.group(1))
        return default

# USAGE:
#  _bp_safe_float("1.5m")          → 1.5
#  _bp_safe_float("0.25 meters")   → 0.25
#  _bp_safe_float({"x": 1})        → 0.0 (default)
#  _bp_safe_float(None, 5.0)       → 5.0 (custom default)
```

### 2. Safe Object Pose Unpacking

```python
# BEFORE (Broken):
position, rotation = object_pose

# AFTER (Safe):
_op = object_pose if isinstance(object_pose, (list, tuple)) else (object_pose, None)
position = _op[0] if len(_op) >= 1 else (0, 0, 0)
rotation = _op[1] if len(_op) >= 2 else (1, 0, 0, 0)

# Position assignment (safe):
_pos = list(position) if hasattr(position, '__iter__') else [0, 0, 0]
rsp.object_pose.position.x = _bp_safe_float(_pos[0]) if len(_pos) > 0 else 0.0
rsp.object_pose.position.y = _bp_safe_float(_pos[1]) if len(_pos) > 1 else 0.0
rsp.object_pose.position.z = _bp_safe_float(_pos[2]) if len(_pos) > 2 else 0.0

# Rotation assignment (safe):
_rot = list(rotation) if hasattr(rotation, '__iter__') and rotation is not None else [1, 0, 0, 0]
rsp.object_pose.rpy.rw = _bp_safe_float(_rot[0], 1.0) if len(_rot) > 0 else 1.0
rsp.object_pose.rpy.rx = _bp_safe_float(_rot[1]) if len(_rot) > 1 else 0.0
rsp.object_pose.rpy.ry = _bp_safe_float(_rot[2]) if len(_rot) > 2 else 0.0
rsp.object_pose.rpy.rz = _bp_safe_float(_rot[3]) if len(_rot) > 3 else 0.0
```

### 3. String Wrapping for blocking_start_server

```python
# BEFORE (Broken):
# blocking_start_server() may return dict but rsp.msg expects string
rsp.msg = self.server_function.blocking_start_server(command, args)

# AFTER (Safe):
# Wrap ALL blocking_start_server calls with str()
rsp.msg = str(self.server_function.blocking_start_server(command, args) or "")
```

### 4. Joint Position Guards

```python
# BEFORE (Broken):
for joint_name in joint_positions:
    joint_state.position = joint_positions[joint_name]  # May be dict!

# AFTER (Safe):
if not isinstance(joint_positions, dict):
    print(f'[PATCH] get_joint_position got non-dict: {type(joint_positions)}')
    return rsp

for joint_name in joint_positions:
    _jval = joint_positions[joint_name]
    # Handle dict-as-value case
    joint_state.position = float(np.asarray(_jval).flat[0]) \
        if not isinstance(_jval, dict) else 0.0
```

---

## Patch Markers for Idempotency

### Check Pattern (All patches)

```python
PATCH_MARKER = "BlueprintPipeline [TYPE] patch"

if PATCH_MARKER in file_content:
    print(f"[PATCH] Already patched — skipping")
    sys.exit(0)
```

### Available Markers

```python
PATCH_MARKER_EE_POSE       = "BlueprintPipeline ee_pose patch"
PATCH_MARKER_OBJECT_POSE   = "BlueprintPipeline object_pose patch"
PATCH_MARKER_GRPC_SERVER   = "BlueprintPipeline grpc_server patch"
```

### Auto-Detection in Docker

```dockerfile
# Run during build (idempotent):
RUN python3 /tmp/patches/patch_ee_pose_handler.py
RUN python3 /tmp/patches/patch_object_pose_handler.py
RUN python3 /tmp/patches/patch_grpc_server.py

# If already patched: [PATCH] Already patched — skipping
# If not patched: [PATCH] Successfully patched ...
# If source missing: [PATCH] Skipping (server source not available)
```

---

## Testing Patterns

### EE Pose Testing

```python
# Test various return formats:
test_cases = [
    ((0.5, 0.5, 0.5), (1, 0, 0, 0), "2-tuple"),           # Expected
    ((0.5, 0.5, 0.5), (1, 0, 0, 0), "extra", None, "5-tuple"),  # Extra
    ((0.5, 0.5, 0.5), "single"),  # Single element
    (None, None, "None return"),
    (np.array([0.5, 0.5, 0.5]), np.array([1, 0, 0, 0]), "numpy"),
]

# All should safely return (pos, rot) or (None, None)
```

### Object Pose Testing

```python
# Test path resolution:
test_requests = [
    ("/World/cube", True),           # Exact match exists
    ("/obj/model/cube", False),      # Fuzzy match needed
    ("/Env/fixtures/unknown", None), # No match, returns original
    ("/", "invalid"),                # Edge cases
]

# Verify scoring:
# Xformable + /World + low depth > non-xformable/deep paths
```

### GRPC Server Testing

```python
# Test safe float parsing:
test_values = [
    ("1.5", 1.5),
    ("0.25m", 0.25),
    ("1.0e-3", 0.001),
    ({"x": 1}, 0.0),
    (None, 0.0),
]

# Verify all blocking_start_server calls wrapped with str()
# Verify joint_positions type-guarded
```

---

## Debugging Tips

### Enable Patch Logging

All patches print `[PATCH]` prefixed messages:

```bash
# Check logs for patch output:
docker logs geniesim-server | grep "\[PATCH\]"

# Expected output (successful):
[PATCH] Wrapped robot.get_ee_pose for safe 2-value unpacking
[PATCH] Resolved prim path (exact): /obj/model/cube -> /World/cube
[PATCH] Fixed ee_pose unpacking: pos, rot = ...get_ee_pose(...)
```

### Verification Checklist

- [ ] All patch scripts executed during Docker build
- [ ] Markers present in code after build
- [ ] No "[PATCH]" warnings about unmatched patterns
- [ ] Robot.get_ee_pose wrapped successfully
- [ ] Prim path resolution working (check logs)
- [ ] GRPC calls wrapping string fields

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| "Already patched" skip | Markers detected in source | Normal - idempotent behavior |
| "Could not find pattern" | Upstream code structure differs | Helper methods injected anyway |
| Path resolution failing | USD stage not initialized | Graceful fallback to original path |
| Wrapped twice | Docker re-run without cache | Add `--no-cache` flag |

---

## Integration Checklist

- [x] patch_ee_pose_handler.py creates safe unpacking
- [x] patch_ee_pose_handler.py injects monkey-patch wrapper
- [x] patch_object_pose_handler.py adds _bp_resolve_prim_path helper
- [x] patch_grpc_server.py wraps blocking_start_server calls
- [x] All patches are idempotent (marker-based)
- [x] All patches include error logging
- [x] Mock local server (geniesim_server.py) provides test fallback

