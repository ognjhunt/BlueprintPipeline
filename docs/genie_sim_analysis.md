# GenieSim Server EE Pose and Object Pose Code Paths Analysis

## Summary
This analysis documents the EE pose and object pose handling in the GenieSim server, including upstream bugs and the BlueprintPipeline patches that fix them.

## Key Files

### Local BlueprintPipeline Files
1. `/Users/nijelhunt_1/workspace/BlueprintPipeline/tools/geniesim_adapter/deployment/patches/patch_ee_pose_handler.py`
2. `/Users/nijelhunt_1/workspace/BlueprintPipeline/tools/geniesim_adapter/deployment/patches/patch_object_pose_handler.py`
3. `/Users/nijelhunt_1/workspace/BlueprintPipeline/tools/geniesim_adapter/deployment/patches/patch_grpc_server.py`
4. `/Users/nijelhunt_1/workspace/BlueprintPipeline/tools/geniesim_adapter/geniesim_server.py`

### Upstream Server Files (in Docker container)
- `/opt/geniesim/source/data_collection/server/command_controller.py`
- `/opt/geniesim/source/data_collection/server/grpc_server.py`

---

## EE POSE CODE PATH

### Upstream Implementation Issue
The upstream `command_controller.py` has a critical bug in the EE pose handler:

```python
# BROKEN - Too many values to unpack (expected 2)
pos, rot = self.robot.get_ee_pose(...)
```

**Problem**: The robot controller's `get_ee_pose()` method can return:
- 2 values: `(position, orientation)` 
- 3+ values: `(position, orientation, extra_data)`
- Inconsistent tuple/list formats across different robot implementations

### BlueprintPipeline Patch: `patch_ee_pose_handler.py`

The patch addresses this through two mechanisms:

#### 1. **Safe Unpacking Pattern**
Replaces the faulty unpacking with a robust version:

```python
# PATCHED
_ee_result = self.robot.get_ee_pose(...)
if isinstance(_ee_result, (list, tuple)) and len(_ee_result) >= 2:
    pos, rot = _ee_result[0], _ee_result[1]
elif isinstance(_ee_result, (list, tuple)) and len(_ee_result) == 1:
    pos, rot = _ee_result[0], None
else:
    pos, rot = _ee_result, None
```

**Benefits**:
- Handles variable-length returns safely
- Gracefully degrades with None values
- Preserves first 2 values regardless of extra data
- Extra variables assigned to None

#### 2. **Monkey-Patch Wrapper**
Injects a class method to wrap the robot's `get_ee_pose`:

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

**Auto-wiring**:
The patch automatically injects a call after robot initialization:
```python
self.robot = ...
self._bp_wrap_ee_pose(self.robot)  # BlueprintPipeline ee_pose patch
```

---

## OBJECT POSE CODE PATH

### Upstream Implementation Issue
The upstream `command_controller.py` has issues with object pose retrieval:

1. **Path Mismatch**: Requested prim paths don't match the actual USD stage hierarchy
2. **Missing Prim Resolution**: No fuzzy matching when exact paths don't exist
3. **Return Format Issues**: May return empty/identity poses

### BlueprintPipeline Patch: `patch_object_pose_handler.py`

The patch adds a **fuzzy path resolution helper method**:

```python
def _bp_resolve_prim_path(self, requested_path):
    """Resolve a prim path by fuzzy matching against the USD stage.
    
    If the exact path exists, return it. Otherwise, search for prims
    whose name (last path component) matches the requested path's name,
    then score candidates: prefer Xformable prims under /World/ with
    shorter paths. Falls back to substring matching if no exact name
    match is found.
    """
```

#### Resolution Strategy (Priority Order)

1. **Exact Path Match**
   - Check if the requested path exists directly in the stage
   - Return immediately if found

2. **Exact Name Match**
   - Extract target name: last path component of requested path (e.g., "cube" from "/path/to/cube")
   - Search all prims in stage for exact name match
   - Score candidates with preferences:
     - **+100 points**: Geometric prims (Xformable - includes Mesh, Xform, etc.)
     - **+50 points**: Prims under `/World/` (main content location)
     - **-depth**: Penalize deeply nested paths (fewer slashes = better)
   - Return highest-scoring match

3. **Substring Match**
   - If no exact name match, search for substring matches
   - Apply same scoring system
   - Return highest-scoring match

4. **Fallback**
   - If no matches found, return original path
   - Print warning in logs

#### Example Resolution Flow

```
Requested: "/obj/model/cube"
Target name: "cube"

Found candidates:
  Exact: ["/World/cube", "/Env/fixtures/cube"]
  Substring: ["/World/assembly/cuboid", "/Obstacles/cubic_stand"]

Scoring:
  "/World/cube": +100 (Xformable) + 50 (in /World) - 2 (depth) = 148 ✓ WINNER
  "/Env/fixtures/cube": +100 (Xformable) - 3 (depth) = 97
  
Resolution: "/obj/model/cube" → "/World/cube"
```

#### Auto-injection
The patch automatically injects a call in the `handle_get_object_pose` handler:
```python
prim_path = request.prim_path
prim_path = self._bp_resolve_prim_path(prim_path)  # BlueprintPipeline object_pose patch
```

---

## GRPC SERVER PATCHES

The `patch_grpc_server.py` file fixes additional issues in the upstream `grpc_server.py`:

### Key Fixes

1. **Object Pose Unpacking**
   - Safely unpacks `position, rotation = object_pose`
   - Handles non-tuple/list returns
   - Safe extraction: `_pos = list(position) if hasattr(position, '__iter__') else [0, 0, 0]`

2. **Protobuf String Fields**
   - Wraps all `rsp.msg = blocking_start_server(...)` calls with `str()`
   - Reason: `blocking_start_server()` often returns dicts but `rsp.msg` expects string

3. **Recording State Assignment**
   - Fixes: `rsp.recordingState = result` → `rsp.recordingState = str(result) or ""`
   - Handles dict/None returns gracefully

4. **Joint Position Extraction**
   - Guards against string returns (error messages)
   - Safe float extraction: `float(np.asarray(_jval).flat[0])`
   - Handles dict-as-value cases

5. **Safe Float Helper**
   - Injected `_bp_safe_float()` function
   - Parses unit-suffixed values (e.g., "1.5m" → 1.5)
   - Falls back to 0.0 for unparseable values

---

## DEPLOYMENT FLOW

### Docker Build Process
1. Dockerfile copies patch scripts to `/tmp/patches/`
2. During image build, patches are applied to source code:
   ```bash
   python3 /tmp/patches/patch_ee_pose_handler.py
   python3 /tmp/patches/patch_object_pose_handler.py
   python3 /tmp/patches/patch_grpc_server.py
   ```

### Runtime Verification
Patches are **idempotent** — they include a marker string to skip if already applied:
- `PATCH_MARKER = "BlueprintPipeline ee_pose patch"`
- `PATCH_MARKER = "BlueprintPipeline object_pose patch"`
- `PATCH_MARKER = "BlueprintPipeline grpc_server patch"`

This prevents double-patching and allows safe re-runs.

---

## KEY INSIGHTS

### EE Pose Robustness
- **Monkey-patch wrapper approach** is powerful because:
  1. Works even if primary unpacking pattern isn't matched
  2. Catches runtime exceptions gracefully
  3. Standardizes all robot implementations to 2-value return
  4. Can be applied at any point, even after robot initialization

### Object Pose Flexibility
- **Fuzzy path resolution** is critical because:
  1. Different scenes have different USD hierarchies
  2. Asset names stay consistent even if paths differ
  3. Scoring system balances multiple criteria:
     - Geometric relevance (Xformable)
     - Location (under /World/ is standard)
     - Nesting depth (simpler hierarchies preferred)
  4. Substring fallback handles partial path mismatches

### Data Validation Philosophy
- Both patches add **defensive unpacking**:
  - Check return types before destructuring
  - Provide sensible defaults (None, (0,0,0), etc.)
  - Log all decisions for debugging
  - Never silently fail — print warnings to console

### Idempotency & Safety
- Marker-based detection prevents re-patching
- Wrapped/flagged functions prevent double-wrapping
- Original functions preserved for debugging
- All patches include logging for transparency

---

## Files Referenced

- **EE Pose Patch**: `/Users/nijelhunt_1/workspace/BlueprintPipeline/tools/geniesim_adapter/deployment/patches/patch_ee_pose_handler.py`
- **Object Pose Patch**: `/Users/nijelhunt_1/workspace/BlueprintPipeline/tools/geniesim_adapter/deployment/patches/patch_object_pose_handler.py`
- **GRPC Server Patch**: `/Users/nijelhunt_1/workspace/BlueprintPipeline/tools/geniesim_adapter/deployment/patches/patch_grpc_server.py`
- **Local Mock Server**: `/Users/nijelhunt_1/workspace/BlueprintPipeline/tools/geniesim_adapter/geniesim_server.py` (297 lines)

