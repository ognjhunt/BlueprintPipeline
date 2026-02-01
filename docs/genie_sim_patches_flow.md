# GenieSim Patches: Code Flow Diagrams

## EE Pose Handler Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Client Request: GET_EE_POSE                                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ command_controller.py: handle_get_ee_pose()                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ UPSTREAM (BROKEN):     │
        │ pos, rot =             │
        │  robot.get_ee_pose()   │
        │                        │
        │ ❌ Crashes if >2 values│
        └────────────────────────┘
                     │
                     ▼
     ┌──────────────────────────────┐
     │ BLUEPRINT PATCH Applied:     │
     │ _ee_result = get_ee_pose()   │
     │ if isinstance(...) and       │
     │    len(...) >= 2:            │
     │     pos, rot = result[0:2]   │
     │ elif len(...) == 1:          │
     │     pos, rot = result, None  │
     │ else:                        │
     │     pos, rot = None, None    │
     │                              │
     │ ✓ Handles variable returns  │
     └──────────────────────────────┘
                     │
                     ▼
     ┌──────────────────────────────┐
     │ ALSO: Monkey-Patch Wrapper   │
     │ Applied at robot init:       │
     │ self._bp_wrap_ee_pose(robot) │
     │                              │
     │ ✓ Wraps original method      │
     │ ✓ Always returns 2 values    │
     │ ✓ Catches exceptions         │
     └──────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ Safe Response:         │
        │ pos: [x, y, z]        │
        │ rot: [qx, qy, qz, qw] │
        │ (or None if failed)    │
        └────────────────────────┘
```

## Object Pose Handler Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Client Request: GET_OBJECT_POSE                                 │
│ Requested prim_path: "/obj/model/cube"                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ command_controller.py: handle_get_object_pose()                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────┐
    │ UPSTREAM (BROKEN):             │
    │ stage.GetPrimAtPath(prim_path) │
    │                                │
    │ ❌ Exact path may not exist    │
    │ ❌ Returns empty/identity pose │
    └────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────────┐
    │ BLUEPRINT PATCH: _bp_resolve_prim_path()           │
    │                                                    │
    │ Step 1: Try exact path match                       │
    │         → stage.GetPrimAtPath("/obj/model/cube")  │
    │         → Not found                               │
    │                                                    │
    │ Step 2: Extract target name                        │
    │         → target_name = "cube" (last component)   │
    │                                                    │
    │ Step 3: Search stage for exact name matches        │
    │         Found:                                     │
    │         - "/World/cube" [Xformable]               │
    │         - "/Env/fixtures/cube" [Xformable]        │
    │                                                    │
    │ Step 4: Score candidates                          │
    │         "/World/cube":                            │
    │         + 100 (Xformable)                         │
    │         +  50 (under /World)                      │
    │         -   2 (depth) = 148 ✓ WINNER             │
    │                                                    │
    │         "/Env/fixtures/cube":                     │
    │         + 100 (Xformable)                         │
    │         -   3 (depth) = 97                        │
    │                                                    │
    │ Step 5: Return best match                         │
    │         "/World/cube"                             │
    └────────────────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ USD Stage GetPrimAtPath    │
        │ prim = stage.GetPrimAtPath │
        │        ("/World/cube")     │
        │                            │
        │ ✓ Now found successfully   │
        └────────────────────────────┘
                     │
                     ▼
     ┌──────────────────────────────┐
     │ Extract Pose from Prim:      │
     │ position, rotation =         │
     │   prim.get_world_pose()      │
     │                              │
     │ + GRPC Server Patch also     │
     │   applies safe unpacking here│
     └──────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ Return Response:       │
        │ pos: [x, y, z]        │
        │ rot: [qx, qy, qz, qw] │
        └────────────────────────┘
```

## Patch Application Timeline

```
┌──────────────────────────────────────────────────────────────┐
│ Docker Build Phase                                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ 1. Base Image: geniesim:latest (Ubuntu + Isaac Sim)          │
│    └─ Contains upstream code with bugs                       │
│                                                              │
│ 2. Copy patch scripts                                         │
│    └─ COPY patches/ /tmp/patches/                            │
│                                                              │
│ 3. Apply patch_ee_pose_handler.py                            │
│    └─ Python3 /tmp/patches/patch_ee_pose_handler.py          │
│       ├─ Check marker: "BlueprintPipeline ee_pose patch"     │
│       ├─ If not found: Apply regex patterns                  │
│       ├─ Fix unpacking pattern                               │
│       ├─ Inject monkey-patch wrapper method                  │
│       └─ Inject wrapper call in robot init                   │
│                                                              │
│ 4. Apply patch_object_pose_handler.py                        │
│    └─ Python3 /tmp/patches/patch_object_pose_handler.py      │
│       ├─ Check marker: "BlueprintPipeline object_pose patch" │
│       ├─ If not found: Inject helper method                  │
│       └─ Inject prim path resolution call                    │
│                                                              │
│ 5. Apply patch_grpc_server.py                                │
│    └─ Python3 /tmp/patches/patch_grpc_server.py              │
│       ├─ Check marker: "BlueprintPipeline grpc_server patch" │
│       ├─ Fix object_pose unpacking                           │
│       ├─ Wrap blocking_start_server() calls                  │
│       ├─ Guard joint position access                         │
│       └─ Inject safe_float() helper                          │
│                                                              │
│ 6. Result: blueprint-geniesim-server:latest                  │
│    └─ All patches applied, idempotent markers in place       │
│                                                              │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ Runtime Phase                                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ 1. Container starts with patched code                        │
│                                                              │
│ 2. Robot initialization:                                     │
│    └─ self.robot = controller.init_robot()                   │
│    └─ self._bp_wrap_ee_pose(self.robot)  # PATCHED          │
│       └─ Robot.get_ee_pose is now wrapped                    │
│                                                              │
│ 3. EE Pose requests:                                         │
│    └─ handle_get_ee_pose() calls robot.get_ee_pose()         │
│    └─ Wrapped version returns exactly (pos, rot)             │
│    └─ Safe unpacking also handles variable-length returns    │
│                                                              │
│ 4. Object Pose requests:                                     │
│    └─ handle_get_object_pose() gets prim_path               │
│    └─ Calls self._bp_resolve_prim_path(prim_path)           │
│    └─ Fuzzy matching finds actual prim in stage             │
│    └─ Returns correct pose                                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Patch Marker Detection (Idempotency)

```python
# In each patch script:

if PATCH_MARKER in file_content:
    print("[PATCH] Already patched — skipping")
    sys.exit(0)

# After patching, marker is added:
content = f"# {PATCH_MARKER} applied\n" + content

# This ensures:
# ✓ Re-running patch scripts is safe (no-op)
# ✓ Multiple Docker builds don't double-patch
# ✓ Can safely include in idempotent infrastructure code
```

## Error Handling Philosophy

```
┌────────────────────────────────────────┐
│ Upstream: Silent Failure               │
│ pos, rot = (complicated_return)        │
│ ❌ ValueError if >2 values              │
│ ❌ TypeError if not iterable            │
│ ❌ No logging                           │
└────────────────────────────────────────┘
                    ▼
┌────────────────────────────────────────┐
│ Blueprint Patch: Defensive Strategy    │
│                                        │
│ 1. Check types before unpacking        │
│    if isinstance(result, (list,tuple)) │
│                                        │
│ 2. Provide sensible defaults           │
│    else: pos, rot = None, None         │
│                                        │
│ 3. Always log decisions                │
│    print(f'[PATCH] Resolved...')       │
│                                        │
│ 4. Return usable fallback              │
│    return original_path                │
│                                        │
│ ✓ Never crashes                        │
│ ✓ Always logs errors                   │
│ ✓ Graceful degradation                 │
└────────────────────────────────────────┘
```

## Path Resolution Scoring Example

```
Scene Structure:
  /World/
    ├─ objects/
    │  └─ cube_001 (Mesh, under /World)
    ├─ cube (Xform, under /World)
    └─ fixture_cube (Xformable)
  /Env/
    └─ fixtures/
       └─ cube (Mesh, nested)
  /Obstacles/
    └─ cubic_obstacle (no match)

Request: "/obj/model/cube"
Target: "cube"

Scoring Results:
┌────────────────────────────────────┐
│ Candidate         │ Score │ Reason │
├────────────────────────────────────┤
│ /World/cube       │  148  │ +100 Xformable  │
│                   │       │ +50 /World      │
│                   │       │ -2 depth        │
│                   │       │ ✓ WINNER        │
├────────────────────────────────────┤
│ /World/obj/cube   │  147  │ +100 Xformable  │
│                   │       │ +50 /World      │
│                   │       │ -3 depth        │
├────────────────────────────────────┤
│ /Env/fixtures/... │   95  │ +100 Xformable  │
│                   │       │ -5 depth        │
├────────────────────────────────────┤
│ /Obstacles/cubic  │   N/A │ Substring only  │
│   _obstacle       │       │ Not name match  │
└────────────────────────────────────┘
```

