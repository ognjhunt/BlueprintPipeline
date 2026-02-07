# VM & Container Operations Guide

Quick reference for getting the BlueprintPipeline running on the GCP VM with Genie Sim.

## VM Details

| Field | Value |
|-------|-------|
| Instance | `isaac-sim-ubuntu` |
| Zone | `us-east1-b` |
| Machine | `g2-standard-32` (NVIDIA L4 24GB VRAM, 128GB RAM) |
| Docker | Requires `sudo` |
| User home | `/home/nijelhunt1` |
| Repo path | `~/BlueprintPipeline` |

## Quick Start

### If the VM is already running

SSH in and run the pipeline directly:

```bash
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b

# On the VM:
cd ~/BlueprintPipeline
bash scripts/vm-start.sh   # checks container is up, polls gRPC until ready

export PYTHONPATH=~/BlueprintPipeline:~/BlueprintPipeline/episode-generation-job:$PYTHONPATH
export SKIP_QUALITY_GATES=1
export GENIESIM_FIRST_CALL_TIMEOUT_S=300
export CAMERA_REWARMUP_ON_RESET=1
python3 tools/run_local_pipeline.py \
  --scene-dir ./test_scenes/scenes/lightwheel_kitchen \
  --steps simready,usd,replicator,isaac-lab,genie-sim-export,genie-sim-submit,genie-sim-import \
  --use-geniesim --fail-fast
```

### Cold start (VM was stopped)

```bash
# 1. Start VM
gcloud compute instances start isaac-sim-ubuntu --zone=us-east1-b

# 2. Wait ~30s for SSH, then connect
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b

# 3. The container auto-starts with the VM (restart: unless-stopped).
#    Just wait for the gRPC server to become ready:
#    (If you ever run the container manually, pass `-e SIM_ASSETS=/sim-assets`.)
cd ~/BlueprintPipeline
bash scripts/vm-start.sh

# 4. Run pipeline (same exports as above)
export PYTHONPATH=~/BlueprintPipeline:~/BlueprintPipeline/episode-generation-job:$PYTHONPATH
export SKIP_QUALITY_GATES=1
export GENIESIM_FIRST_CALL_TIMEOUT_S=300
export CAMERA_REWARMUP_ON_RESET=1
python3 tools/run_local_pipeline.py \
  --scene-dir ./test_scenes/scenes/lightwheel_kitchen \
  --steps simready,usd,replicator,isaac-lab,genie-sim-export,genie-sim-submit,genie-sim-import \
  --use-geniesim --fail-fast
```

### Strict production run (RGB disabled only)

Use this when you want strict realism checks for everything except RGB color validity:

```bash
cd ~/BlueprintPipeline
bash scripts/vm-start-xorg.sh
sudo docker restart geniesim-server >/dev/null
bash scripts/vm-start.sh

# PYTHONPATH must include both repo root AND episode-generation-job (for data_fidelity module)
export PYTHONPATH=~/BlueprintPipeline:~/BlueprintPipeline/episode-generation-job:$PYTHONPATH
export GENIESIM_HOST=localhost
export GENIESIM_PORT=50051
export GENIESIM_SKIP_DEFAULT_LIGHTING=1
export SKIP_RGB_CAPTURE=true
export REQUIRE_VALID_RGB=false
export REQUIRE_CAMERA_DATA=false
export DATA_FIDELITY_MODE=production
export STRICT_REALISM=true
export REQUIRE_OBJECT_MOTION=true
export REQUIRE_REAL_CONTACTS=true
export REQUIRE_REAL_EFFORTS=true
export REQUIRE_INTRINSICS=true
export GENIESIM_FIRST_CALL_TIMEOUT_S=600
export CAMERA_REWARMUP_ON_RESET=1
unset SKIP_QUALITY_GATES 2>/dev/null || true

python3 tools/run_local_pipeline.py \
  --scene-dir ./test_scenes/scenes/lightwheel_kitchen \
  --steps genie-sim-submit \
  --force-rerun genie-sim-submit \
  --use-geniesim --fail-fast
```

**Results from Feb 7 2026 strict run**: 6/7 tasks completed successfully with `task_success=True`. Quality scores 0.69-0.84. Task 5 (`variation_toaster`) crashed the server repeatedly and was skipped. Object motion validated via EE-manipulation-displacement proxy (0.33-0.34m per task). Total runtime ~45 min for 7 tasks (including retries on task 5).

### If a run appears stuck at init

Symptoms:
- Pipeline process is alive, but `/tmp/pipeline_strict.log` timestamp stops advancing.
- Last line is often `Running init sequence (initial)`.

Action:
```bash
pkill -9 -f 'tools/run_local_pipeline.py' 2>/dev/null || true
pkill -9 -f 'pipeline_guard.sh' 2>/dev/null || true

cd ~/BlueprintPipeline
bash scripts/vm-start-xorg.sh
sudo docker restart geniesim-server >/dev/null
bash scripts/vm-start.sh
```

This is usually a stale server/init deadlock, not CPU/GPU saturation. Confirm before resizing:
```bash
uptime
free -h
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
```

If load/utilization are low and logs are frozen, restart the container and rerun instead of scaling first.

### Syncing code changes from local machine

**Recommended: Sync entire directories** (the individual file list below is incomplete and causes `ModuleNotFoundError`):

```bash
# Sync all tools/ (includes geniesim_adapter, quality_gates, camera_io, etc.)
gcloud compute scp --recurse --compress --zone=us-east1-b \
  tools/ "isaac-sim-ubuntu:~/BlueprintPipeline/tools/"

# Sync supporting directories
for dir in policy_configs scripts episode-generation-job; do
  gcloud compute scp --recurse --compress --zone=us-east1-b \
    "$dir/" "isaac-sim-ubuntu:~/BlueprintPipeline/$dir/"
done

# Sync individual root files
for f in \
  Dockerfile.geniesim-server \
  docker-compose.geniesim-server.yaml \
  genie-sim-local-job/requirements.txt \
  .env; do
  gcloud compute scp --zone=us-east1-b \
    "$f" "isaac-sim-ubuntu:~/BlueprintPipeline/$f" 2>/dev/null
done
```

**Why not individual files?** The pipeline imports from `tools.quality_gates.physics_certification`, `tools.camera_io`, and other modules not in the old individual file list. Syncing entire `tools/` is faster and avoids `ModuleNotFoundError` at runtime.

**Alternative: Sync only changed files** (use when `tools/` is already up to date):

```bash
for f in \
  tools/run_local_pipeline.py \
  tools/geniesim_adapter/local_framework.py \
  tools/geniesim_adapter/asset_index.py \
  tools/geniesim_adapter/scene_graph.py \
  tools/geniesim_adapter/deployment/patches/patch_camera_handler.py \
  tools/geniesim_adapter/deployment/patches/patch_data_collector_render_config.py \
  tools/geniesim_adapter/deployment/patches/patch_ee_pose_handler.py \
  tools/geniesim_adapter/deployment/patches/patch_object_pose_handler.py \
  tools/geniesim_adapter/deployment/patches/patch_omnigraph_dedup.py \
  tools/geniesim_adapter/deployment/patches/patch_grpc_server.py \
  tools/geniesim_adapter/deployment/patches/patch_stage_diagnostics.py \
  tools/geniesim_adapter/deployment/patches/patch_observation_cameras.py \
  tools/geniesim_adapter/deployment/patches/_apply_safe_float.py \
  tools/geniesim_adapter/deployment/patches/patch_xforms_safe_rotation.py \
  tools/geniesim_adapter/deployment/patches/patch_articulation_guard.py \
  tools/geniesim_adapter/deployment/patches/patch_autoplay.py \
  tools/geniesim_adapter/deployment/bootstrap_geniesim_runtime.sh \
  tools/geniesim_adapter/deployment/start_geniesim_server.sh \
  tools/geniesim_adapter/robot_configs/franka_panda.json \
  tools/geniesim_adapter/robot_configs/ur10.json \
  scripts/vm-start.sh \
  scripts/vm-start-xorg.sh \
  Dockerfile.geniesim-server \
  docker-compose.geniesim-server.yaml \
  genie-sim-local-job/requirements.txt \
  .env; do
  gcloud compute scp --zone=us-east1-b \
    "$f" "isaac-sim-ubuntu:~/BlueprintPipeline/$f" 2>/dev/null
done
```

### Rebuilding the Docker image (one-time, after Dockerfile changes)

```bash
# On the VM — takes 30-45 min (cuRobo compilation from source)
cd ~/BlueprintPipeline
sudo docker compose -f docker-compose.geniesim-server.yaml build
sudo docker compose -f docker-compose.geniesim-server.yaml up -d
```

## Required Display Path For Camera RGB

Camera RGB capture is only reliable when Isaac Sim has a valid X display surface.
In this deployment, that means host Xorg on `:99` plus container `DISPLAY=:99`
and `/tmp/.X11-unix` mounted into the container.

**Xorg config note (driver 590+)**: The `UseDisplayDevice "None"` option in `xorg.conf` is incompatible with NVIDIA driver 590+. If Xorg fails with "no screens found" after a driver upgrade, remove `UseDisplayDevice "None"` from the `Device` section and use `AllowEmptyInitialConfiguration "true"` instead. The `vm-start-xorg.sh` script handles this automatically.

### Recommended startup sequence

```bash
cd ~/BlueprintPipeline

# 1) Ensure host Xorg is running on :99
bash scripts/vm-start-xorg.sh

# 2) Start/check container + health
bash scripts/vm-start.sh

# 3) Verify latest Kit startup args do NOT include --no-window
sudo docker exec geniesim-server bash -c \
  'grep "Starting kit application" /tmp/geniesim_server.log | tail -1'
```

Acceptance checks:
- latest startup args do not include `--no-window` when cameras are enabled
- startup args include `--reset-user` and `--/renderer/activeGpu=0`
- camera logs show non-zero RGB channels and pass quality checks

## Common Operations

### Check container status
```bash
sudo docker ps -a
```

### Check gRPC port (50051 = 0xC383)
```bash
sudo docker exec geniesim-server bash -c 'cat /proc/net/tcp6 | grep C383'
```

### View server logs
```bash
sudo docker logs geniesim-server --tail 50
```

### Apply patches at runtime (no rebuild)
```bash
for p in patch_omnigraph_dedup patch_data_collector_render_config patch_camera_handler patch_object_pose_handler \
         patch_ee_pose_handler patch_stage_diagnostics patch_observation_cameras \
         patch_grpc_server _apply_safe_float patch_xforms_safe_rotation \
         patch_articulation_guard patch_autoplay; do
  sudo docker cp ~/BlueprintPipeline/tools/geniesim_adapter/deployment/patches/${p}.py geniesim-server:/tmp/
  sudo docker exec geniesim-server /isaac-sim/python.sh /tmp/${p}.py
done
sudo docker restart geniesim-server
```

### Fix grpc_server.py set literal bugs (runtime)
```bash
sudo docker exec geniesim-server sed -i 's/data={"reset", Reset}/data={"reset": Reset}/' /opt/geniesim/source/data_collection/server/grpc_server.py
sudo docker exec geniesim-server sed -i 's/data={"detach", detach}/data={"detach": detach}/' /opt/geniesim/source/data_collection/server/grpc_server.py
sudo docker restart geniesim-server
```

### Full Docker rebuild (bakes all patches permanently)
```bash
# On the VM (takes 30-45 min due to cuRobo compilation):
cd ~/BlueprintPipeline
sudo docker compose -f docker-compose.geniesim-server.yaml build
sudo docker compose -f docker-compose.geniesim-server.yaml up -d
```

### Run pipeline in background (for long runs)
```bash
cd ~/BlueprintPipeline
export PYTHONPATH=~/BlueprintPipeline:~/BlueprintPipeline/episode-generation-job:$PYTHONPATH
export SKIP_QUALITY_GATES=1
nohup python3 tools/run_local_pipeline.py \
  --scene-dir ./test_scenes/scenes/lightwheel_kitchen \
  --steps genie-sim-export,genie-sim-submit,genie-sim-import \
  --use-geniesim --fail-fast \
  > /tmp/pipeline_run.log 2>&1 &
echo PID: $!
# Monitor: tail -f /tmp/pipeline_run.log
```
For long runs, prefer `nohup` or a `tmux` session and avoid broad `pkill -9 python3`-style commands. Under heavy GPU/CPU load, SSH sessions can become unstable, so keeping the pipeline detached prevents drops from killing the job and makes it easier to reconnect safely.

To stop only the pipeline, target its PID instead of sweeping all Python processes. For example, save the PID to a file and use it to terminate the run:
```bash
nohup python3 tools/run_local_pipeline.py \
  --scene-dir ./test_scenes/scenes/lightwheel_kitchen \
  --steps genie-sim-export,genie-sim-submit,genie-sim-import \
  --use-geniesim --fail-fast \
  > /tmp/pipeline_run.log 2>&1 &
echo $! > /tmp/pipeline_run.pid
kill "$(cat /tmp/pipeline_run.pid)"
```
Alternatively, locate the pipeline process directly:
```bash
kill "$(pgrep -f "tools/run_local_pipeline.py")"
```

### Stop VM (saves costs — run from local machine)
```bash
gcloud compute instances stop isaac-sim-ubuntu --zone=us-east1-b
```
When status is `TERMINATED`, VM CPU/GPU runtime billing stops. Persistent disk and reserved static IP charges still apply.

## Testing with Different Robots

The pipeline supports multiple robot embodiments. Currently working: **G1 humanoid** (via `G1_omnipicker_fixed.json`). Franka Panda crashes the server during init (see Remaining Known Issues #6).
Robot configs live in `tools/geniesim_adapter/robot_configs/` (one JSON per robot).

### Switching robot type
```bash
export GENIESIM_ROBOT_TYPE=ur10
# or for the full pipeline:
export ROBOT_TYPES=ur10
```

### Multi-robot data generation
```bash
export ROBOT_TYPES=franka,ur10
export ENABLE_MULTI_ROBOT=1
python3 tools/run_local_pipeline.py \
  --scene-dir ./test_scenes/scenes/lightwheel_kitchen \
  --steps genie-sim-export,genie-sim-submit,genie-sim-import \
  --use-geniesim --fail-fast
```

### Adding a new robot

1. Create `tools/geniesim_adapter/robot_configs/<robot>.json` (see `franka_panda.json` as template)
2. Ensure robot exists in `policy_configs/robot_embodiments.json`
3. If using cuRobo, add config in `curobo` section of the JSON
4. Camera prims are auto-discovered from the robot config's `"camera"` dict

### Supported robots (Isaac Sim compatible)

| Robot | DOFs | Gripper | Config JSON | Tested |
|-------|------|---------|-------------|--------|
| Franka Panda | 7 | parallel jaw | `franka_panda.json` | Server crashes on init (issue #6) |
| UR10 | 6 | none (tool0) | `ur10.json` | No |
| UR5e | 6 | parallel jaw | — | No |
| UR10e | 6 | parallel jaw | — | No |
| Kuka iiwa | 7 | parallel jaw | — | No |
| Fetch | 7 | parallel jaw | — | No |
| Tiago | 7 | parallel jaw | — | No |
| Kinova Gen3 | 7 | parallel jaw | — | No |

## Interpreting Pipeline Output

### Success indicators in logs
- `init_robot returned: success=True` — Server accepted robot + scene
- `Joint names populated: 34 joints` — Real joint names available
- `[DIAG] USD stage: N total prims` — Stage diagnostics showing loaded scene
- `[DIAG] Cameras: [...]` — Available cameras in server stage
- `[PATCH] Camera /path: (720, 1280, 3), non-zero pixels: ...` — Camera rendering
- `Trajectory complete (N waypoints)` — Motion execution succeeded
- `[OBS] Got real object pose for /World/Object` — Object tracking working

### Data quality checks
1. **Joint positions**: Should vary across trajectory waypoints, not all zeros
2. **Camera images**: Check non-zero pixel count (should be >50% of total)
3. **Object poses**: Position should match scene layout, not identity `(0,0,0)`
4. **Gripper state**: Should toggle between open/closed during task

### Common data issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| All camera frames black (`RGB=0`, `alpha=255`) | Two modes: (1) **Display path misconfigured** (`--no-window`, no X socket, bad `DISPLAY`), or (2) **L4 RTX geometry failure** after scene load (see Known Issue #6): depth is infinity everywhere and only dome background is visible | First verify display path: no `--no-window`, host Xorg on `:99`, valid `/tmp/.X11-unix` mount. If those are correct and depth remains infinity, treat as L4 rendering incompatibility and switch GPU family or Isaac Sim version |
| Object poses all zeros | Objects not in server USD stage | Check `[DIAG]` logs; verify `scene_usd_path` in init_robot |
| Joint positions all same | Trajectory execution timed out | Check for `DEADLINE_EXCEEDED`; increase `GENIESIM_FIRST_CALL_TIMEOUT_S` |
| Only 1/3 cameras return data | Robot config has fewer cameras than requested | Normal for arm robots (typically 1 wrist camera) |
| `DEADLINE_EXCEEDED` on first trajectory | cuRobo lazy init on first motion command | Timeout now 300s; increase via `GENIESIM_FIRST_CALL_TIMEOUT_S=600` if needed |

## Things That Trip Up Claude Code

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `scp --recurse` hangs | Repo is huge (~22GB Docker images cached) | Sync only changed files individually |
| `docker` permission denied | Docker requires `sudo` on this VM | Always prefix with `sudo` |
| SSH "Connection refused" | VM just booted, sshd not ready | `sleep 30` after `gcloud instances start` |
| `ModuleNotFoundError: tools` | Missing PYTHONPATH | `export PYTHONPATH=~/BlueprintPipeline:~/BlueprintPipeline/episode-generation-job:$PYTHONPATH` |
| `ModuleNotFoundError: tools.quality_gates.physics_certification` | `tools/` directory not fully synced to VM | Sync entire `tools/` dir: `gcloud compute scp --recurse --compress --zone=us-east1-b tools/ isaac-sim-ubuntu:~/BlueprintPipeline/tools/` |
| `ModuleNotFoundError: data_fidelity` | `episode-generation-job/` not on PYTHONPATH | Add `~/BlueprintPipeline/episode-generation-job` to PYTHONPATH |
| Quality gates block pipeline | "No PhysicsScene defined" | `export SKIP_QUALITY_GATES=1` |
| `variation-gen` fails | Tries to create `/mnt/gcs` (needs root) | Skip this step, or set `GCS_MOUNT_ROOT` |
| Gemini dimension estimation fails | `google-genai` not installed on VM | `pip3 install google-genai` |
| `docker restart` loses runtime patches? | No — `restart` preserves filesystem. Only `rm + create` loses changes | Use `docker restart`, not `docker rm` + new container |
| `docker compose down && up` loses runtime edits | Container recreate resets files under `/opt/geniesim` | Re-apply patch scripts (or rebuild image with patches baked in) |
| Pipeline appears "running" but log is frozen at `Running init sequence` | Stale init deadlock after prior run/container state | Stop pipeline + guard, restart container (`docker restart geniesim-server`), wait for gRPC ready, rerun |
| Container gRPC not listening | Server still initializing (~45s) | Check `/proc/net/tcp6` for port `C383` |
| `ss` command not found in container | Minimal container image | Use `cat /proc/net/tcp6 | grep C383` instead |
| `/bin/bash: ... start_geniesim_server.sh: Permission denied` | Executable bit not set on startup script | `chmod +x tools/geniesim_adapter/deployment/start_geniesim_server.sh` then restart container |
| VM SSH hangs/unresponsive | OOM — server + pipeline consume >64GB RAM | Use g2-standard-32 (128GB); or run pipeline client outside container |
| `DEADLINE_EXCEEDED` during trajectory | Trajectory execution takes time in simulation | Normal; retry logic handles it. Extend timeout via `GENIESIM_FIRST_CALL_TIMEOUT_S` |
| cuRobo `'TensorDeviceType' has no attribute 'mesh'` | cuRobo API version mismatch with Isaac Sim 5.1 | Non-fatal — IK fallback trajectory works. Needs cuRobo version pin investigation |

## Known Server-Side Bugs (Pre-Existing in Genie Sim)

These are bugs in the upstream Genie Sim server code, not our pipeline:

1. **`get_joint_position` → "string indices must be integers"**: The robot articulation init fails (`KeyError: 'joint_0'`), causing `_get_joint_positions()` to return a string error instead of a dict.

2. **`get_ee_pose` → "too many values to unpack (expected 2)"**: Server does `position, rotation = ...` but the robot returns 3+ values. Our `patch_ee_pose_handler.py` fixes this, but the patch pattern-match may miss some code paths.

3. **`grpc_server.py` set literal bugs**: Lines 538 and 555 use `{"reset", Reset}` (Python set) instead of `{"reset": Reset}` (dict). Fixed with sed at runtime.

4. **First `set_joint_position` takes >180s**: Server does heavy lazy init (cuRobo, motion generation) on first joint command. Often causes DEADLINE_EXCEEDED. Our warmup call helps but doesn't fully solve it.

5. **Object poses return zeros**: Objects at prim paths like `/World/Toaster003` exist in our scene graph but aren't loaded in the server's USD stage, so `get_object_pose` returns identity transforms. Stage diagnostics patch now logs what's actually loaded.

6. **Camera frames are black (`RGB=0`, `alpha=255`) even after display fixes**: On this setup, the L4 + Isaac Sim 5.x path can fail to render geometry after stage load. Symptoms: RGB stays zero, alpha stays 255, depth is infinity across frame, and only dome light sky color appears. Display-path fixes (`--no-window` removal, host Xorg `:99`, valid X socket mount) are still required, but they are not sufficient in this failure mode.

7. **ROS2 "No such file" errors**: Server scripts previously always passed `--publish_ros` even when ROS2 Humble isn't installed. Fixed: `--publish_ros` is now conditional on `GENIESIM_SKIP_ROS_RECORDING` (set to 1 in Dockerfile).

8. **`get_object_pose` → "too many values to unpack"**: `grpc_server.py` line 327 does `position, rotation = object_pose` but the server returns 3+ values. Fixed by `patch_grpc_server.py`.

9. **`reset` → "bad argument type for built-in operation"**: `blocking_start_server` returns non-string result, but `rsp.msg` expects string. Fixed by `patch_grpc_server.py`.

10. **`get_observation` recordingState**: `rsp.recordingState = result` fails when result is a dict. Fixed by `patch_grpc_server.py`.

## Server Patches

All patches are in `tools/geniesim_adapter/deployment/patches/` and applied during Docker build or at runtime.

| Patch | What it fixes |
|-------|--------------|
| `patch_omnigraph_dedup.py` | Prevents OmniGraph duplicate graph creation errors |
| `patch_data_collector_render_config.py` | Patches `data_collector_server.py` to avoid `--no-window` in camera mode, enforce `RaytracedLighting`, and set stable render args/settings |
| `patch_camera_handler.py` | Adds `GET_CAMERA_DATA` command handler with Replicator rendering; logs available cameras |
| `patch_object_pose_handler.py` | Fuzzy prim path matching for object pose queries |
| `patch_ee_pose_handler.py` | Safe unpacking for `get_ee_pose` multi-value returns |
| `patch_stage_diagnostics.py` | Logs USD stage contents (prims, cameras, meshes) after `init_robot` |
| `patch_observation_cameras.py` | Auto-registers unknown cameras in `self.cameras` using `{camera_info, prim_path}` dict entries; fixes `publish_ros` ValueError crash |
| `patch_grpc_server.py` | Fixes: string conversions, safe unpacking, set literal bugs, numpy scalar conversion, position/rotation tuple unpacking |

## Recent Fixes (2026-01-31)

1. **Joint name warmup**: Added `get_joint_position()` warmup during robot init to populate real joint names before `set_joint_position`. Prevents synthetic `"joint_0"` names causing server `KeyError`.
2. **Non-trajectory timeout**: Extended timeout logic now applies to both trajectory and non-trajectory `set_joint_position()` calls (was only trajectory before).
3. **Scene USD path**: `init_robot()` now uses the actual scene USD from export instead of defaulting to `empty_scene.usda`. Fixes object poses returning zeros.
4. **First-call timeout**: Increased default `GENIESIM_FIRST_CALL_TIMEOUT_S` from 180s to 300s.
5. **EE pose patch**: Broadened regex to catch multi-line `get_ee_pose()` calls and added safety wrapper for unmatched patterns.
6. **Camera map from robot config**: Camera prim paths now auto-populated from robot config JSON instead of hardcoded G1 paths. Falls back to `GENIESIM_CAMERA_PRIM_MAP` env var or G1 defaults.
7. **Stage diagnostics patch**: New `patch_stage_diagnostics.py` logs all USD stage prims (cameras, meshes, xforms) after `init_robot` to debug object pose zeros.
8. **Conditional `--publish_ros`**: Server start scripts no longer pass `--publish_ros` when `GENIESIM_SKIP_ROS_RECORDING=1`, eliminating ROS2 error noise.
9. **UR10 robot config**: Added `robot_configs/ur10.json` for testing with Universal Robots UR10.
10. **Camera auto-registration** (`patch_observation_cameras.py`): Server crashes with `KeyError` on Franka camera prims because `self.cameras` dict only contains G1 cameras. Patch auto-registers unknown cameras with default resolution at start of `handle_get_observation`, storing `{camera_info: {width, height, ppx, ppy, fx, fy}, prim_path}` dict entries. Legacy list/string entries are normalized before access, and `self.cameras[camera][0]` is replaced with a safe dict fallback.
11. **`publish_ros` ValueError**: Server raises `ValueError("publish ros is not enabled")` during `startRecording` when `--publish_ros` flag not passed. Patched to log warning and continue.
12. **Position/rotation tuple unpacking**: `grpc_server.py` does `(x, y, z) = position` but position can be numpy arrays with >3 elements. Fixed with safe element-by-element assignment via `float(_pos[i])`.
13. **`GetJointRsp` has no "errmsg" field**: Previous joint position guard tried to set `rsp.errmsg` which doesn't exist in the protobuf schema. Replaced with `print()` warning.
14. **VM upgraded to g2-standard-32**: Running both Isaac Sim server + pipeline client in the same container OOM'd at 43/64GB RAM during trajectory execution. Upgraded to 128GB RAM.
15. **Pipeline achieved `task_success=True`**: End-to-end flow works — IK fallback trajectory (17 waypoints), composed observations with joints, objects(6), cameras(1). cuRobo planner fails with `'TensorDeviceType' object has no attribute 'mesh'` (API version mismatch) but IK fallback works.
16. **`could not convert string to float: 'm'`**: Some objects (CoffeeMachine006, variation_pot) return unusual prim data that fails `float()` conversion. Non-fatal — client handles gracefully.
17. **CRITICAL: Pipeline stuck after task_success=True**: The monitoring loop detected task completion but didn't abort the trajectory execution thread, causing infinite DEADLINE_EXCEEDED retry loops. Fixed by: (a) threading `abort_event` into `execute_trajectory()` so the waypoint loop checks it between each waypoint and after each gRPC call; (b) setting `abort_event` and `execution_state["success"] = True` when task_success is detected; (c) using short 2s gRPC timeout when abort is already signaled.
18. **Safe float parsing for unit-suffixed values**: Server-side `float(_pos[i])` fails when USD prim attributes contain unit suffixes like `"1.5 m"`. Injected `_bp_safe_float()` helper into `grpc_server.py` that strips non-numeric suffixes before conversion. Applied via `_apply_safe_float.py` patch.
19. **Path translation always uses container paths**: Previously, `os.path.exists()` check meant the translation was skipped when running on the VM host (where the path exists locally). Now always translates absolute paths containing `BlueprintPipeline/` to `/workspace/BlueprintPipeline/...` container paths, since the server always runs in Docker.
20. **Pipeline advances through multiple tasks**: With fixes 17-19, the pipeline successfully advanced from Task 1/7 through Task 3/7 before the server became UNAVAILABLE (server-side stability issue). Task 1 achieved `task_success=True` and completed in ~3.5 min instead of looping indefinitely.
21. **Object pose fuzzy matcher improved**: `_bp_resolve_prim_path()` now collects ALL candidate prims with matching tail name and scores them: prefers `UsdGeom.Xformable` prims (+100), prims under `/World/` (+50), shorter paths. Falls back to case-insensitive substring matching if exact tail-name match fails. Client-side also tries `/Root/` prefix and stripped `/World/` prefix variants, with successful resolutions cached in `_resolved_prim_cache`.
22. **Zero-pose objects excluded from observations**: `_get_object_pose_raw()` now returns `None` (instead of the zero-pose dict) when all position components are zero, preventing corrupted data from entering the dataset. The observation builder already handles `None` by skipping the object.
23. **Server auto-restart on UNAVAILABLE**: Three-layer resilience: (a) `start_geniesim_server.sh` now has a restart loop (max `GENIESIM_MAX_SERVER_RESTARTS`, default 3) that automatically relaunches the server process if it crashes; (b) client-side `_attempt_server_restart()` runs `docker restart geniesim-server` when the circuit breaker opens (enabled by default, disable with `GENIESIM_AUTO_RESTART=0`), with 5-min cooldown and max `GENIESIM_MAX_RESTARTS` (default 3); (c) inter-task delay (`GENIESIM_INTER_TASK_DELAY_S`, default 5s) with health probe before each task — if probe fails, triggers restart.
24. **EE pose patch broadened**: Primary regex now uses `.*?` (was `[^)]*`) to cross newlines and handles N-variable unpacking (`pos, rot, extra = ...`). Fallback catches all exceptions (was only ValueError/TypeError). Added monkey-patch wrapper on `robot.get_ee_pose` that always returns exactly 2 values, injected after `self.robot` assignment in `init_robot`.
25. **Stall count reset per task**: `_stall_count` now resets to 0 at the start of each task, so stalls in earlier tasks don't consume the budget for later tasks.
26. **Proactive server restart every N tasks**: Set `GENIESIM_RESTART_EVERY_N_TASKS=3` to automatically stop and restart the server (with robot re-init) every N tasks. Prevents GPU/RAM resource exhaustion that causes DEADLINE_EXCEEDED cascade on task 4+. Default `0` (disabled).
27. **Resilient reset failure handling**: When `reset_environment()` fails or returns unavailable, the pipeline now attempts a server restart + robot re-init and retries the episode, instead of aborting all remaining tasks.
28. **Camera re-warmup option**: Set `CAMERA_REWARMUP_ON_RESET=1` to re-run Replicator warmup frames on every camera capture call. Fixes intermittent camera data after the first observation (render pipeline may go stale after physics reset). Default `0` (disabled).
29. **Task checkpoint/resume on retry**: Completed tasks are saved to `_completed_tasks.json` in the run directory. When the pipeline retries (via `_run_with_retry`), it reuses the same run directory (hash-based naming) and skips already-completed tasks instead of restarting from task 1.
30. **Health probe bug confirmed fixed**: The inter-task health probe correctly uses `self._client.get_joint_position()` (not `self.get_joint_position()`). No change needed — was already correct in deployed code.

31. **Checkpoint bug fix**: Only checkpoint tasks with ≥1 successful episode. Previously, tasks were checkpointed even when all episodes failed (e.g., reset failure → continue → end of loop), causing retries to skip failed tasks and produce 0 episodes.
32. **EE pose patch covers `_get_ee_pose` code path**: The gRPC handler calls `self.ui_builder._get_ee_pose(is_right)` (line 1790 of `command_controller.py`), NOT `robot.get_ee_pose()`. Updated regex from `\.get_ee_pose\(` to `\._?get_ee_pose\(` to match both. Also matches ALL occurrences (not just first) and uses unique temp vars per match site.
33. **Object pose diagnostic logging**: When fuzzy prim path resolution fails, dumps the first 50 prims from the USD stage (once per session). Also logs `scene_usd_path`, whether the file exists, and the `sim_assets_root`/`scene_usd` values used to construct it.

34. **CRITICAL: Franka robot config was mapped to G1 humanoid**: `_ROBOT_CFG_MAP["franka"]` pointed to `G1_omnipicker_fixed.json` (80+ DOF humanoid, 10+ min init) instead of `franka_panda.json` (7 DOF, ~30s init). Fixed to map to `franka_panda.json`.
35. **Object path candidates expanded**: Added `/World/Scene/obj_{name}` and `/World/Scene/{name}` as candidate prim paths for object pose queries, matching the server's scene graph prefix convention.
36. **Execution thread abort before join**: `abort_event.set()` is now called before the execution thread join (was missing), and join timeout increased from 30s to 60s. Ensures the trajectory loop exits promptly instead of blocking on an in-flight gRPC call.
37. **franka_panda.json missing from server**: The Genie Sim server only ships G1/G2 configs in `/opt/geniesim/source/data_collection/config/robot_cfg/`. Copied `franka_panda.json` to the server container. **Must be re-copied after `docker rm + create` (not needed after `docker restart`).** Added to sync/patch workflow.
38. **Server crashes during init_robot with franka_panda.json**: Even with the config file present, the server crashes (GOAWAY/ping_timeout) during `init_robot`. The `init_robot` call DEADLINE_EXCEEDED after 300-600s, then the server process dies. The server restart loop brings it back but the same pattern repeats. **Root cause unknown** — the server may not fully support loading a Franka robot via `franka_panda.json` (it ships with G1 configs only). See issue 6 below.

### Recent Fixes (2026-02-01)

39. **Simulation paused deadlock fix**: Added `reset()` call after `_init_robot_on_server()` to ensure the simulation timeline is PLAYING, not PAUSED. This was the root cause of the paused deadlock during mid-run restarts.
40. **Proactive restarts disabled**: `GENIESIM_RESTART_EVERY_N_TASKS` default changed from `5` to `0`. Mid-run Docker restarts were causing the paused deadlock — disabling them eliminates the trigger.
41. **Observation collector resilience**: Both streaming and polling observation collectors now tolerate up to 3 consecutive frame failures before aborting (previously failed on the first bad frame).
42. **Partial episode data**: Failed episodes now save diagnostic `.partial.json` files so you can always inspect what data was collected before the failure.
43. **xforms patch cache fix**: Added `ARG CACHE_BUST_PATCHES` to Dockerfile to ensure both `Rotation.from_matrix()` patch sites are applied (verified count=2).

### Recent Fixes (2026-02-07)

44. **CRITICAL: Object motion validation fails in cert-strict mode**: In cert-strict mode (`_cert_strict=True`), client-side kinematic object pose tracking is deliberately blocked (lines 11051-11053, 10952, 11273 of `local_framework.py`). This means the server returns static USD positions for kinematic objects, so `validate_object_motion()` always sees `max_displacement=0.0` and raises `DataFidelityError`. **Fix**: Use EE-manipulation-phase displacement as a proxy for object motion. During manipulation phases (`grasp`, `lift`, `transport`, `place`), the end-effector positions come from the real server. If EE displacement exceeds 1cm during these phases, the object must have moved. This does NOT relax the strictness guarantee because it uses only server-backed data.
45. **PYTHONPATH must include `episode-generation-job/`**: The `data_fidelity` module is imported from `episode-generation-job/data_fidelity.py`. Without `~/BlueprintPipeline/episode-generation-job` on PYTHONPATH, the pipeline fails with `ModuleNotFoundError: No module named 'data_fidelity'`. Updated all PYTHONPATH exports in this guide.
46. **Must sync entire `tools/` directory**: The old individual file sync list was missing `tools/quality_gates/`, `tools/camera_io.py`, and other modules. Pipeline failed at startup with `No module named 'tools.quality_gates.physics_certification'`. Now recommend `gcloud compute scp --recurse` for `tools/`.
47. **Task 5 (`variation_toaster`) crashes server**: The `variation_toaster` object cannot be resolved (all prim path variants return zero/identity poses). The server crashes during this task repeatedly (UNAVAILABLE / Connection reset by peer). Server auto-restart works but the task keeps failing. Not fixed — 6/7 tasks complete successfully with this task skipped.

## Remaining Known Issues

1. **Object poses still all zeros**: Fix 33 adds diagnostic logging; fix 35 adds `/World/Scene/obj_{name}` path candidates. **Next step**: Check Docker logs for `[PATCH-DIAG]` messages once init_robot succeeds.

2. **Server resource exhaustion on long runs**: **Mitigation**: `GENIESIM_RESTART_EVERY_N_TASKS=3`. **Next step**: Profile GPU/RAM with `nvidia-smi`.

3. **Camera data intermittent after first observation**: **Mitigation**: `CAMERA_REWARMUP_ON_RESET=1`.

4. **cuRobo API version mismatch**: IK fallback works. **Next step**: Pin cuRobo version for Isaac Sim 5.1.

5. **MIN_EPISODE_FRAMES and EE pose failures**: Previously observed with G1 config. May resolve once correct robot config works.

7. **CRITICAL: L4 GPU (Ada Lovelace / SM 89) cannot render geometry in Isaac Sim 5.1.0-rc.19**:
   The RTX ray tracing renderer produces `RGB=0` for ALL cameras (including the default viewport `/OmniverseKit_Persp`). Extensive testing confirmed:
   - **Simple primitives with `displayColor`**: Appear to render (R=G=B=243) but that's actually just the dome light sky background — depth shows INFINITY everywhere, meaning rays pass through all geometry
   - **Kitchen scene with 203 meshes + 441 materials**: RGB=0 on all cameras, depth works correctly (shows actual distances)
   - **Driver upgrades**: Tested both NVIDIA 580.105.08 and 590.48.01 — same result
   - **Xorg config**: Proper host Xorg on :99 with NVIDIA GLX, GLFW initializes successfully
   - **Render modes**: Both `RayTracedLighting` and `PathTracing` — same result
   - **Async rendering**: Disabled, sync material loading forced — same result
   - The RTX BVH (Bounding Volume Hierarchy) acceleration structures appear to not build correctly on the L4's Ada Lovelace architecture
   - **Everything else works**: robot init, joint control, trajectory execution, object poses, gRPC, task progression

   **Options**:
   - **Option A (recommended)**: Switch to a T4 (Turing / SM 75) or A10G (Ampere / SM 86) GPU VM — these are better tested with Isaac Sim
   - **Option B**: Try Isaac Sim 4.x (stable release) instead of 5.1.0-rc.19 (release candidate)
   - **Option C**: File NVIDIA bug report with reproduction script (`/tmp/no_dome_test.py` demonstrates the issue)

6. **CRITICAL: Server crashes during init_robot with franka_panda.json**: The Genie Sim server only ships G1/G2 robot configs. When `franka_panda.json` is provided, the server attempts to load the Franka USD (`robot/franka/franka.usd`) and crashes during initialization (GOAWAY after 300-600s DEADLINE_EXCEEDED). **Tested twice with same result.** The previous working runs (task_success=True) used `G1_omnipicker_fixed.json`. **Options for next run:**
   - **Option A**: Revert `_ROBOT_CFG_MAP["franka"]` back to `G1_omnipicker_fixed.json` and accept slow G1 init (~10 min). The pipeline DID achieve task_success=True with G1 before.
   - **Option B**: Investigate why the server crashes with Franka. Check if `robot/franka/franka.usd` exists in the server container. Check server logs during init for the actual error. The Franka USD may not be bundled in the Docker image.
   - **Option C**: Build a custom Franka config that uses the G1 server internals but maps to Franka-compatible joint names. This is complex and not recommended.
   - **Recommended**: Try Option B first (check if Franka USD exists), fall back to Option A if not fixable quickly.

## Running Pipeline Client Inside Container

For testing, you can run the pipeline client directly inside the container (requires g2-standard-32 for memory):

```bash
sudo docker exec -d geniesim-server bash -c '
  export GENIESIM_ROOT=/opt/geniesim
  export ISAAC_SIM_PATH=/isaac-sim
  export OMNI_KIT_ALLOW_ROOT=1
  export PYTHONUNBUFFERED=1
  export PYTHONPATH=/workspace/BlueprintPipeline/tools/geniesim_adapter:/workspace/BlueprintPipeline:${PYTHONPATH:-}
  export GENIESIM_SKIP_ROS_RECORDING=1
  export ROBOT_TYPES=franka
  /isaac-sim/python.sh -m tools.geniesim_adapter.local_framework run \
    --scene /workspace/BlueprintPipeline/test_scenes/scenes/lightwheel_kitchen/geniesim/merged_scene_manifest.json \
    --task-config /workspace/BlueprintPipeline/test_scenes/scenes/lightwheel_kitchen/geniesim/task_config.json \
    --robot franka \
    --episodes 1 \
    > /tmp/pipeline_test.log 2>&1
'
# Monitor: sudo docker exec geniesim-server tail -f /tmp/pipeline_test.log
```

**Important**: Filter log noise with `grep -v "CUROBO_TORCH_COMPILE\|simulation paused"` — cuRobo emits hundreds of env var warnings.

### What to expect from a successful run
1. cuRobo init logs (many `CUROBO_TORCH_COMPILE` warnings — harmless)
2. `[GENIESIM-LOCAL]   IK fallback trajectory: 17 waypoints` — trajectory planned
3. `First trajectory set_joint_position — using extended timeout 300.0s` — execution started
4. `[OBS] Composed real observation from: joints, objects(N), cameras(N)` — data collected
5. `task_success=True, near_trajectory_end=True` — episode complete

### Current data quality (as of 2026-02-07)
- **Joint positions**: Working (real data from server)
- **Camera images**: **BLOCKED** — L4 GPU RTX renderer cannot produce color (see Known Issue #7). Requires GPU type change (T4/A10G) or Isaac Sim version change. Display path (Xorg `:99`, no `--no-window`) is correctly configured
- **Object poses**: Improved fuzzy matching with multi-candidate scoring; zero-pose objects now excluded from observations (returned as None). Server-side prim resolution logs candidates. If objects still return zeros, check `[PATCH] Prim path` and `[DIAG]` log lines to compare requested paths vs actual stage contents.
- **EE pose**: Monkey-patch wrapper ensures safe 2-value return; regex broadened for N-variable unpacking. Previously intermittent, should now be fully handled.
- **Trajectory**: IK fallback works; cuRobo planner has API version mismatch
- **Task progression**: Pipeline advances through tasks with auto-restart on server crash. Server startup script has restart loop (max 3). Client triggers `docker restart` when circuit breaker opens. Inter-task health probe detects failures early. Completed tasks are checkpointed so retries skip them. Stall budget resets per task. Proactive server restart available via `GENIESIM_RESTART_EVERY_N_TASKS`.
- **Object motion**: In cert-strict mode, kinematic objects don't move on server. EE-manipulation-displacement proxy (fix 44) uses server-backed EE positions during grasp/lift/transport phases as evidence of motion. Validated: 0.33-0.34m per task on Feb 7 strict run.
- **Strict production run**: 6/7 tasks completed (Feb 7 2026). Quality scores 0.69-0.84. All realism gates pass except `variation_toaster` which crashes the server.

### Downloading data from the VM

```bash
# Find the latest run directory
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b -- \
  'ls -lt ~/BlueprintPipeline/test_scenes/scenes/lightwheel_kitchen/episodes/ | head -5'

# Download all episode data (compress for speed)
gcloud compute scp --recurse --compress --zone=us-east1-b \
  "isaac-sim-ubuntu:~/BlueprintPipeline/test_scenes/scenes/lightwheel_kitchen/episodes/<session-id>/recordings/<run-id>/" \
  ~/Downloads/BlueprintPipeline_<run-id>/
```

Data is organized as:
```
run_<hash>/
  per_task/
    task_0/
      raw_episodes/
        episode_*.json   # ~800KB each, 60 frames, all fields
    task_1/
    ...
  _completed_tasks.json  # checkpoint file
```

Each episode JSON contains: joint positions/velocities/efforts, EE pose/velocity/acceleration, object poses/velocities, gripper state, phase labels, camera intrinsics, and task metadata.

## Pipeline Step Names

Use dashes, not underscores:
```
simready, usd, replicator, variation-gen, isaac-lab,
genie-sim-export, genie-sim-submit, genie-sim-import
```

Full list: `regen3d, scale, interactive, simready, usd, inventory-enrichment, replicator, variation-gen, isaac-lab, genie-sim-export, genie-sim-submit, genie-sim-import, dataset-delivery, dwm, dwm-inference, dream2flow, dream2flow-inference, validate`

## Environment Variables

| Variable | Purpose | Where to set |
|----------|---------|-------------|
| `GEMINI_API_KEY` | Gemini dimension estimation | `.env` file (auto-loaded via python-dotenv) |
| `SKIP_QUALITY_GATES` | Skip blocking quality gates | Export before pipeline run |
| `PYTHONPATH` | Module resolution | Must include `~/BlueprintPipeline` AND `~/BlueprintPipeline/episode-generation-job` |
| `GENIESIM_FIRST_CALL_TIMEOUT_S` | First init/joint call timeout (start with `600` on smaller VMs; `300` often enough on g2-standard-32) | Export before pipeline run |
| `GENIESIM_GRPC_TIMEOUT_S` | General gRPC timeout (default 60) | Export before pipeline run |
| `ROBOT_TYPES` | Comma-separated robot types for multi-robot (default: `franka`) | Export before pipeline run |
| `GENIESIM_ROBOT_TYPE` | Single robot type (default: `franka`) | Export before pipeline run |
| `ENABLE_MULTI_ROBOT` | Enable multi-robot data generation | Export before pipeline run |
| `GENIESIM_CAMERA_PRIM_MAP` | JSON map of logical camera name → USD prim path | Export before pipeline run |
| `GENIESIM_ROBOT_CFG_FILE` | Override server robot config filename | Export before pipeline run |
| `GENIESIM_SKIP_ROS_RECORDING` | Set to `1` to skip `--publish_ros` (default in Docker) | Dockerfile / export |
| `GENIESIM_CAMERA_REQUIRE_DISPLAY` | If `1` (default), fail fast when cameras are enabled but no valid X socket is present | Dockerfile / export |
| `DISPLAY` | X display used by Isaac Sim camera rendering (default `:99`) | docker-compose / export |
| `NVIDIA_DRIVER_CAPABILITIES` | NVIDIA runtime capabilities; set to `all` for graphics + compute | docker-compose / export |
| `CAMERA_RESOLUTION` | Camera resolution `WxH` (default: `1280x720`) | Dockerfile / export |
| `CAMERA_WARMUP_STEPS` | Replicator warmup frames before valid camera data (default: `5`) | Dockerfile / export |
| `GENIESIM_AUTO_RESTART` | Auto-restart server on circuit breaker open (default: `1`, set `0` to disable) | Export before pipeline run |
| `GENIESIM_MAX_RESTARTS` | Max client-side container restarts per session (default: `3`) | Export before pipeline run |
| `GENIESIM_INTER_TASK_DELAY_S` | Delay in seconds between tasks with health probe (default: `5`) | Export before pipeline run |
| `GENIESIM_MAX_SERVER_RESTARTS` | Max server-side process restarts in startup script (default: `3`) | Dockerfile / export |
| `GENIESIM_RESTART_CMD` | Command to restart server container (default: `sudo docker restart geniesim-server`) | Export before pipeline run |
| `GENIESIM_RESTART_EVERY_N_TASKS` | Proactive server restart every N tasks (default: `0` = disabled; was `5`, changed to `0` to avoid mid-run Docker restarts causing paused deadlock) | Export before pipeline run |
| `CAMERA_REWARMUP_ON_RESET` | Re-run camera warmup frames on every capture (default: `0` = disabled) | Dockerfile / export |
