# VM & Container Operations Guide

Quick reference for getting the BlueprintPipeline running on the GCP VM with Genie Sim.

## VM Details

| Field | Value |
|-------|-------|
| Instance | `isaac-sim-ubuntu` |
| Zone | `us-east1-b` |
| Machine | `g2-standard-16` (NVIDIA L4, 24GB VRAM) |
| Docker | Requires `sudo` |
| User home | `/home/nijelhunt1` |
| Repo path | `~/BlueprintPipeline` |

## Quick Start (from scratch)

```bash
# 1. Start VM
gcloud compute instances start isaac-sim-ubuntu --zone=us-east1-b

# 2. Wait for SSH (VM needs ~30s after boot)
sleep 30
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="echo ready"

# 3. Sync code changes (only modified files - full rsync is too slow)
for f in \
  tools/run_local_pipeline.py \
  tools/geniesim_adapter/local_framework.py \
  tools/geniesim_adapter/deployment/patches/patch_camera_handler.py \
  tools/geniesim_adapter/deployment/patches/patch_ee_pose_handler.py \
  tools/geniesim_adapter/deployment/patches/patch_object_pose_handler.py \
  tools/geniesim_adapter/deployment/patches/patch_omnigraph_dedup.py \
  tools/geniesim_adapter/deployment/patches/patch_grpc_server.py \
  tools/geniesim_adapter/deployment/patches/patch_stage_diagnostics.py \
  tools/geniesim_adapter/deployment/bootstrap_geniesim_runtime.sh \
  tools/geniesim_adapter/deployment/start_geniesim_server.sh \
  tools/geniesim_adapter/robot_configs/franka_panda.json \
  tools/geniesim_adapter/robot_configs/ur10.json \
  Dockerfile.geniesim-server \
  genie-sim-local-job/requirements.txt \
  .env; do
  gcloud compute scp --zone=us-east1-b \
    "/Users/nijelhunt_1/workspace/BlueprintPipeline/$f" \
    "isaac-sim-ubuntu:~/BlueprintPipeline/$f" 2>/dev/null
done

# 4. Start container (if not running)
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="
  sudo docker start geniesim-server 2>/dev/null || \
  (cd ~/BlueprintPipeline && \
   sudo -E docker-compose -f docker-compose.geniesim-server.yaml up -d)
"

# 5. Wait for gRPC server (~45s after container start)
sleep 45
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="
  sudo docker exec geniesim-server bash -c \
    'cat /proc/net/tcp6 | grep C383 && echo SERVER_READY || echo NOT_READY'
"

# 6. Run pipeline
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="
  cd ~/BlueprintPipeline &&
  export PYTHONPATH=~/BlueprintPipeline:\$PYTHONPATH &&
  export SKIP_QUALITY_GATES=1 &&
  python3 tools/run_local_pipeline.py \
    --scene-dir ./test_scenes/scenes/lightwheel_kitchen \
    --steps simready,usd,replicator,isaac-lab,genie-sim-export,genie-sim-submit,genie-sim-import \
    --use-geniesim \
    --fail-fast
"
```

## Common Operations

### Check container status
```bash
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="sudo docker ps -a"
```

### Check gRPC port (50051 = 0xC383)
```bash
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="
  sudo docker exec geniesim-server bash -c 'cat /proc/net/tcp6 | grep C383'
"
```

### View server logs
```bash
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="sudo docker logs geniesim-server --tail 50"
```

### Apply patches at runtime (no rebuild)
```bash
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="
  for p in patch_omnigraph_dedup patch_camera_handler patch_object_pose_handler \
           patch_ee_pose_handler patch_stage_diagnostics patch_grpc_server; do
    sudo docker cp ~/BlueprintPipeline/tools/geniesim_adapter/deployment/patches/\${p}.py geniesim-server:/tmp/
    sudo docker exec geniesim-server /isaac-sim/python.sh /tmp/\${p}.py
  done
  sudo docker restart geniesim-server
"
```

### Fix grpc_server.py set literal bugs (runtime)
```bash
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="
  sudo docker exec geniesim-server sed -i 's/data={\"reset\", Reset}/data={\"reset\": Reset}/' /opt/geniesim/source/data_collection/server/grpc_server.py
  sudo docker exec geniesim-server sed -i 's/data={\"detach\", detach}/data={\"detach\": detach}/' /opt/geniesim/source/data_collection/server/grpc_server.py
  sudo docker restart geniesim-server
"
```

### Full Docker rebuild (bakes all patches permanently)
```bash
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="
  cd ~/BlueprintPipeline &&
  sudo docker build -f Dockerfile.geniesim-server -t geniesim-server:latest .
"
# Takes 30-45 min (cuRobo compilation)
```

### Run pipeline in background (for long runs)
```bash
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="
  cd ~/BlueprintPipeline &&
  export PYTHONPATH=~/BlueprintPipeline:\$PYTHONPATH &&
  export SKIP_QUALITY_GATES=1 &&
  nohup python3 tools/run_local_pipeline.py \
    --scene-dir ./test_scenes/scenes/lightwheel_kitchen \
    --steps genie-sim-export,genie-sim-submit,genie-sim-import \
    --use-geniesim --fail-fast \
    > /tmp/pipeline_run.log 2>&1 &
  echo PID: \$!
"
# Monitor: tail -f /tmp/pipeline_run.log
```

### Stop VM (saves costs)
```bash
gcloud compute instances stop isaac-sim-ubuntu --zone=us-east1-b
```

## Testing with Different Robots

The pipeline supports multiple robot embodiments. Currently tested: **Franka Panda**.
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
| Franka Panda | 7 | parallel jaw | `franka_panda.json` | Yes |
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
| All camera frames black | Render product not created for camera prim | Check `[PATCH] Available cameras in stage` log; set `GENIESIM_CAMERA_PRIM_MAP` |
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
| `ModuleNotFoundError: tools` | Missing PYTHONPATH | `export PYTHONPATH=~/BlueprintPipeline:$PYTHONPATH` |
| Quality gates block pipeline | "No PhysicsScene defined" | `export SKIP_QUALITY_GATES=1` |
| `variation-gen` fails | Tries to create `/mnt/gcs` (needs root) | Skip this step, or set `GCS_MOUNT_ROOT` |
| Gemini dimension estimation fails | `google-genai` not installed on VM | `pip3 install google-genai` |
| `docker restart` loses runtime patches? | No — `restart` preserves filesystem. Only `rm + create` loses changes | Use `docker restart`, not `docker rm` + new container |
| Container gRPC not listening | Server still initializing (~45s) | Check `/proc/net/tcp6` for port `C383` |
| `ss` command not found in container | Minimal container image | Use `cat /proc/net/tcp6 | grep C383` instead |

## Known Server-Side Bugs (Pre-Existing in Genie Sim)

These are bugs in the upstream Genie Sim server code, not our pipeline:

1. **`get_joint_position` → "string indices must be integers"**: The robot articulation init fails (`KeyError: 'joint_0'`), causing `_get_joint_positions()` to return a string error instead of a dict.

2. **`get_ee_pose` → "too many values to unpack (expected 2)"**: Server does `position, rotation = ...` but the robot returns 3+ values. Our `patch_ee_pose_handler.py` fixes this, but the patch pattern-match may miss some code paths.

3. **`grpc_server.py` set literal bugs**: Lines 538 and 555 use `{"reset", Reset}` (Python set) instead of `{"reset": Reset}` (dict). Fixed with sed at runtime.

4. **First `set_joint_position` takes >180s**: Server does heavy lazy init (cuRobo, motion generation) on first joint command. Often causes DEADLINE_EXCEEDED. Our warmup call helps but doesn't fully solve it.

5. **Object poses return zeros**: Objects at prim paths like `/World/Toaster003` exist in our scene graph but aren't loaded in the server's USD stage, so `get_object_pose` returns identity transforms. Stage diagnostics patch now logs what's actually loaded.

6. **Camera frames are black**: Replicator `step()` returns black frames — annotators may need warm-up or the render pipeline isn't fully initialized. Camera patch now logs available cameras and warm-up status.

7. **ROS2 "No such file" errors**: Server scripts previously always passed `--publish_ros` even when ROS2 Humble isn't installed. Fixed: `--publish_ros` is now conditional on `GENIESIM_SKIP_ROS_RECORDING` (set to 1 in Dockerfile).

8. **`get_object_pose` → "too many values to unpack"**: `grpc_server.py` line 327 does `position, rotation = object_pose` but the server returns 3+ values. Fixed by `patch_grpc_server.py`.

9. **`reset` → "bad argument type for built-in operation"**: `blocking_start_server` returns non-string result, but `rsp.msg` expects string. Fixed by `patch_grpc_server.py`.

10. **`get_observation` recordingState**: `rsp.recordingState = result` fails when result is a dict. Fixed by `patch_grpc_server.py`.

## Server Patches

All patches are in `tools/geniesim_adapter/deployment/patches/` and applied during Docker build or at runtime.

| Patch | What it fixes |
|-------|--------------|
| `patch_omnigraph_dedup.py` | Prevents OmniGraph duplicate graph creation errors |
| `patch_camera_handler.py` | Adds `GET_CAMERA_DATA` command handler with Replicator rendering; logs available cameras |
| `patch_object_pose_handler.py` | Fuzzy prim path matching for object pose queries |
| `patch_ee_pose_handler.py` | Safe unpacking for `get_ee_pose` multi-value returns |
| `patch_stage_diagnostics.py` | Logs USD stage contents (prims, cameras, meshes) after `init_robot` |
| `patch_grpc_server.py` | 26 fixes: string conversions, safe unpacking, set literal bugs, numpy scalar conversion |

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
| `PYTHONPATH` | Module resolution | Must include `~/BlueprintPipeline` |
| `GENIESIM_FIRST_CALL_TIMEOUT_S` | First joint call timeout (default 300) | Export before pipeline run |
| `GENIESIM_GRPC_TIMEOUT_S` | General gRPC timeout (default 60) | Export before pipeline run |
| `ROBOT_TYPES` | Comma-separated robot types for multi-robot (default: `franka`) | Export before pipeline run |
| `GENIESIM_ROBOT_TYPE` | Single robot type (default: `franka`) | Export before pipeline run |
| `ENABLE_MULTI_ROBOT` | Enable multi-robot data generation | Export before pipeline run |
| `GENIESIM_CAMERA_PRIM_MAP` | JSON map of logical camera name → USD prim path | Export before pipeline run |
| `GENIESIM_ROBOT_CFG_FILE` | Override server robot config filename | Export before pipeline run |
| `GENIESIM_SKIP_ROS_RECORDING` | Set to `1` to skip `--publish_ros` (default in Docker) | Dockerfile / export |
| `CAMERA_RESOLUTION` | Camera resolution `WxH` (default: `1280x720`) | Dockerfile / export |
| `CAMERA_WARMUP_STEPS` | Replicator warmup frames before valid camera data (default: `5`) | Dockerfile / export |
