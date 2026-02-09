# VM & Container Operations Guide

Quick reference for getting the BlueprintPipeline running on the GCP VM with GenieSim.

## VM Details

| Field | Value |
|-------|-------|
| Instance | `isaac-sim-ubuntu` |
| Zone | `us-east1-b` |
| Machine | `g2-standard-32` (NVIDIA L4 24GB VRAM, 128GB RAM) |
| Docker | Requires `sudo` |
| User home | `/home/nijelhunt1` |
| Repo path | `~/BlueprintPipeline` |

## One-Shot Cold Start (from local machine)

```bash
# 1. Start VM and wait for SSH
gcloud compute instances start isaac-sim-ubuntu --zone=us-east1-b
sleep 35

# 2. Sync code
gcloud compute scp --recurse --compress --zone=us-east1-b \
  tools/ "isaac-sim-ubuntu:~/BlueprintPipeline/tools/"
for dir in policy_configs scripts episode-generation-job configs; do
  gcloud compute scp --recurse --compress --zone=us-east1-b \
    "$dir/" "isaac-sim-ubuntu:~/BlueprintPipeline/$dir/"
done
for f in Dockerfile.geniesim-server docker-compose.geniesim-server.yaml \
         genie-sim-local-job/requirements.txt .env run_pipeline.sh; do
  gcloud compute scp --zone=us-east1-b \
    "$f" "isaac-sim-ubuntu:~/BlueprintPipeline/$f" 2>/dev/null
done

# 3. CRITICAL: Remove nested dirs created by scp --recurse
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="
  rm -rf ~/BlueprintPipeline/tools/tools/
  rm -rf ~/BlueprintPipeline/policy_configs/policy_configs/
  rm -rf ~/BlueprintPipeline/scripts/scripts/
  rm -rf ~/BlueprintPipeline/episode-generation-job/episode-generation-job/
  rm -rf ~/BlueprintPipeline/configs/configs/
"

# 4. Start Xorg + restart container + wait for gRPC
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="
  cd ~/BlueprintPipeline && bash scripts/vm-start-xorg.sh
"
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="
  sudo docker restart geniesim-server >/dev/null 2>&1
"
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="
  cd ~/BlueprintPipeline && bash scripts/vm-start.sh
"

# 5. Run pipeline (uses checked-in run_pipeline.sh which sources configs/realism_strict.env)
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="
  cd ~/BlueprintPipeline && bash run_pipeline.sh
"

# 6. Monitor
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="tail -f /tmp/pipeline_strict.log"
```

## Quick Start

### If the VM is already running

```bash
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b
cd ~/BlueprintPipeline
bash scripts/vm-start.sh   # checks container is up, polls gRPC until ready
bash run_pipeline.sh        # sources configs/realism_strict.env, runs pipeline
# Monitor: tail -f /tmp/pipeline_strict.log
```

### Cold start (VM was stopped)

```bash
# 1. Start VM
gcloud compute instances start isaac-sim-ubuntu --zone=us-east1-b

# 2. Wait ~30s for SSH, then connect
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b

# 3. Container auto-starts with the VM (restart: unless-stopped).
cd ~/BlueprintPipeline
bash scripts/vm-start.sh

# 4. Run pipeline
bash run_pipeline.sh
# Monitor: tail -f /tmp/pipeline_strict.log
```

### If a run appears stuck at init

Symptoms: pipeline alive but log frozen at `Running init sequence (initial)`.

```bash
pkill -9 -f 'tools/run_local_pipeline.py' 2>/dev/null || true
pkill -9 -f 'pipeline_guard.sh' 2>/dev/null || true

cd ~/BlueprintPipeline
bash scripts/vm-start-xorg.sh
sudo docker restart geniesim-server >/dev/null
bash scripts/vm-start.sh
```

Low CPU/GPU utilization with frozen logs = stale init deadlock. Restart container, don't scale.

## Pipeline Configuration

All pipeline env vars are in `configs/realism_strict.env` (sourced by `run_pipeline.sh`).

### Current settings (Feb 2026)

| Setting | Value | Reason |
|---------|-------|--------|
| `REQUIRE_REAL_EFFORTS` | `true` | PhysX joint torques confirmed working |
| `REQUIRE_REAL_CONTACTS` | `false` | Objects remain kinematic (dynamic fall-through unsolved) |
| `REQUIRE_OBJECT_MOTION` | `false` | Kinematic objects don't move on grasp |
| `GENIESIM_ALLOW_IK_FAILURE_FALLBACK` | `1` | cuRobo may not init; allow joint-space interpolation fallback |
| `GENIESIM_STRICT_RUNTIME_READINESS` | `0` | Strict probe fails with kinematic objects |
| `GENIESIM_ENABLE_SCENE_PHYSICS_PROBE` | `0` | Kinematic objects ignore gravity |
| `GENIESIM_INIT_ROBOT_FAILOVER_TIMEOUT_S` | `180` | cuRobo failover can take 2+ min |
| `SKIP_RGB_CAPTURE` | `true` | L4 GPU cannot render color (Known Issue #7) |

### What data is real vs fallback

| Data | Source | Status |
|------|--------|--------|
| Joint efforts | PhysX (`measured_joint_forces`) | Real |
| Joint positions/velocities | Server articulation | Real |
| EE pose | Server (monkey-patched safe unpacking) | Real |
| Object positions | Manifest transform fallback | Static (kinematic) |
| Object motion | N/A | Disabled (kinematic) |
| Contact data | N/A | Disabled (kinematic) |
| Camera RGB | N/A | Disabled (L4 rendering bug) |

## Robot Configuration

The server ONLY supports the **G1 humanoid** (via `G1_omnipicker_fixed.json`). All defaults use `robot_type="g1"`.

Franka Panda crashes the server during init (the Franka USD is not bundled in the Docker image). Do NOT set `robot_type="franka"`.

## Syncing Code Changes

```bash
# Full sync (recommended)
gcloud compute scp --recurse --compress --zone=us-east1-b \
  tools/ "isaac-sim-ubuntu:~/BlueprintPipeline/tools/"
for dir in policy_configs scripts episode-generation-job configs; do
  gcloud compute scp --recurse --compress --zone=us-east1-b \
    "$dir/" "isaac-sim-ubuntu:~/BlueprintPipeline/$dir/"
done

# CRITICAL: Remove nested dirs created by scp --recurse
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b --command="
  rm -rf ~/BlueprintPipeline/tools/tools/
  rm -rf ~/BlueprintPipeline/policy_configs/policy_configs/
  rm -rf ~/BlueprintPipeline/scripts/scripts/
  rm -rf ~/BlueprintPipeline/episode-generation-job/episode-generation-job/
  rm -rf ~/BlueprintPipeline/configs/configs/
"
```

**scp nesting bug**: `gcloud compute scp --recurse dir/ remote:path/dir/` creates `dir/dir/`. Python resolves modules from the nested copy, causing `FileNotFoundError`. Always run cleanup after syncing.

## Server Patches

Patches are applied automatically on container start by `start_geniesim_server.sh`. They modify GenieSim server files in `/opt/geniesim/` at runtime.

### Patch chain (applied in order)

| Patch | Critical | Purpose |
|-------|----------|---------|
| `patch_camera_handler.py` | No | Camera data command handler |
| `patch_observation_cameras.py` | No | Auto-register unknown cameras |
| `patch_object_pose_handler.py` | No | Fuzzy prim path matching |
| `patch_ee_pose_handler.py` | No | Safe EE pose unpacking |
| `patch_data_collector_render_config.py` | No | Import fix, render settings |
| `patch_contact_report.py` | Yes | PhysX contact reporting |
| `patch_joint_efforts_handler.py` | Yes | Real joint effort extraction |
| `patch_enable_contacts_on_init.py` | Yes | Enable contacts during init |
| `patch_sim_thread_physics_cache.py` | Yes | Cache efforts+contacts on sim thread |
| `patch_register_scene_objects.py` | Yes | Register scene objects for dynamics (v3) |
| `patch_deferred_dynamic_restore.py` | Yes | Deferred dynamic restore (v4) |
| `patch_dynamic_teleport_v5.py` | Yes | Dynamic teleport fix (v5) |
| `patch_fix_dynamic_prims_overwrite.py` | Yes | Prevent dynamic prims overwrite (v6) |
| `patch_scene_collision.py` | Yes | Add CollisionAPI to meshes, validate coverage |
| `patch_ui_builder_time_import.py` | Yes | Fix `time` import in ui_builder.py (prevents cuRobo init crash) |

Bootstrap (`bootstrap_geniesim_runtime.sh`) applies additional first-time patches:
`patch_articulation_physics_wait.py`, `patch_ui_builder_time_import.py`, `patch_set_joint_guard.py`, `patch_camera_crash_guard.py`, `patch_autoplay.py`, plus collision baking and grpc_server.py fixes.

### Key patch notes

- **v3** (`patch_register_scene_objects.py`): Restores objects from kinematic to dynamic after articulation init. Uses `GENIESIM_CC_PATH` env var (defaults to `/opt/geniesim/...`).
- **v5** (`patch_scene_collision.py`): Adds CollisionAPI + convexHull approximation to ALL mesh prims. Runs BEFORE v3's dynamic restore. Validates 100% collision coverage.
- **v7** (`patch_keep_objects_kinematic.py`): Keeps objects kinematic. Only applied when `GENIESIM_KEEP_OBJECTS_KINEMATIC=1`. Currently DISABLED (=0) in strict mode.
- **`patch_ui_builder_time_import.py`**: Fixes `UnboundLocalError: cannot access local variable 'time'` at ui_builder.py:437. The `patch_articulation_physics_wait.py` added `import time` inside an `except` block, which Python treats as a local variable for the entire function scope. Without this fix, cuRobo init crashes and triggers a failover restart loop.

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

### Full Docker rebuild (bakes all patches permanently)
```bash
cd ~/BlueprintPipeline
sudo docker compose -f docker-compose.geniesim-server.yaml build
sudo docker compose -f docker-compose.geniesim-server.yaml up -d
```

### Run pipeline in background
```bash
cd ~/BlueprintPipeline
bash run_pipeline.sh
# Monitor: tail -f /tmp/pipeline_strict.log
# Stop: kill "$(cat /tmp/pipeline_run.pid)"
```

### Stop VM (saves ~$1.17/hr)
```bash
gcloud compute instances stop isaac-sim-ubuntu --zone=us-east1-b
```
The g2-standard-32 with L4 GPU costs **$1.17/hr on-demand** ($0.47/hr spot) in us-east1. When status is `TERMINATED`, CPU/GPU billing stops. Disk and static IP charges still apply.

## Display Path For Camera RGB

Camera RGB requires valid X display: host Xorg on `:99`, container `DISPLAY=:99`, `/tmp/.X11-unix` mounted.

**Xorg config note (driver 590+)**: Remove `UseDisplayDevice "None"` and use `AllowEmptyInitialConfiguration "true"`. The `vm-start-xorg.sh` script handles this.

## Downloading Data

```bash
# Find latest run directory
gcloud compute ssh isaac-sim-ubuntu --zone=us-east1-b -- \
  'ls -lt ~/BlueprintPipeline/test_scenes/scenes/lightwheel_kitchen/episodes/ | head -5'

# Download
gcloud compute scp --recurse --compress --zone=us-east1-b \
  "isaac-sim-ubuntu:~/BlueprintPipeline/test_scenes/scenes/lightwheel_kitchen/episodes/<session-id>/" \
  ~/Downloads/BlueprintPipeline_episodes/
```

Data layout:
```
run_<hash>/per_task/task_N/raw_episodes/episode_*.json
```
Each episode JSON (~800KB, 60 frames): joint positions/velocities/efforts, EE pose, object poses, gripper state, phase labels, camera intrinsics, task metadata.

## Things That Trip Up Claude Code

| Issue | Fix |
|-------|-----|
| `scp --recurse` creates nested dirs (`tools/tools/`) | Run `rm -rf ~/BlueprintPipeline/tools/tools/` after sync |
| SSH quoting breaks `nohup`/`$!` | Upload a shell script via scp, then run it |
| `docker` permission denied | Always use `sudo` |
| SSH "Connection refused" | `sleep 30` after `gcloud instances start` |
| `ModuleNotFoundError: tools` | `export PYTHONPATH=~/BlueprintPipeline:~/BlueprintPipeline/episode-generation-job:$PYTHONPATH` |
| Pipeline stuck at `Running init sequence` | Stale init deadlock — restart container, rerun |
| `DEADLINE_EXCEEDED` on first trajectory | cuRobo lazy init — timeout 600s handles it |
| Container gRPC not listening | Server initializing (~45s) — check `/proc/net/tcp6` for `C383` |
| `start_geniesim_server.sh: Permission denied` | `chmod +x tools/geniesim_adapter/deployment/start_geniesim_server.sh` |

## Known Issues (Feb 2026)

1. **Dynamic objects fall through surfaces**: Even with 100% collision coverage (202/202 mesh prims), objects restored to dynamic by v3 fall through tables. Unsolved — we accept kinematic objects and use manifest positions.

2. **cuRobo API version mismatch**: `'TensorDeviceType' has no attribute 'mesh'`. IK fallback chain works: server cuRobo -> local numerical IK -> joint-space interpolation.

3. **Franka robot crashes server**: Server only ships G1/G2 configs. Franka USD not bundled. Use `robot_type="g1"` only.

4. **L4 GPU cannot render color**: RTX ray tracing produces RGB=0 on all cameras. Depth works. Options: switch to T4/A10G GPU, or use Isaac Sim 4.x instead of 5.1.0-rc.19.

5. **`variation_toaster` crashes server**: Object cannot be resolved; server crashes repeatedly during this task. 6/7 tasks complete with it skipped.

6. **init_robot hang (FIXED)**: `UnboundLocalError: cannot access local variable 'time'` in ui_builder.py caused cuRobo init crash and failover loop. Fixed by `patch_ui_builder_time_import.py` (now wired into both startup scripts).

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `GENIESIM_HOST` | gRPC server host | `localhost` |
| `GENIESIM_PORT` | gRPC server port | `50051` |
| `GENIESIM_FIRST_CALL_TIMEOUT_S` | First init/joint call timeout | `600` |
| `GENIESIM_GRPC_TIMEOUT_S` | General gRPC timeout | `120` |
| `GENIESIM_OBJECT_POSE_TIMEOUT_S` | Object pose query timeout | `30` |
| `GENIESIM_INIT_ROBOT_FAILOVER_TIMEOUT_S` | cuRobo failover timeout | `180` |
| `GENIESIM_AUTO_RESTART` | Auto-restart on circuit breaker open | `1` |
| `GENIESIM_MAX_RESTARTS` | Max client-side container restarts | `3` |
| `GENIESIM_INTER_TASK_DELAY_S` | Delay between tasks with health probe | `5` |
| `GENIESIM_RESTART_EVERY_N_TASKS` | Proactive restart every N tasks | `0` (disabled) |
| `CAMERA_REWARMUP_ON_RESET` | Re-run camera warmup on each capture | `1` |
| `SKIP_RGB_CAPTURE` | Skip camera RGB | `true` |
| `DATA_FIDELITY_MODE` | Fidelity mode (`production`/`development`) | `production` |
| `STRICT_REALISM` | Fail on realism violations | `true` |

See `configs/realism_strict.env` for the full set.

## Pipeline Step Names

Use dashes, not underscores:
```
simready, usd, replicator, variation-gen, isaac-lab,
genie-sim-export, genie-sim-submit, genie-sim-import
```
