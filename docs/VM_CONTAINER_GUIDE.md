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
  tools/geniesim_adapter/deployment/bootstrap_geniesim_runtime.sh \
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
  sudo docker cp ~/BlueprintPipeline/tools/geniesim_adapter/deployment/patches/patch_camera_handler.py geniesim-server:/tmp/
  sudo docker cp ~/BlueprintPipeline/tools/geniesim_adapter/deployment/patches/patch_ee_pose_handler.py geniesim-server:/tmp/
  sudo docker cp ~/BlueprintPipeline/tools/geniesim_adapter/deployment/patches/patch_object_pose_handler.py geniesim-server:/tmp/
  sudo docker exec geniesim-server /isaac-sim/python.sh /tmp/patch_camera_handler.py
  sudo docker exec geniesim-server /isaac-sim/python.sh /tmp/patch_ee_pose_handler.py
  sudo docker exec geniesim-server /isaac-sim/python.sh /tmp/patch_object_pose_handler.py
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

5. **Object poses return zeros**: Objects at prim paths like `/World/Toaster003` exist in our scene graph but aren't loaded in the server's USD stage, so `get_object_pose` returns identity transforms.

6. **Camera frames are black**: Replicator `step()` returns black frames — annotators may need warm-up or the render pipeline isn't fully initialized.

7. **ROS2 "No such file" errors**: Container uses `--publish_ros` but ROS2 Humble isn't installed in the base Isaac Sim 4.5 image. Non-fatal — only affects camera topic republishing.

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
| `GENIESIM_FIRST_CALL_TIMEOUT_S` | First joint call timeout (default 180) | Export before pipeline run |
| `GENIESIM_GRPC_TIMEOUT_S` | General gRPC timeout (default 60) | Export before pipeline run |
