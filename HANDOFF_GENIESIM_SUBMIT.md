# Genie Sim Submit - Handoff Document

## What We're Doing
Testing the BlueprintPipeline end-to-end using a Lightwheel KitchenRoom SimReady scene, bypassing Phase 1 (3D-RE-GEN). The pipeline runs robot data collection via a Genie Sim gRPC server on a GCP VM.

## Current Status (2026-02-12)
The pipeline uses **Franka Panda** (`franka_panda.json`). If robot init fails, the run fails (no automatic robot failover). See `docs/VM_CONTAINER_GUIDE.md` for the full operational runbook and current known issues.

## Infrastructure
- **Client**: VM (SSH in), runs `tools/run_local_pipeline.py`
- **Server**: GCP VM `isaac-sim-ubuntu` (zone `us-east1-b`, IP `34.138.160.175`), Docker container `geniesim-server` running Isaac Sim 5.1.0 + Genie Sim gRPC server on port 50051
- **Scene**: `test_scenes/scenes/lightwheel_kitchen/` with SimReady USD assets

## What's Done (Parts 1-2)
1. **Scene setup** - Lightwheel kitchen scene configured with manifest, objects at z=-2.3 to -2.5
2. **genie-sim-export** - PASSED. Generated `geniesim/task_config.json` with 4 tasks (2 pick_place, 1 organize, 1 interact). Robot at base_position [0.5, 0.5, -2.3], workspace_bounds [[-0.5,-0.5,-2.8],[1.5,1.5,-1.8]]

## Previously Blocked (Part 3: genie-sim-submit) — RESOLVED

The following issues were resolved:
1. **`SIM_ASSETS` env var** — Set in `docker-compose.geniesim-server.yaml`
2. **Robot USD assets** — Pre-baked into Docker image via `download_robot_assets.sh`
3. **Robot configs** — Franka Panda (`franka_panda.json`) installed and selected

### Code Changes Already Made

**`tools/geniesim_adapter/local_framework.py`:**
- `ping()` (~line 1042): Changed from `get_observation()` RPC to TCP socket check (server crashes if robot not loaded)
- `check_simulation_ready()` (~line 3242): Changed from `reset()+get_observation()` to socket+channel check
- `run_data_collection()` (~line 3391): Added robot init before the episode loop using Franka Panda (`franka_panda.json`).

**`tools/run_local_pipeline.py`:**
- `_steps_require_geniesim_preflight()` (~line 1836): Removed `GENIESIM_EXPORT` from preflight check

**`tools/geniesim_adapter/task_config.py`:**
- `base_position` z-coordinate (~line 946): Fixed to compute from workspace bounds center instead of hardcoded 0.0

**`docker-compose.geniesim-server.yaml`:**
- Added `SIM_ASSETS: /sim-assets` env var
- Added volume mount `${SIM_ASSETS_HOST_PATH:-./sim-assets}:/sim-assets:ro`

**`test_scenes/scenes/lightwheel_kitchen/assets/scene_manifest.json`:**
- Room bounds emptied (`"room": {}`) to allow `GENIESIM_WORKSPACE_BOUNDS_JSON` override

### Assets Status
- GenieSimAssets cloned to VM at `/home/ohstnhunt/GenieSimAssets` (14.25 GB, CC BY-NC-SA 4.0)
- User wants to keep only `robot/` subdir (delete objects/scenes/materials/background)
- For commercial use: need a custom Franka config (`tools/geniesim_adapter/robot_configs/franka_panda.json`)

## Steps to Complete

### Primary Path (Franka Panda)
1. On the VM, clean up GenieSimAssets: `cd ~/GenieSimAssets && rm -rf objects scenes materials background`
2. Set up SIM_ASSETS for the container. Either:
   - `sudo docker stop geniesim-server && sudo docker rm geniesim-server`
   - `cd ~/BlueprintPipeline && SIM_ASSETS_HOST_PATH=/home/ohstnhunt/GenieSimAssets docker compose -f docker-compose.geniesim-server.yaml up -d`
   - OR just set env on existing: stop container, re-create with `-v /home/ohstnhunt/GenieSimAssets:/sim-assets:ro -e SIM_ASSETS=/sim-assets`
3. Wait ~90s for server startup ("simulation paused" in logs = ready)
4. Copy `franka_panda.json` to server's robot_cfg dir: `sudo docker cp tools/geniesim_adapter/robot_configs/franka_panda.json geniesim-server:/opt/geniesim/source/data_collection/config/robot_cfg/franka_panda.json`
5. Ensure the Franka USD exists under `SIM_ASSETS`: `franka_panda.json` references `"robot_usd": "robot/franka/franka.usd"`. If it is missing, the server will fail init and the client will error (no robot failover).

### Selecting a Different Robot (Optional)
If you need to use a different robot config explicitly:
1. Do steps 1-3 above
2. Run with `GENIESIM_ROBOT_TYPE=<robot>` (or set `GENIESIM_ROBOT_CFG_FILE=<robot_cfg.json>`)

### For production Franka support
1. Create proper Franka robot_description YAML with joint limits, collision spheres (use an existing robot_description yaml as a template)
2. Copy Franka USD from Isaac Sim into SIM_ASSETS: `mkdir -p $SIM_ASSETS/robot/franka && cp /isaac-sim/.../franka.usd $SIM_ASSETS/robot/franka/`
3. Validate `franka_panda.json` prim paths match the actual Franka USD hierarchy
4. Test `init_robot()` → `reset()` → `get_observation()` sequence

## Key Environment Variables for Running Submit
```bash
GENIESIM_HOST=34.138.160.175
GENIESIM_PORT=50051
GENIESIM_WORKSPACE_BOUNDS_JSON='[[-0.5,-0.5,-2.8],[1.5,1.5,-1.8]]'
ALLOW_GENIESIM_MOCK=1
GENIESIM_ROOT=/tmp/geniesim-stub
ISAAC_SIM_PATH=/tmp/isaac-sim-stub
```
Stubs at `/tmp/geniesim-stub` and `/tmp/isaac-sim-stub/python.sh` must exist on the Mac for preflight to pass.

## Key Files
- `tools/geniesim_adapter/local_framework.py` - Core framework, gRPC client, data collection loop
- `tools/run_local_pipeline.py` - Pipeline orchestrator
- `tools/geniesim_adapter/task_config.py` - Task generation, robot placement
- `docker-compose.geniesim-server.yaml` - Server Docker config
- `test_scenes/scenes/lightwheel_kitchen/geniesim/task_config.json` - Generated task config (4 tasks)
- `tools/geniesim_adapter/robot_configs/franka_panda.json` - Draft Franka config for Genie Sim server

## Server-Side Key Files (inside container)
- `/opt/geniesim/source/data_collection/server/command_controller.py` - Line 68: `self.sim_assets_root = os.environ.get("SIM_ASSETS")`, Line 222-225: loads robot USD
- `/opt/geniesim/source/data_collection/config/robot_cfg/` - Robot configs (franka_panda.json, etc.)
- `/opt/geniesim/source/data_collection/scripts/data_collector_server.py` - gRPC server entry point
