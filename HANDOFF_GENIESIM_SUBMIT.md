# Genie Sim Submit - Handoff Document

## What We're Doing
Testing the BlueprintPipeline end-to-end using a Lightwheel KitchenRoom SimReady scene, bypassing Phase 1 (3D-RE-GEN). The pipeline has multiple steps; we're stuck on **Part 3: genie-sim-submit** which runs robot data collection via a remote Genie Sim server.

## Infrastructure
- **Client**: Mac (local), runs `tools/run_local_pipeline.py`
- **Server**: GCP VM `isaac-sim-ubuntu` (zone `us-east1-b`, IP `34.138.160.175`), Docker container `geniesim-server` running Isaac Sim 4.5.0 + Genie Sim gRPC server on port 50051
- **Scene**: `test_scenes/scenes/lightwheel_kitchen/` with SimReady USD assets

## What's Done (Parts 1-2)
1. **Scene setup** - Lightwheel kitchen scene configured with manifest, objects at z=-2.3 to -2.5
2. **genie-sim-export** - PASSED. Generated `geniesim/task_config.json` with 4 tasks (2 pick_place, 1 organize, 1 interact). Robot at base_position [0.5, 0.5, -2.3], workspace_bounds [[-0.5,-0.5,-2.8],[1.5,1.5,-1.8]]

## What's Blocked (Part 3: genie-sim-submit)

### The Problem
When the pipeline calls `reset()` or `init_robot()` gRPC RPCs, the server's `CommandController` crashes because:
1. **`SIM_ASSETS` env var was not set** - The server reads `os.environ.get("SIM_ASSETS")` (line 68 of `command_controller.py`) and gets `None`
2. **No robot USD assets** - `init_robot()` needs robot USD files pointed to by `SIM_ASSETS`
3. **Server only ships with G1/G2 robot configs** - No Franka config exists on the server

### Code Changes Already Made

**`tools/geniesim_adapter/local_framework.py`:**
- `ping()` (~line 1042): Changed from `get_observation()` RPC to TCP socket check (server crashes if robot not loaded)
- `check_simulation_ready()` (~line 3242): Changed from `reset()+get_observation()` to socket+channel check
- `run_data_collection()` (~line 3391): Added `init_robot()` call before episode loop with robot config mapping:
  ```python
  _ROBOT_CFG_MAP = {"franka": "G1_omnipicker_fixed.json", "g1": "G1_omnipicker_fixed.json", ...}
  ```
  This mapping needs updating once a Franka config is deployed to the server.

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
- For commercial use: need a custom Franka config (draft at `tools/geniesim_adapter/robot_configs/franka_panda.json`)

## Steps to Complete

### Immediate (unblock testing with G1 robot)
1. On the VM, clean up GenieSimAssets: `cd ~/GenieSimAssets && rm -rf objects scenes materials background`
2. Set up SIM_ASSETS for the container. Either:
   - `sudo docker stop geniesim-server && sudo docker rm geniesim-server`
   - `cd ~/BlueprintPipeline && SIM_ASSETS_HOST_PATH=/home/ohstnhunt/GenieSimAssets docker compose -f docker-compose.geniesim-server.yaml up -d`
   - OR just set env on existing: stop container, re-create with `-v /home/ohstnhunt/GenieSimAssets:/sim-assets:ro -e SIM_ASSETS=/sim-assets`
3. Wait ~90s for server startup ("simulation paused" in logs = ready)
4. Copy `franka_panda.json` to server's robot_cfg dir: `sudo docker cp tools/geniesim_adapter/robot_configs/franka_panda.json geniesim-server:/opt/geniesim/source/data_collection/config/robot_cfg/franka_panda.json`
5. Update `_ROBOT_CFG_MAP` in `local_framework.py` line ~3405: change `"franka": "G1_omnipicker_fixed.json"` to `"franka": "franka_panda.json"`
6. **BUT**: The `franka_panda.json` references `"robot_usd": "robot/franka/franka.usd"` which must exist under `SIM_ASSETS`. The Franka USD is in Isaac Sim at `/isaac-sim/exts/isaacsim.asset.browser/data/Isaac/Robots/FrankaEmika/` — you may need to symlink or copy it into the SIM_ASSETS dir, or use a G1 robot for testing first.

### Fastest path to test (use G1 robot)
1. Do steps 1-3 above
2. Keep `_ROBOT_CFG_MAP` as `"franka": "G1_omnipicker_fixed.json"`
3. The G1 robot will be spawned instead of Franka — mismatch with task config but will validate the pipeline flow
4. Run: `GENIESIM_HOST=34.138.160.175 GENIESIM_PORT=50051 GENIESIM_WORKSPACE_BOUNDS_JSON='[[-0.5,-0.5,-2.8],[1.5,1.5,-1.8]]' ALLOW_GENIESIM_MOCK=1 GENIESIM_ROOT=/tmp/geniesim-stub ISAAC_SIM_PATH=/tmp/isaac-sim-stub python -m tools.run_local_pipeline --scene-dir test_scenes/scenes/lightwheel_kitchen --steps genie-sim-submit --mock-geniesim --reset-breaker`

### For production Franka support
1. Create proper Franka robot_description YAML with joint limits, collision spheres (see G1 yaml as template)
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
- `/opt/geniesim/source/data_collection/config/robot_cfg/` - Robot configs (G1_omnipicker_fixed.json, etc.)
- `/opt/geniesim/source/data_collection/scripts/data_collector_server.py` - gRPC server entry point
