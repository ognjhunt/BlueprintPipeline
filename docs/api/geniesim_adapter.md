# `tools/geniesim_adapter` API

## Purpose

`tools/geniesim_adapter` bridges BlueprintPipeline scene outputs with Genie Sim 3.0, handling scene graph export, asset indexing, task configuration, and local data collection orchestration (via Isaac Sim + gRPC). It provides both export utilities (`GenieSimExporter`) and a local framework client (`GenieSimLocalFramework`) so the pipeline can generate Genie Sim episode bundles without relying on a hosted API.【F:tools/geniesim_adapter/__init__.py†L1-L99】【F:tools/geniesim_adapter/local_framework.py†L1-L96】

## Public entrypoints

Importable via `tools.geniesim_adapter`:

- Export and configuration:
  - `GenieSimExporter`
  - `GenieSimExportConfig`
  - `GenieSimExportResult`
- Scene/task asset builders:
  - `SceneGraphConverter`, `GenieSimSceneGraph`, `GenieSimNode`, `GenieSimEdge`
  - `AssetIndexBuilder`, `GenieSimAsset`, `GenieSimAssetIndex`
  - `TaskConfigGenerator`, `GenieSimTaskConfig`, `SuggestedTask`
- Multi-robot configuration helpers:
  - `MultiRobotConfig`, `RobotType`, `RobotCategory`, `RobotSpec`
  - `ROBOT_SPECS`, `DEFAULT_MULTI_ROBOT_CONFIG`, `FULL_ROBOT_CONFIG`
  - `get_robot_spec`, `get_geniesim_robot_config`
- Local data collection framework:
  - `GenieSimLocalFramework`, `GenieSimConfig`, `GenieSimServerStatus`, `DataCollectionResult`
  - `check_geniesim_availability`, `run_local_data_collection`【F:tools/geniesim_adapter/__init__.py†L52-L139】

## Configuration / environment variables

Local framework configuration uses environment variables when instantiating `GenieSimConfig.from_env` or running the local framework CLI:

- `GENIESIM_HOST` (default: `localhost`)
- `GENIESIM_PORT` (default: `50051`)
- `GENIESIM_TIMEOUT` (default: `30` seconds)
- `GENIESIM_ROOT` (default: `/opt/geniesim`)
- `ISAAC_SIM_PATH` (default: `/isaac-sim`)
- `ALLOW_GENIESIM_MOCK` (default: `0`)
- `HEADLESS` (default: `1`)
- `ROBOT_TYPE` (default: `franka`)

These are referenced directly by the local framework adapter, along with additional settings derived from `GenieSimConfig` for episodes per task and robot selection.【F:tools/geniesim_adapter/local_framework.py†L52-L96】【F:tools/geniesim_adapter/local_framework.py†L214-L249】

## Request/response payloads & data models

### Exporter payloads

- **Request**: `GenieSimExportConfig` controls export behavior such as robot selection, multi-robot behavior, task count, and metadata toggles. Key fields include `robot_type`, `enable_multi_robot`, `max_tasks`, and `filter_commercial_only`.【F:tools/geniesim_adapter/exporter.py†L46-L121】
- **Response**: `GenieSimExportResult` describes exported outputs and stats:
  - `scene_graph_path`, `asset_index_path`, `task_config_path`, `scene_config_path`
  - `num_nodes`, `num_edges`, `num_assets`, `num_tasks`
  - `errors`, `warnings`
  - `to_dict()` for JSON serialization【F:tools/geniesim_adapter/exporter.py†L123-L197】

### Local framework payloads

- **Request**: `GenieSimLocalFramework.run_data_collection` expects a `task_config` dict (typically produced by `TaskConfigGenerator`) and optional `scene_config` dict, plus optional `episodes_per_task` override. The request payload is structured to contain `suggested_tasks` and scene metadata such as `usd_path` when available.【F:tools/geniesim_adapter/local_framework.py†L1019-L1078】
- **Response**: `DataCollectionResult` includes success flags, episode metrics, and output locations (`recording_dir`). It also captures quality metrics and errors for auditing runs.【F:tools/geniesim_adapter/local_framework.py†L250-L279】

## Example usage

```python
from pathlib import Path
from tools.geniesim_adapter import (
    GenieSimExporter,
    GenieSimExportConfig,
    GenieSimLocalFramework,
    check_geniesim_availability,
)

# 1) Export BlueprintPipeline scene artifacts for Genie Sim
exporter = GenieSimExporter(
    GenieSimExportConfig(robot_type="franka", max_tasks=25)
)
export_result = exporter.export(
    manifest_path=Path("scenes/kitchen/assets/scene_manifest.json"),
    output_dir=Path("scenes/kitchen/geniesim"),
)
print(export_result.to_dict())

# 2) Run local data collection if Genie Sim is available
status = check_geniesim_availability()
if status["available"]:
    framework = GenieSimLocalFramework()
    result = framework.run_data_collection(
        task_config={"name": "open_drawer", "suggested_tasks": []},
        scene_config={"usd_path": "scenes/kitchen/usd/scene.usda"},
    )
    print(result)
```

