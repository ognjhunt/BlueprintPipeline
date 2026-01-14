# Stub Boundaries & Contracts

This document defines the **explicit stub boundaries** used by BlueprintPipeline’s
free/default smoke tests. It captures the minimum file layout, schemas, and required
fields for (a) **3D‑RE‑GEN inputs** and (b) **Genie Sim gRPC responses** so that
tests can run without external services.

## 1) 3D‑RE‑GEN Input Stub (Fixtures)

**Boundary:** `scenes/{scene_id}/regen3d/` is produced by 3D‑RE‑GEN and treated as
an external dependency. The fixture `fixtures/generate_mock_regen3d.py` generates
the stubbed outputs consumed by the pipeline.
Canonical JSON schemas live in `fixtures/contracts/`:
- `regen3d_scene_info.schema.json`
- `regen3d_object_pose.schema.json`
- `regen3d_object_bounds.schema.json`
- `regen3d_object_material.schema.json`
- `regen3d_camera_intrinsics.schema.json`
- `regen3d_camera_extrinsics.schema.json`

### Required Directory Layout

```
scenes/{scene_id}/regen3d/
├── scene_info.json
├── objects/
│   └── {object_id}/
│       ├── mesh.glb
│       ├── pose.json
│       ├── bounds.json
│       └── material.json
├── background/
│   ├── mesh.glb
│   ├── pose.json
│   └── bounds.json
└── camera/
    ├── intrinsics.json
    └── extrinsics.json
```

### Minimal Schemas (Required Fields)

**`scene_info.json`**
```json
{
  "scene_id": "string",
  "image_size": [width, height],
  "coordinate_frame": "y_up",
  "meters_per_unit": 1.0,
  "confidence": 0.0,
  "version": "1.0",
  "environment_type": "kitchen|office|warehouse",
  "reconstruction_method": "3d-re-gen",
  "generated_at": "ISO-8601 timestamp"
}
```

**`objects/{object_id}/pose.json`**
```json
{
  "transform_matrix": [[4x4 numbers]],
  "translation": [x, y, z],
  "rotation_quaternion": [w, x, y, z],
  "scale": [sx, sy, sz],
  "confidence": 0.0,
  "is_floor_contact": true
}
```

**`objects/{object_id}/bounds.json`**
```json
{
  "min": [x, y, z],
  "max": [x, y, z],
  "center": [x, y, z],
  "size": [sx, sy, sz]
}
```

**`objects/{object_id}/material.json`**
```json
{
  "base_color": [r, g, b],
  "metallic": 0.0,
  "roughness": 0.5,
  "material_type": "generic"
}
```

**`background/pose.json`** and **`background/bounds.json`** follow the same schema as
object pose/bounds. The background `mesh.glb` can be any valid GLB.

**`camera/intrinsics.json`**
```json
{
  "matrix": [[3x3 numbers]],
  "width": 1920,
  "height": 1080
}
```

**`camera/extrinsics.json`**
```json
{ "matrix": [[4x4 numbers]] }
```

**Minimum Contract:** The pipeline requires `scene_info.json`, per-object `mesh.glb`,
`pose.json`, `bounds.json`, and the `objects/` directory. The background and camera
files are required by the tests but optional for downstream adapters if not used.

---

## 2) Genie Sim gRPC Response Stubs (Local Framework)

**Boundary:** The local Genie Sim framework is invoked via gRPC. When running
in smoke/mock mode, stubbed responses must satisfy the minimal response fields
consumed by `tools/geniesim_adapter/local_framework.py`.

The canonical proto schema is defined in
`tools/geniesim_adapter/geniesim_grpc.proto`. These are the **minimum response
fields** required by the pipeline code paths:

### Required Response Contracts

| RPC | Required Fields | Notes |
| --- | --- | --- |
| `GetObservation` | `success`, `robot_state`, `scene_state`, `timestamp` | `robot_state` should include `joint_state.positions` and `joint_state.names` at minimum. `scene_state.objects` can be empty. |
| `SetJointPosition` | `success`, `current_state` | `current_state.positions` should match the requested joint count. |
| `GetJointPosition` | `success`, `joint_state` | Used for verification/polling in local framework. |
| `StartRecording` | `success`, `recording_path` | `recording_path` is persisted into local episode metadata. |
| `StopRecording` | `success`, `frames_recorded`, `duration_seconds`, `recording_path` | Fields are serialized into episode metadata. |
| `Reset` | `success` | Error message optional. |
| `SendCommand` | `success`, `payload` (JSON bytes) | `payload` may be empty for commands not used in tests. |

### Minimal Field Shapes

**`RobotState`**
```json
{
  "joint_state": {
    "positions": [float],
    "names": ["string"]
  }
}
```

**`SceneState`**
```json
{
  "objects": [],
  "simulation_time": 0.0,
  "step_count": 0
}
```

**`StartRecordingResponse`**
```json
{
  "success": true,
  "recording_path": "/tmp/geniesim_recordings/episode_000000.json"
}
```

**`StopRecordingResponse`**
```json
{
  "success": true,
  "frames_recorded": 30,
  "duration_seconds": 1.0,
  "recording_path": "/tmp/geniesim_recordings/episode_000000.json"
}
```

**Minimum Contract:** The pipeline only asserts `success` and consumes the
fields listed above. Any additional proto fields may be omitted or left defaulted
by stub implementations.

---

## 3) Genie Sim Local Output Contracts (Episodes + Metadata)

**Boundary:** When Genie Sim runs locally, episode recordings and metadata are
written to disk for import and validation. Mock generators and schema tests
validate this interface without requiring Isaac Sim or Particulate.

### Required Directory Layout

```
geniesim_local/{run_id}/
├── recordings/
│   └── episode_000000.json
└── metadata/
    ├── dataset_info.json
    └── episodes.jsonl
```

### Canonical Schemas

- `fixtures/contracts/geniesim_local_episode.schema.json`
- `fixtures/contracts/geniesim_local_dataset_info.schema.json`
- `fixtures/contracts/geniesim_local_episodes_index.schema.json`

**Minimum Contract:** Episode JSONs must include `episode_id`, `task_name`,
`frames`, `frame_count`, `quality_score`, and `validation_passed`. Metadata
must include dataset summary plus a JSONL index entry per episode.
