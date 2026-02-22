# Stub Boundaries & Contracts

This document defines the **explicit stub boundaries** used by BlueprintPipeline’s
free/default smoke tests. It captures the minimum file layout, schemas, and required
fields for (a) **Stage 1 text generation inputs** and (b) **Genie Sim gRPC responses** so that
tests can run without external services.

## 1) Stage 1 Output Stub (Fixtures)

**Boundary:** `scenes/{scene_id}/` Stage 1 outputs are treated as an external
dependency for smoke tests. The fixture `fixtures/generate_mock_stage1.py`
generates the canonical outputs consumed by downstream jobs.
Canonical request schema lives in `fixtures/contracts/`:
- `scene_request_v1.schema.json`

### Required Directory Layout

```
scenes/{scene_id}/
├── textgen/
│   ├── package.json
│   ├── request.normalized.json
│   └── .textgen_complete
├── assets/
│   ├── scene_manifest.json
│   ├── .stage1_complete
│   └── objects/{object_id}/mesh.glb
├── layout/
│   └── scene_layout_scaled.json
└── seg/
    └── inventory.json
```

### Minimal Required Fields

**`assets/.stage1_complete`**
```json
{
  "scene_id": "string",
  "status": "completed",
  "marker_type": "stage1_complete",
  "timestamp": "ISO-8601 timestamp"
}
```

**`assets/scene_manifest.json`**
```json
{
  "scene_id": "string",
  "scene": {},
  "objects": [
    {
      "id": "string",
      "category": "string",
      "asset": {"path": "assets/objects/<id>/mesh.glb"}
    }
  ]
}
```

**Minimum Contract:** The pipeline requires `assets/scene_manifest.json`,
`layout/scene_layout_scaled.json`, `seg/inventory.json`, and
`assets/.stage1_complete`.

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
