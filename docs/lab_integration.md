# Lab integration data contract

This document captures the storage layout, dataset formats, and integration points needed to
consume Genie Sim episodes in downstream lab pipelines.

## Dataset directory structure

Generated data is organized under `scenes/{scene_id}/` with separate roots for Genie Sim exports
and episode bundles. The canonical paths are defined in `tools/storage_layout/paths.py` and include
`geniesim/` plus the `episodes/` tree used by downstream training and quality gates.【F:tools/storage_layout/paths.py†L41-L84】

```
scenes/{scene_id}/
├── geniesim/
│   ├── scene_graph.json
│   ├── asset_index.json
│   ├── task_config.json
│   ├── scene_config.yaml
│   └── export_manifest.json
└── episodes/
    ├── meta/
    │   ├── info.json
    │   ├── tasks.jsonl
    │   └── episodes.jsonl
    ├── data/
    │   └── chunk-*/episode_*.parquet
    └── videos/
        └── chunk-*/observation.images.*/*.mp4
```

Notes:
- Genie Sim export artifacts are anchored at `scenes/{scene_id}/geniesim/` and include the export
  manifest for provenance and checksums.【F:tools/storage_layout/paths.py†L61-L73】【F:tools/geniesim_adapter/exporter.py†L300-L367】
- Episode bundles are written under `scenes/{scene_id}/episodes/` with metadata, data, and optional
  videos organized per LeRobot conventions.【F:tools/storage_layout/paths.py†L74-L84】【F:episode-generation-job/lerobot_exporter.py†L20-L57】

## File formats and schemas

### `episodes.jsonl`

A newline-delimited JSON index of episode metadata written alongside `dataset_info.json`.
The import job builds this by serializing each entry from `dataset_info["episodes"]` and writing
one JSON object per line.【F:genie-sim-import-job/import_from_geniesim.py†L710-L772】

Each line includes fields like:
- `episode_id` (string)
- `episode_index` (integer)
- `num_frames` (integer)
- `duration_seconds` (number)
- `quality_score` (number)
- `validation_passed` (boolean)
- `file` (string; `episode_*.parquet` path)

The field requirements for episode entries mirror the dataset info schema used for local Genie Sim
imports.【F:fixtures/contracts/geniesim_local_dataset_info.schema.json†L17-L62】

### `episode_*.parquet`

Parquet files store the per-step (frame) data for each episode. The import job writes
`episode_000000.parquet`, `episode_000001.parquet`, etc., during LeRobot conversion and records the
file name back into `dataset_info.json` and `episodes.jsonl`.【F:genie-sim-import-job/import_from_geniesim.py†L718-L746】【F:genie-sim-import-job/import_from_geniesim.py†L869-L881】

### `dataset_info.json`

Dataset summary and schema metadata for LeRobot conversions. The structure is validated against the
`geniesim_local_dataset_info.schema.json` contract and includes dataset-level metrics plus the full
`episodes` array used by `episodes.jsonl`.【F:genie-sim-import-job/import_from_geniesim.py†L167-L225】【F:fixtures/contracts/geniesim_local_dataset_info.schema.json†L1-L71】

Key fields include:
- `dataset_type`, `format_version`, `schema_version`
- `scene_id`, `job_id`, `run_id`, `pipeline_commit`, `export_schema_version`
- `episodes` (array of episode metadata)
- `total_episodes`, `total_frames`, `average_quality_score`, `min_quality_score`, `max_quality_score`
- `skipped_episodes`, `skip_rate_percent`, `conversion_failures`

### `import_manifest.json`

The import job writes a machine-readable manifest that captures checksums, quality summary, LeRobot
conversion status, and provenance for the imported bundle.【F:genie-sim-import-job/import_from_geniesim.py†L1147-L1370】

The manifest schema is defined in `import_manifest_utils.py` and includes:
- `schema_version`, `generated_at`, `output_dir`, `gcs_output_path`
- `readme_path`, `checksums_path`, `file_inventory`
- `episodes` summary (`downloaded`, `passed_validation`, `filtered`)
- `quality` thresholds and summary
- `lerobot` conversion metrics
- `checksums` + `verification` blocks
- `provenance` snapshot for pipeline and environment lineage【F:genie-sim-import-job/import_manifest_utils.py†L8-L58】

## LeRobot integration

The export path to LeRobot format is handled by `episode-generation-job/lerobot_exporter.py`, which
writes the canonical LeRobot directory structure with `meta/`, `data/`, `videos/`, and
`ground_truth/` (when available).【F:episode-generation-job/lerobot_exporter.py†L20-L57】

The import job `genie-sim-import-job/import_from_geniesim.py` performs conversion to LeRobot format
and writes `dataset_info.json`, `episodes.jsonl`, and `episode_*.parquet` for downstream training
jobs.【F:genie-sim-import-job/import_from_geniesim.py†L718-L772】【F:genie-sim-import-job/import_from_geniesim.py†L869-L921】

Minimal load example (LeRobot dataset):

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Point at the LeRobot dataset directory (e.g., scenes/{scene_id}/episodes/lerobot)
lerobot_dataset = LeRobotDataset("/path/to/scenes/{scene_id}/episodes/lerobot")
print(lerobot_dataset[0].keys())
```

Minimal load example (Parquet episodes):

```python
import pyarrow.parquet as pq

episode = pq.read_table("/path/to/scenes/{scene_id}/episodes/lerobot/episode_000000.parquet")
print(episode.schema)
```

## Quality metrics

Quality gates read thresholds from `tools/quality_gates/quality_config.json` and enforce them in
`tools/quality_gates/quality_gate.py`. Episode-focused metrics include:
- `collision_free_rate_min`
- `quality_score_min`
- `quality_pass_rate_min`
- `min_episodes_required`
- `min_average_quality_score` (SLI for average quality)

These values are applied in the episode quality gate and SLI checks to compute pass/fail decisions
and alerts for generated episodes.【F:tools/quality_gates/quality_config.json†L10-L55】【F:tools/quality_gates/quality_gate.py†L1549-L1660】

## Provenance chain

Provenance is captured at both export and import stages:

1. **Genie Sim export** writes `geniesim/export_manifest.json` with export metadata, checksums, and
   file inventory. The manifest includes `export_info` for timestamp/source pipeline and records
   an `asset_provenance_path` when available.【F:tools/geniesim_adapter/exporter.py†L370-L458】
2. **Genie Sim import** writes `import_manifest.json` containing a `provenance` block with source
   control, pipeline version, environment snapshot, and job metadata, plus references to
   `asset_provenance_path` in the bundle.【F:genie-sim-import-job/import_from_geniesim.py†L1249-L1364】
3. **Checksums** are embedded in both manifests and include a self-checksum entry computed from a
   canonical JSON representation to preserve lineage integrity.【F:tools/geniesim_adapter/exporter.py†L432-L536】【F:genie-sim-import-job/import_manifest_utils.py†L24-L33】

