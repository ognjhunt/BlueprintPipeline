# Genie Sim Import Job

## Updating Python dependencies

The Docker image installs dependencies from `requirements.lock` for reproducible builds. When you change `requirements.txt`, update `tools/requirements-pins.txt` (the single source of truth) and run `python tools/sync_requirements_pins.py` before regenerating the lockfile with a clean virtual environment:

```bash
python -m venv /tmp/genie-sim-import-job-lock
source /tmp/genie-sim-import-job-lock/bin/activate
pip install --upgrade pip
pip install -r genie-sim-import-job/requirements.txt
pip freeze > genie-sim-import-job/requirements.lock
```

Review the updated `requirements.lock`, then rebuild the image to verify `pip check` passes.

## Quality threshold tuning

The minimum quality score for imports is defined in `quality_config.json`. You can override it at runtime with the `MIN_QUALITY_SCORE` environment variable as long as the value stays within the allowed range in that file (currently `0.5` to `0.95`). Update the JSON (and commit it) to formally change the default, and audit the effective threshold in job logs where the configured range and selected value are printed at startup.

## LeRobot enforcement defaults

LeRobot conversion failures are enforced by default in production or service mode. When `REQUIRE_LEROBOT` is unset, the import job defaults to requiring LeRobot conversion if `production_mode` or service mode is active. Set `REQUIRE_LEROBOT=false` to opt out in development workflows, and check the startup logs/import manifest for the resolved default that was applied.

## Episode bundle layout

The import job expects a bundled episode layout that mirrors the directory checks in `import_from_geniesim.py` (see `_resolve_recordings_dir`, `lerobot_dir = output_dir / "lerobot"`, and `_resolve_lerobot_info_path`) so operators can map errors back to the code paths that enforce them.

### Required directories

* `recordings/` containing `episode_*.json` (and any per-episode artifacts produced by Genie Sim).
* `lerobot/` containing the LeRobot export when conversion is required:
  * `dataset_info.json` (always required when LeRobot is enforced).
  * `episodes.jsonl` (LeRobot v3 metadata index).
  * `meta/episode_index.json` (LeRobot v3 index) **or** `episode_*.parquet` files for legacy v2 runs.
* Optional `videos/` directories can live under `lerobot/` and may include robot-specific subdirectories (for example, `lerobot/videos/spot/episode_0001.mp4`) when multi-robot runs are exported.

### Minimum artifacts for a successful import

* **Recordings-only import:** `recordings/episode_*.json` must exist and resolve via `_resolve_recordings_dir`.
* **LeRobot-required import:** `recordings/episode_*.json` **plus** `lerobot/dataset_info.json` must exist. The importer will then resolve additional metadata via `_resolve_lerobot_info_path` (v3 `meta/episode_index.json` or v2 `episode_*.parquet`) and load `episodes.jsonl` when present.

### Example trees

Single-robot run (bundle root):

```
episodes/
  geniesim_<job_id>/
    recordings/
      episode_0001.json
      episode_0002.json
    lerobot/
      dataset_info.json
      episodes.jsonl
      meta/
        episode_index.json
      videos/
        episode_0001.mp4
        episode_0002.mp4
```

Multi-robot run (bundle root):

```
episodes/
  geniesim_<job_id>/
    recordings/
      episode_0001.json
    lerobot/
      dataset_info.json
      episodes.jsonl
      meta/
        episode_index.json
      videos/
        spot/
          episode_0001.mp4
        stretch/
          episode_0001.mp4
```
