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
