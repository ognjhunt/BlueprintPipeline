# GPU Cost Estimation

BlueprintPipeline includes a lightweight GPU-hours estimator in
`tools/cost_tracking/estimate.py` for quick planning before a local run.

## Usage

```bash
python -m tools.cost_tracking.estimate --scene-dir ./scenes/my_scene
```

Optional overrides:

```bash
python -m tools.cost_tracking.estimate \
  --scene-dir ./scenes/my_scene \
  --steps regen3d,simready,replicator \
  --config ./cost_estimate.json
```

Or from the local runner:

```bash
python tools/run_local_pipeline.py \
  --scene-dir ./scenes/my_scene \
  --estimate-costs \
  --estimate-config ./cost_estimate.json
```

## Configuration

The estimator reads JSON config files with optional `rates`, `steps`, and a
`default_instance_type`. Example:

```json
{
  "default_instance_type": "g5.xlarge",
  "rates": {
    "g5.xlarge": {"hourly_rate": 1.006, "gpu_count": 1},
    "g5.12xlarge": {"hourly_rate": 4.384, "gpu_count": 4}
  },
  "steps": {
    "regen3d": {"duration_minutes": 45, "instance_type": "g5.xlarge"},
    "dwm-inference": {"duration_hours": 2, "instance_type": "g5.12xlarge"}
  }
}
```

Notes:
* `hourly_rate` is the instance-hour price in USD for the selected instance
  type (not per GPU).
* `gpu_count` represents the number of GPUs attached to that instance type.
* Step durations can be expressed with `duration_minutes` or `duration_hours`.

## Estimation assumptions

* **GPU-hours** are computed as: `duration_hours × gpu_count`.
* **Cost** is computed as: `duration_hours × hourly_rate`.
* Steps missing duration or instance data are reported as “missing” so you can
  decide whether to add them to the config.
* The default rate table in `tools/cost_tracking/estimate.py` is meant as a
  starting point—update it to reflect your actual cloud provider pricing.
