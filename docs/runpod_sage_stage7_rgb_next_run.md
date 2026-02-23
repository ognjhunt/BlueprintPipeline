# Stage 7 RGB Next-Run Runbook

## What Changed

Before:
- Stage 7 used one chosen display mode (`headless`/`windowed`) and then ran full collection.
- If mode choice was wrong for RGB, full Stage 7 produced black frames and failed late.

Now:
- Stage 7 supports `streaming` as a first-class mode.
- New policy `SAGE_STAGE7_RGB_POLICY=auto_probe_fail` runs a short probe across candidate modes before full collection.
- The probe writes `stage7_mode_probe.json`, selects the first passing mode, updates `plan_bundle.json`, then runs full Stage 7.
- Existing strict RGB contract enforcement is unchanged for the final run.

## One-Command Next Run (Strict RGB Required)

```bash
cd /workspace/BlueprintPipeline
SAGE_REQUIRE_VALID_RGB=1 \
SAGE_STAGE7_RGB_POLICY=auto_probe_fail \
SAGE_STAGE7_MODE_ORDER=auto \
bash scripts/runpod_sage/run_full_pipeline.sh
```

## Key Environment Variables

| Variable | Default | Meaning |
|---|---|---|
| `SAGE_STAGE7_RGB_POLICY` | `auto_probe_fail` | `auto_probe_fail` probes candidate modes and aborts if none pass. `legacy_direct` skips probe and runs directly. |
| `SAGE_STAGE7_MODE_ORDER` | `auto` | Candidate order for probe. `auto` picks `windowed,streaming,headless` when display exists, else `streaming,headless`. |
| `SAGE_STAGE7_PROBE_DEMOS` | `1` | Number of demos per probe attempt. |
| `SAGE_STAGE7_PROBE_TIMEOUT_S` | `600` | Timeout per probe attempt in seconds. |
| `SAGE_STAGE7_STREAMING_ENABLED` | `1` | Enables/disables `streaming` candidate mode. |
| `SAGE_STAGE7_STREAMING_PORT` | `49100` | WebRTC/streaming port used in streaming mode setup. |
| `SAGE_STAGE7_PROBE_KEEP_ARTIFACTS` | `1` | Keep probe per-mode outputs on disk. |

## New Probe Artifact

Path:
- `${LAYOUT_DIR}/quality/stage7_mode_probe.json`

Contains:
- requested mode and resolved mode order
- probe config (`probe_demos`, timeout)
- per-mode attempt records
- return code, timeout, contract issues, quality summary
- selected winner mode (if any)

## How to Read `stage7_mode_probe.json`

Important fields:
- `status`: `pass` or `fail`
- `selected_mode`: winner mode for full run (`windowed`/`streaming`/`headless`)
- `attempts[].pass`: whether that mode passed probe contract
- `attempts[].failure_reasons`: concise reason list
- `attempts[].quality.rgb_std_min`: should be > `0` for valid RGB
- `attempts[].quality.exported_videos`: should meet probe demo count

Typical failure signatures:
- `rgb_std_min=0.0`: black RGB
- `missing_videos=...` or `exported_videos=...`: video export failed/empty
- `timeout>...`: attempt hung past timeout

## Recovery If All Modes Fail

1. Increase probe timeout:
```bash
export SAGE_STAGE7_PROBE_TIMEOUT_S=1200
```
2. Force explicit candidate order:
```bash
export SAGE_STAGE7_MODE_ORDER=windowed,streaming,headless
```
3. If running with real desktop display, force windowed:
```bash
export SAGE_STAGE7_HEADLESS_MODE=windowed
export SAGE_STAGE7_RGB_POLICY=legacy_direct
```
4. If display is virtual/Xvfb only, avoid forcing windowed and keep probe policy.

## Manual Probe CLI (Diagnostics Only)

```bash
python scripts/runpod_sage/stage7_rgb_mode_probe.py \
  --plan-bundle /workspace/SAGE/server/results/<layout_id>/plans/plan_bundle.json \
  --collector /workspace/BlueprintPipeline/scripts/runpod_sage/isaacsim_collect_mobile_franka.py \
  --isaacsim-py /workspace/isaacsim_env/bin/python3 \
  --output-root /workspace/SAGE/server/results/<layout_id>/stage7_probe_manual \
  --mode-order auto \
  --probe-demos 1 \
  --timeout-s 600
```

## Cost Control

After validation/collection, stop billable GPU resources immediately:
- Stop TensorDock/Vast/RunPod instance once artifacts are downloaded.

