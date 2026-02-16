#!/usr/bin/env bash
set -euo pipefail

# Minimal in-container smoke test:
# - 1 pose-aug variant
# - 1 demo
# - strict mode on (no placeholders)

WORKSPACE="${WORKSPACE:-/workspace}"

# shellcheck disable=SC1091
source "${WORKSPACE}/miniconda3/etc/profile.d/conda.sh"
conda activate sage

export ROOM_TYPE="${ROOM_TYPE:-kitchen}"
export ROBOT_TYPE="${ROBOT_TYPE:-mobile_franka}"
export TASK_DESC="${TASK_DESC:-Pick up the mug from the counter and place it on the table}"

export NUM_POSE_SAMPLES=1
export NUM_DEMOS=1
export STRICT_PIPELINE=1
export STRICT_ARTIFACT_CONTRACT=1
export STRICT_PROVENANCE=1
export AUTO_FIX_LAYOUT=1
export SAGE_STRICT_SENSORS=1
export SAGE_SENSOR_MIN_RGB_STD="${SAGE_SENSOR_MIN_RGB_STD:-5.0}"
export SAGE_MIN_DEPTH_FINITE_RATIO="${SAGE_MIN_DEPTH_FINITE_RATIO:-0.98}"
export SAGE_MAX_RGB_SATURATION_RATIO="${SAGE_MAX_RGB_SATURATION_RATIO:-0.85}"
export SAGE_EXPORT_SCENE_USD=1
export SAGE_EXPORT_DEMO_VIDEOS=1

bash "${WORKSPACE}/BlueprintPipeline/scripts/runpod_sage/run_full_pipeline.sh"
