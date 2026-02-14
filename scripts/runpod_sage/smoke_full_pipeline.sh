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

bash "${WORKSPACE}/BlueprintPipeline/scripts/runpod_sage/run_full_pipeline.sh"

