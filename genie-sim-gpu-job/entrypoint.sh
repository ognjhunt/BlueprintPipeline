#!/usr/bin/env bash
set -euo pipefail

PREALLOCATE_MB="${GENIESIM_GPU_PREALLOCATE_MB:-}"

if [[ -n "${PREALLOCATE_MB}" && "${PREALLOCATE_MB}" != "0" ]]; then
  /isaac-sim/python.sh /app/genie-sim-gpu-job/gpu_warmup.py
else
  echo "[entrypoint] GENIESIM_GPU_PREALLOCATE_MB not set or zero; skipping GPU warmup."
fi

exec /isaac-sim/python.sh /app/genie-sim-submit-job/submit_to_geniesim.py
