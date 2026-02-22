#!/bin/bash
# run_pipeline.sh — Single source of truth for running the BlueprintPipeline.
# Run on the VM host (not inside the container).
# Sources configs/realism_strict.env for all strictness/timeout settings.
#
# Usage:
#   bash run_pipeline.sh                                   # default: lightwheel_kitchen
#   bash run_pipeline.sh --scene-dir ./test_scenes/scenes/bedroom
#   bash run_pipeline.sh --env-file configs/realism_rgb_test.env
set -euo pipefail

cd "$(dirname "$0")"

# ── Parse arguments ──
SCENE_DIR="./test_scenes/scenes/lightwheel_kitchen"
ENV_FILE="configs/realism_strict.env"
DATA_BACKEND="auto"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --scene-dir)
      SCENE_DIR="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --backend)
      DATA_BACKEND="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# ── Single-run guard ──
if pgrep -af 'tools/run_local_pipeline.py' >/dev/null 2>&1; then
  echo "ERROR: pipeline already running" >&2
  exit 1
fi

# ── Source env config ──
if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  echo "[run_pipeline] Loaded ${ENV_FILE}"
else
  echo "ERROR: ${ENV_FILE} not found" >&2
  exit 1
fi

# ── Fixed env vars (not in .env file) ──
export PYTHONPATH="${HOME}/BlueprintPipeline:${HOME}/BlueprintPipeline/episode-generation-job:${PYTHONPATH:-}"
export GENIESIM_HOST=localhost
export GENIESIM_PORT=50051
export GENIESIM_SKIP_DEFAULT_LIGHTING=1
unset SKIP_QUALITY_GATES 2>/dev/null || true

# ── Resolve backend mode ──
if [[ "${DATA_BACKEND}" == "auto" ]]; then
  _use_geniesim_raw="$(printf '%s' "${USE_GENIESIM:-true}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${_use_geniesim_raw}" == "false" || "${_use_geniesim_raw}" == "0" || "${_use_geniesim_raw}" == "no" || "${_use_geniesim_raw}" == "off" ]]; then
    DATA_BACKEND="episode"
  else
    DATA_BACKEND="geniesim"
  fi
fi

STEP_ARGS=()
case "${DATA_BACKEND}" in
  geniesim)
    STEP_ARGS=(--steps genie-sim-submit --force-rerun genie-sim-submit --use-geniesim)
    ;;
  episode|blueprint|isaac)
    STEP_ARGS=(--steps isaac-lab --force-rerun isaac-lab)
    ;;
  *)
    echo "ERROR: --backend must be one of auto|geniesim|episode (or blueprint|isaac), got '${DATA_BACKEND}'" >&2
    exit 1
    ;;
esac

# ── Ensure host Xorg is running for RGB capture ──
if [ "${SKIP_RGB_CAPTURE:-true}" = "false" ] && [ "${ENABLE_CAMERAS:-0}" = "1" ]; then
  if [ ! -S /tmp/.X11-unix/X99 ]; then
    echo "[run_pipeline] Starting headless Xorg on :99 for camera RGB rendering..."
    if command -v nvidia-xconfig >/dev/null 2>&1 && command -v Xorg >/dev/null 2>&1; then
      # Generate xorg.conf with NVIDIA driver for headless rendering
      sudo nvidia-xconfig --no-xinerama --use-display-device=None \
        --virtual=1280x720 -o /tmp/xorg-pipeline.conf 2>/dev/null || true
      if [ -f /tmp/xorg-pipeline.conf ]; then
        sudo Xorg :99 -config /tmp/xorg-pipeline.conf -noreset +extension GLX &>/dev/null &
        sleep 2
        if [ -S /tmp/.X11-unix/X99 ]; then
          echo "[run_pipeline] Xorg :99 started with NVIDIA GLX support"
        else
          echo "[run_pipeline] WARNING: Xorg failed to start — camera RGB may fail"
        fi
      fi
    else
      echo "[run_pipeline] WARNING: nvidia-xconfig/Xorg not available — start Xorg manually for RGB"
    fi
  else
    echo "[run_pipeline] Xorg :99 already running"
  fi
  export DISPLAY=:99
fi

# ── gRPC readiness check (use clean PYTHONPATH to avoid import conflicts) ──
if [[ "${DATA_BACKEND}" == "geniesim" ]]; then
  echo "[run_pipeline] Checking gRPC readiness on ${GENIESIM_HOST}:${GENIESIM_PORT}..."
  _grpc_ready=0
  for i in $(seq 1 30); do
    if PYTHONPATH="" python3 -c "import grpc,sys; ch=grpc.insecure_channel('${GENIESIM_HOST}:${GENIESIM_PORT}'); grpc.channel_ready_future(ch).result(timeout=2); sys.exit(0)" 2>/dev/null; then
      _grpc_ready=1
      break
    fi
    echo "  Waiting for gRPC... (attempt $i/30)"
    sleep 5
  done
  if [ "$_grpc_ready" = "0" ]; then
    echo "ERROR: gRPC not ready after 150s" >&2
    exit 1
  fi
  echo "[run_pipeline] gRPC ready"
fi

# ── Run ──
LOG="/tmp/pipeline_strict.log"
echo "[run_pipeline] Starting pipeline (log: $LOG)"
echo "[run_pipeline] Scene: ${SCENE_DIR}"
echo "[run_pipeline] Backend: ${DATA_BACKEND}"
echo "[run_pipeline] RGB: SKIP_RGB_CAPTURE=${SKIP_RGB_CAPTURE:-unset}, ENABLE_CAMERAS=${ENABLE_CAMERAS:-unset}"
nohup python3 tools/run_local_pipeline.py \
  --scene-dir "${SCENE_DIR}" \
  "${STEP_ARGS[@]}" \
  --fail-fast \
  "${EXTRA_ARGS[@]}" \
  > "$LOG" 2>&1 &
PID=$!
echo "$PID" > /tmp/pipeline_run.pid
echo "[run_pipeline] PID: $PID"
echo "[run_pipeline] Monitor: tail -f $LOG"
