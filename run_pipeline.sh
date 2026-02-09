#!/bin/bash
# run_pipeline.sh — Single source of truth for running the BlueprintPipeline.
# Run on the VM host (not inside the container).
# Sources configs/realism_strict.env for all strictness/timeout settings.
set -euo pipefail

cd "$(dirname "$0")"

# ── Single-run guard ──
if pgrep -af 'tools/run_local_pipeline.py' >/dev/null 2>&1; then
  echo "ERROR: pipeline already running" >&2
  exit 1
fi

# ── Source strict env config ──
if [ -f configs/realism_strict.env ]; then
  set -a
  # shellcheck disable=SC1091
  source configs/realism_strict.env
  set +a
  echo "[run_pipeline] Loaded configs/realism_strict.env"
else
  echo "ERROR: configs/realism_strict.env not found" >&2
  exit 1
fi

# ── Fixed env vars (not in .env file) ──
export PYTHONPATH="${HOME}/BlueprintPipeline:${HOME}/BlueprintPipeline/episode-generation-job:${PYTHONPATH:-}"
export GENIESIM_HOST=localhost
export GENIESIM_PORT=50051
export GENIESIM_SKIP_DEFAULT_LIGHTING=1
unset SKIP_QUALITY_GATES 2>/dev/null || true

# ── gRPC readiness check (use clean PYTHONPATH to avoid import conflicts) ──
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

# ── Run ──
LOG="/tmp/pipeline_strict.log"
echo "[run_pipeline] Starting pipeline (log: $LOG)"
nohup python3 tools/run_local_pipeline.py \
  --scene-dir ./test_scenes/scenes/lightwheel_kitchen \
  --steps genie-sim-submit \
  --force-rerun genie-sim-submit \
  --use-geniesim --fail-fast \
  > "$LOG" 2>&1 &
PID=$!
echo "$PID" > /tmp/pipeline_run.pid
echo "[run_pipeline] PID: $PID"
echo "[run_pipeline] Monitor: tail -f $LOG"
