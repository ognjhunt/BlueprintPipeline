#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# Ensure the Genie Sim server container is running and ready on the VM.
#
# Usage (after SSH into VM):
#   cd ~/BlueprintPipeline && bash scripts/vm-start.sh
#
# What it does:
#   1. Checks if geniesim-server container is already running
#   2. If not, starts it via docker compose
#   3. Polls Docker health status until the server's gRPC port is ready
#   4. Prints READY when done
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTAINER_NAME="geniesim-server"
TIMEOUT_S=${VM_START_TIMEOUT_S:-120}

# ---- Pre-step: Ensure host Xorg display exists for camera RGB ----
if [ "${VM_SKIP_XORG_BOOTSTRAP:-0}" != "1" ]; then
  echo "[vm-start] Ensuring host Xorg display is ready..."
  "${SCRIPT_DIR}/vm-start-xorg.sh"
fi

# Compose defaults to :99; keep env explicit for docker compose -E usage.
export DISPLAY="${DISPLAY:-:99}"

grpc_listening() {
  sudo docker exec "${CONTAINER_NAME}" bash -lc \
    'grep -qi C383 /proc/net/tcp /proc/net/tcp6 2>/dev/null'
}

# ---- Step 1: Check if container is already running ----
if sudo docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "[vm-start] Container '${CONTAINER_NAME}' is already running."
else
  # Try to start an existing stopped container first
  if sudo docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[vm-start] Starting existing container '${CONTAINER_NAME}'..."
    sudo docker start "${CONTAINER_NAME}"
  else
    echo "[vm-start] No existing container found. Running docker compose up..."
    cd "${REPO_ROOT}"
    sudo -E docker compose -f docker-compose.geniesim-server.yaml up -d
  fi
fi

# ---- Step 2: Wait for gRPC server readiness via Docker health status ----
echo "[vm-start] Waiting for server readiness (timeout: ${TIMEOUT_S}s)..."

_elapsed=0
_interval=5

while [ "${_elapsed}" -lt "${TIMEOUT_S}" ]; do
  _health=$(sudo docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${CONTAINER_NAME}" 2>/dev/null || echo "missing")

  case "${_health}" in
    healthy)
      echo "[vm-start] READY - server is healthy (${_elapsed}s elapsed)"
      exit 0
      ;;
    unhealthy)
      # Some images report unhealthy due a broken healthcheck command even
      # while gRPC is actually ready. Treat active port 50051 as authoritative.
      if grpc_listening; then
        echo "[vm-start] READY - healthcheck reports unhealthy but gRPC is listening on 50051 (${_elapsed}s elapsed)"
        exit 0
      fi
      ;;
    none)
      # No healthcheck configured — fall back to TCP port check
      if grpc_listening; then
        echo "[vm-start] READY - gRPC server is listening on port 50051 (${_elapsed}s elapsed)"
        exit 0
      fi
      ;;
  esac

  sleep "${_interval}"
  _elapsed=$((_elapsed + _interval))
  echo "[vm-start]   ...still waiting (${_elapsed}s / ${TIMEOUT_S}s) [status: ${_health}]"
done

echo "[vm-start] ERROR: Server did not become ready within ${TIMEOUT_S}s" >&2
echo "[vm-start] Check logs: sudo docker logs ${CONTAINER_NAME} --tail 50" >&2
exit 1
