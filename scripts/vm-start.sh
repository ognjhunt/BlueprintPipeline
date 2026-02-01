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
#   3. Polls gRPC port 50051 until the server is accepting connections
#   4. Prints READY when done
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTAINER_NAME="geniesim-server"
GRPC_PORT_HEX="C383"  # 50051 in hex
TIMEOUT_S=${VM_START_TIMEOUT_S:-120}

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

# ---- Step 2: Wait for gRPC server readiness ----
echo "[vm-start] Waiting for gRPC server on port 50051 (timeout: ${TIMEOUT_S}s)..."

_elapsed=0
_interval=5

while [ "${_elapsed}" -lt "${TIMEOUT_S}" ]; do
  if sudo docker exec "${CONTAINER_NAME}" bash -c "cat /proc/net/tcp6 2>/dev/null | grep -q ${GRPC_PORT_HEX}" 2>/dev/null; then
    echo "[vm-start] READY - gRPC server is listening on port 50051 (${_elapsed}s elapsed)"
    exit 0
  fi
  sleep "${_interval}"
  _elapsed=$((_elapsed + _interval))
  echo "[vm-start]   ...still waiting (${_elapsed}s / ${TIMEOUT_S}s)"
done

echo "[vm-start] ERROR: Server did not become ready within ${TIMEOUT_S}s" >&2
echo "[vm-start] Check logs: sudo docker logs ${CONTAINER_NAME} --tail 50" >&2
exit 1
