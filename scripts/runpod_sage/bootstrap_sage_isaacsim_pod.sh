#!/usr/bin/env bash
set -euo pipefail

log() { echo "[sage-isaac-bootstrap $(date -u +%FT%TZ)] $*"; }

WORKSPACE=${WORKSPACE:-/workspace}
SAGE_DIR="${WORKSPACE}/SAGE"
MINICONDA_DIR="${WORKSPACE}/miniconda3"
BP_DIR="${WORKSPACE}/BlueprintPipeline"
PREFLIGHT_SCRIPT="${BP_DIR}/scripts/runpod_sage/isaacsim_runtime_preflight.sh"
SECRETS_ENV_PATH="${WORKSPACE}/.sage_runpod_secrets.env"
ISAAC_SIM_IMAGE="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}"
ISAAC_CONTAINER_NAME="${ISAAC_CONTAINER_NAME:-sage-isaacsim}"
MCP_RESOLVER="${BP_DIR}/scripts/runpod_sage/mcp_extension_paths.py"

resolve_mcp_extension_src() {
  if [[ -f "${MCP_RESOLVER}" ]]; then
    local resolved=""
    resolved="$(python3 "${MCP_RESOLVER}" --sage-dir "${SAGE_DIR}" || true)"
    if [[ -n "${resolved}" ]]; then
      echo "${resolved}"
      return 0
    fi
  fi
  local candidate
  for candidate in \
    "${SAGE_DIR}/server/isaacsim_mcp_ext/isaac.sim.mcp_extension" \
    "${SAGE_DIR}/server/isaacsim/isaac.sim.mcp_extension"; do
    if [[ -d "${candidate}" ]]; then
      if [[ "${candidate}" == "${SAGE_DIR}/server/isaacsim/isaac.sim.mcp_extension" ]]; then
        log "WARNING: Using deprecated MCP extension path: ${candidate}"
      fi
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ -f "${SECRETS_ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${SECRETS_ENV_PATH}"
  log "Loaded secrets env: ${SECRETS_ENV_PATH}"
fi

OPENAI_API_KEY_EFFECTIVE="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-}}"
if [[ -z "${OPENAI_API_KEY_EFFECTIVE}" ]]; then
  log "ERROR: OPENAI_API_KEY or OPENROUTER_API_KEY is required."
  exit 2
fi
if [[ -z "${NGC_API_KEY:-}" ]]; then
  log "ERROR: NGC_API_KEY is required to pull ${ISAAC_SIM_IMAGE}."
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  log "ERROR: SLURM_JOB_ID is required (used by SAGE MCP port hashing)."
  exit 2
fi
if [[ -z "${TRELLIS_SERVER_URL:-}" ]]; then
  log "ERROR: TRELLIS_SERVER_URL is required (public URL of Pod A / port 8080)."
  exit 2
fi

OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENROUTER_BASE_URL:-https://api.openai.com/v1}}"
OPENAI_WEBSOCKET_BASE_URL="${OPENAI_WEBSOCKET_BASE_URL:-}"
if [[ -z "${OPENAI_WEBSOCKET_BASE_URL}" && "${OPENAI_BASE_URL,,}" == *"api.openai.com"* ]]; then
  OPENAI_WEBSOCKET_BASE_URL="wss://api.openai.com/ws/v1/realtime?provider=openai"
fi
OPENAI_USE_WEBSOCKET="${OPENAI_USE_WEBSOCKET:-1}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"
OPENAI_MODEL_QWEN="${OPENAI_MODEL_QWEN:-qwen/qwen3.5-397b-a17b}"
OPENAI_MODEL_OPENAI="${OPENAI_MODEL_OPENAI:-moonshotai/kimi-k2.5}"
OPENAI_MODEL_GLMV="${OPENAI_MODEL_GLMV:-${OPENAI_MODEL}}"
OPENAI_MODEL_CLAUDE="${OPENAI_MODEL_CLAUDE:-${OPENAI_MODEL}}"
export OPENAI_API_KEY_EFFECTIVE
export OPENAI_BASE_URL
export OPENAI_WEBSOCKET_BASE_URL
export OPENAI_USE_WEBSOCKET

log "Preflight: GPU visibility"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "ERROR: nvidia-smi not found (GPU not available in this pod)."
  exit 3
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1 || true

log "Installing base packages"
apt-get update -qq
apt-get install -y -qq \
  ca-certificates curl wget git jq gnupg lsb-release \
  iproute2 psmisc libvulkan1 vulkan-tools >/dev/null

if [[ -x "${PREFLIGHT_SCRIPT}" ]]; then
  log "Running host Vulkan preflight before Isaac Sim container launch..."
  REQUIRE_LOCAL_ROBOT_ASSET="${REQUIRE_LOCAL_ROBOT_ASSET:-0}" "${PREFLIGHT_SCRIPT}"
fi

log "Installing / starting Docker (dockerd) if needed"
if ! command -v docker >/dev/null 2>&1; then
  apt-get install -y -qq docker.io >/dev/null
fi

start_dockerd() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  log "Starting dockerd..."
  mkdir -p /var/run
  # Keep Docker's image layers and build cache on the persistent /workspace volume,
  # not the (smaller) container disk. Isaac Sim pulls are large.
  DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-${WORKSPACE}/docker}"
  mkdir -p "${DOCKER_DATA_ROOT}"
  nohup dockerd --data-root "${DOCKER_DATA_ROOT}" --host=unix:///var/run/docker.sock >/tmp/dockerd.log 2>&1 &
  for _ in $(seq 1 60); do
    if docker info >/dev/null 2>&1; then
      log "dockerd is ready"
      return 0
    fi
    sleep 2
  done
  log "ERROR: dockerd did not become ready. Last 80 lines:"
  tail -n 80 /tmp/dockerd.log || true
  return 1
}

start_dockerd

log "Installing NVIDIA Container Toolkit (for --gpus all) if needed"
if ! command -v nvidia-container-cli >/dev/null 2>&1; then
  distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
    sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  apt-get update -qq
  apt-get install -y -qq nvidia-container-toolkit >/dev/null
fi

# Configure Docker runtime to use NVIDIA hooks. This may fail on some RunPod templates;
# we attempt it and continue if it errors, but GPU-in-container will fail without it.
if command -v nvidia-ctk >/dev/null 2>&1; then
  nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1 || true
fi

# Restart dockerd to pick up runtime changes
pkill dockerd >/dev/null 2>&1 || true
sleep 1
start_dockerd

log "NGC docker login + pull Isaac Sim image"
echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin >/dev/null
docker pull "${ISAAC_SIM_IMAGE}"

log "Cloning NVlabs/sage into ${SAGE_DIR} (if needed)"
if [[ ! -d "${SAGE_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/NVlabs/sage.git "${SAGE_DIR}"
fi

log "Installing Miniconda under ${MINICONDA_DIR} (if needed)"
if [[ ! -x "${MINICONDA_DIR}/bin/conda" ]]; then
  cd "${WORKSPACE}"
  wget -q -O Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3.sh -b -p "${MINICONDA_DIR}"
  rm -f Miniconda3.sh
fi

# shellcheck disable=SC1091
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"

log "Creating conda env 'sage' from SAGE client/environment.yml (if needed)"
if ! conda env list | awk '{print $1}' | grep -qx sage; then
  conda env create -n sage -f "${SAGE_DIR}/client/environment.yml"
fi

conda activate sage

log "Writing SAGE client key.json"
python3 - <<PY
import json, os, pathlib
client_key = {
  "API_TOKEN": os.environ["OPENAI_API_KEY_EFFECTIVE"],
  "API_URL_QWEN": os.environ.get("OPENAI_BASE_URL", "${OPENAI_BASE_URL}"),
  "MODEL_NAME": os.environ.get("OPENAI_MODEL_QWEN", "qwen/qwen3.5-397b-a17b"),
}
path = pathlib.Path("${SAGE_DIR}/client/key.json")
path.write_text(json.dumps(client_key, indent=4) + "\\n", encoding="utf-8")
path.chmod(0o600)
print("wrote", str(path))
PY

log "Writing SAGE server key.json"
python3 - <<PY
import json, os, pathlib
model_qwen = os.environ.get("OPENAI_MODEL_QWEN", "qwen/qwen3.5-397b-a17b")
model_openai = os.environ.get("OPENAI_MODEL_OPENAI", "moonshotai/kimi-k2.5")
model_glmv = os.environ.get("OPENAI_MODEL_GLMV", os.environ.get("OPENAI_MODEL", "${OPENAI_MODEL}"))
model_claude = os.environ.get("OPENAI_MODEL_CLAUDE", os.environ.get("OPENAI_MODEL", "${OPENAI_MODEL}"))
server_key = {
  "ANTHROPIC_API_KEY": "",
  "API_TOKEN": os.environ["OPENAI_API_KEY_EFFECTIVE"],
  "API_URL_QWEN": os.environ.get("OPENAI_BASE_URL", "${OPENAI_BASE_URL}"),
  "API_URL_OPENAI": os.environ.get("OPENAI_BASE_URL", "${OPENAI_BASE_URL}"),
  "OPENAI_WEBSOCKET_BASE_URL": os.environ.get("OPENAI_WEBSOCKET_BASE_URL", ""),
  "OPENAI_USE_WEBSOCKET": os.environ.get("OPENAI_USE_WEBSOCKET", "1"),
  "MODEL_DICT": {
    "qwen": model_qwen,
    "openai": model_openai,
    "glmv": model_glmv,
    "claude": model_claude,
  },
  "TRELLIS_SERVER_URL": os.environ["TRELLIS_SERVER_URL"],
  "FLUX_SERVER_URL": "",
}
path = pathlib.Path("${SAGE_DIR}/server/key.json")
path.write_text(json.dumps(server_key, indent=4) + "\\n", encoding="utf-8")
path.chmod(0o600)
print("wrote", str(path))
PY

log "Starting Isaac Sim container (${ISAAC_SIM_IMAGE}) with MCP extension enabled"
docker rm -f "${ISAAC_CONTAINER_NAME}" >/dev/null 2>&1 || true
mkdir -p "${WORKSPACE}/isaac-sim-cache" "${WORKSPACE}/isaac-sim-data"
MCP_EXT_SRC="$(resolve_mcp_extension_src || true)"

docker run -d --name "${ISAAC_CONTAINER_NAME}" \
  --gpus all \
  --network=host \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e SLURM_JOB_ID="${SLURM_JOB_ID}" \
  -e MCP_EXT_SRC="${MCP_EXT_SRC}" \
  -v "${SAGE_DIR}:/workspace/SAGE:rw" \
  -v "${WORKSPACE}/isaac-sim-cache:/root/.cache/ov:rw" \
  -v "${WORKSPACE}/isaac-sim-data:/root/.local/share/ov:rw" \
  "${ISAAC_SIM_IMAGE}" \
  bash -lc "set -euo pipefail; if [[ -z \"\${VK_ICD_FILENAMES:-}\" && -z \"\${VK_DRIVER_FILES:-}\" ]]; then for c in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do if [[ -f \"\${c}\" ]]; then export VK_ICD_FILENAMES=\"\${c}\"; export VK_DRIVER_FILES=\"\${c}\"; break; fi; done; fi; if [[ -n \"\${MCP_EXT_SRC:-}\" && -d \"\${MCP_EXT_SRC}\" ]]; then ln -sf \"\${MCP_EXT_SRC}\" /isaac-sim/exts/isaac.sim.mcp_extension; else echo '[bootstrap] WARN: MCP extension path missing'; fi; /isaac-sim/kit/kit /isaac-sim/apps/omni.isaac.sim.kit --no-window --enable isaac.sim.mcp_extension"

log "Waiting for MCP socket port to be listening (derived from SLURM_JOB_ID=${SLURM_JOB_ID})"
MCP_PORT="$(python3 - <<'PY'
import hashlib, os
job_id = os.environ.get("SLURM_JOB_ID")
port_start, port_end = 8080, 40000
h = int(hashlib.md5(str(job_id).encode()).hexdigest(), 16)
print(port_start + (h % (port_end - port_start + 1)))
PY
)"
log "Expected MCP port: ${MCP_PORT}"

deadline=$(( $(date +%s) + 900 ))
while [[ "$(date +%s)" -lt "${deadline}" ]]; do
  if ss -lnt 2>/dev/null | awk '{print $4}' | grep -q ":${MCP_PORT}\$"; then
    log "MCP port is listening"
    exit 0
  fi
  sleep 5
done

log "ERROR: MCP port was not listening within timeout."
log "Isaac Sim container logs (last 120 lines):"
docker logs --tail 120 "${ISAAC_CONTAINER_NAME}" || true
exit 4
