#!/usr/bin/env bash
set -euo pipefail

log() { echo "[sage-isaac-bootstrap $(date -u +%FT%TZ)] $*"; }

WORKSPACE=${WORKSPACE:-/workspace}
SAGE_DIR="${WORKSPACE}/SAGE"
MINICONDA_DIR="${WORKSPACE}/miniconda3"
SECRETS_ENV_PATH="${WORKSPACE}/.sage_runpod_secrets.env"
ISAAC_SIM_IMAGE="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}"
ISAAC_CONTAINER_NAME="${ISAAC_CONTAINER_NAME:-sage-isaacsim}"

if [[ -f "${SECRETS_ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${SECRETS_ENV_PATH}"
  log "Loaded secrets env: ${SECRETS_ENV_PATH}"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  log "ERROR: OPENAI_API_KEY is required."
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

OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"

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
  iproute2 psmisc >/dev/null

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
  "API_TOKEN": os.environ["OPENAI_API_KEY"],
  "API_URL_QWEN": os.environ.get("OPENAI_BASE_URL", "${OPENAI_BASE_URL}"),
  "MODEL_NAME": os.environ.get("OPENAI_MODEL", "${OPENAI_MODEL}"),
}
path = pathlib.Path("${SAGE_DIR}/client/key.json")
path.write_text(json.dumps(client_key, indent=4) + "\\n", encoding="utf-8")
path.chmod(0o600)
print("wrote", str(path))
PY

log "Writing SAGE server key.json"
python3 - <<PY
import json, os, pathlib
model = os.environ.get("OPENAI_MODEL", "${OPENAI_MODEL}")
server_key = {
  "ANTHROPIC_API_KEY": "",
  "API_TOKEN": os.environ["OPENAI_API_KEY"],
  "API_URL_QWEN": os.environ.get("OPENAI_BASE_URL", "${OPENAI_BASE_URL}"),
  "API_URL_OPENAI": os.environ.get("OPENAI_BASE_URL", "${OPENAI_BASE_URL}"),
  "MODEL_DICT": {
    "qwen": model,
    "openai": model,
    "glmv": model,
    "claude": model,
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

docker run -d --name "${ISAAC_CONTAINER_NAME}" \
  --gpus all \
  --network=host \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e SLURM_JOB_ID="${SLURM_JOB_ID}" \
  -v "${SAGE_DIR}:/workspace/SAGE:rw" \
  -v "${WORKSPACE}/isaac-sim-cache:/root/.cache/ov:rw" \
  -v "${WORKSPACE}/isaac-sim-data:/root/.local/share/ov:rw" \
  "${ISAAC_SIM_IMAGE}" \
  bash -lc "set -euo pipefail; ln -sf /workspace/SAGE/server/isaacsim/isaac.sim.mcp_extension /isaac-sim/exts/isaac.sim.mcp_extension; /isaac-sim/kit/kit /isaac-sim/apps/omni.isaac.sim.kit --no-window --enable isaac.sim.mcp_extension"

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
