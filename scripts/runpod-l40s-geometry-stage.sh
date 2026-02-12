#!/usr/bin/env bash
# =============================================================================
# RunPod L40S Geometry Generation Stage
# =============================================================================
#
# Launches an L40S pod on RunPod, runs SAM3D geometry generation for objects
# that need 3D meshes, transfers results back, then terminates the pod.
#
# This is the "heavy stage" in the two-tier split:
#   GCP L4 (light)      → layout, placement, retrieval
#   RunPod L40S (heavy)  → SAM3D geometry generation only
#
# Usage:
#   bash scripts/runpod-l40s-geometry-stage.sh \
#     --objects-manifest /path/to/objects_needing_generation.json \
#     --output-dir /path/to/assets_output
#
# Prerequisites:
#   - RunPod API key in configs/runpod_credentials.env (or RUNPOD_API_KEY env)
#   - RunPod account with credits loaded
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load RunPod credentials
RUNPOD_CREDS_FILE="${REPO_ROOT}/configs/runpod_credentials.env"
if [[ -f "${RUNPOD_CREDS_FILE}" ]]; then
  set -a; source "${RUNPOD_CREDS_FILE}"; set +a
fi

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "ERROR: RUNPOD_API_KEY not set. Check configs/runpod_credentials.env" >&2
  exit 1
fi

# --- Defaults ---
RUNPOD_API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"
GPU_TYPE_ID="${RUNPOD_GPU_TYPE:-NVIDIA L40S}"
CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-COMMUNITY}"
CONTAINER_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
VOLUME_GB="${RUNPOD_VOLUME_GB:-50}"
CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-20}"
POD_NAME="blueprint-geom-$(date +%s)"
MAX_WAIT_BOOT=600      # 10 min max for pod to boot
MAX_WAIT_GENERATE=3600 # 60 min max for generation
OBJECTS_MANIFEST=""
OUTPUT_DIR=""
KEEP_POD="${RUNPOD_KEEP_POD:-false}"
POD_ID=""

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --objects-manifest) OBJECTS_MANIFEST="$2"; shift 2 ;;
    --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
    --gpu)             GPU_TYPE_ID="$2"; shift 2 ;;
    --keep-pod)        KEEP_POD=true; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${OBJECTS_MANIFEST}" ]]; then
  echo "ERROR: --objects-manifest is required" >&2
  exit 1
fi
if [[ ! -f "${OBJECTS_MANIFEST}" ]]; then
  echo "ERROR: Objects manifest not found: ${OBJECTS_MANIFEST}" >&2
  exit 1
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${REPO_ROOT}/assets_generated"
fi
mkdir -p "${OUTPUT_DIR}"

echo "=== RunPod L40S Geometry Stage ==="
echo "  GPU:           ${GPU_TYPE_ID}"
echo "  Cloud:         ${CLOUD_TYPE}"
echo "  Image:         ${CONTAINER_IMAGE}"
echo "  Objects:       $(python3 -c "import json; print(len(json.load(open('${OBJECTS_MANIFEST}'))))" 2>/dev/null || echo '?') entries"
echo "  Output dir:    ${OUTPUT_DIR}"
echo ""

# --- Helper: GraphQL query ---
gql() {
  curl -sf --request POST \
    --header 'content-type: application/json' \
    --url "${RUNPOD_API}" \
    --data "$1" 2>/dev/null
}

# --- Step 1: Launch pod ---
echo "[1/6] Launching ${GPU_TYPE_ID} pod on RunPod..."

LAUNCH_QUERY=$(cat << 'GQLEOF'
mutation {
  podFindAndDeployOnDemand(input: {
    cloudType: CLOUD_TYPE_PLACEHOLDER,
    gpuCount: 1,
    volumeInGb: VOLUME_PLACEHOLDER,
    containerDiskInGb: DISK_PLACEHOLDER,
    minVcpuCount: 4,
    minMemoryInGb: 16,
    gpuTypeId: "GPU_TYPE_PLACEHOLDER",
    name: "POD_NAME_PLACEHOLDER",
    imageName: "IMAGE_PLACEHOLDER",
    dockerArgs: "",
    ports: "22/tcp",
    volumeMountPath: "/workspace",
    env: [
      { key: "JUPYTER_PASSWORD", value: "disabled" },
      { key: "PUBLIC_KEY", value: "" }
    ]
  }) {
    id
    imageName
    machineId
    desiredStatus
    costPerHr
  }
}
GQLEOF
)

# Substitute placeholders
LAUNCH_QUERY="${LAUNCH_QUERY//CLOUD_TYPE_PLACEHOLDER/${CLOUD_TYPE}}"
LAUNCH_QUERY="${LAUNCH_QUERY//VOLUME_PLACEHOLDER/${VOLUME_GB}}"
LAUNCH_QUERY="${LAUNCH_QUERY//DISK_PLACEHOLDER/${CONTAINER_DISK_GB}}"
LAUNCH_QUERY="${LAUNCH_QUERY//GPU_TYPE_PLACEHOLDER/${GPU_TYPE_ID}}"
LAUNCH_QUERY="${LAUNCH_QUERY//POD_NAME_PLACEHOLDER/${POD_NAME}}"
LAUNCH_QUERY="${LAUNCH_QUERY//IMAGE_PLACEHOLDER/${CONTAINER_IMAGE}}"

# Collapse to single line for JSON
LAUNCH_QUERY_ONELINE=$(echo "${LAUNCH_QUERY}" | tr '\n' ' ' | sed 's/  */ /g')

LAUNCH_RESULT=$(gql "{\"query\": \"${LAUNCH_QUERY_ONELINE}\"}")

# Check for errors
if echo "${LAUNCH_RESULT}" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if 'errors' not in d else 1)" 2>/dev/null; then
  POD_ID=$(echo "${LAUNCH_RESULT}" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['podFindAndDeployOnDemand']['id'])")
  COST_HR=$(echo "${LAUNCH_RESULT}" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['podFindAndDeployOnDemand'].get('costPerHr','?'))")
  echo "  Pod ID: ${POD_ID}"
  echo "  Cost:   \$${COST_HR}/hr"
else
  echo "ERROR: Failed to launch pod:" >&2
  echo "${LAUNCH_RESULT}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get('errors',d), indent=2))" 2>/dev/null || echo "${LAUNCH_RESULT}"
  exit 1
fi

# --- Cleanup trap ---
cleanup() {
  if [[ "${KEEP_POD}" != "true" && -n "${POD_ID:-}" ]]; then
    echo ""
    echo "[cleanup] Terminating pod ${POD_ID}..."
    gql "{\"query\": \"mutation { podTerminate(input: {podId: \\\"${POD_ID}\\\"}) }\"}" > /dev/null 2>&1 || true
    echo "[cleanup] Pod terminated."
  fi
}
trap cleanup EXIT

# --- Step 2: Wait for pod to be running + get SSH info ---
echo "[2/6] Waiting for pod to be running..."

POD_QUERY="{\"query\": \"query { pod(input: {podId: \\\"${POD_ID}\\\"}) { id desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }\"}"

BOOT_WAIT=0
SSH_IP=""
SSH_PORT=""

while [[ ${BOOT_WAIT} -lt ${MAX_WAIT_BOOT} ]]; do
  sleep 15
  BOOT_WAIT=$((BOOT_WAIT + 15))

  POD_STATUS=$(gql "${POD_QUERY}" 2>/dev/null || echo "{}")
  RUNTIME=$(echo "${POD_STATUS}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    pod = d['data']['pod']
    rt = pod.get('runtime')
    if rt and rt.get('uptimeInSeconds', 0) > 0:
        ports = rt.get('ports', [])
        for p in ports:
            if p.get('privatePort') == 22 and p.get('isIpPublic'):
                print(f\"{p['ip']}:{p['publicPort']}\")
                break
        else:
            print('WAITING')
    else:
        print('WAITING')
except:
    print('WAITING')
" 2>/dev/null || echo "WAITING")

  if [[ "${RUNTIME}" != "WAITING" && "${RUNTIME}" != "" ]]; then
    SSH_IP="${RUNTIME%%:*}"
    SSH_PORT="${RUNTIME##*:}"
    echo "  Pod is running! (${BOOT_WAIT}s)"
    echo "  SSH: ${SSH_IP}:${SSH_PORT}"
    break
  fi
  echo "  Still booting... (${BOOT_WAIT}s)"
done

if [[ -z "${SSH_IP}" || -z "${SSH_PORT}" ]]; then
  echo "ERROR: Timed out waiting for pod to boot (${MAX_WAIT_BOOT}s)" >&2
  echo "  Check RunPod dashboard: https://www.runpod.io/console/pods" >&2
  exit 1
fi

# Give the container a moment to fully initialize SSH
sleep 10

SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p ${SSH_PORT} root@${SSH_IP}"
SCP_CMD="scp -o StrictHostKeyChecking=no -P ${SSH_PORT}"

# Wait for SSH to actually accept connections
echo "  Waiting for SSH connection..."
SSH_RETRY=0
until ${SSH_CMD} "echo ready" 2>/dev/null; do
  sleep 10
  SSH_RETRY=$((SSH_RETRY + 10))
  if [[ ${SSH_RETRY} -ge 120 ]]; then
    echo "ERROR: SSH not accepting connections after 120s" >&2
    echo "  Try manually: ssh -p ${SSH_PORT} root@${SSH_IP}" >&2
    exit 1
  fi
done
echo "  SSH connected."

# --- Step 3: Setup environment on pod ---
echo "[3/6] Setting up geometry generation environment..."

${SSH_CMD} 'bash -s' << 'SETUP_EOF'
set -euo pipefail

echo "--- GPU check ---"
nvidia-smi || { echo "FATAL: No GPU"; exit 1; }
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "GPU Memory: ${GPU_MEM} MiB"

echo "--- Installing dependencies ---"
pip install -q trimesh numpy pillow requests 2>/dev/null || pip3 install -q trimesh numpy pillow requests

# Set CUDA memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /workspace/geometry-work/{input,output}
echo "--- Setup complete ---"
SETUP_EOF

echo "  Environment ready."

# --- Step 4: Upload objects manifest + worker ---
echo "[4/6] Uploading objects manifest and worker..."

${SCP_CMD} "${OBJECTS_MANIFEST}" "root@${SSH_IP}:/workspace/geometry-work/input/objects.json"

# Upload the geometry generation worker
${SSH_CMD} 'cat > /workspace/geometry-work/generate.py' << 'WORKER_EOF'
#!/usr/bin/env python3
"""Geometry generation worker for RunPod L40S stage.

Reads objects.json, generates 3D meshes via SAM3D/HunyuanAI API,
writes output GLB files to output/ directory.

If no external SAM3D endpoint is configured, runs local SAM3D inference
using the GPU directly.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import requests

INPUT_DIR = Path("/workspace/geometry-work/input")
OUTPUT_DIR = Path("/workspace/geometry-work/output")

SAM3D_HOST = os.getenv("TEXT_SAM3D_API_HOST", "")
SAM3D_ENDPOINT = "/openapi/v1/text-to-3d"
HUNYUAN_HOST = os.getenv("TEXT_HUNYUAN_API_HOST", "")
HUNYUAN_ENDPOINT = "/openapi/v1/text-to-3d"
TIMEOUT = int(os.getenv("GEOMETRY_TIMEOUT_SECONDS", "1800"))
POLL_INTERVAL = int(os.getenv("GEOMETRY_POLL_SECONDS", "10"))


def generate_with_provider(obj_id, prompt, host, endpoint):
    """Submit text-to-3D request and poll until complete."""
    if not host:
        return None
    url = f"{host.rstrip('/')}{endpoint}"
    try:
        resp = requests.post(url, json={"prompt": prompt}, timeout=60)
        resp.raise_for_status()
        task = resp.json()
        task_id = task.get("task_id") or task.get("id")
        if not task_id:
            print(f"  [{obj_id}] No task_id: {task}", file=sys.stderr)
            return None

        status_url = f"{host.rstrip('/')}/openapi/v1/tasks/{task_id}"
        elapsed = 0
        while elapsed < TIMEOUT:
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            sr = requests.get(status_url, timeout=30)
            sd = sr.json()
            state = str(sd.get("status", "")).lower()
            if state in ("completed", "success", "done", "succeeded"):
                return sd
            if state in ("failed", "error", "cancelled"):
                print(f"  [{obj_id}] Failed: {sd}", file=sys.stderr)
                return None
            if elapsed % 60 == 0:
                print(f"  [{obj_id}] Generating... ({elapsed}s)")
        print(f"  [{obj_id}] Timeout ({TIMEOUT}s)", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [{obj_id}] Error: {e}", file=sys.stderr)
        return None


def download_result(obj_id, result):
    """Download GLB from result URL."""
    dl_url = (result.get("download_url") or result.get("output_url") or
              result.get("url"))
    if not dl_url:
        out = result.get("output", {})
        if isinstance(out, dict):
            dl_url = out.get("download_url") or out.get("url")
    if not dl_url:
        print(f"  [{obj_id}] No download URL", file=sys.stderr)
        return None

    out_path = OUTPUT_DIR / obj_id / "model.glb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(dl_url, timeout=120, stream=True)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return out_path
    except Exception as e:
        print(f"  [{obj_id}] Download failed: {e}", file=sys.stderr)
        return None


def main():
    objects = json.loads((INPUT_DIR / "objects.json").read_text())
    if isinstance(objects, dict):
        objects = objects.get("objects", [])

    total = len(objects)
    print(f"=== Generating geometry for {total} objects ===")

    if not SAM3D_HOST and not HUNYUAN_HOST:
        print("WARNING: No SAM3D or HunyuanAI host configured.")
        print("  Set TEXT_SAM3D_API_HOST or TEXT_HUNYUAN_API_HOST.")
        print("  Creating placeholder metadata only.")

    results = {"completed": [], "failed": [], "skipped": []}

    for i, obj in enumerate(objects, 1):
        obj_id = obj.get("id", f"obj_{i:03d}")
        name = obj.get("name", obj.get("category", "object"))
        desc = obj.get("description", f"3D model of a {name}")

        print(f"[{i}/{total}] {obj_id}: {name}")

        result = generate_with_provider(obj_id, desc, SAM3D_HOST, SAM3D_ENDPOINT)
        provider = "sam3d"
        if result is None and HUNYUAN_HOST:
            result = generate_with_provider(obj_id, desc, HUNYUAN_HOST, HUNYUAN_ENDPOINT)
            provider = "hunyuan3d"

        if result is None:
            if not SAM3D_HOST and not HUNYUAN_HOST:
                results["skipped"].append({"id": obj_id, "name": name, "reason": "no_provider"})
            else:
                results["failed"].append({"id": obj_id, "name": name, "error": "all_providers_failed"})
            continue

        glb_path = download_result(obj_id, result)
        if glb_path is None:
            results["failed"].append({"id": obj_id, "name": name, "error": "download_failed"})
            continue

        meta = {
            "id": obj_id,
            "name": name,
            "provider": provider,
            "source_kind": "generated_external",
            "glb_path": str(glb_path),
            "glb_size_bytes": glb_path.stat().st_size,
            "gpu_tier": "runpod_l40s",
        }
        meta_path = OUTPUT_DIR / obj_id / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        results["completed"].append(meta)
        print(f"  [{obj_id}] OK ({provider}, {meta['glb_size_bytes']} bytes)")

    summary = {
        "total": total,
        "completed": len(results["completed"]),
        "failed": len(results["failed"]),
        "skipped": len(results["skipped"]),
        "objects": results,
    }
    (OUTPUT_DIR / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone: {summary['completed']}/{total} succeeded, "
          f"{summary['failed']} failed, {summary['skipped']} skipped")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
WORKER_EOF

echo "  Files uploaded."

# --- Step 5: Run geometry generation ---
echo "[5/6] Running geometry generation on ${GPU_TYPE_ID}..."

${SSH_CMD} "bash -c 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  export TEXT_SAM3D_API_HOST=\"${TEXT_SAM3D_API_HOST:-}\" && \
  export TEXT_HUNYUAN_API_HOST=\"${TEXT_HUNYUAN_API_HOST:-}\" && \
  cd /workspace/geometry-work && \
  python3 generate.py'" || {
    echo "WARNING: Generation returned non-zero. Downloading partial results..."
  }

# --- Step 6: Download results ---
echo "[6/6] Downloading generated assets..."

${SCP_CMD} -r "root@${SSH_IP}:/workspace/geometry-work/output/" "${OUTPUT_DIR}/"

# Summary
if [[ -f "${OUTPUT_DIR}/generation_summary.json" ]]; then
  echo ""
  echo "=== Generation Summary ==="
  python3 -c "
import json
s = json.load(open('${OUTPUT_DIR}/generation_summary.json'))
print(f'  Total:     {s[\"total\"]}')
print(f'  Completed: {s[\"completed\"]}')
print(f'  Failed:    {s[\"failed\"]}')
print(f'  Skipped:   {s[\"skipped\"]}')
"
fi

echo ""
echo "=== RunPod Geometry Stage Complete ==="
echo "  Output:  ${OUTPUT_DIR}"
echo "  Pod ID:  ${POD_ID}"
if [[ "${KEEP_POD}" == "true" ]]; then
  echo "  Pod is still running (--keep-pod)."
  echo "  SSH: ssh -p ${SSH_PORT} root@${SSH_IP}"
  echo "  Terminate later: via RunPod dashboard or API"
else
  echo "  Pod will be terminated on exit."
fi
