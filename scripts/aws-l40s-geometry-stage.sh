#!/usr/bin/env bash
# =============================================================================
# AWS L40S Geometry Generation Stage
# =============================================================================
#
# Launches a g6e.xlarge (L40S 48GB) spot or on-demand instance on AWS,
# runs only the GPU-heavy SAM3D / HunyuanAI geometry generation for objects
# that need 3D mesh creation, then transfers results back.
#
# This is the "heavy stage" in the two-tier split:
#   GCP L4 (light) → layout, placement, retrieval
#   AWS L40S (heavy) → SAM3D geometry generation only
#
# Usage:
#   bash scripts/aws-l40s-geometry-stage.sh \
#     --objects-manifest /path/to/objects_needing_generation.json \
#     --output-dir /path/to/assets_output
#
# Prerequisites:
#   - AWS CLI configured (or source configs/aws_credentials.env)
#   - SSH private key at ~/.ssh/blueprint-pipeline.pem
#   - GPU quota approved (L-DB2E81BA >= 4 vCPUs)
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load AWS credentials if present
AWS_CREDS_FILE="${REPO_ROOT}/configs/aws_credentials.env"
if [[ -f "${AWS_CREDS_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${AWS_CREDS_FILE}"
  set +a
fi

# --- Defaults (overridable via env or flags) ---
AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
INSTANCE_TYPE="${AWS_EC2_INSTANCE_TYPE:-g6e.xlarge}"
AMI_ID="${AWS_EC2_AMI_ID:-ami-0fe59b4f6e7e66c3e}"
SUBNET_ID="${AWS_EC2_SUBNET_ID:-subnet-08e8bc987e392bc41}"
SECURITY_GROUP_ID="${AWS_EC2_SECURITY_GROUP_ID:-sg-071f1d3800d03bd57}"
KEY_NAME="${AWS_EC2_KEY_NAME:-Blueprint Pipeline}"
SSH_KEY_PATH="${AWS_EC2_SSH_KEY_PATH:-$HOME/.ssh/blueprint-pipeline.pem}"
ROOT_VOLUME_GB="${AWS_L40S_ROOT_VOLUME_GB:-100}"
USE_SPOT="${AWS_L40S_USE_SPOT:-false}"
INSTANCE_TAG="blueprint-l40s-geometry"
MAX_WAIT_BOOT=300      # 5 min max wait for instance to boot
MAX_WAIT_SETUP=600     # 10 min max for setup script
MAX_WAIT_GENERATE=3600 # 60 min max for geometry generation
OBJECTS_MANIFEST=""
OUTPUT_DIR=""
KEEP_INSTANCE="${AWS_L40S_KEEP_INSTANCE:-false}"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --objects-manifest) OBJECTS_MANIFEST="$2"; shift 2 ;;
    --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
    --spot)            USE_SPOT=true; shift ;;
    --keep-instance)   KEEP_INSTANCE=true; shift ;;
    --instance-type)   INSTANCE_TYPE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${OBJECTS_MANIFEST}" ]]; then
  echo "ERROR: --objects-manifest is required" >&2
  echo "  This JSON file lists objects needing SAM3D geometry generation." >&2
  exit 1
fi
if [[ ! -f "${OBJECTS_MANIFEST}" ]]; then
  echo "ERROR: objects manifest not found: ${OBJECTS_MANIFEST}" >&2
  exit 1
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${REPO_ROOT}/assets_generated"
fi
mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${SSH_KEY_PATH}" ]]; then
  echo "ERROR: SSH key not found at ${SSH_KEY_PATH}" >&2
  echo "  Download your private key from the AWS Console:" >&2
  echo "  EC2 → Key pairs → 'Blueprint Pipeline' → download .pem" >&2
  echo "  Then: mv ~/Downloads/Blueprint\\ Pipeline.pem ${SSH_KEY_PATH}" >&2
  echo "         chmod 600 ${SSH_KEY_PATH}" >&2
  exit 1
fi

echo "=== AWS L40S Geometry Stage ==="
echo "  Instance type: ${INSTANCE_TYPE}"
echo "  AMI:           ${AMI_ID}"
echo "  Region:        ${AWS_REGION}"
echo "  Objects:       $(wc -l < "${OBJECTS_MANIFEST}" | tr -d ' ') entries"
echo "  Output dir:    ${OUTPUT_DIR}"
echo ""

# --- Step 1: Launch instance ---
echo "[1/6] Launching ${INSTANCE_TYPE} instance..."

LAUNCH_ARGS=(
  aws ec2 run-instances
  --region "${AWS_REGION}"
  --instance-type "${INSTANCE_TYPE}"
  --image-id "${AMI_ID}"
  --key-name "${KEY_NAME}"
  --subnet-id "${SUBNET_ID}"
  --security-group-ids "${SECURITY_GROUP_ID}"
  --associate-public-ip-address
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${ROOT_VOLUME_GB},VolumeType=gp3,DeleteOnTermination=true}"
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_TAG}},{Key=Project,Value=BlueprintPipeline},{Key=Stage,Value=geometry-generation}]"
  --count 1
)

if [[ "${USE_SPOT}" == "true" ]]; then
  LAUNCH_ARGS+=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}')
  echo "  (Using spot pricing)"
fi

LAUNCH_OUTPUT=$("${LAUNCH_ARGS[@]}" 2>&1)
INSTANCE_ID=$(echo "${LAUNCH_OUTPUT}" | python3 -c "import sys,json; print(json.load(sys.stdin)['Instances'][0]['InstanceId'])")
echo "  Instance ID: ${INSTANCE_ID}"

# --- Cleanup trap ---
cleanup() {
  if [[ "${KEEP_INSTANCE}" != "true" && -n "${INSTANCE_ID:-}" ]]; then
    echo ""
    echo "[cleanup] Terminating instance ${INSTANCE_ID}..."
    aws ec2 terminate-instances --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# --- Step 2: Wait for running + public IP ---
echo "[2/6] Waiting for instance to be running..."
aws ec2 wait instance-running --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}"
echo "  Instance is running."

PUBLIC_IP=$(aws ec2 describe-instances --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "  Public IP: ${PUBLIC_IP}"

# Wait for SSH to be ready
echo "  Waiting for SSH..."
BOOT_WAIT=0
until ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes \
  -i "${SSH_KEY_PATH}" "ubuntu@${PUBLIC_IP}" "echo ready" 2>/dev/null; do
  sleep 10
  BOOT_WAIT=$((BOOT_WAIT + 10))
  if [[ ${BOOT_WAIT} -ge ${MAX_WAIT_BOOT} ]]; then
    echo "ERROR: Timed out waiting for SSH (${MAX_WAIT_BOOT}s)" >&2
    exit 1
  fi
done
echo "  SSH is ready."

SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -i ${SSH_KEY_PATH} ubuntu@${PUBLIC_IP}"
SCP_CMD="scp -o StrictHostKeyChecking=no -i ${SSH_KEY_PATH}"

# --- Step 3: Bootstrap instance ---
echo "[3/6] Bootstrapping instance with CUDA + SAM3D dependencies..."

${SSH_CMD} 'bash -s' << 'SETUP_EOF'
set -euo pipefail

echo "--- Checking GPU ---"
nvidia-smi || { echo "FATAL: No GPU detected"; exit 1; }
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "GPU Memory: ${GPU_MEM} MiB"
if [[ "${GPU_MEM}" -lt 40000 ]]; then
  echo "WARNING: GPU has ${GPU_MEM} MiB, expected >= 48000 MiB for L40S"
fi

echo "--- Installing Python deps ---"
sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1

# Create venv for geometry generation
python3 -m venv /home/ubuntu/geom-venv
source /home/ubuntu/geom-venv/bin/activate

pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --quiet trimesh numpy pillow requests flask

# Set CUDA allocator config for reduced fragmentation
echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> /home/ubuntu/.bashrc

mkdir -p /home/ubuntu/geometry-work/{input,output}
echo "--- Setup complete ---"
SETUP_EOF

echo "  Bootstrap complete."

# --- Step 4: Upload objects manifest ---
echo "[4/6] Uploading objects manifest..."
${SCP_CMD} "${OBJECTS_MANIFEST}" "ubuntu@${PUBLIC_IP}:/home/ubuntu/geometry-work/input/objects.json"

# Also upload the geometry generation worker script
cat << 'WORKER_EOF' | ${SSH_CMD} 'cat > /home/ubuntu/geometry-work/generate.py'
#!/usr/bin/env python3
"""Geometry generation worker for AWS L40S stage.

Reads objects.json, generates 3D meshes via SAM3D/HunyuanAI,
writes output GLB files to output/ directory.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import requests

INPUT_DIR = Path("/home/ubuntu/geometry-work/input")
OUTPUT_DIR = Path("/home/ubuntu/geometry-work/output")

# SAM3D / HunyuanAI endpoints (set via env or defaults)
SAM3D_HOST = os.getenv("TEXT_SAM3D_API_HOST", "")
SAM3D_ENDPOINT = "/openapi/v1/text-to-3d"
HUNYUAN_HOST = os.getenv("TEXT_HUNYUAN_API_HOST", "")
HUNYUAN_ENDPOINT = "/openapi/v1/text-to-3d"
TIMEOUT = int(os.getenv("GEOMETRY_TIMEOUT_SECONDS", "1800"))
POLL_INTERVAL = int(os.getenv("GEOMETRY_POLL_SECONDS", "10"))


def generate_with_provider(obj_id: str, prompt: str, host: str, endpoint: str) -> dict | None:
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
            print(f"  [{obj_id}] No task_id in response: {task}", file=sys.stderr)
            return None

        # Poll for completion
        status_url = f"{host.rstrip('/')}/openapi/v1/tasks/{task_id}"
        elapsed = 0
        while elapsed < TIMEOUT:
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            status_resp = requests.get(status_url, timeout=30)
            status_data = status_resp.json()
            state = str(status_data.get("status", "")).lower()
            if state in ("completed", "success", "done"):
                return status_data
            if state in ("failed", "error", "cancelled"):
                print(f"  [{obj_id}] Generation failed: {status_data}", file=sys.stderr)
                return None
            if elapsed % 60 == 0:
                print(f"  [{obj_id}] Still generating... ({elapsed}s)", file=sys.stderr)

        print(f"  [{obj_id}] Timed out after {TIMEOUT}s", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"  [{obj_id}] Provider error: {exc}", file=sys.stderr)
        return None


def download_result(obj_id: str, result: dict) -> Path | None:
    """Download GLB from result URL."""
    download_url = result.get("download_url") or result.get("output_url") or result.get("url")
    if not download_url:
        # Check nested output
        output = result.get("output", {})
        if isinstance(output, dict):
            download_url = output.get("download_url") or output.get("url")
    if not download_url:
        print(f"  [{obj_id}] No download URL in result", file=sys.stderr)
        return None

    out_path = OUTPUT_DIR / obj_id / "model.glb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(download_url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return out_path
    except Exception as exc:
        print(f"  [{obj_id}] Download failed: {exc}", file=sys.stderr)
        return None


def main():
    objects = json.loads((INPUT_DIR / "objects.json").read_text())
    if not isinstance(objects, list):
        objects = objects.get("objects", [])

    total = len(objects)
    print(f"Generating geometry for {total} objects...")

    results = {"completed": [], "failed": [], "skipped": []}

    for i, obj in enumerate(objects, 1):
        obj_id = obj.get("id", f"obj_{i:03d}")
        name = obj.get("name", obj.get("category", "object"))
        prompt = f"3D model of a {name}"
        if obj.get("description"):
            prompt = obj["description"]

        print(f"[{i}/{total}] {obj_id}: {name}")

        # Try SAM3D first, then HunyuanAI
        result = generate_with_provider(obj_id, prompt, SAM3D_HOST, SAM3D_ENDPOINT)
        provider = "sam3d"
        if result is None and HUNYUAN_HOST:
            result = generate_with_provider(obj_id, prompt, HUNYUAN_HOST, HUNYUAN_ENDPOINT)
            provider = "hunyuan3d"

        if result is None:
            print(f"  [{obj_id}] FAILED - no provider succeeded")
            results["failed"].append({"id": obj_id, "name": name, "error": "all_providers_failed"})
            continue

        glb_path = download_result(obj_id, result)
        if glb_path is None:
            results["failed"].append({"id": obj_id, "name": name, "error": "download_failed"})
            continue

        # Write metadata
        meta = {
            "id": obj_id,
            "name": name,
            "provider": provider,
            "source_kind": "generated_external",
            "glb_path": str(glb_path),
            "glb_size_bytes": glb_path.stat().st_size,
        }
        meta_path = OUTPUT_DIR / obj_id / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        results["completed"].append(meta)
        print(f"  [{obj_id}] OK ({provider}, {meta['glb_size_bytes']} bytes)")

    # Write summary
    summary_path = OUTPUT_DIR / "generation_summary.json"
    summary = {
        "total": total,
        "completed": len(results["completed"]),
        "failed": len(results["failed"]),
        "skipped": len(results["skipped"]),
        "objects": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nDone: {summary['completed']}/{total} succeeded, {summary['failed']} failed")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
WORKER_EOF

echo "  Objects manifest and worker script uploaded."

# --- Step 5: Run geometry generation ---
echo "[5/6] Running geometry generation on L40S..."

${SSH_CMD} "bash -c 'source /home/ubuntu/geom-venv/bin/activate && \
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  export TEXT_SAM3D_API_HOST=\"${TEXT_SAM3D_API_HOST:-}\" && \
  export TEXT_HUNYUAN_API_HOST=\"${TEXT_HUNYUAN_API_HOST:-}\" && \
  cd /home/ubuntu/geometry-work && \
  python3 generate.py'" || {
    echo "WARNING: Geometry generation returned non-zero exit. Downloading partial results..."
  }

# --- Step 6: Download results ---
echo "[6/6] Downloading generated assets..."

${SCP_CMD} -r "ubuntu@${PUBLIC_IP}:/home/ubuntu/geometry-work/output/" "${OUTPUT_DIR}/"

# Show summary
if [[ -f "${OUTPUT_DIR}/generation_summary.json" ]]; then
  echo ""
  echo "=== Generation Summary ==="
  python3 -c "
import json, sys
s = json.load(open('${OUTPUT_DIR}/generation_summary.json'))
print(f\"  Total:     {s['total']}\")
print(f\"  Completed: {s['completed']}\")
print(f\"  Failed:    {s['failed']}\")
"
fi

echo ""
echo "=== L40S Geometry Stage Complete ==="
echo "  Output: ${OUTPUT_DIR}"
if [[ "${KEEP_INSTANCE}" == "true" ]]; then
  echo "  Instance ${INSTANCE_ID} is still running (--keep-instance)."
  echo "  Don't forget to terminate: aws ec2 terminate-instances --instance-ids ${INSTANCE_ID}"
else
  echo "  Instance will be terminated on exit."
fi
