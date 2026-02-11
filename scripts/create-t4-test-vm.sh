#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# Create a T4 GPU VM for testing Isaac Sim RGB rendering (Stage 4).
#
# This creates a NEW VM separate from the existing L4 VM (isaac-sim-ubuntu).
# The T4 has RT cores (Turing) and is well-tested with Isaac Sim 5.1.0.
#
# Usage:
#   bash scripts/create-t4-test-vm.sh
#
# After creation, run:
#   bash scripts/setup-t4-vm.sh
#
# Cost: ~$0.95/hr (n1-standard-8 + T4). STOP when not in use.
# =============================================================================

VM_NAME="${T4_VM_NAME:-geniesim-t4-test}"
VM_ZONE="${T4_VM_ZONE:-asia-east1-c}"
MACHINE_TYPE="${T4_MACHINE_TYPE:-n1-standard-8}"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="200GB"
BOOT_DISK_TYPE="pd-ssd"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"

echo "============================================================"
echo " Creating T4 GPU VM for RGB Rendering Test"
echo "============================================================"
echo "VM Name:      ${VM_NAME}"
echo "Zone:         ${VM_ZONE}"
echo "Machine:      ${MACHINE_TYPE}"
echo "GPU:          ${GPU_TYPE} x${GPU_COUNT}"
echo "Disk:         ${BOOT_DISK_SIZE} ${BOOT_DISK_TYPE}"
echo "Est. cost:    ~\$0.95/hr"
echo "============================================================"

# Check if VM already exists
if gcloud compute instances describe "${VM_NAME}" --zone="${VM_ZONE}" &>/dev/null; then
  echo "[create-t4] VM '${VM_NAME}' already exists in ${VM_ZONE}."
  echo "[create-t4] Current status:"
  gcloud compute instances describe "${VM_NAME}" --zone="${VM_ZONE}" --format="value(status)"
  echo ""
  echo "To delete and recreate: gcloud compute instances delete ${VM_NAME} --zone=${VM_ZONE}"
  exit 0
fi

# Check GPU quota
echo "[create-t4] Checking GPU quota in ${VM_ZONE}..."
_region="${VM_ZONE%-*}"
_quota=$(gcloud compute regions describe "${_region}" \
  --format="json(quotas)" 2>/dev/null | \
  python3 -c "
import json, sys
data = json.load(sys.stdin)
for q in data.get('quotas', []):
    if q.get('metric') == 'NVIDIA_T4_GPUS':
        avail = q.get('limit', 0) - q.get('usage', 0)
        print(f'T4 quota: {avail} available (limit={q[\"limit\"]}, used={q[\"usage\"]})')
        sys.exit(0 if avail >= 1 else 1)
print('T4 quota not found in region — may need to request quota')
sys.exit(1)
" 2>&1) || true
echo "[create-t4] ${_quota}"

# Create the VM
echo "[create-t4] Creating VM..."
gcloud compute instances create "${VM_NAME}" \
  --zone="${VM_ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
  --maintenance-policy=TERMINATE \
  --boot-disk-size="${BOOT_DISK_SIZE}" \
  --boot-disk-type="${BOOT_DISK_TYPE}" \
  --image-family="${IMAGE_FAMILY}" \
  --image-project="${IMAGE_PROJECT}" \
  --scopes=default,storage-ro

echo ""
echo "============================================================"
echo " VM Created Successfully"
echo "============================================================"
echo "VM Name: ${VM_NAME}"
echo "Zone:    ${VM_ZONE}"
echo ""
echo "Next steps:"
echo "  1. Wait ~60s for VM to boot"
echo "  2. Run: bash scripts/setup-t4-vm.sh"
echo "  3. After setup: bash scripts/run-t4-rgb-test.sh"
echo ""
echo "To stop (save money):  gcloud compute instances stop ${VM_NAME} --zone=${VM_ZONE}"
echo "To delete:             gcloud compute instances delete ${VM_NAME} --zone=${VM_ZONE}"
echo "============================================================"
