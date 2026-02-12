#!/usr/bin/env bash
# =============================================================================
# Two-Tier SceneSmith Pipeline
# =============================================================================
#
# Tier 1 (GCP L4 — cheap):
#   SceneSmith layout + furniture placement + object retrieval
#   → Produces scene layout with object list
#
# Tier 2 (AWS L40S — expensive, on-demand):
#   SAM3D/HunyuanAI geometry generation for objects that couldn't be retrieved
#   → Produces GLB meshes for each generated object
#
# Back to Tier 1 (GCP L4):
#   Asset integration, catalog publish, embedding queue, replication
#
# Usage:
#   bash scripts/run-two-tier-scenesmith.sh --prompt "A modern kitchen with ..."
#
# This script coordinates between GCP and AWS automatically.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load configs
if [[ -f "${REPO_ROOT}/configs/text_source.env" ]]; then
  set -a; source "${REPO_ROOT}/configs/text_source.env"; set +a
fi
if [[ -f "${REPO_ROOT}/configs/aws_credentials.env" ]]; then
  set -a; source "${REPO_ROOT}/configs/aws_credentials.env"; set +a
fi

# --- Args ---
PROMPT=""
SCENE_ID=""
QUALITY_TIER="premium"
SKIP_AWS="${SKIP_AWS_STAGE:-false}"
GCP_VM_NAME="${TEXT_GEN_VM_NAME:-isaac-sim-ubuntu}"
GCP_VM_ZONE="${TEXT_GEN_VM_ZONE:-us-east1-c}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)        PROMPT="$2"; shift 2 ;;
    --scene-id)      SCENE_ID="$2"; shift 2 ;;
    --quality-tier)  QUALITY_TIER="$2"; shift 2 ;;
    --skip-aws)      SKIP_AWS=true; shift ;;
    --gcp-vm)        GCP_VM_NAME="$2"; shift 2 ;;
    --gcp-zone)      GCP_VM_ZONE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${PROMPT}" ]]; then
  echo "ERROR: --prompt is required" >&2
  echo "  Example: --prompt 'A modern kitchen with marble countertops'" >&2
  exit 1
fi

if [[ -z "${SCENE_ID}" ]]; then
  SCENE_ID="scene_$(date +%Y%m%d_%H%M%S)"
fi

WORK_DIR="/tmp/two-tier-scenesmith/${SCENE_ID}"
mkdir -p "${WORK_DIR}"

echo "============================================="
echo "  Two-Tier SceneSmith Pipeline"
echo "============================================="
echo "  Scene ID:      ${SCENE_ID}"
echo "  Quality:       ${QUALITY_TIER}"
echo "  Prompt:        ${PROMPT:0:80}..."
echo "  Work dir:      ${WORK_DIR}"
echo "  Skip AWS:      ${SKIP_AWS}"
echo "============================================="
echo ""

# =========================================================================
# TIER 1: GCP L4 — Scene Layout + Object Placement + Retrieval
# =========================================================================
echo "=== TIER 1: GCP L4 — Layout & Placement ==="

# Build the request payload
cat > "${WORK_DIR}/scene_request.json" << REQEOF
{
  "scene_id": "${SCENE_ID}",
  "prompt": $(python3 -c "import json; print(json.dumps('${PROMPT}'))"),
  "quality_tier": "${QUALITY_TIER}",
  "seed": 1,
  "constraints": {
    "object_density": "high"
  },
  "provider_policy": "openai_primary"
}
REQEOF

echo "[Tier1] Sending request to SceneSmith service..."

SCENESMITH_URL="${SCENESMITH_SERVER_URL:-http://127.0.0.1:8081/v1/generate}"

# Check if SceneSmith service is reachable
if curl -sf "${SCENESMITH_URL%/v1/generate}/health" > /dev/null 2>&1 || \
   curl -sf "${SCENESMITH_URL}" --max-time 5 > /dev/null 2>&1; then
  echo "[Tier1] SceneSmith service reachable at ${SCENESMITH_URL}"
else
  echo "[Tier1] SceneSmith service not reachable. Attempting GCP VM start..."
  gcloud compute instances start "${GCP_VM_NAME}" --zone="${GCP_VM_ZONE}" 2>/dev/null || true
  echo "[Tier1] Waiting for VM SSH..."
  for i in $(seq 1 30); do
    if gcloud compute ssh "${GCP_VM_NAME}" --zone="${GCP_VM_ZONE}" -- "echo ready" 2>/dev/null; then
      break
    fi
    sleep 10
  done
  # Assume SceneSmith starts via systemd on the VM
  echo "[Tier1] VM is up. Waiting 30s for services to initialize..."
  sleep 30
fi

# Call SceneSmith — layout + placement only (no geometry generation)
TIER1_RESPONSE=$(curl -sf -X POST "${SCENESMITH_URL}" \
  -H "Content-Type: application/json" \
  -d @"${WORK_DIR}/scene_request.json" \
  --max-time 1800) || {
  echo "ERROR: SceneSmith Tier 1 failed." >&2
  echo "  Check: ${SCENESMITH_URL}" >&2
  exit 1
}

echo "${TIER1_RESPONSE}" > "${WORK_DIR}/tier1_response.json"
OBJECT_COUNT=$(echo "${TIER1_RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('package',d).get('objects',[])))" 2>/dev/null || echo "?")
echo "[Tier1] Got ${OBJECT_COUNT} objects from SceneSmith."

# Extract objects that need geometry generation (no retrieved asset)
python3 - << 'EXTRACT_EOF' "${WORK_DIR}/tier1_response.json" "${WORK_DIR}/objects_needing_generation.json"
import json
import sys

response_path = sys.argv[1]
output_path = sys.argv[2]

with open(response_path) as f:
    data = json.load(f)

package = data.get("package", data)
objects = package.get("objects", [])

# Objects that need generation: no existing asset, not thin-covering, not placeholder
needs_gen = []
for obj in objects:
    strategy = obj.get("asset_strategy", "")
    source = obj.get("source_backend", "")
    # If asset_strategy is "retrieved" but no actual asset file exists, it needs generation
    # For now, include all objects — the L40S worker will skip those with existing assets
    needs_gen.append({
        "id": obj.get("id", ""),
        "name": obj.get("name", ""),
        "category": obj.get("category", ""),
        "description": obj.get("description", ""),
        "sim_role": obj.get("sim_role", ""),
        "dimensions_est": obj.get("dimensions_est", {}),
    })

with open(output_path, "w") as f:
    json.dump(needs_gen, f, indent=2)

print(f"  {len(needs_gen)} objects need geometry generation")
EXTRACT_EOF

NEEDS_GEN=$(python3 -c "import json; print(len(json.load(open('${WORK_DIR}/objects_needing_generation.json'))))")

if [[ "${NEEDS_GEN}" == "0" || "${SKIP_AWS}" == "true" ]]; then
  if [[ "${SKIP_AWS}" == "true" ]]; then
    echo "[Tier1] --skip-aws flag set, skipping AWS L40S stage."
  else
    echo "[Tier1] All objects retrieved from library. Skipping AWS stage."
  fi
else
  # =========================================================================
  # TIER 2: AWS L40S — SAM3D Geometry Generation
  # =========================================================================
  echo ""
  echo "=== TIER 2: AWS L40S — Geometry Generation ==="
  echo "  ${NEEDS_GEN} objects need 3D mesh generation"

  bash "${SCRIPT_DIR}/aws-l40s-geometry-stage.sh" \
    --objects-manifest "${WORK_DIR}/objects_needing_generation.json" \
    --output-dir "${WORK_DIR}/generated_assets"

  echo "[Tier2] Geometry generation complete."

  # Show what was generated
  if [[ -f "${WORK_DIR}/generated_assets/generation_summary.json" ]]; then
    python3 -c "
import json
s = json.load(open('${WORK_DIR}/generated_assets/generation_summary.json'))
print(f'  Generated: {s[\"completed\"]}/{s[\"total\"]} objects')
if s['failed'] > 0:
    print(f'  Failed:    {s[\"failed\"]} objects')
"
  fi
fi

# =========================================================================
# TIER 1 RESUME: GCP L4 — Asset Integration
# =========================================================================
echo ""
echo "=== TIER 1 RESUME: Asset Integration ==="
echo "[Tier1] Merging generated assets back into scene package..."

# Merge generated assets into the tier1 response
python3 - << 'MERGE_EOF' "${WORK_DIR}/tier1_response.json" "${WORK_DIR}/generated_assets" "${WORK_DIR}/final_package.json"
import json
import sys
from pathlib import Path

response_path = sys.argv[1]
gen_dir = Path(sys.argv[2])
output_path = sys.argv[3]

with open(response_path) as f:
    data = json.load(f)

package = data.get("package", data)
objects = package.get("objects", [])

# Index generated assets by object ID
generated = {}
if gen_dir.exists():
    for meta_file in gen_dir.glob("*/metadata.json"):
        with open(meta_file) as f:
            meta = json.load(f)
        generated[meta["id"]] = meta

# Update objects with generation metadata
for obj in objects:
    obj_id = obj.get("id", "")
    if obj_id in generated:
        gen = generated[obj_id]
        obj["asset_strategy"] = "generated"
        obj["generation_metadata"] = {
            "provider": gen.get("provider", "unknown"),
            "source_kind": "generated_external",
            "glb_size_bytes": gen.get("glb_size_bytes", 0),
            "tier": "aws_l40s",
        }

package["objects"] = objects
package["two_tier"] = {
    "enabled": True,
    "tier1_backend": "gcp_l4",
    "tier2_backend": "aws_l40s",
    "objects_generated": len(generated),
    "objects_total": len(objects),
}

with open(output_path, "w") as f:
    json.dump(package if "package" not in data else {"package": package}, f, indent=2)

print(f"  Final package: {len(objects)} objects ({len(generated)} generated on L40S)")
MERGE_EOF

echo ""
echo "============================================="
echo "  Two-Tier Pipeline Complete"
echo "============================================="
echo "  Scene:     ${SCENE_ID}"
echo "  Output:    ${WORK_DIR}/final_package.json"
echo "  Assets:    ${WORK_DIR}/generated_assets/"
echo "============================================="
