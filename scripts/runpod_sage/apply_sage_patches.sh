#!/usr/bin/env bash
# =============================================================================
# Apply SAGE patches for GPT 5.1 compatibility + import guards.
# Idempotent — safe to run multiple times.
# =============================================================================
set -euo pipefail

log() { echo "[sage-patches $(date -u +%FT%TZ)] $*"; }

SAGE_DIR="${WORKSPACE:-/workspace}/SAGE"

if [[ ! -d "${SAGE_DIR}" ]]; then
    log "ERROR: SAGE not found at ${SAGE_DIR}"
    exit 1
fi

# ── Helper: apply a sed patch only if the old pattern exists ─────────────────
patch_file() {
    local file="$1" old_pattern="$2" new_pattern="$3" desc="$4"
    if [[ ! -f "${file}" ]]; then
        log "  SKIP: ${file} not found"
        return
    fi
    if grep -qF "${old_pattern}" "${file}" 2>/dev/null; then
        sed -i "s|${old_pattern}|${new_pattern}|g" "${file}"
        log "  PATCHED: ${desc} in $(basename "${file}")"
    else
        log "  OK: ${desc} already applied in $(basename "${file}")"
    fi
}

# ── Patch 1: vlm.py — max_completion_tokens + reasoning_effort ──────────────
log "Patch 1: vlm.py — GPT 5.1 API compatibility"
VLM="${SAGE_DIR}/server/vlm.py"
if [[ -f "${VLM}" ]]; then
    # Replace max_tokens with max_completion_tokens
    if grep -q 'max_tokens=' "${VLM}" && ! grep -q 'max_completion_tokens=' "${VLM}"; then
        sed -i 's/max_tokens=/max_completion_tokens=/g' "${VLM}"
        log "  PATCHED: max_tokens → max_completion_tokens"
    else
        log "  OK: max_completion_tokens already set"
    fi

    # Add reasoning_effort if not present
    if ! grep -q 'reasoning_effort' "${VLM}"; then
        # Insert reasoning_effort after max_completion_tokens
        sed -i '/max_completion_tokens=/s/$/\n            reasoning_effort="high",/' "${VLM}" 2>/dev/null || \
            log "  WARNING: Could not auto-insert reasoning_effort (apply manually)"
    else
        log "  OK: reasoning_effort already present"
    fi
fi

# ── Patch 2: layout.py — import guards ──────────────────────────────────────
log "Patch 2: layout.py — import guards for matfuse + isaaclab"
LAYOUT="${SAGE_DIR}/server/layout.py"
if [[ -f "${LAYOUT}" ]]; then
    if grep -q '^from floor_plan_materials' "${LAYOUT}"; then
        # Wrap the import in try/except
        python3 - "${LAYOUT}" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, 'r') as f:
    content = f.read()

# Guard matfuse import
old = "from floor_plan_materials.material_generator import"
if old in content and "try:" not in content.split(old)[0][-50:]:
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if line.startswith("from floor_plan_materials"):
            new_lines.append("try:")
            new_lines.append("    " + line)
            new_lines.append("except (ImportError, FileNotFoundError, Exception):")
            new_lines.append("    pass  # matfuse not available, falls back to defaults")
        elif line.startswith("from isaaclab.correct_mobile_franka"):
            new_lines.append("try:")
            new_lines.append("    " + line)
            new_lines.append("except (ImportError, FileNotFoundError, Exception):")
            new_lines.append("    pass  # isaaclab not available")
        else:
            new_lines.append(line)
    with open(path, 'w') as f:
        f.write('\n'.join(new_lines))
    print("  PATCHED: import guards in layout.py")
else:
    print("  OK: import guards already applied in layout.py")
PYEOF
    else
        log "  OK: layout.py already patched"
    fi
fi

# ── Patch 3: layout_wo_robot.py — same import guards ────────────────────────
log "Patch 3: layout_wo_robot.py — import guards"
LAYOUT_WO="${SAGE_DIR}/server/layout_wo_robot.py"
if [[ -f "${LAYOUT_WO}" ]]; then
    if grep -q '^from floor_plan_materials' "${LAYOUT_WO}"; then
        python3 - "${LAYOUT_WO}" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, 'r') as f:
    content = f.read()

old = "from floor_plan_materials"
if old in content and "try:" not in content.split(old)[0][-50:]:
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if line.startswith("from floor_plan_materials"):
            new_lines.append("try:")
            new_lines.append("    " + line)
            new_lines.append("except (ImportError, FileNotFoundError, Exception):")
            new_lines.append("    pass")
        elif line.startswith("from isaaclab.correct_mobile_franka"):
            new_lines.append("try:")
            new_lines.append("    " + line)
            new_lines.append("except (ImportError, FileNotFoundError, Exception):")
            new_lines.append("    pass")
        else:
            new_lines.append(line)
    with open(path, 'w') as f:
        f.write('\n'.join(new_lines))
    print("  PATCHED: import guards in layout_wo_robot.py")
else:
    print("  OK: import guards already applied in layout_wo_robot.py")
PYEOF
    else
        log "  OK: layout_wo_robot.py already patched"
    fi
fi

# ── Patch 4: client_generation_room_desc.py — GPT 5.1 compat ────────────────
log "Patch 4: client_generation_room_desc.py — GPT 5.1 compatibility"
CLIENT="${SAGE_DIR}/client/client_generation_room_desc.py"
if [[ -f "${CLIENT}" ]]; then
    # max_tokens → max_completion_tokens
    if grep -q 'max_tokens=' "${CLIENT}" && ! grep -q 'max_completion_tokens=' "${CLIENT}"; then
        sed -i 's/max_tokens=/max_completion_tokens=/g' "${CLIENT}"
        log "  PATCHED: max_tokens → max_completion_tokens"
    fi

    # tool_choice: "none" → "auto"
    if grep -q '"tool_choice": "none"' "${CLIENT}" || grep -q "'tool_choice': 'none'" "${CLIENT}"; then
        sed -i "s/\"tool_choice\": \"none\"/\"tool_choice\": \"auto\"/g" "${CLIENT}"
        sed -i "s/'tool_choice': 'none'/'tool_choice': 'auto'/g" "${CLIENT}"
        log "  PATCHED: tool_choice none → auto"
    fi

    # Fix conda env name: simgen → sage
    if grep -q 'conda activate simgen' "${CLIENT}"; then
        sed -i 's/conda activate simgen/conda activate sage/g' "${CLIENT}"
        log "  PATCHED: simgen → sage"
    fi

    # Fix bash init path
    if grep -q 'source ~/.bashrc' "${CLIENT}"; then
        sed -i 's|source ~/.bashrc|source /workspace/miniconda3/etc/profile.d/conda.sh|g' "${CLIENT}"
        log "  PATCHED: bashrc → conda.sh"
    fi
fi

# ── Patch 5: Physics crash guard — handle Isaac Sim unavailable ──────────────
log "Patch 5: Physics crash guard — fix 'string indices must be integers'"
PATCH_SCRIPT="${WORKSPACE:-/workspace}/BlueprintPipeline/scripts/runpod_sage/patch_physics_crash.py"
if [[ -f "${PATCH_SCRIPT}" ]]; then
    SAGE_DIR="${SAGE_DIR}" python3 "${PATCH_SCRIPT}"
else
    log "  SKIP: patch_physics_crash.py not found at ${PATCH_SCRIPT}"
    log "  (This is normal if running outside the Docker image)"
fi

# ── Patch 6: Ensure texture baking dependencies ──────────────────────────────
log "Patch 6: Texture baking dependencies (nvdiffrast)"
if python3 -c "import nvdiffrast" 2>/dev/null; then
    log "  OK: nvdiffrast already installed"
else
    log "  Installing nvdiffrast for SAM3D texture baking..."
    pip install nvdiffrast 2>&1 | tail -5 || log "  WARNING: nvdiffrast install failed (texture baking will use vertex colors)"
fi

log "All patches applied."
