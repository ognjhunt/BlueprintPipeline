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

# ── Patch 7: client_generation_robot_task.py — GPT 5.1 compat ────────────────
log "Patch 7: client_generation_robot_task.py — GPT 5.1 compatibility"
ROBOT_CLIENT="${SAGE_DIR}/client/client_generation_robot_task.py"
if [[ -f "${ROBOT_CLIENT}" ]]; then
    # max_tokens → max_completion_tokens
    if grep -q '"max_tokens":' "${ROBOT_CLIENT}" && ! grep -q '"max_completion_tokens":' "${ROBOT_CLIENT}"; then
        sed -i 's/"max_tokens":/"max_completion_tokens":/g' "${ROBOT_CLIENT}"
        log "  PATCHED: max_tokens → max_completion_tokens"
    else
        log "  OK: max_completion_tokens already set"
    fi

    # Add reasoning_effort after max_completion_tokens if not present
    if ! grep -q 'reasoning_effort' "${ROBOT_CLIENT}"; then
        sed -i '/"max_completion_tokens":/a\                    "reasoning_effort": "medium",' "${ROBOT_CLIENT}"
        log "  PATCHED: added reasoning_effort=medium"
    else
        log "  OK: reasoning_effort already present"
    fi

    # tool_choice: "none" → "auto"
    if grep -q '"tool_choice"] = "none"' "${ROBOT_CLIENT}"; then
        sed -i 's/"tool_choice"\] = "none"/"tool_choice"] = "auto"/g' "${ROBOT_CLIENT}"
        log "  PATCHED: tool_choice none → auto"
    else
        log "  OK: tool_choice already auto"
    fi

    # Fix conda env name: simgen → sage
    if grep -q 'conda activate simgen' "${ROBOT_CLIENT}"; then
        sed -i 's/conda activate simgen/conda activate sage/g' "${ROBOT_CLIENT}"
        log "  PATCHED: simgen → sage"
    fi

    # Fix bash init path
    if grep -q 'source ~/.bashrc' "${ROBOT_CLIENT}"; then
        sed -i 's|source ~/.bashrc|source /workspace/miniconda3/etc/profile.d/conda.sh|g' "${ROBOT_CLIENT}"
        log "  PATCHED: bashrc → conda.sh"
    fi
fi

# ── Patch 8: client_generation_scene_aug.py — GPT 5.1 compat ────────────────
log "Patch 8: client_generation_scene_aug.py — GPT 5.1 compatibility"
AUG_CLIENT="${SAGE_DIR}/client/client_generation_scene_aug.py"
if [[ -f "${AUG_CLIENT}" ]]; then
    # max_tokens → max_completion_tokens
    if grep -q '"max_tokens":' "${AUG_CLIENT}" && ! grep -q '"max_completion_tokens":' "${AUG_CLIENT}"; then
        sed -i 's/"max_tokens":/"max_completion_tokens":/g' "${AUG_CLIENT}"
        log "  PATCHED: max_tokens → max_completion_tokens"
    else
        log "  OK: max_completion_tokens already set"
    fi

    # Add reasoning_effort after max_completion_tokens if not present
    if ! grep -q 'reasoning_effort' "${AUG_CLIENT}"; then
        sed -i '/"max_completion_tokens":/a\                    "reasoning_effort": "medium",' "${AUG_CLIENT}"
        log "  PATCHED: added reasoning_effort=medium"
    else
        log "  OK: reasoning_effort already present"
    fi

    # tool_choice: "none" → "auto"
    if grep -q '"tool_choice"\] = "none"' "${AUG_CLIENT}"; then
        sed -i 's/"tool_choice"\] = "none"/"tool_choice"] = "auto"/g' "${AUG_CLIENT}"
        log "  PATCHED: tool_choice none → auto"
    else
        log "  OK: tool_choice already auto"
    fi

    # Fix conda env name: simgen → sage
    if grep -q 'conda activate simgen' "${AUG_CLIENT}"; then
        sed -i 's/conda activate simgen/conda activate sage/g' "${AUG_CLIENT}"
        log "  PATCHED: simgen → sage"
    fi

    # Fix bash init path
    if grep -q 'source ~/.bashrc' "${AUG_CLIENT}"; then
        sed -i 's|source ~/.bashrc|source /workspace/miniconda3/etc/profile.d/conda.sh|g' "${AUG_CLIENT}"
        log "  PATCHED: bashrc → conda.sh"
    fi
fi

# ── Patch 9: Ensure isaaclab fallback modules are symlinked ──────────────────
log "Patch 9: Isaac Lab fallback modules"
BP_FALLBACK="${WORKSPACE:-/workspace}/BlueprintPipeline/scripts/runpod_sage/isaaclab_fallback"
ISAACLAB_DIR="${SAGE_DIR}/server/isaaclab"
if [[ -d "${BP_FALLBACK}" ]]; then
    # Create isaaclab package dir if it doesn't exist
    mkdir -p "${ISAACLAB_DIR}"
    # Create __init__.py if missing
    if [[ ! -f "${ISAACLAB_DIR}/__init__.py" ]]; then
        echo "# Isaac Lab fallback package" > "${ISAACLAB_DIR}/__init__.py"
        log "  CREATED: ${ISAACLAB_DIR}/__init__.py"
    fi
    # Symlink fallback modules (only if the real module doesn't exist)
    for fallback_file in "${BP_FALLBACK}"/*.py; do
        module_name=$(basename "${fallback_file}")
        if [[ "${module_name}" == "__init__.py" ]]; then
            continue
        fi
        target="${ISAACLAB_DIR}/${module_name}"
        if [[ ! -f "${target}" ]]; then
            ln -sf "${fallback_file}" "${target}"
            log "  LINKED: ${module_name} → isaaclab/"
        else
            log "  OK: ${module_name} already exists (using original)"
        fi
    done
else
    log "  SKIP: isaaclab_fallback not found (will be created later)"
fi

log "All patches applied."
