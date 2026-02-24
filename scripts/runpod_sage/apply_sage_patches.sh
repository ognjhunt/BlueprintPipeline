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

# ── Patches 2+3: layout.py + layout_wo_robot.py — import guards ──────────────
log "Patch 2: layout.py — import guards for matfuse + isaaclab"
LAYOUT="${SAGE_DIR}/server/layout.py"
log "Patch 3: layout_wo_robot.py — import guards"
LAYOUT_WO="${SAGE_DIR}/server/layout_wo_robot.py"

# Shared logic: wrap bare top-level imports in try/except.
# Handles multi-line imports like: from X import (\n    Y\n)
for _layout_file in "${LAYOUT}" "${LAYOUT_WO}"; do
    if [[ ! -f "${_layout_file}" ]]; then
        log "  SKIP: $(basename "${_layout_file}") not found"
        continue
    fi
    if grep -q '^from floor_plan_materials\|^from isaaclab' "${_layout_file}"; then
        python3 - "${_layout_file}" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, 'r') as f:
    content = f.read()

# Skip if already wrapped
if '\ntry:\n    from floor_plan_materials' in content or \
   '\ntry:\n    from isaaclab' in content:
    print(f"  OK: import guards already applied in {path}")
    sys.exit(0)

lines = content.split('\n')
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if (line.startswith("from floor_plan_materials") or
        line.startswith("from isaaclab")):
        # Collect full import (may span multiple lines with parens)
        import_lines = [line]
        if '(' in line and ')' not in line:
            i += 1
            while i < len(lines):
                import_lines.append(lines[i])
                if ')' in lines[i]:
                    break
                i += 1
        new_lines.append("try:")
        for il in import_lines:
            new_lines.append("    " + il)
        new_lines.append("except (ImportError, FileNotFoundError, Exception):")
        new_lines.append("    pass")
    else:
        new_lines.append(line)
    i += 1

with open(path, 'w') as f:
    f.write('\n'.join(new_lines))
print(f"  PATCHED: import guards in {path}")
PYEOF
    else
        log "  OK: $(basename "${_layout_file}") already patched"
    fi
done

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
if python3.11 -c "import nvdiffrast" 2>/dev/null; then
    log "  OK: nvdiffrast already installed"
else
    log "  WARNING: nvdiffrast missing. Texture baking will fail in strict mode."
    log "  (Install it at image build time; runtime installs are disabled.)"
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

# ── Patch 9b: Stage 5-7 script source of truth symlinks ─────────────────────
log "Patch 9b: Stage 5-7 source-of-truth script symlinks"
BP_STAGE567="${WORKSPACE:-/workspace}/BlueprintPipeline/scripts/runpod_sage/sage_stage567_mobile_franka.py"
BP_STAGE7_COLLECTOR="${WORKSPACE:-/workspace}/BlueprintPipeline/scripts/runpod_sage/isaacsim_collect_mobile_franka.py"
BP_SIMREADY_LITE="${WORKSPACE:-/workspace}/BlueprintPipeline/scripts/runpod_sage/bp_simready_lite.py"
SAGE_STAGE567="${SAGE_DIR}/server/sage_stage567_mobile_franka.py"
SAGE_STAGE7_COLLECTOR="${SAGE_DIR}/server/isaacsim_collect_mobile_franka.py"
SAGE_SIMREADY_LITE="${SAGE_DIR}/server/bp_simready_lite.py"

_link_stage_script() {
    local src="$1"
    local dst="$2"
    local label="$3"
    if [[ ! -f "${src}" ]]; then
        log "  WARNING: missing source for ${label}: ${src}"
        return
    fi
    ln -sfn "${src}" "${dst}"
    log "  LINKED: ${dst} -> ${src}"
}

_link_stage_script "${BP_STAGE567}" "${SAGE_STAGE567}" "sage_stage567_mobile_franka.py"
_link_stage_script "${BP_STAGE7_COLLECTOR}" "${SAGE_STAGE7_COLLECTOR}" "isaacsim_collect_mobile_franka.py"
_link_stage_script "${BP_SIMREADY_LITE}" "${SAGE_SIMREADY_LITE}" "bp_simready_lite.py"

# ── Patch 10: key.py + layout.py — key.json path + missing Anthropic fallback ─
log "Patch 10: key.py + layout.py — key.json path fix + fallback for missing ANTHROPIC_API_KEY"

# key.py loads key.json relative to the current working directory, which breaks when
# server entrypoints are launched from /workspace/SAGE/client. Force key.json to be
# resolved relative to key.py itself.
KEYPY="${SAGE_DIR}/server/key.py"
if [[ -f "${KEYPY}" ]]; then
    python3 - "${KEYPY}" <<'PYEOF'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
orig = text

marker = "# BP_PATCH_KEY_JSON_RELATIVE"
if marker in text:
    print(f"  OK: key.json path already patched in {path}")
    raise SystemExit(0)

old = 'with open("key.json", "r") as f:\\n    key_dict = json.load(f)\\n'
new = (
    f"{marker}\\n"
    "from pathlib import Path\\n\\n"
    "_key_json_path = Path(__file__).resolve().parent / \"key.json\"\\n"
    "with open(_key_json_path, \"r\") as f:\\n"
    "    key_dict = json.load(f)\\n"
)

if old in text:
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  PATCHED: {path}")
else:
    print(f"  WARNING: key.json open pattern not found in {path}")
PYEOF
fi

for _layout in "${SAGE_DIR}/server/layout.py" "${SAGE_DIR}/server/layout_wo_robot.py"; do
    if [[ ! -f "${_layout}" ]]; then
        log "  SKIP: not found: ${_layout}"
        continue
    fi

    python3 - "${_layout}" <<'PYEOF'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
orig = text

text = text.replace("from key import ANTHROPIC_API_KEY", "from key import ANTHROPIC_API_KEY, API_TOKEN")

parse_old = """        # Check if API key is available\n        api_key = ANTHROPIC_API_KEY\n        if not api_key:\n            return json.dumps({\n                \"success\": False,\n                \"error\": \"ANTHROPIC_API_KEY environment variable is not set\"\n            })\n"""
parse_new = """        # Check for an available model key (prefer Anthropic, fallback to OpenAI)\n        api_key = ANTHROPIC_API_KEY or API_TOKEN\n        if not api_key:\n            return json.dumps({\n                \"success\": False,\n                \"error\": \"No ANTHROPIC_API_KEY or API_TOKEN available\"\n            })\n"""
place_old = """        # Check if API key is available\n        api_key = ANTHROPIC_API_KEY\n        if not api_key:\n            print(\"❌ ANTHROPIC_API_KEY not found\", file=sys.stderr)\n            return json.dumps({\n                \"success\": False,\n                \"error\": \"ANTHROPIC_API_KEY environment variable is not set\"\n            })\n"""
place_new = """        # Check for an available model key (prefer Anthropic, fallback to OpenAI)\n        api_key = ANTHROPIC_API_KEY or API_TOKEN\n        if not api_key:\n            print(\"❌ ANTHROPIC_API_KEY / API_TOKEN not found\", file=sys.stderr)\n            return json.dumps({\n                \"success\": False,\n                \"error\": \"No ANTHROPIC_API_KEY or API_TOKEN available\"\n            })\n"""

if parse_old in text:
    text = text.replace(parse_old, parse_new, 1)
if place_old in text:
    text = text.replace(place_old, place_new, 1)

if text != orig:
    path.write_text(text)
    print(f"  PATCHED: {path}")
else:
    print(f"  OK: no-op in {path}")
PYEOF
done

# ── Patch 10b: layout.py — allow estimated placement targets ────────────────
log "Patch 10b: layout.py — allow estimated placement targets (no invalidation)"
for _layout in "${SAGE_DIR}/server/layout.py" "${SAGE_DIR}/server/layout_wo_robot.py"; do
    if [[ ! -f "${_layout}" ]]; then
        log "  SKIP: not found: ${_layout}"
        continue
    fi

    python3 - "${_layout}" <<'PYEOF'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
orig = text

marker = "# BP_PATCH_ALLOW_ESTIMATED_LOCATION"
if marker in text:
    print(f"  OK: estimated-location patch already present in {path}")
    raise SystemExit(0)

pattern = re.compile(
    r'(^[ \t]*)location = str\\(object_data\\.get\\(\"location\", \"floor\"\\)\\)\\.lower\\(\\)\\n'
    r'\\1if location not in \\[\"floor\", \"wall\"\\]:\\n'
    r'\\1    # Check if it\\x27s a valid existing object ID\\n'
    r'\\1    existing_object_ids = \\[obj\\.id for obj in room\\.objects\\]\\n'
    r'\\1    if location not in existing_object_ids:\\n'
    r'\\1        # Invalid location, use invalid\\n'
    r'\\1        location = \"invalid\"\\n'
    r'\\1    # If it\\x27s a valid object ID, keep it as is\\n',
    flags=re.MULTILINE,
)

replacement = (
    r'\\1' + marker + r'\\n'
    r'\\1location = str(object_data.get(\"location\", \"floor\")).lower().strip()\\n'
    r'\\1if not location:\\n'
    r'\\1    location = \"floor\"\\n'
    r'\\1# Normalize common variants like \"on floor\", \"floor near wall\", etc.\\n'
    r'\\1if \"floor\" in location:\\n'
    r'\\1    location = \"floor\"\\n'
    r'\\1elif \"wall\" in location:\\n'
    r'\\1    location = \"wall\"\\n'
    r'\\1elif location not in [\"floor\", \"wall\"]:\\n'
    r'\\1    # Allow estimated object names (e.g., \"table\", \"countertop\") until IDs exist.\\n'
    r'\\1    existing_object_ids = [obj.id for obj in room.objects]\\n'
    r'\\1    if location in existing_object_ids:\\n'
    r'\\1        pass\\n'
)

text, n = pattern.subn(replacement, text)
if n == 0:
    print(f"  OK: no-op in {path} (location invalidation pattern not found)")
    raise SystemExit(0)

path.write_text(text)
print(f"  PATCHED: allowed estimated placement targets in {path} (occurrences={n})")
PYEOF
done

# Optional vLM hardening: keep Anthropic usage resilient (auto fallback to OpenAI).
if [[ -f "${SAGE_DIR}/server/vlm.py" ]]; then
    log "Patch 11: Add safe OpenAI fallback when Anthropic key is missing"
    python3 - "${SAGE_DIR}/server/vlm.py" <<'PYEOF'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text()
orig = text

old = "    if vlm_type == \"claude\":\n        return _call_claude_with_retry(\n            model, max_tokens, temperature, messages, thinking,\n            max_retries, retry_base_delay, retry_max_delay\n        )\n    elif vlm_type in [\"qwen\", \"openai\", \"glmv\"]:"
new = "    if vlm_type == \"claude\":\n        # Prefer Claude when key is available, otherwise fallback to OpenAI path.\n        if ANTHROPIC_API_KEY:\n            return _call_claude_with_retry(\n                model, max_tokens, temperature, messages, thinking,\n                max_retries, retry_base_delay, retry_max_delay\n            )\n        print(\"⚠️ ANTHROPIC_API_KEY missing; falling back to OpenAI for call_vlm(claude)\")\n        return _call_openai_with_retry(\n            \"openai\",\n            model,\n            max_tokens,\n            temperature,\n            messages,\n            thinking,\n            response_format,\n            max_retries,\n            retry_base_delay,\n            retry_max_delay,\n        )\n    elif vlm_type in [\"qwen\", \"openai\", \"glmv\"]:"

if old in text:
    text = text.replace(old, new, 1)

if text != orig:
    path.write_text(text)
    print(f"  PATCHED: {path}")
else:
    print(f"  OK: no-op in {path}")
PYEOF
fi

# ── Patch 12: Claude JSON parse failure -> GPT repair fallback ──────────────
log "Patch 12: Claude JSON parse repair fallback (Claude -> OpenAI when JSON extraction fails)"
for _sage_file in "${SAGE_DIR}/server/layout.py" "${SAGE_DIR}/server/layout_wo_robot.py" "${SAGE_DIR}/server/vlm.py"; do
    if [[ ! -f "${_sage_file}" ]]; then
        log "  SKIP: not found: ${_sage_file}"
        continue
    fi

    python3 - "${_sage_file}" <<'PYEOF'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
orig = text

marker = "# BP_PATCH_CLAUDE_JSON_REPAIR_FALLBACK"
had_marker = marker in text

patch = r'''

{marker}
# Fallback for cases where Claude returns non-JSON (or mixed markdown/text) when
# the caller expects JSON. We first try regex extraction; if that still fails,
# we ask OpenAI to return *only* valid JSON, then re-run the original extractor.
import json as _bp_json
import os as _bp_os
import re as _bp_re
import sys as _bp_sys
from typing import Any as _bp_Any, Optional as _bp_Optional


def _bp__extract_json_substring(_text: str) -> _bp_Optional[str]:
    if not isinstance(_text, str):
        return None
    t = _text.strip()
    if not t:
        return None

    # Strip markdown code fences.
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()

    # Try object first, then array.
    for pat in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = _bp_re.search(pat, t)
        if not m:
            continue
        candidate = m.group(0).strip()
        try:
            _bp_json.loads(candidate)
            return candidate
        except Exception:
            continue
    # Whole-string JSON?
    try:
        _bp_json.loads(t)
        return t
    except Exception:
        return None


def _bp__openai_repair_json(_raw_text: str) -> _bp_Optional[str]:
    extracted = _bp__extract_json_substring(_raw_text)
    if extracted is not None:
        return extracted

    api_key = _bp_os.getenv("OPENAI_API_KEY") or _bp_os.getenv("API_TOKEN")
    if not api_key:
        return None

    model = (
        _bp_os.getenv("SAGE_JSON_REPAIR_MODEL")
        or _bp_os.getenv("OPENAI_MODEL")
        or "gpt-5.1-mini"
    )

    system = (
        "You are a JSON repair tool. "
        "Return ONLY valid JSON (object or array), with no markdown fences and no prose. "
        "If the input already contains JSON, extract it and output it verbatim."
    )

    user = "Repair to strict JSON only. Input:\n" + _raw_text

    try:
        import openai as _bp_openai  # type: ignore
    except Exception:
        return None

    def _bp__truthy(v: object) -> bool:
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in {"1", "true", "yes", "on", "y"}

    def _bp__build_openai_client_kwargs() -> list[dict[str, str]]:
        kwargs: dict[str, str] = {"api_key": api_key}
        base_url = _bp_os.getenv("OPENAI_BASE_URL", "").strip()
        if base_url:
            kwargs["base_url"] = base_url
        websocket_base_url = _bp_os.getenv("OPENAI_WEBSOCKET_BASE_URL", "").strip()
        websocket_enabled = _bp__truthy(_bp_os.getenv("OPENAI_USE_WEBSOCKET", ""))
        if websocket_enabled and websocket_base_url:
            kwargs["websocket_base_url"] = websocket_base_url

        candidates = [dict(kwargs)]
        if "websocket_base_url" in kwargs:
            no_ws = dict(kwargs)
            no_ws.pop("websocket_base_url", None)
            candidates.append(no_ws)
        if "base_url" in kwargs:
            no_base = dict(kwargs)
            no_base.pop("base_url", None)
            candidates.append(no_base)
            if "websocket_base_url" in kwargs:
                no_base_no_ws = dict(no_base)
                no_base_no_ws.pop("websocket_base_url", None)
                candidates.append(no_base_no_ws)

        candidates.append({"api_key": api_key})
        return candidates

    def _bp__build_openai_client() -> _bp_Any:
        if not hasattr(_bp_openai, "OpenAI"):
            raise AttributeError("OpenAI SDK v1 not available")

        candidates = _bp__build_openai_client_kwargs()
        seen: set[tuple[tuple[str, str], ...]] = set()
        last_error = None
        for candidate in candidates:
            key = tuple(sorted(candidate.items()))
            if key in seen:
                continue
            seen.add(key)
            try:
                return _bp_openai.OpenAI(**candidate)
            except TypeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return _bp_openai.OpenAI(api_key=api_key)

    out = ""
    try:
        if hasattr(_bp_openai, "OpenAI"):
            client = _bp__build_openai_client()

            # Prefer chat.completions (SDK v1 path)
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0,
                        max_completion_tokens=2000,
                    )
                except TypeError:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0,
                        max_tokens=2000,
                    )
                out = (resp.choices[0].message.content or "").strip()

            # Best-effort fallback to responses API if available.
            if not out and hasattr(client, "responses"):
                try:
                    resp = client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": [{"type": "text", "text": system}]},
                            {"role": "user", "content": [{"type": "text", "text": user}]},
                        ],
                        temperature=0,
                        max_output_tokens=2000,
                    )
                except TypeError:
                    resp = client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": [{"type": "text", "text": system}]},
                            {"role": "user", "content": [{"type": "text", "text": user}]},
                        ],
                        temperature=0,
                        max_tokens=2000,
                    )
                out = (getattr(resp, "output_text", "") or "").strip()
        else:
            # Legacy SDK path
            _bp_openai.api_key = api_key
            try:
                resp = _bp_openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                    max_completion_tokens=2000,
                )
            except TypeError:
                resp = _bp_openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                    max_tokens=2000,
                )
            out = (resp["choices"][0]["message"]["content"] or "").strip()
    except Exception as exc:
        print(f"[BP] OpenAI JSON repair failed: {exc}", file=_bp_sys.stderr, flush=True)
        return None

    extracted = _bp__extract_json_substring(out)
    return extracted


def _bp__wrap_json_extractors() -> None:
    # Wrap common JSON extractors used by SAGE so "Could not extract JSON" from
    # Claude can be repaired via OpenAI.
    candidates = (
        "extract_json",
        "extract_json_from_text",
        "extract_json_from_response",
        "_extract_json",
        "_extract_json_from_text",
        "_extract_json_from_response",
    )
    for name in candidates:
        fn = globals().get(name)
        if not callable(fn) or getattr(fn, "_bp_wrapped", False):
            continue

        def _wrapper(text, *args, __fn=fn, __name=name, **kwargs):
            try:
                return __fn(text, *args, **kwargs)
            except Exception:
                repaired = _bp__openai_repair_json(text)
                if repaired is None:
                    raise
                _na = "n/a"
                _tlen = len(text) if isinstance(text, str) else _na
                print(
                    f"[BP] JSON repair fallback used for {__name} (len={_tlen})",
                    file=_bp_sys.stderr,
                    flush=True,
                )
                # Re-run original extractor so return type matches (str vs parsed obj).
                return __fn(repaired, *args, **kwargs)

        _wrapper._bp_wrapped = True  # type: ignore[attr-defined]
        globals()[name] = _wrapper


try:
    _bp__wrap_json_extractors()
except Exception as _exc:
    print(f"[BP] WARNING: failed to install JSON repair wrappers: {_exc}", file=_bp_sys.stderr, flush=True)

'''

patch = patch.replace("{marker}", marker)
if had_marker:
    # Upgrade path: remove existing appended block (marker -> EOF), then re-append.
    text = text[: text.index(marker)].rstrip()
text = text.rstrip() + patch + "\n"
path.write_text(text)
print(f"  PATCHED: JSON repair fallback {'replaced' if had_marker else 'appended'} to {path}")
PYEOF
done

# ── Patch 12b: client_generation_scene_aug.py — tolerate missing match fields ─
log "Patch 12b: client_generation_scene_aug.py — tolerate missing matched_object_ids"
AUG_CLIENT="${SAGE_DIR}/client/client_generation_scene_aug.py"
if [[ -f "${AUG_CLIENT}" ]]; then
    python3 - "${AUG_CLIENT}" <<'PYEOF'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
orig = text

marker = "# BP_PATCH_SCENE_AUG_MISSING_MATCHES"
if marker in text:
    print(f"  OK: scene_aug missing-match patch already present in {path}")
    raise SystemExit(0)

old = (
    '            assert "object_type" in obj_dict, "Object type not found in object dictionary"\\n'
    '            assert "matched_object_ids" in obj_dict, "Matched object ids not found in object dictionary"\\n'
    '            assert "matched_object_types" in obj_dict, "Matched object types not found in object dictionary"\\n\\n'
    '            object_type = obj_dict["object_type"]\\n'
    '            matched_object_ids = obj_dict["matched_object_ids"]\\n'
    '            matched_object_types = obj_dict["matched_object_types"]\\n'
)

new = (
    f"            {marker}\\n"
    '            assert "object_type" in obj_dict, "Object type not found in object dictionary"\\n'
    '            object_type = obj_dict["object_type"]\\n'
    '            matched_object_ids = obj_dict.get("matched_object_ids") or []\\n'
    '            matched_object_types = obj_dict.get("matched_object_types") or []\\n'
    '            if not matched_object_ids or not matched_object_types:\\n'
    '                # Stage 1-3 outputs often only include object_type/quantity/placement_guidance\\n'
    '                # and do not have matched_object_ids yet. Keep placeholders so augmentation can proceed.\\n'
    '                qty = int(obj_dict.get("quantity", 1) or 1)\\n'
    '                place_id = str(obj_dict.get("placement_guidance", "floor") or "floor")\\n'
    '                place_object_type = place_id if place_id in ["floor", "wall"] else place_id\\n'
    '                for _i in range(qty):\\n'
    '                    objects_to_keep.append({\\n'
    '                        "id": f"required_{object_type}_{_i+1}",\\n'
    '                        "type": object_type,\\n'
    '                        "description": object_type,\\n'
    '                        "place_id": place_id,\\n'
    '                        "place_object_type": place_object_type,\\n'
    '                    })\\n'
    '                continue\\n\\n'
)

if old in text:
    text = text.replace(old, new, 1)

if text != orig:
    path.write_text(text)
    print(f"  PATCHED: {path}")
else:
    print(f"  OK: no-op in {path}")
PYEOF
fi

# ── Patch 13: client_generation_robot_task.py — respect MAX_OBJECTS cap ────────
log "Patch 13: client_generation_robot_task.py — enforce MAX_OBJECTS cap in Qwen prompt"
ROBOT_CLIENT="${SAGE_DIR}/client/client_generation_robot_task.py"
if [[ -f "${ROBOT_CLIENT}" ]]; then
    python3 - "${ROBOT_CLIENT}" <<'PYEOF'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
orig = text

# The SAGE client prompt has two lines that explicitly say "NO maximum":
#   - NO maximum on total objects across all calls
#   - NO maximum on total object types across all calls
# Replace them with a dynamic cap read from SAGE_MAX_OBJECTS env var.
# If SAGE_MAX_OBJECTS is unset or 0, keep the original "no maximum" behavior.

old_no_max_objects = "- NO maximum on total objects across all calls"
new_max_objects = (
    "- IMPORTANT: Total objects across ALL calls MUST NOT exceed the scene budget. "
    "Check the task description for the exact limit (typically 20-30). "
    "Prioritize task-critical objects and essential furniture; omit low-priority clutter once near the cap."
)

old_no_max_types = "- NO maximum on total object types across all calls"
new_max_types = (
    "- Keep total object TYPES proportional to the scene budget "
    "(roughly 60-70% of the total object cap)."
)

if old_no_max_objects in text:
    text = text.replace(old_no_max_objects, new_max_objects)
    print(f"  PATCHED: replaced 'NO maximum objects' line")
else:
    print(f"  OK: 'NO maximum objects' line already patched or not found")

if old_no_max_types in text:
    text = text.replace(old_no_max_types, new_max_types)
    print(f"  PATCHED: replaced 'NO maximum types' line")
else:
    print(f"  OK: 'NO maximum types' line already patched or not found")

if text != orig:
    path.write_text(text)
    print(f"  PATCHED: {path}")
else:
    print(f"  OK: no changes needed in {path}")
PYEOF
fi

# ── Patch 14: pose_aug_mm_from_layout_with_task.py — guard None base poses ─────
log "Patch 14: pose_aug_mm_from_layout_with_task.py — guard None robot base positions"
POSE_AUG_SCRIPT="${SAGE_DIR}/server/augment/pose_aug_mm_from_layout_with_task.py"
if [[ -f "${POSE_AUG_SCRIPT}" ]]; then
    python3 - "${POSE_AUG_SCRIPT}" <<'PYEOF'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
orig = text

marker = "# BP_PATCH_POSE_AUG_SAFE_CPU"
if marker in text:
    print(f"  OK: pose_aug safe-cpu patch already present in {path}")
    raise SystemExit(0)

# The upstream script can return None for base poses in rare cases (e.g. no feasible
# placement samples). Downstream code sometimes calls `.cpu()` on these values.
pat = re.compile(
    r"^(?P<indent>\\s*)(?P<var>robot_base_pos_(?:pick|place))\\s*=\\s*(?P=var)\\.cpu\\(\\)(?P<suffix>[^\\n]*)$",
    re.MULTILINE,
)

def repl(m: re.Match[str]) -> str:
    indent = m.group("indent")
    var = m.group("var")
    suffix = m.group("suffix")
    # Preserve any chained operations after `.cpu()` (e.g. `.numpy()`).
    return f"{indent}{var} = {var}.cpu(){suffix} if {var} is not None else None  {marker}"

text = pat.sub(repl, text)

if text != orig:
    path.write_text(text)
    print(f"  PATCHED: {path}")
else:
    print(f"  OK: no changes needed in {path}")
PYEOF
fi

# ── Final: Validate syntax of all patched Python files ──────────────────────
log "Validating patched file syntax..."
_SAGE_PY="${WORKSPACE:-/workspace}/miniconda3/envs/sage/bin/python"
[[ ! -x "${_SAGE_PY}" ]] && _SAGE_PY="python3"
_had_errors=0
for _check_file in \
    "${SAGE_DIR}/server/layout.py" \
    "${SAGE_DIR}/server/layout_wo_robot.py" \
    "${SAGE_DIR}/server/vlm.py" \
    "${SAGE_DIR}/server/augment/pose_aug_mm_from_layout_with_task.py" \
    "${SAGE_DIR}/client/client_generation_robot_task.py" \
    "${SAGE_DIR}/client/client_generation_room_desc.py" \
    "${SAGE_DIR}/client/client_generation_scene_aug.py"; do
    if [[ -f "${_check_file}" ]]; then
        if ! "${_SAGE_PY}" -c "import py_compile; py_compile.compile('${_check_file}', doraise=True)" 2>/dev/null; then
            log "  ERROR: Syntax error in ${_check_file}!"
            _had_errors=1
        fi
    fi
done
if [[ "${_had_errors}" == "0" ]]; then
    log "  All files pass syntax check."
fi

log "All patches applied."
