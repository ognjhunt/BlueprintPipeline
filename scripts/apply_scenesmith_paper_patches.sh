#!/usr/bin/env bash
# =============================================================================
# Apply portable SceneSmith paper-stack runtime patches (idempotent)
# =============================================================================
set -euo pipefail

log() { echo "[scenesmith-patches $(date -u +%FT%TZ)] $*"; }

is_truthy() {
  local raw="${1:-}"
  local normalized
  normalized="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')"
  case "${normalized}" in
    1|true|yes|on|y) return 0 ;;
    *) return 1 ;;
  esac
}

STRICT=0
if is_truthy "${SCENESMITH_PATCH_STRICT:-0}"; then
  STRICT=1
fi

warn_or_fail() {
  local msg="$1"
  if (( STRICT )); then
    log "ERROR: ${msg}"
    exit 1
  fi
  log "WARNING: ${msg}"
}

SCENESMITH_DIR="${SCENESMITH_PAPER_REPO_DIR:-${SCENESMITH_DIR:-/workspace/scenesmith}}"
SCENESMITH_DIR="${SCENESMITH_DIR/#\~/${HOME}}"
PYTHON_BIN="${SCENESMITH_PAPER_PYTHON_BIN:-${SCENESMITH_DIR}/.venv/bin/python}"

if [[ ! -d "${SCENESMITH_DIR}" ]]; then
  warn_or_fail "SceneSmith repo not found at ${SCENESMITH_DIR}"
  exit 0
fi

log "Applying SceneSmith paper-stack runtime patches"
log "  repo=${SCENESMITH_DIR}"
log "  python=${PYTHON_BIN}"

ensure_material_dirs() {
  mkdir -p "${SCENESMITH_DIR}/data/materials" "${SCENESMITH_DIR}/data/materials/embeddings" || \
    warn_or_fail "Failed to create materials directories"
}

ensure_sam3_repo() {
  local sam3_dir="${SCENESMITH_DIR}/external/SAM3"
  if [[ -d "${sam3_dir}/.git" ]]; then
    log "  OK: SAM3 repo present (${sam3_dir})"
    return 0
  fi

  if ! is_truthy "${SCENESMITH_PATCH_AUTO_CLONE_SAM3:-1}"; then
    warn_or_fail "SAM3 repo missing and auto-clone disabled: ${sam3_dir}"
    return 0
  fi

  if ! command -v git >/dev/null 2>&1; then
    warn_or_fail "git is required to clone SAM3 (missing: ${sam3_dir})"
    return 0
  fi

  mkdir -p "${SCENESMITH_DIR}/external"
  log "  Cloning SAM3 into ${sam3_dir}"
  if git clone --depth 1 https://github.com/facebookresearch/sam3.git "${sam3_dir}" >/tmp/scenesmith_patch_sam3.log 2>&1; then
    log "  OK: cloned SAM3"
  else
    tail -n 20 /tmp/scenesmith_patch_sam3.log || true
    warn_or_fail "Failed to clone SAM3"
  fi
}

patch_indoor_scene_generation() {
  local indoor_file="${SCENESMITH_DIR}/scenesmith/experiments/indoor_scene_generation.py"
  if [[ ! -f "${indoor_file}" ]]; then
    warn_or_fail "indoor_scene_generation.py not found at ${indoor_file}"
    return 0
  fi

  if python3 - "${indoor_file}" <<'PYEOF'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = "BlueprintPipeline materials server patch"
if marker in text:
    print(f"  OK: materials server patch already present in {path}")
    raise SystemExit(0)

start_token = "    def _start_materials_server(self) -> None:\n"
end_token = "    def _stop_materials_server(self) -> None:\n"
start = text.find(start_token)
end = text.find(end_token)
if start < 0 or end < 0 or end <= start:
    print(f"  ERROR: could not locate _start/_stop materials server functions in {path}")
    raise SystemExit(2)

replacement = '''    def _start_materials_server(self) -> None:
        """Start materials retrieval server."""
        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.materials_retrieval_server

        # BlueprintPipeline materials server patch:
        # Skip startup when required materials data is unavailable.
        data_path_raw = str(server_config.get("data_path", "")).strip()
        embeddings_path_raw = str(server_config.get("embeddings_path", "")).strip()
        data_path = Path(data_path_raw).expanduser() if data_path_raw else Path()
        embeddings_path = (
            Path(embeddings_path_raw).expanduser() if embeddings_path_raw else Path()
        )
        required_embedding_files = (
            embeddings_path / "clip_embeddings.npy",
            embeddings_path / "embedding_index.yaml",
            embeddings_path / "metadata_index.yaml",
        )
        if (
            not data_path_raw
            or not embeddings_path_raw
            or not data_path.exists()
            or not embeddings_path.exists()
            or any(not file_path.exists() for file_path in required_embedding_files)
        ):
            console_logger.warning(
                "BlueprintPipeline materials server patch: missing materials data; "
                "skipping materials retrieval server startup "
                f"(data_path={data_path_raw!r}, embeddings_path={embeddings_path_raw!r})"
            )
            return

        retrieval_device = _get_retrieval_gpu_device()
        console_logger.info(
            f"Starting materials retrieval server on "
            f"{server_config.host}:{server_config.port} "
            f"(CLIP device: {retrieval_device or 'default'})"
        )

        self.materials_server = MaterialsRetrievalServer(
            host=server_config.host,
            port=server_config.port,
            preload_retriever=True,  # Always preload CLIP for consistent performance.
            materials_config=server_config,  # Pass DictConfig directly.
            clip_device=retrieval_device,
        )

        self.materials_server.start()
        # Longer timeout for CLIP loading.
        self.materials_server.wait_until_ready(timeout_s=60.0)
        console_logger.info("Materials retrieval server ready")
'''

patched = text[:start] + replacement + "\n" + text[end:]
path.write_text(patched, encoding="utf-8")
print(f"  PATCHED: materials server guard in {path}")
PYEOF
  then
    return 0
  fi

  warn_or_fail "Failed to patch materials server startup guard"
}

patch_floor_plan_material_fallbacks() {
  if ! is_truthy "${SCENESMITH_PATCH_ENABLE_FLOOR_PLAN_FALLBACK_PATCH:-0}"; then
    log "  SKIP: floor plan material fallback patch disabled (set SCENESMITH_PATCH_ENABLE_FLOOR_PLAN_FALLBACK_PATCH=1 to enable)"
    return 0
  fi

  local resolver_file="${SCENESMITH_DIR}/scenesmith/floor_plan_agents/tools/materials_resolver.py"
  if [[ ! -f "${resolver_file}" ]]; then
    warn_or_fail "materials_resolver.py not found at ${resolver_file}"
  elif python3 - "${resolver_file}" <<'PYEOF'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = "BlueprintPipeline material id fallback patch"
if marker in text:
    print(f"  OK: material id fallback patch already present in {path}")
    raise SystemExit(0)

if "class MaterialsResolver" not in text:
    print(f"  SKIP: class MaterialsResolver not found in {path}")
    raise SystemExit(0)

append_block = '''
# BlueprintPipeline material id fallback patch:
# Unknown/empty material ids fallback to default or first available material id.

def _bp_material_nonempty(value):
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return None


def _bp_material_lookup(container, key):
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            pass
    return getattr(container, key, None)


def _bp_material_candidates_from(container, kind):
    if container is None:
        return []

    keys = []
    if kind == "wall":
        keys.extend(("default_wall_material_id", "wall_material_id", "default_material_id"))
    elif kind == "floor":
        keys.extend(("default_floor_material_id", "floor_material_id", "default_material_id"))
    else:
        keys.extend(("default_material_id", "default_wall_material_id", "default_floor_material_id"))
    keys.extend(("default_material_id", "default_wall_material_id", "default_floor_material_id"))

    out = []
    for key in keys:
        candidate = _bp_material_nonempty(_bp_material_lookup(container, key))
        if candidate:
            out.append(candidate)

    materials_node = _bp_material_lookup(container, "materials")
    if materials_node is not None and materials_node is not container:
        for key in keys:
            candidate = _bp_material_nonempty(_bp_material_lookup(materials_node, key))
            if candidate:
                out.append(candidate)
    return out


if "MaterialsResolver" in globals() and hasattr(MaterialsResolver, "_resolve_material_id"):
    _bp_original_resolve_material_id = MaterialsResolver._resolve_material_id
    if getattr(_bp_original_resolve_material_id, "__name__", "") != "_bp_resolve_material_id_with_fallback":

        def _bp_resolve_material_id_with_fallback(self, material_id):
            requested = _bp_material_nonempty(material_id)
            if requested:
                try:
                    resolved = _bp_original_resolve_material_id(self, requested)
                    resolved = _bp_material_nonempty(resolved)
                    if resolved:
                        return resolved
                except Exception:
                    pass

            fallback_candidates = []
            for container in (
                self,
                getattr(self, "materials_config", None),
                getattr(self, "cfg", None),
                getattr(self, "config", None),
            ):
                fallback_candidates.extend(_bp_material_candidates_from(container, "any"))

            for mapping_name in (
                "materials_by_id",
                "_materials_by_id",
                "materials",
                "material_id_to_metadata",
                "material_id_to_path",
            ):
                mapping = getattr(self, mapping_name, None)
                if isinstance(mapping, dict):
                    for candidate in mapping.keys():
                        candidate = _bp_material_nonempty(candidate)
                        if candidate:
                            fallback_candidates.append(candidate)
                            break

            seen = set()
            for candidate in fallback_candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                try:
                    resolved = _bp_original_resolve_material_id(self, candidate)
                    resolved = _bp_material_nonempty(resolved)
                    if resolved:
                        if resolved != requested:
                            import logging

                            logging.getLogger(__name__).warning(
                                "BlueprintPipeline material id fallback patch: requested=%r fallback=%r",
                                material_id,
                                resolved,
                            )
                        return resolved
                except Exception:
                    continue

            return _bp_original_resolve_material_id(self, material_id)

        MaterialsResolver._resolve_material_id = _bp_resolve_material_id_with_fallback
'''

path.write_text(text.rstrip() + "\n\n" + append_block.lstrip("\n"), encoding="utf-8")
print(f"  PATCHED: material id fallback in {path}")
PYEOF
  then
    :
  else
    warn_or_fail "Failed to patch materials_resolver.py fallback behavior"
  fi

  local tools_file="${SCENESMITH_DIR}/scenesmith/floor_plan_agents/tools/floor_plan_tools.py"
  if [[ ! -f "${tools_file}" ]]; then
    warn_or_fail "floor_plan_tools.py not found at ${tools_file}"
    return 0
  fi

  if python3 - "${tools_file}" <<'PYEOF'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = "BlueprintPipeline floor plan material defaults patch"
if marker in text:
    print(f"  OK: floor plan material defaults patch already present in {path}")
    raise SystemExit(0)

if "class FloorPlanTools" not in text:
    print(f"  SKIP: class FloorPlanTools not found in {path}")
    raise SystemExit(0)

append_block = '''
# BlueprintPipeline floor plan material defaults patch:
# Empty wall/floor material ids are replaced with defaults before validation.

def _bp_fp_nonempty(value):
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return None


def _bp_fp_lookup(container, key):
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            pass
    return getattr(container, key, None)


def _bp_fp_default_material_from_tools(tools_obj, kind):
    resolver = getattr(tools_obj, "materials_resolver", None)
    if resolver is not None and hasattr(resolver, "_resolve_material_id"):
        try:
            resolved = resolver._resolve_material_id("")
            resolved = _bp_fp_nonempty(resolved)
            if resolved:
                return resolved
        except Exception:
            pass

    if kind == "wall":
        keys = ["default_wall_material_id", "wall_material_id", "default_material_id"]
    else:
        keys = ["default_floor_material_id", "floor_material_id", "default_material_id"]
    keys += ["default_material_id", "default_wall_material_id", "default_floor_material_id"]

    for container in (
        tools_obj,
        getattr(tools_obj, "materials_config", None),
        getattr(tools_obj, "cfg", None),
        getattr(tools_obj, "config", None),
        getattr(tools_obj, "floor_plan_cfg", None),
        getattr(tools_obj, "floor_plan_config", None),
    ):
        if container is None:
            continue
        for key in keys:
            candidate = _bp_fp_nonempty(_bp_fp_lookup(container, key))
            if candidate:
                return candidate
        nested = _bp_fp_lookup(container, "materials")
        if nested is not None and nested is not container:
            for key in keys:
                candidate = _bp_fp_nonempty(_bp_fp_lookup(nested, key))
                if candidate:
                    return candidate

    if resolver is not None:
        for mapping_name in (
            "materials_by_id",
            "_materials_by_id",
            "materials",
            "material_id_to_metadata",
            "material_id_to_path",
        ):
            mapping = getattr(resolver, mapping_name, None)
            if isinstance(mapping, dict):
                for candidate in mapping.keys():
                    candidate = _bp_fp_nonempty(candidate)
                    if candidate:
                        return candidate
    return None


if "FloorPlanTools" in globals() and hasattr(FloorPlanTools, "_set_room_materials_impl"):
    import inspect as _bp_fp_inspect

    _bp_original_set_room_materials_impl = FloorPlanTools._set_room_materials_impl
    if getattr(_bp_original_set_room_materials_impl, "__name__", "") != "_bp_set_room_materials_impl_with_defaults":
        _bp_signature = _bp_fp_inspect.signature(_bp_original_set_room_materials_impl)
        _bp_positional_names = []
        for _bp_name, _bp_param in _bp_signature.parameters.items():
            if _bp_name == "self":
                continue
            if _bp_param.kind in (
                _bp_fp_inspect.Parameter.POSITIONAL_ONLY,
                _bp_fp_inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                _bp_positional_names.append(_bp_name)
        _bp_positional_index = {
            name: idx for idx, name in enumerate(_bp_positional_names)
        }

        def _bp_set_room_materials_impl_with_defaults(self, *args, **kwargs):
            args_list = list(args)

            def _read_arg(name):
                if name in kwargs:
                    return kwargs.get(name)
                idx = _bp_positional_index.get(name)
                if idx is None or idx >= len(args_list):
                    return None
                return args_list[idx]

            def _write_arg(name, value):
                if name in kwargs:
                    kwargs[name] = value
                    return
                idx = _bp_positional_index.get(name)
                if idx is None:
                    kwargs[name] = value
                    return
                while len(args_list) <= idx:
                    args_list.append(None)
                args_list[idx] = value

            for arg_name, kind in (
                ("wall_material_id", "wall"),
                ("floor_material_id", "floor"),
            ):
                current = _bp_fp_nonempty(_read_arg(arg_name))
                if current:
                    continue
                fallback = _bp_fp_default_material_from_tools(self, kind)
                if fallback:
                    _write_arg(arg_name, fallback)
                    import logging

                    logging.getLogger(__name__).warning(
                        "BlueprintPipeline floor plan material defaults patch: "
                        "using %s fallback material_id=%r",
                        kind,
                        fallback,
                    )

            return _bp_original_set_room_materials_impl(
                self,
                *tuple(args_list),
                **kwargs,
            )

        FloorPlanTools._set_room_materials_impl = _bp_set_room_materials_impl_with_defaults
'''

path.write_text(text.rstrip() + "\n\n" + append_block.lstrip("\n"), encoding="utf-8")
print(f"  PATCHED: floor plan material defaults in {path}")
PYEOF
  then
    return 0
  fi

  warn_or_fail "Failed to patch floor_plan_tools.py defaults handling"
}

patch_parallel_start_method() {
  local parallel_file="${SCENESMITH_DIR}/scenesmith/utils/parallel.py"
  if [[ ! -f "${parallel_file}" ]]; then
    warn_or_fail "parallel.py not found at ${parallel_file}"
    return 0
  fi

  if python3 - "${parallel_file}" <<'PYEOF'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = "BlueprintPipeline multiprocessing start-method patch"
if marker in text:
    print(f"  OK: multiprocessing start-method patch already present in {path}")
    raise SystemExit(0)

if "def run_parallel_isolated(" not in text:
    print(f"  SKIP: run_parallel_isolated not found in {path}")
    raise SystemExit(0)

append_block = '''
# BlueprintPipeline multiprocessing start-method patch:
# Default to fork (stable in the current container runtime), while allowing override
# to forkserver/spawn via SCENESMITH_MP_START_METHOD.
def _bp_get_mp_context() -> tuple[multiprocessing.context.BaseContext, str]:
    import os as _bp_os

    requested = str(
        _bp_os.getenv("SCENESMITH_MP_START_METHOD", "fork")
    ).strip().lower()
    if requested in {"", "auto", "default"}:
        requested = "fork"

    candidates = [requested]
    if "forkserver" not in candidates:
        candidates.append("forkserver")
    if "spawn" not in candidates:
        candidates.append("spawn")

    for method in candidates:
        try:
            return multiprocessing.get_context(method), method
        except Exception:
            continue

    # Last-resort fallback (should not happen in normal Linux/python builds).
    return multiprocessing.get_context(), "default"


if "run_parallel_isolated" in globals():
    _bp_original_run_parallel_isolated = run_parallel_isolated
    if getattr(_bp_original_run_parallel_isolated, "__name__", "") != "_bp_run_parallel_isolated":
        def _bp_run_parallel_isolated(
            tasks: list[tuple[str, Callable, dict]],
            max_workers: int,
            return_values: bool = False,
        ) -> dict[str, tuple[bool, Any]]:
            ctx, start_method = _bp_get_mp_context()
            console_logger.info(
                "BlueprintPipeline multiprocessing start-method patch: using %s",
                start_method,
            )

            result_queue = ctx.Queue()
            pending = list(tasks)
            active: dict[int, tuple[multiprocessing.Process, str]] = {}
            results: dict[str, tuple[bool, Any]] = {}

            while pending or active:
                while len(active) < max_workers and pending:
                    task_id, target, kwargs = pending.pop(0)
                    proc = ctx.Process(
                        target=_worker_wrapper,
                        args=(target, kwargs, task_id, result_queue, return_values),
                    )
                    proc.start()
                    active[proc.pid] = (proc, task_id)
                    console_logger.info(
                        f"Started {task_id} (pid={proc.pid}, start_method={start_method})"
                    )

                if active:
                    sentinels = [proc.sentinel for proc, _ in active.values()]
                    wait(sentinels, timeout=1.0)

                while True:
                    try:
                        result_task_id, success, result_or_error = result_queue.get_nowait()
                        results[result_task_id] = (success, result_or_error)
                        status = "completed" if success else f"failed: {result_or_error}"
                        console_logger.info(f"{result_task_id} {status}")
                    except queue.Empty:
                        break

                for pid, (proc, task_id) in list(active.items()):
                    if not proc.is_alive():
                        proc.join()
                        del active[pid]
                        if task_id not in results:
                            signal_name = _get_signal_name(proc.exitcode)
                            results[task_id] = (
                                False,
                                f"Process crashed (exitcode={proc.exitcode}{signal_name})",
                            )
                            console_logger.error(
                                f"{task_id} crashed (exitcode={proc.exitcode}{signal_name})"
                            )

            return results

        run_parallel_isolated = _bp_run_parallel_isolated
'''

path.write_text(text.rstrip() + "\n\n" + append_block.lstrip("\n"), encoding="utf-8")
print(f"  PATCHED: multiprocessing start-method in {path}")
PYEOF
  then
    return 0
  fi

  warn_or_fail "Failed to patch parallel.py multiprocessing start method"
}

patch_agents_tool_typing() {
  if is_truthy "${SCENESMITH_PATCH_SKIP_AGENTS_TOOL_PATCH:-0}"; then
    log "  SKIP: agents/tool.py patch disabled by SCENESMITH_PATCH_SKIP_AGENTS_TOOL_PATCH"
    return 0
  fi

  if [[ ! -x "${PYTHON_BIN}" ]]; then
    warn_or_fail "Python binary not found/executable: ${PYTHON_BIN}"
    return 0
  fi

  local purelib
  if ! purelib="$(${PYTHON_BIN} - <<'PYEOF'
import sysconfig
print(sysconfig.get_paths().get("purelib", ""))
PYEOF
)"; then
    warn_or_fail "Failed to resolve site-packages path"
    return 0
  fi

  local tool_file="${purelib}/agents/tool.py"
  if [[ ! -f "${tool_file}" ]]; then
    log "  SKIP: agents/tool.py not found at ${tool_file}"
    return 0
  fi

  if python3 - "${tool_file}" <<'PYEOF'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = "BlueprintPipeline typing patch"
if marker in text:
    print(f"  OK: typing patch already present in {path}")
    raise SystemExit(0)

old_block = """ToolFunction = Union[
    ToolFunctionWithoutContext[ToolParams],
    ToolFunctionWithContext[ToolParams],
    ToolFunctionWithToolContext[ToolParams],
]
"""
new_block = """ToolFunction = Union[
    # BlueprintPipeline typing patch: avoid Python 3.11.0 ParamSpec KeyError (~TContext).
    ToolFunctionWithoutContext,
    ToolFunctionWithContext,
    ToolFunctionWithToolContext,
]
"""

if old_block in text:
    patched = text.replace(old_block, new_block, 1)
else:
    patterns = (
        r"ToolFunctionWithoutContext\[ToolParams\]",
        r"ToolFunctionWithContext\[ToolParams\]",
        r"ToolFunctionWithToolContext\[ToolParams\]",
    )
    if not all(re.search(p, text) for p in patterns):
        print(f"  SKIP: expected ToolFunction type pattern not found in {path}")
        raise SystemExit(0)

    patched = text
    patched = re.sub(
        r"ToolFunctionWithoutContext\[ToolParams\]",
        "ToolFunctionWithoutContext",
        patched,
        count=1,
    )
    patched = re.sub(
        r"ToolFunctionWithContext\[ToolParams\]",
        "ToolFunctionWithContext",
        patched,
        count=1,
    )
    patched = re.sub(
        r"ToolFunctionWithToolContext\[ToolParams\]",
        "ToolFunctionWithToolContext",
        patched,
        count=1,
    )

path.write_text(patched, encoding="utf-8")
print(f"  PATCHED: typing workaround in {path}")
PYEOF
  then
    return 0
  fi

  warn_or_fail "Failed to patch agents/tool.py typing aliases"
}

install_missing_python_modules() {
  if is_truthy "${SCENESMITH_PATCH_SKIP_PYTHON_INSTALL:-0}"; then
    log "  SKIP: python module install disabled by SCENESMITH_PATCH_SKIP_PYTHON_INSTALL"
    return 0
  fi

  if [[ ! -x "${PYTHON_BIN}" ]]; then
    warn_or_fail "Cannot check python modules, binary missing: ${PYTHON_BIN}"
    return 0
  fi

  local missing_modules=()
  while IFS= read -r module_name; do
    [[ -n "${module_name}" ]] && missing_modules+=("${module_name}")
  done < <("${PYTHON_BIN}" - <<'PYEOF'
import importlib.util

for module_name in ("decord", "pycocotools", "utils3d"):
    if importlib.util.find_spec(module_name) is None:
        print(module_name)
PYEOF
)

  if [[ ${#missing_modules[@]} -eq 0 ]]; then
    log "  OK: decord + pycocotools + utils3d already installed"
    return 0
  fi

  log "  Installing missing python modules: ${missing_modules[*]}"

  local install_rc=0
  if command -v uv >/dev/null 2>&1; then
    (
      cd "${SCENESMITH_DIR}"
      uv pip install "${missing_modules[@]}"
    ) >/tmp/scenesmith_patch_pip.log 2>&1 || install_rc=$?
  else
    install_rc=1
  fi

  if [[ ${install_rc} -ne 0 ]]; then
    if "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
      "${PYTHON_BIN}" -m pip install "${missing_modules[@]}" >/tmp/scenesmith_patch_pip.log 2>&1 || install_rc=$?
    fi
  fi

  if [[ ${install_rc} -ne 0 ]]; then
    tail -n 40 /tmp/scenesmith_patch_pip.log || true
    warn_or_fail "Failed to install python modules: ${missing_modules[*]}"
    return 0
  fi

  log "  OK: installed missing python modules"
}

ensure_timm_and_kaolin() {
  if is_truthy "${SCENESMITH_PATCH_SKIP_TIMM_KAOLIN_CHECK:-0}"; then
    log "  SKIP: timm/kaolin checks disabled by SCENESMITH_PATCH_SKIP_TIMM_KAOLIN_CHECK"
    return 0
  fi

  if [[ ! -x "${PYTHON_BIN}" ]]; then
    warn_or_fail "Cannot check timm/kaolin, python missing: ${PYTHON_BIN}"
    return 0
  fi

  local check_output
  if ! check_output="$("${PYTHON_BIN}" - <<'PYEOF'
from importlib import metadata

def parse_tuple(version_raw: str) -> tuple[int, ...]:
    parts = []
    for token in version_raw.split("."):
        digits = ""
        for ch in token:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits == "":
            parts.append(0)
        else:
            parts.append(int(digits))
    return tuple(parts)

def version_of(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return ""

timm_version = version_of("timm")
kaolin_version = version_of("kaolin")
needs_timm = not timm_version or parse_tuple(timm_version) < (1, 0, 25)
needs_kaolin = not kaolin_version

print(f"TIMM_VERSION={timm_version}")
print(f"KAOLIN_VERSION={kaolin_version}")
print(f"NEEDS_TIMM={1 if needs_timm else 0}")
print(f"NEEDS_KAOLIN={1 if needs_kaolin else 0}")
PYEOF
)"; then
    warn_or_fail "Failed to evaluate timm/kaolin package state"
    return 0
  fi

  local timm_version kaolin_version needs_timm needs_kaolin
  timm_version="$(printf '%s\n' "${check_output}" | awk -F= '/^TIMM_VERSION=/{print $2}')"
  kaolin_version="$(printf '%s\n' "${check_output}" | awk -F= '/^KAOLIN_VERSION=/{print $2}')"
  needs_timm="$(printf '%s\n' "${check_output}" | awk -F= '/^NEEDS_TIMM=/{print $2}')"
  needs_kaolin="$(printf '%s\n' "${check_output}" | awk -F= '/^NEEDS_KAOLIN=/{print $2}')"

  log "  timm=${timm_version:-<missing>} kaolin=${kaolin_version:-<missing>}"

  local install_failed=0
  local pip_log="/tmp/scenesmith_patch_timm_kaolin.log"

  if [[ "${needs_timm}" == "1" ]]; then
    log "  Enforcing timm>=1.0.25"
    if command -v uv >/dev/null 2>&1; then
      (
        cd "${SCENESMITH_DIR}"
        uv pip install 'timm>=1.0.25'
      ) >"${pip_log}" 2>&1 || install_failed=1
    else
      "${PYTHON_BIN}" -m pip install 'timm>=1.0.25' >"${pip_log}" 2>&1 || install_failed=1
    fi
  fi

  if [[ "${needs_kaolin}" == "1" ]]; then
    log "  Installing kaolin"
    local kaolin_rc=0
    if command -v uv >/dev/null 2>&1; then
      (
        cd "${SCENESMITH_DIR}"
        uv pip install kaolin \
          -f "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html"
      ) >"${pip_log}" 2>&1 || kaolin_rc=$?
      if [[ ${kaolin_rc} -ne 0 ]]; then
        (
          cd "${SCENESMITH_DIR}"
          uv pip install kaolin
        ) >"${pip_log}" 2>&1 || kaolin_rc=$?
      fi
    else
      "${PYTHON_BIN}" -m pip install kaolin \
        -f "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html" \
        >"${pip_log}" 2>&1 || kaolin_rc=$?
      if [[ ${kaolin_rc} -ne 0 ]]; then
        "${PYTHON_BIN}" -m pip install kaolin >"${pip_log}" 2>&1 || kaolin_rc=$?
      fi
    fi
    if [[ ${kaolin_rc} -ne 0 ]]; then
      install_failed=1
    fi
  fi

  if (( install_failed )); then
    tail -n 40 "${pip_log}" || true
    warn_or_fail "Failed to enforce timm/kaolin requirements"
    return 0
  fi

  if [[ "${needs_timm}" == "1" || "${needs_kaolin}" == "1" ]]; then
    log "  OK: timm/kaolin requirements enforced"
  else
    log "  OK: timm/kaolin already satisfy requirements"
  fi
}

validate_sam3_imports() {
  if is_truthy "${SCENESMITH_PATCH_SKIP_IMPORT_VALIDATION:-0}"; then
    log "  SKIP: import validation disabled by SCENESMITH_PATCH_SKIP_IMPORT_VALIDATION"
    return 0
  fi

  if [[ ! -x "${PYTHON_BIN}" ]]; then
    return 0
  fi

  if "${PYTHON_BIN}" - "${SCENESMITH_DIR}" <<'PYEOF'
import importlib
import sys
from pathlib import Path

repo_dir = Path(sys.argv[1]).resolve()
for rel in ("external/SAM3", "external/sam-3d-objects"):
    candidate = repo_dir / rel
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

missing = []
for module_name in ("sam3.model_builder", "sam3d_objects.pipeline.inference_pipeline_pointmap"):
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{module_name}: {exc}")

if missing:
    print("IMPORT_VALIDATION_FAILED")
    for entry in missing:
        print(entry)
    raise SystemExit(1)

print("IMPORT_VALIDATION_OK")
PYEOF
  then
    log "  OK: SAM3 + SAM3D imports validated"
  else
    warn_or_fail "SAM3/SAM3D import validation failed"
  fi
}

ensure_material_dirs
ensure_sam3_repo
patch_indoor_scene_generation
patch_floor_plan_material_fallbacks
patch_parallel_start_method
patch_agents_tool_typing
install_missing_python_modules
ensure_timm_and_kaolin
validate_sam3_imports

log "SceneSmith runtime patch pass completed"
