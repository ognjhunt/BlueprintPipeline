#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-start}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

load_env_file() {
  local file_path="$1"
  if [[ -z "${file_path}" ]]; then
    return 0
  fi
  local resolved_path="${file_path/#\~/${HOME}}"
  if [[ ! -f "${resolved_path}" ]]; then
    echo "Env file not found: ${resolved_path}" >&2
    return 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "${resolved_path}"
  set +a
  echo "Loaded env: ${resolved_path}"
}

if [[ "${TEXT_BACKEND_SKIP_DOTENV:-0}" != "1" ]]; then
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    load_env_file "${REPO_ROOT}/.env"
  fi
fi

if [[ -n "${TEXT_BACKEND_ENV_FILE:-}" ]]; then
  load_env_file "${TEXT_BACKEND_ENV_FILE}"
elif [[ -f "${REPO_ROOT}/configs/text_backends.env" ]]; then
  load_env_file "${REPO_ROOT}/configs/text_backends.env"
fi

PYTHON_BIN=${PYTHON_BIN:-python3}

RUN_DIR=${TEXT_BACKEND_RUN_DIR:-/tmp/blueprint-text-backends}
LOG_DIR=${TEXT_BACKEND_LOG_DIR:-${RUN_DIR}/logs}
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

SCENESMITH_PORT=${SCENESMITH_PORT:-8081}
SAGE_PORT=${SAGE_PORT:-8082}

SCENESMITH_PID_FILE="${RUN_DIR}/scenesmith-service.pid"
SAGE_PID_FILE="${RUN_DIR}/sage-service.pid"
SCENESMITH_LOG="${LOG_DIR}/scenesmith-service.log"
SAGE_LOG="${LOG_DIR}/sage-service.log"

is_running() {
  local pid_file="$1"
  if [[ ! -f "${pid_file}" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    return 1
  fi
  kill -0 "${pid}" >/dev/null 2>&1
}

check_runtime_dependencies() {
  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import flask  # noqa: F401
PY
  then
    echo "Missing python dependency: flask" >&2
    echo "Run ./scripts/setup_text_backend_services.sh and set PYTHON_BIN to that venv." >&2
    return 1
  fi
}

start_one() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  local port="$4"
  local mode_env_name="$5"
  local module_path="$6"

  if is_running "${pid_file}"; then
    echo "${name} already running (pid=$(cat "${pid_file}"))"
    return 0
  fi

  local mode
  mode="${!mode_env_name:-internal}"

  if [[ "${name}" == "scenesmith-service" ]] && [[ "${mode}" == "paper" || "${mode}" == "paper_stack" ]]; then
    local paper_repo_dir
    paper_repo_dir="${SCENESMITH_PAPER_REPO_DIR:-${HOME}/scenesmith}"
    paper_repo_dir="${paper_repo_dir/#\~/${HOME}}"
    if [[ ! -d "${paper_repo_dir}" ]]; then
      echo "SceneSmith paper repo not found at ${paper_repo_dir}; falling back to internal mode." >&2
      mode="internal"
    else
      export SCENESMITH_PAPER_REPO_DIR="${paper_repo_dir}"
    fi
  fi

  echo "Starting ${name} on port ${port} (mode=${mode})"
  nohup env PORT="${port}" "${mode_env_name}=${mode}" "${PYTHON_BIN}" "${module_path}" >"${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${pid_file}"
  sleep 1

  if kill -0 "${pid}" >/dev/null 2>&1; then
    echo "Started ${name} (pid=${pid})"
  else
    echo "Failed to start ${name}; check ${log_file}" >&2
    return 1
  fi
}

stop_one() {
  local name="$1"
  local pid_file="$2"

  if ! is_running "${pid_file}"; then
    echo "${name} not running"
    rm -f "${pid_file}"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}")"
  echo "Stopping ${name} (pid=${pid})"
  kill "${pid}" >/dev/null 2>&1 || true
  sleep 1
  if kill -0 "${pid}" >/dev/null 2>&1; then
    echo "Force killing ${name} (pid=${pid})"
    kill -9 "${pid}" >/dev/null 2>&1 || true
  fi
  rm -f "${pid_file}"
}

status_one() {
  local name="$1"
  local pid_file="$2"
  local port="$3"
  local log_file="$4"

  if is_running "${pid_file}"; then
    echo "${name}: running pid=$(cat "${pid_file}") port=${port} log=${log_file}"
  else
    echo "${name}: stopped"
  fi
}

case "${ACTION}" in
  start)
    check_runtime_dependencies
    start_one "scenesmith-service" "${SCENESMITH_PID_FILE}" "${SCENESMITH_LOG}" "${SCENESMITH_PORT}" "SCENESMITH_SERVICE_MODE" "${REPO_ROOT}/scenesmith-service/scenesmith_service.py"
    start_one "sage-service" "${SAGE_PID_FILE}" "${SAGE_LOG}" "${SAGE_PORT}" "SAGE_SERVICE_MODE" "${REPO_ROOT}/sage-service/sage_service.py"
    ;;
  stop)
    stop_one "scenesmith-service" "${SCENESMITH_PID_FILE}"
    stop_one "sage-service" "${SAGE_PID_FILE}"
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  status)
    status_one "scenesmith-service" "${SCENESMITH_PID_FILE}" "${SCENESMITH_PORT}" "${SCENESMITH_LOG}"
    status_one "sage-service" "${SAGE_PID_FILE}" "${SAGE_PORT}" "${SAGE_LOG}"
    ;;
  logs)
    echo "--- scenesmith-service (${SCENESMITH_LOG}) ---"
    tail -n 80 "${SCENESMITH_LOG}" 2>/dev/null || true
    echo "--- sage-service (${SAGE_LOG}) ---"
    tail -n 80 "${SAGE_LOG}" 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}" >&2
    exit 1
    ;;
esac
