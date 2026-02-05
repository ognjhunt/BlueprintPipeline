#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# Ensure a host-side NVIDIA-backed Xorg display is running for Isaac Sim camera
# rendering. This script runs on the VM host (not inside the container).
#
# Usage:
#   bash scripts/vm-start-xorg.sh
#
# Optional env vars:
#   VM_XORG_DISPLAY_NUM=99      # display number (default 99 -> :99)
#   VM_XORG_WIDTH=1280          # virtual screen width
#   VM_XORG_HEIGHT=720          # virtual screen height
#   VM_XORG_CONFIG_PATH=/etc/X11/xorg.conf
#   VM_XORG_LOG_PATH=/tmp/vm-xorg-99.log
#   VM_XORG_REFRESH_CONFIG=0    # set 1 to rewrite xorg.conf
# =============================================================================

DISPLAY_NUM=${VM_XORG_DISPLAY_NUM:-99}
DISPLAY_VALUE=":${DISPLAY_NUM}"
XSOCK_PATH="/tmp/.X11-unix/X${DISPLAY_NUM}"
XORG_CONFIG_PATH=${VM_XORG_CONFIG_PATH:-/etc/X11/xorg.conf}
XORG_LOG_PATH=${VM_XORG_LOG_PATH:-/tmp/vm-xorg-${DISPLAY_NUM}.log}
XORG_WIDTH=${VM_XORG_WIDTH:-1280}
XORG_HEIGHT=${VM_XORG_HEIGHT:-720}
VM_XORG_REFRESH_CONFIG=${VM_XORG_REFRESH_CONFIG:-0}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[vm-start-xorg] ERROR: nvidia-smi not found; cannot determine GPU bus ID." >&2
  exit 1
fi

if ! command -v Xorg >/dev/null 2>&1; then
  echo "[vm-start-xorg] ERROR: Xorg not found on host." >&2
  exit 1
fi

if pgrep -fa "Xorg ${DISPLAY_VALUE}" >/dev/null 2>&1 && [ -S "${XSOCK_PATH}" ]; then
  echo "[vm-start-xorg] Xorg already running on ${DISPLAY_VALUE} (${XSOCK_PATH})."
  exit 0
fi

if pgrep -fa "Xorg ${DISPLAY_VALUE}" >/dev/null 2>&1 && [ ! -S "${XSOCK_PATH}" ]; then
  echo "[vm-start-xorg] Found stale Xorg process on ${DISPLAY_VALUE}; restarting..."
  sudo pkill -f "Xorg ${DISPLAY_VALUE}" || true
  sleep 1
fi

if [ ! -f "${XORG_CONFIG_PATH}" ] || [ "${VM_XORG_REFRESH_CONFIG}" = "1" ]; then
  GPU_BUS_ID_RAW="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr -d '[:space:]')"
  if [ -z "${GPU_BUS_ID_RAW}" ]; then
    echo "[vm-start-xorg] ERROR: Could not detect GPU PCI bus ID." >&2
    exit 1
  fi

  IFS=':.' read -r _domain_hex _bus_hex _dev_hex _func_dec <<< "${GPU_BUS_ID_RAW}"
  if [ -z "${_bus_hex:-}" ] || [ -z "${_dev_hex:-}" ] || [ -z "${_func_dec:-}" ]; then
    echo "[vm-start-xorg] ERROR: Unexpected GPU bus format: ${GPU_BUS_ID_RAW}" >&2
    exit 1
  fi

  GPU_BUS_DEC=$((16#${_bus_hex}))
  GPU_DEV_DEC=$((16#${_dev_hex}))
  GPU_FUNC_DEC=$((10#${_func_dec}))
  XORG_BUS_ID="PCI:${GPU_BUS_DEC}:${GPU_DEV_DEC}:${GPU_FUNC_DEC}"

  echo "[vm-start-xorg] Writing ${XORG_CONFIG_PATH} with BusID=${XORG_BUS_ID}"
  sudo tee "${XORG_CONFIG_PATH}" >/dev/null <<EOF
Section "ServerLayout"
    Identifier "Layout0"
    Screen 0 "Screen0" 0 0
EndSection

Section "Device"
    Identifier "GPU0"
    Driver "nvidia"
    BusID "${XORG_BUS_ID}"
    Option "AllowEmptyInitialConfiguration" "true"
    Option "UseDisplayDevice" "None"
EndSection

Section "Screen"
    Identifier "Screen0"
    Device "GPU0"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Virtual ${XORG_WIDTH} ${XORG_HEIGHT}
    EndSubSection
EndSection
EOF
fi

echo "[vm-start-xorg] Starting Xorg on ${DISPLAY_VALUE} (log: ${XORG_LOG_PATH})"
sudo bash -lc "nohup Xorg ${DISPLAY_VALUE} -config ${XORG_CONFIG_PATH} -noreset +extension GLX +extension RANDR +extension RENDER > ${XORG_LOG_PATH} 2>&1 &"
sleep 3

if [ ! -S "${XSOCK_PATH}" ] || ! pgrep -fa "Xorg ${DISPLAY_VALUE}" >/dev/null 2>&1; then
  echo "[vm-start-xorg] ERROR: Failed to start Xorg on ${DISPLAY_VALUE}" >&2
  if [ -f "${XORG_LOG_PATH}" ]; then
    echo "[vm-start-xorg] Last 40 log lines:" >&2
    tail -40 "${XORG_LOG_PATH}" >&2 || true
  fi
  exit 1
fi

echo "[vm-start-xorg] READY ${DISPLAY_VALUE} (${XSOCK_PATH})"
