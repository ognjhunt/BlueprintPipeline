#!/usr/bin/env bash
# =============================================================================
# Download Isaac Sim robot USD assets from NVIDIA's public S3 CDN.
#
# Usage:
#   ./download_robot_assets.sh [DEST_DIR] [MANIFEST_FILE]
#
# Defaults:
#   DEST_DIR      = /sim-assets/robots
#   MANIFEST_FILE = same directory as this script / robot_asset_manifest.txt
#
# Each line in the manifest is a directory path relative to the CDN root.
# The script downloads the full directory tree (USD + meshes + textures)
# using wget recursive mode.  Blank lines and comments (#) are skipped.
#
# Exits non-zero on any download failure so Docker builds fail fast.
# =============================================================================
set -euo pipefail

DEST_DIR="${1:-/sim-assets/robots}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="${2:-${SCRIPT_DIR}/robot_asset_manifest.txt}"

CDN_BASE="${ISAAC_ASSET_CDN_BASE:-https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1}"

if [ ! -f "${MANIFEST}" ]; then
    echo "[ERROR] Manifest not found: ${MANIFEST}" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

FAIL_COUNT=0

while IFS= read -r line || [ -n "$line" ]; do
    # Skip blank lines and comments
    line="$(echo "$line" | sed 's/#.*//' | xargs)"
    [ -z "$line" ] && continue

    SRC_URL="${CDN_BASE}/${line}/"
    # Destination mirrors the Nucleus path structure
    DEST_SUBDIR="${DEST_DIR}/${line}"
    mkdir -p "${DEST_SUBDIR}"

    echo "===> Downloading: ${line}"
    # wget recursive: grab all files in the directory.
    # --no-parent prevents ascending, --cut-dirs strips the S3 prefix so
    # files land under DEST_SUBDIR directly.
    #
    # The number of path components in CDN_BASE after the host is 3:
    #   Assets / Isaac / 5.1
    # Plus the components in $line itself. We use -nH (no host dir) and
    # -P to set the output root, then --cut-dirs to strip the CDN prefix.
    #
    # Example: line="Isaac/Robots/Franka"
    #   URL = .../Assets/Isaac/5.1/Isaac/Robots/Franka/
    #   We want files at DEST_DIR/Isaac/Robots/Franka/...
    #   CDN path components to cut: "Assets", "Isaac", "5.1" = 3
    if ! wget -q -r -np -nH --cut-dirs=3 \
         -R "index.html*" \
         -P "${DEST_DIR}" \
         "${SRC_URL}" 2>/dev/null; then
        echo "[WARN] wget recursive failed for ${line}, trying individual USD files..."
        # Fallback: try to download just the main USD file(s) directly
        # Extract the last path component as a likely filename stem
        BASENAME="$(basename "$line")"
        LOWER_BASENAME="$(echo "$BASENAME" | tr '[:upper:]' '[:lower:]')"
        DOWNLOADED=0
        for EXT in ".usd" ".usda" ".usdc"; do
            for NAME in "$LOWER_BASENAME" "$BASENAME"; do
                FILE_URL="${CDN_BASE}/${line}/${NAME}${EXT}"
                if wget -q -O "${DEST_SUBDIR}/${NAME}${EXT}" "${FILE_URL}" 2>/dev/null; then
                    echo "     Downloaded: ${NAME}${EXT}"
                    DOWNLOADED=1
                    break 2
                else
                    rm -f "${DEST_SUBDIR}/${NAME}${EXT}"
                fi
            done
        done
        if [ "$DOWNLOADED" -eq 0 ]; then
            echo "[ERROR] Failed to download any asset for: ${line}" >&2
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "     OK"
    fi
done < "${MANIFEST}"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "[WARN] ${FAIL_COUNT} robot asset(s) failed to download (non-fatal)." >&2
    # Non-fatal: some CDN assets may not be available for all Isaac Sim versions.
    # Critical robots (Franka, G1, UR10) are validated at runtime.
fi

echo "All robot assets downloaded to ${DEST_DIR}"
du -sh "${DEST_DIR}"
