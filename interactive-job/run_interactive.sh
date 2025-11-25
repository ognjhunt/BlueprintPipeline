#!/bin/bash
set -euo pipefail

echo "[PHYSX] Starting interactive asset generation"
python /app/run_interactive_assets.py
echo "[PHYSX] Done interactive asset generation"
