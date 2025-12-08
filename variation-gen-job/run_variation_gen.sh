#!/usr/bin/env bash
# =============================================================================
# Variation Asset Generator - Entry Point Script
# =============================================================================
# Generates reference images for variation assets using Gemini 3.0 Pro Image (Nano Banana Pro).
#
# Environment Variables:
#   BUCKET              - GCS bucket name
#   SCENE_ID            - Scene identifier
#   REPLICATOR_PREFIX   - Path to replicator bundle (default: scenes/{SCENE_ID}/replicator)
#   VARIATION_ASSETS_PREFIX - Output path (default: scenes/{SCENE_ID}/variation_assets)
#   GEMINI_API_KEY      - Gemini API key (required)
#   MAX_ASSETS          - Optional limit on number of assets to generate
#   PRIORITY_FILTER     - Optional filter: "required", "recommended", or "optional"
#   DRY_RUN             - Set to "1" or "true" to skip actual generation
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "Variation Asset Generator"
echo "=============================================="
echo "Scene ID: ${SCENE_ID:-<not set>}"
echo "Bucket: ${BUCKET:-<not set>}"
echo "Replicator prefix: ${REPLICATOR_PREFIX:-scenes/${SCENE_ID}/replicator}"
echo "Output prefix: ${VARIATION_ASSETS_PREFIX:-scenes/${SCENE_ID}/variation_assets}"
echo "=============================================="

# Verify required environment variables
if [ -z "${SCENE_ID:-}" ]; then
    echo "ERROR: SCENE_ID is required" >&2
    exit 1
fi

if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "ERROR: GEMINI_API_KEY is required" >&2
    exit 1
fi

# Run the Python script
exec python3 /app/generate_variation_assets.py
