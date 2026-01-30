#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# One-time VM setup for BlueprintPipeline GPU instances.
# Installs all Python dependencies so the pipeline can run immediately
# after the VM boots and the Docker container starts.
#
# Usage:
#   ssh into VM, cd ~/BlueprintPipeline, then:
#   bash scripts/setup-vm.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[setup-vm] Installing pipeline Python dependencies..."
pip3 install --quiet --upgrade pip

# Core pipeline requirements
pip3 install --quiet -r "${REPO_ROOT}/genie-sim-local-job/requirements.txt"

# GCP / Firebase / telemetry deps used by the pipeline runner
pip3 install --quiet \
    google-cloud-storage \
    google-cloud-firestore \
    google-cloud-secret-manager \
    google-api-core \
    google-auth \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-gcp-trace \
    opentelemetry-exporter-otlp \
    opentelemetry-exporter-otlp-proto-grpc \
    opentelemetry-propagator-gcp \
    PyYAML \
    Pillow \
    scipy

# Ensure scripts are executable
chmod +x "${REPO_ROOT}/tools/geniesim_adapter/deployment/start_geniesim_server.sh"
chmod +x "${REPO_ROOT}/tools/geniesim_adapter/deployment/bootstrap_geniesim_runtime.sh"

echo "[setup-vm] Done. Python deps installed for pipeline execution."
echo "[setup-vm] To start the server:  cd ${REPO_ROOT} && sudo docker compose -f docker-compose.geniesim-server.yaml up -d"
echo "[setup-vm] To run the pipeline:  GENIESIM_HOST=localhost GENIESIM_PORT=50051 GEMINI_API_KEY=<key> python3 tools/run_local_pipeline.py --scene-dir test_scenes/scenes/lightwheel_kitchen --steps genie-sim-submit"
