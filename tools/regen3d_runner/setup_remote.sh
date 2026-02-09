#!/usr/bin/env bash
# Bootstrap 3D-RE-GEN on a remote GPU VM.
#
# This script is uploaded and executed on the VM to set up the 3D-RE-GEN
# environment. It is idempotent — safe to run multiple times.
#
# Usage (called by runner.py via SSH):
#   bash setup_remote.sh /home/user/3D-RE-GEN
#
# Reference:
#   https://github.com/cgtuebingen/3D-RE-GEN
#   https://github.com/cgtuebingen/3D-RE-GEN/blob/main/INSTALLATION.md

set -euo pipefail

REPO_DIR="${1:-$HOME/3D-RE-GEN}"
VENV_DIR="${REPO_DIR}/venv_py310"
SENTINEL_FILE="${REPO_DIR}/.bp_regen3d_setup_ok"

echo "[SETUP] 3D-RE-GEN bootstrap starting..."
echo "[SETUP] Repo dir: ${REPO_DIR}"
echo "[SETUP] Venv dir: ${VENV_DIR}"

# ─── Step 1: Clone or update the repository ─────────────────────────────────
if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "[SETUP] Cloning 3D-RE-GEN repository..."
    git clone --recursive https://github.com/cgtuebingen/3D-RE-GEN.git "${REPO_DIR}"
else
    echo "[SETUP] Repository exists, pulling latest..."
    cd "${REPO_DIR}"
    git pull --ff-only 2>/dev/null || echo "[SETUP] Pull skipped (detached HEAD or conflict)"
    git submodule update --init --recursive
fi

cd "${REPO_DIR}"

# ─── Step 2: Install mamba if not present ────────────────────────────────────
if ! command -v mamba &>/dev/null; then
    if ! command -v conda &>/dev/null; then
        echo "[SETUP] Installing Miniforge (mamba)..."
        INSTALLER="/tmp/miniforge.sh"
        curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "${INSTALLER}"
        bash "${INSTALLER}" -b -p "$HOME/miniforge3"
        rm -f "${INSTALLER}"
        export PATH="$HOME/miniforge3/bin:$PATH"
        echo "[SETUP] Miniforge installed"
    else
        echo "[SETUP] conda found, installing mamba into base..."
        conda install -y -n base -c conda-forge mamba
    fi
fi

# Ensure mamba/conda is on PATH
for p in "$HOME/miniforge3/bin" "$HOME/mambaforge/bin" "$HOME/miniconda3/bin" "$HOME/anaconda3/bin"; do
    [ -d "$p" ] && export PATH="$p:$PATH"
done

# ─── Step 3: Create Python 3.10 environment ─────────────────────────────────
if [ ! -d "${VENV_DIR}" ]; then
    echo "[SETUP] Creating mamba environment (Python 3.10)..."
    mamba create -p "${VENV_DIR}" python=3.10 -y
else
    echo "[SETUP] Mamba environment already exists"
fi

# ─── Step 4: Install dependencies ───────────────────────────────────────────
echo "[SETUP] Installing PyTorch + CUDA 12.1..."
mamba run -p "${VENV_DIR}" pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "[SETUP] Installing PyTorch3D..."
mamba run -p "${VENV_DIR}" pip install --no-cache-dir fvcore iopath
mamba run -p "${VENV_DIR}" pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/pytorch3d.git"

echo "[SETUP] Installing requirements.txt..."
if [ -f "${REPO_DIR}/requirements.txt" ]; then
    mamba run -p "${VENV_DIR}" pip install --no-cache-dir \
        -r "${REPO_DIR}/requirements.txt"
fi

# Force numpy < 2.0 for compatibility
mamba run -p "${VENV_DIR}" pip install --no-cache-dir "numpy<2.0"

# Install google-genai for nanoBanana inpainting
mamba run -p "${VENV_DIR}" pip install --no-cache-dir google-genai

# ─── Step 5: Download SAM weights ───────────────────────────────────────────
SAM_WEIGHTS="${REPO_DIR}/segmentor/sam_vit_h_4b8939.pth"
if [ ! -f "${SAM_WEIGHTS}" ]; then
    echo "[SETUP] Downloading SAM ViT-H weights (2.56 GB)..."
    mkdir -p "${REPO_DIR}/segmentor"
    wget -q --show-progress \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
        -O "${SAM_WEIGHTS}"
    echo "[SETUP] SAM weights downloaded"
else
    echo "[SETUP] SAM weights already present"
fi

# ─── Step 6: Verify GPU access ──────────────────────────────────────────────
echo "[SETUP] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
    echo "[SETUP] WARNING: nvidia-smi failed - GPU may not be available"
}

# ─── Step 7: Quick sanity check ─────────────────────────────────────────────
echo "[SETUP] Verifying PyTorch CUDA..."
mamba run -p "${VENV_DIR}" python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

date -u +"%Y-%m-%dT%H:%M:%SZ" > "${SENTINEL_FILE}"
echo "[SETUP] Wrote setup sentinel: ${SENTINEL_FILE}"

echo "[SETUP] 3D-RE-GEN bootstrap complete!"
