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
# Add known conda/mamba paths to PATH FIRST (previous installs survive across SSH sessions)
for p in "$HOME/miniforge3/bin" "$HOME/mambaforge/bin" "$HOME/miniconda3/bin" "$HOME/anaconda3/bin"; do
    [ -d "$p" ] && export PATH="$p:$PATH"
done

if ! command -v mamba &>/dev/null; then
    if ! command -v conda &>/dev/null; then
        echo "[SETUP] Installing Miniforge (mamba)..."
        INSTALLER="/tmp/miniforge.sh"
        curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "${INSTALLER}"
        bash "${INSTALLER}" -b -u -p "$HOME/miniforge3"
        rm -f "${INSTALLER}"
        export PATH="$HOME/miniforge3/bin:$PATH"
        echo "[SETUP] Miniforge installed"
    else
        echo "[SETUP] conda found, installing mamba into base..."
        conda install -y -n base -c conda-forge mamba
    fi
fi

# ─── Step 3: Create Python 3.10 environment ─────────────────────────────────
if [ ! -d "${VENV_DIR}" ]; then
    echo "[SETUP] Creating mamba environment (Python 3.10)..."
    mamba create -p "${VENV_DIR}" python=3.10 -y
else
    echo "[SETUP] Mamba environment already exists"
fi

# Fast idempotency guard: if base runtime is healthy, skip heavyweight bootstrap.
BASE_SETUP_HEALTHY=0
if [ -x "${VENV_DIR}/bin/python" ] && \
   PYTHONNOUSERSITE=1 "${VENV_DIR}/bin/python" -c \
   "import torch, pytorch3d, pytorch3d._C" >/dev/null 2>&1; then
    BASE_SETUP_HEALTHY=1
    if [ ! -f "${SENTINEL_FILE}" ]; then
        date -u +"%Y-%m-%dT%H:%M:%SZ" > "${SENTINEL_FILE}"
        echo "[SETUP] Recreated missing base setup sentinel: ${SENTINEL_FILE}"
    fi
    echo "[SETUP] Base environment already healthy; skipping base dependency bootstrap."
fi

if [ "${BASE_SETUP_HEALTHY}" -eq 0 ]; then
    # ─── Step 4: Install dependencies ───────────────────────────────────────
    # Detect CUDA version and select matching PyTorch index
    CUDA_MAJOR=$(nvcc --version 2>/dev/null | grep -oP 'release \K\d+' || echo "12")
    echo "[SETUP] Detected CUDA major version: ${CUDA_MAJOR}"
    if [ "${CUDA_MAJOR}" -ge 13 ]; then
        # CUDA 13.x — use cu124 builds (forward compatible)
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        echo "[SETUP] Using PyTorch CUDA 12.4 index (forward-compatible with CUDA ${CUDA_MAJOR})"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        echo "[SETUP] Using PyTorch CUDA 12.1 index"
    fi

    echo "[SETUP] Installing PyTorch..."
    mamba run -p "${VENV_DIR}" pip install --no-cache-dir \
        torch torchvision torchaudio --index-url "${TORCH_INDEX}"

    echo "[SETUP] Installing PyTorch3D prerequisites..."
    mamba run -p "${VENV_DIR}" pip install --no-cache-dir fvcore iopath

    if PYTHONNOUSERSITE=1 "${VENV_DIR}/bin/python" -c \
       "import pytorch3d, pytorch3d._C" >/dev/null 2>&1; then
        echo "[SETUP] PyTorch3D already importable; skipping source rebuild."
    else
        # PyTorch3D build strategy:
        # 1. Use system CUDA (nvcc) with CUDA_HOME=/usr/local/cuda
        # 2. Bypass PyTorch's CUDA version mismatch check (torch cu124 vs system nvcc 13.x)
        # 3. Patch PyTorch3D setup.py to apply CUDA 13 visibility flags regardless of torch.version.cuda
        # 4. Disable conda's bundled linker (too old for CUDA 13 object files)
        echo "[SETUP] Building PyTorch3D from source with system CUDA..."

        # Patch PyTorch's CUDA mismatch check to allow system nvcc != torch CUDA version
        CPP_EXT="${VENV_DIR}/lib/python3.10/site-packages/torch/utils/cpp_extension.py"
        if [ -f "${CPP_EXT}" ] && ! grep -q 'PYTORCH_SKIP_CUDA_MISMATCH_CHECK' "${CPP_EXT}"; then
            echo "[SETUP] Patching PyTorch CUDA version check..."
            cp "${CPP_EXT}" "${CPP_EXT}.bak"
            python3 -c "
content = open('${CPP_EXT}').read()
old = 'if cuda_ver.major != torch_cuda_version.major:'
new = 'if cuda_ver.major != torch_cuda_version.major and os.environ.get(\"PYTORCH_SKIP_CUDA_MISMATCH_CHECK\") not in [\"1\", \"true\", \"yes\"]:'
if old in content:
    open('${CPP_EXT}', 'w').write(content.replace(old, new))
    print('[SETUP] PyTorch CUDA check patched')
"
        fi

        # Disable conda's bundled linker (causes 'hidden symbol' errors with CUDA 13 objects)
        CONDA_LD="${VENV_DIR}/compiler_compat/ld"
        if [ -f "${CONDA_LD}" ] && [ ! -f "${CONDA_LD}.bak" ]; then
            echo "[SETUP] Disabling conda linker (using system ld instead)..."
            mv "${CONDA_LD}" "${CONDA_LD}.bak"
        fi

        # Clone PyTorch3D, patch setup.py for CUDA 13 visibility flags, then build
        PT3D_BUILD="/tmp/pytorch3d_build"
        rm -rf "${PT3D_BUILD}"
        git clone --filter=blob:none https://github.com/facebookresearch/pytorch3d.git "${PT3D_BUILD}"

        # Patch: force CUDA 13 visibility flags even when torch.version.cuda reports 12.x
        python3 -c "
f = '${PT3D_BUILD}/setup.py'
content = open(f).read()
old = 'if major >= 13:'
new = 'if major >= 12:  # Patched: force CUDA 13 visibility flags (system nvcc is 13.x)'
if old in content:
    open(f, 'w').write(content.replace(old, new))
    print('[SETUP] PyTorch3D setup.py patched for CUDA 13 visibility')
"

        cd "${PT3D_BUILD}"
        FORCE_CUDA=1 \
        PYTORCH_SKIP_CUDA_MISMATCH_CHECK=1 \
        CUDA_HOME=/usr/local/cuda \
        TORCH_CUDA_ARCH_LIST="8.9" \
        PATH="/usr/local/cuda/bin:${PATH}" \
        "${VENV_DIR}/bin/pip" install --no-cache-dir --no-build-isolation . || {
            echo "[SETUP] ERROR: PyTorch3D build failed"
            tail -30 /tmp/pytorch3d_build.log 2>/dev/null || true
            exit 1
        }
        cd "${REPO_DIR}"
        rm -rf "${PT3D_BUILD}"
    fi

    echo "[SETUP] Bootstrapping pip inside conda env..."
    PYTHONNOUSERSITE=1 "${VENV_DIR}/bin/python" -m ensurepip --upgrade 2>/dev/null || true

    echo "[SETUP] Installing requirements.txt..."
    if [ -f "${REPO_DIR}/requirements.txt" ]; then
        PYTHONNOUSERSITE=1 "${VENV_DIR}/bin/python" -m pip install --no-cache-dir \
            -r "${REPO_DIR}/requirements.txt"
    fi

    # Install segmentor-specific requirements
    if [ -f "${REPO_DIR}/segmentor/requirements.txt" ]; then
        echo "[SETUP] Installing segmentor requirements..."
        PYTHONNOUSERSITE=1 "${VENV_DIR}/bin/python" -m pip install --no-cache-dir \
            -r "${REPO_DIR}/segmentor/requirements.txt"
    fi

    # Force numpy < 2.0 for compatibility
    PYTHONNOUSERSITE=1 "${VENV_DIR}/bin/python" -m pip install --no-cache-dir "numpy<2.0"

    # Install google-genai for nanoBanana inpainting
    PYTHONNOUSERSITE=1 "${VENV_DIR}/bin/python" -m pip install --no-cache-dir google-genai

    # Install rembg for background removal
    PYTHONNOUSERSITE=1 "${VENV_DIR}/bin/python" -m pip install --no-cache-dir rembg segment_anything

    # ─── Step 5: Download SAM weights ───────────────────────────────────────
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

    # ─── Step 6: Verify GPU access ──────────────────────────────────────────
    echo "[SETUP] Checking GPU..."
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
        echo "[SETUP] WARNING: nvidia-smi failed - GPU may not be available"
    }

    # ─── Step 7: Quick sanity check ─────────────────────────────────────────
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
fi

# ─── Step 8: SAM3 segmentation environment (Python 3.12) ─────────────────
# SAM3 requires Python 3.12+, PyTorch 2.7+, and transformers 5.1+.
# We create a separate venv so the 3D-RE-GEN venv (Python 3.10) is untouched.

SAM3_VENV="${REPO_DIR}/venv_sam3"
SAM3_SENTINEL="${REPO_DIR}/.bp_sam3_setup_ok"

if [ -f "${SAM3_SENTINEL}" ]; then
    echo "[SETUP] SAM3 environment already set up"
else
    echo "[SETUP] Creating SAM3 environment (Python 3.12)..."

    if [ ! -d "${SAM3_VENV}" ]; then
        mamba create -p "${SAM3_VENV}" python=3.12 -y
    fi

    echo "[SETUP] Installing SAM3 dependencies (PyTorch 2.7 + CUDA 12.6)..."
    PYTHONNOUSERSITE=1 "${SAM3_VENV}/bin/pip" install --no-cache-dir \
        torch==2.7.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126

    echo "[SETUP] Installing HuggingFace transformers (SAM3 support)..."
    PYTHONNOUSERSITE=1 "${SAM3_VENV}/bin/pip" install --no-cache-dir \
        "transformers>=5.1.0" \
        pillow accelerate timm

    echo "[SETUP] Pre-downloading SAM3 model weights (~3.4 GB)..."
    PYTHONNOUSERSITE=1 "${SAM3_VENV}/bin/python" -c "
from transformers import Sam3Model, Sam3Processor
print('[SETUP] Downloading SAM3 model...')
Sam3Model.from_pretrained('facebook/sam3')
Sam3Processor.from_pretrained('facebook/sam3')
print('[SETUP] SAM3 model cached successfully')
" || {
        echo "[SETUP] WARNING: SAM3 model download failed — will retry at first use"
    }

    echo "[SETUP] Pre-downloading DepthAnythingV2 model..."
    PYTHONNOUSERSITE=1 "${SAM3_VENV}/bin/python" -c "
from transformers import pipeline
print('[SETUP] Downloading DepthAnythingV2...')
pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Large-hf')
print('[SETUP] DepthAnythingV2 cached successfully')
" || {
        echo "[SETUP] WARNING: DepthAnythingV2 download failed — will retry at first use"
    }

    # Verify SAM3 venv works
    echo "[SETUP] Verifying SAM3 venv..."
    PYTHONNOUSERSITE=1 "${SAM3_VENV}/bin/python" -c "
import torch
print(f'SAM3 venv: Python {__import__(\"sys\").version}')
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
import transformers
print(f'Transformers {transformers.__version__}')
"

    date -u +"%Y-%m-%dT%H:%M:%SZ" > "${SAM3_SENTINEL}"
    echo "[SETUP] SAM3 environment ready (sentinel: ${SAM3_SENTINEL})"
fi

echo "[SETUP] 3D-RE-GEN bootstrap complete!"
