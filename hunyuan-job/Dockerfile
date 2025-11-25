FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# --- System deps ----------------------------------------------------
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-venv python3-dev \
    git wget curl \
    build-essential cmake ninja-build \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 libegl1 \
    && rm -rf /var/lib/apt/lists/*

# --- Python venv (Python 3.10 from Ubuntu 22.04) --------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# --- PyTorch 2.5.1 + CUDA 12.4 -------------------------------------
# Matches Hunyuan3D 2.1 recommendations.
RUN pip install --no-cache-dir --prefer-binary \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Help CUDA extensions (like custom_rasterizer) compile without a GPU
# visible at build time (L4 / A100-ish archs).
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

# --- Clone Hunyuan3D-2.1 -------------------------------------------
ENV HUNYUAN_REPO_ROOT=/app/Hunyuan3D-2.1

RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git ${HUNYUAN_REPO_ROOT} || true

WORKDIR ${HUNYUAN_REPO_ROOT}

# --- Install Python deps for Hunyuan3D-2.1 --------------------------
# Upstream requirements.txt pins:
#   - bpy==4.0        (Blender; not usable in Cloud Run)
#   - onnxruntime==1.16.3 (old pin, no longer published for some platforms)
#
# We drop JUST those lines, then install a CURRENT onnxruntime explicitly
# so rembg/background removal still works.
RUN sed -e '/^bpy==/d' -e '/^onnxruntime==/d' requirements.txt > /tmp/requirements.no_bpy_onnx.txt \
 && pip install --no-cache-dir --prefer-binary -r /tmp/requirements.no_bpy_onnx.txt \
 && pip install --no-cache-dir --prefer-binary "onnxruntime>=1.17.0"

# Optional: usd-core for GLB -> USDZ conversion (usd_from_gltf CLI)
RUN pip install --no-cache-dir --prefer-binary "usd-core>=24.8"

# --- Patch mesh_utils to make Blender optional ----------------------
# Hunyuan's mesh_utils.py imports bpy at top-level, which crashes if Blender
# is not installed. Here we rewrite that import into a try/except so the
# module can be imported without bpy, and GLB export is simply disabled.
RUN python - << 'PY'
from pathlib import Path

path = Path("hy3dpaint/DifferentiableRenderer/mesh_utils.py")
text = path.read_text()

if "import bpy" in text and "Bpy IO CAN NOT BE Imported!!! GLB export is disabled (no Blender)." not in text:
    text = text.replace(
        "import bpy",
        "try:\\n"
        "    import bpy\\n"
        "except Exception:\\n"
        "    bpy = None\\n"
        "    print('Bpy IO CAN NOT BE Imported!!! GLB export is disabled (no Blender).', flush=True)"
    )
    path.write_text(text)
PY

# --- Build custom rasterizer + inpainting extension -----------------
# IMPORTANT: --no-build-isolation lets setup.py see torch from /opt/venv,
# avoiding "ModuleNotFoundError: No module named 'torch'" during build.
RUN cd ${HUNYUAN_REPO_ROOT}/hy3dpaint/custom_rasterizer && \
    pip install --no-cache-dir --prefer-binary --no-build-isolation -e . && \
    cd ${HUNYUAN_REPO_ROOT}/hy3dpaint/DifferentiableRenderer && \
    bash compile_mesh_painter.sh

# --- Download RealESRGAN weights for super-res in paint pipeline ----
RUN mkdir -p ${HUNYUAN_REPO_ROOT}/hy3dpaint/ckpt && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
         -O ${HUNYUAN_REPO_ROOT}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth

# --- Common runtime env ---------------------------------------------
ENV HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/mnt/gcs/hf-cache \
    TRANSFORMERS_CACHE=/mnt/gcs/hf-cache \
    TORCH_HOME=/tmp/torch-cache \
    CUDA_HOME=/usr/local/cuda \
    CONDA_PREFIX=/usr/local/cuda \
    HUNYUAN_MODEL_PATH="tencent/Hunyuan3D-2.1"

# Make sure Python can see the Hunyuan packages
ENV PYTHONPATH="${HUNYUAN_REPO_ROOT}:${HUNYUAN_REPO_ROOT}/hy3dshape:${HUNYUAN_REPO_ROOT}/hy3dpaint:${PYTHONPATH}"

WORKDIR /app

# --- Job entry scripts ----------------------------------------------
COPY run_hunyuan_from_assets.py /app/run_hunyuan_from_assets.py
COPY run_hunyuan.sh /app/run_hunyuan.sh

RUN chmod +x /app/run_hunyuan.sh

ENTRYPOINT ["/app/run_hunyuan.sh"]
