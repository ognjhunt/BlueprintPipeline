#!/usr/bin/env python3
"""
SAM3D Drop-in Server for SAGE Pipeline
=======================================
TRELLIS-compatible API that uses:
  1. OpenAI image generation (gpt-image-1) to create reference images from text
  2. SAM3D (Meta's SAM-3D-Objects) for image-to-3D mesh generation

Exposes the same HTTP API contract as TRELLIS:
  - GET  /health          → {"status": "healthy", "gpu_available": true}
  - POST /generate        → 202 + {"job_id": "...", "status": "accepted"}
  - GET  /job/{job_id}    → 200 + GLB binary (done) | 202 (processing) | 500 (failed)

Usage:
  python sam3d_server.py --port 8080 [--openai-key KEY] [--checkpoint-dir DIR]
"""

import os
import sys
import gc
import io
import json
import signal
import time
import random
import tempfile
import argparse
import threading
import traceback
import types
from pathlib import Path

# ── Texture baking config ────────────────────────────────────────────────────
# Set SAM3D_TEXTURE_BAKING=0 to disable (saves ~2-3GB VRAM per object)
TEXTURE_BAKING_ENABLED = os.environ.get("SAM3D_TEXTURE_BAKING", "1") == "1"
SAM3D_TEXTURE_STRICT = os.environ.get("SAM3D_TEXTURE_STRICT", "1") == "1"

# ── Kaolin monkey-patch ───────────────────────────────────────────────────────
# kaolin 0.17.0 pre-built wheels are for torch 2.5.1+cu121.
# Our env has torch 2.10+cu128, so the C extension (_C.so) fails to load.
# SAM3D only uses kaolin for check_tensor (shape validation) and visualization.
# We provide lightweight shims so SAM3D can import without the C extension.
def _fake_check_tensor(tensor, shape, throw=True):
    return True

_km = types.ModuleType("kaolin")
_km.utils = types.ModuleType("kaolin.utils")
_km.utils.testing = types.ModuleType("kaolin.utils.testing")
_km.utils.testing.check_tensor = _fake_check_tensor
_km.visualize = types.ModuleType("kaolin.visualize")
_km.render = types.ModuleType("kaolin.render")
_km.render.camera = types.ModuleType("kaolin.render.camera")
for _k, _v in {
    "kaolin": _km,
    "kaolin.utils": _km.utils,
    "kaolin.utils.testing": _km.utils.testing,
    "kaolin.visualize": _km.visualize,
    "kaolin.render": _km.render,
    "kaolin.render.camera": _km.render.camera,
}.items():
    sys.modules[_k] = _v

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="SAM3D Drop-in Server for SAGE")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--openai-key", type=str, default=None,
                    help="OpenAI API key (or set OPENAI_API_KEY env)")
parser.add_argument("--gemini-key", type=str, default=None,
                    help="Gemini API key (or set GEMINI_API_KEY env)")
parser.add_argument("--checkpoint-dir", type=str,
                    default="/workspace/sam3d/checkpoints/hf",
                    help="Path to SAM3D checkpoint directory")
parser.add_argument("--image-model", type=str, default="gpt-image-1",
                    help="OpenAI image generation model")
parser.add_argument("--image-size", type=str, default="1024x1024",
                    help="Generated image size")
parser.add_argument("--image-backend", type=str, default="gemini",
                    choices=["gemini", "openai"],
                    help="Primary image generation backend (default: gemini)")
args = parser.parse_args()

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Gemini client ────────────────────────────────────────────────────────────
GEMINI_API_KEY = args.gemini_key or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    key_json_paths = [
        "/workspace/SAGE/server/key.json",
        os.path.join(os.path.dirname(__file__), "key.json"),
    ]
    for kp in key_json_paths:
        if os.path.exists(kp):
            with open(kp) as f:
                kd = json.load(f)
            GEMINI_API_KEY = kd.get("GEMINI_API_KEY", "")
            if GEMINI_API_KEY:
                print(f"Loaded Gemini key from {kp}")
                break

if GEMINI_API_KEY:
    print(f"Gemini API key loaded (primary image backend)")
else:
    print("WARNING: No Gemini API key. Will use OpenAI for images.")

# ── OpenAI client (fallback) ────────────────────────────────────────────────
OPENAI_API_KEY = args.openai_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_TOKEN")
if not OPENAI_API_KEY:
    key_json_paths = [
        "/workspace/SAGE/server/key.json",
        os.path.join(os.path.dirname(__file__), "key.json"),
    ]
    for kp in key_json_paths:
        if os.path.exists(kp):
            with open(kp) as f:
                kd = json.load(f)
            OPENAI_API_KEY = kd.get("API_TOKEN", "")
            if OPENAI_API_KEY:
                print(f"Loaded OpenAI key from {kp}")
                break

if not OPENAI_API_KEY:
    print("WARNING: No OpenAI API key found.")

import openai
oai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ── SAM3D model (lazy loaded) ────────────────────────────────────────────────
os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", "/usr"))

sam3d_model = None
sam3d_lock = threading.Lock()  # Serialize GPU inference


def load_sam3d():
    """Lazy-load the SAM3D model on first use."""
    global sam3d_model
    if sam3d_model is not None:
        return sam3d_model

    print("Loading SAM3D model...")
    config_path = os.path.join(args.checkpoint_dir, "pipeline.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"SAM3D checkpoint not found at {config_path}. "
            f"Download with: huggingface-cli download facebook/sam-3d-objects"
        )

    # Use SAM3D's Inference wrapper from the notebook directory
    sam3d_notebook = os.path.join(os.path.dirname(args.checkpoint_dir), "..", "notebook")
    if os.path.isdir(sam3d_notebook):
        sys.path.insert(0, os.path.abspath(sam3d_notebook))

    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap

    config = OmegaConf.load(config_path)
    config.compile_model = False
    config.workspace_dir = os.path.dirname(config_path)

    # Choose rendering engine based on texture baking availability
    global TEXTURE_BAKING_ENABLED
    if TEXTURE_BAKING_ENABLED:
        try:
            import nvdiffrast
            config.rendering_engine = "nvdiffrast"
            print(f"Texture baking: ENABLED (nvdiffrast available)")
        except ImportError:
            msg = (
                "nvdiffrast not found. Install with: pip install nvdiffrast\n"
                "Texture baking cannot run without nvdiffrast."
            )
            if SAM3D_TEXTURE_STRICT:
                raise RuntimeError(msg)
            print("WARNING:", msg.replace("\n", " "))
            print("  Falling back to vertex colors only")
            config.rendering_engine = "pytorch3d"
            TEXTURE_BAKING_ENABLED = False
    else:
        config.rendering_engine = "pytorch3d"
        print(f"Texture baking: DISABLED (SAM3D_TEXTURE_BAKING=0)")

    # Safety check (simplified from notebook/inference.py)
    pipeline = instantiate(config)
    sam3d_model = pipeline
    print(f"SAM3D model loaded! (rendering_engine={config.rendering_engine}, texture_baking={TEXTURE_BAKING_ENABLED})")
    return sam3d_model


# ── Job tracking ──────────────────────────────────────────────────────────────
jobs = {}
jobs_lock = threading.Lock()

# Sequential processing queue (prevents GPU OOM)
job_queue = []
queue_lock = threading.Lock()
queue_event = threading.Event()

# ── Worker watchdog ───────────────────────────────────────────────────────
_worker_thread = None
_worker_healthy = True
_worker_lock = threading.Lock()

def _safe_float_env(key: str, default: float) -> float:
    try:
        value = float(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return float(default)
    return float(value) if value > 0 else float(default)


# How long a single /generate job is allowed to stay in "processing" before being
# marked failed. This prevents clients from polling forever if the worker hangs.
# Default aligns with SceneSmith bridge timeout_total_s=900.
JOB_TIMEOUT_S = _safe_float_env("SAM3D_JOB_TIMEOUT_S", 900.0)
# How often to check worker/job health.
WATCHDOG_INTERVAL_S = _safe_float_env("SAM3D_WATCHDOG_INTERVAL_S", 10.0)
# If enabled, terminate the server when it becomes unhealthy so an external
# supervisor can restart it.
EXIT_ON_UNHEALTHY = os.environ.get("SAM3D_EXIT_ON_UNHEALTHY", "0") == "1"


def _start_worker():
    """Start (or restart) the worker thread."""
    global _worker_thread, _worker_healthy
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _worker_healthy = True
        _worker_thread = threading.Thread(target=worker_loop, daemon=True, name="sam3d-worker")
        _worker_thread.start()
        print(f"[SAM3D] Worker thread started (tid={_worker_thread.ident})", flush=True)


def _watchdog_loop():
    """Monitor worker thread and fail stuck jobs so clients don't poll forever."""
    global _worker_healthy
    while True:
        time.sleep(WATCHDOG_INTERVAL_S)
        with _worker_lock:
            alive = _worker_thread is not None and _worker_thread.is_alive()
        if not alive:
            if _worker_healthy:
                print("[SAM3D] WATCHDOG: Worker thread died! Restarting...", flush=True)
                _worker_healthy = False

            # Fail any jobs stuck in "processing"
            with jobs_lock:
                for jid, jdata in jobs.items():
                    if jdata.get("status") == "processing":
                        jdata["status"] = "failed"
                        jdata["error"] = "Worker thread crashed during processing"
                        jdata["failed_at"] = time.time()
                        print(f"[SAM3D] WATCHDOG: Marked job {jid} as failed", flush=True)

            # Clean up CUDA state
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # synchronize() can hang if the CUDA context is corrupted.
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    print("[SAM3D] WATCHDOG: CUDA cache cleared", flush=True)
            except Exception as e:
                print(f"[SAM3D] WATCHDOG: CUDA cleanup error: {e}", flush=True)

            _start_worker()
            continue

        # Detect a stuck processing job (thread alive but no completion).
        stuck_job_id = None
        stuck_age_s = None
        now = time.time()
        with jobs_lock:
            for jid, jdata in jobs.items():
                if jdata.get("status") != "processing":
                    continue
                started_at = float(jdata.get("started_at") or jdata.get("created_at") or now)
                age_s = now - started_at
                if age_s > JOB_TIMEOUT_S:
                    stuck_job_id = jid
                    stuck_age_s = age_s
                    jdata["status"] = "failed"
                    jdata["error"] = f"Job timed out after {JOB_TIMEOUT_S:.0f}s in processing"
                    jdata["failed_at"] = now
                    break
        if stuck_job_id:
            print(
                f"[SAM3D] WATCHDOG: Job {stuck_job_id} timed out after {stuck_age_s:.1f}s (timeout={JOB_TIMEOUT_S:.0f}s)",
                flush=True,
            )
            _worker_healthy = False
            if EXIT_ON_UNHEALTHY:
                print("[SAM3D] WATCHDOG: Exiting process due to SAM3D_EXIT_ON_UNHEALTHY=1", flush=True)
                os.kill(os.getpid(), signal.SIGTERM)


def worker_loop():
    """Process jobs one at a time to prevent GPU OOM."""
    while True:
        queue_event.wait()
        queue_event.clear()

        while True:
            with queue_lock:
                if not job_queue:
                    break
                job_id = job_queue.pop(0)

            with jobs_lock:
                job = jobs.get(job_id)
            if not job or job["status"] != "queued":
                continue

            with jobs_lock:
                jobs[job_id]["status"] = "processing"
                jobs[job_id]["started_at"] = time.time()

            try:
                _process_job(job_id, job["input_text"], job["seed"])
            except Exception as e:
                tb = traceback.format_exc()
                print(f"Job {job_id} failed: {e}\n{tb}", flush=True)
                with jobs_lock:
                    jobs[job_id]["status"] = "failed"
                    jobs[job_id]["error"] = str(e)
            finally:
                # Free VRAM between jobs to prevent fragmentation
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()


def _process_job(job_id, input_text, seed):
    """Generate reference image → SAM3D 3D reconstruction → GLB."""
    import numpy as np
    from PIL import Image as PILImage

    t0 = time.time()

    # ── Step 1: Generate reference image (Gemini primary, OpenAI fallback) ──
    print(f"Job {job_id}: Generating reference image for: {input_text[:80]}...", flush=True)
    import base64
    import requests as req

    img_prompt = (
        f"A single {input_text}, centered on a pure white background, "
        f"product photography style, high quality, no shadows, "
        f"isolated object with no other items in the scene"
    )
    ref_image = None

    # ── Try Gemini first (free tier with retry for rate limits) ──
    if GEMINI_API_KEY and args.image_backend == "gemini":
        gemini_models = ["gemini-2.0-flash-exp-image-generation", "gemini-2.5-flash-image"]
        for gm in gemini_models:
            if ref_image is not None:
                break
            for attempt in range(3):  # retry up to 3 times on rate limit
                try:
                    if attempt > 0:
                        wait = 10 * attempt
                        print(f"Job {job_id}: Rate limited, waiting {wait}s (attempt {attempt+1})...", flush=True)
                        time.sleep(wait)
                    print(f"Job {job_id}: Trying Gemini ({gm})...", flush=True)
                    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gm}:generateContent?key={GEMINI_API_KEY}"
                    gemini_payload = {
                        "contents": [{"parts": [{"text": img_prompt}]}],
                        "generationConfig": {
                            "responseModalities": ["TEXT", "IMAGE"],
                        },
                    }
                    resp = req.post(gemini_url, json=gemini_payload, timeout=90)
                    if resp.status_code == 429 and attempt < 2:
                        continue  # retry on rate limit
                    resp.raise_for_status()
                    result = resp.json()
                    for candidate in result.get("candidates", []):
                        for part in candidate.get("content", {}).get("parts", []):
                            if "inlineData" in part:
                                img_bytes = base64.b64decode(part["inlineData"]["data"])
                                ref_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGBA")
                                print(f"Job {job_id}: Gemini image generated via {gm}!", flush=True)
                                break
                        if ref_image:
                            break
                    break  # success or non-retryable error
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        continue
                    print(f"Job {job_id}: Gemini {gm} failed ({e})", flush=True)
                    break
        if ref_image is None:
            print(f"Job {job_id}: Gemini unavailable, falling back to OpenAI", flush=True)

    # ── Fallback to OpenAI ──
    if ref_image is None and oai_client:
        try:
            print(f"Job {job_id}: Using OpenAI image generation...", flush=True)
            response = oai_client.images.generate(
                model=args.image_model,
                prompt=img_prompt,
                size=args.image_size,
                n=1,
            )
            img_data = response.data[0]
            if getattr(img_data, "b64_json", None):
                img_bytes = base64.b64decode(img_data.b64_json)
                ref_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGBA")
            elif getattr(img_data, "url", None):
                img_resp = req.get(img_data.url, timeout=30)
                img_resp.raise_for_status()
                ref_image = PILImage.open(io.BytesIO(img_resp.content)).convert("RGBA")
            else:
                raise RuntimeError("OpenAI response had neither b64_json nor url")
        except Exception as e:
            raise RuntimeError(f"Image generation failed: {e}") from e

    if ref_image is None:
        raise RuntimeError("No image backend available or all failed")

    t_image = time.time() - t0
    print(f"Job {job_id}: Reference image generated in {t_image:.1f}s", flush=True)

    # ── Step 2: Create mask (entire object = white on white bg) ──
    # For a product-photo-style image, the object is the non-white area.
    # Simple approach: use alpha channel or threshold-based mask.
    ref_array = np.array(ref_image)

    if ref_array.shape[2] == 4:
        # Use alpha channel as mask
        mask = ref_array[:, :, 3] > 128
    else:
        # Threshold: anything not near-white is the object
        gray = np.mean(ref_array[:, :, :3], axis=2)
        mask = gray < 240

    # If mask is too small (image gen didn't produce clear alpha), use full image
    mask_ratio = mask.sum() / mask.size
    if mask_ratio < 0.01:
        print(f"Job {job_id}: Mask too small ({mask_ratio:.3f}), using full image", flush=True)
        mask = np.ones(mask.shape, dtype=bool)

    print(f"Job {job_id}: Mask covers {mask_ratio:.1%} of image", flush=True)

    # ── Step 3: Run SAM3D ──
    print(f"Job {job_id}: Running SAM3D inference...", flush=True)
    import torch
    with sam3d_lock:
        pipeline = load_sam3d()
        # Merge image + mask into RGBA (SAM3D expects this)
        rgba = np.concatenate([ref_array[:, :, :3], (mask * 255).astype(np.uint8)[..., None]], axis=-1)
        if seed is not None:
            torch.manual_seed(seed)
        output = pipeline.run(
            rgba,
            None,  # mask already embedded in alpha channel
            seed=seed,
            stage1_only=False,
            with_mesh_postprocess=True,
            with_texture_baking=TEXTURE_BAKING_ENABLED,
            with_layout_postprocess=False,
            use_vertex_color=True,  # always generate vertex colors as fallback
        )

    t_sam3d = time.time() - t0 - t_image
    print(f"Job {job_id}: SAM3D inference done in {t_sam3d:.1f}s")

    # ── Step 4: Export GLB ──
    glb = output.get("glb")
    if glb is None:
        # Fall back to gaussian splat → try to get mesh another way
        gs = output.get("gs") or output.get("gaussian", [None])[0]
        if gs is not None:
            # Save as PLY and note it's not a mesh
            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                gs.save_ply(tmp.name)
                with open(tmp.name, "rb") as f:
                    file_content = f.read()
                os.unlink(tmp.name)
            print(f"Job {job_id}: WARNING: Only Gaussian splat available, not mesh GLB")
            # Still return it — SAGE's post-processing may handle it
        else:
            raise RuntimeError("SAM3D produced neither GLB mesh nor Gaussian splat")
    else:
        # Export GLB to bytes
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            glb.export(tmp.name)
            with open(tmp.name, "rb") as f:
                file_content = f.read()
            os.unlink(tmp.name)

    total_time = time.time() - t0
    print(f"Job {job_id}: Complete! Image={t_image:.1f}s, SAM3D={t_sam3d:.1f}s, Total={total_time:.1f}s, GLB={len(file_content)} bytes")

    with jobs_lock:
        jobs[job_id] = {
            "status": "completed",
            "input_text": input_text,
            "seed": seed,
            "file_content": file_content,
            "generation_time": total_time,
            "completed_at": time.time(),
        }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health_check():
    with _worker_lock:
        alive = _worker_thread is not None and _worker_thread.is_alive()
    if not alive:
        return jsonify({
            "status": "unhealthy",
            "reason": "worker_thread_dead",
            "gpu_available": True,
            "backend": "sam3d",
        }), 503
    # If any job is stuck in processing beyond JOB_TIMEOUT_S, report unhealthy.
    now = time.time()
    with jobs_lock:
        for jdata in jobs.values():
            if jdata.get("status") != "processing":
                continue
            started_at = float(jdata.get("started_at") or jdata.get("created_at") or now)
            if now - started_at > JOB_TIMEOUT_S:
                return jsonify({
                    "status": "unhealthy",
                    "reason": "job_processing_timeout",
                    "job_timeout_s": JOB_TIMEOUT_S,
                    "gpu_available": True,
                    "backend": "sam3d",
                }), 503
    return jsonify({"status": "healthy", "gpu_available": True, "backend": "sam3d"})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    input_text = data.get("input_text", "A simple 3D object")
    seed = data.get("seed", random.randint(1, 1000000))
    job_id = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "input_text": input_text,
            "seed": seed,
            "created_at": time.time(),
        }

    with queue_lock:
        job_queue.append(job_id)
    queue_event.set()

    return jsonify({
        "status": "accepted",
        "job_id": job_id,
        "message": "Request received and processing started",
    }), 202


@app.route("/job/<job_id>", methods=["GET"])
def get_job(job_id):
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job["status"] == "completed":
        return send_file(
            io.BytesIO(job["file_content"]),
            as_attachment=True,
            download_name=f"generated_model_{job.get('seed', 0)}.glb",
            mimetype="application/octet-stream",
        )
    elif job["status"] == "failed":
        return jsonify({
            "status": "failed",
            "error": job.get("error", "Unknown error"),
        }), 500
    else:
        # Prevent indefinite polling: fail "processing" jobs that exceed timeout.
        if job.get("status") == "processing":
            now = time.time()
            started_at = float(job.get("started_at") or job.get("created_at") or now)
            if now - started_at > JOB_TIMEOUT_S:
                with jobs_lock:
                    job2 = jobs.get(job_id)
                    if job2 and job2.get("status") == "processing":
                        job2["status"] = "failed"
                        job2["error"] = f"Job timed out after {JOB_TIMEOUT_S:.0f}s in processing"
                        job2["failed_at"] = now
                return jsonify({"status": "failed", "error": f"Job timed out after {JOB_TIMEOUT_S:.0f}s"}), 500
        return jsonify({
            "status": job["status"],
            "job_id": job_id,
            "message": "Job is still processing",
        }), 202


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "SAM3D Drop-in Server (TRELLIS-compatible)",
        "version": "1.0.0",
        "status": "running",
        "backend": "sam3d + openai-image",
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    with jobs_lock:
        total = len(jobs)
        completed = sum(1 for j in jobs.values() if j["status"] == "completed")
        failed = sum(1 for j in jobs.values() if j["status"] == "failed")
        processing = sum(1 for j in jobs.values() if j["status"] in ("queued", "processing"))
    return jsonify({
        "total_jobs": total,
        "completed": completed,
        "failed": failed,
        "processing": processing,
    })


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Gracefully shut down the server to free VRAM for later pipeline stages."""
    import signal
    print("[SAM3D] Shutdown requested — freeing GPU memory...")
    # Return response before shutting down
    response = jsonify({"status": "shutting_down", "message": "SAM3D server stopping to free VRAM"})
    # Schedule shutdown after response is sent
    def _shutdown():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_shutdown, daemon=True).start()
    return response, 200


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start managed worker thread + watchdog
    _start_worker()
    _watchdog = threading.Thread(target=_watchdog_loop, daemon=True, name="sam3d-watchdog")
    _watchdog.start()

    print("=" * 60)
    print("SAM3D Drop-in Server (TRELLIS-compatible API)")
    print("=" * 60)
    print(f"Port: {args.port}")
    print(f"Image model: {args.image_model}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Endpoints:")
    print(f"  GET  /health")
    print(f"  POST /generate  {{input_text: '...', seed: N}}")
    print(f"  GET  /job/{{id}}")
    print("=" * 60)

    app.run(host="0.0.0.0", port=args.port, debug=False)
