import base64
import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional, List

from flask import Flask, jsonify, request

app = Flask(__name__)

# Where your PhysX-Anything code lives inside the container.
# You will set this in the Dockerfile or as an env var.
PHYSX_ROOT = Path(os.environ.get("PHYSX_ROOT", "/opt/physx_anything"))

# Script that runs the full PhysX-Anything pipeline.
# You will implement this (or adjust the path) based on the repo’s docs.
PHYSX_ENTRY = PHYSX_ROOT / "run_physx_anything_pipeline.py"


class PhysxError(RuntimeError):
    pass


def run_physx_anything(image_bytes: bytes) -> Tuple[bytes, bytes]:
    """
    Given raw PNG/JPEG bytes, run the PhysX-Anything pipeline and
    return (mesh_glb_bytes, urdf_bytes).

    You MUST fill in the command that actually runs the pipeline,
    based on the official PhysX-Anything repo instructions.
    """
    if not PHYSX_ENTRY.is_file():
        raise PhysxError(f"PhysX entry script not found at {PHYSX_ENTRY}")

    # 1) Make temp dirs for input + output
    tmp_root = Path("/tmp/physx_anything")
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Use the PID as a simple per-request subdir to avoid collisions.
    req_dir = tmp_root / f"req_{os.getpid()}"
    in_dir = req_dir / "input"
    out_dir = req_dir / "output"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Save the image
    input_path = in_dir / "input.png"
    input_path.write_bytes(image_bytes)

    # 3) Run the pipeline
    #
    # >>> IMPORTANT <<<
    # Replace the command list below with the actual command(s)
    # PhysX-Anything uses on your machine. Example pattern:
    #
    #   python run_physx_anything_pipeline.py \
    #      --input_image /path/to/input.png \
    #      --output_dir /path/to/output
    #
    # You may need additional flags for category, seed, etc.
    #
    cmd: List[str] = [
        sys.executable,
        str(PHYSX_ENTRY),
        "--input_image",
        str(input_path),
        "--output_dir",
        str(out_dir),
    ]
    app.logger.info("[PHYSX-SERVICE] Running pipeline: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        app.logger.info("[PHYSX-SERVICE] Pipeline output:\n%s", result.stdout)
    except subprocess.CalledProcessError as e:
        raise PhysxError(
            f"PhysX-Anything pipeline failed with code {e.returncode}: {e.stdout}"
        ) from e

    # 4) Find GLB and URDF in out_dir
    def find_first(patterns: List[str]) -> Optional[Path]:
        for pattern in patterns:
            for p in out_dir.rglob(pattern):
                return p
        return None

    mesh_path = find_first(["*.glb", "*.gltf"])
    urdf_path = find_first(["*.urdf"])

    if mesh_path is None or urdf_path is None:
        raise PhysxError(
            f"Could not find both mesh and URDF in {out_dir} "
            f"(mesh={mesh_path}, urdf={urdf_path})"
        )

    mesh_bytes = mesh_path.read_bytes()
    urdf_bytes = urdf_path.read_bytes()
    return mesh_bytes, urdf_bytes


def encode_assets(mesh_bytes: bytes, urdf_bytes: bytes) -> dict:
    """
    Convert raw bytes into the JSON schema interactive-job expects:
      - mesh_base64
      - urdf_base64
      - placeholder=False
    """
    return {
        "mesh_base64": base64.b64encode(mesh_bytes).decode("ascii"),
        "urdf_base64": base64.b64encode(urdf_bytes).decode("ascii"),
        "placeholder": False,
        "generator": "physx-anything",
    }


@app.route("/", methods=["GET"])
def healthcheck():
    return "ok", 200


@app.route("/", methods=["POST"])
def handle_request():
    """
    Main endpoint used by interactive-job.

    Expects JSON:
        { "image_base64": "<base64 png/jpg bytes>" }

    Returns JSON:
        {
          "mesh_base64": "<...>",
          "urdf_base64": "<...>",
          "placeholder": false,
          "generator": "physx-anything"
        }
    """
    data = request.get_json(force=True, silent=True) or {}
    img_b64 = data.get("image_base64")

    if not img_b64:
        return jsonify({"error": "image_base64 is required"}), 400

    try:
        image_bytes = base64.b64decode(img_b64)
    except Exception:
        return jsonify({"error": "image_base64 is not valid base64"}), 400

    try:
        mesh_bytes, urdf_bytes = run_physx_anything(image_bytes)
    except PhysxError as e:
        app.logger.error("[PHYSX-SERVICE] %s", e)
        # You can choose to return 500 here and let interactive-job
        # fall back to placeholders, or return placeholder directly.
        return jsonify({"error": str(e)}), 500

    resp = encode_assets(mesh_bytes, urdf_bytes)
    return jsonify(resp), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
