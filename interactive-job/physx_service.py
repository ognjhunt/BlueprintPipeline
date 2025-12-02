import base64
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional, List

from flask import Flask, jsonify, request

app = Flask(__name__)

# Where the PhysX-Anything repo is installed.
PHYSX_ROOT = Path(os.environ.get("PHYSX_ROOT", "/opt/physx_anything"))

# Our small driver script that orchestrates 1_vlm_demo.py → 4_simready_gen.py
PHYSX_ENTRY = PHYSX_ROOT / "run_physx_anything_pipeline.py"


class PhysxError(RuntimeError):
    """Raised when the PhysX-Anything pipeline fails."""


def run_physx_anything(image_bytes: bytes) -> Tuple[bytes, bytes]:
    """
    Given raw PNG/JPEG bytes, run the PhysX-Anything pipeline and
    return (mesh_glb_bytes, urdf_bytes).
    """
    if not PHYSX_ENTRY.is_file():
        raise PhysxError(f"PhysX entry script not found at {PHYSX_ENTRY}")

    # 1) Make temp dirs for input + output
    tmp_root = Path("/tmp/physx_anything")
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Use PID to avoid collisions between concurrent requests.
    req_dir = tmp_root / f"req_{os.getpid()}"
    in_dir = req_dir / "input"
    out_dir = req_dir / "output"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Save the image
    input_path = in_dir / "input.png"
    input_path.write_bytes(image_bytes)

    # 3) Run the driver script
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

    # 4) Read mesh + URDF from out_dir
    mesh_path = out_dir / "part.glb"
    urdf_path = out_dir / "part.urdf"

    if not mesh_path.is_file() or not urdf_path.is_file():
        raise PhysxError(
            f"Expected part.glb and part.urdf in {out_dir}, "
            f"found mesh={mesh_path.is_file()}, urdf={urdf_path.is_file()}"
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

    If anything fails inside the PhysX pipeline, we return HTTP 500 and
    interactive-job will fall back to placeholder assets.
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
        return jsonify({"error": str(e)}), 500

    resp = encode_assets(mesh_bytes, urdf_bytes)
    return jsonify(resp), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
