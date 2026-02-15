#!/usr/bin/env python3
"""
PhysX-Anything Service for BlueprintPipeline.

Wraps VictorTao1998/PhysX-Anything to produce a simulation-ready articulated
asset payload from a single RGB image.

This service returns a base64-encoded zip containing:
- a URDF file
- any referenced meshes/textures produced by PhysX-Anything

Request (POST /):
{
  "image_base64": "<base64 bytes>",
  "seed": 1001,
  "remove_bg": false,
  "voxel_define": 32,
  "fixed_base": 0,
  "deformable": 0
}
"""

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional
import xml.etree.ElementTree as ET

from flask import Flask, jsonify, request

app = Flask(__name__)

PHYSX_ROOT = Path(os.environ.get("PHYSX_ANYTHING_ROOT", "/opt/physx-anything"))
CKPT_DIR = Path(os.environ.get("PHYSX_ANYTHING_CKPT", str(PHYSX_ROOT / "pretrain" / "vlm")))
TMP_ROOT = Path(os.environ.get("PHYSX_ANYTHING_TMP_DIR", "/tmp/physx-anything"))
PROCESS_TIMEOUT_S = int(os.environ.get("PHYSX_ANYTHING_PROCESS_TIMEOUT_S", "1200") or "1200")

_gpu_lock = threading.Semaphore(1)


def _pick_latest_file(root: Path, pattern: str) -> Optional[Path]:
    candidates = list(root.rglob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _zip_directory(root: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            zf.write(path, arcname=str(path.relative_to(root)))
    return buf.getvalue()


def _count_joints(urdf_path: Path) -> int:
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        joints = root.findall("joint")
        count = 0
        for j in joints:
            if (j.attrib.get("type") or "").strip().lower() != "fixed":
                count += 1
        return count
    except Exception:
        return 0


@app.get("/")
def health():
    ready = bool((PHYSX_ROOT / "1_vlm_demo.py").is_file() and CKPT_DIR.exists())
    return jsonify(
        {
            "status": "ok",
            "ready": ready,
            "physx_root": str(PHYSX_ROOT),
            "ckpt_dir": str(CKPT_DIR),
        }
    )


@app.post("/")
def infer():
    payload: Dict[str, Any] = request.get_json(force=True, silent=False) or {}
    image_b64 = payload.get("image_base64")
    if not isinstance(image_b64, str) or not image_b64.strip():
        return jsonify({"status": "error", "message": "image_base64 required"}), 400

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"invalid image_base64: {exc}"}), 400

    seed = int(payload.get("seed", 1001) or 1001)
    remove_bg = bool(payload.get("remove_bg", False))
    voxel_define = int(payload.get("voxel_define", 32) or 32)
    fixed_base = int(payload.get("fixed_base", 0) or 0)
    deformable = int(payload.get("deformable", 0) or 0)

    req_id = uuid.uuid4().hex[:12]
    work_dir = TMP_ROOT / f"req_{req_id}"
    demo_dir = work_dir / "demo"

    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    demo_img = demo_dir / "input.png"
    demo_img.write_bytes(image_bytes)

    # PhysX-Anything scripts expect prompt assets under ./dataset in the working directory.
    dataset_src = PHYSX_ROOT / "dataset"
    if dataset_src.is_dir():
        dataset_dst = work_dir / "dataset"
        if not dataset_dst.exists():
            try:
                dataset_dst.symlink_to(dataset_src)
            except Exception:
                dataset_dst.mkdir(parents=True, exist_ok=True)
                prompt_src = dataset_src / "overall_prompt.txt"
                if prompt_src.is_file():
                    shutil.copy2(prompt_src, dataset_dst / "overall_prompt.txt")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PHYSX_ROOT}:{env.get('PYTHONPATH', '')}"
    env["PYTHONHASHSEED"] = str(seed)

    with _gpu_lock:
        try:
            step1 = [
                sys.executable,
                str(PHYSX_ROOT / "1_vlm_demo.py"),
                "--demo_path",
                str(demo_dir),
                "--save_part_ply",
                "True",
                "--remove_bg",
                "True" if remove_bg else "False",
                "--ckpt",
                str(CKPT_DIR),
            ]
            subprocess.run(
                step1,
                check=True,
                cwd=str(work_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                timeout=PROCESS_TIMEOUT_S,
            )

            for script_name in ["2_decoder.py", "3_split.py"]:
                subprocess.run(
                    [sys.executable, str(PHYSX_ROOT / script_name)],
                    check=True,
                    cwd=str(work_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                    timeout=PROCESS_TIMEOUT_S,
                )

            step4 = [
                sys.executable,
                str(PHYSX_ROOT / "4_simready_gen.py"),
                "--voxel_define",
                str(voxel_define),
                "--basepath",
                str(work_dir / "test_demo"),
                "--process",
                "0",
                "--fixed_base",
                str(fixed_base),
                "--deformable",
                str(deformable),
            ]
            subprocess.run(
                step4,
                check=True,
                cwd=str(work_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                timeout=PROCESS_TIMEOUT_S,
            )
        except subprocess.CalledProcessError as exc:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"physx-anything failed: {exc.returncode}",
                        "stdout": (exc.stdout or "")[:4000],
                    }
                ),
                500,
            )
        except subprocess.TimeoutExpired:
            return jsonify({"status": "error", "message": "physx-anything timed out"}), 504
        except Exception as exc:
            return jsonify({"status": "error", "message": f"physx-anything exception: {exc}"}), 500

    urdf_path = _pick_latest_file(work_dir, "*.urdf")
    if not urdf_path or not urdf_path.is_file():
        return jsonify({"status": "error", "message": "no urdf produced"}), 500

    payload_dir = urdf_path.parent
    zip_bytes = _zip_directory(payload_dir)
    joint_count = _count_joints(urdf_path)

    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass

    return jsonify(
        {
            "payload_zip_base64": base64.b64encode(zip_bytes).decode("ascii"),
            "placeholder": False,
            "generator": "physx-anything",
            "articulation": {
                "joint_count": joint_count,
                "is_articulated": joint_count > 0,
            },
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080") or "8080")
    app.run(host="0.0.0.0", port=port, debug=False)
