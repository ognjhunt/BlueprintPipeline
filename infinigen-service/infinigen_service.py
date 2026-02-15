#!/usr/bin/env python3
"""
Infinigen Service for BlueprintPipeline.

Runs Infinigen's asset generator/exporter and returns a base64-encoded zip
payload containing the URDF + referenced meshes for an articulated asset.

Request (POST /):
{
  "asset_name": "refrigerators",
  "seed": 1001,
  "collision": true,
  "export": "urdf"
}
"""

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional
import xml.etree.ElementTree as ET

from flask import Flask, jsonify, request

app = Flask(__name__)

INFINIGEN_ROOT = Path(os.environ.get("INFINIGEN_ROOT", "/opt/infinigen"))
SPAWN_ASSET = INFINIGEN_ROOT / "scripts" / "spawn_asset.py"
TMP_ROOT = Path(os.environ.get("INFINIGEN_TMP_DIR", "/tmp/infinigen"))
PROCESS_TIMEOUT_S = int(os.environ.get("INFINIGEN_PROCESS_TIMEOUT_S", "1200") or "1200")


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
    ready = bool(SPAWN_ASSET.is_file())
    return jsonify({"status": "ok", "ready": ready, "infinigen_root": str(INFINIGEN_ROOT)})


@app.post("/")
def generate():
    payload: Dict[str, Any] = request.get_json(force=True, silent=False) or {}
    asset_name = str(payload.get("asset_name", "") or "").strip()
    if not asset_name:
        return jsonify({"status": "error", "message": "asset_name required"}), 400

    seed = int(payload.get("seed", 1001) or 1001)
    collision = bool(payload.get("collision", True))
    export = str(payload.get("export", "urdf") or "urdf").strip().lower()
    if export not in {"urdf"}:
        return jsonify({"status": "error", "message": f"unsupported export: {export}"}), 400

    req_id = uuid.uuid4().hex[:12]
    work_dir = TMP_ROOT / f"req_{req_id}"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SPAWN_ASSET),
        "-exp",
        export,
        "-n",
        asset_name,
        "-s",
        str(seed),
        "-dir",
        str(work_dir),
    ]
    if collision:
        cmd.append("-c")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{INFINIGEN_ROOT}:{env.get('PYTHONPATH', '')}"

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            timeout=PROCESS_TIMEOUT_S,
            cwd=str(INFINIGEN_ROOT),
        )
    except subprocess.CalledProcessError as exc:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"infinigen failed: {exc.returncode}",
                    "stdout": (exc.stdout or "")[:4000],
                }
            ),
            500,
        )
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "message": "infinigen timed out"}), 504
    except Exception as exc:
        return jsonify({"status": "error", "message": f"infinigen exception: {exc}"}), 500

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
            "generator": "infinigen",
            "articulation": {
                "joint_count": joint_count,
                "is_articulated": joint_count > 0,
            },
            "meta": {"asset_name": asset_name, "seed": seed, "collision": collision},
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080") or "8080")
    app.run(host="0.0.0.0", port=port, debug=False)

