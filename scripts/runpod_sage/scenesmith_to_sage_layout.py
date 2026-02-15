#!/usr/bin/env python3
"""
SceneSmith (official paper stack) -> SAGE layout directory bridge.

Goal:
  - Keep SAGE stages 5-7 as the "factory" (grasps/plans/capture)
  - Allow generating a single SAGE-compatible layout dir from SceneSmith output

This script:
  1) Runs the official SceneSmith paper stack via:
       BlueprintPipeline/scenesmith-service/scenesmith_paper_command.py
  2) Converts objects + transforms into a SAGE-style room dict JSON (z-up).
  3) Writes a minimal pose augmentation folder:
       pose_aug_0/meta.json (list with 1 variant)
       pose_aug_0/variant_000.json (room dict)
  4) Writes meshes into generation/*.obj:
       - Default: key_only (manipulables + key surfaces) use SAM3D server
       - Others: primitive unit-box OBJ (scaled in Isaac Sim by dimensions)

Output:
  Prints the new layout_id to stdout (and logs to stderr).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_SURFACE_TOKENS = ("table", "counter", "desk", "island", "bench")


def _log(msg: str) -> None:
    print(f"[scenesmith->sage {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", file=sys.stderr, flush=True)


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _slug(s: str, default: str) -> str:
    import re

    text = str(s or "").strip()
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    text = re.sub(r"_{2,}", "_", text).strip("_")
    if not text:
        return default
    return text[:80]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_unit_box_obj(path: Path) -> None:
    """
    Write a canonical unit box OBJ with:
      x,y in [-0.5, +0.5] (centered footprint)
      z in [0.0, 1.0]      (bottom at 0)
    This matches the convention used by the SAGE collector in this repo:
      - position.x/y is footprint center
      - position.z is bottom
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # 8 vertices, 12 triangles (two per face)
    obj = """# unit box (centered XY, bottom at Z=0)
v -0.5 -0.5 0.0
v  0.5 -0.5 0.0
v  0.5  0.5 0.0
v -0.5  0.5 0.0
v -0.5 -0.5 1.0
v  0.5 -0.5 1.0
v  0.5  0.5 1.0
v -0.5  0.5 1.0
f 1 2 3
f 1 3 4
f 5 8 7
f 5 7 6
f 1 5 6
f 1 6 2
f 2 6 7
f 2 7 3
f 3 7 8
f 3 8 4
f 4 8 5
f 4 5 1
"""
    path.write_text(obj, encoding="utf-8")


# SceneSmith (y-up) -> SAGE (z-up) basis transform:
#   x_g = x_s
#   y_g = -z_s
#   z_g = y_s
_A = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)


def _mat3_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = float(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
    return out


def _mat3_T(a: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(a[j][i]) for j in range(3)] for i in range(3)]


def _quat_to_mat3(qw: float, qx: float, qy: float, qz: float) -> List[List[float]]:
    # Normalize (avoid drift)
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n <= 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def _scenesmith_pos_to_sage(px: float, py: float, pz: float) -> Tuple[float, float, float]:
    return (float(px), float(-pz), float(py))


def _scenesmith_rot_to_sage_yaw_deg(q: Mapping[str, Any]) -> float:
    qw = _safe_float(q.get("w"), 1.0)
    qx = _safe_float(q.get("x"), 0.0)
    qy = _safe_float(q.get("y"), 0.0)
    qz = _safe_float(q.get("z"), 0.0)
    R_s = _quat_to_mat3(qw, qx, qy, qz)
    A = _A
    AT = _mat3_T(A)
    # Similarity transform: R_g = A * R_s * A^T
    R_g = _mat3_mul(_mat3_mul(A, R_s), AT)
    yaw = math.degrees(math.atan2(R_g[1][0], R_g[0][0]))
    # Wrap to [-180, 180]
    while yaw > 180.0:
        yaw -= 360.0
    while yaw < -180.0:
        yaw += 360.0
    return float(yaw)


def _is_surface_type(t: str) -> bool:
    s = str(t or "").strip().lower().replace(" ", "_")
    return any(tok in s for tok in _SURFACE_TOKENS)


def _pick_mesh_candidates(
    raw_objs: Sequence[Mapping[str, Any]],
    *,
    mesh_policy: str,
    max_meshes: int,
) -> List[int]:
    if max_meshes <= 0:
        return []
    if mesh_policy == "all":
        return list(range(min(len(raw_objs), max_meshes)))

    # key_only: manipulables first, then key surfaces.
    manipulables: List[int] = []
    surfaces: List[int] = []
    for idx, o in enumerate(raw_objs):
        role = str(o.get("sim_role", "")).strip().lower()
        t = str(o.get("category") or o.get("name") or "")
        if role == "manipulable_object":
            manipulables.append(idx)
        elif _is_surface_type(t):
            surfaces.append(idx)

    ordered = manipulables + [i for i in surfaces if i not in manipulables]
    return ordered[:max_meshes]


def _http_post_json(url: str, payload: Dict[str, Any], *, timeout_s: int) -> Tuple[int, Dict[str, Any]]:
    try:
        import requests  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"requests is required for SAM3D calls: {exc}") from exc

    r = requests.post(url, json=payload, timeout=timeout_s)
    status = int(r.status_code)
    try:
        data = r.json()
    except Exception:
        data = {"_raw": r.text}
    return status, data


def _http_get_bytes(url: str, *, timeout_s: int) -> Tuple[int, bytes, Dict[str, Any]]:
    try:
        import requests  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"requests is required for SAM3D calls: {exc}") from exc

    r = requests.get(url, timeout=timeout_s)
    status = int(r.status_code)
    if status == 200 and (r.headers.get("content-type") or "").lower().startswith("application"):
        return status, r.content, {}
    # Non-200 or JSON status.
    try:
        data = r.json()
    except Exception:
        data = {"_raw": r.text}
    return status, b"", data


def _sam3d_health(server_base: str) -> bool:
    try:
        import requests  # type: ignore
    except Exception:
        return False
    try:
        r = requests.get(server_base.rstrip("/") + "/health", timeout=3)
        return bool(r.status_code == 200)
    except Exception:
        return False


def _sam3d_shutdown(server_base: str) -> None:
    try:
        import requests  # type: ignore
    except Exception:
        return
    try:
        requests.post(server_base.rstrip("/") + "/shutdown", timeout=5)
    except Exception:
        pass


def _sam3d_start_if_needed(server_base: str) -> None:
    """
    Best-effort: start sam3d_server.py if the health endpoint is down.

    This keeps SceneSmith generation and SAM3D mesh synthesis sequential on a
    single GPU, avoiding concurrent VRAM allocation.
    """
    if _sam3d_health(server_base):
        return

    # Try to launch the server as a detached subprocess.
    port = int(server_base.rsplit(":", 1)[-1])
    bp_root = Path(__file__).resolve().parents[2]
    server_script = bp_root / "scripts" / "runpod_sage" / "sam3d_server.py"
    if not server_script.exists():
        raise FileNotFoundError(f"Cannot start SAM3D server; missing: {server_script}")

    python_bin = "python3.11" if shutil.which("python3.11") else sys.executable
    image_backend = os.getenv("SAM3D_IMAGE_BACKEND", "gemini")
    checkpoint_dir = os.getenv("SAM3D_CHECKPOINT_DIR", "/workspace/sam3d/checkpoints/hf")

    log_path = Path("/tmp/sam3d_server_from_scenesmith.log")
    _log(f"Starting SAM3D server for mesh generation on :{port} (backend={image_backend})")
    with log_path.open("ab") as f:
        proc = subprocess.Popen(
            [python_bin, str(server_script), "--port", str(port), "--image-backend", image_backend, "--checkpoint-dir", checkpoint_dir],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    deadline = time.time() + 300.0
    while time.time() < deadline:
        if _sam3d_health(server_base):
            _log("SAM3D server healthy")
            return
        if proc.poll() is not None:
            raise RuntimeError(f"SAM3D server died while starting (log={log_path})")
        time.sleep(2.0)
    raise TimeoutError(f"SAM3D server did not become healthy within 300s (log={log_path})")


def _sam3d_generate_obj(
    *,
    prompt: str,
    out_obj_path: Path,
    seed: int,
    server_base: str,
    timeout_total_s: int = 900,
) -> bool:
    gen_url = server_base.rstrip("/") + "/generate"
    job_base = server_base.rstrip("/") + "/job/"

    status, data = _http_post_json(gen_url, {"input_text": prompt, "seed": int(seed)}, timeout_s=30)
    if status not in (200, 202):
        raise RuntimeError(f"SAM3D /generate failed (status={status}) data={data}")
    job_id = str(data.get("job_id") or "").strip()
    if not job_id:
        raise RuntimeError(f"SAM3D /generate returned no job_id: {data}")

    deadline = time.time() + float(timeout_total_s)
    while time.time() < deadline:
        st, blob, j = _http_get_bytes(job_base + job_id, timeout_s=30)
        if st == 200 and blob:
            # Convert GLB -> OBJ
            try:
                import trimesh  # type: ignore
            except Exception as exc:
                raise RuntimeError(f"trimesh required to convert GLB->OBJ: {exc}") from exc

            loaded = trimesh.load(io.BytesIO(blob), file_type="glb")
            if hasattr(loaded, "geometry") and hasattr(loaded, "dump"):
                # Scene -> single mesh
                mesh = loaded.dump(concatenate=True)
            else:
                mesh = loaded
            out_obj_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(out_obj_path))
            return True
        if st == 202:
            time.sleep(2.0)
            continue
        if st >= 400:
            raise RuntimeError(f"SAM3D job failed (status={st}) data={j}")
        time.sleep(2.0)
    raise TimeoutError(f"SAM3D job timed out after {timeout_total_s}s (prompt={prompt!r})")


def _run_scenesmith_paper_stack(
    *,
    scene_id: str,
    prompt: str,
    room_type: str,
    seed: int,
) -> Dict[str, Any]:
    bp_root = Path(__file__).resolve().parents[2]  # BlueprintPipeline/
    paper_cmd = bp_root / "scenesmith-service" / "scenesmith_paper_command.py"
    if not paper_cmd.exists():
        raise FileNotFoundError(f"Missing SceneSmith command bridge: {paper_cmd}")

    # Defaults expected in the hybrid image.
    os.environ.setdefault("SCENESMITH_PAPER_REPO_DIR", "/opt/scenesmith")
    os.environ.setdefault("SCENESMITH_PAPER_PYTHON_BIN", "/opt/scenesmith/.venv/bin/python")
    os.environ.setdefault("SCENESMITH_PAPER_ALL_SAM3D", "true")

    payload = {
        "request_id": uuid.uuid4().hex,
        "scene_id": scene_id,
        "prompt": prompt,
        "quality_tier": "paper",
        "seed": int(seed),
        "constraints": {"room_type": room_type},
    }

    proc = subprocess.run(
        [sys.executable, str(paper_cmd)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=os.environ.copy(),
        cwd=str(bp_root),
        check=False,
        timeout=int(os.getenv("SCENESMITH_PAPER_TIMEOUT_SECONDS", "5400")),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "SceneSmith paper stack failed: "
            f"exit={proc.returncode} stderr_tail={(proc.stderr or '')[-4000:]!r} "
            f"stdout_tail={(proc.stdout or '')[-4000:]!r}"
        )
    out = json.loads(proc.stdout)
    if not isinstance(out, Mapping):
        raise RuntimeError("SceneSmith paper stack returned non-object JSON")
    return dict(out)


def _convert_objects_to_sage(
    raw_objs: Sequence[Mapping[str, Any]],
    *,
    margin_m: float,
    room_type: str,
    seed: int,
) -> Dict[str, Any]:
    objs: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_objs, start=1):
        transform = raw.get("transform") if isinstance(raw.get("transform"), Mapping) else {}
        pos = transform.get("position") if isinstance(transform.get("position"), Mapping) else {}
        quat = transform.get("rotation_quaternion") if isinstance(transform.get("rotation_quaternion"), Mapping) else {}
        dims = raw.get("dimensions_est") if isinstance(raw.get("dimensions_est"), Mapping) else {}

        px = _safe_float(pos.get("x"), 0.0)
        py = _safe_float(pos.get("y"), 0.0)
        pz = _safe_float(pos.get("z"), 0.0)
        xg, yg, zg = _scenesmith_pos_to_sage(px, py, pz)

        width_s = max(0.01, abs(_safe_float(dims.get("width"), 0.3)))
        height_s = max(0.01, abs(_safe_float(dims.get("height"), 0.3)))
        depth_s = max(0.01, abs(_safe_float(dims.get("depth"), 0.3)))
        yaw_deg = _scenesmith_rot_to_sage_yaw_deg(quat)

        obj_type = str(raw.get("category") or raw.get("name") or "object").strip() or "object"
        obj_id = str(raw.get("id") or f"obj_{idx:03d}").strip() or f"obj_{idx:03d}"
        source_id = f"sm_{idx:04d}_{_slug(obj_type, default='object')}"

        objs.append(
            {
                "type": obj_type,
                "id": obj_id,
                "source_id": source_id,
                "position": {"x": float(xg), "y": float(yg), "z": float(zg)},
                "rotation": {"x": 0.0, "y": 0.0, "z": float(yaw_deg)},
                "dimensions": {"width": float(width_s), "length": float(depth_s), "height": float(height_s)},
            }
        )

    # Shift into positive room coordinates and compute room bounds.
    min_x = 1e9
    min_y = 1e9
    max_x = -1e9
    max_y = -1e9
    max_z = 0.0
    for o in objs:
        p = o["position"]
        d = o["dimensions"]
        cx, cy, cz = float(p["x"]), float(p["y"]), float(p["z"])
        w, l, h = float(d["width"]), float(d["length"]), float(d["height"])
        min_x = min(min_x, cx - 0.5 * w)
        max_x = max(max_x, cx + 0.5 * w)
        min_y = min(min_y, cy - 0.5 * l)
        max_y = max(max_y, cy + 0.5 * l)
        max_z = max(max_z, cz + h)

    if not math.isfinite(min_x):
        min_x, min_y, max_x, max_y = 0.0, 0.0, 6.0, 6.0

    shift_x = float(margin_m - min_x)
    shift_y = float(margin_m - min_y)
    for o in objs:
        o["position"]["x"] = float(o["position"]["x"] + shift_x)
        o["position"]["y"] = float(o["position"]["y"] + shift_y)
        # Keep z as-is (SceneSmith already y>=0 -> z>=0).

    room_w = float((max_x - min_x) + 2.0 * margin_m)
    room_l = float((max_y - min_y) + 2.0 * margin_m)
    room_h = float(max(3.0, max_z + 0.5))

    return {
        "room_type": room_type,
        "dimensions": {"width": room_w, "length": room_l, "height": room_h},
        "seed": int(seed),
        "objects": objs,
        "scene_source": "scenesmith",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SceneSmith paper stack and emit a SAGE layout dir")
    parser.add_argument("--results_dir", default=os.getenv("SAGE_RESULTS_DIR", "/workspace/SAGE/server/results"))
    parser.add_argument("--layout_id", default="")
    parser.add_argument("--room_type", default=os.getenv("ROOM_TYPE", "generic_room"))
    parser.add_argument("--task_desc", default=os.getenv("TASK_DESC", ""))
    parser.add_argument("--prompt", default=os.getenv("SCENESMITH_PROMPT", ""))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "0") or "0"))

    parser.add_argument("--pose_aug_name", default="pose_aug_0")
    parser.add_argument("--mesh_policy", default=os.getenv("SCENESMITH_TO_SAGE_MESH_POLICY", "key_only"), choices=["key_only", "all"])
    parser.add_argument("--max_meshes", type=int, default=int(os.getenv("SCENESMITH_TO_SAGE_MAX_MESHES", "30")))
    parser.add_argument("--sam3d_server", default=os.getenv("SAM3D_SERVER_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--stop_sam3d_before_paper", action="store_true", default=os.getenv("SCENESMITH_TO_SAGE_STOP_SAM3D_BEFORE_PAPER", "1") == "1")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    layout_id = str(args.layout_id).strip()
    if not layout_id:
        layout_id = f"layout_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    layout_dir = results_dir / layout_id
    if layout_dir.exists():
        raise FileExistsError(f"layout_dir already exists: {layout_dir}")

    prompt = str(args.prompt).strip()
    if not prompt:
        room_type = str(args.room_type).strip() or "generic_room"
        task = str(args.task_desc).strip()
        if task:
            prompt = f"A realistic {room_type}. {task}."
        else:
            prompt = f"A realistic {room_type}."

    seed = int(args.seed)
    if seed == 0:
        seed = random.randint(1, 10_000_000)

    _log(f"Generating SceneSmith scene: layout_id={layout_id} room_type={args.room_type} seed={seed}")
    # Avoid concurrent VRAM usage: stop SAM3D server during SceneSmith generation.
    if bool(args.stop_sam3d_before_paper) and _sam3d_health(str(args.sam3d_server)):
        _log("Stopping SAM3D server before SceneSmith paper stack (free VRAM)...")
        _sam3d_shutdown(str(args.sam3d_server))
        # Best-effort wait for it to go down.
        for _ in range(60):
            if not _sam3d_health(str(args.sam3d_server)):
                break
            time.sleep(1.0)

    response = _run_scenesmith_paper_stack(scene_id=layout_id, prompt=prompt, room_type=str(args.room_type), seed=seed)
    raw_objects = response.get("objects") if isinstance(response.get("objects"), list) else []
    if not raw_objects:
        raise RuntimeError("SceneSmith returned 0 objects")

    # Convert objects to SAGE room dict.
    room = _convert_objects_to_sage(raw_objects, margin_m=0.6, room_type=str(args.room_type), seed=seed)
    objs = list(room.get("objects", []) or [])

    # Create layout dir structure.
    (layout_dir / "generation").mkdir(parents=True, exist_ok=True)
    (layout_dir / args.pose_aug_name).mkdir(parents=True, exist_ok=True)

    # Mesh generation (bounded).
    candidates = _pick_mesh_candidates(raw_objects, mesh_policy=str(args.mesh_policy), max_meshes=int(args.max_meshes))
    candidate_set = set(candidates)
    mesh_prompts: Dict[int, str] = {}
    for idx in candidates:
        raw = raw_objects[idx]
        name = str(raw.get("name") or raw.get("category") or "object").strip() or "object"
        mesh_prompts[idx] = name

    _log(f"Meshes: policy={args.mesh_policy} max_meshes={args.max_meshes} candidates={len(candidates)}/{len(objs)}")

    for i, obj in enumerate(objs):
        source_id = str(obj.get("source_id") or "").strip()
        if not source_id:
            continue
        out_obj = layout_dir / "generation" / f"{source_id}.obj"
        if i in candidate_set:
            prompt_i = mesh_prompts.get(i, str(obj.get("type") or "object"))
            try:
                _sam3d_start_if_needed(str(args.sam3d_server))
                _log(f"SAM3D mesh [{i+1}/{len(objs)}]: {prompt_i!r} -> {out_obj.name}")
                _sam3d_generate_obj(prompt=prompt_i, out_obj_path=out_obj, seed=seed + i, server_base=str(args.sam3d_server))
                continue
            except Exception as exc:
                _log(f"WARNING: SAM3D mesh failed for {prompt_i!r}: {exc} (falling back to unit box)")
        _write_unit_box_obj(out_obj)

    # Write room jsons and pose-aug meta/variant.
    room_json = layout_dir / "room_0.json"
    _write_json(room_json, room)

    base_layout_json = layout_dir / f"{layout_id}.json"
    _write_json(base_layout_json, room)

    variant_json_rel = "variant_000.json"
    variant_json = layout_dir / args.pose_aug_name / variant_json_rel
    _write_json(variant_json, room)

    meta_json = layout_dir / args.pose_aug_name / "meta.json"
    _write_json(meta_json, [variant_json_rel])

    # Keep a copy of raw response for debugging.
    _write_json(layout_dir / "scenesmith_response.json", response)

    # Print layout id only (consumed by bash).
    sys.stdout.write(layout_id + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
