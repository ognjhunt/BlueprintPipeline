#!/usr/bin/env python3
"""
Stage 5 (STRICT): Render-free M2T2 grasp inference using SAGE's native utilities.

This script intentionally does NOT:
- talk to Isaac Sim MCP
- fall back to Open3D
- generate placeholder grasps

It relies on SAGE's nvdiffrast-based pipeline:
  /workspace/SAGE/server/m2t2_utils/data.py::generate_m2t2_data
  /workspace/SAGE/server/m2t2_utils/infer.py::{load_m2t2, infer_m2t2}

Output:
  results/<layout_id>/grasps/grasp_transforms.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


SAGE_SERVER_DIR = os.environ.get("SAGE_SERVER_DIR", "/workspace/SAGE/server")
sys.path.insert(0, SAGE_SERVER_DIR)


def _log(msg: str) -> None:
    print(f"[stage5 {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def _call_generate_m2t2_data(generate_fn, *, layout_id: str, layout_dir: Path, layout_dict_path: Path, num_views: int) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    sig = inspect.signature(generate_fn)
    params = sig.parameters
    kwargs: Dict[str, Any] = {}
    if "layout_id" in params:
        kwargs["layout_id"] = layout_id
    if "layout_dir" in params:
        kwargs["layout_dir"] = str(layout_dir)
    if "results_dir" in params:
        kwargs["results_dir"] = str(layout_dir.parent)
    if "layout_dict_path" in params:
        kwargs["layout_dict_path"] = str(layout_dict_path)
    if "layout_dict_save_path" in params:
        kwargs["layout_dict_save_path"] = str(layout_dict_path)
    if "room_dict_save_path" in params:
        kwargs["room_dict_save_path"] = str(layout_dict_path)
    if "num_views" in params:
        kwargs["num_views"] = int(num_views)
    if "num_viewpoints" in params:
        kwargs["num_viewpoints"] = int(num_views)

    missing = [name for name, p in params.items() if p.default is inspect._empty and name not in kwargs]
    if missing:
        raise RuntimeError(f"generate_m2t2_data signature mismatch. Missing required params: {missing}. Full signature: {sig}")

    out = generate_fn(**kwargs)

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    if isinstance(out, tuple) and len(out) == 2:
        meta, vis = out
        if isinstance(meta, list) and isinstance(vis, list):
            for m, v in zip(meta, vis):
                pairs.append((m, v))
        elif isinstance(meta, dict) and isinstance(vis, dict):
            pairs.append((meta, vis))
    elif isinstance(out, dict):
        if "meta_data" in out and "vis_data" in out:
            pairs.append((out["meta_data"], out["vis_data"]))
        elif "meta_datas" in out and "vis_datas" in out:
            for m, v in zip(out["meta_datas"], out["vis_datas"]):
                pairs.append((m, v))
    elif isinstance(out, list):
        for item in out:
            if isinstance(item, dict) and "meta_data" in item and "vis_data" in item:
                pairs.append((item["meta_data"], item["vis_data"]))

    if not pairs:
        raise RuntimeError(f"generate_m2t2_data returned unsupported structure: {type(out).__name__}")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE Stage 5 (M2T2 grasps) — strict")
    parser.add_argument("--layout_id", required=True)
    parser.add_argument("--results_dir", default="/workspace/SAGE/server/results")
    parser.add_argument("--layout_dict_path", default="")
    parser.add_argument("--num_views", type=int, default=1)
    args = parser.parse_args()

    layout_dir = Path(args.results_dir) / args.layout_id
    if not layout_dir.exists():
        raise FileNotFoundError(f"Layout dir not found: {layout_dir}")

    if args.layout_dict_path:
        layout_dict_path = Path(args.layout_dict_path)
    else:
        # Use the first room_*.json by default.
        room_jsons = sorted(layout_dir.glob("room_*.json"))
        if not room_jsons:
            raise FileNotFoundError(f"No room_*.json found in {layout_dir}")
        layout_dict_path = room_jsons[0]

    if not layout_dict_path.exists():
        raise FileNotFoundError(f"layout_dict_path not found: {layout_dict_path}")

    from m2t2_utils.data import generate_m2t2_data
    from m2t2_utils.infer import load_m2t2, infer_m2t2

    _log("Loading M2T2 model...")
    model, cfg = load_m2t2()

    pairs = _call_generate_m2t2_data(
        generate_m2t2_data,
        layout_id=args.layout_id,
        layout_dir=layout_dir,
        layout_dict_path=layout_dict_path,
        num_views=args.num_views,
    )

    all_grasps = []
    all_contacts = []
    for meta, vis in pairs:
        out = infer_m2t2(meta, vis, model, cfg, return_contacts=True)
        if isinstance(out, tuple) and len(out) == 2:
            grasps, contacts = out
        else:
            grasps, contacts = out, np.zeros((0, 3), dtype=np.float32)
        grasps = np.asarray(grasps)
        contacts = np.asarray(contacts)
        if grasps.size == 0:
            continue
        all_grasps.append(grasps)
        all_contacts.append(contacts if contacts.size else np.zeros((grasps.shape[0], 3), dtype=np.float32))

    if not all_grasps:
        raise RuntimeError("Stage 5 produced 0 grasps (strict)")

    grasps = np.concatenate(all_grasps, axis=0).astype(np.float32)
    contacts = np.concatenate(all_contacts, axis=0).astype(np.float32) if all_contacts else np.zeros((grasps.shape[0], 3), dtype=np.float32)

    out_dir = layout_dir / "grasps"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_grasps": int(grasps.shape[0]),
        "grasps": grasps.tolist(),
        "contacts": contacts.tolist(),
        "source": "m2t2",
        "layout_dict_path": str(layout_dict_path),
        "num_views": int(args.num_views),
    }
    (out_dir / "grasp_transforms.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _log(f"Wrote {grasps.shape[0]} grasps -> {out_dir}/grasp_transforms.json")


if __name__ == "__main__":
    main()

