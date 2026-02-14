#!/usr/bin/env python3
"""
SAGE → BlueprintPipeline Post-Processing Bridge.

Takes SAGE pipeline outputs and applies BlueprintPipeline quality layers:
  1. Asset Validation (VLM-based scoring of generated 3D objects)
  2. SimReady Lite Analysis (physics/collision recommendations)
  3. Grasp Quality Analysis (basic sanity + distribution)
  4. Demo Analysis (HDF5 structure + sensor sanity)
  5. Diversity Metrics (scene/object/pose diversity)
  6. Quality Report (aggregate)

Usage:
    python sage_to_bp_postprocess.py \
        --sage_results /workspace/SAGE/server/results/layout_XXXXXXXX \
        --output_dir /workspace/outputs/layout_XXXXXXXX_bp

Output:
    <output_dir>/
        quality_report.json          - Aggregate quality report
        asset_validation/             - Per-asset VLM scores
        scene_manifest.json           - BP-format scene manifest
        certification/                - Per-episode certification results
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Add BlueprintPipeline to path
BP_DIR = os.environ.get("BP_DIR", "/workspace/BlueprintPipeline")
sys.path.insert(0, BP_DIR)

from scripts.runpod_sage.bp_simready_lite import recommend_simready_lite, preset_to_dict


def load_sage_room(sage_results_dir):
    """Load SAGE room JSON."""
    results = Path(sage_results_dir)

    # Find room JSON
    room_jsons = list(results.glob("room_*.json"))
    if not room_jsons:
        # Try inside layout subdirectory
        for subdir in results.iterdir():
            if subdir.is_dir():
                room_jsons = list(subdir.glob("room_*.json"))
                if room_jsons:
                    break

    if not room_jsons:
        print("[BP-POST] No room_*.json found")
        return None

    with open(room_jsons[0]) as f:
        return json.load(f)


def convert_to_scene_manifest(room_data, sage_results_dir):
    """Convert SAGE room JSON to BlueprintPipeline scene_manifest.json format."""
    results = Path(sage_results_dir)
    gen_dir = results / "generation"

    objects = []
    for obj in room_data.get("objects", []):
        source_id = obj.get("source_id", "")
        obj_path = gen_dir / f"{source_id}.obj"

        bp_obj = {
            "name": obj.get("type", "unknown"),
            "category": obj.get("type", "unknown"),
            "id": obj.get("id", ""),
            "source_id": source_id,
            "position": obj.get("position", {}),
            "rotation": obj.get("rotation", {}),
            "dimensions_est": {
                "width": obj.get("dimensions", {}).get("width", 0.5),
                "height": obj.get("dimensions", {}).get("height", 0.5),
                "depth": obj.get("dimensions", {}).get("length", 0.5),
            },
            "source_kind": "sage_sam3d",
            "source_path": str(obj_path) if obj_path.exists() else "",
            "mesh_exists": obj_path.exists(),
        }
        objects.append(bp_obj)

    manifest = {
        "scene_id": Path(sage_results_dir).name,
        "room_type": room_data.get("room_type", "unknown"),
        "dimensions": room_data.get("dimensions", {}),
        "objects": objects,
        "source": "sage",
        "pipeline_version": "sage+bp_v1",
    }

    return manifest


def run_asset_validation(manifest, sage_results_dir, output_dir):
    """Run VLM-based asset validation on each generated 3D object."""
    results = []

    try:
        from tools.source_pipeline.asset_validation import (
            validate_asset_candidate,
            validate_asset_with_vlm,
        )
    except ImportError:
        print("[BP-POST] asset_validation not available — using deterministic scoring")
        from tools.source_pipeline.asset_validation import validate_asset_candidate
        validate_asset_with_vlm = None

    gen_dir = Path(sage_results_dir) / "generation"
    val_dir = Path(output_dir) / "asset_validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    for obj in manifest.get("objects", []):
        obj_dir = gen_dir / obj.get("source_id", "")

        if validate_asset_with_vlm:
            result = validate_asset_with_vlm(
                obj=obj,
                asset_dir=gen_dir,
                source_kind="sage_sam3d",
                source_path=obj.get("source_path", ""),
                room_type=manifest.get("room_type", ""),
            )
        else:
            result = validate_asset_candidate(
                obj=obj,
                source_kind="sage_sam3d",
                source_path=obj.get("source_path", ""),
                room_type=manifest.get("room_type", ""),
            )

        result["object_name"] = obj.get("name", "unknown")
        result["object_id"] = obj.get("id", "")
        results.append(result)

    # Save per-asset results
    with open(val_dir / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    passed = sum(1 for r in results if r.get("passed", False))
    avg_score = sum(r.get("score", 0) for r in results) / max(len(results), 1)

    return {
        "total_assets": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "avg_score": round(avg_score, 4),
        "results_file": str(val_dir / "validation_results.json"),
    }


def run_simready_analysis(manifest, output_dir):
    """
    Analyze assets for "SimReady Lite" (pre-collection physics presets).

    NOTE: This does NOT run V-HACD or bake full SimReady USD assets. It only
    recommends values that can be applied in Isaac Sim prior to capture.
    """
    analysis = {
        "objects": [],
        "summary": {},
    }

    for obj in manifest.get("objects", []):
        category = obj.get("category", "unknown").lower()
        dims = obj.get("dimensions_est", {})
        preset = recommend_simready_lite(category=category, dimensions={
            "width": dims.get("width", 0.0),
            "length": dims.get("depth", 0.0),
            "height": dims.get("height", 0.0),
        })
        preset_dict = preset_to_dict(preset)

        volume = float(
            (dims.get("width", 0.0) or 0.0)
            * (dims.get("height", 0.0) or 0.0)
            * (dims.get("depth", 0.0) or 0.0)
        )

        obj_analysis = {
            "name": obj.get("name", "unknown"),
            "id": obj.get("id", ""),
            "mesh_exists": obj.get("mesh_exists", False),
            "recommended_physics": preset_dict,
            "estimated_volume_m3": round(volume, 6),
            "collision_shape": preset.collision_approximation,
            "collision_max_hulls": preset.collision_max_hulls,
            "is_articulated": False,  # SAGE objects are rigid
            "simready_ready": obj.get("mesh_exists", False),
        }
        analysis["objects"].append(obj_analysis)

    ready = sum(1 for o in analysis["objects"] if o["simready_ready"])
    analysis["summary"] = {
        "total": len(analysis["objects"]),
        "simready_ready": ready,
        "needs_mesh": len(analysis["objects"]) - ready,
    }

    sr_dir = Path(output_dir) / "simready"
    sr_dir.mkdir(parents=True, exist_ok=True)
    with open(sr_dir / "simready_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    return analysis["summary"]


def run_grasp_quality_analysis(sage_results_dir, output_dir):
    """Analyze grasp quality from M2T2 outputs."""
    grasp_file = Path(sage_results_dir) / "grasps" / "grasp_transforms.json"

    if not grasp_file.exists():
        return {"status": "no_grasps", "num_grasps": 0}

    with open(grasp_file) as f:
        grasp_data = json.load(f)

    num_grasps = grasp_data.get("num_grasps", 0)
    if num_grasps == 0:
        return {"status": "no_grasps", "num_grasps": 0}

    # Basic grasp analysis (without full force closure computation)
    grasps = grasp_data.get("grasps", [])
    import numpy as np

    grasp_positions = []
    grasp_orientations = []
    for g in grasps:
        g_np = np.array(g)
        if g_np.shape == (4, 4):
            grasp_positions.append(g_np[:3, 3])
            grasp_orientations.append(g_np[:3, :3])

    if grasp_positions:
        positions = np.array(grasp_positions)
        # Spread metric: how diverse are the grasp positions?
        spread = np.std(positions, axis=0).mean()
        # Height distribution
        height_mean = positions[:, 2].mean()
        height_std = positions[:, 2].std()
    else:
        spread = 0.0
        height_mean = 0.0
        height_std = 0.0

    result = {
        "status": "analyzed",
        "num_grasps": num_grasps,
        "source": grasp_data.get("source", "unknown"),
        "position_spread_m": round(float(spread), 4),
        "height_mean_m": round(float(height_mean), 4),
        "height_std_m": round(float(height_std), 4),
        "quality_estimate": "good" if num_grasps >= 5 else "limited",
    }

    gq_dir = Path(output_dir) / "grasp_quality"
    gq_dir.mkdir(parents=True, exist_ok=True)
    with open(gq_dir / "grasp_quality.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_demo_analysis(sage_results_dir, output_dir):
    """Analyze generated demos (HDF5 or JSON)."""
    demo_dir = Path(sage_results_dir) / "demos"

    if not demo_dir.exists():
        return {"status": "no_demos"}

    # Check for HDF5
    hdf5_file = demo_dir / "dataset.hdf5"
    meta_file = demo_dir / "demo_metadata.json"
    decomp_file = demo_dir / "step_decomposition.json"
    capture_manifest = demo_dir / "capture_manifest.json"

    result = {"status": "analyzed"}

    if hdf5_file.exists():
        try:
            import h5py
            import numpy as np
            with h5py.File(str(hdf5_file), "r") as f:
                data = f["data"]
                num_demos = len(data.keys())
                total_steps = sum(
                    data[k]["states"].shape[0] for k in data.keys()
                )
                result["format"] = "hdf5"
                result["num_demos"] = num_demos
                result["total_steps"] = total_steps
                result["avg_steps_per_demo"] = total_steps / max(num_demos, 1)

                # Check observation keys
                first_demo = list(data.keys())[0]
                obs_keys = list(data[first_demo]["obs"].keys())
                result["observation_keys"] = obs_keys
                result["has_rgb"] = any("rgb" in k for k in obs_keys)
                result["has_depth"] = any("depth" in k for k in obs_keys)

                # Sanity: ensure at least one RGB and one depth stream is non-degenerate.
                rgb_ok = False
                depth_ok = False
                for key in obs_keys:
                    if ("rgb" in key) and (not rgb_ok):
                        arr = data[first_demo]["obs"][key]
                        # Read first frame only.
                        frame0 = np.array(arr[0])
                        rgb_ok = bool(frame0.size > 0 and frame0.std() > 1.0)
                        result.setdefault("rgb_sanity", {})[key] = {
                            "std": float(frame0.std()) if frame0.size else 0.0,
                            "shape": list(frame0.shape),
                        }
                    if ("depth" in key) and (not depth_ok):
                        arr = data[first_demo]["obs"][key]
                        frame0 = np.array(arr[0], dtype=np.float32)
                        finite = np.isfinite(frame0)
                        depth_ok = bool(frame0.size > 0 and finite.any() and frame0[finite].std() > 1e-4)
                        result.setdefault("depth_sanity", {})[key] = {
                            "finite_frac": float(finite.mean()) if frame0.size else 0.0,
                            "std": float(frame0[finite].std()) if finite.any() else 0.0,
                            "shape": list(frame0.shape),
                        }
                result["rgb_non_degenerate"] = rgb_ok
                result["depth_non_degenerate"] = depth_ok
        except ImportError:
            result["format"] = "hdf5"
            result["note"] = "h5py not available for detailed analysis"
    elif meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        result["format"] = "json+npy"
        result["num_demos"] = meta.get("num_demos", 0)

    if decomp_file.exists():
        with open(decomp_file) as f:
            decomp = json.load(f)
        result["step_decomposition"] = {
            "num_demos": decomp.get("num_demos", 0),
            "phase_labels": decomp.get("phase_labels", []),
        }

    if capture_manifest.exists():
        try:
            with open(capture_manifest) as f:
                manifest = json.load(f)
            result["capture_manifest"] = {
                "cameras": manifest.get("cameras", []),
                "resolution": manifest.get("resolution"),
                "modalities": manifest.get("modalities", []),
            }
        except Exception:
            result["capture_manifest"] = {"error": "failed_to_read"}

    demo_analysis_dir = Path(output_dir) / "demo_analysis"
    demo_analysis_dir.mkdir(parents=True, exist_ok=True)
    with open(demo_analysis_dir / "demo_analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_diversity_metrics(manifest, sage_results_dir, output_dir):
    """Compute scene diversity metrics."""
    objects = manifest.get("objects", [])

    # Object type diversity
    categories = [obj.get("category", "unknown").lower() for obj in objects]
    unique_categories = set(categories)

    # Spatial diversity
    positions = []
    for obj in objects:
        pos = obj.get("position", {})
        if "x" in pos and "y" in pos:
            positions.append((pos["x"], pos["y"]))

    import numpy as np
    spatial_spread = 0.0
    if len(positions) > 1:
        pos_array = np.array(positions)
        spatial_spread = float(np.std(pos_array, axis=0).mean())

    result = {
        "num_objects": len(objects),
        "unique_categories": len(unique_categories),
        "category_diversity": round(len(unique_categories) / max(len(objects), 1), 4),
        "categories": sorted(unique_categories),
        "spatial_spread_m": round(spatial_spread, 4),
        "diversity_score": round(
            0.5 * len(unique_categories) / max(len(objects), 1) +
            0.5 * min(spatial_spread / 2.0, 1.0),
            4,
        ),
    }

    div_dir = Path(output_dir) / "diversity"
    div_dir.mkdir(parents=True, exist_ok=True)
    with open(div_dir / "diversity_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def validate_required_outputs(
    room_dir: Path,
    manifest: dict,
    asset_val: dict,
    simready: dict,
    grasp_quality: dict,
    demo_analysis: dict,
    output_dir: str,
) -> None:
    """Fail fast when mandatory outputs are missing or obviously invalid."""
    strict = os.getenv("BP_QUALITY_GATE", "").lower() in {"1", "true", "yes"} or \
        os.getenv("BP_STRICT", "").lower() in {"1", "true", "yes"}
    if not strict:
        return

    errors: list[str] = []

    if not manifest.get("objects"):
        errors.append("No objects in scene manifest.")

    if asset_val.get("total_assets", 0) <= 0:
        errors.append("No assets validated.")

    grasp_num = int(grasp_quality.get("num_grasps", 0) or 0)
    if grasp_num <= 0:
        errors.append("No grasps available.")

    if demo_analysis.get("format") == "hdf5":
        if demo_analysis.get("num_demos", 0) <= 0:
            errors.append("No demos in HDF5 dataset.")
        if demo_analysis.get("has_rgb") and demo_analysis.get("rgb_non_degenerate") is False:
            errors.append("RGB observation stream is degenerate.")
        if demo_analysis.get("has_depth") and demo_analysis.get("depth_non_degenerate") is False:
            errors.append("Depth observation stream is degenerate.")
    else:
        errors.append("Demo analysis missing HDF5 output.")

    # Require all mandatory files generated by the collector for strict mode.
    required_files = [
        room_dir / "grasps" / "grasp_transforms.json",
        room_dir / "demos" / "dataset.hdf5",
        room_dir / "demos" / "demo_metadata.json",
        room_dir / "demos" / "step_decomposition.json",
        room_dir / "demos" / "capture_manifest.json",
    ]
    for path in required_files:
        if not path.exists():
            errors.append(f"Missing required artifact: {path}")

    if simready.get("summary", {}).get("total", 0) < len(manifest.get("objects", [])):
        errors.append("SimReady analysis missing entries for some objects.")

    if errors:
        raise RuntimeError("BP strict validation failed: " + "; ".join(errors))


def generate_quality_report(
    manifest, asset_val, simready, grasp_quality,
    demo_analysis, diversity, output_dir, sage_results_dir,
    elapsed_time,
):
    """Generate aggregate quality report."""
    report = {
        "pipeline": "SAGE + BlueprintPipeline",
        "sage_version": "NVlabs/sage@main",
        "bp_version": "ognjhunt/BlueprintPipeline@main",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processing_time_seconds": round(elapsed_time, 1),
        "scene": {
            "layout_id": Path(sage_results_dir).name,
            "room_type": manifest.get("room_type", "unknown"),
            "dimensions": manifest.get("dimensions", {}),
            "num_objects": len(manifest.get("objects", [])),
        },
        "quality_layers": {
            "asset_validation": asset_val,
            "simready_analysis": simready,
            "grasp_quality": grasp_quality,
            "demo_analysis": demo_analysis,
            "diversity_metrics": diversity,
        },
        "overall_score": 0.0,
    }

    # Compute overall score
    scores = []
    if asset_val.get("avg_score", 0) > 0:
        scores.append(asset_val["avg_score"])
    if diversity.get("diversity_score", 0) > 0:
        scores.append(diversity["diversity_score"])
    if grasp_quality.get("num_grasps", 0) > 0:
        scores.append(0.8)  # Grasps exist → reasonable score
    if demo_analysis.get("num_demos", 0) > 0:
        scores.append(0.7)  # Demos exist → reasonable score

    report["overall_score"] = round(sum(scores) / max(len(scores), 1), 4)

    # Save report
    output_path = Path(output_dir) / "quality_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description="SAGE → BP post-processing")
    parser.add_argument("--sage_results", required=True,
                        help="SAGE results directory (layout_XXXXXXXX)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for BP quality results")
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 60)
    print("SAGE → BlueprintPipeline Post-Processing")
    print("=" * 60)
    print(f"SAGE results: {args.sage_results}")
    print(f"BP output:    {args.output_dir}")
    print()

    # Load SAGE room data
    room = load_sage_room(args.sage_results)
    if room is None:
        print("[ERROR] Could not load SAGE room data")
        sys.exit(1)

    # Convert to BP manifest
    manifest = convert_to_scene_manifest(room, args.sage_results)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "scene_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Scene: {manifest['room_type']} ({len(manifest['objects'])} objects)")
    print()

    # Run quality layers
    print("1/6 Asset Validation...")
    try:
        asset_val = run_asset_validation(manifest, args.sage_results, args.output_dir)
        print(f"     {asset_val['passed']}/{asset_val['total_assets']} passed (avg: {asset_val['avg_score']:.3f})")
    except Exception as e:
        print(f"     FAILED: {e}")
        asset_val = {"error": str(e)}

    print("2/6 SimReady Analysis...")
    try:
        simready = run_simready_analysis(manifest, args.output_dir)
        print(f"     {simready['simready_ready']}/{simready['total']} ready")
    except Exception as e:
        print(f"     FAILED: {e}")
        simready = {"error": str(e)}

    print("3/6 Grasp Quality...")
    try:
        grasp_quality = run_grasp_quality_analysis(args.sage_results, args.output_dir)
        print(f"     {grasp_quality.get('num_grasps', 0)} grasps — {grasp_quality.get('quality_estimate', 'unknown')}")
    except Exception as e:
        print(f"     FAILED: {e}")
        grasp_quality = {"error": str(e)}

    print("4/6 Demo Analysis...")
    try:
        demo_analysis = run_demo_analysis(args.sage_results, args.output_dir)
        print(f"     {demo_analysis.get('num_demos', 0)} demos")
    except Exception as e:
        print(f"     FAILED: {e}")
        demo_analysis = {"error": str(e)}

    print("5/6 Diversity Metrics...")
    try:
        diversity = run_diversity_metrics(manifest, args.sage_results, args.output_dir)
        print(f"     Score: {diversity.get('diversity_score', 0):.3f} ({diversity.get('unique_categories', 0)} unique categories)")
    except Exception as e:
        print(f"     FAILED: {e}")
        diversity = {"error": str(e)}

    print("6/6 Quality Report...")
    elapsed = time.time() - t0
    report = generate_quality_report(
        manifest, asset_val, simready, grasp_quality,
        demo_analysis, diversity, args.output_dir,
        args.sage_results, elapsed,
    )
    print(f"     Overall score: {report['overall_score']:.3f}")

    # Strict postprocess gate: fail early when required artifacts are absent/invalid.
    validate_required_outputs(
        Path(args.sage_results),
        manifest,
        asset_val,
        simready,
        grasp_quality,
        demo_analysis,
        args.output_dir,
    )

    # Optional strict gating
    if os.getenv("BP_QUALITY_GATE", "").lower() in {"1", "true", "yes"}:
        min_score = float(os.getenv("BP_QUALITY_MIN_SCORE", "0.7"))
        if report["overall_score"] < min_score:
            print(
                f"[BP-POST] ERROR: Quality gate failed: score={report['overall_score']:.3f} < {min_score:.3f}",
                file=sys.stderr,
            )
            sys.exit(2)

    print()
    print("=" * 60)
    print(f"Quality Report: {args.output_dir}/quality_report.json")
    print(f"Overall Score:  {report['overall_score']:.3f}")
    print(f"Processing:     {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
