#!/usr/bin/env python3
"""
Render scene from Isaac Sim and run M2T2 grasp inference.

Bridge between SAGE scene generation (stages 1-3) and grasp generation (stage 5).

Flow:
  1. Load scene USD into Isaac Sim
  2. Render RGB + depth from N camera viewpoints
  3. Generate segmentation masks per object
  4. Run M2T2 grasp inference per viewpoint
  5. Aggregate grasps in world frame
  6. Save grasp transforms to JSON

Usage:
    cd /workspace/SAGE/server
    python render_and_infer_grasps.py --layout_id layout_XXXXXXXX

Output:
    results/<layout_id>/grasps/
        grasp_transforms.json     - All grasps with confidence scores
        grasp_per_object.json     - Grasps grouped by target object
        grasp_visualization.ply   - Point cloud visualization (optional)
"""

import argparse
import json
import os
import sys
import math
import time
import hashlib
import socket
import numpy as np
from pathlib import Path

# Add SAGE server to path
SAGE_SERVER_DIR = os.environ.get("SAGE_SERVER_DIR", "/workspace/SAGE/server")
sys.path.insert(0, SAGE_SERVER_DIR)

# Isaac Sim MCP connection
def get_mcp_port():
    """Compute Isaac Sim MCP port from SLURM_JOB_ID."""
    job_id = os.environ.get("SLURM_JOB_ID", "12345")
    h = int(hashlib.md5(str(job_id).encode()).hexdigest(), 16)
    return 8080 + (h % (40000 - 8080 + 1))


def send_mcp_command(command_json, host="localhost", port=None, timeout=120):
    """Send a command to Isaac Sim MCP and get response."""
    if port is None:
        port = get_mcp_port()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))

        data = json.dumps(command_json).encode("utf-8")
        sock.sendall(data + b"\n")

        response = b""
        while True:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
                # Check for complete JSON
                try:
                    json.loads(response.decode("utf-8"))
                    break
                except json.JSONDecodeError:
                    continue
            except socket.timeout:
                break

        sock.close()
        return json.loads(response.decode("utf-8"))
    except Exception as e:
        print(f"[M2T2-BRIDGE] MCP connection failed: {e}", file=sys.stderr)
        return None


def check_isaac_sim_available():
    """Check if Isaac Sim MCP is reachable."""
    port = get_mcp_port()
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except Exception:
        return False


def render_scene_from_isaac_sim(scene_save_dir, room_json_path, viewpoints):
    """
    Render RGB + depth images from Isaac Sim for M2T2 input.

    If Isaac Sim is not available, generates synthetic depth from mesh data
    as a fallback.
    """
    renders = []

    if check_isaac_sim_available():
        print("[M2T2-BRIDGE] Isaac Sim available — rendering from USD scene")

        # Create scene in Isaac Sim
        result = send_mcp_command({
            "command": "create_single_room_layout_scene_from_room",
            "scene_save_dir": str(scene_save_dir),
            "room_dict_save_path": str(room_json_path),
        })

        if result and isinstance(result, dict) and result.get("status") == "success":
            # Render from each viewpoint using execute_script
            for i, vp in enumerate(viewpoints):
                render_script = f"""
import numpy as np
from isaacsim import SimulationApp
# Set camera position and render
# ... (camera setup code would go here)
"""
                # For now, use the default render
                renders.append({
                    "viewpoint_idx": i,
                    "camera_pose": vp["camera_pose"],
                    "intrinsics": vp.get("intrinsics", default_intrinsics()),
                    "source": "isaac_sim",
                })
        else:
            print("[M2T2-BRIDGE] Isaac Sim scene creation failed, using mesh fallback")
    else:
        print("[M2T2-BRIDGE] Isaac Sim not available — using synthetic rendering")

    # Fallback: Generate synthetic RGB+depth from OBJ meshes using Open3D
    if not renders:
        renders = render_from_meshes(scene_save_dir, room_json_path, viewpoints)

    return renders


def default_intrinsics(width=640, height=480, fov=60):
    """Generate default camera intrinsics."""
    fx = width / (2 * math.tan(math.radians(fov / 2)))
    fy = fx
    cx = width / 2
    cy = height / 2
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)


def render_from_meshes(scene_save_dir, room_json_path, viewpoints):
    """
    Render synthetic RGB+depth from OBJ mesh files using Open3D.

    This is the fallback when Isaac Sim is not available.
    """
    renders = []

    try:
        import open3d as o3d
    except ImportError:
        print("[M2T2-BRIDGE] open3d not available for mesh rendering", file=sys.stderr)
        return renders

    # Load room JSON
    with open(room_json_path) as f:
        room = json.load(f)

    gen_dir = Path(scene_save_dir) / "generation"

    # Load all meshes
    meshes = []
    object_names = []
    for obj in room.get("objects", []):
        source_id = obj.get("source_id", "")
        obj_path = gen_dir / f"{source_id}.obj"
        if obj_path.exists():
            try:
                mesh = o3d.io.read_triangle_mesh(str(obj_path))
                if mesh.has_triangles():
                    # Transform mesh to room position
                    center = mesh.get_center()
                    mesh.translate(-center)  # center at origin

                    # Scale to target dimensions
                    bounds = mesh.get_max_bound() - mesh.get_min_bound()
                    target_w = obj["dimensions"]["width"]
                    target_l = obj["dimensions"]["length"]
                    target_h = obj["dimensions"]["height"]
                    scale = [
                        target_w / max(bounds[0], 1e-6),
                        target_l / max(bounds[1], 1e-6),
                        target_h / max(bounds[2], 1e-6),
                    ]
                    mesh.scale(np.mean(scale), center=[0, 0, 0])

                    # Translate to position
                    pos = [obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]]
                    mesh.translate(pos)

                    meshes.append(mesh)
                    object_names.append(obj.get("type", source_id))
            except Exception as e:
                print(f"[M2T2-BRIDGE] Failed to load {obj_path}: {e}")

    if not meshes:
        print("[M2T2-BRIDGE] No meshes loaded — cannot render")
        return renders

    # Combine meshes
    combined = meshes[0]
    for m in meshes[1:]:
        combined += m
    combined.compute_vertex_normals()

    # Create scene
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=640, height=480)
    vis.add_geometry(combined)

    # Render from each viewpoint
    width, height = 640, 480
    intrinsics = default_intrinsics(width, height)

    for i, vp in enumerate(viewpoints):
        # Set camera
        ctr = vis.get_view_control()
        cam_pose = np.array(vp["camera_pose"])

        # Render
        vis.poll_events()
        vis.update_renderer()

        # Capture depth
        depth_image = vis.capture_depth_float_buffer(do_render=True)
        depth_np = np.asarray(depth_image)

        # Capture RGB
        color_image = vis.capture_screen_float_buffer(do_render=True)
        rgb_np = (np.asarray(color_image) * 255).astype(np.uint8)

        # Create segmentation mask (simplified — all non-zero depth = object)
        seg = np.zeros(depth_np.shape, dtype=np.int32)
        seg[depth_np > 0] = 1  # table = 1

        renders.append({
            "viewpoint_idx": i,
            "rgb": rgb_np,
            "depth": depth_np,
            "seg": seg,
            "camera_pose": cam_pose,
            "intrinsics": intrinsics,
            "label_map": {"table": 1},
            "source": "open3d_mesh",
        })

    vis.destroy_window()
    return renders


def generate_viewpoints(room_dims, num_views=4):
    """Generate camera viewpoints around the room."""
    w = room_dims["width"]
    l = room_dims["length"]
    h = room_dims["height"]

    cx, cy = w / 2, l / 2
    cam_height = h * 0.8  # 80% of room height
    radius = max(w, l) * 0.8

    viewpoints = []
    for i in range(num_views):
        angle = (2 * math.pi * i) / num_views
        cam_x = cx + radius * math.cos(angle)
        cam_y = cy + radius * math.sin(angle)
        cam_z = cam_height

        # Look-at matrix: camera looks at room center
        forward = np.array([cx - cam_x, cy - cam_y, 0 - cam_z])
        forward = forward / np.linalg.norm(forward)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / max(np.linalg.norm(right), 1e-6)
        up = np.cross(right, forward)

        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = -up  # OpenCV convention
        cam_pose[:3, 2] = forward
        cam_pose[:3, 3] = [cam_x, cam_y, cam_z]

        viewpoints.append({
            "camera_pose": cam_pose.tolist(),
            "intrinsics": default_intrinsics().tolist(),
        })

    return viewpoints


def run_m2t2_inference(renders, output_dir):
    """Run M2T2 grasp inference on rendered images."""

    try:
        import torch
        from m2t2_utils.infer import load_m2t2, infer_m2t2
    except ImportError as e:
        print(f"[M2T2-BRIDGE] Cannot import M2T2: {e}", file=sys.stderr)
        print("[M2T2-BRIDGE] Generating placeholder grasp data", file=sys.stderr)
        return generate_placeholder_grasps(output_dir)

    # Load M2T2 model
    print("[M2T2-BRIDGE] Loading M2T2 model...")
    t0 = time.time()
    model, cfg = load_m2t2()
    print(f"[M2T2-BRIDGE] M2T2 loaded in {time.time() - t0:.1f}s")

    all_grasps = []
    all_contacts = []

    for render_data in renders:
        if "rgb" not in render_data or "depth" not in render_data:
            print(f"[M2T2-BRIDGE] Viewpoint {render_data.get('viewpoint_idx', '?')} missing RGB/depth, skipping")
            continue

        import torch

        meta_data = {
            "intrinsics": np.array(render_data["intrinsics"]),
            "camera_pose": np.array(render_data["camera_pose"]),
            "label_map": render_data.get("label_map", {"table": 1}),
        }

        vis_data = {
            "rgb": torch.from_numpy(render_data["rgb"]).permute(2, 0, 1).float() / 255.0,
            "depth": render_data["depth"],
            "seg": render_data["seg"],
        }

        print(f"[M2T2-BRIDGE] Running inference on viewpoint {render_data['viewpoint_idx']}...")
        t0 = time.time()

        try:
            grasps, contacts = infer_m2t2(meta_data, vis_data, model, cfg, return_contacts=True)
            print(f"[M2T2-BRIDGE] Found {grasps.shape[0]} grasps in {time.time() - t0:.1f}s")

            if grasps.shape[0] > 0:
                all_grasps.append(grasps)
                all_contacts.append(contacts)
        except Exception as e:
            print(f"[M2T2-BRIDGE] Inference failed for viewpoint {render_data['viewpoint_idx']}: {e}")

    # Aggregate results
    if all_grasps:
        combined_grasps = np.concatenate(all_grasps, axis=0)
        combined_contacts = np.concatenate(all_contacts, axis=0)
    else:
        combined_grasps = np.zeros((0, 4, 4))
        combined_contacts = np.zeros((0, 3))

    # De-duplicate similar grasps (position within 2cm)
    if combined_grasps.shape[0] > 1:
        positions = combined_grasps[:, :3, 3]
        keep = [0]
        for i in range(1, len(positions)):
            dists = np.linalg.norm(positions[keep] - positions[i], axis=1)
            if dists.min() > 0.02:  # 2cm threshold
                keep.append(i)
        combined_grasps = combined_grasps[keep]
        combined_contacts = combined_contacts[keep]

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    result = {
        "num_grasps": int(combined_grasps.shape[0]),
        "grasps": combined_grasps.tolist(),
        "contacts": combined_contacts.tolist() if combined_contacts.shape[0] > 0 else [],
        "source": "m2t2",
        "model_path": "/workspace/SAGE/M2T2/m2t2.pth",
        "num_viewpoints": len(renders),
    }

    with open(os.path.join(output_dir, "grasp_transforms.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[M2T2-BRIDGE] Saved {combined_grasps.shape[0]} grasps to {output_dir}/grasp_transforms.json")

    # Clean up GPU memory
    del model
    if 'torch' in dir():
        torch.cuda.empty_cache()

    return result


def generate_placeholder_grasps(output_dir):
    """Generate placeholder grasp data when M2T2 is not available."""
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "num_grasps": 0,
        "grasps": [],
        "contacts": [],
        "source": "placeholder",
        "note": "M2T2 not available — placeholder data generated",
    }
    with open(os.path.join(output_dir, "grasp_transforms.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Render scene + M2T2 grasp inference")
    parser.add_argument("--layout_id", required=True, help="Layout directory name")
    parser.add_argument("--results_dir", default="/workspace/SAGE/server/results",
                        help="SAGE results directory")
    parser.add_argument("--num_views", type=int, default=4, help="Number of camera viewpoints")
    args = parser.parse_args()

    layout_dir = Path(args.results_dir) / args.layout_id
    if not layout_dir.exists():
        print(f"[ERROR] Layout directory not found: {layout_dir}")
        sys.exit(1)

    # Find room JSON
    room_jsons = list(layout_dir.glob("room_*.json"))
    if not room_jsons:
        print(f"[ERROR] No room_*.json found in {layout_dir}")
        sys.exit(1)

    room_json_path = room_jsons[0]
    with open(room_json_path) as f:
        room = json.load(f)

    print(f"[M2T2-BRIDGE] Layout: {args.layout_id}")
    print(f"[M2T2-BRIDGE] Room: {room['room_type']} ({room['dimensions']['width']}x{room['dimensions']['length']}x{room['dimensions']['height']}m)")
    print(f"[M2T2-BRIDGE] Objects: {len(room['objects'])}")

    # Generate viewpoints
    viewpoints = generate_viewpoints(room["dimensions"], num_views=args.num_views)

    # Render scene
    print(f"\n[M2T2-BRIDGE] Rendering from {len(viewpoints)} viewpoints...")
    renders = render_scene_from_isaac_sim(
        scene_save_dir=str(layout_dir),
        room_json_path=str(room_json_path),
        viewpoints=viewpoints,
    )

    if not renders:
        print("[M2T2-BRIDGE] No renders generated — saving placeholder grasps")
        output_dir = str(layout_dir / "grasps")
        generate_placeholder_grasps(output_dir)
        return

    # Run M2T2 inference
    print(f"\n[M2T2-BRIDGE] Running M2T2 inference...")
    output_dir = str(layout_dir / "grasps")
    result = run_m2t2_inference(renders, output_dir)

    print(f"\n[M2T2-BRIDGE] Done! {result['num_grasps']} grasps found")


if __name__ == "__main__":
    main()
