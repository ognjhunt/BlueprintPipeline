#!/usr/bin/env python3
"""
Assemble a SAGE room into a single viewable 3D scene.

Reads room_*.json + generation/*.obj and creates a combined scene
that can be opened in Blender, macOS Preview, or any GLB viewer.

Usage:
    python assemble_scene.py /path/to/SAGE_Run6_SAM3D_LivingRoom

Outputs:
    /path/to/SAGE_Run6_SAM3D_LivingRoom/combined_scene.glb
    /path/to/SAGE_Run6_SAM3D_LivingRoom/combined_scene.obj  (fallback)
"""

import json
import sys
import os
import math
import glob
from pathlib import Path

def rotation_matrix_z(angle_deg):
    """Create a 3x3 rotation matrix around Z axis."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]

def transform_vertex(vx, vy, vz, rot_z_deg, tx, ty, tz):
    """Apply rotation (Z-axis) then translation to a vertex."""
    rot = rotation_matrix_z(rot_z_deg)
    rx = rot[0][0] * vx + rot[0][1] * vy + rot[0][2] * vz
    ry = rot[1][0] * vx + rot[1][1] * vy + rot[1][2] * vz
    rz = rot[2][0] * vx + rot[2][1] * vy + rot[2][2] * vz
    return rx + tx, ry + ty, rz + tz

def read_obj(filepath):
    """Read vertices and faces from an OBJ file."""
    vertices = []
    faces = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v' and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                face = []
                for p in parts[1:]:
                    # Handle f v, f v/vt, f v/vt/vn, f v//vn
                    idx = int(p.split('/')[0])
                    face.append(idx)
                faces.append(face)
    return vertices, faces

def get_obj_bounds(vertices):
    """Get min/max bounds of vertices."""
    if not vertices:
        return [0,0,0], [0,0,0]
    mins = [min(v[i] for v in vertices) for i in range(3)]
    maxs = [max(v[i] for v in vertices) for i in range(3)]
    return mins, maxs

def assemble_scene(run_dir):
    run_dir = Path(run_dir)

    # Find layout directory
    results_dir = run_dir / "results"
    if not results_dir.exists():
        print(f"ERROR: No 'results' directory in {run_dir}")
        sys.exit(1)

    layout_dirs = list(results_dir.glob("layout_*"))
    if not layout_dirs:
        print(f"ERROR: No layout_* directory found in {results_dir}")
        sys.exit(1)

    layout_dir = layout_dirs[0]
    print(f"Layout: {layout_dir.name}")

    # Find room JSON
    room_jsons = list(layout_dir.glob("room_*.json"))
    if not room_jsons:
        print(f"ERROR: No room_*.json found in {layout_dir}")
        sys.exit(1)

    room_json_path = room_jsons[0]
    print(f"Room: {room_json_path.name}")

    with open(room_json_path) as f:
        room = json.load(f)

    gen_dir = layout_dir / "generation"

    print(f"\nRoom: {room['room_type']} ({room['dimensions']['width']}m x {room['dimensions']['length']}m x {room['dimensions']['height']}m)")
    print(f"Objects: {len(room['objects'])}")
    print()

    # Collect all transformed geometry
    all_vertices = []
    all_faces = []
    all_colors = []  # per-object color for identification
    vertex_offset = 0

    placed_count = 0
    missing_count = 0

    # Add room floor as a quad
    w = room['dimensions']['width']
    l = room['dimensions']['length']
    h = room['dimensions']['height']

    floor_verts = [[0, 0, 0], [w, 0, 0], [w, l, 0], [0, l, 0]]
    floor_faces = [[1, 2, 3, 4]]
    all_vertices.extend(floor_verts)
    all_faces.extend([[vi + vertex_offset for vi in face] for face in floor_faces])
    vertex_offset += len(floor_verts)

    # Add walls as thin boxes
    wall_h = h
    wall_t = 0.05  # 5cm thick walls
    wall_verts = []
    wall_faces_list = []

    # South wall (y=0)
    wall_verts.extend([[0, -wall_t, 0], [w, -wall_t, 0], [w, 0, 0], [0, 0, 0],
                       [0, -wall_t, wall_h], [w, -wall_t, wall_h], [w, 0, wall_h], [0, 0, wall_h]])
    nv = len(wall_verts) - 8
    wall_faces_list.extend([[nv+1,nv+2,nv+6,nv+5], [nv+2,nv+3,nv+7,nv+6], [nv+3,nv+4,nv+8,nv+7], [nv+4,nv+1,nv+5,nv+8]])

    # North wall (y=l)
    wall_verts.extend([[0, l, 0], [w, l, 0], [w, l+wall_t, 0], [0, l+wall_t, 0],
                       [0, l, wall_h], [w, l, wall_h], [w, l+wall_t, wall_h], [0, l+wall_t, wall_h]])
    nv = len(wall_verts) - 8
    wall_faces_list.extend([[nv+1,nv+2,nv+6,nv+5], [nv+2,nv+3,nv+7,nv+6], [nv+3,nv+4,nv+8,nv+7], [nv+4,nv+1,nv+5,nv+8]])

    # West wall (x=0)
    wall_verts.extend([[-wall_t, 0, 0], [0, 0, 0], [0, l, 0], [-wall_t, l, 0],
                       [-wall_t, 0, wall_h], [0, 0, wall_h], [0, l, wall_h], [-wall_t, l, wall_h]])
    nv = len(wall_verts) - 8
    wall_faces_list.extend([[nv+1,nv+2,nv+6,nv+5], [nv+2,nv+3,nv+7,nv+6], [nv+3,nv+4,nv+8,nv+7], [nv+4,nv+1,nv+5,nv+8]])

    # East wall (x=w)
    wall_verts.extend([[w, 0, 0], [w+wall_t, 0, 0], [w+wall_t, l, 0], [w, l, 0],
                       [w, 0, wall_h], [w+wall_t, 0, wall_h], [w+wall_t, l, wall_h], [w, l, wall_h]])
    nv = len(wall_verts) - 8
    wall_faces_list.extend([[nv+1,nv+2,nv+6,nv+5], [nv+2,nv+3,nv+7,nv+6], [nv+3,nv+4,nv+8,nv+7], [nv+4,nv+1,nv+5,nv+8]])

    all_vertices.extend(wall_verts)
    all_faces.extend([[vi + vertex_offset for vi in face] for face in wall_faces_list])
    vertex_offset += len(wall_verts)

    # Place each object
    for obj in room['objects']:
        source_id = obj.get('source_id', '')
        obj_file = gen_dir / f"{source_id}.obj"

        if not obj_file.exists():
            print(f"  MISSING: {obj['type']} ({obj['id']}) — {source_id}.obj not found")
            missing_count += 1
            continue

        vertices, faces = read_obj(str(obj_file))
        if not vertices:
            print(f"  EMPTY: {obj['type']} ({obj['id']}) — no vertices")
            continue

        # Get OBJ bounds for scaling
        mins, maxs = get_obj_bounds(vertices)
        obj_size = [maxs[i] - mins[i] for i in range(3)]

        # Target size from room JSON (in meters)
        target_w = obj['dimensions']['width']
        target_l = obj['dimensions']['length']
        target_h = obj['dimensions']['height']

        # Calculate scale factors
        sx = target_w / max(obj_size[0], 1e-6)
        sy = target_l / max(obj_size[1], 1e-6)
        sz = target_h / max(obj_size[2], 1e-6)

        # Use uniform scale (average) to avoid distortion
        # Actually use per-axis to match room layout exactly

        # Object position (room coordinates, in meters)
        px = obj['position']['x']
        py = obj['position']['y']
        pz = obj['position']['z']

        # Rotation (degrees around Z axis)
        rot_z = obj['rotation']['z']

        # Transform: center, scale, rotate, translate
        center = [(mins[i] + maxs[i]) / 2 for i in range(3)]

        transformed = []
        for v in vertices:
            # Center
            vx = v[0] - center[0]
            vy = v[1] - center[1]
            vz = v[2] - mins[2]  # Bottom at z=0

            # Scale
            vx *= sx
            vy *= sy
            vz *= sz

            # Rotate + translate
            fx, fy, fz = transform_vertex(vx, vy, vz, rot_z, px, py, pz)
            transformed.append([fx, fy, fz])

        # Add to combined mesh with offset
        all_vertices.extend(transformed)
        for face in faces:
            all_faces.append([vi + vertex_offset for vi in face])
        vertex_offset += len(transformed)

        placed_count += 1
        print(f"  ✅ {obj['type']:12s} ({source_id}) — {len(vertices)} verts, pos=({px:.1f}, {py:.1f}, {pz:.1f}), rot={rot_z}°")

    print(f"\nPlaced: {placed_count}/{len(room['objects'])} objects")
    if missing_count:
        print(f"Missing: {missing_count} OBJ files")

    # Write combined OBJ
    out_obj = run_dir / "combined_scene.obj"
    out_mtl = run_dir / "combined_scene.mtl"

    with open(out_obj, 'w') as f:
        f.write(f"# SAGE Combined Scene — {room['room_type']}\n")
        f.write(f"# {placed_count} objects + walls + floor\n")
        f.write(f"# Room: {w}m x {l}m x {h}m\n")
        f.write(f"mtllib combined_scene.mtl\n\n")

        for v in all_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write(f"\n# {len(all_faces)} faces\n")
        for face in all_faces:
            f.write("f " + " ".join(str(vi) for vi in face) + "\n")

    with open(out_mtl, 'w') as f:
        f.write("# SAGE Combined Scene Materials\n")
        f.write("newmtl default\n")
        f.write("Kd 0.7 0.7 0.7\n")

    print(f"\n✅ Combined scene written to:")
    print(f"   {out_obj}")
    print(f"   ({len(all_vertices)} vertices, {len(all_faces)} faces)")

    # Try to also create GLB using trimesh if available
    try:
        import trimesh
        scene = trimesh.Scene()

        # Load and place each object
        for obj_data in room['objects']:
            source_id = obj_data.get('source_id', '')
            obj_file = gen_dir / f"{source_id}.obj"
            if not obj_file.exists():
                continue

            try:
                mesh = trimesh.load(str(obj_file), force='mesh')
            except Exception:
                try:
                    loaded = trimesh.load(str(obj_file))
                    if isinstance(loaded, trimesh.Scene):
                        meshes = list(loaded.geometry.values())
                        if meshes:
                            mesh = trimesh.util.concatenate(meshes)
                        else:
                            continue
                    else:
                        mesh = loaded
                except Exception:
                    continue

            # Scale to target dimensions
            bounds = mesh.bounds
            current_size = bounds[1] - bounds[0]
            target = [obj_data['dimensions']['width'],
                      obj_data['dimensions']['length'],
                      obj_data['dimensions']['height']]

            scale = [target[i] / max(current_size[i], 1e-6) for i in range(3)]

            # Center mesh at origin, bottom at z=0
            center = (bounds[0] + bounds[1]) / 2
            mesh.vertices -= center
            mesh.vertices[:, 2] -= mesh.vertices[:, 2].min()

            # Apply scale
            mesh.vertices[:, 0] *= scale[0]
            mesh.vertices[:, 1] *= scale[1]
            mesh.vertices[:, 2] *= scale[2]

            # Create transform matrix
            rot_z = obj_data['rotation']['z']
            angle = math.radians(rot_z)
            transform = trimesh.transformations.euler_matrix(0, 0, angle)
            transform[0, 3] = obj_data['position']['x']
            transform[1, 3] = obj_data['position']['y']
            transform[2, 3] = obj_data['position']['z']

            scene.add_geometry(mesh, node_name=obj_data['id'], transform=transform)

        # Add floor
        floor = trimesh.creation.box(extents=[w, l, 0.02])
        floor_transform = trimesh.transformations.translation_matrix([w/2, l/2, -0.01])
        scene.add_geometry(floor, node_name='floor', transform=floor_transform)

        out_glb = run_dir / "combined_scene.glb"
        scene.export(str(out_glb))
        print(f"   {out_glb} (GLB — open in Blender/macOS Quick Look)")

    except ImportError:
        print(f"\n   (Install trimesh for GLB output: pip install trimesh)")
    except Exception as e:
        print(f"\n   GLB export failed: {e}")
        print(f"   Use the .obj file instead")

    print(f"\nOpen in:")
    print(f"  • Blender: File → Import → Wavefront (.obj)")
    print(f"  • macOS: open {out_obj}  (Preview/Quick Look)")
    print(f"  • Online: https://3dviewer.net (drag & drop)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/SAGE_Run6_SAM3D_LivingRoom")
        sys.exit(1)
    assemble_scene(sys.argv[1])
