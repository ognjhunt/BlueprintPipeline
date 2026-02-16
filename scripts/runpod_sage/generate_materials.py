#!/usr/bin/env python3
"""Generate missing material files for a SAGE layout so that tex_utils can build mesh_dict_list.

SAGE's material generation normally relies on MatFuse (heavy diffusion model) + Holodeck assets,
neither of which are available in our lean Docker image.  This script generates the required
material files procedurally:

  materials/{floor_material}.png        – solid-color floor texture
  materials/{wall_material}.png         – solid-color wall texture
  materials/{window_id}_texture.png     – window glass/frame texture
  materials/{window_id}_tex_coords.pkl  – window UV map (vts, fts)
  materials/{door_material}_texture.png – door panel texture
  materials/{door_material}_tex_coords.pkl – door UV map (vts, fts)

The textures are simple procedural images (not photorealistic), but they are geometrically
correct so that M2T2 grasp inference and Isaac Sim rendering can proceed.

Usage:
  python generate_materials.py /workspace/SAGE/server/results/layout_XXXX
"""
import json
import numpy as np
import os
import pickle
import sys
from PIL import Image

if len(sys.argv) != 2:
    print("Usage: python generate_materials.py /path/to/layout_dir", file=sys.stderr)
    raise SystemExit(2)

LAYOUT_DIR = sys.argv[1]
MATERIALS_DIR = os.path.join(LAYOUT_DIR, "materials")
os.makedirs(MATERIALS_DIR, exist_ok=True)

# Find the layout JSON
layout_json = None
layout_id = os.path.basename(LAYOUT_DIR)
candidate = os.path.join(LAYOUT_DIR, f"{layout_id}.json")
if os.path.exists(candidate):
    layout_json = candidate
else:
    for f in os.listdir(LAYOUT_DIR):
        if f.endswith(".json") and not f.endswith(".bak"):
            layout_json = os.path.join(LAYOUT_DIR, f)
            break

if not layout_json:
    print("ERROR: No layout JSON found", file=sys.stderr)
    raise SystemExit(1)

d = json.load(open(layout_json))
rooms = d.get("rooms", [])
if not rooms:
    print("ERROR: No rooms in layout", file=sys.stderr)
    raise SystemExit(1)


# --- Texture generation helpers ---

def make_solid_texture(color_rgb, size=512):
    """Create a solid-color texture image."""
    img = np.full((size, size, 3), color_rgb, dtype=np.uint8)
    return Image.fromarray(img)


def make_window_texture(grid_xy=(1, 1), glass_rgb=(180, 210, 240), frame_rgb=(77, 77, 77), size=512):
    """Create a window texture with glass panes and frame grid."""
    nx, ny = grid_xy
    img = np.full((size, size, 3), frame_rgb, dtype=np.uint8)

    cell_w = size // nx
    cell_h = size // ny
    border = max(4, size // 64)  # frame thickness

    for i in range(nx):
        for j in range(ny):
            x0 = i * cell_w + border
            x1 = (i + 1) * cell_w - border
            y0 = j * cell_h + border
            y1 = (j + 1) * cell_h - border
            img[y0:y1, x0:x1] = glass_rgb

    return Image.fromarray(img)


def make_door_texture(color_rgb=(139, 90, 43), panel_rgb=(120, 75, 35), size=512):
    """Create a door panel texture with raised panels."""
    img = np.full((size, size, 3), color_rgb, dtype=np.uint8)

    # Two raised panels
    border = size // 10
    mid = size // 2
    panel_inset = size // 8

    for y0, y1 in [(border, mid - border // 2), (mid + border // 2, size - border)]:
        img[y0:y1, panel_inset:size - panel_inset] = panel_rgb

    return Image.fromarray(img)


def make_box_tex_coords(n_faces=12):
    """Create UV tex_coords for a trimesh box (12 triangular faces, 8 vertices).

    This matches the format that tex_utils.py expects when loading from pkl:
      vts: (N, 2) float32 – texture UV coordinates
      fts: (M, 3) int32  – face-to-texcoord indices

    For a box, trimesh creates 12 triangles (2 per face × 6 faces).
    We assign simple planar UV mapping for each face.
    """
    # A box in trimesh has 8 unique vertices and 12 faces.
    # For UV mapping, we need per-face-vertex UVs.
    # Simplest approach: create 4 UV coords per quad face (24 total for 6 faces)
    # and index them with face-texture indices.

    vts_list = []
    fts_list = []

    # For each of 6 box faces (2 triangles each), create 4 UV corners
    for face_idx in range(6):
        base = face_idx * 4
        # 4 UV corners for this quad face
        vts_list.extend([
            [0.0, 0.0],  # bottom-left
            [1.0, 0.0],  # bottom-right
            [1.0, 1.0],  # top-right
            [0.0, 1.0],  # top-left
        ])
        # Two triangles covering the quad
        fts_list.append([base + 0, base + 1, base + 2])
        fts_list.append([base + 0, base + 2, base + 3])

    vts = np.array(vts_list, dtype=np.float32)
    fts = np.array(fts_list, dtype=np.int32)

    return {"vts": vts, "fts": fts}


# --- Floor and wall material color lookup ---

MATERIAL_COLORS = {
    # Floors
    "hardwood": (160, 120, 80),
    "wood": (160, 120, 80),
    "tile": (200, 200, 200),
    "carpet": (140, 130, 120),
    "marble": (220, 215, 210),
    "concrete": (180, 180, 180),
    "laminate": (170, 140, 100),
    "vinyl": (190, 185, 175),
    "bamboo": (180, 150, 90),
    "stone": (165, 160, 155),
    "linoleum": (185, 180, 170),
    # Walls
    "drywall": (235, 230, 220),
    "plaster": (230, 225, 215),
    "brick": (160, 75, 55),
    "wallpaper": (220, 210, 200),
    "paint": (240, 235, 225),
    "stucco": (225, 218, 205),
    "paneling": (155, 115, 75),
    "wood_paneling": (155, 115, 75),
}


def get_material_color(material_name):
    """Get RGB color for a material name, with fuzzy matching."""
    key = material_name.lower().strip().replace(" ", "_")
    if key in MATERIAL_COLORS:
        return MATERIAL_COLORS[key]
    # Fuzzy match
    for k, v in MATERIAL_COLORS.items():
        if k in key or key in k:
            return v
    # Default neutral
    return (200, 195, 185)


# --- Generate materials for each room ---

generated = []

for room in rooms:
    room_id = room.get("id", "unknown")

    # 1. Floor material
    floor_mat = room.get("floor_material", "hardwood")
    floor_path = os.path.join(MATERIALS_DIR, f"{floor_mat}.png")
    if not os.path.exists(floor_path):
        color = get_material_color(floor_mat)
        make_solid_texture(color).save(floor_path)
        generated.append(f"floor: {floor_mat}.png")
        print(f"  Generated floor texture: {floor_mat}.png ({color})")

    # 2. Wall materials
    for wall in room.get("walls", []):
        wall_mat = wall.get("material", "drywall")
        wall_path = os.path.join(MATERIALS_DIR, f"{wall_mat}.png")
        if not os.path.exists(wall_path):
            color = get_material_color(wall_mat)
            make_solid_texture(color).save(wall_path)
            generated.append(f"wall: {wall_mat}.png")
            print(f"  Generated wall texture: {wall_mat}.png ({color})")

    # 3. Window materials
    for window in room.get("windows", []):
        win_id = window.get("id", "unknown_window")
        tex_path = os.path.join(MATERIALS_DIR, f"{win_id}_texture.png")
        coords_path = os.path.join(MATERIALS_DIR, f"{win_id}_tex_coords.pkl")

        if not os.path.exists(tex_path) or not os.path.exists(coords_path):
            grid = window.get("window_grid", [1, 1])
            if not isinstance(grid, (list, tuple)) or len(grid) != 2:
                grid = [1, 1]

            glass_color = window.get("glass_color", [180, 210, 240])
            frame_color = window.get("frame_color", [77, 77, 77])

            make_window_texture(
                grid_xy=tuple(grid),
                glass_rgb=tuple(glass_color[:3]) if len(glass_color) >= 3 else (180, 210, 240),
                frame_rgb=tuple(frame_color[:3]) if len(frame_color) >= 3 else (77, 77, 77),
            ).save(tex_path)

            tex_coords = make_box_tex_coords()
            with open(coords_path, "wb") as f:
                pickle.dump(tex_coords, f)

            generated.append(f"window: {win_id}")
            print(f"  Generated window material: {win_id} (grid={grid})")

    # 4. Door materials
    for door in room.get("doors", []):
        door_mat = door.get("door_material", "standard")
        tex_path = os.path.join(MATERIALS_DIR, f"{door_mat}_texture.png")
        coords_path = os.path.join(MATERIALS_DIR, f"{door_mat}_tex_coords.pkl")

        if not os.path.exists(tex_path) or not os.path.exists(coords_path):
            make_door_texture().save(tex_path)

            tex_coords = make_box_tex_coords()
            with open(coords_path, "wb") as f:
                pickle.dump(tex_coords, f)

            generated.append(f"door: {door_mat}")
            print(f"  Generated door material: {door_mat}")

print(f"\nGenerated {len(generated)} material files in {MATERIALS_DIR}")
print("Material generation complete.")
