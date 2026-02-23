import json
import math
import sys
import types
from pathlib import Path

import pytest
import numpy as np


@pytest.mark.unit
def test_sage_scene_quality_push_apart_resolves_overlap():
    from scripts.runpod_sage import sage_scene_quality as q

    room = {
        "room_type": "test",
        "dimensions": {"width": 10.0, "length": 10.0, "height": 3.0},
        "objects": [
            # Support surface (not involved in collision)
            {
                "type": "table",
                "id": "table_0",
                "source_id": "table_0",
                "position": {"x": 1.0, "y": 1.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "dimensions": {"width": 2.0, "length": 1.0, "height": 0.75},
            },
            # Two manipulables overlapping in XY and Z
            {
                "type": "mug",
                "id": "mug_0",
                "source_id": "mug_0",
                "position": {"x": 1.0, "y": 1.0, "z": 0.75},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "dimensions": {"width": 0.10, "length": 0.10, "height": 0.10},
            },
            {
                "type": "cup",
                "id": "cup_0",
                "source_id": "cup_0",
                "position": {"x": 1.05, "y": 1.0, "z": 0.75},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "dimensions": {"width": 0.10, "length": 0.10, "height": 0.10},
            },
        ],
    }

    before = q.evaluate_room(room)
    assert before["num_colliding_pairs"] >= 1

    rep = q.repair_room_inplace(room, profile=q._PROFILES["strict"], max_iters=10, max_corrected_ratio=1.0)
    assert rep["pass_after"] is True

    after = rep["after"]
    assert after["num_colliding_pairs"] == 0


@pytest.mark.unit
def test_sage_scene_quality_normalizes_condiment_mass_and_snaps_tabletop():
    from scripts.runpod_sage import sage_scene_quality as q

    room = {
        "room_type": "kitchen",
        "dimensions": {"width": 6.0, "length": 6.0, "height": 3.0},
        "objects": [
            {
                "type": "table",
                "id": "table_0",
                "source_id": "table_0",
                "position": {"x": 2.0, "y": 2.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "dimensions": {"width": 1.2, "length": 1.0, "height": 0.58},
            },
            {
                "type": "pepper_shaker",
                "id": "pepper_0",
                "source_id": "pepper_0",
                "position": {"x": 2.0, "y": 2.0, "z": 0.03},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "dimensions": {"width": 0.06, "length": 0.06, "height": 0.11},
                "physics": {"mass": 3.2},
            },
        ],
    }

    rep = q.repair_room_inplace(
        room,
        profile=q._PROFILES["strict"],
        max_iters=4,
        auto_fix=True,
        max_corrected_ratio=1.0,
    )
    assert rep["pass_after"] is True

    pepper = next(obj for obj in room["objects"] if obj["id"] == "pepper_0")
    assert pepper["physics"]["mass"] == pytest.approx(0.3, abs=1e-6)
    assert pepper["position"]["z"] == pytest.approx(0.585, abs=1e-3)
    correction_reasons = {entry["reason"] for entry in rep["corrections"]}
    assert "mass_preset_normalization" in correction_reasons
    assert "tabletop_floor_correction" in correction_reasons or "surface_snap" in correction_reasons


@pytest.mark.unit
def test_sage_scene_quality_corrected_ratio_gate():
    from scripts.runpod_sage import sage_scene_quality as q

    objects = [
        {
            "type": "table",
            "id": "table_0",
            "source_id": "table_0",
            "position": {"x": 2.0, "y": 2.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "dimensions": {"width": 1.2, "length": 1.0, "height": 0.58},
        }
    ]
    for idx in range(8):
        objects.append(
            {
                "type": "salt_shaker",
                "id": f"salt_{idx}",
                "source_id": f"salt_{idx}",
                "position": {"x": 2.0, "y": 2.0, "z": -0.02},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "dimensions": {"width": 0.05, "length": 0.05, "height": 0.10},
                "physics": {"mass": 4.0},
            }
        )
    room = {
        "room_type": "kitchen",
        "dimensions": {"width": 6.0, "length": 6.0, "height": 3.0},
        "objects": objects,
    }
    rep = q.repair_room_inplace(
        room,
        profile=q._PROFILES["strict"],
        max_iters=3,
        auto_fix=True,
        max_corrected_ratio=0.2,
    )
    assert rep["pass_after"] is False
    assert "corrected_object_ratio_exceeds_threshold" in rep["errors_after"]


@pytest.mark.unit
def test_scenesmith_to_sage_axis_conversion_right_handed():
    from scripts.runpod_sage import scenesmith_to_sage_layout as conv

    A = conv._A

    # Orthonormal: A * A^T == I
    AT = conv._mat3_T(A)
    I = conv._mat3_mul(A, AT)
    for i in range(3):
        for j in range(3):
            if i == j:
                assert abs(I[i][j] - 1.0) < 1e-9
            else:
                assert abs(I[i][j]) < 1e-9

    # Right-handed: det(A) == +1
    det = (
        A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
    )
    assert abs(det - 1.0) < 1e-9

    # Cross product consistency: A(e_x) x A(e_y) == A(e_z)
    ex_g = (A[0][0], A[1][0], A[2][0])  # A * (1,0,0)
    ey_g = (A[0][1], A[1][1], A[2][1])  # A * (0,1,0)
    ez_g = (A[0][2], A[1][2], A[2][2])  # A * (0,0,1)

    cx = (
        ex_g[1] * ey_g[2] - ex_g[2] * ey_g[1],
        ex_g[2] * ey_g[0] - ex_g[0] * ey_g[2],
        ex_g[0] * ey_g[1] - ex_g[1] * ey_g[0],
    )
    assert all(abs(cx[i] - ez_g[i]) < 1e-9 for i in range(3))


@pytest.mark.unit
def test_scenesmith_to_sage_room_bounds_and_shift_positive():
    from scripts.runpod_sage import scenesmith_to_sage_layout as conv

    raw = [
        {
            "id": "a",
            "category": "table",
            "sim_role": "static",
            "transform": {
                "position": {"x": -1.0, "y": 0.0, "z": -2.0},  # y-up (y==height)
                "rotation_quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            },
            "dimensions_est": {"width": 1.0, "height": 0.75, "depth": 1.5},
        },
        {
            "id": "b",
            "category": "mug",
            "sim_role": "manipulable_object",
            "transform": {
                "position": {"x": 0.5, "y": 0.0, "z": -1.0},
                "rotation_quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            },
            "dimensions_est": {"width": 0.1, "height": 0.1, "depth": 0.1},
        },
    ]

    margin = 0.6
    room = conv._convert_objects_to_sage(raw, margin_m=margin, room_type="kitchen", seed=123)
    dims = room["dimensions"]
    assert dims["width"] > 2.0 * margin
    assert dims["length"] > 2.0 * margin

    min_x = 1e9
    min_y = 1e9
    max_x = -1e9
    max_y = -1e9
    for o in room["objects"]:
        p = o["position"]
        d = o["dimensions"]
        cx = float(p["x"])
        cy = float(p["y"])
        w = float(d["width"])
        l = float(d["length"])
        min_x = min(min_x, cx - 0.5 * w)
        max_x = max(max_x, cx + 0.5 * w)
        min_y = min(min_y, cy - 0.5 * l)
        max_y = max(max_y, cy + 0.5 * l)

    assert min_x >= margin - 1e-6
    assert min_y >= margin - 1e-6
    assert max_x <= float(dims["width"]) - margin + 1e-6
    assert max_y <= float(dims["length"]) - margin + 1e-6


@pytest.mark.unit
def test_scenesmith_to_sage_reuses_existing_glb_matches_by_object_tokens(tmp_path):
    from scripts.runpod_sage import scenesmith_to_sage_layout as conv

    run_dir = tmp_path / "run"
    house_state_dir = run_dir / "scene_demo_resume" / "scene_000" / "combined_house"
    house_state_dir.mkdir(parents=True, exist_ok=True)
    house_state_path = house_state_dir / "house_state.json"
    house_state_path.write_text(
        json.dumps(
            {
                "rooms": {
                    "kitchen": {
                        "objects": [
                            {
                                "id": "salt_shaker_1",
                                "semantic_class": "salt_shaker",
                                "pose": {"position": {"x": 0, "y": 0, "z": 0}},
                                "extent": {"x": 0.1, "y": 0.1, "z": 0.2},
                            },
                            {
                                "id": "mug_0",
                                "semantic_class": "mug",
                                "pose": {"position": {"x": 1, "y": 0, "z": 0}},
                                "extent": {"x": 0.1, "y": 0.1, "z": 0.1},
                            },
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    glb_salt = run_dir / "generated_assets" / "salt_shaker_1771814906.glb"
    glb_mug = run_dir / "generated_assets" / "mug_1771814907.glb"
    glb_salt.parent.mkdir(parents=True, exist_ok=True)
    glb_salt.write_bytes(b"dummy")
    glb_mug.write_bytes(b"dummy")

    raw_objects = [
        {"id": "salt_shaker_1", "name": "salt shaker", "category": "salt_shaker"},
        {"id": "mug_0", "name": "mug", "category": "mug"},
    ]
    response = {
        "paper_stack": {
            "run_dir": str(run_dir),
            "house_state_path": str(house_state_path),
        }
    }

    pool = conv._collect_existing_glb_pool(response=response, raw_objects=raw_objects)
    first = conv._pick_existing_glb_for_object(pool, raw_objects[0])
    second = conv._pick_existing_glb_for_object(pool, raw_objects[1])

    assert first is not None
    assert second is not None
    assert first.name.startswith("salt_shaker")
    assert second.name.startswith("mug")
    assert first != second


@pytest.mark.unit
def test_scenesmith_to_sage_glb_match_avoids_partial_token_false_positive(tmp_path):
    from scripts.runpod_sage import scenesmith_to_sage_layout as conv

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    wall_glb = run_dir / "generated_assets" / "north_wall_1771814907.glb"
    wall_glb.parent.mkdir(parents=True, exist_ok=True)
    wall_glb.write_bytes(b"dummy")

    response = {"paper_stack": {"run_dir": str(run_dir), "house_state_path": ""}}
    raw_objects = [{"id": "wall_clock_1", "name": "wall clock", "category": "wall_clock"}]
    pool = conv._collect_existing_glb_pool(response=response, raw_objects=raw_objects)
    picked = conv._pick_existing_glb_for_object(pool, raw_objects[0])
    assert picked is None


@pytest.mark.unit
def test_scenesmith_to_sage_glb_warm_match_avoids_substring_false_positive(tmp_path):
    from scripts.runpod_sage import scenesmith_to_sage_layout as conv

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    template_glb = run_dir / "generated_assets" / "template_plate_holder.glb"
    template_glb.parent.mkdir(parents=True, exist_ok=True)
    template_glb.write_bytes(b"dummy")

    response = {"paper_stack": {"run_dir": str(run_dir), "house_state_path": ""}}
    raw_objects = [{"id": "plate_1", "name": "plate", "category": "plate"}]
    pool = conv._collect_existing_glb_pool(response=response, raw_objects=raw_objects)
    picked = conv._pick_existing_glb_for_object(pool, raw_objects[0])
    assert picked is None


@pytest.mark.unit
def test_scenesmith_to_sage_layout_payload_wraps_room_with_rooms_key():
    from scripts.runpod_sage import scenesmith_to_sage_layout as conv

    room = {
        "room_type": "kitchen",
        "dimensions": {"width": 6.0, "length": 5.0, "height": 3.0},
        "seed": 123,
        "scene_source": "scenesmith",
        "objects": [{"id": "salt_0", "type": "salt_shaker"}],
        "policy_analysis": {"task": "pick and place"},
    }
    payload = conv._single_room_layout_payload(room, layout_id="layout_test")
    assert payload["layout_id"] == "layout_test"
    assert isinstance(payload.get("rooms"), list) and len(payload["rooms"]) == 1
    assert payload["rooms"][0]["objects"][0]["id"] == "salt_0"
    assert payload["policy_analysis"]["task"] == "pick and place"


@pytest.mark.unit
def test_sage_scene_quality_extract_room_payload_accepts_layout_wrapper(tmp_path):
    from scripts.runpod_sage import sage_scene_quality as q

    payload = {
        "layout_id": "layout_test",
        "rooms": [
            {
                "room_type": "kitchen",
                "dimensions": {"width": 4.0, "length": 4.0, "height": 3.0},
                "objects": [],
            }
        ],
    }
    full, room = q._extract_room_payload(payload, path=tmp_path / "variant.json")
    assert full is payload
    assert room is payload["rooms"][0]


@pytest.mark.unit
def test_stage567_build_mesh_dict_list_accepts_flat_room_variant(tmp_path, monkeypatch):
    from scripts.runpod_sage import sage_stage567_mobile_franka as stage567

    variant_json = tmp_path / "variant_000.json"
    variant_json.write_text(
        json.dumps(
            {
                "room_type": "kitchen",
                "dimensions": {"width": 4.0, "length": 4.0, "height": 3.0},
                "objects": [{"id": "salt_0", "type": "salt_shaker"}],
            }
        ),
        encoding="utf-8",
    )

    seen = {}

    def _fake_dict_to_floor_plan(layout_data):
        seen["layout_data"] = layout_data
        room = types.SimpleNamespace(id="room_0")
        return types.SimpleNamespace(rooms=[room])

    def _fake_export_single_room_layout_to_mesh_dict_list(layout, rid):
        return {"rid": rid, "room_count": len(layout.rooms)}

    monkeypatch.setitem(
        sys.modules,
        "tex_utils",
        types.SimpleNamespace(
            dict_to_floor_plan=_fake_dict_to_floor_plan,
            export_single_room_layout_to_mesh_dict_list=_fake_export_single_room_layout_to_mesh_dict_list,
        ),
    )

    out = stage567._build_mesh_dict_list(variant_json, "room_0")
    assert out["rid"] == "room_0"
    assert out["room_count"] == 1
    assert isinstance(seen["layout_data"].get("rooms"), list)


@pytest.mark.unit
def test_stage567_ensure_mesh_textures_fills_missing_texture(tmp_path):
    from scripts.runpod_sage import sage_stage567_mobile_franka as stage567

    class _Mesh:
        def __init__(self):
            self.vertices = np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            )
            self.faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    variant_json = tmp_path / "layout_foo" / "pose_aug_0" / "variant_000.json"
    variant_json.parent.mkdir(parents=True, exist_ok=True)
    variant_json.write_text("{}", encoding="utf-8")

    mesh_info = {"salt_shaker_0": {"mesh": _Mesh(), "texture": None}}
    out = stage567._ensure_mesh_textures(mesh_info, variant_json)
    tex = out["salt_shaker_0"]["texture"]
    assert isinstance(tex["vts"], np.ndarray)
    assert isinstance(tex["fts"], np.ndarray)
    assert Path(tex["texture_map_path"]).exists()
