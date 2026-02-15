import math

import pytest


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
                "position": {"x": 7.0, "y": 7.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "dimensions": {"width": 2.0, "length": 1.0, "height": 0.75},
            },
            # Two manipulables overlapping in XY and Z
            {
                "type": "mug",
                "id": "mug_0",
                "source_id": "mug_0",
                "position": {"x": 1.0, "y": 1.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "dimensions": {"width": 0.10, "length": 0.10, "height": 0.10},
            },
            {
                "type": "cup",
                "id": "cup_0",
                "source_id": "cup_0",
                "position": {"x": 1.05, "y": 1.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "dimensions": {"width": 0.10, "length": 0.10, "height": 0.10},
            },
        ],
    }

    before = q.evaluate_room(room)
    assert before["num_colliding_pairs"] >= 1

    rep = q.repair_room_inplace(room, profile=q._PROFILES["strict"], max_iters=10)
    assert rep["pass_after"] is True

    after = rep["after"]
    assert after["num_colliding_pairs"] == 0


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

