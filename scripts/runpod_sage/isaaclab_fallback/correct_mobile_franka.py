"""
Fallback replacement for isaaclab.correct_mobile_franka.

The original module is NVIDIA-internal and not included in the public SAGE repo.
This provides stub implementations that:
  1. Log a warning that real robot feasibility correction is not available
  2. Return a "success" result so the pipeline can continue
  3. Preserve the existing layout without modification

The two functions imported by layout.py:
  - correct_mobile_franka_standalone(layout, room_id)
  - robot_task_feasibility_correction_for_room_standalone(layout, room_id)
"""

import json
import sys


async def correct_mobile_franka_standalone(layout, room_id=""):
    """
    Stub: Correct mobile Franka placement for feasibility.

    In the full NVIDIA implementation, this adjusts object positions
    to ensure the mobile Franka can reach pick/place targets with
    collision-free navigation paths.

    This fallback skips correction and returns the layout unchanged.

    Args:
        layout: FloorPlan object
        room_id: str, the room to correct

    Returns:
        JSON string with correction result
    """
    print(
        "[FALLBACK] correct_mobile_franka_standalone: Real robot feasibility "
        "correction not available (NVIDIA-internal module). "
        "Layout preserved without modification.",
        file=sys.stderr
    )
    return json.dumps({
        "status": "success",
        "message": "Robot feasibility correction skipped (fallback mode). "
                   "Layout preserved as-is. For full feasibility correction, "
                   "the NVIDIA-internal isaaclab.correct_mobile_franka module is required.",
        "corrections_applied": 0,
        "fallback": True,
    })


async def robot_task_feasibility_correction_for_room_standalone(layout, room_id=""):
    """
    Stub: Check and correct robot task feasibility for a room.

    In the full NVIDIA implementation, this:
    1. Spawns a mobile Franka in Isaac Sim
    2. Tests if pick/place objects are reachable
    3. Plans collision-free navigation paths
    4. Adjusts object positions if needed
    5. Removes objects that block the robot

    This fallback skips all checks and returns success.

    Args:
        layout: FloorPlan object
        room_id: str, the room to check

    Returns:
        JSON string with feasibility result
    """
    print(
        "[FALLBACK] robot_task_feasibility_correction_for_room_standalone: "
        "Real robot feasibility correction not available (NVIDIA-internal module). "
        "Returning success — objects preserved as placed by DFS solver.",
        file=sys.stderr
    )

    # Count objects in the specified room for the report
    num_objects = 0
    room_type = "unknown"
    if layout is not None:
        try:
            rooms = layout.rooms if hasattr(layout, 'rooms') else []
            for room in rooms:
                rid = getattr(room, 'id', '') or getattr(room, 'room_id', '')
                if not room_id or rid == room_id:
                    num_objects = len(getattr(room, 'objects', []))
                    room_type = getattr(room, 'room_type', 'unknown')
                    break
        except Exception:
            pass

    return json.dumps({
        "status": "success",
        "message": f"Robot task feasibility check skipped (fallback mode). "
                   f"{num_objects} objects in {room_type} room preserved as-is.",
        "room_id": room_id,
        "room_type": room_type,
        "num_objects": num_objects,
        "corrections_applied": 0,
        "objects_removed": 0,
        "objects_adjusted": 0,
        "fallback": True,
        "note": "For full robot feasibility correction with collision-free path "
                "planning, the NVIDIA-internal isaaclab.correct_mobile_franka "
                "module and a running Isaac Sim instance are required.",
    })
