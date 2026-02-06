#!/usr/bin/env python3
"""Add convexHull collision approximation to all kinematic objects in scene.usda."""

import sys

scene_path = sys.argv[1] if len(sys.argv) > 1 else "test_scenes/scenes/lightwheel_kitchen/usd/scene.usda"

with open(scene_path, "r") as f:
    content = f.read()

kinematic_objects = [
    "obj_Refrigerator001",
    "obj_Dishwasher054",
    "obj_Microwave017",
    "obj_Sink054",
    "obj_Stovetop012",
    "obj_RangeHood015",
    "obj_Kitchen_Cabinet001",
    "obj_Kitchen_Cabinet002",
    "obj_Table049",
    "obj_WallStackOven004",
    "obj_CoffeeMachine006",
    "obj_Pot057",
]

count = 0
for obj_name in kinematic_objects:
    marker = '"' + obj_name + '"'
    idx = content.find(marker)
    if idx == -1:
        print("  %s: not found" % obj_name)
        continue

    # Find kinematicEnabled within the next 500 chars of this object
    search_region = content[idx:idx + 500]
    kin_pos = search_region.find("bool physics:kinematicEnabled = true")
    if kin_pos == -1:
        print("  %s: no kinematicEnabled found" % obj_name)
        continue

    # Check if already has approximation
    if "physics:approximation" in search_region:
        print("  %s: already has approximation" % obj_name)
        continue

    # Insert physics:approximation after the kinematicEnabled line
    abs_pos = idx + kin_pos
    end_of_line = content.find("\n", abs_pos)
    insert = '\n            uniform token physics:approximation = "convexHull"'
    content = content[:end_of_line] + insert + content[end_of_line:]
    count += 1
    print("  %s: added convexHull collision approximation" % obj_name)

with open(scene_path, "w") as f:
    f.write(content)
print("\nDone. Updated %d objects." % count)
