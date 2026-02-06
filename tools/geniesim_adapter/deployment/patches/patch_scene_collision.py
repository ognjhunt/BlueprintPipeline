#!/usr/bin/env python3
"""
Patch: Add collision geometry to scene objects that have PhysicsCollisionAPI
on their Xform but lack collision on child mesh prims.

This runs inside the Isaac Sim container and patches the command_controller
to inject collision setup after init_robot loads the scene.
"""
import re

TARGET = "/opt/geniesim/source/data_collection/server/command_controller.py"
MARKER = "# [PATCH] scene_collision_injected"


def _read(path):
    with open(path, "r") as f:
        return f.read()


def _write(path, content):
    with open(path, "w") as f:
        f.write(content)


COLLISION_SETUP_CODE = '''
    # [PATCH] scene_collision_injected
    def _patch_add_scene_collision(self):
        """Add convexHull collision to kinematic scene objects after scene load."""
        try:
            from pxr import Usd, UsdGeom, UsdPhysics
            stage = self.my_world.stage
            if stage is None:
                print("[PATCH] scene_collision: no stage available")
                return

            scene_root = stage.GetPrimAtPath("/World/Scene")
            if not scene_root or not scene_root.IsValid():
                print("[PATCH] scene_collision: /World/Scene not found")
                return

            patched = 0
            for child in scene_root.GetChildren():
                # Only process kinematic objects (skip dynamic ones)
                if not child.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue

                rigid_api = UsdPhysics.RigidBodyAPI(child)
                kin_attr = rigid_api.GetKinematicEnabledAttr()
                if not kin_attr or not kin_attr.Get():
                    continue  # dynamic object, skip

                # Check if this Xform has CollisionAPI but its children dont
                if not child.HasAPI(UsdPhysics.CollisionAPI):
                    continue

                # Walk all descendant mesh prims and add CollisionAPI + convexHull
                from pxr import Sdf
                for desc in Usd.PrimRange(child):
                    if not desc.IsA(UsdGeom.Mesh):
                        continue
                    if desc.HasAPI(UsdPhysics.CollisionAPI):
                        continue  # already has collision
                    col_api = UsdPhysics.CollisionAPI.Apply(desc)
                    col_api.CreateCollisionEnabledAttr().Set(True)
                    desc.CreateAttribute(
                        "physics:approximation",
                        Sdf.ValueTypeNames.Token,
                    ).Set("convexHull")
                    patched += 1

            print("[PATCH] scene_collision: added collision to %d mesh prims" % patched)
        except Exception as e:
            print("[PATCH] scene_collision error: %s" % e)
'''

HOOK_CODE = '''
        # [PATCH] scene_collision_hook
        self._patch_add_scene_collision()
'''


def apply():
    code = _read(TARGET)

    if MARKER in code:
        print("[PATCH] scene_collision already applied -- skipping")
        return

    # 1. Add the method to the class
    # Find "def handle_init_robot" and insert our method before it
    anchor = "def handle_init_robot(self"
    idx = code.find(anchor)
    if idx == -1:
        print("[PATCH] scene_collision: could not find handle_init_robot")
        return

    # Find the start of the line (go back to the indentation)
    line_start = code.rfind("\n", 0, idx) + 1
    code = code[:line_start] + COLLISION_SETUP_CODE + "\n" + code[line_start:]

    # 2. Hook into handle_init_robot to call our method after init completes
    # Find the end of handle_init_robot where data_to_send is set
    # Look for the pattern: self.data_to_send = ... in handle_init_robot
    init_method_start = code.find("def handle_init_robot(self")
    if init_method_start == -1:
        print("[PATCH] scene_collision: could not find handle_init_robot after insertion")
        return

    # Find "self.data_to_send" within handle_init_robot (first occurrence after method start)
    send_idx = code.find("self.data_to_send", init_method_start)
    if send_idx == -1:
        print("[PATCH] scene_collision: could not find data_to_send in handle_init_robot")
        return

    # Insert our hook call just before self.data_to_send
    line_start2 = code.rfind("\n", 0, send_idx) + 1
    code = code[:line_start2] + HOOK_CODE + "\n" + code[line_start2:]

    _write(TARGET, code)
    print("[PATCH] scene_collision: injected collision setup into handle_init_robot")


if __name__ == "__main__":
    apply()
