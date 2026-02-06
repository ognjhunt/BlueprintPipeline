#!/usr/bin/env python3
"""Patch: Enable PhysX contact reporting after init_robot.

Inserts a call to _bp_enable_contact_reporting() at the end of init_robot(),
so that PhysX ContactReportAPI is activated on prims after the scene is loaded.
"""
import re
import sys

SERVER_FILE = "/opt/geniesim/source/data_collection/server/grpc_server.py"

with open(SERVER_FILE, "r") as f:
    content = f.read()

# First, remove any duplicate bad patches from prior sed runs
content = re.sub(
    r'        # Enable PhysX contact reporting after robot init\n'
    r'        try:\n'
    r'            self\._bp_enable_contact_reporting\(\)\n'
    r'            print\("\[INIT_ROBOT\] PhysX contact reporting enabled\."\)\n'
    r'        except Exception as e:\n'
    r'            print\(f"\[INIT_ROBOT\] Failed to enable contact reporting: \{e\}"\)\n',
    '',
    content,
)

# Find the init_robot method and insert the call before 'return rsp'
# Pattern: the init_robot method ends with:
#   ) or "")
#         return rsp
#
#     def set_object_pose
pattern = r'(    def init_robot\(self, req, rsp\):.*?)(        return rsp\n\n    def set_object_pose)'
match = re.search(pattern, content, re.DOTALL)
if match:
    insert_point = match.start(2)
    injection = (
        '        # Enable PhysX contact reporting after robot init\n'
        '        try:\n'
        '            self._bp_enable_contact_reporting()\n'
        '            print("[INIT_ROBOT] PhysX contact reporting enabled.")\n'
        '        except Exception as e:\n'
        '            print(f"[INIT_ROBOT] Failed to enable contact reporting: {e}")\n'
    )
    content = content[:insert_point] + injection + content[insert_point:]
    with open(SERVER_FILE, "w") as f:
        f.write(content)
    print("[PATCH] Injected _bp_enable_contact_reporting() into init_robot()")
    # Verify single occurrence
    count = content.count("_bp_enable_contact_reporting()")
    print(f"[PATCH] Total occurrences of _bp_enable_contact_reporting(): {count}")
else:
    print("[PATCH] ERROR: Could not find init_robot pattern. Checking manually...")
    idx = content.find("def init_robot")
    if idx < 0:
        print("[PATCH] init_robot not found in file!")
        sys.exit(1)
    idx2 = content.find("def set_object_pose", idx)
    if idx2 < 0:
        print("[PATCH] set_object_pose not found after init_robot!")
        sys.exit(1)
    print(f"[PATCH] init_robot at {idx}, set_object_pose at {idx2}")
    block = content[idx:idx2]
    returns = [m.start() for m in re.finditer(r'return rsp', block)]
    print(f"[PATCH] Found {len(returns)} 'return rsp' in init_robot block")
    if returns:
        # Insert before last return rsp
        abs_pos = idx + returns[-1]
        injection = (
            '        # Enable PhysX contact reporting after robot init\n'
            '        try:\n'
            '            self._bp_enable_contact_reporting()\n'
            '            print("[INIT_ROBOT] PhysX contact reporting enabled.")\n'
            '        except Exception as e:\n'
            '            print(f"[INIT_ROBOT] Failed to enable contact reporting: {e}")\n'
        )
        content = content[:abs_pos] + injection + content[abs_pos:]
        with open(SERVER_FILE, "w") as f:
            f.write(content)
        print("[PATCH] Injected via fallback method")
    else:
        print("[PATCH] No return rsp found - cannot patch")
        sys.exit(1)
