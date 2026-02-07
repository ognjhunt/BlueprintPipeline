#!/usr/bin/env python3
"""Patch: Enable PhysX contact reporting after init_robot.

IMPORTANT: Contact reporting must be enabled on the SIM THREAD (command_controller),
NOT the gRPC thread (grpc_server), because USD stage operations deadlock from
non-main threads in Isaac Sim.

This patch:
1. Removes any old gRPC-thread injection from grpc_server.py
2. Adds the contact reporting call to handle_init_robot in command_controller.py
   (which runs on the sim thread via on_command_step dispatch)
"""
import os
import re
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")

# ── Step 1: Remove old injection from grpc_server.py (if present) ──────────────
GRPC_FILE = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "server", "grpc_server.py",
)
if not os.path.isfile(GRPC_FILE):
    print(f"[PATCH] grpc_server.py not found at {GRPC_FILE}")
    print("[PATCH] Skipping enable_contacts_on_init patch (server source not available)")
    sys.exit(0)

with open(GRPC_FILE, "r") as f:
    grpc = f.read()

# Remove any block between blocking_start_server call and return rsp that
# contains _bp_enable_contact_reporting
old_block_re = re.compile(
    r'        # Enable PhysX contact reporting after robot init.*?(?=        return rsp)',
    re.DOTALL,
)
if old_block_re.search(grpc):
    grpc = old_block_re.sub('', grpc)
    with open(GRPC_FILE, "w") as f:
        f.write(grpc)
    print("[PATCH] Removed old gRPC-thread contact reporting injection from grpc_server.py")
else:
    print("[PATCH] No gRPC-thread contact injection found in grpc_server.py (clean)")

# ── Step 2: Inject into command_controller.py handle_init_robot (sim thread) ───
CC_FILE = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "server", "command_controller.py",
)
if not os.path.isfile(CC_FILE):
    print(f"[PATCH] command_controller.py not found at {CC_FILE}")
    print("[PATCH] Skipping enable_contacts_on_init patch (command_controller not available)")
    sys.exit(0)

with open(CC_FILE, "r") as f:
    cc = f.read()

MARKER = "# BlueprintPipeline contact_reporting_on_init patch"

if MARKER in cc:
    print("[PATCH] Contact reporting already injected in command_controller.py — skipping")
    sys.exit(0)

# Find handle_init_robot's data_to_send = "success" line
# We insert the contact reporting BEFORE setting data_to_send so it runs on the sim thread
old_pattern = '        self.data_to_send = "success"\n\n    def handle_add_camera'
if old_pattern not in cc:
    # Try more flexible match
    match = re.search(
        r'(    def handle_init_robot\(self\):.*?)(        self\.data_to_send = "success"\n)',
        cc,
        re.DOTALL,
    )
    if not match:
        print("[PATCH] ERROR: Could not find handle_init_robot data_to_send pattern")
        sys.exit(1)
    insert_pos = match.end(2)
    injection = (
        f'        {MARKER}\n'
        '        try:\n'
        '            import os as _os\n'
        '            if _os.environ.get("BP_ENABLE_CONTACTS_ON_INIT", "1") == "1":\n'
        '                grpc_srv = getattr(self, "_grpc_server_ref", None)\n'
        '                if grpc_srv and hasattr(grpc_srv, "_bp_enable_contact_reporting"):\n'
        '                    grpc_srv._bp_enable_contact_reporting()\n'
        '                    print("[INIT_ROBOT] PhysX contact reporting enabled (sim thread).")\n'
        '                else:\n'
        '                    print("[INIT_ROBOT] No gRPC server ref — skipping contact reporting")\n'
        '        except Exception as _cr_e:\n'
        '            print(f"[INIT_ROBOT] Contact reporting failed: {_cr_e}")\n'
    )
    cc = cc[:insert_pos] + injection + cc[insert_pos:]
else:
    # Replace the simple pattern
    replacement = (
        f'        {MARKER}\n'
        '        try:\n'
        '            import os as _os\n'
        '            if _os.environ.get("BP_ENABLE_CONTACTS_ON_INIT", "1") == "1":\n'
        '                grpc_srv = getattr(self, "_grpc_server_ref", None)\n'
        '                if grpc_srv and hasattr(grpc_srv, "_bp_enable_contact_reporting"):\n'
        '                    grpc_srv._bp_enable_contact_reporting()\n'
        '                    print("[INIT_ROBOT] PhysX contact reporting enabled (sim thread).")\n'
        '                else:\n'
        '                    print("[INIT_ROBOT] No gRPC server ref — skipping contact reporting")\n'
        '        except Exception as _cr_e:\n'
        '            print(f"[INIT_ROBOT] Contact reporting failed: {_cr_e}")\n'
        '        self.data_to_send = "success"\n\n    def handle_add_camera'
    )
    cc = cc.replace(old_pattern, replacement, 1)

with open(CC_FILE, "w") as f:
    f.write(cc)

# ── Step 3: Wire up _grpc_server_ref on CommandController ──────────────────────
# The GrpcServer already has server_function pointing to CommandController.
# We need CommandController to have a back-reference to GrpcServer.
# Inject this into grpc_server.py's __init__ or start method.
with open(GRPC_FILE, "r") as f:
    grpc = f.read()

BACKREF_MARKER = "# BlueprintPipeline grpc_server_backref"
if BACKREF_MARKER not in grpc:
    # Find where rpc_server.start() is called or where server_function is set
    # In GrpcServer.__init__, self.server_function = server_function
    old_init = 'self.server_function = server_function'
    if old_init in grpc:
        new_init = (
            'self.server_function = server_function\n'
            f'        {BACKREF_MARKER}\n'
            '        self.server_function._grpc_server_ref = self'
        )
        grpc = grpc.replace(old_init, new_init, 1)
        with open(GRPC_FILE, "w") as f:
            f.write(grpc)
        print("[PATCH] Added _grpc_server_ref backref in grpc_server.py")
    else:
        print("[PATCH] WARNING: Could not find server_function assignment for backref")
else:
    print("[PATCH] grpc_server_ref backref already present")

print("[PATCH] Contact reporting now runs on SIM THREAD via handle_init_robot.")
print("[PATCH] Set BP_ENABLE_CONTACTS_ON_INIT=0 to disable.")
