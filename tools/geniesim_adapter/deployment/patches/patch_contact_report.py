#!/usr/bin/env python3
"""
Patch the Genie Sim server's grpc_server.py to add get_contact_report.

This enables PhysX contact reporting via SimObservationService/get_contact_report.
The patch is idempotent and safe to run multiple times.
"""
import os
import re
import sys
import textwrap

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
GRPC_SERVER = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "server", "grpc_server.py",
)

PATCH_MARKER = "BlueprintPipeline contact_report patch"

CONTACT_HANDLER = textwrap.dedent("""\

    # --- BEGIN BlueprintPipeline contact_report patch ---
    def get_contact_report(self, request, context):
        \"\"\"Return a PhysX contact report for the current simulation step.\"\"\"
        try:
            try:
                from aimdk.protocol import geniesim_grpc_pb2 as _pb2
            except Exception:
                try:
                    import geniesim_grpc_pb2 as _pb2
                except Exception:
                    from tools.geniesim_adapter import geniesim_grpc_pb2 as _pb2

            ContactReportRsp = _pb2.ContactReportRsp
            ContactPoint = _pb2.ContactPoint
        except Exception:
            # If protobuf classes are unavailable, return empty response.
            return None

        contacts = []
        total_force = 0.0
        max_penetration = 0.0
        include_points = getattr(request, "include_points", False)

        try:
            from omni.physx import get_physx_interface
            physx = get_physx_interface()
            contact_data = None
            try:
                from omni.physx import get_physx_simulation_interface
                sim = get_physx_simulation_interface()
                if sim is not None and hasattr(sim, "get_contact_report"):
                    contact_data = sim.get_contact_report()
            except Exception:
                contact_data = None

            if contact_data is None and physx is not None and hasattr(physx, "get_contact_report"):
                contact_data = physx.get_contact_report()

            if contact_data:
                for contact in contact_data:
                    if isinstance(contact, dict):
                        body_a = str(contact.get("actor0", ""))
                        body_b = str(contact.get("actor1", ""))
                        impulse = float(contact.get("impulse", 0.0))
                        separation = float(contact.get("separation", 0.0))
                        position = contact.get("position", [0, 0, 0])
                        normal = contact.get("normal", [0, 0, 1])
                    else:
                        body_a = str(getattr(contact, "actor0", ""))
                        body_b = str(getattr(contact, "actor1", ""))
                        impulse = float(getattr(contact, "impulse", 0.0))
                        separation = float(getattr(contact, "separation", 0.0))
                        position = getattr(contact, "position", [0, 0, 0])
                        normal = getattr(contact, "normal", [0, 0, 1])

                    penetration = abs(separation)
                    total_force += impulse
                    max_penetration = max(max_penetration, penetration)

                    contacts.append(ContactPoint(
                        body_a=body_a,
                        body_b=body_b,
                        normal_force=impulse,
                        penetration_depth=penetration,
                        position=list(position) if include_points else [],
                        normal=list(normal) if include_points else [],
                    ))

            return ContactReportRsp(
                contacts=contacts,
                total_normal_force=total_force,
                max_penetration_depth=max_penetration,
            )
        except Exception:
            return ContactReportRsp(
                contacts=[],
                total_normal_force=0.0,
                max_penetration_depth=0.0,
            )
    # --- END BlueprintPipeline contact_report patch ---
""")


def patch_file() -> None:
    if not os.path.isfile(GRPC_SERVER):
        print(f"[PATCH] grpc_server.py not found at {GRPC_SERVER}")
        print("[PATCH] Skipping contact_report patch (server source not available)")
        sys.exit(0)

    with open(GRPC_SERVER, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] contact_report already patched — skipping")
        sys.exit(0)

    class_match = re.search(r"^class\s+SimObservationServiceServicer\\b.*?:", content, re.MULTILINE)
    if not class_match:
        print("[PATCH] SimObservationServiceServicer class not found — skipping")
        sys.exit(0)

    insert_start = class_match.end()
    next_class = re.search(r"^class\s+\\w+", content[insert_start:], re.MULTILINE)
    insert_at = insert_start + next_class.start() if next_class else len(content)

    method_indent = "    "
    method_match = re.search(r"^([ \\t]+)def\\s+\\w+\\(self", content[insert_start:insert_at], re.MULTILINE)
    if method_match:
        method_indent = method_match.group(1)

    indented_handler = "\n".join(
        (method_indent + line) if line.strip() else line
        for line in CONTACT_HANDLER.splitlines()
    ) + "\n"

    patched = content[:insert_at].rstrip() + "\n\n" + indented_handler + content[insert_at:]

    with open(GRPC_SERVER, "w") as f:
        f.write(patched)

    print("[PATCH] Injected get_contact_report into grpc_server.py")


if __name__ == "__main__":
    patch_file()
