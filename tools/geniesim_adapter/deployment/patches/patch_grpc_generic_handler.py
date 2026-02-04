#!/usr/bin/env python3
"""
Patch the Genie Sim server to add a GenericRpcHandler that routes custom RPC methods
to their implementations in the patched service classes.

This is necessary because:
1. patch_contact_report.py adds get_contact_report method to ObservationService class
2. But gRPC only routes methods defined in the proto-generated *_pb2_grpc.py
3. This patch adds a GenericRpcHandler that intercepts calls to our custom methods
   and routes them to the patched class implementations

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

PATCH_MARKER = "BlueprintPipeline grpc_generic_handler patch"

# The GenericRpcHandler class that intercepts custom method calls
# NOTE: This implementation avoids importing external pb2 files to prevent
# protobuf version conflicts. Instead, it uses the pb2 classes already loaded
# by the contact_report patch.
GENERIC_HANDLER_CLASS = textwrap.dedent("""\

# --- BEGIN BlueprintPipeline grpc_generic_handler patch ---
class _BPGenericRpcHandler(grpc.GenericRpcHandler):
    \"\"\"Generic RPC handler for BlueprintPipeline-patched methods.

    Routes custom gRPC methods (not defined in server's proto) to their
    implementations in patched service classes.
    \"\"\"

    def __init__(self, observation_service):
        self._obs_svc = observation_service

    def service(self, handler_call_details):
        method = handler_call_details.method

        # Route get_contact_report to ObservationService
        if method.endswith("/get_contact_report"):
            return grpc.unary_unary_rpc_method_handler(
                self._handle_contact_report,
                request_deserializer=self._deserialize_contact_req,
                response_serializer=self._serialize_contact_rsp,
            )

        # Route get_ee_wrench to ObservationService (if implemented)
        if method.endswith("/get_ee_wrench"):
            if hasattr(self._obs_svc, "get_ee_wrench"):
                return grpc.unary_unary_rpc_method_handler(
                    self._handle_ee_wrench,
                    request_deserializer=self._deserialize_ee_wrench_req,
                    response_serializer=self._serialize_ee_wrench_rsp,
                )

        return None

    def _get_pb2(self):
        \"\"\"Lazily get pb2 module using direct file import to avoid path shadowing.\"\"\"
        if hasattr(self, '_cached_pb2'):
            return self._cached_pb2
        # Direct import from /opt/geniesim to avoid /workspace path shadowing
        try:
            import importlib.util
            import os
            _pb2_path = "/opt/geniesim/source/data_collection/common/aimdk/protocol/contact_report_pb2.py"
            if os.path.exists(_pb2_path):
                _spec = importlib.util.spec_from_file_location("_bp_contact_pb2", _pb2_path)
                if _spec and _spec.loader:
                    _module = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_module)
                    self._cached_pb2 = _module
                    return _module
        except Exception as _e:
            print(f"[GRPC_GENERIC_HANDLER] Direct pb2 import failed: {_e}")
        # Fallback to standard imports
        try:
            from aimdk.protocol import contact_report_pb2 as geniesim_grpc_pb2
            self._cached_pb2 = geniesim_grpc_pb2
            return geniesim_grpc_pb2
        except Exception:
            pass
        try:
            import contact_report_pb2 as geniesim_grpc_pb2
            self._cached_pb2 = geniesim_grpc_pb2
            return geniesim_grpc_pb2
        except Exception:
            pass
        self._cached_pb2 = None
        return None

    def _deserialize_contact_req(self, data):
        pb2 = self._get_pb2()
        if pb2 is None:
            return None
        req = pb2.ContactReportReq()
        req.ParseFromString(data)
        return req

    def _serialize_contact_rsp(self, rsp):
        if rsp is None:
            pb2 = self._get_pb2()
            if pb2 is None:
                return b""
            return pb2.ContactReportRsp().SerializeToString()
        return rsp.SerializeToString()

    def _handle_contact_report(self, request, context):
        try:
            return self._obs_svc.get_contact_report(request, context)
        except Exception as e:
            print(f"[GRPC_GENERIC_HANDLER] get_contact_report failed: {e}")
            pb2 = self._get_pb2()
            if pb2 is None:
                return None
            return pb2.ContactReportRsp(
                contacts=[],
                total_normal_force=0.0,
                max_penetration_depth=0.0,
            )

    def _deserialize_ee_wrench_req(self, data):
        pb2 = self._get_pb2()
        if pb2 is None:
            return None
        req = pb2.EEWrenchReq()
        req.ParseFromString(data)
        return req

    def _serialize_ee_wrench_rsp(self, rsp):
        if rsp is None:
            pb2 = self._get_pb2()
            if pb2 is None:
                return b""
            return pb2.EEWrenchRsp().SerializeToString()
        return rsp.SerializeToString()

    def _handle_ee_wrench(self, request, context):
        try:
            return self._obs_svc.get_ee_wrench(request, context)
        except Exception as e:
            print(f"[GRPC_GENERIC_HANDLER] get_ee_wrench failed: {e}")
            pb2 = self._get_pb2()
            if pb2 is None:
                return None
            return pb2.EEWrenchRsp()
# --- END BlueprintPipeline grpc_generic_handler patch ---
""")

# Code to add after sim_observation_service registration
HANDLER_REGISTRATION = textwrap.dedent("""\
        # BlueprintPipeline grpc_generic_handler patch — register custom RPC handler
        _bp_obs_svc = ObservationService(self.server_function)
        sim_observation_service_pb2_grpc.add_SimObservationServiceServicer_to_server(
            _bp_obs_svc, self._server
        )
        self._server.add_generic_rpc_handlers([_BPGenericRpcHandler(_bp_obs_svc)])
""")


def patch_file():
    if not os.path.isfile(GRPC_SERVER):
        print(f"[PATCH] grpc_server.py not found at {GRPC_SERVER}")
        print("[PATCH] Skipping grpc_generic_handler patch")
        sys.exit(0)

    with open(GRPC_SERVER, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] grpc_generic_handler already patched — skipping")
        sys.exit(0)

    changes = 0

    # Step 1: Add the _BPGenericRpcHandler class definition before GrpcServer class
    grpc_server_class = re.search(r"^class GrpcServer:", content, re.MULTILINE)
    if not grpc_server_class:
        print("[PATCH] ERROR: Could not find 'class GrpcServer:' in grpc_server.py")
        sys.exit(1)

    insert_pos = grpc_server_class.start()
    content = content[:insert_pos] + GENERIC_HANDLER_CLASS + "\n\n" + content[insert_pos:]
    changes += 1
    print("[PATCH] Injected _BPGenericRpcHandler class definition")

    # Step 2: Modify the server() method to use our custom registration
    # Find the original sim_observation_service registration and replace it
    old_registration = (
        "sim_observation_service_pb2_grpc.add_SimObservationServiceServicer_to_server(\n"
        "            ObservationService(self.server_function), self._server\n"
        "        )"
    )

    new_registration = (
        "# " + PATCH_MARKER + " — custom registration with generic handler\n"
        "        _bp_obs_svc = ObservationService(self.server_function)\n"
        "        sim_observation_service_pb2_grpc.add_SimObservationServiceServicer_to_server(\n"
        "            _bp_obs_svc, self._server\n"
        "        )\n"
        "        self._server.add_generic_rpc_handlers([_BPGenericRpcHandler(_bp_obs_svc)])"
    )

    if old_registration in content:
        content = content.replace(old_registration, new_registration, 1)
        changes += 1
        print("[PATCH] Replaced ObservationService registration with custom handler")
    else:
        # Try flexible regex matching
        pattern = re.compile(
            r'sim_observation_service_pb2_grpc\.add_SimObservationServiceServicer_to_server\(\s*'
            r'ObservationService\(self\.server_function\),\s*self\._server\s*\)',
            re.DOTALL
        )
        m = pattern.search(content)
        if m:
            content = content[:m.start()] + new_registration.lstrip("# " + PATCH_MARKER + " — custom registration with generic handler\n") + content[m.end():]
            changes += 1
            print("[PATCH] Replaced ObservationService registration (regex)")
        else:
            print("[PATCH] WARNING: Could not find sim_observation_service registration pattern")
            # Try to insert after the last service registration as fallback
            add_port = re.search(r'self\._server\.add_insecure_port\(', content)
            if add_port:
                insert_before = add_port.start()
                # Check if we already have ObservationService registered
                if "ObservationService" in content[:insert_before]:
                    # Just add the generic handler after existing registrations
                    inject = (
                        "\n        # " + PATCH_MARKER + " — add generic handler for custom methods\n"
                        "        # Note: ObservationService already registered; adding generic handler separately\n"
                        "        _bp_obs_svc_for_handler = ObservationService(self.server_function)\n"
                        "        self._server.add_generic_rpc_handlers([_BPGenericRpcHandler(_bp_obs_svc_for_handler)])\n"
                    )
                    content = content[:insert_before] + inject + content[insert_before:]
                    changes += 1
                    print("[PATCH] Injected generic handler registration (fallback)")

    if changes == 0:
        print("[PATCH] ERROR: No changes made — patch failed")
        sys.exit(1)

    with open(GRPC_SERVER, "w") as f:
        f.write(content)

    print(f"[PATCH] Successfully patched grpc_server.py for generic RPC handling ({changes} changes)")


if __name__ == "__main__":
    patch_file()
