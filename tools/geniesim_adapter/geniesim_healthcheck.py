#!/usr/bin/env python3
"""
Genie Sim Health Check CLI.

Validates Isaac Sim availability, gRPC dependencies, and server readiness.
"""

import argparse
import json
import sys
from typing import Any, Dict

from tools.geniesim_adapter.local_framework import (
    build_geniesim_preflight_report,
    check_geniesim_availability,
)


def _render_human_report(report: Dict[str, Any]) -> None:
    status = report.get("status", {})
    print("Genie Sim Health Check")
    print("=======================")
    print(f"Isaac Sim available: {status.get('isaac_sim_available', False)}")
    print(f"gRPC available: {status.get('grpc_available', False)}")
    print(f"gRPC stubs available: {status.get('grpc_stubs_available', False)}")
    print(f"Server running: {status.get('server_running', False)}")
    print(f"Server ready: {report.get('server_ready', False)}")
    print(f"Overall: {'PASS' if report.get('ok') else 'FAIL'}")
    if report.get("missing"):
        print("\nMissing requirements:")
        for item in report["missing"]:
            print(f"  - {item}")
    if report.get("remediation"):
        print("\nRemediation steps:")
        for step in report["remediation"]:
            print(f"  - {step}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Genie Sim health check")
    parser.add_argument("--timeout", type=float, default=5.0, help="Ping timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    status = check_geniesim_availability()
    report = build_geniesim_preflight_report(
        status,
        require_server=True,
        require_ready=True,
        ping_timeout=args.timeout,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _render_human_report(report)

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
