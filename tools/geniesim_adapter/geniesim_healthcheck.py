#!/usr/bin/env python3
"""
Genie Sim Health Check CLI.

Validates Isaac Sim availability, gRPC dependencies, and server readiness.
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict

from tools.logging_config import init_logging

logger = logging.getLogger(__name__)


def _render_human_report(report: Dict[str, Any]) -> None:
    status = report.get("status", {})
    print("Genie Sim Health Check")
    print("=======================")
    print("Isaac Sim available: %s" % status.get("isaac_sim_available", False))
    print("gRPC available: %s" % status.get("grpc_available", False))
    print("gRPC stubs available: %s" % status.get("grpc_stubs_available", False))
    print("Server running: %s" % status.get("server_running", False))
    print("Server ready: %s" % report.get("server_ready", False))
    print(
        "Overall: %s" % ("PASS" if report.get("ok") else "FAIL"),
    )
    if report.get("missing"):
        print("Missing requirements: %s" % ", ".join(report["missing"]))
    if report.get("remediation"):
        print("Remediation steps: %s" % " ".join(report["remediation"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Genie Sim health check")
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Readiness probe timeout in seconds",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    init_logging()
    if args.json:
        logging.disable(logging.CRITICAL)

    from tools.geniesim_adapter.local_framework import (
        build_geniesim_preflight_report,
        check_geniesim_availability,
    )

    status = check_geniesim_availability()
    report = build_geniesim_preflight_report(
        status,
        require_server=True,
        require_ready=True,
        readiness_timeout=args.timeout,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _render_human_report(report)

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
