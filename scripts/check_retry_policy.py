#!/usr/bin/env python3
"""Validate workflow retry settings against shared defaults."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / "workflows"
RETRY_POLICY_PATH = REPO_ROOT / "policy_configs" / "retry_policy.yaml"


def load_default_max_retries() -> int:
    payload = yaml.safe_load(RETRY_POLICY_PATH.read_text()) or {}
    if not isinstance(payload, dict) or "max_retries" not in payload:
        raise ValueError(f"Missing max_retries in {RETRY_POLICY_PATH}")
    return int(payload["max_retries"])


def main() -> int:
    default_max_retries = load_default_max_retries()
    errors: list[str] = []
    for workflow_path in sorted(WORKFLOWS_DIR.glob("*-pipeline.yaml")):
        for line_number, line in enumerate(workflow_path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            match = re.search(r"max_retries:\s*(\d+)(.*)", line)
            if not match:
                continue
            value = int(match.group(1))
            comment = match.group(2) or ""
            if value == default_max_retries:
                if "policy_configs/retry_policy.yaml" not in comment:
                    errors.append(
                        f"{workflow_path}:{line_number} max_retries uses default {default_max_retries} "
                        "but is missing a reference to policy_configs/retry_policy.yaml"
                    )
            else:
                if "override" not in comment.lower():
                    errors.append(
                        f"{workflow_path}:{line_number} max_retries={value} diverges from "
                        f"default {default_max_retries} without an inline override comment"
                    )
    if errors:
        print("Retry policy compliance check failed:\n" + "\n".join(errors))
        return 1
    print("Retry policy compliance check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
