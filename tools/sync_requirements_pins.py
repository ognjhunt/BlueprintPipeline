#!/usr/bin/env python3
"""Sync per-job requirements.txt files with tools/requirements-pins.txt."""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PINS_FILE = ROOT / "tools" / "requirements-pins.txt"

ALIASES = {
    "pytorch_lightning": "pytorch-lightning",
    "huggingface_hub": "huggingface-hub",
    "opencv_python": "opencv-python",
    "opencv_python_headless": "opencv-python-headless",
    "flash_attn": "flash-attn",
    "Pillow": "pillow",
    "PyYAML": "pyyaml",
}


def load_pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in PINS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        pins[name.lower()] = version
    return pins


def sync_requirements(pins: dict[str, str]) -> None:
    for path in ROOT.rglob("requirements.txt"):
        lines = path.read_text().splitlines()
        new_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                new_lines.append(line)
                continue
            if stripped.startswith("-"):
                new_lines.append(line)
                continue
            parts = stripped.split(";", 1)
            req_part = parts[0].strip()
            marker = ";" + parts[1] if len(parts) > 1 else ""
            match = re.match(r"([A-Za-z0-9_.-]+)", req_part)
            if not match:
                new_lines.append(line)
                continue
            name = match.group(1)
            canonical = ALIASES.get(name, name)
            version = pins.get(canonical.lower())
            if version is None:
                raise SystemExit(f"Missing pin for {canonical} in {path}")
            new_lines.append(f"{canonical}=={version}{marker}")
        path.write_text("\n".join(new_lines) + "\n")


if __name__ == "__main__":
    sync_requirements(load_pins())
