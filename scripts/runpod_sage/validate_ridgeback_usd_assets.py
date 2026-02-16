#!/usr/bin/env python3
"""
Validate transitive local USD dependencies for RidgebackFranka assets.

This avoids "single file exists" false positives by recursively scanning USD
asset references and verifying all referenced local files are present.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


USD_EXTS = {".usd", ".usda", ".usdc", ".usdz"}
REF_PATTERN = re.compile(r"@([^@]+)@")
SKIP_PREFIXES = ("http://", "https://", "data:", "anon:", "materialx:")


def _read_text_for_reference_scan(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        pass

    raw = path.read_bytes()
    if raw.startswith(b"PXR-USDC"):
        if shutil.which("usdcat"):
            try:
                proc = subprocess.run(
                    ["usdcat", str(path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                return proc.stdout
            except Exception:
                pass
        if shutil.which("strings"):
            try:
                proc = subprocess.run(
                    ["strings", str(path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                return proc.stdout
            except Exception:
                pass
    return raw.decode("utf-8", errors="ignore")


def _extract_refs(text: str) -> List[str]:
    refs: List[str] = []
    for match in REF_PATTERN.finditer(text):
        ref = match.group(1).strip()
        if ref:
            refs.append(ref)
    return refs


def _resolve_ref_path(ref: str, *, owner_file: Path, assets_root: Path) -> Optional[Path]:
    if not ref or ref.startswith(SKIP_PREFIXES):
        return None
    ref = ref.split("#", 1)[0].strip()
    if not ref:
        return None

    # Omniverse-style absolute references mapped into local assets root.
    if ref.startswith("omniverse://"):
        tail = ref.split("://", 1)[1]
        slash = tail.find("/")
        if slash >= 0:
            ref = tail[slash:]
        else:
            return None

    if ref.startswith("/Isaac/"):
        return assets_root / ref.lstrip("/")

    if ref.startswith("/NVIDIA/"):
        # Keep "/NVIDIA" marker support for environments mirroring Nucleus roots.
        return assets_root.parent.parent.parent / ref.lstrip("/")

    p = Path(ref)
    if p.is_absolute():
        return p
    return (owner_file.parent / p).resolve()


def _scan_transitive_dependencies(root_usd: Path, *, assets_root: Path) -> Tuple[Set[Path], List[Dict[str, str]], int]:
    queue: List[Path] = [root_usd.resolve()]
    visited_usd: Set[Path] = set()
    missing: List[Dict[str, str]] = []
    checked_refs = 0

    while queue:
        usd_path = queue.pop()
        if usd_path in visited_usd:
            continue
        visited_usd.add(usd_path)

        if not usd_path.exists():
            missing.append({"owner": str(usd_path), "ref": "(root)", "resolved": str(usd_path)})
            continue

        text = _read_text_for_reference_scan(usd_path)
        refs = _extract_refs(text)
        for ref in refs:
            resolved = _resolve_ref_path(ref, owner_file=usd_path, assets_root=assets_root)
            if resolved is None:
                continue
            checked_refs += 1
            if not resolved.exists():
                missing.append({"owner": str(usd_path), "ref": ref, "resolved": str(resolved)})
                continue
            if resolved.suffix.lower() in USD_EXTS:
                queue.append(resolved)

    return visited_usd, missing, checked_refs


def validate_ridgeback_assets(root_usd: Path, *, assets_root: Path) -> Dict[str, object]:
    visited_usd, missing, checked_refs = _scan_transitive_dependencies(root_usd, assets_root=assets_root)
    return {
        "root_usd": str(root_usd),
        "assets_root": str(assets_root),
        "visited_usd_files": sorted(str(p) for p in visited_usd),
        "visited_usd_count": len(visited_usd),
        "checked_references_count": checked_refs,
        "missing_references": missing,
        "missing_count": len(missing),
        "ok": len(missing) == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate transitive RidgebackFranka local USD dependencies")
    parser.add_argument("--root-usd", required=True, help="Path to ridgeback_franka.usd")
    parser.add_argument("--assets-root", required=True, help="Local ISAAC_ASSETS_ROOT")
    parser.add_argument("--report-path", default="", help="Optional JSON report output path")
    parser.add_argument("--warn-only", action="store_true", default=False)
    args = parser.parse_args()

    root_usd = Path(args.root_usd).resolve()
    assets_root = Path(args.assets_root).resolve()

    report = validate_ridgeback_assets(root_usd, assets_root=assets_root)

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": report["ok"],
                "visited_usd_count": report["visited_usd_count"],
                "checked_references_count": report["checked_references_count"],
                "missing_count": report["missing_count"],
            }
        )
    )
    if report["ok"]:
        return 0
    if args.warn_only:
        return 0
    for item in report["missing_references"][:20]:
        print(
            f"missing ref owner={item['owner']} ref={item['ref']} resolved={item['resolved']}",
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
