#!/usr/bin/env python3
"""Validate transitive local USD dependencies for RidgebackFranka assets.

Uses structured USD layer/reference traversal when available, with a guarded
text fallback. The fallback is defensive against malformed binary scrape tokens
to avoid path-length and decode crashes.
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
MAX_REF_LEN = 4096
REF_PATTERN = re.compile(r"@([^@]+)@")
SKIP_PREFIXES = ("http://", "https://", "data:", "anon:", "materialx:")


def _read_text_for_reference_scan(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        pass

    try:
        raw = path.read_bytes()
    except Exception:
        return ""
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
    return raw.decode("utf-8", errors="ignore")


def _is_plausible_ref(ref: str) -> bool:
    token = str(ref or "").strip()
    if not token:
        return False
    if len(token) > MAX_REF_LEN:
        return False
    if "\x00" in token:
        return False
    return True


def _structured_refs_from_usd(path: Path) -> Optional[List[str]]:
    try:
        from pxr import Sdf  # type: ignore
    except Exception:
        return None

    try:
        layer = Sdf.Layer.FindOrOpen(str(path))
    except Exception:
        return None
    if layer is None:
        return None

    refs: Set[str] = set()
    try:
        for sub in list(getattr(layer, "subLayerPaths", []) or []):
            if isinstance(sub, str) and sub.strip():
                refs.add(sub.strip())
    except Exception:
        pass

    def _walk_prim_spec(prim_spec: object) -> None:
        for list_attr in ("referenceList", "payloadList"):
            try:
                list_op = getattr(prim_spec, list_attr, None)
            except Exception:
                list_op = None
            if list_op is None:
                continue
            for item_attr in ("prependedItems", "addedItems", "appendedItems", "explicitItems"):
                try:
                    items = list(getattr(list_op, item_attr, []) or [])
                except Exception:
                    items = []
                for item in items:
                    try:
                        asset = str(getattr(item, "assetPath", "") or "").strip()
                    except Exception:
                        asset = ""
                    if asset:
                        refs.add(asset)
        children = {}
        try:
            children = dict(getattr(prim_spec, "nameChildren", {}) or {})
        except Exception:
            children = {}
        for child in children.values():
            _walk_prim_spec(child)

    try:
        for prim_spec in list(getattr(layer, "rootPrims", []) or []):
            _walk_prim_spec(prim_spec)
    except Exception:
        return None

    return sorted(refs)


def _extract_refs(text: str) -> List[str]:
    refs: List[str] = []
    for match in REF_PATTERN.finditer(text):
        ref = match.group(1).strip()
        if _is_plausible_ref(ref):
            refs.append(ref)
    return refs


def _resolve_ref_path(ref: str, *, owner_file: Path, assets_root: Path) -> Optional[Path]:
    if not ref or ref.startswith(SKIP_PREFIXES):
        return None
    ref = ref.split("#", 1)[0].strip()
    if not _is_plausible_ref(ref):
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

    try:
        p = Path(ref)
        if p.is_absolute():
            return p
        return (owner_file.parent / p).resolve()
    except Exception:
        return None


def _refs_for_usd(path: Path) -> List[str]:
    structured = _structured_refs_from_usd(path)
    if structured is not None:
        return [r for r in structured if _is_plausible_ref(r)]
    return _extract_refs(_read_text_for_reference_scan(path))


def _scan_transitive_dependencies(
    root_usd: Path,
    *,
    assets_root: Path,
) -> Tuple[Set[Path], List[Dict[str, str]], int, int]:
    queue: List[Path] = [root_usd.resolve()]
    visited_usd: Set[Path] = set()
    missing: List[Dict[str, str]] = []
    checked_refs = 0
    skipped_refs = 0

    while queue:
        usd_path = queue.pop()
        if usd_path in visited_usd:
            continue
        visited_usd.add(usd_path)

        if not usd_path.exists():
            missing.append({"owner": str(usd_path), "ref": "(root)", "resolved": str(usd_path)})
            continue

        refs = _refs_for_usd(usd_path)
        for ref in refs:
            resolved = _resolve_ref_path(ref, owner_file=usd_path, assets_root=assets_root)
            if resolved is None:
                skipped_refs += 1
                continue
            checked_refs += 1
            try:
                exists = resolved.exists()
            except OSError as exc:
                missing.append(
                    {
                        "owner": str(usd_path),
                        "ref": ref,
                        "resolved": str(resolved),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if not exists:
                missing.append({"owner": str(usd_path), "ref": ref, "resolved": str(resolved)})
                continue
            if resolved.suffix.lower() in USD_EXTS:
                queue.append(resolved)

    return visited_usd, missing, checked_refs, skipped_refs


def validate_ridgeback_assets(root_usd: Path, *, assets_root: Path) -> Dict[str, object]:
    visited_usd, missing, checked_refs, skipped_refs = _scan_transitive_dependencies(root_usd, assets_root=assets_root)
    return {
        "root_usd": str(root_usd),
        "assets_root": str(assets_root),
        "visited_usd_files": sorted(str(p) for p in visited_usd),
        "visited_usd_count": len(visited_usd),
        "checked_references_count": checked_refs,
        "skipped_references_count": skipped_refs,
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
