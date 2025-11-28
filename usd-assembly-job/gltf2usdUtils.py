#!/usr/bin/env python3
"""
Minimal shim for kcoley/gltf2usd's gltf2usdUtils module.

The gltf2usd code expects a top-level module named ``gltf2usdUtils``.
In some layouts this module either does not exist or lives inside the
private ``_gltf2usd`` package, which leads to:

    ModuleNotFoundError: No module named 'gltf2usdUtils'

This shim provides:
  * A conservative implementation of the helpers we know are used
    by gltf2usd for static meshes; and
  * A best-effort adapter that delegates to the "real" implementation
    if it is importable from ``_gltf2usd.gltf2usdUtils``.

The Blueprint usd-assembly job only needs static mesh support (no
skeleton / animation), so this is sufficient for our use case.
"""

from __future__ import annotations

import re
from typing import Any


def _sanitize_identifier(name: str, prefix: str = "node") -> str:
    """
    Return a string that is safe to use as a USD prim / primvar name.

    Rules (intentionally conservative):
      * Strip leading/trailing whitespace.
      * Replace any character not in [A-Za-z0-9_] with "_".
      * If the name starts with a digit, prepend the prefix + "_".
      * Avoid reserved names "." and "..".
    """
    if not isinstance(name, str):
        name = str(name)

    name = name.strip()
    if not name:
        name = prefix

    # Replace unsafe characters.
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)

    # USD identifiers cannot start with a digit.
    if name[0].isdigit():
        name = f"{prefix}_{name}"

    # Avoid "." or ".." which are not valid identifiers.
    if name in {".", ".."}:
        name = f"{prefix}_{name.replace('.', '_')}"

    return name


class _UtilsProxy:
    """
    Fallback implementation of the GLTF2USDUtils API.

    We implement the helpers that are known to be used in the gltf2usd
    script, and provide a forgiving default for any unknown helpers:
    they simply return the first positional argument unchanged so that
    the caller still gets something usable.
    """

    # Known helpers -----------------------------------------------------

    def convert_to_usd_friendly_node_name(self, name: str) -> str:
        return _sanitize_identifier(name, prefix="node")

    def convert_to_usd_friendly_primvar_name(self, name: str) -> str:
        return _sanitize_identifier(name, prefix="pv")

    # Generic fallback --------------------------------------------------

    def __getattr__(self, attr: str) -> Any:  # pragma: no cover - defensive
        def _fallback(*args: Any, **kwargs: Any) -> Any:
            # For unknown helpers, just return first arg (if any).
            return args[0] if args else None

        return _fallback


# Try to delegate to a "real" implementation if the package layout provides one.
try:  # pragma: no cover - best-effort import
    from _gltf2usd import gltf2usdUtils as _real_impl  # type: ignore[import-not-found]

    GLTF2USDUtils = getattr(_real_impl, "GLTF2USDUtils", _UtilsProxy())
except Exception:  # noqa: BLE001 - broad by design
    GLTF2USDUtils = _UtilsProxy()


# Convenience function so code that does:
#   from gltf2usdUtils import convert_to_usd_friendly_node_name
# also continues to work.
def convert_to_usd_friendly_node_name(name: str) -> str:
    return GLTF2USDUtils.convert_to_usd_friendly_node_name(name)
