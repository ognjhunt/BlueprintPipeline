#!/usr/bin/env python3
"""
Python 3 compatibility shim for the old stdlib `sets` module.

The original Python 2 code in kcoley/gltf2usd does:

    from sets import Set

In Python 3, the `sets` module was removed and `set` / `frozenset`
are built‑in types. This shim provides a minimal `Set` alias so that
the legacy code continues to work.
"""

from __future__ import annotations

# In Python 2.4+ the old `Set` type and the built‑in `set` are equivalent.
# We simply alias Set to the built‑in `set` for compatibility.
Set = set
