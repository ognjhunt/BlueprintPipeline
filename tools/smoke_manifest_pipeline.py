"""Minimal smoke-style examples for moving a scene manifest through the pipeline.

These helpers intentionally avoid heavy external dependencies and only run when
required inputs are present. They can be used as executable documentation for
future adapters (e.g., BlueprintRecipe) that want to drive the USD assembly,
SimReady, and Replicator bundle steps programmatically.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from blueprint_sim.manifest import DEFAULT_ROOT


def _ensure_sample_inputs(tmp_root: Path) -> None:
    assets_root = tmp_root / "scenes/example/assets"
    seg_root = tmp_root / "scenes/example/seg"
    usd_root = tmp_root / "scenes/example/usd"

    assets_root.mkdir(parents=True, exist_ok=True)
    seg_root.mkdir(parents=True, exist_ok=True)
    usd_root.mkdir(parents=True, exist_ok=True)

    # Minimal manifest and layout placeholders
    manifest = {
        "scene_id": "example",
        "objects": [],
        "schema_version": "0.0.1",
    }
    layout = {"objects": [], "camera_trajectory": []}
    inventory = {"scene_type": "generic", "objects": []}

    (assets_root / "scene_manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2), encoding="utf-8"
    )
    (tmp_root / "scenes/example/layout/scene_layout_scaled.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (tmp_root / "scenes/example/layout/scene_layout_scaled.json").write_text(
        __import__("json").dumps(layout, indent=2), encoding="utf-8"
    )
    (seg_root / "inventory.json").write_text(
        __import__("json").dumps(inventory, indent=2), encoding="utf-8"
    )


def smoke_manifest_to_usd(root: Path = DEFAULT_ROOT) -> Optional[int]:
    try:
        from blueprint_sim import assembly
    except ImportError as exc:  # pragma: no cover - optional dependency
        print(f"[SMOKE] Skipping USD assembly example: {exc}")
        return None

    layout_prefix = "scenes/example/layout"
    assets_prefix = "scenes/example/assets"
    return assembly.assemble_scene(
        layout_prefix=layout_prefix,
        assets_prefix=assets_prefix,
        usd_prefix="scenes/example/usd",
        bucket="local",
        scene_id="example",
        convert_only=True,
        root=root,
    )


def smoke_simready(root: Path = DEFAULT_ROOT) -> Optional[int]:
    try:
        from blueprint_sim.simready import prepare_simready_assets
    except ImportError as exc:  # pragma: no cover
        print(f"[SMOKE] Skipping SimReady example: {exc}")
        return None

    return prepare_simready_assets(
        assets_prefix="scenes/example/assets", bucket="local", scene_id="example", root=root
    )


def smoke_replicator_bundle(root: Path = DEFAULT_ROOT) -> Optional[int]:
    try:
        from blueprint_sim.replicator import generate_replicator_bundle
    except ImportError as exc:  # pragma: no cover
        print(f"[SMOKE] Skipping Replicator example: {exc}")
        return None

    return generate_replicator_bundle(
        scene_id="example",
        seg_prefix="scenes/example/seg",
        assets_prefix="scenes/example/assets",
        usd_prefix="scenes/example/usd",
        replicator_prefix="scenes/example/replicator",
        bucket="local",
        requested_policies=None,
        root=root,
    )


def main() -> None:
    tmp_root = Path(os.environ.get("SMOKE_ROOT", "/tmp/blueprint_smoke"))
    _ensure_sample_inputs(tmp_root)

    print("[SMOKE] Prepared sample manifest at", tmp_root)
    smoke_manifest_to_usd(tmp_root)
    smoke_simready(tmp_root)
    smoke_replicator_bundle(tmp_root)


if __name__ == "__main__":
    main()
