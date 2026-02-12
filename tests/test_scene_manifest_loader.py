from __future__ import annotations

import json
from pathlib import Path

from tools.scene_manifest.loader import load_manifest_or_scene_assets


def _write_manifest(assets_root: Path, manifest: dict) -> None:
    assets_root.mkdir(parents=True, exist_ok=True)
    (assets_root / "scene_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_loader_preserves_articulation_candidate_and_required(tmp_path: Path) -> None:
    assets_root = tmp_path / "assets"
    manifest = {
        "version": "1.0.0",
        "scene_id": "scene_loader_flags",
        "scene": {"coordinate_frame": "y_up", "meters_per_unit": 1.0},
        "objects": [
            {
                "id": "cabinet_001",
                "name": "base cabinet",
                "category": "cabinet",
                "sim_role": "articulated_furniture",
                "asset": {"path": "scenes/scene_loader_flags/assets/cabinet_001/model.usd"},
                "transform": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "articulation": {
                    "candidate": True,
                    "required": False,
                    "backend_hint": "particulate_optional",
                },
            }
        ],
    }
    _write_manifest(assets_root, manifest)

    loaded = load_manifest_or_scene_assets(assets_root)
    assert loaded is not None
    assert loaded["scene_id"] == "scene_loader_flags"
    assert len(loaded["objects"]) == 1
    obj = loaded["objects"][0]
    assert obj["type"] == "interactive"
    assert obj["sim_role"] == "articulated_furniture"
    assert obj["articulation_candidate"] is True
    assert obj["articulation_required"] is False
    assert obj["articulation"]["candidate"] is True
