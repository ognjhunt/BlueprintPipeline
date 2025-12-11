#!/usr/bin/env python3
"""Index NVIDIA pack metadata and ZeroScene assets into Firestore.

This tool builds canonical ``assets`` and ``scenes`` collection entries without
requiring meshes to be present locally. It computes optional text embeddings
for retrieval if GEMINI_API_KEY is provided.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import importlib

from tools.asset_catalog import AssetCatalogClient, AssetDocument, SceneDocument, EmbeddingInfo
from tools.scene_manifest.loader import load_manifest_or_scene_assets

GCS_ROOT = Path("/mnt/gcs")


def load_pack_assets(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "assets" in data:
        return list(data.get("assets") or [])
    if isinstance(data, list):
        return list(data)
    raise ValueError("Pack metadata must be a list or dict with an 'assets' key")


def build_embedding(text: str) -> Optional[EmbeddingInfo]:
    if not text:
        return None

    genai_spec = importlib.util.find_spec("google.genai")
    if genai_spec is None:
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    genai = importlib.import_module("google.genai")
    client = genai.Client(api_key=api_key)

    try:
        response = client.models.embed_content(model="text-embedding-004", contents=text)
        vector = getattr(response, "values", None)
        if vector is None and hasattr(response, "embedding"):
            vector = getattr(response.embedding, "values", response.embedding)
        if vector:
            return EmbeddingInfo(model="text-embedding-004", text=text, vector=list(vector))
    except Exception as exc:  # pragma: no cover - network/API failures
        print(f"[INDEXER] WARNING: embedding failed: {exc}", file=sys.stderr)

    return None


def safe_slug(value: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


def build_embedding_text(entry: Dict[str, Any], tags: Iterable[str]) -> str:
    parts = [
        f"name: {entry.get('name', '')}",
        f"description: {entry.get('description', '')}",
        f"class: {entry.get('class_name') or entry.get('category', '')}",
    ]
    role_list = entry.get("sim_roles") or entry.get("role") or []
    if role_list:
        parts.append(f"roles: {','.join(role_list if isinstance(role_list, list) else [role_list])}")
    tag_list = list(tags)
    if tag_list:
        parts.append(f"tags: {', '.join(tag_list)}")
    return " | ".join([p for p in parts if p.strip()])


def build_asset_document(entry: Dict[str, Any], source: str) -> AssetDocument:
    asset_id = str(entry.get("asset_id") or entry.get("id") or safe_slug(entry.get("name", "asset")))
    logical_id = entry.get("logical_id") or entry.get("logical_asset_id")
    sim_roles = entry.get("sim_roles") or entry.get("role") or []
    if isinstance(sim_roles, str):
        sim_roles = [sim_roles]

    tags: List[str] = []
    if entry.get("category"):
        tags.append(str(entry.get("category")))
    if entry.get("class_name"):
        tags.append(str(entry.get("class_name")))

    embedding_text = build_embedding_text(entry, tags)
    embedding = build_embedding(embedding_text)

    ignored_keys = {
        "asset_id",
        "logical_id",
        "logical_asset_id",
        "usd_path",
        "asset_path",
        "gcs_uri",
        "thumbnail",
        "thumbnail_uri",
        "sim_roles",
        "role",
        "class_name",
        "category",
        "description",
        "physics",
        "physics_profile",
        "articulation",
        "articulation_profile",
    }

    return AssetDocument(
        asset_id=asset_id,
        logical_id=logical_id,
        source=source,
        usd_path=entry.get("usd_path") or entry.get("asset_path"),
        gcs_uri=entry.get("gcs_uri"),
        thumbnail_uri=entry.get("thumbnail") or entry.get("thumbnail_uri"),
        sim_roles=list(sim_roles),
        class_name=entry.get("class_name") or entry.get("category"),
        description=entry.get("description"),
        physics_profile=entry.get("physics") or entry.get("physics_profile"),
        articulation_profile=entry.get("articulation") or entry.get("articulation_profile"),
        embedding=embedding,
        extra_metadata={
            "tags": tags,
            **{k: v for k, v in entry.items() if k not in ignored_keys},
        },
    )


def index_pack(metadata_path: Path, client: AssetCatalogClient) -> List[str]:
    assets = load_pack_assets(metadata_path)
    registered: List[str] = []

    for entry in assets:
        doc = build_asset_document(entry, source="nvidia_pack")
        client.upsert_asset_document(doc)
        registered.append(doc.asset_id)
        print(f"[INDEXER] Registered pack asset {doc.asset_id} ({doc.class_name})")

    return registered


def map_type_to_role(obj: Dict[str, Any]) -> List[str]:
    role = obj.get("sim_role") or obj.get("type")
    if role in {"interactive", "manipulable_object"}:
        return ["manipulable_object"]
    if role in {"articulated", "articulated_furniture", "articulated_appliance"}:
        return ["articulated"]
    return [role] if role else []


def index_zeroscene_assets(
    assets_prefix: str,
    scene_id: Optional[str],
    client: AssetCatalogClient,
    root: Path = GCS_ROOT,
) -> List[str]:
    assets_root = root / assets_prefix
    manifest = load_manifest_or_scene_assets(assets_root)
    if manifest is None:
        print(f"[INDEXER] No manifest found under {assets_root}", file=sys.stderr)
        return []

    registered: List[str] = []
    objects = manifest.get("objects", [])
    for obj in objects:
        logical_id = obj.get("logical_asset_id") or obj.get("id")
        asset_id = f"{scene_id}_obj_{logical_id}" if scene_id else str(logical_id)
        entry = {
            "asset_id": asset_id,
            "logical_id": logical_id,
            "name": obj.get("class_name") or f"obj_{logical_id}",
            "class_name": obj.get("class_name"),
            "description": obj.get("description"),
            "usd_path": obj.get("asset_path"),
            "thumbnail_uri": obj.get("thumbnail") or obj.get("thumbnail_uri"),
            "sim_roles": map_type_to_role(obj),
            "physics": obj.get("physics"),
            "articulation": obj.get("articulation"),
        }
        doc = build_asset_document(entry, source="zeroscene")
        client.upsert_asset_document(doc)
        registered.append(doc.asset_id)
        print(f"[INDEXER] Registered ZeroScene asset {doc.asset_id} ({doc.class_name})")

    if scene_id:
        scene_doc = SceneDocument(
            scene_id=scene_id,
            assets_prefix=assets_prefix,
            asset_ids=registered,
            usd_path=f"{assets_prefix}/usd/scene.usda",
        )
        client.upsert_scene_document(scene_doc)

    return registered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index pack metadata and ZeroScene assets")
    parser.add_argument("--pack-metadata", type=Path, help="Path to NVIDIA pack metadata JSON", required=False)
    parser.add_argument("--assets-prefix", type=str, help="ZeroScene assets prefix (e.g., scenes/123/assets)")
    parser.add_argument("--scene-id", type=str, default=None, help="Scene identifier for ZeroScene registration")
    parser.add_argument("--root", type=Path, default=GCS_ROOT, help="Root path for storage")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = AssetCatalogClient()

    registered: List[str] = []
    if args.pack_metadata:
        registered.extend(index_pack(args.pack_metadata, client))

    if args.assets_prefix:
        registered.extend(index_zeroscene_assets(args.assets_prefix, args.scene_id, client, args.root))

    if not registered:
        print("[INDEXER] No assets registered")
        return 1

    print(f"[INDEXER] Indexed {len(registered)} assets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
