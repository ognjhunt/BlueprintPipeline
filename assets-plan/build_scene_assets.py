import os, json
from pathlib import Path

INTERACTIVE_KEYWORDS = {
    "door", "drawer", "cabinet", "fridge", "refrigerator", "oven", "freezer", "handle", "knob",
    "faucet", "tap", "switch", "lever", "hinge"
}

def parse_interactive_ids(raw: str | None):
    if not raw:
        return set()
    ids = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            ids.add(int(tok))
        except ValueError:
            pass
    return ids

def classify_type(class_name: str, phrase: str | None, interactive_ids, obj_id: int, static_pipeline: str):
    text = (class_name or "").lower()
    if phrase:
        text += " " + phrase.lower()
    if obj_id in interactive_ids:
        return "interactive", "physx"
    if any(k in text for k in INTERACTIVE_KEYWORDS):
        return "interactive", "physx"
    return "static", static_pipeline

def main():
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    layout_prefix = os.getenv("LAYOUT_PREFIX")       # scenes/<sceneId>/layout
    multiview_prefix = os.getenv("MULTIVIEW_PREFIX") # scenes/<sceneId>/multiview
    assets_prefix = os.getenv("ASSETS_PREFIX")       # scenes/<sceneId>/assets
    layout_file = os.getenv("LAYOUT_FILE_NAME", "scene_layout_scaled.json")
    interactive_ids_env = os.getenv("INTERACTIVE_OBJECT_IDS", "")
    static_pipeline = os.getenv("STATIC_ASSET_PIPELINE", "sam3d")
    physx_endpoint = os.getenv("PHYSX_ENDPOINT")

    if not (layout_prefix and multiview_prefix and assets_prefix):
        raise SystemExit("[ASSETS] LAYOUT_PREFIX, MULTIVIEW_PREFIX, ASSETS_PREFIX required")

    interactive_ids = parse_interactive_ids(interactive_ids_env)

    root = Path("/mnt/gcs")
    layout_path = root / layout_prefix / layout_file
    multiview_root = root / multiview_prefix
    assets_root = root / assets_prefix

    print(f"[ASSETS] Bucket={bucket}")
    print(f"[ASSETS] Scene={scene_id}")
    print(f"[ASSETS] Layout={layout_path}")
    print(f"[ASSETS] Multiview root={multiview_root}")
    print(f"[ASSETS] Assets root={assets_root}")
    print(f"[ASSETS] interactive_ids={interactive_ids}")
    print(f"[ASSETS] static_pipeline={static_pipeline}")
    if physx_endpoint:
        print(f"[ASSETS] physx_endpoint={physx_endpoint}")

    with layout_path.open("r") as f:
        layout = json.load(f)

    objects = layout.get("objects", [])
    plan_objects = []

    for obj in objects:
        oid = obj.get("id")
        cls = obj.get("class_name", f"class_{obj.get('class_id', 0)}")
        phrase = obj.get("object_phrase")  # optional future field

        mv_dir = multiview_root / f"obj_{oid}"
        crop_path = mv_dir / "crop.png"
        if not crop_path.is_file():
            print(f"[ASSETS] WARNING: missing crop for obj {oid} at {crop_path}")
            continue

        obj_type, pipeline = classify_type(cls, phrase, interactive_ids, oid, static_pipeline=static_pipeline)

        entry = {
            "id": oid,
            "class_name": cls,
            "type": obj_type,
            "pipeline": pipeline,
            "multiview_dir": f"{multiview_prefix}/obj_{oid}",
            "crop_path": f"{multiview_prefix}/obj_{oid}/crop.png",
            "polygon": obj.get("polygon"),
        }
        if obj_type == "interactive":
            entry["interactive_output"] = f"{assets_prefix}/interactive/obj_{oid}"
            if physx_endpoint:
                entry["physx_endpoint"] = physx_endpoint
        plan_objects.append(entry)

    assets_root.mkdir(parents=True, exist_ok=True)
    out_path = assets_root / "scene_assets.json"
    plan = {
        "scene_id": scene_id,
        "objects": plan_objects,
        "physx_endpoint": physx_endpoint,
        "interactive_count": sum(1 for o in plan_objects if o["type"] == "interactive"),
    }
    with out_path.open("w") as f:
        json.dump(plan, f, indent=2)

    print(f"[ASSETS] Wrote {out_path} with {len(plan_objects)} objects")

if __name__ == "__main__":
    main()
