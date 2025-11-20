import os, sys, json, base64, time
from pathlib import Path

import requests

MESHY_BASE = "https://api.meshy.ai"

def load_assets_plan(path: Path):
    with path.open("r") as f:
        return json.load(f)

def image_to_data_uri(img_path: Path):
    mime = "image/png"
    if img_path.suffix.lower() in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"

def create_image_to_3d_task(api_key: str, data_uri: str) -> str:
    url = f"{MESHY_BASE}/openapi/v1/image-to-3d"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "image_url": data_uri,
        "ai_model": "latest",          # Meshy 6 preview
        "topology": "triangle",
        "should_remesh": False,        # highest precision mesh :contentReference[oaicite:8]{index=8}
        "should_texture": True,
        "enable_pbr": True,
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    task_id = resp.json()["result"]
    print(f"[MESHY] Created task {task_id}")
    return task_id

def wait_for_task(api_key: str, task_id: str, poll_seconds: float = 10.0, timeout: float = 1800.0):
    url = f"{MESHY_BASE}/openapi/v1/image-to-3d/{task_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    t0 = time.time()
    while True:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        progress = data.get("progress")
        print(f"[MESHY] {task_id}: status={status}, progress={progress}")
        if status == "SUCCEEDED":
            return data
        if status in ("FAILED", "CANCELED"):
            raise RuntimeError(f"Meshy task {task_id} failed: {data.get('task_error')}")
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Meshy task {task_id} timed out")
        time.sleep(poll_seconds)

def download_file(url: str, out_path: Path):
    import urllib.request
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[MESHY] Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)

def main():
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    assets_prefix = os.getenv("ASSETS_PREFIX")  # e.g. scenes/<sceneId>/assets
    meshy_api_key = os.getenv("MESHY_API_KEY")

    if not assets_prefix or not meshy_api_key:
        print("[MESHY] ASSETS_PREFIX and MESHY_API_KEY required", file=sys.stderr)
        sys.exit(1)

    root = Path("/mnt/gcs")
    assets_root = root / assets_prefix
    plan_path = assets_root / "scene_assets.json"

    if not plan_path.is_file():
        print(f"[MESHY] ERROR: assets plan not found: {plan_path}", file=sys.stderr)
        sys.exit(1)

    plan = load_assets_plan(plan_path)
    objs = plan.get("objects", [])
    print(f"[MESHY] Loaded plan with {len(objs)} objects")

    for obj in objs:
        if obj.get("pipeline") != "meshy":
            continue

        oid = obj["id"]
        crop_path = root / obj["crop_path"]
        if not crop_path.is_file():
            print(f"[MESHY] WARNING: missing crop for obj {oid}: {crop_path}", file=sys.stderr)
            continue

        data_uri = image_to_data_uri(crop_path)
        task_id = create_image_to_3d_task(meshy_api_key, data_uri)
        task = wait_for_task(meshy_api_key, task_id)

        model_urls = task.get("model_urls", {})
        usdz_url = model_urls.get("usdz") or model_urls.get("glb")
        if not usdz_url:
            print(f"[MESHY] WARNING: no model URL for obj {oid}, task {task_id}", file=sys.stderr)
            continue

        out_dir = assets_root / "static" / f"obj_{oid}"
        out_model = out_dir / ("model.usdz" if "usdz" in usdz_url else "model.glb")
        download_file(usdz_url, out_model)

        # Optionally textures
        for i, tex in enumerate(task.get("texture_urls", [])):
            base_color = tex.get("base_color")
            if base_color:
                download_file(base_color, out_dir / f"texture_{i}_basecolor.png")

    print("[MESHY] Done.")

if __name__ == "__main__":
    main()
