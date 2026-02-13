from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

from .asset_validation import is_validation_enabled, should_soft_repair, validate_asset_candidate
from .asset_retrieval import AssetRetrievalService
from .asset_retrieval_rollout import effective_retrieval_mode, update_rollout_state


_ASSET_EXTENSIONS = {".usd", ".usda", ".usdz"}
_TEXTURE_DIR_NAMES = {"texture", "textures", "tex", "texutre", "materials"}
_CATEGORY_SYNONYMS: Dict[str, List[str]] = {
    "refrigerator": ["fridge", "refrigerator"],
    "cabinet": ["cabinet", "drawer", "closet", "wardrobe"],
    "counter": ["counter", "island", "worktop"],
    "table": ["table", "desk", "workbench"],
    "sink": ["sink", "basin"],
    "mug": ["mug", "cup", "glass"],
    "plate": ["plate", "dish", "tray"],
    "bottle": ["bottle", "flask", "jar"],
    "bookshelf": ["bookshelf", "shelf", "rack"],
    "monitor": ["monitor", "screen", "display"],
    "chair": ["chair", "stool", "seat"],
    "sofa": ["sofa", "couch"],
}

_SAM3D_DEFAULT_BASE_URL = ""
_HUNYUAN_DEFAULT_BASE_URL = ""


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dims_for_object(obj: Mapping[str, Any]) -> Dict[str, float]:
    dims = obj.get("dimensions_est") if isinstance(obj.get("dimensions_est"), Mapping) else {}
    width = max(0.02, _safe_float(dims.get("width"), 0.25))
    height = max(0.02, _safe_float(dims.get("height"), 0.25))
    depth = max(0.02, _safe_float(dims.get("depth"), 0.25))
    return {
        "width": width,
        "height": height,
        "depth": depth,
    }


def _write_placeholder_model_usd(path: Path, *, dims: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sx = round(float(dims.get("width", 0.25)), 6)
    sy = round(float(dims.get("height", 0.25)), 6)
    sz = round(float(dims.get("depth", 0.25)), 6)
    payload = f'''#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{{
    def Cube "Geom"
    {{
        double size = 1
        double3 xformOp:scale = ({sx}, {sy}, {sz})
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }}
}}
'''
    path.write_text(payload, encoding="utf-8")


def _write_reference_model_usd(path: Path, *, reference_rel_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ref = reference_rel_path.replace("\\", "/")
    payload = f'''#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root" (
    prepend references = @{ref}@
)
{{
}}
'''
    path.write_text(payload, encoding="utf-8")


_THIN_COVERING_CATEGORIES = frozenset({
    "rug", "mat", "poster", "painting", "curtain", "tablecloth",
    "placemat", "coaster", "doormat", "bathmat", "tapestry",
    "wall_art", "floor_mat", "mousepad",
})


def _is_thin_covering(obj: Mapping[str, Any]) -> bool:
    """Return True if *obj* should be materialised as a flat textured quad."""
    cat = str(obj.get("category") or "").lower().replace(" ", "_")
    name = str(obj.get("name") or "").lower().replace(" ", "_")
    return cat in _THIN_COVERING_CATEGORIES or name in _THIN_COVERING_CATEGORIES


def _write_thin_covering_usd(path: Path, *, dims: Mapping[str, float]) -> None:
    """Write a flat quad USD — thin covering for rugs, posters, mats, etc."""
    path.parent.mkdir(parents=True, exist_ok=True)
    hw = round(float(dims.get("width", 1.0)) / 2, 6)
    hd = round(float(dims.get("depth", 1.0)) / 2, 6)
    thickness = 0.005  # 5 mm
    payload = f'''#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{{
    def Mesh "Geom"
    {{
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [({-hw}, 0, {-hd}), ({hw}, 0, {-hd}), ({hw}, {thickness}, {hd}), ({-hw}, {thickness}, {hd})]
        normal3f[] normals = [(0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
    }}
}}
'''
    path.write_text(payload, encoding="utf-8")


def _tokenize(text: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) >= 2]


def _library_prefixes() -> List[str]:
    raw = (os.getenv("TEXT_ASSET_LIBRARY_PREFIXES") or "scenes").strip()
    prefixes = [segment.strip().strip("/") for segment in raw.split(",") if segment.strip()]
    cache_prefix = (os.getenv("TEXT_ASSET_GENERATED_CACHE_PREFIX") or "asset-library/generated-text").strip().strip("/")
    if _is_truthy(os.getenv("TEXT_ASSET_GENERATED_CACHE_ENABLED"), default=True) and cache_prefix:
        if cache_prefix not in prefixes:
            prefixes.append(cache_prefix)
    return prefixes


def _is_truthy(raw: Optional[str], *, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _asset_generation_provider_chain() -> List[str]:
    if not _is_truthy(os.getenv("TEXT_ASSET_GENERATION_ENABLED"), default=True):
        return []

    configured_chain = (os.getenv("TEXT_ASSET_GENERATION_PROVIDER_CHAIN") or "").strip().lower()
    if configured_chain:
        return [part.strip() for part in configured_chain.split(",") if part.strip()]

    provider = (os.getenv("TEXT_ASSET_GENERATION_PROVIDER") or "sam3d").strip().lower()
    if provider == "sam3d":
        return ["sam3d", "hunyuan3d"]
    if provider == "hunyuan3d":
        return ["hunyuan3d", "sam3d"]
    return [provider]


def _mesh_generation_prompt(obj: Mapping[str, Any]) -> str:
    category = str(obj.get("category") or "object").strip().lower()
    name = str(obj.get("name") or category).strip().lower()
    description = str(obj.get("description") or "").strip()
    sim_role = str(obj.get("sim_role") or "manipulable_object").strip().lower()
    role_hint = "robot-manipulable prop" if sim_role == "manipulable_object" else "furniture/appliance"
    base = f"A realistic {category} ({name}) {role_hint} for indoor robotic simulation, clean topology, physically plausible scale."
    if description:
        return f"{base} Details: {description}"
    return base


def _http_json_request(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: Optional[Mapping[str, Any]] = None,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=body, method=method.upper())
    for key, value in headers.items():
        req.add_header(key, value)

    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:  # noqa: S310 - URL is controlled by env/config
        raw = resp.read()
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"Unexpected response payload type from {url}: {type(decoded).__name__}")
    return decoded


def _provider_base_url(provider: str) -> str:
    if provider == "sam3d":
        base = (
            os.getenv("TEXT_SAM3D_API_HOST")
            or os.getenv("SAM3D_API_HOST")
            or os.getenv("TEXT_SAM3D_BASE_URL")
            or _SAM3D_DEFAULT_BASE_URL
        )
        return base.rstrip("/")
    if provider == "hunyuan3d":
        base = (
            os.getenv("TEXT_HUNYUAN_API_HOST")
            or os.getenv("HUNYUAN_API_HOST")
            or os.getenv("TEXT_HUNYUAN_BASE_URL")
            or _HUNYUAN_DEFAULT_BASE_URL
        )
        return base.rstrip("/")
    return ""


def _provider_api_key(provider: str) -> str:
    if provider == "sam3d":
        return (os.getenv("SAM3D_API_KEY") or os.getenv("TEXT_SAM3D_API_KEY") or "").strip()
    if provider == "hunyuan3d":
        return (os.getenv("HUNYUAN_API_KEY") or os.getenv("TEXT_HUNYUAN_API_KEY") or "").strip()
    return ""


def _provider_model(provider: str) -> str:
    if provider == "sam3d":
        return (os.getenv("SAM3D_MODEL") or os.getenv("TEXT_SAM3D_MODEL") or "sam3d").strip()
    if provider == "hunyuan3d":
        return (os.getenv("HUNYUAN_MODEL") or os.getenv("TEXT_HUNYUAN_MODEL") or "hunyuan3d-2.1").strip()
    return provider


def _provider_text_endpoints(provider: str) -> List[str]:
    if provider == "sam3d":
        raw = (
            os.getenv("TEXT_SAM3D_TEXT_ENDPOINTS")
            or os.getenv("SAM3D_TEXT_ENDPOINTS")
            or "/openapi/v1/text-to-3d,/v1/text-to-3d"
        ).strip()
        return [segment.strip() for segment in raw.split(",") if segment.strip()]
    if provider == "hunyuan3d":
        raw = (
            os.getenv("TEXT_HUNYUAN_TEXT_ENDPOINTS")
            or os.getenv("HUNYUAN_TEXT_ENDPOINTS")
            or "/openapi/v1/text-to-3d,/v1/text-to-3d"
        ).strip()
        return [segment.strip() for segment in raw.split(",") if segment.strip()]
    return []


def _provider_timeout_seconds(provider: str) -> int:
    if provider == "sam3d":
        raw = (os.getenv("TEXT_SAM3D_TIMEOUT_SECONDS") or "1800").strip()
    elif provider == "hunyuan3d":
        raw = (os.getenv("TEXT_HUNYUAN_TIMEOUT_SECONDS") or "1800").strip()
    else:
        raw = "1800"
    try:
        return max(60, int(raw))
    except ValueError:
        return 1800


def _provider_poll_seconds(provider: str) -> int:
    if provider == "sam3d":
        raw = (os.getenv("TEXT_SAM3D_POLL_SECONDS") or "10").strip()
    elif provider == "hunyuan3d":
        raw = (os.getenv("TEXT_HUNYUAN_POLL_SECONDS") or "10").strip()
    else:
        raw = "10"
    try:
        return max(2, int(raw))
    except ValueError:
        return 10


def _extract_task_id(payload: Mapping[str, Any]) -> Optional[str]:
    for field in ("result", "task_id", "id"):
        value = payload.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _extract_model_url(payload: Mapping[str, Any]) -> Optional[str]:
    model_urls = payload.get("model_urls")
    if isinstance(model_urls, Mapping):
        for key in ("usdz", "glb", "obj", "fbx", "usd"):
            value = model_urls.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for field in ("model_url", "download_url", "output_url"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    output = payload.get("output")
    if isinstance(output, Mapping):
        for field in ("model_url", "download_url"):
            value = output.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def _extract_texture_urls(payload: Mapping[str, Any]) -> List[Mapping[str, Any] | str]:
    texture_urls = payload.get("texture_urls")
    if isinstance(texture_urls, list):
        return texture_urls
    output = payload.get("output")
    if isinstance(output, Mapping):
        nested = output.get("texture_urls")
        if isinstance(nested, list):
            return nested
    return []


def _mesh_filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    if not name:
        return "model.glb"
    suffix = Path(name).suffix.lower()
    if suffix in {".glb", ".usdz", ".usd", ".usda", ".obj", ".fbx"}:
        return name
    return f"{Path(name).stem}.glb"


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_path)  # noqa: S310 - URL is controlled by provider response


def _cache_generated_asset_bundle(
    *,
    root: Path,
    bundle_dir: Path,
    obj: Mapping[str, Any],
    provider: str,
) -> Optional[str]:
    if not _is_truthy(os.getenv("TEXT_ASSET_GENERATED_CACHE_ENABLED"), default=True):
        return None

    cache_prefix = (os.getenv("TEXT_ASSET_GENERATED_CACHE_PREFIX") or "asset-library/generated-text").strip().strip("/")
    if not cache_prefix:
        return None

    category = str(obj.get("category") or "object").strip().lower()
    name = str(obj.get("name") or category).strip().lower()
    description = str(obj.get("description") or "").strip().lower()
    digest = sha256(f"{provider}|{category}|{name}|{description}".encode("utf-8")).hexdigest()[:12]
    cache_dir = root / cache_prefix / category / digest
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle_dir, cache_dir, dirs_exist_ok=True)
    return cache_dir.relative_to(root).as_posix()


def _create_and_poll_provider_task(
    *,
    provider: str,
    base_url: str,
    headers: Mapping[str, str],
    prompt: str,
) -> Optional[Dict[str, Any]]:
    timeout_seconds = _provider_timeout_seconds(provider)
    poll_seconds = _provider_poll_seconds(provider)
    model_name = _provider_model(provider)
    create_error: Optional[str] = None

    for endpoint in _provider_text_endpoints(provider):
        create_url = f"{base_url}{endpoint}"
        try:
            payload = {
                "prompt": prompt,
                "model": model_name,
                "ai_model": model_name,
                "topology": "triangle",
                "should_texture": True,
                "enable_pbr": True,
            }
            response = _http_json_request(
                method="POST",
                url=create_url,
                headers=headers,
                payload=payload,
                timeout_seconds=90,
            )

            direct_model_url = _extract_model_url(response)
            if direct_model_url:
                return {
                    "endpoint_used": endpoint,
                    "task_id": _extract_task_id(response),
                    "status_url": response.get("status_url"),
                    "create_error": create_error,
                    "result": response,
                }

            task_id = _extract_task_id(response)
            if not task_id:
                create_error = f"missing_task_id:{endpoint}"
                continue

            status_url_raw = response.get("status_url")
            status_url = (
                str(status_url_raw).strip()
                if isinstance(status_url_raw, str) and str(status_url_raw).strip()
                else f"{create_url.rstrip('/')}/{task_id}"
            )
            t0 = time.time()
            while True:
                poll_response = _http_json_request(
                    method="GET",
                    url=status_url,
                    headers=headers,
                    timeout_seconds=60,
                )
                status = str(poll_response.get("status") or poll_response.get("state") or "").strip().upper()
                if status in {"SUCCEEDED", "SUCCESS", "COMPLETED"}:
                    return {
                        "endpoint_used": endpoint,
                        "task_id": task_id,
                        "status_url": status_url,
                        "create_error": create_error,
                        "result": poll_response,
                    }
                if status in {"FAILED", "CANCELED", "CANCELLED", "ERROR"}:
                    create_error = f"task_failed:{endpoint}:{task_id}:{status}"
                    break
                if time.time() - t0 >= timeout_seconds:
                    create_error = f"task_timeout:{endpoint}:{task_id}"
                    break
                time.sleep(poll_seconds)
        except Exception as exc:
            create_error = f"{endpoint}:{exc}"
    return None


def _generate_asset_with_api_provider(
    *,
    root: Path,
    obj: Mapping[str, Any],
    obj_dir: Path,
    provider: str,
) -> Optional[Dict[str, Any]]:
    api_key = _provider_api_key(provider)
    base_url = _provider_base_url(provider)
    if not api_key or not base_url:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }
    prompt = _mesh_generation_prompt(obj)
    task_meta = _create_and_poll_provider_task(
        provider=provider,
        base_url=base_url,
        headers=headers,
        prompt=prompt,
    )
    if task_meta is None:
        return None

    final_payload = task_meta.get("result")
    if not isinstance(final_payload, Mapping):
        return None

    model_url = _extract_model_url(final_payload)
    if not model_url:
        return None

    bundle_dir = obj_dir / "generated_asset"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model_filename = _mesh_filename_from_url(model_url)
    downloaded_asset = bundle_dir / model_filename
    _download_file(model_url, downloaded_asset)

    # Best-effort texture download for richer materials.
    texture_urls = _extract_texture_urls(final_payload)
    tex_dir = bundle_dir / "textures"
    for idx, entry in enumerate(texture_urls):
        if isinstance(entry, str) and entry.strip():
            _download_file(entry.strip(), tex_dir / f"texture_{idx}.png")
        elif isinstance(entry, Mapping):
            for key in ("base_color", "normal", "roughness", "metallic"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    _download_file(value.strip(), tex_dir / f"texture_{idx}_{key}.png")

    provider_task_json = bundle_dir / f"{provider}_task.json"
    provider_task_json.write_text(
        json.dumps(
            {
                "provider": provider,
                "prompt": prompt,
                "endpoint": task_meta.get("endpoint_used"),
                "task_id": task_meta.get("task_id"),
                "status_url": task_meta.get("status_url"),
                "create_error": task_meta.get("create_error"),
                "result": final_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cached_rel_path = _cache_generated_asset_bundle(
        root=root,
        bundle_dir=bundle_dir,
        obj=obj,
        provider=provider,
    )
    source_path = cached_rel_path or downloaded_asset.relative_to(root).as_posix()

    return {
        "reference_rel": downloaded_asset.relative_to(obj_dir).as_posix(),
        "source_path": source_path,
        "source_kind": f"generated_{provider}",
        "model_or_library": f"{provider}_text_to_3d",
    }


def _generate_asset_with_sam3d(
    *,
    root: Path,
    obj: Mapping[str, Any],
    obj_dir: Path,
) -> Optional[Dict[str, Any]]:
    return _generate_asset_with_api_provider(root=root, obj=obj, obj_dir=obj_dir, provider="sam3d")


def _generate_asset_with_hunyuan3d(
    *,
    root: Path,
    obj: Mapping[str, Any],
    obj_dir: Path,
) -> Optional[Dict[str, Any]]:
    return _generate_asset_with_api_provider(root=root, obj=obj, obj_dir=obj_dir, provider="hunyuan3d")


def _check_generated_asset_cache(
    *,
    root: Path,
    obj: Mapping[str, Any],
    obj_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Check if a previously generated asset exists in the cache."""
    if not _is_truthy(os.getenv("TEXT_ASSET_GENERATED_CACHE_ENABLED"), default=True):
        return None
    cache_prefix = (os.getenv("TEXT_ASSET_GENERATED_CACHE_PREFIX") or "asset-library/generated-text").strip().strip("/")
    if not cache_prefix:
        return None
    category = str(obj.get("category") or "object").strip().lower()
    name = str(obj.get("name") or category).strip().lower()
    description = str(obj.get("description") or "").strip().lower()
    for provider in _asset_generation_provider_chain():
        digest = sha256(f"{provider}|{category}|{name}|{description}".encode("utf-8")).hexdigest()[:12]
        cache_dir = root / cache_prefix / category / digest
        if cache_dir.is_dir() and any(cache_dir.iterdir()):
            logger.info("[ADAPTER] cache hit for %s/%s (provider=%s, digest=%s)", category, name, provider, digest)
            shutil.copytree(cache_dir, obj_dir, dirs_exist_ok=True)
            glb_files = list(obj_dir.glob("*.glb"))
            return {
                "asset_path": glb_files[0].relative_to(root).as_posix() if glb_files else None,
                "source_kind": "generated_cached",
                "provider": provider,
                "cache_digest": digest,
            }
    return None


def _generate_asset_with_provider(
    *,
    root: Path,
    obj: Mapping[str, Any],
    obj_dir: Path,
) -> Optional[Dict[str, Any]]:
    # Check cache before calling provider APIs
    cached = _check_generated_asset_cache(root=root, obj=obj, obj_dir=obj_dir)
    if cached is not None:
        return cached

    providers = _asset_generation_provider_chain()
    for provider in providers:
        if provider == "sam3d":
            generated = _generate_asset_with_sam3d(root=root, obj=obj, obj_dir=obj_dir)
            if generated is not None:
                return generated
        elif provider == "hunyuan3d":
            generated = _generate_asset_with_hunyuan3d(root=root, obj=obj, obj_dir=obj_dir)
            if generated is not None:
                return generated
    return None


def _scan_asset_library(root: Path) -> List[Dict[str, Any]]:
    if not _is_truthy(os.getenv("TEXT_ASSET_RETRIEVAL_ENABLED"), default=True):
        return []

    max_files_raw = (os.getenv("TEXT_ASSET_LIBRARY_MAX_FILES") or "2500").strip()
    try:
        max_files = max(1, int(max_files_raw))
    except ValueError:
        max_files = 2500

    entries: List[Dict[str, Any]] = []
    for prefix in _library_prefixes():
        prefix_path = root / prefix
        if not prefix_path.exists():
            continue

        for dirpath, _, filenames in os.walk(prefix_path):
            if len(entries) >= max_files:
                break
            for filename in filenames:
                if len(entries) >= max_files:
                    break
                candidate = Path(dirpath) / filename
                if candidate.suffix.lower() not in _ASSET_EXTENSIONS:
                    continue
                if filename.startswith("."):
                    continue
                rel = candidate.relative_to(root).as_posix()
                tokens = set(_tokenize(candidate.stem) + _tokenize(rel))
                entries.append(
                    {
                        "path": candidate,
                        "path_rel": rel,
                        "tokens": tokens,
                    }
                )
        if len(entries) >= max_files:
            break
    return entries


def _query_terms(*, category: str, name: str) -> List[str]:
    base = _tokenize(category) + _tokenize(name)
    if category in _CATEGORY_SYNONYMS:
        base.extend(_CATEGORY_SYNONYMS[category])
    return sorted(set(base))


def _asset_match_score(*, terms: List[str], entry_tokens: set[str]) -> float:
    if not terms:
        return 0.0
    overlap = len([term for term in terms if term in entry_tokens])
    if overlap == 0:
        return 0.0
    return overlap / len(terms)


def _choose_retrieved_asset(
    *,
    obj: Mapping[str, Any],
    library_entries: List[Dict[str, Any]],
    used_paths: set[str],
) -> Optional[Dict[str, Any]]:
    if not library_entries:
        return None

    category = str(obj.get("category") or "object").strip().lower()
    name = str(obj.get("name") or category).strip().lower()
    terms = _query_terms(category=category, name=name)

    min_score_raw = (os.getenv("TEXT_ASSET_LIBRARY_MIN_SCORE") or "0.25").strip()
    try:
        min_score = max(0.0, min(1.0, float(min_score_raw)))
    except ValueError:
        min_score = 0.25

    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for entry in library_entries:
        path_rel = str(entry["path_rel"])
        if path_rel in used_paths:
            continue
        score = _asset_match_score(terms=terms, entry_tokens=set(entry["tokens"]))
        if score < min_score:
            continue
        if score > best_score:
            best = entry
            best_score = score

    if best is None:
        return None
    selected = dict(best)
    selected["score"] = round(best_score, 4)
    return selected


def _copy_asset_bundle(source_asset: Path, bundle_dir: Path) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    source_dir = source_asset.parent
    dest_asset = bundle_dir / source_asset.name
    shutil.copy2(source_asset, dest_asset)

    # Copy direct sibling USD files that may be referenced.
    for sibling in source_dir.iterdir():
        if sibling == source_asset:
            continue
        if sibling.is_file() and sibling.suffix.lower() in _ASSET_EXTENSIONS:
            shutil.copy2(sibling, bundle_dir / sibling.name)

    # Copy texture/material folders when present.
    for sibling in source_dir.iterdir():
        if not sibling.is_dir():
            continue
        if sibling.name.lower() not in _TEXTURE_DIR_NAMES:
            continue
        shutil.copytree(sibling, bundle_dir / sibling.name, dirs_exist_ok=True)

    return dest_asset


def _materialize_metadata(
    path: Path,
    *,
    obj: Mapping[str, Any],
    dims: Mapping[str, float],
    source_path: Optional[str] = None,
    source_kind: str = "placeholder",
) -> None:
    metadata = {
        "id": obj.get("id"),
        "class_name": obj.get("category") or obj.get("name") or "object",
        "mesh_bounds": {
            "export": {
                "center": [0.0, 0.0, 0.0],
                "size": [dims["width"], dims["height"], dims["depth"]],
            }
        },
        "source": {
            "type": "text",
            "asset_strategy": obj.get("asset_strategy", "generated"),
            "materialization": source_kind,
        },
    }
    if source_path:
        metadata["source"]["asset_path"] = source_path
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _object_transform_to_layout(obj: Mapping[str, Any]) -> Dict[str, Any]:
    transform = obj.get("transform") if isinstance(obj.get("transform"), Mapping) else {}
    position = transform.get("position") if isinstance(transform.get("position"), Mapping) else {}
    center3d = [
        _safe_float(position.get("x"), 0.0),
        _safe_float(position.get("y"), 0.0),
        _safe_float(position.get("z"), 0.0),
    ]

    layout_obj: Dict[str, Any] = {
        "id": str(obj.get("id")),
        "class_name": str(obj.get("category") or obj.get("name") or "object"),
        "center3d": center3d,
    }

    if obj.get("placement_stage") is not None:
        layout_obj["placement_stage"] = str(obj.get("placement_stage"))
    if obj.get("parent_support_id") is not None:
        layout_obj["parent_support_id"] = str(obj.get("parent_support_id"))
    if isinstance(obj.get("surface_local_se2"), Mapping):
        layout_obj["surface_local_se2"] = dict(obj.get("surface_local_se2") or {})
    if isinstance(obj.get("asset_validation"), Mapping):
        layout_obj["asset_validation"] = dict(obj.get("asset_validation") or {})

    if "rotation_quaternion" in transform and isinstance(transform["rotation_quaternion"], Mapping):
        quat = transform["rotation_quaternion"]
        layout_obj["rotation_quaternion"] = {
            "w": _safe_float(quat.get("w"), 1.0),
            "x": _safe_float(quat.get("x"), 0.0),
            "y": _safe_float(quat.get("y"), 0.0),
            "z": _safe_float(quat.get("z"), 0.0),
        }

    if "obb" in obj and isinstance(obj["obb"], Mapping):
        layout_obj["obb"] = obj["obb"]

    return layout_obj


def materialize_placeholder_assets(
    *,
    root: Path,
    scene_id: str,
    assets_prefix: str,
    objects: List[Dict[str, Any]],
    room_type: str = "",
    retrieval_report: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Create placeholder model assets for text-generated objects.

    Returns a list of provenance entries with materialized asset paths.
    """

    assets_root = root / assets_prefix
    assets_root.mkdir(parents=True, exist_ok=True)

    provenance_assets: List[Dict[str, Any]] = []
    library_entries = _scan_asset_library(root)
    used_paths: set[str] = set()
    retrieval_mode = effective_retrieval_mode(root)
    retrieval_service = AssetRetrievalService(root=root, retrieval_mode=retrieval_mode)
    retrieval_decisions: List[Dict[str, Any]] = []
    ann_attempted = 0
    ann_errors = 0
    ann_candidates = 0
    validation_enabled = is_validation_enabled()

    for obj in objects:
        oid = str(obj.get("id") or "")
        if not oid:
            continue
        obj_dir = assets_root / oid
        obj_dir.mkdir(parents=True, exist_ok=True)

        dims = _dims_for_object(obj)
        model_path = obj_dir / "model.usd"
        metadata_path = obj_dir / "metadata.json"

        # Fast-path: thin coverings (rugs, posters, mats) get a flat quad
        # instead of a full 3D retrieval/generation round-trip.
        if _is_thin_covering(obj):
            _write_thin_covering_usd(model_path, dims=dims)
            _materialize_metadata(
                metadata_path,
                obj=obj,
                dims=dims,
                source_kind="thin_covering_usd",
            )
            provenance_assets.append({
                "object_id": oid,
                "strategy": "generated",
                "source_type": "generated",
                "model_or_library": "thin_covering",
                "materialization": "thin_covering_usd",
                "source_path": "",
                "retrieval_method": "thin_covering",
                "semantic_score": None,
                "lexical_score": None,
                "fallback_reason": None,
                "validation": {"passed": True, "status": "pass", "source_kind": "thin_covering_usd"},
            })
            continue

        decision = retrieval_service.select_asset(
            scene_id=scene_id,
            room_type=room_type,
            obj=obj,
            library_entries=library_entries,
            used_paths=used_paths,
            lexical_selector=_choose_retrieved_asset,
        )
        decision_payload = decision.to_dict()
        retrieval_decisions.append(decision_payload)
        if bool(decision_payload.get("ann_attempted")):
            ann_attempted += 1
        if decision_payload.get("ann_error"):
            ann_errors += 1
        ann_candidates += int(decision_payload.get("ann_candidate_count") or 0)

        retrieved = decision.to_retrieved_entry(root=root)
        if retrieved is not None:
            source_path = Path(retrieved["path"])
            used_paths.add(str(retrieved["path_rel"]))
            bundle_dir = obj_dir / "retrieved_asset"
            copied_asset = _copy_asset_bundle(source_path, bundle_dir)
            reference_rel = copied_asset.relative_to(obj_dir).as_posix()
            _write_reference_model_usd(model_path, reference_rel_path=reference_rel)
            _materialize_metadata(
                metadata_path,
                obj=obj,
                dims=dims,
                source_path=str(retrieved["path_rel"]),
                source_kind="retrieved_asset_bundle",
            )
            source_type = "retrieved"
            model_or_library = "text_asset_library"
            materialization = "retrieved_asset_bundle"
            source_path_hint = str(retrieved["path_rel"])
            retrieval_method = str(decision.method)
            semantic_score = decision.semantic_score
            lexical_score = decision.lexical_score
            fallback_reason = decision.fallback_reason
        else:
            generated = _generate_asset_with_provider(root=root, obj=obj, obj_dir=obj_dir)
            if generated is not None:
                _write_reference_model_usd(model_path, reference_rel_path=str(generated["reference_rel"]))
                _materialize_metadata(
                    metadata_path,
                    obj=obj,
                    dims=dims,
                    source_path=str(generated.get("source_path") or ""),
                    source_kind=str(generated.get("source_kind") or "generated_external"),
                )
                source_type = "generated"
                model_or_library = str(generated.get("model_or_library") or "generated_external")
                materialization = str(generated.get("source_kind") or "generated_external")
                source_path_hint = str(generated.get("source_path") or "")
                retrieval_method = str(decision.method) if decision.method != "none" else "generated"
                semantic_score = decision.semantic_score
                lexical_score = decision.lexical_score
                fallback_reason = decision.fallback_reason or "retrieval_miss_generated"
            else:
                _write_placeholder_model_usd(model_path, dims=dims)
                _materialize_metadata(
                    metadata_path,
                    obj=obj,
                    dims=dims,
                    source_kind="placeholder_usd",
                )
                source_type = "generated" if obj.get("asset_strategy") != "retrieved" else "retrieved"
                model_or_library = "partnet_mobility" if obj.get("asset_strategy") == "retrieved" else "textgen_placeholder"
                materialization = "placeholder_usd"
                source_path_hint = ""
                retrieval_method = str(decision.method) if decision.method != "none" else "placeholder"
                semantic_score = decision.semantic_score
                lexical_score = decision.lexical_score
                fallback_reason = decision.fallback_reason or "retrieval_miss_placeholder"

        validation = validate_asset_candidate(
            obj=obj,
            source_kind=materialization,
            source_path=source_path_hint,
            room_type=room_type,
        )
        if validation_enabled and should_soft_repair(validation) and materialization != "placeholder_usd":
            _write_placeholder_model_usd(model_path, dims=dims)
            _materialize_metadata(
                metadata_path,
                obj=obj,
                dims=dims,
                source_kind="placeholder_usd",
            )
            source_type = "generated"
            model_or_library = "textgen_placeholder"
            materialization = "placeholder_usd"
            source_path_hint = ""
            fallback_reason = "asset_validation_soft_repair"
            validation = dict(validation)
            validation["soft_repair_applied"] = True
            validation["passed"] = True
            validation["status"] = "pass"

        obj["asset_validation"] = validation
        if obj.get("placement_stage") is None:
            obj["placement_stage"] = (
                "manipulands" if str(obj.get("sim_role") or "").strip().lower() == "manipulable_object" else "furniture"
            )
        if obj.get("parent_support_id") is None and str(obj.get("sim_role") or "").strip().lower() == "manipulable_object":
            obj["parent_support_id"] = "room_floor"
        if obj.get("surface_local_se2") is None and str(obj.get("sim_role") or "").strip().lower() == "manipulable_object":
            obj["surface_local_se2"] = {"x": 0.0, "z": 0.0, "yaw_rad": 0.0}

        object_files = [
            file_path.relative_to(root).as_posix()
            for file_path in sorted(obj_dir.rglob("*"))
            if file_path.is_file()
        ]

        provenance_assets.append(
            {
                "object_id": oid,
                "asset_id": f"text::{scene_id}::{oid}",
                "path": f"{assets_prefix}/{oid}/model.usd",
                "source": source_type,
                "strategy": obj.get("asset_strategy", "generated"),
                "model_or_library": model_or_library,
                "materialization": materialization,
                "source_path": source_path_hint,
                "retrieval_method": retrieval_method,
                "retrieval_mode": retrieval_mode,
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "fallback_reason": fallback_reason,
                "asset_validation_score": validation.get("score"),
                "asset_validation_passed": bool(validation.get("passed")),
                "files": object_files,
            }
        )

    if retrieval_report is not None:
        method_counts: Dict[str, int] = {}
        latencies: List[float] = []
        for payload in retrieval_decisions:
            method = str(payload.get("method") or "none")
            method_counts[method] = method_counts.get(method, 0) + 1
            latency_raw = payload.get("latency_ms")
            try:
                latencies.append(float(latency_raw))
            except (TypeError, ValueError):
                continue
        sorted_latencies = sorted(latencies)
        p95_index = int((len(sorted_latencies) - 1) * 0.95) if sorted_latencies else 0
        latency_p95 = sorted_latencies[p95_index] if sorted_latencies else 0.0
        retrieval_report.update(
            {
                "schema_version": "v1",
                "scene_id": scene_id,
                "retrieval_mode": retrieval_mode,
                "object_count": len(retrieval_decisions),
                "library_entries_scanned": len(library_entries),
                "method_counts": method_counts,
                "ann_attempted_count": ann_attempted,
                "ann_error_count": ann_errors,
                "ann_candidate_count_total": ann_candidates,
                "latency_ms": {
                    "mean": round(sum(sorted_latencies) / max(1, len(sorted_latencies)), 4),
                    "p95": round(float(latency_p95), 4),
                },
                "decisions": retrieval_decisions,
            }
        )

    return provenance_assets


def _bucket_name() -> str:
    return (os.getenv("BUCKET") or "").strip()


def _to_gcs_uri(path_rel: str) -> Optional[str]:
    bucket = _bucket_name()
    if not bucket:
        return None
    return f"gs://{bucket}/{path_rel.lstrip('/')}"


def _asset_catalog_enabled() -> bool:
    return _is_truthy(os.getenv("TEXT_ASSET_CATALOG_ENABLED"), default=True)


def _asset_replication_enabled() -> bool:
    return _is_truthy(os.getenv("TEXT_ASSET_REPLICATION_ENABLED"), default=True)


def _asset_embedding_enabled() -> bool:
    return _is_truthy(os.getenv("TEXT_ASSET_ANN_ENABLED"), default=True)


def _enqueue_asset_embedding_request(
    *,
    root: Path,
    scene_id: str,
    assets_prefix: str,
    source_request: Mapping[str, Any],
    catalog_assets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not _asset_embedding_enabled():
        return {
            "enabled": False,
            "queued": False,
        }
    if not catalog_assets:
        return {
            "enabled": True,
            "queued": False,
            "error": "no_catalog_assets",
        }

    queue_prefix = (os.getenv("TEXT_ASSET_EMBEDDING_QUEUE_PREFIX") or "automation/asset_embedding/queue").strip().strip("/")
    if not queue_prefix:
        return {
            "enabled": True,
            "queued": False,
            "error": "missing_queue_prefix",
        }

    model_name = (os.getenv("TEXT_ASSET_EMBEDDING_MODEL") or "text-embedding-3-small").strip()
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    suffix = sha256(f"{scene_id}|{stamp}|{len(catalog_assets)}".encode("utf-8")).hexdigest()[:10]
    queue_object = f"{queue_prefix}/{stamp}-{scene_id}-{suffix}.json"
    queue_path = root / queue_object

    prompt = str(source_request.get("prompt") or "")
    entries: List[Dict[str, Any]] = []
    for item in catalog_assets:
        asset_id = str(item.get("asset_id") or "").strip()
        if not asset_id:
            continue
        class_name = str(item.get("class_name") or "object")
        sim_roles = [str(role) for role in item.get("sim_roles") or [] if role is not None]
        description = str(item.get("description") or "")
        tags = [str(tag) for tag in item.get("tags") or [] if tag is not None]
        descriptor_parts = [
            class_name,
            description,
            " ".join(sim_roles),
            " ".join(tags),
            prompt,
        ]
        descriptor_text = " ".join(part.strip() for part in descriptor_parts if part and part.strip())
        descriptor_hash = sha256(descriptor_text.encode("utf-8")).hexdigest()
        idempotency_key = sha256(f"{asset_id}|{model_name}|{descriptor_hash}".encode("utf-8")).hexdigest()
        entries.append(
            {
                "asset_id": asset_id,
                "descriptor_text": descriptor_text,
                "descriptor_hash": descriptor_hash,
                "idempotency_key": idempotency_key,
                "class_name": class_name,
                "sim_roles": sim_roles,
                "usd_path": str(item.get("usd_path") or item.get("path") or ""),
                "gcs_uri": str(item.get("gcs_uri") or ""),
                "source": str(item.get("source") or ""),
                "tags": tags,
                "dimensions": item.get("dimensions"),
            }
        )

    if not entries:
        return {
            "enabled": True,
            "queued": False,
            "error": "no_valid_assets",
        }

    payload = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "created_at": now.isoformat(),
        "embedding_model": model_name,
        "source_pipeline": "text-scene-adapter-job",
        "assets_prefix": assets_prefix,
        "assets": entries,
    }
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    marker = root / assets_prefix / ".asset_embedding_enqueued"
    marker.write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "queue_object": queue_object,
                "assets_count": len(entries),
                "embedding_model": model_name,
                "status": "queued",
                "timestamp": now.isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "enabled": True,
        "queued": True,
        "queue_object": queue_object,
        "assets_count": len(entries),
        "embedding_model": model_name,
    }


def _write_asset_retrieval_report(
    *,
    root: Path,
    assets_prefix: str,
    report: Mapping[str, Any],
) -> str:
    report_path = root / assets_prefix / ".asset_retrieval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(dict(report), indent=2), encoding="utf-8")
    return report_path.relative_to(root).as_posix()


def _publish_assets_to_catalog(
    *,
    scene_id: str,
    assets_prefix: str,
    objects: List[Dict[str, Any]],
    provenance_assets: List[Dict[str, Any]],
    source_request: Mapping[str, Any],
    quality_tier: str,
    provider_used: str,
    seed: int,
) -> Dict[str, Any]:
    if not _asset_catalog_enabled():
        return {
            "enabled": False,
            "assets_upserted": 0,
            "scene_upserted": False,
            "assets": [],
        }

    try:
        from tools.asset_catalog import AssetCatalogClient, AssetDocument, SceneDocument
    except Exception as exc:  # pragma: no cover - optional dependency
        return {
            "enabled": True,
            "assets_upserted": 0,
            "scene_upserted": False,
            "error": f"asset_catalog_import_failed:{exc}",
            "assets": [],
        }

    object_by_id: Dict[str, Dict[str, Any]] = {}
    for obj in objects:
        oid = str(obj.get("id") or "")
        if oid:
            object_by_id[oid] = obj

    client = AssetCatalogClient()
    asset_ids: List[str] = []
    assets_upserted = 0
    catalog_assets: List[Dict[str, Any]] = []
    for item in provenance_assets:
        oid = str(item.get("object_id") or "")
        if not oid:
            continue
        obj = object_by_id.get(oid, {})
        asset_id = str(item.get("asset_id") or f"text::{scene_id}::{oid}")
        asset_path = str(item.get("path") or f"{assets_prefix}/{oid}/model.usd")
        gcs_uri = _to_gcs_uri(asset_path)
        physics_hints = obj.get("physics_hints") if isinstance(obj.get("physics_hints"), Mapping) else None
        articulation = obj.get("articulation") if isinstance(obj.get("articulation"), Mapping) else None

        doc = AssetDocument(
            asset_id=asset_id,
            logical_id=oid,
            source=f"text_{item.get('source', 'unknown')}",
            usd_path=asset_path,
            gcs_uri=gcs_uri,
            sim_roles=[str(obj.get("sim_role") or "manipulable_object")],
            class_name=str(obj.get("category") or obj.get("name") or "object"),
            description=str(obj.get("description") or ""),
            physics_profile=dict(physics_hints) if isinstance(physics_hints, Mapping) else None,
            articulation_profile=dict(articulation) if isinstance(articulation, Mapping) else None,
            extra_metadata={
                "scene_id": scene_id,
                "object_id": oid,
                "asset_strategy": item.get("strategy"),
                "materialization": item.get("materialization"),
                "source_path": item.get("source_path"),
                "model_or_library": item.get("model_or_library"),
                "generation_tier": quality_tier,
                "provider": provider_used,
                "seed": seed,
                "source_pipeline": "text-scene-adapter-job",
            },
        )
        client.upsert_asset_document(doc)
        assets_upserted += 1
        asset_ids.append(asset_id)
        catalog_assets.append(
            {
                "asset_id": asset_id,
                "usd_path": asset_path,
                "gcs_uri": gcs_uri,
                "class_name": str(obj.get("category") or obj.get("name") or "object"),
                "description": str(obj.get("description") or ""),
                "sim_roles": [str(obj.get("sim_role") or "manipulable_object")],
                "source": f"text_{item.get('source', 'unknown')}",
                "tags": [str(obj.get("category") or ""), str(obj.get("name") or "")],
                "dimensions": _dims_for_object(obj),
            }
        )

    scene_doc = SceneDocument(
        scene_id=scene_id,
        assets_prefix=assets_prefix,
        asset_ids=asset_ids,
        description=str(source_request.get("prompt") or ""),
        extra_metadata={
            "source_type": "text",
            "quality_tier": quality_tier,
            "provider": provider_used,
            "seed": seed,
            "source_pipeline": "text-scene-adapter-job",
        },
    )
    client.upsert_scene_document(scene_doc)

    return {
        "enabled": True,
        "assets_upserted": assets_upserted,
        "scene_upserted": True,
        "assets_collection": client.config.assets_collection,
        "scenes_collection": client.config.scenes_collection,
        "assets": catalog_assets,
    }


def _enqueue_replication_request(
    *,
    root: Path,
    scene_id: str,
    assets_prefix: str,
    provenance_assets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not _asset_replication_enabled():
        return {
            "enabled": False,
            "queued": False,
        }

    queue_prefix = (os.getenv("TEXT_ASSET_REPLICATION_QUEUE_PREFIX") or "automation/asset_replication/queue").strip().strip("/")
    if not queue_prefix:
        return {
            "enabled": True,
            "queued": False,
            "error": "missing_queue_prefix",
        }

    target = (os.getenv("TEXT_ASSET_REPLICATION_TARGET") or "backblaze_b2").strip()
    target_prefix = (os.getenv("TEXT_ASSET_REPLICATION_TARGET_PREFIX") or "assets").strip().strip("/")

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    suffix = sha256(f"{scene_id}|{stamp}|{len(provenance_assets)}".encode("utf-8")).hexdigest()[:10]
    queue_object = f"{queue_prefix}/{stamp}-{scene_id}-{suffix}.json"
    queue_path = root / queue_object

    assets_payload: List[Dict[str, Any]] = []
    for item in provenance_assets:
        files = [str(path) for path in item.get("files") or [] if isinstance(path, str)]
        if not files:
            continue
        file_entries: List[Dict[str, Any]] = []
        for rel in files:
            file_entries.append(
                {
                    "path": rel,
                    "gcs_uri": _to_gcs_uri(rel),
                    "target_key": f"{target_prefix}/{rel}" if target_prefix else rel,
                }
            )
        assets_payload.append(
            {
                "asset_id": str(item.get("asset_id") or ""),
                "object_id": str(item.get("object_id") or ""),
                "source": str(item.get("source") or ""),
                "model_or_library": str(item.get("model_or_library") or ""),
                "materialization": str(item.get("materialization") or ""),
                "files": file_entries,
            }
        )

    payload = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "created_at": now.isoformat(),
        "source_pipeline": "text-scene-adapter-job",
        "storage_mode": "gcs_hot_with_async_replication",
        "primary_storage": {
            "provider": "gcs",
            "bucket": _bucket_name(),
            "assets_prefix": assets_prefix,
        },
        "replication_target": {
            "provider": target,
            "target_prefix": target_prefix,
        },
        "assets": assets_payload,
    }

    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    marker = root / assets_prefix / ".asset_replication_enqueued"
    marker.write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "queue_object": queue_object,
                "assets_count": len(assets_payload),
                "status": "queued",
                "target": target,
                "timestamp": now.isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "enabled": True,
        "queued": True,
        "queue_object": queue_object,
        "assets_count": len(assets_payload),
        "target": target,
    }


def _build_manifest(
    *,
    scene_id: str,
    room_type: str,
    objects: List[Dict[str, Any]],
    assets_prefix: str,
    source_request: Mapping[str, Any],
    layout_plan: Optional[Mapping[str, Any]],
    provider_used: str,
    quality_tier: str,
    seed: int,
    provenance_assets: List[Dict[str, Any]],
    text_backend: str,
    backend_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    source_type_map = {
        "internal": "text",
        "sage": "text_sage",
        "scenesmith": "text_scenesmith",
        "hybrid_serial": "text_hybrid_serial",
    }
    source_type = source_type_map.get(text_backend, "text")
    manifest_objects: List[Dict[str, Any]] = []
    room_min, room_max = _room_box_from_layout_plan(layout_plan)
    scene_room = {
        "bounds": {
            "width": round(max(0.1, room_max[0] - room_min[0]), 4),
            "depth": round(max(0.1, room_max[2] - room_min[2]), 4),
            "height": round(max(0.1, room_max[1] - room_min[1]), 4),
        },
        "origin": [
            round((room_min[0] + room_max[0]) / 2.0, 4),
            round(room_min[1], 4),
            round((room_min[2] + room_max[2]) / 2.0, 4),
        ],
    }

    for obj in objects:
        oid = str(obj.get("id") or "")
        if not oid:
            continue
        dims = _dims_for_object(obj)
        transform = obj.get("transform") if isinstance(obj.get("transform"), Mapping) else {}
        position = transform.get("position") if isinstance(transform.get("position"), Mapping) else {}
        scale = transform.get("scale") if isinstance(transform.get("scale"), Mapping) else {}

        source_articulation = obj.get("articulation") if isinstance(obj.get("articulation"), Mapping) else {}
        manifest_articulation: Dict[str, Any] = {
            "required": bool(source_articulation.get("required", False)),
            "backend_hint": str(source_articulation.get("backend_hint", "none")),
        }
        candidate_value = source_articulation.get("candidate")
        if candidate_value is not None:
            manifest_articulation["candidate"] = bool(candidate_value)
        if source_articulation.get("detection_source") is not None:
            manifest_articulation["detection_source"] = str(source_articulation.get("detection_source"))
        if source_articulation.get("requirement_source") is not None:
            manifest_articulation["requirement_source"] = str(source_articulation.get("requirement_source"))

        manifest_obj: Dict[str, Any] = {
            "id": oid,
            "name": str(obj.get("name") or oid),
            "category": str(obj.get("category") or "object"),
            "description": str(obj.get("description") or "text-generated object"),
            "sim_role": str(obj.get("sim_role") or "manipulable_object"),
            "asset": {
                "path": f"{assets_prefix}/{oid}/model.usd",
                "source": "text_scene_gen",
                "format": "usd",
            },
            "transform": {
                "position": {
                    "x": _safe_float(position.get("x"), 0.0),
                    "y": _safe_float(position.get("y"), 0.0),
                    "z": _safe_float(position.get("z"), 0.0),
                },
                "scale": {
                    "x": _safe_float(scale.get("x"), 1.0),
                    "y": _safe_float(scale.get("y"), 1.0),
                    "z": _safe_float(scale.get("z"), 1.0),
                },
            },
            "dimensions_est": dims,
            "physics_hints": dict(obj.get("physics_hints") or {}),
            "articulation": manifest_articulation,
            "source": {
                "type": source_type,
                "generation_tier": quality_tier,
                "provider": provider_used,
                "seed": seed,
                "text_backend": text_backend,
            },
        }

        rotation_quaternion = transform.get("rotation_quaternion")
        if isinstance(rotation_quaternion, Mapping):
            manifest_obj["transform"]["rotation_quaternion"] = {
                "w": _safe_float(rotation_quaternion.get("w"), 1.0),
                "x": _safe_float(rotation_quaternion.get("x"), 0.0),
                "y": _safe_float(rotation_quaternion.get("y"), 0.0),
                "z": _safe_float(rotation_quaternion.get("z"), 0.0),
            }

        relationships = obj.get("relationships")
        if isinstance(relationships, list):
            manifest_obj["relationships"] = relationships

        if obj.get("placement_stage") is not None:
            manifest_obj["placement_stage"] = str(obj.get("placement_stage"))
        if obj.get("parent_support_id") is not None:
            manifest_obj["parent_support_id"] = str(obj.get("parent_support_id"))
        if isinstance(obj.get("surface_local_se2"), Mapping):
            manifest_obj["surface_local_se2"] = dict(obj.get("surface_local_se2") or {})
        if isinstance(obj.get("asset_validation"), Mapping):
            manifest_obj["asset_validation"] = dict(obj.get("asset_validation") or {})

        manifest_objects.append(manifest_obj)

    return {
        "version": "1.0.0",
        "scene_id": scene_id,
        "scene": {
            "coordinate_frame": "y_up",
            "meters_per_unit": 1.0,
            "environment_type": room_type,
            "room": scene_room,
        },
        "objects": manifest_objects,
        "metadata": {
            "source": {
                "type": source_type,
                "request_id": scene_id,
                "seed": seed,
                "provider": provider_used,
                "generation_tier": quality_tier,
                "text_backend": text_backend,
            },
            "provenance": {
                "assets": provenance_assets,
                "request": source_request,
                "backends": backend_runs,
            },
            "source_pipeline": "text-scene-adapter-job",
        },
    }


def _room_box_from_layout_plan(layout_plan: Optional[Mapping[str, Any]]) -> tuple[List[float], List[float]]:
    room_min = [-3.0, 0.0, -3.0]
    room_max = [3.0, 3.0, 3.0]
    if isinstance(layout_plan, Mapping):
        room_box = layout_plan.get("room_box")
        if isinstance(room_box, Mapping):
            candidate_min = room_box.get("min")
            candidate_max = room_box.get("max")
            if isinstance(candidate_min, list) and len(candidate_min) == 3:
                room_min = [
                    _safe_float(candidate_min[0], room_min[0]),
                    _safe_float(candidate_min[1], room_min[1]),
                    _safe_float(candidate_min[2], room_min[2]),
                ]
            if isinstance(candidate_max, list) and len(candidate_max) == 3:
                room_max = [
                    _safe_float(candidate_max[0], room_max[0]),
                    _safe_float(candidate_max[1], room_max[1]),
                    _safe_float(candidate_max[2], room_max[2]),
                ]

    normalized_min = [min(room_min[idx], room_max[idx]) for idx in range(3)]
    normalized_max = [max(room_min[idx], room_max[idx]) for idx in range(3)]
    return normalized_min, normalized_max


def _build_layout(
    *,
    scene_id: str,
    objects: List[Dict[str, Any]],
    layout_plan: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    room_min, room_max = _room_box_from_layout_plan(layout_plan)
    wall_thickness_m = 0.12
    openings: List[Dict[str, Any]] = []

    if isinstance(layout_plan, Mapping):
        wall_thickness_m = max(0.06, _safe_float(layout_plan.get("wall_thickness_m"), wall_thickness_m))
        raw_openings = layout_plan.get("openings")
        if isinstance(raw_openings, list):
            openings = [entry for entry in raw_openings if isinstance(entry, Mapping)]

    layout_objects = [_object_transform_to_layout(obj) for obj in objects if obj.get("id")]

    return {
        "scene_id": scene_id,
        "objects": layout_objects,
        "room_box": {
            "min": room_min,
            "max": room_max,
        },
        "wall_thickness_m": round(wall_thickness_m, 4),
        "openings": openings,
    }


def _build_inventory(
    *,
    scene_id: str,
    objects: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "scene_id": scene_id,
        "source": "text_scene_gen",
        "objects": [
            {
                "id": str(obj.get("id")),
                "name": str(obj.get("name") or obj.get("id") or "object"),
                "category": str(obj.get("category") or "object"),
                "sim_role": str(obj.get("sim_role") or "manipulable_object"),
                "description": str(obj.get("description") or ""),
                "asset_strategy": str(obj.get("asset_strategy") or "generated"),
                "articulation_required": bool((obj.get("articulation") or {}).get("required", False)),
                "placement_stage": str(obj.get("placement_stage") or ""),
                "parent_support_id": str(obj.get("parent_support_id") or ""),
                "asset_validation": dict(obj.get("asset_validation") or {}) if isinstance(obj.get("asset_validation"), Mapping) else {},
            }
            for obj in objects
            if obj.get("id") is not None
        ],
    }


def build_manifest_layout_inventory(
    *,
    root: Path,
    scene_id: str,
    assets_prefix: str,
    layout_prefix: str,
    seg_prefix: str,
    textgen_payload: Mapping[str, Any],
    source_request: Mapping[str, Any],
) -> Dict[str, Any]:
    """Materialize canonical artifacts and compatibility markers from textgen payload."""

    objects = [dict(obj) for obj in (textgen_payload.get("objects") or []) if isinstance(obj, Mapping)]
    constraints = source_request.get("constraints") if isinstance(source_request.get("constraints"), Mapping) else {}
    # Prefer explicit request room_type, but fall back to the textgen payload / prompt-diversity archetype.
    # This keeps downstream asset retrieval + manifest environment_type aligned with what was generated.
    raw_room_type = (
        str(constraints.get("room_type") or "").strip()
        or str(textgen_payload.get("room_type") or "").strip()
        or str(((constraints.get("prompt_diversity") or {}).get("dimensions") or {}).get("archetype") or "").strip()
    )
    if not raw_room_type:
        raw_room_type = "generic_room"
    try:
        from .generator import _canonicalize_room_type  # local import to avoid incidental cycles

        room_type = _canonicalize_room_type(raw_room_type)
    except Exception:
        room_type = raw_room_type.lower().replace(" ", "_").replace("-", "_")
    retrieval_report: Dict[str, Any] = {}

    provenance_assets = materialize_placeholder_assets(
        root=root,
        scene_id=scene_id,
        assets_prefix=assets_prefix,
        objects=objects,
        room_type=room_type,
        retrieval_report=retrieval_report,
    )

    quality_tier = str(textgen_payload.get("quality_tier") or "standard")
    provider_used = str(textgen_payload.get("provider_used") or "openai")
    seed = int(textgen_payload.get("seed") or 1)
    text_backend = str(textgen_payload.get("text_backend") or "internal").strip().lower() or "internal"
    backend_payload = textgen_payload.get("backend")
    backend_runs_raw = backend_payload.get("backends") if isinstance(backend_payload, Mapping) else None
    backend_runs = (
        [dict(item) for item in backend_runs_raw if isinstance(item, Mapping)]
        if isinstance(backend_runs_raw, list)
        else []
    )
    layout_plan = textgen_payload.get("layout_plan") if isinstance(textgen_payload.get("layout_plan"), Mapping) else None

    manifest = _build_manifest(
        scene_id=scene_id,
        room_type=room_type,
        objects=objects,
        assets_prefix=assets_prefix,
        source_request=source_request,
        layout_plan=layout_plan,
        provider_used=provider_used,
        quality_tier=quality_tier,
        seed=seed,
        provenance_assets=provenance_assets,
        text_backend=text_backend,
        backend_runs=backend_runs,
    )
    layout = _build_layout(
        scene_id=scene_id,
        objects=objects,
        layout_plan=layout_plan,
    )
    inventory = _build_inventory(scene_id=scene_id, objects=objects)

    assets_root = root / assets_prefix
    layout_root = root / layout_prefix
    seg_root = root / seg_prefix

    assets_root.mkdir(parents=True, exist_ok=True)
    layout_root.mkdir(parents=True, exist_ok=True)
    seg_root.mkdir(parents=True, exist_ok=True)

    manifest_path = assets_root / "scene_manifest.json"
    layout_path = layout_root / "scene_layout_scaled.json"
    inventory_path = seg_root / "inventory.json"

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")
    inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    completion_marker = assets_root / ".regen3d_complete"
    completion_marker.write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "status": "completed",
                "source": "text_scene_adapter",
                "objects_count": len(objects),
                "quality_tier": quality_tier,
                "provider": provider_used,
                "seed": seed,
                "marker_type": "stage1_complete",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    adapter_complete = assets_root / ".text_adapter_complete"
    adapter_complete.write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "status": "completed",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rollout_status = {}
    try:
        rollout_status = update_rollout_state(root=root, decisions=list(retrieval_report.get("decisions") or []))
    except Exception as rollout_exc:  # pragma: no cover - fail-open telemetry
        rollout_status = {
            "error": f"rollout_update_failed:{rollout_exc}",
        }
    retrieval_report["rollout"] = rollout_status
    retrieval_report_path = _write_asset_retrieval_report(root=root, assets_prefix=assets_prefix, report=retrieval_report)

    catalog_status = _publish_assets_to_catalog(
        scene_id=scene_id,
        assets_prefix=assets_prefix,
        objects=objects,
        provenance_assets=provenance_assets,
        source_request=source_request,
        quality_tier=quality_tier,
        provider_used=provider_used,
        seed=seed,
    )
    embedding_status = _enqueue_asset_embedding_request(
        root=root,
        scene_id=scene_id,
        assets_prefix=assets_prefix,
        source_request=source_request,
        catalog_assets=list(catalog_status.get("assets") or []),
    )
    replication_status = _enqueue_replication_request(
        root=root,
        scene_id=scene_id,
        assets_prefix=assets_prefix,
        provenance_assets=provenance_assets,
    )

    return {
        "scene_id": scene_id,
        "objects_count": len(objects),
        "manifest_path": str(manifest_path),
        "layout_path": str(layout_path),
        "inventory_path": str(inventory_path),
        "completion_marker": str(completion_marker),
        "asset_retrieval_report": retrieval_report_path,
        "catalog": catalog_status,
        "embedding": embedding_status,
        "replication": replication_status,
    }
