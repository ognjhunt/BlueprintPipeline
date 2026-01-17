import importlib.util
import json
import logging
import os
import sys
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.scene_manifest.loader import load_manifest_or_scene_assets
from monitoring.alerting import send_alert
from tools.validation.entrypoint_checks import (
    validate_required_env_vars,
    validate_scene_manifest,
)
from tools.workflow.failure_markers import FailureMarkerWriter

# GAP-PHYSICS-011 FIX: Import physics profile selector
try:
    from blueprint_sim.recipe_compiler.physics_profiles_selector import create_profile_selector
    HAVE_PROFILE_SELECTOR = True
except ImportError:
    HAVE_PROFILE_SELECTOR = False
    print("[SIMREADY] WARNING: Physics profile selector unavailable", file=sys.stderr)

import numpy as np
from PIL import Image  # type: ignore
from tools.asset_catalog import AssetCatalogClient

# Deterministic material-based physics hints
_material_transfer_spec = importlib.util.find_spec("tools.material_transfer.material_transfer")
if _material_transfer_spec:
    from tools.material_transfer.material_transfer import (  # type: ignore
        infer_material_type,
        MATERIAL_PHYSICS,
        MaterialType,
    )
    HAVE_MATERIAL_TRANSFER = True
else:  # pragma: no cover
    HAVE_MATERIAL_TRANSFER = False
    infer_material_type = None
    MATERIAL_PHYSICS = {}
    MaterialType = None
    print("[SIMREADY] WARNING: Material transfer module unavailable; using generic physics", file=sys.stderr)

# Secret Manager for secure API key storage
try:
    from tools.secrets import get_secret_or_env, SecretIds
    HAVE_SECRET_MANAGER = True
except ImportError:  # pragma: no cover
    HAVE_SECRET_MANAGER = False
    get_secret_or_env = None
    SecretIds = None
    print("[SIMREADY] WARNING: Secret Manager not available - using env vars only", file=sys.stderr)

SECRET_ID_GEMINI = SecretIds.GEMINI_API_KEY if SecretIds else "gemini-api-key"
SECRET_ID_OPENAI = SecretIds.OPENAI_API_KEY if SecretIds else "openai-api-key"

try:
    # Google GenAI SDK for Gemini 3.x
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None

# Unified LLM client supporting Gemini + OpenAI
try:
    from tools.llm_client import create_llm_client, LLMProvider, LLMResponse
    HAVE_LLM_CLIENT = True
except ImportError:  # pragma: no cover
    HAVE_LLM_CLIENT = False
    create_llm_client = None
    LLMProvider = None
    LLMResponse = None

# Parallel processing utilities
try:
    from tools.performance import process_parallel_threaded, ParallelProcessor
    HAVE_PARALLEL_PROCESSING = True
except ImportError:  # pragma: no cover
    HAVE_PARALLEL_PROCESSING = False
    print("[SIMREADY] WARNING: Parallel processing not available - will use sequential processing", file=sys.stderr)

GCS_ROOT = Path("/mnt/gcs")
JOB_NAME = "simready-job"


# ---------- Small helpers ----------

def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_path_join(root: Path, rel: str) -> Path:
    rel_path = rel.lstrip("/")
    return root / rel_path


def _get_env_value(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return default
    return value


@lru_cache(maxsize=8)
def _load_secret_value(
    secret_id: str,
    env_var: str,
    *,
    allow_env_fallback: bool = True,
) -> Optional[str]:
    env_value = os.environ.get(env_var)
    if HAVE_SECRET_MANAGER and get_secret_or_env is not None:
        try:
            value = get_secret_or_env(
                secret_id,
                env_var=env_var,
                fallback_to_env=allow_env_fallback,
            )
            if value and not env_value:
                logger.info(
                    "[SIMREADY] Using Secret Manager for %s credentials.",
                    env_var,
                )
            return value
        except Exception as exc:
            if not allow_env_fallback:
                raise
            logger.warning(
                "[SIMREADY] Failed to fetch secret '%s'; falling back to env var '%s': %s",
                secret_id,
                env_var,
                exc,
            )
    return env_value if allow_env_fallback else None


def _get_secret_value(
    secret_id: str,
    env_var: str,
    *,
    purpose: str,
    production_mode: bool = False,
    required: bool = False,
    log_missing: bool = True,
) -> Optional[str]:
    if production_mode:
        if not HAVE_SECRET_MANAGER or get_secret_or_env is None:
            message = (
                f"[SIMREADY] Secret Manager is required in production for {purpose}. "
                f"Env var '{env_var}' is not allowed."
            )
            logger.error(message)
            raise RuntimeError(message)
        try:
            value = _load_secret_value(
                secret_id,
                env_var,
                allow_env_fallback=False,
            )
        except Exception as exc:
            message = (
                f"[SIMREADY] Missing Secret Manager secret '{secret_id}' for {purpose}. "
                f"Env var '{env_var}' is not allowed in production."
            )
            logger.error(message)
            raise RuntimeError(message) from exc
    else:
        value = _load_secret_value(secret_id, env_var)
    if not value and log_missing:
        message = (
            f"[SIMREADY] Missing {purpose} credentials. "
            f"Set Secret Manager ID '{secret_id}' or env var '{env_var}'."
        )
        if required:
            logger.error(message)
        else:
            logger.warning(message)
    return value


def choose_reference_image_path(root: Path, obj: Dict[str, Any]) -> Optional[Path]:
    """
    Pick the best available reference image for an object.

    Preference (same as other jobs): preferred_view -> multiview/view_0.png -> crop.
    """

    candidates: List[Path] = []

    preferred_rel = obj.get("preferred_view")
    mv_rel = obj.get("multiview_dir")
    crop_rel = obj.get("crop_path")

    if isinstance(preferred_rel, str):
        candidates.append(safe_path_join(root, preferred_rel))
    if isinstance(mv_rel, str):
        candidates.append(safe_path_join(root, mv_rel) / "view_0.png")
    if isinstance(crop_rel, str):
        candidates.append(safe_path_join(root, crop_rel))

    for p in candidates:
        if p.is_file():
            return p

    return None


def load_image_for_gemini(image_path: Path) -> Optional["Image.Image"]:
    """
    Load an RGB PIL image for Gemini input.
    """

    try:
        return Image.open(str(image_path)).convert("RGB")
    except Exception as exc:  # pragma: no cover - image decoding errors
        print(f"[SIMREADY] WARNING: failed to load reference image {image_path}: {exc}", file=sys.stderr)
        return None


def _as_float_list(v: Any, n: int) -> Optional[List[float]]:
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    out: List[float] = []
    for x in v:
        try:
            out.append(float(x))
        except Exception:
            return None
    return out


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))


def load_object_metadata(
    root: Path,
    obj: Dict[str, Any],
    assets_prefix: str,
    catalog_client: Optional[AssetCatalogClient] = None,
) -> Optional[dict]:
    """
    Same lookup strategy as usd-assembly build_scene_usd:
    1. catalog lookup by asset_id or asset_path
    2. explicit metadata_path
    3. next to asset_path
    4. fallback: assets/static/obj_{id}/metadata.json
    """
    if catalog_client is not None:
        logical_id = obj.get("logical_asset_id") or obj.get("logical_id")
        try:
            catalog_meta = catalog_client.lookup_metadata(
                asset_id=obj.get("id"),
                asset_path=obj.get("asset_path"),
                logical_id=logical_id,
            )
            if catalog_meta:
                return catalog_meta
        except Exception as exc:  # pragma: no cover - network errors
            print(f"[SIMREADY] WARNING: catalog lookup failed: {exc}", file=sys.stderr)

    metadata_rel = obj.get("metadata_path")
    if metadata_rel:
        candidate = safe_path_join(root, metadata_rel)
        if candidate.is_file():
            return json.loads(candidate.read_text())

    asset_path = obj.get("asset_path")
    if asset_path:
        asset_dir = safe_path_join(root, asset_path).parent
        candidate = asset_dir / "metadata.json"
        if candidate.is_file():
            return json.loads(candidate.read_text())

    oid = obj.get("id")
    if oid is not None:
        static_dir = safe_path_join(root, f"{assets_prefix}/static/obj_{oid}")
        candidate = static_dir / "metadata.json"
        if candidate.is_file():
            return json.loads(candidate.read_text())

    return None


def extract_mesh_bounds_from_metadata(metadata: Optional[dict]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Try to read an axis-aligned bounding box from metadata.
    Returns: (size_xyz_m, center_xyz_m)
    """
    if not metadata:
        return None, None

    mesh_bounds = metadata.get("mesh_bounds") or {}
    export_bounds = mesh_bounds.get("export") or mesh_bounds.get("bounds") or mesh_bounds

    size = _as_float_list(export_bounds.get("size"), 3)

    center = _as_float_list(export_bounds.get("center"), 3)

    bmin = _as_float_list(export_bounds.get("min") or export_bounds.get("minimum"), 3)
    bmax = _as_float_list(export_bounds.get("max") or export_bounds.get("maximum"), 3)

    if size is None and bmin is not None and bmax is not None:
        size = [float(bmax[i] - bmin[i]) for i in range(3)]

    if center is None and bmin is not None and bmax is not None:
        center = [float((bmin[i] + bmax[i]) * 0.5) for i in range(3)]

    # Sanitize obvious nonsense
    if size is not None:
        size = [max(float(s), 1e-3) for s in size]

    return size, center


def extract_obb_bounds_from_obj(obj: Dict[str, Any]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Best-effort OBB fallback.
    Many pipelines store OBB "extents" as half-extents. We assume half-extents and multiply by 2.
    Returns: (size_xyz_m, center_xyz_m)
    """
    obb = obj.get("obb") or {}
    extents = _as_float_list(obb.get("extents") or obb.get("half_extents"), 3)
    center = _as_float_list(obb.get("center"), 3)

    size = None
    if extents is not None:
        size = [max(2.0 * float(e), 1e-3) for e in extents]

    return size, center


def compute_bounds(
    obj: Dict[str, Any],
    metadata: Optional[dict],
    gemini_estimated_size: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compute a usable axis-aligned bounds estimate in meters:
      - prefer metadata mesh_bounds
      - fallback to obj.obb
      - fallback to gemini_estimated_size (if provided)
      - else default 10cm cube
    """
    size, center = extract_mesh_bounds_from_metadata(metadata)
    source = "metadata"

    if size is None:
        size2, center2 = extract_obb_bounds_from_obj(obj)
        if size2 is not None:
            size, center = size2, center2
            source = "obb"

    if size is None and gemini_estimated_size is not None:
        size = gemini_estimated_size
        source = "gemini_estimated"

    if size is None:
        size = [0.10, 0.10, 0.10]  # 10cm cube fallback
        source = "default"

    if center is None:
        center = [0.0, 0.0, 0.0]

    sx, sy, sz = [max(float(s), 1e-3) for s in size]
    volume = float(sx * sy * sz)

    return {
        "size_m": [sx, sy, sz],
        "center_m": [float(center[0]), float(center[1]), float(center[2])],
        "volume_m3": volume,
        "source": source,
    }


def padded_size(size_m: List[float]) -> List[float]:
    """
    Apply a small collision padding so fast contacts are less likely to tunnel.
    Tunable via env vars:
      SIMREADY_COLLIDER_PAD_RATIO (default 0.01 == 1%)
      SIMREADY_COLLIDER_PAD_MIN_M (default 0.001 == 1mm)
      SIMREADY_COLLIDER_PAD_MAX_M (default 0.02 == 2cm)
    """
    ratio = float(_get_env_value("SIMREADY_COLLIDER_PAD_RATIO", "0.01"))
    pad_min = float(_get_env_value("SIMREADY_COLLIDER_PAD_MIN_M", "0.001"))
    pad_max = float(_get_env_value("SIMREADY_COLLIDER_PAD_MAX_M", "0.02"))

    mx = max(size_m)
    pad = _clamp(mx * ratio, pad_min, pad_max)
    return [max(float(s) + 2.0 * pad, 1e-3) for s in size_m]


# ---------- Generic fallback (used only when Gemini fails) ----------

GENERIC_FALLBACK: Dict[str, Any] = {
    # Effective bulk density (includes air gaps); "generic solid-ish household thing"
    "bulk_density_kg_per_m3": 600.0,
    # Very wide range: 1g to 1000kg
    "mass_range_kg": (0.001, 1000.0),
    "static_friction": 0.6,
    "dynamic_friction": 0.5,
    "restitution": 0.1,
    "dynamic": True,
    "collision_shape": "box",  # box/sphere/capsule
    "material_name": "generic",
    # New robotics-focused properties
    "semantic_class": "object",
    "center_of_mass_offset": [0.0, 0.0, 0.0],  # offset from geometric center
    "graspable": True,
    "grasp_regions": [],  # list of {"position": [x,y,z], "type": "pinch|power|wrap", "width_m": float}
    "contact_offset_m": 0.005,  # PhysX contact detection margin
    "rest_offset_m": 0.001,  # PhysX rest separation
    "surface_roughness": 0.5,  # 0=smooth, 1=rough (for tactile simulation)
}

VALID_SEMANTIC_CLASSES = {
    "object", "container", "tool", "food", "electronics", "furniture",
    "toy", "clothing", "book", "kitchenware", "bottle", "can", "box",
    "bag", "utensil", "appliance", "decoration", "plant", "sporting_goods",
    "office_supply", "hygiene", "cleaning", "storage", "hardware",
}


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"1", "true", "yes", "y"}:
            return True
        if lower in {"0", "false", "no", "n"}:
            return False
    return None


def _collect_material_candidates(obj: Dict[str, Any], metadata: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    for key in (
        "material_name",
        "material",
        "surface_material",
        "dominant_material",
        "materialType",
    ):
        value = metadata.get(key) or obj.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    for key in ("category", "class_name", "label", "name", "description"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    tags = obj.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.strip():
                candidates.append(tag.strip())

    return candidates


def _infer_material_profile(obj: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[Optional[str], Optional[dict]]:
    candidates = _collect_material_candidates(obj, metadata)
    if not candidates:
        return None, None

    if HAVE_MATERIAL_TRANSFER and infer_material_type is not None and MaterialType is not None:
        for candidate in candidates:
            material_type = infer_material_type(candidate)
            if material_type in MATERIAL_PHYSICS:
                return candidate, MATERIAL_PHYSICS[material_type]

    # Fallback keyword lookup
    lowered = " ".join(candidates).lower()
    keyword_map = {
        "metal": "metal",
        "steel": "metal",
        "aluminum": "metal",
        "plastic": "plastic",
        "rubber": "rubber",
        "wood": "wood",
        "glass": "glass",
        "ceramic": "ceramic",
        "fabric": "fabric",
        "cloth": "fabric",
        "leather": "leather",
        "paper": "paper",
        "cardboard": "paper",
        "stone": "stone",
        "concrete": "concrete",
    }

    for key, mat_name in keyword_map.items():
        if key in lowered:
            props = MATERIAL_PHYSICS.get(MaterialType(mat_name)) if MaterialType else None
            if props:
                return mat_name, props

    return None, None


def _extract_metadata_physics(metadata: Dict[str, Any]) -> Dict[str, Any]:
    if not metadata:
        return {}
    for key in ("physics", "simready", "simready_metadata"):
        value = metadata.get(key)
        if isinstance(value, dict):
            return value
    return {}


# ---------- Sim2Real Distribution Parameters ----------
# These define uncertainty ranges for physics properties to enable
# domain randomization during training, improving sim2real transfer.

def compute_physics_distributions(physics: Dict[str, Any], bounds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute distribution ranges for physics properties to enable domain randomization.

    Returns a dict with *_range fields for key physics properties:
    - mass_kg_range: [min, max] for mass randomization
    - friction_static_range: [min, max] for static friction
    - friction_dynamic_range: [min, max] for dynamic friction
    - restitution_range: [min, max] for bounciness
    - center_of_mass_noise: [x, y, z] noise magnitude per axis
    - contact_offset_range: [min, max] for PhysX contact offset

    These ranges are designed for realistic domain randomization that
    improves policy robustness without creating implausible physics.
    """
    distributions: Dict[str, Any] = {}

    # Mass distribution: ±20% of estimated mass (common manufacturing variance)
    mass = float(physics.get("mass_kg", 1.0))
    mass_variance = 0.20
    distributions["mass_kg_range"] = [
        max(0.001, mass * (1.0 - mass_variance)),
        mass * (1.0 + mass_variance)
    ]

    # Friction distributions: ±15% variance (surface condition variance)
    static_f = float(physics.get("static_friction", 0.6))
    dynamic_f = float(physics.get("dynamic_friction", 0.5))
    friction_variance = 0.15

    distributions["friction_static_range"] = [
        max(0.1, static_f * (1.0 - friction_variance)),
        min(1.5, static_f * (1.0 + friction_variance))
    ]
    distributions["friction_dynamic_range"] = [
        max(0.05, dynamic_f * (1.0 - friction_variance)),
        min(1.2, dynamic_f * (1.0 + friction_variance))
    ]

    # Restitution distribution: ±30% (material property variance)
    restitution = float(physics.get("restitution", 0.1))
    restitution_variance = 0.30
    distributions["restitution_range"] = [
        max(0.0, restitution * (1.0 - restitution_variance)),
        min(1.0, restitution * (1.0 + restitution_variance))
    ]

    # Center of mass noise: proportional to object size (5% of each dimension)
    size_m = list(bounds.get("size_m") or [0.1, 0.1, 0.1])
    com_noise_ratio = 0.05
    distributions["center_of_mass_noise"] = [
        size_m[0] * com_noise_ratio,
        size_m[1] * com_noise_ratio,
        size_m[2] * com_noise_ratio
    ]

    # Contact offset range: ±50% (solver sensitivity variance)
    contact_offset = float(physics.get("contact_offset_m", 0.005))
    distributions["contact_offset_range"] = [
        max(0.001, contact_offset * 0.5),
        min(0.02, contact_offset * 1.5)
    ]

    # Surface roughness range: ±20%
    roughness = float(physics.get("surface_roughness", 0.5))
    roughness_variance = 0.20
    distributions["surface_roughness_range"] = [
        max(0.0, roughness * (1.0 - roughness_variance)),
        min(1.0, roughness * (1.0 + roughness_variance))
    ]

    return distributions


# ---------- Physics config (default + Gemini) ----------

def estimate_default_physics(obj: Dict[str, Any], bounds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Baseline physics estimate used only when Gemini is unavailable or fails.
    """
    volume = float(bounds.get("volume_m3") or 0.0)
    rho = float(GENERIC_FALLBACK["bulk_density_kg_per_m3"])

    if volume > 0.0:
        mass_est = volume * rho
        mass = float(np.clip(mass_est, GENERIC_FALLBACK["mass_range_kg"][0], GENERIC_FALLBACK["mass_range_kg"][1]))
    else:
        mass = 1.0

    notes = f"fallback estimate; bulk_density={rho:.1f} kg/m^3; volume={volume:.6f} m^3; bounds_source={bounds.get('source')}"

    return {
        "dynamic": bool(GENERIC_FALLBACK["dynamic"]),
        "collision_shape": str(GENERIC_FALLBACK["collision_shape"]),
        "mass_kg": float(mass),
        "bulk_density_kg_per_m3": float(rho),
        "static_friction": float(GENERIC_FALLBACK["static_friction"]),
        "dynamic_friction": float(GENERIC_FALLBACK["dynamic_friction"]),
        "restitution": float(GENERIC_FALLBACK["restitution"]),
        "material_name": str(GENERIC_FALLBACK["material_name"]),
        "notes": notes,
        # Robotics-focused properties
        "semantic_class": str(GENERIC_FALLBACK["semantic_class"]),
        "center_of_mass_offset": list(GENERIC_FALLBACK["center_of_mass_offset"]),
        "graspable": bool(GENERIC_FALLBACK["graspable"]),
        "grasp_regions": list(GENERIC_FALLBACK["grasp_regions"]),
        "contact_offset_m": float(GENERIC_FALLBACK["contact_offset_m"]),
        "rest_offset_m": float(GENERIC_FALLBACK["rest_offset_m"]),
        "surface_roughness": float(GENERIC_FALLBACK["surface_roughness"]),
        "estimation_source": "heuristic_default",
    }


def estimate_deterministic_physics(
    obj: Dict[str, Any],
    bounds: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deterministic physics estimate that uses metadata and material priors.

    This path is LLM-free and can be enabled in production for predictable results.
    """
    metadata = metadata or {}
    volume = float(bounds.get("volume_m3") or 0.0)
    base = estimate_default_physics(obj, bounds)
    base["estimation_source"] = "deterministic_default"

    notes: List[str] = ["deterministic"]

    metadata_physics = _extract_metadata_physics(metadata)
    has_metadata_overrides = False
    friction_override = False
    restitution_override = False
    density_override_flag = False
    mass_override = None

    # Apply explicit physics overrides from metadata if present.
    if metadata_physics:
        mass_override = _coerce_float(metadata_physics.get("mass_kg") or metadata_physics.get("mass"))
        density_override = _coerce_float(
            metadata_physics.get("bulk_density_kg_per_m3")
            or metadata_physics.get("density_kg_per_m3")
            or metadata_physics.get("density")
        )
        if mass_override is not None:
            base["mass_kg"] = _clamp(mass_override, 0.001, 1000.0)
            has_metadata_overrides = True
        if density_override is not None:
            base["bulk_density_kg_per_m3"] = _clamp(density_override, 0.5, 20000.0)
            if volume > 0.0 and mass_override is None:
                base["mass_kg"] = _clamp(base["bulk_density_kg_per_m3"] * volume, 0.001, 1000.0)
            has_metadata_overrides = True
            density_override_flag = True

        static_override = _coerce_float(metadata_physics.get("static_friction"))
        dynamic_override = _coerce_float(metadata_physics.get("dynamic_friction"))
        restitution_override = _coerce_float(metadata_physics.get("restitution"))
        if static_override is not None:
            base["static_friction"] = _clamp(static_override, 0.0, 2.0)
            has_metadata_overrides = True
            friction_override = True
        if dynamic_override is not None:
            base["dynamic_friction"] = _clamp(dynamic_override, 0.0, 2.0)
            has_metadata_overrides = True
            friction_override = True
        if restitution_override is not None:
            base["restitution"] = _clamp(restitution_override, 0.0, 1.0)
            has_metadata_overrides = True
            restitution_override = True

        material_override = metadata_physics.get("material_name") or metadata_physics.get("material")
        if isinstance(material_override, str) and material_override.strip():
            base["material_name"] = material_override.strip()
            has_metadata_overrides = True

        collision_override = metadata_physics.get("collision_shape") or metadata_physics.get("collision_approximation")
        if isinstance(collision_override, str) and collision_override.strip():
            collision_shape = collision_override.strip().lower()
            if collision_shape in {"box", "sphere", "capsule", "convex_hull", "convex_decomposition"}:
                base["collision_shape"] = collision_shape
                has_metadata_overrides = True

        semantic_override = metadata_physics.get("semantic_class")
        if isinstance(semantic_override, str) and semantic_override.strip():
            semantic = semantic_override.strip().lower().replace(" ", "_")
            if semantic in VALID_SEMANTIC_CLASSES:
                base["semantic_class"] = semantic
                has_metadata_overrides = True

        com_override = metadata_physics.get("center_of_mass_offset")
        if _as_float_list(com_override, 3):
            base["center_of_mass_offset"] = [float(v) for v in com_override]
            has_metadata_overrides = True

        graspable_override = _coerce_bool(metadata_physics.get("graspable"))
        if graspable_override is not None:
            base["graspable"] = graspable_override
            has_metadata_overrides = True

        if "grasp_regions" in metadata_physics and isinstance(metadata_physics["grasp_regions"], list):
            base["grasp_regions"] = list(metadata_physics["grasp_regions"])
            has_metadata_overrides = True

        roughness_override = _coerce_float(metadata_physics.get("surface_roughness"))
        if roughness_override is not None:
            base["surface_roughness"] = _clamp(roughness_override, 0.0, 1.0)
            has_metadata_overrides = True

    material_name, material_props = _infer_material_profile(obj, metadata)
    if material_props:
        if material_name:
            base["material_name"] = material_name
        if not friction_override:
            base["static_friction"] = float(material_props.get("friction_static", base["static_friction"]))
            base["dynamic_friction"] = float(material_props.get("friction_dynamic", base["dynamic_friction"]))
        if not restitution_override:
            base["restitution"] = float(material_props.get("restitution", base["restitution"]))
        if volume > 0.0 and not density_override_flag and mass_override is None:
            density = float(material_props.get("density", base["bulk_density_kg_per_m3"]))
            base["bulk_density_kg_per_m3"] = _clamp(density, 0.5, 20000.0)
            base["mass_kg"] = _clamp(base["bulk_density_kg_per_m3"] * volume, 0.001, 1000.0)
        if base["estimation_source"] == "deterministic_default":
            base["estimation_source"] = "deterministic_material"
            notes.append(f"material={base['material_name']}")

    if has_metadata_overrides:
        base["estimation_source"] = "deterministic_metadata"
        notes.append("metadata_overrides")

    if mass_override is not None and volume > 0.0 and not density_override_flag:
        base["bulk_density_kg_per_m3"] = _clamp(mass_override / volume, 0.5, 20000.0)

    # Enforce friction constraint
    if base["dynamic_friction"] > base["static_friction"]:
        base["dynamic_friction"] = max(0.0, base["static_friction"] - 0.05)

    if base["mass_kg"] > 25.0:
        base["graspable"] = False

    notes.append(f"volume={volume:.6f} m^3")
    base["notes"] = " | ".join([note for note in notes if note])

    return base


def _validate_non_llm_physics_quality(
    physics_cfg: Dict[str, Any],
    bounds: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Validate deterministic/heuristic physics for minimum simulation fidelity.

    Returns (passed, reasons). This is intended for non-LLM paths.
    """
    reasons: List[str] = []

    estimation_source = str(physics_cfg.get("estimation_source") or "")
    if estimation_source in {"deterministic_default", "heuristic_default"}:
        reasons.append("default_estimation_source")

    mass = _coerce_float(physics_cfg.get("mass_kg"))
    if mass is None or mass <= 0.0:
        reasons.append("mass_missing_or_nonpositive")
    elif mass > 2000.0:
        reasons.append("mass_too_large")

    volume = float(bounds.get("volume_m3") or 0.0)
    if mass is not None and volume > 0.0:
        density = mass / volume
        if density < 50.0 or density > 20000.0:
            reasons.append("density_out_of_range")

    static_f = _coerce_float(physics_cfg.get("static_friction"))
    dynamic_f = _coerce_float(physics_cfg.get("dynamic_friction"))
    if static_f is None or dynamic_f is None:
        reasons.append("friction_missing")
    else:
        if not (0.0 <= static_f <= 2.0) or not (0.0 <= dynamic_f <= 2.0):
            reasons.append("friction_out_of_range")
        if dynamic_f > static_f:
            reasons.append("dynamic_friction_gt_static")

    restitution = _coerce_float(physics_cfg.get("restitution"))
    if restitution is None or restitution < 0.0 or restitution > 1.0:
        reasons.append("restitution_out_of_range")

    collision_shape = str(physics_cfg.get("collision_shape") or "").lower()
    valid_shapes = {"box", "sphere", "capsule", "convex_hull", "convex_decomposition"}
    if collision_shape and collision_shape not in valid_shapes:
        reasons.append("collision_shape_invalid")

    return len(reasons) == 0, reasons


def _compact_obj_description(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep prompt context rich but bounded (avoid dumping huge dicts).
    """
    keep: Dict[str, Any] = {}
    priority_keys = [
        "id",
        "name",
        "label",
        "type",
        "class_name",
        "category",
        "pipeline",
        "asset_path",
        "metadata_path",
        "source",
        "tags",
        "description",
        "prompt",
        "caption",
    ]

    for k in priority_keys:
        if k in obj:
            v = obj.get(k)
            if isinstance(v, (str, int, float, bool)):
                keep[k] = v
            elif isinstance(v, list) and len(v) <= 20:
                # keep short lists of primitives/strings
                if all(isinstance(x, (str, int, float, bool)) for x in v):
                    keep[k] = v

    # Add a small sample of other short strings
    for k, v in obj.items():
        if k in keep or k in {"obb"}:
            continue
        if isinstance(v, str) and 0 < len(v) <= 120:
            keep[k] = v
            if len(keep) >= 20:
                break

    # Keep OBB hint, but not huge matrices
    obb = obj.get("obb")
    if isinstance(obb, dict):
        for kk in ("extents", "center"):
            if kk in obb:
                vv = obb.get(kk)
                if isinstance(vv, (list, tuple)) and len(vv) == 3 and all(isinstance(x, (int, float)) for x in vv):
                    keep[f"obb_{kk}"] = [float(x) for x in vv]

    return keep


def make_gemini_prompt(
    oid: Any,
    obj: Dict[str, Any],
    bounds: Dict[str, Any],
    base_cfg: Dict[str, Any],
    has_image: bool = False,
) -> str:
    """
    Prompt Gemini to estimate realistic physics parameters based on object metadata + size.
    """
    size = bounds.get("size_m")
    center = bounds.get("center_m")
    volume = bounds.get("volume_m3")

    obj_desc = _compact_obj_description(obj)
    obj_desc["id"] = oid

    if has_image:
        image_context = (
            "You must estimate REALISTIC physics parameters for the *specific object* shown in the attached image,"
            " using its appearance as the primary signal and the text metadata below as context."
        )
        primary_guidance = (
            "You are given a reference image of the object. Base your estimates primarily on that photo (shape, material, heft), "
            "and use the metadata below only as supporting context."
        )
    else:
        image_context = (
            "You must estimate REALISTIC physics parameters for the object using the metadata and scale information below."
        )
        primary_guidance = (
            "No photo was provided. Rely on the metadata below plus the size to estimate realistic properties, "
            "but avoid generic placeholders."
        )

    skeleton = {
        "dynamic": True,
        "collision_shape": "box",  # box|sphere|capsule
        "mass_kg": 1.0,  # total mass of this object at this scale
        "bulk_density_kg_per_m3": 600.0,  # effective density incl. voids; should be consistent with mass_kg/volume
        "static_friction": 0.6,
        "dynamic_friction": 0.5,
        "restitution": 0.1,
        "material_name": "generic",
        "notes": "",
        # Robotics-focused properties
        "semantic_class": "object",  # e.g. container, tool, food, electronics, furniture, toy, clothing, book, kitchenware
        "center_of_mass_offset": [0.0, 0.0, 0.0],  # offset from geometric center in meters [x, y, z]
        "graspable": True,  # can a robot gripper pick this up?
        "grasp_regions": [],  # list of grasp points: [{"position": [x,y,z], "type": "pinch|power|wrap", "width_m": 0.05}]
        "surface_roughness": 0.5,  # 0.0=glass-smooth, 1.0=sandpaper-rough (affects tactile sensing)
    }

    prompt = f"""
You are configuring physics for USD assets to be simulated in NVIDIA Isaac Sim (PhysX / USD Physics).
These assets will be used by ROBOTICS COMPANIES for manipulation training, so accuracy matters.

{image_context}

{primary_guidance}

Important implementation notes (affect what "realistic" means here):
- The sim uses meters and kilograms.
- We will generate an *analytic collision proxy* (box/sphere/capsule) from the object's bounds.
- In USD Physics, mass and density can both exist, but *explicit mass overrides density*.
  Provide values that are consistent with the provided volume (when available).

### Object bounds (meters)
- size_m (AABB): {json.dumps(size)}
- center_m (AABB center): {json.dumps(center)}
- approx_volume_m3: {json.dumps(volume)}
- bounds_source: {json.dumps(bounds.get("source"))}

### Guidance (use as reality checks, not as constraints)
- Typical solid material densities (kg/m^3): plastics ~900-1400, wood ~400-900, glass/ceramic ~2000-3000,
  aluminum ~2700, steel ~7800. Note: many consumer objects are hollow, so effective/bulk density can be much lower.
- Static friction is usually >= dynamic friction. Most household objects have low restitution (~0.0-0.2).
- Collision shape: choose "box" for blocky/general shapes, "sphere" for near-spherical, "capsule" for elongated.

### ROBOTICS-SPECIFIC GUIDANCE (critical for manipulation tasks):

1. **semantic_class**: Choose the most specific applicable category:
   - container, tool, food, electronics, furniture, toy, clothing, book, kitchenware,
   - bottle, can, box, bag, utensil, appliance, decoration, plant, sporting_goods, office_supply

2. **center_of_mass_offset**: Offset from geometric center in meters [x, y, z].
   - For uniform objects: [0, 0, 0]
   - For objects heavier at bottom (e.g., bottles, vases): negative Y offset like [0, -0.02, 0]
   - For objects with asymmetric mass (e.g., hammer): offset toward heavy end
   - Consider: filled containers have CoM based on contents, not shell

3. **graspable**: Can a parallel-jaw or suction gripper pick this up?
   - False for: very large furniture, fixtures, things bolted down
   - True for: most household objects under ~20kg

4. **grasp_regions**: Suggest 1-3 good grasp locations relative to object center.
   Each region: {{"position": [x, y, z], "type": "pinch|power|wrap", "width_m": <gripper_opening>}}
   - pinch: precision grip with fingertips (small objects, edges)
   - power: full-hand cylindrical grip (handles, bottles)
   - wrap: enveloping grip (irregular shapes)
   - width_m: required gripper opening in meters (0.02-0.15 typical)
   - For simple objects, one central grasp is fine. For tools/handles, suggest multiple.

5. **surface_roughness**: 0.0 (glass-smooth) to 1.0 (sandpaper-rough)
   - Polished metal/glass: 0.1-0.2
   - Painted/coated plastic: 0.3-0.4
   - Bare wood: 0.5-0.6
   - Textured rubber/fabric: 0.7-0.9

CRITICAL:
- Do NOT return generic placeholder values.
- Make mass_kg plausible for this object *at this scale*.
- If you output both mass_kg and bulk_density_kg_per_m3, they should approximately agree with:
    mass_kg ≈ bulk_density_kg_per_m3 * approx_volume_m3
  (within a few times is fine because bounds include air/voids).
- For grasp_regions, be SPECIFIC about where on this particular object a robot should grip.

Return ONLY valid JSON (no markdown, no comments, no extra text) matching this structure:

{json.dumps(skeleton, indent=2)}

Object info (cropped to relevant fields):

{json.dumps(obj_desc, indent=2)}
"""
    return prompt


def have_gemini() -> bool:
    if genai is None or types is None:
        return False
    return bool(_load_secret_value(SECRET_ID_GEMINI, "GEMINI_API_KEY"))


def have_openai() -> bool:
    """Check if OpenAI is available."""
    return bool(_load_secret_value(SECRET_ID_OPENAI, "OPENAI_API_KEY"))


def have_any_llm() -> bool:
    """Check if any LLM provider is available."""
    return have_gemini() or have_openai()


def get_llm_provider() -> Optional[str]:
    """Get the preferred LLM provider based on environment."""
    provider = (_get_env_value("LLM_PROVIDER", "auto") or "auto").lower()
    if provider == "openai" and have_openai():
        return "openai"
    elif provider == "gemini" and have_gemini():
        return "gemini"
    elif provider == "auto":
        if have_gemini():
            return "gemini"
        elif have_openai():
            return "openai"
    return None


def load_multiview_images_for_gemini(
    root: Path, obj: Dict[str, Any], max_views: int = 4
) -> List["Image.Image"]:
    """
    Load multiple view images from the multiview directory for better dimension estimation.
    Returns a list of PIL images (up to max_views).
    """
    images: List["Image.Image"] = []
    mv_rel = obj.get("multiview_dir")
    if not isinstance(mv_rel, str):
        return images

    mv_dir = safe_path_join(root, mv_rel)
    if not mv_dir.is_dir():
        return images

    # Try to load views 0-3 (front, side, back, top typically)
    for i in range(max_views):
        view_path = mv_dir / f"view_{i}.png"
        if view_path.is_file():
            img = load_image_for_gemini(view_path)
            if img is not None:
                images.append(img)

    return images


def make_dimension_estimation_prompt(obj: Dict[str, Any], has_multiple_views: bool = False) -> str:
    """
    Create a prompt for Gemini to estimate real-world dimensions of an object from images.
    """
    obj_desc = _compact_obj_description(obj)

    if has_multiple_views:
        image_context = (
            "You are provided with MULTIPLE VIEWS of the same 3D object. "
            "Use all views together to estimate the real-world dimensions more accurately. "
            "The views typically include front, side, and other angles."
        )
    else:
        image_context = (
            "You are provided with a single image of a 3D object. "
            "Estimate its real-world dimensions based on visual appearance."
        )

    prompt = f"""
You are an expert at estimating real-world dimensions of objects from images.

{image_context}

Your task is to estimate the REAL-WORLD physical dimensions (width, height, depth) of the object in METERS.

Important guidelines:
- Think about what this object actually is and what its typical real-world size would be.
- Consider common reference objects: a book is typically 0.15-0.30m tall, a chair seat is ~0.45m high,
  a coffee mug is ~0.10m tall, a dining table is ~0.75m high and 1.0-2.0m wide.
- For objects like furniture, appliances, or decor items, use your knowledge of typical real-world sizes.
- The dimensions should represent the axis-aligned bounding box of the object.

Object metadata (for context):
{json.dumps(obj_desc, indent=2)}

Return ONLY valid JSON (no markdown, no comments, no extra text) with this exact structure:
{{
    "width_m": <float>,
    "height_m": <float>,
    "depth_m": <float>,
    "confidence": "<low|medium|high>",
    "reasoning": "<brief explanation of how you determined the size>"
}}

Where:
- width_m: extent along X axis (left-right)
- height_m: extent along Y axis (up-down)
- depth_m: extent along Z axis (front-back)
- All values in meters (e.g., 0.30 for 30cm)
"""
    return prompt


def estimate_scale_gemini(
    client: "genai.Client",
    obj: Dict[str, Any],
    reference_image: "Image.Image",
    class_name: Optional[str] = None,
) -> Optional[List[float]]:
    """
    Estimate real-world scale of an object using Gemini vision.

    This is a convenience wrapper around call_gemini_for_dimensions that
    works with a single reference image.

    Args:
        client: Gemini API client
        obj: Object dict with metadata
        reference_image: PIL Image of the object
        class_name: Optional class name override

    Returns:
        [width, height, depth] in meters, or None if estimation fails
    """
    if client is None or reference_image is None:
        return None

    # Add class_name to obj if provided
    obj_copy = dict(obj)
    if class_name:
        obj_copy["class_name"] = class_name

    return call_gemini_for_dimensions(client, obj_copy, [reference_image])


def call_gemini_for_dimensions(
    client: "genai.Client",
    obj: Dict[str, Any],
    images: List["Image.Image"],
) -> Optional[List[float]]:
    """
    Ask Gemini to estimate real-world dimensions of an object from images.
    Returns [width, height, depth] in meters, or None if estimation fails.
    """
    if client is None or not images:
        return None

    prompt = make_dimension_estimation_prompt(obj, has_multiple_views=len(images) > 1)

    try:
        model_name = _get_env_value("GEMINI_MODEL", "gemini-3-pro-preview")
        if model_name.startswith("gemini-1") or model_name.startswith("gemini-2"):
            print(
                f"[SIMREADY] Overriding legacy Gemini model '{model_name}' with gemini-3-pro-preview",
                file=sys.stderr,
            )
            model_name = "gemini-3-pro-preview"

        cfg_kwargs: Dict[str, Any] = {
            "response_mime_type": "application/json",
        }

        # Enable grounding for Gemini 3.x models
        grounding_enabled = (
            _get_env_value("GEMINI_GROUNDING_ENABLED", "true").lower() in {"1", "true", "yes"}
        )
        if model_name.startswith("gemini-3") and grounding_enabled:
            if hasattr(types, "GroundingConfig") and hasattr(types, "GoogleSearch"):
                cfg_kwargs["grounding"] = types.GroundingConfig(
                    google_search=types.GoogleSearch()
                )

        try:
            config = types.GenerateContentConfig(**cfg_kwargs)
        except Exception:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
            )

        # Build content: images first, then prompt
        contents = list(images) + [prompt]

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        raw = _strip_code_fences(response.text or "")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Gemini response was not a JSON object")

        width = float(data.get("width_m", 0))
        height = float(data.get("height_m", 0))
        depth = float(data.get("depth_m", 0))
        confidence = data.get("confidence", "low")
        reasoning = data.get("reasoning", "")

        # Sanity checks: dimensions should be reasonable (1mm to 10m)
        if not (0.001 <= width <= 10.0 and 0.001 <= height <= 10.0 and 0.001 <= depth <= 10.0):
            print(f"[SIMREADY] WARNING: Gemini dimension estimate out of range: {width}x{height}x{depth}m", file=sys.stderr)
            return None

        print(f"[SIMREADY] Gemini estimated dimensions: {width:.3f}x{height:.3f}x{depth:.3f}m (confidence: {confidence})")
        if reasoning:
            print(f"[SIMREADY]   Reasoning: {reasoning[:200]}")

        return [width, height, depth]

    except Exception as e:
        print(f"[SIMREADY] WARNING: Gemini dimension estimation failed: {e}", file=sys.stderr)
        return None


def _strip_code_fences(s: str) -> str:
    s2 = (s or "").strip()
    if s2.startswith("```"):
        # Remove ```json ... ``` wrappers if they appear
        lines = s2.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        return "\n".join(lines).strip()
    return s2


def call_gemini_for_object(
    client: "genai.Client",
    oid: Any,
    obj: Dict[str, Any],
    bounds: Dict[str, Any],
    reference_image: Optional["Image.Image"] = None,
) -> Dict[str, Any]:
    """
    Ask Gemini to estimate realistic physics parameters for the object, then
    post-validate so results are usable for simulation.
    """
    base_cfg = estimate_default_physics(obj, bounds)

    if client is None:
        return base_cfg

    prompt = make_gemini_prompt(oid, obj, bounds, base_cfg, has_image=reference_image is not None)

    try:
        model_name = _get_env_value("GEMINI_MODEL", "gemini-3-pro-preview")

        cfg_kwargs: Dict[str, Any] = {
            "response_mime_type": "application/json",
        }

        # Enable grounding for Gemini 3.x models (default: enabled)
        grounding_enabled = (
            _get_env_value("GEMINI_GROUNDING_ENABLED", "true").lower() in {"1", "true", "yes"}
        )
        if model_name.startswith("gemini-3") and grounding_enabled:
            if hasattr(types, "GroundingConfig") and hasattr(types, "GoogleSearch"):
                cfg_kwargs["grounding"] = types.GroundingConfig(
                    google_search=types.GoogleSearch()
                )

        # Only use thinking_config when the SDK exposes it
        if hasattr(types, "ThinkingConfig"):
            ThinkingConfig = getattr(types, "ThinkingConfig")
            ThinkingLevel = getattr(types, "ThinkingLevel", None)

            if model_name.startswith("gemini-3") and ThinkingLevel is not None:
                cfg_kwargs["thinking_config"] = ThinkingConfig(
                    thinking_level=getattr(ThinkingLevel, "HIGH", "HIGH")
                )

        try:
            config = types.GenerateContentConfig(**cfg_kwargs)
        except Exception:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
            )

        contents = [prompt]
        if reference_image is not None:
            contents = [reference_image, prompt]

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        raw = _strip_code_fences(response.text or "")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Gemini response was not a JSON object")

        merged = dict(base_cfg)

        def _get_float(keys: List[str], default: float) -> float:
            for k in keys:
                if k in data:
                    try:
                        return float(data[k])
                    except Exception:
                        pass
            return float(default)

        def _get_bool(keys: List[str], default: bool) -> bool:
            for k in keys:
                if k in data and isinstance(data[k], bool):
                    return bool(data[k])
            return bool(default)

        def _get_str(keys: List[str], default: str) -> str:
            for k in keys:
                if k in data and isinstance(data[k], str) and data[k].strip():
                    return str(data[k]).strip()
            return str(default)

        # dynamic + collision shape
        merged["dynamic"] = _get_bool(["dynamic"], bool(merged.get("dynamic", True)))

        collision_shape = _get_str(["collision_shape", "collisionShape"], str(merged.get("collision_shape", "box"))).lower()
        # GAP-PHYSICS-010 FIX: Support convex decomposition for complex shapes
        valid_shapes = {"box", "sphere", "capsule", "convex_hull", "convex_decomposition"}
        if collision_shape not in valid_shapes:
            # Keep it safe/stable: fall back to box collider
            collision_shape = "box"
        merged["collision_shape"] = collision_shape

        # restitution
        merged["restitution"] = _clamp(
            _get_float(["restitution", "bounce"], float(merged.get("restitution", 0.1))),
            0.0,
            1.0,
        )

        # friction: prefer explicit static/dynamic, else map from a single friction field
        static_f = _get_float(["static_friction", "staticFriction"], float(merged.get("static_friction", 0.6)))
        dynamic_f = _get_float(["dynamic_friction", "dynamicFriction"], float(merged.get("dynamic_friction", 0.5)))

        if "friction" in data and ("static_friction" not in data and "staticFriction" not in data and
                                  "dynamic_friction" not in data and "dynamicFriction" not in data):
            try:
                f = float(data["friction"])
                static_f = f * 1.1
                dynamic_f = f * 0.9
            except Exception:
                pass

        # GAP-PHYSICS-002 FIX: Physics-consistent friction validation
        static_f = _clamp(static_f, 0.0, 2.0)
        dynamic_f = _clamp(dynamic_f, 0.0, 2.0)

        # Enforce physical constraint: dynamic friction <= static friction
        if dynamic_f > static_f:
            logger.warning(
                f"[SIMREADY] obj_{oid}: Dynamic friction ({dynamic_f:.3f}) > static friction ({static_f:.3f}), "
                "adjusting dynamic friction"
            )
            dynamic_f = max(0.0, static_f - 0.05)

        # Additional validation: both should be non-negative
        if static_f < 0 or dynamic_f < 0:
            logger.warning(f"[SIMREADY] obj_{oid}: Negative friction detected, resetting to defaults")
            static_f = 0.6
            dynamic_f = 0.5

        merged["static_friction"] = static_f
        merged["dynamic_friction"] = dynamic_f

        # material name
        merged["material_name"] = _get_str(["material_name", "material"], str(merged.get("material_name", "generic")))

        # mass + bulk density, cross-checked with bounds volume when available
        volume = float(bounds.get("volume_m3") or 0.0)

        mass_val = _get_float(["mass_kg", "massKg", "mass"], float(merged.get("mass_kg", 1.0)))
        bulk_rho = _get_float(["bulk_density_kg_per_m3", "bulkDensity_kg_per_m3", "density_kg_per_m3", "density"], float(merged.get("bulk_density_kg_per_m3", 600.0)))

        # Keep density sane but wide (effective density can be low for hollow objects)
        bulk_rho = _clamp(bulk_rho, 0.5, 20000.0)

        # Either value could be missing/garbled; choose a stable mass:
        have_mass = "mass_kg" in data or "massKg" in data or "mass" in data
        have_rho = any(k in data for k in ["bulk_density_kg_per_m3", "bulkDensity_kg_per_m3", "density_kg_per_m3", "density"])

        mass_from_density = None
        if volume > 0.0 and have_rho:
            mass_from_density = bulk_rho * volume

        chosen_mass = mass_val
        if volume > 0.0 and mass_from_density is not None and have_mass:
            # If they disagree wildly, prefer density-derived mass (it tends to scale with size),
            # unless density-derived looks obviously broken.
            ratio = (mass_val / mass_from_density) if mass_from_density > 1e-9 else 1.0
            if ratio < 0.33 or ratio > 3.0:
                chosen_mass = mass_from_density
        elif volume > 0.0 and mass_from_density is not None and not have_mass:
            chosen_mass = mass_from_density

        # GAP-PHYSICS-001 FIX: Prevent zero mass which crashes PhysX
        # Clamp mass to wide safety bounds (minimum 1g to prevent physics errors)
        chosen_mass = _clamp(chosen_mass, 0.001, 1000.0)

        # Additional safety check: if mass is exactly zero or negative, set to minimum
        if chosen_mass <= 0.0:
            logger.warning(f"[SIMREADY] obj_{oid}: Invalid mass {chosen_mass}, setting to 0.001kg (1g)")
            chosen_mass = 0.001

        merged["mass_kg"] = float(chosen_mass)

        # Recompute effective bulk density for logging/consistency
        if volume > 0.0:
            merged["bulk_density_kg_per_m3"] = float(_clamp(chosen_mass / volume, 0.5, 20000.0))
        else:
            merged["bulk_density_kg_per_m3"] = float(bulk_rho)

        # Notes
        notes = _get_str(["notes"], "")
        extra = f"gemini_estimated; volume={volume:.6f} m^3; eff_bulk_density={merged['bulk_density_kg_per_m3']:.1f} kg/m^3"
        merged["notes"] = (notes + " | " + extra).strip(" |")

        # ========== ROBOTICS-FOCUSED PROPERTIES ==========

        # Get size for validation bounds
        size_m = list(bounds.get("size_m") or [0.1, 0.1, 0.1])

        # Semantic class
        semantic_class = _get_str(
            ["semantic_class", "semanticClass", "category", "object_type"],
            str(merged.get("semantic_class", "object"))
        ).lower().replace(" ", "_")
        # Validate against known classes
        valid_classes = {
            "object", "container", "tool", "food", "electronics", "furniture",
            "toy", "clothing", "book", "kitchenware", "bottle", "can", "box",
            "bag", "utensil", "appliance", "decoration", "plant", "sporting_goods",
            "office_supply", "hygiene", "cleaning", "storage", "hardware"
        }
        if semantic_class not in valid_classes:
            semantic_class = "object"
        merged["semantic_class"] = semantic_class

        # Center of mass offset
        def _get_float_list(keys: List[str], default: List[float], n: int = 3) -> List[float]:
            for k in keys:
                if k in data:
                    val = data[k]
                    if isinstance(val, (list, tuple)) and len(val) == n:
                        try:
                            return [float(x) for x in val]
                        except Exception:
                            pass
            return list(default)

        com_offset = _get_float_list(
            ["center_of_mass_offset", "centerOfMassOffset", "com_offset"],
            merged.get("center_of_mass_offset", [0.0, 0.0, 0.0])
        )
        # Clamp CoM offset to reasonable range (within half the object size)
        max_offset = max(size_m) * 0.5 if size_m else 0.1
        com_offset = [_clamp(v, -max_offset, max_offset) for v in com_offset]
        merged["center_of_mass_offset"] = com_offset

        # Graspable flag
        merged["graspable"] = _get_bool(
            ["graspable", "is_graspable", "pickable"],
            bool(merged.get("graspable", True))
        )
        # Auto-set graspable=False for very heavy objects
        if merged["mass_kg"] > 25.0:
            merged["graspable"] = False

        # Grasp regions
        grasp_regions_raw = data.get("grasp_regions") or data.get("graspRegions") or []
        grasp_regions: List[Dict[str, Any]] = []
        if isinstance(grasp_regions_raw, list):
            for gr in grasp_regions_raw[:5]:  # limit to 5 grasp regions
                if isinstance(gr, dict):
                    pos = gr.get("position") or gr.get("pos") or [0.0, 0.0, 0.0]
                    gtype = str(gr.get("type", "power")).lower()
                    width = float(gr.get("width_m", 0.08))
                    if gtype not in {"pinch", "power", "wrap", "suction"}:
                        gtype = "power"
                    if isinstance(pos, (list, tuple)) and len(pos) == 3:
                        try:
                            grasp_regions.append({
                                "position": [float(p) for p in pos],
                                "type": gtype,
                                "width_m": _clamp(width, 0.005, 0.30),
                            })
                        except Exception:
                            pass
        merged["grasp_regions"] = grasp_regions if grasp_regions else merged.get("grasp_regions", [])

        # Surface roughness
        merged["surface_roughness"] = _clamp(
            _get_float(["surface_roughness", "surfaceRoughness", "roughness"],
                      float(merged.get("surface_roughness", 0.5))),
            0.0,
            1.0,
        )

        # Contact offsets (PhysX-specific, use sensible defaults based on object size)
        min_dim = min(size_m) if size_m else 0.1
        default_contact = _clamp(min_dim * 0.02, 0.002, 0.01)  # 2% of smallest dimension
        default_rest = _clamp(min_dim * 0.005, 0.0005, 0.003)  # 0.5% of smallest dimension

        merged["contact_offset_m"] = _clamp(
            _get_float(["contact_offset_m", "contactOffset"], default_contact),
            0.001,
            0.02,
        )
        merged["rest_offset_m"] = _clamp(
            _get_float(["rest_offset_m", "restOffset"], default_rest),
            0.0001,
            0.005,
        )

        return merged

    except Exception as e:  # pragma: no cover
        print(f"[SIMREADY] WARNING: Gemini failed for obj {oid}: {e}", file=sys.stderr)
        return base_cfg


# ---------- Physics configuration builders ----------


def build_physics_config(
    obj: Dict[str, Any],
    bounds: Dict[str, Any],
    mesh_bounds: Optional[List[float]] = None,
    mesh_center: Optional[List[float]] = None,
    gemini_client: Optional["genai.Client"] = None,
    reference_image: Optional["Image.Image"] = None,
    metadata: Optional[Dict[str, Any]] = None,
    deterministic_physics: bool = False,
) -> Dict[str, Any]:
    """
    Build complete physics configuration for an object.

    This function orchestrates physics estimation using:
    1. Gemini AI (if client and reference image available)
    2. Fallback to heuristic defaults based on size/metadata

    Args:
        obj: Object dict with metadata
        bounds: Computed bounds dict with size_m, center_m, volume_m3
        mesh_bounds: Optional mesh bounds from metadata [w, h, d]
        mesh_center: Optional mesh center from metadata [x, y, z]
        gemini_client: Optional Gemini API client for AI estimation
        reference_image: Optional PIL image for Gemini

    Returns:
        Complete physics configuration dict
    """
    # Try Gemini-based estimation first unless deterministic is requested
    if not deterministic_physics and gemini_client is not None and have_gemini():
        try:
            physics_cfg = call_gemini_for_object(
                client=gemini_client,
                oid=obj.get("id"),
                obj=obj,
                bounds=bounds,
                reference_image=reference_image,
            )
            physics_cfg["estimation_source"] = "gemini"
            # Store mesh bounds info for catalog
            if mesh_bounds:
                physics_cfg["mesh_bounds"] = {
                    "size": mesh_bounds,
                    "center": mesh_center or [0.0, 0.0, 0.0],
                }
            return physics_cfg
        except Exception as e:
            print(f"[SIMREADY] WARNING: Gemini physics estimation failed: {e}", file=sys.stderr)
            logger.warning(
                "[SIMREADY] Falling back to heuristic physics estimates for obj %s due to Gemini failure.",
                obj.get("id"),
            )

    if deterministic_physics:
        physics_cfg = estimate_deterministic_physics(obj, bounds, metadata)
    else:
        # Fallback to heuristic defaults
        if gemini_client is None or not have_gemini():
            logger.warning(
                "[SIMREADY] Using heuristic physics estimates for obj %s (no Gemini client available).",
                obj.get("id"),
            )
        physics_cfg = estimate_default_physics(obj, bounds)

    # Store mesh bounds info for catalog
    if mesh_bounds:
        physics_cfg["mesh_bounds"] = {
            "size": mesh_bounds,
            "center": mesh_center or [0.0, 0.0, 0.0],
        }

    return physics_cfg


def emit_usd(
    visual_path: Path,
    physics_cfg: Dict[str, Any],
    bounds: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Emit a simready USD file for an asset.

    This function writes a USD wrapper with physics properties
    next to the visual asset.

    Args:
        visual_path: Path to the visual USD/USDZ asset
        physics_cfg: Physics configuration dict
        bounds: Optional bounds dict (extracted from physics_cfg if not provided)

    Returns:
        Path to the written simready.usda file
    """
    # Determine output path (sibling to visual asset)
    out_path = visual_path.parent / "simready.usda"

    # Compute asset reference relative to output location
    asset_rel = "./" + visual_path.name

    # Extract or compute bounds
    if bounds is None:
        # Try to reconstruct bounds from physics config
        mesh_bounds_info = physics_cfg.get("mesh_bounds", {})
        size = mesh_bounds_info.get("size") or [0.1, 0.1, 0.1]
        center = mesh_bounds_info.get("center") or [0.0, 0.0, 0.0]
        volume = size[0] * size[1] * size[2]
        bounds = {
            "size_m": size,
            "center_m": center,
            "volume_m3": volume,
            "source": "physics_cfg",
        }

    # Write the simready USD
    write_simready_usd(out_path, asset_rel, physics_cfg, bounds)

    return out_path


# ---------- USD writing ----------

def choose_static_visual_asset(assets_root: Path, oid: Any) -> Optional[Tuple[Path, str]]:
    """
    For a static object, pick the visual asset file to reference.

    Preference (USD-compatible formats only - GLB cannot be referenced directly):
    1) model.usdz
    2) asset.usdz
    3) model.usd / model.usdc
    """
    base_dir = assets_root / f"obj_{oid}"

    candidates = [
        base_dir / "model.usdz",
        base_dir / "asset.usdz",
        base_dir / "model.usd",
        base_dir / "model.usdc",
    ]

    for p in candidates:
        if p.is_file():
            rel = os.path.relpath(p, base_dir).replace("\\", "/")
            if not rel.startswith("."):
                rel = "./" + rel
            return p, rel

    legacy_dir = assets_root / "static" / f"obj_{oid}"
    candidates = [
        legacy_dir / "model.usdz",
        legacy_dir / "asset.usdz",
        legacy_dir / "model.usd",
        legacy_dir / "model.usdc",
    ]
    for p in candidates:
        if p.is_file():
            rel = os.path.relpath(p, legacy_dir).replace("\\", "/")
            if not rel.startswith("."):
                rel = "./" + rel
            return p, rel

    return None


def _validate_simready_usd(usd_path: Path, physics: Dict[str, Any]) -> None:
    """
    Validate the written USD file for common physics errors.

    GAP-USD-001 FIX: Validate USD stage after modifications to catch errors early.
    """
    if not usd_path.exists():
        logger.error(f"[SIMREADY] USD file not found after write: {usd_path}")
        return

    # Basic validation: check file size
    file_size = usd_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Empty USD file written: {usd_path}")
    if file_size < 100:  # Suspiciously small
        logger.warning(f"[SIMREADY] USD file very small ({file_size} bytes): {usd_path}")

    # Validate required physics attributes are present
    content = usd_path.read_text()

    # Check for required physics attributes
    mass = physics.get("mass_kg", 0)
    if mass <= 0:
        raise ValueError(f"Invalid mass {mass} in physics config for {usd_path}")

    if physics.get("dynamic", True):
        if "PhysicsRigidBodyAPI" not in content:
            logger.warning(f"[SIMREADY] Dynamic object missing PhysicsRigidBodyAPI: {usd_path}")
        if "physics:mass" not in content:
            logger.warning(f"[SIMREADY] Dynamic object missing physics:mass: {usd_path}")
        if "PhysicsMassAPI" not in content:
            logger.warning(f"[SIMREADY] Dynamic object missing PhysicsMassAPI: {usd_path}")

    # Check for physics material
    if "PhysicsMaterialAPI" not in content:
        logger.warning(f"[SIMREADY] Missing PhysicsMaterialAPI: {usd_path}")

    # Check for collision API
    if "PhysicsCollisionAPI" not in content:
        logger.warning(f"[SIMREADY] Missing PhysicsCollisionAPI: {usd_path}")

    logger.debug(f"[SIMREADY] USD validation passed: {usd_path}")


def write_simready_usd(out_path: Path, asset_rel: str, physics: Dict[str, Any], bounds: Dict[str, Any]) -> None:
    """
    Create a small USD wrapper that is actually "sim-ready" in Isaac Sim:

    - If dynamic:
        - Apply PhysicsRigidBodyAPI + PhysicsMassAPI to /Asset
        - Author bool physics:rigidBodyEnabled
        - Author physics:mass (and do NOT also author density to avoid precedence confusion)
        - Author physics:centerOfMass for accurate manipulation dynamics

    - Always:
        - Reference the visual USD/Z under /Asset/Visual
        - Create a collision proxy prim with PhysicsCollisionAPI + PhysxCollisionAPI under /Asset/Collision
        - Define a UsdShade Material with PhysicsMaterialAPI under /Asset/Looks
        - Bind it to the collider using rel material:binding:physics
        - Add robotics-focused metadata (semantic class, graspability, grasp regions, etc.)
    """
    add_proxy = _get_env_value("SIMREADY_ADD_PROXY_COLLIDER", "true").lower() in {"1", "true", "yes"}

    # Core physics properties
    dynamic = bool(physics.get("dynamic", True))
    mass = float(physics.get("mass_kg", 1.0))
    static_friction = float(physics.get("static_friction", 0.6))
    dynamic_friction = float(physics.get("dynamic_friction", 0.5))
    restitution = float(physics.get("restitution", 0.1))
    collision_shape = str(physics.get("collision_shape", "box")).lower()

    # GAP-PHYSICS-010 FIX: Add kinematic flag and GPU collision support
    kinematic_enabled = bool(physics.get("kinematic_enabled", False))
    gpu_collision = bool(physics.get("gpu_collision", True))  # Default enabled for performance

    # Robotics-focused properties
    semantic_class = str(physics.get("semantic_class", "object"))
    material_name = str(physics.get("material_name", "generic"))
    graspable = bool(physics.get("graspable", True))
    grasp_regions = list(physics.get("grasp_regions", []))
    surface_roughness = float(physics.get("surface_roughness", 0.5))
    contact_offset = float(physics.get("contact_offset_m", 0.005))
    rest_offset = float(physics.get("rest_offset_m", 0.001))

    # Center of mass (geometric center + offset)
    com_offset = list(physics.get("center_of_mass_offset", [0.0, 0.0, 0.0]))

    size_m = list(bounds.get("size_m") or [0.1, 0.1, 0.1])
    center_m = list(bounds.get("center_m") or [0.0, 0.0, 0.0])

    # Compute absolute center of mass (geometric center + offset)
    center_of_mass = [center_m[i] + com_offset[i] for i in range(3)]

    # Use padded bounds for collision proxy (if any)
    size_pad = padded_size(size_m)

    lines: List[str] = []
    lines.append("#usda 1.0")
    lines.append("(\n    metersPerUnit = 1\n    kilogramsPerUnit = 1\n)")
    lines.append("")

    # Asset root
    if dynamic:
        lines.append('def Xform "Asset" (')
        lines.append('    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]')
        lines.append(")")
    else:
        # Static colliders don't need a rigid body; they are static when no RigidBodyAPI is in the ancestry.
        lines.append('def Xform "Asset"')
    lines.append("{")

    if dynamic:
        lines.append("    bool physics:rigidBodyEnabled = 1")
        lines.append(f"    float physics:mass = {mass:.6f}")
        # Center of mass for accurate manipulation dynamics
        lines.append(f"    point3f physics:centerOfMass = ({center_of_mass[0]:.6f}, {center_of_mass[1]:.6f}, {center_of_mass[2]:.6f})")
        # (Intentionally not authoring physics:density here. Mass has precedence over density in USD Physics.)

        # GAP-PHYSICS-010 FIX: Add physics:approximation for collision type
        if collision_shape in {"convex_hull", "convex_decomposition"}:
            approximation_token = "convexDecomposition" if collision_shape == "convex_decomposition" else "convexHull"
            lines.append(f'    token physics:approximation = "{approximation_token}"')

        # GAP-PHYSICS-010 FIX: Add kinematic enabled flag for Isaac Sim
        if kinematic_enabled:
            lines.append("    bool physics:kinematicEnabled = 1")

    # Visual reference
    lines.append("")
    lines.append('    def Xform "Visual" (')
    lines.append(f"        prepend references = @{asset_rel}@")
    lines.append("    )")
    lines.append("    {")
    lines.append("    }")

    # Physics material (UsdShade.Material + PhysicsMaterialAPI)
    lines.append("")
    lines.append('    def Scope "Looks"')
    lines.append("    {")
    lines.append('        def Material "PhysicsMaterial" (')
    lines.append('            prepend apiSchemas = ["PhysicsMaterialAPI"]')
    lines.append("        )")
    lines.append("        {")
    lines.append(f"            float physics:staticFriction = {static_friction:.4f}")
    lines.append(f"            float physics:dynamicFriction = {dynamic_friction:.4f}")
    lines.append(f"            float physics:restitution = {restitution:.4f}")
    lines.append("        }")
    lines.append("    }")

    # Collision proxy
    if add_proxy:
        lines.append("")
        lines.append('    def Xform "Collision"')
        lines.append("    {")
        material_path = "</Asset/Looks/PhysicsMaterial>"

        # GAP-PHYSICS-010 FIX: Support convex decomposition and hull-based collision
        if collision_shape in {"convex_hull", "convex_decomposition"}:
            # For complex shapes, reference the visual mesh with convex approximation
            lines.append(f'        def "{collision_shape.upper()}_Collider" (')
            lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]')
            lines.append("        )")
            lines.append("        {")
            lines.append(f"            rel material:binding:physics = {material_path}")
            lines.append(f"            prepend references = @{asset_rel}@")
            lines.append(f"            double3 xformOp:translate = ({center_m[0]:.6f}, {center_m[1]:.6f}, {center_m[2]:.6f})")
            lines.append('            uniform token[] xformOpOrder = ["xformOp:translate"]')
            # PhysX contact properties for stable simulation
            lines.append(f"            float physxCollision:contactOffset = {contact_offset:.6f}")
            lines.append(f"            float physxCollision:restOffset = {rest_offset:.6f}")
            # Set the approximation method for the collider
            approximation_token = "convexDecomposition" if collision_shape == "convex_decomposition" else "convexHull"
            lines.append(f'            token physxCollision:approximation = "{approximation_token}"')
            # GAP-PHYSICS-010 FIX: Add GPU collision support
            if gpu_collision:
                lines.append(f'            bool physxCollision:gpuCollision = true')
            lines.append("        }")

        elif collision_shape == "sphere":
            radius = 0.5 * max(size_pad)
            lines.append('        def Sphere "Collider" (')
            lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]')
            lines.append("        )")
            lines.append("        {")
            lines.append(f"            rel material:binding:physics = {material_path}")
            lines.append(f"            double radius = {radius:.6f}")
            lines.append(f"            double3 xformOp:translate = ({center_m[0]:.6f}, {center_m[1]:.6f}, {center_m[2]:.6f})")
            lines.append('            uniform token[] xformOpOrder = ["xformOp:translate"]')
            # PhysX contact properties for stable simulation
            lines.append(f"            float physxCollision:contactOffset = {contact_offset:.6f}")
            lines.append(f"            float physxCollision:restOffset = {rest_offset:.6f}")
            # GAP-PHYSICS-010 FIX: Add GPU collision support
            if gpu_collision:
                lines.append(f'            bool physxCollision:gpuCollision = true')
            lines.append("        }")

        elif collision_shape == "capsule":
            # Capsule attributes: radius + height (shaft, excludes caps) + axis
            # Choose axis along the largest dimension
            sx, sy, sz = size_pad
            dims = {"X": sx, "Y": sy, "Z": sz}
            axis = max(dims, key=dims.get)
            # Radius based on the smaller transverse dimensions
            if axis == "X":
                r = 0.5 * min(sy, sz)
                max_dim = sx
            elif axis == "Y":
                r = 0.5 * min(sx, sz)
                max_dim = sy
            else:
                r = 0.5 * min(sx, sy)
                max_dim = sz

            r = max(r, 1e-4)
            height = max(max_dim - 2.0 * r, 0.0)

            lines.append('        def Capsule "Collider" (')
            lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]')
            lines.append("        )")
            lines.append("        {")
            lines.append(f"            rel material:binding:physics = {material_path}")
            lines.append(f'            uniform token axis = "{axis}"')
            lines.append(f"            double radius = {r:.6f}")
            lines.append(f"            double height = {height:.6f}")
            lines.append(f"            double3 xformOp:translate = ({center_m[0]:.6f}, {center_m[1]:.6f}, {center_m[2]:.6f})")
            lines.append('            uniform token[] xformOpOrder = ["xformOp:translate"]')
            # PhysX contact properties for stable simulation
            lines.append(f"            float physxCollision:contactOffset = {contact_offset:.6f}")
            lines.append(f"            float physxCollision:restOffset = {rest_offset:.6f}")
            # GAP-PHYSICS-010 FIX: Add GPU collision support
            if gpu_collision:
                lines.append(f'            bool physxCollision:gpuCollision = true')
            lines.append("        }")

        else:
            # Default: box proxy collider (Cube) with non-uniform scale.
            lines.append('        def Cube "Collider" (')
            lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]')
            lines.append("        )")
            lines.append("        {")
            lines.append(f"            rel material:binding:physics = {material_path}")
            lines.append("            double size = 1")
            lines.append(f"            double3 xformOp:translate = ({center_m[0]:.6f}, {center_m[1]:.6f}, {center_m[2]:.6f})")
            lines.append(f"            float3 xformOp:scale = ({size_pad[0]:.6f}, {size_pad[1]:.6f}, {size_pad[2]:.6f})")
            # xformOpOrder is applied in reverse; this ordering yields scale first, then translate.
            lines.append('            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
            # PhysX contact properties for stable simulation
            lines.append(f"            float physxCollision:contactOffset = {contact_offset:.6f}")
            lines.append(f"            float physxCollision:restOffset = {rest_offset:.6f}")
            # GAP-PHYSICS-010 FIX: Add GPU collision support
            if gpu_collision:
                lines.append(f'            bool physxCollision:gpuCollision = true')
            lines.append("        }")

        lines.append("    }")

    # ========== ROBOTICS-FOCUSED METADATA ==========
    lines.append("")
    lines.append("    # Robotics metadata for perception and manipulation")

    # Semantic class (for object detection/segmentation)
    lines.append(f'    string semantic:class = "{semantic_class}"')

    # Material name (for tactile/visual sensing)
    safe_material = material_name.replace('"', '\\"')
    lines.append(f'    string simready:material = "{safe_material}"')

    # Graspability flag
    graspable_val = "true" if graspable else "false"
    lines.append(f"    bool simready:graspable = {graspable_val}")

    # Surface roughness (for tactile simulation: 0=smooth glass, 1=sandpaper)
    lines.append(f"    float simready:surfaceRoughness = {surface_roughness:.4f}")

    # Grasp regions (for manipulation planning)
    if grasp_regions:
        lines.append("")
        lines.append('    def Scope "GraspRegions"')
        lines.append("    {")
        for idx, gr in enumerate(grasp_regions):
            pos = gr.get("position", [0.0, 0.0, 0.0])
            gtype = gr.get("type", "power")
            width = gr.get("width_m", 0.08)
            lines.append(f'        def Xform "Grasp_{idx}" {{')
            lines.append(f"            double3 xformOp:translate = ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})")
            lines.append('            uniform token[] xformOpOrder = ["xformOp:translate"]')
            lines.append(f'            string simready:graspType = "{gtype}"')
            lines.append(f"            float simready:graspWidth = {width:.4f}")
            lines.append("        }")
        lines.append("    }")

    # Optional notes for debugging
    notes = str(physics.get("notes", "")).strip()
    if notes:
        safe_notes = notes.replace('"', '\\"')
        lines.append("")
        lines.append(f'    string simready:notes = "{safe_notes}"')

    lines.append("}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")

    # GAP-USD-001 FIX: Validate USD after writing
    _validate_simready_usd(out_path, physics)


# ---------- Main pipeline ----------


def _env_flag(name: str, default: bool = False) -> bool:
    value = _get_env_value(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _resolve_production_mode() -> bool:
    pipeline_env = _get_env_value("PIPELINE_ENV", "").strip().lower()
    return _env_flag("SIMREADY_PRODUCTION_MODE") or pipeline_env in {"prod", "production"}


def prepare_simready_assets_job(
    bucket: str,
    scene_id: str,
    assets_prefix: str,
    root: Path = GCS_ROOT,
    allow_heuristic_fallback: Optional[bool] = None,
    production_mode: Optional[bool] = None,
) -> int:
    if not assets_prefix:
        print("[SIMREADY] ASSETS_PREFIX is required", file=sys.stderr)
        return 1

    assets_root = root / assets_prefix
    manifest_path = assets_root / "scene_manifest.json"

    if production_mode is None:
        production_mode = _resolve_production_mode()
    if allow_heuristic_fallback is None:
        allow_heuristic_fallback = _env_flag("SIMREADY_ALLOW_HEURISTIC_FALLBACK")
    physics_mode = (_get_env_value("SIMREADY_PHYSICS_MODE", "auto") or "auto").strip().lower()
    allow_deterministic_physics = _env_flag("SIMREADY_ALLOW_DETERMINISTIC_PHYSICS")
    pipeline_env_raw = (_get_env_value("PIPELINE_ENV", "") or "").strip().lower()
    production_mode_set = _env_flag("SIMREADY_PRODUCTION_MODE") or pipeline_env_raw in {
        "prod",
        "production",
    }
    if physics_mode == "deterministic" and not production_mode_set:
        logger.warning(
            "[SIMREADY] Deterministic physics requested but production mode is unset. "
            "If this run is meant to mirror production, set PIPELINE_ENV=production "
            "or SIMREADY_PRODUCTION_MODE=1."
        )

    print(f"[SIMREADY] Bucket={bucket}")
    print(f"[SIMREADY] Scene={scene_id}")
    print(f"[SIMREADY] Assets root={assets_root}")
    print(f"[SIMREADY] Loading {manifest_path} (or legacy scene_assets.json)")
    print(f"[SIMREADY] Production mode={'true' if production_mode else 'false'}")
    print(f"[SIMREADY] Heuristic fallback allowed={'true' if allow_heuristic_fallback else 'false'}")
    print(f"[SIMREADY] Physics estimation mode={physics_mode}")
    print(f"[SIMREADY] Deterministic physics allowed={'true' if allow_deterministic_physics else 'false'}")
    try:
        fallback_min_coverage = float(_get_env_value("SIMREADY_FALLBACK_MIN_COVERAGE", "0.6"))
    except ValueError:
        fallback_min_coverage = 0.6
    fallback_min_coverage = _clamp(fallback_min_coverage, 0.0, 1.0)
    if fallback_min_coverage:
        print(f"[SIMREADY] Fallback physics min coverage={fallback_min_coverage:.2f}")

    try:
        non_llm_min_quality = float(_get_env_value("SIMREADY_NON_LLM_MIN_QUALITY", "0.85"))
    except ValueError:
        non_llm_min_quality = 0.85
    non_llm_min_quality = _clamp(non_llm_min_quality, 0.0, 1.0)
    if non_llm_min_quality:
        print(f"[SIMREADY] Non-LLM physics min quality={non_llm_min_quality:.2f}")

    scene_assets = load_manifest_or_scene_assets(assets_root)
    if scene_assets is None:
        legacy_path = assets_root / "scene_assets.json"
        print(
            f"[SIMREADY] scene manifest missing at {manifest_path} and {legacy_path}",
            file=sys.stderr,
        )
        return 1
    objects = scene_assets.get("objects", [])
    print(f"[SIMREADY] Found {len(objects)} objects in manifest")

    catalog_client = AssetCatalogClient()

    # GAP-PHYSICS-011 FIX: Initialize physics profile selector
    profile_selector = None
    if HAVE_PROFILE_SELECTOR:
        try:
            profile_selector = create_profile_selector()
            print("[SIMREADY] Physics profile selector initialized")
        except Exception as e:
            print(f"[SIMREADY] WARNING: Failed to initialize profile selector: {e}", file=sys.stderr)

    if production_mode and allow_heuristic_fallback:
        logger.warning(
            "[SIMREADY] SIMREADY_ALLOW_HEURISTIC_FALLBACK is ignored in production mode."
        )

    use_deterministic_physics = physics_mode == "deterministic"
    if physics_mode not in {"auto", "gemini", "deterministic"}:
        logger.warning(
            "[SIMREADY] Unknown SIMREADY_PHYSICS_MODE '%s'; defaulting to auto.",
            physics_mode,
        )
        physics_mode = "auto"

    client = None
    if physics_mode == "gemini" or (physics_mode == "auto" and have_gemini()):
        gemini_api_key = _get_secret_value(
            SECRET_ID_GEMINI,
            "GEMINI_API_KEY",
            purpose="Gemini API",
            production_mode=production_mode,
            required=production_mode,
        )
        if gemini_api_key:
            client = genai.Client(api_key=gemini_api_key)
            print("[SIMREADY] Gemini client initialized")
        elif production_mode and physics_mode != "deterministic":
            logger.error(
                "[SIMREADY] Gemini API key is required in production mode. "
                "Set Secret Manager ID '%s' or env var 'GEMINI_API_KEY'.",
                SECRET_ID_GEMINI,
            )
            return 2
        else:
            logger.warning(
                "[SIMREADY] Gemini API key missing; continuing with non-LLM physics."
            )

    if client is None and physics_mode == "gemini":
        logger.error(
            "[SIMREADY] SIMREADY_PHYSICS_MODE=gemini but Gemini client is unavailable."
        )
        return 2

    if client is None and physics_mode == "auto" and allow_deterministic_physics:
        use_deterministic_physics = True

    if client is None and physics_mode == "auto" and not allow_deterministic_physics:
        if production_mode:
            logger.error(
                "[SIMREADY] Production mode requires Gemini or deterministic physics. "
                "Set SIMREADY_PHYSICS_MODE=deterministic or SIMREADY_ALLOW_DETERMINISTIC_PHYSICS=1, "
                "or provide Gemini credentials."
            )
            return 2
        if allow_heuristic_fallback:
            logger.warning(
                "[SIMREADY] Gemini unavailable; using heuristic-only physics estimation (CI/testing fallback enabled)."
            )
        else:
            logger.warning(
                "[SIMREADY] Gemini unavailable; using heuristic-only physics estimation. "
                "Set SIMREADY_ALLOW_HEURISTIC_FALLBACK=1 to acknowledge this fallback for CI/testing."
            )

    if production_mode and physics_mode == "deterministic":
        use_deterministic_physics = True

    simready_paths: Dict[Any, str] = {}
    fallback_stats = {"total": 0, "covered": 0}
    quality_stats = {"total": 0, "passed": 0}
    fallback_mode = use_deterministic_physics

    # GAP-PERF-002 FIX: Process objects in parallel for 10-50x speedup
    def process_single_object(obj: Dict[str, Any]) -> Optional[Tuple[str, str, str, bool, Optional[bool]]]:
        """
        Process a single object. Returns (oid, sim_rel, sim_path, fallback_covered, quality_passed)
        or None on failure.

        This function is thread-safe and can be called in parallel.
        """
        oid = obj.get("id")
        if oid is None:
            return None

        try:
            visual = choose_static_visual_asset(assets_root, oid)
            if visual is None:
                print(f"[SIMREADY] WARNING: no visual asset found for obj {oid}", file=sys.stderr)
                return None

            visual_path, visual_rel = visual
            obj_metadata = load_object_metadata(root, obj, assets_prefix, catalog_client)
            obj_metadata = obj_metadata or {}

            mesh_bounds_metadata, mesh_center_metadata = extract_mesh_bounds_from_metadata(
                obj_metadata
            )
            obb_bounds, obb_center = extract_obb_bounds_from_obj(obj)

            gemini_estimated_size = None
            ref_img = None
            ref_image = choose_reference_image_path(root, obj)
            if ref_image:
                ref_img = load_image_for_gemini(ref_image)
                if client and ref_img:
                    gemini_estimated_size = estimate_scale_gemini(
                        client, obj, ref_img, obj_metadata.get("class_name")
                    )

            # Compute bounds using metadata and Gemini estimates
            bounds = compute_bounds(
                obj=obj,
                metadata=obj_metadata,
                gemini_estimated_size=gemini_estimated_size,
            )

            # Build physics config with Gemini AI or heuristics
            physics_cfg = build_physics_config(
                obj=obj,
                bounds=bounds,
                mesh_bounds=mesh_bounds_metadata,
                mesh_center=mesh_center_metadata,
                gemini_client=client,
                reference_image=ref_img,
                metadata=obj_metadata,
                deterministic_physics=use_deterministic_physics,
            )

            estimation_source = str(physics_cfg.get("estimation_source", ""))
            fallback_covered = estimation_source in {"deterministic_metadata", "deterministic_material"}
            quality_passed: Optional[bool] = None
            if use_deterministic_physics:
                quality_passed, reasons = _validate_non_llm_physics_quality(physics_cfg, bounds)
                if not quality_passed:
                    print(
                        f"[SIMREADY] WARNING: deterministic physics quality check failed for obj {oid}: "
                        + ", ".join(reasons),
                        file=sys.stderr,
                    )

            # Compute sim2real distribution ranges for domain randomization
            physics_distributions = compute_physics_distributions(physics_cfg, bounds)
            physics_cfg.update(physics_distributions)

            # GAP-PHYSICS-011 FIX: Apply physics profile based on scene/task characteristics
            if profile_selector:
                try:
                    # Get task/scene hint from object metadata or use default
                    task_hint = obj.get("task") or scene_assets.get("task") or obj.get("category", "unknown")
                    physics_cfg = profile_selector.apply_profile_to_physics(physics_cfg, profile_selector.select_profile(task_hint))
                except Exception as e:
                    logger.warning(f"[SIMREADY] Failed to apply physics profile for obj {oid}: {e}")

            # Emit the simready USD with physics
            sim_path = emit_usd(visual_path, physics_cfg, bounds)

            if catalog_client:
                try:
                    export_bounds = physics_cfg.get("mesh_bounds") or {}
                    catalog_payload = {
                        "id": obj.get("id"),
                        "class_name": obj.get("class_name"),
                        "asset_path": obj.get("asset_path") or visual_rel,
                        "mesh_bounds": {"export": export_bounds},
                        "physics": physics_cfg,
                        "material_name": physics_cfg.get("material_name"),
                    }
                    catalog_client.publish_metadata(
                        oid, catalog_payload, asset_path=catalog_payload["asset_path"]
                    )
                except Exception as exc:  # pragma: no cover - network errors
                    print(
                        f"[SIMREADY] WARNING: failed to publish catalog metadata for obj {oid}: {exc}",
                        file=sys.stderr,
                    )

            sim_rel = f"{assets_prefix}/obj_{oid}/simready.usda"
            if "static/obj_" in str(visual_path):
                sim_rel = f"{assets_prefix}/static/obj_{oid}/simready.usda"

            print(f"[SIMREADY] ✓ Processed obj {oid}")
            return (oid, sim_rel, str(sim_path), fallback_covered, quality_passed)

        except Exception as e:
            print(f"[SIMREADY] ERROR: Failed to process obj {oid}: {e}", file=sys.stderr)
            return None

    # Use parallel processing if available
    if HAVE_PARALLEL_PROCESSING and len(objects) > 5:  # Only parallelize if >5 objects
        print(f"[SIMREADY] Processing {len(objects)} objects in parallel (max 10 workers)...")
        result = process_parallel_threaded(
            objects,
            process_fn=process_single_object,
            max_workers=10,  # Limit to 10 concurrent Gemini calls
        )

        # Collect successful results
        for success_item in result.successful_results:
            if success_item:
                oid, sim_rel, sim_path, fallback_covered, quality_passed = success_item
                simready_paths[oid] = sim_rel
                print(f"[SIMREADY] Wrote simready asset for obj {oid} -> {sim_path}")
                if fallback_mode:
                    fallback_stats["total"] += 1
                    if fallback_covered:
                        fallback_stats["covered"] += 1
                if quality_passed is not None:
                    quality_stats["total"] += 1
                    if quality_passed:
                        quality_stats["passed"] += 1

        # Report failures
        if result.failed_count > 0:
            print(f"[SIMREADY] WARNING: {result.failed_count}/{result.total_count} objects failed", file=sys.stderr)

        print(f"[SIMREADY] Parallel processing complete: {result.success_count}/{result.total_count} succeeded")

    else:
        # Sequential fallback for small batches or when parallel processing unavailable
        print(f"[SIMREADY] Processing {len(objects)} objects sequentially...")
        for obj in objects:
            result = process_single_object(obj)
            if result:
                oid, sim_rel, sim_path, fallback_covered, quality_passed = result
                simready_paths[oid] = sim_rel
                print(f"[SIMREADY] Wrote simready asset for obj {oid} -> {sim_path}")
                if fallback_mode:
                    fallback_stats["total"] += 1
                    if fallback_covered:
                        fallback_stats["covered"] += 1
                if quality_passed is not None:
                    quality_stats["total"] += 1
                    if quality_passed:
                        quality_stats["passed"] += 1

    if fallback_mode and fallback_stats["total"] > 0 and fallback_min_coverage > 0.0:
        coverage = fallback_stats["covered"] / float(fallback_stats["total"])
        print(
            f"[SIMREADY] Fallback physics coverage: {coverage * 100.0:.1f}% "
            f"({fallback_stats['covered']}/{fallback_stats['total']})"
        )
        if coverage < fallback_min_coverage:
            logger.error(
                "[SIMREADY] Fallback physics coverage %.1f%% is below the minimum %.1f%%. "
                "Increase metadata/material coverage or lower SIMREADY_FALLBACK_MIN_COVERAGE.",
                coverage * 100.0,
                fallback_min_coverage * 100.0,
            )
            return 3

    if use_deterministic_physics and quality_stats["total"] > 0 and non_llm_min_quality > 0.0:
        quality_ratio = quality_stats["passed"] / float(quality_stats["total"])
        print(
            f"[SIMREADY] Non-LLM physics quality: {quality_ratio * 100.0:.1f}% "
            f"({quality_stats['passed']}/{quality_stats['total']})"
        )
        if quality_ratio < non_llm_min_quality:
            logger.error(
                "[SIMREADY] Non-LLM physics quality %.1f%% is below the minimum %.1f%%. "
                "Improve metadata/material coverage or lower SIMREADY_NON_LLM_MIN_QUALITY.",
                quality_ratio * 100.0,
                non_llm_min_quality * 100.0,
            )
            return 4

    marker_path = assets_root / ".simready_complete"
    if simready_paths:
        marker_content = {
            "status": "complete",
            "simready_assets": simready_paths,
            "count": len(simready_paths),
        }
        marker_path.write_text(json.dumps(marker_content, indent=2), encoding="utf-8")
        print(f"[SIMREADY] Created completion marker at {marker_path}")
        print(f"[SIMREADY] Generated {len(simready_paths)} simready assets")
    else:
        marker_content = {
            "status": "complete",
            "simready_assets": {},
            "count": 0,
            "note": "No objects required simready processing (no visual assets found)",
        }
        marker_path.write_text(json.dumps(marker_content, indent=2), encoding="utf-8")
        print(
            "[SIMREADY] No simready assets were created; created completion marker anyway",
            file=sys.stderr,
        )

    return 0


def main() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))

    from blueprint_sim.simready import run_from_env

    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
            "ASSETS_PREFIX": "Path prefix for assets (scenes/<sceneId>/assets)",
        },
        label="[SIMREADY]",
    )

    bucket = os.environ["BUCKET"]
    scene_id = os.environ["SCENE_ID"]
    assets_prefix = os.environ["ASSETS_PREFIX"]
    input_params = {
        "bucket": bucket,
        "scene_id": scene_id,
        "assets_prefix": assets_prefix,
    }
    partial_results = {
        "simready_prefix": assets_prefix,
        "simready_marker": (
            f"{assets_prefix}/.simready_complete" if assets_prefix else None
        ),
    }

    def _write_failure_marker(exc: Exception, failed_step: str) -> None:
        if not bucket or not scene_id:
            print(
                "[SIMREADY] WARNING: Skipping failure marker; BUCKET/SCENE_ID missing.",
                file=sys.stderr,
            )
            return
        FailureMarkerWriter(bucket, scene_id, JOB_NAME).write_failure(
            exception=exc,
            failed_step=failed_step,
            input_params=input_params,
            partial_results=partial_results,
        )

    validated = False
    try:
        assets_root = GCS_ROOT / assets_prefix
        validate_scene_manifest(assets_root / "scene_manifest.json", label="[SIMREADY]")
        validated = True

        sys.exit(run_from_env(root=GCS_ROOT))
    except SystemExit as exc:
        if exc.code not in (0, None):
            failed_step = "entrypoint_validation" if not validated else "entrypoint_exit"
            _write_failure_marker(RuntimeError("Job exited early"), failed_step)
        raise
    except Exception as exc:
        _write_failure_marker(exc, "entrypoint")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        send_alert(
            event_type="simready_job_fatal_exception",
            summary="SimReady job failed with an unhandled exception",
            details={
                "job": "simready-job",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            severity=os.getenv("ALERT_JOB_EXCEPTION_SEVERITY", "critical"),
        )
        raise
