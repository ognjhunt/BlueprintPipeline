import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

SAM3D_REPO_ROOT = Path(os.getenv("SAM3D_REPO_ROOT", Path("/app/sam3d-objects")))
SAM3D_CHECKPOINT_ROOT = Path(
    os.getenv("SAM3D_CHECKPOINT_ROOT", SAM3D_REPO_ROOT / "checkpoints")
)
SAM3D_HF_REPO = os.getenv("SAM3D_HF_REPO", "facebook/sam-3d-objects")
SAM3D_HF_REVISION = os.getenv("SAM3D_HF_REVISION")

# ---------------------------------------------------------------------------
# Ensure upstream SAM 3D notebook code (inference.py) sees a CONDA-like env.
# That file does:
#     os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
# which raises KeyError if CONDA_PREFIX is unset. We are NOT using conda,
# so we synthesize a sensible default based on CUDA_HOME or /usr/local/cuda.
# ---------------------------------------------------------------------------
if "CONDA_PREFIX" not in os.environ:
    # Prefer an existing CUDA_HOME if present, otherwise default to /usr/local/cuda
    cuda_home_default = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    # Ensure CUDA_HOME is set to something coherent
    os.environ.setdefault("CUDA_HOME", cuda_home_default)
    # Make CONDA_PREFIX match CUDA_HOME so inference.py's assignment is safe
    os.environ["CONDA_PREFIX"] = os.environ["CUDA_HOME"]

# ---------------------------------------------------------------------------
# Make the cloned SAM 3D repo importable.
#
# - inference.py lives under   <repo_root>/notebook
#
# We add BOTH the repo root and the notebook folder to sys.path so that:
#   from inference import Inference   -> finds notebook/inference.py
# ---------------------------------------------------------------------------
SAM3D_PYTHON_PATHS = [
    SAM3D_REPO_ROOT,                      # e.g. /app/sam3d-objects
    SAM3D_REPO_ROOT / "notebook",         # e.g. /app/sam3d-objects/notebook (inference.py)
    Path("/workspace/sam3d-objects"),
    Path("/workspace/sam3d-objects/notebook"),
    Path(__file__).parent / "notebook",
]

for p in SAM3D_PYTHON_PATHS:
    if p.is_dir():
        sys.path.append(str(p))


def _ensure_pytorch3d_stub() -> None:
    """
    Ensure that either the real pytorch3d package is importable, or register a
    minimal stub that provides just the pieces SAM 3D uses:

      - pytorch3d.transforms.quaternion_multiply
      - pytorch3d.transforms.quaternion_invert
      - pytorch3d.transforms.Transform3d
      - pytorch3d.renderer.look_at_view_transform

    This lets the Cloud Run job proceed even if installing full pytorch3d (with
    CUDA extensions) is not practical in the container.
    """
    try:
        import pytorch3d  # type: ignore  # noqa: F401
        # Real pytorch3d is available; nothing to do.
        return
    except Exception:
        # Fall through and register a small stub.
        pass

    try:
        import types
        import math
        import torch

        print(
            "[SAM3D] pytorch3d not found; installing minimal stub in sys.modules",
            file=sys.stderr,
        )

        # Create a fake top-level module and mark it as a *package* so that
        # "import pytorch3d.renderer" and "import pytorch3d.transforms" work.
        p3d_mod = types.ModuleType("pytorch3d")
        p3d_mod.__all__ = ["transforms", "renderer"]  # type: ignore[attr-defined]
        p3d_mod.__path__ = []  # type: ignore[attr-defined]

        # -------------------------
        # transforms submodule stub
        # -------------------------
        transforms_mod = types.ModuleType("pytorch3d.transforms")

        def quaternion_multiply(q1, q2):
            """
            Broadcast-friendly quaternion multiplication.

            Expects tensors/arrays of shape (..., 4) with (w, x, y, z) layout,
            matching pytorch3d.transforms.
            """
            q1_t = torch.as_tensor(q1)
            q2_t = torch.as_tensor(q2)

            if q1_t.shape[-1] != 4 or q2_t.shape[-1] != 4:
                raise ValueError(
                    "quaternion_multiply expects inputs with last dimension 4 "
                    f"(got {q1_t.shape}, {q2_t.shape})"
                )

            w1, x1, y1, z1 = torch.unbind(q1_t, dim=-1)
            w2, x2, y2, z2 = torch.unbind(q2_t, dim=-1)

            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return torch.stack((w, x, y, z), dim=-1)

        def quaternion_invert(q, eps: float = 1e-8):
            """
            Quaternion inverse with basic numerical stability.

            Expects (..., 4) (w, x, y, z) quaternions. Matches the usual
            definition: q^{-1} = conjugate(q) / ||q||^2.
            """
            q_t = torch.as_tensor(q)

            if q_t.shape[-1] != 4:
                raise ValueError(
                    "quaternion_invert expects inputs with last dimension 4 "
                    f"(got {q_t.shape})"
                )

            w, x, y, z = torch.unbind(q_t, dim=-1)
            mag_sq = (w * w + x * x + y * y + z * z).clamp_min(eps)
            conj = torch.stack((w, -x, -y, -z), dim=-1)
            return conj / mag_sq.unsqueeze(-1)

        class Transform3d:
            """
            Very small subset of pytorch3d.transforms.Transform3d used by the
            SAM 3D inference pipeline.

            This implementation stores a 4x4 matrix and provides:

              - get_matrix()
              - compose(other)
              - inverse()
              - transform_points(points)
            """

            def __init__(self, matrix=None, device=None, dtype=torch.float32):
                if matrix is None:
                    m = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
                else:
                    m = torch.as_tensor(matrix, device=device, dtype=dtype)
                    if m.ndim == 2:
                        m = m.unsqueeze(0)
                self._matrix = m

            def get_matrix(self):
                return self._matrix

            def compose(self, other: "Transform3d"):
                """
                Compose this transform with another: result = other ∘ self.
                """
                if not isinstance(other, Transform3d):
                    raise TypeError("compose expects another Transform3d")
                m = other._matrix @ self._matrix
                return Transform3d(m)

            def inverse(self):
                m_inv = torch.inverse(self._matrix)
                return Transform3d(m_inv)

            def transform_points(self, points):
                """
                Apply the transform to 3D points.

                points: (..., 3)
                """
                pts = torch.as_tensor(
                    points,
                    dtype=self._matrix.dtype,
                    device=self._matrix.device,
                )
                orig_shape = pts.shape
                pts = pts.reshape(-1, 3)

                ones = torch.ones(
                    pts.shape[0], 1, dtype=pts.dtype, device=pts.device
                )
                homo = torch.cat([pts, ones], dim=-1)  # (N, 4)

                # For simplicity, use the first matrix if we have a batch.
                M = self._matrix[0]  # (4, 4)
                out = (M @ homo.T).T[..., :3]
                return out.reshape(orig_shape)

        transforms_mod.quaternion_multiply = quaternion_multiply
        transforms_mod.quaternion_invert = quaternion_invert
        transforms_mod.Transform3d = Transform3d

        # -------------------------
        # renderer submodule stub
        # -------------------------
        renderer_mod = types.ModuleType("pytorch3d.renderer")

        def look_at_view_transform(
            dist=1.0,
            elev=0.0,
            azim=0.0,
            at=None,
            up=None,
            eye=None,
            degrees: bool = True,
            device=None,
        ):
            """
            Minimal stand‑in for pytorch3d.renderer.look_at_view_transform.

            Returns (R, T) where:
              - R is a (1, 3, 3) rotation matrix
              - T is a (1, 3) translation vector

            For now we implement a simple "camera at distance dist looking at origin"
            parameterization. It won't match the real implementation exactly but is
            sufficient to keep the pipeline running.
            """
            device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

            if eye is not None:
                eye_t = torch.as_tensor(eye, dtype=torch.float32, device=device)
                if eye_t.ndim == 1:
                    eye_t = eye_t.unsqueeze(0)
            else:
                # Compute eye position from spherical coords (elev, azim).
                d = torch.as_tensor(dist, dtype=torch.float32, device=device)
                elev_t = torch.as_tensor(elev, dtype=torch.float32, device=device)
                azim_t = torch.as_tensor(azim, dtype=torch.float32, device=device)

                if degrees:
                    elev_t = elev_t * math.pi / 180.0
                    azim_t = azim_t * math.pi / 180.0

                # Simple spherical -> Cartesian, looking at origin.
                x = d * torch.cos(elev_t) * torch.sin(azim_t)
                y = d * torch.sin(elev_t)
                z = d * torch.cos(elev_t) * torch.cos(azim_t)
                eye_t = torch.stack([x, y, z], dim=-1)
                if eye_t.ndim == 1:
                    eye_t = eye_t.unsqueeze(0)

            # "at" (look-at target) defaults to origin
            if at is None:
                at_t = torch.zeros_like(eye_t)
            else:
                at_t = torch.as_tensor(at, dtype=torch.float32, device=device)
                if at_t.ndim == 1:
                    at_t = at_t.unsqueeze(0)

            # "up" vector defaults to +Y
            if up is None:
                up_t = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
                up_t = up_t.unsqueeze(0).expand_as(eye_t)
            else:
                up_t = torch.as_tensor(up, dtype=torch.float32, device=device)
                if up_t.ndim == 1:
                    up_t = up_t.unsqueeze(0)

            # Build a simple look-at rotation matrix.
            z_axis = eye_t - at_t
            z_axis = z_axis / (z_axis.norm(dim=-1, keepdim=True) + 1e-8)

            x_axis = torch.cross(up_t, z_axis, dim=-1)
            x_axis = x_axis / (x_axis.norm(dim=-1, keepdim=True) + 1e-8)

            y_axis = torch.cross(z_axis, x_axis, dim=-1)

            R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (B, 3, 3)
            T = -torch.bmm(R, eye_t.unsqueeze(-1)).squeeze(-1)  # (B, 3)

            return R, T

        renderer_mod.look_at_view_transform = look_at_view_transform

        # Wire modules together and register them.
        p3d_mod.transforms = transforms_mod
        p3d_mod.renderer = renderer_mod

        sys.modules["pytorch3d"] = p3d_mod
        sys.modules["pytorch3d.transforms"] = transforms_mod
        sys.modules["pytorch3d.renderer"] = renderer_mod

    except Exception as exc:  # pragma: no cover - best-effort safety net
        print(f"[SAM3D] WARNING: failed to register pytorch3d stub: {exc}", file=sys.stderr)


# Make sure the stub (or real pytorch3d) is available before importing Inference.
_ensure_pytorch3d_stub()

try:
    from inference import Inference
except Exception as e:  # pragma: no cover - defensive import
    print(f"[SAM3D] ERROR: failed to import SAM 3D inference utilities: {e}", file=sys.stderr)
    raise


def build_alpha_mask(crop: np.ndarray, background: int = 240) -> np.ndarray:
    gray = crop.mean(axis=-1)
    mask = (np.abs(gray - background) > 1).astype(np.uint8)
    return mask


def load_rgba_with_mask(image_path: Path, polygon: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an RGBA view for SAM 3D. We primarily rely on the background-masked
    crop produced upstream; if a polygon is provided and appears normalized,
    we fall back to the image-derived alpha to avoid coordinate mismatches.
    """
    image = Image.open(image_path).convert("RGB")
    rgb = np.array(image)
    mask = build_alpha_mask(rgb)
    return rgb, mask


def maybe_build_asset_plan(
    assets_root: Path,
    layout_prefix: Optional[str],
    multiview_prefix: Optional[str],
    scene_id: str,
    layout_file_name: str = "scene_layout_scaled.json",
) -> Optional[Path]:
    """Best-effort builder for scene_assets.json if none exists yet."""

    plan_path = assets_root / "scene_assets.json"
    if plan_path.is_file():
        return plan_path

    if not (layout_prefix and multiview_prefix):
        print(
            "[SAM3D] No scene_assets.json present and LAYOUT_PREFIX/MULTIVIEW_PREFIX missing; "
            "cannot auto-build asset plan.",
            file=sys.stderr,
        )
        return None

    root = Path("/mnt/gcs")
    layout_path = root / layout_prefix / layout_file_name
    multiview_root = root / multiview_prefix

    if not layout_path.is_file():
        print(f"[SAM3D] Cannot build asset plan; missing layout at {layout_path}", file=sys.stderr)
        return None

    if not multiview_root.is_dir():
        print(
            f"[SAM3D] Cannot build asset plan; multiview root missing at {multiview_root}",
            file=sys.stderr,
        )
        return None

    try:
        layout = json.loads(layout_path.read_text())
    except Exception as exc:  # pragma: no cover - runtime helper
        print(f"[SAM3D] Failed to load layout for asset plan: {exc}", file=sys.stderr)
        return None

    objects = layout.get("objects", [])
    entries = []

    for obj in objects:
        oid = obj.get("id")
        cls = obj.get("class_name", f"class_{obj.get('class_id', 0)}")
        if oid is None:
            continue

        mv_dir = multiview_root / f"obj_{oid}"
        crop_path = mv_dir / "crop.png"
        preferred_view = mv_dir / "view_0.png"

        if not (preferred_view.is_file() or crop_path.is_file()):
            print(
                f"[SAM3D] Skipping obj {oid}: no crop/view found under {mv_dir}",
                file=sys.stderr,
            )
            continue

        entry = {
            "id": oid,
            "class_name": cls,
            "type": "static",
            "pipeline": "sam3d",
            "multiview_dir": f"{multiview_prefix}/obj_{oid}",
            "crop_path": f"{multiview_prefix}/obj_{oid}/crop.png",
            "polygon": obj.get("polygon"),
        }
        if preferred_view.is_file():
            entry["preferred_view"] = f"{multiview_prefix}/obj_{oid}/view_0.png"

        entries.append(entry)

    if not entries:
        print(
            "[SAM3D] Asset plan auto-build produced zero entries; cannot continue.",
            file=sys.stderr,
        )
        return None

    assets_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "scene_id": scene_id,
        "objects": entries,
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2))
    print(f"[SAM3D] Auto-built scene_assets.json with {len(entries)} object(s) -> {plan_path}")
    return plan_path


def save_basecolor_texture(texture: np.ndarray, out_path: Path) -> Optional[Path]:
    """Persist a basecolor texture if the inference output provides one."""

    if texture is None:
        return None

    try:
        tex = np.asarray(texture)
        if tex.dtype != np.uint8:
            tex = np.clip(tex * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(tex)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        print(f"[SAM3D] Saved basecolor texture -> {out_path}")
        return out_path
    except Exception as e:  # pragma: no cover - best-effort export
        print(f"[SAM3D] WARNING: failed to save texture {out_path.name}: {e}", file=sys.stderr)
        return None


def getenv_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def mesh_bounds(mesh) -> Optional[dict]:
    """Return a simple bounding-box summary for a trimesh-like mesh."""

    try:
        bounds = getattr(mesh, "bounds", None)
        if bounds is None:
            return None
        arr = np.asarray(bounds, dtype=np.float64)
        if arr.shape != (2, 3):
            return None
        mn, mx = arr
        size = mx - mn
        center = (mx + mn) * 0.5
        return {
            "min": mn.tolist(),
            "max": mx.tolist(),
            "size": size.tolist(),
            "center": center.tolist(),
        }
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        print(f"[SAM3D] WARNING: failed to compute mesh bounds: {exc}", file=sys.stderr)
        return None


def normalize_mesh_inplace(mesh, bounds: dict) -> Optional[dict]:
    """
    Normalize a mesh so its center is at the origin and the largest dimension is 1.

    Returns a metadata dict describing the applied transform, or None if skipped.
    """

    size = np.array(bounds.get("size") or [], dtype=np.float64)
    if size.size != 3:
        return None

    max_extent = float(size.max())
    if max_extent <= 0:
        return None

    center = np.array(bounds.get("center"), dtype=np.float64)
    if center.size != 3:
        center = np.zeros(3, dtype=np.float64)

    try:
        mesh.apply_translation(-center)
        mesh.apply_scale(1.0 / max_extent)
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(f"[SAM3D] WARNING: failed to normalize mesh: {exc}", file=sys.stderr)
        return None

    return {
        "translation": (-center).tolist(),
        "scale": 1.0 / max_extent,
        "reference_extent": max_extent,
    }


def convert_glb_to_usdz(glb_path: Path, usdz_path: Path) -> bool:
    """Convert a GLB into USDZ using the usd_from_gltf CLI if available."""

    usd_from_gltf = shutil.which("usd_from_gltf")
    if usd_from_gltf is None:
        print("[SAM3D] WARNING: usd_from_gltf not available; skipping USDZ export", file=sys.stderr)
        return False

    usdz_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [usd_from_gltf, str(glb_path), "-o", str(usdz_path)]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print(f"[SAM3D] Saved USDZ -> {usdz_path}")
        return True
    except subprocess.CalledProcessError as e:  # pragma: no cover - runtime dependency
        print(
            f"[SAM3D] WARNING: usd_from_gltf failed for {glb_path}: {e.stderr or e}",
            file=sys.stderr,
        )
        return False


def download_sam3d_checkpoints(target_root: Path) -> Optional[Path]:
    """Fetch checkpoints from Hugging Face if they are not already present."""

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - optional dependency
        print(
            f"[SAM3D] WARNING: huggingface_hub not available, cannot download checkpoints ({exc})",
            file=sys.stderr,
        )
        return None

    allow_patterns = ["checkpoints/hf/**"]
    target_root.mkdir(parents=True, exist_ok=True)

    try:
        local_path = snapshot_download(
            repo_id=SAM3D_HF_REPO,
            repo_type="model",
            revision=SAM3D_HF_REVISION,
            allow_patterns=allow_patterns,
            local_dir=target_root.parent,
            local_dir_use_symlinks=False,
        )
        config_path = Path(local_path) / "checkpoints/hf/pipeline.yaml"
        if config_path.is_file():
            print(
                f"[SAM3D] Downloaded SAM 3D checkpoints from {SAM3D_HF_REPO} -> {config_path.parent}"
            )
            return config_path

        print(
            f"[SAM3D] WARNING: Download finished but pipeline.yaml not found under {config_path.parent}",
            file=sys.stderr,
        )
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(
            f"[SAM3D] WARNING: Failed to download checkpoints from {SAM3D_HF_REPO}: {exc}",
            file=sys.stderr,
        )

    return None


def validate_checkpoint_bundle(config_path: Path) -> bool:
    """Ensure the pipeline config and nearby weights are present."""

    if not config_path.is_file():
        print(
            f"[SAM3D] ERROR: pipeline config is missing: {config_path}", file=sys.stderr
        )
        return False

    weights_dir = config_path.parent
    weight_files = sorted(weights_dir.glob("*.ckpt")) + sorted(
        weights_dir.glob("*.safetensors")
    )
    if not weight_files:
        print(
            f"[SAM3D] ERROR: no weight files found next to {config_path}. "
            "Download checkpoints with HUGGINGFACE_TOKEN/HF_TOKEN set or provide a custom SAM3D_CONFIG_PATH.",
            file=sys.stderr,
        )
        return False

    print(
        f"[SAM3D] Found {len(weight_files)} checkpoint files under {weights_dir}"
    )
    return True


def main() -> None:
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    assets_prefix = os.getenv("ASSETS_PREFIX")  # scenes/<sceneId>/assets
    sam3d_config_path = os.getenv("SAM3D_CONFIG_PATH")
    layout_prefix = os.getenv("LAYOUT_PREFIX")
    multiview_prefix = os.getenv("MULTIVIEW_PREFIX")
    layout_file_name = os.getenv("LAYOUT_FILE_NAME", "scene_layout_scaled.json")
    normalize_meshes = getenv_bool("SAM3D_NORMALIZE_MESH", "0")

    if not assets_prefix:
        print("[SAM3D] ASSETS_PREFIX is required", file=sys.stderr)
        sys.exit(1)

    root = Path("/mnt/gcs")
    assets_root = root / assets_prefix
    plan_path = maybe_build_asset_plan(
        assets_root,
        layout_prefix=layout_prefix,
        multiview_prefix=multiview_prefix,
        scene_id=scene_id,
        layout_file_name=layout_file_name,
    )

    print(f"[SAM3D] Bucket={bucket}")
    print(f"[SAM3D] Scene={scene_id}")
    print(f"[SAM3D] Assets root={assets_root}")

    if plan_path is None or not plan_path.is_file():
        print(
            f"[SAM3D] ERROR: assets plan not found or failed to build: {plan_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = None
    if sam3d_config_path:
        candidate = Path(sam3d_config_path)
        if candidate.is_file():
            config_path = candidate
    if config_path is None:
        default_cfg = SAM3D_CHECKPOINT_ROOT / "hf/pipeline.yaml"
        legacy_cfg = Path("/mnt/gcs/sam3d/checkpoints/hf/pipeline.yaml")
        fallback_cfg = Path("/workspace/sam3d-objects/checkpoints/hf/pipeline.yaml")
        for candidate in (default_cfg, legacy_cfg, fallback_cfg):
            if candidate.is_file():
                config_path = candidate
                break

    if config_path is None or not config_path.is_file():
        print("[SAM3D] No local pipeline.yaml found; attempting download from Hugging Face")
        config_path = download_sam3d_checkpoints(SAM3D_CHECKPOINT_ROOT)

    if config_path is None or not validate_checkpoint_bundle(config_path):
        print(
            "[SAM3D] ERROR: SAM 3D checkpoints unavailable. Set SAM3D_CONFIG_PATH to a valid pipeline.yaml, "
            "or ensure HUGGINGFACE_TOKEN/HF_TOKEN is set so checkpoints can be downloaded.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[SAM3D] Using config: {config_path}")
    inference = Inference(str(config_path), compile=False)

    plan = json.loads(plan_path.read_text())
    objs = plan.get("objects", [])
    print(f"[SAM3D] Loaded plan with {len(objs)} objects")
    if normalize_meshes:
        print("[SAM3D] Mesh normalization enabled")

    for obj in objs:
        if obj.get("pipeline") != "sam3d":
            continue

        oid = obj.get("id")
        crop_rel = obj.get("crop_path")
        preferred_rel = obj.get("preferred_view")
        mv_rel = obj.get("multiview_dir")

        candidate_paths = []
        if preferred_rel:
            candidate_paths.append(root / preferred_rel)
        if mv_rel:
            candidate_paths.append(root / mv_rel / "view_0.png")
        if crop_rel:
            candidate_paths.append(root / crop_rel)

        crop_path = next((p for p in candidate_paths if p.is_file()), None)
        if crop_path is None:
            print(f"[SAM3D] WARNING: no crop or Gemini view found for obj {oid}", file=sys.stderr)
            continue

        print(f"[SAM3D] Reconstructing object {oid} from {crop_path}")
        rgb, mask = load_rgba_with_mask(crop_path)
        output = inference(rgb, mask, seed=42)

        out_dir = assets_root / f"obj_{oid}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if "gs" in output:
            output["gs"].save_ply(str(out_dir / "splat.ply"))
            print(f"[SAM3D] Saved gaussian splat for obj {oid}")

        mesh = output.get("mesh")
        if mesh is not None and hasattr(mesh, "export"):
            metadata = {}
            original_bounds = mesh_bounds(mesh)
            if original_bounds:
                metadata["mesh_bounds"] = {"original": original_bounds}
            if normalize_meshes and original_bounds:
                norm_info = normalize_mesh_inplace(mesh, original_bounds)
                if norm_info:
                    metadata["normalized"] = True
                    metadata["normalization"] = norm_info

            export_bounds = mesh_bounds(mesh)
            if export_bounds:
                metadata.setdefault("mesh_bounds", {})["export"] = export_bounds

            mesh_glb_path = out_dir / "mesh.glb"
            mesh.export(str(mesh_glb_path))
            print(f"[SAM3D] Saved mesh for obj {oid}")

            asset_glb_path = out_dir / "asset.glb"
            if not asset_glb_path.exists():
                shutil.copy(mesh_glb_path, asset_glb_path)
                print(f"[SAM3D] Saved mesh copy -> {asset_glb_path.name}")

            model_glb_path = out_dir / "model.glb"
            if not model_glb_path.exists():
                shutil.copy(asset_glb_path, model_glb_path)
                print(f"[SAM3D] Saved mesh copy -> {model_glb_path.name}")

            texture = output.get("texture") or output.get("texture_image")
            save_basecolor_texture(texture, out_dir / "texture_0_basecolor.png")

            if metadata:
                metadata_path = out_dir / "metadata.json"
                metadata_path.write_text(json.dumps(metadata, indent=2))
                print(f"[SAM3D] Wrote mesh metadata -> {metadata_path}")

            usdz_path = out_dir / "model.usdz"
            convert_glb_to_usdz(asset_glb_path, usdz_path)

    print("[SAM3D] Done.")


if __name__ == "__main__":
    main()
