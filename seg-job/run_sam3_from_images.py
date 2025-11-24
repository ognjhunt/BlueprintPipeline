import json
import os
import shutil
import subprocess
import sys
import math
import types
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
    cuda_home_default = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("CUDA_HOME", cuda_home_default)
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
    Register a minimal-but-complete-enough pytorch3d stub in sys.modules so
    SAM-3D can import the pieces it needs when the real library is unavailable.
    """
    try:
        import pytorch3d  # type: ignore  # noqa: F401
        # Real pytorch3d is available; nothing to do.
        return
    except Exception:
        pass

    try:
        import torch
    except Exception:
        # For safety: if torch is somehow missing (should not happen in your
        # container), we define a tiny dummy that is enough for import-time
        # usage. In the real container, torch is installed before this code.
        class _DummyTorch:
            float32 = "float32"
            int64 = "int64"

            def as_tensor(self, x, dtype=None, device=None):
                return x

            def eye(self, n, device=None, dtype=None):
                import numpy as _np
                return _np.eye(n, dtype=float)

            def ones(self, *shape, **kwargs):
                import numpy as _np
                return _np.ones(shape, dtype=float)

            def zeros_like(self, x):
                import numpy as _np
                return _np.zeros_like(x)

            def stack(self, xs, dim=0):
                import numpy as _np
                return _np.stack(xs, axis=dim)

            def unbind(self, x, dim=-1):
                import numpy as _np
                return [x.take(i, axis=dim) for i in range(x.shape[dim])]

            def norm(self, x, dim=-1, keepdim=False):
                import numpy as _np
                n = _np.linalg.norm(x, axis=dim)
                if keepdim:
                    n = _np.expand_dims(n, axis=dim)
                return n

            def sqrt(self, x):
                import numpy as _np
                return _np.sqrt(x)

            def cos(self, x):
                import numpy as _np
                return _np.cos(x)

            def sin(self, x):
                import numpy as _np
                return _np.sin(x)

            def cross(self, a, b, dim=-1):
                import numpy as _np
                return _np.cross(a, b)

            def bmm(self, a, b):
                import numpy as _np
                return _np.matmul(a, b)

            def tensor(self, x, dtype=None, device=None):
                return x

            def cat(self, xs, dim=0):
                import numpy as _np
                return _np.concatenate(xs, axis=dim)

        torch = _DummyTorch()  # type: ignore

    print(
        "[SAM3D] pytorch3d not found; installing minimal stub in sys.modules",
        file=sys.stderr,
    )

    # Create fake top-level package
    p3d_mod = types.ModuleType("pytorch3d")
    p3d_mod.__all__ = ["transforms", "renderer", "structures"]  # type: ignore[attr-defined]
    p3d_mod.__path__ = []  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # transforms submodule
    # ------------------------------------------------------------------
    transforms_mod = types.ModuleType("pytorch3d.transforms")

    def _ensure_tensor(x, dtype=getattr(torch, "float32", None), device=None):
        # type: ignore[arg-type]
        return torch.as_tensor(x, dtype=dtype, device=device)

    # --- Quaternion utilities ----------------------------------------

    def quaternion_multiply(q1, q2):
        q1_t = _ensure_tensor(q1)
        q2_t = _ensure_tensor(q2)

        w1, x1, y1, z1 = torch.unbind(q1_t, dim=-1)  # type: ignore[attr-defined]
        w2, x2, y2, z2 = torch.unbind(q2_t, dim=-1)  # type: ignore[attr-defined]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack((w, x, y, z), dim=-1)  # type: ignore[attr-defined]

    def quaternion_invert(q, eps: float = 1e-8):
        q_t = _ensure_tensor(q)
        w, x, y, z = torch.unbind(q_t, dim=-1)  # type: ignore[attr-defined]
        mag_sq = (w * w + x * x + y * y + z * z).clamp_min(eps)  # type: ignore[attr-defined]
        conj = torch.stack((w, -x, -y, -z), dim=-1)  # type: ignore[attr-defined]
        return conj / mag_sq.unsqueeze(-1)  # type: ignore[attr-defined]

    def standardize_quaternion(q, eps: float = 1e-8):
        """
        Make a quaternion have non‑negative real part.
        """
        q_t = _ensure_tensor(q)
        w = q_t[..., 0]
        # +1 where w>=0, -1 otherwise
        sign = (w >= 0).to(q_t.dtype) * 2 - 1  # type: ignore[attr-defined]
        return q_t * sign.unsqueeze(-1)  # type: ignore[attr-defined]

    def quaternion_to_matrix(q, eps: float = 1e-8):
        q_t = _ensure_tensor(q)
        # type: ignore[attr-defined]
        q_t = q_t / (q_t.norm(dim=-1, keepdim=True).clamp_min(eps))

        w, x, y, z = torch.unbind(q_t, dim=-1)  # type: ignore[attr-defined]

        ww = w * w
        xx = x * x
        yy = y * y
        zz = z * z

        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        m00 = ww + xx - yy - zz
        m01 = 2 * (xy - wz)
        m02 = 2 * (xz + wy)

        m10 = 2 * (xy + wz)
        m11 = ww - xx + yy - zz
        m12 = 2 * (yz - wx)

        m20 = 2 * (xz - wy)
        m21 = 2 * (yz + wx)
        m22 = ww - xx - yy + zz

        row0 = torch.stack([m00, m01, m02], dim=-1)  # type: ignore[attr-defined]
        row1 = torch.stack([m10, m11, m12], dim=-1)  # type: ignore[attr-defined]
        row2 = torch.stack([m20, m21, m22], dim=-1)  # type: ignore[attr-defined]
        return torch.stack([row0, row1, row2], dim=-2)  # type: ignore[attr-defined]

    def matrix_to_quaternion(matrix, eps: float = 1e-8):
        m = _ensure_tensor(matrix)
        if m.shape[-2:] != (3, 3):
            raise ValueError("matrix_to_quaternion expects (..., 3, 3) input")

        orig_shape = m.shape[:-2]
        m_flat = m.reshape(-1, 3, 3)
        q_out = []

        for i in range(m_flat.shape[0]):
            M = m_flat[i]
            m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
            m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
            m20, m21, m22 = M[2, 0], M[2, 1], M[2, 2]

            trace = m00 + m11 + m22

            if trace > 0.0:
                s = math.sqrt(float(trace + 1.0)) * 2.0
                qw = 0.25 * s
                qx = (m21 - m12) / (s + eps)
                qy = (m02 - m20) / (s + eps)
                qz = (m10 - m01) / (s + eps)
            elif m00 > m11 and m00 > m22:
                s = math.sqrt(float(1.0 + m00 - m11 - m22)) * 2.0
                qw = (m21 - m12) / (s + eps)
                qx = 0.25 * s
                qy = (m01 + m10) / (s + eps)
                qz = (m02 + m20) / (s + eps)
            elif m11 > m22:
                s = math.sqrt(float(1.0 + m11 - m00 - m22)) * 2.0
                qw = (m02 - m20) / (s + eps)
                qx = (m01 + m10) / (s + eps)
                qy = 0.25 * s
                qz = (m12 + m21) / (s + eps)
            else:
                s = math.sqrt(float(1.0 + m22 - m00 - m11)) * 2.0
                qw = (m10 - m01) / (s + eps)
                qx = (m02 + m20) / (s + eps)
                qy = (m12 + m21) / (s + eps)
                qz = 0.25 * s

            q_out.append(torch.stack([qw, qx, qy, qz]))  # type: ignore[attr-defined]

        q = torch.stack(q_out, dim=0)  # type: ignore[attr-defined]
        return q.reshape(*orig_shape, 4)

    def axis_angle_to_quaternion(aa, eps: float = 1e-8):
        aa_t = _ensure_tensor(aa)
        angle = aa_t.norm(dim=-1, keepdim=True).clamp_min(eps)  # type: ignore[attr-defined]
        axis = aa_t / angle
        half = 0.5 * angle
        sin_half = torch.sin(half)  # type: ignore[attr-defined]
        cos_half = torch.cos(half).squeeze(-1)  # type: ignore[attr-defined]
        x, y, z = torch.unbind(axis * sin_half, dim=-1)  # type: ignore[attr-defined]
        return torch.stack([cos_half, x, y, z], dim=-1)  # type: ignore[attr-defined]

    def quaternion_to_axis_angle(q, eps: float = 1e-8):
        # For the stub we keep this simple: we standardize the quaternion and
        # return an axis-angle vector pointing along +Z with the appropriate angle.
        q_t = standardize_quaternion(_ensure_tensor(q))
        w = q_t[..., 0].clamp(-1.0, 1.0)  # type: ignore[attr-defined]
        if hasattr(torch, "acos"):
            angle = 2.0 * torch.acos(w)  # type: ignore[attr-defined]
        else:
            angle = 0.0
        if hasattr(torch, "zeros_like"):
            zero = torch.zeros_like(w)  # type: ignore[attr-defined]
            one = torch.ones_like(w)    # type: ignore[attr-defined]
        else:
            zero = w * 0
            one = w * 0 + 1.0
        axis = torch.stack([zero, zero, one], dim=-1)  # type: ignore[attr-defined]
        return axis * angle.unsqueeze(-1)  # type: ignore[attr-defined]

    # --- Rotation helpers ----------------------------------------------------

    def axis_angle_to_matrix(aa, eps: float = 1e-8):
        """
        Convert axis‑angle vectors (..., 3) to rotation matrices (..., 3, 3).
        """
        aa_t = _ensure_tensor(aa)
        angle = aa_t.norm(dim=-1, keepdim=True).clamp_min(eps)  # type: ignore[attr-defined]
        axis = aa_t / angle

        x, y, z = torch.unbind(axis, dim=-1)  # type: ignore[attr-defined]
        c = torch.cos(angle)  # type: ignore[attr-defined]
        s = torch.sin(angle)  # type: ignore[attr-defined]
        C = 1.0 - c

        m00 = c + x * x * C
        m01 = x * y * C - z * s
        m02 = x * z * C + y * s

        m10 = y * x * C + z * s
        m11 = c + y * y * C
        m12 = y * z * C - x * s

        m20 = z * x * C - y * s
        m21 = z * y * C + x * s
        m22 = c + z * z * C

        row0 = torch.stack([m00, m01, m02], dim=-1)  # type: ignore[attr-defined]
        row1 = torch.stack([m10, m11, m12], dim=-1)  # type: ignore[attr-defined]
        row2 = torch.stack([m20, m21, m22], dim=-1)  # type: ignore[attr-defined]
        return torch.stack([row0, row1, row2], dim=-2)  # type: ignore[attr-defined]

    def euler_angles_to_matrix(euler_angles, convention="XYZ"):
        """
        Basic implementation supporting common conventions like 'XYZ'.
        Angles are in radians.
        """
        angles = _ensure_tensor(euler_angles)
        if angles.shape[-1] != 3:
            raise ValueError("euler_angles_to_matrix expects (..., 3) input")
        x, y, z = torch.unbind(angles, dim=-1)  # type: ignore[attr-defined]

        def _rot_x(a):
            ca = torch.cos(a); sa = torch.sin(a)  # type: ignore[attr-defined]
            row0 = torch.stack([torch.ones_like(ca), 0 * ca, 0 * ca], dim=-1)  # type: ignore[attr-defined]
            row1 = torch.stack([0 * ca, ca, -sa], dim=-1)  # type: ignore[attr-defined]
            row2 = torch.stack([0 * ca, sa, ca], dim=-1)  # type: ignore[attr-defined]
            return torch.stack([row0, row1, row2], dim=-2)  # type: ignore[attr-defined]

        def _rot_y(a):
            ca = torch.cos(a); sa = torch.sin(a)  # type: ignore[attr-defined]
            row0 = torch.stack([ca, 0 * ca, sa], dim=-1)  # type: ignore[attr-defined]
            row1 = torch.stack([0 * ca, torch.ones_like(ca), 0 * ca], dim=-1)  # type: ignore[attr-defined]
            row2 = torch.stack([-sa, 0 * ca, ca], dim=-1)  # type: ignore[attr-defined]
            return torch.stack([row0, row1, row2], dim=-2)  # type: ignore[attr-defined]

        def _rot_z(a):
            ca = torch.cos(a); sa = torch.sin(a)  # type: ignore[attr-defined]
            row0 = torch.stack([ca, -sa, 0 * ca], dim=-1)  # type: ignore[attr-defined]
            row1 = torch.stack([sa, ca, 0 * ca], dim=-1)  # type: ignore[attr-defined]
            row2 = torch.stack([0 * ca, 0 * ca, torch.ones_like(ca)], dim=-1)  # type: ignore[attr-defined]
            return torch.stack([row0, row1, row2], dim=-2)  # type: ignore[attr-defined]

        mats = {"X": _rot_x, "Y": _rot_y, "Z": _rot_z}
        R = None
        for axis, angle in zip(convention, (x, y, z)):
            Rx = mats[axis](angle)
            R = Rx if R is None else torch.bmm(R, Rx)  # type: ignore[attr-defined]
        return R

    def random_quaternions(n, device=None, dtype=getattr(torch, "float32", None)):
        """
        Very simple random unit quaternions.
        """
        import random
        qs = []
        for _ in range(int(n)):
            u1, u2, u3 = random.random(), random.random(), random.random()
            q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
            q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
            q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
            q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
            qs.append(torch.tensor([q4, q1, q2, q3], dtype=dtype, device=device))  # type: ignore[attr-defined]
        return torch.stack(qs, dim=0)  # type: ignore[attr-defined]

    def random_rotation(device=None, dtype=getattr(torch, "float32", None)):
        q = random_quaternions(1, device=device, dtype=dtype)
        return quaternion_to_matrix(q)[0]

    def random_rotations(n, device=None, dtype=getattr(torch, "float32", None)):
        q = random_quaternions(n, device=device, dtype=dtype)
        return quaternion_to_matrix(q)

    def rotation_6d_to_matrix(x):
        """
        Simplified 6D rotation representation to matrix, as in Zhou et al.
        x: (..., 6)
        """
        x_t = _ensure_tensor(x)
        a1 = x_t[..., 0:3]
        a2 = x_t[..., 3:6]

        b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # type: ignore[attr-defined]
        proj = (b1 * a2).sum(dim=-1, keepdim=True)  # type: ignore[attr-defined]
        b2 = a2 - proj * b1
        b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # type: ignore[attr-defined]
        b3 = torch.cross(b1, b2, dim=-1)  # type: ignore[attr-defined]

        row0 = b1
        row1 = b2
        row2 = b3
        return torch.stack([row0, row1, row2], dim=-2)  # type: ignore[attr-defined]

    # --- Transform3d and derived transforms --------------------------

    class Transform3d:
        """
        Small subset of pytorch3d.transforms.Transform3d:
        - stores batch of 4x4 matrices
        - get_matrix()
        - compose(other)
        - inverse()
        - transform_points(points)
        """

        def __init__(self, matrix=None, device=None, dtype=getattr(torch, "float32", None)):
            if matrix is None:
                m = torch.eye(4, device=device, dtype=dtype)  # type: ignore[attr-defined]
                m = m.reshape(1, 4, 4)
            else:
                m = _ensure_tensor(matrix, dtype=dtype, device=device)
                if len(getattr(m, "shape", ())) == 2:
                    m = m.reshape(1, *m.shape)
                shape = m.shape
                if shape[-2:] == (3, 3):
                    eye = torch.eye(4, device=device, dtype=dtype)  # type: ignore[attr-defined]
                    eye = eye.reshape(1, 4, 4)
                    eye = eye.repeat(shape[0], 1, 1)  # type: ignore[attr-defined]
                    eye[..., :3, :3] = m
                    m = eye
            self._matrix = m

        def get_matrix(self):
            return self._matrix

        def compose(self, other: "Transform3d"):
            m = other._matrix @ self._matrix
            return Transform3d(m)

        def inverse(self):
            try:
                inv = torch.inverse(self._matrix)  # type: ignore[attr-defined]
            except Exception:
                inv = self._matrix.transpose(-1, -2)  # type: ignore[attr-defined]
            return Transform3d(inv)

        def transform_points(self, points):
            pts = _ensure_tensor(points, dtype=getattr(self._matrix, "dtype", None))
            if len(getattr(pts, "shape", ())) > 2:
                orig = pts.shape
                pts = pts.reshape(-1, 3)
            else:
                orig = pts.shape

            ones = torch.ones(pts.shape[0], 1, dtype=getattr(self._matrix, "dtype", None))  # type: ignore[attr-defined]
            homo = torch.cat([pts, ones], dim=-1)  # type: ignore[attr-defined]

            M = self._matrix[0]
            out = (M @ homo.T).T[..., :3]
            return out.reshape(orig)

    class Rotate(Transform3d):
        def __init__(self, R=None, axis=None, angle=None, degrees=True, device=None, dtype=getattr(torch, "float32", None)):
            if R is not None:
                mat = R
            elif axis is not None and angle is not None:
                ang = angle
                if degrees:
                    ang = ang * math.pi / 180.0
                aa = _ensure_tensor(axis) * ang
                mat = axis_angle_to_matrix(aa)
            else:
                mat = None
            super().__init__(matrix=mat, device=device, dtype=dtype)

    class Translate(Transform3d):
        def __init__(self, t=None, device=None, dtype=getattr(torch, "float32", None)):
            if t is None:
                super().__init__(device=device, dtype=dtype)
                return
            t_t = _ensure_tensor(t, dtype=dtype, device=device)
            if len(getattr(t_t, "shape", ())) == 1:
                t_t = t_t.reshape(1, 3)
            eye = torch.eye(4, device=device, dtype=dtype)  # type: ignore[attr-defined]
            eye = eye.reshape(1, 4, 4)
            eye[..., :3, 3] = t_t
            super().__init__(matrix=eye, device=device, dtype=dtype)

    class Scale(Transform3d):
        def __init__(self, s=None, device=None, dtype=getattr(torch, "float32", None)):
            if s is None:
                super().__init__(device=device, dtype=dtype)
                return
            s_t = _ensure_tensor(s, dtype=dtype, device=device)
            if len(getattr(s_t, "shape", ())) == 0:
                s_t = s_t.reshape(1)
            eye = torch.eye(4, device=device, dtype=dtype)  # type: ignore[attr-defined]
            eye = eye.reshape(1, 4, 4)
            eye[..., 0, 0] = s_t[..., 0]
            eye[..., 1, 1] = s_t[..., -2]
            eye[..., 2, 2] = s_t[..., -1]
            super().__init__(matrix=eye, device=device, dtype=dtype)

    def Compose(transforms):
        """
        Compose a sequence of Transform3d objects (right‑to‑left).
        """
        out = Transform3d()
        for t in transforms:
            out = out.compose(t)
        return out

    # Register transform symbols
    for name, obj in {
        "quaternion_multiply": quaternion_multiply,
        "quaternion_invert": quaternion_invert,
        "quaternion_to_matrix": quaternion_to_matrix,
        "matrix_to_quaternion": matrix_to_quaternion,
        "axis_angle_to_quaternion": axis_angle_to_quaternion,
        "quaternion_to_axis_angle": quaternion_to_axis_angle,
        "axis_angle_to_matrix": axis_angle_to_matrix,
        "euler_angles_to_matrix": euler_angles_to_matrix,
        "random_quaternions": random_quaternions,
        "random_rotation": random_rotation,
        "random_rotations": random_rotations,
        "rotation_6d_to_matrix": rotation_6d_to_matrix,
        "standardize_quaternion": standardize_quaternion,
        "Transform3d": Transform3d,
        "Rotate": Rotate,
        "Translate": Translate,
        "Scale": Scale,
        "Compose": Compose,
    }.items():
        setattr(transforms_mod, name, obj)

    # ------------------------------------------------------------------
    # renderer submodule (look_at_view_transform only)
    # ------------------------------------------------------------------
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
        Returns (R, T): R (B,3,3), T (B,3)
        """
        import torch as _torch  # ensure real torch in container

        if device is None:
            device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

        if eye is not None:
            eye_t = _torch.as_tensor(eye, dtype=_torch.float32, device=device)
            if eye_t.ndim == 1:
                eye_t = eye_t.unsqueeze(0)
        else:
            d = _torch.as_tensor(dist, dtype=_torch.float32, device=device)
            elev_t = _torch.as_tensor(elev, dtype=_torch.float32, device=device)
            azim_t = _torch.as_tensor(azim, dtype=_torch.float32, device=device)

            if degrees:
                elev_t = elev_t * math.pi / 180.0
                azim_t = azim_t * math.pi / 180.0

            x = d * _torch.cos(elev_t)
            y = d * _torch.sin(elev_t)
            z = d * _torch.cos(elev_t) * 0 + d

            eye_t = _torch.stack([x, y, z], dim=-1)
            if eye_t.ndim == 1:
                eye_t = eye_t.unsqueeze(0)

        if at is None:
            at_t = _torch.zeros_like(eye_t)
        else:
            at_t = _torch.as_tensor(at, dtype=_torch.float32, device=device)
            if at_t.ndim == 1:
                at_t = at_t.unsqueeze(0)

        if up is None:
            up_t = _torch.tensor([0.0, 1.0, 0.0], dtype=_torch.float32, device=device)
            up_t = up_t.unsqueeze(0).expand_as(eye_t)
        else:
            up_t = _torch.as_tensor(up, dtype=_torch.float32, device=device)
            if up_t.ndim == 1:
                up_t = up_t.unsqueeze(0)

        z_axis = eye_t - at_t
        z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        x_axis = _torch.cross(up_t, z_axis, dim=-1)
        x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        y_axis = _torch.cross(z_axis, x_axis, dim=-1)

        row0 = x_axis
        row1 = y_axis
        row2 = z_axis
        R = _torch.stack([row0, row1, row2], dim=-2)

        T = -_torch.bmm(R, eye_t.unsqueeze(-1)).squeeze(-1)
        return R, T

    renderer_mod.look_at_view_transform = look_at_view_transform

    # ------------------------------------------------------------------
    # structures submodule (Meshes)
    # ------------------------------------------------------------------
    structures_mod = types.ModuleType("pytorch3d.structures")

    class Meshes:
        def __init__(self, verts=None, faces=None, textures=None):
            self._verts = verts
            self._faces = faces
            self._textures = textures

        @property
        def verts_list(self):
            return self._verts

        @property
        def faces_list(self):
            return self._faces

        @property
        def textures(self):
            return self._textures

        def verts_packed(self):
            if self._verts is None:
                return None
            if isinstance(self._verts, (list, tuple)):
                return torch.cat(self._verts, dim=0)  # type: ignore[attr-defined]
            return self._verts

        def faces_packed(self):
            if self._faces is None:
                return None
            if isinstance(self._faces, (list, tuple)):
                return torch.cat(self._faces, dim=0)  # type: ignore[attr-defined]
            return self._faces

        def num_verts_per_mesh(self):
            if self._verts is None:
                return None
            if isinstance(self._verts, (list, tuple)):
                return torch.tensor([v.shape[0] for v in self._verts], dtype=torch.int64)  # type: ignore[attr-defined]
            return torch.tensor([self._verts.shape[0]], dtype=torch.int64)  # type: ignore[attr-defined]

        def num_faces_per_mesh(self):
            if self._faces is None:
                return None
            if isinstance(self._faces, (list, tuple)):
                return torch.tensor([f.shape[0] for f in self._faces], dtype=torch.int64)  # type: ignore[attr-defined]
            return torch.tensor([self._faces.shape[0]], dtype=torch.int64)  # type: ignore[attr-defined]

    structures_mod.Meshes = Meshes

    # Wire submodules into top-level package and register everything.
    p3d_mod.transforms = transforms_mod  # type: ignore[attr-defined]
    p3d_mod.renderer = renderer_mod      # type: ignore[attr-defined]
    p3d_mod.structures = structures_mod  # type: ignore[attr-defined]

    sys.modules["pytorch3d"] = p3d_mod
    sys.modules["pytorch3d.transforms"] = transforms_mod
    sys.modules["pytorch3d.renderer"] = renderer_mod
    sys.modules["pytorch3d.structures"] = structures_mod


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
