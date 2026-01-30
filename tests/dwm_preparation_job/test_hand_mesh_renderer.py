import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DWM_ROOT = REPO_ROOT / "dwm-preparation-job"


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Save state before loading dwm modules (they pollute sys.path and sys.modules)
_pre_keys = set(sys.modules)
_saved_path = sys.path[:]

renderer = _load_module(
    "dwm_hand_mesh_renderer",
    DWM_ROOT / "hand_motion" / "hand_mesh_renderer.py",
)
models = _load_module(
    "dwm_models",
    DWM_ROOT / "models.py",
)

# Clean up sys.modules pollution (dwm scripts add their dir to sys.path and cache
# a generic "models" module that conflicts with other jobs' models.py)
for _k in list(sys.modules):
    if _k not in _pre_keys and _k not in ("dwm_hand_mesh_renderer", "dwm_models"):
        del sys.modules[_k]
sys.path[:] = _saved_path


@pytest.mark.unit
def test_simple_hand_mesh_applies_pose():
    mesh = renderer.SimpleHandMesh()
    base_vertices = mesh.vertices.copy()
    hand_pose = models.HandPose(
        frame_idx=0,
        position=np.array([1.0, 2.0, 3.0]),
        rotation=np.eye(3),
    )

    transformed = mesh.get_vertices(hand_pose)

    assert transformed.shape == base_vertices.shape
    assert np.allclose(transformed, base_vertices + hand_pose.position)


@pytest.mark.unit
def test_create_hand_mesh_fallback_and_require_mano(monkeypatch):
    class FakeMANO:
        def __init__(self, *args, **kwargs):
            raise renderer.MANOUnavailableError("missing")

    monkeypatch.setattr(renderer, "MANOHandMesh", FakeMANO)

    mesh = renderer.create_hand_mesh(renderer.HandModel.MANO, require_mano=False)
    assert isinstance(mesh, renderer.SimpleHandMesh)

    with pytest.raises(renderer.MANOUnavailableError):
        renderer.create_hand_mesh(renderer.HandModel.MANO, require_mano=True)
