from __future__ import annotations

from typing import Iterable, List, Optional, Sequence


DEFAULT_ROBOT_KEYWORDS = (
    "robot",
    "franka",
    "panda",
    "ur5",
    "ur10",
    "kinova",
    "fetch",
    "iiwa",
)


def _get_usd_stage(scene_usd_path: Optional[str] = None):
    try:
        from pxr import Usd
    except ImportError:
        return None

    if scene_usd_path:
        try:
            return Usd.Stage.Open(scene_usd_path)
        except Exception:
            return None

    try:
        import omni.usd

        return omni.usd.get_context().get_stage()
    except Exception:
        return None


def _iter_prims(stage) -> Iterable:
    for prim in stage.Traverse():
        yield prim


def discover_robot_prim_paths(
    scene_usd_path: Optional[str] = None,
    *,
    stage=None,
    keywords: Sequence[str] = DEFAULT_ROBOT_KEYWORDS,
) -> List[str]:
    """Discover robot prim paths from a USD stage or file."""
    stage = stage or _get_usd_stage(scene_usd_path)
    if stage is None:
        return []

    try:
        from pxr import UsdGeom, UsdPhysics
    except ImportError:
        UsdGeom = None
        UsdPhysics = None

    prim_paths: List[str] = []

    if UsdPhysics is not None:
        for prim in _iter_prims(stage):
            try:
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    prim_paths.append(str(prim.GetPath()))
            except Exception:
                continue

    if not prim_paths:
        keyword_set = {kw.lower() for kw in keywords}
        for prim in _iter_prims(stage):
            name = prim.GetName().lower()
            if not any(kw in name for kw in keyword_set):
                continue
            if UsdGeom is not None and not UsdGeom.Xformable(prim):
                continue
            prim_paths.append(str(prim.GetPath()))

    return sorted(set(prim_paths))


def resolve_robot_prim_paths(
    configured_paths: Optional[Sequence[str]],
    scene_usd_path: Optional[str] = None,
    *,
    stage=None,
    keywords: Sequence[str] = DEFAULT_ROBOT_KEYWORDS,
) -> List[str]:
    """Resolve robot prim paths from config or auto-discovery."""
    configured = [path for path in (configured_paths or []) if path]
    if configured:
        stage = stage or _get_usd_stage(scene_usd_path)
        if stage is None:
            return configured
        valid = []
        for path in configured:
            try:
                if stage.GetPrimAtPath(path).IsValid():
                    valid.append(path)
            except Exception:
                continue
        return valid

    return discover_robot_prim_paths(
        scene_usd_path=scene_usd_path,
        stage=stage,
        keywords=keywords,
    )
