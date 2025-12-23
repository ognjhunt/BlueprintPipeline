"""
Scene rendering for DWM conditioning.

Renders static 3D scenes along camera trajectories to produce
the static scene video used as DWM conditioning input.
"""

from .static_scene_renderer import (
    RenderBackend,
    SceneRenderer,
    render_trajectory_to_video,
)

__all__ = [
    "RenderBackend",
    "SceneRenderer",
    "render_trajectory_to_video",
]
