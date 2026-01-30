"""
Scene rendering for DWM conditioning.

Renders static 3D scenes along camera trajectories to produce
the static scene video used as DWM conditioning input.
"""

from .static_scene_renderer import (
    RenderBackend,
    RenderConfig,
    SceneRenderer,
    render_trajectory_to_video,
)

__all__ = [
    "RenderBackend",
    "RenderConfig",
    "SceneRenderer",
    "render_trajectory_to_video",
]
