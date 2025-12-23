"""
DWM Preparation Job - Dexterous World Model conditioning data generation.

This job generates conditioning inputs for DWM (Dexterous World Models)
from BlueprintPipeline scenes. DWM is a video diffusion model that
generates egocentric interaction videos given:
1. Static scene video (rendered from 3D scene along camera trajectory)
2. Hand mesh video (rendered hand meshes along same trajectory)
3. Text prompt (semantic description of action)

Reference:
- Paper: "DWM: Dexterous World Models" (arXiv:2512.17907)
- Project: https://snuvclab.github.io/dwm/

Usage:
    from dwm_preparation_job import prepare_dwm_bundles

    output = prepare_dwm_bundles(
        manifest_path=Path("scene_manifest.json"),
        scene_usd_path=Path("scene.usda"),
        output_dir=Path("dwm_output"),
        num_trajectories=5,
    )
"""

from .prepare_dwm_bundle import (
    DWMPreparationJob,
    prepare_dwm_bundles,
    run_dwm_preparation,
)

__all__ = [
    "DWMPreparationJob",
    "prepare_dwm_bundles",
    "run_dwm_preparation",
]
