"""3D-RE-GEN to BlueprintPipeline adapter.

Converts 3D-RE-GEN reconstruction outputs into formats expected by downstream jobs.

3D-RE-GEN (arXiv:2512.17459) is a modular, compositional pipeline for
"image → sim-ready 3D reconstruction" with explicit physical constraints.

Key Features:
- Object separation via Grounded-SAM segmentation
- Occlusion handling via Application-Querying (A-Q)
- Camera + scene point cloud extraction via VGGT
- Per-object 2D→3D mesh generation via Hunyuan3D 2.0
- Differentiable scene assembly with PyTorch3D
- 4-DoF ground-alignment constraints for physics plausibility

Reference:
- Paper: https://arxiv.org/abs/2512.17459
- Project: https://3dregen.jdihlmann.com/
- GitHub: https://github.com/cgtuebingen/3D-RE-GEN

NOTE: 3D-RE-GEN code is pending public release (~Q1 2025).
This adapter is ready for integration once the code becomes available.
"""

from tools.regen3d_adapter.adapter import (
    Regen3DPose,
    Regen3DMaterial,
    Regen3DObject,
    Regen3DOutput,
    Regen3DAdapter,
    manifest_from_regen3d,
    layout_from_regen3d,
)

__all__ = [
    "Regen3DPose",
    "Regen3DMaterial",
    "Regen3DObject",
    "Regen3DOutput",
    "Regen3DAdapter",
    "manifest_from_regen3d",
    "layout_from_regen3d",
]
