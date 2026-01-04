"""
UltraShape-1.0 Integration for BlueprintPipeline

UltraShape is a scalable two-stage diffusion framework for high-quality 3D geometry generation.
This module provides integration with the BlueprintPipeline variation asset generation system.

Reference: https://github.com/PKU-YuanGroup/UltraShape-1.0
Paper: arXiv:2512.21185
"""

from .ultrashape_refiner import (
    UltraShapeRefiner,
    UltraShapeConfig,
    download_ultrashape_weights,
    clone_ultrashape_repo,
)

__all__ = [
    "UltraShapeRefiner",
    "UltraShapeConfig",
    "download_ultrashape_weights",
    "clone_ultrashape_repo",
]

__version__ = "1.0.0"
