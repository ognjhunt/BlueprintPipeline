"""
Flow Extractor module for Dream2Flow.

This module extracts 3D object flow from generated videos by:
1. Object masks - segmenting the object of interest
2. Video depth - estimating per-frame depth
3. Point tracking - tracking points across frames (CoTracker, TAPIR, etc.)
4. 3D lifting - projecting 2D tracks to 3D using depth and camera geometry

The extracted 3D object flow represents the motion of tracked points on the
object over time, which serves as the bridge between video generation and
robot control.

Reference: Dream2Flow (arXiv:2512.24766)
"""

from .flow_extractor import (
    FlowExtractor,
    FlowExtractorConfig,
    MockFlowExtractor,
    extract_object_flow,
)

__all__ = [
    "FlowExtractor",
    "FlowExtractorConfig",
    "MockFlowExtractor",
    "extract_object_flow",
]
