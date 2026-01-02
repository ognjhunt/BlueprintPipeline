"""
Video Generator module for Dream2Flow.

This module generates "dreamed" videos of task execution from RGB-D observations
and natural language instructions using video diffusion models.

Dream2Flow uses image-to-video models to imagine what the task completion
looks like. The generated video is then processed to extract 3D object flow.

Note: The actual video generation model (from arXiv:2512.24766) is not yet
publicly released. This module provides scaffolding that will be updated
when the model becomes available.
"""

from .video_generator import (
    VideoGenerator,
    VideoGeneratorConfig,
    MockVideoGenerator,
    generate_task_video,
)

__all__ = [
    "VideoGenerator",
    "VideoGeneratorConfig",
    "MockVideoGenerator",
    "generate_task_video",
]
