"""
Video Generator for Dream2Flow.

Generates "dreamed" task execution videos from RGB-D observations and
natural language instructions using video diffusion models.

Based on Dream2Flow (arXiv:2512.24766):
- Input: RGB-D observation + task instruction
- Output: Video showing imagined task execution

Key insight from the paper:
Video models trained on human videos can generate plausible object motion
even when the robot depiction is imperfect. The object motion is what
matters for downstream flow extraction.

Note: This module provides scaffolding for when the Dream2Flow model
is publicly released. Currently uses placeholder/mock generation.
"""

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    GeneratedVideo,
    RGBDObservation,
    TaskInstruction,
    TaskType,
)


@dataclass
class VideoGeneratorConfig:
    """Configuration for video generation."""

    # Output resolution
    resolution: tuple[int, int] = (720, 480)

    # Number of frames to generate
    num_frames: int = 49

    # Target FPS
    fps: float = 24.0

    # Model to use (placeholder until Dream2Flow model is released)
    model_name: str = "placeholder"

    # API endpoint (for remote inference)
    api_endpoint: Optional[str] = None

    # Local checkpoint path
    checkpoint_path: Optional[Path] = None

    # Generation parameters
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    seed: Optional[int] = None

    # Quality settings
    high_quality: bool = False

    # Debug
    verbose: bool = True
    save_debug_info: bool = False


class VideoGenerator:
    """
    Base class for video generation.

    Generates "dreamed" videos of task execution from initial observations
    and language instructions.
    """

    def __init__(self, config: VideoGeneratorConfig):
        self.config = config
        self.model = None
        self._initialized = False

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.config.verbose:
            print(f"[VIDEO-GEN] [{level}] {msg}")

    def initialize(self) -> bool:
        """
        Initialize the video generation model.

        Returns True if initialization successful.
        Currently returns True as placeholder.
        """
        if self._initialized:
            return True

        # TODO: Initialize actual model when Dream2Flow is released
        # Placeholder: model will be loaded here
        self.log("Video generator initialized (placeholder mode)")
        self._initialized = True
        return True

    def generate(
        self,
        observation: RGBDObservation,
        instruction: TaskInstruction,
        output_dir: Path,
        video_id: Optional[str] = None,
    ) -> GeneratedVideo:
        """
        Generate a task execution video.

        Args:
            observation: Initial RGB-D observation of the scene
            instruction: Natural language task instruction
            output_dir: Directory to save outputs
            video_id: Optional identifier for the video

        Returns:
            GeneratedVideo with generated content
        """
        if not self._initialized:
            self.initialize()

        video_id = video_id or f"video_{uuid.uuid4().hex[:8]}"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.log(f"Generating video for: '{instruction.text}'")

        try:
            # Try API-based generation first
            if self.config.api_endpoint:
                return self._generate_via_api(
                    observation=observation,
                    instruction=instruction,
                    output_dir=output_dir,
                    video_id=video_id,
                )

            # Fall back to local model
            if self.model is not None:
                return self._generate_local(
                    observation=observation,
                    instruction=instruction,
                    output_dir=output_dir,
                    video_id=video_id,
                )

            # Placeholder generation
            return self._generate_placeholder(
                observation=observation,
                instruction=instruction,
                output_dir=output_dir,
                video_id=video_id,
            )

        except Exception as e:
            self.log(f"Video generation failed: {e}", "ERROR")
            traceback.print_exc()
            # Return failed video
            return GeneratedVideo(
                video_id=video_id,
                instruction=instruction,
                initial_observation=observation,
                resolution=self.config.resolution,
                num_frames=0,
                fps=self.config.fps,
                model_name=self.config.model_name,
                quality_score=0.0,
                metadata={"error": str(e)},
            )

    def _generate_via_api(
        self,
        observation: RGBDObservation,
        instruction: TaskInstruction,
        output_dir: Path,
        video_id: str,
    ) -> GeneratedVideo:
        """Generate video via remote API call."""
        import requests
        import base64

        self.log("Generating via API...")

        # Prepare request
        files = {}
        if observation.rgb is not None:
            rgb_bytes = Image.fromarray(observation.rgb)
            import io
            buffer = io.BytesIO()
            rgb_bytes.save(buffer, format="PNG")
            files["image"] = ("observation.png", buffer.getvalue(), "image/png")
        elif observation.rgb_path and observation.rgb_path.exists():
            files["image"] = (
                observation.rgb_path.name,
                observation.rgb_path.read_bytes(),
                "image/png",
            )

        if observation.depth is not None:
            # Normalize and encode depth
            depth_normalized = (observation.depth * 1000).astype(np.uint16)
            import io
            buffer = io.BytesIO()
            Image.fromarray(depth_normalized).save(buffer, format="PNG")
            files["depth"] = ("depth.png", buffer.getvalue(), "image/png")
        elif observation.depth_path and observation.depth_path.exists():
            files["depth"] = (
                observation.depth_path.name,
                observation.depth_path.read_bytes(),
                "image/png",
            )

        data = {
            "instruction": instruction.text,
            "num_frames": self.config.num_frames,
            "guidance_scale": self.config.guidance_scale,
        }
        if self.config.seed is not None:
            data["seed"] = self.config.seed

        # Call API
        response = requests.post(
            self.config.api_endpoint,
            files=files,
            data=data,
            timeout=300,
        )
        response.raise_for_status()

        # Parse response
        result = response.json()
        video_bytes = None

        if "video_base64" in result:
            video_bytes = base64.b64decode(result["video_base64"])
        elif "video_url" in result:
            video_response = requests.get(result["video_url"], timeout=120)
            video_response.raise_for_status()
            video_bytes = video_response.content

        # Save video
        video_path = output_dir / f"{video_id}.mp4"
        if video_bytes:
            video_path.write_bytes(video_bytes)
        else:
            raise RuntimeError("API did not return video content")

        # Extract frames
        frames_dir = output_dir / f"{video_id}_frames"
        frames_dir.mkdir(exist_ok=True)
        frames = self._extract_frames(video_path, frames_dir)

        return GeneratedVideo(
            video_id=video_id,
            video_path=video_path,
            frames_dir=frames_dir,
            resolution=self.config.resolution,
            num_frames=len(frames),
            fps=self.config.fps,
            instruction=instruction,
            initial_observation=observation,
            model_name=result.get("model", self.config.model_name),
            quality_score=result.get("quality_score", 0.8),
            has_morphing_artifacts=result.get("has_morphing_artifacts", False),
            has_hallucinations=result.get("has_hallucinations", False),
            metadata={
                "api_response": result,
                "generated_at": datetime.utcnow().isoformat() + "Z",
            },
        )

    def _generate_local(
        self,
        observation: RGBDObservation,
        instruction: TaskInstruction,
        output_dir: Path,
        video_id: str,
    ) -> GeneratedVideo:
        """Generate video using local model checkpoint."""
        # TODO: Implement when Dream2Flow model is released
        raise NotImplementedError(
            "Local model generation not yet implemented. "
            "Waiting for Dream2Flow model release."
        )

    def _generate_placeholder(
        self,
        observation: RGBDObservation,
        instruction: TaskInstruction,
        output_dir: Path,
        video_id: str,
    ) -> GeneratedVideo:
        """
        Generate placeholder video for testing/scaffolding.

        Creates a simple video with the initial observation and instruction overlay.
        """
        self.log("Generating placeholder video (model not available)", "WARN")

        # Get base image
        if observation.rgb is not None:
            base_image = Image.fromarray(observation.rgb)
        elif observation.rgb_path and observation.rgb_path.exists():
            base_image = Image.open(observation.rgb_path)
        else:
            # Create blank image
            base_image = Image.new(
                "RGB",
                self.config.resolution,
                color=(40, 40, 60),
            )

        # Resize to target resolution
        base_image = base_image.resize(self.config.resolution, Image.Resampling.LANCZOS)

        # Generate frames with simulated motion
        frames_dir = output_dir / f"{video_id}_frames"
        frames_dir.mkdir(exist_ok=True)

        frames = []
        for i in range(self.config.num_frames):
            # Create frame with slight variation to simulate motion
            frame = self._create_placeholder_frame(
                base_image=base_image,
                instruction=instruction,
                frame_idx=i,
                total_frames=self.config.num_frames,
            )
            frames.append(frame)

            # Save frame
            frame_path = frames_dir / f"frame_{i:04d}.png"
            frame.save(frame_path)

        # Create video
        video_path = output_dir / f"{video_id}.mp4"
        if imageio is not None:
            writer = imageio.get_writer(video_path, fps=self.config.fps)
            for frame in frames:
                writer.append_data(np.array(frame))
            writer.close()

        return GeneratedVideo(
            video_id=video_id,
            video_path=video_path if imageio is not None else None,
            frames_dir=frames_dir,
            resolution=self.config.resolution,
            num_frames=len(frames),
            fps=self.config.fps,
            instruction=instruction,
            initial_observation=observation,
            model_name="placeholder",
            quality_score=0.5,  # Placeholder has lower quality
            metadata={
                "placeholder": True,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "note": "Placeholder video - Dream2Flow model not yet available",
            },
        )

    def _create_placeholder_frame(
        self,
        base_image: Image.Image,
        instruction: TaskInstruction,
        frame_idx: int,
        total_frames: int,
    ) -> Image.Image:
        """Create a single placeholder frame with instruction overlay."""
        # Copy base image
        frame = base_image.copy()
        draw = ImageDraw.Draw(frame)

        # Progress indicator
        progress = frame_idx / max(total_frames - 1, 1)

        # Add instruction overlay at bottom
        margin = 10
        box_height = 60
        draw.rectangle(
            [
                margin,
                frame.height - box_height - margin,
                frame.width - margin,
                frame.height - margin,
            ],
            fill=(0, 0, 0, 180),
        )

        # Instruction text
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        instruction_text = instruction.text
        if len(instruction_text) > 80:
            instruction_text = instruction_text[:77] + "..."

        draw.text(
            (margin + 10, frame.height - box_height - margin + 10),
            f"Dream2Flow: {instruction_text}",
            fill=(255, 255, 255),
            font=font,
        )

        # Frame counter and progress
        draw.text(
            (margin + 10, frame.height - margin - 20),
            f"Frame {frame_idx + 1}/{total_frames} | Progress: {progress:.0%}",
            fill=(200, 200, 200),
            font=font,
        )

        # Placeholder indicator at top
        draw.rectangle([margin, margin, frame.width - margin, margin + 30], fill=(60, 60, 80))
        draw.text(
            (margin + 10, margin + 5),
            "Dream2Flow Placeholder (model pending release)",
            fill=(255, 200, 100),
            font=font,
        )

        # Simulated motion: slightly shift/transform to indicate progression
        # In real implementation, this would be actual generated video

        return frame

    def _extract_frames(self, video_path: Path, frames_dir: Path) -> list[np.ndarray]:
        """Extract frames from video file."""
        if imageio is None:
            return []

        frames = []
        reader = imageio.get_reader(video_path)
        try:
            for i, frame in enumerate(reader):
                frames.append(frame)
                frame_path = frames_dir / f"frame_{i:04d}.png"
                imageio.imwrite(frame_path, frame)
        finally:
            reader.close()

        return frames


class MockVideoGenerator(VideoGenerator):
    """
    Mock video generator for testing and CI.

    Generates simple placeholder videos without requiring the actual model.
    """

    def __init__(self, config: Optional[VideoGeneratorConfig] = None):
        config = config or VideoGeneratorConfig(model_name="mock")
        super().__init__(config)

    def initialize(self) -> bool:
        self.log("Mock video generator initialized")
        self._initialized = True
        return True

    def generate(
        self,
        observation: RGBDObservation,
        instruction: TaskInstruction,
        output_dir: Path,
        video_id: Optional[str] = None,
    ) -> GeneratedVideo:
        """Generate mock video for testing."""
        return self._generate_placeholder(
            observation=observation,
            instruction=instruction,
            output_dir=output_dir,
            video_id=video_id or f"mock_{uuid.uuid4().hex[:8]}",
        )


def generate_task_video(
    rgb_image: np.ndarray,
    depth_image: Optional[np.ndarray],
    instruction_text: str,
    output_dir: Path,
    config: Optional[VideoGeneratorConfig] = None,
) -> GeneratedVideo:
    """
    Convenience function to generate a task video.

    Args:
        rgb_image: RGB observation as numpy array (H, W, 3)
        depth_image: Optional depth map as numpy array (H, W)
        instruction_text: Natural language instruction
        output_dir: Directory to save outputs
        config: Optional generator configuration

    Returns:
        GeneratedVideo with results
    """
    config = config or VideoGeneratorConfig()

    observation = RGBDObservation(
        rgb=rgb_image,
        depth=depth_image,
    )

    instruction = TaskInstruction(text=instruction_text)

    generator = VideoGenerator(config)
    return generator.generate(
        observation=observation,
        instruction=instruction,
        output_dir=output_dir,
    )
