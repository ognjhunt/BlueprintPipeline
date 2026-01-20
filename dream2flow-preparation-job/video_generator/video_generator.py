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

import importlib
import inspect
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
from tools.config.constants import DEFAULT_HTTP_TIMEOUT_S


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
    # Feature flags
    enabled: bool = True
    allow_placeholder: bool = True
    require_real_backend: bool = False


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
        self._disabled_reason: Optional[str] = None

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.config.verbose:
            print(f"[VIDEO-GEN] [{level}] {msg}")

    def initialize(self) -> bool:
        """
        Initialize the video generation model.

        Returns True if initialization successful.
        """
        if self._initialized:
            return True
        if not self.config.enabled:
            self._disabled_reason = "Video generation disabled by configuration."
            self.log(self._disabled_reason, level="WARNING")
            self._initialized = True
            return False

        if not self.config.api_endpoint and not self.config.checkpoint_path:
            if self.config.require_real_backend or not self.config.allow_placeholder:
                self._disabled_reason = (
                    "Video generation disabled: no API endpoint or checkpoint configured "
                    "and placeholders are not allowed."
                )
                self.log(self._disabled_reason, level="ERROR")
                self._initialized = True
                return False
            self.log("Video generator initialized in placeholder mode (no backend configured)", level="WARNING")
            self._initialized = True
            return True

        try:
            if self.config.checkpoint_path:
                if not self.config.checkpoint_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {self.config.checkpoint_path}")
                self.model = self._load_local_model(self.config.checkpoint_path)
                if self.model is None:
                    raise RuntimeError("Unable to load Dream2Flow model from checkpoint.")
                self.log("Video generator initialized with local Dream2Flow checkpoint")
            else:
                self.log("Video generator initialized with API backend")
            self._initialized = True
            return True
        except (ImportError, FileNotFoundError, RuntimeError) as e:
            if self.config.allow_placeholder and not self.config.require_real_backend:
                self.log(f"Dream2Flow backend not available: {e}", level="WARNING")
                self._initialized = True  # Still proceed in placeholder mode
                return True
            self._disabled_reason = f"Video generation disabled: {e}"
            self.log(self._disabled_reason, level="ERROR")
            self._initialized = True
            return False

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
        if self._disabled_reason:
            self.log(self._disabled_reason, level="ERROR")
            return GeneratedVideo(
                video_id=video_id or f"video_{uuid.uuid4().hex[:8]}",
                instruction=instruction,
                initial_observation=observation,
                resolution=self.config.resolution,
                num_frames=0,
                fps=self.config.fps,
                model_name=self.config.model_name,
                quality_score=0.0,
                metadata={"error": self._disabled_reason, "disabled": True},
            )

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
            timeout=DEFAULT_HTTP_TIMEOUT_S,  # allow long-running video generation
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
        if self.model is None:
            raise RuntimeError(
                "Local video generation is disabled: no Dream2Flow model available. "
                "Provide a valid checkpoint and integration or set allow_placeholder=True."
            )
        self.log("Generating via local Dream2Flow checkpoint...")
        rgb, depth = self._load_observation_arrays(observation)
        generation_inputs = {
            "rgb": rgb,
            "depth": depth,
            "instruction": instruction.text,
            "num_frames": self.config.num_frames,
            "fps": self.config.fps,
            "guidance_scale": self.config.guidance_scale,
            "num_inference_steps": self.config.num_inference_steps,
            "seed": self.config.seed,
            "resolution": self.config.resolution,
        }

        result = self._invoke_local_model(generation_inputs)
        return self._save_model_output(
            result=result,
            observation=observation,
            instruction=instruction,
            output_dir=output_dir,
            video_id=video_id,
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

    def _load_local_model(self, checkpoint_path: Path) -> Any:
        """Load a local Dream2Flow model checkpoint using a compatible backend."""
        model_module = self.config.model_name
        module_candidates = []
        if model_module and model_module not in {"placeholder", "none"}:
            module_candidates.append(model_module)
        module_candidates.extend(["dream2flow", "dream2flow.video", "dream2flow.video_generator"])

        last_error = None
        for module_name in module_candidates:
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                last_error = exc
                continue

            model_cls = None
            for attr in (
                "Dream2FlowVideoGenerator",
                "Dream2FlowVideoPipeline",
                "VideoGenerator",
                "Dream2Flow",
            ):
                if hasattr(module, attr):
                    model_cls = getattr(module, attr)
                    break

            if model_cls is None:
                continue

            if hasattr(model_cls, "from_pretrained"):
                model = model_cls.from_pretrained(str(checkpoint_path))
            elif hasattr(model_cls, "from_checkpoint"):
                model = model_cls.from_checkpoint(str(checkpoint_path))
            elif hasattr(model_cls, "load_from_checkpoint"):
                model = model_cls.load_from_checkpoint(str(checkpoint_path))
            else:
                model = model_cls(str(checkpoint_path))

            if hasattr(model, "eval"):
                model.eval()
            return model

        if last_error:
            raise ImportError(f"Unable to import Dream2Flow backend: {last_error}") from last_error
        raise ImportError("No compatible Dream2Flow backend found.")

    def _invoke_local_model(self, inputs: dict[str, Any]) -> Any:
        """Invoke the local Dream2Flow model with flexible signatures."""
        if hasattr(self.model, "generate"):
            method = self.model.generate
        elif callable(self.model):
            method = self.model
        else:
            raise RuntimeError("Dream2Flow model does not expose a callable interface.")

        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            signature = None

        if signature is None:
            return method(**inputs)

        supported = {}
        for name in signature.parameters:
            if name in inputs:
                supported[name] = inputs[name]
        if "instruction" in signature.parameters and "instruction" not in supported:
            supported["instruction"] = inputs["instruction"]
        if "text" in signature.parameters and "instruction" in inputs:
            supported["text"] = inputs["instruction"]
        if "rgbd" in signature.parameters and inputs.get("rgb") is not None and inputs.get("depth") is not None:
            supported["rgbd"] = np.concatenate(
                [inputs["rgb"], inputs["depth"][..., None].astype(inputs["rgb"].dtype)],
                axis=-1,
            )

        return method(**supported)

    def _save_model_output(
        self,
        result: Any,
        observation: RGBDObservation,
        instruction: TaskInstruction,
        output_dir: Path,
        video_id: str,
    ) -> GeneratedVideo:
        """Persist model output into video/frames and build GeneratedVideo."""
        frames = None
        video_bytes = None
        video_path = None

        if isinstance(result, dict):
            frames = result.get("frames") or result.get("video")
            video_path = result.get("video_path")
            video_bytes = result.get("video_bytes")
        elif isinstance(result, (list, tuple, np.ndarray)):
            frames = result
        elif isinstance(result, (bytes, bytearray)):
            video_bytes = result

        if video_path:
            video_path = Path(video_path)
        else:
            video_path = output_dir / f"{video_id}.mp4"

        frames_dir = output_dir / f"{video_id}_frames"
        frames_dir.mkdir(exist_ok=True)

        if frames is not None:
            frame_list = self._normalize_frames(frames)
            for idx, frame in enumerate(frame_list):
                frame_path = frames_dir / f"frame_{idx:04d}.png"
                frame.save(frame_path)
            if imageio is not None:
                writer = imageio.get_writer(video_path, fps=self.config.fps)
                for frame in frame_list:
                    writer.append_data(np.array(frame))
                writer.close()
        elif video_bytes is not None:
            video_path.write_bytes(video_bytes)
            self._extract_frames(video_path, frames_dir)
        elif video_path.exists():
            self._extract_frames(video_path, frames_dir)
        else:
            raise RuntimeError("Dream2Flow model did not return frames or video output.")

        num_frames = len(list(frames_dir.glob("frame_*.png")))
        return GeneratedVideo(
            video_id=video_id,
            video_path=video_path,
            frames_dir=frames_dir,
            resolution=self.config.resolution,
            num_frames=num_frames,
            fps=self.config.fps,
            instruction=instruction,
            initial_observation=observation,
            model_name=self.config.model_name,
            quality_score=0.8,
            metadata={
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "backend": "local",
            },
        )

    def _normalize_frames(self, frames: Any) -> list[Image.Image]:
        """Normalize model frames into PIL Images."""
        if isinstance(frames, np.ndarray):
            if frames.ndim == 4:
                frames = list(frames)
            else:
                frames = [frames]

        normalized = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                normalized.append(frame.resize(self.config.resolution, Image.Resampling.LANCZOS))
            else:
                frame_array = np.asarray(frame)
                if frame_array.dtype != np.uint8:
                    frame_array = np.clip(frame_array * 255, 0, 255).astype(np.uint8)
                normalized.append(Image.fromarray(frame_array).resize(
                    self.config.resolution, Image.Resampling.LANCZOS
                ))
        return normalized

    def _load_observation_arrays(self, observation: RGBDObservation) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load RGB and depth arrays from observation."""
        rgb = None
        depth = None
        if observation.rgb is not None:
            rgb = observation.rgb
        elif observation.rgb_path and observation.rgb_path.exists():
            rgb = np.array(Image.open(observation.rgb_path))

        if observation.depth is not None:
            depth = observation.depth
        elif observation.depth_path and observation.depth_path.exists():
            depth_image = Image.open(observation.depth_path)
            depth = np.array(depth_image).astype(np.float32)
        return rgb, depth


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
