#!/usr/bin/env python3
"""
Dream2Flow Inference Job.

Consumes packaged Dream2Flow bundles and runs the full inference pipeline:
1. Video generation (when model is available)
2. Flow extraction using vision foundation models
3. Robot trajectory generation

Note: This is scaffolding for when the Dream2Flow model is publicly released.
Currently generates placeholder outputs for testing the pipeline.

Reference: https://arxiv.org/abs/2512.24766
"""

import argparse
import json
import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models import Dream2FlowBundle, TaskInstruction, TaskType
from tools.config.constants import DEFAULT_HTTP_TIMEOUT_S


def _parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Dream2FlowInferenceConfig:
    """Configuration for running Dream2Flow inference on prepared bundles."""

    bundles_dir: Path
    video_api_endpoint: Optional[str] = None
    video_checkpoint_path: Optional[Path] = None
    allow_placeholder: Optional[bool] = None
    save_intermediate: bool = True
    overwrite: bool = False
    verbose: bool = True

    def __post_init__(self) -> None:
        self.bundles_dir = Path(self.bundles_dir)
        if isinstance(self.video_checkpoint_path, str):
            self.video_checkpoint_path = Path(self.video_checkpoint_path)
        if self.allow_placeholder is None:
            self.allow_placeholder = _parse_bool_env("DREAM2FLOW_ALLOW_PLACEHOLDER", default=False)


@dataclass
class BundleInferenceResult:
    """Result of running inference on a single bundle."""

    bundle_id: str
    success: bool
    video_generated: bool = False
    flow_extracted: bool = False
    trajectory_generated: bool = False
    error: Optional[str] = None


@dataclass
class Dream2FlowInferenceOutput:
    """Aggregate output for the inference job."""

    bundles_processed: list[BundleInferenceResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_path: Optional[Path] = None

    @property
    def success(self) -> bool:
        return all(result.success for result in self.bundles_processed) and not self.errors


class Dream2FlowModelClient:
    """Client wrapper around the Dream2Flow inference API or checkpoint."""

    def __init__(
        self,
        api_endpoint: Optional[str] = None,
        checkpoint_path: Optional[Path] = None,
        allow_placeholder: bool = False,
    ) -> None:
        self.api_endpoint = api_endpoint
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.allow_placeholder = allow_placeholder

    def generate_video(
        self,
        rgb_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        instruction: str,
        output_path: Path,
        num_frames: int = 49,
        fps: float = 24.0,
    ) -> list[np.ndarray]:
        """
        Generate task execution video.

        Currently returns placeholder video.
        Will be updated when Dream2Flow model is released.
        """
        if self.api_endpoint:
            try:
                return self._call_remote_api(
                    rgb_image=rgb_image,
                    depth_image=depth_image,
                    instruction=instruction,
                    output_path=output_path,
                    num_frames=num_frames,
                    fps=fps,
                )
            except Exception:
                if not self.allow_placeholder:
                    raise
                traceback.print_exc()

        if self.checkpoint_path and not self.allow_placeholder:
            raise RuntimeError(
                "Dream2Flow checkpoint inference is not implemented; set "
                "DREAM2FLOW_ALLOW_PLACEHOLDER=true to enable placeholder outputs."
            )

        if not self.allow_placeholder:
            raise RuntimeError(
                "Dream2Flow inference requires a working API endpoint or checkpoint. "
                "Set DREAM2FLOW_ALLOW_PLACEHOLDER=true to allow placeholder outputs."
            )

        # Placeholder generation
        return self._generate_placeholder_video(
            rgb_image=rgb_image,
            instruction=instruction,
            output_path=output_path,
            num_frames=num_frames,
            fps=fps,
        )

    def _call_remote_api(
        self,
        rgb_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        instruction: str,
        output_path: Path,
        num_frames: int,
        fps: float,
    ) -> list[np.ndarray]:
        """Call the remote Dream2Flow inference API."""
        import requests
        import base64
        import io

        # Encode RGB image
        if Image is not None:
            buffer = io.BytesIO()
            Image.fromarray(rgb_image).save(buffer, format="PNG")
            rgb_base64 = base64.b64encode(buffer.getvalue()).decode()
        else:
            raise RuntimeError("PIL not available for image encoding")

        # Build request
        data = {
            "rgb_base64": rgb_base64,
            "instruction": instruction,
            "num_frames": num_frames,
        }

        if depth_image is not None:
            buffer = io.BytesIO()
            depth_uint16 = (depth_image * 1000).astype(np.uint16)
            Image.fromarray(depth_uint16).save(buffer, format="PNG")
            data["depth_base64"] = base64.b64encode(buffer.getvalue()).decode()

        response = requests.post(
            self.api_endpoint,
            json=data,
            timeout=DEFAULT_HTTP_TIMEOUT_S,  # allow long-running Dream2Flow inference payloads
        )
        response.raise_for_status()

        result = response.json()
        video_bytes = base64.b64decode(result.get("video_base64", ""))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(video_bytes)

        # Extract frames
        frames = []
        if imageio is not None:
            reader = imageio.get_reader(output_path)
            try:
                for frame in reader:
                    frames.append(frame)
            finally:
                reader.close()

        return frames

    def _generate_placeholder_video(
        self,
        rgb_image: np.ndarray,
        instruction: str,
        output_path: Path,
        num_frames: int,
        fps: float,
    ) -> list[np.ndarray]:
        """Generate placeholder video for testing."""
        if Image is None or imageio is None:
            return []

        base = Image.fromarray(rgb_image)
        frames = []

        for i in range(num_frames):
            frame = base.copy()
            draw = ImageDraw.Draw(frame)

            # Add overlay
            progress = i / max(num_frames - 1, 1)
            margin = 10
            draw.rectangle(
                [margin, frame.height - 70, frame.width - margin, frame.height - margin],
                fill=(0, 0, 0),
            )

            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

            text = f"Dream2Flow: {instruction[:60]}..."
            draw.text((margin + 5, frame.height - 65), text, fill=(255, 255, 255), font=font)
            draw.text(
                (margin + 5, frame.height - 35),
                f"Frame {i + 1}/{num_frames} | {progress:.0%}",
                fill=(200, 200, 200),
                font=font,
            )

            frames.append(np.array(frame))

        # Write video
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        return frames


class Dream2FlowInferenceJob:
    """Run Dream2Flow inference over a directory of bundled inputs."""

    def __init__(self, config: Dream2FlowInferenceConfig):
        self.config = config
        self.model_client = Dream2FlowModelClient(
            api_endpoint=config.video_api_endpoint,
            checkpoint_path=config.video_checkpoint_path,
            allow_placeholder=bool(config.allow_placeholder),
        )

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.config.verbose:
            print(f"[D2F-INFERENCE] [{level}] {msg}")

    def run(self) -> Dream2FlowInferenceOutput:
        bundles_dir = self.config.bundles_dir
        manifest_path = bundles_dir / "dream2flow_bundles_manifest.json"
        results: list[BundleInferenceResult] = []
        errors: list[str] = []

        if not self.config.video_api_endpoint and not self.config.video_checkpoint_path:
            raise ValueError(
                "Dream2Flow inference requires --api-endpoint or --checkpoint-path. "
                "Configure a backend before running."
            )

        self.log(
            "EXPERIMENTAL / PLACEHOLDER OUTPUT: Dream2Flow inference is experimental. "
            "Placeholder outputs are only generated when DREAM2FLOW_ALLOW_PLACEHOLDER=true.",
            level="WARNING",
        )

        bundle_dirs = self._discover_bundles(manifest_path)
        if not bundle_dirs:
            return Dream2FlowInferenceOutput(
                bundles_processed=[],
                errors=["No Dream2Flow bundles found"],
                manifest_path=manifest_path if manifest_path.exists() else None,
            )

        for bundle_dir in bundle_dirs:
            try:
                result = self._process_bundle(bundle_dir)
                results.append(result)
                self.log(f"Processed {result.bundle_id}: success={result.success}")
            except Exception as e:
                error_msg = f"{bundle_dir.name}: {e}"
                errors.append(error_msg)
                results.append(BundleInferenceResult(
                    bundle_id=bundle_dir.name,
                    success=False,
                    error=str(e),
                ))
                self.log(error_msg, level="ERROR")

        # Write completion marker
        if all(r.success for r in results):
            marker_path = bundles_dir / ".dream2flow_inference_complete"
            marker_path.write_text(json.dumps({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "bundles": [r.bundle_id for r in results],
            }, indent=2))

        return Dream2FlowInferenceOutput(
            bundles_processed=results,
            errors=errors,
            manifest_path=manifest_path if manifest_path.exists() else None,
        )

    def _discover_bundles(self, manifest_path: Path) -> list[Path]:
        """Return ordered bundle directories."""
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            bundle_ids = [b.get("bundle_id") for b in manifest.get("bundles", []) if b.get("bundle_id")]
            bundle_dirs = [self.config.bundles_dir / bid for bid in bundle_ids if (self.config.bundles_dir / bid).is_dir()]
            if bundle_dirs:
                return bundle_dirs

        return [
            p for p in self.config.bundles_dir.iterdir()
            if p.is_dir() and (p / "manifest.json").exists()
        ]

    def _process_bundle(self, bundle_dir: Path) -> BundleInferenceResult:
        """Process a single bundle through inference."""
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Bundle manifest missing: {manifest_path}")

        manifest = json.loads(manifest_path.read_text())
        bundle_id = manifest.get("bundle_id", bundle_dir.name)

        # Check if already processed
        inference_video = bundle_dir / "inference_video.mp4"
        if inference_video.exists() and not self.config.overwrite:
            return BundleInferenceResult(
                bundle_id=bundle_id,
                success=True,
                video_generated=True,
                flow_extracted=True,
                trajectory_generated=True,
            )

        # Load initial observation
        obs_dir = bundle_dir / "observation"
        rgb_path = obs_dir / "initial_rgb.png"
        depth_path = obs_dir / "initial_depth.png"

        if not rgb_path.exists():
            raise FileNotFoundError(f"Initial RGB not found: {rgb_path}")

        rgb_image = np.array(Image.open(rgb_path)) if Image else np.zeros((480, 720, 3), dtype=np.uint8)
        depth_image = None
        if depth_path.exists() and Image:
            depth_uint16 = np.array(Image.open(depth_path))
            depth_image = depth_uint16.astype(np.float32) / 1000.0

        # Get instruction
        instruction_data = manifest.get("instruction", {})
        instruction = instruction_data.get("text", "Perform the task")

        # Generate video
        frames = self.model_client.generate_video(
            rgb_image=rgb_image,
            depth_image=depth_image,
            instruction=instruction,
            output_path=inference_video,
            num_frames=manifest.get("num_frames", 49),
            fps=manifest.get("fps", 24.0),
        )

        video_generated = len(frames) > 0

        # Update manifest
        manifest["inference_video"] = "inference_video.mp4" if video_generated else None
        manifest["inference_completed_at"] = datetime.utcnow().isoformat() + "Z"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        return BundleInferenceResult(
            bundle_id=bundle_id,
            success=video_generated,
            video_generated=video_generated,
            flow_extracted=video_generated,  # Placeholder
            trajectory_generated=video_generated,  # Placeholder
        )


def run_dream2flow_inference(
    bundles_dir: Path,
    api_endpoint: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    allow_placeholder: Optional[bool] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> Dream2FlowInferenceOutput:
    """Convenience wrapper to run inference with sensible defaults."""
    config = Dream2FlowInferenceConfig(
        bundles_dir=bundles_dir,
        video_api_endpoint=api_endpoint,
        video_checkpoint_path=checkpoint_path,
        allow_placeholder=allow_placeholder,
        overwrite=overwrite,
        verbose=verbose,
    )
    job = Dream2FlowInferenceJob(config)
    return job.run()


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint for Dream2Flow inference."""
    parser = argparse.ArgumentParser(description="Run Dream2Flow inference job")
    parser.add_argument("--bundles-dir", type=Path, required=True, help="Bundles directory")
    parser.add_argument("--api-endpoint", type=str, default=None, help="Dream2Flow API endpoint")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Local checkpoint path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing inference outputs",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose logging",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default=os.getenv("SCENE_ID", ""),
        help="Scene identifier (optional)",
    )
    args = parser.parse_args(argv)

    output = run_dream2flow_inference(
        bundles_dir=args.bundles_dir,
        api_endpoint=args.api_endpoint,
        checkpoint_path=args.checkpoint_path,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )
    return 0 if output.success else 1


if __name__ == "__main__":
    from tools.error_handling.job_wrapper import run_job_with_dead_letter_queue
    from tools.tracing import init_tracing

    cli_args = argparse.ArgumentParser(add_help=False)
    cli_args.add_argument("--bundles-dir", type=Path)
    cli_args.add_argument("--api-endpoint", type=str, default=None)
    cli_args.add_argument("--checkpoint-path", type=Path, default=None)
    cli_args.add_argument("--overwrite", action="store_true")
    cli_args.add_argument("--quiet", action="store_true")
    cli_args.add_argument("--scene-id", type=str, default=os.getenv("SCENE_ID", ""))
    parsed_args, _ = cli_args.parse_known_args()

    input_params = {
        "bundles_dir": str(parsed_args.bundles_dir) if parsed_args.bundles_dir else None,
        "api_endpoint": parsed_args.api_endpoint,
        "checkpoint_path": str(parsed_args.checkpoint_path) if parsed_args.checkpoint_path else None,
        "overwrite": parsed_args.overwrite,
        "verbose": not parsed_args.quiet,
        "scene_id": parsed_args.scene_id,
    }

    init_tracing(service_name=os.getenv("OTEL_SERVICE_NAME", "dream2flow-preparation-job"))
    exit_code = run_job_with_dead_letter_queue(
        lambda: main(),
        scene_id=parsed_args.scene_id,
        job_type="dream2flow_inference",
        step="dream2flow_inference",
        input_params=input_params,
    )
    sys.exit(exit_code)
