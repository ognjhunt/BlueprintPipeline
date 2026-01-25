#!/usr/bin/env python3
"""
DWM Inference Job.

Consumes packaged DWM conditioning bundles, runs the released DWM model
API/checkpoint, and writes the generated interaction video + frames back
into each bundle directory.
"""

import argparse
import base64
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import imageio.v2 as imageio

from models import DWMConditioningBundle
from tools.config.constants import DEFAULT_HTTP_TIMEOUT_S

logger = logging.getLogger(__name__)


def _safe_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class DWMInferenceConfig:
    """Configuration for running DWM inference on prepared bundles."""

    bundles_dir: Path
    api_endpoint: Optional[str] = None
    checkpoint_path: Optional[Path] = None
    allow_placeholder: Optional[bool] = None
    save_frames: bool = True
    overwrite: bool = False
    fps_override: Optional[float] = None
    prompt_overlay: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        self.bundles_dir = Path(self.bundles_dir)
        if isinstance(self.checkpoint_path, str):
            self.checkpoint_path = Path(self.checkpoint_path)
        if self.allow_placeholder is None:
            self.allow_placeholder = _parse_bool_env("DWM_ALLOW_PLACEHOLDER", default=False)


@dataclass
class BundleInferenceResult:
    """Result of running inference on a single bundle."""

    bundle_id: str
    success: bool
    interaction_video: Optional[Path] = None
    interaction_frames_dir: Optional[Path] = None
    error: Optional[str] = None
    frames_written: int = 0


@dataclass
class DWMInferenceOutput:
    """Aggregate output for the inference job."""

    bundles_processed: list[BundleInferenceResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_path: Optional[Path] = None

    @property
    def success(self) -> bool:
        return all(result.success for result in self.bundles_processed) and not self.errors


class DWMModelClient:
    """Client wrapper around the DWM inference API or checkpoint."""

    def __init__(
        self,
        api_endpoint: Optional[str] = None,
        checkpoint_path: Optional[Path] = None,
        allow_placeholder: bool = False,
    ) -> None:
        self.api_endpoint = api_endpoint
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.allow_placeholder = allow_placeholder

    def generate_interaction(
        self,
        bundle: DWMConditioningBundle,
        output_video_path: Path,
        frames_dir: Optional[Path],
        overlay_prompt: bool = True,
    ) -> int:
        """
        Generate an interaction video for the given bundle.

        Returns:
            Number of frames written.
        """
        if self.api_endpoint:
            try:
                frames = self._call_remote_api(
                    static_scene_video=bundle.static_scene_video_path,
                    hand_mesh_video=bundle.hand_mesh_video_path,
                    text_prompt=bundle.text_prompt,
                    output_video_path=output_video_path,
                    overlay_prompt=overlay_prompt,
                    fps=bundle.fps,
                )
                if frames_dir:
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    for idx, frame in enumerate(frames):
                        imageio.imwrite(frames_dir / f"{idx:05d}.png", frame)
                return len(frames)
            except Exception:
                if not self.allow_placeholder:
                    raise
                # Fall through to placeholder generation
                traceback.print_exc()

        if self.checkpoint_path:
            # Placeholder: checkpoint inference can be wired here when available.
            if not self.allow_placeholder:
                raise RuntimeError(
                    "DWM checkpoint inference is not implemented; set DWM_ALLOW_PLACEHOLDER=true "
                    "to enable placeholder outputs."
                )

        if not self.allow_placeholder:
            raise RuntimeError(
                "DWM inference requires a working API endpoint or checkpoint. "
                "Set DWM_ALLOW_PLACEHOLDER=true to allow placeholder outputs."
            )

        return self._generate_placeholder(
            bundle=bundle,
            output_video_path=output_video_path,
            frames_dir=frames_dir,
            overlay_prompt=overlay_prompt,
        )

    def _call_remote_api(
        self,
        static_scene_video: Optional[Path],
        hand_mesh_video: Optional[Path],
        text_prompt: str,
        output_video_path: Path,
        overlay_prompt: bool,
        fps: float,
    ) -> list[np.ndarray]:
        """Call the remote DWM inference API and return frames."""
        import requests

        files = {}
        if static_scene_video and static_scene_video.exists():
            files["static_scene_video"] = (
                static_scene_video.name,
                static_scene_video.read_bytes(),
                "video/mp4",
            )
        if hand_mesh_video and hand_mesh_video.exists():
            files["hand_mesh_video"] = (
                hand_mesh_video.name,
                hand_mesh_video.read_bytes(),
                "video/mp4",
            )

        data = {"prompt": text_prompt}

        response = requests.post(
            self.api_endpoint,
            files=files,
            data=data,
            timeout=DEFAULT_HTTP_TIMEOUT_S,  # allow long-running DWM inference payloads
        )
        response.raise_for_status()

        video_bytes: Optional[bytes] = None
        content_type = response.headers.get("content-type", "")

        if content_type.startswith("application/json"):
            payload = response.json()
            if "video_base64" in payload:
                video_bytes = base64.b64decode(payload["video_base64"])
            elif "video_bytes" in payload:
                video_bytes = base64.b64decode(payload["video_bytes"])
            elif "video_url" in payload:
                video_url = payload["video_url"]
                download = requests.get(video_url, timeout=120)
                download.raise_for_status()
                video_bytes = download.content
        else:
            video_bytes = response.content

        if not video_bytes:
            raise RuntimeError("DWM API did not return video content")

        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        output_video_path.write_bytes(video_bytes)

        frames = _video_to_frames(output_video_path)

        if overlay_prompt:
            frames = [self._overlay_prompt(f, text_prompt) for f in frames]

        # Normalize FPS for downstream consumers by re-encoding with imageio
        writer = imageio.get_writer(output_video_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        return frames

    def _generate_placeholder(
        self,
        bundle: DWMConditioningBundle,
        output_video_path: Path,
        frames_dir: Optional[Path],
        overlay_prompt: bool,
    ) -> int:
        """Generate a placeholder interaction video when the API is unavailable."""
        base_frames = list(_iter_bundle_frames(bundle))
        if not base_frames:
            base_frames = [np.zeros((bundle.resolution[1], bundle.resolution[0], 3), dtype=np.uint8)]

        interaction_frames = []
        for idx, frame in enumerate(base_frames):
            if bundle.hand_mesh_video_path and bundle.hand_mesh_video_path.exists():
                try:
                    hand_reader = imageio.get_reader(bundle.hand_mesh_video_path)
                    hand_frame = hand_reader.get_data(min(idx, hand_reader.count_frames() - 1))
                    frame = _blend_frames(frame, hand_frame)
                    hand_reader.close()
                except Exception:
                    logger.warning(
                        "Failed to overlay hand mesh video frame (video=%s, frame_index=%s).",
                        bundle.hand_mesh_video_path,
                        idx,
                        exc_info=True,
                    )

            if overlay_prompt:
                frame = self._overlay_prompt(frame, bundle.text_prompt, idx)
            interaction_frames.append(frame)

        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(output_video_path, fps=bundle.fps)
        for frame in interaction_frames:
            writer.append_data(frame)
        writer.close()

        if frames_dir:
            frames_dir.mkdir(parents=True, exist_ok=True)
            for idx, frame in enumerate(interaction_frames):
                imageio.imwrite(frames_dir / f"{idx:05d}.png", frame)

        return len(interaction_frames)

    @staticmethod
    def _overlay_prompt(frame: np.ndarray, prompt: str, idx: int | None = None) -> np.ndarray:
        """Overlay the prompt text onto a frame."""
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        text = prompt if len(prompt) < 120 else prompt[:117] + "..."
        if idx is not None:
            text = f"DWM interaction frame {idx}: {text}"
        else:
            text = f"DWM interaction: {text}"

        font = ImageFont.load_default()
        margin = 10
        draw.rectangle(
            [
                (margin, image.height - 80),
                (image.width - margin, image.height - margin),
            ],
            fill=(0, 0, 0, 128),
        )
        draw.text((margin + 5, image.height - 70), text, fill=(255, 255, 255), font=font)
        return np.array(image)


def _video_to_frames(video_path: Path) -> list[np.ndarray]:
    """Decode a video into frames."""
    frames: list[np.ndarray] = []
    reader = imageio.get_reader(video_path)
    try:
        for frame in reader:
            frames.append(frame)
    finally:
        reader.close()
    return frames


def _iter_bundle_frames(bundle: DWMConditioningBundle):
    """Yield base frames from the bundle (static video or frames directory)."""
    if bundle.static_scene_frames_dir and bundle.static_scene_frames_dir.exists():
        for frame_path in sorted(bundle.static_scene_frames_dir.glob("*")):
            if frame_path.is_file():
                yield imageio.imread(frame_path)
        return

    if bundle.static_scene_video_path and bundle.static_scene_video_path.exists():
        reader = imageio.get_reader(bundle.static_scene_video_path)
        try:
            for frame in reader:
                yield frame
        finally:
            reader.close()


def _blend_frames(frame_a: np.ndarray, frame_b: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Blend two frames with simple alpha compositing."""
    try:
        frame_b_resized = Image.fromarray(frame_b).resize((frame_a.shape[1], frame_a.shape[0]))
        frame_b = np.array(frame_b_resized)
    except Exception:
        frame_b = frame_b
    blended = (frame_a.astype(np.float32) * (1 - alpha)) + (frame_b.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


class DWMInferenceJob:
    """Run DWM inference over a directory of bundled conditioning inputs."""

    def __init__(self, config: DWMInferenceConfig):
        self.config = config
        self.model_client = DWMModelClient(
            api_endpoint=config.api_endpoint,
            checkpoint_path=config.checkpoint_path,
            allow_placeholder=bool(config.allow_placeholder),
        )

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.config.verbose:
            print(f"[DWM-INFERENCE] [{level}] {msg}")

    def run(self) -> DWMInferenceOutput:
        bundles_dir = self.config.bundles_dir
        manifest_path = bundles_dir / "dwm_bundles_manifest.json"
        results: list[BundleInferenceResult] = []
        errors: list[str] = []

        if not self.config.api_endpoint and not self.config.checkpoint_path:
            raise ValueError(
                "DWM inference requires --api-endpoint or --checkpoint-path. "
                "Configure a backend before running."
            )

        self.log(
            "EXPERIMENTAL / PLACEHOLDER OUTPUT: DWM inference is experimental. "
            "Placeholder outputs are only generated when DWM_ALLOW_PLACEHOLDER=true.",
            level="WARNING",
        )

        bundle_dirs = self._discover_bundles(manifest_path)
        if not bundle_dirs:
            return DWMInferenceOutput(
                bundles_processed=[],
                errors=["No DWM bundles found"],
                manifest_path=manifest_path if manifest_path.exists() else None,
            )

        fps_override = self.config.fps_override or _safe_float_env("DWM_FPS", 0.0)

        for bundle_dir in bundle_dirs:
            try:
                bundle = self._load_bundle(bundle_dir)
                if fps_override:
                    bundle.fps = fps_override

                interaction_video_path = bundle_dir / "interaction_video.mp4"
                interaction_frames_dir = bundle_dir / "frames" / "interaction" if self.config.save_frames else None

                if interaction_video_path.exists() and not self.config.overwrite:
                    self.log(f"Skipping {bundle.bundle_id} (interaction video already exists)")
                    results.append(BundleInferenceResult(
                        bundle_id=bundle.bundle_id,
                        success=True,
                        interaction_video=interaction_video_path,
                        interaction_frames_dir=interaction_frames_dir,
                        frames_written=0,
                    ))
                    continue

                frame_count = self.model_client.generate_interaction(
                    bundle=bundle,
                    output_video_path=interaction_video_path,
                    frames_dir=interaction_frames_dir,
                    overlay_prompt=self.config.prompt_overlay,
                )

                bundle.interaction_video_path = interaction_video_path
                bundle.interaction_frames_dir = interaction_frames_dir
                self._update_bundle_manifest(bundle_dir, bundle)

                results.append(BundleInferenceResult(
                    bundle_id=bundle.bundle_id,
                    success=True,
                    interaction_video=interaction_video_path,
                    interaction_frames_dir=interaction_frames_dir,
                    frames_written=frame_count,
                ))
                self.log(f"Generated interaction video for {bundle.bundle_id} ({frame_count} frames)")
            except Exception as e:
                error_msg = f"{bundle_dir.name}: {e}"
                errors.append(error_msg)
                results.append(BundleInferenceResult(
                    bundle_id=bundle_dir.name,
                    success=False,
                    error=str(e),
                ))
                self.log(error_msg, level="ERROR")
                self.log(traceback.format_exc(), level="DEBUG")

        self._update_top_level_manifest(manifest_path, results)

        if self.config.overwrite or all(r.success for r in results):
            marker_path = bundles_dir / ".dwm_inference_complete"
            marker_path.write_text(json.dumps({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "bundles": [r.bundle_id for r in results],
            }, indent=2))

        return DWMInferenceOutput(
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

        # Fallback to directory listing
        return [
            p for p in self.config.bundles_dir.iterdir()
            if p.is_dir() and (p / "manifest.json").exists()
        ]

    def _load_bundle(self, bundle_dir: Path) -> DWMConditioningBundle:
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Bundle manifest missing: {manifest_path}")

        manifest = json.loads(manifest_path.read_text())
        bundle = DWMConditioningBundle(
            bundle_id=manifest.get("bundle_id", bundle_dir.name),
            scene_id=manifest.get("scene_id", "unknown"),
            camera_trajectory=None,  # Not needed for inference
            hand_trajectory=None,
            static_scene_video_path=(
                bundle_dir / manifest["static_scene_video"]
                if manifest.get("static_scene_video") else None
            ),
            hand_mesh_video_path=(
                bundle_dir / manifest["hand_mesh_video"]
                if manifest.get("hand_mesh_video") else None
            ),
            text_prompt=manifest.get("text_prompt", ""),
            static_scene_frames_dir=(
                bundle_dir / manifest["static_scene_frames_dir"]
                if manifest.get("static_scene_frames_dir") else None
            ),
            hand_mesh_frames_dir=(
                bundle_dir / manifest["hand_mesh_frames_dir"]
                if manifest.get("hand_mesh_frames_dir") else None
            ),
            resolution=tuple(manifest.get("resolution", (720, 480))),
            num_frames=manifest.get("num_frames", 0),
            fps=manifest.get("fps", 24.0),
            metadata=manifest,
        )

        return bundle

    def _update_bundle_manifest(self, bundle_dir: Path, bundle: DWMConditioningBundle) -> None:
        manifest_path = bundle_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["interaction_video"] = bundle.interaction_video_path.name if bundle.interaction_video_path else None
        manifest["interaction_frames_dir"] = (
            "frames/interaction" if bundle.interaction_frames_dir else None
        )
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def _update_top_level_manifest(
        self,
        manifest_path: Path,
        results: list[BundleInferenceResult],
    ) -> None:
        """Annotate the top-level bundles manifest with interaction outputs."""
        if not manifest_path.exists():
            return

        manifest = json.loads(manifest_path.read_text())
        result_lookup = {r.bundle_id: r for r in results}
        for bundle_entry in manifest.get("bundles", []):
            result = result_lookup.get(bundle_entry.get("bundle_id"))
            if result and result.success:
                bundle_entry["interaction_video"] = (
                    result.interaction_video.name if result.interaction_video else None
                )
                bundle_entry["interaction_frames_dir"] = (
                    "frames/interaction" if result.interaction_frames_dir else None
                )

        manifest_path.write_text(json.dumps(manifest, indent=2))


def run_dwm_inference(
    bundles_dir: Path,
    api_endpoint: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    allow_placeholder: Optional[bool] = None,
    save_frames: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> DWMInferenceOutput:
    """Convenience wrapper to run inference with sensible defaults."""
    config = DWMInferenceConfig(
        bundles_dir=bundles_dir,
        api_endpoint=api_endpoint,
        checkpoint_path=checkpoint_path,
        allow_placeholder=allow_placeholder,
        save_frames=save_frames,
        overwrite=overwrite,
        verbose=verbose,
    )
    job = DWMInferenceJob(config)
    return job.run()


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint for DWM inference."""
    parser = argparse.ArgumentParser(description="Run DWM inference job")
    parser.add_argument("--bundles-dir", type=Path, required=True, help="Bundles directory")
    parser.add_argument("--api-endpoint", type=str, default=None, help="DWM API endpoint")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Local checkpoint path")
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save interaction frames (default)",
    )
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Disable saving interaction frames",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing interaction outputs",
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

    save_frames = args.save_frames or not args.no_save_frames
    output = run_dwm_inference(
        bundles_dir=args.bundles_dir,
        api_endpoint=args.api_endpoint,
        checkpoint_path=args.checkpoint_path,
        save_frames=save_frames,
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
    cli_args.add_argument("--save-frames", action="store_true")
    cli_args.add_argument("--no-save-frames", action="store_true")
    cli_args.add_argument("--overwrite", action="store_true")
    cli_args.add_argument("--quiet", action="store_true")
    cli_args.add_argument("--scene-id", type=str, default=os.getenv("SCENE_ID", ""))
    parsed_args, _ = cli_args.parse_known_args()

    save_frames_value = parsed_args.save_frames or not parsed_args.no_save_frames
    input_params = {
        "bundles_dir": str(parsed_args.bundles_dir) if parsed_args.bundles_dir else None,
        "api_endpoint": parsed_args.api_endpoint,
        "checkpoint_path": str(parsed_args.checkpoint_path) if parsed_args.checkpoint_path else None,
        "save_frames": save_frames_value,
        "overwrite": parsed_args.overwrite,
        "verbose": not parsed_args.quiet,
        "scene_id": parsed_args.scene_id,
    }

    init_tracing(service_name=os.getenv("OTEL_SERVICE_NAME", "dwm-preparation-job"))
    exit_code = run_job_with_dead_letter_queue(
        lambda: main(),
        scene_id=parsed_args.scene_id,
        job_type="dwm_inference",
        step="dwm_inference",
        input_params=input_params,
    )
    sys.exit(exit_code)
