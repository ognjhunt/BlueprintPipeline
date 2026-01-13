"""
Flow Extractor for Dream2Flow.

Extracts 3D object flow from generated task videos using:
1. Object segmentation (SAM, Grounded-SAM, etc.)
2. Depth estimation (DepthAnything, ZoeDepth, etc.)
3. Point tracking (CoTracker, TAPIR, RAFT, etc.)
4. 3D projection (using camera intrinsics/extrinsics)

Based on Dream2Flow (arXiv:2512.24766):
- Extract geometry + motion cues from generated video
- Track points across frames
- Lift 2D tracks to 3D using depth and camera geometry
- Output: 3D object flow (tracked point trajectories)

Flow extraction failures are the second major bottleneck in the Dream2Flow
pipeline (after video generation artifacts), per the paper's failure analysis.
"""

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import uuid

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    FlowExtractionMethod,
    FlowExtractionResult,
    GeneratedVideo,
    ObjectFlow3D,
    ObjectMask,
    TaskInstruction,
    TrackedPoint,
)


@dataclass
class FlowExtractorConfig:
    """Configuration for flow extraction."""

    # Extraction method
    method: FlowExtractionMethod = FlowExtractionMethod.POINT_TRACKING_WITH_DEPTH

    # Number of points to track on the object
    num_tracking_points: int = 100

    # Point sampling strategy: "grid", "random", "edge", "saliency"
    point_sampling: str = "grid"

    # Segmentation model
    segmentation_model: str = "sam"  # Options: sam, grounded_sam, mask_rcnn
    segmentation_prompt: Optional[str] = None  # For text-guided segmentation

    # Depth estimation model
    depth_model: str = "depth_anything"  # Options: depth_anything, zoedepth, midas

    # Point tracking model
    tracking_model: str = "cotracker"  # Options: cotracker, tapir, raft

    # Camera parameters (default pinhole model)
    camera_focal_length: float = 500.0  # pixels
    camera_cx: Optional[float] = None  # Principal point x (default: width/2)
    camera_cy: Optional[float] = None  # Principal point y (default: height/2)

    # Quality thresholds
    min_visibility_ratio: float = 0.5  # Minimum fraction of frames point must be visible
    min_confidence: float = 0.5  # Minimum tracking confidence

    # API endpoints (for remote inference)
    segmentation_api: Optional[str] = None
    depth_api: Optional[str] = None
    tracking_api: Optional[str] = None

    # Output options
    save_intermediate: bool = True
    save_visualizations: bool = True

    # Debug
    verbose: bool = True
    # Feature flags
    enabled: bool = True
    allow_placeholder: bool = True


class FlowExtractor:
    """
    Extracts 3D object flow from generated videos.

    Pipeline:
    1. Segment the manipulated object
    2. Estimate depth for each frame
    3. Track points on the object across frames
    4. Lift 2D tracks to 3D using depth and camera geometry
    """

    def __init__(self, config: FlowExtractorConfig):
        self.config = config
        self._segmentation_model = None
        self._depth_model = None
        self._tracking_model = None
        self._initialized = False
        self._disabled_reason: Optional[str] = None

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.config.verbose:
            print(f"[FLOW-EXTRACT] [{level}] {msg}")

    def initialize(self) -> bool:
        """
        Initialize models for flow extraction.

        Returns True if initialization successful.
        Currently returns True as placeholder.
        """
        if self._initialized:
            return True
        if not self.config.enabled:
            self._disabled_reason = "Flow extraction disabled by configuration."
            self.log(self._disabled_reason, level="WARNING")
            self._initialized = True
            return False

        # PLACEHOLDER: Awaiting Dream2Flow model release (arXiv:2512.24766)
        # When available, implement integration with:
        # - Segmentation: SAM / Grounded-SAM for semantic object segmentation
        # - Depth estimation: DepthAnything / ZoeDepth for dense depth prediction
        # - Point tracking: CoTracker / TAPIR for robust optical flow estimation
        # - 3D reconstruction: 3D-RE-GEN (arXiv:2512.17459) for SDF scene representation
        # Expected workflow:
        # 1. Load and initialize each foundation model
        # 2. Create inference pipelines with batching support
        # 3. Validate compatibility between model outputs
        # 4. Set up GPU memory management
        try:
            if self.config.segmentation_api:
                self._segmentation_model = "api"
            if self.config.depth_api:
                self._depth_model = "api"
            if self.config.tracking_api:
                self._tracking_model = "api"

            if not any([self._segmentation_model, self._depth_model, self._tracking_model]):
                if not self.config.allow_placeholder:
                    self._disabled_reason = (
                        "Flow extraction disabled: no model backends configured "
                        "(set segmentation_api/depth_api/tracking_api)."
                    )
                    self.log(self._disabled_reason, level="ERROR")
                    self._initialized = True
                    return False
            if not self.config.allow_placeholder and not all(
                [self._segmentation_model, self._depth_model, self._tracking_model]
            ):
                self._disabled_reason = (
                    "Flow extraction disabled: missing backends for "
                    "segmentation, depth, or tracking."
                )
                self.log(self._disabled_reason, level="ERROR")
                self._initialized = True
                return False

            self.log("Flow extractor initialized")
            if self.config.allow_placeholder and not all([self._segmentation_model, self._depth_model, self._tracking_model]):
                self.log(
                    "Flow extractor running with placeholders for missing backends.",
                    level="WARNING",
                )
            self._initialized = True
            return True
        except ImportError as e:
            if self.config.allow_placeholder:
                self.log(f"Dream2Flow not available: {e}", level="WARNING")
                self._initialized = True  # Still proceed in placeholder mode
                return True
            self._disabled_reason = f"Flow extraction disabled: {e}"
            self.log(self._disabled_reason, level="ERROR")
            self._initialized = True
            return False

    def extract(
        self,
        video: GeneratedVideo,
        output_dir: Path,
        target_object: Optional[str] = None,
    ) -> FlowExtractionResult:
        """
        Extract 3D object flow from a generated video.

        Args:
            video: Generated video to extract flow from
            output_dir: Directory to save outputs
            target_object: Optional object to focus on (for segmentation prompt)

        Returns:
            FlowExtractionResult with extracted flows
        """
        if not self._initialized:
            self.initialize()
        if self._disabled_reason:
            self.log(self._disabled_reason, level="ERROR")
            return FlowExtractionResult(
                video_id=video.video_id,
                success=False,
                error=self._disabled_reason,
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.log(f"Extracting flow from video: {video.video_id}")

        try:
            # Step 1: Load video frames
            frames = self._load_frames(video)
            if not frames:
                raise ValueError("No frames found in video")

            self.log(f"Loaded {len(frames)} frames")

            # Step 2: Segment the manipulated object
            masks = self._segment_object(
                frames=frames,
                target_object=target_object or self._get_target_from_instruction(video.instruction),
                output_dir=output_dir / "masks" if self.config.save_intermediate else None,
            )

            self.log(f"Segmented {len(masks)} objects")

            # Step 3: Estimate depth for each frame
            depth_maps = self._estimate_depth(
                frames=frames,
                output_dir=output_dir / "depth" if self.config.save_intermediate else None,
            )

            self.log(f"Estimated depth for {len(depth_maps)} frames")

            # Step 4: Track points on the object
            tracked_points = self._track_points(
                frames=frames,
                masks=masks,
                output_dir=output_dir / "tracks" if self.config.save_intermediate else None,
            )

            self.log(f"Tracked {len(tracked_points)} points")

            # Step 5: Lift 2D tracks to 3D
            object_flows = self._lift_to_3d(
                tracked_points=tracked_points,
                depth_maps=depth_maps,
                masks=masks,
                frame_shape=(frames[0].shape[0], frames[0].shape[1]) if frames else (480, 720),
            )

            self.log(f"Created {len(object_flows)} 3D object flows")

            # Save visualization if requested
            if self.config.save_visualizations:
                self._save_visualization(
                    frames=frames,
                    object_flows=object_flows,
                    output_dir=output_dir / "visualization",
                )

            # Calculate quality metrics
            extraction_confidence = self._calculate_confidence(object_flows)

            return FlowExtractionResult(
                video_id=video.video_id,
                object_flows=object_flows,
                object_masks=masks,
                depth_maps=depth_maps,
                method=self.config.method,
                success=len(object_flows) > 0,
                extraction_confidence=extraction_confidence,
                masks_dir=output_dir / "masks" if self.config.save_intermediate else None,
                depth_dir=output_dir / "depth" if self.config.save_intermediate else None,
                tracks_dir=output_dir / "tracks" if self.config.save_intermediate else None,
            )

        except Exception as e:
            self.log(f"Flow extraction failed: {e}", "ERROR")
            traceback.print_exc()
            return FlowExtractionResult(
                video_id=video.video_id,
                success=False,
                error=str(e),
            )

    def _load_frames(self, video: GeneratedVideo) -> list[np.ndarray]:
        """Load frames from video file or directory."""
        frames = []

        # Try frames directory first
        if video.frames_dir and video.frames_dir.exists():
            frame_paths = sorted(video.frames_dir.glob("*.png")) + sorted(video.frames_dir.glob("*.jpg"))
            for path in frame_paths:
                if Image is not None:
                    frame = np.array(Image.open(path))
                    frames.append(frame)

        # Try video file
        elif video.video_path and video.video_path.exists() and imageio is not None:
            reader = imageio.get_reader(video.video_path)
            try:
                for frame in reader:
                    frames.append(frame)
            finally:
                reader.close()

        return frames

    def _get_target_from_instruction(self, instruction: Optional[TaskInstruction]) -> Optional[str]:
        """Extract target object from instruction."""
        if instruction is None:
            return None
        return instruction.target_object or instruction.text

    def _segment_object(
        self,
        frames: list[np.ndarray],
        target_object: Optional[str],
        output_dir: Optional[Path],
    ) -> list[ObjectMask]:
        """
        Segment the manipulated object in each frame.

        Uses SAM, Grounded-SAM, or other segmentation models.
        Currently uses placeholder segmentation.
        """
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.segmentation_api:
            try:
                return self._segment_object_via_api(frames, target_object, output_dir)
            except Exception as e:
                if not self.config.allow_placeholder:
                    raise
                self.log(f"Segmentation API failed, using placeholder: {e}", level="WARNING")

        # Placeholder: create simple center-based masks

        masks = []
        object_id = "manipulated_object"

        frame_masks = []
        for i, frame in enumerate(frames):
            # Placeholder: create elliptical mask in center region
            h, w = frame.shape[:2]
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            radius_y, radius_x = h // 4, w // 4

            # Create elliptical mask
            mask = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1
            mask = mask.astype(np.uint8) * 255

            frame_masks.append((i, mask))

            # Save if requested
            if output_dir and Image is not None:
                Image.fromarray(mask).save(output_dir / f"mask_{i:04d}.png")

        masks.append(ObjectMask(
            object_id=object_id,
            category=target_object or "object",
            frame_masks=frame_masks,
            confidence_scores=[0.8] * len(frame_masks),  # Placeholder confidence
            is_manipulated=True,
        ))

        return masks

    def _estimate_depth(
        self,
        frames: list[np.ndarray],
        output_dir: Optional[Path],
    ) -> list[Optional[np.ndarray]]:
        """
        Estimate depth for each frame.

        Uses DepthAnything, ZoeDepth, or other depth estimation models.
        Currently uses placeholder depth.
        """
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.depth_api:
            try:
                return self._estimate_depth_via_api(frames, output_dir)
            except Exception as e:
                if not self.config.allow_placeholder:
                    raise
                self.log(f"Depth API failed, using placeholder: {e}", level="WARNING")

        # Placeholder: create simple gradient depth

        depth_maps = []
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]

            # Placeholder: linear depth gradient (closer at bottom, farther at top)
            # Real depth would be estimated by depth model
            y_coords = np.linspace(0.5, 3.0, h)  # Depth from 0.5m to 3m
            depth = np.tile(y_coords[:, np.newaxis], (1, w))

            # Add some noise for realism
            noise = np.random.normal(0, 0.05, depth.shape)
            depth = depth + noise
            depth = np.clip(depth, 0.1, 10.0).astype(np.float32)

            depth_maps.append(depth)

            # Save if requested
            if output_dir:
                # Save as 16-bit PNG (depth in millimeters)
                depth_mm = (depth * 1000).astype(np.uint16)
                if Image is not None:
                    Image.fromarray(depth_mm).save(output_dir / f"depth_{i:04d}.png")

        return depth_maps

    def _track_points(
        self,
        frames: list[np.ndarray],
        masks: list[ObjectMask],
        output_dir: Optional[Path],
    ) -> list[TrackedPoint]:
        """
        Track points on the object across frames.

        Uses CoTracker, TAPIR, or other point tracking models.
        Currently uses placeholder tracking.
        """
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.tracking_api:
            try:
                return self._track_points_via_api(frames, masks, output_dir)
            except Exception as e:
                if not self.config.allow_placeholder:
                    raise
                self.log(f"Tracking API failed, using placeholder: {e}", level="WARNING")

        # Placeholder: sample points and add simulated motion

        tracked_points = []
        num_frames = len(frames)

        if not masks or not masks[0].frame_masks:
            return tracked_points

        # Get initial mask to sample points from
        initial_mask = masks[0].frame_masks[0][1]
        h, w = initial_mask.shape

        # Sample points within the mask
        mask_coords = np.where(initial_mask > 0)
        if len(mask_coords[0]) == 0:
            return tracked_points

        # Subsample points
        num_points = min(self.config.num_tracking_points, len(mask_coords[0]))
        indices = np.random.choice(len(mask_coords[0]), num_points, replace=False)
        initial_points = np.stack([mask_coords[1][indices], mask_coords[0][indices]], axis=1)  # (x, y)

        # Track each point (placeholder: add simulated motion)
        for point_idx, (x, y) in enumerate(initial_points):
            positions_2d = np.zeros((num_frames, 2))
            visibility = np.ones(num_frames, dtype=bool)
            confidence = np.ones(num_frames) * 0.9

            for frame_idx in range(num_frames):
                # Placeholder: add small random motion to simulate tracking
                progress = frame_idx / max(num_frames - 1, 1)

                # Simulate object moving slightly (would be actual tracked positions)
                dx = np.sin(progress * np.pi) * 20  # Move right then back
                dy = progress * -10  # Move slightly up

                new_x = np.clip(x + dx, 0, w - 1)
                new_y = np.clip(y + dy, 0, h - 1)

                positions_2d[frame_idx] = [new_x, new_y]

                # Random occlusion for realism
                if np.random.random() < 0.05:
                    visibility[frame_idx] = False
                    confidence[frame_idx] = 0.3

            tracked_point = TrackedPoint(
                point_id=point_idx,
                positions_2d=positions_2d,
                visibility=visibility,
                confidence=confidence,
                object_id=masks[0].object_id if masks else None,
            )
            tracked_points.append(tracked_point)

        # Save tracking visualization
        if output_dir and len(tracked_points) > 0:
            self._save_tracking_visualization(frames, tracked_points, output_dir)

        return tracked_points

    def _segment_object_via_api(
        self,
        frames: list[np.ndarray],
        target_object: Optional[str],
        output_dir: Optional[Path],
    ) -> list[ObjectMask]:
        """Segment objects via remote API."""
        if Image is None:
            raise RuntimeError("PIL not available for segmentation API payload encoding.")
        import base64
        import io
        import requests

        self.log("Segmenting objects via API...")
        payload_frames = []
        for frame in frames:
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, format="PNG")
            payload_frames.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

        response = requests.post(
            self.config.segmentation_api,
            json={
                "frames": payload_frames,
                "prompt": target_object,
                "model": self.config.segmentation_model,
            },
            timeout=300,
        )
        response.raise_for_status()
        result = response.json()

        masks_payload = result.get("masks_base64") or result.get("masks")
        if not masks_payload:
            raise ValueError("Segmentation API did not return masks")

        masks = []
        frame_masks = []
        for i, mask_item in enumerate(masks_payload):
            if isinstance(mask_item, str):
                mask_bytes = base64.b64decode(mask_item)
                mask = np.array(Image.open(io.BytesIO(mask_bytes)).convert("L"))
            else:
                mask = np.array(mask_item, dtype=np.uint8)
            frame_masks.append((i, mask))
            if output_dir and Image is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(output_dir / f"mask_{i:04d}.png")

        masks.append(
            ObjectMask(
                object_id="manipulated_object",
                category=target_object or "object",
                frame_masks=frame_masks,
                confidence_scores=[result.get("confidence", 0.9)] * len(frame_masks),
                is_manipulated=True,
            )
        )
        return masks

    def _estimate_depth_via_api(
        self,
        frames: list[np.ndarray],
        output_dir: Optional[Path],
    ) -> list[Optional[np.ndarray]]:
        """Estimate depth via remote API."""
        if Image is None:
            raise RuntimeError("PIL not available for depth API payload encoding.")
        import base64
        import io
        import requests

        self.log("Estimating depth via API...")
        payload_frames = []
        for frame in frames:
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, format="PNG")
            payload_frames.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

        response = requests.post(
            self.config.depth_api,
            json={
                "frames": payload_frames,
                "model": self.config.depth_model,
            },
            timeout=300,
        )
        response.raise_for_status()
        result = response.json()

        depth_payload = result.get("depths_base64") or result.get("depths")
        if not depth_payload:
            raise ValueError("Depth API did not return depth maps")

        depth_maps = []
        for i, depth_item in enumerate(depth_payload):
            if isinstance(depth_item, str):
                depth_bytes = base64.b64decode(depth_item)
                depth = np.array(Image.open(io.BytesIO(depth_bytes)))
                if depth.dtype != np.float32:
                    depth = depth.astype(np.float32) / 1000.0
            else:
                depth = np.array(depth_item, dtype=np.float32)
            depth_maps.append(depth)

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                depth_mm = (depth * 1000).astype(np.uint16)
                if Image is not None:
                    Image.fromarray(depth_mm).save(output_dir / f"depth_{i:04d}.png")

        return depth_maps

    def _track_points_via_api(
        self,
        frames: list[np.ndarray],
        masks: list[ObjectMask],
        output_dir: Optional[Path],
    ) -> list[TrackedPoint]:
        """Track points via remote API."""
        if Image is None:
            raise RuntimeError("PIL not available for tracking API payload encoding.")
        import base64
        import io
        import requests

        self.log("Tracking points via API...")
        payload_frames = []
        for frame in frames:
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, format="PNG")
            payload_frames.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

        mask_payload = []
        if masks:
            for _, mask in masks[0].frame_masks:
                buffer = io.BytesIO()
                Image.fromarray(mask).save(buffer, format="PNG")
                mask_payload.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

        response = requests.post(
            self.config.tracking_api,
            json={
                "frames": payload_frames,
                "masks": mask_payload,
                "num_points": self.config.num_tracking_points,
                "model": self.config.tracking_model,
            },
            timeout=300,
        )
        response.raise_for_status()
        result = response.json()

        tracks_payload = result.get("tracks")
        if not tracks_payload:
            raise ValueError("Tracking API did not return tracks")

        tracked_points = []
        for idx, track in enumerate(tracks_payload):
            positions_2d = np.array(track.get("positions_2d"), dtype=np.float32)
            visibility = np.array(track.get("visibility", np.ones(len(positions_2d))), dtype=bool)
            confidence = np.array(track.get("confidence", np.ones(len(positions_2d))), dtype=np.float32)
            tracked_points.append(
                TrackedPoint(
                    point_id=track.get("point_id", idx),
                    positions_2d=positions_2d,
                    visibility=visibility,
                    confidence=confidence,
                    object_id=track.get("object_id", masks[0].object_id if masks else None),
                )
            )

        if output_dir and tracked_points:
            self._save_tracking_visualization(frames, tracked_points, output_dir)

        return tracked_points

    def _lift_to_3d(
        self,
        tracked_points: list[TrackedPoint],
        depth_maps: list[Optional[np.ndarray]],
        masks: list[ObjectMask],
        frame_shape: tuple[int, int],
    ) -> list[ObjectFlow3D]:
        """
        Lift 2D tracked points to 3D using depth and camera geometry.

        Uses pinhole camera model to project 2D points to 3D.
        """
        if not tracked_points or not depth_maps:
            return []

        h, w = frame_shape
        fx = self.config.camera_focal_length
        fy = fx  # Assume square pixels
        cx = self.config.camera_cx if self.config.camera_cx is not None else w / 2
        cy = self.config.camera_cy if self.config.camera_cy is not None else h / 2

        # Lift each tracked point to 3D
        for tracked_point in tracked_points:
            num_frames = tracked_point.positions_2d.shape[0]
            positions_3d = np.zeros((num_frames, 3))

            for frame_idx in range(num_frames):
                x, y = tracked_point.positions_2d[frame_idx]
                x_int, y_int = int(round(x)), int(round(y))

                # Get depth at this pixel
                if frame_idx < len(depth_maps) and depth_maps[frame_idx] is not None:
                    depth = depth_maps[frame_idx]
                    if 0 <= y_int < depth.shape[0] and 0 <= x_int < depth.shape[1]:
                        z = depth[y_int, x_int]
                    else:
                        z = 1.0  # Default depth
                else:
                    z = 1.0

                # Project to 3D using pinhole model
                # X = (x - cx) * Z / fx
                # Y = (y - cy) * Z / fy
                # Z = depth
                X = (x - cx) * z / fx
                Y = (y - cy) * z / fy
                Z = z

                positions_3d[frame_idx] = [X, Y, Z]

            tracked_point.positions_3d = positions_3d

        # Group tracked points by object
        object_flows = []
        for mask in masks:
            object_points = [
                tp for tp in tracked_points
                if tp.object_id == mask.object_id
            ]

            if object_points:
                # Calculate quality metrics
                mean_confidence = np.mean([
                    np.mean(tp.confidence) for tp in object_points
                    if tp.confidence is not None
                ])
                coverage_ratio = np.mean([
                    np.mean(tp.visibility) for tp in object_points
                    if tp.visibility is not None
                ])

                flow = ObjectFlow3D(
                    flow_id=f"flow_{mask.object_id}",
                    object_id=mask.object_id,
                    object_category=mask.category,
                    tracked_points=object_points,
                    num_frames=len(depth_maps),
                    fps=24.0,  # Will be set from video
                    extraction_method=self.config.method,
                    mean_confidence=mean_confidence,
                    coverage_ratio=coverage_ratio,
                    metadata={
                        "num_points": len(object_points),
                        "camera_params": {
                            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                        },
                    },
                )
                object_flows.append(flow)

        return object_flows

    def _save_tracking_visualization(
        self,
        frames: list[np.ndarray],
        tracked_points: list[TrackedPoint],
        output_dir: Path,
    ) -> None:
        """Save visualization of tracked points."""
        if Image is None:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create visualization for each frame
        for frame_idx, frame in enumerate(frames):
            img = Image.fromarray(frame).convert("RGB")
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)

            # Draw tracked points
            for tp in tracked_points:
                if frame_idx < tp.positions_2d.shape[0]:
                    x, y = tp.positions_2d[frame_idx]
                    visible = tp.visibility[frame_idx] if tp.visibility is not None else True

                    color = (0, 255, 0) if visible else (255, 0, 0)
                    radius = 3
                    draw.ellipse(
                        [x - radius, y - radius, x + radius, y + radius],
                        fill=color,
                        outline=color,
                    )

                    # Draw trail
                    if frame_idx > 0:
                        prev_x, prev_y = tp.positions_2d[frame_idx - 1]
                        draw.line([(prev_x, prev_y), (x, y)], fill=(100, 200, 100), width=1)

            img.save(output_dir / f"track_vis_{frame_idx:04d}.png")

    def _save_visualization(
        self,
        frames: list[np.ndarray],
        object_flows: list[ObjectFlow3D],
        output_dir: Path,
    ) -> None:
        """Save overall flow visualization."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # For each flow, visualize the 3D trajectory
        for flow in object_flows:
            trajectory = flow.get_center_trajectory()
            if len(trajectory) > 0:
                # Save trajectory as JSON
                import json
                trajectory_path = output_dir / f"{flow.flow_id}_trajectory.json"
                trajectory_path.write_text(json.dumps({
                    "flow_id": flow.flow_id,
                    "object_id": flow.object_id,
                    "num_points": len(flow.tracked_points),
                    "center_trajectory": trajectory.tolist(),
                    "mean_confidence": flow.mean_confidence,
                }, indent=2))

    def _calculate_confidence(self, object_flows: list[ObjectFlow3D]) -> float:
        """Calculate overall extraction confidence."""
        if not object_flows:
            return 0.0

        confidences = [flow.mean_confidence for flow in object_flows]
        return np.mean(confidences) if confidences else 0.0


class MockFlowExtractor(FlowExtractor):
    """Mock flow extractor for testing and CI."""

    def __init__(self, config: Optional[FlowExtractorConfig] = None):
        config = config or FlowExtractorConfig()
        super().__init__(config)

    def initialize(self) -> bool:
        self.log("Mock flow extractor initialized")
        self._initialized = True
        return True


def extract_object_flow(
    video: GeneratedVideo,
    output_dir: Path,
    config: Optional[FlowExtractorConfig] = None,
    target_object: Optional[str] = None,
) -> FlowExtractionResult:
    """
    Convenience function to extract 3D object flow from a video.

    Args:
        video: Generated video to process
        output_dir: Directory to save outputs
        config: Optional extractor configuration
        target_object: Optional object to focus on

    Returns:
        FlowExtractionResult with extracted flows
    """
    config = config or FlowExtractorConfig()
    extractor = FlowExtractor(config)

    return extractor.extract(
        video=video,
        output_dir=output_dir,
        target_object=target_object,
    )
