"""GCS Sync — Download inputs from and upload outputs to Google Cloud Storage.

Bridges the local pipeline runner with GCS so that:
  1. Input images can be downloaded from ``gs://bucket/scenes/{scene_id}/images/{any_name}.(png|jpg|jpeg)``
  2. Each step's outputs are uploaded back after completion.
  3. Completion / failure markers are written for EventArc triggers.

Uses ``tools/gcs_upload.py:upload_blob_from_filename`` for uploads to inherit
its retry and MD5-verification logic.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional GCS import (graceful fallback for tests / local dev)
# ---------------------------------------------------------------------------

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    gcs_storage = None  # type: ignore[assignment]

try:
    from tools.gcs_upload import upload_blob_from_filename, UploadResult
except ImportError:
    # Allow importing this module in environments where tools.gcs_upload is not
    # on sys.path (e.g. unit-testing gcs_sync in isolation).
    upload_blob_from_filename = None  # type: ignore[assignment]
    UploadResult = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Directories that correspond to pipeline step outputs and should be synced
# back to GCS after the corresponding step completes.
# A step can produce outputs in multiple directories (list of dir names).
STEP_OUTPUT_DIRS: Dict[str, List[str]] = {
    "text-scene-gen": ["textgen"],
    "text-scene-adapter": ["assets", "layout", "seg", "textgen"],
    "scale": ["assets"],
    "interactive": ["assets"],
    "simready": ["assets", "layout"],
    "usd": ["usd", "assets"],
    "inventory-enrichment": ["seg"],
    "replicator": ["replicator"],
    "variation-gen": ["variation_assets"],
    "isaac-lab": ["isaac_lab"],
    "genie-sim-export": ["geniesim"],
    "genie-sim-submit": ["geniesim"],
    "genie-sim-import": ["geniesim", "episodes"],
    "dataset-delivery": ["geniesim"],
    "dwm": ["dwm"],
    "dwm-inference": ["dwm"],
    "dream2flow": ["dream2flow"],
    "dream2flow-inference": ["dream2flow"],
    "validate": [],
}

# Additional directories that are always uploaded (if they exist)
ALWAYS_UPLOAD_DIRS = {"seg", "input"}

# Supported image formats for auto-triggered reconstruction.
SUPPORTED_INPUT_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Legacy input image names to try under input/ when no image object is specified.
FALLBACK_INPUT_IMAGE_NAMES = ["room.png", "room.jpg", "room.jpeg"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyncResult:
    """Result of a single sync operation."""
    success: bool
    files_synced: int
    errors: List[str] = field(default_factory=list)
    gcs_uris: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MarkerResult:
    """Result of writing a completion/failure marker."""
    success: bool
    marker_uri: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# GCSSync class
# ---------------------------------------------------------------------------

class GCSSync:
    """Sync scene data between GCS and the local filesystem.

    Usage::

        sync = GCSSync("my-bucket", "scene_123", Path("/tmp/scenes/scene_123"))
        sync.download_inputs()
        # ... run pipeline ...
        sync.upload_step_outputs("text-scene-gen", Path("/tmp/scenes/scene_123/textgen"))
        sync.write_completion_marker(".reconstruction_complete")
    """

    def __init__(
        self,
        bucket_name: str,
        scene_id: str,
        local_scene_dir: Path,
        *,
        concurrency: int = 4,
        verify_uploads: bool = False,
        input_object: Optional[str] = None,
        input_generation: Optional[str] = None,
    ):
        if gcs_storage is None:
            raise ImportError(
                "google-cloud-storage is required for GCS sync. "
                "Install with: pip install google-cloud-storage"
            )

        self.bucket_name = bucket_name
        self.scene_id = scene_id
        self.local_scene_dir = Path(local_scene_dir)
        self.concurrency = max(1, int(concurrency))
        self.verify_uploads = verify_uploads
        self.input_object = input_object
        self.input_generation = input_generation

        self._client = gcs_storage.Client()
        self._bucket = self._client.bucket(bucket_name)
        self._gcs_prefix = f"scenes/{scene_id}"

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_inputs(self, preferred_object: Optional[str] = None) -> Path:
        """Download the input image from GCS to ``scene_dir/input/``.

        Prefers a specific object when provided. Otherwise downloads the latest
        image from ``scenes/{scene_id}/images/`` with extension in
        ``{png,jpg,jpeg}``, then falls back to legacy ``input/room.*`` paths.

        Returns:
            Path to the downloaded local image file.

        Raises:
            FileNotFoundError: If no input image is found in GCS.
        """
        input_dir = self.local_scene_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        explicit_object = preferred_object or self.input_object
        if explicit_object:
            try:
                return self._download_exact_input_object(explicit_object, input_dir)
            except FileNotFoundError:
                logger.warning(
                    "[GCS] Preferred input object not found; falling back to latest image: %s",
                    explicit_object,
                )

        latest_image = self._download_latest_image(input_dir)
        if latest_image is not None:
            return latest_image

        # Legacy fallback: scenes/{scene_id}/input/room.{png,jpg,jpeg}
        for image_name in FALLBACK_INPUT_IMAGE_NAMES:
            gcs_path = f"{self._gcs_prefix}/input/{image_name}"
            blob = self._bucket.blob(gcs_path)
            if blob.exists():
                local_path = input_dir / image_name
                logger.info(f"[GCS] Downloading gs://{self.bucket_name}/{gcs_path}")
                blob.download_to_filename(str(local_path))
                logger.info(f"[GCS] Downloaded to {local_path} ({local_path.stat().st_size} bytes)")
                self.input_object = gcs_path
                if getattr(blob, "generation", None) is not None:
                    self.input_generation = str(blob.generation)
                return local_path

        raise FileNotFoundError(
            f"No input image found in GCS at gs://{self.bucket_name}/{self._gcs_prefix}/images/ "
            "with extension .png/.jpg/.jpeg"
        )

    def download_file(self, gcs_relative_path: str, local_path: Path) -> Path:
        """Download a single file from GCS.

        Args:
            gcs_relative_path: Path relative to the scene prefix (e.g., "assets/scene_manifest.json").
            local_path: Local destination path.

        Returns:
            The local path of the downloaded file.
        """
        gcs_path = f"{self._gcs_prefix}/{gcs_relative_path}"
        blob = self._bucket.blob(gcs_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        return local_path

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_step_outputs(self, step_name: str, local_dir: Optional[Path] = None) -> SyncResult:
        """Upload outputs from a single pipeline step to GCS.

        A step can write to multiple directories (e.g., ``text-scene-adapter`` writes to
        ``assets/``, ``layout/``, and ``seg/``).  All mapped directories are
        uploaded.

        Args:
            step_name: Pipeline step name (e.g., ``"text-scene-gen"``).
            local_dir: Override — upload this single directory instead of the
                mapped directories.

        Returns:
            Aggregated SyncResult across all uploaded directories.
        """
        if local_dir is not None:
            local_dir = Path(local_dir)
            if not local_dir.is_dir():
                logger.warning(f"[GCS] Output directory does not exist: {local_dir}")
                return SyncResult(success=True, files_synced=0)
            try:
                rel_dir = local_dir.relative_to(self.local_scene_dir)
            except ValueError:
                rel_dir = Path(local_dir.name)
            return self._upload_directory(local_dir, str(rel_dir))

        dir_names = STEP_OUTPUT_DIRS.get(step_name)
        if dir_names is None:
            logger.warning(f"[GCS] No output directory mapping for step '{step_name}'")
            return SyncResult(success=True, files_synced=0)

        total_files = 0
        all_errors: List[str] = []
        all_uris: List[str] = []

        for dir_name in dir_names:
            step_dir = self.local_scene_dir / dir_name
            if not step_dir.is_dir():
                continue
            result = self._upload_directory(step_dir, dir_name)
            total_files += result.files_synced
            all_errors.extend(result.errors)
            all_uris.extend(result.gcs_uris)

        return SyncResult(
            success=len(all_errors) == 0,
            files_synced=total_files,
            errors=all_errors,
            gcs_uris=all_uris,
        )

    def upload_all_outputs(self) -> Dict[str, SyncResult]:
        """Upload all known output directories to GCS.

        Returns:
            Dict mapping directory name to SyncResult.
        """
        results: Dict[str, SyncResult] = {}

        # Upload step output dirs
        mapped_dirs: Set[str] = {
            dir_name
            for dir_names in STEP_OUTPUT_DIRS.values()
            for dir_name in dir_names
        }
        all_dirs: Set[str] = mapped_dirs | ALWAYS_UPLOAD_DIRS
        for dir_name in sorted(all_dirs):
            local_dir = self.local_scene_dir / dir_name
            if local_dir.is_dir() and any(local_dir.iterdir()):
                result = self._upload_directory(local_dir, dir_name)
                results[dir_name] = result

        return results

    def _upload_directory(self, local_dir: Path, gcs_sub_prefix: str) -> SyncResult:
        """Upload all files in a local directory to GCS."""
        files = [f for f in local_dir.rglob("*") if f.is_file()]
        if not files:
            return SyncResult(success=True, files_synced=0)

        uploaded = 0
        errors: List[str] = []
        gcs_uris: List[str] = []

        def _record_upload_result(local_file: Path) -> None:
            nonlocal uploaded
            success, gcs_uri, error_msg = self._upload_file(local_file, local_dir, gcs_sub_prefix)
            if success:
                uploaded += 1
                gcs_uris.append(gcs_uri)
            elif error_msg:
                errors.append(error_msg)

        if self.concurrency <= 1 or len(files) == 1:
            for local_file in files:
                _record_upload_result(local_file)
        else:
            workers = min(self.concurrency, len(files))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_file = {
                    executor.submit(self._upload_file, local_file, local_dir, gcs_sub_prefix): local_file
                    for local_file in files
                }
                for future in as_completed(future_to_file):
                    success, gcs_uri, error_msg = future.result()
                    if success:
                        uploaded += 1
                        gcs_uris.append(gcs_uri)
                    elif error_msg:
                        errors.append(error_msg)

        logger.info(
            f"[GCS] Uploaded {uploaded}/{len(files)} files from {local_dir.name}/ "
            f"to gs://{self.bucket_name}/{self._gcs_prefix}/{gcs_sub_prefix}/"
        )

        return SyncResult(
            success=len(errors) == 0,
            files_synced=uploaded,
            errors=errors,
            gcs_uris=gcs_uris,
        )

    def _upload_file(
        self,
        local_file: Path,
        local_dir: Path,
        gcs_sub_prefix: str,
    ) -> tuple[bool, str, Optional[str]]:
        rel_path = local_file.relative_to(local_dir)
        gcs_path = f"{self._gcs_prefix}/{gcs_sub_prefix}/{rel_path}"
        gcs_uri = f"gs://{self.bucket_name}/{gcs_path}"
        blob = self._bucket.blob(gcs_path)
        content_type = mimetypes.guess_type(str(local_file))[0]

        try:
            if upload_blob_from_filename is not None:
                result = upload_blob_from_filename(
                    blob,
                    local_file,
                    gcs_uri,
                    logger=logger,
                    content_type=content_type,
                    verify_upload=self.verify_uploads,
                )
                if result.success:
                    return True, gcs_uri, None
                return False, gcs_uri, f"{local_file.name}: {result.error}"

            # Fallback: direct upload without retry wrapper.
            blob.upload_from_filename(str(local_file), content_type=content_type)
            return True, gcs_uri, None
        except Exception as exc:
            error_msg = f"{local_file.name}: {exc}"
            logger.error(f"[GCS] Upload failed: {error_msg}")
            return False, gcs_uri, error_msg

    # ------------------------------------------------------------------
    # Completion markers
    # ------------------------------------------------------------------

    def write_completion_marker(
        self,
        marker_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MarkerResult:
        """Write a completion marker file to GCS.

        The marker is a JSON file stored at
        ``scenes/{scene_id}/{marker_name}``.

        Args:
            marker_name: Marker filename (e.g., ``.reconstruction_complete``).
            metadata: Optional metadata to include in the marker JSON.

        Returns:
            MarkerResult indicating success or failure.
        """
        gcs_path = f"{self._gcs_prefix}/{marker_name}"
        gcs_uri = f"gs://{self.bucket_name}/{gcs_path}"

        payload: Dict[str, Any] = {
            "scene_id": self.scene_id,
            "status": "completed",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if self.input_object:
            payload["input_object"] = self.input_object
        if self.input_generation:
            payload["input_generation"] = self.input_generation
        if metadata:
            payload["metadata"] = metadata

        try:
            blob = self._bucket.blob(gcs_path)
            blob.upload_from_string(
                json.dumps(payload, indent=2),
                content_type="application/json",
            )
            logger.info(f"[GCS] Wrote completion marker: {gcs_uri}")
            return MarkerResult(success=True, marker_uri=gcs_uri)
        except Exception as exc:
            logger.error(f"[GCS] Failed to write marker {gcs_uri}: {exc}")
            return MarkerResult(success=False, marker_uri=gcs_uri, error=str(exc))

    def write_failure_marker(
        self,
        marker_name: str,
        error_message: str,
        error_code: str = "pipeline_failed",
        context: Optional[Dict[str, Any]] = None,
    ) -> MarkerResult:
        """Write a failure marker file to GCS.

        Args:
            marker_name: Marker filename (e.g., ``.reconstruction_failed``).
            error_message: Human-readable error description.
            error_code: Machine-readable error code.
            context: Optional additional context.

        Returns:
            MarkerResult indicating success or failure.
        """
        gcs_path = f"{self._gcs_prefix}/{marker_name}"
        gcs_uri = f"gs://{self.bucket_name}/{gcs_path}"

        payload: Dict[str, Any] = {
            "scene_id": self.scene_id,
            "status": "failed",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": {
                "code": error_code,
                "message": error_message,
                "type": "pipeline_failure",
            },
        }
        if self.input_object:
            payload["input_object"] = self.input_object
        if self.input_generation:
            payload["input_generation"] = self.input_generation
        if context:
            payload["context"] = context

        try:
            blob = self._bucket.blob(gcs_path)
            blob.upload_from_string(
                json.dumps(payload, indent=2),
                content_type="application/json",
            )
            logger.info(f"[GCS] Wrote failure marker: {gcs_uri}")
            return MarkerResult(success=True, marker_uri=gcs_uri)
        except Exception as exc:
            logger.error(f"[GCS] Failed to write failure marker {gcs_uri}: {exc}")
            return MarkerResult(success=False, marker_uri=gcs_uri, error=str(exc))

    def check_marker_exists(self, marker_name: str) -> bool:
        """Check if a completion marker already exists (idempotence guard).

        Args:
            marker_name: Marker filename (e.g., ``.reconstruction_complete``).

        Returns:
            True if the marker exists in GCS.
        """
        gcs_path = f"{self._gcs_prefix}/{marker_name}"
        blob = self._bucket.blob(gcs_path)
        return blob.exists()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_gcs_uri(self, relative_path: str) -> str:
        """Build a ``gs://`` URI for a scene-relative path."""
        return f"gs://{self.bucket_name}/{self._gcs_prefix}/{relative_path}"

    def _download_exact_input_object(self, input_object: str, input_dir: Path) -> Path:
        gcs_path = input_object.strip()
        prefix = f"gs://{self.bucket_name}/"
        if gcs_path.startswith(prefix):
            gcs_path = gcs_path[len(prefix):]
        gcs_path = gcs_path.lstrip("/")

        expected_prefix = f"{self._gcs_prefix}/images/"
        if not gcs_path.startswith(expected_prefix):
            raise ValueError(
                f"Input object must be under {expected_prefix} (got {gcs_path})"
            )
        if Path(gcs_path).suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
            raise ValueError(
                f"Unsupported input image extension for {gcs_path}. "
                f"Expected one of: {sorted(SUPPORTED_INPUT_EXTENSIONS)}"
            )

        blob = self._bucket.blob(gcs_path)
        if not blob.exists():
            raise FileNotFoundError(
                f"Input object not found: gs://{self.bucket_name}/{gcs_path}"
            )

        local_path = input_dir / Path(gcs_path).name
        logger.info(f"[GCS] Downloading preferred input: gs://{self.bucket_name}/{gcs_path}")
        blob.download_to_filename(str(local_path))
        logger.info(f"[GCS] Downloaded to {local_path} ({local_path.stat().st_size} bytes)")
        self.input_object = gcs_path
        if getattr(blob, "generation", None) is not None:
            self.input_generation = str(blob.generation)
        return local_path

    def _download_latest_image(self, input_dir: Path) -> Optional[Path]:
        images_prefix = f"{self._gcs_prefix}/images/"
        blobs = list(self._bucket.list_blobs(prefix=images_prefix))
        image_blobs = [
            blob
            for blob in blobs
            if getattr(blob, "name", "").startswith(images_prefix)
            and Path(blob.name).suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
        ]
        if not image_blobs:
            return None

        def _generation_value(blob: Any) -> int:
            try:
                return int(getattr(blob, "generation", "0"))
            except (TypeError, ValueError):
                return 0

        # Prefer the highest generation (newest object version).
        image_blobs.sort(key=lambda blob: (_generation_value(blob), blob.name), reverse=True)
        selected_blob = image_blobs[0]
        selected_name = Path(selected_blob.name).name
        local_path = input_dir / selected_name

        logger.info(f"[GCS] Downloading latest image: gs://{self.bucket_name}/{selected_blob.name}")
        selected_blob.download_to_filename(str(local_path))
        logger.info(f"[GCS] Downloaded to {local_path} ({local_path.stat().st_size} bytes)")
        self.input_object = selected_blob.name
        if getattr(selected_blob, "generation", None) is not None:
            self.input_generation = str(selected_blob.generation)
        return local_path
