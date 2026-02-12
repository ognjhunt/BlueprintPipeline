import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from tools.config.env import parse_bool_env
from tools.lerobot_format import LeRobotExportFormat

from episode_generation.helpers import _is_production_run

logger = logging.getLogger(__name__)


@dataclass
class EpisodeGenerationEnvConfig:
    bucket: str
    scene_id: str
    assets_prefix: str
    episodes_prefix: str
    enable_firebase_upload: bool
    firebase_upload_prefix: str
    robot_type: str
    episodes_per_variation: int
    max_variations: Optional[int]
    fps: float
    use_llm: bool
    use_cpgen: bool
    min_quality_score: float
    min_success_rate: float
    data_pack_tier: str
    num_cameras: int
    image_resolution: Tuple[int, int]
    capture_sensor_data: bool
    use_mock_capture: bool
    allow_mock_capture: bool
    bundle_tier: str
    lerobot_export_format: Optional[Union[str, LeRobotExportFormat]]
    lerobot_parquet_compression: Optional[str]


def parse_env_config(*, allow_mock_capture: bool) -> EpisodeGenerationEnvConfig:
    bucket = os.environ["BUCKET"]
    scene_id = os.environ["SCENE_ID"]

    assets_prefix = os.getenv("ASSETS_PREFIX", f"scenes/{scene_id}/assets")
    episodes_prefix = os.getenv("EPISODES_PREFIX", f"scenes/{scene_id}/episodes")
    enable_firebase_upload = parse_bool_env(
        os.getenv("ENABLE_FIREBASE_UPLOAD"),
        default=_is_production_run(),
    )
    firebase_upload_prefix = os.getenv("FIREBASE_UPLOAD_PREFIX", "datasets")

    robot_type = os.getenv("ROBOT_TYPE", "franka")

    try:
        episodes_per_variation = int(os.getenv("EPISODES_PER_VARIATION", "10"))
        if episodes_per_variation <= 0:
            raise ValueError("EPISODES_PER_VARIATION must be positive")
    except ValueError as exc:
        logger.error("[EPISODE-GEN-JOB] Invalid EPISODES_PER_VARIATION: %s", exc)
        raise SystemExit(1) from exc

    max_variations = os.getenv("MAX_VARIATIONS")
    if max_variations:
        try:
            max_variations_int = int(max_variations)
            if max_variations_int <= 0:
                raise ValueError("MAX_VARIATIONS must be positive")
        except ValueError as exc:
            logger.error("[EPISODE-GEN-JOB] Invalid MAX_VARIATIONS: %s", exc)
            raise SystemExit(1) from exc
        max_variations = max_variations_int
    else:
        max_variations = None

    try:
        fps = float(os.getenv("FPS", "30"))
        if fps <= 0 or fps > 240:
            raise ValueError("FPS must be between 0 and 240")
    except ValueError as exc:
        logger.error("[EPISODE-GEN-JOB] Invalid FPS: %s", exc)
        raise SystemExit(1) from exc

    use_llm = parse_bool_env(os.getenv("USE_LLM"), default=True)
    use_cpgen = parse_bool_env(os.getenv("USE_CPGEN"), default=True)

    try:
        min_quality_score = float(os.getenv("MIN_QUALITY_SCORE", "0.85"))
        if not (0.0 <= min_quality_score <= 1.0):
            raise ValueError("MIN_QUALITY_SCORE must be between 0.0 and 1.0")
    except ValueError as exc:
        logger.error("[EPISODE-GEN-JOB] Invalid MIN_QUALITY_SCORE: %s", exc)
        raise SystemExit(1) from exc

    try:
        min_success_rate = float(os.getenv("MIN_SUCCESS_RATE", "0.5"))
        if not (0.0 <= min_success_rate <= 1.0):
            raise ValueError("MIN_SUCCESS_RATE must be between 0.0 and 1.0")
    except ValueError as exc:
        logger.error("[EPISODE-GEN-JOB] Invalid MIN_SUCCESS_RATE: %s", exc)
        raise SystemExit(1) from exc

    data_pack_tier = os.getenv("DATA_PACK_TIER", "core")
    lerobot_export_format = os.getenv("LEROBOT_EXPORT_FORMAT")
    lerobot_parquet_compression = os.getenv("LEROBOT_PARQUET_COMPRESSION")

    try:
        num_cameras = int(os.getenv("NUM_CAMERAS", "1"))
        if num_cameras < 1 or num_cameras > 8:
            raise ValueError("NUM_CAMERAS must be between 1 and 8")
    except ValueError as exc:
        logger.error("[EPISODE-GEN-JOB] Invalid NUM_CAMERAS: %s", exc)
        raise SystemExit(1) from exc

    resolution_str = os.getenv("IMAGE_RESOLUTION", "640,480")
    try:
        resolution_parts = resolution_str.split(",")
        if len(resolution_parts) != 2:
            raise ValueError("IMAGE_RESOLUTION must be in format 'width,height'")
        image_resolution = tuple(map(int, resolution_parts))
        if image_resolution[0] <= 0 or image_resolution[1] <= 0:
            raise ValueError("IMAGE_RESOLUTION dimensions must be positive")
    except (ValueError, TypeError) as exc:
        logger.error("[EPISODE-GEN-JOB] Invalid IMAGE_RESOLUTION: %s", exc)
        raise SystemExit(1) from exc

    capture_sensor_data = parse_bool_env(os.getenv("CAPTURE_SENSOR_DATA"), default=True)
    use_mock_capture = parse_bool_env(os.getenv("USE_MOCK_CAPTURE"), default=False)

    bundle_tier = os.getenv("BUNDLE_TIER", "standard")

    return EpisodeGenerationEnvConfig(
        bucket=bucket,
        scene_id=scene_id,
        assets_prefix=assets_prefix,
        episodes_prefix=episodes_prefix,
        enable_firebase_upload=enable_firebase_upload,
        firebase_upload_prefix=firebase_upload_prefix,
        robot_type=robot_type,
        episodes_per_variation=episodes_per_variation,
        max_variations=max_variations,
        fps=fps,
        use_llm=use_llm,
        use_cpgen=use_cpgen,
        min_quality_score=min_quality_score,
        min_success_rate=min_success_rate,
        data_pack_tier=data_pack_tier,
        num_cameras=num_cameras,
        image_resolution=image_resolution,
        capture_sensor_data=capture_sensor_data,
        use_mock_capture=use_mock_capture,
        allow_mock_capture=allow_mock_capture,
        bundle_tier=bundle_tier,
        lerobot_export_format=lerobot_export_format,
        lerobot_parquet_compression=lerobot_parquet_compression,
    )
