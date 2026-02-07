import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from tools.config.env import parse_bool_env
from tools.metrics.pipeline_metrics import get_metrics
from tools.validation.entrypoint_checks import validate_required_env_vars

from episode_generation.config import EpisodeGenerationEnvConfig, parse_env_config
from episode_generation.constants import (
    JOB_NAME,
    REQUIRED_EXTENSIONS,
    REQUIRED_ISAAC_SIM_CONTAINER,
    REQUIRED_ISAAC_SIM_VERSION,
)
from episode_generation.helpers import _is_production_run
from episode_generation.runner import run_episode_generation_job

logger = logging.getLogger(__name__)


def _run_main() -> None:
    """Main entry point."""
    from generate_episodes import check_sensor_capture_environment
    from generate_episodes import HAVE_QUALITY_SYSTEM

    logger.info("[EPISODE-GEN-JOB] ================================")
    logger.info("[EPISODE-GEN-JOB] Episode Generation Job (SOTA)")
    logger.info("[EPISODE-GEN-JOB] ================================")

    def _is_production_preflight() -> bool:
        from tools.config.production_mode import resolve_production_mode

        env_flags = resolve_production_mode()
        if HAVE_QUALITY_SYSTEM:
            try:
                from generate_episodes import get_data_quality_level

                return get_data_quality_level().value == "production" or env_flags
            except Exception:
                logger.warning(
                    "[EPISODE-GEN-JOB] Failed to read data quality level; falling back to env flags.",
                    exc_info=True,
                )
                return env_flags
        return env_flags

    def _collect_preflight_capabilities() -> Dict[str, Any]:
        capabilities: Dict[str, Any] = {
            "isaac_sim_available": False,
            "replicator_available": False,
            "physx_available": False,
            "gpu_available": os.path.exists("/dev/nvidia0")
            or os.path.exists("/proc/driver/nvidia"),
        }
        if HAVE_QUALITY_SYSTEM:
            try:
                from generate_episodes import get_environment_capabilities

                detected = get_environment_capabilities()
                capabilities.update(
                    {
                        "isaac_sim_available": detected.isaac_sim_available,
                        "replicator_available": detected.replicator_available,
                        "physx_available": detected.physx_available,
                        "gpu_available": detected.gpu_available,
                        "production_mode": detected.production_mode,
                        "allow_mock_capture": detected.allow_mock_capture,
                    }
                )
                return capabilities
            except Exception:
                logger.warning(
                    "[EPISODE-GEN-JOB] Failed to collect environment capabilities from quality system.",
                    exc_info=True,
                )
        if check_sensor_capture_environment is not None:
            try:
                status = check_sensor_capture_environment()
                capabilities.update(
                    {
                        "isaac_sim_available": bool(status.get("isaac_sim_available")),
                        "replicator_available": bool(status.get("replicator_available")),
                    }
                )
            except Exception:
                logger.warning(
                    "[EPISODE-GEN-JOB] Failed to collect environment capabilities from capture check.",
                    exc_info=True,
                )
        return capabilities

    def _emit_preflight_failure(capabilities: Dict[str, Any]) -> None:
        required_extensions = []
        for extension in REQUIRED_EXTENSIONS:
            if extension == "omni.replicator.core":
                available = capabilities.get("replicator_available", False)
            elif extension == "omni.physx":
                available = capabilities.get("physx_available", False)
            else:
                available = capabilities.get("isaac_sim_available", False)
            required_extensions.append(
                {
                    "name": extension,
                    "required": True,
                    "minimum_version": f"bundled with Isaac Sim {REQUIRED_ISAAC_SIM_VERSION}",
                    "available": bool(available),
                }
            )

        payload = {
            "error": "isaac_sim_preflight_failed",
            "mode": "production",
            "required": {
                "isaac_sim_version": REQUIRED_ISAAC_SIM_VERSION,
                "runtime": "Isaac Sim Python (/isaac-sim/python.sh)",
                "container_image": REQUIRED_ISAAC_SIM_CONTAINER,
                "extensions": required_extensions,
                "gpu": "NVIDIA RTX/Tesla-class GPU with drivers + nvidia-container-toolkit",
            },
            "detected": {
                "isaac_sim_available": capabilities.get("isaac_sim_available"),
                "replicator_available": capabilities.get("replicator_available"),
                "physx_available": capabilities.get("physx_available"),
                "gpu_available": capabilities.get("gpu_available"),
            },
            "remediation": [
                "Run inside the Isaac Sim container or lab runtime.",
                "Ensure the Replicator extension (omni.replicator.core) is enabled.",
                "Verify GPU drivers and NVIDIA Container Toolkit are installed.",
                "See docs/ISAAC_SIM_SETUP.md for setup steps.",
            ],
        }
        logger.error(json.dumps(payload))
        sys.exit(1)

    if _is_production_preflight():
        preflight_capabilities = _collect_preflight_capabilities()
        if (
            not preflight_capabilities.get("isaac_sim_available")
            or not preflight_capabilities.get("replicator_available")
        ):
            _emit_preflight_failure(preflight_capabilities)

    isaac_sim_available = False
    allow_mock_capture = False

    if HAVE_QUALITY_SYSTEM:
        try:
            from generate_episodes import (
                enforce_isaac_sim_for_production,
                get_data_quality_level,
                get_environment_capabilities,
                IsaacSimRequirementError,
                print_environment_report,
                ProductionDataQualityError,
            )

            capabilities = get_environment_capabilities()
            isaac_sim_available = (
                capabilities.get("isaac_sim_available", False)
                if isinstance(capabilities, dict)
                else False
            )

            print_environment_report(capabilities)

            required_quality = get_data_quality_level()
            logger.info(
                "[EPISODE-GEN-JOB] Required quality level: %s", required_quality.value
            )

            capabilities = enforce_isaac_sim_for_production(required_quality)

            logger.info("[EPISODE-GEN-JOB] ✅ Environment check passed")

            allow_mock_capture_env = parse_bool_env(
                os.getenv("ALLOW_MOCK_CAPTURE", os.getenv("ALLOW_MOCK_DATA")),
                default=False,
            )
            if required_quality.value == "production":
                allow_mock_capture = False
                if allow_mock_capture_env:
                    logger.warning(
                        "[EPISODE-GEN-JOB] ⚠️  ALLOW_MOCK_CAPTURE ignored in production quality mode"
                    )
            else:
                allow_mock_capture = allow_mock_capture_env

            if not isaac_sim_available:
                logger.warning(
                    "[EPISODE-GEN-JOB] ⚠️  WARNING: Isaac Sim not confirmed available"
                )

        except IsaacSimRequirementError as exc:
            logger.error("❌ ISAAC SIM REQUIREMENT ERROR:\n%s", exc)
            sys.exit(1)
        except ProductionDataQualityError as exc:
            logger.error("❌ PRODUCTION DATA QUALITY ERROR:\n%s", exc)
            sys.exit(1)
    else:
        logger.warning(
            "[EPISODE-GEN-JOB] Using legacy Isaac Sim check (quality system unavailable)"
        )

        replicator_available = False
        if check_sensor_capture_environment is not None:
            status = check_sensor_capture_environment()
            isaac_sim_available = status.get("isaac_sim_available", False)
            replicator_available = status.get("replicator_available", False)
            logger.info(
                "[EPISODE-GEN-JOB] Isaac Sim available: %s", isaac_sim_available
            )
        else:
            isaac_sim_available = False
            logger.warning(
                "[EPISODE-GEN-JOB] ⚠️  WARNING: Cannot verify Isaac Sim availability"
            )

        from tools.config.production_mode import resolve_production_mode

        production_env_flags = resolve_production_mode()
        is_production = (
            os.getenv("KUBERNETES_SERVICE_HOST") is not None
            or os.getenv("K_SERVICE") is not None
            or os.path.exists("/.dockerenv")
            or production_env_flags
        )

        if is_production:
            default_require_real = "true"
        else:
            default_require_real = "false"

        require_real_physics = parse_bool_env(
            os.getenv("REQUIRE_REAL_PHYSICS", default_require_real),
            default=False,
        )
        if production_env_flags:
            require_real_physics = True

        if is_production:
            allow_mock_data = False
            if parse_bool_env(os.getenv("ALLOW_MOCK_DATA"), default=False):
                logger.error("%s", "=" * 80)
                logger.error(
                    "❌ FATAL ERROR: ALLOW_MOCK_DATA not permitted in production"
                )
                logger.error("%s", "=" * 80)
                logger.error("ALLOW_MOCK_DATA=true is IGNORED in production environments.")
                logger.error("Production training data MUST use real Isaac Sim physics.")
                logger.error("This protection prevents:")
                logger.error("  - Training models on random noise images")
                logger.error("  - Wasting GPU hours on useless data")
                logger.error("  - Shipping low-quality data to labs/customers")
                logger.error("%s", "=" * 80)
        else:
            allow_mock_data = parse_bool_env(os.getenv("ALLOW_MOCK_DATA"), default=False)

        logger.info("[EPISODE-GEN-JOB] Production mode: %s", is_production)
        logger.info("[EPISODE-GEN-JOB] Require real physics: %s", require_real_physics)
        logger.info("[EPISODE-GEN-JOB] Allow mock data: %s", allow_mock_data)
        allow_mock_capture = allow_mock_data

        if require_real_physics and (not isaac_sim_available or not replicator_available):
            logger.error("%s", "=" * 80)
            if not isaac_sim_available:
                logger.error("❌ FATAL ERROR: Isaac Sim not available in production mode")
            else:
                logger.error(
                    "❌ FATAL ERROR: Replicator extension not available in production mode"
                )
            logger.error("%s", "=" * 80)
            logger.error("Episode generation requires NVIDIA Isaac Sim for:")
            logger.error("  ✓ Real physics simulation (PhysX)")
            logger.error("  ✓ Actual sensor data capture (Replicator)")
            logger.error("  ✓ Physics-validated trajectories")
            logger.error("  ✓ Collision detection with real geometry")
            logger.error("Without Isaac Sim, the pipeline would generate:")
            logger.error("  ✗ Random noise RGB images (NOT real sensor data)")
            logger.error("  ✗ Heuristic-based validation (NOT physics-verified)")
            logger.error("  ✗ Mock contact/collision data (NOT accurate)")
            logger.error("This data is USELESS for:")
            logger.error("  • Training production ML models")
            logger.error("  • Real-world robot deployment")
            logger.error("  • Lab testing and evaluation")
            logger.error("To fix this:")
            logger.error(
                "  1. Install Isaac Sim: https://developer.nvidia.com/isaac-sim"
            )
            logger.error(
                "  2. Run with Isaac Sim Python: /isaac-sim/python.sh generate_episodes.py"
            )
            logger.error(
                "  3. Or use the Isaac Sim container: see docs/ISAAC_SIM_SETUP.md"
            )
            if is_production:
                logger.error(
                    "NOTE: ALLOW_MOCK_DATA override is DISABLED in production."
                )
                logger.error(
                    "      There is no way to bypass this check in production mode."
                )
            else:
                logger.error("To bypass in development (NOT for production data):")
                logger.error("  export REQUIRE_REAL_PHYSICS=false")
            logger.error("%s", "=" * 80)
            sys.exit(1)

    if not isaac_sim_available:
        if HAVE_QUALITY_SYSTEM:
            required_quality = None
            try:
                from generate_episodes import get_data_quality_level

                required_quality = get_data_quality_level().value
            except Exception:
                required_quality = None
            if required_quality == "production" or _is_production_run():
                logger.warning(
                    "[EPISODE-GEN-JOB] WARNING: Isaac Sim not available in production mode"
                )
        if not parse_bool_env(os.getenv("REQUIRE_REAL_PHYSICS"), default=False):
            logger.warning("[EPISODE-GEN-JOB] ========================================")
            logger.warning("[EPISODE-GEN-JOB] WARNING: Isaac Sim not available")
            logger.warning(
                "[EPISODE-GEN-JOB] Running with MOCK DATA (random noise)"
            )
            logger.warning("[EPISODE-GEN-JOB] For real data: /isaac-sim/python.sh")
            logger.warning("[EPISODE-GEN-JOB] ========================================")

    env_config: EpisodeGenerationEnvConfig = parse_env_config(
        allow_mock_capture=allow_mock_capture
    )

    logger.info("[EPISODE-GEN-JOB] Configuration:")
    logger.info("[EPISODE-GEN-JOB]   Bucket: %s", env_config.bucket)
    logger.info("[EPISODE-GEN-JOB]   Scene ID: %s", env_config.scene_id)
    logger.info("[EPISODE-GEN-JOB]   Pipeline: SOTA (CP-Gen + Validation)")
    logger.info("[EPISODE-GEN-JOB]   Data Pack: %s", env_config.data_pack_tier)
    logger.info("[EPISODE-GEN-JOB]   Bundle Tier: %s", env_config.bundle_tier)
    logger.info(
        "[EPISODE-GEN-JOB]   Firebase Uploads Enabled: %s",
        env_config.enable_firebase_upload,
    )
    if env_config.lerobot_export_format:
        logger.info(
            "[EPISODE-GEN-JOB]   LeRobot export format: %s",
            env_config.lerobot_export_format,
        )
    if env_config.lerobot_parquet_compression:
        logger.info(
            "[EPISODE-GEN-JOB]   LeRobot Parquet compression: %s",
            env_config.lerobot_parquet_compression,
        )
    logger.info("[EPISODE-GEN-JOB]   Cameras: %s", env_config.num_cameras)
    logger.info("[EPISODE-GEN-JOB]   Resolution: %s", env_config.image_resolution)
    logger.info(
        "[EPISODE-GEN-JOB]   Min success rate: %.1f%%",
        env_config.min_success_rate * 100,
    )
    logger.info("[EPISODE-GEN-JOB]   Allow mock capture: %s", allow_mock_capture)

    gcs_root = Path("/mnt/gcs")

    metrics = get_metrics()
    with metrics.track_job("episode-generation-job", env_config.scene_id):
        exit_code = run_episode_generation_job(
            root=gcs_root,
            bucket=env_config.bucket,
            scene_id=env_config.scene_id,
            assets_prefix=env_config.assets_prefix,
            episodes_prefix=env_config.episodes_prefix,
            enable_firebase_upload=env_config.enable_firebase_upload,
            firebase_upload_prefix=env_config.firebase_upload_prefix,
            robot_type=env_config.robot_type,
            episodes_per_variation=env_config.episodes_per_variation,
            max_variations=env_config.max_variations,
            fps=env_config.fps,
            use_llm=env_config.use_llm,
            use_cpgen=env_config.use_cpgen,
            min_quality_score=env_config.min_quality_score,
            min_success_rate=env_config.min_success_rate,
            data_pack_tier=env_config.data_pack_tier,
            num_cameras=env_config.num_cameras,
            image_resolution=env_config.image_resolution,
            capture_sensor_data=env_config.capture_sensor_data,
            use_mock_capture=env_config.use_mock_capture,
            allow_mock_capture=env_config.allow_mock_capture,
            bundle_tier=env_config.bundle_tier,
            lerobot_export_format=env_config.lerobot_export_format,
            lerobot_parquet_compression=env_config.lerobot_parquet_compression,
        )

    sys.exit(exit_code)


def main() -> None:
    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
        },
        label="[EPISODE-GEN-JOB]",
    )
    from generate_episodes import FailureMarkerWriter, validate_scene_manifest

    bucket = os.environ["BUCKET"]
    scene_id = os.environ["SCENE_ID"]
    assets_prefix = os.getenv(
        "ASSETS_PREFIX",
        f"scenes/{scene_id}/assets",
    )
    episodes_prefix = os.getenv(
        "EPISODES_PREFIX",
        f"scenes/{scene_id}/episodes",
    )
    input_params = {
        "bucket": bucket,
        "scene_id": scene_id,
        "assets_prefix": assets_prefix,
        "episodes_prefix": episodes_prefix,
        "enable_firebase_upload": os.getenv("ENABLE_FIREBASE_UPLOAD"),
        "firebase_upload_prefix": os.getenv("FIREBASE_UPLOAD_PREFIX", "datasets"),
        "robot_type": os.getenv("ROBOT_TYPE", "g1"),
        "episodes_per_variation": os.getenv("EPISODES_PER_VARIATION", "10"),
        "max_variations": os.getenv("MAX_VARIATIONS"),
        "fps": os.getenv("FPS", "30"),
        "use_llm": os.getenv("USE_LLM", "true"),
        "use_cpgen": os.getenv("USE_CPGEN", "true"),
        "min_quality_score": os.getenv("MIN_QUALITY_SCORE", "0.85"),
        "min_success_rate": os.getenv("MIN_SUCCESS_RATE", "0.5"),
        "data_pack_tier": os.getenv("DATA_PACK_TIER", "core"),
        "num_cameras": os.getenv("NUM_CAMERAS", "1"),
        "image_resolution": os.getenv("IMAGE_RESOLUTION", "640,480"),
        "capture_sensor_data": os.getenv("CAPTURE_SENSOR_DATA", "true"),
        "use_mock_capture": os.getenv("USE_MOCK_CAPTURE", "false"),
        "bundle_tier": os.getenv("BUNDLE_TIER", "standard"),
        "lerobot_export_format": os.getenv("LEROBOT_EXPORT_FORMAT"),
        "lerobot_parquet_compression": os.getenv("LEROBOT_PARQUET_COMPRESSION"),
    }
    partial_results = {
        "episodes_prefix": episodes_prefix,
        "quality_report": (
            f"{episodes_prefix}/quality/validation_report.json" if episodes_prefix else None
        ),
    }

    def _write_failure_marker(exc: Exception, failed_step: str) -> None:
        if not bucket or not scene_id:
            logger.warning(
                "[EPISODE-GEN-JOB] WARNING: Skipping failure marker; BUCKET/SCENE_ID missing."
            )
            return
        FailureMarkerWriter(bucket, scene_id, JOB_NAME).write_failure(
            exception=exc,
            failed_step=failed_step,
            input_params=input_params,
            partial_results=partial_results,
        )

    validated = False
    try:
        assets_root = Path("/mnt/gcs") / assets_prefix
        validate_scene_manifest(assets_root / "scene_manifest.json", label="[EPISODE-GEN-JOB]")
        validated = True
        _run_main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            failed_step = "entrypoint_validation" if not validated else "entrypoint_exit"
            _write_failure_marker(RuntimeError("Job exited early"), failed_step)
        raise
    except Exception as exc:
        _write_failure_marker(exc, "entrypoint")
        raise
