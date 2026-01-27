import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

from tools.config.production_mode import resolve_production_mode
from tools.lerobot_format import LeRobotExportFormat, parse_lerobot_export_format
from tools.quality_gates.quality_gate import (
    QualityGateCheckpoint,
    QualityGateRegistry,
)

from episode_generation.constants import JOB_NAME
from episode_generation.helpers import (
    _compute_collision_free_rate,
    _gate_report_path,
    _is_production_run,
    _should_bypass_quality_gates,
)

logger = logging.getLogger(__name__)


def run_episode_generation_job(
    root: Path,
    bucket: str,
    scene_id: str,
    assets_prefix: str,
    episodes_prefix: str,
    enable_firebase_upload: bool = False,
    firebase_upload_prefix: str = "datasets",
    robot_type: str = "franka",
    episodes_per_variation: int = 10,
    max_variations: Optional[int] = None,
    fps: float = 30.0,
    use_llm: bool = True,
    use_cpgen: bool = True,
    min_quality_score: float = 0.85,
    min_success_rate: float = 0.5,
    data_pack_tier: str = "core",
    num_cameras: int = 1,
    image_resolution: Tuple[int, int] = (640, 480),
    capture_sensor_data: bool = True,
    use_mock_capture: bool = False,
    allow_mock_capture: bool = False,
    bundle_tier: str = "standard",
    lerobot_export_format: Optional[Union[str, LeRobotExportFormat]] = None,
    lerobot_parquet_compression: Optional[str] = None,
) -> int:
    """Run the episode generation job (SOTA Pipeline)."""
    from generate_episodes import (
        EpisodeGenerationConfig,
        EpisodeGenerator,
        FailureMarkerWriter,
        _load_camera_specs,
        _load_scene_config,
        _resolve_scene_usd_path,
    )

    logger.info(
        "[EPISODE-GEN-JOB] Starting SOTA episode generation for scene: %s", scene_id
    )
    logger.info("[EPISODE-GEN-JOB] Bundle tier: %s", bundle_tier)
    logger.info("[EPISODE-GEN-JOB] Assets prefix: %s", assets_prefix)
    logger.info("[EPISODE-GEN-JOB] Episodes prefix: %s", episodes_prefix)
    logger.info("[EPISODE-GEN-JOB] Firebase upload enabled: %s", enable_firebase_upload)
    logger.info("[EPISODE-GEN-JOB] Robot type: %s", robot_type)
    logger.info("[EPISODE-GEN-JOB] Episodes per variation: %s", episodes_per_variation)
    logger.info("[EPISODE-GEN-JOB] CP-Gen augmentation: %s", use_cpgen)
    logger.info("[EPISODE-GEN-JOB] Min quality score: %s", min_quality_score)
    logger.info("[EPISODE-GEN-JOB] Min success rate: %.1f%%", min_success_rate * 100)
    logger.info("[EPISODE-GEN-JOB] Data pack: %s", data_pack_tier)
    logger.info("[EPISODE-GEN-JOB] Cameras: %s", num_cameras)
    logger.info("[EPISODE-GEN-JOB] Resolution: %s", image_resolution)
    logger.info("[EPISODE-GEN-JOB] Sensor capture: %s", capture_sensor_data)
    logger.info(
        "[EPISODE-GEN-JOB] LeRobot export format: %s",
        parse_lerobot_export_format(
            lerobot_export_format,
            default=LeRobotExportFormat.LEROBOT_V2,
        ).value,
    )

    assets_dir = root / assets_prefix
    output_dir = root / episodes_prefix

    manifest_path = assets_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        logger.error("[EPISODE-GEN-JOB] Manifest not found: %s", manifest_path)
        return 1

    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        logger.info(
            "[EPISODE-GEN-JOB] Loaded manifest: %s objects",
            len(manifest.get("objects", [])),
        )
    except Exception as exc:
        logger.error("[EPISODE-GEN-JOB] Failed to load manifest: %s", exc)
        return 1

    scene_dir = root / f"scenes/{scene_id}"
    scene_config = _load_scene_config(scene_dir)
    env_scene_usd_path = os.getenv("SCENE_USD_PATH") or os.getenv("USD_SCENE_PATH")
    scene_usd_path = env_scene_usd_path or _resolve_scene_usd_path(scene_dir)
    if env_scene_usd_path:
        logger.info(
            "[EPISODE-GEN-JOB] Using USD scene path from environment: %s",
            env_scene_usd_path,
        )
    production_requested = resolve_production_mode()
    if production_requested and not scene_usd_path:
        logger.error(
            "[EPISODE-GEN-JOB] ERROR: Production runs require a USD scene path for PhysX validation. "
            "Set SCENE_USD_PATH or ensure scenes/<scene_id>/usd contains a USD file."
        )
        return 1
    camera_specs = _load_camera_specs(scene_config)
    robot_urdf_path = scene_config.get("robot_urdf_path") or os.getenv("ROBOT_URDF_PATH")

    robot_prim_paths = scene_config.get("robot_prim_paths")
    if not robot_prim_paths:
        robot_prim_path = scene_config.get("robot_prim_path")
        if robot_prim_path:
            robot_prim_paths = [robot_prim_path]
    if isinstance(robot_prim_paths, str):
        robot_prim_paths = [robot_prim_paths]

    config = EpisodeGenerationConfig(
        scene_id=scene_id,
        manifest_path=manifest_path,
        robot_type=robot_type,
        robot_prim_paths=robot_prim_paths,
        camera_specs=camera_specs,
        scene_usd_path=scene_usd_path,
        robot_urdf_path=robot_urdf_path,
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
        lerobot_export_format=parse_lerobot_export_format(
            lerobot_export_format,
            default=LeRobotExportFormat.LEROBOT_V2,
        ),
        lerobot_parquet_compression=lerobot_parquet_compression,
        output_dir=output_dir,
    )

    try:
        generator = EpisodeGenerator(config, verbose=True)
        output = generator.generate(manifest)

        if output.success:
            logger.info("[EPISODE-GEN-JOB] Episode generation completed successfully")
            logger.info("[EPISODE-GEN-JOB]   Total episodes: %s", output.total_episodes)
            logger.info("[EPISODE-GEN-JOB]   Valid episodes: %s", output.valid_episodes)
            logger.info("[EPISODE-GEN-JOB]   Pass rate: %.1f%%", output.pass_rate * 100)
            logger.info(
                "[EPISODE-GEN-JOB]   Avg quality: %.2f", output.average_quality_score
            )
            logger.info("[EPISODE-GEN-JOB]   Total frames: %s", output.total_frames)
            logger.info(
                "[EPISODE-GEN-JOB]   Duration: %.1fs",
                output.total_duration_seconds,
            )
            logger.info("[EPISODE-GEN-JOB]   Output: %s", output.output_dir)

            if bundle_tier != "standard":
                logger.info(
                    "[EPISODE-GEN-JOB] Running upsell post-processing (%s tier)...",
                    bundle_tier,
                )
                try:
                    from generate_episodes import REPO_ROOT

                    upsell_module_path = REPO_ROOT / "upsell-features-job"
                    if str(upsell_module_path) not in sys.path:
                        sys.path.insert(0, str(upsell_module_path))

                    from post_processor import run_upsell_post_processing

                    upsell_result = run_upsell_post_processing(
                        scene_dir=scene_dir,
                        tier=bundle_tier,
                        robot_type=robot_type,
                        verbose=True,
                    )

                    if upsell_result.get("success"):
                        logger.info(
                            "[EPISODE-GEN-JOB] Upsell post-processing completed successfully"
                        )
                        features = upsell_result.get("features_applied", [])
                        if features:
                            logger.info(
                                "[EPISODE-GEN-JOB]   Features applied: %s",
                                ", ".join(features),
                            )
                    else:
                        logger.warning(
                            "[EPISODE-GEN-JOB] Upsell post-processing had errors"
                        )
                        for err in upsell_result.get("errors", []):
                            logger.warning("[EPISODE-GEN-JOB]     - %s", err)

                except ImportError as exc:
                    logger.warning(
                        "[EPISODE-GEN-JOB] Upsell module not available: %s", exc
                    )
                except Exception as exc:
                    logger.warning(
                        "[EPISODE-GEN-JOB] Upsell post-processing failed: %s", exc
                    )

            if enable_firebase_upload:
                firebase_bucket = os.getenv("FIREBASE_STORAGE_BUCKET")
                firebase_service_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
                firebase_service_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
                missing_firebase = []
                if not firebase_bucket:
                    missing_firebase.append("FIREBASE_STORAGE_BUCKET")
                if not firebase_service_json and not firebase_service_path:
                    missing_firebase.append(
                        "FIREBASE_SERVICE_ACCOUNT_JSON or FIREBASE_SERVICE_ACCOUNT_PATH"
                    )
                if missing_firebase:
                    message = (
                        "Firebase upload requested but missing required configuration: "
                        + ", ".join(missing_firebase)
                    )
                    if _is_production_run():
                        logger.error("[EPISODE-GEN-JOB] %s", message)
                        output.errors.append(message)
                        return 1
                    logger.warning("[EPISODE-GEN-JOB] %s", message)
                    output.warnings.append(message)
                else:
                    logger.info(
                        "[EPISODE-GEN-JOB] Uploading episodes to Firebase Storage..."
                    )
                    try:
                        from tools.firebase_upload.uploader import (
                            upload_episodes_to_firebase,
                        )

                        if output.output_dir is None:
                            raise RuntimeError(
                                "Output directory missing; cannot upload episodes."
                            )

                        upload_summary = upload_episodes_to_firebase(
                            output.output_dir,
                            scene_id,
                            prefix=firebase_upload_prefix,
                        )
                        output.firebase_upload_summary = upload_summary
                        logger.info(
                            "[EPISODE-GEN-JOB] Firebase upload complete: %s/%s files",
                            upload_summary["uploaded"],
                            upload_summary["total_files"],
                        )
                    except Exception as exc:
                        error_message = f"Firebase upload failed: {exc}"
                        output.firebase_upload_error = error_message
                        if _is_production_run():
                            logger.error("[EPISODE-GEN-JOB] %s", error_message)
                            output.errors.append(error_message)
                            return 1
                        logger.warning("[EPISODE-GEN-JOB] %s", error_message)
                        output.warnings.append(error_message)

            if _should_bypass_quality_gates():
                logger.warning(
                    "[EPISODE-GEN-JOB] ⚠️  BYPASS_QUALITY_GATES enabled - skipping quality gates"
                )
                return 0

            collision_free_rate = _compute_collision_free_rate(
                output.validation_report_path,
                output.pass_rate,
            )
            episode_stats = {
                "total_generated": output.total_episodes,
                "passed_quality_filter": output.valid_episodes,
                "average_quality_score": output.average_quality_score,
                "collision_free_rate": collision_free_rate,
            }

            quality_gates = QualityGateRegistry(verbose=True)
            quality_gates.run_checkpoint(
                QualityGateCheckpoint.EPISODES_GENERATED,
                context={
                    "episode_stats": episode_stats,
                    "scene_id": scene_id,
                    "lerobot_dataset_path": str(output.lerobot_dataset_path)
                    if output.lerobot_dataset_path
                    else None,
                    "episode_metadata_path": str(output.lerobot_dataset_path / "meta" / "info.json")
                    if output.lerobot_dataset_path
                    else None,
                },
            )
            report_path = _gate_report_path(root, scene_id, JOB_NAME)
            quality_gates.save_report(scene_id, report_path)

            if not quality_gates.can_proceed():
                logger.error(
                    "[EPISODE-GEN-JOB] ❌ Quality gates blocked downstream pipeline"
                )
                FailureMarkerWriter(bucket, scene_id, JOB_NAME).write_failure(
                    exception=RuntimeError("Quality gates blocked: episode validation failed"),
                    failed_step="quality_gates",
                    input_params={
                        "scene_id": scene_id,
                        "episodes_prefix": episodes_prefix,
                    },
                    partial_results={"quality_gate_report": str(report_path)},
                    recommendations=[
                        "Review episode quality metrics before proceeding.",
                        f"Review quality gate report: {report_path}",
                    ],
                )
                return 1

            return 0
        logger.error(
            "[EPISODE-GEN-JOB] Generation failed with %s errors",
            len(output.errors),
        )
        for err in output.errors:
            logger.error("[EPISODE-GEN-JOB]   - %s", err)
        return 1

    except Exception as exc:
        logger.exception("[EPISODE-GEN-JOB] ERROR: %s", exc)
        return 1
