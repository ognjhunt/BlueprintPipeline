#!/usr/bin/env python3
"""
Genie Sim Export Job for BlueprintPipeline.

Cloud Run job that converts BlueprintPipeline scenes to Genie Sim 3.0 format,
enabling data generation using AGIBOT's simulation platform.

This job:
1. Loads the BlueprintPipeline scene manifest
2. Converts to Genie Sim scene graph format
3. Builds asset index for RAG retrieval
4. Generates task configuration hints
5. Generates MULTI-ROBOT configuration (DEFAULT: ENABLED)
6. Generates enhanced features config (VLA, annotations, bimanual)
7. Generates PREMIUM ANALYTICS manifests (DEFAULT: ENABLED - NO LONGER UPSELL!)
8. Outputs files ready for Genie Sim data generation

Pipeline Position:
    3D-RE-GEN → simready → usd-assembly → replicator → [THIS JOB] → Genie Sim

Enhanced Features (DEFAULT: ENABLED):
    - Multi-robot embodiment data (franka, g2, ur10, gr1, fetch, etc.)
    - Bimanual manipulation tasks
    - Multi-robot coordination scenarios
    - Rich ground truth annotations
    - VLA fine-tuning package configs

Premium Analytics (DEFAULT: ENABLED - NO LONGER UPSELL!):
    - Per-step telemetry (rewards, collisions, grasps, forces, torques)
    - Failure analysis (timeout/collision breakdown, phase-level tracking)
    - Grasp analytics (event timeline, force profiles, contact tracking)
    - Parallel eval metrics (GPU utilization, cross-env variance, throughput)

    Previously $115k-$260k upsell - NOW INCLUDED BY DEFAULT!

Environment Variables:
    BUCKET: GCS bucket name
    SCENE_ID: Scene identifier
    ASSETS_PREFIX: Path to scene assets (scene_manifest.json)
    GENIESIM_PREFIX: Output path for Genie Sim files
    ROBOT_TYPE: Primary robot type (franka, g2, ur10) - default: franka
    MAX_TASKS: Maximum suggested tasks - default: 50
    GENERATE_EMBEDDINGS: Generate semantic embeddings - default: false (true in production)
    REQUIRE_EMBEDDINGS: Require real embeddings (no placeholders) - default: false (true in production)
    FILTER_COMMERCIAL: Only include commercial-use assets - default: true
    COPY_USD: Copy USD files to output - default: true
    ENABLE_MULTI_ROBOT: Generate for multiple robot types - default: true
    ENABLE_BIMANUAL: Generate bimanual tasks - default: true
    ENABLE_VLA_PACKAGES: Generate VLA fine-tuning configs - default: true
    ENABLE_RICH_ANNOTATIONS: Generate rich annotation configs - default: true
    ENABLE_PREMIUM_ANALYTICS: Enable premium analytics capture - default: true (NO LONGER UPSELL!)
    STRICT_PREMIUM_FEATURES: Fail fast on premium feature export errors - default: false
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monitoring.alerting import send_alert
from tools.geniesim_adapter import (
    GenieSimExporter,
    GenieSimExportConfig,
    GenieSimExportResult,
)
from tools.quality_reports import generate_asset_provenance
from tools.metrics.pipeline_metrics import get_metrics
from tools.workflow.failure_markers import FailureMarkerWriter
from tools.validation.entrypoint_checks import (
    validate_required_env_vars,
    validate_scene_manifest,
)
from tools.config.env import parse_bool_env
from tools.config.production_mode import resolve_production_mode

# Import quality gates for validation before export
try:
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from quality_gates.quality_gate import (
        QualityGate,
        QualityGateCheckpoint,
        QualityGateRegistry,
        QualityGateResult,
        QualityGateSeverity,
    )
    HAVE_QUALITY_GATES = True
except ImportError:
    HAVE_QUALITY_GATES = False
    print("[GENIESIM-EXPORT-JOB] WARNING: Quality gates not available")

# Import default premium analytics (DEFAULT: ENABLED)
try:
    from .default_premium_analytics import (
        create_default_premium_analytics_exporter,
        DefaultPremiumAnalyticsConfig,
    )
    PREMIUM_ANALYTICS_AVAILABLE = True
except ImportError:
    PREMIUM_ANALYTICS_AVAILABLE = False
    print("[GENIESIM-EXPORT-JOB] WARNING: Premium analytics module not available")

# Import ALL default premium features (DEFAULT: ENABLED - NO LONGER UPSELL!)
try:
    from .default_sim2real_fidelity import create_default_sim2real_fidelity_exporter
    SIM2REAL_AVAILABLE = True
except ImportError:
    SIM2REAL_AVAILABLE = False

try:
    from .default_embodiment_transfer import create_default_embodiment_transfer_exporter
    EMBODIMENT_TRANSFER_AVAILABLE = True
except ImportError:
    EMBODIMENT_TRANSFER_AVAILABLE = False

try:
    from .default_trajectory_optimality import create_default_trajectory_optimality_exporter
    TRAJECTORY_OPTIMALITY_AVAILABLE = True
except ImportError:
    TRAJECTORY_OPTIMALITY_AVAILABLE = False

try:
    from .default_policy_leaderboard import create_default_policy_leaderboard_exporter
    POLICY_LEADERBOARD_AVAILABLE = True
except ImportError:
    POLICY_LEADERBOARD_AVAILABLE = False

try:
    from .default_tactile_sensor_sim import create_default_tactile_sensor_exporter
    TACTILE_SENSOR_AVAILABLE = True
except ImportError:
    TACTILE_SENSOR_AVAILABLE = False

try:
    from .default_language_annotations import create_default_language_annotations_exporter
    LANGUAGE_ANNOTATIONS_AVAILABLE = True
except ImportError:
    LANGUAGE_ANNOTATIONS_AVAILABLE = False

try:
    from .default_generalization_analyzer import create_default_generalization_analyzer_exporter
    GENERALIZATION_ANALYZER_AVAILABLE = True
except ImportError:
    GENERALIZATION_ANALYZER_AVAILABLE = False

try:
    from .default_sim2real_validation import create_default_sim2real_validation_exporter
    SIM2REAL_VALIDATION_AVAILABLE = True
except ImportError:
    SIM2REAL_VALIDATION_AVAILABLE = False

try:
    from .default_audio_narration import create_default_audio_narration_exporter
    AUDIO_NARRATION_AVAILABLE = True
except ImportError:
    AUDIO_NARRATION_AVAILABLE = False

JOB_NAME = "genie-sim-export-job"
COMMERCIAL_LICENSE_ALLOWLIST = {"cc0", "cc-by", "mit", "apache-2.0"}


def _normalize_license_type(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _is_commercial_license(license_type: Optional[str]) -> bool:
    return _normalize_license_type(license_type) in COMMERCIAL_LICENSE_ALLOWLIST


def parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_service_mode() -> bool:
    return (
        os.getenv("SERVICE_MODE", "").lower() in {"1", "true", "yes", "y"}
        or os.getenv("K_SERVICE") is not None
        or os.getenv("KUBERNETES_SERVICE_HOST") is not None
    )


def _resolve_scene_root_for_provenance(assets_dir: Path) -> Path:
    if (assets_dir / "assets" / "scene_manifest.json").is_file():
        return assets_dir
    if (assets_dir / "scene_manifest.json").is_file():
        return assets_dir.parent
    return assets_dir


def _embedding_provider_available() -> bool:
    return bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )


def _resolve_embedding_model() -> Optional[str]:
    if os.getenv("QWEN_EMBEDDING_MODEL"):
        return os.getenv("QWEN_EMBEDDING_MODEL")
    if os.getenv("OPENAI_EMBEDDING_MODEL"):
        return os.getenv("OPENAI_EMBEDDING_MODEL")
    if os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"):
        return "qwen-text-embedding-v4"
    if os.getenv("OPENAI_API_KEY"):
        return "text-embedding-3-large"
    return "qwen-text-embedding-v4"


def _fail_variation_assets_requirement(
    *,
    bucket: str,
    scene_id: str,
    variation_assets_prefix: Optional[str],
    reason: str,
    filter_commercial: bool,
    service_mode: bool,
) -> int:
    requirement_message = (
        "Sellable datasets require variation-gen-job assets. "
        "Run variation-gen-job and set VARIATION_ASSETS_PREFIX to its output."
    )
    print(f"[GENIESIM-EXPORT-JOB] ❌ ERROR: {reason}")
    print(f"[GENIESIM-EXPORT-JOB] ❌ ERROR: {requirement_message}")
    send_alert(
        event_type="geniesim_export_commercial_gate_failed",
        summary="Genie Sim export blocked by commercial asset gate",
        details={
            "scene_id": scene_id,
            "variation_assets_prefix": variation_assets_prefix,
            "reason": reason,
            "filter_commercial": filter_commercial,
            "service_mode": service_mode,
        },
        severity=os.getenv("ALERT_PROVENANCE_GATE_SEVERITY", "error"),
    )
    FailureMarkerWriter(bucket, scene_id, "genie-sim-export-job").write_failure(
        exception=RuntimeError(requirement_message),
        failed_step="variation_assets_validation",
        input_params={
            "scene_id": scene_id,
            "variation_assets_prefix": variation_assets_prefix,
            "filter_commercial": filter_commercial,
            "service_mode": service_mode,
        },
        recommendations=[
            "Run variation-gen-job to generate commercial-safe variation assets.",
            "Set VARIATION_ASSETS_PREFIX to the variation-gen-job output path.",
        ],
        error_code="missing_variation_assets",
    )
    return 1


def run_geniesim_export_job(
    root: Path,
    scene_id: str,
    assets_prefix: str,
    geniesim_prefix: str,
    robot_type: str = "franka",
    urdf_path: Optional[str] = None,
    max_tasks: int = 50,
    generate_embeddings: bool = False,
    require_embeddings: bool = False,
    embedding_model: Optional[str] = None,
    filter_commercial: bool = True,  # Default TRUE for commercial use
    copy_usd: bool = True,
    enable_multi_robot: bool = True,  # DEFAULT: ENABLED
    enable_bimanual: bool = True,  # DEFAULT: ENABLED
    enable_vla_packages: bool = True,  # DEFAULT: ENABLED
    enable_rich_annotations: bool = True,  # DEFAULT: ENABLED
    variation_assets_prefix: Optional[str] = None,  # Path to variation assets
    replicator_prefix: Optional[str] = None,  # Path to replicator bundle
    enable_premium_analytics: bool = True,  # DEFAULT: ENABLED (no longer upsell!)
    strict_premium_features: bool = False,
    require_quality_gates: bool = True,
    bucket: str = "",
) -> int:
    """
    Run the Genie Sim export job.

    Args:
        root: Root path (e.g., /mnt/gcs)
        scene_id: Scene identifier
        assets_prefix: Path to scene assets
        geniesim_prefix: Output path for Genie Sim files
        robot_type: Primary robot type (franka, g2, ur10)
        urdf_path: Custom URDF path for robot
        max_tasks: Maximum suggested tasks
        generate_embeddings: Generate semantic embeddings
        require_embeddings: Require real embeddings (no placeholders)
        embedding_model: Embedding model name (e.g., qwen-text-embedding-v4)
        filter_commercial: Only include commercial-use assets (DEFAULT: True)
        copy_usd: Copy USD files to output
        enable_multi_robot: Generate for multiple robot types (DEFAULT: True)
        enable_bimanual: Generate bimanual task configs (DEFAULT: True)
        enable_vla_packages: Generate VLA fine-tuning configs (DEFAULT: True)
        enable_rich_annotations: Generate rich annotation configs (DEFAULT: True)
        variation_assets_prefix: Path to variation assets (YOUR commercial assets)
        replicator_prefix: Path to replicator bundle
        enable_premium_analytics: Enable premium analytics capture (DEFAULT: True - NO LONGER UPSELL!)
        strict_premium_features: Fail fast on premium feature export errors (DEFAULT: False)
        require_quality_gates: Fail when quality gates are unavailable or error (DEFAULT: True)
        bucket: GCS bucket for failure markers (optional)

    Returns:
        0 on success, 1 on failure
    """
    production_mode = resolve_production_mode()
    require_quality_gates_env = os.getenv("REQUIRE_QUALITY_GATES")
    env_override = (
        parse_bool(require_quality_gates_env, True)
        if require_quality_gates_env is not None
        else None
    )
    if production_mode:
        if require_quality_gates is False or env_override is False:
            print(
                "[GENIESIM-EXPORT-JOB] Production mode enabled; "
                "REQUIRE_QUALITY_GATES override rejected and quality gates enforced."
            )
        require_quality_gates = True
        if not filter_commercial:
            print(
                "[GENIESIM-EXPORT-JOB] Production mode enabled; "
                "FILTER_COMMERCIAL override rejected and commercial licensing checks enforced."
            )
        filter_commercial = True
    else:
        if require_quality_gates is False or env_override is False:
            print(
                "[GENIESIM-EXPORT-JOB] Non-production mode; "
                "REQUIRE_QUALITY_GATES override honored."
            )

    print(f"[GENIESIM-EXPORT-JOB] Starting Genie Sim export for scene: {scene_id}")
    print(f"[GENIESIM-EXPORT-JOB] Assets prefix: {assets_prefix}")
    print(f"[GENIESIM-EXPORT-JOB] Output prefix: {geniesim_prefix}")
    print(f"[GENIESIM-EXPORT-JOB] Variation assets prefix: {variation_assets_prefix}")
    print(f"[GENIESIM-EXPORT-JOB] Replicator prefix: {replicator_prefix}")
    print(f"[GENIESIM-EXPORT-JOB] Primary robot type: {robot_type}")
    print(f"[GENIESIM-EXPORT-JOB] Max tasks: {max_tasks}")
    print(f"[GENIESIM-EXPORT-JOB] Generate embeddings: {generate_embeddings}")
    print(f"[GENIESIM-EXPORT-JOB] Require embeddings: {require_embeddings}")
    print(f"[GENIESIM-EXPORT-JOB] Filter commercial: {filter_commercial}")
    print(f"[GENIESIM-EXPORT-JOB] Copy USD: {copy_usd}")
    print(f"[GENIESIM-EXPORT-JOB] Multi-robot enabled: {enable_multi_robot}")
    print(f"[GENIESIM-EXPORT-JOB] Bimanual enabled: {enable_bimanual}")
    print(f"[GENIESIM-EXPORT-JOB] VLA packages enabled: {enable_vla_packages}")
    print(f"[GENIESIM-EXPORT-JOB] Rich annotations enabled: {enable_rich_annotations}")
    print(f"[GENIESIM-EXPORT-JOB] Premium analytics enabled: {enable_premium_analytics} (DEFAULT - NO LONGER UPSELL!)")
    print(f"[GENIESIM-EXPORT-JOB] Strict premium features: {strict_premium_features}")
    print(f"[GENIESIM-EXPORT-JOB] Require quality gates: {require_quality_gates}")

    assets_dir = root / assets_prefix
    output_dir = root / geniesim_prefix
    service_mode = _is_service_mode()
    commercial_checks_required = filter_commercial or service_mode or production_mode

    # Validate upstream job completion before starting export
    print("\n[GENIESIM-EXPORT-JOB] Validating upstream job completion...")
    upstream_errors = []

    # Check for USD assembly completion marker
    usd_assembly_marker = assets_dir / ".usd_assembly_complete"
    if not usd_assembly_marker.exists():
        # Try alternative location (parent directory)
        usd_assembly_marker_alt = assets_dir.parent / ".usd_assembly_complete"
        if not usd_assembly_marker_alt.exists():
            upstream_errors.append(
                "USD assembly job incomplete: .usd_assembly_complete marker not found. "
                f"Expected at: {usd_assembly_marker} or {usd_assembly_marker_alt}"
            )
        else:
            print(f"[GENIESIM-EXPORT-JOB] ✓ USD assembly complete (found at {usd_assembly_marker_alt})")
    else:
        print(f"[GENIESIM-EXPORT-JOB] ✓ USD assembly complete")

    # Check for Replicator completion marker
    replicator_marker = None
    if replicator_prefix:
        replicator_dir = root / replicator_prefix
        replicator_marker = replicator_dir / ".replicator_complete"
        if not replicator_marker.exists():
            # Try alternative location
            replicator_marker_alt = assets_dir / ".replicator_complete"
            if not replicator_marker_alt.exists():
                upstream_errors.append(
                    "Replicator job incomplete: .replicator_complete marker not found. "
                    f"Expected at: {replicator_marker} or {replicator_marker_alt}"
                )
            else:
                print(f"[GENIESIM-EXPORT-JOB] ✓ Replicator complete (found at {replicator_marker_alt})")
        else:
            print(f"[GENIESIM-EXPORT-JOB] ✓ Replicator complete")
    else:
        print("[GENIESIM-EXPORT-JOB] ⚠️  Replicator prefix not specified, skipping replicator validation")

    # Block export if upstream jobs are not complete
    if upstream_errors:
        print("\n[GENIESIM-EXPORT-JOB] ❌ ERROR: Upstream jobs not complete")
        for error in upstream_errors:
            print(f"[GENIESIM-EXPORT-JOB]   - {error}")
        print("\n[GENIESIM-EXPORT-JOB] Cannot proceed with export until upstream jobs complete.")
        print("[GENIESIM-EXPORT-JOB] Please ensure:")
        print("[GENIESIM-EXPORT-JOB]   1. usd-assembly-job has completed successfully")
        print("[GENIESIM-EXPORT-JOB]   2. replicator-job has completed successfully (if applicable)")
        return 1

    print("[GENIESIM-EXPORT-JOB] ✓ All upstream jobs validated\n")

    # Load manifest
    manifest_path = assets_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        print(f"[GENIESIM-EXPORT-JOB] ERROR: Manifest not found: {manifest_path}")
        return 1

    # Load variation assets and apply commercial filtering BEFORE merging
    # This is CRITICAL for commercial use - Genie Sim's assets are CC BY-NC-SA 4.0
    variation_assets_dir = None
    variation_objects = []
    if not variation_assets_prefix or not variation_assets_prefix.strip():
        if commercial_checks_required:
            return _fail_variation_assets_requirement(
                bucket=bucket,
                scene_id=scene_id,
                variation_assets_prefix=variation_assets_prefix,
                reason="Missing VARIATION_ASSETS_PREFIX for commercial/service export.",
                filter_commercial=filter_commercial,
                service_mode=service_mode,
            )
        print("[GENIESIM-EXPORT-JOB] WARNING: No variation_assets_prefix specified")
        print("[GENIESIM-EXPORT-JOB] WARNING: Without YOUR variation assets, you cannot sell the data commercially!")
    else:
        variation_assets_dir = root / variation_assets_prefix
        variation_assets_json = variation_assets_dir / "variation_assets.json"
        if variation_assets_json.is_file():
            print(f"[GENIESIM-EXPORT-JOB] Loading variation assets from: {variation_assets_json}")
            try:
                with open(variation_assets_json) as f:
                    variation_data = json.load(f)
                raw_variation_objects = variation_data.get("objects", [])
                print(f"[GENIESIM-EXPORT-JOB] Found {len(raw_variation_objects)} variation assets")
                if not raw_variation_objects:
                    if commercial_checks_required:
                        return _fail_variation_assets_requirement(
                            bucket=bucket,
                            scene_id=scene_id,
                            variation_assets_prefix=variation_assets_prefix,
                            reason="variation_assets.json contains no variation assets.",
                            filter_commercial=filter_commercial,
                            service_mode=service_mode,
                        )
                    print("[GENIESIM-EXPORT-JOB] WARNING: variation_assets.json has no assets")
                    raw_variation_objects = []

                # Mark these as YOUR commercial assets
                for obj in raw_variation_objects:
                    if "asset" not in obj:
                        obj["asset"] = {}
                    obj["asset"]["source"] = "blueprintpipeline_generated"
                    obj["asset"]["commercial_ok"] = True
                    obj["is_variation_asset"] = True

                # Apply commercial filtering to variation assets BEFORE merging
                if commercial_checks_required:
                    filtered_variation_objects = []
                    non_commercial_count = 0
                    non_commercial_assets = []
                    for obj in raw_variation_objects:
                        license_type = obj.get("asset", {}).get("license", "unknown")
                        license_is_commercial = _is_commercial_license(license_type)
                        if production_mode:
                            is_commercial = license_is_commercial
                        else:
                            is_commercial = obj.get("asset", {}).get("commercial_ok", False)

                        # Only include assets with commercial_ok=True or permissive licenses
                        if is_commercial or license_is_commercial:
                            filtered_variation_objects.append(obj)
                        else:
                            non_commercial_count += 1
                            asset_label = (
                                obj.get("asset", {}).get("path")
                                or obj.get("name")
                                or obj.get("id")
                                or "unknown"
                            )
                            non_commercial_assets.append(f"{asset_label} (license={license_type})")

                    variation_objects = filtered_variation_objects
                    if non_commercial_count > 0:
                        if production_mode:
                            print(
                                "[GENIESIM-EXPORT-JOB] ❌ ERROR: Production mode ignores asset.commercial_ok. "
                                "Non-permissive or missing licenses detected in variation assets."
                            )
                            print(
                                "[GENIESIM-EXPORT-JOB] ❌ ERROR: Update asset.license to one of "
                                f"{sorted(COMMERCIAL_LICENSE_ALLOWLIST)}, remove the asset, or rerun "
                                "variation-gen-job with commercial-safe assets."
                            )
                            print(
                                "[GENIESIM-EXPORT-JOB] ❌ ERROR: Example non-compliant assets: "
                                f"{non_commercial_assets[:5]}"
                            )
                        print(
                            f"[GENIESIM-EXPORT-JOB] ✓ Filtered out {non_commercial_count} NC-licensed variation assets"
                        )
                        print(
                            f"[GENIESIM-EXPORT-JOB] ✓ Retained {len(variation_objects)} commercial-safe variation assets"
                        )
                    if not variation_objects and commercial_checks_required:
                        return _fail_variation_assets_requirement(
                            bucket=bucket,
                            scene_id=scene_id,
                            variation_assets_prefix=variation_assets_prefix,
                            reason="All variation assets were filtered out as non-commercial.",
                            filter_commercial=filter_commercial,
                            service_mode=service_mode,
                        )
                else:
                    variation_objects = raw_variation_objects
                    print(
                        "[GENIESIM-EXPORT-JOB] WARNING: Commercial filtering disabled - "
                        "NC-licensed assets may be included"
                    )

            except Exception as e:
                if commercial_checks_required:
                    return _fail_variation_assets_requirement(
                        bucket=bucket,
                        scene_id=scene_id,
                        variation_assets_prefix=variation_assets_prefix,
                        reason=f"Failed to load variation assets: {e}",
                        filter_commercial=filter_commercial,
                        service_mode=service_mode,
                    )
                print(f"[GENIESIM-EXPORT-JOB] WARNING: Failed to load variation assets: {e}")
                variation_objects = []
        else:
            if commercial_checks_required:
                return _fail_variation_assets_requirement(
                    bucket=bucket,
                    scene_id=scene_id,
                    variation_assets_prefix=variation_assets_prefix,
                    reason=f"Variation assets file not found: {variation_assets_json}",
                    filter_commercial=filter_commercial,
                    service_mode=service_mode,
                )
            print(f"[GENIESIM-EXPORT-JOB] No variation assets found at: {variation_assets_json}")
            variation_objects = []

    # Find USD source directory
    usd_source_dir = None
    for possible_usd_dir in [
        assets_dir.parent / "usd",
        assets_dir / "usd",
        root / f"scenes/{scene_id}/usd",
    ]:
        if possible_usd_dir.is_dir():
            usd_source_dir = possible_usd_dir
            print(f"[GENIESIM-EXPORT-JOB] Found USD directory: {usd_source_dir}")
            break

    # Load manifest and merge with variation assets
    print(f"[GENIESIM-EXPORT-JOB] Loading manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    original_object_count = len(manifest.get("objects", []))
    print(f"[GENIESIM-EXPORT-JOB] Original manifest has {original_object_count} objects")

    # Inject USD scene path if available
    usd_path = None
    usd_search_dirs = [path for path in [usd_source_dir, output_dir / "usd"] if path]
    for usd_dir in usd_search_dirs:
        for usd_filename in ("scene.usda", "scene.usd", "scene.usdc"):
            candidate = usd_dir / usd_filename
            if candidate.is_file():
                usd_path = candidate
                break
        if usd_path:
            break

    if usd_path:
        if not manifest.get("usd_path") or not Path(str(manifest.get("usd_path"))).is_file():
            manifest["usd_path"] = str(usd_path)
            print(f"[GENIESIM-EXPORT-JOB] Added usd_path to manifest: {usd_path}")
    else:
        print("[GENIESIM-EXPORT-JOB] No USD scene file found for manifest injection")

    # Merge variation assets into manifest
    if variation_objects:
        print(f"[GENIESIM-EXPORT-JOB] Merging {len(variation_objects)} variation assets into manifest")
        if "objects" not in manifest:
            manifest["objects"] = []
        manifest["objects"].extend(variation_objects)

        # Also add to a separate key for reference
        manifest["variation_assets"] = {
            "count": len(variation_objects),
            "source": "variation-gen-job",
            "commercial_ok": True,
        }
        print(f"[GENIESIM-EXPORT-JOB] Merged manifest now has {len(manifest['objects'])} objects")
    else:
        print("[GENIESIM-EXPORT-JOB] WARNING: No variation assets to merge")
        print("[GENIESIM-EXPORT-JOB] WARNING: Scene will only have original objects")
        print("[GENIESIM-EXPORT-JOB] WARNING: For domain randomization in commercial use, you need variation assets!")

    # Write merged manifest to output directory for the exporter
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_manifest_path = output_dir / "merged_scene_manifest.json"
    with open(merged_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[GENIESIM-EXPORT-JOB] Wrote merged manifest to: {merged_manifest_path}")

    print("[GENIESIM-EXPORT-JOB] Validating merged manifest...")
    try:
        validate_scene_manifest(merged_manifest_path, label="[GENIESIM-EXPORT-JOB]")
    except SystemExit as exc:
        print(
            "[GENIESIM-EXPORT-JOB] ❌ ERROR: Merged manifest validation failed; "
            "aborting export.",
        )
        print(
            f"[GENIESIM-EXPORT-JOB] ❌ ERROR: "
            f"validate_scene_manifest exited with code {exc.code}",
        )
        return 1
    print("[GENIESIM-EXPORT-JOB] ✅ Merged manifest validation passed.")

    print("\n[GENIESIM-EXPORT-JOB] Generating asset provenance...")
    provenance_scene_root = _resolve_scene_root_for_provenance(assets_dir)
    asset_provenance_path = output_dir / "legal" / "asset_provenance.json"
    try:
        generate_asset_provenance(
            scene_dir=provenance_scene_root,
            output_path=asset_provenance_path,
            scene_id=scene_id,
            manifest_path=merged_manifest_path,
        )
        print(f"[GENIESIM-EXPORT-JOB] ✓ Asset provenance written: {asset_provenance_path}")
    except Exception as exc:
        print(f"[GENIESIM-EXPORT-JOB] ❌ ERROR: Failed to generate asset provenance: {exc}")
        return 1
    provenance_gate = {}
    try:
        asset_provenance_payload = json.loads(asset_provenance_path.read_text())
        license_info = asset_provenance_payload.get("license", {})
        commercial_use_ok = bool(license_info.get("commercial_ok", False))
        blockers = license_info.get("blockers") or []
        provenance_gate = {
            "commercial_use_ok": commercial_use_ok,
            "commercial_blockers": blockers,
        }
        print(
            "[GENIESIM-EXPORT-JOB] Asset provenance gate: "
            f"commercial_use_ok={commercial_use_ok}, blockers={len(blockers)}"
        )
        if production_mode or service_mode:
            if not commercial_use_ok or blockers:
                message = (
                    "Asset provenance gate failed for commercial export; "
                    "non-commercial or unknown licenses detected."
                )
                print(f"[GENIESIM-EXPORT-JOB] ❌ ERROR: {message}")
                if blockers:
                    print(
                        "[GENIESIM-EXPORT-JOB] ❌ ERROR: Commercial blockers: "
                        f"{blockers[:5]}"
                    )
                send_alert(
                    event_type="geniesim_export_provenance_gate_failed",
                    summary="Genie Sim export blocked by asset provenance gate",
                    details={
                        "scene_id": scene_id,
                        "asset_provenance_path": str(asset_provenance_path),
                        "commercial_use_ok": commercial_use_ok,
                        "commercial_blockers": blockers,
                        "filter_commercial": filter_commercial,
                        "service_mode": service_mode,
                        "production_mode": production_mode,
                    },
                    severity=os.getenv("ALERT_PROVENANCE_GATE_SEVERITY", "error"),
                )
                if bucket and scene_id:
                    FailureMarkerWriter(bucket, scene_id, JOB_NAME).write_failure(
                        exception=RuntimeError(message),
                        failed_step="asset_provenance_gate",
                        input_params={
                            "scene_id": scene_id,
                            "assets_prefix": assets_prefix,
                            "geniesim_prefix": geniesim_prefix,
                            "filter_commercial": filter_commercial,
                            "service_mode": service_mode,
                            "production_mode": production_mode,
                        },
                        partial_results={
                            "asset_provenance_path": str(asset_provenance_path),
                            "commercial_use_ok": commercial_use_ok,
                            "commercial_blockers": blockers,
                        },
                        recommendations=[
                            "Remove or replace non-commercial assets.",
                            "Update asset license metadata for unknown licenses.",
                        ],
                        error_code="asset_provenance_blocked",
                    )
                return 1
    except Exception as exc:
        print(f"[GENIESIM-EXPORT-JOB] ❌ ERROR: Failed to evaluate asset provenance: {exc}")
        return 1

    # Run quality gates before export
    gate_names = [
        "manifest_completeness",
        "asset_existence",
        "physics_properties",
        "scale_sanity",
    ]
    if HAVE_QUALITY_GATES:
        print("\n[GENIESIM-EXPORT-JOB] Running quality gates before export...")
        try:
            checkpoint = QualityGateCheckpoint.GENIESIM_EXPORT_READY
            registry = QualityGateRegistry(verbose=True)

            def _build_result(
                gate_id: str,
                passed: bool,
                severity: QualityGateSeverity,
                message: str,
                details: Optional[dict] = None,
            ) -> QualityGateResult:
                return QualityGateResult(
                    gate_id=gate_id,
                    checkpoint=checkpoint,
                    passed=passed,
                    severity=severity,
                    message=message,
                    details=details or {},
                )

            def _check_manifest(ctx: dict) -> QualityGateResult:
                required_fields = ["objects", "scene"]
                missing_fields = [f for f in required_fields if f not in ctx["manifest"]]
                if missing_fields:
                    return _build_result(
                        gate_id="manifest_completeness",
                        passed=False,
                        severity=QualityGateSeverity.ERROR,
                        message=f"Manifest missing required fields: {missing_fields}",
                        details={"missing_fields": missing_fields},
                    )
                return _build_result(
                    gate_id="manifest_completeness",
                    passed=True,
                    severity=QualityGateSeverity.INFO,
                    message="Manifest contains all required fields",
                )

            def _check_assets(ctx: dict) -> QualityGateResult:
                missing_assets = []
                for obj in ctx["manifest"].get("objects", []):
                    asset_path = obj.get("asset", {}).get("path")
                    if asset_path:
                        if asset_path.startswith("assets/"):
                            full_path = ctx["assets_dir"] / asset_path.removeprefix("assets/")
                        elif asset_path.startswith("variation_assets/"):
                            full_path = ctx["assets_dir"].parent / asset_path
                        else:
                            full_path = ctx["assets_dir"] / asset_path
                        if not full_path.exists():
                            missing_assets.append(asset_path)
                if missing_assets:
                    severity = (
                        QualityGateSeverity.WARNING
                        if len(missing_assets) < 5
                        else QualityGateSeverity.ERROR
                    )
                    return _build_result(
                        gate_id="asset_existence",
                        passed=False,
                        severity=severity,
                        message=(
                            f"Missing {len(missing_assets)} asset files "
                            f"(first few: {missing_assets[:3]})"
                        ),
                        details={"missing_assets": missing_assets},
                    )
                return _build_result(
                    gate_id="asset_existence",
                    passed=True,
                    severity=QualityGateSeverity.INFO,
                    message="All asset files exist",
                )

            def _check_physics(ctx: dict) -> QualityGateResult:
                objects_with_physics = sum(
                    1 for obj in ctx["manifest"].get("objects", []) if obj.get("physics")
                )
                if objects_with_physics == 0:
                    return _build_result(
                        gate_id="physics_properties",
                        passed=False,
                        severity=QualityGateSeverity.WARNING,
                        message="No objects have physics properties - scene may not simulate properly",
                        details={"objects_with_physics": objects_with_physics},
                    )
                return _build_result(
                    gate_id="physics_properties",
                    passed=True,
                    severity=QualityGateSeverity.INFO,
                    message=f"{objects_with_physics} objects have physics properties",
                    details={"objects_with_physics": objects_with_physics},
                )

            def _check_scale(ctx: dict) -> QualityGateResult:
                scale_issues = []
                for obj in ctx["manifest"].get("objects", []):
                    scale = obj.get("transform", {}).get("scale", [1, 1, 1])
                    if isinstance(scale, dict):
                        scale_values = [scale.get("x", 1), scale.get("y", 1), scale.get("z", 1)]
                    else:
                        scale_values = list(scale)
                    if any(s < 0.001 or s > 1000 for s in scale_values):
                        scale_issues.append(f"{obj.get('name', 'unknown')}: {scale_values}")
                if scale_issues:
                    return _build_result(
                        gate_id="scale_sanity",
                        passed=False,
                        severity=QualityGateSeverity.WARNING,
                        message=f"Objects with suspicious scale: {scale_issues[:3]}",
                        details={"scale_issues": scale_issues},
                    )
                return _build_result(
                    gate_id="scale_sanity",
                    passed=True,
                    severity=QualityGateSeverity.INFO,
                    message="All objects have reasonable scale",
                )

            registry.register(QualityGate(
                id="manifest_completeness",
                name="Manifest Completeness",
                checkpoint=checkpoint,
                severity=QualityGateSeverity.ERROR,
                description="Ensure required manifest sections are present.",
                check_fn=_check_manifest,
            ))
            registry.register(QualityGate(
                id="asset_existence",
                name="Asset Existence",
                checkpoint=checkpoint,
                severity=QualityGateSeverity.WARNING,
                description="Verify referenced assets exist on disk.",
                check_fn=_check_assets,
            ))
            registry.register(QualityGate(
                id="physics_properties",
                name="Physics Properties",
                checkpoint=checkpoint,
                severity=QualityGateSeverity.WARNING,
                description="Ensure objects include physics metadata.",
                check_fn=_check_physics,
            ))
            registry.register(QualityGate(
                id="scale_sanity",
                name="Scale Sanity",
                checkpoint=checkpoint,
                severity=QualityGateSeverity.WARNING,
                description="Check for out-of-range object scales.",
                check_fn=_check_scale,
            ))

            context = {
                "manifest": manifest,
                "assets_dir": assets_dir,
                "scene_id": scene_id,
                "replicator_bundle_dir": str(root / replicator_prefix) if replicator_prefix else None,
            }
            results = registry.run_checkpoint(checkpoint, context)
            found_gate_ids = {result.gate_id for result in results}
            missing_gates = [gate for gate in gate_names if gate not in found_gate_ids]
            if missing_gates:
                print(
                    "[GENIESIM-EXPORT-JOB] ❌ Quality gate evaluation incomplete; "
                    f"missing gates: {missing_gates}"
                )
                if require_quality_gates:
                    return 1
                print(
                    "[GENIESIM-EXPORT-JOB] ⚠️  Continuing without full quality gate coverage\n"
                )
            total_checks = len(results)
            checks_passed = sum(1 for result in results if result.passed)
            error_count = sum(
                1 for result in results
                if not result.passed and result.severity == QualityGateSeverity.ERROR
            )
            warning_count = sum(
                1 for result in results
                if not result.passed and result.severity == QualityGateSeverity.WARNING
            )
            status = "passed"
            if error_count:
                status = "blocked"
            elif warning_count:
                status = "warning"

            print(f"[GENIESIM-EXPORT-JOB] Quality gate result: {status}")
            print(f"[GENIESIM-EXPORT-JOB]   Passed: {checks_passed}/{total_checks}")

            if status == "blocked":
                print("\n[GENIESIM-EXPORT-JOB] ❌ Quality gates BLOCKED export")
                print(f"[GENIESIM-EXPORT-JOB] Errors: {error_count}")
                for result in results:
                    if result.severity == QualityGateSeverity.ERROR and not result.passed:
                        print(f"[GENIESIM-EXPORT-JOB]   ERROR: {result.gate_id}: {result.message}")
                return 1

            if status == "warning":
                print(f"[GENIESIM-EXPORT-JOB] ⚠️  Quality gates passed with warnings ({warning_count})")

            print("[GENIESIM-EXPORT-JOB] ✅ Quality gates passed\n")
        except Exception as e:
            print(
                "[GENIESIM-EXPORT-JOB] ❌ Quality gate evaluation failed; "
                f"gates impacted: {gate_names}. Error: {e}"
            )
            if require_quality_gates:
                return 1
            print("[GENIESIM-EXPORT-JOB] ⚠️  Continuing without quality gate validation\n")
    else:
        print(
            "[GENIESIM-EXPORT-JOB] ❌ Quality gates not available; "
            f"expected gates: {gate_names}."
        )
        print(f"[GENIESIM-EXPORT-JOB] ❌ Missing gate implementations: {gate_names}")
        if require_quality_gates:
            return 1
        print("[GENIESIM-EXPORT-JOB] ⚠️  Continuing without quality gate validation\n")

    embedding_provider_available = _embedding_provider_available()
    embedding_mode = (
        "provider"
        if generate_embeddings and embedding_provider_available
        else "placeholder"
    )

    # Configure exporter with enhanced features
    resolved_embedding_model = embedding_model or _resolve_embedding_model()
    if resolved_embedding_model:
        print(f"[GENIESIM-EXPORT-JOB] Embedding model: {resolved_embedding_model}")
    config = GenieSimExportConfig(
        robot_type=robot_type,
        urdf_path=urdf_path,
        generate_embeddings=generate_embeddings,
        require_embeddings=require_embeddings,
        embedding_model=resolved_embedding_model,
        max_tasks=max_tasks,
        copy_usd_files=copy_usd,
        filter_commercial_only=filter_commercial,
        # Enhanced features (DEFAULT: ENABLED)
        enable_multi_robot=enable_multi_robot,
        enable_bimanual=enable_bimanual,
        enable_vla_packages=enable_vla_packages,
        enable_rich_annotations=enable_rich_annotations,
        enable_multi_robot_coordination=enable_multi_robot,  # Tied to multi_robot
    )

    try:
        exporter = GenieSimExporter(config, verbose=True)
        # Use merged manifest that includes YOUR variation assets
        result = exporter.export(
            manifest_path=merged_manifest_path,
            output_dir=output_dir,
            usd_source_dir=usd_source_dir if copy_usd else None,
        )

        if result.success:
            print("\n[GENIESIM-EXPORT-JOB] Export completed successfully")
            print(f"[GENIESIM-EXPORT-JOB]   Scene Graph: {result.scene_graph_path}")
            print(f"[GENIESIM-EXPORT-JOB]   Asset Index: {result.asset_index_path}")
            print(f"[GENIESIM-EXPORT-JOB]   Task Config: {result.task_config_path}")
            print(f"[GENIESIM-EXPORT-JOB]   Nodes: {result.num_nodes}")
            print(f"[GENIESIM-EXPORT-JOB]   Edges: {result.num_edges}")
            print(f"[GENIESIM-EXPORT-JOB]   Assets: {result.num_assets}")
            print(f"[GENIESIM-EXPORT-JOB]   Tasks: {result.num_tasks}")

            # Export premium analytics manifests (DEFAULT: ENABLED)
            premium_analytics_manifests = {}
            premium_feature_counts = {}
            premium_feature_status = {}
            export_warnings = []

            def record_premium_warning(feature_name: str, exc: Exception, feature_output_dir: Path) -> None:
                warning = {
                    "feature": feature_name,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "output_dir": str(feature_output_dir),
                }
                export_warnings.append(warning)
                print(
                    "[GENIESIM-EXPORT-JOB] ⚠️  Premium feature failed but continuing: "
                    f"{feature_name} ({warning['exception_type']}: {warning['exception_message']})"
                )
                print(f"[GENIESIM-EXPORT-JOB] ⚠️  Output dir: {warning['output_dir']}")
                if strict_premium_features:
                    print(
                        "[GENIESIM-EXPORT-JOB] ❌ Premium feature failure blocked export "
                        "due to STRICT_PREMIUM_FEATURES=1"
                    )
                    raise exc

            if enable_premium_analytics and PREMIUM_ANALYTICS_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting premium analytics manifests (DEFAULT - NO LONGER UPSELL)")
                analytics_dir = output_dir / "premium_analytics"
                try:
                    analytics_config = DefaultPremiumAnalyticsConfig(enabled=True)
                    analytics_exporter = create_default_premium_analytics_exporter(
                        scene_id=scene_id,
                        output_dir=analytics_dir,
                        config=analytics_config,
                    )
                    premium_analytics_manifests = analytics_exporter.export_all_manifests()
                    premium_feature_counts["premium_analytics"] = len(premium_analytics_manifests)
                    premium_feature_status["premium_analytics"] = True
                    print(
                        "[GENIESIM-EXPORT-JOB]   ✓ Premium analytics: "
                        f"{len(premium_analytics_manifests)} manifests exported"
                    )
                    print("[GENIESIM-EXPORT-JOB]   ✓ Per-step telemetry capture enabled")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Failure analysis enabled")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Grasp analytics enabled")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Parallel eval metrics enabled")
                except Exception as exc:
                    record_premium_warning("premium_analytics", exc, analytics_dir)
            elif not enable_premium_analytics:
                print("\n[GENIESIM-EXPORT-JOB] Premium analytics disabled (not recommended)")
            elif not PREMIUM_ANALYTICS_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] WARNING: Premium analytics module not available")

            # Export ALL additional premium features (DEFAULT: ENABLED - NO LONGER UPSELL!)
            all_premium_features_manifests = {}

            # 1. Sim2Real Fidelity Matrix ($20k-$50k value)
            if SIM2REAL_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Sim2Real Fidelity Matrix ($20k-$50k value - NOW FREE)")
                sim2real_dir = output_dir / "sim2real_fidelity"
                try:
                    sim2real_manifests = create_default_sim2real_fidelity_exporter(
                        scene_id=scene_id,
                        robot_type=robot_type,
                        output_dir=sim2real_dir,
                    )
                    all_premium_features_manifests.update({"sim2real": sim2real_manifests})
                    premium_feature_counts["sim2real_fidelity"] = len(sim2real_manifests)
                    premium_feature_status["sim2real_fidelity"] = True
                    print(
                        "[GENIESIM-EXPORT-JOB]   ✓ Sim2Real Fidelity: "
                        f"{len(sim2real_manifests)} manifests exported"
                    )
                    print("[GENIESIM-EXPORT-JOB]   ✓ Physics/Visual/Sensor fidelity scoring")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Transfer confidence score")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Trust matrix for deployment")
                except Exception as exc:
                    record_premium_warning("sim2real_fidelity", exc, sim2real_dir)

            # 2. Embodiment Transfer Analysis ($20k-$100k value)
            if EMBODIMENT_TRANSFER_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Embodiment Transfer Analysis ($20k-$100k value - NOW FREE)")
                embodiment_dir = output_dir / "embodiment_transfer"
                try:
                    embodiment_manifests = create_default_embodiment_transfer_exporter(
                        scene_id=scene_id,
                        source_robot=robot_type,
                        output_dir=embodiment_dir,
                    )
                    all_premium_features_manifests.update({"embodiment": embodiment_manifests})
                    premium_feature_counts["embodiment_transfer"] = len(embodiment_manifests)
                    premium_feature_status["embodiment_transfer"] = True
                    print(
                        "[GENIESIM-EXPORT-JOB]   ✓ Embodiment Transfer: "
                        f"{len(embodiment_manifests)} manifests exported"
                    )
                    print("[GENIESIM-EXPORT-JOB]   ✓ Cross-robot compatibility matrix")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Multi-robot data multiplier")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Transfer strategy recommendations")
                except Exception as exc:
                    record_premium_warning("embodiment_transfer", exc, embodiment_dir)

            # 3. Trajectory Optimality Analysis ($10k-$25k value)
            if TRAJECTORY_OPTIMALITY_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Trajectory Optimality Analysis ($10k-$25k value - NOW FREE)")
                trajectory_dir = output_dir / "trajectory_optimality"
                try:
                    trajectory_manifests = create_default_trajectory_optimality_exporter(
                        scene_id=scene_id,
                        output_dir=trajectory_dir,
                    )
                    all_premium_features_manifests.update({"trajectory": trajectory_manifests})
                    premium_feature_counts["trajectory_optimality"] = len(trajectory_manifests)
                    premium_feature_status["trajectory_optimality"] = True
                    print(
                        "[GENIESIM-EXPORT-JOB]   ✓ Trajectory Optimality: "
                        f"{len(trajectory_manifests)} manifests exported"
                    )
                    print("[GENIESIM-EXPORT-JOB]   ✓ Path efficiency scoring")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Smoothness/jerk analysis")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Training suitability assessment")
                except Exception as exc:
                    record_premium_warning("trajectory_optimality", exc, trajectory_dir)

            # 4. Policy Leaderboard ($20k-$40k value)
            if POLICY_LEADERBOARD_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Policy Leaderboard ($20k-$40k value - NOW FREE)")
                leaderboard_dir = output_dir / "policy_leaderboard"
                try:
                    leaderboard_manifests = create_default_policy_leaderboard_exporter(
                        scene_id=scene_id,
                        output_dir=leaderboard_dir,
                    )
                    all_premium_features_manifests.update({"leaderboard": leaderboard_manifests})
                    premium_feature_counts["policy_leaderboard"] = len(leaderboard_manifests)
                    premium_feature_status["policy_leaderboard"] = True
                    print(
                        "[GENIESIM-EXPORT-JOB]   ✓ Policy Leaderboard: "
                        f"{len(leaderboard_manifests)} manifests exported"
                    )
                    print("[GENIESIM-EXPORT-JOB]   ✓ Multi-policy comparison with confidence intervals")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Statistical significance testing")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Pairwise comparison matrix")
                except Exception as exc:
                    record_premium_warning("policy_leaderboard", exc, leaderboard_dir)

            # 5. Tactile Sensor Simulation ($15k-$30k value)
            if TACTILE_SENSOR_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Tactile Sensor Simulation ($15k-$30k value - NOW FREE)")
                tactile_dir = output_dir / "tactile_sensors"
                try:
                    tactile_manifests = create_default_tactile_sensor_exporter(
                        scene_id=scene_id,
                        output_dir=tactile_dir,
                    )
                    all_premium_features_manifests.update({"tactile": tactile_manifests})
                    premium_feature_counts["tactile_sensors"] = len(tactile_manifests)
                    premium_feature_status["tactile_sensors"] = True
                    print(
                        "[GENIESIM-EXPORT-JOB]   ✓ Tactile Sensors: "
                        f"{len(tactile_manifests)} manifests exported"
                    )
                    print("[GENIESIM-EXPORT-JOB]   ✓ GelSlim/GelSight/DIGIT simulation")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Contact force maps")
                    print("[GENIESIM-EXPORT-JOB]   ✓ 81%+ success vs 50% vision-only")
                except Exception as exc:
                    record_premium_warning("tactile_sensors", exc, tactile_dir)

            # 6. Language Annotations ($10k-$25k value)
            if LANGUAGE_ANNOTATIONS_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Language Annotations ($10k-$25k value - NOW FREE)")
                language_dir = output_dir / "language_annotations"
                try:
                    language_manifests = create_default_language_annotations_exporter(
                        scene_id=scene_id,
                        output_dir=language_dir,
                    )
                    all_premium_features_manifests.update({"language": language_manifests})
                    premium_feature_counts["language_annotations"] = len(language_manifests)
                    premium_feature_status["language_annotations"] = True
                    print(
                        "[GENIESIM-EXPORT-JOB]   ✓ Language Annotations: "
                        f"{len(language_manifests)} manifests exported"
                    )
                    print("[GENIESIM-EXPORT-JOB]   ✓ Template + LLM-powered generation")
                    print("[GENIESIM-EXPORT-JOB]   ✓ 10+ variations per task")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Required for VLA training (OpenVLA, Pi0, RT-2)")
                except Exception as exc:
                    record_premium_warning("language_annotations", exc, language_dir)

            # 7. Generalization Analyzer ($15k-$35k value)
            if GENERALIZATION_ANALYZER_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Generalization Analyzer ($15k-$35k value - NOW FREE)")
                generalization_dir = output_dir / "generalization_analysis"
                try:
                    generalization_manifests = create_default_generalization_analyzer_exporter(
                        scene_id=scene_id,
                        output_dir=generalization_dir,
                    )
                    all_premium_features_manifests.update({"generalization": generalization_manifests})
                    premium_feature_counts["generalization_analyzer"] = len(generalization_manifests)
                    premium_feature_status["generalization_analyzer"] = True
                    print(
                        "[GENIESIM-EXPORT-JOB]   ✓ Generalization Analyzer: "
                        f"{len(generalization_manifests)} manifests exported"
                    )
                    print("[GENIESIM-EXPORT-JOB]   ✓ Per-object success rate analysis")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Learning curve computation")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Curriculum learning recommendations")
                except Exception as exc:
                    record_premium_warning("generalization_analyzer", exc, generalization_dir)

            # 8. Sim2Real Validation Service ($5k-$25k/study value)
            if SIM2REAL_VALIDATION_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Sim2Real Validation Service ($5k-$25k/study - NOW FREE)")
                sim2real_validation_dir = output_dir / "sim2real_validation"
                try:
                    sim2real_validation_manifests = create_default_sim2real_validation_exporter(
                        scene_id=scene_id,
                        robot_type=robot_type,
                        output_dir=sim2real_validation_dir,
                    )
                    all_premium_features_manifests.update({"sim2real_validation": sim2real_validation_manifests})
                    premium_feature_counts["sim2real_validation"] = len(sim2real_validation_manifests)
                    premium_feature_status["sim2real_validation"] = True
                    print(
                        "[GENIESIM-EXPORT-JOB]   ✓ Sim2Real Validation: "
                        f"{len(sim2real_validation_manifests)} manifests exported"
                    )
                    print("[GENIESIM-EXPORT-JOB]   ✓ Real-world validation trial tracking")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Sim vs real success rate comparison")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Quality guarantee certificates (50%/70%/85%)")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Failure mode comparison (sim vs real)")
                except Exception as exc:
                    record_premium_warning("sim2real_validation", exc, sim2real_validation_dir)

            # 9. Audio Narration ($5k-$15k value)
            if AUDIO_NARRATION_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Audio Narration ($5k-$15k value - NOW FREE)")
                audio_narration_dir = output_dir / "audio_narration"
                try:
                    audio_narration_manifests = create_default_audio_narration_exporter(
                        scene_id=scene_id,
                        output_dir=audio_narration_dir,
                    )
                    all_premium_features_manifests.update({"audio_narration": audio_narration_manifests})
                    premium_feature_counts["audio_narration"] = len(audio_narration_manifests)
                    premium_feature_status["audio_narration"] = True
                    print(
                        "[GENIESIM-EXPORT-JOB]   ✓ Audio Narration: "
                        f"{len(audio_narration_manifests)} manifests exported"
                    )
                    print("[GENIESIM-EXPORT-JOB]   ✓ Text-to-speech narration (Google Cloud TTS + local)")
                    print("[GENIESIM-EXPORT-JOB]   ✓ Multi-voice presets (narrator, instructor, casual, robot)")
                    print("[GENIESIM-EXPORT-JOB]   ✓ MP3/WAV/OGG audio output")
                    print("[GENIESIM-EXPORT-JOB]   ✓ VLA audio modality training (RT-2, PaLM-E)")
                except Exception as exc:
                    record_premium_warning("audio_narration", exc, audio_narration_dir)

            # Summary of premium features
            if any([SIM2REAL_AVAILABLE, EMBODIMENT_TRANSFER_AVAILABLE, TRAJECTORY_OPTIMALITY_AVAILABLE,
                   POLICY_LEADERBOARD_AVAILABLE, TACTILE_SENSOR_AVAILABLE, LANGUAGE_ANNOTATIONS_AVAILABLE,
                   GENERALIZATION_ANALYZER_AVAILABLE, SIM2REAL_VALIDATION_AVAILABLE, AUDIO_NARRATION_AVAILABLE]):
                print("\n" + "="*80)
                print("  🎉 PREMIUM FEATURES EXPORTED (DEFAULT - FREE)")
                print("="*80)
                total_value = 0
                features_exported = []
                if premium_feature_status.get("sim2real_fidelity"):
                    features_exported.append("Sim2Real Fidelity Matrix ($20k-$50k)")
                    total_value += 35000
                if premium_feature_status.get("embodiment_transfer"):
                    features_exported.append("Embodiment Transfer Analysis ($20k-$100k)")
                    total_value += 60000
                if premium_feature_status.get("trajectory_optimality"):
                    features_exported.append("Trajectory Optimality Analysis ($10k-$25k)")
                    total_value += 17500
                if premium_feature_status.get("policy_leaderboard"):
                    features_exported.append("Policy Leaderboard ($20k-$40k)")
                    total_value += 30000
                if premium_feature_status.get("tactile_sensors"):
                    features_exported.append("Tactile Sensor Simulation ($15k-$30k)")
                    total_value += 22500
                if premium_feature_status.get("language_annotations"):
                    features_exported.append("Language Annotations ($10k-$25k)")
                    total_value += 17500
                if premium_feature_status.get("generalization_analyzer"):
                    features_exported.append("Generalization Analyzer ($15k-$35k)")
                    total_value += 25000
                if premium_feature_status.get("sim2real_validation"):
                    features_exported.append("Sim2Real Validation Service ($5k-$25k/study)")
                    total_value += 15000
                if premium_feature_status.get("audio_narration"):
                    features_exported.append("Audio Narration ($5k-$15k)")
                    total_value += 10000

                for feature in features_exported:
                    print(f"  ✓ {feature}")
                print(f"\n  💰 Total Value Delivered: ${total_value:,} (NOW FREE BY DEFAULT!)")
                print("="*80 + "\n")

            # Write completion marker with schema version tracking
            import datetime
            marker_path = output_dir / "_GENIESIM_EXPORT_COMPLETE"
            metrics = get_metrics()
            metrics_summary = {
                "backend": metrics.backend.value,
                "stats": metrics.get_stats(),
            }
            marker_data = {
                "scene_id": scene_id,
                "robot_type": robot_type,
                "success": True,
                "commercial_data": filter_commercial,
                "premium_analytics_enabled": enable_premium_analytics and PREMIUM_ANALYTICS_AVAILABLE,
                # Add schema version tracking
                "export_schema_version": "1.0.0",  # BlueprintPipeline export schema version
                "geniesim_schema_version": "3.0.0",  # Genie Sim API version compatibility
                "blueprintpipeline_version": "1.0.0",  # Pipeline version
                "export_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "schema_compatibility": {
                    "min_geniesim_version": "3.0.0",
                    "max_geniesim_version": "3.x.x",
                    "breaking_changes_since": "2.x.x",
                },
                "stats": {
                    "nodes": result.num_nodes,
                    "edges": result.num_edges,
                    "assets": result.num_assets,
                    "tasks": result.num_tasks,
                    "original_objects": original_object_count,
                    "variation_assets": len(variation_objects),
                },
                "metrics_summary": metrics_summary,
                "embedding_requirements": {
                    "generate_embeddings": generate_embeddings,
                    "require_embeddings": require_embeddings,
                    "production_mode": production_mode,
                },
                "embedding_provider_available": embedding_provider_available,
                "embedding_mode": embedding_mode,
                # Track which premium features were exported
                "premium_features_exported": {
                    "premium_analytics": premium_feature_status.get("premium_analytics", False),
                    "sim2real_fidelity": premium_feature_status.get("sim2real_fidelity", False),
                    "embodiment_transfer": premium_feature_status.get("embodiment_transfer", False),
                    "trajectory_optimality": premium_feature_status.get("trajectory_optimality", False),
                    "policy_leaderboard": premium_feature_status.get("policy_leaderboard", False),
                    "tactile_sensors": premium_feature_status.get("tactile_sensors", False),
                    "language_annotations": premium_feature_status.get("language_annotations", False),
                    "generalization_analyzer": premium_feature_status.get("generalization_analyzer", False),
                    "sim2real_validation": premium_feature_status.get("sim2real_validation", False),
                    "audio_narration": premium_feature_status.get("audio_narration", False),
                },
                "premium_features_failed": export_warnings,
            }
            if "premium_analytics" in premium_feature_counts:
                marker_data["premium_analytics_manifests"] = premium_feature_counts["premium_analytics"]
            if premium_feature_counts:
                marker_data["premium_feature_counts"] = premium_feature_counts
            marker_path.write_text(json.dumps(marker_data, indent=2))
            print(f"\n[GENIESIM-EXPORT-JOB] ✓ Completion marker written with schema v{marker_data['export_schema_version']}")

            return 0
        else:
            print(f"[GENIESIM-EXPORT-JOB] ERROR: Export failed")
            for error in result.errors:
                print(f"[GENIESIM-EXPORT-JOB]   - {error}")
            return 1

    except Exception as e:
        print(f"[GENIESIM-EXPORT-JOB] ERROR: {e}")
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    # Validate credentials at startup
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        from startup_validation import validate_and_fail_fast
        # Genie Sim credentials not required for export (only for import)
        validate_and_fail_fast(
            job_name="GENIESIM-EXPORT-JOB",
            require_geniesim=False,
            require_gemini=False,
            validate_gcs=True,
        )
    except ImportError as e:
        print(f"[GENIESIM-EXPORT-JOB] WARNING: Startup validation unavailable: {e}")

    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
        },
        label="[GENIESIM-EXPORT-JOB]",
    )

    bucket = os.environ["BUCKET"]
    scene_id = os.environ["SCENE_ID"]

    # Prefixes with defaults
    assets_prefix = os.getenv(
        "ASSETS_PREFIX",
        f"scenes/{scene_id}/assets" if scene_id else "",
    )
    geniesim_prefix = os.getenv(
        "GENIESIM_PREFIX",
        f"scenes/{scene_id}/geniesim" if scene_id else "",
    )
    variation_assets_prefix = os.getenv(
        "VARIATION_ASSETS_PREFIX",
        f"scenes/{scene_id}/variation_assets" if scene_id else "",
    )
    replicator_prefix = os.getenv(
        "REPLICATOR_PREFIX",
        f"scenes/{scene_id}/replicator" if scene_id else "",
    )

    # Configuration
    production_mode = resolve_production_mode()
    robot_type = os.getenv("ROBOT_TYPE", "franka")
    urdf_path = os.getenv("URDF_PATH")  # Optional custom URDF
    max_tasks = int(os.getenv("MAX_TASKS", "50"))
    generate_embeddings = parse_bool_env(
        os.getenv("GENERATE_EMBEDDINGS"),
        default=production_mode,
    )
    require_embeddings = parse_bool_env(
        os.getenv("REQUIRE_EMBEDDINGS"),
        default=production_mode,
    )
    filter_commercial = parse_bool_env(os.getenv("FILTER_COMMERCIAL"), default=True)
    copy_usd = parse_bool_env(os.getenv("COPY_USD"), default=True)
    enable_multi_robot = parse_bool_env(os.getenv("ENABLE_MULTI_ROBOT"), default=True)
    enable_bimanual = parse_bool_env(os.getenv("ENABLE_BIMANUAL"), default=True)
    enable_vla_packages = parse_bool_env(os.getenv("ENABLE_VLA_PACKAGES"), default=True)
    enable_rich_annotations = parse_bool_env(os.getenv("ENABLE_RICH_ANNOTATIONS"), default=True)
    enable_premium_analytics = parse_bool_env(os.getenv("ENABLE_PREMIUM_ANALYTICS"), default=True)
    require_quality_gates = parse_bool(os.getenv("REQUIRE_QUALITY_GATES"), True)
    strict_premium_features = parse_bool_env(os.getenv("STRICT_PREMIUM_FEATURES"), default=False)
    embedding_model = _resolve_embedding_model()
    if production_mode and not require_quality_gates:
        require_quality_gates = True

    input_params = {
        "bucket": bucket,
        "scene_id": scene_id,
        "assets_prefix": assets_prefix,
        "geniesim_prefix": geniesim_prefix,
        "variation_assets_prefix": variation_assets_prefix,
        "replicator_prefix": replicator_prefix,
        "robot_type": robot_type,
        "urdf_path": urdf_path,
        "max_tasks": max_tasks,
        "generate_embeddings": generate_embeddings,
        "require_embeddings": require_embeddings,
        "embedding_model": embedding_model,
        "filter_commercial": filter_commercial,
        "copy_usd": copy_usd,
        "enable_multi_robot": enable_multi_robot,
        "enable_bimanual": enable_bimanual,
        "enable_vla_packages": enable_vla_packages,
        "enable_rich_annotations": enable_rich_annotations,
        "enable_premium_analytics": enable_premium_analytics,
        "strict_premium_features": strict_premium_features,
        "require_quality_gates": require_quality_gates,
    }
    partial_results = {
        "geniesim_output_prefix": geniesim_prefix,
        "merged_manifest_path": (
            f"{geniesim_prefix}/merged_scene_manifest.json" if geniesim_prefix else None
        ),
    }

    def _write_failure_marker(exc: Exception, failed_step: str) -> None:
        if not bucket or not scene_id:
            print(
                "[GENIESIM-EXPORT-JOB] WARNING: Skipping failure marker; BUCKET/SCENE_ID missing.",
            )
            return
        FailureMarkerWriter(bucket, scene_id, JOB_NAME).write_failure(
            exception=exc,
            failed_step=failed_step,
            input_params=input_params,
            partial_results=partial_results,
        )

    embedding_provider_available = _embedding_provider_available()
    if production_mode and require_embeddings and not embedding_provider_available:
        message = (
            "Production mode requires embeddings, but no embedding provider keys were found. "
            "Set OPENAI_API_KEY or QWEN_API_KEY/DASHSCOPE_API_KEY, or explicitly disable "
            "REQUIRE_EMBEDDINGS."
        )
        print(f"[GENIESIM-EXPORT-JOB] ❌ ERROR: {message}")
        _write_failure_marker(RuntimeError(message), "embedding_provider_validation")
        sys.exit(1)
    if generate_embeddings and not require_embeddings and not embedding_provider_available:
        print(
            "[GENIESIM-EXPORT-JOB] ⚠️  Embedding provider unavailable; "
            "placeholder embeddings will be used."
        )

    validated = False
    try:
        assets_root = Path("/mnt/gcs") / assets_prefix
        validate_scene_manifest(assets_root / "scene_manifest.json", label="[GENIESIM-EXPORT-JOB]")
        validated = True

        print("[GENIESIM-EXPORT-JOB] Configuration:")
        print(f"[GENIESIM-EXPORT-JOB]   Bucket: {bucket}")
        print(f"[GENIESIM-EXPORT-JOB]   Scene ID: {scene_id}")
        print(f"[GENIESIM-EXPORT-JOB]   Variation Assets: {variation_assets_prefix}")
        print(f"[GENIESIM-EXPORT-JOB]   Replicator Bundle: {replicator_prefix}")
        print(f"[GENIESIM-EXPORT-JOB]   Primary Robot Type: {robot_type}")
        print(f"[GENIESIM-EXPORT-JOB]   Max Tasks: {max_tasks}")
        print(f"[GENIESIM-EXPORT-JOB]   Require Embeddings: {require_embeddings}")
        print(f"[GENIESIM-EXPORT-JOB]   Multi-Robot: {enable_multi_robot}")
        print(f"[GENIESIM-EXPORT-JOB]   Bimanual: {enable_bimanual}")
        print(f"[GENIESIM-EXPORT-JOB]   VLA Packages: {enable_vla_packages}")
        print(f"[GENIESIM-EXPORT-JOB]   Rich Annotations: {enable_rich_annotations}")
        print(f"[GENIESIM-EXPORT-JOB]   Commercial Filter: {filter_commercial}")
        print(f"[GENIESIM-EXPORT-JOB]   Premium Analytics: {enable_premium_analytics} (DEFAULT - NO LONGER UPSELL!)")
        print(f"[GENIESIM-EXPORT-JOB]   Strict Premium Features: {strict_premium_features}")
        print(f"[GENIESIM-EXPORT-JOB]   Require Quality Gates: {require_quality_gates}")

        GCS_ROOT = Path("/mnt/gcs")

        metrics = get_metrics()
        with metrics.track_job(JOB_NAME, scene_id):
            exit_code = run_geniesim_export_job(
                root=GCS_ROOT,
                scene_id=scene_id,
                assets_prefix=assets_prefix,
                geniesim_prefix=geniesim_prefix,
                robot_type=robot_type,
                urdf_path=urdf_path,
                max_tasks=max_tasks,
                generate_embeddings=generate_embeddings,
                require_embeddings=require_embeddings,
                embedding_model=embedding_model,
                filter_commercial=filter_commercial,
                copy_usd=copy_usd,
                # Enhanced features (DEFAULT: ENABLED)
                enable_multi_robot=enable_multi_robot,
                enable_bimanual=enable_bimanual,
                enable_vla_packages=enable_vla_packages,
                enable_rich_annotations=enable_rich_annotations,
                # YOUR commercial assets for domain randomization
                variation_assets_prefix=variation_assets_prefix,
                replicator_prefix=replicator_prefix,
                # Premium analytics (DEFAULT: ENABLED - NO LONGER UPSELL!)
                enable_premium_analytics=enable_premium_analytics,
                strict_premium_features=strict_premium_features,
                require_quality_gates=require_quality_gates,
                bucket=bucket,
            )

        sys.exit(exit_code)
    except SystemExit as exc:
        if exc.code not in (0, None):
            failed_step = "entrypoint_validation" if not validated else "entrypoint_exit"
            _write_failure_marker(RuntimeError("Job exited early"), failed_step)
        raise
    except Exception as exc:
        _write_failure_marker(exc, "entrypoint")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        send_alert(
            event_type="geniesim_export_job_fatal_exception",
            summary="Genie Sim export job failed with an unhandled exception",
            details={
                "job": "genie-sim-export-job",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            severity=os.getenv("ALERT_JOB_EXCEPTION_SEVERITY", "critical"),
        )
        raise
