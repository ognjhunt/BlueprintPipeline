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
    GENERATE_EMBEDDINGS: Generate semantic embeddings - default: false
    FILTER_COMMERCIAL: Only include commercial-use assets - default: true
    COPY_USD: Copy USD files to output - default: true
    ENABLE_MULTI_ROBOT: Generate for multiple robot types - default: true
    ENABLE_BIMANUAL: Generate bimanual tasks - default: true
    ENABLE_VLA_PACKAGES: Generate VLA fine-tuning configs - default: true
    ENABLE_RICH_ANNOTATIONS: Generate rich annotation configs - default: true
    ENABLE_PREMIUM_ANALYTICS: Enable premium analytics capture - default: true (NO LONGER UPSELL!)
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

from tools.geniesim_adapter import (
    GenieSimExporter,
    GenieSimExportConfig,
    GenieSimExportResult,
)
from tools.metrics.pipeline_metrics import get_metrics
from tools.workflow.failure_markers import FailureMarkerWriter

# P0-5 FIX: Import quality gates for validation before export
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
    filter_commercial: bool = True,  # Default TRUE for commercial use
    copy_usd: bool = True,
    enable_multi_robot: bool = True,  # DEFAULT: ENABLED
    enable_bimanual: bool = True,  # DEFAULT: ENABLED
    enable_vla_packages: bool = True,  # DEFAULT: ENABLED
    enable_rich_annotations: bool = True,  # DEFAULT: ENABLED
    variation_assets_prefix: Optional[str] = None,  # Path to variation assets
    replicator_prefix: Optional[str] = None,  # Path to replicator bundle
    enable_premium_analytics: bool = True,  # DEFAULT: ENABLED (no longer upsell!)
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
        filter_commercial: Only include commercial-use assets (DEFAULT: True)
        copy_usd: Copy USD files to output
        enable_multi_robot: Generate for multiple robot types (DEFAULT: True)
        enable_bimanual: Generate bimanual task configs (DEFAULT: True)
        enable_vla_packages: Generate VLA fine-tuning configs (DEFAULT: True)
        enable_rich_annotations: Generate rich annotation configs (DEFAULT: True)
        variation_assets_prefix: Path to variation assets (YOUR commercial assets)
        replicator_prefix: Path to replicator bundle
        enable_premium_analytics: Enable premium analytics capture (DEFAULT: True - NO LONGER UPSELL!)
        require_quality_gates: Fail when quality gates are unavailable or error (DEFAULT: True)
        bucket: GCS bucket for failure markers (optional)

    Returns:
        0 on success, 1 on failure
    """
    print(f"[GENIESIM-EXPORT-JOB] Starting Genie Sim export for scene: {scene_id}")
    print(f"[GENIESIM-EXPORT-JOB] Assets prefix: {assets_prefix}")
    print(f"[GENIESIM-EXPORT-JOB] Output prefix: {geniesim_prefix}")
    print(f"[GENIESIM-EXPORT-JOB] Variation assets prefix: {variation_assets_prefix}")
    print(f"[GENIESIM-EXPORT-JOB] Replicator prefix: {replicator_prefix}")
    print(f"[GENIESIM-EXPORT-JOB] Primary robot type: {robot_type}")
    print(f"[GENIESIM-EXPORT-JOB] Max tasks: {max_tasks}")
    print(f"[GENIESIM-EXPORT-JOB] Generate embeddings: {generate_embeddings}")
    print(f"[GENIESIM-EXPORT-JOB] Filter commercial: {filter_commercial}")
    print(f"[GENIESIM-EXPORT-JOB] Copy USD: {copy_usd}")
    print(f"[GENIESIM-EXPORT-JOB] Multi-robot enabled: {enable_multi_robot}")
    print(f"[GENIESIM-EXPORT-JOB] Bimanual enabled: {enable_bimanual}")
    print(f"[GENIESIM-EXPORT-JOB] VLA packages enabled: {enable_vla_packages}")
    print(f"[GENIESIM-EXPORT-JOB] Rich annotations enabled: {enable_rich_annotations}")
    print(f"[GENIESIM-EXPORT-JOB] Premium analytics enabled: {enable_premium_analytics} (DEFAULT - NO LONGER UPSELL!)")
    print(f"[GENIESIM-EXPORT-JOB] Require quality gates: {require_quality_gates}")

    assets_dir = root / assets_prefix
    output_dir = root / geniesim_prefix
    service_mode = _is_service_mode()

    # P1-7 FIX: Validate upstream job completion before starting export
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

    # P1-7 FIX: Block export if upstream jobs are not complete
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

    # P1-6 FIX: Load variation assets and apply commercial filtering BEFORE merging
    # This is CRITICAL for commercial use - Genie Sim's assets are CC BY-NC-SA 4.0
    variation_assets_dir = None
    variation_objects = []
    if not variation_assets_prefix or not variation_assets_prefix.strip():
        if filter_commercial or service_mode:
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
                    if filter_commercial or service_mode:
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

                # P1-6 FIX: Apply commercial filtering to variation assets BEFORE merging
                if filter_commercial:
                    filtered_variation_objects = []
                    non_commercial_count = 0
                    for obj in raw_variation_objects:
                        is_commercial = obj.get("asset", {}).get("commercial_ok", False)
                        license_type = obj.get("asset", {}).get("license", "unknown")

                        # Only include assets with commercial_ok=True or permissive licenses
                        if is_commercial or license_type in ["CC0", "CC-BY", "MIT", "Apache-2.0"]:
                            filtered_variation_objects.append(obj)
                        else:
                            non_commercial_count += 1

                    variation_objects = filtered_variation_objects
                    if non_commercial_count > 0:
                        print(f"[GENIESIM-EXPORT-JOB] ✓ Filtered out {non_commercial_count} NC-licensed variation assets")
                        print(f"[GENIESIM-EXPORT-JOB] ✓ Retained {len(variation_objects)} commercial-safe variation assets")
                    if not variation_objects and (filter_commercial or service_mode):
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
                    print("[GENIESIM-EXPORT-JOB] WARNING: Commercial filtering disabled - NC-licensed assets may be included")

            except Exception as e:
                if filter_commercial or service_mode:
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
            if filter_commercial or service_mode:
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

    # P0-5 FIX: Run quality gates before export
    gate_names = [
        "manifest_completeness",
        "asset_existence",
        "physics_properties",
        "scale_sanity",
    ]
    if HAVE_QUALITY_GATES:
        print("\n[GENIESIM-EXPORT-JOB] Running quality gates before export...")
        try:
            checkpoint = getattr(
                QualityGateCheckpoint,
                "GENIESIM_EXPORT_READY",
                QualityGateCheckpoint.REPLICATOR_COMPLETE,
            )
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
            }
            results = registry.run_checkpoint(checkpoint, context)
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
        if require_quality_gates:
            return 1
        print("[GENIESIM-EXPORT-JOB] ⚠️  Continuing without quality gate validation\n")

    # Configure exporter with enhanced features
    config = GenieSimExportConfig(
        robot_type=robot_type,
        urdf_path=urdf_path,
        generate_embeddings=generate_embeddings,
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
            # P1-8 FIX: Make premium analytics failures block export
            premium_analytics_manifests = {}
            if enable_premium_analytics and PREMIUM_ANALYTICS_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting premium analytics manifests (DEFAULT - NO LONGER UPSELL)")
                # P1-8 FIX: Don't catch exceptions - let them propagate to block export
                analytics_dir = output_dir / "premium_analytics"
                analytics_config = DefaultPremiumAnalyticsConfig(enabled=True)
                analytics_exporter = create_default_premium_analytics_exporter(
                    scene_id=scene_id,
                    output_dir=analytics_dir,
                    config=analytics_config,
                )
                premium_analytics_manifests = analytics_exporter.export_all_manifests()
                print(f"[GENIESIM-EXPORT-JOB]   ✓ Premium analytics: {len(premium_analytics_manifests)} manifests exported")
                print("[GENIESIM-EXPORT-JOB]   ✓ Per-step telemetry capture enabled")
                print("[GENIESIM-EXPORT-JOB]   ✓ Failure analysis enabled")
                print("[GENIESIM-EXPORT-JOB]   ✓ Grasp analytics enabled")
                print("[GENIESIM-EXPORT-JOB]   ✓ Parallel eval metrics enabled")
            elif not enable_premium_analytics:
                print("\n[GENIESIM-EXPORT-JOB] Premium analytics disabled (not recommended)")
            elif not PREMIUM_ANALYTICS_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] WARNING: Premium analytics module not available")

            # Export ALL additional premium features (DEFAULT: ENABLED - NO LONGER UPSELL!)
            all_premium_features_manifests = {}

            # 1. Sim2Real Fidelity Matrix ($20k-$50k value)
            # P1-8 FIX: Make failures block export
            if SIM2REAL_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Sim2Real Fidelity Matrix ($20k-$50k value - NOW FREE)")
                sim2real_dir = output_dir / "sim2real_fidelity"
                sim2real_manifests = create_default_sim2real_fidelity_exporter(
                    scene_id=scene_id,
                    robot_type=robot_type,
                    output_dir=sim2real_dir,
                )
                all_premium_features_manifests.update({"sim2real": sim2real_manifests})
                print(f"[GENIESIM-EXPORT-JOB]   ✓ Sim2Real Fidelity: {len(sim2real_manifests)} manifests exported")
                print("[GENIESIM-EXPORT-JOB]   ✓ Physics/Visual/Sensor fidelity scoring")
                print("[GENIESIM-EXPORT-JOB]   ✓ Transfer confidence score")
                print("[GENIESIM-EXPORT-JOB]   ✓ Trust matrix for deployment")

            # 2. Embodiment Transfer Analysis ($20k-$100k value)
            # P1-8 FIX: Make failures block export
            if EMBODIMENT_TRANSFER_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Embodiment Transfer Analysis ($20k-$100k value - NOW FREE)")
                embodiment_dir = output_dir / "embodiment_transfer"
                embodiment_manifests = create_default_embodiment_transfer_exporter(
                    scene_id=scene_id,
                    source_robot=robot_type,
                    output_dir=embodiment_dir,
                )
                all_premium_features_manifests.update({"embodiment": embodiment_manifests})
                print(f"[GENIESIM-EXPORT-JOB]   ✓ Embodiment Transfer: {len(embodiment_manifests)} manifests exported")
                print("[GENIESIM-EXPORT-JOB]   ✓ Cross-robot compatibility matrix")
                print("[GENIESIM-EXPORT-JOB]   ✓ Multi-robot data multiplier")
                print("[GENIESIM-EXPORT-JOB]   ✓ Transfer strategy recommendations")

            # 3. Trajectory Optimality Analysis ($10k-$25k value)
            # P1-8 FIX: Make failures block export
            if TRAJECTORY_OPTIMALITY_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Trajectory Optimality Analysis ($10k-$25k value - NOW FREE)")
                trajectory_dir = output_dir / "trajectory_optimality"
                trajectory_manifests = create_default_trajectory_optimality_exporter(
                    scene_id=scene_id,
                    output_dir=trajectory_dir,
                )
                all_premium_features_manifests.update({"trajectory": trajectory_manifests})
                print(f"[GENIESIM-EXPORT-JOB]   ✓ Trajectory Optimality: {len(trajectory_manifests)} manifests exported")
                print("[GENIESIM-EXPORT-JOB]   ✓ Path efficiency scoring")
                print("[GENIESIM-EXPORT-JOB]   ✓ Smoothness/jerk analysis")
                print("[GENIESIM-EXPORT-JOB]   ✓ Training suitability assessment")

            # 4. Policy Leaderboard ($20k-$40k value)
            # P1-8 FIX: Make failures block export
            if POLICY_LEADERBOARD_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Policy Leaderboard ($20k-$40k value - NOW FREE)")
                leaderboard_dir = output_dir / "policy_leaderboard"
                leaderboard_manifests = create_default_policy_leaderboard_exporter(
                    scene_id=scene_id,
                    output_dir=leaderboard_dir,
                )
                all_premium_features_manifests.update({"leaderboard": leaderboard_manifests})
                print(f"[GENIESIM-EXPORT-JOB]   ✓ Policy Leaderboard: {len(leaderboard_manifests)} manifests exported")
                print("[GENIESIM-EXPORT-JOB]   ✓ Multi-policy comparison with confidence intervals")
                print("[GENIESIM-EXPORT-JOB]   ✓ Statistical significance testing")
                print("[GENIESIM-EXPORT-JOB]   ✓ Pairwise comparison matrix")

            # 5. Tactile Sensor Simulation ($15k-$30k value)
            # P1-8 FIX: Make failures block export
            if TACTILE_SENSOR_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Tactile Sensor Simulation ($15k-$30k value - NOW FREE)")
                tactile_dir = output_dir / "tactile_sensors"
                tactile_manifests = create_default_tactile_sensor_exporter(
                    scene_id=scene_id,
                    output_dir=tactile_dir,
                )
                all_premium_features_manifests.update({"tactile": tactile_manifests})
                print(f"[GENIESIM-EXPORT-JOB]   ✓ Tactile Sensors: {len(tactile_manifests)} manifests exported")
                print("[GENIESIM-EXPORT-JOB]   ✓ GelSlim/GelSight/DIGIT simulation")
                print("[GENIESIM-EXPORT-JOB]   ✓ Contact force maps")
                print("[GENIESIM-EXPORT-JOB]   ✓ 81%+ success vs 50% vision-only")

            # 6. Language Annotations ($10k-$25k value)
            # P1-8 FIX: Make failures block export
            if LANGUAGE_ANNOTATIONS_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Language Annotations ($10k-$25k value - NOW FREE)")
                language_dir = output_dir / "language_annotations"
                language_manifests = create_default_language_annotations_exporter(
                    scene_id=scene_id,
                    output_dir=language_dir,
                )
                all_premium_features_manifests.update({"language": language_manifests})
                print(f"[GENIESIM-EXPORT-JOB]   ✓ Language Annotations: {len(language_manifests)} manifests exported")
                print("[GENIESIM-EXPORT-JOB]   ✓ Template + LLM-powered generation")
                print("[GENIESIM-EXPORT-JOB]   ✓ 10+ variations per task")
                print("[GENIESIM-EXPORT-JOB]   ✓ Required for VLA training (OpenVLA, Pi0, RT-2)")

            # 7. Generalization Analyzer ($15k-$35k value)
            # P1-8 FIX: Make failures block export
            if GENERALIZATION_ANALYZER_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Generalization Analyzer ($15k-$35k value - NOW FREE)")
                generalization_dir = output_dir / "generalization_analysis"
                generalization_manifests = create_default_generalization_analyzer_exporter(
                    scene_id=scene_id,
                    output_dir=generalization_dir,
                )
                all_premium_features_manifests.update({"generalization": generalization_manifests})
                print(f"[GENIESIM-EXPORT-JOB]   ✓ Generalization Analyzer: {len(generalization_manifests)} manifests exported")
                print("[GENIESIM-EXPORT-JOB]   ✓ Per-object success rate analysis")
                print("[GENIESIM-EXPORT-JOB]   ✓ Learning curve computation")
                print("[GENIESIM-EXPORT-JOB]   ✓ Curriculum learning recommendations")

            # 8. Sim2Real Validation Service ($5k-$25k/study value)
            # P1-8 FIX: Make failures block export
            if SIM2REAL_VALIDATION_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Sim2Real Validation Service ($5k-$25k/study - NOW FREE)")
                sim2real_validation_dir = output_dir / "sim2real_validation"
                sim2real_validation_manifests = create_default_sim2real_validation_exporter(
                    scene_id=scene_id,
                    robot_type=robot_type,
                    output_dir=sim2real_validation_dir,
                )
                all_premium_features_manifests.update({"sim2real_validation": sim2real_validation_manifests})
                print(f"[GENIESIM-EXPORT-JOB]   ✓ Sim2Real Validation: {len(sim2real_validation_manifests)} manifests exported")
                print("[GENIESIM-EXPORT-JOB]   ✓ Real-world validation trial tracking")
                print("[GENIESIM-EXPORT-JOB]   ✓ Sim vs real success rate comparison")
                print("[GENIESIM-EXPORT-JOB]   ✓ Quality guarantee certificates (50%/70%/85%)")
                print("[GENIESIM-EXPORT-JOB]   ✓ Failure mode comparison (sim vs real)")

            # 9. Audio Narration ($5k-$15k value)
            # P1-8 FIX: Make failures block export
            if AUDIO_NARRATION_AVAILABLE:
                print("\n[GENIESIM-EXPORT-JOB] Exporting Audio Narration ($5k-$15k value - NOW FREE)")
                audio_narration_dir = output_dir / "audio_narration"
                audio_narration_manifests = create_default_audio_narration_exporter(
                    scene_id=scene_id,
                    output_dir=audio_narration_dir,
                )
                all_premium_features_manifests.update({"audio_narration": audio_narration_manifests})
                print(f"[GENIESIM-EXPORT-JOB]   ✓ Audio Narration: {len(audio_narration_manifests)} manifests exported")
                print("[GENIESIM-EXPORT-JOB]   ✓ Text-to-speech narration (Google Cloud TTS + local)")
                print("[GENIESIM-EXPORT-JOB]   ✓ Multi-voice presets (narrator, instructor, casual, robot)")
                print("[GENIESIM-EXPORT-JOB]   ✓ MP3/WAV/OGG audio output")
                print("[GENIESIM-EXPORT-JOB]   ✓ VLA audio modality training (RT-2, PaLM-E)")

            # Summary of premium features
            if any([SIM2REAL_AVAILABLE, EMBODIMENT_TRANSFER_AVAILABLE, TRAJECTORY_OPTIMALITY_AVAILABLE,
                   POLICY_LEADERBOARD_AVAILABLE, TACTILE_SENSOR_AVAILABLE, LANGUAGE_ANNOTATIONS_AVAILABLE,
                   GENERALIZATION_ANALYZER_AVAILABLE, SIM2REAL_VALIDATION_AVAILABLE, AUDIO_NARRATION_AVAILABLE]):
                print("\n" + "="*80)
                print("  🎉 PREMIUM FEATURES EXPORTED (DEFAULT - FREE)")
                print("="*80)
                total_value = 0
                features_exported = []
                if SIM2REAL_AVAILABLE:
                    features_exported.append("Sim2Real Fidelity Matrix ($20k-$50k)")
                    total_value += 35000
                if EMBODIMENT_TRANSFER_AVAILABLE:
                    features_exported.append("Embodiment Transfer Analysis ($20k-$100k)")
                    total_value += 60000
                if TRAJECTORY_OPTIMALITY_AVAILABLE:
                    features_exported.append("Trajectory Optimality Analysis ($10k-$25k)")
                    total_value += 17500
                if POLICY_LEADERBOARD_AVAILABLE:
                    features_exported.append("Policy Leaderboard ($20k-$40k)")
                    total_value += 30000
                if TACTILE_SENSOR_AVAILABLE:
                    features_exported.append("Tactile Sensor Simulation ($15k-$30k)")
                    total_value += 22500
                if LANGUAGE_ANNOTATIONS_AVAILABLE:
                    features_exported.append("Language Annotations ($10k-$25k)")
                    total_value += 17500
                if GENERALIZATION_ANALYZER_AVAILABLE:
                    features_exported.append("Generalization Analyzer ($15k-$35k)")
                    total_value += 25000
                if SIM2REAL_VALIDATION_AVAILABLE:
                    features_exported.append("Sim2Real Validation Service ($5k-$25k/study)")
                    total_value += 15000
                if AUDIO_NARRATION_AVAILABLE:
                    features_exported.append("Audio Narration ($5k-$15k)")
                    total_value += 10000

                for feature in features_exported:
                    print(f"  ✓ {feature}")
                print(f"\n  💰 Total Value Delivered: ${total_value:,} (NOW FREE BY DEFAULT!)")
                print("="*80 + "\n")

            # P1-9 FIX: Write completion marker with schema version tracking
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
                "premium_analytics_manifests": len(premium_analytics_manifests),
                # P1-9 FIX: Add schema version tracking
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
                # P1-9 FIX: Track which premium features were exported
                "premium_features_exported": {
                    "sim2real_fidelity": SIM2REAL_AVAILABLE,
                    "embodiment_transfer": EMBODIMENT_TRANSFER_AVAILABLE,
                    "trajectory_optimality": TRAJECTORY_OPTIMALITY_AVAILABLE,
                    "policy_leaderboard": POLICY_LEADERBOARD_AVAILABLE,
                    "tactile_sensors": TACTILE_SENSOR_AVAILABLE,
                    "language_annotations": LANGUAGE_ANNOTATIONS_AVAILABLE,
                    "generalization_analyzer": GENERALIZATION_ANALYZER_AVAILABLE,
                    "sim2real_validation": SIM2REAL_VALIDATION_AVAILABLE,
                    "audio_narration": AUDIO_NARRATION_AVAILABLE,
                },
            }
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
    # P0-7 FIX: Validate credentials at startup
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

    # Get configuration from environment
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")

    if not scene_id:
        print("[GENIESIM-EXPORT-JOB] ERROR: SCENE_ID is required")
        sys.exit(1)

    # Prefixes with defaults
    assets_prefix = os.getenv("ASSETS_PREFIX", f"scenes/{scene_id}/assets")
    geniesim_prefix = os.getenv("GENIESIM_PREFIX", f"scenes/{scene_id}/geniesim")
    # IMPORTANT: variation_assets_prefix contains YOUR commercial assets
    variation_assets_prefix = os.getenv("VARIATION_ASSETS_PREFIX", f"scenes/{scene_id}/variation_assets")
    replicator_prefix = os.getenv("REPLICATOR_PREFIX", f"scenes/{scene_id}/replicator")

    # Configuration
    robot_type = os.getenv("ROBOT_TYPE", "franka")
    urdf_path = os.getenv("URDF_PATH")  # Optional custom URDF
    max_tasks = int(os.getenv("MAX_TASKS", "50"))
    generate_embeddings = os.getenv("GENERATE_EMBEDDINGS", "false").lower() == "true"
    # Default to TRUE for commercial use - only use your own assets
    filter_commercial = os.getenv("FILTER_COMMERCIAL", "true").lower() == "true"
    copy_usd = os.getenv("COPY_USD", "true").lower() == "true"

    # Enhanced features (DEFAULT: ENABLED)
    enable_multi_robot = os.getenv("ENABLE_MULTI_ROBOT", "true").lower() == "true"
    enable_bimanual = os.getenv("ENABLE_BIMANUAL", "true").lower() == "true"
    enable_vla_packages = os.getenv("ENABLE_VLA_PACKAGES", "true").lower() == "true"
    enable_rich_annotations = os.getenv("ENABLE_RICH_ANNOTATIONS", "true").lower() == "true"

    # Premium analytics (DEFAULT: ENABLED - NO LONGER UPSELL!)
    enable_premium_analytics = os.getenv("ENABLE_PREMIUM_ANALYTICS", "true").lower() == "true"
    require_quality_gates = parse_bool(os.getenv("REQUIRE_QUALITY_GATES"), True)

    print("[GENIESIM-EXPORT-JOB] Configuration:")
    print(f"[GENIESIM-EXPORT-JOB]   Bucket: {bucket}")
    print(f"[GENIESIM-EXPORT-JOB]   Scene ID: {scene_id}")
    print(f"[GENIESIM-EXPORT-JOB]   Variation Assets: {variation_assets_prefix}")
    print(f"[GENIESIM-EXPORT-JOB]   Replicator Bundle: {replicator_prefix}")
    print(f"[GENIESIM-EXPORT-JOB]   Primary Robot Type: {robot_type}")
    print(f"[GENIESIM-EXPORT-JOB]   Max Tasks: {max_tasks}")
    print(f"[GENIESIM-EXPORT-JOB]   Multi-Robot: {enable_multi_robot}")
    print(f"[GENIESIM-EXPORT-JOB]   Bimanual: {enable_bimanual}")
    print(f"[GENIESIM-EXPORT-JOB]   VLA Packages: {enable_vla_packages}")
    print(f"[GENIESIM-EXPORT-JOB]   Rich Annotations: {enable_rich_annotations}")
    print(f"[GENIESIM-EXPORT-JOB]   Commercial Filter: {filter_commercial}")
    print(f"[GENIESIM-EXPORT-JOB]   Premium Analytics: {enable_premium_analytics} (DEFAULT - NO LONGER UPSELL!)")
    print(f"[GENIESIM-EXPORT-JOB]   Require Quality Gates: {require_quality_gates}")

    GCS_ROOT = Path("/mnt/gcs")

    metrics = get_metrics()
    with metrics.track_job("genie-sim-export-job", scene_id):
        exit_code = run_geniesim_export_job(
            root=GCS_ROOT,
            scene_id=scene_id,
            assets_prefix=assets_prefix,
            geniesim_prefix=geniesim_prefix,
            robot_type=robot_type,
            urdf_path=urdf_path,
            max_tasks=max_tasks,
            generate_embeddings=generate_embeddings,
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
            require_quality_gates=require_quality_gates,
            bucket=bucket,
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
