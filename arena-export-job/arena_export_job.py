"""
Arena Export Job - Pipeline job for Isaac Lab-Arena integration.

This job converts Blueprint Pipeline scenes to Isaac Lab-Arena format,
enabling standardized policy evaluation and benchmark creation.

Usage:
    # As Cloud Run Job
    BUCKET=my-bucket SCENE_ID=kitchen_123 ASSETS_PREFIX=scenes/kitchen_123/assets \
        python arena_export_job.py

    # Local execution
    python arena_export_job.py --scene-dir /path/to/scene --output-dir /path/to/output

Environment Variables:
    BUCKET: GCS bucket name
    SCENE_ID: Scene identifier
    ASSETS_PREFIX: Path prefix for assets in bucket
    GEMINI_API_KEY: (Optional) For LLM-based affordance detection
    HF_TOKEN: (Optional) For LeRobot Hub registration
    ENABLE_HUB_REGISTRATION: (Optional) Enable auto-registration with Hub
    ENABLE_PREMIUM_ANALYTICS: (Optional) Enable premium analytics capture - default: true (NO LONGER UPSELL!)

Premium Analytics (DEFAULT: ENABLED - NO LONGER UPSELL!):
    - Per-step telemetry (rewards, collisions, grasps, forces, torques)
    - Failure analysis (timeout/collision breakdown, phase-level tracking)
    - Grasp analytics (event timeline, force profiles, contact tracking)
    - Parallel eval metrics (GPU utilization, cross-env variance, throughput)

    Previously $115k-$260k upsell - NOW INCLUDED BY DEFAULT!
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arena_integration import (
    ArenaExportConfig,
    ArenaSceneExporter,
    export_scene_to_arena,
    AffordanceDetector,
    LeRobotHubRegistrar,
    HubConfig,
)
from tools.scene_manifest.loader import load_manifest_or_scene_assets

# Import default premium analytics (DEFAULT: ENABLED)
try:
    from .default_premium_analytics import (
        create_default_premium_analytics_exporter,
        DefaultPremiumAnalyticsConfig,
    )
    PREMIUM_ANALYTICS_AVAILABLE = True
except ImportError:
    PREMIUM_ANALYTICS_AVAILABLE = False
    print("[ARENA] WARNING: Premium analytics module not available")

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

# Default paths
GCS_ROOT = Path("/mnt/gcs")


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    """Save JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def enhance_manifest_with_affordances(
    manifest: dict[str, Any],
    use_llm: bool = True
) -> dict[str, Any]:
    """
    Enhance scene manifest with affordance annotations.

    This modifies the manifest in-place, adding affordances and
    affordance_params to each object's semantics.
    """
    detector = AffordanceDetector(use_llm=use_llm)

    for obj in manifest.get("objects", []):
        obj_id = obj.get("id", "unknown")
        print(f"[ARENA] Detecting affordances for object: {obj_id}")

        # Detect affordances
        affordances = detector.detect(obj)

        # Convert to manifest format
        manifest_format = detector.to_manifest_format(affordances)

        # Update object semantics
        if "semantics" not in obj:
            obj["semantics"] = {}

        obj["semantics"]["affordances"] = manifest_format["affordances"]
        obj["semantics"]["affordance_params"] = manifest_format["affordance_params"]

        print(f"[ARENA]   -> Found affordances: {manifest_format['affordances']}")

    return manifest


def run_arena_export(
    scene_dir: Path,
    output_dir: Optional[Path] = None,
    use_llm: bool = True,
    enable_hub_registration: bool = False,
    hub_namespace: str = "blueprint-robotics",
    enable_premium_analytics: bool = True,  # DEFAULT: ENABLED (NO LONGER UPSELL!)
) -> dict[str, Any]:
    """
    Run Arena export for a scene.

    Args:
        scene_dir: Path to scene directory (contains scene_manifest.json, usd/, etc.)
        output_dir: Output directory (defaults to scene_dir/arena)
        use_llm: Use Gemini for affordance detection
        enable_hub_registration: Register with LeRobot Hub
        hub_namespace: Hugging Face namespace for Hub
        enable_premium_analytics: Enable premium analytics capture (DEFAULT: True - NO LONGER UPSELL!)

    Returns:
        Export result dictionary
    """
    result = {
        "success": False,
        "scene_id": None,
        "output_dir": None,
        "files_generated": [],
        "affordance_count": 0,
        "task_count": 0,
        "hub_registration": None,
        "premium_analytics_enabled": False,
        "premium_analytics_manifests": 0,
        "errors": [],
    }

    # Load manifest
    assets_dir = scene_dir / "assets"
    manifest = load_manifest_or_scene_assets(assets_dir)
    if manifest is None:
        manifest = load_manifest_or_scene_assets(scene_dir)

    if manifest is None:
        result["errors"].append(f"Could not load manifest from {scene_dir}")
        return result

    scene_id = manifest.get("scene_id", scene_dir.name)
    result["scene_id"] = scene_id
    print(f"[ARENA] Processing scene: {scene_id}")

    # Determine paths
    output_dir = output_dir or scene_dir
    scene_path = "usd/scene.usda"

    # Check for USD file
    usd_file = scene_dir / "usd" / "scene.usda"
    if not usd_file.exists():
        # Try alternative paths
        for alt_path in ["scene.usda", "scene.usd", "usd/scene.usd"]:
            alt_file = scene_dir / alt_path
            if alt_file.exists():
                scene_path = alt_path
                break
        else:
            result["errors"].append(f"No USD scene file found in {scene_dir}")
            # Continue anyway - Arena export can still work for evaluation planning

    # Get environment type
    scene_meta = manifest.get("scene", {})
    env_type = scene_meta.get("environment_type", "generic")

    # Step 1: Enhance manifest with affordances
    print("[ARENA] Step 1: Detecting affordances...")
    manifest = enhance_manifest_with_affordances(manifest, use_llm=use_llm)

    # Save enhanced manifest
    enhanced_manifest_path = output_dir / "assets" / "scene_manifest_enhanced.json"
    enhanced_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(manifest, enhanced_manifest_path)
    result["files_generated"].append(str(enhanced_manifest_path))
    print(f"[ARENA]   -> Saved enhanced manifest: {enhanced_manifest_path}")

    # Step 2: Export to Arena format
    print("[ARENA] Step 2: Exporting to Arena format...")
    config = ArenaExportConfig(
        scene_id=scene_id,
        scene_path=scene_path,
        output_dir=output_dir,
        environment_type=env_type,
        use_llm_affordances=use_llm,
        generate_hub_config=enable_hub_registration,
        hub_namespace=hub_namespace,
    )

    exporter = ArenaSceneExporter(config)
    export_result = exporter.export(manifest)

    result["output_dir"] = str(export_result.output_dir)
    result["files_generated"].extend(export_result.generated_files)
    result["affordance_count"] = export_result.affordance_count
    result["task_count"] = export_result.task_count
    result["errors"].extend(export_result.errors)

    if export_result.success:
        print(f"[ARENA]   -> Export successful!")
        print(f"[ARENA]   -> Generated {export_result.task_count} tasks")
        print(f"[ARENA]   -> Detected {export_result.affordance_count} affordances")
    else:
        print(f"[ARENA]   -> Export had errors: {export_result.errors}")

    # Step 3: Register with Hub (if enabled)
    if enable_hub_registration and os.getenv("HF_TOKEN"):
        print("[ARENA] Step 3: Registering with LeRobot Hub...")
        hub_config = HubConfig(
            scene_id=scene_id,
            namespace=hub_namespace,
            environment_type=env_type,
        )

        registrar = LeRobotHubRegistrar(hub_config)
        hub_result = registrar.register(export_result.output_dir)

        result["hub_registration"] = {
            "success": hub_result.success,
            "repo_url": hub_result.repo_url,
            "files_uploaded": hub_result.files_uploaded,
            "errors": hub_result.errors,
        }

        if hub_result.success:
            print(f"[ARENA]   -> Registered at: {hub_result.repo_url}")
        else:
            print(f"[ARENA]   -> Hub registration failed: {hub_result.errors}")
    else:
        print("[ARENA] Step 3: Hub registration skipped (disabled or no HF_TOKEN)")

    # Step 4: Export premium analytics manifests (DEFAULT: ENABLED)
    premium_analytics_manifests = {}
    if enable_premium_analytics and PREMIUM_ANALYTICS_AVAILABLE:
        print("\n[ARENA] Step 4: Exporting premium analytics manifests (DEFAULT - NO LONGER UPSELL)")
        try:
            analytics_dir = output_dir / "arena" / "premium_analytics"
            analytics_config = DefaultPremiumAnalyticsConfig(enabled=True)
            analytics_exporter = create_default_premium_analytics_exporter(
                scene_id=scene_id,
                output_dir=analytics_dir,
                config=analytics_config,
            )
            premium_analytics_manifests = analytics_exporter.export_all_manifests()
            result["premium_analytics_enabled"] = True
            result["premium_analytics_manifests"] = len(premium_analytics_manifests)
            result["files_generated"].extend([str(p) for p in premium_analytics_manifests.values()])
            print(f"[ARENA]   ✓ Premium analytics: {len(premium_analytics_manifests)} manifests exported")
            print("[ARENA]   ✓ Per-step telemetry capture enabled")
            print("[ARENA]   ✓ Failure analysis enabled")
            print("[ARENA]   ✓ Grasp analytics enabled")
            print("[ARENA]   ✓ Parallel eval metrics enabled")
        except Exception as e:
            print(f"[ARENA] WARNING: Premium analytics export failed: {e}")
            import traceback
            traceback.print_exc()
    elif not enable_premium_analytics:
        print("\n[ARENA] Step 4: Premium analytics disabled (not recommended)")
    elif not PREMIUM_ANALYTICS_AVAILABLE:
        print("\n[ARENA] WARNING: Premium analytics module not available")

    # Export ALL additional premium features (DEFAULT: ENABLED - NO LONGER UPSELL!)
    # Same pattern as genie-sim-export-job
    robot_type = os.getenv("ROBOT_TYPE", "franka")
    if SIM2REAL_AVAILABLE:
        try:
            sim2real_dir = output_dir / "arena" / "sim2real_fidelity"
            sim2real_manifests = create_default_sim2real_fidelity_exporter(scene_id=scene_id, robot_type=robot_type, output_dir=sim2real_dir)
            result["files_generated"].extend([str(p) for p in sim2real_manifests.values()])
            print(f"[ARENA]   ✓ Sim2Real Fidelity: {len(sim2real_manifests)} manifests")
        except Exception as e:
            print(f"[ARENA] WARNING: Sim2Real export failed: {e}")

    if EMBODIMENT_TRANSFER_AVAILABLE:
        try:
            embodiment_dir = output_dir / "arena" / "embodiment_transfer"
            embodiment_manifests = create_default_embodiment_transfer_exporter(scene_id=scene_id, source_robot=robot_type, output_dir=embodiment_dir)
            result["files_generated"].extend([str(p) for p in embodiment_manifests.values()])
            print(f"[ARENA]   ✓ Embodiment Transfer: {len(embodiment_manifests)} manifests")
        except Exception as e:
            print(f"[ARENA] WARNING: Embodiment transfer failed: {e}")

    if TRAJECTORY_OPTIMALITY_AVAILABLE:
        try:
            trajectory_dir = output_dir / "arena" / "trajectory_optimality"
            trajectory_manifests = create_default_trajectory_optimality_exporter(scene_id=scene_id, output_dir=trajectory_dir)
            result["files_generated"].extend([str(p) for p in trajectory_manifests.values()])
            print(f"[ARENA]   ✓ Trajectory Optimality: {len(trajectory_manifests)} manifests")
        except Exception as e:
            print(f"[ARENA] WARNING: Trajectory optimality failed: {e}")

    if POLICY_LEADERBOARD_AVAILABLE:
        try:
            leaderboard_dir = output_dir / "arena" / "policy_leaderboard"
            leaderboard_manifests = create_default_policy_leaderboard_exporter(scene_id=scene_id, output_dir=leaderboard_dir)
            result["files_generated"].extend([str(p) for p in leaderboard_manifests.values()])
            print(f"[ARENA]   ✓ Policy Leaderboard: {len(leaderboard_manifests)} manifests")
        except Exception as e:
            print(f"[ARENA] WARNING: Policy leaderboard failed: {e}")

    if TACTILE_SENSOR_AVAILABLE:
        try:
            tactile_dir = output_dir / "arena" / "tactile_sensors"
            tactile_manifests = create_default_tactile_sensor_exporter(scene_id=scene_id, output_dir=tactile_dir)
            result["files_generated"].extend([str(p) for p in tactile_manifests.values()])
            print(f"[ARENA]   ✓ Tactile Sensors: {len(tactile_manifests)} manifests")
        except Exception as e:
            print(f"[ARENA] WARNING: Tactile sensor failed: {e}")

    if LANGUAGE_ANNOTATIONS_AVAILABLE:
        try:
            language_dir = output_dir / "arena" / "language_annotations"
            language_manifests = create_default_language_annotations_exporter(scene_id=scene_id, output_dir=language_dir)
            result["files_generated"].extend([str(p) for p in language_manifests.values()])
            print(f"[ARENA]   ✓ Language Annotations: {len(language_manifests)} manifests")
        except Exception as e:
            print(f"[ARENA] WARNING: Language annotations failed: {e}")

    if GENERALIZATION_ANALYZER_AVAILABLE:
        try:
            generalization_dir = output_dir / "arena" / "generalization_analysis"
            generalization_manifests = create_default_generalization_analyzer_exporter(scene_id=scene_id, output_dir=generalization_dir)
            result["files_generated"].extend([str(p) for p in generalization_manifests.values()])
            print(f"[ARENA]   ✓ Generalization Analyzer: {len(generalization_manifests)} manifests")
        except Exception as e:
            print(f"[ARENA] WARNING: Generalization analyzer failed: {e}")

    # Write completion marker
    marker_path = output_dir / "arena" / ".arena_export_complete"
    marker_content = {
        "status": "complete",
        "scene_id": scene_id,
        "task_count": result["task_count"],
        "affordance_count": result["affordance_count"],
        "files_generated": len(result["files_generated"]),
        "premium_analytics_enabled": result["premium_analytics_enabled"],
        "premium_analytics_manifests": result["premium_analytics_manifests"],
    }
    save_json(marker_content, marker_path)
    result["files_generated"].append(str(marker_path))

    result["success"] = len(result["errors"]) == 0 or export_result.success
    return result


def run_from_env(root: Path = GCS_ROOT) -> int:
    """
    Run Arena export from environment variables (Cloud Run Job mode).

    Environment Variables:
        BUCKET: GCS bucket name
        SCENE_ID: Scene identifier
        ASSETS_PREFIX: Path prefix for assets
        USE_LLM_AFFORDANCES: Enable LLM affordance detection (default: true)
        ENABLE_HUB_REGISTRATION: Enable Hub registration (default: false)
        HUB_NAMESPACE: Hugging Face namespace (default: blueprint-robotics)
    """
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    assets_prefix = os.getenv("ASSETS_PREFIX", "")

    use_llm = os.getenv("USE_LLM_AFFORDANCES", "true").lower() in ("true", "1", "yes")
    enable_hub = os.getenv("ENABLE_HUB_REGISTRATION", "false").lower() in ("true", "1", "yes")
    hub_namespace = os.getenv("HUB_NAMESPACE", "blueprint-robotics")
    enable_premium_analytics = os.getenv("ENABLE_PREMIUM_ANALYTICS", "true").lower() in ("true", "1", "yes")

    if not assets_prefix:
        print("[ARENA] ERROR: ASSETS_PREFIX is required", file=sys.stderr)
        return 1

    scene_dir = root / assets_prefix.rstrip("/assets")
    if not scene_dir.exists():
        # Try with assets prefix directly
        scene_dir = root / assets_prefix
        if not scene_dir.exists():
            print(f"[ARENA] ERROR: Scene directory not found: {scene_dir}", file=sys.stderr)
            return 1

    print(f"[ARENA] Running Arena export job")
    print(f"[ARENA]   Bucket: {bucket}")
    print(f"[ARENA]   Scene ID: {scene_id}")
    print(f"[ARENA]   Scene dir: {scene_dir}")
    print(f"[ARENA]   Use LLM: {use_llm}")
    print(f"[ARENA]   Hub registration: {enable_hub}")
    print(f"[ARENA]   Premium analytics: {enable_premium_analytics} (DEFAULT - NO LONGER UPSELL!)")

    result = run_arena_export(
        scene_dir=scene_dir,
        use_llm=use_llm,
        enable_hub_registration=enable_hub,
        hub_namespace=hub_namespace,
        enable_premium_analytics=enable_premium_analytics,
    )

    if result["success"]:
        print(f"[ARENA] Arena export completed successfully")
        print(f"[ARENA]   Tasks generated: {result['task_count']}")
        print(f"[ARENA]   Affordances detected: {result['affordance_count']}")
        return 0
    else:
        print(f"[ARENA] Arena export failed: {result['errors']}", file=sys.stderr)
        return 1


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Export Blueprint scene to Isaac Lab-Arena format"
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        help="Path to scene directory (contains scene_manifest.json, usd/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (defaults to scene_dir)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM-based affordance detection",
    )
    parser.add_argument(
        "--enable-hub",
        action="store_true",
        help="Enable LeRobot Hub registration",
    )
    parser.add_argument(
        "--hub-namespace",
        type=str,
        default="blueprint-robotics",
        help="Hugging Face namespace for Hub registration",
    )
    parser.add_argument(
        "--disable-premium-analytics",
        action="store_true",
        help="Disable premium analytics capture (not recommended - previously $115k-$260k upsell)",
    )
    parser.add_argument(
        "--env-mode",
        action="store_true",
        help="Run in environment variable mode (Cloud Run Job)",
    )

    args = parser.parse_args()

    if args.env_mode or args.scene_dir is None:
        # Environment variable mode
        sys.exit(run_from_env())

    # CLI mode
    if not args.scene_dir.exists():
        print(f"ERROR: Scene directory not found: {args.scene_dir}", file=sys.stderr)
        sys.exit(1)

    result = run_arena_export(
        scene_dir=args.scene_dir,
        output_dir=args.output_dir,
        use_llm=not args.no_llm,
        enable_hub_registration=args.enable_hub,
        hub_namespace=args.hub_namespace,
        enable_premium_analytics=not args.disable_premium_analytics,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("ARENA EXPORT SUMMARY")
    print("=" * 60)
    print(f"Scene ID: {result['scene_id']}")
    print(f"Success: {result['success']}")
    print(f"Tasks Generated: {result['task_count']}")
    print(f"Affordances Detected: {result['affordance_count']}")
    print(f"Files Generated: {len(result['files_generated'])}")

    if result["hub_registration"]:
        hub = result["hub_registration"]
        print(f"\nHub Registration:")
        print(f"  Success: {hub['success']}")
        print(f"  URL: {hub['repo_url']}")

    if result["errors"]:
        print(f"\nErrors:")
        for error in result["errors"]:
            print(f"  - {error}")

    print("=" * 60)

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
