#!/usr/bin/env python3
"""
Bundle Configuration System.

Defines Standard, Pro, and Enterprise bundle tiers with automated
feature activation based on bundle selection.

Bundle Pricing:
- Standard: $5,499 (current)
- Pro: $12,499
- Enterprise: $25,000
- Foundation: $500k-$2M+/year (custom)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class BundleTier(str, Enum):
    """Available bundle tiers."""
    STANDARD = "standard"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    FOUNDATION = "foundation"


@dataclass
class BundleFeatures:
    """Features included in a bundle."""
    # Core features
    simready_usd: bool = True
    variations: int = 250
    episodes: int = 2500
    episodes_per_variation: int = 10

    # Data formats
    lerobot_format: bool = True
    lerobot_version: str = "v2.0"
    streaming_enabled: bool = False

    # Sensor data
    rgb_cameras: int = 1
    depth_enabled: bool = True
    segmentation_enabled: bool = False

    # Upsell features
    language_annotations: bool = False
    num_language_variations: int = 0
    vla_finetuning_package: bool = False
    vla_models: List[str] = field(default_factory=list)

    # Sim2Real
    sim2real_validation: bool = False
    sim2real_tier: str = "none"  # none, basic, comprehensive, certification
    sim2real_trials: int = 0

    # Advanced capabilities
    contact_rich_tasks: bool = False
    tactile_simulation: bool = False
    tactile_sensor_type: str = "none"

    multi_robot: bool = False
    max_robots: int = 1

    deformable_objects: bool = False
    bimanual: bool = False
    custom_robot_support: bool = False

    # Quality & validation
    min_quality_score: float = 0.7
    isaac_lab_runtime_validation: bool = False

    # DWM
    dwm_conditioning: bool = False

    # Audio and Subtitle features
    audio_narration: bool = False
    audio_voice_preset: str = "narrator"  # narrator, instructor, robot
    subtitle_generation: bool = False
    subtitle_style: str = "descriptive"  # minimal, descriptive, technical, instructional
    subtitle_formats: List[str] = field(default_factory=lambda: ["srt", "vtt", "json"])

    # Support
    priority_support: bool = False
    dedicated_support: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core": {
                "simready_usd": self.simready_usd,
                "variations": self.variations,
                "episodes": self.episodes,
                "episodes_per_variation": self.episodes_per_variation,
            },
            "data_format": {
                "lerobot_format": self.lerobot_format,
                "lerobot_version": self.lerobot_version,
                "streaming_enabled": self.streaming_enabled,
            },
            "sensors": {
                "rgb_cameras": self.rgb_cameras,
                "depth_enabled": self.depth_enabled,
                "segmentation_enabled": self.segmentation_enabled,
            },
            "language": {
                "annotations_enabled": self.language_annotations,
                "num_variations": self.num_language_variations,
            },
            "vla": {
                "finetuning_package": self.vla_finetuning_package,
                "models": self.vla_models,
            },
            "sim2real": {
                "validation_enabled": self.sim2real_validation,
                "tier": self.sim2real_tier,
                "trials": self.sim2real_trials,
            },
            "advanced": {
                "contact_rich_tasks": self.contact_rich_tasks,
                "tactile_simulation": self.tactile_simulation,
                "tactile_sensor_type": self.tactile_sensor_type,
                "multi_robot": self.multi_robot,
                "max_robots": self.max_robots,
                "deformable_objects": self.deformable_objects,
                "bimanual": self.bimanual,
                "custom_robot_support": self.custom_robot_support,
            },
            "quality": {
                "min_quality_score": self.min_quality_score,
                "isaac_lab_runtime_validation": self.isaac_lab_runtime_validation,
            },
            "dwm": {
                "conditioning_enabled": self.dwm_conditioning,
            },
            "audio": {
                "narration_enabled": self.audio_narration,
                "voice_preset": self.audio_voice_preset,
            },
            "subtitles": {
                "enabled": self.subtitle_generation,
                "style": self.subtitle_style,
                "formats": self.subtitle_formats,
            },
            "support": {
                "priority": self.priority_support,
                "dedicated": self.dedicated_support,
            },
        }


# Bundle tier definitions
BUNDLE_CONFIGS: Dict[BundleTier, BundleFeatures] = {
    BundleTier.STANDARD: BundleFeatures(
        # Core
        simready_usd=True,
        variations=250,
        episodes=2500,
        episodes_per_variation=10,
        # Data
        lerobot_format=True,
        lerobot_version="v2.0",
        # Sensors
        rgb_cameras=1,
        depth_enabled=True,
        segmentation_enabled=False,
        # Quality
        min_quality_score=0.7,
    ),
    BundleTier.PRO: BundleFeatures(
        # Core - enhanced
        simready_usd=True,
        variations=500,
        episodes=5000,
        episodes_per_variation=10,
        # Data
        lerobot_format=True,
        lerobot_version="v2.0",
        streaming_enabled=False,
        # Sensors - enhanced
        rgb_cameras=2,
        depth_enabled=True,
        segmentation_enabled=True,
        # Language - NEW
        language_annotations=True,
        num_language_variations=10,
        # VLA - NEW
        vla_finetuning_package=True,
        vla_models=["openvla", "pi0"],
        # Quality - enhanced
        min_quality_score=0.8,
        isaac_lab_runtime_validation=True,
        # Audio/Subtitles - NEW
        audio_narration=True,
        audio_voice_preset="narrator",
        subtitle_generation=True,
        subtitle_style="descriptive",
        # Support
        priority_support=True,
    ),
    BundleTier.ENTERPRISE: BundleFeatures(
        # Core - maximum
        simready_usd=True,
        variations=1000,
        episodes=10000,
        episodes_per_variation=10,
        # Data
        lerobot_format=True,
        lerobot_version="v2.0",
        streaming_enabled=True,
        # Sensors - full
        rgb_cameras=4,
        depth_enabled=True,
        segmentation_enabled=True,
        # Language - full
        language_annotations=True,
        num_language_variations=15,
        # VLA - full
        vla_finetuning_package=True,
        vla_models=["openvla", "pi0", "smolvla", "groot_n1"],
        # Sim2Real - NEW
        sim2real_validation=True,
        sim2real_tier="basic",
        sim2real_trials=20,
        # Advanced - NEW
        contact_rich_tasks=True,
        tactile_simulation=True,
        tactile_sensor_type="gelslim",
        # Quality - highest
        min_quality_score=0.85,
        isaac_lab_runtime_validation=True,
        # DWM
        dwm_conditioning=True,
        # Audio/Subtitles - Enhanced
        audio_narration=True,
        audio_voice_preset="instructor",
        subtitle_generation=True,
        subtitle_style="instructional",
        # Support
        priority_support=True,
        dedicated_support=True,
    ),
    BundleTier.FOUNDATION: BundleFeatures(
        # Everything + custom scale
        simready_usd=True,
        variations=2000,
        episodes=50000,
        episodes_per_variation=25,
        # Data
        lerobot_format=True,
        lerobot_version="v3.0",
        streaming_enabled=True,
        # Sensors - full
        rgb_cameras=4,
        depth_enabled=True,
        segmentation_enabled=True,
        # Language - full
        language_annotations=True,
        num_language_variations=20,
        # VLA - all
        vla_finetuning_package=True,
        vla_models=["openvla", "pi0", "smolvla", "groot_n1"],
        # Sim2Real - certification
        sim2real_validation=True,
        sim2real_tier="certification",
        sim2real_trials=100,
        # Advanced - everything
        contact_rich_tasks=True,
        tactile_simulation=True,
        tactile_sensor_type="gelslim",
        multi_robot=True,
        max_robots=4,
        deformable_objects=True,
        bimanual=True,
        custom_robot_support=True,
        # Quality - highest
        min_quality_score=0.9,
        isaac_lab_runtime_validation=True,
        # DWM
        dwm_conditioning=True,
        # Audio/Subtitles - Full
        audio_narration=True,
        audio_voice_preset="instructor",
        subtitle_generation=True,
        subtitle_style="instructional",
        subtitle_formats=["srt", "vtt", "json"],
        # Support
        priority_support=True,
        dedicated_support=True,
    ),
}


@dataclass
class BundlePricing:
    """Pricing for a bundle."""
    base_price: int
    per_scene_price: int
    volume_discount_10: float = 0.0
    volume_discount_25: float = 0.0
    volume_discount_50: float = 0.0


BUNDLE_PRICING: Dict[BundleTier, BundlePricing] = {
    BundleTier.STANDARD: BundlePricing(
        base_price=5499,
        per_scene_price=5499,
        volume_discount_10=0.11,
        volume_discount_25=0.16,
        volume_discount_50=0.27,
    ),
    BundleTier.PRO: BundlePricing(
        base_price=12499,
        per_scene_price=12499,
        volume_discount_10=0.10,
        volume_discount_25=0.15,
        volume_discount_50=0.25,
    ),
    BundleTier.ENTERPRISE: BundlePricing(
        base_price=25000,
        per_scene_price=25000,
        volume_discount_10=0.10,
        volume_discount_25=0.15,
        volume_discount_50=0.20,
    ),
    BundleTier.FOUNDATION: BundlePricing(
        base_price=500000,  # Starting price for annual license
        per_scene_price=0,  # Included in license
    ),
}


class BundleConfigManager:
    """
    Manages bundle configuration for pipeline runs.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[BUNDLE] {msg}")

    def get_bundle_config(
        self,
        tier: BundleTier,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> BundleFeatures:
        """Get bundle configuration with optional overrides."""
        config = BUNDLE_CONFIGS[tier]

        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        return config

    def get_pricing(
        self,
        tier: BundleTier,
        num_scenes: int = 1,
    ) -> Dict[str, Any]:
        """Calculate pricing for a bundle."""
        pricing = BUNDLE_PRICING[tier]

        # Calculate volume discount
        if num_scenes >= 50:
            discount = pricing.volume_discount_50
        elif num_scenes >= 25:
            discount = pricing.volume_discount_25
        elif num_scenes >= 10:
            discount = pricing.volume_discount_10
        else:
            discount = 0.0

        per_scene_with_discount = pricing.per_scene_price * (1 - discount)
        total = per_scene_with_discount * num_scenes

        return {
            "tier": tier.value,
            "num_scenes": num_scenes,
            "base_per_scene": pricing.per_scene_price,
            "discount_percent": discount * 100,
            "per_scene_with_discount": per_scene_with_discount,
            "total": total,
            "savings": pricing.per_scene_price * num_scenes - total,
        }

    def generate_pipeline_config(
        self,
        tier: BundleTier,
        scene_id: str,
        robot_type: str = "franka",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate complete pipeline configuration from bundle."""
        features = self.get_bundle_config(tier, overrides)

        config = {
            "scene_id": scene_id,
            "bundle_tier": tier.value,
            "robot_type": robot_type,

            # Episode generation
            "episode_generation": {
                "max_variations": features.variations,
                "episodes_per_variation": features.episodes_per_variation,
                "total_episodes": features.episodes,
                "fps": 30.0,
                "min_quality_score": features.min_quality_score,
                "use_cpgen": True,
                "use_validation": True,
            },

            # Sensor capture
            "sensor_capture": {
                "num_cameras": features.rgb_cameras,
                "depth_enabled": features.depth_enabled,
                "segmentation_enabled": features.segmentation_enabled,
                "resolution": [640, 480],
            },

            # Data export
            "data_export": {
                "format": "lerobot",
                "version": features.lerobot_version,
                "streaming_enabled": features.streaming_enabled,
            },

            # Language annotations
            "language": {
                "enabled": features.language_annotations,
                "num_variations": features.num_language_variations,
                "use_llm": features.language_annotations,
            },

            # VLA fine-tuning
            "vla_finetuning": {
                "enabled": features.vla_finetuning_package,
                "models": features.vla_models,
            },

            # Sim2Real validation
            "sim2real": {
                "enabled": features.sim2real_validation,
                "tier": features.sim2real_tier,
                "min_trials": features.sim2real_trials,
            },

            # Advanced capabilities
            "advanced": {
                "contact_rich": features.contact_rich_tasks,
                "tactile": {
                    "enabled": features.tactile_simulation,
                    "sensor_type": features.tactile_sensor_type,
                },
                "multi_robot": {
                    "enabled": features.multi_robot,
                    "max_robots": features.max_robots,
                },
                "deformable": features.deformable_objects,
                "bimanual": features.bimanual,
            },

            # Isaac Lab
            "isaac_lab": {
                "generate_package": True,
                "runtime_validation": features.isaac_lab_runtime_validation,
            },

            # DWM
            "dwm": {
                "enabled": features.dwm_conditioning,
            },
        }

        return config

    def save_config(
        self,
        config: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """Save configuration to file."""
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
        self.log(f"Saved config to {output_path}")

    def print_tier_comparison(self) -> None:
        """Print comparison of all tiers."""
        print("\n" + "=" * 80)
        print("BLUEPRINT PIPELINE BUNDLE COMPARISON")
        print("=" * 80)

        headers = ["Feature", "Standard", "Pro", "Enterprise", "Foundation"]
        print(f"\n{'Feature':<35} {'Standard':>12} {'Pro':>12} {'Enterprise':>12} {'Foundation':>12}")
        print("-" * 85)

        features_to_compare = [
            ("Price", lambda f, t: f"${BUNDLE_PRICING[t].base_price:,}"),
            ("Variations", lambda f, t: f"{f.variations:,}"),
            ("Episodes", lambda f, t: f"{f.episodes:,}"),
            ("RGB Cameras", lambda f, t: str(f.rgb_cameras)),
            ("Depth", lambda f, t: "Yes" if f.depth_enabled else "No"),
            ("Segmentation", lambda f, t: "Yes" if f.segmentation_enabled else "No"),
            ("Language Annotations", lambda f, t: "Yes" if f.language_annotations else "No"),
            ("VLA Package", lambda f, t: "Yes" if f.vla_finetuning_package else "No"),
            ("VLA Models", lambda f, t: str(len(f.vla_models)) if f.vla_models else "-"),
            ("Audio Narration", lambda f, t: f.audio_voice_preset if f.audio_narration else "No"),
            ("Subtitles", lambda f, t: f.subtitle_style if f.subtitle_generation else "No"),
            ("Sim2Real Validation", lambda f, t: f.sim2real_tier if f.sim2real_validation else "No"),
            ("Contact-Rich Tasks", lambda f, t: "Yes" if f.contact_rich_tasks else "No"),
            ("Tactile Simulation", lambda f, t: "Yes" if f.tactile_simulation else "No"),
            ("Multi-Robot", lambda f, t: f"Up to {f.max_robots}" if f.multi_robot else "No"),
            ("Bimanual", lambda f, t: "Yes" if f.bimanual else "No"),
            ("Deformable Objects", lambda f, t: "Yes" if f.deformable_objects else "No"),
            ("DWM Conditioning", lambda f, t: "Yes" if f.dwm_conditioning else "No"),
            ("Priority Support", lambda f, t: "Yes" if f.priority_support else "No"),
        ]

        for feature_name, getter in features_to_compare:
            row = f"{feature_name:<35}"
            for tier in [BundleTier.STANDARD, BundleTier.PRO, BundleTier.ENTERPRISE, BundleTier.FOUNDATION]:
                features = BUNDLE_CONFIGS[tier]
                value = getter(features, tier)
                row += f" {value:>12}"
            print(row)

        print("\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Bundle configuration management"
    )

    subparsers = parser.add_subparsers(dest="command")

    # Compare tiers
    compare_parser = subparsers.add_parser("compare", help="Compare bundle tiers")

    # Generate config
    gen_parser = subparsers.add_parser("generate", help="Generate pipeline config")
    gen_parser.add_argument("--tier", choices=["standard", "pro", "enterprise", "foundation"], required=True)
    gen_parser.add_argument("--scene-id", required=True)
    gen_parser.add_argument("--robot-type", default="franka")
    gen_parser.add_argument("--output", type=Path, required=True)

    # Pricing
    price_parser = subparsers.add_parser("pricing", help="Calculate pricing")
    price_parser.add_argument("--tier", choices=["standard", "pro", "enterprise"], required=True)
    price_parser.add_argument("--scenes", type=int, default=1)

    args = parser.parse_args()

    manager = BundleConfigManager()

    if args.command == "compare":
        manager.print_tier_comparison()

    elif args.command == "generate":
        config = manager.generate_pipeline_config(
            tier=BundleTier(args.tier),
            scene_id=args.scene_id,
            robot_type=args.robot_type,
        )
        manager.save_config(config, args.output)
        print(f"Generated {args.tier} bundle config for {args.scene_id}")

    elif args.command == "pricing":
        pricing = manager.get_pricing(
            tier=BundleTier(args.tier),
            num_scenes=args.scenes,
        )
        print(f"\nPricing for {args.tier.upper()} bundle ({args.scenes} scenes):")
        print(f"  Base per scene: ${pricing['base_per_scene']:,}")
        print(f"  Volume discount: {pricing['discount_percent']:.0f}%")
        print(f"  Per scene after discount: ${pricing['per_scene_with_discount']:,.0f}")
        print(f"  Total: ${pricing['total']:,.0f}")
        print(f"  Savings: ${pricing['savings']:,.0f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
