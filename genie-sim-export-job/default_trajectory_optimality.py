#!/usr/bin/env python3
"""
Default Trajectory Optimality Analysis for Genie Sim 3.0 & Arena.

Previously $10,000-$25,000 upsell - NOW INCLUDED BY DEFAULT!

Analyzes trajectory quality to ensure training data is optimal.

Features (DEFAULT - FREE):
- Path length efficiency (actual vs optimal)
- Jerk analysis (smoothness scoring)
- Energy efficiency metrics
- Velocity profile analysis
- Training suitability score
- Outlier trajectory detection

Output:
- trajectory_optimality_config.json - Analysis configuration
"""

import json
from pathlib import Path
from typing import Dict


def create_default_trajectory_optimality_exporter(
    scene_id: str,
    output_dir: Path,
) -> Dict[str, Path]:
    """Create trajectory optimality analysis config (DEFAULT)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "trajectory_optimality_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "enabled": True,
            "scene_id": scene_id,
            "analysis_config": {
                "compute_path_efficiency": True,
                "compute_smoothness_jerk": True,
                "compute_energy_efficiency": True,
                "compute_velocity_profiles": True,
                "detect_outliers": True,
                "quality_thresholds": {
                    "excellent_jerk_threshold": 100,
                    "good_jerk_threshold": 300,
                    "path_efficiency_excellent": 0.8,
                    "path_efficiency_good": 0.6,
                },
            },
            "output_manifests": {
                "trajectory_quality_report": "trajectory_quality_report.json",
                "outlier_trajectories": "outlier_trajectories.json",
                "training_suitability": "training_suitability_score.json",
            },
            "value": "Previously $10,000-$25,000 upsell - NOW FREE BY DEFAULT",
        }, f, indent=2)

    return {"trajectory_optimality_config": config_path}
