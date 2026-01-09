#!/usr/bin/env python3
"""
Default Policy Leaderboard for Genie Sim 3.0 & Arena.

Previously $20,000-$40,000 upsell - NOW INCLUDED BY DEFAULT!

Multi-policy comparison with statistical rigor.

Features (DEFAULT - FREE):
- Policy rankings with confidence intervals (Wilson score, bootstrap)
- Statistical significance testing (t-test, Mann-Whitney U)
- Performance comparison across tasks
- Rank stability analysis
- Pairwise comparison matrix

Output:
- policy_leaderboard_config.json - Leaderboard configuration
"""

import json
from pathlib import Path
from typing import Dict


def create_default_policy_leaderboard_exporter(
    scene_id: str,
    output_dir: Path,
) -> Dict[str, Path]:
    """Create policy leaderboard config (DEFAULT)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "policy_leaderboard_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "enabled": True,
            "scene_id": scene_id,
            "leaderboard_config": {
                "confidence_level": 0.95,
                "significance_alpha": 0.05,
                "bootstrap_samples": 10000,
                "metrics": [
                    "success_rate",
                    "mean_reward",
                    "mean_episode_length",
                    "grasp_success_rate",
                    "collision_rate",
                    "composite_score",
                ],
                "confidence_methods": {
                    "success_rate": "wilson",
                    "mean_reward": "bootstrap",
                    "episode_length": "normal",
                },
                "significance_tests": {
                    "success_rate": "two_proportion_z_test",
                    "reward": "independent_ttest",
                    "non_parametric": "mann_whitney_u",
                },
            },
            "output_manifests": {
                "leaderboard": "policy_leaderboard.json",
                "pairwise_comparisons": "pairwise_comparison_matrix.json",
                "rank_stability": "rank_stability_analysis.json",
            },
            "value": "Previously $20,000-$40,000 upsell - NOW FREE BY DEFAULT",
        }, f, indent=2)

    return {"policy_leaderboard_config": config_path}
