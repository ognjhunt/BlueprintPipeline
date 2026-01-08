"""
Baseline Benchmarks Package for BlueprintPipeline.

Provides baseline success rate benchmarks that labs can use to:
1. Verify their environment works correctly
2. Compare their trained policies against
3. Justify purchase to procurement ("we bought scenes with verified baselines")

Baseline Types:
- Scripted: Hand-coded waypoint following (proves scene works)
- Heuristic: Simple baselines (random, nearest-neighbor)
- Pretrained: Foundation model zero-shot (e.g., OpenVLA)

This is an UPSELL differentiator:
- Scenes WITHOUT baselines are "unverified assets"
- Scenes WITH baselines are "benchmark-ready environments"
"""

from .baseline_benchmarks import (
    SceneBaselines,
    TaskBaseline,
    BaselineResult,
    BaselineBenchmarkGenerator,
    generate_scene_baselines,
    BaselineType,
    TaskCategory,
    EXPECTED_BASELINES,
    get_expected_baseline_range,
)

__all__ = [
    "SceneBaselines",
    "TaskBaseline",
    "BaselineResult",
    "BaselineBenchmarkGenerator",
    "generate_scene_baselines",
    "BaselineType",
    "TaskCategory",
    "EXPECTED_BASELINES",
    "get_expected_baseline_range",
]
