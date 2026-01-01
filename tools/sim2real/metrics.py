"""Sim-to-Real Transfer Metrics.

Functions for computing and analyzing sim-to-real transfer quality.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple


def compute_transfer_gap(
    sim_success_rate: float,
    real_success_rate: float,
) -> float:
    """Compute transfer gap between sim and real success rates.

    A lower gap indicates better transfer.

    Args:
        sim_success_rate: Success rate in simulation (0-1)
        real_success_rate: Success rate in real world (0-1)

    Returns:
        Transfer gap (sim - real)
    """
    return sim_success_rate - real_success_rate


def compute_success_rate(
    outcomes: List[str],
    success_value: str = "success",
) -> float:
    """Compute success rate from list of outcomes.

    Args:
        outcomes: List of outcome strings
        success_value: Value indicating success

    Returns:
        Success rate (0-1)
    """
    if not outcomes:
        return 0.0

    successes = sum(1 for o in outcomes if o.lower() == success_value.lower())
    return successes / len(outcomes)


def compute_policy_divergence(
    sim_actions: List[List[float]],
    real_actions: List[List[float]],
) -> Dict[str, float]:
    """Compute divergence between sim and real policy actions.

    Measures how differently the policy behaves in sim vs real,
    which can indicate domain gap issues.

    Args:
        sim_actions: List of action vectors from simulation
        real_actions: List of action vectors from real world

    Returns:
        Dict with divergence metrics
    """
    if not sim_actions or not real_actions:
        return {"error": "Insufficient data"}

    # Compute per-dimension statistics
    dim = len(sim_actions[0])

    sim_means = []
    sim_stds = []
    real_means = []
    real_stds = []

    for d in range(dim):
        sim_dim = [a[d] for a in sim_actions if len(a) > d]
        real_dim = [a[d] for a in real_actions if len(a) > d]

        if sim_dim and real_dim:
            sim_means.append(statistics.mean(sim_dim))
            sim_stds.append(statistics.stdev(sim_dim) if len(sim_dim) > 1 else 0)
            real_means.append(statistics.mean(real_dim))
            real_stds.append(statistics.stdev(real_dim) if len(real_dim) > 1 else 0)

    # Mean divergence
    mean_divergence = sum(
        abs(s - r) for s, r in zip(sim_means, real_means)
    ) / len(sim_means) if sim_means else 0

    # Variance ratio (how much more variable is real vs sim)
    variance_ratios = []
    for s, r in zip(sim_stds, real_stds):
        if s > 0:
            variance_ratios.append(r / s)

    avg_variance_ratio = (
        statistics.mean(variance_ratios) if variance_ratios else 1.0
    )

    return {
        "mean_divergence": mean_divergence,
        "variance_ratio": avg_variance_ratio,
        "action_dimensions": dim,
        "sim_sample_count": len(sim_actions),
        "real_sample_count": len(real_actions),
    }


def compute_timing_ratio(
    sim_times: List[float],
    real_times: List[float],
) -> Dict[str, float]:
    """Compute timing comparison between sim and real.

    Args:
        sim_times: Completion times in simulation
        real_times: Completion times in real world

    Returns:
        Dict with timing metrics
    """
    if not sim_times or not real_times:
        return {"error": "Insufficient data"}

    sim_mean = statistics.mean(sim_times)
    real_mean = statistics.mean(real_times)

    ratio = real_mean / sim_mean if sim_mean > 0 else float('inf')

    return {
        "sim_mean_time": sim_mean,
        "real_mean_time": real_mean,
        "time_ratio": ratio,
        "sim_std": statistics.stdev(sim_times) if len(sim_times) > 1 else 0,
        "real_std": statistics.stdev(real_times) if len(real_times) > 1 else 0,
    }


def compute_failure_mode_distribution(
    failure_modes: List[str],
) -> Dict[str, Any]:
    """Analyze failure mode distribution.

    Args:
        failure_modes: List of failure mode strings

    Returns:
        Dict with failure analysis
    """
    if not failure_modes:
        return {"total_failures": 0}

    # Count occurrences
    counts: Dict[str, int] = {}
    for mode in failure_modes:
        mode = mode or "unknown"
        counts[mode] = counts.get(mode, 0) + 1

    # Sort by frequency
    sorted_modes = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "total_failures": len(failure_modes),
        "unique_modes": len(counts),
        "distribution": dict(sorted_modes),
        "top_mode": sorted_modes[0][0] if sorted_modes else None,
        "top_mode_percentage": sorted_modes[0][1] / len(failure_modes) if sorted_modes else 0,
    }


def compute_confidence_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for success rate.

    Args:
        successes: Number of successful trials
        total: Total number of trials
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (lower, upper) bounds
    """
    if total == 0:
        return (0.0, 0.0)

    import math

    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)

    p = successes / total

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) / total + z**2 / (4 * total**2))) / denominator

    return (
        max(0.0, center - margin),
        min(1.0, center + margin)
    )


def compute_sample_size_for_power(
    expected_gap: float = 0.10,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    """Compute required sample size for detecting transfer gap.

    Args:
        expected_gap: Expected difference to detect
        power: Statistical power (default 0.80)
        alpha: Significance level (default 0.05)

    Returns:
        Required sample size per condition
    """
    import math

    # Z-values
    z_alpha = 1.96 if alpha == 0.05 else 2.576
    z_beta = 0.84 if power == 0.80 else 1.28

    # Assume baseline success rate of 0.80
    p1 = 0.80
    p2 = p1 - expected_gap

    # Pooled proportion
    p_bar = (p1 + p2) / 2

    numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) +
                 z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
    denominator = (p1 - p2)**2

    return int(math.ceil(numerator / denominator))


def interpret_transfer_quality(
    transfer_gap: float,
    real_success_rate: float,
    real_trials: int,
) -> Dict[str, Any]:
    """Interpret transfer quality and provide recommendations.

    Args:
        transfer_gap: Difference between sim and real success
        real_success_rate: Actual real-world success rate
        real_trials: Number of real-world trials

    Returns:
        Dict with interpretation and recommendations
    """
    # Quality rating
    if transfer_gap < 0.05:
        quality = "excellent"
        color = "green"
    elif transfer_gap < 0.15:
        quality = "good"
        color = "blue"
    elif transfer_gap < 0.30:
        quality = "moderate"
        color = "yellow"
    else:
        quality = "poor"
        color = "red"

    # Recommendations
    recommendations = []

    if real_trials < 10:
        recommendations.append(
            f"Need more real-world trials (currently {real_trials}, recommend 20+)"
        )

    if transfer_gap > 0.15:
        recommendations.extend([
            "Increase domain randomization (lighting, textures, poses)",
            "Verify physics properties match real objects",
            "Check camera calibration and observation alignment",
        ])

    if real_success_rate < 0.50:
        recommendations.extend([
            "Consider sim2real fine-tuning with real data",
            "Review failure modes for systematic issues",
            "Validate basic motion primitives work correctly",
        ])

    if transfer_gap > 0.30:
        recommendations.append(
            "Consider collecting real-world demonstrations for imitation learning"
        )

    # Confidence assessment
    if real_trials >= 20:
        lower, upper = compute_confidence_interval(
            int(real_success_rate * real_trials),
            real_trials
        )
        confidence_note = f"95% CI for real success: [{lower:.1%}, {upper:.1%}]"
    else:
        confidence_note = "Insufficient trials for reliable confidence interval"

    return {
        "quality": quality,
        "quality_color": color,
        "transfer_gap": f"{transfer_gap:.1%}",
        "real_success_rate": f"{real_success_rate:.1%}",
        "confidence_note": confidence_note,
        "recommendations": recommendations,
        "production_ready": quality in ["excellent", "good"] and real_trials >= 20,
    }
