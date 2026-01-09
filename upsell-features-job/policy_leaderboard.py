"""
Policy Leaderboard and Statistical Analysis Module
===================================================

Provides multi-policy comparison rankings with confidence intervals
and statistical significance testing - currently NOT captured in standard pipeline.

Premium Analytics Feature - Upsell Value: $20,000 - $40,000

Features:
- Multi-policy leaderboard generation
- Confidence interval calculation (Wilson score, bootstrap)
- Statistical significance testing (t-test, Mann-Whitney U)
- Performance comparison across tasks and conditions
- Ranking stability analysis

Author: BlueprintPipeline Premium Analytics
"""

import json
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
from pathlib import Path
import statistics
import hashlib


class ConfidenceMethod(Enum):
    """Methods for calculating confidence intervals."""
    WILSON = "wilson"  # Wilson score interval (best for proportions)
    BOOTSTRAP = "bootstrap"  # Bootstrap resampling
    NORMAL = "normal"  # Normal approximation
    CLOPPER_PEARSON = "clopper_pearson"  # Exact binomial


class SignificanceTest(Enum):
    """Statistical significance tests."""
    TTEST_IND = "ttest_independent"  # Independent samples t-test
    TTEST_PAIRED = "ttest_paired"  # Paired samples t-test
    MANN_WHITNEY = "mann_whitney_u"  # Non-parametric
    PERMUTATION = "permutation"  # Permutation test


class RankingMetric(Enum):
    """Metrics for ranking policies."""
    SUCCESS_RATE = "success_rate"
    MEAN_REWARD = "mean_reward"
    MEAN_EPISODE_LENGTH = "mean_episode_length"
    GRASP_SUCCESS_RATE = "grasp_success_rate"
    COLLISION_RATE = "collision_rate"
    TIME_TO_COMPLETION = "time_to_completion"
    COMPOSITE_SCORE = "composite_score"


@dataclass
class PolicyEvalResult:
    """Evaluation results for a single policy."""
    policy_id: str
    policy_name: str
    policy_version: str
    checkpoint_path: Optional[str] = None

    # Task info
    task_name: str = ""
    task_variant: str = ""

    # Episode counts
    total_episodes: int = 0
    successful_episodes: int = 0
    failed_episodes: int = 0

    # Primary metrics
    success_rate: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_episode_length: float = 0.0
    std_episode_length: float = 0.0

    # Detailed metrics
    grasp_attempts: int = 0
    successful_grasps: int = 0
    grasp_success_rate: float = 0.0
    total_collisions: int = 0
    collision_rate: float = 0.0
    mean_time_to_completion: float = 0.0
    std_time_to_completion: float = 0.0

    # Raw data for statistical tests
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_outcomes: List[bool] = field(default_factory=list)  # True = success
    completion_times: List[float] = field(default_factory=list)

    # Metadata
    eval_timestamp: datetime = field(default_factory=datetime.now)
    num_parallel_envs: int = 0
    total_wall_time: float = 0.0


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float  # e.g., 0.95 for 95%
    method: ConfidenceMethod
    sample_size: int


@dataclass
class SignificanceResult:
    """Result of statistical significance test."""
    test_type: SignificanceTest
    statistic: float
    p_value: float
    significant: bool  # at alpha level
    alpha: float
    effect_size: float
    effect_size_interpretation: str  # "small", "medium", "large"
    policy_a_id: str
    policy_b_id: str
    metric: str
    conclusion: str


@dataclass
class PolicyRanking:
    """Ranking entry for a policy."""
    rank: int
    policy_id: str
    policy_name: str
    metric_value: float
    confidence_interval: ConfidenceInterval
    rank_stability: float  # Probability this is the true rank
    significantly_better_than: List[str]  # Policy IDs this beats
    significantly_worse_than: List[str]  # Policy IDs that beat this
    rank_change_from_previous: Optional[int] = None


@dataclass
class Leaderboard:
    """Complete policy leaderboard."""
    leaderboard_id: str
    task_name: str
    ranking_metric: RankingMetric
    confidence_level: float
    generated_at: datetime

    rankings: List[PolicyRanking] = field(default_factory=list)
    pairwise_comparisons: List[SignificanceResult] = field(default_factory=list)

    # Metadata
    total_policies: int = 0
    total_episodes_evaluated: int = 0
    evaluation_conditions: Dict[str, Any] = field(default_factory=dict)


class PolicyLeaderboardGenerator:
    """
    Generates policy leaderboards with statistical rigor.

    This module provides multi-policy comparison that is NOT
    captured in the standard Isaac Lab Arena output.
    """

    def __init__(
        self,
        output_dir: str = "./leaderboards",
        confidence_level: float = 0.95,
        alpha: float = 0.05,
        bootstrap_samples: int = 10000
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_level = confidence_level
        self.alpha = alpha
        self.bootstrap_samples = bootstrap_samples
        self.policy_results: Dict[str, PolicyEvalResult] = {}

    def add_policy_result(self, result: PolicyEvalResult) -> None:
        """Add policy evaluation result."""
        self.policy_results[result.policy_id] = result

    def compute_wilson_interval(
        self,
        successes: int,
        total: int,
        confidence: float = 0.95
    ) -> ConfidenceInterval:
        """
        Compute Wilson score confidence interval for proportions.

        This is the recommended method for success rate CIs as it
        handles edge cases better than normal approximation.
        """
        if total == 0:
            return ConfidenceInterval(
                point_estimate=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                confidence_level=confidence,
                method=ConfidenceMethod.WILSON,
                sample_size=0
            )

        p = successes / total
        z = self._z_score(confidence)

        denominator = 1 + z**2 / total
        centre = p + z**2 / (2 * total)
        margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)

        lower = (centre - margin) / denominator
        upper = (centre + margin) / denominator

        return ConfidenceInterval(
            point_estimate=p,
            lower_bound=max(0.0, lower),
            upper_bound=min(1.0, upper),
            confidence_level=confidence,
            method=ConfidenceMethod.WILSON,
            sample_size=total
        )

    def compute_bootstrap_interval(
        self,
        data: List[float],
        statistic: str = "mean",
        confidence: float = 0.95
    ) -> ConfidenceInterval:
        """
        Compute bootstrap confidence interval for any statistic.

        Non-parametric method that works well for non-normal distributions.
        """
        if not data:
            return ConfidenceInterval(
                point_estimate=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                confidence_level=confidence,
                method=ConfidenceMethod.BOOTSTRAP,
                sample_size=0
            )

        data_array = np.array(data)
        n = len(data)

        # Generate bootstrap samples
        bootstrap_stats = []
        for _ in range(self.bootstrap_samples):
            sample = np.random.choice(data_array, size=n, replace=True)
            if statistic == "mean":
                bootstrap_stats.append(np.mean(sample))
            elif statistic == "median":
                bootstrap_stats.append(np.median(sample))
            elif statistic == "std":
                bootstrap_stats.append(np.std(sample))

        bootstrap_stats = np.array(bootstrap_stats)

        # Percentile method
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

        if statistic == "mean":
            point = np.mean(data_array)
        elif statistic == "median":
            point = np.median(data_array)
        else:
            point = np.std(data_array)

        return ConfidenceInterval(
            point_estimate=float(point),
            lower_bound=float(lower),
            upper_bound=float(upper),
            confidence_level=confidence,
            method=ConfidenceMethod.BOOTSTRAP,
            sample_size=n
        )

    def compute_normal_interval(
        self,
        data: List[float],
        confidence: float = 0.95
    ) -> ConfidenceInterval:
        """Compute normal approximation confidence interval."""
        if not data:
            return ConfidenceInterval(
                point_estimate=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                confidence_level=confidence,
                method=ConfidenceMethod.NORMAL,
                sample_size=0
            )

        n = len(data)
        mean = statistics.mean(data)
        std = statistics.stdev(data) if n > 1 else 0.0

        z = self._z_score(confidence)
        margin = z * std / math.sqrt(n)

        return ConfidenceInterval(
            point_estimate=mean,
            lower_bound=mean - margin,
            upper_bound=mean + margin,
            confidence_level=confidence,
            method=ConfidenceMethod.NORMAL,
            sample_size=n
        )

    def _z_score(self, confidence: float) -> float:
        """Get z-score for confidence level."""
        # Common confidence levels
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        return z_scores.get(confidence, 1.96)

    def independent_ttest(
        self,
        policy_a: PolicyEvalResult,
        policy_b: PolicyEvalResult,
        metric: str = "reward"
    ) -> SignificanceResult:
        """
        Perform independent samples t-test between two policies.

        Tests if there's a statistically significant difference
        in performance between policies.
        """
        if metric == "reward":
            data_a = policy_a.episode_rewards
            data_b = policy_b.episode_rewards
        elif metric == "length":
            data_a = [float(x) for x in policy_a.episode_lengths]
            data_b = [float(x) for x in policy_b.episode_lengths]
        elif metric == "completion_time":
            data_a = policy_a.completion_times
            data_b = policy_b.completion_times
        else:
            data_a = policy_a.episode_rewards
            data_b = policy_b.episode_rewards

        if not data_a or not data_b:
            return SignificanceResult(
                test_type=SignificanceTest.TTEST_IND,
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                effect_size=0.0,
                effect_size_interpretation="none",
                policy_a_id=policy_a.policy_id,
                policy_b_id=policy_b.policy_id,
                metric=metric,
                conclusion="Insufficient data for comparison"
            )

        # Compute t-statistic manually
        n_a, n_b = len(data_a), len(data_b)
        mean_a, mean_b = np.mean(data_a), np.mean(data_b)
        var_a, var_b = np.var(data_a, ddof=1), np.var(data_b, ddof=1)

        # Pooled standard error
        pooled_se = math.sqrt(var_a / n_a + var_b / n_b)

        if pooled_se == 0:
            t_stat = 0.0
            p_value = 1.0
        else:
            t_stat = (mean_a - mean_b) / pooled_se
            # Approximate p-value using normal distribution for large samples
            p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

        # Cohen's d effect size
        pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

        # Interpret effect size
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            interpretation = "negligible"
        elif abs_effect < 0.5:
            interpretation = "small"
        elif abs_effect < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        significant = p_value < self.alpha

        if significant:
            if mean_a > mean_b:
                conclusion = f"{policy_a.policy_name} significantly outperforms {policy_b.policy_name} (p={p_value:.4f})"
            else:
                conclusion = f"{policy_b.policy_name} significantly outperforms {policy_a.policy_name} (p={p_value:.4f})"
        else:
            conclusion = f"No significant difference between {policy_a.policy_name} and {policy_b.policy_name} (p={p_value:.4f})"

        return SignificanceResult(
            test_type=SignificanceTest.TTEST_IND,
            statistic=float(t_stat),
            p_value=float(p_value),
            significant=significant,
            alpha=self.alpha,
            effect_size=float(effect_size),
            effect_size_interpretation=interpretation,
            policy_a_id=policy_a.policy_id,
            policy_b_id=policy_b.policy_id,
            metric=metric,
            conclusion=conclusion
        )

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF using error function approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def mann_whitney_test(
        self,
        policy_a: PolicyEvalResult,
        policy_b: PolicyEvalResult,
        metric: str = "reward"
    ) -> SignificanceResult:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).

        Better for non-normal distributions or ordinal data.
        """
        if metric == "reward":
            data_a = policy_a.episode_rewards
            data_b = policy_b.episode_rewards
        elif metric == "length":
            data_a = [float(x) for x in policy_a.episode_lengths]
            data_b = [float(x) for x in policy_b.episode_lengths]
        else:
            data_a = policy_a.episode_rewards
            data_b = policy_b.episode_rewards

        if not data_a or not data_b:
            return SignificanceResult(
                test_type=SignificanceTest.MANN_WHITNEY,
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                effect_size=0.0,
                effect_size_interpretation="none",
                policy_a_id=policy_a.policy_id,
                policy_b_id=policy_b.policy_id,
                metric=metric,
                conclusion="Insufficient data for comparison"
            )

        n_a, n_b = len(data_a), len(data_b)

        # Compute U statistic
        combined = [(x, 'a') for x in data_a] + [(x, 'b') for x in data_b]
        combined.sort(key=lambda x: x[0])

        # Assign ranks (handling ties)
        ranks = {}
        i = 0
        while i < len(combined):
            j = i
            while j < len(combined) and combined[j][0] == combined[i][0]:
                j += 1
            avg_rank = (i + j + 1) / 2  # Average rank for ties
            for k in range(i, j):
                if combined[k][1] not in ranks:
                    ranks[combined[k][1]] = []
                ranks[combined[k][1]].append(avg_rank)
            i = j

        r_a = sum(ranks.get('a', []))
        u_a = r_a - n_a * (n_a + 1) / 2
        u_b = n_a * n_b - u_a
        u_stat = min(u_a, u_b)

        # Normal approximation for p-value (large sample)
        mean_u = n_a * n_b / 2
        std_u = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)

        if std_u > 0:
            z = (u_stat - mean_u) / std_u
            p_value = 2 * (1 - self._normal_cdf(abs(z)))
        else:
            p_value = 1.0

        # Effect size (rank-biserial correlation)
        effect_size = 1 - (2 * u_stat) / (n_a * n_b)

        abs_effect = abs(effect_size)
        if abs_effect < 0.1:
            interpretation = "negligible"
        elif abs_effect < 0.3:
            interpretation = "small"
        elif abs_effect < 0.5:
            interpretation = "medium"
        else:
            interpretation = "large"

        significant = p_value < self.alpha

        if significant:
            median_a = np.median(data_a)
            median_b = np.median(data_b)
            if median_a > median_b:
                conclusion = f"{policy_a.policy_name} ranks significantly higher than {policy_b.policy_name} (p={p_value:.4f})"
            else:
                conclusion = f"{policy_b.policy_name} ranks significantly higher than {policy_a.policy_name} (p={p_value:.4f})"
        else:
            conclusion = f"No significant rank difference between {policy_a.policy_name} and {policy_b.policy_name}"

        return SignificanceResult(
            test_type=SignificanceTest.MANN_WHITNEY,
            statistic=float(u_stat),
            p_value=float(p_value),
            significant=significant,
            alpha=self.alpha,
            effect_size=float(effect_size),
            effect_size_interpretation=interpretation,
            policy_a_id=policy_a.policy_id,
            policy_b_id=policy_b.policy_id,
            metric=metric,
            conclusion=conclusion
        )

    def success_rate_comparison(
        self,
        policy_a: PolicyEvalResult,
        policy_b: PolicyEvalResult
    ) -> SignificanceResult:
        """
        Compare success rates between policies using proportion test.

        Uses two-proportion z-test.
        """
        n_a = policy_a.total_episodes
        n_b = policy_b.total_episodes
        x_a = policy_a.successful_episodes
        x_b = policy_b.successful_episodes

        if n_a == 0 or n_b == 0:
            return SignificanceResult(
                test_type=SignificanceTest.TTEST_IND,
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                effect_size=0.0,
                effect_size_interpretation="none",
                policy_a_id=policy_a.policy_id,
                policy_b_id=policy_b.policy_id,
                metric="success_rate",
                conclusion="Insufficient data"
            )

        p_a = x_a / n_a
        p_b = x_b / n_b

        # Pooled proportion
        p_pooled = (x_a + x_b) / (n_a + n_b)

        # Standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))

        if se == 0:
            z_stat = 0.0
            p_value = 1.0
        else:
            z_stat = (p_a - p_b) / se
            p_value = 2 * (1 - self._normal_cdf(abs(z_stat)))

        # Effect size (Cohen's h)
        effect_size = 2 * (math.asin(math.sqrt(p_a)) - math.asin(math.sqrt(p_b)))

        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            interpretation = "small"
        elif abs_effect < 0.5:
            interpretation = "medium"
        else:
            interpretation = "large"

        significant = p_value < self.alpha

        if significant:
            if p_a > p_b:
                conclusion = f"{policy_a.policy_name} has significantly higher success rate ({p_a*100:.1f}% vs {p_b*100:.1f}%, p={p_value:.4f})"
            else:
                conclusion = f"{policy_b.policy_name} has significantly higher success rate ({p_b*100:.1f}% vs {p_a*100:.1f}%, p={p_value:.4f})"
        else:
            conclusion = f"No significant difference in success rates ({p_a*100:.1f}% vs {p_b*100:.1f}%, p={p_value:.4f})"

        return SignificanceResult(
            test_type=SignificanceTest.TTEST_IND,
            statistic=float(z_stat),
            p_value=float(p_value),
            significant=significant,
            alpha=self.alpha,
            effect_size=float(effect_size),
            effect_size_interpretation=interpretation,
            policy_a_id=policy_a.policy_id,
            policy_b_id=policy_b.policy_id,
            metric="success_rate",
            conclusion=conclusion
        )

    def generate_leaderboard(
        self,
        task_name: str,
        metric: RankingMetric = RankingMetric.SUCCESS_RATE,
        run_significance_tests: bool = True
    ) -> Leaderboard:
        """
        Generate complete policy leaderboard with rankings and confidence intervals.

        This is the KEY UPSELL FEATURE - not captured in standard Arena output.
        """
        leaderboard_id = hashlib.sha256(
            f"{task_name}_{metric.value}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        leaderboard = Leaderboard(
            leaderboard_id=leaderboard_id,
            task_name=task_name,
            ranking_metric=metric,
            confidence_level=self.confidence_level,
            generated_at=datetime.now(),
            total_policies=len(self.policy_results),
            total_episodes_evaluated=sum(p.total_episodes for p in self.policy_results.values())
        )

        # Filter to relevant task
        relevant_policies = [
            p for p in self.policy_results.values()
            if p.task_name == task_name or not task_name
        ]

        if not relevant_policies:
            return leaderboard

        # Compute metric values and CIs for each policy
        policy_metrics: List[Tuple[PolicyEvalResult, float, ConfidenceInterval]] = []

        for policy in relevant_policies:
            if metric == RankingMetric.SUCCESS_RATE:
                value = policy.success_rate
                ci = self.compute_wilson_interval(
                    policy.successful_episodes,
                    policy.total_episodes,
                    self.confidence_level
                )
            elif metric == RankingMetric.MEAN_REWARD:
                value = policy.mean_reward
                ci = self.compute_bootstrap_interval(
                    policy.episode_rewards,
                    "mean",
                    self.confidence_level
                )
            elif metric == RankingMetric.MEAN_EPISODE_LENGTH:
                value = policy.mean_episode_length
                ci = self.compute_bootstrap_interval(
                    [float(x) for x in policy.episode_lengths],
                    "mean",
                    self.confidence_level
                )
            elif metric == RankingMetric.GRASP_SUCCESS_RATE:
                value = policy.grasp_success_rate
                ci = self.compute_wilson_interval(
                    policy.successful_grasps,
                    policy.grasp_attempts,
                    self.confidence_level
                )
            elif metric == RankingMetric.COLLISION_RATE:
                value = policy.collision_rate
                ci = self.compute_wilson_interval(
                    policy.total_collisions,
                    policy.total_episodes,
                    self.confidence_level
                )
            elif metric == RankingMetric.TIME_TO_COMPLETION:
                value = policy.mean_time_to_completion
                ci = self.compute_bootstrap_interval(
                    policy.completion_times,
                    "mean",
                    self.confidence_level
                )
            else:  # COMPOSITE_SCORE
                # Weighted combination of metrics
                value = (
                    0.4 * policy.success_rate +
                    0.3 * policy.grasp_success_rate +
                    0.2 * (1 - policy.collision_rate) +
                    0.1 * (1 - min(1.0, policy.mean_time_to_completion / 10.0))
                )
                ci = ConfidenceInterval(
                    point_estimate=value,
                    lower_bound=value * 0.9,
                    upper_bound=min(1.0, value * 1.1),
                    confidence_level=self.confidence_level,
                    method=ConfidenceMethod.NORMAL,
                    sample_size=policy.total_episodes
                )

            policy_metrics.append((policy, value, ci))

        # Sort by metric value (descending for most metrics, ascending for time/collision)
        reverse = metric not in [RankingMetric.COLLISION_RATE, RankingMetric.TIME_TO_COMPLETION]
        policy_metrics.sort(key=lambda x: x[1], reverse=reverse)

        # Run pairwise significance tests
        pairwise_results: Dict[Tuple[str, str], SignificanceResult] = {}

        if run_significance_tests:
            for i, (policy_a, _, _) in enumerate(policy_metrics):
                for j, (policy_b, _, _) in enumerate(policy_metrics):
                    if i >= j:
                        continue

                    if metric == RankingMetric.SUCCESS_RATE:
                        result = self.success_rate_comparison(policy_a, policy_b)
                    else:
                        result = self.independent_ttest(policy_a, policy_b, "reward")

                    pairwise_results[(policy_a.policy_id, policy_b.policy_id)] = result
                    leaderboard.pairwise_comparisons.append(result)

        # Build rankings
        for rank, (policy, value, ci) in enumerate(policy_metrics, 1):
            better_than = []
            worse_than = []

            for (a_id, b_id), result in pairwise_results.items():
                if result.significant:
                    if a_id == policy.policy_id:
                        if (reverse and result.effect_size > 0) or (not reverse and result.effect_size < 0):
                            better_than.append(b_id)
                        else:
                            worse_than.append(b_id)
                    elif b_id == policy.policy_id:
                        if (reverse and result.effect_size < 0) or (not reverse and result.effect_size > 0):
                            worse_than.append(a_id)
                        else:
                            better_than.append(a_id)

            # Estimate rank stability using bootstrap
            rank_stability = self._estimate_rank_stability(
                policy, value, ci, policy_metrics
            )

            ranking = PolicyRanking(
                rank=rank,
                policy_id=policy.policy_id,
                policy_name=policy.policy_name,
                metric_value=value,
                confidence_interval=ci,
                rank_stability=rank_stability,
                significantly_better_than=better_than,
                significantly_worse_than=worse_than
            )

            leaderboard.rankings.append(ranking)

        return leaderboard

    def _estimate_rank_stability(
        self,
        policy: PolicyEvalResult,
        value: float,
        ci: ConfidenceInterval,
        all_policies: List[Tuple[PolicyEvalResult, float, ConfidenceInterval]]
    ) -> float:
        """
        Estimate probability that current rank is the true rank.

        Uses overlap of confidence intervals as proxy for rank stability.
        """
        current_rank = next(
            i for i, (p, _, _) in enumerate(all_policies, 1)
            if p.policy_id == policy.policy_id
        )

        # Count overlapping CIs with adjacent ranks
        overlaps = 0
        total_adjacent = 0

        for i, (other_policy, other_value, other_ci) in enumerate(all_policies, 1):
            if other_policy.policy_id == policy.policy_id:
                continue

            # Only consider adjacent ranks
            if abs(i - current_rank) > 2:
                continue

            total_adjacent += 1

            # Check CI overlap
            if ci.upper_bound >= other_ci.lower_bound and ci.lower_bound <= other_ci.upper_bound:
                overlaps += 1

        if total_adjacent == 0:
            return 1.0  # No adjacent policies, rank is stable

        # Stability decreases with more overlaps
        stability = 1.0 - (overlaps / total_adjacent) * 0.5

        return max(0.0, min(1.0, stability))

    def generate_comparison_matrix(
        self,
        task_name: str
    ) -> Dict[str, Any]:
        """
        Generate pairwise comparison matrix between all policies.

        Shows which policies significantly outperform others.
        """
        relevant_policies = [
            p for p in self.policy_results.values()
            if p.task_name == task_name or not task_name
        ]

        matrix = {
            "task_name": task_name,
            "policies": [p.policy_id for p in relevant_policies],
            "policy_names": {p.policy_id: p.policy_name for p in relevant_policies},
            "success_rate_matrix": {},
            "reward_matrix": {},
            "significant_wins": {p.policy_id: 0 for p in relevant_policies},
            "significant_losses": {p.policy_id: 0 for p in relevant_policies}
        }

        for policy_a in relevant_policies:
            matrix["success_rate_matrix"][policy_a.policy_id] = {}
            matrix["reward_matrix"][policy_a.policy_id] = {}

            for policy_b in relevant_policies:
                if policy_a.policy_id == policy_b.policy_id:
                    matrix["success_rate_matrix"][policy_a.policy_id][policy_b.policy_id] = None
                    matrix["reward_matrix"][policy_a.policy_id][policy_b.policy_id] = None
                    continue

                # Success rate comparison
                sr_result = self.success_rate_comparison(policy_a, policy_b)
                matrix["success_rate_matrix"][policy_a.policy_id][policy_b.policy_id] = {
                    "p_value": sr_result.p_value,
                    "significant": sr_result.significant,
                    "effect_size": sr_result.effect_size,
                    "winner": policy_a.policy_id if sr_result.effect_size > 0 and sr_result.significant else (
                        policy_b.policy_id if sr_result.effect_size < 0 and sr_result.significant else None
                    )
                }

                # Reward comparison
                reward_result = self.independent_ttest(policy_a, policy_b, "reward")
                matrix["reward_matrix"][policy_a.policy_id][policy_b.policy_id] = {
                    "p_value": reward_result.p_value,
                    "significant": reward_result.significant,
                    "effect_size": reward_result.effect_size,
                    "winner": policy_a.policy_id if reward_result.effect_size > 0 and reward_result.significant else (
                        policy_b.policy_id if reward_result.effect_size < 0 and reward_result.significant else None
                    )
                }

                # Track wins/losses
                if sr_result.significant:
                    if sr_result.effect_size > 0:
                        matrix["significant_wins"][policy_a.policy_id] += 1
                        matrix["significant_losses"][policy_b.policy_id] += 1
                    else:
                        matrix["significant_wins"][policy_b.policy_id] += 1
                        matrix["significant_losses"][policy_a.policy_id] += 1

        return matrix

    def save_leaderboard(self, leaderboard: Leaderboard) -> str:
        """Save leaderboard to JSON file."""
        output_file = self.output_dir / f"leaderboard_{leaderboard.leaderboard_id}.json"

        data = {
            "leaderboard_id": leaderboard.leaderboard_id,
            "task_name": leaderboard.task_name,
            "ranking_metric": leaderboard.ranking_metric.value,
            "confidence_level": leaderboard.confidence_level,
            "generated_at": leaderboard.generated_at.isoformat(),
            "total_policies": leaderboard.total_policies,
            "total_episodes_evaluated": leaderboard.total_episodes_evaluated,
            "rankings": [
                {
                    "rank": r.rank,
                    "policy_id": r.policy_id,
                    "policy_name": r.policy_name,
                    "metric_value": r.metric_value,
                    "confidence_interval": {
                        "point_estimate": r.confidence_interval.point_estimate,
                        "lower_bound": r.confidence_interval.lower_bound,
                        "upper_bound": r.confidence_interval.upper_bound,
                        "confidence_level": r.confidence_interval.confidence_level,
                        "method": r.confidence_interval.method.value
                    },
                    "rank_stability": r.rank_stability,
                    "significantly_better_than": r.significantly_better_than,
                    "significantly_worse_than": r.significantly_worse_than
                }
                for r in leaderboard.rankings
            ],
            "pairwise_comparisons": [
                {
                    "policy_a": c.policy_a_id,
                    "policy_b": c.policy_b_id,
                    "test_type": c.test_type.value,
                    "p_value": c.p_value,
                    "significant": c.significant,
                    "effect_size": c.effect_size,
                    "effect_size_interpretation": c.effect_size_interpretation,
                    "conclusion": c.conclusion
                }
                for c in leaderboard.pairwise_comparisons
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        return str(output_file)

    def generate_premium_report(
        self,
        task_name: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive premium leaderboard report.

        KEY UPSELL VALUE - multi-policy comparison with statistical rigor.
        """
        # Generate leaderboards for different metrics
        success_lb = self.generate_leaderboard(task_name, RankingMetric.SUCCESS_RATE)
        reward_lb = self.generate_leaderboard(task_name, RankingMetric.MEAN_REWARD)
        composite_lb = self.generate_leaderboard(task_name, RankingMetric.COMPOSITE_SCORE)

        # Generate comparison matrix
        comparison_matrix = self.generate_comparison_matrix(task_name)

        report = {
            "report_type": "policy_leaderboard_premium",
            "generated_at": datetime.now().isoformat(),
            "task_name": task_name,
            "summary": {
                "total_policies": len(self.policy_results),
                "total_episodes": sum(p.total_episodes for p in self.policy_results.values()),
                "best_policy_by_success": success_lb.rankings[0].policy_name if success_lb.rankings else None,
                "best_policy_by_reward": reward_lb.rankings[0].policy_name if reward_lb.rankings else None,
                "best_overall": composite_lb.rankings[0].policy_name if composite_lb.rankings else None
            },
            "leaderboards": {
                "success_rate": {
                    "rankings": [
                        {
                            "rank": r.rank,
                            "policy": r.policy_name,
                            "value": f"{r.metric_value * 100:.1f}%",
                            "ci_95": f"[{r.confidence_interval.lower_bound * 100:.1f}%, {r.confidence_interval.upper_bound * 100:.1f}%]",
                            "stability": f"{r.rank_stability * 100:.0f}%"
                        }
                        for r in success_lb.rankings
                    ]
                },
                "mean_reward": {
                    "rankings": [
                        {
                            "rank": r.rank,
                            "policy": r.policy_name,
                            "value": f"{r.metric_value:.2f}",
                            "ci_95": f"[{r.confidence_interval.lower_bound:.2f}, {r.confidence_interval.upper_bound:.2f}]",
                            "stability": f"{r.rank_stability * 100:.0f}%"
                        }
                        for r in reward_lb.rankings
                    ]
                },
                "composite": {
                    "rankings": [
                        {
                            "rank": r.rank,
                            "policy": r.policy_name,
                            "value": f"{r.metric_value:.3f}",
                            "stability": f"{r.rank_stability * 100:.0f}%"
                        }
                        for r in composite_lb.rankings
                    ]
                }
            },
            "significant_findings": [],
            "recommendations": [],
            "comparison_matrix": comparison_matrix
        }

        # Add significant findings
        for comparison in success_lb.pairwise_comparisons:
            if comparison.significant and abs(comparison.effect_size) >= 0.5:
                report["significant_findings"].append({
                    "finding": comparison.conclusion,
                    "effect_size": comparison.effect_size_interpretation,
                    "p_value": comparison.p_value
                })

        # Add recommendations
        if success_lb.rankings:
            top_policy = success_lb.rankings[0]
            report["recommendations"].append(
                f"Deploy {top_policy.policy_name} for production - highest success rate with {top_policy.rank_stability*100:.0f}% rank confidence"
            )

            if len(success_lb.rankings) > 1:
                runner_up = success_lb.rankings[1]
                if runner_up.rank_stability < 0.7:
                    report["recommendations"].append(
                        f"Consider additional evaluation for {runner_up.policy_name} - rank stability only {runner_up.rank_stability*100:.0f}%"
                    )

        report["upsell_opportunities"] = [
            "Run embodiment transfer analysis to deploy top policy on additional robots",
            "Generate sim2real fidelity matrix for real-world deployment confidence",
            "Perform failure mode analysis on lower-ranked policies to identify improvement areas",
            "Expand evaluation to additional task variations for generalization assessment"
        ]

        return report


def create_policy_leaderboard_generator(
    output_dir: str = "./leaderboards",
    confidence_level: float = 0.95
) -> PolicyLeaderboardGenerator:
    """Factory function to create PolicyLeaderboardGenerator instance."""
    return PolicyLeaderboardGenerator(output_dir=output_dir, confidence_level=confidence_level)
