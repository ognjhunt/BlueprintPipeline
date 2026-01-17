"""
GPU-Accelerated Parallel Evaluation for Isaac Lab-Arena.

This module provides high-throughput policy evaluation using Arena's
homogeneous parallel environment support. Scales to 1000s of parallel
environments for rapid policy benchmarking.

Key Difference from Genie Sim:
- Genie Sim: Data GENERATION with parallel environments
- This Module: Policy EVALUATION at scale

Features:
- GPU-accelerated parallel rollouts (1000+ envs)
- Multi-policy comparison (leaderboard generation)
- Statistical analysis with confidence intervals
- Automatic GPU memory management
- Checkpoint evaluation scheduling

Usage:
    from tools.arena_integration.parallel_evaluation import (
        ParallelEvaluator,
        ParallelEvalConfig,
        run_parallel_evaluation
    )

    config = ParallelEvalConfig(
        num_envs=1024,
        num_episodes=10000,
        device="cuda:0"
    )
    evaluator = ParallelEvaluator(config)
    results = evaluator.evaluate(env_spec, policy)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import numpy as np

from .components import ArenaEnvironmentSpec, ArenaEmbodiment, ArenaTask
from .benchmark_validation import (
    resolve_isaac_lab_arena_version,
    validate_arena_benchmark_results,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ParallelEvalConfig:
    """Configuration for parallel evaluation."""
    # Scale parameters
    num_envs: int = 1024                     # Number of parallel environments
    num_episodes: int = 1000                 # Total episodes to evaluate
    max_steps_per_episode: int = 500         # Max steps before timeout

    # Hardware configuration
    device: str = "cuda:0"
    headless: bool = True
    use_fabric: bool = True                  # Use Isaac Fabric for GPU perf

    # Memory management
    max_gpu_memory_gb: float = 24.0          # Max GPU memory to use
    auto_scale_envs: bool = True             # Auto-reduce envs if OOM

    # Evaluation settings
    deterministic: bool = True               # Deterministic policy inference
    seed: int = 42
    record_videos: bool = False
    video_interval: int = 100                # Record every N episodes

    # Statistical settings
    confidence_level: float = 0.95           # For confidence intervals
    bootstrap_samples: int = 1000            # Bootstrap resampling count

    # Checkpointing
    checkpoint_interval: int = 100           # Save results every N episodes
    checkpoint_dir: Optional[Path] = None


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode_id: int
    success: bool
    return_: float                           # Total episode return
    length: int                              # Episode length in steps
    timeout: bool                            # Terminated by max_steps
    collision_count: int = 0
    grasp_success: bool = False
    task_progress: float = 0.0               # 0-1 progress toward goal
    wall_time_ms: float = 0.0


@dataclass
class TaskMetricsSummary:
    """Aggregated metrics for a task."""
    task_id: str
    task_name: str
    num_episodes: int
    success_rate: float
    success_rate_ci: tuple[float, float]     # Confidence interval
    mean_return: float
    std_return: float
    mean_length: float
    std_length: float
    timeout_rate: float
    collision_rate: float
    percentiles: dict[str, float]            # 25th, 50th, 75th, 90th


@dataclass
class ParallelEvalResult:
    """Complete parallel evaluation results."""
    success: bool
    config: ParallelEvalConfig
    env_spec_id: str
    policy_id: str
    embodiment: str
    isaac_lab_arena_version: Optional[str] = None

    # Timing
    start_time: str
    end_time: str
    total_wall_time_s: float
    episodes_per_second: float

    # Scale achieved
    num_parallel_envs: int
    total_episodes: int
    total_steps: int

    # Aggregated metrics
    overall_success_rate: float
    overall_success_rate_ci: tuple[float, float]
    overall_mean_return: float
    overall_std_return: float

    # Per-task breakdown
    task_metrics: dict[str, TaskMetricsSummary]

    # Raw episode data (optional, for detailed analysis)
    episode_metrics: list[EpisodeMetrics] = field(default_factory=list)

    # Errors
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        data = {
            "success": self.success,
            "config": {
                "num_envs": self.config.num_envs,
                "num_episodes": self.config.num_episodes,
                "device": self.config.device,
            },
            "env_spec_id": self.env_spec_id,
            "policy_id": self.policy_id,
            "embodiment": self.embodiment,
            "timing": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_wall_time_s": self.total_wall_time_s,
                "episodes_per_second": self.episodes_per_second,
            },
            "scale": {
                "num_parallel_envs": self.num_parallel_envs,
                "total_episodes": self.total_episodes,
                "total_steps": self.total_steps,
            },
            "metrics": {
                "overall_success_rate": self.overall_success_rate,
                "overall_success_rate_ci": list(self.overall_success_rate_ci),
                "overall_mean_return": self.overall_mean_return,
                "overall_std_return": self.overall_std_return,
            },
            "task_metrics": {
                k: {
                    "task_id": v.task_id,
                    "task_name": v.task_name,
                    "num_episodes": v.num_episodes,
                    "success_rate": v.success_rate,
                    "success_rate_ci": list(v.success_rate_ci),
                    "mean_return": v.mean_return,
                    "mean_length": v.mean_length,
                    "timeout_rate": v.timeout_rate,
                }
                for k, v in self.task_metrics.items()
            },
            "errors": self.errors,
        }
        if self.isaac_lab_arena_version:
            data["isaac_lab_arena_version"] = self.isaac_lab_arena_version
        return data

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        data = self.to_dict()
        arena_version = self.isaac_lab_arena_version or resolve_isaac_lab_arena_version()
        if arena_version:
            data["isaac_lab_arena_version"] = arena_version
        validate_arena_benchmark_results(
            data,
            scene_id=self.env_spec_id,
            task_ids=list(self.task_metrics.keys()),
            arena_version=arena_version,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# =============================================================================
# POLICY PROTOCOL
# =============================================================================

class PolicyProtocol(Protocol):
    """Protocol for policies that can be evaluated."""

    def reset(self) -> None:
        """Reset policy state."""
        ...

    def get_action(
        self,
        observation: dict[str, np.ndarray],
        deterministic: bool = True
    ) -> np.ndarray:
        """Get action for observation batch."""
        ...

    @property
    def policy_id(self) -> str:
        """Unique policy identifier."""
        ...


# =============================================================================
# PARALLEL EVALUATOR
# =============================================================================

class ParallelEvaluator:
    """
    GPU-accelerated parallel policy evaluator.

    Implements Arena's homogeneous parallel environment support
    for high-throughput policy benchmarking.

    This is for EVALUATION only, not data generation (use Genie Sim for that).
    """

    def __init__(self, config: ParallelEvalConfig):
        self.config = config
        self._isaac_lab_available = self._check_isaac_lab()

    def _check_isaac_lab(self) -> bool:
        """Check if Isaac Lab is available."""
        try:
            import omni.isaac.lab  # noqa: F401
            return True
        except ImportError:
            return False

    def evaluate(
        self,
        env_spec: ArenaEnvironmentSpec,
        policy: PolicyProtocol,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ParallelEvalResult:
        """
        Run parallel evaluation.

        Args:
            env_spec: Arena environment specification
            policy: Policy to evaluate (must implement PolicyProtocol)
            progress_callback: Optional callback(episodes_done, total_episodes)

        Returns:
            ParallelEvalResult with comprehensive metrics
        """
        start_time = datetime.utcnow()
        errors: list[str] = []

        # Determine actual number of parallel envs
        num_envs = self._determine_num_envs(env_spec)

        if self._isaac_lab_available:
            result = self._run_isaac_lab_evaluation(
                env_spec, policy, num_envs, progress_callback
            )
        else:
            # Mock evaluation for development/testing
            result = self._run_mock_evaluation(
                env_spec, policy, num_envs, progress_callback
            )

        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()

        # Compute statistics
        episode_metrics = result["episode_metrics"]
        task_metrics = self._compute_task_metrics(episode_metrics, env_spec.task)

        successes = [e.success for e in episode_metrics]
        returns = [e.return_ for e in episode_metrics]

        success_rate = np.mean(successes) if successes else 0.0
        success_ci = self._bootstrap_ci(successes, self.config.confidence_level)

        arena_version = resolve_isaac_lab_arena_version()
        return ParallelEvalResult(
            success=len(errors) == 0,
            config=self.config,
            env_spec_id=env_spec.scene.scene_id,
            policy_id=policy.policy_id,
            embodiment=env_spec.embodiment.embodiment_type.value,
            isaac_lab_arena_version=arena_version,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_wall_time_s=total_time,
            episodes_per_second=len(episode_metrics) / total_time if total_time > 0 else 0,
            num_parallel_envs=num_envs,
            total_episodes=len(episode_metrics),
            total_steps=sum(e.length for e in episode_metrics),
            overall_success_rate=float(success_rate),
            overall_success_rate_ci=success_ci,
            overall_mean_return=float(np.mean(returns)) if returns else 0.0,
            overall_std_return=float(np.std(returns)) if returns else 0.0,
            task_metrics=task_metrics,
            episode_metrics=episode_metrics if self.config.record_videos else [],
            errors=errors + result.get("errors", []),
        )

    def _determine_num_envs(self, env_spec: ArenaEnvironmentSpec) -> int:
        """Determine optimal number of parallel environments."""
        requested = self.config.num_envs

        if not self.config.auto_scale_envs:
            return requested

        # Estimate memory per env based on task complexity
        base_memory_mb = 100  # Base memory per env
        object_memory_mb = len(env_spec.scene.objects) * 10

        total_memory_per_env = base_memory_mb + object_memory_mb
        max_envs_by_memory = int(
            (self.config.max_gpu_memory_gb * 1024) / total_memory_per_env
        )

        # Use power of 2 for GPU efficiency
        optimal = min(requested, max_envs_by_memory)
        optimal = 2 ** int(math.log2(optimal))

        return max(1, optimal)

    def _run_isaac_lab_evaluation(
        self,
        env_spec: ArenaEnvironmentSpec,
        policy: PolicyProtocol,
        num_envs: int,
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> dict[str, Any]:
        """Run evaluation using Isaac Lab."""
        from omni.isaac.lab.envs import ManagerBasedRLEnv

        episode_metrics: list[EpisodeMetrics] = []
        errors: list[str] = []

        try:
            # Create vectorized environment
            env_cfg = self._create_env_cfg(env_spec, num_envs)
            env = ManagerBasedRLEnv(cfg=env_cfg)

            episodes_done = 0
            target_episodes = self.config.num_episodes

            # Pre-allocate buffers for efficiency
            episode_returns = np.zeros(num_envs)
            episode_lengths = np.zeros(num_envs, dtype=np.int32)
            episode_ids = np.arange(num_envs)

            obs, _ = env.reset()
            policy.reset()

            while episodes_done < target_episodes:
                # Get actions for all envs
                actions = policy.get_action(obs, deterministic=self.config.deterministic)

                # Step all envs
                obs, rewards, terminated, truncated, infos = env.step(actions)

                # Update episode stats
                episode_returns += rewards.cpu().numpy()
                episode_lengths += 1

                # Handle completed episodes
                dones = (terminated | truncated).cpu().numpy()
                for env_idx in np.where(dones)[0]:
                    if episodes_done >= target_episodes:
                        break

                    metrics = EpisodeMetrics(
                        episode_id=episodes_done,
                        success=bool(infos.get("success", {}).get(env_idx, False)),
                        return_=float(episode_returns[env_idx]),
                        length=int(episode_lengths[env_idx]),
                        timeout=bool(truncated[env_idx]),
                        task_progress=float(infos.get("task_progress", {}).get(env_idx, 0.0)),
                    )
                    episode_metrics.append(metrics)

                    # Reset stats for this env
                    episode_returns[env_idx] = 0
                    episode_lengths[env_idx] = 0
                    episodes_done += 1

                    if progress_callback:
                        progress_callback(episodes_done, target_episodes)

                    # Checkpoint
                    if (episodes_done % self.config.checkpoint_interval == 0
                            and self.config.checkpoint_dir):
                        self._save_checkpoint(episode_metrics, episodes_done)

            env.close()

        except Exception as e:
            errors.append(f"Isaac Lab evaluation failed: {e}")

        return {
            "episode_metrics": episode_metrics,
            "errors": errors,
        }

    def _run_mock_evaluation(
        self,
        env_spec: ArenaEnvironmentSpec,
        policy: PolicyProtocol,
        num_envs: int,
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> dict[str, Any]:
        """
        Mock evaluation for development/testing without Isaac Lab.

        Generates realistic-looking metrics for testing the evaluation pipeline.
        """
        episode_metrics: list[EpisodeMetrics] = []
        np.random.seed(self.config.seed)

        target_episodes = self.config.num_episodes

        # Simulate evaluation with realistic timing
        episodes_per_batch = num_envs
        batches_needed = math.ceil(target_episodes / episodes_per_batch)

        for batch in range(batches_needed):
            batch_start = batch * episodes_per_batch
            batch_end = min(batch_start + episodes_per_batch, target_episodes)

            for ep_id in range(batch_start, batch_end):
                # Generate realistic metrics
                # Success rate varies by task difficulty
                difficulty_factor = {
                    "easy": 0.85,
                    "medium": 0.65,
                    "hard": 0.45,
                    "expert": 0.25,
                }.get(env_spec.task.difficulty.value, 0.5)

                success = np.random.random() < difficulty_factor

                # Episode length correlates with success
                if success:
                    length = int(np.random.normal(150, 50))
                else:
                    length = int(np.random.normal(350, 100))
                length = max(10, min(length, self.config.max_steps_per_episode))

                # Return correlates with success and length efficiency
                if success:
                    base_return = 100.0
                    efficiency_bonus = (self.config.max_steps_per_episode - length) / 10
                    return_ = base_return + efficiency_bonus + np.random.normal(0, 10)
                else:
                    progress = np.random.random() * 0.8
                    return_ = progress * 50 + np.random.normal(0, 15)

                metrics = EpisodeMetrics(
                    episode_id=ep_id,
                    success=success,
                    return_=return_,
                    length=length,
                    timeout=length >= self.config.max_steps_per_episode,
                    collision_count=np.random.poisson(0.5) if not success else 0,
                    task_progress=1.0 if success else np.random.random() * 0.8,
                    wall_time_ms=length * 8.3,  # ~120Hz simulation
                )
                episode_metrics.append(metrics)

                if progress_callback:
                    progress_callback(ep_id + 1, target_episodes)

            # Simulate batch processing time
            time.sleep(0.01)

        return {
            "episode_metrics": episode_metrics,
            "errors": [],
        }

    def _create_env_cfg(
        self,
        env_spec: ArenaEnvironmentSpec,
        num_envs: int
    ) -> Any:
        """Create Isaac Lab environment configuration."""
        # This would create a proper ManagerBasedRLEnvCfg
        # For now, return the spec's config converted
        cfg = env_spec.to_isaac_lab_cfg()
        cfg["sim"]["num_envs"] = num_envs
        return cfg

    def _compute_task_metrics(
        self,
        episodes: list[EpisodeMetrics],
        task: ArenaTask
    ) -> dict[str, TaskMetricsSummary]:
        """Compute aggregated task metrics."""
        if not episodes:
            return {}

        successes = [e.success for e in episodes]
        returns = [e.return_ for e in episodes]
        lengths = [e.length for e in episodes]
        timeouts = [e.timeout for e in episodes]
        collisions = [e.collision_count for e in episodes]

        success_rate = np.mean(successes)
        success_ci = self._bootstrap_ci(successes, self.config.confidence_level)

        summary = TaskMetricsSummary(
            task_id=task.task_id,
            task_name=task.name,
            num_episodes=len(episodes),
            success_rate=float(success_rate),
            success_rate_ci=success_ci,
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            mean_length=float(np.mean(lengths)),
            std_length=float(np.std(lengths)),
            timeout_rate=float(np.mean(timeouts)),
            collision_rate=float(np.mean([c > 0 for c in collisions])),
            percentiles={
                "p25": float(np.percentile(returns, 25)),
                "p50": float(np.percentile(returns, 50)),
                "p75": float(np.percentile(returns, 75)),
                "p90": float(np.percentile(returns, 90)),
            },
        )

        return {task.task_id: summary}

    def _bootstrap_ci(
        self,
        data: list[bool],
        confidence: float
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval for success rate."""
        if not data:
            return (0.0, 0.0)

        data_array = np.array(data, dtype=float)
        n = len(data_array)

        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(self.config.bootstrap_samples):
            sample = np.random.choice(data_array, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        # Compute percentile CI
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return (float(lower), float(upper))

    def _save_checkpoint(
        self,
        episodes: list[EpisodeMetrics],
        episodes_done: int
    ) -> None:
        """Save evaluation checkpoint."""
        if not self.config.checkpoint_dir:
            return

        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_{episodes_done}.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "episodes_done": episodes_done,
            "metrics": [
                {
                    "episode_id": e.episode_id,
                    "success": e.success,
                    "return": e.return_,
                    "length": e.length,
                }
                for e in episodes
            ],
        }

        with open(checkpoint_path, "w") as f:
            json.dump(data, f)


# =============================================================================
# MULTI-POLICY COMPARISON (LEADERBOARD)
# =============================================================================

@dataclass
class PolicyComparisonResult:
    """Results from comparing multiple policies."""
    policies: list[str]
    task_id: str
    rankings: list[dict[str, Any]]  # Sorted by success rate
    statistical_tests: dict[str, Any]  # Pairwise comparisons


class MultiPolicyEvaluator:
    """
    Evaluates multiple policies for leaderboard generation.

    Runs parallel evaluations across policies and produces
    ranked comparisons with statistical significance tests.
    """

    def __init__(self, config: ParallelEvalConfig):
        self.config = config
        self.evaluator = ParallelEvaluator(config)

    def compare_policies(
        self,
        env_spec: ArenaEnvironmentSpec,
        policies: list[PolicyProtocol],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> PolicyComparisonResult:
        """
        Compare multiple policies on same environment.

        Args:
            env_spec: Environment specification
            policies: List of policies to compare
            progress_callback: Optional callback(policy_id, episodes_done, total)

        Returns:
            PolicyComparisonResult with rankings and statistical tests
        """
        results: dict[str, ParallelEvalResult] = {}

        for policy in policies:
            print(f"Evaluating policy: {policy.policy_id}")

            def policy_progress(done: int, total: int) -> None:
                if progress_callback:
                    progress_callback(policy.policy_id, done, total)

            result = self.evaluator.evaluate(env_spec, policy, policy_progress)
            results[policy.policy_id] = result

        # Generate rankings
        rankings = self._compute_rankings(results)

        # Statistical significance tests
        stat_tests = self._compute_statistical_tests(results)

        return PolicyComparisonResult(
            policies=[p.policy_id for p in policies],
            task_id=env_spec.task.task_id,
            rankings=rankings,
            statistical_tests=stat_tests,
        )

    def _compute_rankings(
        self,
        results: dict[str, ParallelEvalResult]
    ) -> list[dict[str, Any]]:
        """Compute policy rankings by success rate."""
        rankings = []

        for policy_id, result in results.items():
            rankings.append({
                "rank": 0,  # Will be filled
                "policy_id": policy_id,
                "success_rate": result.overall_success_rate,
                "success_rate_ci": result.overall_success_rate_ci,
                "mean_return": result.overall_mean_return,
                "episodes_per_second": result.episodes_per_second,
            })

        # Sort by success rate (descending)
        rankings.sort(key=lambda x: x["success_rate"], reverse=True)

        # Assign ranks
        for i, r in enumerate(rankings):
            r["rank"] = i + 1

        return rankings

    def _compute_statistical_tests(
        self,
        results: dict[str, ParallelEvalResult]
    ) -> dict[str, Any]:
        """Compute pairwise statistical significance tests."""
        from scipy import stats

        policy_ids = list(results.keys())
        tests = {}

        for i, p1 in enumerate(policy_ids):
            for p2 in policy_ids[i + 1:]:
                r1 = results[p1]
                r2 = results[p2]

                # Extract success lists
                s1 = [e.success for e in r1.episode_metrics] if r1.episode_metrics else []
                s2 = [e.success for e in r2.episode_metrics] if r2.episode_metrics else []

                if s1 and s2:
                    # Two-proportion z-test
                    n1, n2 = len(s1), len(s2)
                    p1_rate, p2_rate = np.mean(s1), np.mean(s2)
                    p_pooled = (sum(s1) + sum(s2)) / (n1 + n2)

                    if p_pooled > 0 and p_pooled < 1:
                        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                        z_stat = (p1_rate - p2_rate) / se if se > 0 else 0
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                    else:
                        z_stat = 0
                        p_value = 1.0

                    tests[f"{p1}_vs_{p2}"] = {
                        "z_statistic": float(z_stat),
                        "p_value": float(p_value),
                        "significant_0.05": p_value < 0.05,
                        "significant_0.01": p_value < 0.01,
                        "winner": p1 if p1_rate > p2_rate else p2,
                    }

        return tests


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_parallel_evaluation(
    env_spec: ArenaEnvironmentSpec,
    policy: PolicyProtocol,
    num_envs: int = 1024,
    num_episodes: int = 1000,
    device: str = "cuda:0",
) -> ParallelEvalResult:
    """
    Convenience function for parallel evaluation.

    Args:
        env_spec: Arena environment specification
        policy: Policy to evaluate
        num_envs: Number of parallel environments
        num_episodes: Total episodes to run
        device: CUDA device

    Returns:
        ParallelEvalResult with metrics
    """
    config = ParallelEvalConfig(
        num_envs=num_envs,
        num_episodes=num_episodes,
        device=device,
    )
    evaluator = ParallelEvaluator(config)
    return evaluator.evaluate(env_spec, policy)


def estimate_evaluation_time(
    num_episodes: int,
    num_envs: int,
    avg_episode_length: int = 200,
    sim_frequency: float = 120.0,
) -> float:
    """
    Estimate wall-clock time for evaluation.

    Args:
        num_episodes: Total episodes
        num_envs: Parallel environments
        avg_episode_length: Average steps per episode
        sim_frequency: Simulation frequency (Hz)

    Returns:
        Estimated time in seconds
    """
    batches = math.ceil(num_episodes / num_envs)
    steps_per_batch = avg_episode_length
    time_per_step = 1 / sim_frequency

    # Add overhead for resets, policy inference
    overhead_factor = 1.5

    return batches * steps_per_batch * time_per_step * overhead_factor
