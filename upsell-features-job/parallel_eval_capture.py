"""
Parallel Evaluation Results Capture Module
===========================================

Captures GPU-accelerated parallel evaluation metrics from Isaac Lab Arena
that are currently NOT captured in the standard pipeline.

Premium Analytics Feature - Upsell Value: $25,000 - $50,000

Features:
- 1000+ environment parallel evaluation tracking
- GPU utilization and throughput metrics
- Cross-environment variance analysis
- Evaluation reproducibility assessment
- Hardware efficiency benchmarking

Author: BlueprintPipeline Premium Analytics
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
from pathlib import Path
import statistics
import hashlib


class GPUBackend(Enum):
    """GPU compute backends supported."""
    CUDA = "cuda"
    VULKAN = "vulkan"
    CPU = "cpu"  # Fallback


class ParallelizationStrategy(Enum):
    """How environments are distributed across GPUs."""
    SINGLE_GPU = "single_gpu"
    MULTI_GPU_SPLIT = "multi_gpu_split"
    MULTI_GPU_REPLICATED = "multi_gpu_replicated"
    DISTRIBUTED = "distributed"


class EvaluationMode(Enum):
    """Type of evaluation being performed."""
    POLICY_BENCHMARK = "policy_benchmark"
    ABLATION_STUDY = "ablation_study"
    HYPERPARAMETER_SWEEP = "hyperparameter_sweep"
    GENERALIZATION_TEST = "generalization_test"
    STRESS_TEST = "stress_test"


@dataclass
class GPUMetrics:
    """GPU utilization metrics during evaluation."""
    gpu_id: int
    gpu_name: str
    memory_total_gb: float
    memory_used_gb: float
    memory_utilization: float  # 0-1
    compute_utilization: float  # 0-1
    temperature_celsius: float
    power_watts: float
    environments_hosted: int


@dataclass
class EnvironmentInstance:
    """Metrics for a single parallel environment instance."""
    env_idx: int
    gpu_id: int
    seed: int

    # Episode statistics
    episodes_completed: int = 0
    total_steps: int = 0
    successful_episodes: int = 0

    # Timing
    total_sim_time: float = 0.0
    total_wall_time: float = 0.0
    avg_step_time_ms: float = 0.0

    # Per-episode data
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_outcomes: List[bool] = field(default_factory=list)

    # Variance tracking
    reward_variance: float = 0.0
    length_variance: float = 0.0


@dataclass
class ParallelEvalConfig:
    """Configuration for parallel evaluation run."""
    config_id: str
    policy_id: str
    task_name: str

    # Parallelization
    num_environments: int
    num_gpus: int
    envs_per_gpu: int
    strategy: ParallelizationStrategy

    # Evaluation parameters
    episodes_per_env: int
    max_episode_steps: int
    evaluation_mode: EvaluationMode

    # Seeds for reproducibility
    base_seed: int
    env_seeds: List[int] = field(default_factory=list)

    # Scene variations
    pose_variations: int = 1
    lighting_variations: int = 1
    object_variations: int = 1


@dataclass
class ParallelEvalResults:
    """Complete results from parallel evaluation run."""
    results_id: str
    config: ParallelEvalConfig

    # Timing
    start_time: datetime
    end_time: datetime
    total_wall_time: float
    total_sim_time: float

    # Environment instances
    environments: List[EnvironmentInstance] = field(default_factory=list)

    # GPU metrics (sampled during evaluation)
    gpu_metrics_samples: List[List[GPUMetrics]] = field(default_factory=list)

    # Aggregate statistics
    total_episodes: int = 0
    successful_episodes: int = 0
    success_rate: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_episode_length: float = 0.0
    std_episode_length: float = 0.0

    # Throughput
    episodes_per_second: float = 0.0
    steps_per_second: float = 0.0
    sim_to_real_ratio: float = 0.0  # Simulation speedup factor

    # Cross-environment statistics
    inter_env_reward_variance: float = 0.0
    inter_env_success_variance: float = 0.0
    reproducibility_score: float = 0.0

    # Hardware efficiency
    gpu_utilization_mean: float = 0.0
    gpu_memory_efficiency: float = 0.0
    power_efficiency: float = 0.0  # Episodes per watt-hour


class ParallelEvalCapture:
    """
    Captures and analyzes parallel evaluation results from Isaac Lab Arena.

    This module fills the gap of GPU-accelerated parallel benchmark data
    that is NOT captured in the standard pipeline output.
    """

    def __init__(self, output_dir: str = "./parallel_eval"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_config: Optional[ParallelEvalConfig] = None
        self.current_results: Optional[ParallelEvalResults] = None
        self.gpu_sample_interval: float = 1.0  # seconds

    def initialize_evaluation(
        self,
        policy_id: str,
        task_name: str,
        num_environments: int = 1024,
        num_gpus: int = 1,
        episodes_per_env: int = 10,
        max_episode_steps: int = 500,
        evaluation_mode: EvaluationMode = EvaluationMode.POLICY_BENCHMARK,
        base_seed: int = 42,
        pose_variations: int = 1,
        lighting_variations: int = 1,
        object_variations: int = 1
    ) -> str:
        """Initialize a new parallel evaluation run."""
        config_id = hashlib.sha256(
            f"{policy_id}_{task_name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        envs_per_gpu = num_environments // num_gpus if num_gpus > 0 else num_environments

        strategy = ParallelizationStrategy.SINGLE_GPU
        if num_gpus > 1:
            strategy = ParallelizationStrategy.MULTI_GPU_SPLIT

        # Generate deterministic seeds for each environment
        np.random.seed(base_seed)
        env_seeds = list(np.random.randint(0, 2**31, size=num_environments))

        self.current_config = ParallelEvalConfig(
            config_id=config_id,
            policy_id=policy_id,
            task_name=task_name,
            num_environments=num_environments,
            num_gpus=num_gpus,
            envs_per_gpu=envs_per_gpu,
            strategy=strategy,
            episodes_per_env=episodes_per_env,
            max_episode_steps=max_episode_steps,
            evaluation_mode=evaluation_mode,
            base_seed=base_seed,
            env_seeds=env_seeds,
            pose_variations=pose_variations,
            lighting_variations=lighting_variations,
            object_variations=object_variations
        )

        results_id = f"{config_id}_results"

        self.current_results = ParallelEvalResults(
            results_id=results_id,
            config=self.current_config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_wall_time=0.0,
            total_sim_time=0.0
        )

        # Initialize environment instances
        for env_idx in range(num_environments):
            gpu_id = env_idx // envs_per_gpu if envs_per_gpu > 0 else 0
            env = EnvironmentInstance(
                env_idx=env_idx,
                gpu_id=gpu_id,
                seed=env_seeds[env_idx]
            )
            self.current_results.environments.append(env)

        return config_id

    def record_gpu_metrics(
        self,
        gpu_metrics: List[Dict[str, Any]]
    ) -> None:
        """Record a sample of GPU metrics during evaluation."""
        if not self.current_results:
            return

        metrics = []
        for gm in gpu_metrics:
            metrics.append(GPUMetrics(
                gpu_id=gm.get("gpu_id", 0),
                gpu_name=gm.get("gpu_name", "Unknown"),
                memory_total_gb=gm.get("memory_total_gb", 0.0),
                memory_used_gb=gm.get("memory_used_gb", 0.0),
                memory_utilization=gm.get("memory_utilization", 0.0),
                compute_utilization=gm.get("compute_utilization", 0.0),
                temperature_celsius=gm.get("temperature_celsius", 0.0),
                power_watts=gm.get("power_watts", 0.0),
                environments_hosted=gm.get("environments_hosted", 0)
            ))

        self.current_results.gpu_metrics_samples.append(metrics)

    def record_episode_result(
        self,
        env_idx: int,
        reward: float,
        length: int,
        success: bool,
        sim_time: float,
        wall_time: float
    ) -> None:
        """Record result from a single episode in a parallel environment."""
        if not self.current_results:
            return

        if env_idx >= len(self.current_results.environments):
            return

        env = self.current_results.environments[env_idx]
        env.episodes_completed += 1
        env.total_steps += length
        env.episode_rewards.append(reward)
        env.episode_lengths.append(length)
        env.episode_outcomes.append(success)
        env.total_sim_time += sim_time
        env.total_wall_time += wall_time

        if success:
            env.successful_episodes += 1

    def record_batch_results(
        self,
        batch_results: List[Dict[str, Any]]
    ) -> None:
        """Record results from a batch of parallel episodes."""
        for result in batch_results:
            self.record_episode_result(
                env_idx=result["env_idx"],
                reward=result["reward"],
                length=result["length"],
                success=result["success"],
                sim_time=result.get("sim_time", 0.0),
                wall_time=result.get("wall_time", 0.0)
            )

    def finalize_evaluation(self) -> ParallelEvalResults:
        """Complete evaluation and compute all aggregate statistics."""
        if not self.current_results:
            raise ValueError("No active evaluation")

        results = self.current_results
        results.end_time = datetime.now()
        results.total_wall_time = (results.end_time - results.start_time).total_seconds()

        # Aggregate from environments
        all_rewards = []
        all_lengths = []
        all_outcomes = []

        for env in results.environments:
            all_rewards.extend(env.episode_rewards)
            all_lengths.extend(env.episode_lengths)
            all_outcomes.extend(env.episode_outcomes)
            results.total_sim_time += env.total_sim_time

            # Compute per-environment variance
            if len(env.episode_rewards) > 1:
                env.reward_variance = statistics.variance(env.episode_rewards)
                env.length_variance = statistics.variance([float(x) for x in env.episode_lengths])

            if env.total_steps > 0:
                env.avg_step_time_ms = (env.total_wall_time * 1000) / env.total_steps

        results.total_episodes = len(all_rewards)
        results.successful_episodes = sum(all_outcomes)
        results.success_rate = results.successful_episodes / results.total_episodes if results.total_episodes > 0 else 0.0

        if all_rewards:
            results.mean_reward = statistics.mean(all_rewards)
            results.std_reward = statistics.stdev(all_rewards) if len(all_rewards) > 1 else 0.0

        if all_lengths:
            results.mean_episode_length = statistics.mean(all_lengths)
            results.std_episode_length = statistics.stdev(all_lengths) if len(all_lengths) > 1 else 0.0

        # Throughput
        if results.total_wall_time > 0:
            results.episodes_per_second = results.total_episodes / results.total_wall_time
            total_steps = sum(env.total_steps for env in results.environments)
            results.steps_per_second = total_steps / results.total_wall_time

        if results.total_wall_time > 0 and results.total_sim_time > 0:
            results.sim_to_real_ratio = results.total_sim_time / results.total_wall_time

        # Cross-environment analysis
        self._compute_cross_environment_statistics(results)

        # GPU efficiency
        self._compute_gpu_efficiency(results)

        self.current_results = None
        self.current_config = None

        return results

    def _compute_cross_environment_statistics(
        self,
        results: ParallelEvalResults
    ) -> None:
        """Compute variance across parallel environments."""
        env_mean_rewards = []
        env_success_rates = []

        for env in results.environments:
            if env.episodes_completed > 0:
                env_mean_rewards.append(statistics.mean(env.episode_rewards) if env.episode_rewards else 0.0)
                env_success_rates.append(env.successful_episodes / env.episodes_completed)

        if len(env_mean_rewards) > 1:
            results.inter_env_reward_variance = statistics.variance(env_mean_rewards)
            results.inter_env_success_variance = statistics.variance(env_success_rates)

            # Reproducibility score: lower variance = higher reproducibility
            # Normalized to 0-1 range where 1 = perfect reproducibility
            max_expected_variance = 0.25  # Expected variance for random outcomes
            reward_cv = results.inter_env_reward_variance / (results.mean_reward ** 2) if results.mean_reward != 0 else 0
            results.reproducibility_score = max(0.0, 1.0 - (reward_cv / max_expected_variance))

    def _compute_gpu_efficiency(
        self,
        results: ParallelEvalResults
    ) -> None:
        """Compute GPU efficiency metrics."""
        if not results.gpu_metrics_samples:
            return

        utilizations = []
        memory_efficiencies = []
        total_power = 0.0
        sample_count = 0

        for sample in results.gpu_metrics_samples:
            for gpu in sample:
                utilizations.append(gpu.compute_utilization)
                if gpu.memory_total_gb > 0:
                    memory_efficiencies.append(gpu.memory_used_gb / gpu.memory_total_gb)
                total_power += gpu.power_watts
                sample_count += 1

        if utilizations:
            results.gpu_utilization_mean = statistics.mean(utilizations)

        if memory_efficiencies:
            results.gpu_memory_efficiency = statistics.mean(memory_efficiencies)

        # Power efficiency: episodes per watt-hour
        if sample_count > 0 and results.total_wall_time > 0:
            avg_power = total_power / sample_count
            watt_hours = avg_power * (results.total_wall_time / 3600)
            if watt_hours > 0:
                results.power_efficiency = results.total_episodes / watt_hours

    def compute_environment_correlation_matrix(
        self,
        results: ParallelEvalResults
    ) -> Dict[str, Any]:
        """
        Compute correlation matrix across parallel environments.

        UPSELL VALUE: Identifies if certain env configurations systematically
        produce different results.
        """
        n_envs = min(100, len(results.environments))  # Limit for visualization

        # Build reward matrix (episodes x environments)
        max_episodes = max(len(env.episode_rewards) for env in results.environments[:n_envs])
        reward_matrix = np.zeros((max_episodes, n_envs))
        reward_matrix[:] = np.nan

        for i, env in enumerate(results.environments[:n_envs]):
            for j, reward in enumerate(env.episode_rewards):
                if j < max_episodes:
                    reward_matrix[j, i] = reward

        # Compute correlation matrix (environment x environment)
        valid_envs = []
        for i in range(n_envs):
            if not np.all(np.isnan(reward_matrix[:, i])):
                valid_envs.append(i)

        if len(valid_envs) < 2:
            return {"error": "Not enough valid environments for correlation analysis"}

        # Use numpy for correlation computation
        valid_matrix = reward_matrix[:, valid_envs]

        # Handle missing values by using pairwise complete observations
        n_valid = len(valid_envs)
        correlation_matrix = np.zeros((n_valid, n_valid))

        for i in range(n_valid):
            for j in range(n_valid):
                mask = ~(np.isnan(valid_matrix[:, i]) | np.isnan(valid_matrix[:, j]))
                if np.sum(mask) > 2:
                    x = valid_matrix[mask, i]
                    y = valid_matrix[mask, j]
                    if np.std(x) > 0 and np.std(y) > 0:
                        correlation_matrix[i, j] = np.corrcoef(x, y)[0, 1]
                    else:
                        correlation_matrix[i, j] = 1.0 if i == j else 0.0
                else:
                    correlation_matrix[i, j] = np.nan

        # Identify clusters of correlated environments
        clusters = self._identify_env_clusters(correlation_matrix, valid_envs)

        return {
            "num_environments_analyzed": n_valid,
            "mean_correlation": float(np.nanmean(correlation_matrix[np.triu_indices(n_valid, k=1)])),
            "min_correlation": float(np.nanmin(correlation_matrix[np.triu_indices(n_valid, k=1)])),
            "max_correlation": float(np.nanmax(correlation_matrix[np.triu_indices(n_valid, k=1)])),
            "correlation_matrix": correlation_matrix.tolist(),
            "environment_indices": valid_envs,
            "clusters": clusters,
            "interpretation": self._interpret_correlation_results(
                np.nanmean(correlation_matrix[np.triu_indices(n_valid, k=1)])
            )
        }

    def _identify_env_clusters(
        self,
        correlation_matrix: np.ndarray,
        env_indices: List[int],
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Identify clusters of highly correlated environments."""
        n = len(env_indices)
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)

            for j in range(i + 1, n):
                if j in visited:
                    continue
                if not np.isnan(correlation_matrix[i, j]) and correlation_matrix[i, j] >= threshold:
                    cluster.append(j)
                    visited.add(j)

            if len(cluster) > 1:
                clusters.append({
                    "env_indices": [env_indices[idx] for idx in cluster],
                    "size": len(cluster),
                    "mean_internal_correlation": float(np.nanmean([
                        correlation_matrix[a, b]
                        for a in cluster for b in cluster if a < b
                    ]))
                })

        return clusters

    def _interpret_correlation_results(self, mean_corr: float) -> str:
        """Interpret correlation analysis results."""
        if np.isnan(mean_corr):
            return "Insufficient data for correlation analysis"
        elif mean_corr > 0.9:
            return "Very high correlation - environments are highly consistent, evaluation is reliable"
        elif mean_corr > 0.7:
            return "Good correlation - reasonable consistency across environments"
        elif mean_corr > 0.5:
            return "Moderate correlation - some variability between environments, consider investigation"
        elif mean_corr > 0.3:
            return "Low correlation - significant variability, may indicate seed sensitivity or bugs"
        else:
            return "Very low correlation - results may not be reproducible, investigate environment setup"

    def compute_throughput_analysis(
        self,
        results: ParallelEvalResults
    ) -> Dict[str, Any]:
        """
        Analyze evaluation throughput and scaling efficiency.

        UPSELL VALUE: Hardware efficiency metrics for optimizing evaluation costs.
        """
        analysis = {
            "results_id": results.results_id,
            "configuration": {
                "num_environments": results.config.num_environments,
                "num_gpus": results.config.num_gpus,
                "envs_per_gpu": results.config.envs_per_gpu,
                "parallelization_strategy": results.config.strategy.value
            },
            "throughput_metrics": {
                "total_episodes": results.total_episodes,
                "total_steps": sum(env.total_steps for env in results.environments),
                "wall_clock_time_seconds": results.total_wall_time,
                "simulation_time_seconds": results.total_sim_time,
                "episodes_per_second": results.episodes_per_second,
                "steps_per_second": results.steps_per_second,
                "realtime_factor": results.sim_to_real_ratio
            },
            "per_environment_metrics": {
                "avg_episodes_per_env": results.total_episodes / results.config.num_environments if results.config.num_environments > 0 else 0,
                "avg_step_time_ms": 0.0,
                "step_time_variance": 0.0
            },
            "gpu_efficiency": {
                "mean_utilization": results.gpu_utilization_mean,
                "memory_efficiency": results.gpu_memory_efficiency,
                "power_efficiency_eps_per_wh": results.power_efficiency
            },
            "scaling_analysis": {},
            "recommendations": []
        }

        # Per-environment step timing
        step_times = [env.avg_step_time_ms for env in results.environments if env.avg_step_time_ms > 0]
        if step_times:
            analysis["per_environment_metrics"]["avg_step_time_ms"] = statistics.mean(step_times)
            if len(step_times) > 1:
                analysis["per_environment_metrics"]["step_time_variance"] = statistics.variance(step_times)

        # Scaling analysis
        theoretical_speedup = results.config.num_environments
        if results.config.num_environments > 1:
            # Estimate single-env throughput from per-env timing
            single_env_eps = 1.0 / (results.mean_episode_length * analysis["per_environment_metrics"]["avg_step_time_ms"] / 1000) if analysis["per_environment_metrics"]["avg_step_time_ms"] > 0 else 1.0
            actual_speedup = results.episodes_per_second / single_env_eps if single_env_eps > 0 else 1.0

            analysis["scaling_analysis"] = {
                "theoretical_speedup": theoretical_speedup,
                "actual_speedup": actual_speedup,
                "parallel_efficiency": actual_speedup / theoretical_speedup if theoretical_speedup > 0 else 0.0,
                "overhead_factor": 1.0 - (actual_speedup / theoretical_speedup) if theoretical_speedup > 0 else 0.0
            }

            # Recommendations
            efficiency = analysis["scaling_analysis"]["parallel_efficiency"]
            if efficiency < 0.5:
                analysis["recommendations"].append(
                    "Low parallel efficiency (<50%) - consider reducing environments per GPU or upgrading GPU memory"
                )
            elif efficiency < 0.8:
                analysis["recommendations"].append(
                    "Moderate parallel efficiency - some overhead from synchronization or memory constraints"
                )
            else:
                analysis["recommendations"].append(
                    "Good parallel efficiency (>80%) - current configuration is well-optimized"
                )

        if results.gpu_utilization_mean < 0.7:
            analysis["recommendations"].append(
                f"GPU utilization is only {results.gpu_utilization_mean*100:.0f}% - consider adding more parallel environments"
            )

        if results.gpu_memory_efficiency > 0.9:
            analysis["recommendations"].append(
                "GPU memory near capacity - reduce environments per GPU if experiencing OOM errors"
            )

        return analysis

    def compute_reproducibility_report(
        self,
        results: ParallelEvalResults
    ) -> Dict[str, Any]:
        """
        Generate reproducibility analysis report.

        UPSELL VALUE: Critical for validating policy performance claims.
        """
        report = {
            "results_id": results.results_id,
            "policy_id": results.config.policy_id,
            "task_name": results.config.task_name,
            "reproducibility_score": results.reproducibility_score,
            "interpretation": "",
            "seed_analysis": {
                "base_seed": results.config.base_seed,
                "num_unique_seeds": len(set(results.config.env_seeds)),
                "seed_distribution": "uniform"
            },
            "variance_analysis": {
                "overall_success_rate": results.success_rate,
                "inter_env_success_variance": results.inter_env_success_variance,
                "inter_env_reward_variance": results.inter_env_reward_variance,
                "coefficient_of_variation": results.std_reward / results.mean_reward if results.mean_reward != 0 else 0
            },
            "environment_outliers": [],
            "recommendations": []
        }

        # Interpret reproducibility score
        if results.reproducibility_score >= 0.9:
            report["interpretation"] = "Excellent reproducibility - results are highly consistent across environments"
        elif results.reproducibility_score >= 0.75:
            report["interpretation"] = "Good reproducibility - results are reasonably consistent"
        elif results.reproducibility_score >= 0.5:
            report["interpretation"] = "Moderate reproducibility - consider investigating sources of variance"
        else:
            report["interpretation"] = "Poor reproducibility - results vary significantly, investigation needed"

        # Identify outlier environments
        if results.environments:
            mean_success_rate = results.success_rate
            success_rates = [
                env.successful_episodes / env.episodes_completed if env.episodes_completed > 0 else 0.0
                for env in results.environments
            ]

            if len(success_rates) > 1 and statistics.stdev(success_rates) > 0:
                std_success_rate = statistics.stdev(success_rates)
                for i, env in enumerate(results.environments):
                    sr = success_rates[i]
                    z_score = (sr - mean_success_rate) / std_success_rate
                    if abs(z_score) > 2:
                        report["environment_outliers"].append({
                            "env_idx": env.env_idx,
                            "seed": env.seed,
                            "success_rate": sr,
                            "z_score": z_score,
                            "direction": "high" if z_score > 0 else "low"
                        })

        # Recommendations
        if len(report["environment_outliers"]) > results.config.num_environments * 0.1:
            report["recommendations"].append(
                f"Found {len(report['environment_outliers'])} outlier environments (>{10}%) - investigate seed sensitivity"
            )

        if report["variance_analysis"]["coefficient_of_variation"] > 0.5:
            report["recommendations"].append(
                "High coefficient of variation in rewards - policy may have high variance, consider ensemble methods"
            )

        if results.reproducibility_score < 0.75:
            report["recommendations"].append(
                "Run additional evaluation with different base seed to confirm results"
            )

        return report

    def save_results(
        self,
        results: ParallelEvalResults,
        include_per_env_data: bool = True
    ) -> str:
        """Save parallel evaluation results to JSON file."""
        output_file = self.output_dir / f"parallel_eval_{results.results_id}.json"

        data = {
            "results_id": results.results_id,
            "config": {
                "config_id": results.config.config_id,
                "policy_id": results.config.policy_id,
                "task_name": results.config.task_name,
                "num_environments": results.config.num_environments,
                "num_gpus": results.config.num_gpus,
                "envs_per_gpu": results.config.envs_per_gpu,
                "strategy": results.config.strategy.value,
                "episodes_per_env": results.config.episodes_per_env,
                "max_episode_steps": results.config.max_episode_steps,
                "evaluation_mode": results.config.evaluation_mode.value,
                "base_seed": results.config.base_seed,
                "pose_variations": results.config.pose_variations,
                "lighting_variations": results.config.lighting_variations,
                "object_variations": results.config.object_variations
            },
            "timing": {
                "start_time": results.start_time.isoformat(),
                "end_time": results.end_time.isoformat(),
                "total_wall_time": results.total_wall_time,
                "total_sim_time": results.total_sim_time
            },
            "aggregate_statistics": {
                "total_episodes": results.total_episodes,
                "successful_episodes": results.successful_episodes,
                "success_rate": results.success_rate,
                "mean_reward": results.mean_reward,
                "std_reward": results.std_reward,
                "mean_episode_length": results.mean_episode_length,
                "std_episode_length": results.std_episode_length
            },
            "throughput": {
                "episodes_per_second": results.episodes_per_second,
                "steps_per_second": results.steps_per_second,
                "sim_to_real_ratio": results.sim_to_real_ratio
            },
            "cross_environment": {
                "inter_env_reward_variance": results.inter_env_reward_variance,
                "inter_env_success_variance": results.inter_env_success_variance,
                "reproducibility_score": results.reproducibility_score
            },
            "gpu_efficiency": {
                "gpu_utilization_mean": results.gpu_utilization_mean,
                "gpu_memory_efficiency": results.gpu_memory_efficiency,
                "power_efficiency": results.power_efficiency
            }
        }

        if include_per_env_data:
            data["environments"] = [
                {
                    "env_idx": env.env_idx,
                    "gpu_id": env.gpu_id,
                    "seed": env.seed,
                    "episodes_completed": env.episodes_completed,
                    "successful_episodes": env.successful_episodes,
                    "success_rate": env.successful_episodes / env.episodes_completed if env.episodes_completed > 0 else 0.0,
                    "mean_reward": statistics.mean(env.episode_rewards) if env.episode_rewards else 0.0,
                    "reward_variance": env.reward_variance,
                    "avg_step_time_ms": env.avg_step_time_ms
                }
                for env in results.environments
            ]

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        return str(output_file)

    def generate_premium_report(
        self,
        results: ParallelEvalResults
    ) -> Dict[str, Any]:
        """
        Generate comprehensive premium parallel evaluation report.

        KEY UPSELL VALUE - GPU-accelerated benchmark analytics.
        """
        throughput = self.compute_throughput_analysis(results)
        reproducibility = self.compute_reproducibility_report(results)
        correlation = self.compute_environment_correlation_matrix(results)

        report = {
            "report_type": "parallel_eval_premium",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": {
                "policy_id": results.config.policy_id,
                "task_name": results.config.task_name,
                "total_episodes": results.total_episodes,
                "success_rate": f"{results.success_rate * 100:.1f}%",
                "throughput": f"{results.episodes_per_second:.1f} episodes/sec",
                "reproducibility": f"{results.reproducibility_score * 100:.0f}%",
                "gpu_utilization": f"{results.gpu_utilization_mean * 100:.0f}%"
            },
            "benchmark_configuration": {
                "parallel_environments": results.config.num_environments,
                "gpus_used": results.config.num_gpus,
                "evaluation_mode": results.config.evaluation_mode.value,
                "total_wall_time": f"{results.total_wall_time:.1f}s",
                "realtime_factor": f"{results.sim_to_real_ratio:.1f}x"
            },
            "performance_metrics": {
                "success_rate": results.success_rate,
                "success_rate_95ci": self._compute_wilson_ci(
                    results.successful_episodes,
                    results.total_episodes
                ),
                "mean_reward": results.mean_reward,
                "reward_std": results.std_reward,
                "mean_episode_length": results.mean_episode_length
            },
            "throughput_analysis": throughput,
            "reproducibility_analysis": reproducibility,
            "environment_correlation": correlation,
            "hardware_recommendations": throughput.get("recommendations", []),
            "upsell_opportunities": [
                "Run policy leaderboard analysis to compare against baseline policies",
                "Perform embodiment transfer analysis for multi-robot deployment",
                "Generate failure mode analysis for episodes that failed",
                "Expand evaluation across additional task variations for generalization metrics"
            ]
        }

        return report

    def _compute_wilson_ci(
        self,
        successes: int,
        total: int,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """Compute Wilson score confidence interval."""
        if total == 0:
            return {"lower": 0.0, "upper": 0.0}

        import math
        p = successes / total
        z = 1.96 if confidence == 0.95 else 1.645

        denominator = 1 + z**2 / total
        centre = p + z**2 / (2 * total)
        margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)

        return {
            "lower": max(0.0, (centre - margin) / denominator),
            "upper": min(1.0, (centre + margin) / denominator)
        }


def create_parallel_eval_capture(output_dir: str = "./parallel_eval") -> ParallelEvalCapture:
    """Factory function to create ParallelEvalCapture instance."""
    return ParallelEvalCapture(output_dir=output_dir)
