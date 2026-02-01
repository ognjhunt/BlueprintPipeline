"""
Arena Evaluation Runner - Run policy evaluations on Blueprint scenes.

This module integrates with Isaac Lab-Arena's evaluation infrastructure
to benchmark robot policies on Blueprint-generated environments.

Features:
- Single-policy evaluation
- Multi-policy comparison (leaderboard generation)
- Parallel GPU-accelerated evaluation
- Metric collection and export
- Integration with LeRobot evaluation hub
"""

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tools.config.env import parse_bool_env

from .benchmark_validation import (
    resolve_isaac_lab_arena_version,
    validate_arena_benchmark_results,
)

@dataclass
class EvaluationConfig:
    """Configuration for Arena policy evaluation."""
    scene_id: str
    policy_path: str                          # Path to policy checkpoint
    policy_type: str = "gr00t_closedloop"     # Policy architecture type
    num_envs: int = 10                        # Parallel environments
    num_episodes: int = 100                   # Total evaluation episodes
    max_steps_per_episode: int = 500
    tasks: list[str] = field(default_factory=list)  # Specific tasks to evaluate
    embodiment: str = "franka"
    seed: int = 42
    record_video: bool = False
    output_dir: Optional[Path] = None

    # Isaac Sim configuration
    headless: bool = True
    gpu_id: int = 0

    # Advanced options
    domain_randomization: bool = False
    success_threshold: float = 0.9


@dataclass
class TaskMetrics:
    """Metrics for a single task evaluation."""
    task_name: str
    success_rate: float
    mean_steps_to_success: float
    std_steps_to_success: float
    mean_reward: float
    std_reward: float
    num_episodes: int
    num_successes: int
    timeout_rate: float
    collision_rate: float = 0.0
    additional_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    success: bool
    scene_id: str
    policy_path: str
    policy_type: str
    embodiment: str
    timestamp: str
    total_episodes: int
    overall_success_rate: float
    task_metrics: dict[str, TaskMetrics]
    summary: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    video_paths: list[str] = field(default_factory=list)


class ArenaEvaluationRunner:
    """
    Runs Arena policy evaluations on Blueprint scenes.

    This runner:
    1. Loads Arena scene and tasks
    2. Instantiates policy models
    3. Runs parallel evaluations
    4. Collects and aggregates metrics
    5. Exports results for leaderboard/comparison
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.arena_available = self._check_arena_available()

    def _check_arena_available(self) -> bool:
        """Check if Isaac Lab-Arena is available."""
        try:
            # Try importing Arena modules
            import isaaclab_arena  # noqa: F401
            return True
        except ImportError:
            return False

    def run(self, arena_dir: Path) -> EvaluationResult:
        """
        Run policy evaluation.

        Args:
            arena_dir: Path to exported Arena scene directory

        Returns:
            EvaluationResult with evaluation metrics
        """
        timestamp = datetime.utcnow().isoformat()
        errors: list[str] = []

        if not self.arena_available:
            # Run in subprocess mode (Isaac Sim container)
            return self._run_subprocess_evaluation(arena_dir, timestamp)

        # Direct evaluation (Arena available in current environment)
        return self._run_direct_evaluation(arena_dir, timestamp)

    def _run_direct_evaluation(
        self,
        arena_dir: Path,
        timestamp: str
    ) -> EvaluationResult:
        """Run evaluation directly using Arena APIs."""
        errors: list[str] = []
        task_metrics: dict[str, TaskMetrics] = {}

        try:
            from isaaclab_arena import ArenaEnvBuilder
            from isaaclab_arena.evaluation import PolicyRunner, EvaluationMetrics

            # Load scene module
            scene_module_path = arena_dir / "scene_module.py"
            if not scene_module_path.exists():
                errors.append(f"Scene module not found: {scene_module_path}")
                return self._create_error_result(timestamp, errors)

            # Import scene dynamically
            import importlib.util
            spec = importlib.util.spec_from_file_location("scene_module", scene_module_path)
            scene_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scene_module)
            scene = scene_module.get_scene()

            # Discover tasks
            tasks_to_run = self.config.tasks or self._discover_tasks(arena_dir)

            # Load policy
            policy = self._load_policy()

            # Run evaluation for each task
            total_successes = 0
            total_episodes = 0

            for task_name in tasks_to_run:
                try:
                    metrics = self._evaluate_task(
                        scene, task_name, policy, arena_dir
                    )
                    task_metrics[task_name] = metrics
                    total_successes += metrics.num_successes
                    total_episodes += metrics.num_episodes
                except Exception as e:
                    errors.append(f"Task {task_name} failed: {e}")

            overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0

            return EvaluationResult(
                success=len(errors) == 0,
                scene_id=self.config.scene_id,
                policy_path=self.config.policy_path,
                policy_type=self.config.policy_type,
                embodiment=self.config.embodiment,
                timestamp=timestamp,
                total_episodes=total_episodes,
                overall_success_rate=overall_success_rate,
                task_metrics=task_metrics,
                summary=self._generate_summary(task_metrics),
                errors=errors,
            )

        except Exception as e:
            errors.append(f"Evaluation failed: {e}")
            return self._create_error_result(timestamp, errors)

    def _run_subprocess_evaluation(
        self,
        arena_dir: Path,
        timestamp: str
    ) -> EvaluationResult:
        """
        Run evaluation in Isaac Sim Docker container via subprocess.

        This is used when Arena is not directly available but we want
        to trigger evaluation in an external environment.
        """
        errors: list[str] = []

        # Create evaluation script
        eval_script = self._generate_evaluation_script(arena_dir)
        script_path = arena_dir / "run_evaluation.py"
        script_path.write_text(eval_script)

        # Output path for results
        output_dir = self.config.output_dir or arena_dir / "evaluation_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f"eval_{timestamp.replace(':', '-')}.json"

        # Build Docker command if using containerized Isaac Sim
        docker_image = os.getenv("ISAAC_SIM_DOCKER_IMAGE", "nvcr.io/nvidia/isaac-sim:5.1.0")
        use_docker = parse_bool_env(os.getenv("USE_DOCKER_EVALUATION"), default=True)

        if use_docker:
            cmd = [
                "docker", "run", "--rm",
                "--gpus", f'"device={self.config.gpu_id}"',
                "-v", f"{arena_dir}:/workspace/arena",
                "-v", f"{output_dir}:/workspace/output",
                "-e", f"POLICY_PATH={self.config.policy_path}",
                "-e", f"NUM_ENVS={self.config.num_envs}",
                "-e", f"NUM_EPISODES={self.config.num_episodes}",
                docker_image,
                "python", "/workspace/arena/run_evaluation.py",
                "--output", f"/workspace/output/eval_{timestamp.replace(':', '-')}.json"
            ]
        else:
            # Direct execution (assumes Isaac Sim in PATH)
            cmd = [
                sys.executable, str(script_path),
                "--output", str(results_path),
                "--policy-path", self.config.policy_path,
                "--num-envs", str(self.config.num_envs),
                "--num-episodes", str(self.config.num_episodes),
            ]

        try:
            # Execute evaluation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                errors.append(f"Evaluation subprocess failed: {result.stderr}")
                return self._create_error_result(timestamp, errors)

            # Load results
            if results_path.exists():
                with open(results_path) as f:
                    eval_data = json.load(f)
                return self._parse_subprocess_results(eval_data, timestamp)
            else:
                errors.append("Results file not generated")
                return self._create_error_result(timestamp, errors)

        except subprocess.TimeoutExpired:
            errors.append("Evaluation timed out after 1 hour")
            return self._create_error_result(timestamp, errors)
        except Exception as e:
            errors.append(f"Subprocess execution failed: {e}")
            return self._create_error_result(timestamp, errors)

    def _generate_evaluation_script(self, arena_dir: Path) -> str:
        """Generate Python script for subprocess evaluation."""
        return f'''#!/usr/bin/env python3
"""
Arena Evaluation Script
Auto-generated by BlueprintPipeline

Run this script in an Isaac Sim environment to evaluate policies.
"""

import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--policy-path", default="{self.config.policy_path}")
    parser.add_argument("--num-envs", type=int, default={self.config.num_envs})
    parser.add_argument("--num-episodes", type=int, default={self.config.num_episodes})
    parser.add_argument("--embodiment", default="{self.config.embodiment}")
    args = parser.parse_args()

    try:
        from isaaclab_arena import ArenaEnvBuilder
        from isaaclab_arena.evaluation import PolicyRunner

        # Import scene
        sys.path.insert(0, str(Path(__file__).parent))
        from scene_module import get_scene

        scene = get_scene()

        # Run evaluation
        runner = PolicyRunner(
            policy_path=args.policy_path,
            num_envs=args.num_envs,
        )

        results = runner.evaluate(
            scene=scene,
            num_episodes=args.num_episodes,
            embodiment=args.embodiment,
        )

        # Save results
        with open(args.output, "w") as f:
            json.dump(results.to_dict(), f, indent=2)

        print(f"Evaluation complete. Results saved to {{args.output}}")

    except Exception as e:
        error_result = {{
            "success": False,
            "error": str(e),
        }}
        with open(args.output, "w") as f:
            json.dump(error_result, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
'''

    def _discover_tasks(self, arena_dir: Path) -> list[str]:
        """Discover available tasks from Arena export."""
        tasks_dir = arena_dir / "tasks"
        if not tasks_dir.exists():
            return []

        tasks = []
        for task_file in tasks_dir.glob("*.py"):
            if task_file.stem != "__init__":
                tasks.append(task_file.stem)
        return tasks

    def _load_policy(self) -> Any:
        """Load policy from checkpoint."""
        # Policy loading depends on policy type
        policy_type = self.config.policy_type
        non_gr00t_policy_load = False

        if policy_type == "gr00t_closedloop":
            try:
                from gr00t.policy import GR00TPolicy
                return GR00TPolicy.load(self.config.policy_path)
            except ImportError as exc:
                non_gr00t_policy_load = True
                missing_module = exc.name or "gr00t.policy"
                logger = logging.getLogger(__name__)
                logger.warning(
                    "GR00T policy import failed for module '%s'. "
                    "Install the GR00T package or ensure it is on PYTHONPATH. "
                    "Falling back to non-GR00T policy load. "
                    "policy_type=%s policy_path=%s non_gr00t_policy_load=%s",
                    missing_module,
                    policy_type,
                    self.config.policy_path,
                    non_gr00t_policy_load,
                )

        # Fallback: try to load as torch checkpoint
        try:
            import torch
            return torch.load(self.config.policy_path)
        except Exception as exc:
            raise ValueError(
                "Failed to load policy "
                f"(policy_type={policy_type}, policy_path={self.config.policy_path}, "
                f"non_gr00t_policy_load={non_gr00t_policy_load})"
            ) from exc

    def _evaluate_task(
        self,
        scene: Any,
        task_name: str,
        policy: Any,
        arena_dir: Path
    ) -> TaskMetrics:
        """Evaluate policy on a single task."""
        # This is a placeholder - actual implementation would use Arena APIs
        from isaaclab_arena import ArenaEnvBuilder
        from isaaclab_arena.evaluation import run_episodes

        # Load task
        task_module_path = arena_dir / "tasks" / f"{task_name}.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location(task_name, task_module_path)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)

        task = task_module.create_task()

        # Build environment
        builder = ArenaEnvBuilder(
            scene=scene,
            task=task,
            embodiment=self.config.embodiment,
            num_envs=self.config.num_envs,
        )
        env = builder.build()

        # Run episodes
        results = run_episodes(
            env=env,
            policy=policy,
            num_episodes=self.config.num_episodes // len(self._discover_tasks(arena_dir)),
            max_steps=self.config.max_steps_per_episode,
        )

        return TaskMetrics(
            task_name=task_name,
            success_rate=results.success_rate,
            mean_steps_to_success=results.mean_steps,
            std_steps_to_success=results.std_steps,
            mean_reward=results.mean_reward,
            std_reward=results.std_reward,
            num_episodes=results.num_episodes,
            num_successes=results.num_successes,
            timeout_rate=results.timeout_rate,
        )

    def _generate_summary(self, task_metrics: dict[str, TaskMetrics]) -> dict[str, Any]:
        """Generate evaluation summary statistics."""
        if not task_metrics:
            return {}

        success_rates = [m.success_rate for m in task_metrics.values()]
        mean_rewards = [m.mean_reward for m in task_metrics.values()]

        return {
            "num_tasks": len(task_metrics),
            "mean_success_rate": sum(success_rates) / len(success_rates),
            "min_success_rate": min(success_rates),
            "max_success_rate": max(success_rates),
            "mean_reward": sum(mean_rewards) / len(mean_rewards),
            "best_task": max(task_metrics.items(), key=lambda x: x[1].success_rate)[0],
            "worst_task": min(task_metrics.items(), key=lambda x: x[1].success_rate)[0],
        }

    def _create_error_result(
        self,
        timestamp: str,
        errors: list[str]
    ) -> EvaluationResult:
        """Create an error result."""
        return EvaluationResult(
            success=False,
            scene_id=self.config.scene_id,
            policy_path=self.config.policy_path,
            policy_type=self.config.policy_type,
            embodiment=self.config.embodiment,
            timestamp=timestamp,
            total_episodes=0,
            overall_success_rate=0.0,
            task_metrics={},
            summary={},
            errors=errors,
        )

    def _parse_subprocess_results(
        self,
        data: dict[str, Any],
        timestamp: str
    ) -> EvaluationResult:
        """Parse results from subprocess evaluation."""
        if not data.get("success", False):
            return self._create_error_result(timestamp, [data.get("error", "Unknown error")])

        task_metrics = {}
        for task_name, metrics_data in data.get("task_metrics", {}).items():
            task_metrics[task_name] = TaskMetrics(
                task_name=task_name,
                success_rate=metrics_data.get("success_rate", 0.0),
                mean_steps_to_success=metrics_data.get("mean_steps", 0.0),
                std_steps_to_success=metrics_data.get("std_steps", 0.0),
                mean_reward=metrics_data.get("mean_reward", 0.0),
                std_reward=metrics_data.get("std_reward", 0.0),
                num_episodes=metrics_data.get("num_episodes", 0),
                num_successes=metrics_data.get("num_successes", 0),
                timeout_rate=metrics_data.get("timeout_rate", 0.0),
            )

        return EvaluationResult(
            success=True,
            scene_id=self.config.scene_id,
            policy_path=self.config.policy_path,
            policy_type=self.config.policy_type,
            embodiment=self.config.embodiment,
            timestamp=timestamp,
            total_episodes=data.get("total_episodes", 0),
            overall_success_rate=data.get("overall_success_rate", 0.0),
            task_metrics=task_metrics,
            summary=data.get("summary", {}),
        )

    def export_results(
        self,
        result: EvaluationResult,
        output_path: Path,
        arena_dir: Optional[Path] = None,
    ) -> None:
        """Export evaluation results to JSON."""
        arena_version = resolve_isaac_lab_arena_version(arena_dir)
        data = {
            "success": result.success,
            "scene_id": result.scene_id,
            "policy_path": result.policy_path,
            "policy_type": result.policy_type,
            "embodiment": result.embodiment,
            "timestamp": result.timestamp,
            "total_episodes": result.total_episodes,
            "overall_success_rate": result.overall_success_rate,
            "task_metrics": {
                name: {
                    "task_name": m.task_name,
                    "success_rate": m.success_rate,
                    "mean_steps_to_success": m.mean_steps_to_success,
                    "std_steps_to_success": m.std_steps_to_success,
                    "mean_reward": m.mean_reward,
                    "std_reward": m.std_reward,
                    "num_episodes": m.num_episodes,
                    "num_successes": m.num_successes,
                    "timeout_rate": m.timeout_rate,
                }
                for name, m in result.task_metrics.items()
            },
            "summary": result.summary,
            "errors": result.errors,
        }

        if arena_version:
            data["isaac_lab_arena_version"] = arena_version

        validate_arena_benchmark_results(
            data,
            scene_id=result.scene_id,
            task_ids=list(result.task_metrics.keys()),
            arena_dir=arena_dir,
            arena_version=arena_version,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


def run_arena_evaluation(
    arena_dir: Path,
    policy_path: str,
    scene_id: str,
    num_episodes: int = 100,
    embodiment: str = "franka",
) -> EvaluationResult:
    """
    Convenience function to run Arena evaluation.

    Args:
        arena_dir: Path to Arena export directory
        policy_path: Path to policy checkpoint
        scene_id: Scene identifier
        num_episodes: Number of evaluation episodes
        embodiment: Robot embodiment to use

    Returns:
        EvaluationResult with metrics
    """
    config = EvaluationConfig(
        scene_id=scene_id,
        policy_path=policy_path,
        num_episodes=num_episodes,
        embodiment=embodiment,
    )

    runner = ArenaEvaluationRunner(config)
    return runner.run(arena_dir)
