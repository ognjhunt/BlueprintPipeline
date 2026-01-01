"""Experiment Tracking for Sim-to-Real Validation.

Provides convenience functions for tracking experiments and logging results.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .validation import (
    Sim2RealValidator,
    Sim2RealExperiment,
    Sim2RealResult,
    TaskType,
    RobotType,
    TrialOutcome,
    Trial,
)


# Global validator instance
_validator: Optional[Sim2RealValidator] = None


def get_validator() -> Sim2RealValidator:
    """Get or create the global validator instance."""
    global _validator
    if _validator is None:
        experiments_dir = Path(os.getenv(
            "SIM2REAL_EXPERIMENTS_DIR",
            "./sim2real_experiments"
        ))
        _validator = Sim2RealValidator(experiments_dir=experiments_dir)
    return _validator


class ExperimentTracker:
    """High-level experiment tracker for convenient usage."""

    def __init__(
        self,
        experiments_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.validator = Sim2RealValidator(
            experiments_dir=experiments_dir,
            verbose=verbose,
        )

    def create(
        self,
        name: str,
        scene_id: str,
        task_type: str,
        robot_type: str = "franka",
        policy_source: str = "",
        description: str = "",
        **kwargs
    ) -> Sim2RealExperiment:
        """Create a new experiment.

        Args:
            name: Experiment name
            scene_id: Scene identifier
            task_type: Task type (pick_place, open_door, etc.)
            robot_type: Robot type (franka, ur10, fetch, etc.)
            policy_source: Path to trained policy
            description: Experiment description
            **kwargs: Additional experiment properties

        Returns:
            Created experiment
        """
        return self.validator.create_experiment(
            name=name,
            scene_id=scene_id,
            task_type=TaskType(task_type),
            robot_type=RobotType(robot_type),
            policy_source=policy_source,
            description=description,
            **kwargs
        )

    def log_sim(
        self,
        experiment_id: str,
        success: bool,
        duration: float = 0.0,
        quality: float = 0.0,
        **kwargs
    ) -> Trial:
        """Log a simulation trial.

        Args:
            experiment_id: Experiment identifier
            success: Whether trial succeeded
            duration: Trial duration in seconds
            quality: Quality score (0-1)
            **kwargs: Additional trial properties

        Returns:
            Created trial
        """
        return self.validator.log_sim_trial(
            experiment_id=experiment_id,
            outcome=TrialOutcome.SUCCESS if success else TrialOutcome.FAILURE,
            duration_seconds=duration,
            quality_score=quality,
            **kwargs
        )

    def log_real(
        self,
        experiment_id: str,
        success: bool,
        duration: float = 0.0,
        quality: float = 0.0,
        failure_mode: Optional[str] = None,
        failure_description: Optional[str] = None,
        video_path: Optional[str] = None,
        notes: str = "",
        **kwargs
    ) -> Trial:
        """Log a real-world trial.

        Args:
            experiment_id: Experiment identifier
            success: Whether trial succeeded
            duration: Trial duration in seconds
            quality: Quality score (0-1)
            failure_mode: Category of failure (if failed)
            failure_description: Detailed failure description
            video_path: Path to video recording
            notes: Additional notes
            **kwargs: Additional trial properties

        Returns:
            Created trial
        """
        return self.validator.log_real_trial(
            experiment_id=experiment_id,
            outcome=TrialOutcome.SUCCESS if success else TrialOutcome.FAILURE,
            duration_seconds=duration,
            quality_score=quality,
            failure_mode=failure_mode,
            failure_description=failure_description,
            video_path=video_path,
            notes=notes,
            **kwargs
        )

    def analyze(self, experiment_id: str) -> Sim2RealResult:
        """Analyze an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Analysis result
        """
        return self.validator.analyze_experiment(experiment_id)

    def get_stats(self) -> dict:
        """Get aggregate statistics."""
        return self.validator.get_aggregate_stats()


# Convenience functions using global validator
def register_experiment(
    name: str,
    scene_id: str,
    task_type: str,
    robot_type: str = "franka",
    policy_source: str = "",
    **kwargs
) -> str:
    """Register a new experiment and return its ID.

    Example:
        exp_id = register_experiment(
            name="Kitchen Mug Pick-Place",
            scene_id="kitchen_001",
            task_type="pick_place",
            robot_type="franka",
            policy_source="./policies/mug_pick_place.pt"
        )
    """
    validator = get_validator()
    experiment = validator.create_experiment(
        name=name,
        scene_id=scene_id,
        task_type=TaskType(task_type),
        robot_type=RobotType(robot_type),
        policy_source=policy_source,
        **kwargs
    )
    return experiment.experiment_id


def log_real_world_result(
    experiment_id: str,
    success: bool,
    duration_seconds: float = 0.0,
    failure_mode: Optional[str] = None,
    notes: str = "",
    video_path: Optional[str] = None,
) -> Trial:
    """Log a real-world trial result.

    Example:
        log_real_world_result(
            experiment_id="abc123",
            success=True,
            duration_seconds=12.5,
            notes="Clean execution, object placed accurately"
        )

        log_real_world_result(
            experiment_id="abc123",
            success=False,
            duration_seconds=8.0,
            failure_mode="grasp_failure",
            notes="Gripper slipped on object surface"
        )
    """
    validator = get_validator()
    return validator.log_real_trial(
        experiment_id=experiment_id,
        outcome=TrialOutcome.SUCCESS if success else TrialOutcome.FAILURE,
        duration_seconds=duration_seconds,
        failure_mode=failure_mode,
        notes=notes,
        video_path=video_path,
    )


def analyze_experiment(experiment_id: str) -> Sim2RealResult:
    """Analyze an experiment and get results.

    Example:
        result = analyze_experiment("abc123")
        print(f"Transfer gap: {result.transfer_gap:.1%}")
        print(f"Quality: {result.transfer_quality}")
    """
    validator = get_validator()
    return validator.analyze_experiment(experiment_id)


def get_aggregate_stats() -> dict:
    """Get aggregate statistics across all experiments.

    Example:
        stats = get_aggregate_stats()
        print(f"Total experiments: {stats['total_experiments']}")
        print(f"Avg transfer gap: {stats['avg_transfer_gap']:.1%}")
    """
    validator = get_validator()
    return validator.get_aggregate_stats()


# CLI interface
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Sim-to-Real Experiment Tracking")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Create experiment
    create_parser = subparsers.add_parser("create", help="Create new experiment")
    create_parser.add_argument("--name", required=True)
    create_parser.add_argument("--scene-id", required=True)
    create_parser.add_argument("--task-type", required=True)
    create_parser.add_argument("--robot-type", default="franka")
    create_parser.add_argument("--policy-source", default="")

    # Log trial
    log_parser = subparsers.add_parser("log", help="Log trial result")
    log_parser.add_argument("--experiment-id", required=True)
    log_parser.add_argument("--success", action="store_true")
    log_parser.add_argument("--failure", action="store_true")
    log_parser.add_argument("--sim", action="store_true", help="Simulation trial")
    log_parser.add_argument("--real", action="store_true", help="Real-world trial")
    log_parser.add_argument("--duration", type=float, default=0.0)
    log_parser.add_argument("--failure-mode", default=None)
    log_parser.add_argument("--notes", default="")

    # Analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze experiment")
    analyze_parser.add_argument("--experiment-id", required=True)

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Get aggregate stats")

    args = parser.parse_args()

    if args.command == "create":
        exp_id = register_experiment(
            name=args.name,
            scene_id=args.scene_id,
            task_type=args.task_type,
            robot_type=args.robot_type,
            policy_source=args.policy_source,
        )
        print(f"Created experiment: {exp_id}")

    elif args.command == "log":
        validator = get_validator()
        success = args.success or not args.failure

        if args.real:
            trial = validator.log_real_trial(
                experiment_id=args.experiment_id,
                outcome=TrialOutcome.SUCCESS if success else TrialOutcome.FAILURE,
                duration_seconds=args.duration,
                failure_mode=args.failure_mode,
                notes=args.notes,
            )
        else:
            trial = validator.log_sim_trial(
                experiment_id=args.experiment_id,
                outcome=TrialOutcome.SUCCESS if success else TrialOutcome.FAILURE,
                duration_seconds=args.duration,
                notes=args.notes,
            )
        print(f"Logged trial: {trial.trial_id}")

    elif args.command == "analyze":
        result = analyze_experiment(args.experiment_id)
        print(json.dumps(result.to_dict(), indent=2))

    elif args.command == "stats":
        stats = get_aggregate_stats()
        print(json.dumps(stats, indent=2))

    else:
        parser.print_help()
