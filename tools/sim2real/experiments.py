"""Experiment Tracking for Sim-to-Real Validation.

Provides convenience functions for tracking experiments and logging results.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

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
_default_assignments_path: Optional[Path] = None


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


def _get_assignments_path(assignments_path: Optional[Path] = None) -> Path:
    """Resolve the assignments store path for A/B tests."""
    global _default_assignments_path
    if assignments_path is not None:
        return assignments_path
    if _default_assignments_path is None:
        experiments_dir = Path(os.getenv(
            "SIM2REAL_EXPERIMENTS_DIR",
            "./sim2real_experiments"
        ))
        _default_assignments_path = experiments_dir / "ab_assignments.json"
    return _default_assignments_path


def _load_assignments(assignments_path: Path) -> Dict[str, Dict[str, str]]:
    """Load persisted scene assignments."""
    if not assignments_path.exists():
        return {}
    try:
        with assignments_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _persist_assignments(assignments_path: Path, assignments: Dict[str, Dict[str, str]]) -> None:
    """Persist assignments to disk using atomic write."""
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = assignments_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(assignments, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(assignments_path)


def _normalize_split_ratios(split_ratios: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Normalize split ratios for A/B variants."""
    ratios = split_ratios or {"A": 0.5, "B": 0.5}
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return {variant: weight / total for variant, weight in ratios.items() if weight > 0}


def assign_scene_variant(
    scene_id: str,
    *,
    enabled: bool = True,
    split_ratios: Optional[Dict[str, float]] = None,
    assignments_path: Optional[Path] = None,
    seed: Optional[int] = None,
) -> str:
    """Assign a scene to an A/B testing variant and persist the assignment.

    Args:
        scene_id: Unique scene identifier.
        enabled: Whether A/B testing is enabled. If disabled, returns "A".
        split_ratios: Mapping of variant labels to weights (e.g., {"A": 0.5, "B": 0.5}).
        assignments_path: Optional path to store assignments.
        seed: Optional seed for deterministic assignment.

    Returns:
        Variant label (e.g., "A" or "B").
    """
    if not enabled:
        return "A"

    assignments_path = _get_assignments_path(assignments_path)
    assignments = _load_assignments(assignments_path)

    if scene_id in assignments and isinstance(assignments[scene_id], dict):
        variant = assignments[scene_id].get("variant")
        if isinstance(variant, str) and variant:
            return variant

    normalized = _normalize_split_ratios(split_ratios)
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    roll = rng.random()
    cumulative = 0.0
    variant = "A"
    for label, weight in normalized.items():
        cumulative += weight
        if roll <= cumulative:
            variant = label
            break

    assignments[scene_id] = {
        "variant": variant,
        "assigned_at": datetime.utcnow().isoformat() + "Z",
    }
    _persist_assignments(assignments_path, assignments)
    return variant


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
