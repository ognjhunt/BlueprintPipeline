"""Sim-to-Real Validation Framework.

This framework provides:
1. Structured experiment tracking for sim-to-real transfer
2. Metrics for measuring transfer quality
3. A/B comparison between sim-trained and other policies
4. Real-world trial logging and analysis

The goal is to answer: "Does training on BlueprintPipeline data transfer to real robots?"
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics


class RobotType(str, Enum):
    """Supported robot types."""
    FRANKA = "franka"
    UR10 = "ur10"
    FETCH = "fetch"
    KUKA = "kuka"
    SPOT = "spot"
    OTHER = "other"


class TaskType(str, Enum):
    """Task categories for comparison."""
    PICK_PLACE = "pick_place"
    OPEN_DOOR = "open_door"
    OPEN_DRAWER = "open_drawer"
    POUR = "pour"
    PUSH = "push"
    STACK = "stack"
    INSERT = "insert"
    ARTICULATED_ACCESS = "articulated_access"
    OTHER = "other"


class TrialOutcome(str, Enum):
    """Outcome of a single trial."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"  # Task partially completed
    ABORTED = "aborted"  # Trial stopped early
    ERROR = "error"  # System error


@dataclass
class TransferMetrics:
    """Metrics measuring sim-to-real transfer quality."""
    # Core success metrics
    sim_success_rate: float = 0.0  # Success rate in simulation
    real_success_rate: float = 0.0  # Success rate in real world
    transfer_gap: float = 0.0  # sim - real (lower is better)

    # Trial counts
    sim_trials: int = 0
    real_trials: int = 0

    # Timing metrics
    sim_avg_completion_time: float = 0.0  # seconds
    real_avg_completion_time: float = 0.0

    # Quality metrics
    sim_avg_quality_score: float = 0.0
    real_avg_quality_score: float = 0.0

    # Failure analysis
    real_failure_modes: Dict[str, int] = field(default_factory=dict)

    # Statistical confidence
    confidence_interval_95: Optional[Tuple[float, float]] = None  # For transfer gap

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sim_success_rate": self.sim_success_rate,
            "real_success_rate": self.real_success_rate,
            "transfer_gap": self.transfer_gap,
            "sim_trials": self.sim_trials,
            "real_trials": self.real_trials,
            "sim_avg_completion_time": self.sim_avg_completion_time,
            "real_avg_completion_time": self.real_avg_completion_time,
            "sim_avg_quality_score": self.sim_avg_quality_score,
            "real_avg_quality_score": self.real_avg_quality_score,
            "real_failure_modes": self.real_failure_modes,
            "confidence_interval_95": self.confidence_interval_95,
        }


@dataclass
class Trial:
    """A single experiment trial (sim or real)."""
    trial_id: str
    is_simulation: bool
    outcome: TrialOutcome
    task_type: TaskType
    scene_id: str

    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0

    # Quality
    quality_score: float = 0.0  # 0-1

    # Failure analysis
    failure_mode: Optional[str] = None
    failure_description: Optional[str] = None

    # Observations
    notes: str = ""
    video_path: Optional[str] = None
    sensor_log_path: Optional[str] = None

    # Environment conditions (for real-world)
    lighting_condition: Optional[str] = None
    object_variations: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.trial_id:
            self.trial_id = str(uuid.uuid4())[:8]
        if not self.start_time:
            self.start_time = datetime.utcnow().isoformat() + "Z"


@dataclass
class Sim2RealExperiment:
    """A complete sim-to-real transfer experiment."""
    experiment_id: str
    name: str
    description: str

    # Configuration
    scene_id: str
    task_type: TaskType
    robot_type: RobotType
    policy_source: str  # Path to trained policy

    # Experiment metadata
    created_at: str = ""
    updated_at: str = ""
    status: str = "created"  # created, running, completed, analyzed

    # Training info
    training_scene_source: str = ""  # Which pipeline generated the training data
    training_episodes: int = 0
    training_hours: float = 0.0

    # Trials
    sim_trials: List[Trial] = field(default_factory=list)
    real_trials: List[Trial] = field(default_factory=list)

    # Results
    metrics: Optional[TransferMetrics] = None

    # Partner/lab info (for external validation)
    partner_name: Optional[str] = None
    partner_contact: Optional[str] = None

    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"

    def add_sim_trial(self, trial: Trial) -> None:
        """Add a simulation trial."""
        trial.is_simulation = True
        self.sim_trials.append(trial)
        self.updated_at = datetime.utcnow().isoformat() + "Z"

    def add_real_trial(self, trial: Trial) -> None:
        """Add a real-world trial."""
        trial.is_simulation = False
        self.real_trials.append(trial)
        self.updated_at = datetime.utcnow().isoformat() + "Z"

    def compute_metrics(self) -> TransferMetrics:
        """Compute transfer metrics from trials."""
        metrics = TransferMetrics()

        # Simulation metrics
        if self.sim_trials:
            sim_successes = sum(
                1 for t in self.sim_trials
                if t.outcome == TrialOutcome.SUCCESS
            )
            metrics.sim_trials = len(self.sim_trials)
            metrics.sim_success_rate = sim_successes / len(self.sim_trials)

            sim_times = [
                t.duration_seconds for t in self.sim_trials
                if t.outcome == TrialOutcome.SUCCESS and t.duration_seconds > 0
            ]
            if sim_times:
                metrics.sim_avg_completion_time = statistics.mean(sim_times)

            sim_scores = [t.quality_score for t in self.sim_trials if t.quality_score > 0]
            if sim_scores:
                metrics.sim_avg_quality_score = statistics.mean(sim_scores)

        # Real-world metrics
        if self.real_trials:
            real_successes = sum(
                1 for t in self.real_trials
                if t.outcome == TrialOutcome.SUCCESS
            )
            metrics.real_trials = len(self.real_trials)
            metrics.real_success_rate = real_successes / len(self.real_trials)

            real_times = [
                t.duration_seconds for t in self.real_trials
                if t.outcome == TrialOutcome.SUCCESS and t.duration_seconds > 0
            ]
            if real_times:
                metrics.real_avg_completion_time = statistics.mean(real_times)

            real_scores = [t.quality_score for t in self.real_trials if t.quality_score > 0]
            if real_scores:
                metrics.real_avg_quality_score = statistics.mean(real_scores)

            # Failure mode analysis
            for trial in self.real_trials:
                if trial.outcome != TrialOutcome.SUCCESS and trial.failure_mode:
                    mode = trial.failure_mode
                    metrics.real_failure_modes[mode] = metrics.real_failure_modes.get(mode, 0) + 1

        # Transfer gap
        metrics.transfer_gap = metrics.sim_success_rate - metrics.real_success_rate

        # Confidence interval (Wilson score interval approximation)
        if metrics.real_trials >= 10:
            p = metrics.real_success_rate
            n = metrics.real_trials
            z = 1.96  # 95% confidence

            denominator = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denominator
            margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denominator

            metrics.confidence_interval_95 = (
                max(0, center - margin),
                min(1, center + margin)
            )

        self.metrics = metrics
        return metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "scene_id": self.scene_id,
            "task_type": self.task_type.value,
            "robot_type": self.robot_type.value,
            "policy_source": self.policy_source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "training_scene_source": self.training_scene_source,
            "training_episodes": self.training_episodes,
            "training_hours": self.training_hours,
            "sim_trials": [
                {
                    "trial_id": t.trial_id,
                    "outcome": t.outcome.value,
                    "duration_seconds": t.duration_seconds,
                    "quality_score": t.quality_score,
                    "failure_mode": t.failure_mode,
                    "notes": t.notes,
                }
                for t in self.sim_trials
            ],
            "real_trials": [
                {
                    "trial_id": t.trial_id,
                    "outcome": t.outcome.value,
                    "duration_seconds": t.duration_seconds,
                    "quality_score": t.quality_score,
                    "failure_mode": t.failure_mode,
                    "failure_description": t.failure_description,
                    "notes": t.notes,
                    "video_path": t.video_path,
                    "lighting_condition": t.lighting_condition,
                    "object_variations": t.object_variations,
                }
                for t in self.real_trials
            ],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "partner_name": self.partner_name,
            "partner_contact": self.partner_contact,
        }


@dataclass
class Sim2RealResult:
    """Summary result of sim-to-real validation."""
    experiment_id: str
    scene_id: str
    task_type: str

    # Key metrics
    sim_success_rate: float
    real_success_rate: float
    transfer_gap: float

    # Verdict
    transfer_successful: bool  # True if transfer gap < threshold
    transfer_quality: str  # excellent, good, moderate, poor

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "scene_id": self.scene_id,
            "task_type": self.task_type,
            "sim_success_rate": self.sim_success_rate,
            "real_success_rate": self.real_success_rate,
            "transfer_gap": self.transfer_gap,
            "transfer_successful": self.transfer_successful,
            "transfer_quality": self.transfer_quality,
            "recommendations": self.recommendations,
        }


class Sim2RealValidator:
    """Validator for sim-to-real transfer experiments."""

    # Transfer gap thresholds
    EXCELLENT_THRESHOLD = 0.05  # < 5% gap
    GOOD_THRESHOLD = 0.15  # < 15% gap
    MODERATE_THRESHOLD = 0.30  # < 30% gap

    def __init__(
        self,
        experiments_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.experiments_dir = Path(experiments_dir or "./sim2real_experiments")
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.experiments: Dict[str, Sim2RealExperiment] = {}

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SIM2REAL] {msg}")

    def create_experiment(
        self,
        name: str,
        scene_id: str,
        task_type: TaskType,
        robot_type: RobotType,
        policy_source: str,
        description: str = "",
        **kwargs
    ) -> Sim2RealExperiment:
        """Create a new sim-to-real experiment."""
        experiment = Sim2RealExperiment(
            experiment_id="",  # Will be auto-generated
            name=name,
            description=description,
            scene_id=scene_id,
            task_type=task_type,
            robot_type=robot_type,
            policy_source=policy_source,
            **kwargs
        )

        self.experiments[experiment.experiment_id] = experiment
        self._save_experiment(experiment)
        self.log(f"Created experiment: {experiment.experiment_id} - {name}")

        return experiment

    def log_sim_trial(
        self,
        experiment_id: str,
        outcome: TrialOutcome,
        duration_seconds: float = 0.0,
        quality_score: float = 0.0,
        **kwargs
    ) -> Trial:
        """Log a simulation trial."""
        if experiment_id not in self.experiments:
            self.experiments[experiment_id] = self._load_experiment(experiment_id)

        experiment = self.experiments[experiment_id]

        trial = Trial(
            trial_id="",
            is_simulation=True,
            outcome=outcome,
            task_type=experiment.task_type,
            scene_id=experiment.scene_id,
            duration_seconds=duration_seconds,
            quality_score=quality_score,
            **kwargs
        )

        experiment.add_sim_trial(trial)
        self._save_experiment(experiment)

        return trial

    def log_real_trial(
        self,
        experiment_id: str,
        outcome: TrialOutcome,
        duration_seconds: float = 0.0,
        quality_score: float = 0.0,
        failure_mode: Optional[str] = None,
        failure_description: Optional[str] = None,
        video_path: Optional[str] = None,
        notes: str = "",
        **kwargs
    ) -> Trial:
        """Log a real-world trial."""
        if experiment_id not in self.experiments:
            self.experiments[experiment_id] = self._load_experiment(experiment_id)

        experiment = self.experiments[experiment_id]

        trial = Trial(
            trial_id="",
            is_simulation=False,
            outcome=outcome,
            task_type=experiment.task_type,
            scene_id=experiment.scene_id,
            duration_seconds=duration_seconds,
            quality_score=quality_score,
            failure_mode=failure_mode,
            failure_description=failure_description,
            video_path=video_path,
            notes=notes,
            **kwargs
        )

        experiment.add_real_trial(trial)
        self._save_experiment(experiment)

        self.log(f"Logged real trial: {trial.trial_id} - {outcome.value}")

        return trial

    def analyze_experiment(self, experiment_id: str) -> Sim2RealResult:
        """Analyze an experiment and generate results."""
        if experiment_id not in self.experiments:
            self.experiments[experiment_id] = self._load_experiment(experiment_id)

        experiment = self.experiments[experiment_id]
        metrics = experiment.compute_metrics()

        # Determine transfer quality
        gap = metrics.transfer_gap
        if gap < self.EXCELLENT_THRESHOLD:
            quality = "excellent"
            successful = True
        elif gap < self.GOOD_THRESHOLD:
            quality = "good"
            successful = True
        elif gap < self.MODERATE_THRESHOLD:
            quality = "moderate"
            successful = True
        else:
            quality = "poor"
            successful = False

        # Generate recommendations
        recommendations = []

        if not successful:
            recommendations.append(
                f"Transfer gap of {gap:.1%} is too high. "
                "Consider domain randomization improvements."
            )

        if metrics.real_trials < 10:
            recommendations.append(
                f"Only {metrics.real_trials} real trials. "
                "Need at least 10 for statistical significance."
            )

        if metrics.real_failure_modes:
            top_mode = max(metrics.real_failure_modes, key=metrics.real_failure_modes.get)
            recommendations.append(
                f"Most common failure: {top_mode} "
                f"({metrics.real_failure_modes[top_mode]} occurrences)"
            )

        if metrics.sim_avg_completion_time > 0 and metrics.real_avg_completion_time > 0:
            time_ratio = metrics.real_avg_completion_time / metrics.sim_avg_completion_time
            if time_ratio > 1.5:
                recommendations.append(
                    f"Real-world execution {time_ratio:.1f}x slower than sim. "
                    "Check for conservative motion planning."
                )

        result = Sim2RealResult(
            experiment_id=experiment_id,
            scene_id=experiment.scene_id,
            task_type=experiment.task_type.value,
            sim_success_rate=metrics.sim_success_rate,
            real_success_rate=metrics.real_success_rate,
            transfer_gap=gap,
            transfer_successful=successful,
            transfer_quality=quality,
            recommendations=recommendations,
        )

        # Update experiment status
        experiment.status = "analyzed"
        self._save_experiment(experiment)

        self.log(f"Experiment {experiment_id} analyzed:")
        self.log(f"  Sim success: {metrics.sim_success_rate:.1%}")
        self.log(f"  Real success: {metrics.real_success_rate:.1%}")
        self.log(f"  Transfer gap: {gap:.1%}")
        self.log(f"  Quality: {quality}")

        return result

    def get_all_experiments(self) -> List[Sim2RealExperiment]:
        """Get all experiments."""
        experiments = []
        for path in self.experiments_dir.glob("*.json"):
            try:
                exp = self._load_experiment(path.stem)
                experiments.append(exp)
            except Exception:
                pass
        return experiments

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics across all experiments."""
        experiments = self.get_all_experiments()

        if not experiments:
            return {"error": "No experiments found"}

        analyzed = [e for e in experiments if e.metrics]

        if not analyzed:
            return {"error": "No analyzed experiments"}

        # Aggregate metrics
        total_sim_trials = sum(e.metrics.sim_trials for e in analyzed)
        total_real_trials = sum(e.metrics.real_trials for e in analyzed)

        avg_transfer_gap = statistics.mean(e.metrics.transfer_gap for e in analyzed)
        avg_real_success = statistics.mean(e.metrics.real_success_rate for e in analyzed)

        # By task type
        by_task = {}
        for exp in analyzed:
            task = exp.task_type.value
            if task not in by_task:
                by_task[task] = {"experiments": 0, "gaps": []}
            by_task[task]["experiments"] += 1
            by_task[task]["gaps"].append(exp.metrics.transfer_gap)

        for task, data in by_task.items():
            data["avg_gap"] = statistics.mean(data["gaps"])
            del data["gaps"]

        return {
            "total_experiments": len(experiments),
            "analyzed_experiments": len(analyzed),
            "total_sim_trials": total_sim_trials,
            "total_real_trials": total_real_trials,
            "avg_transfer_gap": avg_transfer_gap,
            "avg_real_success_rate": avg_real_success,
            "by_task_type": by_task,
            "experiments_with_poor_transfer": sum(
                1 for e in analyzed
                if e.metrics.transfer_gap >= self.MODERATE_THRESHOLD
            ),
        }

    def _save_experiment(self, experiment: Sim2RealExperiment) -> None:
        """Save experiment to disk."""
        path = self.experiments_dir / f"{experiment.experiment_id}.json"
        path.write_text(json.dumps(experiment.to_dict(), indent=2))

    def _load_experiment(self, experiment_id: str) -> Sim2RealExperiment:
        """Load experiment from disk."""
        path = self.experiments_dir / f"{experiment_id}.json"
        if not path.exists():
            raise ValueError(f"Experiment not found: {experiment_id}")

        data = json.loads(path.read_text())

        experiment = Sim2RealExperiment(
            experiment_id=data["experiment_id"],
            name=data["name"],
            description=data.get("description", ""),
            scene_id=data["scene_id"],
            task_type=TaskType(data["task_type"]),
            robot_type=RobotType(data["robot_type"]),
            policy_source=data["policy_source"],
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            status=data.get("status", "created"),
            training_scene_source=data.get("training_scene_source", ""),
            training_episodes=data.get("training_episodes", 0),
            training_hours=data.get("training_hours", 0.0),
            partner_name=data.get("partner_name"),
            partner_contact=data.get("partner_contact"),
        )

        # Load trials
        for trial_data in data.get("sim_trials", []):
            trial = Trial(
                trial_id=trial_data["trial_id"],
                is_simulation=True,
                outcome=TrialOutcome(trial_data["outcome"]),
                task_type=experiment.task_type,
                scene_id=experiment.scene_id,
                duration_seconds=trial_data.get("duration_seconds", 0.0),
                quality_score=trial_data.get("quality_score", 0.0),
                failure_mode=trial_data.get("failure_mode"),
                notes=trial_data.get("notes", ""),
            )
            experiment.sim_trials.append(trial)

        for trial_data in data.get("real_trials", []):
            trial = Trial(
                trial_id=trial_data["trial_id"],
                is_simulation=False,
                outcome=TrialOutcome(trial_data["outcome"]),
                task_type=experiment.task_type,
                scene_id=experiment.scene_id,
                duration_seconds=trial_data.get("duration_seconds", 0.0),
                quality_score=trial_data.get("quality_score", 0.0),
                failure_mode=trial_data.get("failure_mode"),
                failure_description=trial_data.get("failure_description"),
                notes=trial_data.get("notes", ""),
                video_path=trial_data.get("video_path"),
                lighting_condition=trial_data.get("lighting_condition"),
                object_variations=trial_data.get("object_variations", []),
            )
            experiment.real_trials.append(trial)

        # Load metrics if available
        if data.get("metrics"):
            m = data["metrics"]
            experiment.metrics = TransferMetrics(
                sim_success_rate=m.get("sim_success_rate", 0.0),
                real_success_rate=m.get("real_success_rate", 0.0),
                transfer_gap=m.get("transfer_gap", 0.0),
                sim_trials=m.get("sim_trials", 0),
                real_trials=m.get("real_trials", 0),
                sim_avg_completion_time=m.get("sim_avg_completion_time", 0.0),
                real_avg_completion_time=m.get("real_avg_completion_time", 0.0),
                sim_avg_quality_score=m.get("sim_avg_quality_score", 0.0),
                real_avg_quality_score=m.get("real_avg_quality_score", 0.0),
                real_failure_modes=m.get("real_failure_modes", {}),
                confidence_interval_95=tuple(m["confidence_interval_95"]) if m.get("confidence_interval_95") else None,
            )

        return experiment
