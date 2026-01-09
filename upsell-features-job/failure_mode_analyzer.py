#!/usr/bin/env python3
"""
Failure Mode Analysis Module for BlueprintPipeline.

Provides comprehensive failure analysis for robotics labs including:
- Failure taxonomy and categorization
- Root cause analysis
- Frame-by-frame failure detection
- Failure mode distribution statistics
- Recovery pattern analysis
- Actionable recommendations

Upsell Value: $10,000-$50,000 per dataset
- Labs save 10x debugging time
- "Don't train on this failure mode" guidance
- Essential for data quality assessment
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid


class FailureCategory(str, Enum):
    """High-level failure categories."""
    GRASP_FAILURE = "grasp_failure"
    COLLISION = "collision"
    PLACEMENT_ERROR = "placement_error"
    TIMEOUT = "timeout"
    JOINT_LIMIT = "joint_limit"
    MOTION_PLANNING = "motion_planning"
    PERCEPTION_ERROR = "perception_error"
    PHYSICS_VIOLATION = "physics_violation"
    UNKNOWN = "unknown"


class FailureSeverity(str, Enum):
    """Failure severity levels."""
    CRITICAL = "critical"      # Task cannot continue, may damage robot
    MAJOR = "major"            # Task failed, but recoverable
    MINOR = "minor"            # Task degraded, may still succeed
    WARNING = "warning"        # Potential issue detected


@dataclass
class FailureEvent:
    """Individual failure event with detailed context."""
    event_id: str
    episode_id: str
    frame_idx: int
    timestamp: float

    # Classification
    category: FailureCategory
    subcategory: str
    severity: FailureSeverity

    # Context
    phase: str  # Motion phase when failure occurred
    robot_state: Dict[str, Any]
    contact_info: Optional[Dict[str, Any]] = None

    # Analysis
    root_cause: Optional[str] = None
    contributing_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    confidence: float = 1.0  # How confident is this detection
    recoverable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "episode_id": self.episode_id,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "category": self.category.value,
            "subcategory": self.subcategory,
            "severity": self.severity.value,
            "phase": self.phase,
            "root_cause": self.root_cause,
            "contributing_factors": self.contributing_factors,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "recoverable": self.recoverable,
        }


@dataclass
class FailureTaxonomy:
    """Complete failure taxonomy for a dataset."""
    # Grasp failures
    grasp_miss: int = 0          # Gripper closed on empty space
    grasp_slip: int = 0          # Object slipped during manipulation
    grasp_collision: int = 0     # Collision during grasp attempt
    grasp_unstable: int = 0      # Grasp achieved but unstable
    grasp_force_exceeded: int = 0  # Excessive grasp force

    # Collision failures
    collision_environment: int = 0  # Hit table, walls, etc.
    collision_object: int = 0       # Hit non-target objects
    collision_self: int = 0         # Robot self-collision
    collision_target: int = 0       # Unintended collision with target

    # Placement failures
    placement_miss: int = 0         # Object not at target
    placement_unstable: int = 0     # Object falls after release
    placement_collision: int = 0    # Collision during placement
    placement_orientation: int = 0  # Wrong orientation at placement

    # Motion failures
    joint_limit_exceeded: int = 0
    singularity_approach: int = 0
    path_blocked: int = 0
    velocity_exceeded: int = 0

    # Timing failures
    timeout_approach: int = 0
    timeout_grasp: int = 0
    timeout_transport: int = 0
    timeout_place: int = 0

    # Physics violations
    physics_penetration: int = 0
    physics_unstable: int = 0
    physics_constraint_violated: int = 0

    def total_failures(self) -> int:
        """Total number of failures."""
        return sum([
            self.grasp_miss, self.grasp_slip, self.grasp_collision,
            self.grasp_unstable, self.grasp_force_exceeded,
            self.collision_environment, self.collision_object,
            self.collision_self, self.collision_target,
            self.placement_miss, self.placement_unstable,
            self.placement_collision, self.placement_orientation,
            self.joint_limit_exceeded, self.singularity_approach,
            self.path_blocked, self.velocity_exceeded,
            self.timeout_approach, self.timeout_grasp,
            self.timeout_transport, self.timeout_place,
            self.physics_penetration, self.physics_unstable,
            self.physics_constraint_violated,
        ])

    def to_dict(self) -> Dict[str, int]:
        return {
            "grasp_failures": {
                "miss": self.grasp_miss,
                "slip": self.grasp_slip,
                "collision": self.grasp_collision,
                "unstable": self.grasp_unstable,
                "force_exceeded": self.grasp_force_exceeded,
            },
            "collision_failures": {
                "environment": self.collision_environment,
                "object": self.collision_object,
                "self": self.collision_self,
                "target": self.collision_target,
            },
            "placement_failures": {
                "miss": self.placement_miss,
                "unstable": self.placement_unstable,
                "collision": self.placement_collision,
                "orientation": self.placement_orientation,
            },
            "motion_failures": {
                "joint_limit": self.joint_limit_exceeded,
                "singularity": self.singularity_approach,
                "path_blocked": self.path_blocked,
                "velocity_exceeded": self.velocity_exceeded,
            },
            "timeout_failures": {
                "approach": self.timeout_approach,
                "grasp": self.timeout_grasp,
                "transport": self.timeout_transport,
                "place": self.timeout_place,
            },
            "physics_violations": {
                "penetration": self.physics_penetration,
                "unstable": self.physics_unstable,
                "constraint_violated": self.physics_constraint_violated,
            },
        }


@dataclass
class FailureAnalysisReport:
    """Complete failure analysis report for a dataset."""
    report_id: str
    scene_id: str
    created_at: str

    # Episode stats
    total_episodes: int
    failed_episodes: int
    success_rate: float

    # Failure taxonomy
    taxonomy: FailureTaxonomy

    # Failure events (detailed)
    failure_events: List[FailureEvent] = field(default_factory=list)

    # Aggregated analysis
    failure_distribution: Dict[str, float] = field(default_factory=dict)
    top_failure_mode: Optional[str] = None
    top_failure_count: int = 0

    # Per-phase analysis
    phase_failure_rates: Dict[str, float] = field(default_factory=dict)

    # Per-object analysis (for pick-place)
    object_failure_rates: Dict[str, float] = field(default_factory=dict)

    # Root cause summary
    root_cause_distribution: Dict[str, int] = field(default_factory=dict)

    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

    # Data quality assessment
    data_quality_score: float = 0.0
    recommended_filter_threshold: Optional[float] = None
    episodes_to_filter: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "scene_id": self.scene_id,
            "created_at": self.created_at,
            "summary": {
                "total_episodes": self.total_episodes,
                "failed_episodes": self.failed_episodes,
                "success_rate": f"{self.success_rate:.1%}",
                "total_failure_events": len(self.failure_events),
            },
            "taxonomy": self.taxonomy.to_dict(),
            "failure_distribution": {
                k: f"{v:.1%}" for k, v in self.failure_distribution.items()
            },
            "top_failure_mode": {
                "mode": self.top_failure_mode,
                "count": self.top_failure_count,
                "percentage": f"{self.top_failure_count / max(1, len(self.failure_events)):.1%}",
            },
            "phase_failure_rates": {
                k: f"{v:.1%}" for k, v in self.phase_failure_rates.items()
            },
            "object_failure_rates": {
                k: f"{v:.1%}" for k, v in self.object_failure_rates.items()
            },
            "root_cause_distribution": self.root_cause_distribution,
            "recommendations": self.recommendations,
            "data_quality": {
                "score": self.data_quality_score,
                "filter_threshold": self.recommended_filter_threshold,
                "episodes_to_filter_count": len(self.episodes_to_filter),
            },
        }


class FailureModeAnalyzer:
    """
    Comprehensive failure mode analysis for robotics training data.

    Analyzes episode trajectories to detect, classify, and explain failures.
    Generates actionable insights for robotics labs.
    """

    # Failure detection thresholds
    GRASP_THRESHOLD = 0.02  # meters - gripper < this = closed
    SLIP_VELOCITY_THRESHOLD = 0.1  # m/s - object velocity during grasp
    COLLISION_FORCE_THRESHOLD = 10.0  # N - unexpected contact force
    PLACEMENT_ACCURACY_THRESHOLD = 0.05  # meters
    JOINT_LIMIT_MARGIN = 0.05  # radians from limit
    TIMEOUT_DURATION = 30.0  # seconds per phase

    # Root cause mappings
    ROOT_CAUSE_MAPPINGS = {
        "grasp_miss": [
            "Object pose estimation error",
            "Gripper approach angle incorrect",
            "Object moved during approach",
        ],
        "grasp_slip": [
            "Insufficient gripper force",
            "Object surface too slippery",
            "Grasp point selection suboptimal",
        ],
        "collision_environment": [
            "Motion planning clearance insufficient",
            "Scene geometry mismatch",
            "Obstacle not in scene model",
        ],
        "placement_miss": [
            "Target pose estimation error",
            "Accumulated positioning error",
            "Release timing incorrect",
        ],
        "timeout": [
            "Motion planning taking too long",
            "Policy inference latency",
            "Waiting for stable state",
        ],
    }

    # Recommendations by failure type
    FAILURE_RECOMMENDATIONS = {
        FailureCategory.GRASP_FAILURE: {
            "severity": "high",
            "training_impact": "Filter grasp failures from training data",
            "actions": [
                "Increase gripper force domain randomization",
                "Add more grasp point variation in training",
                "Verify object friction coefficients",
                "Consider tactile sensing for closed-loop grasping",
            ],
        },
        FailureCategory.COLLISION: {
            "severity": "critical",
            "training_impact": "NEVER train on collision episodes",
            "actions": [
                "Increase motion planning clearance margins",
                "Add more obstacle variation during training",
                "Verify collision geometry accuracy",
                "Consider adding collision prediction to policy",
            ],
        },
        FailureCategory.PLACEMENT_ERROR: {
            "severity": "medium",
            "training_impact": "May use for curriculum learning (hard examples)",
            "actions": [
                "Increase placement accuracy domain randomization",
                "Verify target pose estimation pipeline",
                "Consider visual servoing for final placement",
            ],
        },
        FailureCategory.TIMEOUT: {
            "severity": "medium",
            "training_impact": "Filter timeout episodes",
            "actions": [
                "Optimize motion planning parameters",
                "Check for policy inference bottlenecks",
                "Reduce conservatism in motion constraints",
            ],
        },
        FailureCategory.JOINT_LIMIT: {
            "severity": "high",
            "training_impact": "Filter joint limit violations",
            "actions": [
                "Add joint limit awareness to reward function",
                "Verify robot model joint limits match reality",
                "Consider workspace analysis for task feasibility",
            ],
        },
    }

    def __init__(
        self,
        joint_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        verbose: bool = True,
    ):
        self.joint_limits = joint_limits or {}
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[FAILURE-ANALYZER] {msg}")

    def analyze_episode(
        self,
        episode_data: Dict[str, Any],
        episode_id: str,
    ) -> List[FailureEvent]:
        """
        Analyze a single episode for failure events.

        Args:
            episode_data: Episode trajectory data (frames, states, contacts)
            episode_id: Unique episode identifier

        Returns:
            List of detected failure events
        """
        failures = []

        frames = episode_data.get("frames", [])
        if not frames:
            return failures

        # Track state across frames
        prev_gripper_state = None
        grasp_started_frame = None
        object_grasped = False

        for frame_idx, frame in enumerate(frames):
            timestamp = frame.get("timestamp", frame_idx / 30.0)
            phase = frame.get("phase", "unknown")

            # Extract state
            gripper_pos = frame.get("gripper_position", 0.04)
            joint_positions = frame.get("joint_positions", [])
            contacts = frame.get("contacts", [])
            object_poses = frame.get("object_poses", {})

            robot_state = {
                "gripper_position": gripper_pos,
                "joint_positions": joint_positions,
                "ee_position": frame.get("ee_position"),
                "phase": phase,
            }

            # Check for grasp failures
            grasp_failure = self._check_grasp_failure(
                frame, prev_gripper_state, object_grasped, phase
            )
            if grasp_failure:
                failures.append(FailureEvent(
                    event_id=str(uuid.uuid4())[:8],
                    episode_id=episode_id,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    category=FailureCategory.GRASP_FAILURE,
                    subcategory=grasp_failure["type"],
                    severity=FailureSeverity.MAJOR,
                    phase=phase,
                    robot_state=robot_state,
                    contact_info={"contacts": contacts},
                    root_cause=grasp_failure.get("root_cause"),
                    confidence=grasp_failure.get("confidence", 0.8),
                ))

            # Check for collisions
            collision_failure = self._check_collision_failure(
                contacts, phase, object_grasped
            )
            if collision_failure:
                failures.append(FailureEvent(
                    event_id=str(uuid.uuid4())[:8],
                    episode_id=episode_id,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    category=FailureCategory.COLLISION,
                    subcategory=collision_failure["type"],
                    severity=FailureSeverity.CRITICAL if collision_failure["type"] == "self" else FailureSeverity.MAJOR,
                    phase=phase,
                    robot_state=robot_state,
                    contact_info=collision_failure.get("contact_info"),
                    root_cause=collision_failure.get("root_cause"),
                    confidence=collision_failure.get("confidence", 0.9),
                ))

            # Check for joint limit violations
            joint_failure = self._check_joint_limits(joint_positions)
            if joint_failure:
                failures.append(FailureEvent(
                    event_id=str(uuid.uuid4())[:8],
                    episode_id=episode_id,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    category=FailureCategory.JOINT_LIMIT,
                    subcategory="limit_exceeded",
                    severity=FailureSeverity.CRITICAL,
                    phase=phase,
                    robot_state=robot_state,
                    root_cause=f"Joint {joint_failure['joint']} exceeded limit",
                    confidence=1.0,
                ))

            # Update tracking state
            prev_gripper_state = gripper_pos
            if gripper_pos < self.GRASP_THRESHOLD and phase in ["grasp", "lift", "transport"]:
                object_grasped = True

        # Check for placement failure at end
        final_frame = frames[-1] if frames else {}
        if final_frame.get("phase") in ["place", "release", "retract"]:
            placement_failure = self._check_placement_failure(
                episode_data, final_frame
            )
            if placement_failure:
                failures.append(FailureEvent(
                    event_id=str(uuid.uuid4())[:8],
                    episode_id=episode_id,
                    frame_idx=len(frames) - 1,
                    timestamp=final_frame.get("timestamp", 0),
                    category=FailureCategory.PLACEMENT_ERROR,
                    subcategory=placement_failure["type"],
                    severity=FailureSeverity.MAJOR,
                    phase=final_frame.get("phase", "unknown"),
                    robot_state={},
                    root_cause=placement_failure.get("root_cause"),
                    confidence=placement_failure.get("confidence", 0.7),
                ))

        # Check for timeout
        if frames:
            duration = frames[-1].get("timestamp", 0) - frames[0].get("timestamp", 0)
            if duration > self.TIMEOUT_DURATION:
                failures.append(FailureEvent(
                    event_id=str(uuid.uuid4())[:8],
                    episode_id=episode_id,
                    frame_idx=len(frames) - 1,
                    timestamp=duration,
                    category=FailureCategory.TIMEOUT,
                    subcategory="episode_timeout",
                    severity=FailureSeverity.MAJOR,
                    phase="overall",
                    robot_state={},
                    root_cause="Episode exceeded maximum duration",
                    confidence=1.0,
                ))

        return failures

    def _check_grasp_failure(
        self,
        frame: Dict[str, Any],
        prev_gripper: Optional[float],
        object_grasped: bool,
        phase: str,
    ) -> Optional[Dict[str, Any]]:
        """Check for grasp-related failures."""
        gripper_pos = frame.get("gripper_position", 0.04)

        # Grasp miss: Gripper closed but no object contact
        if phase == "grasp" and gripper_pos < self.GRASP_THRESHOLD:
            contacts = frame.get("contacts", [])
            has_object_contact = any(
                "gripper" in str(c.get("body_a", "")).lower() or
                "gripper" in str(c.get("body_b", "")).lower()
                for c in contacts
            )
            if not has_object_contact:
                return {
                    "type": "miss",
                    "root_cause": "Gripper closed without object contact",
                    "confidence": 0.85,
                }

        # Grasp slip: Object was grasped but contact lost
        if object_grasped and phase in ["lift", "transport"]:
            contacts = frame.get("contacts", [])
            gripper_contacts = [
                c for c in contacts
                if "gripper" in str(c.get("body_a", "")).lower() or
                   "gripper" in str(c.get("body_b", "")).lower()
            ]
            if not gripper_contacts and gripper_pos < self.GRASP_THRESHOLD:
                return {
                    "type": "slip",
                    "root_cause": "Object slipped from gripper during manipulation",
                    "confidence": 0.9,
                }

        return None

    def _check_collision_failure(
        self,
        contacts: List[Dict[str, Any]],
        phase: str,
        object_grasped: bool,
    ) -> Optional[Dict[str, Any]]:
        """Check for collision failures."""
        for contact in contacts:
            body_a = str(contact.get("body_a", "")).lower()
            body_b = str(contact.get("body_b", "")).lower()
            force = contact.get("force_magnitude", 0)

            # Expected contacts during grasp
            if object_grasped and "gripper" in body_a or "gripper" in body_b:
                continue

            # Self-collision
            if "link" in body_a and "link" in body_b:
                return {
                    "type": "self",
                    "contact_info": contact,
                    "root_cause": f"Self-collision between {body_a} and {body_b}",
                    "confidence": 1.0,
                }

            # Environment collision with high force
            if force > self.COLLISION_FORCE_THRESHOLD:
                collision_type = "environment"
                if "object" in body_a or "object" in body_b:
                    collision_type = "object"

                return {
                    "type": collision_type,
                    "contact_info": contact,
                    "root_cause": f"Unexpected collision ({force:.1f}N) with {body_a if 'link' not in body_a else body_b}",
                    "confidence": 0.8,
                }

        return None

    def _check_joint_limits(
        self,
        joint_positions: List[float],
    ) -> Optional[Dict[str, Any]]:
        """Check for joint limit violations."""
        if not self.joint_limits or not joint_positions:
            return None

        for i, pos in enumerate(joint_positions):
            joint_name = f"joint_{i}"
            if joint_name in self.joint_limits:
                lower, upper = self.joint_limits[joint_name]
                if pos < lower - self.JOINT_LIMIT_MARGIN:
                    return {"joint": joint_name, "position": pos, "limit": lower}
                if pos > upper + self.JOINT_LIMIT_MARGIN:
                    return {"joint": joint_name, "position": pos, "limit": upper}

        return None

    def _check_placement_failure(
        self,
        episode_data: Dict[str, Any],
        final_frame: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Check for placement failures."""
        # Check if object is at target
        target_pos = episode_data.get("target_position")
        object_poses = final_frame.get("object_poses", {})

        if target_pos and object_poses:
            for obj_id, pose in object_poses.items():
                obj_pos = pose.get("position", [0, 0, 0])
                distance = sum((a - b) ** 2 for a, b in zip(obj_pos, target_pos)) ** 0.5

                if distance > self.PLACEMENT_ACCURACY_THRESHOLD:
                    return {
                        "type": "miss",
                        "root_cause": f"Object {distance:.3f}m from target (threshold: {self.PLACEMENT_ACCURACY_THRESHOLD}m)",
                        "confidence": 0.9,
                    }

        return None

    def analyze_dataset(
        self,
        episodes: List[Dict[str, Any]],
        scene_id: str,
    ) -> FailureAnalysisReport:
        """
        Analyze entire dataset for failure modes.

        Args:
            episodes: List of episode data dictionaries
            scene_id: Scene identifier

        Returns:
            Complete failure analysis report
        """
        self.log(f"Analyzing {len(episodes)} episodes for failure modes...")

        all_failures: List[FailureEvent] = []
        taxonomy = FailureTaxonomy()
        failed_episode_ids = set()

        # Phase and object tracking
        phase_failures: Dict[str, int] = {}
        phase_totals: Dict[str, int] = {}
        object_failures: Dict[str, int] = {}
        object_totals: Dict[str, int] = {}

        for episode in episodes:
            episode_id = episode.get("episode_id", str(uuid.uuid4())[:8])

            # Analyze episode
            failures = self.analyze_episode(episode, episode_id)

            if failures:
                failed_episode_ids.add(episode_id)
                all_failures.extend(failures)

                # Update taxonomy
                for failure in failures:
                    self._update_taxonomy(taxonomy, failure)

                    # Track by phase
                    phase = failure.phase
                    phase_failures[phase] = phase_failures.get(phase, 0) + 1

            # Track phase totals
            for frame in episode.get("frames", []):
                phase = frame.get("phase", "unknown")
                phase_totals[phase] = phase_totals.get(phase, 0) + 1

            # Track object totals
            target_obj = episode.get("target_object", "unknown")
            object_totals[target_obj] = object_totals.get(target_obj, 0) + 1
            if episode_id in failed_episode_ids:
                object_failures[target_obj] = object_failures.get(target_obj, 0) + 1

        # Compute distributions
        total_failures = len(all_failures)
        failure_distribution = {}
        root_cause_distribution = {}

        for failure in all_failures:
            cat = failure.category.value
            failure_distribution[cat] = failure_distribution.get(cat, 0) + 1

            if failure.root_cause:
                root_cause_distribution[failure.root_cause] = \
                    root_cause_distribution.get(failure.root_cause, 0) + 1

        # Normalize distributions
        for k in failure_distribution:
            failure_distribution[k] /= max(1, total_failures)

        # Find top failure mode
        top_mode = max(failure_distribution.items(), key=lambda x: x[1]) if failure_distribution else (None, 0)

        # Compute phase failure rates
        phase_failure_rates = {
            phase: phase_failures.get(phase, 0) / max(1, total)
            for phase, total in phase_totals.items()
        }

        # Compute object failure rates
        object_failure_rates = {
            obj: object_failures.get(obj, 0) / max(1, total)
            for obj, total in object_totals.items()
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(
            taxonomy, failure_distribution, all_failures
        )

        # Compute data quality score
        success_rate = 1 - (len(failed_episode_ids) / max(1, len(episodes)))
        data_quality_score = self._compute_data_quality_score(
            success_rate, taxonomy, len(episodes)
        )

        # Determine episodes to filter
        episodes_to_filter = list(failed_episode_ids)
        filter_threshold = 0.7 if data_quality_score < 0.8 else None

        report = FailureAnalysisReport(
            report_id=str(uuid.uuid4())[:12],
            scene_id=scene_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            total_episodes=len(episodes),
            failed_episodes=len(failed_episode_ids),
            success_rate=success_rate,
            taxonomy=taxonomy,
            failure_events=all_failures,
            failure_distribution=failure_distribution,
            top_failure_mode=top_mode[0],
            top_failure_count=int(top_mode[1] * total_failures) if top_mode[0] else 0,
            phase_failure_rates=phase_failure_rates,
            object_failure_rates=object_failure_rates,
            root_cause_distribution=root_cause_distribution,
            recommendations=recommendations,
            data_quality_score=data_quality_score,
            recommended_filter_threshold=filter_threshold,
            episodes_to_filter=episodes_to_filter,
        )

        self.log(f"Analysis complete: {len(all_failures)} failures in {len(failed_episode_ids)} episodes")
        return report

    def _update_taxonomy(self, taxonomy: FailureTaxonomy, failure: FailureEvent) -> None:
        """Update taxonomy counts based on failure event."""
        cat = failure.category
        sub = failure.subcategory

        if cat == FailureCategory.GRASP_FAILURE:
            if sub == "miss":
                taxonomy.grasp_miss += 1
            elif sub == "slip":
                taxonomy.grasp_slip += 1
            elif sub == "collision":
                taxonomy.grasp_collision += 1
            elif sub == "unstable":
                taxonomy.grasp_unstable += 1
            elif sub == "force_exceeded":
                taxonomy.grasp_force_exceeded += 1

        elif cat == FailureCategory.COLLISION:
            if sub == "environment":
                taxonomy.collision_environment += 1
            elif sub == "object":
                taxonomy.collision_object += 1
            elif sub == "self":
                taxonomy.collision_self += 1
            elif sub == "target":
                taxonomy.collision_target += 1

        elif cat == FailureCategory.PLACEMENT_ERROR:
            if sub == "miss":
                taxonomy.placement_miss += 1
            elif sub == "unstable":
                taxonomy.placement_unstable += 1
            elif sub == "collision":
                taxonomy.placement_collision += 1
            elif sub == "orientation":
                taxonomy.placement_orientation += 1

        elif cat == FailureCategory.JOINT_LIMIT:
            taxonomy.joint_limit_exceeded += 1

        elif cat == FailureCategory.TIMEOUT:
            if "approach" in sub:
                taxonomy.timeout_approach += 1
            elif "grasp" in sub:
                taxonomy.timeout_grasp += 1
            elif "transport" in sub:
                taxonomy.timeout_transport += 1
            elif "place" in sub:
                taxonomy.timeout_place += 1

    def _generate_recommendations(
        self,
        taxonomy: FailureTaxonomy,
        distribution: Dict[str, float],
        failures: List[FailureEvent],
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on failure analysis."""
        recommendations = []

        # Sort by frequency
        sorted_cats = sorted(distribution.items(), key=lambda x: x[1], reverse=True)

        for cat_name, freq in sorted_cats[:3]:  # Top 3 failure modes
            try:
                cat = FailureCategory(cat_name)
            except ValueError:
                continue

            if cat in self.FAILURE_RECOMMENDATIONS:
                rec_template = self.FAILURE_RECOMMENDATIONS[cat]

                # Get specific subcategory recommendations
                cat_failures = [f for f in failures if f.category == cat]
                subcategories = {}
                for f in cat_failures:
                    subcategories[f.subcategory] = subcategories.get(f.subcategory, 0) + 1

                recommendations.append({
                    "failure_category": cat_name,
                    "frequency": f"{freq:.1%}",
                    "severity": rec_template["severity"],
                    "training_impact": rec_template["training_impact"],
                    "actions": rec_template["actions"],
                    "subcategory_breakdown": subcategories,
                })

        # Add general recommendations
        total_failures = taxonomy.total_failures()
        if total_failures > 0:
            if taxonomy.collision_self > 0:
                recommendations.append({
                    "priority": "CRITICAL",
                    "message": f"Found {taxonomy.collision_self} self-collision events - these episodes MUST be filtered",
                    "action": "Add self-collision check to episode validation",
                })

            if taxonomy.joint_limit_exceeded > 5:
                recommendations.append({
                    "priority": "HIGH",
                    "message": f"Found {taxonomy.joint_limit_exceeded} joint limit violations",
                    "action": "Verify robot joint limits match Isaac Sim configuration",
                })

        return recommendations

    def _compute_data_quality_score(
        self,
        success_rate: float,
        taxonomy: FailureTaxonomy,
        total_episodes: int,
    ) -> float:
        """Compute overall data quality score (0-1)."""
        score = success_rate  # Base score is success rate

        # Penalties for critical failures
        if taxonomy.collision_self > 0:
            score -= 0.1  # Self-collisions are very bad

        if taxonomy.physics_penetration > 0:
            score -= 0.05

        # Bonus for having enough data even with some failures
        if total_episodes > 100:
            score += 0.05

        return max(0.0, min(1.0, score))

    def save_report(
        self,
        report: FailureAnalysisReport,
        output_path: Path,
    ) -> Path:
        """Save failure analysis report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        self.log(f"Saved failure analysis report to {output_path}")
        return output_path


def analyze_episode_failures(
    episodes_dir: Path,
    scene_id: str,
    output_dir: Optional[Path] = None,
) -> FailureAnalysisReport:
    """
    Convenience function to analyze failures in an episodes directory.

    Args:
        episodes_dir: Path to episodes directory
        scene_id: Scene identifier
        output_dir: Optional output directory for report

    Returns:
        FailureAnalysisReport
    """
    episodes_dir = Path(episodes_dir)

    # Load episodes from parquet or metadata
    episodes = []

    # Try to load episode metadata
    meta_dir = episodes_dir / "meta"
    if meta_dir.exists():
        episodes_file = meta_dir / "episodes.jsonl"
        if episodes_file.exists():
            with open(episodes_file) as f:
                for line in f:
                    try:
                        ep = json.loads(line)
                        episodes.append(ep)
                    except json.JSONDecodeError:
                        continue

    # Also check for individual episode JSON files
    for ep_file in episodes_dir.glob("episode_*.json"):
        try:
            with open(ep_file) as f:
                episodes.append(json.load(f))
        except (json.JSONDecodeError, IOError):
            continue

    if not episodes:
        # Create placeholder episode data
        print(f"[FAILURE-ANALYZER] No episode data found in {episodes_dir}")
        episodes = [{"episode_id": "placeholder", "frames": []}]

    analyzer = FailureModeAnalyzer(verbose=True)
    report = analyzer.analyze_dataset(episodes, scene_id)

    if output_dir:
        output_path = Path(output_dir) / "failure_analysis_report.json"
        analyzer.save_report(report, output_path)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze episode failures")
    parser.add_argument("episodes_dir", type=Path, help="Path to episodes directory")
    parser.add_argument("--scene-id", required=True, help="Scene identifier")
    parser.add_argument("--output-dir", type=Path, help="Output directory")

    args = parser.parse_args()

    report = analyze_episode_failures(
        episodes_dir=args.episodes_dir,
        scene_id=args.scene_id,
        output_dir=args.output_dir,
    )

    print(f"\n=== Failure Analysis Summary ===")
    print(f"Total Episodes: {report.total_episodes}")
    print(f"Failed Episodes: {report.failed_episodes}")
    print(f"Success Rate: {report.success_rate:.1%}")
    print(f"Top Failure Mode: {report.top_failure_mode}")
    print(f"Data Quality Score: {report.data_quality_score:.2f}")
