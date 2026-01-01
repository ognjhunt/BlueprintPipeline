"""Quality Gate Framework.

Defines quality gates at key pipeline checkpoints with severity levels
and human-in-the-loop validation support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .notification_service import NotificationService, NotificationPayload


class QualityGateSeverity(str, Enum):
    """Quality gate severity levels."""
    ERROR = "error"      # Blocks pipeline, requires human approval to continue
    WARNING = "warning"  # Flags for review but doesn't block
    INFO = "info"        # Informational only


class QualityGateCheckpoint(str, Enum):
    """Pipeline checkpoints where quality gates are evaluated."""
    # 3D Reconstruction
    RECONSTRUCTION_COMPLETE = "reconstruction_complete"

    # USD Assembly Pipeline
    MANIFEST_VALIDATED = "manifest_validated"
    SIMREADY_COMPLETE = "simready_complete"
    USD_ASSEMBLED = "usd_assembled"
    REPLICATOR_COMPLETE = "replicator_complete"
    ISAAC_LAB_GENERATED = "isaac_lab_generated"

    # Episode Generation Pipeline
    PRE_EPISODE_VALIDATION = "pre_episode_validation"
    EPISODES_GENERATED = "episodes_generated"

    # DWM Pipeline
    DWM_PREPARED = "dwm_prepared"
    DWM_INFERENCE_COMPLETE = "dwm_inference_complete"

    # Final Delivery
    SCENE_READY = "scene_ready"
    DELIVERED = "delivered"


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_id: str
    checkpoint: QualityGateCheckpoint
    passed: bool
    severity: QualityGateSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = ""
    requires_human_approval: bool = False

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

        # Auto-set human approval requirement based on severity
        if not self.passed and self.severity == QualityGateSeverity.ERROR:
            self.requires_human_approval = True


@dataclass
class QualityGate:
    """Definition of a quality gate check."""
    id: str
    name: str
    checkpoint: QualityGateCheckpoint
    severity: QualityGateSeverity
    description: str
    check_fn: Optional[Callable[[Dict[str, Any]], QualityGateResult]] = None

    # Notification settings
    notify_on_pass: bool = False
    notify_on_fail: bool = True
    require_human_approval: bool = False  # Always require human approval

    def check(self, context: Dict[str, Any]) -> QualityGateResult:
        """Run the quality gate check."""
        if self.check_fn is None:
            # Default pass if no check function
            return QualityGateResult(
                gate_id=self.id,
                checkpoint=self.checkpoint,
                passed=True,
                severity=self.severity,
                message=f"{self.name}: No check function defined (auto-pass)",
            )

        result = self.check_fn(context)

        # Override human approval if gate requires it
        if self.require_human_approval:
            result.requires_human_approval = True

        return result


class QualityGateRegistry:
    """Registry and executor for quality gates."""

    def __init__(self, verbose: bool = True):
        self.gates: Dict[str, QualityGate] = {}
        self.results: List[QualityGateResult] = []
        self.verbose = verbose

        # Register default gates
        self._register_default_gates()

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[QUALITY-GATE] {msg}")

    def register(self, gate: QualityGate) -> None:
        """Register a quality gate."""
        self.gates[gate.id] = gate
        self.log(f"Registered gate: {gate.id}")

    def get_gates_for_checkpoint(
        self, checkpoint: QualityGateCheckpoint
    ) -> List[QualityGate]:
        """Get all gates for a checkpoint."""
        return [g for g in self.gates.values() if g.checkpoint == checkpoint]

    def run_checkpoint(
        self,
        checkpoint: QualityGateCheckpoint,
        context: Dict[str, Any],
        notification_service: Optional["NotificationService"] = None,
    ) -> List[QualityGateResult]:
        """Run all gates for a checkpoint.

        Args:
            checkpoint: The checkpoint to evaluate
            context: Context data for gate checks
            notification_service: Optional notification service for alerts

        Returns:
            List of QualityGateResult
        """
        gates = self.get_gates_for_checkpoint(checkpoint)
        results = []

        self.log(f"Running {len(gates)} gates for checkpoint: {checkpoint.value}")

        for gate in gates:
            try:
                result = gate.check(context)
            except Exception as e:
                result = QualityGateResult(
                    gate_id=gate.id,
                    checkpoint=checkpoint,
                    passed=False,
                    severity=gate.severity,
                    message=f"{gate.name}: Check failed with error: {e}",
                    requires_human_approval=True,
                )

            results.append(result)
            self.results.append(result)

            # Log result
            status = "PASSED" if result.passed else "FAILED"
            self.log(f"  [{status}] {gate.name}: {result.message}")

            # Send notification if needed
            if notification_service:
                should_notify = (
                    (not result.passed and gate.notify_on_fail) or
                    (result.passed and gate.notify_on_pass)
                )

                if should_notify:
                    self._send_notification(
                        gate, result, context, notification_service
                    )

        return results

    def get_blocking_failures(self) -> List[QualityGateResult]:
        """Get all blocking (ERROR severity) failures."""
        return [
            r for r in self.results
            if not r.passed and r.severity == QualityGateSeverity.ERROR
        ]

    def can_proceed(self) -> bool:
        """Check if pipeline can proceed (no blocking failures)."""
        return len(self.get_blocking_failures()) == 0

    def to_report(self, scene_id: str) -> Dict[str, Any]:
        """Generate a validation report."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        blocking = len(self.get_blocking_failures())

        return {
            "scene_id": scene_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_gates": len(self.results),
                "passed": passed,
                "failed": failed,
                "blocking_failures": blocking,
                "can_proceed": self.can_proceed(),
            },
            "results": [
                {
                    "gate_id": r.gate_id,
                    "checkpoint": r.checkpoint.value,
                    "passed": r.passed,
                    "severity": r.severity.value,
                    "message": r.message,
                    "details": r.details,
                    "recommendations": r.recommendations,
                    "requires_human_approval": r.requires_human_approval,
                    "timestamp": r.timestamp,
                }
                for r in self.results
            ],
        }

    def save_report(self, scene_id: str, output_path: Path) -> None:
        """Save validation report to file."""
        report = self.to_report(scene_id)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        self.log(f"Report saved to: {output_path}")

    def _send_notification(
        self,
        gate: QualityGate,
        result: QualityGateResult,
        context: Dict[str, Any],
        notification_service: "NotificationService",
    ) -> None:
        """Send notification for a gate result."""
        from .notification_service import NotificationPayload

        scene_id = context.get("scene_id", "unknown")

        payload = NotificationPayload(
            subject=f"{'PASSED' if result.passed else 'FAILED'}: {gate.name}",
            body=result.message,
            scene_id=scene_id,
            checkpoint=result.checkpoint.value,
            severity=result.severity.value,
            qa_context={
                "Gate": gate.name,
                "Description": gate.description,
                "Recommendations": result.recommendations or ["No specific recommendations"],
                "Details": result.details,
            },
            action_required=result.requires_human_approval,
        )

        notification_service.send(payload)

    def _register_default_gates(self) -> None:
        """Register default quality gates for the pipeline."""

        # QG-1: Manifest Validation
        def check_manifest(ctx: Dict[str, Any]) -> QualityGateResult:
            manifest = ctx.get("manifest", {})
            objects = manifest.get("objects", [])

            issues = []
            if not objects:
                issues.append("No objects in manifest")

            for obj in objects:
                if not obj.get("id"):
                    issues.append(f"Object missing ID")
                if not obj.get("asset", {}).get("path"):
                    issues.append(f"Object {obj.get('id', 'unknown')} missing asset path")

            passed = len(issues) == 0

            return QualityGateResult(
                gate_id="qg-1-manifest",
                checkpoint=QualityGateCheckpoint.MANIFEST_VALIDATED,
                passed=passed,
                severity=QualityGateSeverity.ERROR,
                message=f"Manifest validation: {len(objects)} objects, {len(issues)} issues",
                details={"object_count": len(objects), "issues": issues},
                recommendations=[
                    "Ensure all objects have unique IDs",
                    "Verify all asset paths point to valid files",
                ] if not passed else [],
            )

        self.register(QualityGate(
            id="qg-1-manifest",
            name="Manifest Integrity",
            checkpoint=QualityGateCheckpoint.MANIFEST_VALIDATED,
            severity=QualityGateSeverity.ERROR,
            description="Validates scene manifest schema, object count, and required fields",
            check_fn=check_manifest,
            notify_on_fail=True,
        ))

        # QG-2: Physics Plausibility
        def check_physics(ctx: Dict[str, Any]) -> QualityGateResult:
            physics_data = ctx.get("physics_properties", {})
            objects = physics_data.get("objects", [])

            warnings = []
            for obj in objects:
                obj_id = obj.get("id", "unknown")
                mass = obj.get("mass", 0)
                friction = obj.get("friction", 0)

                if mass < 0.01 or mass > 500:
                    warnings.append(f"{obj_id}: Mass {mass}kg outside expected range [0.01, 500]")
                if friction < 0 or friction > 2.0:
                    warnings.append(f"{obj_id}: Friction {friction} outside expected range [0, 2.0]")

            passed = len(warnings) == 0

            return QualityGateResult(
                gate_id="qg-2-physics",
                checkpoint=QualityGateCheckpoint.SIMREADY_COMPLETE,
                passed=passed,
                severity=QualityGateSeverity.WARNING,
                message=f"Physics plausibility: {len(warnings)} warnings",
                details={"warnings": warnings},
                recommendations=[
                    "Review mass estimates for realism",
                    "Verify friction coefficients match expected materials",
                    "Consider measuring real-world properties for critical objects",
                ] if not passed else [],
            )

        self.register(QualityGate(
            id="qg-2-physics",
            name="Physics Plausibility",
            checkpoint=QualityGateCheckpoint.SIMREADY_COMPLETE,
            severity=QualityGateSeverity.WARNING,
            description="Checks mass, friction, and other physics properties are within realistic ranges",
            check_fn=check_physics,
            notify_on_fail=True,
        ))

        # QG-3: USD Validity
        def check_usd(ctx: Dict[str, Any]) -> QualityGateResult:
            usd_path = ctx.get("usd_path")
            issues = []

            if not usd_path or not Path(usd_path).is_file():
                return QualityGateResult(
                    gate_id="qg-3-usd",
                    checkpoint=QualityGateCheckpoint.USD_ASSEMBLED,
                    passed=False,
                    severity=QualityGateSeverity.ERROR,
                    message="USD scene file not found",
                    recommendations=["Check USD assembly job completed successfully"],
                )

            usd_content = Path(usd_path).read_text()

            # Check header
            if not usd_content.startswith("#usda 1.0"):
                issues.append("Invalid USD header")

            # Check for critical components
            if "PhysicsScene" not in usd_content:
                issues.append("No PhysicsScene defined")

            # Check for broken references
            import re
            refs = re.findall(r'references\s*=\s*@([^@]+)@', usd_content)
            usd_dir = Path(usd_path).parent
            for ref in refs:
                if ref.startswith("./") or ref.startswith("../"):
                    ref_path = (usd_dir / ref).resolve()
                    if not ref_path.is_file():
                        issues.append(f"Missing reference: {ref}")

            passed = len(issues) == 0

            return QualityGateResult(
                gate_id="qg-3-usd",
                checkpoint=QualityGateCheckpoint.USD_ASSEMBLED,
                passed=passed,
                severity=QualityGateSeverity.ERROR,
                message=f"USD validation: {len(issues)} issues" if issues else "USD validated successfully",
                details={"issues": issues, "reference_count": len(refs)},
                recommendations=[
                    "Fix broken references before proceeding",
                    "Ensure PhysicsScene is properly defined",
                ] if not passed else [],
            )

        self.register(QualityGate(
            id="qg-3-usd",
            name="USD Validity",
            checkpoint=QualityGateCheckpoint.USD_ASSEMBLED,
            severity=QualityGateSeverity.ERROR,
            description="Validates USD scene is parseable and all references are valid",
            check_fn=check_usd,
            notify_on_fail=True,
            require_human_approval=True,  # Critical gate
        ))

        # QG-4: Isaac Lab Code Generation
        def check_isaac_lab(ctx: Dict[str, Any]) -> QualityGateResult:
            isaac_lab_dir = ctx.get("isaac_lab_dir")
            issues = []

            if not isaac_lab_dir or not Path(isaac_lab_dir).is_dir():
                return QualityGateResult(
                    gate_id="qg-4-isaac-lab",
                    checkpoint=QualityGateCheckpoint.ISAAC_LAB_GENERATED,
                    passed=False,
                    severity=QualityGateSeverity.WARNING,
                    message="Isaac Lab directory not found",
                    recommendations=["Check Isaac Lab task generation completed"],
                )

            isaac_lab_dir = Path(isaac_lab_dir)

            # Check required files
            required = ["env_cfg.py"]
            for fname in required:
                if not (isaac_lab_dir / fname).is_file():
                    issues.append(f"Missing: {fname}")

            # Syntax check
            import ast
            for py_file in isaac_lab_dir.glob("*.py"):
                try:
                    ast.parse(py_file.read_text())
                except SyntaxError as e:
                    issues.append(f"Syntax error in {py_file.name}: {e}")

            # Check for task files
            task_files = list(isaac_lab_dir.glob("task_*.py"))
            if not task_files:
                issues.append("No task files generated")

            passed = len(issues) == 0

            return QualityGateResult(
                gate_id="qg-4-isaac-lab",
                checkpoint=QualityGateCheckpoint.ISAAC_LAB_GENERATED,
                passed=passed,
                severity=QualityGateSeverity.WARNING,
                message=f"Isaac Lab validation: {len(task_files)} tasks, {len(issues)} issues",
                details={"task_count": len(task_files), "issues": issues},
                recommendations=[
                    "Fix syntax errors in generated code",
                    "Verify generated tasks match environment policies",
                ] if not passed else [],
            )

        self.register(QualityGate(
            id="qg-4-isaac-lab",
            name="Isaac Lab Code Validation",
            checkpoint=QualityGateCheckpoint.ISAAC_LAB_GENERATED,
            severity=QualityGateSeverity.WARNING,
            description="Validates generated Isaac Lab code syntax and structure",
            check_fn=check_isaac_lab,
            notify_on_fail=True,
        ))

        # QG-5: Pre-Episode Simulation Check
        def check_pre_episode(ctx: Dict[str, Any]) -> QualityGateResult:
            sim_check = ctx.get("simulation_check", {})

            if not sim_check:
                return QualityGateResult(
                    gate_id="qg-5-pre-episode",
                    checkpoint=QualityGateCheckpoint.PRE_EPISODE_VALIDATION,
                    passed=False,
                    severity=QualityGateSeverity.ERROR,
                    message="Simulation pre-check not performed",
                    recommendations=[
                        "Run scene load test in Isaac Sim",
                        "Verify physics simulation is stable",
                    ],
                    requires_human_approval=True,
                )

            scene_loads = sim_check.get("scene_loads", False)
            physics_stable = sim_check.get("physics_stable", False)
            steps_completed = sim_check.get("steps_completed", 0)

            passed = scene_loads and physics_stable and steps_completed >= 10

            return QualityGateResult(
                gate_id="qg-5-pre-episode",
                checkpoint=QualityGateCheckpoint.PRE_EPISODE_VALIDATION,
                passed=passed,
                severity=QualityGateSeverity.ERROR,
                message=f"Simulation check: loads={scene_loads}, stable={physics_stable}, steps={steps_completed}",
                details=sim_check,
                recommendations=[
                    "Check for floating/exploding objects" if not physics_stable else "",
                    "Verify USD scene loads without errors" if not scene_loads else "",
                ] if not passed else [],
                requires_human_approval=True,
            )

        self.register(QualityGate(
            id="qg-5-pre-episode",
            name="Pre-Episode Simulation Check",
            checkpoint=QualityGateCheckpoint.PRE_EPISODE_VALIDATION,
            severity=QualityGateSeverity.ERROR,
            description="Verifies scene loads in simulator and physics is stable",
            check_fn=check_pre_episode,
            notify_on_fail=True,
            require_human_approval=True,
        ))

        # QG-6: Episode Quality
        def check_episodes(ctx: Dict[str, Any]) -> QualityGateResult:
            episode_stats = ctx.get("episode_stats", {})

            total = episode_stats.get("total_generated", 0)
            passed_quality = episode_stats.get("passed_quality_filter", 0)
            avg_quality = episode_stats.get("average_quality_score", 0)
            collision_free = episode_stats.get("collision_free_rate", 0)

            issues = []
            if total == 0:
                issues.append("No episodes generated")
            if passed_quality < total * 0.5:
                issues.append(f"Low quality pass rate: {passed_quality}/{total}")
            if collision_free < 0.8:
                issues.append(f"Low collision-free rate: {collision_free:.1%}")

            passed = len(issues) == 0 and total > 0

            return QualityGateResult(
                gate_id="qg-6-episodes",
                checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
                passed=passed,
                severity=QualityGateSeverity.WARNING,
                message=f"Episodes: {passed_quality}/{total} passed (avg quality: {avg_quality:.2f})",
                details=episode_stats,
                recommendations=[
                    "Review failed episodes for common patterns",
                    "Consider adjusting MIN_QUALITY_SCORE threshold",
                    "Check collision detection parameters",
                ] if not passed else [],
            )

        self.register(QualityGate(
            id="qg-6-episodes",
            name="Episode Quality Filter",
            checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
            severity=QualityGateSeverity.WARNING,
            description="Validates generated episodes meet quality thresholds",
            check_fn=check_episodes,
            notify_on_fail=True,
            notify_on_pass=True,  # Also notify on success for this one
        ))

        # QG-7: Scene Ready (Final Gate)
        def check_scene_ready(ctx: Dict[str, Any]) -> QualityGateResult:
            checklist = ctx.get("readiness_checklist", {})

            required = [
                ("usd_valid", "USD scene is valid"),
                ("physics_stable", "Physics simulation stable"),
                ("episodes_generated", "Training episodes available"),
            ]

            optional = [
                ("replicator_ready", "Replicator bundle available"),
                ("isaac_lab_ready", "Isaac Lab tasks generated"),
                ("dwm_ready", "DWM inference complete"),
            ]

            missing_required = []
            missing_optional = []

            for key, desc in required:
                if not checklist.get(key, False):
                    missing_required.append(desc)

            for key, desc in optional:
                if not checklist.get(key, False):
                    missing_optional.append(desc)

            passed = len(missing_required) == 0

            return QualityGateResult(
                gate_id="qg-7-ready",
                checkpoint=QualityGateCheckpoint.SCENE_READY,
                passed=passed,
                severity=QualityGateSeverity.ERROR,
                message=f"Scene readiness: {len(missing_required)} required missing, {len(missing_optional)} optional missing",
                details={
                    "missing_required": missing_required,
                    "missing_optional": missing_optional,
                    "checklist": checklist,
                },
                recommendations=missing_required + missing_optional if not passed else [],
                requires_human_approval=True,
            )

        self.register(QualityGate(
            id="qg-7-ready",
            name="Scene Readiness Check",
            checkpoint=QualityGateCheckpoint.SCENE_READY,
            severity=QualityGateSeverity.ERROR,
            description="Final validation before scene delivery",
            check_fn=check_scene_ready,
            notify_on_fail=True,
            notify_on_pass=True,
            require_human_approval=True,
        ))
