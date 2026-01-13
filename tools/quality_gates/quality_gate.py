"""Quality Gate Framework.

Defines quality gates at key pipeline checkpoints with severity levels
and human-in-the-loop validation support.

LABS P1 FIX: This module now supports:
- Configurable thresholds via quality_config.json
- Human approval workflow for blocked gates
- Manual override with audit logging
- Environment variable overrides (BP_QUALITY_*)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .notification_service import NotificationService, NotificationPayload

# Import configuration
try:
    from tools.config import load_quality_config, QualityConfig
    HAVE_CONFIG = True
except ImportError:
    HAVE_CONFIG = False
    QualityConfig = None


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


# =============================================================================
# LABS P1 FIX: Human Approval Workflow
# =============================================================================


class ApprovalStatus(str, Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    OVERRIDDEN = "overridden"
    EXPIRED = "expired"


@dataclass
class ApprovalRequest:
    """Request for human approval of a failed quality gate."""
    request_id: str
    gate_id: str
    scene_id: str
    checkpoint: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    failure_message: str = ""
    failure_details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    created_at: str = ""
    expires_at: str = ""
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    override_reason: Optional[str] = None
    approval_notes: Optional[str] = None

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"apr_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "gate_id": self.gate_id,
            "scene_id": self.scene_id,
            "checkpoint": self.checkpoint,
            "status": self.status.value,
            "failure_message": self.failure_message,
            "failure_details": self.failure_details,
            "recommendations": self.recommendations,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "override_reason": self.override_reason,
            "approval_notes": self.approval_notes,
        }


class HumanApprovalManager:
    """
    LABS P1 FIX: Manager for human-in-the-loop approval workflow.

    Handles:
    - Creating approval requests for failed gates
    - Storing pending approvals
    - Processing approvals/rejections
    - Override with audit logging
    - Timeout handling

    Usage:
        manager = HumanApprovalManager(scene_id="kitchen_001")

        # Create approval request for a failed gate
        request = manager.create_request(gate_result)

        # Wait for approval (blocking)
        approved = manager.wait_for_approval(request, timeout_hours=24)

        # Or check status non-blocking
        status = manager.check_status(request.request_id)

        # Process approval via API/dashboard
        manager.approve(request.request_id, approver="lab_admin@example.com")

        # Or override with reason
        manager.override(request.request_id, reason="Known issue, safe to proceed",
                        approver="admin@example.com")
    """

    # Storage directory for pending approvals
    # Can be overridden via BP_APPROVALS_DIR environment variable
    APPROVALS_DIR = Path(os.getenv("BP_APPROVALS_DIR", "/tmp/blueprintpipeline/approvals"))

    def __init__(
        self,
        scene_id: str,
        config: Optional["QualityConfig"] = None,
        notification_service: Optional["NotificationService"] = None,
        verbose: bool = True,
    ):
        self.scene_id = scene_id
        self.verbose = verbose
        self.notification_service = notification_service

        # Load config
        if config:
            self.config = config
        elif HAVE_CONFIG:
            self.config = load_quality_config()
        else:
            self.config = None

        # Get settings from config
        if self.config and hasattr(self.config, 'human_approval'):
            self.timeout_hours = self.config.human_approval.timeout_hours
            self.auto_approve_on_timeout = self.config.human_approval.auto_approve_on_timeout
        else:
            self.timeout_hours = float(os.getenv("APPROVAL_TIMEOUT_HOURS", "24"))
            self.auto_approve_on_timeout = os.getenv("AUTO_APPROVE_ON_TIMEOUT", "false").lower() == "true"

        # Ensure approvals directory exists
        self.approvals_dir = self.APPROVALS_DIR / scene_id
        self.approvals_dir.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[APPROVAL] [{level}] {msg}")

    def create_request(self, gate_result: QualityGateResult) -> ApprovalRequest:
        """Create an approval request for a failed quality gate.

        Args:
            gate_result: The failed QualityGateResult

        Returns:
            ApprovalRequest that can be tracked and approved
        """
        expires_at = datetime.utcnow() + timedelta(hours=self.timeout_hours)

        request = ApprovalRequest(
            request_id=f"apr_{uuid.uuid4().hex[:12]}",
            gate_id=gate_result.gate_id,
            scene_id=self.scene_id,
            checkpoint=gate_result.checkpoint.value,
            failure_message=gate_result.message,
            failure_details=gate_result.details,
            recommendations=gate_result.recommendations,
            expires_at=expires_at.isoformat() + "Z",
        )

        # Save to disk
        self._save_request(request)

        self.log(f"Created approval request {request.request_id} for gate {gate_result.gate_id}")

        # Send notification if service available
        if self.notification_service:
            self._send_notification(request)

        return request

    def check_status(self, request_id: str) -> Optional[ApprovalRequest]:
        """Check the current status of an approval request.

        Args:
            request_id: The approval request ID

        Returns:
            ApprovalRequest with current status, or None if not found
        """
        request = self._load_request(request_id)

        if request and request.status == ApprovalStatus.PENDING:
            # Check if expired
            if request.expires_at:
                expires = datetime.fromisoformat(request.expires_at.replace("Z", "+00:00"))
                if datetime.now(expires.tzinfo) > expires:
                    if self.auto_approve_on_timeout:
                        request.status = ApprovalStatus.APPROVED
                        request.approved_by = "SYSTEM:auto_timeout"
                        request.approved_at = datetime.utcnow().isoformat() + "Z"
                        request.approval_notes = f"Auto-approved after {self.timeout_hours}h timeout"
                    else:
                        request.status = ApprovalStatus.EXPIRED
                    self._save_request(request)

        return request

    def wait_for_approval(
        self,
        request: ApprovalRequest,
        timeout_hours: Optional[float] = None,
        poll_interval_seconds: float = 30.0,
    ) -> bool:
        """Wait for an approval request to be processed (blocking).

        Args:
            request: The ApprovalRequest to wait for
            timeout_hours: Override timeout (default: use config)
            poll_interval_seconds: How often to check status

        Returns:
            True if approved/overridden, False if rejected/expired
        """
        timeout = timeout_hours or self.timeout_hours
        start_time = time.time()
        max_wait_seconds = timeout * 3600

        self.log(f"Waiting for approval of {request.request_id} (timeout: {timeout}h)")
        self.log(f"Approval can be submitted via dashboard, email, or API")

        while True:
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed >= max_wait_seconds:
                self.log(f"Approval request {request.request_id} timed out", "WARNING")
                current = self.check_status(request.request_id)
                if current:
                    return current.status in (ApprovalStatus.APPROVED, ApprovalStatus.OVERRIDDEN)
                return False

            # Check status
            current = self.check_status(request.request_id)

            if not current:
                self.log(f"Approval request {request.request_id} not found", "ERROR")
                return False

            if current.status == ApprovalStatus.APPROVED:
                self.log(f"Request {request.request_id} APPROVED by {current.approved_by}")
                return True

            if current.status == ApprovalStatus.OVERRIDDEN:
                self.log(f"Request {request.request_id} OVERRIDDEN by {current.approved_by}")
                self.log(f"Override reason: {current.override_reason}")
                return True

            if current.status == ApprovalStatus.REJECTED:
                self.log(f"Request {request.request_id} REJECTED by {current.approved_by}")
                return False

            if current.status == ApprovalStatus.EXPIRED:
                self.log(f"Request {request.request_id} EXPIRED")
                return False

            # Still pending - wait and poll again
            time.sleep(poll_interval_seconds)

    def approve(
        self,
        request_id: str,
        approver: str,
        notes: Optional[str] = None,
    ) -> bool:
        """Approve a pending approval request.

        Args:
            request_id: The approval request ID
            approver: Email or ID of the approver
            notes: Optional approval notes

        Returns:
            True if successfully approved
        """
        request = self._load_request(request_id)

        if not request:
            self.log(f"Approval request {request_id} not found", "ERROR")
            return False

        if request.status != ApprovalStatus.PENDING:
            self.log(f"Cannot approve {request_id}: status is {request.status.value}", "WARNING")
            return False

        request.status = ApprovalStatus.APPROVED
        request.approved_by = approver
        request.approved_at = datetime.utcnow().isoformat() + "Z"
        request.approval_notes = notes

        self._save_request(request)
        self._log_audit("APPROVED", request, approver)

        self.log(f"Request {request_id} approved by {approver}")
        return True

    def reject(
        self,
        request_id: str,
        approver: str,
        notes: Optional[str] = None,
    ) -> bool:
        """Reject a pending approval request.

        Args:
            request_id: The approval request ID
            approver: Email or ID of the rejector
            notes: Optional rejection notes

        Returns:
            True if successfully rejected
        """
        request = self._load_request(request_id)

        if not request:
            self.log(f"Approval request {request_id} not found", "ERROR")
            return False

        if request.status != ApprovalStatus.PENDING:
            self.log(f"Cannot reject {request_id}: status is {request.status.value}", "WARNING")
            return False

        request.status = ApprovalStatus.REJECTED
        request.approved_by = approver
        request.approved_at = datetime.utcnow().isoformat() + "Z"
        request.approval_notes = notes

        self._save_request(request)
        self._log_audit("REJECTED", request, approver)

        self.log(f"Request {request_id} rejected by {approver}")
        return True

    def override(
        self,
        request_id: str,
        reason: str,
        approver: str,
    ) -> bool:
        """Override a pending approval request (bypass with audit).

        LABS P1 FIX: Allows labs to proceed past failed gates with documented reason.

        Args:
            request_id: The approval request ID
            reason: Reason for override (REQUIRED for audit)
            approver: Email or ID of the overrider

        Returns:
            True if successfully overridden
        """
        if not reason or len(reason.strip()) < 10:
            self.log("Override requires a reason (min 10 characters)", "ERROR")
            return False

        request = self._load_request(request_id)

        if not request:
            self.log(f"Approval request {request_id} not found", "ERROR")
            return False

        if request.status not in (ApprovalStatus.PENDING, ApprovalStatus.EXPIRED):
            self.log(f"Cannot override {request_id}: status is {request.status.value}", "WARNING")
            return False

        request.status = ApprovalStatus.OVERRIDDEN
        request.approved_by = approver
        request.approved_at = datetime.utcnow().isoformat() + "Z"
        request.override_reason = reason

        self._save_request(request)
        self._log_audit("OVERRIDDEN", request, approver, reason)

        self.log(f"Request {request_id} overridden by {approver}: {reason}")
        return True

    def list_pending(self) -> List[ApprovalRequest]:
        """List all pending approval requests for this scene."""
        requests = []
        for path in self.approvals_dir.glob("*.json"):
            request = self._load_request(path.stem)
            if request and request.status == ApprovalStatus.PENDING:
                requests.append(request)
        return requests

    def _save_request(self, request: ApprovalRequest) -> None:
        """Save approval request to disk."""
        path = self.approvals_dir / f"{request.request_id}.json"
        with open(path, "w") as f:
            json.dump(request.to_dict(), f, indent=2)

    def _load_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Load approval request from disk."""
        path = self.approvals_dir / f"{request_id}.json"
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        return ApprovalRequest(
            request_id=data["request_id"],
            gate_id=data["gate_id"],
            scene_id=data["scene_id"],
            checkpoint=data["checkpoint"],
            status=ApprovalStatus(data["status"]),
            failure_message=data.get("failure_message", ""),
            failure_details=data.get("failure_details", {}),
            recommendations=data.get("recommendations", []),
            created_at=data.get("created_at", ""),
            expires_at=data.get("expires_at", ""),
            approved_by=data.get("approved_by"),
            approved_at=data.get("approved_at"),
            override_reason=data.get("override_reason"),
            approval_notes=data.get("approval_notes"),
        )

    def _send_notification(self, request: ApprovalRequest) -> None:
        """Send notification for new approval request."""
        if not self.notification_service:
            return

        from .notification_service import NotificationPayload

        payload = NotificationPayload(
            subject=f"Approval Required: {request.gate_id}",
            body=f"Quality gate '{request.gate_id}' failed and requires approval.\n\n{request.failure_message}",
            scene_id=request.scene_id,
            checkpoint=request.checkpoint,
            severity="error",
            qa_context={
                "Request ID": request.request_id,
                "Gate": request.gate_id,
                "Expires": request.expires_at,
                "Recommendations": request.recommendations,
            },
            action_required=True,
            dashboard_url=f"/approvals/{request.request_id}",
        )

        self.notification_service.send(payload)

    def _log_audit(
        self,
        action: str,
        request: ApprovalRequest,
        actor: str,
        reason: Optional[str] = None,
    ) -> None:
        """Log audit trail for approval actions."""
        audit_dir = self.approvals_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)

        audit_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "request_id": request.request_id,
            "gate_id": request.gate_id,
            "scene_id": request.scene_id,
            "actor": actor,
            "reason": reason,
        }

        audit_path = audit_dir / f"{request.request_id}_{action.lower()}.json"
        with open(audit_path, "w") as f:
            json.dump(audit_entry, f, indent=2)


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
    """Registry and executor for quality gates.

    LABS P1 FIX: Now supports configurable thresholds via quality_config.json
    and integrates with HumanApprovalManager for blocked gates.

    Usage:
        # Use default config
        registry = QualityGateRegistry()

        # Use custom thresholds
        registry = QualityGateRegistry(
            config_overrides={"episodes": {"collision_free_rate_min": 0.7}}
        )

        # Run checkpoint with approval handling
        results = registry.run_checkpoint_with_approval(
            checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
            context={"episode_stats": {...}},
            scene_id="kitchen_001",
        )
    """

    def __init__(
        self,
        verbose: bool = True,
        config: Optional["QualityConfig"] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.gates: Dict[str, QualityGate] = {}
        self.results: List[QualityGateResult] = []
        self.verbose = verbose
        self.approval_manager: Optional[HumanApprovalManager] = None

        # Load configuration with overrides
        if config:
            self.config = config
        elif HAVE_CONFIG:
            self.config = load_quality_config(overrides=config_overrides)
        else:
            self.config = None

        # Register default gates with configurable thresholds
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

    def run_checkpoint_with_approval(
        self,
        checkpoint: QualityGateCheckpoint,
        context: Dict[str, Any],
        scene_id: str,
        notification_service: Optional["NotificationService"] = None,
        wait_for_approval: bool = True,
        approval_timeout_hours: Optional[float] = None,
    ) -> Tuple[List[QualityGateResult], bool]:
        """
        LABS P1 FIX: Run checkpoint with human approval workflow for failures.

        This method runs all gates for a checkpoint and handles approval
        workflow for any gates that require human approval.

        Args:
            checkpoint: The checkpoint to evaluate
            context: Context data for gate checks
            scene_id: Scene identifier for approval tracking
            notification_service: Optional notification service
            wait_for_approval: If True, block until approval (or timeout)
            approval_timeout_hours: Override timeout for approval

        Returns:
            Tuple of (results, can_proceed)
            - results: List of QualityGateResult
            - can_proceed: True if all gates passed or were approved
        """
        # Run the checkpoint
        results = self.run_checkpoint(checkpoint, context, notification_service)

        # Check for failures requiring approval
        failures_needing_approval = [
            r for r in results
            if not r.passed and r.requires_human_approval
        ]

        if not failures_needing_approval:
            return results, self.can_proceed()

        # Initialize approval manager if needed
        if self.approval_manager is None:
            self.approval_manager = HumanApprovalManager(
                scene_id=scene_id,
                config=self.config,
                notification_service=notification_service,
                verbose=self.verbose,
            )

        self.log(f"Found {len(failures_needing_approval)} failures requiring approval")

        # Create approval requests
        approval_requests = []
        for result in failures_needing_approval:
            request = self.approval_manager.create_request(result)
            approval_requests.append(request)

        if not wait_for_approval:
            # Return immediately - caller must handle approval externally
            self.log("Approval required - returning without waiting")
            return results, False

        # Wait for all approvals
        all_approved = True
        for request in approval_requests:
            approved = self.approval_manager.wait_for_approval(
                request,
                timeout_hours=approval_timeout_hours,
            )
            if not approved:
                all_approved = False
                self.log(f"Approval {request.request_id} was not granted", "WARNING")

        return results, all_approved

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
        # LABS P1 FIX: Use configurable thresholds
        def check_physics(ctx: Dict[str, Any]) -> QualityGateResult:
            physics_data = ctx.get("physics_properties", {})
            objects = physics_data.get("objects", [])

            # Get thresholds from config or use defaults
            if self.config and hasattr(self.config, 'physics'):
                mass_min = self.config.physics.mass_min_kg
                mass_max = self.config.physics.mass_max_kg
                friction_min = self.config.physics.friction_min
                friction_max = self.config.physics.friction_max
            else:
                mass_min = float(os.getenv("BP_QUALITY_PHYSICS_MASS_MIN_KG", "0.01"))
                mass_max = float(os.getenv("BP_QUALITY_PHYSICS_MASS_MAX_KG", "500"))
                friction_min = float(os.getenv("BP_QUALITY_PHYSICS_FRICTION_MIN", "0"))
                friction_max = float(os.getenv("BP_QUALITY_PHYSICS_FRICTION_MAX", "2.0"))

            warnings = []
            for obj in objects:
                obj_id = obj.get("id", "unknown")
                mass = obj.get("mass", 0)
                friction = obj.get("friction", 0)

                if mass < mass_min or mass > mass_max:
                    warnings.append(f"{obj_id}: Mass {mass}kg outside expected range [{mass_min}, {mass_max}]")
                if friction < friction_min or friction > friction_max:
                    warnings.append(f"{obj_id}: Friction {friction} outside expected range [{friction_min}, {friction_max}]")

            passed = len(warnings) == 0

            return QualityGateResult(
                gate_id="qg-2-physics",
                checkpoint=QualityGateCheckpoint.SIMREADY_COMPLETE,
                passed=passed,
                severity=QualityGateSeverity.WARNING,
                message=f"Physics plausibility: {len(warnings)} warnings",
                details={
                    "warnings": warnings,
                    "thresholds": {
                        "mass_range_kg": [mass_min, mass_max],
                        "friction_range": [friction_min, friction_max],
                    },
                },
                recommendations=[
                    "Review mass estimates for realism",
                    "Verify friction coefficients match expected materials",
                    "Consider measuring real-world properties for critical objects",
                    f"Thresholds can be adjusted in quality_config.json or via BP_QUALITY_* env vars",
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
        # LABS P1 FIX: Use configurable thresholds
        def check_episodes(ctx: Dict[str, Any]) -> QualityGateResult:
            episode_stats = ctx.get("episode_stats", {})

            # Get thresholds from config or use defaults
            if self.config and hasattr(self.config, 'episodes'):
                collision_free_min = self.config.episodes.collision_free_rate_min
                quality_pass_rate_min = self.config.episodes.quality_pass_rate_min
                quality_score_min = self.config.episodes.quality_score_min
                min_episodes = self.config.episodes.min_episodes_required
            else:
                collision_free_min = float(os.getenv("BP_QUALITY_EPISODES_COLLISION_FREE_RATE_MIN", "0.8"))
                quality_pass_rate_min = float(os.getenv("BP_QUALITY_EPISODES_QUALITY_PASS_RATE_MIN", "0.5"))
                quality_score_min = float(os.getenv("BP_QUALITY_EPISODES_QUALITY_SCORE_MIN", "0.85"))
                min_episodes = int(os.getenv("BP_QUALITY_EPISODES_MIN_EPISODES_REQUIRED", "1"))

            total = episode_stats.get("total_generated", 0)
            passed_quality = episode_stats.get("passed_quality_filter", 0)
            avg_quality = episode_stats.get("average_quality_score", 0)
            collision_free = episode_stats.get("collision_free_rate", 0)

            issues = []
            if total < min_episodes:
                issues.append(f"Insufficient episodes: {total} < {min_episodes} required")
            if total > 0 and passed_quality < total * quality_pass_rate_min:
                issues.append(f"Low quality pass rate: {passed_quality}/{total} ({passed_quality/total:.1%} < {quality_pass_rate_min:.0%})")
            if collision_free < collision_free_min:
                issues.append(f"Low collision-free rate: {collision_free:.1%} < {collision_free_min:.0%}")
            if avg_quality > 0 and avg_quality < quality_score_min:
                issues.append(f"Low average quality: {avg_quality:.2f} < {quality_score_min}")

            passed = len(issues) == 0 and total >= min_episodes

            return QualityGateResult(
                gate_id="qg-6-episodes",
                checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
                passed=passed,
                severity=QualityGateSeverity.WARNING,
                message=f"Episodes: {passed_quality}/{total} passed (avg quality: {avg_quality:.2f})",
                details={
                    **episode_stats,
                    "thresholds": {
                        "collision_free_rate_min": collision_free_min,
                        "quality_pass_rate_min": quality_pass_rate_min,
                        "quality_score_min": quality_score_min,
                        "min_episodes_required": min_episodes,
                    },
                },
                recommendations=[
                    "Review failed episodes for common patterns",
                    f"Current quality_score_min threshold: {quality_score_min}",
                    "Check collision detection parameters",
                    "Thresholds can be adjusted in quality_config.json or via BP_QUALITY_* env vars",
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

        # LABS P1 FIX: QG-6b - Sim2Real Transfer Validation
        # This gate validates that generated episodes will transfer well to real robots
        def check_sim2real(ctx: Dict[str, Any]) -> QualityGateResult:
            sim2real_metrics = ctx.get("sim2real_metrics", {})

            if not sim2real_metrics:
                # Sim2Real validation not performed - this is OK, just a warning
                return QualityGateResult(
                    gate_id="qg-6b-sim2real",
                    checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
                    passed=True,  # Don't block if not configured
                    severity=QualityGateSeverity.INFO,
                    message="Sim2Real validation not configured (optional for Enterprise+ tiers)",
                    details={"note": "Enable sim2real_validation in bundle tier config for transfer metrics"},
                    recommendations=[
                        "Consider enabling Sim2Real validation for production deployments",
                        "Enterprise and Foundation tiers include automatic transfer validation",
                    ],
                )

            # Extract metrics
            transfer_gap = sim2real_metrics.get("transfer_gap", 1.0)
            sim_success_rate = sim2real_metrics.get("sim_success_rate", 0.0)
            real_success_rate = sim2real_metrics.get("real_success_rate", 0.0)
            production_ready = sim2real_metrics.get("production_ready", False)
            confidence_interval = sim2real_metrics.get("confidence_interval")
            failure_modes = sim2real_metrics.get("failure_modes", {})

            # Determine quality level based on transfer gap
            # Transfer gap = |sim_success - real_success| / max(sim_success, 0.01)
            # <10%: Excellent, 10-20%: Good, 20-35%: Moderate, >35%: Poor
            if transfer_gap < 0.10:
                quality_level = "excellent"
            elif transfer_gap < 0.20:
                quality_level = "good"
            elif transfer_gap < 0.35:
                quality_level = "moderate"
            else:
                quality_level = "poor"

            # Validation criteria
            issues = []
            recommendations = []

            # Check transfer gap (configurable thresholds)
            transfer_gap_threshold = sim2real_metrics.get("transfer_gap_threshold", 0.25)
            if transfer_gap > transfer_gap_threshold:
                issues.append(f"Transfer gap {transfer_gap:.1%} exceeds threshold {transfer_gap_threshold:.0%}")
                recommendations.append("Review domain randomization settings")
                recommendations.append("Consider fine-tuning on more diverse scenes")

            # Check if real-world success rate is acceptable
            min_real_success_rate = sim2real_metrics.get("min_real_success_rate", 0.6)
            if real_success_rate < min_real_success_rate:
                issues.append(f"Real-world success rate {real_success_rate:.1%} below threshold {min_real_success_rate:.0%}")
                recommendations.append("Increase training episodes")
                recommendations.append("Review task difficulty and gripper settings")

            # Analyze failure modes if available
            if failure_modes:
                top_failure = max(failure_modes.items(), key=lambda x: x[1]) if failure_modes else None
                if top_failure and top_failure[1] > 0.3:  # >30% of failures from one mode
                    issues.append(f"Dominant failure mode: {top_failure[0]} ({top_failure[1]:.0%})")
                    recommendations.append(f"Address {top_failure[0]} failure mode specifically")

            passed = len(issues) == 0
            severity = QualityGateSeverity.WARNING if not passed else QualityGateSeverity.INFO

            return QualityGateResult(
                gate_id="qg-6b-sim2real",
                checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
                passed=passed,
                severity=severity,
                message=f"Sim2Real transfer: {quality_level} (gap: {transfer_gap:.1%}, real: {real_success_rate:.1%})",
                details={
                    "transfer_gap": transfer_gap,
                    "transfer_gap_threshold": transfer_gap_threshold,
                    "quality_level": quality_level,
                    "sim_success_rate": sim_success_rate,
                    "real_success_rate": real_success_rate,
                    "production_ready": production_ready,
                    "confidence_interval": confidence_interval,
                    "failure_modes": failure_modes,
                    "issues": issues,
                },
                recommendations=recommendations if not passed else [
                    f"Transfer quality is {quality_level} - suitable for real-world deployment"
                ],
            )

        self.register(QualityGate(
            id="qg-6b-sim2real",
            name="Sim2Real Transfer Validation",
            checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
            severity=QualityGateSeverity.WARNING,
            description="LABS P1 FIX: Validates sim-to-real transfer fidelity for production deployment",
            check_fn=check_sim2real,
            notify_on_fail=True,
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
