"""Quality Gate Framework.

Defines quality gates at key pipeline checkpoints with severity levels
and human-in-the-loop validation support.

This module now supports:
- Configurable thresholds via quality_config.json
- Human approval workflow for blocked gates
- Manual override with audit logging
- Environment variable overrides (BP_QUALITY_*)
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .notification_service import NotificationService, NotificationPayload

# Import configuration
try:
    from tools.config import load_quality_config, QualityConfig
    from tools.config.env import parse_bool_env, parse_float_env
    from tools.quality_gates.notification_validation import ensure_production_notification_channels
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
    # Genie Sim Integration
    GENIESIM_EXPORT_READY = "geniesim_export_ready"
    GENIESIM_IMPORT_COMPLETE = "geniesim_import_complete"
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
# Human Approval Workflow
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
    override_metadata: Optional[Dict[str, Any]] = None
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
            "override_metadata": self.override_metadata,
            "approval_notes": self.approval_notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalRequest":
        return cls(
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
            override_metadata=data.get("override_metadata"),
            approval_notes=data.get("approval_notes"),
        )


class ApprovalStore(Protocol):
    """Storage interface for approval requests and audit entries."""

    def save_request(self, request: ApprovalRequest) -> None:
        ...

    def load_request(self, request_id: str) -> Optional[ApprovalRequest]:
        ...

    def list_requests(self, status: Optional[ApprovalStatus] = None) -> List[ApprovalRequest]:
        ...

    def write_audit_entry(self, audit_entry: Dict[str, Any]) -> None:
        ...

    def migrate_from_filesystem(self, base_dir: Path) -> int:
        ...


class FilesystemApprovalStore:
    """Filesystem-backed approval store."""

    DEFAULT_APPROVALS_DIR = Path("/var/lib/blueprintpipeline/approvals")

    def __init__(self, scene_id: str, base_dir: Optional[Path] = None) -> None:
        self.scene_id = scene_id
        self.base_dir = base_dir or self.DEFAULT_APPROVALS_DIR
        self.scene_dir = self.base_dir / scene_id
        self.scene_dir.mkdir(parents=True, exist_ok=True)

    def save_request(self, request: ApprovalRequest) -> None:
        path = self.scene_dir / f"{request.request_id}.json"
        with open(path, "w") as f:
            json.dump(request.to_dict(), f, indent=2)

    def load_request(self, request_id: str) -> Optional[ApprovalRequest]:
        path = self.scene_dir / f"{request_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return ApprovalRequest.from_dict(data)

    def list_requests(self, status: Optional[ApprovalStatus] = None) -> List[ApprovalRequest]:
        requests: List[ApprovalRequest] = []
        for path in self.scene_dir.glob("*.json"):
            request = self.load_request(path.stem)
            if not request:
                continue
            if status and request.status != status:
                continue
            requests.append(request)
        return requests

    def write_audit_entry(self, audit_entry: Dict[str, Any]) -> None:
        audit_dir = self.scene_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"{audit_entry['request_id']}_{audit_entry['action'].lower()}.json"
        with open(audit_path, "w") as f:
            json.dump(audit_entry, f, indent=2)
        log_path = audit_dir / "audit.log.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")

    def migrate_from_filesystem(self, base_dir: Path) -> int:
        if base_dir.resolve() == self.base_dir.resolve():
            return 0
        migrated = 0
        source_dir = base_dir / self.scene_id
        if not source_dir.exists():
            return 0
        for path in source_dir.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            self.save_request(ApprovalRequest.from_dict(data))
            migrated += 1
        audit_log = source_dir / "audit" / "audit.log.jsonl"
        if audit_log.exists():
            with open(audit_log) as f:
                for line in f:
                    if not line.strip():
                        continue
                    self.write_audit_entry(json.loads(line))
        return migrated


class HumanApprovalManager:
    """
    Manager for human-in-the-loop approval workflow.

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
        manager.override(
            request.request_id,
            override_metadata={
                "approver_id": "admin@example.com",
                "category": "known_issue",
                "ticket": "OPS-1234",
                "justification": "Known issue; verified acceptable for this delivery and documented mitigation steps.",
                "timestamp": "2024-01-10T12:00:00Z",
            },
        )
    """

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
            auto_approve_on_timeout = self.config.human_approval.auto_approve_on_timeout
            allow_auto_approve_non_prod = (
                self.config.human_approval.allow_auto_approve_on_timeout_non_production
            )
        else:
            self.timeout_hours = parse_float_env(
                os.getenv("APPROVAL_TIMEOUT_HOURS"),
                default=24,
                min_value=0,
                name="APPROVAL_TIMEOUT_HOURS",
            )
            auto_approve_on_timeout = parse_bool_env(os.getenv("AUTO_APPROVE_ON_TIMEOUT"), default=False)
            allow_auto_approve_non_prod = (
                parse_bool_env(os.getenv("ALLOW_AUTO_APPROVE_ON_TIMEOUT_NON_PROD"), default=False)
            )

        auto_approve_requested = auto_approve_on_timeout
        if self._is_production_mode():
            if auto_approve_requested:
                self.log(
                    "Auto-approve on timeout requested but disabled in production; "
                    "requests will expire instead.",
                    "WARNING",
                )
            self.auto_approve_on_timeout = False
        else:
            if auto_approve_requested and not allow_auto_approve_non_prod:
                self.log(
                    "Auto-approve on timeout requested but not permitted; "
                    "set allow_auto_approve_on_timeout_non_production to enable.",
                    "WARNING",
                )
            self.auto_approve_on_timeout = auto_approve_requested and allow_auto_approve_non_prod

        self.store = self._init_store(scene_id)
        self._run_store_migration_if_needed()

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
        self.store.save_request(request)

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
        request = self.store.load_request(request_id)

        if request and request.status == ApprovalStatus.PENDING:
            # Check if expired
            if request.expires_at:
                expires = datetime.fromisoformat(request.expires_at.replace("Z", "+00:00"))
                if datetime.now(expires.tzinfo) > expires:
                    allow_auto_approve = (
                        self.auto_approve_on_timeout and not self._is_production_mode()
                    )
                    if allow_auto_approve:
                        request.status = ApprovalStatus.APPROVED
                        request.approved_by = "SYSTEM:auto_timeout"
                        request.approved_at = datetime.utcnow().isoformat() + "Z"
                        request.approval_notes = f"Auto-approved after {self.timeout_hours}h timeout"
                        self.store.save_request(request)
                        self._log_audit(
                            "AUTO_APPROVED_TIMEOUT",
                            request,
                            request.approved_by,
                            reason=request.approval_notes,
                        )
                        self.log(
                            f"Request {request.request_id} auto-approved after timeout",
                            "WARNING",
                        )
                        return request

                    request.status = ApprovalStatus.EXPIRED
                    if self._is_production_mode():
                        request.approval_notes = (
                            f"Expired after {self.timeout_hours}h timeout; "
                            "explicit human approval required to proceed"
                        )
                    else:
                        request.approval_notes = f"Expired after {self.timeout_hours}h timeout"
                    self.store.save_request(request)
                    self._log_audit(
                        "EXPIRED",
                        request,
                        "SYSTEM:timeout",
                        reason=request.approval_notes,
                    )
                    self.log(f"Request {request.request_id} expired after timeout", "WARNING")

        return request

    def wait_for_approval(
        self,
        request: ApprovalRequest,
        timeout_hours: Optional[float] = None,
        poll_interval_seconds: float = 30.0,
        cancel_event: Optional["threading.Event"] = None,
    ) -> bool:
        """Wait for an approval request to be processed (blocking).

        Args:
            request: The ApprovalRequest to wait for
            timeout_hours: Override timeout (default: use config)
            poll_interval_seconds: How often to check status
            cancel_event: Optional threading.Event to cancel the wait early

        Returns:
            True if approved/overridden, False if rejected/expired
        """
        timeout = timeout_hours or self.timeout_hours
        start_time = time.time()
        max_wait_seconds = timeout * 3600
        elapsed = 0.0

        self.log(f"Waiting for approval of {request.request_id} (timeout: {timeout}h)")
        self.log(f"Approval can be submitted via dashboard, email, or API")

        while elapsed < max_wait_seconds:
            elapsed = time.time() - start_time
            if cancel_event and cancel_event.is_set():
                self.log(f"Approval request {request.request_id} canceled", "WARNING")
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
            wait_seconds = min(poll_interval_seconds, max_wait_seconds - elapsed)
            if wait_seconds > 0:
                if cancel_event:
                    if cancel_event.wait(wait_seconds):
                        self.log(f"Approval request {request.request_id} canceled", "WARNING")
                        return False
                else:
                    time.sleep(wait_seconds)
            elapsed = time.time() - start_time

        self.log(f"Approval request {request.request_id} timed out", "WARNING")
        current = self.check_status(request.request_id)
        if current:
            return current.status in (ApprovalStatus.APPROVED, ApprovalStatus.OVERRIDDEN)
        return False

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
        request = self.store.load_request(request_id)

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

        self.store.save_request(request)
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
        request = self.store.load_request(request_id)

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

        self.store.save_request(request)
        self._log_audit("REJECTED", request, approver)

        self.log(f"Request {request_id} rejected by {approver}")
        return True

    def override(
        self,
        request_id: str,
        override_metadata: Dict[str, Any],
    ) -> bool:
        """Override a pending approval request (bypass with audit).

        Allows labs to proceed past failed gates with documented reason.

        Args:
            request_id: The approval request ID
            override_metadata: Structured metadata containing approver_id, category,
                ticket, justification, and timestamp.

        Returns:
            True if successfully overridden
        """
        if not self._validate_override_metadata(override_metadata):
            return False

        request = self.store.load_request(request_id)

        if not request:
            self.log(f"Approval request {request_id} not found", "ERROR")
            return False

        if self.config and hasattr(self.config, "gate_overrides"):
            if not self.config.gate_overrides.allow_manual_override:
                self.log("Manual overrides are disabled by policy", "ERROR")
                return False

        if self._is_production_mode():
            allowed_overriders = []
            if self.config and hasattr(self.config, "gate_overrides"):
                if not self.config.gate_overrides.allow_override_in_production:
                    self.log("Overrides are disabled in production by policy", "ERROR")
                    return False
                allowed_overriders = self.config.gate_overrides.allowed_overriders
            if not allowed_overriders:
                self.log("Overrides are not permitted in production", "ERROR")
                return False
            approver_id = str(override_metadata["approver_id"]).strip()
            if approver_id not in allowed_overriders:
                self.log("Override requires a privileged approver in production", "ERROR")
                return False

        if request.status not in (ApprovalStatus.PENDING, ApprovalStatus.EXPIRED):
            self.log(f"Cannot override {request_id}: status is {request.status.value}", "WARNING")
            return False

        category = str(override_metadata["category"]).strip()
        ticket = str(override_metadata["ticket"]).strip()
        justification = str(override_metadata["justification"]).strip()

        request.status = ApprovalStatus.OVERRIDDEN
        request.approved_by = str(override_metadata["approver_id"]).strip()
        request.approved_at = datetime.utcnow().isoformat() + "Z"
        request.override_reason = f"{category}: {justification}"
        request.override_metadata = {
            "approver_id": request.approved_by,
            "category": category,
            "ticket": ticket,
            "justification": justification,
            "timestamp": str(override_metadata["timestamp"]).strip(),
        }

        self.store.save_request(request)
        self._log_audit(
            "OVERRIDDEN",
            request,
            request.approved_by,
            request.override_reason,
            request.override_metadata,
        )

        self.log(f"Request {request_id} overridden by {request.approved_by}: {request.override_reason}")
        return True

    def list_pending(self) -> List[ApprovalRequest]:
        """List all pending approval requests for this scene."""
        return self.store.list_requests(status=ApprovalStatus.PENDING)

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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log audit trail for approval actions."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "request_id": request.request_id,
            "gate_id": request.gate_id,
            "scene_id": request.scene_id,
            "actor": actor,
            "reason": reason,
            "metadata": metadata,
        }
        if action == "OVERRIDDEN" and metadata:
            audit_entry["override_metadata_hash"] = self._hash_override_metadata(metadata)
        self.store.write_audit_entry(audit_entry)

    def _init_store(self, scene_id: str) -> ApprovalStore:
        store_backend = None
        store_config = None
        if self.config and hasattr(self.config, "approval_store"):
            store_config = self.config.approval_store
            store_backend = store_config.backend
        env_backend = os.getenv("BP_QUALITY_APPROVAL_STORE_BACKEND") or os.getenv(
            "APPROVAL_STORE_BACKEND",
        )
        if env_backend:
            store_backend = env_backend

        backend = str(store_backend).lower() if store_backend else None
        firestore_available = self._firestore_available()
        if not backend or backend == "auto":
            backend = (
                "firestore"
                if firestore_available and not self._is_local_mode()
                else "filesystem"
            )

        if backend == "firestore":
            if not firestore_available:
                if self._is_local_mode():
                    self.log(
                        "Firestore backend requested but unavailable; falling back to filesystem for local runs.",
                        "WARNING",
                    )
                    backend = "filesystem"
                else:
                    raise RuntimeError(
                        "Firestore backend requested but google-cloud-firestore is not available."
                    )
            if backend == "firestore":
                from .approval_store_firestore import FirestoreApprovalStore

                collection = "quality_gate_approvals"
                if store_config and store_config.firestore_collection:
                    collection = store_config.firestore_collection
                collection = os.getenv(
                    "BP_QUALITY_APPROVAL_STORE_FIRESTORE_COLLECTION",
                    collection,
                )
                return FirestoreApprovalStore(scene_id=scene_id, collection=collection)
        elif backend != "filesystem":
            raise ValueError(f"Unknown approval store backend: {backend}")

        base_dir = FilesystemApprovalStore.DEFAULT_APPROVALS_DIR
        if store_config and store_config.filesystem_path:
            base_dir = Path(store_config.filesystem_path)
        base_dir = Path(
            os.getenv(
                "QUALITY_APPROVAL_PATH",
                os.getenv(
                    "BP_QUALITY_APPROVAL_STORE_FILESYSTEM_PATH",
                    os.getenv("APPROVAL_STORE_FILESYSTEM_PATH", str(base_dir)),
                ),
            )
        )
        return FilesystemApprovalStore(scene_id=scene_id, base_dir=base_dir)


    def _run_store_migration_if_needed(self) -> None:
        if not self.config or not hasattr(self.config, "approval_store"):
            return
        store_config = self.config.approval_store
        if not store_config.migrate_from_filesystem:
            return
        if store_config.backend != "firestore":
            return
        source_dir = FilesystemApprovalStore.DEFAULT_APPROVALS_DIR
        if store_config.filesystem_path:
            source_dir = Path(store_config.filesystem_path)
        migrated = self.store.migrate_from_filesystem(source_dir)
        if migrated:
            self.log(f"Migrated {migrated} approval request(s) into {store_config.backend} store")

    def _hash_override_metadata(self, metadata: Dict[str, Any]) -> str:
        """Return a deterministic hash of override metadata."""
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _validate_ticket_reference(self, ticket_id: str) -> bool:
        """Validate that a ticket reference looks like an ID or URL."""
        if not ticket_id:
            return False
        if ticket_id.startswith(("http://", "https://")):
            return True
        return re.match(r"^[A-Za-z][A-Za-z0-9._-]*-\d+$", ticket_id) is not None

    def _validate_override_metadata(self, override_metadata: Dict[str, Any]) -> bool:
        schema = None
        if self.config and hasattr(self.config, "gate_overrides"):
            schema = self.config.gate_overrides.override_reason_schema

        required_fields = ["approver_id", "category", "ticket", "justification", "timestamp"]
        if schema and schema.required_fields:
            required_fields = list(dict.fromkeys(list(schema.required_fields) + ["approver_id", "timestamp"]))

        missing_fields = [field for field in required_fields if not override_metadata.get(field)]
        if missing_fields:
            self.log(
                f"Override requires metadata fields: {', '.join(missing_fields)}",
                "ERROR",
            )
            return False

        category = str(override_metadata["category"]).strip()
        if schema and schema.categories and category not in schema.categories:
            self.log(
                f"Override category must be one of: {', '.join(schema.categories)}",
                "ERROR",
            )
            return False

        ticket = str(override_metadata["ticket"]).strip()
        ticket_pattern = schema.ticket_pattern if schema else None
        if ticket_pattern:
            if not re.match(ticket_pattern, ticket):
                self.log("Override requires a ticket that matches policy", "ERROR")
                return False
        elif not self._validate_ticket_reference(ticket):
            self.log("Override requires a valid ticket URL or ID", "ERROR")
            return False

        justification = str(override_metadata["justification"]).strip()
        min_length = schema.justification_min_length if schema else 50
        if len(justification) < min_length:
            self.log(
                f"Override justification must be at least {min_length} characters",
                "ERROR",
            )
            return False

        if not self._validate_timestamp(str(override_metadata["timestamp"])):
            self.log("Override requires a valid ISO-8601 timestamp", "ERROR")
            return False

        return True

    def _validate_timestamp(self, value: str) -> bool:
        """Validate ISO-8601 timestamp format."""
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        return True

    def _firestore_available(self) -> bool:
        try:
            return importlib.util.find_spec("google.cloud.firestore") is not None
        except (ModuleNotFoundError, ValueError):
            return False

    def _is_local_mode(self) -> bool:
        pipeline_env = os.getenv("PIPELINE_ENV", "").lower()
        bp_env = os.getenv("BP_ENV", "").lower()
        local_envs = {"local", "development", "dev", "test"}
        return pipeline_env in local_envs or bp_env in local_envs

    def _is_production_mode(self) -> bool:
        """Return True when running in production mode."""
        pipeline_env = os.getenv("PIPELINE_ENV", "").lower()
        bp_env = os.getenv("BP_ENV", "").lower()
        return pipeline_env == "production" or bp_env == "production"


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

    Now supports configurable thresholds via quality_config.json
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

        if HAVE_CONFIG:
            ensure_production_notification_channels(self.config)

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
        Run checkpoint with human approval workflow for failures.

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

        def _parse_env_json(value: str) -> Optional[Any]:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None

        def _parse_env_list(value: str) -> List[str]:
            parsed = _parse_env_json(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            return [item.strip() for item in value.split(",") if item.strip()]

        def _get_nested_value(data: Dict[str, Any], path: str) -> Tuple[bool, Any]:
            current: Any = data
            for segment in path.split("."):
                if not isinstance(current, dict) or segment not in current:
                    return False, None
                current = current[segment]
            return True, current

        def _resolve_first_key(data: Dict[str, Any], keys: List[str]) -> Tuple[Optional[str], Any]:
            for key in keys:
                if key in data:
                    return key, data.get(key)
            return None, None

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
        # Use configurable thresholds
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
                mass_min = parse_float_env(
                    os.getenv("BP_QUALITY_PHYSICS_MASS_MIN_KG"),
                    default=0.01,
                    min_value=0,
                    name="BP_QUALITY_PHYSICS_MASS_MIN_KG",
                )
                mass_max = parse_float_env(
                    os.getenv("BP_QUALITY_PHYSICS_MASS_MAX_KG"),
                    default=500,
                    min_value=0,
                    name="BP_QUALITY_PHYSICS_MASS_MAX_KG",
                )
                friction_min = parse_float_env(
                    os.getenv("BP_QUALITY_PHYSICS_FRICTION_MIN"),
                    default=0,
                    min_value=0,
                    name="BP_QUALITY_PHYSICS_FRICTION_MIN",
                )
                friction_max = parse_float_env(
                    os.getenv("BP_QUALITY_PHYSICS_FRICTION_MAX"),
                    default=2.0,
                    min_value=0,
                    name="BP_QUALITY_PHYSICS_FRICTION_MAX",
                )

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

        # QG-8: USD File Size
        def check_usd_size(ctx: Dict[str, Any]) -> QualityGateResult:
            usd_path = ctx.get("usd_path")
            if not usd_path or not Path(usd_path).is_file():
                return QualityGateResult(
                    gate_id="qg-8-usd-size",
                    checkpoint=QualityGateCheckpoint.USD_ASSEMBLED,
                    passed=False,
                    severity=QualityGateSeverity.ERROR,
                    message="USD scene file not found for size validation",
                    details={"usd_path": usd_path},
                    recommendations=["Ensure USD assembly output is available before validating size"],
                )

            if self.config and hasattr(self.config, "usd"):
                max_size_bytes = self.config.usd.max_usd_size_bytes
            else:
                max_size_bytes = int(os.getenv("BP_QUALITY_USD_MAX_USD_SIZE_BYTES", "500000000"))

            size_bytes = Path(usd_path).stat().st_size
            passed = size_bytes <= max_size_bytes
            return QualityGateResult(
                gate_id="qg-8-usd-size",
                checkpoint=QualityGateCheckpoint.USD_ASSEMBLED,
                passed=passed,
                severity=QualityGateSeverity.ERROR,
                message=(
                    f"USD size {size_bytes} bytes within limit {max_size_bytes}"
                    if passed
                    else f"USD size {size_bytes} exceeds limit {max_size_bytes}"
                ),
                details={
                    "usd_path": usd_path,
                    "size_bytes": size_bytes,
                    "max_size_bytes": max_size_bytes,
                },
                recommendations=[
                    "Remove unused assets or layers from the USD stage",
                    "Split large assets into referenced sublayers",
                    "Adjust max_usd_size_bytes in quality_config.json or BP_QUALITY_USD_MAX_USD_SIZE_BYTES if justified",
                ] if not passed else [],
            )

        self.register(QualityGate(
            id="qg-8-usd-size",
            name="USD File Size",
            checkpoint=QualityGateCheckpoint.USD_ASSEMBLED,
            severity=QualityGateSeverity.ERROR,
            description="Checks assembled USD file size against configured thresholds",
            check_fn=check_usd_size,
            notify_on_fail=True,
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

        # QG-9: Replicator Bundle Sensor Metadata
        def check_replicator_bundle(ctx: Dict[str, Any]) -> QualityGateResult:
            bundle_dir = ctx.get("replicator_bundle_dir")
            if not bundle_dir or not Path(bundle_dir).is_dir():
                return QualityGateResult(
                    gate_id="qg-9-replicator-sensors",
                    checkpoint=QualityGateCheckpoint.REPLICATOR_COMPLETE,
                    passed=False,
                    severity=QualityGateSeverity.ERROR,
                    message="Replicator bundle directory not found",
                    details={"replicator_bundle_dir": bundle_dir},
                    recommendations=[
                        "Ensure replicator-job completed and bundle directory is available",
                        "Provide replicator_bundle_dir in quality gate context",
                    ],
                )

            bundle_dir = Path(bundle_dir)
            missing_files = []
            bundle_metadata_path = bundle_dir / "bundle_metadata.json"
            if not bundle_metadata_path.is_file():
                missing_files.append(str(bundle_metadata_path))

            config_dir = bundle_dir / "configs"
            config_files = sorted(config_dir.glob("*.json")) if config_dir.is_dir() else []
            if not config_files:
                missing_files.append(str(config_dir / "*.json"))

            required_sensor_fields = None
            if self.config and hasattr(self.config, "replicator"):
                required_sensor_fields = self.config.replicator.required_sensor_fields
            else:
                env_value = os.getenv("BP_QUALITY_REPLICATOR_REQUIRED_SENSOR_FIELDS")
                if env_value:
                    parsed = _parse_env_json(env_value)
                    if isinstance(parsed, dict):
                        required_sensor_fields = parsed
                    elif isinstance(parsed, list):
                        required_sensor_fields = {item: [item] for item in parsed}

            if not required_sensor_fields:
                required_sensor_fields = {
                    "camera_list": ["cameras", "camera_list"],
                    "resolution": ["resolution"],
                    "modalities": ["modalities", "annotations"],
                    "stream_ids": ["stream_ids", "streams"],
                }
            if not isinstance(required_sensor_fields, dict):
                required_sensor_fields = {
                    "camera_list": ["cameras", "camera_list"],
                    "resolution": ["resolution"],
                    "modalities": ["modalities", "annotations"],
                    "stream_ids": ["stream_ids", "streams"],
                }

            issues = []
            config_issues: List[Dict[str, Any]] = []
            if not missing_files:
                for config_path in config_files:
                    try:
                        config_data = json.loads(config_path.read_text())
                    except json.JSONDecodeError as exc:
                        issues.append(f"Invalid JSON in {config_path.name}: {exc}")
                        continue

                    capture_config = config_data.get("capture_config", {})
                    if not isinstance(capture_config, dict):
                        issues.append(f"{config_path.name}: capture_config must be an object")
                        continue

                    config_missing = []
                    config_invalid = []

                    camera_key, camera_value = _resolve_first_key(
                        capture_config, required_sensor_fields.get("camera_list", [])
                    )
                    if not camera_key:
                        config_missing.append("camera_list")
                    elif not isinstance(camera_value, list) or not camera_value:
                        config_invalid.append("camera_list")

                    resolution_key, resolution_value = _resolve_first_key(
                        capture_config, required_sensor_fields.get("resolution", [])
                    )
                    if not resolution_key:
                        config_missing.append("resolution")
                    elif not (
                        isinstance(resolution_value, (list, tuple))
                        and len(resolution_value) == 2
                        and all(isinstance(v, int) for v in resolution_value)
                    ):
                        config_invalid.append("resolution")

                    modalities_key, modalities_value = _resolve_first_key(
                        capture_config, required_sensor_fields.get("modalities", [])
                    )
                    if not modalities_key:
                        config_missing.append("modalities")
                    elif not (
                        isinstance(modalities_value, list)
                        and all(isinstance(v, str) for v in modalities_value)
                    ):
                        config_invalid.append("modalities")

                    stream_key, stream_value = _resolve_first_key(
                        capture_config, required_sensor_fields.get("stream_ids", [])
                    )
                    if not stream_key:
                        config_missing.append("stream_ids")
                    elif not (
                        isinstance(stream_value, list)
                        and all(isinstance(v, str) for v in stream_value)
                    ):
                        config_invalid.append("stream_ids")

                    if config_missing or config_invalid:
                        config_issues.append({
                            "config": config_path.name,
                            "missing_fields": config_missing,
                            "invalid_fields": config_invalid,
                        })

            passed = not missing_files and not issues and not config_issues

            recommendations = []
            if missing_files:
                recommendations.append("Regenerate replicator bundle metadata and configs")
            if config_issues:
                recommendations.append(
                    "Update capture_config to include cameras, resolution, modalities, and stream IDs"
                )
            if issues:
                recommendations.append("Fix invalid JSON in replicator config files")

            return QualityGateResult(
                gate_id="qg-9-replicator-sensors",
                checkpoint=QualityGateCheckpoint.REPLICATOR_COMPLETE,
                passed=passed,
                severity=QualityGateSeverity.ERROR,
                message=(
                    "Replicator bundle sensor metadata validated"
                    if passed
                    else f"Replicator bundle sensor metadata issues: {len(missing_files) + len(issues) + len(config_issues)}"
                ),
                details={
                    "replicator_bundle_dir": str(bundle_dir),
                    "missing_files": missing_files,
                    "config_issues": config_issues,
                    "issues": issues,
                    "required_sensor_fields": required_sensor_fields,
                },
                recommendations=recommendations,
            )

        self.register(QualityGate(
            id="qg-9-replicator-sensors",
            name="Replicator Bundle Sensor Metadata",
            checkpoint=QualityGateCheckpoint.REPLICATOR_COMPLETE,
            severity=QualityGateSeverity.ERROR,
            description="Validates Replicator bundle capture_config and sensor metadata completeness",
            check_fn=check_replicator_bundle,
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

        # QG-10: Pre-Episode Scene Reference Integrity
        def check_pre_episode_references(ctx: Dict[str, Any]) -> QualityGateResult:
            usd_path = ctx.get("usd_path")
            manifest = ctx.get("manifest", {})
            issues = []

            if not usd_path or not Path(usd_path).is_file():
                return QualityGateResult(
                    gate_id="qg-10-pre-episode-refs",
                    checkpoint=QualityGateCheckpoint.PRE_EPISODE_VALIDATION,
                    passed=False,
                    severity=QualityGateSeverity.ERROR,
                    message="USD scene file not found for reference integrity check",
                    details={"usd_path": usd_path},
                    recommendations=[
                        "Ensure usd_path is provided and points to a valid USD file",
                        "Re-run USD assembly if the file is missing",
                    ],
                )

            usd_content = Path(usd_path).read_text()
            refs = re.findall(r'references\s*=\s*@([^@]+)@', usd_content)
            usd_dir = Path(usd_path).parent
            missing_refs = []

            for ref in refs:
                ref_path = Path(ref)
                if not ref_path.is_absolute():
                    ref_path = (usd_dir / ref).resolve()
                if not ref_path.is_file():
                    missing_refs.append(ref)

            missing_manifest_assets = []
            for obj in manifest.get("objects", []) if isinstance(manifest, dict) else []:
                asset_path = obj.get("asset", {}).get("path") if isinstance(obj, dict) else None
                if not asset_path:
                    continue
                candidate = Path(asset_path)
                if not candidate.is_absolute():
                    candidate = (usd_dir / asset_path).resolve()
                if not candidate.is_file():
                    missing_manifest_assets.append(asset_path)

            if missing_refs:
                issues.append(f"Missing USD references: {len(missing_refs)}")
            if missing_manifest_assets:
                issues.append(f"Missing manifest assets: {len(missing_manifest_assets)}")

            passed = not missing_refs and not missing_manifest_assets

            return QualityGateResult(
                gate_id="qg-10-pre-episode-refs",
                checkpoint=QualityGateCheckpoint.PRE_EPISODE_VALIDATION,
                passed=passed,
                severity=QualityGateSeverity.ERROR,
                message=(
                    "Scene references resolved"
                    if passed
                    else f"Scene references missing: {len(missing_refs)} refs, {len(missing_manifest_assets)} assets"
                ),
                details={
                    "usd_path": usd_path,
                    "reference_count": len(refs),
                    "missing_references": missing_refs,
                    "missing_manifest_assets": missing_manifest_assets,
                },
                recommendations=[
                    "Update USD references to valid asset paths",
                    "Ensure manifest asset paths are synced with assembled USD",
                    "Rebuild USD layers or copy required assets into the bundle",
                ] if not passed else [],
            )

        self.register(QualityGate(
            id="qg-10-pre-episode-refs",
            name="Pre-Episode Scene Reference Integrity",
            checkpoint=QualityGateCheckpoint.PRE_EPISODE_VALIDATION,
            severity=QualityGateSeverity.ERROR,
            description="Validates all USD references and manifest assets resolve before episode generation",
            check_fn=check_pre_episode_references,
            notify_on_fail=True,
        ))

        # QG-6: Episode Quality
        # Use configurable thresholds
        def check_episodes(ctx: Dict[str, Any]) -> QualityGateResult:
            episode_stats = ctx.get("episode_stats", {})

            def is_production_mode() -> bool:
                pipeline_env = os.getenv("PIPELINE_ENV", "").lower()
                bp_env = os.getenv("BP_ENV", "").lower()
                return pipeline_env == "production" or bp_env == "production"

            # Get thresholds from config or use defaults
            if self.config and hasattr(self.config, 'episodes'):
                base_thresholds = {
                    "collision_free_rate_min": self.config.episodes.collision_free_rate_min,
                    "quality_pass_rate_min": self.config.episodes.quality_pass_rate_min,
                    "quality_score_min": self.config.episodes.quality_score_min,
                    "min_episodes_required": self.config.episodes.min_episodes_required,
                }
            else:
                base_thresholds = {
                    "collision_free_rate_min": float(os.getenv("BP_QUALITY_EPISODES_COLLISION_FREE_RATE_MIN", "0.90")),
                    "quality_pass_rate_min": float(os.getenv("BP_QUALITY_EPISODES_QUALITY_PASS_RATE_MIN", "0.70")),
                    "quality_score_min": float(os.getenv("BP_QUALITY_EPISODES_QUALITY_SCORE_MIN", "0.90")),
                    "min_episodes_required": int(os.getenv("BP_QUALITY_EPISODES_MIN_EPISODES_REQUIRED", "3")),
                }

            env_threshold_keys = (
                "BP_QUALITY_EPISODES_COLLISION_FREE_RATE_MIN",
                "BP_QUALITY_EPISODES_QUALITY_PASS_RATE_MIN",
                "BP_QUALITY_EPISODES_QUALITY_SCORE_MIN",
                "BP_QUALITY_EPISODES_MIN_EPISODES_REQUIRED",
            )
            env_overrides_used = any(key in os.environ for key in env_threshold_keys)
            env_overrides_used = env_overrides_used or bool(os.getenv("BP_QUALITY_EPISODES_TIER_THRESHOLDS"))
            context_overrides_used = bool(ctx.get("episode_thresholds_override")) or bool(ctx.get("episode_thresholds_by_tier"))
            explicit_thresholds_configured = env_overrides_used or context_overrides_used

            production_floor_applied = False
            production_minimums = {
                "collision_free_rate_min": 0.9,
                "quality_pass_rate_min": 0.75,
                "quality_score_min": 0.92,
                "min_episodes_required": 5,
            }

            tier = (
                ctx.get("data_pack_tier")
                or episode_stats.get("data_pack_tier")
                or os.getenv("DATA_PACK_TIER", "core")
            ).lower()
            tier_thresholds: Dict[str, Dict[str, Any]] = {}
            tier_thresholds_source = "defaults"
            if ctx.get("episode_thresholds_by_tier"):
                tier_thresholds = ctx["episode_thresholds_by_tier"]
                tier_thresholds_source = "context"
            elif os.getenv("BP_QUALITY_EPISODES_TIER_THRESHOLDS"):
                tier_thresholds_source = "environment"
                try:
                    tier_thresholds = json.loads(os.getenv("BP_QUALITY_EPISODES_TIER_THRESHOLDS", "{}"))
                except json.JSONDecodeError:
                    tier_thresholds = {}

            default_tier_minimums = (
                getattr(self.config.episodes, "tier_thresholds", {}) if self.config else {}
            ) or {
                "core": {
                    "collision_free_rate_min": 0.90,
                    "quality_pass_rate_min": 0.70,
                    "quality_score_min": 0.90,
                    "min_episodes_required": 3,
                },
                "plus": {
                    "collision_free_rate_min": 0.90,
                    "quality_pass_rate_min": 0.70,
                    "quality_score_min": 0.90,
                    "min_episodes_required": 4,
                },
                "full": {
                    "collision_free_rate_min": 0.90,
                    "quality_pass_rate_min": 0.70,
                    "quality_score_min": 0.90,
                    "min_episodes_required": 5,
                },
            }

            tier_minimums = tier_thresholds.get(tier) or default_tier_minimums.get(tier, {})
            tier_minimums_applied = any(
                tier_minimums.get(key, base_thresholds[key]) > base_thresholds[key]
                for key in base_thresholds
            )
            effective_thresholds = {
                key: max(base_thresholds[key], tier_minimums.get(key, base_thresholds[key]))
                for key in base_thresholds
            }
            if is_production_mode():
                production_floor_applied = any(
                    production_minimums[key] > effective_thresholds[key]
                    for key in effective_thresholds
                )
                effective_thresholds = {
                    key: max(effective_thresholds[key], production_minimums[key])
                    for key in effective_thresholds
                }

            if env_overrides_used or tier_thresholds_source in {"environment", "context"} or context_overrides_used:
                relaxed_below_production = {
                    key: effective_thresholds[key] < production_minimums[key]
                    for key in effective_thresholds
                }
                self.log(
                    "Episode quality thresholds overridden: "
                    f"env_overrides={env_overrides_used}, "
                    f"tier_source={tier_thresholds_source}, "
                    f"context_overrides={context_overrides_used}, "
                    f"relaxed_below_production={relaxed_below_production}, "
                    f"effective_thresholds={effective_thresholds}"
                )

            if explicit_thresholds_configured:
                quality_thresholds_source = "overrides"
            elif tier_minimums_applied:
                quality_thresholds_source = "tier_minimums"
            else:
                quality_thresholds_source = "defaults"

            collision_free_min = effective_thresholds["collision_free_rate_min"]
            quality_pass_rate_min = effective_thresholds["quality_pass_rate_min"]
            quality_score_min = effective_thresholds["quality_score_min"]
            min_episodes = effective_thresholds["min_episodes_required"]

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
                    "thresholds": effective_thresholds,
                    "quality_thresholds_source": quality_thresholds_source,
                    "thresholds_audit": {
                        "base_thresholds": base_thresholds,
                        "tier": tier,
                        "tier_minimums": tier_minimums,
                        "tier_thresholds_source": tier_thresholds_source,
                        "production_floor_applied": production_floor_applied,
                        "production_minimums": production_minimums if production_floor_applied else {},
                        "explicit_thresholds_configured": explicit_thresholds_configured,
                        "env_overrides_used": env_overrides_used,
                        "context_overrides_used": context_overrides_used,
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

        # QG-11: Episode Metadata Schema Completeness
        def check_episode_metadata(ctx: Dict[str, Any]) -> QualityGateResult:
            metadata_path = ctx.get("episode_metadata_path")
            lerobot_dataset_path = ctx.get("lerobot_dataset_path")

            candidate_paths = []
            if metadata_path:
                candidate_paths.append(Path(metadata_path))
            if lerobot_dataset_path:
                lerobot_path = Path(lerobot_dataset_path)
                candidate_paths.extend([
                    lerobot_path / "metadata.json",
                    lerobot_path / "meta" / "info.json",
                    lerobot_path / "info.json",
                ])

            metadata_file = next((path for path in candidate_paths if path and path.is_file()), None)
            if not metadata_file:
                return QualityGateResult(
                    gate_id="qg-11-episode-metadata",
                    checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
                    passed=False,
                    severity=QualityGateSeverity.ERROR,
                    message="Episode metadata file not found",
                    details={
                        "episode_metadata_path": metadata_path,
                        "lerobot_dataset_path": lerobot_dataset_path,
                        "searched_paths": [str(p) for p in candidate_paths],
                    },
                    recommendations=[
                        "Export episode metadata (metadata.json or meta/info.json) during episode generation",
                        "Provide episode_metadata_path in quality gate context if stored elsewhere",
                    ],
                )

            try:
                metadata = json.loads(metadata_file.read_text())
            except json.JSONDecodeError as exc:
                return QualityGateResult(
                    gate_id="qg-11-episode-metadata",
                    checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
                    passed=False,
                    severity=QualityGateSeverity.ERROR,
                    message=f"Episode metadata JSON invalid: {exc}",
                    details={"metadata_file": str(metadata_file)},
                    recommendations=["Fix episode metadata JSON formatting and regenerate export"],
                )

            required_fields = None
            if self.config and hasattr(self.config, "episode_metadata"):
                required_fields = self.config.episode_metadata.required_fields
            else:
                env_value = os.getenv("BP_QUALITY_EPISODE_METADATA_REQUIRED_FIELDS")
                if env_value:
                    parsed = _parse_env_json(env_value)
                    if isinstance(parsed, dict):
                        required_fields = parsed

            if not required_fields:
                required_fields = {
                    "dataset_name": {"paths": ["dataset_name", "name"], "type": "string"},
                    "scene_id": {"paths": ["scene_id", "scene.scene_id"], "type": "string"},
                    "robot_type": {"paths": ["robot_type", "robot.type"], "type": "string"},
                    "camera_specs": {
                        "paths": ["camera_specs", "data_pack.cameras", "cameras"],
                        "type": "array_or_object",
                    },
                    "fps": {"paths": ["fps"], "type": "number"},
                    "action_space": {"paths": ["action_space", "action_space_info"], "type": "array_or_object"},
                    "episode_stats": {"paths": ["episode_stats", "stats"], "type": "object"},
                }
            if not isinstance(required_fields, dict):
                required_fields = {
                    "dataset_name": {"paths": ["dataset_name", "name"], "type": "string"},
                    "scene_id": {"paths": ["scene_id", "scene.scene_id"], "type": "string"},
                    "robot_type": {"paths": ["robot_type", "robot.type"], "type": "string"},
                    "camera_specs": {
                        "paths": ["camera_specs", "data_pack.cameras", "cameras"],
                        "type": "array_or_object",
                    },
                    "fps": {"paths": ["fps"], "type": "number"},
                    "action_space": {"paths": ["action_space", "action_space_info"], "type": "array_or_object"},
                    "episode_stats": {"paths": ["episode_stats", "stats"], "type": "object"},
                }

            missing_fields = []
            type_errors = []
            for field_name, spec in required_fields.items():
                paths = spec.get("paths", [])
                expected_type = spec.get("type", "object")
                found = False
                value = None
                for path in paths:
                    has_value, value = _get_nested_value(metadata, path)
                    if has_value:
                        found = True
                        break
                if not found:
                    missing_fields.append(field_name)
                    continue

                valid_type = False
                if expected_type == "string":
                    valid_type = isinstance(value, str)
                elif expected_type == "number":
                    valid_type = isinstance(value, (int, float))
                elif expected_type == "object":
                    valid_type = isinstance(value, dict)
                elif expected_type == "array":
                    valid_type = isinstance(value, list)
                elif expected_type == "array_or_object":
                    valid_type = isinstance(value, (list, dict))
                elif expected_type == "boolean":
                    valid_type = isinstance(value, bool)
                else:
                    valid_type = value is not None

                if not valid_type:
                    type_errors.append({
                        "field": field_name,
                        "expected_type": expected_type,
                        "actual_type": type(value).__name__,
                    })

            passed = not missing_fields and not type_errors

            recommendations = []
            if missing_fields:
                recommendations.append(
                    f"Populate episode metadata fields: {', '.join(missing_fields)}"
                )
            if type_errors:
                recommendations.append(
                    "Ensure metadata fields match expected types (string/number/object)"
                )

            return QualityGateResult(
                gate_id="qg-11-episode-metadata",
                checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
                passed=passed,
                severity=QualityGateSeverity.ERROR,
                message=(
                    "Episode metadata schema complete"
                    if passed
                    else f"Episode metadata missing {len(missing_fields)} fields and {len(type_errors)} type issues"
                ),
                details={
                    "metadata_file": str(metadata_file),
                    "missing_fields": missing_fields,
                    "type_errors": type_errors,
                    "required_fields": required_fields,
                },
                recommendations=recommendations,
            )

        self.register(QualityGate(
            id="qg-11-episode-metadata",
            name="Episode Metadata Schema Completeness",
            checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
            severity=QualityGateSeverity.ERROR,
            description="Validates episode metadata contains required schema fields and types",
            check_fn=check_episode_metadata,
            notify_on_fail=True,
        ))

        # QG-6a: Average Quality Score SLI (blocking)
        def check_quality_score_sli(ctx: Dict[str, Any]) -> QualityGateResult:
            episode_stats = ctx.get("episode_stats", {})

            if self.config and hasattr(self.config, "data_quality"):
                min_avg_quality = self.config.data_quality.min_average_quality_score
            else:
                min_avg_quality = float(os.getenv("BP_QUALITY_AVG_SCORE_MIN", "0.85"))

            avg_quality = episode_stats.get("average_quality_score", 0.0) or 0.0
            total = episode_stats.get("total_generated", 0) or 0
            passed = total > 0 and avg_quality >= min_avg_quality

            return QualityGateResult(
                gate_id="qg-6a-quality-score-sli",
                checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
                passed=passed,
                severity=QualityGateSeverity.ERROR,
                message=(
                    f"Average quality score: {avg_quality:.2f} "
                    f"(min {min_avg_quality:.2f})"
                ),
                details={
                    "average_quality_score": avg_quality,
                    "total_episodes": total,
                    "threshold": min_avg_quality,
                },
                recommendations=[
                    "Increase episodes or adjust task constraints",
                    "Review validation failures and sensor capture quality",
                ] if not passed else [],
            )

        self.register(QualityGate(
            id="qg-6a-quality-score-sli",
            name="Average Quality Score SLI",
            checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
            severity=QualityGateSeverity.ERROR,
            description="Ensures average episode quality meets production SLI thresholds",
            check_fn=check_quality_score_sli,
            notify_on_fail=True,
            require_human_approval=True,
        ))

        # QG-6c: Sensor Capture Source SLI
        def check_sensor_capture_sli(ctx: Dict[str, Any]) -> QualityGateResult:
            data_quality = ctx.get("data_quality", {})

            if self.config and hasattr(self.config, "data_quality"):
                min_sensor_rate = self.config.data_quality.min_sensor_capture_rate
                allowed_sources = self.config.data_quality.allowed_sensor_sources
            else:
                min_sensor_rate = float(os.getenv("BP_QUALITY_SENSOR_CAPTURE_RATE_MIN", "0.90"))
                allowed_sources = [
                    source.strip()
                    for source in os.getenv(
                        "BP_QUALITY_SENSOR_SOURCES",
                        "isaac_sim_replicator,simulation",
                    ).split(",")
                    if source.strip()
                ]

            sensor_rate = float(data_quality.get("sensor_capture_rate", 0.0) or 0.0)
            sources = data_quality.get("sensor_sources", [])
            unexpected_sources = [
                source for source in sources
                if source and source not in allowed_sources
            ]

            passed = sensor_rate >= min_sensor_rate and not unexpected_sources

            return QualityGateResult(
                gate_id="qg-6c-sensor-capture",
                checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
                passed=passed,
                severity=QualityGateSeverity.ERROR,
                message=(
                    f"Sensor capture rate: {sensor_rate:.1%} "
                    f"(min {min_sensor_rate:.1%})"
                ),
                details={
                    "sensor_capture_rate": sensor_rate,
                    "allowed_sources": allowed_sources,
                    "observed_sources": sources,
                    "unexpected_sources": unexpected_sources,
                },
                recommendations=[
                    "Ensure Isaac Sim Replicator capture is enabled",
                    "Disable mock capture in production",
                    "Verify sensor backend configuration",
                ] if not passed else [],
            )

        self.register(QualityGate(
            id="qg-6c-sensor-capture",
            name="Sensor Capture Source SLI",
            checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
            severity=QualityGateSeverity.ERROR,
            description="Validates sensor capture source and coverage meet SLI thresholds",
            check_fn=check_sensor_capture_sli,
            notify_on_fail=True,
            require_human_approval=True,
        ))

        # QG-6d: Physics Validation SLI
        def check_physics_validation_sli(ctx: Dict[str, Any]) -> QualityGateResult:
            data_quality = ctx.get("data_quality", {})

            if self.config and hasattr(self.config, "data_quality"):
                min_physics_rate = self.config.data_quality.min_physics_validation_rate
                allowed_backends = self.config.data_quality.allowed_physics_backends
            else:
                min_physics_rate = float(os.getenv("BP_QUALITY_PHYSICS_VALIDATION_RATE_MIN", "0.90"))
                allowed_backends = [
                    backend.strip()
                    for backend in os.getenv(
                        "BP_QUALITY_PHYSICS_BACKENDS",
                        "isaac_sim,isaac_lab",
                    ).split(",")
                    if backend.strip()
                ]

            physics_rate = float(data_quality.get("physics_validation_rate", 0.0) or 0.0)
            backends = data_quality.get("physics_backends", [])
            unexpected_backends = [
                backend for backend in backends
                if backend and backend not in allowed_backends
            ]

            passed = physics_rate >= min_physics_rate and not unexpected_backends

            return QualityGateResult(
                gate_id="qg-6d-physics-validation",
                checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
                passed=passed,
                severity=QualityGateSeverity.ERROR,
                message=(
                    f"Physics validation rate: {physics_rate:.1%} "
                    f"(min {min_physics_rate:.1%})"
                ),
                details={
                    "physics_validation_rate": physics_rate,
                    "allowed_backends": allowed_backends,
                    "observed_backends": backends,
                    "unexpected_backends": unexpected_backends,
                },
                recommendations=[
                    "Run physics validation with Isaac Sim or Isaac Lab",
                    "Disable heuristic-only validation in production",
                ] if not passed else [],
            )

        self.register(QualityGate(
            id="qg-6d-physics-validation",
            name="Physics Validation SLI",
            checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
            severity=QualityGateSeverity.ERROR,
            description="Validates physics-backed episode validation coverage",
            check_fn=check_physics_validation_sli,
            notify_on_fail=True,
            require_human_approval=True,
        ))

        # QG-6b - Sim2Real Transfer Validation
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
            description="Validates sim-to-real transfer fidelity for production deployment",
            check_fn=check_sim2real,
            notify_on_fail=True,
        ))

        # QG-12: DWM Conditioning Data Format
        def check_dwm_bundle(ctx: Dict[str, Any]) -> QualityGateResult:
            dwm_output_dir = ctx.get("dwm_output_dir")
            if not dwm_output_dir or not Path(dwm_output_dir).is_dir():
                return QualityGateResult(
                    gate_id="qg-12-dwm-format",
                    checkpoint=QualityGateCheckpoint.DWM_PREPARED,
                    passed=False,
                    severity=QualityGateSeverity.ERROR,
                    message="DWM output directory not found",
                    details={"dwm_output_dir": dwm_output_dir},
                    recommendations=[
                        "Ensure dwm-preparation-job produced bundles",
                        "Provide dwm_output_dir in quality gate context",
                    ],
                )

            if self.config and hasattr(self.config, "dwm"):
                required_files = self.config.dwm.required_files
            else:
                env_value = os.getenv("BP_QUALITY_DWM_REQUIRED_FILES")
                required_files = _parse_env_list(env_value) if env_value else []
            if not required_files:
                required_files = [
                    "manifest.json",
                    "static_scene_video.mp4",
                    "camera_trajectory.json",
                    "metadata/scene_info.json",
                    "metadata/prompt.txt",
                ]
            if not isinstance(required_files, list):
                required_files = [
                    "manifest.json",
                    "static_scene_video.mp4",
                    "camera_trajectory.json",
                    "metadata/scene_info.json",
                    "metadata/prompt.txt",
                ]

            dwm_output_dir = Path(dwm_output_dir)
            bundle_dirs = [path for path in dwm_output_dir.iterdir() if path.is_dir()]
            if not bundle_dirs:
                return QualityGateResult(
                    gate_id="qg-12-dwm-format",
                    checkpoint=QualityGateCheckpoint.DWM_PREPARED,
                    passed=False,
                    severity=QualityGateSeverity.ERROR,
                    message="No DWM bundles found in output directory",
                    details={"dwm_output_dir": str(dwm_output_dir)},
                    recommendations=[
                        "Verify DWM bundle packager created bundle directories",
                        "Check dwm-preparation-job logs for bundle generation errors",
                    ],
                )

            missing_by_bundle = {}
            manifest_issues = []
            hand_required_missing = {}

            for bundle_dir in bundle_dirs:
                missing_files = []
                for rel_path in required_files:
                    if not (bundle_dir / rel_path).is_file():
                        missing_files.append(rel_path)

                manifest_path = bundle_dir / "manifest.json"
                manifest_data = {}
                if manifest_path.is_file():
                    try:
                        manifest_data = json.loads(manifest_path.read_text())
                    except json.JSONDecodeError as exc:
                        manifest_issues.append({
                            "bundle": bundle_dir.name,
                            "issue": f"Invalid manifest.json: {exc}",
                        })

                action_type = str(manifest_data.get("action_type", "unknown")).lower()
                requires_hand = action_type not in {"none", "static", "unknown", "camera_only"}
                if requires_hand:
                    hand_missing = []
                    if not (bundle_dir / "hand_mesh_video.mp4").is_file():
                        hand_missing.append("hand_mesh_video.mp4")
                    if not (bundle_dir / "hand_trajectory.json").is_file():
                        hand_missing.append("hand_trajectory.json")
                    if hand_missing:
                        hand_required_missing[bundle_dir.name] = hand_missing

                if manifest_data:
                    static_video = manifest_data.get("static_scene_video")
                    camera_file = manifest_data.get("camera_trajectory_file")
                    hand_video = manifest_data.get("hand_mesh_video")
                    hand_traj = manifest_data.get("hand_trajectory_file")

                    if static_video and not (bundle_dir / static_video).is_file():
                        manifest_issues.append({
                            "bundle": bundle_dir.name,
                            "issue": f"static_scene_video missing at {static_video}",
                        })
                    if camera_file and not (bundle_dir / camera_file).is_file():
                        manifest_issues.append({
                            "bundle": bundle_dir.name,
                            "issue": f"camera_trajectory_file missing at {camera_file}",
                        })
                    if requires_hand and hand_video and not (bundle_dir / hand_video).is_file():
                        manifest_issues.append({
                            "bundle": bundle_dir.name,
                            "issue": f"hand_mesh_video missing at {hand_video}",
                        })
                    if requires_hand and hand_traj and not (bundle_dir / hand_traj).is_file():
                        manifest_issues.append({
                            "bundle": bundle_dir.name,
                            "issue": f"hand_trajectory_file missing at {hand_traj}",
                        })

                if missing_files:
                    missing_by_bundle[bundle_dir.name] = missing_files

            passed = not missing_by_bundle and not manifest_issues and not hand_required_missing

            recommendations = []
            if missing_by_bundle:
                recommendations.append("Ensure DWM bundle packager writes all required files")
            if hand_required_missing:
                recommendations.append("Render hand mesh videos and trajectories for interactive actions")
            if manifest_issues:
                recommendations.append("Fix manifest.json entries to match actual bundle files")

            return QualityGateResult(
                gate_id="qg-12-dwm-format",
                checkpoint=QualityGateCheckpoint.DWM_PREPARED,
                passed=passed,
                severity=QualityGateSeverity.ERROR,
                message=(
                    "DWM bundle structure validated"
                    if passed
                    else "DWM bundle structure issues detected"
                ),
                details={
                    "dwm_output_dir": str(dwm_output_dir),
                    "required_files": required_files,
                    "missing_files": missing_by_bundle,
                    "hand_required_missing": hand_required_missing,
                    "manifest_issues": manifest_issues,
                },
                recommendations=recommendations,
            )

        self.register(QualityGate(
            id="qg-12-dwm-format",
            name="DWM Conditioning Data Format",
            checkpoint=QualityGateCheckpoint.DWM_PREPARED,
            severity=QualityGateSeverity.ERROR,
            description="Validates DWM conditioning bundle structure and manifest consistency",
            check_fn=check_dwm_bundle,
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
