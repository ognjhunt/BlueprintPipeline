from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings

from pydantic.warnings import PydanticDeprecatedSince20

warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.config import (
    ApprovalStoreConfig,
    GateOverrideConfig,
    HumanApprovalConfig,
    OverrideReasonSchema,
    QualityConfig,
)
from tools.quality_gates.quality_gate import (
    ApprovalStatus,
    HumanApprovalManager,
    QualityGateCheckpoint,
    QualityGateResult,
    QualityGateSeverity,
)


def _build_manager(tmp_path, monkeypatch, config):
    config.approval_store = ApprovalStoreConfig(
        backend="filesystem",
        filesystem_path=str(tmp_path),
    )
    return HumanApprovalManager(scene_id="scene-123", config=config, verbose=False)


def _make_failure_result():
    return QualityGateResult(
        gate_id="gate-1",
        checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
        passed=False,
        severity=QualityGateSeverity.ERROR,
        message="quality gate failed",
    )


def test_timeout_expires_in_production(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_ENV", "production")
    config = QualityConfig(
        human_approval=HumanApprovalConfig(
            timeout_hours=0.001,
            auto_approve_on_timeout=True,
            allow_auto_approve_on_timeout_non_production=True,
        ),
    )
    manager = _build_manager(tmp_path, monkeypatch, config)
    request = manager.create_request(_make_failure_result())
    request.expires_at = (datetime.utcnow() - timedelta(hours=1)).isoformat() + "Z"
    manager.store.save_request(request)

    current = manager.check_status(request.request_id)

    assert current is not None
    assert current.status == ApprovalStatus.EXPIRED
    assert current.approved_by is None
    assert "explicit human approval required" in (current.approval_notes or "")


def test_override_metadata_schema_validation(monkeypatch, tmp_path):
    monkeypatch.delenv("PIPELINE_ENV", raising=False)
    schema = OverrideReasonSchema(
        required_fields=["category", "ticket", "justification"],
        categories=["known_issue"],
        ticket_pattern=r"^OPS-\d+$",
        justification_min_length=20,
    )
    config = QualityConfig(
        gate_overrides=GateOverrideConfig(
            allow_manual_override=True,
            override_reason_schema=schema,
        )
    )
    manager = _build_manager(tmp_path, monkeypatch, config)
    request = manager.create_request(_make_failure_result())

    missing_category = {
        "approver_id": "admin@example.com",
        "ticket": "OPS-1234",
        "justification": "Valid justification with enough length",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    assert manager.override(request.request_id, missing_category) is False

    invalid_category = {
        "approver_id": "admin@example.com",
        "category": "other",
        "ticket": "OPS-1234",
        "justification": "Valid justification with enough length",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    assert manager.override(request.request_id, invalid_category) is False

    valid_metadata = {
        "approver_id": "admin@example.com",
        "category": "known_issue",
        "ticket": "OPS-1234",
        "justification": "Valid justification with enough length",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    assert manager.override(request.request_id, valid_metadata) is True

    current = manager.check_status(request.request_id)
    assert current is not None
    assert current.status == ApprovalStatus.OVERRIDDEN
    assert current.override_metadata["category"] == "known_issue"
    assert current.override_metadata["ticket"] == "OPS-1234"
