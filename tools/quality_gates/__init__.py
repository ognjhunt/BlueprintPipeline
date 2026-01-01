"""Quality Gates Package.

Human-in-the-loop quality validation with notifications.
"""

from .notification_service import (
    NotificationService,
    NotificationChannel,
    send_email_notification,
    send_sms_notification,
)
from .quality_gate import (
    QualityGate,
    QualityGateResult,
    QualityGateSeverity,
    QualityGateCheckpoint,
    QualityGateRegistry,
)
from .ai_qa_context import (
    QAContextGenerator,
    generate_qa_context,
)

__all__ = [
    "NotificationService",
    "NotificationChannel",
    "send_email_notification",
    "send_sms_notification",
    "QualityGate",
    "QualityGateResult",
    "QualityGateSeverity",
    "QualityGateCheckpoint",
    "QualityGateRegistry",
    "QAContextGenerator",
    "generate_qa_context",
]
