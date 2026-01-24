"""Notification Service for Quality Gate Alerts.

Supports email and SMS notifications for human-in-the-loop QA.

Environment Variables:
    QA_EMAIL: Default email for QA notifications (default: ohstnhunt@gmail.com)
    QA_PHONE: Default phone for SMS notifications (default: 9196389913)

    # Email (SendGrid)
    SENDGRID_API_KEY: SendGrid API key for email notifications
    SENDGRID_FROM_EMAIL: From email address (default: qa@blueprintpipeline.com)

    # SMS (Twilio)
    TWILIO_ACCOUNT_SID: Twilio account SID
    TWILIO_AUTH_TOKEN: Twilio auth token
    TWILIO_FROM_PHONE: Twilio phone number for sending SMS

    # Slack
    QA_SLACK_WEBHOOK_URL: Slack incoming webhook URL

    # Fallback (SMTP)
    SMTP_HOST: SMTP server hostname
    SMTP_PORT: SMTP server port (default: 587)
    SMTP_USER: SMTP username
    SMTP_PASSWORD: SMTP password
"""

from __future__ import annotations

import json
import os
import smtplib
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.config import QualityConfig


class NotificationChannel(str, Enum):
    """Notification channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    CONSOLE = "console"


@dataclass
class NotificationResult:
    """Result of a notification attempt."""
    success: bool
    channel: NotificationChannel
    recipient: str
    message: str
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


@dataclass
class NotificationPayload:
    """Payload for notifications."""
    subject: str
    body: str
    scene_id: str
    checkpoint: str
    severity: str
    qa_context: Optional[Dict[str, Any]] = None
    action_required: bool = True
    dashboard_url: Optional[str] = None

    def to_email_html(self) -> str:
        """Convert to HTML email format."""
        severity_colors = {
            "error": "#dc3545",
            "warning": "#ffc107",
            "info": "#17a2b8",
        }
        color = severity_colors.get(self.severity.lower(), "#6c757d")

        qa_section = ""
        if self.qa_context:
            items = []
            for key, value in self.qa_context.items():
                if isinstance(value, list):
                    value = "<br>".join(f"  - {item}" for item in value)
                items.append(f"<li><strong>{key}:</strong> {value}</li>")
            qa_section = f"""
            <h3>What to Review</h3>
            <ul>{"".join(items)}</ul>
            """

        action_section = ""
        if self.action_required:
            action_section = """
            <div style="background:#fff3cd;padding:15px;border-radius:5px;margin:20px 0">
                <strong>Action Required:</strong> Please review and validate before proceeding.
            </div>
            """

        dashboard_section = ""
        if self.dashboard_url:
            dashboard_section = f"""
            <p><a href="{self.dashboard_url}" style="background:#007bff;color:white;padding:10px 20px;text-decoration:none;border-radius:5px">View in Dashboard</a></p>
            """

        return f"""
        <html>
        <body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:20px">
            <div style="background:{color};color:white;padding:15px;border-radius:5px 5px 0 0">
                <h2 style="margin:0">{self.subject}</h2>
                <p style="margin:5px 0 0 0;opacity:0.9">Scene: {self.scene_id} | Checkpoint: {self.checkpoint}</p>
            </div>
            <div style="border:1px solid #ddd;border-top:none;padding:20px;border-radius:0 0 5px 5px">
                <p>{self.body}</p>
                {qa_section}
                {action_section}
                {dashboard_section}
                <hr style="border:none;border-top:1px solid #eee;margin:20px 0">
                <p style="color:#666;font-size:12px">
                    BlueprintPipeline QA Notification<br>
                    Timestamp: {datetime.utcnow().isoformat()}Z
                </p>
            </div>
        </body>
        </html>
        """

    def to_sms_text(self) -> str:
        """Convert to SMS text format (160 char limit awareness)."""
        # Priority info for SMS
        action = "[ACTION REQUIRED] " if self.action_required else ""
        return f"{action}BlueprintPipeline QA: {self.checkpoint} - {self.subject}. Scene: {self.scene_id}. Check email for details."

    def to_plain_text(self) -> str:
        """Convert to plain text format."""
        lines = [
            f"Subject: {self.subject}",
            f"Scene: {self.scene_id}",
            f"Checkpoint: {self.checkpoint}",
            f"Severity: {self.severity}",
            "",
            self.body,
        ]

        if self.qa_context:
            lines.extend(["", "What to Review:"])
            for key, value in self.qa_context.items():
                if isinstance(value, list):
                    lines.append(f"  {key}:")
                    for item in value:
                        lines.append(f"    - {item}")
                else:
                    lines.append(f"  {key}: {value}")

        if self.action_required:
            lines.extend(["", "ACTION REQUIRED: Please review and validate before proceeding."])

        return "\n".join(lines)

    def to_slack_text(self) -> str:
        """Convert to Slack-friendly text format."""
        action_required = "Yes" if self.action_required else "No"
        lines = [
            f"*{self.subject}*",
            f"*Scene ID:* {self.scene_id}",
            f"*Checkpoint:* {self.checkpoint}",
            f"*Severity:* {self.severity}",
            f"*Action Required:* {action_required}",
        ]

        if self.body:
            lines.extend(["", self.body])

        if self.qa_context:
            lines.append("")
            lines.append("*QA Context:*")
            for key, value in self.qa_context.items():
                if isinstance(value, list):
                    lines.append(f"• *{key}:*")
                    for item in value:
                        lines.append(f"    • {item}")
                else:
                    lines.append(f"• *{key}:* {value}")

        if self.dashboard_url:
            lines.extend(["", f"<{self.dashboard_url}|View in Dashboard>"])

        return "\n".join(lines)


class NotificationService:
    """Service for sending QA notifications via multiple channels."""

    # Default recipients from user request
    DEFAULT_EMAIL = "ohstnhunt@gmail.com"
    DEFAULT_PHONE = "9196389913"

    def __init__(
        self,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        slack_webhook_url: Optional[str] = None,
        channels: Optional[List[NotificationChannel]] = None,
        verbose: bool = True,
    ):
        self.email = email or os.getenv("QA_EMAIL", self.DEFAULT_EMAIL)
        self.phone = phone or os.getenv("QA_PHONE", self.DEFAULT_PHONE)
        self.channels = channels or [NotificationChannel.EMAIL, NotificationChannel.SMS]
        self.verbose = verbose

        # API keys
        self.sendgrid_key = os.getenv("SENDGRID_API_KEY")
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_from = os.getenv("TWILIO_FROM_PHONE")
        self.slack_webhook_url = (
            slack_webhook_url
            or os.getenv("QA_SLACK_WEBHOOK_URL")
            or os.getenv("BP_QUALITY_NOTIFICATIONS_SLACK_WEBHOOK_URL")
        )

        # SMTP fallback
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[NOTIFICATION] {msg}")

    def send(self, payload: NotificationPayload) -> List[NotificationResult]:
        """Send notification via all configured channels."""
        results = []

        for channel in self.channels:
            if channel == NotificationChannel.EMAIL:
                result = self._send_email(payload)
            elif channel == NotificationChannel.SMS:
                result = self._send_sms(payload)
            elif channel == NotificationChannel.SLACK:
                result = self._send_slack(payload)
            elif channel == NotificationChannel.CONSOLE:
                result = self._send_console(payload)
            else:
                result = NotificationResult(
                    success=False,
                    channel=channel,
                    recipient="",
                    message=payload.subject,
                    error=f"Unsupported channel: {channel}",
                )

            results.append(result)

            if result.success:
                self.log(f"Sent {channel.value} to {result.recipient}")
            else:
                self.log(f"Failed {channel.value}: {result.error}")

        return results

    def _send_email(self, payload: NotificationPayload) -> NotificationResult:
        """Send email notification."""
        # Try SendGrid first
        if self.sendgrid_key:
            return self._send_email_sendgrid(payload)

        # Fallback to SMTP
        if self.smtp_host and self.smtp_user:
            return self._send_email_smtp(payload)

        # Console fallback
        self.log(f"Email notification (no provider configured):\n{payload.to_plain_text()}")
        return NotificationResult(
            success=True,  # Console is "successful"
            channel=NotificationChannel.EMAIL,
            recipient=self.email,
            message=payload.subject,
            error="No email provider configured - logged to console",
        )

    def _send_email_sendgrid(self, payload: NotificationPayload) -> NotificationResult:
        """Send email via SendGrid."""
        try:
            from_email = os.getenv("SENDGRID_FROM_EMAIL", "qa@blueprintpipeline.com")

            data = {
                "personalizations": [{"to": [{"email": self.email}]}],
                "from": {"email": from_email, "name": "BlueprintPipeline QA"},
                "subject": f"[QA] {payload.subject}",
                "content": [
                    {"type": "text/plain", "value": payload.to_plain_text()},
                    {"type": "text/html", "value": payload.to_email_html()},
                ],
            }

            req = urllib.request.Request(
                "https://api.sendgrid.com/v3/mail/send",
                data=json.dumps(data).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.sendgrid_key}",
                    "Content-Type": "application/json",
                },
            )

            urllib.request.urlopen(req, timeout=30)

            return NotificationResult(
                success=True,
                channel=NotificationChannel.EMAIL,
                recipient=self.email,
                message=payload.subject,
            )

        except Exception as e:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.EMAIL,
                recipient=self.email,
                message=payload.subject,
                error=str(e),
            )

    def _send_email_smtp(self, payload: NotificationPayload) -> NotificationResult:
        """Send email via SMTP."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[QA] {payload.subject}"
            msg["From"] = self.smtp_user
            msg["To"] = self.email

            msg.attach(MIMEText(payload.to_plain_text(), "plain"))
            msg.attach(MIMEText(payload.to_email_html(), "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            return NotificationResult(
                success=True,
                channel=NotificationChannel.EMAIL,
                recipient=self.email,
                message=payload.subject,
            )

        except Exception as e:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.EMAIL,
                recipient=self.email,
                message=payload.subject,
                error=str(e),
            )

    def _send_sms(self, payload: NotificationPayload) -> NotificationResult:
        """Send SMS notification via Twilio."""
        if not (self.twilio_sid and self.twilio_token and self.twilio_from):
            self.log(f"SMS notification (Twilio not configured): {payload.to_sms_text()}")
            return NotificationResult(
                success=True,  # Console is "successful"
                channel=NotificationChannel.SMS,
                recipient=self.phone,
                message=payload.subject,
                error="Twilio not configured - logged to console",
            )

        try:
            # Format phone number
            phone = self.phone
            if not phone.startswith("+"):
                phone = f"+1{phone}"  # Assume US

            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_sid}/Messages.json"

            data = urllib.parse.urlencode({
                "To": phone,
                "From": self.twilio_from,
                "Body": payload.to_sms_text(),
            }).encode("utf-8")

            # Basic auth
            import base64
            auth = base64.b64encode(f"{self.twilio_sid}:{self.twilio_token}".encode()).decode()

            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Authorization": f"Basic {auth}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )

            urllib.request.urlopen(req, timeout=30)

            return NotificationResult(
                success=True,
                channel=NotificationChannel.SMS,
                recipient=phone,
                message=payload.subject,
            )

        except Exception as e:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.SMS,
                recipient=self.phone,
                message=payload.subject,
                error=str(e),
            )

    def _send_slack(self, payload: NotificationPayload) -> NotificationResult:
        """Send Slack notification via incoming webhook."""
        if not self.slack_webhook_url:
            self.log(f"Slack notification (no webhook configured):\n{payload.to_slack_text()}")
            return NotificationResult(
                success=True,  # Console is "successful"
                channel=NotificationChannel.SLACK,
                recipient="slack-webhook",
                message=payload.subject,
                error="Slack webhook not configured - logged to console",
            )

        try:
            data = {"text": payload.to_slack_text()}
            req = urllib.request.Request(
                self.slack_webhook_url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

            urllib.request.urlopen(req, timeout=30)

            return NotificationResult(
                success=True,
                channel=NotificationChannel.SLACK,
                recipient="slack-webhook",
                message=payload.subject,
            )

        except Exception as e:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.SLACK,
                recipient="slack-webhook",
                message=payload.subject,
                error=str(e),
            )

    def _send_console(self, payload: NotificationPayload) -> NotificationResult:
        """Log notification to console."""
        print("\n" + "=" * 60)
        print("QA NOTIFICATION")
        print("=" * 60)
        print(payload.to_plain_text())
        print("=" * 60 + "\n")

        return NotificationResult(
            success=True,
            channel=NotificationChannel.CONSOLE,
            recipient="console",
            message=payload.subject,
        )


def _normalize_recipients(raw_recipients: Optional[object]) -> list[str]:
    if raw_recipients is None:
        return []
    if isinstance(raw_recipients, str):
        return [item.strip() for item in raw_recipients.split(",") if item.strip()]
    if isinstance(raw_recipients, Sequence):
        return [str(item).strip() for item in raw_recipients if str(item).strip()]
    if isinstance(raw_recipients, Iterable):
        return [str(item).strip() for item in raw_recipients if str(item).strip()]
    return []


def _normalize_channels(raw_channels: Optional[object]) -> list[str]:
    if raw_channels is None:
        return []
    if isinstance(raw_channels, str):
        return [item.strip().lower() for item in raw_channels.split(",") if item.strip()]
    if isinstance(raw_channels, Sequence):
        return [str(item).strip().lower() for item in raw_channels if str(item).strip()]
    if isinstance(raw_channels, Iterable):
        return [str(item).strip().lower() for item in raw_channels if str(item).strip()]
    return []


def _channel_from_string(value: str) -> Optional[NotificationChannel]:
    normalized = value.strip().lower()
    for channel in NotificationChannel:
        if channel.value == normalized:
            return channel
    return None


def build_notification_service(
    config: Optional["QualityConfig"],
    *,
    env: Optional[Mapping[str, str]] = None,
    verbose: bool = True,
) -> Optional["NotificationService"]:
    """Build a NotificationService from quality config and environment variables."""
    env_map = dict(env) if env is not None else os.environ
    notifications = getattr(config, "notifications", None)

    email_recipients = _normalize_recipients(
        getattr(notifications, "email_recipients", None) if notifications else None
    )
    sms_recipients = _normalize_recipients(
        getattr(notifications, "sms_recipients", None) if notifications else None
    )
    slack_webhook_url = (
        getattr(notifications, "slack_webhook_url", None) if notifications else None
    )

    env_email = env_map.get("BP_QUALITY_NOTIFICATIONS_EMAIL_RECIPIENTS")
    if env_email:
        email_recipients = _normalize_recipients(env_email)

    env_sms = env_map.get("BP_QUALITY_NOTIFICATIONS_SMS_RECIPIENTS")
    if env_sms:
        sms_recipients = _normalize_recipients(env_sms)

    env_slack = env_map.get("BP_QUALITY_NOTIFICATIONS_SLACK_WEBHOOK_URL")
    if env_slack:
        slack_webhook_url = env_slack

    if env_map.get("QA_SLACK_WEBHOOK_URL") and not slack_webhook_url:
        slack_webhook_url = env_map.get("QA_SLACK_WEBHOOK_URL")

    channels = _normalize_channels(env_map.get("BP_QUALITY_HUMAN_APPROVAL_NOTIFICATION_CHANNELS"))
    if not channels and config and getattr(config, "human_approval", None):
        channels = _normalize_channels(config.human_approval.notification_channels)

    if not channels:
        if email_recipients:
            channels.append(NotificationChannel.EMAIL.value)
        if sms_recipients:
            channels.append(NotificationChannel.SMS.value)
        if slack_webhook_url:
            channels.append(NotificationChannel.SLACK.value)

    if not channels:
        return None

    resolved_channels: List[NotificationChannel] = []
    seen = set()
    for entry in channels:
        channel = _channel_from_string(entry)
        if channel and channel.value not in seen:
            resolved_channels.append(channel)
            seen.add(channel.value)

    if not resolved_channels:
        return None

    email = email_recipients[0] if email_recipients else None
    phone = sms_recipients[0] if sms_recipients else None

    return NotificationService(
        email=email,
        phone=phone,
        slack_webhook_url=slack_webhook_url,
        channels=resolved_channels,
        verbose=verbose,
    )


# Convenience functions
def send_email_notification(
    subject: str,
    body: str,
    scene_id: str,
    checkpoint: str,
    severity: str = "info",
    qa_context: Optional[Dict[str, Any]] = None,
    action_required: bool = False,
    email: Optional[str] = None,
) -> NotificationResult:
    """Send an email notification.

    Args:
        subject: Notification subject
        body: Notification body
        scene_id: Scene identifier
        checkpoint: Pipeline checkpoint name
        severity: Severity level (error, warning, info)
        qa_context: AI-generated QA guidance
        action_required: Whether human action is needed
        email: Override recipient email

    Returns:
        NotificationResult
    """
    service = NotificationService(
        email=email,
        channels=[NotificationChannel.EMAIL],
    )

    payload = NotificationPayload(
        subject=subject,
        body=body,
        scene_id=scene_id,
        checkpoint=checkpoint,
        severity=severity,
        qa_context=qa_context,
        action_required=action_required,
    )

    results = service.send(payload)
    return results[0] if results else NotificationResult(
        success=False,
        channel=NotificationChannel.EMAIL,
        recipient=email or service.email,
        message=subject,
        error="No results returned",
    )


def send_sms_notification(
    subject: str,
    scene_id: str,
    checkpoint: str,
    severity: str = "info",
    action_required: bool = False,
    phone: Optional[str] = None,
) -> NotificationResult:
    """Send an SMS notification.

    Args:
        subject: Short notification message
        scene_id: Scene identifier
        checkpoint: Pipeline checkpoint name
        severity: Severity level
        action_required: Whether human action is needed
        phone: Override recipient phone

    Returns:
        NotificationResult
    """
    service = NotificationService(
        phone=phone,
        channels=[NotificationChannel.SMS],
    )

    payload = NotificationPayload(
        subject=subject,
        body="",  # SMS is short
        scene_id=scene_id,
        checkpoint=checkpoint,
        severity=severity,
        action_required=action_required,
    )

    results = service.send(payload)
    return results[0] if results else NotificationResult(
        success=False,
        channel=NotificationChannel.SMS,
        recipient=phone or service.phone,
        message=subject,
        error="No results returned",
    )
