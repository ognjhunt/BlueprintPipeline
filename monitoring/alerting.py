from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Dict, Optional

SEVERITY_LEVELS = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "critical": 50,
}


def _normalize_severity(severity: Optional[str]) -> str:
    if not severity:
        return "info"
    return severity.strip().lower()


def _severity_allowed(severity: str) -> bool:
    minimum = _normalize_severity(os.getenv("ALERT_MIN_SEVERITY", "warning"))
    severity_level = SEVERITY_LEVELS.get(_normalize_severity(severity), 20)
    minimum_level = SEVERITY_LEVELS.get(minimum, 30)
    return severity_level >= minimum_level


def _build_payload(
    event_type: str,
    summary: str,
    details: Optional[Dict[str, Any]],
    severity: str,
) -> Dict[str, Any]:
    return {
        "event_type": event_type,
        "summary": summary,
        "details": details or {},
        "severity": severity,
        "source": os.getenv("ALERT_SOURCE", "blueprint_pipeline"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _send_webhook(payload: Dict[str, Any]) -> bool:
    url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    if not url:
        print("[alerting] ALERT_WEBHOOK_URL not set; skipping webhook alert.")
        return False

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status = response.status
            if 200 <= status < 300:
                return True
            print(f"[alerting] Webhook returned status {status}.")
            return False
    except Exception as exc:
        print(f"[alerting] Webhook failed: {exc}")
        return False


def send_alert(
    event_type: str,
    summary: str,
    details: Optional[Dict[str, Any]] = None,
    severity: str = "warning",
) -> bool:
    backend = os.getenv("ALERT_BACKEND", "none").strip().lower()
    severity = _normalize_severity(severity)

    if backend == "none":
        return False
    if not _severity_allowed(severity):
        return False

    payload = _build_payload(event_type, summary, details, severity)

    if backend == "webhook":
        return _send_webhook(payload)

    print(f"[alerting] Unknown ALERT_BACKEND '{backend}'; skipping alert.")
    return False
