import json
import os
import hmac
import hashlib

from flask import Flask, jsonify, request
from google.cloud.workflows import executions_v1
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

app = Flask(__name__)


def _get_project_id() -> str:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT must be set")
    return project_id


@app.get("/healthz")
def health_check():
    return jsonify({"status": "ok"})


def _log_invalid_request(reason: str) -> None:
    auth_header = request.headers.get("Authorization", "")
    auth_scheme = auth_header.split(" ", 1)[0] if auth_header else ""
    metadata = {
        "remote_addr": request.remote_addr,
        "method": request.method,
        "path": request.path,
        "user_agent": request.headers.get("User-Agent"),
        "content_type": request.content_type,
        "content_length": request.content_length,
        "auth_scheme": auth_scheme,
        "has_signature": bool(request.headers.get("X-Webhook-Signature")),
    }
    app.logger.warning("Rejected webhook request: %s | metadata=%s", reason, metadata)


def _verify_hmac_signature(body: bytes, secret: str) -> bool:
    signature = request.headers.get("X-Webhook-Signature", "")
    if not signature.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    provided = signature.split("=", 1)[1]
    return hmac.compare_digest(provided, expected)


def _verify_oidc_token(token: str, audience: str) -> bool:
    try:
        id_token.verify_oauth2_token(token, google_requests.Request(), audience=audience)
    except ValueError:
        return False
    return True


def _is_authenticated(body: bytes) -> bool:
    secret = os.getenv("WEBHOOK_HMAC_SECRET")
    audience = os.getenv("WEBHOOK_OIDC_AUDIENCE")
    if not secret and not audience:
        _log_invalid_request("authentication not configured")
        return False

    if secret and _verify_hmac_signature(body, secret):
        return True

    if audience:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            if token and _verify_oidc_token(token, audience):
                return True

    _log_invalid_request("invalid authentication")
    return False


def _validate_payload(payload: dict) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload must be a JSON object"
    job_id = payload.get("job_id")
    if not isinstance(job_id, str):
        return False, "job_id must be a string"
    status = payload.get("status")
    if status is not None and not isinstance(status, str):
        return False, "status must be a string"
    scene_id = payload.get("scene_id")
    if scene_id is not None and not isinstance(scene_id, str):
        return False, "scene_id must be a string"
    return True, ""


@app.post("/webhooks/geniesim/job-complete")
def handle_job_complete():
    raw_body = request.get_data()
    if not _is_authenticated(raw_body):
        return jsonify({"error": "unauthorized"}), 401

    payload = request.get_json(silent=True)
    is_valid, error = _validate_payload(payload)
    if not is_valid:
        return jsonify({"error": error}), 400

    payload = payload.copy()
    payload.setdefault("status", "completed")

    workflow_name = os.getenv("WORKFLOW_NAME", "genie-sim-import-pipeline")
    region = os.getenv("WORKFLOW_REGION", "us-central1")

    client = executions_v1.ExecutionsClient()
    parent = f"projects/{_get_project_id()}/locations/{region}/workflows/{workflow_name}"
    execution = executions_v1.Execution(argument=json.dumps(payload))
    response = client.create_execution(request={"parent": parent, "execution": execution})

    return jsonify({"status": "triggered", "execution": response.name}), 202


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
