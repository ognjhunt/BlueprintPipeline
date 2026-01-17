import json
import os
import hmac
import hashlib
import ipaddress
import subprocess
import urllib.error
import urllib.request
from urllib.parse import urlparse

from flask import Flask, jsonify, request
from google.cloud.workflows import executions_v1
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

app = Flask(__name__)

_BLOCKED_HEALTHCHECK_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("169.254.169.254/32"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),
]


def _parse_allowed_health_hosts() -> set[str]:
    raw_hosts = os.getenv("HEALTHCHECK_ALLOWED_HOSTS", "")
    if not raw_hosts:
        return set()
    return {item.strip().lower() for item in raw_hosts.split(",") if item.strip()}


def _validate_health_url(url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "invalid_url"

    scheme = (parsed.scheme or "").lower()
    if scheme != "https":
        return False, "invalid_scheme"

    hostname = parsed.hostname
    if not hostname:
        return False, "missing_hostname"

    hostname = hostname.lower()
    if hostname == "localhost":
        return False, "localhost_not_allowed"

    allowed_hosts = _parse_allowed_health_hosts()
    if allowed_hosts and hostname not in allowed_hosts:
        return False, "host_not_allowed"

    try:
        host_ip = ipaddress.ip_address(hostname)
    except ValueError:
        return True, ""

    for blocked_network in _BLOCKED_HEALTHCHECK_NETWORKS:
        if host_ip in blocked_network:
            return False, "blocked_ip_range"

    return True, ""


def _http_probe(url: str, timeout_s: float) -> dict:
    result = {"ok": False, "url": url}
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            status_code = response.getcode()
            result["status_code"] = status_code
            result["ok"] = 200 <= status_code < 300
            if not result["ok"]:
                result["error"] = f"non_2xx_status:{status_code}"
    except urllib.error.HTTPError as exc:
        result["status_code"] = exc.code
        result["error"] = f"http_error:{exc.code}"
    except Exception as exc:
        result["error"] = f"request_failed:{exc}"
    return result


def _process_probe(pattern: str, timeout_s: float) -> dict:
    result = {"ok": False, "pattern": pattern}
    try:
        completed = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        result["ok"] = completed.returncode == 0
        if not result["ok"]:
            result["error"] = "process_not_found"
    except FileNotFoundError:
        result["error"] = "pgrep_not_available"
    except Exception as exc:
        result["error"] = f"process_probe_failed:{exc}"
    return result


def _gpu_probe(timeout_s: float) -> dict:
    result = {"ok": False}
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        result["stdout"] = completed.stdout.strip()
        result["stderr"] = completed.stderr.strip()
        result["ok"] = completed.returncode == 0
        if not result["ok"]:
            result["error"] = "nvidia_smi_failed"
    except FileNotFoundError:
        result["error"] = "nvidia_smi_not_found"
    except Exception as exc:
        result["error"] = f"gpu_probe_failed:{exc}"
    return result


def _llm_probe(timeout_s: float) -> dict:
    result = {"ok": False}
    health_url = os.getenv("LLM_HEALTH_URL")
    if health_url:
        is_valid, error = _validate_health_url(health_url)
        if not is_valid:
            result["url"] = health_url
            result["error"] = "invalid_health_url"
            result["details"] = {"reason": error}
            return result
        probe = _http_probe(health_url, timeout_s)
        result.update(probe)
        return result

    token_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ]
    for key in token_keys:
        if os.getenv(key):
            result["ok"] = True
            result["status"] = "token_present"
            result["token_source"] = key
            return result

    result["error"] = "missing_credentials"
    return result


def _isaac_sim_probe(timeout_s: float) -> dict:
    result = {"ok": False}
    health_url = os.getenv("ISAAC_SIM_HEALTH_URL")
    if health_url:
        is_valid, error = _validate_health_url(health_url)
        if not is_valid:
            result["url"] = health_url
            result["error"] = "invalid_health_url"
            result["details"] = {"reason": error}
            return result
        probe = _http_probe(health_url, timeout_s)
        result.update(probe)
        return result

    process_pattern = os.getenv("ISAAC_SIM_PROCESS_PATTERN")
    if process_pattern:
        probe = _process_probe(process_pattern, timeout_s)
        result.update(probe)
        return result

    result["ok"] = True
    result["status"] = "skipped"
    result["reason"] = "not_configured"
    return result


def _dependency_health() -> tuple[bool, dict]:
    timeout_s = float(os.getenv("HEALTH_PROBE_TIMEOUT_S", "2.0"))
    gpu_required = os.getenv("GPU_HEALTH_REQUIRED", "false").lower() == "true"
    isaac_required = os.getenv("ISAAC_SIM_HEALTH_REQUIRED", "false").lower() == "true"
    llm_required = os.getenv("LLM_HEALTH_REQUIRED", "false").lower() == "true"

    dependencies = {
        "gpu": _gpu_probe(timeout_s),
        "isaac_sim": _isaac_sim_probe(timeout_s),
        "llm": _llm_probe(timeout_s),
    }
    errors = []

    for name, required in [
        ("gpu", gpu_required),
        ("isaac_sim", isaac_required),
        ("llm", llm_required),
    ]:
        details = dependencies[name]
        details["required"] = required
        if required and not details.get("ok", False):
            errors.append({
                "dependency": name,
                "error": details.get("error", "dependency_unavailable"),
                "details": details,
            })

    return not errors, {"dependencies": dependencies, "errors": errors}


def _get_project_id() -> str:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT must be set")
    return project_id


def _is_production_mode() -> bool:
    return os.getenv("ENV", "").lower() == "production"


def _validate_startup_auth_configuration() -> None:
    secret = os.getenv("WEBHOOK_HMAC_SECRET")
    audience = os.getenv("WEBHOOK_OIDC_AUDIENCE")
    if secret or audience:
        return
    message = (
        "Webhook authentication is not configured. Set WEBHOOK_HMAC_SECRET for HMAC "
        "verification or WEBHOOK_OIDC_AUDIENCE for OIDC token validation to secure requests."
    )
    if _is_production_mode():
        raise RuntimeError(
            "Missing webhook authentication configuration for production. "
            "Set WEBHOOK_HMAC_SECRET or WEBHOOK_OIDC_AUDIENCE."
        )
    app.logger.warning(message)


_validate_startup_auth_configuration()


@app.get("/healthz")
def health_check():
    deps_ok, dep_details = _dependency_health()
    if not deps_ok:
        return jsonify({
            "status": "error",
            "message": "Dependency check failed",
            "dependencies": dep_details.get("dependencies"),
            "errors": dep_details.get("errors"),
        }), 503

    return jsonify({
        "status": "ok",
        "dependencies": dep_details.get("dependencies"),
    })


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
        if _is_production_mode():
            raise RuntimeError(
                "Missing webhook authentication configuration for production. "
                "Set WEBHOOK_HMAC_SECRET or WEBHOOK_OIDC_AUDIENCE."
            )
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
