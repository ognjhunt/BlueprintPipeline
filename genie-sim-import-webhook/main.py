import json
import os
import hmac
import hashlib
import ipaddress
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, jsonify, request
from google.api_core import exceptions as google_exceptions
from google.api_core import retry
from google.cloud import firestore
from google.cloud.workflows import executions_v1
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from pydantic import BaseModel, Field, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.config import load_pipeline_config
from tools.config.env import parse_bool_env
from tools.tracing import init_tracing

app = Flask(__name__)
init_tracing(service_name=os.getenv("OTEL_SERVICE_NAME", "genie-sim-import-webhook"))

_SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def _parse_allowed_origins() -> set[str]:
    raw_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
    if not raw_origins:
        return set()
    return {origin.strip() for origin in raw_origins.split(",") if origin.strip()}


def _allowed_origin(origin: str, allowed: set[str]) -> str | None:
    if not origin or not allowed:
        return None
    if "*" in allowed:
        return "*"
    if origin in allowed:
        return origin
    return None


@app.after_request
def _apply_security_headers(response):  # type: ignore[override]
    """Apply API security headers.

    These endpoints are API-only. If they are ever exposed to browsers for
    state-changing actions, enforce CSRF tokens on those routes.
    """
    for header, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)

    allowed_origins = _parse_allowed_origins()
    origin = request.headers.get("Origin", "")
    allow_origin = _allowed_origin(origin, allowed_origins)
    if allow_origin:
        response.headers["Access-Control-Allow-Origin"] = allow_origin
        response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        request_headers = request.headers.get("Access-Control-Request-Headers")
        if request_headers:
            response.headers.setdefault("Access-Control-Allow-Headers", request_headers)
        if allow_origin != "*":
            response.headers.setdefault("Vary", "Origin")
    return response

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
    timeout_s = _get_health_probe_timeout_s()
    gpu_required = parse_bool_env(os.getenv("GPU_HEALTH_REQUIRED"), default=False)
    isaac_required = parse_bool_env(os.getenv("ISAAC_SIM_HEALTH_REQUIRED"), default=False)
    llm_required = parse_bool_env(os.getenv("LLM_HEALTH_REQUIRED"), default=False)

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


def _get_health_probe_timeout_s() -> float:
    env_value = os.getenv("HEALTH_PROBE_TIMEOUT_S")
    if env_value is not None:
        return float(env_value)
    pipeline_config = load_pipeline_config()
    return float(pipeline_config.health_checks.probe_timeout_s)


def _get_project_id() -> str:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT must be set")
    return project_id


def _is_production_mode() -> bool:
    return os.getenv("ENV", "").lower() == "production"


def _parse_healthz_bearer_token() -> str | None:
    token = os.getenv("HEALTHZ_BEARER_TOKEN", "").strip()
    return token or None


def _parse_healthz_ip_allowlist() -> list[ipaddress._BaseNetwork]:
    raw_allowlist = os.getenv("HEALTHZ_IP_ALLOWLIST", "")
    if not raw_allowlist:
        return []
    allowlist: list[ipaddress._BaseNetwork] = []
    for item in raw_allowlist.split(","):
        entry = item.strip()
        if not entry:
            continue
        try:
            allowlist.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            app.logger.warning("Invalid healthz allowlist entry: %s", entry)
    return allowlist


def _healthz_client_ip() -> str | None:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip() or None
    return request.remote_addr


def _authorize_health_check():
    if not _is_production_mode():
        return True, None

    bearer_token = _parse_healthz_bearer_token()
    if bearer_token:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return False, (jsonify({
                "status": "error",
                "message": "Missing bearer token",
            }), 401)
        provided_token = auth_header.split(" ", 1)[1]
        if not hmac.compare_digest(provided_token, bearer_token):
            return False, (jsonify({
                "status": "error",
                "message": "Invalid bearer token",
            }), 403)
        return True, None

    allowlist = _parse_healthz_ip_allowlist()
    if allowlist:
        client_ip = _healthz_client_ip()
        if not client_ip:
            return False, (jsonify({
                "status": "error",
                "message": "Client IP not available",
            }), 403)
        try:
            ip_obj = ipaddress.ip_address(client_ip)
        except ValueError:
            return False, (jsonify({
                "status": "error",
                "message": "Invalid client IP",
            }), 403)
        if not any(ip_obj in network for network in allowlist):
            return False, (jsonify({
                "status": "error",
                "message": "Client IP not allowed",
            }), 403)
        return True, None

    return False, (jsonify({
        "status": "error",
        "message": "Health check authentication not configured",
    }), 403)


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
    authorized, response = _authorize_health_check()
    if not authorized:
        return response

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


def _request_metadata() -> dict:
    return {
        "remote_addr": request.remote_addr,
        "method": request.method,
        "path": request.path,
        "user_agent": request.headers.get("User-Agent"),
        "content_type": request.content_type,
        "content_length": request.content_length,
        "request_id": request.headers.get("X-Request-Id") or request.headers.get("X-Correlation-Id"),
    }


def _log_webhook_event(event: str, payload: dict, extra: dict) -> None:
    audit_payload = {
        "event": event,
        "request_metadata": _request_metadata(),
        "job_id": payload.get("job_id"),
        "scene_id": payload.get("scene_id"),
        "status": payload.get("status"),
    }
    if extra:
        audit_payload.update(extra)
    app.logger.info(json.dumps(audit_payload, sort_keys=True))


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


class JobCompletePayload(BaseModel):
    job_id: str = Field(..., min_length=1)
    status: str | None = None
    scene_id: str | None = None


def _format_payload_errors(errors: list[dict]) -> list[str]:
    formatted = []
    for err in errors:
        loc = ".".join(str(part) for part in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        formatted.append(f"{loc}: {msg}" if loc else msg)
    return formatted


def _parse_job_complete_payload(payload: object) -> tuple[dict | None, list[str]]:
    try:
        parsed = JobCompletePayload.model_validate(payload)
    except ValidationError as exc:
        return None, _format_payload_errors(exc.errors())
    return parsed.model_dump(), []


def _dedup_collection_name() -> str:
    return os.getenv("FIRESTORE_DEDUP_COLLECTION", "webhook_dedup")


def _rate_limit_collection_name() -> str:
    return "webhook_rate_limits"


def _get_rate_limit_per_min() -> int | None:
    raw_limit = os.getenv("WEBHOOK_RATE_LIMIT_PER_MIN", "100")
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise RuntimeError("WEBHOOK_RATE_LIMIT_PER_MIN must be an integer") from exc
    if limit <= 0:
        return None
    return limit


def _rate_limit_key(scene_id: str | None, remote_addr: str | None, minute_bucket: int) -> str:
    scoped_scene = scene_id if scene_id else "ip-only"
    scoped_ip = remote_addr if remote_addr else "unknown"
    return f"{scoped_scene}:{scoped_ip}:{minute_bucket}"


def _seconds_until_next_minute(now: datetime) -> int:
    epoch_seconds = int(now.timestamp())
    next_bucket = ((epoch_seconds // 60) + 1) * 60
    return max(0, next_bucket - epoch_seconds)


def _increment_rate_limit(payload: dict) -> dict:
    limit = _get_rate_limit_per_min()
    if limit is None:
        return {"allowed": True, "limit": None, "remaining": None, "retry_after": None}

    now = datetime.now(tz=timezone.utc)
    minute_bucket = int(now.timestamp() // 60)
    key = _rate_limit_key(payload.get("scene_id"), request.remote_addr, minute_bucket)

    client = firestore.Client(project=_get_project_id())
    collection = client.collection(_rate_limit_collection_name())
    document = collection.document(key)
    transaction = client.transaction()

    @firestore.transactional
    def _update_rate_limit(txn):
        snapshot = document.get(transaction=txn)
        current_count = snapshot.get("count", 0) if snapshot.exists else 0
        new_count = current_count + 1
        data = {
            "count": new_count,
            "limit": limit,
            "minute_bucket": minute_bucket,
            "scene_id": payload.get("scene_id"),
            "remote_addr": request.remote_addr,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }
        if snapshot.exists:
            txn.update(document, data)
        else:
            data["created_at"] = firestore.SERVER_TIMESTAMP
            txn.set(document, data)
        return new_count

    new_count = _update_rate_limit(transaction)
    remaining = max(limit - new_count, 0)
    return {
        "allowed": new_count <= limit,
        "limit": limit,
        "remaining": remaining,
        "retry_after": _seconds_until_next_minute(now),
    }


def _rate_limit_headers(rate_limit: dict) -> dict[str, str]:
    headers: dict[str, str] = {}
    limit = rate_limit.get("limit")
    remaining = rate_limit.get("remaining")
    if limit is not None:
        headers["X-RateLimit-Limit"] = str(limit)
        if remaining is not None:
            headers["X-RateLimit-Remaining"] = str(remaining)
    if not rate_limit.get("allowed", True):
        retry_after = rate_limit.get("retry_after")
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
    return headers


def _apply_rate_limit_headers(response, rate_limit: dict):  # type: ignore[override]
    headers = _rate_limit_headers(rate_limit)
    if headers:
        response.headers.update(headers)
    return response


def _idempotency_key(payload: dict) -> str:
    header_key = request.headers.get("X-Request-Id") or request.headers.get("X-Correlation-Id")
    if header_key:
        return header_key.strip()
    job_id = payload.get("job_id", "")
    status = payload.get("status", "")
    digest = hashlib.sha256(f"{job_id}:{status}".encode("utf-8")).hexdigest()
    return digest


def _reserve_dedup_marker(payload: dict) -> tuple[bool, dict]:
    dedup_key = _idempotency_key(payload)
    client = firestore.Client(project=_get_project_id())
    collection = client.collection(_dedup_collection_name())
    document = collection.document(dedup_key)
    marker = {
        "idempotency_key": dedup_key,
        "job_id": payload.get("job_id"),
        "status": payload.get("status"),
        "request_id": request.headers.get("X-Request-Id") or request.headers.get("X-Correlation-Id"),
        "execution_name": None,
        "created_at": firestore.SERVER_TIMESTAMP,
    }
    try:
        document.create(marker)
    except google_exceptions.AlreadyExists:
        existing = document.get()
        return False, existing.to_dict() if existing.exists else {}
    return True, marker


@app.post("/webhooks/geniesim/job-complete")
def handle_job_complete():
    raw_body = request.get_data()
    if not _is_authenticated(raw_body):
        return jsonify({"error": "unauthorized"}), 401

    payload, errors = _parse_job_complete_payload(request.get_json(silent=True))
    if errors:
        return jsonify({"error": "invalid_payload", "details": errors}), 400

    payload = payload.copy()
    payload.setdefault("status", "completed")

    storage_transient_exceptions = (
        google_exceptions.ServiceUnavailable,
        google_exceptions.DeadlineExceeded,
        google_exceptions.TooManyRequests,
        google_exceptions.InternalServerError,
    )

    try:
        rate_limit = _increment_rate_limit(payload)
    except storage_transient_exceptions as exc:
        app.logger.warning(
            "Failed to update rate limit due to transient storage error: %s",
            exc,
        )
        return jsonify({"error": "rate_limit_unavailable"}), 503
    except Exception as exc:
        app.logger.exception("Failed to update rate limit: %s", exc)
        return jsonify({"error": "rate_limit_failed"}), 500

    if not rate_limit.get("allowed", True):
        response = jsonify({
            "error": "rate_limited",
            "retry_after": rate_limit.get("retry_after"),
        })
        response.status_code = 429
        return _apply_rate_limit_headers(response, rate_limit)

    try:
        is_new_marker, marker = _reserve_dedup_marker(payload)
    except storage_transient_exceptions as exc:
        app.logger.warning(
            "Failed to reserve dedup marker due to transient storage error: %s",
            exc,
        )
        response = jsonify({"error": "dedup_unavailable"})
        response.status_code = 503
        return _apply_rate_limit_headers(response, rate_limit)
    except Exception as exc:
        app.logger.exception("Failed to reserve dedup marker: %s", exc)
        response = jsonify({"error": "dedup_failed"})
        response.status_code = 500
        return _apply_rate_limit_headers(response, rate_limit)

    if not is_new_marker:
        execution_name = marker.get("execution_name")
        response = jsonify({
            "status": "duplicate",
            "execution": execution_name,
            "idempotency_key": marker.get("idempotency_key"),
        })
        response.status_code = 200
        return _apply_rate_limit_headers(response, rate_limit)

    _log_webhook_event(
        "webhook.accepted",
        payload,
        {
            "idempotency_key": marker.get("idempotency_key"),
            "request_id": marker.get("request_id"),
            "rate_limit": {
                "limit": rate_limit.get("limit"),
                "remaining": rate_limit.get("remaining"),
                "retry_after": rate_limit.get("retry_after"),
            },
        },
    )

    workflow_name = os.getenv("WORKFLOW_NAME", "genie-sim-import-pipeline")
    region = os.getenv("WORKFLOW_REGION", "us-central1")
    retry_policy = retry.Retry(
        predicate=retry.if_exception_type(
            google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded,
            google_exceptions.TooManyRequests,
            google_exceptions.InternalServerError,
        ),
        initial=0.5,
        maximum=5.0,
        multiplier=2.0,
        deadline=15.0,
    )
    transient_exceptions = (
        google_exceptions.ServiceUnavailable,
        google_exceptions.DeadlineExceeded,
        google_exceptions.TooManyRequests,
        google_exceptions.InternalServerError,
    )

    client = executions_v1.ExecutionsClient()
    parent = f"projects/{_get_project_id()}/locations/{region}/workflows/{workflow_name}"
    execution = executions_v1.Execution(argument=json.dumps(payload))
    try:
        response = client.create_execution(
            request={"parent": parent, "execution": execution},
            retry=retry_policy,
        )
    except Exception as exc:
        try:
            firestore.Client(project=_get_project_id()).collection(
                _dedup_collection_name()
            ).document(_idempotency_key(payload)).delete()
        except Exception:
            app.logger.warning("Failed to delete dedup marker after execution failure.")
        error_details = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "request_metadata": _request_metadata(),
            "workflow": {
                "name": workflow_name,
                "region": region,
                "parent": parent,
            },
        }
        app.logger.exception("Failed to create workflow execution | details=%s", error_details)
        status_code = 502 if isinstance(exc, transient_exceptions) else 500
        response = jsonify({
            "error": "execution_create_failed",
            "details": {
                "message": str(exc),
                "type": type(exc).__name__,
            },
        })
        response.status_code = status_code
        return _apply_rate_limit_headers(response, rate_limit)

    _log_webhook_event(
        "workflow.triggered",
        payload,
        {
            "execution_name": response.name,
            "workflow_name": workflow_name,
            "region": region,
        },
    )

    try:
        firestore.Client(project=_get_project_id()).collection(
            _dedup_collection_name()
        ).document(_idempotency_key(payload)).update({
            "execution_name": response.name,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
    except Exception:
        app.logger.warning("Failed to update dedup marker with execution name.")

    response_body = jsonify({"status": "triggered", "execution": response.name})
    response_body.status_code = 202
    return _apply_rate_limit_headers(response_body, rate_limit)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
