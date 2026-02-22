#!/usr/bin/env python3
"""SceneSmith-compatible Stage 1 backend service.

This service exposes the HTTP contract expected by
`tools/source_pipeline/generator.py` when `SCENESMITH_SERVER_URL` is set.

Modes:
- internal: use in-repo SceneSmith strategy implementation
- command:  invoke external process defined by SCENESMITH_COMMAND
- http_forward: forward request payload to SCENESMITH_UPSTREAM_URL
- paper_stack: command bridge that runs official SceneSmith and converts outputs
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shlex
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping
from urllib import error as url_error
from urllib import request as url_request

from flask import Flask, jsonify, request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.source_pipeline.generator import SceneSmithGeneratorStrategy, TextGenerationContext
from tools.source_pipeline.request import QualityTier

app = Flask(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

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


def _json_error(status_code: int, message: str) -> tuple[Any, int]:
    return jsonify({"status": "error", "error": message}), status_code


@contextlib.contextmanager
def _temporary_env(key: str, value: str) -> Iterator[None]:
    previous = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _safe_int(raw: Any, *, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _safe_float(raw: Any, *, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return value


def _parse_quality_tier(raw: Any) -> QualityTier:
    if raw is None:
        return QualityTier.STANDARD
    try:
        return QualityTier(str(raw).strip().lower())
    except ValueError:
        return QualityTier.STANDARD


def _as_constraints(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


def _build_context(payload: Mapping[str, Any]) -> TextGenerationContext:
    scene_id = str(payload.get("scene_id") or "").strip() or f"scene_{uuid.uuid4().hex[:12]}"
    prompt = str(payload.get("prompt") or "").strip()
    quality_tier = _parse_quality_tier(payload.get("quality_tier"))
    seed = _safe_int(payload.get("seed"), default=1)
    provider_policy = (
        str(payload.get("provider_policy") or "openrouter_qwen_primary").strip()
        or "openrouter_qwen_primary"
    )
    constraints = _as_constraints(payload.get("constraints"))
    return TextGenerationContext(
        scene_id=scene_id,
        prompt=prompt,
        quality_tier=quality_tier,
        seed=seed,
        provider_policy=provider_policy,
        constraints=constraints,
    )


def _build_internal_package(context: TextGenerationContext) -> Dict[str, Any]:
    # Force local generation path inside the service itself.
    with _temporary_env("SCENESMITH_SERVER_URL", ""):
        strategy = SceneSmithGeneratorStrategy()
        package = strategy.generate(context)

    report = package.get("quality_gate_report")
    if isinstance(report, dict):
        report["provider"] = "scenesmith"
        report["generation_mode"] = "scenesmith_service_internal"

    scenesmith_payload = package.get("scenesmith")
    if not isinstance(scenesmith_payload, dict):
        scenesmith_payload = {}
    scenesmith_payload["service_mode"] = "internal"
    package["scenesmith"] = scenesmith_payload
    package["text_backend"] = "scenesmith"
    return package


def _run_command_backend(payload: Mapping[str, Any]) -> Dict[str, Any]:
    command = os.getenv("SCENESMITH_COMMAND", "").strip()
    if not command:
        raise RuntimeError("SCENESMITH_COMMAND is required when SCENESMITH_SERVICE_MODE=command")

    timeout = _safe_float(os.getenv("SCENESMITH_SERVICE_TIMEOUT_SECONDS", "3600"), default=3600.0)
    completed = subprocess.run(
        shlex.split(command),
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(f"SceneSmith command failed ({completed.returncode}): {stderr}")

    stdout = (completed.stdout or "").strip()
    if not stdout:
        raise RuntimeError("SceneSmith command produced empty stdout")

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"SceneSmith command returned invalid JSON: {exc}") from exc

    if not isinstance(parsed, Mapping):
        raise RuntimeError("SceneSmith command output must be a JSON object")
    return dict(parsed)


def _run_http_forward(payload: Mapping[str, Any]) -> Dict[str, Any]:
    endpoint = os.getenv("SCENESMITH_UPSTREAM_URL", "").strip()
    if not endpoint:
        raise RuntimeError("SCENESMITH_UPSTREAM_URL is required when SCENESMITH_SERVICE_MODE=http_forward")

    timeout = _safe_float(os.getenv("SCENESMITH_SERVICE_TIMEOUT_SECONDS", "3600"), default=3600.0)
    req = url_request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with url_request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except (url_error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(f"SceneSmith upstream request failed: {exc}") from exc

    if not raw:
        raise RuntimeError("SceneSmith upstream returned empty response")

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SceneSmith upstream returned invalid JSON: {exc}") from exc

    if not isinstance(parsed, Mapping):
        raise RuntimeError("SceneSmith upstream response must be a JSON object")
    return dict(parsed)


def _run_paper_stack(payload: Mapping[str, Any]) -> Dict[str, Any]:
    paper_command = os.getenv("SCENESMITH_PAPER_COMMAND", "").strip()
    if not paper_command:
        adapter = REPO_ROOT / "scenesmith-service" / "scenesmith_paper_command.py"
        paper_command = f"{shlex.quote(sys.executable)} {shlex.quote(str(adapter))}"

    with _temporary_env("SCENESMITH_COMMAND", paper_command):
        return _run_command_backend(payload)


def _normalize_backend_response(result: Mapping[str, Any], *, request_id: str) -> Dict[str, Any]:
    response = dict(result)
    if not isinstance(response.get("package"), Mapping) and not isinstance(response.get("objects"), list):
        # Treat plain object payload as a package body.
        response = {"package": dict(result)}
    response.setdefault("schema_version", "v1")
    response.setdefault("request_id", request_id)
    return response


def _invoke_backend(payload: Mapping[str, Any], context: TextGenerationContext, request_id: str) -> Dict[str, Any]:
    mode = os.getenv("SCENESMITH_SERVICE_MODE", "internal").strip().lower() or "internal"

    if mode == "internal":
        return {
            "schema_version": "v1",
            "request_id": request_id,
            "package": _build_internal_package(context),
            "service": {"name": "scenesmith", "mode": mode},
        }
    if mode == "command":
        return _normalize_backend_response(_run_command_backend(payload), request_id=request_id)
    if mode == "http_forward":
        return _normalize_backend_response(_run_http_forward(payload), request_id=request_id)
    if mode in {"paper", "paper_stack"}:
        return _normalize_backend_response(_run_paper_stack(payload), request_id=request_id)

    raise RuntimeError(f"Unsupported SCENESMITH_SERVICE_MODE={mode!r}")


@app.route("/", methods=["GET"])
def root() -> tuple[Any, int]:
    return healthz()


@app.route("/healthz", methods=["GET"])
def healthz() -> tuple[Any, int]:
    mode = os.getenv("SCENESMITH_SERVICE_MODE", "internal").strip().lower() or "internal"
    payload = {
        "status": "ok",
        "service": "scenesmith",
        "mode": mode,
    }
    return jsonify(payload), 200


@app.route("/v1/generate", methods=["POST"])
def generate() -> tuple[Any, int]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, Mapping):
        return _json_error(400, "Request body must be a JSON object")

    context = _build_context(payload)
    if not context.prompt:
        return _json_error(400, "prompt is required")

    request_id = str(payload.get("request_id") or uuid.uuid4().hex)
    try:
        response = _invoke_backend(payload, context, request_id)
    except Exception as exc:  # pragma: no cover - defensive runtime path
        logger.exception("SceneSmith service request failed")
        return _json_error(500, str(exc))

    return jsonify(response), 200


if __name__ == "__main__":
    port = _safe_int(os.getenv("PORT", "8081"), default=8081)
    app.run(host="0.0.0.0", port=port)
