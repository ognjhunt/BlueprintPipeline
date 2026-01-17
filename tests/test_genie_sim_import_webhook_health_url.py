import importlib.util
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
WEBHOOK_PATH = ROOT / "genie-sim-import-webhook" / "main.py"


def _install_google_cloud_mocks() -> None:
    google = ModuleType("google")
    cloud = ModuleType("google.cloud")
    workflows = ModuleType("google.cloud.workflows")
    executions_v1 = ModuleType("google.cloud.workflows.executions_v1")
    workflows.executions_v1 = executions_v1

    auth = ModuleType("google.auth")
    auth_transport = ModuleType("google.auth.transport")
    auth_requests = ModuleType("google.auth.transport.requests")
    auth_transport.requests = auth_requests

    oauth2 = ModuleType("google.oauth2")
    id_token = ModuleType("google.oauth2.id_token")
    oauth2.id_token = id_token

    sys.modules.setdefault("google", google)
    sys.modules.setdefault("google.cloud", cloud)
    sys.modules.setdefault("google.cloud.workflows", workflows)
    sys.modules.setdefault("google.cloud.workflows.executions_v1", executions_v1)
    sys.modules.setdefault("google.auth", auth)
    sys.modules.setdefault("google.auth.transport", auth_transport)
    sys.modules.setdefault("google.auth.transport.requests", auth_requests)
    sys.modules.setdefault("google.oauth2", oauth2)
    sys.modules.setdefault("google.oauth2.id_token", id_token)


_install_google_cloud_mocks()

spec = importlib.util.spec_from_file_location(
    "genie_sim_import_webhook_main",
    WEBHOOK_PATH,
)
webhook_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(webhook_main)


def test_validate_health_url_allows_https_hostname(monkeypatch):
    monkeypatch.delenv("HEALTHCHECK_ALLOWED_HOSTS", raising=False)

    ok, error = webhook_main._validate_health_url("https://valid.example.com/healthz")

    assert ok is True
    assert error == ""


def test_validate_health_url_blocks_metadata_http(monkeypatch):
    monkeypatch.delenv("HEALTHCHECK_ALLOWED_HOSTS", raising=False)

    ok, error = webhook_main._validate_health_url("http://169.254.169.254")

    assert ok is False
    assert error == "invalid_scheme"


def test_validate_health_url_enforces_allowlist(monkeypatch):
    monkeypatch.setenv("HEALTHCHECK_ALLOWED_HOSTS", "internal.example.com,valid.example.com")

    ok, error = webhook_main._validate_health_url("https://valid.example.com/healthz")
    blocked_ok, blocked_error = webhook_main._validate_health_url(
        "https://blocked.example.com/healthz"
    )

    assert ok is True
    assert error == ""
    assert blocked_ok is False
    assert blocked_error == "host_not_allowed"
