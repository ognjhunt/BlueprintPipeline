import importlib.util
from pathlib import Path


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    if spec and spec.loader:
        spec.loader.exec_module(module)
    return module


def _load_integrity_audit_module():
    module_path = Path(__file__).resolve().parents[1] / "dataset-delivery-job" / "integrity_audit.py"
    return _load_module("integrity_audit", module_path)


def _load_import_manifest_utils_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "genie-sim-import-job"
        / "import_manifest_utils.py"
    )
    return _load_module("import_manifest_utils", module_path)


def _base_payload():
    return {
        "generated_at": "2024-01-01T00:00:00Z",
        "root": ".",
        "algorithm": "sha256",
        "files": {
            "episode-000.parquet": {"sha256": "abc123", "size_bytes": 10},
        },
    }


def test_checksums_signature_valid():
    manifest_utils = _load_import_manifest_utils_module()
    key = "supersecret"
    payload = _base_payload()
    payload["signature"] = manifest_utils.compute_checksums_signature(
        payload,
        key,
        key_id="unit-test",
    )
    result = manifest_utils.verify_checksums_signature(payload, key)
    assert result["success"] is True


def test_checksums_signature_modified_payload_fails():
    manifest_utils = _load_import_manifest_utils_module()
    key = "supersecret"
    payload = _base_payload()
    payload["signature"] = manifest_utils.compute_checksums_signature(payload, key)
    payload["files"]["episode-000.parquet"]["sha256"] = "tampered"
    result = manifest_utils.verify_checksums_signature(payload, key)
    assert result["success"] is False


def test_missing_signature_fails_in_production():
    integrity_audit = _load_integrity_audit_module()
    payload = _base_payload()
    result = integrity_audit._verify_checksums_signature(
        payload,
        production_mode=True,
        hmac_key="supersecret",
    )
    assert result["success"] is False
