import importlib.util
import json
import sys
from pathlib import Path


def _load_utils_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "genie-sim-import-job" / "import_manifest_utils.py"
    module_dir = module_path.parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    spec = importlib.util.spec_from_file_location("import_manifest_utils", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load import_manifest_utils module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_manifest(utils_module):
    return {
        "schema_version": utils_module.MANIFEST_SCHEMA_VERSION,
        "output_dir": ".",
        "checksums": {
            "metadata": {},
            "episodes": {},
        },
    }


def test_verify_import_manifest_checksum_valid(tmp_path):
    utils_module = _load_utils_module()
    manifest = _build_manifest(utils_module)
    checksum = utils_module.compute_manifest_checksum(manifest)
    manifest["checksums"]["metadata"]["import_manifest.json"] = {"sha256": checksum}

    manifest_path = tmp_path / "import_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    result = utils_module.verify_import_manifest_checksum(manifest_path)

    assert result["success"] is True
    assert result["expected"] == checksum
    assert result["actual"] == checksum
    assert result["errors"] == []


def test_verify_import_manifest_checksum_invalid(tmp_path):
    utils_module = _load_utils_module()
    manifest = _build_manifest(utils_module)
    manifest["checksums"]["metadata"]["import_manifest.json"] = {"sha256": "bad"}

    manifest_path = tmp_path / "import_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    result = utils_module.verify_import_manifest_checksum(manifest_path)

    assert result["success"] is False
    assert result["expected"] == "bad"
    assert result["actual"] is not None
    assert result["errors"]


def test_verify_import_manifest_checksum_missing_entry(tmp_path):
    utils_module = _load_utils_module()
    manifest = _build_manifest(utils_module)

    manifest_path = tmp_path / "import_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    result = utils_module.verify_import_manifest_checksum(manifest_path)

    assert result["success"] is False
    assert result["expected"] is None
    assert result["errors"]
