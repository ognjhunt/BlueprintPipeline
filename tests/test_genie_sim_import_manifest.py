import importlib.util
import json
import sys
from pathlib import Path


def _load_verify_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "genie-sim-import-job" / "verify_import_manifest.py"
    module_dir = module_path.parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    spec = importlib.util.spec_from_file_location("verify_import_manifest", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load verify_import_manifest module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_verify_manifest_missing_required_fields(tmp_path, capsys):
    verify_module = _load_verify_module()
    manifest_path = tmp_path / "import_manifest.json"
    manifest = {
        "output_dir": str(tmp_path),
        "checksums": {
            "metadata": {},
            "episodes": {},
            "missing_episode_ids": [],
            "missing_metadata_files": [],
        },
    }
    manifest_path.write_text(json.dumps(manifest))

    exit_code = verify_module.verify_manifest(manifest_path)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Unsupported schema_version" in captured.out


def test_verify_manifest_reports_invalid_paths_and_metadata(tmp_path, capsys):
    verify_module = _load_verify_module()
    utils_module = _load_utils_module()

    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    present_file = output_dir / "present.txt"
    present_file.write_text("data")

    manifest = {
        "schema_version": utils_module.MANIFEST_SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "file_inventory": [
            {"path": "present.txt", "size_bytes": present_file.stat().st_size + 1},
            {"path": "missing.txt", "size_bytes": 1},
        ],
        "checksums": {
            "metadata": {
                "present.txt": {
                    "sha256": utils_module.compute_sha256(present_file),
                    "size_bytes": present_file.stat().st_size + 1,
                },
                "missing.txt": {
                    "sha256": "deadbeef",
                    "size_bytes": 1,
                },
            },
            "episodes": {},
            "missing_episode_ids": [],
            "missing_metadata_files": ["missing_meta.json"],
        },
    }
    manifest_path = tmp_path / "import_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    exit_code = verify_module.verify_manifest(manifest_path)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Missing inventory file: missing.txt" in captured.out
    assert "Size mismatch for present.txt" in captured.out
    assert "Missing checksum file: missing.txt" in captured.out
    assert "Missing metadata files recorded in manifest" in captured.out
