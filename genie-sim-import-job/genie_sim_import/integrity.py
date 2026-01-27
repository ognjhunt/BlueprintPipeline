import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from tools.config.env import parse_bool_env
from tools.config.production_mode import resolve_production_mode
from tools.secrets.secret_manager import get_secret_or_env

from import_manifest_utils import compute_checksums_signature

from genie_sim_import.constants import (
    CHECKSUMS_HMAC_KEY_ENV_VAR,
    CHECKSUMS_HMAC_KEY_ID_ENV_VAR,
    CHECKSUMS_HMAC_KEY_ID_SECRET_ID,
    CHECKSUMS_HMAC_KEY_SECRET_ID,
)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_firebase_verify_checksums(raw_value: Optional[str]) -> bool:
    return parse_bool_env(raw_value, default=True)


def _format_checksums_verification_errors(
    checksums_verification: Mapping[str, Any],
) -> List[str]:
    verification_errors: List[str] = []
    for error in checksums_verification.get("errors") or []:
        verification_errors.append(str(error))
    if checksums_verification.get("missing_files"):
        verification_errors.append(
            "Missing files: " + ", ".join(checksums_verification["missing_files"])
        )
    if checksums_verification.get("checksum_mismatches"):
        mismatch_paths = [
            mismatch["path"]
            for mismatch in checksums_verification["checksum_mismatches"]
            if isinstance(mismatch, dict) and "path" in mismatch
        ]
        if mismatch_paths:
            verification_errors.append(
                "Checksum mismatches: " + ", ".join(mismatch_paths)
            )
    if checksums_verification.get("size_mismatches"):
        mismatch_paths = [
            mismatch["path"]
            for mismatch in checksums_verification["size_mismatches"]
            if isinstance(mismatch, dict) and "path" in mismatch
        ]
        if mismatch_paths:
            verification_errors.append("Size mismatches: " + ", ".join(mismatch_paths))
    return verification_errors


def _write_checksums_file(
    output_dir: Path,
    checksums: Dict[str, Any],
) -> tuple[Path, Optional[Dict[str, str]]]:
    checksums_path = output_dir / "checksums.json"
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "root": ".",
        "algorithm": "sha256",
        "files": checksums,
    }
    signature = None
    production_mode = resolve_production_mode()
    hmac_key = get_secret_or_env(
        CHECKSUMS_HMAC_KEY_SECRET_ID,
        env_var=CHECKSUMS_HMAC_KEY_ENV_VAR,
        fallback_to_env=not production_mode,
    )
    if hmac_key:
        key_id = get_secret_or_env(
            CHECKSUMS_HMAC_KEY_ID_SECRET_ID,
            env_var=CHECKSUMS_HMAC_KEY_ID_ENV_VAR,
            fallback_to_env=not production_mode,
        )
        signature = compute_checksums_signature(payload, hmac_key, key_id=key_id)
        payload["signature"] = signature
    with open(checksums_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return checksums_path, signature
