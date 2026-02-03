#!/usr/bin/env python3
"""Verify that geniesim_grpc.proto matches the pinned Genie Sim ref."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GENIESIM_REPO = os.environ.get("GENIESIM_REPO", "https://github.com/AgibotTech/genie_sim.git")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_geniesim_ref() -> str:
    ref_path = _repo_root() / "genie-sim-gpu-job" / "GENIESIM_REF"
    if not ref_path.exists():
        raise SystemExit(f"GENIESIM_REF not found at {ref_path}")
    ref = ref_path.read_text(encoding="utf-8").strip()
    if not ref:
        raise SystemExit("GENIESIM_REF is empty.")
    return ref


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "(no output)"
        raise SystemExit(f"Command failed ({' '.join(command)}): {details}")
    return result.stdout


def _ensure_git(repo_root: Path) -> None:
    if not (repo_root / ".git").exists():
        raise SystemExit(f"{repo_root} is not a git checkout. Set GENIESIM_ROOT to a git clone.")


def _fetch_repo(temp_dir: Path, ref: str) -> Path:
    _run(["git", "init"], cwd=temp_dir)
    _run(["git", "remote", "add", "origin", GENIESIM_REPO], cwd=temp_dir)
    _run(["git", "fetch", "--depth", "1", "origin", ref], cwd=temp_dir)
    _run(["git", "checkout", "FETCH_HEAD"], cwd=temp_dir)
    return temp_dir


def _resolve_repo(ref: str) -> tuple[Path, bool]:
    geniesim_root = os.environ.get("GENIESIM_ROOT")
    if geniesim_root:
        repo_path = Path(geniesim_root).expanduser().resolve()
        if not repo_path.exists():
            raise SystemExit(f"GENIESIM_ROOT does not exist: {repo_path}")
        _ensure_git(repo_path)
        return repo_path, False

    temp_dir = Path(tempfile.mkdtemp(prefix="geniesim-proto-")).resolve()
    try:
        _fetch_repo(temp_dir, ref)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return temp_dir, True


def _candidate_proto_paths(repo_path: Path, ref: str) -> list[str]:
    override = os.environ.get("GENIESIM_PROTO_PATHS")
    if override:
        return [path.strip() for path in override.split(",") if path.strip()]

    output = _run(["git", "ls-tree", "-r", "--name-only", ref], cwd=repo_path)
    return [line for line in output.splitlines() if line.endswith("geniesim_grpc.proto")]


def _select_proto_path(repo_path: Path, ref: str) -> str:
    candidates = _candidate_proto_paths(repo_path, ref)
    if not candidates:
        raise SystemExit(
            "Could not locate geniesim_grpc.proto in the Genie Sim repo. "
            "Set GENIESIM_PROTO_PATHS to the correct path(s)."
        )
    if len(candidates) > 1:
        listing = "\n".join(f"  - {path}" for path in candidates)
        raise SystemExit(
            "Multiple geniesim_grpc.proto files found:\n"
            f"{listing}\n"
            "Set GENIESIM_PROTO_PATHS to the authoritative path."
        )
    return candidates[0]


def _read_proto_content(repo_path: Path, ref: str, proto_path: str) -> bytes:
    output = _run(["git", "show", f"{ref}:{proto_path}"], cwd=repo_path)
    return output.encode("utf-8")


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _cleanup_temp_repo(repo_path: Path, is_temp: bool) -> None:
    if is_temp:
        shutil.rmtree(repo_path, ignore_errors=True)


def main() -> None:
    ref = _read_geniesim_ref()
    repo_path, is_temp = _resolve_repo(ref)
    try:
        proto_path = _select_proto_path(repo_path, ref)
        expected_content = _read_proto_content(repo_path, ref, proto_path)
    finally:
        _cleanup_temp_repo(repo_path, is_temp)

    local_proto = Path(__file__).resolve().parent / "geniesim_grpc.proto"
    if not local_proto.exists():
        raise SystemExit(f"Local proto not found: {local_proto}")

    local_content = local_proto.read_bytes()

    expected_hash = _hash(expected_content)
    local_hash = _hash(local_content)

    if expected_hash != local_hash:
        allow_patch = os.getenv("ALLOW_GENIESIM_PROTO_PATCH", "").strip().lower() in {"1", "true", "yes", "y", "on"}
        if allow_patch:
            print(
                "Genie Sim proto differs from GENIESIM_REF, but ALLOW_GENIESIM_PROTO_PATCH is set. "
                "Skipping strict hash check."
            )
            print(f"Expected (from {ref}): {expected_hash}")
            print(f"Found (local): {local_hash}")
            return
        message = (
            "Genie Sim proto is out of sync with GENIESIM_REF.\n"
            f"Expected (from {ref}): {expected_hash}\n"
            f"Found (local): {local_hash}\n"
            "Run: python tools/geniesim_adapter/update_geniesim_proto.py"
        )
        raise SystemExit(message)

    print("Genie Sim proto is in sync with GENIESIM_REF.")


if __name__ == "__main__":
    main()
