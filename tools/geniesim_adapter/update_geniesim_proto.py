#!/usr/bin/env python3
"""Sync Genie Sim proto and regenerate gRPC stubs."""

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
    candidates = [line for line in output.splitlines() if line.endswith("geniesim_grpc.proto")]
    return candidates


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


def _write_proto(target: Path, content: bytes) -> None:
    target.write_bytes(content)


def _hash_content(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _print_hash(label: str, content: bytes) -> None:
    print(f"{label}: {_hash_content(content)}")


def _run_regenerate_stubs(proto_root: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "tools.geniesim_adapter.regenerate_stubs",
    ]
    result = subprocess.run(command, check=False, cwd=str(_repo_root()))
    if result.returncode != 0:
        raise SystemExit("Failed to regenerate gRPC stubs. See output above for details.")


def _cleanup_temp_repo(repo_path: Path, is_temp: bool) -> None:
    if is_temp:
        shutil.rmtree(repo_path, ignore_errors=True)


def main() -> None:
    ref = _read_geniesim_ref()
    repo_path, is_temp = _resolve_repo(ref)
    try:
        proto_path = _select_proto_path(repo_path, ref)
        proto_content = _read_proto_content(repo_path, ref, proto_path)
        target = Path(__file__).resolve().parent / "geniesim_grpc.proto"
        _write_proto(target, proto_content)
        print(f"Synced {proto_path} to {target}")
        _print_hash("Proto SHA256", proto_content)
        _run_regenerate_stubs(target.parent)
    finally:
        _cleanup_temp_repo(repo_path, is_temp)


if __name__ == "__main__":
    main()
