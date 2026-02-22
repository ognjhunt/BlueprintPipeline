"""Remote VM execution abstraction for GPU workloads.

Provides SSH/SCP utilities for running commands on a remote GCE VM,
designed for Stage 1 text generation reconstruction but reusable for any GPU step.

Uses gcloud compute ssh/scp to avoid managing SSH keys directly.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VMConfig:
    """Configuration for a remote GCE VM."""
    host: str = "isaac-sim-ubuntu"
    zone: str = "us-east1-c"
    user: str = ""  # empty = gcloud default
    ssh_timeout_s: int = 30
    max_ssh_retries: int = 3
    retry_backoff_s: float = 5.0


class VMExecutorError(Exception):
    """Base error for VM execution failures."""
    pass


class SSHConnectionError(VMExecutorError):
    """SSH connection could not be established."""
    pass


class CommandTimeoutError(VMExecutorError):
    """Remote command exceeded its timeout."""
    pass


class GPUNotAvailableError(VMExecutorError):
    """GPU is not accessible on the remote VM."""
    pass


class VMExecutor:
    """Execute commands on a remote GCE VM via gcloud SSH."""

    _SECRET_PATTERNS = (
        re.compile(r"(GEMINI_API_KEY\s*=\s*)([^ \n;&|]+)"),
        re.compile(r"((?:HF_TOKEN|HF_HUB_TOKEN|HUGGINGFACE_HUB_TOKEN)\s*=\s*)([^ \n;&|]+)"),
        re.compile(r"(Authorization:\s*Bearer\s+)([^ \n\"']+)"),
        re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
        re.compile(r"hf_[0-9A-Za-z]{20,}"),
    )

    def __init__(self, config: VMConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose

    def _build_ssh_target(self) -> str:
        if self.config.user:
            return f"{self.config.user}@{self.config.host}"
        return self.config.host

    @classmethod
    def _sanitize_command_for_log(cls, command: str) -> str:
        redacted = command
        for pattern in cls._SECRET_PATTERNS:
            if pattern.pattern.startswith("AIza"):
                redacted = pattern.sub("AIza<REDACTED>", redacted)
            elif pattern.pattern.startswith("hf_"):
                redacted = pattern.sub("hf_<REDACTED>", redacted)
            else:
                redacted = pattern.sub(r"\1<REDACTED>", redacted)
        return redacted

    def _build_ssh_cmd(self, command: str, timeout: Optional[int] = None) -> List[str]:
        """Build gcloud compute ssh command."""
        cmd = [
            "gcloud", "compute", "ssh",
            self._build_ssh_target(),
            f"--zone={self.config.zone}",
            "--",
        ]
        # Wrap remote command in timeout if specified
        if timeout:
            cmd.append(f"timeout {timeout} bash -c {shlex.quote(command)}")
        else:
            cmd.append(f"bash -c {shlex.quote(command)}")
        return cmd

    def ssh_exec(
        self,
        command: str,
        timeout: Optional[int] = None,
        stream_logs: bool = True,
        check: bool = True,
    ) -> Tuple[int, str, str]:
        """Execute a command on the remote VM via SSH.

        Args:
            command: Shell command to run remotely.
            timeout: Command timeout in seconds (None = no timeout).
            stream_logs: If True, stream stdout/stderr in real time.
            check: If True, raise on non-zero exit code.

        Returns:
            Tuple of (return_code, stdout, stderr).

        Raises:
            SSHConnectionError: If SSH connection fails after retries.
            CommandTimeoutError: If command exceeds timeout.
            VMExecutorError: If command fails and check=True.
        """
        ssh_cmd = self._build_ssh_cmd(command, timeout)
        last_error: Optional[BaseException] = None

        for attempt in range(1, self.config.max_ssh_retries + 1):
            try:
                safe_command = self._sanitize_command_for_log(command)
                if self.verbose:
                    logger.info(
                        f"[VM] SSH attempt {attempt}/{self.config.max_ssh_retries}: "
                        f"{safe_command[:120]}..."
                    )

                if stream_logs:
                    rc, stdout, stderr = self._exec_streaming(ssh_cmd, timeout, check)
                else:
                    rc, stdout, stderr = self._exec_capture(ssh_cmd, timeout, check)

                # Retry transport-level SSH failures even when check=False.
                if rc == 255:
                    err = SSHConnectionError(
                        f"SSH connection failed (exit 255): {stderr[-300:] if stderr else ''}"
                    )
                    if attempt < self.config.max_ssh_retries:
                        wait = self.config.retry_backoff_s * attempt
                        logger.warning(
                            f"[VM] SSH connection failed (exit 255), retrying in {wait}s..."
                        )
                        time.sleep(wait)
                        last_error = err
                        continue
                    if check:
                        raise err

                return rc, stdout, stderr

            except subprocess.TimeoutExpired:
                raise CommandTimeoutError(
                    f"Command timed out after {timeout}s: "
                    f"{self._sanitize_command_for_log(command)[:100]}"
                )
            except SSHConnectionError as exc:
                last_error = exc
                if attempt < self.config.max_ssh_retries:
                    wait = self.config.retry_backoff_s * attempt
                    logger.warning(
                        f"[VM] SSH connection failed, retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue
                break

        raise SSHConnectionError(
            f"SSH connection failed after {self.config.max_ssh_retries} attempts: "
            f"{last_error}"
        )

    def _exec_streaming(
        self, cmd: List[str], timeout: Optional[int], check: bool
    ) -> Tuple[int, str, str]:
        """Execute with real-time log streaming."""
        stdout_lines = []
        stderr_lines = []

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            import selectors
            sel = selectors.DefaultSelector()
            sel.register(proc.stdout, selectors.EVENT_READ)
            sel.register(proc.stderr, selectors.EVENT_READ)

            start = time.monotonic()
            while proc.poll() is None:
                if timeout and (time.monotonic() - start) > timeout:
                    proc.kill()
                    raise subprocess.TimeoutExpired(cmd, timeout)

                for key, _ in sel.select(timeout=1.0):
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    if key.fileobj is proc.stdout:
                        stdout_lines.append(line)
                        if self.verbose:
                            print(f"[Stage 1 text generation] {line}", end="")
                    else:
                        stderr_lines.append(line)
                        if self.verbose:
                            print(f"[Stage 1 text generation:err] {line}", end="")

            # Drain remaining output
            remaining_out = proc.stdout.read()
            remaining_err = proc.stderr.read()
            if remaining_out:
                stdout_lines.append(remaining_out)
            if remaining_err:
                stderr_lines.append(remaining_err)

            sel.close()
        except Exception:
            proc.kill()
            raise

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        rc = proc.returncode

        if check and rc != 0:
            if rc == 255:
                raise SSHConnectionError(
                    f"SSH connection failed (exit 255): {stderr[-300:] if stderr else ''}"
                )
            # Check for OOM
            combined = stdout + stderr
            if "CUDA out of memory" in combined or "OutOfMemoryError" in combined:
                raise VMExecutorError(
                    f"GPU OOM on remote VM. Consider reducing image resolution "
                    f"or octree_resolution. Exit code: {rc}"
                )
            raise VMExecutorError(
                f"Remote command failed with exit code {rc}.\n"
                f"stderr: {stderr[-500:]}"
            )

        return rc, stdout, stderr

    def _exec_capture(
        self, cmd: List[str], timeout: Optional[int], check: bool
    ) -> Tuple[int, str, str]:
        """Execute and capture output without streaming."""
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if check and result.returncode != 0:
            combined = result.stdout + result.stderr
            if "CUDA out of memory" in combined or "OutOfMemoryError" in combined:
                raise VMExecutorError(
                    f"GPU OOM on remote VM. Exit code: {result.returncode}"
                )
            if result.returncode == 255:
                raise SSHConnectionError(
                    f"SSH connection failed: {result.stderr[-300:]}"
                )
            raise VMExecutorError(
                f"Remote command failed (exit {result.returncode}): "
                f"{result.stderr[-500:]}"
            )

        return result.returncode, result.stdout, result.stderr

    def scp_upload(
        self,
        local_path: Path,
        remote_path: str,
    ) -> None:
        """Upload a single file to the remote VM.

        Uses individual file SCP (not --recurse) to avoid the nested
        directory bug documented in MEMORY.md.
        """
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        target = f"{self._build_ssh_target()}:{remote_path}"
        cmd = [
            "gcloud", "compute", "scp",
            str(local_path),
            target,
            f"--zone={self.config.zone}",
        ]

        logger.info(f"[VM] SCP upload: {local_path.name} -> {remote_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise VMExecutorError(
                f"SCP upload failed: {result.stderr[:300]}"
            )

    def scp_download(
        self,
        remote_path: str,
        local_path: Path,
    ) -> None:
        """Download a single file from the remote VM.

        Uses individual file SCP to avoid nested directory bug.
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        source = f"{self._build_ssh_target()}:{remote_path}"
        cmd = [
            "gcloud", "compute", "scp",
            source,
            str(local_path),
            f"--zone={self.config.zone}",
        ]

        logger.info(f"[VM] SCP download: {remote_path} -> {local_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise VMExecutorError(
                f"SCP download failed: {result.stderr[:300]}"
            )

    def scp_download_dir(
        self,
        remote_dir: str,
        local_dir: Path,
    ) -> List[Path]:
        """Download all files from a remote directory.

        Lists files first, then downloads individually to avoid the
        nested directory bug from MEMORY.md.
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # List remote files. Use -L and include links so symlinked compatibility
        # outputs (e.g., {name}.glb -> {name}_shape.glb) are downloaded too.
        rc, stdout, _ = self.ssh_exec(
            f"find -L {shlex.quote(remote_dir)} \\( -type f -o -type l \\) 2>/dev/null",
            stream_logs=False,
            check=False,
        )

        if rc != 0 or not stdout.strip():
            logger.warning(f"[VM] No files found in remote dir: {remote_dir}")
            return []

        remote_files = [f.strip() for f in stdout.strip().split("\n") if f.strip()]
        downloaded = []

        for remote_file in remote_files:
            # Preserve relative path structure
            rel_path = remote_file.replace(remote_dir.rstrip("/") + "/", "", 1)
            local_file = local_dir / rel_path

            try:
                self.scp_download(remote_file, local_file)
                downloaded.append(local_file)
            except VMExecutorError as exc:
                logger.warning(f"[VM] Failed to download {remote_file}: {exc}")

        logger.info(f"[VM] Downloaded {len(downloaded)}/{len(remote_files)} files")
        return downloaded

    def check_vm_running(self) -> bool:
        """Check if the VM is currently running."""
        cmd = [
            "gcloud", "compute", "instances", "describe",
            self.config.host,
            f"--zone={self.config.zone}",
            "--format=value(status)",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        status = result.stdout.strip()
        return status == "RUNNING"

    def check_gpu_available(self) -> bool:
        """Verify GPU is accessible on the remote VM."""
        try:
            rc, stdout, _ = self.ssh_exec(
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                timeout=15,
                stream_logs=False,
                check=False,
            )
            if rc == 0 and stdout.strip():
                logger.info(f"[VM] GPU available: {stdout.strip()}")
                return True
            return False
        except (SSHConnectionError, CommandTimeoutError):
            return False

    def ensure_directory(self, remote_path: str) -> None:
        """Create a directory on the remote VM if it doesn't exist."""
        self.ssh_exec(
            f"mkdir -p {shlex.quote(remote_path)}",
            stream_logs=False,
            check=True,
        )
