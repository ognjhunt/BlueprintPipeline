"""3D-RE-GEN runner — executes 3D reconstruction on a remote GPU VM.

Manages the full lifecycle of a 3D-RE-GEN reconstruction:
1. Ensure the remote VM is ready (SSH, GPU, repo cloned, deps installed)
2. Upload the source image
3. Generate a config.yaml tailored to this scene
4. Execute the 3D-RE-GEN pipeline (steps 1-7)
5. Download and harvest outputs into adapter-expected format

Reference:
    - Paper: https://arxiv.org/abs/2512.17459
    - Project: https://3dregen.jdihlmann.com/
    - GitHub: https://github.com/cgtuebingen/3D-RE-GEN
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.regen3d_runner.vm_executor import (
    VMConfig,
    VMExecutor,
    VMExecutorError,
    SSHConnectionError,
    CommandTimeoutError,
    GPUNotAvailableError,
)
from tools.regen3d_runner.output_harvester import (
    harvest_regen3d_native_output,
    HarvestResult,
)

logger = logging.getLogger(__name__)

# Path to the config template relative to this file
_TEMPLATE_PATH = Path(__file__).parent / "config_template.yaml"
_SETUP_SCRIPT_PATH = Path(__file__).parent / "setup_remote.sh"


@dataclass
class Regen3DConfig:
    """Configuration for 3D-RE-GEN reconstruction."""

    # Remote VM
    vm_host: str = "isaac-sim-ubuntu"
    vm_zone: str = "us-east1-b"
    repo_path: str = "/home/nikhil/3D-RE-GEN"

    # Pipeline steps (1-9; typically 1-7 to skip render+eval)
    steps: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])

    # Execution
    timeout_s: int = 1800  # 30 minutes
    device: str = "cuda:0"
    jobs_per_gpu: int = 1
    use_all_available_cuda: bool = False

    # Model choices
    use_vggt: bool = True
    use_hunyuan21: bool = True

    # Gemini for nanoBanana inpainting
    gemini_api_key: str = ""
    gemini_model_id: str = "gemini-2.5-flash-image-preview"

    # Segmentation labels
    labels: List[str] = field(
        default_factory=lambda: [
            "chair", "table", "sofa", "plant in pot", "lamp", "floor"
        ]
    )

    # Auto-start VM if stopped
    auto_start_vm: bool = False

    @classmethod
    def from_env(cls) -> "Regen3DConfig":
        """Load configuration from environment variables."""
        steps_str = os.getenv("REGEN3D_STEPS", "1,2,3,4,5,6,7")
        steps = [int(s.strip()) for s in steps_str.split(",") if s.strip()]

        labels_str = os.getenv(
            "REGEN3D_LABELS", "chair,table,sofa,plant in pot,lamp,floor"
        )
        labels = [l.strip() for l in labels_str.split(",") if l.strip()]

        return cls(
            vm_host=os.getenv("REGEN3D_VM_HOST", "isaac-sim-ubuntu"),
            vm_zone=os.getenv("REGEN3D_VM_ZONE", "us-east1-b"),
            repo_path=os.getenv("REGEN3D_REPO_PATH", "/home/nikhil/3D-RE-GEN"),
            steps=steps,
            timeout_s=int(os.getenv("REGEN3D_TIMEOUT_S", "1800")),
            device=os.getenv("REGEN3D_DEVICE", "cuda:0"),
            jobs_per_gpu=int(os.getenv("REGEN3D_JOBS_PER_GPU", "1")),
            use_all_available_cuda=os.getenv("REGEN3D_USE_ALL_CUDA", "").lower()
            in ("true", "1", "yes"),
            use_vggt=os.getenv("REGEN3D_USE_VGGT", "true").lower()
            in ("true", "1", "yes"),
            use_hunyuan21=os.getenv("REGEN3D_USE_HUNYUAN21", "true").lower()
            in ("true", "1", "yes"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model_id=os.getenv(
                "REGEN3D_GEMINI_MODEL", "gemini-2.5-flash-image-preview"
            ),
            labels=labels,
            auto_start_vm=os.getenv("REGEN3D_AUTO_START_VM", "").lower()
            in ("true", "1", "yes"),
        )


@dataclass
class ReconstructionResult:
    """Result of a 3D-RE-GEN reconstruction run."""
    success: bool
    scene_id: str
    objects_count: int
    duration_seconds: float
    output_dir: Path
    harvest_result: Optional[HarvestResult] = None
    error: Optional[str] = None
    remote_log: str = ""


class Regen3DRunner:
    """Run 3D-RE-GEN reconstruction on a remote GPU VM."""

    def __init__(
        self,
        config: Optional[Regen3DConfig] = None,
        verbose: bool = True,
    ):
        self.config = config or Regen3DConfig.from_env()
        self.verbose = verbose

        self._vm = VMExecutor(
            VMConfig(
                host=self.config.vm_host,
                zone=self.config.vm_zone,
            ),
            verbose=verbose,
        )

    def log(self, msg: str, level: str = "INFO") -> None:
        getattr(logger, level.lower(), logger.info)(msg)
        if self.verbose:
            print(f"[REGEN3D-RUNNER] {msg}")

    # ─── Public API ──────────────────────────────────────────────────────────

    def run_reconstruction(
        self,
        input_image: Path,
        scene_id: str,
        output_dir: Path,
        environment_type: str = "generic",
    ) -> ReconstructionResult:
        """Run the full 3D-RE-GEN pipeline and harvest outputs.

        Args:
            input_image: Path to the source image (local).
            scene_id: Scene identifier.
            output_dir: Local directory to write regen3d/ output into.
            environment_type: Environment type hint for downstream jobs.

        Returns:
            ReconstructionResult with success status and output details.
        """
        input_image = Path(input_image)
        output_dir = Path(output_dir)
        start_time = time.monotonic()

        if not input_image.is_file():
            return ReconstructionResult(
                success=False,
                scene_id=scene_id,
                objects_count=0,
                duration_seconds=0.0,
                output_dir=output_dir,
                error=f"Input image not found: {input_image}",
            )

        try:
            # Step 1: Ensure VM is ready
            self._ensure_vm_ready()

            # Step 2: Setup 3D-RE-GEN on remote VM
            self._setup_remote()

            # Step 3: Upload input image
            remote_image_path = self._upload_input_image(input_image, scene_id)

            # Step 4: Generate and upload config
            remote_output_dir = f"{self.config.repo_path}/output_{scene_id}"
            self._generate_and_upload_config(
                remote_image_path, remote_output_dir, scene_id
            )

            # Step 5: Run the pipeline
            self.log(
                f"Starting 3D-RE-GEN pipeline (steps {self.config.steps}) "
                f"for scene: {scene_id}"
            )
            rc, stdout, stderr = self._execute_pipeline(
                scene_id, remote_output_dir
            )

            if rc != 0:
                return ReconstructionResult(
                    success=False,
                    scene_id=scene_id,
                    objects_count=0,
                    duration_seconds=time.monotonic() - start_time,
                    output_dir=output_dir,
                    error=f"3D-RE-GEN pipeline failed (exit {rc})",
                    remote_log=stderr[-2000:] if stderr else "",
                )

            # Step 6: Download outputs
            local_native_dir = output_dir / "_native_output"
            self._download_outputs(remote_output_dir, local_native_dir)

            # Step 7: Harvest into adapter format
            regen3d_dir = output_dir
            harvest = harvest_regen3d_native_output(
                native_dir=local_native_dir,
                target_dir=regen3d_dir,
                scene_id=scene_id,
                source_image_path=str(input_image),
                environment_type=environment_type,
            )

            duration = time.monotonic() - start_time
            self.log(
                f"Reconstruction complete: {harvest.objects_count} objects, "
                f"{duration:.1f}s, bg={harvest.has_background}, "
                f"cam={harvest.has_camera}"
            )

            return ReconstructionResult(
                success=True,
                scene_id=scene_id,
                objects_count=harvest.objects_count,
                duration_seconds=duration,
                output_dir=regen3d_dir,
                harvest_result=harvest,
            )

        except SSHConnectionError as exc:
            return ReconstructionResult(
                success=False,
                scene_id=scene_id,
                objects_count=0,
                duration_seconds=time.monotonic() - start_time,
                output_dir=output_dir,
                error=f"SSH connection failed: {exc}",
            )
        except CommandTimeoutError as exc:
            return ReconstructionResult(
                success=False,
                scene_id=scene_id,
                objects_count=0,
                duration_seconds=time.monotonic() - start_time,
                output_dir=output_dir,
                error=f"Command timed out: {exc}",
            )
        except Exception as exc:
            return ReconstructionResult(
                success=False,
                scene_id=scene_id,
                objects_count=0,
                duration_seconds=time.monotonic() - start_time,
                output_dir=output_dir,
                error=f"Unexpected error: {exc}",
            )

    # ─── Internal Methods ────────────────────────────────────────────────────

    def _ensure_vm_ready(self) -> None:
        """Ensure the remote VM is running and accessible."""
        if not self._vm.check_vm_running():
            if self.config.auto_start_vm:
                self.log("VM is stopped. Starting...")
                from subprocess import run
                run(
                    [
                        "gcloud", "compute", "instances", "start",
                        self.config.vm_host,
                        f"--zone={self.config.vm_zone}",
                    ],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                # Wait for SSH to become available
                self.log("Waiting for VM to boot...")
                for attempt in range(12):
                    time.sleep(10)
                    try:
                        self._vm.ssh_exec("echo ready", timeout=10, stream_logs=False)
                        self.log("VM is ready")
                        return
                    except (SSHConnectionError, VMExecutorError):
                        continue
                raise VMExecutorError("VM started but SSH not available after 2 minutes")
            else:
                raise VMExecutorError(
                    f"VM {self.config.vm_host} is not running. "
                    f"Start with: gcloud compute instances start "
                    f"{self.config.vm_host} --zone={self.config.vm_zone}"
                )

        # Verify GPU
        if not self._vm.check_gpu_available():
            raise GPUNotAvailableError(
                f"GPU not available on {self.config.vm_host}. "
                f"Check nvidia-smi on the VM."
            )

    def _setup_remote(self) -> None:
        """Bootstrap 3D-RE-GEN on the remote VM if needed."""
        # Check if repo already exists and has venv
        rc, stdout, _ = self._vm.ssh_exec(
            f"test -d {shlex.quote(self.config.repo_path)}/venv_py310 && echo EXISTS",
            stream_logs=False,
            check=False,
        )

        if "EXISTS" in stdout:
            self.log("3D-RE-GEN environment already set up on VM")
            return

        self.log("Setting up 3D-RE-GEN on remote VM (first time only)...")

        # Upload setup script
        self._vm.scp_upload(
            _SETUP_SCRIPT_PATH,
            "/tmp/setup_regen3d.sh",
        )

        # Execute setup script
        self._vm.ssh_exec(
            f"bash /tmp/setup_regen3d.sh {shlex.quote(self.config.repo_path)}",
            timeout=1200,  # 20 minutes for first-time setup
            stream_logs=True,
            check=True,
        )

        self.log("3D-RE-GEN setup complete")

    def _upload_input_image(self, input_image: Path, scene_id: str) -> str:
        """Upload the source image to the remote VM.

        Returns:
            Remote path to the uploaded image.
        """
        ext = input_image.suffix or ".jpg"
        remote_path = f"{self.config.repo_path}/input_images/{scene_id}{ext}"

        # Ensure remote directory exists
        self._vm.ensure_directory(f"{self.config.repo_path}/input_images")

        self._vm.scp_upload(input_image, remote_path)
        self.log(f"Uploaded input image to {remote_path}")
        return remote_path

    def _generate_and_upload_config(
        self,
        remote_image_path: str,
        remote_output_dir: str,
        scene_id: str,
    ) -> None:
        """Generate config.yaml from template and upload to VM."""
        template = _TEMPLATE_PATH.read_text()

        # Format labels as YAML list
        labels_yaml = "\n".join(f" - {label}" for label in self.config.labels)

        # Substitute placeholders
        config_content = template.format(
            input_image_path=remote_image_path,
            output_dir=remote_output_dir,
            temp_dir=f"{self.config.repo_path}/tmp_{scene_id}",
            device=self.config.device,
            use_all_available_cuda=str(self.config.use_all_available_cuda).lower(),
            jobs_per_gpu=self.config.jobs_per_gpu,
            labels=labels_yaml,
            gemini_model_id=self.config.gemini_model_id,
            use_vggt=str(self.config.use_vggt),
            use_hunyuan21=str(self.config.use_hunyuan21).lower(),
        )

        # Write to temp file and upload
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(config_content)
            temp_path = Path(f.name)

        try:
            remote_config = f"{self.config.repo_path}/src/config.yaml"
            self._vm.scp_upload(temp_path, remote_config)
            self.log(f"Uploaded config.yaml for scene: {scene_id}")
        finally:
            temp_path.unlink(missing_ok=True)

        # Set GEMINI_API_KEY on the remote VM if available
        if self.config.gemini_api_key:
            # Write to a temp env file that run.py can source
            env_cmd = (
                f"echo 'export GEMINI_API_KEY={shlex.quote(self.config.gemini_api_key)}' "
                f"> {shlex.quote(self.config.repo_path)}/.env_keys"
            )
            self._vm.ssh_exec(env_cmd, stream_logs=False, check=True)

    def _execute_pipeline(
        self, scene_id: str, remote_output_dir: str
    ) -> tuple:
        """Execute the 3D-RE-GEN pipeline on the remote VM.

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        steps_arg = " ".join(str(s) for s in self.config.steps)
        repo = shlex.quote(self.config.repo_path)
        venv = f"{self.config.repo_path}/venv_py310"

        # Build the command
        # Source env keys if present, create output dir, run pipeline
        command = (
            f"cd {repo} && "
            f"[ -f .env_keys ] && source .env_keys ; "
            f"mkdir -p {shlex.quote(remote_output_dir)} && "
            f"mamba run -p {shlex.quote(venv)} python run.py -p {steps_arg}"
        )

        return self._vm.ssh_exec(
            command,
            timeout=self.config.timeout_s,
            stream_logs=self.verbose,
            check=False,
        )

    def _download_outputs(
        self, remote_output_dir: str, local_dir: Path
    ) -> None:
        """Download 3D-RE-GEN outputs from the remote VM."""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # Key directories to download
        subdirs = [
            "3D",           # Hunyuan3D meshes
            "glb",          # Optimized scene GLBs
            "pointclouds",  # Point clouds
            "vggt",         # VGGT camera + sparse reconstruction
            "pre_3D",       # Camera parameters
            "findings",     # Segmentation masks, depth
            "masks",        # Binary masks
        ]

        for subdir in subdirs:
            remote_path = f"{remote_output_dir}/{subdir}"
            local_subdir = local_dir / subdir

            # Check if remote directory exists
            rc, stdout, _ = self._vm.ssh_exec(
                f"test -d {shlex.quote(remote_path)} && echo EXISTS",
                stream_logs=False,
                check=False,
            )

            if "EXISTS" not in stdout:
                self.log(f"Remote dir not found (skipping): {subdir}", "WARNING")
                continue

            downloaded = self._vm.scp_download_dir(remote_path, local_subdir)
            self.log(f"Downloaded {len(downloaded)} files from {subdir}/")

    # ─── Cleanup ─────────────────────────────────────────────────────────────

    def cleanup_remote(self, scene_id: str) -> None:
        """Clean up remote output directory after successful harvest."""
        remote_output_dir = f"{self.config.repo_path}/output_{scene_id}"
        remote_tmp_dir = f"{self.config.repo_path}/tmp_{scene_id}"

        self._vm.ssh_exec(
            f"rm -rf {shlex.quote(remote_output_dir)} {shlex.quote(remote_tmp_dir)}",
            stream_logs=False,
            check=False,
        )
        self.log(f"Cleaned up remote directories for {scene_id}")
