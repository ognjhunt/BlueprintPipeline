"""3D-RE-GEN runner — executes 3D reconstruction on a remote GPU VM.

Manages the full lifecycle of a 3D-RE-GEN reconstruction:
1. Ensure the remote VM is ready (SSH, GPU, repo cloned, deps installed)
2. Upload the source image
3. Generate a config.yaml tailored to this scene
4. (Optional) Run SAM3 segmentation to replace Step 1
5. Execute the 3D-RE-GEN pipeline (steps 2-7 or 1-7)
6. Download and harvest outputs into adapter-expected format

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
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

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
_SAM3_SCRIPT_PATH = Path(__file__).parent / "segmentation_sam3.py"
_SETUP_SENTINEL = ".bp_regen3d_setup_ok"
_SAM3_SETUP_SENTINEL = ".bp_sam3_setup_ok"


@dataclass
class Regen3DConfig:
    """Configuration for 3D-RE-GEN reconstruction."""

    # Remote VM
    vm_host: str = "isaac-sim-ubuntu"
    vm_zone: str = "us-east1-d"
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
    enable_texture_hy21: bool = False  # Disabled: Hunyuan3D UNet type mismatch hangs

    # Gemini for nanoBanana inpainting
    gemini_api_key: str = ""
    gemini_model_id: str = "gemini-2.5-flash-image"

    # Segmentation labels (populated dynamically by Gemini auto-labeling)
    labels: List[str] = field(default_factory=list)

    # Segmentation backend: "sam3" (new) or "grounded_sam" (original Step 1)
    seg_backend: str = "sam3"

    # SAM3-specific settings
    sam3_threshold: float = 0.4
    sam3_model: str = "facebook/sam3"

    # Require background mesh (full scene mode)
    require_background: bool = True

    # Auto-start VM if stopped
    auto_start_vm: bool = False

    @classmethod
    def from_env(cls) -> "Regen3DConfig":
        """Load configuration from environment variables."""
        steps_str = os.getenv("REGEN3D_STEPS", "1,2,3,4,5,6,7")
        steps = [int(s.strip()) for s in steps_str.split(",") if s.strip()]

        labels_str = os.getenv("REGEN3D_LABELS", "")
        labels = [l.strip() for l in labels_str.split(",") if l.strip()]

        seg_backend = os.getenv("REGEN3D_SEG_BACKEND", "sam3").lower()

        return cls(
            vm_host=os.getenv("REGEN3D_VM_HOST", "isaac-sim-ubuntu"),
            vm_zone=os.getenv("REGEN3D_VM_ZONE", "us-east1-d"),
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
            enable_texture_hy21=os.getenv("REGEN3D_ENABLE_TEXTURE", "false").lower()
            in ("true", "1", "yes"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model_id=os.getenv(
                "REGEN3D_GEMINI_MODEL", "gemini-2.5-flash-image"
            ),
            labels=labels,
            seg_backend=seg_backend,
            sam3_threshold=float(os.getenv("REGEN3D_SAM3_THRESHOLD", "0.4")),
            sam3_model=os.getenv("REGEN3D_SAM3_MODEL", "facebook/sam3"),
            require_background=os.getenv("REGEN3D_REQUIRE_BACKGROUND", "true").lower()
            in ("true", "1", "yes"),
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

            # Step 4a: Auto-detect labels with Gemini
            if not self.config.labels:
                self._resolve_auto_labels(input_image)

            if not self.config.labels:
                return ReconstructionResult(
                    success=False,
                    scene_id=scene_id,
                    objects_count=0,
                    duration_seconds=time.monotonic() - start_time,
                    output_dir=output_dir,
                    error=(
                        "No segmentation labels available. "
                        "Gemini auto-labeling failed and no REGEN3D_LABELS configured."
                    ),
                )

            self._generate_and_upload_config(
                remote_image_path, remote_output_dir, scene_id
            )

            # Step 4b: Run SAM3 segmentation (replaces 3D-RE-GEN Step 1)
            if self.config.seg_backend == "sam3":
                self._run_sam3_segmentation(
                    remote_image_path, remote_output_dir
                )

            # Step 5: Run the pipeline
            effective_steps = self.config.steps
            if self.config.seg_backend == "sam3" and 1 in effective_steps:
                effective_steps = [s for s in effective_steps if s != 1]
                self.log("SAM3 active — skipping 3D-RE-GEN Step 1")
            self.log(
                f"Starting 3D-RE-GEN pipeline (steps {effective_steps}) "
                f"for scene: {scene_id}"
            )
            rc, stdout, stderr = self._execute_pipeline(
                scene_id, remote_output_dir, steps_override=effective_steps
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
            if local_native_dir.exists():
                shutil.rmtree(local_native_dir)
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

            # Validate background mesh for full scene mode
            if not harvest.has_background:
                msg = (
                    "Background mesh not produced by 3D-RE-GEN. "
                    "Full scene mode requires background for lighting and physics."
                )
                if self.config.require_background:
                    return ReconstructionResult(
                        success=False,
                        scene_id=scene_id,
                        objects_count=harvest.objects_count,
                        duration_seconds=duration,
                        output_dir=regen3d_dir,
                        error=msg,
                        harvest_result=harvest,
                    )
                else:
                    self.log(msg, "WARNING")

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
        # Require explicit setup completion sentinel, not only venv presence.
        setup_sentinel = f"{self.config.repo_path}/{_SETUP_SENTINEL}"
        rc, stdout, _ = self._vm.ssh_exec(
            f"test -f {shlex.quote(setup_sentinel)} && echo EXISTS",
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
        labels_yaml = "\n".join(
            f"  - {json.dumps(label)}" for label in self.config.labels
        )

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
            use_vggt=str(self.config.use_vggt).lower(),
            use_hunyuan21=str(self.config.use_hunyuan21).lower(),
            enable_texture_hy21=str(self.config.enable_texture_hy21).lower(),
            conda_env=f"{self.config.repo_path}/venv_py310",
        )

        # Validate rendered YAML before uploading to remote.
        try:
            yaml.safe_load(config_content)
        except Exception as exc:
            raise ValueError(f"Generated config.yaml is invalid: {exc}") from exc

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
        self,
        scene_id: str,
        remote_output_dir: str,
        steps_override: Optional[List[int]] = None,
    ) -> tuple:
        """Execute the 3D-RE-GEN pipeline on the remote VM.

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        steps = steps_override or self.config.steps
        steps_arg = " ".join(str(s) for s in steps)
        repo = shlex.quote(self.config.repo_path)
        venv = f"{self.config.repo_path}/venv_py310"

        # Build the command
        # Source env keys if present, create output dir, run pipeline
        # PYTHONNOUSERSITE=1 is critical: the VM has stale packages in
        # ~/.local/lib/python3.10/ that conflict with the conda env.
        command = (
            f"cd {repo} && "
            f"[ -f .env_keys ] && source .env_keys ; "
            f"export PATH=$HOME/miniforge3/bin:$PATH && "
            f"export PYTHONNOUSERSITE=1 && "
            f"export CUDA_HOME=/usr/local/cuda && "
            f"export PYTORCH_SKIP_CUDA_MISMATCH_CHECK=1 && "
            f"mkdir -p {shlex.quote(remote_output_dir)} && "
            f"./venv_py310/bin/python run.py -p {steps_arg}"
        )

        return self._vm.ssh_exec(
            command,
            timeout=self.config.timeout_s,
            stream_logs=self.verbose,
            check=False,
        )

    def _resolve_auto_labels(self, input_image: Path) -> None:
        """Use Gemini to auto-detect object labels from the input image."""
        self.log("Running Gemini auto-labeling...")
        from tools.regen3d_runner.gemini_label_generator import (
            generate_labels_from_image,
        )
        labels = generate_labels_from_image(str(input_image))
        self.config.labels = labels
        self.log(f"Auto-labels ({len(labels)}): {labels}")

    def _setup_sam3_env(self) -> None:
        """Ensure the SAM3 Python 3.12 venv exists on the remote VM."""
        sentinel = f"{self.config.repo_path}/{_SAM3_SETUP_SENTINEL}"
        rc, stdout, _ = self._vm.ssh_exec(
            f"test -f {shlex.quote(sentinel)} && echo EXISTS",
            stream_logs=False,
            check=False,
        )
        if "EXISTS" in stdout:
            self.log("SAM3 environment already set up on VM")
            return

        self.log("Setting up SAM3 environment on remote VM...")
        # The setup_remote.sh handles SAM3 venv creation.
        # Re-run it to ensure SAM3 section is executed.
        self._vm.scp_upload(_SETUP_SCRIPT_PATH, "/tmp/setup_regen3d.sh")
        self._vm.ssh_exec(
            f"bash /tmp/setup_regen3d.sh {shlex.quote(self.config.repo_path)}",
            timeout=1200,
            stream_logs=True,
            check=True,
        )
        self.log("SAM3 environment setup complete")

    def _run_sam3_segmentation(
        self,
        remote_image_path: str,
        remote_output_dir: str,
    ) -> None:
        """Run SAM3 text-prompted segmentation on the remote VM.

        Uploads segmentation_sam3.py and executes it in the venv_sam3
        environment. Populates {remote_output_dir}/findings/ with masks
        and depth, replacing 3D-RE-GEN's built-in Step 1.
        """
        # Ensure SAM3 venv is ready
        self._setup_sam3_env()

        # Upload the SAM3 segmentation script
        remote_script = "/tmp/segmentation_sam3.py"
        self._vm.scp_upload(_SAM3_SCRIPT_PATH, remote_script)

        # Build labels argument
        labels_csv = ",".join(self.config.labels)

        # Run SAM3 in the venv_sam3 environment
        venv_sam3 = f"{self.config.repo_path}/venv_sam3"
        command = (
            f"export PYTHONNOUSERSITE=1 && "
            f"{shlex.quote(venv_sam3)}/bin/python {shlex.quote(remote_script)} "
            f"--image {shlex.quote(remote_image_path)} "
            f"--output {shlex.quote(remote_output_dir)} "
            f"--labels {shlex.quote(labels_csv)} "
            f"--threshold {self.config.sam3_threshold} "
            f"--model {shlex.quote(self.config.sam3_model)} "
            f"--device {shlex.quote(self.config.device)}"
        )

        self.log("Running SAM3 segmentation...")
        rc, stdout, stderr = self._vm.ssh_exec(
            command,
            timeout=600,  # 10 minutes for segmentation + depth
            stream_logs=self.verbose,
            check=False,
        )

        if rc != 0:
            # Check for failure marker
            failed_marker = f"{remote_output_dir}/findings/.sam3_failed"
            rc2, marker_content, _ = self._vm.ssh_exec(
                f"cat {shlex.quote(failed_marker)} 2>/dev/null",
                stream_logs=False,
                check=False,
            )
            if rc2 == 0 and marker_content.strip():
                self.log(
                    f"SAM3 failed: {marker_content.strip()}. "
                    f"Falling back to 3D-RE-GEN Step 1.",
                    "WARNING",
                )
                # Re-add Step 1 so the original pipeline handles segmentation
                if 1 not in self.config.steps:
                    self.config.steps = [1] + self.config.steps
                self.config.seg_backend = "grounded_sam"
                return

            raise VMExecutorError(
                f"SAM3 segmentation failed (exit {rc}): "
                f"{stderr[-1000:] if stderr else 'no stderr'}"
            )

        # Verify output
        check_cmd = (
            f"ls {shlex.quote(remote_output_dir)}/findings/fullSize/*.png "
            f"2>/dev/null | wc -l"
        )
        rc, count_str, _ = self._vm.ssh_exec(
            check_cmd, stream_logs=False, check=False
        )
        mask_count = int(count_str.strip()) if count_str.strip().isdigit() else 0
        self.log(f"SAM3 produced {mask_count} object masks")

        if mask_count == 0:
            self.log(
                "SAM3 produced 0 masks — falling back to 3D-RE-GEN Step 1",
                "WARNING",
            )
            if 1 not in self.config.steps:
                self.config.steps = [1] + self.config.steps
            self.config.seg_backend = "grounded_sam"

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
