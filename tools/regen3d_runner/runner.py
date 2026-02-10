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
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

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
    vm_zone: str = "us-east1-c"
    repo_path: str = "/home/nijelhunt1/3D-RE-GEN"

    # Pipeline steps (1-9; typically 1-7 to skip render+eval)
    steps: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])

    # Execution
    setup_timeout_s: int = 3600  # First-time setup can take >20m (PyTorch3D build)
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
    huggingface_token: str = ""

    # Segmentation labels (populated dynamically by Gemini auto-labeling)
    labels: List[str] = field(default_factory=list)

    # Segmentation backend: "sam3" (new) or "grounded_sam" (original Step 1)
    seg_backend: str = "grounded_sam"

    # SAM3-specific settings
    sam3_threshold: float = 0.4
    sam3_model: str = "facebook/sam3"

    # Require background mesh (full scene mode)
    require_background: bool = True

    # Optionally repair missing VGGT background cloud before Step 7
    repair_missing_emptyroom_pointcloud: bool = False

    # Auto-start VM if stopped
    auto_start_vm: bool = False

    @classmethod
    def from_env(cls) -> "Regen3DConfig":
        """Load configuration from environment variables."""
        steps_str = os.getenv("REGEN3D_STEPS", "1,2,3,4,5,6,7")
        steps = [int(s.strip()) for s in steps_str.split(",") if s.strip()]

        labels_str = os.getenv("REGEN3D_LABELS", "")
        labels = [l.strip() for l in labels_str.split(",") if l.strip()]

        seg_backend = os.getenv("REGEN3D_SEG_BACKEND", "grounded_sam").lower()

        return cls(
            vm_host=os.getenv("REGEN3D_VM_HOST", "isaac-sim-ubuntu"),
            vm_zone=os.getenv("REGEN3D_VM_ZONE", "us-east1-c"),
            repo_path=os.getenv("REGEN3D_REPO_PATH", "/home/nijelhunt1/3D-RE-GEN"),
            steps=steps,
            setup_timeout_s=int(os.getenv("REGEN3D_SETUP_TIMEOUT_S", "3600")),
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
            huggingface_token=(
                os.getenv("HF_TOKEN", "")
                or os.getenv("HF_HUB_TOKEN", "")
                or os.getenv("HUGGINGFACE_HUB_TOKEN", "")
            ),
            labels=labels,
            seg_backend=seg_backend,
            sam3_threshold=float(os.getenv("REGEN3D_SAM3_THRESHOLD", "0.4")),
            sam3_model=os.getenv("REGEN3D_SAM3_MODEL", "facebook/sam3"),
            require_background=os.getenv("REGEN3D_REQUIRE_BACKGROUND", "true").lower()
            in ("true", "1", "yes"),
            repair_missing_emptyroom_pointcloud=os.getenv(
                "REGEN3D_REPAIR_EMPTYROOM_PC", "false"
            ).lower()
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

            rc = 0
            stderr = ""
            if 7 in effective_steps:
                pre_steps = [s for s in effective_steps if s != 7]
                if pre_steps:
                    self.log(
                        f"Starting 3D-RE-GEN pipeline (steps {pre_steps}) "
                        f"for scene: {scene_id}"
                    )
                    rc, _, stderr = self._execute_pipeline(
                        scene_id,
                        remote_output_dir,
                        steps_override=pre_steps,
                    )
                    if rc != 0:
                        return ReconstructionResult(
                            success=False,
                            scene_id=scene_id,
                            objects_count=0,
                            duration_seconds=time.monotonic() - start_time,
                            output_dir=output_dir,
                            error=f"3D-RE-GEN pipeline failed before Step 7 (exit {rc})",
                            remote_log=stderr[-2000:] if stderr else "",
                        )

                if self.config.repair_missing_emptyroom_pointcloud:
                    self._ensure_step7_background_pointcloud(remote_output_dir)

                self.log(
                    f"Starting 3D-RE-GEN pipeline (steps [7]) "
                    f"for scene: {scene_id}"
                )
                rc, _, stderr = self._execute_pipeline(
                    scene_id,
                    remote_output_dir,
                    steps_override=[7],
                )
            else:
                self.log(
                    f"Starting 3D-RE-GEN pipeline (steps {effective_steps}) "
                    f"for scene: {scene_id}"
                )
                rc, _, stderr = self._execute_pipeline(
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

            # Step 5.5: Create GLB symlinks on VM before downloading
            self._create_glb_symlinks(remote_output_dir)

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
            timeout=self.config.setup_timeout_s,
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

        # Upload API keys for remote steps if configured.
        self._upload_remote_env_keys()

    def _upload_remote_env_keys(self) -> None:
        """Upload optional API keys to remote .env_keys with restrictive perms."""
        exports: List[str] = []
        if self.config.gemini_api_key:
            exports.append(
                f"export GEMINI_API_KEY={shlex.quote(self.config.gemini_api_key)}"
            )
        if self.config.huggingface_token:
            hf_quoted = shlex.quote(self.config.huggingface_token)
            exports.extend(
                [
                    f"export HF_TOKEN={hf_quoted}",
                    f"export HF_HUB_TOKEN={hf_quoted}",
                    f"export HUGGINGFACE_HUB_TOKEN={hf_quoted}",
                ]
            )

        if not exports:
            return

        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False
        ) as env_file:
            env_file.write("\n".join(exports) + "\n")
            env_path = Path(env_file.name)

        try:
            remote_env_path = f"{self.config.repo_path}/.env_keys"
            self._vm.scp_upload(env_path, remote_env_path)
            self._vm.ssh_exec(
                f"chmod 600 {shlex.quote(remote_env_path)}",
                stream_logs=False,
                check=True,
            )
        finally:
            env_path.unlink(missing_ok=True)

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
            f"mkdir -p {repo}/tmp_{shlex.quote(scene_id)} && "
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
        self._upload_remote_env_keys()
        # The setup_remote.sh handles SAM3 venv creation.
        # Re-run it to ensure SAM3 section is executed.
        self._vm.scp_upload(_SETUP_SCRIPT_PATH, "/tmp/setup_regen3d.sh")
        repo = shlex.quote(self.config.repo_path)
        self._vm.ssh_exec(
            f"cd {repo} && [ -f .env_keys ] && source .env_keys ; "
            f"bash /tmp/setup_regen3d.sh {repo}",
            timeout=self.config.setup_timeout_s,
            stream_logs=True,
            check=True,
        )
        self.log("SAM3 environment setup complete")

    def _run_sam3_segmentation(
        self,
        remote_image_path: str,
        remote_output_dir: str,
        *,
        strict_failure: bool = False,
    ) -> Dict[str, Any]:
        """Run SAM3 text-prompted segmentation on the remote VM.

        Uploads segmentation_sam3.py and executes it in the venv_sam3
        environment. Populates {remote_output_dir}/findings/ with masks
        and depth, replacing 3D-RE-GEN's built-in Step 1.
        """
        result: Dict[str, Any] = {
            "mask_count": 0,
            "fallback_used": False,
            "error": None,
        }

        # Ensure SAM3 venv is ready
        self._setup_sam3_env()

        # Upload the SAM3 segmentation script
        remote_script = "/tmp/segmentation_sam3.py"
        self._vm.scp_upload(_SAM3_SCRIPT_PATH, remote_script)

        # Build labels argument
        labels_csv = ",".join(self.config.labels)

        # Run SAM3 in the venv_sam3 environment
        venv_sam3 = f"{self.config.repo_path}/venv_sam3"
        repo = shlex.quote(self.config.repo_path)
        command = (
            f"cd {repo} && [ -f .env_keys ] && source .env_keys ; "
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
            marker_error = marker_content.strip() if rc2 == 0 else ""
            error_msg = marker_error or (
                stderr[-1000:] if stderr else f"SAM3 segmentation failed (exit {rc})"
            )
            result["error"] = error_msg

            if strict_failure:
                raise VMExecutorError(f"SAM3 segmentation failed: {error_msg}")

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
                result["fallback_used"] = True
                return result

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
        result["mask_count"] = mask_count
        self.log(f"SAM3 produced {mask_count} object masks")

        if mask_count == 0:
            if strict_failure:
                result["error"] = "SAM3 produced 0 masks"
                raise VMExecutorError("SAM3 produced 0 masks")
            self.log(
                "SAM3 produced 0 masks — falling back to 3D-RE-GEN Step 1",
                "WARNING",
            )
            if 1 not in self.config.steps:
                self.config.steps = [1] + self.config.steps
            self.config.seg_backend = "grounded_sam"
            result["fallback_used"] = True
            result["error"] = "SAM3 produced 0 masks"

        return result

    def _ensure_step7_background_pointcloud(self, remote_output_dir: str) -> None:
        """Create points_emptyRoom.ply from available VGGT sparse clouds if missing."""
        sparse_dir = f"{remote_output_dir}/vggt/sparse"
        empty_room = f"{sparse_dir}/points_emptyRoom.ply"
        rc, stdout, _ = self._vm.ssh_exec(
            f"test -f {shlex.quote(empty_room)} && echo EXISTS",
            stream_logs=False,
            check=False,
        )
        if "EXISTS" in stdout:
            return

        for candidate in ("points_merged.ply", "points.ply"):
            candidate_path = f"{sparse_dir}/{candidate}"
            rc, stdout, _ = self._vm.ssh_exec(
                f"test -f {shlex.quote(candidate_path)} && echo EXISTS",
                stream_logs=False,
                check=False,
            )
            if "EXISTS" not in stdout:
                continue

            rc, _, stderr = self._vm.ssh_exec(
                f"cp {shlex.quote(candidate_path)} {shlex.quote(empty_room)}",
                stream_logs=False,
                check=False,
            )
            if rc == 0:
                self.log(
                    f"Created missing points_emptyRoom.ply from {candidate}",
                    "WARNING",
                )
            else:
                self.log(
                    f"Failed to create points_emptyRoom.ply from {candidate}: "
                    f"{stderr[-200:] if stderr else 'unknown error'}",
                    "WARNING",
                )
            return

        self.log(
            "points_emptyRoom.ply missing and no compatible fallback point cloud found",
            "WARNING",
        )

    def _create_glb_symlinks(self, remote_output_dir: str) -> None:
        """Create {name}.glb -> {name}_shape.glb symlinks on the remote VM.

        Hunyuan3D-2.1 outputs files as {name}_shape.glb but downstream code
        expects {name}.glb. This creates symlinks so both names resolve.
        """
        cmd = (
            f"for dir in {remote_output_dir}/3D/*/; do "
            f"name=$(basename \"$dir\"); "
            f"shape_glb=\"$dir/${{name}}_shape.glb\"; "
            f"link_glb=\"$dir/${{name}}.glb\"; "
            f"if [ -f \"$shape_glb\" ] && [ ! -e \"$link_glb\" ]; then "
            f"ln -s \"${{name}}_shape.glb\" \"$link_glb\" && "
            f"echo \"Symlinked: $link_glb -> ${{name}}_shape.glb\"; "
            f"fi; "
            f"done"
        )
        rc, stdout, stderr = self._vm.ssh_exec(
            cmd, stream_logs=False, check=False
        )
        if stdout.strip():
            for line in stdout.strip().splitlines():
                self.log(line)

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

    def setup_sam3_only(self) -> None:
        """Set up the SAM3 environment on the remote VM without running a reconstruction.

        Useful for pre-warming the VM so the first SAM3 reconstruction doesn't
        spend 10-15 minutes on setup.
        """
        self.log("Pre-warming SAM3 environment on remote VM...")
        self._ensure_vm_ready()
        self._setup_sam3_env()
        self.log("SAM3 environment is ready. Set REGEN3D_SEG_BACKEND=sam3 to use it.")

    def _count_remote_masks(self, remote_output_dir: str) -> int:
        """Count masks in findings/fullSize on the remote VM."""
        cmd = (
            f"ls {shlex.quote(remote_output_dir)}/findings/fullSize/*.png "
            f"2>/dev/null | wc -l"
        )
        _, count_str, _ = self._vm.ssh_exec(cmd, stream_logs=False, check=False)
        return int(count_str.strip()) if count_str.strip().isdigit() else 0

    def _download_findings_if_present(
        self,
        remote_output_dir: str,
        local_output_dir: Path,
    ) -> bool:
        """Download findings/ artifacts for A/B compare if they exist.

        For `--compare`, we only need `findings/fullSize` (masks) and any
        top-level marker files (e.g. `.sam3_failed`). Downloading the full
        findings tree is slow and doesn't change the visual decision.
        """
        rc, stdout, _ = self._vm.ssh_exec(
            f"test -d {shlex.quote(remote_output_dir)}/findings && echo EXISTS",
            stream_logs=False,
            check=False,
        )
        if "EXISTS" not in stdout:
            return False
        local_output_dir.mkdir(parents=True, exist_ok=True)

        any_downloaded = False

        # 1) Download masks (what we visually compare)
        remote_fullsize = f"{remote_output_dir}/findings/fullSize"
        rc, stdout, _ = self._vm.ssh_exec(
            f"test -d {shlex.quote(remote_fullsize)} && echo EXISTS",
            stream_logs=False,
            check=False,
        )
        if "EXISTS" in stdout:
            self._vm.scp_download_dir(
                remote_fullsize,
                local_output_dir / "findings" / "fullSize",
            )
            any_downloaded = True

        # 2) Download any small marker/metadata files at findings root.
        rc, stdout, _ = self._vm.ssh_exec(
            f"find {shlex.quote(remote_output_dir)}/findings "
            f"-maxdepth 1 -type f 2>/dev/null",
            stream_logs=False,
            check=False,
        )
        if rc == 0 and stdout.strip():
            for remote_file in [l.strip() for l in stdout.splitlines() if l.strip()]:
                rel = remote_file.replace(
                    f"{remote_output_dir.rstrip('/')}/findings/", "", 1
                )
                self._vm.scp_download(
                    remote_file,
                    local_output_dir / "findings" / rel,
                )
                any_downloaded = True

        return any_downloaded

    def compare_backends(
        self,
        input_image: Path,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Run both segmentation backends on the same image and compare outputs."""
        input_image = Path(input_image)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_image.is_file():
            raise FileNotFoundError(f"Input image not found: {input_image}")

        self.log("Starting A/B segmentation comparison...")
        self._ensure_vm_ready()
        self._setup_remote()

        # Generate labels once (shared between both backends)
        self._resolve_auto_labels(input_image)
        self.log(f"Labels for comparison: {self.config.labels}")

        # Upload image
        scene_id = f"ab_compare_{int(time.time())}"
        remote_image_path = f"/tmp/{scene_id}_input{input_image.suffix}"
        self._vm.scp_upload(input_image, remote_image_path)

        local_out_a = output_dir / "grounded_sam"
        local_out_b = output_dir / "sam3"
        remote_out_a = f"{self.config.repo_path}/output_{scene_id}_grounded_sam"
        remote_out_b = f"{self.config.repo_path}/output_{scene_id}_sam3"

        results: Dict[str, Any] = {
            "scene_id": scene_id,
            "input_image": str(input_image),
            "labels": self.config.labels,
            "grounded_sam": {
                "status": "failed",
                "error": "not_run",
                "fallback_used": False,
                "mask_count": 0,
                "duration_s": 0.0,
                "exit_code": 1,
                "output_dir": str(local_out_a),
            },
            "sam3": {
                "status": "failed",
                "error": "not_run",
                "fallback_used": False,
                "mask_count": 0,
                "duration_s": 0.0,
                "exit_code": 1,
                "output_dir": str(local_out_b),
            },
        }

        try:
            # --- Backend A: grounded_sam (3D-RE-GEN Step 1) ---
            self.log("=" * 60)
            self.log("Running backend A: grounded_sam (GroundingDINO + SAM1)")
            self.log("=" * 60)
            t0 = time.monotonic()
            try:
                self._generate_and_upload_config(
                    remote_image_path, remote_out_a, scene_id
                )
                rc_a, _, stderr_a = self._execute_pipeline(
                    scene_id, remote_out_a, steps_override=[1]
                )
                dur_a = time.monotonic() - t0
                count_a = self._count_remote_masks(remote_out_a)
                self._download_findings_if_present(remote_out_a, local_out_a)
                results["grounded_sam"].update(
                    {
                        "status": "success" if rc_a == 0 else "failed",
                        "error": None if rc_a == 0 else (
                            stderr_a[-800:] if stderr_a else f"exit {rc_a}"
                        ),
                        "mask_count": count_a,
                        "duration_s": round(dur_a, 1),
                        "exit_code": rc_a,
                    }
                )
            except Exception as exc:
                dur_a = time.monotonic() - t0
                results["grounded_sam"].update(
                    {
                        "status": "failed",
                        "error": str(exc),
                        "duration_s": round(dur_a, 1),
                        "exit_code": 1,
                    }
                )
            self.log(
                f"grounded_sam: {results['grounded_sam']['mask_count']} masks "
                f"in {results['grounded_sam']['duration_s']:.1f}s "
                f"(status={results['grounded_sam']['status']})"
            )

            # --- Backend B: SAM3 ---
            self.log("=" * 60)
            self.log("Running backend B: sam3 (SAM3 text-prompted)")
            self.log("=" * 60)
            t0 = time.monotonic()
            try:
                self._vm.ssh_exec(
                    f"mkdir -p {shlex.quote(remote_out_b)}",
                    stream_logs=False,
                    check=True,
                )
                sam3_result = self._run_sam3_segmentation(
                    remote_image_path,
                    remote_out_b,
                    strict_failure=True,
                )
                dur_b = time.monotonic() - t0
                count_b = self._count_remote_masks(remote_out_b)
                self._download_findings_if_present(remote_out_b, local_out_b)
                results["sam3"].update(
                    {
                        "status": "success",
                        "error": sam3_result["error"],
                        "fallback_used": sam3_result["fallback_used"],
                        "mask_count": count_b,
                        "duration_s": round(dur_b, 1),
                        "exit_code": 0,
                    }
                )
            except Exception as exc:
                dur_b = time.monotonic() - t0
                count_b = self._count_remote_masks(remote_out_b)
                self._download_findings_if_present(remote_out_b, local_out_b)
                results["sam3"].update(
                    {
                        "status": "failed",
                        "error": str(exc),
                        "mask_count": count_b,
                        "duration_s": round(dur_b, 1),
                        "exit_code": 1,
                    }
                )
            self.log(
                f"sam3: {results['sam3']['mask_count']} masks "
                f"in {results['sam3']['duration_s']:.1f}s "
                f"(status={results['sam3']['status']})"
            )
        finally:
            # Cleanup remote
            self._vm.ssh_exec(
                f"rm -rf {shlex.quote(remote_out_a)} {shlex.quote(remote_out_b)} "
                f"{shlex.quote(remote_image_path)}",
                stream_logs=False, check=False,
            )

        results["success"] = (
            results["grounded_sam"]["status"] == "success"
            and results["sam3"]["status"] == "success"
        )

        # Write comparison summary
        summary_path = output_dir / "comparison.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        self.log("=" * 60)
        self.log("A/B COMPARISON RESULTS")
        self.log(
            f"  grounded_sam: {results['grounded_sam']['mask_count']} masks, "
            f"{results['grounded_sam']['duration_s']:.1f}s, "
            f"status={results['grounded_sam']['status']}"
        )
        self.log(
            f"  sam3:         {results['sam3']['mask_count']} masks, "
            f"{results['sam3']['duration_s']:.1f}s, "
            f"status={results['sam3']['status']}"
        )
        self.log(f"  Output:       {output_dir}")
        self.log(
            f"  Compare masks visually in {local_out_a}/findings/fullSize/ vs "
            f"{local_out_b}/findings/fullSize/"
        )
        self.log("=" * 60)
        return results


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _load_runner_env_defaults() -> None:
    """Load repository .env defaults for standalone runner CLI."""
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=repo_root / ".env", override=False)
    load_dotenv(
        dotenv_path=repo_root / "configs" / "regen3d_reconstruct.env",
        override=False,
    )


def _cli():
    """Minimal CLI for maintenance tasks."""
    import argparse

    parser = argparse.ArgumentParser(
        description="3D-RE-GEN runner — maintenance utilities",
    )
    parser.add_argument(
        "--setup-sam3",
        action="store_true",
        help="Set up the SAM3 segmentation environment on the remote VM "
        "(Python 3.12 venv, PyTorch 2.7, SAM3 model download). "
        "Takes ~10-15 min on first run, instant thereafter.",
    )
    parser.add_argument(
        "--compare",
        metavar="IMAGE",
        help="Run both segmentation backends (grounded_sam and sam3) on the "
        "given image and produce a side-by-side comparison of masks.",
    )
    parser.add_argument(
        "--compare-output",
        metavar="DIR",
        default=None,
        help="Output directory for --compare results "
        "(default: ~/Downloads/seg_comparison/)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    _load_runner_env_defaults()

    if args.setup_sam3:
        runner = Regen3DRunner()
        runner.setup_sam3_only()
    elif args.compare:
        from pathlib import Path as _Path
        image_path = _Path(args.compare)
        out_dir = _Path(
            args.compare_output
            or _Path.home() / "Downloads" / "seg_comparison"
        )
        runner = Regen3DRunner()
        results = runner.compare_backends(image_path, out_dir)
        print(json.dumps(results, indent=2))
        if not results.get("success", False):
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
