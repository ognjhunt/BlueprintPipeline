#!/usr/bin/env python3
"""
Local Pipeline Runner for BlueprintPipeline.

Runs the full pipeline locally without requiring GCS or Cloud Run.
Useful for development, testing, and debugging.

Usage:
    # Generate mock scene and run pipeline
    python fixtures/generate_mock_regen3d.py --scene-id test_kitchen --output-dir ./test_scenes
    python tools/run_local_pipeline.py --scene-dir ./test_scenes/scenes/test_kitchen

    # Run with specific steps
    python tools/run_local_pipeline.py --scene-dir ./scene --steps regen3d,simready,usd

    # Run with validation
    python tools/run_local_pipeline.py --scene-dir ./scene --validate

    # Genie Sim submit now auto-triggers import by default when import not in steps
    python tools/run_local_pipeline.py --scene-dir ./scene \
        --steps genie-sim-export,genie-sim-submit
    # To explicitly disable auto-import:
    python tools/run_local_pipeline.py --scene-dir ./scene \
        --steps genie-sim-export,genie-sim-submit --no-auto-trigger-import
    # Run a local poller to trigger import once job.json reports completed
    python tools/run_local_pipeline.py --scene-dir ./scene --import-poller

Pipeline Steps:
    0. regen3d-reconstruct - (Optional) Run 3D-RE-GEN on GPU VM (image -> 3D)
    1. regen3d   - Adapt 3D-RE-GEN outputs to BlueprintPipeline format
    2. scale     - (Optional) Scale calibration
    3. interactive - Particulate articulation (requires PARTICULATE_ENDPOINT)
    4. simready  - Prepare physics-ready assets
    5. usd       - Assemble scene.usda
    6. inventory-enrichment - (Optional) Enrich inventory metadata
    7. replicator - Generate Replicator bundle
    8. variation-gen - Generate variation assets for Genie Sim export
    9. isaac-lab - Generate Isaac Lab task package
    10. genie-sim-export - Export scene bundle for Genie Sim
    11. genie-sim-submit - Submit/run Genie Sim generation (API or local)
    12. genie-sim-import - Import Genie Sim episodes into local bundle
    13. dataset-delivery - (Optional) Deliver datasets to lab buckets
    14. validate  - QA validation

    Auto-trigger import: By default, a successful genie-sim-submit step
    (status=completed) will automatically run genie-sim-import if it is
    not already scheduled. Use --no-auto-trigger-import to disable this.

Experimental Steps (disabled by default; enable with --enable-dwm,
--enable-dream2flow, or --enable-experimental):
    - dwm       - Generate DWM conditioning data (egocentric videos + hand meshes)
    - dwm-inference - Run DWM model to generate interaction videos for each bundle
    - dream2flow - Generate Dream2Flow conditioning bundles
    - dream2flow-inference - Run Dream2Flow model inference

Note: Experimental steps are hidden from default help and require explicit enablement.

Environment overrides:
    - DEFAULT_CAMERA_IDS: Comma-separated camera IDs for replicator capture config.
    - DEFAULT_CAMERA_RESOLUTION: Resolution as "WIDTHxHEIGHT" or "WIDTH,HEIGHT".
    - DEFAULT_STREAM_IDS: Comma-separated stream IDs for replicator capture config.
    - ALLOW_PARTIAL_FIREBASE_UPLOADS: Allow Firebase uploads to proceed for
      successful robots even if others fail.
    - FIREBASE_REQUIRE_ATOMIC_UPLOAD: When true, rollback all successful uploads if any robot fails.
    - GENIESIM_SUBMIT_TIMEOUT_S: Total wall-clock timeout for the genie-sim-submit step.
    - GENIESIM_IMPORT_POLL_INTERVAL: Poll interval in seconds for genie-sim-import status checks.
    - GENIESIM_IMPORT_POLL_TIMEOUT: Timeout in seconds for genie-sim-import status polling.
    - GCS_MOUNT_ROOT: Base path used for local GCS-style mount mapping.
    - PIPELINE_FAIL_FAST: Stop the pipeline on the first failure when true.
    - PIPELINE_STEP_TIMEOUTS_JSON: JSON mapping of pipeline step names to timeout seconds.
    - ADAPTIVE_TIMEOUT_BUNDLE_TIER: Bundle tier for adaptive default timeouts.
    - ADAPTIVE_TIMEOUT_SCENE_COMPLEXITY: Scene complexity for adaptive default timeouts.
    - REQUIRE_BALANCED_ROBOT_EPISODES: Fail validation when cross-robot episode counts mismatch.
    - RELEASE_PATH_RUN: Enforce release-path contracts (requires production mode).
    - REGEN3D_ALLOW_MATERIALLESS: Dev/test override to bypass Stage 1 material checks.

References:
- 3D-RE-GEN (arXiv:2512.17459): "image → sim-ready 3D reconstruction"
  Paper: https://arxiv.org/abs/2512.17459
  Project: https://3dregen.jdihlmann.com/

- DWM (arXiv:2512.17907): Dexterous World Models for egocentric interaction
  Paper: https://arxiv.org/abs/2512.17907
  Project: https://snuvclab.github.io/dwm/
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import logging
import re
import os
import shutil
import struct
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

DEFAULT_DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(
    dotenv_path=DEFAULT_DOTENV_PATH,
    override=False,
)  # load repo .env (GEMINI_API_KEY, etc.) before any os.environ reads

import numpy as np
import yaml

from tools.checkpoint import get_checkpoint_store
from tools.checkpoint.hash_config import resolve_checkpoint_hash_setting
from tools.cost_tracking.estimate import (
    estimate_gpu_costs,
    format_estimate_summary,
    load_estimate_config,
)
from tools.config import (
    load_adaptive_timeout_config,
    load_pipeline_config,
    load_pipeline_step_timeouts,
    load_quality_config as load_quality_gate_config,
)
from tools.config.env import parse_bool_env, parse_float_env, parse_int_env
from tools.config.production_mode import resolve_pipeline_environment, resolve_production_mode
from tools.config.seed_manager import configure_pipeline_seed
from tools.error_handling.errors import ErrorCategory, ErrorContext, PipelineError
from tools.error_handling.errors import classify_exception
from tools.error_handling.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from tools.error_handling.logging import log_pipeline_error
from tools.error_handling.retry import (
    NonRetryableError,
    RetryConfig,
    RetryableError,
    retry_with_backoff,
)
from tools.error_handling.timeout import TimeoutError, timeout_thread
from tools.inventory_enrichment import enrich_inventory_file, InventoryEnrichmentError
from tools.geniesim_adapter.mock_mode import resolve_geniesim_mock_mode
from tools.lerobot_validation import validate_lerobot_dataset
from tools.metrics.job_metrics_exporter import export_job_metrics
from tools.geniesim_idempotency import (
    build_geniesim_idempotency_inputs,
    build_quality_thresholds,
)
from tools.quality.quality_config import resolve_quality_settings
from tools.quality_gates import QualityGateCheckpoint, QualityGateRegistry, build_notification_service
from tools.startup_validation import (
    validate_firebase_credentials,
    validate_gcs_credentials,
)
from tools.validation.geniesim_export import ExportConsistencyError, validate_export_consistency
from tools.tracing.correlation import ensure_request_id

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RETRY_POLICY_FALLBACK = {
    "max_retries": 5,
    "base_delay_seconds": 1.0,
    "max_delay_seconds": 60.0,
    "backoff_factor": 2.0,
}


def _load_json(path: Path, context: str) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise NonRetryableError(
            f"Failed to parse JSON for {context} at {path}: {exc}"
        ) from exc


def _safe_write_text(path: Path, payload: str, context: str) -> None:
    try:
        path.write_text(payload)
    except Exception as exc:
        raise NonRetryableError(
            f"Failed to write {context} to {path}: {exc}"
        ) from exc


class PipelineStep(str, Enum):
    """Pipeline steps in execution order."""
    REGEN3D_RECONSTRUCT = "regen3d-reconstruct"  # Run 3D-RE-GEN on remote GPU VM
    REGEN3D = "regen3d"
    SCALE = "scale"
    INTERACTIVE = "interactive"
    SIMREADY = "simready"
    USD = "usd"
    INVENTORY_ENRICHMENT = "inventory-enrichment"
    REPLICATOR = "replicator"
    VARIATION_GEN = "variation-gen"
    ISAAC_LAB = "isaac-lab"
    GENIESIM_EXPORT = "genie-sim-export"
    GENIESIM_SUBMIT = "genie-sim-submit"
    GENIESIM_IMPORT = "genie-sim-import"
    DATASET_DELIVERY = "dataset-delivery"
    DWM = "dwm"  # Dexterous World Model preparation
    DWM_INFERENCE = "dwm-inference"  # Run DWM model on prepared bundles
    DREAM2FLOW = "dream2flow"  # Dream2Flow preparation (arXiv:2512.24766)
    DREAM2FLOW_INFERENCE = "dream2flow-inference"  # Run Dream2Flow inference
    VALIDATE = "validate"


@dataclass
class StepResult:
    """Result of a pipeline step."""
    step: PipelineStep
    success: bool
    duration_seconds: float
    message: str = ""
    outputs: Dict[str, Any] = None

    def __post_init__(self):
        if self.outputs is None:
            self.outputs = {}


class LocalPipelineRunner:
    """Run BlueprintPipeline locally for testing."""

    # Default step order (free/default workflow)
    DEFAULT_STEPS = [
        PipelineStep.REGEN3D,
        PipelineStep.SCALE,
        PipelineStep.INTERACTIVE,
        PipelineStep.SIMREADY,
        PipelineStep.USD,
        PipelineStep.REPLICATOR,
        PipelineStep.GENIESIM_EXPORT,
        PipelineStep.GENIESIM_SUBMIT,
        PipelineStep.GENIESIM_IMPORT,
    ]

    DWM_STEPS = [
        PipelineStep.DWM,  # DWM conditioning data generation
        PipelineStep.DWM_INFERENCE,  # Run DWM model on bundles
    ]

    DREAM2FLOW_STEPS = [
        PipelineStep.DREAM2FLOW,  # Dream2Flow conditioning data (arXiv:2512.24766)
        PipelineStep.DREAM2FLOW_INFERENCE,  # Run Dream2Flow inference
    ]

    def __init__(
        self,
        scene_dir: Path,
        verbose: bool = True,
        json_logging: Optional[bool] = None,
        skip_interactive: bool = True,
        environment_type: str = "kitchen",
        enable_dwm: bool = False,
        enable_dream2flow: bool = False,
        enable_inventory_enrichment: Optional[bool] = None,
        enable_dataset_delivery: bool = False,
        disable_articulated_assets: bool = False,
        fail_fast: Optional[bool] = None,
        step_circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        """Initialize the local pipeline runner.

        Args:
            scene_dir: Path to scene directory (contains regen3d/, etc.)
            verbose: Print detailed progress
            skip_interactive: Skip interactive-job (requires external service)
            environment_type: Environment type for policy selection
            enable_dwm: Include optional DWM steps in default pipeline
            enable_dream2flow: Include optional Dream2Flow steps in default pipeline
            enable_inventory_enrichment: Enable optional inventory enrichment step
            enable_dataset_delivery: Enable optional dataset delivery step
            step_circuit_breaker: Optional shared step-level circuit breaker to reuse across runs
        """
        self.scene_dir = Path(scene_dir).resolve()
        self.verbose = verbose
        if json_logging is None:
            self.json_logging = parse_bool_env(
                os.getenv("BP_JSON_LOGS"),
                default=True,
            )
        else:
            self.json_logging = json_logging
        self.skip_interactive = skip_interactive
        self.environment_type = environment_type
        self.enable_dwm = enable_dwm
        self.enable_dream2flow = enable_dream2flow
        if enable_inventory_enrichment is None:
            self.enable_inventory_enrichment = os.getenv(
                "ENABLE_INVENTORY_ENRICHMENT",
                "0",
            ).lower() in {"1", "true", "yes", "y"}
        else:
            self.enable_inventory_enrichment = enable_inventory_enrichment
        self.enable_dataset_delivery = enable_dataset_delivery
        self.disable_articulated_assets = disable_articulated_assets
        if fail_fast is None:
            # Default to True in production mode for faster failure detection
            production_default = resolve_production_mode()
            self.fail_fast = parse_bool_env(
                os.getenv("PIPELINE_FAIL_FAST"),
                default=production_default,
            )
        else:
            self.fail_fast = fail_fast
        self.environment = resolve_pipeline_environment()
        self.debug = os.getenv("BP_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
        self.enable_checkpoint_hashes = resolve_checkpoint_hash_setting()

        # GCS sync (set up later via configure_gcs_sync)
        self._gcs_sync = None
        self._gcs_download_inputs = False
        self._gcs_upload_outputs = False
        self._gcs_input_object: Optional[str] = None
        self._gcs_input_generation: Optional[str] = None

        # Derive scene ID from directory name
        self.scene_id = self.scene_dir.name
        self.run_id = os.getenv("BP_RUN_ID") or str(uuid4())
        os.environ.setdefault("RUN_ID", self.run_id)
        self._current_step: Optional[PipelineStep] = None
        self._logger = self._configure_logger()

        # Setup paths
        self.regen3d_dir = self.scene_dir / "regen3d"
        self.assets_dir = self.scene_dir / "assets"
        self.layout_dir = self.scene_dir / "layout"
        self.seg_dir = self.scene_dir / "seg"
        self.usd_dir = self.scene_dir / "usd"
        self.replicator_dir = self.scene_dir / "replicator"
        self.isaac_lab_dir = self.scene_dir / "isaac_lab"
        self.geniesim_dir = self.scene_dir / "geniesim"
        self.episodes_dir = self.scene_dir / "episodes"
        self.dwm_dir = self.scene_dir / "dwm"  # DWM conditioning data
        self.dream2flow_dir = self.scene_dir / "dream2flow"  # Dream2Flow conditioning data

        self.results: List[StepResult] = []
        self._geniesim_preflight_report: Optional[Dict[str, Any]] = None
        self._pending_articulation_preflight = False
        self._path_redaction_regex = re.compile(r"(?:[A-Za-z]:\\\\|/)[^\\s]+")
        self._quality_gates = QualityGateRegistry(verbose=self.verbose)
        self._notification_service = build_notification_service(
            self._quality_gates.config,
            verbose=self.verbose,
        )
        self._quality_gate_report_path = self._resolve_quality_gate_report_path()
        self.retry_config = self._resolve_retry_config()
        self._step_timeouts = self._resolve_step_timeouts()
        self._geniesim_local_run_results: Dict[str, Any] = {}
        self._geniesim_output_dirs: Dict[str, Path] = {}

        # Step-level circuit breaker to prevent cascading failures (P1)
        # Opens after consecutive failures to avoid wasting time on doomed pipelines
        if step_circuit_breaker is not None:
            self._step_circuit_breaker = step_circuit_breaker
            self.log(
                f"Using provided step-level circuit breaker '{self._step_circuit_breaker.name}'.",
                "INFO",
            )
        else:
            self._step_circuit_breaker = None
        self._step_circuit_breaker_persistence_path: Optional[Path] = None

    def configure_gcs_sync(
        self,
        bucket_name: str,
        download_inputs: bool = False,
        upload_outputs: bool = False,
        input_object: Optional[str] = None,
        input_generation: Optional[str] = None,
        upload_concurrency: Optional[int] = None,
    ) -> None:
        """Configure GCS sync for downloading inputs and uploading outputs.

        Args:
            bucket_name: GCS bucket name (e.g., ``blueprint-8c1ca.appspot.com``).
            download_inputs: Download input image from GCS before running.
            upload_outputs: Upload step outputs to GCS after each step.
            input_object: Preferred scene image object path (relative or gs:// URI).
            input_generation: Source GCS object generation for idempotence tracking.
            upload_concurrency: Number of parallel uploads for each synced directory.
        """
        try:
            from tools.gcs_sync import GCSSync
        except ImportError:
            from gcs_sync import GCSSync  # type: ignore[no-redef]

        if upload_concurrency is None:
            try:
                upload_concurrency = parse_int_env(
                    os.getenv("GCS_UPLOAD_CONCURRENCY"),
                    default=4,
                    min_value=1,
                    name="GCS_UPLOAD_CONCURRENCY",
                )
            except ValueError as exc:
                raise NonRetryableError(str(exc)) from exc

        self._gcs_sync = GCSSync(
            bucket_name=bucket_name,
            scene_id=self.scene_id,
            local_scene_dir=self.scene_dir,
            concurrency=upload_concurrency,
            input_object=input_object,
            input_generation=input_generation,
        )
        self._gcs_download_inputs = download_inputs
        self._gcs_upload_outputs = upload_outputs
        self._gcs_input_object = input_object
        self._gcs_input_generation = input_generation
        self.log(
            "[GCS] Sync configured: "
            f"bucket={bucket_name}, download={download_inputs}, upload={upload_outputs}, "
            f"input_object={input_object}, input_generation={input_generation}, "
            f"upload_concurrency={upload_concurrency}"
        )

    def _resolve_step_dependencies(
        self,
        steps: List[PipelineStep],
    ) -> Dict[PipelineStep, List[PipelineStep]]:
        """Return step dependency map for the current run."""
        dependencies: Dict[PipelineStep, List[PipelineStep]] = {
            PipelineStep.REGEN3D: (
                [PipelineStep.REGEN3D_RECONSTRUCT]
                if PipelineStep.REGEN3D_RECONSTRUCT in steps
                else []
            ),
            PipelineStep.SCALE: [PipelineStep.REGEN3D],
            PipelineStep.INTERACTIVE: [PipelineStep.REGEN3D],
            PipelineStep.SIMREADY: [PipelineStep.REGEN3D],
            PipelineStep.USD: [PipelineStep.SIMREADY],
            PipelineStep.INVENTORY_ENRICHMENT: [PipelineStep.REGEN3D],
            PipelineStep.REPLICATOR: [PipelineStep.USD],
            PipelineStep.VARIATION_GEN: [PipelineStep.REPLICATOR],
            PipelineStep.ISAAC_LAB: [PipelineStep.USD],
            PipelineStep.DWM: [PipelineStep.USD],
            PipelineStep.DWM_INFERENCE: [PipelineStep.DWM],
            PipelineStep.DREAM2FLOW: [PipelineStep.USD],
            PipelineStep.DREAM2FLOW_INFERENCE: [PipelineStep.DREAM2FLOW],
            PipelineStep.GENIESIM_EXPORT: [PipelineStep.SIMREADY],
            PipelineStep.GENIESIM_SUBMIT: [PipelineStep.GENIESIM_EXPORT],
            PipelineStep.GENIESIM_IMPORT: [PipelineStep.GENIESIM_SUBMIT],
            PipelineStep.DATASET_DELIVERY: [PipelineStep.GENIESIM_IMPORT],
        }

        if self._variation_assets_expected(steps):
            dependencies.setdefault(PipelineStep.GENIESIM_EXPORT, []).append(
                PipelineStep.VARIATION_GEN
            )

        return dependencies

    def _variation_assets_expected(self, steps: List[PipelineStep]) -> bool:
        """Return True if variation assets are expected for Genie Sim export."""
        if PipelineStep.VARIATION_GEN in steps:
            return True
        manifest_path = self.replicator_dir / "variation_assets" / "manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = _load_json(manifest_path, "variation assets manifest")
        except NonRetryableError as exc:
            self.log(f"Unable to parse variation assets manifest: {exc}", "WARNING")
            return True
        assets = manifest.get("assets", []) if isinstance(manifest, dict) else []
        return bool(assets)

    def _find_step_result(self, step: PipelineStep) -> Optional[StepResult]:
        for result in reversed(self.results):
            if result.step == step:
                return result
        return None

    def _output_is_nonempty(self, path: Path) -> bool:
        if path.is_dir():
            return any(path.iterdir())
        if not path.exists():
            return False
        return path.stat().st_size > 0

    def _missing_expected_outputs(self, step: PipelineStep) -> List[Path]:
        missing: List[Path] = []
        for path in self._expected_output_paths(step):
            if not self._output_is_nonempty(path):
                missing.append(path)
        return missing

    def _check_step_prerequisites(
        self,
        step: PipelineStep,
        dependencies: Dict[PipelineStep, List[PipelineStep]],
        requested_steps: List[PipelineStep],
    ) -> Optional[StepResult]:
        required_steps = dependencies.get(step, [])
        if not required_steps:
            return None

        issues: List[str] = []
        for prereq in required_steps:
            prereq_result = self._find_step_result(prereq)
            if prereq_result is None and prereq in requested_steps:
                issues.append(f"{prereq.value} did not run yet")
            elif prereq_result is not None and not prereq_result.success:
                issues.append(f"{prereq.value} failed")

            missing_outputs = self._missing_expected_outputs(prereq)
            if missing_outputs:
                missing_list = ", ".join(str(path) for path in missing_outputs)
                issues.append(f"{prereq.value} outputs missing: {missing_list}")

        if not issues:
            return None

        message = "Skipped due to missing prerequisites: " + "; ".join(issues)
        return StepResult(
            step=step,
            success=False,
            duration_seconds=0,
            message=message,
        )

    def log(self, msg: str, level: str = "INFO") -> None:
        """Log a message."""
        if not self.verbose:
            return
        level_name = level.upper()
        log_level = logging._nameToLevel.get(level_name, logging.INFO)
        step_value = None
        if self._current_step is not None:
            step_value = self._current_step.value
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level_name,
            "scene_id": self.scene_id,
            "step": step_value,
            "run_id": self.run_id,
            "message": msg,
        }
        if self.json_logging:
            print(json.dumps(payload))
        else:
            print(f"[run_id={self.run_id}] {msg}")

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"local-pipeline.{self.scene_id}.{self.run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()
        handler = logging.StreamHandler(stream=sys.stdout)
        if self.json_logging:
            formatter = logging.Formatter("%(message)s")
        else:
            formatter = logging.Formatter("[LOCAL-PIPELINE] [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _sanitize_error_message(self, message: str) -> str:
        if not message:
            return message
        return self._path_redaction_regex.sub("<redacted-path>", message)

    def _summarize_exception(self, exc: Exception) -> str:
        sanitized_message = self._sanitize_error_message(str(exc))
        if sanitized_message:
            return f"{type(exc).__name__}: {sanitized_message}"
        return type(exc).__name__

    def _load_retry_policy_defaults(self) -> Dict[str, float]:
        policy_path = REPO_ROOT / "policy_configs" / "retry_policy.yaml"
        if not policy_path.exists():
            message = f"Retry policy defaults not found at {policy_path}; using fallback defaults."
            if self._is_production_mode():
                raise NonRetryableError(message)
            self.log(message, "WARNING")
            return dict(RETRY_POLICY_FALLBACK)
        try:
            payload = yaml.safe_load(policy_path.read_text()) or {}
        except yaml.YAMLError as exc:
            message = f"Failed to parse retry policy defaults at {policy_path}: {exc}"
            if self._is_production_mode():
                raise NonRetryableError(message) from exc
            self.log(message, "WARNING")
            return dict(RETRY_POLICY_FALLBACK)

        if not isinstance(payload, dict):
            message = f"Retry policy defaults at {policy_path} must be a mapping."
            if self._is_production_mode():
                raise NonRetryableError(message)
            self.log(message, "WARNING")
            return dict(RETRY_POLICY_FALLBACK)

        def _coerce(name: str, cast, fallback_key: str) -> float:
            value = payload.get(name, RETRY_POLICY_FALLBACK[fallback_key])
            try:
                return cast(value)
            except (TypeError, ValueError) as exc:
                message = (
                    f"Invalid retry policy value for '{name}' in {policy_path}: {value}"
                )
                if self._is_production_mode():
                    raise NonRetryableError(message) from exc
                self.log(message + "; using fallback.", "WARNING")
                return RETRY_POLICY_FALLBACK[fallback_key]

        return {
            "max_retries": int(_coerce("max_retries", int, "max_retries")),
            "base_delay_seconds": float(_coerce("base_delay_seconds", float, "base_delay_seconds")),
            "max_delay_seconds": float(_coerce("max_delay_seconds", float, "max_delay_seconds")),
            "backoff_factor": float(_coerce("backoff_factor", float, "backoff_factor")),
        }

    def _resolve_retry_config(self) -> RetryConfig:
        defaults = self._load_retry_policy_defaults()
        max_retries_source = "policy_configs/retry_policy.yaml"
        base_delay_source = "policy_configs/retry_policy.yaml"
        max_delay_source = "policy_configs/retry_policy.yaml"

        if os.getenv("PIPELINE_RETRY_MAX"):
            max_retries = self._parse_env_int(
                "PIPELINE_RETRY_MAX",
                default=int(defaults["max_retries"]),
            )
            max_retries_source = "PIPELINE_RETRY_MAX"
        else:
            max_retries = int(defaults["max_retries"])

        if os.getenv("PIPELINE_RETRY_BASE_DELAY"):
            base_delay = self._parse_env_float(
                "PIPELINE_RETRY_BASE_DELAY",
                default=float(defaults["base_delay_seconds"]),
            )
            base_delay_source = "PIPELINE_RETRY_BASE_DELAY"
        else:
            base_delay = float(defaults["base_delay_seconds"])

        if os.getenv("PIPELINE_RETRY_MAX_DELAY"):
            max_delay = self._parse_env_float(
                "PIPELINE_RETRY_MAX_DELAY",
                default=float(defaults["max_delay_seconds"]),
            )
            max_delay_source = "PIPELINE_RETRY_MAX_DELAY"
        else:
            max_delay = float(defaults["max_delay_seconds"])

        backoff_factor = float(defaults["backoff_factor"])

        self.log(
            "Resolved retry policy: "
            f"max_retries={max_retries} ({max_retries_source}), "
            f"base_delay={base_delay}s ({base_delay_source}), "
            f"max_delay={max_delay}s ({max_delay_source}), "
            f"backoff_factor={backoff_factor} (policy_configs/retry_policy.yaml)",
            "INFO",
        )

        return RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
        )

    def _is_production_mode(self) -> bool:
        return self.environment == "production" or resolve_production_mode()

    def _is_release_path_run(self) -> bool:
        return bool(
            parse_bool_env(os.getenv("RELEASE_PATH_RUN"), default=False)
            or parse_bool_env(os.getenv("BP_RELEASE_PATH_RUN"), default=False)
        )

    def _parse_env_int(self, name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            if self._is_production_mode():
                raise NonRetryableError(
                    f"Invalid {name} value '{raw}' in production; expected integer."
                )
            self.log(f"Invalid {name} value '{raw}', defaulting to {default}", "WARNING")
            return default

    def _parse_env_float(self, name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            if self._is_production_mode():
                raise NonRetryableError(
                    f"Invalid {name} value '{raw}' in production; expected number."
                )
            self.log(f"Invalid {name} value '{raw}', defaulting to {default}", "WARNING")
            return default

    @staticmethod
    def _parse_csv(value: Optional[str]) -> List[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    def _parse_resolution_env(self, name: str, default: List[int]) -> List[int]:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        parts = [part.strip() for part in re.split(r"[x,]", raw) if part.strip()]
        if len(parts) != 2:
            if self._is_production_mode():
                raise NonRetryableError(
                    f"Invalid {name} value '{raw}' in production; expected WIDTHxHEIGHT."
                )
            self.log(
                f"Invalid {name} value '{raw}', expected WIDTHxHEIGHT; defaulting to {default}",
                "WARNING",
            )
            return default
        try:
            width = int(parts[0])
            height = int(parts[1])
        except ValueError:
            if self._is_production_mode():
                raise NonRetryableError(
                    f"Invalid {name} value '{raw}' in production; expected integers."
                )
            self.log(
                f"Invalid {name} value '{raw}', expected integers; defaulting to {default}",
                "WARNING",
            )
            return default
        return [width, height]

    def _cleanup_local_output_dirs(
        self,
        output_dirs: Dict[str, Path],
        robots: List[str],
        *,
        enabled: bool,
    ) -> Dict[str, Dict[str, Any]]:
        report = {
            "cleaned": {},
            "left_behind": {},
        }
        for robot in robots:
            output_dir = output_dirs.get(robot)
            output_path = str(output_dir) if output_dir else None
            if not enabled:
                report["left_behind"][robot] = {
                    "path": output_path,
                    "reason": "cleanup_disabled",
                }
                continue
            if not output_dir:
                report["left_behind"][robot] = {
                    "path": None,
                    "reason": "missing_output_dir",
                }
                continue
            if not output_dir.exists():
                report["left_behind"][robot] = {
                    "path": output_path,
                    "reason": "missing_path",
                }
                continue
            try:
                shutil.rmtree(output_dir)
                report["cleaned"][robot] = {"path": output_path}
            except Exception as exc:
                report["left_behind"][robot] = {
                    "path": output_path,
                    "reason": str(exc),
                }
        return report

    def _firebase_prefix_for_robot(
        self,
        firebase_prefix: str,
        robot: str,
        job_id: str,
        *,
        multi_robot: bool,
    ) -> str:
        del job_id
        from tools.firebase_upload.firebase_upload_orchestrator import build_firebase_upload_prefix

        return build_firebase_upload_prefix(
            self.scene_id,
            robot_type=robot if multi_robot else None,
            prefix=firebase_prefix,
        )

    def _cleanup_firebase_artifacts(
        self,
        *,
        firebase_prefix: str,
        job_id: str,
        robots: List[str],
        upload_summaries: Dict[str, Any],
        enabled: bool,
        multi_robot: bool,
    ) -> Dict[str, Dict[str, Any]]:
        report = {
            "cleaned": {},
            "left_behind": {},
        }
        if not robots:
            return report
        if not enabled:
            for robot in robots:
                report["left_behind"][robot] = {
                    "reason": "cleanup_disabled",
                }
            return report

        from tools.firebase_upload.uploader import cleanup_firebase_paths

        for robot in robots:
            summary = upload_summaries.get(robot) or {}
            file_statuses = summary.get("file_statuses", []) or []
            recorded_paths = [
                status.get("remote_path")
                for status in file_statuses
                if status.get("status") in {"uploaded", "reuploaded"}
            ]
            if recorded_paths:
                cleanup_result = cleanup_firebase_paths(paths=recorded_paths)
                report["cleaned"][robot] = cleanup_result
                if cleanup_result.get("failed"):
                    report["left_behind"][robot] = {
                        "reason": "delete_failed",
                        "details": cleanup_result.get("failed"),
                    }
                continue

            prefix = self._firebase_prefix_for_robot(
                firebase_prefix,
                robot,
                job_id,
                multi_robot=multi_robot,
            )
            cleanup_result = cleanup_firebase_paths(prefix=prefix)
            report["cleaned"][robot] = cleanup_result
            if cleanup_result.get("failed"):
                report["left_behind"][robot] = {
                    "reason": "delete_failed",
                    "details": cleanup_result.get("failed"),
                }
        return report

    @staticmethod
    def _merge_cleanup_reports(
        base_report: Optional[Dict[str, Any]],
        extra_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not base_report:
            return extra_report
        merged = {
            "cleaned": {},
            "left_behind": {},
        }
        merged["cleaned"].update(base_report.get("cleaned", {}))
        merged["cleaned"].update(extra_report.get("cleaned", {}))
        merged["left_behind"].update(base_report.get("left_behind", {}))
        merged["left_behind"].update(extra_report.get("left_behind", {}))
        return merged

    @staticmethod
    def _merge_firebase_cleanup_by_robot(
        base_report: Dict[str, Any],
        extra_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        cleaned = extra_report.get("cleaned", {})
        left_behind = extra_report.get("left_behind", {})
        for robot in set(cleaned) | set(left_behind):
            existing = base_report.get(robot)
            if existing is None:
                merged_entry: Dict[str, Any] = {}
            elif isinstance(existing, dict) and ("cleanup" in existing or "left_behind" in existing):
                merged_entry = dict(existing)
            else:
                merged_entry = {"cleanup": existing}
            if robot in cleaned:
                merged_entry["cleanup"] = cleaned[robot]
            if robot in left_behind:
                merged_entry["left_behind"] = left_behind[robot]
            base_report[robot] = merged_entry
        return base_report

    def _resolve_default_capture_config(self) -> Dict[str, Any]:
        default_cameras = ["wrist", "overhead"]
        default_resolution = [1280, 720]
        default_stream_ids = ["rgb", "depth", "segmentation"]

        cameras = self._parse_csv(os.getenv("DEFAULT_CAMERA_IDS")) or default_cameras
        stream_ids = self._parse_csv(os.getenv("DEFAULT_STREAM_IDS")) or default_stream_ids
        resolution = self._parse_resolution_env("DEFAULT_CAMERA_RESOLUTION", default_resolution)

        return {
            "cameras": cameras,
            "resolution": resolution,
            "modalities": ["rgb", "depth"],
            "stream_ids": stream_ids,
        }

    def _resolve_geniesim_robot_types(self) -> List[str]:
        raw_robot_types = os.getenv("ROBOT_TYPES") or os.getenv("GENIESIM_ROBOT_TYPES")
        if raw_robot_types is not None:
            parsed = self._parse_csv(raw_robot_types)
            if parsed:
                return parsed
        legacy_robot = os.getenv("GENIESIM_ROBOT_TYPE", "franka")
        return [legacy_robot] if legacy_robot else ["franka"]

    def _resolve_step_timeouts(self) -> Dict[PipelineStep, Optional[float]]:
        adaptive_config = load_adaptive_timeout_config()
        default_timeout_seconds = adaptive_config.selected_timeout_seconds
        raw_timeouts = load_pipeline_step_timeouts()
        resolved: Dict[PipelineStep, Optional[float]] = {}
        unknown: Dict[str, Optional[float]] = {}
        for step_name, timeout_value in raw_timeouts.items():
            try:
                step = PipelineStep(step_name)
            except ValueError:
                unknown[step_name] = timeout_value
                continue
            resolved[step] = timeout_value

        if unknown:
            formatted = ", ".join(
                f"{key}={value if value is not None else 'disabled'}"
                for key, value in sorted(unknown.items())
            )
            self.log(
                f"Unrecognized pipeline step timeouts ignored: {formatted}",
                "WARNING",
            )

        for step in PipelineStep:
            if step not in resolved:
                resolved[step] = default_timeout_seconds

        adaptive_details = [
            f"default_timeout_seconds={adaptive_config.default_timeout_seconds:.0f}s",
        ]
        if adaptive_config.selected_bundle_tier:
            bundle_value = adaptive_config.bundle_tier[adaptive_config.selected_bundle_tier]
            adaptive_details.append(
                f"bundle_tier={adaptive_config.selected_bundle_tier} ({bundle_value:.0f}s)"
            )
        if adaptive_config.selected_scene_complexity:
            complexity_value = adaptive_config.scene_complexity[
                adaptive_config.selected_scene_complexity
            ]
            adaptive_details.append(
                f"scene_complexity={adaptive_config.selected_scene_complexity} ({complexity_value:.0f}s)"
            )
        adaptive_details.append(f"selected_timeout_seconds={default_timeout_seconds:.0f}s")
        overrides_summary = (
            "none"
            if not raw_timeouts
            else ", ".join(
                f"{key}={value if value is not None else 'disabled'}"
                for key, value in sorted(raw_timeouts.items())
            )
        )
        final_summary = ", ".join(
            f"{step.value}={value if value is not None else 'disabled'}"
            for step, value in resolved.items()
        )
        self.log(
            "Resolved pipeline step timeouts using adaptive defaults "
            f"({'; '.join(adaptive_details)}), "
            f"overrides=({overrides_summary}); "
            f"final=({final_summary}).",
            "INFO",
        )
        return resolved

    def _run_with_timeout(self, step: PipelineStep, action: Any) -> Any:
        timeout_seconds = self._step_timeouts.get(step)
        if timeout_seconds is None:
            return action()
        try:
            with timeout_thread(
                timeout_seconds,
                f"{step.value} timed out after {timeout_seconds:.0f}s",
            ):
                return action()
        except TimeoutError as exc:
            context = ErrorContext(
                scene_id=self.scene_id,
                step=step.value,
                max_attempts=1,
                additional={"timeout_seconds": timeout_seconds},
            )
            message = (
                f"{step.value} timed out after {timeout_seconds:.0f}s "
                f"for scene {self.scene_id}"
            )
            raise PipelineError(
                message,
                category=ErrorCategory.EXTERNAL_SERVICE,
                retryable=False,
                context=context,
                cause=exc,
            ) from exc

    def _run_with_retry(self, step: PipelineStep, action: Any) -> Any:
        config = self.retry_config
        timeout_seconds = self._step_timeouts.get(step)

        def on_retry(attempt: int, exc: Exception, delay: float) -> None:
            self.log(
                (
                    f"{step.value} retry {attempt}/{config.max_retries} after {delay:.2f}s: "
                    f"{self._summarize_exception(exc)}"
                ),
                "WARNING",
            )

        def on_failure(attempt: int, exc: Exception) -> None:
            self.log(
                f"{step.value} failed after {attempt} attempts: {self._summarize_exception(exc)}",
                "ERROR",
            )

        def action_with_timeout() -> Any:
            if timeout_seconds is None:
                return action()
            try:
                with timeout_thread(
                    timeout_seconds,
                    f"{step.value} timed out after {timeout_seconds:.0f}s",
                ):
                    return action()
            except TimeoutError as exc:
                context = ErrorContext(
                    scene_id=self.scene_id,
                    step=step.value,
                    max_attempts=config.max_retries,
                    additional={"timeout_seconds": timeout_seconds},
                )
                message = (
                    f"{step.value} timed out after {timeout_seconds:.0f}s "
                    f"for scene {self.scene_id}"
                )
                raise PipelineError(
                    message,
                    category=ErrorCategory.EXTERNAL_SERVICE,
                    retryable=True,
                    context=context,
                    cause=exc,
                ) from exc

        decorator = retry_with_backoff(
            max_retries=config.max_retries,
            base_delay=config.base_delay,
            max_delay=config.max_delay,
            backoff_factor=config.backoff_factor,
            jitter=config.jitter,
            on_retry=on_retry,
            on_failure=on_failure,
        )
        return decorator(action_with_timeout)()

    def _log_exception_traceback(self, context: str, exc: Exception) -> None:
        self.log(f"{context}: {self._summarize_exception(exc)}", "ERROR")
        if self.debug:
            self.log(traceback.format_exc(), "DEBUG")

    def _handle_step_exception(
        self,
        step: PipelineStep,
        exc: Exception,
        context: str,
        *,
        duration_seconds: float = 0,
    ) -> StepResult:
        pipeline_error = classify_exception(exc)
        log_pipeline_error(pipeline_error, f"{step.value} failed")
        self._log_exception_traceback(context, exc)
        error_context = None
        if getattr(pipeline_error, "context", None) is not None:
            try:
                error_context = pipeline_error.context.to_dict()
            except Exception:
                error_context = pipeline_error.context
        detail = (
            f"error_type={pipeline_error.__class__.__name__} "
            f"retryable={pipeline_error.retryable}"
        )
        if error_context:
            detail = f"{detail} context={error_context}"
        self.log(f"{step.value} classified error: {detail}", "ERROR")
        if pipeline_error.retryable:
            raise RetryableError(pipeline_error.message) from exc
        user_message = getattr(pipeline_error, "user_message", None) or pipeline_error.message
        sanitized_message = self._sanitize_error_message(user_message)
        summary_message = self._summarize_exception(pipeline_error)
        message = sanitized_message or summary_message
        return StepResult(
            step=step,
            success=False,
            duration_seconds=duration_seconds,
            message=f"Error: {message}",
        )

    def _handle_retryable_exception(
        self,
        step: PipelineStep,
        exc: Exception,
        context: str,
        *,
        duration_seconds: float = 0,
    ) -> StepResult:
        pipeline_error = classify_exception(exc)
        self._log_exception_traceback(context, exc)
        error_context = None
        if getattr(pipeline_error, "context", None) is not None:
            try:
                error_context = pipeline_error.context.to_dict()
            except Exception:
                error_context = pipeline_error.context
        detail = (
            f"error_type={pipeline_error.__class__.__name__} "
            f"retryable={pipeline_error.retryable}"
        )
        if error_context:
            detail = f"{detail} context={error_context}"
        self.log(f"{step.value} classified error: {detail}", "ERROR")
        user_message = getattr(pipeline_error, "user_message", None) or pipeline_error.message
        sanitized_message = self._sanitize_error_message(user_message)
        summary_message = self._summarize_exception(pipeline_error)
        message = sanitized_message or summary_message
        return StepResult(
            step=step,
            success=False,
            duration_seconds=duration_seconds,
            message=f"Error: {message}",
        )

    def _write_marker(
        self,
        marker_path: Path,
        status: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a simple JSON marker file."""
        marker_payload: Dict[str, Any] = {
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "scene_id": self.scene_id,
            "run_id": self.run_id,
        }
        if payload:
            marker_payload.update(payload)
        _safe_write_text(
            marker_path,
            json.dumps(marker_payload, indent=2),
            context="marker file",
        )

    def _load_marker_payload(
        self,
        marker_path: Path,
        marker_label: str,
    ) -> Optional[Dict[str, Any]]:
        if not marker_path.is_file():
            return None
        try:
            payload = _load_json(marker_path, marker_label)
        except NonRetryableError as exc:
            self.log(f"Ignoring unreadable {marker_label}: {exc}", "WARNING")
            return None
        if not isinstance(payload, dict):
            self.log(
                f"Ignoring malformed {marker_label}: expected JSON object at {marker_path}",
                "WARNING",
            )
            return None
        return payload

    def _clear_stale_marker_if_mismatched(
        self,
        marker_path: Path,
        marker_label: str,
        *,
        expected_job_id: Optional[str],
        expected_run_id: Optional[str],
    ) -> bool:
        """Return True when marker matches expected job/run, else remove stale marker."""
        if not marker_path.is_file():
            return False
        marker_payload = self._load_marker_payload(marker_path, marker_label)
        if not marker_payload:
            try:
                marker_path.unlink(missing_ok=True)
            except Exception as exc:
                self.log(f"Failed to remove malformed marker {marker_path}: {exc}", "WARNING")
            return False
        marker_job_id = str(marker_payload.get("job_id") or "").strip()
        marker_run_id = str(marker_payload.get("run_id") or "").strip()
        if (
            expected_job_id
            and expected_run_id
            and marker_job_id == expected_job_id
            and marker_run_id == expected_run_id
        ):
            return True
        self.log(
            f"Ignoring stale {marker_label} at {marker_path} "
            f"(expected job_id={expected_job_id!r}, run_id={expected_run_id!r}; "
            f"found job_id={marker_job_id!r}, run_id={marker_run_id!r}).",
            "WARNING",
        )
        try:
            marker_path.unlink(missing_ok=True)
        except Exception as exc:
            self.log(f"Failed to remove stale marker {marker_path}: {exc}", "WARNING")
        return False

    def _resolve_geniesim_job_identity(self) -> Tuple[Optional[str], Optional[str]]:
        """Return current Genie Sim (job_id, run_id) when job metadata is available."""
        job_path = self.geniesim_dir / "job.json"
        if not job_path.is_file():
            return None, None
        try:
            payload = _load_json(job_path, "Genie Sim job payload")
        except NonRetryableError as exc:
            self.log(f"Failed to load Genie Sim job metadata: {exc}", "WARNING")
            return None, None
        if not isinstance(payload, dict):
            self.log(
                f"Failed to parse Genie Sim job metadata at {job_path}: expected JSON object.",
                "WARNING",
            )
            return None, None
        job_id = str(payload.get("job_id") or "").strip() or None
        run_id = str(payload.get("run_id") or "").strip() or self.run_id
        return job_id, run_id

    def _resolve_quality_gate_report_path(self) -> Path:
        report_dir = self.scene_dir / "quality_gates"
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir / "quality_gate_report.json"

    def _should_skip_quality_gates(self) -> bool:
        skip_requested = parse_bool_env(os.getenv("SKIP_QUALITY_GATES"), default=False)
        if not skip_requested:
            return False
        if self._is_production_mode():
            raise NonRetryableError(
                "SKIP_QUALITY_GATES is not allowed in production mode."
            )
        return True

    @staticmethod
    def _dir_has_files(path: Path) -> bool:
        return path.is_dir() and any(path.iterdir())

    def _build_readiness_checklist(self) -> Dict[str, bool]:
        steps_success = {result.step.value: result.success for result in self.results}
        return {
            "usd_valid": (self.usd_dir / "scene.usda").exists(),
            "physics_stable": steps_success.get(PipelineStep.SIMREADY.value, False),
            "episodes_generated": self._dir_has_files(self.episodes_dir),
            "replicator_ready": self._dir_has_files(self.replicator_dir),
            "isaac_lab_ready": self._dir_has_files(self.isaac_lab_dir),
            "dwm_ready": self._dir_has_files(self.dwm_dir),
        }

    @staticmethod
    def _resolve_episode_metadata_path(lerobot_dir: Path) -> Path:
        preferred = lerobot_dir / "meta" / "info.json"
        if preferred.is_file():
            return preferred
        fallback = lerobot_dir / "metadata.json"
        if fallback.is_file():
            return fallback
        return preferred

    @staticmethod
    def _build_episode_stats(result: Any) -> Dict[str, Any]:
        episodes_collected = getattr(result, "episodes_collected", 0) if result else 0
        episodes_passed = getattr(result, "episodes_passed", 0) if result else 0
        episode_stats = {
            "total_generated": episodes_collected,
            "passed_quality_filter": episodes_passed,
        }
        if result is None:
            return episode_stats
        average_quality = getattr(result, "average_quality_score", None)
        if average_quality is not None:
            episode_stats["average_quality_score"] = float(average_quality)
        collision_free_rate = getattr(result, "collision_free_rate", None)
        if collision_free_rate is not None:
            episode_stats["collision_free_rate"] = float(collision_free_rate)
        return episode_stats

    def _quality_gate_context_for_step(
        self,
        step: PipelineStep,
    ) -> Optional[Dict[str, Any]]:
        if step == PipelineStep.REGEN3D:
            manifest_path = self.assets_dir / "scene_manifest.json"
            if not manifest_path.is_file():
                return None
            return {
                "checkpoint": QualityGateCheckpoint.MANIFEST_VALIDATED,
                "context": {
                    "scene_id": self.scene_id,
                    "manifest": _load_json(manifest_path, "quality gate manifest"),
                },
            }
        if step == PipelineStep.SCALE:
            report_path = self.assets_dir / "scale_report.json"
            return {
                "checkpoint": QualityGateCheckpoint.SCALE_COMPLETE,
                "context": {
                    "scene_id": self.scene_id,
                    "scale_report_path": str(report_path),
                },
            }
        if step == PipelineStep.INTERACTIVE:
            manifest_path = self.assets_dir / "scene_manifest.json"
            required_count = 0
            if manifest_path.is_file():
                manifest = _load_json(manifest_path, "interactive manifest")
                required_count = len([
                    obj for obj in manifest.get("objects", [])
                    if obj.get("type") == "interactive"
                ])
            results_path = self.assets_dir / "interactive" / "interactive_results.json"
            return {
                "checkpoint": QualityGateCheckpoint.INTERACTIVE_COMPLETE,
                "context": {
                    "scene_id": self.scene_id,
                    "interactive_results_path": str(results_path),
                    "required_interactive_count": required_count,
                },
            }
        if step == PipelineStep.SIMREADY:
            physics_summary_path = self.assets_dir / "simready_physics.json"
            manifest_path = self.assets_dir / "scene_manifest.json"
            physics_objects: List[Dict[str, Any]] = []
            source = "manifest"

            # Prefer simready_physics.json generated by simready-job when present.
            if physics_summary_path.is_file():
                try:
                    physics_summary = _load_json(
                        physics_summary_path,
                        "quality gate simready physics summary",
                    )
                    for obj in physics_summary.get("objects", []):
                        static_friction = obj.get("static_friction")
                        dynamic_friction = obj.get("dynamic_friction")
                        friction = (
                            static_friction
                            if static_friction is not None
                            else dynamic_friction if dynamic_friction is not None else 0.0
                        )
                        physics_objects.append({
                            "id": obj.get("id"),
                            "mass": obj.get("mass_kg", obj.get("mass", 0.0)),
                            "friction": friction,
                        })
                    if physics_objects:
                        source = "simready_physics"
                except NonRetryableError as exc:
                    self.log(
                        f"WARNING: Failed to parse simready physics summary ({exc}); "
                        "falling back to scene manifest physics",
                        "WARNING",
                    )

            if not physics_objects:
                if not manifest_path.is_file():
                    return None
                manifest = _load_json(manifest_path, "quality gate simready manifest")
                for obj in manifest.get("objects", []):
                    physics = obj.get("physics", {}) or {}
                    physics_objects.append({
                        "id": obj.get("id"),
                        "mass": physics.get("mass_kg", 0.0),
                        "friction": physics.get(
                            "friction_static",
                            physics.get("friction_dynamic", 0.0),
                        ),
                    })
            return {
                "checkpoint": QualityGateCheckpoint.SIMREADY_COMPLETE,
                "context": {
                    "scene_id": self.scene_id,
                    "physics_properties": {
                        "objects": physics_objects,
                        "source": source,
                    },
                },
            }
        if step == PipelineStep.USD:
            usd_path = self.usd_dir / "scene.usda"
            return {
                "checkpoint": QualityGateCheckpoint.USD_ASSEMBLED,
                "context": {
                    "scene_id": self.scene_id,
                    "usd_path": str(usd_path),
                },
            }
        if step == PipelineStep.INVENTORY_ENRICHMENT:
            inventory_path = self.seg_dir / "inventory.json"
            enriched_path = self.seg_dir / "inventory_enriched.json"
            return {
                "checkpoint": QualityGateCheckpoint.INVENTORY_ENRICHMENT_COMPLETE,
                "context": {
                    "scene_id": self.scene_id,
                    "inventory_path": str(inventory_path),
                    "inventory_enriched_path": str(enriched_path),
                },
            }
        if step == PipelineStep.REPLICATOR:
            return {
                "checkpoint": QualityGateCheckpoint.REPLICATOR_COMPLETE,
                "context": {
                    "scene_id": self.scene_id,
                    "replicator_bundle_dir": str(self.replicator_dir),
                },
            }
        if step == PipelineStep.VARIATION_GEN:
            variation_assets_path = self.scene_dir / "variation_assets" / "variation_assets.json"
            return {
                "checkpoint": QualityGateCheckpoint.VARIATION_GEN_COMPLETE,
                "context": {
                    "scene_id": self.scene_id,
                    "variation_assets_path": str(variation_assets_path),
                },
            }
        if step == PipelineStep.ISAAC_LAB:
            return {
                "checkpoint": QualityGateCheckpoint.ISAAC_LAB_GENERATED,
                "context": {
                    "scene_id": self.scene_id,
                    "isaac_lab_dir": str(self.isaac_lab_dir),
                },
            }
        if step == PipelineStep.DWM:
            return {
                "checkpoint": QualityGateCheckpoint.DWM_PREPARED,
                "context": {
                    "scene_id": self.scene_id,
                    "dwm_output_dir": str(self.dwm_dir),
                },
            }
        if step == PipelineStep.GENIESIM_IMPORT:
            return {
                "checkpoint": QualityGateCheckpoint.SCENE_READY,
                "context": {
                    "scene_id": self.scene_id,
                    "readiness_checklist": self._build_readiness_checklist(),
                },
            }
        if step == PipelineStep.GENIESIM_SUBMIT:
            if not self._geniesim_output_dirs:
                return None
            contexts: List[Dict[str, Any]] = []
            for robot, output_dir in self._geniesim_output_dirs.items():
                if not output_dir:
                    continue
                lerobot_dir = output_dir / "lerobot"
                metadata_path = self._resolve_episode_metadata_path(lerobot_dir)
                result = self._geniesim_local_run_results.get(robot)
                contexts.append({
                    "scene_id": self.scene_id,
                    "episode_stats": self._build_episode_stats(result),
                    "lerobot_dataset_path": str(lerobot_dir),
                    "episode_metadata_path": str(metadata_path),
                    "robot_type": robot,
                })
            if not contexts:
                return None
            return {
                "checkpoint": QualityGateCheckpoint.EPISODES_GENERATED,
                "context": contexts[0] if len(contexts) == 1 else contexts,
            }
        return None

    def _apply_quality_gates(
        self,
        step: PipelineStep,
        result: StepResult,
        *,
        checkpointed: bool = False,
    ) -> StepResult:
        try:
            gate_payload = self._quality_gate_context_for_step(step)
        except NonRetryableError as exc:
            result.success = False
            result.message = str(exc)
            return result
        if not gate_payload:
            return result

        checkpoint = gate_payload["checkpoint"]
        context = gate_payload["context"]
        self._quality_gates.register_required_checkpoint(checkpoint)
        outputs = result.outputs
        outputs["quality_gate_checkpoint"] = checkpoint.value
        outputs["quality_gate_report"] = str(self._quality_gate_report_path)
        if checkpointed:
            outputs["quality_gate_step_source"] = "checkpoint_resume"

        try:
            should_skip = self._should_skip_quality_gates()
        except NonRetryableError as exc:
            reason = str(exc)
            self._quality_gates.register_skipped_checkpoint(checkpoint, reason)
            self._quality_gates.save_report(self.scene_id, self._quality_gate_report_path)
            report = self._quality_gates.to_report(self.scene_id)
            outputs["quality_gate_summary"] = report.get("summary", {})
            outputs["quality_gate_skipped"] = True
            outputs["quality_gate_skip_reason"] = reason
            result.success = False
            result.message = reason
            outputs["quality_gate_blocked"] = True
            return result

        if should_skip:
            skip_reason = "SKIP_QUALITY_GATES environment override"
            if checkpointed:
                skip_reason += " on checkpoint-resumed step"
            self.log(
                f"SKIP_QUALITY_GATES enabled - skipping quality gates for {checkpoint.value}",
                "WARNING",
            )
            self._quality_gates.register_skipped_checkpoint(checkpoint, skip_reason)
            self._quality_gates.save_report(self.scene_id, self._quality_gate_report_path)
            report = self._quality_gates.to_report(self.scene_id)
            outputs["quality_gate_summary"] = report.get("summary", {})
            outputs["quality_gate_skipped"] = True
            outputs["quality_gate_skip_reason"] = skip_reason
            return result

        if isinstance(context, list):
            gate_results = []
            for entry in context:
                gate_results.extend(
                    self._quality_gates.run_checkpoint(
                        checkpoint=checkpoint,
                        context=entry,
                        notification_service=self._notification_service,
                    )
                )
        else:
            gate_results = self._quality_gates.run_checkpoint(
                checkpoint=checkpoint,
                context=context,
                notification_service=self._notification_service,
            )
        self._quality_gates.save_report(self.scene_id, self._quality_gate_report_path)
        report = self._quality_gates.to_report(self.scene_id)
        outputs["quality_gate_summary"] = report.get("summary", {})
        outputs["quality_gate_skipped"] = False

        blocked = any((not entry.passed and entry.severity == "error") for entry in gate_results)
        if blocked:
            result.success = False
            result.message = f"Quality gates blocked at {checkpoint.value}"
            outputs["quality_gate_blocked"] = True
        return result

    @staticmethod
    def _is_cloud_path(path: Path, gcs_root: Path) -> bool:
        path_str = str(path)
        if path_str.startswith("gs://"):
            return True
        try:
            path.resolve().relative_to(gcs_root)
        except ValueError:
            return False
        return True

    def _validate_production_startup(self) -> Dict[str, Any]:
        production_mode = resolve_production_mode()
        release_path_run = self._is_release_path_run()
        report: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "production_mode": production_mode,
            "release_path_run": release_path_run,
            "ok": True,
            "errors": [],
            "warnings": [],
            "checks": {},
        }

        if release_path_run and not production_mode:
            report["errors"].append(
                "RELEASE_PATH_RUN requires production mode. Set PIPELINE_ENV=production."
            )

        if production_mode:
            skip_quality_gates_requested = parse_bool_env(
                os.getenv("SKIP_QUALITY_GATES"),
                default=False,
            )
            report["checks"]["quality_gate_policy"] = {
                "skip_requested": skip_quality_gates_requested,
            }
            if skip_quality_gates_requested:
                report["errors"].append(
                    "SKIP_QUALITY_GATES cannot be enabled in production mode."
                )

            report["checks"]["checkpoint_hashes"] = {
                "enabled": self.enable_checkpoint_hashes,
                "env": os.getenv("BP_CHECKPOINT_HASHES"),
            }
            if not self.enable_checkpoint_hashes:
                report["errors"].append(
                    "Checkpoint output hashes must be enabled in production. "
                    "Set BP_CHECKPOINT_HASHES=1 to continue."
                )

            gcs_report = validate_gcs_credentials()
            firebase_report = validate_firebase_credentials(required=True)
            report["checks"]["gcs_credentials"] = gcs_report
            report["checks"]["firebase_credentials"] = firebase_report
            report["warnings"].extend(gcs_report.get("warnings", []))
            report["warnings"].extend(firebase_report.get("warnings", []))
            if not gcs_report.get("valid", False):
                report["errors"].extend(gcs_report.get("errors", []))
            if not firebase_report.get("valid", False):
                report["errors"].extend(firebase_report.get("errors", []))

            gcs_root = Path(os.getenv("GCS_MOUNT_ROOT", "/mnt/gcs")).resolve()
            path_checks: Dict[str, Any] = {
                "scene_dir": self.scene_dir,
                "assets_dir": self.assets_dir,
                "layout_dir": self.layout_dir,
                "usd_dir": self.usd_dir,
                "replicator_dir": self.replicator_dir,
                "geniesim_dir": self.geniesim_dir,
                "episodes_dir": self.episodes_dir,
            }
            path_results: Dict[str, Any] = {}
            for label, path in path_checks.items():
                is_cloud_path = self._is_cloud_path(path, gcs_root)
                path_results[label] = {
                    "path": str(path),
                    "cloud_path": is_cloud_path,
                }
                if not is_cloud_path:
                    report["errors"].append(
                        f"{label} must be a GCS-mounted path under {gcs_root} (got {path})"
                    )
            report["checks"]["path_validation"] = {
                "gcs_mount_root": str(gcs_root),
                "paths": path_results,
            }

            production_floors = {
                "collision_free_rate_min": 0.90,
                "quality_pass_rate_min": 0.75,
                "quality_score_min": 0.92,
                "min_episodes_required": 5,
            }
            try:
                quality_config = load_quality_gate_config()
                episode_thresholds = quality_config.episodes
                tier_thresholds = episode_thresholds.tier_thresholds or {}
                report["checks"]["quality_gate_config"] = {
                    "loaded": True,
                    "episode_thresholds": {
                        "collision_free_rate_min": episode_thresholds.collision_free_rate_min,
                        "quality_pass_rate_min": episode_thresholds.quality_pass_rate_min,
                        "quality_score_min": episode_thresholds.quality_score_min,
                        "min_episodes_required": episode_thresholds.min_episodes_required,
                    },
                    "tier_thresholds": tier_thresholds,
                }
                for key, floor_value in production_floors.items():
                    current_value = getattr(episode_thresholds, key)
                    if current_value < floor_value:
                        report["errors"].append(
                            f"quality_gate_config.thresholds.episodes.{key} must be >= {floor_value} "
                            f"in production (got {current_value})"
                        )
                for tier_name, threshold_values in tier_thresholds.items():
                    if not isinstance(threshold_values, dict):
                        continue
                    for key, floor_value in production_floors.items():
                        if key in threshold_values and threshold_values[key] < floor_value:
                            report["errors"].append(
                                "quality_gate_config.thresholds.episodes.tier_thresholds."
                                f"{tier_name}.{key} must be >= {floor_value} in production "
                                f"(got {threshold_values[key]})"
                            )
            except Exception as exc:
                report["checks"]["quality_gate_config"] = {
                    "loaded": False,
                    "error": self._summarize_exception(exc),
                }
                report["errors"].append(
                    f"Quality gate config failed to load: {self._summarize_exception(exc)}"
                )

        report["ok"] = not report["errors"]
        report_path = self.scene_dir / "production_validation.json"
        _safe_write_text(
            report_path,
            json.dumps(report, indent=2),
            context="production validation report",
        )

        if report["errors"]:
            error_summary = "\n".join(f"  - {err}" for err in report["errors"])
            raise NonRetryableError(
                "Production startup validation failed:\n"
                f"{error_summary}\n"
                f"Report: {report_path}"
            )

        return report

    def _resolve_step_circuit_breaker_path(self) -> Path:
        return self.scene_dir / ".circuit_breaker.json"

    def _rotate_circuit_breaker_state(self, persistence_path: Path, reason: str) -> None:
        if persistence_path.exists():
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            rotated_path = persistence_path.with_name(
                f"{persistence_path.name}.{timestamp}.bak"
            )
            persistence_path.replace(rotated_path)
            self.log(
                f"[CIRCUIT-BREAKER] Rotated breaker state to {rotated_path} ({reason}).",
                "INFO",
            )
        tmp_path = persistence_path.with_suffix(f"{persistence_path.suffix}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()

    def _prepare_step_circuit_breaker_persistence(self, reset_breaker: bool) -> Path:
        persistence_path = self._resolve_step_circuit_breaker_path()
        persistence_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = persistence_path.with_suffix(".meta.json")
        previous_scene_id = None
        if meta_path.exists():
            try:
                payload = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                previous_scene_id = payload.get("scene_id")

        if reset_breaker:
            self._rotate_circuit_breaker_state(persistence_path, "reset flag")
        elif previous_scene_id and previous_scene_id != self.scene_id:
            self._rotate_circuit_breaker_state(
                persistence_path,
                f"scene changed from {previous_scene_id} to {self.scene_id}",
            )

        meta_payload = {
            "scene_id": self.scene_id,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        _safe_write_text(
            meta_path,
            json.dumps(meta_payload, indent=2),
            context="circuit breaker metadata",
        )

        return persistence_path

    def _ensure_step_circuit_breaker(self, *, reset_breaker: bool) -> None:
        if self._step_circuit_breaker is not None:
            if reset_breaker:
                self.log("[CIRCUIT-BREAKER] Resetting provided circuit breaker.", "INFO")
                self._step_circuit_breaker.reset()
            return

        persistence_path = self._prepare_step_circuit_breaker_persistence(reset_breaker)
        self._step_circuit_breaker_persistence_path = persistence_path
        self._step_circuit_breaker = self._initialize_step_circuit_breaker(
            persistence_path=persistence_path
        )

    def _initialize_step_circuit_breaker(
        self,
        *,
        persistence_path: Optional[Path] = None,
    ) -> CircuitBreaker:
        """
        Initialize circuit breaker for step-level failure protection (P1).

        Opens after consecutive step failures to prevent cascading failures.
        When open, remaining steps are skipped until recovery timeout.
        """
        failure_threshold = int(os.getenv("PIPELINE_CIRCUIT_BREAKER_FAILURES", "3"))
        success_threshold = int(os.getenv("PIPELINE_CIRCUIT_BREAKER_SUCCESSES", "2"))
        recovery_timeout = float(os.getenv("PIPELINE_CIRCUIT_BREAKER_RECOVERY_S", "60"))
        should_log_load = bool(persistence_path and persistence_path.exists())

        def on_open(name: str, failure_count: int) -> None:
            self.log(
                f"[CIRCUIT-BREAKER] Pipeline circuit breaker opened after {failure_count} "
                f"consecutive step failures. Remaining steps will be skipped until recovery.",
                "ERROR",
            )

        def on_half_open(name: str) -> None:
            self.log(
                "[CIRCUIT-BREAKER] Pipeline circuit breaker half-open. Testing if pipeline can recover.",
                "WARNING",
            )

        def on_close(name: str) -> None:
            self.log(
                "[CIRCUIT-BREAKER] Pipeline circuit breaker closed. Normal operation resumed.",
                "INFO",
            )

        breaker = CircuitBreaker(
            name=f"pipeline_{self.scene_id}",
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            recovery_timeout=recovery_timeout,
            on_open=on_open,
            on_half_open=on_half_open,
            on_close=on_close,
            persistence_path=persistence_path,
        )
        if should_log_load:
            self.log(
                f"[CIRCUIT-BREAKER] Loaded breaker state from {persistence_path}.",
                "INFO",
            )
        return breaker

    def run(
        self,
        steps: Optional[List[PipelineStep]] = None,
        run_validation: bool = False,
        resume_from: Optional[PipelineStep] = None,
        force_rerun_steps: Optional[List[PipelineStep]] = None,
        reset_breaker: bool = False,
        auto_trigger_import: Optional[bool] = None,
    ) -> bool:
        """Run the pipeline.

        Args:
            steps: Specific steps to run (default: all applicable)
            run_validation: Run QA validation at the end
            resume_from: Resume from the given step (skip completed steps with checkpoints)
            force_rerun_steps: Steps to rerun even if checkpoints exist
            reset_breaker: Reset the step circuit breaker persistence for this run
            auto_trigger_import: Run genie-sim-import after a completed submit when not scheduled.
                Defaults to True if genie-sim-submit is in steps and import is not.

        Returns:
            True if all steps succeeded
        """
        seed = configure_pipeline_seed()
        if seed is not None:
            self.log(f"Using pipeline seed: {seed}")

        self._apply_labs_flags(run_validation=run_validation)
        try:
            self._validate_production_startup()
        except NonRetryableError as exc:
            self.log(f"ERROR: {exc}", "ERROR")
            return False

        if steps is None:
            steps = self._resolve_default_steps()

        # Default auto_trigger_import to True if submit is in steps and import is not
        if auto_trigger_import is None:
            auto_trigger_import = (
                PipelineStep.GENIESIM_SUBMIT in steps
                and PipelineStep.GENIESIM_IMPORT not in steps
            )

        if run_validation and PipelineStep.VALIDATE not in steps:
            steps.append(PipelineStep.VALIDATE)

        self._ensure_step_circuit_breaker(reset_breaker=reset_breaker)

        if resume_from is not None:
            checkpoint_store = get_checkpoint_store(self.scene_dir, self.scene_id)
            if resume_from not in steps:
                self.log(
                    f"ERROR: resume-from step {resume_from.value} is not in requested steps",
                    "ERROR",
                )
                return False
            resume_index = steps.index(resume_from)
            prior_steps = steps[:resume_index]
            for step in prior_steps:
                expected_outputs = self._expected_output_paths(step)
                step_has_checkpoint = checkpoint_store.should_skip_step(
                    step.value,
                    expected_outputs=expected_outputs,
                    require_nonempty=True,
                    require_fresh_outputs=True,
                    validate_sidecar_metadata=True,
                )
                if step == PipelineStep.GENIESIM_IMPORT and step_has_checkpoint:
                    expected_job_id, expected_run_id = self._resolve_geniesim_job_identity()
                    marker_ok = self._clear_stale_marker_if_mismatched(
                        self.geniesim_dir / ".geniesim_import_complete",
                        "Genie Sim import completion marker",
                        expected_job_id=expected_job_id,
                        expected_run_id=expected_run_id,
                    )
                    step_has_checkpoint = marker_ok
                if step_has_checkpoint:
                    continue
                self.log(
                    (
                        "ERROR: resume-from requires a completed checkpoint and expected outputs "
                        f"for prior step {step.value}"
                    ),
                    "ERROR",
                )
                return False
            steps = steps[resume_index:]

        self.log("=" * 60)
        self.log("BlueprintPipeline Local Runner")
        self.log("=" * 60)
        self.log(f"Scene directory: {self.scene_dir}")
        self.log(f"Scene ID: {self.scene_id}")
        self.log(f"Steps: {[s.value for s in steps]}")
        if self._gcs_sync:
            self.log(f"GCS bucket: {self._gcs_sync.bucket_name}")
            self.log(f"GCS upload: {self._gcs_upload_outputs}")
        self.log("=" * 60)

        # Download inputs from GCS if requested
        if self._gcs_sync and self._gcs_download_inputs:
            try:
                self.log("[GCS] Downloading input image from GCS...")
                input_path = self._gcs_sync.download_inputs(preferred_object=self._gcs_input_object)
                self.log(f"[GCS] Input image downloaded: {input_path}")
            except FileNotFoundError as exc:
                self.log(f"[GCS] ERROR: {exc}", "ERROR")
                return False
            except Exception as exc:
                self.log(f"[GCS] ERROR: Failed to download inputs: {exc}", "ERROR")
                return False

        # Check prerequisites — regen3d/ is only required by steps that
        # consume raw reconstruction data.  Downstream-only workflows
        # (e.g. replicator, isaac-lab, episode-gen) that operate from
        # assets/seg/usd dirs can skip this check.  This also allows
        # BlueprintCapturePipeline-produced scenes (which store
        # reconstruction under nurec/ rather than regen3d/) to drive
        # downstream data-generation steps directly.
        _regen3d_consuming_steps = {
            PipelineStep.REGEN3D_RECONSTRUCT,
            PipelineStep.REGEN3D,
            PipelineStep.SCALE,
            PipelineStep.INTERACTIVE,
            PipelineStep.VALIDATE,
        }
        needs_regen3d = bool(set(steps) & _regen3d_consuming_steps)

        if not self.regen3d_dir.is_dir():
            if PipelineStep.REGEN3D_RECONSTRUCT in steps:
                self.log("regen3d/ dir will be created by regen3d-reconstruct step")
                self.regen3d_dir.mkdir(parents=True, exist_ok=True)
            elif needs_regen3d:
                self.log(f"ERROR: 3D-RE-GEN output not found at {self.regen3d_dir}", "ERROR")
                self.log(
                    "Run: python fixtures/generate_mock_regen3d.py first, "
                    "or add regen3d-reconstruct to --steps",
                    "ERROR",
                )
                return False
            else:
                self.log(
                    f"regen3d/ not found at {self.regen3d_dir} — "
                    "skipping (not required by requested steps)",
                    "WARNING",
                )

        if not self._preflight_articulation_requirements(steps):
            return False

        if self._steps_require_geniesim_preflight(steps):
            require_server = self._geniesim_requires_server(steps)
            if not self._run_geniesim_preflight(require_server=require_server):
                return False

        # Create output directories
        for d in [self.assets_dir, self.layout_dir, self.seg_dir,
                  self.usd_dir, self.replicator_dir, self.isaac_lab_dir,
                  self.geniesim_dir, self.episodes_dir,
                  self.dwm_dir, self.dream2flow_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Run each step
        all_success = True
        checkpoint_store = get_checkpoint_store(self.scene_dir, self.scene_id)
        forced_steps = set(force_rerun_steps or [])
        dependencies = self._resolve_step_dependencies(steps)
        def _execute_step(step: PipelineStep) -> bool:
            nonlocal all_success
            # Check circuit breaker before running step (P1)
            if not self._step_circuit_breaker.allow_request():
                time_until_retry = self._step_circuit_breaker.get_time_until_retry()
                self.log(
                    f"[CIRCUIT-BREAKER] Step {step.value} skipped - circuit breaker open. "
                    f"Retry in {time_until_retry:.1f}s after {self._step_circuit_breaker.config.failure_threshold} "
                    "consecutive failures.",
                    "WARNING",
                )
                self.results.append(
                    StepResult(
                        step=step,
                        success=False,
                        duration_seconds=0,
                        message=f"Skipped (circuit breaker open, retry in {time_until_retry:.1f}s)",
                        outputs={"circuit_breaker_open": True},
                    )
                )
                all_success = False
                return True

            prerequisite_result = self._check_step_prerequisites(
                step,
                dependencies,
                steps,
            )
            if prerequisite_result is not None:
                self.results.append(prerequisite_result)
                all_success = False
                self.log(f"Step {step.value} skipped: {prerequisite_result.message}", "ERROR")
                # Record as failure for circuit breaker
                self._step_circuit_breaker.record_failure(
                    Exception(f"Prerequisite check failed: {prerequisite_result.message}")
                )
                if self.fail_fast:
                    self.log("Fail-fast enabled; stopping pipeline.", "ERROR")
                    return False
                return True
            if resume_from is not None:
                expected_outputs = self._expected_output_paths(step)
                if step in forced_steps:
                    self.log(f"Force rerun requested for step {step.value}; skipping checkpoint.", "INFO")
                else:
                    should_skip_step = checkpoint_store.should_skip_step(
                        step.value,
                        expected_outputs=expected_outputs,
                        require_nonempty=True,
                        require_fresh_outputs=True,
                        validate_sidecar_metadata=True,
                    )
                    if step == PipelineStep.GENIESIM_IMPORT and should_skip_step:
                        expected_job_id, expected_run_id = self._resolve_geniesim_job_identity()
                        marker_path = self.geniesim_dir / ".geniesim_import_complete"
                        marker_ok = self._clear_stale_marker_if_mismatched(
                            marker_path,
                            "Genie Sim import completion marker",
                            expected_job_id=expected_job_id,
                            expected_run_id=expected_run_id,
                        )
                        if not marker_ok:
                            should_skip_step = False
                            self.log(
                                "Ignoring checkpoint resume for genie-sim-import because "
                                "completion marker did not match current job/run.",
                                "WARNING",
                            )
                    if should_skip_step:
                        checkpoint = checkpoint_store.load_checkpoint(step.value)
                        self.log(f"Skipping step {step.value} (checkpoint found)", "INFO")
                        checkpoint_outputs = dict(checkpoint.outputs) if checkpoint else {}
                        checkpoint_result = StepResult(
                            step=step,
                            success=True,
                            duration_seconds=0,
                            message="Skipped (checkpointed)",
                            outputs=checkpoint_outputs,
                        )
                        checkpoint_result = self._apply_quality_gates(
                            step,
                            checkpoint_result,
                            checkpointed=True,
                        )
                        self.results.append(checkpoint_result)
                        if not checkpoint_result.success:
                            all_success = False
                            self.log(f"Step {step.value} failed: {checkpoint_result.message}", "ERROR")
                            self._step_circuit_breaker.record_failure(
                                Exception(checkpoint_result.message or "Quality gate failed")
                            )
                            if self.fail_fast:
                                self.log("Fail-fast enabled; stopping pipeline.", "ERROR")
                                return False
                        else:
                            # Checkpointed step counts as success for circuit breaker
                            self._step_circuit_breaker.record_success()
                        return True

            started_at = datetime.utcnow().isoformat() + "Z"
            result = self._run_step(step)
            completed_at = datetime.utcnow().isoformat() + "Z"
            if result.success:
                result = self._apply_quality_gates(step, result)
            self.results.append(result)

            if not result.success:
                all_success = False
                self.log(f"Step {step.value} failed: {result.message}", "ERROR")
                # Record failure for circuit breaker (P1)
                self._step_circuit_breaker.record_failure(Exception(result.message or "Step failed"))
                # Continue with remaining steps for partial results
                if self.fail_fast:
                    self.log("Fail-fast enabled; stopping pipeline.", "ERROR")
                    return False
            else:
                # Record success for circuit breaker (P1)
                self._step_circuit_breaker.record_success()
                checkpoint_store.write_checkpoint(
                    step.value,
                    status="completed",
                    started_at=started_at,
                    completed_at=completed_at,
                    outputs=result.outputs,
                    output_paths=self._expected_output_paths(step),
                    store_output_hashes=self.enable_checkpoint_hashes,
                )
                # Upload step outputs to GCS if configured
                if self._gcs_sync and self._gcs_upload_outputs:
                    try:
                        self.log(f"[GCS] Uploading outputs for step {step.value}...")
                        sync_result = self._gcs_sync.upload_step_outputs(step.value)
                        self.log(
                            f"[GCS] Uploaded {sync_result.files_synced} files for {step.value}"
                        )
                        if sync_result.errors:
                            for err in sync_result.errors[:5]:
                                self.log(f"[GCS] Upload error: {err}", "WARNING")
                    except Exception as exc:
                        self.log(
                            f"[GCS] WARNING: GCS upload failed for {step.value}: {exc}",
                            "WARNING",
                        )
            if step == PipelineStep.REGEN3D and self._pending_articulation_preflight:
                if not self._preflight_articulation_requirements(steps):
                    return False
            return True

        for index, step in enumerate(steps):
            should_continue = _execute_step(step)
            if not should_continue:
                break
            result = self._find_step_result(step)
            if (
                auto_trigger_import
                and step == PipelineStep.GENIESIM_SUBMIT
                and result is not None
                and result.success
                and result.outputs.get("job_status") == "completed"
                and PipelineStep.GENIESIM_IMPORT not in steps[index + 1 :]
            ):
                should_continue = _execute_step(PipelineStep.GENIESIM_IMPORT)
                if not should_continue:
                    break

        # Write GCS completion/failure marker
        if self._gcs_sync and self._gcs_upload_outputs:
            total_time = sum(r.duration_seconds for r in self.results)
            passed = sum(1 for r in self.results if r.success)
            metadata = {
                "steps_passed": passed,
                "steps_total": len(self.results),
                "duration_seconds": round(total_time, 1),
                "steps": [
                    {"name": r.step.value, "success": r.success, "duration_s": round(r.duration_seconds, 1)}
                    for r in self.results
                ],
            }
            if self._gcs_sync.input_object:
                metadata["input_object"] = self._gcs_sync.input_object
            if self._gcs_sync.input_generation:
                metadata["input_generation"] = self._gcs_sync.input_generation
            try:
                if all_success:
                    self._gcs_sync.write_completion_marker(
                        ".reconstruction_complete", metadata=metadata,
                    )
                    self.log("[GCS] Wrote .reconstruction_complete marker")
                else:
                    failed_steps = [r.step.value for r in self.results if not r.success]
                    self._gcs_sync.write_failure_marker(
                        ".reconstruction_failed",
                        error_message=f"Pipeline steps failed: {', '.join(failed_steps)}",
                        error_code="pipeline_step_failure",
                        context=metadata,
                    )
                    self.log("[GCS] Wrote .reconstruction_failed marker")
            except Exception as exc:
                self.log(f"[GCS] WARNING: Failed to write marker: {exc}", "WARNING")

        # Print summary
        self._print_summary()

        return all_success

    def _steps_require_geniesim_preflight(self, steps: List[PipelineStep]) -> bool:
        """Return True if a Genie Sim step that needs the runtime is requested.

        genie-sim-export only generates JSON configs from the scene manifest and
        does not need Isaac Sim, Genie Sim, or a running server.
        """
        return any(
            step in {
                PipelineStep.GENIESIM_SUBMIT,
                PipelineStep.GENIESIM_IMPORT,
            }
            for step in steps
        )

    def _geniesim_requires_server(self, steps: List[PipelineStep]) -> bool:
        """Return True if Genie Sim server is required for requested steps."""
        if PipelineStep.GENIESIM_SUBMIT not in steps:
            return False
        mock_decision = resolve_geniesim_mock_mode()
        if mock_decision.requested and mock_decision.production_mode:
            raise NonRetryableError(
                "GENIESIM_MOCK_MODE cannot be used in production. "
                "Disable GENIESIM_MOCK_MODE in production to continue."
            )
        if mock_decision.enabled:
            return False
        try:
            from tools.geniesim_adapter.local_framework import GenieSimConfig
        except ImportError:
            return True
        config = GenieSimConfig.from_env()
        if config.host not in {"localhost", "127.0.0.1"}:
            return True
        return False

    def _run_geniesim_preflight(self, *, require_server: bool) -> bool:
        """Run Genie Sim preflight checks for local steps."""
        try:
            from tools.geniesim_adapter.local_framework import GenieSimConfig, run_geniesim_preflight
        except ImportError as e:
            self.log(f"ERROR: Genie Sim preflight dependencies not found: {e}", "ERROR")
            return False

        report = run_geniesim_preflight(
            "local-pipeline",
            require_server=require_server,
        )
        self._geniesim_preflight_report = report
        self.geniesim_dir.mkdir(parents=True, exist_ok=True)
        preflight_report_path = self.geniesim_dir / "preflight_report.json"
        _safe_write_text(
            preflight_report_path,
            json.dumps(report, indent=2),
            context="geniesim preflight report",
        )
        if not require_server and not report.get("status", {}).get("server_running", False):
            self.log(
                "Genie Sim server is not running; the local framework will start it automatically if needed.",
                "WARNING",
            )

        # Run Firebase connectivity preflight if Firebase upload is configured
        firebase_preflight_result = self._run_firebase_preflight()
        if firebase_preflight_result is not None:
            report["firebase_preflight"] = firebase_preflight_result
            _safe_write_text(
                preflight_report_path,
                json.dumps(report, indent=2),
                context="geniesim preflight report (with firebase)",
            )
            if not firebase_preflight_result.get("success", False):
                firebase_required = parse_bool_env(
                    os.getenv("FIREBASE_UPLOAD_REQUIRED"),
                    default=resolve_production_mode(),
                )
                if firebase_required:
                    self.log(
                        f"[FIREBASE-PREFLIGHT] Firebase connectivity check failed: "
                        f"{firebase_preflight_result.get('error', 'unknown error')}",
                        "ERROR",
                    )
                    return False
                else:
                    self.log(
                        f"[FIREBASE-PREFLIGHT] Firebase connectivity check failed (non-fatal): "
                        f"{firebase_preflight_result.get('error', 'unknown error')}",
                        "WARNING",
                    )

        if report.get("ok", False):
            return True

        config = GenieSimConfig.from_env()
        server_command = (
            f"{config.isaac_sim_path}/python.sh "
            f"{config.geniesim_root}/source/data_collection/scripts/data_collector_server.py "
            f"--headless --port {config.port}"
        )
        message = (
            "[GENIESIM-PREFLIGHT] Genie Sim prerequisites missing. "
            f"Set ISAAC_SIM_PATH (current: {config.isaac_sim_path}), "
            f"set GENIESIM_ROOT (current: {config.geniesim_root}), "
            "install grpcio (pip install grpcio), "
            f"and start the Genie Sim server: {server_command}"
        )
        self.log(message, "ERROR")
        return False

    def _run_firebase_preflight(self) -> Optional[Dict[str, Any]]:
        """
        Run Firebase connectivity preflight check.

        Returns None if Firebase upload is disabled, otherwise returns the
        connectivity check result dict.
        """
        # Check if Firebase upload is enabled
        firebase_enabled = parse_bool_env(os.getenv("ENABLE_FIREBASE_UPLOAD"), default=True)
        firebase_disabled = parse_bool_env(os.getenv("DISABLE_FIREBASE_UPLOAD"), default=False)

        if firebase_disabled or not firebase_enabled:
            self.log("[FIREBASE-PREFLIGHT] Firebase upload is disabled, skipping connectivity check.", "INFO")
            return None

        # Check if Firebase bucket is configured
        firebase_bucket = os.getenv("FIREBASE_STORAGE_BUCKET")
        if not firebase_bucket:
            self.log("[FIREBASE-PREFLIGHT] FIREBASE_STORAGE_BUCKET not configured, skipping connectivity check.", "INFO")
            return None

        self.log(f"[FIREBASE-PREFLIGHT] Testing Firebase connectivity to bucket: {firebase_bucket}", "INFO")

        try:
            from tools.firebase_upload import preflight_firebase_connectivity

            result = preflight_firebase_connectivity(timeout_seconds=15.0)
            if result.get("success"):
                self.log(
                    f"[FIREBASE-PREFLIGHT] Firebase connectivity verified "
                    f"(bucket={result.get('bucket_name')}, latency={result.get('latency_ms')}ms)",
                    "INFO",
                )
            return result
        except ImportError as e:
            return {
                "success": False,
                "error": f"Firebase upload module not available: {e}",
                "bucket_name": firebase_bucket,
            }
        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
                "bucket_name": firebase_bucket,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error during Firebase preflight: {e}",
                "bucket_name": firebase_bucket,
            }

    def _apply_labs_flags(self, run_validation: bool) -> None:
        """Apply production/labs flags for staging or lab validation runs."""
        staging_e2e = os.environ.get("RUN_STAGING_E2E") == "1"
        labs_validation = os.environ.get("LABS_VALIDATION", "").lower() in {"1", "true", "yes"}
        production_mode = resolve_production_mode()

        if not (staging_e2e or production_mode or (labs_validation and run_validation)):
            return

        os.environ.setdefault("PIPELINE_ENV", "production")
        os.environ.setdefault("DATA_QUALITY_LEVEL", "production")
        os.environ.setdefault("ISAAC_SIM_REQUIRED", "true")
        self.log("Applied labs/production flags for staging or lab validation runs")

    def _resolve_default_steps(self) -> List[PipelineStep]:
        """Resolve default steps, using pipeline selector for Genie Sim mode."""
        try:
            from tools.pipeline_selector.selector import PipelineSelector
        except ImportError:
            self.log("Pipeline selector not available; falling back to default steps", "WARNING")
            steps = self.DEFAULT_STEPS.copy()
            return self._apply_optional_steps(steps)

        selector = PipelineSelector(scene_root=self.scene_dir)
        decision = selector.select(self.scene_dir)
        steps = self._map_jobs_to_steps(decision.job_sequence)
        if PipelineStep.GENIESIM_SUBMIT in steps and PipelineStep.GENIESIM_IMPORT not in steps:
            submit_index = steps.index(PipelineStep.GENIESIM_SUBMIT)
            steps.insert(submit_index + 1, PipelineStep.GENIESIM_IMPORT)

        steps = self._apply_optional_steps(steps)

        return steps

    def _apply_optional_steps(self, steps: List[PipelineStep]) -> List[PipelineStep]:
        """Append optional steps gated by explicit flags."""
        if self.enable_inventory_enrichment:
            steps = self._inject_inventory_enrichment_step(steps)
        if self.enable_dataset_delivery:
            steps = self._inject_dataset_delivery_step(steps)
        if self.enable_dwm:
            steps.extend(self.DWM_STEPS)
        if self.enable_dream2flow:
            steps.extend(self.DREAM2FLOW_STEPS)
        return steps

    @staticmethod
    def _inject_inventory_enrichment_step(steps: List[PipelineStep]) -> List[PipelineStep]:
        if PipelineStep.INVENTORY_ENRICHMENT in steps:
            return steps
        if PipelineStep.REPLICATOR in steps:
            insert_at = steps.index(PipelineStep.REPLICATOR)
            steps.insert(insert_at, PipelineStep.INVENTORY_ENRICHMENT)
        else:
            steps.append(PipelineStep.INVENTORY_ENRICHMENT)
        return steps

    @staticmethod
    def _inject_dataset_delivery_step(steps: List[PipelineStep]) -> List[PipelineStep]:
        if PipelineStep.DATASET_DELIVERY in steps:
            return steps
        if PipelineStep.GENIESIM_IMPORT in steps:
            insert_at = steps.index(PipelineStep.GENIESIM_IMPORT) + 1
            steps.insert(insert_at, PipelineStep.DATASET_DELIVERY)
        else:
            steps.append(PipelineStep.DATASET_DELIVERY)
        return steps

    def _map_jobs_to_steps(self, job_sequence: List[str]) -> List[PipelineStep]:
        """Map pipeline selector job names to local runner steps."""
        mapping = {
            "regen3d-reconstruct-job": PipelineStep.REGEN3D_RECONSTRUCT,
            "regen3d-job": PipelineStep.REGEN3D,
            "scale-job": PipelineStep.SCALE,
            "interactive-job": PipelineStep.INTERACTIVE,
            "simready-job": PipelineStep.SIMREADY,
            "usd-assembly-job": PipelineStep.USD,
            "replicator-job": PipelineStep.REPLICATOR,
            "variation-gen-job": PipelineStep.VARIATION_GEN,
            "isaac-lab-job": PipelineStep.ISAAC_LAB,
            "genie-sim-export-job": PipelineStep.GENIESIM_EXPORT,
            "genie-sim-submit-job": PipelineStep.GENIESIM_SUBMIT,
            "genie-sim-import-job": PipelineStep.GENIESIM_IMPORT,
            "dataset-delivery-job": PipelineStep.DATASET_DELIVERY,
        }
        steps: List[PipelineStep] = []
        for job_name in job_sequence:
            step = mapping.get(job_name)
            if step is None:
                self.log(f"Skipping unsupported local job: {job_name}", "WARNING")
                continue
            steps.append(step)
        return steps

    def _preflight_articulation_requirements(self, steps: List[PipelineStep]) -> bool:
        """Validate articulation requirements before running steps."""
        manifest_path = self.assets_dir / "scene_manifest.json"
        if not manifest_path.is_file():
            if PipelineStep.REGEN3D in steps:
                self._pending_articulation_preflight = True
                return True
            return True

        try:
            manifest = _load_json(manifest_path, "articulation preflight manifest")
        except NonRetryableError as exc:
            self.log(str(exc), "ERROR")
            return False
        required_ids = self._required_articulation_ids(manifest)
        if not required_ids:
            self._pending_articulation_preflight = False
            return True

        if self.disable_articulated_assets:
            self._disable_articulations_in_manifest(manifest, manifest_path, required_ids)
            self.log(
                "Articulated assets disabled; proceeding without interactive processing.",
                "WARNING",
            )
            self._pending_articulation_preflight = False
            return True

        endpoint = os.getenv("PARTICULATE_ENDPOINT", "").strip()
        particulate_mode = (os.getenv("PARTICULATE_MODE", "remote") or "remote").strip().lower()
        local_endpoint = (os.getenv("PARTICULATE_LOCAL_ENDPOINT", "") or "").strip()
        articulation_backend = (os.getenv("ARTICULATION_BACKEND", "particulate") or "particulate").strip().lower()
        interactive_requested = PipelineStep.INTERACTIVE in steps

        if self.skip_interactive or not interactive_requested:
            self.log(
                "ERROR: Articulated assets detected but interactive processing is not enabled.",
                "ERROR",
            )
            self.log(
                "Enable articulation by running with --with-interactive and setting PARTICULATE_ENDPOINT, "
                "or explicitly disable articulated assets with DISABLE_ARTICULATED_ASSETS=true.",
                "ERROR",
            )
            self.log(f"Articulated object IDs: {required_ids}", "ERROR")
            return False

        if articulation_backend not in {"particulate", "auto"}:
            self.log(
                "ERROR: Required articulated assets require Particulate backend. "
                f"Current ARTICULATION_BACKEND={articulation_backend}.",
                "ERROR",
            )
            self.log(f"Articulated object IDs: {required_ids}", "ERROR")
            return False

        if particulate_mode in {"mock", "skip"}:
            self.log(
                "ERROR: Required articulated assets cannot run with "
                f"PARTICULATE_MODE={particulate_mode}.",
                "ERROR",
            )
            self.log(f"Articulated object IDs: {required_ids}", "ERROR")
            return False

        if particulate_mode == "local":
            resolved_local_endpoint = local_endpoint or "http://localhost:8080"
            if not resolved_local_endpoint:
                self.log(
                    "ERROR: PARTICULATE_MODE=local set but no local endpoint available.",
                    "ERROR",
                )
                self.log(f"Articulated object IDs: {required_ids}", "ERROR")
                return False
        elif not endpoint:
            self.log(
                "ERROR: Articulated assets detected but PARTICULATE_ENDPOINT is not set.",
                "ERROR",
            )
            self.log(
                "Set PARTICULATE_ENDPOINT (remote mode) or use PARTICULATE_MODE=local "
                "with PARTICULATE_LOCAL_ENDPOINT, or set DISABLE_ARTICULATED_ASSETS=true "
                "to proceed without articulated assets.",
                "ERROR",
            )
            self.log(f"Articulated object IDs: {required_ids}", "ERROR")
            return False

        self._pending_articulation_preflight = False
        return True

    def _disable_articulations_in_manifest(
        self,
        manifest: Dict[str, Any],
        manifest_path: Path,
        required_ids: List[str],
    ) -> None:
        """Disable articulation requirements in the manifest."""
        disabled_ids = set(required_ids)
        for obj in manifest.get("objects", []):
            if obj.get("id") not in disabled_ids:
                continue
            articulation = obj.get("articulation") or {}
            articulation["required"] = False
            articulation["disabled"] = True
            obj["articulation"] = articulation
            if obj.get("sim_role") in {"articulated_furniture", "articulated_appliance"}:
                obj["sim_role"] = "static"

        metadata = manifest.get("metadata") or {}
        metadata["articulation_disabled"] = {
            "disabled_count": len(disabled_ids),
            "disabled_objects": sorted(disabled_ids),
            "disabled_at": datetime.utcnow().isoformat() + "Z",
            "reason": "DISABLE_ARTICULATED_ASSETS=true",
        }
        manifest["metadata"] = metadata
        _safe_write_text(
            manifest_path,
            json.dumps(manifest, indent=2),
            context="articulation-disabled manifest",
        )

    def _run_step(self, step: PipelineStep) -> StepResult:
        """Run a single pipeline step."""
        import time
        start_time = time.time()

        self._current_step = step
        self.log(f"\n--- Running step: {step.value} ---")

        try:
            if step == PipelineStep.REGEN3D_RECONSTRUCT:
                result = self._run_with_timeout(
                    PipelineStep.REGEN3D_RECONSTRUCT,
                    self._run_regen3d_reconstruct,
                )
            elif step == PipelineStep.REGEN3D:
                result = self._run_with_timeout(PipelineStep.REGEN3D, self._run_regen3d_adapter)
            elif step == PipelineStep.SCALE:
                result = self._run_with_retry(PipelineStep.SCALE, self._run_scale)
            elif step == PipelineStep.INTERACTIVE:
                result = self._run_with_retry(PipelineStep.INTERACTIVE, self._run_interactive)
            elif step == PipelineStep.SIMREADY:
                result = self._run_with_timeout(PipelineStep.SIMREADY, self._run_simready)
            elif step == PipelineStep.USD:
                result = self._run_with_timeout(PipelineStep.USD, self._run_usd_assembly)
            elif step == PipelineStep.INVENTORY_ENRICHMENT:
                result = self._run_with_timeout(
                    PipelineStep.INVENTORY_ENRICHMENT,
                    self._run_inventory_enrichment,
                )
            elif step == PipelineStep.REPLICATOR:
                result = self._run_with_timeout(PipelineStep.REPLICATOR, self._run_replicator)
            elif step == PipelineStep.VARIATION_GEN:
                result = self._run_with_timeout(PipelineStep.VARIATION_GEN, self._run_variation_gen)
            elif step == PipelineStep.ISAAC_LAB:
                result = self._run_with_timeout(PipelineStep.ISAAC_LAB, self._run_isaac_lab)
            elif step == PipelineStep.GENIESIM_EXPORT:
                result = self._run_with_timeout(
                    PipelineStep.GENIESIM_EXPORT,
                    self._run_geniesim_export,
                )
            elif step == PipelineStep.GENIESIM_SUBMIT:
                result = self._run_with_timeout(
                    PipelineStep.GENIESIM_SUBMIT,
                    self._run_geniesim_submit,
                )
            elif step == PipelineStep.GENIESIM_IMPORT:
                result = self._run_with_timeout(
                    PipelineStep.GENIESIM_IMPORT,
                    self._run_geniesim_import,
                )
            elif step == PipelineStep.DATASET_DELIVERY:
                result = self._run_with_timeout(
                    PipelineStep.DATASET_DELIVERY,
                    self._run_dataset_delivery,
                )
            elif step == PipelineStep.DWM:
                result = self._run_with_retry(PipelineStep.DWM, self._run_dwm)
            elif step == PipelineStep.DWM_INFERENCE:
                result = self._run_with_timeout(
                    PipelineStep.DWM_INFERENCE,
                    self._run_dwm_inference,
                )
            elif step == PipelineStep.DREAM2FLOW:
                result = self._run_with_retry(PipelineStep.DREAM2FLOW, self._run_dream2flow)
            elif step == PipelineStep.DREAM2FLOW_INFERENCE:
                result = self._run_with_timeout(
                    PipelineStep.DREAM2FLOW_INFERENCE,
                    self._run_dream2flow_inference,
                )
            elif step == PipelineStep.VALIDATE:
                result = self._run_with_timeout(PipelineStep.VALIDATE, self._run_validation)
            else:
                result = StepResult(
                    step=step,
                    success=False,
                    duration_seconds=0,
                    message=f"Unknown step: {step.value}",
                )
        except RetryableError as e:
            result = self._handle_retryable_exception(
                step,
                e,
                f"Step {step.value} failed after retries",
                duration_seconds=time.time() - start_time,
            )
        except NonRetryableError as e:
            result = StepResult(
                step=step,
                success=False,
                duration_seconds=time.time() - start_time,
                message=str(e),
            )
        except Exception as e:
            self._log_exception_traceback(f"Step {step.value} failed", e)
            result = StepResult(
                step=step,
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Exception: {self._summarize_exception(e)}",
            )

        result.duration_seconds = time.time() - start_time
        status = "OK" if result.success else "FAILED"
        self.log(f"Step {step.value}: {status} ({result.duration_seconds:.2f}s)")
        self._current_step = None

        return result

    def _find_input_image(self) -> Optional[Path]:
        """Find the best source image for regen3d reconstruction."""
        supported_exts = {".png", ".jpg", ".jpeg"}
        input_dir = self.scene_dir / "input"
        candidates: List[Path] = []

        if input_dir.is_dir():
            candidates.extend(
                path for path in input_dir.iterdir()
                if path.is_file() and path.suffix.lower() in supported_exts
            )

        # Backwards compatibility: allow source image in scene root.
        if not candidates and self.scene_dir.is_dir():
            candidates.extend(
                path for path in self.scene_dir.iterdir()
                if path.is_file() and path.suffix.lower() in supported_exts
            )

        if not candidates:
            return None

        preferred_name = None
        if self._gcs_sync and self._gcs_sync.input_object:
            preferred_name = Path(self._gcs_sync.input_object).name
        if preferred_name:
            for path in candidates:
                if path.name == preferred_name:
                    return path

        if len(candidates) > 1:
            names = ", ".join(path.name for path in sorted(candidates)[:5])
            self.log(
                f"Multiple input images found; choosing newest by mtime. Candidates: {names}",
                "WARNING",
            )

        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0]

    def _run_regen3d_reconstruct(self) -> StepResult:
        """Run 3D-RE-GEN reconstruction on a remote GPU VM.

        Takes a source image from scene_dir/input/ and runs the full
        3D-RE-GEN pipeline (arXiv:2512.17459) to produce 3D assets in
        the regen3d/ directory.
        """
        from tools.regen3d_runner import Regen3DRunner, Regen3DConfig

        # Find input image.
        input_dir = self.scene_dir / "input"
        input_image = self._find_input_image()

        if input_image is None:
            return StepResult(
                step=PipelineStep.REGEN3D_RECONSTRUCT,
                success=False,
                duration_seconds=0,
                message=(
                    f"No input image found. Place an image at "
                    f"{input_dir}/<any_name>.png|jpg|jpeg"
                ),
            )

        self.log(f"Input image: {input_image}")

        # Load reconstruction defaults from configs/regen3d_reconstruct.env
        regen3d_env_path = Path(__file__).resolve().parents[1] / "configs" / "regen3d_reconstruct.env"
        if regen3d_env_path.is_file():
            load_dotenv(dotenv_path=regen3d_env_path, override=False)
            self.log(f"Loaded REGEN3D env defaults from {regen3d_env_path}")
        else:
            self.log(
                f"REGEN3D env defaults not found at {regen3d_env_path}; using current process env",
                "WARNING",
            )

        config = Regen3DConfig.from_env()
        self.log(
            "REGEN3D config: "
            f"vm={config.vm_host} zone={config.vm_zone} repo={config.repo_path} "
            f"steps={config.steps} timeout_s={config.timeout_s} device={config.device}"
        )
        runner = Regen3DRunner(config=config, verbose=self.verbose)

        result = runner.run_reconstruction(
            input_image=input_image,
            scene_id=self.scene_id,
            output_dir=self.regen3d_dir,
            environment_type=self.environment_type,
        )

        if not result.success:
            return StepResult(
                step=PipelineStep.REGEN3D_RECONSTRUCT,
                success=False,
                duration_seconds=result.duration_seconds,
                message=f"3D-RE-GEN reconstruction failed: {result.error}",
            )

        self.log(
            f"3D-RE-GEN reconstruction complete: "
            f"{result.objects_count} objects in {result.duration_seconds:.1f}s"
        )

        return StepResult(
            step=PipelineStep.REGEN3D_RECONSTRUCT,
            success=True,
            duration_seconds=result.duration_seconds,
            message=f"3D-RE-GEN reconstruction: {result.objects_count} objects",
            outputs={
                "objects_count": result.objects_count,
                "output_dir": str(result.output_dir),
                "input_image": str(input_image),
            },
        )

    def _run_regen3d_adapter(self) -> StepResult:
        """Run the 3D-RE-GEN adapter."""
        from tools.regen3d_adapter import Regen3DAdapter

        adapter = Regen3DAdapter(verbose=self.verbose)

        # Load 3D-RE-GEN outputs
        regen3d_output = adapter.load_regen3d_output(self.regen3d_dir)
        self.log(f"Loaded {len(regen3d_output.objects)} objects from 3D-RE-GEN")

        # Generate manifest
        manifest = adapter.create_manifest(
            regen3d_output,
            scene_id=self.scene_id,
            environment_type=self.environment_type,
        )
        manifest_path = self.assets_dir / "scene_manifest.json"
        _safe_write_text(
            manifest_path,
            json.dumps(manifest, indent=2),
            context="scene manifest",
        )
        manifest = self._annotate_articulation_requirements(manifest, manifest_path)
        self.log(f"Wrote manifest: {manifest_path}")

        # Generate layout
        layout = adapter.create_layout(regen3d_output, apply_scale_factor=1.0)
        layout_path = self.layout_dir / "scene_layout_scaled.json"
        _safe_write_text(
            layout_path,
            json.dumps(layout, indent=2),
            context="scene layout",
        )
        self.log(f"Wrote layout: {layout_path}")

        # Copy assets
        asset_paths = adapter.copy_assets(
            regen3d_output,
            self.scene_dir,
            assets_prefix="assets",
        )
        self.log(f"Copied {len(asset_paths)} assets")

        # Generate inventory
        inventory = self._generate_inventory(regen3d_output, manifest=manifest)
        inventory_path = self.seg_dir / "inventory.json"
        _safe_write_text(
            inventory_path,
            json.dumps(inventory, indent=2),
            context="inventory",
        )
        self.log(f"Wrote inventory: {inventory_path}")

        quality_ok, quality_message, quality_details = self._validate_stage1_quality(
            regen3d_output=regen3d_output,
            manifest=manifest,
            layout=layout,
        )
        if self._should_skip_quality_gates():
            # Stage 1 quality is a gate; allow local/test workflows to proceed when
            # SKIP_QUALITY_GATES is set (never allowed in production).
            quality_details = dict(quality_details)
            quality_details["skipped"] = True
            quality_details["skip_reason"] = "SKIP_QUALITY_GATES environment override"
            quality_ok = True
            quality_message = "Stage 1 quality gate skipped (SKIP_QUALITY_GATES)"
        quality_report_path = self._write_stage1_quality_report(
            quality_ok=quality_ok,
            quality_message=quality_message,
            quality_details=quality_details,
        )
        if not quality_ok:
            failure_outputs = dict(quality_details)
            failure_outputs["stage1_quality_report"] = str(quality_report_path)
            return StepResult(
                step=PipelineStep.REGEN3D,
                success=False,
                duration_seconds=0,
                message=quality_message,
                outputs=failure_outputs,
            )

        # Write completion marker
        marker_path = self.assets_dir / ".regen3d_complete"
        marker_content = {
            "status": "complete",
            "scene_id": self.scene_id,
            "objects_count": len(regen3d_output.objects),
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }
        _safe_write_text(
            marker_path,
            json.dumps(marker_content, indent=2),
            context="regen3d completion marker",
        )

        return StepResult(
            step=PipelineStep.REGEN3D,
            success=True,
            duration_seconds=0,
            message="3D-RE-GEN adapter completed",
            outputs={
                "manifest": str(manifest_path),
                "layout": str(layout_path),
                "objects_count": len(regen3d_output.objects),
                "stage1_quality_report": str(quality_report_path),
            },
        )

    def _validate_stage1_quality(
        self,
        *,
        regen3d_output: Any,
        manifest: Dict[str, Any],
        layout: Dict[str, Any],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate Stage 1 reconstruction outputs before emitting completion marker."""
        issues: List[str] = []
        details: Dict[str, Any] = {}

        foreground_count = len(getattr(regen3d_output, "objects", []) or [])
        has_background = getattr(regen3d_output, "background", None) is not None
        manifest_objects = manifest.get("objects") if isinstance(manifest, dict) else None
        layout_objects = layout.get("objects") if isinstance(layout, dict) else None

        details["foreground_object_count"] = foreground_count
        details["has_background"] = has_background
        details["manifest_object_count"] = (
            len(manifest_objects) if isinstance(manifest_objects, list) else 0
        )
        details["layout_object_count"] = (
            len(layout_objects) if isinstance(layout_objects, list) else 0
        )
        quality_mode = os.getenv("REGEN3D_QUALITY_MODE", "quality").strip().lower()
        if quality_mode not in {"quality", "compat"}:
            quality_mode = "quality"
        allow_textureless = parse_bool_env(
            os.getenv("REGEN3D_ALLOW_TEXTURELESS"),
            default=False,
        )
        allow_materialless_requested = parse_bool_env(
            os.getenv("REGEN3D_ALLOW_MATERIALLESS"),
            default=False,
        )
        allow_materialless = bool(
            allow_materialless_requested and not self._is_production_mode()
        )
        details["quality_mode"] = quality_mode
        details["allow_textureless_override"] = allow_textureless
        details["allow_materialless_requested"] = bool(allow_materialless_requested)
        details["allow_materialless_override"] = allow_materialless
        if allow_materialless_requested and self._is_production_mode():
            issues.append("REGEN3D_ALLOW_MATERIALLESS is not allowed in production mode.")

        max_objects_quality = parse_int_env(
            os.getenv("REGEN3D_MAX_OBJECTS_QUALITY"),
            default=18,
            min_value=1,
            name="REGEN3D_MAX_OBJECTS_QUALITY",
        )
        max_objects_compat = parse_int_env(
            os.getenv("REGEN3D_MAX_OBJECTS_COMPAT"),
            default=40,
            min_value=1,
            name="REGEN3D_MAX_OBJECTS_COMPAT",
        )
        max_instances_quality = parse_int_env(
            os.getenv("REGEN3D_MAX_INSTANCES_PER_CLASS_QUALITY"),
            default=4,
            min_value=1,
            name="REGEN3D_MAX_INSTANCES_PER_CLASS_QUALITY",
        )
        max_instances_compat = parse_int_env(
            os.getenv("REGEN3D_MAX_INSTANCES_PER_CLASS_COMPAT"),
            default=10,
            min_value=1,
            name="REGEN3D_MAX_INSTANCES_PER_CLASS_COMPAT",
        )
        max_masks_quality = parse_int_env(
            os.getenv("REGEN3D_MAX_MASKS"),
            default=40,
            min_value=1,
            name="REGEN3D_MAX_MASKS",
        )
        min_mask_ratio_quality = parse_float_env(
            os.getenv("REGEN3D_MIN_MASK_AREA_RATIO_QUALITY"),
            default=0.0002,
            min_value=0.0,
            max_value=1.0,
            name="REGEN3D_MIN_MASK_AREA_RATIO_QUALITY",
        )
        max_tiny_masks_quality = parse_int_env(
            os.getenv("REGEN3D_MAX_TINY_MASKS_QUALITY"),
            default=2,
            min_value=0,
            name="REGEN3D_MAX_TINY_MASKS_QUALITY",
        )
        details["max_object_limit"] = (
            max_objects_quality if quality_mode == "quality" else max_objects_compat
        )
        details["max_instances_per_class_limit"] = (
            max_instances_quality if quality_mode == "quality" else max_instances_compat
        )
        details["max_mask_limit"] = max_masks_quality
        details["min_mask_area_ratio_quality"] = min_mask_ratio_quality
        details["max_tiny_masks_quality"] = max_tiny_masks_quality

        if foreground_count <= 0:
            issues.append("No reconstructed foreground objects were produced.")
        if not has_background:
            issues.append("Background mesh is missing.")
        if not isinstance(manifest_objects, list) or not manifest_objects:
            issues.append("scene_manifest.json has no objects.")
        if not isinstance(layout_objects, list) or not layout_objects:
            issues.append("scene_layout_scaled.json has no objects.")
        max_object_limit = (
            max_objects_quality if quality_mode == "quality" else max_objects_compat
        )
        if foreground_count > max_object_limit:
            issues.append(
                "Foreground object count exceeds Stage 1 cap: "
                f"{foreground_count} > {max_object_limit}."
            )

        def _class_name_for_object(obj: Any) -> str:
            raw_id = str(getattr(obj, "id", "") or "").strip()
            if not raw_id:
                mesh_path = str(getattr(obj, "mesh_path", "") or "").strip()
                raw_id = Path(mesh_path).stem if mesh_path else "unknown"
            return re.sub(r"__\(\d+,\s*\d+\)$", "", raw_id)

        class_counts = Counter(
            _class_name_for_object(obj)
            for obj in (getattr(regen3d_output, "objects", []) or [])
        )
        details["foreground_class_counts"] = dict(class_counts)
        max_instances_limit = (
            max_instances_quality
            if quality_mode == "quality"
            else max_instances_compat
        )
        repeat_offenders = {
            label: count
            for label, count in class_counts.items()
            if count > max_instances_limit
        }
        details["foreground_class_repeat_offenders"] = repeat_offenders
        if repeat_offenders:
            joined = ", ".join(
                f"{label}={count}" for label, count in sorted(repeat_offenders.items())
            )
            issues.append(
                "Foreground class repetition exceeds Stage 1 cap: "
                f"{joined} (limit={max_instances_limit})."
            )

        mesh_stats: Dict[str, Any] = {
            "checked": 0,
            "with_materials": 0,
            "with_textures": 0,
            "parse_errors": [],
        }
        for obj in (getattr(regen3d_output, "objects", []) or []):
            mesh_path_value = getattr(obj, "mesh_path", "")
            if not mesh_path_value:
                continue
            mesh_path = Path(mesh_path_value)
            info = self._inspect_glb_metadata(mesh_path)
            if info["exists"]:
                mesh_stats["checked"] += 1
            if info["parse_error"]:
                mesh_stats["parse_errors"].append(
                    f"{mesh_path.name}: {info['parse_error']}"
                )
                continue
            if info["materials"] > 0:
                mesh_stats["with_materials"] += 1
            if info["textures"] > 0:
                mesh_stats["with_textures"] += 1
            if info["meshes"] <= 0:
                mesh_stats["parse_errors"].append(
                    f"{mesh_path.name}: no mesh primitives found"
                )

        details["mesh_stats"] = mesh_stats
        if mesh_stats["checked"] <= 0:
            issues.append("No readable object GLB assets were found.")
        elif mesh_stats["with_materials"] <= 0 and not allow_materialless:
            issues.append("Object GLBs contain no material definitions.")
        elif (
            quality_mode == "quality"
            and not allow_textureless
            and mesh_stats["with_textures"] <= 0
        ):
            issues.append(
                "Object GLBs contain no texture definitions "
                "(quality mode requires textures unless "
                "REGEN3D_ALLOW_TEXTURELESS=true)."
            )

        combined_scene_path = self.regen3d_dir / "glb" / "scene" / "combined_scene.glb"
        combined_scene_info = self._inspect_glb_metadata(combined_scene_path)
        details["combined_scene"] = {
            "path": str(combined_scene_path),
            "exists": bool(combined_scene_info.get("exists")),
            "meshes": int(combined_scene_info.get("meshes", 0)),
            "materials": int(combined_scene_info.get("materials", 0)),
            "textures": int(combined_scene_info.get("textures", 0)),
            "parse_error": combined_scene_info.get("parse_error"),
        }
        if not combined_scene_info["exists"]:
            issues.append(
                "Combined scene GLB is missing at regen3d/glb/scene/combined_scene.glb."
            )
        elif combined_scene_info["parse_error"]:
            issues.append(
                "Combined scene GLB is unreadable: "
                f"{combined_scene_info['parse_error']}."
            )
        elif combined_scene_info["meshes"] <= 0:
            issues.append("Combined scene GLB contains no mesh primitives.")
        elif combined_scene_info["materials"] <= 0 and not allow_materialless:
            issues.append("Combined scene GLB contains no material definitions.")
        elif (
            quality_mode == "quality"
            and not allow_textureless
            and combined_scene_info["textures"] <= 0
        ):
            issues.append(
                "Combined scene GLB contains no texture definitions "
                "(quality mode requires textures unless "
                "REGEN3D_ALLOW_TEXTURELESS=true)."
            )

        mask_stats = self._inspect_mask_quality(
            self.regen3d_dir / "masks",
            tiny_ratio_threshold=float(min_mask_ratio_quality),
        )
        details["mask_stats"] = mask_stats
        if mask_stats.get("file_count", 0) > max_masks_quality:
            issues.append(
                "Mask count exceeds Stage 1 cap: "
                f"{mask_stats.get('file_count', 0)} > {max_masks_quality}."
            )
        if mask_stats.get("zero_area", 0) > 0:
            issues.append(
                f"Detected {mask_stats.get('zero_area')} zero-area mask(s)."
            )
        if (
            quality_mode == "quality"
            and mask_stats.get("tiny_area", 0) > max_tiny_masks_quality
        ):
            issues.append(
                "Detected too many tiny masks for quality mode: "
                f"{mask_stats.get('tiny_area')} > {max_tiny_masks_quality} "
                f"(threshold={min_mask_ratio_quality:.6f})."
            )

        details["issues"] = list(issues)

        if issues:
            return (
                False,
                "Stage 1 quality gate failed: " + " ".join(issues),
                details,
            )
        return True, "Stage 1 quality gate passed", details

    def _write_stage1_quality_report(
        self,
        *,
        quality_ok: bool,
        quality_message: str,
        quality_details: Dict[str, Any],
    ) -> Path:
        """Write a machine-readable Stage 1 quality report."""
        report_path = self.assets_dir / "stage1_quality_report.json"
        report_payload = {
            "scene_id": self.scene_id,
            "status": "pass" if quality_ok else "fail",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "message": quality_message,
            "details": quality_details,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_write_text(
            report_path,
            json.dumps(report_payload, indent=2),
            context="stage1 quality report",
        )
        return report_path

    def _inspect_mask_quality(
        self,
        mask_dir: Path,
        *,
        tiny_ratio_threshold: float,
    ) -> Dict[str, Any]:
        """Inspect binary masks for zero-area and tiny-fragment issues."""
        stats: Dict[str, Any] = {
            "path": str(mask_dir),
            "available": mask_dir.is_dir(),
            "file_count": 0,
            "checked": 0,
            "zero_area": 0,
            "tiny_area": 0,
            "tiny_ratio_threshold": tiny_ratio_threshold,
            "zero_area_samples": [],
            "tiny_area_samples": [],
            "read_errors": [],
            "pillow_available": True,
        }
        if not mask_dir.is_dir():
            return stats

        mask_paths = sorted(mask_dir.glob("*.png"))
        stats["file_count"] = len(mask_paths)
        if not mask_paths:
            return stats

        try:
            from PIL import Image
        except Exception:
            stats["pillow_available"] = False
            return stats

        for path in mask_paths:
            try:
                arr = np.array(Image.open(path).convert("L"))
            except Exception as exc:
                stats["read_errors"].append(f"{path.name}: {exc}")
                continue
            area = int(np.count_nonzero(arr))
            total = int(arr.size)
            ratio = (area / total) if total else 0.0
            stats["checked"] += 1
            if area == 0:
                stats["zero_area"] += 1
                if len(stats["zero_area_samples"]) < 10:
                    stats["zero_area_samples"].append(path.name)
                continue
            if ratio < tiny_ratio_threshold:
                stats["tiny_area"] += 1
                if len(stats["tiny_area_samples"]) < 10:
                    stats["tiny_area_samples"].append(path.name)
        return stats

    def _inspect_glb_metadata(self, glb_path: Path) -> Dict[str, Any]:
        """Inspect high-level GLB metadata required by Stage 1 quality gating."""
        info: Dict[str, Any] = {
            "exists": glb_path.is_file(),
            "meshes": 0,
            "materials": 0,
            "textures": 0,
            "parse_error": None,
        }
        if not info["exists"]:
            info["parse_error"] = "missing file"
            return info
        try:
            data = glb_path.read_bytes()
            if data[:4] != b"glTF":
                info["parse_error"] = "not a GLB file"
                return info
            if len(data) < 20:
                info["parse_error"] = "truncated GLB header"
                return info
            json_chunk_len, _ = struct.unpack_from("<II", data, 12)
            json_start = 20
            json_end = json_start + json_chunk_len
            gltf_json = json.loads(data[json_start:json_end].decode("utf-8"))
            info["meshes"] = len(gltf_json.get("meshes", []) or [])
            info["materials"] = len(gltf_json.get("materials", []) or [])
            info["textures"] = len(gltf_json.get("textures", []) or [])
        except Exception as exc:
            info["parse_error"] = str(exc)
        return info

    def _annotate_articulation_requirements(
        self,
        manifest: Dict[str, Any],
        manifest_path: Path,
    ) -> Dict[str, Any]:
        """Detect articulated objects and annotate the manifest."""
        from tools.articulation import (
            detect_scene_articulations,
            infer_primary_joint_type,
            parse_label_articulation_hint,
        )

        results = detect_scene_articulations(
            manifest,
            use_llm=False,
            verbose=self.verbose,
        )

        required_ids: List[str] = []
        required_from_hints: List[str] = []
        required_from_detector: List[str] = []
        for obj in manifest.get("objects", []):
            obj_id = obj.get("id")
            if not obj_id or obj.get("sim_role") in {"background", "scene_shell"}:
                continue

            articulation = obj.get("articulation") or {}
            category_text = (
                obj.get("category")
                or (obj.get("semantics") or {}).get("class")
                or obj.get("name")
                or obj_id
            )
            hint_payload = parse_label_articulation_hint(str(category_text))

            if hint_payload.get("is_articulated"):
                hints = articulation.get("hints")
                if not isinstance(hints, list):
                    hints = []
                hint_entry = {
                    "source": "label_hint",
                    "text": str(category_text),
                    "confidence": float(hint_payload.get("confidence", 0.0)),
                    "joint_types": hint_payload.get("joint_types", []),
                    "parts": hint_payload.get("parts", []),
                    "matched_keywords": hint_payload.get("matched_keywords", []),
                }
                hints.append(hint_entry)
                articulation["hints"] = hints

                primary_joint_type = infer_primary_joint_type(hint_payload)
                if primary_joint_type and not articulation.get("type"):
                    articulation["type"] = primary_joint_type

                if float(hint_payload.get("confidence", 0.0)) >= 0.55:
                    required_ids.append(obj_id)
                    required_from_hints.append(obj_id)
                    articulation["required"] = True
                    articulation["required_reason"] = (
                        f"label_hint:{','.join(hint_payload.get('parts', []))}"
                    )
                    articulation["required_source"] = "label_hint"
                    articulation["detection"] = {
                        "type": primary_joint_type or articulation.get("type"),
                        "confidence": float(hint_payload.get("confidence", 0.0)),
                        "method": "label_hint",
                    }
                    obj["articulation"] = articulation
                    if obj.get("sim_role") in {"unknown", "static", None, ""}:
                        obj["sim_role"] = self._infer_articulation_role(obj)
                    continue

            result = results.get(obj_id)
            if result and result.has_articulation:
                required_ids.append(obj_id)
                required_from_detector.append(obj_id)
                detection_payload = {
                    "type": result.articulation_type.value,
                    "confidence": result.confidence,
                    "method": result.detection_method,
                }
                if result.joint_axis is not None:
                    detection_payload["axis"] = [float(v) for v in result.joint_axis.tolist()]
                if result.joint_range is not None:
                    detection_payload["range"] = [float(result.joint_range[0]), float(result.joint_range[1])]

                articulation.update({
                    "required": True,
                    "required_reason": "detector_match",
                    "required_source": "detector",
                    "detection": detection_payload,
                })
                articulation.setdefault("type", result.articulation_type.value)
                obj["articulation"] = articulation
                if obj.get("sim_role") in {"unknown", "static", None, ""}:
                    obj["sim_role"] = self._infer_articulation_role(obj)
                continue

            obj["articulation"] = articulation

        metadata = manifest.get("metadata") or {}
        required_ids = sorted(set(required_ids))
        metadata["articulation_detection"] = {
            "required_count": len(required_ids),
            "required_objects": required_ids,
            "required_from_label_hints": sorted(set(required_from_hints)),
            "required_from_detector": sorted(set(required_from_detector)),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source": "label_hint+heuristic",
        }
        manifest["metadata"] = metadata
        _safe_write_text(
            manifest_path,
            json.dumps(manifest, indent=2),
            context="articulation-annotated manifest",
        )

        if required_ids:
            self.log(f"Detected {len(required_ids)} articulated objects requiring interactive processing")

        return manifest

    def _infer_articulation_role(self, obj: Dict[str, Any]) -> str:
        """Infer an articulated sim_role based on object category."""
        category = (obj.get("category")
                    or (obj.get("semantics") or {}).get("category")
                    or obj.get("id", ""))
        category = category.lower()

        appliance_keywords = {
            "refrigerator",
            "fridge",
            "oven",
            "microwave",
            "dishwasher",
            "washing_machine",
            "washer",
            "dryer",
            "stove",
            "range",
        }
        if any(keyword in category for keyword in appliance_keywords):
            return "articulated_appliance"
        return "articulated_furniture"

    def _required_articulation_ids(self, manifest: Dict[str, Any]) -> List[str]:
        """Return object IDs that require articulation."""
        required = []
        for obj in manifest.get("objects", []):
            obj_id = obj.get("id")
            if not obj_id:
                continue
            if obj.get("sim_role") in {"articulated_furniture", "articulated_appliance"}:
                required.append(obj_id)
                continue
            articulation = obj.get("articulation") or {}
            if articulation.get("required"):
                required.append(obj_id)
        return required

    def _generate_inventory(
        self,
        regen3d_output,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate semantic inventory from 3D-RE-GEN output."""
        manifest_by_id: Dict[str, Dict[str, Any]] = {}
        if manifest:
            manifest_by_id = {
                str(obj.get("id")): obj
                for obj in manifest.get("objects", [])
                if obj.get("id") is not None
            }

        objects = []
        for obj in regen3d_output.objects:
            manifest_obj = manifest_by_id.get(str(obj.id), {})
            manifest_articulation = manifest_obj.get("articulation") or {}
            articulation_hint = manifest_articulation.get("type")
            articulation_hint_source = manifest_articulation.get("required_source")
            if not articulation_hint:
                hints = manifest_articulation.get("hints")
                if isinstance(hints, list) and hints:
                    first_hint = hints[0] if isinstance(hints[0], dict) else {}
                    joint_types = first_hint.get("joint_types", [])
                    parts = first_hint.get("parts", [])
                    if joint_types:
                        articulation_hint = joint_types[0]
                    elif parts:
                        articulation_hint = parts[0]
                    articulation_hint_source = articulation_hint_source or first_hint.get("source")

            inv_obj = {
                "id": obj.id,
                "category": obj.category or "object",
                "short_description": obj.description or f"Object {obj.id}",
                "sim_role": (
                    manifest_obj.get("sim_role")
                    or (obj.sim_role if obj.sim_role != "unknown" else "static")
                ),
                "articulation_hint": articulation_hint,
                "articulation_hint_source": articulation_hint_source,
                "bounds": obj.bounds,
                "is_floor_contact": obj.pose.is_floor_contact,
            }
            objects.append(inv_obj)

        return {
            "scene_id": self.scene_id,
            "environment_type": self.environment_type,
            "total_objects": len(objects),
            "objects": objects,
            "metadata": {
                "source": "regen3d_adapter",
                "generated_at": datetime.utcnow().isoformat() + "Z",
            },
        }

    def _run_scale(self) -> StepResult:
        """Run scale calibration (optional)."""
        start_time = time.time()
        production_mode = self.environment == "production"
        marker_path = self.assets_dir / ".scale_complete"
        report_path = self.assets_dir / "scale_report.json"
        warnings: List[str] = []

        try:
            scale_module_path = REPO_ROOT / "scale-job" / "run_scale_from_layout.py"
            spec = importlib.util.spec_from_file_location("run_scale_from_layout", scale_module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Unable to load scale module at {scale_module_path}")
            scale_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scale_module)

            layout_path = self.layout_dir / "scene_layout.json"
            if not layout_path.is_file():
                message = f"Scale calibration failed: missing layout at {layout_path}"
                if production_mode:
                    return StepResult(
                        step=PipelineStep.SCALE,
                        success=False,
                        duration_seconds=time.time() - start_time,
                        message=message,
                        outputs={"completion_marker": str(marker_path), "warnings": [message]},
                    )
                warnings.append(message)
                return StepResult(
                    step=PipelineStep.SCALE,
                    success=True,
                    duration_seconds=time.time() - start_time,
                    message=message,
                    outputs={"completion_marker": str(marker_path), "warnings": warnings},
                )

            layout = _load_json(layout_path, "scale calibration layout")
            objects = layout.get("objects", [])
            room_box = layout.get("room_box")

            manifest_path = self.assets_dir / "scene_manifest.json"
            manifest = None
            if manifest_path.is_file():
                try:
                    manifest = _load_json(manifest_path, "scale calibration manifest")
                except NonRetryableError as exc:
                    warnings.append(str(exc))

            metric_metadata, metadata_path = scale_module.load_metric_metadata(self.layout_dir)

            scales: List[float] = []
            used_samples: List[dict] = []

            ref_scales, ref_samples = scale_module.gather_reference_scales(metric_metadata, objects)
            scales.extend(ref_scales)
            used_samples.extend(ref_samples)

            metric_scales, metric_samples = scale_module.gather_scene_metric_scales(metric_metadata, room_box)
            scales.extend(metric_scales)
            used_samples.extend(metric_samples)

            up_axis = 1
            for obj in objects:
                name = obj.get("class_name", "")
                prior_h = scale_module.get_prior_for_class(name)
                obb = obj.get("obb")

                if prior_h is None or obb is None:
                    continue

                extents = obb.get("extents")
                if not isinstance(extents, list) or len(extents) != 3:
                    continue

                measured_h = 2.0 * float(extents[up_axis])
                if measured_h <= 1e-6:
                    continue

                scale_sample = prior_h / measured_h
                scales.append(scale_sample)
                used_samples.append(
                    {
                        "source": "class_prior",
                        "class_name": name,
                        "prior_h_m": prior_h,
                        "measured_h_units": measured_h,
                        "scale_sample": scale_sample,
                    }
                )

            scale_factor = 1.0
            confidence = 0.0
            outliers: List[dict] = []

            if not scales:
                warnings.append("No scale cues available; defaulting to scale factor 1.0.")
            else:
                scales_arr = np.array(scales, dtype=np.float32)
                scale_factor = float(np.median(scales_arr))
                if scale_factor <= 0:
                    warnings.append("Computed non-positive scale factor; defaulting to 1.0.")
                    scale_factor = 1.0
                else:
                    rel_dev = np.abs(scales_arr - scale_factor) / scale_factor
                    outlier_mask = rel_dev > 0.25
                    for idx in np.where(outlier_mask)[0].tolist():
                        sample = dict(used_samples[idx])
                        sample["relative_deviation"] = float(rel_dev[idx])
                        outliers.append(sample)

                cv = float(np.std(scales_arr) / scale_factor) if scale_factor > 0 else 1.0
                sample_factor = min(1.0, len(scales) / 5.0)
                confidence = max(0.0, min(1.0, sample_factor * (1.0 - min(cv, 1.0))))

            if outliers:
                warnings.append(f"{len(outliers)} outlier scale samples detected.")

            low_confidence = confidence < 0.5
            if low_confidence:
                warnings.append(f"Low scale confidence ({confidence:.2f}).")

            report = {
                "scale_factor": scale_factor,
                "confidence": confidence,
                "n_samples": len(scales),
                "reference_samples": [
                    sample for sample in used_samples if sample.get("source") != "class_prior"
                ],
                "class_prior_samples": [
                    sample for sample in used_samples if sample.get("source") == "class_prior"
                ],
                "outliers": outliers,
                "warnings": warnings,
                "layout_path": str(layout_path),
                "metadata_path": str(metadata_path) if metadata_path else None,
                "manifest_path": str(manifest_path) if manifest is not None else None,
                "manifest_summary": {
                    "object_count": len(manifest.get("objects", [])) if isinstance(manifest, dict) else None,
                },
            }

            _safe_write_text(
                report_path,
                json.dumps(report, indent=2),
                context="scale report",
            )
            self._write_marker(marker_path, status="completed")

            if low_confidence or not scales:
                message = "Scale calibration warnings detected."
                if production_mode:
                    return StepResult(
                        step=PipelineStep.SCALE,
                        success=False,
                        duration_seconds=time.time() - start_time,
                        message=message,
                        outputs={
                            "completion_marker": str(marker_path),
                            "scale_report": str(report_path),
                            "scale_factor": scale_factor,
                            "reference_samples": report["reference_samples"],
                            "warnings": warnings,
                        },
                    )
                return StepResult(
                    step=PipelineStep.SCALE,
                    success=True,
                    duration_seconds=time.time() - start_time,
                    message=message,
                    outputs={
                        "completion_marker": str(marker_path),
                        "scale_report": str(report_path),
                        "scale_factor": scale_factor,
                        "reference_samples": report["reference_samples"],
                        "warnings": warnings,
                    },
                )

            return StepResult(
                step=PipelineStep.SCALE,
                success=True,
                duration_seconds=time.time() - start_time,
                message="Scale calibration report generated",
                outputs={
                    "completion_marker": str(marker_path),
                    "scale_report": str(report_path),
                    "scale_factor": scale_factor,
                    "reference_samples": report["reference_samples"],
                    "warnings": warnings,
                },
            )
        except Exception as exc:
            self._log_exception_traceback("Scale calibration failed", exc)
            message = f"Scale calibration failed: {self._summarize_exception(exc)}"
            success = not production_mode
            return StepResult(
                step=PipelineStep.SCALE,
                success=success,
                duration_seconds=time.time() - start_time,
                message=message,
                outputs={
                    "completion_marker": str(marker_path),
                    "scale_report": str(report_path),
                    "warnings": [message],
                },
            )

    def _run_interactive(self) -> StepResult:
        """Run interactive asset processing."""
        production_mode = self.environment == "production"
        manifest_path = self.assets_dir / "scene_manifest.json"
        required_ids: List[str] = []
        if manifest_path.is_file():
            try:
                manifest = _load_json(manifest_path, "interactive manifest")
                required_ids = self._required_articulation_ids(manifest)
            except NonRetryableError as exc:
                return StepResult(
                    step=PipelineStep.INTERACTIVE,
                    success=False,
                    duration_seconds=0,
                    message=str(exc),
                )

        if self.skip_interactive:
            if production_mode:
                return StepResult(
                    step=PipelineStep.INTERACTIVE,
                    success=False,
                    duration_seconds=0,
                    message=(
                        "Interactive job is required in production. "
                        "Disable skip_interactive and provide PARTICULATE_ENDPOINT."
                    ),
                    outputs={"required_articulations": required_ids},
                )
            if required_ids:
                return StepResult(
                    step=PipelineStep.INTERACTIVE,
                    success=False,
                    duration_seconds=0,
                    message=(
                        "Articulation required but interactive job is disabled. "
                        "Run with --with-interactive and configure Particulate "
                        "(PARTICULATE_ENDPOINT or PARTICULATE_MODE=local)."
                    ),
                    outputs={"required_articulations": required_ids},
                )
            self.log("Interactive job skipped (requires PARTICULATE_ENDPOINT)")
            marker_path = self.assets_dir / ".interactive_complete"
            self._write_marker(marker_path, status="completed")
            return StepResult(
                step=PipelineStep.INTERACTIVE,
                success=True,
                duration_seconds=0,
                message="Skipped (no PARTICULATE_ENDPOINT)",
                outputs={"completion_marker": str(marker_path)},
            )

        particulate_mode = (os.getenv("PARTICULATE_MODE", "remote") or "remote").strip().lower()
        endpoint = (os.getenv("PARTICULATE_ENDPOINT", "") or "").strip()
        local_endpoint = (os.getenv("PARTICULATE_LOCAL_ENDPOINT", "") or "").strip()
        articulation_backend = (
            os.getenv("ARTICULATION_BACKEND", "particulate") or "particulate"
        ).strip().lower()

        if required_ids and articulation_backend not in {"particulate", "auto"}:
            return StepResult(
                step=PipelineStep.INTERACTIVE,
                success=False,
                duration_seconds=0,
                message=(
                    "Required articulated assets need Particulate backend. "
                    f"ARTICULATION_BACKEND={articulation_backend} is not allowed."
                ),
                outputs={"required_articulations": required_ids},
            )
        if required_ids and particulate_mode in {"mock", "skip"}:
            return StepResult(
                step=PipelineStep.INTERACTIVE,
                success=False,
                duration_seconds=0,
                message=(
                    "Required articulated assets cannot run with "
                    f"PARTICULATE_MODE={particulate_mode}."
                ),
                outputs={"required_articulations": required_ids},
            )

        resolved_endpoint = endpoint
        if particulate_mode == "local":
            resolved_endpoint = local_endpoint or "http://localhost:8080"

        if particulate_mode != "local" and not endpoint:
            if production_mode:
                return StepResult(
                    step=PipelineStep.INTERACTIVE,
                    success=False,
                    duration_seconds=0,
                    message="PARTICULATE_ENDPOINT is required in production",
                    outputs={"required_articulations": required_ids},
                )
            if required_ids:
                return StepResult(
                    step=PipelineStep.INTERACTIVE,
                    success=False,
                    duration_seconds=0,
                    message=(
                        "Articulation required but PARTICULATE endpoint is not set "
                        "for remote mode."
                    ),
                    outputs={"required_articulations": required_ids},
                )
            marker_path = self.assets_dir / ".interactive_complete"
            self._write_marker(marker_path, status="completed")
            return StepResult(
                step=PipelineStep.INTERACTIVE,
                success=True,
                duration_seconds=0,
                message="Skipped (PARTICULATE_ENDPOINT not set)",
                outputs={"completion_marker": str(marker_path)},
            )

        interactive_script = REPO_ROOT / "interactive-job" / "run_interactive_assets.py"
        env = os.environ.copy()
        env.update({
            "BUCKET": os.getenv("BUCKET", "local"),
            "ASSETS_PREFIX": str(self.assets_dir),
            "REGEN3D_PREFIX": str(self.regen3d_dir),
            "SCENE_ID": self.scene_id,
            "PARTICULATE_ENDPOINT": endpoint,
            "PARTICULATE_MODE": particulate_mode,
            "PARTICULATE_LOCAL_ENDPOINT": local_endpoint or "http://localhost:8080",
            "PARTICULATE_LOCAL_MODEL": os.getenv("PARTICULATE_LOCAL_MODEL", ""),
            "APPROVED_PARTICULATE_MODELS": os.getenv("APPROVED_PARTICULATE_MODELS", "pat_b"),
            "ARTICULATION_BACKEND": articulation_backend,
            "PIPELINE_ENV": os.getenv("PIPELINE_ENV", "development"),
            "DISALLOW_PLACEHOLDER_URDF": os.getenv("DISALLOW_PLACEHOLDER_URDF", "false"),
        })
        if particulate_mode == "local":
            env["PARTICULATE_ENDPOINT"] = resolved_endpoint

        self.log("Running interactive-job entrypoint locally")
        proc = subprocess.run(
            [sys.executable, str(interactive_script)],
            cwd=str(REPO_ROOT),
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            return StepResult(
                step=PipelineStep.INTERACTIVE,
                success=False,
                duration_seconds=0,
                message=f"interactive-job failed with exit code {proc.returncode}",
            )

        failure_marker = self.assets_dir / ".interactive_failed"
        if failure_marker.is_file():
            failure_payload = _load_json(failure_marker, "interactive failure payload")
            return StepResult(
                step=PipelineStep.INTERACTIVE,
                success=False,
                duration_seconds=0,
                message=f"interactive-job reported failure: {failure_payload.get('reason')}",
                outputs={"failure_payload": failure_payload},
            )

        results_path = self.assets_dir / "interactive" / "interactive_results.json"
        if not results_path.is_file():
            return StepResult(
                step=PipelineStep.INTERACTIVE,
                success=False,
                duration_seconds=0,
                message="interactive-job results not found",
            )

        results_data = _load_json(results_path, "interactive results")
        if required_ids:
            result_by_id = {
                str(r.get("id")): r for r in results_data.get("objects", [])
            }
            missing = [
                obj_id for obj_id in required_ids
                if not result_by_id.get(str(obj_id), {}).get("is_articulated")
            ]
            if missing:
                return StepResult(
                    step=PipelineStep.INTERACTIVE,
                    success=False,
                    duration_seconds=0,
                    message="Articulation required but not available for all objects",
                    outputs={"missing_articulations": missing},
                )

        marker_path = self.assets_dir / ".interactive_complete"
        self._write_marker(marker_path, status="completed")
        return StepResult(
            step=PipelineStep.INTERACTIVE,
            success=True,
            duration_seconds=0,
            message="Interactive job completed",
            outputs={
                "interactive_results": str(results_path),
                "articulated_count": results_data.get("articulated_count", 0),
                "completion_marker": str(marker_path),
            },
        )

    def _run_simready(self) -> StepResult:
        """Run simready preparation."""
        # Reload repo-local .env so key discovery does not depend on caller cwd.
        load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)

        # Convert GLB→USDZ first so SimReady can find visual assets.
        self._convert_glb_assets_to_usdz()

        manifest_path = self.assets_dir / "scene_manifest.json"
        if not manifest_path.is_file():
            return StepResult(
                step=PipelineStep.SIMREADY,
                success=False,
                duration_seconds=0,
                message="Manifest not found - run regen3d step first",
            )

        missing_usdz = self._find_missing_usdz_assets()
        if missing_usdz:
            preview = ", ".join(missing_usdz[:5])
            if len(missing_usdz) > 5:
                preview = f"{preview}, ..."
            if self._is_production_mode() or not self._should_skip_quality_gates():
                return StepResult(
                    step=PipelineStep.SIMREADY,
                    success=False,
                    duration_seconds=0,
                    message=(
                        "GLB→USDZ conversion incomplete before SimReady; "
                        f"missing asset.usdz in {len(missing_usdz)} asset directories: {preview}"
                    ),
                    outputs={"missing_usdz_assets": missing_usdz},
                )
            self.log(
                (
                    "GLB→USDZ conversion incomplete before SimReady; "
                    f"missing asset.usdz in {len(missing_usdz)} asset directories: {preview}. "
                    "Continuing because SKIP_QUALITY_GATES is enabled."
                ),
                "WARNING",
            )
        try:
            from blueprint_sim.simready import run_from_env
        except ImportError as exc:
            return StepResult(
                step=PipelineStep.SIMREADY,
                success=False,
                duration_seconds=0,
                message=f"SimReady dependencies missing: {self._summarize_exception(exc)}",
            )

        assets_prefix = "assets"
        os.environ.setdefault("BUCKET", "local")
        os.environ.setdefault("SCENE_ID", self.scene_id)
        os.environ.setdefault("ASSETS_PREFIX", assets_prefix)
        production_mode = self._is_production_mode()
        # Default physics policy:
        # - prod: Gemini-first (fail closed if configured that way in simready-job)
        # - dev/test: auto + heuristic fallback allowed (CI-friendly)
        os.environ.setdefault("SIMREADY_PHYSICS_MODE", "gemini" if production_mode else "auto")
        os.environ.setdefault("SIMREADY_ALLOW_DETERMINISTIC_PHYSICS", "0" if production_mode else "1")
        os.environ.setdefault("SIMREADY_ALLOW_HEURISTIC_FALLBACK", "0" if production_mode else "1")
        physics_mode = (os.environ.get("SIMREADY_PHYSICS_MODE") or "").strip().lower()
        if not production_mode and physics_mode == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            return StepResult(
                step=PipelineStep.SIMREADY,
                success=False,
                duration_seconds=0,
                message=(
                    "SIMREADY_PHYSICS_MODE=gemini requires GEMINI_API_KEY, "
                    "but it was not found in the environment after loading .env."
                ),
                outputs={"physics_mode": physics_mode},
            )

        return_code = run_from_env(root=self.scene_dir)
        if return_code != 0:
            return StepResult(
                step=PipelineStep.SIMREADY,
                success=False,
                duration_seconds=0,
                message=f"SimReady job failed with exit code {return_code}",
                outputs={"exit_code": return_code},
            )

        marker_path = self.assets_dir / ".simready_complete"
        physics_summary_path = self.assets_dir / "simready_physics.json"
        outputs = {
            "completion_marker": str(marker_path) if marker_path.exists() else None,
            "assets_prefix": assets_prefix,
            "physics_summary": (
                str(physics_summary_path) if physics_summary_path.exists() else None
            ),
        }
        return StepResult(
            step=PipelineStep.SIMREADY,
            success=True,
            duration_seconds=0,
            message="SimReady preparation completed",
            outputs=outputs,
        )

    def _convert_glb_assets_to_usdz(self) -> None:
        """Convert GLB assets to USDZ for USD scene assembly.

        Iterates over asset directories and converts asset.glb → asset.usdz
        using the existing glb_to_usd converter. Skips if USDZ already exists.
        """
        try:
            from usd_assembly_job.glb_to_usd import convert_glb_to_usd
        except ImportError:
            # Try alternate import path
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "glb_to_usd",
                    Path(__file__).parent.parent / "usd-assembly-job" / "glb_to_usd.py",
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                convert_glb_to_usd = mod.convert_glb_to_usd
            except Exception as exc:
                self.log(f"WARNING: GLB→USDZ conversion unavailable: {exc}")
                return

        converted = 0
        skipped = 0
        failed = 0

        for obj_dir in sorted(self.assets_dir.iterdir()):
            if not obj_dir.is_dir() or not obj_dir.name.startswith("obj_"):
                continue

            glb_path = obj_dir / "asset.glb"
            usdz_path = obj_dir / "asset.usdz"

            if not glb_path.is_file():
                continue
            if usdz_path.is_file():
                skipped += 1
                continue

            try:
                success = convert_glb_to_usd(glb_path, usdz_path, create_usdz=True)
                if success:
                    converted += 1
                    self.log(f"Converted {obj_dir.name}/asset.glb → asset.usdz")
                else:
                    failed += 1
                    self.log(f"WARNING: GLB→USDZ failed for {obj_dir.name}")
            except Exception as exc:
                failed += 1
                self.log(f"WARNING: GLB→USDZ error for {obj_dir.name}: {exc}")

        self.log(
            f"GLB→USDZ conversion: {converted} converted, {skipped} skipped, {failed} failed"
        )

    def _find_missing_usdz_assets(self) -> List[str]:
        """Return obj_* directories that have asset.glb but no asset.usdz."""
        missing: List[str] = []
        if not self.assets_dir.is_dir():
            return missing

        for obj_dir in sorted(self.assets_dir.iterdir()):
            if not obj_dir.is_dir() or not obj_dir.name.startswith("obj_"):
                continue
            glb_path = obj_dir / "asset.glb"
            usdz_path = obj_dir / "asset.usdz"
            if glb_path.is_file() and not usdz_path.is_file():
                missing.append(obj_dir.name)
        return missing

    def _run_usd_assembly(self) -> StepResult:
        """Run USD assembly."""
        # Check prerequisites
        manifest_path = self.assets_dir / "scene_manifest.json"
        layout_path = self.layout_dir / "scene_layout_scaled.json"

        if not manifest_path.is_file():
            return StepResult(
                step=PipelineStep.USD,
                success=False,
                duration_seconds=0,
                message="Manifest not found",
            )

        if not layout_path.is_file():
            return StepResult(
                step=PipelineStep.USD,
                success=False,
                duration_seconds=0,
                message="Layout not found",
            )

        try:
            manifest = _load_json(manifest_path, "usd assembly manifest")
            layout = _load_json(layout_path, "usd assembly layout")
        except NonRetryableError as exc:
            return StepResult(
                step=PipelineStep.USD,
                success=False,
                duration_seconds=0,
                message=str(exc),
            )

        # GLB→USDZ conversion already ran in _run_simready(); skip if already done
        self._convert_glb_assets_to_usdz()

        # Generate a minimal scene.usda
        usda_content = self._generate_usda(manifest, layout)
        usda_path = self.usd_dir / "scene.usda"
        _safe_write_text(
            usda_path,
            usda_content,
            context="scene usda",
        )
        self._write_marker(self.assets_dir / ".usd_assembly_complete", status="completed")

        self.log(f"Generated USD: {usda_path}")

        return StepResult(
            step=PipelineStep.USD,
            success=True,
            duration_seconds=0,
            message="USD assembly completed",
            outputs={"scene_usda": str(usda_path)},
        )

    def _select_usd_reference_asset(self, obj_asset_dir: Path) -> Tuple[str, str]:
        """Pick the best USD reference path for scene assembly.

        Priority:
        simready.usda -> model.usdz -> asset.usdz ->
        model.usd/usda/usdc -> asset.usd/usda/usdc
        """
        preferred_names = [
            "simready.usda",
            "model.usdz",
            "asset.usdz",
            "model.usd",
            "model.usda",
            "model.usdc",
            "asset.usd",
            "asset.usda",
            "asset.usdc",
        ]
        for name in preferred_names:
            candidate = obj_asset_dir / name
            if candidate.is_file():
                return os.path.relpath(str(candidate), str(self.usd_dir)), name
        return "", ""

    def _generate_usda(self, manifest: Dict, layout: Dict) -> str:
        """Generate a scene.usda file with asset references and physics."""
        # Production pipeline hardcodes metersPerUnit=1.0 and upAxis="Y".
        # Match that to avoid mismatches with assets that assume meters.
        meters_per_unit = 1.0
        coord_frame = "Y"

        lines = [
            '#usda 1.0',
            '(',
            f'    metersPerUnit = {meters_per_unit}',
            f'    upAxis = "{coord_frame}"',
            '    defaultPrim = "World"',
            ')',
            '',
            'def Xform "World" {',
            '    def Xform "Scene" {',
        ]

        # Add objects with USD asset references and physics APIs
        for obj in manifest.get("objects", []):
            obj_id = obj.get("id", "unknown")
            asset_path = obj.get("asset", {}).get("path", "")

            # Find the USD asset file for this object
            obj_asset_dir = self.assets_dir / f"obj_{obj_id}"
            usd_ref = ""
            usd_ref_name = ""
            if obj_asset_dir.is_dir():
                usd_ref, usd_ref_name = self._select_usd_reference_asset(obj_asset_dir)

            # Get transform (position, rotation, scale)
            transform = obj.get("transform", {})
            pos = transform.get("position", {})
            px = pos.get("x", 0)
            py = pos.get("y", 0)
            pz = pos.get("z", 0)

            rot = transform.get("rotation_quaternion", {})
            rw = rot.get("w", 1.0)
            rx = rot.get("x", 0.0)
            ry = rot.get("y", 0.0)
            rz = rot.get("z", 0.0)

            scl = transform.get("scale", {})
            sx = scl.get("x", 1.0)
            sy = scl.get("y", 1.0)
            sz = scl.get("z", 1.0)

            use_simready_wrapper = usd_ref_name == "simready.usda"

            if usd_ref:
                lines.append(f'        def Xform "obj_{obj_id}" (')
                if not use_simready_wrapper:
                    lines.append(
                        '            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"]'
                    )
                lines.append(f'            prepend references = @{usd_ref}@')
                lines.append(f'        ) {{')
                if not use_simready_wrapper:
                    # Physics properties from manifest are only fallback defaults.
                    # SimReady wrappers already author object-level physics.
                    mass = obj.get("physics", {}).get("mass", 1.0)
                    lines.append(f'            float physics:mass = {mass}')
                    lines.append(f'            bool physics:collisionEnabled = true')
                    # Local non-simready assets often lack collision meshes,
                    # so keep them kinematic to avoid falling through ground.
                    lines.append(f'            bool physics:kinematicEnabled = true')
            else:
                lines.append(f'        def Xform "obj_{obj_id}" (')
                lines.append(f'            # No USD asset found for: {asset_path}')
                lines.append(f'        ) {{')
            lines.append(f'            double3 xformOp:translate = ({px}, {py}, {pz})')
            lines.append(f'            quatd xformOp:orient = ({rw}, {rx}, {ry}, {rz})')
            lines.append(f'            double3 xformOp:scale = ({sx}, {sy}, {sz})')
            lines.append(f'            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]')
            lines.append(f'        }}')
            lines.append('')

        lines.append('    }')

        # Add default dome light for camera rendering
        # Without lighting, cameras capture black/uniform images
        skip_lighting = os.environ.get("USD_SKIP_DEFAULT_LIGHTING", "0") == "1"
        if not skip_lighting:
            dome_intensity = float(os.environ.get("USD_DOME_LIGHT_INTENSITY", "200.0"))
            dome_color_temp = float(os.environ.get("USD_DOME_LIGHT_COLOR_TEMP", "6500.0"))
            lines.append('')
            lines.append('    def DomeLight "DefaultDomeLight" {')
            lines.append(f'        float inputs:intensity = {dome_intensity}')
            lines.append(f'        float inputs:colorTemperature = {dome_color_temp}')
            lines.append('        bool inputs:enableColorTemperature = true')
            lines.append('    }')

        # Add PhysicsScene for PhysX simulation
        lines.append('')
        lines.append('    def PhysicsScene "PhysicsScene" {')
        if coord_frame == "Z":
            lines.append('        vector3f physics:gravityDirection = (0, 0, -1)')
        else:
            lines.append('        vector3f physics:gravityDirection = (0, -1, 0)')
        lines.append('        float physics:gravityMagnitude = 9.81')
        lines.append('    }')

        # Add a ground plane with physics collision for PhysX
        if coord_frame == "Z":
            gp_points = '[(-500, -500, 0), (500, -500, 0), (500, 500, 0), (-500, 500, 0)]'
        else:
            gp_points = '[(-500, 0, -500), (500, 0, -500), (500, 0, 500), (-500, 0, 500)]'
        lines.append('')
        lines.append('    def Mesh "GroundPlane" (')
        lines.append('        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsRigidBodyAPI"]')
        lines.append('    ) {')
        lines.append(f'        float3[] points = {gp_points}')
        lines.append('        int[] faceVertexCounts = [4]')
        lines.append('        int[] faceVertexIndices = [0, 1, 2, 3]')
        lines.append('        bool physics:collisionEnabled = true')
        lines.append('        bool physics:kinematicEnabled = true')
        lines.append('        uniform token purpose = "default"')
        lines.append('    }')

        # Add camera prims referenced by replicator configs
        # Compute scene centroid from object bounds for camera targeting
        objects = manifest.get("objects", [])
        scene_center = [0.0, 0.5, 0.0]
        if objects:
            cx_sum, cy_sum, cz_sum = 0.0, 0.0, 0.0
            for obj in objects:
                t = obj.get("transform", {}).get("position", {})
                cx_sum += t.get("x", 0.0)
                cy_sum += t.get("y", 0.0)
                cz_sum += t.get("z", 0.0)
            n = len(objects)
            scene_center = [cx_sum / n, cy_sum / n + 0.5, cz_sum / n]

        # Front camera: positioned 2m in front of scene center, looking at it
        front_cam_pos = (scene_center[0], scene_center[1] + 0.5, scene_center[2] + 2.0)
        lines.append('')
        lines.append('    def Camera "front_camera" {')
        lines.append(f'        double3 xformOp:translate = ({front_cam_pos[0]:.4f}, {front_cam_pos[1]:.4f}, {front_cam_pos[2]:.4f})')
        lines.append('        float3 xformOp:rotateXYZ = (-15.0, 0.0, 0.0)')
        lines.append('        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]')
        lines.append('        float focalLength = 24.0')
        lines.append('        float horizontalAperture = 36.0')
        lines.append('        float verticalAperture = 24.0')
        lines.append('        float2 clippingRange = (0.01, 1000.0)')
        lines.append('    }')

        # Wrist camera: placeholder at origin, typically attached to robot EE at runtime
        lines.append('')
        lines.append('    def Camera "wrist_camera" {')
        lines.append('        double3 xformOp:translate = (0.0, 1.0, 0.0)')
        lines.append('        float3 xformOp:rotateXYZ = (-90.0, 0.0, 0.0)')
        lines.append('        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]')
        lines.append('        float focalLength = 18.0')
        lines.append('        float horizontalAperture = 36.0')
        lines.append('        float verticalAperture = 24.0')
        lines.append('        float2 clippingRange = (0.01, 100.0)')
        lines.append('    }')

        lines.append('}')

        return '\n'.join(lines)

    def _run_inventory_enrichment(self) -> StepResult:
        """Run inventory enrichment before replicator generation."""
        inventory_path = self.seg_dir / "inventory.json"
        if not inventory_path.is_file():
            message = "Inventory not found; skipping enrichment"
            self.log(message, "WARNING")
            return StepResult(
                step=PipelineStep.INVENTORY_ENRICHMENT,
                success=True,
                duration_seconds=0,
                message=message,
            )

        output_path = self.seg_dir / "inventory_enriched.json"
        try:
            enrich_inventory_file(inventory_path, output_path=output_path, mode="gemini")
        except InventoryEnrichmentError as exc:
            message = f"Inventory enrichment skipped: {exc}"
            self.log(message, "WARNING")
            return StepResult(
                step=PipelineStep.INVENTORY_ENRICHMENT,
                success=True,
                duration_seconds=0,
                message=message,
            )

        return StepResult(
            step=PipelineStep.INVENTORY_ENRICHMENT,
            success=True,
            duration_seconds=0,
            message="Inventory enrichment completed",
            outputs={"inventory_enriched": str(output_path)},
        )

    def _run_replicator(self) -> StepResult:
        """Run replicator bundle generation using the full Gemini-powered pipeline."""
        start_time = time.time()

        # Try to use the full replicator-job with Gemini scene analysis
        try:
            replicator_module_path = REPO_ROOT / "replicator-job" / "generate_replicator_bundle.py"
            spec = importlib.util.spec_from_file_location(
                "generate_replicator_bundle", replicator_module_path,
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {replicator_module_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            self.log("Running Gemini-powered replicator scene analysis...")

            bundle, analysis_result = mod.process_scene(
                root=self.scene_dir,
                scene_id=self.scene_id,
                seg_prefix="seg",
                assets_prefix="assets",
                usd_prefix="usd",
                replicator_prefix="replicator",
            )

            mod.write_replicator_bundle(bundle, analysis_result, self.replicator_dir)

            # Post-process configs to ensure required sensor metadata fields
            self._ensure_replicator_sensor_metadata()

            policy_count = len(bundle.policies) if hasattr(bundle, "policies") else 0
            self._write_marker(self.replicator_dir / ".replicator_complete", status="completed")
            self.log(f"Gemini replicator bundle: {policy_count} policies generated")

            return StepResult(
                step=PipelineStep.REPLICATOR,
                success=True,
                duration_seconds=time.time() - start_time,
                message=f"Replicator bundle generated ({policy_count} policies via Gemini)",
                outputs={
                    "policies_count": policy_count,
                    "bundle_dir": str(self.replicator_dir),
                },
            )

        except Exception as exc:
            self.log(f"Gemini replicator failed ({exc}); falling back to stub", "WARNING")
            return self._run_replicator_stub()

    def _ensure_replicator_sensor_metadata(self) -> None:
        """Ensure all replicator config JSONs have camera_list and stream_ids."""
        configs_dir = self.replicator_dir / "configs"
        if not configs_dir.is_dir():
            return
        default_cameras = ["front_camera", "wrist_camera"]
        for config_path in configs_dir.glob("*.json"):
            try:
                with config_path.open("r") as f:
                    config = json.load(f)
                capture = config.get("capture_config", {})
                changed = False
                if "camera_list" not in capture and "cameras" not in capture:
                    capture["camera_list"] = default_cameras
                    changed = True
                if "stream_ids" not in capture and "streams" not in capture:
                    annotations = capture.get("annotations", capture.get("modalities", ["rgb", "depth"]))
                    capture["stream_ids"] = annotations
                    changed = True
                if changed:
                    config["capture_config"] = capture
                    with config_path.open("w") as f:
                        json.dump(config, f, indent=2)
                    self.log(f"Patched sensor metadata in {config_path.name}")
            except Exception as exc:
                self.log(f"Warning: failed to patch {config_path.name}: {exc}", "WARNING")

    def _run_replicator_stub(self) -> StepResult:
        """Fallback: generate minimal replicator bundle without Gemini."""
        manifest_path = self.assets_dir / "scene_manifest.json"
        if not manifest_path.is_file():
            return StepResult(
                step=PipelineStep.REPLICATOR, success=False,
                duration_seconds=0, message="Manifest not found",
            )

        manifest = _load_json(manifest_path, "replicator manifest")
        inventory = {}
        inv_path = self.seg_dir / "inventory_enriched.json"
        if not inv_path.is_file():
            inv_path = self.seg_dir / "inventory.json"
        if inv_path.is_file():
            inventory = _load_json(inv_path, "replicator inventory")

        policies_dir = self.replicator_dir / "policies"
        policies_dir.mkdir(parents=True, exist_ok=True)
        configs_dir = self.replicator_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)

        capture_config = self._resolve_default_capture_config()
        _safe_write_text(
            configs_dir / "default_policy.json",
            json.dumps({"policy_id": "default_policy", "capture_config": capture_config}, indent=2),
            context="default policy config",
        )

        placement_regions = self._generate_placement_regions(manifest)
        _safe_write_text(
            self.replicator_dir / "placement_regions.usda", placement_regions,
            context="placement regions",
        )

        variation_dir = self.replicator_dir / "variation_assets"
        variation_dir.mkdir(parents=True, exist_ok=True)
        variation_manifest = self._generate_variation_manifest(manifest, inventory)
        _safe_write_text(
            variation_dir / "manifest.json",
            json.dumps(variation_manifest, indent=2),
            context="variation asset manifest",
        )
        self._write_marker(self.replicator_dir / ".replicator_complete", status="completed")

        self.log("Generated minimal replicator bundle (stub fallback)")
        return StepResult(
            step=PipelineStep.REPLICATOR, success=True,
            duration_seconds=0, message="Replicator bundle generated (stub fallback)",
        )

    def _run_variation_gen(self) -> StepResult:
        """Run variation asset generation for local testing."""
        manifest_path = self.replicator_dir / "variation_assets" / "manifest.json"
        if not manifest_path.is_file():
            return StepResult(
                step=PipelineStep.VARIATION_GEN,
                success=False,
                duration_seconds=0,
                message="Variation manifest not found - run replicator step first",
            )

        variation_assets_dir = self.scene_dir / "variation_assets"
        variation_assets_dir.mkdir(parents=True, exist_ok=True)

        try:
            manifest = _load_json(manifest_path, "variation assets manifest")
        except NonRetryableError as exc:
            return StepResult(
                step=PipelineStep.VARIATION_GEN,
                success=False,
                duration_seconds=0,
                message=str(exc),
            )
        assets = manifest.get("assets", [])
        if not assets:
            return StepResult(
                step=PipelineStep.VARIATION_GEN,
                success=False,
                duration_seconds=0,
                message="Variation manifest contains no assets",
            )

        use_mock = parse_bool_env(os.getenv("VARIATION_GEN_USE_MOCK"), default=False)
        if use_mock:
            self._generate_mock_variation_assets(assets, variation_assets_dir)
            marker_path = variation_assets_dir / ".variation_pipeline_complete"
            self._write_marker(marker_path, status="completed")

            return StepResult(
                step=PipelineStep.VARIATION_GEN,
                success=True,
                duration_seconds=0,
                message="Variation assets generated (mock)",
                outputs={
                    "variation_assets_manifest": str(variation_assets_dir / "variation_assets.json"),
                    "variation_marker": str(marker_path),
                },
            )

        # Hybrid VM mode: Gemini images locally + Hunyuan3D on GPU VM
        hybrid_mode = os.getenv("VARIATION_GEN_MODE", "").lower()
        if hybrid_mode == "hybrid_vm":
            return self._run_variation_gen_hybrid(variation_assets_dir)

        gcs_scene_dir = self._ensure_gcs_scene_link()
        if gcs_scene_dir is None:
            return StepResult(
                step=PipelineStep.VARIATION_GEN,
                success=False,
                duration_seconds=0,
                message="Unable to prepare GCS mount mapping for variation assets "
                "(check GCS_MOUNT_ROOT)",
            )

        variation_assets_prefix = f"scenes/{self.scene_id}/variation_assets"
        replicator_prefix = f"scenes/{self.scene_id}/replicator"
        env = os.environ.copy()
        env.update({
            "SCENE_ID": self.scene_id,
            "BUCKET": env.get("BUCKET", "local"),
            "REPLICATOR_PREFIX": replicator_prefix,
            "VARIATION_ASSETS_PREFIX": variation_assets_prefix,
        })
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "variation-asset-pipeline-job" / "run_variation_asset_pipeline.py"),
                ],
                check=True,
                env=env,
                cwd=str(REPO_ROOT),
            )
        except subprocess.CalledProcessError as exc:
            return StepResult(
                step=PipelineStep.VARIATION_GEN,
                success=False,
                duration_seconds=0,
                message=f"Variation asset pipeline failed: {self._summarize_exception(exc)}",
            )

        variation_assets_json = variation_assets_dir / "variation_assets.json"
        simready_assets_json = variation_assets_dir / "simready_assets.json"
        pipeline_summary = variation_assets_dir / "pipeline_summary.json"
        marker_path = variation_assets_dir / ".variation_pipeline_complete"

        if not variation_assets_json.is_file():
            try:
                self._write_variation_assets_manifest(
                    variation_assets_json,
                    simready_assets_json,
                    gcs_scene_dir / "variation_assets",
                )
            except FileNotFoundError:
                return StepResult(
                    step=PipelineStep.VARIATION_GEN,
                    success=False,
                    duration_seconds=0,
                    message="variation_assets.json missing after variation asset pipeline run",
                )
            except NonRetryableError as exc:
                return StepResult(
                    step=PipelineStep.VARIATION_GEN,
                    success=False,
                    duration_seconds=0,
                    message=str(exc),
                )

        if not marker_path.is_file():
            self._write_marker(marker_path, status="completed")

        usdz_assets = sorted(
            asset_path.as_posix() for asset_path in variation_assets_dir.glob("*.usdz")
        )

        return StepResult(
            step=PipelineStep.VARIATION_GEN,
            success=True,
            duration_seconds=0,
            message="Variation assets generated",
            outputs={
                "variation_assets_manifest": str(variation_assets_json),
                "simready_assets_manifest": str(simready_assets_json),
                "pipeline_summary": str(pipeline_summary),
                "variation_marker": str(marker_path),
                "variation_usdz_assets": usdz_assets,
            },
        )

    def _generate_mock_variation_assets(
        self,
        assets: List[Dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """Generate mock variation assets for local runs."""
        objects = []
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
            b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01"
            b"\xe2!\xbc\x33\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        for asset in assets:
            name = asset.get("name") or asset.get("id") or "variation_asset"
            category = asset.get("category", "object")
            description = asset.get("description") or f"Variation of {category}"
            asset_dir = output_dir / name
            asset_dir.mkdir(parents=True, exist_ok=True)
            reference_path = asset_dir / "reference.png"
            if not reference_path.exists():
                reference_path.write_bytes(png_bytes)

            objects.append({
                "id": name,
                "name": name,
                "category": category,
                "short_description": description,
                "sim_role": "manipulable_object",
                "must_be_separate_asset": True,
                "preferred_view": f"variation_assets/{name}/reference.png",
                "multiview_dir": None,
                "crop_path": None,
                "physics_hints": asset.get("physics_hints", {}),
                "semantic_class": asset.get("semantic_class", category),
                "asset": {
                    "license": "CC0",
                    "commercial_ok": True,
                },
            })

        payload = {
            "scene_id": self.scene_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source": "variation-gen-job-mock",
            "objects": objects,
            "metadata": {
                "total_objects": len(objects),
                "generation_type": "variation_assets",
            },
        }
        _safe_write_text(
            output_dir / "variation_assets.json",
            json.dumps(payload, indent=2),
            context="variation assets manifest",
        )

    def _run_variation_gen_hybrid(self, variation_assets_dir: Path) -> StepResult:
        """Run variation asset generation via hybrid local+VM orchestration."""
        t0 = time.time()
        try:
            from tools.variation_asset_runner import VariationAssetRunner
            runner = VariationAssetRunner(
                scene_dir=self.scene_dir,
                vm_zone=os.getenv("VM_ZONE", "us-east1-c"),
                vm_host=os.getenv("VM_HOST", "isaac-sim-ubuntu"),
            )
            summary = runner.run()
        except Exception as exc:
            self._log_exception_traceback("Hybrid variation gen failed", exc)
            return StepResult(
                step=PipelineStep.VARIATION_GEN,
                success=False,
                duration_seconds=time.time() - t0,
                message=f"Hybrid variation gen failed: {self._summarize_exception(exc)}",
            )

        succeeded = summary.get("succeeded", 0)
        total = summary.get("total", 0)
        marker_path = variation_assets_dir / ".variation_pipeline_complete"

        return StepResult(
            step=PipelineStep.VARIATION_GEN,
            success=succeeded > 0,
            duration_seconds=time.time() - t0,
            message=f"Variation assets generated via hybrid VM ({succeeded}/{total} succeeded)",
            outputs={
                "variation_assets_manifest": str(variation_assets_dir / "variation_assets.json"),
                "simready_assets_manifest": str(variation_assets_dir / "simready_assets.json"),
                "pipeline_summary": str(variation_assets_dir / "pipeline_summary.json"),
                "variation_marker": str(marker_path),
            },
        )

    def _ensure_gcs_scene_link(self) -> Optional[Path]:
        gcs_root = Path(os.getenv("GCS_MOUNT_ROOT", "/mnt/gcs"))
        gcs_scene_dir = gcs_root / "scenes" / self.scene_id
        try:
            gcs_scene_dir.parent.mkdir(parents=True, exist_ok=True)
            if not gcs_scene_dir.exists():
                gcs_scene_dir.symlink_to(self.scene_dir)
        except OSError as exc:
            self._log_exception_traceback(
                f"Failed to map {gcs_root} for variation assets (set GCS_MOUNT_ROOT to override)",
                exc,
            )
            return None
        return gcs_scene_dir

    def _write_variation_assets_manifest(
        self,
        output_path: Path,
        simready_manifest_path: Path,
        gcs_variation_dir: Path,
    ) -> None:
        if not simready_manifest_path.is_file():
            raise FileNotFoundError(str(simready_manifest_path))

        simready_manifest = _load_json(simready_manifest_path, "variation simready manifest")
        objects: List[Dict[str, Any]] = []
        for asset in simready_manifest.get("assets", []):
            name = asset.get("name") or "variation_asset"
            metadata = asset.get("metadata") or {}
            category = metadata.get("category", "object")
            description = metadata.get("description") or metadata.get("short_description") or ""
            license_name = metadata.get("license", "CC0")
            asset_path = asset.get("path")
            resolved_path = None
            if asset_path:
                asset_path = Path(asset_path)
                if asset_path.is_absolute():
                    try:
                        relative_path = asset_path.relative_to(gcs_variation_dir)
                        resolved_path = f"variation_assets/{relative_path.as_posix()}"
                    except ValueError:
                        resolved_path = f"variation_assets/{asset_path.name}"
                else:
                    asset_str = asset_path.as_posix()
                    if asset_str.startswith("variation_assets/"):
                        resolved_path = asset_str
                    else:
                        resolved_path = f"variation_assets/{asset_str}"

            objects.append({
                "id": name,
                "name": name,
                "category": category,
                "description": description,
                "asset": {
                    "path": resolved_path,
                    "license": license_name,
                    "commercial_ok": True,
                },
            })

        payload = {
            "scene_id": self.scene_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source": "variation-asset-pipeline",
            "objects": objects,
        }
        _safe_write_text(
            output_path,
            json.dumps(payload, indent=2),
            context="variation assets export manifest",
        )

    def _generate_placement_regions(self, manifest: Dict) -> str:
        """Generate minimal placement_regions.usda."""
        lines = [
            '#usda 1.0',
            '(',
            '    defaultPrim = "PlacementRegions"',
            ')',
            '',
            'def Xform "PlacementRegions" {',
        ]

        # Find surfaces for placement
        for obj in manifest.get("objects", []):
            category = (obj.get("category") or "").lower()
            if category in ["counter", "table", "shelf", "desk"]:
                obj_id = obj.get("id", "unknown")
                pos = obj.get("transform", {}).get("position", {})
                dims = obj.get("dimensions_est", {})

                # Create a placement region on top of this surface
                height = dims.get("height", 0.9)
                width = dims.get("width", 0.6)
                depth = dims.get("depth", 0.6)

                lines.append(f'    def Xform "region_{obj_id}" {{')
                lines.append(f'        double3 xformOp:translate = ({pos.get("x", 0)}, {pos.get("y", 0) + height}, {pos.get("z", 0)})')
                lines.append(f'        uniform token[] xformOpOrder = ["xformOp:translate"]')
                lines.append(f'        custom float3 extent = ({width * 0.8}, 0.02, {depth * 0.8})')
                lines.append(f'        custom string region_type = "placement_surface"')
                lines.append(f'    }}')

        lines.append('}')
        return '\n'.join(lines)

    def _generate_variation_manifest(self, manifest: Dict, inventory: Dict) -> Dict:
        """Generate variation asset manifest."""
        assets = []

        # Identify what variation assets might be needed based on scene
        for obj in inventory.get("objects", []):
            if obj.get("sim_role") in ["manipulable_object", "clutter"]:
                category = obj.get("category", "object")
                assets.append({
                    "name": f"variation_{category}",
                    "category": category,
                    "description": f"Variation of {category}",
                    "priority": "recommended",
                    "source_hint": "generate",
                })

        if not assets:
            for obj in manifest.get("objects", []):
                if obj.get("sim_role") in {"background", "scene_shell"}:
                    continue
                category = (obj.get("category")
                            or (obj.get("semantics") or {}).get("category")
                            or "object")
                assets.append({
                    "name": f"variation_{category}",
                    "category": category,
                    "description": f"Variation of {category}",
                    "priority": "recommended",
                    "source_hint": "generate",
                })
                if len(assets) >= 3:
                    break

        return {
            "scene_id": self.scene_id,
            "assets": assets,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    def _run_isaac_lab(self) -> StepResult:
        """Run Isaac Lab task generation."""
        try:
            from tools.isaac_lab_tasks.task_generator import IsaacLabTaskGenerator, TaskConfig

            # Load policy config
            policy_config_path = REPO_ROOT / "policy_configs" / "environment_policies.json"
            if policy_config_path.is_file():
                policy_config = _load_json(policy_config_path, "Isaac Lab policy config")
            else:
                policy_config = {"policies": {}, "environments": {}}

            # Load manifest as recipe
            manifest_path = self.assets_dir / "scene_manifest.json"
            if not manifest_path.is_file():
                return StepResult(
                    step=PipelineStep.ISAAC_LAB,
                    success=False,
                    duration_seconds=0,
                    message="Manifest not found",
                )

            manifest = _load_json(manifest_path, "Isaac Lab manifest")

            # Build recipe from manifest
            recipe = {
                "metadata": {
                    "environment_type": self.environment_type,
                    "scene_path": str(self.usd_dir / "scene.usda"),
                },
                "room": manifest.get("scene", {}).get("room", {}),
                "objects": manifest.get("objects", []),
            }

            # Generate task
            generator = IsaacLabTaskGenerator(policy_config)

            # Select policy based on environment
            policy_id = "manipulation"  # Default
            if self.environment_type == "kitchen":
                policy_id = "dish_loading"
            elif self.environment_type == "warehouse":
                policy_id = "pick_place"

            task = generator.generate(
                recipe=recipe,
                policy_id=policy_id,
                robot_type="franka",
                num_envs=1024,
            )

            # Save task files
            saved_files = generator.save(task, str(self.isaac_lab_dir))
            marker_path = self.isaac_lab_dir / ".isaac_lab_complete"
            self._write_marker(marker_path, status="completed")

            self.log(f"Generated Isaac Lab task: {task.task_name}")
            self.log(f"Files: {list(saved_files.keys())}")

            return StepResult(
                step=PipelineStep.ISAAC_LAB,
                success=True,
                duration_seconds=0,
                message=f"Generated task: {task.task_name}",
                outputs={"files": list(saved_files.keys()), "completion_marker": str(marker_path)},
            )

        except ImportError as e:
            return StepResult(
                step=PipelineStep.ISAAC_LAB,
                success=False,
                duration_seconds=0,
                message=f"Import error: {self._summarize_exception(e)}",
            )
        except Exception as e:
            return self._handle_step_exception(
                PipelineStep.ISAAC_LAB,
                e,
                "Isaac Lab task generation failed",
            )

    def _run_geniesim_export(self) -> StepResult:
        """Run Genie Sim export job locally."""
        start_time = time.time()
        try:
            sys.path.insert(0, str(REPO_ROOT / "genie-sim-export-job"))
            from export_to_geniesim import run_geniesim_export_job
        except ImportError as e:
            return StepResult(
                step=PipelineStep.GENIESIM_EXPORT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Import error (Genie Sim export job not found): {self._summarize_exception(e)}",
            )

        manifest_path = self.assets_dir / "scene_manifest.json"
        try:
            if not manifest_path.is_file():
                raise NonRetryableError("Manifest not found - run regen3d step first")
        except NonRetryableError as exc:
            return StepResult(
                step=PipelineStep.GENIESIM_EXPORT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=str(exc),
            )

        root = self.scene_dir.parent
        assets_prefix = f"{self.scene_id}/assets"
        geniesim_prefix = f"{self.scene_id}/geniesim"
        variation_assets_prefix = f"{self.scene_id}/variation_assets"
        replicator_prefix = f"{self.scene_id}/replicator"
        robot_type = os.getenv("GENIESIM_ROBOT_TYPE", "franka")
        os.environ.setdefault("VARIATION_ASSETS_PREFIX", variation_assets_prefix)

        def _export_job() -> None:
            exit_code = run_geniesim_export_job(
                root=root,
                scene_id=self.scene_id,
                assets_prefix=assets_prefix,
                geniesim_prefix=geniesim_prefix,
                robot_type=robot_type,
                variation_assets_prefix=variation_assets_prefix,
                replicator_prefix=replicator_prefix,
                copy_usd=True,
            )
            if exit_code != 0:
                raise RetryableError(f"Genie Sim export failed with exit code {exit_code}")

        try:
            self._run_with_retry(PipelineStep.GENIESIM_EXPORT, _export_job)
        except NonRetryableError as exc:
            return StepResult(
                step=PipelineStep.GENIESIM_EXPORT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=str(exc),
            )
        except Exception as exc:
            return self._handle_step_exception(
                PipelineStep.GENIESIM_EXPORT,
                exc,
                "Genie Sim export failed",
                duration_seconds=time.time() - start_time,
            )

        return StepResult(
            step=PipelineStep.GENIESIM_EXPORT,
            success=True,
            duration_seconds=time.time() - start_time,
            message="Genie Sim export completed",
            outputs={
                "geniesim_dir": str(self.geniesim_dir),
                "scene_graph": str(self.geniesim_dir / "scene_graph.json"),
                "task_config": str(self.geniesim_dir / "task_config.json"),
            },
        )

    def _run_geniesim_submit(self) -> StepResult:
        """Submit Genie Sim generation (API or local framework)."""
        start_time = time.time()
        try:
            from tools.geniesim_adapter.local_framework import (
                format_geniesim_preflight_failure,
                run_geniesim_preflight,
                run_local_data_collection,
            )
        except ImportError as e:
            return StepResult(
                step=PipelineStep.GENIESIM_SUBMIT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Import error (Genie Sim submit dependencies not found): {self._summarize_exception(e)}",
            )

        scene_graph_path = self.geniesim_dir / "scene_graph.json"
        asset_index_path = self.geniesim_dir / "asset_index.json"
        task_config_path = self.geniesim_dir / "task_config.json"
        try:
            if not scene_graph_path.is_file() or not asset_index_path.is_file() or not task_config_path.is_file():
                raise NonRetryableError(
                    "Genie Sim export outputs missing - run genie-sim-export first"
                )
            try:
                validate_export_consistency(
                    scene_graph_path=scene_graph_path,
                    asset_index_path=asset_index_path,
                    task_config_path=task_config_path,
                )
            except ExportConsistencyError as exc:
                raise NonRetryableError(str(exc)) from exc
        except NonRetryableError as exc:
            return StepResult(
                step=PipelineStep.GENIESIM_SUBMIT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=str(exc),
            )

        scene_graph = _load_json(scene_graph_path, "Genie Sim scene graph")
        asset_index = _load_json(asset_index_path, "Genie Sim asset index")
        task_config = _load_json(task_config_path, "Genie Sim task config")

        pipeline_config = load_pipeline_config()
        robot_types = self._resolve_geniesim_robot_types()
        if not robot_types:
            raise NonRetryableError("No robot types configured")
        robot_type = robot_types[0]
        multi_robot = len(robot_types) > 1
        episodes_per_task_env = os.getenv("EPISODES_PER_TASK")
        episodes_per_task_source = (
            "environment variable EPISODES_PER_TASK"
            if episodes_per_task_env not in (None, "")
            else "pipeline config episode_generation.episodes_per_task"
        )
        episodes_per_task = parse_int_env(
            episodes_per_task_env if episodes_per_task_env not in (None, "") else None,
            default=pipeline_config.episode_generation.episodes_per_task,
            min_value=1,
            name="EPISODES_PER_TASK",
        )
        self.log(
            f"Resolved episodes_per_task={episodes_per_task} from {episodes_per_task_source}.",
            "INFO",
        )
        num_variations = parse_int_env(
            os.getenv("NUM_VARIATIONS"),
            default=5,
            min_value=1,
            name="NUM_VARIATIONS",
        )
        quality_settings = resolve_quality_settings()
        min_quality_score = quality_settings.min_quality_score
        quality_thresholds = build_quality_thresholds(
            min_quality_score=min_quality_score,
            filter_low_quality=quality_settings.filter_low_quality,
        )
        def _parse_timeout_env(env_var: str) -> Optional[float]:
            raw_value = os.getenv(env_var)
            if raw_value in (None, ""):
                return None
            try:
                timeout_value = float(raw_value)
            except ValueError as exc:
                raise NonRetryableError(
                    f"{env_var} must be a number of seconds; got {raw_value!r}."
                ) from exc
            if timeout_value <= 0:
                raise NonRetryableError(
                    f"{env_var} must be greater than 0 seconds; got {raw_value!r}."
                )
            return timeout_value

        step_timeout_seconds = self._step_timeouts.get(PipelineStep.GENIESIM_SUBMIT)
        collection_timeout_seconds = _parse_timeout_env("GENIESIM_COLLECTION_TIMEOUT_S")
        submit_timeout_seconds = _parse_timeout_env("GENIESIM_SUBMIT_TIMEOUT_S")
        if submit_timeout_seconds is None and step_timeout_seconds is not None:
            submit_timeout_seconds = step_timeout_seconds
        if collection_timeout_seconds is None and step_timeout_seconds is not None:
            collection_timeout_seconds = step_timeout_seconds

        idempotency_payload = build_geniesim_idempotency_inputs(
            scene_id=self.scene_id,
            task_config=task_config,
            robot_types=robot_types,
            episodes_per_task=episodes_per_task,
            quality_thresholds=quality_thresholds,
        )
        idempotency_key = hashlib.sha256(
            json.dumps(idempotency_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        job_path = self.geniesim_dir / "job.json"
        idempotency_path = self.geniesim_dir / "idempotency.json"
        submission_marker = self.geniesim_dir / ".geniesim_submitted"

        def _write_idempotency_payload() -> None:
            idempotency_payload["idempotency_key"] = idempotency_key
            _safe_write_text(
                idempotency_path,
                json.dumps(idempotency_payload, indent=2, sort_keys=True),
                context="geniesim idempotency payload",
            )
        if job_path.is_file():
            prior_job_payload = _load_json(job_path, "Genie Sim job payload")
            prior_key = prior_job_payload.get("idempotency_key")
            if not prior_key and idempotency_path.is_file():
                prior_idempotency = _load_json(idempotency_path, "Genie Sim idempotency payload")
                prior_key = prior_idempotency.get("idempotency_key")
            prior_status = prior_job_payload.get("status")
            if prior_key == idempotency_key and prior_status == "completed":
                self.log(
                    "Found completed Genie Sim job with matching idempotency key; reusing outputs.",
                    "INFO",
                )
                self._write_marker(
                    submission_marker,
                    status="submitted",
                    payload={
                        "job_id": prior_job_payload.get("job_id"),
                        "run_id": str(prior_job_payload.get("run_id") or self.run_id),
                        "job_status": prior_status,
                    },
                )
                return StepResult(
                    step=PipelineStep.GENIESIM_SUBMIT,
                    success=True,
                    duration_seconds=time.time() - start_time,
                    message="Reusing completed Genie Sim job outputs.",
                    outputs={
                        "job_id": prior_job_payload.get("job_id"),
                        "job_status": prior_status,
                        "job_payload": str(job_path),
                        "preflight_report_path": prior_job_payload.get("preflight_report_path"),
                    },
                )
            if prior_key == idempotency_key and prior_status == "failed":
                self.log(
                    "Prior Genie Sim job failed with matching idempotency key; rerunning.",
                    "WARNING",
                )

        job_id = None
        submission_message = None
        local_run_results: Dict[str, Any] = {}
        local_run_ends: Dict[str, datetime] = {}
        preflight_report = None
        failure_details: Dict[str, Any] = {}
        failure_reason = None
        firebase_upload_status = "skipped"
        # Default to False: never delete completed episode data just because
        # a later task failed.  The output directory contains per-task
        # checkpoints (_completed_tasks.json) that allow retries to resume
        # from where they left off.  Cleaning up destroys that data.
        cleanup_enabled = parse_bool_env(
            os.getenv("GENIESIM_CLEANUP_ON_FAILURE"),
            default=False,
        )
        cleanup_details: Dict[str, Any] = {"enabled": cleanup_enabled}
        import uuid

        job_id = f"local-{uuid.uuid4()}"
        submission_message = "Local Genie Sim execution started."
        submit_step_start = time.time()
        preflight_report = run_geniesim_preflight(
            "genie-sim-local-runner",
            require_server=True,
        )
        preflight_report_path = self.geniesim_dir / "preflight_report.json"
        _safe_write_text(
            preflight_report_path,
            json.dumps(preflight_report, indent=2),
            context="geniesim preflight report",
        )
        try:
            if not preflight_report.get("ok", False):
                raise NonRetryableError(
                    format_geniesim_preflight_failure(
                        "genie-sim-local-runner",
                        preflight_report,
                    )
                )
        except NonRetryableError as exc:
            return StepResult(
                step=PipelineStep.GENIESIM_SUBMIT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=str(exc),
                outputs={
                    "preflight": preflight_report,
                    "preflight_report_path": str(preflight_report_path),
                },
            )

        merged_manifest_path = self.geniesim_dir / "merged_scene_manifest.json"
        if merged_manifest_path.is_file():
            scene_manifest = _load_json(merged_manifest_path, "Genie Sim merged scene manifest")
        else:
            scene_manifest = {"scene_graph": scene_graph}
        # Ensure scene graph nodes are available for local runs (object metadata + aliases)
        if "nodes" not in scene_manifest:
            _sg = scene_manifest.get("scene_graph") or scene_graph
            if isinstance(_sg, dict) and _sg.get("nodes"):
                scene_manifest["nodes"] = _sg.get("nodes")
        # Propagate meters_per_unit if missing at top-level
        if "meters_per_unit" not in scene_manifest:
            _sg = scene_manifest.get("scene_graph") or scene_graph
            if isinstance(_sg, dict) and _sg.get("meters_per_unit") is not None:
                scene_manifest["meters_per_unit"] = _sg.get("meters_per_unit")

        output_dirs: Dict[str, Path] = {}
        robot_failures: Dict[str, Dict[str, Any]] = {}
        for robot_index, current_robot in enumerate(robot_types):
            remaining_budget_seconds = None
            if submit_timeout_seconds is not None:
                elapsed_seconds = time.time() - submit_step_start
                remaining_budget_seconds = submit_timeout_seconds - elapsed_seconds
                if remaining_budget_seconds <= 0:
                    timeout_message = (
                        "Genie Sim submit timed out after "
                        f"{elapsed_seconds:.1f}s (budget {submit_timeout_seconds:.1f}s, "
                        "including preflight) "
                        f"before starting robot {current_robot}."
                    )
                    self.log(timeout_message, "ERROR")
                    failure_reason = "Genie Sim submit timed out"
                    failure_details["timeout"] = {
                        "message": timeout_message,
                        "budget_seconds": submit_timeout_seconds,
                        "elapsed_seconds": elapsed_seconds,
                        "robot": current_robot,
                    }
                    for skipped_robot in robot_types[robot_index:]:
                        robot_failures[skipped_robot] = {
                            "error": timeout_message,
                            "type": "total_timeout",
                        }
                        local_run_results.setdefault(skipped_robot, None)
                    break
            if multi_robot:
                output_dir = self.episodes_dir / current_robot / f"geniesim_{job_id}"
            else:
                output_dir = self.episodes_dir / f"geniesim_{job_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dirs[current_robot] = output_dir
            config_dir = output_dir / "config"
            config_dir.mkdir(parents=True, exist_ok=True)

            scene_manifest_path = config_dir / "scene_manifest.json"
            task_config_local_path = config_dir / "task_config.json"
            _safe_write_text(
                scene_manifest_path,
                json.dumps(scene_manifest, indent=2),
                context="geniesim scene manifest",
            )
            _safe_write_text(
                task_config_local_path,
                json.dumps(task_config, indent=2),
                context="geniesim task config",
            )

            effective_max_duration_seconds = collection_timeout_seconds
            if remaining_budget_seconds is not None and remaining_budget_seconds > 0:
                effective_max_duration_seconds = (
                    remaining_budget_seconds
                    if effective_max_duration_seconds is None
                    else min(effective_max_duration_seconds, remaining_budget_seconds)
                )

            def _collect_data(
                scene_manifest_path: Path = scene_manifest_path,
                task_config_local_path: Path = task_config_local_path,
                output_dir: Path = output_dir,
                current_robot: str = current_robot,
                max_duration_seconds: Optional[float] = effective_max_duration_seconds,
            ) -> Any:
                result = run_local_data_collection(
                    scene_manifest_path=scene_manifest_path,
                    task_config_path=task_config_local_path,
                    output_dir=output_dir,
                    robot_type=current_robot,
                    episodes_per_task=episodes_per_task,
                    max_duration_seconds=max_duration_seconds,
                    verbose=True,
                )
                if getattr(result, "timed_out", False):
                    self.log(
                        f"Genie Sim data collection timed out for robot {current_robot}.",
                        "WARNING",
                    )
                if bool(getattr(result, "fatal_realism_failure", False)):
                    _fatal_code = str(
                        getattr(result, "fatal_realism_code", None)
                        or "STRICT_REALISM_FAILURE"
                    )
                    _fatal_message = str(
                        getattr(result, "fatal_realism_message", None)
                        or "strict realism violation"
                    )
                    raise NonRetryableError(
                        f"Fatal realism violation ({_fatal_code}): {_fatal_message}"
                    )
                if not result or not result.success:
                    raise RetryableError("Local Genie Sim execution failed")
                return result

            try:
                local_run_results[current_robot] = self._run_with_retry(
                    PipelineStep.GENIESIM_SUBMIT,
                    _collect_data,
                )
                local_run_ends[current_robot] = datetime.utcnow()
            except NonRetryableError as exc:
                robot_failures[current_robot] = {"error": str(exc)}
                local_run_results[current_robot] = None
            except Exception as exc:
                robot_failures[current_robot] = {"error": self._summarize_exception(exc)}
                local_run_results[current_robot] = None

        if robot_failures:
            failure_details["by_robot"] = robot_failures
            if failure_reason is None:
                failure_reason = "Local Genie Sim execution failed"
            cleanup_details["local"] = self._merge_cleanup_reports(
                cleanup_details.get("local"),
                self._cleanup_local_output_dirs(
                    output_dirs,
                    list(robot_failures.keys()),
                    enabled=cleanup_enabled,
                ),
            )
            failure_details["cleanup"] = cleanup_details
        submission_message = (
            "Local Genie Sim execution completed."
            if all(result and result.success for result in local_run_results.values())
            else "Local Genie Sim execution failed."
        )

        job_status = (
            "completed"
            if all(result and result.success for result in local_run_results.values())
            else "failed"
        )
        if preflight_report and not preflight_report.get("ok", False):
            job_status = "failed"

        job_payload = {
            "job_id": job_id,
            "scene_id": self.scene_id,
            "run_id": self.run_id,
            "status": job_status,
            "submitted_at": datetime.utcnow().isoformat() + "Z",
            "message": submission_message,
            "idempotency_key": idempotency_key,
            "preflight_report_path": str(preflight_report_path),
            "bundle": {
                "scene_graph": str(scene_graph_path),
                "asset_index": str(asset_index_path),
                "task_config": str(task_config_path),
            },
            "generation_params": {
                "robot_type": robot_type,
                "robot_types": robot_types,
                "episodes_per_task": episodes_per_task,
                "num_variations": num_variations,
                "min_quality_score": min_quality_score,
            },
            "quality_config": {
                "min_quality_score": min_quality_score,
                "filter_low_quality": quality_settings.filter_low_quality,
                "range": {
                    "min_allowed": quality_settings.config.min_allowed,
                    "max_allowed": quality_settings.config.max_allowed,
                },
                "defaults": {
                    "min_quality_score": quality_settings.config.default_min_quality_score,
                    "filter_low_quality": quality_settings.config.default_filter_low_quality,
                },
                "source_path": quality_settings.config.source_path,
            },
        }
        task_count = len(task_config.get("suggested_tasks", task_config.get("tasks", []))) or 1
        total_episodes_per_robot = max(1, episodes_per_task * task_count)
        job_metrics_by_robot: Dict[str, Any] = {}
        robot_failure_details = failure_details.get("by_robot", {})
        for current_robot in robot_types:
            result = local_run_results.get(current_robot)
            episodes_collected = getattr(result, "episodes_collected", 0) if result else 0
            episodes_passed = getattr(result, "episodes_passed", 0) if result else 0
            job_metrics_by_robot[current_robot] = {
                "job_id": job_id,
                "status": "completed" if result and result.success else "failed",
                "created_at": job_payload["submitted_at"],
                "completed_at": (
                    local_run_ends[current_robot].isoformat() + "Z"
                    if current_robot in local_run_ends
                    else None
                ),
                "total_episodes": total_episodes_per_robot,
                "episodes_collected": episodes_collected,
                "episodes_passed": episodes_passed,
                "quality_pass_rate": (
                    episodes_passed / episodes_collected if episodes_collected else None
                ),
                "failure_reason": failure_reason if current_robot in robot_failure_details else None,
                "failure_details": robot_failure_details.get(current_robot),
            }
        job_payload["job_metrics_by_robot"] = job_metrics_by_robot
        completed_times = [
            metrics.get("completed_at")
            for metrics in job_metrics_by_robot.values()
            if metrics.get("completed_at")
        ]
        completed_at = max(completed_times) if completed_times else None
        job_payload["job_metrics_summary"] = {
            "job_id": job_id,
            "status": job_status,
            "created_at": job_payload["submitted_at"],
            "completed_at": completed_at,
            "duration_seconds": time.time() - start_time,
            "total_episodes": total_episodes_per_robot * len(robot_types),
            "episodes_collected": sum(
                metrics.get("episodes_collected", 0) for metrics in job_metrics_by_robot.values()
            ),
            "episodes_passed": sum(
                metrics.get("episodes_passed", 0) for metrics in job_metrics_by_robot.values()
            ),
            "quality_pass_rate": (
                (
                    sum(
                        metrics.get("episodes_passed", 0)
                        for metrics in job_metrics_by_robot.values()
                    )
                    / sum(
                        metrics.get("episodes_collected", 0)
                        for metrics in job_metrics_by_robot.values()
                    )
                )
                if sum(
                    metrics.get("episodes_collected", 0)
                    for metrics in job_metrics_by_robot.values()
                )
                else None
            ),
        }
        if not multi_robot:
            job_payload["job_metrics"] = job_metrics_by_robot[robot_type]

        allow_partial_firebase_uploads = parse_bool_env(
            os.getenv("ALLOW_PARTIAL_FIREBASE_UPLOADS"),
            default=False,
        )
        fail_on_partial_error = parse_bool_env(
            os.getenv("FAIL_ON_PARTIAL_ERROR"),
            default=False,
        )
        production_mode = resolve_production_mode()
        require_lerobot = parse_bool_env(os.getenv("REQUIRE_LEROBOT"), default=False)
        allow_partial_failures = parse_bool_env(
            os.getenv("ALLOW_PARTIAL_FAILURES"), default=False
        )
        require_balanced_robot_episodes = parse_bool_env(
            os.getenv("REQUIRE_BALANCED_ROBOT_EPISODES"),
            default=False,
        )
        artifact_validation: Dict[str, Any] = {
            "require_lerobot": require_lerobot,
            "allow_partial_failures": allow_partial_failures,
            "require_balanced_robot_episodes": require_balanced_robot_episodes,
            "by_robot": {},
        }
        artifacts_valid = True
        artifact_validation_status = "passed"
        artifact_structure_by_robot: Dict[str, Any] = {}
        for current_robot, output_dir in output_dirs.items():
            recordings_dir = output_dir / "recordings"
            lerobot_dir = output_dir / "lerobot"
            dataset_info_path = lerobot_dir / "dataset_info.json"
            missing_paths = []
            episode_files: List[Path] = []
            if not recordings_dir.is_dir():
                missing_paths.append(str(recordings_dir))
            else:
                episode_files = list(recordings_dir.rglob("*.json"))
                if not episode_files:
                    missing_paths.append(f"{recordings_dir}/**/*.json")
            if require_lerobot:
                if not lerobot_dir.is_dir():
                    missing_paths.append(str(lerobot_dir))
                if not dataset_info_path.is_file():
                    missing_paths.append(str(dataset_info_path))
            validation_errors = []
            lerobot_validation_errors: List[str] = []
            if lerobot_dir.is_dir():
                lerobot_validation_errors = validate_lerobot_dataset(lerobot_dir)
                if lerobot_validation_errors:
                    validation_errors.append(
                        {
                            "type": "lerobot_validation",
                            "errors": lerobot_validation_errors,
                        }
                    )
            expected_episodes = job_metrics_by_robot.get(current_robot, {}).get(
                "episodes_collected"
            )
            actual_episodes = len(episode_files)
            has_recordings_dir = recordings_dir.is_dir()
            has_episode_files = bool(episode_files)
            has_lerobot_dir = lerobot_dir.is_dir()
            has_dataset_info = dataset_info_path.is_file()
            count_mismatch = (
                expected_episodes is not None and expected_episodes != actual_episodes
            )
            artifact_structure_by_robot[current_robot] = {
                "output_dir": str(output_dir),
                "recordings_dir": str(recordings_dir),
                "lerobot_dir": str(lerobot_dir),
                "dataset_info_path": str(dataset_info_path),
                "has_recordings_dir": has_recordings_dir,
                "has_episode_files": has_episode_files,
                "has_lerobot_dir": has_lerobot_dir,
                "has_dataset_info": has_dataset_info,
                "episodes_expected": expected_episodes,
                "episodes_found": actual_episodes,
            }
            if missing_paths:
                validation_errors.append(
                    {
                        "type": "missing_paths",
                        "paths": missing_paths,
                    }
                )
            if count_mismatch:
                validation_errors.append(
                    {
                        "type": "episode_count_mismatch",
                        "expected": expected_episodes,
                        "found": actual_episodes,
                    }
                )
            artifact_validation["by_robot"][current_robot] = {
                "status": "passed" if not validation_errors else "failed",
                "output_dir": str(output_dir),
                "recordings_dir": str(recordings_dir),
                "lerobot_dir": str(lerobot_dir),
                "dataset_info_path": str(dataset_info_path),
                "episodes_found": actual_episodes,
                "episodes_expected": expected_episodes,
                "errors": validation_errors,
            }
            if validation_errors:
                artifacts_valid = False
                artifact_validation_status = "failed"
                job_metrics_by_robot[current_robot]["status"] = "failed"
                failure_details.setdefault("by_robot", {})
                failure_details["by_robot"].setdefault(current_robot, {})
                failure_details["by_robot"][current_robot]["artifact_validation"] = {
                    "errors": validation_errors,
                    "episodes_found": actual_episodes,
                    "episodes_expected": expected_episodes,
                    "missing_paths": missing_paths,
                    "require_lerobot": require_lerobot,
                    "lerobot_errors": lerobot_validation_errors,
                }
                failure_details.setdefault("artifact_validation", {})
                failure_details["artifact_validation"].setdefault("by_robot", {})
                failure_details["artifact_validation"]["by_robot"][current_robot] = {
                    "errors": lerobot_validation_errors,
                    "output_dir": str(output_dir),
                    "lerobot_dir": str(lerobot_dir),
                }
                if not job_metrics_by_robot[current_robot].get("failure_reason"):
                    job_metrics_by_robot[current_robot][
                        "failure_reason"
                    ] = "Artifact validation failed"
                job_metrics_by_robot[current_robot][
                    "failure_details"
                ] = failure_details["by_robot"][current_robot]

        expected_episode_counts = {
            details["episodes_expected"]
            for details in artifact_structure_by_robot.values()
            if details["episodes_expected"] is not None
        }
        canonical_expected_episodes = (
            expected_episode_counts.pop() if len(expected_episode_counts) == 1 else None
        )
        lerobot_required = require_lerobot or any(
            details["has_lerobot_dir"] for details in artifact_structure_by_robot.values()
        )
        dataset_info_required = require_lerobot or any(
            details["has_dataset_info"] for details in artifact_structure_by_robot.values()
        )
        canonical_schema = {
            "required_subpaths": ["recordings"],
            "lerobot_required": lerobot_required,
            "dataset_info_required": dataset_info_required,
            "expected_episodes": canonical_expected_episodes,
        }
        cross_robot_validation: Dict[str, Any] = {
            "status": "passed",
            "canonical_schema": canonical_schema,
            "by_robot": {},
            "errors": [],
        }
        if len(expected_episode_counts) > 1:
            cross_robot_validation["errors"].append(
                {
                    "type": "expected_episode_counts_inconsistent",
                    "expected_counts": {
                        robot: details["episodes_expected"]
                        for robot, details in artifact_structure_by_robot.items()
                    },
                }
            )
        for current_robot, details in artifact_structure_by_robot.items():
            discrepancies = []
            if not details["has_recordings_dir"]:
                discrepancies.append(
                    {
                        "type": "missing_required_path",
                        "path": details["recordings_dir"],
                    }
                )
            if not details["has_episode_files"]:
                discrepancies.append(
                    {
                        "type": "missing_episodes",
                        "path": f"{details['recordings_dir']}/**/*.json",
                    }
                )
            if lerobot_required and not details["has_lerobot_dir"]:
                discrepancies.append(
                    {
                        "type": "missing_lerobot_dir",
                        "path": details["lerobot_dir"],
                    }
                )
            if dataset_info_required and not details["has_dataset_info"]:
                discrepancies.append(
                    {
                        "type": "missing_dataset_info",
                        "path": details["dataset_info_path"],
                    }
                )
            if canonical_expected_episodes is not None:
                if details["episodes_expected"] is None:
                    discrepancies.append(
                        {
                            "type": "missing_expected_episode_count",
                            "expected": canonical_expected_episodes,
                        }
                    )
                elif details["episodes_expected"] != canonical_expected_episodes:
                    discrepancies.append(
                        {
                            "type": "expected_episode_count_mismatch",
                            "expected": canonical_expected_episodes,
                            "found": details["episodes_expected"],
                        }
                    )
                if details["episodes_found"] != canonical_expected_episodes:
                    discrepancies.append(
                        {
                            "type": "episode_count_mismatch",
                            "expected": canonical_expected_episodes,
                            "found": details["episodes_found"],
                        }
                    )
            cross_robot_validation["by_robot"][current_robot] = {
                "schema": {
                    "episodes_path": details["output_dir"],
                    "recordings_path": details["recordings_dir"],
                    "lerobot_path": details["lerobot_dir"],
                    "dataset_info_path": details["dataset_info_path"],
                },
                "episodes_found": details["episodes_found"],
                "episodes_expected": details["episodes_expected"],
                "discrepancies": discrepancies,
            }
            if discrepancies:
                cross_robot_validation["status"] = "failed"

        mismatch_types = {
            "expected_episode_counts_inconsistent",
            "expected_episode_count_mismatch",
            "episode_count_mismatch",
        }
        cross_robot_episode_mismatch = any(
            error.get("type") in mismatch_types
            for error in cross_robot_validation.get("errors", [])
        ) or any(
            discrepancy.get("type") in mismatch_types
            for details in cross_robot_validation.get("by_robot", {}).values()
            for discrepancy in details.get("discrepancies", [])
        )

        if cross_robot_validation["errors"] or cross_robot_validation["status"] == "failed":
            if require_balanced_robot_episodes and cross_robot_episode_mismatch:
                artifacts_valid = False
                artifact_validation_status = "failed"
                failure_details.setdefault("artifact_validation", {})
                failure_details["artifact_validation"]["cross_robot"] = cross_robot_validation
                if not job_payload.get("failure_reason"):
                    job_payload["failure_reason"] = (
                        "Cross-robot episode counts mismatch"
                    )
                job_payload["failure_details"] = failure_details.get("artifact_validation")
            elif allow_partial_failures:
                if artifact_validation_status == "passed":
                    artifact_validation_status = "warning"
                self.log(
                    "Cross-robot artifact validation found inconsistencies; "
                    "continuing due to ALLOW_PARTIAL_FAILURES.",
                    "WARNING",
                )
            else:
                artifacts_valid = False
                artifact_validation_status = "failed"
                failure_details.setdefault("artifact_validation", {})
                failure_details["artifact_validation"]["cross_robot"] = cross_robot_validation
                if not job_payload.get("failure_reason"):
                    job_payload["failure_reason"] = "Cross-robot artifact validation failed"
                job_payload["failure_details"] = failure_details.get("artifact_validation")

        artifact_validation["cross_robot"] = cross_robot_validation
        artifact_validation["status"] = artifact_validation_status
        job_payload["artifact_validation"] = artifact_validation
        job_payload["job_metrics_summary"]["artifact_validation"] = {
            "status": artifact_validation_status,
            "cross_robot": cross_robot_validation,
        }
        job_payload["firebase_upload_policy"] = {
            "allow_partial_uploads": allow_partial_firebase_uploads,
            "suppress_on_partial_failure": production_mode or fail_on_partial_error,
            "production_mode": production_mode,
            "fail_on_partial_error": fail_on_partial_error,
        }
        if not artifacts_valid:
            job_status = "failed"
            job_payload["status"] = job_status
            job_payload["job_metrics_summary"]["status"] = job_status

        artifacts_by_robot = {}
        for current_robot, output_dir in output_dirs.items():
            artifacts_by_robot[current_robot] = {
                "episodes_path": str(output_dir),
                "episodes_prefix": str(output_dir),
                "lerobot_path": str(output_dir / "lerobot"),
                "lerobot_prefix": str(output_dir / "lerobot"),
            }
        if multi_robot:
            job_payload["artifacts_by_robot"] = artifacts_by_robot
            job_payload["artifacts"] = artifacts_by_robot[robot_type]
        else:
            job_payload["artifacts"] = artifacts_by_robot[robot_type]

        episodes_collected_total = sum(
            metrics.get("episodes_collected", 0) for metrics in job_metrics_by_robot.values()
        )
        episodes_passed_total = sum(
            metrics.get("episodes_passed", 0) for metrics in job_metrics_by_robot.values()
        )
        local_execution = {
            "success": job_status == "completed",
            "episodes_collected": episodes_collected_total,
            "episodes_passed": episodes_passed_total,
            "preflight": preflight_report,
            "generation_duration_seconds": time.time() - start_time,
        }
        if local_run_results:
            local_execution["by_robot"] = {
                current_robot: {
                    "success": bool(result and result.success),
                    "episodes_collected": getattr(result, "episodes_collected", 0) if result else 0,
                    "episodes_passed": getattr(result, "episodes_passed", 0) if result else 0,
                    "output_dir": str(output_dirs.get(current_robot)) if output_dirs else None,
                }
                for current_robot, result in local_run_results.items()
            }
        job_payload["local_execution"] = local_execution
        if local_run_results and job_status != "failed":
            firebase_upload_status = "pending"
            self.log(
                "Firebase upload will occur during import after validation.",
                "INFO",
            )
        job_payload["firebase_upload_status"] = firebase_upload_status

        job_payload["message"] = submission_message or job_payload.get("message")
        job_payload["status"] = job_status

        if job_status == "failed":
            job_payload["failure_reason"] = failure_reason or "Genie Sim submission failed"
            job_payload["failure_details"] = failure_details or None

        _write_idempotency_payload()
        _safe_write_text(
            job_path,
            json.dumps(job_payload, indent=2),
            context="geniesim job payload",
        )
        if job_status != "failed":
            self._write_marker(
                submission_marker,
                status="submitted",
                payload={
                    "job_id": job_id,
                    "run_id": self.run_id,
                    "job_status": job_status,
                },
            )
        try:
            export_job_metrics(job_json_path=job_path)
        except Exception as exc:
            self.log(
                f"Failed to export Genie Sim job metrics summary: {exc}",
                "WARNING",
            )
        self._geniesim_local_run_results = local_run_results
        self._geniesim_output_dirs = output_dirs

        return StepResult(
            step=PipelineStep.GENIESIM_SUBMIT,
            success=job_status != "failed",
            duration_seconds=time.time() - start_time,
            message=submission_message or "Genie Sim submission completed",
            outputs={
                "job_id": job_id,
                "job_status": job_status,
                "job_payload": str(job_path),
                "preflight_report_path": str(preflight_report_path),
            },
        )

    def _run_geniesim_import(self) -> StepResult:
        """Import Genie Sim episodes into the local bundle."""
        start_time = time.time()
        try:
            sys.path.insert(0, str(REPO_ROOT / "genie-sim-import-job"))
            from import_from_geniesim import ImportConfig, run_local_import_job
        except ImportError as e:
            return StepResult(
                step=PipelineStep.GENIESIM_IMPORT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Import error (Genie Sim import job not found): {self._summarize_exception(e)}",
            )

        job_path = self.geniesim_dir / "job.json"
        try:
            if not job_path.is_file():
                raise NonRetryableError(
                    "Genie Sim job metadata missing - run genie-sim-submit first"
                )
        except NonRetryableError as exc:
            return StepResult(
                step=PipelineStep.GENIESIM_IMPORT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=str(exc),
            )

        try:
            job_payload = _load_json(job_path, "Genie Sim job payload")
            job_id = job_payload.get("job_id")
            if not job_id:
                raise NonRetryableError("Genie Sim job metadata missing job_id")
        except NonRetryableError as exc:
            return StepResult(
                step=PipelineStep.GENIESIM_IMPORT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=str(exc),
            )
        try:
            job_payload, job_status = self._poll_geniesim_job_status(job_path, job_id)
        except NonRetryableError as exc:
            return StepResult(
                step=PipelineStep.GENIESIM_IMPORT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=str(exc),
                outputs={
                    "job_id": job_id,
                    "job_status": job_payload.get("status") if job_payload else None,
                    "job_payload": str(job_path),
                },
            )
        local_execution = job_payload.get("local_execution", {}) if isinstance(job_payload, dict) else {}
        local_success = local_execution.get("success")
        if local_success is None:
            local_success = job_status == "completed"
        firebase_status = job_payload.get("firebase_upload_status")
        firebase_upload_required = parse_bool_env(
            os.getenv("FIREBASE_UPLOAD_REQUIRED"),
            default=resolve_production_mode(),
        )
        if firebase_status == "failed":
            if firebase_upload_required:
                return StepResult(
                    step=PipelineStep.GENIESIM_IMPORT,
                    success=False,
                    duration_seconds=time.time() - start_time,
                    message=(
                        "Genie Sim Firebase upload failed and is required; "
                        "aborting import."
                    ),
                )
            self.log(
                "Genie Sim Firebase upload failed; continuing import with local recordings.",
                "WARNING",
            )
        artifacts = job_payload.get("artifacts", {})
        local_episodes_prefix = (
            artifacts.get("episodes_prefix")
            or artifacts.get("episodes_path")
            or str(self.episodes_dir / f"geniesim_{job_id}")
        )

        output_dir = Path(local_episodes_prefix)
        recordings_dir = output_dir / "recordings"
        lerobot_dir = output_dir / "lerobot"
        dataset_info_path = lerobot_dir / "dataset_info.json"
        require_lerobot = parse_bool_env(os.getenv("REQUIRE_LEROBOT"), default=False)
        lerobot_validation_errors: List[str] = []

        try:
            if not local_success:
                raise NonRetryableError(
                    f"Genie Sim job status is {job_status}; import requires successful local execution"
                )
            if not recordings_dir.is_dir():
                raise NonRetryableError(
                    "Genie Sim recordings directory missing for job "
                    f"{job_id}: expected {recordings_dir}"
                )
            episode_files = list(recordings_dir.rglob("*.json"))
            if not episode_files:
                raise NonRetryableError(
                    "Genie Sim recordings missing for job "
                    f"{job_id}: expected *.json episodes under {recordings_dir}"
                )
            if require_lerobot and (not lerobot_dir.is_dir() or not dataset_info_path.is_file()):
                missing = []
                if not lerobot_dir.is_dir():
                    missing.append(str(lerobot_dir))
                if not dataset_info_path.is_file():
                    missing.append(str(dataset_info_path))
                raise NonRetryableError(
                    "Genie Sim lerobot artifacts missing for job "
                    f"{job_id}: expected {', '.join(missing)}"
                )
            if require_lerobot and lerobot_dir.is_dir():
                lerobot_validation_errors = validate_lerobot_dataset(lerobot_dir)
                if lerobot_validation_errors:
                    raise NonRetryableError(
                        "Genie Sim lerobot validation failed for job "
                        f"{job_id}: {'; '.join(lerobot_validation_errors)}"
                    )
        except NonRetryableError as exc:
            return StepResult(
                step=PipelineStep.GENIESIM_IMPORT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=str(exc),
                outputs={
                    "job_id": job_id,
                    "run_id": str(job_payload.get("run_id") or self.run_id),
                    "job_status": job_status,
                    "local_execution_success": bool(local_success),
                    "output_dir": str(output_dir),
                    "recordings_path": str(recordings_dir),
                    "lerobot_path": str(lerobot_dir),
                    "lerobot_dataset_info": str(dataset_info_path),
                    "lerobot_validation_errors": lerobot_validation_errors,
                },
            )
        quality_settings = resolve_quality_settings()
        config = ImportConfig(
            job_id=job_id,
            output_dir=output_dir,
            min_quality_score=quality_settings.min_quality_score,
            enable_validation=parse_bool_env(os.getenv("ENABLE_VALIDATION"), default=True),
            filter_low_quality=quality_settings.filter_low_quality,
            require_lerobot=require_lerobot,
            wait_for_completion=True,
            poll_interval=0,
            job_metadata_path=str(job_path),
            local_episodes_prefix=local_episodes_prefix,
        )
        def _import_job() -> Any:
            result = run_local_import_job(config, job_metadata=job_payload)
            if not result or not result.success:
                raise RetryableError("Genie Sim import failed")
            return result

        try:
            result = self._run_with_retry(PipelineStep.GENIESIM_IMPORT, _import_job)
        except NonRetryableError as exc:
            return StepResult(
                step=PipelineStep.GENIESIM_IMPORT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=str(exc),
            )
        except Exception as exc:
            return self._handle_step_exception(
                PipelineStep.GENIESIM_IMPORT,
                exc,
                "Genie Sim import failed",
                duration_seconds=time.time() - start_time,
            )

        strict_manifest_contract = self._is_production_mode() or self._is_release_path_run()
        manifest_contract_errors: List[str] = []
        if result.success and result.import_manifest_path:
            manifest_ok, manifest_contract_errors = self._validate_import_manifest_contract(
                Path(result.import_manifest_path),
                strict_release=strict_manifest_contract,
            )
            if not manifest_ok:
                message = (
                    "Import manifest contract validation failed: "
                    + "; ".join(manifest_contract_errors[:6])
                )
                if strict_manifest_contract:
                    return StepResult(
                        step=PipelineStep.GENIESIM_IMPORT,
                        success=False,
                        duration_seconds=time.time() - start_time,
                        message=message,
                        outputs={
                            "job_id": job_id,
                            "run_id": str(job_payload.get("run_id") or self.run_id),
                            "job_status": job_status,
                            "import_manifest": str(result.import_manifest_path),
                            "import_manifest_contract_errors": manifest_contract_errors,
                            "strict_release_contract": True,
                        },
                    )
                self.log(message, "WARNING")

        marker_path = None
        if result.success:
            marker_path = self.geniesim_dir / ".geniesim_import_complete"
            import_manifest_path = (
                str(result.import_manifest_path)
                if result.import_manifest_path
                else None
            )
            self._write_marker(
                marker_path,
                status="completed",
                payload={
                    "job_id": job_id,
                    "run_id": str(job_payload.get("run_id") or self.run_id),
                    "import_manifest": import_manifest_path,
                },
            )
        duration_seconds = time.time() - start_time
        local_execution = job_payload.get("local_execution", {})
        local_execution["import_duration_seconds"] = duration_seconds
        job_payload["local_execution"] = local_execution
        _safe_write_text(
            job_path,
            json.dumps(job_payload, indent=2),
            context="geniesim import job payload",
        )

        return StepResult(
            step=PipelineStep.GENIESIM_IMPORT,
            success=result.success,
            duration_seconds=duration_seconds,
            message="Genie Sim import completed" if result.success else "Genie Sim import failed",
            outputs={
                "job_id": job_id,
                "run_id": str(job_payload.get("run_id") or self.run_id),
                "import_manifest": str(result.import_manifest_path) if result.import_manifest_path else None,
                "output_dir": str(output_dir),
                "recordings_path": str(recordings_dir),
                "lerobot_path": str(lerobot_dir),
                "lerobot_dataset_info": str(dataset_info_path),
                "completion_marker": str(marker_path) if result.success else None,
                "strict_release_contract": strict_manifest_contract,
                "import_manifest_contract_errors": manifest_contract_errors,
            },
        )

    def _validate_import_manifest_contract(
        self,
        manifest_path: Path,
        *,
        strict_release: bool,
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        try:
            manifest_payload = _load_json(manifest_path, "import manifest")
        except NonRetryableError as exc:
            return False, [str(exc)]

        utils_path = REPO_ROOT / "genie-sim-import-job" / "import_manifest_utils.py"
        if not utils_path.is_file():
            errors.append(f"import_manifest_utils.py not found at {utils_path}")
            return False, errors
        try:
            spec = importlib.util.spec_from_file_location(
                "import_manifest_utils_runtime",
                utils_path,
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"unable to load spec from {utils_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:
            errors.append(
                "failed to load import manifest validator: "
                f"{self._summarize_exception(exc)}"
            )
            return False, errors

        validator = getattr(module, "validate_import_manifest_contract", None)
        if not callable(validator):
            return False, ["validate_import_manifest_contract is unavailable"]

        try:
            validation_errors = validator(
                manifest_payload,
                strict_release=strict_release,
            )
        except Exception as exc:
            return False, [f"manifest contract validator failed: {self._summarize_exception(exc)}"]

        if not isinstance(validation_errors, list):
            return False, ["manifest contract validator returned invalid result"]
        normalized_errors = [str(item) for item in validation_errors if str(item).strip()]
        return len(normalized_errors) == 0, normalized_errors

    def _run_dataset_delivery(self) -> StepResult:
        """Invoke the dataset delivery job using the current Genie Sim context."""
        start_time = time.time()
        delivery_script = REPO_ROOT / "dataset-delivery-job" / "dataset_delivery.py"
        if not delivery_script.is_file():
            return StepResult(
                step=PipelineStep.DATASET_DELIVERY,
                success=False,
                duration_seconds=time.time() - start_time,
                message="dataset-delivery-job entrypoint not found",
            )

        job_path = self.geniesim_dir / "job.json"
        if not job_path.is_file():
            return StepResult(
                step=PipelineStep.DATASET_DELIVERY,
                success=False,
                duration_seconds=time.time() - start_time,
                message="Genie Sim job metadata missing - run genie-sim-submit/import first",
            )

        try:
            job_payload = _load_json(job_path, "Genie Sim job payload")
        except NonRetryableError as exc:
            return StepResult(
                step=PipelineStep.DATASET_DELIVERY,
                success=False,
                duration_seconds=time.time() - start_time,
                message=str(exc),
            )

        job_id = str(job_payload.get("job_id") or "")
        if not job_id:
            return StepResult(
                step=PipelineStep.DATASET_DELIVERY,
                success=False,
                duration_seconds=time.time() - start_time,
                message="Genie Sim job metadata missing job_id",
            )

        artifacts = job_payload.get("artifacts", {})
        local_episodes_prefix = (
            artifacts.get("episodes_prefix")
            or artifacts.get("episodes_path")
            or str(self.episodes_dir / f"geniesim_{job_id}")
        )
        output_dir = Path(local_episodes_prefix)
        import_manifest_path = output_dir / "import_manifest.json"
        if not import_manifest_path.is_file():
            return StepResult(
                step=PipelineStep.DATASET_DELIVERY,
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Import manifest not found at {import_manifest_path}",
                outputs={"import_manifest_path": str(import_manifest_path)},
            )

        manifest_env_path = str(import_manifest_path)
        gcs_root = Path(os.getenv("GCS_MOUNT_ROOT", "/mnt/gcs")).resolve()
        try:
            relative_manifest = import_manifest_path.resolve().relative_to(gcs_root)
        except ValueError:
            relative_manifest = None
        if relative_manifest is not None:
            manifest_env_path = f"gs://{relative_manifest.as_posix()}"

        env = os.environ.copy()
        env.update({
            "SCENE_ID": self.scene_id,
            "JOB_ID": job_id,
            "IMPORT_MANIFEST_PATH": manifest_env_path,
            "BUCKET": env.get("BUCKET", "local"),
            "RUN_ID": self.run_id,
        })
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

        self.log("Running dataset-delivery-job entrypoint locally")
        proc = subprocess.run(
            [sys.executable, str(delivery_script)],
            cwd=str(REPO_ROOT),
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            return StepResult(
                step=PipelineStep.DATASET_DELIVERY,
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"dataset-delivery-job failed with exit code {proc.returncode}",
                outputs={"import_manifest_path": manifest_env_path},
            )

        marker_path = self.geniesim_dir / ".dataset_delivery_complete"
        self._write_marker(marker_path, status="completed")
        return StepResult(
            step=PipelineStep.DATASET_DELIVERY,
            success=True,
            duration_seconds=time.time() - start_time,
            message="Dataset delivery completed",
            outputs={
                "job_id": job_id,
                "import_manifest_path": manifest_env_path,
                "completion_marker": str(marker_path),
            },
        )

    def _poll_geniesim_job_status(
        self,
        job_path: Path,
        job_id: str,
        *,
        poll_interval_override: Optional[int] = None,
        poll_timeout_override: Optional[int] = None,
    ) -> tuple[Dict[str, Any], str]:
        poll_interval = poll_interval_override
        if poll_interval is None:
            poll_interval = parse_int_env(
                os.getenv("GENIESIM_IMPORT_POLL_INTERVAL"),
                default=10,
            )
        poll_interval = max(1, poll_interval)
        poll_timeout = poll_timeout_override
        if poll_timeout is None:
            poll_timeout = parse_int_env(
                os.getenv("GENIESIM_IMPORT_POLL_TIMEOUT"),
                default=3600,
            )
        poll_start = time.time()
        last_status = None
        last_payload: Dict[str, Any] = {}
        while True:
            payload = _load_json(job_path, "Genie Sim job payload")
            status = str(payload.get("status", "submitted")).strip().lower()
            last_payload = payload
            if status != last_status and status:
                self.log(
                    f"Genie Sim job {job_id} status: {status}",
                    "INFO",
                )
                last_status = status
            if status in {"completed", "failed"}:
                if status == "failed":
                    failure_reason = payload.get("failure_reason") or "unknown failure"
                    raise NonRetryableError(
                        f"Genie Sim job {job_id} failed: {failure_reason}."
                    )
                return payload, status
            if poll_timeout and (time.time() - poll_start) >= poll_timeout:
                raise NonRetryableError(
                    "Timed out waiting for Genie Sim job "
                    f"{job_id} to complete after {poll_timeout}s "
                    f"(last status: {status})."
                )
            time.sleep(poll_interval)

    def run_geniesim_import_poller(self, *, poll_interval: Optional[int] = None) -> bool:
        """Poll Genie Sim job status and trigger local import once completed."""
        self.geniesim_dir.mkdir(parents=True, exist_ok=True)
        marker_path = self.geniesim_dir / ".geniesim_import_triggered"
        completion_marker = self.geniesim_dir / ".geniesim_import_complete"
        submission_marker = self.geniesim_dir / ".geniesim_submitted"
        if submission_marker.is_file():
            try:
                submission_payload = _load_json(
                    submission_marker,
                    "Genie Sim submission marker",
                )
            except Exception as exc:
                self.log(
                    f"Found Genie Sim submission marker but failed to read it: {exc}",
                    "WARNING",
                )
            else:
                marker_job_id = submission_payload.get("job_id")
                marker_status = submission_payload.get("job_status") or submission_payload.get("status")
                self.log(
                    "Found Genie Sim submission marker"
                    + (f" for job {marker_job_id}" if marker_job_id else "")
                    + (f" (status: {marker_status})." if marker_status else "."),
                    "INFO",
                )
        job_path = self.geniesim_dir / "job.json"
        resolved_interval = poll_interval
        if resolved_interval is None:
            resolved_interval = parse_int_env(
                os.getenv("GENIESIM_IMPORT_POLL_INTERVAL"),
                default=10,
            )
        resolved_interval = max(1, resolved_interval)
        if not job_path.is_file():
            wait_message = f"Waiting for Genie Sim job metadata at {job_path}..."
            if submission_marker.is_file():
                wait_message += (
                    f" Submission marker exists at {submission_marker},"
                    " but job.json is still required for polling."
                )
            self.log(wait_message, "INFO")
        while not job_path.is_file():
            time.sleep(resolved_interval)
        job_payload = _load_json(job_path, "Genie Sim job payload")
        job_id = job_payload.get("job_id")
        if not job_id:
            self.log(
                f"Genie Sim job metadata missing job_id at {job_path}",
                "ERROR",
            )
            return False
        run_id = str(job_payload.get("run_id") or self.run_id)
        if self._clear_stale_marker_if_mismatched(
            completion_marker,
            "Genie Sim import completion marker",
            expected_job_id=job_id,
            expected_run_id=run_id,
        ):
            self.log(
                f"Genie Sim import already completed (marker: {completion_marker}).",
                "INFO",
            )
            return True
        if self._clear_stale_marker_if_mismatched(
            marker_path,
            "Genie Sim import trigger marker",
            expected_job_id=job_id,
            expected_run_id=run_id,
        ):
            self.log(
                f"Genie Sim import already triggered (marker: {marker_path}).",
                "INFO",
            )
            return True
        try:
            _, job_status = self._poll_geniesim_job_status(
                job_path,
                job_id,
                poll_interval_override=resolved_interval,
            )
        except NonRetryableError as exc:
            self.log(str(exc), "ERROR")
            return False
        if job_status != "completed":
            self.log(
                f"Genie Sim job {job_id} status is {job_status}; import not triggered.",
                "ERROR",
            )
            return False
        if self._clear_stale_marker_if_mismatched(
            completion_marker,
            "Genie Sim import completion marker",
            expected_job_id=job_id,
            expected_run_id=run_id,
        ):
            self.log(
                f"Genie Sim import already completed (marker: {completion_marker}).",
                "INFO",
            )
            return True
        self._write_marker(
            marker_path,
            status="triggered",
            payload={"job_id": job_id, "run_id": run_id},
        )
        result = self._run_geniesim_import()
        return bool(result.success)

    def _run_dwm(self) -> StepResult:
        """
        Run DWM (Dexterous World Model) preparation.

        Generates conditioning data for DWM video diffusion model:
        - Egocentric camera trajectories
        - Static scene video renders
        - Hand mesh video renders
        - Text prompts

        Reference: DWM paper (arXiv:2512.17907)
        """
        try:
            # Import DWM preparation job
            sys.path.insert(0, str(REPO_ROOT / "dwm-preparation-job"))
            from prepare_dwm_bundle import run_dwm_preparation

            # Check prerequisites
            manifest_path = self.assets_dir / "scene_manifest.json"
            if not manifest_path.is_file():
                return StepResult(
                    step=PipelineStep.DWM,
                    success=False,
                    duration_seconds=0,
                    message="Manifest not found - run regen3d step first",
                )

            # Run DWM preparation
            output = run_dwm_preparation(
                scene_dir=self.scene_dir,
                output_dir=self.dwm_dir,
                num_trajectories=5,  # Default number of trajectories
                verbose=self.verbose,
            )

            if output.success:
                marker_path = self.dwm_dir / ".dwm_complete"
                self._write_marker(marker_path, status="completed")

                return StepResult(
                    step=PipelineStep.DWM,
                    success=True,
                    duration_seconds=output.generation_time_seconds,
                    message=f"Generated {output.num_bundles} DWM bundles",
                    outputs={
                        "bundles_count": output.num_bundles,
                        "total_frames": output.total_frames,
                        "output_dir": str(output.output_dir),
                        "manifest": str(output.manifest_path) if output.manifest_path else None,
                        "completion_marker": str(marker_path),
                    },
                )
            else:
                return StepResult(
                    step=PipelineStep.DWM,
                    success=False,
                    duration_seconds=output.generation_time_seconds,
                    message=f"DWM generation failed: {output.errors[:1]}",
                    outputs={"errors": output.errors},
                )

        except ImportError as e:
            return StepResult(
                step=PipelineStep.DWM,
                success=False,
                duration_seconds=0,
                message=f"Import error (DWM job not found): {self._summarize_exception(e)}",
            )
        except Exception as e:
            return self._handle_step_exception(
                PipelineStep.DWM,
                e,
                "DWM preparation failed",
            )

    def _run_dwm_inference(self) -> StepResult:
        """Run the DWM inference job on prepared bundles."""
        try:
            sys.path.insert(0, str(REPO_ROOT / "dwm-preparation-job"))
            from dwm_inference_job import run_dwm_inference

            manifest_path = self.dwm_dir / "dwm_bundles_manifest.json"
            if not manifest_path.is_file():
                return StepResult(
                    step=PipelineStep.DWM_INFERENCE,
                    success=False,
                    duration_seconds=0,
                    message="DWM bundles manifest not found - run dwm step first",
                )

            output = run_dwm_inference(
                bundles_dir=self.dwm_dir,
                api_endpoint=os.environ.get("DWM_API_ENDPOINT"),
                checkpoint_path=os.environ.get("DWM_CHECKPOINT_PATH"),
                verbose=self.verbose,
            )

            if output.success:
                marker_path = self.dwm_dir / ".dwm_inference_complete"
                self._write_marker(marker_path, status="completed")
                return StepResult(
                    step=PipelineStep.DWM_INFERENCE,
                    success=True,
                    duration_seconds=0,
                    message=f"Generated interaction videos for {len(output.bundles_processed)} bundles",
                    outputs={
                        "bundles": len(output.bundles_processed),
                        "manifest": str(output.manifest_path) if output.manifest_path else None,
                        "completion_marker": str(marker_path),
                    },
                )

            return StepResult(
                step=PipelineStep.DWM_INFERENCE,
                success=False,
                duration_seconds=0,
                message=f"DWM inference failed: {output.errors[:1]}",
                outputs={"errors": output.errors},
            )

        except ImportError as e:
            return StepResult(
                step=PipelineStep.DWM_INFERENCE,
                success=False,
                duration_seconds=0,
                message=f"Import error (DWM inference job not found): {self._summarize_exception(e)}",
            )
        except Exception as e:
            return self._handle_step_exception(
                PipelineStep.DWM_INFERENCE,
                e,
                "DWM inference failed",
            )

    def _run_dream2flow(self) -> StepResult:
        """
        Run Dream2Flow preparation.

        Generates Dream2Flow bundles from scene for video-to-flow robot control:
        - Task instructions from scene manifest
        - Initial RGB-D observations
        - Generated task videos (via video diffusion)
        - 3D object flow extraction
        - Robot tracking targets

        Reference: Dream2Flow paper (arXiv:2512.24766)
        """
        try:
            sys.path.insert(0, str(REPO_ROOT / "dream2flow-preparation-job"))
            from prepare_dream2flow_bundle import run_dream2flow_preparation

            # Check prerequisites
            manifest_path = self.assets_dir / "scene_manifest.json"
            if not manifest_path.is_file():
                return StepResult(
                    step=PipelineStep.DREAM2FLOW,
                    success=False,
                    duration_seconds=0,
                    message="Manifest not found - run regen3d step first",
                )

            # Run Dream2Flow preparation
            output = run_dream2flow_preparation(
                scene_dir=self.scene_dir,
                output_dir=self.dream2flow_dir,
                num_tasks=5,  # Default number of tasks
                verbose=self.verbose,
            )

            if output.success:
                marker_path = self.dream2flow_dir / ".dream2flow_complete"
                self._write_marker(marker_path, status="completed")

                return StepResult(
                    step=PipelineStep.DREAM2FLOW,
                    success=True,
                    duration_seconds=output.generation_time_seconds,
                    message=f"Generated {len(output.bundles)} Dream2Flow bundles",
                    outputs={
                        "bundles_count": len(output.bundles),
                        "output_dir": str(output.output_dir),
                        "manifest": str(output.manifest_path) if output.manifest_path else None,
                        "completion_marker": str(marker_path),
                    },
                )
            else:
                return StepResult(
                    step=PipelineStep.DREAM2FLOW,
                    success=False,
                    duration_seconds=output.generation_time_seconds,
                    message=f"Dream2Flow generation failed: {output.errors[:1] if output.errors else 'Unknown error'}",
                    outputs={"errors": output.errors},
                )

        except ImportError as e:
            return StepResult(
                step=PipelineStep.DREAM2FLOW,
                success=False,
                duration_seconds=0,
                message=f"Import error (Dream2Flow job not found): {self._summarize_exception(e)}",
            )
        except Exception as e:
            return self._handle_step_exception(
                PipelineStep.DREAM2FLOW,
                e,
                "Dream2Flow preparation failed",
            )

    def _run_dream2flow_inference(self) -> StepResult:
        """Run the Dream2Flow inference job on prepared bundles."""
        try:
            sys.path.insert(0, str(REPO_ROOT / "dream2flow-preparation-job"))
            from dream2flow_inference_job import run_dream2flow_inference

            manifest_path = self.dream2flow_dir / "dream2flow_bundles_manifest.json"
            if not manifest_path.is_file():
                return StepResult(
                    step=PipelineStep.DREAM2FLOW_INFERENCE,
                    success=False,
                    duration_seconds=0,
                    message="Dream2Flow bundles manifest not found - run dream2flow step first",
                )

            output = run_dream2flow_inference(
                bundles_dir=self.dream2flow_dir,
                api_endpoint=os.environ.get("DREAM2FLOW_API_ENDPOINT"),
                checkpoint_path=os.environ.get("DREAM2FLOW_CHECKPOINT_PATH"),
                verbose=self.verbose,
            )

            if output.success:
                marker_path = self.dream2flow_dir / ".dream2flow_inference_complete"
                self._write_marker(marker_path, status="completed")
                return StepResult(
                    step=PipelineStep.DREAM2FLOW_INFERENCE,
                    success=True,
                    duration_seconds=0,
                    message=f"Generated flow extractions for {len(output.bundles_processed)} bundles",
                    outputs={
                        "bundles": len(output.bundles_processed),
                        "manifest": str(output.manifest_path) if output.manifest_path else None,
                        "completion_marker": str(marker_path),
                    },
                )

            return StepResult(
                step=PipelineStep.DREAM2FLOW_INFERENCE,
                success=False,
                duration_seconds=0,
                message=f"Dream2Flow inference failed: {output.errors[:1] if output.errors else 'Unknown error'}",
                outputs={"errors": output.errors},
            )

        except ImportError as e:
            return StepResult(
                step=PipelineStep.DREAM2FLOW_INFERENCE,
                success=False,
                duration_seconds=0,
                message=f"Import error (Dream2Flow inference job not found): {self._summarize_exception(e)}",
            )
        except Exception as e:
            return self._handle_step_exception(
                PipelineStep.DREAM2FLOW_INFERENCE,
                e,
                "Dream2Flow inference failed",
            )

    def _run_validation(self) -> StepResult:
        """Run QA validation."""
        try:
            from tools.qa_validation.validator import run_qa_validation

            report_path = self.scene_dir / "validation_report.json"
            report = run_qa_validation(
                scene_dir=self.scene_dir,
                scene_id=self.scene_id,
                output_report=report_path,
                verbose=self.verbose,
            )

            if report.passed:
                return StepResult(
                    step=PipelineStep.VALIDATE,
                    success=True,
                    duration_seconds=0,
                    message=f"Validation passed: {report.summary.get('passed', 0)}/{report.summary.get('total_checks', 0)} checks",
                    outputs={"report": str(report_path), "summary": report.summary},
                )
            else:
                return StepResult(
                    step=PipelineStep.VALIDATE,
                    success=False,
                    duration_seconds=0,
                    message=f"Validation failed: {len(report.issues)} errors",
                    outputs={"report": str(report_path), "issues": report.issues[:5]},
                )

        except ImportError as e:
            return StepResult(
                step=PipelineStep.VALIDATE,
                success=False,
                duration_seconds=0,
                message=f"Import error: {self._summarize_exception(e)}",
            )

    def _expected_output_paths(self, step: PipelineStep) -> List[Path]:
        """Expected output paths for checkpoint validation."""
        if step == PipelineStep.REGEN3D_RECONSTRUCT:
            return [
                self.regen3d_dir / "scene_info.json",
                self.regen3d_dir / "objects",
            ]
        if step == PipelineStep.REGEN3D:
            return [
                self.assets_dir / "scene_manifest.json",
                self.layout_dir / "scene_layout_scaled.json",
                self.seg_dir / "inventory.json",
                self.assets_dir / ".regen3d_complete",
            ]
        if step == PipelineStep.SCALE:
            return [self.assets_dir / ".scale_complete"]
        if step == PipelineStep.INTERACTIVE:
            return [self.assets_dir / ".interactive_complete"]
        if step == PipelineStep.SIMREADY:
            return [
                self.assets_dir / "scene_manifest.json",
                self.assets_dir / ".simready_complete",
            ]
        if step == PipelineStep.USD:
            return [
                self.usd_dir / "scene.usda",
                self.assets_dir / ".usd_assembly_complete",
            ]
        if step == PipelineStep.INVENTORY_ENRICHMENT:
            return [self.seg_dir / "inventory_enriched.json"]
        if step == PipelineStep.REPLICATOR:
            return [
                self.replicator_dir / "bundle_metadata.json",
                self.replicator_dir / "placement_regions.usda",
                self.replicator_dir / ".replicator_complete",
            ]
        if step == PipelineStep.VARIATION_GEN:
            return [
                self.scene_dir / "variation_assets" / "variation_assets.json",
                self.scene_dir / "variation_assets" / ".variation_pipeline_complete",
            ]
        if step == PipelineStep.ISAAC_LAB:
            return [self.isaac_lab_dir / ".isaac_lab_complete"]
        if step == PipelineStep.GENIESIM_EXPORT:
            return [
                self.geniesim_dir / "scene_graph.json",
                self.geniesim_dir / "task_config.json",
                self.geniesim_dir / "asset_index.json",
                self.geniesim_dir / "merged_scene_manifest.json",
            ]
        if step == PipelineStep.GENIESIM_SUBMIT:
            return [
                self.geniesim_dir / "job.json",
                self.geniesim_dir / ".geniesim_submitted",
            ]
        if step == PipelineStep.GENIESIM_IMPORT:
            return [self.geniesim_dir / ".geniesim_import_complete"]
        if step == PipelineStep.DATASET_DELIVERY:
            return [self.geniesim_dir / ".dataset_delivery_complete"]
        if step == PipelineStep.DWM:
            return [self.dwm_dir / ".dwm_complete"]
        if step == PipelineStep.DWM_INFERENCE:
            return [self.dwm_dir / ".dwm_inference_complete"]
        if step == PipelineStep.DREAM2FLOW:
            return [self.dream2flow_dir / ".dream2flow_complete"]
        if step == PipelineStep.DREAM2FLOW_INFERENCE:
            return [self.dream2flow_dir / ".dream2flow_inference_complete"]
        if step == PipelineStep.VALIDATE:
            return [self.scene_dir / "validation_report.json"]
        return []

    def _print_summary(self) -> None:
        """Print pipeline execution summary."""
        self.log("\n" + "=" * 60)
        self.log("PIPELINE SUMMARY")
        self.log("=" * 60)

        total_time = sum(r.duration_seconds for r in self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed

        for result in self.results:
            status = "PASS" if result.success else "FAIL"
            self.log(f"  {result.step.value:15} [{status}] {result.duration_seconds:.2f}s - {result.message[:50]}")

        self.log("-" * 60)
        self.log(f"Total: {passed} passed, {failed} failed in {total_time:.2f}s")

        # Print key outputs
        self.log("\nKey Outputs:")
        for result in self.results:
            for key, value in result.outputs.items():
                if key in ["manifest", "layout", "scene_usda", "report"]:
                    self.log(f"  {key}: {value}")

        self.log("=" * 60)


def main():
    os.environ["REQUEST_ID"] = ensure_request_id()

    # Write PID file so the pipeline can be stopped safely via
    # kill "$(cat /tmp/pipeline_run.pid)" instead of broad pkill commands.
    _pid_path = os.environ.get("PIPELINE_PID_FILE", "/tmp/pipeline_run.pid")
    try:
        with open(_pid_path, "w") as _pf:
            _pf.write(str(os.getpid()))
    except OSError:
        pass

    # Graceful SIGTERM handler — lets current task finish before exiting.
    import signal

    def _sigterm_handler(signum, frame):
        logging.warning("SIGTERM received — completing current task then exiting")
        # Setting this env var is checked by the pipeline runner between steps.
        os.environ["_PIPELINE_SHUTDOWN"] = "1"

    signal.signal(signal.SIGTERM, _sigterm_handler)

    parser = argparse.ArgumentParser(
        description="Run BlueprintPipeline locally for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Path to scene directory (contains regen3d/, etc.)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        help="Comma-separated list of steps to run (default: all)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run QA validation at the end",
    )
    parser.add_argument(
        "--environment",
        default="kitchen",
        choices=["kitchen", "office", "warehouse", "laundry", "bedroom"],
        help="Environment type for policy selection",
    )
    parser.add_argument(
        "--with-interactive",
        action="store_true",
        help=(
            "Run interactive job "
            "(requires PARTICULATE_ENDPOINT for remote mode or "
            "PARTICULATE_MODE=local)"
        ),
    )
    parser.add_argument(
        "--enable-dwm",
        action="store_true",
        help="Include optional DWM preparation/inference steps (experimental) in the default pipeline",
    )
    parser.add_argument(
        "--enable-dream2flow",
        action="store_true",
        help="Include optional Dream2Flow preparation/inference steps (experimental) in the default pipeline",
    )
    parser.add_argument(
        "--enable-experimental",
        action="store_true",
        help="Enable experimental steps (DWM + Dream2Flow) in the default pipeline",
    )
    parser.add_argument(
        "--deliver",
        action="store_true",
        help="Include optional dataset delivery step in the pipeline",
    )
    parser.add_argument(
        "--disable-articulations",
        action="store_true",
        help="Explicitly disable articulated assets (avoids requiring PARTICULATE_ENDPOINT)",
    )
    parser.add_argument(
        "--use-geniesim",
        action="store_true",
        help="Use Genie Sim execution mode (overrides USE_GENIESIM for this run)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume pipeline from the specified step (skip completed steps with checkpoints)",
    )
    parser.add_argument(
        "--force-rerun",
        action="append",
        default=[],
        help=(
            "Comma-separated list of steps to rerun even if checkpoints exist. "
            "Use 'all' to rerun every step. Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--mock-geniesim",
        action="store_true",
        help="Run Genie Sim steps in mock mode (no external services required)",
    )
    parser.add_argument(
        "--auto-trigger-import",
        action="store_true",
        default=None,
        help=(
            "Run genie-sim-import automatically after a completed submit when import "
            "is not in the remaining steps. Defaults to True if submit is requested "
            "but import is not."
        ),
    )
    parser.add_argument(
        "--no-auto-trigger-import",
        action="store_true",
        dest="no_auto_trigger_import",
        help="Disable automatic import triggering after submit.",
    )
    parser.add_argument(
        "--import-poller",
        action="store_true",
        help=(
            "Run a local poller that waits for geniesim/job.json to reach status=completed "
            "and then triggers genie-sim-import once."
        ),
    )
    parser.add_argument(
        "--import-poller-interval",
        type=int,
        help=(
            "Polling interval in seconds for --import-poller. Defaults to "
            "GENIESIM_IMPORT_POLL_INTERVAL when unset."
        ),
    )
    parser.add_argument(
        "--estimate-costs",
        action="store_true",
        help="Print estimated GPU-hours and costs before running",
    )
    parser.add_argument(
        "--estimate-config",
        type=str,
        help="Path to JSON config for GPU rate + duration overrides",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=None,
        help="Stop the pipeline immediately on the first failure",
    )
    parser.add_argument(
        "--reset-breaker",
        action="store_true",
        help="Reset the step circuit breaker state before running",
    )
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        default=None,
        help="GCS bucket for input download / output upload (e.g., blueprint-8c1ca.appspot.com)",
    )
    parser.add_argument(
        "--gcs-download-inputs",
        action="store_true",
        help="Download input image from GCS before running (requires --gcs-bucket)",
    )
    parser.add_argument(
        "--gcs-upload-outputs",
        action="store_true",
        help="Upload step outputs to GCS after each step (requires --gcs-bucket)",
    )
    parser.add_argument(
        "--gcs-input-object",
        type=str,
        default=None,
        help=(
            "Preferred triggering input object path in GCS "
            '(e.g., "scenes/{scene_id}/images/my_image.jpeg")'
        ),
    )
    parser.add_argument(
        "--gcs-input-generation",
        type=str,
        default=None,
        help="GCS generation for the triggering input object (for idempotence metadata)",
    )
    parser.add_argument(
        "--gcs-upload-concurrency",
        type=int,
        default=None,
        help=(
            "Parallel upload workers for GCS sync (defaults to GCS_UPLOAD_CONCURRENCY "
            "or 4)"
        ),
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    if args.enable_experimental:
        args.enable_dwm = True
        args.enable_dream2flow = True

    if args.use_geniesim:
        os.environ["USE_GENIESIM"] = "true"
    if args.mock_geniesim:
        os.environ["GENIESIM_MOCK_MODE"] = "true"
        os.environ["USE_GENIESIM"] = "true"

    resume_from = None
    if args.resume_from:
        try:
            resume_from = PipelineStep(args.resume_from.strip().lower())
        except ValueError:
            print(f"Unknown resume-from step: {args.resume_from}")
            print(f"Available: {[s.value for s in PipelineStep]}")
            sys.exit(1)

    # Parse steps
    steps = None
    if args.steps:
        step_names = [s.strip().lower() for s in args.steps.split(",")]
        steps = []
        for name in step_names:
            try:
                steps.append(PipelineStep(name))
            except ValueError:
                print(f"Unknown step: {name}")
                print(f"Available: {[s.value for s in PipelineStep]}")
                sys.exit(1)

    if args.deliver and steps is not None:
        steps = LocalPipelineRunner._inject_dataset_delivery_step(steps)

    force_rerun_steps: List[PipelineStep] = []
    if args.force_rerun:
        requested = []
        for entry in args.force_rerun:
            requested.extend([item.strip().lower() for item in entry.split(",") if item.strip()])
        if "all" in requested:
            force_rerun_steps = list(PipelineStep)
        else:
            for name in requested:
                try:
                    force_rerun_steps.append(PipelineStep(name))
                except ValueError:
                    print(f"Unknown force-rerun step: {name}")
                    print(f"Available: {[s.value for s in PipelineStep]}")
                    sys.exit(1)

    # Create and run pipeline
    runner = LocalPipelineRunner(
        scene_dir=args.scene_dir,
        verbose=not args.quiet,
        skip_interactive=not args.with_interactive,
        environment_type=args.environment,
        enable_dwm=args.enable_dwm,
        enable_dream2flow=args.enable_dream2flow,
        enable_dataset_delivery=args.deliver,
        disable_articulated_assets=(
            args.disable_articulations
            or os.getenv("DISABLE_ARTICULATED_ASSETS", "").lower() in {"1", "true", "yes", "y"}
        ),
        fail_fast=args.fail_fast,
    )

    # Configure GCS sync if requested
    if args.gcs_bucket:
        runner.configure_gcs_sync(
            bucket_name=args.gcs_bucket,
            download_inputs=args.gcs_download_inputs,
            upload_outputs=args.gcs_upload_outputs,
            input_object=args.gcs_input_object,
            input_generation=args.gcs_input_generation,
            upload_concurrency=args.gcs_upload_concurrency,
        )
    elif (
        args.gcs_download_inputs
        or args.gcs_upload_outputs
        or args.gcs_input_object
        or args.gcs_input_generation
        or args.gcs_upload_concurrency is not None
    ):
        print(
            "ERROR: --gcs-download-inputs, --gcs-upload-outputs, --gcs-input-object, "
            "--gcs-input-generation, and --gcs-upload-concurrency require --gcs-bucket"
        )
        sys.exit(1)

    if args.import_poller:
        success = runner.run_geniesim_import_poller(
            poll_interval=args.import_poller_interval,
        )
        sys.exit(0 if success else 1)

    if args.estimate_costs:
        config_path = Path(args.estimate_config) if args.estimate_config else None
        config = load_estimate_config(
            config_path,
            include_dwm=args.enable_dwm,
            include_dream2flow=args.enable_dream2flow,
        )
        resolved_steps = steps or runner._resolve_default_steps()
        step_names = [
            step.value if isinstance(step, PipelineStep) else str(step)
            for step in resolved_steps
        ]
        summary = estimate_gpu_costs(step_names, config)
        print(format_estimate_summary(summary))

    # Resolve auto_trigger_import: explicit flags override smart default
    if args.no_auto_trigger_import:
        auto_trigger_import = False
    elif args.auto_trigger_import:
        auto_trigger_import = True
    else:
        auto_trigger_import = None  # Let run() determine smart default

    success = runner.run(
        steps=steps,
        run_validation=args.validate,
        resume_from=resume_from,
        force_rerun_steps=force_rerun_steps,
        reset_breaker=args.reset_breaker,
        auto_trigger_import=auto_trigger_import,
    )

    # Clean up PID file
    try:
        os.remove(_pid_path)
    except OSError:
        pass

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
