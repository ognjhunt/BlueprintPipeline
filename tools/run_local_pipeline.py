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

Pipeline Steps:
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
    13. dwm       - Generate DWM conditioning data (egocentric videos + hand meshes)
    14. dwm-inference - Run DWM model to generate interaction videos for each bundle
    15. validate  - QA validation

Note: DWM and Dream2Flow steps are optional and only included by default when
--enable-dwm or --enable-dream2flow is set.

References:
- 3D-RE-GEN (arXiv:2512.17459): "image → sim-ready 3D reconstruction"
  Paper: https://arxiv.org/abs/2512.17459
  Project: https://3dregen.jdihlmann.com/

- DWM (arXiv:2512.17907): Dexterous World Models for egocentric interaction
  Paper: https://arxiv.org/abs/2512.17907
  Project: https://snuvclab.github.io/dwm/
"""

import argparse
import importlib.util
import json
import re
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tools.checkpoint import load_checkpoint, should_skip_step, write_checkpoint
from tools.cost_tracking.estimate import (
    estimate_gpu_costs,
    format_estimate_summary,
    load_estimate_config,
)
from tools.config.env import parse_bool_env
from tools.config.production_mode import resolve_production_mode
from tools.config.seed_manager import configure_pipeline_seed
from tools.error_handling import (
    NonRetryableError,
    RetryConfig,
    RetryableError,
    retry_with_backoff,
)
from tools.inventory_enrichment import enrich_inventory_file, InventoryEnrichmentError
from tools.geniesim_adapter.mock_mode import resolve_geniesim_mock_mode
from tools.quality_gates import QualityGateCheckpoint, QualityGateRegistry

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class PipelineStep(str, Enum):
    """Pipeline steps in execution order."""
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
        PipelineStep.SIMREADY,
        PipelineStep.USD,
        PipelineStep.REPLICATOR,
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
        skip_interactive: bool = True,
        environment_type: str = "kitchen",
        enable_dwm: bool = False,
        enable_dream2flow: bool = False,
        enable_inventory_enrichment: Optional[bool] = None,
        disable_articulated_assets: bool = False,
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
        """
        self.scene_dir = Path(scene_dir).resolve()
        self.verbose = verbose
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
        self.disable_articulated_assets = disable_articulated_assets
        self.environment = os.getenv("BP_ENV", "development").lower()
        self.debug = os.getenv("BP_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
        self.enable_checkpoint_hashes = os.getenv("BP_CHECKPOINT_HASHES", "0").lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

        # Derive scene ID from directory name
        self.scene_id = self.scene_dir.name

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
        self._quality_gate_report_path = self._resolve_quality_gate_report_path()
        self.retry_config = self._resolve_retry_config()

    def log(self, msg: str, level: str = "INFO") -> None:
        """Log a message."""
        if self.verbose:
            print(f"[LOCAL-PIPELINE] [{level}] {msg}")

    def _sanitize_error_message(self, message: str) -> str:
        if not message:
            return message
        return self._path_redaction_regex.sub("<redacted-path>", message)

    def _summarize_exception(self, exc: Exception) -> str:
        sanitized_message = self._sanitize_error_message(str(exc))
        if sanitized_message:
            return f"{type(exc).__name__}: {sanitized_message}"
        return type(exc).__name__

    def _resolve_retry_config(self) -> RetryConfig:
        max_retries = self._parse_env_int("PIPELINE_RETRY_MAX", default=3)
        base_delay = self._parse_env_float("PIPELINE_RETRY_BASE_DELAY", default=1.0)
        max_delay = self._parse_env_float("PIPELINE_RETRY_MAX_DELAY", default=60.0)
        return RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )

    def _parse_env_int(self, name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            self.log(f"Invalid {name} value '{raw}', defaulting to {default}", "WARNING")
            return default

    def _parse_env_float(self, name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            self.log(f"Invalid {name} value '{raw}', defaulting to {default}", "WARNING")
            return default

    def _run_with_retry(self, step: PipelineStep, action: Any) -> Any:
        config = self.retry_config

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

        decorator = retry_with_backoff(
            max_retries=config.max_retries,
            base_delay=config.base_delay,
            max_delay=config.max_delay,
            backoff_factor=config.backoff_factor,
            jitter=config.jitter,
            on_retry=on_retry,
            on_failure=on_failure,
        )
        return decorator(action)()

    def _log_exception_traceback(self, context: str, exc: Exception) -> None:
        self.log(f"{context}: {self._summarize_exception(exc)}", "ERROR")
        if self.debug:
            self.log(traceback.format_exc(), "DEBUG")

    def _write_marker(self, marker_path: Path, status: str) -> None:
        """Write a simple JSON marker file."""
        marker_path.write_text(json.dumps({
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "scene_id": self.scene_id,
        }, indent=2))

    def _resolve_quality_gate_report_path(self) -> Path:
        report_dir = self.scene_dir / "quality_gates"
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir / "quality_gate_report.json"

    def _should_skip_quality_gates(self) -> bool:
        if not parse_bool_env(os.getenv("SKIP_QUALITY_GATES"), default=False):
            return False
        if resolve_production_mode():
            self.log(
                "SKIP_QUALITY_GATES requested but production mode detected; "
                "quality gates will still run.",
                "WARNING",
            )
            return False
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
                    "manifest": json.loads(manifest_path.read_text()),
                },
            }
        if step == PipelineStep.SIMREADY:
            manifest_path = self.assets_dir / "scene_manifest.json"
            if not manifest_path.is_file():
                return None
            manifest = json.loads(manifest_path.read_text())
            physics_objects = []
            for obj in manifest.get("objects", []):
                physics = obj.get("physics", {}) or {}
                physics_objects.append({
                    "id": obj.get("id"),
                    "mass": physics.get("mass_kg", 0.0),
                    "friction": physics.get("friction_static", physics.get("friction_dynamic", 0.0)),
                })
            return {
                "checkpoint": QualityGateCheckpoint.SIMREADY_COMPLETE,
                "context": {
                    "scene_id": self.scene_id,
                    "physics_properties": {"objects": physics_objects},
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
        if step == PipelineStep.ISAAC_LAB:
            return {
                "checkpoint": QualityGateCheckpoint.ISAAC_LAB_GENERATED,
                "context": {
                    "scene_id": self.scene_id,
                    "isaac_lab_dir": str(self.isaac_lab_dir),
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
        return None

    def _apply_quality_gates(self, step: PipelineStep, result: StepResult) -> StepResult:
        gate_payload = self._quality_gate_context_for_step(step)
        if not gate_payload:
            return result

        checkpoint = gate_payload["checkpoint"]
        context = gate_payload["context"]
        outputs = result.outputs
        outputs["quality_gate_checkpoint"] = checkpoint.value
        outputs["quality_gate_report"] = str(self._quality_gate_report_path)

        if self._should_skip_quality_gates():
            self.log(
                f"SKIP_QUALITY_GATES enabled - skipping quality gates for {checkpoint.value}",
                "WARNING",
            )
            self._quality_gates.save_report(self.scene_id, self._quality_gate_report_path)
            report = self._quality_gates.to_report(self.scene_id)
            outputs["quality_gate_summary"] = report.get("summary", {})
            outputs["quality_gate_skipped"] = True
            return result

        gate_results = self._quality_gates.run_checkpoint(
            checkpoint=checkpoint,
            context=context,
        )
        self._quality_gates.save_report(self.scene_id, self._quality_gate_report_path)
        report = self._quality_gates.to_report(self.scene_id)
        outputs["quality_gate_summary"] = report.get("summary", {})

        blocked = any((not entry.passed and entry.severity == "error") for entry in gate_results)
        if blocked:
            result.success = False
            result.message = f"Quality gates blocked at {checkpoint.value}"
            outputs["quality_gate_blocked"] = True
        return result

    def run(
        self,
        steps: Optional[List[PipelineStep]] = None,
        run_validation: bool = False,
        resume_from: Optional[PipelineStep] = None,
        force_rerun_steps: Optional[List[PipelineStep]] = None,
    ) -> bool:
        """Run the pipeline.

        Args:
            steps: Specific steps to run (default: all applicable)
            run_validation: Run QA validation at the end
            resume_from: Resume from the given step (skip completed steps with checkpoints)
            force_rerun_steps: Steps to rerun even if checkpoints exist

        Returns:
            True if all steps succeeded
        """
        seed = configure_pipeline_seed()
        if seed is not None:
            self.log(f"Using pipeline seed: {seed}")

        if steps is None:
            steps = self._resolve_default_steps()

        if run_validation and PipelineStep.VALIDATE not in steps:
            steps.append(PipelineStep.VALIDATE)

        if PipelineStep.GENIESIM_SUBMIT in steps and PipelineStep.GENIESIM_IMPORT not in steps:
            submit_index = steps.index(PipelineStep.GENIESIM_SUBMIT)
            steps.insert(submit_index + 1, PipelineStep.GENIESIM_IMPORT)

        self._apply_labs_flags(run_validation=run_validation)

        if resume_from is not None:
            if resume_from not in steps:
                self.log(
                    f"ERROR: resume-from step {resume_from.value} is not in requested steps",
                    "ERROR",
                )
                return False
            steps = steps[steps.index(resume_from):]

        self.log("=" * 60)
        self.log("BlueprintPipeline Local Runner")
        self.log("=" * 60)
        self.log(f"Scene directory: {self.scene_dir}")
        self.log(f"Scene ID: {self.scene_id}")
        self.log(f"Steps: {[s.value for s in steps]}")
        self.log("=" * 60)

        # Check prerequisites
        if not self.regen3d_dir.is_dir():
            self.log(f"ERROR: 3D-RE-GEN output not found at {self.regen3d_dir}", "ERROR")
            self.log("Run: python fixtures/generate_mock_regen3d.py first", "ERROR")
            return False

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
        forced_steps = set(force_rerun_steps or [])
        for step in steps:
            if resume_from is not None:
                expected_outputs = self._expected_output_paths(step)
                if step in forced_steps:
                    self.log(f"Force rerun requested for step {step.value}; skipping checkpoint.", "INFO")
                elif should_skip_step(
                    self.scene_dir,
                    step.value,
                    expected_outputs=expected_outputs,
                    require_nonempty=True,
                    require_fresh_outputs=True,
                    validate_sidecar_metadata=True,
                ):
                    checkpoint = load_checkpoint(self.scene_dir, step.value)
                    self.log(f"Skipping step {step.value} (checkpoint found)", "INFO")
                    self.results.append(
                        StepResult(
                            step=step,
                            success=True,
                            duration_seconds=0,
                            message="Skipped (checkpointed)",
                            outputs=checkpoint.outputs if checkpoint else {},
                        )
                    )
                    continue

            started_at = datetime.utcnow().isoformat() + "Z"
            result = self._run_step(step)
            completed_at = datetime.utcnow().isoformat() + "Z"
            if result.success:
                result = self._apply_quality_gates(step, result)
            self.results.append(result)

            if not result.success:
                all_success = False
                self.log(f"Step {step.value} failed: {result.message}", "ERROR")
                # Continue with remaining steps for partial results
            else:
                write_checkpoint(
                    self.scene_dir,
                    step.value,
                    status="completed",
                    started_at=started_at,
                    completed_at=completed_at,
                    outputs=result.outputs,
                    output_paths=self._expected_output_paths(step),
                    scene_id=self.scene_id,
                    store_output_hashes=self.enable_checkpoint_hashes,
                )
            if step == PipelineStep.REGEN3D and self._pending_articulation_preflight:
                if not self._preflight_articulation_requirements(steps):
                    return False

        # Print summary
        self._print_summary()

        return all_success

    def _steps_require_geniesim_preflight(self, steps: List[PipelineStep]) -> bool:
        """Return True if any Genie Sim step is requested."""
        return any(
            step in {
                PipelineStep.GENIESIM_EXPORT,
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
            self.log(
                "GENIESIM_MOCK_MODE requested but production mode detected; "
                "mock mode is ignored and a Genie Sim server is required.",
                "WARNING",
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
        if not require_server and not report.get("status", {}).get("server_running", False):
            self.log(
                "Genie Sim server is not running; the local framework will start it automatically if needed.",
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

    def _apply_labs_flags(self, run_validation: bool) -> None:
        """Apply production/labs flags for staging or lab validation runs."""
        staging_e2e = os.environ.get("RUN_STAGING_E2E") == "1"
        labs_staging = os.environ.get("LABS_STAGING", "").lower() in {"1", "true", "yes"}
        labs_validation = os.environ.get("LABS_VALIDATION", "").lower() in {"1", "true", "yes"}

        if not (staging_e2e or labs_staging or (labs_validation and run_validation)):
            return

        os.environ.setdefault("LABS_STAGING", "1")
        os.environ.setdefault("DATA_QUALITY_LEVEL", "production")
        os.environ.setdefault("ISAAC_SIM_REQUIRED", "true")
        os.environ.setdefault("PRODUCTION_MODE", "true")
        self.log("Applied labs/production flags for staging or lab validation runs")

    def _resolve_default_steps(self) -> List[PipelineStep]:
        """Resolve default steps, using pipeline selector for Genie Sim mode."""
        try:
            from tools.pipeline_selector.selector import PipelineSelector, DataGenerationBackend
        except ImportError:
            self.log("Pipeline selector not available; falling back to default steps", "WARNING")
            steps = self.DEFAULT_STEPS.copy()
            return self._apply_optional_steps(steps)

        selector = PipelineSelector(scene_root=self.scene_dir)
        decision = selector.select(self.scene_dir)
        if decision.data_backend == DataGenerationBackend.GENIESIM:
            steps = self._map_jobs_to_steps(decision.job_sequence)
            if PipelineStep.GENIESIM_SUBMIT in steps and PipelineStep.GENIESIM_IMPORT not in steps:
                submit_index = steps.index(PipelineStep.GENIESIM_SUBMIT)
                steps.insert(submit_index + 1, PipelineStep.GENIESIM_IMPORT)
        else:
            steps = self.DEFAULT_STEPS.copy()

        steps = self._apply_optional_steps(steps)

        return steps

    def _apply_optional_steps(self, steps: List[PipelineStep]) -> List[PipelineStep]:
        """Append optional steps gated by explicit flags."""
        if self.enable_inventory_enrichment:
            steps = self._inject_inventory_enrichment_step(steps)
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

    def _map_jobs_to_steps(self, job_sequence: List[str]) -> List[PipelineStep]:
        """Map pipeline selector job names to local runner steps."""
        mapping = {
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

        manifest = json.loads(manifest_path.read_text())
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

        if not endpoint:
            self.log(
                "ERROR: Articulated assets detected but PARTICULATE_ENDPOINT is not set.",
                "ERROR",
            )
            self.log(
                "Set PARTICULATE_ENDPOINT and re-run, or set DISABLE_ARTICULATED_ASSETS=true to proceed "
                "without articulated assets.",
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
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def _run_step(self, step: PipelineStep) -> StepResult:
        """Run a single pipeline step."""
        import time
        start_time = time.time()

        self.log(f"\n--- Running step: {step.value} ---")

        try:
            if step == PipelineStep.REGEN3D:
                result = self._run_regen3d_adapter()
            elif step == PipelineStep.SCALE:
                result = self._run_scale()
            elif step == PipelineStep.INTERACTIVE:
                result = self._run_interactive()
            elif step == PipelineStep.SIMREADY:
                result = self._run_simready()
            elif step == PipelineStep.USD:
                result = self._run_usd_assembly()
            elif step == PipelineStep.INVENTORY_ENRICHMENT:
                result = self._run_inventory_enrichment()
            elif step == PipelineStep.REPLICATOR:
                result = self._run_replicator()
            elif step == PipelineStep.VARIATION_GEN:
                result = self._run_variation_gen()
            elif step == PipelineStep.ISAAC_LAB:
                result = self._run_isaac_lab()
            elif step == PipelineStep.GENIESIM_EXPORT:
                result = self._run_geniesim_export()
            elif step == PipelineStep.GENIESIM_SUBMIT:
                result = self._run_geniesim_submit()
            elif step == PipelineStep.GENIESIM_IMPORT:
                result = self._run_geniesim_import()
            elif step == PipelineStep.DWM:
                result = self._run_dwm()
            elif step == PipelineStep.DWM_INFERENCE:
                result = self._run_dwm_inference()
            elif step == PipelineStep.DREAM2FLOW:
                result = self._run_dream2flow()
            elif step == PipelineStep.DREAM2FLOW_INFERENCE:
                result = self._run_dream2flow_inference()
            elif step == PipelineStep.VALIDATE:
                result = self._run_validation()
            else:
                result = StepResult(
                    step=step,
                    success=False,
                    duration_seconds=0,
                    message=f"Unknown step: {step.value}",
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

        return result

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
        manifest_path.write_text(json.dumps(manifest, indent=2))
        manifest = self._annotate_articulation_requirements(manifest, manifest_path)
        self.log(f"Wrote manifest: {manifest_path}")

        # Generate layout
        layout = adapter.create_layout(regen3d_output, apply_scale_factor=1.0)
        layout_path = self.layout_dir / "scene_layout_scaled.json"
        layout_path.write_text(json.dumps(layout, indent=2))
        self.log(f"Wrote layout: {layout_path}")

        # Copy assets
        asset_paths = adapter.copy_assets(
            regen3d_output,
            self.scene_dir,
            assets_prefix="assets",
        )
        self.log(f"Copied {len(asset_paths)} assets")

        # Generate inventory
        inventory = self._generate_inventory(regen3d_output)
        inventory_path = self.seg_dir / "inventory.json"
        inventory_path.write_text(json.dumps(inventory, indent=2))
        self.log(f"Wrote inventory: {inventory_path}")

        # Write completion marker
        marker_path = self.assets_dir / ".regen3d_complete"
        marker_content = {
            "status": "complete",
            "scene_id": self.scene_id,
            "objects_count": len(regen3d_output.objects),
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }
        marker_path.write_text(json.dumps(marker_content, indent=2))

        return StepResult(
            step=PipelineStep.REGEN3D,
            success=True,
            duration_seconds=0,
            message="3D-RE-GEN adapter completed",
            outputs={
                "manifest": str(manifest_path),
                "layout": str(layout_path),
                "objects_count": len(regen3d_output.objects),
            },
        )

    def _annotate_articulation_requirements(
        self,
        manifest: Dict[str, Any],
        manifest_path: Path,
    ) -> Dict[str, Any]:
        """Detect articulated objects and annotate the manifest."""
        from tools.articulation import detect_scene_articulations

        results = detect_scene_articulations(
            manifest,
            use_llm=False,
            verbose=self.verbose,
        )

        required_ids = []
        for obj in manifest.get("objects", []):
            obj_id = obj.get("id")
            if not obj_id or obj.get("sim_role") in {"background", "scene_shell"}:
                continue

            result = results.get(obj_id)
            if not result or not result.has_articulation:
                continue

            required_ids.append(obj_id)
            articulation = obj.get("articulation") or {}
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
                "detection": detection_payload,
            })
            obj["articulation"] = articulation

            if obj.get("sim_role") in {"unknown", "static", None, ""}:
                obj["sim_role"] = self._infer_articulation_role(obj)

        metadata = manifest.get("metadata") or {}
        metadata["articulation_detection"] = {
            "required_count": len(required_ids),
            "required_objects": required_ids,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source": "heuristic",
        }
        manifest["metadata"] = metadata
        manifest_path.write_text(json.dumps(manifest, indent=2))

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

    def _generate_inventory(self, regen3d_output) -> Dict[str, Any]:
        """Generate semantic inventory from 3D-RE-GEN output."""
        objects = []
        for obj in regen3d_output.objects:
            inv_obj = {
                "id": obj.id,
                "category": obj.category or "object",
                "short_description": obj.description or f"Object {obj.id}",
                "sim_role": obj.sim_role if obj.sim_role != "unknown" else "static",
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

            layout = json.loads(layout_path.read_text())
            objects = layout.get("objects", [])
            room_box = layout.get("room_box")

            manifest_path = self.assets_dir / "scene_manifest.json"
            manifest = None
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text())
                except json.JSONDecodeError:
                    warnings.append(f"Failed to parse manifest at {manifest_path}")

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

            report_path.write_text(json.dumps(report, indent=2))
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
            manifest = json.loads(manifest_path.read_text())
            required_ids = self._required_articulation_ids(manifest)

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
                        "Run with --with-interactive and set PARTICULATE_ENDPOINT."
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

        # Would need PARTICULATE_ENDPOINT to run
        endpoint = os.getenv("PARTICULATE_ENDPOINT")
        if not endpoint:
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
                    message="Articulation required but PARTICULATE_ENDPOINT not set",
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
            "ASSETS_PREFIX": str(self.assets_dir),
            "REGEN3D_PREFIX": str(self.regen3d_dir),
            "SCENE_ID": self.scene_id,
            "PARTICULATE_ENDPOINT": endpoint,
            "PRODUCTION_MODE": "false",
            "DISALLOW_PLACEHOLDER_URDF": os.getenv("DISALLOW_PLACEHOLDER_URDF", "false"),
        })

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
            failure_payload = json.loads(failure_marker.read_text())
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

        results_data = json.loads(results_path.read_text())
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
        # Load manifest
        manifest_path = self.assets_dir / "scene_manifest.json"
        if not manifest_path.is_file():
            return StepResult(
                step=PipelineStep.SIMREADY,
                success=False,
                duration_seconds=0,
                message="Manifest not found - run regen3d step first",
            )

        manifest = json.loads(manifest_path.read_text())

        # For local testing, add basic physics properties to each object
        for obj in manifest.get("objects", []):
            if not obj.get("physics"):
                mass = self._estimate_mass(obj)
                friction_static = 0.5
                friction_dynamic = 0.4
                restitution = 0.1

                obj["physics"] = {
                    "dynamic": obj.get("sim_role") in ["manipulable_object", "clutter"],
                    "mass_kg": mass,
                    "friction_static": friction_static,
                    "friction_dynamic": friction_dynamic,
                    "restitution": restitution,
                    "center_of_mass_offset": [0.0, 0.0, 0.0],
                    # Sim2Real distribution ranges for domain randomization
                    "mass_kg_range": [mass * 0.8, mass * 1.2],
                    "friction_static_range": [friction_static * 0.85, min(1.5, friction_static * 1.15)],
                    "friction_dynamic_range": [friction_dynamic * 0.85, min(1.2, friction_dynamic * 1.15)],
                    "restitution_range": [max(0.0, restitution * 0.7), min(1.0, restitution * 1.3)],
                    "center_of_mass_noise": [0.005, 0.005, 0.005],
                }

        # Write updated manifest
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Write simready completion marker
        marker_path = self.assets_dir / ".simready_complete"
        marker_path.write_text(f"completed at {datetime.utcnow().isoformat()}Z\n")

        self.log(f"Added physics to {len(manifest.get('objects', []))} objects")

        return StepResult(
            step=PipelineStep.SIMREADY,
            success=True,
            duration_seconds=0,
            message="SimReady preparation completed",
            outputs={"objects_processed": len(manifest.get("objects", []))},
        )

    def _estimate_mass(self, obj: Dict[str, Any]) -> float:
        """Estimate mass from dimensions and material hints."""
        dims = obj.get("dimensions_est", {})
        volume = (
            dims.get("width", 0.1) *
            dims.get("height", 0.1) *
            dims.get("depth", 0.1)
        )

        # Default density (like plastic)
        density = 1000  # kg/m^3

        material = obj.get("physics_hints", {}).get("material_type", "")
        if material in ["metal", "steel"]:
            density = 7800
        elif material in ["wood"]:
            density = 600
        elif material in ["ceramic", "stone"]:
            density = 2500
        elif material in ["fabric", "plastic"]:
            density = 200

        return max(0.01, volume * density)  # Minimum 10g

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

        manifest = json.loads(manifest_path.read_text())
        layout = json.loads(layout_path.read_text())

        # Generate a minimal scene.usda
        usda_content = self._generate_usda(manifest, layout)
        usda_path = self.usd_dir / "scene.usda"
        usda_path.write_text(usda_content)
        self._write_marker(self.assets_dir / ".usd_assembly_complete", status="completed")

        self.log(f"Generated USD: {usda_path}")

        return StepResult(
            step=PipelineStep.USD,
            success=True,
            duration_seconds=0,
            message="USD assembly completed",
            outputs={"scene_usda": str(usda_path)},
        )

    def _generate_usda(self, manifest: Dict, layout: Dict) -> str:
        """Generate a minimal scene.usda file."""
        meters_per_unit = manifest.get("scene", {}).get("meters_per_unit", 1.0)
        coord_frame = manifest.get("scene", {}).get("coordinate_frame", "Y")

        lines = [
            '#usda 1.0',
            '(',
            f'    metersPerUnit = {meters_per_unit}',
            f'    upAxis = "{coord_frame.upper()}"',
            '    defaultPrim = "World"',
            ')',
            '',
            'def Xform "World" {',
            '    def Xform "Scene" {',
        ]

        # Add objects as references
        for obj in manifest.get("objects", []):
            obj_id = obj.get("id", "unknown")
            asset_path = obj.get("asset", {}).get("path", f"assets/obj_{obj_id}/asset.glb")

            # Get transform
            transform = obj.get("transform", {})
            pos = transform.get("position", {})
            px = pos.get("x", 0)
            py = pos.get("y", 0)
            pz = pos.get("z", 0)

            lines.append(f'        def Xform "obj_{obj_id}" (')
            lines.append(f'            # Reference to: {asset_path}')
            lines.append(f'        ) {{')
            lines.append(f'            double3 xformOp:translate = ({px}, {py}, {pz})')
            lines.append(f'            uniform token[] xformOpOrder = ["xformOp:translate"]')
            lines.append(f'        }}')
            lines.append('')

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
            enrich_inventory_file(inventory_path, output_path=output_path)
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
        """Run replicator bundle generation."""
        # Load manifest and inventory
        manifest_path = self.assets_dir / "scene_manifest.json"
        inventory_path = self.seg_dir / "inventory_enriched.json"
        if not inventory_path.is_file():
            inventory_path = self.seg_dir / "inventory.json"

        if not manifest_path.is_file():
            return StepResult(
                step=PipelineStep.REPLICATOR,
                success=False,
                duration_seconds=0,
                message="Manifest not found",
            )

        manifest = json.loads(manifest_path.read_text())
        inventory = {}
        if inventory_path.is_file():
            inventory = json.loads(inventory_path.read_text())

        # Create replicator directories
        policies_dir = self.replicator_dir / "policies"
        policies_dir.mkdir(parents=True, exist_ok=True)

        # Generate bundle metadata
        bundle_metadata = {
            "scene_id": self.scene_id,
            "environment_type": self.environment_type,
            "policies": ["manipulation", "navigation"],
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        (self.replicator_dir / "bundle_metadata.json").write_text(
            json.dumps(bundle_metadata, indent=2)
        )

        # Generate minimal placement regions
        placement_regions = self._generate_placement_regions(manifest)
        (self.replicator_dir / "placement_regions.usda").write_text(placement_regions)

        # Generate variation asset manifest
        variation_dir = self.replicator_dir / "variation_assets"
        variation_dir.mkdir(parents=True, exist_ok=True)
        variation_manifest = self._generate_variation_manifest(manifest, inventory)
        (variation_dir / "manifest.json").write_text(
            json.dumps(variation_manifest, indent=2)
        )
        self._write_marker(self.replicator_dir / ".replicator_complete", status="completed")

        self.log("Generated replicator bundle")

        return StepResult(
            step=PipelineStep.REPLICATOR,
            success=True,
            duration_seconds=0,
            message="Replicator bundle generated",
            outputs={
                "placement_regions": str(self.replicator_dir / "placement_regions.usda"),
                "bundle_metadata": str(self.replicator_dir / "bundle_metadata.json"),
            },
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

        manifest = json.loads(manifest_path.read_text())
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

        gcs_scene_dir = self._ensure_gcs_scene_link()
        if gcs_scene_dir is None:
            return StepResult(
                step=PipelineStep.VARIATION_GEN,
                success=False,
                duration_seconds=0,
                message="Unable to prepare /mnt/gcs mapping for variation assets",
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
        (output_dir / "variation_assets.json").write_text(json.dumps(payload, indent=2))

    def _ensure_gcs_scene_link(self) -> Optional[Path]:
        gcs_root = Path("/mnt/gcs")
        gcs_scene_dir = gcs_root / "scenes" / self.scene_id
        try:
            gcs_scene_dir.parent.mkdir(parents=True, exist_ok=True)
            if not gcs_scene_dir.exists():
                gcs_scene_dir.symlink_to(self.scene_dir)
        except OSError as exc:
            self._log_exception_traceback("Failed to map /mnt/gcs for variation assets", exc)
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

        simready_manifest = json.loads(simready_manifest_path.read_text())
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
        output_path.write_text(json.dumps(payload, indent=2))

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
                policy_config = json.loads(policy_config_path.read_text())
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

            manifest = json.loads(manifest_path.read_text())

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
            self._log_exception_traceback("Isaac Lab task generation failed", e)
            return StepResult(
                step=PipelineStep.ISAAC_LAB,
                success=False,
                duration_seconds=0,
                message=f"Error: {self._summarize_exception(e)}",
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
            return StepResult(
                step=PipelineStep.GENIESIM_EXPORT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Genie Sim export failed: {self._summarize_exception(exc)}",
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
        except NonRetryableError as exc:
            return StepResult(
                step=PipelineStep.GENIESIM_SUBMIT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=str(exc),
            )

        scene_graph = json.loads(scene_graph_path.read_text())
        asset_index = json.loads(asset_index_path.read_text())
        task_config = json.loads(task_config_path.read_text())

        robot_type = os.getenv("GENIESIM_ROBOT_TYPE", "franka")
        episodes_per_task = int(os.getenv("EPISODES_PER_TASK", "10"))
        num_variations = int(os.getenv("NUM_VARIATIONS", "5"))
        min_quality_score = float(os.getenv("MIN_QUALITY_SCORE", "0.85"))

        job_id = None
        submission_message = None
        local_run_result = None
        preflight_report = None
        import uuid

        job_id = f"local-{uuid.uuid4()}"
        submission_message = "Local Genie Sim execution started."
        preflight_report = run_geniesim_preflight(
            "genie-sim-local-runner",
            require_server=False,
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
                outputs={"preflight": preflight_report},
            )

        output_dir = self.episodes_dir / f"geniesim_{job_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        config_dir = output_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        merged_manifest_path = self.geniesim_dir / "merged_scene_manifest.json"
        if merged_manifest_path.is_file():
            scene_manifest = json.loads(merged_manifest_path.read_text())
        else:
            scene_manifest = {"scene_graph": scene_graph}

        scene_manifest_path = config_dir / "scene_manifest.json"
        task_config_local_path = config_dir / "task_config.json"
        scene_manifest_path.write_text(json.dumps(scene_manifest, indent=2))
        task_config_local_path.write_text(json.dumps(task_config, indent=2))

        def _collect_data() -> Any:
            result = run_local_data_collection(
                scene_manifest_path=scene_manifest_path,
                task_config_path=task_config_local_path,
                output_dir=output_dir,
                robot_type=robot_type,
                episodes_per_task=episodes_per_task,
                verbose=True,
            )
            if not result or not result.success:
                raise RetryableError("Local Genie Sim execution failed")
            return result

        try:
            local_run_result = self._run_with_retry(PipelineStep.GENIESIM_SUBMIT, _collect_data)
        except NonRetryableError as exc:
            return StepResult(
                step=PipelineStep.GENIESIM_SUBMIT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=str(exc),
                outputs={"preflight": preflight_report},
            )
        except Exception as exc:
            return StepResult(
                step=PipelineStep.GENIESIM_SUBMIT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Genie Sim submit failed: {self._summarize_exception(exc)}",
                outputs={"preflight": preflight_report},
            )
        submission_message = (
            "Local Genie Sim execution completed."
            if local_run_result and local_run_result.success
            else "Local Genie Sim execution failed."
        )

        job_status = (
            "completed"
            if local_run_result and local_run_result.success
            else "failed"
        )
        if preflight_report and not preflight_report.get("ok", False):
            job_status = "failed"

        job_payload = {
            "job_id": job_id,
            "scene_id": self.scene_id,
            "status": job_status,
            "submitted_at": datetime.utcnow().isoformat() + "Z",
            "message": submission_message,
            "bundle": {
                "scene_graph": str(scene_graph_path),
                "asset_index": str(asset_index_path),
                "task_config": str(task_config_path),
            },
            "generation_params": {
                "robot_type": robot_type,
                "episodes_per_task": episodes_per_task,
                "num_variations": num_variations,
                "min_quality_score": min_quality_score,
            },
        }

        episodes_path = str(self.episodes_dir / f"geniesim_{job_id}")
        job_payload["artifacts"] = {
            "episodes_path": episodes_path,
            "episodes_prefix": episodes_path,
            "lerobot_path": str(Path(episodes_path) / "lerobot"),
            "lerobot_prefix": str(Path(episodes_path) / "lerobot"),
        }
        job_payload["local_execution"] = {
            "success": bool(local_run_result and local_run_result.success),
            "episodes_collected": getattr(local_run_result, "episodes_collected", 0) if local_run_result else 0,
            "episodes_passed": getattr(local_run_result, "episodes_passed", 0) if local_run_result else 0,
            "preflight": preflight_report,
            "generation_duration_seconds": time.time() - start_time,
        }

        job_path = self.geniesim_dir / "job.json"
        job_path.write_text(json.dumps(job_payload, indent=2))

        return StepResult(
            step=PipelineStep.GENIESIM_SUBMIT,
            success=job_status != "failed",
            duration_seconds=time.time() - start_time,
            message=submission_message or "Genie Sim submission completed",
            outputs={
                "job_id": job_id,
                "job_status": job_status,
                "job_payload": str(job_path),
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

        job_payload = json.loads(job_path.read_text())
        try:
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
        job_status = job_payload.get("status", "submitted")
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

        try:
            if job_status != "completed":
                raise NonRetryableError(
                    f"Genie Sim job status is {job_status}; import requires completed job"
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
        except NonRetryableError as exc:
            return StepResult(
                step=PipelineStep.GENIESIM_IMPORT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=str(exc),
                outputs={
                    "job_id": job_id,
                    "job_status": job_status,
                    "output_dir": str(output_dir),
                    "recordings_path": str(recordings_dir),
                    "lerobot_path": str(lerobot_dir),
                    "lerobot_dataset_info": str(dataset_info_path),
                },
            )
        config = ImportConfig(
            job_id=job_id,
            output_dir=output_dir,
            min_quality_score=float(os.getenv("MIN_QUALITY_SCORE", "0.85")),
            enable_validation=parse_bool_env(os.getenv("ENABLE_VALIDATION"), default=True),
            filter_low_quality=parse_bool_env(os.getenv("FILTER_LOW_QUALITY"), default=True),
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
            return StepResult(
                step=PipelineStep.GENIESIM_IMPORT,
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Genie Sim import failed: {self._summarize_exception(exc)}",
            )

        marker_path = None
        if result.success:
            marker_path = self.geniesim_dir / ".geniesim_import_complete"
            self._write_marker(marker_path, status="completed")
        duration_seconds = time.time() - start_time
        local_execution = job_payload.get("local_execution", {})
        local_execution["import_duration_seconds"] = duration_seconds
        job_payload["local_execution"] = local_execution
        job_path.write_text(json.dumps(job_payload, indent=2))

        return StepResult(
            step=PipelineStep.GENIESIM_IMPORT,
            success=result.success,
            duration_seconds=duration_seconds,
            message="Genie Sim import completed" if result.success else "Genie Sim import failed",
            outputs={
                "job_id": job_id,
                "import_manifest": str(result.import_manifest_path) if result.import_manifest_path else None,
                "output_dir": str(output_dir),
                "recordings_path": str(recordings_dir),
                "lerobot_path": str(lerobot_dir),
                "lerobot_dataset_info": str(dataset_info_path),
                "completion_marker": str(marker_path) if result.success else None,
            },
        )

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
            self._log_exception_traceback("DWM preparation failed", e)
            return StepResult(
                step=PipelineStep.DWM,
                success=False,
                duration_seconds=0,
                message=f"Error: {self._summarize_exception(e)}",
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
            self._log_exception_traceback("DWM inference failed", e)
            return StepResult(
                step=PipelineStep.DWM_INFERENCE,
                success=False,
                duration_seconds=0,
                message=f"Error: {self._summarize_exception(e)}",
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
            self._log_exception_traceback("Dream2Flow preparation failed", e)
            return StepResult(
                step=PipelineStep.DREAM2FLOW,
                success=False,
                duration_seconds=0,
                message=f"Error: {self._summarize_exception(e)}",
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
            self._log_exception_traceback("Dream2Flow inference failed", e)
            return StepResult(
                step=PipelineStep.DREAM2FLOW_INFERENCE,
                success=False,
                duration_seconds=0,
                message=f"Error: {self._summarize_exception(e)}",
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
            ]
        if step == PipelineStep.GENIESIM_SUBMIT:
            return [self.geniesim_dir / "job.json"]
        if step == PipelineStep.GENIESIM_IMPORT:
            return [self.geniesim_dir / ".geniesim_import_complete"]
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
        choices=["kitchen", "office", "warehouse", "laundry"],
        help="Environment type for policy selection",
    )
    parser.add_argument(
        "--with-interactive",
        action="store_true",
        help="Run interactive job (requires PARTICULATE_ENDPOINT)",
    )
    parser.add_argument(
        "--enable-dwm",
        action="store_true",
        help="Include optional DWM preparation/inference steps in the default pipeline",
    )
    parser.add_argument(
        "--enable-dream2flow",
        action="store_true",
        help="Include optional Dream2Flow preparation/inference steps in the default pipeline",
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
        "-q", "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

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
        disable_articulated_assets=(
            args.disable_articulations
            or os.getenv("DISABLE_ARTICULATED_ASSETS", "").lower() in {"1", "true", "yes", "y"}
        ),
    )

    if args.estimate_costs:
        config_path = Path(args.estimate_config) if args.estimate_config else None
        config = load_estimate_config(config_path)
        resolved_steps = steps or runner._resolve_default_steps()
        step_names = [
            step.value if isinstance(step, PipelineStep) else str(step)
            for step in resolved_steps
        ]
        summary = estimate_gpu_costs(step_names, config)
        print(format_estimate_summary(summary))

    success = runner.run(
        steps=steps,
        run_validation=args.validate,
        resume_from=resume_from,
        force_rerun_steps=force_rerun_steps,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
