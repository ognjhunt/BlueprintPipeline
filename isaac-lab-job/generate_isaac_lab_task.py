#!/usr/bin/env python3
"""
Isaac Lab Task Generation Job.

Generates complete Isaac Lab task packages from BlueprintPipeline scenes.
This is the final step in the pipeline, producing RL training-ready packages.

Pipeline Position:
    3D-RE-GEN → simready → usd-assembly → replicator → [THIS JOB]

Outputs:
    isaac_lab/
        env_cfg.py         - ManagerBasedEnv configuration
        task_{policy}.py   - Task implementation
        train_cfg.yaml     - Training hyperparameters
        randomizations.py  - EventManager-compatible hooks
        reward_functions.py - Reward modules
        __init__.py        - Package init

Environment Variables:
    BUCKET: GCS bucket name
    SCENE_ID: Scene identifier
    ASSETS_PREFIX: Path to assets (contains scene_manifest.json)
    USD_PREFIX: Path to USD files (contains scene.usda)
    REPLICATOR_PREFIX: Path to Replicator bundle
    ISAAC_LAB_PREFIX: Output path for Isaac Lab package
    ENVIRONMENT_TYPE: Environment type hint (kitchen, office, etc.)
    POLICY_ID: Specific policy to generate (optional, auto-selects based on env)
    ROBOT_TYPE: Robot type (franka, ur10, fetch) - default: franka
    NUM_ENVS: Number of parallel environments (default: 1024)
    STRICT_CONFIG_LOADING: Fail fast if policy config or replicator metadata is missing/invalid
"""

import ast
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.isaac_lab_tasks.task_generator import IsaacLabTaskGenerator
from tools.isaac_lab_tasks.runtime_validator import (
    IsaacLabRuntimeValidator,
    RuntimeValidationResult,
)
from tools.validation.entrypoint_checks import validate_required_env_vars

logger = logging.getLogger(__name__)


# =============================================================================
# Code Validation
# =============================================================================


@dataclass
class CodeValidationResult:
    """Result of Python code validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_python_syntax(code: str, filename: str = "<string>") -> CodeValidationResult:
    """
    Validate Python code for syntax errors.

    Args:
        code: Python source code
        filename: Filename for error messages

    Returns:
        CodeValidationResult with syntax errors if any
    """
    result = CodeValidationResult(is_valid=True)

    try:
        ast.parse(code)
    except SyntaxError as e:
        result.is_valid = False
        result.errors.append(f"{filename}:{e.lineno}: SyntaxError: {e.msg}")

    return result


def validate_isaac_lab_env_config(code: str) -> CodeValidationResult:
    """
    Validate Isaac Lab env_cfg.py structure.

    Checks for:
    - Required class definitions
    - Required imports
    - Proper configuration structure

    Args:
        code: Python source code of env_cfg.py

    Returns:
        CodeValidationResult with any issues found
    """
    result = validate_python_syntax(code, "env_cfg.py")
    if not result.is_valid:
        return result

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return result

    # Track what we find
    found_classes = set()
    found_imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            found_classes.add(node.name)

            # Check for required base classes
            if node.name.endswith("EnvCfg"):
                has_manager_based = any(
                    "ManagerBasedEnvCfg" in ast.unparse(base) if hasattr(ast, 'unparse') else True
                    for base in node.bases
                )
                if not has_manager_based:
                    result.warnings.append(
                        f"Class {node.name} should inherit from ManagerBasedEnvCfg"
                    )

        elif isinstance(node, ast.Import):
            for alias in node.names:
                found_imports.add(alias.name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found_imports.add(node.module)

    # Check for required elements
    env_cfg_found = any(name.endswith("EnvCfg") for name in found_classes)
    if not env_cfg_found:
        result.errors.append("Missing EnvCfg class definition")
        result.is_valid = False

    # Check for Isaac Lab imports
    required_imports = ["omni.isaac.lab"]
    for req in required_imports:
        if not any(req in imp for imp in found_imports):
            result.warnings.append(f"Missing expected import: {req}")

    return result


def validate_isaac_lab_task(code: str) -> CodeValidationResult:
    """
    Validate Isaac Lab task Python file.

    Args:
        code: Python source code of task file

    Returns:
        CodeValidationResult with any issues found
    """
    result = validate_python_syntax(code, "task.py")
    if not result.is_valid:
        return result

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return result

    found_classes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            found_classes.add(node.name)

    # Check for task class
    task_found = any(
        name.endswith("Env") or name.endswith("Task")
        for name in found_classes
    )
    if not task_found:
        result.warnings.append("No Env or Task class found in task file")

    return result


def validate_reward_functions(code: str) -> CodeValidationResult:
    """
    Validate reward functions module.

    Args:
        code: Python source code of reward_functions.py

    Returns:
        CodeValidationResult with any issues found
    """
    result = validate_python_syntax(code, "reward_functions.py")
    if not result.is_valid:
        return result

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return result

    found_functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found_functions.add(node.name)

    if not found_functions:
        result.warnings.append("No reward functions defined")

    return result


def validate_randomizations(code: str) -> CodeValidationResult:
    """
    Validate randomizations module.

    Checks for:
    - Valid Python syntax
    - Event-compatible function signatures
    - Required imports

    Args:
        code: Python source code of randomizations.py

    Returns:
        CodeValidationResult with any issues found
    """
    result = validate_python_syntax(code, "randomizations.py")
    if not result.is_valid:
        return result

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return result

    found_functions = set()
    found_imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found_functions.add(node.name)
            # Check for env parameter (EventManager compatible)
            if node.args.args:
                first_arg = node.args.args[0].arg
                if first_arg != "env":
                    result.warnings.append(
                        f"Function '{node.name}' should have 'env' as first parameter for EventManager"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            found_imports.add(node.module)

    if not found_functions:
        result.warnings.append("No randomization functions defined")

    return result


def validate_train_config_yaml(content: str) -> CodeValidationResult:
    """
    Validate training configuration YAML.

    Checks for:
    - Valid YAML syntax
    - Required sections (runner, experiment, etc.)

    Args:
        content: YAML content

    Returns:
        CodeValidationResult with any issues found
    """
    result = CodeValidationResult(is_valid=True)

    try:
        import yaml
        config = yaml.safe_load(content)
    except ImportError:
        # yaml module not available, skip validation
        result.warnings.append("YAML module not available, skipping validation")
        return result
    except yaml.YAMLError as e:
        result.add_error(f"YAML parse error: {e}")
        result.is_valid = False
        return result

    if not isinstance(config, dict):
        result.add_error("Training config must be a dictionary")
        result.is_valid = False
        return result

    # Check for expected top-level keys
    expected_keys = {"runner", "experiment"}
    optional_keys = {"algorithm", "observation", "reward", "policy"}

    for key in expected_keys:
        if key not in config:
            result.warnings.append(f"Missing expected key: {key}")

    # Check runner config
    runner = config.get("runner", {})
    if runner:
        if "num_envs" not in runner:
            result.warnings.append("runner.num_envs not specified")
        if "max_iterations" not in runner:
            result.warnings.append("runner.max_iterations not specified")

    return result


def validate_generated_isaac_lab_code(
    saved_files: Dict[str, str]
) -> CodeValidationResult:
    """
    Validate all generated Isaac Lab code files.

    Args:
        saved_files: Dict mapping filenames to their content

    Returns:
        Combined CodeValidationResult for all files
    """
    result = CodeValidationResult(is_valid=True)

    # Python file validators
    python_validators = {
        "env_cfg.py": validate_isaac_lab_env_config,
        "reward_functions.py": validate_reward_functions,
        "randomizations.py": validate_randomizations,
    }

    # YAML file validators
    yaml_validators = {
        "train_cfg.yaml": validate_train_config_yaml,
    }

    for filename, content in saved_files.items():
        file_result = None

        if filename.endswith(".py"):
            # Use specific validator if available
            if filename in python_validators:
                file_result = python_validators[filename](content)
            elif filename.startswith("task_"):
                file_result = validate_isaac_lab_task(content)
            else:
                file_result = validate_python_syntax(content, filename)

        elif filename.endswith(".yaml") or filename.endswith(".yml"):
            # Use specific YAML validator if available
            if filename in yaml_validators:
                file_result = yaml_validators[filename](content)
            # else: skip generic YAML validation to avoid dependency issues

        # Merge results if we got any
        if file_result:
            if not file_result.is_valid:
                result.is_valid = False
            result.errors.extend([f"[{filename}] {e}" for e in file_result.errors])
            result.warnings.extend([f"[{filename}] {w}" for w in file_result.warnings])

    return result

GCS_ROOT = Path("/mnt/gcs")


# =============================================================================
# Environment → Policy Mapping
# =============================================================================

ENVIRONMENT_POLICIES = {
    "kitchen": ["dish_loading", "articulated_access", "manipulation"],
    "warehouse": ["pick_place", "palletizing", "bin_picking"],
    "office": ["desk_organization", "manipulation"],
    "laundry": ["fabric_handling", "manipulation"],
    "generic": ["manipulation"],
}


def select_policy(environment_type: str, manifest: Dict[str, Any]) -> str:
    """Select the best policy based on environment and scene content."""
    policies = ENVIRONMENT_POLICIES.get(environment_type, ["manipulation"])

    # Check if scene has articulated objects
    has_articulated = any(
        obj.get("sim_role") in ["articulated_furniture", "articulated_appliance"]
        for obj in manifest.get("objects", [])
    )

    if has_articulated and "articulated_access" in policies:
        return "articulated_access"

    # Check if scene has manipulable objects
    has_manipulable = any(
        obj.get("sim_role") == "manipulable_object"
        for obj in manifest.get("objects", [])
    )

    if has_manipulable:
        if environment_type == "kitchen":
            return "dish_loading"
        elif environment_type == "warehouse":
            return "pick_place"

    return policies[0]


# =============================================================================
# Main Job
# =============================================================================

def run_isaac_lab_job(
    root: Path,
    scene_id: str,
    assets_prefix: str,
    usd_prefix: str,
    replicator_prefix: str,
    isaac_lab_prefix: str,
    environment_type: str = "generic",
    policy_id: Optional[str] = None,
    robot_type: str = "franka",
    num_envs: int = 1024,
    run_runtime_validation: bool = True,
    skip_sanity_rollout: bool = False,
    strict_config_loading: bool = False,
) -> int:
    """Run the Isaac Lab task generation job.

    Returns:
        0 on success, 1 on failure
    """
    logger.info("[ISAAC-LAB-JOB] Starting task generation for scene: %s", scene_id)
    logger.info("[ISAAC-LAB-JOB] Assets prefix: %s", assets_prefix)
    logger.info("[ISAAC-LAB-JOB] USD prefix: %s", usd_prefix)
    logger.info("[ISAAC-LAB-JOB] Replicator prefix: %s", replicator_prefix)
    logger.info("[ISAAC-LAB-JOB] Output prefix: %s", isaac_lab_prefix)
    logger.info("[ISAAC-LAB-JOB] Environment type: %s", environment_type)
    logger.info("[ISAAC-LAB-JOB] Robot type: %s", robot_type)
    logger.info("[ISAAC-LAB-JOB] Num envs: %s", num_envs)

    assets_dir = root / assets_prefix
    usd_dir = root / usd_prefix
    replicator_dir = root / replicator_prefix
    isaac_lab_dir = root / isaac_lab_prefix

    # Load manifest
    manifest_path = assets_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        logger.error("[ISAAC-LAB-JOB] Manifest not found at %s", manifest_path)
        return 1

    try:
        manifest = json.loads(manifest_path.read_text())
        logger.info(
            "[ISAAC-LAB-JOB] Loaded manifest with %s objects",
            len(manifest.get("objects", [])),
        )
    except Exception as e:
        logger.error("[ISAAC-LAB-JOB] Failed to load manifest: %s", e)
        return 1

    # Check USD scene exists
    scene_usda = usd_dir / "scene.usda"
    if not scene_usda.is_file():
        logger.warning("[ISAAC-LAB-JOB] scene.usda not found at %s", scene_usda)
        # Continue anyway, USD path will be used as reference

    # Select policy if not specified
    if not policy_id:
        policy_id = select_policy(environment_type, manifest)
        logger.info("[ISAAC-LAB-JOB] Auto-selected policy: %s", policy_id)
    else:
        logger.info("[ISAAC-LAB-JOB] Using specified policy: %s", policy_id)

    # Load policy config
    policy_config_path = REPO_ROOT / "policy_configs" / "environment_policies.json"
    if policy_config_path.is_file():
        try:
            policy_config = json.loads(policy_config_path.read_text())
            logger.info("[ISAAC-LAB-JOB] Loaded policy configuration")
        except (OSError, json.JSONDecodeError) as e:
            message = f"[ISAAC-LAB-JOB] WARNING: Failed to load policy config: {e}"
            if strict_config_loading:
                logger.error(message.replace("WARNING", "ERROR"))
                return 1
            logger.warning(message)
            policy_config = {"policies": {}, "environments": {}}
    else:
        message = (
            "[ISAAC-LAB-JOB] WARNING: Policy config not found at "
            f"{policy_config_path}, using defaults"
        )
        if strict_config_loading:
            logger.error(message.replace("WARNING", "ERROR"))
            return 1
        logger.warning(message)
        policy_config = {"policies": {}, "environments": {}}

    # Load replicator metadata if available
    replicator_metadata_path = replicator_dir / "bundle_metadata.json"
    if replicator_metadata_path.is_file():
        try:
            replicator_metadata = json.loads(replicator_metadata_path.read_text())
            logger.info("[ISAAC-LAB-JOB] Loaded replicator metadata")
        except (OSError, json.JSONDecodeError) as e:
            message = f"[ISAAC-LAB-JOB] WARNING: Failed to load replicator metadata: {e}"
            if strict_config_loading:
                logger.error(message.replace("WARNING", "ERROR"))
                return 1
            logger.warning(message)
            replicator_metadata = {}
    else:
        message = (
            "[ISAAC-LAB-JOB] WARNING: Replicator metadata not found at "
            f"{replicator_metadata_path}, using defaults"
        )
        if strict_config_loading:
            logger.error(message.replace("WARNING", "ERROR"))
            return 1
        logger.warning(message)
        replicator_metadata = {}

    # Build recipe from manifest + replicator data
    recipe = {
        "metadata": {
            "environment_type": environment_type,
            "scene_path": str(scene_usda),
            "scene_id": scene_id,
            "replicator_bundle": str(replicator_dir),
        },
        "room": manifest.get("scene", {}).get("room", {}),
        "objects": manifest.get("objects", []),
        "replicator": replicator_metadata,
    }

    # Create output directory
    isaac_lab_dir.mkdir(parents=True, exist_ok=True)

    # Generate task
    try:
        generator = IsaacLabTaskGenerator(policy_config)
        task = generator.generate(
            recipe=recipe,
            policy_id=policy_id,
            robot_type=robot_type,
            num_envs=num_envs,
        )
        logger.info("[ISAAC-LAB-JOB] Generated task: %s", task.task_name)
    except Exception as e:
        logger.exception("[ISAAC-LAB-JOB] Task generation failed: %s", e)
        return 1

    # Validate generated code BEFORE writing files
    logger.info("[ISAAC-LAB-JOB] Validating generated code...")
    validation_result = validate_generated_isaac_lab_code(task.files)

    if validation_result.warnings:
        logger.warning(
            "[ISAAC-LAB-JOB] Validation warnings (%s):", len(validation_result.warnings)
        )
        for warning in validation_result.warnings:
            logger.warning("[ISAAC-LAB-JOB]   WARNING: %s", warning)

    if not validation_result.is_valid:
        logger.error(
            "[ISAAC-LAB-JOB] Code validation failed (%s errors):",
            len(validation_result.errors),
        )
        for error in validation_result.errors:
            logger.error("[ISAAC-LAB-JOB]   ERROR: %s", error)
        logger.error("[ISAAC-LAB-JOB] Files NOT written due to validation errors")
        return 1

    logger.info("[ISAAC-LAB-JOB] Code validation passed")

    # Save task files (only after validation passes)
    try:
        saved_files = generator.save(task, str(isaac_lab_dir))
        logger.info("[ISAAC-LAB-JOB] Saved %s files:", len(saved_files))
        for filename in saved_files:
            logger.info("[ISAAC-LAB-JOB]   - %s", filename)
    except Exception as e:
        logger.error("[ISAAC-LAB-JOB] Failed to save task files: %s", e)
        return 1

    # ==========================================================================
    # RUNTIME VALIDATION - Actually test that the generated code works
    # ==========================================================================
    runtime_result: Optional[RuntimeValidationResult] = None

    if run_runtime_validation:
        logger.info("[ISAAC-LAB-JOB] Running runtime validation...")
        runtime_validator = IsaacLabRuntimeValidator(verbose=True)

        runtime_result = runtime_validator.validate(
            isaac_lab_dir=isaac_lab_dir,
            run_sanity_rollout=not skip_sanity_rollout,
            num_rollout_steps=10,
            num_envs=4,  # Use small number for validation
        )

        if not runtime_result.is_valid:
            logger.error("[ISAAC-LAB-JOB] Runtime validation failed!")
            for error in runtime_result.errors:
                logger.error("[ISAAC-LAB-JOB]   ERROR: %s", error)
            for warning in runtime_result.warnings:
                logger.warning("[ISAAC-LAB-JOB]   WARNING: %s", warning)
            # Don't fail the job - mark it as partial success
            # Generated code is syntactically valid but may not run
            logger.warning("[ISAAC-LAB-JOB] Code generated but runtime validation failed")
            logger.warning("[ISAAC-LAB-JOB] The generated code may need manual adjustments")
        else:
            logger.info("[ISAAC-LAB-JOB] ✓ Runtime validation passed!")
            if runtime_result.rollout_fps > 0:
                logger.info(
                    "[ISAAC-LAB-JOB]   Rollout FPS: %.1f", runtime_result.rollout_fps
                )
            if runtime_result.observation_shapes:
                logger.info(
                    "[ISAAC-LAB-JOB]   Observation shapes: %s",
                    runtime_result.observation_shapes,
                )
            if runtime_result.action_shape:
                logger.info(
                    "[ISAAC-LAB-JOB]   Action shape: %s", runtime_result.action_shape
                )

    # Write metadata
    metadata = {
        "scene_id": scene_id,
        "task_name": task.task_name,
        "policy_id": policy_id,
        "robot_type": robot_type,
        "num_envs": num_envs,
        "environment_type": environment_type,
        "files": list(saved_files.keys()),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "generator_version": "1.0.0",
        "runtime_validation": runtime_result.to_dict() if runtime_result else None,
    }
    metadata_path = isaac_lab_dir / "generation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("[ISAAC-LAB-JOB] Wrote metadata: %s", metadata_path)

    # Write completion marker
    marker_path = isaac_lab_dir / ".isaac_lab_complete"
    marker_content = {
        "status": "complete",
        "task_name": task.task_name,
        "policy_id": policy_id,
        "completed_at": datetime.utcnow().isoformat() + "Z",
        "runtime_validated": runtime_result.is_valid if runtime_result else None,
        "validation_errors": runtime_result.errors if runtime_result else [],
    }
    marker_path.write_text(json.dumps(marker_content, indent=2))

    logger.info("[ISAAC-LAB-JOB] ✓ Isaac Lab task generation completed successfully")
    logger.info("[ISAAC-LAB-JOB]   Task: %s", task.task_name)
    logger.info("[ISAAC-LAB-JOB]   Policy: %s", policy_id)
    logger.info("[ISAAC-LAB-JOB]   Files: %s", len(saved_files))
    logger.info("[ISAAC-LAB-JOB]   Output: %s", isaac_lab_dir)

    return 0


def main():
    """Main entry point."""
    # Get configuration from environment
    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
        },
        label="[ISAAC-LAB-JOB]",
    )
    bucket = os.environ["BUCKET"]
    scene_id = os.environ["SCENE_ID"]

    # Prefixes with defaults
    assets_prefix = os.getenv("ASSETS_PREFIX", f"scenes/{scene_id}/assets")
    usd_prefix = os.getenv("USD_PREFIX", f"scenes/{scene_id}/usd")
    replicator_prefix = os.getenv("REPLICATOR_PREFIX", f"scenes/{scene_id}/replicator")
    isaac_lab_prefix = os.getenv("ISAAC_LAB_PREFIX", f"scenes/{scene_id}/isaac_lab")

    # Optional configuration
    environment_type = os.getenv("ENVIRONMENT_TYPE", "generic")
    policy_id = os.getenv("POLICY_ID", None)
    robot_type = os.getenv("ROBOT_TYPE", "franka")
    num_envs = int(os.getenv("NUM_ENVS", "1024"))

    # Runtime validation settings
    # Set RUN_RUNTIME_VALIDATION=false to skip validation
    # Set SKIP_SANITY_ROLLOUT=true to skip environment steps (faster, less thorough)
    run_runtime_validation = os.getenv("RUN_RUNTIME_VALIDATION", "true").lower() == "true"
    skip_sanity_rollout = os.getenv("SKIP_SANITY_ROLLOUT", "false").lower() == "true"
    strict_config_loading = os.getenv("STRICT_CONFIG_LOADING", "false").lower() == "true"

    logger.info("[ISAAC-LAB-JOB] Configuration:")
    logger.info("[ISAAC-LAB-JOB]   Bucket: %s", bucket)
    logger.info("[ISAAC-LAB-JOB]   Scene ID: %s", scene_id)
    logger.info("[ISAAC-LAB-JOB]   Runtime validation: %s", run_runtime_validation)
    logger.info("[ISAAC-LAB-JOB]   Skip sanity rollout: %s", skip_sanity_rollout)
    logger.info("[ISAAC-LAB-JOB]   Strict config loading: %s", strict_config_loading)

    exit_code = run_isaac_lab_job(
        root=GCS_ROOT,
        scene_id=scene_id,
        assets_prefix=assets_prefix,
        usd_prefix=usd_prefix,
        replicator_prefix=replicator_prefix,
        isaac_lab_prefix=isaac_lab_prefix,
        environment_type=environment_type,
        policy_id=policy_id,
        robot_type=robot_type,
        num_envs=num_envs,
        run_runtime_validation=run_runtime_validation,
        skip_sanity_rollout=skip_sanity_rollout,
        strict_config_loading=strict_config_loading,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    from tools.startup_validation import validate_and_fail_fast

    validate_and_fail_fast(job_name="ISAAC-LAB", validate_gcs=True)
    main()
