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
"""

import ast
import json
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
) -> int:
    """Run the Isaac Lab task generation job.

    Returns:
        0 on success, 1 on failure
    """
    print(f"[ISAAC-LAB-JOB] Starting task generation for scene: {scene_id}")
    print(f"[ISAAC-LAB-JOB] Assets prefix: {assets_prefix}")
    print(f"[ISAAC-LAB-JOB] USD prefix: {usd_prefix}")
    print(f"[ISAAC-LAB-JOB] Replicator prefix: {replicator_prefix}")
    print(f"[ISAAC-LAB-JOB] Output prefix: {isaac_lab_prefix}")
    print(f"[ISAAC-LAB-JOB] Environment type: {environment_type}")
    print(f"[ISAAC-LAB-JOB] Robot type: {robot_type}")
    print(f"[ISAAC-LAB-JOB] Num envs: {num_envs}")

    assets_dir = root / assets_prefix
    usd_dir = root / usd_prefix
    replicator_dir = root / replicator_prefix
    isaac_lab_dir = root / isaac_lab_prefix

    # Load manifest
    manifest_path = assets_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        print(f"[ISAAC-LAB-JOB] ERROR: Manifest not found at {manifest_path}")
        return 1

    try:
        manifest = json.loads(manifest_path.read_text())
        print(f"[ISAAC-LAB-JOB] Loaded manifest with {len(manifest.get('objects', []))} objects")
    except Exception as e:
        print(f"[ISAAC-LAB-JOB] ERROR: Failed to load manifest: {e}")
        return 1

    # Check USD scene exists
    scene_usda = usd_dir / "scene.usda"
    if not scene_usda.is_file():
        print(f"[ISAAC-LAB-JOB] WARNING: scene.usda not found at {scene_usda}")
        # Continue anyway, USD path will be used as reference

    # Select policy if not specified
    if not policy_id:
        policy_id = select_policy(environment_type, manifest)
        print(f"[ISAAC-LAB-JOB] Auto-selected policy: {policy_id}")
    else:
        print(f"[ISAAC-LAB-JOB] Using specified policy: {policy_id}")

    # Load policy config
    policy_config_path = REPO_ROOT / "policy_configs" / "environment_policies.json"
    if policy_config_path.is_file():
        policy_config = json.loads(policy_config_path.read_text())
        print("[ISAAC-LAB-JOB] Loaded policy configuration")
    else:
        print("[ISAAC-LAB-JOB] WARNING: Policy config not found, using defaults")
        policy_config = {"policies": {}, "environments": {}}

    # Load replicator metadata if available
    replicator_metadata_path = replicator_dir / "bundle_metadata.json"
    if replicator_metadata_path.is_file():
        try:
            replicator_metadata = json.loads(replicator_metadata_path.read_text())
            print("[ISAAC-LAB-JOB] Loaded replicator metadata")
        except Exception:
            replicator_metadata = {}
    else:
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
        print(f"[ISAAC-LAB-JOB] Generated task: {task.task_name}")
    except Exception as e:
        print(f"[ISAAC-LAB-JOB] ERROR: Task generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Validate generated code BEFORE writing files
    print("[ISAAC-LAB-JOB] Validating generated code...")
    validation_result = validate_generated_isaac_lab_code(task.files)

    if validation_result.warnings:
        print(f"[ISAAC-LAB-JOB] Validation warnings ({len(validation_result.warnings)}):")
        for warning in validation_result.warnings:
            print(f"[ISAAC-LAB-JOB]   WARNING: {warning}")

    if not validation_result.is_valid:
        print(f"[ISAAC-LAB-JOB] ERROR: Code validation failed ({len(validation_result.errors)} errors):")
        for error in validation_result.errors:
            print(f"[ISAAC-LAB-JOB]   ERROR: {error}")
        print("[ISAAC-LAB-JOB] Files NOT written due to validation errors")
        return 1

    print("[ISAAC-LAB-JOB] Code validation passed")

    # Save task files (only after validation passes)
    try:
        saved_files = generator.save(task, str(isaac_lab_dir))
        print(f"[ISAAC-LAB-JOB] Saved {len(saved_files)} files:")
        for filename in saved_files:
            print(f"[ISAAC-LAB-JOB]   - {filename}")
    except Exception as e:
        print(f"[ISAAC-LAB-JOB] ERROR: Failed to save task files: {e}")
        return 1

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
    }
    metadata_path = isaac_lab_dir / "generation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[ISAAC-LAB-JOB] Wrote metadata: {metadata_path}")

    # Write completion marker
    marker_path = isaac_lab_dir / ".isaac_lab_complete"
    marker_content = {
        "status": "complete",
        "task_name": task.task_name,
        "policy_id": policy_id,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
    marker_path.write_text(json.dumps(marker_content, indent=2))

    print("[ISAAC-LAB-JOB] ✓ Isaac Lab task generation completed successfully")
    print(f"[ISAAC-LAB-JOB]   Task: {task.task_name}")
    print(f"[ISAAC-LAB-JOB]   Policy: {policy_id}")
    print(f"[ISAAC-LAB-JOB]   Files: {len(saved_files)}")
    print(f"[ISAAC-LAB-JOB]   Output: {isaac_lab_dir}")

    return 0


def main():
    """Main entry point."""
    # Get configuration from environment
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")

    if not scene_id:
        print("[ISAAC-LAB-JOB] ERROR: SCENE_ID is required")
        sys.exit(1)

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

    print(f"[ISAAC-LAB-JOB] Configuration:")
    print(f"[ISAAC-LAB-JOB]   Bucket: {bucket}")
    print(f"[ISAAC-LAB-JOB]   Scene ID: {scene_id}")

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
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
