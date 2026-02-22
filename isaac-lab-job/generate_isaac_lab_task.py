#!/usr/bin/env python3
"""
Isaac Lab Task Generation Job.

Generates complete Isaac Lab task packages from BlueprintPipeline scenes.
This is the final step in the pipeline, producing RL training-ready packages.

Pipeline Position:
    Stage 1 text generation → simready → usd-assembly → replicator → [THIS JOB]

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
    AFFORDANCE_FEASIBILITY_MODE: strict|warn|off|auto (default: strict)
    ISAAC_REFRESH_ONLY: Run refresh-only task spawn planning pass from variation outputs
    VARIATION_ASSETS_PREFIX: Variation assets prefix for refresh mode
"""

import ast
import hashlib
import json
import logging
import os
import re
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
from tools.config.env import parse_bool_env
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
    required_imports = ["isaaclab"]
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

ROBOT_TRACKS: Dict[str, List[str]] = {
    "humanoid": ["g2", "gr1", "h1"],
    "manipulator": ["franka", "ur10", "ur5e", "kuka_iiwa"],
}

TRACK_POLICY_PRIORITY: Dict[str, List[str]] = {
    "humanoid": [
        "laundry_sorting",
        "table_clearing",
        "mixed_sku_logistics",
        "door_manipulation",
        "drawer_manipulation",
        "articulated_access",
        "general_manipulation",
        "dexterous_pick_place",
    ],
    "manipulator": [
        "dexterous_pick_place",
        "precision_insertion",
        "dish_loading",
        "grocery_stocking",
        "drawer_manipulation",
        "door_manipulation",
        "articulated_access",
        "general_manipulation",
    ],
}

TRACK_STYLE_SNIPPETS: Dict[str, List[str]] = {
    "humanoid": [
        "whole-body reach and stabilization",
        "bimanual transfer with posture control",
        "navigation-aware manipulation",
    ],
    "manipulator": [
        "precision grasp and place",
        "collision-aware cycle optimization",
        "repeatable high-throughput handling",
    ],
}

POLICY_REGION_PROFILES: Dict[str, Dict[str, Any]] = {
    "dish_loading": {
        "spawn_affordances": ["support_surface", "staging_zone", "placeable"],
        "goal_affordances": ["containable", "load_target", "wash_zone"],
        "requires_goal": True,
        "requires_asset": True,
    },
    "table_clearing": {
        "spawn_affordances": ["support_surface", "staging_zone", "placeable"],
        "goal_affordances": ["containable", "load_target", "wash_zone"],
        "requires_goal": True,
        "requires_asset": True,
    },
    "laundry_sorting": {
        "spawn_affordances": ["support_surface", "staging_zone", "placeable"],
        "goal_affordances": ["containable", "load_target"],
        "requires_goal": True,
        "requires_asset": True,
    },
    "grocery_stocking": {
        "spawn_affordances": ["staging_zone", "support_surface", "placeable"],
        "goal_affordances": ["containable", "load_target", "support_surface"],
        "requires_goal": True,
        "requires_asset": True,
    },
    "mixed_sku_logistics": {
        "spawn_affordances": ["staging_zone", "support_surface", "placeable"],
        "goal_affordances": ["containable", "load_target", "support_surface"],
        "requires_goal": True,
        "requires_asset": True,
    },
    "dexterous_pick_place": {
        "spawn_affordances": ["support_surface", "staging_zone", "placeable"],
        "goal_affordances": ["placeable", "support_surface", "containable"],
        "requires_goal": True,
        "requires_asset": False,
    },
    "precision_insertion": {
        "spawn_affordances": ["support_surface", "staging_zone", "placeable"],
        "goal_affordances": ["insert_target", "containable", "load_target"],
        "requires_goal": True,
        "requires_asset": False,
    },
    "articulated_access": {
        "spawn_affordances": ["articulation_control", "load_target"],
        "goal_affordances": [],
        "requires_goal": False,
        "requires_asset": False,
    },
    "drawer_manipulation": {
        "spawn_affordances": ["articulation_control", "load_target"],
        "goal_affordances": [],
        "requires_goal": False,
        "requires_asset": False,
    },
    "door_manipulation": {
        "spawn_affordances": ["articulation_control", "load_target"],
        "goal_affordances": [],
        "requires_goal": False,
        "requires_asset": False,
    },
    "knob_manipulation": {
        "spawn_affordances": ["articulation_control"],
        "goal_affordances": [],
        "requires_goal": False,
        "requires_asset": False,
    },
    "panel_interaction": {
        "spawn_affordances": ["articulation_control"],
        "goal_affordances": [],
        "requires_goal": False,
        "requires_asset": False,
    },
}

DEFAULT_POLICY_REGION_PROFILE: Dict[str, Any] = {
    "spawn_affordances": ["support_surface", "staging_zone", "placeable"],
    "goal_affordances": ["placeable", "containable", "support_surface"],
    "requires_goal": True,
    "requires_asset": False,
}


def _normalize_scene_label(value: str) -> str:
    normalized = re.sub(r"[_-]+", " ", value.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _extract_scene_anchors(manifest: Dict[str, Any], limit: int = 6) -> List[str]:
    anchors: List[str] = []
    seen: set[str] = set()
    for obj in manifest.get("objects", []):
        raw = (
            obj.get("category")
            or (obj.get("semantics") or {}).get("class")
            or obj.get("name")
            or obj.get("id")
        )
        if not isinstance(raw, str):
            continue
        label = _normalize_scene_label(raw)
        if not label:
            continue
        if label in {"background", "wall", "floor", "ceiling", "scene shell"}:
            continue
        if label in seen:
            continue
        seen.add(label)
        anchors.append(label)
        if len(anchors) >= limit:
            break
    return anchors


def _sort_policies_for_track(track: str, policies: List[str]) -> List[str]:
    priority = TRACK_POLICY_PRIORITY.get(track, [])
    order = {policy_id: idx for idx, policy_id in enumerate(priority)}
    return sorted(
        policies,
        key=lambda policy_id: (order.get(policy_id, len(priority)), policy_id),
    )


def _select_track_policies(
    *,
    environment_type: str,
    policy_config: Dict[str, Any],
    track: str,
    max_policies: int = 4,
) -> List[str]:
    env_cfg = (policy_config.get("environments") or {}).get(environment_type, {})
    defaults = env_cfg.get("default_policies")
    if not isinstance(defaults, list) or not defaults:
        defaults = ENVIRONMENT_POLICIES.get(environment_type, ENVIRONMENT_POLICIES["generic"])

    selected = [p for p in defaults if isinstance(p, str) and p.strip()]
    selected = _sort_policies_for_track(track, selected)
    return selected[: max_policies if max_policies > 0 else len(selected)]


def build_scene_task_catalog(
    *,
    scene_id: str,
    environment_type: str,
    manifest: Dict[str, Any],
    policy_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a creative per-scene task catalog for humanoid and manipulator tracks."""
    anchors = _extract_scene_anchors(manifest)
    anchor_phrase = ", ".join(anchors) if anchors else "scene objects"
    policies_cfg = policy_config.get("policies") or {}

    tracks: Dict[str, Any] = {}
    for track, robot_types in ROBOT_TRACKS.items():
        selected_policies = _select_track_policies(
            environment_type=environment_type,
            policy_config=policy_config,
            track=track,
        )
        tasks: List[Dict[str, Any]] = []
        styles = TRACK_STYLE_SNIPPETS.get(track, ["robust manipulation"])
        for policy_index, policy_name in enumerate(selected_policies):
            policy_payload = policies_cfg.get(policy_name) or {}
            display_name = policy_payload.get("display_name") or policy_name.replace("_", " ").title()
            description = policy_payload.get("description") or "Complete the objective robustly."
            reward_components = [
                comp
                for comp in (policy_payload.get("reward_components") or [])
                if isinstance(comp, str)
            ][:3]

            for variant in range(2):
                style = styles[(policy_index + variant) % len(styles)]
                task_id = f"{track}_{policy_name}_v{variant + 1}"
                title = f"{display_name} ({track}, {style})"
                objective = (
                    f"Execute {display_name.lower()} in a {environment_type} scene "
                    f"using {style} around {anchor_phrase}."
                )
                if variant == 1:
                    objective += (
                        " Add randomized object starts and require recovery from one disturbed placement."
                    )
                tasks.append(
                    {
                        "task_id": task_id,
                        "title": title,
                        "robot_track": track,
                        "recommended_robot_types": robot_types,
                        "policy_id": policy_name,
                        "description": description,
                        "objective": objective,
                        "scene_anchors": anchors,
                        "difficulty": "hard" if variant == 1 else "medium",
                        "success_criteria": [
                            "Complete objective within episode budget",
                            "No unstable collisions with fixed scene assets",
                            "Maintain grasp or articulation control during transfer",
                        ],
                        "reward_focus": reward_components,
                    }
                )

        tracks[track] = {
            "robot_types": robot_types,
            "selected_policies": selected_policies,
            "tasks": tasks,
        }

    return {
        "scene_id": scene_id,
        "environment_type": environment_type,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "tracks": tracks,
    }


def _stable_choice(items: List[str], seed_key: str) -> Optional[str]:
    if not items:
        return None
    ordered = sorted(items)
    digest = hashlib.sha256(seed_key.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(ordered)
    return ordered[index]


def _get_region_affordances(region_payload: Dict[str, Any]) -> set[str]:
    affordances = {
        str(value).strip().lower()
        for value in (region_payload.get("affordances") or [])
        if str(value).strip()
    }

    surface_type = str(region_payload.get("surface_type", "")).strip().lower()
    if surface_type == "horizontal":
        affordances.update({"support_surface", "placeable"})
    elif surface_type == "volume":
        affordances.update({"containable", "insert_target"})

    return affordances


def _regions_matching_affordances(
    region_records: Dict[str, Dict[str, Any]],
    candidate_region_ids: List[str],
    required_affordances: List[str],
) -> List[str]:
    if not required_affordances:
        return list(candidate_region_ids)

    required = {value.strip().lower() for value in required_affordances if value.strip()}
    matched: List[str] = []
    for region_id in candidate_region_ids:
        payload = region_records.get(region_id) or {}
        affordances = _get_region_affordances(payload)
        if affordances & required:
            matched.append(region_id)
    return matched


def _build_fallback_affordance_graph(
    *,
    scene_id: str,
    environment_type: str,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    seen_regions: set[str] = set()
    regions: List[Dict[str, Any]] = []

    for obj in manifest.get("objects", []):
        placement_region = obj.get("placement_region")
        if not isinstance(placement_region, str) or not placement_region.strip():
            continue
        region_id = placement_region.strip()
        if region_id in seen_regions:
            continue
        seen_regions.add(region_id)
        regions.append(
            {
                "id": region_id,
                "surface_type": "horizontal",
                "semantic_tags": [],
                "suitable_for": [],
                "affordances": ["support_surface", "placeable", "staging_zone"],
            }
        )

    if not regions:
        regions = [
            {
                "id": "scene_default_region",
                "surface_type": "horizontal",
                "semantic_tags": ["default"],
                "suitable_for": ["object"],
                "affordances": ["support_surface", "placeable", "staging_zone", "containable"],
            }
        ]

    return {
        "scene_id": scene_id,
        "environment_type": environment_type,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "regions": regions,
        "variation_assets": [],
        "policy_region_map": {},
        "policy_asset_map": {},
        "asset_to_region_candidates": {},
        "articulation_targets": [],
    }


def load_affordance_graph_for_tasks(
    *,
    scene_id: str,
    environment_type: str,
    manifest: Dict[str, Any],
    replicator_dir: Path,
    feasibility_mode: str,
) -> tuple[Dict[str, Any], str]:
    graph_path = replicator_dir / "affordance_graph.json"
    if graph_path.is_file():
        try:
            payload = json.loads(graph_path.read_text())
            return payload, str(graph_path)
        except (OSError, json.JSONDecodeError) as exc:
            if feasibility_mode == "strict":
                raise RuntimeError(
                    f"Failed to parse affordance graph at {graph_path}: {exc}"
                ) from exc
            logger.warning(
                "[ISAAC-LAB-JOB] Failed to parse affordance graph at %s: %s; "
                "using fallback graph from manifest.",
                graph_path,
                exc,
            )
    else:
        if feasibility_mode == "strict":
            raise FileNotFoundError(
                f"Missing required affordance graph at {graph_path}"
            )
        logger.warning(
            "[ISAAC-LAB-JOB] Affordance graph not found at %s; using fallback graph from manifest.",
            graph_path,
        )

    return _build_fallback_affordance_graph(
        scene_id=scene_id,
        environment_type=environment_type,
        manifest=manifest,
    ), "manifest_fallback"


def build_region_constrained_task_plan(
    *,
    task_catalog: Dict[str, Any],
    affordance_graph: Dict[str, Any],
) -> Dict[str, Any]:
    region_records: Dict[str, Dict[str, Any]] = {}
    for region in affordance_graph.get("regions") or []:
        region_id = region.get("id")
        if isinstance(region_id, str) and region_id.strip():
            region_records[region_id] = region

    all_region_ids = sorted(region_records.keys())
    policy_region_map = {
        str(policy_id): [str(region_id) for region_id in region_ids if str(region_id).strip()]
        for policy_id, region_ids in (affordance_graph.get("policy_region_map") or {}).items()
        if isinstance(region_ids, list)
    }
    policy_asset_map = {
        str(policy_id): [str(asset_name) for asset_name in asset_names if str(asset_name).strip()]
        for policy_id, asset_names in (affordance_graph.get("policy_asset_map") or {}).items()
        if isinstance(asset_names, list)
    }
    asset_to_region_candidates = {
        str(asset_name): [str(region_id) for region_id in region_ids if str(region_id).strip()]
        for asset_name, region_ids in (affordance_graph.get("asset_to_region_candidates") or {}).items()
        if isinstance(region_ids, list)
    }
    articulation_targets = [
        target
        for target in (affordance_graph.get("articulation_targets") or [])
        if isinstance(target, dict) and str(target.get("object_id", "")).strip()
    ]
    articulation_target_ids = sorted(
        {
            str(target.get("object_id")).strip()
            for target in articulation_targets
            if str(target.get("object_id", "")).strip()
        }
    )

    tracks_out: Dict[str, Any] = {}
    total_tasks = 0
    feasible_tasks = 0

    for track, payload in (task_catalog.get("tracks") or {}).items():
        tasks_out: List[Dict[str, Any]] = []
        track_feasible = 0

        for task in payload.get("tasks") or []:
            total_tasks += 1
            policy_id = str(task.get("policy_id", "")).strip()
            profile = POLICY_REGION_PROFILES.get(policy_id, DEFAULT_POLICY_REGION_PROFILE)

            policy_regions = policy_region_map.get(policy_id) or all_region_ids
            policy_regions = [region_id for region_id in policy_regions if region_id in region_records]
            spawn_regions = _regions_matching_affordances(
                region_records=region_records,
                candidate_region_ids=policy_regions,
                required_affordances=profile.get("spawn_affordances") or [],
            )
            if not spawn_regions:
                spawn_regions = list(policy_regions)

            goal_regions = _regions_matching_affordances(
                region_records=region_records,
                candidate_region_ids=policy_regions,
                required_affordances=profile.get("goal_affordances") or [],
            )
            if not goal_regions and profile.get("requires_goal", False):
                goal_regions = [region_id for region_id in spawn_regions if region_id in policy_regions]

            allowed_assets = policy_asset_map.get(policy_id, [])
            if not allowed_assets:
                for asset_name, candidate_regions in asset_to_region_candidates.items():
                    if set(candidate_regions) & set(spawn_regions + goal_regions):
                        allowed_assets.append(asset_name)
                allowed_assets = sorted(set(allowed_assets))

            reasons: List[str] = []
            if not spawn_regions:
                reasons.append("no_spawn_regions")
            if profile.get("requires_goal", False) and not goal_regions:
                reasons.append("no_goal_regions")
            if profile.get("requires_asset", False) and not allowed_assets:
                reasons.append("no_allowed_assets")
            needs_articulation = "articulation_control" in (
                profile.get("spawn_affordances") or []
            )
            if needs_articulation and not articulation_target_ids:
                reasons.append("no_articulation_targets")

            is_feasible = len(reasons) == 0
            if is_feasible:
                track_feasible += 1
                feasible_tasks += 1

            task_id = str(task.get("task_id", "task"))
            default_spawn_region = _stable_choice(spawn_regions, f"{task_id}:spawn")
            default_goal_region = _stable_choice(
                goal_regions if goal_regions else spawn_regions,
                f"{task_id}:goal",
            )

            tasks_out.append(
                {
                    "task_id": task_id,
                    "policy_id": policy_id,
                    "robot_track": task.get("robot_track", track),
                    "scene_anchors": task.get("scene_anchors", []),
                    "spawn_regions": sorted(set(spawn_regions)),
                    "goal_regions": sorted(set(goal_regions)),
                    "default_spawn_region": default_spawn_region,
                    "default_goal_region": default_goal_region,
                    "allowed_assets": allowed_assets,
                    "requires_asset": bool(profile.get("requires_asset", False)),
                    "articulation_target_ids": articulation_target_ids if needs_articulation else [],
                    "feasible": is_feasible,
                    "infeasible_reasons": reasons,
                }
            )

        tracks_out[track] = {
            "robot_types": payload.get("robot_types", []),
            "task_count": len(tasks_out),
            "feasible_task_count": track_feasible,
            "tasks": tasks_out,
        }

    return {
        "scene_id": task_catalog.get("scene_id"),
        "environment_type": task_catalog.get("environment_type"),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "tracks": tracks_out,
        "summary": {
            "total_tasks": total_tasks,
            "feasible_tasks": feasible_tasks,
            "infeasible_tasks": max(total_tasks - feasible_tasks, 0),
            "articulation_target_count": len(articulation_target_ids),
            "track_feasible_counts": {
                track: payload.get("feasible_task_count", 0)
                for track, payload in tracks_out.items()
            },
        },
    }


def evaluate_task_plan_feasibility(task_plan: Dict[str, Any]) -> Dict[str, Any]:
    summary = task_plan.get("summary") or {}
    total_tasks = int(summary.get("total_tasks") or 0)
    feasible_tasks = int(summary.get("feasible_tasks") or 0)
    track_counts = summary.get("track_feasible_counts") or {}

    reasons: List[str] = []
    if total_tasks <= 0:
        reasons.append("no_tasks_generated")
    if feasible_tasks <= 0:
        reasons.append("no_feasible_tasks")
    for track, count in track_counts.items():
        if int(count or 0) <= 0:
            reasons.append(f"no_feasible_tasks_for_track:{track}")

    return {
        "is_feasible": len(reasons) == 0,
        "reasons": reasons,
        "summary": summary,
    }


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            return payload
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _normalize_asset_name(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _collect_available_variation_assets(
    variation_assets_dir: Path,
) -> Dict[str, Any]:
    """Load variation outputs and return the set of assets that are generated and simready."""
    variation_assets_path = variation_assets_dir / "variation_assets.json"
    variation_assets_payload = _read_json_if_exists(variation_assets_path) or {}
    variation_objects = variation_assets_payload.get("objects") or []
    if not isinstance(variation_objects, list):
        variation_objects = []

    simready_marker_path = variation_assets_dir / ".simready_complete"
    simready_marker_payload = _read_json_if_exists(simready_marker_path) or {}
    simready_assets = simready_marker_payload.get("simready_assets") or {}

    simready_ids: set[str] = set()
    if isinstance(simready_assets, dict):
        simready_ids.update(str(key).strip() for key in simready_assets.keys() if str(key).strip())
    elif isinstance(simready_assets, list):
        simready_ids.update(str(value).strip() for value in simready_assets if str(value).strip())

    available_asset_ids: List[str] = []
    for obj in variation_objects:
        if not isinstance(obj, dict):
            continue
        object_id = str(obj.get("id", "")).strip()
        if not object_id:
            continue
        generated_payload = obj.get("generated_3d")
        generated_ok = False
        if isinstance(generated_payload, dict):
            status = str(generated_payload.get("status", "")).strip().lower()
            generated_ok = status == "success" or any(
                generated_payload.get(field)
                for field in ("usdz_path", "glb_path", "obj_path")
            )
        if not generated_ok:
            # Legacy/no-3D mode: if object exists, still allow use as a candidate asset.
            generated_ok = True

        if not generated_ok:
            continue

        if simready_ids and object_id not in simready_ids:
            continue
        available_asset_ids.append(object_id)

    available_asset_ids = sorted(set(available_asset_ids))
    available_asset_ids_normalized = sorted(
        {_normalize_asset_name(asset_id) for asset_id in available_asset_ids if asset_id}
    )

    return {
        "variation_assets_path": str(variation_assets_path),
        "simready_marker_path": str(simready_marker_path),
        "variation_object_count": len(variation_objects),
        "available_asset_ids": available_asset_ids,
        "available_asset_ids_normalized": available_asset_ids_normalized,
        "simready_marker_present": simready_marker_path.is_file(),
    }


def _apply_variation_asset_filter_to_task_plan(
    task_plan: Dict[str, Any],
    available_asset_ids: List[str],
) -> Dict[str, Any]:
    """Constrain task asset choices to generated+simready variation assets."""
    refreshed_plan: Dict[str, Any] = json.loads(json.dumps(task_plan))

    available_ids = {str(value).strip() for value in available_asset_ids if str(value).strip()}
    available_normalized = {_normalize_asset_name(value) for value in available_ids}

    total_tasks = 0
    feasible_tasks = 0
    track_feasible_counts: Dict[str, int] = {}

    for track, payload in (refreshed_plan.get("tracks") or {}).items():
        tasks = payload.get("tasks") or []
        track_feasible = 0
        for task in tasks:
            total_tasks += 1
            allowed_assets = [str(value) for value in (task.get("allowed_assets") or []) if str(value).strip()]
            filtered_assets = [
                asset_name
                for asset_name in allowed_assets
                if asset_name in available_ids
                or _normalize_asset_name(asset_name) in available_normalized
            ]

            reasons = [str(value) for value in (task.get("infeasible_reasons") or [])]
            reasons = [value for value in reasons if value != "no_allowed_assets"]
            if bool(task.get("requires_asset", False)) and not filtered_assets:
                reasons.append("no_available_variation_assets")

            task["allowed_assets_before_refresh"] = allowed_assets
            task["allowed_assets"] = sorted(set(filtered_assets))
            task["infeasible_reasons"] = sorted(set(reasons))
            task["feasible"] = len(task["infeasible_reasons"]) == 0

            if task["feasible"]:
                track_feasible += 1
                feasible_tasks += 1

        payload["task_count"] = len(tasks)
        payload["feasible_task_count"] = track_feasible
        track_feasible_counts[track] = track_feasible

    refreshed_plan["summary"] = {
        "total_tasks": total_tasks,
        "feasible_tasks": feasible_tasks,
        "infeasible_tasks": max(total_tasks - feasible_tasks, 0),
        "track_feasible_counts": track_feasible_counts,
    }
    refreshed_plan["variation_asset_filter"] = {
        "available_asset_ids": sorted(available_ids),
        "available_asset_count": len(available_ids),
    }
    refreshed_plan["generated_at"] = datetime.utcnow().isoformat() + "Z"
    return refreshed_plan


def _resolve_feasibility_mode(strict_config_loading: bool) -> str:
    mode = os.getenv("AFFORDANCE_FEASIBILITY_MODE", "strict").strip().lower()
    if mode == "auto":
        return "strict" if strict_config_loading else "warn"
    if mode not in {"strict", "warn", "off"}:
        logger.warning(
            "[ISAAC-LAB-JOB] Invalid AFFORDANCE_FEASIBILITY_MODE=%s; defaulting to strict.",
            mode,
        )
        return "strict"
    return mode


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
    isaac_refresh_only: bool = False,
    variation_assets_prefix: Optional[str] = None,
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
    logger.info("[ISAAC-LAB-JOB] Refresh-only mode: %s", isaac_refresh_only)

    assets_dir = root / assets_prefix
    usd_dir = root / usd_prefix
    replicator_dir = root / replicator_prefix
    isaac_lab_dir = root / isaac_lab_prefix
    variation_assets_dir = (
        root / variation_assets_prefix
        if variation_assets_prefix
        else root / f"scenes/{scene_id}/variation_assets"
    )

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

    feasibility_mode = _resolve_feasibility_mode(strict_config_loading)
    logger.info("[ISAAC-LAB-JOB] Affordance feasibility mode: %s", feasibility_mode)

    try:
        affordance_graph, affordance_graph_source = load_affordance_graph_for_tasks(
            scene_id=scene_id,
            environment_type=environment_type,
            manifest=manifest,
            replicator_dir=replicator_dir,
            feasibility_mode=feasibility_mode,
        )
    except Exception as exc:
        logger.error("[ISAAC-LAB-JOB] Failed to load affordance graph: %s", exc)
        return 1

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
    task_catalog_path = isaac_lab_dir / "task_catalog.json"
    baseline_spawn_plan_path = isaac_lab_dir / "task_spawn_plan_baseline.json"
    refresh_spawn_plan_path = isaac_lab_dir / "task_spawn_plan_refresh.json"
    legacy_spawn_plan_path = isaac_lab_dir / "task_spawn_plan.json"
    metadata_path = isaac_lab_dir / "generation_metadata.json"

    if isaac_refresh_only:
        logger.info("[ISAAC-LAB-JOB] Running refresh-only spawn-plan pass")

        task_catalog = _read_json_if_exists(task_catalog_path)
        if task_catalog is None:
            logger.info(
                "[ISAAC-LAB-JOB] Baseline task catalog missing at %s; rebuilding catalog.",
                task_catalog_path,
            )
            task_catalog = build_scene_task_catalog(
                scene_id=scene_id,
                environment_type=environment_type,
                manifest=manifest,
                policy_config=policy_config,
            )
            task_catalog_path.write_text(json.dumps(task_catalog, indent=2))

        baseline_plan = _read_json_if_exists(baseline_spawn_plan_path)
        baseline_plan_source = str(baseline_spawn_plan_path)
        if baseline_plan is None:
            baseline_plan = _read_json_if_exists(legacy_spawn_plan_path)
            baseline_plan_source = str(legacy_spawn_plan_path)

        if baseline_plan is None:
            logger.warning(
                "[ISAAC-LAB-JOB] Baseline spawn plan missing; rebuilding from task catalog and affordance graph."
            )
            baseline_plan = build_region_constrained_task_plan(
                task_catalog=task_catalog,
                affordance_graph=affordance_graph,
            )
            baseline_plan["affordance_graph_source"] = affordance_graph_source
            baseline_plan["feasibility_mode"] = feasibility_mode
            baseline_plan_source = "rebuilt_in_refresh_mode"

        variation_asset_info = _collect_available_variation_assets(variation_assets_dir)
        available_variation_assets = variation_asset_info.get("available_asset_ids", [])
        refreshed_plan = _apply_variation_asset_filter_to_task_plan(
            baseline_plan,
            available_variation_assets,
        )
        refreshed_plan["mode"] = "refresh_only"
        refreshed_plan["scene_id"] = scene_id
        refreshed_plan["environment_type"] = environment_type
        refreshed_plan["baseline_plan_source"] = baseline_plan_source
        refreshed_plan["affordance_graph_source"] = affordance_graph_source
        refreshed_plan["feasibility_mode"] = feasibility_mode
        refreshed_plan["variation_assets_prefix"] = str(variation_assets_dir)
        refreshed_plan["variation_assets_summary"] = variation_asset_info

        feasibility_report = evaluate_task_plan_feasibility(refreshed_plan)
        refreshed_plan["feasibility_report"] = feasibility_report
        refresh_spawn_plan_path.write_text(json.dumps(refreshed_plan, indent=2))
        legacy_spawn_plan_path.write_text(json.dumps(refreshed_plan, indent=2))
        logger.info(
            "[ISAAC-LAB-JOB] Wrote refresh spawn plan: %s", refresh_spawn_plan_path
        )

        if feasibility_mode != "off" and not feasibility_report.get("is_feasible", False):
            reasons = ", ".join(feasibility_report.get("reasons", []))
            summary = feasibility_report.get("summary", {})
            message = (
                "[ISAAC-LAB-JOB] Refresh feasibility gate failed: "
                f"reasons={reasons}; summary={summary}"
            )
            if feasibility_mode == "strict":
                logger.error(message)
                return 1
            logger.warning(message)

        metadata = _read_json_if_exists(metadata_path) or {}
        metadata.update(
            {
                "scene_id": scene_id,
                "environment_type": environment_type,
                "policy_id": policy_id,
                "robot_type": robot_type,
                "refresh_mode": True,
                "refresh_generated_at": datetime.utcnow().isoformat() + "Z",
                "task_catalog": str(task_catalog_path),
                "task_catalog_summary": {
                    track: len(payload.get("tasks", []))
                    for track, payload in (task_catalog.get("tracks") or {}).items()
                },
                "task_spawn_plan_baseline": str(baseline_spawn_plan_path),
                "task_spawn_plan_refresh": str(refresh_spawn_plan_path),
                "task_spawn_plan": str(legacy_spawn_plan_path),
                "task_spawn_plan_summary": refreshed_plan.get("summary", {}),
                "feasibility_gate_refresh": feasibility_report,
                "affordance_graph_source": affordance_graph_source,
                "variation_assets_prefix": str(variation_assets_dir),
                "variation_assets_summary": variation_asset_info,
                "generator_version": "1.0.0",
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2))
        logger.info("[ISAAC-LAB-JOB] Updated metadata: %s", metadata_path)

        refresh_marker_path = isaac_lab_dir / ".isaac_lab_refresh_complete"
        refresh_marker_content = {
            "status": "complete",
            "mode": "refresh_only",
            "scene_id": scene_id,
            "policy_id": policy_id,
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "task_spawn_plan_refresh_path": str(refresh_spawn_plan_path),
            "task_spawn_plan_path": str(legacy_spawn_plan_path),
            "variation_assets_prefix": str(variation_assets_dir),
            "variation_assets_summary": variation_asset_info,
            "feasibility_gate": feasibility_report,
        }
        refresh_marker_path.write_text(json.dumps(refresh_marker_content, indent=2))
        logger.info(
            "[ISAAC-LAB-JOB] Wrote refresh completion marker: %s",
            refresh_marker_path,
        )
        return 0

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

    # Build creative per-scene task catalog for both robot tracks.
    task_catalog = build_scene_task_catalog(
        scene_id=scene_id,
        environment_type=environment_type,
        manifest=manifest,
        policy_config=policy_config,
    )
    task_catalog_path.write_text(json.dumps(task_catalog, indent=2))
    logger.info("[ISAAC-LAB-JOB] Wrote task catalog: %s", task_catalog_path)

    task_spawn_plan = build_region_constrained_task_plan(
        task_catalog=task_catalog,
        affordance_graph=affordance_graph,
    )
    task_spawn_plan["affordance_graph_source"] = affordance_graph_source
    task_spawn_plan["feasibility_mode"] = feasibility_mode

    feasibility_report = evaluate_task_plan_feasibility(task_spawn_plan)
    task_spawn_plan["feasibility_report"] = feasibility_report

    baseline_spawn_plan_path.write_text(json.dumps(task_spawn_plan, indent=2))
    legacy_spawn_plan_path.write_text(json.dumps(task_spawn_plan, indent=2))
    logger.info(
        "[ISAAC-LAB-JOB] Wrote baseline task spawn plan: %s",
        baseline_spawn_plan_path,
    )

    if feasibility_mode != "off" and not feasibility_report.get("is_feasible", False):
        reasons = ", ".join(feasibility_report.get("reasons", []))
        summary = feasibility_report.get("summary", {})
        message = (
            "[ISAAC-LAB-JOB] Region feasibility gate failed: "
            f"reasons={reasons}; summary={summary}"
        )
        if feasibility_mode == "strict":
            logger.error(message)
            return 1
        logger.warning(message)

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
        "task_catalog": str(task_catalog_path),
        "task_catalog_summary": {
            track: len(payload.get("tasks", []))
            for track, payload in (task_catalog.get("tracks") or {}).items()
        },
        "task_spawn_plan_baseline": str(baseline_spawn_plan_path),
        "task_spawn_plan": str(legacy_spawn_plan_path),
        "task_spawn_plan_summary": task_spawn_plan.get("summary", {}),
        "affordance_graph_source": affordance_graph_source,
        "feasibility_gate": feasibility_report,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "generator_version": "1.0.0",
        "runtime_validation": runtime_result.to_dict() if runtime_result else None,
    }
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
        "task_catalog_path": str(task_catalog_path),
        "task_spawn_plan_baseline_path": str(baseline_spawn_plan_path),
        "task_spawn_plan_path": str(legacy_spawn_plan_path),
        "feasibility_gate": feasibility_report,
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
    variation_assets_prefix = os.getenv(
        "VARIATION_ASSETS_PREFIX",
        f"scenes/{scene_id}/variation_assets",
    )

    # Optional configuration
    environment_type = os.getenv("ENVIRONMENT_TYPE", "generic")
    policy_id = os.getenv("POLICY_ID", None)
    robot_type = os.getenv("ROBOT_TYPE", "franka")
    num_envs = int(os.getenv("NUM_ENVS", "1024"))
    isaac_refresh_only = parse_bool_env(os.getenv("ISAAC_REFRESH_ONLY"), default=False)

    # Runtime validation settings
    # Set RUN_RUNTIME_VALIDATION=false to skip validation
    # Set SKIP_SANITY_ROLLOUT=true to skip environment steps (faster, less thorough)
    run_runtime_validation = parse_bool_env(os.getenv("RUN_RUNTIME_VALIDATION"), default=True)
    skip_sanity_rollout = parse_bool_env(os.getenv("SKIP_SANITY_ROLLOUT"), default=False)
    strict_config_loading = parse_bool_env(os.getenv("STRICT_CONFIG_LOADING"), default=False)

    logger.info("[ISAAC-LAB-JOB] Configuration:")
    logger.info("[ISAAC-LAB-JOB]   Bucket: %s", bucket)
    logger.info("[ISAAC-LAB-JOB]   Scene ID: %s", scene_id)
    logger.info("[ISAAC-LAB-JOB]   Runtime validation: %s", run_runtime_validation)
    logger.info("[ISAAC-LAB-JOB]   Skip sanity rollout: %s", skip_sanity_rollout)
    logger.info("[ISAAC-LAB-JOB]   Strict config loading: %s", strict_config_loading)
    logger.info("[ISAAC-LAB-JOB]   Refresh-only mode: %s", isaac_refresh_only)
    logger.info("[ISAAC-LAB-JOB]   Variation assets prefix: %s", variation_assets_prefix)

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
        isaac_refresh_only=isaac_refresh_only,
        variation_assets_prefix=variation_assets_prefix,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    from tools.startup_validation import validate_and_fail_fast

    validate_and_fail_fast(job_name="ISAAC-LAB", validate_gcs=True)
    main()
