"""
LeRobot Environment Hub Integration for Isaac Lab-Arena.

This module provides integration with the LeRobot Environment Hub,
enabling Blueprint scenes to be published as standard benchmarks.

Key Features:
- Updated LeRobot Hub format (v2.0 compatible)
- Benchmark metadata generation
- Automatic environment registration
- Leaderboard integration
- Dataset linking

Distinction from Genie Sim:
- Genie Sim: Exports DATA in LeRobot format
- This Module: Publishes ENVIRONMENTS/BENCHMARKS to Hub

Usage:
    from tools.arena_integration.lerobot_hub import (
        LeRobotHubPublisher,
        HubEnvironmentSpec,
        publish_to_hub
    )

    publisher = LeRobotHubPublisher(HfToken="...")
    result = publisher.publish(env_spec, benchmark_config)
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

from .components import (
    ArenaScene,
    ArenaTask,
    ArenaEmbodiment,
    ArenaEnvironmentSpec,
    ArenaObject,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

class HubVisibility(str, Enum):
    """Hub repository visibility."""
    PUBLIC = "public"
    PRIVATE = "private"
    ORGANIZATION = "organization"


class BenchmarkCategory(str, Enum):
    """Benchmark categories for Hub organization."""
    MANIPULATION = "manipulation"
    NAVIGATION = "navigation"
    MOBILE_MANIPULATION = "mobile_manipulation"
    HUMANOID = "humanoid"
    MULTI_ROBOT = "multi_robot"
    DEFORMABLE = "deformable"
    TOOL_USE = "tool_use"
    LONG_HORIZON = "long_horizon"


@dataclass
class HubPublishConfig:
    """Configuration for Hub publishing."""
    # HuggingFace settings
    namespace: str = "blueprint-robotics"
    visibility: HubVisibility = HubVisibility.PUBLIC
    hf_token: Optional[str] = None

    # Repository settings
    repo_prefix: str = "arena-"
    create_dataset_card: bool = True
    create_model_card: bool = True

    # Benchmark settings
    include_baseline_results: bool = True
    include_evaluation_code: bool = True

    # Versioning
    version: str = "1.0.0"
    changelog: Optional[str] = None


@dataclass
class HubEnvironmentSpec:
    """
    LeRobot Hub Environment Specification (v2.0 format).

    This matches the updated Hub format for environment registration.
    """
    # Required fields
    env_id: str                                 # Unique environment ID
    display_name: str                           # Human-readable name
    description: str
    category: BenchmarkCategory

    # Source info
    source: str = "BlueprintPipeline"
    source_version: str = "1.0.0"

    # Environment details
    observation_space: dict[str, Any] = field(default_factory=dict)
    action_space: dict[str, Any] = field(default_factory=dict)
    reward_type: str = "sparse"                 # sparse, dense, shaped

    # Task info
    tasks: list[dict[str, Any]] = field(default_factory=list)
    default_task: Optional[str] = None

    # Robot support
    supported_embodiments: list[str] = field(default_factory=list)
    default_embodiment: str = "franka"

    # Difficulty and metrics
    difficulty: str = "medium"
    success_metric: str = "task_success"
    primary_metric: str = "success_rate"

    # Assets
    scene_usd_path: Optional[str] = None
    thumbnail_path: Optional[str] = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    license: str = "apache-2.0"

    def to_hub_yaml(self) -> str:
        """Convert to Hub YAML format."""
        data = {
            "env_id": self.env_id,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category.value,
            "source": {
                "name": self.source,
                "version": self.source_version,
            },
            "spaces": {
                "observation": self.observation_space,
                "action": self.action_space,
            },
            "reward_type": self.reward_type,
            "tasks": self.tasks,
            "default_task": self.default_task,
            "robots": {
                "supported": self.supported_embodiments,
                "default": self.default_embodiment,
            },
            "evaluation": {
                "difficulty": self.difficulty,
                "success_metric": self.success_metric,
                "primary_metric": self.primary_metric,
            },
            "metadata": {
                "tags": self.tags,
                "license": self.license,
            },
        }

        if self.citations:
            data["metadata"]["citations"] = self.citations

        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def to_hub_json(self) -> dict[str, Any]:
        """Convert to Hub JSON format."""
        return {
            "env_id": self.env_id,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category.value,
            "source": self.source,
            "source_version": self.source_version,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "reward_type": self.reward_type,
            "tasks": self.tasks,
            "default_task": self.default_task,
            "supported_embodiments": self.supported_embodiments,
            "default_embodiment": self.default_embodiment,
            "difficulty": self.difficulty,
            "success_metric": self.success_metric,
            "primary_metric": self.primary_metric,
            "tags": self.tags,
            "license": self.license,
        }


@dataclass
class HubPublishResult:
    """Result of Hub publishing operation."""
    success: bool
    env_id: str
    repo_url: Optional[str]
    files_uploaded: list[str]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# =============================================================================
# ENVIRONMENT SPEC BUILDER
# =============================================================================

class HubEnvironmentBuilder:
    """
    Builds LeRobot Hub environment specifications from Arena components.

    Converts Arena scenes and tasks to Hub-compatible format.
    """

    def __init__(self, config: Optional[HubPublishConfig] = None):
        self.config = config or HubPublishConfig()

    def from_arena_spec(
        self,
        env_spec: ArenaEnvironmentSpec,
        benchmark_name: Optional[str] = None,
    ) -> HubEnvironmentSpec:
        """
        Create Hub spec from Arena environment specification.

        Args:
            env_spec: Arena environment specification
            benchmark_name: Optional custom benchmark name

        Returns:
            HubEnvironmentSpec for Hub registration
        """
        scene = env_spec.scene
        task = env_spec.task
        embodiment = env_spec.embodiment

        # Generate env_id
        env_id = benchmark_name or f"arena_{scene.scene_id}_{task.task_id}"
        env_id = env_id.lower().replace(" ", "_").replace("-", "_")

        # Determine category
        category = self._infer_category(task, embodiment)

        # Build observation space
        obs_space = self._build_observation_space(env_spec)

        # Build action space
        action_space = self._build_action_space(embodiment)

        # Build task list
        tasks = [self._task_to_hub_format(task)]

        # Generate tags
        tags = self._generate_tags(scene, task, embodiment)

        return HubEnvironmentSpec(
            env_id=env_id,
            display_name=f"Arena: {scene.scene_id} - {task.name}",
            description=self._generate_description(scene, task),
            category=category,
            observation_space=obs_space,
            action_space=action_space,
            reward_type="shaped" if task.config.reward_scale != 1.0 else "sparse",
            tasks=tasks,
            default_task=task.task_id,
            supported_embodiments=[e.value for e in [
                embodiment.embodiment_type
            ]] + ["franka", "ur10", "gr1"],  # Common defaults
            default_embodiment=embodiment.embodiment_type.value,
            difficulty=task.difficulty.value,
            scene_usd_path=scene.config.usd_path,
            tags=tags,
        )

    def from_scene_all_tasks(
        self,
        scene: ArenaScene,
        embodiment: ArenaEmbodiment,
    ) -> list[HubEnvironmentSpec]:
        """
        Create Hub specs for all possible tasks in a scene.

        Args:
            scene: Arena scene
            embodiment: Default robot embodiment

        Returns:
            List of HubEnvironmentSpec for each valid task
        """
        from .components import ArenaEnvironmentBuilder, ArenaTask

        builder = ArenaEnvironmentBuilder()
        env_specs = builder.build_all_tasks_for_scene(scene, embodiment)

        hub_specs = []
        for env_spec in env_specs:
            hub_spec = self.from_arena_spec(env_spec)
            hub_specs.append(hub_spec)

        return hub_specs

    def _infer_category(
        self,
        task: ArenaTask,
        embodiment: ArenaEmbodiment
    ) -> BenchmarkCategory:
        """Infer benchmark category from task and embodiment."""
        from .affordances import AffordanceType
        from .components import EmbodimentType

        # Check embodiment type first
        if embodiment.embodiment_type in [EmbodimentType.GR1, EmbodimentType.G1]:
            if "mobile" in embodiment.capabilities:
                return BenchmarkCategory.MOBILE_MANIPULATION
            return BenchmarkCategory.HUMANOID

        # Check task affordances
        if AffordanceType.FOLDABLE in task.required_affordances:
            return BenchmarkCategory.DEFORMABLE

        # Check for tool use indicators
        if any(kw in task.task_id for kw in ["tool", "cut", "write"]):
            return BenchmarkCategory.TOOL_USE

        # Default to manipulation
        return BenchmarkCategory.MANIPULATION

    def _build_observation_space(
        self,
        env_spec: ArenaEnvironmentSpec
    ) -> dict[str, Any]:
        """Build observation space specification."""
        obs_keys = env_spec.observation_space_keys

        space = {
            "type": "dict",
            "spaces": {},
        }

        # Common observation spaces
        obs_shapes = {
            "image": {"type": "box", "shape": [224, 224, 3], "dtype": "uint8"},
            "depth": {"type": "box", "shape": [224, 224], "dtype": "float32"},
            "joint_pos": {"type": "box", "shape": [env_spec.embodiment.config.dof], "dtype": "float32"},
            "joint_vel": {"type": "box", "shape": [env_spec.embodiment.config.dof], "dtype": "float32"},
            "ee_pos": {"type": "box", "shape": [3], "dtype": "float32"},
            "ee_quat": {"type": "box", "shape": [4], "dtype": "float32"},
            "gripper_state": {"type": "box", "shape": [2], "dtype": "float32"},
            "object_pos": {"type": "box", "shape": [3], "dtype": "float32"},
            "object_quat": {"type": "box", "shape": [4], "dtype": "float32"},
            "target_pos": {"type": "box", "shape": [3], "dtype": "float32"},
        }

        for key in obs_keys:
            if key in obs_shapes:
                space["spaces"][key] = obs_shapes[key]
            else:
                # Default shape
                space["spaces"][key] = {"type": "box", "shape": [1], "dtype": "float32"}

        return space

    def _build_action_space(self, embodiment: ArenaEmbodiment) -> dict[str, Any]:
        """Build action space specification."""
        return {
            "type": "box",
            "shape": [embodiment.action_dim],
            "low": -1.0,
            "high": 1.0,
            "dtype": "float32",
        }

    def _task_to_hub_format(self, task: ArenaTask) -> dict[str, Any]:
        """Convert Arena task to Hub format."""
        return {
            "task_id": task.task_id,
            "name": task.name,
            "description": task.description,
            "required_affordances": [aff.value for aff in task.required_affordances],
            "max_steps": task.config.max_steps,
            "success_threshold": task.config.success_threshold,
            "difficulty": task.difficulty.value,
        }

    def _generate_description(self, scene: ArenaScene, task: ArenaTask) -> str:
        """Generate Hub description."""
        return f"""
## {scene.scene_id} - {task.name}

**Environment Type:** {scene.environment_type.value}
**Task:** {task.description}
**Difficulty:** {task.difficulty.value}

### Scene Information
- Objects: {len(scene.objects)}
- Environment: {scene.environment_type.value}

### Task Details
- Max Steps: {task.config.max_steps}
- Success Threshold: {task.config.success_threshold}
- Required Affordances: {', '.join(aff.value for aff in task.required_affordances)}

Generated by BlueprintPipeline Arena Integration.
""".strip()

    def _generate_tags(
        self,
        scene: ArenaScene,
        task: ArenaTask,
        embodiment: ArenaEmbodiment
    ) -> list[str]:
        """Generate Hub tags."""
        tags = [
            "arena",
            "isaac-lab",
            "blueprint-pipeline",
            scene.environment_type.value,
            task.task_id,
            embodiment.embodiment_type.value,
        ]

        # Add affordance tags
        for aff in task.required_affordances:
            tags.append(aff.value.lower())

        return list(set(tags))


# =============================================================================
# HUB PUBLISHER
# =============================================================================

class LeRobotHubPublisher:
    """
    Publishes environments to LeRobot Environment Hub.

    Handles:
    - Repository creation
    - File uploads
    - Metadata generation
    - Leaderboard setup
    """

    def __init__(self, config: HubPublishConfig):
        self.config = config
        self._hf_api = None

    def _get_hf_api(self):
        """Get HuggingFace API client."""
        if self._hf_api is None:
            try:
                from huggingface_hub import HfApi
                token = self.config.hf_token or os.getenv("HF_TOKEN")
                self._hf_api = HfApi(token=token)
            except ImportError:
                raise ImportError("huggingface_hub is required for Hub publishing")
        return self._hf_api

    def publish(
        self,
        hub_spec: HubEnvironmentSpec,
        arena_export_dir: Optional[Path] = None,
        evaluation_results: Optional[dict[str, Any]] = None,
    ) -> HubPublishResult:
        """
        Publish environment to LeRobot Hub.

        Args:
            hub_spec: Hub environment specification
            arena_export_dir: Optional directory with Arena export files
            evaluation_results: Optional baseline evaluation results

        Returns:
            HubPublishResult with publish status
        """
        errors: list[str] = []
        warnings: list[str] = []
        files_uploaded: list[str] = []

        repo_id = f"{self.config.namespace}/{self.config.repo_prefix}{hub_spec.env_id}"

        try:
            api = self._get_hf_api()

            # Create repository
            repo_url = api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",  # Environments are datasets in Hub
                exist_ok=True,
                private=self.config.visibility == HubVisibility.PRIVATE,
            )

            # Create temporary directory for files
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Generate and upload environment spec
                spec_path = tmpdir / "environment.yaml"
                spec_path.write_text(hub_spec.to_hub_yaml())
                files_uploaded.append("environment.yaml")

                # Generate JSON version
                json_path = tmpdir / "environment.json"
                with open(json_path, "w") as f:
                    json.dump(hub_spec.to_hub_json(), f, indent=2)
                files_uploaded.append("environment.json")

                # Generate README
                if self.config.create_dataset_card:
                    readme_path = tmpdir / "README.md"
                    readme_path.write_text(self._generate_readme(hub_spec, evaluation_results))
                    files_uploaded.append("README.md")

                # Copy Arena export files if provided
                if arena_export_dir and arena_export_dir.exists():
                    arena_files = self._copy_arena_files(arena_export_dir, tmpdir)
                    files_uploaded.extend(arena_files)

                # Add evaluation results if provided
                if evaluation_results:
                    results_path = tmpdir / "baseline_results.json"
                    with open(results_path, "w") as f:
                        json.dump(evaluation_results, f, indent=2)
                    files_uploaded.append("baseline_results.json")

                # Add evaluation code if configured
                if self.config.include_evaluation_code:
                    eval_code = self._generate_evaluation_code(hub_spec)
                    eval_path = tmpdir / "evaluate.py"
                    eval_path.write_text(eval_code)
                    files_uploaded.append("evaluate.py")

                # Upload all files
                api.upload_folder(
                    folder_path=str(tmpdir),
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Publish {hub_spec.env_id} v{self.config.version}",
                )

            return HubPublishResult(
                success=True,
                env_id=hub_spec.env_id,
                repo_url=str(repo_url),
                files_uploaded=files_uploaded,
                warnings=warnings,
            )

        except Exception as e:
            errors.append(str(e))
            return HubPublishResult(
                success=False,
                env_id=hub_spec.env_id,
                repo_url=None,
                files_uploaded=files_uploaded,
                errors=errors,
            )

    def _generate_readme(
        self,
        hub_spec: HubEnvironmentSpec,
        evaluation_results: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate README.md for Hub repository."""
        readme = f"""---
tags:
{chr(10).join(f'- {tag}' for tag in hub_spec.tags)}
license: {hub_spec.license}
task_categories:
- robotics
---

# {hub_spec.display_name}

{hub_spec.description}

## Quick Start

```python
from lerobot.env import make_env

env = make_env("{hub_spec.env_id}")
obs = env.reset()

for _ in range(1000):
    action = policy.get_action(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
```

## Environment Details

| Property | Value |
|----------|-------|
| Category | {hub_spec.category.value} |
| Difficulty | {hub_spec.difficulty} |
| Default Robot | {hub_spec.default_embodiment} |
| Reward Type | {hub_spec.reward_type} |

### Observation Space

```
{json.dumps(hub_spec.observation_space, indent=2)}
```

### Action Space

```
{json.dumps(hub_spec.action_space, indent=2)}
```

## Supported Robots

{chr(10).join(f'- {robot}' for robot in hub_spec.supported_embodiments)}

## Tasks

"""
        for task in hub_spec.tasks:
            readme += f"""
### {task['name']}

- **ID:** {task['task_id']}
- **Description:** {task['description']}
- **Max Steps:** {task['max_steps']}
- **Difficulty:** {task['difficulty']}
"""

        if evaluation_results:
            readme += f"""
## Baseline Results

| Metric | Value |
|--------|-------|
| Success Rate | {evaluation_results.get('success_rate', 'N/A'):.2%} |
| Mean Return | {evaluation_results.get('mean_return', 'N/A'):.2f} |
| Mean Episode Length | {evaluation_results.get('mean_length', 'N/A'):.1f} |
"""

        readme += f"""
## Citation

If you use this environment, please cite:

```bibtex
@misc{{blueprint_{hub_spec.env_id},
  title = {{{hub_spec.display_name}}},
  author = {{BlueprintPipeline}},
  year = {{{datetime.now().year}}},
  publisher = {{LeRobot Environment Hub}},
}}
```

## License

{hub_spec.license}
"""
        return readme

    def _copy_arena_files(
        self,
        arena_dir: Path,
        target_dir: Path
    ) -> list[str]:
        """Copy Arena export files to upload directory."""
        import shutil

        files_copied = []
        arena_target = target_dir / "arena"
        arena_target.mkdir(exist_ok=True)

        for file_path in arena_dir.glob("**/*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(arena_dir)
                target_path = arena_target / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, target_path)
                files_copied.append(f"arena/{rel_path}")

        return files_copied

    def _generate_evaluation_code(self, hub_spec: HubEnvironmentSpec) -> str:
        """Generate evaluation script."""
        return f'''#!/usr/bin/env python3
"""
Evaluation script for {hub_spec.env_id}

This script evaluates a policy on the {hub_spec.display_name} environment.

Usage:
    python evaluate.py --policy-path /path/to/policy.pt --num-episodes 100
"""

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True, help="Path to policy checkpoint")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()

    # Load environment
    try:
        from lerobot.env import make_env
        env = make_env("{hub_spec.env_id}")
    except ImportError:
        print("Please install lerobot: pip install lerobot")
        return

    # Load policy
    import torch
    policy = torch.load(args.policy_path)

    # Evaluate
    successes = []
    returns = []
    lengths = []

    for ep in range(args.num_episodes):
        obs = env.reset()
        episode_return = 0
        episode_length = 0

        for step in range({hub_spec.tasks[0]['max_steps'] if hub_spec.tasks else 500}):
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)

            episode_return += reward
            episode_length += 1

            if done:
                break

        successes.append(info.get("success", False))
        returns.append(episode_return)
        lengths.append(episode_length)

        print(f"Episode {{ep+1}}/{{args.num_episodes}}: success={{successes[-1]}}, return={{returns[-1]:.2f}}")

    # Save results
    results = {{
        "env_id": "{hub_spec.env_id}",
        "num_episodes": args.num_episodes,
        "success_rate": sum(successes) / len(successes),
        "mean_return": sum(returns) / len(returns),
        "mean_length": sum(lengths) / len(lengths),
    }}

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\\nResults saved to {{args.output}}")
    print(f"Success rate: {{results['success_rate']:.2%}}")


if __name__ == "__main__":
    main()
'''


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def publish_to_hub(
    env_spec: ArenaEnvironmentSpec,
    namespace: str = "blueprint-robotics",
    hf_token: Optional[str] = None,
) -> HubPublishResult:
    """
    Convenience function to publish Arena environment to Hub.

    Args:
        env_spec: Arena environment specification
        namespace: HuggingFace namespace
        hf_token: HuggingFace token (or set HF_TOKEN env var)

    Returns:
        HubPublishResult
    """
    config = HubPublishConfig(
        namespace=namespace,
        hf_token=hf_token,
    )

    builder = HubEnvironmentBuilder(config)
    hub_spec = builder.from_arena_spec(env_spec)

    publisher = LeRobotHubPublisher(config)
    return publisher.publish(hub_spec)


def generate_hub_spec(
    env_spec: ArenaEnvironmentSpec,
    benchmark_name: Optional[str] = None,
) -> HubEnvironmentSpec:
    """
    Generate Hub specification without publishing.

    Args:
        env_spec: Arena environment specification
        benchmark_name: Optional custom name

    Returns:
        HubEnvironmentSpec
    """
    builder = HubEnvironmentBuilder()
    return builder.from_arena_spec(env_spec, benchmark_name)
