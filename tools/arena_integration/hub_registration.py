"""
LeRobot Hub Registration - Register Blueprint environments with Hugging Face.

This module handles automatic registration of Blueprint-generated Arena
environments with the LeRobot Environment Hub on Hugging Face.

Features:
- Automatic environment registration
- Metadata generation and upload
- Version management
- Evaluation result integration
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .benchmark_validation import (
    resolve_isaac_lab_arena_version,
    validate_arena_benchmark_results,
)

@dataclass
class HubConfig:
    """Configuration for LeRobot Hub registration."""
    scene_id: str
    namespace: str = "blueprint-robotics"
    visibility: str = "public"                # public, private, organization
    repo_type: str = "environment"            # environment, dataset, model
    environment_type: str = "generic"
    description: str = ""
    tags: list[str] = field(default_factory=lambda: ["blueprint", "arena", "simulation"])
    license: str = "apache-2.0"

    # Hugging Face credentials (from env or explicit)
    hf_token: Optional[str] = None

    # Repo settings
    create_if_missing: bool = True
    update_existing: bool = True


@dataclass
class HubRegistrationResult:
    """Result of hub registration."""
    success: bool
    repo_id: str
    repo_url: str
    files_uploaded: list[str]
    errors: list[str] = field(default_factory=list)


class LeRobotHubRegistrar:
    """
    Handles registration of Blueprint environments with LeRobot Hub.

    The LeRobot Environment Hub (https://huggingface.co/spaces/lerobot/environment-hub)
    provides a registry of simulation environments for robot learning.

    This registrar:
    1. Creates/updates Hugging Face repos
    2. Uploads Arena scene files
    3. Generates standardized metadata
    4. Links evaluation results
    """

    def __init__(self, config: HubConfig):
        self.config = config
        self.hf_token = config.hf_token or os.getenv("HF_TOKEN")
        self._hf_available = self._check_hf_available()

    def _check_hf_available(self) -> bool:
        """Check if Hugging Face Hub is available."""
        try:
            from huggingface_hub import HfApi
            return True
        except ImportError:
            return False

    @property
    def repo_id(self) -> str:
        """Get the full repo ID."""
        return f"{self.config.namespace}/{self.config.scene_id}"

    @property
    def repo_url(self) -> str:
        """Get the repo URL."""
        return f"https://huggingface.co/{self.repo_id}"

    def register(
        self,
        arena_dir: Path,
        evaluation_results: Optional[dict[str, Any]] = None
    ) -> HubRegistrationResult:
        """
        Register environment with LeRobot Hub.

        Args:
            arena_dir: Path to Arena export directory
            evaluation_results: Optional evaluation results to include

        Returns:
            HubRegistrationResult with registration status
        """
        errors: list[str] = []
        uploaded_files: list[str] = []

        if not self._hf_available:
            return HubRegistrationResult(
                success=False,
                repo_id=self.repo_id,
                repo_url=self.repo_url,
                files_uploaded=[],
                errors=["huggingface_hub not installed. Run: pip install huggingface_hub"],
            )

        if not self.hf_token:
            return HubRegistrationResult(
                success=False,
                repo_id=self.repo_id,
                repo_url=self.repo_url,
                files_uploaded=[],
                errors=["HF_TOKEN not set. Get a token from https://huggingface.co/settings/tokens"],
            )

        if evaluation_results:
            arena_version = resolve_isaac_lab_arena_version(arena_dir)
            validated_results = dict(evaluation_results)
            if arena_version and "isaac_lab_arena_version" not in validated_results:
                validated_results["isaac_lab_arena_version"] = arena_version
            tasks_dir = arena_dir / "tasks"
            task_ids = None
            if tasks_dir.exists():
                task_ids = [
                    task_file.stem
                    for task_file in tasks_dir.glob("*.py")
                    if task_file.stem != "__init__"
                ]
            validate_arena_benchmark_results(
                validated_results,
                scene_id=self.config.scene_id,
                task_ids=task_ids,
                arena_dir=arena_dir,
                arena_version=arena_version,
            )
            evaluation_results = validated_results

        try:
            from huggingface_hub import HfApi, create_repo, upload_folder

            api = HfApi(token=self.hf_token)

            # Step 1: Create or verify repo exists
            if self.config.create_if_missing:
                try:
                    create_repo(
                        repo_id=self.repo_id,
                        token=self.hf_token,
                        repo_type="dataset",  # Environments stored as datasets
                        exist_ok=True,
                        private=(self.config.visibility == "private"),
                    )
                except Exception as e:
                    errors.append(f"Failed to create repo: {e}")

            # Step 2: Generate metadata files
            metadata_files = self._generate_metadata_files(arena_dir, evaluation_results)
            for filename, content in metadata_files.items():
                file_path = arena_dir / filename
                if isinstance(content, str):
                    file_path.write_text(content)
                else:
                    with open(file_path, "w") as f:
                        json.dump(content, f, indent=2)

            # Step 3: Upload files
            try:
                upload_result = upload_folder(
                    repo_id=self.repo_id,
                    folder_path=str(arena_dir),
                    repo_type="dataset",
                    token=self.hf_token,
                    commit_message=f"Update from BlueprintPipeline - {datetime.utcnow().isoformat()}",
                )
                uploaded_files = list(arena_dir.rglob("*"))
                uploaded_files = [str(f.relative_to(arena_dir)) for f in uploaded_files if f.is_file()]
            except Exception as e:
                errors.append(f"Failed to upload files: {e}")

            # Step 4: Update repo metadata
            try:
                self._update_repo_metadata(api)
            except Exception as e:
                errors.append(f"Failed to update repo metadata: {e}")

            return HubRegistrationResult(
                success=len(errors) == 0,
                repo_id=self.repo_id,
                repo_url=self.repo_url,
                files_uploaded=uploaded_files,
                errors=errors,
            )

        except Exception as e:
            errors.append(f"Registration failed: {e}")
            return HubRegistrationResult(
                success=False,
                repo_id=self.repo_id,
                repo_url=self.repo_url,
                files_uploaded=[],
                errors=errors,
            )

    def _generate_metadata_files(
        self,
        arena_dir: Path,
        evaluation_results: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Generate metadata files for Hub upload."""
        files = {}

        # README.md for repo landing page
        files["README.md"] = self._generate_readme(arena_dir, evaluation_results)

        # environment_info.json - structured metadata for Hub indexing
        env_info = self._generate_environment_info(arena_dir, evaluation_results)
        files["environment_info.json"] = env_info

        # LEROBOT_ENV.yaml - LeRobot-specific config
        files["LEROBOT_ENV.yaml"] = self._generate_lerobot_config()

        return files

    def _generate_readme(
        self,
        arena_dir: Path,
        evaluation_results: Optional[dict[str, Any]] = None
    ) -> str:
        """Generate README.md for the Hub repo."""
        # Load arena manifest for details
        manifest_path = arena_dir / "arena_manifest.json"
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())

        task_count = manifest.get("task_count", 0)
        objects = manifest.get("objects", [])
        affordances = set()
        for obj in objects:
            affordances.update(obj.get("affordances", []))

        readme = f'''---
tags:
{self._format_yaml_tags(self.config.tags)}
license: {self.config.license}
task_categories:
  - robotics
  - reinforcement-learning
---

# {self.config.scene_id}

**Blueprint Pipeline Arena Environment**

This environment was auto-generated from a real-world scene using [BlueprintPipeline](https://github.com/blueprint-robotics/pipeline).

## Overview

- **Environment Type**: {self.config.environment_type}
- **Task Count**: {task_count}
- **Object Count**: {len(objects)}
- **Unique Affordances**: {len(affordances)}

## Supported Robots

{self._format_robot_list(manifest.get("supported_embodiments", ["franka"]))}

## Available Affordances

{self._format_affordance_list(affordances)}

## Quick Start

```python
from isaaclab_arena import ArenaEnvBuilder
from huggingface_hub import snapshot_download

# Download environment
env_path = snapshot_download(repo_id="{self.repo_id}", repo_type="dataset")

# Load and build environment
import sys
sys.path.insert(0, env_path)
from scene_module import get_scene

scene = get_scene()
builder = ArenaEnvBuilder(scene=scene, embodiment="franka")
env = builder.build()

# Run episode
obs = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
```

## Evaluation Results

'''

        if evaluation_results:
            readme += f'''
| Metric | Value |
|--------|-------|
| Success Rate | {evaluation_results.get("overall_success_rate", 0):.1%} |
| Total Episodes | {evaluation_results.get("total_episodes", 0)} |
| Best Task | {evaluation_results.get("summary", {}).get("best_task", "N/A")} |
'''
        else:
            readme += "*No evaluation results available yet.*\n"

        readme += f'''

## Source

Generated by BlueprintPipeline on {datetime.utcnow().strftime("%Y-%m-%d")}.

## License

This environment is released under the {self.config.license} license.
'''
        return readme

    def _generate_environment_info(
        self,
        arena_dir: Path,
        evaluation_results: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Generate structured environment info for indexing."""
        manifest_path = arena_dir / "arena_manifest.json"
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())

        info = {
            "version": "1.0.0",
            "schema_version": "lerobot-env-1.0",
            "environment": {
                "name": self.config.scene_id,
                "display_name": f"Blueprint {self.config.environment_type.title()} - {self.config.scene_id}",
                "type": self.config.environment_type,
                "source": "blueprint-pipeline",
                "source_version": "1.0.0",
            },
            "simulation": {
                "platform": "isaac-sim",
                "arena_version": manifest.get("arena_version", "0.1.0"),
                "physics_engine": "physx",
            },
            "assets": {
                "object_count": len(manifest.get("objects", [])),
                "task_count": manifest.get("task_count", 0),
            },
            "supported_embodiments": manifest.get("supported_embodiments", ["franka"]),
            "affordances": list(set(
                aff for obj in manifest.get("objects", [])
                for aff in obj.get("affordances", [])
            )),
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "tags": self.config.tags,
                "license": self.config.license,
            },
        }

        if evaluation_results:
            info["evaluation"] = {
                "success_rate": evaluation_results.get("overall_success_rate"),
                "total_episodes": evaluation_results.get("total_episodes"),
                "evaluated_at": evaluation_results.get("timestamp"),
            }

        return info

    def _generate_lerobot_config(self) -> str:
        """Generate LeRobot-specific configuration."""
        return f'''# LeRobot Environment Configuration
# Auto-generated by BlueprintPipeline

environment:
  name: {self.config.scene_id}
  type: {self.config.environment_type}
  source: blueprint-pipeline

arena:
  version: "0.1.0"
  scene_module: scene_module.py
  tasks_dir: tasks/

loading:
  method: huggingface
  repo_id: {self.repo_id}
  repo_type: dataset

compatibility:
  lerobot_version: ">=0.1.0"
  isaac_sim_version: ">=5.0.0"
  isaac_lab_arena_version: ">=0.1.0"

metadata:
  generated_by: BlueprintPipeline
  generated_at: {datetime.utcnow().isoformat()}
'''

    def _update_repo_metadata(self, api: Any) -> None:
        """Update repository metadata on Hugging Face."""
        # This would update repo card data, tags, etc.
        # Implementation depends on HF Hub API specifics
        pass

    def _format_yaml_tags(self, tags: list[str]) -> str:
        """Format tags for YAML front matter."""
        return "\n".join(f"  - {tag}" for tag in tags)

    def _format_robot_list(self, robots: list[str]) -> str:
        """Format robot list for README."""
        robot_names = {
            "franka": "Franka Emika Panda",
            "ur10": "Universal Robots UR10",
            "fetch": "Fetch Robotics",
            "gr1": "Fourier GR1",
            "g1": "Unitree G1",
        }
        lines = []
        for robot in robots:
            name = robot_names.get(robot, robot.title())
            lines.append(f"- **{name}** (`{robot}`)")
        return "\n".join(lines)

    def _format_affordance_list(self, affordances: set[str]) -> str:
        """Format affordance list for README."""
        if not affordances:
            return "*No affordances detected.*"
        return "\n".join(f"- {aff}" for aff in sorted(affordances))


def register_with_hub(
    arena_dir: Path,
    scene_id: str,
    namespace: str = "blueprint-robotics",
    evaluation_results: Optional[dict[str, Any]] = None,
) -> HubRegistrationResult:
    """
    Convenience function to register with LeRobot Hub.

    Args:
        arena_dir: Path to Arena export directory
        scene_id: Scene identifier
        namespace: Hugging Face namespace/organization
        evaluation_results: Optional evaluation results

    Returns:
        HubRegistrationResult with registration status
    """
    config = HubConfig(
        scene_id=scene_id,
        namespace=namespace,
    )

    registrar = LeRobotHubRegistrar(config)
    return registrar.register(arena_dir, evaluation_results)
